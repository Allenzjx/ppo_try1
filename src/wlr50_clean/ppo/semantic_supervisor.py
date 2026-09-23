"""Task semantics independent of the frozen FSM's imitation guards.

No Isaac imports, legacy-controller state mutation, Recording observations, or
clock-derived completion. The same live TaskEvaluator can accompany A, B and C.
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import yaml

from wlr50_clean.fsm.controller import ControllerEvent, ControllerFrame
from wlr50_clean.fsm.motion_executor import MotionExecutor
from wlr50_clean.fsm.state_spec import Lifecycle, load_fsm_spec
from wlr50_clean.fsm.task_result import TaskResult, TaskTermination
from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER, SERVO_ORDER, WHEEL_ORDER, Full12Command, servo_limits_deg,
)
from wlr50_clean.reference.motion_contract import load_motion_contract
from .semantic_workspace_potential import (
    MODE as WORKSPACE_POTENTIAL_MODE, interval_distance_progress,
)
from .semantic_transfer_roles import MODE as TRANSFER_ROLES_MODE, TransferRoleTracker, validate_config as validate_transfer_roles

DEFAULT_TASK_SPEC_PATH = Path(__file__).resolve().parents[3] / "configs/ppo_semantic_v2/stage_task_spec.yaml"
LEG_ORDER = ("FL", "FR", "RL", "RR")
PLACEMENT_PREDECESSORS = {"FR":(), "FL":("FR",), "RR":("FR","FL"), "RL":("FR","FL","RR")}
PHASE_IDS = tuple(f"P{i:02d}" for i in range(1, 14))
GOAL_FEATURE_KEYS = tuple(
    f"{leg}_{feature}" for leg in LEG_ORDER
    for feature in ("clearance_m", "front_distance_m", "load_fraction")
) + ("support_count", "body_forward_m", "body_linear_speed_m_s",
     "body_angular_speed_rad_s", "maximum_wheel_speed_rad_s")
ZERO12 = (0.0,) * 12
P06_RETIREMENT_MODE = "measured_workspace_interior_peak"
LIFT_CREDIT_MODE = "measured_air_process_current_top_gap"
PREPARATION_CREDIT_MODE = "current_workspace_before_predecessor_placement"
CAPTURE_RETENTION_MODE = "current_platform_region_after_placement"
CAPTURE_APPROACH_MODE = "post_cross_current_surface_proximity_plus_real_contact_v1"
FL_CAPTURE_APPROACH_MODE = "post_cross_FL_multiscale_positive_gap_plus_real_contact_v1"
CAPTURE_APPROACH_MODES = (CAPTURE_APPROACH_MODE, FL_CAPTURE_APPROACH_MODE)
STOP_PROGRESS_MODE = "per_wheel_four_type_threshold_ratio_v1"
P06_TAIL_MODE = "measured_workspace_retirement_after_finite_source"
P09_LIFT_MODE = "functional_lift_edge_v2"
P09_FREE_AIR_LIFT_MODE = "functional_free_air_lift_v3"
FUNCTIONAL_RR_MODES = (P09_LIFT_MODE, P09_FREE_AIR_LIFT_MODE)
RR_CARRY_SOURCE_MODE = "current_free_lift_before_pending_knee_and_roll_v1"
RR_WORKSPACE_RETIREMENT_MODE = "current_qualified_RR_over_top_receiver_retirement_v1"
RR_WORKSPACE_RETIREMENT_MODE_V2 = "established_RR_over_top_receiver_retirement_v2"
CAPTURE_CONTINUATION_MODE = "p05_hip_only_continuation_v1"
P05_PREEDGE_RECOVERY_MODE = "p05_preedge_approach_recovery_v1"
P05_SAME_AIR_RECROSS_MODE = "p05_preedge_same_air_recross_v2"
P05_COMPLETED_SOURCE_RECOVERY_MODE = "p05_completed_source_first_approach_recross_v3"
P05_FINITE_RECOVERY_TIMEOUT_MODE = "current_capture_or_completed_handoff_after_original_window_v1"


def _capture_continuation_enabled(spec: Mapping[str, Any]) -> bool:
    mode = spec.get("capture_continuation_semantics")
    if mode not in (None, CAPTURE_CONTINUATION_MODE):
        raise ValueError("unknown FL capture continuation semantics")
    if mode is not None and (spec.get("physical_acceptance_version") != "all_stage_v1"
            or spec.get("potential_definition") != "global_physical_progress_v3"
            or spec.get("reference_nominal_semantics") != "successful_fsm_derived_v2"
            or spec["nominal"].get("continuous_channel_inheritance") is not True
            or spec["nominal"].get("sequence_semantics") != "source_partial_order_physical_ready_v1"):
        raise ValueError("FL capture continuation requires continuous source and measured all-stage progress")
    return mode is not None


def _p05_preedge_recovery_enabled(spec: Mapping[str, Any]) -> bool:
    mode = spec.get("nominal", {}).get("p05_preedge_approach_recovery")
    if mode not in (None, P05_PREEDGE_RECOVERY_MODE, P05_SAME_AIR_RECROSS_MODE,
                    P05_COMPLETED_SOURCE_RECOVERY_MODE):
        raise ValueError("unknown P05 pre-edge approach recovery semantics")
    timeout = spec.get("p05_finite_recovery_timeout_semantics")
    recross_mode = mode in (P05_SAME_AIR_RECROSS_MODE, P05_COMPLETED_SOURCE_RECOVERY_MODE)
    if ((recross_mode and timeout != P05_FINITE_RECOVERY_TIMEOUT_MODE)
            or (not recross_mode and timeout is not None)):
        raise ValueError("same-AIR P05 recross and finite effective recovery must be explicitly paired")
    if mode is not None and (not _capture_continuation_enabled(spec)
            or spec["nominal"].get("p05_pending_capture") != "current_FL_capture_wheel_continuation_to_handoff_v2"
            or spec.get("local_timeout_policy", {}).get("mode") != "bounded_current_progress_allowance_v1"
            or not 0. < _number(spec["local_timeout_policy"].get("maximum_extension_s"), "pre-edge window") <= 10.):
        raise ValueError("P05 pre-edge recovery requires the existing finite continuous capture schedule")
    return mode is not None


def placement_predecessors_satisfied(spec: Mapping[str, Any], history: Mapping[str, Any], leg: str) -> bool:
    """Credit real rear motion without manufacturing a missing FL placement.

    This is only progress eligibility. Hard Q/C/P measurements and RR_FIRST
    remain in TaskEvaluator, and whole-task success still requires all four.
    """
    predecessors = PLACEMENT_PREDECESSORS[leg]
    if spec.get("capture_continuation_semantics") == CAPTURE_CONTINUATION_MODE and leg in ("RR", "RL"):
        predecessors = tuple(p for p in predecessors if p != "FL")
    return all(history["placed"][p] for p in predecessors)


def _capture_continuation_status(spec: Mapping[str, Any], evaluation: Mapping[str, Any],
                                 stage_id: str) -> dict[str, Any]:
    """Current recoverability, never a substituted touchdown or support flag."""
    history = evaluation.get("history", {})
    placed = history.get("placed", {})
    legs = evaluation.get("current_legs", {})
    fl = legs.get("FL", {})
    pending = bool(history.get("active_lift", {}).get("FL") is True
        and history.get("front_edge_crossed", {}).get("FL") is True
        and placed.get("FL") is False)
    contact = bool(fl.get("top_surface_contact") is True and fl.get("obstacle_pair_active") is True
        and fl.get("air") is False and fl.get("ground_contact") is False)
    # During rear swing, its own loaded wheel is not an 'other support'.
    excluded = {"FL"}
    if stage_id in ("P07", "P08", "P09", "P10", "P11", "P12"):
        excluded.add(spec["stages"][stage_id]["active_leg"])
    supports = tuple(leg for leg, row in legs.items() if leg not in excluded
        and row.get("support") is True and row.get("bearing_verified") is True
        and row.get("air") is False
        and (row.get("ground_contact") is True or row.get("top_surface_contact") is True))
    gap = fl.get("clearance_m")
    legal_path = bool(pending and fl.get("within_top_xy") is True
        and fl.get("within_lateral_span") is True and fl.get("ground_contact") is False
        and (fl.get("air") is True or contact)
        and isinstance(gap, (int, float)) and not isinstance(gap, bool) and math.isfinite(gap)
        and gap >= spec["geometry"]["top_gap_min_m"])
    allow = bool(evaluation.get("valid") is True and evaluation.get("termination_reason") is None
        and evaluation.get("physical_evidence_status") in ("VERIFIED", "CONTACT_BEARING_UNVERIFIED")
        and placed.get("FR") is True and legal_path
        and len(supports) >= spec["support"]["minimum_other_supports"])
    return {"mode": CAPTURE_CONTINUATION_MODE, "fl_contact_observed": contact,
        "fl_capture_pending": pending, "allow_capture_continuation": allow,
        "legal_capture_path": legal_path, "observed_other_support_contacts": supports,
        "excluded_support_legs": tuple(sorted(excluded)), "placed_FL": placed.get("FL") is True,
        "support_or_placement_awarded": False}


class SemanticObservationError(ValueError):
    """Missing/unverified physical data is an interface failure, not feasibility."""


def _validate_rr_carry_source_semantics(spec):
    nominal = spec.get("nominal", {})
    mode = nominal.get("rr_carry_source_semantics")
    if mode not in (None, RR_CARRY_SOURCE_MODE):
        raise ValueError("unknown RR carry source semantics")
    if mode is not None and (spec.get("p09_lift_semantics") != P09_FREE_AIR_LIFT_MODE
            or spec.get("physical_acceptance_version") != "all_stage_v1"
            or spec.get("reference_nominal_semantics") != "successful_fsm_derived_v2"
            or nominal.get("sequence_semantics") != "source_partial_order_physical_ready_v1"
            or nominal.get("continuous_channel_inheritance") is not True):
        raise ValueError("RR carry source readiness requires continuous free-AIR v3 source semantics")
    return mode


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SemanticObservationError(f"{name} must be a finite measurement")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SemanticObservationError(f"missing finite measurement: {name}") from exc
    if not math.isfinite(result):
        raise SemanticObservationError(f"nonfinite measurement: {name}")
    return result


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise SemanticObservationError(f"{name} requires {size} measured components")
    return tuple(_number(item, name) for item in value)


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _current_rr_placement_usable(evaluation: Mapping[str, Any]) -> bool:
    """Real placement plus current region; never a historical support claim."""
    current = evaluation["current_legs"]["RR"]
    history = evaluation["history"]
    return bool(evaluation["valid"] and evaluation["termination_reason"] is None
        and history["placed"]["RR"] and history["front_edge_crossed"]["RR"]
        and current.get("lift_established") and not current["ground_contact"]
        and current["within_top_xy"]
        and (current["top_contact"] or (current.get("current_lift_valid")
            and current["air"] and current["clearance_m"] >= 0.)))


def _p06_retirement_bounds(spec: Mapping[str,Any]) -> tuple[float,float] | None:
    mode=spec["nominal"].get("p06_rolling_retirement")
    if mode is None: return None
    if mode != P06_RETIREMENT_MODE:
        raise ValueError("unrecognized P06 rolling retirement semantics")
    if spec["nominal"].get("continuous_channel_inheritance") is not True:
        raise ValueError("P06 retirement requires continuous layer ownership")
    geo=spec["geometry"]
    lower=_number(geo["workspace_min_m"],"workspace lower bound")
    upper=_number(geo["workspace_max_m"],"workspace upper bound")
    width=_number(geo["xy_measurement_tolerance_m"],"workspace blending width")
    if not 0 < width < upper-lower:
        raise ValueError("P06 retirement requires positive measurement width inside workspace")
    return lower,width


def _p06_tail_enabled(spec: Mapping[str, Any]) -> bool:
    mode = spec["nominal"].get("p06_wheel_tail_semantics")
    if mode is None:
        return False
    if mode != P06_TAIL_MODE:
        raise ValueError("unrecognized P06 wheel tail semantics")
    if _p06_retirement_bounds(spec) is None:
        raise ValueError("P06 wheel tail requires continuous measured workspace retirement")
    return True


def _capture_approach_enabled(spec: Mapping[str, Any]) -> bool:
    mode = spec.get("capture_approach_semantics")
    if mode is None:
        return False
    if mode not in CAPTURE_APPROACH_MODES:
        raise ValueError("unrecognized capture approach semantics")
    if (spec.get("potential_definition") != "global_physical_progress_v3"
            or spec.get("capture_retention_semantics") != CAPTURE_RETENTION_MODE):
        raise ValueError("capture approach requires global progress and measured current platform geometry")
    if _number(spec["geometry"]["top_gap_max_m"], "capture approach existing gap scale") <= 0.:
        raise ValueError("capture approach requires a positive existing top gap scale")
    if mode == FL_CAPTURE_APPROACH_MODE:
        cfg = spec.get("fl_capture_potential", {})
        if cfg != {"coarse_gap_scale_m": .025, "fine_gap_scale_m": .003, "fine_fraction": .5}:
            raise ValueError("FL capture v1 requires explicit fixed coarse/fine engineering scales")
    elif "fl_capture_potential" in spec:
        raise ValueError("FL capture scales require their explicit potential version")
    return True


def _workspace_potential_enabled(spec: Mapping[str, Any]) -> bool:
    mode = spec.get("workspace_potential_semantics")
    if mode is None:
        return False
    if mode != WORKSPACE_POTENTIAL_MODE:
        raise ValueError("unrecognized workspace potential semantics")
    if spec.get("potential_definition") != "global_physical_progress_v3":
        raise ValueError("workspace potential requires global physical progress")
    lower = _number(spec["geometry"]["workspace_min_m"], "workspace potential lower")
    upper = _number(spec["geometry"]["workspace_max_m"], "workspace potential upper")
    if not lower < upper:
        raise ValueError("workspace potential requires an ordered existing interval")
    return True


def _rr_workspace_retirement_enabled(spec: Mapping[str, Any]) -> bool:
    mode = spec.get("rr_postcross_workspace_semantics")
    if mode is None:
        return False
    if mode not in (RR_WORKSPACE_RETIREMENT_MODE, RR_WORKSPACE_RETIREMENT_MODE_V2):
        raise ValueError("unknown RR post-cross workspace semantics")
    if (spec.get("workspace_potential_semantics") != WORKSPACE_POTENTIAL_MODE
            or spec.get("potential_definition") != "global_physical_progress_v3"
            or not spec.get("transfer_roles")
            or spec.get("physical_acceptance_version") != "all_stage_v1"
            or spec.get("p09_lift_semantics") not in FUNCTIONAL_RR_MODES):
        raise ValueError("RR post-cross workspace retirement requires measured functional RR progress")
    return True


def _current_rr_receiver_preparation_retired(spec: Mapping[str, Any], leg: str,
                                           evaluation: Mapping[str, Any]) -> bool:
    # Stateless potential only: no evaluator/event/contact/nominal mutation.
    if (leg != "RR" or not _rr_workspace_retirement_enabled(spec)
            or evaluation.get("valid") is not True
            or evaluation.get("termination_reason") is not None):
        return False
    history, current = evaluation["history"], evaluation["current_legs"]["RR"]
    eligible = current.get("current_lift_valid") is True
    if spec.get("rr_postcross_workspace_semantics") == RR_WORKSPACE_RETIREMENT_MODE_V2:
        # Preserve only already-earned receiver preparation, not current lift
        # or support validity. GROUND revokes lift_established in the evaluator.
        # Keep currentQ's independent geometry condition; only its short-window
        # other-support/body-control conjunction is irrelevant to this credit.
        for key in ("ground_relative_lift_m", "front_distance_m", "clearance_m"):
            value = current.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                return False
        eligible = bool(current.get("lift_established") is True
            and current.get("motion_continuation_allowed") is True
            and current["ground_relative_lift_m"] >=
                spec["history"]["minimum_initial_clearance_gain_m"])
    return bool(history["active_lift"]["RR"] is True
        and history["front_edge_crossed"]["RR"] is True
        and eligible
        and current.get("ground_contact") is False
        and current.get("within_top_xy") is True
        and current.get("within_lateral_span") is True
        and current["front_distance_m"] >= 0.
        and ((current.get("air") is True
              and current["clearance_m"] >= spec["geometry"]["top_gap_min_m"])
             or (current.get("top_surface_contact") is True
                 and current.get("top_contact") is True)))


def _verified_p06_rolling_source(contract: Any, expected: tuple[float, ...]) -> tuple[float, ...]:
    """Bind the advisory extension to the frozen wheel-only source, not a pose gate."""
    phase = contract.phase("P06")
    rows = tuple(phase.waypoints)
    if len(rows) != 3:
        raise ValueError("P06 wheel tail requires the verified three-waypoint rolling source")
    start = _vector(tuple(phase.start_full12), 12, "P06 source start")
    end = _vector(tuple(phase.end_full12), 12, "P06 source end")
    values = tuple(_vector(tuple(row.full12), 12, "P06 source waypoint") for row in rows)
    times = tuple(_number(row.time_s, "P06 source time") for row in rows)
    duration = _number(phase.active_duration_s, "P06 source duration")
    if (values[0] != start or values[2] != end
            or start[8:] != ZERO12[8:] or end[8:] != ZERO12[8:]
            or any(value[:8] != start[:8] for value in values)
            or values[1][8:] != expected or any(value <= 0. for value in expected)
            or tuple(phase.active_channels) != WHEEL_ORDER
            or tuple(row.kind for row in rows) != ("phase_entry", "reference_waypoint", "reference_waypoint")
            or rows[0].changed_channels or rows[0].atomic_channels
            or any(tuple(row.changed_channels) != WHEEL_ORDER or tuple(row.atomic_channels) != WHEEL_ORDER for row in rows[1:])
            or times[:2] != (0., 0.) or duration <= 0. or times[2] <= 0.
            or round(times[2] * contract.physics_hz) != round(duration * contract.physics_hz)
            or round(duration * contract.physics_hz) < 1):
        raise ValueError("P06 wheel tail source is not the verified positive rolling / zero endpoint structure")
    return values[1][8:]


def load_task_spec(path: Path | str = DEFAULT_TASK_SPEC_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict) or spec.get("schema") != "wlr50_clean.semantic_stage_task_spec.v2":
        raise ValueError("invalid semantic task schema")
    if spec.get("rear_leg_order") != "RR_FIRST" or tuple(spec.get("stages", {})) != PHASE_IDS:
        raise ValueError("semantic task must contain ordered P01-P13 and RR_FIRST")
    if spec.get("physics_hz") != 120.0 or spec.get("decision_hz") != 15.0:
        raise ValueError("semantic timing must remain 120/15 Hz")
    if not 0 < float(spec["episode_maximum_duration_s"]) <= 200:
        raise ValueError("semantic task horizon must be at most 200 seconds")
    if spec["history"].get("lift_motion_evidence") not in (None,"whole_body_actuation"):
        raise ValueError("unrecognized lift evidence semantics")
    if spec["history"].get("crossing_evidence_semantics") not in (None,"qualified_air_pending_geometry"):
        raise ValueError("unrecognized crossing evidence semantics")
    if spec.get("potential_definition") not in (None,"global_physical_progress_v3"):
        raise ValueError("unrecognized global potential semantics")
    if spec.get("preparation_credit_semantics") not in (None,PREPARATION_CREDIT_MODE):
        raise ValueError("unrecognized preparation credit semantics")
    if (spec.get("preparation_credit_semantics") == PREPARATION_CREDIT_MODE
            and spec.get("potential_definition") != "global_physical_progress_v3"):
        raise ValueError("preparation credit requires global physical progress potential")
    if spec.get("capture_retention_semantics") not in (None,CAPTURE_RETENTION_MODE):
        raise ValueError("unrecognized capture retention semantics")
    if spec.get("capture_retention_semantics") == CAPTURE_RETENTION_MODE:
        if spec.get("potential_definition") != "global_physical_progress_v3":
            raise ValueError("capture retention requires global physical progress potential")
        if _number(spec["history"]["minimum_lift_gain_m"], "capture retention scale") <= 0:
            raise ValueError("capture retention requires positive existing lift scale")
        if _number(spec["geometry"]["xy_measurement_tolerance_m"], "capture XY tolerance") < 0:
            raise ValueError("capture retention requires nonnegative existing XY tolerance")
        _number(spec["geometry"]["top_gap_min_m"], "capture existing top gap")
    if "rolling_capture_retention" in spec or spec.get("revision") == "task_conditioned_hip_wheel_v1":
        expected = {"rear_preparation_near_m": -.22, "blend_distance_m": .05,
                    "contact_fraction": .5, "positive_gap_scale_m": .003}
        if (spec.get("revision") != "task_conditioned_hip_wheel_v1"
                or spec.get("rolling_capture_retention") != expected
                or spec.get("capture_retention_semantics") != CAPTURE_RETENTION_MODE
                or spec.get("physical_acceptance_version") != "all_stage_v1"
                or spec.get("p09_lift_semantics") != P09_FREE_AIR_LIFT_MODE):
            raise ValueError("rolling retention requires its explicit current-contact task-quality version")
    _capture_approach_enabled(spec)
    _capture_continuation_enabled(spec)
    _p05_preedge_recovery_enabled(spec)
    _workspace_potential_enabled(spec)
    _rr_workspace_retirement_enabled(spec)
    validate_transfer_roles(spec)
    if spec.get("physical_acceptance_version") not in (None, "all_stage_v1"):
        raise ValueError("unknown physical acceptance version")
    if spec.get("p09_lift_semantics") not in (None, *FUNCTIONAL_RR_MODES):
        raise ValueError("unknown P09 functional lift semantics")
    if spec.get("p09_lift_semantics") in FUNCTIONAL_RR_MODES and spec.get("physical_acceptance_version") != "all_stage_v1":
        raise ValueError("functional P09 lift requires verified all-stage physical sensing")
    _validate_rr_carry_source_semantics(spec)
    if spec.get("physical_acceptance_version") == "all_stage_v1":
        timeout = spec.get("local_timeout_policy", {})
        if (timeout.get("mode") != "bounded_current_progress_allowance_v1"
                or not 0 < _number(timeout.get("maximum_extension_s"), "local extension") <= 10.
                or not 0 < _number(timeout.get("fraction_of_original_limit"), "local fraction") <= .5):
            raise ValueError("all-stage local recovery must be finite and within reviewed bounds")
        if not 0 < _number(spec["final"].get("post_completion_observation_s"), "post observation") <= 1.:
            raise ValueError("all-stage post observation requires a positive fixed window no longer than1s")
    if spec["final"].get("stop_pose_semantics") not in (None,"physical_stable_pose"):
        raise ValueError("unrecognized final stop pose semantics")
    if spec["final"].get("stop_command_progress") not in (None,"reciprocal_physical_stop_tolerance"):
        raise ValueError("unrecognized final command progress semantics")
    if (spec["final"].get("stop_command_progress") is not None
            and _number(spec["final"]["maximum_commanded_wheel_speed_rad_s"], "stop command tolerance") <= 0):
        raise ValueError("command progress requires a positive existing physical stop tolerance")
    if spec["final"].get("stop_progress_semantics") not in (None, STOP_PROGRESS_MODE):
        raise ValueError("unrecognized continuous stop progress semantics")
    if spec["final"].get("stop_progress_semantics") == STOP_PROGRESS_MODE:
        if (spec.get("potential_definition") != "global_physical_progress_v3"
                or spec["final"].get("stop_pose_semantics") != "physical_stable_pose"):
            raise ValueError("continuous stop progress requires global physical progress and physical stable pose")
        for key in ("maximum_wheel_speed_rad_s", "maximum_commanded_wheel_speed_rad_s",
                    "maximum_body_linear_speed_m_s", "maximum_body_angular_speed_rad_s"):
            if _number(spec["final"][key], key) <= 0:
                raise ValueError("continuous stop progress requires positive existing physical tolerances")
    _p06_retirement_bounds(spec)
    _p06_tail_enabled(spec)
    if spec.get("lift_credit_semantics") not in (None,LIFT_CREDIT_MODE):
        raise ValueError("unrecognized lift credit semantics")
    if spec.get("lift_credit_semantics") == LIFT_CREDIT_MODE:
        if spec["history"].get("lift_motion_evidence") != "whole_body_actuation":
            raise ValueError("soft AIR credit requires whole-body physical evidence")
        for key in ("minimum_initial_clearance_gain_m","minimum_lift_gain_m"):
            if _number(spec["history"][key],key) <= 0:
                raise ValueError("lift credit physical scales must be positive")
    required = {"purpose", "valid_start_conditions", "goal_features", "completion_predicates",
                "progress_potential", "allowed_action_channels", "physical_limits",
                "stall_diagnostic", "maximum_task_duration", "next_phase", "active_leg"}
    predicates = {"physical_valid", "whole_task_success", "rear_approach"} | {
        f"{kind}_{leg}" for kind in ("placed", "lifted", "clear", "approach", "workspace", "edge_proximity", "role_prepared", "transfer_ready", "support", "load_ready")
        for leg in LEG_ORDER
    }
    for index, phase in enumerate(PHASE_IDS):
        row = spec["stages"][phase]
        if not required <= row.keys():
            raise ValueError(f"{phase} lacks required semantic fields")
        if row["next_phase"] != (PHASE_IDS[index + 1] if index < 12 else "SUCCESS"):
            raise ValueError("semantic graph must be forward-only; repeated stage credit is forbidden")
        if not row["completion_predicates"] or not row["valid_start_conditions"]:
            raise ValueError("entry and completion require physical predicates")
        for key in ("completion_predicates", "valid_start_conditions", "goal_features"):
            if not set(row[key]) <= predicates:
                raise ValueError(f"unknown semantic predicate in {phase}")
        if not row["allowed_action_channels"] or not set(row["allowed_action_channels"]) <= set(FULL12_ORDER):
            raise ValueError(f"invalid action channels in {phase}")
        if not 0 < _number(row["maximum_task_duration"], "stage duration") <= 200:
            raise ValueError("invalid task duration")
    # P10 prepares workspace: workspace cannot be required before entering it.
    if any("workspace" in item for item in spec["stages"]["P10"]["valid_start_conditions"]):
        raise ValueError("P10 must create, not presuppose, RL workspace")
    return spec


class TaskEvaluator:
    """Independent episode-local measured lift -> crossing -> placement history.

    Initializing this class does not import snapshot success latches. Curriculum
    history must be established by real prefix observations; no history setter
    is exposed. Invalid geometry/contact cannot be replaced by knee numbers.
    V2 uses own-leg motion; explicit v3 uses measured whole-body actuation and
    distinguishes initial clearance from AIR above the obstacle top. The
    existing observable active_lift bits are current
    uninterrupted crossing qualifications until crossing, then completed lift
    history. Ground contact before crossing revokes qualification, not evidence.
    """

    def __init__(self, task_spec_path: Path | str = DEFAULT_TASK_SPEC_PATH, *, spec: Mapping[str, Any] | None = None):
        self.spec = dict(spec) if spec is not None else load_task_spec(task_spec_path)
        version = self.spec.get("physical_acceptance_version")
        if version not in (None, "all_stage_v1"):
            raise ValueError("unsupported physical acceptance version")
        self._all_stage = version == "all_stage_v1"
        self._functional_rr = self.spec.get("p09_lift_semantics") in FUNCTIONAL_RR_MODES
        self._free_air_rr = self.spec.get("p09_lift_semantics") == P09_FREE_AIR_LIFT_MODE
        self._rr_last_contact_bottom = None
        self._rr_free_air_count = 0
        self._rr_previous_bottom_sample = None
        self._rr_ground_bottom: float | None = None
        self._rr_non_ground_count = 0
        self._rr_edge_count = 0
        self._rr_established = False
        self._rr_geometry_samples: deque = deque()
        self._rr_support_samples: deque = deque(maxlen=self.spec["history"]["minimum_air_samples"])
        self._termination_source: str | None = None
        self._traversal_event_time: float | None = None
        self._completion_observation_since: float | None = None
        self._post_completion_loss = False
        self._attempt_active = dict.fromkeys(LEG_ORDER, False)
        self._attempt_joint_demand = dict.fromkeys(LEG_ORDER, False)
        self._wall_ascent = {leg: deque() for leg in LEG_ORDER}
        if self._all_stage:
            if self.spec["history"].get("lift_motion_evidence") != "whole_body_actuation":
                raise ValueError("all-stage acceptance requires whole-body measured actuation")
            duration = _number(self.spec["final"].get("post_completion_observation_s"), "post completion observation duration")
            if not 0. < duration <= 200.:
                raise ValueError("post completion observation must be finite and bounded")
        self._last_tick: int | None = None
        self._last_time: float | None = None
        self._stable_since: float | None = None
        self._samples = {leg: deque() for leg in LEG_ORDER}
        self._air_count = dict.fromkeys(LEG_ORDER, 0)
        self._top_count = dict.fromkeys(LEG_ORDER, 0)
        self._initial_clearance = dict.fromkeys(LEG_ORDER, False)
        self._obstacle_before_clearance = dict.fromkeys(LEG_ORDER, False)
        self._soft_air_progress_enabled = self.spec.get("lift_credit_semantics") == LIFT_CREDIT_MODE
        self._soft_air_actuation_earned = dict.fromkeys(LEG_ORDER,False)
        self._soft_air_earned_tick = dict.fromkeys(LEG_ORDER,None)
        self._history = {key: dict.fromkeys(LEG_ORDER, False) for key in ("active_lift", "front_edge_crossed", "placed")}
        self._event_ticks = {key: {} for key in self._history}
        self._lift_attempt_events: list[dict[str, Any]] = []
        self._snapshot: dict[str, Any] = {"valid": False, "success": False, "termination_reason": None,
            "reason": "no live observation", "goal_features": dict.fromkeys(GOAL_FEATURE_KEYS, 0.0)}
        self._failure: str | None = None
        self._failure_reason = ""
        self._transfer_tracker = TransferRoleTracker(self.spec) if self.spec.get("transfer_roles") else None

    @property
    def snapshot(self) -> dict[str, Any]:
        result = {**self._snapshot, "goal_features": dict(self._snapshot["goal_features"]),
            "history": {**{k: dict(v) for k, v in self._history.items()},
                        "event_ticks": {k: dict(v) for k, v in self._event_ticks.items()},
                        "active_lift_semantics": ("RR_same_attempt_establishment_with_separate_current_validity_and_contact_mode"
                            if self._functional_rr else "current_active_attempt_air_or_verified_early_top_until_crossing_then_history"
                            if self._all_stage else "current_uninterrupted_airborne_above_top_qualification_until_crossing_then_completed_history"),
                         "lift_attempt_events": [dict(event) for event in self._lift_attempt_events]}}
        if self._soft_air_progress_enabled and "current_legs" in result:
            result["current_legs"]={leg:{**row,
                "soft_air_actuation_earned":self._soft_air_actuation_earned[leg],
                "soft_air_earned_tick":self._soft_air_earned_tick[leg],
                "soft_air_evidence_semantics":"current_uninterrupted_measured_actuated_air_for_shaping_not_hard_qualification"}
                for leg,row in result["current_legs"].items()}
        return result

    def _fail(self, result: TaskResult, reason: str) -> None:
        if self._failure is None:
            self._failure, self._failure_reason = result.value, reason
            if self._all_stage:
                self._termination_source = (
                    "BODY_CONTACT" if result is TaskResult.TASK_FAILURE_BODY_COLLISION else
                    "WHEEL_ONLY_PROCESS" if result is TaskResult.TASK_FAILURE_WHEEL_ONLY_CLIMB else
                    "RR_FIRST_ORDER" if "RR_FIRST" in reason else
                    "NUMERICAL" if "nonfinite" in reason else
                    "HARD_JOINT_LIMIT" if "hard joint limit" in reason else
                    "UNVERIFIED_SENSOR" if result is TaskResult.INFRASTRUCTURE_ERROR else "PHYSICAL_SAFETY")
        for leg in LEG_ORDER:
            self._soft_air_actuation_earned[leg]=False
            self._soft_air_earned_tick[leg]=None

    def _unverified_sensor(self, reason: str, tick: int, now: float) -> dict[str, Any]:
        self._fail(TaskResult.INFRASTRUCTURE_ERROR, "unverified sensor: " + reason)
        self._snapshot.update(valid=False, success=False, termination_reason=self._failure,
            reason=self._failure_reason, evaluator_version="all_stage_v1", run_validity="UNVERIFIED",
            termination_source=self._termination_source, physical_evidence_status="UNVERIFIED_SENSOR",
            traversal_event_observed=self._traversal_event_time is not None,
            traversal_task_complete=False, task_completed_controlled=False)
        self._last_tick, self._last_time = tick, now
        return self.snapshot

    def _all_stage_body_bounds(self, observation: Any, contacts: Mapping[str, Any]) -> tuple[tuple, tuple]:
        base_pair = _get(contacts.get("base_link"), "obstacle")
        if (_get(base_pair, "pair_verified") is not True or type(_get(base_pair, "active")) is not bool
                or _get(base_pair, "sensor_body") != "base_link"
                or _get(base_pair, "other_body") != "/World/Obstacle"):
            raise SemanticObservationError("exact base_link/obstacle contact evidence unavailable")
        _vector(_get(base_pair, "force_w_n"), 3, "base pair force")
        if _get(observation, "geometry_pose_aware") is not True:
            raise SemanticObservationError("current-pose collider geometry unavailable")
        bounds = _get(_get(observation, "body_bounds_w_m", {}), "base_link")
        low = _vector(_get(bounds, "minimum_m"), 3, "base collider minimum")
        high = _vector(_get(bounds, "maximum_m"), 3, "base collider maximum")
        if any(a > b for a, b in zip(low, high)):
            raise SemanticObservationError("base collider bounds unordered")
        return low, high

    def _update_soft_air_process(self, leg: str, *, air: bool, in_task_region: bool, tick: int) -> None:
        """Acquire from a fresh AIR suffix, retain during hover, revoke on contact.

        Commands/response are measured evidence, not a causal motor-work proof.
        This state never participates in initial/qualified/cross/placed gates.
        """
        if not self._soft_air_progress_enabled: return
        if not air or self._failure is not None:
            self._soft_air_actuation_earned[leg]=False; self._soft_air_earned_tick[leg]=None
            return
        if self._soft_air_actuation_earned[leg]: return  # No sliding-window expiry.
        if not in_task_region or not placement_predecessors_satisfied(self.spec, self._history, leg): return
        suffix=[]
        for sample in reversed(self._samples[leg]):
            if not sample[4]: break  # Exclude all earlier ground/wall motion.
            suffix.append(sample)
        suffix.reverse(); cfg=self.spec["history"]
        if len(suffix)<max(2,cfg["minimum_air_samples"]): return
        upward=suffix[-1][1]-min(sample[1] for sample in suffix[:-1])
        if (upward<cfg["minimum_initial_clearance_gain_m"] or suffix[-1][1]<=suffix[-2][1]): return
        joint_motion=sum(sum(abs(x-y) for x,y in zip(a[5],b[5])) for a,b in zip(suffix,suffix[1:]))
        command_motion=sum(sum(abs(x-y) for x,y in zip(a[6],b[6])) for a,b in zip(suffix,suffix[1:]))
        tracking_error=max(abs(a-b) for a,b in zip(suffix[-1][6],suffix[-1][5]))
        gravity_motion=_norm(tuple(a-b for a,b in zip(suffix[-1][7],suffix[0][7])))
        # Reuse existing whole-body evidence scales. Wheel spin alone is not
        # a joint command demand; constant targets with tracking error are.
        demand=command_motion>=cfg.get("minimum_command_motion_deg",.1) or tracking_error>=.5
        response=joint_motion>=cfg["minimum_joint_motion_deg"] or gravity_motion>=cfg.get("minimum_gravity_direction_change",.02)
        if demand and response:
            self._soft_air_actuation_earned[leg]=True; self._soft_air_earned_tick[leg]=tick

    def observe(self, observation: Any) -> dict[str, Any]:
        tick = _get(observation, "physics_tick")
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise SemanticObservationError("physics_tick must be a nonnegative integer")
        now = _number(_get(observation, "simulation_time_s"), "simulation time")
        if self._last_tick == tick:
            if self._last_time != now:
                raise SemanticObservationError("same observation tick has inconsistent time")
            return self.snapshot
        if self._last_tick is not None and (tick != self._last_tick + 1 or not math.isclose(now-self._last_time,1/120.,rel_tol=0.,abs_tol=1e-9)):
            raise SemanticObservationError("task history requires contiguous advancing physical observations")
        if _get(observation, "all_finite") is not True:
            self._fail(TaskResult.SAFETY_ABORT, "nonfinite authoritative observation")
            self._snapshot.update(valid=False, success=False, termination_reason=self._failure, reason=self._failure_reason)
            if self._all_stage:
                self._snapshot.update(evaluator_version="all_stage_v1", termination_source=self._termination_source,
                    run_validity="NUMERICAL_SAFETY_ABORT", physical_evidence_status="NONFINITE",
                    task_completed_controlled=False, traversal_task_complete=False)
            self._last_tick, self._last_time = tick, now
            return self.snapshot
        base = _get(observation, "base")
        base_position = _vector(_get(base, "position_w_m"), 3, "base position")
        base_linear = _vector(_get(base, "linear_velocity_w_m_s"), 3, "base linear velocity")
        base_angular = _vector(_get(base, "angular_velocity_w_rad_s"), 3, "base angular velocity")
        _vector(_get(base, "orientation_wxyz"), 4, "base quaternion")
        gravity = _vector(_get(_get(observation, "imu"), "projected_gravity_b"), 3, "gravity")
        obstacle = _get(observation, "obstacle")
        front, top = (_number(_get(obstacle, key), key) for key in ("front_x_m", "top_z_m"))
        back, left, right = (_number(_get(obstacle,key),key) for key in ("back_x_m","left_y_m","right_y_m"))
        if back <= front or left <= right:
            raise SemanticObservationError("invalid measured obstacle plane ordering")
        joints, wheels, contacts = (_get(observation, key) for key in ("joints", "wheels", "contacts"))
        if not all(isinstance(v, Mapping) for v in (joints, wheels, contacts)):
            raise SemanticObservationError("full joint/wheel/exact-contact measurements are required")
        positions = {}
        for name in SERVO_ORDER:
            joint = joints.get(name)
            positions[name] = _number(_get(joint, "position_deg"), name)
            _number(_get(joint, "velocity_deg_s"), f"{name} velocity")
            lo, hi = servo_limits_deg(name)
            if not lo <= positions[name] <= hi:
                self._fail(TaskResult.SAFETY_ABORT, f"hard joint limit: {name}")
        collision = _get(_get(observation, "body_collision"), "detected")
        if not isinstance(collision, bool):
            raise SemanticObservationError("authoritative body collision status is required")
        if collision:
            self._fail(TaskResult.TASK_FAILURE_BODY_COLLISION, "central body/obstacle collision")
        if base_position[2] < .015 or base_position[2] > 1 or _norm(base_linear) > 5 or _norm(base_angular) > 20 or gravity[2] > -.30:
            self._fail(TaskResult.SAFETY_ABORT, "fall or physics explosion")
        body_bounds = None
        if self._all_stage:
            try:
                body_bounds = self._all_stage_body_bounds(observation, contacts)
            except SemanticObservationError as exc:
                return self._unverified_sensor(str(exc), tick, now)
        hist_cfg, geo = self.spec["history"], self.spec["geometry"]
        current = {}; forces = {}; speeds = []; commands = []
        for index, leg in enumerate(LEG_ORDER):
            wheel = wheels.get(WHEEL_ORDER[index]); body_name = _get(wheel, "body_name")
            if _get(wheel, "geometry_verified") is not True:
                if self._all_stage:
                    return self._unverified_sensor(f"unverified {leg} wheel geometry", tick, now)
                raise SemanticObservationError(f"unverified {leg} wheel geometry")
            center = _vector(_get(wheel, "center_w_m"), 3, f"{leg} center")
            bottom = _vector(_get(wheel, "bottom_w_m"), 3, f"{leg} bottom")
            speeds.append(_number(_get(wheel, "velocity_rad_s"), f"{leg} wheel speed"))
            commands.append(_number(_get(wheel, "command_rad_s"), f"{leg} wheel command"))
            contact = contacts.get(body_name); ground = _get(contact, "ground"); obstacle_pair = _get(contact, "obstacle")
            for pair in (ground, obstacle_pair):
                if _get(pair, "pair_verified") is not True or not isinstance(_get(pair, "active"), bool):
                    if self._all_stage:
                        return self._unverified_sensor(f"{leg} requires verified exact ground and obstacle contact pairs", tick, now)
                    raise SemanticObservationError(f"{leg} requires verified exact ground and obstacle contact pairs")
                _number(_get(pair, "normal_force_n"), f"{leg} pair normal force")
            ground_active, top_active = _get(ground, "active"), _get(obstacle_pair, "active")
            force = sum(max(0., _get(pair, "normal_force_n")) for pair in (ground, obstacle_pair) if _get(pair, "active"))
            reaction_force = force
            surfaces = None
            if self._all_stage:
                from .semantic_physical_sensing import physical_contact_surface
                surfaces = [physical_contact_surface(pair, kind=kind, obstacle=obstacle,
                    tolerance_m=geo["xy_measurement_tolerance_m"],
                    force_noise_floor_n=self.spec["support"]["force_noise_floor_n"])
                    for pair, kind in ((ground, "ground"), (obstacle_pair, "obstacle"))]
                if not all(v["pair_valid"] for v in surfaces):
                    return self._unverified_sensor(f"{leg} exact-pair force vector unavailable", tick, now)
                force = sum(v["bearing_force_n"] for v in surfaces if v["bearing_verified"])
            forces[leg] = force
            air = not ground_active and not top_active
            self._air_count[leg] = self._air_count[leg] + 1 if air else 0
            samples = self._samples[leg]
            if ground_active and (not self._history["front_edge_crossed"][leg] or (self._functional_rr and leg == "RR")):
                if self._all_stage:
                    self._attempt_active[leg] = False
                    self._attempt_joint_demand[leg] = False
                    self._wall_ascent[leg].clear()
                if self._history["active_lift"][leg] and not self._history["front_edge_crossed"][leg]:
                    self._history["active_lift"][leg] = False
                    self._lift_attempt_events.append({"leg": leg, "event": "qualification_revoked_ground_before_cross",
                        "physics_tick": tick, "simulation_time_s": now})
                # A later lift must earn new upward/actuation evidence, not reuse
                # the old hop still present in the half-second window.
                samples.clear()
            whole_body = hist_cfg.get("lift_motion_evidence") == "whole_body_actuation"
            if whole_body:
                joint_targets = tuple(_number(_get(joints[name], "command_deg"), f"{name} command") for name in SERVO_ORDER)
                wheel_targets = tuple(_number(_get(wheels[name], "command_rad_s"), f"{name} command") for name in WHEEL_ORDER)
                if ground_active and (not self._history["front_edge_crossed"][leg] or (self._functional_rr and leg == "RR")):
                    self._initial_clearance[leg] = False
                    self._obstacle_before_clearance[leg] = bool(top_active)
                elif top_active and not self._initial_clearance[leg]:
                    self._obstacle_before_clearance[leg] = True
            else:
                joint_targets, wheel_targets = (), ()
            samples.append((now, bottom[2], positions[SERVO_ORDER[index*2]], positions[SERVO_ORDER[index*2+1]], air,
                            tuple(positions.values()), joint_targets, gravity, wheel_targets))
            while len(samples) > 1 and now - samples[0][0] > hist_cfg["window_s"]:
                samples.popleft()
            minimum_so_far = samples[0][1]; gain = 0.
            for sample in samples:
                gain = max(gain, sample[1]-minimum_so_far)
                minimum_so_far = min(minimum_so_far, sample[1])
            movement = sum(abs(b[j]-a[j]) for a,b in zip(samples, list(samples)[1:]) for j in (2,3))
            distance = center[0] - front
            all_joint_motion = sum(sum(abs(x-y) for x,y in zip(a[5],b[5])) for a,b in zip(samples,list(samples)[1:]))
            command_motion = (sum(sum(abs(x-y) for x,y in zip(a[6],b[6])) for a,b in zip(samples,list(samples)[1:]))
                              if whole_body else 0.)
            gravity_motion = _norm(tuple(a-b for a,b in zip(samples[-1][7],samples[0][7])))
            commanded_effort = whole_body and (command_motion >= hist_cfg.get("minimum_command_motion_deg", .1)
                or max(abs(a-b) for a,b in zip(joint_targets, positions.values())) >= .5
                or max(map(abs,wheel_targets)) >= .02)
            motion_evidence = movement >= hist_cfg["minimum_joint_motion_deg"]
            if whole_body:
                motion_evidence = bool(commanded_effort and (all_joint_motion >= hist_cfg["minimum_joint_motion_deg"]
                    or gravity_motion >= hist_cfg.get("minimum_gravity_direction_change", .02)))
                if (not self._all_stage and self._failure is None and air and self._air_count[leg] >= hist_cfg["minimum_air_samples"]
                        and gain >= hist_cfg.get("minimum_initial_clearance_gain_m", .003)
                        and motion_evidence
                        # Earlier wheel/obstacle contact is allowed. If it
                        # precedes any free clearance, require fresh commanded
                        # whole-body joint motion rather than wheel spin alone.
                        and (not self._obstacle_before_clearance[leg]
                             or command_motion>=hist_cfg["minimum_joint_motion_deg"])
                        and not self._initial_clearance[leg]):
                    self._initial_clearance[leg] = True
                    self._lift_attempt_events.append({"leg":leg,"event":"whole_body_initial_clearance",
                        "physics_tick":tick,"simulation_time_s":now,"upward_excursion_m":gain,
                        "own_joint_motion_deg":movement,"whole_body_joint_motion_deg":all_joint_motion,
                        "command_motion_deg":command_motion,"gravity_direction_change":gravity_motion})
            lift = (hist_cfg["near_front_min_m"] <= distance <= hist_cfg["near_front_max_m"]
                and self._air_count[leg] >= hist_cfg["minimum_air_samples"]
                and bottom[2] >= top
                and gain >= hist_cfg["minimum_lift_gain_m"] and motion_evidence
                and (not whole_body or (self._initial_clearance[leg] and self._failure is None)))
            top_surface = bool(surfaces and surfaces[1]["surface"] == "TOP")
            if self._all_stage:
                # Wheel demand by itself is not proof of active joint work.
                # Whole-body joint demand + measured response + actual upward
                # motion can qualify before a ballistic AIR-above-top sample.
                joint_demand = bool(command_motion >= hist_cfg.get("minimum_command_motion_deg", .1)
                    or max(abs(a-b) for a,b in zip(joint_targets, positions.values())) >= .5)
                self._attempt_joint_demand[leg] |= joint_demand
                active_response = bool(joint_demand and (all_joint_motion >= hist_cfg["minimum_joint_motion_deg"]
                    or gravity_motion >= hist_cfg.get("minimum_gravity_direction_change", .02)))
                motion_evidence = active_response
                initial_now = bool(not ground_active and self._failure is None and active_response
                    and gain >= hist_cfg.get("minimum_initial_clearance_gain_m", .003)
                    and (self._air_count[leg] >= hist_cfg["minimum_air_samples"] or top_surface))
                functional = None
                if self._functional_rr and leg == "RR":
                    functional = self._observe_functional_rr(now=now, tick=tick, bottom=bottom,
                        center=center, ground=ground_active, obstacle_contact=top_active,
                        top_surface=top_surface, surface=surfaces[1]["surface"],
                        active_response=active_response, current_other_legs=current)
                    initial_now = functional["initial_now"]
                if initial_now:
                    self._attempt_active[leg] = True
                    if not self._initial_clearance[leg]:
                        self._initial_clearance[leg] = True
                        self._lift_attempt_events.append({"leg": leg, "event": "whole_body_initial_clearance",
                            "physics_tick": tick, "simulation_time_s": now,
                            "upward_excursion_m": gain, "whole_body_joint_motion_deg": all_joint_motion,
                            "command_motion_deg": command_motion,
                            **({"evidence": "continuous_unsupported_AIR_free_rise_v3",
                                "unsupported_free_lift_m": functional["unsupported_free_lift_m"]}
                               if self._free_air_rr and leg == "RR" else {"evidence": "active_air_or_early_top"})})
                lift = bool(self._failure is None and not ground_active and self._attempt_active[leg]
                    and gain >= hist_cfg["minimum_lift_gain_m"]
                    and hist_cfg["near_front_min_m"] <= distance <= hist_cfg["near_front_max_m"]
                    and (self._air_count[leg] >= hist_cfg["minimum_air_samples"] or top_surface))
                if functional is not None:
                    lift = functional["lift_established_now"]
                if surfaces[1]["surface"] == "FRONT_WALL" and not ground_active:
                    self._wall_ascent[leg].append((tick, bottom[2], max(map(abs, wheel_targets))))
                elif not top_surface:
                    self._wall_ascent[leg].clear()
            if lift and not self._history["active_lift"][leg]:
                self._history["active_lift"][leg] = True
                self._event_ticks["active_lift"].setdefault(leg, tick)
                self._lift_attempt_events.append({"leg": leg, "event": "qualified_measured_upward_lift",
                    "physics_tick": tick, "simulation_time_s": now, "upward_excursion_m": gain,
                    "joint_motion_deg": movement, "airborne_clearance_above_top_m": bottom[2]-top,
                    **({"qualification_evidence": "continuous_unsupported_AIR_free_rise_v3",
                        "unsupported_free_lift_m": functional["unsupported_free_lift_m"]}
                       if self._free_air_rr and leg == "RR" else {})})
            xy_tolerance = geo["xy_measurement_tolerance_m"]
            within_lateral_span = right-xy_tolerance <= center[1] <= left+xy_tolerance
            self._update_soft_air_process(leg,air=air,tick=tick,in_task_region=(within_lateral_span
                and hist_cfg["near_front_min_m"]<=distance<=hist_cfg["near_front_max_m"]))
            within_top_xy = within_lateral_span and front-xy_tolerance <= center[0] <= back+xy_tolerance
            top_geometry = within_top_xy and geo["top_gap_min_m"] <= bottom[2]-top <= geo["top_gap_max_m"]
            loaded = bool(top_active and top_geometry and distance >= 0)
            if self._all_stage:
                # TOP contact may precede the wheel CENTER plane. It is a
                # measured surface event, not a synonym for crossed/placed.
                loaded = bool(top_surface and not ground_active and within_top_xy)
            # The observable qualification proves this same uninterrupted
            # active lift reached AIR clearance. A finite-radius wheel may
            # land near the edge before its center crosses; do not demand a
            # ballistic AIR sample immediately before center-plane crossing.
            # Exact body-pair contact plus live bottom/ROI geometry is the
            # available fallback: it does not claim a verified contact point
            # or label an earlier corner contact as a front-wall climb.
            crossing_geometry = (not ground_active and within_top_xy
                and ((air and bottom[2]>=top) or loaded))
            crossing_geometry_pending = False
            if distance >= 0 and not self._history["front_edge_crossed"][leg]:
                # A previously qualified, still airborne wheel can straddle
                # the center plane with a slightly negative measured gap.
                # Missing CURRENT crossing geometry is not evidence that it
                # climbed a wall. Keep it unfinished; do not relax crossing,
                # placement, ground revocation, lateral ROI, or RR_FIRST.
                crossing_geometry_pending = bool(
                    hist_cfg.get("crossing_evidence_semantics") == "qualified_air_pending_geometry"
                    and self._failure is None
                    and self._history["active_lift"][leg] and air and within_top_xy
                    and not crossing_geometry
                    and (leg != "RL" or self._history["placed"]["RR"]))
                if self._all_stage:
                    accepted = bool(self._history["active_lift"][leg] and crossing_geometry)
                    wall = self._wall_ascent[leg]
                    positive_wheel_process = bool(len(wall) >= 2 and not self._attempt_joint_demand[leg]
                        and not self._attempt_active[leg] and all(item[2] >= .02 for item in wall)
                        and wall[-1][1]-min(item[1] for item in wall) >= hist_cfg["minimum_lift_gain_m"])
                    if accepted and leg == "RL" and not self._history["placed"]["RR"]:
                        self._fail(TaskResult.INCOMPLETE_CONTROLLER_BLOCKED, "RR_FIRST order violated: RL crossed before RR placement")
                    elif accepted:
                        self._history["front_edge_crossed"][leg] = True
                        self._event_ticks["front_edge_crossed"][leg] = tick
                    elif positive_wheel_process:
                        self._fail(TaskResult.TASK_FAILURE_WHEEL_ONLY_CLIMB,
                            f"{leg} measured powered front-wall ascent without active joint demand in this attempt")
                    else:
                        crossing_geometry_pending = True  # Missing Q is NOT a causal failure.
                elif crossing_geometry_pending:
                    pass
                elif not self._history["active_lift"][leg] or not crossing_geometry:
                    self._fail(TaskResult.TASK_FAILURE_WHEEL_ONLY_CLIMB,
                        f"{leg} crossed front without uninterrupted above-top active lift and current AIR/TOP geometry")
                elif leg == "RL" and not self._history["placed"]["RR"]:
                    self._fail(TaskResult.INCOMPLETE_CONTROLLER_BLOCKED, "RR_FIRST order violated: RL crossed before RR placement")
                else:
                    self._history["front_edge_crossed"][leg] = True; self._event_ticks["front_edge_crossed"][leg] = tick
            self._top_count[leg] = self._top_count[leg]+1 if loaded else 0
            if self._history["front_edge_crossed"][leg] and self._top_count[leg] >= hist_cfg["minimum_top_samples"] and not self._history["placed"][leg]:
                self._history["placed"][leg] = True; self._event_ticks["placed"][leg] = tick
            current[leg] = {"front_distance_m": distance, "clearance_m": bottom[2]-top,
                "top_geometry": top_geometry, "top_contact": loaded, "air": air,
                "crossing_geometry_pending": crossing_geometry_pending,
                "recent_joint_motion_deg": movement, "recent_clearance_gain_m": gain,
                "recent_whole_body_joint_motion_deg":all_joint_motion,
                "whole_body_actuation_evidence":bool(motion_evidence) if whole_body else None,
                "initial_clearance":bool(self._initial_clearance[leg]) if whole_body else False,
                "ground_contact":ground_active,"obstacle_pair_active":top_active,
                "consecutive_air_samples": self._air_count[leg], "consecutive_top_samples": self._top_count[leg],
                "within_top_xy": within_top_xy, "within_lateral_span": within_lateral_span,
                "support": force >= self.spec["support"]["force_noise_floor_n"]}
            if self._all_stage:
                current[leg].update(contact_surface=surfaces[1]["surface"],
                    top_surface_contact=top_surface, contact_reaction=bool(ground_active or top_active),
                    contact_reaction_force_n=reaction_force, bearing_force_n=force,
                    bearing_verified=all(v["bearing_verified"] for v in surfaces),
                    active_attempt=self._attempt_active[leg],
                    crossing_evidence_status=("VERIFIED" if self._history["front_edge_crossed"][leg]
                        else "UNVERIFIED" if crossing_geometry_pending else "PENDING"))
                if self._functional_rr and leg == "RR":
                    current[leg].update(functional)
                    current[leg].update(lift_established=self._rr_established,
                        initial_lift_observed=bool(self._initial_clearance[leg]),
                        front_edge_crossed=self._history["front_edge_crossed"][leg],
                        placed_on_top=self._history["placed"][leg])
            if self.spec.get("capture_retention_semantics") == CAPTURE_RETENTION_MODE:
                # Distance to the SAME measured, tolerance-expanded rectangle.
                # This extra current diagnostic never changes contact/history,
                # crossing, placement, stage entry or task success predicates.
                dx = max(front-xy_tolerance-center[0], 0., center[0]-back-xy_tolerance)
                dy = max(right-xy_tolerance-center[1], 0., center[1]-left-xy_tolerance)
                current[leg]["top_xy_outside_distance_m"] = _number(
                    math.hypot(dx, dy), f"{leg} current platform outside distance")
        total_force = sum(forces.values())
        features = {}
        for leg in LEG_ORDER:
            fraction = forces[leg]/total_force if total_force > 0 else 0.
            current[leg]["load_fraction"] = fraction
            if self._all_stage:
                # Keep the encoder finite, but never call unknown denominator
                # or unclassified wall reaction a measured low load.
                current[leg]["load_fraction_valid"] = all(v["bearing_verified"] for v in current.values()) and total_force > 0.
            features.update({f"{leg}_clearance_m": current[leg]["clearance_m"],
                             f"{leg}_front_distance_m": current[leg]["front_distance_m"],
                             f"{leg}_load_fraction": fraction})
        features.update(support_count=float(sum(v["support"] for v in current.values())),
                        body_forward_m=base_position[0]-front, body_linear_speed_m_s=_norm(base_linear),
                        body_angular_speed_rad_s=_norm(base_angular), maximum_wheel_speed_rad_s=max(map(abs,speeds)))
        final = self.spec["final"]
        base_region = front+final["minimum_body_forward_m"] <= base_position[0] <= back+geo["xy_measurement_tolerance_m"] and right-geo["xy_measurement_tolerance_m"] <= base_position[1] <= left+geo["xy_measurement_tolerance_m"]
        final_region = all(v["top_geometry"] and v["front_distance_m"] >= final["minimum_rear_wheel_forward_m"] for v in current.values()) and base_region
        home_error = max(abs(positions[name]-target) for name,target in zip(SERVO_ORDER,final["home_servo_pose_deg"]))
        controlled = (_norm(base_linear) <= final["maximum_body_linear_speed_m_s"]
            and _norm(base_angular) <= final["maximum_body_angular_speed_rad_s"]
            and max(map(abs,speeds)) <= final["maximum_wheel_speed_rad_s"]
            and max(map(abs,commands)) <= final["maximum_commanded_wheel_speed_rad_s"]
            and (final.get("stop_pose_semantics") == "physical_stable_pose"
                 or home_error <= final["home_tolerance_deg"]))
        current_support = sum(v["support"] and v["top_contact"] for v in current.values()) >= self.spec["support"]["minimum_other_supports"]
        eligible = self._failure is None and all(self._history["placed"].values()) and final_region and controlled and current_support
        self._stable_since = (now if self._stable_since is None else self._stable_since) if eligible else None
        success = self._stable_since is not None and now-self._stable_since+1e-12 >= final["stable_duration_s"]
        self._snapshot = {"valid": True, "success": success, "termination_reason": self._failure,
            "reason": self._failure_reason, "goal_features": features, "current_legs": current,
            "final_region_valid": final_region, "final_controlled": controlled, "final_support_available": current_support,
            "home_maximum_servo_error_deg": home_error, "maximum_commanded_wheel_speed_rad_s": max(map(abs,commands)),
            "final_stable_for_s": 0. if self._stable_since is None else now-self._stable_since,
            "crossing_contact_evidence": "verified_body_pair_and_live_wheel_geometry_no_contact_point_classification",
            "source": "current_episode_live_joint_geometry_exact_contact_history",
            "physics_tick": tick, "simulation_time_s": now}
        if final.get("stop_progress_semantics") == STOP_PROGRESS_MODE:
            # These are the same measured speeds and canonical applied ACK
            # commands used above, not nominal/raw actions or native readback.
            # Keep diagnostics outside the fixed 17-key actor goal features.
            self._snapshot.update(measured_wheel_velocity_rad_s=tuple(speeds),
                applied_wheel_command_rad_s=tuple(commands),
                stop_progress_wheel_order=tuple(WHEEL_ORDER))
        if self._all_stage:
            self._all_stage_finish(now=now, current=current, body_bounds=body_bounds,
                front=front, back=back, left=left, right=right,
                base_linear=base_linear, base_angular=base_angular,
                speeds=speeds, commands=commands, old_controlled=controlled)
        if self._transfer_tracker is not None:
            self._snapshot["transfer_roles"] = self._transfer_tracker.observe(observation, self.snapshot)
        self._last_tick, self._last_time = tick, now
        return self.snapshot

    def _observe_functional_rr(self, *, now, tick, bottom, center, ground,
                               obstacle_contact, top_surface, surface,
                               active_response, current_other_legs):
        """Measured RR attempt, not a hover timer or a static balance certificate.

        Reuse the existing two physics-sample noise rejection, 3/8 mm lift
        scales, whole-body actuation test and short physical history. The body
        must remain in the unchanged live safety envelope, with two verified
        OTHER ground/TOP supports; no named support set, flat posture or zero
        velocity is required. This is corroborating control evidence, not a
        proof that every future action is feasible. Exploration can continue
        while response evidence is incomplete, without claiming establishment.
        """
        cfg = self.spec["history"]
        free_diagnostic = {}
        if self._free_air_rr:
            previous = self._rr_previous_bottom_sample
            adjacent = bool(previous is not None and tick == previous[0] + 1
                and math.isclose(now-previous[1], 1./self.spec["physics_hz"],
                                 rel_tol=0., abs_tol=1e-8))
            vz = (bottom[2]-previous[2])/(now-previous[1]) if adjacent else None
            self._rr_previous_bottom_sample = (tick, now, bottom[2])
            air = not ground and not obstacle_contact
            if not adjacent:
                # An unobserved interval cannot certify continuous unsupported
                # motion or conceal a contact reset. Missing is never zero vz.
                self._rr_last_contact_bottom = None
                self._rr_free_air_count = 0
            if not air:
                self._rr_last_contact_bottom = (tick, bottom[2])
                self._rr_free_air_count = 0
            else:
                self._rr_free_air_count += 1
            reference = self._rr_last_contact_bottom
            free_gain = bottom[2]-reference[1] if air and reference is not None else None
            free_diagnostic = dict(
                unsupported_free_lift_m=free_gain,
                free_air_reference_tick=None if reference is None else reference[0],
                free_air_reference_bottom_z_m=None if reference is None else reference[1],
                consecutive_free_air_samples=self._rr_free_air_count,
                free_lift_evidence_semantics="current_continuous_AIR_rise_from_last_actual_contact",
                wheel_bottom_vz_m_s=vz, wheel_bottom_vz_observation_tick=tick,
                wheel_bottom_vz_interval_s=now-previous[1] if adjacent else None,
                wheel_bottom_vz_semantics="adjacent_measured_collider_bottom_world_z_finite_difference")
        if ground:
            if self._rr_established:
                self._lift_attempt_events.append(dict(leg="RR", event="current_lift_revoked_ground",
                    physics_tick=tick, simulation_time_s=now,
                    completed_crossing_history_retained=self._history["front_edge_crossed"]["RR"]))
            self._rr_ground_bottom = bottom[2]
            self._rr_non_ground_count = 0
            self._rr_established = False
            self._rr_geometry_samples.clear()
        else:
            self._rr_non_ground_count += 1
        edge = bool(obstacle_contact and not ground and not top_surface)
        self._rr_edge_count = self._rr_edge_count+1 if edge else 0
        self._rr_geometry_samples.append((now, tuple(center)))
        while len(self._rr_geometry_samples) > 1 and now-self._rr_geometry_samples[0][0] > cfg["window_s"]:
            self._rr_geometry_samples.popleft()
        displacement = max((_norm(tuple(a-b for a,b in zip(center, point)))
                            for _, point in self._rr_geometry_samples), default=0.)
        supports = tuple(leg for leg, row in current_other_legs.items()
            if row["support"] and (row["ground_contact"] or row["top_contact"])
            and (not self._free_air_rr or row.get("bearing_verified") is True))
        supported = len(supports) >= self.spec["support"]["minimum_other_supports"]
        self._rr_support_samples.append(supported)
        body_evidence = bool(self._failure is None and supported
            and len(self._rr_support_samples) >= cfg["minimum_air_samples"] and all(self._rr_support_samples))
        current_gain = (None if self._rr_ground_bottom is None else bottom[2]-self._rr_ground_bottom)
        repeated_non_ground = self._rr_non_ground_count >= cfg["minimum_air_samples"]
        initial = bool(self._failure is None and not ground and repeated_non_ground and active_response
            and current_gain is not None and current_gain >= cfg["minimum_initial_clearance_gain_m"])
        qualification_gain = current_gain
        if self._free_air_rr:
            qualification_gain = free_diagnostic["unsupported_free_lift_m"]
            initial = bool(self._failure is None and air and active_response
                and self._rr_free_air_count >= cfg["minimum_air_samples"]
                and qualification_gain is not None
                and qualification_gain >= cfg["minimum_initial_clearance_gain_m"])
        established_now = bool(initial and body_evidence and qualification_gain >= cfg["minimum_lift_gain_m"])
        self._rr_established |= established_now
        # An edge-hung wheel with no measured response cannot acquire or keep
        # a CURRENT swing-availability claim merely because time passes. Keep
        # the earlier event, and allow bounded probing instead of parking.
        edge_response = bool(active_response and displacement >= cfg["minimum_initial_clearance_gain_m"])
        geometry_valid = current_gain is not None and current_gain >= cfg["minimum_initial_clearance_gain_m"]
        valid = bool(self._rr_established and not ground and body_evidence and geometry_valid
            and (not edge or edge_response))
        contact_mode = ("GROUND_AND_OBSTACLE" if ground and obstacle_contact else "GROUND" if ground
                        else "AIR" if not obstacle_contact else "TOP" if top_surface else surface)
        return dict(initial_now=initial, lift_established_now=established_now,
            current_lift_valid=valid, contact_mode=contact_mode,
            motion_continuation_allowed=self._failure is None,
            motion_continuation_reason=("physical_safety_abort" if self._failure is not None
                else "current_functional_lift" if valid else "bounded_adjustment_to_acquire_measured_response"),
            ground_relative_lift_m=current_gain, recent_wheel_displacement_m=displacement,
            edge_adjustment_response_observed=edge_response if edge else None,
            body_control_evidence=body_evidence,
            body_control_evidence_semantics="unchanged_live_safety_and_short_verified_other_supports_not_static_stability_proof",
            observed_other_support_contacts=supports,
            air_duration_s=self._air_count["RR"]/self.spec["physics_hz"],
            edge_contact_duration_s=self._rr_edge_count/self.spec["physics_hz"],
            durations_are_diagnostic_only=True, **free_diagnostic)

    def _all_stage_finish(self, *, now, current, body_bounds, front, back, left, right,
                          base_linear, base_angular, speeds, commands, old_controlled):
        final, geo = self.spec["final"], self.spec["geometry"]
        low, high = body_bounds
        tolerance = geo["xy_measurement_tolerance_m"]
        body_through = bool(low[0] >= front-tolerance and high[0] <= back+tolerance
            and low[1] >= right-tolerance and high[1] <= left+tolerance)
        outside_x = max(front-tolerance-low[0], 0., high[0]-back-tolerance)
        outside_y = max(right-tolerance-low[1], 0., high[1]-left-tolerance)
        # Keep the platform target, not a request to drive off its far edge.
        # Current AIR above it is not disqualified by a thin recovery z band.
        region = body_through and all(v["within_top_xy"] and v["clearance_m"] >= geo["top_gap_min_m"]
                                     for v in current.values())
        support = sum(v["support"] and v["top_surface_contact"] for v in current.values()) >= self.spec["support"]["minimum_other_supports"]
        measured_controlled = (_norm(base_linear) <= final["maximum_body_linear_speed_m_s"]
            and _norm(base_angular) <= final["maximum_body_angular_speed_rad_s"]
            and max(map(abs, speeds)) <= final["maximum_wheel_speed_rad_s"])
        history_complete = all(self._history["placed"].values())
        evidence_ok = all(v["load_fraction_valid"] for v in current.values())
        event_now = bool(self._failure is None and evidence_ok and history_complete and region)
        if event_now and self._traversal_event_time is None:
            self._traversal_event_time = now
        controlled_now = bool(event_now and measured_controlled and support)
        if controlled_now and self._completion_observation_since is None:
            self._completion_observation_since = now
        if self._completion_observation_since is not None and not region:
            self._post_completion_loss = True
        observed = (0. if self._completion_observation_since is None else now-self._completion_observation_since)
        post_complete = self._completion_observation_since is not None and observed+1e-12 >= final["post_completion_observation_s"]
        if post_complete and self._post_completion_loss and self._failure is None:
            self._fail(TaskResult.INCOMPLETE_CONTROLLER_BLOCKED, "controlled completion lost during fixed post-completion observation")
            self._termination_source = "POST_COMPLETION_LOSS"
        elif post_complete and not controlled_now and self._failure is None:
            self._fail(TaskResult.INCOMPLETE_CONTROLLER_BLOCKED, "not currently controlled at fixed post-completion observation end")
            self._termination_source = "POST_COMPLETION_LOSS"
        success = bool(post_complete and controlled_now and not self._post_completion_loss and self._failure is None)
        self._snapshot.update(
            evaluator_version="all_stage_v1", run_validity="VALID",
            physical_evidence_status="VERIFIED" if evidence_ok else "CONTACT_BEARING_UNVERIFIED",
            traversal_event_observed=self._traversal_event_time is not None,
            traversal_event_time_s=self._traversal_event_time,
            traversal_task_complete=controlled_now and self._failure is None,
            task_completed_controlled=controlled_now and self._failure is None,
            body_traversal_geometry=dict(valid=True, minimum_w_m=low, maximum_w_m=high,
                whole_body_in_platform_region=body_through,
                outside_platform_distance_m=math.hypot(outside_x, outside_y)),
            final_region_valid=region, final_controlled=measured_controlled,
            final_support_available=support,
            strict_recovery_quality=dict(passed=bool(old_controlled and self._snapshot["final_stable_for_s"] >= final["stable_duration_s"]),
                controlled_stop=old_controlled, stable_for_s=self._snapshot["final_stable_for_s"],
                commanded_wheels_within_tolerance=max(map(abs, commands)) <= final["maximum_commanded_wheel_speed_rad_s"],
                home_error_deg=self._snapshot["home_maximum_servo_error_deg"]),
            post_completion_observation_s=observed,
            post_completion_observation_started=self._completion_observation_since is not None,
            post_completion_elapsed_s=min(observed, final["post_completion_observation_s"]),
            post_completion_observation_complete=post_complete,
            post_completion_loss_observed=self._post_completion_loss,
            success=success, termination_reason=self._failure, reason=self._failure_reason,
            termination_source=self._termination_source or ("POST_COMPLETION_OBSERVATION" if success else None),
            crossing_contact_evidence="active_attempt_plus_surface_contact_or_air_geometry_v1")
        # The existing completion progress consumes this elapsed field. Strict
        # recovery retains its independent old stable-time value above.
        self._snapshot["final_stable_for_s"] = min(observed, final["post_completion_observation_s"])

    observe_and_update = observe


class TaskStageSupervisor:
    """Forward-only semantic stages; motion endpoints never veto exploration."""
    def __init__(self, task_spec_path: Path | str = DEFAULT_TASK_SPEC_PATH, *,
                 evaluator: TaskEvaluator | None = None, initial_stage_id: str = "P01"):
        self.spec = load_task_spec(task_spec_path)
        if initial_stage_id not in PHASE_IDS:
            raise ValueError("unknown initial task stage")
        self.evaluator = evaluator or TaskEvaluator(spec=self.spec)
        self.stage_id = initial_stage_id
        self.completed_stage_ids: list[str] = []
        self.transition_evidence: list[dict[str, Any]] = []
        self.stage_started_s: float | None = None
        self.episode_started_s: float | None = None
        self.termination_reason: str | None = None
        self.termination_source: str | None = None
        self._snapshot: dict[str, Any] = {}
        self._last_observation_tick: int | None = None
        self._progress_samples: deque = deque()
        self._capture_continuation = _capture_continuation_enabled(self.spec)
        self._fl_pending_handoff: dict[str, Any] | None = None
        self._p05_local_deadline_warning = False
        self.rr_capture_feedback = None
        rr_mode = self.spec.get("rr_capture_continuation_semantics")
        if rr_mode not in (None, "rr_capture_then_rl_transfer_v1"):
            raise ValueError("unknown RR capture continuation semantics")
        from .semantic_rr_capture_context import rr_contact_handoff_window_s
        self._rr_contact_handoff_window_s = rr_contact_handoff_window_s(self.spec)

    def predicate(self, name: str, evaluation: Mapping[str, Any]) -> float:
        if name == "physical_valid":
            return float(evaluation["valid"] and evaluation["termination_reason"] is None)
        if name == "whole_task_success":
            if evaluation["success"]: return 1.
            final=self.spec["final"]; features=evaluation["goal_features"]
            if self.spec.get("physical_acceptance_version") == "all_stage_v1":
                # Existing phase_progress slot carries a reversible finite
                # acceptance state, separately from the dense physical Phi.
                # Disjoint bands expose the timer start (including elapsed=0)
                # and irreversible region-loss latch without changing372.
                observed = _clip(float(evaluation.get("post_completion_elapsed_s", 0.)) / final["post_completion_observation_s"])
                if evaluation.get("post_completion_observation_started", False):
                    return (.80+.19*observed if evaluation.get("post_completion_loss_observed", False)
                            else .50+.25*observed)
                return .25 if evaluation.get("traversal_event_observed", False) else 0.
            history_fraction=sum(evaluation["history"]["placed"].values())/4.
            forward=min(_clip(features["body_forward_m"]/final["minimum_body_forward_m"]),
                        min(_clip(features[f"{leg}_front_distance_m"]/final["minimum_rear_wheel_forward_m"]) for leg in LEG_ORDER))
            if final.get("stop_progress_semantics") == STOP_PROGRESS_MODE:
                order = evaluation.get("stop_progress_wheel_order")
                if not isinstance(order, (tuple, list)) or tuple(order) != WHEEL_ORDER:
                    raise SemanticObservationError("continuous stop progress requires canonical measured wheel order")
                speeds = _vector(evaluation.get("measured_wheel_velocity_rad_s"), 4, "measured wheel velocities")
                commands = _vector(evaluation.get("applied_wheel_command_rad_s"), 4, "applied wheel commands")

                def ratio(value, tolerance_key):
                    magnitude = abs(_number(value, "continuous stop measurement"))
                    tolerance = _number(final[tolerance_key], "continuous stop tolerance")
                    if tolerance <= 0:
                        raise SemanticObservationError("continuous stop tolerance must be positive")
                    return tolerance / (tolerance + magnitude)

                # Four types, not ten separately weighted components. Replace
                # the legacy aggregation; do not append its command term again.
                # q(T)=.5 is soft progress, not a changed hard stopping threshold.
                stop_terms = [sum(ratio(v, "maximum_wheel_speed_rad_s") for v in speeds) / 4.,
                    sum(ratio(v, "maximum_commanded_wheel_speed_rad_s") for v in commands) / 4.,
                    ratio(features["body_linear_speed_m_s"], "maximum_body_linear_speed_m_s"),
                    ratio(features["body_angular_speed_rad_s"], "maximum_body_angular_speed_rad_s")]
            else:
                stop_terms=[_clip(1.-features["maximum_wheel_speed_rad_s"]/final["maximum_wheel_speed_rad_s"]),
                          _clip(1.-features["body_linear_speed_m_s"]/final["maximum_body_linear_speed_m_s"]),
                          _clip(1.-features["body_angular_speed_rad_s"]/final["maximum_body_angular_speed_rad_s"])]
                if final.get("stop_pose_semantics") != "physical_stable_pose":
                    stop_terms.append(_clip(1.-evaluation["home_maximum_servo_error_deg"]/final["home_tolerance_deg"]))
                if final.get("stop_command_progress") == "reciprocal_physical_stop_tolerance":
                    command = _number(evaluation["maximum_commanded_wheel_speed_rad_s"], "maximum wheel command")
                    tolerance = final["maximum_commanded_wheel_speed_rad_s"]
                    # Preserve the previous opt-in formula exactly for old MDPs.
                    stop_terms.append(1. if command <= tolerance else tolerance/command)
            stop=sum(stop_terms)/len(stop_terms)
            settle=_clip(evaluation["final_stable_for_s"]/final["stable_duration_s"])
            return min(.99,.4*history_fraction+.3*forward+.2*stop+.1*settle)
        if name == "rear_approach":
            return min(self.predicate("edge_proximity_RL", evaluation), self.predicate("edge_proximity_RR", evaluation))
        kind, leg = name.rsplit("_", 1)
        history = evaluation["history"]
        current = evaluation["current_legs"][leg]; geo = self.spec["geometry"]
        if kind in ("role_prepared", "transfer_ready"):
            # A genuinely qualified downstream motion is already an entrance,
            # not a reason to land/recreate a preparation pose. Current support
            # remains independently measured; historical placement is not force.
            if self.spec.get("p09_lift_semantics") in FUNCTIONAL_RR_MODES and leg == "RR":
                if current.get("current_lift_valid") or _current_rr_placement_usable(evaluation):
                    return 1.
            elif (evaluation["termination_reason"] is None and
                    (history["placed"][leg] or (history["active_lift"][leg] and current["air"]))):
                return 1.
            row = evaluation.get("transfer_roles", {}).get(leg, {})
            flag = "preparation_ready" if kind == "role_prepared" else "transfer_ready"
            progress = "preparation_progress" if kind == "role_prepared" else "transfer_progress"
            return 1. if row.get(flag) is True else min(.99, float(row.get(progress, 0.)))
        if kind == "lifted":
            functional_rr = self.spec.get("p09_lift_semantics") in FUNCTIONAL_RR_MODES and leg == "RR"
            if functional_rr:
                if current.get("current_lift_valid") or _current_rr_placement_usable(evaluation): return 1.
            elif history["active_lift"][leg]: return 1.
            cfg = self.spec["history"]
            # Dense measurements provide a gradient of task progress, but only
            # the joint+clearance+AIR chronology can produce completion (=1).
            if cfg.get("lift_motion_evidence") == "whole_body_actuation":
                return .99 * (_clip(current["recent_clearance_gain_m"]/cfg["minimum_lift_gain_m"])
                    + float(current.get("whole_body_actuation_evidence",False))
                    + _clip(current["consecutive_air_samples"]/cfg["minimum_air_samples"])) / 3.
            return .99 * ( _clip(current["recent_clearance_gain_m"]/cfg["minimum_lift_gain_m"])
                + _clip(current["recent_joint_motion_deg"]/cfg["minimum_joint_motion_deg"])
                + _clip(current["consecutive_air_samples"]/cfg["minimum_air_samples"]) ) / 3.
        if kind == "placed":
            if history["placed"][leg]:
                if not (self.spec.get("p09_lift_semantics") in FUNCTIONAL_RR_MODES and leg == "RR"):
                    return 1.
                if _current_rr_placement_usable(evaluation): return 1.
            lift = self.predicate(f"lifted_{leg}", evaluation)
            crossing = 1. if history["front_edge_crossed"][leg] else _clip(1.+current["front_distance_m"]/.20)
            captured = (.5 * float(current["top_geometry"])
                + .5 * _clip(current["consecutive_top_samples"]/self.spec["history"]["minimum_top_samples"]))
            return min(.99, .35*lift + .35*crossing + .30*captured)
        support = sum(v["support"] for key,v in evaluation["current_legs"].items() if key != leg)
        support_available = support >= self.spec["support"]["minimum_other_supports"]
        if kind == "support": return float(support_available)
        if kind == "load_ready":
            if not support_available: return 0.
            if self.spec.get("p09_lift_semantics") in FUNCTIONAL_RR_MODES and current.get("load_fraction_valid") is False:
                return 0.  # Missing denominator is not measured unloading.
            limit = self.spec["support"]["unloaded_leg_maximum_load_fraction"]
            return _clip((1.-current["load_fraction"])/(1.-limit))
        if kind in ("approach", "workspace", "edge_proximity"):
            if not current["within_lateral_span"]: return 0.
            if kind == "edge_proximity": kind = "workspace"  # Legacy config keys are explicitly geometry-only.
            lower, upper = geo[f"{kind}_min_m"], geo[f"{kind}_max_m"]
            x = current["front_distance_m"]
            if self.spec.get("physical_acceptance_version") == "all_stage_v1":
                # A current downstream crossing/capture is a legitimate entry,
                # not a demand to reverse to an obsolete narrow interval.
                advanced = (history["front_edge_crossed"][leg] or history["placed"][leg])
                if advanced and not current["ground_contact"] and (current["air"] or current["top_contact"]):
                    return 1.
                # Reuse the declared measurement tolerance symmetrically, not
                # the previous run's exact sub-millimetre deficit as a cutoff.
                margin = geo["xy_measurement_tolerance_m"]
                lower, upper = lower-margin, upper+margin
            return 1. if lower <= x <= upper else _clip(1.-min(abs(x-lower),abs(x-upper))/.25)
        if kind == "clear":
            if (self.spec.get("physical_acceptance_version") == "all_stage_v1"
                    and (history["front_edge_crossed"][leg] or history["placed"][leg])
                    and current["within_top_xy"] and not current["ground_contact"]
                    and (current["top_contact"] or (current["air"] and current["clearance_m"] >= 0.))):
                return 1.  # Already carried/captured: do not demand a second hop.
            return _clip(max(0.,current["clearance_m"])/geo["airborne_clearance_above_top_m"])
        raise ValueError(f"unknown task predicate {name}")

    def entry_report(self, stage_id: str, evaluation: Mapping[str, Any] | None = None) -> dict[str, Any]:
        ev = self.evaluator.snapshot if evaluation is None else evaluation
        conditions = self.spec["stages"][stage_id]["valid_start_conditions"]
        values = {name: self.predicate(name, ev) if ev["valid"] else 0. for name in conditions}
        waived = []
        if (self._capture_continuation and stage_id in PHASE_IDS[5:]
                and values.get("placed_FL", 1.) < 1.
                and _capture_continuation_status(self.spec, ev, stage_id)["allow_capture_continuation"]):
            waived.append("placed_FL")
        reasons = [k for k, v in values.items() if v < 1. and k not in waived]
        result = {"valid": not reasons, "reasons": reasons, "values": values}
        if self._capture_continuation:
            result.update(continuation_waived_conditions=waived,
                          actual_conditions_satisfied=all(v >= 1. for v in values.values()),
                          waiver_semantics="scheduling_only_no_physical_completion_credit")
        return result

    def physical_potential(self, evaluation: Mapping[str,Any]) -> float:
        """One phase-label-independent potential over physical progress/history."""
        if not evaluation.get("valid"): return 0.
        history=evaluation["history"]; legs=evaluation["current_legs"]
        values=[]
        for leg in LEG_ORDER:
            current=legs[leg]
            if history["placed"][leg]:
                # Keep completed events/ordering, but reuse the original .2
                # capture share for current region retention. AIR above the
                # platform is fully eligible: no fixed contact or stop pose.
                retention = (self._current_capture_retention(leg,evaluation)
                    if self.spec.get("capture_retention_semantics") == CAPTURE_RETENTION_MODE else 1.)
                values.append(.8+.2*retention); continue
            if not placement_predecessors_satisfied(self.spec, history, leg):
                # Workspace preparation can precede another leg's placement.
                # Reuse only its existing weight; unload/lift/carry/capture
                # remain predecessor-gated and no history is awarded here.
                preparation=(self._workspace_potential_progress(leg,evaluation)
                    if self.spec.get("preparation_credit_semantics") == PREPARATION_CREDIT_MODE else 0.)
                values.append(.1*preparation); continue
            if leg == "RL" and self.spec["nominal"].get("rear_policy_timing"):
                from .semantic_rear_policy_timing import rear_dependency
                dependency = rear_dependency({"physical_evaluator": evaluation}, self.spec["support"])
                if not (dependency["support_transfer_permitted"] or dependency["rl_current_swing"]):
                    # Historical RR placement cannot reward a fresh RL unload
                    # after losing the support on which that transfer depends.
                    # Keep the existing small preparation share and allow an
                    # already qualified live RL swing to continue/land.
                    values.append(.1*self._workspace_potential_progress(leg, evaluation))
                    continue
            workspace=self._workspace_potential_progress(leg,evaluation)
            unload=(evaluation.get("transfer_roles", {}).get(leg, {}).get("transfer_progress", 0.)
                    if self.spec.get("transfer_roles") else self.predicate(f"load_ready_{leg}",evaluation))
            initial=float(current.get("initial_clearance",False))
            hard_lift=bool(history["active_lift"][leg])
            if self.spec.get("transfer_roles") and hard_lift and history["front_edge_crossed"][leg]:
                unload = 1.  # Retire transfer credit independently of capture shaping variant.
            lift_credit=float(hard_lift)
            if self.spec.get("lift_credit_semantics") == LIFT_CREDIT_MODE:
                lift_credit=self._current_lift_credit(leg,evaluation)
            carry=(1. if history["front_edge_crossed"][leg] else _clip(1.+current["front_distance_m"]/.25)) if hard_lift else 0.
            capture=_clip(current["consecutive_top_samples"]/self.spec["history"]["minimum_top_samples"]) if history["front_edge_crossed"][leg] else 0.
            if self.spec.get("capture_approach_semantics") in CAPTURE_APPROACH_MODES:
                # Retire preparation's existing .1 share after qualified
                # crossing: subsequent capture loading is not regression.
                # This local progress credit does not claim that measured
                # load_ready was ever 1 or alter current support/history.
                if hard_lift and history["front_edge_crossed"][leg]:
                    unload=1.
                capture=self._current_capture_progress(leg,evaluation,capture)
            values.append(.1*workspace+.1*unload+.25*(.25*initial+.75*lift_credit)+.35*carry+.2*capture)
        finish = ((self._current_finish_progress(evaluation)
                   if self.spec.get("physical_acceptance_version") == "all_stage_v1"
                   else self.predicate("whole_task_success", evaluation))
                  if all(history["placed"].values()) else 0.)
        return min(1.,.85*sum(values)/4.+.15*finish)

    def _current_finish_progress(self, evaluation: Mapping[str, Any]) -> float:
        """Single existing finish share: current geometry, actual rates, post time.

        This is not an acceptance predicate. Neither historical traversal nor
        matching command/home targets can substitute for present measurements.
        """
        if evaluation.get("physical_evidence_status") != "VERIFIED":
            return 0.
        geometry = evaluation["body_traversal_geometry"]
        outside = _number(geometry["outside_platform_distance_m"], "body platform outside distance")
        if not geometry["valid"] or outside < 0.:
            raise SemanticObservationError("finish progress requires valid current body rectangle")
        body = .25/(.25+outside)  # Existing carry/retention distance scale.
        final = self.spec["final"]; features = evaluation["goal_features"]
        if tuple(evaluation.get("stop_progress_wheel_order", ())) != WHEEL_ORDER:
            raise SemanticObservationError("finish progress requires measured wheel order")
        speeds = _vector(evaluation.get("measured_wheel_velocity_rad_s"), 4, "finish measured wheels")

        def ratio(value, key):
            magnitude = abs(_number(value, "finish measured rate"))
            threshold = _number(final[key], "finish rate threshold")
            if threshold <= 0.:
                raise SemanticObservationError("finish rate threshold must be positive")
            return 1. if magnitude <= threshold else threshold/magnitude

        speed = (sum(ratio(v, "maximum_wheel_speed_rad_s") for v in speeds)/4.
                 + ratio(features["body_linear_speed_m_s"], "maximum_body_linear_speed_m_s")
                 + ratio(features["body_angular_speed_rad_s"], "maximum_body_angular_speed_rad_s"))/3.
        support = _clip(sum(v["support"] and v["top_surface_contact"]
                            for v in evaluation["current_legs"].values())
                        / self.spec["support"]["minimum_other_supports"])
        controlled = body*speed*support
        observed = _clip(evaluation["post_completion_elapsed_s"]/final["post_completion_observation_s"])
        if evaluation["post_completion_loss_observed"]:
            observed = 0.
        return .65*body + .25*controlled + .10*controlled*observed

    def _workspace_potential_progress(self, leg: str, evaluation: Mapping[str, Any]) -> float:
        """Soft preparation credit only; never an entry/completion predicate.

        The reciprocal can round to 1 just outside a floating-point boundary.
        Public predicate/phase progress and nominal supervision deliberately
        retain their original physical interval rule and never call this path.
        """
        if self.spec.get("workspace_potential_semantics") is None:
            return self.predicate(f"workspace_{leg}", evaluation)
        if not _workspace_potential_enabled(self.spec):
            raise ValueError("workspace potential mode unavailable")
        current, geometry = evaluation["current_legs"][leg], self.spec["geometry"]
        edge = interval_distance_progress(
            current["front_distance_m"], geometry["workspace_min_m"], geometry["workspace_max_m"],
            current["within_lateral_span"])
        if self.spec.get("transfer_roles"):
            # Replace part of the existing .1 preparation budget, not a new reward.
            role = evaluation.get("transfer_roles", {}).get(leg, {})
            receiver = float(role.get("workspace_progress", 0.))
            if _current_rr_receiver_preparation_retired(self.spec, leg, evaluation):
                # Qualified current crossing has demonstrated usable space.
                # Preserve its existing credit, not ongoing FL contraction.
                # Keep the independent current RR edge/capture/lift goals.
                receiver = 1.
            return .5*edge+.5*receiver
        return edge

    def _current_capture_progress(self, leg: str, evaluation: Mapping[str,Any], contact_fraction: float) -> float:
        """Soft first-capture approach; never a touchdown or phase completion."""
        history=evaluation["history"]
        if not (history["active_lift"][leg] and history["front_edge_crossed"][leg]):
            return 0.
        current=evaluation["current_legs"][leg]
        outside=_number(current.get("top_xy_outside_distance_m"), f"{leg} capture approach outside distance")
        if outside < 0.:
            raise SemanticObservationError("capture approach outside distance must be nonnegative")
        clearance=_number(current["clearance_m"], f"{leg} capture approach clearance")
        if leg == "FL" and self.spec.get("capture_approach_semantics") == FL_CAPTURE_APPROACH_MODE:
            # Replace only the old FL geometric half-share. Outside the legal
            # top region, workspace/carry already reward horizontal approach;
            # do not reward descent toward a wall. This never awards placed.
            proximity = 0.
            if current.get("within_top_xy") is True and current.get("within_lateral_span") is True:
                cfg = self.spec["fl_capture_potential"]
                gap = max(0., clearance)  # Penetration receives no extra credit.
                fine = cfg["fine_fraction"]
                proximity = ((1.-fine)/(1.+gap/cfg["coarse_gap_scale_m"])
                             + fine/(1.+gap/cfg["fine_gap_scale_m"]))
            # Already-placed legs bypass this helper in physical_potential;
            # their current retention target has no further descent incentive.
            return .5*proximity+.5*contact_fraction
        scale=self.spec["geometry"]["top_gap_max_m"]
        xy=_clip(1.-outside/.25)  # Reuse the existing carry/retention decay length.
        proximity=xy*scale/(scale+abs(clearance))
        # Reallocate the existing .2 capture share, not an extra reward. AIR
        # geometry earns at most half; the rest still needs real TOP samples.
        # Placed legs bypass this helper so later legitimate AIR is unchanged.
        return .5*proximity+.5*contact_fraction

    def _current_capture_retention(self, leg: str, evaluation: Mapping[str,Any]) -> float:
        current=evaluation["current_legs"][leg]
        from .semantic_rear_policy_timing import RECAPTURE_MODE, rear_dependency
        if leg == "RR" and self.spec["nominal"].get("rear_policy_timing") == RECAPTURE_MODE:
            dependency = rear_dependency({"physical_evaluator": evaluation}, self.spec["support"])
            if not (dependency["rl_current_swing"] or evaluation["history"]["placed"].get("RL") is True):
                # Only the existing .2 retention share changes. Historical .8
                # event credit, qualification and completion history stay put.
                # A legitimate RL swing/placed continuation retains legacy AIR.
                legal = bool(evaluation.get("valid") is True
                    and evaluation.get("termination_reason") is None
                    and current.get("within_top_xy") is True
                    and current.get("within_lateral_span") is True
                    and current.get("ground_contact") is False
                    and (dependency["rr_top_contact"] or
                         (current.get("air") is True and current.get("obstacle_pair_active") is False)))
                if not legal:
                    return 0.
                clearance = _number(current["clearance_m"], "RR current recapture clearance")
                if clearance < self.spec["geometry"]["top_gap_min_m"]:
                    return 0.
                contact = (_clip(current["consecutive_top_samples"] / self.spec["history"]["minimum_top_samples"])
                           if dependency["rr_current_bearing"] else 0.)
                # Reuse existing bounded XY/gap and geometric/contact halves.
                # AIR can approach the top, but cannot earn the contact half.
                return self._current_capture_progress("RR", evaluation, contact)
        outside=_number(current.get("top_xy_outside_distance_m"), f"{leg} current platform outside distance")
        if outside < 0.:
            raise SemanticObservationError("current platform outside distance must be nonnegative")
        clearance=_number(current["clearance_m"], f"{leg} capture clearance")
        xy=_clip(1.-outside/.25)  # Existing workspace/carry decay length.
        scale=self.spec["history"]["minimum_lift_gain_m"]
        gap=max(0.,self.spec["geometry"]["top_gap_min_m"]-clearance)
        vertical=scale/(scale+gap)
        retention = min(xy,vertical)
        cfg = self.spec.get("rolling_capture_retention")
        if cfg is None or leg not in ("FR", "FL"):
            return retention
        placed, legs = evaluation["history"]["placed"], evaluation["current_legs"]
        if (not all(placed[p] for p in ("FR", "FL")) or placed["RR"]
                or legs["RR"].get("current_lift_valid") is True):
            return retention
        rear = max(_number(legs[p]["front_distance_m"], f"{p} rolling edge distance") for p in ("RL", "RR"))
        weight = _clip((cfg["rear_preparation_near_m"]-rear)/cfg["blend_distance_m"])
        # Current support is a boolean measured condition, never a reward for
        # greater force. Reuse the existing capture share, with no new event.
        contact = bool(current.get("top_contact") is True and current.get("top_surface_contact") is True
            and current.get("support") is True and current.get("bearing_verified") is True
            and current.get("air") is False and current.get("ground_contact") is False)
        proximity = 1./(1.+max(0.,clearance)/cfg["positive_gap_scale_m"])
        usable = (1.-cfg["contact_fraction"])*proximity+cfg["contact_fraction"]*float(contact)
        return retention*((1.-weight)+weight*usable)

    def _current_lift_credit(self, leg: str, evaluation: Mapping[str,Any]) -> float:
        if evaluation.get("valid") is not True or evaluation.get("termination_reason") is not None: return 0.
        current=evaluation["current_legs"][leg]; history=evaluation["history"]
        clearance=_number(current["clearance_m"],f"{leg} current top clearance")
        scale=self.spec["history"]["minimum_lift_gain_m"]
        fraction=scale/(scale+max(0.,-clearance))
        if self.spec.get("p09_lift_semantics") in FUNCTIONAL_RR_MODES and leg == "RR":
            if current.get("current_lift_valid"):
                return 1. if history["front_edge_crossed"][leg] else fraction
            return 0.
        if history["active_lift"][leg]:
            return 1. if history["front_edge_crossed"][leg] else fraction
        cfg=self.spec["history"]
        eligible=(current.get("soft_air_actuation_earned") is True and current["initial_clearance"]
            and current["air"] and not current["ground_contact"] and not current["obstacle_pair_active"]
            and current["within_lateral_span"]
            and cfg["near_front_min_m"]<=current["front_distance_m"]<=cfg["near_front_max_m"])
        return .99*fraction if eligible else 0.

    def observe_and_update(self, observation: Any, *, sim_time_s: float | None = None) -> dict[str, Any]:
        evaluation = self.evaluator.observe(observation)
        if self._last_observation_tick == _get(observation, "physics_tick"):
            return dict(self._snapshot)
        now = _number(_get(observation,"simulation_time_s") if sim_time_s is None else sim_time_s,"task time")
        if self.stage_started_s is None: self.stage_started_s = now
        if self.episode_started_s is None: self.episode_started_s = now
        continuation = (_capture_continuation_status(self.spec, evaluation, self.stage_id)
                        if self._capture_continuation else None)
        # A delayed true measurement completes the old task in place. It does
        # not rewind the scheduler, duplicate a source layer, or claim touchdown
        # at the earlier permission-to-continue tick.
        if (self._fl_pending_handoff is not None and evaluation["history"]["placed"]["FL"]
                and "P05" not in self.completed_stage_ids):
            self.completed_stage_ids.append("P05")
            self.completed_stage_ids.sort(key=PHASE_IDS.index)
            self.transition_evidence.append({"from_stage": "P05", "to_stage": self.stage_id,
                "sim_time_s": now, "physics_tick": _get(observation, "physics_tick"),
                "reason": "late measured FL placement completes pending capture without scheduler rewind",
                "event_kind": "pending_capture_physically_completed", "scheduler_transition": False,
                "physical_completion_awarded": True, "history": evaluation["history"]})
        stage = self.spec["stages"][self.stage_id]
        entry = self.entry_report(self.stage_id, evaluation)
        goal_values = {name: self.predicate(name,evaluation) if evaluation["valid"] else 0. for name in stage["completion_predicates"]}
        progress = sum(goal_values.values())/len(goal_values)
        takeover = False
        if (self.spec.get("transfer_roles") and evaluation["valid"]
                and self.stage_id in ("P01", "P04", "P06", "P07", "P08", "P10", "P11")):
            leg = stage["active_leg"]; current = evaluation["current_legs"][leg]
            history = evaluation["history"]
            takeover = bool(history["active_lift"][leg] and not current["ground_contact"]
                            and current["within_lateral_span"] and (current["air"] or current["top_contact"]))
            if self.spec.get("p09_lift_semantics") in FUNCTIONAL_RR_MODES and leg == "RR":
                takeover = bool(current.get("current_lift_valid") and current["within_lateral_span"])
        if evaluation["termination_reason"] is not None:
            self.termination_reason = evaluation["termination_reason"]
            self.termination_source = evaluation.get("termination_source", "PHYSICAL_EVALUATOR")
        pending_handoff = bool(continuation and self.stage_id == "P05"
            and continuation["allow_capture_continuation"]
            and now-self.stage_started_s >= float(stage["maximum_task_duration"]))
        if continuation and self.stage_id == "P05" and now-self.stage_started_s >= float(stage["maximum_task_duration"]):
            self._p05_local_deadline_warning = True
        # At most one forward scheduler transition per physical observation.
        # Late physical completion above is separate, never replayed source
        # motion or a phase-label success bonus.
        if self.termination_reason is None and entry["valid"] and (pending_handoff or takeover or all(v >= 1. for v in goal_values.values())) and _get(observation,"physics_tick") % 8 == 0:
            previous = self.stage_id
            if not pending_handoff and previous not in self.completed_stage_ids:
                self.completed_stage_ids.append(previous)
            next_stage = stage["next_phase"]
            self.transition_evidence.append({"from_stage": previous, "to_stage": next_stage, "sim_time_s": now,
                "physics_tick": _get(observation,"physics_tick"), "reason": (
                    "P05 local warning escalated to measured pending-capture continuation; FL not placed" if pending_handoff else
                    "qualified downstream motion already active; continuous takeover" if takeover else "current physical goal set satisfied"),
                "continuous_takeover": takeover,
                "entry": entry, "completion_values": goal_values, "history": evaluation["history"]})
            if self._capture_continuation:
                self.transition_evidence[-1].update(scheduler_transition=True,
                    physical_completion_awarded=not pending_handoff, fl_capture_pending=bool(pending_handoff))
            if pending_handoff:
                self._fl_pending_handoff = {"physics_tick": _get(observation, "physics_tick"), "sim_time_s": now}
            if next_stage == "SUCCESS":
                self.termination_reason = "SUCCESS"
                self.termination_source = "COMMON_PHYSICAL_TASK_COMPLETE"
            else:
                self.stage_id = next_stage; self.stage_started_s = now; self._progress_samples.clear()
                stage = self.spec["stages"][self.stage_id]; entry = self.entry_report(self.stage_id,evaluation)
                goal_values = {name:self.predicate(name,evaluation) for name in stage["completion_predicates"]}
                progress = sum(goal_values.values())/len(goal_values)
        age = now-self.stage_started_s; episode_age = now-self.episode_started_s
        if self._capture_continuation:
            continuation = _capture_continuation_status(self.spec, evaluation, self.stage_id)
        local_warning_only = bool(self._capture_continuation and
            (self.stage_id == "P05" or (self.stage_id in PHASE_IDS[5:]
                and continuation["fl_capture_pending"])))
        if (self.stage_id == "P05" and self.spec.get("p05_finite_recovery_timeout_semantics")
                == P05_FINITE_RECOVERY_TIMEOUT_MODE):
            # Preserve the original finite approach opportunity, not an
            # unlimited exemption based on old crossing history. Current
            # legal capture already covers safe descent and can hand over
            # at the next decision tick. Never advance an assist here.
            recovery_end = (float(stage["maximum_task_duration"])
                + float(self.spec["local_timeout_policy"]["maximum_extension_s"]))
            completed_handoff_pending = bool(entry["valid"] and evaluation["valid"]
                and self.termination_reason is None and all(v >= 1. for v in goal_values.values())
                and _get(observation, "physics_tick") % 8 != 0)
            local_warning_only = bool(age < recovery_end
                or continuation["allow_capture_continuation"] or completed_handoff_pending)
        rr_recovery = None
        if self.spec.get("rr_capture_continuation_semantics") is not None:
            from .semantic_rr_capture_context import rr_capture_transfer_context
            feedback = self.rr_capture_feedback
            fresh = bool(isinstance(feedback, Mapping)
                and feedback.get("episode_observation_tick") == _get(observation, "physics_tick"))
            rr_recovery = rr_capture_transfer_context(
                task={"physical_evaluator":evaluation,"termination_reason":self.termination_reason,
                      "stage_id":self.stage_id,"entry_valid":entry["valid"],
                      "completion_values":goal_values},
                observation=observation, support_spec=self.spec["support"],
                assist_snapshot=feedback["state"] if fresh else None,
                contact_handoff_window_s=self._rr_contact_handoff_window_s)
            rr_recovery["committed_feedback_matches_current_tick"] = fresh
            rr_recovery["local_warning_only"] = bool(self.stage_id == "P09" and fresh
                and rr_recovery["rr_capture_recovery_allowed"])
            local_warning_only = local_warning_only or rr_recovery["local_warning_only"]
        local_limit = float(stage["maximum_task_duration"])
        allowance = 0.
        post_window_allowance = 0.
        if self.spec.get("physical_acceptance_version") == "all_stage_v1":
            policy = self.spec["local_timeout_policy"]
            # Stateless, bounded recovery allowance from current physical goal
            # progress. No jitter-reset clock, hidden renewals or global inflate.
            # All operands (phase, progress, elapsed) already occur in372.
            allowance = min(float(policy["maximum_extension_s"]),
                local_limit * float(policy["fraction_of_original_limit"])) * _clip(progress)**2
            if self.stage_id == "P13" and evaluation.get("post_completion_observation_started", False):
                # Preserve the already-started fixed observation, never renew
                # it. Both age and remaining post time are observable. The
                # global200s deadline is still authoritative.
                remaining_post = max(0., self.spec["final"]["post_completion_observation_s"]
                                     - evaluation["post_completion_elapsed_s"])
                post_window_allowance = max(0., age+remaining_post-local_limit-allowance)
        if self.termination_reason is None:
            if episode_age >= self.spec["episode_maximum_duration_s"]:
                self.termination_reason = TaskResult.INCOMPLETE_CONTROLLER_BLOCKED.value
                self.termination_source = "GLOBAL_FINITE_TASK_DEADLINE"
            elif not local_warning_only and age >= local_limit + allowance + post_window_allowance:
                self.termination_reason = TaskResult.INCOMPLETE_CONTROLLER_BLOCKED.value
                self.termination_source = "LOCAL_BOUNDED_RECOVERY_EXHAUSTED" if allowance else "LOCAL_TASK_DEADLINE"
        self._progress_samples.append((now,progress))
        stall = stage["stall_diagnostic"]
        while len(self._progress_samples)>1 and now-self._progress_samples[0][0]>stall["window_s"]:
            self._progress_samples.popleft()
        stalled = bool(self._progress_samples and now-self._progress_samples[0][0] >= stall["window_s"]-1/120
            and max(v for _,v in self._progress_samples)-min(v for _,v in self._progress_samples)<stall["minimum_potential_change"])
        histories=evaluation["history"]
        self._snapshot={"schema":"wlr50_clean.semantic_task.v2", "stage_id":self.stage_id,"purpose":stage["purpose"],
            "phase_progress":progress,"task_progress_potential":min(1.,(len(self.completed_stage_ids)+(0. if self.termination_reason=="SUCCESS" else progress))/13),
            "goal_features":evaluation["goal_features"],"history":histories,
            "active_lift_history":histories["active_lift"],"front_edge_crossed_history":histories["front_edge_crossed"],"placed_history":histories["placed"],
            "completed_stage_ids":list(self.completed_stage_ids),"success":self.termination_reason=="SUCCESS",
            "termination_reason":self.termination_reason,"entry_valid":entry["valid"],"entry_reasons":entry["reasons"],
            "termination_source":self.termination_source,
            "local_timeout": {"nominal_limit_s":local_limit, "current_progress_allowance_s":allowance,
                "fixed_post_window_allowance_s":post_window_allowance,
                "effective_limit_s":local_limit+allowance+post_window_allowance, "nominal_limit_exceeded":age>=local_limit,
                "classification":"finite_task_terminal_not_external_truncation",
                "timer_inputs_observable_in_existing372":True},
            "completion_values":goal_values,"stage_age_s":age,"stage_elapsed_s":age,
            "remaining_task_time_s":max(0.,self.spec["episode_maximum_duration_s"]-episode_age),
            "substage":"CAPTURE" if progress >= .8 else ("TRANSFER" if self.stage_id in ("P01","P04","P08","P10","P11") else "EXECUTION"),
            "stall_diagnostic":stalled,"transition_evidence":list(self.transition_evidence),"physical_evaluator":evaluation}
        if rr_recovery is not None:
            self._snapshot["rr_capture_continuation"] = rr_recovery
        if (self.spec.get("physical_acceptance_version") == "all_stage_v1" and self.stage_id == "P13"
                and evaluation.get("post_completion_observation_started", False)):
            self._snapshot["substage"] = "CAPTURE"
        if self.spec.get("potential_definition") == "global_physical_progress_v3":
            self._snapshot["task_progress_potential"] = self.physical_potential(evaluation)
            # Physical airborne/load-transfer evidence overlaps phase labels.
            # Smooth attitude costs use this continuous coefficient, not a label edge.
            # A future unloaded leg must not discount the current task's costs.
            # Eligibility follows measured placement history, never phase labels
            # or a prescribed support-leg posture/contact template.
            unfinished=[v for leg,v in evaluation.get("current_legs",{}).items()
                if not histories["placed"][leg]
                and placement_predecessors_satisfied(self.spec, histories, leg)]
            self._snapshot["physical_transfer_fraction"] = max((
                _clip(1.-v["load_fraction"]/.35) for v in unfinished),default=0.)
        if self.spec.get("transfer_roles"):
            roles = evaluation.get("transfer_roles", {})
            active = roles.get(stage["active_leg"], {})
            self._snapshot.update(transfer_roles_version=TRANSFER_ROLES_MODE, transfer_roles=roles,
                                  transfer_role_context=active, pending_capture=bool(active.get("pending_capture")))
            eligible = [leg for leg in LEG_ORDER if not histories["placed"][leg]
                        and placement_predecessors_satisfied(self.spec, histories, leg)]
            self._snapshot["physical_transfer_fraction"] = max((roles.get(leg, {}).get("motion_fraction", 0.)
                                                                for leg in eligible), default=0.)
        if self.spec.get("p09_lift_semantics") in FUNCTIONAL_RR_MODES and evaluation["valid"]:
            rr = evaluation["current_legs"]["RR"]
            # Reuse the existing RR qualification bit for CURRENT validity.
            # Complete Q/C/P event history remains independently in history.
            # No extra hidden action-deciding timer or actor dimension is added.
            self._snapshot["active_lift_history"] = {
                **histories["active_lift"], "RR": bool(rr.get("current_lift_valid"))}
            self._snapshot["p09_lift_semantics"] = self.spec["p09_lift_semantics"]
            self._snapshot["rr_placed_currently_usable"] = _current_rr_placement_usable(evaluation)
        if continuation is not None:
            cross_tick = histories.get("event_ticks", {}).get("front_edge_crossed", {}).get("FL")
            pending_elapsed = (max(0., now-cross_tick/self.spec["physics_hz"])
                if continuation["fl_capture_pending"] and type(cross_tick) is int else 0.)
            continuation.update(scheduler_advanced_pending=self._fl_pending_handoff is not None,
                submode="P06_CAPTURE_PENDING" if self.stage_id == "P06" and continuation["fl_capture_pending"] else "FL_CAPTURE_PENDING" if continuation["fl_capture_pending"] else "NO_PENDING_FL_CAPTURE",
                pending_handoff=self._fl_pending_handoff,
                strict_historical_predecessors={leg: all(histories["placed"][p]
                    for p in PLACEMENT_PREDECESSORS[leg]) for leg in LEG_ORDER},
                recovery_action=("P06_source_wheel_overlap" if self.stage_id == "P06" and continuation["allow_capture_continuation"]
                    else "continue_current_physical_tasks_and_FL_recovery" if self.stage_id != "P05"
                    else "hip_only_capture_recovery_before_local_escalation" if age < local_limit
                    else "local_warning_waiting_current_safe_capture_path"))
            self._snapshot.update(capture_continuation=continuation,
                fl_contact_observed=continuation["fl_contact_observed"],
                fl_capture_pending=continuation["fl_capture_pending"],
                allow_capture_continuation=continuation["allow_capture_continuation"],
                p05_local_deadline_warning=self._p05_local_deadline_warning,
                capture_pending_elapsed_s=pending_elapsed,
                pending_capture=bool(self._snapshot.get("pending_capture") or continuation["fl_capture_pending"]))
            self._snapshot["local_timeout"].update(
                classification="scheduler_warning_not_episode_end" if local_warning_only else "finite_task_terminal_not_external_truncation",
                local_episode_terminal_enabled=not local_warning_only,
                timer_inputs_observable_in_existing372=False,
                observation_semantics="legacy372_times_plus_explicit_capture_append17",
                current_schema_observable=True)
        self._last_observation_tick = _get(observation, "physics_tick")
        return dict(self._snapshot)

    @property
    def snapshot(self) -> dict[str, Any]: return dict(self._snapshot)


class NominalMotionProvider:
    """Read-only compact actions as finite suggestions, without entry vetoes.

    The existing motion executor owns source waypoint/tracking scheduling only.
    A persistent nominal approaches these advisory targets under physical slew
    limits; it is never reset to a historical entry anchor. The first handoff
    sample retains both nominal and tracking. Adapter compensation is never
    reset. After the suggestion tail, PPO remains active under task deadlines.
    P02 may reuse P01's rolling suggestion from current measured task goals;
    this does not replay P01 or make source duration an entry/finish condition.
    """
    @staticmethod
    def _validated_servo_rate_overrides(spec: Mapping[str, Any]) -> dict[str, tuple[float, ...]]:
        """Optional advisory-only slew; never a physical limit or task gate."""
        nominal = spec["nominal"]
        overrides = nominal.get("phase_servo_rate_overrides_deg_s", {})
        if not isinstance(overrides, Mapping):
            raise ValueError("nominal servo rate overrides require a phase mapping")
        result = {}
        for phase, channels in overrides.items():
            if phase not in PHASE_IDS or not isinstance(channels, Mapping):
                raise ValueError("nominal servo rate override has an unknown phase or invalid channels")
            default = _number(nominal["servo_handoff_rate_deg_s"], "default nominal servo rate")
            rates = [default] * len(SERVO_ORDER)
            for name, value in channels.items():
                if name not in SERVO_ORDER or isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("nominal servo rate override requires a servo name and numeric rate")
                rate = _number(value, "nominal servo rate override")
                if not 0. < rate <= default:
                    raise ValueError("nominal servo rate override must be positive and no faster than the default")
                rates[SERVO_ORDER.index(name)] = rate
            result[phase] = tuple(rates)
        return result

    def __init__(self, contract: Any, *, spec: Mapping[str, Any] | None = None,
                 fsm_spec: Any = None):
        self.contract=contract; self.spec=dict(spec) if spec is not None else load_task_spec()
        self.physics_hz=float(contract.physics_hz)
        self.servo_rate_limit_deg_s=float(contract.servo_rate_limit_deg_s)
        self.nominal_full12=tuple(contract.phases[0].start_full12)
        self.state_id: str | None=None; self.elapsed_s=0.; self.endpoint_issued=False
        self.tracking_servo_names: tuple[str,...]=()
        nominal=self.spec["nominal"]
        self._capture_continuation = _capture_continuation_enabled(self.spec)
        self._capture_continuation_diagnostic: dict[str, Any] = {}
        self._p05_preedge_recovery = _p05_preedge_recovery_enabled(self.spec)
        self._p05_preedge_diagnostic: dict[str, Any] = {}
        reference_mode = self.spec.get("reference_nominal_semantics")
        if reference_mode not in (None, "successful_fsm_derived_v2"):
            raise ValueError("unknown successful-FSM nominal semantics")
        self._reference_nominal = reference_mode is not None
        self._sequence_mode = nominal.get("sequence_semantics")
        from .semantic_rear_policy_timing import MODES as REAR_TIMING_MODES
        self._rear_policy_timing_mode = nominal.get("rear_policy_timing")
        if self._rear_policy_timing_mode not in (None, *REAR_TIMING_MODES):
            raise ValueError("unknown rear policy timing mode")
        if self._rear_policy_timing_mode and self._sequence_mode != "source_partial_order_physical_ready_v1":
            raise ValueError("rear policy timing requires existing continuous physical source order")
        self._final_stop_mode = nominal.get("final_stop_owner")
        if self._final_stop_mode not in (None, "current_physical_stop_nominal_owner_v1",
                                         "source_home_after_physical_stop_v1",
                                         "source_home_after_physical_stop_v2"):
            raise ValueError("unknown nominal final stop ownership")
        if self._final_stop_mode and (not self._reference_nominal
                or nominal.get("continuous_channel_inheritance") is not True
                or self.spec.get("physical_acceptance_version") != "all_stage_v1"):
            raise ValueError("final stop ownership requires continuous source-derived all-stage nominal")
        self._final_stop_owner = None
        self._final_stop_owner_retired = False
        self._final_stop_diagnostic = {}
        self._final_home_recovery = None
        from .semantic_height_recovery import validate_height_candidate
        self._height_candidate = validate_height_candidate(nominal.get("height_recovery"))
        self._height_recovery_offsets = {"front_left_hip": 0., "rear_left_hip": 0.}
        self._height_diagnostic = {}
        if self._sequence_mode not in (None, "source_partial_order_physical_ready_v1"):
            raise ValueError("unknown nominal sequence semantics")
        self._rr_carry_source_mode = _validate_rr_carry_source_semantics(self.spec)
        self._rr_carry_knee_source_times = ()
        self._rr_carry_roll_source_time = None
        if self._rr_carry_source_mode:
            phase = contract.phase("P09")
            rolling = [w for w in phase.waypoints if set(w.atomic_channels) == set(WHEEL_ORDER)
                       and all(value > 0. for value in w.full12[8:])
                       and any(g.time_s == w.time_s and set(g.channels) == set(WHEEL_ORDER)
                               for g in phase.atomic_groups)]
            if not rolling:
                raise ValueError("RR carry source requires an authored positive four-wheel atomic group")
            self._rr_carry_roll_source_time = min(w.time_s for w in rolling)
            previous = phase.start_full12
            knee_times = []
            for waypoint in phase.waypoints:
                # Identify this source's pending carry events, not a physical
                # descent from a knee angle and not fixed historical timestamps.
                if (waypoint.time_s < self._rr_carry_roll_source_time
                        and set(waypoint.changed_channels) == {"rear_right_knee"}
                        and set(waypoint.atomic_channels) == {"rear_right_knee"}
                        and waypoint.full12[7] < previous[7]):
                    knee_times.append(waypoint.time_s)
                previous = waypoint.full12
            if not knee_times:
                raise ValueError("RR carry source requires identifiable authored knee carry events")
            self._rr_carry_knee_source_times = tuple(knee_times)
        self._p05_pending_capture_mode = nominal.get("p05_pending_capture")
        if self._p05_pending_capture_mode not in (None,
                "current_FL_capture_wheel_continuation_v1",
                "current_FL_capture_wheel_continuation_to_handoff_v2"):
            raise ValueError("unknown P05 capture-wheel continuation semantics")
        if (self._p05_pending_capture_mode == "current_FL_capture_wheel_continuation_to_handoff_v2"
                and (not self._reference_nominal
                     or nominal.get("continuous_channel_inheritance") is not True
                     or self.spec.get("physical_acceptance_version") != "all_stage_v1")):
            raise ValueError("P05 capture handoff requires continuous source-derived physical ownership")
        if self._sequence_mode and (not self._reference_nominal
                or nominal.get("continuous_channel_inheritance") is not True):
            raise ValueError("physical source order requires continuous source-derived nominal")
        self._p09_late_source = None
        if self._sequence_mode and self.spec.get("p09_lift_semantics") in FUNCTIONAL_RR_MODES:
            phase = contract.phase("P09")
            groups = [g for g in phase.atomic_groups if g.source_full12_atomic]
            if len(groups) != 1:
                raise ValueError("P09 late reconfiguration requires one authored full12 atomic group")
            group = groups[0]
            index = next(i for i, w in enumerate(phase.waypoints)
                         if math.isclose(w.time_s, group.time_s, rel_tol=0., abs_tol=1e-9))
            waypoint, before = phase.waypoints[index], phase.waypoints[index-1]
            changed = {"front_left_hip", "front_left_knee", "rear_left_hip",
                       "rear_left_knee", "front_left_ankle"}
            if (index == 0 or set(waypoint.changed_channels) != changed
                    or set(group.required_runtime_channels) != set(SERVO_ORDER+WHEEL_ORDER)
                    or set(before.atomic_channels) != set(WHEEL_ORDER)
                    or before.full12[8:] != (0.,)*4
                    or not any(g.time_s == before.time_s and set(g.channels) == set(WHEEL_ORDER)
                               for g in phase.atomic_groups)):
                raise ValueError("P09 late reconfiguration must follow the authored four-wheel stop")
            self._p09_late_source = (before.time_s, group.time_s)
        self._reference_fsm_spec = None
        self.normal_drive_bias_full12 = ZERO12
        if self._reference_nominal:
            # Read the same immutable source specification used by the frozen
            # controller. Its correction fractions describe nominal tuning,
            # never residual bounds, phase gates or task acceptance.
            self._reference_fsm_spec = (fsm_spec if fsm_spec is not None else
                load_fsm_spec(Path(__file__).resolve().parents[3]/"configs/fsm_states.yaml"))
            if (tuple(state.state_id for state in self._reference_fsm_spec.states) != PHASE_IDS
                    or self._reference_fsm_spec.rear_leg_order != self.spec["rear_leg_order"]
                    or self._reference_fsm_spec.motion_hz != self.physics_hz):
                raise ValueError("successful-FSM nominal source specification differs")
        self._servo_rate_overrides = self._validated_servo_rate_overrides(self.spec)
        self._all_stage_acceptance = self.spec.get("physical_acceptance_version") == "all_stage_v1"
        if self._all_stage_acceptance and nominal.get("continuous_channel_inheritance") is not True:
            raise ValueError("all-stage nominal ownership requires continuous channel inheritance")
        self._observed_captured_legs: set[str] = set()
        self._capture_owner_holds: dict[str, dict[str, Any]] = {}
        self._approach_wheel_prior=_vector(nominal["approach_wheel_prior_rad_s"],4,"approach wheel prior rad/s")
        source_wheels={tuple(waypoint.full12[8:]) for waypoint in contract.phase("P01").waypoints
                       if any(value != 0. for value in waypoint.full12[8:])}
        if (nominal.get("approach_wheel_prior_source")!="P01"
                or nominal.get("approach_wheel_prior_scope")!="P02_current_physical_goal_feedback_only"
                or any(value <= 0. for value in self._approach_wheel_prior)
                or source_wheels!={self._approach_wheel_prior}):
            raise ValueError("approach wheel prior must equal the verified P01 rolling waypoint in rad/s")
        self._source_motion=MotionExecutor(physics_hz=self.physics_hz,
            servo_rate_limit_deg_s=self.servo_rate_limit_deg_s,initial_full12=self.nominal_full12)
        self._continuous_layers: list[dict[str,Any]] = []
        self._retirement_bounds=_p06_retirement_bounds(self.spec)
        self._p06_tail_source = (_verified_p06_rolling_source(contract, self._approach_wheel_prior)
                                if _p06_tail_enabled(self.spec) else None)
        self._tail_diagnostic: dict[str, Any] = {
            "schema": "wlr50_clean.p06_wheel_tail.v1", "enabled": self._p06_tail_source is not None,
            "timing": "P06_layer_contribution_before_later_owners_and_slew_not_applied_target",
            "source_rolling_rad_s": self._p06_tail_source, "layer_present": False,
            "source_endpoint_issued": False, "finite_source_tail_replaced": False,
            "wheel_gain": None, "status": "not_applicable_no_P06_layer"}
        self._retirement_diagnostic: dict[str,Any] = {
            "schema":"wlr50_clean.p06_rolling_retirement.v1",
            "timing":"current_nominal_suggestion_before_slew_not_applied_target",
            "enabled":self._retirement_bounds is not None,"layer_present":False,
            "status":"not_applicable_no_P06_layer","origin":None,
            "source_observation_tick":None,"source_sim_time_s":None,
            "front_distance_rl_m":None,"front_distance_rr_m":None,
            "lateral_valid":None,"measured_fraction":None,"peak_fraction":None,"wheel_gain":None}

    @property
    def nominal_suggestion_diagnostics(self) -> dict[str,Any]:
        result = ({"p06_rolling_retirement":dict(self._retirement_diagnostic)}
                  if self._retirement_bounds is not None else {})
        if self._reference_nominal:
            result["successful_fsm_nominal"] = {
                "mode":"successful_fsm_derived_v2",
                "source_state_spec_path":str(self._reference_fsm_spec.path.resolve()),
                "request_semantics":"held_source_owner_requests_single_mature_mapper",
                "tracking_semantics":"current_and_unfinished_owners_not_stale_handoff_tracking",
                "normal_drive_bias_full12":self.normal_drive_bias_full12,
                "dormant_reference_rebound_trigger_enabled":False,
                "source_entry_and_completion_gates_enabled":False,
                "residual_dependent_controller_selection":False,
            }
        if self._sequence_mode:
            result["source_partial_order"] = {
                "mode": self._sequence_mode, "default_assist": "RL_source_method",
                "RR_contact_backup_enabled": False, "residual_channels_restricted": False,
                "layers": [{"stage": layer["stage"], "source_ticks": layer["ticks"],
                    **layer.get("sequence_diagnostic", {})} for layer in self._continuous_layers
                    if layer["stage"] in (("P07", "P08", "P09", "P12")
                                          if self._rear_policy_timing_mode else ("P07", "P08", "P09"))],
            }
        if self._p06_tail_source is not None:
            result["p06_wheel_tail"] = dict(self._tail_diagnostic)
        if self._final_stop_mode:
            result["final_stop_owner"] = dict(self._final_stop_diagnostic)
        if hasattr(self, "_rr_carry_diagnostic"):
            result["rr_carry_continuation"] = dict(self._rr_carry_diagnostic)
        if self._height_candidate is not None:
            result["height_recovery"] = dict(self._height_diagnostic)
        if self._capture_continuation:
            result["capture_continuation"] = dict(self._capture_continuation_diagnostic)
        if self._p05_preedge_recovery:
            result["p05_preedge_approach_recovery"] = dict(self._p05_preedge_diagnostic)
        if self._all_stage_acceptance:
            result["capture_owner_hold"] = {
                "schema": "wlr50_clean.nominal_capture_owner_hold.v1",
                "enabled": True,
                "semantics": "retire_captured_owners_skip_redundant_swing_allow_later_cooperation",
                "held_value_semantics": "current_nominal_not_actual_q_or_final_drive",
                "captures": {leg: dict(record) for leg, record in self._capture_owner_holds.items()},
                "suppressed_new_swing_layers": tuple(dict(layer["capture_suppressed_at_creation"])
                    for layer in self._continuous_layers if "capture_suppressed_at_creation" in layer),
            }
        return result

    def rear_policy_timing(self, task: Mapping[str, Any]) -> dict[str, Any]:
        from .semantic_rear_policy_timing import public_timing
        return public_timing(task, self._continuous_layers, self.spec["support"], self.physics_hz,
                             mode=self._rear_policy_timing_mode)

    def _start_source_motion(self, motion: MotionExecutor, phase: Any) -> None:
        """Reuse successful source action tuning without importing its gates."""
        if not self._reference_nominal:
            motion.start_phase(phase)
            return
        from wlr50_clean.fsm.motion_executor import FeedbackCorrection
        state = self._reference_fsm_spec.state(phase.state_id)
        correction = FeedbackCorrection(state.normal_correction_fractions
            if state.normal_correction_domain == "logical_command" else ZERO12)
        motion.start_phase(phase, correction, time_scale=state.normal_time_scale)

    def _source_normal_bias(self, motion: MotionExecutor, sample: Any) -> tuple[float, ...]:
        """Same source formula, including its endpoint tick but not held tail.

        This is a finite nominal actuator suggestion. Unlike the legacy
        controller it cannot require a historical entry angle or rebound rate.
        """
        if not self._reference_nominal:
            return ZERO12
        state = self._reference_fsm_spec.state(sample.state_id)
        if (state.normal_correction_domain != "post_mapper_drive"
                or sample.elapsed_s > motion.effective_active_duration_s + 1e-12):
            return ZERO12
        return tuple(.5*delta*fraction for delta,fraction in zip(
            motion.phase.delta_full12,state.normal_correction_fractions,strict=True))

    def _capture_hold_measurement(self, task: Mapping[str, Any], observation: Any) -> dict[str, Any] | None:
        """Validate live placement history before source clocks or owner mutation."""
        if not self._all_stage_acceptance:
            return None
        ev = task.get("physical_evaluator")
        if task.get("termination_reason") is not None or (
                isinstance(ev, Mapping) and ev.get("termination_reason") is not None):
            return None
        if not isinstance(ev, Mapping) or ev.get("valid") is not True:
            raise SemanticObservationError("capture ownership requires a valid current evaluator")
        history = ev.get("history")
        placed = history.get("placed") if isinstance(history, Mapping) else None
        if not isinstance(placed, Mapping) or any(type(placed.get(leg)) is not bool for leg in LEG_ORDER):
            raise SemanticObservationError("capture ownership requires four real placement history flags")
        tick = ev.get("physics_tick")
        now = _number(ev.get("simulation_time_s"), "capture ownership source time")
        if (type(tick) is not int or tick < 0
                or not math.isclose(now, tick/self.physics_hz, rel_tol=0., abs_tol=1e-9)):
            raise SemanticObservationError("capture ownership source clock is invalid")
        if observation is not None and (tick != _get(observation, "physics_tick") or not math.isclose(
                now, _number(_get(observation, "simulation_time_s"), "observation time"), rel_tol=0., abs_tol=1e-9)):
            raise SemanticObservationError("capture ownership source differs from current observation")
        new_swing_hold = None
        stage_id = str(task["stage_id"])
        target = {"P02": "FR", "P05": "FL", "P09": "RR", "P12": "RL"}.get(stage_id)
        new_layer = not self._continuous_layers or self._continuous_layers[-1]["stage"] != stage_id
        if new_layer and target is not None and placed[target]:
            legs = ev.get("current_legs")
            current = legs.get(target) if isinstance(legs, Mapping) else None
            flags = ("within_top_xy", "ground_contact", "top_contact", "air")
            if not isinstance(current, Mapping) or any(type(current.get(key)) is not bool for key in flags):
                raise SemanticObservationError("new captured swing owner requires current region/contact flags")
            clearance = _number(current.get("clearance_m"), "captured swing current clearance")
            if (current["within_top_xy"] and not current["ground_contact"]
                    and (current["top_contact"] or (current["air"] and clearance >= 0.))):
                new_swing_hold = {"stage_id": stage_id, "leg": target, "source_physics_tick": tick,
                    **{key: current[key] for key in flags}, "clearance_m": clearance}
        return {"physics_tick": tick, "placed": {leg: placed[leg] for leg in LEG_ORDER},
                "new_swing_hold": new_swing_hold}

    def _retirement_measurement(self, task: Mapping[str,Any], observation: Any) -> dict[str,Any] | None:
        if (self._retirement_bounds is None or (task["stage_id"]!="P06"
                and not any(layer["stage"]=="P06" for layer in self._continuous_layers))):
            return None
        ev=task.get("physical_evaluator")
        # Preserve the existing terminal path (including nonfinite failures).
        # Terminal samples cannot earn retirement or masquerade as live inputs.
        if task.get("termination_reason") is not None or (isinstance(ev,Mapping) and ev.get("termination_reason") is not None):
            return {"status":"terminal_no_new_retirement","source_observation_tick":None,
                    "source_sim_time_s":None,"measured_fraction":None,
                    "front_distance_rl_m":None,"front_distance_rr_m":None,"lateral_valid":None}
        if not isinstance(ev,Mapping) or ev.get("valid") is not True:
            raise SemanticObservationError("P06 retirement requires a valid live evaluator")
        tick=ev.get("physics_tick"); now=_number(ev.get("simulation_time_s"),"retirement source time")
        if type(tick) is not int or tick<0 or not math.isclose(now,tick/self.physics_hz,rel_tol=0.,abs_tol=1e-9):
            raise SemanticObservationError("P06 retirement source clock is invalid")
        if observation is not None and (tick!=_get(observation,"physics_tick") or not math.isclose(
                now,_number(_get(observation,"simulation_time_s"),"observation time"),rel_tol=0.,abs_tol=1e-9)):
            raise SemanticObservationError("P06 retirement source differs from current observation")
        legs=ev.get("current_legs")
        if not isinstance(legs,Mapping) or any(not isinstance(legs.get(leg),Mapping) for leg in ("RL","RR")):
            raise SemanticObservationError("P06 retirement lacks measured rear geometry")
        distances=tuple(_number(legs[leg].get("front_distance_m"),f"{leg} front distance") for leg in ("RL","RR"))
        lateral=tuple(legs[leg].get("within_lateral_span") for leg in ("RL","RR"))
        if any(type(value) is not bool for value in lateral):
            raise SemanticObservationError("P06 retirement requires measured lateral flags")
        lower,width=self._retirement_bounds
        fraction=_clip((min(distances)-lower)/width) if all(lateral) else 0.
        return {"status":"live_measured","source_observation_tick":tick,"source_sim_time_s":now,
                "front_distance_rl_m":distances[0],"front_distance_rr_m":distances[1],
                "lateral_valid":all(lateral),"measured_fraction":fraction}

    @classmethod
    def from_handoff(cls, contract: Any, *, spec: Mapping[str,Any], stage_id: str,
                     nominal_full12: Sequence[float], tracking_servo_names: Sequence[str],
                     fsm_spec: Any = None):
        command=_vector(tuple(nominal_full12),12,"executed handoff nominal")
        if Full12Command.from_full12(command).clamped().to_full12()!=command:
            raise SemanticObservationError("handoff nominal is outside actuator limits")
        names=tuple(tracking_servo_names)
        if len(names)!=len(set(names)) or any(name not in SERVO_ORDER for name in names):
            raise SemanticObservationError("invalid executed handoff tracking")
        result=cls(contract,spec=spec,fsm_spec=fsm_spec)
        result.nominal_full12=command; result.tracking_servo_names=names; result.state_id=stage_id
        result._source_motion=MotionExecutor(physics_hz=result.physics_hz,
            servo_rate_limit_deg_s=result.servo_rate_limit_deg_s,initial_full12=command)
        result._start_source_motion(result._source_motion,contract.phase(stage_id))
        return result

    def _sequence_permission(self, layer: dict[str, Any], task: Mapping[str, Any],
                             observation: Any) -> bool:
        """Dispatch readiness, not phase completion or a historical pose gate.

        Retain the finite source cadence (including its atomic FR pulse/stop),
        but do not consume a pending owner's clock. Physical waits hold existing
        targets while the mapper, residual and simulation continue normally.
        """
        if self._capture_continuation and layer["stage"] == "P06":
            status = _capture_continuation_status(self.spec, task.get("physical_evaluator", {}), task["stage_id"])
            if status["fl_capture_pending"] and not status["allow_capture_continuation"]:
                layer["sequence_diagnostic"] = {"status": "holding", "wait_reason": "current_FL_capture_path_or_other_support_unavailable",
                    "observation_tick": task.get("physical_evaluator", {}).get("physics_tick")}
                return False  # Resume this clock, never catch up expired events.
        if self._rear_policy_timing_mode and layer["stage"] == "P12":
            from .semantic_rear_policy_timing import rear_dependency
            dep = rear_dependency(task, self.spec["support"])
            permitted = dep["support_transfer_permitted"] or dep["rl_current_swing"]
            layer["rl_dependency_wait"] = not permitted
            layer["sequence_diagnostic"] = dict(dep,
                observation_tick=task.get("physical_evaluator", {}).get("physics_tick"),
                status="active" if permitted else "holding_RL_joint_lane",
                wait_reason=None if permitted else "current_RR_support_before_new_RL_unload",
                wheel_source_clock_continues_after_start=layer["ticks"] > 0)
            # Synchronize the initial source knee/wheel sequence. Once started,
            # wheel pulses and explicit stops must complete even after drop-load.
            # Only the independent RL joint cursor pauses, without catch-up.
            return bool(layer["ticks"] > 0 or permitted)
        if not self._sequence_mode or layer["stage"] not in ("P07", "P08", "P09"):
            return True
        ev = task.get("physical_evaluator", {})
        legs = ev.get("current_legs", {})
        tick = ev.get("physics_tick")
        diag = layer.setdefault("sequence_diagnostic", {})
        diag.update(observation_tick=tick, status="active", wait_reason=None)
        if layer["stage"] == "P09" and self._rr_carry_source_mode:
            diag["current_free_lift_source_readiness"] = None
        if ev.get("valid") is not True or task.get("termination_reason") is not None:
            diag.update(status="holding", wait_reason="no_live_physical_readiness")
            return False
        joints = _get(observation, "joints", {}) if observation is not None else {}
        q = tuple(_number(_get(joints.get(name, {}), "position_deg"),
                          "sequence measured joint position") for name in SERVO_ORDER)
        fl, fr, rr = (legs.get(leg, {}) for leg in ("FL", "FR", "RR"))
        support_count = sum(row.get("support") is True and
                            (row.get("ground_contact") is True or row.get("top_contact") is True)
                            for row in legs.values())
        other_support_count = sum(row.get("support") is True and
            (row.get("ground_contact") is True or row.get("top_contact") is True)
            for leg, row in legs.items() if leg != "RR")
        enough_support = (support_count if layer["stage"] == "P07" else other_support_count
                          ) >= self.spec["support"]["minimum_other_supports"]
        base = _vector(_get(_get(observation, "base", {}), "position_w_m"), 3, "sequence body position")
        wheel = _vector(_get(_get(observation, "wheels", {}).get(WHEEL_ORDER[0], {}),
                            "center_w_m"), 3, "sequence FL position")
        fl_radius = math.dist(base, wheel)
        preparation = next((x for x in self._continuous_layers if x["stage"] == "P07"), layer)
        preparation.setdefault("FL_initial_body_radius_m", fl_radius)
        contraction = preparation["FL_initial_body_radius_m"] - fl_radius
        # Space is measured, not inferred from FL's command or an imagined load.
        fl_space = bool(fl.get("within_lateral_span") is True and
            ((fl.get("air") is True and not fl.get("ground_contact")
              and fl.get("clearance_m", -1.) >= self.spec["history"]["minimum_initial_clearance_gain_m"])
             or contraction >= self.spec["history"]["minimum_initial_clearance_gain_m"]))
        rr_continuing = bool(rr.get("current_lift_valid") is True
                             and rr.get("motion_continuation_allowed") is True
                             and not rr.get("ground_contact"))
        minimum_motion = self.spec["history"]["minimum_joint_motion_deg"]
        ready, reason = True, None
        if layer["stage"] == "P07":
            group = next(g for g in layer["motion"].phase.atomic_groups
                         if "front_right_knee" in g.channels and "front_right_ankle" in g.channels)
            group_tick = layer["motion"]._scaled_source_tick(group.time_s)
            if layer["ticks"] == group_tick and "fr_group_start_tick" not in diag:
                ready = enough_support and (fl_space or rr_continuing)
                reason = "FL_measured_space_before_FR_atomic_group"
                if ready:
                    layer["fr_start_q"] = q[3]
                    diag["fr_group_start_tick"] = tick
        elif layer["ticks"] == 0:
            predecessor_id = "P07" if layer["stage"] == "P08" else "P08"
            predecessor = next((x for x in self._continuous_layers
                                if x["stage"] == predecessor_id), None)
            # Executed suffix handoffs contain no reconstructed predecessor.
            # In that case use current physical readiness, never replay a prefix.
            ended = predecessor is None or bool(predecessor.get("sample")
                                                 and predecessor["sample"].endpoint_issued)
            if layer["stage"] == "P08":
                fr_response = predecessor is None or (
                    "fr_start_q" in predecessor and
                    predecessor["fr_start_q"] - q[3] >= minimum_motion)
                ready = enough_support and (rr_continuing or
                    (ended and fl_space and fr.get("support") is True and fr_response))
                reason = "P07_source_groups_and_measured_FL_space_FR_response"
            else:
                rl_response = predecessor is None or (
                    "start_q" in predecessor and q[4] - predecessor["start_q"][4] >= minimum_motion)
                unloaded = (rr.get("load_fraction_valid") is True and
                            rr.get("load_fraction", 1.) <= self.spec["support"]["unloaded_leg_maximum_load_fraction"])
                ready = enough_support and (rr_continuing or (ended and rl_response and unloaded))
                reason = "RL_source_assist_response_and_RR_measured_unload_or_lift"
        elif layer["stage"] == "P09" and self._p09_late_source is not None:
            group_tick = layer["motion"]._scaled_source_tick(self._p09_late_source[1])
            pending = self._rr_pending_carry_readiness(layer, task, observation) if self._rr_carry_source_mode else None
            if pending is not None:
                ready, reason = pending["ready"], pending["reason"]
                diag["current_free_lift_source_readiness"] = pending
            elif layer["ticks"] == group_tick:
                if self._rear_policy_timing_mode:
                    from .semantic_rear_policy_timing import rear_dependency
                    ready = rear_dependency(task, self.spec["support"])["support_transfer_permitted"]
                    reason = "current_RR_bearing_before_FL_RL_transfer"
                else:
                    ready = self._rr_late_reconfiguration_ready(task)
                    reason = "current_RR_over_top_before_late_reconfiguration"
                diag.update(late_group_source_tick=group_tick,
                    RR_late_front_distance_m=rr.get("front_distance_m"),
                    RR_late_clearance_m=rr.get("clearance_m"),
                    RR_late_within_top_xy=rr.get("within_top_xy"))
                if ready:
                    diag["late_group_start_tick"] = tick
        diag.update(support_count=support_count, RR_other_support_count=other_support_count, FL_space_measured=fl_space,
                    FL_body_radial_contraction_m=contraction, RR_current_continuation=rr_continuing)
        if not ready:
            diag.update(status="holding" if layer["ticks"] else "pending", wait_reason=reason)
            diag["wait_ticks"] = diag.get("wait_ticks", 0) + 1
        elif layer["ticks"] == 0:
            layer["start_q"] = q
            diag["actual_start_tick"] = tick
        return ready

    def _rr_pending_carry_readiness(self, layer, task, observation):
        """Only pending authored events wait; no replay, stop mask or clock catch-up."""
        local_tick = layer["ticks"]
        motion = layer["motion"]
        knee = any(local_tick == motion._scaled_source_tick(t) for t in self._rr_carry_knee_source_times)
        roll = local_tick == motion._scaled_source_tick(self._rr_carry_roll_source_time)
        if not (knee or roll):
            return None  # In particular, no explicit stop group is gated here.
        ev = task.get("physical_evaluator", {})
        legs = ev.get("current_legs", {})
        rr = legs.get("RR", {})
        observed_tick = _get(observation, "physics_tick")
        observed_time = _get(observation, "simulation_time_s")
        ev_tick, ev_time = ev.get("physics_tick"), ev.get("simulation_time_s")
        fresh = (type(ev_tick) is int and ev_tick == observed_tick
            and isinstance(ev_time, (float, int)) and isinstance(observed_time, (float, int))
            and math.isfinite(ev_time) and math.isfinite(observed_time)
            and math.isclose(ev_time, observed_time, rel_tol=0., abs_tol=1e-9)
            and math.isclose(ev_time, ev_tick/self.physics_hz, rel_tol=0., abs_tol=1e-9))
        support_count = sum(self._rr_verified_bearing(row) for leg, row in legs.items() if leg != "RR")
        valid = bool(fresh and ev.get("valid") is True and task.get("termination_reason") is None
            and ev.get("termination_reason") is None
            and ev.get("physical_evidence_status") in ("VERIFIED", "CONTACT_BEARING_UNVERIFIED")
            and rr.get("current_lift_valid") is True and rr.get("motion_continuation_allowed") is True
            and rr.get("ground_contact") is False and rr.get("within_lateral_span") is True
            and support_count >= self.spec["support"]["minimum_other_supports"])
        try:
            gap = _number(rr.get("clearance_m"), "RR pending carry clearance")
        except (SemanticObservationError, TypeError):
            gap = None
        vz = rr.get("wheel_bottom_vz_m_s")
        try:
            vz = _number(vz, "RR measured collider-bottom vertical velocity")
            if rr.get("wheel_bottom_vz_observation_tick") != ev.get("physics_tick"):
                vz = None
        except (SemanticObservationError, TypeError):
            vz = None
        reason = "current_free_lift_before_pending_knee" if knee else "current_free_lift_before_pending_roll"
        if knee:
            # XY alone includes under-platform/edge states. Only the existing
            # current qualified drop-region test permits intentional lowering.
            safe_drop = valid and self._rr_late_reconfiguration_ready(task)
            ready = valid and (rr.get("within_top_xy") is not True or safe_drop)
            if ready and not safe_drop and rr.get("air") is True:
                if gap is None or (gap < self.spec["geometry"]["airborne_clearance_above_top_m"]
                                    and (vz is None or vz < 0.)):
                    ready = False
                    reason = ("current_AIR_clearance_unavailable" if gap is None else
                              "current_AIR_low_clearance_vertical_response_unavailable" if vz is None else
                              "current_AIR_clearance_decreasing_before_pending_knee")
        else:
            # A finite-radius TOP corner can support continued forwarding before
            # center crossing. Unlike the post-source helper this has no endpoint
            # prerequisite, and AIR has no above-top or stationary-pose gate.
            top = (gap is not None and rr.get("contact_surface") == "TOP"
                and rr.get("top_surface_contact") is True and rr.get("obstacle_pair_active") is True
                and self._rr_verified_bearing(rr)
                and self.spec["geometry"]["top_gap_min_m"] <= gap <= self.spec["geometry"]["top_gap_max_m"])
            ready = valid and (rr.get("air") is True or top)
        return dict(mode=self._rr_carry_source_mode, source_tick=local_tick,
            observation_tick=ev.get("physics_tick"), pending_kind="knee_waypoint" if knee else "positive_fourwheel_group",
            ready=bool(ready), reason=reason, current_free_lift_valid=valid, current_evidence_fresh=bool(fresh),
            RR_clearance_m=gap, RR_bottom_vz_m_s=vz, RR_other_verified_supports=support_count,
            source_clock_may_advance=bool(ready), physical_success_awarded=False)

    def _rr_verified_bearing(self, row: Any) -> bool:
        if not isinstance(row, Mapping):
            return False
        try:
            force = _number(row.get("bearing_force_n"), "RR continuation bearing")
        except (SemanticObservationError, TypeError):
            return False
        return (row.get("support") is True and row.get("bearing_verified") is True
                and row.get("air") is False and force >= self.spec["support"]["force_noise_floor_n"]
                and (row.get("ground_contact") is True or row.get("top_surface_contact") is True))

    def _rr_late_reconfiguration_ready(self, task: Mapping[str, Any]) -> bool:
        """Current safe drop region, not capture/history-pose or initial-lift height.

        AIR above the platform may be lowered by this very atomic group. TOP
        uses the existing contact gap tolerance; a front-corner touch alone is
        not an over-platform entry. No task event or support is manufactured.
        """
        ev = task.get("physical_evaluator", {})
        legs = ev.get("current_legs", {})
        rr = legs.get("RR", {})
        try:
            distance = _number(rr.get("front_distance_m"), "RR late front distance")
            clearance = _number(rr.get("clearance_m"), "RR late clearance")
        except (SemanticObservationError, TypeError):
            return False
        geo = self.spec["geometry"]
        legal_gap = ((rr.get("air") is True and clearance >= 0.) or
            (rr.get("contact_surface") == "TOP" and rr.get("top_surface_contact") is True
             and rr.get("top_geometry") is True and rr.get("obstacle_pair_active") is True
             and self._rr_verified_bearing(rr)
             and geo["top_gap_min_m"] <= clearance <= geo["top_gap_max_m"]))
        return bool(ev.get("valid") is True and task.get("termination_reason") is None
            and ev.get("termination_reason") is None
            and ev.get("physical_evidence_status") in ("VERIFIED", "CONTACT_BEARING_UNVERIFIED")
            and rr.get("current_lift_valid") is True and rr.get("motion_continuation_allowed") is True
            and rr.get("ground_contact") is False and rr.get("within_lateral_span") is True
            and rr.get("within_top_xy") is True and distance >= 0. and legal_gap
            and sum(self._rr_verified_bearing(legs.get(leg, {})) for leg in LEG_ORDER if leg != "RR")
                >= self.spec["support"]["minimum_other_supports"])

    def _rr_waiting_late_group(self, layer: Mapping[str, Any]) -> bool:
        """Only this pending group relinquishes wheel ownership, not fresh stops."""
        if self._p09_late_source is None or layer["stage"] != "P09" or layer.get("sample") is None:
            return False
        stop, group = (layer["motion"]._scaled_source_tick(t) for t in self._p09_late_source)
        return bool(layer["ticks"] == group and layer["sample"].tick_index > stop
            and layer.get("advanced_this_tick") is False
            and layer.get("sequence_diagnostic", {}).get("wait_reason")
                in ("current_RR_over_top_before_late_reconfiguration",
                    "current_RR_bearing_before_FL_RL_transfer"))

    def _rr_top_continuation_allowed(self, task: Mapping[str, Any]) -> bool:
        """Existing wheel suggestion may create crossing after a real TOP contact.

        TOP surface sensing is independent of center-plane crossing: requiring
        top_contact/top_geometry/within_top_xy here would wait for the very
        forward motion being suggested. No Q/C/P event is changed or awarded.
        """
        ev = task.get("physical_evaluator", {})
        if (task.get("stage_id") != "P09" or task.get("termination_reason") is not None
                or not isinstance(ev, Mapping)
                or ev.get("valid") is not True or ev.get("termination_reason") is not None
                or ev.get("physical_evidence_status") not in ("VERIFIED", "CONTACT_BEARING_UNVERIFIED")
                or not any(layer["stage"] == "P09" and layer.get("sample") is not None
                           and (layer["sample"].endpoint_issued or self._rr_waiting_late_group(layer))
                           for layer in self._continuous_layers)):
            return False
        legs, history = ev.get("current_legs", {}), ev.get("history", {})
        if not isinstance(legs, Mapping) or not isinstance(history, Mapping):
            return False
        rr = legs.get("RR", {})
        geo, support = self.spec["geometry"], self.spec["support"]

        bearing = self._rr_verified_bearing

        try:
            distance = _number(rr.get("front_distance_m"), "RR continuation front distance")
            clearance = _number(rr.get("clearance_m"), "RR continuation current gap")
        except (SemanticObservationError, TypeError):
            return False
        return bool(history.get("active_lift", {}).get("RR") is True
            and history.get("placed", {}).get("RR") is False and rr.get("active_attempt") is True
            and rr.get("current_lift_valid") is True and rr.get("motion_continuation_allowed") is True
            and rr.get("ground_contact") is False and rr.get("within_lateral_span") is True
            and rr.get("contact_surface") == "TOP" and rr.get("top_surface_contact") is True
            and rr.get("obstacle_pair_active") is True and bearing(rr)
            and geo["top_gap_min_m"] <= clearance <= geo["top_gap_max_m"]
            and geo["workspace_min_m"] <= distance < geo["approach_max_m"]
            and sum(bearing(legs.get(leg, {})) for leg in LEG_ORDER if leg != "RR")
                >= support["minimum_other_supports"])

    def _continuous_advisory(self, task: Mapping[str,Any], retirement: Mapping[str,Any] | None=None,
                            capture: Mapping[str,Any] | None=None,
                            observation: Any=None) -> tuple[tuple[float,...],tuple[str,...]]:
        """Continue unfinished predecessor suggestions; never restore an entry vector.

        Each layer owns only channels it actually changes, and newer changes
        take precedence. This is nominal scheduling, not a success gate or a
        policy trajectory: all twelve residual channels stay available.
        """
        stage_id=str(task["stage_id"])
        if capture is not None:
            # Stop only owners that existed before this newly observed capture.
            # The new phase may deliberately reopen the same leg for another
            # task. Never pin a leg forever or restore a historical entry pose.
            for leg in LEG_ORDER:
                if not capture["placed"][leg] or leg in self._observed_captured_legs:
                    continue
                indices = (2*LEG_ORDER.index(leg), 2*LEG_ORDER.index(leg)+1)
                for layer in self._continuous_layers:
                    layer.setdefault("capture_retired_servo_indices", set()).update(indices)
                self._capture_owner_holds[leg] = {
                    "observed_capture_tick": capture["physics_tick"],
                    "stage_id": stage_id,
                    "held_nominal_servo_deg": tuple(self.nominal_full12[i] for i in indices),
                    "retired_preexisting_layers": tuple(layer["stage"] for layer in self._continuous_layers),
                }
                self._observed_captured_legs.add(leg)
        if not self._continuous_layers or self._continuous_layers[-1]["stage"]!=stage_id:
            phase=self.contract.phase(stage_id)
            motion=MotionExecutor(physics_hz=self.physics_hz,servo_rate_limit_deg_s=self.servo_rate_limit_deg_s,
                                  initial_full12=self.nominal_full12)
            self._start_source_motion(motion,phase)
            self._continuous_layers.append({"stage":stage_id,"motion":motion,"last":tuple(phase.start_full12),
                "touched":set(),"sample":None,"ticks":0})
            if self._rear_policy_timing_mode and stage_id == "P12":
                # P12 contains wheel-only atomic groups; no mixed joint/wheel
                # atomic group is split. Declare the independent RL pair cursor
                # so loss of support cannot indefinitely retain a wheel pulse.
                if any(set(g.channels).intersection(SERVO_ORDER[4:6])
                       and set(g.channels).intersection(WHEEL_ORDER) for g in phase.atomic_groups):
                    raise ValueError("rear lane cannot split a mixed RL/wheel atomic source group")
                rl_motion = MotionExecutor(physics_hz=self.physics_hz,
                    servo_rate_limit_deg_s=self.servo_rate_limit_deg_s, initial_full12=self.nominal_full12)
                self._start_source_motion(rl_motion, phase)
                self._continuous_layers[-1].update(rl_motion=rl_motion, rl_ticks=0,
                    rl_sample=None, rl_last=tuple(phase.start_full12), rl_touched=set(), rl_dependency_wait=False)
            if capture is not None and capture["new_swing_hold"] is not None:
                # A currently captured target needs no redundant new swing.
                # Suppress only this layer's target pair, not its collaborators
                # or later receiving-side/P13 owners. History alone is not enough.
                record = dict(capture["new_swing_hold"])
                first = 2*LEG_ORDER.index(record["leg"])
                indices = (first, first+1)
                record["held_nominal_servo_deg"] = tuple(self.nominal_full12[i] for i in indices)
                self._continuous_layers[-1]["capture_retired_servo_indices"] = set(indices)
                self._continuous_layers[-1]["capture_suppressed_at_creation"] = record
        proposed=list(self.nominal_full12)
        # Rebuild declared tracking owners in the source-derived mode. An
        # ended source segment must not acquire another feedback sample merely
        # because the semantic phase changed. Unfinished older layers below
        # still retain their owned targets and live tracking on the same tick.
        tracking=set() if self._reference_nominal else set(self.tracking_servo_names)
        normal_bias=list(ZERO12)
        ev=task.get("physical_evaluator",{}); legs=ev.get("current_legs",{}); history=ev.get("history",{})
        pending_status = (_capture_continuation_status(self.spec, ev, stage_id)
                          if self._capture_continuation else None)
        if pending_status is not None:
            self._capture_continuation_diagnostic = {**pending_status,
                "source_observation_tick": ev.get("physics_tick"), "P06_layer_present": False,
                "P06_wheel_contribution_enabled": False,
                "semantics": "current_measured_P06_owner_permission_before_later_source_owners_and_residual",
                "policy_residual_restricted": False, "source_clock_catch_up": False}
        if self._height_candidate is not None:
            from .semantic_height_recovery import current_rr_recovery_permission
            permitted = current_rr_recovery_permission(task, support_spec=self.spec["support"])
            step = self._height_candidate["recovery_rate_deg_s"]/self.physics_hz
            for name, previous_offset in self._height_recovery_offsets.items():
                goal = self._height_candidate["post_lift_recovery_deg"][name] if permitted else 0.
                self._height_recovery_offsets[name] = previous_offset+max(-step, min(step, goal-previous_offset))
            self._height_diagnostic = {"mode": self._height_candidate["mode"],
                "candidate_id": self._height_candidate["candidate_id"], "source_observation_tick": ev.get("physics_tick"),
                "current_RR_carry_recovery_permitted": permitted,
                "post_lift_recovery_offsets_deg": dict(self._height_recovery_offsets), "owners": [],
                "source_clocks_and_atomic_groups_unchanged": True, "task_or_support_credit_awarded": False,
                "exit": "later source owner replaces this segment; no permanent hip lock or home restore"}
        if self._p06_tail_source is not None:
            self._tail_diagnostic.update(layer_present=False, source_endpoint_issued=False,
                finite_source_tail_replaced=False, wheel_gain=None, status="not_applicable_no_P06_layer")
        for layer in self._continuous_layers:
            # A decreasing logical knee/hip angle is not a physical descent
            # predicate. Measured A uses these segments to carry an airborne
            # rear wheel toward/across the front plane. Requiring that endpoint
            # before dispatch would suppress the action that can create it.
            # Keep this finite sequence advisory; actual clearance/crossing and
            # placement are evaluated independently, with every residual open.
            # Only the separately verified late whole-body drop group waits
            # for a current over-platform entry; earlier RR carry keeps running.
            advance = self._sequence_permission(layer, task, observation)
            layer["advanced_this_tick"] = advance
            if advance:
                sample=layer["motion"].tick(); layer["ticks"]+=1
                layer["touched"].update(i for i,(a,b) in enumerate(zip(sample.full12,layer["last"])) if abs(a-b)>1e-9)
                if self._sequence_mode:
                    # An explicit wheel stop owns all authored wheel channels,
                    # even when this layer's previous local value was already 0.
                    # Do not restore the unchanged servos in Full12 snapshots.
                    layer["touched"].update(8+WHEEL_ORDER.index(name)
                        for group in sample.atomic_groups for name in group.channels if name in WHEEL_ORDER)
                layer["last"]=sample.full12; layer["sample"]=sample
            else:
                sample = layer["sample"]
                if sample is None:
                    if layer["stage"] == stage_id:
                        self.elapsed_s=0.; self.endpoint_issued=False
                    continue
            source_bias=self._source_normal_bias(layer["motion"],sample)
            if "rl_motion" in layer:
                if not layer["rl_dependency_wait"]:
                    rl_sample = layer["rl_motion"].tick()
                    layer["rl_ticks"] += 1
                    layer["rl_touched"].update(i for i in (4, 5)
                        if abs(rl_sample.full12[i] - layer["rl_last"][i]) > 1e-9)
                    layer["rl_sample"], layer["rl_last"] = rl_sample, rl_sample.full12
                layer["touched"].difference_update((4, 5))
            gain=1.
            if layer["stage"]=="P06" and retirement is not None:
                peak=layer.get("rolling_retirement_peak",0.)
                if retirement["measured_fraction"] is not None:
                    peak=max(peak,retirement["measured_fraction"])
                layer["rolling_retirement_peak"]=peak; gain=1.-peak
                if self._all_stage_acceptance:
                    # Current geometry can recover the original bounded rolling
                    # suggestion after retreat. Peak remains diagnostic only;
                    # later owners, original slew and task deadlines still win.
                    if retirement["measured_fraction"] is not None:
                        gain = 1.-retirement["measured_fraction"]
                        layer["rolling_last_live_gain"] = gain
                    else:
                        gain = layer.get("rolling_last_live_gain", gain)
                    self._retirement_diagnostic["gain_semantics"] = "bounded_current_geometry_recoverable"
                self._retirement_diagnostic.update(retirement,layer_present=True,
                    origin="current_live_P06_layer",peak_fraction=peak,wheel_gain=gain)
            if layer["stage"] == "P06" and pending_status is not None:
                if pending_status["fl_capture_pending"] and not pending_status["allow_capture_continuation"]:
                    gain = 0.  # Only this source owner's contribution; no residual mask.
                self._capture_continuation_diagnostic.update(P06_layer_present=True,
                    P06_source_tick=layer["ticks"], P06_source_advanced=advance,
                    P06_wheel_contribution_enabled=gain > 0., P06_wheel_gain=gain)
            replace_tail = False
            if self._p06_tail_source is not None and layer["stage"] == "P06":
                live = (retirement is not None and retirement["status"] == "live_measured"
                        and retirement["measured_fraction"] is not None and ev.get("valid") is True
                        and task.get("termination_reason") is None and ev.get("termination_reason") is None)
                replace_tail = live and sample.endpoint_issued and gain > 0.
                self._tail_diagnostic.update(layer_present=True, source_endpoint_issued=sample.endpoint_issued,
                    finite_source_tail_replaced=replace_tail, wheel_gain=gain,
                    status=("terminal_no_tail" if not live else "retired" if gain == 0. else
                            "live_endpoint_tail" if replace_tail else "finite_source_before_endpoint"))
            for i in layer["touched"]:
                if i in layer.get("capture_retired_servo_indices", ()):
                    tracking.discard(SERVO_ORDER[i])
                    continue
                # Scale this owner's contribution, never later owners or residuals.
                # Keep layer.last/sample as the original source (including its
                # zero endpoint); only this local owner's contribution changes.
                source_value = self._p06_tail_source[i-8] if replace_tail and i>=8 else sample.full12[i]
                if self._height_candidate is not None:
                    from .semantic_height_recovery import OWNERS, source_owner_height_target
                    owner = OWNERS.get(layer["stage"])
                    if owner is not None and i == owner[0]:
                        entry = layer["motion"].phase.start_full12[i]
                        original = source_value
                        source_value = source_owner_height_target(source_value=original, source_entry=entry,
                            reduction_deg=self._height_candidate["preparation_reduction_deg"][owner[1]],
                            recovery_deg=self._height_recovery_offsets[owner[1]])
                        self._height_diagnostic["owners"].append({"stage": layer["stage"], "channel": owner[1],
                            "fixed_source_entry_deg": entry, "original_source_target_deg": original,
                            "candidate_source_target_deg": source_value, "reduction_deg": original-source_value})
                proposed[i]=source_value*(gain if i>=8 else 1.)
                if self._reference_nominal:
                    normal_bias[i]=source_bias[i]
                if i<8:
                    source_tracking_eligible = (not self._reference_nominal or
                        layer["stage"] == stage_id or not sample.endpoint_issued)
                    if source_tracking_eligible and SERVO_ORDER[i] in sample.tracking_servo_names: tracking.add(SERVO_ORDER[i])
                    else: tracking.discard(SERVO_ORDER[i])
            if layer["stage"]==stage_id:
                self.elapsed_s=sample.elapsed_s; self.endpoint_issued=sample.endpoint_issued
            if "rl_motion" in layer and layer["rl_sample"] is not None:
                rl_sample = layer["rl_sample"]
                rl_bias = self._source_normal_bias(layer["rl_motion"], rl_sample)
                for i in layer["rl_touched"]:
                    if i in layer.get("capture_retired_servo_indices", ()):
                        tracking.discard(SERVO_ORDER[i])
                        continue
                    proposed[i], normal_bias[i] = rl_sample.full12[i], rl_bias[i]
                    if not rl_sample.endpoint_issued and SERVO_ORDER[i] in rl_sample.tracking_servo_names:
                        tracking.add(SERVO_ORDER[i])
                    else:
                        tracking.discard(SERVO_ORDER[i])
                if layer["stage"] == stage_id:
                    self.endpoint_issued = self.endpoint_issued and rl_sample.endpoint_issued
        if self._reference_nominal:
            self.normal_drive_bias_full12=tuple(normal_bias)
        if stage_id in ("P09","P12"):
            leg="RR" if stage_id=="P09" else "RL"; current=legs.get(leg,{})
            functional_rr = (stage_id == "P09"
                and self.spec.get("p09_lift_semantics") in FUNCTIONAL_RR_MODES)
            if functional_rr:
                # Source hip/knee/whole-body owners continue without a Q/height
                # dispatch gate. This extra P01-derived rolling suggestion is
                # retained for verified AIR; after the finite source, real
                # TOP-corner contact may also continue toward crossing/capture.
                # FRONT_WALL or unknown EDGE never enables that TOP extension.
                distance = current.get("front_distance_m", 1.)
                late_wait = any(self._rr_waiting_late_group(layer) for layer in self._continuous_layers)
                clearance = current.get("clearance_m", -1.)
                if late_wait:
                    try:
                        distance = _number(distance, "pending RR carry distance")
                        clearance = _number(clearance, "pending RR carry clearance")
                    except (SemanticObservationError, TypeError):
                        distance, clearance = math.inf, -math.inf
                rolling = bool(ev.get("valid") is True and task.get("termination_reason") is None
                    and ev.get("termination_reason") is None and current.get("current_lift_valid")
                    and current.get("motion_continuation_allowed") and current.get("air")
                    and not current.get("ground_contact") and current.get("within_lateral_span")
                    and (not late_wait or ev.get("physical_evidence_status")
                        in ("VERIFIED", "CONTACT_BEARING_UNVERIFIED"))
                    and (not late_wait or sum(self._rr_verified_bearing(legs.get(leg, {}))
                        for leg in LEG_ORDER if leg != "RR") >= self.spec["support"]["minimum_other_supports"])
                    and distance < self.spec["geometry"]["approach_max_m" if late_wait else "approach_min_m"]
                    and (distance < self.spec["geometry"]["workspace_min_m"]
                         or clearance >= (self.spec["geometry"]["top_gap_min_m"]
                            if self._rear_policy_timing_mode and current.get("within_top_xy") is True else 0.)))
                ordered_wheel_owner = bool(self._sequence_mode and any(
                    layer["sample"] is not None and
                    ((layer["stage"] in ("P07", "P09") and not layer["sample"].endpoint_issued
                      and not self._rr_waiting_late_group(layer))
                     or (layer.get("advanced_this_tick") and any(
                         set(group.channels).intersection(WHEEL_ORDER)
                         for group in layer["sample"].atomic_groups)))
                    for layer in self._continuous_layers))
                # Extra task feedback must not overwrite the finite source's
                # FR joint/wheel pulse or its authored stops just because the
                # task label has already advanced to P09. It remains available
                # after finite source execution, when current geometry permits.
                top_rolling = self._rr_top_continuation_allowed(task)
                rolling = (rolling or top_rolling) and not ordered_wheel_owner
                if rolling:
                    proposed[8:] = self._approach_wheel_prior
                self._rr_carry_diagnostic = dict(
                    source_joint_owners_continued=True, added_rolling_suggestion=rolling,
                    ordered_source_wheel_owner=ordered_wheel_owner,
                    late_reconfiguration_waiting=late_wait,
                    current_TOP_continuation_eligible=top_rolling,
                    reason="finite_ordered_source_wheel_owner" if ordered_wheel_owner else
                        "current_TOP_corner_continuation" if top_rolling and rolling else
                        "current_AIR_safe_approach" if rolling else
                        "no_extra_wall_push_source_and_residual_adjustment_continue",
                    fixed_lift_timer_gate=False, above_top_15mm_action_gate=False)
            elif (task.get("termination_reason") is None and ev.get("termination_reason") is None
                    and history.get("active_lift",{}).get(leg,False)
                    and current.get("clearance_m",-1.)>=self.spec["geometry"]["airborne_clearance_above_top_m"]
                    and current.get("front_distance_m",1.)<self.spec["geometry"]["approach_min_m"]):
                proposed[8:]=self._approach_wheel_prior
        return tuple(proposed),tuple(name for name in SERVO_ORDER if name in tracking)

    def _p05_preedge_recovery_status(self, task: Mapping[str, Any], *,
            prior_source_endpoint_issued: bool, prior_observation_tick: Any) -> dict[str, Any]:
        """Finite P05-only wheel advice, not a capture/handoff or residual gate.

        The prior endpoint plus a subsequent physical tick uses the backend's
        established one-write/one-step order. This is NOT an independent ACK
        verification. No recovery latch or renewed clock is introduced.
        """
        ev = task.get("physical_evaluator", {})
        legs = ev.get("current_legs", {})
        fl = legs.get("FL", {})
        history = ev.get("history", {})
        tick, now = ev.get("physics_tick"), ev.get("simulation_time_s")
        age = task.get("stage_elapsed_s")
        start = self.spec["stages"]["P05"]["maximum_task_duration"]
        end = start + self.spec["local_timeout_policy"]["maximum_extension_s"]
        mode = self.spec["nominal"].get("p05_preedge_approach_recovery", P05_PREEDGE_RECOVERY_MODE)
        # v3 permits the same measured advice once the complete source has
        # actually issued and a subsequent tick is observed. It does not wait
        # for the task's nominal timeout to begin using a still-positive gap.
        # Endpoint/subsequent-tick/fresh-wheel-owner checks below still apply;
        # the absolute end and all v1/v2 behavior remain unchanged.
        window_start = 0. if mode == P05_COMPLETED_SOURCE_RECOVERY_MODE else start
        finite = lambda value: type(value) in (int, float) and math.isfinite(value)
        gap, distance = fl.get("clearance_m"), fl.get("front_distance_m")
        geometry = self.spec["geometry"]
        supports = tuple(leg for leg in LEG_ORDER if leg != "FL"
            and legs.get(leg, {}).get("support") is True
            and legs[leg].get("bearing_verified") is True and legs[leg].get("air") is False
            and (legs[leg].get("ground_contact") is True or legs[leg].get("top_surface_contact") is True))
        fresh = tuple({"stage": layer["stage"], "source_tick": layer["sample"].tick_index,
                       "channels": tuple(name for name in group.channels if name in WHEEL_ORDER)}
            for layer in self._continuous_layers if layer.get("advanced_this_tick", True)
            and layer.get("sample") is not None for group in layer["sample"].atomic_groups
            if set(group.channels).intersection(WHEEL_ORDER))
        consecutive = bool(type(tick) is int and tick >= 0 and type(prior_observation_tick) is int
            and tick == prior_observation_tick + 1 and finite(now)
            and math.isclose(now, tick/self.physics_hz, rel_tol=0., abs_tol=1e-9))
        checks = {
            "P05_only": task.get("stage_id") == "P05",
            "absolute_stage_age_window": finite(age) and window_start <= age < end,
            "valid_no_abort": ev.get("valid") is True and ev.get("termination_reason") is None
                and task.get("termination_reason") is None
                and ev.get("physical_evidence_status") in ("VERIFIED", "CONTACT_BEARING_UNVERIFIED"),
            "prior_qualified_FL_lift": history.get("active_lift", {}).get("FL") is True,
            "FR_placement_history": history.get("placed", {}).get("FR") is True,
            "not_crossed_or_placed_FL": history.get("front_edge_crossed", {}).get("FL") is False
                and history.get("placed", {}).get("FL") is False,
            "current_FL_AIR_no_contact": fl.get("air") is True and fl.get("ground_contact") is False
                and fl.get("obstacle_pair_active") is False and fl.get("top_surface_contact") is False,
            "positive_conservative_top_gap": finite(gap) and gap > 0.,
            "current_lateral_valid": fl.get("within_lateral_span") is True,
            "bounded_preedge_distance": finite(distance)
                and -geometry["approach_max_m"] <= distance <= geometry["xy_measurement_tolerance_m"],
            "verified_other_supports": len(supports) >= self.spec["support"]["minimum_other_supports"],
            "source_endpoint_issued_then_subsequent_tick": prior_source_endpoint_issued is True
                and self.endpoint_issued is True and consecutive,
            "fresh_source_wheel_owner_absent": not fresh,
        }
        recross = None
        if mode in (P05_SAME_AIR_RECROSS_MODE, P05_COMPLETED_SOURCE_RECOVERY_MODE):
            events = history.get("event_ticks", {})
            crossings = events.get("front_edge_crossed", {}) if isinstance(events, Mapping) else {}
            cross_tick = crossings.get("FL") if isinstance(crossings, Mapping) else None
            air_count = fl.get("consecutive_air_samples")
            counter_valid = bool(type(tick) is int and tick >= 0
                and type(air_count) is int and 1 <= air_count <= tick + 1)
            air_start = tick - air_count + 1 if counter_valid else None
            valid_span = bool(counter_valid and type(cross_tick) is int
                and 0 <= air_start <= cross_tick <= tick)
            first_approach = history.get("front_edge_crossed", {}).get("FL") is False
            same_air_recross = bool(history.get("front_edge_crossed", {}).get("FL") is True
                and fl.get("active_attempt") is True and valid_span
                and fl.get("within_top_xy") is False and finite(distance)
                and distance < -geometry["xy_measurement_tolerance_m"])
            # FL's old active_attempt may survive a post-cross GROUND event.
            # The uninterrupted measured AIR streak must contain this cross;
            # no history clearing, contact credit or renewed timer is used.
            checks.pop("not_crossed_or_placed_FL")
            checks.update(unplaced_FL=history.get("placed", {}).get("FL") is False,
                first_approach_or_same_air_recross=first_approach or same_air_recross)
            recross = dict(branch="first_approach" if first_approach else
                "same_air_recross" if same_air_recross else "ineligible_history_or_current_geometry",
                consecutive_air_samples=air_count, air_start_tick=air_start,
                cross_event_tick=cross_tick, same_air_span_valid=valid_span)
        result = {"mode": mode, "eligible": all(checks.values()), "checks": checks,
            "reasons": tuple(key for key, passed in checks.items() if not passed),
            "source_observation_tick": tick, "prior_source_observation_tick": prior_observation_tick,
            "stage_elapsed_s": age, "absolute_window_s": (window_start, end), "window_end_exclusive": True,
            "source_endpoint_issued": self.endpoint_issued,
            "prior_source_endpoint_issued": prior_source_endpoint_issued,
            "source_elapsed_s": self.elapsed_s, "source_tick_consecutive": consecutive,
            "endpoint_evidence": "issued_source_endpoint_then_one_write_one_step_subsequent_tick",
            "independent_ack_verified": False, "fresh_source_wheel_owners": fresh,
            "current_FL_gap_m": gap, "current_FL_front_distance_m": distance,
            "observed_other_support_contacts": supports, "wheel_prior_rad_s": self._approach_wheel_prior,
            "timing": "nominal_suggestion_before_mapper_residual_and_actuator_dispatch",
            "actor_observability": "current_nominal_recovery_advice_encoded_source_internals_and_full_contact_classes_not_individually_encoded",
            "recovery_clock_reset_or_latch": False, "policy_residual_restricted": False,
            "phase_or_capture_credit_awarded": False}
        if recross is not None:
            result["same_air_recross"] = recross
        return result

    def _approach_assist_required(self, task: Mapping[str,Any]) -> bool:
        evaluation=task.get("physical_evaluator")
        if (task.get("termination_reason") is not None or not isinstance(evaluation,Mapping)
                or evaluation.get("valid") is not True or evaluation.get("termination_reason") is not None):
            return False
        history=evaluation.get("history",{}).get("active_lift",{})
        legs=evaluation.get("current_legs",{})
        if history.get("FR") is not True or "FR" not in legs:
            return False
        other_supports=sum(legs.get(leg,{}).get("support") is True for leg in LEG_ORDER if leg!="FR")
        current=legs["FR"]; geometry=self.spec["geometry"]
        return (other_supports >= self.spec["support"]["minimum_other_supports"]
            and _number(current.get("clearance_m"),"current FR clearance") >= geometry["airborne_clearance_above_top_m"]
            and _number(current.get("front_distance_m"),"current FR front distance") < geometry["approach_min_m"])

    def _observe_final_stop_owner(self, task: Mapping[str, Any], observation: Any) -> bool:
        """Continue stopping once a real completion observation has started.

        Capture the preceding nominal request, never actual q/final drive or
        this tick's future forward/home pulse. A P12 physical completion may
        precede the first P13 source sample; do not restart forward advice just
        because that source has not issued its historical stop yet. The live
        evaluator still owns the unchanged window and every failure. Current
        speed loss is diagnostic, not a reason to postpone nominal stopping.
        """
        if self._final_stop_mode is None:
            return False
        ev = task.get("physical_evaluator", {})
        terminal = task.get("termination_reason") is not None or ev.get("termination_reason") is not None
        in_phase = task.get("stage_id") == "P13"
        tick, now = ev.get("physics_tick"), ev.get("simulation_time_s")
        elapsed = ev.get("post_completion_elapsed_s")
        finite = lambda x: type(x) in (int, float) and math.isfinite(x)
        fresh = bool(observation is not None and type(tick) is int and tick >= 0 and finite(now)
            and math.isclose(now, tick/self.physics_hz, rel_tol=0., abs_tol=1e-9)
            and _get(observation, "physics_tick") == tick
            and finite(_get(observation, "simulation_time_s"))
            and math.isclose(_get(observation, "simulation_time_s"), now, rel_tol=0., abs_tol=1e-9))
        placed = ev.get("history", {}).get("placed", {})
        legs = ev.get("current_legs", {})
        leg_evidence = all(isinstance(row, Mapping) and row.get("load_fraction_valid") is True
            and row.get("bearing_verified") is True and finite(row.get("bearing_force_n"))
            and row["bearing_force_n"] >= 0. and finite(row.get("load_fraction"))
            and 0. <= row["load_fraction"] <= 1.
            and type(row.get("air")) is bool and type(row.get("support")) is bool
            and not (row["air"] and row["support"])
            and row.get("within_top_xy") is True and finite(row.get("clearance_m"))
            and row["clearance_m"] >= self.spec["geometry"]["top_gap_min_m"]
            for row in (legs.get(leg) for leg in LEG_ORDER))
        top_supports = sum(isinstance(row, Mapping) and row.get("top_surface_contact") is True
            and self._rr_verified_bearing(row)
            for row in (legs.get(leg, {}) for leg in LEG_ORDER))
        current = {key: ev.get(key) for key in ("valid", "physical_evidence_status", "final_region_valid",
            "final_controlled", "final_support_available", "task_completed_controlled",
            "post_completion_observation_started", "post_completion_observation_complete",
            "post_completion_loss_observed")}
        takeover_valid = bool(fresh and leg_evidence and ev.get("valid") is True and not terminal
            and ev.get("physical_evidence_status") == "VERIFIED"
            and all(placed.get(leg) is True for leg in LEG_ORDER)
            and top_supports >= self.spec["support"]["minimum_other_supports"]
            and all(ev.get(key) is True for key in ("final_region_valid",
                "final_support_available", "post_completion_observation_started"))
            and ev.get("post_completion_observation_complete") is False
            and ev.get("post_completion_loss_observed") is False
            and finite(elapsed) and 0. <= elapsed < self.spec["final"]["post_completion_observation_s"]
            and elapsed <= now)
        current_valid = bool(takeover_valid and ev.get("final_controlled") is True
            and ev.get("task_completed_controlled") is True)
        issued_stop = bool(self.state_id == "P13" and any(layer["stage"] == "P13"
            and layer.get("sample") is not None for layer in self._continuous_layers)
            and max(map(abs, self.nominal_full12[8:])) <= self.spec["final"]["maximum_commanded_wheel_speed_rad_s"])
        if self._final_stop_owner is not None and (terminal or not in_phase):
            self._final_stop_owner_retired = True
        if (self._final_stop_owner is None and not self._final_stop_owner_retired
                and in_phase and takeover_valid):
            self._final_stop_owner = {"entry_observation_tick": tick,
                "entry_sim_time_s": now, "post_window_start_s": now-elapsed,
                "entry_post_elapsed_s": elapsed, "issued_nominal_full12": self.nominal_full12,
                "held_nominal_servo_deg": self.nominal_full12[:8],
                "held_tracking_servo_names": self.tracking_servo_names,
                "held_normal_servo_bias_deg": self.normal_drive_bias_full12[:8],
                "entry_top_bearing_support_count": top_supports, "entry_evidence": current}
        active = bool(self._final_stop_owner is not None and not self._final_stop_owner_retired
                      and in_phase and not terminal)
        self._final_stop_diagnostic = {"mode": self._final_stop_mode, "active": active,
            "acquisition_semantics": "post_window_triggered_nominal_stop_takeover_v2",
            "status": ("holding_currently_eligible" if current_valid else "holding_live_eligibility_lost") if active
                else "retired_episode_or_phase" if self._final_stop_owner_retired else "not_acquired",
            "entry": self._final_stop_owner, "current_observation_tick": tick,
            "current_evidence_fresh": fresh, "current_entry_eligibility": current_valid,
            "current_post_window_takeover_eligibility": takeover_valid,
            "current_entry_eligibility_semantics": "strict_current_measured_control_not_takeover_gate",
            "current_evidence": current, "current_top_bearing_support_count": top_supports,
            "current_four_leg_evidence_valid": leg_evidence,
            "preceding_issued_nominal_wheels_stopped": issued_stop,
            "source_clocks_continue_contributions_retired": active,
            "raw_residual_mapper_or_evaluator_reset": False, "task_success_awarded": False}
        return active

    def _final_stop_request(self):
        """Stop first, then issue the source's home-like servo request once.

        The opt-in refinement retires the historical differential wheel pulse,
        not the home intent. This is a nominal request through the SAME mature
        mapper: no actual-q write, mapper reset, residual mask, or new success
        predicate. The old mode continues to hold its preceding nominal pose.
        """
        owner = self._final_stop_owner
        diagnostic = self._final_stop_diagnostic
        smooth = self._final_stop_mode == "source_home_after_physical_stop_v2"
        refine = self._final_stop_mode in ("source_home_after_physical_stop_v1",
                                          "source_home_after_physical_stop_v2")
        if (refine and self._final_home_recovery is None
                and diagnostic["current_entry_eligibility"]):
            target = _vector(self.contract.phase("P13").end_full12, 12, "source P13 home")
            self._final_home_recovery = {
                "entry_observation_tick": diagnostic["current_observation_tick"],
                "target_servo_deg": target[:8],
                "source": "recording_motion_contract.P13.end_full12.servo8",
                "source_wheel_pulse_replayed": False,
            }
            if smooth:
                # The previous step request excited loaded wheel contact even
                # after body speed settled. Shape this nominal request only;
                # never change measured stop limits or the fixed observation.
                self._final_home_recovery.update(
                    start_servo_deg=self.nominal_full12[:8],
                    ramp_duration_s=float(self.spec["final"]["stable_duration_s"]),
                    ramp_semantics="quintic_source_home_nominal_then_hold_v1")
        if refine:
            diagnostic["home_recovery"] = {
                "enabled": True, "active": self._final_home_recovery is not None,
                "entry": self._final_home_recovery,
                "status": "source_home_request_then_measured_settle" if self._final_home_recovery
                    else "stopped_waiting_current_physical_control",
                "home_pose_is_success_gate": False,
                "physical_observation_window_changed": False,
                "mapper_history_reset": False, "residual_channels_restricted": False,
            }
        if self._final_home_recovery is not None:
            # The reference home group explicitly owns all eight servos. Its
            # preceding carry tracking/bias owners no longer apply, while the
            # mapper's physical slew and previous applied-command state remain.
            home = self._final_home_recovery
            target = home["target_servo_deg"]
            if smooth:
                elapsed = max(0., (diagnostic["current_observation_tick"]
                    - home["entry_observation_tick"])/self.physics_hz)
                u = _clip(elapsed/home["ramp_duration_s"])
                fraction = u*u*u*(10.+u*(-15.+6.*u))
                if u < 1.:
                    target = tuple(start+fraction*(end-start)
                        for start, end in zip(home["start_servo_deg"], target))
                diagnostic["home_recovery"].update(
                    nominal_ramp_elapsed_s=elapsed, nominal_ramp_fraction=fraction,
                    nominal_ramp_complete=u >= 1.)
            return target+(0.,)*4, (), (0.,)*12
        return (owner["held_nominal_servo_deg"]+(0.,)*4,
                owner["held_tracking_servo_names"],
                owner["held_normal_servo_bias_deg"]+(0.,)*4)

    def evaluate(self, stage: str | Mapping[str,Any], observation: Any=None) -> tuple[float,...]:
        stage_id=stage if isinstance(stage,str) else str(stage["stage_id"])
        prior_endpoint = bool(self.state_id == "P05" and self.endpoint_issued)
        prior_tick = self._capture_continuation_diagnostic.get("source_observation_tick")
        # Validate before any layer/source clock, ownership or nominal mutation.
        retirement=self._retirement_measurement(stage,observation) if isinstance(stage,Mapping) else None
        capture=self._capture_hold_measurement(stage,observation) if isinstance(stage,Mapping) else None
        final_stop = self._observe_final_stop_owner(stage, observation) if isinstance(stage, Mapping) else False
        phase=self.contract.phase(stage_id)
        handoff=self.state_id is not None and self.state_id!=stage_id
        if self.state_id!=stage_id:
            self.state_id=stage_id
            self._start_source_motion(self._source_motion,phase)
        source=self._source_motion.tick()
        self.elapsed_s=source.elapsed_s
        proposed=source.full12
        self.endpoint_issued=source.endpoint_issued
        proposed_tracking=source.tracking_servo_names
        self.normal_drive_bias_full12=self._source_normal_bias(self._source_motion,source)
        if isinstance(stage,Mapping) and self.spec["nominal"].get("continuous_channel_inheritance") is True:
            proposed,proposed_tracking=self._continuous_advisory(stage,retirement,capture,observation)
        # The string-only test seam remains the unmodified finite advisory.
        # All feedback here is in the current task snapshot, with no timer,
        # historical posture or reference-force gate. The opted-in P06 layer
        # separately retains its audited peak of measured workspace retirement.
        if stage_id=="P02" and isinstance(stage,Mapping) and self._approach_assist_required(stage):
            proposed=proposed[:8]+self._approach_wheel_prior
        if (stage_id == "P05" and isinstance(stage, Mapping)
                and self._p05_pending_capture_mode is not None):
            ev = stage.get("physical_evaluator", {})
            current = ev.get("current_legs", {}).get("FL", {})
            role = stage.get("transfer_roles", {}).get("FL", {})
            captured_handoff = False
            if self._p05_pending_capture_mode == "current_FL_capture_wheel_continuation_to_handoff_v2":
                # Placement is observed at 120 Hz, while phase handoff happens
                # at the next 15 Hz decision. Do not re-expose an old P05 stop
                # in those intervening ticks. Continue only an already rolling
                # suggestion with CURRENT verified capture; no timer or latch.
                # These layers have already advanced once above. A newly
                # authored wheel event (including a stop) retains precedence.
                fresh_wheel_event = any(set(group.channels).intersection(WHEEL_ORDER)
                    for layer in self._continuous_layers if layer.get("advanced_this_tick", True)
                    and layer["sample"] is not None for group in layer["sample"].atomic_groups)
                captured_handoff = bool(
                    ev.get("history", {}).get("placed", {}).get("FL") is True
                    and current.get("top_contact") is True and current.get("support") is True
                    and current.get("bearing_verified") is True and current.get("within_top_xy") is True
                    and current.get("air") is False and current.get("ground_contact") is False
                    and type(ev.get("physics_tick")) is int and ev["physics_tick"] % 8 != 0
                    and self.nominal_full12[8:] == self._approach_wheel_prior
                    and not fresh_wheel_event)
            # Same P06-class wheel suggestion before placement; no phase skip,
            # fake contact, servo reset, or residual suppression. Existing servo
            # layers continue. v2 bridges a current capture only until handoff;
            # unsafe/current-contact loss and fresh source events still win.
            if ((role.get("pending_capture") or captured_handoff) and stage.get("termination_reason") is None
                    and ev.get("valid") is True and ev.get("termination_reason") is None
                    and len([leg for leg in role.get("observed_support_contacts", ()) if leg != "FL"]) >= 2
                    and current.get("clearance_m", -1.) >= self.spec["geometry"]["top_gap_min_m"]
                    and current.get("front_distance_m", 1.) < self.spec["geometry"]["approach_max_m"]):
                if not self._capture_continuation or (
                        (captured_handoff or _capture_continuation_status(self.spec, ev, stage_id)["allow_capture_continuation"])
                        and not any(set(group.channels).intersection(WHEEL_ORDER)
                            for layer in self._continuous_layers if layer.get("advanced_this_tick", True)
                            and layer["sample"] is not None for group in layer["sample"].atomic_groups)):
                    proposed = proposed[:8]+self._approach_wheel_prior
        if self._p05_preedge_recovery and isinstance(stage, Mapping):
            self._p05_preedge_diagnostic = self._p05_preedge_recovery_status(stage,
                prior_source_endpoint_issued=prior_endpoint, prior_observation_tick=prior_tick)
            if self._p05_preedge_diagnostic["eligible"]:
                proposed = proposed[:8]+self._approach_wheel_prior
        if stage_id=="P13" and self.endpoint_issued and not final_stop:
            proposed=tuple(self.spec["final"]["home_servo_pose_deg"])+(0.,)*4
        if final_stop:
            proposed, proposed_tracking, self.normal_drive_bias_full12 = self._final_stop_request()
            self._final_stop_diagnostic.update(source_elapsed_s=self.elapsed_s,
                source_endpoint_issued=self.endpoint_issued,
                source_layer_ticks={layer["stage"]: layer["ticks"] for layer in self._continuous_layers})
        if self._reference_nominal:
            # Logical source requests hold between authored events. The one
            # mature mapper below owns physical servo slew and load feedback;
            # changing a logical target every tick would repeatedly reset it.
            # Only touched source channels were replaced above: this is not a
            # phase.start pose restore or a reset of residual/mapper history.
            self.nominal_full12=Full12Command.from_full12(proposed).clamped().to_full12()
            self.tracking_servo_names=proposed_tracking
        elif not handoff:
            servo_rates = self._servo_rate_overrides.get(stage_id,
                (self.spec["nominal"]["servo_handoff_rate_deg_s"],) * 8)
            rates = servo_rates + (self.spec["nominal"]["wheel_handoff_rate_rad_s2"],) * 4
            self.nominal_full12=Full12Command.from_full12(tuple(old+max(-rate/self.physics_hz,min(rate/self.physics_hz,target-old)) for old,target,rate in zip(self.nominal_full12,proposed,rates))).clamped().to_full12()
            self.tracking_servo_names=proposed_tracking
        return self.nominal_full12


class SemanticControllerAdapter:
    """Explicit startup replacement producing the established ControllerFrame."""
    def __init__(self, spec: Any, contract: Any, *, task_spec_path: Path | str = DEFAULT_TASK_SPEC_PATH,
                 supervisor: TaskStageSupervisor | None = None, nominal_provider: NominalMotionProvider | None = None):
        self.spec=spec; self.contract=contract
        self.supervisor=TaskStageSupervisor(task_spec_path) if supervisor is None else supervisor
        self.evaluator=self.supervisor.evaluator
        self.nominal_provider=NominalMotionProvider(contract,spec=self.supervisor.spec,fsm_spec=spec) if nominal_provider is None else nominal_provider
        self.motion=self.nominal_provider
        self.physics_tick=0; self.lifecycle=Lifecycle.EXECUTE_MOTION
        self.history:list[ControllerEvent]=[]; self.termination:TaskTermination|None=None
        self.first_blocker=None; self._last_time:float|None=None

    @classmethod
    def from_paths(cls, fsm_path: Path | str, motion_contract_path: Path | str, *, task_spec_path: Path | str=DEFAULT_TASK_SPEC_PATH):
        return cls(load_fsm_spec(Path(fsm_path)),load_motion_contract(Path(motion_contract_path)),task_spec_path=task_spec_path)

    @classmethod
    def from_live_prefix(cls, spec: Any, contract: Any, *, supervisor: TaskStageSupervisor,
                         nominal_full12: Sequence[float], tracking_servo_names: Sequence[str],
                         physics_tick: int, sim_time_s: float):
        task=supervisor.snapshot; ev=task.get("physical_evaluator",{})
        if (type(physics_tick) is not int or physics_tick<=0 or physics_tick%8
                or supervisor._last_observation_tick!=physics_tick
                or not math.isclose(sim_time_s,physics_tick/120.,rel_tol=0.,abs_tol=1e-9)
                or supervisor.episode_started_s!=0. or task.get("termination_reason") is not None
                or ev.get("valid") is not True or ev.get("termination_reason") is not None
                or task.get("stage_id")!=supervisor.stage_id):
            raise SemanticObservationError("handoff must be a valid live natural-prefix decision tick")
        provider=NominalMotionProvider.from_handoff(contract,spec=supervisor.spec,stage_id=supervisor.stage_id,
            nominal_full12=nominal_full12,tracking_servo_names=tracking_servo_names,fsm_spec=spec)
        result=cls(spec,contract,supervisor=supervisor,nominal_provider=provider)
        result.physics_tick=physics_tick+1; result._last_time=sim_time_s
        task=result.task_snapshot
        frame=ControllerFrame(physics_tick,sim_time_s,supervisor.stage_id,result.lifecycle,
            provider.nominal_full12,True,True,False,provider.tracking_servo_names,ZERO12,ZERO12,
            {"mode":"semantic_live_prefix_handoff","semantic_task":task},False,None,None,())
        return result,frame

    @property
    def state(self): return SimpleNamespace(state_id=self.supervisor.stage_id)
    @property
    def phase(self): return self.contract.phase(self.supervisor.stage_id)
    @property
    def task_progress(self): return float(self.supervisor.snapshot.get("phase_progress",0.))
    @property
    def task_snapshot(self):
        task=self.supervisor.snapshot
        diagnostics=self.nominal_provider.nominal_suggestion_diagnostics
        return {**task,"nominal_provider_diagnostics":diagnostics} if diagnostics else task

    def step(self, observation: Any, *, sim_time_s: float | None=None) -> ControllerFrame:
        now=self.physics_tick/120. if sim_time_s is None else float(sim_time_s)
        if self._last_time is not None and now <= self._last_time:
            raise SemanticObservationError("controller time must advance monotonically")
        old_events=len(self.supervisor.transition_evidence)
        task=self.supervisor.observe_and_update(observation,sim_time_s=now)
        command=self.nominal_provider.evaluate(task,observation)
        task=self.task_snapshot  # Copied current suggestion diagnostics, not applied ACKs.
        events=[]
        for row in self.supervisor.transition_evidence[old_events:]:
            event=ControllerEvent(now,row["from_stage"],Lifecycle.EXECUTE_MOTION.value,Lifecycle.DONE.value,row["reason"],row)
            events.append(event); self.history.append(event)
        self.lifecycle=Lifecycle.EXECUTE_MOTION if task["entry_valid"] else Lifecycle.WAIT_ENTRY
        if task["termination_reason"] is not None:
            self.lifecycle=Lifecycle.DONE
            self.termination=TaskTermination(TaskResult(task["termination_reason"]),task["stage_id"],self.lifecycle.value,now,
                "physical task success" if task["success"] else (task["physical_evaluator"].get("reason") or "semantic task duration exhausted"),task)
            command=command[:8]+(0.,)*4
        source_bias=(self.nominal_provider.normal_drive_bias_full12 if self.termination is None else ZERO12)
        feedback_details={"mode":"semantic_no_reference_feedback","semantic_task":task}
        if self.nominal_provider._reference_nominal:
            feedback_details.update(mode="successful_fsm_derived_nominal",source_normal_drive_bias_full12=source_bias)
        frame=ControllerFrame(self.physics_tick,now,task["stage_id"],self.lifecycle,command,self.physics_tick%8==0,True,False,
            self.nominal_provider.tracking_servo_names,ZERO12,source_bias,
            feedback_details,
            self.nominal_provider.endpoint_issued,self.termination,None,tuple(events))
        self._last_time=now; self.physics_tick+=1
        return frame

"""Five disjoint reward families for the finite-horizon physical task.

Potential uses augmented physical task/history state, not a phase-local origin.
All costs are bounded and integrated in actual simulated seconds. No reference
trajectory, support-leg template, or incentive to produce nonzero actions exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .semantic_observation import CONFIG_ROOT, SemanticObservationFrame, finite, vector
from .semantic_task_quality import (OBJECTIVE as TASK_CONDITIONED_QUALITY_OBJECTIVE,
                                    SPACE_CONFIG, task_space_quality_sample)

FAMILIES = ("task_progress", "body_stability", "contact_motion_quality",
            "control_smoothness", "control_regularization")
DEFAULT_REWARD_CONFIG = CONFIG_ROOT / "reward_config.yaml"
ROLE_TRANSFER_VERSION = "diagonal_transfer_roles_v1"

CARRY_BODY_ALLOWANCE_MODE = "current_functional_carry_and_capture_settle_v1"
TASK_FIRST_OBJECTIVE = "task_first_recovery_v1"
FRONT_QUALITY_OBJECTIVE = "fl_capture_front_body_quality_v1"


def _front_quality_substate(task: Mapping[str, Any]) -> str:
    """Audit labels only: none can disable or zero the front quality cost."""
    # This is the production semantic-task envelope, not the inner evaluator
    # snapshot. Historical capture is distinct from current support/load.
    evaluation = task.get("physical_evaluator", {})
    history = evaluation.get("history", {})
    if history.get("placed", {}).get("FR") is True:
        return "CAPTURED"
    if history.get("front_edge_crossed", {}).get("FR") is True:
        return "POST_CROSS_CAPTURE"
    current = evaluation.get("current_legs", {}).get("FR", {})
    if history.get("active_lift", {}).get("FR") is True:
        if (current.get("air") is True and not current.get("ground_contact")
                and current.get("clearance_m", -1.) >= .015):
            return "FUNCTIONAL_CARRY"
        return "QUALIFIED_LIFT_OR_TRANSFER"
    return "UNQUALIFIED_PREPARATION"


def _validate_task_priority(values: Mapping[str, Any]) -> None:
    """The recovery profile disables reward preferences, never physical safety.

    We keep the existing five-family arithmetic and measured diagnostics. The
    explicit epsilon is a versioned assertion about those family weights, not a
    second hidden multiplier or a phase-dependent quality switch. A later
    nonzero-quality candidate needs its own explicit objective version.
    """
    objective = values.get("objective_profile")
    if "task_space_quality" in values and objective != TASK_CONDITIONED_QUALITY_OBJECTIVE:
        raise ValueError("task-space quality settings require their explicit objective version")
    if objective == TASK_CONDITIONED_QUALITY_OBJECTIVE:
        expected_front = {"phases": ["P01", "P02"], "transfer_weight_floor": .5,
                          "attitude_fraction": .5, "rate_fraction": .5, "sample_audit": True}
        if (values.get("revision") != TASK_CONDITIONED_QUALITY_OBJECTIVE
                or values.get("task_space_quality") != SPACE_CONFIG
                or values.get("front_body_quality") != expected_front
                or isinstance(values.get("quality_epsilon"), bool)
                or finite(values.get("quality_epsilon"), "task quality epsilon") != .06
                or values["family_weights"] != dict(zip(FAMILIES, (1., .06, 0., 0., 0.), strict=True))
                or values["attitude_scale_rad"] != .5 or values["euler_rate_scale_rad_s"] != .5
                or values["control_regularization_enabled"] is not False):
            raise ValueError("task-quality v1 binds preserved front cost and bounded measured geometry only")
        return
    if objective == FRONT_QUALITY_OBJECTIVE:
        expected = {"phases": ["P01", "P02"], "transfer_weight_floor": .5,
                    "attitude_fraction": .5, "rate_fraction": .5, "sample_audit": True}
        if values.get("front_body_quality") != expected:
            raise ValueError("front quality v1 requires explicit P01/P02 positive-floor body-only contract")
        if (isinstance(values.get("quality_epsilon"), bool)
                or finite(values.get("quality_epsilon"), "front quality epsilon") != .03
                or values["family_weights"] != dict(zip(FAMILIES, (1., .03, 0., 0., 0.), strict=True))
                or values["attitude_scale_rad"] != .5 or values["euler_rate_scale_rad_s"] != .5
                or values["control_regularization_enabled"] is not False):
            raise ValueError("front quality v1 binds only .03 body coefficient and declared physical scales")
        return
    if "front_body_quality" in values:
        raise ValueError("front quality settings require their explicit objective version")
    if objective is None:
        if "quality_epsilon" in values:
            raise ValueError("quality epsilon requires an explicit objective profile")
        return
    if objective != TASK_FIRST_OBJECTIVE:
        raise ValueError("unknown task-priority objective profile")
    epsilon = values.get("quality_epsilon")
    if isinstance(epsilon, bool) or finite(epsilon, "quality epsilon") != 0.:
        raise ValueError("task-first recovery v1 requires quality epsilon zero")
    if (values["family_weights"]["task_progress"] != 1.
            or any(values["family_weights"][name] != 0. for name in FAMILIES[1:])):
        raise ValueError("task-first recovery requires task weight one and all quality weights zero")


def _validate_carry_body_allowance(values: Mapping[str, Any]) -> None:
    mode = values.get("carry_body_allowance")
    if mode is None:
        if "capture_settle_window_s" in values:
            raise ValueError("capture settle window requires carry body allowance mode")
        return
    if mode != CARRY_BODY_ALLOWANCE_MODE:
        raise ValueError("unknown carry body allowance mode")
    window = values.get("capture_settle_window_s")
    if isinstance(window, bool) or not 0. < finite(window, "capture settle window") <= .5:
        raise ValueError("capture settle window must be positive and at most 0.5 s")


def _current_functional_body_allowance(task: Mapping[str, Any], window_s: float) -> float:
    """Body-cost floor only; existing current evidence, never a task/bonus gate.

    The v1 engineering limits mirror the current task: 15 mm non-RR airborne
    clearance, near-front [-.40, .15] m or current platform XY, and two verified
    other supports. No motion derivative, normalized load or phase label is a
    substitute for earned qualification and current usable geometry/contact.
    """
    if task.get("transfer_roles_version") != ROLE_TRANSFER_VERSION:
        return 0.
    ev = task.get("physical_evaluator", {})
    if (not isinstance(ev, Mapping) or ev.get("valid") is not True
            or ev.get("physical_evidence_status") not in ("VERIFIED", "CONTACT_BEARING_UNVERIFIED")
            or task.get("termination_reason") is not None
            or ev.get("termination_reason") is not None):
        return 0.
    legs, history = ev.get("current_legs", {}), ev.get("history", {})
    tick = ev.get("physics_tick")
    if not isinstance(legs, Mapping) or not isinstance(history, Mapping) or type(tick) is not int:
        return 0.

    def measured(value):
        try:
            return None if isinstance(value, bool) else finite(value, "carry measured value")
        except (TypeError, ValueError):
            return None

    def support(row):
        if not isinstance(row, Mapping):
            return False
        force = measured(row.get("bearing_force_n"))
        return (row.get("support") is True and row.get("bearing_verified") is True
                and force is not None and force >= .2 and row.get("air") is False
                and (row.get("ground_contact") is True or
                     (row.get("top_contact") is True and row.get("top_surface_contact") is True)))

    order = ("FR", "FL", "RR", "RL")
    qualified, placed = history.get("active_lift", {}), history.get("placed", {})
    crossed = history.get("front_edge_crossed", {})
    placement_ticks = history.get("event_ticks", {}).get("placed", {})
    allowance = 0.
    for index, leg in enumerate(order):
        row = legs.get(leg, {})
        if (not isinstance(row, Mapping) or qualified.get(leg) is not True
                or row.get("active_attempt") is not True
                or row.get("ground_contact") is not False
                or row.get("within_lateral_span") is not True):
            continue
        distance, clearance = measured(row.get("front_distance_m")), measured(row.get("clearance_m"))
        in_region = row.get("within_top_xy") is True or (
            distance is not None and -.40 <= distance <= .15)
        if (distance is None or clearance is None or not in_region
                or sum(support(legs.get(other, {})) for other in order if other != leg) < 2):
            continue
        top_contact = (row.get("top_contact") is True and row.get("top_surface_contact") is True
                       and row.get("within_top_xy") is True and row.get("air") is False
                       and row.get("top_geometry") is True and -.015 <= clearance <= .025)
        if placed.get(leg) is True:
            event_tick = placement_ticks.get(leg)
            if (crossed.get(leg) is True and top_contact and support(row)
                    and type(event_tick) is int and 0 <= event_tick <= tick):
                # Real placement age, not reward-instance time or phase age.
                allowance = max(allowance, max(0., 1.-(tick-event_tick)/(120.*window_s)))
            continue
        if placed.get(leg) is not False or not all(placed.get(p) is True for p in order[:index]):
            continue
        if leg == "RR":
            usable = (row.get("current_lift_valid") is True
                      and row.get("motion_continuation_allowed") is True
                      and (row.get("air") is True or row.get("obstacle_pair_active") is True))
        else:
            usable = top_contact or (row.get("air") is True
                and row.get("obstacle_pair_active") is False
                and clearance is not None and clearance >= .015)
        if usable:
            allowance = 1.
    return allowance



@dataclass(frozen=True)
class SemanticRewardConfig:
    values: Mapping[str, Any]
    path: Path

    @property
    def gamma(self) -> float:
        return float(self.values["gamma"])

    @property
    def failure_avoidance_bound(self) -> float:
        v = self.values
        costs = sum(v["family_weights"][name] for name in FAMILIES[1:])
        costs += v["time_cost_per_s"] * v["family_weights"]["task_progress"]
        # Infinite discounted future is an upper bound on the finite task.
        # This only bounds costs avoidable by ending now. It is NOT a theorem
        # ordering all successful/failed trajectories of different durations:
        # delaying a discounted negative terminal event can still improve return.
        return costs / v["decision_hz"] / (1-self.gamma) + v["potential_weight"]


def load_semantic_reward_config(path: Path | str = DEFAULT_REWARD_CONFIG) -> SemanticRewardConfig:
    selected = Path(path).resolve()
    v = yaml.safe_load(selected.read_text(encoding="utf-8"))
    if v.get("schema") != "wlr50_clean.semantic_reward.v2" or tuple(v["family_weights"]) != FAMILIES:
        raise ValueError("semantic reward requires exactly five ordered families")
    if (v["physics_hz"], v["decision_hz"], v["maximum_task_duration_s"]) != (120,15,200):
        raise ValueError("semantic task timing must be 120/15 Hz and 200 seconds")
    if not 0 < finite(v["gamma"],"gamma") < 1 or v["potential_terminal_value"] != 0:
        raise ValueError("potential requires gamma in (0,1) and zero terminal value")
    from .semantic_return_profile import reward_return_profile
    reward_return_profile(v)
    if v["time_limit_semantics"] != "finite_horizon_task_termination_no_bootstrap":
        raise ValueError("task deadline is terminal, not a rollout cutoff")
    for key in ("reference_imitation_reward", "phase_label_transition_bonus", "nonzero_residual_bonus"):
        if v.get(key) is not False:
            raise ValueError(f"{key} is forbidden")
    for key,value in v.items():
        if isinstance(value,(int,float)) and not isinstance(value,bool) and finite(value,key) < 0:
            raise ValueError(f"negative reward configuration {key}")
    if any(finite(value,key) < 0 for key,value in v["family_weights"].items()):
        raise ValueError("negative family weight")
    for key in ("attitude_scale_rad", "euler_rate_scale_rad_s", "angular_acceleration_scale_rad_s2",
                "touchdown_speed_scale_m_s", "slip_speed_scale_m_s"):
        if finite(v[key],key) <= 0:
            raise ValueError(f"{key} must be positive")
    if type(v["control_regularization_enabled"]) is not bool:
        raise ValueError("regularization switch must be boolean")
    _validate_carry_body_allowance(v)
    _validate_task_priority(v)
    config = SemanticRewardConfig(v,selected)
    if v["failure_cost"]*v["family_weights"]["task_progress"] <= config.failure_avoidance_bound:
        raise ValueError("failure event must exceed avoidable future costs and potential")
    return config


def _square_cost(values: Sequence[float], scales: Sequence[float]) -> float:
    return min(1.0, sum(min(1.0,(x/s)**2) for x,s in zip(values,scales,strict=True))/max(1,len(values)))


@dataclass(frozen=True)
class SemanticRewardSample:
    previous: SemanticObservationFrame
    current: SemanticObservationFrame
    dt_s: float
    nominal: tuple[float, ...]
    previous_nominal: tuple[float, ...]
    residual: tuple[float, ...]
    previous_residual: tuple[float, ...]
    actual_drive: tuple[float, ...]
    previous_actual_drive: tuple[float, ...]
    previous_previous_actual_drive: tuple[float, ...]
    residual_caps: tuple[float, ...]


class SemanticRewardCalculator:
    def __init__(self, config: SemanticRewardConfig | None = None):
        self.config = config or load_semantic_reward_config()
        self.reset()

    def reset(self) -> None:
        """Episode reset only; a task phase change never clears landing history."""
        self._clock_s = 0.
        self._touchdowns: dict[int,tuple[float,tuple[float,...]]] = {}
        self._command_excursion = dict.fromkeys(range(4),0.)

    def evaluate(self, previous: SemanticObservationFrame, current: SemanticObservationFrame,
                 samples: Sequence[SemanticRewardSample], *, termination_reason: str | None,
                 task_success: bool) -> dict[str, Any]:
        if not samples:
            raise ValueError("reward requires an actually executed physical tick")
        if task_success != (termination_reason == "SUCCESS"):
            raise ValueError("success event and termination reason disagree")
        v = self.config.values
        costs = {name:0.0 for name in FAMILIES[1:]}
        diagnostics: dict[str,float] = {}
        task_quality = v.get("objective_profile") == TASK_CONDITIONED_QUALITY_OBJECTIVE
        front_quality = v.get("objective_profile") == FRONT_QUALITY_OBJECTIVE or task_quality
        front_sample_audit = []
        geometry_sample_audit = []
        total_dt = 0.0
        for sample in samples:
            dt = finite(sample.dt_s,"reward dt")
            if not 0 < dt <= 1/120+1e-9:
                raise ValueError("reward samples must be actual 120 Hz ticks")
            role_motion_weighting = sample.current.task.get("transfer_roles_version") == ROLE_TRANSFER_VERSION
            if role_motion_weighting:
                role_fraction = finite(sample.current.task.get("physical_transfer_fraction"), "role transfer fraction")
                if not 0. <= role_fraction <= 1.:
                    raise ValueError("role transfer fraction must be in [0,1]")
            total_dt += dt
            self._clock_s += dt
            metrics = sample.current.metrics
            transfer = sample.current.task["substage"] == "TRANSFER"
            transfer_fraction = (max(0.,min(1.,float(sample.current.task.get("physical_transfer_fraction",float(transfer)))))
                                 if v.get("physical_transfer_weighting") else float(transfer))
            if role_motion_weighting:
                transfer_fraction = role_fraction
            body_transfer_fraction = transfer_fraction
            if v.get("carry_body_allowance") == CARRY_BODY_ALLOWANCE_MODE:
                body_transfer_fraction = max(body_transfer_fraction,
                    _current_functional_body_allowance(sample.current.task, v["capture_settle_window_s"]))
            attitude = _square_cost(metrics["rpy"][:2], (v["attitude_scale_rad"],)*2)
            rates = _square_cost(metrics["euler_roll_pitch_rate"], (v["euler_rate_scale_rad_s"],)*2)
            acceleration = _square_cost(metrics["body_angular_acceleration"], (v["angular_acceleration_scale_rad_s2"],)*3)
            motion_weight = 1.-(1.-v["transfer_attitude_weight"])*body_transfer_fraction
            attitude *= motion_weight
            if role_motion_weighting:
                # The versioned measured role window, not AIR/load or a phase
                # label, permits necessary transfer dynamics. Capture/settle
                # restores all three costs continuously as activity returns to 0.
                rates *= motion_weight
                acceleration *= motion_weight
            body = (attitude+rates+acceleration)/3
            if front_quality:
                # Small positive cost during preparation AND carry. Never gate
                # on success, AIR, lift loss, or a zero-cost transfer substage.
                # Do not reuse the old carry exemption or acceleration cost.
                cfg = v["front_body_quality"]
                phase = sample.current.task["stage_id"]
                eligible = phase in cfg["phases"]
                weight = 1.-(1.-cfg["transfer_weight_floor"])*transfer_fraction
                tilt_raw = _square_cost(metrics["rpy"][:2], (v["attitude_scale_rad"],)*2)
                rate_raw = _square_cost(metrics["euler_roll_pitch_rate"], (v["euler_rate_scale_rad_s"],)*2)
                body = (weight*(cfg["attitude_fraction"]*tilt_raw+cfg["rate_fraction"]*rate_raw)
                        if eligible else 0.)
                attitude = weight*tilt_raw if eligible else 0.
                rates = weight*rate_raw if eligible else 0.
                acceleration = 0.  # Measured elsewhere; not this objective.
                front_coefficient = v["family_weights"]["body_stability"]
                if task_quality:
                    front_coefficient *= v["task_space_quality"]["front_fraction"]
                if eligible:
                    beta = front_coefficient*weight
                    front_sample_audit.append({
                        "sim_time_s": metrics["sim_time_s"], "dt_s": dt, "phase": phase,
                        "substate": _front_quality_substate(sample.current.task),
                        "physical_transfer_fraction": transfer_fraction, "effective_beta_per_s": beta,
                        "roll_pitch_rad": tuple(metrics["rpy"][:2]),
                        "roll_pitch_rate_rad_s": tuple(metrics["euler_roll_pitch_rate"]),
                        "raw_tilt_cost": tilt_raw, "raw_rate_cost": rate_raw,
                        "weighted_quality_cost": front_coefficient*body*dt,
                    })
                    for key, value in (("front_beta_time_integral", beta),
                                       ("front_raw_tilt_cost", tilt_raw),
                                       ("front_raw_rate_cost", rate_raw),
                                       ("front_weighted_tilt_cost", beta*cfg["attitude_fraction"]*tilt_raw),
                                       ("front_weighted_rate_cost", beta*cfg["rate_fraction"]*rate_raw)):
                        diagnostics[key] = diagnostics.get(key, 0.)+value*dt
                if task_quality:
                    body *= v["task_space_quality"]["front_fraction"]
            if task_quality:
                space_cfg = v["task_space_quality"]
                geometry = task_space_quality_sample(sample.current.task, metrics, space_cfg)
                geometry.update(dt_s=dt, effective_beta_per_s=(v["family_weights"]["body_stability"]
                    *space_cfg["geometry_fraction"] if geometry["eligible"] else 0.),
                    weighted_geometry_cost=None, terminal_measurement_omitted=False)
                if geometry["eligible"] and not geometry["valid"]:
                    if termination_reason is None:
                        raise ValueError("task-space quality measurement unavailable: " + geometry["reason"])
                    # Preserve already classified safety/task termination and
                    # Phi=0. Unknown geometry is audited null, never good zero.
                    geometry["terminal_measurement_omitted"] = True
                elif geometry["valid"]:
                    raw_geometry = geometry["raw_geometry_cost"]
                    body += space_cfg["geometry_fraction"]*raw_geometry
                    geometry["weighted_geometry_cost"] = geometry["effective_beta_per_s"]*raw_geometry*dt
                    for key, value in (("task_space_raw_geometry_cost", raw_geometry*dt),
                                       ("task_space_weighted_geometry_cost", geometry["weighted_geometry_cost"])):
                        diagnostics[key] = diagnostics.get(key, 0.)+value
                geometry_sample_audit.append(geometry)
            contact_terms = []
            for index,(before,after) in enumerate(zip(sample.previous.metrics["wheels"],metrics["wheels"],strict=True)):
                touchdown = after["contact"] and not before["contact"]
                descent = max(0.0,-before["vertical_velocity"]-v["touchdown_speed_allowance_m_s"])
                impact = min(1.0,(descent/v["touchdown_speed_scale_m_s"])**2) if touchdown else 0.0
                load = min(1.0,after["force"]/(metrics["mass_kg"]*9.81)) if touchdown else 0.0
                # A contact loss/rebound is not forbidden: penalize only excess
                # upward speed at a just-lost contact, independent of leg identity.
                rebound = min(1.0,(max(0.0,after["vertical_velocity"]-.15)/.5)**2) if before["contact"] and not after["contact"] else 0.0
                slip = min(1.0,(max(0.0,after["slip_speed"]-v["slip_speed_allowance_m_s"])/v["slip_speed_scale_m_s"])**2) if after["contact"] else 0.0
                if v.get("contact_loss_semantics") == "recent_touchdown_without_active_command_change":
                    command_change=sum(abs(a-b) for a,b in zip(sample.actual_drive[:8],sample.previous_actual_drive[:8]))
                    command_change+=sum(abs(a-b)*60./1.8 for a,b in zip(sample.actual_drive[8:],sample.previous_actual_drive[8:]))
                    self._command_excursion[index]+=command_change
                    if touchdown:
                        self._touchdowns[index]=(self._clock_s,sample.actual_drive)
                        # This command was actually dispatched during the
                        # landing tick; do not erase active takeoff evidence
                        # merely because contact was established concurrently.
                        self._command_excursion[index]=command_change
                    last=self._touchdowns.get(index)
                    passive_recent_landing=(last is not None and self._clock_s-last[0]<=v["rebound_window_s"]
                        and self._command_excursion[index]<=v["rebound_command_motion_deg"])
                    # Departure alone proves neither intention nor rebound. A
                    # recent touchdown followed by upward reversal under an
                    # unchanged whole-body command is the conservative cost.
                    # Actuated/ambiguous departures stay diagnostic, not taxed.
                    if not passive_recent_landing: rebound=0.
                    contact_terms.append((impact*(.5+.5*load)+rebound+slip)/3)
                    diagnostics["confirmed_post_touchdown_rebound"] = diagnostics.get("confirmed_post_touchdown_rebound",0.)+rebound*dt
                    diagnostics["contact_chatter_diagnostic"] = diagnostics.get("contact_chatter_diagnostic",0.)+after["chatter"]*dt
                    diagnostics["touchdown_events"] = diagnostics.get("touchdown_events",0.)+float(touchdown)
                else:
                    contact_terms.append((impact*(.5+.5*load)+rebound+after["chatter"]+slip)/4)
            contact = sum(contact_terms)/4 * (1.-(1.-v["transfer_contact_weight"])*transfer_fraction)
            rates12 = (60.0,)*8+(1.8,)*4
            changes = {
                "nominal_first_difference":tuple((a-b)/dt for a,b in zip(sample.nominal,sample.previous_nominal,strict=True)),
                "residual_first_difference":tuple((a-b)/dt for a,b in zip(sample.residual,sample.previous_residual,strict=True)),
                "actual_drive_first_difference":tuple((a-b)/dt for a,b in zip(sample.actual_drive,sample.previous_actual_drive,strict=True)),
                "actual_drive_second_difference":tuple((a-2*b+c)/(dt*dt) for a,b,c in zip(sample.actual_drive,sample.previous_actual_drive,sample.previous_previous_actual_drive,strict=True)),
            }
            smooth_parts = {key:_square_cost(values,tuple(x*120 for x in rates12) if key.endswith("second_difference") else rates12)
                            for key,values in changes.items()}
            smooth = ((smooth_parts["actual_drive_first_difference"]+smooth_parts["actual_drive_second_difference"])/2
                      if v.get("smoothness_components")=="applied_only" else sum(smooth_parts.values())/4)
            regularization = _square_cost(sample.residual,sample.residual_caps) if v["control_regularization_enabled"] else 0.0
            for name,cost in zip(FAMILIES[1:],(body,contact,smooth,regularization),strict=True):
                costs[name] += min(1.0,max(0.0,cost))*dt
            for name,cost in {**smooth_parts,"gravity_attitude":attitude,"euler_rate":rates,
                              "angular_acceleration":acceleration,"contact_quality":contact}.items():
                diagnostics[name] = diagnostics.get(name,0.0)+cost*dt
        phi_before = finite(previous.task["task_progress_potential"],"potential before")
        phi_after = 0.0 if termination_reason else finite(current.task["task_progress_potential"],"potential after")
        potential = v["potential_weight"]*(self.config.gamma*phi_after-phi_before)
        event = v["success_reward"] if task_success else (-v["failure_cost"] if termination_reason else 0.0)
        unweighted = {"task_progress":potential+event-v["time_cost_per_s"]*total_dt,
                      **{name:-costs[name] for name in FAMILIES[1:]}}
        families = {name:unweighted[name]*v["family_weights"][name] for name in FAMILIES}
        result = {"total":sum(families.values()),"families":families,"unweighted_families":unweighted,
                "objective_profile":v.get("objective_profile", "legacy_quality_weighted"),
                "quality_epsilon":v.get("quality_epsilon"),
                "potential_before":phi_before,"potential_after":phi_after,"potential_shaping":potential,
                "terminal_event":event,"elapsed_physics_s":total_dt,"cost_components":diagnostics,
                "discount_convention":"one_gamma_per_policy_decision; short_N1_terminal_interval_has_zero_next_potential_and_no_bootstrap",
                "terminal_bootstrap_allowed":False if termination_reason else True}
        if front_quality:
            result["front_quality_sample_audit"] = front_sample_audit
            result["front_quality_semantics"] = "P01_P02_tilt_rate_only_positive_transfer_floor_v1"
        if task_quality:
            result["task_space_quality_sample_audit"] = geometry_sample_audit
            result["task_space_quality_semantics"] = "bounded_current_collider_obstacle_AABB_separation_deficit_v1"
        return result

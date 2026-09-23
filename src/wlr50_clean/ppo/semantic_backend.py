"""Explicit B/C backend, retaining the frozen A backend as a read-only control.

This class reuses physical scene/reset primitives and the one-write/one-step
actuation method. It never constructs SensorFsmController, never restores its
private lifecycle, and never calls the legacy reference entry/endpoint gates.
"""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .action_projection import ActionProjector, SafetyProjection, load_action_projection_config
from .isaac_fsm_backend import (
    IsaacFSMBackend, IsaacFSMBackendError, SensorContractFailure, PHASE_IDS,
    PHYSICS_DT_S, SETTLE_TICKS, ZERO_FULL12, _load_live_dependencies,
    _validate_reset_options, _non_negative_seed, _require_running,
    _validate_sensor_contract, _validate_rate_contract, _validate_controller_clock,
    _live_source_mapper_state, _full12, _member, _enum_value, _fall_and_explosion,
    _guard_asserted, _level_measurement, _sha256_file, _frame_is_terminal,
    build_residual_actuation_plan,
)
from .observation_schema import NonFiniteObservationError, PPOObservationFrame
from .ppo_env_adapter import AuthoritativeFrame
from .reward_terms import RewardSignals
from .termination import TerminationSignals
from .semantic_nominal_geometry import MODE as NOMINAL_GEOMETRY_MODE, FUNCTIONAL_RR_MODE, BOUNDED_RR_MODE
from .semantic_headroom import HEADROOM_MODE, validate_semantic_servo_headroom_config
from .semantic_capture_assist import CAPTURE_ASSIST_MODE, HipOnlyCaptureAssist, capture_assist_context
from .semantic_rr_capture_assist import (RR_CAPTURE_ASSIST_MODE, RR_CAPTURE_FEEDBACK_REVISION,
    RRHipOnlyCaptureAssist,
    rr_capture_assist_context)
from .semantic_rr_capture_context import rr_capture_transfer_context, rr_contact_handoff_window_s
from .semantic_rr_carry_wheel import MODE as RR_CARRY_WHEEL_MODE, build_rr_carry_wheel_context

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "ppo_semantic_v2"
DEFAULT_EXECUTION_PROFILE = CONFIG_ROOT / "execution_profile.yaml"


def load_execution_profile(path: Path | str = DEFAULT_EXECUTION_PROFILE) -> dict[str, Any]:
    profile = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema") != "wlr50_clean.semantic_execution_profile.v1":
        raise ValueError("unexpected semantic execution profile")
    if (profile["physics_hz"], profile["decision_hz"], profile["episode_timeout_s"]) != (120, 15, 200):
        raise ValueError("semantic timing must remain 120/15 Hz and a 200 second task horizon")
    if profile.get("nominal_geometry_advisory") not in (None, NOMINAL_GEOMETRY_MODE, FUNCTIONAL_RR_MODE, BOUNDED_RR_MODE):
        raise ValueError("unknown nominal geometry advisory version")
    if (profile.get("nominal_geometry_advisory") is not None
            and profile["residual"].get("composition") != "independent_post_mapper_residual.v1"):
        raise ValueError("nominal geometry requires independent post-mapper residual composition")
    headroom = profile["residual"].get("policy_headroom_mode")
    if headroom not in (None, HEADROOM_MODE):
        raise ValueError("unknown policy servo headroom mode")
    if (headroom is not None
            and profile["residual"].get("composition") != "independent_post_mapper_residual.v1"):
        raise ValueError("same-tick headroom requires independent post-mapper residual composition")
    if headroom is not None:
        base = load_action_projection_config()
        margins = yaml.safe_load(base.path.read_text(encoding="utf-8"))["joint_safety_margin_deg"]
        validate_semantic_servo_headroom_config(headroom, margins)
    reference_mode = profile["residual"].get("tracking_reference_mode")
    if reference_mode is not None:
        from .semantic_tracking_reference import MODE
        if (reference_mode != MODE or headroom != HEADROOM_MODE or
                profile["residual"].get("composition") != "independent_post_mapper_residual.v1"):
            raise ValueError("requested tracking reference requires independent residual and same-tick headroom")
    assist = profile.get("capture_assist_mode")
    if assist not in (None, CAPTURE_ASSIST_MODE):
        raise ValueError("unknown capture assist mode")
    if assist is not None and (headroom != HEADROOM_MODE or
            profile["residual"].get("composition") != "independent_post_mapper_residual.v1"):
        raise ValueError("capture assist requires unchanged post-mapper composition and physical headroom")
    rr_assist = profile.get("rr_capture_assist_mode")
    if rr_assist not in (None, RR_CAPTURE_ASSIST_MODE):
        raise ValueError("unknown RR capture assist mode")
    if rr_assist is not None and (assist != CAPTURE_ASSIST_MODE or headroom != HEADROOM_MODE):
        raise ValueError("RR capture continuation must preserve the FL and physical headroom path")
    if rr_assist is not None and profile.get("rr_capture_feedback_revision") != RR_CAPTURE_FEEDBACK_REVISION:
        raise ValueError("RR capture requires its explicit current feedback revision")
    if profile.get("rr_capture_wheel_mode", "off") not in ("off", RR_CARRY_WHEEL_MODE):
        raise ValueError("unknown declared RR support-wheel projection")
    if profile.get("rr_capture_wheel_mode", "off") != "off" and rr_assist != RR_CAPTURE_ASSIST_MODE:
        raise ValueError("RR support-wheel projection requires the observable RR continuation")
    from .semantic_rear_policy_timing import MODES as REAR_TIMING_MODES
    rear_timing = profile.get("rear_policy_timing_mode")
    if rear_timing not in (None, *REAR_TIMING_MODES):
        raise ValueError("unknown rear policy timing execution profile")
    if rear_timing and (rr_assist is not None or profile.get("nominal_geometry_advisory") is not None
                       or profile.get("rr_capture_wheel_mode", "off") != "off"):
        raise ValueError("rear policy learning forbids rear task assist, geometry and forced wheel shaping")
    return profile


def build_semantic_projector(path: Path | str = DEFAULT_EXECUTION_PROFILE) -> ActionProjector:
    profile = load_execution_profile(path)
    values = profile["residual"]
    base = load_action_projection_config()
    caps = (float(values["initial_servo_cap_deg"]),) * 8 + (float(values["initial_wheel_cap_rad_s"]),) * 4
    if any(not math.isfinite(x) or x <= 0 for x in caps):
        raise ValueError("residual engineering scales must be finite and positive")
    fractions = tuple(cap / span for cap, span in zip(caps, base.physical_residual_scale_full12, strict=True))
    if any(x > 1 for x in fractions):
        raise ValueError("residual cap exceeds physical actuator span")
    phase_fractions = {phase: fractions for phase in PHASE_IDS}
    if "phase_caps_full12" in values:
        configured = values["phase_caps_full12"]
        if tuple(configured) != PHASE_IDS:
            raise ValueError("phase caps must contain ordered P01-P13")
        previous = None
        for phase in PHASE_IDS:
            row = tuple(float(x) for x in configured[phase])
            if len(row)!=12 or any(not math.isfinite(x) or x<=0 for x in row):
                raise ValueError("phase/channel ranges must contain twelve finite positive values")
            if previous is not None and any(a<b for a,b in zip(row,previous)):
                raise ValueError("this range version cannot shrink feasible residual history at phase handoff")
            fractions = tuple(a/b for a,b in zip(row,base.physical_residual_scale_full12))
            if any(x>1 for x in fractions):
                raise ValueError("phase residual cap exceeds actuator span")
            phase_fractions[phase] = fractions
            previous = row
    config = replace(
        base, action_schema_name="wlr50_clean.semantic_residual_full12",
        action_schema_version=1, training_enabled=True, path=Path(path).resolve(),
        phase_scale_full12=phase_fractions,
        phase_mask_full12={phase: (1,) * 12 for phase in PHASE_IDS},
        servo_residual_rate_deg_s=float(values["servo_rate_deg_s"]),
        wheel_residual_rate_rad_s2=float(values["wheel_rate_rad_s2"]),
        recording_envelope_initialization_suggestion=False,
        policy_headroom_mode=values.get("policy_headroom_mode"),
    )
    # The reused projector computes legacy percentage diagnostics but never
    # applies them as a cap. No legacy mask or max-initial-scale gate is used.
    return ActionProjector(config=config)


class SemanticIsaacBackend(IsaacFSMBackend):
    """Task-semantic runtime chosen before the first control frame."""

    def __init__(self, simulation_app: Any = None, *, execution_profile: Path | str = DEFAULT_EXECUTION_PROFILE,
                 task_spec_path: Path | str = CONFIG_ROOT / "stage_task_spec.yaml",
                 controller_factory: Any = None, **kwargs: Any) -> None:
        super().__init__(simulation_app, **kwargs)
        self.execution_profile_path = Path(execution_profile).resolve()
        self.execution_profile = load_execution_profile(execution_profile)
        composition = self.execution_profile["residual"].get("composition")
        if composition not in (None, "independent_post_mapper_residual.v1"):
            raise ValueError("unknown semantic residual composition")
        self._independent_policy_residual = composition is not None
        self._policy_headroom_mode = self.execution_profile["residual"].get("policy_headroom_mode")
        self._tracking_reference_mode = self.execution_profile["residual"].get("tracking_reference_mode")
        self._capture_assist = HipOnlyCaptureAssist() if self.execution_profile.get("capture_assist_mode") else None
        self._rr_capture_assist = RRHipOnlyCaptureAssist() if self.execution_profile.get("rr_capture_assist_mode") else None
        self._rr_carry_wheel_mode = self.execution_profile.get("rr_capture_wheel_mode", "off")
        self.task_spec_path = Path(task_spec_path).resolve()
        rr_task_spec = yaml.safe_load(self.task_spec_path.read_text(encoding="utf-8"))
        self._rear_policy_timing_mode = self.execution_profile.get("rear_policy_timing_mode")
        if self._rear_policy_timing_mode != rr_task_spec.get("nominal", {}).get("rear_policy_timing"):
            raise ValueError("rear timing task and execution profile must agree")
        self._rr_support_spec = rr_task_spec["support"]
        self._rr_contact_handoff_window_s = rr_contact_handoff_window_s(rr_task_spec)
        self._physical_acceptance_version = yaml.safe_load(self.task_spec_path.read_text(encoding="utf-8")).get("physical_acceptance_version")
        self._nominal_geometry_mode = self.execution_profile.get("nominal_geometry_advisory")
        self._nominal_geometry_margin_m = None
        self._functional_geometry_parameters = {}
        if self._nominal_geometry_mode is not None:
            task_spec = yaml.safe_load(self.task_spec_path.read_text(encoding="utf-8"))
            self._nominal_geometry_margin_m = float(task_spec["geometry"]["airborne_clearance_above_top_m"])
            if not math.isfinite(self._nominal_geometry_margin_m) or self._nominal_geometry_margin_m <= 0.:
                raise ValueError("nominal geometry requires the existing positive physical clearance margin")
            if self._nominal_geometry_mode in (FUNCTIONAL_RR_MODE, BOUNDED_RR_MODE):
                if task_spec.get("p09_lift_semantics") not in (
                        "functional_lift_edge_v2", "functional_free_air_lift_v3"):
                    raise ValueError("functional RR geometry requires matching current-lift semantics")
                self._functional_geometry_parameters = dict(mode=self._nominal_geometry_mode,
                    minimum_lift_gain_m=float(task_spec["history"]["minimum_lift_gain_m"]),
                    workspace_min_m=float(task_spec["geometry"]["workspace_min_m"]))
                if self._nominal_geometry_mode == BOUNDED_RR_MODE:
                    self._functional_geometry_parameters["top_gap_min_m"] = float(task_spec["geometry"]["top_gap_min_m"])
        self._semantic_controller_factory = controller_factory
        self._semantic_actuation_plan = None
        self._level_fixed = tuple(float(x) for x in self.execution_profile["level_reference_orientation_wxyz"])

    def reset(self, *, seed: int, options: Mapping[str, Any]) -> AuthoritativeFrame:
        self._reset_generation += 1
        self._poison_episode_state_for_reset(clear_evidence=True)
        if self.execution_profile.get("capture_assist_mode"):
            self._capture_assist = HipOnlyCaptureAssist()
        if self.execution_profile.get("rr_capture_assist_mode"):
            self._rr_capture_assist = RRHipOnlyCaptureAssist()
        reset_seed = _non_negative_seed(seed)
        reset_options = dict(options)
        _validate_reset_options(reset_options)
        if reset_options.get("training_phase_snapshot") not in (None, "P01"):
            raise IsaacFSMBackendError("semantic snapshots require task-state validation; use natural P01 until available")
        requested = str(reset_options.get("start_phase", "P01"))
        if requested != "P01":
            raise IsaacFSMBackendError("initial semantic runtime only accepts natural P01; no historical entry reconstruction")
        dependencies = self._dependencies or _load_live_dependencies()
        self._dependencies = dependencies
        try:
            if self._scene is None:
                self._scene = dependencies.create_scene(
                    simulation_app=self.simulation_app,
                    before_reset=lambda sim, robot: dependencies.create_sensing_backends(sim=sim, robot=robot),
                )
            scene = self._scene
            reset_writes = dict(dependencies.reset_scene(scene, self._canonical_reset_state))
            backends = getattr(scene, "instrumentation", None)
            contact = getattr(backends, "contact_backend", None)
            if backends is None or contact is None or not bool(getattr(contact, "initialized", False)):
                raise SensorContractFailure("exact contact sensor bank unavailable")
            adapter = dependencies.adapter_from_scene(scene)
            if self._canonical_reset_state is None:
                self._canonical_reset_state = dependencies.capture_reset_state(scene)
            # Keep the same physical settle and unchanged mature mapper.
            # Measured post-settle state is accepted as a set, not byte equality.
            ack = None
            for tick in range(SETTLE_TICKS):
                _require_running(scene, "semantic reset settle")
                ack = self._atomic_apply(adapter, ZERO_FULL12, physics_tick=tick,
                                         tracking_servo_names=(), drive_feedback_bias_full12=ZERO_FULL12)
                scene.sim.step(render=False)
                adapter.update_readback()
            adapter.verify_authoritative_servo_limits_adopted()
            reader = dependencies.reader_from_scene(scene, adapter, backends)
            if self._physical_acceptance_version == "all_stage_v1":
                from .semantic_physical_sensing import SemanticSensorReader
                reader = SemanticSensorReader.from_live_scene(scene, adapter, backends=backends,
                    physical_acceptance_version="all_stage_v1")
            observation = reader.read(physics_tick=0, simulation_time_s=0.0, commanded_full12=ack["drive_target_full12"])
            _validate_sensor_contract(observation, dependencies.expected_contact_bodies, require_finite=True)
            if self._semantic_controller_factory is None:
                from .semantic_supervisor import SemanticControllerAdapter
                controller = SemanticControllerAdapter.from_paths(
                    self.fsm_path, self.motion_contract_path,
                    task_spec_path=self.task_spec_path,
                )
            else:
                controller = self._semantic_controller_factory(self.fsm_path, self.motion_contract_path)
            _validate_rate_contract(adapter, controller)
            frame = controller.step(observation, sim_time_s=0.0)
            _validate_controller_clock(frame, physics_tick=0, sim_time_s=0.0)
            self._adapter, self._reader, self._controller = adapter, reader, controller
            self._raw_observation, self._controller_frame = observation, frame
            self._last_atomic_ack = ack
            self._level_reference_orientation = self._level_fixed
            self._done = False
            self._snapshot_restoration = {
                "mode": "semantic_natural_P01", "historical_state_equality_required": False,
                "requested_phase": "P01", "reset_only_state_writes": dict(reset_writes),
                "policy_credit_excludes_settle": True,
            }
            self._reset_metadata = self._make_reset_metadata(observation, seed=reset_seed, options=reset_options, reset_writes=reset_writes)
            self._reset_metadata.update({
                "execution_mode": "semantic_B_or_C", "supervisor_schema": "task_semantic_v2",
                "controller_hash": _sha256_file(Path(__file__).with_name("semantic_supervisor.py")),
                "stage_task_spec_hash": _sha256_file(self.task_spec_path),
                "execution_profile_hash": _sha256_file(self.execution_profile_path),
                "effective_phase_entry_semantics": "current_physical_validity_no_reference_entry_gate",
                "level_calibration_window_s": 0.0, "level_calibration_sample_count": 0,
                "level_reference_source": "versioned_fixed_chassis_axes",
                "reset_generation_commit": "semantic_physical_validity",
            })
            if self._capture_assist is not None:
                self._reset_metadata.update(capture_assist_mode=CAPTURE_ASSIST_MODE,
                    capture_assist_applies_identically_B_C_train_det_stoch=True,
                    capture_assist_policy_sample_log_probability_unchanged=True)
            if self._rr_capture_assist is not None:
                self._reset_metadata.update(rr_capture_assist_mode=RR_CAPTURE_ASSIST_MODE,
                    rr_capture_feedback_revision=RR_CAPTURE_FEEDBACK_REVISION,
                    rr_capture_assist_applies_identically_B_C_train_det_stoch=True,
                    rr_capture_assist_is_policy_learning=False)
            result = self._build_authoritative_frame(observation, frame, previous_frame=None)
            if _frame_is_terminal(result):
                raise SensorContractFailure("semantic reset produced physically invalid/terminal initial state")
            self._reset_count += 1
            self._authoritative_frame = result
            self._committed_reset_generation = self._reset_generation
            return result
        except Exception:
            self._poison_episode_state_for_reset(clear_evidence=False)
            raise

    def step_physics(self, applied_action_full12):
        # Capture the same pure plan as the inherited one-write/one-step path,
        # before its _atomic_apply seam; never alter the mapper's nominal input.
        if not self._independent_policy_residual:
            return super().step_physics(applied_action_full12)
        self._require_committed_reset_generation("step_physics")
        source = self._controller_frame
        if source is None:
            raise IsaacFSMBackendError("reset must precede semantic step_physics")
        if self._semantic_actuation_plan is not None:
            raise IsaacFSMBackendError("nested semantic physics dispatch")
        self._semantic_actuation_plan = build_residual_actuation_plan(
            applied_action_full12, frozen_nominal_full12=source.full12,
            drive_feedback_bias_full12=source.drive_feedback_bias_full12,
            normal_drive_bias_full12=source.normal_drive_bias_full12)
        try:
            return super().step_physics(applied_action_full12)
        finally:
            self._semantic_actuation_plan = None

    def _atomic_apply(self, adapter, command, *, physics_tick, tracking_servo_names,
                      drive_feedback_bias_full12):
        plan = self._semantic_actuation_plan
        if plan is not None:
            from .semantic_residual_adapter import SemanticActuationDispatch
            geometry = None
            if getattr(self, "_nominal_geometry_mode", None) is not None:
                controller = self._controller
                source = self._controller_frame
                prefix_mode = getattr(controller, "mode", None)
                handoff_tick = getattr(controller, "_handoff_tick", None)
                # Frozen teacher, bias takeover and the exact receipt handoff
                # stay untouched. READY continuation uses the same B/C path.
                if ((prefix_mode is None or prefix_mode == "READY")
                        and source.physics_tick != handoff_tick):
                    from .semantic_nominal_geometry import capture_nominal_geometry_context
                    geometry_options = dict(getattr(self, "_functional_geometry_parameters", {}))
                    if self._nominal_geometry_mode == BOUNDED_RR_MODE and source.state_id == "P09":
                        provider = self._reader.geometry_backend.provider
                        geometry_options["collider_local_points"] = provider._body_local_points.get("rear_right_wheel")
                    geometry = capture_nominal_geometry_context(
                        adapter=adapter, observation=self._raw_observation,
                        source_frame=source, task_snapshot=controller.task_snapshot,
                        clearance_margin_m=self._nominal_geometry_margin_m,
                        physics_tick=physics_tick,
                        **geometry_options)
            assist = getattr(self, "_capture_assist", None)
            assist_context = None
            # The explicitly uncredited frozen-A reset prefix is not B/C.
            # Keep its teacher/takeover receipts unchanged; every READY B/C
            # dispatch uses the same assist rule irrespective of residual.
            if getattr(self._controller, "mode", None) in ("TEACHER", "TAKEOVER"):
                assist = None
            if assist is not None:
                active_controller = getattr(self._controller, "_semantic", None) or self._controller
                assist_context = capture_assist_context(task=self._controller.task_snapshot,
                    observation=self._raw_observation, source_frame=self._controller_frame,
                    previous_ack=adapter.last_ack or {},
                    nominal_provider=getattr(active_controller, "nominal_provider", None), physics_tick=physics_tick)
            rr_assist = getattr(self, "_rr_capture_assist", None)
            rr_context = None
            if getattr(self._controller, "mode", None) in ("TEACHER", "TAKEOVER"):
                rr_assist = None
            if rr_assist is not None:
                active_controller = getattr(self._controller, "_semantic", None) or self._controller
                rr_context = rr_capture_assist_context(task=self._controller.task_snapshot,
                    observation=self._raw_observation, source_frame=self._controller_frame,
                    previous_ack=adapter.last_ack or {}, physics_tick=physics_tick,
                    support_spec=active_controller.supervisor.spec["support"])
            wheel_context = self._rr_carry_pre_dispatch_context(physics_tick)
            adapter = SemanticActuationDispatch(adapter, plan, nominal_geometry_context=geometry,
                policy_headroom_mode=getattr(self, "_policy_headroom_mode", None),
                tracking_reference_mode=getattr(self, "_tracking_reference_mode", None),
                tracking_reference_bootstrap_tick=SETTLE_TICKS + self._reset_prime_tick_count,
                capture_assist=assist, capture_assist_context=assist_context,
                rr_capture_assist=rr_assist, rr_capture_assist_context=rr_context,
                rr_carry_wheel_context=wheel_context)
        ack = super()._atomic_apply(adapter, command, physics_tick=physics_tick,
            tracking_servo_names=tracking_servo_names,
            drive_feedback_bias_full12=drive_feedback_bias_full12)
        if plan is not None and getattr(self, "_rr_capture_assist", None) is not None:
            active_controller = getattr(self._controller, "_semantic", None) or self._controller
            if "rr_capture_assist_evidence" in ack:
                active_controller.supervisor.rr_capture_feedback = {
                    "episode_observation_tick": self._controller_frame.physics_tick+1,
                    "dispatch_physics_tick": physics_tick,
                    "state": self._rr_capture_assist.snapshot(),
                }
        return ack

    def _rr_carry_pre_dispatch_context(self, physics_tick):
        """Read-only shared input, independently captured before native audit/write."""
        if (getattr(self, "_rr_carry_wheel_mode", "off") == "off"
                or getattr(self._controller, "mode", None) in ("TEACHER", "TAKEOVER")):
            return None
        active = getattr(self._controller, "_semantic", None) or self._controller
        return build_rr_carry_wheel_context(task=self._controller.task_snapshot,
            observation=self._raw_observation, source_frame=self._controller_frame,
            nominal_provider=active.nominal_provider, support_spec=self._rr_support_spec,
            physics_tick=physics_tick, source_ack=self._adapter.last_ack,
            previous_write_count=self._adapter.write_count, mode=self._rr_carry_wheel_mode,
            wheel_rate_rad_s2=self.execution_profile["residual"]["wheel_rate_rad_s2"])

    def _termination_signals(self, observation: Any, controller_frame: Any):
        result = _enum_value(_member(_member(controller_frame, "termination"), "result"))
        task = getattr(self._controller, "task_snapshot", {})
        source = task.get("termination_source") if isinstance(task, Mapping) else None
        fall, explosion, physics = _fall_and_explosion(observation)
        self._body_collision_seen |= bool(_member(_member(observation, "body_collision"), "detected", False)) or result == "TASK_FAILURE_BODY_COLLISION"
        self._wheel_only_seen |= result == "TASK_FAILURE_WHEEL_ONLY_CLIMB"
        self._nan_inf_seen |= not bool(_member(observation, "all_finite", False))
        self._joint_limit_seen |= _guard_asserted(observation, "joint_hard_limit_violation")
        self._fall_seen |= fall
        self._physics_explosion_seen |= explosion
        unsafe = any((self._body_collision_seen, self._wheel_only_seen, self._nan_inf_seen,
                      self._joint_limit_seen, self._fall_seen, self._physics_explosion_seen))
        if result in ("INFRASTRUCTURE_ERROR", "VIDEO_OR_ARTIFACT_ERROR"):
            raise IsaacFSMBackendError(f"semantic infrastructure fault: {controller_frame.termination}")
        signals = TerminationSignals(
            success=result == "SUCCESS" and not unsafe,
            body_collision=self._body_collision_seen, wheel_only_climb=self._wheel_only_seen,
            nan_inf=self._nan_inf_seen, hard_joint_limit=self._joint_limit_seen,
            fall=self._fall_seen, physics_explosion=self._physics_explosion_seen,
            timeout=result == "INCOMPLETE_CONTROLLER_BLOCKED" or float(controller_frame.sim_time_s) >= 200.0,
            reference_conformance_outside_30pct=False,
        )
        reasons = [name for name, active in (
            ("NAN_INF", signals.nan_inf), ("PHYSICS_EXPLOSION", signals.physics_explosion),
            ("BODY_COLLISION", signals.body_collision), ("WHEEL_ONLY_CLIMB", signals.wheel_only_climb),
            ("FALL", signals.fall), ("HARD_JOINT_LIMIT", signals.hard_joint_limit),
            ("SUCCESS", signals.success), (source or "TASK_DEADLINE_OR_STALL", signals.timeout),
        ) if active]
        return signals, {
            "schema": "wlr50_clean.semantic_termination.v1", "controller_result": result,
            "controller_reason": _member(_member(controller_frame, "termination"), "reason"),
            "primary_source": reasons[0] if reasons else None, "active_sources": reasons,
            "physics_guard_values": physics, "timeout_is_task_terminal": True,
            "termination_source": source,
            "finite_mdp_terminal": bool(result), "external_truncation": False,
            "legacy_wheel_only_guard_diagnostic": _guard_asserted(observation, "wheel_only_climb_detected"),
        }

    def _build_authoritative_frame(self, observation: Any, controller_frame: Any, *, previous_frame: AuthoritativeFrame | None):
        controller = self._controller
        if controller is None:
            raise IsaacFSMBackendError("semantic controller unavailable")
        state_id = str(controller_frame.state_id)
        progress = float(controller.task_progress)
        nominal = _full12(controller_frame.full12, "semantic nominal")
        termination, details = self._termination_signals(observation, controller_frame)
        level = _level_measurement(observation, self._level_fixed)
        self._level_calibration = level
        try:
            transport = PPOObservationFrame.from_live_observation(
                observation, state_id=state_id, macro_phase=int(state_id[1:]),
                phase_progress=progress, previous_action_full12=self._previous_action_full12,
            )
            self._last_valid_actor_observation = transport
        except NonFiniteObservationError:
            if not termination.nan_inf or self._last_valid_actor_observation is None:
                raise
            transport = replace(self._last_valid_actor_observation, state_id=state_id,
                                macro_phase=int(state_id[1:]), phase_progress=progress)
        # This transport DTO is never the semantic actor schema. The actor
        # encoder consumes the real observation and task/mapper summaries below.
        ack = dict(self._last_atomic_ack or {})
        mapper = _live_source_mapper_state(self._adapter, source_control_physics_tick=controller_frame.physics_tick)
        info = {
            **self._reset_metadata, "raw_observation": observation,
            "raw_controller_frame": controller_frame, "atomic_ack": ack,
            "semantic_task": dict(controller.task_snapshot),
            "level_calibration": level, "mapper_state_summary": mapper,
            "mapped_nominal_full12": list(ack.get("native_drive_target_full12", ZERO_FULL12)),
            "mapped_nominal_time_semantics": "previous_dispatch_native_before_post_mapper_bias",
            "drive_target_full12": list(ack.get("drive_target_full12", ZERO_FULL12)),
            "controller_lifecycle": _enum_value(controller_frame.lifecycle),
            "controller_task_result": _enum_value(_member(controller_frame.termination, "result")),
            "controller_termination": controller_frame.termination, "termination_mapping": details,
            "in_episode_root_pose_writes": 0, "in_episode_root_velocity_writes": 0,
            "in_episode_force_or_impulse_writes": 0, "in_episode_gravity_writes": 0,
            "recording_accesses": 0,
        }
        if "actuator_target_effect_audit" in ack:
            info["actuator_target_effect_audit"] = ack["actuator_target_effect_audit"]
        if getattr(self, "_capture_assist", None) is not None:
            info["capture_assist"] = self._capture_assist.snapshot()
            info["capture_assist_evidence"] = ack.get("capture_assist_evidence")
        rear_learning = bool(getattr(self, "_rear_policy_timing_mode", None))
        if getattr(self, "_rr_capture_assist", None) is not None or rear_learning:
            # Preserve the old 410-feature prefix with a disabled WAIT snapshot.
            # This object is metadata only: never advance it or pass it to the
            # actuation dispatch. No fabricated contact or assist owner exists.
            info["rr_capture_assist"] = (RRHipOnlyCaptureAssist().snapshot() if rear_learning
                                         else self._rr_capture_assist.snapshot())
            info["rr_capture_assist_evidence"] = ack.get("rr_capture_assist_evidence")
            from .semantic_rr_capture_profile import RR_TASK_FIELDS
            rear_context = rr_capture_transfer_context(
                task=controller.task_snapshot, observation=observation,
                support_spec=self._rr_support_spec,
                assist_snapshot=None if rear_learning else info["rr_capture_assist"],
                wheel_mode="off",
                contact_handoff_window_s=self._rr_contact_handoff_window_s)
            wheel_context = self._rr_carry_pre_dispatch_context(self._adapter._last_physics_tick + 1)
            if wheel_context is not None:
                rear_context["fl_wheel_guidance_active"] = wheel_context["envelope_active"]
                info["rr_carry_wheel_context"] = wheel_context
                info["rr_carry_wheel_evidence"] = ack.get("rr_carry_wheel_evidence")
            info["rr_capture_transfer_context"] = {key: rear_context[key] for key in RR_TASK_FIELDS}
            info["rr_capture_transfer_diagnostics"] = rear_context
        if rear_learning:
            from .semantic_rear_policy_timing import public_timing
            active = getattr(controller, "_semantic", None) or controller
            provider = getattr(active, "nominal_provider", None)
            info["rear_policy_timing"] = (provider.rear_policy_timing(controller.task_snapshot)
                if provider is not None else public_timing(controller.task_snapshot, [], self._rr_support_spec, 120.,
                    mode=self._rear_policy_timing_mode))
            info["rear_task_assist_disabled"] = True
        unsafe = any((termination.body_collision, termination.wheel_only_climb,
                      termination.fall, termination.nan_inf, termination.hard_joint_limit, termination.physics_explosion))
        return AuthoritativeFrame(
            physics_tick=controller_frame.physics_tick, sim_time_s=controller_frame.sim_time_s,
            state_id=state_id, macro_phase=int(state_id[1:]), phase_progress=progress,
            observation=transport, nominal_action_full12=nominal,
            reference_action_full12=nominal, reference_delta_full12=ZERO_FULL12,
            action_mask_full12=(1,) * 12, reward_signals=RewardSignals(),
            termination_signals=termination,
            safety_projection=SafetyProjection(
                residual_enabled=not unsafe, force_wheels_zero=unsafe,
                body_collision_detected=termination.body_collision,
                wheel_only_climb_detected=termination.wheel_only_climb,
                reason=details["primary_source"] if unsafe else None,
            ), info=info,
        )

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
)
from .observation_schema import NonFiniteObservationError, PPOObservationFrame
from .ppo_env_adapter import AuthoritativeFrame
from .reward_terms import RewardSignals
from .termination import TerminationSignals

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "ppo_semantic_v2"
DEFAULT_EXECUTION_PROFILE = CONFIG_ROOT / "execution_profile.yaml"


def load_execution_profile(path: Path | str = DEFAULT_EXECUTION_PROFILE) -> dict[str, Any]:
    profile = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if profile.get("schema") != "wlr50_clean.semantic_execution_profile.v1":
        raise ValueError("unexpected semantic execution profile")
    if (profile["physics_hz"], profile["decision_hz"], profile["episode_timeout_s"]) != (120, 15, 200):
        raise ValueError("semantic timing must remain 120/15 Hz and a 200 second task horizon")
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
    config = replace(
        base, action_schema_name="wlr50_clean.semantic_residual_full12",
        action_schema_version=1, training_enabled=True, path=Path(path).resolve(),
        phase_scale_full12={phase: fractions for phase in PHASE_IDS},
        phase_mask_full12={phase: (1,) * 12 for phase in PHASE_IDS},
        servo_residual_rate_deg_s=float(values["servo_rate_deg_s"]),
        wheel_residual_rate_rad_s2=float(values["wheel_rate_rad_s2"]),
        recording_envelope_initialization_suggestion=False,
    )
    # The reused projector computes legacy percentage diagnostics but never
    # applies them as a cap. No legacy mask or max-initial-scale gate is used.
    return ActionProjector(config=config)


class SemanticIsaacBackend(IsaacFSMBackend):
    """Task-semantic runtime chosen before the first control frame."""

    def __init__(self, simulation_app: Any = None, *, execution_profile: Path | str = DEFAULT_EXECUTION_PROFILE,
                 controller_factory: Any = None, **kwargs: Any) -> None:
        super().__init__(simulation_app, **kwargs)
        self.execution_profile_path = Path(execution_profile).resolve()
        self.execution_profile = load_execution_profile(execution_profile)
        self._semantic_controller_factory = controller_factory
        self._level_fixed = tuple(float(x) for x in self.execution_profile["level_reference_orientation_wxyz"])

    def reset(self, *, seed: int, options: Mapping[str, Any]) -> AuthoritativeFrame:
        self._reset_generation += 1
        self._poison_episode_state_for_reset(clear_evidence=True)
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
            observation = reader.read(physics_tick=0, simulation_time_s=0.0, commanded_full12=ack["drive_target_full12"])
            _validate_sensor_contract(observation, dependencies.expected_contact_bodies, require_finite=True)
            if self._semantic_controller_factory is None:
                from .semantic_supervisor import SemanticControllerAdapter
                controller = SemanticControllerAdapter.from_paths(
                    self.fsm_path, self.motion_contract_path,
                    task_spec_path=CONFIG_ROOT / "stage_task_spec.yaml",
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
                "stage_task_spec_hash": _sha256_file(CONFIG_ROOT / "stage_task_spec.yaml"),
                "execution_profile_hash": _sha256_file(self.execution_profile_path),
                "effective_phase_entry_semantics": "current_physical_validity_no_reference_entry_gate",
                "level_calibration_window_s": 0.0, "level_calibration_sample_count": 0,
                "level_reference_source": "versioned_fixed_chassis_axes",
                "reset_generation_commit": "semantic_physical_validity",
            })
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

    def _termination_signals(self, observation: Any, controller_frame: Any):
        result = _enum_value(_member(_member(controller_frame, "termination"), "result"))
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
            ("SUCCESS", signals.success), ("TASK_DEADLINE_OR_STALL", signals.timeout),
        ) if active]
        return signals, {
            "schema": "wlr50_clean.semantic_termination.v1", "controller_result": result,
            "controller_reason": _member(_member(controller_frame, "termination"), "reason"),
            "primary_source": reasons[0] if reasons else None, "active_sources": reasons,
            "physics_guard_values": physics, "timeout_is_task_terminal": True,
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

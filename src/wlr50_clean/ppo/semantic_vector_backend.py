"""Eight semantic clones, one physical advance, no legacy control gates.

This module reuses frozen scene/actuator primitives, not the legacy controller.
It supports only synchronous P01 reset. Never call row semantic reset/step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.sensing.contact_classifier import SENSED_BODIES
from .actuator_target_effect import build_actuator_target_effect_audit, actuator_target_audit_request
from .isaac_fsm_backend import (
    SETTLE_TICKS, PHYSICS_DT_S, _validate_controller_clock,
    _validate_sensor_contract, _validate_rate_contract, build_residual_actuation_plan,
)
from .semantic_backend import (
    CONFIG_ROOT, DEFAULT_EXECUTION_PROFILE, SemanticIsaacBackend, load_execution_profile,
)
from .semantic_env import _terminal_reason
from .semantic_supervisor import SemanticControllerAdapter
from .vectorized_isaac_backend import (
    VectorizedIsaacFSMBackend, VectorizedIsaacBackendError, _BatchedCommandAdapter,
    _BatchAck, _RowRobotData, ZERO12,
)


class _RowTargetRobot:
    """Read-only one-row tensor facade; no write method is exposed."""
    def __init__(self, robot: Any, row: int, origin: Any, count: int):
        self._robot, self._row, self._count = robot, row, count
        self.joint_names = tuple(robot.joint_names)
        self.body_names = tuple(robot.body_names)
        self.data = _RowRobotData(robot.data, row, origin, count)

    def __getattr__(self, name: str) -> Any:
        if name not in ("_joint_pos_target_sim", "_joint_vel_target_sim"):
            raise AttributeError(name)
        tensor = getattr(self._robot, name)
        if tensor.ndim != 2 or tensor.shape[0] != self._count:
            raise VectorizedIsaacBackendError("dispatch target tensor lost its row dimension")
        return tensor[self._row:self._row+1]


class _RowMapperAdapter:
    """Existing audit/source-mapper helpers inspect this row without advancing it."""
    def __init__(self, owner: Any, row: int):
        self.robot = _RowTargetRobot(owner.robot, row, owner.origins[row], owner.num_envs)
        self.joint_map = owner.joint_map
        self.standing_pose_deg = owner.standing_pose_deg[row]
        self.servo_target_mapper = owner.mappers[row]
        self._final_drive_servo_deg = owner._final_drive[row]


class _SemanticBatchedCommandAdapter(_BatchedCommandAdapter):
    def __init__(self, robot: Any, count: int, origins: Any):
        super().__init__(robot, count)
        self.origins = origins

    def row_adapter(self, row: int) -> _RowMapperAdapter:
        if not 0 <= row < self.num_envs:
            raise ValueError("invalid row")
        return _RowMapperAdapter(self, row)

    def apply_batch(self, commands, *, physics_tick, tracking_servo_names, drive_feedback_bias_full12):
        # Exactly one inherited mapper.advance per row and one write_data_to_sim.
        batch = super().apply_batch(commands, physics_tick=physics_tick,
            tracking_servo_names=tracking_servo_names,
            drive_feedback_bias_full12=drive_feedback_bias_full12)
        rows = []
        for index, ack in enumerate(batch.rows):
            mapper = self.mappers[index]
            native = tuple(mapper._applied[name] for name in SERVO_ORDER) + tuple(commands[index][8:])
            rows.append({**ack, "native_drive_target_full12":list(native),
                "drive_feedback_final_slew_limit_deg_per_tick":mapper.maximum_delta_deg,
                "env_index":index})
        return _BatchAck(tuple(rows), batch.articulation_writes_this_call, batch.physics_tick)


class SemanticVectorIsaacBackend(VectorizedIsaacFSMBackend):
    """Eight isolated semantic B/C rows; global hard reset only at episode barrier."""
    def __init__(self, simulation_app: Any, *, num_envs: int = 8,
                 execution_profile: Path | str = DEFAULT_EXECUTION_PROFILE, **kwargs):
        if type(num_envs) is not int or num_envs != 8:
            raise ValueError("first semantic vector revision supports only N=8")
        self.execution_profile_path = Path(execution_profile).resolve()
        self.execution_profile = load_execution_profile(self.execution_profile_path)
        if self.execution_profile["residual"].get("policy_headroom_mode") is not None:
            raise ValueError("same-tick policy servo headroom currently requires the audited N=1 dispatch")
        self._policy_requests = (None,)*8
        self._row_adapters = ()
        # Creates only the one physical scene/contact bank, not legacy controllers.
        super().__init__(simulation_app, num_envs=num_envs, **kwargs)

    def reset_all(self, *, seeds: Sequence[int] | None = None,
                  options: Sequence[Mapping[str, Any]] | None = None):
        seed_rows = tuple(range(1001,1009)) if seeds is None else tuple(seeds)
        option_rows = ({},)*8 if options is None else tuple(options)
        if len(seed_rows)!=8 or any(type(seed) is not int or seed<0 for seed in seed_rows):
            raise ValueError("exactly eight nonnegative integer seeds required")
        if len(option_rows)!=8 or any(dict(option) for option in option_rows):
            raise ValueError("semantic vector v1 supports natural P01 only, no snapshots")
        if self._reset_transaction_poisoned:
            raise VectorizedIsaacBackendError("previous reset transaction failed")
        self._policy_requests = (None,)*8
        try:
            lifecycle = self._prepare_physical_reset()
            self.command_adapter = _SemanticBatchedCommandAdapter(self.robot,8,self.scene.env_origins)
            self.contact_bank.reset()
            zeros = (ZERO12,)*8
            ack = None
            for _ in range(SETTLE_TICKS):
                ack = self.command_adapter.apply_batch(zeros,
                    physics_tick=self.global_physics_step_count, tracking_servo_names=((),)*8,
                    drive_feedback_bias_full12=zeros)
                self._advance_global_physics()
            if ack is None:
                raise VectorizedIsaacBackendError("settle produced no atomic dispatch")
            self.readers = self._make_readers()
            self.controllers = tuple(SemanticControllerAdapter.from_paths(
                self.fsm_path,self.motion_contract_path,
                task_spec_path=CONFIG_ROOT/"stage_task_spec.yaml") for _ in range(8))
            self._assert_independent_python_state()
            self._row_adapters = tuple(self.command_adapter.row_adapter(row) for row in range(8))
            self._semantics = tuple(SemanticIsaacBackend(
                execution_profile=self.execution_profile_path,
                audit_actuator_target_effect=False) for _ in range(8))
            if len({id(x.supervisor.evaluator) for x in self.controllers})!=8:
                raise VectorizedIsaacBackendError("semantic evaluator history is shared")
            frames, controller_frames = [],[]
            for row,(reader,controller,semantic) in enumerate(zip(
                    self.readers,self.controllers,self._semantics,strict=True)):
                observation = reader.read(physics_tick=0,simulation_time_s=0.,
                                         commanded_full12=ack.rows[row]["drive_target_full12"])
                _validate_sensor_contract(observation,SENSED_BODIES,require_finite=True)
                _validate_rate_contract(self._row_adapters[row],controller)
                control = controller.step(observation,sim_time_s=0.)
                _validate_controller_clock(control,physics_tick=0,sim_time_s=0.)
                semantic._controller = controller
                semantic._adapter = self._row_adapters[row]
                semantic._level_reference_orientation = semantic._level_fixed
                semantic._reset_metadata = {
                    **self._reset_metadata(row=row,seed=seed_rows[row],options={},
                        reset_lifecycle=lifecycle,logical_state_id="P01"),
                    "schema":"wlr50_clean.semantic_vector_reset.v1",
                    "execution_mode":"semantic_B_or_C",
                    "level_calibration_sample_count":0,
                    "level_reference_source":"versioned_fixed_chassis_axes",
                    "effective_phase_entry_semantics":"current_physical_validity_no_reference_entry_gate",
                    "reset_only_prime_physics_ticks":SETTLE_TICKS,
                    "independent_semantic_evaluator_per_environment":True,
                    "policy_credit_excludes_settle":True,
                }
                semantic._last_atomic_ack = ack.rows[row]
                semantic._raw_observation = observation
                semantic._controller_frame = control
                frame = semantic._build_authoritative_frame(observation,control,previous_frame=None)
                if _terminal_reason(frame) is not None:
                    raise VectorizedIsaacBackendError(f"row {row} natural P01 reset produced a terminal physical state")
                semantic._authoritative_frame = frame
                semantic._done = False
                frames.append(frame)
                controller_frames.append(control)
            self._frames,self._controller_frames = tuple(frames),tuple(controller_frames)
            self._done = (False,)*8
            self._episode_tick = 0
            self._reset_count += 1
            return self._batched_frame()
        except Exception:
            self._reset_transaction_poisoned = True
            raise

    def set_actuator_target_audit_requests(self, phase_ids, raw_actions, masks):
        if len(phase_ids)!=8 or len(raw_actions)!=8 or len(masks)!=8:
            raise ValueError("audit request must contain exactly eight rows")
        self._policy_requests = tuple(actuator_target_audit_request(phase,raw,mask)
            for phase,raw,mask in zip(phase_ids,raw_actions,masks,strict=True))

    def step_physics_batch(self, applied_actions_full12):
        if not self._frames or any(self._done):
            raise VectorizedIsaacBackendError("valid whole-batch reset is required")
        if len(applied_actions_full12)!=8 or any(request is None for request in self._policy_requests):
            raise ValueError("one full audited action request per row is required")
        plans = tuple(build_residual_actuation_plan(action,
            frozen_nominal_full12=control.full12,
            drive_feedback_bias_full12=control.drive_feedback_bias_full12,
            normal_drive_bias_full12=control.normal_drive_bias_full12)
            for action,control in zip(applied_actions_full12,self._controller_frames,strict=True))
        if any(not control.full12_atomic_write_required for control in self._controller_frames):
            raise VectorizedIsaacBackendError("semantic controller did not request atomic Full12")
        previous_final = tuple(tuple(self.command_adapter._final_drive[row][name]
                                     for name in SERVO_ORDER) for row in range(8))
        raw_ack = self.command_adapter.apply_batch(
            tuple(plan.frozen_nominal_full12 for plan in plans),
            physics_tick=self.global_physics_step_count,
            tracking_servo_names=tuple(control.tracking_servo_names for control in self._controller_frames),
            drive_feedback_bias_full12=tuple(plan.combined_post_mapper_bias_full12 for plan in plans))
        acks = []
        for row,plan in enumerate(plans):
            audit = build_actuator_target_effect_audit(adapter=self._row_adapters[row],
                actuation=plan,raw_ack=raw_ack.rows[row],
                previous_final_drive_servo_deg=previous_final[row],
                source_phase_id=self._frames[row].state_id,
                policy_request=self._policy_requests[row])
            acks.append({**plan.annotate_ack(raw_ack.rows[row]),"actuator_target_effect_audit":audit})
        before_global = self.global_physics_step_count
        self._advance_global_physics()
        if self.global_physics_step_count!=before_global+1:
            raise VectorizedIsaacBackendError("batch must advance the one SimulationContext once")
        tick = self._episode_tick+1
        frames,controls = [],[]
        for row,(reader,controller,semantic,previous) in enumerate(zip(
                self.readers,self.controllers,self._semantics,self._frames,strict=True)):
            observation = reader.read(physics_tick=tick,simulation_time_s=tick*PHYSICS_DT_S,
                                     commanded_full12=acks[row]["drive_target_full12"])
            _validate_sensor_contract(observation,SENSED_BODIES,require_finite=False)
            control = controller.step(observation,sim_time_s=tick*PHYSICS_DT_S)
            _validate_controller_clock(control,physics_tick=tick,sim_time_s=tick*PHYSICS_DT_S)
            semantic._last_atomic_ack = acks[row]
            semantic._previous_action_full12 = tuple(applied_actions_full12[row])
            semantic._raw_observation,semantic._controller_frame = observation,control
            semantic._episode_tick = tick
            frame = semantic._build_authoritative_frame(observation,control,previous_frame=previous)
            semantic._authoritative_frame = frame
            semantic._done = _terminal_reason(frame) is not None
            frames.append(frame)
            controls.append(control)
        self._episode_tick = tick
        self._frames,self._controller_frames = tuple(frames),tuple(controls)
        self._done = tuple(_terminal_reason(frame) is not None for frame in frames)
        return self._batched_frame()

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import pytest

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.action_projection import SafetyProjection
from wlr50_clean.ppo.semantic_backend import build_semantic_projector
from wlr50_clean.ppo.semantic_env import SemanticEpisodeEnv, WRITE_COUNTERS, _terminal_reason
from wlr50_clean.ppo.semantic_observation import (
    GOAL_KEYS, HISTORY_GROUPS, LEGS, STAGES, SemanticObservationBuilder,
    SemanticObservationError, load_semantic_observation_schema,
)
from wlr50_clean.ppo.semantic_reward import (
    FAMILIES, SemanticRewardCalculator, SemanticRewardSample, load_semantic_reward_config,
)
from wlr50_clean.ppo.semantic_supervisor import GOAL_FEATURE_KEYS, TaskStageSupervisor
from wlr50_clean.ppo.termination import TerminationSignals
from wlr50_clean.sensing.geometry import locked_obstacle_planes, WHEEL_JOINT_TO_BODY
from wlr50_clean.sensing.observation import (
    BodyCollisionStatus, BodyContactObservation, CenterOfMassObservation, CollisionRole,
    ContactClass, ImuObservation, JointObservation, Observation, PairContactObservation,
    RigidBodyObservation, SupportDiagnostics, WheelObservation,
)

ZERO12 = (0.0,)*12


def _raw(tick=0, *, pitch=0.0, roll=0.0, omega=(0.0,0.0,0.0), joint=10.0):
    # Production geometry convention is LEFT positive y > RIGHT negative y.
    planes = locked_obstacle_planes()
    cr,sr,cp,sp = math.cos(roll/2),math.sin(roll/2),math.cos(pitch/2),math.sin(pitch/2)
    q = (cr*cp,sr*cp,cr*sp,-sr*sp)
    base = RigidBodyObservation("base_link",(0.,0.,.3),q,(0.,0.,0.),omega)
    joints = {name:JointObservation(name,joint,2.,0.,-999.) for name in SERVO_ORDER}
    wheels,contacts,bodies = {},{}, {"base_link":base}
    for name in WHEEL_ORDER:
        body_name = WHEEL_JOINT_TO_BODY[name]
        wheels[name] = WheelObservation(name,body_name,0.,0.,(.2,0.,.05),(.2,0.,0.),"real_fixture_collider",True)
        bodies[body_name] = RigidBodyObservation(body_name,(.2,0.,.05),(1.,0.,0.,0.),(0.,0.,0.),(0.,0.,0.))
        def pair(active,other):
            return PairContactObservation(body_name,other,active,(0.,0.,25. if active else 0.),
                25. if active else 0.,0.,(.2,0.,0.),((0.,0.,25. if active else 0.),)*8,
                (active,)*8,8 if active else 0,"exact_pair_fixture",True)
        contacts[body_name] = BodyContactObservation(body_name,CollisionRole.WHEEL,ContactClass.GROUND,
                                                   pair(True,"ground"),pair(False,"obstacle"))
    imu = ImuObservation(q,omega,(0.,0.,0.),(0.,0.,9.81),(0.,0.,-1.),"fixture")
    com = CenterOfMassObservation((0.,0.,.25),(0.,0.,0.),10.,tuple(bodies),"fixture",True)
    support = SupportDiagnostics(tuple(WHEEL_JOINT_TO_BODY.values()),(),(),(0.,0.),.1,True,4,"fixture",True)
    return Observation("fixture",tick,tick/120,1/120,joints,wheels,contacts,bodies,base,imu,planes,com,support,
        BodyCollisionStatus(False,False,False,0.,"clear"),actual_full12=(joint,)*8+(0.,)*4,
        commanded_full12=ZERO12,measured_wheel_velocity_rad_s=(0.,)*4,joint_positions_deg=(joint,)*8)


def _frame(tick=0, *, stage="P01", phi=.02, success=False, reason=None, raw=None):
    completed = list(STAGES[:int(stage[1:])-1])
    if success:
        stage,completed,phi = "P13",list(STAGES),1.0
    task = {"schema":"wlr50_clean.semantic_task.v2","stage_id":stage,"purpose":"fixture",
        "phase_progress":.26,"task_progress_potential":phi,"goal_features":dict.fromkeys(GOAL_KEYS,0.),
        "completed_stage_ids":completed,"success":success,"termination_reason":"SUCCESS" if success else reason,
        "substage":"EXECUTION","stage_elapsed_s":tick/120,"remaining_task_time_s":200-tick/120,
        **{key:dict.fromkeys(LEGS,False) for key in ("active_lift_history","front_edge_crossed_history","placed_history")}}
    mapper = {key:(0.,)*8 for key in ("requested_servo_deg","tracking_compensation_deg","applied_drive_command_deg","final_drive_servo_deg")}
    mapper.update(nominal_target_reached=(True,)*8,tracking_active=(False,)*8,retiring_stale_bias=(False,)*8,feedback_tick=tick)
    return SimpleNamespace(physics_tick=tick,sim_time_s=tick/120,state_id=stage,
        nominal_action_full12=ZERO12,action_mask_full12=(1,)*12,safety_projection=SafetyProjection(),
        termination_signals=TerminationSignals(success=success),info={
            "raw_observation":raw or _raw(tick),"semantic_task":task,"mapper_state_summary":mapper,
            "mapped_nominal_full12":ZERO12,"drive_target_full12":ZERO12,"controller_task_result":None})


def _built(frame=None):
    schema = load_semantic_observation_schema()
    builder = SemanticObservationBuilder(schema)
    return builder.build(frame or _frame(),dict.fromkeys(HISTORY_GROUPS,ZERO12))


def _sample(before,after, **kwargs):
    values = {key:ZERO12 for key in ("nominal","previous_nominal","residual","previous_residual",
                                    "actual_drive","previous_actual_drive","previous_previous_actual_drive")}
    values.update(residual_caps=(4.,)*8+(.12,)*4,**kwargs)
    return SemanticRewardSample(before,after,1/120,**values)


def test_live_dataclasses_and_real_obstacle_planes_encode_automatic_dimension():
    raw = _raw()
    assert raw.obstacle.left_y_m > raw.obstacle.right_y_m
    built = _built(_frame(raw=raw))
    schema = load_semantic_observation_schema()
    assert len(schema.encode(built.groups)) == sum(row["size"] for row in schema.groups)
    assert schema.dimension not in (85,125)
    assert built.groups["actual_joint_position_deg"] == (10.,)*8
    assert built.groups["actual_joint_velocity_deg_s"] == (2.,)*8
    assert built.groups["actual_joint_position_deg"] != tuple(j.error_deg for j in raw.joints.values())
    assert GOAL_KEYS == GOAL_FEATURE_KEYS


def test_episode_start_tilt_is_not_redefined_as_horizontal():
    built = _built(_frame(raw=_raw(pitch=.15,roll=.2)))
    assert built.groups["chassis_rpy_rad"][:2] == pytest.approx((.2,.15))
    assert built.groups["projected_gravity_chassis"] != (0.,0.,-1.)
    assert built.groups["orientation_derivative_valid"] == (0.,)


def test_real_task_supervisor_metadata_encodes_without_fixture_contract_shortcuts():
    raw = _raw(joint=0.)
    supervisor = TaskStageSupervisor()
    task = supervisor.observe_and_update(raw)
    frame = _frame(raw=raw)
    frame.info["semantic_task"] = task
    frame.state_id = task["stage_id"]
    built = _built(frame)
    assert built.task["termination_reason"] is None
    assert built.groups["task_progress"][1] == task["task_progress_potential"]
    assert built.groups["goal_features"] == tuple(task["goal_features"][key] for key in GOAL_FEATURE_KEYS)


def test_euler_pitch_derivative_comes_from_pose_not_body_omega_y():
    schema = load_semantic_observation_schema()
    builder = SemanticObservationBuilder(schema)
    history = dict.fromkeys(HISTORY_GROUPS,ZERO12)
    builder.build(_frame(raw=_raw(pitch=.1,roll=.3,omega=(0.,3.,0.))),history)
    built = builder.build(_frame(1,raw=_raw(1,pitch=.101,roll=.3,omega=(0.,3.,0.))),history)
    assert built.groups["euler_roll_pitch_rate_rad_s"][1] == pytest.approx(.12)
    assert built.groups["body_angular_velocity"][1] != pytest.approx(.12)


@pytest.mark.parametrize("key",["semantic_task","mapper_state_summary","mapped_nominal_full12"])
def test_missing_live_metadata_fails_closed(key):
    frame = _frame()
    del frame.info[key]
    with pytest.raises(SemanticObservationError):
        _built(frame)


def test_unverified_geometry_fails_closed_and_reversed_planes_are_rejected():
    raw = _raw()
    wheels = dict(raw.wheels)
    wheels[WHEEL_ORDER[0]] = replace(wheels[WHEEL_ORDER[0]],geometry_verified=False)
    with pytest.raises(SemanticObservationError,match="unverified"):
        _built(_frame(raw=replace(raw,wheels=wheels)))
    with pytest.raises(SemanticObservationError,match="planes"):
        _built(_frame(raw=replace(raw,obstacle=replace(raw.obstacle,left_y_m=-1.,right_y_m=1.))))


def test_reward_has_only_five_families_and_standing_still_cannot_profit():
    before,after = _built(),_built(_frame(1))
    result = SemanticRewardCalculator().evaluate(before,after,[_sample(before,after)],termination_reason=None,task_success=False)
    assert tuple(result["families"]) == FAMILIES
    assert result["total"] < 0
    assert result["terminal_event"] == 0


def test_phase_label_change_is_not_a_reward_event_or_potential_reset():
    before = _built(_frame(stage="P09",phi=.65))
    same = _built(_frame(1,stage="P09",phi=.65))
    crossed = _built(_frame(1,stage="P10",phi=.65))
    calculator = SemanticRewardCalculator()
    a = calculator.evaluate(before,same,[_sample(before,same)],termination_reason=None,task_success=False)
    b = calculator.evaluate(before,crossed,[_sample(before,crossed)],termination_reason=None,task_success=False)
    assert a == b
    assert b["potential_before"] == b["potential_after"] == .65


def test_discounted_closed_potential_cycle_cannot_generate_positive_credit():
    states = [_built(_frame(i,phi=phi)) for i,phi in enumerate((.2,.8,.2))]
    calculator = SemanticRewardCalculator()
    rewards = [calculator.evaluate(a,b,[_sample(a,b)],termination_reason=None,task_success=False)["potential_shaping"]
               for a,b in zip(states,states[1:])]
    assert rewards[0]+calculator.config.gamma*rewards[1] == pytest.approx(5*(calculator.config.gamma**2*.2-.2))
    assert rewards[0]+calculator.config.gamma*rewards[1] < 0


@pytest.mark.parametrize("reason",["BODY_COLLISION","WHEEL_ONLY_CLIMB","TASK_TIMEOUT","FALL","HARD_JOINT_LIMIT","NAN_INF"])
def test_failure_and_task_timeout_have_zero_potential_no_bootstrap(reason):
    before,after = _built(),_built(_frame(1))
    result = SemanticRewardCalculator().evaluate(before,after,[_sample(before,after)],termination_reason=reason,task_success=False)
    assert result["potential_after"] == 0
    assert result["terminal_bootstrap_allowed"] is False
    assert result["terminal_event"] == -40
    assert result["total"] < -40


def test_success_is_one_terminal_event_and_failure_exceeds_avoidable_cost_bound():
    before,after = _built(),_built(_frame(1,success=True))
    calculator = SemanticRewardCalculator()
    result = calculator.evaluate(before,after,[_sample(before,after)],termination_reason="SUCCESS",task_success=True)
    assert result["terminal_event"] == 40
    assert result["potential_after"] == 0
    assert calculator.config.values["failure_cost"] > calculator.config.failure_avoidance_bound
    with pytest.raises(ValueError,match="disagree"):
        calculator.evaluate(before,after,[_sample(before,after)],termination_reason=None,task_success=True)


def test_regularizer_can_be_disabled_and_transfer_attitude_is_less_restrictive():
    before = _built(_frame(raw=_raw(pitch=.3)))
    after = _built(_frame(1,raw=_raw(1,pitch=.3)))
    sample = _sample(before,after,residual=(1.,)*8+(.01,)*4)
    config = load_semantic_reward_config()
    config = replace(config,values={**config.values,"control_regularization_enabled":False})
    calculator = SemanticRewardCalculator(config)
    a = calculator.evaluate(before,after,[sample],termination_reason=None,task_success=False)
    transfer = replace(after,task={**after.task,"substage":"TRANSFER"})
    b = calculator.evaluate(before,transfer,[replace(sample,current=transfer)],termination_reason=None,task_success=False)
    assert a["families"]["control_regularization"] == 0
    assert b["families"]["body_stability"] > a["families"]["body_stability"]


class Backend:
    def __init__(self, *, terminal_tick=None, reason=None, start_tick=0, transition=False):
        self.terminal_tick,self.reason,self.start_tick,self.transition = terminal_tick,reason,start_tick,transition
        self.calls = []
        self.requests = []
    def reset(self, *, seed, options):
        self.tick = self.start_tick
        self.calls = []
        return _frame(self.tick)
    def set_actuator_target_audit_request(self,phase_id,raw_policy_action_full12,phase_mask_full12):
        self.requests.append((phase_id,raw_policy_action_full12,phase_mask_full12))
    def step_physics(self,applied_action_full12):
        self.tick += 1
        self.calls.append(applied_action_full12)
        terminal = self.tick == self.terminal_tick
        frame = _frame(self.tick,stage="P02" if self.transition and self.tick>=4 else "P01",
                       reason=self.reason if terminal else None,success=terminal and self.reason=="SUCCESS")
        frame.info["drive_target_full12"] = applied_action_full12
        return frame


def test_actual_env_projector_reward_signature_and_eight_tick_action_hold():
    backend = Backend(transition=True)
    observed = []
    env = SemanticEpisodeEnv(backend,projector=build_semantic_projector(),tick_observer=lambda a,b,p:observed.append((a,b,p)))
    initial = env.reset()
    raw = (.05,)*12
    step = env.step(raw)
    assert len(initial) == len(step.observation) == env.observation_dimension
    assert len(backend.calls) == len(observed) == 8
    assert step.terminated is step.truncated is False
    assert step.info["raw_policy_action_full12"] == raw
    assert step.info["phase_transition_action_jump"][0]["handoff_hold_used"] is True
    assert tuple(step.info["reward"]["families"]) == FAMILIES
    assert backend.requests == [("P01",raw,(1,)*12)]
    assert any(x != 0 for x in backend.calls[-1])


@pytest.mark.parametrize("reason",["BODY_COLLISION","SUCCESS","INCOMPLETE_CONTROLLER_BLOCKED"])
def test_env_valid_terminal_breaks_repeat_and_never_bootstraps(reason):
    backend = Backend(terminal_tick=3,reason=reason)
    env = SemanticEpisodeEnv(backend)
    env.reset()
    result = env.step(ZERO12)
    assert result.terminated is True and result.truncated is False
    assert len(backend.calls) == 3
    assert result.info["time_outs"] is False
    assert result.info["terminal_bootstrap_allowed"] is False
    with pytest.raises(RuntimeError,match="reset"):
        env.step(ZERO12)
    env.reset()
    assert env.done is False and env.decision_count == 0
    assert env.bridge.previous_projected_residual_full12 == ZERO12


def test_200_seconds_is_task_terminal_and_partial_policy_step_cost_uses_actual_dt():
    backend = Backend(start_tick=23998)
    env = SemanticEpisodeEnv(backend)
    env.reset()
    step = env.step(ZERO12)
    assert len(backend.calls) == 2
    assert step.info["termination_reason"] == "TASK_TIMEOUT"
    assert step.info["reward"]["elapsed_physics_s"] == pytest.approx(2/120)
    assert step.terminated is True and step.truncated is False


@pytest.mark.parametrize("result",["SAFETY_ABORT","SOME_UNKNOWN_RESULT","INFRASTRUCTURE_ERROR"])
def test_unknown_abort_cannot_continue_collecting_policy_credit(result):
    frame = _frame()
    frame.info["controller_task_result"] = result
    with pytest.raises(RuntimeError):
        _terminal_reason(frame)


def test_repeated_completed_stage_cannot_appear_in_actor_history():
    frame = _frame()
    frame.info["semantic_task"]["completed_stage_ids"] = ["P01","P01"]
    with pytest.raises(SemanticObservationError,match="unique ordered prefix"):
        _built(frame)


class AuditedBackend(Backend):
    def __init__(self, *, bad_tick=None, missing_tick=None, write_tick=None):
        super().__init__()
        self.bad_tick,self.missing_tick,self.write_tick = bad_tick,missing_tick,write_tick
    def step_physics(self,applied_action_full12):
        frame = super().step_physics(applied_action_full12)
        request_phase,raw,mask = self.requests[-1]
        # Command ticks include settle offset and intentionally differ from episode ticks.
        command_tick = self.tick+300
        changed = (self.tick==1,)+(False,)*11
        audit = {"schema":"wlr50_clean.actuator_target_effect_audit.v1","verified":self.tick!=self.bad_tick,
            "actual_mapping_matches_dispatch":True,"setter_dispatch_targets_equal":True,"same_tick_counterfactual":True,
            "physics_tick":command_tick,"source_phase_id":frame.state_id,"policy_request_phase":request_phase,
            "raw_policy_action_full12":list(raw),"phase_mask_full12":list(mask),
            "changed_channels_full12":changed,"changed_target_channel_count":sum(changed)}
        frame.info["atomic_ack"] = {"physics_tick":command_tick}
        if self.tick!=self.missing_tick:
            frame.info["actuator_target_effect_audit"] = audit
        frame.info.update(dict.fromkeys(WRITE_COUNTERS,0))
        if self.tick==self.write_tick:
            frame.info[WRITE_COUNTERS[0]] = 1
        frame.info["semantic_task"]["transition_evidence"] = [{"physics_tick":2,"from_stage":"P01","to_stage":"P02"}] if self.tick>=2 else []
        return frame


def test_all_physics_audits_preserve_early_native_effect_and_stage_evidence_once():
    env = SemanticEpisodeEnv(AuditedBackend())
    env.reset()
    step = env.step((.05,)*12)
    summary = step.info["actuator_target_effect_audit_summary"]
    assert summary == {"physics_ticks":8,"verified_tick_count":8,"all_ticks_verified":True,
                       "actual_native_effect_tick_count":1,"own_phase_request_effect_tick_count":1}
    assert step.info["actuator_target_effect_audit"]["changed_target_channel_count"] == 0
    assert len(step.info["actuator_target_effect_audit_ticks"]) == 8
    assert step.info["no_in_episode_state_writes_verified"] is True
    assert len(step.info["stage_transition_evidence"]) == 1
    assert env.step((.05,)*12).info["stage_transition_evidence"] == []


@pytest.mark.parametrize("setting",[{"bad_tick":3},{"missing_tick":3}])
def test_intermediate_missing_or_invalid_audit_cannot_be_hidden_by_valid_final_tick(setting):
    env = SemanticEpisodeEnv(AuditedBackend(**setting))
    env.reset()
    info = env.step((.05,)*12).info
    assert info["actuator_target_effect_audit"]["verified"] is True
    assert info["actuator_target_effect_audit_summary"]["all_ticks_verified"] is False
    assert info["actuator_target_effect_audit_summary"]["verified_tick_count"] == 7


def test_intermediate_forbidden_write_and_missing_counter_fail_closed():
    env = SemanticEpisodeEnv(AuditedBackend(write_tick=2))
    env.reset()
    info = env.step((.05,)*12).info
    assert info[WRITE_COUNTERS[0]] == 1
    assert info["no_in_episode_state_writes_verified"] is False
    env = SemanticEpisodeEnv(Backend())
    env.reset()
    info = env.step(ZERO12).info
    assert info[WRITE_COUNTERS[0]] is None
    assert info["no_in_episode_state_writes_verified"] is False

"""CPU contract checks; synthetic sensor sequences are not live suffix availability."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_semantic_supervisor import observation, advance, leg_state
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor, SemanticControllerAdapter
from wlr50_clean.ppo.semantic_prefix import (
    DispatchReceipt, PrefixRequest, PrefixUnavailable, ResetOnlyPrefixController,
    PrefixSemanticIsaacBackend, PrefixCreditCore, ZERO12,
)
from wlr50_clean.ppo.isaac_fsm_backend import DEFAULT_FSM_PATH, DEFAULT_MOTION_CONTRACT_PATH
from wlr50_clean.fsm.state_spec import load_fsm_spec, Lifecycle
from wlr50_clean.reference.motion_contract import load_motion_contract
from wlr50_clean.fsm.controller import ControllerFrame
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER


def resources():
    return load_fsm_spec(DEFAULT_FSM_PATH),load_motion_contract(DEFAULT_MOTION_CONTRACT_PATH)


def receipt(tick, nominal=None, bias=ZERO12, tracking=()):
    if nominal is None: nominal=resources()[1].phases[0].start_full12
    ack={"physics_tick":tick+600,"articulation_writes_this_call":1,
         "applied_full12":tuple(nominal),"drive_feedback_bias_requested_full12":bias,
         "native_drive_target_full12":tuple(nominal),"drive_target_full12":tuple(nominal)}
    return DispatchReceipt.from_ack(source_tick=tick,command=nominal,tracking=tracking,bias=bias,ack=ack)


class Teacher:
    def __init__(self,nominal):
        self.nominal=tuple(nominal);self.motion=SimpleNamespace();self.calls=0
    def step(self,observation,*,sim_time_s):
        tick=observation["physics_tick"];self.calls+=1
        return ControllerFrame(tick,sim_time_s,"P01",Lifecycle.EXECUTE_MOTION,self.nominal,
            tick%8==0,True,False,(SERVO_ORDER[0],),ZERO12,(.1,)*8+(0.,)*4,
            {"teacher":True},False,None,None,())


def synthetic_prefix(target="P09"):
    """Every supervisor update consumes a contiguous measured sensor sample."""
    spec,contract=resources();teacher=Teacher(contract.phases[0].start_full12)
    controller=ResetOnlyPrefixController(teacher,TaskStageSupervisor(),spec,contract,PrefixRequest(target))
    obs=observation()
    schedules={
        "P01":("FR",{"air":True,"bottom":.012,"hip":1.5}),
        "P02":("FR",{"air":True,"bottom":.075,"hip":4.,"x":.499}),
        "P03":("FR",{"top":True,"bottom":.05,"x":.53}),
        "P04":("FL",{"air":True,"bottom":.012,"hip":1.5}),
        "P05":("FL",None),
        "P08":("RR",{"air":True,"bottom":.012,"hip":1.5}),
        "P09":("RR",None),
        "P11":("RL",{"air":True,"bottom":.012,"hip":1.5}),
        "P12":("RL",None),
    }
    phase_age={}
    for tick in range(180):
        if tick: obs=advance(obs)
        stage=controller.supervisor.stage_id
        phase_age[stage]=phase_age.get(stage,0)+1
        if stage in schedules:
            leg,values=schedules[stage]
            if values is None:
                values=({"air":True,"bottom":.075,"hip":4.,"x":.499}
                        if phase_age[stage]<=3 else {"top":True,"bottom":.05,"x":.53})
            leg_state(obs,leg,**values)
        frame=controller.step(obs,sim_time_s=tick/120)
        controller.record_verified_dispatch(receipt(tick,frame.full12,
            tuple(a+b for a,b in zip(frame.normal_drive_bias_full12,frame.drive_feedback_bias_full12)),
            frame.tracking_servo_names))
        if controller.mode=="READY":
            return controller,teacher,obs,frame
        assert controller.task_snapshot["termination_reason"] is None, controller.task_snapshot
    raise AssertionError(controller.task_snapshot)


@pytest.mark.parametrize("phase",["P06","P07","P08","P09","P10","P11","P12","P13"])
def test_actual_shadow_history_reaches_handoff_without_label_restore(phase):
    controller,teacher,obs,frame=synthetic_prefix(phase)
    assert controller.handoff_record["actual_phase"]==phase
    assert controller.teacher is None and controller._semantic.supervisor is controller.supervisor
    assert controller._semantic.evaluator is controller.evaluator
    assert controller.handoff_record["atomic_ack"]["physics_tick"]==controller.handoff_record["source_control_tick"]+600
    assert controller.handoff_record["handoff_tick"]==controller.handoff_record["source_control_tick"]+1
    assert controller.supervisor.episode_started_s==0.
    assert frame.physics_tick==obs["physics_tick"] and frame.sim_time_s>0
    assert controller.task_snapshot["remaining_task_time_s"]==pytest.approx(200-frame.sim_time_s)
    assert controller.task_snapshot["placed_history"]["FR"]
    assert controller.task_snapshot["placed_history"]["FL"]
    if int(phase[1:])>=10: assert controller.task_snapshot["placed_history"]["RR"]
    if phase in ("P06","P07"):
        assert controller.handoff_record["semantic_task"]["active_lift_history"]["RR"] is False
    calls=teacher.calls
    next_obs=advance(obs)
    controller.step(next_obs,sim_time_s=next_obs["simulation_time_s"])
    assert teacher.calls==calls


def test_public_handoff_preserves_seed_and_does_not_observe_twice():
    spec,contract=resources();supervisor=TaskStageSupervisor();obs=observation()
    for _ in range(9):
        task=supervisor.observe_and_update(obs)
        if obs["physics_tick"]<8:obs=advance(obs)
    before=deepcopy(supervisor.snapshot)
    seed=contract.phases[0].start_full12
    controller,frame=SemanticControllerAdapter.from_live_prefix(spec,contract,supervisor=supervisor,
        nominal_full12=seed,tracking_servo_names=(SERVO_ORDER[0],),physics_tick=8,sim_time_s=8/120)
    assert supervisor.snapshot==before
    assert frame.full12==tuple(seed) and frame.tracking_servo_names==(SERVO_ORDER[0],)
    assert controller.physics_tick==9 and controller._last_time==8/120
    controller.step(advance(obs),sim_time_s=9/120)
    assert controller.physics_tick==10


@pytest.mark.parametrize("field,value",[("physics_tick",True),("articulation_writes_this_call",0),
                                       ("applied_full12",(1.,)*12)])
def test_receipt_rejects_unbound_dispatch(field,value):
    good=receipt(0);bad=dict(good.ack);bad[field]=value
    with pytest.raises(ValueError):
        DispatchReceipt.from_ack(source_tick=0,command=good.nominal,tracking=(),bias=ZERO12,ack=bad)


def test_controller_rejects_new_observation_without_dispatch():
    spec,contract=resources()
    c=ResetOnlyPrefixController(Teacher(contract.phases[0].start_full12),TaskStageSupervisor(),spec,contract,PrefixRequest())
    c.step(observation(),sim_time_s=0.)
    with pytest.raises(ValueError,match="dispatch receipt"):c.step(observation(1),sim_time_s=1/120)


@pytest.mark.parametrize("target",["P01","P02","SUCCESS"])
def test_initial_target_scope_is_explicit(target):
    with pytest.raises(ValueError):PrefixRequest(target)


def test_configure_prefix_is_not_a_snapshot_restore():
    backend=object.__new__(PrefixSemanticIsaacBackend)
    with pytest.raises(ValueError):backend.configure_prefix("true")
    backend.configure_prefix(False)
    assert backend._prefix_enabled is False


def real_prefix_runtime(monkeypatch, target="P06", *, task_spec_path=None):
    """Production reset/controller/mapper/core seam with synthetic sensors and tensor physics.

    No causal claim that these scripted wheel motions follow the teacher.
    Scene/reader, teacher commands and USD limit installation are simulated;
    the real frozen teacher factory is separately checked before substitution.
    """
    from test_actuator_target_effect import _adapter
    from test_isaac_fsm_backend import FakeRuntime, FakeAdapter
    from test_semantic_observation_reward_env import _raw
    from wlr50_clean.ppo.semantic_legacy_evaluation import physical_json
    from wlr50_clean.sensing.contact_classifier import SENSED_BODIES, DEFAULT_ROLE_MAP
    from wlr50_clean.sensing.observation import (
        BodyContactObservation, PairContactObservation, RigidBodyObservation, ContactClass,
    )
    runtime=FakeRuntime()
    holder={}
    def make_adapter(scene):
        adapter=_adapter()
        adapter.verify_authoritative_servo_limits_adopted=lambda:None
        adapter.joint_limit_initialization_evidence=FakeAdapter(runtime).joint_limit_initialization_evidence
        runtime.adapter=adapter
        return adapter
    class Reader:
        def __init__(self):
            raw=_raw(joint=0.)
            bodies,contacts=dict(raw.bodies),dict(raw.contacts)
            for name in SENSED_BODIES:
                if name not in bodies:
                    bodies[name]=RigidBodyObservation(name,(0.,0.,.2),(1.,0.,0.,0.),(0.,)*3,(0.,)*3)
                if name not in contacts:
                    def pair(kind):
                        return PairContactObservation(name,kind,False,(0.,)*3,0.,0.,None,
                            ((0.,)*3,)*8,(False,)*8,0,"offline_exact_fixture",True)
                    contacts[name]=BodyContactObservation(name,DEFAULT_ROLE_MAP[name],ContactClass.AIR,
                                                        pair("ground"),pair("obstacle"))
            raw=replace(raw,bodies=bodies,contacts=contacts,
                        center_of_mass=replace(raw.center_of_mass,included_bodies=tuple(bodies)))
            self.raw=physical_json(raw)
            self.ages={}
        def read(self,*,physics_tick,simulation_time_s,commanded_full12):
            assert simulation_time_s==pytest.approx(physics_tick/120,abs=1e-12)
            obs=self.raw
            obs.update(physics_tick=physics_tick,simulation_time_s=simulation_time_s,
                       commanded_full12=list(commanded_full12))
            controller=holder["backend"].prefix_controller
            stage=controller.supervisor.stage_id if controller is not None else "P01"
            self.ages[stage]=self.ages.get(stage,0)+1
            front=obs["obstacle"]["front_x_m"]
            for leg in ("RR","RL"):
                leg_state(obs,leg,x=front-(.4 if target=="P06" else .1))
            values={"P01":("FR",dict(air=True,bottom=.012,hip=1.5)),
                    "P02":("FR",dict(air=True,bottom=.075,hip=4.,x=front-.001)),
                    "P03":("FR",dict(top=True,bottom=.05,x=front+.03)),
                    "P04":("FL",dict(air=True,bottom=.012,hip=1.5,x=front-.1)),
                    "P05":("FL",dict(air=True,bottom=.075,hip=4.,x=front-.001)
                            if self.ages[stage]<=3 else dict(top=True,bottom=.05,x=front+.03))}
            if physics_tick and stage in values:
                leg,setting=values[stage]
                leg_state(obs,leg,**setting)
            for wheel in obs["wheels"].values():
                contact=obs["contacts"][wheel["body_name"]]
                contact["contact_class"]=("GROUND" if contact["ground"]["active"] else
                                           "OBSTACLE" if contact["obstacle"]["active"] else "AIR")
            return deepcopy(obs)
    dependencies=replace(runtime.dependencies(),adapter_from_scene=make_adapter,
        reader_from_scene=lambda scene,adapter,backends:Reader(),expected_contact_bodies=tuple(SENSED_BODIES))
    options={} if task_spec_path is None else {"task_spec_path":task_spec_path}
    backend=PrefixSemanticIsaacBackend(prefix_request=PrefixRequest(target),dependencies=dependencies,**options)
    holder["backend"]=backend
    factory=backend._new_prefix_controller
    def synthetic_teacher_factory(fsm_path,contract_path):
        from wlr50_clean.fsm.controller import SensorFsmController
        controller=factory(fsm_path,contract_path)
        if isinstance(controller,ResetOnlyPrefixController):
            assert isinstance(controller.teacher,SensorFsmController)
            teacher=Teacher(controller.contract.phases[0].start_full12)
            teacher.motion=SimpleNamespace(servo_rate_limit_deg_s=150.)
            controller.teacher=teacher
            controller.motion=teacher.motion
        return controller
    backend._semantic_controller_factory=synthetic_teacher_factory
    return runtime,backend


def test_teacher_task_snapshot_getter_stays_read_only_without_nominal_diagnostics():
    cfg=Path(__file__).resolve().parents[2]/"configs/ppo_semantic_v3"
    spec,contract=resources()
    teacher=Teacher(contract.phases[0].start_full12)
    supervisor=TaskStageSupervisor(cfg/"stage_task_spec.yaml")
    controller=ResetOnlyPrefixController(teacher,supervisor,spec,contract,PrefixRequest("P06"))
    raw=observation()
    for name,command in zip(SERVO_ORDER,teacher.nominal[:8],strict=True):
        raw["joints"][name]["command_deg"]=command
    controller.step(raw,sim_time_s=0.)
    before=deepcopy(supervisor.snapshot)
    clock=(controller.physics_tick,supervisor._last_observation_tick,
           supervisor.episode_started_s,controller.teacher_calls,teacher.calls)
    assert controller._semantic is None and controller.mode=="TEACHER"
    for _ in range(3):
        task=controller.task_snapshot
        assert task==before
        assert "nominal_provider_diagnostics" not in task
        task["stage_id"]="not_a_real_stage"
    assert supervisor.snapshot==before
    assert clock==(controller.physics_tick,supervisor._last_observation_tick,
                   supervisor.episode_started_s,controller.teacher_calls,teacher.calls)


def test_semantic_task_snapshot_forwards_live_diagnostics_without_alias_clock_or_encoding_changes(monkeypatch):
    import torch
    from wlr50_clean.ppo.semantic_env import SemanticEpisodeEnv
    from wlr50_clean.ppo.semantic_observation import HISTORY_GROUPS,SemanticObservationBuilder
    from wlr50_clean.ppo.semantic_prefix import PrefixRslAdapter
    cfg=Path(__file__).resolve().parents[2]/"configs/ppo_semantic_v3"
    runtime,backend=real_prefix_runtime(monkeypatch,"P06",task_spec_path=cfg/"stage_task_spec.yaml")
    core=SemanticEpisodeEnv(backend,collect_trace=False,
        action_config=cfg/"execution_profile.yaml",reward_config_path=cfg/"reward_config.yaml",
        observation_schema_path=cfg/"observation_schema.json")
    env=PrefixRslAdapter(core,seed=1001,device="cpu",evidence_sink=lambda record:None)
    encoded,_,done,_=env.step(torch.zeros((1,12)))
    controller=backend.prefix_controller
    inner=controller._semantic
    assert controller.mode=="READY" and inner is not None and not done.item()
    assert encoded["policy"].shape==(1,372)  # v3 includes the existing 48 transfer-role features.
    expected=inner.task_snapshot
    diagnostics=expected["nominal_provider_diagnostics"]
    assert "p06_rolling_retirement" in diagnostics and "p06_wheel_tail" in diagnostics
    assert core.frame.info["semantic_task"]["nominal_provider_diagnostics"]==diagnostics
    before=deepcopy(controller.supervisor.snapshot)
    def read_state():
        return (controller.physics_tick,inner.physics_tick,inner._last_time,
                controller.supervisor._last_observation_tick,controller.supervisor.episode_started_s,
                controller.teacher_calls,core.frame.physics_tick,core.decision_count,env.total_decisions,
                runtime.sim.step_count,backend._adapter.write_count,inner.nominal_provider.nominal_full12,
                inner.nominal_provider._source_motion._tick_index,
                tuple(layer["ticks"] for layer in inner.nominal_provider._continuous_layers))
    state=read_state()
    for _ in range(3):
        task=controller.task_snapshot
        assert task==expected
        task["nominal_provider_diagnostics"]["p06_rolling_retirement"]["wheel_gain"]=123.
        task["nominal_provider_diagnostics"]["p06_wheel_tail"]["status"]="not_a_real_status"
        assert controller.task_snapshot==expected
        assert inner.task_snapshot==expected
    assert read_state()==state and controller.supervisor.snapshot==before

    # Suggestion metadata is visible to log consumers but does not add or
    # replace any actor feature, reset its derivative history, or advance time.
    info_without=dict(core.frame.info)
    info_without["semantic_task"]={key:value for key,value in expected.items()
                                   if key!="nominal_provider_diagnostics"}
    frame_without=replace(core.frame,info=info_without)
    history=dict.fromkeys(HISTORY_GROUPS,ZERO12)
    schema=core.observation_schema
    with_diagnostics=SemanticObservationBuilder(schema).build(core.frame,history)
    without_diagnostics=SemanticObservationBuilder(schema).build(frame_without,history)
    assert schema.dimension==372
    assert schema.encode(with_diagnostics.groups)==schema.encode(without_diagnostics.groups)
    assert read_state()==state and controller.supervisor.snapshot==before


@pytest.mark.parametrize("target",["P06","P07"])
def test_real_backend_core_rsl_credit_seam_two_resets_and_native_mapping(monkeypatch,target):
    import torch
    from wlr50_clean.ppo.semantic_env import SemanticEpisodeEnv
    from wlr50_clean.ppo.semantic_prefix import PrefixRslAdapter
    from wlr50_clean.fsm.controller import SensorFsmController
    runtime,backend=real_prefix_runtime(monkeypatch,target)
    records=[]
    core=SemanticEpisodeEnv(backend,collect_trace=False)
    env=PrefixRslAdapter(core,seed=1001,device="cpu",evidence_sink=records.append)
    assert env.total_decisions==0
    miss=env.core.attempts[-1]["miss"] or {}
    assert env.core.start_record["from_P01_current_policy"] is False, (
        miss.get("reason"),miss.get("terminal_info",{}).get("termination_reason"),
        miss.get("terminal_info",{}).get("semantic_task",{}).get("physical_evaluator",{}).get("reason"),
        miss.get("terminal_info",{}).get("end_phase_id"),
        miss.get("terminal_info",{}).get("semantic_task",{}).get("completion_values"),
        miss.get("semantic_task",{}).get("stage_id"))
    assert env.core.start_record["requested_phase"]==target
    assert env.core.prefix_decisions>0 and env.core.prefix_ticks==core.frame.physics_tick
    assert env.core.credited_decisions==0
    assert backend.prefix_controller.teacher_calls>0
    assert backend.prefix_controller._semantic.supervisor is backend.prefix_controller.supervisor
    assert backend.prefix_controller.handoff_record["semantic_task"]["active_lift_history"]["RR"] is False
    mapper=backend._adapter.servo_target_mapper
    reader=backend._reader
    clock=core.frame.physics_tick
    steps,writes=runtime.sim.step_count,backend._adapter.write_count
    obs,reward,done,extras=env.step(torch.full((1,12),.05))
    assert obs["policy"].shape==(1,324)
    assert not done.item() and env.total_decisions==1
    assert backend._adapter.servo_target_mapper is mapper and backend._reader is reader
    assert runtime.sim.step_count==steps+8 and backend._adapter.write_count==writes+8
    info=extras["semantic_decisions"][0]
    assert info["physics_tick"]==clock+8 and info["decision_count"]==1
    assert info["physical_core_decision_count_including_prefix"]>1
    assert info["actuator_target_effect_audit_summary"]["all_ticks_verified"]
    assert info["no_in_episode_state_writes_verified"]
    assert info["task_result_scope"]=="teacher_initialized_suffix"
    assert "entry_observation" not in info["curriculum_start"]
    assert "handoff" not in info["curriculum_start"]
    assert info["curriculum_start"]["prefix_attempt_index"]==0
    assert not info["full_task_success"]
    assert info["semantic_task"]["remaining_task_time_s"]==pytest.approx(200-(clock+8)/120)
    assert all(not row["policy_credit"] for row in records)
    before_prefix=env.core.prefix_decisions
    env.core.reset(seed=1001)
    assert env.core.prefix_decisions>before_prefix
    assert backend._reader is not reader and backend._adapter.servo_target_mapper is not mapper
    assert runtime.reset_scene_count==2
    assert env.core.credited_decisions==1


@pytest.mark.parametrize("offset",[-1,True,1800])
def test_teacher_offset_is_bounded_initialization_not_a_historical_posture(offset):
    with pytest.raises(ValueError):PrefixRequest("P06",teacher_offset_decisions=offset)


def test_real_official_update_contains_only_credited_suffix_decisions(monkeypatch,tmp_path):
    import torch
    from wlr50_clean.ppo.semantic_env import SemanticEpisodeEnv
    from wlr50_clean.ppo.semantic_prefix import PrefixRslAdapter
    from wlr50_clean.ppo.semantic_training import construct_semantic_runner,train_semantic,parameter_hash
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        runtime,backend=real_prefix_runtime(monkeypatch,"P06")
        records=[]
        core=SemanticEpisodeEnv(backend,collect_trace=False)
        env=PrefixRslAdapter(core,seed=1001,device="cpu",evidence_sink=records.append)
        prefix_decisions=env.core.prefix_decisions
        prefix_tick=core.frame.physics_tick
        runner,_=construct_semantic_runner(env,seed=1001,device="cpu")
        before=parameter_hash(runner.alg.actor)
        result=train_semantic(runner,env,run_dir=tmp_path/"run",output_root=tmp_path/"outputs",
            stage="phase_suffix",decisions=128,contract={"test_only":True},seed=1001)
        assert env.total_decisions==128 and env.core.credited_decisions==128
        assert env.core.prefix_decisions==prefix_decisions
        assert core.decision_count==128+prefix_decisions
        assert core.frame.physics_tick==prefix_tick+128*8
        assert result["optimizer_steps_this_run"]==20
        assert parameter_hash(runner.alg.actor)!=before
        assert env.telemetry_summary()["core"]["decisions"]==128
        assert env.telemetry_summary()["core"]["physical_core_including_prefix"]["decisions"]==128+prefix_decisions
        assert env.telemetry_summary()["success_count"]==0
        assert runner.alg.storage.step==0
    finally:
        torch.set_num_threads(previous_threads)


def test_workspace_response_uses_three_real_reset_seams_without_optimizer(monkeypatch,tmp_path):
    from wlr50_clean.ppo import semantic_workspace_probe as probe
    from wlr50_clean.ppo import semantic_training
    runtime,backend=real_prefix_runtime(monkeypatch,"P06")
    monkeypatch.setattr(semantic_training,"construct_semantic_runner",
                        lambda *a,**kw:pytest.fail("probe must never construct an optimizer"))
    args=SimpleNamespace(run_dir=tmp_path,seed=1001,from_phase="P06",teacher_offset_decisions=0,
                         decisions=2,raw_magnitude=.5)
    result=probe.run_segments(None,args,{"offline_fixture":True},backend=backend)
    assert runtime.reset_scene_count==3
    assert result["optimizer_updates"]==result["optimized_policy_decisions"]==0
    assert result["automatic_training_gate"] is False
    assert [row["segment"] for row in result["segments"]]==["nominal_zero","old_range","new_range"]
    assert result["segments"][0]["actual_native_effect_ticks"]==0
    assert result["segments"][1]["raw_stimulus_full12"]==result["segments"][2]["raw_stimulus_full12"]
    for row in result["segments"]:
        assert row["response_physics_ticks"]==16
        assert row["actual_response_decisions"]==2
        assert row["prefix_decisions"]>0
        assert row["start"]["from_P01_current_policy"] is False
        assert set(row["artifacts"])=={"prefix_evidence.jsonl","native_response_120hz.jsonl","response_decisions.jsonl"}


@pytest.mark.parametrize("value",[float("nan"),float("inf"),0.,1.01])
def test_probe_raw_range_is_explicit_and_finite(value):
    from wlr50_clean.ppo.semantic_workspace_probe import response_actions
    with pytest.raises(ValueError):response_actions(value)

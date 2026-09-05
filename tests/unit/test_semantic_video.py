"""CPU contract tests for the additive video path; not physical success evidence."""
from dataclasses import dataclass
from types import SimpleNamespace as NS

import pytest
from wlr50_clean.ppo import semantic_video as video
from wlr50_clean.ppo.semantic_cli import parser
from wlr50_clean.ppo.semantic_video_cli import validate_video_args


@pytest.mark.parametrize("ticks,frames",[(1,32),(8,33),(12944,1650),(23751,3000),(23752,3001)])
def test_full_context_uses_native_120_to_15_grid(ticks,frames):
    assert video.frames_for_episode(ticks)==frames


@pytest.mark.parametrize("ticks",[0,-1,1.5,True])
def test_bad_episode_ticks_fail_closed(ticks):
    with pytest.raises(video.SemanticVideoError):
        video.frames_for_episode(ticks)


def test_observer_stops_exact_tick_not_next_policy_boundary():
    events=[]
    evaluator=NS(snapshot={"success":False,"termination_reason":None})
    def observe(before,after,projection):
        events.append(("metrics",after.physics_tick))
        evaluator.snapshot["success"]=after.physics_tick==3
    observer=video.EndpointObserver(NS(evaluator=evaluator,observe=observe),None,None)
    for tick in (1,2):
        observer(NS(physics_tick=tick-1),NS(physics_tick=tick),None)
    with pytest.raises(video._PhysicalEndpoint):
        observer(NS(physics_tick=2),NS(physics_tick=3),None)
    assert observer.reason=="SUCCESS"
    assert observer.last_frame.physics_tick==3
    assert observer.last_global_tick==67
    assert events==[("metrics",1),("metrics",2),("metrics",3)]


def test_observer_metrics_then_native_render_then_endpoint(monkeypatch):
    events=[]
    physical=NS(evaluator=NS(snapshot={"success":True,"termination_reason":None}),
                observe=lambda *_:events.append("metrics"))
    monkeypatch.setattr(video,"capture_frame",lambda *args:events.append(("render",args[2])))
    observer=video.EndpointObserver(physical,object(),object())
    with pytest.raises(video._PhysicalEndpoint):
        observer(NS(physics_tick=7),NS(physics_tick=8),None)
    assert events==["metrics",("render",72)]


def test_physical_failure_is_not_success_or_post_roll():
    physical=NS(evaluator=NS(snapshot={"success":False,"termination_reason":"TASK_FAILURE_WHEEL_ONLY_CLIMB"}),
                observe=lambda *_:None)
    observer=video.EndpointObserver(physical,None,None)
    with pytest.raises(video._PhysicalEndpoint):
        observer(NS(physics_tick=0),NS(physics_tick=1),None)
    assert observer.reason=="TASK_FAILURE_WHEEL_ONLY_CLIMB"


def test_native_capture_call_order_without_extra_physics():
    events=[]
    recorder=NS(before_render=lambda **kw:events.append(("before",kw)),
                after_render=lambda:events.append("after"),
                require_healthy=lambda:events.append("healthy"))
    backend=NS(render_video_frame=lambda:events.append("render"))
    video.capture_frame(recorder,backend,72)
    assert events==[("before",{"sim_step":72,"sim_time_s":.6}),"render","after","healthy"]
    with pytest.raises(video.SemanticVideoError):
        video.capture_frame(recorder,backend,73)


def post_backend(monkeypatch):
    from wlr50_clean.ppo import isaac_fsm_backend as original
    events=[]
    source=NS(drive_feedback_bias_full12=(.1,)*12,normal_drive_bias_full12=(.2,)*12,
              tracking_servo_names=("rear_right_hip",))
    controller=NS(physics_tick=31,termination="LEGACY_RUNNING")
    raw={"physics_tick":32,"simulation_time_s":32/120}
    adapter=NS(update_readback=lambda:events.append("readback"))
    scene=NS(sim=NS(step=lambda **kw:events.append(("step",kw))))
    reader=NS(read=lambda **kw:(events.append(("read",kw)) or raw))
    def atomic(adapter,command,**kwargs):
        events.append(("atomic",tuple(command),kwargs))
        return {"physics_tick":kwargs["physics_tick"],"drive_target_full12":tuple(command),
                "applied_full12":list(command),
                "drive_feedback_bias_requested_full12":list(kwargs["drive_feedback_bias_full12"]),
                "tracking_servo_names":list(kwargs["tracking_servo_names"])}
    backend=NS(_scene=scene,_adapter=adapter,_reader=reader,_controller=controller,
        _controller_frame=source,_episode_tick=31,_done=False,_previous_action_full12=tuple(range(12)),
        _episode_sensor_tick_offset=0,_last_atomic_ack={"physics_tick":443,
            "ppo_actuation_contract":"frozen_nominal_plus_post_mapper_residual.v1",
            "fsm_nominal_mapper_input_full12":[2.]*12,
            "combined_post_mapper_bias_full12":[.75]*12,
            "applied_full12":[2.]*12,"drive_feedback_bias_requested_full12":[.75]*12,
            "tracking_servo_names":["front_left_hip"]},
        _video_post_terminal_tick_count=0,_atomic_apply=atomic,
        _dependencies=NS(expected_contact_bodies=("body",)))
    monkeypatch.setattr(original,"_require_running",lambda *_:None)
    monkeypatch.setattr(original,"_validate_sensor_contract",lambda *a,**kw:events.append("validate"))
    monkeypatch.setattr(video,"measured_observation",lambda raw:raw)
    return backend,events


def test_post_roll_uses_common_success_without_faking_old_success(monkeypatch):
    backend,events=post_backend(monkeypatch)
    result={"success":True,"termination_reason":None}
    evaluator=NS(snapshot=result,observe=lambda raw:(events.append("evaluate") or result))
    controller=backend._controller
    row=video.common_post_success_tick(backend,evaluator,episode_ticks=31,post_index=1)
    assert backend._controller is controller and controller.termination=="LEGACY_RUNNING"
    assert controller.physics_tick==31 and backend._episode_tick==31 and backend._done is False
    assert row["task_credit"] is False and row["controller_step_count"]==0
    assert row["root_state_write_count"]==0
    assert [e[0] if isinstance(e,tuple) else e for e in events]==[
        "atomic","step","readback","read","validate","evaluate"]
    assert events[0][1]==(2.,)*8+(0.,)*4
    assert events[0][2]["drive_feedback_bias_full12"]==(.75,)*8+(0.,)*4
    assert events[0][2]["tracking_servo_names"]==("front_left_hip",)
    assert row["atomic_ack"]["fsm_nominal_mapper_input_full12"]==[2.]*8+[0.]*4
    assert row["atomic_ack"]["combined_post_mapper_bias_full12"]==[.75]*8+[0.]*4
    assert events[0][2]["physics_tick"]==444
    assert events[3][1]["physics_tick"]==32



def test_missing_or_mismatched_receipt_fails_before_physics(monkeypatch):
    backend,events=post_backend(monkeypatch)
    backend._last_atomic_ack["combined_post_mapper_bias_full12"]=[99.]*12
    with pytest.raises(video.SemanticVideoError,match="actual adapter input"):
        video.common_post_success_tick(backend,NS(snapshot={"success":True}),
                                       episode_ticks=31,post_index=1)
    assert events==[]


def test_post_roll_retains_real_mapper_compensation_and_nonzero_post_mapper_residual(monkeypatch):
    import numpy as np
    from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
    from wlr50_clean.infrastructure.robot_adapter import RobotAdapter
    from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapper
    from wlr50_clean.ppo.isaac_fsm_backend import IsaacFSMBackend, build_residual_actuation_plan

    # Only articulation buffers/physics are fake. All nominal shaping, sampled
    # feedback, bounded post-mapper bias and physical target conversion are real.
    staged={"writes":0}
    robot=NS(data=NS(joint_pos=np.zeros((1,12),dtype=np.float32),
                    joint_vel=np.zeros((1,12),dtype=np.float32)),
        set_joint_position_target=lambda targets,joint_ids:staged.update(position=targets.copy()),
        set_joint_velocity_target=lambda targets,joint_ids:staged.update(velocity=targets.copy()),
        write_data_to_sim=lambda:staged.update(writes=staged["writes"]+1),
        update=lambda dt:None)
    adapter=RobotAdapter.__new__(RobotAdapter)
    adapter.robot=robot;adapter.physics_dt_s=1/120
    adapter.joint_map=NS(servo_ids=tuple(range(8)),wheel_ids=tuple(range(8,12)))
    adapter._standing_servo_tensor=np.zeros((1,8),dtype=np.float32)
    adapter.standing_pose_deg={name:0. for name in SERVO_ORDER}
    adapter.servo_target_mapper=ServoTargetMapper(adapter.standing_pose_deg)
    adapter._final_drive_servo_deg={name:0. for name in SERVO_ORDER}
    adapter.write_count=0;adapter._last_physics_tick=None;adapter.last_ack=None
    nominal=(10.,)+(0.,)*7+(.2,)*4
    residual=(1.5,)+(0.,)*7+(.03,)*4
    plan=build_residual_actuation_plan(tuple(a+b for a,b in zip(nominal,residual)),
        frozen_nominal_full12=nominal,drive_feedback_bias_full12=(.25,)+(0.,)*11,
        normal_drive_bias_full12=video.ZERO12)
    for tick in range(80):
        ack=adapter.apply_full12(nominal,physics_tick=tick,
            tracking_servo_names=("front_left_hip",),
            drive_feedback_bias_full12=plan.combined_post_mapper_bias_full12)
    assert ack["servo_tracking_compensation_deg"][0]>0.
    expected_targets=staged["position"].copy()
    expected_compensation=tuple(ack["servo_tracking_compensation_deg"])
    backend,events=post_backend(monkeypatch)
    backend._adapter=adapter
    backend._last_atomic_ack=plan.annotate_ack(ack)
    backend._previous_action_full12=plan.projected_applied_full12
    backend._atomic_apply=lambda *args,**kwargs:IsaacFSMBackend._atomic_apply(backend,*args,**kwargs)
    # Incoming controller frame intentionally differs from the last dispatch.
    backend._controller_frame.tracking_servo_names=("rear_right_hip",)
    result={"success":True,"termination_reason":None}
    for post_index in (1,2):
        row=video.common_post_success_tick(backend,NS(snapshot=result,observe=lambda raw:result),
            episode_ticks=31,post_index=post_index)
        assert np.array_equal(staged["position"],expected_targets)
        assert tuple(row["atomic_ack"]["servo_tracking_compensation_deg"])==expected_compensation
        assert row["atomic_ack"]["fsm_nominal_mapper_input_full12"]==list(nominal[:8]) + [0.]*4
        assert row["atomic_ack"]["combined_post_mapper_bias_full12"]==list(plan.combined_post_mapper_bias_full12[:8])+[0.]*4
        assert row["atomic_ack"]["tracking_servo_names"]==["front_left_hip"]
        assert np.count_nonzero(staged["velocity"])==0
    assert adapter.write_count==82 and staged["writes"]==82
    assert len([event for event in events if isinstance(event,tuple) and event[0]=="step"])==2

def test_post_roll_losing_real_stability_is_diagnostic_failure(monkeypatch):
    backend,events=post_backend(monkeypatch)
    evaluator=NS(snapshot={"success":True},
                 observe=lambda raw:{"success":False,"termination_reason":None})
    with pytest.raises(video.SemanticVideoError,match="stability"):
        video.common_post_success_tick(backend,evaluator,episode_ticks=31,post_index=1)
    assert backend._done is False
    assert len([e for e in events if isinstance(e,tuple) and e[0]=="step"])==1


def test_post_roll_requires_common_success_before_dispatch(monkeypatch):
    backend,events=post_backend(monkeypatch)
    with pytest.raises(video.SemanticVideoError,match="common success"):
        video.common_post_success_tick(backend,NS(snapshot={"success":False}),
                                       episode_ticks=31,post_index=1)
    assert events==[]


def test_evidence_hash_change_rejected(tmp_path):
    source=tmp_path/"source"
    source.mkdir()
    path=source/"raw"
    path.write_text("first")
    record=video.file_record(path)
    assert video.inside(source,record)==path.resolve()
    path.write_text("changed")
    with pytest.raises(video.SemanticVideoError,match="hash"):
        video.inside(source,record)


def test_evidence_path_escape_rejected(tmp_path):
    root=tmp_path/"source";root.mkdir()
    outside=tmp_path/"outside";outside.write_text("x")
    with pytest.raises(video.SemanticVideoError,match="escaped"):
        video.inside(root,video.file_record(outside))


@pytest.mark.parametrize("role,name,allowed",[
    ("A","fsm_original_baseline.mp4",True),
    ("A","semantic_A_success.mp4",True),
    ("B","semantic_B_success.mp4",True),
    ("C","semantic_C_candidate.mp4",True),
    ("C","semantic_C_success.mp4",False),
    ("C","ppo_semantic_improved.mp4",False),
    ("A","ppo_semantic_improved.mp4",False),
])
def test_publication_names_do_not_turn_task_success_into_improvement(role,name,allowed):
    assert video.publication_name_allowed(role,name) is allowed


def test_requested_baseline_filename_cannot_skip_physical_validation(tmp_path,monkeypatch):
    def reject(_source):
        raise video.SemanticVideoError("failed physical source")
    monkeypatch.setattr(video,"validate_semantic_video_source",reject)
    with pytest.raises(video.SemanticVideoError,match="failed physical"):
        video.publish_success_source(tmp_path,tmp_path/"fsm_original_baseline.mp4")


def test_no_improved_filename_from_capture_alone(tmp_path,monkeypatch):
    monkeypatch.setattr(video,"validate_semantic_video_source",
                        lambda source:({"role":"C"},None))
    with pytest.raises(video.SemanticVideoError,match="improvement"):
        video.publish_success_source(tmp_path,tmp_path/"ppo_improved.mp4")


def test_video_args_reject_headless_or_training_before_live(monkeypatch,tmp_path):
    from wlr50_clean.ppo import semantic_video_cli
    monkeypatch.setattr(semantic_video_cli,"validate_request",lambda args:None)
    argv=["eval","--run-dir",str(tmp_path),"--expected-head","a"*40,
          "--seed","4001","--mode","semantic_prior_eval"]
    with pytest.raises(video.SemanticVideoError,match="no-headless"):
        validate_video_args(parser().parse_args(argv))
    assert validate_video_args(parser().parse_args(argv+["--no-headless"]))=="B"
    bad=parser().parse_args(argv+["--no-headless"]);bad.command="train"
    with pytest.raises(video.SemanticVideoError):
        validate_video_args(bad)


def test_refresh_recreates_only_semantic_pre_episode_state(monkeypatch):
    @dataclass
    class Frame:
        physics_tick:int
        state_id:str
        nominal_action_full12:tuple
        info:dict
    events=[]
    old_controller=NS()
    old_frame=Frame(0,"P01",video.ZERO12,{"raw_observation":{"current":True},
        "video_pre_action_refresh":{"physical_pre_action_ticks":64}})
    backend=NS(_controller=old_controller,_controller_frame=object(),_episode_tick=0,
        _committed_reset_generation=7,_reset_count=1,_adapter=object(),
        fsm_path="frozen_fsm",motion_contract_path="frozen_motion",
        refresh_video_pre_action_frame=lambda:(events.append("refresh_current_reader") or old_frame))
    fresh_controller=NS()
    fresh_controller.step=lambda raw,**kw:(events.append(("new_controller_tick0",raw,kw)) or old_frame)
    monkeypatch.setattr(video.SemanticControllerAdapter,"from_paths",
                        lambda *a,**kw:fresh_controller)
    fresh_info={"drive_target_full12":video.ZERO12,"semantic_task":{"schema":"wlr50_clean.semantic_task.v2","stage_id":"P01",
        "transition_evidence":[],"active_lift_history":[False]*4,"completed_stage_ids":[],"phase_progress":0.,"task_progress_potential":0.,"success":False,"substage":"TRANSFER"}}
    backend._build_authoritative_frame=lambda *a,**kw:Frame(0,"P01",video.ZERO12,fresh_info)
    builder=NS(reset=lambda:events.append("reset_encoder"),
               build=lambda frame,history:NS(groups={}))
    core=NS(backend=backend,decision_count=0,frame=old_frame,done=False,
            observation_builder=builder,
            observation_schema=NS(encode=lambda groups:(0.,)*324),
            bridge=NS(reset=lambda **kw:events.append("reset_bridge")))
    obs,info=video.refresh_semantic_core_after_preroll(core)
    assert len(obs)==324 and backend._controller is fresh_controller
    assert old_controller.__dict__=={}
    assert backend._committed_reset_generation==7 and backend._reset_count==1
    assert core.frame.physics_tick==0 and core._transition_evidence_count==0
    assert info["semantic_video_initialization"]["prefix_credit_imported"] is False
    assert [e if isinstance(e,str) else e[0] for e in events]==[
        "refresh_current_reader","new_controller_tick0","reset_encoder","reset_bridge"]

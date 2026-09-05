"""Task-first entry/finish tests using complete synthetic sensor observations."""
from copy import deepcopy
from pathlib import Path

import pytest

from wlr50_clean.ppo.semantic_supervisor import (
    DEFAULT_TASK_SPEC_PATH, GOAL_FEATURE_KEYS, LEG_ORDER, NominalMotionProvider,
    SemanticControllerAdapter, SemanticObservationError, TaskEvaluator,
    TaskStageSupervisor, load_task_spec,
)
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.reference.motion_contract import load_motion_contract
from wlr50_clean.fsm.state_spec import Lifecycle

PROJECT=Path(__file__).resolve().parents[2]


def observation(tick=0):
    result={"physics_tick":tick,"simulation_time_s":tick/120.,"all_finite":True,
        "base":{"position_w_m":[.3,0.,.11],"orientation_wxyz":[1.,0.,0.,0.],
                "linear_velocity_w_m_s":[0.,0.,0.],"angular_velocity_w_rad_s":[0.,0.,0.]},
        "imu":{"projected_gravity_b":[0.,0.,-1.]},
        "obstacle":{"front_x_m":.5,"back_x_m":2.,"left_y_m":.8,"right_y_m":-.8,"top_z_m":.05},
        "body_collision":{"detected":False},
        "joints":{name:{"position_deg":0.,"velocity_deg_s":0.} for name in SERVO_ORDER},
        "wheels":{},"contacts":{}}
    for index,name in enumerate(WHEEL_ORDER):
        body=name.replace("ankle","wheel")
        result["wheels"][name]={"body_name":body,"geometry_verified":True,
            "center_w_m":[.4,index*.1,.05],"bottom_w_m":[.4,index*.1,0.],
            "velocity_rad_s":0.,"command_rad_s":0.}
        result["contacts"][body]={"ground":{"active":True,"pair_verified":True,"normal_force_n":7.},
                                  "obstacle":{"active":False,"pair_verified":True,"normal_force_n":0.}}
    return result


def advance(obs):
    result=deepcopy(obs); result["physics_tick"]+=1
    result["simulation_time_s"]=result["physics_tick"]/120.
    return result


def leg_state(obs,leg,*,x=None,bottom=None,air=False,top=False,hip=None):
    i=LEG_ORDER.index(leg); w=obs["wheels"][WHEEL_ORDER[i]]
    if x is not None:w["center_w_m"][0]=x;w["bottom_w_m"][0]=x
    if bottom is not None:w["bottom_w_m"][2]=bottom;w["center_w_m"][2]=bottom+.05
    c=obs["contacts"][w["body_name"]]
    c["ground"].update(active=not air and not top,normal_force_n=0. if air or top else 7.)
    c["obstacle"].update(active=top,normal_force_n=7. if top else 0.)
    if hip is not None:obs["joints"][SERVO_ORDER[2*i]]["position_deg"]=hip


def place(ev,obs,leg):
    for data in ({"bottom":.012,"air":True,"hip":1.5},
                 {"bottom":.020,"air":True,"hip":3.},
                 {"x":.53,"bottom":.075,"air":True,"hip":4.},
                 {"x":.53,"bottom":.05,"top":True},
                 {"x":.53,"bottom":.05,"top":True}):
        obs=advance(obs);leg_state(obs,leg,**data);ev.observe(obs)
    assert ev.snapshot["history"]["placed"][leg]
    return obs


def rr_placed():
    ev=TaskEvaluator();obs=observation();ev.observe(obs)
    for leg in ("FR","FL","RR"):obs=place(ev,obs,leg)
    return ev,obs


@pytest.fixture
def p02_airborne_sensor_prefix():
    """Complete sensor fixture; lift is earned through the real evaluator API.

    Only the final FR horizontal position varies in the paired tests. These
    synthetic contiguous measurements are not a claim of a new physical run.
    """
    obs=observation()
    obs["obstacle"]["front_x_m"]=.5213121737735307
    front=obs["obstacle"]["front_x_m"]
    leg_state(obs,"FR",x=front-.15)
    frames=[obs]
    for tick in range(1,9):
        obs=advance(obs)
        leg_state(obs,"FR",bottom=.075*tick/8,air=True,hip=3.*tick/8)
        frames.append(obs)
    return frames


@pytest.mark.parametrize("front_distance,completed",[
    (-.15,False),(-.005001,False),(-.005,True),(-.001,True),
])
def test_p02_descent_handoff_requires_front_edge_capture_geometry(
        p02_airborne_sensor_prefix,front_distance,completed):
    frames=deepcopy(p02_airborne_sensor_prefix)
    final=frames[-1]
    leg_state(final,"FR",x=final["obstacle"]["front_x_m"]+front_distance,air=True)
    sup=TaskStageSupervisor(initial_stage_id="P02")
    for obs in frames:snap=sup.observe_and_update(obs)
    evaluation=sup.evaluator.snapshot
    assert evaluation["history"]["active_lift"]["FR"]
    assert not evaluation["history"]["front_edge_crossed"]["FR"]
    assert sup.predicate("lifted_FR",evaluation)==1.
    assert sup.predicate("clear_FR",evaluation)==1.
    assert evaluation["current_legs"]["FR"]["clearance_m"]==pytest.approx(.025)
    assert evaluation["current_legs"]["FR"]["front_distance_m"]==pytest.approx(front_distance)
    assert snap["stage_id"]==("P03" if completed else "P02")
    assert snap["completed_stage_ids"]==(["P02"] if completed else [])
    assert snap["termination_reason"] is None and not snap["success"]
    if completed:
        assert snap["transition_evidence"][0]["completion_values"]=={
            "lifted_FR":1.,"clear_FR":1.,"approach_FR":1.}
    else:
        assert snap["completion_values"]["approach_FR"]<1.
        assert not snap["transition_evidence"]


@pytest.mark.parametrize("missing_requirement",["lift","clearance"])
def test_p02_front_edge_geometry_does_not_replace_lift_or_clearance(
        p02_airborne_sensor_prefix,missing_requirement):
    frames=deepcopy(p02_airborne_sensor_prefix)
    if missing_requirement=="lift":
        for obs in frames:
            obs["joints"]["front_right_hip"]["position_deg"]=0.
    final=frames[-1]
    leg_state(final,"FR",x=final["obstacle"]["front_x_m"]-.005,air=True,
              bottom=.05 if missing_requirement=="clearance" else None)
    sup=TaskStageSupervisor(initial_stage_id="P02")
    for obs in frames:snap=sup.observe_and_update(obs)
    assert snap["completion_values"]["approach_FR"]==1.
    assert snap["completion_values"]["lifted_FR" if missing_requirement=="lift" else "clear_FR"]<1.
    assert snap["stage_id"]=="P02" and snap["completed_stage_ids"]==[]
    assert snap["termination_reason"] is None


@pytest.mark.parametrize("angle,velocity",[(-30.,0.),(-43.,-3.),(-50.3975984,0.),(10.,-20.)])
def test_p10_entry_accepts_actual_rr_placement_without_historical_rebound(angle,velocity):
    ev,obs=rr_placed();sup=TaskStageSupervisor(evaluator=ev,initial_stage_id="P10")
    obs=advance(obs)
    obs["joints"]["rear_right_knee"].update(position_deg=angle,velocity_deg_s=velocity)
    leg_state(obs,"RL",x=-.1)  # Workspace does NOT already exist at entry.
    snap=sup.observe_and_update(obs)
    assert snap["entry_valid"]
    assert snap["stage_id"]=="P10"
    assert snap["completion_values"]["workspace_RL"]<1.
    assert snap["termination_reason"] is None


def test_p10_historical_knee_values_do_not_replace_actual_rr_history():
    ev=TaskEvaluator();obs=observation();ev.observe(obs)
    for leg in ("FR","FL"):obs=place(ev,obs,leg)
    obs=advance(obs)
    obs["joints"]["rear_right_knee"].update(position_deg=-50.397598397883456,velocity_deg_s=23.585333053160202)
    sup=TaskStageSupervisor(evaluator=ev,initial_stage_id="P10")
    snap=sup.observe_and_update(obs)
    assert not snap["entry_valid"] and "placed_RR" in snap["entry_reasons"]
    assert snap["completed_stage_ids"]==[] and not snap["success"]


def test_p10_collision_rejected_even_with_history_and_exact_historical_knee():
    ev,obs=rr_placed();obs=advance(obs);obs["body_collision"]["detected"]=True
    obs["joints"]["rear_right_knee"].update(position_deg=-50.3976,velocity_deg_s=23.5853)
    snap=TaskStageSupervisor(evaluator=ev,initial_stage_id="P10").observe_and_update(obs)
    assert not snap["entry_valid"]
    assert snap["termination_reason"]=="TASK_FAILURE_BODY_COLLISION"


@pytest.mark.parametrize("missing",["contacts","wheels","imu"])
def test_p10_needs_complete_current_physical_observation(missing):
    ev,obs=rr_placed();obs=advance(obs);del obs[missing]
    with pytest.raises(SemanticObservationError):
        TaskStageSupervisor(evaluator=ev,initial_stage_id="P10").observe_and_update(obs)


def test_p10_completion_waits_for_workspace_then_handoffs_without_velocity_gate():
    ev,obs=rr_placed();sup=TaskStageSupervisor(evaluator=ev,initial_stage_id="P10")
    for _ in range(9):
        obs=advance(obs)
        leg_state(obs,"RL",x=.4 if obs["physics_tick"]==24 else -.1)
        snap=sup.observe_and_update(obs)
    assert snap["stage_id"]=="P11"
    assert snap["completed_stage_ids"]==["P10"]
    assert snap["transition_evidence"][0]["completion_values"]=={"workspace_RL":1.,"support_RL":1.}


def test_raw_joint_command_or_air_without_measured_lift_never_credits_lift():
    ev=TaskEvaluator();obs=observation();ev.observe(obs)
    for _ in range(3):
        obs=advance(obs);leg_state(obs,"FR",air=True)
        obs["joints"]["front_right_hip"]["command_deg"]=90.
        ev.observe(obs)
    assert not ev.snapshot["history"]["active_lift"]["FR"]
    obs=advance(obs);leg_state(obs,"FR",x=.51,air=True)
    assert ev.observe(obs)["termination_reason"]=="TASK_FAILURE_WHEEL_ONLY_CLIMB"


def test_single_air_sample_is_not_active_lift():
    ev=TaskEvaluator();obs=observation();ev.observe(obs)
    obs=advance(obs);leg_state(obs,"FR",bottom=.02,air=True,hip=8.)
    assert not ev.observe(obs)["history"]["active_lift"]["FR"]


def test_rr_first_order_is_actual_history_not_phase_name():
    ev=TaskEvaluator();obs=observation();ev.observe(obs)
    for step in range(1,4):
        obs=advance(obs);leg_state(obs,"RL",x=.53 if step==3 else .4,bottom=.02*step,air=True,hip=2.*step)
        evaluation=ev.observe(obs)
    assert evaluation["termination_reason"]=="INCOMPLETE_CONTROLLER_BLOCKED"
    assert "RR_FIRST" in evaluation["reason"]
    assert not evaluation["history"]["placed"]["RL"]


def test_evaluator_requires_final_region_home_and_stable_physical_stop():
    ev,obs=rr_placed();obs=place(ev,obs,"RL")
    assert not ev.snapshot["success"]
    for _ in range(63):
        obs=advance(obs);obs["base"]["position_w_m"][0]=.9
        for leg in LEG_ORDER:leg_state(obs,leg,x=.8,bottom=.05,top=True,hip=0.)
        snap=ev.observe(obs)
    assert snap["success"]
    assert tuple(snap["goal_features"])==GOAL_FEATURE_KEYS
    assert all(snap["history"]["placed"].values())


def test_final_static_placement_without_active_history_is_not_success():
    obs=observation();obs["base"]["position_w_m"][0]=.9
    for leg in LEG_ORDER:leg_state(obs,leg,x=.8,bottom=.05,top=True)
    snap=TaskEvaluator().observe(obs)
    assert not snap["success"]
    assert snap["termination_reason"]=="TASK_FAILURE_WHEEL_ONLY_CLIMB"


@pytest.mark.parametrize("command,accepted",[(.01,True),(.2,False)])
def test_physical_stop_tolerance_is_not_an_exact_zero_actor_requirement(command,accepted):
    ev,obs=rr_placed();obs=place(ev,obs,"RL")
    for _ in range(63):
        obs=advance(obs);obs["base"]["position_w_m"][0]=.9
        for leg in LEG_ORDER:leg_state(obs,leg,x=.8,bottom=.05,top=True,hip=0.)
        for wheel in obs["wheels"].values():wheel["command_rad_s"]=command
        snap=ev.observe(obs)
    assert snap["success"] is accepted
    assert snap["maximum_commanded_wheel_speed_rad_s"]==command


@pytest.mark.parametrize("outlier",["wheel_y","wheel_back","base_y","base_back"])
def test_final_top_height_alone_cannot_fake_obstacle_region(outlier):
    ev,obs=rr_placed();obs=place(ev,obs,"RL")
    for _ in range(63):
        obs=advance(obs);obs["base"]["position_w_m"][0]=.9
        for leg in LEG_ORDER:leg_state(obs,leg,x=.8,bottom=.05,top=True,hip=0.)
        if outlier=="wheel_y":obs["wheels"][WHEEL_ORDER[0]]["center_w_m"][1]=2.
        if outlier=="wheel_back":obs["wheels"][WHEEL_ORDER[0]]["center_w_m"][0]=2.1
        if outlier=="base_y":obs["base"]["position_w_m"][1]=2.
        if outlier=="base_back":obs["base"]["position_w_m"][0]=2.1
        snap=ev.observe(obs)
    assert not snap["final_region_valid"] and not snap["success"]


def test_unfinished_task_ends_at_declared_task_deadline_not_legacy_half_second():
    sup=TaskStageSupervisor();sup.spec["stages"]["P01"]["maximum_task_duration"]=.1
    for tick in range(13):snap=sup.observe_and_update(observation(tick))
    assert snap["termination_reason"]=="INCOMPLETE_CONTROLLER_BLOCKED"
    assert not snap["success"] and snap["completed_stage_ids"]==[]


def test_history_reset_does_not_inherit_success_from_another_evaluator():
    ev,_=rr_placed()
    assert ev.snapshot["history"]["placed"]["RR"]
    fresh=TaskEvaluator().observe(observation())
    assert not any(fresh["history"]["placed"].values())


def test_repeated_same_observation_cannot_mine_stage_credit():
    ev,obs=rr_placed();obs=advance(obs)
    leg_state(obs,"RL",air=True)
    sup=TaskStageSupervisor(evaluator=ev,initial_stage_id="P10")
    first=sup.observe_and_update(obs)
    assert first["stage_id"]=="P11"
    for _ in range(20):
        again=sup.observe_and_update(obs)
        assert again==first
    assert sup.completed_stage_ids==["P10"]


def test_final_stop_without_current_physical_support_never_succeeds():
    ev,obs=rr_placed();obs=place(ev,obs,"RL")
    for _ in range(63):
        obs=advance(obs);obs["base"]["position_w_m"][0]=.9
        for leg in LEG_ORDER:leg_state(obs,leg,x=.8,bottom=.05,air=True,hip=0.)
        snap=ev.observe(obs)
    assert not snap["success"] and not snap["final_support_available"]


def test_missing_or_unverified_geometry_is_not_feasibility():
    obs=observation();obs["wheels"]["rear_right_ankle"]["geometry_verified"]=False
    with pytest.raises(SemanticObservationError,match="geometry"):
        TaskEvaluator().observe(obs)


def test_nonfinite_and_hard_limits_remain_terminal():
    obs=observation();obs["all_finite"]=False
    assert TaskEvaluator().observe(obs)["termination_reason"]=="SAFETY_ABORT"
    obs=observation();obs["joints"]["rear_right_knee"]["position_deg"]=211.
    assert TaskEvaluator().observe(obs)["termination_reason"]=="SAFETY_ABORT"


def test_goal_history_requires_contiguous_real_observation_ticks():
    ev=TaskEvaluator();ev.observe(observation())
    with pytest.raises(SemanticObservationError,match="contiguous"):
        ev.observe(observation(2))


def test_stage_spec_covers_all_tasks_without_reference_predicates():
    cfg=load_task_spec()
    assert len(cfg["stages"])==13
    for phase,row in cfg["stages"].items():
        assert len(row["allowed_action_channels"])==12
        assert "reference" not in " ".join(row["valid_start_conditions"]+row["completion_predicates"])
    assert not any("workspace" in item for item in cfg["stages"]["P10"]["valid_start_conditions"])


def test_nominal_handoff_is_continuous_and_does_not_reapply_increment():
    contract=load_motion_contract(PROJECT/"configs/recording_motion_contract.json")
    provider=NominalMotionProvider(contract)
    for _ in range(40):provider.evaluate("P09")
    previous=provider.nominal_full12
    assert provider.evaluate("P10")==previous
    for _ in range(100):last=provider.evaluate("P10")
    for _ in range(100):again=provider.evaluate("P10")
    assert last==again
    assert provider.endpoint_issued


def test_early_handoff_still_approaches_usable_absolute_advisory_not_a_lost_delta():
    contract=load_motion_contract(PROJECT/"configs/recording_motion_contract.json")
    provider=NominalMotionProvider(contract)
    provider.evaluate("P01")
    previous=provider.nominal_full12
    assert provider.evaluate("P02")==previous
    for _ in range(500):provider.evaluate("P02")
    assert provider.nominal_full12==contract.phase("P02").end_full12


def test_placement_progress_is_continuous_but_does_not_forge_completion():
    ev=TaskEvaluator();sup=TaskStageSupervisor(evaluator=ev,initial_stage_id="P03")
    obs=observation();before=ev.observe(obs)
    first=sup.predicate("placed_FR",before)
    obs=advance(obs);leg_state(obs,"FR",bottom=.012,air=True,hip=1.5)
    middle=sup.predicate("placed_FR",ev.observe(obs))
    assert first<middle<1.
    assert not ev.snapshot["history"]["placed"]["FR"]


def test_nominal_endpoint_does_not_complete_unfinished_task_or_stop_residual():
    controller=SemanticControllerAdapter.from_paths(PROJECT/"configs/fsm_states.yaml",PROJECT/"configs/recording_motion_contract.json")
    obs=observation()
    # Initial evenly loaded FR is not load-ready. The legacy P01 tail is not a goal.
    for tick in range(1800):
        obs["physics_tick"]=tick;obs["simulation_time_s"]=tick/120.
        frame=controller.step(obs,sim_time_s=tick/120.)
    assert controller.nominal_provider.endpoint_issued
    assert frame.state_id=="P01" and frame.lifecycle is Lifecycle.EXECUTE_MOTION
    assert frame.termination is None
    assert controller.task_progress<1.


def test_adapter_produces_full12_and_preserves_public_rate_contract():
    controller=SemanticControllerAdapter.from_paths(PROJECT/"configs/fsm_states.yaml",PROJECT/"configs/recording_motion_contract.json")
    frame=controller.step(observation(),sim_time_s=0.)
    assert frame.physics_tick==0 and frame.full12_atomic_write_required
    assert frame.drive_feedback_bias_full12==(0.,)*12
    assert frame.normal_drive_bias_full12==(0.,)*12
    assert controller.motion.physics_hz==120. and controller.motion.servo_rate_limit_deg_s==150.
    assert controller.phase.macro_phase==1
    assert "physical_evaluator" in controller.task_snapshot

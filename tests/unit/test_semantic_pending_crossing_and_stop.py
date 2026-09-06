"""Boundary counterexamples; synthetic measurements, not new physical runs."""
from copy import deepcopy
from pathlib import Path

import pytest

from test_semantic_supervisor import observation, advance, leg_state, place
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator, TaskStageSupervisor, load_task_spec

ROOT=Path(__file__).resolve().parents[2]
V3=ROOT/'configs/ppo_semantic_v3/stage_task_spec.yaml'


def live():
    obs=observation()
    for joint in obs['joints'].values():joint['command_deg']=0.
    return obs


def qualified(ev,leg='FL'):
    obs=live();ev.observe(obs)
    for bottom,angle in ((.004,1.5),(.009,3.),(.05026418027465104,4.)):
        obs=advance(obs);leg_state(obs,leg,bottom=bottom,air=True,hip=angle)
        ev.observe(obs)
    assert ev.snapshot['history']['active_lift'][leg]
    return obs


@pytest.mark.parametrize('consumer',['direct','semantic_controller','shared_A_B_C_recorder'])
def test_real_fl_boundary_values_remain_pending_until_actual_crossing(consumer,tmp_path):
    # 5036e116 training P05 tick2650: +0.178519mm front, -0.104683mm
    # gap, AIR without either contact, with earned unrevoked lift. Only the
    # boundary values are replayed here; prior history is earned via observe.
    recorder=None
    if consumer=='shared_A_B_C_recorder':
        from wlr50_clean.ppo.semantic_legacy_evaluation import PhysicalEvaluationRecorder
        recorder=PhysicalEvaluationRecorder(tmp_path,task_spec_path=V3)
        ev=recorder.evaluator
    elif consumer=='semantic_controller':ev=TaskStageSupervisor(V3).evaluator
    else:ev=TaskEvaluator(V3)
    try:
        obs=qualified(ev)
        for _ in range(24):
            obs=advance(obs)
            leg_state(obs,'FL',x=.5001785193188155,bottom=.04989531693045854,air=True,hip=4.)
            snap=ev.observe(obs)
            assert snap['current_legs']['FL']['crossing_geometry_pending']
            assert snap['termination_reason'] is None
            assert not snap['history']['front_edge_crossed']['FL']
            assert not snap['history']['placed']['FL'] and not snap['success']
        # Current above-top AIR can finally establish crossing, but not load.
        obs=advance(obs);leg_state(obs,'FL',x=.501,bottom=.0501,air=True)
        snap=ev.observe(obs)
        assert snap['history']['front_edge_crossed']['FL']
        assert not snap['current_legs']['FL']['crossing_geometry_pending']
        assert not snap['history']['placed']['FL']
        for _ in range(2):
            obs=advance(obs);leg_state(obs,'FL',x=.501,bottom=.05,top=True)
            snap=ev.observe(obs)
        assert snap['history']['placed']['FL'] and not snap['success']
    finally:
        if recorder is not None:recorder.close()


@pytest.mark.parametrize('fault',['ground_revoked','side_bypass','RL_before_RR'])
def test_pending_does_not_waive_ground_roi_or_rear_order(fault):
    leg='RL' if fault=='RL_before_RR' else 'FL'
    ev=TaskEvaluator(V3);obs=qualified(ev,leg)
    if fault=='ground_revoked':
        obs=advance(obs);leg_state(obs,leg,x=.49,bottom=0.)
        assert not ev.observe(obs)['history']['active_lift'][leg]
    obs=advance(obs);leg_state(obs,leg,x=.501,bottom=.0498953,air=True)
    if fault=='side_bypass':
        wheel=obs['wheels'][('rear_left' if leg=='RL' else 'front_left')+'_ankle']
        wheel['center_w_m'][1]=wheel['bottom_w_m'][1]=.9
    snap=ev.observe(obs)
    assert snap['termination_reason'] is not None
    assert not snap['current_legs'][leg]['crossing_geometry_pending']
    assert not snap['history']['front_edge_crossed'][leg]
    assert not snap['history']['placed'][leg]


def test_actual_rr_loaded_crossing_without_qualified_lift_remains_rejected():
    # cp12672 fresh P01 eval reached RR obstacle load with only an initial
    # clearance event, not above-top qualification. Do not grant the exemption.
    ev=TaskEvaluator(V3);obs=live();ev.observe(obs)
    for bottom,angle in ((.004,1.5),(.009,3.),(.012,4.)):
        obs=advance(obs);leg_state(obs,'RR',bottom=bottom,air=True,hip=angle);ev.observe(obs)
    assert ev.snapshot['current_legs']['RR']['initial_clearance']
    assert not ev.snapshot['history']['active_lift']['RR']
    obs=advance(obs);leg_state(obs,'RR',x=.501,bottom=.049,top=True)
    snap=ev.observe(obs)
    assert snap['termination_reason']=='TASK_FAILURE_WHEEL_ONLY_CLIMB'
    assert not snap['history']['front_edge_crossed']['RR']
    assert not snap['history']['placed']['RR']


def placed_all(spec_path=V3):
    ev=TaskEvaluator(spec_path);obs=live();ev.observe(obs)
    for leg in ('FR','FL','RR','RL'):obs=place(ev,obs,leg)
    return ev,obs


POSES=((12.,-18.,20.,31.,-12.,24.,16.,-20.),
       (-24.,32.,-16.,18.,35.,-22.,-27.,29.))


def stopped_observation(obs,pose):
    obs=advance(obs);obs['base']['position_w_m'][0]=.9
    for leg in ('FR','FL','RR','RL'):leg_state(obs,leg,x=.8,bottom=.05,top=True)
    for name,value in zip(SERVO_ORDER,pose):
        obs['joints'][name].update(position_deg=value,command_deg=value,velocity_deg_s=0.)
    return obs


def test_two_distinct_safe_joint_configs_have_same_physical_final_potential_and_success():
    traces=[]
    for pose in POSES:
        ev,obs=placed_all();sup=TaskStageSupervisor(V3,evaluator=ev,initial_stage_id='P13')
        trace=[]
        for _ in range(64):
            obs=stopped_observation(obs,pose);snap=ev.observe(obs)
            trace.append((sup.predicate('whole_task_success',snap),sup.physical_potential(snap),snap['success']))
        assert snap['success'] and snap['final_controlled']
        assert snap['home_maximum_servo_error_deg']>ev.spec['final']['home_tolerance_deg']
        assert trace[0][2] is False  # Required physical settle time is retained.
        traces.append(trace)
    assert traces[0]==traces[1]


@pytest.mark.parametrize('fault',['body_speed','body_rotation','wheel_speed','wheel_command',
    'support','region','joint_limit','fall','collision'])
def test_nonhome_physical_stop_still_rejects_real_bad_conditions(fault):
    ev,obs=placed_all()
    for _ in range(64):
        obs=stopped_observation(obs,POSES[0])
        if fault=='body_speed':obs['base']['linear_velocity_w_m_s']=[.06,0.,0.]
        if fault=='body_rotation':obs['base']['angular_velocity_w_rad_s']=[.31,0.,0.]
        if fault=='wheel_speed':obs['wheels']['front_left_ankle']['velocity_rad_s']=.26
        if fault=='wheel_command':obs['wheels']['front_left_ankle']['command_rad_s']=.03
        if fault=='support':
            for contact in obs['contacts'].values():contact['obstacle']['normal_force_n']=0.
        if fault=='region':obs['base']['position_w_m'][0]=.55
        if fault=='joint_limit':obs['joints']['front_left_hip']['position_deg']=136.
        if fault=='fall':obs['base']['position_w_m'][2]=.01
        if fault=='collision':obs['body_collision']['detected']=True
        snap=ev.observe(obs)
    assert not snap['success']


def test_v2_retains_home_stop_and_pose_potential_by_default():
    old=ROOT/'configs/ppo_semantic_v2/stage_task_spec.yaml'
    ev,obs=placed_all(old);sup=TaskStageSupervisor(old,evaluator=ev,initial_stage_id='P13')
    for _ in range(64):
        obs=stopped_observation(obs,POSES[0]);snap=ev.observe(obs)
    assert not snap['success'] and not snap['final_controlled']
    other=deepcopy(snap);other['home_maximum_servo_error_deg']=0.
    assert sup.predicate('whole_task_success',other)>sup.predicate('whole_task_success',snap)


def test_unknown_final_pose_semantics_fails_closed(tmp_path):
    import yaml
    spec=load_task_spec(V3);spec['final']['stop_pose_semantics']='ignore_physics'
    path=tmp_path/'invalid.yaml';path.write_text(yaml.safe_dump(spec,sort_keys=False))
    with pytest.raises(ValueError,match='stop pose semantics'):load_task_spec(path)


def test_unknown_crossing_semantics_fails_closed(tmp_path):
    import yaml
    spec=load_task_spec(V3);spec['history']['crossing_evidence_semantics']='allow_any_air'
    path=tmp_path/'invalid-crossing.yaml';path.write_text(yaml.safe_dump(spec,sort_keys=False))
    with pytest.raises(ValueError,match='crossing evidence semantics'):load_task_spec(path)

"""Focused physical counterexamples for continuous whole-body exploration."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from test_semantic_supervisor import observation, advance, leg_state, place
from test_semantic_observation_reward_env import _built, _frame, _sample, Backend
from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator, TaskStageSupervisor, NominalMotionProvider, load_task_spec
from wlr50_clean.ppo.semantic_reward import SemanticRewardCalculator, load_semantic_reward_config
from wlr50_clean.ppo.semantic_backend import build_semantic_projector
from wlr50_clean.ppo.semantic_env import SemanticEpisodeEnv
from wlr50_clean.ppo.phase_action_masks_v2 import PhaseTransitionBridge
from wlr50_clean.reference.motion_contract import load_motion_contract

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/'configs/ppo_semantic_v3'


def live(tick=0):
    obs=observation(tick)
    for joint in obs['joints'].values(): joint['command_deg']=0.
    return obs


def evaluator(): return TaskEvaluator(CFG/'stage_task_spec.yaml')


def whole_body_lift(ev,obs,leg='RR',offset=0.,top_clearance=.060):
    for bottom,angle in ((.004,1.5),(.009,3.),(top_clearance,4.)):
        obs=advance(obs); leg_state(obs,leg,bottom=bottom,air=True)
        obs['joints']['front_left_hip'].update(position_deg=offset+angle,command_deg=offset+angle+1.)
        ev.observe(obs)
    return obs


def test_whole_body_initial_clearance_does_not_require_rr_joint_motion_or_count_crossing():
    ev=evaluator(); obs=live(); ev.observe(obs)
    obs=whole_body_lift(ev,obs,top_clearance=.012)
    current=ev.snapshot['current_legs']['RR']
    assert current['initial_clearance'] and current['recent_joint_motion_deg']==0.
    assert not ev.snapshot['history']['active_lift']['RR']
    assert not ev.snapshot['history']['front_edge_crossed']['RR']
    assert not ev.snapshot['history']['placed']['RR']


@pytest.mark.parametrize('support_offset',[-20.,15.])
def test_two_nonhistorical_whole_body_entries_earn_same_qualification(support_offset):
    ev=evaluator(); obs=live(); obs['joints']['front_left_hip']['position_deg']=support_offset; ev.observe(obs)
    obs=whole_body_lift(ev,obs,offset=support_offset)
    assert ev.snapshot['history']['active_lift']['RR']
    assert ev.snapshot['current_legs']['RR']['recent_joint_motion_deg']==0.
    assert ev.snapshot['termination_reason'] is None


@pytest.mark.parametrize('invalid',['collision','fall','wheel_wall','no_motion','descent'])
def test_invalid_airborne_motion_never_earns_active_crossing(invalid):
    ev=evaluator(); obs=live()
    if invalid=='descent': leg_state(obs,'RR',bottom=.1,air=True)
    if invalid=='wheel_wall': leg_state(obs,'RR',bottom=0.,top=True)
    ev.observe(obs)
    for i in range(1,4):
        obs=advance(obs); leg_state(obs,'RR',bottom=.1-i*.01 if invalid=='descent' else .02*i,air=True)
        if invalid not in ('no_motion','wheel_wall'): obs['joints']['front_left_hip'].update(position_deg=2.*i,command_deg=3.*i)
        if invalid=='wheel_wall':
            for wheel in obs['wheels'].values(): wheel.update(command_rad_s=.3,velocity_rad_s=.3)
        if invalid=='collision': obs['body_collision']['detected']=True
        if invalid=='fall': obs['imu']['projected_gravity_b']=[0.,0.,1.]
        ev.observe(obs)
    assert not ev.snapshot['history']['active_lift']['RR']


def test_obstacle_contact_does_not_forbid_fresh_actuated_corrective_lift():
    ev=evaluator(); obs=live(); leg_state(obs,'RR',bottom=.001,top=True); ev.observe(obs)
    obs=whole_body_lift(ev,obs)
    assert ev.snapshot['history']['active_lift']['RR']
    assert ev.snapshot['termination_reason'] is None


def test_ground_revokes_current_whole_body_qualification_but_retry_keeps_history():
    ev=evaluator(); obs=live(); ev.observe(obs); obs=whole_body_lift(ev,obs)
    obs=advance(obs); leg_state(obs,'RR',bottom=0.); ev.observe(obs)
    assert not ev.snapshot['history']['active_lift']['RR']
    assert not ev.snapshot['current_legs']['RR']['initial_clearance']
    count=len(ev.snapshot['history']['lift_attempt_events'])
    obs=whole_body_lift(ev,obs,offset=4.)
    assert ev.snapshot['history']['active_lift']['RR']
    assert len(ev.snapshot['history']['lift_attempt_events'])>count


def test_fl_preparation_air_preserves_placement_not_current_load():
    ev=evaluator(); obs=live(); ev.observe(obs)
    obs=place(ev,obs,'FR'); obs=place(ev,obs,'FL')
    obs=advance(obs); leg_state(obs,'FL',bottom=.08,air=True); snap=ev.observe(obs)
    assert snap['history']['placed']['FL']
    assert snap['current_legs']['FL']['air']
    assert not snap['current_legs']['FL']['support']
    assert snap['current_legs']['FL']['load_fraction']==0.
    assert snap['termination_reason'] is None


def test_p08_air_history_survives_p09_handoff_and_phi_does_not_depend_on_label():
    ev=evaluator(); obs=live(); ev.observe(obs)
    obs=place(ev,obs,'FR'); obs=place(ev,obs,'FL'); obs=whole_body_lift(ev,obs)
    sup=TaskStageSupervisor(CFG/'stage_task_spec.yaml',evaluator=ev,initial_stage_id='P08')
    while obs['physics_tick']%8 or sup.stage_id=='P08':
        obs=advance(obs); snap=sup.observe_and_update(obs)
    assert snap['stage_id']=='P09' and snap['termination_reason'] is None
    assert snap['history']['active_lift']['RR']
    value=sup.physical_potential(ev.snapshot)
    sup.stage_id='P10'
    assert sup.physical_potential(ev.snapshot)==value


def task(stage,*,front=-.2,clear=.03,lift=True):
    return {'stage_id':stage,'termination_reason':None,'physical_evaluator':{
        'valid':True,'termination_reason':None,'history':{'active_lift':{'RR':lift,'RL':lift},'front_edge_crossed':{}},
        'current_legs':{'RR':{'clearance_m':clear,'front_distance_m':front},'RL':{'clearance_m':clear,'front_distance_m':front}}}}


def test_unfinished_p08_support_motion_continues_under_p09_without_old_entry_restore():
    contract=load_motion_contract(ROOT/'configs/recording_motion_contract.json')
    provider=NominalMotionProvider(contract,spec=load_task_spec(CFG/'stage_task_spec.yaml'))
    for _ in range(8): provider.evaluate(task('P08',lift=False))
    before=provider.nominal_full12
    assert provider.evaluate(task('P09'))==before
    for _ in range(80): provider.evaluate(task('P09'))
    assert provider.nominal_full12[4]>before[4]+5.
    assert provider.nominal_full12[6]>20.
    # No timed knee descent while RR still far from the placement region.
    assert provider.nominal_full12[7]==pytest.approx(0.)
    assert min(provider.nominal_full12[8:])>0.
    for _ in range(20): provider.evaluate(task('P09',front=.0))
    assert provider.nominal_full12[7]<-1.


@pytest.mark.parametrize('pair',[('P01','P02'),('P02','P03'),('P05','P06'),('P07','P08'),('P08','P09'),('P09','P10'),('P10','P11'),('P11','P12'),('P12','P13')])
def test_phase_bridge_keeps_all_live_residual_channels_without_global_zero(pair):
    projector=build_semantic_projector(CFG/'execution_profile.yaml'); bridge=PhaseTransitionBridge(projector)
    before=(.25,)*8+(.01,)*4; bridge.reset(state_id=pair[0],projected_residual_full12=before,applied_action_full12=before)
    out=bridge.project_tick((0.,)*12,state_id=pair[1],nominal_action_full12=(0.,)*12,
                           reference_action_full12=(0.,)*12,reference_delta_full12=(0.,)*12)
    assert out.projection.safe_projected_residual_full12==pytest.approx(before)
    assert out.transition_metric.handoff_hold_used


def calc(): return SemanticRewardCalculator(load_semantic_reward_config(CFG/'reward_config.yaml'))


def contact_frame(tick,contact,vertical):
    frame=_built(_frame(tick))
    metrics=deepcopy(frame.metrics)
    for row in metrics['wheels']: row.update(contact=contact,vertical_velocity=vertical,chatter=.5,slip_speed=0.)
    return replace(frame,metrics=metrics)


def reward(calculator,a,b,**kwargs):
    sample=replace(_sample(a,b),**kwargs)
    return calculator.evaluate(a,b,[sample],termination_reason=None,task_success=False)


def test_initial_departure_is_not_rebound_but_post_touchdown_passive_rebound_is():
    a,b,c=contact_frame(0,False,-.4),contact_frame(1,True,0.),contact_frame(2,False,.5)
    initial=reward(calc(),b,c)
    assert initial['cost_components']['confirmed_post_touchdown_rebound']==0.
    calculator=calc(); reward(calculator,a,b)
    bounced=reward(calculator,b,c)
    assert bounced['cost_components']['confirmed_post_touchdown_rebound']>0.
    calculator=calc(); reward(calculator,a,b)
    acted=reward(calculator,b,c,actual_drive=(1.,)+(0.,)*11)
    assert acted['cost_components']['confirmed_post_touchdown_rebound']==0.
    calculator.reset()
    assert reward(calculator,b,c)['cost_components']['confirmed_post_touchdown_rebound']==0.


@pytest.mark.parametrize('command,passive',[
    ((0.,)*12,True),
    ((1.,)+(0.,)*11,False),
    ((0.,)*8+(.01,)+(0.,)*3,False),
])
def test_touchdown_tick_retains_real_whole_body_command_change(command,passive):
    a,b,c=contact_frame(0,False,-.4),contact_frame(1,True,0.),contact_frame(2,False,.5)
    calculator=calc()
    landing=reward(calculator,a,b,actual_drive=command,previous_actual_drive=(0.,)*12)
    # The target is now held, but the mechanism can still be responding to the
    # active command dispatched during touchdown. It is not a passive rebound.
    departure=reward(calculator,b,c,actual_drive=command,previous_actual_drive=command,
                     previous_previous_actual_drive=(0.,)*12)
    assert landing['families']['contact_motion_quality']<0.  # Impact remains assessed.
    assert (departure['cost_components']['confirmed_post_touchdown_rebound']>0.) is passive


@pytest.mark.parametrize('stage',['P01','P06','P08','P09','P13'])
def test_future_unloaded_leg_does_not_discount_current_physical_target(stage):
    obs=live()
    leg_state(obs,'RR',air=True)
    obs['contacts']['front_right_wheel']['ground']['normal_force_n']=14.
    supervisor=TaskStageSupervisor(CFG/'stage_task_spec.yaml',initial_stage_id=stage)
    snap=supervisor.observe_and_update(obs)
    assert snap['physical_evaluator']['current_legs']['FR']['load_fraction']==pytest.approx(.5)
    assert snap['physical_evaluator']['current_legs']['RR']['load_fraction']==0.
    assert snap['physical_transfer_fraction']==0.
    assert snap['termination_reason'] is None


@pytest.mark.parametrize('placed,target',[
    ((),'FR'), (('FR',),'FL'), (('FR','FL'),'RR'),
    (('FR','FL','RR'),'RL'), (('FR','FL','RR','RL'),None),
])
def test_transfer_eligibility_uses_actual_placement_history_not_phase(placed,target):
    ev=evaluator(); obs=live(); ev.observe(obs)
    for leg in placed: obs=place(ev,obs,leg)
    obs=advance(obs)
    if target is not None: leg_state(obs,target,air=True)
    # Deliberately retain P01 as label: measured predecessor placement, not a
    # stage counter or nominal clock, identifies the next eligible target.
    supervisor=TaskStageSupervisor(CFG/'stage_task_spec.yaml',evaluator=ev,initial_stage_id='P01')
    snap=supervisor.observe_and_update(obs)
    assert all(snap['history']['placed'][leg] for leg in placed)
    assert snap['physical_transfer_fraction']==(1. if target is not None else 0.)
    assert snap['termination_reason'] is None


def test_nominal_and_residual_changes_are_diagnostic_when_applied_is_constant():
    a,b=_built(),_built(_frame(1))
    r=reward(calc(),a,b,nominal=(1.,)*12,residual=(-1.,)*12)
    assert r['families']['control_smoothness']==0.
    assert r['families']['control_regularization']==0.
    assert r['cost_components']['nominal_first_difference']>0.
    assert r['total']<0.


def test_real_env_phase_change_is_not_done_and_new_ranges_reach_meaningful_targets():
    env=SemanticEpisodeEnv(Backend(transition=True),action_config=CFG/'execution_profile.yaml',reward_config_path=CFG/'reward_config.yaml')
    env.reset(); step=env.step((.4,)*12)
    assert not step.terminated and not step.truncated
    assert max(step.info['projected_residual_full12'][:8])>3.
    assert step.info['physics_ticks']==8


def test_v3_failure_bound_success_and_no_cycle_or_stall_bonus():
    calculator=calc(); assert calculator.config.values['failure_cost']>calculator.config.failure_avoidance_bound
    states=[_built(_frame(i,phi=p)) for i,p in enumerate((.2,.8,.2))]
    rs=[reward(calculator,a,b) for a,b in zip(states,states[1:])]
    assert rs[0]['potential_shaping']+calculator.config.gamma*rs[1]['potential_shaping']<0.
    a,b=states[:2]
    success=calculator.evaluate(a,b,[_sample(a,b)],termination_reason='SUCCESS',task_success=True)
    failure=calculator.evaluate(a,b,[_sample(a,b)],termination_reason='BODY_COLLISION',task_success=False)
    assert success['total']>failure['total']
    assert failure['potential_after']==0. and not failure['terminal_bootstrap_allowed']

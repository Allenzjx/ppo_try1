"""Real evaluator AIR-process shaping, independent of unchanged hard history."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from test_semantic_supervisor import observation, advance, leg_state, place
from test_semantic_observation_reward_env import _built, _frame, _sample, _raw
from wlr50_clean.fsm.state_spec import load_fsm_spec
from wlr50_clean.ppo.semantic_supervisor import (
    DEFAULT_TASK_SPEC_PATH, LEG_ORDER, PHASE_IDS, SemanticControllerAdapter,
    SemanticObservationError, TaskEvaluator, TaskStageSupervisor, load_task_spec,
)
from wlr50_clean.ppo.semantic_reward import FAMILIES, SemanticRewardCalculator, load_semantic_reward_config
from wlr50_clean.reference.motion_contract import load_motion_contract

ROOT=Path(__file__).resolve().parents[2]
CFG=ROOT/'configs/ppo_semantic_v3'
SPEC=CFG/'stage_task_spec.yaml'
VALUES=load_task_spec(SPEC)


def prepared(*, enabled=True):
    spec=deepcopy(VALUES)
    if not enabled: spec.pop('lift_credit_semantics')
    ev=TaskEvaluator(spec=spec);obs=observation()
    for joint in obs['joints'].values(): joint['command_deg']=0.
    ev.observe(obs)
    for leg in ('FR','FL'): obs=place(ev,obs,leg)
    return ev,obs


def sample(ev,obs,*,bottom,fl_position=4.,fl_target=15.,air=True,wall=False,x=.4,leg='RR'):
    obs=advance(obs);leg_state(obs,leg,x=x,bottom=bottom,air=air,top=wall)
    obs['joints']['front_left_hip'].update(position_deg=fl_position,command_deg=fl_target)
    snapshot=ev.observe(obs)
    return obs,snapshot


def earned(*, enabled=True):
    ev,obs=prepared(enabled=enabled)
    obs,_=sample(ev,obs,bottom=.001,fl_position=2.)
    obs,snap=sample(ev,obs,bottom=.006,fl_position=4.)
    if enabled: assert snap['current_legs']['RR']['soft_air_actuation_earned']
    assert snap['current_legs']['RR']['initial_clearance']
    assert not snap['history']['active_lift']['RR']
    return ev,obs


def credit(ev,snapshot=None):
    return TaskStageSupervisor(SPEC,evaluator=ev)._current_lift_credit('RR',snapshot or ev.snapshot)


def test_constant_target_other_leg_response_can_earn_soft_air_without_rr_motion():
    ev,obs=earned()
    current=ev.snapshot['current_legs']['RR']
    suffix=list(ev._samples['RR'])[-2:]
    assert suffix[0][6]==suffix[1][6]  # No new joint target change.
    assert current['recent_joint_motion_deg']==0.  # RR itself did not move.
    assert current['recent_whole_body_joint_motion_deg']>=2.
    assert current['soft_air_earned_tick']==obs['physics_tick']
    assert 0.<credit(ev)<.99


def test_initial_to_top_height_has_strict_progress_but_soft_never_enables_carry():
    ev,obs=earned();sup=TaskStageSupervisor(SPEC,evaluator=ev)
    values=[]
    for height in (.003,.008,.015,.030,.049):
        obs,snap=sample(ev,obs,bottom=height)
        current=snap['current_legs']['RR']
        expected=.99*.008/(.008+.05-height)
        assert credit(ev)==pytest.approx(expected)
        assert current['air'] and current['load_fraction']==0.
        assert not snap['history']['active_lift']['RR']
        assert not snap['history']['front_edge_crossed']['RR']
        assert not snap['history']['placed']['RR'] and not snap['success']
        # Front legs completed, RL ineligible. RR workspace/unload are each1.
        expected_phi=.85*(2.+.1+.1+.25*(.25+.75*expected))/4.
        values.append(sup.physical_potential(snap))
        assert values[-1]==pytest.approx(expected_phi)  # Carry remains exactly0.
    assert all(a<b for a,b in zip(values,values[1:]))


def test_hover_past_window_and_phase_handoff_retain_earned_state_without_bonus():
    # Shadow the measured prefix from episode time zero; never inject a handoff
    # clock or history bits after the fact.
    sup=TaskStageSupervisor(SPEC,initial_stage_id='P08');ev=sup.evaluator;obs=observation()
    for joint in obs['joints'].values(): joint['command_deg']=0.
    sup.observe_and_update(obs)
    for leg in ('FR','FL'):
        for data in ({'bottom':.012,'air':True,'hip':1.5},
                     {'bottom':.020,'air':True,'hip':3.},
                     {'x':.53,'bottom':.075,'air':True,'hip':4.},
                     {'x':.53,'bottom':.05,'top':True},
                     {'x':.53,'bottom':.05,'top':True}):
            obs=advance(obs);leg_state(obs,leg,**data);sup.observe_and_update(obs)
        assert ev.snapshot['history']['placed'][leg]
    for height,position in ((.001,2.),(.006,4.),(.025,4.)):
        obs,snap=sample(ev,obs,bottom=height,fl_position=position)
        sup.observe_and_update(obs)
    earned_tick=snap['current_legs']['RR']['soft_air_earned_tick']
    assert earned_tick is not None and sup.episode_started_s==0.
    before=sup.physical_potential(snap)
    for _ in range(250):
        obs=advance(obs);task=sup.observe_and_update(obs)
        assert task['physical_evaluator']['current_legs']['RR']['soft_air_earned_tick']==earned_tick
        assert task['task_progress_potential']==pytest.approx(before)
    assert sup.stage_id=='P09'
    current=ev.snapshot['current_legs']['RR']
    assert current['recent_clearance_gain_m']==0.
    assert current['recent_whole_body_joint_motion_deg']==0.
    assert current['soft_air_actuation_earned']
    # Public live-prefix handoff retains the actual same evaluator, not bits by phase.
    while obs['physics_tick']%8: obs=advance(obs);sup.observe_and_update(obs)
    contract=load_motion_contract(ROOT/'configs/recording_motion_contract.json')
    controller,_=SemanticControllerAdapter.from_live_prefix(load_fsm_spec(ROOT/'configs/fsm_states.yaml'),
        contract,supervisor=sup,nominal_full12=(0.,)*12,tracking_servo_names=(),
        physics_tick=obs['physics_tick'],sim_time_s=obs['simulation_time_s'])
    assert controller.evaluator is ev
    assert ev.snapshot['current_legs']['RR']['soft_air_earned_tick']==earned_tick


def test_soft_descent_lowers_current_credit_without_expiring_air_process():
    ev,obs=earned();obs,high=sample(ev,obs,bottom=.030)
    high_credit=credit(ev);earned_tick=high['current_legs']['RR']['soft_air_earned_tick']
    obs,low=sample(ev,obs,bottom=.012)
    assert 0.<credit(ev)<high_credit
    assert low['current_legs']['RR']['soft_air_actuation_earned']
    assert low['current_legs']['RR']['soft_air_earned_tick']==earned_tick
    assert high['history']==low['history']


@pytest.mark.parametrize('contact',['wall','ground'])
def test_contact_revokes_soft_but_only_existing_hard_rules_control_history(contact):
    ev,obs=earned()
    obs,snap=sample(ev,obs,bottom=.025 if contact=='wall' else 0.,air=False,wall=contact=='wall')
    assert not snap['current_legs']['RR']['soft_air_actuation_earned']
    assert snap['current_legs']['RR']['soft_air_earned_tick'] is None
    assert credit(ev)==0.
    if contact=='wall': assert snap['current_legs']['RR']['initial_clearance']
    else: assert not snap['current_legs']['RR']['initial_clearance']


def test_old_initial_wall_height_then_passive_air_cannot_reuse_contact_gain_but_fresh_correction_can():
    ev,obs=earned()
    for height in (.025,.040):
        obs,snap=sample(ev,obs,bottom=height,wall=True,air=False,fl_position=5.,fl_target=5.)
        assert snap['current_legs']['RR']['initial_clearance']
        assert credit(ev)==0.
    for height in (.041,.045,.046):
        obs,snap=sample(ev,obs,bottom=height,fl_position=5.,fl_target=5.)
        assert snap['current_legs']['RR']['recent_clearance_gain_m']>.02  # Mixed window is misleading.
        assert not snap['current_legs']['RR']['soft_air_actuation_earned']
        assert credit(ev)==0.
    obs,_=sample(ev,obs,bottom=.047,fl_position=6.,fl_target=15.)
    obs,snap=sample(ev,obs,bottom=.049,fl_position=8.,fl_target=15.)
    assert snap['current_legs']['RR']['soft_air_actuation_earned']
    assert 0.<credit(ev)<.99 and not snap['history']['active_lift']['RR']


def test_passive_ascent_then_late_command_on_descent_cannot_backdate_acquisition():
    ev,obs=prepared()
    for height in (.004,.010,.020):
        obs,snap=sample(ev,obs,bottom=height,fl_position=4.,fl_target=4.)
        assert not snap['current_legs']['RR']['soft_air_actuation_earned']
    obs,snap=sample(ev,obs,bottom=.019,fl_position=7.,fl_target=15.)
    assert snap['current_legs']['RR']['recent_clearance_gain_m']>=.008
    assert snap['current_legs']['RR']['whole_body_actuation_evidence']
    assert not snap['current_legs']['RR']['soft_air_actuation_earned']
    assert credit(ev)==0.


@pytest.mark.parametrize('kind',['wheel_spin','command_only','motion_without_demand'])
def test_spin_new_targets_or_passive_response_alone_do_not_authorize_soft_height(kind):
    ev,obs=prepared()
    # Set every joint's target to its actual pose; avoid unrelated tracking error.
    for name,joint in obs['joints'].items(): joint['command_deg']=joint['position_deg']
    for i,height in enumerate((.001,.007,.015)):
        obs=advance(obs);leg_state(obs,'RR',x=.4,bottom=height,air=True)
        if kind=='wheel_spin':
            for wheel in obs['wheels'].values(): wheel.update(command_rad_s=1.,velocity_rad_s=1.)
        elif kind=='command_only': obs['joints']['front_left_hip']['command_deg']=15.+i
        elif kind=='motion_without_demand':
            # Actual motion with only subthreshold target variation and error.
            obs['joints']['front_left_hip']['position_deg']=4.+i*.02
            obs['joints']['front_left_hip']['command_deg']=4.+i*.02
        snap=ev.observe(obs)
        assert not snap['current_legs']['RR']['soft_air_actuation_earned']


def test_qualified_below_top_loses_lift_credit_without_clearing_hard_or_carry():
    ev,obs=earned();obs,high=sample(ev,obs,bottom=.070,fl_position=7.)
    assert high['history']['active_lift']['RR'] and credit(ev)==1.
    sup=TaskStageSupervisor(SPEC,evaluator=ev);high_phi=sup.physical_potential(high)
    obs,low=sample(ev,obs,bottom=.025,fl_position=7.)
    assert low['history']['active_lift']['RR'] and not low['history']['front_edge_crossed']['RR']
    fraction=.008/(.008+.025)
    assert credit(ev)==pytest.approx(fraction)
    assert high_phi-sup.physical_potential(low)==pytest.approx(.85/4*.25*.75*(1-fraction))
    assert high['history']==low['history']
    # A grounded observation still invokes the original hard revocation.
    obs,ground=sample(ev,obs,bottom=0.,air=False,fl_position=7.)
    assert not ground['history']['active_lift']['RR']


def test_crossed_qualified_retains_achieved_credit_and_top_contact_does_not_undo_hard_state():
    ev,obs=earned();obs,_=sample(ev,obs,bottom=.070,fl_position=7.)
    obs,crossed=sample(ev,obs,bottom=.060,x=.53,fl_position=7.)
    assert crossed['history']['front_edge_crossed']['RR']
    obs,low=sample(ev,obs,bottom=.040,x=.53,fl_position=7.,wall=True,air=False)
    assert low['history']['active_lift']['RR'] and low['history']['front_edge_crossed']['RR']
    assert not low['current_legs']['RR']['soft_air_actuation_earned']
    assert credit(ev)==1. and not low['success']


@pytest.mark.parametrize('failure',['collision','nonfinite','unverified_pair','nonfinite_clearance'])
def test_invalid_or_failed_physics_cannot_bypass_soft_credit(failure):
    ev,obs=earned();obs=advance(obs)
    if failure=='collision': obs['body_collision']['detected']=True
    elif failure=='nonfinite': obs['all_finite']=False
    elif failure=='unverified_pair': obs['contacts']['rear_right_wheel']['ground']['pair_verified']=False
    elif failure=='nonfinite_clearance': obs['wheels']['rear_right_ankle']['bottom_w_m'][2]=float('nan')
    if failure in ('unverified_pair','nonfinite_clearance'):
        with pytest.raises(SemanticObservationError): ev.observe(obs)
    else:
        snap=ev.observe(obs)
        assert snap['termination_reason'] is not None
        assert not snap['current_legs']['RR']['soft_air_actuation_earned']
        assert credit(ev)==0.


def test_hard_history_and_events_unchanged_in_real_observation_replay():
    new,obs=prepared();old=TaskEvaluator(spec={k:v for k,v in VALUES.items() if k!='lift_credit_semantics'})
    # Both evaluators receive the exact complete sequence from their fresh reset.
    initial=observation()
    for joint in initial['joints'].values(): joint['command_deg']=0.
    old.observe(initial)
    for leg in ('FR','FL'): initial=place(old,initial,leg)
    assert initial==obs
    for height,air,wall,x in ((.001,True,False,.4),(.006,True,False,.4),(.030,False,True,.4),
                              (.040,True,False,.4),(.049,True,False,.4),(.070,True,False,.4),
                              (.060,True,False,.53),(.050,False,True,.53),(.050,False,True,.53)):
        obs,snap=sample(new,obs,bottom=height,fl_position=obs['joints']['front_left_hip']['position_deg']+2.,
                        air=air,wall=wall,x=x)
        reference=old.observe(obs)
        assert snap['history']==reference['history']
        assert snap['success']==reference['success'] and snap['termination_reason']==reference['termination_reason']
        for leg in LEG_ORDER:
            stripped={k:v for k,v in snap['current_legs'][leg].items() if not k.startswith('soft_air_')}
            assert stripped==reference['current_legs'][leg]


def test_reward_difference_occurs_once_and_discounted_closed_height_cycle_cannot_farm_bonus():
    ev,obs=earned();sup=TaskStageSupervisor(SPEC,evaluator=ev)
    phis=[]
    for height in (.015,.030,.015):
        obs,snap=sample(ev,obs,bottom=height);phis.append(sup.physical_potential(snap))
    config=load_semantic_reward_config(CFG/'reward_config.yaml')
    a,b,c=[_built(_frame(i,phi=phi)) for i,phi in enumerate(phis)]
    unchanged=replace(b,task={**b.task,'task_progress_potential':phis[0]})
    def reward(left,right):
        return SemanticRewardCalculator(config).evaluate(left,right,[_sample(left,right)],termination_reason=None,task_success=False)
    base,improved=reward(a,unchanged),reward(a,b)
    expected=config.values['potential_weight']*config.gamma*(phis[1]-phis[0])
    assert improved['total']-base['total']==pytest.approx(expected)
    assert improved['families']['task_progress']-base['families']['task_progress']==pytest.approx(expected)
    for family in FAMILIES[1:]: assert improved['families'][family]==base['families'][family]
    assert improved['potential_shaping']+config.gamma*reward(b,c)['potential_shaping']<=0.
    terminal=SemanticRewardCalculator(config).evaluate(b,c,[_sample(b,c)],termination_reason='FALL',task_success=False)
    assert terminal['potential_after']==0.


def test_current_same_state_phase_independence_v2_and_unopted_v3_compatibility(tmp_path):
    ev,obs=earned();obs,snap=sample(ev,obs,bottom=.030)
    values=[TaskStageSupervisor(SPEC,evaluator=ev,initial_stage_id=phase).physical_potential(snap) for phase in PHASE_IDS]
    assert values==[values[0]]*13
    old_values=deepcopy(VALUES);old_values.pop('lift_credit_semantics')
    path=tmp_path/'old.yaml';path.write_text(yaml.safe_dump(old_values,sort_keys=False),encoding='utf-8')
    old=TaskStageSupervisor(path,evaluator=ev)
    assert old.physical_potential(snap)==pytest.approx(.48078125)
    obs,lower=sample(ev,obs,bottom=.010)
    assert old.physical_potential(lower)==old.physical_potential(snap)
    assert 'lift_credit_semantics' not in load_task_spec(DEFAULT_TASK_SPEC_PATH)


def test_shaping_diagnostics_do_not_change_324_actor_encoding():
    from wlr50_clean.ppo.semantic_observation import HISTORY_GROUPS,SemanticObservationBuilder,load_semantic_observation_schema
    raw=_raw(joint=0.);sup=TaskStageSupervisor(SPEC);task=sup.observe_and_update(raw)
    frame=_frame(raw=raw);frame.state_id=task['stage_id'];frame.info['semantic_task']=task
    schema=load_semantic_observation_schema(CFG/'observation_schema.json');history=dict.fromkeys(HISTORY_GROUPS,(0.,)*12)
    before=SemanticObservationBuilder(schema).build(frame,history)
    clean=deepcopy(task)
    for row in clean['physical_evaluator']['current_legs'].values():
        for key in list(row):
            if key.startswith('soft_air_'): row.pop(key)
    frame.info['semantic_task']=clean
    after=SemanticObservationBuilder(schema).build(frame,history)
    assert schema.dimension==324 and schema.encode(before.groups)==schema.encode(after.groups)


@pytest.mark.parametrize('mode,scale',[('unknown',.008),('measured_air_process_current_top_gap',0.),
                                     ('measured_air_process_current_top_gap',float('nan'))])
def test_bad_progress_revision_or_physical_scale_rejected(tmp_path,mode,scale):
    spec=deepcopy(VALUES);spec['lift_credit_semantics']=mode;spec['history']['minimum_lift_gain_m']=scale
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(spec),encoding='utf-8')
    with pytest.raises(ValueError): load_task_spec(path)

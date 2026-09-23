"""Directed synthetic wiring/credit cases, not physical capture evidence."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from test_semantic_source_partial_order_v1 import contract, layer
from test_semantic_rr_capture_context import measured, top_rr, SUPPORT_SPEC
from test_semantic_rear_policy_timing_control import potential_pair, current, position_source
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.ppo.semantic_rear_policy_timing import (
    MODE, RECAPTURE_MODE, public_timing, rear_dependency,
)
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider
from wlr50_clean.ppo.semantic_backend import load_execution_profile

CONFIG = Path(__file__).resolve().parents[2]/'configs/ppo_rr_rl_timing_policy_learning_v1'


def post_capture(*, phase='P10', lift_valid=False):
    item, _ = measured()
    item['stage_id'] = phase
    ev = item['physical_evaluator']
    ev['history']['placed']['RR'] = True
    ev['current_legs']['RR']['current_lift_valid'] = lift_valid
    layers = [dict(stage='P09', ticks=658,
                   sequence_diagnostic={'late_group_start_tick':8439}),
              dict(stage='P12', ticks=17, rl_ticks=3, rl_dependency_wait=True)]
    return item, layers


@pytest.mark.parametrize('phase', ['P09','P10','P11','P12'])
def test_v2_current_loss_reopens_task_without_rewinding_or_faking_lift(phase):
    item, layers = post_capture(phase=phase)
    before = deepcopy((item,layers))
    old = public_timing(item,layers,SUPPORT_SPEC,120.,mode=MODE)
    new = public_timing(item,layers,SUPPORT_SPEC,120.,mode=RECAPTURE_MODE)
    assert old['rl_prep_transfer'] and not old['rr_carry_capture']
    assert new['rr_carry_capture'] and not new['rl_prep_transfer']
    assert not new['rr_support_handoff']
    assert (item,layers) == before
    for field in ('p09_source_time_s','p12_source_time_s','p12_rl_source_time_s',
                  'p09_dependency_wait','p12_dependency_wait'):
        assert new[field] == old[field]
    assert not item['physical_evaluator']['current_legs']['RR']['current_lift_valid']
    assert not rear_dependency(item,SUPPORT_SPEC)['rr_current_bearing']


def test_v1_default_output_and_pre_capture_v2_are_unchanged():
    item,_ = measured(); item['stage_id']='P09'
    old=public_timing(item,[],SUPPORT_SPEC,120.,mode=MODE)
    assert old == public_timing(item,[],SUPPORT_SPEC,120.)
    assert old == public_timing(item,[],SUPPORT_SPEC,120.,mode=RECAPTURE_MODE)
    with pytest.raises(ValueError):
        public_timing(item,[],SUPPORT_SPEC,120.,mode='unknown')


@pytest.mark.parametrize('permission',['current_RR_bearing','current_RL_air','RL_placed'])
def test_real_support_and_legal_rl_continuation_do_not_force_recapture(permission):
    item,layers=post_capture()
    ev=item['physical_evaluator']
    if permission=='current_RR_bearing':
        top_rr(item)
    elif permission=='current_RL_air':
        ev['current_legs']['RL'].update(air=True,ground_contact=False,
            current_lift_valid=True,motion_continuation_allowed=True,support=False)
    else:
        ev['history']['placed']['RL']=True
    old=public_timing(item,layers,SUPPORT_SPEC,120.,mode=MODE)
    new=public_timing(item,layers,SUPPORT_SPEC,120.,mode=RECAPTURE_MODE)
    assert new == old
    assert not new['rr_carry_capture']


def retention_case(gap=.07):
    sup,_,ev=potential_pair()
    sup.spec['nominal']['rear_policy_timing']=RECAPTURE_MODE
    rr=ev['current_legs']['RR']
    rr.update(clearance_m=gap,top_xy_outside_distance_m=0.,
        within_top_xy=True,within_lateral_span=True,consecutive_top_samples=0,
        ground_contact=False,air=True,obstacle_pair_active=False)
    legacy=deepcopy(sup)
    legacy.spec['nominal']['rear_policy_timing']=MODE
    return sup,legacy,ev


@pytest.mark.parametrize('gap',[0.,.007,.07,1.])
def test_air_approach_has_bounded_geometry_credit_but_never_contact_half(gap):
    sup,legacy,ev=retention_case(gap)
    before=deepcopy(ev)
    scale=sup.spec['geometry']['top_gap_max_m']
    expected=.5*scale/(scale+gap)
    assert sup._current_capture_retention('RR',ev)==pytest.approx(expected)
    assert 0. <= expected <= .5
    assert legacy._current_capture_retention('RR',ev)==1.
    # Existing .8 historical credit and all other leg/finish budgets unchanged.
    assert legacy.physical_potential(ev)-sup.physical_potential(ev)==pytest.approx(
        .85/4.*.2*(1.-expected))
    assert ev==before and ev['history']['placed']['RR']


def test_exact_current_top_bearing_can_earn_contact_share_without_new_event():
    sup,_,ev=retention_case(0.)
    rr=ev['current_legs']['RR']
    rr.update(air=False,support=True,bearing_verified=True,bearing_force_n=2.,
        top_contact=True,top_surface_contact=True,obstacle_pair_active=True,
        contact_surface='TOP',consecutive_top_samples=sup.spec['history']['minimum_top_samples'])
    before=deepcopy(ev)
    assert sup._current_capture_retention('RR',ev)==1.
    assert ev==before


@pytest.mark.parametrize('field,value',[
    ('bearing_force_n',0.),('bearing_force_n',float('nan')),
    ('bearing_verified',False),('support',False),
])
def test_top_without_verified_current_bearing_never_earns_full_retention(field,value):
    sup,_,ev=retention_case(0.)
    rr=ev['current_legs']['RR']
    rr.update(air=False,support=True,bearing_verified=True,bearing_force_n=2.,
        top_contact=True,top_surface_contact=True,obstacle_pair_active=True,
        contact_surface='TOP',consecutive_top_samples=2)
    rr[field]=value
    assert sup._current_capture_retention('RR',ev)==.5


@pytest.mark.parametrize('case',['outside_xy','outside_lateral','ground','wall','below_band'])
def test_illegal_descent_geometry_cannot_receive_recapture_reward(case):
    sup,_,ev=retention_case(0.)
    rr=ev['current_legs']['RR']
    if case=='outside_xy': rr['within_top_xy']=False
    elif case=='outside_lateral': rr['within_lateral_span']=False
    elif case=='ground': rr.update(ground_contact=True,air=False,support=True)
    elif case=='wall': rr.update(air=False,obstacle_pair_active=True,contact_surface='FRONT_WALL')
    else: rr['clearance_m']=sup.spec['geometry']['top_gap_min_m']-.001
    assert sup._current_capture_retention('RR',ev)==0.


@pytest.mark.parametrize('case',['RL_current_air','RL_placed'])
def test_later_legal_rl_task_keeps_legacy_rr_retention(case):
    sup,legacy,ev=retention_case(.07)
    if case=='RL_current_air':
        ev['current_legs']['RL'].update(current_lift_valid=True,
            motion_continuation_allowed=True,air=True,ground_contact=False)
    else: ev['history']['placed']['RL']=True
    assert sup._current_capture_retention('RR',ev)==legacy._current_capture_retention('RR',ev)


def test_provider_passes_explicit_new_mode_to_existing_nine_fields(contract):
    spec=yaml.safe_load((CONFIG/'stage_task_spec.yaml').read_text())
    spec['nominal']['rear_policy_timing']=RECAPTURE_MODE
    p=NominalMotionProvider.from_handoff(contract,spec=spec,stage_id='P10',
        nominal_full12=contract.phase('P10').start_full12,tracking_servo_names=())
    item,_=post_capture()
    result=p.rear_policy_timing(item)
    assert len(result)==9 and result['rr_carry_capture'] and not result['rl_prep_transfer']


@pytest.mark.parametrize('rl_swing', [False, True])
def test_drop_after_late_keeps_stop_and_couples_task_retention_and_rl_lane(contract, rl_swing):
    # Synthetic scheduler inputs only: source cursor positioning and the P12
    # task label below do not claim a legal physical episode/prefix or contact.
    spec=yaml.safe_load((CONFIG/'stage_task_spec.yaml').read_text())
    spec['nominal']['rear_policy_timing']=RECAPTURE_MODE
    p=NominalMotionProvider.from_handoff(contract,spec=spec,stage_id='P09',
        nominal_full12=contract.phase('P09').start_full12,tracking_servo_names=())
    sup,_,_=retention_case()

    def send(tick, phase, *, bearing=False, swing=False, placed=True):
        item,raw=current(contract,tick,phase,rr_bearing=bearing,rl_swing=swing,
                         gap=0. if bearing else .025)
        ev=item['physical_evaluator']
        ev['history']['placed']['RR']=placed
        ev['history']['placed']['RL']=False
        ev['current_legs']['RR'].update(top_xy_outside_distance_m=0.,
            consecutive_top_samples=spec['history']['minimum_top_samples'] if bearing else 0)
        before=deepcopy((item,raw))
        command=p.evaluate(item,raw)
        assert (item,raw)==before
        return command,item

    send(0,'P09',placed=False)
    p09=layer(p,'P09')
    late_tick=p09['motion']._scaled_source_tick(p._p09_late_source[1])
    position_source(p09,late_tick)
    atomic_before=p09['motion'].source_atomic_emitted
    _,contact=send(1,'P09',bearing=True)
    assert p09['sample'].tick_index==late_tick
    assert p09['motion'].source_atomic_emitted==atomic_before+1
    assert sup._current_capture_retention('RR',contact['physical_evaluator'])==1.
    send(2,'P12',bearing=True)
    p12=layer(p,'P12')
    assert p12['ticks']==p12['rl_ticks']==1
    joint_tick=p12['rl_ticks']; held_pair=p.nominal_full12[4:6]
    stop=next(w for w in contract.phase('P09').waypoints if w.time_s>p._p09_late_source[1]
        and set(w.atomic_channels)==set(WHEEL_ORDER) and w.full12[8:]==(0.,)*4)
    stop_tick=p09['motion']._scaled_source_tick(stop.time_s)
    stop_count=0; drop_steps=stop_tick-p09['ticks']+3
    assert drop_steps>0
    for tick in range(3,3+drop_steps):
        old_p09,old_p12=p09['ticks'],p12['ticks']
        command,item=send(tick,'P12',swing=rl_swing)
        ev=item['physical_evaluator']; timing=p.rear_policy_timing(item)
        assert p09['ticks']==old_p09+1 and p12['ticks']==old_p12+1
        assert ev['history']['placed']['RR'] and not rear_dependency(item,spec['support'])['rr_current_bearing']
        assert timing['rr_carry_capture'] is (not rl_swing)
        assert timing['rl_swing_capture'] is rl_swing
        assert timing['p12_dependency_wait'] is (not rl_swing)
        if rl_swing:
            assert p12['rl_ticks']==joint_tick+(tick-2)
            assert sup._current_capture_retention('RR',ev)==1.
        else:
            assert p12['rl_ticks']==joint_tick and command[4:6]==held_pair
            assert 0.<sup._current_capture_retention('RR',ev)<.5
        groups=[g for g in p09['sample'].atomic_groups
                if set(g.channels)==set(WHEEL_ORDER)]
        if groups:
            assert p09['sample'].tick_index==stop_tick
            assert p09['sample'].full12[8:]==(0.,)*4
            stop_count+=len(groups)
    # Later P12 wheel ownership can override a P09 stop in final composition;
    # count the actual P09 source event, not an invented final all-wheel zero.
    assert stop_count==1 and p09['sample'].endpoint_issued
    assert p09['motion'].source_atomic_emitted==atomic_before+1
    assert p09['sequence_diagnostic']['late_group_start_tick']==1
    if not rl_swing:
        _,reacquired=send(3+drop_steps,'P12',bearing=True)
        assert p12['rl_ticks']==joint_tick+1
        assert p12['rl_sample'].tick_index==joint_tick  # No catch-up replay.
        assert not p.rear_policy_timing(reacquired)['rr_carry_capture']
        assert sup._current_capture_retention('RR',reacquired['physical_evaluator'])==1.


def test_profile_accepts_new_mode_but_keeps_rear_task_assists_off(tmp_path):
    data=yaml.safe_load((CONFIG/'execution_profile.yaml').read_text())
    data['rear_policy_timing_mode']=RECAPTURE_MODE
    path=tmp_path/'new_mode.yaml';path.write_text(yaml.safe_dump(data))
    profile=load_execution_profile(path)
    assert profile['rear_policy_timing_mode']==RECAPTURE_MODE
    assert profile['rr_capture_assist_mode'] is None
    assert profile['rr_capture_wheel_mode']=='off'
    from wlr50_clean.ppo.semantic_nominal_geometry import MODE as GEOMETRY_MODE
    data['nominal_geometry_advisory']=GEOMETRY_MODE
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError): load_execution_profile(path)

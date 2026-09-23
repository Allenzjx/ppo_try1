"""Finite CPU dependency/owner tests, not physical rollout or success evidence."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from test_semantic_source_partial_order_v1 import contract, layer
from test_semantic_rr_top_continuation import top_input
from test_semantic_rr_capture_context import measured, top_rr, SUPPORT_SPEC
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider
from wlr50_clean.ppo.semantic_rear_policy_timing import MODE, rear_dependency, public_timing
from wlr50_clean.ppo.semantic_backend import load_execution_profile
from wlr50_clean.ppo.semantic_rr_capture_assist import RR_CAPTURE_ASSIST_MODE, RR_CAPTURE_FEEDBACK_REVISION
from wlr50_clean.ppo.semantic_rr_carry_wheel import MODE as WHEEL_MODE

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'configs/ppo_rr_rl_timing_policy_learning_v1'


def provider(contract, phase='P09', enabled=True):
    spec = yaml.safe_load((CONFIG/'stage_task_spec.yaml').read_text())
    if not enabled: spec['nominal'].pop('rear_policy_timing')
    return NominalMotionProvider.from_handoff(contract,spec=spec,stage_id=phase,
        nominal_full12=contract.phase(phase).start_full12,tracking_servo_names=())


def current(contract,tick,phase='P09',*,rr_bearing=False,rl_swing=False,gap=.025):
    item,raw = top_input(contract,tick)
    item['stage_id'] = phase
    ev = item['physical_evaluator']; rows = ev['current_legs']
    ev['history']['front_edge_crossed'] = dict.fromkeys(rows,False)
    ev['history']['front_edge_crossed']['RR'] = True
    rr = rows['RR']
    rr.update(within_top_xy=True,within_lateral_span=True,front_distance_m=.02,clearance_m=gap,
        current_lift_valid=True,motion_continuation_allowed=True,wheel_bottom_vz_m_s=-.01,
        wheel_bottom_vz_observation_tick=tick,ground_contact=False)
    if rr_bearing:
        rr.update(air=False,support=True,bearing_verified=True,bearing_force_n=3.,top_contact=True,
            top_surface_contact=True,obstacle_pair_active=True,contact_surface='TOP')
    else:
        rr.update(air=True,support=False,bearing_force_n=0.,top_contact=False,
            top_surface_contact=False,obstacle_pair_active=False,contact_surface='NONE')
    if rl_swing:
        rows['RL'].update(air=True,ground_contact=False,support=False,bearing_force_n=0.,
            current_lift_valid=True,motion_continuation_allowed=True,clearance_m=.02,
            front_distance_m=.03,within_top_xy=True)
        ev['history']['active_lift']['RL'] = True
    return item,raw


def issue(p,contract,tick,phase='P09',**kwargs):
    item,raw = current(contract,tick,phase,**kwargs)
    before = deepcopy((item,raw))
    command = p.evaluate(item,raw)
    assert (item,raw) == before
    return command,item


def position_source(source,tick,*,rl=False):
    # Only a CPU scheduler fixture, never an episode teleport or recording edit.
    key,motion = ('rl_ticks','rl_motion') if rl else ('ticks','motion')
    source[key] = source[motion]._tick_index = tick


@pytest.mark.parametrize('placed', [False,True])
@pytest.mark.parametrize('gap', [-.014,0.,.025,.07])
def test_air_and_history_never_become_rr_bearing_or_transfer(placed,gap):
    item,_ = measured(); item['stage_id']='P09'
    item['physical_evaluator']['history']['placed']['RR'] = placed
    item['physical_evaluator']['current_legs']['RR']['clearance_m'] = gap
    result = rear_dependency(item,SUPPORT_SPEC)
    assert not result['rr_top_contact'] and not result['rr_current_bearing']
    assert not result['support_transfer_permitted']
    timing = public_timing(item,[],SUPPORT_SPEC,120.)
    assert timing['rr_carry_capture'] and not timing['rl_prep_transfer']


@pytest.mark.parametrize('field,value', [('top_contact',False),('top_surface_contact',False),
    ('air',True),('ground_contact',True),('within_top_xy',False),('obstacle_pair_active',False),
    ('bearing_verified',False),('support',False),('bearing_force_n',0.),
    ('bearing_force_n',float('nan')),('bearing_force_n',True),('contact_surface','FRONT_WALL')])
def test_current_sensor_contradictions_do_not_enable_support_transfer(field,value):
    item,_ = measured(); top_rr(item)
    item['physical_evaluator']['current_legs']['RR'][field] = value
    assert not rear_dependency(item,SUPPORT_SPEC)['support_transfer_permitted']


def test_real_top_can_exist_before_bearing_and_requires_other_measured_bridge():
    item,_ = measured(); top_rr(item,support=False)
    result = rear_dependency(item,SUPPORT_SPEC)
    assert result['rr_top_contact'] and not result['rr_current_bearing']
    top_rr(item)
    assert rear_dependency(item,SUPPORT_SPEC)['support_transfer_permitted']
    for leg in ('FL','FR'):
        item['physical_evaluator']['current_legs'][leg].update(air=True,support=False,ground_contact=False)
    result = rear_dependency(item,SUPPORT_SPEC)
    assert result['rr_current_bearing'] and not result['support_transfer_permitted']


@pytest.mark.parametrize('terminal', ['task','evaluator','invalid'])
def test_invalid_or_terminal_state_does_not_authorize_dependencies(terminal):
    item,_ = measured(); top_rr(item)
    if terminal=='task': item['termination_reason']='SAFETY_ABORT'
    elif terminal=='evaluator': item['physical_evaluator']['termination_reason']='SAFETY_ABORT'
    else: item['physical_evaluator']['valid']=False
    assert not rear_dependency(item,SUPPORT_SPEC)['support_transfer_permitted']


@pytest.mark.parametrize('gap', [0.,.025])
def test_p09_pending_carry_knee_can_run_before_capture_and_gap_zero_is_not_support(contract,gap):
    p = provider(contract); issue(p,contract,0)
    source = layer(p,'P09')
    pending = source['motion']._scaled_source_tick(p._rr_carry_knee_source_times[0])
    position_source(source,pending)
    _,item = issue(p,contract,1,gap=gap)
    assert source['ticks'] == pending+1
    assert source['sequence_diagnostic']['current_free_lift_source_readiness']['ready']
    assert not rear_dependency(item,p.spec['support'])['rr_current_bearing']


def test_pending_knee_preserves_old_safe_drop_negative_air_guard(contract):
    p = provider(contract); issue(p,contract,0)
    source=layer(p,'P09')
    pending=source['motion']._scaled_source_tick(p._rr_carry_knee_source_times[0])
    position_source(source,pending)
    issue(p,contract,1,gap=-.010)
    assert source['ticks']==pending


def test_p09_late_waits_one_group_while_carry_wheels_keep_safe_feedback_and_then_runs_once(contract):
    p = provider(contract); issue(p,contract,0)
    source = layer(p,'P09')
    pending = source['motion']._scaled_source_tick(p._p09_late_source[1])
    stop = source['motion']._scaled_source_tick(p._p09_late_source[0])
    position_source(source,stop)
    stopped,_ = issue(p,contract,1)
    assert stopped[8:] == (0.,)*4
    position_source(source,pending-1)
    issue(p,contract,2)
    assert source['sample'].tick_index == pending-1
    initial_count = source['motion'].source_atomic_emitted
    for tick in range(3,7):
        command,item = issue(p,contract,tick,gap=-.001)
        assert source['ticks'] == pending
        assert source['motion'].source_atomic_emitted == initial_count
        assert source['sequence_diagnostic']['wait_reason'] == 'current_RR_bearing_before_FL_RL_transfer'
        assert command[8:] == (.3,)*4
        assert p.rear_policy_timing(item)['p09_dependency_wait']
        assert not item['physical_evaluator']['history']['placed']['RR']
    command,item = issue(p,contract,7,rr_bearing=True,gap=-.001)
    assert source['sample'].tick_index == pending and source['ticks'] == pending+1
    assert source['motion'].source_atomic_emitted == initial_count+1
    assert len(source['sample'].atomic_groups) == 1
    assert source['sample'].atomic_groups[0].source_full12_atomic
    assert command[8:] == (-1.07,0.,0.,0.)
    assert source['sequence_diagnostic']['late_group_start_tick'] == 7
    assert not p.rear_policy_timing(item)['p09_dependency_wait']
    issue(p,contract,8,rr_bearing=True,gap=-.001)
    assert source['motion'].source_atomic_emitted == initial_count+1


def test_old_opt_out_late_keeps_old_current_over_top_permission(contract):
    p = provider(contract,enabled=False); issue(p,contract,0)
    source = layer(p,'P09')
    pending = source['motion']._scaled_source_tick(p._p09_late_source[1])
    position_source(source,pending)
    _,item = issue(p,contract,1)
    assert source['ticks'] == pending+1
    assert not rear_dependency(item,p.spec['support'])['rr_current_bearing']


def test_p12_new_unload_pauses_but_started_wheel_stop_runs_and_joint_clock_does_not_catch_up(contract):
    p = provider(contract,'P12')
    for tick in range(3): issue(p,contract,tick,'P12')
    source = layer(p,'P12')
    assert source['ticks'] == source['rl_ticks'] == 0
    assert source['sample'] is source['rl_sample'] is None
    issue(p,contract,3,'P12',rr_bearing=True)
    assert source['ticks'] == source['rl_ticks'] == 1
    phase = contract.phase('P12')
    pulse = next(w for w in phase.waypoints if w.atomic_channels and any(w.full12[i] for i in range(8,12)))
    position_source(source,source['motion']._scaled_source_tick(pulse.time_s))
    pulse_command,_ = issue(p,contract,4,'P12',rr_bearing=True)
    assert any(pulse_command[i] for i in range(8,12))
    joint_ticks = source['rl_ticks']; joint_target = p.nominal_full12[4:6]
    stop = next(w for w in phase.waypoints if w.time_s > pulse.time_s and set(w.atomic_channels)==set(WHEEL_ORDER)
        and w.full12[8:] == (0.,)*4)
    stop_tick = source['motion']._scaled_source_tick(stop.time_s)
    position_source(source,stop_tick)
    command,item = issue(p,contract,5,'P12')
    assert source['sample'].tick_index == stop_tick and command[8:] == (0.,)*4
    assert source['rl_ticks'] == joint_ticks and command[4:6] == joint_target
    assert source['rl_dependency_wait'] and p.rear_policy_timing(item)['p12_dependency_wait']
    for tick in (6,7): issue(p,contract,tick,'P12')
    assert source['rl_ticks'] == joint_ticks
    issue(p,contract,8,'P12',rr_bearing=True)
    assert source['rl_ticks'] == joint_ticks+1
    assert source['rl_sample'].tick_index == joint_ticks


def test_already_air_rl_continues_even_if_rr_load_is_lost_but_ground_cannot_use_history(contract):
    p = provider(contract,'P12')
    issue(p,contract,0,'P12',rl_swing=True)
    source = layer(p,'P12')
    assert source['ticks'] == source['rl_ticks'] == 1
    _,item = issue(p,contract,1,'P12',rl_swing=True)
    assert source['rl_ticks'] == 2 and p.rear_policy_timing(item)['rl_swing_capture']
    item,raw = current(contract,2,'P12')
    item['physical_evaluator']['history']['active_lift']['RL'] = True
    item['physical_evaluator']['history']['placed']['RL'] = True
    p.evaluate(item,raw)
    assert source['ticks'] == 3 and source['rl_ticks'] == 2 and source['rl_dependency_wait']


def test_public_clocks_are_actual_independent_progress_and_flags_not_phase_credit():
    item,_ = measured(); item['stage_id']='P12'
    layers = [dict(stage='P09',ticks=600,sequence_diagnostic={'late_group_start_tick':20}),
        dict(stage='P12',ticks=120,rl_ticks=15,rl_dependency_wait=True)]
    timing = public_timing(item,layers,SUPPORT_SPEC,120.)
    assert timing['p09_source_time_s']==5. and timing['p12_source_time_s']==1.
    assert timing['p12_rl_source_time_s']==.125 and timing['p12_dependency_wait']
    assert timing['rl_prep_transfer'] and not rear_dependency(item,SUPPORT_SPEC)['rr_current_bearing']
    assert not item['physical_evaluator']['history']['placed']['RR']


@pytest.mark.parametrize('kind', ['RR_assist','wheel_projection','geometry'])
def test_new_backend_profile_rejects_every_rear_task_assist(tmp_path,kind):
    data = yaml.safe_load((CONFIG/'execution_profile.yaml').read_text())
    if kind=='RR_assist':
        data['rr_capture_assist_mode']=RR_CAPTURE_ASSIST_MODE
        data['rr_capture_feedback_revision']=RR_CAPTURE_FEEDBACK_REVISION
    elif kind=='wheel_projection': data['rr_capture_wheel_mode']=WHEEL_MODE
    else:
        from wlr50_clean.ppo.semantic_nominal_geometry import MODE as GEOMETRY_MODE
        data['nominal_geometry_advisory']=GEOMETRY_MODE
    path=tmp_path/'bad.yaml'; path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError): load_execution_profile(path)


def test_current_profile_is_opt_in_front_declared_rear_off_and_prefix_needs_no_proxy():
    profile = load_execution_profile(CONFIG/'execution_profile.yaml')
    assert profile['rear_policy_timing_mode']==MODE and profile['capture_assist_mode'] is not None
    assert profile['rr_capture_assist_mode'] is None and profile['rr_capture_wheel_mode']=='off'
    assert profile['nominal_geometry_advisory'] is None
    # Backend uses this explicit unwrap for ResetOnlyPrefixController, while
    # checkpoint-prefix retains the ordinary semantic backend/controller.
    p = object(); inner = SimpleNamespace(nominal_provider=p)
    wrapper = SimpleNamespace(_semantic=inner)
    assert (getattr(wrapper,'_semantic',None) or wrapper).nominal_provider is p


@pytest.mark.parametrize('channel', [6,7])
@pytest.mark.parametrize('raw_value', [-.002,.002])
def test_rear_raw_channel_retains_actual_final_target_authority_without_task_assist(channel,raw_value):
    from test_actuator_target_effect import _adapter
    from test_semantic_residual_adapter import plan
    from wlr50_clean.ppo.semantic_backend import build_semantic_projector
    from wlr50_clean.ppo.semantic_residual_adapter import SemanticActuationDispatch
    from wlr50_clean.ppo.isaac_fsm_backend import IsaacFSMBackend
    from wlr50_clean.infrastructure.command_batch import SERVO_ORDER,servo_limits_deg
    zero=(0.,)*12
    profile=load_execution_profile(CONFIG/'execution_profile.yaml')
    projector=build_semantic_projector(CONFIG/'execution_profile.yaml')
    raw=list(zero);raw[channel]=raw_value
    projection=projector.project(raw,state_id='P09',nominal_action_full12=zero,
        reference_action_full12=zero,reference_delta_full12=zero,previous_projected_residual_full12=zero)
    residual=projection.safe_projected_residual_full12
    assert residual[channel]*raw_value>0 and sum(x!=0 for x in residual)==1
    adapters=[_adapter(),_adapter()]
    receipts=[]
    for adapter,request in zip(adapters,(zero,residual)):
        # Matching original nominal/mapper/actuator history; real adapter writes
        # into tensor fixtures, not a physical robot or simulated response.
        adapter.apply_full12(zero,physics_tick=0,tracking_servo_names=())
        before=tuple(adapter._final_drive_servo_deg.values())
        actuation=plan(request)
        bound=SemanticActuationDispatch(adapter,actuation,
            policy_headroom_mode=profile['residual']['policy_headroom_mode'],
            tracking_reference_mode=profile['residual']['tracking_reference_mode'],
            tracking_reference_bootstrap_tick=1)
        assert bound.rr_capture_assist is None and bound.nominal_geometry_context is None
        assert bound.rr_carry_wheel_context is None
        receipt=IsaacFSMBackend._atomic_apply(None,bound,zero,physics_tick=1,tracking_servo_names=(),
            drive_feedback_bias_full12=actuation.combined_post_mapper_bias_full12)
        assert receipt['articulation_writes_this_call']==1
        for i,name in enumerate(SERVO_ORDER):
            lower,upper=servo_limits_deg(name)
            assert lower<=receipt['drive_target_full12'][i]<=upper
            assert abs(receipt['drive_target_full12'][i]-before[i])<=adapter.servo_target_mapper.maximum_delta_deg+1e-12
        assert 'rr_capture_assist_evidence' not in receipt and 'rr_carry_wheel_evidence' not in receipt
        receipts.append(receipt)
    baseline,changed=(row['drive_target_full12'] for row in receipts)
    assert (changed[channel]-baseline[channel])*raw_value>0
    assert all(changed[i]==baseline[i] for i in range(12) if i!=channel)


def potential_pair():
    from test_semantic_all_stage_physical_acceptance import new_observation
    from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor
    sup=TaskStageSupervisor(CONFIG/'stage_task_spec.yaml')
    sup.evaluator.observe(new_observation())
    snap=deepcopy(sup.evaluator.snapshot)
    for leg in ('FR','FL','RR'):
        snap['history']['placed'][leg]=True
        snap['history']['active_lift'][leg]=True
        snap['history']['front_edge_crossed'][leg]=True
    rr=snap['current_legs']['RR']
    rr.update(air=True,support=False,ground_contact=False,top_contact=False,top_surface_contact=False,
        obstacle_pair_active=False,bearing_force_n=0.,within_top_xy=True,within_lateral_span=True,clearance_m=.03)
    rl=snap['current_legs']['RL']
    rl.update(current_lift_valid=False,motion_continuation_allowed=False,initial_clearance=True,
        soft_air_actuation_earned=True,air=True,ground_contact=False,clearance_m=.025,
        within_lateral_span=True,within_top_xy=True,front_distance_m=.02,consecutive_top_samples=0)
    snap['history']['active_lift']['RL']=True
    snap['history']['front_edge_crossed']['RL']=True
    snap['transfer_roles']['RL']['transfer_progress']=1.
    old=deepcopy(sup);old.spec['nominal'].pop('rear_policy_timing')
    return sup,old,snap


def test_reward_historical_rr_placement_does_not_reward_new_rl_unload_without_current_support():
    sup,old,snap=potential_pair()
    base=sup.physical_potential(snap)
    assert base < old.physical_potential(snap)
    weak=deepcopy(snap)
    weak['current_legs']['RL'].update(initial_clearance=False,soft_air_actuation_earned=False,clearance_m=0.)
    weak['history']['active_lift']['RL']=weak['history']['front_edge_crossed']['RL']=False
    weak['transfer_roles']['RL']['transfer_progress']=0.
    assert sup.physical_potential(weak)==base
    assert old.physical_potential(weak)<old.physical_potential(snap)
    assert sup.predicate('placed_RR',snap)==old.predicate('placed_RR',snap)


@pytest.mark.parametrize('permission', ['actual_RR_support','already_qualified_RL_air','RL_already_placed'])
def test_reward_keeps_old_progress_for_current_support_existing_swing_or_completed_retention(permission):
    sup,old,snap=potential_pair()
    if permission=='actual_RR_support':
        snap['current_legs']['RR'].update(air=False,ground_contact=False,support=True,bearing_verified=True,
            bearing_force_n=3.,top_contact=True,top_surface_contact=True,obstacle_pair_active=True,contact_surface='TOP')
        snap['current_legs']['FL'].update(support=True,bearing_verified=True,bearing_force_n=3.,air=False,ground_contact=True)
    elif permission=='already_qualified_RL_air':
        snap['current_legs']['RL'].update(current_lift_valid=True,motion_continuation_allowed=True,air=True,ground_contact=False)
    else: snap['history']['placed']['RL']=True
    assert sup.physical_potential(snap)==old.physical_potential(snap)

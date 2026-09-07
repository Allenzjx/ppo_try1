"""Targeted measured/synthetic role counterexamples, never teacher rollouts."""
from copy import deepcopy
from pathlib import Path
import math

import pytest

from test_semantic_supervisor import observation, advance, leg_state, place
from test_semantic_continuous_v3 import whole_body_lift
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator, TaskStageSupervisor, NominalMotionProvider, load_task_spec
from wlr50_clean.reference.motion_contract import load_motion_contract

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT/'configs/ppo_semantic_v3/stage_task_spec.yaml'


def measured():
    obs = observation()
    obs['center_of_mass'] = dict(valid=True, position_w_m=[.30, 0., .10],
        velocity_w_m_s=[0., 0., 0.], total_mass_kg=3., included_bodies=['synthetic_13_body_sum'])
    for j in obs['joints'].values():
        j['command_deg'] = 0.
    for name, xy in zip(WHEEL_ORDER, ((.4,.12),(.4,-.12),(.32,.12),(.32,-.12))):
        obs['wheels'][name]['center_w_m'][:2] = xy
        obs['wheels'][name]['bottom_w_m'][:2] = xy
    return obs


def prefix(placed=()):
    ev = TaskEvaluator(SPEC); obs = measured(); ev.observe(obs)
    for leg in placed:
        obs = place(ev, obs, leg)
    return ev, obs


def transfer(ev, obs, target, *, move_com=True, move_receiver=False, n=10):
    receiver = {'FR':'RL','FL':'RR','RR':'FL','RL':'FR'}[target]
    wheel = WHEEL_ORDER[('FL','FR','RL','RR').index(receiver)]
    c = obs['center_of_mass']['position_w_m']
    r = obs['wheels'][wheel]['center_w_m']
    d = [r[0]-c[0], r[1]-c[1], 0.]; length = math.hypot(*d[:2]); d = [x/length for x in d]
    for i in range(n):
        obs = advance(obs)
        obs['joints']['front_left_hip'].update(position_deg=i*.2, command_deg=i*.2+1.)
        obs['wheels']['rear_left_ankle']['command_rad_s'] = .1
        if move_com:
            obs['center_of_mass']['position_w_m'] = [a+.0002*x for a,x in zip(obs['center_of_mass']['position_w_m'],d)]
            obs['center_of_mass']['velocity_w_m_s'] = [x*.024 for x in d]
        if move_receiver:
            for key in ('center_w_m','bottom_w_m'):
                obs['wheels'][wheel][key] = [a-.0002*x for a,x in zip(obs['wheels'][wheel][key],d)]
        snap = ev.observe(obs)
    return obs, snap['transfer_roles'][target]


def test_fr_current_top_enables_first_fl_preparation_without_fl_air():
    ev, obs = prefix(('FR',))
    obs, role = transfer(ev, obs, 'FL')
    assert role['diagonal_receiving_side'] == 'RR'
    assert 'FR' in role['observed_support_contacts']
    assert role['preparation_ready'] and not ev.snapshot['current_legs']['FL']['air']


def test_history_fr_air_is_not_current_support():
    ev, obs = prefix(('FR',)); obs = advance(obs); leg_state(obs,'FR',air=True,bottom=.08)
    role = ev.observe(obs)['transfer_roles']['FL']
    assert ev.snapshot['history']['placed']['FR']
    assert 'FR' not in role['observed_support_contacts']
    assert 'FR' in role['historical_placement_without_current_support']


def test_air_receiver_and_dynamic_two_contacts_allow_measured_rr_transfer():
    ev, obs = prefix(('FR','FL'))
    obs = advance(obs); leg_state(obs,'FL',air=True,bottom=.08); leg_state(obs,'RR',air=True,bottom=.004)
    ev.observe(obs)
    obs = whole_body_lift(ev,obs,'RR',top_clearance=.012)
    obs, role = transfer(ev,obs,'RR')
    assert role['observed_support_contacts'] == ['FR','RL']
    assert role['transfer_ready']
    assert not role['receiver_workspace_state']['receiver_current_support']
    assert not role['transfer_direction_context']['two_contact_static_stability_proven']
    assert not ev.snapshot['history']['front_edge_crossed']['RR']


def test_receiver_motion_does_not_invent_com_translation():
    ev, obs = prefix(('FR','FL'))
    obs, role = transfer(ev,obs,'RR',move_com=False,move_receiver=True)
    ctx = role['transfer_direction_context']
    assert ctx['com_toward_receiver_m'] == 0.
    assert ctx['com_world_displacement_m'] == (0.,0.,0.)
    assert math.dist(ctx['receiver_world_displacement_m'],(0.,0.,0.)) > 0.
    assert role['transfer_progress'] == 0.


def test_air_down_command_cannot_claim_ground_impulse():
    ev, obs = prefix(); obs = advance(obs); leg_state(obs,'RR',air=True,bottom=.03)
    role = ev.observe(obs)['transfer_roles']['RR']
    assert not role['transfer_direction_context']['target_contact_reaction_possible']
    assert role['transfer_direction_context']['ground_reaction_impulse'] is None


@pytest.mark.parametrize('knee',[-40.,-10.,20.])
def test_static_candidate_angle_and_instant_lowload_do_not_complete_transfer(knee):
    ev = TaskEvaluator(SPEC); obs = measured()
    obs['joints']['front_right_knee'].update(position_deg=knee, command_deg=knee)
    leg_state(obs,'RR',air=True,bottom=.001)
    for i in range(10):
        if i: obs = advance(obs)
        role = ev.observe(obs)['transfer_roles']['RR']
    assert not role['transfer_ready'] and role['transfer_progress'] == 0.


def test_p05_pending_capture_can_roll_without_fake_placement():
    ev, obs = prefix(('FR',))
    obs = whole_body_lift(ev,obs,'FL',top_clearance=.08)
    sup = TaskStageSupervisor(SPEC,evaluator=ev,initial_stage_id='P05')
    obs = advance(obs); task = sup.observe_and_update(obs)
    assert task['pending_capture'] and not task['history']['placed']['FL']
    provider = NominalMotionProvider(load_motion_contract(ROOT/'configs/recording_motion_contract.json'), spec=load_task_spec(SPEC))
    for _ in range(20): action = provider.evaluate(task)
    assert min(action[8:]) > 0.
    assert sup.stage_id == 'P05' and not task['history']['placed']['FL']
    obs = advance(obs); leg_state(obs,'FL',x=.53,bottom=.05,top=True)
    first_contact = sup.observe_and_update(obs)
    assert first_contact['pending_capture'] and not first_contact['history']['placed']['FL']
    assert min(provider.evaluate(first_contact)[8:]) > 0.
    obs = advance(obs); completed = sup.observe_and_update(obs)
    assert completed['history']['placed']['FL']


def test_one_new_dropload_does_not_borrow_age_of_old_motion_window():
    ev, obs = prefix()
    obs, _ = transfer(ev,obs,'RR',move_com=False,n=35)
    obs = advance(obs); leg_state(obs,'RR',air=True,bottom=.001)
    role = ev.observe(obs)['transfer_roles']['RR']
    assert not role['transfer_ready']
    assert role['transfer_direction_context']['continued_response_duration_s'] == 0.
    assert role['transfer_progress'] == 0.


def test_old_contact_gap_does_not_force_half_second_stationary_wait():
    ev, obs = prefix(('FR','FL'))
    obs = advance(obs)
    for leg in ('FL','RR','RL'): leg_state(obs,leg,air=True,bottom=.001)
    ev.observe(obs)  # One real short interval with only FR support.
    obs = advance(obs); leg_state(obs,'RL'); ev.observe(obs)
    obs = whole_body_lift(ev,obs,'RR',top_clearance=.012)
    obs, role = transfer(ev,obs,'RR',n=12)
    assert role['transfer_direction_context']['support_continuity_fraction'] < 1.
    assert role['transfer_direction_context']['short_support_continuity_fraction'] == 1.
    assert role['transfer_ready']


def test_p08_lift_continues_into_p09_and_ground_revokes_before_cross():
    ev, obs = prefix(('FR','FL'))
    obs = whole_body_lift(ev,obs,'RR')
    obs, _ = transfer(ev,obs,'RR')
    sup = TaskStageSupervisor(SPEC,evaluator=ev,initial_stage_id='P08')
    for _ in range(16):
        obs = advance(obs); snap = sup.observe_and_update(obs)
        if snap['stage_id'] == 'P09': break
    assert snap['stage_id'] == 'P09' and snap['history']['active_lift']['RR']
    assert snap['termination_reason'] is None
    reference = snap['transfer_roles']['RR']['transfer_direction_context']['reference_tick']
    assert reference == 0  # Whole physical window survives phase handoff.
    phi = sup.physical_potential(ev.snapshot); sup.stage_id='P10'
    assert sup.physical_potential(ev.snapshot) == phi
    obs = advance(obs); leg_state(obs,'RR',bottom=0.)
    assert not ev.observe(obs)['history']['active_lift']['RR']


def test_rr_historical_placement_retreat_changes_rl_current_support():
    ev, obs = prefix(('FR','FL','RR'))
    obs = advance(obs); leg_state(obs,'RR',x=.35,bottom=.08,air=True)
    snap = ev.observe(obs); role = snap['transfer_roles']['RL']
    assert snap['history']['placed']['RR'] and 'RR' not in role['observed_support_contacts']
    assert 'RR' in role['historical_placement_without_current_support']


@pytest.mark.parametrize('failure',['body','wheel_only'])
def test_roles_never_mask_real_task_failures(failure):
    ev, obs = prefix()
    obs = advance(obs)
    if failure == 'body': obs['body_collision']['detected'] = True
    else: leg_state(obs,'RR',x=.53,top=True,bottom=.05)
    assert ev.observe(obs)['termination_reason'] is not None


def test_role_configuration_replaces_not_renames_old_completion_gates():
    spec = load_task_spec(SPEC)
    used = [p for row in spec['stages'].values() for p in row['completion_predicates']]
    assert not any(p.startswith(('workspace_', 'support_', 'load_ready_')) for p in used)
    assert 'role_prepared_RR' in used and 'transfer_ready_RR' in used
    assert all(len(row['allowed_action_channels']) == 12 for row in spec['stages'].values())


def test_existing_qualified_or_placed_motion_never_requires_recreating_preparation():
    ev, obs = prefix(('FR','FL'))
    obs = whole_body_lift(ev,obs,'RR')
    sup = TaskStageSupervisor(SPEC,evaluator=ev,initial_stage_id='P08')
    assert sup.predicate('transfer_ready_RR',ev.snapshot) == 1.
    assert sup.predicate('role_prepared_RR',ev.snapshot) == 1.
    obs = advance(obs); leg_state(obs,'RR',bottom=0.); ev.observe(obs)
    assert not ev.snapshot['history']['active_lift']['RR']
    assert sup.predicate('transfer_ready_RR',ev.snapshot) < 1.


@pytest.mark.parametrize('stage',['P06','P07','P08'])
def test_qualified_crossing_outside_old_edge_interval_continuously_takes_over(stage):
    ev, obs = prefix(('FR','FL'))
    obs = whole_body_lift(ev,obs,'RR')
    obs = advance(obs); leg_state(obs,'RR',x=.58,bottom=.075,air=True); ev.observe(obs)
    assert ev.snapshot['history']['front_edge_crossed']['RR']
    sup = TaskStageSupervisor(SPEC,evaluator=ev,initial_stage_id=stage)
    assert sup.predicate('edge_proximity_RR',ev.snapshot) < 1.
    for _ in range(8):
        obs=advance(obs); snap=sup.observe_and_update(obs)
        if snap['stage_id']!=stage: break
    assert snap['stage_id']!=stage and snap['termination_reason'] is None
    transition=snap['transition_evidence'][-1]
    assert transition['continuous_takeover']
    assert any(v<1. for v in transition['completion_values'].values())  # No fake interval measurement.

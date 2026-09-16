"""Synthetic P13 source ownership tests, never a simulated success receipt."""
from copy import deepcopy

import pytest
import yaml

from test_semantic_source_partial_order_v1 import ROOT, contract, current_input, layer
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider, SemanticObservationError

MODE = "current_physical_stop_nominal_owner_v1"


def provider(contract, *, enabled=True):
    spec = yaml.safe_load((ROOT/'configs/ppo_fsm_reference_p09_stable_v2/stage_task_spec.yaml').read_text())
    if enabled:
        spec['nominal']['final_stop_owner'] = MODE
    else:
        spec['nominal'].pop('final_stop_owner', None)
    return NominalMotionProvider.from_handoff(contract, spec=spec, stage_id='P13',
        nominal_full12=contract.phase('P13').start_full12, tracking_servo_names=())


def stop_task(contract, tick, *, phase='P13', started=True):
    t, raw = current_input(contract, phase, tick)
    ev = t['physical_evaluator']
    ev.update(physical_evidence_status='VERIFIED', final_region_valid=True, final_controlled=True,
        final_support_available=True, task_completed_controlled=True,
        post_completion_observation_started=started, post_completion_observation_complete=False,
        post_completion_loss_observed=False, post_completion_elapsed_s=.05)
    ev['history']['placed'] = {leg: True for leg in ('FL','FR','RL','RR')}
    for row in ev['current_legs'].values():
        row.update(load_fraction_valid=True, load_fraction=.25, support=True, bearing_verified=True,
            bearing_force_n=5., top_surface_contact=True, top_contact=True, air=False,
            ground_contact=False, within_top_xy=True, clearance_m=-.001)
    # Real positive class: FL may be AIR while three OTHER top contacts bear.
    ev['current_legs']['FL'].update(air=True, ground_contact=False, support=False,
        bearing_force_n=0., load_fraction=0., top_contact=False, top_surface_contact=False, clearance_m=.014)
    return t, raw


def issue(p, contract, tick, *, change=None, phase='P13', started=True):
    t, raw = stop_task(contract, tick, phase=phase, started=started)
    if change: change(t, raw)
    before = deepcopy((t, raw))
    result = p.evaluate(t, raw)
    assert (t, raw) == before
    return result


def at_first_source_stop(contract, *, enabled=True):
    p = provider(contract, enabled=enabled)
    issue(p, contract, 100, started=False)
    s = layer(p, 'P13')
    stop = next(w for w in s['motion'].phase.waypoints
                if 16. < w.time_s < 16.8 and w.full12[8:] == (0.,)*4)
    tick = s['motion']._scaled_source_tick(stop.time_s)
    # Jump only the synthetic executor clocks; no real physics snapshot.
    s['ticks'] = s['motion']._tick_index = p._source_motion._tick_index = tick
    prior = p.nominal_full12
    assert any(prior[8:])
    # The real successful-zero window starts only after this issued stop is
    # physically observed; a merely authored stop does not fabricate it.
    stopped = issue(p, contract, 101, started=False)
    assert stopped[8:] == (0.,)*4
    assert p._final_stop_owner is None  # The physical post window has not begun.
    return p


def test_true_issued_stop_with_FL_air_retires_future_home_pulse_not_current_N(contract):
    p = at_first_source_stop(contract)
    old = at_first_source_stop(contract, enabled=False)
    held = p.nominal_full12[:8]
    p.tracking_servo_names = ('rear_right_knee',)
    p.normal_drive_bias_full12 = (.125,)*8+(0.,)*4
    assert held != tuple(p.spec['final']['home_servo_pose_deg'])
    out = issue(p, contract, 102)
    assert out == held+(0.,)*4
    entry = p.nominal_suggestion_diagnostics['final_stop_owner']['entry']
    assert entry['entry_observation_tick'] == 102 and entry['entry_top_bearing_support_count'] == 3
    assert entry['held_nominal_servo_deg'] == held
    assert entry['held_tracking_servo_names'] == ('rear_right_knee',)
    assert entry['held_normal_servo_bias_deg'] == (.125,)*8
    issue(old, contract, 102)
    pulse_seen = endpoint_seen = False
    for tick in range(103, 255):
        out = issue(p, contract, tick)
        baseline = issue(old, contract, tick)
        s, b = layer(p, 'P13'), layer(old, 'P13')
        assert s['ticks'] == b['ticks'] and s['sample'].atomic_groups == b['sample'].atomic_groups
        assert out == held+(0.,)*4
        assert p.tracking_servo_names == ('rear_right_knee',)
        assert p.normal_drive_bias_full12 == (.125,)*8+(0.,)*4
        if s['sample'].source_full12_atomic:
            pulse_seen = True
            assert baseline[8:] == (1.09, -.72, -.43, -.39)
            assert baseline[:8] != held
        endpoint_seen |= s['sample'].endpoint_issued
    assert pulse_seen and endpoint_seen
    d = p.nominal_suggestion_diagnostics['final_stop_owner']
    assert d['active'] and d['source_endpoint_issued']
    assert d['source_clocks_continue_contributions_retired'] and not d['task_success_awarded']


@pytest.mark.parametrize('field,value', [
    ('final_region_valid',False), ('final_support_available',False),
    ('post_completion_observation_started',False),
    ('post_completion_observation_complete',True), ('post_completion_loss_observed',True),
    ('physical_evidence_status','CONTACT_BEARING_UNVERIFIED'),
    ('post_completion_elapsed_s',1.), ('post_completion_elapsed_s',-.01),
])
def test_old_started_or_ineligible_current_state_cannot_acquire(contract, field, value):
    p = at_first_source_stop(contract)
    issue(p, contract, 102, change=lambda t, _: t['physical_evaluator'].update({field:value}))
    assert p._final_stop_owner is None


@pytest.mark.parametrize('fault', ['not_placed','one_top','wall','FL_load_unknown','FL_fake_support',
                                  'FL_missing_force','FL_outside','FL_missing_leg','stale_observation'])
def test_real_four_capture_evidence_and_current_top_support_required(contract, fault):
    p = at_first_source_stop(contract)
    def change(t, raw):
        ev=t['physical_evaluator']; fl=ev['current_legs']['FL']
        if fault == 'not_placed': ev['history']['placed']['RL']=False
        elif fault == 'one_top':
            for leg in ('FR','RL'): ev['current_legs'][leg]['top_surface_contact']=False
        elif fault == 'wall':
            for row in ev['current_legs'].values(): row.update(top_surface_contact=False,ground_contact=False)
        elif fault == 'FL_load_unknown': fl['load_fraction_valid']=False
        elif fault == 'FL_fake_support': fl['support']=True
        elif fault == 'FL_missing_force': fl['bearing_force_n']=None
        elif fault == 'FL_outside': fl['within_top_xy']=False
        elif fault == 'FL_missing_leg': ev['current_legs']['FL']=None
        else: raw['physics_tick']-=1
    if fault == 'stale_observation':
        with pytest.raises(SemanticObservationError,match='differs from current observation'):
            issue(p,contract,102,change=change)
    else: issue(p,contract,102,change=change)
    assert p._final_stop_owner is None


@pytest.mark.parametrize('loss', ['final_controlled','final_region_valid','final_support_available'])
def test_after_acquisition_live_loss_is_visible_but_does_not_replay_source(contract, loss):
    p=at_first_source_stop(contract); held=issue(p,contract,102)
    for tick in range(103,120):
        assert issue(p,contract,tick,change=lambda t,_:t['physical_evaluator'].update({loss:False,
            'task_completed_controlled':False})) == held
    d=p.nominal_suggestion_diagnostics['final_stop_owner']
    assert d['active'] and not d['current_entry_eligibility'] and d['current_evidence'][loss] is False
    assert d['status']=='holding_live_eligibility_lost' and not d['task_success_awarded']
    assert layer(p,'P13')['sample'].tick_index > 2016  # Original pulse clock passed, not rewound.


def test_terminal_or_leaving_P13_retires_owner_and_old_started_cannot_reacquire(contract):
    p=at_first_source_stop(contract); issue(p,contract,102)
    issue(p,contract,103,change=lambda t,_:t.update(termination_reason='INCOMPLETE_CONTROLLER_BLOCKED'))
    d=p.nominal_suggestion_diagnostics['final_stop_owner']
    assert not d['active'] and p._final_stop_owner_retired
    issue(p,contract,104)
    assert not p.nominal_suggestion_diagnostics['final_stop_owner']['active']
    fresh=at_first_source_stop(contract); issue(fresh,contract,102)
    issue(fresh,contract,103,phase='P12')
    assert fresh._final_stop_owner_retired
    issue(fresh,contract,104)
    assert not fresh.nominal_suggestion_diagnostics['final_stop_owner']['active']
    assert provider(contract)._final_stop_owner is None


def test_unknown_mode_rejected_and_unrelated_phase_does_not_capture(contract):
    p=provider(contract); p.spec['nominal']['final_stop_owner']='unknown'
    with pytest.raises(ValueError,match='unknown nominal final stop'):
        NominalMotionProvider(contract,spec=p.spec)
    p=provider(contract)
    issue(p,contract,100,phase='P12')
    assert p._final_stop_owner is None


def test_raw_residual_bridge_remains_independent_of_nominal_stop_owner(contract):
    from wlr50_clean.ppo.phase_action_masks_v2 import PhaseTransitionBridge
    from wlr50_clean.ppo.semantic_backend import build_semantic_projector
    p=at_first_source_stop(contract); nominal=issue(p,contract,102)
    bridge=PhaseTransitionBridge(build_semantic_projector(ROOT/'configs/ppo_fsm_reference_p09_stable_v2/execution_profile.yaml'))
    raw=(.2,)*12
    projection=bridge.project_tick(raw,state_id='P13',nominal_action_full12=nominal,
        reference_action_full12=nominal,reference_delta_full12=(0.,)*12,dt_s=1/120).projection
    assert projection.raw_residual_full12==raw and projection.effective_action_mask_full12==(1,)*12
    assert all(x!=0. for x in projection.safe_projected_residual_full12)
    assert all(x!=0. for x in projection.applied_action_full12[8:])


@pytest.mark.parametrize('currently_controlled', [True, False])
def test_real_started_window_can_stop_first_P13_before_any_P13_source_sample(contract, currently_controlled):
    spec = yaml.safe_load((ROOT/'configs/ppo_fsm_reference_p09_stable_v2/stage_task_spec.yaml').read_text())
    prior = tuple(contract.phase('P12').end_full12[:8]) + (0.,)*4
    p = NominalMotionProvider.from_handoff(contract, spec=spec, stage_id='P12',
        nominal_full12=prior, tracking_servo_names=('rear_left_knee',))
    assert p.state_id == 'P12'
    assert not any(s['stage']=='P13' for s in p._continuous_layers)
    before = p.nominal_full12[:8]
    def changed(t, raw):
        t['physical_evaluator'].update(final_controlled=currently_controlled,
            task_completed_controlled=currently_controlled)
    result = issue(p, contract, 6064, change=changed)
    assert result == before+(0.,)*4
    assert layer(p,'P13')['sample'].full12[8:] == (.3,)*4
    d=p.nominal_suggestion_diagnostics['final_stop_owner']
    assert d['active'] and not d['preceding_issued_nominal_wheels_stopped']
    assert d['current_post_window_takeover_eligibility']
    assert d['current_entry_eligibility'] is currently_controlled
    assert d['acquisition_semantics']=='post_window_triggered_nominal_stop_takeover_v2'
    assert d['entry']['held_nominal_servo_deg']==before
    assert d['entry']['held_tracking_servo_names']==('rear_left_knee',)
    assert not d['task_success_awarded']


def test_all_placed_and_currently_slow_without_started_window_cannot_skip_source(contract):
    p=provider(contract)
    result=issue(p,contract,6064,started=False)
    assert result[8:]==(.3,)*4
    assert p._final_stop_owner is None
    assert not p.nominal_suggestion_diagnostics['final_stop_owner']['current_post_window_takeover_eligibility']


@pytest.mark.parametrize('terminal_at', ['task','evaluator'])
def test_first_P13_started_window_cannot_override_a_terminal_or_hard_failure(contract, terminal_at):
    p=provider(contract)
    def changed(t,raw):
        destination=t if terminal_at=='task' else t['physical_evaluator']
        destination['termination_reason']='TASK_FAILURE_BODY_COLLISION'
    issue(p,contract,6064,change=changed)
    assert p._final_stop_owner is None
    assert not p.nominal_suggestion_diagnostics['final_stop_owner']['active']

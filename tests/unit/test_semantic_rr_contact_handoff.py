"""Synthetic contact/deadline regression; never Isaac or learned-task credit."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_semantic_all_stage_physical_acceptance import set_leg
from test_semantic_rr_capture_context import measured, top_rr, assist_state, SUPPORT_SPEC
from test_semantic_rr_capture_supervisor import airborne_rr
from test_semantic_supervisor import advance
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.semantic_rr_capture_assist import (
    RRHipOnlyCaptureAssist, rr_capture_assist_context, apply_rr_capture_assist_snapshot,
)
from wlr50_clean.ppo.semantic_rr_capture_context import (
    RR_CONTACT_HANDOFF_MODE, rr_capture_transfer_context, rr_contact_handoff_window_s,
)


WINDOW = 10. / 120.


class ContactTrace:
    """Adjacent physical-fixture sensing with production assist/evaluator.

    Deadline origins are positioned explicitly, not simulated wall time.
    Feedback is a synthetic committed-envelope fixture, not a native ACK.
    """
    def __init__(self, pair):
        self.sup, self.obs = pair
        self.assist = RRHipOnlyCaptureAssist()
        self.previous = tuple(self.obs['joints'][name]['position_deg'] for name in SERVO_ORDER) + (0.,) * 4

    def step(self, surface='AIR', *, local_age=20., episode_age=100., weak=False,
             feedback_offset=0, missing_feedback=False, fault=None):
        sup, old = self.sup, self.obs
        obs = advance(old)
        context = rr_capture_assist_context(task=sup.snapshot, observation=old,
            source_frame=SimpleNamespace(state_id=sup.stage_id), physics_tick=obs['physics_tick'],
            support_spec=sup.spec['support'])
        receipt = self.assist.advance(context=context, previous_final_full12=self.previous, physics_dt_s=1. / 120.)
        self.previous = apply_rr_capture_assist_snapshot(self.previous, receipt['state_after'])
        set_leg(obs, 'RR', x=.53, bottom=.05 if surface == 'TOP' else 0. if surface == 'GROUND' else .075,
                surface=surface)
        if weak:
            body = obs['wheels'][WHEEL_ORDER[3]]['body_name']
            pair = obs['contacts'][body]['obstacle']
            force = sup.spec['support']['force_noise_floor_n'] / 2.
            pair.update(normal_force_n=force, force_w_n=[0., 0., force])
        if fault == 'invalid':
            obs['geometry_pose_aware'] = False
        elif fault == 'body_collision':
            obs['body_collision']['detected'] = True
            obs['contacts']['base_link']['obstacle'].update(
                active=True, normal_force_n=1., force_w_n=[1., 0., 0.])
        now = obs['simulation_time_s']
        if sup.stage_id == 'P09':
            sup.stage_started_s = now - local_age
        sup.episode_started_s = now - episode_age
        sup.rr_capture_feedback = None if missing_feedback else {
            'episode_observation_tick': obs['physics_tick'] + feedback_offset,
            'state': receipt['state_after'],
        }
        task = sup.observe_and_update(obs)
        self.obs = obs
        return task, receipt['state_after']

    def align_second_top_mod8(self, value):
        count = (value - (self.obs['physics_tick'] + 2)) % 8
        for _ in range(count):
            assert self.step()[0]['termination_reason'] is None


@pytest.mark.parametrize('placed_mod8', [1, 3, 7])
def test_expired_local_capture_waits_for_normal_phase_boundary(airborne_rr, placed_mod8):
    trace = ContactTrace(airborne_rr)
    assert trace.sup.spec['rr_contact_handoff_semantics'] == RR_CONTACT_HANDOFF_MODE
    assert trace.sup._rr_contact_handoff_window_s == WINDOW
    trace.align_second_top_mod8(placed_mod8)
    first, _ = trace.step('TOP', local_age=45.)
    rr = first['physical_evaluator']['current_legs']['RR']
    assert rr['consecutive_top_samples'] == 1 and not first['placed_history']['RR']
    assert first['termination_reason'] is None and first['stage_id'] == 'P09'
    second, state = trace.step('TOP', local_age=45.)
    assert second['placed_history']['RR'] and state['mode_name'] == 'HOLD'
    assert second['termination_reason'] is None and second['stage_id'] == 'P09'
    assert second['rr_capture_continuation']['rr_completed_contact_handoff_pending']
    assert not second['success'] and 'P09' not in second['completed_stage_ids']
    for _ in range(8 - placed_mod8):
        last, _ = trace.step('TOP', local_age=45.)
        assert last['termination_reason'] is None
    assert trace.obs['physics_tick'] % 8 == 0 and last['stage_id'] == 'P10'
    assert 'P09' in last['completed_stage_ids'] and not last['success']


def test_weak_pair_alone_is_not_top_but_following_real_top_has_confirmation_opportunity(airborne_rr):
    trace = ContactTrace(airborne_rr); trace.align_second_top_mod8(3)
    weak, _ = trace.step('TOP', weak=True)
    assert not weak['physical_evaluator']['current_legs']['RR']['top_contact']
    assert not weak['placed_history']['RR']
    assert not weak['rr_capture_continuation']['rr_contact_confirmation_allowed']
    first, state = trace.step('TOP', local_age=45.)
    assert state['mode_name'] == 'HOLD'
    assert first['physical_evaluator']['current_legs']['RR']['consecutive_top_samples'] == 1
    assert not first['placed_history']['RR'] and first['termination_reason'] is None
    assert first['rr_capture_continuation']['rr_contact_confirmation_allowed']
    second, _ = trace.step('TOP', local_age=45.)
    assert second['placed_history']['RR'] and second['termination_reason'] is None


@pytest.mark.parametrize('surface', ['AIR', 'GROUND'])
def test_current_contact_loss_does_not_borrow_real_placement_history(airborne_rr, surface):
    trace = ContactTrace(airborne_rr); trace.align_second_top_mod8(1)
    assert trace.step('TOP')[0]['termination_reason'] is None
    assert trace.step('TOP')[0]['placed_history']['RR']
    last, _ = trace.step(surface, local_age=45.)
    assert not last['physical_evaluator']['current_legs']['RR']['top_contact']
    assert not last['rr_capture_continuation']['local_warning_only']
    assert not last['rr_capture_continuation']['rr_completed_contact_handoff_pending']
    assert last['termination_reason'] == 'INCOMPLETE_CONTROLLER_BLOCKED'
    if surface == 'AIR':
        assert last['completion_values']['placed_RR'] == 1.  # Deliberate negative case.


@pytest.mark.parametrize('placed_mod8', [0, 3])
def test_global_deadline_still_wins_after_real_placement(airborne_rr, placed_mod8):
    trace = ContactTrace(airborne_rr); trace.align_second_top_mod8(placed_mod8)
    assert trace.step('TOP')[0]['termination_reason'] is None
    last, _ = trace.step('TOP', local_age=45., episode_age=200.)
    assert last['placed_history']['RR']
    assert last['termination_source'] == 'GLOBAL_FINITE_TASK_DEADLINE'
    assert not last['success']


@pytest.mark.parametrize('kind', ['missing', 'stale', 'future'])
def test_missing_or_uncommitted_feedback_cannot_waive_contact_deadline(airborne_rr, kind):
    trace = ContactTrace(airborne_rr); trace.align_second_top_mod8(3)
    assert trace.step('TOP')[0]['termination_reason'] is None
    last, _ = trace.step('TOP', local_age=45., missing_feedback=kind == 'missing',
                        feedback_offset=-1 if kind == 'stale' else 1 if kind == 'future' else 0)
    assert last['placed_history']['RR']
    assert not last['rr_capture_continuation']['committed_feedback_matches_current_tick']
    assert not last['rr_capture_continuation']['local_warning_only']
    assert last['termination_source'] == 'LOCAL_BOUNDED_RECOVERY_EXHAUSTED'


@pytest.mark.parametrize('fault,source', [('invalid', 'UNVERIFIED_SENSOR'), ('body_collision', 'BODY_CONTACT')])
def test_current_sensor_or_physical_failure_wins_over_new_contact_permission(airborne_rr, fault, source):
    trace = ContactTrace(airborne_rr); trace.align_second_top_mod8(3)
    assert trace.step('TOP')[0]['termination_reason'] is None
    last, _ = trace.step('TOP', local_age=45., episode_age=200., fault=fault)
    assert last['termination_source'] == source
    assert not last['rr_capture_continuation']['rr_contact_confirmation_allowed']
    assert not last['rr_capture_continuation']['rr_completed_contact_handoff_pending']


def test_contact_toggle_preserves_existing_cumulative_hold_clock(airborne_rr):
    trace = ContactTrace(airborne_rr); trace.align_second_top_mod8(3)
    trace.step('TOP')
    _, second = trace.step('TOP')
    _, loss = trace.step('AIR')
    _, recontact = trace.step('TOP')
    _, confirmed = trace.step('TOP')
    assert second['hold_elapsed_s'] == pytest.approx(1. / 120.)
    assert loss['hold_elapsed_s'] == pytest.approx(2. / 120.)
    assert recontact['hold_elapsed_s'] == loss['hold_elapsed_s']
    assert confirmed['hold_elapsed_s'] == pytest.approx(3. / 120.)


def context_inputs(*, placed=False, elapsed=0., mode='HOLD', modulo=3, **rr_changes):
    task, obs = measured(); top_rr(task)
    task.update(stage_id='P09', entry_valid=True,
                completion_values={'placed_RR': 1. if placed else .99})
    task['physical_evaluator']['physics_tick'] = 104 + modulo
    task['physical_evaluator']['history']['placed']['RR'] = placed
    task['physical_evaluator']['current_legs']['RR'].update(rr_changes)
    assist = RRHipOnlyCaptureAssist.from_snapshot(assist_state(mode))
    assist.state['hold_elapsed_s'] = elapsed
    return dict(task=task, observation=obs, support_spec=SUPPORT_SPEC,
                assist_snapshot=assist.snapshot(), contact_handoff_window_s=WINDOW)


def context(**kwargs):
    return rr_capture_transfer_context(**context_inputs(**kwargs))


@pytest.mark.parametrize('elapsed,allowed', [(0., True), (WINDOW, True), (WINDOW + 1. / 120., False)])
def test_unfinished_capture_confirmation_uses_existing_finite_cumulative_clock(elapsed, allowed):
    value = context(elapsed=elapsed)
    assert value['rr_contact_confirmation_allowed'] is allowed
    assert not value['rr_completed_contact_handoff_pending']
    assert value['rr_capture_recovery_allowed'] is allowed


def test_completed_current_top_can_wait_next_boundary_after_confirmation_budget_spent():
    value = context(placed=True, elapsed=2., modulo=3)
    assert not value['rr_contact_confirmation_allowed']
    assert value['rr_completed_contact_handoff_pending'] and value['rr_capture_recovery_allowed']
    value = context(placed=True, elapsed=2., modulo=0)
    assert not value['rr_contact_confirmation_allowed']
    assert not value['rr_completed_contact_handoff_pending']
    assert not value['rr_capture_recovery_allowed']


@pytest.mark.parametrize('changes', [
    {'air': True}, {'ground_contact': True}, {'top_contact': False},
    {'top_surface_contact': False}, {'bearing_verified': False},
    {'bearing_force_n': 0.}, {'bearing_force_n': float('nan')},
    {'within_top_xy': False}, {'within_lateral_span': False},
    {'current_lift_valid': False}, {'obstacle_pair_active': False},
    {'contact_surface': 'FRONT'},
])
def test_current_evidence_not_historical_completion_controls_handoff(changes):
    value = context(placed=True, **changes)
    assert not value['rr_contact_confirmation_allowed']
    assert not value['rr_completed_contact_handoff_pending']
    assert not value['rr_capture_recovery_allowed']


@pytest.mark.parametrize('mode', ['WAIT', 'RELEASE', 'RELEASED'])
def test_nonowned_capture_modes_cannot_extend_local_episode(mode):
    assert not context(placed=True, mode=mode)['rr_capture_recovery_allowed']


@pytest.mark.parametrize('mode', ['DESCEND', 'HOLD', 'BLOCKED'])
def test_owned_capture_modes_can_confirm_actual_first_top(mode):
    assert context(mode=mode)['rr_contact_confirmation_allowed']


@pytest.mark.parametrize('kind', ['not_qualified', 'not_crossed', 'one_other_support', 'retired', 'physical_abort'])
def test_completion_does_not_replace_current_qualification_support_or_safety(kind):
    args = context_inputs(placed=True)
    ev = args['task']['physical_evaluator']
    if kind == 'not_qualified':
        ev['history']['active_lift']['RR'] = False
    elif kind == 'not_crossed':
        ev['history']['front_edge_crossed']['RR'] = False
    elif kind == 'one_other_support':
        for leg in ('FR', 'FL'):
            ev['current_legs'][leg].update(air=True, support=False, ground_contact=False, bearing_force_n=0.)
    elif kind == 'retired':
        assist = RRHipOnlyCaptureAssist.from_snapshot(args['assist_snapshot'])
        assist.state['retired'] = 1.
        args['assist_snapshot'] = assist.snapshot()
    else:
        ev['termination_reason'] = 'BODY_COLLISION'
    value = rr_capture_transfer_context(**args)
    assert not value['rr_contact_confirmation_allowed']
    assert not value['rr_completed_contact_handoff_pending']
    assert not value['rr_capture_recovery_allowed']


@pytest.mark.parametrize('kind', ['wrong_phase', 'entry_invalid', 'goal_incomplete', 'empty_goals', 'not_placed'])
def test_completed_pending_needs_actual_complete_task_not_only_contact(kind):
    args = context_inputs(placed=True, elapsed=2.)
    task = args['task']
    if kind == 'wrong_phase': task['stage_id'] = 'P08'
    elif kind == 'entry_invalid': task['entry_valid'] = False
    elif kind == 'goal_incomplete': task['completion_values']['placed_RR'] = .99
    elif kind == 'empty_goals': task['completion_values'] = {}
    else: task['physical_evaluator']['history']['placed']['RR'] = False
    value = rr_capture_transfer_context(**args)
    assert not value['rr_contact_confirmation_allowed']
    assert not value['rr_completed_contact_handoff_pending']
    assert not value['rr_capture_recovery_allowed']


def test_context_has_no_mutations_and_requires_explicit_opt_in():
    args = context_inputs(placed=True)
    before = deepcopy(args)
    assert rr_capture_transfer_context(**args)['rr_capture_recovery_allowed']
    assert args == before
    del args['contact_handoff_window_s']
    value = rr_capture_transfer_context(**args)
    assert value['rr_contact_handoff_window_s'] == 0.
    assert not value['rr_capture_recovery_allowed']


@pytest.mark.parametrize('window', [True, None, -1., float('nan'), float('inf')])
def test_context_rejects_invalid_window(window):
    args = context_inputs()
    args['contact_handoff_window_s'] = window
    with pytest.raises(ValueError, match='finite and nonnegative'):
        rr_capture_transfer_context(**args)


def test_opt_in_window_uses_existing_top_samples_and_physics_decision_clocks(airborne_rr):
    sup, _ = airborne_rr
    spec = deepcopy(sup.spec)
    assert rr_contact_handoff_window_s(spec) == WINDOW
    del spec['rr_contact_handoff_semantics']
    assert rr_contact_handoff_window_s(spec) == 0.


@pytest.mark.parametrize('key,value', [('rr_contact_handoff_semantics', 'unknown'),
    ('rr_capture_continuation_semantics', None), ('physical_acceptance_version', None),
    ('physics_hz', 60.), ('decision_hz', 30.), ('minimum_top_samples', 0),
    ('minimum_top_samples', True)])
def test_opt_in_contract_rejects_unreviewed_mode_or_invalid_cadence(airborne_rr, key, value):
    sup, _ = airborne_rr
    spec = deepcopy(sup.spec)
    if key == 'minimum_top_samples': spec['history'][key] = value
    else: spec[key] = value
    with pytest.raises(ValueError, match='RR contact handoff'):
        rr_contact_handoff_window_s(spec)

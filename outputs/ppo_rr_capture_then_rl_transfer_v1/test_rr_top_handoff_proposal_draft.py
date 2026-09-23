"""UNEXECUTED DRAFT for the unapplied contact-handoff candidate.

No Isaac/actor/checkpoint work. Expect failure on unchanged e24a production.
Run only after the explicit candidate mode is implemented and opted into.
Synthetic adjacent measurements are not physical success evidence.
"""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/unit'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_semantic_rr_capture_supervisor import airborne_rr
from test_capture_handoff_counterexample import Trace
from test_semantic_rr_capture_context import measured, top_rr, assist_state, SUPPORT_SPEC
from wlr50_clean.ppo.semantic_rr_capture_context import rr_capture_transfer_context


WINDOW = 10. / 120.


@pytest.mark.parametrize('placed_mod8', [1, 3, 7])
def test_expired_local_capture_waits_for_normal_phase_boundary(airborne_rr, placed_mod8):
    trace = Trace(airborne_rr)
    assert trace.sup.spec['rr_contact_handoff_semantics'] == 'current_TOP_cumulative_HOLD_next_decision_v1'
    trace.align_second_top_mod8(placed_mod8)
    first = trace.step('TOP', local_age=45.)
    assert first['consecutive_top_samples'] == 1 and not first['placed_history']
    assert first['termination'] is None and first['stage'] == 'P09'
    second = trace.step('TOP', local_age=45.)
    assert second['placed_history'] and second['current_TOP']
    assert second['assist_mode'] == 'HOLD' and second['termination'] is None
    assert second['stage'] == 'P09'
    for _ in range(8 - placed_mod8):
        last = trace.step('TOP', local_age=45.)
        assert last['termination'] is None
    assert last['tick_mod8'] == 0 and last['stage'] == 'P10'


def test_weak_pair_is_not_top_but_following_real_top_has_confirmation_opportunity(airborne_rr):
    trace = Trace(airborne_rr); trace.align_second_top_mod8(3)
    weak = trace.step('TOP', weak=True)
    assert not weak['current_TOP'] and not weak['placed_history']
    first = trace.step('TOP', local_age=45.)
    assert first['assist_mode'] == 'HOLD' and first['consecutive_top_samples'] == 1
    assert not first['placed_history'] and first['termination'] is None
    second = trace.step('TOP', local_age=45.)
    assert second['placed_history'] and second['termination'] is None


@pytest.mark.parametrize('surface', ['AIR', 'GROUND'])
def test_current_contact_loss_does_not_borrow_real_placement_history(airborne_rr, surface):
    trace = Trace(airborne_rr); trace.align_second_top_mod8(1)
    assert trace.step('TOP')['termination'] is None
    assert trace.step('TOP')['placed_history']
    last = trace.step(surface, local_age=45.)
    assert not last['current_TOP'] and not last['warning_only']
    assert last['termination'] == 'INCOMPLETE_CONTROLLER_BLOCKED'


def test_global_deadline_still_wins_after_real_placement(airborne_rr):
    trace = Trace(airborne_rr); trace.align_second_top_mod8(3)
    assert trace.step('TOP')['termination'] is None
    last = trace.step('TOP', local_age=45., episode_age=200.)
    assert last['placed_history']
    assert last['termination_source'] == 'GLOBAL_FINITE_TASK_DEADLINE'


def context(*, placed=False, elapsed=0., mode='HOLD', modulo=3, **rr_changes):
    task, obs = measured(); top_rr(task)
    task.update(stage_id='P09', entry_valid=True,
                completion_values={'placed_RR': 1. if placed else .99})
    task['physical_evaluator']['physics_tick'] = 104 + modulo
    task['physical_evaluator']['history']['placed']['RR'] = placed
    task['physical_evaluator']['current_legs']['RR'].update(rr_changes)
    snapshot = assist_state(mode)
    # A fixture state must be rebuilt, not mutate the metadata-derived snapshot.
    from wlr50_clean.ppo.semantic_rr_capture_assist import RRHipOnlyCaptureAssist
    assist = RRHipOnlyCaptureAssist.from_snapshot(snapshot)
    assist.state['hold_elapsed_s'] = elapsed
    snapshot = assist.snapshot()
    return rr_capture_transfer_context(task=task, observation=obs, support_spec=SUPPORT_SPEC,
        assist_snapshot=snapshot, contact_handoff_window_s=WINDOW)


@pytest.mark.parametrize('elapsed,allowed', [(0., True), (WINDOW, True), (WINDOW+1./120., False)])
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


@pytest.mark.parametrize('changes', [
    {'air': True}, {'ground_contact': True}, {'top_contact': False},
    {'top_surface_contact': False}, {'bearing_verified': False},
    {'bearing_force_n': 0.}, {'bearing_force_n': float('nan')},
    {'within_top_xy': False}, {'within_lateral_span': False},
    {'current_lift_valid': False},
])
def test_current_evidence_not_historical_completion_controls_handoff(changes):
    value = context(placed=True, **changes)
    assert not value['rr_contact_confirmation_allowed']
    assert not value['rr_completed_contact_handoff_pending']
    assert not value['rr_capture_recovery_allowed']


@pytest.mark.parametrize('mode', ['WAIT', 'RELEASE', 'RELEASED'])
def test_nonowned_capture_modes_cannot_extend_local_episode(mode):
    value = context(placed=True, mode=mode)
    assert not value['rr_capture_recovery_allowed']

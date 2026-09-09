"""Synthetic current-body PBRS / reversible final timer, not a live success."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from test_all_stage_supervisor_contract import SPEC
from test_semantic_all_stage_physical_acceptance import finished_setup
from test_semantic_supervisor import advance
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER


def setup():
    ev, obs = finished_setup()
    return TaskStageSupervisor(SPEC, evaluator=ev, initial_stage_id="P13"), obs


def test_current_body_approach_has_gradient_without_early_event_or_completion():
    sup, obs = setup()
    obs['body_bounds_w_m']['base_link']['minimum_m'][0] = .40
    first = sup.evaluator.observe(obs)
    obs = advance(obs)
    obs['body_bounds_w_m']['base_link']['minimum_m'][0] = .45
    second = sup.evaluator.observe(obs)
    assert .85 < sup.physical_potential(first) < sup.physical_potential(second) < 1.
    assert not second['traversal_event_observed'] and not second['success']
    assert sup.predicate('whole_task_success', second) == 0.
    obs = advance(obs)
    obs['body_bounds_w_m']['base_link']['minimum_m'][0] = .40
    assert sup.physical_potential(sup.evaluator.observe(obs)) == pytest.approx(sup.physical_potential(first))


@pytest.mark.parametrize('side', ['back', 'left', 'right'])
def test_body_crossed_front_but_outside_other_rectangle_edges_is_not_full(side):
    sup, obs = setup()
    bound = obs['body_bounds_w_m']['base_link']
    if side == 'back': bound['maximum_m'][0] = 2.1
    elif side == 'left': bound['maximum_m'][1] = .9
    else: bound['minimum_m'][1] = -.9
    snap = sup.evaluator.observe(obs)
    assert snap['body_traversal_geometry']['outside_platform_distance_m'] > 0.
    assert not snap['traversal_event_observed']
    assert sup._current_finish_progress(snap) < .9


def test_nonmaximum_wheel_deceleration_has_credit_but_hard_max_not_passed():
    sup, obs = setup()
    obs['wheels'][WHEEL_ORDER[0]]['velocity_rad_s'] = 2.
    obs['wheels'][WHEEL_ORDER[1]]['velocity_rad_s'] = 1.
    first = sup.evaluator.observe(obs)
    obs = advance(obs)
    obs['wheels'][WHEEL_ORDER[1]]['velocity_rad_s'] = .5
    second = sup.evaluator.observe(obs)
    assert sup.physical_potential(second) > sup.physical_potential(first)
    assert not second['task_completed_controlled'] and not second['success']


def test_command_home_and_latched_event_cannot_replace_current_finish():
    sup, obs = setup()
    snap = sup.evaluator.observe(obs)
    other = deepcopy(snap)
    other['home_maximum_servo_error_deg'] = 55.
    other['applied_wheel_command_rad_s'] = (1., 1., 1., 1.)
    other['maximum_commanded_wheel_speed_rad_s'] = 1.
    assert sup.physical_potential(other) == sup.physical_potential(snap)
    other['body_traversal_geometry']['outside_platform_distance_m'] = .1
    assert other['traversal_event_observed']
    assert sup.physical_potential(other) < sup.physical_potential(snap)
    other['physical_evidence_status'] = 'CONTACT_BEARING_UNVERIFIED'
    assert sup._current_finish_progress(other) == 0.


@pytest.mark.parametrize('loss', [False, True])
def test_existing_progress_slot_exposes_full_timer_including_final_tenth(loss):
    sup, obs = setup()
    snap = sup.evaluator.observe(obs)
    assert snap['post_completion_observation_started']
    values = []
    for elapsed in (0., .5, .90, .95, .99, 1.):
        current = deepcopy(snap)
        current.update(post_completion_elapsed_s=elapsed, post_completion_loss_observed=loss)
        p = sup.predicate('whole_task_success', current)
        assert (p - (.8 if loss else .5)) / (.19 if loss else .25) == pytest.approx(elapsed)
        assert p < 1.
        values.append(p)
    assert all(a < b for a, b in zip(values, values[1:]))
    current['success'] = True
    assert sup.predicate('whole_task_success', current) == 1.


def test_timer_start_zero_is_distinct_from_event_without_timer():
    sup, obs = setup()
    snap = sup.evaluator.observe(obs)
    assert sup.predicate('whole_task_success', snap) == .5
    snap['post_completion_observation_started'] = False
    assert sup.predicate('whole_task_success', snap) == .25
    snap['traversal_event_observed'] = False
    assert sup.predicate('whole_task_success', snap) == 0.


def test_late_started_fixed_post_window_is_not_cut_by_local_limit():
    sup, obs = setup()
    snap = sup.evaluator.observe(obs)
    sup.evaluator = SimpleNamespace(observe=lambda _: deepcopy(snap), snapshot=snap)
    sup.stage_started_s = sup.episode_started_s = 0.
    obs['physics_tick'] = 67*120
    obs['simulation_time_s'] = 67.
    first = sup.observe_and_update(obs)
    assert first['termination_reason'] is None and first['substage'] == 'CAPTURE'
    assert first['local_timeout']['effective_limit_s'] == pytest.approx(68.)
    snap['post_completion_elapsed_s'] = .5
    obs['physics_tick'] += 60; obs['simulation_time_s'] += .5
    second = sup.observe_and_update(obs)
    assert second['local_timeout']['effective_limit_s'] == pytest.approx(68.)
    assert second['termination_reason'] is None
    snap['success'] = True; snap['post_completion_elapsed_s'] = 1.
    obs['physics_tick'] += 60; obs['simulation_time_s'] += .5
    assert sup.observe_and_update(obs)['termination_reason'] == 'SUCCESS'


def test_fixed_post_window_cannot_extend_global_200_seconds():
    sup, obs = setup()
    snap = sup.evaluator.observe(obs)
    sup.evaluator = SimpleNamespace(observe=lambda _: deepcopy(snap), snapshot=snap)
    sup.stage_started_s = 135.; sup.episode_started_s = 0.
    obs['physics_tick'] = 24000; obs['simulation_time_s'] = 200.
    assert sup.observe_and_update(obs)['termination_source'] == 'GLOBAL_FINITE_TASK_DEADLINE'

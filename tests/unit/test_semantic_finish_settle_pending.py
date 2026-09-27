"""Five bounded groups; synthetic sensing is not a new physical success."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_all_stage_supervisor_contract import SPEC
from test_semantic_all_stage_physical_acceptance import new_spec, new_observation, early_top_place, set_leg
from test_semantic_supervisor import advance
from test_semantic_p13_final_stop_owner import provider, stop_task
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator, TaskStageSupervisor, LEG_ORDER
from wlr50_clean.reference.motion_contract import load_motion_contract
from test_semantic_source_partial_order_v1 import ROOT


def ready(enabled=True):
    ev = TaskEvaluator(spec=new_spec(), finish_recovery_enabled=enabled)
    obs = new_observation()
    ev.observe(obs)
    for leg in ('FR', 'FL', 'RR', 'RL'):
        obs = early_top_place(ev, obs, leg)
    obs = advance(obs)
    for index, leg in enumerate(LEG_ORDER):
        set_leg(obs, leg, x=.7+.03*index, bottom=.05, surface='TOP')
    obs['base']['position_w_m'] = [.8, 0., .12]
    obs['body_bounds_w_m']['base_link'] = {'minimum_m': [.7, -.05, .07], 'maximum_m': [.9, .05, .17]}
    ev.observe(obs)
    return ev, obs


def steps(ev, obs, count):
    for _ in range(count):
        obs = advance(obs)
        ev.observe(obs)
    return obs


def miss(kind='wheel', enabled=True):
    ev, obs = ready(enabled)
    if kind == 'linear':
        obs['base']['linear_velocity_w_m_s'] = [.05001, 0., 0.]
    elif kind == 'angular':
        obs['base']['angular_velocity_w_rad_s'] = [0., .30001, 0.]
    else:
        obs['wheels'][WHEEL_ORDER[-1]]['velocity_rad_s'] = .25001
    return ev, steps(ev, obs, 120)


def stopped(obs):
    obs = deepcopy(obs)
    obs['base']['linear_velocity_w_m_s'] = [.05, 0., 0.]
    obs['base']['angular_velocity_w_rad_s'] = [0., .30, 0.]
    for wheel in obs['wheels'].values():
        wheel.update(velocity_rad_s=.25, command_rad_s=0.)
    return obs


def test_default_unchanged_and_optin_prefix_matches_every_existing_field():
    old, obs = ready(False)
    new, _ = ready(True)
    extra = {key for key in new.snapshot if key.startswith('finish_') or key == 'evaluation_timing_version'}
    assert extra == {'finish_settle_pending', 'finish_endpoint_missed', 'finish_clean_elapsed_norm',
        'finish_recovery_semantics', 'finish_endpoint_miss', 'finish_clean_since_s', 'finish_clean_restarts',
        'evaluation_timing_version'}
    assert {key: value for key, value in new.snapshot.items() if key not in extra} == old.snapshot
    obs['wheels'][WHEEL_ORDER[-1]]['velocity_rad_s'] = .251
    for index in range(1, 121):
        obs = advance(obs)
        a, b = old.observe(obs), new.observe(obs)
        if index < 120:
            assert a == {key: value for key, value in b.items() if key not in extra}
    assert a['termination_reason'] == 'INCOMPLETE_CONTROLLER_BLOCKED'
    assert a['reason'] == 'not currently controlled at fixed post-completion observation end'
    assert not any(key.startswith('finish_') for key in a)
    assert b['termination_reason'] is None and b['finish_settle_pending']
    assert b['finish_endpoint_missed'] and not b['success']


def test_three_original_speed_thresholds_and_clean_window_reset_not_command_credit():
    for kind in ('linear', 'angular', 'wheel'):
        ev, obs = miss(kind)
        first_clock = ev._completion_observation_since
        record = deepcopy(ev.snapshot['finish_endpoint_miss'])
        assert ev.snapshot['finish_settle_pending'] and not ev.snapshot['success']
        assert all(wheel['command_rad_s'] == 0. for wheel in obs['wheels'].values())
        obs = stopped(obs)
        obs = steps(ev, obs, 61)
        assert ev.snapshot['finish_clean_elapsed_norm'] == pytest.approx(.5)
        obs['wheels'][WHEEL_ORDER[0]]['velocity_rad_s'] = .25001
        obs = steps(ev, obs, 1)
        assert ev.snapshot['finish_clean_elapsed_norm'] == 0.
        assert ev.snapshot['finish_clean_restarts'] == 1
        obs = stopped(obs)
        obs = steps(ev, obs, 120)
        assert ev.snapshot['finish_clean_elapsed_norm'] == pytest.approx(119/120)
        assert not ev.snapshot['success']
        obs = steps(ev, obs, 1)
        assert ev.snapshot['success'] and ev.snapshot['termination_reason'] is None
        assert ev.snapshot['finish_settle_pending'] is False
        assert ev.snapshot['finish_endpoint_missed'] is True
        assert ev.snapshot['finish_clean_elapsed_norm'] == 1.
        assert ev._completion_observation_since == first_clock
        assert ev.snapshot['finish_endpoint_miss'] == record


def test_region_support_evidence_and_real_safety_are_never_recoverable():
    for fault in ('region', 'support', 'evidence', 'collision', 'hard_limit', 'nonfinite', 'fall'):
        ev, obs = miss()
        healthy = deepcopy(obs)
        if fault == 'region':
            obs['body_bounds_w_m']['base_link']['minimum_m'][0] = .4
        elif fault == 'support':
            for leg in ('FR', 'FL', 'RL'):
                set_leg(obs, leg, surface='AIR')
        elif fault == 'evidence':
            obs['contacts']['rear_right_wheel']['obstacle']['pair_verified'] = False
        elif fault == 'collision':
            obs['body_collision']['detected'] = True
        elif fault == 'hard_limit':
            obs['joints']['front_left_knee']['position_deg'] = -61.
        elif fault == 'nonfinite':
            obs['all_finite'] = False
        else:
            obs['base']['position_w_m'][2] = .014
        obs = steps(ev, obs, 1)
        first_failure = ev.snapshot['termination_reason']
        assert first_failure is not None and not ev.snapshot['success']
        assert not ev.snapshot['finish_settle_pending']
        # Repairing the synthetic sensor cannot erase a submitted failure.
        healthy.update(physics_tick=obs['physics_tick'], simulation_time_s=obs['simulation_time_s'])
        steps(ev, stopped(healthy), 1)
        assert ev.snapshot['termination_reason'] == first_failure
        assert not ev.snapshot['success']
    # Region loss before the fixed endpoint remains latched, even on return.
    ev, obs = ready(True)
    obs['body_bounds_w_m']['base_link']['minimum_m'][0] = .4
    obs = steps(ev, obs, 1)
    obs['body_bounds_w_m']['base_link']['minimum_m'][0] = .7
    steps(ev, obs, 119)
    assert ev.snapshot['post_completion_loss_observed']
    assert ev.snapshot['termination_reason'] == 'INCOMPLETE_CONTROLLER_BLOCKED'
    assert not ev.snapshot['finish_endpoint_missed']


def test_existing_stop_owner_can_start_home_late_once_without_reacquisition():
    contract = load_motion_contract(ROOT/'configs/recording_motion_contract.json')
    nominal = provider(contract)
    nominal._final_stop_mode = 'source_home_after_physical_stop_v2'
    task, raw = stop_task(contract, 100)
    task['physical_evaluator'].update(final_controlled=False, task_completed_controlled=False)
    out = nominal.evaluate(task, raw)
    assert out[8:] == (0.,)*4 and nominal._final_home_recovery is None
    entry = deepcopy(nominal._final_stop_owner)
    task, raw = stop_task(contract, 260)
    task['physical_evaluator'].update(post_completion_elapsed_s=1.,
        post_completion_observation_complete=True, finish_settle_pending=True,
        finish_endpoint_missed=True,
        finish_recovery_semantics='speed_only_pending_then_continuous_original_window_v1')
    assert nominal.evaluate(task, raw)[8:] == (0.,)*4
    home = deepcopy(nominal._final_home_recovery)
    assert home is not None and home['entry_observation_tick'] == 260
    assert nominal._final_stop_owner == entry
    assert nominal.nominal_suggestion_diagnostics['final_stop_owner']['pending_home_eligibility']
    assert not nominal.nominal_suggestion_diagnostics['final_stop_owner']['current_post_window_takeover_eligibility']
    fresh = provider(contract)
    fresh._final_stop_mode = 'source_home_after_physical_stop_v2'
    fresh.evaluate(task, raw)
    assert fresh._final_stop_owner is None  # Pending cannot acquire a new owner.
    for tick in (261, 320, 321):
        task, raw = stop_task(contract, tick)
        task['physical_evaluator'].update(post_completion_elapsed_s=1.,
            post_completion_observation_complete=True, finish_settle_pending=True,
            finish_endpoint_missed=True,
            finish_recovery_semantics='speed_only_pending_then_continuous_original_window_v1')
        assert nominal.evaluate(task, raw)[8:] == (0.,)*4
        assert nominal._final_home_recovery == home
    diagnostic = nominal.nominal_suggestion_diagnostics['final_stop_owner']
    assert diagnostic['home_recovery']['nominal_ramp_complete']
    assert not diagnostic['home_recovery']['residual_channels_restricted']
    assert not diagnostic['raw_residual_mapper_or_evaluator_reset']


def test_global200_and_explicit_factory_optin_are_not_bypassed():
    ev, obs = miss()
    snap = ev.snapshot
    sup = TaskStageSupervisor(SPEC, evaluator=SimpleNamespace(
        observe=lambda _: deepcopy(snap), snapshot=snap), initial_stage_id='P13')
    sup.stage_started_s = 135.
    sup.episode_started_s = 0.
    obs.update(physics_tick=24000, simulation_time_s=200.)
    assert sup.observe_and_update(obs)['termination_source'] == 'GLOBAL_FINITE_TASK_DEADLINE'
    # Direct evaluator recovery cannot award a clean-window success at200.
    ev._finish_clean_since = 198.
    ev._snapshot['physics_tick'] = 24000
    ev._all_stage_finish(now=200., current=snap['current_legs'], body_bounds=((.7,-.05,.07),(.9,.05,.17)),
        front=.5, back=2., left=.8, right=-.8, base_linear=(0.,)*3,
        base_angular=(0.,)*3, speeds=(0.,)*4, commands=(0.,)*4, old_controlled=True)
    assert ev.snapshot['termination_source'] == 'GLOBAL_FINITE_TASK_DEADLINE'
    assert not ev.snapshot['success']
    from wlr50_clean.ppo.semantic_post_rr_front_prep_source import controller_factory
    with pytest.raises(ValueError, match='explicit bool'):
        controller_factory(task_spec_path=SPEC, read_local_state=lambda: {},
            read_post_rr_state=lambda: {}, finish_recovery_enabled='yes')

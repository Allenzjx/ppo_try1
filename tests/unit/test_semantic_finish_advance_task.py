"""Bounded synthetic wiring/physical counterexamples; not Isaac success."""
from copy import deepcopy

import pytest

from test_semantic_all_stage_physical_acceptance import new_spec, new_observation, early_top_place, set_leg
from test_semantic_supervisor import advance
from test_semantic_post_rr_front_prep_task import frame, CAPS
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator, LEG_ORDER
from wlr50_clean.ppo.semantic_finish_advance_task import (
    ADVANCE, HOME, SETTLE, DONE, FINISH_FIELDS, FINISH_OBSERVATION_DIM,
    FinishAdvanceConfig, FinishAdvanceTask, home_duration,
)
from wlr50_clean.ppo.semantic_post_rr_front_prep_task import PostRRFrontPrepTask


def bounds(obs, x=.8):
    obs['body_bounds_w_m'] = {name: dict(minimum_m=[x-.10,-.05,.06], maximum_m=[x+.10,.05,.18])
        for name in ('base_link','front_left_wheel','front_right_wheel','rear_left_wheel','rear_right_wheel')}
    obs['base']['position_w_m'] = [x, 0., .12]


def ready():
    ev = TaskEvaluator(spec=new_spec(), finish_recovery_enabled=True, finish_advance_config=FinishAdvanceConfig())
    obs = new_observation()
    bounds(obs)
    ev.observe(obs)
    for leg in ('FR','FL','RR','RL'):
        obs = early_top_place(ev, obs, leg)
    for index, leg in enumerate(LEG_ORDER):
        set_leg(obs, leg, x=.7+.03*index, bottom=.05, surface='TOP')
    obs = advance(obs)
    ev.observe(obs)
    assert ev.snapshot['requested_finish']['mode'] == ADVANCE
    return ev, obs


def step(ev, obs, count=1):
    for _ in range(count):
        obs = advance(obs)
        ev.observe(obs)
    return obs


def enter_home(ev, obs):
    bounds(obs, 1.)
    obs['joints'][SERVO_ORDER[0]].update(position_deg=20., command_deg=20.)
    obs = step(ev, obs)
    assert ev.snapshot['requested_finish']['mode'] == HOME
    return obs


def stopped(obs, error=8.):
    for name, joint in obs['joints'].items():
        joint.update(position_deg=error if name == SERVO_ORDER[0] else 0., command_deg=0.)
    obs['base']['linear_velocity_w_m_s'] = [.05,0.,0.]
    obs['base']['angular_velocity_w_rad_s'] = [0.,.3,0.]
    for wheel in obs['wheels'].values():
        wheel.update(velocity_rad_s=.25, command_rad_s=.02)
    return obs


def test_default_off_prefix_exact_and_positive_distance_not_wheel_integral():
    old = TaskEvaluator(spec=new_spec(), finish_recovery_enabled=True)
    new = TaskEvaluator(spec=new_spec(), finish_recovery_enabled=True, finish_advance_config={})
    obs = new_observation(); bounds(obs)
    for _ in range(3):
        a, b = old.observe(obs), new.observe(obs)
        b.pop('requested_finish'); b['evaluation_timing_version'] = a['evaluation_timing_version']
        assert a == b
        obs = advance(obs)
    ev, obs = ready()
    origin = ev.snapshot['requested_finish']['origin_body_position_w_m']
    for wheel in obs['wheels'].values(): wheel.update(velocity_rad_s=.3, command_rad_s=.3)
    obs = step(ev, obs, 180)
    s = ev.snapshot
    assert not s['success'] and s['termination_reason'] is None
    assert s['requested_finish']['mode'] == ADVANCE
    assert s['requested_finish']['advance_progress_m'] == 0.
    assert not s['post_completion_observation_started']
    # A brief AIR is no support force and cannot reset the displacement origin.
    set_leg(obs, 'RL', x=.79, bottom=.057, surface='AIR')
    bounds(obs, .99); obs = step(ev, obs)
    assert ev.snapshot['requested_finish']['origin_body_position_w_m'] == origin
    assert ev.snapshot['requested_finish']['advance_progress_m'] == pytest.approx(.19)
    assert ev.snapshot['requested_finish']['current_top_support_count'] == 3
    assert ev.snapshot['requested_finish']['mode'] == ADVANCE
    bounds(obs, .90); obs = step(ev, obs)
    assert ev.snapshot['requested_finish']['advance_progress_m'] == pytest.approx(.10)
    assert not ev.snapshot['requested_finish']['post_RL_forward_completed']
    with pytest.raises(ValueError): FinishAdvanceConfig(advance_distance_m=0.)


def test_geometry_and_actual_home_are_separate_nonterminal_gates():
    ev, obs = ready()
    bounds(obs, 1.)
    obs['body_bounds_w_m']['rear_left_wheel']['minimum_m'][0] = .54
    obs = step(ev, obs)
    assert ev.snapshot['requested_finish']['mode'] == ADVANCE
    bounds(obs, 1.)
    obs['body_bounds_w_m']['front_right_wheel']['maximum_m'][0] = 1.95
    obs = step(ev, obs)
    assert ev.snapshot['requested_finish']['mode'] == ADVANCE
    obs = enter_home(ev, obs)
    duration = ev.snapshot['requested_finish']['home_duration_s']
    assert duration == home_duration(FinishAdvanceConfig(), [20.,0.,3.,0.,3.,0.,3.,0.])
    obs = step(ev, obs, 130)
    assert ev.snapshot['termination_reason'] is None and not ev.snapshot['success']
    assert not ev.snapshot['requested_finish']['current_home_within_tolerance']
    assert ev.snapshot['requested_finish']['mode'] == HOME
    assert ev.snapshot['home_maximum_servo_error_deg'] == 20.


def test_original_thresholds_home8_clean1s_and_no_phase_done():
    ev, obs = ready(); obs = enter_home(ev, obs)
    obs = stopped(obs, error=8.00001); obs = step(ev, obs, 140)
    assert not ev.snapshot['success'] and ev.snapshot['requested_finish']['clean_elapsed_s'] == 0.
    obs = stopped(obs); obs = step(ev, obs, 61)
    assert ev.snapshot['requested_finish']['mode'] == SETTLE
    assert ev.snapshot['requested_finish']['clean_elapsed_s'] == pytest.approx(.5)
    for fault in ('wheel','linear','angular','command'):
        bad = deepcopy(obs)
        if fault == 'wheel': bad['wheels'][WHEEL_ORDER[0]]['velocity_rad_s'] = .250001
        elif fault == 'linear': bad['base']['linear_velocity_w_m_s'][0] = .050001
        elif fault == 'angular': bad['base']['angular_velocity_w_rad_s'][1] = .300001
        else: bad['wheels'][WHEEL_ORDER[0]]['command_rad_s'] = .020001
        obs = step(ev, bad)
        assert ev.snapshot['requested_finish']['clean_elapsed_s'] == 0.
        assert ev.snapshot['termination_reason'] is None
        obs = stopped(obs); obs = step(ev, obs, 20)
    obs = stopped(obs); obs = step(ev, obs, 121)
    assert ev.snapshot['success'] and ev.snapshot['requested_finish']['mode'] == DONE
    assert ev.snapshot['strict_recovery_quality']['passed']
    assert ev.snapshot['requested_finish_completed']


def test_real_safety_global200_and_latched_region_failure_survive():
    for fault in ('collision','hard_limit','nonfinite','fall','region'):
        ev, obs = ready(); obs = enter_home(ev, obs)
        obs = stopped(obs); obs = step(ev, obs, 3)
        healthy = deepcopy(obs)
        if fault == 'collision': obs['body_collision']['detected'] = True
        elif fault == 'hard_limit': obs['joints']['front_left_knee']['position_deg'] = -61.
        elif fault == 'nonfinite': obs['all_finite'] = False
        elif fault == 'fall': obs['base']['position_w_m'][2] = .014
        else: obs['body_bounds_w_m']['base_link']['minimum_m'][0] = .4
        obs = step(ev, obs)
        first = ev.snapshot['termination_reason']
        assert first and not ev.snapshot['success']
        healthy.update(physics_tick=obs['physics_tick'], simulation_time_s=obs['simulation_time_s'])
        step(ev, healthy)
        assert ev.snapshot['termination_reason'] == first and not ev.snapshot['success']
    ev, obs = ready(); obs = enter_home(ev, obs)
    # Direct task clock uses the same finite horizon; not a local timeout reset.
    obs.update(physics_tick=23999, simulation_time_s=200.-1/120.)
    ev._last_tick, ev._last_time = 23999, 200.-1/120.
    step(ev, obs)
    assert ev.snapshot['termination_source'] == 'GLOBAL_FINITE_TASK_DEADLINE'


def test_wrapper_delegates_prefix_and_exposes_exact_ack_references_and_rebase():
    task = FinishAdvanceTask(accepted_task=PostRRFrontPrepTask(phase_caps_full12={'P11':CAPS,'P12':CAPS}),
                            phase_caps_full12={'P11':CAPS,'P12':CAPS})
    ev, obs = ready()
    state = deepcopy(ev.snapshot['requested_finish'])
    s0 = deepcopy(state); s0.update(active=False, mode=None, observation_tick=0)
    f = frame(0); f.info['semantic_task']['physical_evaluator']['requested_finish'] = s0
    task.observe(f)
    assert task.finish_observation() == (0.,)*41
    assert task.post_rr_observation() == task.accepted.post_rr_observation()
    state['observation_tick'] = 1
    f = frame(1, top=True, bearing=True, rl_top=True)
    f.info['semantic_task']['physical_evaluator']['requested_finish'] = state
    before = task.snapshot(); task.observe(f)
    assert not task.reward(before)['on_policy_finish_sample']
    assert FINISH_FIELDS[0] == 'finish_active' and FINISH_OBSERVATION_DIM == 41
    assert task.finish_observation()[1:13] == tuple(a/b for a,b in zip(task.reference_request,CAPS))
    assert task.snapshot()['finish_context']['reference_final_full12'] == tuple(f.info['drive_target_full12'])
    f2 = frame(2, top=True, bearing=True, rl_top=True)
    h = deepcopy(state); h.update(observation_tick=2, mode=HOME, home_permitted=True,
        home_entry_tick=2, home_duration_s=home_duration(task.finish_config, f2.info['drive_target_full12'][:8]))
    f2.info['semantic_task']['physical_evaluator']['requested_finish'] = h
    task.observe(f2)
    assert len(task.reference_history) == 2 and task.reference_tick == 2
    assert not task.snapshot()['finish_context']['config'].get('HISTORY_reset',False)
    damaged = frame(3, top=True, bearing=True, rl_top=True)
    damaged.info['atomic_ack']['physics_tick'] += 1
    damaged.info['semantic_task']['physical_evaluator']['requested_finish'] = dict(h, observation_tick=3)
    # A reset reference must be an actual adjacent ACK, never an independently reconstructed N.
    task._reference_mode = ADVANCE
    with pytest.raises(ValueError, match='clocks differ'): task.observe(damaged)

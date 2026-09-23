"""Staged finite P05 endpoint recovery; synthetic evidence, never Isaac success.

Not run while the source recording is active. Normal imports require v3.
"""
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path

import pytest

from test_semantic_p05_preedge_recovery import KEY, ROLL, clone, mature, status, task
from wlr50_clean.ppo.semantic_supervisor import (
    P05_COMPLETED_SOURCE_RECOVERY_MODE, P05_SAME_AIR_RECROSS_MODE,
    P05_PREEDGE_RECOVERY_MODE, P05_FINITE_RECOVERY_TIMEOUT_MODE,
    _p05_preedge_recovery_enabled, load_task_spec,
)

TIMEOUT_KEY = 'p05_finite_recovery_timeout_semantics'


@pytest.fixture(scope='module')
def mature_v3(mature):
    provider = clone(mature[0])
    provider.spec['nominal'][KEY] = P05_COMPLETED_SOURCE_RECOVERY_MODE
    provider.spec[TIMEOUT_KEY] = P05_FINITE_RECOVERY_TIMEOUT_MODE
    assert _p05_preedge_recovery_enabled(provider.spec)
    return provider


def same_air(tick=3600, age=9.8):
    value = task(tick, age)
    ev = value['physical_evaluator']
    ev['history']['front_edge_crossed']['FL'] = True
    ev['history']['event_ticks'] = {'front_edge_crossed': {'FL': tick-60}}
    ev['current_legs']['FL'].update(active_attempt=True, consecutive_air_samples=61)
    return value


def test_experiment_selects_explicit_new_mode_with_existing_finite_timeout():
    root = Path(__file__).resolve().parents[2]
    spec = load_task_spec(root/'configs/ppo_rr_capture_then_rl_transfer_v1/stage_task_spec.yaml')
    assert spec['nominal'][KEY] == P05_COMPLETED_SOURCE_RECOVERY_MODE
    assert spec[TIMEOUT_KEY] == P05_FINITE_RECOVERY_TIMEOUT_MODE
    assert spec['stages']['P05']['maximum_task_duration'] == 30.
    assert spec['local_timeout_policy']['maximum_extension_s'] == 10.
    spec.pop(TIMEOUT_KEY)
    with pytest.raises(ValueError, match='explicitly paired'):
        _p05_preedge_recovery_enabled(spec)


@pytest.mark.parametrize('mode', [P05_PREEDGE_RECOVERY_MODE, P05_SAME_AIR_RECROSS_MODE])
def test_old_opt_ins_keep_original_lower_age(mature_v3, mode):
    old = clone(mature_v3); old.spec['nominal'][KEY] = mode
    if mode == P05_PREEDGE_RECOVERY_MODE: old.spec.pop(TIMEOUT_KEY)
    assert _p05_preedge_recovery_enabled(old.spec)
    assert not status(old, task(age=9.8))['eligible']
    assert status(old, task(age=30.))['eligible']
    assert not status(old, task(age=40.))['eligible']


@pytest.mark.parametrize('age,eligible', [(-.001, False), (0., True), (9.8, True),
    (29.999, True), (30., True), (39.999, True), (40., False), (41., False),
    (math.nan, False), (None, False), (True, False)])
def test_only_lower_age_changes_and_absolute_end_never_renews(mature_v3, age, eligible):
    info = status(mature_v3, task(age=age))
    assert info['eligible'] is eligible
    assert info['absolute_window_s'] == (0., 40.)


def test_measured_positive_gap_window_is_not_forced_to_wait_until_age30(mature_v3):
    # Measured CP222720v10 geometry at31.3333s; other data is this test's
    # synthetic valid source/context fixture, not a whole physical replay.
    value = task(age=9.8)
    value['physical_evaluator']['current_legs']['FL'].update(
        front_distance_m=-.02416935222316141, clearance_m=.008952820481185078)
    info = status(mature_v3, value)
    assert info['eligible'] and info['reasons'] == ()
    assert info['same_air_recross']['branch'] == 'first_approach'
    assert not info['phase_or_capture_credit_awarded']
    assert not info['policy_residual_restricted']
    assert not info['independent_ack_verified']


def test_full_source_endpoint_and_subsequent_tick_remain_mandatory(mature_v3):
    value = task(age=9.8)
    assert not status(mature_v3, value, previous_endpoint=False)['eligible']
    before_endpoint = clone(mature_v3); before_endpoint.endpoint_issued = False
    assert not status(before_endpoint, value)['eligible']
    for prior in (None, 3598, 3600, 3601, True):
        assert not status(mature_v3, value, previous_tick=prior)['eligible']


def test_explicit_stop_owns_its_tick_then_later_valid_tick_can_advise(mature, mature_v3, monkeypatch):
    provider, stop = clone(mature_v3), mature[1]
    original = provider._continuous_advisory
    def with_stop(*args, **kwargs):
        proposed, tracking = original(*args, **kwargs)
        layer = provider._continuous_layers[-1]
        layer['sample'] = replace(layer['sample'], atomic_groups=stop.atomic_groups)
        return proposed[:8]+(0.,)*4, tracking
    monkeypatch.setattr(provider, '_continuous_advisory', with_stop)
    assert provider.evaluate(task(age=9.8))[8:] == (0.,)*4
    info = provider.nominal_suggestion_diagnostics[KEY]
    assert not info['eligible'] and 'fresh_source_wheel_owner_absent' in info['reasons']
    monkeypatch.setattr(provider, '_continuous_advisory', original)
    assert provider.evaluate(task(3601, 9.8+1/120.))[8:] == ROLL


@pytest.mark.parametrize('change', [dict(ground_contact=True), dict(obstacle_pair_active=True),
    dict(top_surface_contact=True), dict(air=False), dict(clearance_m=0.),
    dict(clearance_m=-.000001), dict(clearance_m=math.nan),
    dict(within_lateral_span=False), dict(front_distance_m=-.10001)])
def test_no_relaxation_for_ground_wall_nonpositive_gap_or_invalid_roi(mature_v3, change):
    value = task(age=9.8)
    value['physical_evaluator']['current_legs']['FL'].update(change)
    assert not status(mature_v3, value)['eligible']


@pytest.mark.parametrize('fault', ['support', 'bearing', 'air', 'physical_invalid', 'abort', 'FR_unplaced'])
def test_support_and_live_task_still_required(mature_v3, fault):
    value = task(age=9.8); ev = value['physical_evaluator']
    if fault in ('support', 'bearing', 'air'):
        key = {'support':'support', 'bearing':'bearing_verified', 'air':'air'}[fault]
        for leg in ('RR', 'RL'): ev['current_legs'][leg][key] = fault == 'air'
    elif fault == 'physical_invalid': ev['valid'] = False
    elif fault == 'abort': value['termination_reason'] = 'TASK_FAILURE_BODY_COLLISION'
    else: ev['history']['placed']['FR'] = False
    assert not status(mature_v3, value)['eligible']


def test_first_approach_and_recross_keep_different_history_requirements(mature_v3):
    first = task(age=9.8)
    assert status(mature_v3, first)['same_air_recross']['branch'] == 'first_approach'
    recross = same_air()
    assert status(mature_v3, recross)['eligible']
    assert status(mature_v3, recross)['same_air_recross']['branch'] == 'same_air_recross'
    # A post-cross ground/contact interval cannot borrow old history.
    recross['physical_evaluator']['current_legs']['FL']['consecutive_air_samples'] = 60
    assert not status(mature_v3, recross)['eligible']
    missing = same_air(); missing['physical_evaluator']['history'].pop('event_ticks')
    assert not status(mature_v3, missing)['eligible']
    placed = task(age=9.8); placed['physical_evaluator']['history']['placed']['FL'] = True
    assert not status(mature_v3, placed)['eligible']


def test_intercept_changes_wheel_advice_only_not_source_servo_owners_or_history(mature_v3):
    enabled, disabled = clone(mature_v3), clone(mature_v3)
    disabled._p05_preedge_recovery = False
    value = task(age=9.8); before = deepcopy(value)
    expected, actual = disabled.evaluate(deepcopy(value)), enabled.evaluate(value)
    assert actual[:8] == expected[:8] and actual[8:] == ROLL and expected[8:] == (0.,)*4
    assert value == before and enabled.state_id == 'P05'
    assert enabled.tracking_servo_names == disabled.tracking_servo_names
    assert enabled.normal_drive_bias_full12 == disabled.normal_drive_bias_full12
    assert enabled._continuous_layers[-1]['sample'] == disabled._continuous_layers[-1]['sample']
    assert enabled._source_motion._tick_index == disabled._source_motion._tick_index


def test_permission_loss_and_original_absolute_end_return_original_source_tail(mature_v3):
    provider = clone(mature_v3)
    assert provider.evaluate(task(age=9.8))[8:] == ROLL
    lost = task(3601, 9.8+1/120.)
    lost['physical_evaluator']['current_legs']['FL']['clearance_m'] = 0.
    assert provider.evaluate(lost)[8:] == (0.,)*4
    assert provider.evaluate(task(3602, 9.8+2/120.))[8:] == ROLL
    assert provider.evaluate(task(3603, 40.))[8:] == (0.,)*4
    assert provider.evaluate(task(3604, 40.+1/120.))[8:] == (0.,)*4

"""Bounded synthetic timeout counterexamples, NOT Isaac/task success evidence.

Uses contiguous measured-fixture observations and the production assist/evaluator/
supervisor. Only deadline clock origins are positioned as in the existing unit
fixture; no production code, thresholds, predicate results or placement history
are patched. The feedback envelope is a synthetic API fixture, not a native ACK.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/unit'))
from test_semantic_rr_capture_supervisor import airborne_rr
from test_semantic_all_stage_physical_acceptance import set_leg
from test_semantic_supervisor import advance
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.semantic_rr_capture_assist import (
    RRHipOnlyCaptureAssist, rr_capture_assist_context, apply_rr_capture_assist_snapshot,
)


class Trace:
    def __init__(self, pair):
        self.sup, self.obs = pair
        self.assist = RRHipOnlyCaptureAssist()
        self.previous = tuple(self.obs['joints'][name]['position_deg'] for name in SERVO_ORDER) + (0.,) * 4
        self.rows = []
        assert self.sup.spec['history']['minimum_top_samples'] == 2

    def step(self, surface='AIR', *, local_age=20., episode_age=100., weak=False):
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
            body = obs['wheels'][WHEEL_ORDER[3]]['body_name']  # Canonical order FL/FR/RL/RR.
            pair = obs['contacts'][body]['obstacle']
            force = sup.spec['support']['force_noise_floor_n'] / 2.
            pair.update(normal_force_n=force, force_w_n=[0., 0., force])
        now = obs['simulation_time_s']
        if sup.stage_id == 'P09':
            sup.stage_started_s = now - local_age
        sup.episode_started_s = now - episode_age
        sup.rr_capture_feedback = {'episode_observation_tick': obs['physics_tick'], 'state': receipt['state_after']}
        task = sup.observe_and_update(obs)
        self.obs = obs
        ev, rr = task['physical_evaluator'], task['physical_evaluator']['current_legs']['RR']
        recovery = task['rr_capture_continuation']
        row = {'tick': obs['physics_tick'], 'tick_mod8': obs['physics_tick'] % 8,
            'stage': task['stage_id'], 'assist_mode': receipt['state_after']['mode_name'],
            'assist_hold_elapsed_s': receipt['state_after']['hold_elapsed_s'],
            'pre_physics_top_surface': context['top_surface_contact'], 'pre_physics_bearing': context['current_top_bearing'],
            'current_TOP': rr['top_contact'], 'current_TOP_surface': rr['top_surface_contact'],
            'current_bearing_verified': rr['bearing_verified'], 'current_ground': rr['ground_contact'], 'current_AIR': rr['air'],
            'current_obstacle_pair': rr['obstacle_pair_active'],
            'consecutive_top_samples': rr['consecutive_top_samples'], 'placed_history': ev['history']['placed']['RR'],
            'placed_goal': sup.predicate('placed_RR', ev),
            'fresh_feedback': recovery['committed_feedback_matches_current_tick'],
            'recovery_allowed': recovery['rr_capture_recovery_allowed'], 'warning_only': recovery['local_warning_only'],
            'termination': task['termination_reason'], 'termination_source': task['termination_source']}
        self.rows.append(row)
        return row

    def align_second_top_mod8(self, value):
        count = (value - (self.obs['physics_tick'] + 2)) % 8
        for _ in range(count):
            assert self.step()['termination'] is None

    def evidence(self, case):
        rows = [row for row in self.rows if row['current_obstacle_pair'] or row is self.rows[-1]]
        print('SYNTHETIC_EVIDENCE ' + json.dumps({'case': case, 'rows': rows}, sort_keys=True))


@pytest.mark.parametrize('placed_mod8', [1, 3, 7])
def test_after_deadline_fresh_hold_terminates_at_real_placement_before_next_decision(airborne_rr, placed_mod8):
    trace = Trace(airborne_rr); trace.align_second_top_mod8(placed_mod8)
    first = trace.step('TOP', local_age=45.)
    assert first['assist_mode'] == 'DESCEND' and first['consecutive_top_samples'] == 1
    assert not first['placed_history'] and first['warning_only'] and first['termination'] is None
    second = trace.step('TOP', local_age=45.)
    assert second['tick_mod8'] == placed_mod8 and second['assist_mode'] == 'HOLD'
    assert second['current_TOP'] and second['consecutive_top_samples'] == 2 and second['placed_history']
    assert second['placed_goal'] == 1. and second['fresh_feedback'] and not second['warning_only']
    assert second['stage'] == 'P09' and second['termination'] == 'INCOMPLETE_CONTROLLER_BLOCKED'
    assert second['termination_source'] == 'LOCAL_BOUNDED_RECOVERY_EXHAUSTED'
    assert second['assist_hold_elapsed_s'] == pytest.approx(1. / 120.)
    # A real run stops here; do not fabricate post-terminal motion to P10.
    trace.evidence(f'placement_mod8_{placed_mod8}_deadline_expired')


def test_before_deadline_current_contact_survives_to_next_decision_handoff(airborne_rr):
    trace = Trace(airborne_rr); trace.align_second_top_mod8(1)
    first, second = trace.step('TOP'), trace.step('TOP')
    assert first['consecutive_top_samples'] == 1 and not first['placed_history']
    assert second['placed_history'] and second['assist_mode'] == 'HOLD' and second['termination'] is None
    for _ in range(7):
        last = trace.step('TOP')
        assert last['termination'] is None
    assert last['tick_mod8'] == 0 and last['stage'] == 'P10'
    trace.evidence('before_deadline_natural_next_decision_handoff')


def test_capture_exactly_at_decision_tick_hands_off_before_local_timeout(airborne_rr):
    trace = Trace(airborne_rr); trace.align_second_top_mod8(0)
    assert trace.step('TOP', local_age=45.)['termination'] is None
    last = trace.step('TOP', local_age=45.)
    assert last['placed_history'] and last['assist_mode'] == 'HOLD'
    assert last['tick_mod8'] == 0 and last['stage'] == 'P10' and last['termination'] is None
    trace.evidence('expired_P09_capture_exact_decision_tick')


def test_global_deadline_still_terminates_even_with_capture_at_decision_tick(airborne_rr):
    trace = Trace(airborne_rr); trace.align_second_top_mod8(0)
    assert trace.step('TOP')['termination'] is None
    last = trace.step('TOP', local_age=45., episode_age=200.)
    assert last['placed_history'] and last['termination'] == 'INCOMPLETE_CONTROLLER_BLOCKED'
    assert last['termination_source'] == 'GLOBAL_FINITE_TASK_DEADLINE'
    trace.evidence('global200_wins_over_true_capture_handoff')


def test_ground_recontact_does_not_receive_a_contact_hold_waiver(airborne_rr):
    trace = Trace(airborne_rr); trace.align_second_top_mod8(1)
    assert trace.step('TOP')['termination'] is None
    last = trace.step('GROUND', local_age=45.)
    assert last['current_ground'] and not last['current_TOP'] and not last['placed_history']
    assert not last['recovery_allowed'] and not last['warning_only']
    assert last['termination'] == 'INCOMPLETE_CONTROLLER_BLOCKED'
    trace.evidence('ground_not_capture_hold')


def test_historical_placement_followed_by_air_is_not_current_contact_hold(airborne_rr):
    trace = Trace(airborne_rr); trace.align_second_top_mod8(1)
    assert trace.step('TOP')['termination'] is None
    assert trace.step('TOP')['placed_history']
    last = trace.step('AIR', local_age=45.)
    assert last['placed_history'] and last['current_AIR'] and not last['current_TOP']
    assert last['assist_mode'] == 'HOLD' and last['fresh_feedback']
    assert not last['warning_only'] and last['termination'] == 'INCOMPLETE_CONTROLLER_BLOCKED'
    trace.evidence('old_placed_AIR_is_not_current_TOP_hold')


def test_weak_unclassified_pair_then_first_loaded_top_can_timeout_before_placement(airborne_rr):
    trace = Trace(airborne_rr); trace.align_second_top_mod8(3)
    weak = trace.step('TOP', weak=True)
    assert weak['current_obstacle_pair'] and not weak['current_TOP_surface']
    assert not weak['current_TOP'] and not weak['placed_history']
    last = trace.step('TOP', local_age=45.)
    assert last['assist_mode'] == 'HOLD' and last['current_TOP'] and last['consecutive_top_samples'] == 1
    assert not last['placed_history'] and not last['warning_only']
    assert last['termination'] == 'INCOMPLETE_CONTROLLER_BLOCKED'
    trace.evidence('first_loaded_TOP_after_weak_pair_before_placement')

"""Offline sizing on sealed physical states, never counterfactual task credit."""
from __future__ import annotations
import json
import math
from pathlib import Path

from event_quality import rows, _multiply, _quaternion, _rpy, load_semantic_observation_schema, ROOT


def analyze(summary_path):
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    source = Path(summary['source'])
    run = json.loads((source.parent / 'run_manifest.json').read_text(encoding='utf-8'))
    assert run.get('completed_at_utc') and run['lifecycle'] != 'RUNNING'
    end = summary['events_first_observed']['placed']['FR']
    schema = load_semantic_observation_schema(ROOT / 'configs' / ('ppo_'+summary['experiment_id']) / 'observation_schema.json')
    decisions = []
    transitions = []
    for row in rows(source / 'video_policy_decisions.jsonl'):
        if row['start_tick'] >= end:
            break
        decisions.append(row)
        evidence = row['step_info'].get('stage_transition_evidence') or []
        if isinstance(evidence, dict):
            evidence = [evidence]
        for transition in evidence:
            if transition.get('to_stage'):
                transitions.append((transition['physics_tick'], transition['to_stage']))
    transitions.sort()
    phase = 'P01'
    transition_idx = decision_idx = 0
    previous = None
    raw_integral = tilt_integral = rate_integral = eligible_dt = 0.
    discounted_integral = endpoint_beta_estimate = 0.
    states = 0
    for row in rows(source / 'physical_observations.jsonl'):
        tick = row['physics_tick']
        if tick > end:
            break
        q = _quaternion(row['base']['orientation_wxyz'])
        rpy = _rpy(_quaternion(_multiply(q, schema.fixed_chassis_to_body_wxyz)))
        while transition_idx < len(transitions) and transitions[transition_idx][0] <= tick:
            phase = transitions[transition_idx][1]
            transition_idx += 1
        if previous is not None:
            assert tick == previous[0] + 1
            dt = row['simulation_time_s'] - previous[1]
            rates = [math.atan2(math.sin(a-b), math.cos(a-b))/dt for a,b in zip(rpy[:2], previous[2][:2])]
            if phase in ('P01','P02'):
                while decisions[decision_idx]['end_tick'] < tick:
                    decision_idx += 1
                # Raw 120 Hz geometry is exact; old logs only retain the
                # transfer fraction at policy endpoints, so report bounds.
                tilt = sum(min(1., (x/.5)**2) for x in rpy[:2])/2
                rate = sum(min(1., (x/.5)**2) for x in rates)/2
                raw = .5*(tilt+rate)
                fraction = decisions[decision_idx]['step_info']['semantic_task']['physical_transfer_fraction']
                raw_integral += raw*dt
                tilt_integral += .5*tilt*dt
                rate_integral += .5*rate*dt
                discounted_integral += (.9985**decision_idx)*raw*dt
                endpoint_beta_estimate += .03*(1.-.5*fraction)*raw*dt
                eligible_dt += dt
                states += 1
        previous = tick, row['simulation_time_s'], rpy
    task_rows = [row for row in decisions if row['request_phase'] in ('P01','P02')]
    task = {'positive_potential_sum': 0., 'negative_potential_sum': 0.,
            'time_cost_sum': 0., 'terminal_event_sum': 0., 'net_task_reward_sum': 0.}
    for row in task_rows:
        reward = row['step_info']['reward']
        value = reward['potential_shaping']
        task['positive_potential_sum' if value > 0 else 'negative_potential_sum'] += value
        task['time_cost_sum'] += .02*reward['elapsed_physics_s']
        task['terminal_event_sum'] += reward['terminal_event']
        task['net_task_reward_sum'] += reward['families']['task_progress']
    return {
        'sealed_source': str(source), 'FR_event_window': summary['windows']['FR_preparation_to_capture'],
        'eligible_current_phase_samples': states, 'eligible_physics_duration_s': eligible_dt,
        'P01_P02_raw_integral_s': raw_integral,
        'half_tilt_integral_s': tilt_integral, 'half_rate_integral_s': rate_integral,
        'prospective_new_quality_cost_bounds': [.015*raw_integral, .03*raw_integral],
        'prospective_discounted_cost_bounds': [.015*discounted_integral, .03*discounted_integral],
        'endpoint_transfer_fraction_proxy_cost_not_exact': endpoint_beta_estimate,
        'original_actual_P01_P02_request_decision_reward': task,
        'quality_upper_bound_over_recorded_positive_potential_sum': .03*raw_integral/task['positive_potential_sum'],
        'old_actual_quality_weight_was_zero': True,
        'new_reward_was_not_used_by_this_policy': True,
        'does_not_establish_new_policy_return_or_learning_improvement': True,
        'bound_reason': 'Exact 120 Hz pose/rate integral; transfer fraction only logged at 15 Hz endpoints. New beta is bounded .015..03 independently of substate. P01/P02 task totals use whole requested decision intervals and may include the final transition sample.'}


if __name__ == '__main__':
    folder = Path(__file__).resolve().parent
    result = {'schema': 'wlr50_clean.front_quality_initial_budget.v1',
              'beta_max_per_s': .03, 'transfer_floor': .5,
              'attitude_scale_rad': .5, 'euler_rate_scale_rad_s': .5,
              'tilt_fraction': .5, 'rate_fraction': .5,
              'no_new_physical_trajectory_generated': True,
              'references_are_descriptive_different_RR_versions': True,
              'trajectories': {name: analyze(folder / (name+'_event_quality.json')) for name in ('N_ref','CP178432')}}
    out = folder / 'front_quality_initial_budget.json'
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))

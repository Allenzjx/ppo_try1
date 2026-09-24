"""Sealed P10 only. Endpoint counterfactual screen, NOT a 120-Hz reward replay.

Stdlib only; emits a compact JSON result, writes no checkpoint or runtime file.
"""
import collections
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0130272021180Z_g49eb23163a6e_5e6b94b24cce4a2fb9edc117565be855'


def numeric(d, key):
    x = d[key]
    assert type(x) in (float, int) and math.isfinite(x), key
    return x


def scalar_config(path, key):
    text = path.read_text(encoding='utf-8')
    values = re.findall(r'^\s*' + re.escape(key) + r':\s*([0-9.]+)\s*$', text, re.M)
    assert len(values) == 1, (key, values)
    return float(values[0])


def main():
    manifest = json.loads((RUN / 'training_manifest.json').read_text())
    assert manifest['lifecycle'] == 'SUCCEEDED' and manifest['actual_policy_decisions'] == 384
    config = ROOT / 'configs/ppo_rr_rl_timing_policy_learning_v1'
    floor = scalar_config(config / 'reward_config.yaml', 'force_noise_floor_n')
    velocity_scale = scalar_config(config / 'stage_task_spec.yaml', 'velocity_scale_m_s')
    beta = scalar_config(config / 'reward_config.yaml', 'counterroll_cost_per_s')
    time_cost = scalar_config(config / 'reward_config.yaml', 'time_cost_per_s')
    counts = collections.Counter()
    failures = collections.Counter()
    cases = {}
    candidates = []
    previous = None
    actual_cost = 0.
    cap_values = set()
    with (RUN / 'residual_and_projection_audit.jsonl').open(encoding='utf-8') as stream:
        for i, line in enumerate(stream):
            r = json.loads(line)
            assert r['global_policy_decision'] == 223617 + i
            a = r['applied_audit']; t = a['semantic_task']; ev = t['physical_evaluator']
            rr, fl = ev['current_legs']['RR'], ev['current_legs']['FL']
            diag = t['cooperative_preparation']
            assert a['physics_tick'] == 6144 + i * 8 and not r['terminal']
            n = numeric({'n': a['nominal_action_full12'][8]}, 'n')
            target = numeric({'n': a['actual_drive_target_full12'][8]}, 'n')
            actual = numeric({'n': ev['measured_wheel_velocity_rad_s'][0]}, 'n')
            request = a['actuator_target_effect_audit']['policy_headroom_evidence']['requested_policy_residual_full12'][8]
            cap = r['policy_request']['current_cap_full12'][8]
            assert cap > 0.; cap_values.add(cap)
            source_class = 'zero' if n == 0. else ('negative' if n < 0. else 'positive')
            counts['N_' + source_class] += 1
            counts['all_endpoints'] += 1
            reward_rows = a['reward_breakdown']['cooperative_preparation_sample_audit']
            assert len(reward_rows) == 8
            for sample in reward_rows:
                counts['actual_reward_ticks'] += 1
                counts['actual_reward_eligible'] += int(sample['eligible'])
                counts['actual_reward_positive'] += int(sample['raw_cost'] > 0.)
            actual_cost += a['reward_breakdown']['cooperative_counterroll_cost']
            gates = {
                'rear_phase': t['stage_id'] not in ('P01', 'P02', 'P03', 'P04', 'P05', 'P06'),
                'preparation_relevant': diag.get('relevant') is True,
                'no_qualified_RL_swing': diag['rl_actual_swing'] is False,
                'physical_valid': ev['valid'] is True and ev['termination_reason'] is None,
                'RR_current_qualified_AIR': rr['current_lift_valid'] is True and rr['air'] is True,
                'RR_no_current_bearing': diag['current_rr_bearing'] is False,
                'FL_real_support': fl['air'] is False and fl['bearing_verified'] is True
                    and fl['support'] is True and (fl['ground_contact'] is True or fl['top_surface_contact'] is True)
                    and numeric(fl, 'bearing_force_n') >= floor,
                'no_authored_reverse': n >= 0.,
            }
            for key, ok in gates.items():
                if not ok: failures[key] += 1
            eligible = all(gates.values())
            reverse = min(max(0., -target), max(0., -actual))
            counts['static_gates_except_source'] += int(all(v for k, v in gates.items() if k != 'no_authored_reverse'))
            counts['endpoint_candidate_static'] += int(eligible)
            counts['endpoint_candidate_static_N_' + source_class] += int(eligible)
            counts['N_zero_request_target_actual_reverse'] += int(n == 0. and request < 0. and target < 0. and actual < 0.)
            common = dict(tick=a['physics_tick'], time_s=a['sim_time_s'], phase=t['stage_id'],
                nominal_FL=n, request_FL=request, final_FL=target, actual_FL=actual,
                RR_gap_mm=rr['clearance_m'] * 1000., RR_current_Q=rr['current_lift_valid'],
                RR_AIR=rr['air'], FL_AIR=fl['air'], FL_force_n=fl['bearing_force_n'],
                RL_swing=diag['rl_actual_swing'], candidate_static=eligible)
            if n < 0. and reverse > 0.:
                cases.setdefault('authored_reverse_excluded', common)
            if fl['air'] is True and n == 0. and target < 0.:
                cases.setdefault('FL_AIR_excluded', common)
            if diag['rl_actual_swing']:
                cases.setdefault('RL_swing_excluded', common)
            if eligible and reverse > 0.:
                counts['endpoint_supported_reverse_candidate'] += 1
                if previous is not None:
                    if previous['nominal_action_full12'][8] < 0.:
                        counts['proxy_intervals_excluded_authored_reverse_at_previous_endpoint'] += 1
                        previous = a
                        continue
                    old = previous['semantic_task']['physical_evaluator']['current_legs']['RR']
                    dt = a['sim_time_s'] - previous['sim_time_s']
                    assert a['physics_tick'] - previous['physics_tick'] == 8 and abs(dt - 8 / 120) < 1e-8
                    # Evaluator defines distance = wheel center x - fixed obstacle front.
                    forward = max(0., (rr['front_distance_m'] - old['front_distance_m']) / dt)
                    legal = all(x['within_top_xy'] is True and x['within_lateral_span'] is True
                                and x['ground_contact'] is False for x in (old, rr))
                    descent = (max(0., (max(0., old['clearance_m']) - max(0., rr['clearance_m'])) / dt)
                               if legal else 0.)
                    progress = max(forward, descent)
                    lack = 1. - min(1., progress / velocity_scale)
                    proxy = min(1., reverse / cap) * lack
                    row = dict(common, previous_tick=previous['physics_tick'], interval_s=dt,
                        legal_XY_at_both_endpoints=legal, forward_m_s=forward, legal_descent_m_s=descent,
                        endpoint_proxy_raw_cost=proxy)
                    candidates.append(row)
                    counts['proxy_intervals_both_source_endpoints_nonnegative'] += 1
                    counts['proxy_intervals_legal_XY_at_both_endpoints'] += int(legal)
                    category = ('no_net_progress' if progress == 0. else
                                'positive_below_existing_scale' if progress < velocity_scale else
                                'progress_retires_cost')
                    counts['candidate_' + category] += 1
                    cases.setdefault(category, row)
            previous = a
    assert counts['all_endpoints'] == 384
    print(json.dumps(dict(schema='readonly.stopped_source_counterroll_screen.v1', run=str(RUN),
        counts=dict(counts), independent_gate_failures=dict(failures),
        existing_config=dict(force_floor_n=floor, FL_cap_values=sorted(cap_values),
            progress_scale_m_s=velocity_scale, coefficient_per_s=beta, existing_time_cost_per_s=time_cost),
        actual_counterroll_cost=actual_cost, endpoint_examples=cases,
        candidate_endpoint_ticks=[r['tick'] for r in candidates],
        endpoint_proxy_raw_cost_sum=sum(r['endpoint_proxy_raw_cost'] for r in candidates),
        limitations='384 endpoint screen and 8-tick net-progress proxy only. Interior states absent after reward early-return; no exact 3072-tick counterfactual, discounted return, or optimizer effect claimed.'), indent=2))


if __name__ == '__main__':
    main()

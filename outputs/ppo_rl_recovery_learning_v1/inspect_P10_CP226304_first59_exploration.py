"""Fixed sealed first episode only; stdlib, no runtime/model imports or writes."""
from pathlib import Path
import collections
import itertools
import json
import math

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T1717242165088Z_g892385cba8a7_afdc21482bbc480a9692fb63fe8ad858'
with (RUN / 'residual_and_projection_audit.jsonl').open(encoding='utf8') as stream:
    rows = [json.loads(line) for line in itertools.islice(stream, 59)]
assert len(rows) == 59 and rows[-1]['global_policy_decision'] == 226363
assert rows[-1]['applied_audit']['termination_reason'] == 'BODY_COLLISION'
names = ['FLh','FLk','FRh','FRk','RLh','RLk','RRh','RRk','FLw','FRw','RLw','RRw']
clamp = lambda value, lo, hi: max(lo, min(hi, value))
unique = lambda values: len({round(x, 6) for x in values})
span = lambda values: [min(values), max(values)]
stats = []
request_errors = []
mean_errors = []
fl = []
for i, name in enumerate(names):
    values = collections.defaultdict(list)
    for j, row in enumerate(rows):
        s, q = row['applied_audit'], row['policy_request']
        h = s['actuator_target_effect_audit']['policy_headroom_evidence']
        request = h['requested_policy_residual_full12'][i]
        final = s['actual_drive_target_full12'][i]
        raw, mean = q['selected_raw_full12'][i], q['conditional_mean_full12'][i]
        old = q['previous_filtered_request_full12'][i]
        cap = q['current_cap_full12'][i]
        holds = sum(t['handoff_hold_used'] for t in s['actuator_target_effect_audit_ticks'])
        maximum = (60. if i < 8 else 1.8)*(s['physics_ticks']-holds)/120.
        request_replay = clamp(math.tanh(raw)*cap, old-maximum, old+maximum)
        mean_request = clamp(math.tanh(mean)*cap, old-maximum, old+maximum)
        request_errors.append(abs(request_replay-request))
        mean_errors.append(abs(mean-(.9*q['history_center_full12'][i]+.1*q['base_mean_full12'][i])))
        for key, value in dict(raw=raw, mean=mean, sigma=q['effective_sigma_full12'][i],
                request=request, effective=h['effective_policy_residual_full12'][i], final=final,
                headclip=i in h['clipped_servo_indices'], final_minus58=abs(final+58)<1e-6,
                downstream_changed=abs(final-h['candidate_native_target_before_final_slew_full12'][i])>1e-5,
                rate_limited=abs(math.tanh(raw)*cap-request)>1e-5,
                sample_request_differs_mean=abs(request-mean_request)>1e-5,
                request_positive_change=request-old>1e-5,
                sigma_multiplier=q['rear_local_sigma_multiplier_full12'][i]).items():
            values[key].append(value)
        if i == 1:
            feature = q['rear_owner_observed_features']
            previous_final = s['actuator_target_effect_audit']['previous_final_drive_servo_deg'][i]
            item = dict(index=j, tick=s['physics_tick'], mean=mean, sigma=q['effective_sigma_full12'][i],
                raw=raw, request=request, mean_request_same_history=mean_request, final=final,
                previous_final_last_native_tick=previous_final, headroom_clip=i in h['clipped_servo_indices'])
            # Stable recorded input anchor window; not a rerun of contact/state transitions.
            if j >= 20:
                assert feature[9] == 1 and feature == rows[20]['policy_request']['rear_owner_observed_features']
                anchor, anchor_request = feature[1]*180., feature[5]*180.
                def fixed_owner(req):
                    relative = clamp(req-anchor_request, -cap, cap)
                    return anchor+clamp(relative, min(0.,-58.-anchor), max(0.,58.-anchor))
                item.update(anchor=anchor, anchor_request=anchor_request,
                    request_exit_threshold=-58.-anchor+anchor_request,
                    raw_goal_exit_threshold=math.atanh((-58.-anchor+anchor_request)/cap),
                    frozen_owner_final_replay=fixed_owner(request),
                    frozen_owner_mean_final=fixed_owner(mean_request))
            fl.append(item)
    aliases = collections.defaultdict(list)
    for j, value in enumerate(values['final']): aliases[round(value,6)].append(j)
    alias_groups = [dict(final=value, rows=len(indices), distinct_requests=unique(values['request'][j] for j in indices),
        distinct_raw=unique(values['raw'][j] for j in indices)) for value,indices in aliases.items()
        if len(indices)>1 and unique(values['raw'][j] for j in indices)>1]
    stats.append(dict(channel=name, cap=rows[0]['policy_request']['current_cap_full12'][i],
        ranges={key:span(values[key]) for key in ('mean','sigma','raw','request','effective','final')},
        counts={key:sum(values[key]) for key in ('headclip','final_minus58','downstream_changed',
            'rate_limited','sample_request_differs_mean','request_positive_change')},
        distinct={key:unique(values[key]) for key in ('raw','request','effective','final')},
        sigma_gate_counts=dict(collections.Counter(values['sigma_multiplier'])), alias_groups=alias_groups))
stable = fl[20:]
escapes = [fl[j] for j in range(1,59) if abs(fl[j-1]['final']+58)<1e-6 and fl[j]['final']>-58+1e-5]
rr_modes = collections.Counter()
for row in rows:
    leg=row['applied_audit']['semantic_task']['physical_evaluator']['current_legs']['RR']
    rr_modes['ground' if leg['ground_contact'] else 'legal_TOP_bearing' if leg['top_surface_contact'] and leg['within_top_xy'] and leg['bearing_verified'] and leg['bearing_force_n']>0 else 'other']+=1
result = dict(run=str(RUN.relative_to(ROOT)), fixed_global_window=[226305,226363], sampled_decisions=59,
    optimized_credit_claimed=0, prefix_credit=0, end_tick=6606, end_s=55.05, terminal='BODY_COLLISION',
    lifecycle_scope='first completed student episode only; outer 1024-decision run is continuing',
    request_replay_max_abs_error=max(request_errors), conditional_mean_max_abs_error=max(mean_errors),
    request_replay='cap*tanh(raw), then 60deg/s or 1.8rad/s2 slew from recorded REQUEST; subtract two recorded one-tick phase handoff holds',
    raw_to_request_not_direct_cap_product=True, channels=stats,
    rear_preparation_gate_counts=dict(collections.Counter(r['policy_request']['cooperative_prep_allowed'] for r in rows)),
    gate_switches=[dict(tick=r['applied_audit']['physics_tick'],
        rr_carry=r['policy_request']['cooperative_observed_rr_carry_capture'],
        rr_reachable=r['policy_request']['cooperative_observed_rr_top_reachable'],
        rl_prep=r['policy_request']['cooperative_observed_rl_prep_transfer'],
        multiplier=r['policy_request']['rear_local_sigma_multiplier_full12'])
        for j,r in enumerate(rows) if j==0 or r['policy_request']['rear_local_sigma_multiplier_full12']
        !=rows[j-1]['policy_request']['rear_local_sigma_multiplier_full12']],
    all_recorded_phase_masks_one=all(all(x==1 for x in r['applied_audit']['actuator_target_effect_audit']['phase_mask_full12']) for r in rows),
    FLk_boundary_headroom_clip_overlap=sum(x['headroom_clip'] and abs(x['final']+58)<1e-6 for x in fl),
    FLk_RR_ground=dict(rows=sum(r['applied_audit']['semantic_task']['physical_evaluator']['current_legs']['RR']['ground_contact'] for r in rows),
        at_minus58=sum(r['applied_audit']['semantic_task']['physical_evaluator']['current_legs']['RR']['ground_contact'] and abs(r['applied_audit']['actual_drive_target_full12'][1]+58)<1e-6 for r in rows)),
    RR_endpoint_modes=dict(rr_modes),
    stable_owner_window=dict(first_tick=stable[0]['tick'],last_tick=stable[-1]['tick'], rows=len(stable),
        anchor=stable[0]['anchor'], anchor_request=stable[0]['anchor_request'],
        request_exit_threshold=stable[0]['request_exit_threshold'], raw_goal_exit_threshold=stable[0]['raw_goal_exit_threshold'],
        reconstruction_max_error=max(abs(x['frozen_owner_final_replay']-x['final']) for x in stable),
        sample_at_boundary=sum(abs(x['final']+58)<1e-6 for x in stable),
        same_history_mean_at_boundary=sum(abs(x['frozen_owner_mean_final']+58)<1e-6 for x in stable),
        sample_escapes_while_mean_at_boundary=[x['tick'] for x in stable if x['final']>-58+1e-5 and abs(x['frozen_owner_mean_final']+58)<1e-6],
        sample_boundary_while_mean_escapes=[x['tick'] for x in stable if abs(x['final']+58)<1e-6 and x['frozen_owner_mean_final']>-58+1e-5],
        limits='same recorded anchors/history only, no counterfactual state/contact/mapper rerun; no native 120Hz owner transition receipt'),
    actual_FLk_boundary_exits=escapes,
    selected_FLk=[fl[i] for i in (0,10,20,30,35,40,45,50,55,58)])
print(json.dumps(result, ensure_ascii=False, indent=2))

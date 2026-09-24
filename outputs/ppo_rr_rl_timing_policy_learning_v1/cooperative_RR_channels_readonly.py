"""Bounded RR6/7 append audit of the two sealed384 rear blocks, stdlib only."""
import collections,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUNS={
 'P07':('20260924T0038084596548Z_g49eb23163a6e_baf0b6006fea46a3a731db4c4d63d633',239),
 'P10':('20260924T0130272021180Z_g49eb23163a6e_5e6b94b24cce4a2fb9edc117565be855',143),
}
def stats(xs):
    return dict(n=len(xs),min=min(xs),max=max(xs),mean=sum(xs)/len(xs))
results=[]
for label,(runid,expected) in RUNS.items():
    run=ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/train'/runid
    m=json.loads((run/'training_manifest.json').read_text())
    assert m['lifecycle']=='SUCCEEDED' and m['actual_policy_decisions']==384
    channels={6:[],7:[]}; count=0;terminals=0;last=None
    with (run/'residual_and_projection_audit.jsonl').open() as f:
        for line in f:
            r=json.loads(line);p=r['policy_request'];a=r['applied_audit'];last=a
            terminals+=int(a['termination_reason'] is not None)
            if not p['cooperative_prep_allowed']:continue
            count+=1; effect=a['actuator_target_effect_audit'];h=effect['policy_headroom_evidence']
            for i in (6,7):
                final=a['actual_drive_target_full12'][i]
                hardlo,hardhi=h['servo_hard_limits_deg'][i]
                safelo,safehi=h['servo_safety_limits_deg'][i]
                requested=h['requested_policy_residual_full12'][i]
                desired=math.tanh(p['selected_raw_full12'][i])*p['current_cap_full12'][i]
                baseline=h['baseline_native_plus_controller_full12'][i]
                previous=effect['previous_final_drive_servo_deg'][i]
                candidate=h['candidate_native_target_before_final_slew_full12'][i]
                maxdelta=effect['tracking_reference_evidence']['maximum_delta_deg']
                channels[i].append(dict(raw=p['selected_raw_full12'][i],base_mean=p['base_mean_full12'][i],
                    conditional_mean=p['conditional_mean_full12'][i],sigma=p['effective_sigma_full12'][i],
                    residual_desired=desired,residual_requested=requested,
                    residual_effective=h['effective_policy_residual_full12'][i],baseline=baseline,
                    unprojected_requested_target=baseline+requested,final=final,
                    headroom_lower=safelo,hard_lower=hardlo,
                    rate_clipped=abs(desired-requested)>1e-5,
                    headroom_clipped=i in h['clipped_servo_indices'],
                    final_slew_clipped=abs(max(hardlo,min(hardhi,candidate))-previous)>maxdelta+1e-9,
                    at_headroom_lower=abs(final-safelo)<1e-8,at_true_hard_lower=abs(final-hardlo)<1e-8,
                    above_headroom_lower=final>safelo+1e-8,
                    requested_target_above_headroom_lower=baseline+requested>safelo+1e-8,
                    carry_reachable=bool(p['cooperative_observed_rr_carry_capture'] and p['cooperative_observed_rr_top_reachable']),
                    parent_receiving=bool(p['cooperative_parent_receiving_continuation_active']),
                    rl_prep=bool(p['cooperative_observed_rl_prep_transfer']),
                    rear_sigma_multiplier=p['rear_local_sigma_multiplier_full12'][i]))
    assert count==expected and terminals==0
    result=dict(run_label=label,run_dir=str(run),prep_samples=count,terminal_count=terminals,
        end_time_s=last['sim_time_s'],end_phase=last['phase_id'],task_success=last['task_success'],channels=[])
    for i,rs in channels.items():
        fields=('raw','base_mean','conditional_mean','sigma','residual_desired','residual_requested',
            'residual_effective','baseline','unprojected_requested_target','final')
        counts=('rate_clipped','headroom_clipped','final_slew_clipped','at_headroom_lower','at_true_hard_lower',
            'above_headroom_lower','requested_target_above_headroom_lower')
        result['channels'].append(dict(index=i,name='RR_hip' if i==6 else 'RR_knee',
            statistics={k:stats([x[k] for x in rs]) for k in fields},
            counts={k:sum(x[k] for x in rs) for k in counts},
            headroom_lower=sorted({x['headroom_lower'] for x in rs}),
            true_hard_lower=sorted({x['hard_lower'] for x in rs}),
            raw_positive=sum(x['raw']>0 for x in rs),conditional_mean_positive=sum(x['conditional_mean']>0 for x in rs),
            distinct_raw=len({x['raw'] for x in rs}),distinct_final_6dp=len({round(x['final'],6) for x in rs}),
            sigma_multiplier_counts=dict(collections.Counter(x['rear_sigma_multiplier'] for x in rs))))
    result['RR_knee_observed_gate_subgroups']=[]
    for label2,predicate in (
        ('carry_reachable_parent_not_receiving',lambda x:x['carry_reachable'] and not x['parent_receiving']),
        ('carry_reachable_parent_receiving',lambda x:x['carry_reachable'] and x['parent_receiving']),
        ('rl_prep_transfer',lambda x:x['rl_prep'])):
        subset=[x for x in channels[7] if predicate(x)]
        result['RR_knee_observed_gate_subgroups'].append(dict(label=label2,n=len(subset),
            at_headroom_lower=sum(x['at_headroom_lower'] for x in subset),
            raw_positive=sum(x['raw']>0 for x in subset),
            ranges={k:stats([x[k] for x in subset]) for k in ('baseline','raw','conditional_mean','sigma','residual_requested','final')} if subset else {}))
    results.append(result)
print(json.dumps(dict(schema='readonly.cooperative_RR6_RR7_prep.v1',runs=results,
    interpretation='All statistics conditional on observed request-start prep gate; endpoint final targets, not actual joint positions or causal sigma ablation.',
    scope='Only named sealed P07/P10 384 blocks; no activeDET, Torch/Isaac, production edits or model queries.'),separators=(',',':')))

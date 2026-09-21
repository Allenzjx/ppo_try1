"""First durable new-kernel128 only. Saved tensors/arithmetic, no actor forward."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import torch
import yaml
from audit_first_completed_update import read_prefix, stats

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/ppo_fl_capture_quality_v1/train/20260918T0627100878188Z_g3a50657a96c9_92db59ba0d774fa78e4cc53d5fad8c77'
VERSION='cap_transition_request_history_heteroscedastic_log_temperature_quarter_v1'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--update',type=int,choices=(1376,1377),default=1376)
    args=parser.parse_args()
    completed_blocks=args.update-1375
    torch.set_num_threads(1)
    updates,uh=read_prefix(RUN/'optimizer_updates.jsonl',completed_blocks)
    update=updates[-1]; number=update['ppo_update']; assert number==args.update
    all_rows,rh=read_prefix(RUN/'residual_and_projection_audit.jsonl',128*completed_blocks)
    rs=all_rows[-128:]
    likelihood=json.loads((RUN/f'rollouts/update_{number:06d}_likelihood.json').read_text())
    saved=torch.load(RUN/f'rollouts/rollout_{number:06d}.pt',map_location='cpu',weights_only=False)
    obs=saved['observations']['policy'][:,0]
    assert obs.shape==(128,372) and saved['actions'].shape==(128,1,12)
    assert rs[-1]['global_policy_decision']==update['global_policy_decisions']==180480+128*completed_blocks
    req=[r['policy_request'] for r in rs]
    tensor=lambda key:torch.tensor([r[key] for r in req],dtype=torch.float32)
    cap_cfg=yaml.safe_load((ROOT/'configs/ppo_fl_capture_quality_v1/execution_profile.yaml').read_text())['residual']['phase_caps_full12']
    caps=torch.tensor([cap_cfg[f'P{i:02d}'] for i in range(1,14)],dtype=torch.float32)
    stages=obs[:,:13].argmax(dim=-1)
    current=caps[stages]; prior=caps[(stages-1).clamp(min=0)]
    predecessor=obs[torch.arange(128),158+(stages-1).clamp(min=0)]>0.5
    gate=(obs[:,20:21]==0)&(stages[:,None]>0)&predecessor[:,None]&(current>prior)
    previous_raw=obs[:,195:207].clamp(-20,20)
    previous_request=obs[:,207:219]*torch.tensor([4,4,4,6,4,4,4,4,.12,.12,.12,.12])
    assert bool((~gate | (previous_request.abs()<=prior+1e-5)).all())
    ratio=torch.where(gate,previous_request/current,torch.zeros_like(previous_raw))
    assert bool((ratio.abs()<1).all())
    center=torch.where(gate,torch.atanh(ratio),previous_raw)
    mean=.1*tensor('base_mean_full12')+.9*center
    checks={
        'all_new_policy_version':all(r.get('policy_version')==VERSION for r in req),
        'raw_saved_exact':torch.equal(tensor('selected_raw_full12'),saved['actions'][:,0]),
        'collected_raw_saved_exact':torch.equal(torch.tensor([r['raw_policy_action_full12'] for r in rs]),saved['actions'][:,0]),
        'mean_saved_exact':torch.equal(tensor('conditional_mean_full12'),saved['distribution_params'][0][:,0]),
        'sigma_saved_exact':torch.equal(tensor('effective_sigma_full12'),saved['distribution_params'][1][:,0]),
        'logp_saved_exact':torch.equal(torch.tensor([r['old_log_probability'] for r in rs]),saved['actions_log_prob'][:,0,0]),
        'request_logp_equals_collection':all(r['policy_request']['selected_raw_log_probability']==r['old_log_probability'] for r in rs),
        'raw_history_from_same_saved_obs_exact':torch.equal(tensor('previous_raw_from_current_observation_full12'),previous_raw),
        'request_from_same_saved_obs_exact':torch.equal(tensor('previous_filtered_request_full12'),previous_request),
        'actual_gate_matches_saved_obs':torch.equal(tensor('cap_transition_gate_full12').bool(),gate),
        'current_caps_exact':torch.equal(tensor('current_cap_full12'),current),
        'prior_caps_exact':torch.equal(tensor('predecessor_cap_full12'),prior),
        'same_phase_center_equals_raw_exact':torch.equal(tensor('history_center_full12')[~gate],previous_raw[~gate]),
        'center_algebra_error_below_2e7':float((tensor('history_center_full12')-center).abs().max())<2e-7,
        'conditional_formula_error_below_2e7':float((tensor('conditional_mean_full12')-mean).abs().max())<2e-7,
        'one_sample_no_extra_forward_rng':all(r['sampling_draws']==1 and r['extra_model_forwards']==r['extra_random_draws']==0 for r in req),
        'all_likelihood_new_obs_provenance':all(m['history_source']=='this_saved_observation_stage0_13_age20_completed158_171_raw195_207_request207_219_not_shuffled_neighbor' for m in likelihood['minibatches']),
        'all_physical_ticks_verified':all(r['applied_audit']['actuator_target_effect_audit_summary']['all_ticks_verified'] for r in rs),
    }
    first=likelihood['minibatches'][0]
    checks['first_unupdated_minibatch_ratio_near_one']=max(abs(v-1) for v in first['ratio'])<2e-5
    checks['first_unupdated_minibatch_no_strict_clipping']=not any(first['clipped_branch_strictly_active'])
    uses=Counter(i for m in likelihood['minibatches'] for indices in m['rollout_flat_indices'] for i in indices)
    checks['each_sample_used_5_times']=uses==Counter({i:5 for i in range(128)})
    logp_recomputed=torch.distributions.Normal(saved['distribution_params'][0][:,0],saved['distribution_params'][1][:,0]).log_prob(saved['actions'][:,0]).sum(-1)
    logp_error=float((logp_recomputed-saved['actions_log_prob'][:,0,0]).abs().max())
    checks['cpu_saved_gaussian_logp_error_below_2e5']=logp_error<2e-5
    gates=[]
    age_zero=[]
    handoffs=[]
    for i,(r,q) in enumerate(zip(rs,req)):
        a=r['applied_audit']
        if q['encoded_stage_age']==0:
            age_zero.append(dict(index=i,phase=a['phase_id'],predecessor_completed=q['predecessor_completed'],
                                 gated_channels=[j for j,x in enumerate(q['cap_transition_gate_full12']) if x]))
        if any(q['cap_transition_gate_full12']):
            gates.append(dict(index=i,global_decision=r['global_policy_decision'],phase=a['phase_id'],
                policy_request=q,projected_request=a['projected_residual_full12'],native_verified=a['actuator_target_effect_audit']['verified']))
        if a['phase_transition_action_jump']:
            handoffs.append(dict(index=i,phase=a['phase_id'],end_phase=a['end_phase_id'],
                evidence=a['phase_transition_action_jump']))
    result=dict(schema='wlr50_clean.actual_first_request_kernel_update_audit.v1',run=str(RUN),
        scope=f'completed_update{number}_128_only; saved arithmetic and actual optimizer hook; no actor forward',
        source_sha256=dict(optimizer_prefix_lines=completed_blocks,optimizer_prefix_sha256=uh,
                          audit_prefix_lines=128*completed_blocks,audit_prefix_sha256=rh),
        update=update,counts=dict(Counter(r['applied_audit']['phase_id'] for r in rs)),checks=checks,
        all_checks_passed=all(checks.values()),center_max_error=float((tensor('history_center_full12')-center).abs().max()),
        conditional_max_error=float((tensor('conditional_mean_full12')-mean).abs().max()),gaussian_logp_cpu_max_error=logp_error,
        first_minibatch_ratio=stats(first['ratio']),first_minibatch_max_abs_ratio_minus_one=max(abs(v-1) for v in first['ratio']),
        first_minibatch_scope='only its actual32 samples before first optimizer step; not all128 at original parameters',
        optimizer_minibatches=len(likelihood['minibatches']),gated_decision_count=len(gates),gate_events=gates,age_zero_cases=age_zero,
        ordinary_handoffs=handoffs,observed_P05_to_P06=any(r['applied_audit']['phase_id']=='P05' and r['applied_audit']['end_phase_id']=='P06' for r in rs),
        prefix_not_training_credit=True,extra_forwards=0,extra_rng_draws=0,extra_optimizer_steps=0,
        limits=['Only observed gate positives are real examples; absent P06 handoff is not a passed physical test.',
                'Saved-observation arithmetic validates wiring, not a counterfactual trajectory or task success.',
                'RNG/Adam migration equality belongs to official load receipt; optimizer data here are newly changed by the actual update.'])
    dest=OUT/f'update{number}_actual_request_kernel_audit.json'
    with dest.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({key:result[key] for key in ('counts','all_checks_passed','checks','gated_decision_count','age_zero_cases','observed_P05_to_P06','first_minibatch_max_abs_ratio_minus_one','gaussian_logp_cpu_max_error')},indent=2))
    if not result['all_checks_passed']:raise RuntimeError('saved audit mismatch; see diagnostic, no production changes made')


if __name__=='__main__':main()

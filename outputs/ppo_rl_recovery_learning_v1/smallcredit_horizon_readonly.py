"""Bounded two-sealed-run credit check. Stdlib only; no tensor/model access."""
from collections import Counter
import json
import math
from pathlib import Path
import runpy

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
helper=runpy.run_path(str(OUT/'inspect_course.py'))
rows,small,batches=helper['rows'],helper['small_json'],helper['minibatches']
RUNS={
    'P07': '20260924T0038084596548Z_g49eb23163a6e_baf0b6006fea46a3a731db4c4d63d633',
    'P10': '20260924T0130272021180Z_g49eb23163a6e_5e6b94b24cce4a2fb9edc117565be855'}
SELECT={'P07':(5680,5688,5696,5768,5776,5800,5808,5816,5824,6184,6192,6208,6224,6232),
        'P10':(6144,6184,6192,6200,6208,6232,7184,7192,7296,7304,7384,7392,7496)}
result={}
for label,name in RUNS.items():
    run=ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/train'/name
    manifest=small(run/'training_manifest.json')
    assert manifest['lifecycle']=='SUCCEEDED' and manifest['actual_policy_decisions']==384
    updates=list(rows(run/'optimizer_updates.jsonl')); advantages=list(rows(run/'advantage_audit.jsonl'))
    assert len(updates)==len(advantages)==3
    data=[]
    for r in rows(run/'residual_and_projection_audit.jsonl'):
        assert len(data)<384
        a=r['applied_audit']; task=a['semantic_task']; ev=task['physical_evaluator']
        rr,rl=ev['current_legs']['RR'],ev['current_legs']['RL']; rb=a['reward_breakdown']
        data.append(dict(decision=r['global_policy_decision'],tick=a['physics_tick'],time_s=a['sim_time_s'],
            request_phase=a['phase_id'],end_phase=a['end_phase_id'],reward=r['reward'],old_value=r['old_value'],
            terminal=r['terminal'],RR_bearing=task['rr_capture_continuation']['rr_current_bearing'],
            RR_contact=rr['contact_mode'],RR_force_n=rr['bearing_force_n'],RR_gap_m=rr['clearance_m'],
            RL_current_qualified=rl['current_lift_valid'],RL_contact=rl['contact_mode'],
            potential_before=rb['potential_before'],potential_after=rb['potential_after'],
            potential_shaping=rb['potential_shaping'],counterroll_cost=rb.get('cooperative_counterroll_cost'),
            old_logp=r['old_log_probability'],raw_GAE=None,
            raw_GAE_status='not_saved_per_nonterminal_row_in_JSON; not_reconstructed'))
    assert len(data)==384 and not any(r['terminal'] for r in data)
    summaries=[]
    for k,(u,a) in enumerate(zip(updates,advantages)):
        assert a['sample_count']==128 and a['ppo_update_intended']==u['ppo_update']
        assert a['last_global_policy_decision']==u['global_policy_decisions']==data[k*128+127]['decision']
        assert a['teacher_prefix_samples_included'] is False
        assert a['stored_advantage_semantics']=='official_whole_rollout_standardized_GAE'
        visits=Counter()
        for mb in batches(run/'rollouts'/f"update_{u['ppo_update']:06d}_likelihood.json"):
            for j,idxs in enumerate(mb['rollout_flat_indices']):
                assert len(idxs)==1
                i=idxs[0]; row=data[k*128+i]; visits[i]+=1
                value=mb['actual_advantage'][j]
                while isinstance(value,list): assert len(value)==1; value=value[0]
                if 'actual_standardized_advantage' in row: assert row['actual_standardized_advantage']==value
                row['actual_standardized_advantage']=value
                old=mb['old_log_probability'][j]
                while isinstance(old,list): assert len(old)==1; old=old[0]
                assert old==row['old_logp']
        assert visits==Counter({i:5 for i in range(128)})
        block=data[k*128:(k+1)*128]; gamma=a['gamma']; lam=a['lambda']
        discounted=sum((gamma**i)*r['reward'] for i,r in enumerate(block))
        summary=dict(update=u['ppo_update'],first_decision=block[0]['decision'],last_decision=block[-1]['decision'],
            time_start_s=block[0]['time_s']-1/15,time_end_s=block[-1]['time_s'],
            contiguous_128_reward_sum=sum(r['reward'] for r in block),
            discounted_128_rewards_only=discounted,no_bootstrap_added=True,
            first_logged_old_value=block[0]['old_value'],last_pre_action_old_value=block[-1]['old_value'],
            actual_tail_bootstrap_value=None,tail_evidence=a['tail_bootstrap'],
            next_block_first_value_after_critic_update=data[(k+1)*128]['old_value'] if k<2 else None,
            next_value_is_not_previous_rollout_bootstrap=True,
            raw_GAE=a['overall']['raw_gae_returns_minus_old_values'],
            standardized_advantage=a['overall']['stored_advantages'],
            old_values=a['overall']['old_values'],return_targets=a['overall']['returns'],
            value_loss=u['value_loss'],kl=u['kl_mean'],clip_fraction=u['clip_fraction'])
        summaries.append(summary)
    windows=[]
    for i,row in enumerate(data):
        if row['tick'] not in SELECT[label]: continue
        end=min(len(data),(i//128+1)*128)
        following=data[i:end]
        windows.append({**row,'samples_including_this_to_rollout_end':len(following),
            'discounted_observed_rewards_to_rollout_end_only':sum(gamma**j*r['reward'] for j,r in enumerate(following)),
            'no_terminal_reached_in_this_run':True})
    changes=[]
    for left,right in zip(data,data[1:]):
        changed=[]
        if left['RR_bearing']!=right['RR_bearing']: changed.append('RR_gain' if right['RR_bearing'] else 'RR_loss')
        if left['RL_current_qualified']!=right['RL_current_qualified']: changed.append('RL_qualification_gain' if right['RL_current_qualified'] else 'RL_qualification_loss')
        if changed:
            changes.append(dict(previous_tick=left['tick'],tick=right['tick'],event=changed,
                decision=right['decision'],reward=right['reward'],old_value=right['old_value'],
                standardized_advantage=right['actual_standardized_advantage'],
                same_rollout_as_previous=((right['decision']-data[0]['decision'])//128
                    ==(left['decision']-data[0]['decision'])//128)))
    result[label]=dict(run=str(run),optimized_decisions=384,updates=3,optimizer_steps=60,
        terminal_count=0,final_nonterminal_budget_boundary=True,blocks=summaries,
        event_examples=windows,contact_qualification_transition_brackets=changes,
        within_rollout_transitions=sum(c['same_rollout_as_previous'] for c in changes),
        cross_rollout_transitions=sum(not c['same_rollout_as_previous'] for c in changes),
        first_rollout_capture_RL_events_precede_first_update=True if label=='P07' else None,
        initial_RR_capture_is_zero_credit_prefix=True if label=='P10' else False)

credit=dict(gamma=gamma,gae_lambda=lam,decision_hz=15,rollout_decisions=128,rollout_seconds=128/15,
    TD_residual_trace_weight_per_decision=gamma*lam,
    trace_half_life_decisions=math.log(.5)/math.log(gamma*lam),
    trace_half_life_seconds=math.log(.5)/math.log(gamma*lam)/15,
    trace_weights_at_decisions={str(n):(gamma*lam)**n for n in (15,30,60,128,256)},
    reward_discount_weights_at_decisions={str(n):gamma**n for n in (15,30,60,128,256)},
    mathematical_not_measured_counterfactual=True)
conclusion=("No strong causal evidence that 128 alone caused RR/RL failure: the first learner RR capture/loss/RL lift "
    "and most repeated gain/loss transitions occur within an optimized rollout; ordinary phase changes are nonterminal. "
    "There is a real bootstrap dependence and limited outcome coverage: both 384-decision rear blocks end nonterminal, "
    "and no numerical same-policy tail V or terminal outcome is logged. Longer continuous rear collection is supported; "
    "changing rollout128 to256 is a separate testable variance/credit choice, not an established fix. "
    "Previously proven issued-owner persistence, current-support loss and saturated action/proxy coverage are more direct "
    "implementation/state issues. Test their already-versioned correction before attributing residual failure to horizon.")
payload=dict(schema='readonly.sealed_P07_P10_credit_horizon.v1',scope='only prior49eb P07/P10 sealed384 each',
    no_torch_or_tensor_access=True,no_model_forward=True,no_production_changes=True,
    learning_parameters=credit,runs=result,conclusion=conclusion,
    caveats=['All per-row standardized advantages are actual likelihood audit values, signs unchanged.',
        'Per-row nonterminal raw GAE is unavailable in these JSON receipts; only exact aggregate raw GAE is reported.',
        'Rewards-only discounted sums are not PPO return targets; no missing bootstrap was invented.',
        'The next-block critic has already updated and cannot replace the prior rollout bootstrap.',
        'Positive raw GAE means better than the old critic prediction, not success or physical benefit.',
        'No terminal-to-event return comparison is possible: both selected rear blocks have zero task terminals.'])
jsonpath=OUT/'smallcredit_horizon_readonly.json'; mdpath=OUT/'smallcredit_horizon_readonly.md'
assert not jsonpath.exists() and not mdpath.exists()
with jsonpath.open('x',encoding='utf-8') as f: json.dump(payload,f,indent=2,ensure_ascii=False,allow_nan=False)
text=['# 128-decision credit horizon: prior sealed P07/P10 only','',conclusion,'',
    f'Actual gamma={gamma}, lambda={lam}; 128 decisions={128/15:.6f}s. GAE TD-residual trace half-life={credit["trace_half_life_seconds"]:.3f}s; weight at128={credit["trace_weights_at_decisions"]["128"]:.6f}, at256={credit["trace_weights_at_decisions"]["256"]:.6f}. These are formula weights, not a measured alternative-run result.','',
    '| Run/update | physical window s | observed rewards sum | discounted rewards only | mean old V | mean return target | raw GAE + / - |','| --- | --- | ---: | ---: | ---: | ---: | --- |']
for label,r in result.items():
    for b in r['blocks']:
        text.append(f'| {label}/{b["update"]} | {b["time_start_s"]:.3f}–{b["time_end_s"]:.3f} | {b["contiguous_128_reward_sum"]:.6f} | {b["discounted_128_rewards_only"]:.6f} | {b["old_values"]["mean"]:.5f} | {b["return_targets"]["mean"]:.5f} | {b["raw_GAE"]["positive_count"]}/{b["raw_GAE"]["negative_count"]} |')
text+=['','## Actual candidate-event examples','',
    'Reward and standardized advantage are original logged values. They are not relabeled from contact success/failure. Nonterminal raw GAE remains unavailable per row.','',
    '| Run/tick | RR contact / force N | RL qualified | reward | old V | actual std advantage | rows left in rollout incl. this |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
for label,ticks in {'P07':(5688,5696,5776,5808,5824,6208,6232),'P10':(6144,6208,7192,7304,7392,7496)}.items():
    for e in result[label]['event_examples']:
        if e['tick'] in ticks:
            text.append(f'| {label}/{e["tick"]} | {e["RR_contact"]} / {e["RR_force_n"]:.3f} | {e["RL_current_qualified"]} | {e["reward"]:.6f} | {e["old_value"]:.5f} | {e["actual_standardized_advantage"]:.5f} | {e["samples_including_this_to_rollout_end"]} |')
text+=['','## What this does and does not support','',
    'P07 learner RR placement5682 and RL qualification5772 both precede the first course update1710; the resulting capture/loss/lift transitions are already in its first128 samples. P10 initial RR placement6133 is in the zero-credit N prefix, while later contact losses and RL attempts are learner samples.',
    f'P07 sampled contact/qualification transitions within/across rollout boundaries: {result["P07"]["within_rollout_transitions"]}/{result["P07"]["cross_rollout_transitions"]}; P10: {result["P10"]["within_rollout_transitions"]}/{result["P10"]["cross_rollout_transitions"]}. These are endpoint transition brackets, not all120Hz events.',
    'Both runs stop after384 decisions/25.6s of learner collection, with task incomplete and no terminal. Therefore there is no observed terminal return to propagate across the full rear task. All six128 tails bootstrap. Last pre-action V is not the unlogged end-state V; next-block V is measured after a critic update.',
    'Keep the distinction: extending continuous collection to a real rear outcome addresses demonstrated missing outcome coverage; increasing rollout length may reduce reliance on a tail bootstrap but is not proven to repair the already-observed owner/state/action-limit problems.',
    'Source evidence: cooperative_sealed384_summary.md, cooperative_P10_384_summary.md, their exact run journals plus six advantage/optimizer/likelihood records. No current active probe, other history, tensor, or model was read.']
with mdpath.open('x',encoding='utf-8') as f: f.write('\n'.join(text)+'\n')
print(json.dumps(dict(json=str(jsonpath),markdown=str(mdpath),optimized_decisions=768,physics_started=False)))

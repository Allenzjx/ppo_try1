"""Completed update 1726 only; stdlib streaming, no checkpoint/model load."""
from collections import Counter, defaultdict
from itertools import islice
import json
import math
from pathlib import Path
import runpy

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUN=ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7'
U=runpy.run_path(str(OUT/'inspect_course.py'))
def first_json(path):
    with path.open(encoding='utf-8-sig') as f:
        return json.loads(next(f))
opt=first_json(RUN/'optimizer_updates.jsonl')
adv=first_json(RUN/'advantage_audit.jsonl')
assert opt['ppo_update']==adv['ppo_update_intended']==1726
assert opt['global_policy_decisions']==adv['last_global_policy_decision']==225408
assert opt['optimizer_steps']==20 and adv['sample_count']==128
assert adv['first_global_policy_decision']==225281 and adv['teacher_prefix_samples_included'] is False
with (RUN/'residual_and_projection_audit.jsonl').open(encoding='utf-8-sig') as f:
    rows=[json.loads(line) for line in islice(f,128)]
assert len(rows)==128 and [x['global_policy_decision'] for x in rows]==list(range(225281,225409))
manifest=U['small_json'](RUN/'run_manifest.started.json')
support=U['bound_support'](manifest['runtime_contract'])
exposure=Counter(); stored_adv={}; latest={}; first={}; max_current_logp_error=0.; minibatches=0
for mb in U['minibatches'](RUN/'rollouts/update_001726_likelihood.json'):
    assert mb['minibatch_index']==minibatches
    minibatches+=1
    for j,candidates in enumerate(mb['rollout_flat_indices']):
        assert len(candidates)==1, 'ambiguous row identity; do not guess'
        i=candidates[0]; row=rows[i]
        assert abs(mb['old_log_probability'][j]-row['old_log_probability'])<1e-6
        if i in stored_adv: assert stored_adv[i]==mb['actual_advantage'][j]
        stored_adv[i]=mb['actual_advantage'][j]; exposure[i]+=1
        current_mean=mb['current_conditional_mean'][j]; current_sigma=mb['current_conditional_sigma'][j]
        err=abs(U['gaussian_logp'](row['raw_policy_action_full12'],current_mean,current_sigma)-mb['optimization_log_probability'][j])
        max_current_logp_error=max(max_current_logp_error,err)
        entry=dict(minibatch_index=mb['minibatch_index'], current_mean=current_mean, current_sigma=current_sigma,
            mean_delta_from_collection=[a-b for a,b in zip(current_mean,row['old_distribution_mean_full12'])],
            ratio=mb['ratio'][j], clipped=mb['clipped_branch_strictly_active'][j],
            loss_gradient_wrt_network_mean=mb['loss_gradient_wrt_network_mean_full12'][j])
        latest[i]=entry
        first.setdefault(i,entry)
assert minibatches==20 and len(exposure)==128 and set(exposure.values())=={5}
assert max_current_logp_error<.0001
assert max(abs(sum(stored_adv.values())/128-adv['overall']['stored_advantages']['mean']),0)<1e-8

def stats(values):
    values=list(values)
    return dict(n=len(values), minimum=min(values), maximum=max(values), mean=sum(values)/len(values),
        positive=sum(x>0 for x in values),negative=sum(x<0 for x in values)) if values else dict(n=0)

groups=defaultdict(list); facts=[]; phases=Counter()
for i,row in enumerate(rows):
    a=row['applied_audit']; task=a['semantic_task']; ev=task['physical_evaluator']; legs=ev['current_legs']
    flags,_=U['physical_windows'](task,support)
    h=a['actuator_target_effect_audit']['policy_headroom_evidence']
    margins=task['transfer_roles']['RR']['receiver_workspace_state']['joint_range_margin_deg']['front_left_knee']
    near=margins['negative_deg']<=h['servo_reserve_deg']
    rr_top=U['bearing'](legs['RR'],support['force_noise_floor_n'],top=True)
    fl_clip=1 in h['clipped_servo_indices'] and h['requested_policy_residual_full12'][1]<h['effective_policy_residual_full12'][1]
    rl_initial=any(x.get('leg')=='RL' and x.get('event')=='whole_body_initial_clearance'
        for x in ev['history'].get('lift_attempt_events',[]))
    # First episode's explicit initial-clearance event is at tick 6099; do not
    # confuse initial clearance with qualified AIR, support or full success.
    if i==117:
        assert a['physics_tick']==6103 and legs['RL']['air'] is True and rl_initial
        assert any(x.get('leg')=='RL' and x.get('event')=='whole_body_initial_clearance'
            and x.get('physics_tick')==6099 for x in ev['history']['lift_attempt_events'])
    membership=['whole_update']
    if i<118: membership.append('completed_episode0')
    else: membership.append('next_episode_first10_nonterminal')
    if rr_top: membership.append('RR_current_TOP_bearing')
    if 'FR_directed_body_CoM_motion_with_RL_unload' in flags: membership.append('FR_window_motion_and_RL_unload')
    if near: membership.append('FL_actual_within_existing_2deg_lower_reserve')
    if fl_clip: membership.append('FL_lower_target_headroom_clipped')
    if rl_initial: membership.append('RL_initial_clearance_not_qualified')
    if i==86: membership.append('RR_first_placement_decision')
    for name in membership: groups[name].append(i)
    phases[a['phase_id']]+=1
    b=a['reward_breakdown']
    facts.append(dict(global_policy_decision=row['global_policy_decision'],episode_index=0 if i<118 else 1,
        input_phase=a['phase_id'],endpoint_tick=a['physics_tick'],reward=row['reward'],old_value=row['old_value'],
        stored_standardized_advantage=stored_adv[i],
        directly_logged_individual_raw_GAE=(adv['terminal_samples'][0]['raw_gae_returns_minus_old_values'] if i==117 else None),
        raw_GAE_availability='only terminal individual and phase/whole aggregate raw GAE are logged; no inverse-normalization fabrication',
        potential_before=b['potential_before'],potential_after=b['potential_after'],potential_shaping=b['potential_shaping'],
        terminal_event=b['terminal_event'],weighted_reward_families=b['families'],
        cooperative_counterroll_cost=b.get('cooperative_counterroll_cost'),
        RR_current_TOP_bearing=rr_top,RR_bearing_force_n=legs['RR']['bearing_force_n'],
        RL_air=legs['RL']['air'],RL_current_lift_valid=legs['RL'].get('current_lift_valid'),
        FL_actual_negative_margin_deg=margins['negative_deg'],
        final_target_FLk_FRk_RRk=[a['actual_drive_target_full12'][j] for j in (1,3,7)],
        original_raw_FLk_FRk_RRk=[row['raw_policy_action_full12'][j] for j in (1,3,7)],
        original_mean_FLk_FRk_RRk=[row['old_distribution_mean_full12'][j] for j in (1,3,7)],
        last_audited_pre_minibatch_mean_FLk_FRk_RRk=[latest[i]['current_mean'][j] for j in (1,3,7)],
        last_audited_mean_delta_FLk_FRk_RRk=[latest[i]['mean_delta_from_collection'][j] for j in (1,3,7)],
        last_audited_minibatch=latest[i]['minibatch_index'],memberships=membership))

summaries={}
for name,indices in groups.items():
    summaries[name]=dict(count=len(indices),global_decisions=[rows[i]['global_policy_decision'] for i in indices],
        raw_reward=stats(rows[i]['reward'] for i in indices),old_value=stats(rows[i]['old_value'] for i in indices),
        standardized_advantage=stats(stored_adv[i] for i in indices),
        raw_GAE_individual_group_summary=None,
        last_audited_mean_delta={joint:stats(latest[i]['mean_delta_from_collection'][j] for i in indices)
            for joint,j in (('FL_knee',1),('FR_knee',3),('RR_knee',7))},
        final_epoch_pre_minibatch_clipped_samples=sum(latest[i]['clipped'] for i in indices))
selected=[86,88,89,94,95,102,110,111,112,115,116,117,118,127]
report=dict(schema='completed_update1726_credit_readonly.v1',run=str(RUN),
    completed_update=opt, counts=dict(real_decisions=128,ppo_updates=1,optimizer_steps=20,
        likelihood_minibatches=20,raw_row_exposures=640,each_raw_row_exposures=5,phases=dict(phases)),
    global_decision_range=[225281,225408],excluded='All rows >=225409, including episode2 terminal 225434; not read or credited.',
    episode_boundary='118 rows completed HARD_JOINT_LIMIT episode plus first10 nonterminal rows after fresh zero-credit successful_nominal prefix',
    gamma=adv['gamma'],lambda_=adv['lambda'],discount_clock=adv['discount_clock'],
    terminal_samples=adv['terminal_samples'],tail_bootstrap=adv['tail_bootstrap'],
    recorded_overall=adv['overall'],recorded_per_phase=adv['by_request_phase'],physical_groups=summaries,
    selected_event_rows=[facts[i] for i in selected],
    maximum_current_conditional_logp_error=max_current_logp_error,
    interpretation_limits=[
        'Every reported sample was actually used five times in this completed update; no new/extra model forward.',
        'Conditional mean differences use each saved input at its last audited minibatch forward, before that minibatch optimizer step. They are NOT an independent final-checkpoint evaluation, per-sample Adam change, or proof of learned physical behavior.',
        'Positive instantaneous contact reward can coexist with negative standardized GAE when followed soon by a safety failure; negative advantage is not evidence contact itself was penalized.',
        'Whole/phase raw GAE and terminal raw GAE are directly logged; arbitrary physical-window individual raw GAE is N/A. No unlogged tail bootstrap or reverse standardized inference is invented.',
        'Overlapping endpoint groups are observed correlations, not causal credit decomposition; RL initial clearance is not qualified lift.'])
base=OUT/'update1726_credit_readonly'
assert not base.with_suffix('.json').exists() and not base.with_suffix('.md').exists()
md=['# Completed PPO update 1726: physical-window credit','',
    'Only globals 225281–225408: 128 real samples / 1 PPO update / 20 optimizer steps. All 128 original raw actions appear exactly five times in 20 likelihood minibatches. No row ≥225409 read or credited.',
    f'Actual request phases: {dict(phases)}. First118 terminate HARD_JOINT_LIMIT; next10 are a new episode, nonterminal at the rollout boundary. Prefix has zero credit.',
    f'LR={opt["optimizer_learning_rate"]}; observed KL mean={opt["kl_mean"]:.6f}, clip fraction={opt["clip_fraction"]:.6f}, value loss={opt["value_loss"]:.6f}. These are learning diagnostics, not physical success.',
    '', '| Physical endpoint group | n | Reward mean [min,max] | Old V mean | Std advantage mean [min,max] | +/− advantage |',
    '|---|---:|---|---:|---|---|']
for name in ('RR_first_placement_decision','RR_current_TOP_bearing','FR_window_motion_and_RL_unload',
             'RL_initial_clearance_not_qualified','FL_lower_target_headroom_clipped','FL_actual_within_existing_2deg_lower_reserve'):
    s=summaries[name];r=s['raw_reward'];v=s['old_value'];a=s['standardized_advantage']
    md.append(f'| {name} | {s["count"]} | {r["mean"]:.6f} [{r["minimum"]:.6f},{r["maximum"]:.6f}] | {v["mean"]:.6f} | {a["mean"]:.6f} [{a["minimum"]:.6f},{a["maximum"]:.6f}] | {a["positive"]}/{a["negative"]} |')
md+=['','Raw GAE: directly recorded whole-rollout mean −14.902727, range [−30.415251,+1.699550]; P12 mean −28.397252. Terminal raw GAE −30.415251. Individual nonterminal raw GAE for these physical groups is **N/A** in JSON; only standardized advantages are bound per row. γ=.9985, λ=.99; terminal gets no bootstrap; the second-episode tail bootstraps from an unlogged official final value.',
    '', '## Direction visible in existing update forwards', '',
    'Δ below is last audited conditional μ minus original collection μ at the **same saved input**, before its last minibatch step (indices16–19), not final checkpoint μ or physical target change.', '',
    '| Group | FL knee Δμ mean | FR knee Δμ mean | RR knee Δμ mean |','|---|---:|---:|---:|']
for name in ('RR_first_placement_decision','RR_current_TOP_bearing','FR_window_motion_and_RL_unload','FL_lower_target_headroom_clipped','whole_update'):
    s=summaries[name]['last_audited_mean_delta']
    md.append(f'| {name} | {s["FL_knee"]["mean"]:+.7f} | {s["FR_knee"]["mean"]:+.7f} | {s["RR_knee"]["mean"]:+.7f} |')
md+=['','## Interpretation','',
    f'RR first placement (225367) reward {facts[86]["reward"]:+.6f}, old V {facts[86]["old_value"]:.6f}, standardized advantage {facts[86]["stored_standardized_advantage"]:+.6f}. Contact makes a positive immediate contribution; subsequent safety failure affects its return, so this is not evidence that reward explicitly discourages touching the top.',
    'Terminal225398 reward −43.350361 = approximately −40 safety event −3.349196 terminal potential removal −0.001167 elapsed-time cost. Weighted smoothness/contact/stability families are zero there; the logged nonzero *unweighted* smoothness diagnostic is not a competing actual penalty.',
    'The observed mean directions/ratios do not establish that the policy has learned a safe knee trajectory. Multiple raw samples still shared clipped final targets in collection; only a subsequent physical evaluation can test changed behavior. This one update cannot isolate whether clipped exploration, source demand, dynamics, or data coverage dominates.',
    '', 'No model/checkpoint/Torch/Isaac load or production edits. Detailed event rows and per-phase directly logged raw-GAE statistics are in the companion JSON.']
base.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
base.with_suffix('.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
print('\n'.join(md))

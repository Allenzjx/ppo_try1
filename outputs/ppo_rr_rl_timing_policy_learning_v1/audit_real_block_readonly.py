"""Read sealed update JSON and a bounded decision prefix; stdlib only, no writes.

This script does not import project code, Torch, CUDA or Isaac, load tensors,
evaluate a model, consume RNG, or modify the live run. Derived JSON is stdout.
"""
import collections
import datetime
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260923T1908118345669Z_gfa4b98ed506e_1e77c2dd353745ebb86708c5095a34c9'
OUTPUT = ROOT / 'outputs/ppo_rr_rl_timing_policy_learning_v1'
FIRST_UPDATE = 1689
FIRST_DECISION = 220545
COUNT = 128


def read(name):
    return json.loads(name.read_text(encoding='utf-8'))


def scalar(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def logp(raw, mean, sigma):
    assert len(raw) == len(mean) == len(sigma) == 12
    assert all(math.isfinite(x) for x in raw + mean + sigma) and min(sigma) > 0.
    return sum(-.5*((x-m)/s)**2 - math.log(s) - .5*math.log(2*math.pi)
               for x,m,s in zip(raw,mean,sigma))


with (RUN/'residual_and_projection_audit.jsonl').open(encoding='utf-8') as handle:
    rows = [json.loads(line) for line in itertools.islice(handle, COUNT)]
assert len(rows) == COUNT
assert [r['global_policy_decision'] for r in rows] == list(range(FIRST_DECISION,FIRST_DECISION+COUNT))
with (RUN/'optimizer_updates.jsonl').open(encoding='utf-8') as handle:
    update = json.loads(next(handle))
with (RUN/'advantage_audit.jsonl').open(encoding='utf-8') as handle:
    advantages = json.loads(next(handle))
assert update['ppo_update'] == FIRST_UPDATE and update['global_policy_decisions'] == FIRST_DECISION+COUNT-1
likelihood = read(RUN/'rollouts/update_001689_likelihood.json')
publication = read(OUTPUT/'initial_publication_gfa4b98ed506e.json')
checkpoint = read(OUTPUT/'checkpoints/history/checkpoint_step_000220672_manifest.json')
runtime = read(RUN/'live_runtime_identity.json')
phases = collections.Counter(r['applied_audit']['phase_id'] for r in rows)
rear_owner_violations=[]
old_error=0.
rr_effect=collections.Counter()
for row in rows:
    request=row['policy_request']; audit=row['applied_audit']['actuator_target_effect_audit']
    assert request['rear_task_assists_enabled'] is False
    assert request['rr_capture_assist_observed_features'] == [0.]*14
    assert request['sampling_draws']==1 and request['extra_model_forwards']==request['extra_random_draws']==0
    assert request['selected_raw_full12']==row['raw_policy_action_full12']
    assert request['conditional_mean_full12']==row['old_distribution_mean_full12']
    assert request['effective_sigma_full12']==row['old_distribution_std_full12']
    assert request['selected_raw_log_probability']==row['old_log_probability']
    assert request['transformed_actuator_targets_are_not_policy_samples'] is True
    owned=audit.get('capture_assist_owned_channels_full12',[False]*12)
    if owned[6] or owned[7] or 'rr_capture_assist_evidence' in audit or 'rr_carry_wheel_evidence' in audit:
        rear_owner_violations.append(row['global_policy_decision'])
    assert audit['verified'] and audit['setter_dispatch_targets_equal'] and audit['actual_mapping_matches_dispatch']
    assert audit['phase_mask_full12']==[1]*12
    for index in (6,7):
        rr_effect[str(index)] += bool(audit['changed_channels_full12'][index])
    old_error=max(old_error,abs(logp(row['raw_policy_action_full12'],row['old_distribution_mean_full12'],
        row['old_distribution_std_full12'])-row['old_log_probability']))
    assert row['applied_audit']['no_in_episode_state_writes_verified'] is True

current_error=ratio_error=old_minibatch_error=0.
sample_visits=collections.Counter()
current_count=0
for batch in likelihood['minibatches']:
    assert batch['sigma_source']=='current_official_Gaussian_cache_after_parent_receiving_and_observed_rear_local_sigma'
    for j,indices in enumerate(batch['rollout_flat_indices']):
        # Ambiguous indices cannot be silently assigned to a different raw sample.
        assert len(indices)==1
        idx=indices[0]; sample_visits[idx]+=1; current_count+=1
        mean=batch['current_conditional_mean'][j]; sigma=batch['current_conditional_sigma'][j]
        old_lp=scalar(batch['old_log_probability'][j]); current_lp=scalar(batch['optimization_log_probability'][j])
        ratio=scalar(batch['ratio'][j])
        old_minibatch_error=max(old_minibatch_error,abs(old_lp-rows[idx]['old_log_probability']))
        current_error=max(current_error,abs(logp(rows[idx]['raw_policy_action_full12'],mean,sigma)-current_lp))
        ratio_error=max(ratio_error,abs(math.exp(current_lp-old_lp)-ratio))
assert not rear_owner_violations and old_error<3e-5 and current_error<3e-5 and ratio_error<3e-5
assert all(sample_visits[i]==5 for i in range(COUNT))
assert checkpoint['actor_parameter_sha256']==update['actor_parameter_sha256_after']
assert checkpoint['global_policy_decisions']==220672 and checkpoint['ppo_updates']==1689
assert checkpoint['optimizer_steps']==33780
changes=[dict(global_policy_decision=r['global_policy_decision'],request_phase=r['applied_audit']['phase_id'],
    end_phase=r['applied_audit']['end_phase_id'],terminal=r['terminal'],
    terminal_bootstrap_allowed=r['applied_audit']['terminal_bootstrap_allowed'])
    for r in rows if r['applied_audit']['phase_id']!=r['applied_audit']['end_phase_id']]
assert all(not r['terminal'] for r in rows) and all(not r['terminal'] for r in changes)
print(json.dumps(dict(
    schema='wlr50_clean.real_block_readonly_audit.v1',
    audited_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    scope='first completed 128-decision natural-P01 PPO update only; not live unfinished samples',
    run=str(RUN), control_head=runtime['source_git_commit'],
    analysis='stdlib JSON arithmetic only; no model forward, no simulation, no CUDA/Torch, no file mutation',
    initial_publication={k:publication[k] for k in ('global_policy_decisions','ppo_updates','optimizer_steps')},
    sealed_checkpoint={k:checkpoint[k] for k in ('global_policy_decisions','ppo_updates','optimizer_steps','actor_parameter_sha256')},
    new_completed=dict(policy_decisions=128,ppo_updates=1,optimizer_steps=20,auxiliary_updates=0),
    decision_range=[FIRST_DECISION,FIRST_DECISION+COUNT-1], actual_request_phase_samples=dict(phases),
    rear_phase_samples={p:phases[p] for p in ('P07','P08','P09','P10','P11','P12','P13')},
    policy_request_checks=dict(count=COUNT,rear_task_assists_enabled_false=COUNT,RR_WAIT14_zero=COUNT,
        one_Gaussian_draw=COUNT,no_extra_forward_or_draw=COUNT,all12_mask_one=COUNT,
        original_raw_mean_sigma_old_logp_match=COUNT,transformed_targets_not_policy_samples=COUNT),
    actuator_checks=dict(final_dispatch_verified=COUNT,rear_assist_owner_violations=rear_owner_violations,
        last_tick_RR_channel_native_effect_decisions=dict(rr_effect),
        evidence_limit='request phase P01/P02 only; no claim of RR-capture/RL-phase physical coverage'),
    likelihood_checks=dict(minibatches=len(likelihood['minibatches']),sample_visits=current_count,
        each_rollout_sample_visits=5,old_raw_Normal_max_abs_error=old_error,
        current_raw_Normal_max_abs_error=current_error,minibatch_old_logp_exact_max_abs_error=old_minibatch_error,
        exp_current_minus_old_ratio_max_abs_error=ratio_error,float32_vs_float64_tolerance=3e-5,
        official_no_extra_model_forwards=likelihood['extra_model_forwards'],
        official_no_extra_random_draws=likelihood['extra_random_draws']),
    ordinary_phase_changes=changes, terminal_samples=0,
    tail_bootstrap=advantages['tail_bootstrap'], teacher_prefix_samples_included=advantages['teacher_prefix_samples_included'],
    optimizer_evidence=update, result='PASS_WITH_EXPLICIT_REAR_PHASE_COVERAGE_NOT_YET_REACHED'),ensure_ascii=False,indent=2))

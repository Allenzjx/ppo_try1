"""One sealed real batch only; CPU tensors/stdout, no checkpoint/runtime writes."""
import hashlib
import itertools
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T0814225788964Z_ga802b24d78df_790387038d25495ea5f001f934d3ade2'
CPDIR = ROOT / 'outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history'
PHASES = [f'P{i:02}' for i in range(1, 14)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(name):
    with (RUN / name).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def state_hash(value):
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                digest.update(repr(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def param_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode()); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


source_path = CPDIR / 'checkpoint_step_000205824.pt'
target_path = CPDIR / 'checkpoint_step_000205952.pt'
source = torch.load(source_path, map_location='cpu', weights_only=False)
target = torch.load(target_path, map_location='cpu', weights_only=False)
sm = read(source_path.with_name(source_path.stem + '_manifest.json'))
tm = read(target_path.with_name(target_path.stem + '_manifest.json'))
assert sha(source_path) == sm['checkpoint_sha256']
assert sha(target_path) == tm['checkpoint_sha256']
for cp, meta in [(source, sm), (target, tm)]:
    assert param_hash(cp['actor_state_dict']) == meta['actor_parameter_sha256']
    assert param_hash(cp['critic_state_dict']) == meta['critic_parameter_sha256']
    assert state_hash(cp['optimizer_state_dict']) == meta['optimizer_state_sha256']
    assert all(cp['infos'][k] == value for k, value in meta.items() if k in cp['infos'])
assert tm['source_run'] == str(RUN) and tm['save_load_round_trip'] is True
assert [tm[k]-sm[k] for k in ['global_policy_decisions','ppo_updates','optimizer_steps']] == [128,1,20]
assert [tm[k] for k in ['global_policy_decisions','ppo_updates','optimizer_steps']] == [205952,1574,31480]
unchanged = ['p05_capture_assist_branch', 'p05_capture_assist_migration',
    'capture_feedback_semantics_branch','capture_feedback_semantics_migration',
    'task_conditioned_hip_wheel_branch','normalizer_state_sha256','normalization',
    'runner_config','policy_contract','runtime_contract']
assert all(tm[k] == sm[k] for k in unchanged)
assert tm['optimizer_learning_rate'] == 1e-5
assert tm['actor_parameter_sha256'] != sm['actor_parameter_sha256']
assert tm['optimizer_state_sha256'] != sm['optimizer_state_sha256']
step_delta = [float(target['optimizer_state_dict']['state'][k]['step']-value['step'])
              for k,value in source['optimizer_state_dict']['state'].items()]
assert len(step_delta) == 12 and set(step_delta) == {20.}
update = next(lines('optimizer_updates.jsonl'))
assert update['global_policy_decisions'] == 205952 and update['ppo_update'] == 1574
assert update['actor_parameter_sha256_before'] == sm['actor_parameter_sha256']
assert update['actor_parameter_sha256_after'] == tm['actor_parameter_sha256']
assert update['optimizer_steps'] == 20 and update['finite_nonzero_gradient_observed']

prefix_phases, prefix_counts = Counter(), Counter()
previous_native = None
for prefix in lines('prefix_evidence.jsonl'):
    assert prefix['policy_credit'] is False
    if prefix['kind'] == 'checkpoint_prefix_decision':
        prefix_phases[prefix['phase_id']] += 1
        prefix_counts['decisions'] += 1; prefix_counts['physics_ticks'] += prefix['physics_ticks']
        assert prefix['actuator_target_effect_audit_summary']['all_ticks_verified']
        assert prefix['no_in_episode_state_writes_verified']
        previous_native = prefix['actuator_target_effect_audit']
    elif prefix['kind'] == 'checkpoint_prefix_result':
        assert prefix['accepted'] is True
    elif prefix['kind'] == 'policy_credit_start':
        start = prefix['start']; break
assert prefix_counts == {'decisions':562,'physics_ticks':4496}
assert start['actual_phase'] == 'P06' and start['physics_tick'] == 4496
provenance = start['prefix_policy_provenance']
assert provenance['checkpoint_sha256'] == sm['checkpoint_sha256']
assert provenance['source_global_policy_decisions'] == 205824
assert provenance['frozen_actor_parameter_sha256'] == sm['actor_parameter_sha256']
assert provenance['frozen_for_entire_training_block']
assert provenance['independent_parameter_and_buffer_storage_verified']
assert provenance['effective_runtime_content_sha256'] == provenance['source_runtime_content_sha256']
assert provenance['source_policy_contract'] == provenance['effective_policy_contract'] == tm['policy_contract']

data = torch.load(RUN/'rollouts/rollout_001574.pt', map_location='cpu', weights_only=False)
obs = data['observations']['policy']; actions = data['actions']; means,stds = data['distribution_params']
assert obs.shape == (128,1,389) and actions.shape == (128,1,12)
assert torch.equal(obs,data['observations']['critic'])
assert data['runtime_contract'] == tm['runtime_contract']
assert data['policy_contract'] == tm['policy_contract']
assert data['curriculum_epoch']['prefix_request']['target_phase'] == 'P06'
assert data['curriculum_epoch']['prefix_policy_provenance']['checkpoint_sha256'] == sm['checkpoint_sha256']
rows = list(itertools.islice(lines('residual_and_projection_audit.jsonl'),128))
counts, requested, endpoints, modes = Counter(), Counter(), Counter(), Counter()
from wlr50_clean.ppo.semantic_capture_assist import capture_assist_features
for index,row in enumerate(rows):
    a=row['applied_audit']; native=a['actuator_target_effect_audit']; receipt=native['capture_assist_evidence']
    assert row['global_policy_decision'] == 205825+index
    assert a['decision_count'] == index+1 and not row['terminal']
    assert a['physical_core_decision_count_including_prefix'] == 563+index
    assert not a['prefix_teacher_data_in_ppo_storage'] and not a['prefix_checkpoint_policy_data_in_ppo_storage']
    assert a['task_result_scope'] == 'checkpoint_policy_initialized_suffix'
    assert row['raw_policy_action_full12'] == row['policy_request']['selected_raw_full12'] == native['raw_policy_action_full12']
    assert row['policy_request']['sampling_draws'] == 1 and row['policy_request']['extra_random_draws'] == 0
    assert row['old_distribution_mean_full12'] == row['policy_request']['conditional_mean_full12']
    assert row['old_distribution_std_full12'] == row['policy_request']['effective_sigma_full12']
    assert native['phase_mask_full12'] == [1]*12
    assert native['verified'] and native['actual_mapping_matches_dispatch']
    assert native['capture_assist_state_transition_independently_reconstructed']
    assert a['actuator_target_effect_audit_summary']['all_ticks_verified']
    assert a['no_in_episode_state_writes_verified']
    assert receipt['owner_indices'] in ([],[0,1])
    assert not receipt['owner_indices'] or receipt['knee_hold_final_verified']
    assert receipt['candidate_before_assist_full12'][2:] == receipt['candidate_after_assist_full12'][2:]
    before = previous_native['capture_assist_evidence']['state_after']
    assert torch.equal(obs[index,0,372:384],torch.tensor(capture_assist_features(before),dtype=obs.dtype))
    assert receipt['state_after']['feedback_revision'] == 'hold_to_air_progress_window_v2'
    previous_native = native
    requested[a['phase_id']] += 1; endpoints[a['end_phase_id']] += 1
    counts['physics_ticks'] += a['physics_ticks']; counts['assist_owned_endpoints'] += bool(receipt['owner_indices'])
    counts['all12_unmodified_endpoints'] += bool(native['all12_policy_channels_unmodified_at_actuator'])
    modes[round(float(obs[index,0,372])*5)] += 1
    rr = a['semantic_task']['physical_evaluator']['current_legs']['RR']
    counts['RR_current_TOP_endpoints'] += bool(rr['top_contact'])
for tensor,key in [(actions,'raw_policy_action_full12'),(means,'old_distribution_mean_full12'),
                   (stds,'old_distribution_std_full12'),(data['actions_log_prob'],'old_log_probability'),
                   (data['values'],'old_value'),(data['rewards'],'reward'),(data['dones'],'terminal')]:
    assert torch.equal(tensor,torch.tensor([r[key] for r in rows],dtype=tensor.dtype).reshape(tensor.shape))
assert [PHASES[i] for i in obs[:,0,:13].argmax(-1)] == [r['applied_audit']['phase_id'] for r in rows]
assert torch.equal(obs[:,0,372:384],torch.tensor([r['policy_request']['capture_assist_observed_features'] for r in rows],dtype=obs.dtype))
assert torch.equal(obs[:,0,384:389],torch.tensor([r['policy_request']['capture_continuation_observed_features'] for r in rows],dtype=obs.dtype))
likelihood=read(RUN/'rollouts/update_001574_likelihood.json')
use=Counter(i for batch in likelihood['minibatches'] for ids in batch['rollout_flat_indices'] for i in ids)
assert len(likelihood['minibatches']) == 20 and use == Counter({i:5 for i in range(128)})
logp_error=float((torch.distributions.Normal(means,stds).log_prob(actions).sum(-1).unsqueeze(-1)-data['actions_log_prob']).abs().max())
for key,idx in [('RR_qualified_inputs',149),('RR_crossed_inputs',153),('RR_placed_inputs',157),
                ('FL_pending_inputs',384),('pending_advanced_inputs',386)]:
    counts[key]=int((obs[:,0,idx]==1).sum())
print(json.dumps(dict(real_checkpoint=str(target_path),checkpoint_sha256=tm['checkpoint_sha256'],
    source_checkpoint_sha256=sm['checkpoint_sha256'],counts=[205952,1574,31480],new_counts=[128,1,20],
    P05_counts=tm['p05_capture_assist_branch_counts'],feedback_v2_counts=tm['capture_feedback_semantics_branch_counts'],
    prefix_counts=dict(prefix_counts),prefix_phases=dict(prefix_phases),credit_start_s=start['sim_time_s'],
    requested_phases=dict(requested),endpoint_phases=dict(endpoints),observed_assist_modes=dict(modes),
    actual_execution_counts=dict(counts),sample_global_range=[205825,205952],
    end_physics_tick=rows[-1]['applied_audit']['physics_tick'],end_sim_time_s=rows[-1]['applied_audit']['sim_time_s'],
    independent_CPU_logp_max_error=logp_error,each_sample_optimized_five_times=True,
    actual_save_reload=True,actor_and_Adam_changed=True,all12_Adam_step_deltas=step_delta,
    Identity_normalizer_and_runner_config_preserved=True,LR=tm['optimizer_learning_rate'],
    AUX_unchanged={k:tm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning'][k]
                   for k in ['accepted_auxiliary_updates_total','attempted_auxiliary_optimizer_steps_total']},
    runtime=tm['runtime_contract']['source_git_commit'],stage=tm['stage'],
    implemented_reset_sampling=tm['implemented_reset_sampling'],all_assertions_passed=True),indent=2))

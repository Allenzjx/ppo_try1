"""First sealed real PPO batch after actual AUX event2; CPU read-only."""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import torch
from wlr50_clean.ppo.semantic_training import state_hash

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1156115839619Z_g5fd88852bf20_a9df5b5d15aa4500a30f12933b1c36fb'
SOURCE = OUT / 'checkpoints/history/checkpoint_aux_detP02_step_000211968_v2.pt'
TARGET = OUT / 'checkpoints/history/checkpoint_step_000212096.pt'
SOURCE_SHA = '27bc7cbdd98775c4602057d7776dd3becc9eef9b42ec62f4691d2dae7bb06d55'
TARGET_SHA = '11a04b69ed31d824ba6d81491a187e438315cd23a073b4b45de93352f9904f95'
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def parameter_sha(state):
    digest=hashlib.sha256()
    for key,tensor in sorted(state.items()):
        tensor=tensor.detach().cpu().contiguous()
        digest.update(key.encode());digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def prefix(path, count):
    with Path(path).open(encoding='utf-8') as stream:
        result=[json.loads(line) for line in itertools.islice(stream,count)]
    assert len(result)==count
    return result


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    sm,tm=(read(path.with_name(path.stem+'_manifest.json')) for path in (SOURCE,TARGET))
    assert sha(SOURCE)==sm['checkpoint_sha256']==SOURCE_SHA
    assert sha(TARGET)==tm['checkpoint_sha256']==TARGET_SHA
    assert tm['source_run']==str(RUN) and tm['save_load_round_trip']
    assert {key:tm[key]-sm[key] for key in COUNTERS}==dict(global_policy_decisions=128,ppo_updates=1,optimizer_steps=20)
    assert {key:tm[key] for key in COUNTERS}==dict(global_policy_decisions=212096,ppo_updates=1622,optimizer_steps=32440)
    preserved=[key for key in sm if key!='resume_migration' and key.endswith(('_branch','_migration'))]
    assert all(tm[key]==sm[key] for key in preserved)
    for key in ('runtime_contract','policy_contract','runner_config','normalization','normalizer_state_sha256'):
        assert sm[key]==tm[key]
    ledger=tm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert len(ledger['events'])==2 and [event['event_index'] for event in ledger['events']]==[1,2]
    assert ledger['accepted_auxiliary_updates_total']==ledger['attempted_auxiliary_optimizer_steps_total']==64
    assert ledger==sm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert ledger['events'][1]['source_checkpoint']['sha256']=='5d0658331204c2bc79d71fd223582637e68dbb986ce4bf57596648f02c77d881'
    old_aux=tm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert old_aux['accepted_auxiliary_updates_total']==7 and old_aux['attempted_auxiliary_optimizer_steps_total']==8
    source,target=(torch.load(path,map_location='cpu',weights_only=False) for path in (SOURCE,TARGET))
    for payload,metadata in ((source,sm),(target,tm)):
        assert parameter_sha(payload['actor_state_dict'])==metadata['actor_parameter_sha256']
        assert parameter_sha(payload['critic_state_dict'])==metadata['critic_parameter_sha256']
        assert state_hash(payload['optimizer_state_dict'])==metadata['optimizer_state_sha256']
        assert all(payload['infos'][key]==value for key,value in metadata.items() if key in payload['infos'])
    assert sm['actor_parameter_sha256']!=tm['actor_parameter_sha256']
    assert sm['optimizer_state_sha256']!=tm['optimizer_state_sha256']
    deltas=[float(target['optimizer_state_dict']['state'][key]['step']-value['step'])
            for key,value in source['optimizer_state_dict']['state'].items()]
    assert len(deltas)==12 and deltas==[20.]*12
    assert tm['optimizer_learning_rate']==1e-5
    batch=torch.load(RUN/'rollouts/rollout_001622.pt',map_location='cpu',weights_only=False)
    rows=prefix(RUN/'residual_and_projection_audit.jsonl',128)
    update=prefix(RUN/'optimizer_updates.jsonl',1)[0]
    assert update['ppo_update']==1622 and update['optimizer_steps']==20 and update['actor_parameters_changed']
    assert update['actor_parameter_sha256_before']==sm['actor_parameter_sha256']
    assert update['actor_parameter_sha256_after']==tm['actor_parameter_sha256']
    assert update['finite_nonzero_gradient_observed']
    obs=batch['observations']['policy'];actions=batch['actions'];means,stds=batch['distribution_params']
    assert obs.shape==(128,1,389) and actions.shape==(128,1,12)
    assert torch.equal(obs,batch['observations']['critic'])
    assert batch['runtime_contract']==tm['runtime_contract'] and batch['policy_contract']==tm['policy_contract']
    assert batch['curriculum_epoch']['prefix_request'] is None
    assert not (RUN/'prefix_evidence.jsonl').exists()
    phase_counts=Counter();physics_ticks=terminal_count=assist_endpoints=0
    for index,row in enumerate(rows):
        assert row['global_policy_decision']==211969+index
        a=row['applied_audit'];p=row['policy_request'];n=a['actuator_target_effect_audit']
        assert p['sampling_draws']==1 and p['extra_random_draws']==0
        assert row['raw_policy_action_full12']==p['selected_raw_full12']==n['raw_policy_action_full12']
        assert row['old_distribution_mean_full12']==p['conditional_mean_full12']
        assert row['old_distribution_std_full12']==p['effective_sigma_full12']
        assert n['verified'] and n['actual_mapping_matches_dispatch'] and n['phase_mask_full12']==[1]*12
        assert a['actuator_target_effect_audit_summary']['all_ticks_verified']
        phase_counts[a['phase_id']]+=1;physics_ticks+=a['physics_ticks'];terminal_count+=bool(row['terminal'])
        assist_endpoints+=bool(n['capture_assist_evidence']['owner_indices'])
    for tensor,key in ((actions,'raw_policy_action_full12'),(means,'old_distribution_mean_full12'),
        (stds,'old_distribution_std_full12'),(batch['actions_log_prob'],'old_log_probability'),
        (batch['rewards'],'reward'),(batch['dones'],'terminal')):
        assert torch.equal(tensor,torch.tensor([row[key] for row in rows],dtype=tensor.dtype).reshape(tensor.shape))
    assert [f'P{int(index)+1:02}' for index in obs[:,0,:13].argmax(-1)]==[row['applied_audit']['phase_id'] for row in rows]
    error=float((torch.distributions.Normal(means,stds).log_prob(actions).sum(-1).unsqueeze(-1)-batch['actions_log_prob']).abs().max())
    likelihood=read(RUN/'rollouts/update_001622_likelihood.json')
    uses=Counter(i for minibatch in likelihood['minibatches'] for indices in minibatch['rollout_flat_indices'] for i in indices)
    assert len(likelihood['minibatches'])==20 and uses==Counter({i:5 for i in range(128)})
    result={'schema':'wlr50_clean.block07_first_actual_event2_PPO_carry_audit.v1','result':'PASS',
        'run':str(RUN),'source_checkpoint':str(SOURCE),'source_sha256':SOURCE_SHA,'checkpoint':str(TARGET),'checkpoint_sha256':TARGET_SHA,
        'actual_counts':{key:tm[key] for key in COUNTERS},'new_counts':{'policy_decisions':128,'ppo_updates':1,'Adam_steps':20},
        'complete_event1_and_event2_ledger_equal_to_actual_source':True,'front_AUX_total':[64,64],
        'historical_AUX_7_8_and_all_origins_migrations_preserved':True,'preserved_lineage_fields':preserved,
        'new_AUX_added':0,'actor_and_Adam_actually_changed':True,'Adam_step_deltas':deltas,'LR':1e-5,
        'Identity_and_runner_runtime_policy_preserved':True,'actual_save_reload_recorded_true':True,
        'actual_source_parameter_and_Adam_hashes_match':True,'first128_fresh_raw_mean_std_logp_reward_done_exact':True,
        'observation_shape':list(obs.shape),'raw_action_shape':list(actions.shape),'each_sample_PPO_uses':5,
        'CPU_Normal_logp_max_error':error,'request_phase_counts':{f'P{i:02}':phase_counts[f'P{i:02}'] for i in range(1,14)},
        'physics_ticks':physics_ticks,'terminal_samples':terminal_count,'assist_owned_endpoints':assist_endpoints,
        'no_prefix':True,'physical_success_claimed':False,'scope':'first sealed128 decisions only; not the still-running block outcome'}
    (OUT/'block07_first_real_event2_PPO_carry_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (OUT/'block07_first_real_event2_PPO_carry_audit.md').write_text(f'''# First genuine PPO carry after actual AUX event2

**PASS.** Source actual `checkpoint_aux_detP02_step_000211968_v2.pt` → sealed ordinary PPO `checkpoint_step_000212096.pt` (SHA256 `{TARGET_SHA}`). Added128 decisions/1 PPO/20 Adam; cumulative212096/1622/32440. This is real training, not a synthetic continuation.

The entire two-event front AUX ledger is equal to source, including source/data/helper/admission/inspection/fit bindings and64/64 totals. Event1 remains intact; earlier limited AUX7/8, all3 origins and historical migration records remain unchanged. This PPO batch adds0 AUX.

Actor and Adam actually changed; all12 Adam states advanced20. LR1e-5, Identity, runner, runtime and policy contract preserved. Actual checkpoint parameter/Adam hashes and embedded/sidecar values pass independent CPU verification; normal save/reload is recorded true.

Fresh storage128×1×389 and raw12/μ/σ/logp/reward/done exactly match synchronous logs; each original sample used5 times in20 minibatches. Recomputed CPU Normal logp maximum difference {error:.9g}. Natural P01/no prefix; input counts {dict(phase_counts)}, physics ticks{physics_ticks}, terminal samples{terminal_count}, assist-owned endpoints{assist_endpoints}; detailed endpoint masks all1/native checks verified.

This closes the previously unverified **first actual ordinary PPO propagation of event2**. It does not establish physical success or summarize the running block. Audit only reads the first sealed batch and actual source/target checkpoints. No simulator, GPU, fit, production edit or checkpoint write; CPU helper exits after report creation.
''',encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('result','actual_counts','new_counts','front_AUX_total',
        'CPU_Normal_logp_max_error','request_phase_counts','physics_ticks','terminal_samples','assist_owned_endpoints')},indent=2))


if __name__=='__main__':
    main()

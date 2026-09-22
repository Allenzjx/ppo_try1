"""Bounded read-only audit: first128 real samples/update1418/CP185984 only."""
from pathlib import Path
from collections import Counter
from itertools import islice
import hashlib
import json
import math
import sys
import torch

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wlr50_clean.ppo.semantic_history_actor import cap_transition_request_history,task_conditioned_effective_log_std
from wlr50_clean.ppo.semantic_training import state_hash

RUN=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T0459512815083Z_gee5a9651591d_8e120f68dc7c4598b08dc79fcd3d34a4'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prefix(path,count):
    with path.open('rb') as f:content=b''.join(islice(f,count))
    rows=[json.loads(line) for line in content.splitlines()]
    assert len(rows)==count
    return rows,hashlib.sha256(content).hexdigest()


def main():
    torch.set_num_threads(1);rng=torch.get_rng_state().clone()
    rows,row_sha=prefix(RUN/'residual_and_projection_audit.jsonl',128)
    updates,update_sha=prefix(RUN/'optimizer_updates.jsonl',1);update=updates[0]
    assert [r['global_policy_decision'] for r in rows]==list(range(185857,185985))
    assert update['ppo_update']==1418 and update['global_policy_decisions']==185984 and update['optimizer_steps']==20
    sp=RUN/'rollouts/rollout_001418.pt';hp=RUN/'rollouts/update_001418_likelihood.json'
    saved=torch.load(sp,map_location='cpu',weights_only=False);hooks=json.loads(hp.read_text())
    assert len(hooks['minibatches'])==20 and hooks['extra_model_forwards']==hooks['extra_random_draws']==0
    obs=saved['observations']['policy'][:,0];history,_=cap_transition_request_history(obs)
    assert obs.shape==(128,372)
    for rk,sk in [('raw_policy_action_full12','actions'),('old_value','values'),('reward','rewards'),('old_log_probability','actions_log_prob')]:
        assert torch.equal(torch.tensor([r[rk] for r in rows]).reshape_as(saved[sk][:,0]),saved[sk][:,0])
    for k,name in enumerate(('old_distribution_mean_full12','old_distribution_std_full12')):
        assert torch.equal(torch.tensor([r[name] for r in rows]),saved['distribution_params'][k][:,0])
    fullmask=[];firstFR=None;request_sigma_error=0.
    for row in rows:
        q=row['policy_request'];e=row['applied_audit']['actuator_target_effect_audit']
        assert q['policy_version']=='task_conditioned_hip_wheel_sigma_v1'
        assert q['sampling_draws']==1 and q['extra_model_forwards']==q['extra_random_draws']==0
        assert e['phase_mask_full12']==[1]*12 and e['verified'] and e['setter_dispatch_targets_equal'] and e['actual_mapping_matches_dispatch']
        assert all(v>0 and math.isfinite(v) for v in q['effective_sigma_full12'])
        request_sigma_error=max(request_sigma_error,max(abs(.25*s*b/c-eff) for s,b,c,eff in zip(q['learned_sigma_full12'],q['physical_equivalent_B_full12'],q['current_cap_full12'],q['effective_sigma_full12'])))
        if firstFR is None and q['task_state_weights']['FR_air_approach_proxy']==1:firstFR=row
    uses=Counter();mean_error=sigma_error=logp_error=0.;mean_grad_max=torch.zeros(12);logsigma_grad_max=torch.zeros(12)
    mean_grad_nonzero=torch.zeros(12,dtype=torch.int64);logsigma_grad_nonzero=torch.zeros(12,dtype=torch.int64)
    norm_rows=[];gradient_semantics=set();sigma_sources=set();actor_norms=[];critic_norms=[]
    for mb in hooks['minibatches']:
        indices=[v[0] for v in mb['rollout_flat_indices']];assert all(len(v)==1 for v in mb['rollout_flat_indices']);uses.update(indices)
        headmu=torch.tensor(mb['current_network_mean_full12']);headlog=torch.tensor(mb['current_network_log_sigma_full12'])
        cond=torch.tensor(mb['current_conditional_mean']);sigma=torch.tensor(mb['current_conditional_sigma'])
        gm=torch.tensor(mb['loss_gradient_wrt_network_mean_full12']);gs=torch.tensor(mb['loss_gradient_wrt_network_log_sigma_full12'])
        rn=torch.tensor(mb['actor_head_postclip_parameter_gradient_norm_by_row'])
        for value in (headmu,headlog,cond,sigma,gm,gs,rn):assert bool(torch.isfinite(value).all())
        assert cond.shape==sigma.shape==gm.shape==gs.shape==(32,12) and rn.shape==(24,) and bool((sigma>0).all())
        assert mb['head_measurement']=='same_official_minibatch_forward_before_HISTORY_and_sigma_schedule'
        log,_=task_conditioned_effective_log_std(headlog,obs[indices],.25)
        mean_error=max(mean_error,float((cond-(.1*headmu+.9*history[indices])).abs().max()))
        sigma_error=max(sigma_error,float((sigma-log.exp()).abs().max()))
        oldlp=saved['actions_log_prob'][indices,0,0];adv=saved['advantages'][indices,0,0]
        assert torch.equal(oldlp,torch.tensor(mb['old_log_probability'])) and torch.equal(adv,torch.tensor(mb['actual_advantage']))
        action=saved['actions'][indices,0].double()
        rebuilt=(-sigma.double().log()-.5*((action-cond.double())/sigma.double()).square()-.5*math.log(2*math.pi)).sum(-1)
        logp_error=max(logp_error,float((rebuilt-torch.tensor(mb['optimization_log_probability']).double()).abs().max()))
        strict=((adv>0)&(torch.tensor(mb['ratio'])>1.2))|((adv<0)&(torch.tensor(mb['ratio'])<.8))
        assert strict.tolist()==mb['clipped_branch_strictly_active']
        mean_grad_max=torch.maximum(mean_grad_max,gm.abs().max(0).values);logsigma_grad_max=torch.maximum(logsigma_grad_max,gs.abs().max(0).values)
        mean_grad_nonzero+=(gm!=0).sum(0);logsigma_grad_nonzero+=(gs!=0).sum(0);norm_rows.append(rn)
        gradient_semantics.add(mb['gradient_semantics']);sigma_sources.add(mb['sigma_source'])
        actor_norms.append(mb['separate_postclip_parameter_gradient_norms']['actor']);critic_norms.append(mb['separate_postclip_parameter_gradient_norms']['critic'])
    assert uses==Counter({i:5 for i in range(128)}) and mean_error<1e-6 and sigma_error<1e-7 and logp_error<1e-5
    cp=OUT/'checkpoints/history/checkpoint_step_000185984.pt';side=cp.with_name(cp.stem+'_manifest.json');metadata=json.loads(side.read_text())
    target=torch.load(cp,map_location='cpu',weights_only=False);source_path=ROOT/'outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000185856.pt'
    source=torch.load(source_path,map_location='cpu',weights_only=False)
    assert sha(cp)==metadata['checkpoint_sha256'] and metadata['save_load_round_trip'] is True
    assert all(metadata[k]==v for k,v in target['infos'].items())
    assert target['infos']['task_conditioned_hip_wheel_branch_counts']==dict(global_policy_decisions=128,ppo_updates=1,optimizer_steps=20)
    assert target['infos']['normalizer_state_sha256']==source['infos']['normalizer_state_sha256']
    assert update['actor_parameter_sha256_before']==source['infos']['actor_parameter_sha256']
    assert update['actor_parameter_sha256_after']==target['infos']['actor_parameter_sha256']
    so=source['optimizer_state_dict'];to=target['optimizer_state_dict']
    assert state_hash(so)==source['infos']['optimizer_state_sha256'] and state_hash(to)==target['infos']['optimizer_state_sha256']
    assert so['param_groups']==to['param_groups']
    assert all(g['lr']==1e-5 for g in to['param_groups']) and target['infos']['optimizer_learning_rate']==source['infos']['optimizer_learning_rate']==1e-5
    assert so['state'].keys()==to['state'].keys();adam=[]
    for key in so['state']:
        before=so['state'][key];after=to['state'][key];assert before.keys()==after.keys()
        assert after['step'].item()-before['step'].item()==20
        for name in ('exp_avg','exp_avg_sq'):
            assert after[name].shape==before[name].shape and bool(torch.isfinite(after[name]).all())
        adam.append(dict(parameter_id=key,source_step=before['step'].item(),target_step=after['step'].item(),
            exp_avg_shape=list(after['exp_avg'].shape),exp_avg_finite=True,exp_avg_sq_finite=True))
    assert torch.equal(rng,torch.get_rng_state())
    e=firstFR['applied_audit']['actuator_target_effect_audit'];q=firstFR['policy_request']
    result=dict(schema='wlr50_clean.first_real_task_conditioned_update_audit.v1',run=str(RUN),
        scope='Only completed first128 collection rows/update1418/CP185984; does not seal or stop the still-running parent run.',
        actual_completed_new_branch=dict(decisions=128,updates=1,optimizer_steps=20),new_training_added_by_audit=0,
        prefix_hashes=dict(first128_decision_rows=row_sha,first_update_line=update_sha),sealed_artifact_sha256={str(p):sha(p) for p in (sp,hp,side)},
        phase_counts=dict(Counter(r['applied_audit']['phase_id'] for r in rows)),update=update,
        checks=dict(collection_tensor_log_exact=True,all_128_full12_mask_ones_and_native_dispatch_verified=True,all12_positive_sigma=True,
            request_sigma_formula_max_error=request_sigma_error,head_to_conditional_mean_max_error=mean_error,
            actual_sigma_kernel_max_error=sigma_error,Gaussian_logp_rebuild_max_error=logp_error,
            all_samples_five_actual_optimizer_uses=True,old_logp_and_advantage_exact=True,strict_clip_flags_exact=True,
            one_real_head_per_actual_minibatch=True,all_head_and_gradient_arrays_finite=True,RNG_unchanged_by_offline_audit=True),
        gradient=dict(source=sorted(gradient_semantics),sigma_sources=sorted(sigma_sources),
            mean_derivative_max_abs_full12=mean_grad_max.tolist(),logsigma_derivative_max_abs_full12=logsigma_grad_max.tolist(),
            mean_derivative_nonzero_sample_uses_full12=mean_grad_nonzero.tolist(),logsigma_derivative_nonzero_sample_uses_full12=logsigma_grad_nonzero.tolist(),
            head_parameter_row_norm_min=float(torch.stack(norm_rows).min()),head_parameter_row_norm_max=float(torch.stack(norm_rows).max()),
            actor_postclip_norm_min=min(actor_norms),actor_postclip_norm_max=max(actor_norms),
            critic_postclip_norm_min=min(critic_norms),critic_postclip_norm_max=max(critic_norms)),
        checkpoint=dict(path=str(cp),roundtrip=True,hash_verified=True,branch_counts=metadata['task_conditioned_hip_wheel_branch_counts'],
            full_Adam_state_entry_count=len(adam),param_groups_unchanged=True,effective_learning_rate=1e-5,Adam_entries=adam,
            source_and_target_optimizer_hashes_verified=True,Identity_normalizer_hash_unchanged=True,
            migration_evidence='Production loader lines877–890 hash-validates loaded source weights/full optimizer/normalizer before any sampling; source/target saved Adam state entry identities/shapes match and all step counters advance20. Post-update moments must change; no optimizer replay performed.'),
        first_FR_state_hit=dict(global_decision=firstFR['global_policy_decision'],B=q['physical_equivalent_B_full12'],
            sigma_multiplier=q['innovation_sigma_multiplier_full12'],wheel_nominal=firstFR['applied_audit']['nominal_action_full12'][8:],
            wheel_baseline=e['policy_headroom_evidence']['baseline_native_plus_controller_full12'][8:],
            wheel_requested=e['policy_headroom_evidence']['requested_policy_residual_full12'][8:],
            wheel_final=firstFR['applied_audit']['actual_drive_target_full12'][8:],native_wheel_target=e['actual_native_targets']['wheel_velocity_rad_s'],
            scope='actual sampled target/dispatch, not proof of measured wheel angular velocity or whole-task success'))
    with (OUT/'first_real_update1418_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('actual_completed_new_branch','phase_counts','checks','gradient')},indent=2))
    print('Adam state entries',len(adam),'all step +20; scalar/group LR 1e-5; CP185984 roundtrip/hash verified.')


if __name__=='__main__':main()

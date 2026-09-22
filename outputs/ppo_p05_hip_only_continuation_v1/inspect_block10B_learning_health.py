"""Bounded sealed-block CPU read-only PPO health review; no optimizer or physics."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
HELPER=OUT/'rr_phase_column_readonly_v5/inspect_rr_phase_column.py'
HELPER_SHA='24a8ee1a670e063e38d5877217dac9f0ba73f7be92702fb0a87c2db38786877a'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==HELPER_SHA
spec=importlib.util.spec_from_file_location('_block10B_frozen_distribution',HELPER)
h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
torch,training=h.torch,h.training
import numpy as np

RUN=ROOT/'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1727064438736Z_g6ac7b553d792_4ae7abef7e2b4f91b5656e48cd35f791'
CPDIR=OUT/'checkpoints/history'
SOURCES={216960:'33c8a37e670513273e4b072bb41845121eacab03cbcd4efa18dbf7edb6a254d0',
         218496:'6f7c572aaa81ea21407a4aac13809967885cfb9119014a78151b4bf0eff9e227'}
DEST=OUT/'block10B_learning_health_readonly.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def rows(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:yield json.loads(line)
def ar(value):return value.detach().cpu().numpy().astype(float) if torch.is_tensor(value) else np.asarray(value,dtype=float)
def stats(values):
    x=ar(values).reshape(-1);assert len(x) and np.isfinite(x).all()
    return dict(n=len(x),mean=float(x.mean()),min=float(x.min()),max=float(x.max()),
        mean_abs=float(np.abs(x).mean()),max_abs=float(np.abs(x).max()),std_population=float(x.std()))
def channel_stats(values):return {name:stats(ar(values)[:,i]) for i,name in enumerate(h.kernel.CHANNELS)}
def channel_fraction(values):return {name:float(np.asarray(values)[:,i].mean()) for i,name in enumerate(h.kernel.CHANNELS)}


def actor_copy(cp,x):
    model=h.kernel.SemanticP05CaptureHistoryMLPModel(h.kernel.tensors(x[:1]),{'actor':['policy'],'critic':['critic']},
        'actor',12,hidden_dims=(256,256),activation='elu',obs_normalization=False,
        distribution_cfg={'class_name':'HeteroscedasticGaussianDistribution','std_type':'log','init_std':.15},
        observation_layout=h.P05_CAPTURE_OBSERVATION_LAYOUT,exploration_std_temperature=.25)
    model.load_state_dict(cp['actor_state_dict'],strict=True)
    return model.eval()


def main():
    assert not torch.cuda.is_available() and not DEST.exists()
    rng=training.capture_training_rng_state(seed=0);threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        audit=read(OUT/'block10B_training_audit.json')
        assert audit['result']=='PASS' and audit['actual_added_counts']['global_policy_decisions']==1536
        updates=list(rows(RUN/'optimizer_updates.jsonl'))
        assert len(updates)==12 and [r['ppo_update'] for r in updates]==list(range(1661,1673))
        for i,u in enumerate(updates):
            assert u['optimizer_steps']==20 and u['actor_parameters_changed'] and u['finite_nonzero_gradient_observed']
            if i:assert u['actor_parameter_sha256_before']==updates[i-1]['actor_parameter_sha256_after']
        stacked={name:[] for name in ('observations','raw','mean','sigma')};selected=None;roster=[]
        for update in updates:
            path=RUN/f"rollouts/rollout_{update['ppo_update']:06d}.pt"
            r=torch.load(path,map_location='cpu',weights_only=False)
            x=r['observations']['policy'][:,0];mask=x[:,8]==1
            assert x.shape==(128,389) and torch.equal(x,r['observations']['critic'][:,0])
            roster.append(dict(update=update['ppo_update'],P09_samples=int(mask.sum()),sha256=h.sha(path)))
            for name,t in (('observations',x),('raw',r['actions'][:,0]),
                ('mean',r['distribution_params'][0][:,0]),('sigma',r['distribution_params'][1][:,0])):
                stacked[name].append(t[mask])
            if selected is None and bool(mask.all()):
                selected=dict(update=update['ppo_update'],x=x.clone(),raw=r['actions'][:,0].clone(),
                    first_decision=update['global_policy_decisions']-127,last_decision=update['global_policy_decisions'])
        stacked={k:torch.cat(v,dim=0) for k,v in stacked.items()}
        assert len(stacked['raw'])==524 and selected is not None
        raw,mean,sigma=(ar(stacked[k]) for k in ('raw','mean','sigma'))
        assert np.isfinite(raw).all() and np.isfinite(mean).all() and np.isfinite(sigma).all() and (sigma>0).all()
        endpoints=[];selected_rows=[]
        for n,row in enumerate(rows(RUN/'residual_and_projection_audit.jsonl'),216961):
            assert row['global_policy_decision']==n
            if row['applied_audit']['phase_id']!='P09':continue
            a=row['applied_audit'];native=a['actuator_target_effect_audit'];headroom=native['policy_headroom_evidence']
            assert native['verified'] and a['actuator_target_effect_audit_summary']['all_ticks_verified']
            p=row['policy_request'];caps=np.asarray(p['current_cap_full12'],float)
            requested=caps*np.tanh(row['raw_policy_action_full12'])
            filtered=np.asarray(headroom['requested_policy_residual_full12'],float)
            effective=np.asarray(headroom['effective_policy_residual_full12'],float)
            candidate=np.asarray(headroom['candidate_native_target_before_final_slew_full12'],float)
            actual=np.asarray(native['native_drive_target_full12'],float)
            hard=np.asarray(headroom['servo_hard_limits_deg']+[[-2.1,2.1]]*4,float)
            tolerance=np.asarray([1e-4]*8+[1e-6]*4)
            clips=np.zeros(12,dtype=bool);clips[headroom['clipped_servo_indices']]=True
            assert np.array_equal(clips,np.abs(effective-filtered)>0)
            assert np.array_equal(np.asarray(native['phase_mask_full12']),np.ones(12))
            endpoints.append(dict(decision=n,headroom_clips=clips,
                endpoint_filter_difference=np.abs(requested-filtered)>tolerance,
                final_clamp_slew_or_cast_difference=np.abs(candidate-actual)>tolerance,
                final_hard_bound_near=np.any(np.abs(actual[:,None]-hard)<=tolerance[:,None],axis=1),
                native_channel_effect=np.asarray(native['changed_channels_full12'],bool),
                assist_owned=bool(native['capture_assist_evidence']['owner_indices']),
                RRgap=a['semantic_task']['physical_evaluator']['current_legs']['RR']['clearance_m'],
                RRplaced=a['semantic_task']['physical_evaluator']['history']['placed']['RR']))
            if selected['first_decision']<=n<=selected['last_decision']:
                selected_rows.append(row)
        assert len(endpoints)==524 and len(selected_rows)==128
        assert torch.equal(selected['raw'],torch.tensor([r['raw_policy_action_full12'] for r in selected_rows],dtype=torch.float32))
        source_binding={};before_after={};protect={}
        for count,expected_sha in SOURCES.items():
            path=CPDIR/f'checkpoint_step_{count:09d}.pt'
            assert h.sha(path)==expected_sha
            meta=h.semantic_migration.checkpoint_metadata(path)
            cp=torch.load(path,map_location='cpu',weights_only=False)
            assert cp['infos']=={k:v for k,v in meta.items() if k not in ('checkpoint_path','checkpoint_sha256','save_load_round_trip')}
            protect[count]=h.prior._protected_payload(cp)
            model=actor_copy(cp,selected['x']);model_hash=training.parameter_hash(model)
            assert model_hash==meta['actor_parameter_sha256']
            with torch.no_grad():
                d=h.distribution(model,h.kernel.tensors(selected['x']))
                assert torch.equal(d['mean'],model(h.kernel.tensors(selected['x']),stochastic_output=False))
            before_after[count]={k:v.detach() for k,v in d.items()}
            assert training.parameter_hash(model)==model_hash and h.prior._protected_payload(cp)==protect[count]
            assert h.sha(path)==expected_sha
            source_binding[count]=dict(path=str(path),sha256=expected_sha,actor_sha256=model_hash,
                runtime=meta['runtime_contract']['source_git_commit'],counts={k:meta[k] for k in ('global_policy_decisions','ppo_updates','optimizer_steps')})
        b,c=before_after[216960],before_after[218496]
        change=h.kernel.gaussian_change(b,c)
        p09_updates=[u for u,r in zip(updates,roster) if r['P09_samples']]
        def update_stats(items):
            return {k:stats([r[k] for r in items]) for k in ('optimizer_learning_rate','surrogate_loss','value_loss',
                'kl_mean','clip_fraction','entropy','gradient_norm_min','gradient_norm_max')}
        report=dict(schema='wlr50_clean.block10B_bounded_learning_health_readonly.v1',
            scope='12 sealed updates; all524 stored P09 raw distributions and last-tick execution evidence; one128-P09 input fixed endpoint comparison',
            source_binding=source_binding,run=str(RUN),helper_sha256=h.sha(__file__),
            reused_distribution_helper_sha256=HELPER_SHA,previous_audit_sha256=h.sha(OUT/'block10B_training_audit.json'),
            optimizer_updates_sha256=h.sha(RUN/'optimizer_updates.jsonl'),rollout_roster=roster,
            updates=updates,all_update_stats=update_stats(updates),updates_with_P09_stats=update_stats(p09_updates),
            selected_input_update=selected['update'],selected_input_decisions=[selected['first_decision'],selected['last_decision']],
            P09_distribution_all524=dict(raw=channel_stats(raw),conditional_mean=channel_stats(mean),sigma=channel_stats(sigma),
                sample_minus_mean_over_sigma=channel_stats((raw-mean)/sigma),
                sampled_tanh_abs_ge_point95=channel_fraction(np.abs(np.tanh(raw))>=.95),
                sampled_tanh_abs_ge_point99=channel_fraction(np.abs(np.tanh(raw))>=.99),
                mean_tanh_abs_ge_point95=channel_fraction(np.abs(np.tanh(mean))>=.95),
                mean_tanh_abs_ge_point99=channel_fraction(np.abs(np.tanh(mean))>=.99)),
            P09_execution_last_tick524={name:channel_fraction(np.stack([r[name] for r in endpoints])) for name in
                ('headroom_clips','endpoint_filter_difference','final_clamp_slew_or_cast_difference','final_hard_bound_near','native_channel_effect')},
            P09_endpoint_assist_owned_count=sum(r['assist_owned'] for r in endpoints),
            P09_endpoint_RR_placed_count=sum(r['RRplaced'] for r in endpoints),
            P09_endpoint_RR_gap_m=stats([r['RRgap'] for r in endpoints]),
            fixed128_same_input=dict(before={k:channel_stats(b[k]) for k in ('mean','sigma','request','network_mean')},
                after={k:channel_stats(c[k]) for k in ('mean','sigma','request','network_mean')},
                delta={k:channel_stats(change[k]) for k in ('raw_mean_delta','log_sigma_delta','requested_residual_delta')},
                sigma_after_over_before=channel_stats(c['sigma']/b['sigma']),
                full_Gaussian_KL_before_to_after=stats(change['kl_original_to_candidate']),
                full_Gaussian_KL_after_to_before=stats(change['kl_candidate_to_original'])),
            limitations=['Existing gradient norms are combined post separate actor/critic clipping, not raw actor-only gradient strength.',
                'Logged surrogate loss is not a reward or physical-success metric; whole-update KL/clip includes all stages in each update.',
                'Tanh .95/.99 are descriptive saturation thresholds, not new guardrails. No reparameterization or noise change.',
                'Headroom fractions are last native tick per P09 decision only; no full-tick clip fraction is invented.',
                'Endpoint filtered REQUEST differs from cap*tanh(raw) because existing filter/rate-limit/history may act; this is not automatically hard clipping.',
                'Final candidate-versus-dispatch differences combine existing clamp/slew/cast, not measured physical joint tracking.',
                'Fixed128 observations remove input-history drift for the endpoint comparison; one distribution slice does not establish closed-loop causation.'],
            model_forward_only=True,optimizer_steps_performed=0,PPO_added=0,AUX_added=0,checkpoint_written=False)
        training.write_json(DEST,report)
        print(json.dumps(dict(report=str(DEST),updates=12,P09_samples=524,fixed_input_update=selected['update'],
            P09_assist_endpoints=report['P09_endpoint_assist_owned_count'],zero_learning=True)))
    finally:
        training.restore_training_rng_state(rng,expected_seed=0);torch.set_num_threads(threads)


if __name__=='__main__':main()

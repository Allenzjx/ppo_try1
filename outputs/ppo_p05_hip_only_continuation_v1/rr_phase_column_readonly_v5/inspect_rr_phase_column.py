"""Read-only P09-column sensitivity on seven sealed actual capture actions.

No fit, optimizer, budget, new network, checkpoint save, or AUX event API.
Run only after the active video has sealed and root explicitly requests it.
"""
from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
ROOT=OUT.parents[1]
P05_HELPER=OUT/'p05_preedge_rehearsal_v5/inspect_p05_preedge.py'
P05_SHA='7fb6ba9e512ada419b7d4db30f3243b2184dc05e50cf5404381aabc8e9488fab'
if hashlib.sha256(P05_HELPER.read_bytes()).hexdigest()!=P05_SHA:
    raise RuntimeError('frozen readonly distribution helper changed')
spec=importlib.util.spec_from_file_location('_rr_readonly_frozen_p05',P05_HELPER)
prior=importlib.util.module_from_spec(spec);sys.modules[spec.name]=prior;spec.loader.exec_module(prior)
kernel,torch,training,require=prior.kernel,prior.torch,prior.training,prior.require
from wlr50_clean.ppo import semantic_cli,semantic_migration
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_OBSERVATION_LAYOUT

SCHEMA='wlr50_clean.seven_actual_RR_capture_phase_column_readonly.v5'
COLUMN=8
NON_P09_PHASES=[1,2,3,4,5,6,7,8,10,11,12,13]
SOURCE=OUT/'checkpoints/history/checkpoint_step_000218496.pt'
SOURCE_SHA='6f7c572aaa81ea21407a4aac13809967885cfb9119014a78151b4bf0eff9e227'
SOURCE_MANIFEST_SHA='ec4c7ad3f8541279511516aa7334c80f7c3201b4a01fc63ce5108197828a9b2c'
HEAD='6ac7b553d79280eddb72f4d0dece586aa2657db7'
OLD_HEAD='336b7c56d2f048d44c866b2357d33e1eb1f21dce'
RUN=ROOT/'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1428131775302Z_g336b7c56d2f0_23b26a4c40a24a178d190def03381413'
ROLLOUT=RUN/'rollouts/rollout_001655.pt'
ROLLOUT_SHA='e5bab6f8839f9a14f7d672f8171ce0f03668fa814b1fb3ec42567e1580175dd2'
CREDIT=OUT/'block09_RR_capture_credit_readonly.json'
CREDIT_SHA='82d9749aa6842711d3007ce0fae56f25a47829f80cc058ffe4b7916f616f52ad'
CREDIT_HELPER=OUT/'block09_RR_capture_credit_readonly.py'
CREDIT_HELPER_SHA='dede2e4ddec6f6132f8a266cc770cf8a617020ad0d2e516c5e767064ae83df4c'
DECISIONS=list(range(216238,216245))
sha=prior.sha


def distribution(actor,obs,leaf=None):
    """Same true389 kernel; only functional existing P09 input column differs."""
    layer=kernel.first_layer(actor);x=obs['policy'];kernel.validate_observation(x)
    require(x.device.type==layer.weight.device.type=='cpu','explicit CPU actor/data copies only')
    latent=actor.get_latent(obs);require(torch.equal(latent,x),'Identity input required')
    center,evidence=kernel.p05_capture_request_history(latent)
    if leaf is None:raw=actor.mlp(latent)
    else:
        require(leaf.shape==(256,) and leaf.dtype==torch.float32 and leaf.device.type=='cpu'
            and bool(torch.isfinite(leaf).all()),'exact finite float32 P09 256-column derivative leaf required')
        weight=torch.cat((layer.weight[:,:COLUMN].detach(),leaf[:,None],layer.weight[:,COLUMN+1:].detach()),dim=1)
        raw=torch.func.functional_call(actor.mlp,{'0.weight':weight},(latent,))
    head=kernel.history_conditioned_head(raw,center,kernel.HISTORY_RHO)
    log_sigma,_=kernel.receiving_wheel_effective_log_std(head[:,1,:],latent[:,:372],.25)
    mu=head[:,0,:];caps=evidence['current_cap_full12']
    return dict(mean=mu,log_sigma=log_sigma,sigma=log_sigma.exp(),network_mean=raw[:,0,:],
                history=center,caps=caps,request=caps*mu.tanh())


def directional_response(actor,obs,leaf,direction):
    def outputs(candidate):
        d=distribution(actor,obs,candidate)
        return torch.cat((d['mean'],d['log_sigma'],d['request']),dim=-1)
    _,delta=torch.autograd.functional.jvp(outputs,leaf,direction,create_graph=False,strict=True)
    require(bool(torch.isfinite(delta).all()),'nonfinite derivative')
    return {name:kernel._values(delta[:,start:start+12]) for name,start in (
        ('raw_mean_full12',0),('log_sigma_full12',12),('requested_residual_full12',24))}


def reviewed_data(metadata,current_contract):
    """Fixed direct389/raw source; no current-state reconstruction or new rollout."""
    for path,expected in ((ROLLOUT,ROLLOUT_SHA),(CREDIT,CREDIT_SHA),(CREDIT_HELPER,CREDIT_HELPER_SHA)):
        require(sha(path)==expected,'sealed source changed: '+str(path))
    stored=torch.load(ROLLOUT,map_location='cpu',weights_only=False)
    require(stored['schema']=='wlr50_clean.semantic_on_policy_rollout.v1'
        and stored['runtime_contract']['source_git_commit']==OLD_HEAD,'wrong old v2 rollout')
    require(stored['policy_contract']==metadata['policy_contract'],'policy shape/kernel contract changed')
    migration=metadata['p05_preedge_approach_recovery_migration'];factor=migration['p05_preedge_approach_recovery_factor']
    require(migration['schema']=='wlr50_clean.p05_preedge_approach_recovery_same389.v1'
        and migration['source_git_commit']==OLD_HEAD and migration['target_git_commit']==HEAD
        and migration['source_contract_sha256']==semantic_migration.digest(stored['runtime_contract'])
        and migration['target_contract_sha256']==semantic_migration.digest(current_contract),
        'old/current control MDP binding differs')
    require(factor['affected_control_phases']==['P05'] and factor['same_mdp_claimed'] is False
        and factor['controller_transition_semantics_changed'] is True and factor['nominal_changed'] is True
        and factor['observation_semantics_changed']==[]
        and all(factor[k] is False for k in ('reward_changed','observation_codec_changed',
            'observation_shape_changed','policy_kernel_changed','caps_changed','sigma_changed',
            'capture_assist_changed','physical_task_acceptance_rules_changed','added_mutable_state')),
        'P09 input admission cannot assume an unreviewed control or observation change')
    plan=Path(migration['plan_path']);require(sha(plan)==migration['plan_sha256'],'migration plan changed')
    require(json.loads(plan.read_text(encoding='utf-8'))=={k:v for k,v in migration.items()
        if k not in ('plan_path','plan_sha256')},'embedded immutable migration differs from plan')
    credit=json.loads(CREDIT.read_text(encoding='utf-8'))['updates']['1655']
    require(credit['hashes']['rollout']==ROLLOUT_SHA and credit['verified_all_selected_quality_families_zero'],
        'existing actual-credit evidence differs')
    rows={r['global_decision']:r for r in (dict(zip(credit['row_columns'],row)) for row in credit['rows'])}
    chosen=[rows[n] for n in DECISIONS];indices=[r['rollout_index'] for r in chosen]
    require(len(set(indices))==7 and indices==list(range(indices[0],indices[0]+7))
        and all(r['request_phase']=='P09' and not r['done'] and r['actual_standardized_advantage']>0 for r in chosen),
        'seven actual consecutive P09 capture actions required')
    require(chosen[-1]['RR_qualified_TOP'] is True and chosen[-1]['RR_placed_history'] is True
        and chosen[-1]['end_phase']=='P10'
        and rows[216245]['RR_qualified_TOP'] is True and rows[216246]['RR_qualified_TOP'] is True
        and rows[216253]['RR_qualified_TOP'] is False,'capture/continuation/contact-loss provenance differs')
    x=stored['observations']['policy'][indices,0].clone();raw=stored['actions'][indices,0].clone()
    require(x.dtype==torch.float32 and x.shape==(7,389) and raw.dtype==torch.float32,
        'source must contain direct float32 389 inputs and raw12 actions')
    kernel.validate_observation(x);kernel._targets(raw,7,'cpu')
    require(bool((x[:,COLUMN]==1).all()),'only P09 direct positive inputs')
    require(not bool(stored['dones'][indices].any()),'terminal sample not admitted')
    require(torch.equal(x,stored['observations']['critic'][indices,0]),'actor/critic source observations differ')
    # No P05 WAIT/initialized==0 requirement: FL state persists in later phases.
    return x,raw,dict(source='sealed_actual_PPO_raw_actions_not_stored_mu_or_final_target',
        source_rollout=str(ROLLOUT),source_rollout_sha256=ROLLOUT_SHA,
        existing_credit_report=str(CREDIT),existing_credit_report_sha256=CREDIT_SHA,
        existing_credit_helper_sha256=CREDIT_HELPER_SHA,global_decisions=DECISIONS,rollout_indices=indices,
        rows=chosen,observation_provenance='direct_saved389_no_reconstruction_no_X17_remapping',
        old_runtime_head=OLD_HEAD,current_runtime_head=HEAD,
        control_MDP_difference='P05 nominal recovery only; new gate P05_only=false on every selected P09 input',
        migration_plan_sha256=migration['plan_sha256'],off_policy_AUX_candidate=True,
        training_admission_or_execution_authorization=False,independent_validation_rows=0,
        seven_temporally_adjacent_rows_one_trajectory=True,after_contact_loss_excluded=True,
        historical_whole_episode_success_claimed=False,fresh_rollout_claimed=False)


def _inspect(actor,x,targets,*,sigma_nullspace):
    obs=kernel.tensors(x);kernel._targets(targets,7,'cpu')
    require(len(x)==7 and bool((x[:,COLUMN]==1).all()),'exact seven P09 positive states required')
    leaf=kernel.first_layer(actor).weight[:,COLUMN].detach().clone().requires_grad_(True)
    d=distribution(actor,obs,leaf)
    require(torch.equal(d['mean'],actor(obs,stochastic_output=False)),'official conditional mean mismatch')
    channel_loss=.5*(d['mean']-targets).square().mean(0);loss=channel_loss.mean()
    gradient,=torch.autograd.grad(loss,leaf,retain_graph=True)
    require(bool(torch.isfinite(gradient).all()),'nonfinite raw mean gradient')
    norm=gradient.norm();direction=-gradient
    unit=direction/norm if float(norm)>0 else torch.zeros_like(direction)
    result=dict(raw_mean_half_MSE=float(loss.detach()),raw_mean_MSE=2*float(loss.detach()),
        raw_mean_half_MSE_per_channel=kernel._values(channel_loss),gradient_full256=kernel._values(gradient),
        gradient_l2=float(norm),negative_gradient_unit_lr_JVP=directional_response(actor,obs,leaf,direction),
        unit_parameter_L2_negative_gradient_JVP=directional_response(actor,obs,leaf,unit),
        unit_direction_does_not_select_learning_rate=True,
        current_distribution_and_target=kernel._summary(kernel._detached(d),targets))
    if sigma_nullspace:
        # Pure local derivative linear algebra: no proposed weight is applied.
        jac=torch.autograd.functional.jacobian(lambda w:distribution(actor,obs,w)['log_sigma'].reshape(-1),
            leaf,create_graph=False,strict=True,vectorize=False).detach()
        require(jac.shape==(84,256) and bool(torch.isfinite(jac).all()),'invalid seven-state sigma Jacobian')
        _,singular,vh=torch.linalg.svd(jac.double(),full_matrices=True)
        tolerance=float(singular[0])*max(jac.shape)*torch.finfo(torch.float32).eps
        rank=int((singular>tolerance).sum())
        original=direction.double();projected=original-vh[:rank].T@(vh[:rank]@original)
        projected32=projected.float()
        result['seven_state_local_sigma_nullspace']=dict(jacobian_shape=[84,256],
            jacobian_sha256=training.state_hash(jac),singular_values=kernel._values(singular),
            numerical_rank=rank,numerical_nullity=256-rank,rank_absolute_tolerance=tolerance,
            rank_tolerance_rule='max(shape)*float32_epsilon*largest_singular_value; double SVD of float32 derivatives',
            projected_negative_gradient_full256=kernel._values(projected32),
            projected_negative_gradient_l2=float(projected32.norm()),
            retained_gradient_norm_fraction=float(projected32.norm()/norm) if float(norm)>0 else 0.,
            predicted_loss_derivative=float(gradient.double()@projected),
            maximum_abs_Jsigma_times_projected_direction=float((jac.double()@projected).abs().max()),
            projected_negative_gradient_unit_lr_JVP=directional_response(actor,obs,leaf,projected32),
            local_derivative_only_not_finite_step_sigma_invariance=True,
            unobserved_P09_sigma_invariance_claimed=False,optimizer_or_projection_update_implemented=False)
    synthetic=torch.zeros(12,389);synthetic[torch.arange(12),torch.tensor(NON_P09_PHASES)-1]=1
    require(bool((synthetic[:,COLUMN]==0).all()),'nonP09 algebra probes must have X8 exactly zero')
    probes=kernel.tensors(synthetic)
    with torch.no_grad():
        exact=kernel._exact_invariance(distribution(actor,probes),distribution(actor,probes,
            leaf.detach()+torch.linspace(-.03125,.03125,256)))
    require(exact,'functional P09-column algebraic probe changed nonP09 Gaussian')
    result.update(nonP09_same_input_math='Delta hidden = Delta W[:,8]*X8 = 0 for exact nonP09 onehot; entire Gaussian unchanged',
        nonP09_synthetic_probe_phases=NON_P09_PHASES,synthetic_entire_Gaussian_exact=exact,
        synthetic_is_not_actual_physical_coverage=True,real_nonP09_holdout_rows=0,
        future_closed_loop_HISTORY_invariance_claimed=False,P09_mean_and_sigma_may_change=True)
    return result


def inspect_actor(actor,x,targets,*,sigma_nullspace=False):
    require(not torch.cuda.is_available(),'CUDA_VISIBLE_DEVICES=-1 required')
    state=deepcopy(actor.state_dict());before=training.state_hash(state)
    grads={k:None if p.grad is None else p.grad.detach().clone() for k,p in actor.named_parameters()}
    rng=training.capture_training_rng_state(seed=0)
    try:return _inspect(actor,x,targets,sigma_nullspace=sigma_nullspace)
    finally:
        unchanged=(training.state_hash(actor.state_dict())==before and training.capture_training_rng_state(seed=0)==rng
            and all(p.grad is None if grads[k] is None else p.grad is not None and torch.equal(p.grad,grads[k])
                for k,p in actor.named_parameters()))
        if not unchanged:
            actor.load_state_dict(state,strict=True)
            for k,p in actor.named_parameters():p.grad=None if grads[k] is None else grads[k].clone()
            training.restore_training_rng_state(rng,expected_seed=0)
            raise RuntimeError('readonly actor/grads/RNG mutation restored; refusing report')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--include-seven-state-sigma-nullspace',action='store_true')
    args=parser.parse_args(argv)
    require(not torch.cuda.is_available(),'readonly CPU actor copy required')
    destination=args.report.resolve();require(destination.is_relative_to(HERE) and not destination.exists(),'new report inside this outputs directory required')
    require(sha(SOURCE)==SOURCE_SHA and sha(SOURCE.with_name(SOURCE.stem+'_manifest.json'))==SOURCE_MANIFEST_SHA,
        'explicit root-selected current source changed')
    metadata=semantic_migration.checkpoint_metadata(SOURCE)
    require(tuple(metadata[k] for k in ('global_policy_decisions','ppo_updates','optimizer_steps'))==(218496,1672,33440),
        'current source counters differ')
    contract=semantic_cli.runtime_contract(expected_head=HEAD,semantic_version='v3',experiment_id='p05_hip_only_continuation_v1')
    require(metadata['runtime_contract']==contract,'current runtime contract differs')
    rng=training.capture_training_rng_state(seed=metadata['seed']);threads=torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        x,raw,receipt=reviewed_data(metadata,contract)
        payload=torch.load(SOURCE,map_location='cpu',weights_only=False)
        require(payload['infos']=={k:v for k,v in metadata.items() if k not in
            ('checkpoint_path','checkpoint_sha256','save_load_round_trip')},'embedded source infos mismatch')
        protected=prior._protected_payload(payload)
        require(protected['Adam']==metadata['optimizer_state_sha256'],'source Adam hash mismatch')
        actor=kernel.SemanticP05CaptureHistoryMLPModel(kernel.tensors(x[:1]),{'actor':['policy'],'critic':['critic']},
            'actor',12,hidden_dims=(256,256),activation='elu',obs_normalization=False,
            distribution_cfg={'class_name':'HeteroscedasticGaussianDistribution','std_type':'log','init_std':.15},
            observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT,exploration_std_temperature=.25)
        actor.load_state_dict(payload['actor_state_dict'],strict=True);actor.eval()
        require(training.parameter_hash(actor)==metadata['actor_parameter_sha256'],'source actor hash mismatch')
        result=inspect_actor(actor,x,raw,sigma_nullspace=args.include_seven_state_sigma_nullspace)
        require(prior._protected_payload(payload)==protected and sha(SOURCE)==SOURCE_SHA,'protected source state changed')
        result.update(schema=SCHEMA,inspection_only=True,source=str(SOURCE),source_sha256=SOURCE_SHA,
            source_manifest_sha256=SOURCE_MANIFEST_SHA,runtime_contract_sha256=semantic_migration.digest(contract),
            helper_sha256=sha(__file__),frozen_p05_helper_sha256=P05_SHA,
            frozen_distribution_helper_sha256=prior.FROZEN_SHA256,data_receipt=receipt,
            current_training_payload_protected=protected,parameter_scope='actor.mlp.0.weight[:,8] only',scalar_count=256,
            conditional_mean_HISTORY_derivative=.1,requested_residual_semantics='cap*tanh(mu); not final drive or nominal',
            channel_order=kernel.CHANNELS,request_units=kernel.UNITS,
            optimizer_steps_performed=0,PPO_decisions_added=0,PPO_updates_added=0,AUX_updates_added=0,
            checkpoint_written=False,budget_selected=False,physical_success_claimed=False,teacher_deployed=False)
        training.write_json(destination,result)
        print(json.dumps({'report':str(destination),'schema':SCHEMA,'rows':7,'optimizer_steps':0,
            'raw_mean_MSE':result['raw_mean_MSE'],'gradient_l2':result['gradient_l2'],
            'sigma_Jacobian_rank':result.get('seven_state_local_sigma_nullspace',{}).get('numerical_rank')}))
        return result
    finally:
        training.restore_training_rng_state(rng,expected_seed=metadata['seed'])
        torch.set_num_threads(threads)


if __name__=='__main__':main()

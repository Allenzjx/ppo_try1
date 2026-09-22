"""Read-only P05-column derivatives; no optimizer, budget, fit or AUX event API.

Labels are independently reviewed actual owner-empty raw actions. Later assisted
FL capture is provenance for local continuation, not an unassisted success or a
label for the owned FL channels. Physical evaluation remains independent.
"""
from copy import deepcopy
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
FROZEN=HERE.parent/'front_rehearsal_v1/front_rehearsal.py'
FROZEN_SHA256='d72ff9dd9de8f2e302d2d41d9660a70ba1c880eeb2505b6cd8db26da37895321'
if hashlib.sha256(FROZEN.read_bytes()).hexdigest()!=FROZEN_SHA256:
    raise RuntimeError('frozen distribution helper changed')
spec=importlib.util.spec_from_file_location('_p05_readonly_frozen_kernel',FROZEN)
kernel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernel;spec.loader.exec_module(kernel)
torch,training,require=kernel.torch,kernel.training,kernel.require
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_OBSERVATION_LAYOUT
SCHEMA='wlr50_clean.P05_preedge_phase_column_readonly.v5'
COLUMN=4
NON_P05_PHASES=[1,2,3,4,6,7,8,9,10,11,12,13]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def distribution(actor,obs,leaf=None):
    """Functional column4 only; true history and learned log-sigma derivatives."""
    layer=kernel.first_layer(actor);x=obs['policy'];kernel.validate_observation(x)
    require(layer.weight.device.type=='cpu' and x.device.type=='cpu','readonly actor/data copies must be CPU')
    latent=actor.get_latent(obs);require(torch.equal(latent,x),'Identity input normalization required')
    center,evidence=kernel.p05_capture_request_history(latent)
    if leaf is None:
        raw=actor.mlp(latent)
    else:
        require(leaf.shape==(256,) and leaf.dtype==torch.float32 and leaf.device.type=='cpu',
                'temporary derivative leaf is exactly the 256 P05 phase-column values')
        require(bool(torch.isfinite(leaf).all()),'finite temporary algebraic leaf required')
        weight=torch.cat((layer.weight[:,:COLUMN].detach(),leaf[:,None],layer.weight[:,COLUMN+1:].detach()),dim=1)
        raw=torch.func.functional_call(actor.mlp,{'0.weight':weight},(latent,))
    head=kernel.history_conditioned_head(raw,center,kernel.HISTORY_RHO)
    log_sigma,_=kernel.receiving_wheel_effective_log_std(head[:,1,:],latent[:,:372],.25)
    mu=head[:,0,:];caps=evidence['current_cap_full12']
    return {'mean':mu,'log_sigma':log_sigma,'sigma':log_sigma.exp(),
        'network_mean':raw[:,0,:],'history':center,'caps':caps,'request':caps*mu.tanh()}


def dataset(actor,data):
    kernel.first_layer(actor)
    obs={name:kernel.tensors(data[key]) for name,key in (
        ('train','train_observations'),('validation','validation_observations'),
        ('protection','protection_observations'))}
    for name in ('train','validation'):
        x=obs[name]['policy']
        require(bool((x[:,COLUMN]==1).all()),'only actual P05 inputs may be positive labels')
        require(bool((x[:,372:374]==0).all()),'positive input requires assist WAIT and initialized0; interval ownership proof belongs to data loader')
    require(bool((obs['protection']['policy'][:,COLUMN]==0).all()),'real invariant holdout must exclude P05')
    trainrows={row.numpy().tobytes() for row in obs['train']['policy']}
    require(not any(row.numpy().tobytes() in trainrows for row in obs['validation']['policy']),
            'train/validation exact observations overlap')
    targets={name:kernel._targets(data[key],len(obs[name]['policy']),'cpu') for name,key in (
        ('train','train_raw_targets'),('validation','validation_raw_targets'))}
    return obs,targets


def response(actor,obs,leaf,direction):
    def outputs(candidate):
        d=distribution(actor,obs,candidate)
        return torch.cat((d['mean'],d['log_sigma'],d['request']),dim=-1)
    _,delta=torch.autograd.functional.jvp(outputs,leaf,direction,create_graph=False,strict=True)
    require(bool(torch.isfinite(delta).all()),'nonfinite readonly JVP')
    return {'raw_mean_per_unit_lr_full12':kernel._values(delta[:,:12]),
        'log_sigma_per_unit_lr_full12':kernel._values(delta[:,12:24]),
        'requested_residual_per_unit_lr_full12':kernel._values(delta[:,24:]),
        'maximum_abs_mean_per_unit_lr_per_channel':kernel._values(delta[:,:12].abs().amax(0)),
        'maximum_abs_log_sigma_per_unit_lr_per_channel':kernel._values(delta[:,12:24].abs().amax(0)),
        'maximum_abs_requested_residual_per_unit_lr_per_channel':kernel._values(delta[:,24:].abs().amax(0)),
        'all_derivatives_exact_zero':not bool(delta.any())}


def _inspect(actor,data):
    obs,targets=dataset(actor,data)
    original=training.parameter_hash(actor)
    with torch.no_grad():
        baseline={name:kernel._detached(distribution(actor,x)) for name,x in obs.items()}
        require(torch.equal(baseline['train']['mean'],actor(obs['train'],stochastic_output=False)),
                'readonly mean differs from current official actor')
    leaf=kernel.first_layer(actor).weight[:,COLUMN].detach().clone().requires_grad_(True)
    d=distribution(actor,obs['train'],leaf)
    losses=.5*(d['mean']-targets['train']).square().mean(dim=0)
    loss=losses.mean()
    gradient,=torch.autograd.grad(loss,leaf,retain_graph=True)
    channel_gradients=torch.stack([torch.autograd.grad(value,leaf,retain_graph=True)[0] for value in losses])
    require(bool(torch.isfinite(gradient).all()) and bool(torch.isfinite(channel_gradients).all()),
            'nonfinite true conditional-mean gradient')
    require(torch.allclose(channel_gradients.mean(0),gradient,atol=1e-8,rtol=1e-5),
            'channel derivative decomposition differs')
    responses={name:response(actor,x,leaf,-gradient) for name,x in obs.items()}
    require(responses['protection']['all_derivatives_exact_zero'],'nonP05 JVP changed')
    # Fixed arbitrary algebraic perturbation, not gradient descent or a budget.
    algebraic=leaf.detach()+torch.linspace(-.03125,.03125,256)
    synthetic=torch.zeros(12,389)
    synthetic[torch.arange(12),torch.tensor(NON_P05_PHASES)-1]=1
    mathobs=kernel.tensors(synthetic)
    with torch.no_grad():
        exact_real=kernel._exact_invariance(baseline['protection'],distribution(actor,obs['protection'],algebraic))
        exact_math=kernel._exact_invariance(distribution(actor,mathobs),distribution(actor,mathobs,algebraic))
    require(exact_real and exact_math,'P05-column algebraic perturbation changed protected full Gaussian')
    actual=sorted(set(int(v)+1 for v in obs['protection']['policy'][:,:13].argmax(-1)))
    return {'schema':SCHEMA,'inspection_only':True,'optimizer_steps_performed':0,
        'derivative_parameter_scope':'actor.mlp.0.weight[:,4] only','derivative_scalar_count':256,
        'training_rows':len(obs['train']['policy']),'validation_rows':len(obs['validation']['policy']),
        'raw_conditional_half_MSE':float(loss.detach()),'raw_conditional_half_MSE_per_channel':kernel._values(losses),
        'gradient_l2':float(gradient.norm()),'gradient_full256':kernel._values(gradient),
        'per_channel_gradient_l2':kernel._values(channel_gradients.norm(dim=1)),
        'per_channel_gradient_full12x256':kernel._values(channel_gradients),
        'gradient_semantics':'conditional mean raw12 MSE; true HISTORY rho=.9 derivative .1, no artificial rescaling',
        'negative_frozen_gradient_unit_lr_JVP':responses,
        'JVP_is_local_derivative_not_finite_step_or_physical_prediction':True,
        'P05_mean_and_log_sigma_may_both_change':True,
        'same_input_nonP05_math':'Delta_first_hidden=DeltaW[:,4]*X[4]; exact nonP05 onehot X[4]=0; all later activations and entire Gaussian unchanged',
        'actual_nonP05_holdout_phases':actual,
        'missing_actual_nonP05_holdout_phases':[p for p in NON_P05_PHASES if p not in actual],
        'real_holdout_entire_Gaussian_bitwise_equal_for_algebraic_probe':exact_real,
        'synthetic_nonP05_phase_probes':NON_P05_PHASES,
        'synthetic_entire_Gaussian_bitwise_equal_for_algebraic_probe':exact_math,
        'synthetic_probes_are_not_real_physical_coverage':True,
        'algebraic_probe_not_fit_or_copied_to_actor':True,
        'same_input_invariance_does_not_guarantee_future_closed_loop_or_history_invariance':True,
        'train':kernel._summary(baseline['train'],targets['train']),
        'validation':kernel._summary(baseline['validation'],targets['validation']),
        'protection':kernel._summary(baseline['protection']),
        'channel_order':kernel.CHANNELS,'requested_residual_units':kernel.UNITS,
        'request_semantics':'current cap*tanh(conditional mean); before mapper/filter/nominal, not final target',
        'source_actor_sha256_unchanged':original,'PPO_decisions_added':0,'PPO_updates_added':0,
        'PPO_optimizer_steps_added':0,'AUX_updates_added':0,'checkpoint_written':False,
        'teacher_deployed':False,'physical_success_claimed':False,
        'later_assisted_capture_not_labelled_unassisted_success':True,
        'execution_or_budget_interface_available':False,'guard_numbers_selected':False}


def inspect_actor(actor,data):
    require(not torch.cuda.is_available(),'CUDA_VISIBLE_DEVICES=-1 required for readonly inspection')
    state=deepcopy(actor.state_dict());before=training.state_hash(state)
    grads={k:None if p.grad is None else p.grad.detach().clone() for k,p in actor.named_parameters()}
    rng=training.capture_training_rng_state(seed=0)
    try:
        return _inspect(actor,data)
    finally:
        unchanged=(training.state_hash(actor.state_dict())==before and training.capture_training_rng_state(seed=0)==rng
            and all(p.grad is None if grads[k] is None else p.grad is not None and torch.equal(p.grad,grads[k])
                    for k,p in actor.named_parameters()))
        if not unchanged:
            actor.load_state_dict(state,strict=True)
            for k,p in actor.named_parameters():p.grad=None if grads[k] is None else grads[k].clone()
            training.restore_training_rng_state(rng,expected_seed=0)
            raise RuntimeError('readonly inspection unexpectedly changed actor/grad/RNG; restored and refused report')


def _protected_payload(payload):
    infos=payload['infos']
    return {'Adam':training.state_hash(payload['optimizer_state_dict']),
        'critic':training.state_hash(payload['critic_state_dict']),
        'saved_full_training_RNG':training.state_hash(infos['training_rng_state']),
        'normalizer_metadata':infos['normalizer_state_sha256'],
        'PPO_counters':{k:infos[k] for k in ('global_policy_decisions','ppo_updates','optimizer_steps')}}


def cpu_inspection(checkpoint,metadata,data):
    require(not torch.cuda.is_available(),'CPU-only actor copy; no CUDA resume')
    rng=training.capture_training_rng_state(seed=metadata['seed'])
    before_sha=sha(checkpoint)
    require(before_sha==metadata['checkpoint_sha256'],'source checkpoint hash mismatch')
    try:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        expected={k:v for k,v in metadata.items() if k not in ('checkpoint_path','checkpoint_sha256','save_load_round_trip')}
        require(payload['infos']==expected,'source embedded metadata and sidecar differ')
        protected=_protected_payload(payload)
        require(protected['Adam']==metadata['optimizer_state_sha256'],'source Adam integrity mismatch')
        # Instantiate ONLY an actor. No runner, critic or PPO optimizer exists here.
        actor=kernel.SemanticP05CaptureHistoryMLPModel(kernel.tensors(data['train_observations'][:1]),
            {'actor':['policy'],'critic':['critic']},'actor',12,hidden_dims=(256,256),activation='elu',
            obs_normalization=False,distribution_cfg={'class_name':'HeteroscedasticGaussianDistribution',
            'std_type':'log','init_std':.15},observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT,
            exploration_std_temperature=.25)
        actor.load_state_dict(payload['actor_state_dict'],strict=True);actor.eval()
        require(training.parameter_hash(actor)==metadata['actor_parameter_sha256'],'source actor hash differs')
        result=inspect_actor(actor,data)
        require(_protected_payload(payload)==protected and sha(checkpoint)==before_sha,
                'readonly inspection changed protected source state')
        return {**result,'source_checkpoint_sha256_unchanged':before_sha,
            'protected_loaded_payload_unchanged':protected,
            'load_semantics':'CPU actor-state copy only; no runner/critic/Adam construction or state load; checkpoint CUDA RNG bytes preserved, not restored on CPU',
            'source_device_metadata_unchanged':True}
    finally:
        training.restore_training_rng_state(rng,expected_seed=metadata['seed'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','report'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--expected-checkpoint-sha256',required=True)
    parser.add_argument('--expected-policy-decisions',type=int,required=True)
    parser.add_argument('--expected-head',required=True)
    args=parser.parse_args(argv)
    require(not torch.cuda.is_available(),'readonly CLI requires CUDA_VISIBLE_DEVICES=-1')
    from wlr50_clean.ppo import semantic_cli,semantic_migration
    source=args.checkpoint.resolve(strict=True);metadata=semantic_migration.checkpoint_metadata(source)
    require(metadata['checkpoint_sha256']==args.expected_checkpoint_sha256
            and metadata['global_policy_decisions']==args.expected_policy_decisions,'explicit source SHA/counters mismatch')
    contract=semantic_cli.runtime_contract(expected_head=args.expected_head,semantic_version='v3',experiment_id='p05_hip_only_continuation_v1')
    require(metadata['runtime_contract']==contract,'source/current runtime contract mismatch')
    loader=importlib.import_module('data_v5');data=loader.load_reviewed_data(metadata,contract)
    destination=args.report.resolve()
    require(destination.is_relative_to(HERE.resolve()) and not destination.exists(),'new readonly report required inside v5 outputs')
    result=cpu_inspection(source,metadata,data)
    result.update(data_receipt=data['receipt'],binding={'source_checkpoint':str(source),'source_sha256':metadata['checkpoint_sha256'],
        'source_manifest_sha256':sha(source.with_name(source.stem+'_manifest.json')),
        'runtime_contract_sha256':training.state_hash(contract),'inspector_sha256':sha(__file__),
        'frozen_distribution_kernel_sha256':FROZEN_SHA256,'data_loader_sha256':sha(loader.__file__),
        'data_manifest_sha256':sha(HERE/'data_manifest.json')})
    training.write_json(destination,result)
    print(json.dumps({'report':str(destination),'mode':'readonly_P05_column','optimizer_steps':0,'checkpoint_written':False}))
    return result


if __name__=='__main__':main()

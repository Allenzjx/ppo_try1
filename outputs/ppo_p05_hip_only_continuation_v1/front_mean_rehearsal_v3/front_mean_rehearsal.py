"""Mean-head AUX parameterization/inspection/guards; no automatic optimization.

Only existing mlp.4 mean rows and mean biases are candidate variables. The
current389 HISTORY/receiving sigma kernel is unchanged. All phases can change
their same-input means; P03+ preservation is bounded by explicit protection,
never asserted as exact mean invariance. This module currently has NO fit or
checkpoint-save entry point. Root chooses an explicit objective and budget.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent/'front_rehearsal_v1/front_rehearsal.py'
FROZEN_SHA256 = 'd72ff9dd9de8f2e302d2d41d9660a70ba1c880eeb2505b6cd8db26da37895321'
if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != FROZEN_SHA256:
    raise RuntimeError('immutable current389 distribution kernel changed')
spec = importlib.util.spec_from_file_location('_mean_v3_immutable_kernel',FROZEN)
kernel = importlib.util.module_from_spec(spec); sys.modules[spec.name] = kernel; spec.loader.exec_module(kernel)
torch, training = kernel.torch, kernel.training
require = kernel.require
SCHEMA = 'wlr50_clean.finite_existing_mean_head_aux.v3'
PARAMETERS = ['actor.mlp.4.weight[:12,:]', 'actor.mlp.4.bias[:12]']


def positive(value):
    return type(value) in (int,float) and math.isfinite(value) and value > 0


def positive12(values):
    return isinstance(values,(list,tuple)) and len(values)==12 and all(positive(v) for v in values)


@dataclass(frozen=True)
class Objective:
    # No defaults: a single P01 row cannot silently receive only 1/86 weight.
    phase_weights: dict[str,float]
    protection_weight: float
    raw_channel_weights_full12: tuple[float,...]

    def validate(self):
        require(set(self.phase_weights)=={'P01','P02'} and all(positive(v) for v in self.phase_weights.values()),
                'explicit positive P01 and P02 objective weights required')
        require(positive(self.protection_weight), 'explicit positive protection-loss weight required')
        require(positive12(self.raw_channel_weights_full12), 'explicit positive twelve raw-channel loss weights required')


@dataclass(frozen=True)
class Budget:
    max_attempts: int
    learning_rate: float
    maximum_train_request_shift_full12: tuple[float,...]
    maximum_validation_request_shift_full12: tuple[float,...]
    maximum_protection_request_shift_full12: tuple[float,...]
    maximum_p01_probe_request_shift_full12: tuple[float,...]
    maximum_per_state_full_gaussian_kl: float
    maximum_protection_full_gaussian_kl: float

    def validate(self):
        require(type(self.max_attempts) is int and 1 <= self.max_attempts <= 32, 'finite explicit 1..32 attempts required')
        for name in ('learning_rate','maximum_per_state_full_gaussian_kl','maximum_protection_full_gaussian_kl'):
            require(positive(getattr(self,name)), 'explicit positive finite budget: '+name)
        for name in ('maximum_train_request_shift_full12','maximum_validation_request_shift_full12',
                     'maximum_protection_request_shift_full12','maximum_p01_probe_request_shift_full12'):
            require(positive12(getattr(self,name)), 'explicit twelve-channel REQUEST bounds: '+name)


def final_layer(actor):
    kernel.first_layer(actor)
    layer,reshape=actor.mlp[4],actor.mlp[5]
    require(tuple(layer.bias.shape)==(24,) and reshape.dim==-1 and tuple(reshape.unflattened_size)==(2,12),
            'requires confirmed flat first12 mean, last12 log-sigma row layout')
    require(dict(actor.named_parameters())['mlp.4.weight'] is layer.weight
            and dict(actor.named_parameters())['mlp.4.bias'] is layer.bias, 'mean-head binding differs')
    return layer


def selected_leaf(actor):
    layer=final_layer(actor)
    return torch.cat((layer.weight[:12].detach(),layer.bias[:12,None].detach()),dim=1).clone()


def distribution(actor,obs,leaf=None):
    layer=final_layer(actor); x=obs['policy']; kernel.validate_observation(x)
    require(x.device==layer.weight.device,'observations retain source actor device')
    latent=actor.get_latent(obs)
    require(torch.equal(latent,x),'Identity observation normalization required')
    if leaf is None:
        raw=actor.mlp(latent)
    else:
        require(leaf.shape==(12,257) and leaf.device==layer.weight.device and leaf.dtype==torch.float32,
                'temporary leaf is the existing 12x(256weight+1bias) mean head')
        w=torch.cat((leaf[:,:256],layer.weight[12:].detach()),dim=0)
        b=torch.cat((leaf[:,256],layer.bias[12:].detach()),dim=0)
        raw=torch.func.functional_call(actor.mlp,{'4.weight':w,'4.bias':b},(latent,))
    center,evidence=kernel.p05_capture_request_history(latent)
    head=kernel.history_conditioned_head(raw,center,kernel.HISTORY_RHO)
    log_sigma,_=kernel.receiving_wheel_effective_log_std(head[:,1,:],latent[:,:372],.25)
    mu=head[:,0,:]; caps=evidence['current_cap_full12']
    return {'mean':mu,'sigma':log_sigma.exp(),'log_sigma':log_sigma,'network_mean':raw[:,0,:],
            'history':center,'caps':caps,'request':caps*mu.tanh()}


def dataset(actor,data):
    device=final_layer(actor).weight.device
    names={'train':'train_observations','validation':'validation_observations',
           'protection':'protection_observations','p01_probe':'p01_observations'}
    obs={name:kernel.tensors(data[key],device=device) for name,key in names.items()}
    phase={name:value['policy'][:,:13].argmax(-1)+1 for name,value in obs.items()}
    require(bool(((phase['train']==1)|(phase['train']==2)).all()) and bool((phase['train']==1).any())
            and bool((phase['train']==2).any()) and bool((phase['validation']==2).all()),
            'separate P01/P02 training groups and independent P02-only validation required')
    require(bool(((phase['protection']>=3)&(phase['protection']<=12)).all())
            and bool((phase['p01_probe']==1).all()), 'real P03-P12 protection and unlabeled P01 probes required')
    groups={name:torch.nonzero(phase['train']==number).reshape(-1).tolist() for name,number in [('P01',1),('P02',2)]}
    require(data['train_phase_groups']==groups, 'declared phase groups differ from exact input onehots')
    trainrows={row.numpy().tobytes() for row in obs['train']['policy'].detach().cpu()}
    require(not any(row.numpy().tobytes() in trainrows for row in obs['validation']['policy'].detach().cpu()),
            'training observations overlap validation')
    targets={name:kernel._targets(data[key],len(obs[name]['policy']),device) for name,key in
             [('train','train_raw_targets'),('validation','validation_raw_targets')]}
    return obs,targets,groups


def loss_terms(actor,obs,targets,groups,leaf,reference_protection,objective):
    objective.validate()
    current=distribution(actor,obs['train'],leaf)['mean']
    channel=current.new_tensor(objective.raw_channel_weights_full12)
    terms={name:.5*((current[ids]-targets['train'][ids]).square()*channel).mean()
           for name,ids in groups.items()}
    protection=distribution(actor,obs['protection'],leaf)['mean']
    require(reference_protection.shape==protection.shape and not reference_protection.requires_grad,
            'protection target must be detached source actor means for these exact inputs')
    terms['protection']=.5*((protection-reference_protection).square()*channel).mean()
    terms['total']=sum(objective.phase_weights[name]*terms[name] for name in ('P01','P02'))+objective.protection_weight*terms['protection']
    return terms


def baselines(actor,obs):
    with torch.no_grad(): return {name:kernel._detached(distribution(actor,value)) for name,value in obs.items()}


def guard_candidate(actor,obs,reference,leaf,*,budget):
    """Pure check against ORIGINAL source baselines, not last accepted proposal."""
    budget.validate(); reasons=[]; changes={}
    names={'train':budget.maximum_train_request_shift_full12,
        'validation':budget.maximum_validation_request_shift_full12,
        'protection':budget.maximum_protection_request_shift_full12,
        'p01_probe':budget.maximum_p01_probe_request_shift_full12}
    with torch.no_grad():
        for name,limit in names.items():
            after=distribution(actor,obs[name],leaf); before=reference[name]
            if not all(bool(torch.isfinite(v).all()) for v in after.values()): reasons.append(name+'_nonfinite')
            # Complete sigma path stays exact, including receiving multipliers.
            if not torch.equal(before['sigma'],after['sigma']) or not torch.equal(before['log_sigma'],after['log_sigma']):
                reasons.append(name+'_sigma_changed')
            delta=kernel.gaussian_change(before,after)
            if bool((delta['requested_residual_delta'].abs().amax(0)>leaf.new_tensor(limit)).any()):
                reasons.append(name+'_REQUEST_bound')
            kl_limit=budget.maximum_protection_full_gaussian_kl if name=='protection' else budget.maximum_per_state_full_gaussian_kl
            if max(float(delta['kl_original_to_candidate'].max()),float(delta['kl_candidate_to_original'].max()))>kl_limit:
                reasons.append(name+'_full_Gaussian_KL_bound')
            changes[name]={key:kernel._values(v) for key,v in delta.items()}
    return {'accepted_by_trust_only':not reasons,'rejection_reasons':reasons,'change_from_original':changes,
            'no_parameters_copied':True,'not_an_optimizer_step':True,
            'P03plus_mean_bitwise_invariance_claimed':False}


def _jvp(actor,obs,leaf,direction):
    def outputs(candidate):
        d=distribution(actor,obs,candidate)
        return torch.cat((d['mean'],d['log_sigma'],d['request']),dim=-1)
    _,delta=torch.autograd.functional.jvp(outputs,leaf,direction,create_graph=False,strict=True)
    require(not bool(delta[:,12:24].any()),'mean-head direction changed same-input log-sigma derivative')
    return {'raw_mean_per_unit_lr_full12':kernel._values(delta[:,:12]),
            'log_sigma_per_unit_lr_exact_zero':True,
            'requested_residual_per_unit_lr_full12':kernel._values(delta[:,24:]),
            'maximum_abs_requested_residual_per_unit_lr_per_channel':kernel._values(delta[:,24:].abs().amax(0))}


def inspect(actor,data,*,objective):
    """No fitting; exact phase-separated gradients and fixed-gradient response."""
    objective.validate(); original_hash=training.parameter_hash(actor)
    rng=training.capture_training_rng_state(seed=0)
    obs,targets,groups=dataset(actor,data); reference=baselines(actor,obs)
    require(torch.equal(reference['train']['mean'],actor(obs['train'],stochastic_output=False)),
            'functional mean differs from current official389 actor')
    leaf=selected_leaf(actor).requires_grad_(True)
    terms=loss_terms(actor,obs,targets,groups,leaf,reference['protection']['mean'],objective)
    gradients={name:torch.autograd.grad(value,leaf,retain_graph=True)[0] for name,value in terms.items()}
    require(all(bool(torch.isfinite(g).all()) for g in gradients.values()),'nonfinite inspection gradient')
    phase_reports={name:{'raw_weighted_half_MSE':float(value.detach()),
        'gradient_l2':float(gradients[name].norm()),'gradient_by_action_row_l2':kernel._values(gradients[name].norm(dim=1)),
        'gradient_full12x257':kernel._values(gradients[name]),'weight':objective.phase_weights.get(name,objective.protection_weight if name=='protection' else 1.)}
        for name,value in terms.items()}
    responses={name:_jvp(actor,value,leaf,-gradients['total']) for name,value in obs.items()}
    # At source the protection gradient is zero; its second derivative along
    # the proposed negative total-gradient direction reveals restraint later.
    pg=responses['protection']['raw_mean_per_unit_lr_full12']
    tangent=torch.tensor(pg,dtype=leaf.dtype,device=leaf.device)
    curvature=float((tangent.square()*leaf.new_tensor(objective.raw_channel_weights_full12)).mean())
    require(training.parameter_hash(actor)==original_hash and training.capture_training_rng_state(seed=0)==rng,
            'readonly inspection changed actor or complete RNG')
    return {'schema':SCHEMA+'.inspection','objective':asdict(objective),'optimized_parameters':PARAMETERS,
        'optimized_scalar_count':3084,'phase_sample_counts':{k:len(v) for k,v in groups.items()},
        'phase_separated_losses_and_gradients':phase_reports,'negative_total_gradient_unit_lr_response':responses,
        'protection_directional_second_derivative_at_source':curvature,
        'protection_weighted_directional_second_derivative_at_source':objective.protection_weight*curvature,
        'protection_gradient_zero_at_source_is_expected':not bool(gradients['protection'].any()),
        'train':kernel._summary(reference['train'],targets['train']),
        'validation':kernel._summary(reference['validation'],targets['validation']),
        'protection':kernel._summary(reference['protection']),'p01_unlabeled_probe':kernel._summary(reference['p01_probe']),
        'actual_protection_phases':sorted(set(int(v)+1 for v in obs['protection']['policy'][:,:13].argmax(-1))),
        'missing_actual_protection_phases':[phase for phase in range(3,14)
            if not bool((obs['protection']['policy'][:,phase-1]==1).any())],
        'P13_actual_protection_coverage':False,
        'same_input_sigma_exact_by_frozen_trunk_and_sigma_rows':True,
        'P03plus_mean_bitwise_invariance_claimed':False,'future_closed_loop_invariance_claimed':False,
        'conditional_mean_gradient_CHAIN':.1,'request_semantics':'cap*tanh(raw_mu), not nominal/final drive',
        'finite_step_or_physical_success_prediction':False,'optimizer_steps_performed':0,
        'AUX_updates_added':0,'PPO_decisions_added':0,'PPO_updates_added':0,'PPO_optimizer_steps_added':0,
        'teacher_deployed':False,'physical_success_claimed':False,'actor_sha256_unchanged':original_hash}

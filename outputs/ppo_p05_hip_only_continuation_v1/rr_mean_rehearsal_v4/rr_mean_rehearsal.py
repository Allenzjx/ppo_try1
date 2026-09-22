"""Bounded RR data/loss adapter over immutable v3 mean-head mechanics.

No automatic fitting or budget. The existing 12 mean rows remain the only
variables; current389/HISTORY/receiving-sigma and all Gaussian guards are reused
without changing their frozen implementation. All same-input means may change.
"""
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
FROZEN_MEAN=HERE.parent/'front_mean_rehearsal_v3/front_mean_rehearsal.py'
FROZEN_MEAN_SHA256='8a1f0967dcb0e13e6f7561b35e8f727e561c85ff469fcd3434f30c18b0fcc2be'
if hashlib.sha256(FROZEN_MEAN.read_bytes()).hexdigest()!=FROZEN_MEAN_SHA256:
    raise RuntimeError('immutable reviewed mean-head mechanics changed')
spec=importlib.util.spec_from_file_location('_rr_v4_frozen_mean',FROZEN_MEAN)
frozen=importlib.util.module_from_spec(spec);sys.modules[spec.name]=frozen;spec.loader.exec_module(frozen)
kernel=frozen.kernel
torch,training,require=frozen.torch,frozen.training,frozen.require
positive,positive12=frozen.positive,frozen.positive12
Budget=frozen.Budget
PARAMETERS=frozen.PARAMETERS.copy()
final_layer,selected_leaf,distribution=frozen.final_layer,frozen.selected_leaf,frozen.distribution
baselines,guard_candidate,_jvp=frozen.baselines,frozen.guard_candidate,frozen._jvp
FROZEN,FROZEN_SHA256=frozen.FROZEN,frozen.FROZEN_SHA256
SCHEMA='wlr50_clean.finite_existing_mean_head_RR_aux.v4'
POSITIVE_PHASES=('P09','P10','P11')
VALIDATION_PHASES=('P09','P11')
PROTECTION_PHASES=(1,2,3,4,5,6,7,8,12)


@dataclass(frozen=True)
class Objective:
    phase_weights: dict[str,float]
    protection_weight: float
    raw_channel_weights_full12: tuple[float,...]

    def validate(self):
        require(set(self.phase_weights)==set(POSITIVE_PHASES)
            and all(positive(v) for v in self.phase_weights.values()),
            'explicit positive P09/P10/P11 weights required; singleton P10 must not be silently diluted')
        require(positive(self.protection_weight),'explicit positive protection-loss weight required')
        require(positive12(self.raw_channel_weights_full12),'all twelve actual raw action channels require positive weights')


def dataset(actor,data):
    device=final_layer(actor).weight.device
    names={'train':'train_observations','validation':'validation_observations',
           'protection':'protection_observations','p01_probe':'p01_observations'}
    obs={name:kernel.tensors(data[key],device=device) for name,key in names.items()}
    phase={name:value['policy'][:,:13].argmax(-1)+1 for name,value in obs.items()}
    sets={name:set(int(v) for v in p) for name,p in phase.items()}
    require(sets['train']=={9,10,11} and sets['validation']=={9,11},
            'actual P09/P10/P11 positives and P09/P11 validation required; no invented P10 validation')
    require(sets['protection']==set(PROTECTION_PHASES) and sets['p01_probe']=={1},
            'real P01-P08/P12 protection and separate unlabeled P01 probes required; no P13 claim')
    groups={name:torch.nonzero(phase['train']==int(name[1:])).reshape(-1).tolist() for name in POSITIVE_PHASES}
    validation={name:torch.nonzero(phase['validation']==int(name[1:])).reshape(-1).tolist()
                for name in VALIDATION_PHASES}
    require(data['train_phase_groups']==groups and data['validation_phase_groups']==validation,
            'declared groups differ from actual observation phase onehots')
    require(len(groups['P10'])==1,'this reviewed source has exactly one P10 positive and no P10 holdout')
    trainrows={row.numpy().tobytes() for row in obs['train']['policy'].detach().cpu()}
    require(not any(row.numpy().tobytes() in trainrows for row in obs['validation']['policy'].detach().cpu()),
            'training and validation observations overlap')
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
            'protection target is detached CURRENT source mean, never old stochastic action')
    terms['protection']=.5*((protection-reference_protection).square()*channel).mean()
    terms['total']=sum(objective.phase_weights[name]*terms[name] for name in POSITIVE_PHASES)
    terms['total']=terms['total']+objective.protection_weight*terms['protection']
    return terms


def inspect(actor,data,*,objective):
    """Read-only per-phase gradients/JVP; no budget, fit, sampling or success."""
    objective.validate();original_hash=training.parameter_hash(actor)
    rng=training.capture_training_rng_state(seed=0)
    obs,targets,groups=dataset(actor,data);reference=baselines(actor,obs)
    require(torch.equal(reference['train']['mean'],actor(obs['train'],stochastic_output=False)),
            'functional mean differs from official current389 actor')
    leaf=selected_leaf(actor).requires_grad_(True)
    terms=loss_terms(actor,obs,targets,groups,leaf,reference['protection']['mean'],objective)
    gradients={name:torch.autograd.grad(value,leaf,retain_graph=True)[0] for name,value in terms.items()}
    require(all(bool(torch.isfinite(g).all()) for g in gradients.values()),'nonfinite inspection gradient')
    losses={name:{'raw_weighted_half_MSE':float(value.detach()),'gradient_l2':float(gradients[name].norm()),
        'gradient_by_action_row_l2':kernel._values(gradients[name].norm(dim=1)),
        'gradient_full12x257':kernel._values(gradients[name]),
        'weight':objective.phase_weights.get(name,objective.protection_weight if name=='protection' else 1.)}
        for name,value in terms.items()}
    responses={name:_jvp(actor,value,leaf,-gradients['total']) for name,value in obs.items()}
    tangent=leaf.new_tensor(responses['protection']['raw_mean_per_unit_lr_full12'])
    curvature=float((tangent.square()*leaf.new_tensor(objective.raw_channel_weights_full12)).mean())
    require(training.parameter_hash(actor)==original_hash and training.capture_training_rng_state(seed=0)==rng,
            'inspection changed source actor or RNG')
    phase=obs['protection']['policy'][:,:13].argmax(-1)+1
    protected_phases=sorted(set(int(v) for v in phase))
    return {'schema':SCHEMA+'.inspection','objective':asdict(objective),'optimized_parameters':PARAMETERS,
        'optimized_scalar_count':3084,'phase_sample_counts':{k:len(v) for k,v in groups.items()},
        'validation_phase_sample_counts':{k:len(v) for k,v in data['validation_phase_groups'].items()},
        'phase_separated_losses_and_gradients':losses,'negative_total_gradient_unit_lr_response':responses,
        'protection_directional_second_derivative_at_source':curvature,
        'protection_weighted_directional_second_derivative_at_source':objective.protection_weight*curvature,
        'protection_gradient_zero_at_source_is_expected':not bool(gradients['protection'].any()),
        'train':kernel._summary(reference['train'],targets['train']),
        'validation':kernel._summary(reference['validation'],targets['validation']),
        'protection':kernel._summary(reference['protection']),'p01_unlabeled_probe':kernel._summary(reference['p01_probe']),
        'actual_protection_phases':protected_phases,
        'missing_actual_protection_phases':[i for i in range(1,14) if i not in protected_phases],
        'P13_actual_protection_coverage':False,'P10_independent_validation_rows':0,
        'single_historical_trajectory_train_validation_correlated':True,
        'same_input_sigma_exact_by_frozen_trunk_and_sigma_rows':True,
        'P03plus_mean_bitwise_invariance_claimed':False,'future_closed_loop_invariance_claimed':False,
        'conditional_mean_gradient_CHAIN':.1,'request_semantics':'cap*tanh(raw_mu), not nominal/final drive',
        'finite_step_or_physical_success_prediction':False,'optimizer_steps_performed':0,
        'AUX_updates_added':0,'PPO_decisions_added':0,'PPO_updates_added':0,'PPO_optimizer_steps_added':0,
        'teacher_deployed':False,'physical_success_claimed':False,'actor_sha256_unchanged':original_hash}

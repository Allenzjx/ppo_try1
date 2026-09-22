"""Output-only finite FL-mean candidate; no simulator or automatic execution.

Importing this module does not load a checkpoint or optimize anything. Real use
requires separately reviewed data and an explicit caller authorization. The only
optimizable objects are two temporary leaves containing 257 copied FL mean-row
scalars. Neither PPO Adam nor an on-policy storage interface is used here.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/'src'))

import torch
from tensordict import TensorDict
from wlr50_clean.ppo.semantic_history_actor import (
    SemanticTaskConditionedHipWheelHistoryMLPModel, cap_transition_request_history,
    history_conditioned_head, task_conditioned_effective_log_std, HISTORY_RHO)
from wlr50_clean.ppo import semantic_training as training

EXPERIMENT = 'task_conditioned_hip_wheel_v1'
LABEL_SCOPE = 'FL_AIR_gap_approach_not_capture'
LEDGER_KEY = 'auxiliary_mean_learning'


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Budget:
    max_steps: int = 16
    learning_rate: float = .05
    maximum_train_request_shift_deg: float = 1.
    maximum_holdout_request_shift_deg: float = .25
    maximum_per_state_conditional_kl: float = .1

    def validate(self):
        require(type(self.max_steps) is int and 1 <= self.max_steps <= 16, 'aux maximum steps must be 1..16')
        limits = {'learning_rate':.05, 'maximum_train_request_shift_deg':1.,
            'maximum_holdout_request_shift_deg':.25, 'maximum_per_state_conditional_kl':.1}
        for key, ceiling in limits.items():
            value = getattr(self,key)
            require(type(value) in (float,int) and math.isfinite(value) and 0 < value <= ceiling,
                'finite reviewed aux budget exceeded: '+key)


def tensors(observations, *, device):
    x = torch.as_tensor(observations, dtype=torch.float32, device=device)
    require(x.ndim == 2 and x.shape[1] == 372 and 1 <= x.shape[0] <= 128,
        'offline states must be a bounded N x 372 array')
    require(bool(torch.isfinite(x).all()) and bool((x.abs() <= 20).all()), 'invalid saved state')
    return TensorDict({'policy':x,'critic':x.clone()}, batch_size=[x.shape[0]], device=device)


def mean_layer(actor):
    require(type(actor) is SemanticTaskConditionedHipWheelHistoryMLPModel,
        'aux supports only the unchanged task-conditioned actor')
    layer = actor.mlp[4]
    require(isinstance(layer,torch.nn.Linear) and tuple(layer.weight.shape)==(24,256)
        and tuple(layer.bias.shape)==(24,) and len(actor.mlp)==6
        and isinstance(actor.mlp[5],torch.nn.Unflatten), 'unexpected canonical mean/log-sigma head')
    names = dict(actor.named_parameters())
    require(names['mlp.4.weight'] is layer.weight and names['mlp.4.bias'] is layer.bias,
        'wrong parameter binding')
    return layer


def distribution(actor, obs):
    """Read current same-numeric-observation mean/sigma; no sample or RNG draw."""
    with torch.no_grad():
        latent = actor.get_latent(obs)
        history, evidence = cap_transition_request_history(latent)
        raw_head = actor.mlp(latent)
        head = history_conditioned_head(raw_head,history,HISTORY_RHO)
        log_sigma,_ = task_conditioned_effective_log_std(head[...,1,:],latent,.25)
        mu = actor(obs,stochastic_output=False)
        require(torch.equal(mu,head[...,0,:]), 'offline mean differs from the actual deterministic actor')
        return {'mean':mu.detach().clone(),'sigma':log_sigma.exp().detach().clone(),
            'network_mean':raw_head[...,0,:].detach().clone(),'history':history.detach().clone(),
            'caps':evidence['current_cap_full12'].detach().clone()}


def functional_mean(actor, obs, row_weight, row_bias):
    """Use the existing full actor forward and its genuine HISTORY kernel."""
    layer = mean_layer(actor)
    replacement = {'mlp.4.weight':torch.cat((row_weight,layer.weight[1:].detach()),dim=0),
        'mlp.4.bias':torch.cat((row_bias,layer.bias[1:].detach()),dim=0)}
    return torch.func.functional_call(actor,replacement,(obs,),{'stochastic_output':False})


def _values(value):
    return value.detach().cpu().tolist()


def linear_budget_prediction(actor, train_obs, targets, holdout_obs, *, budget=Budget()):
    """Analytic frozen-first-gradient response, no optimizer or parameter writes.

    The head is affine in its 257 selected scalars, but subsequent gradients will
    change; the summed-first-gradient prediction is not a trained result.
    """
    budget.validate()
    with torch.no_grad():
        d,h=distribution(actor,train_obs),distribution(actor,holdout_obs)
        f=actor.get_latent(train_obs);fh=actor.get_latent(holdout_obs)
        for index in range(4):f=actor.mlp[index](f);fh=actor.mlp[index](fh)
        f=torch.cat((f,torch.ones_like(f[:,:1])),dim=1)
        fh=torch.cat((fh,torch.ones_like(fh[:,:1])),dim=1)
        residual=d['mean'][:,0]-targets
        gradient=(1.-HISTORY_RHO)*(f.T@residual)/len(f)
        sum_lr=budget.learning_rate*(budget.max_steps+1)/2
        delta=-(1.-HISTORY_RHO)*(f@gradient)*sum_lr
        hd=-(1.-HISTORY_RHO)*(fh@gradient)*sum_lr
        first=delta*budget.learning_rate/sum_lr
        physical=d['caps'][:,0]*((d['mean'][:,0]+delta).tanh()-d['mean'][:,0].tanh())
        hp=h['caps'][:,0]*((h['mean'][:,0]+hd).tanh()-h['mean'][:,0].tanh())
        stats=lambda x:{'min':float(x.min()),'mean':float(x.mean()),'max':float(x.max())}
        return {'semantics':'frozen first-gradient linear response; no optimizer, no parameter writes, no trained/physical result',
            'budget':asdict(budget),'sum_decaying_learning_rates':sum_lr,'true_history_chain_factor':1.-HISTORY_RHO,
            'leaf_feature_norm_with_bias':stats(f.norm(dim=1)),'first_gradient_norm':float(gradient.norm()),
            'first_gradient_weight_norm':float(gradient[:256].norm()),'first_gradient_bias':float(gradient[256]),
            'first_step_predicted_raw_mean_shift':stats(first),'budget_predicted_raw_mean_shift':stats(delta),
            'budget_predicted_request_shift_deg':stats(physical),'budget_predicted_holdout_request_shift_deg':stats(hp),
            'current_raw_loss':float(.5*residual.square().mean()),
            'linear_predicted_raw_loss':float(.5*(residual+delta).square().mean()),
            'PPO_updates_added':0,'auxiliary_updates_added':0}


def inspect(actor, train_obs, target_raw_fl, holdout_obs):
    """Inspect latest compatible weights before deciding whether aux is needed."""
    mean_layer(actor)
    train = distribution(actor,train_obs); holdout = distribution(actor,holdout_obs)
    targets = torch.as_tensor(target_raw_fl,dtype=train['mean'].dtype,device=train['mean'].device)
    require(targets.shape == train['mean'].shape[:1] and bool(torch.isfinite(targets).all()),
        'FL targets must be the saved finite manually issued raw actions')
    cap = train['caps'][:,0]
    error = cap*(train['mean'][:,0].tanh()-targets.tanh())
    return {'schema':'wlr50_clean.finite_auxiliary_inspection.v1', 'label_scope':LABEL_SCOPE,
        'latest_same_state_network_mean_full12':_values(train['network_mean']),
        'actual_history_center_full12':_values(train['history']),
        'latest_same_state_conditional_mean_full12':_values(train['mean']),
        'latest_same_state_sigma_full12':_values(train['sigma']),
        'recorded_FL_raw_target':_values(targets),'FL_request_error_deg':_values(error),
        'FL_request_absolute_error_mean_deg':float(error.abs().mean()),
        'holdout_conditional_mean_full12':_values(holdout['mean']),
        'holdout_sigma_full12':_values(holdout['sigma']),
        'default_budget_linear_prediction':linear_budget_prediction(actor,train_obs,targets,holdout_obs),
        'automatic_aux_enabled':False,'PPO_decisions_added':0,'PPO_updates_added':0}


def fit_mean_row(runner, train_obs, target_raw_fl, holdout_obs, *, budget=Budget(), authorized=False):
    """Finite independent supervised SGD. Caller must authorize after inspection.

    This is not PPO. It changes row0 only, never critic/trunk/std/other rows or the
    existing PPO optimizer. A rejected trust-region step is rolled back in the
    temporary leaves and stops the procedure; policy actions are never clipped.
    """
    require(authorized is True, 'aux is disabled without explicit authorization')
    budget.validate()
    require(runner.alg.storage.step==0 and runner.alg.transition.actions is None,
        'aux requires a complete-update boundary and fresh empty rollout')
    actor, critic = runner.alg.actor, runner.alg.critic
    layer = mean_layer(actor)
    tx,hx = train_obs['policy'],holdout_obs['policy']
    require(tx.ndim==2 and tx.shape[1]==372 and 1<=len(tx)<=128
        and hx.ndim==2 and hx.shape[1]==372 and 1<=len(hx)<=128, 'invalid bounded data shape')
    require(bool((tx[:,:13].argmax(-1)==4).all()) and bool((tx[:,150]==1).all())
        and bool((tx[:,154]==0).all()) and bool((tx[:,131:133]==0).all()) and bool((tx[:,21]>0).all()),
        'training data must retain genuine P05 crossed AIR gap-pending state')
    require(bool((hx[:,:13].argmax(-1)!=4).all()), 'non-P05 holdout is required, not invented zero states')
    before = {k:v.detach().clone() for k,v in actor.state_dict().items()}
    actor_hash = training.parameter_hash(actor)
    critic_hash = training.parameter_hash(critic)
    optimizer_hash = training.state_hash(runner.alg.optimizer.state_dict())
    normalizer_hash = training.state_hash(training._normalizers(runner))
    source_rng = training.capture_training_rng_state(seed=runner.cfg['seed'])
    source_lr = (runner.alg.learning_rate,training.optimizer_learning_rate(runner))
    previous_mode = actor.training
    actor.eval()
    applied = False
    try:
        initial = inspect(actor,train_obs,target_raw_fl,holdout_obs)
        baseline = distribution(actor,train_obs); held = distribution(actor,holdout_obs)
        target = torch.as_tensor(target_raw_fl,dtype=layer.weight.dtype,device=layer.weight.device)
        weight = torch.nn.Parameter(layer.weight[:1].detach().clone())
        bias = torch.nn.Parameter(layer.bias[:1].detach().clone())
        optimizer = torch.optim.SGD([weight,bias],lr=budget.learning_rate,momentum=0.,weight_decay=0.)
        require(torch.equal(functional_mean(actor,train_obs,weight,bias),baseline['mean']),
            'functional row starts differently from the actual full actor')
        records=[]; stop='finite_budget_exhausted'
        for index in range(budget.max_steps):
            rate=budget.learning_rate*(budget.max_steps-index)/budget.max_steps
            optimizer.param_groups[0]['lr']=rate
            mu=functional_mean(actor,train_obs,weight,bias)
            loss=.5*(mu[:,0]-target).square().mean()
            require(bool(torch.isfinite(loss)), 'nonfinite auxiliary loss')
            if float(loss.detach())<1e-12:
                stop='conditional_mean_already_matches_recorded_local_target'; break
            grads=torch.autograd.grad(loss,(weight,bias))
            require(all(bool(torch.isfinite(g).all()) for g in grads), 'nonfinite auxiliary gradient')
            old_weight,old_bias=weight.detach().clone(),bias.detach().clone()
            weight.grad,bias.grad=grads
            optimizer.step()  # This is the independent two-leaf SGD, NOT PPO Adam.
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                proposed=functional_mean(actor,train_obs,weight,bias)
                hp=functional_mean(actor,holdout_obs,weight,bias)
                displacement=baseline['caps'][:,0]*(proposed[:,0].tanh()-baseline['mean'][:,0].tanh())
                hdisp=held['caps'][:,0]*(hp[:,0].tanh()-held['mean'][:,0].tanh())
                kl=.5*((proposed[:,0]-baseline['mean'][:,0])/baseline['sigma'][:,0]).square()
                hkl=.5*((hp[:,0]-held['mean'][:,0])/held['sigma'][:,0]).square()
                next_loss=.5*(proposed[:,0]-target).square().mean()
                finite=all(bool(torch.isfinite(v).all()) for v in (proposed,hp,displacement,hdisp,kl,hkl,next_loss))
                accepted=finite and float(displacement.abs().max())<=budget.maximum_train_request_shift_deg \
                    and float(hdisp.abs().max())<=budget.maximum_holdout_request_shift_deg \
                    and max(float(kl.max()),float(hkl.max()))<=budget.maximum_per_state_conditional_kl \
                    and float(next_loss)<=float(loss.detach())
                row={'attempt':index+1,'learning_rate':rate,'loss_before':float(loss.detach()),
                    'loss_after_candidate':float(next_loss),'accepted':accepted,
                    'max_train_request_shift_deg':float(displacement.abs().max()),
                    'max_holdout_request_shift_deg':float(hdisp.abs().max()),
                    'maximum_per_state_conditional_kl':max(float(kl.max()),float(hkl.max())),
                    'selected_gradient_l2':float(torch.cat([g.flatten() for g in grads]).norm()),
                    'gradient_semantics':'true conditional mean derivative; HISTORY rho=.9, chain=.1; no rescaling'}
                records.append(row)
                if not accepted:
                    weight.copy_(old_weight);bias.copy_(old_bias)
                    stop='candidate_step_rejected_by_finite_loss_or_trust_bound';break
        with torch.no_grad():
            layer.weight[:1].copy_(weight);layer.bias[:1].copy_(bias)
        applied=True
        after=inspect(actor,train_obs,target_raw_fl,holdout_obs)
        final,final_hold=distribution(actor,train_obs),distribution(actor,holdout_obs)
        for key,value in actor.state_dict().items():
            expected=before[key]
            require(torch.equal(value[1:],expected[1:]) if key in ('mlp.4.weight','mlp.4.bias')
                else torch.equal(value,expected), 'aux changed an unselected actor parameter/buffer')
        require(torch.equal(final['sigma'],baseline['sigma']) and torch.equal(final_hold['sigma'],held['sigma'])
            and torch.equal(final['mean'][:,1:],baseline['mean'][:,1:])
            and torch.equal(final_hold['mean'][:,1:],held['mean'][:,1:]), 'aux changed sigma or other11 same-input means')
        require(training.parameter_hash(critic)==critic_hash and training.state_hash(runner.alg.optimizer.state_dict())==optimizer_hash
            and training.state_hash(training._normalizers(runner))==normalizer_hash, 'aux changed critic/PPO Adam/normalizer')
        require((runner.alg.learning_rate,training.optimizer_learning_rate(runner))==source_lr
            and training.capture_training_rng_state(seed=runner.cfg['seed'])==source_rng, 'aux changed PPO LR or training RNG')
        return {'schema':'wlr50_clean.finite_FL_conditional_mean_aux.v1','label_scope':LABEL_SCOPE,
            'auxiliary_optimizer':'independent two-leaf SGD; momentum=0; weight_decay=0',
            'optimized_parameters':['actor.mlp.4.weight[0,:]','actor.mlp.4.bias[0]'],
            'optimized_scalar_count':257,'budget':asdict(budget),'steps':records,
            'accepted_auxiliary_updates':sum(r['accepted'] for r in records),
            'attempted_auxiliary_optimizer_steps':len(records),'stop_reason':stop,
            'before':initial,'after':after,'PPO_decisions_added':0,'PPO_updates_added':0,
            'PPO_optimizer_steps_added':0,'MDP_or_control_changed':False,'kernel_or_sigma_changed':False,
            'actor_parameter_sha256_before':actor_hash,
            'actor_parameter_sha256_after':training.parameter_hash(actor),
            'PPO_Adam_preserved_sha256':optimizer_hash,'PPO_LR_preserved':source_lr[0],
            'training_rng_preserved':True,'fresh_PPO_rollout_required':True,
            'teacher_deployed':False,'physical_success_claimed':False,
            'cross_state_note':'row0 is shared across phases; holdout bounds are not a physical success guarantee'}
    except BaseException:
        if applied:
            actor.load_state_dict(before,strict=True)
        raise
    finally:
        actor.train(previous_mode)


def append_ledger(infos, *, report, data_receipt, source_checkpoint, helper_sha256):
    """Persist explicit auxiliary lineage inside the already-copied branch dict."""
    require(report.get('accepted_auxiliary_updates',0)>0, 'do not save a no-update candidate as learned aux')
    result=copy.deepcopy(infos)
    branch=result.get('task_conditioned_hip_wheel_branch',{})
    require(branch.get('branch_id')==EXPERIMENT and isinstance(branch.get('counter_origin'),dict), 'missing original task branch')
    ledger=copy.deepcopy(branch.get(LEDGER_KEY,{'schema':'wlr50_clean.auxiliary_mean_learning_ledger.v1','events':[]}))
    require(ledger.get('schema')=='wlr50_clean.auxiliary_mean_learning_ledger.v1' and isinstance(ledger.get('events'),list),
        'invalid auxiliary ledger')
    event={'event_index':len(ledger['events'])+1,'kind':'limited_supervised_FL_conditional_mean_not_PPO',
        'source_checkpoint':copy.deepcopy(source_checkpoint),'helper_sha256':helper_sha256,
        'data_receipt':copy.deepcopy(data_receipt),'report':copy.deepcopy(report),
        'PPO_counters_unchanged':{k:infos[k] for k in ('global_policy_decisions','ppo_updates','optimizer_steps')},
        'stage_requested_decisions_unchanged':copy.deepcopy(infos['stage_requested_decisions'])}
    ledger['events'].append(event)
    ledger['accepted_auxiliary_updates_total']=sum(e['report']['accepted_auxiliary_updates'] for e in ledger['events'])
    ledger['attempted_auxiliary_optimizer_steps_total']=sum(e['report']['attempted_auxiliary_optimizer_steps'] for e in ledger['events'])
    ledger['training_lineage_label']='PPO_plus_explicit_finite_auxiliary_mean_supervision'
    branch[LEDGER_KEY]=ledger
    result['task_conditioned_hip_wheel_branch']=branch
    return result


def save_auxiliary_checkpoint(runner, path, infos, *, report, data_receipt, source_checkpoint):
    """Official full-state save/roundtrip, unique aux name, no pointer publishing."""
    path=Path(path).resolve()
    require(path.suffix=='.pt' and path.name.startswith('checkpoint_aux_flmean_')
        and not path.exists() and not path.with_name(path.stem+'_manifest.json').exists(),
        'aux checkpoint must have a new explicit auxiliary name')
    require(runner.alg.storage.step==0 and runner.alg.transition.actions is None, 'old rollout must not survive auxiliary training')
    require(infos['actor_parameter_sha256']==report['actor_parameter_sha256_before']
        and training.parameter_hash(runner.alg.actor)==report['actor_parameter_sha256_after'],
        'auxiliary source/current actor hashes differ from the reviewed fit')
    payload=append_ledger(infos,report=report,data_receipt=data_receipt,source_checkpoint=source_checkpoint,
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    payload['stage']='auxiliary_mean_pretrain_not_PPO'
    payload['resume_ancestry']={'source_checkpoint':copy.deepcopy(source_checkpoint),
        'source_global_policy_decisions':infos['global_policy_decisions'],'source_ppo_updates':infos['ppo_updates'],
        'source_optimizer_steps':infos['optimizer_steps'],'source_actor_parameter_sha256':infos['actor_parameter_sha256'],
        'source_runtime_contract':copy.deepcopy(infos['runtime_contract']),
        'operation':'explicit_auxiliary_mean_supervision_not_PPO'}
    return training.save_semantic_checkpoint(runner,path,payload)

"""One P06 FL Gaussian coordinate around the immutable complete A actor.

Other eleven raw outputs share the same frozen forward. Outside the observed
window all twelve are deterministic A outputs, without a new random draw.
"""
import copy
import json
import math
from pathlib import Path
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import unpad_trajectories
from wlr50_clean.ppo.semantic_rr_capture_local_actor import tensor_state_sha256
from wlr50_clean.ppo.semantic_finish_advance_actor import SemanticFinishAdvanceMLPModel
from .semantic_fl_forward_window import B_FIELDS, FL

FROZEN_DIMENSION=531
OBSERVATION_DIMENSION=FROZEN_DIMENSION+len(B_FIELDS)
OBSERVATION_LAYOUT='accepted531_p06_fl_scalar_v1'
POLICY_VERSION='frozen_A_p06_fl_conditional_scalar_v1'


class WindowFLDistribution(Distribution):
    """Full12 samples, scalar conditional Gaussian, deterministic inactive rows.

    params=(mu_FL[B,1], sigma_FL[B,1], active[B,1]). RSL storage lazily preserves
    these shapes independently of actions[B,12]. Inactive mu/sigma are positive
    placeholders only; no inactive Normal construction or random draw occurs.
    """
    def __init__(self, output_dim, init_std=.03):
        super().__init__(output_dim)
        if output_dim != 12 or not 0. < init_std < 1.:
            raise ValueError('full12 interface, explicit modest scalar std required')
        self.init_std=init_std
        self._mean=self._sigma=self._active=None

    @property
    def input_dim(self):
        return [2,1]

    def update(self, conditional):
        mean,sigma,active=conditional
        if (mean.ndim != 2 or mean.shape[-1] != 12 or sigma.shape != mean.shape[:-1]+(1,)
                or active.shape != sigma.shape or not bool(((active==0)|(active==1)).all())
                or not bool(torch.isfinite(mean).all())
                or not bool((torch.isfinite(sigma)&(sigma>0)).all())):
            raise ValueError('finite full12 mean, positive scalar std, explicit Boolean mask required')
        self._mean,self._sigma,self._active=mean,sigma,active.bool()

    def sample(self):
        result=self._mean.clone()
        take=self._active[:,0]
        if bool(take.any()):
            # No12D Normal, no zero std, one scalar draw per active row only.
            result[take,FL]=torch.distributions.Normal(
                self._mean[take,FL],self._sigma[take,0]).sample()
        return result

    def deterministic_output(self, conditional):
        return conditional[0]

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._sigma  # Stochastic coordinate only, never full12 std.

    @property
    def params(self):
        return self._mean[:,FL:FL+1],self._sigma,self._active.to(self._mean.dtype)

    def log_prob(self, outputs):
        if outputs.shape != self._mean.shape or not bool(torch.isfinite(outputs).all()):
            raise ValueError('the original stored full12 raw actions are required')
        # The frozen network's minibatch kernel can differ by a few float32
        # roundoff units from singleton collection. This tolerance is support
        # verification ONLY, never an action change or likelihood approximation.
        fixed=torch.ones_like(outputs,dtype=torch.bool)
        fixed[:,FL]=~self._active[:,0]
        self.last_deterministic_support_max_abs=float((outputs[fixed]-self._mean[fixed]).detach().abs().max())
        if not torch.allclose(outputs[fixed],self._mean[fixed],atol=2.e-6,rtol=1.e-6):
            raise ValueError('raw action is outside the frozen deterministic support')
        result=self._mean[:,FL]*0.
        take=self._active[:,0]
        if bool(take.any()):
            result=result.clone()
            result[take]=torch.distributions.Normal(self._mean[take,FL],self._sigma[take,0]).log_prob(outputs[take,FL])
        return result

    @property
    def entropy(self):
        result=self._mean[:,FL]*0.
        take=self._active[:,0]
        if bool(take.any()):
            result=result.clone()
            result[take]=torch.distributions.Normal(self._mean[take,FL],self._sigma[take,0]).entropy()
        return result

    def kl_divergence(self, old_params, new_params):
        old_mean,old_std,old_active=old_params
        new_mean,new_std,new_active=new_params
        if (old_mean.shape != new_mean.shape or old_mean.shape[-1] != 1
                or not torch.equal(old_active,new_active)):
            raise ValueError('scalar KL requires the identical observed window mask')
        result=new_mean[:,0]*0.
        take=new_active[:,0].bool()
        if bool(take.any()):
            old=torch.distributions.Normal(old_mean[take,0],old_std[take,0])
            new=torch.distributions.Normal(new_mean[take,0],new_std[take,0])
            result=result.clone()
            result[take]=torch.distributions.kl_divergence(old,new)
        return result

    def as_deterministic_output_module(self):
        raise NotImplementedError('window-routed export not implemented')


class ConstantFLHead(torch.nn.Module):
    """Two new scalars: conditional raw mean offset and log conditional std."""
    def __init__(self,initial_std):
        super().__init__()
        self.mean_delta=torch.nn.Parameter(torch.tensor(0.))
        self.log_std=torch.nn.Parameter(torch.tensor(initial_std).log())

    def forward(self,latent):
        return torch.stack((self.mean_delta,self.log_std)).view(1,2,1).expand(latent.shape[0],2,1)


class WindowFLActor(MLPModel):
    """Serializable standard RSL constructor; exactly two trainable scalars."""
    policy_version=POLICY_VERSION

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(),
                 activation='elu', obs_normalization=False, distribution_cfg=None,
                 observation_layout=OBSERVATION_LAYOUT, frozen_A_configuration=None,
                 expected_frozen_A_sha256=None, initial_std=.03):
        if (obs['policy'].shape[-1] != OBSERVATION_DIMENSION or output_dim != 12
                or obs_set!='actor' or obs_groups.get('actor')!=['policy']
                or observation_layout!=OBSERVATION_LAYOUT or obs_normalization is not False
                or tuple(hidden_dims)!=()):
            raise ValueError('explicit accepted531+4 / scalar-head / Identity layout required')
        if distribution_cfg not in (None,{'class_name':'WindowFLDistribution','init_std':initial_std}):
            raise ValueError('no hidden12D distribution or noise configuration permitted')
        super().__init__(obs,obs_groups,obs_set,12,(1,),activation,False,
                         {'class_name':WindowFLDistribution,'init_std':initial_std})
        self.mlp=ConstantFLHead(initial_std)
        cfg=copy.deepcopy(frozen_A_configuration)
        if not isinstance(cfg,dict) or cfg.get('observation_layout')!='accepted490_post_rl_advance_home_v1':
            raise ValueError('actual complete accepted A construction recipe required')
        cfg.pop('class_name',None)
        self.frozen_A=SemanticFinishAdvanceMLPModel({'policy':obs['policy'][...,:FROZEN_DIMENSION]},
            {'actor':['policy']},'actor',12,**cfg)
        self.frozen_input_dim=FROZEN_DIMENSION
        self.observation_dimension=OBSERVATION_DIMENSION
        self.observation_layout=OBSERVATION_LAYOUT
        self.expected_frozen_A_sha256=expected_frozen_A_sha256
        self._frozen_snapshot=None
        self._freeze()
        self._last_forward_evidence=None

    def _remember_A(self):
        digest=tensor_state_sha256(self.frozen_A.state_dict())
        if self.expected_frozen_A_sha256 not in (None,digest):
            raise ValueError('complete A tensor hash mismatch')
        self.expected_frozen_A_sha256=digest
        self._frozen_snapshot={k:v.detach().cpu().clone() for k,v in self.frozen_A.state_dict().items()}
        self._freeze()
        return digest

    def load_A_state(self,state,*,expected_sha256):
        if self._frozen_snapshot is not None or tensor_state_sha256(state)!=expected_sha256:
            raise ValueError('one-time strict complete A state load required')
        if self.expected_frozen_A_sha256 not in (None,expected_sha256):
            raise ValueError('A identity differs from constructor binding')
        self.expected_frozen_A_sha256=expected_sha256
        self.frozen_A.load_state_dict(state,strict=True)
        return self._remember_A()

    def _freeze(self):
        self.frozen_A.eval(); self.frozen_A.requires_grad_(False)
        for p in self.frozen_A.parameters(): p.grad=None

    def train(self,mode=True):
        super().train(mode)
        if hasattr(self,'frozen_A'): self._freeze()
        return self

    def update_normalization(self,obs):
        if type(self.obs_normalizer) is not torch.nn.Identity:
            raise RuntimeError('B prototype must not alter normalizers')
        # No call that could update any frozen normalizer.

    def trainable_parameters(self):
        return tuple(p for p in self.parameters() if p.requires_grad)

    def assert_frozen_state(self,optimizer=None):
        if self._frozen_snapshot is None:
            raise RuntimeError('verified A state must be loaded before use')
        if self.frozen_A.training or any(p.requires_grad or p.grad is not None for p in self.frozen_A.parameters()):
            raise RuntimeError('frozen A acquired training/gradient state')
        if any(not torch.equal(v.detach().cpu(),self._frozen_snapshot[k]) for k,v in self.frozen_A.state_dict().items()):
            raise RuntimeError('frozen A tensors changed')
        if optimizer is not None:
            frozen={id(p) for p in self.frozen_A.parameters()}
            if any(id(p) in frozen for g in optimizer.param_groups for p in g['params']):
                raise RuntimeError('frozen A cannot enter B optimizer')
        self.frozen_A.assert_frozen_state()
        return self.expected_frozen_A_sha256

    def load_state_dict(self,state_dict,strict=True,assign=False):
        state={k[len('frozen_A.'):]:v for k,v in state_dict.items() if k.startswith('frozen_A.')}
        if (strict is not True or assign is not False or not state
                or self.expected_frozen_A_sha256 not in (None,tensor_state_sha256(state))):
            raise ValueError('strict immutable complete-A load required')
        result=super().load_state_dict(state_dict,strict=True,assign=False)
        # The actual nested composite needs its public strict loader too.
        self.frozen_A.load_state_dict(state,strict=True)
        self._remember_A(); self.assert_frozen_state()
        return result

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        if self._frozen_snapshot is None:
            raise RuntimeError('verified A state must be loaded before forward')
        obs=unpad_trajectories(obs,masks) if masks is not None else obs
        latent=self.get_latent(obs,masks,hidden_state)
        tail=latent[:,self.frozen_input_dim:]
        flags=tail[:,:1]
        if (not bool(torch.isfinite(latent).all()) or not bool(((flags==0)|(flags==1)).all())
                or not bool((flags.sum(-1)<=1).all())):
            raise ValueError('finite observation and explicit P06 window Boolean required')
        active=flags.any(-1,keepdim=True)
        with torch.no_grad():
            accepted=self.frozen_A({'policy':latent[:,:self.frozen_input_dim]},stochastic_output=False)
        mean=accepted.clone()
        sigma=torch.full_like(mean[:,:1],self.distribution.init_std)
        innovation=torch.zeros_like(sigma)
        if bool(active.any()):
            head=self.mlp(latent)
            innovation=head[:,0,:]
            sigma=head[:,1,:].exp()
            # Accepted A has ALREADY applied the existing HISTORY kernel.
            # Delta unit = conditional raw mean. No second HISTORY pass,
            # (1-rho) multiplier, or implicit gain10. No rate/filter reset.
            mean[:,FL]=torch.where(active[:,0],accepted[:,FL]+innovation[:,0],accepted[:,FL])
        self._last_forward_evidence=dict(active=active.detach().clone(),accepted_raw=accepted.detach().clone(),
            conditional_mean=mean.detach().clone(),conditional_std_FL=sigma.detach().clone(),
            innovation_FL=innovation.detach().clone(),density_dimension=1,stochastic_channel=FL,
            history_kernel_applications=1,other11_from_same_frozen_forward=True,
            delta_units='conditional_raw_mean_increment_no_extra_gain',
            runtime_raw_overwrite=False)
        conditional=(mean,sigma,active.to(mean.dtype))
        if stochastic_output:
            self.distribution.update(conditional)
            return self.distribution.sample()
        return self.distribution.deterministic_output(conditional)


def audited_fl_request(actor,observation,action_call,*,stochastic):
    if type(actor) is not WindowFLActor:
        raise ValueError('explicit scalar FL actor required')
    seen=[]
    handle=actor.register_forward_hook(lambda *_:seen.append(actor._last_forward_evidence))
    try: raw=action_call()
    finally: handle.remove()
    if len(seen)!=1 or raw.shape!=(1,12):
        raise RuntimeError('exactly one full12 action forward required')
    evidence=seen[0]; active=bool(evidence['active'][0,0]); other=[i for i in range(12) if i!=FL]
    if not torch.equal(raw[:,other],evidence['accepted_raw'][:,other]):
        raise RuntimeError('B request changed a frozen coordinate')
    if stochastic:
        mean,std,mask=actor.output_distribution_params
        if (not torch.equal(mean,evidence['conditional_mean'][:,FL:FL+1])
                or not torch.equal(std,evidence['conditional_std_FL'])
                or not torch.equal(mask,evidence['active'].to(mean.dtype))):
            raise RuntimeError('scalar likelihood cache differs from actual requested Gaussian')
        logp=float(actor.get_output_log_prob(raw)[0].detach().cpu())
    else:
        if not torch.equal(raw,evidence['conditional_mean']):
            raise RuntimeError('deterministic raw differs from actual conditional mean')
        logp=None
    vector=lambda value:value[0].detach().cpu().tolist()
    return raw,dict(schema='wlr50_clean.p06_FL_scalar_request.v1',policy_version=POLICY_VERSION,
        selected_raw_full12=vector(raw),accepted_raw_full12=vector(evidence['accepted_raw']),
        conditional_mean_full12=vector(evidence['conditional_mean']),
        conditional_mean_FL_raw=float(evidence['conditional_mean'][0,FL].detach().cpu()),
        conditional_std_FL_raw=float(evidence['conditional_std_FL'][0,0].detach().cpu()) if active else None,
        mean_delta_conditional_raw=float(evidence['innovation_FL'][0,0].detach().cpu()),
        active=active,selected_raw_log_probability=logp,stochastic_coordinate=FL,density_dimension=1,
        stochastic_FL_credit=int(stochastic and active),sampling_draws=int(stochastic and active),
        history_kernel_applications=1,extra_rho_or_gain_multiplier=False,other11_frozen=True,
        A_state_sha256=actor.expected_frozen_A_sha256,
        inactive_semantics='deterministic_frozen_A;conditional_density1_logp0_entropy0')


def _same_state(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict): return isinstance(b,dict) and a.keys()==b.keys() and all(_same_state(a[k],b[k]) for k in a)
    return a==b


def audited_fl_ppo_update(runner,*,likelihood_audit_path=None):
    """Observe the OFFICIAL RSL update; no12D-normal or replacement PPO math."""
    alg=runner.alg; actor=alg.actor; actor.assert_frozen_state(alg.optimizer)
    original_generator=alg.storage.mini_batch_generator
    original_logp=actor.get_output_log_prob; original_kl=actor.get_kl_divergence
    context={}; records=[]; steps=[]; live=[]
    stored_rows=int(alg.storage.step)
    active_rows=int(alg.storage.distribution_params[2].sum().item())
    def batches(*args,**kwargs):
        it=original_generator(*args,**kwargs);live.append(it)
        for batch in it:
            context['batch']=batch
            yield batch
    def logp(raw):
        batch=context['batch']
        if not torch.equal(raw,batch.actions): raise RuntimeError('not the stored raw12 sample')
        result=original_logp(raw)
        mu,sigma,active=actor.output_distribution_params
        if not torch.equal(active,batch.old_distribution_params[2]):
            raise RuntimeError('window mask changed for a saved observation')
        pack=lambda value:value.detach().cpu().tolist()
        records.append(dict(raw_full12=pack(raw),old_logp=pack(batch.old_actions_log_prob.squeeze(-1)),
            current_logp=pack(result),mean_FL=pack(mu),std_FL=pack(sigma),active=pack(active),
            ratio=pack(torch.exp(result-batch.old_actions_log_prob.squeeze(-1))),
            advantage=pack(batch.advantages.squeeze(-1)),density_dimension=1,
            deterministic_support_max_abs=actor.distribution.last_deterministic_support_max_abs))
        return result
    def kl(old,new):
        result=original_kl(old,new);take=new[2][:,0].bool()
        context['kl_all']=float(result.detach().mean())
        context['kl_active']=float(result[take].detach().mean()) if bool(take.any()) else None
        return result
    def before(optimizer,args,kwargs):
        all_inactive=not bool(context['batch'].old_distribution_params[2].any())
        params=actor.trainable_parameters()
        if all_inactive and any(p.grad is not None for p in params):
            raise RuntimeError('inactive-only batch must not feed zero gradients to Adam momentum')
        context['before']=[(p.detach().clone(),copy.deepcopy(optimizer.state.get(p,{}))) for p in params]
        context['step']=dict(all_inactive=all_inactive,actor_gradient_present=any(p.grad is not None for p in params),
            actor_gradient_nonzero=any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in params),
            kl_all=context.get('kl_all'),kl_active=context.get('kl_active'),actual_learning_rate=optimizer.param_groups[0]['lr'])
    def after(optimizer,args,kwargs):
        record=context['step']
        if record['all_inactive']:
            if any(not torch.equal(p.detach(),v) or not _same_state(optimizer.state.get(p,{}),s)
                   for p,(v,s) in zip(actor.trainable_parameters(),context['before'])):
                raise RuntimeError('inactive-only Adam moved B scalar or its moments')
        steps.append(record)
    pre=alg.optimizer.register_step_pre_hook(before);post=alg.optimizer.register_step_post_hook(after)
    try:
        alg.storage.mini_batch_generator=batches;actor.get_output_log_prob=logp;actor.get_kl_divergence=kl
        loss=alg.update()
    finally:
        for iterator in live:iterator.close()
        pre.remove();post.remove()
        alg.storage.mini_batch_generator=original_generator
        actor.get_output_log_prob=original_logp;actor.get_kl_divergence=original_kl
    if len(steps)!=alg.num_learning_epochs*alg.num_mini_batches or len(records)!=len(steps):
        raise RuntimeError('official scalar PPO instrumentation count mismatch')
    actor.assert_frozen_state(alg.optimizer)
    report=dict(schema='wlr50_clean.p06_FL_scalar_PPO_update.v1',density_dimension=1,
        total_continuous_storage_rows=stored_rows,stochastic_FL_rows=active_rows,
        critic_only_context_rows=stored_rows-active_rows,optimizer_steps=len(steps),
        actor_optimizer_steps=sum(s['actor_gradient_present'] for s in steps),
        actor_gradient_steps=sum(s['actor_gradient_present'] for s in steps),
        actor_nonzero_gradient_steps=sum(s['actor_gradient_nonzero'] for s in steps),
        inactive_only_steps=sum(s['all_inactive'] for s in steps),steps=steps,
        actual_learning_rate=float(alg.learning_rate),loss={k:float(v) for k,v in loss.items()},
        frozen_A_sha256=actor.expected_frozen_A_sha256,minibatches=records,
        extra_model_forwards=0,extra_random_draws=0)
    if not all(math.isfinite(v) for v in report['loss'].values()): raise RuntimeError('nonfinite official PPO loss')
    if likelihood_audit_path is not None:
        Path(likelihood_audit_path).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report

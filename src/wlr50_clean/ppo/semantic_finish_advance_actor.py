"""One saved CP232960 composite with a post-RL-only finish innovation head.

The accepted490 function and its normalizers remain immutable, including its
learned post-RR head. The new Gaussian is sampled only after measured RL TOP.
REQUEST history is not reset; its observed physical reference defines the new
finish coordinate. Public advance/home targets remain controller contributions.
"""
from __future__ import annotations

import copy
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from rsl_rl.utils import unpad_trajectories
from .semantic_history_actor import HISTORY_RHO
from .semantic_p05_capture_actor import p05_capture_request_history
from .semantic_post_rr_front_prep_actor import SemanticPostRRHistoryMLPModel
from .semantic_rr_capture_local_actor import tensor_state_sha256
from .semantic_finish_advance_task import FINISH_FIELDS

ACCEPTED_DIMENSION = 490
OBSERVATION_DIMENSION = ACCEPTED_DIMENSION + len(FINISH_FIELDS)
OBSERVATION_LAYOUT = 'accepted490_post_rl_advance_home_v1'
POLICY_VERSION = 'frozen_cp232960_advance_then_home_v1'
REFERENCE_FIELDS = tuple('finish_request_ref_ratio_'+str(i) for i in range(12))


class SemanticFinishAdvanceMLPModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(128,128),
                 activation='elu', obs_normalization=False, distribution_cfg=None,
                 observation_layout=OBSERVATION_LAYOUT, frozen_accepted_configuration=None,
                 expected_accepted_state_sha256=None, initial_finish_std=(.04,)*12):
        if (observation_layout != OBSERVATION_LAYOUT or output_dim != 12
                or obs_set != 'actor' or obs_groups.get('actor') != ['policy']
                or obs['policy'].shape[-1] != OBSERVATION_DIMENSION
                or obs_normalization is not False or not isinstance(distribution_cfg,dict)
                or distribution_cfg.get('class_name') != 'HeteroscedasticGaussianDistribution'
                or distribution_cfg.get('std_type') != 'log'):
            raise ValueError('explicit accepted490 + finish Full12 Identity Gaussian required')
        super().__init__(obs,obs_groups,obs_set,output_dim,hidden_dims,activation,
                         obs_normalization,copy.deepcopy(distribution_cfg))
        if type(self.distribution) is not HeteroscedasticGaussianDistribution:
            raise ValueError('official conditional Gaussian required')
        cfg = copy.deepcopy(frozen_accepted_configuration)
        if not isinstance(cfg,dict) or cfg.get('observation_layout') != 'accepted465_post_rr_front_pair490_v2':
            raise ValueError('complete accepted CP232960 actor construction recipe required')
        cfg.pop('class_name',None)
        self.frozen_accepted = SemanticPostRRHistoryMLPModel(
            {'policy':obs['policy'][...,:490]}, {'actor':['policy']},'actor',12,**cfg)
        self.expected_accepted_state_sha256 = expected_accepted_state_sha256
        self._accepted_state = None
        self._last_forward_evidence = None
        self.policy_version = POLICY_VERSION
        self.observation_layout = OBSERVATION_LAYOUT
        self.observation_dimension = OBSERVATION_DIMENSION
        sigma = torch.as_tensor(initial_finish_std,dtype=torch.float32)
        if sigma.shape != (12,) or not bool((torch.isfinite(sigma)&(sigma>0)).all()):
            raise ValueError('twelve positive finish innovation std values required')
        last = [m for m in self.mlp.modules() if isinstance(m,torch.nn.Linear)][-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias[:12].zero_()
            last.bias[12:].copy_(sigma.to(last.bias).log())
        self._freeze_accepted()

    def _freeze_accepted(self):
        self.frozen_accepted.requires_grad_(False)
        self.frozen_accepted.eval()
        for p in self.frozen_accepted.parameters():
            p.grad = None

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self,'frozen_accepted'):
            self._freeze_accepted()
        return self

    def update_normalization(self, obs):
        if self.obs_normalization or type(self.obs_normalizer) is not torch.nn.Identity:
            raise RuntimeError('finish normalizer must remain Identity')
        self.frozen_accepted.update_normalization(None)

    def trainable_parameters(self):
        return tuple(p for p in self.parameters() if p.requires_grad)

    def _remember_accepted(self):
        self._freeze_accepted()
        state = self.frozen_accepted.state_dict()
        digest = tensor_state_sha256(state)
        if self.expected_accepted_state_sha256 not in (None,digest):
            raise ValueError('complete accepted490 tensor identity changed')
        self.expected_accepted_state_sha256 = digest
        self._accepted_state = {k:v.detach().cpu().clone() for k,v in state.items()}
        return digest

    def load_accepted_state(self, state, *, expected_sha256):
        if self._accepted_state is not None or tensor_state_sha256(state) != expected_sha256:
            raise ValueError('one immutable complete accepted state required')
        self.frozen_accepted.load_state_dict(state,strict=True)
        return self._remember_accepted()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict is not True or assign is not False:
            raise ValueError('finish restore requires strict stable parameter objects')
        accepted = {k[len('frozen_accepted.'):]:v for k,v in state_dict.items()
                    if k.startswith('frozen_accepted.')}
        if not accepted or self.expected_accepted_state_sha256 not in (None,tensor_state_sha256(accepted)):
            raise ValueError('saved accepted490 binding differs')
        result = super().load_state_dict(state_dict,strict=True,assign=False)
        # Recursive Torch loading skips each child's public loader.
        self.frozen_accepted.load_state_dict(accepted,strict=True)
        self._remember_accepted()
        return result

    def assert_frozen_state(self, optimizer=None):
        if self._accepted_state is None:
            raise RuntimeError('load verified accepted490 before requesting actions')
        self.update_normalization(None)
        state = self.frozen_accepted.state_dict()
        if self.frozen_accepted.training or any(not torch.equal(v.detach().cpu(),self._accepted_state[k])
                                               for k,v in state.items()):
            raise RuntimeError('accepted complete CP232960 actor changed')
        params = tuple(self.frozen_accepted.parameters())
        if any(p.requires_grad or p.grad is not None for p in params):
            raise RuntimeError('accepted actor acquired gradients')
        if optimizer is not None:
            frozen = {id(p) for p in params}
            if any(id(p) in frozen for g in optimizer.param_groups for p in g['params']):
                raise RuntimeError('accepted actor must be excluded from finish Adam')
        self.frozen_accepted.assert_frozen_state()
        return self.expected_accepted_state_sha256

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if self._accepted_state is None:
            raise RuntimeError('accepted490 state not bound')
        obs = unpad_trajectories(obs,masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs,masks,hidden_state)
        if latent.shape[-1] != OBSERVATION_DIMENSION or not bool(torch.isfinite(latent).all()):
            raise ValueError('finite explicit finish observations required')
        tail = latent[...,490:]
        flag = tail[...,0]
        if not bool(((flag==0)|(flag==1)).all()):
            raise ValueError('finish activation must be a measured event Boolean')
        active = flag.bool()
        if stochastic_output and not bool(active.all()):
            raise ValueError('no new sampling or PPO credit before legal RL placement')
        with torch.no_grad():
            accepted = self.frozen_accepted({'policy':latent[...,:490]},stochastic_output=False)
        center,_ = p05_capture_request_history(latent[...,:389])
        if bool(active.any()):
            new = self.mlp(latent)
            delta,log_std = new[...,0,:],new[...,1,:]
            ratios = torch.stack([tail[...,FINISH_FIELDS.index(n)] for n in REFERENCE_FIELDS],dim=-1)
            if not bool((ratios[active].abs()<1.).all()):
                raise ValueError('finish REQUEST reference must have finite inverse tanh')
            # Keep the actual conditional HISTORY kernel. New coordinates begin
            # at the current committed REQUEST, not raw=0 or the old home bias.
            refs = torch.atanh(ratios)
            finish_mean = HISTORY_RHO*center+(1.-HISTORY_RHO)*(refs+delta)
            mean = torch.where(active.unsqueeze(-1),finish_mean,accepted)
        else:
            delta = torch.zeros_like(accepted)
            mean = accepted  # Literal identity before the authorized event.
            log_std = self.frozen_accepted._last_forward_evidence['head_log_std']
        conditional = torch.stack((mean,log_std),dim=-2)
        self._last_forward_evidence = dict(active=active.detach().clone(),conditional_mean=mean.detach().clone(),
            head_log_std=log_std.detach().clone(),accepted_mean=accepted.detach().clone(),
            finish_delta=delta.detach().clone(),history_center=center.detach().clone())
        if stochastic_output:
            self.distribution.update(conditional)
            return self.distribution.sample()
        return self.distribution.deterministic_output(conditional)

    def as_jit(self):
        raise NotImplementedError('event-routed export is not validated')

    def as_onnx(self, verbose=False):
        raise NotImplementedError('event-routed export is not validated')


def audited_finish_request(actor, observation, action_call, *, stochastic):
    if type(actor) is not SemanticFinishAdvanceMLPModel:
        raise ValueError('finish packaged actor required')
    seen=[]
    hook=actor.register_forward_hook(lambda _m,_i,_o:seen.append(actor._last_forward_evidence))
    try:
        raw=action_call()
    finally:
        hook.remove()
    if len(seen)!=1 or raw.shape!=(1,12):
        raise RuntimeError('exactly one actual action forward required')
    e=seen[0]; active=bool(e['active'][0])
    if stochastic:
        mean,std=actor.output_distribution_params
        if not active or not torch.equal(mean,e['conditional_mean']) or not torch.equal(std,e['head_log_std'].exp()):
            raise RuntimeError('likelihood differs from the actual conditional Gaussian')
        logp=float(actor.get_output_log_prob(raw)[0].detach().cpu())
    else:
        if not torch.equal(raw,e['conditional_mean']):
            raise RuntimeError('deterministic output differs from actual mean')
        logp=None
    vec=lambda x:x[0].detach().cpu().tolist()
    return raw,dict(schema='wlr50_clean.finish_advance_actual_request.v1',policy_version=actor.policy_version,
        finish_active=active,mode='stochastic' if stochastic else 'deterministic',
        selected_raw_full12=vec(raw),selected_tanh_full12=vec(torch.tanh(raw)),selected_raw_log_probability=logp,
        conditional_mean_full12=vec(e['conditional_mean']),accepted_mean_full12=vec(e['accepted_mean']),
        finish_raw_delta_full12=vec(e['finish_delta']),history_center_full12=vec(e['history_center']),
        active_conditional_std_full12=vec(e['head_log_std'].exp()) if active else None,
        accepted_complete_state_sha256=actor.expected_accepted_state_sha256,
        prefix_excluded_from_new_PPO_credit=not active,sampling_draws=int(stochastic),
        history_kernel_applications=1,gain10_applied_again=False,
        public_finish_reference_is_not_neural_output=True)

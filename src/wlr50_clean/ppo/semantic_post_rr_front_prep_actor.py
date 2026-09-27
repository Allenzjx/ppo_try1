"""One packaged, frozen accepted465 predecessor and an event-routed RL head.

The public four-channel preparation reference is a control contribution, not
learned output. Its relative REQUEST origin is visible in the observation.
The underlying physical limits, single REQUEST history and Gaussian likelihood
are unchanged. No checkpoint is loaded or switched during an episode.
"""
from __future__ import annotations

import copy
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from rsl_rl.utils import unpad_trajectories
from .semantic_history_actor import HISTORY_RHO
from .semantic_p05_capture_actor import p05_capture_request_history
from .semantic_rr_capture_local_actor import (
    SemanticRRCaptureLocalHistoryMLPModel, tensor_state_sha256,
    validate_capture_local_latent,
)
from .semantic_post_rr_front_prep_task import POST_RR_FIELDS

POLICY_VERSION = 'frozen_cp231936_post_rr_front_pair_v1'
OBSERVATION_LAYOUT = 'accepted465_post_rr_front_pair487_v1'
ANCHOR_DIMENSION = 465
OBSERVATION_DIMENSION = ANCHOR_DIMENSION + len(POST_RR_FIELDS)
REBASE_INDICES = (1, 3, 8, 9)
RATIO_FIELDS = ('entry_request_ratio_FL_knee', 'entry_request_ratio_FR_knee',
                'entry_request_ratio_FL_wheel', 'entry_request_ratio_FR_wheel')
DEFAULT_STD = (.045, .06, .04, .04, .05, .07, .12, .12, .045, .045, .05, .05)


class SemanticPostRRHistoryMLPModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(128, 128),
                 activation='elu', obs_normalization=False, distribution_cfg=None,
                 observation_layout=OBSERVATION_LAYOUT,
                 frozen_anchor_configuration=None, initial_post_std=DEFAULT_STD,
                 expected_anchor_state_sha256=None):
        if (observation_layout != OBSERVATION_LAYOUT or output_dim != 12
                or obs_set != 'actor' or obs_groups.get('actor') != ['policy']
                or obs['policy'].shape[-1] != OBSERVATION_DIMENSION
                or obs_normalization is not False
                or not isinstance(distribution_cfg, dict)
                or distribution_cfg.get('class_name') != 'HeteroscedasticGaussianDistribution'
                or distribution_cfg.get('std_type') != 'log'):
            raise ValueError('post-RR actor requires explicit487/Full12/Identity Gaussian')
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims,
                         activation, obs_normalization, copy.deepcopy(distribution_cfg))
        if type(self.distribution) is not HeteroscedasticGaussianDistribution:
            raise ValueError('official heteroscedastic distribution required')
        anchor_cfg = copy.deepcopy(frozen_anchor_configuration)
        if not isinstance(anchor_cfg, dict) or anchor_cfg.get('continuation') is not True:
            raise ValueError('complete saved465 anchor construction recipe required')
        anchor_cfg.pop('class_name', None)
        self.frozen_anchor = SemanticRRCaptureLocalHistoryMLPModel(
            {'policy': obs['policy'][..., :465]}, {'actor': ['policy']}, 'actor', 12,
            **anchor_cfg)
        self.expected_anchor_state_sha256 = expected_anchor_state_sha256
        self._anchor_state = None
        self.policy_version = POLICY_VERSION
        self.observation_layout = observation_layout
        self.observation_dimension = OBSERVATION_DIMENSION
        self._last_forward_evidence = None
        std = torch.as_tensor(initial_post_std, dtype=torch.float32)
        if std.shape != (12,) or not bool((torch.isfinite(std) & (std > 0)).all()):
            raise ValueError('twelve positive post-RR innovation standard deviations required')
        last = [m for m in self.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
        if last.out_features != 24:
            raise ValueError('new post head must contain mean12 and logstd12')
        with torch.no_grad():
            last.weight.zero_()
            last.bias[:12].zero_()
            last.bias[12:].copy_(std.to(last.bias).log())
        self._freeze_anchor()

    def _freeze_anchor(self):
        self.frozen_anchor.requires_grad_(False)
        self.frozen_anchor.eval()
        for p in self.frozen_anchor.parameters():
            p.grad = None

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'frozen_anchor'):
            self._freeze_anchor()
        return self

    def update_normalization(self, obs):
        if self.obs_normalization or type(self.obs_normalizer) is not torch.nn.Identity:
            raise RuntimeError('new and accepted normalizers must remain Identity')
        self.frozen_anchor.update_normalization(None)

    def trainable_parameters(self):
        result = tuple(p for p in self.parameters() if p.requires_grad)
        frozen = {id(p) for p in self.frozen_anchor.parameters()}
        if not result or any(id(p) in frozen for p in result):
            raise RuntimeError('accepted predecessor leaked into optimizer')
        return result

    def _remember_anchor(self):
        self._freeze_anchor()
        state = self.frozen_anchor.state_dict()
        digest = tensor_state_sha256(state)
        if self.expected_anchor_state_sha256 not in (None, digest):
            raise ValueError('accepted full465 anchor hash changed')
        self.expected_anchor_state_sha256 = digest
        self._anchor_state = {k: v.detach().cpu().clone() for k, v in state.items()}
        return digest

    def load_frozen_anchor_state(self, state, *, expected_sha256=None):
        if self._anchor_state is not None:
            raise RuntimeError('anchor already bound; cannot replace it during continuation')
        digest = tensor_state_sha256(state)
        if expected_sha256 not in (None, digest):
            raise ValueError('declared anchor state hash mismatch')
        if self.expected_anchor_state_sha256 not in (None, digest):
            raise ValueError('constructor anchor state hash mismatch')
        self.frozen_anchor.load_state_dict(state, strict=True)
        return self._remember_anchor()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict is not True or assign is not False:
            raise ValueError('post-RR restore must preserve exact parameter objects')
        anchor = {k[len('frozen_anchor.'):]: v for k, v in state_dict.items()
                  if k.startswith('frozen_anchor.')}
        if not anchor or self.expected_anchor_state_sha256 not in (None, tensor_state_sha256(anchor)):
            raise ValueError('saved package must contain the same accepted predecessor')
        result = super().load_state_dict(state_dict, strict=True, assign=False)
        # Recursive torch loading does not call a child's public load_state_dict.
        self.frozen_anchor._remember_prior()
        self._remember_anchor()
        return result

    def assert_frozen_state(self, optimizer=None):
        if self._anchor_state is None:
            raise RuntimeError('accepted checkpoint has not been verified and loaded')
        self.update_normalization(None)
        state = self.frozen_anchor.state_dict()
        if self.frozen_anchor.training or any(not torch.equal(v.detach().cpu(), self._anchor_state[k])
                                             for k, v in state.items()):
            raise RuntimeError('accepted complete predecessor changed')
        params = tuple(self.frozen_anchor.parameters())
        if any(p.requires_grad or p.grad is not None for p in params):
            raise RuntimeError('accepted predecessor acquired gradients')
        if optimizer is not None:
            ids = {id(p) for p in params}
            if any(id(p) in ids for g in optimizer.param_groups for p in g['params']):
                raise RuntimeError('accepted predecessor present in Adam/weight decay')
        self.frozen_anchor.assert_frozen_state()
        return self.expected_anchor_state_sha256

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if self._anchor_state is None:
            raise RuntimeError('load immutable full465 anchor before requesting actions')
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        if latent.shape[-1] != OBSERVATION_DIMENSION or not bool(torch.isfinite(latent).all()):
            raise ValueError('finite post-RR observation schema required')
        validate_capture_local_latent(latent[..., :465], continuation=True)
        tail = latent[..., 465:]
        flag = tail[..., 0]
        if not bool(((flag == 0) | (flag == 1)).all()):
            raise ValueError('post-touch activation is a physical-event boolean')
        active = flag.bool()
        if stochastic_output and not bool(active.all()):
            raise ValueError('no post-RR noise or PPO credit in frozen predecessor')
        with torch.no_grad():
            anchor_mean = self.frozen_anchor({'policy': latent[..., :465]}, stochastic_output=False)
        old = self.frozen_anchor._last_forward_evidence
        center, _ = p05_capture_request_history(latent[..., :389])
        if bool(active.any()):
            new = self.mlp(latent)
            delta = new[..., 0, :]
            mean = anchor_mean + (1. - HISTORY_RHO) * delta
            ratios = torch.stack([tail[..., POST_RR_FIELDS.index(n)] for n in RATIO_FIELDS], dim=-1)
            if not bool((ratios[active].abs() < 1.).all()):
                raise ValueError('entry REQUEST reference has no finite inverse tanh')
            refs = torch.atanh(ratios)
            rebased = HISTORY_RHO * center[..., REBASE_INDICES] + (1. - HISTORY_RHO) * (
                refs + delta[..., REBASE_INDICES])
            mean = mean.clone()
            mean[..., REBASE_INDICES] = rebased
            mean = torch.where(active.unsqueeze(-1), mean, anchor_mean)
            log_std = new[..., 1, :]
        else:
            delta = torch.zeros_like(anchor_mean)
            mean = anchor_mean  # Literal identity, including floating-point evaluation order.
            log_std = old['head_log_std']
        conditional = torch.stack((mean, log_std), dim=-2)
        self._last_forward_evidence = dict(active=active.detach().clone(),
            conditional_mean=mean.detach().clone(), head_log_std=log_std.detach().clone(),
            anchor_conditional_mean=anchor_mean.detach().clone(),
            post_raw_mean_delta=delta.detach().clone(), history_center=center.detach().clone())
        if stochastic_output:
            self.distribution.update(conditional)
            return self.distribution.sample()
        return self.distribution.deterministic_output(conditional)

    def as_jit(self):
        raise NotImplementedError('do not export a graph that drops the frozen event route')

    def as_onnx(self, verbose=False):
        raise NotImplementedError('event-routed export has not been validated')


def audited_post_rr_policy_request(actor, observation, action_call, *, stochastic):
    if type(actor) is not SemanticPostRRHistoryMLPModel:
        raise ValueError('explicit post-RR packaged actor required')
    seen = []
    hook = actor.register_forward_hook(lambda _m, _i, _o: seen.append(actor._last_forward_evidence))
    try:
        raw = action_call()
    finally:
        hook.remove()
    if len(seen) != 1 or raw.shape != (1, 12):
        raise RuntimeError('exactly one action forward/draw required')
    e = seen[0]
    vector = lambda x: x[0].detach().cpu().tolist()
    active = bool(e['active'][0])
    if stochastic:
        mu, std = actor.output_distribution_params
        if not active or not torch.equal(mu, e['conditional_mean']) or not torch.equal(std, e['head_log_std'].exp()):
            raise RuntimeError('PPO likelihood must describe the actual sampled conditional distribution')
        logp = float(actor.get_output_log_prob(raw)[0].detach().cpu())
    else:
        if not torch.equal(raw, e['conditional_mean']):
            raise RuntimeError('deterministic action differs from saved policy mean')
        logp = None
    return raw, dict(schema='wlr50_clean.post_rr_actual_request.v1', policy_version=POLICY_VERSION,
        post_rr_active=active, mode='stochastic' if stochastic else 'deterministic',
        selected_raw_full12=vector(raw), selected_tanh_full12=vector(torch.tanh(raw)),
        selected_raw_log_probability=logp, conditional_mean_full12=vector(e['conditional_mean']),
        active_conditional_std_full12=vector(e['head_log_std'].exp()) if active else None,
        anchor_conditional_mean_full12=vector(e['anchor_conditional_mean']),
        post_raw_mean_delta_full12=vector(e['post_raw_mean_delta']), history_center_full12=vector(e['history_center']),
        frozen_anchor_state_sha256=actor.expected_anchor_state_sha256, history_kernel_applications=1,
        gain10_applied_again=False, public_preparation_module=True, relative_request_indices=list(REBASE_INDICES),
        prefix_excluded_from_new_PPO_credit=not active, sampling_draws=int(stochastic),
        task_target_projection_is_not_neural_output=True)

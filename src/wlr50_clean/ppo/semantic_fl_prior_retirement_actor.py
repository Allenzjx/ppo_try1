"""Explicit observed B subclass: explicit observed old-FL-prior retirement.

No sample overwrite. The unchanged scalar distribution samples and scores the
new conditional mean. No actuator, mapper, mask or source-owner code is added.
"""
import torch
from rsl_rl.utils import unpad_trajectories
from wlr50_clean.ppo.semantic_fl_forward_actor import WindowFLActor
from wlr50_clean.ppo.semantic_fl_forward_actor import OBSERVATION_LAYOUT as B_LAYOUT
from wlr50_clean.ppo.semantic_fl_forward_window import FL
from wlr50_clean.ppo.semantic_history_actor import HISTORY_RHO

from .semantic_fl_prior_retirement_task import OBSERVATION_DIMENSION, LAMBDA_INDEX

OBSERVATION_LAYOUT = 'accepted531_p06_fl_scalar_prior_retirement_v1'
POLICY_VERSION = 'frozen_A_p06_fl_conditional_prior_retirement_v1'


class PriorRetirementActor(WindowFLActor):
    policy_version = POLICY_VERSION

    def __init__(self, obs, obs_groups, obs_set, output_dim, *,
                 observation_layout=OBSERVATION_LAYOUT, **kwargs):
        if obs['policy'].shape[-1] != OBSERVATION_DIMENSION or observation_layout != OBSERVATION_LAYOUT:
            raise ValueError('explicit536 prior-retirement observation schema required')
        # ConstantFLHead is dimension-independent; its two parameters and all
        # frozen-A state keys are exactly the existing B535 keys. The temporary
        # base MLP is discarded by the original constructor as before.
        super().__init__({'policy':obs['policy'][...,:LAMBDA_INDEX]}, obs_groups, obs_set, output_dim,
                         observation_layout=B_LAYOUT, **kwargs)
        if type(self.obs_normalizer) is not torch.nn.Identity:
            raise ValueError('retirement cannot change observation normalization')
        self.obs_dim = self.observation_dimension = OBSERVATION_DIMENSION
        self.observation_layout = OBSERVATION_LAYOUT

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        values = obs['policy']
        if values.ndim != 2 or values.shape[-1] != OBSERVATION_DIMENSION or not bool(torch.isfinite(values).all()):
            raise ValueError('finite flat536 observation required')
        lam = values[:, LAMBDA_INDEX:LAMBDA_INDEX+1]
        flags = values[:, 531:532]
        if (not bool(((lam >= 0.) & (lam <= 1.)).all())
                or not bool(((flags == 0.) | (flags == 1.)).all())
                or bool(((flags == 0.) & (lam != 0.)).any())):
            raise ValueError('OFF must reset observed lambda; active lambda must be in [0,1]')
        # Reuse the ORIGINAL single full-A forward and two-scalar head. The
        # hook retains the actual head tensor with gradients, unlike detached
        # logging evidence. No second actor/head evaluation or random draw.
        heads = []
        handle = self.mlp.register_forward_hook(lambda module, args, out: heads.append(out))
        try:
            original_mean = super().forward(obs, masks=None, hidden_state=hidden_state, stochastic_output=False)
        finally:
            handle.remove()
        old = self._last_forward_evidence
        active = old['active']
        center = self.frozen_A._last_forward_evidence['history_center']
        if center.shape != original_mean.shape or not bool(torch.isfinite(center).all()):
            raise ValueError('same executed A forward must expose finite HISTORY center12')
        h = center[:, FL:FL+1]
        accepted = old['accepted_raw']
        innovation = accepted[:, FL:FL+1]-HISTORY_RHO*h
        mean = original_mean.clone()
        if bool(active.any()):
            if len(heads) != 1:
                raise RuntimeError('one original B scalar head forward required')
            delta = heads[0][:, 0, :]
            sigma = heads[0][:, 1, :].exp()
            intermediate = original_mean[:, FL:FL+1]-lam*innovation
            # Literal endpoint expressions preserve lambda0 identity and avoid
            # needless cancellation at the fully-retired endpoint.
            selected = torch.where(lam == 0., original_mean[:, FL:FL+1],
                torch.where(lam == 1., HISTORY_RHO*h+delta, intermediate))
            mean[:, FL] = torch.where(active[:, 0], selected[:, 0], original_mean[:, FL])
        else:
            if heads:
                raise RuntimeError('OFF must not evaluate the new head')
            delta = torch.zeros_like(h)
            sigma = torch.full_like(h, self.distribution.init_std)
        self._last_forward_evidence = dict(old,
            conditional_mean=mean.detach().clone(), conditional_std_FL=sigma.detach().clone(),
            retirement_lambda=lam.detach().clone(), history_center_FL=h.detach().clone(),
            old_A_innovation_FL=innovation.detach().clone(),
            retired_old_A_conditional_raw=(lam*innovation).detach().clone(),
            original_B_conditional_mean=original_mean.detach().clone(),
            innovation_FL=delta.detach().clone(), rho=HISTORY_RHO,
            history_center_source='same_executed_frozen_A_forward:p05_capture_request_history',
            runtime_raw_overwrite=False, public_prior_retirement_is_not_learned_delta=True)
        conditional = mean, sigma, active.to(mean.dtype)
        if stochastic_output:
            self.distribution.update(conditional)
            return self.distribution.sample()
        return self.distribution.deterministic_output(conditional)


def audited_retirement_request(actor, observation, action_call, *, stochastic):
    """Version-specific exact type; production audited_fl_request is unchanged."""
    if type(actor) is not PriorRetirementActor:
        raise ValueError('explicit prior-retirement actor required')
    seen = []
    handle = actor.register_forward_hook(lambda *_: seen.append(actor._last_forward_evidence))
    try:
        raw = action_call()
    finally:
        handle.remove()
    if len(seen) != 1 or raw.shape != (1, 12):
        raise RuntimeError('exactly one full12 action forward required')
    e = seen[0]
    active = bool(e['active'][0, 0])
    other = [i for i in range(12) if i != FL]
    if not torch.equal(raw[:, other], e['accepted_raw'][:, other]):
        raise RuntimeError('retirement changed a frozen coordinate')
    if stochastic:
        mu, sigma, mask = actor.output_distribution_params
        if (not torch.equal(mu, e['conditional_mean'][:, FL:FL+1])
                or not torch.equal(sigma, e['conditional_std_FL'])
                or not torch.equal(mask, e['active'].to(mu.dtype))):
            raise RuntimeError('sampled scalar distribution differs from audited mean/sigma/mask')
        logp = float(actor.get_output_log_prob(raw)[0].detach().cpu())
    else:
        if not torch.equal(raw, e['conditional_mean']):
            raise RuntimeError('DET sample differs from audited mean')
        logp = None
    vector = lambda x: x[0].detach().cpu().tolist()
    scalar = lambda name: float(e[name][0, 0].detach().cpu())
    return raw, dict(schema='wlr50_clean.p06_FL_prior_retirement_request.v1', policy_version=POLICY_VERSION,
        selected_raw_full12=vector(raw), accepted_raw_full12=vector(e['accepted_raw']),
        conditional_mean_full12=vector(e['conditional_mean']),
        conditional_mean_FL_raw=float(e['conditional_mean'][0,FL].detach().cpu()),
        original_B_conditional_mean_full12=vector(e['original_B_conditional_mean']),
        retirement_lambda=scalar('retirement_lambda'), history_center_FL=scalar('history_center_FL'),
        old_A_innovation_FL=scalar('old_A_innovation_FL'),
        retired_old_A_conditional_raw=scalar('retired_old_A_conditional_raw'),
        mean_delta_conditional_raw=scalar('innovation_FL'),
        conditional_std_FL_raw=scalar('conditional_std_FL') if active else None,
        active=active, selected_raw_log_probability=logp, density_dimension=1, stochastic_coordinate=FL,
        stochastic_FL_credit=int(stochastic and active),
        sampling_draws=int(stochastic and active), history_kernel_applications=1, rho=HISTORY_RHO,
        other11_frozen=True, A_state_sha256=actor.expected_frozen_A_sha256,
        public_prior_retirement_is_not_learned_delta=True, extra_rho_or_gain_multiplier=False,
        runtime_raw_overwrite=False, extra_actuator_projection=False,
        inactive_semantics='deterministic_frozen_A;conditional_density1_logp0_entropy0')

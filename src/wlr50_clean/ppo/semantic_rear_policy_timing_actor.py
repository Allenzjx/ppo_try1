"""Shared rear sigma kernel; no rear target override or mutable actor history."""
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from rsl_rl.utils import unpad_trajectories

from .semantic_history_actor import history_conditioned_head, HISTORY_RHO
from .semantic_p05_capture_actor import p05_capture_request_history
from .semantic_receiving_wheel_sigma import receiving_wheel_effective_log_std
from .semantic_rr_capture_profile import RR_TASK_START
from .semantic_rear_policy_timing_profile import (
    REAR_POLICY_TIMING_START, REAR_POLICY_TIMING_OBSERVATION_DIM,
    REAR_POLICY_TIMING_OBSERVATION_LAYOUT,
)


def rear_policy_timing_effective_log_std(head_log_std, latent, temperature=.25):
    """Sampling, current likelihood and audits must use this same pure kernel."""
    if latent.shape[-1] != REAR_POLICY_TIMING_OBSERVATION_DIM or not bool(torch.isfinite(latent).all()):
        raise ValueError('rear timing sigma requires finite 419 observations')
    timing = latent[..., REAR_POLICY_TIMING_START:]
    flags = timing[..., [0, 1, 2, 3, 7, 8]]
    reachable = latent[..., RR_TASK_START + 1]
    if (not bool(((flags == 0) | (flags == 1)).all())
            or not bool(((reachable == 0) | (reachable == 1)).all())
            or not bool(((timing[..., 4:7] >= 0) & (timing[..., 4:7] <= 1)).all())):
        raise ValueError('rear timing sigma requires boolean flags and normalized source clocks')
    parent, evidence = receiving_wheel_effective_log_std(head_log_std, latent[..., :372], temperature)
    rr_gate = (timing[..., 0] == 1) & (reachable == 1)
    rl_gate = timing[..., 2] == 1
    channels = torch.arange(12, device=latent.device)
    rr_channels = rr_gate.unsqueeze(-1) & (channels == 6)
    rl_channels = rl_gate.unsqueeze(-1) & ((channels == 3) | (channels == 1))
    multiplier = torch.where(rr_channels, torch.full_like(parent, 4.),
                            torch.where(rl_channels, torch.full_like(parent, 2.), torch.ones_like(parent)))
    log_std = parent + multiplier.log()
    if not bool((torch.isfinite(log_std.exp()) & (log_std.exp() > 0)).all()):
        raise ValueError('rear timing sigma must be finite positive without clipping')
    return log_std, {**evidence,
        'rear_rr_carry_reachable_sigma_active': rr_gate,
        'rear_rl_prep_sigma_active': rl_gate,
        'rear_policy_timing_observed_features': timing,
        'rear_local_sigma_gate_full12': rr_channels | rl_channels,
        'rear_local_sigma_multiplier_full12': multiplier,
        'effective_innovation_sigma_multiplier_full12':
            evidence['effective_innovation_sigma_multiplier_full12'] * multiplier}


class SemanticRearPolicyTimingHistoryMLPModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256,256), activation='elu',
                 obs_normalization=False, distribution_cfg=None, observation_layout=None,
                 exploration_std_temperature=None):
        if (observation_layout != REAR_POLICY_TIMING_OBSERVATION_LAYOUT or output_dim != 12
                or exploration_std_temperature != .25 or obs_normalization is not False):
            raise ValueError('rear timing requires explicit419/full12/Identity/quarter HISTORY')
        super().__init__(obs,obs_groups,obs_set,output_dim,hidden_dims,activation,obs_normalization,distribution_cfg)
        if self.obs_dim != REAR_POLICY_TIMING_OBSERVATION_DIM or type(self.distribution) is not HeteroscedasticGaussianDistribution:
            raise ValueError('rear timing actor distribution/layout mismatch')
        self.observation_layout = observation_layout

    @property
    def exploration_std_temperature(self): return .25

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs,masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs,masks,hidden_state)
        center,_ = p05_capture_request_history(latent[..., :389])
        head = history_conditioned_head(self.mlp(latent),center,HISTORY_RHO)
        if not stochastic_output:
            return self.distribution.deterministic_output(head)
        log_std,_ = rear_policy_timing_effective_log_std(head[...,1,:],latent,self.exploration_std_temperature)
        self.distribution.update(torch.stack((head[...,0,:],log_std),dim=-2))
        return self.distribution.sample()

    def as_jit(self): raise NotImplementedError('rear timing history JIT export is not implemented')
    def as_onnx(self,verbose=False): raise NotImplementedError('rear timing history ONNX export is not implemented')

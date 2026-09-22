"""Raw Gaussian actor sees assist state; actuator assistance is not a sample."""
from __future__ import annotations

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from rsl_rl.utils import unpad_trajectories

from .semantic_history_actor import cap_transition_request_history, history_conditioned_head, HISTORY_RHO
from .semantic_receiving_wheel_sigma import receiving_wheel_effective_log_std
from .semantic_p05_capture_profile import P05_CAPTURE_OBSERVATION_DIM, P05_CAPTURE_OBSERVATION_LAYOUT


def p05_capture_request_history(latent):
    if latent.shape[-1] != P05_CAPTURE_OBSERVATION_DIM or not bool(torch.isfinite(latent).all()):
        raise ValueError("capture actor requires finite observable 389 state")
    prefix = latent[..., :372]
    # Scheduling permission is not completion. Only this pure history-center
    # computation treats the explicitly observed P05->P06 handoff as an entry;
    # the actor input, actual completion bits, reward and sigma remain untouched.
    pending_entry = ((prefix[...,5] == 1) & (prefix[...,20] == 0)
                     & (latent[...,384] == 1) & (latent[...,386] == 1)
                     & (prefix[...,162] == 0))
    kernel = prefix
    if bool(pending_entry.any()):
        kernel = prefix.clone()
        kernel[...,162] = torch.where(pending_entry, torch.ones_like(kernel[...,162]), kernel[...,162])
    center, evidence = cap_transition_request_history(kernel)
    predecessor=(prefix[...,:13].argmax(-1)-1).clamp(min=0)
    evidence["physical_predecessor_completed"] = prefix[...,158:171].gather(-1,predecessor.unsqueeze(-1)).squeeze(-1)==1
    evidence["pending_scheduler_handoff"] = pending_entry
    return center, evidence


class SemanticP05CaptureHistoryMLPModel(MLPModel):
    """Same 12 Gaussian outputs and 0.9 history, with 17 observable new inputs."""
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256,256),
                 activation="elu", obs_normalization=False, distribution_cfg=None,
                 observation_layout=None, exploration_std_temperature=None):
        if (observation_layout != P05_CAPTURE_OBSERVATION_LAYOUT or output_dim != 12
                or exploration_std_temperature != .25 or obs_normalization is not False):
            raise ValueError("capture actor requires explicit 389/12 Identity quarter-history configuration")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, distribution_cfg)
        if self.obs_dim != P05_CAPTURE_OBSERVATION_DIM or type(self.distribution) is not HeteroscedasticGaussianDistribution:
            raise ValueError("capture actor distribution/layout mismatch")
        self.observation_layout = observation_layout

    @property
    def exploration_std_temperature(self):
        return .25

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs,masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs,masks,hidden_state)
        center,_ = p05_capture_request_history(latent)
        head = history_conditioned_head(self.mlp(latent),center,HISTORY_RHO)
        if not stochastic_output:
            return self.distribution.deterministic_output(head)
        log_std,_ = receiving_wheel_effective_log_std(head[...,1,:],latent[...,:372],.25)
        self.distribution.update(torch.stack((head[...,0,:],log_std),dim=-2))
        return self.distribution.sample()

    def as_jit(self):
        raise NotImplementedError("capture history JIT export is not implemented")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("capture history ONNX export is not implemented")

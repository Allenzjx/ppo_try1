"""Receiving continuation exploration; no learned state or action transform.

Only the official raw Gaussian standard deviation changes. This does not claim
forward traction, contact retention, physical task success, or a deterministic
improvement. Parent mean, HISTORY, capacities and action dispatch are untouched.
"""
from __future__ import annotations

import math
import torch
from tensordict import TensorDict
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories

from .semantic_history_actor import (
    HISTORY_RHO, SemanticTaskConditionedHipWheelHistoryMLPModel,
    cap_transition_request_history, history_conditioned_head,
    task_conditioned_effective_log_std,
)
from .semantic_receiving_wheel_profile import (
    RECEIVING_WHEEL_CHANNEL_INDICES, RECEIVING_WHEEL_PHASE_INDICES,
    RECEIVING_WHEEL_RR_PLACED_HISTORY_INDEX, RECEIVING_WHEEL_SIGMA_MULTIPLIER,
)
from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT, ROLE_OBSERVATION_DIM


def receiving_wheel_effective_log_std(head_log_std: torch.Tensor, latent: torch.Tensor,
                                     temperature: float):
    """Single shared sampling/likelihood/audit kernel; no RNG or hidden state."""
    parent_log_std, evidence = task_conditioned_effective_log_std(head_log_std, latent, temperature)
    stage = evidence["stage_index"]
    stage_gate = torch.zeros_like(stage, dtype=torch.bool)
    for phase in RECEIVING_WHEEL_PHASE_INDICES:
        stage_gate |= stage == phase
    receiving = stage_gate & (latent[..., RECEIVING_WHEEL_RR_PLACED_HISTORY_INDEX] == 1)
    channels = torch.arange(12, device=latent.device)
    channel_gate = torch.zeros_like(channels, dtype=torch.bool)
    for channel in RECEIVING_WHEEL_CHANNEL_INDICES:
        channel_gate |= channels == channel
    gate = receiving.unsqueeze(-1) & channel_gate
    log_std = torch.where(gate, parent_log_std + math.log(RECEIVING_WHEEL_SIGMA_MULTIPLIER),
                          parent_log_std)
    sigma = log_std.exp()
    if not bool((torch.isfinite(sigma) & (sigma > 0)).all()):
        raise ValueError("receiving-wheel sigma must be finite positive without clipping")
    multiplier = torch.where(gate, torch.full_like(log_std, RECEIVING_WHEEL_SIGMA_MULTIPLIER),
                             torch.ones_like(log_std))
    # Preserve the parent B/cap evidence under its original names; the extra
    # factor is separate, not a silently re-labelled physical action capacity.
    return log_std, {**evidence, "receiving_continuation_active": receiving,
        "RR_placed_history": latent[..., RECEIVING_WHEEL_RR_PLACED_HISTORY_INDEX],
        "receiving_sigma_gate_full12": gate,
        "receiving_sigma_multiplier_full12": multiplier,
        "effective_innovation_sigma_multiplier_full12":
            evidence["innovation_sigma_multiplier_full12"] * multiplier}


class SemanticReceivingWheelSigmaHistoryMLPModel(SemanticTaskConditionedHipWheelHistoryMLPModel):
    """Identical parameters and deterministic path; explicit positive sigma."""

    def forward(self, obs: TensorDict, masks: torch.Tensor | None = None,
                hidden_state: HiddenState = None, stochastic_output: bool = False) -> torch.Tensor:
        if not stochastic_output:
            return super().forward(obs, masks, hidden_state, stochastic_output=False)
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        if (getattr(self, "observation_layout", None) != ROLE_OBSERVATION_LAYOUT
                or self.obs_dim != ROLE_OBSERVATION_DIM or latent.shape[-1] != ROLE_OBSERVATION_DIM):
            raise ValueError("receiving-wheel actor requires the unchanged explicit 372 layout")
        history, _ = cap_transition_request_history(latent)
        head = history_conditioned_head(self.mlp(latent), history, HISTORY_RHO)
        log_std, _ = receiving_wheel_effective_log_std(
            head[..., 1, :], latent, self.exploration_std_temperature)
        self.distribution.update(torch.stack((head[..., 0, :], log_std), dim=-2))
        return self.distribution.sample()

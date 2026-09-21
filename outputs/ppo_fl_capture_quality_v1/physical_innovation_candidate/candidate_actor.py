"""OUTPUT-ONLY candidate; not registered in production or valid for real resume.

The only candidate delta is effective FR-knee innovation sigma in P06--P13.
No new parameter, buffer, mutable history, observation feature or action cap.
"""
from __future__ import annotations

import math
import torch
from tensordict import TensorDict
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories
from wlr50_clean.ppo.semantic_history_actor import (
    SemanticCapTransitionQuarterHistoryMLPModel,
    cap_transition_request_history, history_conditioned_head, HISTORY_RHO,
)
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT, ROLE_OBSERVATION_DIM

CANDIDATE_VERSION = "output_only_request_history_FR_knee_P06plus_sigma_24_over_112_v1"
FR_KNEE_INDEX = 3
FIRST_SCALED_PHASE_INDEX = 5  # Zero-based P06; not a history-dependent flag.
FR_KNEE_SIGMA_SCALE = 24.0 / 112.0


class PhysicalInnovationCandidate(SemanticCapTransitionQuarterHistoryMLPModel):
    """Same MLP/mean/full action support; one observation-selected log-sigma delta."""

    def forward(self, obs: TensorDict, masks: torch.Tensor | None = None,
                hidden_state: HiddenState = None, stochastic_output: bool = False) -> torch.Tensor:
        if not stochastic_output:
            # Delegate the original deterministic path, including cache/RNG behavior.
            return super().forward(obs, masks, hidden_state, stochastic_output=False)
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        if (getattr(self, "observation_layout", None) != ROLE_OBSERVATION_LAYOUT
                or self.obs_dim != ROLE_OBSERVATION_DIM or latent.shape[-1] != ROLE_OBSERVATION_DIM):
            raise ValueError("candidate requires the original explicit 372 observation layout")
        history, evidence = cap_transition_request_history(latent)
        head = history_conditioned_head(self.mlp(latent), history, HISTORY_RHO)
        log_std = head[..., 1, :] + math.log(self.exploration_std_temperature)
        gate = ((evidence["stage_index"] >= FIRST_SCALED_PHASE_INDEX).unsqueeze(-1)
                & (torch.arange(12, device=latent.device) == FR_KNEE_INDEX))
        # torch.where preserves every unselected log-sigma value bit-for-bit.
        # Scaling the distribution, not the sampled action, keeps all likelihood
        # and KL consumers on the same official Gaussian cache.
        log_std = torch.where(gate, log_std + math.log(FR_KNEE_SIGMA_SCALE), log_std)
        sigma = log_std.exp()
        if not bool((torch.isfinite(sigma) & (sigma > 0)).all()):
            raise ValueError("effective candidate sigma must remain finite and strictly positive")
        self.distribution.update(torch.stack((head[..., 0, :], log_std), dim=-2))
        return self.distribution.sample()

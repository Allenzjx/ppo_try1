"""OUTPUT-ONLY CPU prototype. Not a registered production policy or checkpoint.

Preserves previous filtered physical REQUEST at cap-expansion entry, not final
actuator displacement/actual joint motion. Learned sigma and execution are intact.
"""
from __future__ import annotations

import math
import torch
from rsl_rl.utils import unpad_trajectories
from wlr50_clean.ppo.semantic_history_actor import (
    SemanticQuarterTemperedHistoryMLPModel, history_conditioned_head,
)

PROTOTYPE_VERSION = 'output_only_quarter_cap_transition_request_history_prototype_v1'
_EARLY = (18.,24.,18.,24.,12.,18.,12.,18.,.6,.6,.6,.6)
_FRONT = (18.,24.,18.,24.,12.,18.,12.,18.,1.,.6,1.,.6)
_LATE = (32.,36.,24.,112.,24.,36.,24.,36.,1.2,1.2,1.,.6)
CAPS = (_EARLY, _EARLY, _FRONT, _FRONT, _FRONT, *(_LATE,)*8)
REQUEST_SCALES = (4.,4.,4.,6.,4.,4.,4.,4.,.12,.12,.12,.12)


def request_transition_history(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure observable function, no history cache and no mutation of observations.

    Current fixed120/15 phase transitions occur at an aligned decision boundary.
    Gate exact encoded age zero, completed immediate predecessor and larger cap.
    Float32 decoding is approximate to source doubles; no statistics fitting.
    """
    if (not isinstance(latent, torch.Tensor) or not latent.is_floating_point()
            or latent.ndim < 2 or latent.shape[-1] != 372
            or not bool(torch.isfinite(latent).all())
            or not bool((latent.abs() <= 20).all())):
        raise ValueError('prototype requires finite fixed-scale-clipped 372 observations')
    stage_bits = latent[..., :13]
    completed = latent[..., 158:171]
    age = latent[..., 20]
    if (not bool(((stage_bits == 0) | (stage_bits == 1)).all())
            or not bool((stage_bits.sum(-1) == 1).all())
            or not bool(((completed == 0) | (completed == 1)).all())
            or not bool((age >= 0).all())):
        raise ValueError('stage, completed-stage or nonnegative age encoding is invalid')
    stage = stage_bits.argmax(-1)
    previous_stage = (stage-1).clamp(min=0)
    caps = latent.new_tensor(CAPS)
    current_cap, previous_cap = caps[stage], caps[previous_stage]
    predecessor_done = completed.gather(-1, previous_stage.unsqueeze(-1)).squeeze(-1) == 1
    first = (age == 0) & (stage > 0) & predecessor_done
    changed = first.unsqueeze(-1) & (current_cap > previous_cap)
    raw = latent[..., 195:207]
    if not bool(changed.any()):
        return raw, changed
    physical_request = latent[..., 207:219] * latent.new_tensor(REQUEST_SCALES)
    # A larger cap leaves inverse tanh strictly interior for every valid old
    # request. Reject bad observation semantics rather than silently clipping.
    valid = physical_request.abs() <= previous_cap + 1e-5
    ratio = torch.where(changed, physical_request / current_cap, torch.zeros_like(raw))
    if not bool((~changed | valid).all()) or not bool((ratio.abs() < 1).all()):
        raise ValueError('previous filtered request violates its predecessor cap')
    centered = torch.atanh(ratio)
    return torch.where(changed, centered, raw), changed


class OutputOnlyCapTransitionQuarterActor(SemanticQuarterTemperedHistoryMLPModel):
    """Same tensor topology; only conditional mean uses the observable gate."""

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        history, _ = request_transition_history(latent)
        head = history_conditioned_head(self.mlp(latent), history, .9)
        if not stochastic_output:
            return self.distribution.deterministic_output(head)
        log_std = head[..., 1, :] + math.log(self.exploration_std_temperature)
        sigma = log_std.exp()
        if not bool((torch.isfinite(sigma) & (sigma > 0)).all()):
            raise ValueError('effective conditional sigma must be finite and strictly positive without clipping')
        self.distribution.update(torch.stack((head[..., 0, :], log_std), dim=-2))
        return self.distribution.sample()

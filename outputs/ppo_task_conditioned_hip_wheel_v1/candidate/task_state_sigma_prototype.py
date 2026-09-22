"""CPU-only exploratory kernel, NOT a production policy or adopted B table.

All decisions depend on the existing fixed-scale 372 observation. Support gates
are explicitly proxies: current TOP/Y-ROI/body-clearance are not in this layout.
No actor forwards, RNG, control writes, acceptance changes or sampling rejection.
"""
from __future__ import annotations

import math
import torch

from wlr50_clean.ppo.semantic_policy_distribution import REQUEST_HISTORY_CAPS


SIGMA_CANDIDATE_VERSION = "UNADOPTED_task_conditioned_physical_B_prototype_v1"
ORDER = ("FL_hip", "FL_knee", "FR_hip", "FR_knee", "RL_hip", "RL_knee",
         "RR_hip", "RR_knee", "FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel")
B_TABLE = {
    "FR_air_approach_proxy": (9.,12.,12.,18.,12.,12.,6.,9.,.6,.6,.6,.6),
    "FL_crossed_air_pending": (24.,24.,9.,12.,6.,9.,6.,9.,1.,.6,1.,.6),
    "P06_front_support_rolling_proxy": (12.,12.,12.,12.,12.,12.,12.,12.,1.2,1.2,1.,.6),
    "P06_front_support_recovery_proxy": (24.,24.,12.,18.,12.,12.,12.,18.,1.2,1.2,1.,.6),
    "RR_preparation": (24.,24.,12.,18.,24.,24.,12.,18.,1.2,1.2,1.,.6),
    "RR_current_valid_lift": (24.,24.,12.,18.,24.,24.,24.,36.,1.2,1.2,1.,.6),
}


def _validate_latent(latent: torch.Tensor) -> None:
    if (not isinstance(latent, torch.Tensor) or not latent.is_floating_point()
            or latent.ndim < 2 or latent.shape[-1] != 372
            or not bool(torch.isfinite(latent).all()) or not bool((latent.abs() <= 20).all())):
        raise ValueError("expected finite fixed-scale-clipped [...,372] observations")
    for start, end in ((0,13), (131,139), (146,171)):
        bits = latent[..., start:end]
        if not bool(((bits == 0) | (bits == 1)).all()):
            raise ValueError("phase/contact/history/completed features must be binary")
    if not bool((latent[..., :13].sum(-1) == 1).all()) or not bool((latent[...,20] >= 0).all()):
        raise ValueError("one phase and nonnegative observed stage age required")


def candidate_physical_scales(latent: torch.Tensor):
    """Return B, actual unchanged request cap, and transparent current gates."""
    _validate_latent(latent)
    stage = latent[..., :13].argmax(-1)
    cap = latent.new_tensor(REQUEST_HISTORY_CAPS)[stage]
    # Reproduce the currently adopted e735 innovation fallback, not a new cap.
    base = cap.clone()
    base[...,3] = torch.where(stage >= 5, 24., base[...,3])
    weights = {}
    no_FR_pairs = (latent[...,133] == 0) & (latent[...,134] == 0)
    no_FL_pairs = (latent[...,131] == 0) & (latent[...,132] == 0)
    fr = ((stage <= 1) & (latent[...,155] == 0) & (latent[...,147] == 1)
          & no_FR_pairs & (latent[...,33] >= 2))
    weights["FR_air_approach_proxy"] = fr * (latent[...,24] / .015).clamp(0.,1.)
    fl = ((stage == 4) & (latent[...,150] == 1) & (latent[...,154] == 0) & no_FL_pairs)
    weights["FL_crossed_air_pending"] = fl * (latent[...,21] / .003).clamp(0.,1.)

    def front_support_proxy(leg_index):
        # A pair reaction near top geometry is NOT exact classified TOP bearing.
        # No current-y/contact-surface/force-vector validity bit is fabricated.
        goal = 21 + 3*leg_index
        geom = 99 + 3*leg_index
        pair = 131 + 2*leg_index
        force = 123 + 2*leg_index
        return ((latent[...,pair] == 0) & (latent[...,pair+1] == 1)
                & (latent[...,force+1]*100. >= .2)
                & (latent[...,goal] >= -.015) & (latent[...,goal] <= .025)
                & (latent[...,goal+1] >= -.005) & (latent[...,geom+1] <= .005))

    fronts_placed = (latent[...,154] == 1) & (latent[...,155] == 1)
    support_proxy = front_support_proxy(0) & front_support_proxy(1)
    rr_pending = latent[...,157] == 0
    p06 = (stage == 5) & fronts_placed & rr_pending
    rolling = p06 & support_proxy
    weights["P06_front_support_rolling_proxy"] = rolling.to(latent.dtype)
    weights["P06_front_support_recovery_proxy"] = (p06 & ~support_proxy).to(latent.dtype)
    # Fade the rolling preference out using current rear edge distance, not time.
    # .05m blend is a candidate exploration schedule, not an acceptance threshold.
    rear_x = torch.maximum(latent[...,28], latent[...,31])
    rear_proximity = ((rear_x + .27) / .05).clamp(0.,1.)
    rr_prepare = ((stage >= 6) & (stage <= 8) & rr_pending).to(latent.dtype)
    rr_prepare = torch.maximum(rr_prepare, p06 * rear_proximity)
    weights["RR_preparation"] = rr_prepare
    # CURRENT functional qualification, already observable and revoked on ground.
    rr_valid = ((stage >= 5) & (stage <= 8) & rr_pending & (latent[...,149] == 1))
    weights["RR_current_valid_lift"] = rr_valid.to(latent.dtype)

    result = base
    for name, weight in weights.items():
        target = latent.new_tensor(B_TABLE[name])
        result = torch.lerp(result, target, weight.unsqueeze(-1))
    if not bool((torch.isfinite(result) & (result > 0)).all()):
        raise ValueError("all twelve physical equivalent scales must stay positive")
    return result, cap, {"weights": weights, "stage_index": stage,
                        "front_support_proxy_not_exact_TOP": support_proxy,
                        "current_RR_qualification": latent[...,149],
                        "current_rear_front_distance_m": rear_x}


def candidate_effective_head(head: torch.Tensor, latent: torch.Tensor):
    """Input head ALREADY includes existing REQUEST-HISTORY conditional mean.

    This preserves that tensor's mean exactly. Sampling/logp must use this one
    head in the same official Gaussian; this function performs no draw or filter.
    """
    B, cap, gates = candidate_physical_scales(latent)
    if (not isinstance(head, torch.Tensor) or tuple(head.shape) != (*latent.shape[:-1],2,12)
            or head.dtype != latent.dtype or head.device != latent.device
            or not bool(torch.isfinite(head).all())):
        raise ValueError("matching finite [...,2,12] Gaussian head required")
    log_std = head[...,1,:] + math.log(.25) + (B/cap).log()
    sigma = log_std.exp()
    if not bool((torch.isfinite(sigma) & (sigma > 0)).all()):
        raise ValueError("candidate effective Gaussian sigma must remain positive finite")
    return torch.stack((head[...,0,:], log_std), dim=-2), {
        **gates, "B_full12": B, "unchanged_cap_full12": cap,
        "sigma_multiplier_full12": B/cap, "effective_sigma_full12": sigma,
        "version": SIGMA_CANDIDATE_VERSION,
    }

"""DRAFT: observed, parameter-free rear sigma candidate. No target transform.

Scalar rules are standard-library-only. The tensor kernel imports Torch only
when explicitly called by the future reviewed actor/likelihood/audit path.
"""
from __future__ import annotations

import math

SIGMA_MODE = 'observable_rear_cooperative_preparation_sigma_v1'
PREP_TOTAL_MULTIPLIERS = ((3, 16.), (1, 2.), (4, 2.), (5, 8.))
RR_HIP_INDEX, RR_HIP_MULTIPLIER = 6, 4.
FL_WHEEL_INDEX, FL_WHEEL_EXTRA_MULTIPLIER = 8, 2.


def cooperative_prep_multipliers(*, rr_carry_capture, rr_top_reachable,
                                rl_prep_transfer, receiving_continuation_active):
    """Scalar reference; total multiplier relative to receiving-wheel PARENT.

    rr_top_reachable is an already observed sensor-based task candidate, NOT
    fabricated contact/bearing or an actuator-space feasibility certificate.
    """
    flags = (rr_carry_capture, rr_top_reachable, rl_prep_transfer, receiving_continuation_active)
    if any(type(flag) is not bool for flag in flags):
        raise ValueError('four exact observable booleans required')
    carry_reachable = rr_carry_capture and rr_top_reachable
    prep_allowed = carry_reachable or rl_prep_transfer
    wheel_extra = carry_reachable and not receiving_continuation_active
    result = [1.] * 12
    if carry_reachable:
        result[RR_HIP_INDEX] = RR_HIP_MULTIPLIER
    if prep_allowed:
        for index, total in PREP_TOTAL_MULTIPLIERS:
            result[index] = total
    if wheel_extra:
        result[FL_WHEEL_INDEX] = FL_WHEEL_EXTRA_MULTIPLIER
    return tuple(result), dict(
        sigma_mode=SIGMA_MODE,
        observed_rr_carry_capture=rr_carry_capture,
        observed_rr_top_reachable=rr_top_reachable,
        observed_rl_prep_transfer=rl_prep_transfer,
        observed_parent_receiving_continuation_active=receiving_continuation_active,
        rr_carry_reachable=carry_reachable, prep_allowed=prep_allowed,
        fl_wheel_extra_active=wheel_extra,
        support_transfer_permission_modified=False,
        multiplier_reference='receiving_wheel_effective_sigma_not_old_rear_sigma',
        target_or_mean_modified=False)


def scalar_effective_log_std(parent_log_std, **flags):
    """Test/reference arithmetic only, never a second deployment likelihood."""
    if len(parent_log_std) != 12 or any(not math.isfinite(x) for x in parent_log_std):
        raise ValueError('finite parent Full12 log sigma required')
    multiplier, evidence = cooperative_prep_multipliers(**flags)
    return tuple(x+math.log(m) for x, m in zip(parent_log_std, multiplier)), evidence


def cooperative_prep_effective_log_std(head_log_std, latent, temperature=.25):
    """ONE tensor kernel for sampling, current likelihood and request audit.

    Keep current422 validation/HISTORY/receiving parent intact. Replace the
    rear multiplier as a TOTAL; do not multiply16 on top of old FR-knee×2.
    No RNG, learned parameters, action projection, clamps or hidden state.
    """
    import torch
    from .semantic_p02_progress_actor import validate_p02_progress_latent
    from .semantic_receiving_wheel_sigma import receiving_wheel_effective_log_std
    from .semantic_rear_policy_timing_profile import REAR_POLICY_TIMING_START
    from .semantic_rr_capture_profile import RR_TASK_START

    validate_p02_progress_latent(latent)
    timing = latent[..., REAR_POLICY_TIMING_START:419]
    flags = timing[..., [0, 1, 2, 3, 7, 8]]
    reachable = latent[..., RR_TASK_START+1]
    if (not bool(((flags == 0) | (flags == 1)).all())
            or not bool(((reachable == 0) | (reachable == 1)).all())
            or not bool(((timing[..., 4:7] >= 0) & (timing[..., 4:7] <= 1)).all())):
        raise ValueError('cooperative sigma requires existing boolean flags and normalized source clocks')
    parent, evidence = receiving_wheel_effective_log_std(head_log_std, latent[..., :372], temperature)
    carry = timing[..., 0] == 1
    rl_prep = timing[..., 2] == 1
    rr_gate = carry & (reachable == 1)
    prep = rr_gate | rl_prep
    receiving = evidence['receiving_continuation_active']
    fl_wheel = rr_gate & ~receiving
    channels = torch.arange(12, device=latent.device)
    multiplier = torch.where(rr_gate.unsqueeze(-1) & (channels == RR_HIP_INDEX),
        torch.full_like(parent, RR_HIP_MULTIPLIER), torch.ones_like(parent))
    for index, total in PREP_TOTAL_MULTIPLIERS:
        multiplier = torch.where(prep.unsqueeze(-1) & (channels == index),
                                 torch.full_like(parent, total), multiplier)
    multiplier = torch.where(fl_wheel.unsqueeze(-1) & (channels == FL_WHEEL_INDEX),
        torch.full_like(parent, FL_WHEEL_EXTRA_MULTIPLIER), multiplier)
    log_std = parent + multiplier.log()
    if not bool((torch.isfinite(log_std.exp()) & (log_std.exp() > 0)).all()):
        raise ValueError('cooperative sigma must be finite positive without clipping')
    return log_std, {**evidence,
        'rear_rr_carry_reachable_sigma_active': rr_gate,
        # Retain the old meaning of this field; do not relabel AIR prep as RL transfer.
        'rear_rl_prep_sigma_active': rl_prep,
        'rear_policy_timing_observed_features': timing,
        'rear_local_sigma_gate_full12': multiplier != 1,
        'rear_local_sigma_multiplier_full12': multiplier,
        'effective_innovation_sigma_multiplier_full12':
            evidence['effective_innovation_sigma_multiplier_full12'] * multiplier,
        'cooperative_sigma_mode': SIGMA_MODE,
        'cooperative_observed_rr_carry_capture': carry,
        'cooperative_observed_rr_top_reachable': reachable == 1,
        'cooperative_observed_rl_prep_transfer': rl_prep,
        'cooperative_parent_receiving_continuation_active': receiving,
        'cooperative_prep_allowed': prep,
        'cooperative_fl_wheel_extra_active': fl_wheel,
        'cooperative_multiplier_reference': 'receiving_wheel_effective_sigma_not_old_rear_sigma',
        'cooperative_support_transfer_permission_modified': False,
        'cooperative_target_or_mean_modified': False}

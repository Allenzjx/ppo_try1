"""Explicit, metadata-only receiving-wheel exploration profile.

Explicit versioned profile. Registration requires an explicitly reviewed
same-MDP, changed-stochastic-kernel migration; this is not a quantity extension.
"""
from __future__ import annotations

RECEIVING_WHEEL_POLICY = "task_conditioned_receiving_wheel_sigma_x3_v1"
RECEIVING_WHEEL_ACTOR_CLASS = (
    "wlr50_clean.ppo.semantic_receiving_wheel_sigma:SemanticReceivingWheelSigmaHistoryMLPModel"
)
RECEIVING_WHEEL_SIGMA_SEMANTICS = (
    "task_conditioned_B_over_cap_then_P10_P12_RR_placed_history_FR_RR_wheel_sigma_x3_same372_v1"
)
RECEIVING_WHEEL_PHASE_INDICES = (9, 10, 11)
RECEIVING_WHEEL_CHANNEL_INDICES = (9, 11)
RECEIVING_WHEEL_RR_PLACED_HISTORY_INDEX = 157
RECEIVING_WHEEL_SIGMA_MULTIPLIER = 3.0


def receiving_wheel_policy_contract(*, observation_layout: str | None = None) -> dict:
    """Complete parent contract plus the sole, explicit stochastic change."""
    from .semantic_policy_distribution import TASK_CONDITIONED_HIP_WHEEL_POLICY, policy_contract
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT

    if observation_layout != ROLE_OBSERVATION_LAYOUT:
        raise ValueError("receiving-wheel sigma requires the unchanged explicit role372 layout")
    contract = policy_contract(TASK_CONDITIONED_HIP_WHEEL_POLICY,
                               observation_layout=observation_layout)
    contract.update({
        "version": RECEIVING_WHEEL_POLICY,
        "actor_class": RECEIVING_WHEEL_ACTOR_CLASS,
        "sigma_scaling_semantics": RECEIVING_WHEEL_SIGMA_SEMANTICS,
        "sigma_profile_parent_version": TASK_CONDITIONED_HIP_WHEEL_POLICY,
        "conditional_std": "parent_task_conditioned_sigma*receiving_wheel_multiplier",
        "effective_log_std": "parent_task_conditioned_effective_log_std+log(receiving_wheel_multiplier)",
        "receiving_continuation_gate": {
            "phase_indices": list(RECEIVING_WHEEL_PHASE_INDICES),
            "RR_placed_history_observation_index": RECEIVING_WHEEL_RR_PLACED_HISTORY_INDEX,
            "RR_placed_history_value": 1,
            "channel_indices": list(RECEIVING_WHEEL_CHANNEL_INDICES),
            "channel_names": ["FR_wheel", "RR_wheel"],
            "conditional_sigma_multiplier": RECEIVING_WHEEL_SIGMA_MULTIPLIER,
            "otherwise_multiplier": 1.0,
            "current_RR_support_required": False,
            "current_RR_lift_required": False,
            "P13_unchanged": True,
        },
        "receiving_scale_units": "dimensionless_conditional_innovation_std_not_cap_or_target",
        "directional_bias": False,
        "extra_filter_or_sampling_rejection": False,
    })
    return contract

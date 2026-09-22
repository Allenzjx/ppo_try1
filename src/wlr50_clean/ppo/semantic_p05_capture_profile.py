"""Explicit observable capture-assist MDP, preserving the receiving sigma kernel."""
from __future__ import annotations

P05_CAPTURE_POLICY = "p05_hip_only_capture_assist_history_v1"
P05_CAPTURE_ACTOR_CLASS = "wlr50_clean.ppo.semantic_p05_capture_actor:SemanticP05CaptureHistoryMLPModel"
P05_CAPTURE_OBSERVATION_LAYOUT = "role372_p05_capture_assist_v1"
P05_CAPTURE_ASSIST_GROUP = "p05_capture_assist_state_full12"
P05_CAPTURE_CONTINUATION_GROUP = "p05_capture_continuation_full5"
P05_CAPTURE_OBSERVATION_DIM = 389
P05_CAPTURE_CONTINUATION_FIELDS = (
    "fl_capture_pending", "allow_capture_continuation", "scheduler_advanced_pending",
    "p05_local_deadline_warning", "capture_pending_elapsed_fraction",
)
P05_CAPTURE_HISTORY_SEMANTICS = "completed_predecessor_or_observed_pending_P05_scheduler_handoff_request_center_v1"


def p05_capture_policy_contract(*, observation_layout=None):
    from .semantic_receiving_wheel_profile import receiving_wheel_policy_contract, RECEIVING_WHEEL_POLICY
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    if observation_layout != P05_CAPTURE_OBSERVATION_LAYOUT:
        raise ValueError("P05 capture policy requires the explicit observable 389 layout")
    result = receiving_wheel_policy_contract(observation_layout=ROLE_OBSERVATION_LAYOUT)
    result.update(version=P05_CAPTURE_POLICY, actor_class=P05_CAPTURE_ACTOR_CLASS,
        observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT,
        observation_dimension=P05_CAPTURE_OBSERVATION_DIM,
        preserved_observation_prefix_dimension=372,
        capture_assist_observation_group=P05_CAPTURE_ASSIST_GROUP,
        capture_assist_observation_slice=[372,384],
        capture_continuation_observation_group=P05_CAPTURE_CONTINUATION_GROUP,
        capture_continuation_observation_slice=[384,389],
        capture_continuation_observation_fields=list(P05_CAPTURE_CONTINUATION_FIELDS),
        history_center_semantics=P05_CAPTURE_HISTORY_SEMANTICS,
        sigma_kernel_source_policy=RECEIVING_WHEEL_POLICY,
        sigma_kernel_observation_slice=[0,372],
        action_transform="declared_state_dependent_FL_hip_feedback_and_final_knee_hold_after_policy_sample",
        raw_sample_and_log_probability="original_Gaussian_before_capture_assist_unchanged",
        mdp_change="observable_capture_assist_and_nonterminal_pending_P05_continuation",
        completion_history="actual_completed_ordered_subset_not_scheduler_success")
    return result

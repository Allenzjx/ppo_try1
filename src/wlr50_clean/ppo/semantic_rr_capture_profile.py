"""Versioned RR capture/real RL transfer observations; unchanged Gaussian kernel."""
from .semantic_rr_capture_assist import RR_CAPTURE_ASSIST_FEATURE_NAMES
from .semantic_p05_capture_profile import P05_CAPTURE_OBSERVATION_DIM

RR_CAPTURE_POLICY = 'rr_capture_then_rl_transfer_history_v1'
RR_CAPTURE_ACTOR_CLASS = 'wlr50_clean.ppo.semantic_rr_capture_actor:SemanticRRCaptureHistoryMLPModel'
RR_CAPTURE_OBSERVATION_LAYOUT = 'role389_rr_capture_transfer_v1'
RR_ASSIST_GROUP = 'rr_capture_assist_state_full14'
RR_TASK_GROUP = 'rr_capture_transfer_context_full7'
RR_TASK_FIELDS = ('rr_lift_carry','rr_top_reachable','rr_top_contact','rr_current_bearing',
                  'rl_transfer_ready','rr_capture_recovery_allowed','fl_wheel_guidance_active')
RR_ASSIST_START = P05_CAPTURE_OBSERVATION_DIM
RR_TASK_START = RR_ASSIST_START + len(RR_CAPTURE_ASSIST_FEATURE_NAMES)
RR_CAPTURE_OBSERVATION_DIM = RR_TASK_START + len(RR_TASK_FIELDS)


def rr_capture_policy_contract(*, observation_layout=None):
    from .semantic_p05_capture_profile import p05_capture_policy_contract, P05_CAPTURE_OBSERVATION_LAYOUT
    if observation_layout != RR_CAPTURE_OBSERVATION_LAYOUT:
        raise ValueError('RR capture policy requires its explicit appended-state layout')
    result = p05_capture_policy_contract(observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)
    result.update(version=RR_CAPTURE_POLICY,actor_class=RR_CAPTURE_ACTOR_CLASS,
        observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT,observation_dimension=RR_CAPTURE_OBSERVATION_DIM,
        preserved_observation_prefix_dimension=P05_CAPTURE_OBSERVATION_DIM,
        rr_capture_assist_observation_group=RR_ASSIST_GROUP,
        rr_capture_assist_observation_slice=[RR_ASSIST_START,RR_TASK_START],
        rr_capture_assist_observation_fields=list(RR_CAPTURE_ASSIST_FEATURE_NAMES),
        rr_capture_task_observation_group=RR_TASK_GROUP,
        rr_capture_task_observation_slice=[RR_TASK_START,RR_CAPTURE_OBSERVATION_DIM],
        rr_capture_task_observation_fields=list(RR_TASK_FIELDS),
        action_transform='declared_separate_FL_and_RR_hip_feedback_knee_hold_after_original_Gaussian_sample',
        mdp_change='observable_RR_capture_and_current_contact_RL_transfer_control',
        old_389_numerical_codec_preserved=True,
        contact_semantics='RR_lift_carry_top_reachable_current_TOP_and_current_bearing_are_distinct',
        same_physical_state_or_trajectory_equivalence_claimed=False)
    return result

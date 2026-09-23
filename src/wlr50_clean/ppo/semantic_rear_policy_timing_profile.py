"""Explicit rear policy authority, observable source timing and local exploration."""
from .semantic_rr_capture_profile import (
    RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_DIM, RR_CAPTURE_OBSERVATION_LAYOUT,
    RR_TASK_START, rr_capture_policy_contract,
)

REAR_POLICY_TIMING_POLICY = 'rr_rl_timing_policy_learning_history_v1'
REAR_POLICY_TIMING_ACTOR_CLASS = (
    'wlr50_clean.ppo.semantic_rear_policy_timing_actor:SemanticRearPolicyTimingHistoryMLPModel'
)
REAR_POLICY_TIMING_OBSERVATION_LAYOUT = 'role410_rear_policy_timing_v1'
REAR_POLICY_TIMING_GROUP = 'rear_policy_timing_full9'
REAR_POLICY_TIMING_FIELDS = (
    'rr_carry_capture', 'rr_support_handoff', 'rl_prep_transfer', 'rl_swing_capture',
    'p09_source_time_s', 'p12_source_time_s', 'p12_rl_source_time_s',
    'p09_dependency_wait', 'p12_dependency_wait',
)
REAR_POLICY_TIMING_START = RR_CAPTURE_OBSERVATION_DIM
REAR_POLICY_TIMING_OBSERVATION_DIM = REAR_POLICY_TIMING_START + len(REAR_POLICY_TIMING_FIELDS)
REAR_POLICY_TIMING_TIME_FIELDS = frozenset(REAR_POLICY_TIMING_FIELDS[4:7])
REAR_POLICY_TIMING_SIGMA_SEMANTICS = (
    'parent_receiving_sigma_then_observed_RR_carry_reachable_hip_x4_RL_prep_FR_FL_knee_x2_v1'
)


def rear_policy_timing_policy_contract(*, observation_layout=None):
    if observation_layout != REAR_POLICY_TIMING_OBSERVATION_LAYOUT:
        raise ValueError('rear policy timing requires its explicit 419 layout')
    result = rr_capture_policy_contract(observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
    result.update(
        version=REAR_POLICY_TIMING_POLICY, actor_class=REAR_POLICY_TIMING_ACTOR_CLASS,
        observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT,
        observation_dimension=REAR_POLICY_TIMING_OBSERVATION_DIM,
        preserved_observation_prefix_dimension=RR_CAPTURE_OBSERVATION_DIM,
        rear_policy_timing_observation_group=REAR_POLICY_TIMING_GROUP,
        rear_policy_timing_observation_slice=[REAR_POLICY_TIMING_START, REAR_POLICY_TIMING_OBSERVATION_DIM],
        rear_policy_timing_observation_fields=list(REAR_POLICY_TIMING_FIELDS),
        rear_policy_timing_time_scale_s=200.0,
        action_transform='declared_FL_capture_assist_only;rear_task_assists_and_rear_task_wheel_projection_OFF;existing_physical_limits_and_mapper_preserved',
        raw_sample_and_log_probability='same_state_conditioned_raw_Gaussian_for_sample_current_likelihood_and_audit_before_physical_dispatch',
        mdp_change='observable_rear_source_dependency_timing_with_policy_owned_rear_hip_knee',
        old_410_numerical_codec_preserved=True,
        rr_assist_off_observation='existing_fourteen_WAIT_zero_features_not_hidden_rear_assistance',
        sigma_profile_parent_version=RR_CAPTURE_POLICY,
        sigma_kernel_observation_slice=[0, REAR_POLICY_TIMING_OBSERVATION_DIM],
        sigma_scaling_semantics=REAR_POLICY_TIMING_SIGMA_SEMANTICS,
        conditional_std='parent_receiving_effective_sigma*observable_rear_local_multiplier',
        effective_log_std='parent_receiving_effective_log_std+log(observable_rear_local_multiplier)',
        rear_local_sigma_gates={
            'RR_hip': {'channel_index': 6, 'carry_index': REAR_POLICY_TIMING_START,
                       'current_top_reachable_index': RR_TASK_START + 1, 'multiplier': 4.0},
            'FR_FL_knee': {'channel_indices': [3, 1], 'rl_prep_index': REAR_POLICY_TIMING_START + 2,
                           'multiplier': 2.0},
            'otherwise_multiplier': 1.0,
        },
        rear_sigma_units='dimensionless_effective_conditional_std_multiplier_not_servo_degrees_or_raw_target',
        rear_sigma_status='versioned_initial_diagnostic_candidate_not_claimed_converged',
        deterministic_mean_change=False,
        directional_bias=False, extra_filter_or_sampling_rejection=False,
    )
    return result

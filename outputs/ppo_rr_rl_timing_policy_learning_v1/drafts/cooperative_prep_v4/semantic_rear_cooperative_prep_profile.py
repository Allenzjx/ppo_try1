"""DRAFT same422 stochastic-kernel version; no controller or observation change."""
from .semantic_p02_progress_profile import (
    P02_PROGRESS_POLICY, P02_PROGRESS_OBSERVATION_LAYOUT, p02_progress_policy_contract)
from .semantic_receiving_wheel_profile import RECEIVING_WHEEL_POLICY
from .semantic_rear_policy_timing_profile import REAR_POLICY_TIMING_START
from .semantic_rr_capture_profile import RR_TASK_START
from .semantic_rear_cooperative_prep_sigma import (
    SIGMA_MODE, PREP_TOTAL_MULTIPLIERS, RR_HIP_INDEX, RR_HIP_MULTIPLIER,
    FL_WHEEL_INDEX, FL_WHEEL_EXTRA_MULTIPLIER)

COOPERATIVE_PREP_POLICY = 'rear_cooperative_prep_history_v1'
COOPERATIVE_PREP_ACTOR_CLASS = (
    'wlr50_clean.ppo.semantic_rear_cooperative_prep_actor:SemanticRearCooperativePrepHistoryMLPModel')
COOPERATIVE_PREP_OBSERVATION_LAYOUT = P02_PROGRESS_OBSERVATION_LAYOUT
COOPERATIVE_PREP_OBSERVATION_DIM = 422
COOPERATIVE_PREP_SIGMA_SEMANTICS = (
    'receiving_parent_then_observed_RRcarry_reachable_OR_RLprep_FRknee16_FLknee2_RLhip2_RLknee8'
    '_RRhip4_unchanged_FLwheel2_only_carry_reachable_without_parent_receiving_v1')


def cooperative_prep_policy_contract(*, observation_layout=None):
    if observation_layout != COOPERATIVE_PREP_OBSERVATION_LAYOUT:
        raise ValueError('cooperative prep requires unchanged explicit role419_p02_progress_v1/422')
    result = p02_progress_policy_contract(observation_layout=observation_layout)
    result.update(version=COOPERATIVE_PREP_POLICY, actor_class=COOPERATIVE_PREP_ACTOR_CLASS,
        source_policy_version=P02_PROGRESS_POLICY,
        observation_layout=COOPERATIVE_PREP_OBSERVATION_LAYOUT, observation_dimension=422,
        preserved_observation_prefix_dimension=422, all_422_numerical_features_unchanged=True,
        observation_schema_change=False, parameter_topology_change=False,
        deterministic_mean_change=False, directional_bias=False,
        stochastic_kernel_change=True, control_or_reward_change=False,
        mdp_change='none_from_this_factor_same_observations_targets_safety_reward_and_source_schedule',
        action_transform='declared_FL_capture_assist_only;rear_task_assists_and_rear_task_wheel_projection_OFF;existing_physical_limits_and_mapper_preserved',
        sigma_mode=SIGMA_MODE, sigma_profile_parent_version=RECEIVING_WHEEL_POLICY,
        sigma_kernel_observation_slice=[0, 422], sigma_parent_observation_slice=[0, 372],
        sigma_scaling_semantics=COOPERATIVE_PREP_SIGMA_SEMANTICS,
        conditional_std='receiving_parent_sigma*declared_TOTAL_rear_cooperative_multiplier',
        effective_log_std='receiving_parent_log_std+log(declared_TOTAL_rear_cooperative_multiplier)',
        rear_local_sigma_gates={
            'RR_hip_unchanged': dict(channel_index=RR_HIP_INDEX, multiplier=RR_HIP_MULTIPLIER,
                gate='rr_carry_capture_AND_rr_top_reachable'),
            'cooperative_preparation': dict(
                gate='(rr_carry_capture_AND_rr_top_reachable)_OR_rl_prep_transfer',
                channel_total_multipliers={str(i): value for i, value in PREP_TOTAL_MULTIPLIERS}),
            'FL_wheel_carry_extra_outside_parent_receiving': dict(channel_index=FL_WHEEL_INDEX,
                multiplier=FL_WHEEL_EXTRA_MULTIPLIER,
                gate='rr_carry_capture_AND_rr_top_reachable_AND_NOT_parent_receiving_continuation_active'),
            'observed_carry_index': REAR_POLICY_TIMING_START,
            'observed_top_reachable_index': RR_TASK_START+1,
            'observed_RL_prep_index': REAR_POLICY_TIMING_START+2,
            'otherwise_multiplier': 1.0,
            'reference': 'receiving_parent_not_previous_rear_multiplier'},
        support_transfer_permission_modified=False, contact_evidence_modified=False,
        no_action_target_mean_shift_rejection_or_owner=True,
        raw_sample_and_log_probability='original_unclipped_raw_Gaussian_sample_and_same_shared_current_kernel_log_probability_before_existing_execution_projection',
        optimizer_migration_requirement='preserve_all_actor_critic_parameters_full_Adam_state_actual_LR_Identity_normalizers_and_RNG;discard_unfinished_old_kernel_rollout',
        rear_sigma_status='bounded_initial_physical_exploration_candidate_not_proven_optimal_or_successful',
        extra_filter_or_sampling_rejection=False)
    return result

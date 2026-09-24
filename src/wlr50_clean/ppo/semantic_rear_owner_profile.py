"""Public fixed owner anchors; the old422 codec/kernel remains the prefix."""
from collections.abc import Mapping
from math import isfinite
from .semantic_rear_cooperative_prep_profile import (
    COOPERATIVE_PREP_POLICY, COOPERATIVE_PREP_OBSERVATION_LAYOUT,
    cooperative_prep_policy_contract)

REAR_OWNER_MODE = 'issued_rear_transfer_owner_suspension_v1'
REAR_OWNER_POLICY = 'rear_owner_recovery_history_v1'
REAR_OWNER_ACTOR_CLASS = 'wlr50_clean.ppo.semantic_rear_owner_actor:SemanticRearOwnerRecoveryHistoryMLPModel'
REAR_OWNER_OBSERVATION_LAYOUT = 'role422_rear_owner_recovery_v1'
REAR_OWNER_GROUP = 'rear_owner_recovery_full17'
REAR_OWNER_START = 422
REAR_OWNER_OBSERVATION_DIM = 439
REAR_OWNER_INDICES = (0, 1, 4, 5)
REAR_OWNER_FIELDS = tuple(f'{kind}_{i}' for kind in ('anchor_final_deg', 'anchor_request_deg',
    'active', 'winning_late_owner') for i in REAR_OWNER_INDICES) + ('rl_edge_recovery_permitted',)
REAR_OWNER_SCALES = (180.,)*8 + (1.,)*9


def rear_owner_features(snapshot):
    keys = ('anchor_final_deg', 'anchor_request_deg', 'active', 'winning_late_owner', 'rl_edge_recovery_permitted')
    if not isinstance(snapshot, Mapping) or any(k not in snapshot for k in keys):
        raise ValueError('all authoritative rear owner fields required')
    numbers, bits = [], []
    for key in keys[:2]:
        values = snapshot[key]
        if (not isinstance(values, (list, tuple)) or len(values) != 4
                or any(type(v) not in (int, float) or not isfinite(v) for v in values)):
            raise ValueError('four finite actual owner anchors required')
        numbers.extend(float(v)/180. for v in values)
    for key in keys[2:4]:
        values = snapshot[key]
        if not isinstance(values, (list, tuple)) or len(values) != 4 or any(type(v) is not bool for v in values):
            raise ValueError('four actual owner flags required')
        bits.extend(float(v) for v in values)
    edge = snapshot[keys[4]]
    if type(edge) is not bool:
        raise ValueError('actual RL edge recovery flag required')
    for j, active in enumerate(snapshot['active']):
        if not active and (numbers[j] != 0. or numbers[j+4] != 0.):
            raise ValueError('inactive owner anchors must be zero')
    return tuple(numbers + bits + [float(edge)])


def rear_owner_policy_contract(*, observation_layout=None):
    if observation_layout != REAR_OWNER_OBSERVATION_LAYOUT:
        raise ValueError('owner recovery requires explicit439 layout')
    result = cooperative_prep_policy_contract(observation_layout=COOPERATIVE_PREP_OBSERVATION_LAYOUT)
    result.update(version=REAR_OWNER_POLICY, actor_class=REAR_OWNER_ACTOR_CLASS,
        source_policy_version=COOPERATIVE_PREP_POLICY,
        observation_layout=REAR_OWNER_OBSERVATION_LAYOUT, observation_dimension=439,
        preserved_observation_prefix_dimension=422, old_422_numerical_codec_preserved=True,
        all_422_numerical_features_unchanged=False,
        observation_schema_change=True, parameter_topology_change=True,
        rear_owner_mode=REAR_OWNER_MODE, rear_owner_group=REAR_OWNER_GROUP,
        rear_owner_slice=[422,439], rear_owner_fields=list(REAR_OWNER_FIELDS),
        rear_owner_scales=list(REAR_OWNER_SCALES),
        dependent_owner_name_semantics='winning_late_owner_includes_P09_late_and_P12_dependent_servo_owners',
        owner_observation_clock='anchors_and_active_last_committed_dispatch;winning_owner_and_edge_current_context;active_without_current_owner_is_visible_pending_release',
        sigma_kernel_observation_slice=[0,422], stochastic_kernel_change=False,
        deterministic_mean_change='old422_columns_preserved_new17_zero; same_numeric_prefix_initial_mean_sigma_exact',
        control_or_reward_change=True,
        mdp_change='public_issued_owner_suspension_RL_edge_recovery_and_versioned_preparation_proxies',
        action_transform='existing_single_mapper; suspended_dependent_FL_RL_servo_owner_uses_committed_FINAL_anchor_plus_bounded_request_change; wheels_and_explicit_stops_preserved; rear_completion_assists_OFF_FLassist_declared',
        no_action_target_mean_shift_rejection_or_owner=False,
        support_transfer_permission_modified='RL_edge_recovery_not_false_bearing_or_full_strong_transfer',
        optimizer_migration_requirement='old_actor_critic_columns_full_Adam_options_steps_LR_Identity_RNG_exact; new17_first_weight_and_moments_zero; empty_rollout')
    return result

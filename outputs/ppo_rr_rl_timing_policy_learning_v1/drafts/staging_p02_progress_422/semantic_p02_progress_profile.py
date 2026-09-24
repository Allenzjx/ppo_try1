"""Explicit P02 progress-credit state appended to the unchanged rear419 codec."""
from collections.abc import Mapping
from math import isfinite
from .semantic_rear_policy_timing_profile import (
    REAR_POLICY_TIMING_OBSERVATION_LAYOUT, rear_policy_timing_policy_contract)

P02_PROGRESS_MODE = 'p02_measured_progress_credit_v1'
P02_PROGRESS_POLICY = 'rear_policy_p02_progress_history_v1'
P02_PROGRESS_ACTOR_CLASS = 'wlr50_clean.ppo.semantic_p02_progress_actor:SemanticP02ProgressHistoryMLPModel'
P02_PROGRESS_OBSERVATION_LAYOUT = 'role419_p02_progress_v1'
P02_PROGRESS_GROUP = 'p02_progress_credit_full3'
P02_PROGRESS_FIELDS = ('p02_best_remaining_m', 'p02_progress_credit_fraction', 'p02_current_eligible')
P02_PROGRESS_SNAPSHOT_FIELDS = ('best_remaining_m', 'credit_fraction', 'current_eligible')
P02_PROGRESS_START = 419
P02_PROGRESS_OBSERVATION_DIM = 422


def p02_progress_features(snapshot, stage_id):
    if not isinstance(snapshot, Mapping) or any(k not in snapshot for k in P02_PROGRESS_SNAPSHOT_FIELDS):
        raise ValueError('P02 credit requires all three authoritative public fields')
    best, credit, eligible = (snapshot[k] for k in P02_PROGRESS_SNAPSHOT_FIELDS)
    if (isinstance(best, bool) or not isinstance(best, (int, float)) or not isfinite(best) or best < 0
            or isinstance(credit, bool) or not isinstance(credit, (int, float)) or not isfinite(credit)
            or not 0 <= credit <= 1 or type(eligible) is not bool):
        raise ValueError('invalid public P02 progress-credit state')
    result = (float(best), float(credit), float(eligible))
    if stage_id != 'P02' and result != (0., 0., 0.):
        raise ValueError('P02 credit features must be zero outside P02')
    return result


def p02_progress_policy_contract(*, observation_layout=None):
    if observation_layout != P02_PROGRESS_OBSERVATION_LAYOUT:
        raise ValueError('P02 progress requires its explicit 422 layout')
    result = rear_policy_timing_policy_contract(observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT)
    result.update(version=P02_PROGRESS_POLICY, actor_class=P02_PROGRESS_ACTOR_CLASS,
        observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT, observation_dimension=422,
        preserved_observation_prefix_dimension=419, p02_progress_mode=P02_PROGRESS_MODE,
        p02_progress_observation_group=P02_PROGRESS_GROUP,
        p02_progress_observation_slice=[419, 422], p02_progress_observation_fields=list(P02_PROGRESS_FIELDS),
        p02_progress_feature_scales=[1., 1., 1.], old_419_numerical_codec_preserved=True,
        sigma_kernel_observation_slice=[0, 419],
        mdp_change='measured_P02_progress_credit_controls_local_continuation_with_all_three_state_fields_public',
        deterministic_mean_change='zero_append_preserves_initial_function; new_columns_learn_in_ordinary_PPO')
    return result

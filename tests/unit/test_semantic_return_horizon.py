"""Versioned return-profile/reward tests; no native simulation evidence."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from test_semantic_observation_reward_env import _built, _frame, _sample
from wlr50_clean.ppo.semantic_policy_distribution import (
    LEGACY_POLICY, STATE_DEPENDENT_POLICY, policy_contract, policy_version_from_metadata,
)
from wlr50_clean.ppo.semantic_return_profile import LEGACY_RETURN_PROFILE, RETURN_PROFILE
from wlr50_clean.ppo.semantic_reward import SemanticRewardCalculator, load_semantic_reward_config
from wlr50_clean.ppo.semantic_training import semantic_runner_config


ROOT = Path(__file__).resolve().parents[2]
V2_REWARD = ROOT / "configs/ppo_semantic_v2/reward_config.yaml"
V3_REWARD = ROOT / "configs/ppo_semantic_v3/reward_config.yaml"


def _metadata(version=STATE_DEPENDENT_POLICY, *, profile=None, semantic_version="v3"):
    config = semantic_runner_config(seed=1001, device="cpu", semantic_version=semantic_version,
                                    policy_version=version, return_profile=profile)
    return {"seed": 1001, "semantic_version": semantic_version,
            "runner_config": config, "policy_contract": policy_contract(version)}


def _reward(previous, current, *, ticks=8, reason=None, success=False, path=V3_REWARD):
    calculator = SemanticRewardCalculator(load_semantic_reward_config(path))
    return calculator.evaluate(previous, current, [_sample(previous, current)] * ticks,
                               termination_reason=reason, task_success=success)


def test_target_gamma_single_source_and_original_failure_guard_survives():
    v2, v3 = load_semantic_reward_config(V2_REWARD), load_semantic_reward_config(V3_REWARD)
    old = semantic_runner_config(seed=1001, device="cpu", semantic_version="v2")
    new = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3")
    assert old["algorithm"]["gamma"] == v2.gamma == .995
    assert old["algorithm"]["lam"] == .95
    assert "semantic_return_profile" not in old
    assert new["algorithm"]["gamma"] == v3.gamma == .9985
    assert new["algorithm"]["lam"] == .99
    assert new["semantic_return_profile"] == v3.values["return_profile"] == RETURN_PROFILE
    # Frozen v2 retains its historical nonzero regularization weight; the old
    # v3 bound was 14.6, which must not be substituted for v2's actual bound.
    assert v2.failure_avoidance_bound == pytest.approx(14.613333333333333)
    assert v3.failure_avoidance_bound == pytest.approx(37)
    assert v3.values["failure_cost"] == v3.values["success_reward"] == 40
    assert v3.values["failure_cost"] > v3.failure_avoidance_bound


@pytest.mark.parametrize("change", ["gamma", "bool_gamma", "nan_gamma", "profile", "missing_profile", "insufficient_failure"])
def test_reward_rejects_unknown_mixed_or_unsafe_profile(tmp_path, change):
    values = copy.deepcopy(dict(load_semantic_reward_config(V3_REWARD).values))
    if change == "gamma": values["gamma"] = .9995
    elif change == "bool_gamma": values["gamma"] = True
    elif change == "nan_gamma": values["gamma"] = float("nan")
    elif change == "profile": values["return_profile"] = "unreviewed_profile"
    elif change == "missing_profile": values.pop("return_profile")
    else: values["failure_cost"] = 37.0
    selected = tmp_path / "invalid_reward.yaml"
    selected.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_semantic_reward_config(selected)


@pytest.mark.parametrize("policy", [LEGACY_POLICY, STATE_DEPENDENT_POLICY])
@pytest.mark.parametrize("profile", [LEGACY_RETURN_PROFILE, RETURN_PROFILE])
def test_historical_and_current_complete_runner_metadata_are_distinct(policy, profile):
    info = _metadata(policy, profile=profile)
    assert policy_version_from_metadata(info) == policy
    cfg = info["runner_config"]
    if profile == LEGACY_RETURN_PROFILE:
        assert "semantic_return_profile" not in cfg
        assert (cfg["algorithm"]["gamma"], cfg["algorithm"]["lam"]) == (.995, .95)
    else:
        assert cfg["semantic_return_profile"] == RETURN_PROFILE
        assert (cfg["algorithm"]["gamma"], cfg["algorithm"]["lam"]) == (.9985, .99)


@pytest.mark.parametrize("change", ["missing_marker", "unknown_marker", "legacy_gamma", "legacy_lambda",
                                    "unknown_gamma", "bool_gamma", "nan_lambda", "other_field"])
def test_current_metadata_cannot_fall_back_to_legacy_or_ignore_nonreturn_fields(change):
    info = _metadata()
    cfg = info["runner_config"]
    if change == "missing_marker": cfg.pop("semantic_return_profile")
    elif change == "unknown_marker": cfg["semantic_return_profile"] = "unreviewed_profile"
    elif change == "legacy_gamma": cfg["algorithm"]["gamma"] = .995
    elif change == "legacy_lambda": cfg["algorithm"]["lam"] = .95
    elif change == "unknown_gamma": cfg["algorithm"]["gamma"] = .9995
    elif change == "bool_gamma": cfg["algorithm"]["gamma"] = True
    elif change == "nan_lambda": cfg["algorithm"]["lam"] = float("nan")
    else: cfg["actor"]["hidden_dims"] = [256, 128]
    with pytest.raises(ValueError):
        policy_version_from_metadata(info)


def test_new_marker_cannot_be_put_on_old_parameters_or_v2():
    legacy = _metadata(profile=LEGACY_RETURN_PROFILE)
    legacy["runner_config"]["semantic_return_profile"] = RETURN_PROFILE
    with pytest.raises(ValueError):
        policy_version_from_metadata(legacy)
    with pytest.raises(ValueError):
        semantic_runner_config(seed=1001, device="cpu", semantic_version="v2", return_profile=RETURN_PROFILE)


def test_profile_change_only_changes_gamma_lambda_and_explicit_marker_in_runner():
    old = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
                                policy_version=STATE_DEPENDENT_POLICY, return_profile=LEGACY_RETURN_PROFILE)
    new = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
                                policy_version=STATE_DEPENDENT_POLICY)
    expected = copy.deepcopy(old)
    expected["algorithm"].update(gamma=.9985, lam=.99)
    expected["semantic_return_profile"] = RETURN_PROFILE
    assert new == expected
    assert new["num_steps_per_env"] == 128


def test_eight_physics_samples_receive_one_new_gamma_and_unchanged_quality_costs():
    before, after = _built(_frame(phi=.2)), _built(_frame(8, phi=.7))
    new = _reward(before, after)
    old = _reward(before, after, path=V2_REWARD)
    assert new["potential_shaping"] == pytest.approx(5 * (.9985 * .7 - .2))
    assert new["elapsed_physics_s"] == pytest.approx(8 / 120)
    assert new["terminal_bootstrap_allowed"] is True
    assert new["terminal_event"] == 0
    # This motion-free fixture has zero costs under both historical and current
    # contact/smoothness semantics; it is not a general old/new physics equivalence.
    for family in ("body_stability", "contact_motion_quality", "control_smoothness", "control_regularization"):
        assert new["families"][family] == old["families"][family] == 0


@pytest.mark.parametrize("source,target", [("P07", "P08"), ("P08", "P09"), ("P09", "P10"),
                                         ("P10", "P11"), ("P11", "P12"), ("P12", "P13")])
def test_phase_labels_do_not_reset_new_global_potential_or_award_transition(source, target):
    before = _built(_frame(stage=source, phi=.65))
    same = _built(_frame(8, stage=source, phi=.65))
    changed = _built(_frame(8, stage=target, phi=.65))
    a, b = _reward(before, same), _reward(before, changed)
    assert a == b
    assert b["potential_after"] == b["potential_before"] == .65
    assert b["terminal_event"] == 0 and b["terminal_bootstrap_allowed"] is True


@pytest.mark.parametrize("reason", ["BODY_COLLISION", "WHEEL_ONLY_CLIMB", "INCOMPLETE_CONTROLLER_BLOCKED",
                                   "TASK_TIMEOUT", "FALL", "HARD_JOINT_LIMIT", "NAN_INF"])
@pytest.mark.parametrize("ticks", [1, 3, 8])
def test_short_and_full_failure_intervals_keep_absorbing_potential_and_original_event(reason, ticks):
    before, after = _built(_frame(phi=.65)), _built(_frame(ticks, phi=.85))
    result = _reward(before, after, ticks=ticks, reason=reason)
    assert result["potential_shaping"] == pytest.approx(-5 * .65)
    assert result["potential_after"] == 0
    assert result["terminal_event"] == -40
    assert result["terminal_bootstrap_allowed"] is False
    assert result["elapsed_physics_s"] == pytest.approx(ticks / 120)


@pytest.mark.parametrize("terminal", [False, True])
def test_new_discounted_cycle_and_absorbing_telescoping_are_exact(terminal):
    phi = (.2, .8, .5, .2)
    states = [_built(_frame(8 * i, phi=p)) for i, p in enumerate(phi)]
    shaped = []
    for i, (previous, current) in enumerate(zip(states, states[1:])):
        reason = "INCOMPLETE_CONTROLLER_BLOCKED" if terminal and i == 2 else None
        shaped.append(_reward(previous, current, reason=reason)["potential_shaping"])
    total = sum(.9985 ** i * value for i, value in enumerate(shaped))
    expected = -5 * phi[0] if terminal else 5 * (.9985 ** 3 * phi[-1] - phi[0])
    assert total == pytest.approx(expected, abs=2e-15)
    assert total < 0


def test_success_keeps_one_positive_event_and_zero_terminal_potential():
    before, after = _built(_frame(phi=.85)), _built(_frame(8, success=True))
    result = _reward(before, after, reason="SUCCESS", success=True)
    assert result["terminal_event"] == 40
    assert result["potential_shaping"] == pytest.approx(-5 * .85)
    assert result["potential_after"] == 0 and result["terminal_bootstrap_allowed"] is False

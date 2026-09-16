"""Task recovery changes reward preferences, not task/safety or residual signs."""
from copy import deepcopy

import pytest
import yaml

from test_semantic_observation_reward_env import _built, _frame, _raw, _sample
from wlr50_clean.ppo.semantic_reward import (
    FAMILIES, SemanticRewardCalculator, _validate_task_priority,
    load_semantic_reward_config,
)

OLD = load_semantic_reward_config().path.parent.parent / "ppo_fsm_reference_p09_stable_v2"
NEW = OLD.parent / "ppo_task_first_recovery_v1"


def calculator():
    return SemanticRewardCalculator(load_semantic_reward_config(NEW / "reward_config.yaml"))


def evaluate(before, after, **sample_values):
    return calculator().evaluate(before, after, [_sample(before, after, **sample_values)],
        termination_reason=after.task.get("termination_reason"),
        task_success=after.task.get("success", False))


@pytest.mark.parametrize("name", ["action_schema.json", "execution_profile.yaml",
    "observation_schema.json", "quality_score.yaml", "stage_task_spec.yaml"])
def test_successful_N_and_physics_and_schema_are_semantically_identical(name):
    assert yaml.safe_load((OLD / name).read_text(encoding="utf-8")) == yaml.safe_load(
        (NEW / name).read_text(encoding="utf-8"))


def test_epsilon_zero_is_explicit_and_all_measured_quality_costs_stay_diagnostic():
    before = _built(_frame(raw=_raw(pitch=.3)))
    after = _built(_frame(1, raw=_raw(1, pitch=.4)))
    sample = _sample(before, after, residual=(1.,)*12, actual_drive=(1.,)*12)
    new = calculator().evaluate(before, after, [sample], termination_reason=None, task_success=False)
    old = SemanticRewardCalculator(load_semantic_reward_config(OLD / "reward_config.yaml")).evaluate(
        before, after, [sample], termination_reason=None, task_success=False)
    assert new["quality_epsilon"] == 0.
    assert all(new["families"][name] == 0. for name in FAMILIES[1:])
    assert new["cost_components"] == old["cost_components"]
    assert new["unweighted_families"] == old["unweighted_families"]
    assert new["total"] == new["families"]["task_progress"] == old["families"]["task_progress"]


@pytest.mark.parametrize("sign", [-1., 0., 1.])
def test_spinning_wheels_without_physical_progress_is_not_task_reward(sign):
    before, after = _built(), _built(_frame(1))
    result = evaluate(before, after, residual=(0.,)*8+(sign,)*4,
                      actual_drive=(0.,)*8+(sign,)*4)
    assert result["terminal_event"] == 0.
    assert result["total"] < 0.


def test_useful_transfer_can_tilt_without_quality_penalty_and_retreat_is_not_new_success():
    before = _built(_frame(phi=.2))
    forward_tilted = _built(_frame(1, phi=.3, raw=_raw(1, pitch=.4, roll=.3)))
    forward_level = _built(_frame(1, phi=.3))
    fallen_back_level = _built(_frame(1, phi=.1))
    assert evaluate(before, forward_tilted)["total"] == evaluate(before, forward_level)["total"]
    assert evaluate(before, forward_tilted)["total"] > 0.
    assert evaluate(before, fallen_back_level)["total"] < 0.


@pytest.mark.parametrize("reason", ["BODY_COLLISION", "WHEEL_ONLY_CLIMB", "FALL", "HARD_JOINT_LIMIT", "NAN_INF", "TASK_TIMEOUT"])
def test_real_failure_terminal_remains_negative_and_no_bootstrap(reason):
    result = evaluate(_built(), _built(_frame(1, reason=reason)))
    assert result["terminal_event"] == -40.
    assert result["potential_after"] == 0.
    assert result["terminal_bootstrap_allowed"] is False
    assert result["total"] < -40.


def test_legal_different_postures_keep_same_success_event():
    before = _built(_frame(phi=.9))
    a = evaluate(before, _built(_frame(1, success=True)))
    b = evaluate(before, _built(_frame(1, success=True, raw=_raw(1, pitch=.1, roll=.1))))
    assert a["total"] == b["total"]
    assert a["terminal_event"] == 40.
    assert not a["terminal_bootstrap_allowed"]


def test_phase_handoff_and_nonterminal_rollout_tail_keep_potential_and_bootstrap():
    before = _built(_frame(stage="P02", phi=.2))
    after = _built(_frame(1, stage="P03", phi=.3))
    result = evaluate(before, after)
    assert result["potential_before"] == .2 and result["potential_after"] == .3
    assert result["terminal_bootstrap_allowed"] is True
    assert result["terminal_event"] == 0.


def test_complete_discounted_shaping_telescopes_even_with_temporary_retreat():
    cfg = calculator().config
    totals = []
    for path in ((.1, .5, .3, .9), (.1, .3, .5, .9)):
        states = [_built(_frame(i, phi=phi)) for i, phi in enumerate(path)]
        states.append(_built(_frame(4, success=True)))
        rows = [evaluate(a, b) for a, b in zip(states, states[1:])]
        shaped = sum(cfg.gamma**i*r["potential_shaping"] for i,r in enumerate(rows))
        assert shaped == pytest.approx(-5*path[0])
        totals.append(sum(cfg.gamma**i*r["total"] for i,r in enumerate(rows)))
    assert totals[0] == pytest.approx(totals[1])


@pytest.mark.parametrize("mutation", ["nonzero", "bool", "unbound", "weight", "task_weight"])
def test_objective_marker_rejects_silent_quality_or_task_weight_changes(mutation):
    values = deepcopy(calculator().config.values)
    if mutation == "nonzero": values["quality_epsilon"] = .1
    elif mutation == "bool": values["quality_epsilon"] = False
    elif mutation == "unbound": values.pop("objective_profile")
    elif mutation == "weight": values["family_weights"]["body_stability"] = .4
    else: values["family_weights"]["task_progress"] = 2.
    with pytest.raises(ValueError):
        _validate_task_priority(values)


def test_failure_bound_is_not_a_guarantee_against_delaying_negative_terminal():
    cfg = calculator().config
    assert cfg.failure_avoidance_bound == pytest.approx(5 + .02/15/(1-.9985))
    def finite_failed_return(n):
        return -sum(cfg.gamma**t*.02/15 for t in range(n))-40*cfg.gamma**(n-1)
    assert finite_failed_return(600) > finite_failed_return(100)

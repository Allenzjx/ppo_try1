"""Versioned role activity changes only bounded body-motion cost weighting."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from test_semantic_observation_reward_env import _built, _frame, _sample
from wlr50_clean.ppo.semantic_reward import (
    FAMILIES, ROLE_TRANSFER_VERSION, SemanticRewardCalculator, load_semantic_reward_config,
)

CFG = Path(__file__).resolve().parents[2] / "configs/ppo_semantic_v3/reward_config.yaml"


def frame(tick=0, *, fraction=0., version=ROLE_TRANSFER_VERSION, stage="P09", substage="EXECUTION",
          phi=.4, contact=True, vertical=0.):
    built = _built(_frame(tick, stage=stage, phi=phi))
    task = {**built.task, "physical_transfer_fraction": fraction, "substage": substage}
    if version is not None:
        task["transfer_roles_version"] = version
    metrics = deepcopy(built.metrics)
    metrics.update(rpy=(.2, -.1, 0.), euler_roll_pitch_rate=(.8, -.4),
                   body_angular_acceleration=(4., -2., 6.))
    for wheel in metrics["wheels"]:
        wheel.update(contact=contact, vertical_velocity=vertical, slip_speed=0., chatter=0.)
    return replace(built, task=task, metrics=metrics)


def calc():
    return SemanticRewardCalculator(load_semantic_reward_config(CFG))


def reward(before, after, *, calculator=None, reason=None, **sample_changes):
    return (calculator or calc()).evaluate(
        before, after, [_sample(before, after, **sample_changes)],
        termination_reason=reason, task_success=reason == "SUCCESS",
    )


@pytest.mark.parametrize("fraction", [0., .25, .5, .75, 1.])
def test_exact_role_version_blends_all_body_components_using_existing_coefficient(fraction):
    before, after = frame(), frame(1, fraction=fraction)
    result = reward(before, after)
    legacy = reward(before, replace(after, task={**after.task, "transfer_roles_version": None}))
    weight = 1. - .8*fraction
    base_attitude = ((.2/.5)**2 + (-.1/.5)**2)/2
    base_rates = ((.8/2)**2 + (-.4/2)**2)/2
    base_acc = ((4/20)**2 + (-2/20)**2 + (6/20)**2)/3
    for name, base in (("gravity_attitude", base_attitude), ("euler_rate", base_rates),
                       ("angular_acceleration", base_acc)):
        assert result["cost_components"][name] == pytest.approx(base*weight/120)
    assert result["cost_components"]["gravity_attitude"] == legacy["cost_components"]["gravity_attitude"]
    assert legacy["cost_components"]["euler_rate"] == pytest.approx(base_rates/120)
    assert legacy["cost_components"]["angular_acceleration"] == pytest.approx(base_acc/120)
    assert result["families"]["body_stability"] == pytest.approx(
        -.4*(base_attitude+base_rates+base_acc)*weight/(3*120))
    assert tuple(result["families"]) == FAMILIES
    for family in FAMILIES:
        if family != "body_stability":
            assert result["families"][family] == legacy["families"][family]


@pytest.mark.parametrize("version", [None, "legacy", "diagonal_transfer_roles_v2", "diagonal_transfer_roles_v1_typo"])
def test_missing_or_other_version_keeps_exact_legacy_result(version):
    before, after = frame(version=None), frame(1, fraction=.8, version=None)
    expected = reward(before, after)
    other = replace(after, task={**after.task, "transfer_roles_version": version})
    assert reward(before, other) == expected


@pytest.mark.parametrize("substage", ["TRANSFER", "CAPTURE", "SETTLE", "EXECUTION"])
def test_zero_measured_activity_restores_cost_even_when_stage_label_says_transfer(substage):
    before, after = frame(), frame(1, fraction=0., substage=substage, stage="P13")
    expected = reward(before, frame(1, fraction=0., version=None, stage="P13"))
    assert reward(before, after) == expected


def test_full_to_half_to_zero_activity_restores_cost_smoothly_without_history_reset():
    calculator = calc()
    states = [frame(i, fraction=f) for i, f in enumerate((1., 1., .5, 0.))]
    costs = [-reward(a, b, calculator=calculator)["families"]["body_stability"]
             for a, b in zip(states, states[1:])]
    assert costs[0]/costs[2] == pytest.approx(.2)
    assert costs[1]/costs[2] == pytest.approx(.6)
    assert calculator._clock_s == pytest.approx(3/120)


@pytest.mark.parametrize("fraction", [None, float("nan"), float("inf"), -0.01, 1.01])
def test_named_role_contract_requires_finite_bounded_measured_activity(fraction):
    calculator = calc()
    before, after = frame(), frame(1, fraction=fraction)
    with pytest.raises(ValueError, match="role transfer fraction"):
        reward(before, after, calculator=calculator)
    assert calculator._clock_s == 0.


def test_named_role_contract_never_silently_falls_back_to_substage_if_activity_missing():
    calculator = calc()
    before, after = frame(), frame(1, substage="TRANSFER")
    task = dict(after.task)
    del task["physical_transfer_fraction"]
    with pytest.raises(ValueError, match="role transfer fraction"):
        reward(before, replace(after, task=task), calculator=calculator)
    assert calculator._clock_s == 0.


@pytest.mark.parametrize("stage", ["P01", "P05", "P08", "P09", "P10", "P12", "P13"])
def test_same_measured_role_activity_is_independent_of_phase_label(stage):
    before, after = frame(fraction=.6), frame(1, fraction=.6)
    relabeled = replace(after, task={**after.task, "stage_id": stage})
    assert reward(before, relabeled) == reward(before, after)


@pytest.mark.parametrize("reason", [None, "SUCCESS", "FALL", "BODY_COLLISION", "INCOMPLETE_CONTROLLER_BLOCKED"])
def test_terminal_and_potential_terms_are_identical_to_legacy(reason):
    before, after = frame(phi=.3), frame(1, fraction=.7, phi=.6)
    current = reward(before, after, reason=reason)
    old = reward(before, replace(after, task={**after.task, "transfer_roles_version": None}), reason=reason)
    for key in ("potential_before", "potential_after", "potential_shaping", "terminal_event",
                "terminal_bootstrap_allowed", "discount_convention"):
        assert current[key] == old[key]
    assert current["families"]["task_progress"] == old["families"]["task_progress"]
    if reason is not None:
        assert current["potential_after"] == 0.
        assert current["terminal_bootstrap_allowed"] is False
        assert current["terminal_event"] == (40. if reason == "SUCCESS" else -40.)


def test_eight_measured_ticks_integrate_each_activity_not_only_last_or_elapsed_label():
    states = [frame(i, fraction=i/8) for i in range(9)]
    samples = [_sample(a, b) for a, b in zip(states, states[1:])]
    result = calc().evaluate(states[0], states[-1], samples, termination_reason=None, task_success=False)
    independent_body = sum(reward(a, b)["families"]["body_stability"] for a, b in zip(states, states[1:]))
    assert result["families"]["body_stability"] == pytest.approx(independent_body)
    assert result["elapsed_physics_s"] == pytest.approx(8/120)


def test_receiver_air_and_force_alone_do_not_create_new_penalty_or_reward():
    before, after = frame(contact=False), frame(1, contact=False, vertical=.8, fraction=1.)
    air = reward(before, after)
    assert air["families"]["contact_motion_quality"] == 0.
    assert air["cost_components"]["confirmed_post_touchdown_rebound"] == 0.
    supported = frame(1, contact=True, fraction=1.)
    metrics = deepcopy(supported.metrics)
    for wheel in metrics["wheels"]:
        wheel["force"] *= 100
    # Continuous stationary contact is neither a touchdown nor a large-force prize.
    high = reward(frame(contact=True), replace(supported, metrics=metrics))
    assert high["families"]["contact_motion_quality"] == 0.


@pytest.mark.parametrize("active", [False, True])
def test_role_blend_keeps_existing_active_departure_vs_passive_rebound_filter(active):
    calculator = calc()
    a = frame(contact=False, vertical=-.4, fraction=1.)
    b = frame(1, contact=True, fraction=1.)
    c = frame(2, contact=False, vertical=.5, fraction=1.)
    reward(a, b, calculator=calculator)
    out = reward(b, c, calculator=calculator, actual_drive=((1.,)+(0.,)*11 if active else (0.,)*12))
    assert (out["cost_components"]["confirmed_post_touchdown_rebound"] > 0.) is (not active)


def test_nominal_residual_relabeling_stays_diagnostic_and_regularizer_stays_off():
    before, after = frame(), frame(1, fraction=1.)
    out = reward(before, after, nominal=(1.,)*12, residual=(-1.,)*12)
    assert out["cost_components"]["nominal_first_difference"] > 0.
    assert out["cost_components"]["residual_first_difference"] > 0.
    assert out["families"]["control_smoothness"] == 0.
    assert out["families"]["control_regularization"] == 0.
    assert calc().config.gamma == .9985
    assert len(out["families"]) == 5

"""New soft learning signals; no altered hard events or nominal controller."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from test_semantic_capture_approach import _qualified, _step, _supervisor, _counterfactual
from test_semantic_observation_reward_env import _built, _frame, _raw, _sample
from wlr50_clean.ppo.semantic_reward import (
    SemanticRewardCalculator, load_semantic_reward_config, _front_quality_substate,
)
from wlr50_clean.ppo.semantic_supervisor import load_task_spec

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs/ppo_fl_capture_quality_v1"
OLD = ROOT / "configs/ppo_residual_rr_fix_v1"


def _new():
    return load_task_spec(CFG / "stage_task_spec.yaml")


def _reward(phase="P02", fraction=1., history=None, current=None, *, roll=.2, rates=(.2, .3)):
    before = _built(_frame(0, stage=phase))
    after = _built(_frame(1, stage=phase, raw=_raw(1, roll=roll, pitch=.1)))
    task = {**after.task, "transfer_roles_version": "diagonal_transfer_roles_v1",
            "physical_transfer_fraction": fraction, "physical_evaluator": {
                "history": history or {}, "current_legs": {"FR": current or {}}}}
    after = replace(after, task=task, metrics={**after.metrics, "euler_roll_pitch_rate": rates})
    calculator = SemanticRewardCalculator(load_semantic_reward_config(CFG / "reward_config.yaml"))
    return calculator.evaluate(before, after, [_sample(before, after)],
                               termination_reason=None, task_success=False)


@pytest.mark.parametrize("phase", ["P01", "P02"])
@pytest.mark.parametrize("fraction", [0., .3, 1.])
def test_front_cost_positive_during_transfer_and_equals_audited_terms(phase, fraction):
    result = _reward(phase, fraction)
    sample = result["front_quality_sample_audit"][0]
    beta = .03 * (1. - .5 * fraction)
    assert sample["effective_beta_per_s"] == pytest.approx(beta)
    assert sample["raw_tilt_cost"] == pytest.approx((.2**2 + .1**2) / (2*.5**2))
    assert sample["raw_rate_cost"] == pytest.approx((.2**2 + .3**2) / (2*.5**2))
    expected = beta * .5 * (sample["raw_tilt_cost"] + sample["raw_rate_cost"]) / 120
    assert expected > 0
    assert sample["weighted_quality_cost"] == pytest.approx(expected)
    assert result["families"]["body_stability"] == pytest.approx(-expected)
    parts = result["cost_components"]
    assert parts["front_weighted_tilt_cost"] + parts["front_weighted_rate_cost"] == pytest.approx(expected)
    assert result["families"]["contact_motion_quality"] == 0
    assert result["families"]["control_smoothness"] == 0
    assert result["families"]["control_regularization"] == 0


@pytest.mark.parametrize("phase", [f"P{i:02d}" for i in range(3, 14)])
def test_no_quality_expansion_to_other_phases(phase):
    result = _reward(phase)
    assert result["families"]["body_stability"] == 0
    assert result["front_quality_sample_audit"] == []


@pytest.mark.parametrize("history,current,label", [
    ({}, {}, "UNQUALIFIED_PREPARATION"),
    ({"active_lift": {"FR": True}}, {"air": False, "ground_contact": True}, "QUALIFIED_LIFT_OR_TRANSFER"),
    ({"active_lift": {"FR": True}}, {"air": True, "ground_contact": False, "clearance_m": .02}, "FUNCTIONAL_CARRY"),
    ({"active_lift": {"FR": True}, "front_edge_crossed": {"FR": True}}, {}, "POST_CROSS_CAPTURE"),
    ({"placed": {"FR": True}}, {"air": True}, "CAPTURED"),
])
def test_real_task_envelope_substates_are_audit_only_not_quality_escape(history, current, label):
    result = _reward(history=history, current=current)
    row = result["front_quality_sample_audit"][0]
    assert row["substate"] == label
    assert row["effective_beta_per_s"] == .015
    assert result["families"]["body_stability"] < 0


def test_actual_supervisor_payload_is_used_for_audit_label():
    ev, _, _ = _qualified("FR")
    sup = _supervisor(_new(), evaluator=ev)
    task = {"physical_evaluator": ev.snapshot}
    assert _front_quality_substate(task) == "POST_CROSS_CAPTURE"
    assert sup.predicate("placed_FR", ev.snapshot) < 1


def test_old_task_first_zero_quality_is_preserved():
    before, after = _built(), _built(_frame(1, raw=_raw(1, roll=.3)))
    calc = SemanticRewardCalculator(load_semantic_reward_config(OLD / "reward_config.yaml"))
    result = calc.evaluate(before, after, [_sample(before, after)], termination_reason=None, task_success=False)
    assert result["quality_epsilon"] == 0
    assert all(value == 0 for family, value in result["families"].items() if family != "task_progress")
    assert "front_quality_sample_audit" not in result


@pytest.mark.parametrize("mutation", [
    lambda v: v["front_body_quality"].update(transfer_weight_floor=0.),
    lambda v: v["family_weights"].update(control_smoothness=.01),
    lambda v: v.update(quality_epsilon=0.),
    lambda v: v.update(objective_profile="task_first_recovery_v1"),
])
def test_unreviewed_quality_expansion_is_rejected(tmp_path, mutation):
    values = yaml.safe_load((CFG / "reward_config.yaml").read_text(encoding="utf-8"))
    mutation(values)
    path = tmp_path / "reward.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_semantic_reward_config(path)


def test_fl_mm_gap_gradient_and_true_contact_remain_separate():
    spec = _new()
    # The established sensor fixture predates RR-v3 wheel-velocity inputs.
    # Earn unchanged FR/FL events with that real evaluator, then exercise only
    # this new potential against its resulting physical snapshot.
    ev, obs, _ = _qualified("FL")
    sup = _supervisor(spec, evaluator=ev)
    prior = deepcopy(ev.snapshot["history"])
    values = []
    for gap in (.025, .005, .0034701283946961467, .001, 0.):
        obs, snap = _step(ev, obs, "FL", x=.53, bottom=.05+gap, air=True)
        expected = .5 * (.5/(1+gap/.025) + .5/(1+gap/.003))
        assert sup._current_capture_progress("FL", snap, 0.) == pytest.approx(expected)
        values.append(sup.physical_potential(snap))
        assert snap["history"] == prior
        assert not snap["history"]["placed"]["FL"]
    assert all(a < b for a, b in zip(values, values[1:]))
    obs, first = _step(ev, obs, "FL", x=.53, bottom=.05, top=True)
    assert not first["history"]["placed"]["FL"]
    obs, captured = _step(ev, obs, "FL", x=.53, bottom=.05, top=True)
    assert captured["history"]["placed"]["FL"]
    # After capture, the legacy current-retention path is used exactly.
    old = _supervisor(load_task_spec(OLD / "stage_task_spec.yaml"), evaluator=ev)
    assert sup.physical_potential(captured) == old.physical_potential(captured)


def test_fl_negative_gap_plateaus_and_outside_region_has_no_descent_signal():
    ev, _, _ = _qualified("FL")
    sup = _supervisor(_new(), evaluator=ev)
    for outside in (False, True):
        values = []
        for gap in (0., -.001, -.010):
            snap = _counterfactual(ev.snapshot, "FL", clearance_m=gap,
                                   within_top_xy=not outside, top_xy_outside_distance_m=.01 if outside else 0.)
            values.append(sup._current_capture_progress("FL", snap, 0.))
        assert values[0] == values[1] == values[2] == (0. if outside else .5)
    for gap in (.001, .025):
        snap = _counterfactual(ev.snapshot, "FL", clearance_m=gap, within_top_xy=False)
        assert sup._current_capture_progress("FL", snap, 0.) == 0
    snap = deepcopy(ev.snapshot)
    snap["history"]["front_edge_crossed"]["FL"] = False
    assert sup._current_capture_progress("FL", snap, 0.) == 0


@pytest.mark.parametrize("leg", ["FR", "RR", "RL"])
def test_non_fl_capture_formula_is_unchanged(leg):
    # These isolated old sensor fixtures exercise formula equivalence, not RR
    # v3 event qualification (which has its own real-data acceptance tests).
    ev, _, _ = _qualified(leg)
    old = _supervisor(load_task_spec(OLD / "stage_task_spec.yaml"), evaluator=ev)
    new = _supervisor(_new(), evaluator=ev)
    for gap in (-.001, 0., .003, .025):
        snap = _counterfactual(ev.snapshot, leg, clearance_m=gap)
        assert new._current_capture_progress(leg, snap, 0.) == old._current_capture_progress(leg, snap, 0.)


def test_new_spec_does_not_change_physical_or_nominal_config():
    old, new = load_task_spec(OLD / "stage_task_spec.yaml"), _new()
    differences = {key for key in old.keys() | new.keys() if old.get(key) != new.get(key)}
    assert differences == {"revision", "capture_approach_semantics", "fl_capture_potential"}

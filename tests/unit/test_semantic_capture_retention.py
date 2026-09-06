"""Current capture shaping, not new placement history or support constraints.

Complete synthetic sensor observations exercise the real evaluator. Snapshot
counterfactuals isolate a direct potential dependency; they are not new physical
rollouts or a causal explanation for the recorded rear-leg failures.
"""
from copy import deepcopy
import math
from pathlib import Path

import pytest
import yaml

from test_semantic_supervisor import advance, leg_state, observation, place
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.ppo.semantic_supervisor import (
    CAPTURE_RETENTION_MODE, DEFAULT_TASK_SPEC_PATH, LEG_ORDER, PHASE_IDS,
    SemanticObservationError, TaskEvaluator, TaskStageSupervisor, load_task_spec,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/ppo_semantic_v3/stage_task_spec.yaml"
FEATURE = "top_xy_outside_distance_m"
WEIGHT = .85 / 4 * .2


def _spec_path(tmp_path, *, enabled=False, **updates):
    values = load_task_spec(SPEC)
    # Explicit pre-approach configuration for this older retention-factor
    # comparison; the new approach mode requires measured retention geometry.
    values.pop("capture_approach_semantics", None)
    values.pop("capture_retention_semantics", None)
    if enabled:
        values["capture_retention_semantics"] = CAPTURE_RETENTION_MODE
    values.update(updates)
    path = tmp_path / "capture_spec.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def _initial_observation():
    obs = observation()
    for joint in obs["joints"].values():
        joint["command_deg"] = 0.
    return obs


def _placed_evaluator(path=SPEC, *, all_legs=True):
    evaluator = TaskEvaluator(path)
    obs = _initial_observation()
    evaluator.observe(obs)
    for leg in ("FR", "FL", "RR", "RL") if all_legs else ("FR", "FL", "RR"):
        obs = place(evaluator, obs, leg)
    assert evaluator.snapshot["termination_reason"] is None
    assert not evaluator.snapshot["success"]  # Base has not entered final ROI.
    return evaluator, obs


def _changed(snapshot, leg, *, outside=0., clearance=.02, **fields):
    result = deepcopy(snapshot)
    result["current_legs"][leg].update(
        {FEATURE: outside, "clearance_m": clearance, **fields})
    return result


def _fraction(outside, clearance, spec):
    # Independent statement of the reviewed existing-scale formula.
    k = spec["history"]["minimum_lift_gain_m"]
    xy = max(0., min(1., 1. - outside / .25))
    z = k / (k + max(0., spec["geometry"]["top_gap_min_m"] - clearance))
    return min(xy, z)


@pytest.mark.parametrize("leg", LEG_ORDER)
def test_every_placed_leg_uses_only_existing_capture_weight(leg):
    ev, _ = _placed_evaluator()
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    original = deepcopy(ev.snapshot)
    reference = _changed(original, leg)
    before = sup.physical_potential(reference)
    for outside, clearance in ((.05, .02), (.125, .02), (0., -.023), (.125, -.055)):
        modified = _changed(original, leg, outside=outside, clearance=clearance)
        saved = deepcopy(modified)
        fraction = _fraction(outside, clearance, sup.spec)
        assert sup.physical_potential(modified) - before == pytest.approx(WEIGHT * (fraction - 1.))
        assert sup.predicate(f"placed_{leg}", modified) == 1.
        assert modified == saved
    assert ev.snapshot == original


@pytest.mark.parametrize("clearance", [.02, .1, .5])
@pytest.mark.parametrize("load,air,top,support", [(0., True, False, False), (.9, False, True, True)])
def test_current_platform_air_has_full_credit_without_load_contact_or_height_ceiling(
        clearance, load, air, top, support):
    ev, _ = _placed_evaluator()
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    original = ev.snapshot
    # Strict direct-component comparison: changing contacts here deliberately
    # does not also change another leg's load fraction or the final predicate.
    current = _changed(original, "RR", clearance=clearance, load_fraction=load,
        air=air, top_contact=top, support=support, ground_contact=False,
        obstacle_pair_active=top, consecutive_top_samples=0 if air else 100)
    assert sup.physical_potential(current) == sup.physical_potential(original)
    assert sup.predicate("placed_RR", current) == 1.


def test_actual_evaluator_air_above_platform_keeps_capture_but_not_final_success():
    ev, obs = _placed_evaluator()
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    before = sup.physical_potential(ev.snapshot)
    obs = advance(obs)
    leg_state(obs, "RR", x=.60, bottom=.20, air=True)
    current = ev.observe(obs)
    rr = current["current_legs"]["RR"]
    assert rr["air"] and not rr["support"] and rr["load_fraction"] == 0.
    assert rr[FEATURE] == 0. and not rr["top_geometry"]
    assert sup.physical_potential(current) == before
    assert not current["final_region_valid"] and not current["success"]
    assert current["history"]["placed"]["RR"]


def test_outside_retreat_and_return_have_continuous_existing_scale_credit():
    ev, _ = _placed_evaluator()
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    snapshot = ev.snapshot
    distances = [0., 1e-9, .01, .05, .125, .249999999, .25, .4]
    phis = [sup.physical_potential(_changed(snapshot, "RR", outside=d)) for d in distances]
    assert all(a > b for a, b in zip(phis[:7], phis[1:7]))
    assert phis[-1] == phis[-2]
    for distance, phi in zip(distances, phis):
        assert phi - phis[0] == pytest.approx(-WEIGHT * min(1., distance / .25))
    assert phis[0] - phis[1] < 1e-8
    assert [sup.physical_potential(_changed(snapshot, "RR", outside=d))
            for d in reversed(distances)] == list(reversed(phis))


def test_vertical_drop_and_recovery_are_continuous_without_contact_requirement():
    ev, _ = _placed_evaluator()
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    snapshot = ev.snapshot
    boundary = sup.spec["geometry"]["top_gap_min_m"]
    gaps = [boundary + .001, boundary, boundary - 1e-9, boundary - .008,
            boundary - .040, boundary - .10]
    phis = [sup.physical_potential(_changed(snapshot, "RR", clearance=gap)) for gap in gaps]
    assert phis[0] == phis[1]
    assert all(a > b for a, b in zip(phis[1:], phis[2:]))
    assert phis[1] - phis[2] < 1e-8
    for gap, phi in zip(gaps, phis):
        assert phi - phis[0] == pytest.approx(WEIGHT * (_fraction(0., gap, sup.spec) - 1.))
    assert [sup.physical_potential(_changed(snapshot, "RR", clearance=gap))
            for gap in reversed(gaps)] == list(reversed(phis))


def test_recorded_7176_rr_only_counterexample_uses_current_region_not_history_reset():
    ev, _ = _placed_evaluator()
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    original = ev.snapshot
    outside = .07316714184451456 - sup.spec["geometry"]["xy_measurement_tolerance_m"]
    after = _changed(original, "RR", outside=outside, clearance=-.050582105998521754,
        air=False, top_contact=False, ground_contact=True, load_fraction=.4682812067889998)
    # Exact RR fields from the recorded retreat, with all other synthetic state
    # held fixed. This is not a replay of its full body/contact configuration.
    assert sup.physical_potential(after) - sup.physical_potential(original) == pytest.approx(
        -.034698633080936195)
    assert after["history"] == original["history"]
    assert sup.predicate("placed_RR", after) == 1.
    assert sup.entry_report("P10", after) == sup.entry_report("P10", original)
    assert not after["success"]


@pytest.mark.parametrize("distance,clearance", [(0., 100.), (.1, -.03), (1e9, -1e9)])
def test_bounds_and_phase_labels_do_not_change_direct_retention(distance, clearance):
    ev, _ = _placed_evaluator()
    snapshot = _changed(ev.snapshot, "RR", outside=distance, clearance=clearance)
    before = deepcopy(snapshot)
    values = [TaskStageSupervisor(SPEC, evaluator=ev, initial_stage_id=phase).physical_potential(snapshot)
              for phase in PHASE_IDS]
    assert values == [values[0]] * len(PHASE_IDS)
    assert 0. <= values[0] <= 1.
    assert snapshot == before


def test_unplaced_and_predecessor_gated_formulas_are_exactly_unchanged(tmp_path):
    old_path = _spec_path(tmp_path)
    ev, _ = _placed_evaluator(all_legs=False)
    original = ev.snapshot
    sup, old = TaskStageSupervisor(SPEC, evaluator=ev), TaskStageSupervisor(old_path, evaluator=ev)
    for outside in (0., .125, .5):
        snapshot = _changed(original, "RL", outside=outside, clearance=-.02)
        assert not snapshot["history"]["placed"]["RL"]
        assert sup.physical_potential(snapshot) == old.physical_potential(snapshot)
        assert sup.predicate("placed_RL", snapshot) == old.predicate("placed_RL", snapshot) < 1.
    # RL has no right to lift/carry/capture before RR, even if diagnostic
    # current-region values would be perfect for an already placed wheel.
    ev = TaskEvaluator(SPEC)
    obs = _initial_observation()
    ev.observe(obs)
    for leg in ("FR", "FL"):
        obs = place(ev, obs, leg)
    snap = ev.snapshot
    snap["current_legs"]["RL"].update({FEATURE: 0., "clearance_m": .2,
        "initial_clearance": True, "soft_air_actuation_earned": True,
        "consecutive_top_samples": 100, "consecutive_air_samples": 100})
    sup, old = TaskStageSupervisor(SPEC, evaluator=ev), TaskStageSupervisor(old_path, evaluator=ev)
    assert sup.physical_potential(snap) == old.physical_potential(snap)
    assert not snap["history"]["active_lift"]["RL"]
    assert not snap["history"]["front_edge_crossed"]["RL"]
    assert not snap["history"]["placed"]["RL"]


@pytest.mark.parametrize("flag", ["absent", None])
def test_optout_keeps_old_placed_formula_without_new_feature(tmp_path, flag):
    path = _spec_path(tmp_path, **({} if flag == "absent" else {"capture_retention_semantics": None}))
    ev, _ = _placed_evaluator(path)
    sup = TaskStageSupervisor(path, evaluator=ev)
    snapshot = ev.snapshot
    assert all(FEATURE not in current for current in snapshot["current_legs"].values())
    expected = .85 + .15 * sup.predicate("whole_task_success", snapshot)
    assert sup.physical_potential(snapshot) == expected
    snapshot["current_legs"]["RR"].update(clearance_m=-.5, front_distance_m=-.5,
        top_contact=False, ground_contact=True, air=False)
    assert sup.physical_potential(snapshot) == expected
    legacy = TaskEvaluator(DEFAULT_TASK_SPEC_PATH)
    assert all(FEATURE not in row for row in legacy.observe(observation())["current_legs"].values())


@pytest.mark.parametrize("bad", [-1e-9, -1., math.nan, math.inf, -math.inf, True, None])
def test_malformed_current_outside_distance_cannot_become_full_credit(bad):
    ev, _ = _placed_evaluator()
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    with pytest.raises((SemanticObservationError, ValueError)):
        sup.physical_potential(_changed(ev.snapshot, "RR", outside=bad))


def test_missing_current_outside_measurement_fails_closed_for_optin():
    ev, _ = _placed_evaluator()
    snapshot = deepcopy(ev.snapshot)
    del snapshot["current_legs"]["RR"][FEATURE]
    with pytest.raises((SemanticObservationError, ValueError, KeyError)):
        TaskStageSupervisor(SPEC, evaluator=ev).physical_potential(snapshot)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, None])
def test_malformed_current_clearance_cannot_become_full_credit(bad):
    ev, _ = _placed_evaluator()
    with pytest.raises((SemanticObservationError, ValueError)):
        TaskStageSupervisor(SPEC, evaluator=ev).physical_potential(_changed(ev.snapshot, "RR", clearance=bad))


@pytest.mark.parametrize("x,y,expected", [
    (.60, 0., 0.), (.495, 0., 0.), (2.005, 0., 0.),
    (.60, -.805, 0.), (.60, .805, 0.), (.495, .805, 0.),
    (.494, 0., .001), (2.006, 0., .001),
    (.60, -.806, .001), (.60, .806, .001),
    (.492, -.809, .005), (2.008, .809, .005),
])
def test_real_evaluator_measures_tolerance_expanded_rectangle_euclidean_distance(x, y, expected):
    ev, obs = _placed_evaluator()
    obs = advance(obs)
    leg_state(obs, "RR", x=x, bottom=.08, air=True)
    wheel = obs["wheels"][WHEEL_ORDER[LEG_ORDER.index("RR")]]
    wheel["center_w_m"][1] = y
    # Geometry contract uses the measured wheel center for XY, not its AABB
    # bottom sample or any historical entry location.
    wheel["bottom_w_m"][0:2] = [-123., 456.]
    current = ev.observe(obs)["current_legs"]["RR"]
    assert current[FEATURE] == pytest.approx(expected, abs=1e-14)
    assert current["front_distance_m"] == pytest.approx(x - .5)
    assert current["within_top_xy"] is (expected == 0.)


@pytest.mark.parametrize("mode", ["", "unknown", True, 1, [], {}])
def test_loader_rejects_unrecognized_capture_semantics(tmp_path, mode):
    path = _spec_path(tmp_path, capture_retention_semantics=mode)
    with pytest.raises(ValueError):
        load_task_spec(path)


def test_loader_requires_existing_global_potential_and_positive_lift_scale(tmp_path):
    path = _spec_path(tmp_path, enabled=True, potential_definition=None)
    with pytest.raises(ValueError):
        load_task_spec(path)
    values = load_task_spec(SPEC)
    values["history"]["minimum_lift_gain_m"] = 0.
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_task_spec(path)


def test_shared_hard_evaluator_events_failure_and_success_are_unchanged(tmp_path):
    from wlr50_clean.ppo.semantic_legacy_evaluation import PhysicalEvaluationRecorder

    old_path = _spec_path(tmp_path)
    recorder_dir = tmp_path / "evaluation"
    recorder_dir.mkdir()
    recorder = PhysicalEvaluationRecorder(recorder_dir, task_spec_path=SPEC)
    evaluators = [TaskEvaluator(old_path), TaskEvaluator(SPEC),
        TaskStageSupervisor(SPEC).evaluator, recorder.evaluator]
    try:
        obs = _initial_observation()
        for ev in evaluators:
            ev.observe(obs)
        # A zero-clearance wheel-driven crossing remains a hard failure in
        # every A/B/C consumer. Region shaping must never invent placed bits.
        obs = advance(obs)
        leg_state(obs, "FR", x=.51, bottom=.004, air=True)
        results = [ev.observe(deepcopy(obs)) for ev in evaluators]
        for result in results:
            for row in result["current_legs"].values():
                row.pop(FEATURE, None)
        assert all(result == results[0] for result in results[1:])
        assert results[0]["termination_reason"] == "TASK_FAILURE_WHEEL_ONLY_CLIMB"
        assert not results[0]["history"]["front_edge_crossed"]["FR"]
        assert not results[0]["history"]["placed"]["FR"]
        assert not results[0]["success"]
    finally:
        recorder.close()


def test_existing_invalid_snapshot_still_returns_zero():
    ev, _ = _placed_evaluator()
    snapshot = deepcopy(ev.snapshot)
    snapshot["valid"] = False
    for row in snapshot["current_legs"].values():
        row.pop(FEATURE, None)
    assert TaskStageSupervisor(SPEC, evaluator=ev).physical_potential(snapshot) == 0.


def test_real_qualification_placement_retreat_recovery_and_controlled_success_are_unchanged(tmp_path):
    old_path = _spec_path(tmp_path)
    old, new = TaskEvaluator(old_path), TaskEvaluator(SPEC)

    def observe_both(obs):
        expected = old.observe(deepcopy(obs))
        actual = new.observe(deepcopy(obs))
        comparable = deepcopy(actual)
        for row in comparable["current_legs"].values():
            row.pop(FEATURE)
        assert comparable == expected
        return actual

    obs = _initial_observation()
    observe_both(obs)
    for leg in ("FR", "FL", "RR", "RL"):
        for fields in ({"bottom": .012, "air": True, "hip": 1.5},
                       {"bottom": .020, "air": True, "hip": 3.},
                       {"x": .53, "bottom": .075, "air": True, "hip": 4.},
                       {"x": .53, "bottom": .05, "top": True},
                       {"x": .53, "bottom": .05, "top": True}):
            obs = advance(obs)
            leg_state(obs, leg, **fields)
            snapshot = observe_both(obs)
        assert all(snapshot["history"][key][leg] for key in ("active_lift", "front_edge_crossed", "placed"))
    history = deepcopy(snapshot["history"])
    obs = advance(obs)
    leg_state(obs, "RR", x=.42683285815548544, bottom=-.000582105998521754)
    retreated = observe_both(obs)
    assert retreated["history"] == history and not retreated["success"]
    assert retreated["current_legs"]["RR"]["ground_contact"]
    assert not retreated["final_region_valid"]
    # Recovery may use any safe pose: this fixture retains the measured joints
    # rather than resetting them to the home suggestion or replaying a lift.
    obs = advance(obs)
    leg_state(obs, "RR", x=.53, bottom=.05, top=True)
    obs["base"]["position_w_m"][0] = .7
    snapshot = observe_both(obs)
    assert not snapshot["success"]
    for index in range(60):
        obs = advance(obs)
        snapshot = observe_both(obs)
        assert snapshot["success"] is (index == 59)
    assert snapshot["final_stable_for_s"] == pytest.approx(.5)
    assert snapshot["history"] == history
    assert TaskStageSupervisor(SPEC, evaluator=new).physical_potential(snapshot) == 1.
    assert TaskStageSupervisor(old_path, evaluator=old).physical_potential(snapshot) == 1.

"""Current workspace shaping without granting predecessor-gated leg events."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from test_semantic_supervisor import advance, leg_state, observation, place
from wlr50_clean.ppo.semantic_supervisor import (
    DEFAULT_TASK_SPEC_PATH, PHASE_IDS,
    PREPARATION_CREDIT_MODE, TaskEvaluator, TaskStageSupervisor, load_task_spec,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/ppo_semantic_v3/stage_task_spec.yaml"


@pytest.fixture(autouse=True)
def legacy_workspace_formula(tmp_path, monkeypatch):
    """Keep these historical preparation-weight tests on their original formula.

    The independent workspace-potential integration tests cover the production
    v3 opt-in. No hard predicate or event assertion below is changed.
    """
    values = load_task_spec(SPEC)
    values["workspace_potential_semantics"] = None
    path = tmp_path / "legacy_workspace_preparation.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    monkeypatch.setitem(globals(), "SPEC", path)


def old_spec_path(tmp_path):
    values = load_task_spec(SPEC)
    values.pop("preparation_credit_semantics")
    path = tmp_path / "without_preparation.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return path


def front_pair_prepared(path=None):
    path = SPEC if path is None else path
    evaluator = TaskEvaluator(path)
    obs = observation()
    for joint in obs["joints"].values():
        joint["command_deg"] = 0.
    evaluator.observe(obs)
    for leg in ("FR", "FL"):
        obs = place(evaluator, obs, leg)
    obs = advance(obs)
    front = obs["obstacle"]["front_x_m"]
    # C46464's P06 workspace mismatch; synthetic complete sensor observations,
    # not a claim to replay its actual full physical trajectory.
    leg_state(obs, "RR", x=front - .2168)
    leg_state(obs, "RL", x=front - .2368)
    evaluator.observe(obs)
    return evaluator, obs


def test_real_rl_workspace_progress_and_retreat_before_rr_placement(tmp_path):
    ev, obs = front_pair_prepared()
    sup = TaskStageSupervisor(SPEC, evaluator=ev, initial_stage_id="P06")
    old = TaskStageSupervisor(old_spec_path(tmp_path), evaluator=ev)
    first = ev.snapshot
    before_history = deepcopy(first["history"])
    assert sup.predicate("workspace_RR", first) == 1.
    assert sup.predicate("workspace_RL", first) == pytest.approx(.9328)
    assert sup.predicate("rear_approach", first) < 1.
    assert not first["history"]["placed"]["RR"]
    phis, old_phis = [], []
    for distance in (-.2368, -.22, -.2368):
        obs = advance(obs)
        leg_state(obs, "RL", x=obs["obstacle"]["front_x_m"] + distance)
        snap = ev.observe(obs)
        phis.append(sup.physical_potential(snap))
        old_phis.append(old.physical_potential(snap))
        assert snap["history"] == before_history
        assert not snap["success"] and snap["termination_reason"] is None
    assert phis[1] - phis[0] == pytest.approx(.85 / 4 * .1 * (1 - .9328))
    assert phis[2] - phis[1] == pytest.approx(-.001428)
    assert phis[0] == phis[2]
    assert old_phis == [old_phis[0]] * 3


def test_only_workspace_component_is_added_for_unplaced_future_leg(tmp_path):
    ev, _ = front_pair_prepared()
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    old = TaskStageSupervisor(old_spec_path(tmp_path), evaluator=ev)
    original = ev.snapshot
    for rich_future_state in (False, True):
        snap = deepcopy(original)
        if rich_future_state:
            # Deliberately maximal future-leg diagnostic fields: the blocked
            # predecessor branch must still omit every non-workspace term.
            row = snap["current_legs"]["RL"]
            row.update(initial_clearance=True, soft_air_actuation_earned=True,
                air=True, ground_contact=False, obstacle_pair_active=False,
                clearance_m=.1, consecutive_air_samples=100,
                consecutive_top_samples=100, load_fraction=0.)
            snap["history"]["active_lift"]["RL"] = True
            snap["history"]["front_edge_crossed"]["RL"] = True
        untouched = deepcopy(snap)
        assert sup.physical_potential(snap) - old.physical_potential(snap) == pytest.approx(
            .85 / 4 * .1 * sup.predicate("workspace_RL", snap))
        assert snap == untouched
    assert ev.snapshot == original


def test_below_top_air_preparation_cannot_award_rl_lift_carry_or_placement(tmp_path):
    ev, obs = front_pair_prepared()
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    old = TaskStageSupervisor(old_spec_path(tmp_path), evaluator=ev)
    for height, position in ((.003, 1.5), (.009, 3.), (.012, 4.)):
        obs = advance(obs)
        leg_state(obs, "RL", bottom=height, air=True)
        obs["joints"]["front_left_hip"].update(position_deg=position, command_deg=position + 1.)
        snap = ev.observe(obs)
        assert snap["current_legs"]["RL"]["air"]
        assert snap["current_legs"]["RL"]["clearance_m"] < 0.
        assert not snap["current_legs"]["RL"]["soft_air_actuation_earned"]
        for key in ("active_lift", "front_edge_crossed", "placed"):
            assert not snap["history"][key]["RL"]
        assert not snap["success"] and snap["termination_reason"] is None
        assert sup.physical_potential(snap) - old.physical_potential(snap) == pytest.approx(
            .85 / 4 * .1 * sup.predicate("workspace_RL", snap))


@pytest.mark.parametrize("distance,lateral", [(-2., True), (-.47, True), (-.2368, True),
    (-.22, True), (.06, True), (.31, True), (2., True), (-.22, False)])
def test_current_workspace_bounds_lateral_exclusion_and_no_new_weights(tmp_path, distance, lateral):
    ev, _ = front_pair_prepared()
    snap = ev.snapshot
    snap["current_legs"]["RL"].update(front_distance_m=distance, within_lateral_span=lateral)
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    old = TaskStageSupervisor(old_spec_path(tmp_path), evaluator=ev)
    phi = sup.physical_potential(snap)
    addition = phi - old.physical_potential(snap)
    assert 0. <= phi <= 1.
    assert addition == pytest.approx(.85 / 4 * .1 * sup.predicate("workspace_RL", snap))
    assert 0. <= addition <= .85 / 4 * .1 + 1e-15
    if not lateral or distance in (-2., -.47, .31, 2.):
        assert addition == pytest.approx(0.)


def test_same_physical_state_phase_labels_and_history_are_unchanged():
    ev, _ = front_pair_prepared()
    snapshot = ev.snapshot
    before = deepcopy(snapshot)
    values = [TaskStageSupervisor(SPEC, evaluator=ev, initial_stage_id=phase).physical_potential(snapshot)
              for phase in PHASE_IDS]
    assert values == [values[0]] * 13
    assert snapshot == before and ev.snapshot == before


def test_placed_history_and_final_potential_keep_exact_old_formula(tmp_path):
    ev, obs = front_pair_prepared()
    for leg in ("RR", "RL"):
        obs = place(ev, obs, leg)
    snapshot = ev.snapshot
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    old = TaskStageSupervisor(old_spec_path(tmp_path), evaluator=ev)
    assert all(snapshot["history"]["placed"].values())
    assert sup.physical_potential(snapshot) == old.physical_potential(snapshot)
    assert sup.physical_potential(snapshot) == pytest.approx(
        .85 + .15 * sup.predicate("whole_task_success", snapshot))


@pytest.mark.parametrize("failure", ["invalid", "body_collision"])
def test_existing_invalid_and_terminal_reward_semantics_preserved(tmp_path, failure):
    from test_semantic_observation_reward_env import _built, _frame, _sample
    from wlr50_clean.ppo.semantic_reward import SemanticRewardCalculator, load_semantic_reward_config

    ev, obs = front_pair_prepared()
    obs = advance(obs)
    if failure == "invalid":
        obs["all_finite"] = False
    else:
        obs["body_collision"]["detected"] = True
    snap = ev.observe(obs)
    assert not snap["success"] and snap["termination_reason"] is not None
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    old = TaskStageSupervisor(old_spec_path(tmp_path), evaluator=ev)
    if failure == "invalid":
        assert sup.physical_potential(snap) == old.physical_potential(snap) == 0.
    # The potential function historically accepts a valid terminal snapshot;
    # its existing reward consumer, not a new gate, zeros terminal potential.
    a, b = _built(_frame(phi=.4)), _built(_frame(1, phi=sup.physical_potential(snap)))
    config = load_semantic_reward_config(ROOT / "configs/ppo_semantic_v3/reward_config.yaml")
    result = SemanticRewardCalculator(config).evaluate(a, b, [_sample(a, b)],
        termination_reason=snap["termination_reason"], task_success=False)
    assert result["potential_after"] == 0.
    assert not result["terminal_bootstrap_allowed"]


@pytest.mark.parametrize("mode", ["unknown", "", True, 1, [], {}])
def test_loader_rejects_invalid_preparation_mode(tmp_path, mode):
    values = load_task_spec(SPEC)
    values["preparation_credit_semantics"] = mode
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="preparation credit"):
        load_task_spec(path)


def test_loader_requires_existing_v3_global_potential_for_optin(tmp_path):
    values = load_task_spec(DEFAULT_TASK_SPEC_PATH)
    assert "preparation_credit_semantics" not in values
    values["preparation_credit_semantics"] = PREPARATION_CREDIT_MODE
    path = tmp_path / "v2_optin.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="global physical progress"):
        load_task_spec(path)


def test_missing_or_null_optin_keeps_v2_and_previous_v3_potential(tmp_path):
    ev, _ = front_pair_prepared()
    snap = ev.snapshot
    old_path = old_spec_path(tmp_path)
    old = TaskStageSupervisor(old_path, evaluator=ev)
    legacy = TaskStageSupervisor(DEFAULT_TASK_SPEC_PATH, evaluator=ev)
    # No unplaced leg has AIR credit in this measured ground fixture, so the
    # legacy and prior-v3 formulas agree exactly and omit RL workspace.
    expected = .85 / 4 * (2. + .1 + .1 * old.predicate("load_ready_RR", snap))
    assert old.physical_potential(snap) == legacy.physical_potential(snap) == expected
    old_values = load_task_spec(old_path)
    old_values["preparation_credit_semantics"] = None
    old_path.write_text(yaml.safe_dump(old_values, sort_keys=False), encoding="utf-8")
    assert TaskStageSupervisor(old_path, evaluator=ev).physical_potential(snap) == expected


def test_shared_a_b_c_hard_evaluator_is_unchanged_by_shaping_optin(tmp_path):
    from wlr50_clean.ppo.semantic_legacy_evaluation import PhysicalEvaluationRecorder

    old_path = old_spec_path(tmp_path)
    recorder_path = tmp_path / "recorder"
    recorder_path.mkdir()
    recorder = PhysicalEvaluationRecorder(recorder_path, task_spec_path=SPEC)
    evaluators = [TaskEvaluator(old_path), TaskEvaluator(SPEC),
                  TaskStageSupervisor(SPEC).evaluator, recorder.evaluator]
    try:
        obs = observation()
        for joint in obs["joints"].values():
            joint["command_deg"] = 0.
        for evaluator in evaluators:
            evaluator.observe(obs)
        # Unqualified low RL crossing before RR placement remains rejected by
        # every consumer, even if current workspace preparation earned credit.
        for distance, height in ((-.2368, 0.), (-.22, .003), (.001, .004)):
            obs = advance(obs)
            leg_state(obs, "RL", x=.5 + distance, bottom=height, air=height > 0.)
            results = [ev.observe(deepcopy(obs)) for ev in evaluators]
            assert all(result == results[0] for result in results[1:])
        assert results[0]["termination_reason"] == "TASK_FAILURE_WHEEL_ONLY_CLIMB"
        assert not results[0]["history"]["front_edge_crossed"]["RL"]
        assert not results[0]["history"]["placed"]["RL"]
        assert not results[0]["success"]
    finally:
        recorder.close()

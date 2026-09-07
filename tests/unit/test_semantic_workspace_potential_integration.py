"""Soft-only workspace revision through real evaluator/supervisor/reward APIs.

Complete synthetic sensor observations are CPU contract tests, not Isaac runs.
The prior mode remains an exact hard-predicate and nominal-behaviour control.
"""
import ast
from copy import deepcopy
import inspect
import math
from pathlib import Path

import pytest
import yaml

from test_semantic_supervisor import advance, leg_state, observation, place
from test_semantic_observation_reward_env import _built, _frame, _raw, _sample
from wlr50_clean.ppo.semantic_observation import (
    HISTORY_GROUPS, SemanticObservationBuilder, load_semantic_observation_schema,
)
from wlr50_clean.ppo.semantic_reward import (
    FAMILIES, SemanticRewardCalculator, load_semantic_reward_config,
)
from wlr50_clean.ppo.semantic_supervisor import (
    DEFAULT_TASK_SPEC_PATH, LEG_ORDER, PHASE_IDS, PLACEMENT_PREDECESSORS,
    WORKSPACE_POTENTIAL_MODE, NominalMotionProvider, TaskEvaluator,
    TaskStageSupervisor, load_task_spec,
)
from wlr50_clean.ppo.semantic_workspace_potential import interval_distance_progress
from wlr50_clean.reference.motion_contract import load_motion_contract


ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs/ppo_semantic_v3"
SPEC = CFG / "stage_task_spec.yaml"
WEIGHT = .85 / 4 * .1


@pytest.fixture
def modes(tmp_path):
    # New side is the actual production default, not only a fabricated config.
    values = load_task_spec(SPEC)
    assert values["workspace_potential_semantics"] == WORKSPACE_POTENTIAL_MODE
    old = deepcopy(values)
    old.pop("workspace_potential_semantics")
    path = tmp_path / "before_workspace_potential.yaml"
    path.write_text(yaml.safe_dump(old, sort_keys=False), encoding="utf-8")
    return SPEC, path


def measured(path, *, placed=()):
    ev = TaskEvaluator(path)
    obs = observation()
    for joint in obs["joints"].values():
        joint["command_deg"] = 0.
    ev.observe(obs)
    for leg in placed:
        obs = place(ev, obs, leg)
    return ev, obs


def set_distance(obs, leg, value):
    leg_state(obs, leg, x=obs["obstacle"]["front_x_m"] + value)


def soft(sup, leg, snapshot):
    row = snapshot["current_legs"][leg]
    geo = sup.spec["geometry"]
    return interval_distance_progress(row["front_distance_m"],
        geo["workspace_min_m"], geo["workspace_max_m"], row["within_lateral_span"])


def supervisors(paths, ev):
    return tuple(TaskStageSupervisor(path, evaluator=ev) for path in paths)


@pytest.mark.parametrize("leg", LEG_ORDER)
def test_actual_far_workspace_progress_and_retreat_have_only_existing_weight(modes, leg):
    ev, obs = measured(modes[0])
    new, old = supervisors(modes, ev)
    history = deepcopy(ev.snapshot["history"])
    phis, old_phis, work = [], [], []
    for distance in (-1.2, -1., -.8, -1.2):
        obs = advance(obs)
        set_distance(obs, leg, distance)
        snap = ev.observe(obs)
        assert snap["valid"] and snap["termination_reason"] is None
        assert snap["history"] == history
        assert new.predicate(f"workspace_{leg}", snap) == old.predicate(f"workspace_{leg}", snap) == 0.
        phis.append(new.physical_potential(snap))
        old_phis.append(old.physical_potential(snap))
        work.append(soft(new, leg, snap))
    assert old_phis == [old_phis[0]] * 4
    assert phis[0] < phis[1] < phis[2]
    assert phis[3] == phis[0]
    for i in range(1, 4):
        assert phis[i] - phis[i-1] == pytest.approx(WEIGHT * (work[i] - work[i-1]))
    assert all(0. <= value <= 1. for value in phis)


@pytest.mark.parametrize("leg", ("FL", "RR", "RL"))
def test_predecessor_blocked_leg_gets_workspace_only_without_fabricated_events(modes, leg):
    ev, obs = measured(modes[0])
    obs = advance(obs)
    set_distance(obs, leg, -.8)
    snap = ev.observe(obs)
    new, old = supervisors(modes, ev)
    assert not all(snap["history"]["placed"][p] for p in PLACEMENT_PREDECESSORS[leg])
    baseline = deepcopy(snap)
    base_phi = new.physical_potential(snap)
    assert base_phi - old.physical_potential(snap) == pytest.approx(WEIGHT * soft(new, leg, snap))
    # Adversarial diagnostic richness cannot open the ordered non-workspace
    # branch. These counterfactual fields are not injected into the evaluator.
    rich = deepcopy(snap)
    rich["current_legs"][leg].update(initial_clearance=True,
        soft_air_actuation_earned=True, air=True, ground_contact=False,
        obstacle_pair_active=False, clearance_m=.1, consecutive_air_samples=100,
        consecutive_top_samples=100, load_fraction=0.)
    rich["history"]["active_lift"][leg] = True
    rich["history"]["front_edge_crossed"][leg] = True
    assert new.physical_potential(rich) == base_phi
    assert ev.snapshot == baseline
    assert not any(baseline["history"][key][leg]
        for key in ("active_lift", "front_edge_crossed", "placed"))


@pytest.mark.parametrize("leg", LEG_ORDER)
def test_predecessor_eligible_ordinary_workspace_branch_uses_same_soft_formula(modes, leg):
    ev, obs = measured(modes[0], placed=PLACEMENT_PREDECESSORS[leg])
    obs = advance(obs)
    set_distance(obs, leg, -.8)
    snap = ev.observe(obs)
    new, old = supervisors(modes, ev)
    assert all(snap["history"]["placed"][p] for p in PLACEMENT_PREDECESSORS[leg])
    assert not snap["history"]["placed"][leg]
    assert new.physical_potential(snap) - old.physical_potential(snap) == pytest.approx(
        WEIGHT * (soft(new, leg, snap) - old.predicate(f"workspace_{leg}", snap)))


@pytest.mark.parametrize("edge", ("workspace_min_m", "workspace_max_m"))
@pytest.mark.parametrize("direction", (-math.inf, math.inf))
def test_nextafter_hard_workspace_entry_and_completion_are_exact_old(modes, edge, direction):
    ev, _ = measured(modes[0], placed=("FR", "FL"))
    new, old = supervisors(modes, ev)
    value = math.nextafter(new.spec["geometry"][edge], direction)
    snap = deepcopy(ev.snapshot)
    snap["current_legs"]["RL"]["front_distance_m"] = value
    # nextafter is applied to the existing measured scalar, avoiding an extra
    # world-coordinate subtraction that could round it back to the boundary.
    for phase in PHASE_IDS:
        assert new.entry_report(phase, snap) == old.entry_report(phase, snap)
        for name in new.spec["stages"][phase]["completion_predicates"]:
            assert new.predicate(name, snap) == old.predicate(name, snap)
    assert new.predicate("rear_approach", snap) == old.predicate("rear_approach", snap)
    # We preserve even the legacy predicate's own floating-point rounding;
    # soft==1 is never substituted as a new completion test.


@pytest.mark.parametrize("distance,lateral", [(-2., True), (.8, True), (-.22, True),
    (.06, True), (-.8, False)])
def test_interval_sides_inside_and_lateral_exclusion(modes, distance, lateral):
    ev, _ = measured(modes[0])
    new, old = supervisors(modes, ev)
    snap = deepcopy(ev.snapshot)
    snap["current_legs"]["RL"].update(front_distance_m=distance, within_lateral_span=lateral)
    expected = WEIGHT * (soft(new, "RL", snap) - old.predicate("workspace_RL", snap))
    assert new.physical_potential(snap) - old.physical_potential(snap) == pytest.approx(expected)
    assert 0. <= new.physical_potential(snap) <= 1.
    if not lateral:
        assert soft(new, "RL", snap) == 0.


@pytest.mark.parametrize("leg", LEG_ORDER)
def test_placed_retention_branch_is_exact_old_including_legitimate_air(modes, leg):
    ev, obs = measured(modes[0], placed=("FR", "FL", "RR", "RL"))
    new, old = supervisors(modes, ev)
    for distance, height in ((.04, .1), (-.5, .1), (-.5, .01)):
        obs = advance(obs)
        leg_state(obs, leg, x=.5 + distance, bottom=height, air=True)
        snap = ev.observe(obs)
        assert snap["history"]["placed"][leg]
        assert snap["current_legs"][leg]["air"]
        assert not snap["current_legs"][leg]["support"]
        assert new.physical_potential(snap) == old.physical_potential(snap)


def test_live_continuous_observations_keep_evaluator_handoffs_history_and_nominal_exact(modes):
    pairs = [measured(path, placed=("FR", "FL")) for path in modes]
    sups = [TaskStageSupervisor(path, evaluator=pair[0], initial_stage_id="P06")
            for path, pair in zip(modes, pairs)]
    contract = load_motion_contract(ROOT / "configs/recording_motion_contract.json")
    providers = [NominalMotionProvider(contract, spec=sup.spec) for sup in sups]
    obs = pairs[0][1]
    seen, different_phi = set(), False
    for i in range(40):
        obs = advance(obs)
        for leg in ("RR", "RL"):
            set_distance(obs, leg, -.8 if i < 6 else -.2)
        if i >= 22:
            leg_state(obs, "RR", bottom=.001, air=True)
        tasks = [sup.observe_and_update(deepcopy(obs)) for sup in sups]
        assert sups[0].evaluator.snapshot == sups[1].evaluator.snapshot
        assert {k: v for k, v in tasks[0].items() if k != "task_progress_potential"} == {
            k: v for k, v in tasks[1].items() if k != "task_progress_potential"}
        different_phi |= tasks[0]["task_progress_potential"] != tasks[1]["task_progress_potential"]
        targets = [provider.evaluate(task, obs) for provider, task in zip(providers, tasks)]
        assert targets[0] == targets[1]
        assert providers[0].tracking_servo_names == providers[1].tracking_servo_names
        seen.add(tasks[0]["stage_id"])
    assert {"P06", "P07", "P08", "P09"} <= seen
    assert different_phi
    assert sups[0].completed_stage_ids == sups[1].completed_stage_ids == ["P06", "P07", "P08"]
    assert not sups[0].snapshot["success"]


def test_global_phi_same_at_every_ordinary_phase_and_no_repeat_event_bonus(modes):
    ev, obs = measured(modes[0])
    obs = advance(obs)
    set_distance(obs, "RL", -.8)
    snap = ev.observe(obs)
    before = deepcopy(snap)
    values = [TaskStageSupervisor(modes[0], evaluator=ev, initial_stage_id=phase).physical_potential(snap)
              for phase in PHASE_IDS]
    assert values == [values[0]] * 13
    assert ev.snapshot == before
    a, b = _built(_frame(phi=values[0])), _built(_frame(1, phi=values[0], stage="P02"))
    cfg = load_semantic_reward_config(CFG / "reward_config.yaml")
    result = SemanticRewardCalculator(cfg).evaluate(a, b, [_sample(a, b)],
        termination_reason=None, task_success=False)
    assert result["potential_shaping"] == pytest.approx(5 * (cfg.gamma - 1.) * values[0])
    assert result["terminal_event"] == 0.


def test_true_reward_receives_phi_once_and_other_four_families_are_unchanged(modes):
    ev, obs = measured(modes[0])
    obs = advance(obs)
    set_distance(obs, "RL", -.8)
    snap = ev.observe(obs)
    new, old = supervisors(modes, ev)
    before = _built(_frame(phi=.1))
    cfg = load_semantic_reward_config(CFG / "reward_config.yaml")
    results = []
    for sup in (new, old):
        after = _built(_frame(1, phi=sup.physical_potential(snap)))
        results.append(SemanticRewardCalculator(cfg).evaluate(before, after, [_sample(before, after)],
            termination_reason=None, task_success=False))
    delta = 5 * cfg.gamma * (new.physical_potential(snap) - old.physical_potential(snap))
    assert results[0]["potential_shaping"] - results[1]["potential_shaping"] == pytest.approx(delta)
    assert tuple(results[0]["families"]) == FAMILIES
    for family in FAMILIES[1:]:
        assert results[0]["families"][family] == results[1]["families"][family]
    assert results[0]["total"] - results[1]["total"] == pytest.approx(delta)
    assert results[0]["cost_components"] == results[1]["cost_components"]


@pytest.mark.parametrize("failure", ("invalid", "body_collision"))
def test_actual_invalid_or_terminal_evaluator_keeps_zero_terminal_phi_no_bootstrap(modes, failure):
    measured_pairs = [measured(path) for path in modes]
    evaluators = [pair[0] for pair in measured_pairs]
    obs = advance(measured_pairs[0][1])
    if failure == "invalid":
        obs["all_finite"] = False
    else:
        obs["body_collision"]["detected"] = True
    snaps = [ev.observe(deepcopy(obs)) for ev in evaluators]
    assert snaps[0] == snaps[1]
    assert snaps[0]["termination_reason"] is not None
    cfg = load_semantic_reward_config(CFG / "reward_config.yaml")
    results = []
    for path, ev, snap in zip(modes, evaluators, snaps):
        phi = TaskStageSupervisor(path, evaluator=ev).physical_potential(snap)
        if failure == "invalid":
            assert phi == 0.
        a, b = _built(_frame(phi=.3)), _built(_frame(1, phi=phi))
        results.append(SemanticRewardCalculator(cfg).evaluate(a, b, [_sample(a, b)],
            termination_reason=snap["termination_reason"], task_success=False))
    assert results[0] == results[1]
    assert results[0]["potential_after"] == 0.
    assert results[0]["potential_shaping"] == pytest.approx(-5 * .3)
    assert results[0]["terminal_event"] == -40.
    assert not results[0]["terminal_bootstrap_allowed"]


@pytest.mark.parametrize("mode", ("unknown", "", True, 1, [], {}))
def test_loader_rejects_unknown_workspace_mode(tmp_path, mode):
    values = load_task_spec(SPEC)
    values["workspace_potential_semantics"] = mode
    path = tmp_path / "bad_workspace_mode.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="workspace"):
        load_task_spec(path)


def test_mode_requires_v3_global_phi_but_missing_null_keep_old_exact(modes, tmp_path):
    ev, obs = measured(modes[0])
    obs = advance(obs)
    set_distance(obs, "RL", -.8)
    snap = ev.observe(obs)
    legacy = load_task_spec(modes[1])
    legacy["workspace_potential_semantics"] = None
    path = tmp_path / "null.yaml"
    path.write_text(yaml.safe_dump(legacy, sort_keys=False), encoding="utf-8")
    assert TaskStageSupervisor(path).physical_potential(snap) == TaskStageSupervisor(modes[1]).physical_potential(snap)
    values = load_task_spec(DEFAULT_TASK_SPEC_PATH)
    assert "workspace_potential_semantics" not in values
    values["workspace_potential_semantics"] = WORKSPACE_POTENTIAL_MODE
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="workspace|global physical"):
        load_task_spec(path)


def test_real_observation_builder_keeps_324_schema_and_only_phi_scalar_changes(modes):
    raw = _raw(joint=0.)
    tasks = [TaskStageSupervisor(path).observe_and_update(raw) for path in modes]
    schema = load_semantic_observation_schema(CFG / "observation_schema.json")
    old_schema = load_semantic_observation_schema(ROOT / "configs/ppo_semantic_v2/observation_schema.json")
    assert schema.groups == old_schema.groups
    assert schema.dimension == old_schema.dimension == 324
    assert schema.clip == old_schema.clip
    history = dict.fromkeys(HISTORY_GROUPS, (0.,) * 12)
    built = []
    for task in tasks:
        frame = _frame(raw=raw)
        frame.info["semantic_task"] = task
        frame.state_id = task["stage_id"]
        built.append(SemanticObservationBuilder(schema).build(frame, history))
    assert tasks[0]["task_progress_potential"] != tasks[1]["task_progress_potential"]
    assert len(built[0].groups["goal_features"]) == 17
    for name in built[0].groups:
        if name != "task_progress":
            assert built[0].groups[name] == built[1].groups[name]
    assert built[0].groups["task_progress"][0] == built[1].groups["task_progress"][0]
    encoded = [schema.encode(frame.groups) for frame in built]
    assert sum(a != b for a, b in zip(*encoded)) == 1


def test_new_helper_is_used_only_twice_inside_physical_potential():
    tree = ast.parse(inspect.getsource(TaskStageSupervisor))
    calls = []
    for method in tree.body[0].body:
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(method):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    if node.func.attr == "_workspace_potential_progress":
                        calls.append(method.name)
    assert calls == ["physical_potential", "physical_potential"]

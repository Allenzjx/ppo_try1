"""Isolated CPU tests for candidate soft quality; production import kept separate."""
import ast
from copy import deepcopy
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/unit"))
from candidate_bootstrap import CANDIDATE, OLD as OLD_MODULES, OLD_FIXTURES
old_reward, old_supervisor, old_observation = (
    OLD_MODULES[name] for name in ("semantic_reward", "semantic_supervisor", "semantic_observation"))
_built, _frame, _raw, _sample, ZERO12 = (
    getattr(OLD_FIXTURES, name) for name in ("_built", "_frame", "_raw", "_sample", "ZERO12"))
quality, reward, supervisor, observation = (CANDIDATE[name] for name in (
    "semantic_task_quality", "semantic_reward", "semantic_supervisor", "semantic_observation"))
assert old_reward is not reward and old_supervisor is not supervisor and old_observation is not observation
NEW = HERE / "configs/ppo_task_conditioned_hip_wheel_v1"
OLD = ROOT / "configs/ppo_fl_capture_quality_v1"
PLANES = (.5, 2., .8, -.8, 0., .05)


def state(phase="P02", *, tick=1, distance=.03, missing=False, reason=None):
    value = _built(_frame(tick, stage=phase, phi=.45, raw=_raw(tick, roll=.2, pitch=.1)))
    ev = dict(valid=True, simulation_time_s=tick/120, history={}, current_legs={},
              body_traversal_geometry=dict(valid=True, minimum_w_m=(.7, -.2, .05+distance), maximum_w_m=(1., .2, .3)))
    if missing: ev.pop("body_traversal_geometry")
    return replace(value, task={**value.task, "termination_reason": reason,
        "transfer_roles_version": "diagonal_transfer_roles_v1", "physical_transfer_fraction": .6,
        "physical_evaluator": ev}, metrics={**value.metrics, "euler_roll_pitch_rate": (.2, .3),
                                            "obstacle_planes_world_m": PLANES})


def calculate(after, *, before=None, cfg=None, module=reward, reason=None):
    before = state(after.task["stage_id"], tick=0) if before is None else before
    cfg = module.load_semantic_reward_config(NEW / "reward_config.yaml") if cfg is None else cfg
    return module.SemanticRewardCalculator(cfg).evaluate(before, after, [_sample(before, after)],
        termination_reason=reason, task_success=False)


def retention_fixture():
    spec = supervisor.load_task_spec(NEW / "stage_task_spec.yaml")
    sup = supervisor.TaskStageSupervisor.__new__(supervisor.TaskStageSupervisor)
    sup.spec = spec
    row = dict(top_xy_outside_distance_m=0., clearance_m=.02, front_distance_m=.1,
               top_contact=False, top_surface_contact=False, support=False, bearing_verified=True,
               air=True, ground_contact=False)
    ev = dict(valid=True, history=dict(placed=dict(FR=True, FL=True, RR=False, RL=False)),
              current_legs={leg: deepcopy(row) for leg in ("FR", "FL", "RL", "RR")})
    for leg in ("RL", "RR"): ev["current_legs"][leg]["front_distance_m"] = -.3
    return sup, ev


def test_new_fixed_config_preserves_failure_avoidance_guard():
    cfg = reward.load_semantic_reward_config(NEW / "reward_config.yaml")
    assert cfg.failure_avoidance_bound == pytest.approx(8.555555555555683)
    assert cfg.failure_avoidance_bound < cfg.values["failure_cost"] == 40.
    assert cfg.values["family_weights"]["body_stability"] == .06


@pytest.mark.parametrize("mutation", [lambda v: v.update(quality_epsilon=.03),
    lambda v: v["task_space_quality"].update(clearance_margin_m=.03),
    lambda v: v["task_space_quality"].update(front_fraction=.1),
    lambda v: v["task_space_quality"].update(sample_audit=False),
    lambda v: v["family_weights"].update(control_smoothness=.01),
    lambda v: v.update(objective_profile="fl_capture_front_body_quality_v1")])
def test_unreviewed_config_rejected(tmp_path, mutation):
    values = yaml.safe_load((NEW / "reward_config.yaml").read_text())
    mutation(values)
    path = tmp_path / "reward.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    with pytest.raises(ValueError): reward.load_semantic_reward_config(path)


def test_old_profiles_outputs_are_exactly_unchanged():
    for config_name in ("ppo_fl_capture_quality_v1", "ppo_residual_rr_fix_v1", "ppo_semantic_v2"):
        path = ROOT / "configs" / config_name / "reward_config.yaml"
        for i in range(1, 14):
            after = state(f"P{i:02d}")
            a = calculate(after, cfg=old_reward.load_semantic_reward_config(path), module=old_reward)
            b = calculate(after, cfg=reward.load_semantic_reward_config(path))
            assert a == b


@pytest.mark.parametrize("phase", ["P01", "P02"])
def test_front_cost_audit_is_bitwise_preserved_without_geometry_deficit(phase):
    after = state(phase)
    old = calculate(after, cfg=old_reward.load_semantic_reward_config(OLD / "reward_config.yaml"), module=old_reward)
    new = calculate(after)
    assert old["front_quality_sample_audit"] == new["front_quality_sample_audit"]
    assert old["families"]["body_stability"] == new["families"]["body_stability"]
    assert new["front_quality_sample_audit"][0]["effective_beta_per_s"] == .03*(1.-.5*.6)
    assert new["task_space_quality_sample_audit"][0]["raw_geometry_cost"] == 0.


@pytest.mark.parametrize("distance,expected", [(0., 1.), (.01, .25), (.02, 0.), (.10, 0.)])
def test_geometry_has_finite_space_floor_no_high_reward(distance, expected):
    result = calculate(state("P06", distance=distance))
    row = result["task_space_quality_sample_audit"][0]
    assert row["raw_geometry_cost"] == pytest.approx(expected)
    assert row["effective_beta_per_s"] == .03
    assert result["families"]["body_stability"] == pytest.approx(-.03*expected/120)
    assert result["families"]["control_smoothness"] == 0.


def test_euclidean_distance_not_global_z_or_swing_support_template():
    s = state("P08", distance=-.02)
    geometry = s.task["physical_evaluator"]["body_traversal_geometry"]
    geometry.update(minimum_w_m=(.7, .81, .03), maximum_w_m=(1., 1., .3))
    row = calculate(s)["task_space_quality_sample_audit"][0]
    assert row["separation_lower_bound_m"] == pytest.approx(.01)
    assert row["raw_geometry_cost"] == pytest.approx(.25)
    assert not s.task["physical_evaluator"]["current_legs"]  # No fixed support template is needed.


@pytest.mark.parametrize("phase", ["P03", "P04", "P10", "P11", "P12", "P13"])
def test_outside_quality_window_has_no_new_cost_or_missing_geometry_gate(phase):
    result = calculate(state(phase, missing=True))
    row = result["task_space_quality_sample_audit"][0]
    assert not row["eligible"] and row["raw_geometry_cost"] is None
    assert result["families"]["body_stability"] == 0.


@pytest.mark.parametrize("kind", ["missing", "nan", "unordered", "clock", "planes"])
def test_missing_or_invalid_nonterminal_geometry_cannot_be_zero_cost(kind):
    s = state("P06", missing=kind == "missing")
    ev = s.task["physical_evaluator"]
    if kind == "nan": ev["body_traversal_geometry"]["minimum_w_m"] = (math.nan, 0., .1)
    if kind == "unordered": ev["body_traversal_geometry"]["minimum_w_m"] = (9., 0., .1)
    if kind == "clock": ev["simulation_time_s"] = 4.
    if kind == "planes": s = replace(s, metrics={**s.metrics, "obstacle_planes_world_m": None})
    with pytest.raises(ValueError, match="measurement unavailable"): calculate(s)


@pytest.mark.parametrize("reason", ["TASK_FAILURE_BODY_COLLISION", "FALL", "NAN_INF"])
def test_classified_terminal_preserves_failure_and_null_unmeasurable_cost(reason):
    result = calculate(state("P09", missing=True, reason=reason), reason=reason)
    row = result["task_space_quality_sample_audit"][0]
    assert row["terminal_measurement_omitted"] and not row["valid"]
    assert row["raw_geometry_cost"] is row["weighted_geometry_cost"] is None
    assert "body_traversal_geometry" in row["reason"]
    assert result["terminal_event"] == -40.
    assert result["potential_after"] == 0. and not result["terminal_bootstrap_allowed"]


def test_per_tick_integration_and_cross_phase_bootstrap_remain_continuous():
    states = [state(phase, tick=i, distance=.01) for i, phase in enumerate(("P05", "P05", "P06"))]
    cfg = reward.load_semantic_reward_config(NEW / "reward_config.yaml")
    result = reward.SemanticRewardCalculator(cfg).evaluate(states[0], states[-1],
        [_sample(a, b) for a, b in zip(states, states[1:])], termination_reason=None, task_success=False)
    assert len(result["task_space_quality_sample_audit"]) == 2
    assert result["elapsed_physics_s"] == 2/120
    assert result["families"]["body_stability"] == pytest.approx(-.03*.25*2/120)
    assert result["potential_shaping"] == 5*(.9985*.45-.45)
    assert result["terminal_bootstrap_allowed"]


def test_rolling_contact_usable_differs_from_history_without_new_force_bonus():
    sup, ev = retention_fixture()
    airborne = sup._current_capture_retention("FL", ev)
    assert airborne == pytest.approx(.5/(1+.02/.003))
    row = ev["current_legs"]["FL"]
    row.update(clearance_m=0., top_contact=True, top_surface_contact=True, air=False, support=True)
    assert sup._current_capture_retention("FL", ev) == 1.
    for force in (.2, 10., 1000.):
        row["bearing_force_n"] = force
        assert sup._current_capture_retention("FL", ev) == 1.
    row["bearing_verified"] = False
    assert sup._current_capture_retention("FL", ev) == .5
    assert ev["history"]["placed"]["FL"]


def test_rolling_retirement_continuous_and_uses_closer_rear_leg():
    sup, ev = retention_fixture()
    start = sup._current_capture_retention("FL", ev)
    values = []
    for distance in (-.3, -.27, -.245, -.22, -.1):
        ev["current_legs"]["RR"]["front_distance_m"] = distance
        values.append(sup._current_capture_retention("FL", ev))
    assert values == pytest.approx([start, start, (1+start)/2, 1., 1.])
    assert ev["current_legs"]["RL"]["front_distance_m"] == -.3
    ev["current_legs"]["RR"]["front_distance_m"] = -.27+1e-10
    assert abs(sup._current_capture_retention("FL", ev)-start) < 1e-8


@pytest.mark.parametrize("escape", ["RR_lift", "RR_placed", "front_unplaced", "old_version", "rear_leg"])
def test_no_permanent_front_support_requirement_or_old_version_change(escape):
    sup, ev = retention_fixture()
    leg = "FL"
    if escape == "RR_lift": ev["current_legs"]["RR"]["current_lift_valid"] = True
    if escape == "RR_placed": ev["history"]["placed"]["RR"] = True
    if escape == "front_unplaced": ev["history"]["placed"]["FR"] = False
    if escape == "old_version": sup.spec = old_supervisor.load_task_spec(OLD / "stage_task_spec.yaml")
    if escape == "rear_leg": leg = "RR"
    saved = deepcopy(ev)
    assert sup._current_capture_retention(leg, ev) == 1.
    assert ev == saved


def test_retention_does_not_reward_penetration_or_depend_on_phase_label():
    sup, ev = retention_fixture()
    row = ev["current_legs"]["FL"]
    for gap in (0., -.001, -.01):
        row["clearance_m"] = gap
        assert sup._current_capture_retention("FL", ev) == .5
    for phase in ("P05", "P06", "P07", "P08", "P09"):
        sup.stage_id = phase
        assert sup._current_capture_retention("FL", ev) == .5


@pytest.mark.parametrize("mutation", [lambda v: v.pop("rolling_capture_retention"),
    lambda v: v["rolling_capture_retention"].update(positive_gap_scale_m=0.),
    lambda v: v.update(revision="fl_capture_quality_v1"),
    lambda v: v.update(capture_retention_semantics=None)])
def test_rolling_config_rejects_silent_or_unreviewed_version(tmp_path, mutation):
    values = yaml.safe_load((NEW / "stage_task_spec.yaml").read_text())
    mutation(values)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(values, sort_keys=False))
    with pytest.raises(ValueError): supervisor.load_task_spec(path)


def test_observation_encoded_groups_unchanged_only_world_planes_metric_added():
    schema = old_observation.load_semantic_observation_schema()
    old = old_observation.SemanticObservationBuilder(schema)
    new = observation.SemanticObservationBuilder(schema)
    for tick in (0, 1, 8):
        frame = _frame(tick, raw=_raw(tick, roll=.2))
        history = dict.fromkeys(old_observation.HISTORY_GROUPS, ZERO12)
        a, b = old.build(frame, history), new.build(frame, history)
        assert a.groups == b.groups and schema.encode(a.groups) == schema.encode(b.groups)
        assert a.task == b.task
        extra = dict(b.metrics)
        live = frame.info["raw_observation"].obstacle
        assert extra.pop("obstacle_planes_world_m") == tuple(getattr(live, key) for key in
            ("front_x_m", "back_x_m", "left_y_m", "right_y_m", "bottom_z_m", "top_z_m"))
        assert a.metrics == extra


def test_hard_evaluator_and_nominal_ast_exactly_unchanged():
    def classes(path):
        tree = ast.parse(path.read_text())
        return {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.ClassDef)}
    old = classes(ROOT / "src/wlr50_clean/ppo/semantic_supervisor.py")
    new = classes(HERE / "src/wlr50_clean/ppo/semantic_supervisor.py")
    assert old["TaskEvaluator"] == new["TaskEvaluator"]
    assert old["NominalMotionProvider"] == new["NominalMotionProvider"]

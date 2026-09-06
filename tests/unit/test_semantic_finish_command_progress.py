"""Opt-in command-stop shaping through real evaluator history and reward APIs."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from test_semantic_supervisor import advance, leg_state, observation, place
from test_semantic_observation_reward_env import _built, _frame, _sample
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.ppo.semantic_reward import FAMILIES, SemanticRewardCalculator, load_semantic_reward_config
from wlr50_clean.ppo.semantic_supervisor import (
    DEFAULT_TASK_SPEC_PATH, LEG_ORDER, PHASE_IDS, SemanticObservationError,
    TaskEvaluator, TaskStageSupervisor, load_task_spec,
)

CONFIG = Path(__file__).resolve().parents[2] / "configs/ppo_semantic_v3"
SPEC = CONFIG / "stage_task_spec.yaml"


def placed_entry(path=SPEC):
    """Earn all history through observations; never install success/history bits."""
    evaluator = TaskEvaluator(path)
    obs = observation()
    for joint in obs["joints"].values():
        joint["command_deg"] = 0.
    evaluator.observe(obs)
    for leg in ("FR", "FL", "RR", "RL"):
        obs = place(evaluator, obs, leg)
    assert all(evaluator.snapshot["history"]["placed"].values())
    assert not evaluator.snapshot["success"]
    obs = advance(obs)
    obs["base"]["position_w_m"][0] = .9
    for leg in LEG_ORDER:
        leg_state(obs, leg, x=.8, bottom=.05, top=True, hip=0.)
    return evaluator, obs


def command_observation(obs, command):
    result = deepcopy(obs)
    # Opposite signs prove the actual maximum magnitude, not signed average.
    for index, wheel in enumerate(result["wheels"].values()):
        wheel["command_rad_s"] = command if index % 2 else -command
    return result


def test_same_physical_state_command_halving_strictly_increases_global_potential():
    original, obs = placed_entry()
    scores, potentials = [], []
    evaluations = []
    for command in (.16, .08, .04, .02):
        evaluator = deepcopy(original)
        evaluation = evaluator.observe(command_observation(obs, command))
        supervisor = TaskStageSupervisor(SPEC, evaluator=evaluator, initial_stage_id="P13")
        scores.append(supervisor.predicate("whole_task_success", evaluation))
        potentials.append(supervisor.physical_potential(evaluation))
        evaluations.append(evaluation)
        assert not evaluation["success"] and evaluation["final_stable_for_s"] == 0.
        assert evaluation["final_controlled"] is (command <= .02)
        assert evaluation["final_region_valid"] and evaluation["final_support_available"]
    assert scores == pytest.approx([.85625, .8625, .875, .9])
    assert potentials == pytest.approx([.9784375, .979375, .98125, .985])
    assert all(a < b for a, b in zip(scores, scores[1:]))
    assert all(a < b for a, b in zip(potentials, potentials[1:]))
    # Every measured physical feature/history is identical; only command differs.
    for evaluation in evaluations[1:]:
        for key in ("goal_features", "current_legs", "history", "home_maximum_servo_error_deg"):
            assert evaluation[key] == evaluations[0][key]


def test_command_progress_is_continuous_at_tolerance_and_does_not_skip_stable_time():
    original, obs = placed_entry()
    scores = []
    for command in (.02 - 1e-9, .02, .02 + 1e-9):
        evaluator = deepcopy(original)
        snapshot = evaluator.observe(command_observation(obs, command))
        scores.append(TaskStageSupervisor(SPEC, evaluator=evaluator).predicate("whole_task_success", snapshot))
        assert not snapshot["success"]
    assert scores[0] == scores[1]
    assert 0 < scores[1] - scores[2] < 3e-9
    evaluator = original
    obs = command_observation(obs, .02)
    snapshot = evaluator.observe(obs)
    assert snapshot["final_controlled"] and not snapshot["success"]
    for _ in range(59):
        obs = advance(obs)
        snapshot = evaluator.observe(obs)
        assert not snapshot["success"]
    obs = advance(obs)
    snapshot = evaluator.observe(obs)
    assert snapshot["final_stable_for_s"] == pytest.approx(.5)
    assert snapshot["success"]


@pytest.mark.parametrize("failure", ["actual_wheel_speed", "body_linear", "body_angular", "region", "support", "collision", "fall"])
def test_command_within_tolerance_does_not_bypass_other_physical_requirements(failure):
    evaluator, obs = placed_entry()
    obs = command_observation(obs, .01)
    if failure == "actual_wheel_speed":
        obs["wheels"][WHEEL_ORDER[0]]["velocity_rad_s"] = .26
    elif failure == "body_linear":
        obs["base"]["linear_velocity_w_m_s"] = [.051, 0., 0.]
    elif failure == "body_angular":
        obs["base"]["angular_velocity_w_rad_s"] = [0., 0., .31]
    elif failure == "region":
        obs["base"]["position_w_m"][1] = 2.
    elif failure == "support":
        for leg in LEG_ORDER:
            leg_state(obs, leg, x=.8, bottom=.05, air=True)
    elif failure == "collision":
        obs["body_collision"]["detected"] = True
    elif failure == "fall":
        obs["imu"]["projected_gravity_b"] = [0., 0., 1.]
    for _ in range(62):
        snapshot = evaluator.observe(obs)
        assert not snapshot["success"]
        assert snapshot["final_stable_for_s"] == 0.
        obs = advance(obs)
    if failure in ("actual_wheel_speed", "body_linear", "body_angular"):
        assert not snapshot["final_controlled"]
    elif failure == "region":
        assert not snapshot["final_region_valid"]
    elif failure == "support":
        assert not snapshot["final_support_available"]
    else:
        assert snapshot["termination_reason"] is not None


def test_ordinary_phase_labels_cannot_change_global_finish_potential():
    evaluator, obs = placed_entry()
    snapshot = evaluator.observe(command_observation(obs, .08))
    values = [TaskStageSupervisor(SPEC, evaluator=evaluator, initial_stage_id=phase).physical_potential(snapshot)
              for phase in PHASE_IDS]
    assert values == [values[0]] * 13


def test_old_v3_without_optin_and_v2_home_stop_semantics_are_unchanged(tmp_path):
    old = load_task_spec(SPEC)
    old["final"].pop("stop_command_progress")
    path = tmp_path / "old_v3.yaml"
    path.write_text(yaml.safe_dump(old, sort_keys=False), encoding="utf-8")
    evaluator, obs = placed_entry(path)
    supervisor = TaskStageSupervisor(path, evaluator=evaluator)
    a = deepcopy(evaluator).observe(command_observation(obs, .16))
    b = deepcopy(evaluator).observe(command_observation(obs, .02))
    assert supervisor.predicate("whole_task_success", a) == supervisor.predicate("whole_task_success", b) == pytest.approx(.9)
    v2, obs2 = placed_entry(DEFAULT_TASK_SPEC_PATH)
    obs2["joints"]["front_left_hip"]["position_deg"] = 20.
    snap = v2.observe(command_observation(obs2, .01))
    legacy_supervisor = TaskStageSupervisor(DEFAULT_TASK_SPEC_PATH, evaluator=v2)
    assert not snap["final_controlled"]  # V2 still requires its home tolerance.
    assert legacy_supervisor.predicate("whole_task_success", snap) == pytest.approx(.85)
    assert "stop_command_progress" not in load_task_spec(DEFAULT_TASK_SPEC_PATH)["final"]


@pytest.mark.parametrize("command", [float("nan"), float("inf"), None, True])
def test_existing_live_command_finite_validation_still_fails_closed(command):
    evaluator, obs = placed_entry()
    obs["wheels"][WHEEL_ORDER[0]]["command_rad_s"] = command
    with pytest.raises(SemanticObservationError, match="finite"):
        evaluator.observe(obs)


def test_optin_config_rejects_unknown_mode_or_invalid_existing_tolerance(tmp_path):
    spec = load_task_spec(SPEC)
    path = tmp_path / "invalid.yaml"
    for mode, tolerance in (("clipped_at_tolerance", .02), ("reciprocal_physical_stop_tolerance", 0.)):
        spec["final"].update(stop_command_progress=mode, maximum_commanded_wheel_speed_rad_s=tolerance)
        path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError):
            load_task_spec(path)


def test_finish_command_progress_enters_only_existing_task_family_once():
    evaluator, obs = placed_entry()
    supervisor = TaskStageSupervisor(SPEC, evaluator=evaluator)
    snapshots = [deepcopy(evaluator).observe(command_observation(obs, value)) for value in (.16, .02)]
    phis = [supervisor.physical_potential(snapshot) for snapshot in snapshots]
    before = _built(_frame(phi=phis[0]))
    after = _built(_frame(1, phi=phis[1]))
    unchanged = replace(after, task={**after.task, "task_progress_potential": phis[0]})
    config = load_semantic_reward_config(CONFIG / "reward_config.yaml")
    baseline = SemanticRewardCalculator(config).evaluate(before, unchanged, [_sample(before, unchanged)],
        termination_reason=None, task_success=False)
    improved = SemanticRewardCalculator(config).evaluate(before, after, [_sample(before, after)],
        termination_reason=None, task_success=False)
    assert tuple(improved["families"]) == FAMILIES
    expected = config.values["potential_weight"] * config.gamma * (phis[1] - phis[0])
    assert improved["total"] - baseline["total"] == pytest.approx(expected)
    assert improved["families"]["task_progress"] - baseline["families"]["task_progress"] == pytest.approx(expected)
    for family in FAMILIES[1:]:
        assert improved["families"][family] == baseline["families"][family]
    assert improved["terminal_event"] == 0.

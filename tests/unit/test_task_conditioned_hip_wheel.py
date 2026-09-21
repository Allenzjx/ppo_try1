"""Task-conditioned exploration and soft quality, using CPU/synthetic data only.

No checkpoint, output-directory bootstrap, physical simulation, or training
credit is required. This file runs unchanged from the project's tests/unit.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from test_semantic_capture_approach import _qualified, _step
from test_semantic_observation_reward_env import _built, _frame, _sample
from wlr50_clean.ppo.semantic_history_actor import (
    SemanticTaskConditionedHipWheelHistoryMLPModel,
    task_conditioned_effective_log_std, task_conditioned_physical_scales,
)
from wlr50_clean.ppo.semantic_policy_distribution import (
    TASK_CONDITIONED_PHYSICAL_B_TABLE, REQUEST_HISTORY_CAPS,
)
from wlr50_clean.ppo.semantic_reward import SemanticRewardCalculator, load_semantic_reward_config
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor, load_task_spec

CONFIGS_ROOT = Path(__file__).resolve().parents[2] / "configs"
CONFIG = CONFIGS_ROOT / "ppo_task_conditioned_hip_wheel_v1"


@pytest.fixture(autouse=True)
def preserve_torch_rng():
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)


def _latent(phase):
    x = torch.zeros(1, 372)
    x[0, phase] = 1
    x[0, 158:158 + phase] = 1
    x[0, 20] = .01
    x[0, 195:207] = torch.linspace(-.2, .2, 12)
    # Both rear wheels are initially outside the preparation region.
    x[0, 28] = x[0, 31] = -.3
    return x


def _rolling():
    x = _latent(5)
    x[0, 154:156] = 1  # Actual placement history is not current support.
    x[0, 132] = x[0, 134] = 1
    x[0, 124] = x[0, 126] = .1
    x[0, 22] = x[0, 25] = .1
    x[0, 100] = x[0, 103] = -.1
    return x


def _assert_B(x, label):
    B, cap, evidence = task_conditioned_physical_scales(x)
    assert torch.equal(B[0], torch.tensor(TASK_CONDITIONED_PHYSICAL_B_TABLE[label]))
    assert torch.equal(cap[0], torch.tensor(REQUEST_HISTORY_CAPS[int(x[0, :13].argmax())]))
    return evidence


def test_all_thirteen_phases_keep_twelve_positive_finite_sigmas():
    for phase in range(13):
        x = _latent(phase)
        log_std, _ = task_conditioned_effective_log_std(torch.full((1, 12), -2.), x, .25)
        sigma = log_std.exp()
        assert sigma.shape == (1, 12)
        assert bool((torch.isfinite(sigma) & (sigma > 0)).all())


@pytest.mark.parametrize("leg", ["FR", "FL"])
def test_approach_sigma_requires_current_air_and_task_history(leg):
    x = _latent(1 if leg == "FR" else 4)
    if leg == "FR":
        x[0, 147], x[0, 24], x[0, 33] = 1, .015, 3
        label, contact, placed = "FR_air_approach_proxy", 133, 155
    else:
        x[0, 150], x[0, 21] = 1, .003
        label, contact, placed = "FL_crossed_air_pending", 131, 154
    _assert_B(x, label)
    for index in (contact, placed):
        incompatible = x.clone()
        incompatible[0, index] = 1
        _, _, evidence = task_conditioned_physical_scales(incompatible)
        assert evidence["task_state_weights"][label].item() == 0


def test_p06_rolling_and_contact_recovery_are_distinct_observable_states():
    x = _rolling()
    _assert_B(x, "P06_front_support_rolling_proxy")
    x[0, 132] = 0  # FL current obstacle contact lost, placement history remains.
    evidence = _assert_B(x, "P06_front_support_recovery_proxy")
    assert evidence["task_state_weights"]["P06_front_support_rolling_proxy"].item() == 0
    assert x[0, 154].item() == 1


@pytest.mark.parametrize("rear_index", [28, 31])
def test_closer_rear_wheel_continuously_releases_p06_rolling_preference(rear_index):
    x = _rolling()
    rolling = torch.tensor(TASK_CONDITIONED_PHYSICAL_B_TABLE["P06_front_support_rolling_proxy"])
    prepare = torch.tensor(TASK_CONDITIONED_PHYSICAL_B_TABLE["RR_preparation"])
    for distance, fraction in ((-.3, 0.), (-.27, 0.), (-.245, .5), (-.22, 1.)):
        x[0, rear_index] = distance
        B, _, evidence = task_conditioned_physical_scales(x)
        assert evidence["task_state_weights"]["RR_preparation"].item() == pytest.approx(fraction, abs=1e-6)
        torch.testing.assert_close(B[0], torch.lerp(rolling, prepare, fraction))


def test_current_valid_rr_lift_overrides_rolling_and_preparation_not_stale_history():
    x = _rolling()
    x[0, 28] = -.22
    x[0, 149] = 1
    _assert_B(x, "RR_current_valid_lift")
    x[0, 149] = 0  # Completed prior stages alone cannot recreate a current lift.
    _assert_B(x, "RR_preparation")
    x[0, 157] = 1
    _, _, evidence = task_conditioned_physical_scales(x)
    assert all(evidence["task_state_weights"][name].item() == 0 for name in
               ("RR_preparation", "RR_current_valid_lift", "P06_front_support_rolling_proxy"))


def test_actual_raw_sample_log_probability_entropy_share_one_gaussian():
    x = _rolling()
    obs = TensorDict({"policy": x, "critic": x.clone()}, batch_size=[1])
    model = SemanticTaskConditionedHipWheelHistoryMLPModel(
        obs, {"actor": ["policy"]}, "actor", 12,
        observation_layout="diagonal_transfer_state_v1", exploration_std_temperature=.25,
        distribution_cfg={"class_name": "HeteroscedasticGaussianDistribution", "std_type": "log", "init_std": .15})
    state = torch.get_rng_state()
    raw = model(obs, stochastic_output=True)
    after = torch.get_rng_state()
    normal = torch.distributions.Normal(model.output_mean, model.output_std)
    torch.set_rng_state(state)
    assert torch.equal(raw, normal.sample()) and torch.equal(after, torch.get_rng_state())
    assert torch.equal(model.get_output_log_prob(raw), normal.log_prob(raw).sum(-1))
    assert torch.equal(model.output_entropy, normal.entropy().sum(-1))


def _reward_frame(tick, *, phase="P06", missing=False):
    frame = _built(_frame(tick, stage=phase, phi=.45))
    evaluator = {"valid": True, "simulation_time_s": tick / 120, "history": {}, "current_legs": {}}
    if not missing:
        evaluator["body_traversal_geometry"] = dict(valid=True,
            minimum_w_m=(.7, -.2, .06), maximum_w_m=(1., .2, .3))
    return replace(frame, task={**frame.task, "physical_evaluator": evaluator},
        metrics={**frame.metrics, "obstacle_planes_world_m": (.5, 2., .8, -.8, 0., .05)})


def _reward_result(*, missing=False, terminal=None, phase="P06"):
    before, after = _reward_frame(0, phase=phase), _reward_frame(1, phase=phase, missing=missing)
    calc = SemanticRewardCalculator(load_semantic_reward_config(CONFIG / "reward_config.yaml"))
    return calc.evaluate(before, after, [_sample(before, after)], termination_reason=terminal, task_success=False)


def test_live_geometry_cost_and_missing_nonterminal_measurement_are_not_confused():
    row = _reward_result()["task_space_quality_sample_audit"][0]
    assert row["separation_lower_bound_m"] == pytest.approx(.01)
    assert row["raw_geometry_cost"] == pytest.approx(.25)
    with pytest.raises(ValueError, match="measurement unavailable"):
        _reward_result(missing=True)
    outside = _reward_result(missing=True, phase="P13")["task_space_quality_sample_audit"][0]
    assert not outside["eligible"] and outside["raw_geometry_cost"] is None


@pytest.mark.parametrize("reason", ["TASK_FAILURE_BODY_COLLISION", "NAN_INF"])
def test_missing_terminal_geometry_stays_null_and_does_not_hide_failure(reason):
    result = _reward_result(missing=True, terminal=reason)
    row = result["task_space_quality_sample_audit"][0]
    assert row["terminal_measurement_omitted"] and not row["valid"]
    assert row["raw_geometry_cost"] is row["weighted_geometry_cost"] is None
    assert result["terminal_event"] == -40. and result["potential_after"] == 0.
    assert not result["terminal_bootstrap_allowed"]


def test_fl_near_surface_air_is_soft_progress_not_hard_capture():
    spec = load_task_spec(CONFIG / "stage_task_spec.yaml")
    # The established sensor fixture predates RR-v3 velocity inputs. Earn the
    # unchanged FR/FL hard events with that evaluator, then apply the new soft
    # potential to its snapshots; do not claim this fixture verifies RR sensors.
    ev, obs, _ = _qualified("FL")
    sup = TaskStageSupervisor(CONFIG / "stage_task_spec.yaml", evaluator=ev)
    obs, airborne = _step(ev, obs, "FL", x=.53, bottom=.0504, air=True)
    assert not airborne["history"]["placed"]["FL"]
    assert sup._current_capture_progress("FL", airborne, 0.) < .5
    assert sup.predicate("placed_FL", airborne) < 1.
    for _ in range(spec["history"]["minimum_top_samples"]):
        obs, captured = _step(ev, obs, "FL", x=.53, bottom=.05, top=True)
    assert captured["history"]["placed"]["FL"] and sup.predicate("placed_FL", captured) == 1.


def test_soft_rolling_retention_does_not_rewrite_history_or_require_permanent_contact():
    sup = TaskStageSupervisor.__new__(TaskStageSupervisor)
    sup.spec = load_task_spec(CONFIG / "stage_task_spec.yaml")
    row = dict(top_xy_outside_distance_m=0., clearance_m=.02, front_distance_m=-.3,
               top_contact=False, top_surface_contact=False, support=False,
               bearing_verified=True, air=True, ground_contact=False)
    ev = dict(valid=True, history=dict(placed=dict(FR=True, FL=True, RR=False, RL=False)),
              current_legs={leg: deepcopy(row) for leg in ("FR", "FL", "RR", "RL")})
    original = deepcopy(ev)
    assert sup._current_capture_retention("FL", ev) == pytest.approx(.5 / (1 + .02 / .003))
    assert ev == original and sup.predicate("placed_FL", ev) == 1.
    ev["current_legs"]["RL"]["front_distance_m"] = -.22
    assert sup._current_capture_retention("FL", ev) == 1.
    ev = deepcopy(original)
    ev["current_legs"]["RR"]["current_lift_valid"] = True
    assert sup._current_capture_retention("FL", ev) == 1.

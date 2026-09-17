"""Isolated zero/home routing; no Isaac launch or learned-policy success claim."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from wlr50_clean.ppo import semantic_cli as cli
from wlr50_clean.ppo import semantic_video as video
from wlr50_clean.ppo.semantic_migration import experiment_namespace
from wlr50_clean.ppo.semantic_policy_distribution import CONFIG_NAMES
from wlr50_clean.ppo.semantic_reward import load_semantic_reward_config
from wlr50_clean.ppo.semantic_supervisor import load_task_spec
from wlr50_clean.ppo.semantic_video_cli import build_video_core, validate_video_args


EXPERIMENT = "non_residual_refine_v1"
SOURCE = cli.PROJECT_ROOT / "configs/ppo_task_first_recovery_v1"
TARGET = cli.PROJECT_ROOT / "configs/ppo_non_residual_refine_v1"


def request(**changes):
    args = cli.parser().parse_args([
        "eval", "--run-dir", str(cli.PROJECT_ROOT / "runs/ppo_non_residual_refine_v1/unit_not_run"),
        "--expected-head", "a" * 40, "--semantic-version", "v3", "--experiment-id", EXPERIMENT,
        "--mode", "semantic_prior_eval", "--seed", "4001", "--max-decisions", "3000", "--no-headless",
    ])
    for name, value in changes.items():
        setattr(args, name, value)
    return args


def test_new_paths_and_video_configuration_are_isolated():
    assert experiment_namespace("v3", EXPERIMENT) == "ppo_non_residual_refine_v1"
    assert cli.version_paths("v3", experiment_id=EXPERIMENT) == (
        cli.PROJECT_ROOT / "runs/ppo_non_residual_refine_v1",
        cli.PROJECT_ROOT / "outputs/ppo_non_residual_refine_v1", TARGET)
    configs = video.video_configuration("v3", experiment_id=EXPERIMENT)
    assert {p.name for p in configs.values()} == CONFIG_NAMES
    assert all(p.parent == TARGET for p in configs.values())
    with pytest.raises(ValueError):
        experiment_namespace("v2", EXPERIMENT)


def test_six_configs_change_only_explicit_terminal_nominal_mode():
    assert {p.name for p in TARGET.iterdir() if p.is_file()} == CONFIG_NAMES
    for name in CONFIG_NAMES - {"stage_task_spec.yaml"}:
        assert (TARGET / name).read_bytes() == (SOURCE / name).read_bytes()
    before = yaml.safe_load((SOURCE / "stage_task_spec.yaml").read_text(encoding="utf-8"))
    expected = deepcopy(before)
    expected["nominal"]["final_stop_owner"] = "source_home_after_physical_stop_v1"
    actual = yaml.safe_load((TARGET / "stage_task_spec.yaml").read_text(encoding="utf-8"))
    assert actual == expected
    assert before["nominal"]["final_stop_owner"] == "current_physical_stop_nominal_owner_v1"
    assert load_task_spec(TARGET / "stage_task_spec.yaml")["final"] == before["final"]
    reward = load_semantic_reward_config(TARGET / "reward_config.yaml").values
    assert reward["quality_epsilon"] == 0.0
    assert reward["family_weights"]["task_progress"] == 1.0
    assert all(reward["family_weights"][key] == 0. for key in (
        "body_stability", "contact_motion_quality", "control_smoothness", "control_regularization"))


def test_prior_natural_P01_and_readonly_preflight_are_allowed():
    cli.validate_request(request())
    cli.validate_request(request(command="preflight", seed=1001))
    assert validate_video_args(request()) == "B"


@pytest.mark.parametrize("changes", [
    {"command": "train"}, {"command": "smoke"},
    {"mode": "semantic_residual_eval"}, {"mode": "legacy_fsm_eval"},
    {"checkpoint": Path("unused_checkpoint.pt")}, {"resume_migration": Path("unused_plan.json")},
    {"new_mdp_warm_start": True}, {"policy_distribution_migration": True},
    {"num_envs": 8}, {"from_phase": "P06"}, {"teacher_offset_decisions": 1},
    {"prefix_source": "successful_nominal"}, {"prefix_source": "checkpoint_policy"},
])
def test_refine_never_routes_training_checkpoint_teacher_or_other_roles(changes):
    with pytest.raises(ValueError, match="prior-only"):
        cli.validate_request(request(**changes))


@pytest.mark.parametrize("role", ["A", "C"])
def test_direct_video_core_cannot_bypass_prior_only_guard(role):
    with pytest.raises(video.SemanticVideoError, match="prior-only role B"):
        build_video_core(None, role=role, semantic_version="v3", experiment_id=EXPERIMENT)


def test_new_video_retains_actual_task_interval_without_padding():
    receipt = video.task_interval_receipt(6116, experiment_id=EXPERIMENT)
    assert receipt["frame_count"] == 765
    assert receipt["last_frame_episode_tick"] == 6116
    assert receipt["extra_physics_ticks"] == 0
    assert receipt["extra_pre_frames"] == receipt["extra_post_frames"] == 0


def test_other_experiment_paths_and_windows_unchanged():
    assert cli.version_paths("v3", experiment_id="task_first_recovery_v1")[2] == SOURCE
    assert cli.version_paths("v3", experiment_id="fsm_reference_p09_stable_v2")[2].name == "ppo_fsm_reference_p09_stable_v2"
    assert video.task_interval_receipt(6116, experiment_id="task_first_recovery_v1")["frame_count"] == 765


@pytest.mark.parametrize("name", ["run_semantic_ppo.ps1", "run_semantic_video.ps1"])
def test_existing_launchers_expose_new_experiment_without_new_engine(name):
    script = (cli.PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")
    assert "'task_first_recovery_v1','non_residual_refine_v1'" in script
    assert '"ppo_$ExperimentId"' in script

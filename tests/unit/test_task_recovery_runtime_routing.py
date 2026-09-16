"""Task-first changes reward identity, not physical/reset/video semantics."""
from pathlib import Path

import pytest

from wlr50_clean.ppo import semantic_cli as cli, semantic_video as video


EXPERIMENT = "task_first_recovery_v1"


def test_new_experiment_routes_all_three_namespaces():
    runs, outputs, configs = cli.version_paths("v3", experiment_id=EXPERIMENT)
    assert runs == cli.PROJECT_ROOT / "runs/ppo_task_first_recovery_v1"
    assert outputs == cli.PROJECT_ROOT / "outputs/ppo_task_first_recovery_v1"
    assert configs == cli.PROJECT_ROOT / "configs/ppo_task_first_recovery_v1"
    args = cli.parser().parse_args(["train", "--run-dir", str(runs / "test"),
        "--expected-head", "a"*40, "--semantic-version", "v3", "--experiment-id", EXPERIMENT])
    assert cli._request_paths(args) == (runs, outputs, configs)


@pytest.mark.parametrize("ticks", [1, 812, 8857, 24000])
def test_task_video_has_exact_same_real_clock_without_added_roll(ticks):
    before = video.task_interval_receipt(ticks)
    after = video.task_interval_receipt(ticks, experiment_id=EXPERIMENT)
    assert after == {**before, "experiment_id": EXPERIMENT}
    assert video.source_frame_count({"experiment_id": EXPERIMENT,
        "episode_physics_ticks": ticks}) == (ticks+7)//8
    assert after["extra_physics_ticks"] == after["extra_pre_frames"] == after["extra_post_frames"] == 0


@pytest.mark.parametrize("role", ["A", "B", "C"])
def test_new_namespace_uses_same_actual_natural_reset_without_teacher(role, monkeypatch):
    from test_semantic_video_natural_reset import actual_backend_receipt, ENTRY
    _, frame = actual_backend_receipt(role, monkeypatch)
    proof = video.current_video_natural_reset_proof(frame.info, role=role,
        contract={"semantic_version": "v3", "experiment_id": EXPERIMENT}, entry=ENTRY)
    assert proof["experiment_id"] == EXPERIMENT
    assert proof["reset_metadata"]["reset_prime_tick_count"] == 0
    assert proof["entry"]["physics_tick"] == 0


def test_selected_video_and_training_config_are_identical():
    configs = video.video_configuration("v3", experiment_id=EXPERIMENT)
    root = cli.version_paths("v3", experiment_id=EXPERIMENT)[2]
    assert Path(configs["execution_profile"]) == root / "execution_profile.yaml"
    assert Path(configs["reward_config_path"]) == root / "reward_config.yaml"

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from wlr50_clean.ppo import semantic_cli
from wlr50_clean.ppo import semantic_video as video
from wlr50_clean.ppo import semantic_video_cli as cli
from wlr50_clean.ppo.semantic_policy_distribution import CONFIG_NAMES


EXPERIMENT = "rr_rl_timing_policy_learning_v1"


def test_rear_video_binds_exact_seventh_curriculum_configuration(tmp_path, monkeypatch):
    config = tmp_path / f"configs/ppo_{EXPERIMENT}"
    config.mkdir(parents=True)
    names = set(CONFIG_NAMES) | {"curriculum_plan.json"}
    for name in names:
        (config / name).write_text(name, encoding="utf-8")
    monkeypatch.setattr(semantic_cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(semantic_cli, "version_paths",
                        lambda *args, **kwargs: (tmp_path, tmp_path, config))
    configs = video.video_configuration("v3", experiment_id=EXPERIMENT)
    files = {path.relative_to(tmp_path).as_posix(): cli._sha256_path(path)
             for path in configs.values()}
    selected = {path.name: {"path": path.relative_to(tmp_path).as_posix(),
                            "sha256": cli._sha256_path(path)}
                for path in configs.values()}
    contract = {"experiment_id": EXPERIMENT, "files": files,
                "selected_configuration": selected}
    assert len(configs) == 7 and {path.name for path in configs.values()} == names
    video._validate_video_configuration_binding(configs, contract, experiment_id=EXPERIMENT)
    for missing in names:
        bad = copy.deepcopy(contract)
        del bad["selected_configuration"][missing]
        with pytest.raises(video.SemanticVideoError, match="seven selected"):
            video._validate_video_configuration_binding(configs, bad, experiment_id=EXPERIMENT)


def media_boundary(tmp_path, monkeypatch):
    source_head, target_head = cli.MEDIA_COMPAT_SOURCE_HEAD, "b" * 40
    source_blobs = {
        "src/wlr50_clean/ppo/semantic_video.py": b"SOURCE_VIDEO = 1\n",
        "src/wlr50_clean/ppo/semantic_video_cli.py":
            b"SOURCE_CLI = 1\ndef build_video_core():\n    return 1\n",
        "src/wlr50_clean/ppo/control.py": b"CONTROL = 1\n",
    }
    target_blobs = {
        **source_blobs,
        "src/wlr50_clean/ppo/semantic_video.py": b"TARGET_VIDEO = 1\n",
        "src/wlr50_clean/ppo/semantic_video_cli.py":
            b"TARGET_CLI = 1\ndef build_video_core():\n    return 1\n",
    }
    selected = {}
    for name in set(CONFIG_NAMES) | {"curriculum_plan.json"}:
        relative = f"configs/ppo_{EXPERIMENT}/{name}"
        source_blobs[relative] = target_blobs[relative] = name.encode()
        selected[name] = {"path": relative,
                          "sha256": cli._sha256_bytes(source_blobs[relative])}
    source_files = {path: cli._sha256_bytes(value) for path, value in source_blobs.items()}
    target_files = {path: cli._sha256_bytes(value) for path, value in target_blobs.items()}
    common = {"schema": "wlr50_clean.semantic_runtime_contract.v1",
              "semantic_version": "v3", "experiment_id": EXPERIMENT,
              "selected_configuration": selected, "physics_hz": 120.0,
              "decision_hz": 15.0, "task_timeout_s": 200.0}
    source = {**common, "source_git_commit": source_head, "files": source_files,
              "runtime_content_sha256": cli._runtime_files_sha256(source_files)}
    target = {**copy.deepcopy(common), "source_git_commit": target_head,
              "files": target_files,
              "runtime_content_sha256": cli._runtime_files_sha256(target_files)}
    checkpoint = tmp_path / "checkpoint_step_000222080.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    manifest.write_text(json.dumps({"runtime_contract": source,
        "checkpoint_sha256": cli._sha256_path(checkpoint),
        "save_load_round_trip": True}), encoding="utf-8")
    blobs = {(source_head, path): value for path, value in source_blobs.items()}
    blobs.update({(target_head, path): value for path, value in target_blobs.items()})
    monkeypatch.setattr(cli, "_git_blob_bytes", lambda head, path: blobs[(head, path)])
    args = NS(checkpoint=checkpoint, command="eval", mode="semantic_residual_eval",
              semantic_version="v3", experiment_id=EXPERIMENT,
              resume_migration=None, new_mdp_warm_start=False,
              policy_distribution_migration=False, checkpoint_output_branch=None)
    return args, source, target, source_blobs, target_blobs


def test_media_only_video_eval_uses_saved_load_contract_and_explicit_receipt(tmp_path, monkeypatch):
    args, source, target, _, _ = media_boundary(tmp_path, monkeypatch)
    load_contract, receipt = cli.reviewed_video_checkpoint_runtime(args, target, role="C")
    assert load_contract == source
    assert receipt["scope"] == "semantic_video_cli_eval_only"
    assert set(receipt["reviewed_media_delta"]) == cli.MEDIA_COMPAT_FILES
    assert receipt["selected_configuration"] == source["selected_configuration"]
    assert receipt["all_other_runtime_files_identical"] is True
    assert receipt["build_video_core_ast_unchanged"] is True
    assert receipt["official_checkpoint_load_uses_source_runtime"] is True
    assert receipt["capture_manifest_uses_evaluation_runtime"] is True
    assert receipt["training_allowed"] is False


@pytest.mark.parametrize("fault", ["third_file", "config", "source_blob", "training"])
def test_media_only_video_eval_fails_closed(tmp_path, monkeypatch, fault):
    args, source, target, source_blobs, _ = media_boundary(tmp_path, monkeypatch)
    if fault == "third_file":
        target["files"]["src/wlr50_clean/ppo/control.py"] = "0" * 64
        target["runtime_content_sha256"] = cli._runtime_files_sha256(target["files"])
    elif fault == "config":
        target["selected_configuration"].pop("curriculum_plan.json")
    elif fault == "source_blob":
        original = cli._git_blob_bytes
        monkeypatch.setattr(cli, "_git_blob_bytes", lambda head, path:
                            b"tampered" if (head == cli.MEDIA_COMPAT_SOURCE_HEAD
                                             and path == "src/wlr50_clean/ppo/semantic_video.py")
                            else original(head, path))
    else:
        args.command = "train"
    with pytest.raises((RuntimeError, KeyError)):
        cli.reviewed_video_checkpoint_runtime(args, target, role="C")


def test_exact_runtime_checkpoint_needs_no_media_exception(tmp_path, monkeypatch):
    args, source, _, _, _ = media_boundary(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_git_blob_bytes",
                        lambda *args: pytest.fail("exact resume should not inspect compatibility blobs"))
    load_contract, receipt = cli.reviewed_video_checkpoint_runtime(args, source, role="C")
    assert load_contract == source and receipt is None

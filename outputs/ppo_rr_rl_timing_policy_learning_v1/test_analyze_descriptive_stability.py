from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("analyze_descriptive_stability.py")
SPEC = importlib.util.spec_from_file_location("_generic_stability", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


HEAD = "1" * 40


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sealed_fixture(tmp_path: Path) -> tuple[Path, str, str, Path]:
    run = tmp_path / "run"
    source = run / "source"
    source.mkdir(parents=True)
    checkpoint = tmp_path / "checkpoint_step_000000128.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    checkpoint_sha = digest(checkpoint)
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    sidecar.write_text(json.dumps({
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "save_load_round_trip": True,
        "global_policy_decisions": 128,
        "ppo_updates": 1,
        "optimizer_steps": 20,
        "actor_parameter_sha256": "a" * 64,
        "runtime_contract": {
            "experiment_id": "rr_rl_timing_policy_learning_v1",
            "source_git_commit": HEAD,
        },
    }), encoding="utf-8")
    manifest = {
        "schema": "wlr50_clean.semantic_video_source.v1",
        "experiment_id": "rr_rl_timing_policy_learning_v1",
        "role": "C", "from_phase": "P01", "episode_count": 1,
        "optimizer_updates": 0,
        "runtime_contract": {"source_git_commit": HEAD},
        "policy_sampling_mode": "deterministic_conditional_mean",
        "checkpoint_load_provenance": {
            "checkpoint_loaded_and_verified": True,
            "stochastic_policy": False,
            "policy_seed": None,
            "saved_global_policy_decisions": 128,
            "parameter_hashes": {"actor_parameter_sha256": "a" * 64},
            "source": {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_sha,
                "manifest": str(sidecar.resolve()),
                "manifest_sha256": digest(sidecar),
            },
        },
    }
    manifest_path = source / "semantic_video_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run_path = run / "run_manifest.json"
    run_path.write_text(json.dumps({
        "completed_at_utc": "2026-09-24T00:00:00Z",
        "lifecycle": "DIAGNOSTIC_FAILURE",
    }), encoding="utf-8")
    return source, digest(manifest_path), digest(run_path), checkpoint


def test_label_is_derived_from_verified_bound_sidecar(tmp_path: Path) -> None:
    source, manifest_sha, run_sha, _ = sealed_fixture(tmp_path)
    identity = module.sealed_checkpoint_identity(source, manifest_sha, run_sha, HEAD)
    assert identity["display_label"] == "CP128"
    assert identity["lifetime_counters"] == {
        "global_policy_decisions": 128,
        "ppo_updates": 1,
        "optimizer_steps": 20,
    }
    assert identity["label_derived_from_verified_sealed_load_provenance"] is True


def test_checkpoint_byte_change_is_rejected(tmp_path: Path) -> None:
    source, manifest_sha, run_sha, checkpoint = sealed_fixture(tmp_path)
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checkpoint/sidecar bytes"):
        module.sealed_checkpoint_identity(source, manifest_sha, run_sha, HEAD)

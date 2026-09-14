"""Deferred current-video natural-reset proof tests. NOT RUN; no live Isaac proof."""
from __future__ import annotations

import copy
import json

import pytest
from wlr50_clean.ppo import semantic_video as video

CONTRACT = {"semantic_version": "v3", "experiment_id": "fsm_reference_p09_stable_v2"}
ENTRY = {"state_id": "P01", "physics_tick": 0, "decision_count": 0, "done": False}


def actual_backend_receipt(role, monkeypatch):
    # Use the real backend reset implementation with existing fake physical
    # dependencies, not the incorrect marker=None semantic capture fixture.
    from test_isaac_fsm_backend import FakeRuntime, _backend
    from test_semantic_backend import SemanticFrameStub
    from wlr50_clean.ppo import semantic_backend
    runtime = FakeRuntime()
    monkeypatch.setattr(semantic_backend, "_sha256_file", lambda path: "a" * 64)
    backend = (_backend(runtime) if role == "A" else semantic_backend.SemanticIsaacBackend(
        dependencies=runtime.dependencies(), controller_factory=lambda *args: SemanticFrameStub()))
    frame = backend.reset(seed=4001, options={})
    return runtime, frame


def source_with_proof(role, info):
    proof = video.current_video_natural_reset_proof(info, role=role, contract=CONTRACT, entry=ENTRY)
    return {"role": role, "runtime_contract": CONTRACT, "from_phase": "P01",
            "natural_reset_proof": proof,
            "reset_evidence": {key: info[key] for key in
                ("reset_count", "reset_options", "training_phase_snapshot")}}


@pytest.mark.parametrize("role", ["A", "B", "C"])
def test_real_backend_natural_receipt_is_accepted_and_persisted_without_side_effects(role, monkeypatch):
    runtime, frame = actual_backend_receipt(role, monkeypatch)
    counts = (runtime.sim.step_count, runtime.adapter.write_count, runtime.reader_count)
    assert frame.physics_tick == 0 and frame.state_id == "P01"
    assert frame.info["training_phase_snapshot"] == (None if role == "A" else "P01")
    assert frame.info["phase_snapshot_restoration"]["mode"] == (
        "normal_p01_reset" if role == "A" else "semantic_natural_P01")
    source = source_with_proof(role, frame.info)
    loaded = json.loads(json.dumps(source))  # Exactly the artifact replay boundary.
    assert video.validate_current_video_natural_reset(loaded) == source["natural_reset_proof"]
    assert (runtime.sim.step_count, runtime.adapter.write_count, runtime.reader_count) == counts
    assert counts[0] == counts[1] == 180
    assert loaded["natural_reset_proof"]["reset_metadata"]["reset_prime_tick_count"] == 0


@pytest.mark.parametrize("role", ["A", "B", "C"])
@pytest.mark.parametrize("fault", [
    "second_reset", "explicit_p01_snapshot", "explicit_start", "loaded_p01_hidden_options",
    "historical_restore", "missing_restoration", "missing_requested_phase", "missing_prime",
    "prime_replay", "sensor_offset", "entry_clock", "marker_mismatch", "missing_execution",
    "snapshot_file", "source_tick", "unknown_mode",
])
def test_current_natural_reset_rejects_actual_or_unproven_snapshots_equally(role, fault, monkeypatch):
    _, frame = actual_backend_receipt(role, monkeypatch)
    info = copy.deepcopy(frame.info)
    restoration = info["phase_snapshot_restoration"]
    if fault == "second_reset":
        info["reset_count"] = 2
    elif fault == "explicit_p01_snapshot":
        info["reset_options"] = {"training_phase_snapshot": "P01"}
    elif fault == "explicit_start":
        info["reset_options"] = {"start_phase": "P01"}
    elif fault == "loaded_p01_hidden_options":
        info["training_phase_snapshot"] = "P01"
        restoration.update(mode="normal_p01_reset", requested_phase="P01", snapshot_validated=True)
    elif fault == "historical_restore":
        restoration.update(mode="phase_entry_snapshot", requested_phase="P09")
    elif fault == "missing_restoration":
        del info["phase_snapshot_restoration"]
    elif fault == "missing_requested_phase":
        del restoration["requested_phase"]
    elif fault == "missing_prime":
        del info["reset_prime_tick_count"]
    elif fault == "prime_replay":
        info["reset_prime_tick_count"] = 1
    elif fault == "sensor_offset":
        info["sensor_tick_at_effective_entry"] = 120
    elif fault == "entry_clock":
        info["fsm_and_episode_clock_at_effective_entry"] = 1
    elif fault == "marker_mismatch":
        info["training_phase_snapshot"] = "P01" if role == "A" else None
    elif fault == "missing_execution":
        if role == "A":
            info["execution_mode"] = "semantic_B_or_C"
        else:
            del info["execution_mode"]
    elif fault == "snapshot_file":
        restoration["snapshot_path"] = "real-snapshot.json"
    elif fault == "source_tick":
        restoration["source_tick"] = 0
    else:
        restoration["mode"] = "unproven_natural_P01"
    with pytest.raises(video.SemanticVideoError):
        video.current_video_natural_reset_proof(info, role=role, contract=CONTRACT, entry=ENTRY)


@pytest.mark.parametrize("entry", [
    dict(ENTRY, state_id="P09"), dict(ENTRY, physics_tick=1),
    dict(ENTRY, decision_count=1), dict(ENTRY, done=True),
])
def test_receipt_cannot_override_actual_noninitial_episode_state(entry, monkeypatch):
    _, frame = actual_backend_receipt("C", monkeypatch)
    with pytest.raises(video.SemanticVideoError, match="initial natural P01"):
        video.current_video_natural_reset_proof(frame.info, role="C", contract=CONTRACT, entry=entry)


@pytest.mark.parametrize("fault", ["missing", "count", "mode", "prime", "role", "entry", "summary"])
def test_persisted_receipt_replay_rejects_missing_or_changed_proof(fault, monkeypatch):
    _, frame = actual_backend_receipt("C", monkeypatch)
    source = json.loads(json.dumps(source_with_proof("C", frame.info)))
    proof = source["natural_reset_proof"]
    if fault == "missing":
        del source["natural_reset_proof"]
    elif fault == "count":
        proof["reset_metadata"]["reset_count"] = 2
    elif fault == "mode":
        proof["reset_metadata"]["phase_snapshot_restoration"]["mode"] = "phase_entry_snapshot"
    elif fault == "prime":
        proof["reset_metadata"]["reset_prime_tick_count"] = 1
    elif fault == "role":
        proof["role"] = "A"
    elif fault == "entry":
        proof["entry"]["physics_tick"] = 1
    else:
        source["reset_evidence"]["training_phase_snapshot"] = None
    with pytest.raises(video.SemanticVideoError):
        video.validate_current_video_natural_reset(source)


def source_until_artifact_boundary(version, experiment):
    return {"schema": "wlr50_clean.semantic_video_source.v1", "role": "B",
        "mode": video.ROLES["B"], "success_candidate": True, "diagnostic_only": False,
        "improved_claim": False, "optimizer_updates": 0, "seed": 4001,
        "episode_count": 1, "fresh_process_single_episode": True,
        "camera": {**video.CAMERA, "resolution": [1280, 720], "fps": 15},
        "stitched": False, "frame_interpolation": False, "speed_modified": False,
        "semantic_version": version, "experiment_id": experiment,
        "pre_action_ticks": video.PRE_TICKS, "performed_post_success_ticks": video.POST_TICKS,
        "artifacts": {"boundary": {}}}


@pytest.mark.parametrize("version,experiment", [
    ("v2", None), ("v3", None), ("v3", "all_stage_acceptance_v1"),
])
def test_older_video_sources_do_not_acquire_the_new_reset_proof_requirement(tmp_path, monkeypatch, version, experiment):
    class ArtifactBoundary(Exception):
        pass
    source = source_until_artifact_boundary(version, experiment)
    (tmp_path / "semantic_video_source_manifest.json").write_text(json.dumps(source))
    monkeypatch.setattr(video, "validate_current_video_natural_reset",
                        lambda *a, **k: pytest.fail("historical video acquired new proof requirement"))
    def boundary(*args):
        raise ArtifactBoundary
    monkeypatch.setattr(video, "inside", boundary)
    with pytest.raises(ArtifactBoundary):
        video.validate_semantic_video_source(tmp_path)


def test_current_independent_source_validator_checks_reset_proof_before_artifacts(tmp_path, monkeypatch):
    source = source_until_artifact_boundary("v3", video.TASK_WINDOW_EXPERIMENT)
    # No proof: a claimed current success cannot get past the common guard.
    (tmp_path / "semantic_video_source_manifest.json").write_text(json.dumps(source))
    monkeypatch.setattr(video, "inside", lambda *a: pytest.fail("missing proof reached artifact processing"))
    with pytest.raises(video.SemanticVideoError, match="persisted natural reset proof"):
        video.validate_semantic_video_source(tmp_path)

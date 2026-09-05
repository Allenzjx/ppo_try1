from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import wlr50_clean.ppo.semantic_backend as semantic_module
from wlr50_clean.ppo.semantic_backend import build_semantic_projector, CONFIG_ROOT, SemanticIsaacBackend
from wlr50_clean.ppo.action_projection import bitwise_full12_equal, SafetyProjection


def project(projector, raw, phase="P01", previous=(0.0,) * 12, **kwargs):
    return projector.project(raw, state_id=phase, nominal_action_full12=(0.0,) * 12,
        reference_action_full12=(99.0,) * 8 + (1.0,) * 4,
        reference_delta_full12=(0.0,) * 12,
        previous_projected_residual_full12=previous, **kwargs)


@pytest.mark.parametrize("phase", [f"P{i:02}" for i in range(1, 14)])
def test_all_phases_open_physical_channels_and_no_reference_hard_cap(phase):
    p = build_semantic_projector()
    assert p.config.mask_for(phase) == (1,) * 12
    row = project(p, (10.0,) * 12, phase)
    assert row.safe_projected_residual_full12 == pytest.approx((0.5,) * 8 + (0.015,) * 4)
    assert all(row.recording_envelope_exceeded_full12[:8])


def test_zero_identity_is_same_history_not_a_trajectory_claim():
    p = build_semantic_projector()
    zero = project(p, (0.0,) * 12)
    assert zero.zero_residual_fast_path
    assert bitwise_full12_equal(zero.applied_action_full12, (0.0,) * 12)
    history = (1.0,) * 8 + (0.06,) * 4
    decay = project(p, (0.0,) * 12, previous=history)
    assert not decay.zero_residual_fast_path
    assert decay.safe_projected_residual_full12 == pytest.approx((0.5,) * 8 + (0.045,) * 4)


def test_phase_change_retains_feasible_exploration_including_FR_wheel():
    p = build_semantic_projector()
    previous = (0.1,) * 8 + (0.02,) * 4
    row = project(p, (0.2,) * 12, "P08", previous=previous)
    assert row.effective_action_mask_full12[9] == 1
    assert row.applied_action_full12[9] > 0


def test_reactive_collision_stop_is_not_predictive_guarantee():
    row = project(build_semantic_projector(), (1.0,) * 12,
                  safety=SafetyProjection(body_collision_detected=True))
    assert row.applied_action_full12[8:] == (0.0,) * 4
    assert row.hard_safety_modified


def test_protected_A_files_are_unchanged_using_original_manifest():
    root = CONFIG_ROOT.parents[1]
    payload = json.loads((root / "artifacts/ppo_phase_v1_start/frozen_fsm_hashes.json").read_text())
    for name, expected in payload["protected_files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name


class SemanticFrameStub:
    motion = SimpleNamespace(servo_rate_limit_deg_s=150.0)
    task_progress = 0.1
    task_snapshot = {"goal_features": {}, "history": {}, "stage_age_s": 0.0}

    def __init__(self):
        self.tick = -1

    def step(self, observation, *, sim_time_s):
        self.tick += 1
        return SimpleNamespace(
            physics_tick=self.tick, sim_time_s=sim_time_s, state_id="P01",
            full12=(0.0,) * 12, lifecycle="EXECUTE_MOTION", termination=None,
            full12_atomic_write_required=True, tracking_servo_names=(),
            drive_feedback_bias_full12=(0.0,) * 12, normal_drive_bias_full12=(0.0,) * 12,
        )


def test_semantic_backend_uses_explicit_factory_and_one_atomic_write(monkeypatch):
    from test_isaac_fsm_backend import FakeRuntime
    runtime = FakeRuntime()
    dependencies = runtime.dependencies()
    dependencies = replace(dependencies, controller_from_paths=lambda *args: pytest.fail("legacy controller constructed"))
    monkeypatch.setattr(semantic_module, "_sha256_file", lambda path: "a" * 64)
    backend = SemanticIsaacBackend(dependencies=dependencies,
        controller_factory=lambda *args: SemanticFrameStub())
    initial = backend.reset(seed=1001, options={})
    assert initial.info["supervisor_schema"] == "task_semantic_v2"
    assert initial.info["level_calibration_sample_count"] == 0
    assert initial.info["level_reference_orientation_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert initial.info["level_calibration"]["pitch_error_to_level_rad"] == pytest.approx(0.2)
    writes = runtime.adapter.write_count
    steps = runtime.sim.step_count
    result = backend.step_physics((0.1,) * 8 + (0.02,) * 4)
    assert runtime.adapter.write_count == writes + 1
    assert runtime.sim.step_count == steps + 1
    assert result.physics_tick == 1
    assert result.info["in_episode_root_pose_writes"] == 0
    assert result.info["atomic_ack"]["ppo_projected_residual_full12"] == pytest.approx((0.1,) * 8 + (0.02,) * 4)
    assert result.info["mapper_state_summary"]["requested_servo_deg"] == [0.0] * 8


def test_failed_semantic_reset_cannot_reuse_previous_generation(monkeypatch):
    from test_isaac_fsm_backend import FakeRuntime
    runtime = FakeRuntime()
    monkeypatch.setattr(semantic_module, "_sha256_file", lambda path: "a" * 64)
    backend = SemanticIsaacBackend(dependencies=runtime.dependencies(),
        controller_factory=lambda *args: SemanticFrameStub())
    backend.reset(seed=1001, options={})
    with pytest.raises(Exception, match="only accepts natural P01"):
        backend.reset(seed=1002, options={"start_phase": "P10"})
    with pytest.raises(Exception, match="committed reset generation"):
        backend.step_physics((0.0,) * 12)

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from test_semantic_observation_reward_env import _frame, _raw
from wlr50_clean.ppo.semantic_legacy_evaluation import (
    PhysicalEvaluationRecorder, WRITE_COUNTERS, _evaluation_legacy,
)

ZERO = (0.,) * 12
PROJECTION = SimpleNamespace(safe_projected_residual_full12=ZERO)


def audited_frame(tick, *, collision=False, controller_success=False):
    raw = _raw(tick)
    if collision:
        raw = replace(raw, body_collision=replace(raw.body_collision, detected=True, real_pair_active=True))
    frame = _frame(tick, raw=raw)
    frame.info.pop("semantic_task")  # Legacy frames have no new supervisor label.
    frame.info.update({key: 0 for key in WRITE_COUNTERS})
    frame.info["atomic_ack"] = {"drive_target_full12": ZERO}
    frame.info["controller_task_result"] = "SUCCESS" if controller_success else None
    frame.info["actuator_target_effect_audit"] = {
        "schema": "wlr50_clean.actuator_target_effect_audit.v1", "verified": True,
        "actual_mapping_matches_dispatch": True, "setter_dispatch_targets_equal": True,
        "same_tick_counterfactual": True, "raw_policy_action_full12": ZERO,
        "target_dtype": "torch.float32", "changed_target_channel_count": 0,
        "source_phase_id": "P01",
    }
    return frame


def test_independent_evaluator_does_not_accept_internal_success_and_raw_replays(tmp_path):
    recorder = PhysicalEvaluationRecorder(tmp_path)
    first, second = audited_frame(0), audited_frame(1, controller_success=True)
    try:
        recorder.start(first)
        recorder.observe(first, second, PROJECTION)
        result = recorder.summary()
        assert result["task_success"] is False
        assert result["quality_metrics"]["fixed_quality_score"] is None
        assert result["physical_task_duration_s"] == pytest.approx(1 / 120)
    finally:
        recorder.close()
    rows = [json.loads(line) for line in (tmp_path / "physical_observations.jsonl").read_text().splitlines()]
    assert rows[1]["base"]["position_w_m"] == [0., 0., .3]
    assert "velocity_deg_s" in rows[1]["joints"]["front_left_hip"]
    assert all(
        "active" in contact["ground"] for contact in rows[1]["contacts"].values())
    from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator
    replay = TaskEvaluator()
    for row in rows:
        replay.observe(row)
    assert replay.snapshot["goal_features"] == result["physical_task_evaluation"]["goal_features"]


def test_real_body_collision_is_independent_failure(tmp_path):
    recorder = PhysicalEvaluationRecorder(tmp_path)
    try:
        a, b = audited_frame(0), audited_frame(1, collision=True)
        recorder.start(a); recorder.observe(a, b, PROJECTION)
        result = recorder.summary()
        assert not result["task_success"]
        assert result["physical_task_evaluation"]["termination_reason"] == "TASK_FAILURE_BODY_COLLISION"
    finally:
        recorder.close()


@pytest.mark.parametrize("fault", ["missing_write", "write", "audit", "source_phase"])
def test_missing_or_invalid_native_and_write_evidence_fails_closed(tmp_path, fault):
    recorder = PhysicalEvaluationRecorder(tmp_path)
    a, b = audited_frame(0), audited_frame(1)
    if fault == "missing_write": del b.info[WRITE_COUNTERS[0]]
    if fault == "write": b.info[WRITE_COUNTERS[0]] = 1
    if fault == "audit": b.info["actuator_target_effect_audit"]["verified"] = False
    if fault == "source_phase": b.info["actuator_target_effect_audit"]["source_phase_id"] = "P02"
    try:
        recorder.start(a)
        with pytest.raises(RuntimeError): recorder.observe(a, b, PROJECTION)
    finally:
        recorder.close()


def test_legacy_adapter_cannot_publish_label_only_success(tmp_path, monkeypatch):
    class Backend:
        def __init__(self, *args, **kwargs): pass
        def set_actuator_target_audit_request(self, **kwargs): assert kwargs["raw_policy_action_full12"] == ZERO
    class Core:
        def __init__(self, backend, **kwargs):
            self.done = False
            self.phase_actions = SimpleNamespace(mask_for=lambda _: (1,) * 12)
        def reset(self, **kwargs): self.frame = audited_frame(0)
        def step(self, action):
            old = self.frame
            self.frame = audited_frame(1, controller_success=True)
            self.tick_callback(old, self.frame, PROJECTION)
            self.done = True
            return SimpleNamespace(info={**self.frame.info, "termination_reason": "SUCCESS"})
    import wlr50_clean.ppo.isaac_fsm_backend as backend_module
    import wlr50_clean.ppo.residual_direct_env as env_module
    monkeypatch.setattr(backend_module, "IsaacFSMBackend", Backend)
    monkeypatch.setattr(env_module, "ResidualEpisodeEnv", Core)
    args = SimpleNamespace(run_dir=tmp_path, seed=2001, max_decisions=3000)
    result = _evaluation_legacy(None, args, {})
    assert result["legacy_controller_result"] == "SUCCESS"
    assert not result["task_success"]
    assert result["termination_reason"] == "LEGACY_ENDED_WITHOUT_PHYSICAL_TASK_COMPLETION"

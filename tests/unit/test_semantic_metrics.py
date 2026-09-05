from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_semantic_observation_reward_env import Backend, _frame, _raw
from wlr50_clean.ppo.semantic_cli import _evaluation, parser
from wlr50_clean.ppo.semantic_env import SemanticEpisodeEnv
from wlr50_clean.ppo.semantic_metrics import PHASES, SemanticMetricsAccumulator


def frame(tick, *, phase="P01", pitch=0.0, omega=(0.0, 0.0, 0.0)):
    result = _frame(tick, stage=phase, raw=_raw(tick, pitch=pitch, roll=0.3, omega=omega))
    result.info["atomic_ack"] = {"drive_target_full12": (0.0,) * 12}
    return result


PROJECTION = SimpleNamespace(safe_projected_residual_full12=(0.0,) * 12)


def test_real_observation_euler_pitch_rate_is_not_body_y_omega():
    metrics = SemanticMetricsAccumulator()
    metrics.observe(frame(0, pitch=0.1, omega=(0.0, 3.0, 0.0)),
                    frame(1, pitch=0.101, omega=(0.0, 3.0, 0.0)), PROJECTION)
    row = metrics.rows[0]
    assert row["pitch_rate_rad_s"] == pytest.approx(0.12)
    assert row["body_y_omega_rad_s"] != pytest.approx(0.12)
    assert metrics.summary()["global"]["pitch_rate_rms_rad_s"] == pytest.approx(0.12)


def test_missing_phase_score_is_unavailable_and_all13_score_is_fixed_weighted_sum():
    metrics = SemanticMetricsAccumulator()
    metrics.observe(frame(0, pitch=0.1), frame(1, pitch=0.101), PROJECTION)
    partial = metrics.summary()
    assert partial["fixed_quality_score"] is None
    assert partial["phases"]["P02"]["sampled"] is False
    assert partial["phases"]["P02"]["quality_score"] is None
    for index, phase in enumerate(PHASES[1:], 1):
        metrics.observe(frame(index, phase=phase, pitch=0.1 + index * 0.001),
                        frame(index + 1, phase=phase, pitch=0.1 + (index + 1) * 0.001), PROJECTION)
    full = metrics.summary()
    assert full["all_phases_sampled"] is True
    # Only Euler pitch changes, 0.12 rad/s; all five other quality dynamics are zero.
    assert full["fixed_quality_score"] == pytest.approx(0.12 * 0.25 / 0.5)
    assert full["global"]["duration_s"] == pytest.approx(13 / 120)
    assert full["score_is_not_success"] is True


def test_physics_dt_and_clock_continuity_are_enforced():
    metrics = SemanticMetricsAccumulator()
    with pytest.raises(ValueError, match="one real physics tick"):
        metrics.observe(frame(0), frame(2), PROJECTION)
    metrics.observe(frame(0), frame(1), PROJECTION)
    with pytest.raises(ValueError, match="continuous"):
        metrics.observe(frame(2), frame(3), PROJECTION)
    metrics.reset()
    metrics.observe(frame(0), frame(1), PROJECTION)
    assert metrics.summary()["global"]["physics_ticks"] == 1


def test_actual_core_eval_installs_quality_callback_and_preserves_failed_task(tmp_path):
    class QualityBackend(Backend):
        def reset(self, *, seed, options):
            result = super().reset(seed=seed, options=options)
            result.info["atomic_ack"] = {"drive_target_full12": result.info["drive_target_full12"]}
            return result

        def step_physics(self, applied_action_full12):
            result = super().step_physics(applied_action_full12)
            if self.tick == self.terminal_tick:
                from dataclasses import replace
                raw = result.info["raw_observation"]
                result.info["raw_observation"] = replace(raw, body_collision=replace(raw.body_collision, detected=True, real_pair_active=True))
            result.info["atomic_ack"] = {"drive_target_full12": result.info["drive_target_full12"]}
            from test_semantic_training import native_audit
            result.info["actuator_target_effect_audit"] = {**native_audit(self.requests[-1][1]), "source_phase_id": "P01"}
            for key in ("in_episode_root_pose_writes", "in_episode_root_velocity_writes",
                        "in_episode_force_or_impulse_writes", "in_episode_gravity_writes"):
                result.info[key] = 0
            return result

    core = SemanticEpisodeEnv(QualityBackend(terminal_tick=3, reason="BODY_COLLISION"), collect_trace=False)
    args = parser().parse_args(["eval", "--run-dir", str(tmp_path), "--expected-head", "a" * 40,
                               "--mode", "semantic_prior_eval", "--seed", "2001", "--max-decisions", "128"])
    result = _evaluation(core, args, contract={})
    assert result["task_success"] is False
    assert result["termination_reason"] == "BODY_COLLISION"
    assert result["quality_metrics"]["global"]["physics_ticks"] == 3
    assert result["quality_metrics"]["global"]["duration_s"] == pytest.approx(3 / 120)
    assert result["quality_metrics"]["fixed_quality_score"] is None
    assert result["physical_task_evaluation"]["termination_reason"] == "TASK_FAILURE_BODY_COLLISION"

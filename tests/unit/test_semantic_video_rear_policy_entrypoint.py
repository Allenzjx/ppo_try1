"""The rear-policy namespace uses the current natural-P01 video path."""
from pathlib import Path
from types import SimpleNamespace

from wlr50_clean.ppo.semantic_video import (
    camera_for_experiment,
    capture_assist_tick_evidence,
    rear_policy_video_control_metadata,
    task_interval_receipt,
    video_configuration,
)


EXPERIMENT = "rr_rl_timing_policy_learning_v1"
ROOT = Path(__file__).resolve().parents[2]


def test_rear_policy_video_uses_current_config_camera_and_task_clock():
    configs = video_configuration("v3", experiment_id=EXPERIMENT)
    assert set(configs) == {
        "task_spec_path", "quality_score_path", "execution_profile",
        "reward_config_path", "observation_schema_path", "action_schema_path",
    }
    assert all(path.parent == ROOT / "configs" / ("ppo_" + EXPERIMENT)
               for path in configs.values())
    assert camera_for_experiment(EXPERIMENT) == camera_for_experiment(
        "rr_capture_then_rl_transfer_v1")
    receipt = task_interval_receipt(24000, experiment_id=EXPERIMENT)
    assert receipt["frame_count"] == 3000
    assert receipt["encoded_duration_s"] == receipt["physical_duration_s"] == 200.
    assert receipt["extra_physics_ticks"] == receipt["extra_pre_frames"] == 0


def test_rear_policy_tick_evidence_marks_real_no_rear_assist_state():
    frame = SimpleNamespace(
        physics_tick=8, sim_time_s=8 / 120., state_id="P01",
        nominal_action_full12=(0.,) * 12, action_mask_full12=(1,) * 12,
        info={
            "raw_observation": {}, "capture_assist": {"active": False},
            "rr_capture_assist": None, "rear_task_assist_disabled": True,
            "rear_policy_timing": {"mode": "RR_CARRY_CAPTURE"},
            "semantic_task": {"physical_evaluator": {}}, "atomic_ack": {},
        })
    row = capture_assist_tick_evidence(frame)
    assert row["capture_assist"] == {"active": False}
    assert row["rr_capture_assist"] is None
    assert row["rear_task_assist_disabled"] is True
    assert row["rear_policy_timing"] == {"mode": "RR_CARRY_CAPTURE"}


def test_rear_policy_source_labels_split_prior_from_policy_without_rear_assist():
    residual = rear_policy_video_control_metadata("C")
    prior = rear_policy_video_control_metadata("B")
    for row in (residual, prior):
        assert row["front_fl_capture_assist"] == {
            "enabled": True, "mode": "p05_hip_only_continuation_v1",
            "is_policy_learning": False}
        assert row["rear_task_assist"]["enabled"] is False
        assert row["rear_task_assist"]["rr_capture_assist_mode"] is None
        assert row["rear_task_assist"]["rr_capture_wheel_mode"] == "off"
        assert row["rear_task_assist"]["nominal_geometry_advisory"] is None
    assert residual["rear_task_assist"]["policy_controls_rear_task_actions"] is True
    assert residual["rear_task_assist"]["rear_action_request"] == "residual_policy"
    assert prior["rear_task_assist"]["policy_controls_rear_task_actions"] is False
    assert prior["rear_task_assist"]["rear_action_request"] == "zero_residual_prior"

"""Correct two analysis-only target statistics; no torch, model, or optimizer."""
import hashlib
import json
import math
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.infrastructure.command_batch import (
    SERVO_ORDER, HIP_LIMIT_DEG, KNEE_LIMIT_DEG,
    WHEEL_VELOCITY_LIMIT_RAD_S, logical_readback_from_physical,
)

SOURCE = OUT / "block10B_learning_health_readonly.json"
DEST = OUT / "block10B_learning_health_readonly_corrected.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not DEST.exists()
    r = json.loads(SOURCE.read_text(encoding="utf-8"))
    run = Path(r["run"])
    assert sha(run / "optimizer_updates.jsonl") == r["optimizer_updates_sha256"]
    channels = list(r["P09_execution_last_tick524"]["headroom_clips"])
    # JSON sort order differs from canonical execution order.
    canonical = ["FL_hip", "FL_knee", "FR_hip", "FR_knee", "RL_hip", "RL_knee",
                 "RR_hip", "RR_knee", "FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel"]
    assert sorted(canonical) == channels
    counts = {k: [0] * 12 for k in ("post_assist_clamp_slew_cast_difference", "final_hard_bound_near")}
    n = 0
    owners = {}
    maximum_after_assist_servo_error = 0.0
    with (run / "residual_and_projection_audit.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            a = row["applied_audit"]
            if a["phase_id"] != "P09":
                continue
            t = a["actuator_target_effect_audit"]
            assert t["verified"] and t["setter_dispatch_targets_equal"] and t["actual_mapping_matches_dispatch"]
            physical = t["actual_native_targets"]
            standing = dict(zip(SERVO_ORDER, t["tracking_reference_evidence"]["standing_pose_deg"], strict=True))
            actual = logical_readback_from_physical(physical["servo_position_rad"],
                physical["wheel_velocity_rad_s"], standing)
            assist = t["capture_assist_evidence"]
            candidate = assist["candidate_after_assist_full12"]
            for i in assist["owner_indices"]:
                owners[canonical[i]] = owners.get(canonical[i], 0) + 1
            maximum_after_assist_servo_error = max(maximum_after_assist_servo_error,
                *(abs(actual[i] - assist["final_servo_target_deg"][i]) for i in range(8)))
            # This compares completed PhysX dispatch targets, not measured q/qd.
            for i, value in enumerate(actual):
                tolerance = 1e-4 if i < 8 else 1e-6
                assert math.isfinite(value)
                limits = (HIP_LIMIT_DEG if i % 2 == 0 else KNEE_LIMIT_DEG) if i < 8 else (
                    -WHEEL_VELOCITY_LIMIT_RAD_S, WHEEL_VELOCITY_LIMIT_RAD_S)
                counts["post_assist_clamp_slew_cast_difference"][i] += abs(candidate[i] - value) > tolerance
                counts["final_hard_bound_near"][i] += min(abs(value - edge) for edge in limits) <= tolerance
            n += 1
    assert n == 524 and owners == {"FL_hip": 9, "FL_knee": 9}
    assert maximum_after_assist_servo_error < 1e-4
    # Preserve the original report and all unaffected model/distribution results.
    invalid = r["P09_execution_last_tick524"].pop("final_clamp_slew_or_cast_difference")
    assert set(invalid.values()) == {1.0}
    for key, values in counts.items():
        r["P09_execution_last_tick524"][key] = dict(zip(canonical, [v / n for v in values], strict=True))
    r["schema"] = "wlr50_clean.block10B_bounded_learning_health_readonly.v2_corrected_dispatch_statistics"
    r["target_statistics_correction"] = {
        "original_report_path": str(SOURCE), "original_report_sha256": sha(SOURCE),
        "correction_helper_path": str(Path(__file__).resolve()), "correction_helper_sha256": sha(__file__),
        "invalid_old_fields": ["P09_execution_last_tick524.final_clamp_slew_or_cast_difference",
                               "P09_execution_last_tick524.final_hard_bound_near"],
        "reason": "native_drive_target_full12 is the mapped nominal input, not the final actuator target; old comparisons were invalid",
        "correct_source": "actual_native_targets servo_position_rad/wheel_velocity_rad_s, inverse canonical mapping with same-row standing pose",
        "physical_axis_signs": {"servos": [1, 1, 1, 1, -1, -1, -1, -1], "wheels": [-1, 1, -1, 1]},
        "units": "canonical servo degrees / canonical wheel rad/s", "wheel_hard_limit_rad_s": WHEEL_VELOCITY_LIMIT_RAD_S,
        "assist_owners_by_channel": owners,
        "maximum_dispatch_servo_inverse_mapping_vs_final_receipt_error_deg": maximum_after_assist_servo_error,
        "new_model_forwards": 0, "new_optimizer_steps": 0,
    }
    r["limitations"] = [x for x in r["limitations"] if not x.startswith("Final candidate-versus-dispatch")]
    r["limitations"].append("Corrected post-assist candidate versus actual dispatch differences combine final clamp/slew/dtype effects, not measured physical tracking. FL assist owns nine of the 524 endpoints and is separated before this comparison.")
    with DEST.open("x", encoding="utf-8") as f:
        json.dump(r, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({"corrected_report": str(DEST), "sha256": sha(DEST), "P09": n,
        "new_forwards": 0, "PPO_added": 0, "AUX_added": 0,
        "corrected_target_stats": {key: r["P09_execution_last_tick524"][key] for key in counts}}))


if __name__ == "__main__":
    main()

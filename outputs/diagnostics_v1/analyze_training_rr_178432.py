"""Bounded, sealed training RR audit. No actor, evaluator, GAE, or simulation."""
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/ppo_residual_rr_fix_v1/train/20260917T0454003203604Z_g6c2121b68654_7079dc3b6d4f471d8d3dfe5404d49924"
OUT = ROOT / "outputs/diagnostics_v1/training_RR_carry_178432.json"
sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, SERVO_COMMAND_SIGN


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main():
    manifest = read_json(RUN / "run_manifest.json")
    assert manifest["lifecycle"] == "SUCCEEDED" and manifest["completed_at_utc"]
    result = manifest["result"]
    assert result["actual_policy_decisions"] == 512
    with (RUN / "residual_and_projection_audit.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    assert len(rows) == 512 and all(not r["terminal"] for r in rows)
    ticks = [r["applied_audit"]["physics_tick"] for r in rows]
    assert ticks == list(range(2688, 6777, 8))
    assert [r["global_policy_decision"] for r in rows] == list(range(177921, 178433))
    with (RUN / "optimizer_updates.jsonl").open(encoding="utf-8") as stream:
        updates = [json.loads(line) for line in stream]
    last_audit = rows[-1]["applied_audit"]
    final_task = last_audit["semantic_task"]
    history = final_task["physical_evaluator"]["history"]
    rr_events = [e for e in history["lift_attempt_events"] if e["leg"] == "RR"]
    transitions = []
    snapshots = []
    precursor_source = []
    for index, row in enumerate(rows):
        a = row["applied_audit"]
        tick = a["physics_tick"]
        t = a["semantic_task"]
        e = t["physical_evaluator"]
        n = t["nominal_provider_diagnostics"]
        layers = n["source_partial_order"]["layers"]
        transitions.extend(a["stage_transition_evidence"])
        if 4000 <= tick < 4264:
            precursor_source.append({"tick": tick, "stage_id": t["stage_id"],
                "RR_current": e["current_legs"]["RR"], "layers": layers,
                "N": a["nominal_action_full12"], "final": a["actual_drive_target_full12"]})
        if tick < 4264:
            continue
        tracking = a["actuator_target_effect_audit"]["tracking_reference_evidence"]
        actual = [(math.degrees(q) - standing) / SERVO_COMMAND_SIGN[name]
            for name, q, standing in zip(SERVO_ORDER, tracking["actual_measured_physical_rad"],
                                         tracking["standing_pose_deg"])]
        raw = row["raw_policy_action_full12"]
        mu = row["old_distribution_mean_full12"]
        prev_raw = rows[index - 1]["raw_policy_action_full12"]
        snapshots.append({
            "global_decision": row["global_policy_decision"], "tick": tick,
            "simulation_time_s": a["sim_time_s"], "request_phase": a["phase_id"],
            "end_phase": a["end_phase_id"], "stage_id": t["stage_id"],
            "RR": e["current_legs"]["RR"],
            "other_legs": {leg: e["current_legs"][leg] for leg in ("FL", "FR", "RL")},
            "RR_transfer_direction": e["transfer_roles"]["RR"]["transfer_direction_context"],
            "RR_history": {k: history_map["RR"] for k, history_map in e["history"].items()
                           if isinstance(history_map, dict) and "RR" in history_map},
            "body_geometry": e["body_traversal_geometry"], "goal_features": e["goal_features"],
            "source_layers": layers, "capture_owner_hold": n.get("capture_owner_hold"),
            "height_recovery": n.get("height_recovery"),
            "raw": raw, "stored_conditional_mean": mu,
            "stored_effective_sigma": row["old_distribution_std_full12"],
            "stored_innovation": [v - m for v, m in zip(raw, mu)],
            "previous_saved_raw": prev_raw,
            "history_carry_inferred_rho_09": [0.9 * v for v in prev_raw],
            "N": a["nominal_action_full12"], "projected_residual": a["projected_residual_full12"],
            "final": a["actual_drive_target_full12"],
            "mapped_N": a["actuator_target_effect_audit"]["native_drive_target_full12"],
            "geometry_adjusted_N": a["actuator_target_effect_audit"].get("geometry_adjusted_native_full12"),
            "geometry_evidence": a["actuator_target_effect_audit"].get("nominal_geometry_evidence"),
            "actual_servo_canonical_deg_before_last_dispatch": actual,
            "actual_servo_sample_tick": tick - 1,
            "actual_wheel_velocity_rad_s": e["measured_wheel_velocity_rad_s"],
            "actual_wheel_order": e["stop_progress_wheel_order"],
            "applied_wheel_commands_rad_s": e["applied_wheel_command_rad_s"],
            "mask": a["actuator_target_effect_audit"]["phase_mask_full12"],
            "dispatch_verified": a["actuator_target_effect_audit"]["verified"],
            "handoff_hold_ticks": [v for v in a["actuator_target_effect_audit_ticks"] if v["handoff_hold_used"]],
            "phase_action_jump": a["phase_transition_action_jump"],
            "reward": row["reward"], "reward_breakdown": a["reward_breakdown"],
            "termination_reason": a["termination_reason"], "terminal": row["terminal"]})
    readiness = []
    for s in precursor_source + snapshots:
        for layer in s.get("source_layers", s.get("layers", [])):
            value = layer.get("current_free_lift_source_readiness")
            if value is not None:
                readiness.append({"saved_tick": s["tick"], "stage": layer["stage"],
                                  "source_ticks": layer["source_ticks"], **value})
    source_changes = []
    previous_key = None
    for s in snapshots:
        key = tuple((v["stage"], v.get("status"), v.get("wait_reason")) for v in s["source_layers"])
        if key != previous_key:
            source_changes.append({"tick": s["tick"], "layers": s["source_layers"]})
            previous_key = key
    post = [s for s in snapshots if s["tick"] >= history["event_ticks"]["placed"]["RR"]]
    first = lambda predicate: next((s["tick"] for s in post if predicate(s)), None)
    summary = {
        "execution_lifecycle": manifest["lifecycle"], "completed_at_utc": manifest["completed_at_utc"],
        "actual_policy_decisions": 512, "global_policy_decisions": 178432,
        "ppo_updates_this_run": result["ppo_updates_this_run"],
        "optimizer_steps_this_run": result["optimizer_steps_this_run"],
        "episode_terminal": last_audit["termination_reason"], "episode_still_nonterminal": True,
        "full_task_success": last_audit["full_task_success"],
        "first_current_incomplete_stage": final_task["stage_id"],
        "completed_stage_ids": final_task["completed_stage_ids"],
        "phase_decision_counts_full_512_block": dict(Counter(r["applied_audit"]["phase_id"] for r in rows)),
        "window_snapshots": len(snapshots), "all12_masks_all_ones": all(s["mask"] == [1.0] * 12 for s in snapshots),
        "all_dispatch_verified": all(s["dispatch_verified"] for s in snapshots),
        "source_readiness_snapshot_count": len(readiness),
        "source_wait_snapshot_count": sum(r.get("ready") is False for r in readiness),
        "RR_event_ticks": {k: v.get("RR") for k, v in history["event_ticks"].items()},
        "first_saved_postplacement_front_distance_negative_tick": first(lambda s: s["RR"]["front_distance_m"] < 0),
        "first_saved_postplacement_air_tick": first(lambda s: s["RR"]["air"]),
        "first_saved_postplacement_not_top_tick": first(lambda s: not s["RR"]["top_contact"]),
        "first_saved_postplacement_ground_tick": first(lambda s: s["RR"]["ground_contact"]),
        "first_saved_postplacement_current_lift_invalid_tick": first(lambda s: not s["RR"]["current_lift_valid"]),
        "max_postplacement_front_distance": max((s["RR"]["front_distance_m"], s["tick"]) for s in post),
        "final_RR": snapshots[-1]["RR"], "final_RL": snapshots[-1]["other_legs"]["RL"]}
    report = {
        "schema": "wlr50_clean.sealed_training_RR_carry_window.v1", "source_run": str(RUN),
        "bounds": [4264, 6776], "preceding_source_context_only": [4000, 4256],
        "scope": "Original saved decision snapshots and exact recorded event ticks; no full physics log exists in this training run. Servo readback is the previous physics tick. Unknown inter-snapshot contacts are not filled in; free-reference ticks identify actual evaluator baseline refreshes.",
        "no_simulation_no_policy_forward_no_GAE_recalculation": True,
        "historical_CP177152_labels_unchanged": True, "summary": summary,
        "updates": updates, "RR_attempt_events": rr_events,
        "stage_transitions": transitions, "source_readiness_snapshots": readiness,
        "source_status_changes": source_changes, "preceding_source_context": precursor_source,
        "snapshots": snapshots}
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"output": str(OUT), "summary": summary,
                      "RR_attempt_events": rr_events, "stage_transitions": transitions,
                      "readiness": readiness}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

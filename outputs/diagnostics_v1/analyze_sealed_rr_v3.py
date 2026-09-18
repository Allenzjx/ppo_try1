"""Read one explicit, bounded RR window of a sealed video evaluation only.

No policy/evaluator imports, simulation, GAE, relabeling, or active-run reads.
Examples: --source <run>/source --start-tick 4000 --end-tick 6000
          --output <new.json>. Output creation is exclusive (never overwrite).
"""
import argparse
from collections import Counter
import itertools
import json
from pathlib import Path


def load(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def last_json(path, require_step_info=False):
    """Read the last complete JSON row without scanning the whole decision log."""
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        chunk = min(size, 131072)
        while chunk:
            stream.seek(size - chunk)
            lines = stream.read(chunk).splitlines()
            complete_lines = lines if chunk == size else lines[1:]
            for line in reversed(complete_lines):
                if line.strip():
                    row = json.loads(line)
                    if not require_step_info or "step_info" in row:
                        return row
            if chunk == size:
                break
            chunk = min(size, chunk * 2)
    raise ValueError("Empty sealed decision log")


def indexed_window(path, start, end, tick_key, first_tick, keep=None):
    """Existing logs are dense one-row-per-tick; verify that invariant, not guess."""
    with path.open(encoding="utf-8") as stream:
        rows = itertools.islice(stream, start - first_tick, end - first_tick + 1)
        count = 0
        for tick, line in enumerate(rows, start):
            count += 1
            if keep is not None and tick not in keep:
                continue
            row = json.loads(line)
            if row.get(tick_key) != tick:
                raise ValueError(f"Non-dense/wrong-index {path.name}: expected {tick}")
            yield row
        if count != end - start + 1:
            raise ValueError(f"Truncated sealed window in {path}")


def analyze(args):
    source = args.source.resolve(strict=True)
    # The seal is checked BEFORE reading any large episode logs.
    run = load(source.parent / "run_manifest.json")
    if not run.get("completed_at_utc") or run.get("lifecycle") not in {
        "SUCCEEDED", "FAILED", "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY"
    }:
        raise ValueError("Refusing unsealed/nonterminal run")
    manifest = load(source / "semantic_video_source_manifest.json")
    if manifest.get("episode_count") != 1:
        raise ValueError("Only one sealed physical episode is supported")
    start, end = args.start_tick, args.end_tick
    terminal_tick = manifest["episode_physics_ticks"]
    if not (1 <= start <= end <= terminal_tick) or end - start > 6000:
        raise ValueError("Require 1 <= start <= end <= sealed endpoint; <=6001 rows")
    decision_path = source / "video_policy_decisions.jsonl"
    last = last_json(decision_path)
    if last["end_tick"] != terminal_tick:
        raise ValueError("Last decision disagrees with sealed endpoint")
    # A fixed endpoint may interrupt the final 8-tick action before env.step returns.
    # The sealed physical evaluator is authoritative; never invent its missing step_info.
    last_returned = last_json(decision_path, require_step_info=True)
    final_task = last_returned["step_info"]["semantic_task"]
    evaluator = manifest.get("physical_episode", {}).get("physical_task_evaluation", {})
    if not evaluator:
        raise ValueError("Missing sealed physical task evaluation")
    history = evaluator.get("history", {})
    decisions = []
    with decision_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["end_tick"] > end:
                break
            if row["end_tick"] >= start and "step_info" in row:
                decisions.append(row)
    if not decisions:
        raise ValueError("No saved decision snapshots in requested window")
    ticks = {row["end_tick"] for row in decisions}
    physical = {}
    previous = None
    contact_segments = []
    for p in indexed_window(source / "physical_observations.jsonl", start, end,
                            "physics_tick", 0):
        tick = p["physics_tick"]
        contact = p["contacts"]["rear_right_wheel"]
        wheel = p["wheels"]["rear_right_ankle"]
        key = (bool(contact["ground"]["active"]), bool(contact["obstacle"]["active"]))
        if not contact_segments or tuple(contact_segments[-1]["ground_obstacle"]) != key:
            contact_segments.append({"first_tick": tick, "last_tick": tick,
                                     "ground_obstacle": key,
                                     "first_bottom_z_m": wheel["bottom_w_m"][2]})
        segment = contact_segments[-1]
        segment["last_tick"] = tick
        segment["last_bottom_z_m"] = wheel["bottom_w_m"][2]
        vz = None
        if previous is not None:
            dt = p["simulation_time_s"] - previous["simulation_time_s"]
            if dt > 0:
                vz = (wheel["bottom_w_m"][2] -
                      previous["wheels"]["rear_right_ankle"]["bottom_w_m"][2]) / dt
        if tick in ticks:
            physical[tick] = (p, vz)
        previous = p
    native = {row["episode_physics_tick"]: row for row in indexed_window(
        source / "native_tick_audit.jsonl", start, end,
        "episode_physics_tick", 1, ticks)}
    snapshots, waits, seen_waits = [], Counter(), set()
    v3_fields_seen = False
    names = ("front_left", "front_right", "rear_left", "rear_right")
    for d in decisions:
        tick = d["end_tick"]
        info = d["step_info"]
        task = info["semantic_task"]
        ev = task["physical_evaluator"]
        rr = ev["current_legs"]["RR"]
        v3_fields_seen |= "unsupported_free_lift_m" in rr
        p, vz = physical[tick]
        n = native[tick]
        audit = n["native_audit"]
        layers = task.get("nominal_provider_diagnostics", {}).get(
            "source_partial_order", {}).get("layers", [])
        pending = []
        for layer in layers:
            entry = layer.get("current_free_lift_source_readiness")
            if entry is None:
                continue
            entry = dict(entry)
            entry["decision_endpoint_age_ticks"] = (
                tick - entry["observation_tick"] if isinstance(entry.get("observation_tick"), int)
                else None)
            # Dispatch observes a preceding physics tick. Preserve age, don't demand end_tick.
            observation_tick = entry.get("observation_tick")
            fresh = (isinstance(observation_tick, int) and
                     d["start_tick"] <= observation_tick <= tick and
                     entry.get("current_evidence_fresh") is True)
            entry["recorded_within_this_decision_interval"] = fresh
            pending.append(entry)
            key = (observation_tick, entry.get("pending_kind"), entry.get("reason"))
            if fresh and entry.get("ready") is False and key not in seen_waits:
                waits[str(entry.get("reason"))] += 1
                seen_waits.add(key)
        roles = ev.get("transfer_roles", {})
        snapshots.append({
            "tick": tick, "time_s": p["simulation_time_s"],
            "request_phase": d.get("request_phase"), "stage_id": task.get("stage_id"),
            "RR_recorded_current": rr,
            "RR_recorded_history": {k: history_map.get("RR") for k, history_map in
                ev.get("history", {}).items() if isinstance(history_map, dict) and "RR" in history_map},
            "other_legs_recorded_current": {leg: ev["current_legs"].get(leg) for leg in ("FL", "FR", "RL")},
            "RR_transfer_context": roles.get("RR", {}).get("transfer_direction_context"),
            "source_readiness": pending,
            "base": p["base"], "center_of_mass": p["center_of_mass"],
            "RR_contacts_raw": p["contacts"]["rear_right_wheel"],
            "FL_contacts_raw": p["contacts"]["front_left_wheel"],
            "RR_wheel_raw": p["wheels"]["rear_right_ankle"],
            "RR_adjacent_measured_bottom_vz_m_s": vz,
            "RR_hip_raw": p["joints"]["rear_right_hip"],
            "RR_knee_raw": p["joints"]["rear_right_knee"],
            "N_full12": n.get("nominal_full12"),
            "raw_full12": audit.get("raw_policy_action_full12"),
            "residual_mask_full12": audit.get("phase_mask_full12"),
            "projected_residual_full12": n.get("projected_residual_full12"),
            "mapped_N_full12": audit.get("native_drive_target_full12"),
            "final_canonical_full12": [p["joints"][name + "_" + joint]["command_deg"]
                for name in names for joint in ("hip", "knee")] +
                [p["wheels"][name + "_ankle"]["command_rad_s"] for name in names],
            "actual_fourwheel_qd_rad_s": [p["wheels"][name + "_ankle"]["velocity_rad_s"] for name in names],
            "dispatch_verified": audit.get("verified")})
    if not v3_fields_seen and not args.allow_legacy:
        raise ValueError("No v3 free-lift diagnostics; legacy run needs explicit --allow-legacy")
    completed = final_task.get("completed_stage_ids", [])
    first_uncredited = next((f"P{i:02}" for i in range(1, 14)
                             if f"P{i:02}" not in completed), None)
    result = {
        "schema": "wlr50_clean.sealed_RR_v3_window_summary.v1",
        "source": str(source), "run_lifecycle": run["lifecycle"],
        "sealed_at": run["completed_at_utc"], "bounds": [start, end],
        "runtime_contract": manifest.get("runtime_contract"),
        "checkpoint_load_provenance": manifest.get("checkpoint_load_provenance"),
        "recorded_mode": final_task.get("p09_lift_semantics"),
        "no_simulation_policy_forward_GAE_or_relabeling": True,
        "original_physical_task_success": manifest.get("physical_task_success"),
        "final": {"stage_id": last.get("request_phase"), "completed_stage_ids": completed,
            "stage_credit_snapshot_tick": last_returned["end_tick"],
            "final_endpoint_tick": last["end_tick"],
            "first_incomplete_task": None if manifest.get("physical_task_success") is True
                else last.get("request_phase"),
            "first_uncredited_stage_by_original_credits": first_uncredited,
            "latest_returned_task_success": final_task.get("success"),
            "termination_reason": last.get("stop_reason") or last.get("step_info", {}).get("termination_reason"),
            "physical_termination_reason": evaluator.get("termination_reason"),
            "physical_reason": evaluator.get("reason"),
            "RR_current": evaluator.get("current_legs", {}).get("RR"),
            "event_ticks": history.get("event_ticks"),
            "RR_attempt_events": [e for e in history.get("lift_attempt_events", []) if e.get("leg") == "RR"]},
        "raw_contact_segments_all_physics_ticks": contact_segments,
        "source_wait_counts_saved_decision_snapshots_only": dict(waits),
        "sampling_caveat": "Free/Q/source snapshots are original decision endpoints, not every physics tick. Wait counts are not seconds; raw contacts cover every requested physics tick. Missing load/CoM/diagnostic values remain unknown. No new success label is computed.",
        "decision_snapshot_count": len(snapshots), "snapshots": snapshots}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return {"output": str(args.output.resolve()), "snapshots": len(snapshots),
            "physical_ticks": end - start + 1,
            "endpoint_tick": terminal_tick, "stage_id": result["final"]["stage_id"],
            "termination_reason": result["final"]["termination_reason"],
            "first_incomplete_task": result["final"]["first_incomplete_task"],
            "original_physical_task_success": result["original_physical_task_success"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--start-tick", type=int, required=True)
    parser.add_argument("--end-tick", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-legacy", action="store_true",
                        help="Read preserved old inputs without v3 diagnostics; never relabel")
    try:
        print(json.dumps(analyze(parser.parse_args()), ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f"Refused: {exc}\n")

"""One-pass, stdlib-only progress snapshot of the latest runs in this namespace.

Reads only small manifests/sidecars and the policy, update, and terminal JSONL
streams of the selected latest run per kind. No Torch, production imports,
checkpoint hashing/loading, teacher-prefix scans, physical/raw scans, or CSV.
Each stream stops at its initial byte size; an incomplete final line is deferred.
Logged samples are a lower bound while writers buffer or a prefix is running.
Use --write for a new timestamped JSON snapshot; existing reports never change.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[1]
RUNS = PROJECT / "runs/ppo_fsm_reference_p09_stable_v2"
PHASES = tuple(f"P{i:02d}" for i in range(1, 14))
KINDS = ("train", "validation", "diagnostics", "locked_test")
BASE = {"global_policy_decisions": 140544, "ppo_updates": 1063, "optimizer_steps": 21260}
RR_FLAGS = ("initial_lift_observed", "lift_established", "current_lift_valid",
            "motion_continuation_allowed", "front_edge_crossed", "placed_on_top",
            "air", "ground_contact", "top_contact", "bearing_verified", "load_fraction_valid")
RR_NUMBERS = ("ground_relative_lift_m", "clearance_m", "front_distance_m",
              "recent_wheel_displacement_m", "air_duration_s", "edge_contact_duration_s")
QUALITY = ("sampled", "physics_ticks", "duration_s", "roll_rms_rad", "pitch_rms_rad",
           "roll_peak_rad", "pitch_peak_rad", "roll_rate_rms_rad_s", "pitch_rate_rms_rad_s",
           "angular_acceleration_rms_rad_s2", "applied_servo_rate_rms_deg_s",
           "applied_wheel_rate_rms_rad_s2", "residual_servo_rms_deg", "residual_wheel_rms_rad_s")
EVENTS = {"whole_body_initial_clearance": "I", "qualified_measured_upward_lift": "Q",
          "qualification_revoked_ground_before_cross": "Q_revoked"}
MAX_MANIFEST = 8 * 1024 * 1024
MAX_LINE = 8 * 1024 * 1024
MAX_STREAM = 1024 * 1024 * 1024


def mapping(value):
    return value if isinstance(value, dict) else {}


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value):
    return type(value) is int and value >= 0


def pick(value, keys):
    value = mapping(value)
    return {key: value.get(key) for key in keys}


def reject_constant(value):
    raise ValueError(f"invalid nonfinite JSON constant: {value}")


def decode(raw):
    value = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


class Reads:
    def __init__(self):
        self.receipts = []
        self.warnings = []

    def small(self, path):
        if not path.is_file():
            return None
        with path.open("rb") as stream:
            raw = stream.read(MAX_MANIFEST + 1)
        if len(raw) > MAX_MANIFEST:
            raise ValueError(f"oversized manifest refused: {path}")
        self.receipts.append({"path": str(path), "bytes_read": len(raw), "scope": "small_json"})
        try:
            return decode(raw)
        except (ValueError, UnicodeError) as exc:
            self.warnings.append(f"Manifest unavailable, possibly being written: {path.name}: {exc}")
            return None

    def rows(self, path):
        if not path.is_file():
            return
        with path.open("rb") as stream:
            initial = path.stat().st_size
            if initial > MAX_STREAM:
                raise ValueError(f"latest-run stream exceeds 1 GiB bound: {path}")
            consumed, count, deferred = 0, 0, 0
            while consumed < initial:
                remaining = initial - consumed
                raw = stream.readline(min(MAX_LINE + 1, remaining))
                if not raw:
                    break
                consumed += len(raw)
                if len(raw) > MAX_LINE:
                    raise ValueError(f"oversized JSONL row refused: {path}")
                if not raw.endswith(b"\n"):
                    deferred = len(raw)
                    break
                if not raw.strip():
                    continue
                count += 1
                yield count, decode(raw)
        self.receipts.append({"path": str(path), "scope": "one_stream_pass_at_initial_byte_boundary",
            "initial_size_bytes": initial, "bytes_read": consumed, "complete_rows": count,
            "deferred_partial_tail_bytes": deferred, "size_after_bytes": path.stat().st_size})


def history(audit):
    task = mapping(audit.get("semantic_task"))
    physical = mapping(task.get("physical_evaluator"))
    return physical, mapping(physical.get("history", task.get("history")))


def physical_summary(audit):
    physical, hist = history(audit)
    rr = mapping(mapping(physical.get("current_legs")).get("RR"))
    placement = mapping(hist.get("placed"))
    first_unplaced = next((leg for leg in ("FR", "FL", "RR", "RL") if placement.get(leg) is not True), None)
    return {"phase_id": audit.get("phase_id"), "end_phase_id": audit.get("end_phase_id"),
        "physics_tick": audit.get("physics_tick"), "sim_time_s": audit.get("sim_time_s"),
        "termination_reason": audit.get("termination_reason"),
        "task_success": audit.get("task_success"), "full_task_success": audit.get("full_task_success"),
        "task_outcome_label": audit.get("task_outcome_label"),
        "first_unfinished_phase": None if audit.get("task_success") is True else audit.get("end_phase_id", audit.get("phase_id")),
        "first_unplaced_leg": first_unplaced if placement else None,
        "first_unplaced_leg_scope": "recorded placement history only; not an inferred stage predicate",
        "physical": pick(physical, ("valid", "success", "run_validity", "physical_evidence_status",
            "termination_reason", "termination_source", "reason", "final_region_valid", "final_controlled",
            "post_completion_elapsed_s", "post_completion_observation_complete")),
        "RR": pick(rr, (*RR_FLAGS, *RR_NUMBERS, "contact_mode", "contact_surface", "motion_continuation_reason")),
        "history_RR": {name: mapping(hist.get(name)).get("RR") for name in ("active_lift", "front_edge_crossed", "placed")}}


def checkpoint_receipt(reader, checkpoint, *, required_root=None):
    if not checkpoint:
        return None
    path = Path(checkpoint).resolve()
    if required_root is None:
        required_root = PROJECT / "outputs"
    if required_root.resolve() not in path.parents:
        raise ValueError(f"checkpoint reference outside permitted outputs: {path}")
    sidecar = path.with_name(path.stem + "_manifest.json")
    data = reader.small(sidecar)
    if data is None:
        return {"checkpoint": str(path), "status": "SIDECAR_UNAVAILABLE"}
    return {"checkpoint": str(path), "manifest": str(sidecar),
        "checkpoint_file_exists": path.is_file(),
        "metadata_path_binding_matches": data.get("checkpoint_path") == str(path),
        **pick(data, (*BASE, "checkpoint_sha256", "save_load_round_trip", "optimizer_learning_rate")),
        "stage_requested_decisions": data.get("stage_requested_decisions"),
        "verification_scope": "saved manifest claims; checkpoint bytes are not loaded or rehashed by this inspector"}


def inspect_run(path, kind, reader):
    start = mapping(reader.small(path / "run_manifest.started.json"))
    final = mapping(reader.small(path / "run_manifest.json"))
    training = mapping(reader.small(path / "training_manifest.json")) if kind == "train" else {}
    evaluation = mapping(reader.small(path / "evaluation_manifest.json")) if kind != "train" else {}
    failure = mapping(reader.small(path / "training_failure.json"))
    args = mapping(start.get("arguments", final.get("arguments")))
    source = checkpoint_receipt(reader, args.get("checkpoint"))
    samples, physics_ticks, terminal_rows, bad_prefix_rows, last_global = 0, 0, 0, 0, None
    counts, modes, present, truth, event_counts, inherited_events = (Counter() for _ in range(6))
    maxima, native, writes = {}, Counter(), Counter()
    seen_events, errors, last = set(), [], None
    terminal_details, episode_index = deque(maxlen=12), 0
    seen_nonzero_raw = seen_nonzero_applied = 0
    for _, row in reader.rows(path / "residual_and_projection_audit.jsonl"):
        if kind == "train":
            if not isinstance(row.get("applied_audit"), dict) or not integer(row.get("global_policy_decision")):
                raise ValueError(f"training audit is not a sampled-policy wrapper: {path}")
            audit, global_step = row["applied_audit"], row["global_policy_decision"]
            if last_global is not None and global_step != last_global + 1:
                errors.append(f"noncontiguous logged global decision: {last_global} -> {global_step}")
            last_global = global_step
            done = row.get("terminal") is True
        else:
            audit = row
            done = audit.get("termination_reason") is not None
        samples += 1
        phase = audit.get("phase_id")
        counts[phase if phase in PHASES else "UNKNOWN"] += 1
        ticks = audit.get("physics_ticks")
        if integer(ticks): physics_ticks += ticks
        else: errors.append(f"missing physics_ticks at logged sample {samples}")
        if audit.get("prefix_teacher_data_in_ppo_storage") is True:
            bad_prefix_rows += 1
        for key, value in audit.items():
            if key.startswith("in_episode_") and key.endswith("_writes") and integer(value):
                writes[key] += value
        for key, field in (("raw", "raw_policy_action_full12"), ("applied", "projected_residual_full12")):
            values = audit.get(field)
            if isinstance(values, list) and len(values) == 12 and all(number(v) for v in values):
                native[f"{key}_residual_rows_available"] += 1
                if any(abs(v) > 0 for v in values):
                    if key == "raw": seen_nonzero_raw += 1
                    else: seen_nonzero_applied += 1
        summary = mapping(audit.get("actuator_target_effect_audit_summary"))
        for key in ("verified_tick_count", "actual_native_effect_tick_count", "own_phase_request_effect_tick_count"):
            if integer(summary.get(key)):
                native[key] += summary[key]
                native[f"{key}_rows_available"] += 1
        effect = mapping(audit.get("actuator_target_effect_audit"))
        if effect.get("verified") is True: native["last_tick_verified_rows"] += 1
        physical, hist = history(audit)
        rr = mapping(mapping(physical.get("current_legs")).get("RR"))
        for key in RR_FLAGS:
            if type(rr.get(key)) is bool:
                present[key] += 1
                truth[key] += int(rr[key])
        mode = rr.get("contact_mode")
        modes[str(mode) if mode is not None else "UNAVAILABLE"] += 1
        for key in RR_NUMBERS:
            if number(rr.get(key)):
                maxima[key] = max(maxima.get(key, rr[key]), rr[key])
        for key in ("active_lift", "front_edge_crossed", "placed"):
            value = mapping(hist.get(key)).get("RR")
            if type(value) is bool:
                present[f"history_{key}"] += 1
                truth[f"history_{key}"] += int(value)
        prefix_tick = mapping(audit.get("curriculum_start")).get("physics_tick", -1)
        events = [event for event in hist.get("lift_attempt_events", []) if isinstance(event, dict)
                  and event.get("leg") == "RR"]
        for key, label in (("front_edge_crossed", "C"), ("placed", "P")):
            tick = mapping(mapping(hist.get("event_ticks")).get(key)).get("RR")
            if integer(tick): events.append({"event": label, "physics_tick": tick})
        for event in events:
            tick, name = event.get("physics_tick"), event.get("event")
            label = EVENTS.get(name, name)
            identity = (episode_index, tick, label)
            if integer(tick) and identity not in seen_events:
                seen_events.add(identity)
                target = inherited_events if integer(prefix_tick) and tick <= prefix_tick else event_counts
                target[label] += 1
        last = physical_summary(audit)
        if done:
            terminal_rows += 1
            terminal_details.append({"episode_index": episode_index, "global_policy_decision": last_global, **last})
            episode_index += 1
    update_rows, opt_steps, update_last = 0, 0, None
    for _, row in reader.rows(path / "optimizer_updates.jsonl"):
        if not all(integer(row.get(key)) for key in ("ppo_update", "global_policy_decisions", "optimizer_steps")):
            raise ValueError(f"invalid completed optimizer-update row: {path}")
        update_rows += 1
        opt_steps += row["optimizer_steps"]
        update_last = pick(row, ("ppo_update", "global_policy_decisions", "optimizer_steps",
            "finite_nonzero_gradient_observed", "actor_parameter_sha256_before", "actor_parameter_sha256_after"))
    completed_rows, completed_success, completed_reasons = 0, 0, Counter()
    last_episode = None
    for _, row in reader.rows(path / "completed_episodes.jsonl"):
        completed_rows += 1
        completed_success += int(row.get("full_task_success") is True)
        completed_reasons[str(row.get("termination_reason"))] += 1
        last_episode = {**pick(row, ("episode_index", "seed", "policy_decisions", "termination_reason",
            "task_success", "full_task_success", "task_outcome_label", "duration_s")),
            "terminal": physical_summary(mapping(row.get("terminal_info")))}
    return {"run_dir": str(path), "kind": kind, "lifecycle": final.get("lifecycle", start.get("lifecycle", "MANIFEST_NOT_YET_AVAILABLE")),
        "arguments": pick(args, ("from_phase", "stage", "seed", "decisions", "mode", "new_mdp_warm_start", "expected_head")),
        "source_checkpoint": source,
        "on_policy_logged_samples_lower_bound": samples if kind == "train" else 0,
        "evaluation_logged_decisions": samples if kind != "train" else 0,
        "logged_physics_ticks": physics_ticks, "phase_samples": {p: counts[p] for p in PHASES},
        "unknown_phase_samples": counts["UNKNOWN"], "last_logged_global_policy_decision": last_global,
        "completed_ppo_updates_logged": update_rows, "optimizer_steps_logged": opt_steps,
        "optimized_decisions_from_complete_128_rollouts": update_rows * 128,
        "last_complete_update": update_last,
        "final_training_manifest_counts": pick(training, ("actual_policy_decisions", "global_policy_decisions", "ppo_updates_this_run", "optimizer_steps_this_run")),
        "terminal_audit_rows": terminal_rows, "completed_episode_rows": completed_rows,
        "completed_full_task_success_rows": completed_success, "completed_termination_reasons": dict(completed_reasons),
        "last_completed_episode": last_episode, "recent_terminal_evidence": list(terminal_details), "last_logged_physical_state": last,
        "RR": {"true_sample_counts": {k: truth[k] for k in present}, "available_sample_counts": dict(present),
            "contact_mode_sample_counts": dict(modes), "decision_end_maxima": maxima,
            "new_event_counts_after_credit_start": dict(event_counts), "inherited_prefix_event_counts": dict(inherited_events),
            "durations_are_diagnostic_only": True},
        "action_effect": {**dict(native), "raw_nonzero_rows": seen_nonzero_raw,
            "applied_residual_nonzero_rows": seen_nonzero_applied, "recorded_state_write_totals": dict(writes)},
        "prefix_teacher_rows_wrongly_in_storage": bad_prefix_rows,
        "evaluation_result": pick(evaluation, ("mode", "seed", "from_phase", "policy_decisions", "duration_s",
            "task_success", "termination_reason", "window_ended_before_task_terminal", "optimizer_updates_during_evaluation",
            "deterministic_policy")),
        "evaluation_quality": {"global": pick(mapping(evaluation.get("quality_metrics")).get("global"), QUALITY),
            "score_is_not_success": mapping(evaluation.get("quality_metrics")).get("score_is_not_success")},
        "failure": pick(failure, ("error_type", "error", "completed_updates_this_run", "partial_rollout_or_update_is_not_resumable")),
        "evidence_errors": errors[:12]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", action="append", choices=KINDS, help="Default: latest run from each kind")
    parser.add_argument("--run-dir", type=Path, help="Inspect only one explicit run in this namespace")
    parser.add_argument("--write", action="store_true", help="Also save an exclusive timestamped report snapshot")
    args = parser.parse_args(argv)
    reader = Reads()
    selected = []
    if args.run_dir:
        path = args.run_dir.resolve()
        if path.parent.parent != RUNS or path.parent.name not in KINDS:
            raise ValueError("explicit run must be directly within an allowed kind of this namespace")
        selected.append((path.parent.name, path))
    else:
        for kind in args.kind or KINDS:
            directory = RUNS / kind
            candidates = [p for p in directory.iterdir() if p.is_dir() and not p.name.endswith("_launcher")] if directory.is_dir() else []
            if candidates:
                selected.append((kind, max(candidates, key=lambda p: p.name)))
    runs = [inspect_run(path, kind, reader) for kind, path in selected]
    pointer = reader.small(OUTPUT / "checkpoints/checkpoint_last_pointer.json")
    saved = checkpoint_receipt(reader, mapping(pointer).get("checkpoint"), required_root=OUTPUT / "checkpoints")
    snapshot = {"schema": "ppo_fsm_reference_p09_stable_v2.live_progress_snapshot.v1",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(), "namespace": str(OUTPUT),
        "status": "NO_RUNS_YET" if not runs else "RECORDED_PROGRESS_SNAPSHOT",
        "verified_resume_baseline_from_parent_receipt": BASE,
        "excluded_old_unsaved_update": {"global_policy_decisions": 140672, "ppo_update": 1064,
            "resumable_credit": False, "reason": "old interrupted run; never use its unsaved update as the resume source"},
        "latest_saved_checkpoint": saved, "selected_latest_runs": runs,
        "new_saved_counts_relative_to_baseline": {key: saved[key] - value if saved and integer(saved.get(key)) else None for key, value in BASE.items()},
        "warnings": reader.warnings, "read_receipts": reader.receipts,
        "limits": ["A sampled/logged terminal before next-prefix reset is not necessarily in an optimized or saved rollout.",
            "Policy, update, episode, and pointer files are read at separate byte boundaries, not an atomic global snapshot.",
            "No teacher-prefix stream contributes policy credit. Only latest selected runs are counted; no cumulative history scan.",
            "Decision-end extrema do not certify all120Hz physical states or center-of-mass/contact causality.",
            "Native-effect counters and checkpoint roundtrip flags are recorded evidence, not independently rerun tests.",
            "Missing measurements remain null or have zero available rows; missing does not mean AIR, zero load, or success.",
            "No task verdict is changed; suffix progress and external diagnostic cutoffs are not full PPO success."]}
    if args.write:
        target = OUTPUT / "progress_snapshots" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot["snapshot_path"] = str(target)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(snapshot, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    json.dump(snapshot, sys.stdout, ensure_ascii=False, indent=2, allow_nan=False)
    print()
    return snapshot


if __name__ == "__main__":
    main()

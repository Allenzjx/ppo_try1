"""Bounded event windows from ONE finalized all-stage N1 TRAIN policy-audit pass.

Stdlib/report only: no production imports, tensors, hashing, CSV, raw trajectory,
prefix-stream reads, simulation, polling, or mutation of the input run. The two
small final manifests must agree before the audit is opened. Original recorded
values/clock names are retained; missing optional evidence is UNAVAILABLE.

Usage: python -B diagnose_completed_rear_windows.py --run-dir <final TRAIN run>
       --output-dir <fresh directory under this experiment's outputs>
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
TRAIN_ROOT = PROJECT / "runs/ppo_all_stage_acceptance_v1/train"
UNAVAILABLE = "UNAVAILABLE"
FINAL = {"SUCCEEDED", "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY"}
PHASES = tuple(f"P{i:02d}" for i in range(1, 14))
REAR = ("RR", "RL")
EVENT_NAMES = {
    "whole_body_initial_clearance": "I",
    "qualified_measured_upward_lift": "Q",
    "qualification_revoked_ground_before_cross": "Q_revoked",
    "active_lift": "Q", "front_edge_crossed": "C", "placed": "P",
}
WRITES = (
    "in_episode_root_pose_writes", "in_episode_root_velocity_writes",
    "in_episode_force_or_impulse_writes", "in_episode_gravity_writes",
)
OBSERVATION_KEYS = (
    "encoded_observation", "encoded_observation_groups", "observation",
    "observations", "final_observation", "final_obs", "raw_observation",
    "moments", "observation_moments", "rpy", "chassis_rpy_rad",
    "projected_gravity_chassis", "gravity_vector", "base_pose",
)
LEG_FIELDS = (
    "front_distance_m", "clearance_m", "within_top_xy", "within_lateral_span",
    "top_geometry", "top_contact", "ground_contact", "air", "support",
    "contact_surface", "top_surface_contact", "contact_reaction",
    "contact_reaction_force_n", "bearing_force_n", "bearing_verified",
    "load_fraction", "load_fraction_valid", "obstacle_pair_active",
    "initial_clearance", "active_attempt", "crossing_evidence_status",
    "crossing_geometry_pending", "consecutive_air_samples", "consecutive_top_samples",
    "recent_joint_motion_deg", "recent_whole_body_joint_motion_deg",
    "recent_clearance_gain_m", "whole_body_actuation_evidence",
    "soft_air_actuation_earned", "soft_air_earned_tick", "top_xy_outside_distance_m",
)
NATIVE_FIELDS = (
    "schema", "verified", "source_phase_id", "policy_request_phase", "physics_tick",
    "canonical_order", "projected_residual_full12", "actual_native_targets",
    "counterfactual_native_targets", "native_target_delta", "target_dtype",
    "actual_mapping_matches_dispatch", "setter_dispatch_targets_equal",
    "same_tick_counterfactual", "counterfactual_scope", "actual_target_source",
    "previous_final_drive_servo_deg", "native_drive_target_full12",
    "controller_drive_bias_full12", "combined_post_mapper_bias_full12",
    "geometry_adjusted_native_full12", "nominal_geometry_adjustment_full12",
    "nominal_geometry_native_target_delta", "nominal_geometry_counterfactual_scope",
)
HEADROOM_FIELDS = (
    "schema", "mode", "servo_reserve_deg", "servo_hard_limits_deg",
    "baseline_native_plus_controller_full12", "requested_policy_residual_full12",
    "effective_policy_residual_full12", "effective_combined_post_mapper_bias_full12",
    "policy_residual_intervals_servo_deg", "clipped_servo_indices",
    "baseline_outside_reserved_servo_indices", "candidate_native_target_before_final_slew_full12",
    "candidate_target_semantics", "wheel_residual_unchanged",
)
TRACKING_FIELDS = (
    "schema", "source_phase_id", "source_control_tick", "dispatch_physics_tick",
    "physics_tick", "feedback_tick", "previous_request_source",
    "previous_requested_full12", "actual_measured_physical_rad", "standing_pose_deg",
    "mapper_computational_reference_rad", "requested_command_deg", "channels",
    "reference_reads_current_policy_residual", "reference_semantics",
)
PHYSICAL_FIELDS = (
    "valid", "success", "termination_reason", "reason", "termination_source",
    "source", "physics_tick", "simulation_time_s", "evaluator_version",
    "run_validity", "physical_evidence_status", "body_traversal_geometry",
    "measured_wheel_velocity_rad_s", "applied_wheel_command_rad_s", "stop_progress_wheel_order",
    "final_region_valid", "final_controlled", "final_support_available", "final_stable_for_s",
    "traversal_event_observed", "task_completed_controlled", "strict_recovery_quality",
    "post_completion_elapsed_s", "post_completion_loss_observed",
)
LIMITS = [
    "Only current PPO credited rows/events. Inherited pre-credit events are counted separately, not learned achievements.",
    "Each event window is centered on its FIRST OBSERVED decision row, not an invented exact event-tick state.",
    "Geometry/controls are decision-end samples; reward covers that action interval; policy Gaussian moments/raw are pre-action.",
    "Native command physics_tick and episode physics_tick are preserved separately; no assumed equality or offset conversion.",
    "Extrema are over credited decision-end samples only, not all 120 Hz states. No minimum-gap/contact claims between samples.",
    "Native flags are recorded evidence, not independently re-executed validation; no full raw/native trajectory is read.",
    "old_distribution_mean/std are policy Gaussian moments, NOT encoded observations or body-pose measurements.",
    "Scalar body speeds, body_forward_m and world AABB do not determine RPY/gravity, signed velocity, or center of mass.",
    "No encoded-observation inversion. Even if optional observation-like fields appear, their schema is not assumed or decoded.",
    "No new success, safety reclassification, dynamics/causal proof, optimizer gate, or parameter recommendation.",
]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def integer(value, label):
    need(type(value) is int and value >= 0, f"{label}: nonnegative integer required")
    return value


def inside(path, root):
    resolved = Path(path).resolve()
    need(root.resolve() in resolved.parents, f"path outside required subtree: {resolved}")
    return resolved


def reject_constant(value):
    raise ValueError(f"nonfinite JSON token: {value}")


def finite_float(value):
    result = float(value)
    need(math.isfinite(result), "nonfinite/overflowed JSON float")
    return result


def decode(text):
    result = json.loads(text, parse_constant=reject_constant, parse_float=finite_float)
    need(isinstance(result, dict), "JSON object required")
    return result


def small_json(path):
    need(path.is_file() and path.stat().st_size <= 8*1024*1024, f"missing/oversized final manifest: {path}")
    return decode(path.read_text(encoding="utf-8-sig"))


def mapping(value):
    return value if isinstance(value, dict) else {}


def pick(value, keys):
    value = mapping(value)
    return {key: value[key] if key in value else UNAVAILABLE for key in keys}


def history_of(audit):
    task = mapping(audit.get("semantic_task"))
    physical = mapping(task.get("physical_evaluator"))
    if isinstance(physical.get("history"), dict):
        return physical["history"], "applied_audit.semantic_task.physical_evaluator.history"
    return mapping(task.get("history")), "applied_audit.semantic_task.history"


def compact_row(row, line_number):
    audit = row["applied_audit"]
    task = mapping(audit.get("semantic_task"))
    physical = mapping(task.get("physical_evaluator"))
    native = mapping(audit.get("actuator_target_effect_audit"))
    hist, history_path = history_of(audit)
    physical_record = pick(physical, PHYSICAL_FIELDS)
    physical_record["goal_features"] = pick(physical.get("goal_features"), (
        "support_count", "body_forward_m", "body_linear_speed_m_s",
        "body_angular_speed_rad_s", "maximum_wheel_speed_rad_s"))
    physical_record["current_legs"] = {leg: pick(mapping(physical.get("current_legs")).get(leg), LEG_FIELDS)
                                        for leg in ("FL", "FR", "RL", "RR")}
    native_record = pick(native, NATIVE_FIELDS)
    native_record["policy_headroom_evidence"] = pick(native.get("policy_headroom_evidence"), HEADROOM_FIELDS)
    native_record["tracking_reference_evidence"] = pick(native.get("tracking_reference_evidence"), TRACKING_FIELDS)
    native_record["nominal_geometry_evidence"] = pick(native.get("nominal_geometry_evidence"), (
        "schema", "mode", "status", "context", "projection", "old_zero_policy_physical_target_rad",
        "desired_zero_policy_canonical_target_deg", "nominal_geometry_adjustment_full12",
        "current_policy_residual_used", "physical_motion_guaranteed"))
    return {
        "source_classification": "current_PPO_policy", "audit_line_1based": line_number,
        **pick(row, ("global_policy_decision", "raw_policy_action_full12", "old_distribution_mean_full12",
                     "old_distribution_std_full12", "old_log_probability", "old_value", "reward", "terminal")),
        "applied_audit": {
            **pick(audit, ("schema", "phase_id", "end_phase_id", "physics_tick", "physics_ticks", "sim_time_s",
                          "decision_count", "physical_core_decision_count_including_prefix", "curriculum_start",
                          "task_result_scope", "prefix_teacher_data_in_ppo_storage", "raw_policy_action_full12", "nominal_action_full12",
                          "projected_residual_full12", "applied_action_full12", "actual_drive_target_full12",
                          "actuator_target_effect_audit_summary", "actuator_target_effect_audit_ticks",
                          "termination_reason", "task_success", "full_task_success", "task_outcome_label",
                          "time_outs", "terminal_bootstrap_allowed", "terminal_observation_finite_fallback",
                          "stage_transition_evidence", "reward_breakdown", "no_in_episode_state_writes_verified", *WRITES)),
            "actuator_target_effect_audit": native_record,
            "semantic_task": {
                **pick(task, ("stage_id", "stage_age_s", "stage_elapsed_s", "phase_progress", "goal_features",
                             "task_progress_potential", "completion_values", "completed_stage_ids", "entry_valid",
                             "entry_reasons", "termination_source", "termination_reason", "physical_transfer_fraction",
                             "transfer_roles_version", "transfer_roles", "transfer_role_context", "pending_capture",
                             "nominal_provider_diagnostics")),
                "history": pick(hist, ("active_lift", "front_edge_crossed", "placed", "event_ticks")),
                "history_source_path": history_path, "physical_evaluator": physical_record,
            },
        },
    }


class Episode:
    def __init__(self, index, row, radius, events_per_kind):
        audit = row["applied_audit"]
        start = mapping(audit.get("curriculum_start"))
        # Ordinary natural-P01 SemanticEpisodeEnv has no prefix facade fields.
        # The explicit tick-zero/P01 check below disambiguates this absence.
        self.scope = audit.get("task_result_scope", "full_task" if not start else UNAVAILABLE)
        mode = start.get("mode")
        need(self.scope in ("full_task", "teacher_initialized_suffix", "checkpoint_policy_initialized_suffix"),
             f"unknown credit scope: {self.scope}")
        if start:
            need(mode in ("teacher_initialized_suffix", "checkpoint_policy_initialized_suffix", "fresh_P01_fallback"),
                 f"unrecognized curriculum_start.mode: {mode}")
            self.credit_tick = integer(start.get("physics_tick"), "credit start physics_tick")
        else:
            need(self.scope == "full_task", "suffix requires recorded credit-start tick")
            self.credit_tick = audit["physics_tick"]-audit["physics_ticks"]
            need(self.credit_tick == 0 and audit["phase_id"] == "P01", "unattributed non-P01 start")
        self.index, self.start, self.first = index, start, row["global_policy_decision"]
        self.radius, self.events_per_kind = radius, events_per_kind
        self.previous = deque(maxlen=radius+1)
        self.windows, self.seen, self.events, self.prefix_events = {}, set(), Counter(), Counter()
        self.extrema, self.ranges = {}, Counter()
        self.phases, self.native, self.observation_keys = Counter(), Counter(), Counter()
        self.decisions = self.ticks = self.handoffs = 0
        self.last_tick = self.credit_tick
        self.last = None

    def window(self, key, center, metadata):
        previous = list(self.previous)[-self.radius:]
        self.windows[key] = {"selection": metadata, "center_global": center["global_policy_decision"],
                             "rows": [*previous, center], "remaining": self.radius,
                             "available_preceding_rows": len(previous)}

    def feed(self, row, line_number):
        audit = row["applied_audit"]
        need(audit.get("curriculum_start", {}) == self.start, "curriculum changed inside credited episode")
        need(audit.get("task_result_scope", "full_task" if not self.start else UNAVAILABLE) == self.scope,
             "credit scope changed inside episode")
        need(audit.get("prefix_teacher_data_in_ppo_storage", UNAVAILABLE if self.start else False) is False,
             "prefix data exclusion is missing or false")
        need(row.get("raw_policy_action_full12") == audit.get("raw_policy_action_full12"), "sample/applied raw binding differs")
        n = integer(audit.get("physics_ticks"), "physics_ticks")
        tick = integer(audit.get("physics_tick"), "episode physics_tick")
        need(1 <= n <= 8 and tick == self.last_tick+n, "credited episode clock gap/reset")
        need(number(audit.get("sim_time_s")) and math.isclose(audit["sim_time_s"], tick/120., abs_tol=1e-8),
             "decision simulation_time/physics_tick disagree")
        need(audit.get("phase_id") in PHASES and audit.get("end_phase_id") in PHASES, "unknown phase")
        need(type(row.get("terminal")) is bool, "explicit terminal flag required")
        need(audit.get("time_outs") is False, "unexpected external/bootstrap timeout")
        need(audit.get("terminal_bootstrap_allowed") is (not row["terminal"]), "terminal/bootstrap mismatch")
        center = compact_row(row, line_number)
        for window in self.windows.values():
            if window["remaining"] > 0:
                window["rows"].append(center)
                window["remaining"] -= 1
        self.decisions += 1
        self.ticks += n
        self.phases[audit["phase_id"]] += 1
        for container_name, container in (("row", row), ("applied_audit", audit),
                                           ("semantic_task", mapping(audit.get("semantic_task")))):
            for key in OBSERVATION_KEYS:
                if key in container:
                    self.observation_keys[f"{container_name}.{key}"] += 1
        native_summary = mapping(audit.get("actuator_target_effect_audit_summary"))
        for key in ("physics_ticks", "verified_tick_count", "actual_native_effect_tick_count", "own_phase_request_effect_tick_count"):
            value = native_summary.get(key)
            if type(value) is int and 0 <= value <= n:
                self.native[key] += value
                self.native[key+"_available_rows"] += 1
            else:
                self.native[key+"_unavailable_rows"] += 1
        self.native["all_ticks_verified_true_rows"] += int(native_summary.get("all_ticks_verified") is True)
        self.native["all_ticks_verified_false_rows"] += int(native_summary.get("all_ticks_verified") is False)
        self.native["all_ticks_verified_unavailable_rows"] += int(type(native_summary.get("all_ticks_verified")) is not bool)
        for key in WRITES:
            value = audit.get(key)
            label = "zero_rows" if type(value) is int and value == 0 else "nonzero_rows" if type(value) is int else "unavailable_rows"
            self.native[key+"_"+label] += 1
        hist, hist_path = history_of(audit)
        events = []
        for item in hist.get("lift_attempt_events", []):
            if isinstance(item, dict) and item.get("leg") in REAR and item.get("event") in EVENT_NAMES:
                events.append((item["leg"], EVENT_NAMES[item["event"]], item.get("physics_tick"),
                               hist_path+".lift_attempt_events", item))
        for name, values in mapping(hist.get("event_ticks")).items():
            if name in EVENT_NAMES:
                for leg, event_tick in mapping(values).items():
                    if leg in REAR:
                        events.append((leg, EVENT_NAMES[name], event_tick, hist_path+".event_ticks."+name,
                                       {"leg": leg, "event": name, "physics_tick": event_tick}))
        for leg, kind, event_tick, path, raw_event in events:
            event_tick = integer(event_tick, "event physics_tick")
            need(event_tick <= tick, "history event occurs after observed row")
            key = leg, kind, event_tick
            if key in self.seen:
                continue
            self.seen.add(key)
            if event_tick <= self.credit_tick:
                self.prefix_events[f"{leg}:{kind}"] += 1
                continue
            category = f"{leg}:{kind}"
            self.events[category] += 1
            occurrence = self.events[category]
            # First N-1 and latest event: I storms cannot consume Q/C/P slots.
            slot = min(occurrence, self.events_per_kind)
            self.window(f"event:{category}:{slot}", center, {
                "type": "rear_event", "leg": leg, "kind": kind,
                "occurrence_1based": occurrence, "source_classification": "current_PPO_policy",
                "physics_tick": event_tick, "first_observed_global": row["global_policy_decision"],
                "source_path": path, "original_event": raw_event,
            })
        if audit["phase_id"] != audit["end_phase_id"]:
            self.handoffs += 1
            need(not row["terminal"], "phase handoff unexpectedly terminal")
            if self.handoffs <= 12:
                self.window(f"handoff:{self.handoffs}", center, {"type": "nonterminal_phase_handoff",
                    "phase_id": audit["phase_id"], "end_phase_id": audit["end_phase_id"],
                    "physics_tick": tick, "stage_transition_evidence": audit.get("stage_transition_evidence", UNAVAILABLE)})
        legs = mapping(mapping(mapping(audit.get("semantic_task")).get("physical_evaluator")).get("current_legs"))
        for leg in REAR:
            for metric in ("front_distance_m", "clearance_m"):
                value = mapping(legs.get(leg)).get(metric)
                label = f"{leg}:{metric}"
                if not number(value):
                    self.ranges[label+":unavailable_rows"] += 1
                    continue
                self.ranges[label+":available_rows"] += 1
                for direction in ("min", "max"):
                    key = f"extreme:{label}:{direction}"
                    old = self.extrema.get(key)
                    if old is None or (value < old if direction == "min" else value > old):
                        self.extrema[key] = value
                        self.window(key, center, {"type": "decision_end_extreme", "leg": leg,
                            "metric": metric, "direction": direction, "value": value, "physics_tick": tick})
        if row["terminal"]:
            self.window("terminal", center, {"type": "task_terminal", "physics_tick": tick,
                "termination_reason": audit.get("termination_reason", UNAVAILABLE)})
        self.previous.append(center)
        self.last, self.last_tick = center, tick

    def finish(self):
        need(self.last is not None, "empty episode")
        if not self.last["terminal"]:
            # Reconstruct preceding deque without double-counting the tail row.
            prior = list(self.previous)
            self.windows["nonterminal_tail"] = {"selection": {"type": "optimized_nonterminal_tail"},
                "center_global": self.last["global_policy_decision"], "rows": prior,
                "remaining": self.radius, "available_preceding_rows": max(0, len(prior)-1)}
        windows, rows = [], {}
        for key, window in sorted(self.windows.items(), key=lambda item: (item[1]["center_global"], item[0])):
            ids = []
            for record in window["rows"]:
                g = record["global_policy_decision"]
                rows[g] = record
                ids.append(g)
            windows.append({"id": key, **window["selection"], "center_global": window["center_global"],
                "selected_global_decisions": ids, "available_preceding_rows": window["available_preceding_rows"],
                "available_following_rows": self.radius-window["remaining"],
                "following_window_clipped_at_episode_or_block_end": window["remaining"] > 0})
        audit = self.last["applied_audit"]
        summary = {
            "episode_index": self.index, "source_classification": "current_PPO_policy",
            "task_result_scope": self.scope, "curriculum_start": self.start,
            "credit_start_episode_physics_tick": self.credit_tick,
            "first_global": self.first, "last_global": self.last["global_policy_decision"],
            "decisions": self.decisions, "physics_ticks": self.ticks,
            "phase_counts": {p: self.phases[p] for p in PHASES}, "nonterminal_handoffs": self.handoffs,
            "handoff_windows_omitted": max(0, self.handoffs-12),
            "terminal": self.last["terminal"], "termination_reason": audit["termination_reason"],
            "task_success": audit["task_success"], "full_task_success": audit["full_task_success"],
            "last_episode_physics_tick": self.last_tick, "last_sim_time_s": audit["sim_time_s"],
            "policy_rear_event_counts": dict(self.events), "inherited_rear_event_counts_excluded": dict(self.prefix_events),
            "event_windows_policy": "First N-1 and latest per leg/kind; counts include omitted intermediate events.",
            "event_windows_omitted": sum(max(0, count-self.events_per_kind) for count in self.events.values()),
            "decision_end_ranges": {f"{leg}:{metric}": {
                "min": self.extrema.get(f"extreme:{leg}:{metric}:min", UNAVAILABLE),
                "max": self.extrema.get(f"extreme:{leg}:{metric}:max", UNAVAILABLE),
                "available_rows": self.ranges[f"{leg}:{metric}:available_rows"],
                "unavailable_rows": self.ranges[f"{leg}:{metric}:unavailable_rows"],
            } for leg in REAR for metric in ("front_distance_m", "clearance_m")},
            "recorded_native_and_write_counts": dict(self.native),
            "optional_observation_like_fields_present_rows": dict(self.observation_keys),
            "rpy_gravity_raw_base_pose": UNAVAILABLE,
            "rpy_gravity_scope": "No verified serialized observation schema decoded; scalar speed/AABB are not raw pose.",
            "window_count": len(windows), "selected_unique_decision_rows": len(rows),
        }
        return {"summary": summary, "windows": windows, "selected_rows": [rows[g] for g in sorted(rows)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--radius", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--events-per-kind", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--max-audit-mib", type=int, default=512)
    args = parser.parse_args()
    run, output = inside(args.run_dir, TRAIN_ROOT), inside(args.output_dir, ROOT)
    need(run.parent == TRAIN_ROOT.resolve() and run.is_dir(), "one direct TRAIN run directory required")
    need(not output.exists(), "exclusive output directory already exists")
    need(1 <= args.max_audit_mib <= 2048, "audit byte budget must be 1..2048 MiB")
    run_manifest = small_json(run/"run_manifest.json")
    manifest = small_json(run/"training_manifest.json")
    need(run_manifest.get("command") == "train" and run_manifest.get("completed_at_utc"), "final TRAIN manifest required")
    need(manifest.get("schema") == "wlr50_clean.semantic_training.v1", "unknown training manifest schema")
    need(run_manifest.get("lifecycle") == manifest.get("lifecycle") and manifest["lifecycle"] in FINAL,
         "run/training manifests are not matching finalized executions")
    need(manifest.get("num_envs") == 1 and manifest.get("semantic_version") == "v3", "only v3 N1 supported")
    runtime = mapping(manifest.get("runtime_contract"))
    need(runtime.get("experiment_id") == "all_stage_acceptance_v1", "wrong experiment runtime")
    need(mapping(run_manifest.get("runtime_contract")) == runtime, "final runtime contracts differ")
    result = mapping(run_manifest.get("result"))
    for key in ("actual_policy_decisions", "global_policy_decisions", "ppo_updates_this_run", "optimizer_steps_this_run"):
        need(result.get(key) == manifest.get(key), f"final result differs: {key}")
    actual = integer(manifest.get("actual_policy_decisions"), "actual decisions")
    end = integer(manifest.get("global_policy_decisions"), "end global")
    need(actual > 0 and actual % 128 == 0 and end >= actual, "not a complete saved N1 rollout boundary")
    source = end-actual
    path = run/"residual_and_projection_audit.jsonl"
    need(path.is_file(), "policy audit missing")
    before = path.stat()
    need(0 < before.st_size <= args.max_audit_mib*1024*1024, "policy audit exceeds explicit byte budget or is empty")
    print(json.dumps({"pre_stream_file": str(path), "bytes": before.st_size, "allowed_bytes": args.max_audit_mib*1024*1024}), flush=True)
    documents, episode, total = [], None, 0
    started = time.perf_counter()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            need(len(line) <= 4*1024*1024 and line.strip(), f"oversized/blank audit line {line_number}")
            row = decode(line)
            need(row.get("global_policy_decision") == source+line_number, "PPO global interval gap/overlap")
            need(isinstance(row.get("applied_audit"), dict), "missing applied audit")
            if episode is None:
                episode = Episode(len(documents), row, args.radius, args.events_per_kind)
            episode.feed(row, line_number)
            total += 1
            need(total <= actual, "audit contains rows beyond published optimized boundary")
            if row["terminal"]:
                documents.append(episode.finish())
                episode = None
    if episode is not None:
        documents.append(episode.finish())
    after = path.stat()
    need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "finalized audit changed during read")
    need(total == actual, "audit count differs from finalized optimized count")
    counts, event_counts, native = Counter(), Counter(), Counter()
    for doc in documents:
        counts.update(doc["summary"]["phase_counts"])
        event_counts.update(doc["summary"]["policy_rear_event_counts"])
        native.update(doc["summary"]["recorded_native_and_write_counts"])
    telemetry = mapping(manifest.get("telemetry"))
    core = mapping(telemetry.get("core"))
    need(core.get("decisions") == total and core.get("physics_ticks") == sum(d["summary"]["physics_ticks"] for d in documents),
         "manifest credited decision/tick totals differ")
    need(all(mapping(core.get("phase_decisions")).get(p, 0) == counts[p] for p in PHASES), "manifest credited phases differ")
    need(telemetry.get("completed_episode_count") == sum(d["summary"]["terminal"] for d in documents), "terminal ledger differs")
    elapsed = time.perf_counter()-started
    physical_core = mapping(core.get("physical_core_including_prefix"))
    prefix_difference = {}
    for key in ("decisions", "physics_ticks"):
        prefix_difference[key] = (physical_core[key]-core[key]
                                  if type(physical_core.get(key)) is int and type(core.get(key)) is int else UNAVAILABLE)
    summary = {
        "schema": "ppo_all_stage_acceptance_v1.completed_rear_windows.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
        "lifecycle": manifest["lifecycle"], "runtime": pick(runtime, ("experiment_id", "source_git_commit", "runtime_content_sha256")),
        "source_global": source, "end_global": end, "actual_policy_decisions": actual,
        "ppo_updates_this_run": manifest["ppo_updates_this_run"], "optimizer_steps_this_run": manifest["optimizer_steps_this_run"],
        "audit": {"path": str(path), "bytes": before.st_size, "full_stream_passes": 1,
                  "rows": total, "analysis_wall_s": elapsed, "hashes_computed": False,
                  "other_large_streams_read": [], "file_metadata_unchanged_during_pass": True},
        "window_bounds": {"radius_decisions": args.radius, "events_per_leg_kind": args.events_per_kind,
                          "maximum_windows_per_episode": 10*args.events_per_kind+12+8+1,
                          "maximum_rows_per_window": 2*args.radius+1},
        "phase_counts": {p: counts[p] for p in PHASES}, "policy_rear_event_counts": dict(event_counts),
        "recorded_native_and_write_counts": dict(native),
        "prefix_excluded_from_policy_rows": True,
        "prefix_counts_from_manifest_physical_minus_credit": prefix_difference,
        "prefix_scope": "Manifest subtraction only, includes any reset/prefix core work; no prefix event/native stream read.",
        "optional_pose_scope": {"encoded_observation": UNAVAILABLE, "raw_base_pose": UNAVAILABLE,
                                "rpy": UNAVAILABLE, "gravity_vector": UNAVAILABLE,
                                "reason": "No observation decoding or reconstruction; inspect per-episode key-presence inventory."},
        "episodes": [doc["summary"] for doc in documents], "limits": LIMITS,
    }
    # All validation finishes before creating an exclusive output directory.
    output.mkdir(parents=True, exist_ok=False)
    for index, doc in enumerate(documents):
        with (output/f"episode_{index:03d}_windows.json").open("x", encoding="utf-8") as stream:
            json.dump(doc, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write("\n")
    with (output/"rear_windows_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    lines = ["# Completed rear-event windows", "", f"Run: `{run.name}`", "",
             f"Actual PPO interval {source+1}–{end}: {actual} decisions; {core['physics_ticks']} physics ticks.",
             f"One audit pass; {before.st_size:,} bytes; {elapsed:.3f}s analysis (not simulation performance).", "",
             "| Episode | Decisions | End phase | Terminal/reason | Windows / selected rows |",
             "|---|---:|---|---|---:|"]
    for doc in documents:
        s = doc["summary"]
        last = doc["selected_rows"][-1]["applied_audit"]
        lines.append(f"| {s['episode_index']} | {s['decisions']} | {last['end_phase_id']} | "
                     f"{s['terminal']} / {s['termination_reason']} | {s['window_count']} / {s['selected_unique_decision_rows']} |")
    lines += ["", "Policy rear events: `"+json.dumps(dict(event_counts), sort_keys=True)+"`.", "", *["- "+x for x in LIMITS], ""]
    with (output/"README.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"output_dir": str(output), "actual_decisions": actual, "physics_ticks": core["physics_ticks"],
                      "episodes": len(documents), "rear_events": dict(event_counts), "stream_passes": 1,
                      "selected_rows": sum(d["summary"]["selected_unique_decision_rows"] for d in documents)}))


if __name__ == "__main__":
    main()

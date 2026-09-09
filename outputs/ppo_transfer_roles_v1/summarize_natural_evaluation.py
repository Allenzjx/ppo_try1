"""Bounded, standard-library-only summary of one completed natural-P01 C eval.

Reads two small manifests and only the first/last decision and physical rows.
Does not load checkpoints, import production modules, rescan trajectories, or
reconstruct simulator state. JSON/Markdown outputs are exclusive, never replaced.

Usage: python summarize_natural_evaluation.py --run-dir <completed_validation_run>
         --output-json <new_experiment_local_path.json>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
EVAL_ROOT = PROJECT / "runs/ppo_transfer_roles_v1/validation"
PHASES = tuple(f"P{i:02}" for i in range(1, 14))
LEGS = ("FL", "FR", "RL", "RR")
WHEELS = dict(zip(LEGS, ("front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle")))
WRITES = ("in_episode_root_pose_writes", "in_episode_root_velocity_writes",
          "in_episode_force_or_impulse_writes", "in_episode_gravity_writes")
METRICS = ("duration_s", "roll_rms_rad", "pitch_rms_rad", "angular_acceleration_rms_rad_s2",
           "applied_servo_rate_rms_deg_s", "applied_wheel_rate_rms_rad_s2",
           "residual_servo_rms_deg", "residual_wheel_rms_rad_s", "quality_score")
LEG_FIELDS = ("front_distance_m", "clearance_m", "top_geometry", "within_top_xy", "within_lateral_span",
              "air", "ground_contact", "obstacle_pair_active", "top_contact", "support", "load_fraction",
              "consecutive_air_samples", "consecutive_top_samples")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, name):
    require(type(value) is int and value >= 0, f"invalid integer: {name}")
    return value


def flag(value, name):
    require(type(value) is bool, f"missing explicit boolean: {name}")
    return value


def close(a, b):
    return (type(a) in (float, int) and type(b) in (float, int)
            and math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8))


def subset(obj, keys):
    return {key: obj.get(key) for key in keys} if isinstance(obj, dict) else None


def termination_classification(formal_reason, physical_reason, *, success, window):
    """Exact semantic_env/TaskResult names; task failure is not safety abort.

    The formal reason is primary (it preserves the environment's signal
    priority); an explicit physical result is a fallback. Unknown names are
    retained by the report and are never inferred to be independent safety.
    """
    if success:
        return "task_success"
    if window:
        return "window_incomplete"
    classes = {
        "BODY_COLLISION": "task_failure", "WHEEL_ONLY_CLIMB": "task_failure",
        "TASK_FAILURE_BODY_COLLISION": "task_failure",
        "TASK_FAILURE_WHEEL_ONLY_CLIMB": "task_failure",
        "FALL": "independent_safety_abort", "NAN_INF": "independent_safety_abort",
        "HARD_JOINT_LIMIT": "independent_safety_abort",
        "PHYSICS_EXPLOSION": "independent_safety_abort", "SAFETY_ABORT": "independent_safety_abort",
        "INCOMPLETE_CONTROLLER_BLOCKED": "incompletion", "TASK_TIMEOUT": "incompletion",
        "TIMEOUT": "incompletion", "INFRASTRUCTURE_ERROR": "infrastructure_error",
        "VIDEO_OR_ARTIFACT_ERROR": "artifact_error",
    }
    for reason in (formal_reason, physical_reason):
        if reason in classes:
            return classes[reason]
    return "unknown_termination" if formal_reason is not None or physical_reason is not None else "incompletion"


def decode(data):
    def bad_constant(value):
        raise ValueError(f"non-standard JSON constant: {value}")
    value = json.loads(data.decode("utf-8-sig"), parse_constant=bad_constant)
    require(isinstance(value, dict), "expected JSON object")
    return value


class Reads:
    def __init__(self):
        self.sources = {}

    def note(self, path, scope):
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns)
        old = self.sources.setdefault(path, {"signature": signature, "scopes": []})
        require(old["signature"] == signature, f"source changed during read: {path.name}")
        old["scopes"].append(scope)

    def small(self, path):
        self.note(path, "complete small JSON")
        require(path.stat().st_size <= 8 * 1024 * 1024, f"manifest exceeds bounded size: {path.name}")
        return decode(path.read_bytes())

    def edge(self, path, last=False):
        self.note(path, "last JSONL row" if last else "first JSONL row")
        maximum = 4 * 1024 * 1024
        with path.open("rb") as stream:
            if not last:
                data = stream.readline(maximum + 1)
                require(len(data) <= maximum and data.strip(), "missing/oversized first row")
                return decode(data)
            position = stream.seek(0, 2)
            data = b""
            while position and len(data) <= maximum:
                length = min(65536, position)
                position -= length
                stream.seek(position)
                data = stream.read(length) + data
                trimmed = data.rstrip(b"\r\n")
                if b"\n" in trimmed or position == 0:
                    row = trimmed.rsplit(b"\n", 1)[-1]
                    require(row and len(row) <= maximum, "missing/oversized last row")
                    return decode(row)
        raise ValueError(f"last row exceeds bounded read: {path.name}")

    def verify_unchanged(self):
        for path, record in self.sources.items():
            stat = path.stat()
            require((stat.st_size, stat.st_mtime_ns) == record["signature"], f"source changed: {path}")

    def receipt(self):
        return [{"path": str(path), "size_bytes": row["signature"][0],
                 "mtime_ns": row["signature"][1], "read_scopes": row["scopes"]}
                for path, row in self.sources.items()]


def quality_row(row, *, phase=False):
    if row is None or row.get("sampled") is False or row.get("physics_ticks") == 0:
        return None
    require(row.get("sampled") is True, "quality availability is not explicit")
    ticks = integer(row.get("physics_ticks"), "quality ticks")
    return {"ticks" if phase else "physics_ticks": ticks, **subset(row, METRICS)}


def summarize(run):
    reads = Reads()
    wrapper = reads.small(run / "run_manifest.json")
    actual = reads.small(run / "evaluation_manifest.json")
    args = wrapper.get("arguments", {})
    require(wrapper.get("command") == "eval" and wrapper.get("lifecycle") == "SUCCEEDED"
            and wrapper.get("completed_at_utc"), "requires completed successful evaluator execution, not a running/failed launcher")
    require(args.get("mode") == actual.get("mode") == "semantic_residual_eval"
            and args.get("from_phase") == actual.get("from_phase") == "P01"
            and type(args.get("num_envs")) is int and args["num_envs"] == 1,
            "requires formal natural-P01 deterministic N1 C evaluation")
    require(actual.get("deterministic_policy") is True and type(actual.get("optimizer_updates_during_evaluation")) is int
            and actual["optimizer_updates_during_evaluation"] == 0, "not deterministic zero-optimizer evaluation")
    require(actual.get("checkpoint") and actual["checkpoint"] == args.get("checkpoint")
            and actual.get("seed") == args.get("seed"), "checkpoint/seed identity mismatch")
    require(actual.get("runtime_contract") == wrapper.get("runtime_contract"), "runtime contract mismatch")
    require(actual["runtime_contract"].get("experiment_id") == "transfer_roles_v1", "wrong experiment runtime")
    require(wrapper.get("result") == actual, "final wrapper result differs from evaluation manifest")
    require(not (run / "prefix_evidence.jsonl").exists(), "prefix evidence is incompatible with this natural-P01 protocol")
    count = integer(actual.get("policy_decisions"), "policy decisions")
    ticks = integer(actual.get("observed_physics_ticks"), "observed physics ticks")
    require(count > 0 and ticks > 0, "empty evaluation has no terminal diagnostic")
    telemetry = actual["telemetry"]
    require(telemetry.get("episodes") == 1 and telemetry.get("decisions") == count
            and telemetry.get("physics_ticks") == ticks, "not a single continuous unprefixed episode")
    for key, value in telemetry.items():
        if key.startswith("prefix"):
            require(value in (None, False, 0, [], {}), "nonzero prefix telemetry")
    phase_counts = telemetry["phase_decisions"]
    require(set(phase_counts) <= set(PHASES), "unknown phase in decision accounting")
    counts = {p: integer(phase_counts.get(p, 0), p) for p in PHASES}
    require(sum(counts.values()) == count, "phase counts do not sum to policy decisions")
    first = reads.edge(run / "residual_and_projection_audit.jsonl")
    last = reads.edge(run / "residual_and_projection_audit.jsonl", last=True)
    initial_raw = reads.edge(run / "physical_observations.jsonl")
    raw = reads.edge(run / "physical_observations.jsonl", last=True)
    require(first.get("phase_id") == "P01" and first.get("decision_count") == 1
            and first.get("physics_tick") == first.get("physics_ticks")
            and initial_raw.get("physics_tick") == 0 and close(initial_raw.get("simulation_time_s"), 0.),
            "first transition did not start from real P01 tick zero")
    for edge in (first, last):
        require(not edge.get("curriculum_start") and edge.get("task_result_scope") in (None, "full_task"),
                "suffix/prefix scope in a purported natural evaluation")
    require(last.get("decision_count") == count and last.get("physics_tick") == raw.get("physics_tick") == ticks
            and close(last.get("sim_time_s"), actual.get("duration_s"))
            and close(raw.get("simulation_time_s"), ticks / 120.) and close(actual.get("duration_s"), ticks / 120.),
            "terminal decision/raw/manifest clocks differ")
    require(actual["runtime_contract"].get("physics_hz") == 120
            and actual["runtime_contract"].get("decision_hz") == 15, "unsupported clock protocol")
    require(last.get("time_outs") is False, "task timeouts must not be time-limit bootstraps")
    window = flag(actual.get("window_ended_before_task_terminal"), "window ended")
    success = flag(actual.get("task_success"), "task success")
    physical = actual["physical_task_evaluation"]
    require(flag(physical.get("success"), "physical success") == success, "formal physical success mismatch")
    require(actual.get("controller_task_success") == last.get("task_success"), "controller outcome mismatch")
    require(last.get("terminal_bootstrap_allowed") is window, "terminal/bootstrap classification mismatch")
    if window:
        require(not success and last.get("termination_reason") is None and actual.get("termination_reason") is None,
                "external window cannot be called terminal or successful")
    else:
        require(last.get("termination_reason") is not None, "completed task lacks a terminal reason")
    if success:
        require(not window and physical.get("valid") is True and last.get("full_task_success") is True
                and actual.get("controller_task_success") is True, "whole-run success lacks full natural task evidence")
    task = last["semantic_task"]
    completed = task["completed_stage_ids"]
    require(completed == list(PHASES[:len(completed)]), "completed stages are not the continuous forward history")
    unfinished = next((p for p in PHASES if p not in completed), None)
    require(not success or unfinished is None, "success with an unfinished stage")
    transitions = task.get("transition_evidence", [])
    require(isinstance(transitions, list), "missing transition history")
    history = physical.get("history", {})
    events = history.get("event_ticks", {})
    qcp = {leg: {name: events.get(key, {}).get(leg) for name, key in
                 (("qualified", "active_lift"), ("crossed", "front_edge_crossed"), ("placed", "placed"))} for leg in LEGS}
    current = physical.get("current_legs", {})
    terminal_legs = {leg: {**(subset(current.get(leg), LEG_FIELDS) or {}),
                          "current_snapshot_available": isinstance(current.get(leg), dict),
                          "active_lift_history": history.get("active_lift", {}).get(leg),
                          "front_edge_crossed_history": history.get("front_edge_crossed", {}).get(leg),
                          "placed_history": history.get("placed", {}).get(leg)} for leg in LEGS}
    pairs = {}
    for leg, wheel_name in WHEELS.items():
        wheel = raw.get("wheels", {}).get(wheel_name, {})
        contact = raw.get("contacts", {}).get(wheel.get("body_name"), {})
        pairs[leg] = {"wheel": subset(wheel, ("body_name", "center_w_m", "bottom_w_m", "velocity_rad_s", "command_rad_s", "geometry_verified")),
                      "contact_class": contact.get("contact_class"),
                      **{pair: subset(contact.get(pair), ("active", "pair_verified", "force_w_n", "normal_force_n",
                         "contact_point_w_m", "consecutive_active_ticks")) for pair in ("ground", "obstacle")}}
    quality = actual["quality_metrics"]
    phases = {p: quality_row(quality["phases"].get(p), phase=True) for p in PHASES}
    skipped = integer(quality.get("nonfinite_terminal_ticks_skipped", 0), "skipped quality ticks")
    require(sum((x or {}).get("ticks", 0) for x in phases.values()) + skipped == ticks,
            "quality phase ticks plus explicitly skipped nonfinite ticks do not match physical duration")
    require(quality.get("all_phases_sampled") == all(x is not None for x in phases.values()), "quality coverage flag mismatch")
    require(quality.get("all_phases_sampled") or quality.get("fixed_quality_score") is None,
            "missing phases cannot have a complete fixed quality score")
    windows = {p: {name: quality_row(quality.get("transfer_capture_split", {}).get(p, {}).get(name), phase=True)
                   for name in ("CAPTURE", "EXECUTION", "LEGACY_UNSEGMENTED", "TRANSFER")} for p in PHASES}
    failure_class = termination_classification(actual.get("termination_reason"), physical.get("termination_reason"),
                                               success=success, window=window)
    hard_abort = failure_class == "independent_safety_abort"
    classification = {
        "task_success": "full natural-P01 physical task success",
        "window_incomplete": "external decision window ended; unfinished, not success",
        "task_failure": "recorded body-collision or wheel-only task failure; not independent safety abort",
        "independent_safety_abort": "recorded independent safety abort; not task completion",
        "incompletion": "task/controller incompletion; no recorded independent safety abort",
        "infrastructure_error": "recorded infrastructure error; not task success or inferred safety abort",
        "artifact_error": "recorded artifact error; not task success or inferred safety abort",
        "unknown_termination": "unclassified recorded termination; not inferred to be a safety abort",
    }[failure_class]
    status = ("FINAL_COMPLETED_TASK_SUCCESS" if success else "FINAL_COMPLETED_WINDOW_INCOMPLETE" if window else
              "FINAL_COMPLETED_TASK_INCOMPLETE" if failure_class == "incompletion" else "FINAL_COMPLETED_PHYSICAL_FAILURE")
    match = re.fullmatch(r"checkpoint_step_(\d+)\.pt", Path(actual["checkpoint"]).name)
    native = last.get("actuator_target_effect_audit") or {}
    report = {
        "schema": "wlr50_clean.transfer_roles_evaluation_diagnosis.v1", "status": status,
        "run": str(run), "runtime_commit": actual["runtime_contract"]["source_git_commit"],
        "source_checkpoint_step": int(match[1]) if match else None,
        "evaluation": {"from_phase": "P01", "natural_episode": True, "prefix_used": False,
                       "prefix_source_argument_is_not_prefix_execution": args.get("prefix_source"),
                       "seed": actual["seed"], "deterministic_policy": True,
                       "observation_dimension": telemetry["observation_dimension"],
                       "observation_layout": (actual.get("policy_contract") or {}).get("observation_layout"),
                       "policy_decisions": count, "observed_physics_ticks": ticks, "duration_s": actual["duration_s"],
                       "optimizer_updates": 0, "task_success": success, "controller_task_success": actual["controller_task_success"],
                       "termination_reason": actual.get("termination_reason"), "window_ended_before_task_terminal": window,
                       "time_outs": False, "reward_total": actual["reward_total"]},
        "physical": {**subset(physical, ("valid", "reason", "termination_reason", "final_region_valid", "final_controlled",
                                        "final_support_available", "final_stable_for_s")),
                     "classification": classification, "failure_class": failure_class,
                     "recorded_hard_abort": hard_abort, "recorded_task_failure": failure_class == "task_failure",
                     "terminal_raw_all_finite": raw.get("all_finite"), "terminal_raw_body_collision": raw.get("body_collision")},
        "first_unfinished": {"phase": unfinished, **subset(task, ("substage", "stage_age_s", "entry_valid", "entry_reasons",
                                                                  "completion_values", "pending_capture")),
                             "interpretation": "Recorded current goals/history, not a reconstructed predicate or imposed entry pose."},
        "history_event_ticks": qcp, "lift_attempt_events": history.get("lift_attempt_events"),
        "terminal_legs": terminal_legs, "terminal_contacts_raw": pairs,
        "terminal_motion": {"base_raw": raw.get("base"), "goal_features": physical.get("goal_features")},
        "terminal_wheels": {"nominal": last.get("nominal_action_full12", [None]*12)[8:],
                            "requested_residual": last.get("projected_residual_full12", [None]*12)[8:],
                            "final_canonical_drive": last.get("actual_drive_target_full12", [None]*12)[8:],
                            "measured_rad_s": physical.get("measured_wheel_velocity_rad_s")},
        "phase_order": list(PHASES), "decisions_by_phase": counts, "phase_transitions": transitions,
        "clock_scope": {"initial_physics_tick": 0, "final_physics_tick": ticks, "final_decision_ticks": last["physics_ticks"],
                        "head_tail_and_manifest_agree": True, "full_trajectory_clock_rescan": False},
        "final_decision_native_scope": {**(last.get("actuator_target_effect_audit_summary") or {}),
                                        "verified": native.get("verified"), "actual_mapping_matches_dispatch": native.get("actual_mapping_matches_dispatch"),
                                        "setter_dispatch_targets_equal": native.get("setter_dispatch_targets_equal"),
                                        **{key: last.get(key) for key in WRITES},
                                        "no_in_episode_state_writes_verified": last.get("no_in_episode_state_writes_verified"),
                                        "terminal_observation_finite_fallback": last.get("terminal_observation_finite_fallback"),
                                        "terminal_bootstrap_allowed": last.get("terminal_bootstrap_allowed"),
                                        "full_native_trajectory_independent_recheck": None},
        "quality": {"source": "evaluation_manifest.json.quality_metrics", "global": quality_row(quality.get("global")),
                    "phases": phases, "windows": windows, "all_phases_sampled": quality["all_phases_sampled"],
                    "fixed_quality_score": quality.get("fixed_quality_score"), "quality_is_not_success": True,
                    "nonfinite_terminal_ticks_skipped": skipped},
        "scope": ["Two manifests plus JSONL head/tail only; no checkpoint load/hash or full native/raw audit.",
                  "Historical Q/C/P is distinct from current support/AIR. Missing contact/quality evidence is not zero.",
                  "No unique failure cause, comparative superiority, or new physical validation is inferred."],
    }
    reads.verify_unchanged()
    report["source_reads"] = reads.receipt()
    return report


def markdown(r):
    e, p, first = r["evaluation"], r["physical"], r["first_unfinished"]
    def fmt(value):
        return "unavailable" if value is None else f"{value:.6g}" if type(value) is float else str(value)
    lines = [f"# Natural-P01 C / checkpoint {fmt(r['source_checkpoint_step'])}", "",
             f"Run `{Path(r['run']).name}`; runtime `{r['runtime_commit'][:12]}`; deterministic N1, seed {e['seed']}, no prefix, optimizer 0.", "",
             f"**{e['policy_decisions']} decisions / {e['observed_physics_ticks']} ticks / {e['duration_s']:.6f}s; task success={e['task_success']}.** "
             f"Reason `{e['termination_reason']}`; window ended={e['window_ended_before_task_terminal']}; physical valid={p['valid']}, physical reason={p['termination_reason']}. "
             f"Class `{p['failure_class']}`; independent safety abort={p['recorded_hard_abort']}.", "",
             f"First unfinished: **{first['phase']}** / {first['substage']}, age {fmt(first['stage_age_s'])}s. "
             f"Current completion values: `{json.dumps(first['completion_values'], ensure_ascii=False)}`. Pending labels are not success.", "",
             "| Leg | historical Q/C/P ticks | current contact / support / load | front / clearance (m) |",
             "|---|---|---|---|"]
    for leg in LEGS:
        h, c, raw = r["history_event_ticks"][leg], r["terminal_legs"][leg], r["terminal_contacts_raw"][leg]
        lines.append(f"|{leg}|{' / '.join(fmt(h[k]) for k in ('qualified','crossed','placed'))}|"
                     f"{fmt(raw['contact_class'])} / {fmt(c.get('support'))} / {fmt(c.get('load_fraction'))}|"
                     f"{fmt(c.get('front_distance_m'))} / {fmt(c.get('clearance_m'))}|")
    q = r["quality"]
    lines += ["", "P01–P13 decision counts: **" + " / ".join(str(r["decisions_by_phase"][phase]) for phase in PHASES) + "**.", "",
              "Global recorded stability: " + "; ".join(f"{key}={fmt((q['global'] or {}).get(key))}" for key in
                  ("roll_rms_rad", "pitch_rms_rad", "angular_acceleration_rms_rad_s2")) + ". "
              f"All-phase quality score={fmt(q['fixed_quality_score'])}; unsampled phases/windows remain null.", "",
              f"Final-decision native recorded verified={r['final_decision_native_scope']['verified']}; "
              f"state-write verification={r['final_decision_native_scope']['no_in_episode_state_writes_verified']}; "
              f"bootstrap={r['final_decision_native_scope']['terminal_bootstrap_allowed']}. "
              f"Terminal raw finite={p['terminal_raw_all_finite']}; collision record=`{json.dumps(p['terminal_raw_body_collision'])}`.", "",
              "Scope: manifest accounting and trajectory head/tail, not a full stream re-audit. Historical placement is not current load; "
              "an unfinished external window is not task success. No causal or stability-superiority claim."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve(strict=True)
    output = args.output_json.resolve()
    md = output.with_suffix(".md")
    require(run.is_dir() and run.is_relative_to(EVAL_ROOT), "run must be inside the experiment validation root")
    require(output.is_relative_to(ROOT) and output.suffix == ".json", "output must be a new experiment-local JSON path")
    require(not output.exists() and not md.exists(), "refusing to overwrite either diagnosis output")
    report = summarize(run)
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    body = markdown(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(text)
    with md.open("x", encoding="utf-8") as stream:
        stream.write(body)
    print(json.dumps({"json": str(output), "markdown": str(md), "status": report["status"],
                      "decisions": report["evaluation"]["policy_decisions"], "task_success": report["evaluation"]["task_success"],
                      "first_unfinished": report["first_unfinished"]["phase"], "checkpoint_loads": 0,
                      "trajectory_rows_read": 4, "full_trajectory_rescans": 0}))


if __name__ == "__main__":
    main()

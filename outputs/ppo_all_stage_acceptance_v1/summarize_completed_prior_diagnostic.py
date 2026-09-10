"""One completed natural-P01 B diagnostic, never credited as PPO training.

One decision-audit pass; only first/last raw physical rows. Standard library,
exclusive JSON/Markdown outputs, no PT, production imports, simulation or CSV.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
INPUT = PROJECT / "runs/ppo_all_stage_acceptance_v1/diagnostics"
OLD = ROOT.parent / "ppo_transfer_roles_v1"
PHASES = tuple(f"P{i:02d}" for i in range(1, 14))
LEGS = ("FR", "FL", "RR", "RL")
EVENT_NAMES = {"whole_body_initial_clearance": "I", "qualified_measured_upward_lift": "Q",
               "front_edge_crossed": "C", "placed": "P", "qualification_revoked_ground_before_cross": "Q_revoked"}
COMMON = ("valid", "evaluator_version", "run_validity", "physical_evidence_status", "termination_reason",
          "termination_source", "traversal_event_observed", "traversal_task_complete", "strict_recovery_quality",
          "post_completion_loss_observed", "final_region_valid", "final_controlled", "final_support_available")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def module(name):
    spec = importlib.util.spec_from_file_location("prior_report_"+name, OLD/(name+".py"))
    result = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(result)
    finally:
        sys.dont_write_bytecode = previous
    return result


def inside(path, root):
    path = path.resolve()
    need(path == root or root in path.parents, f"outside required report scope: {path}")
    return path


def summarize(run):
    training = module("summarize_completed_training")
    edges = module("summarize_natural_evaluation")
    reads = edges.Reads()
    final = reads.small(run/"run_manifest.json")
    actual = reads.small(run/"evaluation_manifest.json")
    args = final.get("arguments", {})
    need(final.get("lifecycle") == "SUCCEEDED" and final.get("completed_at_utc"),
         "requires finalized evaluator execution; running/failed launch is not a complete B report")
    need(final.get("command") == "eval" and final.get("result") == actual, "formal evaluation binding differs")
    need(args.get("mode") == actual.get("mode") == "semantic_prior_eval"
         and args.get("num_envs") == 1 and args.get("from_phase") == actual.get("from_phase") == "P01"
         and args.get("checkpoint") is None and actual.get("checkpoint") is None, "not natural unprefixed B")
    need(actual.get("optimizer_updates_during_evaluation") == 0 and actual.get("deterministic_policy") is False,
         "B must have no loaded actor or optimizer updates")
    need(actual.get("runtime_contract") == final.get("runtime_contract")
         and actual["runtime_contract"].get("experiment_id") == "all_stage_acceptance_v1", "wrong runtime")
    need(not (run/"prefix_evidence.jsonl").exists(), "prefix source is outside this diagnostic protocol")
    counts, native = Counter(), Counter()
    events, transitions, event_seen, transition_seen = [], [], set(), set()
    path = run/"residual_and_projection_audit.jsonl"
    reads.note(path, "single complete decision JSONL pass")
    last, count, ticks = None, 0, 0
    for row in training.lines(path):
        count += 1
        n = row["physics_ticks"]
        need(type(n) is int and 1 <= n <= 8 and row["decision_count"] == count
             and row["physics_tick"] == ticks+n and row["phase_id"] in PHASES, "decision/tick discontinuity")
        need(count != 1 or row["phase_id"] == "P01", "first decision is not P01")
        need(not row.get("curriculum_start") and row.get("task_result_scope") in (None, "full_task"), "suffix data in B")
        need(row["raw_policy_action_full12"] == [0.]*12 and row["projected_residual_full12"] == [0.]*12,
             "B emitted nonzero residual")
        counts[row["phase_id"]] += 1
        ticks += n
        native.update(training.audit_counts(row, n))
        native["nonzero_actual_canonical_drive_decisions"] += int(any(row["actual_drive_target_full12"]))
        task = row["semantic_task"]
        history = task["history"]
        observed = [(event["leg"], event["event"], event["physics_tick"])
                    for event in history.get("lift_attempt_events", [])]
        for kind, values in history.get("event_ticks", {}).items():
            if kind != "active_lift":
                observed.extend((leg, kind, tick) for leg, tick in values.items())
        for leg, kind, tick in observed:
            need(leg in LEGS and type(tick) is int and 0 <= tick <= ticks, "invalid event clock/leg")
            key = (leg, kind, tick)
            if key not in event_seen:
                event_seen.add(key)
                events.append({"leg": leg, "kind": EVENT_NAMES.get(kind, kind), "recorded_event_name": kind,
                               "physics_tick": tick, "first_observed_diagnostic_decision": count,
                               "source": "zero_residual_semantic_prior_B", "PPO_credit": False})
        for transition in task.get("transition_evidence", []):
            key = (transition["from_stage"], transition["to_stage"], transition["physics_tick"])
            if key not in transition_seen:
                transition_seen.add(key)
                transitions.append({**transition, "observed_diagnostic_decision": count,
                    "observed_on_terminal_decision": row.get("termination_reason") is not None,
                    "decision_bootstrap_allowed": row.get("terminal_bootstrap_allowed")})
        last = row
    need(last is not None, "no decision evidence")
    telemetry = actual["telemetry"]
    need(count == actual["policy_decisions"] == telemetry["decisions"] and
         ticks == actual["observed_physics_ticks"] == telemetry["physics_ticks"], "manifest/decision totals differ")
    need(dict(counts) == telemetry["phase_decisions"] and telemetry["episodes"] == 1, "phase/reset mismatch")
    first_raw = reads.edge(run/"physical_observations.jsonl")
    raw = reads.edge(run/"physical_observations.jsonl", last=True)
    need(first_raw.get("physics_tick") == 0 and raw.get("physics_tick") == ticks
         and edges.close(raw.get("simulation_time_s"), ticks/120.)
         and edges.close(actual["duration_s"], ticks/120.), "raw/decision endpoint clocks differ")
    task, physical = last["semantic_task"], actual["physical_task_evaluation"]
    need(actual["task_success"] == physical["success"], "formal physical success differs")
    window = actual["window_ended_before_task_terminal"]
    need(type(window) is bool and last["terminal_bootstrap_allowed"] is window, "terminal/bootstrap mismatch")
    reads.verify_unchanged()
    return {
        "schema": "ppo_all_stage_acceptance_v1.completed_prior_diagnostic.v1", "run": str(run),
        "lifecycle": final["lifecycle"], "mode": "B_semantic_prior_eval", "seed": actual["seed"],
        "diagnostic_decisions": count, "physics_ticks": ticks, "duration_s": actual["duration_s"],
        "new_training_decisions": 0, "PPO_updates": 0, "prefix_decisions": 0,
        "task_success": actual["task_success"], "controller_task_success": actual.get("controller_task_success"),
        "window_ended_before_task_terminal": window, "termination_reason": actual.get("termination_reason"),
        "termination_source": task.get("termination_source"), "common_physical_result": {k: physical.get(k) for k in COMMON},
        "phase_counts": {p: counts[p] for p in PHASES}, "phase_transitions": transitions, "events": events,
        "event_counts_by_leg": {leg: dict(Counter(e["kind"] for e in events if e["leg"] == leg)) for leg in LEGS},
        "native_audit": dict(native), "first_unfinished_stage_label": next((p for p in PHASES if p not in task["completed_stage_ids"]), None),
        "current_task": {k: task.get(k) for k in ("stage_id", "completion_values", "entry_reasons", "local_timeout", "stage_age_s")},
        "current_legs_at_end": physical.get("current_legs"), "history_at_end": physical.get("history"),
        "terminal_raw_contacts": raw.get("contacts"), "terminal_raw_body_collision": raw.get("body_collision"),
        "sources": reads.receipt(),
        "limits": ["B diagnostic, never PPO training or a policy success denominator.",
                   "I/Q/C/P/revoke are logged evaluator events, not independent physical replay.",
                   "Native checks consume saved same-tick audit evidence; zero PPO effect is expected for B, not zero actuator motion.",
                   "Only raw first/last sampled; no full raw/contact rescan. Empty independent native/transition files are not imputed.",
                   "First unfinished label is not an independent physical failure classification; current goals and termination source remain separate.",
                   "Unvisited phases have zero diagnostic samples, not zero quality."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run, output = inside(args.run, INPUT), inside(args.output_dir, ROOT)
    paths = [output/"prior_diagnostic_summary.json", output/"prior_diagnostic_summary.md"]
    need(not any(p.exists() for p in paths), "exclusive output exists")
    report = summarize(run)
    lines = ["# Completed B diagnostic (not training)", "", f"Run: `{run}`", "",
             f"{report['diagnostic_decisions']} diagnostic decisions / {report['physics_ticks']} ticks / {report['duration_s']:.6f}s.",
             f"Task success: {report['task_success']}; external window: {report['window_ended_before_task_terminal']}.",
             f"Termination: {report['termination_reason']}; source: {report['termination_source']}.",
             f"First unfinished label: {report['first_unfinished_stage_label']} (see actual goals in JSON).", "",
             "| Phase | Diagnostic decisions |", "|---|---:|"]
    lines += [f"| {p} | {n} |" for p, n in report["phase_counts"].items()]
    lines += ["", "Leg event counts (recorded I/Q/C/P/revoke):", ""]
    lines += [f"- {leg}: `{json.dumps(values, ensure_ascii=False)}`" for leg, values in report["event_counts_by_leg"].items()]
    lines += ["", "Recorded event ticks (I is initial clearance, not qualification):", ""]
    lines += [f"- {e['leg']} {e['kind']} @ {e['physics_tick']} (no PPO credit)" for e in report["events"]]
    if report["window_ended_before_task_terminal"]:
        lines += ["", "External diagnostic window ended; internal termination reason/source remain as recorded, not inferred."]
    lines += ["", "Native audit: `"+json.dumps(report["native_audit"])+"`", "", *report["limits"]]
    payloads = [json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", "\n".join(lines)+"\n"]
    output.mkdir(parents=True, exist_ok=True)
    for path, payload in zip(paths, payloads):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(json.dumps({"outputs": [str(p) for p in paths], "diagnostic_decisions": report["diagnostic_decisions"], "training_decisions": 0}))


if __name__ == "__main__":
    main()

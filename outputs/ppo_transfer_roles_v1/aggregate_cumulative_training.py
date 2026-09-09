"""Cumulative completed-summary ledger; no fixed check2 endpoint or raw scans.

PREPARED ONLY: do not run until authorized. Standard library only; imports just
the old output helper's JSON/episode validators, never its fixed CHECKS runner.
Default completed inputs are blocks1..6 (7168 decisions). Later use --blocks
1 2 3 4 5 6 7, etc., ONLY after each per-block summary and final manifests exist.
--pending block7:C:/.../run reads one training manifest snapshot, never updates
or trajectories, and excludes that block completely from completed totals.
--plan block8:P06:1536 describes unstarted allocation only. Labels are disjoint.
--eval-report defaults to the two completed C1/C2 diagnoses; their run paths
locate authoritative final evaluation/run manifests. Pass the full report list
when adding later evaluations. Denominator is unique explicitly supplied runs,
including final launcher failures; this is not a search of all historical evals.
Always use a fresh --output-dir under this outputs directory. No overwrite,
checkpoint/PT read, raw JSONL, polling, simulator, torch, or training_manifest edit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import json

from aggregate_completed_blocks import (
    ROOT, HEADER, PHASES, SCHEMA, FINAL_LIFECYCLES,
    require, count, read_document, validate_episodes,
)

PROJECT = ROOT.parents[1]
TRAIN_ROOT = PROJECT / "runs/ppo_transfer_roles_v1/train"
EVAL_ROOT = PROJECT / "runs/ppo_transfer_roles_v1/validation"
BASE_SOURCE, BASE_END, BASE_DECISIONS = 123136, 130304, 7168


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path == root or root in path.parents, f"path outside intended directory: {path}")
    return path


def load(path):
    return read_document(Path(path))


def validate_completed(index, cursor):
    directory = ROOT / f"block{index}_summary"
    summary, receipt = load(directory / "training_block_summary.json")
    coverage, coverage_receipt = load(directory / "coverage_table.json")
    require(summary.get("schema") == SCHEMA and len(summary.get("runs", [])) == 1, "invalid single-block summary")
    run = summary["runs"][0]
    run_dir = inside(run["run"], TRAIN_ROOT)
    manifest, manifest_receipt = load(run_dir / "run_manifest.json")
    training, training_receipt = load(run_dir / "training_manifest.json")
    require(manifest.get("command") == "train", "summary points to a non-training run")
    require(manifest.get("lifecycle") in FINAL_LIFECYCLES and manifest.get("completed_at_utc"),
            f"block{index} is not final; no running/partial data can enter completed totals")
    require(run.get("lifecycle") == training.get("lifecycle") == manifest["lifecycle"], "final lifecycle mismatch")
    fields = {"actual_decisions": "actual_policy_decisions", "end_global": "global_policy_decisions",
              "updates": "ppo_updates_this_run", "optimizer_steps": "optimizer_steps_this_run",
              "planned_decisions": "planned_requested_policy_decisions",
              "unconsumed_decisions": "unconsumed_requested_policy_decisions", "rounding_overrun": "rounding_overrun"}
    for short, full in fields.items():
        require(count(run.get(short), short) == count(training.get(full), full)
                == count(manifest.get("result", {}).get(full), f"final result.{full}"), f"summary/final mismatch: {short}")
    n = run["actual_decisions"]
    require(n > 0 and n % 128 == 0 and run["source_global"] == cursor and run["end_global"] == cursor+n,
            "duplicate/gapped or unverified complete-update interval")
    require(n + run["unconsumed_decisions"] == run["planned_decisions"] + run["rounding_overrun"], "allocation accounting differs")
    require(run["updates"] == n//128 and run["optimizer_steps"] == 20*run["updates"], "update accounting differs")
    phases = validate_episodes(run)
    core = training["telemetry"]["core"]
    require(core["decisions"] == n and core["physics_ticks"] == run["totals"]["physics_ticks"], "core accounting differs")
    require({p: phases.get(p, 0) for p in PHASES} == {p: core["phase_decisions"].get(p, 0) for p in PHASES}, "core phases differ")
    prefix = run["prefix_excluded"]
    for short, full in (("decisions", "prefix_behavior_decisions"), ("physics_ticks", "prefix_physics_ticks")):
        require(count(prefix.get(short, 0), short) == count(core.get(full, 0), full), "teacher exclusion mismatch")
    require(coverage.get("header") == HEADER and len(coverage.get("rows", [])) == 13, "coverage structure differs")
    seen, sums = set(), Counter()
    for values in coverage["rows"]:
        require(isinstance(values, list) and len(values) == len(HEADER), "coverage row shape differs")
        row = dict(zip(HEADER, values)); phase = row["source_phase"]
        require(phase in PHASES and phase not in seen, "duplicate/unknown phase")
        seen.add(phase)
        require(row["run"] == run_dir.name and row["source_global"] == cursor and row["end_global"] == run["end_global"], "coverage interval differs")
        require(row["optimized_decisions"] == phases[phase] and type(row["sampled"]) is bool
                and row["sampled"] == (phases[phase] > 0), "coverage actual/sample flag differs")
        for key in HEADER[4:-1]: sums[key] += count(row[key], key)
    for key in ("optimized_decisions", "physics_ticks", "native_effect_ticks", "own_phase_effect_ticks",
                "native_audit_anomaly_rows", "state_write_anomaly_rows", "nonterminal_handoffs"):
        # The source Counter serializes no handoff key for a single-phase block.
        # Only this event count may be implicitly zero; required audit totals stay strict.
        expected = run["totals"].get(key, 0) if key == "nonterminal_handoffs" else run["totals"][key]
        require(sums[key] == expected, f"coverage total differs: {key}")
    require(run["checkpoints"] and run["checkpoints"][-1]["global_policy_decisions"] == run["end_global"], "no stored final checkpoint receipt")
    info = {"block": index, "run": str(run_dir), "summary": receipt, "coverage": coverage_receipt,
            "run_manifest": manifest_receipt, "training_manifest": training_receipt,
            "runtime_commit": training["runtime_contract"]["source_git_commit"],
            "observation_dimension": (training.get("policy_contract") or {}).get("observation_dimension")}
    return run, phases, coverage["rows"], info


def evaluations(paths):
    records, seen = [], set()
    for path in paths:
        report, report_receipt = load(inside(path if path.is_absolute() else ROOT/path, ROOT))
        location = report.get("run") or (report.get("source") or {}).get("run_directory")
        require(isinstance(location, str), "evaluation report lacks its run identity")
        run = inside(location, EVAL_ROOT)
        require(str(run).casefold() not in seen, "same evaluation counted twice")
        seen.add(str(run).casefold())
        wrapper, receipt = load(run/"run_manifest.json"); args = wrapper.get("arguments", {})
        require(wrapper.get("command") == "eval" and wrapper.get("lifecycle") in {"SUCCEEDED", "FAILED"}
                and wrapper.get("completed_at_utc"), "evaluation is not a completed attempt")
        require(args.get("mode") == "semantic_residual_eval" and args.get("from_phase") == "P01"
                and args.get("num_envs") == 1, "denominator accepts only natural-P01 N1 C evaluations")
        row = {"run": str(run), "report": report_receipt, "run_manifest": receipt,
               "lifecycle": wrapper["lifecycle"], "checkpoint": args.get("checkpoint"), "seed": args.get("seed"),
               "runtime_commit": wrapper["runtime_contract"]["source_git_commit"],
               "task_success": None, "formal_task_result_available": False}
        if (run/"evaluation_manifest.json").exists():
            actual, evidence = load(run/"evaluation_manifest.json")
            require(actual.get("mode") == "semantic_residual_eval" and actual.get("from_phase") == "P01"
                    and actual.get("deterministic_policy") is True and actual.get("optimizer_updates_during_evaluation") == 0,
                    "evaluation protocol differs")
            require(actual.get("checkpoint") == args.get("checkpoint") and actual.get("seed") == args.get("seed"), "eval identity mismatch")
            require(type(actual.get("task_success")) is bool, "missing explicit task outcome")
            row.update(evaluation_manifest=evidence, task_success=actual["task_success"],
                formal_task_result_available=True, termination_reason=actual.get("termination_reason"),
                policy_decisions=actual.get("policy_decisions"), physics_ticks=actual.get("observed_physics_ticks"),
                physical_valid=(actual.get("physical_task_evaluation") or {}).get("valid"),
                window_ended_before_task_terminal=actual.get("window_ended_before_task_terminal"))
        else:
            require(wrapper["lifecycle"] == "FAILED", "successful launcher lacks formal evaluation manifest")
        records.append(row)
    return {"completed_attempt_denominator": len(records),
            "full_task_successes": sum(r["task_success"] is True for r in records),
            "reported_task_non_successes": sum(r["task_success"] is False for r in records),
            "missing_task_result_failures": sum(not r["formal_task_result_available"] for r in records),
            "scope": "unique explicitly supplied completed natural-P01 C attempts, not training or suffix outcomes",
            "attempts": records}


def build(args):
    require(args.blocks == list(range(1, len(args.blocks)+1)) and len(args.blocks) >= 6,
            "cumulative input must contain consecutive blocks1..N with N>=6")
    totals, phases, outcomes, labels, scopes, prefix, events = (Counter() for _ in range(7))
    cursor, rows, runs, sources = BASE_SOURCE, [], [], []
    tails, seen_runs = [], set()
    for index in args.blocks:
        run, counts, coverage, receipt = validate_completed(index, cursor)
        require(run["run"].casefold() not in seen_runs, "duplicate completed run")
        seen_runs.add(run["run"].casefold()); cursor = run["end_global"]
        if index == 6: require(cursor == BASE_END and cursor-BASE_SOURCE == BASE_DECISIONS, "historical7168 anchor differs")
        phases.update(counts); rows.extend(coverage); sources.append(receipt)
        for key in ("actual_decisions", "updates", "optimizer_steps", "planned_decisions", "unconsumed_decisions", "rounding_overrun"):
            totals[key] += run[key]
        for key, value in run["totals"].items(): totals[key] += count(value, key)
        for key, value in run["prefix_excluded"].items(): prefix[key] += count(value, f"prefix.{key}")
        episodes = []
        for ep in run["episodes"]:
            events.update({k: count(v, k) for k, v in ep.get("event_counts_by_leg_and_credit", {}).items()})
            compact = {k: ep.get(k) for k in ("episode_index", "scope", "decisions", "physics_ticks", "terminal",
                "last_active_stage", "first_unfinished_stage", "termination_reason", "task_success", "full_task_success")}
            episodes.append(compact)
            if ep["terminal"]:
                outcomes[ep["termination_reason"]] += 1; labels[ep["task_outcome_label"]] += 1
                scopes[ep["scope"]] += 1
            else: tails.append({"block": index, **compact})
        runs.append({"block": index, **{k: run[k] for k in ("run", "lifecycle", "source_global", "end_global",
            "actual_decisions", "planned_decisions", "unconsumed_decisions", "rounding_overrun", "updates", "optimizer_steps",
            "sampling", "checkpoints", "prefix_excluded")}, "episodes": episodes})
    used = {f"block{i}" for i in args.blocks}; pending, plans = [], []
    for specification in args.pending:
        label, location = specification.split(":", 1)
        require(label and label not in used, "duplicate completed/pending/planned label"); used.add(label)
        run = inside(location, TRAIN_ROOT)
        require(str(run).casefold() not in seen_runs, "pending run already counted"); seen_runs.add(str(run).casefold())
        document, receipt = load(run/"run_manifest.json")
        require(document.get("command") == "train" and document.get("lifecycle") not in FINAL_LIFECYCLES,
                "final pending run needs a completed summary, not a partial label")
        pending.append({"label": label, "run_manifest": receipt, "lifecycle": document.get("lifecycle"),
            "requested_decisions": document.get("arguments", {}).get("decisions"),
            "completed_decisions_counted": 0, "live_optimized_or_collected_decisions": None,
            "scope": "excluded snapshot; no update/raw trajectory read"})
    for specification in args.plan:
        label, phase, raw = specification.split(":"); allocation = int(raw)
        require(label and label not in used and phase in PHASES and allocation > 0, "invalid or duplicate plan")
        used.add(label); plans.append({"label": label, "from_phase": phase, "requested_decisions": allocation,
                                      "actual_decisions": None, "completed_decisions_counted": 0})
    require(totals["actual_decisions"] == cursor-BASE_SOURCE == sum(phases.values()), "aggregate interval differs")
    summary = {"schema": "ppo_transfer_roles_v1.cumulative_completed_summary.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "source_global": BASE_SOURCE,
        "end_global": cursor, "base_completed_decisions": BASE_DECISIONS,
        "additional_completed_decisions": totals["actual_decisions"]-BASE_DECISIONS,
        "completed_totals": dict(totals), "policy_phase_counts": {p: phases[p] for p in PHASES},
        "completed_episode_denominator": sum(outcomes.values()), "termination_counts": dict(outcomes),
        "task_outcome_label_counts": dict(labels), "completed_scope_counts": dict(scopes),
        "nonterminal_update_boundary_tails": tails, "prefix_excluded_totals": dict(prefix),
        "event_counts_by_leg_and_credit": dict(events), "runs": runs, "source_documents": sources,
        "pending_runs_excluded": pending, "unstarted_plans_excluded": plans,
        "formal_evaluation": evaluations(args.eval_report),
        "limits": ["No raw JSONL, checkpoint tensors, GPU or physics read/execution; stored evidence only.",
            "Tails are optimized samples but not terminal failures/successes; prefixes receive no PPO credit.",
            "Prefix *_unavailable_rows retain missing detailed evidence; zero effect fields there do not prove zero native activity.",
            "Unvisited phases have zero samples, not zero quality; runtimes/layouts can differ across completed blocks.",
            "No obsolete check2 endpoint assumed; plan/partial counts are never added to actual or eval denominators."]}
    return {"header": HEADER, "rows": rows, "scope": "completed per-run source-phase coverage; no quality imputation"}, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, nargs="+", default=list(range(1, 7)))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pending", action="append", default=[], metavar="LABEL:RUN_PATH")
    parser.add_argument("--plan", action="append", default=[], metavar="LABEL:PHASE:DECISIONS")
    parser.add_argument("--eval-report", type=Path, nargs="+", default=[Path(f"evaluation_{n}_diagnosis.json") for n in (127232, 130304)])
    args = parser.parse_args(); output = inside(args.output_dir, ROOT)
    paths = [output/"cumulative_coverage_table.json", output/"cumulative_training_summary.json"]
    require(not any(p.exists() for p in paths), "exclusive output exists; use a fresh snapshot directory")
    documents = build(args)  # Validate every input before creating outputs.
    payloads = [json.dumps(d, ensure_ascii=False, allow_nan=False, indent=2)+"\n" for d in documents]
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        streams = [stack.enter_context(p.open("x", encoding="utf-8")) for p in paths]
        for stream, payload in zip(streams, payloads): stream.write(payload)
    print(json.dumps({"outputs": [str(p) for p in paths], "actual_decisions": documents[1]["completed_totals"]["actual_decisions"]}))


if __name__ == "__main__":
    main()

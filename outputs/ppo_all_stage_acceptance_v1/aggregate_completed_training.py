"""Aggregate ONLY published all-stage completed-block JSON summaries.

Repeat --summary for each training_block_summary.json (filename is not special).
Inputs must cover a contiguous interval beginning at 138496/1047/20940.
--output is an exclusive new JSON path under this experiment's outputs root.
No raw/manifest/checkpoint reads, runtime imports, CSV, polling, or simulation.
Missing optional evidence is UNAVAILABLE, not zero or a success claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT.parents[1] / "runs/ppo_all_stage_acceptance_v1/train"
INPUT_SCHEMA = "ppo_all_stage_acceptance_v1.completed_blocks.v1"
OUTPUT_SCHEMA = "ppo_all_stage_acceptance_v1.cumulative_completed_summaries.v1"
PHASES = tuple(f"P{i:02d}" for i in range(1, 14))
BASE = {"global_policy_decisions": 138496, "ppo_updates": 1047, "optimizer_steps": 20940}
UNAVAILABLE = "UNAVAILABLE"
FINAL = {"SUCCEEDED", "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY"}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label):
    need(type(value) is int and value >= 0, f"{label}: expected nonnegative integer, not bool/null")
    return value


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    need(path != root and root in path.parents, f"path outside required subtree: {path}")
    return path


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def load_summary(path):
    path = inside(path, ROOT)
    need(path.suffix.lower() == ".json" and path.is_file(), f"missing summary JSON: {path}")
    raw = path.read_bytes()
    document = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant)
    need(isinstance(document, dict), "summary must be a JSON object")
    return document, {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def phases(value):
    need(isinstance(value, dict) and set(value) <= set(PHASES), "invalid/missing source-phase counts")
    # The existing parser writes a sparse Counter: absent phases here mean zero samples.
    return {p: integer(value.get(p, 0), p) for p in PHASES}


def validate_run(run):
    need(isinstance(run, dict) and run.get("lifecycle") in FINAL, "input run is not finalized")
    identity = str(inside(run["run"], TRAIN_ROOT)).casefold()
    source, end, n = (integer(run.get(k), k) for k in ("source_global", "end_global", "actual_decisions"))
    need(n > 0 and n % 128 == 0 and end == source+n, "invalid completed-update interval")
    need(integer(run.get("updates"), "updates") == n//128
         and integer(run.get("optimizer_steps"), "optimizer_steps") == 20*(n//128), "128/20 update counts differ")
    allocation = [run.get(k) for k in ("planned_decisions", "unconsumed_decisions", "rounding_overrun")]
    if all(v is not None for v in allocation):
        planned, unused, overrun = (integer(v, "allocation") for v in allocation)
        need(n+unused == planned+overrun, "actual/unused/planned/rounding ledger differs")
    totals = run.get("totals")
    need(isinstance(totals, dict) and integer(totals.get("optimized_decisions"), "optimized_decisions") == n,
         "missing/inconsistent optimized totals")
    episodes = run.get("episodes")
    need(isinstance(episodes, list) and episodes, "missing credited episode/tail ledger")
    cursor, ticks, terminals, counts = source+1, 0, 0, Counter()
    for i, ep in enumerate(episodes):
        need(isinstance(ep, dict) and integer(ep.get("episode_index"), "episode_index") == i,
             "duplicate/gapped episode index")
        d = integer(ep.get("decisions"), "episode decisions")
        need(d > 0 and ep.get("first_global") == cursor and ep.get("last_global") == cursor+d-1,
             "episode interval gap/overlap")
        pc = phases(ep.get("phase_samples"))
        need(sum(pc.values()) == d, "episode phase counts differ")
        counts.update(pc)
        t = integer(ep.get("physics_ticks"), "episode physics ticks")
        need(d <= t <= 8*d, "invalid credited physics duration")
        ticks += t
        need(type(ep.get("terminal")) is bool, "missing explicit terminal/tail classification")
        if ep["terminal"]:
            terminals += 1
            if ep.get("task_success") is True:
                need(ep.get("termination_reason") == "SUCCESS", "success/reason contradiction")
            if ep.get("full_task_success") is True:
                need(ep.get("task_success") is True and ep.get("scope") == "full_task"
                     and ep.get("task_outcome_label") == "FULL_TASK_SUCCESS", "suffix labelled full success")
        else:
            need(i == len(episodes)-1 and ep["last_global"] == end, "tail is not the final update-boundary segment")
            need(ep.get("termination_reason") is None and ep.get("task_outcome_label") is None
                 and ep.get("task_success") in (None, False) and ep.get("full_task_success") in (None, False),
                 "tail incorrectly counted as terminal outcome")
        cursor += d
    need(cursor == end+1 and sum(counts.values()) == n, "episode ledger does not cover completed block")
    need(ticks == integer(totals.get("physics_ticks"), "physics_ticks"), "episode/run physics counts differ")
    need(integer(run.get("completed_episode_count"), "completed_episode_count") == terminals,
         "completed count includes/omits a tail")
    need(integer(run.get("completed_episode_rows_outside_boundary", 0), "outside-boundary episodes") == 0,
         "outside-boundary episodes included")
    return identity, counts


def aggregate(documents, expected_end=None):
    """Pure stored-document aggregation; paths in receipts are NEVER dereferenced."""
    runs, sources, identities = [], [], set()
    for document, receipt in documents:
        need(document.get("schema") == INPUT_SCHEMA, "wrong namespace/schema; cumulative outputs are not block inputs")
        block_runs = document.get("runs")
        need(isinstance(block_runs, list) and block_runs, "summary contains no completed runs")
        if "actual_decisions" in document:
            need(document["actual_decisions"] == sum(integer(r.get("actual_decisions"), "actual_decisions")
                                                    for r in block_runs), "document decision total differs")
        sources.append({**receipt, "document_metadata": {k:v for k,v in document.items() if k != "runs"}})
        for run in block_runs:
            identity, _ = validate_run(run)
            need(identity not in identities, "duplicate completed run across inputs")
            identities.add(identity); runs.append(run)
    need(runs, "no completed inputs")
    runs.sort(key=lambda r: r["source_global"])
    cursor, update_cursor, step_cursor = (BASE[k] for k in BASE)
    total, policy, prefix, prefix_phases, events = (Counter() for _ in range(5))
    reasons, labels, terminal_sources = Counter(), Counter(), Counter()
    tails, unavailable, missing_prefix, missing_events = [], [], [], []
    terminal_n = full_n = suffix_n = unknown_success = 0
    latest_receipt = UNAVAILABLE
    for run in runs:
        label = run["run"]
        need(run["source_global"] == cursor, "gap/overlap or wrong source: must start at138496 and remain contiguous")
        _, pc = validate_run(run); policy.update(pc)
        expected_source = {"global_policy_decisions": cursor, "ppo_updates": update_cursor, "optimizer_steps": step_cursor}
        cursor = run["end_global"]; update_cursor += run["updates"]; step_cursor += run["optimizer_steps"]
        for k in ("actual_decisions", "updates", "optimizer_steps"):
            total[k] += run[k]
        total["physics_ticks"] += run["totals"]["physics_ticks"]
        saved = run.get("final_checkpoint_receipt")
        if isinstance(saved, dict):
            need(saved.get("global_policy_decisions") == cursor and saved.get("ppo_updates") == update_cursor
                 and saved.get("optimizer_steps") == step_cursor and saved.get("save_load_round_trip") is True,
                 "recorded final checkpoint lifetime/round-trip receipt differs")
            need(saved.get("source_counters_from_final_minus_this_run") == expected_source,
                 "checkpoint receipt source counters differ from contiguous ledger")
            latest_receipt = saved
        else:
            latest_receipt = UNAVAILABLE
            unavailable.append({"run": label, "field": "final_checkpoint_receipt"})
        p = run.get("prefix_excluded")
        if isinstance(p, dict):
            # This field is explicitly the parser's sparse Counter, including {} for no prefix.
            prefix.update({k: integer(v, f"prefix.{k}") for k,v in p.items()})
            pp = run.get("prefix_phase_samples")
            if isinstance(pp, dict):
                ppc = phases(pp)
                need(sum(ppc.values()) == p.get("decisions", 0), "prefix phase/decision counts differ")
                prefix_phases.update(ppc)
            else:
                unavailable.append({"run": label, "field": "prefix_phase_samples"})
        else:
            missing_prefix.append(label)
        run_events = Counter()
        for ep in run["episodes"]:
            if ep["terminal"]:
                terminal_n += 1
                reasons[ep.get("termination_reason") or UNAVAILABLE] += 1
                labels[ep.get("task_outcome_label") or UNAVAILABLE] += 1
                if type(ep.get("full_task_success")) is not bool or type(ep.get("task_success")) is not bool:
                    unknown_success += 1
                else:
                    full_n += int(ep["full_task_success"])
                    suffix_n += int(ep["task_success"] and not ep["full_task_success"])
                ts = ep.get("termination_sources_at_end") or {}
                terminal_sources[str(ts.get("supervisor") or UNAVAILABLE)] += 1
            else:
                tails.append({"run": label, **{k:ep.get(k, UNAVAILABLE) for k in
                    ("episode_index", "first_global", "last_global", "decisions", "physics_ticks", "last_active_stage")}})
            ee = ep.get("events")
            if not isinstance(ee, list):
                missing_events.append({"run": label, "episode_index": ep["episode_index"]})
                continue
            for event in ee:
                source = event.get("source_classification") or (
                    "current_PPO_policy" if event.get("credit") == "policy" else UNAVAILABLE)
                key = f'{event.get("leg", UNAVAILABLE)}:{event.get("kind", UNAVAILABLE)}:{source}'
                run_events[key] += 1
        recorded = run.get("event_counts_by_leg_and_source")
        if isinstance(recorded, dict):
            need(dict(run_events) == recorded, "recorded event-source aggregate differs from preserved events")
        events.update(run_events)
        for k in ("sampling", "authority_by_source_phase", "changed_actor_updates", "finite_gradient_updates",
                  "actor_chain_matches", "planned_decisions", "unconsumed_decisions", "rounding_overrun"):
            if run.get(k) is None:
                unavailable.append({"run": label, "field": k})
    need(cursor-BASE["global_policy_decisions"] == total["actual_decisions"] == sum(policy.values()), "cumulative credit differs")
    if expected_end is not None:
        need(cursor == expected_end, "actual final checkpoint differs from --expected-end")
    coverage = {"header": ["phase", "credited_policy_decisions", "credited_physics_ticks",
        "PPO_updates_in_revision", "current_version_real_coverage", "task_completed", "evidence_scope"],
        "rows": [[p, policy[p], UNAVAILABLE, UNAVAILABLE,
            "RECORDED_OPTIMIZED_SAMPLES" if policy[p] else "UNSAMPLED", UNAVAILABLE,
            "Supplied completed blocks only; phase tick/revision split and physical completion not inferred from counts."] for p in PHASES],
        "merge_note": "Candidate phase_and_task_chain_coverage table only; does not overwrite audit_tables.json."}
    prefix_totals = {k:prefix[k] for k in ("decisions", "physics_ticks", "attempts", "accepted_attempts")}
    prefix_totals.update(prefix)
    return {"schema": OUTPUT_SCHEMA, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_lifetime_counters": BASE, "end_lifetime_counters_from_ledger": {
            "global_policy_decisions": cursor, "ppo_updates": update_cursor, "optimizer_steps": step_cursor},
        "new_training_totals": dict(total), "policy_phase_counts": {p:policy[p] for p in PHASES},
        "completed_episode_count": terminal_n, "termination_reason_counts": dict(reasons),
        "terminal_supervisor_source_counts": dict(terminal_sources), "task_outcome_label_counts": dict(labels),
        "recorded_full_task_training_successes": full_n, "recorded_suffix_training_successes": suffix_n,
        "terminal_success_fields_unavailable_count": unknown_success,
        "nonterminal_tail_count": len(tails), "nonterminal_tail_decisions": sum(t["decisions"] for t in tails),
        "nonterminal_update_boundary_tails": tails,
        "prefix_excluded": {"known_totals": prefix_totals, "complete": not missing_prefix,
            "totals": prefix_totals if not missing_prefix else UNAVAILABLE, "unavailable_runs": missing_prefix,
            "phase_counts_complete": not missing_prefix and not any(
                x["field"] == "prefix_phase_samples" for x in unavailable),
            "known_source_phase_counts": {p:prefix_phases[p] for p in PHASES}},
        "event_counts_by_leg_and_source": dict(events), "event_lists_unavailable": missing_events,
        "latest_checkpoint_receipt": latest_receipt, "unavailable_evidence": unavailable,
        "phase_and_task_chain_coverage": coverage, "source_documents": sources,
        "runs": runs, "formal_natural_P01_evaluation": UNAVAILABLE,
        "limits": ["Only explicitly supplied completed summary JSONs were read. No JSONL, PT, sidecar, runtime, or simulator access.",
            "Base138496/1047/20940 is excluded from new totals; the previous15360 training decisions are not counted again.",
            "SUCCEEDED is process finality, not task success. Prefix/suffix outcomes and optimized tails are not formal P01 successes.",
            "Checkpoint hashes/reloads are preserved recorded receipts, not newly verified tensors or hashes.",
            "Inherited Q/C/P is not current bearing support; absent failed-prefix full event chains remain unavailable.",
            "Known event/authority records are preserved per run; missing evidence and unvisited quality are not imputed as zero.",
            "All supplied revisions are accumulated; sampled coverage does not certify the current runtime or physical task completion."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-end", type=int)
    args = parser.parse_args()
    inputs = [inside(p, ROOT) for p in args.summary]
    need(len(set(inputs)) == len(inputs), "duplicate summary input")
    output = inside(args.output, ROOT)
    need(output.suffix.lower() == ".json" and not output.exists(), "output must be a fresh exclusive JSON path")
    result = aggregate([load_summary(p) for p in inputs], expected_end=args.expected_end)
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(json.dumps({"output": str(output), "actual_new": result["new_training_totals"],
                      "latest_lifetime": result["end_lifetime_counters_from_ledger"]}))


if __name__ == "__main__":
    main()

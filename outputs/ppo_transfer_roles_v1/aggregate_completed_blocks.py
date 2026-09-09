"""Combine one explicitly bounded group of completed block summaries, once only.

Check 1 (default): blocks 1-4, 1024 each, source 123136 -> 127232.
Run only after block4_summary is genuinely published as complete:
    python outputs/ppo_transfer_roles_v1/aggregate_completed_blocks.py

Check 2: blocks 5-7, allocations 1536/1536/1024, source 127232 -> 131328.
Run only after block6_summary AND block7_summary are genuinely complete:
    python outputs/ppo_transfer_roles_v1/aggregate_completed_blocks.py --check 2
Check 2 writes exclusively inside check2_summary; check 1 files are untouched.

Standard library only. Reads exactly eight (check 1) or six (check 2) existing
JSON summary files, never raw audits, checkpoint tensors, or simulator files.
Does not watch or poll.
This verifies stored-summary consistency, not fresh physical/model correctness.
Both destination names must be absent; existing evidence is never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHECKS = {
    1: {"source": 123136, "final": 127232,
        "allocations": ((1, 1024), (2, 1024), (3, 1024), (4, 1024)),
        "output_directory": "",
        "aggregation_schema": "ppo_transfer_roles_v1.first4096_summary_aggregation.v1"},
    2: {"source": 127232, "final": 131328,
        "allocations": ((5, 1536), (6, 1536), (7, 1024)),
        "output_directory": "check2_summary",
        "aggregation_schema": "ppo_transfer_roles_v1.check2_summary_aggregation.v1"},
}
PHASES = tuple(f"P{i:02d}" for i in range(1, 14))
SCHEMA = "ppo_transfer_roles_v1.completed_blocks.v1"
HEADER = ["run", "source_global", "end_global", "source_phase", "optimized_decisions",
          "physics_ticks", "nonterminal_handoffs", "terminal_handoffs", "native_effect_ticks",
          "own_phase_effect_ticks", "native_audit_anomaly_rows", "state_write_anomaly_rows", "sampled"]
FINAL_LIFECYCLES = {"SUCCEEDED", "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value, name):
    require(type(value) is int and value >= 0, f"{name}: expected a nonnegative integer, not bool/null")
    return value


def reject_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_document(path):
    require(path.is_file(), f"missing completed input: {path}")
    raw = path.read_bytes()
    doc = json.loads(raw.decode("utf-8-sig"), parse_constant=reject_constant)
    require(isinstance(doc, dict), f"expected JSON object: {path}")
    return doc, {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def phase_counts(values, label):
    require(isinstance(values, dict) and set(values) <= set(PHASES), f"invalid phase names: {label}")
    return {phase: count(values.get(phase, 0), f"{label}.{phase}") for phase in PHASES}


def validate_episodes(run):
    episodes = run.get("episodes")
    require(isinstance(episodes, list) and episodes, "missing credited episode/tail ledger")
    cursor = run["source_global"] + 1
    total_phases = Counter()
    physics_ticks = completed = 0
    for index, episode in enumerate(episodes):
        require(isinstance(episode, dict), "invalid episode record")
        require(count(episode.get("episode_index"), "episode index") == index,
                "duplicate/gapped episode indices within one run")
        first = count(episode.get("first_global"), "episode first_global")
        last = count(episode.get("last_global"), "episode last_global")
        decisions = count(episode.get("decisions"), "episode decisions")
        require(decisions > 0 and first == cursor and last == first + decisions - 1,
                "credited episode intervals overlap, have gaps, or disagree with decisions")
        require(last <= run["end_global"], "episode extends beyond the completed block")
        phases = phase_counts(episode.get("phase_samples"), "episode phase_samples")
        require(sum(phases.values()) == decisions, "episode phase counts differ from credited decisions")
        total_phases.update(phases)
        ticks = count(episode.get("physics_ticks"), "episode physics ticks")
        require(decisions <= ticks <= decisions * 8, "invalid credited physics tick count")
        physics_ticks += ticks
        terminal = episode.get("terminal")
        require(type(terminal) is bool, "episode terminal flag is absent/nonboolean")
        if terminal:
            completed += 1
            require(isinstance(episode.get("termination_reason"), str) and episode["termination_reason"],
                    "completed episode lacks its actual termination reason")
            require(isinstance(episode.get("task_outcome_label"), str) and episode["task_outcome_label"],
                    "completed episode lacks its actual outcome label")
            require(type(episode.get("task_success")) is bool and type(episode.get("full_task_success")) is bool,
                    "completed episode lacks explicit task/full-task success booleans")
            if episode["task_success"]:
                require(episode["termination_reason"] == "SUCCESS", "task success contradicts terminal reason")
            if episode["full_task_success"]:
                require(episode["task_success"] and episode.get("scope") == "full_task"
                        and episode["task_outcome_label"] == "FULL_TASK_SUCCESS",
                        "suffix/prefix success must not be labelled full-task success")
        else:
            require(index == len(episodes) - 1 and last == run["end_global"],
                    "nonterminal episode is not the final completed-update tail")
            require(episode.get("termination_reason") is None and episode.get("task_outcome_label") is None
                    and episode.get("task_success") in (None, False)
                    and episode.get("full_task_success") in (None, False),
                    "nonterminal tail is incorrectly assigned a failure/success outcome")
        cursor = last + 1
    require(cursor == run["end_global"] + 1, "episode ledger does not cover the complete block")
    require(count(run.get("completed_episode_count"), "completed_episode_count") == completed,
            "completed episode count includes a tail or omits a terminal episode")
    require(count(run.get("completed_episode_rows_outside_boundary"), "outside-boundary episodes") == 0,
            "summary contains completed episodes outside its declared boundary")
    require(physics_ticks == count(run["totals"].get("physics_ticks"), "run physics ticks"),
            "episode and run physics totals disagree")
    return dict(total_phases)


def validate_block(summary, coverage, index, expected_source, expected_decisions=1024):
    require(summary.get("schema") == SCHEMA, "unrecognized completed-block summary schema")
    runs = summary.get("runs")
    require(isinstance(runs, list) and len(runs) == 1, "each block directory must contain exactly one run")
    run = runs[0]
    require(isinstance(run, dict) and isinstance(run.get("run"), str) and run["run"], "missing run identity")
    require(run.get("lifecycle") in FINAL_LIFECYCLES, f"block {index} is not finalized")
    source = count(run.get("source_global"), "source_global")
    end = count(run.get("end_global"), "end_global")
    decisions = count(run.get("actual_decisions"), "actual_decisions")
    require(source == expected_source and end == source + expected_decisions
            and decisions == expected_decisions, f"block {index}: gap/overlap/wrong actual interval")
    require(count(run.get("planned_decisions"), "planned_decisions") == decisions
            and count(run.get("unconsumed_decisions"), "unconsumed_decisions") == 0
            and count(run.get("rounding_overrun"), "rounding_overrun") == 0,
            f"block {index} did not consume its complete declared allocation")
    require(count(run.get("updates"), "updates") == decisions // 128
            and count(run.get("optimizer_steps"), "optimizer_steps") == (decisions // 128) * 20,
            "actual PPO/optimizer counts do not match the completed 128-transition update blocks")
    totals = run.get("totals")
    require(isinstance(totals, dict), "missing stored run totals")
    require(count(totals.get("optimized_decisions"), "optimized_decisions") == decisions,
            "optimized transitions differ from the credited interval")
    episode_phases = validate_episodes(run)
    checkpoints = run.get("checkpoints")
    require(isinstance(checkpoints, list) and checkpoints, "missing stored checkpoint publication receipts")
    saved = [count(row.get("global_policy_decisions"), "saved global") for row in checkpoints]
    require(saved == sorted(set(saved)) and saved[-1] == end and all(source < value <= end for value in saved),
            "stored checkpoint receipts do not end at the complete block boundary")
    # Paths are preserved as receipts, never opened or mistaken for new tensor verification.
    require(all(isinstance(row.get("checkpoint"), str) and isinstance(row.get("manifest"), str)
                for row in checkpoints), "incomplete stored checkpoint receipt")
    require(coverage.get("header") == HEADER, "coverage header changed; refusing positional reinterpretation")
    rows = coverage.get("rows")
    require(isinstance(rows, list) and len(rows) == len(PHASES), "missing/duplicate phase coverage rows")
    name = run["run"].replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    sums, seen = Counter(), set()
    for values in rows:
        require(isinstance(values, list) and len(values) == len(HEADER), "malformed coverage row")
        row = dict(zip(HEADER, values))
        phase = row["source_phase"]
        require(row["run"] == name and row["source_global"] == source and row["end_global"] == end,
                "coverage row belongs to a different run or interval")
        require(phase in PHASES and phase not in seen, "unknown/duplicated source phase")
        seen.add(phase)
        for key in HEADER[4:-1]:
            sums[key] += count(row[key], f"coverage {phase}.{key}")
        require(type(row["sampled"]) is bool and row["sampled"] == (row["optimized_decisions"] > 0),
                "coverage sampled flag contradicts actual policy count")
        require(row["optimized_decisions"] == episode_phases[phase], "coverage and episode phase counts disagree")
        require(row["own_phase_effect_ticks"] <= row["native_effect_ticks"] <= row["physics_ticks"],
                "coverage native-effect count exceeds observed credited ticks")
    for key in ("optimized_decisions", "physics_ticks", "native_effect_ticks", "own_phase_effect_ticks",
                "native_audit_anomaly_rows", "state_write_anomaly_rows", "nonterminal_handoffs"):
        require(sums[key] == count(totals.get(key), f"run totals.{key}"), f"coverage/run total mismatch: {key}")
    return run, episode_phases


def build_outputs(root=ROOT, *, check=1):
    """No writes or external dereferences; validate all eight/six input documents first."""
    require(type(check) is int and check in CHECKS, "check must be 1 or 2")
    specification = CHECKS[check]
    source, final = specification["source"], specification["final"]
    allocations = specification["allocations"]
    runs, rows, receipts, limits = [], [], [], []
    policy_phases, outcomes, labels, scopes = Counter(), Counter(), Counter(), Counter()
    cursor, updates, optimizers, decisions = source, 0, 0, 0
    terminal_count = tail_count = tail_decisions = full_success = suffix_success = 0
    seen_runs = set()
    for index, expected_decisions in allocations:
        directory = root / f"block{index}_summary"
        summary, summary_receipt = read_document(directory / "training_block_summary.json")
        coverage, coverage_receipt = read_document(directory / "coverage_table.json")
        run, phases = validate_block(summary, coverage, index, cursor, expected_decisions)
        identity = run["run"].replace("\\", "/").casefold().rstrip("/")
        require(identity not in seen_runs, "one run was included more than once")
        seen_runs.add(identity)
        runs.append(run)  # Preserve ALL prefix/outcome/authority/unknown future fields.
        rows.extend(coverage["rows"])
        policy_phases.update(phases)
        cursor = run["end_global"]
        decisions += run["actual_decisions"]
        updates += run["updates"]
        optimizers += run["optimizer_steps"]
        receipts.append({"block": index, "summary": summary_receipt, "coverage": coverage_receipt,
            "summary_fields_except_runs": {key: value for key, value in summary.items() if key != "runs"},
            "coverage_fields_except_rows": {key: value for key, value in coverage.items() if key != "rows"}})
        for note in summary.get("limits", []):
            if note not in limits:
                limits.append(note)
        for episode in run["episodes"]:
            if not episode["terminal"]:
                tail_count += 1
                tail_decisions += episode["decisions"]
                continue
            terminal_count += 1
            outcomes[episode["termination_reason"]] += 1
            labels[episode["task_outcome_label"]] += 1
            scopes[f"{episode['scope']}::{episode['task_outcome_label']}"] += 1
            full_success += int(episode["full_task_success"])
            suffix_success += int(episode["task_success"] and not episode["full_task_success"])
    require(cursor == final and decisions == final - source == 4096 and updates == 32 and optimizers == 640,
            f"check {check}: final boundary or actual update/optimizer totals mismatch")
    require(sum(policy_phases.values()) == decisions, "aggregate policy phase total mismatch")
    limits += [f"Only the {2 * len(allocations)} published block JSON summaries were read; their input SHA256 values bind this aggregation.",
               "Run records are preserved verbatim, including prefix exclusions, unavailable authority fields and failures.",
               "Nonterminal update-boundary tails are retained observations, not completed failures or successes.",
               "Checkpoint publication is stored-summary evidence only; no checkpoint, sidecar or raw audit was reopened.",
               "Zero sampled decisions means unvisited coverage, not perfect phase quality or a task-success claim."]
    summary = {"schema": SCHEMA, "aggregation_schema": specification["aggregation_schema"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_global": source, "end_global": final, "actual_decisions": decisions,
        "updates": updates, "optimizer_steps": optimizers, "block_count": len(allocations),
        "policy_phase_counts": {phase: policy_phases[phase] for phase in PHASES},
        "outcome_counts": {"completed_episodes": terminal_count, "nonterminal_tails": tail_count,
            "nonterminal_tail_decisions": tail_decisions, "tails_counted_as_failures": False,
            "completed_by_termination_reason": dict(sorted(outcomes.items())),
            "completed_by_task_outcome_label": dict(sorted(labels.items())),
            "completed_by_scope_and_outcome": dict(sorted(scopes.items())),
            "full_task_successes": full_success, "suffix_task_successes": suffix_success,
            "completed_non_successes": terminal_count - full_success - suffix_success,
            "prefix_attempt_outcomes_included": False},
        "runs": runs, "source_block_documents": receipts, "limits": limits}
    coverage = {"header": HEADER, "rows": rows,
        "scope": f"{len(allocations)} completed optimized source-phase tables, preserved per run; teacher prefixes excluded. Unvisited quality unavailable.",
        "source_global": source, "end_global": final, "source_block_documents": receipts}
    return coverage, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=int, choices=sorted(CHECKS), default=1,
                        help="completed allocation group; default 1 preserves the first-4096 route")
    args = parser.parse_args()
    output_directory = ROOT / CHECKS[args.check]["output_directory"]
    destinations = [output_directory / "coverage_table.json", output_directory / "training_block_summary.json"]
    require(not any(path.exists() for path in destinations), "aggregate output already exists; refusing overwrite")
    documents = build_outputs(check=args.check)
    payloads = [json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2) + "\n" for doc in documents]
    output_directory.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        with ExitStack() as stack:
            streams = []
            for path in destinations:
                stream = stack.enter_context(path.open("x", encoding="utf-8"))
                created.append(path)
                streams.append(stream)
            for stream, payload in zip(streams, payloads):
                stream.write(payload)
    except BaseException:
        # Remove only exact output files this invocation successfully created;
        # an existing file that caused exclusive creation to fail is untouched.
        for path in created:
            path.unlink(missing_ok=True)
        raise
    print(json.dumps({"written": [str(path) for path in destinations],
                      "source_global": documents[1]["source_global"], "end_global": documents[1]["end_global"],
                      "actual_decisions": documents[1]["actual_decisions"], "updates": documents[1]["updates"],
                      "optimizer_steps": documents[1]["optimizer_steps"]}))


if __name__ == "__main__":
    main()

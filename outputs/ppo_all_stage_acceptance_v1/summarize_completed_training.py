"""New-namespace completed-block reporting; stdlib only, no PT/physics access.

Reuse the existing report parser, scanning each explicitly supplied FINAL run's
policy audit once. No polling, historical rescans, CSV, or runtime imports.
--run may be repeated; --output-dir must be a fresh new-experiment directory.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
TRAIN_ROOT = PROJECT / "runs/ppo_all_stage_acceptance_v1/train"
LEGACY = ROOT.parent / "ppo_transfer_roles_v1/summarize_completed_training.py"
COMMON_FIELDS = (
    "evaluator_version", "run_validity", "physical_evidence_status",
    "traversal_event_observed", "traversal_event_time_s", "traversal_task_complete",
    "task_completed_controlled", "strict_recovery_quality", "post_completion_loss_observed",
    "post_completion_observation_complete", "post_completion_elapsed_s",
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def inside(path, root):
    path, root = Path(path).resolve(), root.resolve()
    need(path == root or root in path.parents, f"path outside experiment scope: {path}")
    return path


def report_parser():
    # This is a new report-only module instance, never a production monkeypatch.
    spec = importlib.util.spec_from_file_location("all_stage_legacy_report_parser", LEGACY)
    need(spec is not None and spec.loader is not None, "existing report parser unavailable")
    module = importlib.util.module_from_spec(spec)
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True  # Do not create/update caches in the legacy output tree.
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    previous_observe = module.observe

    def observe(ep, info, global_id):
        previous_observe(ep, info, global_id)
        task = info["semantic_task"]
        physical = task["physical_evaluator"]
        ep["common_physical_result_at_end"] = {key: physical.get(key) for key in COMMON_FIELDS}
        ep["common_result_unavailable_fields"] = [key for key in COMMON_FIELDS if key not in physical]
        ep["termination_sources_at_end"] = {
            "supervisor": task.get("termination_source"),
            "physical_evaluator": physical.get("termination_source"),
        }
        ep["first_unfinished_stage_scope"] = (
            "First absent completed_stage_ids label; consult actual completion_values and "
            "common physical result, not an independent physical failure verdict."
        )

    module.observe = observe
    return module


def classify_sources(summary):
    counts = Counter()
    for ep in summary["episodes"]:
        start = ep.get("curriculum_start") or {}
        mode = start.get("mode")
        if mode == "teacher_initialized_suffix":
            source = "frozen_FSM_prefix_then_current_PPO"
            inherited_source = "frozen_FSM_prefix"
        elif mode == "checkpoint_policy_initialized_suffix":
            source = "frozen_checkpoint_prefix_then_current_PPO"
            inherited_source = "frozen_checkpoint_policy_prefix"
        elif mode == "fresh_P01_fallback":
            source = "fresh_natural_P01_after_failed_prefix"
            inherited_source = "before_credit_unattributed"
        elif not start and summary.get("sampling") in ("P01_only", "P01_full_task_only_initial_version"):
            source = "natural_P01_current_PPO"
            inherited_source = "before_credit_unattributed"
        else:
            source = "UNAVAILABLE_UNRECOGNIZED_SAMPLING"
            inherited_source = "prefix_or_before_credit_unattributed"
        ep["source_classification"] = source
        for event in ep["events"]:
            # Preserve the old credit classification and historical event ticks.
            event["source_classification"] = "current_PPO_policy" if event["credit"] == "policy" else inherited_source
            counts[f'{event["leg"]}:{event["kind"]}:{event["source_classification"]}'] += 1
    summary["event_counts_by_leg_and_source"] = dict(counts)
    summary["prefix_history_scope"] = (
        "Inherited histories observable after successful takeover are attributed by event tick. "
        "Compact prefix decision rows lack full I/Q/C/P history; failed-prefix complete event "
        "chains are UNAVAILABLE here, not zero. Prefix decisions remain excluded from PPO."
    )


def checkpoint_receipt(summary, parser):
    need(summary["checkpoints"], "final completed block has no checkpoint receipt")
    saved = summary["checkpoints"][-1]
    checkpoint = inside(saved["checkpoint"], ROOT)
    sidecar = inside(saved["manifest"], ROOT)
    need(checkpoint.is_file() and sidecar.is_file(), "recorded final checkpoint/sidecar unavailable")
    metadata = parser.read(sidecar)  # JSON only; never load/hash checkpoint tensors.
    need(saved["global_policy_decisions"] == metadata["global_policy_decisions"] == summary["end_global"],
         "final checkpoint decision boundary mismatch")
    need(metadata.get("save_load_round_trip") is True, "final checkpoint lacks verified save/load receipt")
    need(Path(metadata["checkpoint_path"]).resolve() == checkpoint, "sidecar/checkpoint path binding mismatch")
    updates, steps = metadata["ppo_updates"], metadata["optimizer_steps"]
    need(type(updates) is int and type(steps) is int
         and updates >= summary["updates"] and steps >= summary["optimizer_steps"], "invalid lifetime counters")
    ancestry = metadata.get("resume_ancestry") or {}
    source = {"global_policy_decisions": summary["source_global"],
              "ppo_updates": updates-summary["updates"], "optimizer_steps": steps-summary["optimizer_steps"]}
    for key, value in source.items():
        recorded = ancestry.get("source_"+key)
        need(recorded is None or recorded == value, f"resume ancestry mismatch: {key}")
    summary["final_checkpoint_receipt"] = {
        "checkpoint": str(checkpoint), "sidecar": str(sidecar),
        "checkpoint_sha256_recorded": metadata.get("checkpoint_sha256"),
        "global_policy_decisions": summary["end_global"], "ppo_updates": updates, "optimizer_steps": steps,
        "source_counters_from_final_minus_this_run": source,
        "save_load_round_trip": True,
        "scope": "Recorded save-time official reload evidence only; no new tensor/hash or natural-P01 evaluation.",
    }


def main():
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--run", action="append", type=Path, required=True)
    arguments.add_argument("--output-dir", type=Path, required=True)
    args = arguments.parse_args()
    runs = [inside(path, TRAIN_ROOT) for path in args.run]
    need(len(set(runs)) == len(runs), "duplicate run input")
    output = inside(args.output_dir, ROOT)
    paths = [output/"coverage_table.json", output/"training_block_summary.json"]
    need(not any(path.exists() for path in paths), "exclusive output exists; select a fresh directory")
    parser = report_parser()
    rows, summaries = [], []
    for run in runs:
        coverage, summary = parser.summarize(run)  # Existing finality/128/N1/telemetry checks; one audit pass.
        classify_sources(summary)
        checkpoint_receipt(summary, parser)
        rows.extend(coverage)
        summaries.append(summary)
    ordered = sorted(summaries, key=lambda row: row["source_global"])
    need(all(a["end_global"] <= b["source_global"] for a, b in zip(ordered, ordered[1:])), "overlapping credit ranges")
    documents = [
        {"header": parser.HEADER, "rows": rows, "scope": "Actual optimized source-phase samples; unvisited quality unavailable."},
        {"schema": "ppo_all_stage_acceptance_v1.completed_blocks.v1", "runs": summaries,
         "reused_parser": str(LEGACY), "actual_decisions": sum(row["actual_decisions"] for row in summaries),
         "limits": ["Stored evidence, not new physical verification or evaluation success.",
                    "No torch/PT/Isaac, raw rescans, CSV, or legacy report/runtime modification.",
                    "Planned/unconsumed quantities stay separate; optimized nonterminal tails are not completed episodes.",
                    "Legacy metric definitions unchanged; absent new result fields remain unavailable."]},
    ]
    payloads = [json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2)+"\n" for doc in documents]
    output.mkdir(parents=True, exist_ok=True)
    for path, payload in zip(paths, payloads):
        with path.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    print(json.dumps({"outputs": [str(path) for path in paths], "runs": len(runs),
                      "actual_decisions": documents[1]["actual_decisions"]}))


if __name__ == "__main__":
    main()

"""Thin new-revision media adapter; reuses verified decode/remux/pair exporter.

No differing-runtime pair waiver. No source changes, masks, or physics.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
AFTER_PATH = ROOT / "outputs/ppo_timing_task_priority_v1/after_media.py"
spec = importlib.util.spec_from_file_location("prior_after_media_reuse", AFTER_PATH)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
base = prior.base
SCOPE = "height-geometry-and-P02-recovery-v1"
LABELS = {"B0": "Zero HEIGHT", "C0": "PPO HEIGHT"}


def paired_runtime(left, right):
    for key in ("runtime_contract", "evaluation_configuration", "camera", "seed"):
        base.require(left.get(key) == right.get(key), f"Different {key}: refused paired claim; no runtime-delta waiver")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "compare", "self-test"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--role", choices=("B0", "C0"))
    parser.add_argument("--checkpoint-decisions", type=int)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-runtime-sha")
    parser.add_argument("--require-non-success", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "self-test":
        paired_runtime({"runtime_contract": {"x": 1}}, {"runtime_contract": {"x": 1}})
        try:
            paired_runtime({"runtime_contract": {"x": 1}}, {"runtime_contract": {"x": 2}})
        except RuntimeError:
            print(json.dumps({"self_test": "PASS", "different_runtime_rejected": True, "media_exported": False}))
            return
        raise AssertionError("different runtime accepted")
    base.require(args.output is not None and args.expected_head and args.expected_runtime_sha, "new output and actual reviewed runtime identity required")
    receipt = args.output.resolve().with_suffix(".media.json")
    base.require(not args.output.exists() and not receipt.exists(), "Never overwrite media or receipts")
    extra = {"revision_scope": SCOPE, "after_repair": True, "after_change_scope": SCOPE,
        "learning_gain_claim": False, "diagnostic_intervention": None,
        "export_adapter": str(Path(__file__).resolve()), "export_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reused_media_helper": str(prior.BASE_PATH), "reused_initial_pair_audit": str(AFTER_PATH)}
    if args.command == "export":
        base.require(args.source is not None and args.role is not None, "source/role required")
        source = base.read_json(args.source / "semantic_video_source_manifest.json")
        run = base.read_json(args.source.parent / "run_manifest.json")
        base.require(run.get("completed_at_utc") and run.get("lifecycle") not in (None, "STARTED", "RUNNING")
            and run["runtime_contract"] == source["runtime_contract"], "Source not a finalized runtime-bound run")
        prior.require_after_runtime(source, args.expected_head, args.expected_runtime_sha)
        if args.require_non_success:
            base.require(source.get("physical_task_success") is False, "Filename/result requested non-success; real source succeeded")
        proof = source.get("checkpoint_load_provenance")
        if args.role == "C0":
            base.require(args.checkpoint_decisions is not None and args.checkpoint_decisions > 0
                and isinstance(proof, dict) and proof.get("checkpoint_loaded_and_verified") is True
                and proof.get("saved_global_policy_decisions") == args.checkpoint_decisions,
                "C export must bind actual loaded checkpoint")
        else:
            base.require(proof is None and args.checkpoint_decisions is None, "Zero run cannot claim policy load")
        command = ["export", "--source", str(args.source), "--output", str(args.output), "--label", args.role]
        extra.update(label=LABELS[args.role], base_media_role=args.role,
            height_diagnostics_receipt=source.get("height_diagnostics"), reviewed_runtime_binding={
                "source_git_commit": args.expected_head, "runtime_content_sha256": args.expected_runtime_sha})
    else:
        base.require(args.left is not None and args.right is not None, "Both new revision receipts required")
        items = [base.read_json(p) for p in (args.left, args.right)]
        for item, role in zip(items, ("B0", "C0")):
            base.require(item.get("revision_scope") == SCOPE and item.get("base_media_role") == role
                and item.get("label") == LABELS[role] and item.get("diagnostic_intervention") is None,
                "Not unmasked formal B/C height revision media")
            prior.require_after_runtime(item, args.expected_head, args.expected_runtime_sha)
        paired_runtime(*items)
        extra.update(label="Zero HEIGHT vs PPO HEIGHT", runtime_match_policy="exact runtime and evaluation only",
            initial_physical_state_equality=prior.initial_pair_audit(*items),
            paired_checkpoint_load_provenance={side: item.get("checkpoint_load_provenance") for side, item in zip(("left", "right"), items)})
        command = ["compare", "--left", str(args.left), "--right", str(args.right), "--output", str(args.output)]
    original_argv, original_read, original_run = sys.argv, base.read_json, base.run
    def read_adapter(path):
        item = original_read(path)
        if args.command == "compare" and Path(path).resolve() in (args.left.resolve(), args.right.resolve()):
            return {**item, "label": item["base_media_role"]}
        return item
    def caption_adapter(command):
        if "-filter_complex" in command:
            index = command.index("-filter_complex") + 1
            command[index] = command[index].replace("text='B0 |", "text='Zero HEIGHT |").replace("text='C0 |", "text='PPO HEIGHT |")
        return original_run(command)
    try:
        sys.argv = [str(prior.BASE_PATH), *command]
        base.read_json, base.run = read_adapter, caption_adapter
        base.main()  # Unchanged complete decode, exact PTS/ledger identity, 15fps, <=200s, threads<=2.
    finally:
        sys.argv, base.read_json, base.run = original_argv, original_read, original_run
    prior.annotate_new_receipt(receipt, extra)  # Only receipt newly owned by this invocation.
    print(json.dumps({"receipt": str(receipt), "scope": SCOPE, "label": extra["label"]}))


if __name__ == "__main__":
    main()

"""Thin AFTER adapter over the already verified diagnostic media exporter.

Only new source-bound files; no old helper/source/video is edited. Runtime
identity must be supplied from the actual reviewed post-change run, not guessed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "outputs/ppo_rr_video_diagnosis_v1/diagnostic_media.py"
BEFORE_HEAD = "4b2c038887c4109c639eea720f9e60de2c6d8d93"
spec = importlib.util.spec_from_file_location("existing_diagnostic_media", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
LABELS = {"B0": "Zero AFTER", "C0": "PPO AFTER"}
REWARD_PATH = "configs/ppo_fsm_reference_p09_stable_v2/reward_config.yaml"
REWARD_ONLY_FILES = {REWARD_PATH, *{f"src/wlr50_clean/ppo/{name}.py" for name in
    ("semantic_reward", "semantic_migration", "semantic_training")}}
EVAL_CONFIGS = {"action_schema_path": "action_schema.json", "execution_profile": "execution_profile.yaml",
    "observation_schema_path": "observation_schema.json", "quality_score_path": "quality_score.yaml",
    "reward_config_path": "reward_config.yaml", "task_spec_path": "stage_task_spec.yaml"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require_after_runtime(item, head, runtime_sha):
    contract = item["runtime_contract"]
    base.require(len(head) == 40 and head != BEFORE_HEAD and len(runtime_sha) == 64,
                 "AFTER requires a reviewed changed revision, not the before revision")
    base.require(contract["source_git_commit"] == head and contract["runtime_content_sha256"] == runtime_sha,
                 "Actual source runtime does not equal the explicitly selected AFTER runtime")
    base.require(digest(contract["files"]) == runtime_sha, "Runtime file inventory digest is invalid")


def reward_only_contract_audit(left, right):
    """Compare exact recorded inventories; never waive physics/control changes."""
    a, b = left["runtime_contract"], right["runtime_contract"]
    base.require(a["files"].keys() == b["files"].keys(), "Reward-only cannot add/remove runtime files")
    differences = {path: {"left_sha256": a["files"][path], "right_sha256": b["files"][path]}
                   for path in a["files"] if a["files"][path] != b["files"][path]}
    required = {REWARD_PATH, "src/wlr50_clean/ppo/semantic_reward.py"}
    base.require(required <= differences.keys() <= REWARD_ONLY_FILES,
                 f"Not an explicit reward-only delta: {sorted(differences)}")
    variable = {"files", "source_git_commit", "runtime_content_sha256", "selected_configuration"}
    invariant = [{key: value for key, value in item.items() if key not in variable} for item in (a, b)]
    base.require(invariant[0] == invariant[1], "Reward-only changed other runtime metadata/rates/frozen A")
    for item in (left, right):
        contract, evaluation = item["runtime_contract"], item["evaluation_configuration"]
        selected = contract["selected_configuration"]
        base.require(set(selected) == set(EVAL_CONFIGS.values()) and set(evaluation) == set(EVAL_CONFIGS),
                     "Unknown selected/evaluation configuration key cannot be waived")
        for field, name in EVAL_CONFIGS.items():
            relative = f"configs/ppo_fsm_reference_p09_stable_v2/{name}"
            binding = evaluation[field]
            base.require(set(binding) == {"bytes", "path", "sha256"} and type(binding["bytes"]) is int
                         and binding["bytes"] > 0 and Path(binding["path"]).resolve() == (ROOT / relative).resolve()
                         and selected[name] == {"path": relative, "sha256": binding["sha256"]}
                         and contract["files"][relative] == binding["sha256"], "Configuration inventory binding differs")
    base.require({k:v for k,v in a["selected_configuration"].items() if k != "reward_config.yaml"}
                 == {k:v for k,v in b["selected_configuration"].items() if k != "reward_config.yaml"},
                 "Reward-only changed a nonreward selected configuration")
    evaluation_invariants = [{k:v for k,v in item["evaluation_configuration"].items() if k != "reward_config_path"}
                            for item in (left, right)]
    base.require(evaluation_invariants[0] == evaluation_invariants[1], "Nonreward evaluation configuration differs")
    # This honest invariant projection is ONLY an in-memory comparison input to
    # the reused encoder. Original source/individual receipts stay unchanged.
    projection = {**invariant[0], "files": {k:v for k,v in a["files"].items() if k not in differences},
        "selected_configuration": {k:v for k,v in a["selected_configuration"].items() if k != "reward_config.yaml"}}
    audit = {"different_file_hashes": differences,
        "reward_evaluation_bindings": {"left": left["evaluation_configuration"]["reward_config_path"],
                                       "right": right["evaluation_configuration"]["reward_config_path"]},
        "all_other_runtime_files_selected_configs_and_metadata_exact_equal": True,
        "evaluation_exception_key": "reward_config_path", "same_reward_claim": False,
        "same_actor_claim": False, "entire_runtime_equal_claim": False,
        "encoder_comparison_projection": "in-memory verified invariant fields only; no original receipt rewritten"}
    return audit, projection, evaluation_invariants[0]


def initial_pair_audit(left, right):
    spec = importlib.util.spec_from_file_location("initial_pair_extractor", ROOT / "outputs/ppo_rr_video_diagnosis_v1/rr_tracking_extract.py")
    tracking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracking)
    initial, evidence = [], []
    for item in (left, right):
        path = Path(item["source_manifest"])
        raw = path.read_bytes()
        base.require(hashlib.sha256(raw).hexdigest() == item["source_manifest_sha256"], "Source manifest no longer receipt-bound")
        source = json.loads(raw)
        for key in ("runtime_contract", "camera", "seed", "evaluation_configuration", "natural_reset_proof",
                    "reset_evidence", "checkpoint_load_provenance"):
            base.require(source.get(key) == item.get(key), f"Receipt differs from real source: {key}")
        source_dir, run = tracking.completed_source(path.parent)
        base.require(source_dir == path.parent and run["runtime_contract"] == item["runtime_contract"],
                     "Completed run runtime differs from source/receipt")
        physical_line, physical = next(tracking.lines(source_dir / "physical_observations.jsonl"))
        native_line, native = next(tracking.lines(source_dir / "native_tick_audit.jsonl"))
        tracking.physical_schema(physical); tracking.native_schema(native)
        base.require(physical["physics_tick"] == 0 and native["episode_physics_tick"] == 1, "No true reset tick0/pre-first-step evidence")
        initial.append(tracking.initial_state(physical, native))
        evidence.append({"source_manifest": str(path), "source_manifest_sha256": item["source_manifest_sha256"],
                         "physical_line": physical_line, "native_line": native_line})
    comparison = tracking.compare_initial(*initial)
    arrays = ("joint_position_canonical_deg", "joint_velocity_canonical_deg_s", "wheel_velocity_rad_s",
        "root_position_world_m", "root_quaternion_wxyz", "root_linear_velocity_world_m_s",
        "root_angular_velocity_world_rad_s", "com_position_world_m", "com_velocity_world_m_s",
        "joint_position_native_pre_first_step_rad", "standing_pose_deg")
    base.require(all(comparison.get(key, {}).get("exact_equal") is True for key in arrays)
                 and comparison["contact_pairs_exact_equal"] is True
                 and comparison["com_valid"] == {"zero": True, "ppo": True}, "Actual initial measurements are missing or differ")
    return {"measured_comparison": comparison, "evidence": evidence, "required_arrays": list(arrays),
            "all_initial_arrays_contacts_and_validity_exact_equal": True}


def annotate_new_receipt(path, extra):
    # This path was absent before the same invocation generated it. Existing
    # user/source receipts are never passed here, and the video is never edited.
    payload = base.read_json(path)
    payload.update(extra)
    pending = path.with_name(path.name + ".after.pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(pending, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "compare"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--role", choices=("B0", "C0"))
    parser.add_argument("--checkpoint-decisions", type=int)
    parser.add_argument("--left", type=Path)
    parser.add_argument("--right", type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-runtime-sha", required=True)
    parser.add_argument("--left-expected-head")
    parser.add_argument("--left-expected-runtime-sha")
    parser.add_argument("--allow-reward-only-runtime-delta", action="store_true")
    parser.add_argument("--change-scope", choices=("timing-only", "timing-and-reward"), default="timing-only")
    args = parser.parse_args(argv)
    receipt_path = args.output.resolve().with_suffix(".media.json")
    base.require(not args.output.exists() and not receipt_path.exists(), "AFTER output exists; never overwrite")
    if args.command == "export":
        base.require(not args.allow_reward_only_runtime_delta and args.left_expected_head is None
                     and args.left_expected_runtime_sha is None, "Pair-only flags cannot alter an individual export")
        base.require(args.source is not None and args.role is not None, "Export requires source and role")
        source = base.read_json(args.source / "semantic_video_source_manifest.json")
        require_after_runtime(source, args.expected_head, args.expected_runtime_sha)
        provenance = source.get("checkpoint_load_provenance")
        if args.role == "C0":
            base.require(args.checkpoint_decisions is not None and args.checkpoint_decisions > 0
                         and provenance is not None and provenance.get("checkpoint_loaded_and_verified") is True
                         and provenance["saved_global_policy_decisions"] == args.checkpoint_decisions,
                         "PPO AFTER must bind the actual verified checkpoint load")
        else:
            base.require(provenance is None and args.checkpoint_decisions is None, "Zero AFTER cannot claim a PPO checkpoint")
        command = ["export", "--source", str(args.source), "--output", str(args.output), "--label", args.role]
        extra = {"label": LABELS[args.role], "base_media_role": args.role, "after_repair": True,
                 "after_change_scope": args.change_scope, "learning_gain_claim": False}
    else:
        base.require(args.left is not None and args.right is not None, "Compare requires both AFTER receipts")
        if args.allow_reward_only_runtime_delta:
            base.require(args.left_expected_head is not None and args.left_expected_runtime_sha is not None,
                         "Reward-only pairing needs explicit LEFT and RIGHT runtime identities")
        else:
            base.require(args.left_expected_head is None and args.left_expected_runtime_sha is None,
                         "Different left binding requires the explicit reward-only audit flag")
        items = []
        for path, role in ((args.left, "B0"), (args.right, "C0")):
            item = base.read_json(path)
            base.require(item.get("after_repair") is True and item.get("base_media_role") == role
                         and item.get("label") == LABELS[role], "Comparison inputs must be explicitly labeled AFTER")
            left_override = role == "B0" and args.allow_reward_only_runtime_delta
            require_after_runtime(item, args.left_expected_head if left_override else args.expected_head,
                                  args.left_expected_runtime_sha if left_override else args.expected_runtime_sha)
            items.append(item)
        initial_audit = initial_pair_audit(*items)
        reward_audit = None
        if args.allow_reward_only_runtime_delta:
            reward_audit, runtime_projection, evaluation_projection = reward_only_contract_audit(*items)
        command = ["compare", "--left", str(args.left), "--right", str(args.right), "--output", str(args.output)]
        extra = {"label": "Zero AFTER vs PPO AFTER", "after_repair": True,
                 "runtime_match_policy": "explicit strict reward-only delta" if reward_audit else "exact runtime/evaluation match",
                 "reward_only_runtime_delta_audit": reward_audit, "initial_physical_state_equality": initial_audit,
                 "paired_runtime_bindings": {side: {key:item["runtime_contract"][key] for key in
                    ("source_git_commit", "runtime_content_sha256")} for side,item in zip(("left", "right"), items)},
                 "paired_checkpoint_load_provenance": {side:item.get("checkpoint_load_provenance")
                    for side,item in zip(("left", "right"), items)}, "learning_gain_claim": False}
    original_argv, original_read, original_run = sys.argv, base.read_json, base.run
    def role_adapter(path):
        item = original_read(path)
        if args.command == "compare" and Path(path).resolve() in (args.left.resolve(), args.right.resolve()):
            item = {**item, "label": item["base_media_role"]}
            if args.allow_reward_only_runtime_delta:
                item = {**item, "runtime_contract": runtime_projection, "evaluation_configuration": evaluation_projection}
        return item
    def caption_adapter(command):
        if "-filter_complex" in command:
            index = command.index("-filter_complex") + 1
            command[index] = command[index].replace("text='B0 |", "text='Zero AFTER |").replace("text='C0 |", "text='PPO AFTER |")
            if args.allow_reward_only_runtime_delta:
                command[index] = command[index].replace("1x real elapsed time - 15 Hz interval endpoints",
                    "1x elapsed time - different reward versions")
        return original_run(command)
    try:
        sys.argv = [str(BASE_PATH), *command]
        base.read_json, base.run = role_adapter, caption_adapter
        base.main()  # Existing full decode, ledger/PTS, static-tail and CPU-thread limits.
    finally:
        sys.argv, base.read_json, base.run = original_argv, original_read, original_run
    annotate_new_receipt(receipt_path, {**extra, "after_runtime_binding": {
        "source_git_commit": args.expected_head, "runtime_content_sha256": args.expected_runtime_sha},
        "export_adapter": str(Path(__file__).resolve()), "reused_media_helper": str(BASE_PATH)})
    print(json.dumps({"after_receipt": str(receipt_path), "label": extra["label"], "after_repair": True}))


if __name__ == "__main__":
    main()

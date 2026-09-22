"""Isolated post-deployment adapter for sigma-profile media and receipts.

Importing this module performs no encoding, training, checkpoint loading, or
simulation.  Every formal operation first calls the official sigma-plan and
historical LIMITED AUX validator through ``receiving_wheel_provenance``.
"""
from __future__ import annotations

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from receiving_wheel_provenance import (
    OUT, ROOT, TARGET_POLICY, require, sha, validate_checkpoint,
)


AUX = OUT / "candidate/aux_method_label"
AUX_MEDIA_HASH = "815e4dab5863b3e53c9ccbc26ee21e6a217244a674cad28d12ccd5f013ce2bdb"
TRAINING_RECEIPT_HASH = "c8c7d84e6fdc34253b381d53d24cc46fd8e9dc9f25ad68978d8ed858fb3d4466"
SCHEMA = "wlr50_clean.task_conditioned_limited_aux_receiving_wheel_review.v1"


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def _limited_aux_module(historical_runtime_root: Path):
    import sys
    if str(AUX) not in sys.path:
        sys.path.insert(0, str(AUX))
    path = AUX / "limited_aux_media.py"
    require(sha(path) == AUX_MEDIA_HASH,
            "LIMITED AUX adapter changed; sigma media needs explicit re-review")
    result = module("receiving_wheel_limited_aux_media", path)
    # These are process-local adapter bindings, not edits or a version allowlist.
    result.r.POLICY_VERSION = TARGET_POLICY
    result.SCHEMA = SCHEMA
    result.validate_auxiliary_checkpoint = lambda checkpoint, receipt: validate_checkpoint(
        checkpoint, receipt, historical_runtime_root=historical_runtime_root)
    return result


def _archived_quantity_control(b, c, identity):
    """Apply the old comparison contract to an already archived-verified boundary."""
    require(b["receipt"].get("role") == "B" and b["receipt"].get("mode") == "N_plus_zero"
            and c["receipt"].get("role") == "C"
            and c["receipt"].get("mode") == "deterministic_conditional_mean",
            "quantity pair requires B zero and deterministic C")
    proof = identity.get("historical_quantity_boundary") or {}
    require(proof.get("schema") == "wlr50_clean.historical_quantity_aux_posthoc.v1"
            and proof.get("posthoc_only_no_training") is True,
            "pair lacks the isolated archived quantity proof")
    a, z = b["manifest"], c["manifest"]
    require(a.get("experiment_id") == z.get("experiment_id") == "task_conditioned_hip_wheel_v1"
            and a.get("camera") is not None and a["camera"] == z.get("camera")
            and a.get("seed") == z.get("seed") == 4001,
            "quantity pair experiment/camera/scene seed differs")
    entry = a.get("natural_reset_proof", {}).get("entry")
    require(entry is not None and entry == z.get("natural_reset_proof", {}).get("entry"),
            "quantity pair natural reset entry differs")
    source_identity = identity["sigma_source_checkpoint_identity"]
    require(a.get("runtime_contract") == proof.get("quantity_source_runtime_contract")
            and source_identity.get("runtime_contract") == proof.get("quantity_target_runtime_contract")
            and a.get("evaluation_configuration") == proof.get("source_evaluation_configuration")
            and z.get("evaluation_configuration") == proof.get("target_evaluation_configuration"),
            "B/C evaluation metadata does not match the archived quantity boundary")
    return {"same_control_via_archived_verified_quantity_only_boundary": True,
        "same_runtime_contract": False,
        "source_B_runtime_contract": proof["quantity_source_runtime_contract"],
        "sigma_source_runtime_contract": proof["quantity_target_runtime_contract"],
        "historical_runtime": proof["historical_runtime"],
        "quantity_plan": proof["quantity_plan"],
        "quantity_plan_sha256": proof["quantity_plan_sha256"],
        "same_camera": a["camera"], "same_scene_seed": 4001,
        "same_natural_reset_entry": entry,
        "current_sigma_runtime_verified_separately": True,
        "single_pair_is_not_statistical_evidence": True}


def make_adapter(auxiliary_receipt: Path, historical_runtime_root: Path):
    """Build a lazy adapter; actual calls still fail until official deployment."""
    limited = _limited_aux_module(Path(historical_runtime_root))
    base = limited.make_adapter(Path(auxiliary_receipt))

    def pair(b_path: Path, c_path: Path, destination: Path):
        b = limited.r.checked_review(b_path)
        c = base.checked_review(c_path)
        identity = c["receipt"]["learning_method"]
        require(identity.get("exploration_profile", {}).get("target_policy_version") == TARGET_POLICY,
                "C receipt lacks the exact officially validated sigma identity")
        # The old official builder was already re-run in its real historical
        # checkout. The sigma source is its quantity target; C is boundary two.
        quantity = _archived_quantity_control(b, c, identity)
        comparison = identity["comparison_contract"]
        require(comparison == {
            "quantity_boundary_and_sigma_boundary_are_distinct": True,
            "same_N": 1,
            "same_physics_and_MDP": True,
            "same_nominal_mapper_evaluator_reward_control_chain": True,
            "same_policy_distribution": False,
            "deterministic_and_stochastic_C_must_use_same_checkpoint": True,
            "old_B_must_not_be_relabelled_as_target_sigma_profile": True,
        }, "sigma comparison contract changed")
        destination = Path(destination).resolve()
        require(destination.is_relative_to(OUT) and destination != OUT,
                "use a new isolated receiving-wheel pair directory")
        destination.mkdir(parents=True, exist_ok=False)
        encoder = module("receiving_wheel_encoder_only",
                         ROOT / "outputs/ppo_fl_capture_quality_v1/paired_event_media.py")
        cp = c["receipt"]["checkpoint_identity"]["saved_global_policy_decisions"]
        full = encoder._comparison_video(
            b, c, destination / f"N_vs_CP{cp}_PPO_PLUS_LIMITED_AUX_receiving_wheel_sigma_x3.mp4",
            left_end=b["receipt"]["frame_count"],
            right_end=c["receipt"]["frame_count"],
            kind="full_attempt_same_elapsed_P01_quantity_then_exact_sigma_profile",
        )
        result = {
            "schema": "wlr50_clean.task_conditioned_limited_aux_receiving_wheel_pair.v1",
            "B_receipt": str(b["receipt_path"]),
            "C_receipt": str(c["receipt_path"]),
            "verified_quantity_boundary": quantity,
            "verified_sigma_boundary": identity["exploration_profile"],
            "learning_method": identity,
            "full_episode": full,
            "B_physical_result": b["receipt"]["physical_result"],
            "C_physical_result": c["receipt"]["physical_result"],
            "quality_improvement_claim": None,
            "same_nominal_mapper_evaluator_reward_control_chain": True,
            "same_policy_distribution": False,
            "B_not_relabelled_as_target_sigma_profile": True,
            "task_success_not_inferred_from_media": True,
            "shorter_side_freeze_is_not_new_physics": True,
        }
        limited.r.base.write_new_json(destination / "pair_receipt.json", result)
        return result

    def modes(deterministic_path: Path, stochastic_path: Path, output: Path):
        deterministic = base.checked_review(deterministic_path)
        stochastic = base.checked_review(stochastic_path)
        common = limited.r.strict_policy_modes(deterministic, stochastic)
        method = deterministic["receipt"]["learning_method"]
        require(method == stochastic["receipt"]["learning_method"],
                "deterministic/stochastic captures use different sigma lineages")
        require(deterministic["receipt"]["checkpoint_identity"]
                == stochastic["receipt"]["checkpoint_identity"],
                "deterministic/stochastic captures must use the same checkpoint")
        output = Path(output).resolve()
        require(output.is_relative_to(OUT), "mode receipt must remain inside outputs")
        result = {
            "schema": "wlr50_clean.task_conditioned_limited_aux_receiving_wheel_mode_pair.v1",
            **common,
            "learning_method": method,
            "deterministic_receipt": str(deterministic["receipt_path"]),
            "stochastic_receipt": str(stochastic["receipt_path"]),
            "same_checkpoint_required": True,
        }
        limited.r.base.write_new_json(output, result)
        return result

    return SimpleNamespace(export=base.export, checked_review=base.checked_review,
                           pair=pair, modes=modes)


def make_training_summarizer(auxiliary_receipt: Path, historical_runtime_root: Path):
    """Adapt the frozen post-hoc receipt without weakening its policy check."""
    path = OUT / "training_receipt.py"
    require(sha(path) == TRAINING_RECEIPT_HASH,
            "training receipt changed; sigma receipt needs explicit re-review")
    original = module("receiving_wheel_original_training_receipt", path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = copy.deepcopy(next(node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "summarize"))
    namespace = dict(vars(original))
    namespace["POLICY"] = TARGET_POLICY
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 "<receiving-wheel-training-receipt>", "exec"), namespace)
    strict_summarize = namespace["summarize"]

    def summarize(run: Path):
        result = strict_summarize(run)
        final = Path(result["final_checkpoint"]["path"])
        method = validate_checkpoint(final, Path(auxiliary_receipt),
            historical_runtime_root=historical_runtime_root)
        require(result["new_ppo_updates"] > 0,
                "sigma training receipt requires fresh PPO after the migration boundary")
        result.update(
            schema="wlr50_clean.task_conditioned_receiving_wheel_sealed_training_receipt.v1",
            policy_version=TARGET_POLICY,
            learning_method=method,
            receipt_is_post_hoc_not_optimizer_prerequisite=True,
        )
        return result
    return summarize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aux-receipt", type=Path, required=True)
    parser.add_argument("--historical-runtime-root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--source", type=Path, required=True)
    export.add_argument("--destination", type=Path, required=True)
    pair = sub.add_parser("pair")
    pair.add_argument("--b-receipt", type=Path, required=True)
    pair.add_argument("--c-receipt", type=Path, required=True)
    pair.add_argument("--destination", type=Path, required=True)
    modes = sub.add_parser("modes")
    modes.add_argument("--deterministic-receipt", type=Path, required=True)
    modes.add_argument("--stochastic-receipt", type=Path, required=True)
    modes.add_argument("--output", type=Path, required=True)
    training = sub.add_parser("training")
    training.add_argument("--run", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "training":
        result = make_training_summarizer(
            args.aux_receipt, args.historical_runtime_root)(args.run)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    else:
        adapter = make_adapter(args.aux_receipt, args.historical_runtime_root)
        result = (adapter.export(args.source, args.destination) if args.command == "export" else
                  adapter.pair(args.b_receipt, args.c_receipt, args.destination)
                  if args.command == "pair" else
                  adapter.modes(args.deterministic_receipt, args.stochastic_receipt, args.output))
    print(json.dumps(result, indent=2))

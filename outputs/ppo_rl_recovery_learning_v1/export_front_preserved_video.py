"""Export a sealed CP225280-front-preservation evaluation, outputs-only.

This narrow adapter accepts the explicit ``cp225280_front_preserved_v1``
sibling lineage and then delegates rendering/QA to
``export_rl_recovery_video.py``.  It never imports Torch or simulator code.
Older owner439/reward/65a/collection identities remain owned by the original
exporter and are not reinterpreted here.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent
CORE_PATH = OUTPUT_ROOT / "export_rl_recovery_video.py"

SCHEMA = "wlr50_clean.cp225280_front_preserved439.v1"
IDENTITY = "front_preservation439_branch_identity"
COUNTS_KEY = "front_preservation439_branch_counts"
REPLAY_COUNTS_KEY = "front_replay_counts"
BRANCH = "cp225280_front_preserved_v1"
SOURCE_HEAD = "f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b"
SOURCE_SHA = "fbe28e3718a5796afc2271e7b10aa04017369593b5c62e0bde1973a20f68032f"
SOURCE_MANIFEST_SHA = "f501b7a735c0aaa42be00114e09f80d46f7f009d7717aa416fc30ad3cc837d6d"
ORIGIN = {"global_policy_decisions": 225280,
          "ppo_updates": 1725, "optimizer_steps": 34500}
POLICY = "rear_owner_recovery_history_v1"
LAYOUT = "role422_rear_owner_recovery_v1"
COLLECTION_PROFILE = "n1_rear_owner439_collection512_v1"
REPLAY_SCHEMA = "wlr50_clean.front_replay_regularizer.v1"
REPLAY_VERSION = "cp225280_front_gaussian_kl_v1"
DATASET_SCHEMA = "wlr50_clean.cp225280_front_replay.v1"
FORBIDDEN = ("rr_retention_reward_migration", "front_retention439_runtime_identity",
             "front_retention439_auxiliary", "collection_horizon439")
CODE = "src/wlr50_clean/ppo/"
ALLOWED_RUNTIME_PATHS = frozenset(CODE + name for name in (
    "semantic_reward.py", "semantic_rr_retention_migration.py",
    "semantic_video_cli.py", "semantic_policy_distribution.py",
    "semantic_training.py", "semantic_rear_policy_timing_migration.py",
    "semantic_return_profile.py", "semantic_front_retention439.py",
    "semantic_cli.py", "semantic_migration.py",
    "semantic_front_preservation.py", "semantic_front_replay.py"))
REQUIRED_NEW_PATHS = frozenset((CODE + "semantic_front_preservation.py",
                                CODE + "semantic_front_replay.py"))


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("front-preservation media dependency cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load(CORE_PATH, "_front_preserved_media_core")
require = CORE.require
ORIGINAL_CHECKPOINT_IDENTITY = CORE.checkpoint_identity


def _checked_path_hash(path_value: Any, sha_value: Any, label: str) -> Path:
    require(isinstance(path_value, (str, Path)) and str(path_value),
            label + " path is required")
    path = Path(path_value).resolve(strict=True)
    expected = CORE.BASE.checked_sha(sha_value, label)
    require(CORE.BASE.sha256(path) == expected, label + " bytes differ")
    return path


def _counter_delta(counters: dict[str, int]) -> dict[str, int]:
    require(all(type(counters.get(key)) is int and counters[key] >= ORIGIN[key]
                for key in CORE.BASE.COUNTERS),
            "front-preservation counters are missing or rolled back")
    updates = counters["ppo_updates"] - ORIGIN["ppo_updates"]
    require(counters["global_policy_decisions"] - ORIGIN["global_policy_decisions"] ==
                512 * updates and
            counters["optimizer_steps"] - ORIGIN["optimizer_steps"] == 20 * updates,
            "front-preservation descendants require exact +512/+1/+20 accounting")
    return {key: counters[key] - ORIGIN[key] for key in CORE.BASE.COUNTERS}


def _expected_replay_counts(updates: int) -> dict[str, int]:
    require(type(updates) is int and updates >= 0,
            "front replay update count is invalid")
    return {"ppo_updates_with_replay": updates,
            "optimizer_minibatches": 20 * updates,
            "replay_row_exposures": 640 * updates,
            "on_policy_samples_added": 0,
            "separate_auxiliary_optimizer_steps": 0}


def _runtime_source(runtime: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    source = receipt.get("source_manifest") or {}
    old = source.get("runtime_contract") or {}
    changed = receipt.get("changed_file_hashes")
    require(isinstance(changed, dict), "front-preservation changed-file binding is missing")
    actual_changed = {path for path in set(old.get("files") or {}) |
                      set(runtime.get("files") or {})
                      if (old.get("files") or {}).get(path) !=
                         (runtime.get("files") or {}).get(path)}
    require(actual_changed == set(changed) and
            actual_changed <= ALLOWED_RUNTIME_PATHS and
            REQUIRED_NEW_PATHS <= actual_changed,
            "front-preservation runtime delta is not the reviewed narrow set")
    for path in actual_changed:
        require(changed[path] == {
                    "before": (old.get("files") or {}).get(path),
                    "after": (runtime.get("files") or {}).get(path)},
                "front-preservation changed-file hash differs: " + path)
    variable = {"files", "source_git_commit", "runtime_content_sha256"}
    require({key: value for key, value in old.items() if key not in variable} ==
            {key: value for key, value in runtime.items() if key not in variable} and
            old.get("source_git_commit") == SOURCE_HEAD and
            CORE.BASE.json_digest(old.get("files") or {}) ==
                old.get("runtime_content_sha256") and
            CORE.BASE.json_digest(runtime.get("files") or {}) ==
                runtime.get("runtime_content_sha256"),
            "front-preservation changed configuration/physics/task contract")
    return old


def _validate_dataset(spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    require(isinstance(spec, dict) and spec.get("schema") == REPLAY_SCHEMA and
            spec.get("version", REPLAY_VERSION) == REPLAY_VERSION and
            spec.get("coefficient") == 1.0 and
            spec.get("minibatch_size") == 32,
            "front replay spec is not the reviewed coefficient1/batch32 contract")
    dataset = _checked_path_hash(args.front_replay_dataset,
                                 args.front_replay_dataset_sha256,
                                 "--front-replay-dataset-sha256")
    require(Path(spec.get("dataset_path", "")).resolve(strict=True) == dataset and
            spec.get("dataset_sha256") == args.front_replay_dataset_sha256,
            "front replay spec differs from the explicit dataset binding")
    payload = CORE.BASE.read_json(dataset)
    source = payload.get("source_binding") or {}
    reviewed = source.get("reviewed_f6d_tensor_identical_migration_target") or {}
    require(payload.get("schema") == DATASET_SCHEMA and
            reviewed.get("checkpoint_sha256") == SOURCE_SHA and
            reviewed.get("manifest_sha256") == SOURCE_MANIFEST_SHA and
            reviewed.get("runtime_head") == SOURCE_HEAD and
            reviewed.get("counters") == ORIGIN and
            source.get("policy_version") == POLICY and
            source.get("observation_layout") == LAYOUT,
            "front replay dataset is not bound to reviewed f6d439 CP225280")
    rows = list(payload.get("training_rows") or []) + list(payload.get("heldout_rows") or [])
    require(0 < len(rows) <= 256 and
            all(row.get("phase") in {f"P{index:02d}" for index in range(1, 7)} and
                len(row.get("observation") or []) == 439 and
                len(row.get("reference_mean") or []) == 12 and
                len(row.get("reference_sigma") or []) == 12 and
                all(type(value) in (int, float) and math.isfinite(value)
                    for field in ("observation", "reference_mean", "reference_sigma")
                    for value in row[field])
                for row in rows),
            "front replay dataset row shape/phase/value contract differs")
    return {"path": str(dataset), "sha256": args.front_replay_dataset_sha256,
            "training_rows": len(payload.get("training_rows") or []),
            "heldout_rows": len(payload.get("heldout_rows") or [])}


def _source_identity(receipt: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    binding = receipt.get("source_checkpoint") or {}
    source_checkpoint = _checked_path_hash(binding.get("checkpoint"), SOURCE_SHA,
                                           "source checkpoint")
    source_manifest = _checked_path_hash(binding.get("manifest"), SOURCE_MANIFEST_SHA,
                                         "source manifest")
    source = CORE.BASE.read_json(source_manifest)
    require(binding == {"checkpoint": str(source_checkpoint), "checkpoint_sha256": SOURCE_SHA,
                        "manifest": str(source_manifest),
                        "manifest_sha256": SOURCE_MANIFEST_SHA} and
            receipt.get("source_manifest") == source and
            receipt.get("source_manifest_content_sha256") == CORE.BASE.json_digest(source),
            "front-preservation receipt does not embed the exact f6d source")
    old_args = copy.copy(args)
    old_args.checkpoint = source_checkpoint
    old_args.checkpoint_sha256 = SOURCE_SHA
    old_args.checkpoint_manifest_sha256 = SOURCE_MANIFEST_SHA
    old_args.expected_head = SOURCE_HEAD
    old_args.expected_global_policy_decisions = ORIGIN["global_policy_decisions"]
    old_args.expected_ppo_updates = ORIGIN["ppo_updates"]
    old_args.expected_optimizer_steps = ORIGIN["optimizer_steps"]
    old_args.checkpoint_runtime_head = None
    old_args.rr_retention_publication = None
    old_args.rr_retention_publication_sha256 = None
    old_args.rr_retention_plan_sha256 = None
    old_args.front_retention439_publication = None
    old_args.front_retention439_publication_sha256 = None
    old_args.front_retention439_plan_sha256 = None
    old_args.collection512_publication = None
    old_args.collection512_publication_sha256 = None
    proof = {"policy_sampling_mode": "deterministic_conditional_mean",
        "runtime_contract": source["runtime_contract"],
        "checkpoint_load_provenance": {
            "checkpoint_loaded_and_verified": True, "stochastic_policy": False,
            "policy_seed": None, "saved_global_policy_decisions": ORIGIN["global_policy_decisions"],
            "checkpoint_runtime_compatibility": None,
            "parameter_hashes": {"actor_parameter_sha256": source["actor_parameter_sha256"]},
            "source": copy.deepcopy(binding)}}
    return ORIGINAL_CHECKPOINT_IDENTITY(proof, old_args)


def _validate_publication(receipt: dict[str, Any], runtime: dict[str, Any],
                          route: dict[str, Any], replay_spec: dict[str, Any],
                          args: argparse.Namespace) -> dict[str, Any]:
    path = _checked_path_hash(args.front_preservation_publication,
                              args.front_preservation_publication_sha256,
                              "--front-preservation-publication-sha256")
    payload = CORE.BASE.read_json(path)
    checkpoint = _checked_path_hash(payload.get("checkpoint"),
                                    payload.get("checkpoint_sha256"),
                                    "published checkpoint")
    manifest = _checked_path_hash(payload.get("manifest"),
                                  payload.get("manifest_sha256"),
                                  "published manifest")
    metadata = CORE.BASE.read_json(manifest)
    require(payload.get("save_load_round_trip") is True and
            all(payload.get("added_" + name) == 0 for name in
                ("policy_decisions", "ppo_updates", "optimizer_steps", "auxiliary_updates")) and
            {key: payload.get(key) for key in CORE.BASE.COUNTERS} == ORIGIN and
            metadata.get(IDENTITY) == receipt and
            metadata.get(COUNTS_KEY) == _counter_delta(ORIGIN) and
            metadata.get(REPLAY_COUNTS_KEY) == _expected_replay_counts(0) and
            metadata.get("runtime_contract") == runtime and
            metadata.get("checkpoint_output_routing") == route and
            metadata.get("policy_contract") == receipt.get("target_policy_contract") and
            payload.get("replay_spec") == replay_spec,
            "front-preservation zero-credit publication or reload receipt differs")
    return {"path": str(path), "sha256": args.front_preservation_publication_sha256,
            "checkpoint": str(checkpoint), "checkpoint_sha256": payload["checkpoint_sha256"],
            "manifest": str(manifest), "manifest_sha256": payload["manifest_sha256"]}


def _validate_last_replay(metadata: dict[str, Any], spec: dict[str, Any],
                          updates: int) -> None:
    if updates == 0:
        return
    report = (metadata.get("last_update") or {}).get("front_replay_regularization") or {}
    minibatches = report.get("minibatches")
    require(report.get("schema") == REPLAY_SCHEMA and report.get("spec") == spec and
            isinstance(minibatches, list) and len(minibatches) == 20 and
            report.get("actual_replay_row_exposures") == 640 and
            report.get("on_policy_samples_added") == 0 and
            report.get("separate_auxiliary_optimizer_steps") == 0 and
            report.get("realtime_teacher_deployed") is False and
            all(batch.get("gradient_consumed_once") is True and
                batch.get("on_policy_samples_added") == 0 and
                batch.get("separate_optimizer_steps") == 0
                for batch in minibatches),
            "latest PPO update lacks its actual20/640 front replay evidence")


def checkpoint_identity(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    proof = manifest.get("checkpoint_load_provenance") or {}
    binding = proof.get("source") or {}
    checkpoint = Path(binding.get("checkpoint", "")).resolve(strict=True)
    expected_checkpoint = Path(args.checkpoint).resolve(strict=True)
    sidecar = Path(binding.get("manifest", "")).resolve(strict=True)
    expected_sidecar = expected_checkpoint.with_name(
        expected_checkpoint.stem + "_manifest.json").resolve(strict=True)
    metadata = CORE.BASE.read_json(expected_sidecar)
    if IDENTITY not in metadata:
        require(getattr(args, "front_preservation_publication", None) is None and
                getattr(args, "front_preservation_publication_sha256", None) is None and
                getattr(args, "front_replay_dataset", None) is None and
                getattr(args, "front_replay_dataset_sha256", None) is None,
                "non-front-preservation checkpoints must not carry its explicit bindings")
        return ORIGINAL_CHECKPOINT_IDENTITY(manifest, args)
    require(proof.get("checkpoint_loaded_and_verified") is True and
            manifest.get("policy_sampling_mode") == "deterministic_conditional_mean" and
            proof.get("stochastic_policy") in (None, False) and
            proof.get("policy_seed") is None and
            proof.get("checkpoint_runtime_compatibility") is None and
            checkpoint == expected_checkpoint and sidecar == expected_sidecar,
            "front-preservation video is not an exact-runtime deterministic checkpoint load")
    checkpoint_sha = CORE.BASE.checked_sha(args.checkpoint_sha256,
                                            "--checkpoint-sha256")
    sidecar_sha = CORE.BASE.checked_sha(args.checkpoint_manifest_sha256,
                                       "--checkpoint-manifest-sha256")
    require(CORE.BASE.sha256(checkpoint) == checkpoint_sha ==
                binding.get("checkpoint_sha256") == metadata.get("checkpoint_sha256") and
            CORE.BASE.sha256(sidecar) == sidecar_sha == binding.get("manifest_sha256") and
            Path(metadata.get("checkpoint_path", "")).resolve() == checkpoint,
            "front-preservation video checkpoint bytes/path differ")
    counters = {key: getattr(args, "expected_" + key) for key in CORE.BASE.COUNTERS}
    require(all(metadata.get(key) == value for key, value in counters.items()) and
            proof.get("saved_global_policy_decisions") == counters["global_policy_decisions"],
            "front-preservation explicit counters differ from checkpoint/video")
    expected_head = CORE.BASE.checked_head(args.expected_head)
    runtime = metadata.get("runtime_contract") or {}
    receipt = metadata.get(IDENTITY) or {}
    require(receipt.get("schema") == SCHEMA and
            runtime.get("source_git_commit") == expected_head and
            manifest.get("runtime_contract") == runtime and
            args.checkpoint_runtime_head is None and
            receipt.get("target_git_commit") == expected_head and
            receipt.get("target_runtime_content_sha256") == runtime.get("runtime_content_sha256") and
            receipt.get("target_contract_sha256") == CORE.BASE.json_digest(runtime),
            "front-preservation receipt does not bind the exact evaluation runtime")
    require(receipt.get("control_N_mapper_HISTORY_FLassist_unchanged") is True and
            receipt.get("rear_task_assists_enabled") is False and
            receipt.get("return_estimator_changed") is False and
            receipt.get("old_rollout_inherited") is False and
            receipt.get("fresh_rollout_required") is True and
            all(receipt.get("added_" + name) == 0 for name in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates")),
            "front-preservation boundary changes control/state or invents credit")
    require(all(getattr(args, name, None) is None for name in (
                "rr_retention_publication", "rr_retention_publication_sha256",
                "rr_retention_plan_sha256", "front_retention439_publication",
                "front_retention439_publication_sha256",
                "front_retention439_plan_sha256", "collection512_publication",
                "collection512_publication_sha256")),
            "front-preservation sibling must not borrow later72e/65a/collection arguments")
    old_runtime = _runtime_source(runtime, receipt)
    require(receipt.get("source_contract_sha256") == CORE.BASE.json_digest(old_runtime),
            "front-preservation source contract digest differs")
    source_identity = _source_identity(receipt, args)
    require(not any(key in metadata for key in FORBIDDEN),
            "front-preservation branch borrowed later72e/65a/AUX32/collection lineage")
    source = receipt["source_manifest"]
    for key, value in source.items():
        # Historical branch definitions/receipts are immutable.  Their generic
        # cumulative *_branch_counts legitimately advance on every ordinary
        # PPO save; the new branch and replay counters below are the authority
        # for attributing this sibling's +512/+1/+20 work.
        if key != "resume_migration" and (key.endswith("_migration") or
                                           key.endswith("_branch")):
            require(metadata.get(key) == value,
                    "front-preservation changed legacy source lineage: " + key)
    require(metadata.get("seed") == source.get("seed") and
            metadata.get("normalization") == source.get("normalization") and
            metadata.get("normalizer_state_sha256") == source.get("normalizer_state_sha256") and
            metadata.get("policy_contract") == source.get("policy_contract") ==
                receipt.get("target_policy_contract") and
            metadata.get("runner_config") == receipt.get("target_runner_config") and
            metadata["runner_config"].get("num_steps_per_env") == 512 and
            metadata["runner_config"].get("semantic_collection_profile") == COLLECTION_PROFILE,
            "front-preservation source state/policy or target512 runner differs")
    replay_spec = receipt.get("front_replay_spec") or {}
    dataset_binding = _validate_dataset(replay_spec, args)
    delta = _counter_delta(counters)
    replay_counts = _expected_replay_counts(delta["ppo_updates"])
    require(metadata.get(COUNTS_KEY) == delta and
            metadata.get(REPLAY_COUNTS_KEY) == replay_counts,
            "front-preservation branch/replay counts differ from actual lifetime counters")
    _validate_last_replay(metadata, replay_spec, delta["ppo_updates"])
    route = metadata.get("checkpoint_output_routing") or {}
    branch_root = (ROOT / "outputs" / "ppo_rr_rl_timing_policy_learning_v1" /
                   "branches" / BRANCH).resolve()
    require(route == receipt.get("checkpoint_output_routing") and
            route.get("schema") == "wlr50_clean.checkpoint_output_routing.v1" and
            route.get("branch") == BRANCH and
            Path(route.get("output_root", "")).resolve() == branch_root and
            route.get("main_latest_pointer_promotion") is False and
            checkpoint.parent == branch_root / "checkpoints" / "history",
            "front-preservation checkpoint escaped its explicit sibling route")
    publication = _validate_publication(receipt, runtime, route, replay_spec, args)
    actor_hash = (proof.get("parameter_hashes") or {}).get("actor_parameter_sha256")
    require(actor_hash == metadata.get("actor_parameter_sha256"),
            "loaded actor differs from front-preservation checkpoint sidecar")
    result = copy.deepcopy(source_identity)
    result.update({"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "manifest": str(sidecar), "manifest_sha256": sidecar_sha,
        "actor_parameter_sha256": actor_hash, "policy_version": POLICY,
        "observation_layout": LAYOUT, "observation_dimension": 439,
        "lifetime_counters": counters, "runtime_head": expected_head,
        "checkpoint_runtime_head": expected_head,
        "evaluation_runtime_head": expected_head,
        "reviewed_media_only_runtime_compatibility": None,
        "front_preservation439_branch_identity": {
            "schema": SCHEMA, "branch": BRANCH, "source_runtime_head": SOURCE_HEAD,
            "source_checkpoint_sha256": SOURCE_SHA,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA,
            "revision_counter_origin": copy.deepcopy(ORIGIN),
            "revision_branch_counts": delta, "publication": publication,
            "dataset": dataset_binding, "replay_spec": copy.deepcopy(replay_spec),
            "boundary_added_learning": {"policy_decisions": 0, "ppo_updates": 0,
                                         "optimizer_steps": 0, "auxiliary_updates": 0},
            "physical_retention_proven_by_replay": False},
        "front_replay_counts": replay_counts,
        "legacy_inherited_source_auxiliary":
            copy.deepcopy(source_identity.get("inherited_auxiliary")),
        "migration_added_learning": {"policy_decisions": 0, "ppo_updates": 0,
                                     "optimizer_steps": 0, "auxiliary_updates": 0},
        "this_run_auxiliary_updates": 0,
        "FL_capture_assist_enabled": True, "rear_task_assists_enabled": False,
        "front_preservation_export_adapter": {
            "path": str(Path(__file__).resolve()),
            "sha256": CORE.BASE.sha256(Path(__file__))}})
    return result


def export(args: argparse.Namespace) -> dict[str, Any]:
    previous = CORE.checkpoint_identity
    CORE.checkpoint_identity = checkpoint_identity
    try:
        return CORE.export(args)
    finally:
        CORE.checkpoint_identity = previous


def parser() -> argparse.ArgumentParser:
    result = CORE.parser()
    result.description = __doc__
    result.add_argument("--front-preservation-publication", type=Path, default=None)
    result.add_argument("--front-preservation-publication-sha256", default=None)
    result.add_argument("--front-replay-dataset", type=Path, default=None)
    result.add_argument("--front-replay-dataset-sha256", default=None)
    return result


if __name__ == "__main__":
    export(parser().parse_args())

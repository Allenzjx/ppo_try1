"""Export one sealed owner439 RL-recovery evaluation without starting Isaac.

This is a deliberately narrow adapter over the already validated rear-policy
media pipeline. It accepts only the reviewed learned422 -> owner439 boundary,
the exact reward-only72e boundary, the three-path 65a accounting identity and
its append-only front-retention ledger, the explicit collection512 outer
boundary, the isolated CP225280 front-preservation sibling identity, their
ordinary same-branch descendants, and the fixed historical N reference. The
exporter never loads Torch or a simulator.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent
BASE_EXPORTER = (ROOT / "outputs" / "ppo_rr_rl_timing_policy_learning_v1" /
                 "export_policy_rear_no_assist_video.py")

OWNER_SCHEMA = "wlr50_clean.rear_owner_recovery_append439.v1"
OWNER_FACTOR_KEY = "rear_owner_append439_factor"
OWNER_MIGRATION = "rear_owner_recovery_migration"
OWNER_MODE = "issued_rear_transfer_owner_suspension_v1"
OWNER_POLICY = "rear_owner_recovery_history_v1"
OWNER_ACTOR = (
    "wlr50_clean.ppo.semantic_rear_owner_actor:"
    "SemanticRearOwnerRecoveryHistoryMLPModel")
OWNER_LAYOUT = "role422_rear_owner_recovery_v1"
OWNER_GROUP = "rear_owner_recovery_full17"
OWNER_FIELDS = tuple(
    f"{kind}_{index}"
    for kind in ("anchor_final_deg", "anchor_request_deg", "active",
                 "winning_late_owner")
    for index in (0, 1, 4, 5)
) + ("rl_edge_recovery_permitted",)
OWNER_SCALES = (180.0,) * 8 + (1.0,) * 9
OWNER_SOURCE_HEAD = "49eb23163a6e20bc56301dbafb59b137ecebce66"
OWNER_TARGET_HEAD = "f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b"
OWNER_SOURCE_SHA = "21897a4b2d2a85f01092c187c1b1ac824d8e61b01edaf732a6385ca951ab1dcf"
OWNER_SOURCE_MANIFEST_SHA = "6e05b4e279c1d8ac99447f91d69b7db3f6dd6afaab6a326e610283f783b1f185"
OWNER_ORIGIN = {"global_policy_decisions": 225280,
                "ppo_updates": 1725, "optimizer_steps": 34500}
OWNER_PLAN_SHA = "f7e4238e2f98237172e5266875e9bbc9d692016265616aec4c1e2e17561fd5af"
OWNER_PUBLICATION_SHA = "161fd1c34400d5880f1ceea0d44bef0bfe8dac257dfbf40db172d8a7c5661267"
OWNER_PUBLISHED_CHECKPOINT_SHA = "fbe28e3718a5796afc2271e7b10aa04017369593b5c62e0bde1973a20f68032f"
OWNER_PUBLISHED_MANIFEST_SHA = "f501b7a735c0aaa42be00114e09f80d46f7f009d7717aa416fc30ad3cc837d6d"
OWNER_TARGET_RUNTIME_SHA = "d1561db383c0b359526889f3a2844dbcf97b2ef60a9f30741ecd9c8767a43dba"
OWNER_TARGET_CONTRACT_SHA = "aebdbcbefdad3bc2b525a52d3a6cb478907d5976615e2d598a716160aef8e0dd"
OWNER_PROXY_REVISION = "rr_capture_cooperative_preparation_v5"
OWNER_TIMING_REVISION = "rr_rl_edge_recovery_v4"

RETENTION_SCHEMA = "wlr50_clean.rr_retention_reward_same439.v1"
RETENTION_FACTOR_KEY = "rr_retention_same439_factor"
RETENTION_MIGRATION = "rr_retention_reward_migration"
RETENTION_REWARD_REVISION = "rr_recapture_retention_reward_only_v1"
RETENTION_TARGET_HEAD = "72e63592bdf412d375abfda29b03cf456fbf4f7e"
RETENTION_SOURCE_SHA = "fbd27dea193153c7150881b1258dbf5ce249cfd5d1529e22b066be0abf0da973"
RETENTION_SOURCE_MANIFEST_SHA = "93f41386de6d46fc204f2a22cb2c712ea83eb9b1d6b29c32a7276e2d9da6df53"
RETENTION_ORIGIN = {"global_policy_decisions": 226048,
                    "ppo_updates": 1731, "optimizer_steps": 34620}
RETENTION_PLAN_SHA = "e1050240996e1669731a041f339924ef5ff7eba68983e628b806368e0b5dff9d"
RETENTION_PUBLICATION_SHA = "8c2fd9193ab6d3088cffcd620f9577c660fc2904d11ca2ffe23d5dc76b3a4402"
RETENTION_PUBLISHED_CHECKPOINT_SHA = (
    "15be034b14cfc8741a9dc83e12b6da2220af6d97ccee06b8da109c0e9b706113")
RETENTION_PUBLISHED_MANIFEST_SHA = (
    "80f350f6bb6f5b188f8eec4ea3fa5053de62cf81944014ff5b344a8c44e1f3ed")
RETENTION_TARGET_RUNTIME_SHA = (
    "452c7d15c30486224ee32e0051baee7a714192c9cb66c805d7e872d31e151896")
RETENTION_TARGET_CONTRACT_SHA = (
    "575e7cabc95476b49a9a5c2240aea595ba2e5a496da594aaddfe07908d98ac7b")
RETENTION_CHANGED_PATHS = frozenset({
    "src/wlr50_clean/ppo/semantic_cli.py",
    "src/wlr50_clean/ppo/semantic_migration.py",
    "src/wlr50_clean/ppo/semantic_rear_policy_timing_migration.py",
    "src/wlr50_clean/ppo/semantic_reward.py",
    "src/wlr50_clean/ppo/semantic_rr_retention_migration.py",
    "src/wlr50_clean/ppo/semantic_training.py",
})

FRONT_RETENTION_SCHEMA = "wlr50_clean.front_retention439_runtime_identity.v1"
FRONT_RETENTION_FACTOR_KEY = "front_retention439_identity_factor"
FRONT_RETENTION_IDENTITY = "front_retention439_runtime_identity"
FRONT_RETENTION_LEDGER = "front_retention439_auxiliary"
FRONT_RETENTION_LEDGER_SCHEMA = (
    "wlr50_clean.front_retention439_auxiliary_ledger.v1")
FRONT_RETENTION_FIT_SCHEMA = (
    "wlr50_clean.finite_front_retention439_phase_columns_aux.v1.fit")
FRONT_RETENTION_DATA_SCHEMA = (
    "wlr50_clean.student_probe_v2_front_retention439_rows.v1")
FRONT_RETENTION_TARGET_HEAD = "65a9255be6d9fd4e19590a48a3a650ae606b04f7"
FRONT_RETENTION_SOURCE_SHA = (
    "a02bfd50e17cad3bebe083493b0da9611cfe607fa18dad6a4ed83abdecb91f1c")
FRONT_RETENTION_SOURCE_MANIFEST_SHA = (
    "7c2b8958f695387f9c186978ec6a4b80df496220cfbd4d0eabfe62dc6e683036")
FRONT_RETENTION_ORIGIN = {"global_policy_decisions": 229120,
                          "ppo_updates": 1755, "optimizer_steps": 35100}
FRONT_RETENTION_PLAN_SHA = (
    "faeb48d9d59c0d9ba5225f54bd124857ee348a221decc1c17e9cc3204d1a92a7")
FRONT_RETENTION_PUBLICATION_SHA = (
    "5178c92547a73fe8cdd05c0087a5883b0f05819895de2cbe06b0f658a79b0917")
FRONT_RETENTION_PUBLISHED_CHECKPOINT_SHA = (
    "3e6e646dedcf3e014de91f3741b51c273be5f7ec681778945133e80638a3ffa5")
FRONT_RETENTION_PUBLISHED_MANIFEST_SHA = (
    "6d3cfc15d33add098c01855676297e53895ff9a17ead833cb067f792c803c588")
FRONT_RETENTION_TARGET_RUNTIME_SHA = (
    "863bba75d6623edad17187187e58e9c626f82b953a51bdebd60966ccacb94fc5")
FRONT_RETENTION_TARGET_CONTRACT_SHA = (
    "3496a9bb0981fed786038ba9017756a2823592111ee12f58281c07a2fa2242c6")
FRONT_RETENTION_CHANGED_FILES = {
    "src/wlr50_clean/ppo/semantic_front_retention439.py": {
        "before": None,
        "after": "8663647a721ffa9cdb6a2155eece5ce8261fd3af9d5c1061f3e120a27345554b"},
    "src/wlr50_clean/ppo/semantic_rear_policy_timing_migration.py": {
        "before": "277b618a3bfff06a72975e0849c1f8518c68635dbd9d9188cc2c10bef18d6c9c",
        "after": "524dd2c1b76993e4d932e9b8fd14e7860af56edbe74cd4927e662dc0a8b46dd0"},
    "src/wlr50_clean/ppo/semantic_training.py": {
        "before": "64bee70e1e0d5d90c2484f7de47e97c3c85bfdf2d574ddcd06eab5e45d0fbc18",
        "after": "1eacfe00d67dfa359ca3da6a351e81a87bf9e2b9a81bb637622abf12821b657d"},
}
FRONT_RETENTION_EVENT_KIND = (
    "executed_P02_P05_P06_P09_retention_sparse_phase_columns_not_PPO")
FRONT_RETENTION_PARAMETERS = ["actor.mlp.0.weight[:,[1,4,5,8]]"]

COLLECTION_KEY = "collection_horizon439"
COLLECTION_SCHEMA = "wlr50_clean.collection_horizon439.v1"
COLLECTION_PROFILE_KEY = "semantic_collection_profile"
COLLECTION_PROFILE = "n1_rear_owner439_collection512_v1"
COLLECTION_TARGET_HEAD = "59e868f3e223c589e7645a0f5d63f91fa6119fb6"
COLLECTION_CHANGED_PATHS = frozenset({
    "src/wlr50_clean/ppo/semantic_return_profile.py",
    "src/wlr50_clean/ppo/semantic_training.py",
    "src/wlr50_clean/ppo/semantic_front_retention439.py",
    "src/wlr50_clean/ppo/semantic_cli.py",
    "src/wlr50_clean/ppo/semantic_video_cli.py",
    "src/wlr50_clean/ppo/semantic_policy_distribution.py",
})

HISTORICAL_N_SOURCE_MANIFEST_SHA = "03549f2590759fe22c09ec83ec12282f17f06de722a15ece547873140a083a20"
HISTORICAL_N_RUN_MANIFEST_SHA = "b07d16233a4c56cd9b287434b4b434da90451147b31cd576e6c3a0af218837ee"
HISTORICAL_N_VIDEO_SHA = "80ba92cafd39763fd48956bd3f5901569b31f19f401bcecefd44d3e84d02849d"
HISTORICAL_N_LEDGER_SHA = "2d3e0c12405e2a4ded43121e2ee90124af016ecb7acfe82e0e73a7476b53b8f4"
HISTORICAL_N_HEAD = "ee5a9651591d20bea48be8ceba075fa66594cb36"
HISTORICAL_N_TICKS = 8857


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("_rl_recovery_media_base", BASE_EXPORTER)
    if spec is None or spec.loader is None:
        raise RuntimeError("validated rear-policy media exporter cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
require = BASE.require
BASE_PANEL_LINES = BASE.panel_lines


def reconstruct_owner_source_runtime(runtime: dict[str, Any],
                                     receipt: dict[str, Any]) -> dict[str, Any]:
    """Reverse exactly the published learned422 -> public-owner439 boundary."""
    require(isinstance(receipt, dict) and receipt.get("schema") == OWNER_SCHEMA and
            receipt.get("target_git_commit") == runtime.get("source_git_commit") and
            receipt.get("target_runtime_content_sha256") ==
                runtime.get("runtime_content_sha256") and
            receipt.get("target_contract_sha256") == BASE.json_digest(runtime) and
            receipt.get("discard_old_rollout_storage") is True,
            "owner439 receipt does not bind the checkpoint runtime")
    previous = copy.deepcopy(runtime)
    source_budgets = receipt.get("source_training_budgets")
    require(isinstance(source_budgets, dict),
            "owner439 receipt lacks source training budgets")
    previous["training_budgets"] = copy.deepcopy(source_budgets)
    previous["source_git_commit"] = receipt.get("source_git_commit")
    previous["runtime_content_sha256"] = receipt.get("source_runtime_content_sha256")
    changed = receipt.get("changed_file_hashes")
    files = previous.get("files")
    require(isinstance(changed, dict) and len(changed) == 24 and
            isinstance(files, dict),
            "owner439 receipt lacks its exact finite 24-file delta")
    for path, binding in changed.items():
        require(isinstance(path, str) and isinstance(binding, dict) and
                set(binding) == {"before", "after"} and
                files.get(path) == binding.get("after"),
                "owner439 changed-file binding differs from the current runtime")
        before = binding.get("before")
        if before is None:
            files.pop(path, None)
        else:
            BASE.checked_sha(before, "owner439 source file hash")
            files[path] = before
    source_configuration = receipt.get("source_changed_configuration")
    selected = previous.get("selected_configuration")
    require(isinstance(source_configuration, dict) and
            len(source_configuration) == 5 and isinstance(selected, dict) and
            set(source_configuration) <= set(selected),
            "owner439 source configuration delta differs from the reviewed five files")
    selected.update(copy.deepcopy(source_configuration))
    require(previous.get("source_git_commit") == OWNER_SOURCE_HEAD and
            BASE.json_digest(previous) == receipt.get("source_contract_sha256"),
            "owner439 historical learned422 runtime reconstruction is not exact")
    return previous


def reconstruct_retention_source_runtime(runtime: dict[str, Any],
                                         receipt: dict[str, Any]) -> dict[str, Any]:
    """Reverse only the reviewed six-file reward boundary back to f6d owner439."""
    require(isinstance(receipt, dict) and receipt.get("schema") == RETENTION_SCHEMA and
            receipt.get("target_git_commit") == runtime.get("source_git_commit") and
            receipt.get("target_runtime_content_sha256") ==
                runtime.get("runtime_content_sha256") and
            receipt.get("target_contract_sha256") == BASE.json_digest(runtime) and
            receipt.get("discard_old_rollout_storage") is True,
            "reward-only72e receipt does not bind the checkpoint runtime")
    changed = receipt.get("changed_file_hashes")
    files = runtime.get("files")
    require(isinstance(changed, dict) and set(changed) == RETENTION_CHANGED_PATHS and
            isinstance(files, dict) and BASE.json_digest(files) ==
                runtime.get("runtime_content_sha256"),
            "reward-only72e receipt lacks its exact six-file delta")
    previous = copy.deepcopy(runtime)
    previous["source_git_commit"] = receipt.get("source_git_commit")
    previous["runtime_content_sha256"] = receipt.get("source_runtime_content_sha256")
    previous_files = previous["files"]
    for path, binding in changed.items():
        require(isinstance(binding, dict) and set(binding) == {"before", "after"} and
                previous_files.get(path) == binding.get("after"),
                "reward-only72e target file differs from its receipt: " + str(path))
        before = binding.get("before")
        if before is None:
            previous_files.pop(path, None)
        else:
            BASE.checked_sha(before, "reward-only72e source file hash")
            previous_files[path] = before
    require(previous.get("source_git_commit") == OWNER_TARGET_HEAD and
            previous.get("runtime_content_sha256") == OWNER_TARGET_RUNTIME_SHA and
            receipt.get("source_contract_sha256") == OWNER_TARGET_CONTRACT_SHA and
            BASE.json_digest(previous_files) == OWNER_TARGET_RUNTIME_SHA and
            BASE.json_digest(previous) == OWNER_TARGET_CONTRACT_SHA,
            "reward-only72e historical owner439 runtime reconstruction is not exact")
    return previous


def reconstruct_front_retention439_source_runtime(
        runtime: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Reverse exactly the three-path accounting-only identity to 72e."""
    require(isinstance(receipt, dict) and
            receipt.get("schema") == FRONT_RETENTION_SCHEMA and
            receipt.get("target_git_commit") == runtime.get("source_git_commit") ==
                FRONT_RETENTION_TARGET_HEAD and
            receipt.get("target_runtime_content_sha256") ==
                runtime.get("runtime_content_sha256") ==
                FRONT_RETENTION_TARGET_RUNTIME_SHA and
            receipt.get("target_contract_sha256") ==
                BASE.json_digest(runtime) == FRONT_RETENTION_TARGET_CONTRACT_SHA and
            receipt.get("changed_file_hashes") == FRONT_RETENTION_CHANGED_FILES,
            "front-retention439 identity target or reviewed three-path delta differs")
    files = runtime.get("files")
    require(isinstance(files, dict) and
            BASE.json_digest(files) == FRONT_RETENTION_TARGET_RUNTIME_SHA,
            "front-retention439 target runtime inventory digest differs")
    previous = copy.deepcopy(runtime)
    for path, binding in FRONT_RETENTION_CHANGED_FILES.items():
        require(previous["files"].get(path) == binding["after"],
                "front-retention439 target file differs: " + path)
        if binding["before"] is None:
            previous["files"].pop(path)
        else:
            previous["files"][path] = binding["before"]
    previous["source_git_commit"] = receipt.get("source_git_commit")
    previous["runtime_content_sha256"] = receipt.get(
        "source_runtime_content_sha256")
    require(previous.get("source_git_commit") == RETENTION_TARGET_HEAD and
            previous.get("runtime_content_sha256") == RETENTION_TARGET_RUNTIME_SHA and
            receipt.get("source_contract_sha256") == RETENTION_TARGET_CONTRACT_SHA and
            BASE.json_digest(previous["files"]) == RETENTION_TARGET_RUNTIME_SHA and
            BASE.json_digest(previous) == RETENTION_TARGET_CONTRACT_SHA,
            "front-retention439 historical72e reconstruction is not exact")
    return previous


def reconstruct_collection439_source_runtime(
        runtime: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    """Reverse only the reviewed six-path collection boundary to exact 65a."""
    require(isinstance(receipt, dict) and
            receipt.get("schema") == COLLECTION_SCHEMA and
            runtime.get("source_git_commit") == COLLECTION_TARGET_HEAD and
            receipt.get("target_contract_sha256") == BASE.json_digest(runtime),
            "collection512 receipt does not bind the checkpoint runtime")
    files = runtime.get("files")
    changed = receipt.get("changed_file_hashes")
    require(isinstance(files, dict) and
            BASE.json_digest(files) == runtime.get("runtime_content_sha256") and
            isinstance(changed, dict) and
            set(changed) == COLLECTION_CHANGED_PATHS,
            "collection512 receipt lacks its exact six-path delta")
    previous = copy.deepcopy(runtime)
    for path, binding in changed.items():
        require(isinstance(binding, dict) and
                set(binding) == {"before", "after"} and
                binding.get("before") is not None and
                previous["files"].get(path) == binding.get("after") and
                BASE.checked_sha(binding["before"], "collection512 source file hash") and
                BASE.checked_sha(binding["after"], "collection512 target file hash"),
                "collection512 target file differs from its receipt: " + str(path))
        previous["files"][path] = binding["before"]
    previous["source_git_commit"] = FRONT_RETENTION_TARGET_HEAD
    previous["runtime_content_sha256"] = BASE.json_digest(previous["files"])
    require(previous["runtime_content_sha256"] ==
                FRONT_RETENTION_TARGET_RUNTIME_SHA and
            BASE.json_digest(previous) == receipt.get("source_contract_sha256") ==
                FRONT_RETENTION_TARGET_CONTRACT_SHA,
            "collection512 historical65a reconstruction is not exact")
    return previous


def _empty_front_retention_ledger() -> dict[str, Any]:
    return {"schema": FRONT_RETENTION_LEDGER_SCHEMA, "events": [],
            "accepted_auxiliary_updates_total": 0,
            "attempted_auxiliary_optimizer_steps_total": 0}


def _bound_json(binding: Any, label: str) -> tuple[Path, dict[str, Any]]:
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"},
            label + " requires an exact path/SHA binding")
    path = Path(binding["path"]).resolve(strict=True)
    require(str(path) == binding["path"] and
            BASE.sha256(path) == BASE.checked_sha(binding["sha256"], label),
            label + " bytes differ from the embedded binding")
    return path, BASE.read_json(path)


def _bound_checkpoint(binding: Any, label: str) -> tuple[Path, dict[str, Any]]:
    require(isinstance(binding, dict) and set(binding) == {
                "checkpoint", "checkpoint_sha256", "manifest", "manifest_sha256"},
            label + " requires exact checkpoint/manifest bindings")
    checkpoint = Path(binding["checkpoint"]).resolve(strict=True)
    manifest = checkpoint.with_name(checkpoint.stem + "_manifest.json").resolve(strict=True)
    require(str(checkpoint) == binding["checkpoint"] and
            str(manifest) == binding["manifest"] and
            BASE.sha256(checkpoint) ==
                BASE.checked_sha(binding["checkpoint_sha256"], label) and
            BASE.sha256(manifest) ==
                BASE.checked_sha(binding["manifest_sha256"], label),
            label + " immutable checkpoint bytes changed")
    return checkpoint, BASE.read_json(manifest)


def _validate_front_retention_ledger(metadata: dict[str, Any],
                                     identity_receipt: dict[str, Any],
                                     runtime: dict[str, Any],
                                     counters: dict[str, int]) -> dict[str, Any]:
    """Validate the append-only AUX ledger and return display-only totals."""
    ledger = metadata.get(FRONT_RETENTION_LEDGER,
                          _empty_front_retention_ledger())
    require(isinstance(ledger, dict) and
            ledger.get("schema") == FRONT_RETENTION_LEDGER_SCHEMA and
            isinstance(ledger.get("events"), list),
            "front-retention439 checkpoint lacks its explicit AUX ledger")
    prefix = _empty_front_retention_ledger()
    route = metadata.get("checkpoint_output_routing")
    for index, event in enumerate(ledger["events"], 1):
        require(isinstance(event, dict) and event.get("event_index") == index and
                event.get("kind") == FRONT_RETENTION_EVENT_KIND and
                event.get("optimized_parameters") == FRONT_RETENTION_PARAMETERS and
                event.get("optimized_scalar_count") == 1024 and
                event.get("RR_capture_success_claimed") is False and
                event.get("teacher_deployed") is False and
                event.get("same_future_trajectory_claimed") is False and
                all(type(event.get(key)) is int and event[key] == 0 for key in (
                    "PPO_decisions_added", "PPO_updates_added",
                    "PPO_optimizer_steps_added")),
                "front-retention439 AUX event scope or credit differs")
        _, parent = _bound_checkpoint(
            event.get("source_checkpoint"), "front-retention439 AUX source")
        parent_counters = {key: parent.get(key) for key in BASE.COUNTERS}
        require(all(type(parent_counters[key]) is int for key in BASE.COUNTERS),
                "front-retention439 AUX parent counters are not integers")
        parent_updates = (parent_counters["ppo_updates"] -
                          FRONT_RETENTION_ORIGIN["ppo_updates"])
        require(parent.get(FRONT_RETENTION_IDENTITY) == identity_receipt and
                parent.get("runtime_contract") == runtime and
                parent.get("checkpoint_output_routing") == route and
                parent.get(FRONT_RETENTION_LEDGER,
                           _empty_front_retention_ledger()) == prefix and
                event.get("PPO_counters_unchanged") == parent_counters and
                event.get("source_ledger_sha256") == BASE.json_digest(prefix) and
                event.get("actor_parameter_sha256_before") ==
                    parent.get("actor_parameter_sha256") and
                all(FRONT_RETENTION_ORIGIN[key] <=
                    parent_counters[key] <= counters[key]
                    for key in BASE.COUNTERS) and
                parent_counters["global_policy_decisions"] -
                    FRONT_RETENTION_ORIGIN["global_policy_decisions"] ==
                    128 * parent_updates and
                parent_counters["optimizer_steps"] -
                    FRONT_RETENTION_ORIGIN["optimizer_steps"] ==
                    20 * parent_updates,
                "front-retention439 AUX append source/ledger/counters differ")
        data_binding = event.get("data_receipt")
        _, data = _bound_json(data_binding, "front-retention439 data receipt")
        require(data.get("schema") == FRONT_RETENTION_DATA_SCHEMA and
                data.get("receipt_content_sha256") == BASE.json_digest({
                    key: value for key, value in data.items()
                    if key != "receipt_content_sha256"}) and
                data.get("observation_dimension") == 439 and
                data.get("action_dimension") == 12 and
                data.get("phase_columns") == [1, 4, 5, 8] and
                data.get("decision_index_cutoff_inclusive") == 997 and
                data.get("target_semantics") ==
                    "executed_student_conditional_raw_request" and
                data.get("synthetic_or_intervention_rows_used") is False and
                data.get("teacher_deployed") is False and
                data.get("whole_failed_trajectory_used_as_success_label") is False and
                data.get("targets_are_physical_success_labels") is False and
                data.get("PPO_credit") == 0 and data.get("new_AUX_credit") == 0,
                "front-retention439 data receipt scope differs")
        report_binding = event.get("fit_report")
        _, report = _bound_json(report_binding, "front-retention439 fit report")
        accepted = report.get("accepted_auxiliary_updates")
        attempted = report.get("attempted_auxiliary_optimizer_steps")
        require(report.get("schema") == FRONT_RETENTION_FIT_SCHEMA and
                type(accepted) is int and type(attempted) is int and
                0 < accepted <= attempted <= 32 and
                event.get("accepted_auxiliary_updates") == accepted and
                event.get("attempted_auxiliary_optimizer_steps") == attempted and
                report.get("optimized_parameters") == FRONT_RETENTION_PARAMETERS and
                report.get("optimized_scalar_count") == 1024 and
                report.get("data_receipt") == data_binding and
                report.get("actor_parameter_sha256_before") ==
                    event.get("actor_parameter_sha256_before") and
                report.get("actor_parameter_sha256_after") ==
                    event.get("actor_parameter_sha256_after") and
                report.get("actor_parameter_sha256_after") !=
                    report.get("actor_parameter_sha256_before") and
                report.get("protected_state_before") ==
                    report.get("protected_state_after") and
                all(report.get(key) is True for key in (
                    "real_P10_P12_full_Gaussian_bitwise_unchanged",
                    "synthetic_P13_full_Gaussian_bitwise_unchanged",
                    "zero_selected_phase_columns_P10_P13_algebra_verified",
                    "nonselected_actor_state_unchanged", "fresh_PPO_rollout_required",
                    "targets_were_executed_student_raw_requests")) and
                all(report.get(key) is False for key in (
                    "targets_were_success_or_teacher_labels", "teacher_deployed",
                    "physical_success_claimed", "same_future_trajectory_claimed")) and
                all(type(report.get(key)) is int and report[key] == 0 for key in (
                    "PPO_decisions_added", "PPO_updates_added",
                    "PPO_optimizer_steps_added")),
                "front-retention439 fit report changes protected state or credit")
        prefix["events"].append(copy.deepcopy(event))
        prefix["accepted_auxiliary_updates_total"] += accepted
        prefix["attempted_auxiliary_optimizer_steps_total"] += attempted
    require(ledger == prefix,
            "front-retention439 AUX ledger totals or append-only chain differ")
    if ledger["events"]:
        last = ledger["events"][-1]
        if {key: metadata.get(key) for key in BASE.COUNTERS} == \
                last.get("PPO_counters_unchanged"):
            _, report = _bound_json(last["fit_report"],
                                    "front-retention439 terminal fit report")
            require(metadata.get("actor_parameter_sha256") ==
                        last.get("actor_parameter_sha256_after") and
                    report.get("protected_state_after") == {
                        "critic_parameter_sha256":
                            metadata.get("critic_parameter_sha256"),
                        "optimizer_state_sha256":
                            metadata.get("optimizer_state_sha256"),
                        "normalizer_state_sha256":
                            metadata.get("normalizer_state_sha256"),
                        "optimizer_learning_rate":
                            metadata.get("optimizer_learning_rate"),
                        "training_rng_state_sha256":
                            BASE.json_digest(metadata.get("training_rng_state"))},
                    "same-counter front-retention439 AUX learned state differs")
    return {"schema": FRONT_RETENTION_LEDGER_SCHEMA,
            "event_count": len(ledger["events"]),
            "accepted_auxiliary_updates":
                ledger["accepted_auxiliary_updates_total"],
            "attempted_auxiliary_optimizer_steps":
                ledger["attempted_auxiliary_optimizer_steps_total"],
            "PPO_credit": 0, "physical_success_claimed": False,
            "teacher_deployed": False,
            "optimized_parameters": FRONT_RETENTION_PARAMETERS}


def _validate_publication(args: argparse.Namespace) -> dict[str, Any]:
    publication_path = Path(args.rear_owner_publication).resolve(strict=True)
    require(publication_path.parent == OUTPUT_ROOT.resolve() and
            BASE.sha256(publication_path) ==
                BASE.checked_sha(args.rear_owner_publication_sha256,
                                 "--rear-owner-publication-sha256") ==
                OWNER_PUBLICATION_SHA,
            "owner439 publication receipt differs from the frozen reviewed receipt")
    publication = BASE.read_json(publication_path)
    published_checkpoint = Path(publication.get("checkpoint", "")).resolve(strict=True)
    published_manifest = Path(publication.get("manifest", "")).resolve(strict=True)
    require(published_manifest == published_checkpoint.with_name(
                published_checkpoint.stem + "_manifest.json") and
            BASE.sha256(published_checkpoint) == publication.get("checkpoint_sha256") ==
                OWNER_PUBLISHED_CHECKPOINT_SHA and
            BASE.sha256(published_manifest) == publication.get("manifest_sha256") ==
                OWNER_PUBLISHED_MANIFEST_SHA and
            publication.get("save_load_round_trip") is True and
            all(publication.get(key) == OWNER_ORIGIN[key] for key in BASE.COUNTERS) and
            publication.get("rear_policy_timing_branch_counts") == {
                key: OWNER_ORIGIN[key] - BASE.RECAPTURE_SOURCE_SELECTION["counters"][key]
                for key in BASE.COUNTERS} and
            all(publication.get("added_" + key) == 0 for key in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates")),
            "owner439 publication did not preserve the exact zero-credit full state")
    return {"path": str(publication_path), "sha256": OWNER_PUBLICATION_SHA,
            "receipt": publication, "checkpoint": published_checkpoint,
            "manifest": published_manifest}


def _validate_retention_publication(args: argparse.Namespace) -> dict[str, Any]:
    path_value = getattr(args, "rr_retention_publication", None)
    sha_value = getattr(args, "rr_retention_publication_sha256", None)
    plan_value = getattr(args, "rr_retention_plan_sha256", None)
    require(path_value is not None and sha_value is not None and plan_value is not None,
            "reward-only72e export requires its publication and migration hashes")
    publication_path = Path(path_value).resolve(strict=True)
    require(publication_path.parent == OUTPUT_ROOT.resolve() and
            BASE.sha256(publication_path) ==
                BASE.checked_sha(sha_value,
                                 "--rr-retention-publication-sha256") ==
                RETENTION_PUBLICATION_SHA and
            BASE.checked_sha(plan_value, "--rr-retention-plan-sha256") ==
                RETENTION_PLAN_SHA,
            "reward-only72e publication/plan differs from the reviewed boundary")
    publication = BASE.read_json(publication_path)
    published_checkpoint = Path(publication.get("checkpoint", "")).resolve(strict=True)
    published_manifest = Path(publication.get("manifest", "")).resolve(strict=True)
    expected_branch = (BASE.OUTPUT_ROOT / "branches" /
                       BASE.RECAPTURE_BRANCH / "checkpoints" / "history").resolve()
    require(published_checkpoint.parent == expected_branch and
            published_manifest == published_checkpoint.with_name(
                published_checkpoint.stem + "_manifest.json") and
            BASE.sha256(published_checkpoint) == publication.get("checkpoint_sha256") ==
                RETENTION_PUBLISHED_CHECKPOINT_SHA and
            BASE.sha256(published_manifest) == publication.get("manifest_sha256") ==
                RETENTION_PUBLISHED_MANIFEST_SHA and
            publication.get("save_load_round_trip") is True and
            all(publication.get(key) == RETENTION_ORIGIN[key] for key in BASE.COUNTERS) and
            publication.get("rear_policy_timing_branch_counts") == {
                key: RETENTION_ORIGIN[key] -
                     BASE.RECAPTURE_SOURCE_SELECTION["counters"][key]
                for key in BASE.COUNTERS} and
            all(publication.get("added_" + key) == 0 for key in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates")),
            "reward-only72e publication did not preserve the exact zero-credit full state")
    return {"path": str(publication_path), "sha256": RETENTION_PUBLICATION_SHA,
            "receipt": publication, "checkpoint": published_checkpoint,
            "manifest": published_manifest}


def _validate_front_retention_publication(
        args: argparse.Namespace) -> dict[str, Any]:
    path_value = getattr(args, "front_retention439_publication", None)
    sha_value = getattr(args, "front_retention439_publication_sha256", None)
    plan_value = getattr(args, "front_retention439_plan_sha256", None)
    require(path_value is not None and sha_value is not None and plan_value is not None,
            "65a export requires the front-retention439 identity publication and plan")
    publication_path = Path(path_value).resolve(strict=True)
    require(publication_path.parent == OUTPUT_ROOT.resolve() and
            BASE.sha256(publication_path) ==
                BASE.checked_sha(sha_value,
                                 "--front-retention439-publication-sha256") ==
                FRONT_RETENTION_PUBLICATION_SHA and
            BASE.checked_sha(plan_value, "--front-retention439-plan-sha256") ==
                FRONT_RETENTION_PLAN_SHA,
            "front-retention439 identity publication/plan differs")
    publication = BASE.read_json(publication_path)
    checkpoint = Path(publication.get("checkpoint", "")).resolve(strict=True)
    manifest = Path(publication.get("manifest", "")).resolve(strict=True)
    branch = (BASE.OUTPUT_ROOT / "branches" / BASE.RECAPTURE_BRANCH /
              "checkpoints" / "history").resolve()
    require(checkpoint.parent == branch and
            manifest == checkpoint.with_name(checkpoint.stem + "_manifest.json") and
            BASE.sha256(checkpoint) == publication.get("checkpoint_sha256") ==
                FRONT_RETENTION_PUBLISHED_CHECKPOINT_SHA and
            BASE.sha256(manifest) == publication.get("manifest_sha256") ==
                FRONT_RETENTION_PUBLISHED_MANIFEST_SHA and
            publication.get("save_load_round_trip") is True and
            publication.get("latest_pointer_published") is False and
            all(publication.get(key) == FRONT_RETENTION_ORIGIN[key]
                for key in BASE.COUNTERS) and
            all(publication.get("added_" + key) == 0 for key in (
                "policy_decisions", "ppo_updates", "optimizer_steps",
                "auxiliary_updates")),
            "front-retention439 identity publication did not preserve zero-credit state")
    return {"path": str(publication_path),
            "sha256": FRONT_RETENTION_PUBLICATION_SHA,
            "receipt": publication, "checkpoint": checkpoint,
            "manifest": manifest}


def _validate_collection_publication(
        args: argparse.Namespace, receipt: dict[str, Any]) -> dict[str, Any]:
    path_value = getattr(args, "collection512_publication", None)
    sha_value = getattr(args, "collection512_publication_sha256", None)
    require(path_value is not None and sha_value is not None,
            "collection512 export requires its exact publication receipt hash")
    path = Path(path_value).resolve(strict=True)
    receipt_sha = BASE.checked_sha(
        sha_value, "--collection512-publication-sha256")
    require(path.parent == OUTPUT_ROOT.resolve() and
            BASE.sha256(path) == receipt_sha,
            "collection512 publication bytes differ from the explicit pin")
    publication = BASE.read_json(path)
    checkpoint = Path(publication.get("checkpoint", "")).resolve(strict=True)
    manifest = Path(publication.get("manifest", "")).resolve(strict=True)
    require(manifest == checkpoint.with_name(checkpoint.stem + "_manifest.json") and
            BASE.sha256(checkpoint) == publication.get("checkpoint_sha256") and
            BASE.sha256(manifest) == publication.get("manifest_sha256") and
            publication.get("save_load_round_trip") is True and
            publication.get("latest_pointer_published") is False and
            publication.get("source_checkpoint") == receipt.get("source_checkpoint") and
            publication.get("official_same_input_mean_sigma_value_bitwise_equal") is True and
            publication.get("physical_steps_added") == 0 and
            publication.get("new_auxiliary_updates") == 0 and
            publication.get("actual_collection_shape") == [512, 1, 12] and
            publication.get("auxiliary_ledger_preserved") is True and
            publication.get("training_rng_preserved") is True and
            publication.get("source_latest_pointer_unchanged") is True and
            all(publication.get(key) == receipt.get("counter_origin", {}).get(key)
                for key in BASE.COUNTERS) and
            all(publication.get("added_" + key) == 0 for key in (
                "policy_decisions", "ppo_updates", "optimizer_steps",
                "auxiliary_updates")),
            "collection512 publication is not the successful zero-credit reload")
    published = BASE.read_json(manifest)
    require(published.get(COLLECTION_KEY) == receipt,
            "collection512 publication manifest carries a different receipt")
    return {"path": str(path), "sha256": receipt_sha,
            "receipt": publication, "checkpoint": checkpoint,
            "manifest": manifest, "metadata": published}


def _collection_frozen_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    keys = {FRONT_RETENTION_IDENTITY, FRONT_RETENTION_LEDGER,
            "checkpoint_output_routing", "policy_contract", "seed",
            "normalization", "new_mdp_origin_global_policy_decisions",
            "source_stage_requested_decisions"}
    keys |= {key for key in metadata
             if key != "resume_migration" and
             key.endswith(("_branch", "_migration"))}
    return {key: BASE.json_digest(metadata[key])
            for key in sorted(keys) if key in metadata}


def _collection_counter_updates(counters: dict[str, int],
                                origin: dict[str, int]) -> int:
    require(all(type(counters.get(key)) is int and
                type(origin.get(key)) is int and
                counters[key] >= origin[key] for key in BASE.COUNTERS),
            "collection512 counter rollback")
    updates = counters["ppo_updates"] - origin["ppo_updates"]
    require(counters["global_policy_decisions"] -
                origin["global_policy_decisions"] == 512 * updates and
            counters["optimizer_steps"] - origin["optimizer_steps"] ==
                20 * updates,
            "collection512 descendant counters are not +512/+1/+20")
    return updates


def _collection_boundary(
        metadata: dict[str, Any], runtime: dict[str, Any],
        counters: dict[str, int], checkpoint: Path,
        args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any],
                                           dict[str, Any]]:
    """Validate collection512, returning its exact pre-boundary 65a metadata."""
    receipt = metadata.get(COLLECTION_KEY)
    require(isinstance(receipt, dict) and
            receipt.get("schema") == COLLECTION_SCHEMA,
            "checkpoint lacks its explicit collection512 receipt")
    source_checkpoint, source = _bound_checkpoint(
        receipt.get("source_checkpoint"), "collection512 source")
    source_runtime = reconstruct_collection439_source_runtime(runtime, receipt)
    source_counters = {key: source.get(key) for key in BASE.COUNTERS}
    origin = receipt.get("counter_origin")
    require(source.get(COLLECTION_KEY) is None and
            source.get("checkpoint_sha256") ==
                receipt["source_checkpoint"]["checkpoint_sha256"] and
            source.get("runtime_contract") == source_runtime and
            source.get("policy_contract") == metadata.get("policy_contract") and
            isinstance(origin, dict) and source_counters == origin and
            receipt.get("source_runner_config") == source.get("runner_config") and
            receipt.get("source_stage_requested_decisions") ==
                source.get("stage_requested_decisions"),
            "collection512 source checkpoint or frozen lineage differs")
    target_config = copy.deepcopy(source.get("runner_config") or {})
    require(target_config.get("num_steps_per_env") == 128 and
            COLLECTION_PROFILE_KEY not in target_config and
            (target_config.get("algorithm") or {}).get(
                "num_learning_epochs") == 5 and
            (target_config.get("algorithm") or {}).get(
                "num_mini_batches") == 4,
            "collection512 historical runner is not the reviewed 128/5/4 source")
    target_config["num_steps_per_env"] = 512
    target_config[COLLECTION_PROFILE_KEY] = COLLECTION_PROFILE
    frozen = _collection_frozen_metadata(source)
    require(receipt.get("target_runner_config") == target_config ==
                metadata.get("runner_config") and
            receipt.get("frozen_historical_metadata_sha256") == frozen and
            all(BASE.json_digest(metadata.get(key)) == digest
                for key, digest in frozen.items()) and
            receipt.get("collection_steps_before") == 128 and
            receipt.get("collection_steps_after") == 512 and
            receipt.get("optimizer_steps_per_update") == 20 and
            receipt.get("source_effective_learning_rate") ==
                source.get("optimizer_learning_rate") and
            receipt.get("parameter_mapping") ==
                "identity_all_actor_critic_buffers_full_Adam_options_steps_LR_Identity_rng" and
            receipt.get("same_mdp") is True and
            all(receipt.get(key) is False for key in (
                "return_estimator_changed", "reward_changed", "actor_input_changed",
                "HISTORY_changed", "raw_sample_likelihood_semantics_changed",
                "old_rollout_inherited")) and
            receipt.get("fresh_rollout_required") is True and
            all(receipt.get("added_" + key) == 0 for key in (
                "policy_decisions", "ppo_updates", "optimizer_steps",
                "auxiliary_updates")),
            "collection512 receipt changes protected semantics or historical lineage")
    require(metadata.get("runtime_contract") == runtime and
            metadata.get("checkpoint_output_routing") ==
                source.get("checkpoint_output_routing") and
            checkpoint.parent == source_checkpoint.parent,
            "collection512 checkpoint escaped its source branch")
    stages = metadata.get("stage_requested_decisions") or {}
    require(all(type(stages.get(key)) is int and stages[key] >= value
                for key, value in
                (source.get("stage_requested_decisions") or {}).items()),
            "collection512 descendant rolled back stage budgets")
    updates = _collection_counter_updates(counters, origin)
    for key, branch in metadata.items():
        if (key.endswith("_branch") and isinstance(branch, dict) and
                set(BASE.COUNTERS) <= set(branch.get("counter_origin", {})) and
                key + "_counts" in source):
            require(metadata.get(key + "_counts") == {
                name: counters[name] - branch["counter_origin"][name]
                for name in BASE.COUNTERS},
                "collection512 historical branch totals differ: " + key)
    if updates == 0:
        require(all(metadata.get(key) == source.get(key) for key in (
                    "actor_parameter_sha256", "critic_parameter_sha256",
                    "optimizer_state_sha256", "normalizer_state_sha256",
                    "optimizer_learning_rate", "training_rng_state", "last_update",
                    "stage_requested_decisions")),
                "zero-update collection512 boundary changed learned state")
    else:
        last = metadata.get("last_update") or {}
        require(last.get("global_policy_decisions") ==
                    counters["global_policy_decisions"] and
                last.get("ppo_update") == counters["ppo_updates"] and
                last.get("optimizer_steps") == 20 and
                last.get("actor_parameter_sha256_after") ==
                    metadata.get("actor_parameter_sha256") and
                last.get("optimizer_learning_rate") ==
                    metadata.get("optimizer_learning_rate"),
                "collection512 checkpoint lacks its actual complete update")
    publication = _validate_collection_publication(args, receipt)
    published = publication["metadata"]
    require(published.get("runtime_contract") == runtime and
            published.get("runner_config") == target_config and
            all(published.get(key) == source.get(key) for key in (
                "actor_parameter_sha256", "critic_parameter_sha256",
                "optimizer_state_sha256", "normalizer_state_sha256",
                "optimizer_learning_rate", "training_rng_state", "last_update",
                "stage_requested_decisions")),
            "collection512 publication failed to preserve the actual source state")
    revision = {"schema": COLLECTION_SCHEMA,
        "collection_profile": COLLECTION_PROFILE,
        "source_collection_steps": 128, "collection_steps": 512,
        "optimizer_steps_per_update": 20,
        "revision_counter_origin": copy.deepcopy(origin),
        "revision_branch_counts": {
            key: counters[key] - origin[key] for key in BASE.COUNTERS},
        "source_checkpoint": copy.deepcopy(receipt["source_checkpoint"]),
        "publication": publication["path"],
        "publication_sha256": publication["sha256"],
        "published_checkpoint": str(publication["checkpoint"]),
        "published_checkpoint_sha256":
            publication["receipt"]["checkpoint_sha256"],
        "published_manifest": str(publication["manifest"]),
        "published_manifest_sha256": publication["receipt"]["manifest_sha256"],
        "migration_added_learning": {"policy_decisions": 0, "ppo_updates": 0,
                                     "optimizer_steps": 0,
                                     "auxiliary_updates": 0},
        "historical_128_counters_reinterpreted": False,
        "reward_control_policy_distribution_changed": False}
    return source, source_runtime, revision


def _front_retention_boundary(
        metadata: dict[str, Any], runtime: dict[str, Any],
        counters: dict[str, int], checkpoint: Path,
        args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any],
                                           dict[str, Any], dict[str, Any]]:
    """Validate 65a identity/AUX, then return the exact historical 72e view."""
    publication = _validate_front_retention_publication(args)
    receipt = metadata.get(FRONT_RETENTION_IDENTITY)
    factor = (receipt or {}).get(FRONT_RETENTION_FACTOR_KEY)
    require(isinstance(receipt, dict) and isinstance(factor, dict) and
            receipt.get("schema") == FRONT_RETENTION_SCHEMA and
            factor.get("schema") == FRONT_RETENTION_SCHEMA and
            receipt.get("source_checkpoint_sha256") ==
                FRONT_RETENTION_SOURCE_SHA and
            receipt.get("source_manifest_sha256") ==
                FRONT_RETENTION_SOURCE_MANIFEST_SHA and
            receipt.get("source_git_commit") == RETENTION_TARGET_HEAD and
            receipt.get("target_git_commit") == FRONT_RETENTION_TARGET_HEAD and
            receipt.get("source_runtime_content_sha256") ==
                RETENTION_TARGET_RUNTIME_SHA and
            receipt.get("source_contract_sha256") ==
                RETENTION_TARGET_CONTRACT_SHA and
            receipt.get("target_runtime_content_sha256") ==
                FRONT_RETENTION_TARGET_RUNTIME_SHA and
            receipt.get("target_contract_sha256") ==
                FRONT_RETENTION_TARGET_CONTRACT_SHA,
            "checkpoint lacks the exact front-retention439 runtime identity")
    plan = Path(receipt.get("plan_path", "")).resolve(strict=True)
    require(plan.parent == OUTPUT_ROOT.resolve() and
            BASE.sha256(plan) == receipt.get("plan_sha256") ==
                FRONT_RETENTION_PLAN_SHA and
            BASE.read_json(plan) == {key: value for key, value in receipt.items()
                                     if key not in ("plan_path", "plan_sha256")},
            "front-retention439 identity plan differs from its embedded receipt")
    source_runtime = reconstruct_front_retention439_source_runtime(runtime, receipt)
    policy = metadata.get("policy_contract") or {}
    require(factor.get("revision_counter_origin") == FRONT_RETENTION_ORIGIN and
            factor.get("source_policy_contract") == policy and
            factor.get("target_policy_contract") == policy and
            factor.get("source_runner_config") == metadata.get("runner_config") and
            factor.get("target_runner_config") == metadata.get("runner_config") and
            factor.get("parameter_mapping") ==
                "identity_all_actor_critic_buffers_full_Adam_options_steps_LR_Identity_rng" and
            factor.get("observation_dimension") == 439 and
            factor.get("action_dimension") == 12 and factor.get("num_envs") == 1 and
            factor.get("same_mdp") is True and
            factor.get("old_rollout_inherited") is False and
            all(factor.get("added_" + key) == 0 for key in (
                "policy_decisions", "ppo_updates", "optimizer_steps",
                "auxiliary_updates")),
            "front-retention439 identity changes policy/state or invents learning")
    source_checkpoint = Path(receipt.get("source_checkpoint", "")).resolve(strict=True)
    source_manifest = source_checkpoint.with_name(
        source_checkpoint.stem + "_manifest.json").resolve(strict=True)
    require(BASE.sha256(source_checkpoint) == FRONT_RETENTION_SOURCE_SHA and
            BASE.sha256(source_manifest) == FRONT_RETENTION_SOURCE_MANIFEST_SHA,
            "front-retention439 immutable CP229120 source bytes changed")
    source_metadata = BASE.read_json(source_manifest)
    require(source_metadata.get("checkpoint_sha256") ==
                FRONT_RETENTION_SOURCE_SHA and
            source_metadata.get("runtime_contract") == source_runtime and
            source_metadata.get("policy_contract") == policy and
            source_metadata.get("runner_config") == factor.get("source_runner_config") and
            source_metadata.get("stage_requested_decisions") ==
                factor.get("source_stage_requested_decisions") and
            source_metadata.get("optimizer_learning_rate") ==
                factor.get("source_effective_learning_rate") and
            all(source_metadata.get(key) == FRONT_RETENTION_ORIGIN[key]
                for key in BASE.COUNTERS),
            "front-retention439 immutable source metadata changed")
    preserved = factor.get("preserved_metadata_sha256")
    require(isinstance(preserved, dict) and preserved and
            all(key in source_metadata and
                BASE.json_digest(source_metadata[key]) == digest
                for key, digest in preserved.items()),
            "front-retention439 source full-state preservation receipt changed")
    for key, digest in preserved.items():
        if key.endswith(("_branch", "_migration")) or key in (
                "checkpoint_output_routing", "runner_config", "policy_contract",
                "seed", "normalization"):
            require(BASE.json_digest(metadata.get(key)) == digest,
                    "front-retention439 descendant changed protected lineage: " + key)
    source_stages = source_metadata.get("stage_requested_decisions") or {}
    current_stages = metadata.get("stage_requested_decisions") or {}
    require(all(type(current_stages.get(key)) is int and
                current_stages[key] >= value for key, value in source_stages.items()),
            "front-retention439 descendant reset used training quantities")
    published = BASE.read_json(publication["manifest"])
    route = metadata.get("checkpoint_output_routing")
    branch = (BASE.OUTPUT_ROOT / "branches" / BASE.RECAPTURE_BRANCH).resolve()
    require(published.get(FRONT_RETENTION_IDENTITY) == receipt and
            published.get("runtime_contract") == runtime and
            published.get("policy_contract") == policy and
            route == source_metadata.get("checkpoint_output_routing") ==
                published.get("checkpoint_output_routing") and
            checkpoint.parent == branch / "checkpoints" / "history",
            "front-retention439 checkpoint escaped its identity/source lineage")
    branch_counts = {}
    for key in BASE.COUNTERS:
        require(type(counters[key]) is int and
                counters[key] >= FRONT_RETENTION_ORIGIN[key],
                "invalid front-retention439 " + key + " origin")
        branch_counts[key] = counters[key] - FRONT_RETENTION_ORIGIN[key]
    require(branch_counts["global_policy_decisions"] ==
                128 * branch_counts["ppo_updates"] and
            branch_counts["optimizer_steps"] ==
                20 * branch_counts["ppo_updates"],
            "front-retention439 ordinary PPO counter increments disagree")
    auxiliary = _validate_front_retention_ledger(
        metadata, receipt, runtime, counters)
    historical = {key: copy.deepcopy(value) for key, value in metadata.items()
                  if key not in (FRONT_RETENTION_IDENTITY,
                                 FRONT_RETENTION_LEDGER)}
    historical["runtime_contract"] = source_runtime
    revision = {"schema": FRONT_RETENTION_SCHEMA,
        "accounting_runtime_identity_only": True,
        "revision_counter_origin": FRONT_RETENTION_ORIGIN,
        "revision_branch_counts": branch_counts,
        "migration_plan": str(plan),
        "migration_plan_sha256": FRONT_RETENTION_PLAN_SHA,
        "publication": publication["path"],
        "publication_sha256": FRONT_RETENTION_PUBLICATION_SHA,
        "published_checkpoint": str(publication["checkpoint"]),
        "published_checkpoint_sha256":
            FRONT_RETENTION_PUBLISHED_CHECKPOINT_SHA,
        "published_manifest": str(publication["manifest"]),
        "published_manifest_sha256": FRONT_RETENTION_PUBLISHED_MANIFEST_SHA,
        "added_learning": {"policy_decisions": 0, "ppo_updates": 0,
                           "optimizer_steps": 0, "auxiliary_updates": 0}}
    return historical, source_runtime, revision, auxiliary


def _retention_boundary(metadata: dict[str, Any], runtime: dict[str, Any],
                        counters: dict[str, int], checkpoint: Path,
                        args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any],
                                                           dict[str, Any]]:
    """Validate 72e as reward-only, then return the exact historical f6d view."""
    publication = _validate_retention_publication(args)
    receipt = metadata.get(RETENTION_MIGRATION)
    factor = receipt.get(RETENTION_FACTOR_KEY) if isinstance(receipt, dict) else None
    require(isinstance(factor, dict) and receipt.get("schema") == RETENTION_SCHEMA and
            factor.get("schema") == RETENTION_SCHEMA and
            receipt.get("source_checkpoint_sha256") == RETENTION_SOURCE_SHA and
            receipt.get("source_manifest_sha256") == RETENTION_SOURCE_MANIFEST_SHA and
            receipt.get("source_git_commit") == OWNER_TARGET_HEAD and
            receipt.get("target_git_commit") == RETENTION_TARGET_HEAD and
            receipt.get("source_runtime_content_sha256") == OWNER_TARGET_RUNTIME_SHA and
            receipt.get("source_contract_sha256") == OWNER_TARGET_CONTRACT_SHA and
            receipt.get("target_runtime_content_sha256") == RETENTION_TARGET_RUNTIME_SHA and
            receipt.get("target_contract_sha256") == RETENTION_TARGET_CONTRACT_SHA,
            "checkpoint lacks the exact reward-only72e boundary")
    plan = Path(receipt.get("plan_path", "")).resolve(strict=True)
    require(plan.parent == OUTPUT_ROOT.resolve() and
            BASE.sha256(plan) == receipt.get("plan_sha256") == RETENTION_PLAN_SHA and
            BASE.read_json(plan) == {key: value for key, value in receipt.items()
                                     if key not in ("plan_path", "plan_sha256")},
            "reward-only72e migration plan differs from its embedded receipt")
    source_runtime = reconstruct_retention_source_runtime(runtime, receipt)
    policy = metadata.get("policy_contract") or {}
    require(factor.get("source_policy_contract") == policy and
            factor.get("target_policy_contract") == policy and
            factor.get("source_runner_config") == metadata.get("runner_config") and
            factor.get("target_runner_config") == metadata.get("runner_config") and
            factor.get("revision_counter_origin") == RETENTION_ORIGIN and
            factor.get("original_branch_origin") ==
                BASE.RECAPTURE_SOURCE_SELECTION["counters"] and
            factor.get("parameter_mapping") ==
                "identity_all_actor_critic_buffers_full_Adam_options_steps_LR_Identity_rng" and
            factor.get("observation_dimension") == 439 and
            factor.get("action_dimension") == 12 and factor.get("num_envs") == 1 and
            factor.get("observation_layout") == OWNER_LAYOUT and
            factor.get("reward_only") is True and
            factor.get("reward_revision") == RETENTION_REWARD_REVISION and
            factor.get("reward_local_potential_changed") is True and
            factor.get("shared_task_potential_observation_changed") is False and
            factor.get("observation_semantics_changed") is False and
            factor.get("same_mdp_claimed") is False and
            factor.get("same_numeric_input_Gaussian_and_value_preserved") is True and
            all(factor.get(key) is False for key in (
                "physical_dynamics_changed", "source_pose_commands_changed",
                "control_projection_changed", "HISTORY_changed",
                "conditional_sigma_changed", "old_rollout_inherited")) and
            all(factor.get("added_" + key) == 0 for key in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates")),
            "reward-only72e factor changes policy/control/distribution or invents learning")
    source_checkpoint = Path(receipt.get("source_checkpoint", "")).resolve(strict=True)
    source_manifest = source_checkpoint.with_name(
        source_checkpoint.stem + "_manifest.json").resolve(strict=True)
    require(BASE.sha256(source_checkpoint) == RETENTION_SOURCE_SHA and
            BASE.sha256(source_manifest) == RETENTION_SOURCE_MANIFEST_SHA,
            "reward-only72e immutable CP226048 source bytes changed")
    source_metadata = BASE.read_json(source_manifest)
    require(source_metadata.get("checkpoint_sha256") == RETENTION_SOURCE_SHA and
            source_metadata.get("runtime_contract") == source_runtime and
            source_metadata.get("policy_contract") == policy and
            source_metadata.get("runner_config") == factor.get("source_runner_config") and
            source_metadata.get("stage_requested_decisions") ==
                factor.get("source_stage_requested_decisions") and
            source_metadata.get("optimizer_learning_rate") ==
                factor.get("source_effective_learning_rate") and
            all(source_metadata.get(key) == RETENTION_ORIGIN[key]
                for key in BASE.COUNTERS),
            "reward-only72e immutable source metadata changed")
    preserved = factor.get("preserved_metadata_sha256")
    require(isinstance(preserved, dict) and preserved and
            all(key in source_metadata and BASE.json_digest(source_metadata[key]) == digest
                for key, digest in preserved.items()),
            "reward-only72e source full-state preservation receipt changed")
    for key, digest in preserved.items():
        if key.endswith(("_branch", "_migration")) or key in (
                "checkpoint_output_routing", "runner_config", "policy_contract",
                "seed", "normalization"):
            require(BASE.json_digest(metadata.get(key)) == digest,
                    "reward-only72e descendant changed protected lineage: " + key)
    source_stages = source_metadata.get("stage_requested_decisions") or {}
    current_stages = metadata.get("stage_requested_decisions") or {}
    require(all(type(current_stages.get(key)) is int and current_stages[key] >= value
                for key, value in source_stages.items()),
            "reward-only72e descendant reset used training quantities")
    published_metadata = BASE.read_json(publication["manifest"])
    route = metadata.get("checkpoint_output_routing")
    branch_root = (BASE.OUTPUT_ROOT / "branches" / BASE.RECAPTURE_BRANCH).resolve()
    require(published_metadata.get(RETENTION_MIGRATION) == receipt and
            published_metadata.get("runtime_contract") == runtime and
            published_metadata.get("policy_contract") == policy and
            route == source_metadata.get("checkpoint_output_routing") ==
                published_metadata.get("checkpoint_output_routing") and
            checkpoint.parent == branch_root / "checkpoints" / "history",
            "reward-only72e checkpoint escaped its publication/source lineage")
    branch_counts = {}
    for key in BASE.COUNTERS:
        require(type(counters[key]) is int and counters[key] >= RETENTION_ORIGIN[key],
                "invalid reward-only72e " + key + " origin")
        branch_counts[key] = counters[key] - RETENTION_ORIGIN[key]
    historical = {key: copy.deepcopy(value) for key, value in metadata.items()
                  if key != RETENTION_MIGRATION}
    historical["runtime_contract"] = source_runtime
    revision = {"schema": RETENTION_SCHEMA,
        "reward_revision": RETENTION_REWARD_REVISION,
        "reward_only_boundary": True,
        "policy_control_distribution_changed_at_boundary": False,
        "revision_counter_origin": RETENTION_ORIGIN,
        "revision_branch_counts": branch_counts,
        "migration_plan": str(plan), "migration_plan_sha256": RETENTION_PLAN_SHA,
        "publication": publication["path"],
        "publication_sha256": RETENTION_PUBLICATION_SHA,
        "published_checkpoint": str(publication["checkpoint"]),
        "published_checkpoint_sha256": RETENTION_PUBLISHED_CHECKPOINT_SHA,
        "published_manifest": str(publication["manifest"]),
        "published_manifest_sha256": RETENTION_PUBLISHED_MANIFEST_SHA,
        "added_learning": {"policy_decisions": 0, "ppo_updates": 0,
                           "optimizer_steps": 0, "auxiliary_updates": 0}}
    return historical, source_runtime, revision


def _reject_retention_arguments(args: argparse.Namespace) -> None:
    require(all(getattr(args, name, None) is None for name in (
        "rr_retention_publication", "rr_retention_publication_sha256",
        "rr_retention_plan_sha256")),
        "f6d owner439 export must not carry reward-only72e arguments")


def _reject_front_retention_arguments(args: argparse.Namespace) -> None:
    require(all(getattr(args, name, None) is None for name in (
        "front_retention439_publication",
        "front_retention439_publication_sha256",
        "front_retention439_plan_sha256")),
        "pre-65a export must not carry front-retention439 identity arguments")


def _reject_collection_arguments(args: argparse.Namespace) -> None:
    require(all(getattr(args, name, None) is None for name in (
        "collection512_publication", "collection512_publication_sha256")),
        "pre-collection export must not carry collection512 arguments")


def _inherited_422_identity(metadata: dict[str, Any], runtime: dict[str, Any],
                            policy: dict[str, Any], counters: dict[str, int],
                            checkpoint: Path) -> dict[str, Any]:
    """Run the prior exporter's exact initial->recapture->live->P02->coop chain."""
    progress_receipt = metadata.get(BASE.P02_PROGRESS_MIGRATION)
    cooperative_receipt = metadata.get(BASE.COOPERATIVE_PREP_MIGRATION)
    require(isinstance(progress_receipt, dict) and isinstance(cooperative_receipt, dict),
            "owner439 source lost P02/cooperative ancestry")
    progress_runtime = BASE.reconstruct_cooperative_prep_source_runtime(
        runtime, cooperative_receipt)
    base_runtime = BASE.reconstruct_p02_progress_source_runtime(
        progress_runtime, progress_receipt)
    progress_factor = progress_receipt.get(BASE.P02_PROGRESS_FACTOR_KEY) or {}
    cooperative_factor = cooperative_receipt.get(BASE.COOPERATIVE_PREP_FACTOR_KEY) or {}
    base_policy = progress_factor.get("source_policy_contract") or {}
    progress_policy = cooperative_factor.get("source_policy_contract") or {}
    require(cooperative_factor.get("target_policy_contract") == policy and
            policy.get("version") == BASE.COOPERATIVE_PREP_POLICY_VERSION and
            policy.get("observation_layout") == BASE.P02_PROGRESS_OBSERVATION_LAYOUT and
            policy.get("observation_dimension") == 422,
            "owner439 source is not the exact inherited cooperative422 policy")

    branch = metadata.get("rear_policy_timing_branch")
    counts = metadata.get("rear_policy_timing_branch_counts")
    migration = metadata.get("rear_policy_timing_migration")
    factor = (migration or {}).get("rear_policy_timing_factor")
    require(isinstance(branch, dict) and branch.get("schema") == BASE.MIGRATION_SCHEMA and
            branch.get("migration_added_updates") == 0 and
            isinstance(migration, dict) and migration.get("schema") == BASE.MIGRATION_SCHEMA and
            isinstance(factor, dict) and factor.get("schema") == BASE.MIGRATION_SCHEMA,
            "owner439 source lacks the original rear-policy boundary")
    origin = branch.get("counter_origin")
    require(isinstance(origin, dict) and factor.get("counter_origin") == origin,
            "rear-policy branch origin differs from its migration")
    computed = {}
    for key in BASE.COUNTERS:
        require(type(origin.get(key)) is int and counters[key] >= origin[key],
                f"invalid inherited rear-policy {key} origin")
        computed[key] = counters[key] - origin[key]
    require(counts == computed and factor.get("rear_task_assists_enabled") is False and
            factor.get("FL_capture_assist_enabled") is True and
            factor.get("old_rollout_inherited") is False and
            factor.get("physical_dynamics_changed") is False and
            all(factor.get("added_" + key) == 0 for key in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates")),
            "inherited rear-policy credit/control lineage differs")
    plan = Path(migration.get("plan_path", "")).resolve(strict=True)
    require(BASE.sha256(plan) == migration.get("plan_sha256") and
            BASE.read_json(plan) == {key: value for key, value in migration.items()
                                      if key not in ("plan_path", "plan_sha256")},
            "original rear-policy plan changed after publication")
    initial_source = Path(migration.get("source_checkpoint", "")).resolve(strict=True)
    initial_manifest = initial_source.with_name(
        initial_source.stem + "_manifest.json").resolve(strict=True)
    require(BASE.sha256(initial_source) == migration.get("source_checkpoint_sha256") ==
                branch.get("source_checkpoint_sha256") and
            BASE.sha256(initial_manifest) == migration.get("source_manifest_sha256") and
            factor.get("target_policy_contract") == base_policy,
            "original rear-policy source/policy binding differs")
    preserved = factor.get("preserved_metadata_sha256")
    require(isinstance(preserved, dict), "rear-policy migration lacks preserved lineage")
    auxiliary = BASE.inherited_auxiliary(metadata, preserved)
    live_receipt = metadata.get("rear_live_swing_migration")
    recapture_runtime = BASE.reconstruct_live_swing_source_runtime(base_runtime, live_receipt)
    recapture = BASE.rear_recapture_identity(
        metadata, recapture_runtime, migration, base_policy, counters, checkpoint)
    live_swing = BASE.rear_live_swing_identity(
        metadata, base_runtime, recapture_runtime, base_policy, counters,
        checkpoint, recapture)
    p02_progress = BASE.p02_progress_identity(
        metadata, progress_runtime, base_runtime, base_policy, progress_policy,
        counters, checkpoint, live_swing)
    cooperative = BASE.cooperative_prep_identity(
        metadata, runtime, progress_runtime, progress_policy, policy,
        counters, checkpoint, p02_progress)
    require(all(value is not None for value in
                (recapture, live_swing, p02_progress, cooperative)),
            "owner439 source ancestry is incomplete")
    return {"branch_counter_origin": origin, "rear_policy_branch_counts": computed,
            "rear_recapture_revision": recapture,
            "rear_live_swing_revision": live_swing,
            "p02_progress_revision": p02_progress,
            "cooperative_preparation_revision": cooperative,
            "inherited_auxiliary": auxiliary,
            "nominal_timing": BASE.LIVE_SWING_MODE,
            "migration_plan": str(plan),
            "migration_plan_sha256": migration["plan_sha256"],
            "migration_source_checkpoint": str(initial_source),
            "migration_source_checkpoint_sha256": migration["source_checkpoint_sha256"],
            "migration_source_manifest_sha256": migration["source_manifest_sha256"]}


def checkpoint_identity(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    proof = manifest.get("checkpoint_load_provenance") or {}
    require(proof.get("checkpoint_loaded_and_verified") is True and
            manifest.get("policy_sampling_mode") == "deterministic_conditional_mean" and
            proof.get("stochastic_policy") in (None, False) and
            proof.get("policy_seed") is None,
            "source is not deterministic evaluation of a verified checkpoint")
    binding = proof.get("source") or {}
    checkpoint = Path(binding.get("checkpoint", "")).resolve(strict=True)
    sidecar = Path(binding.get("manifest", "")).resolve(strict=True)
    expected_checkpoint = Path(args.checkpoint).resolve(strict=True)
    expected_sidecar = expected_checkpoint.with_name(
        expected_checkpoint.stem + "_manifest.json").resolve(strict=True)
    checkpoint_sha = BASE.checked_sha(args.checkpoint_sha256, "--checkpoint-sha256")
    sidecar_sha = BASE.checked_sha(args.checkpoint_manifest_sha256,
                                   "--checkpoint-manifest-sha256")
    require(checkpoint == expected_checkpoint and sidecar == expected_sidecar and
            BASE.sha256(checkpoint) == checkpoint_sha == binding.get("checkpoint_sha256") and
            BASE.sha256(sidecar) == sidecar_sha == binding.get("manifest_sha256"),
            "video checkpoint differs from explicit owner439 bytes")
    metadata = BASE.read_json(sidecar)
    counters = {key: getattr(args, "expected_" + key) for key in BASE.COUNTERS}
    require(all(type(value) is int and value >= 0 for value in counters.values()) and
            all(metadata.get(key) == value for key, value in counters.items()) and
            proof.get("saved_global_policy_decisions") == counters["global_policy_decisions"] and
            metadata.get("checkpoint_sha256") == checkpoint_sha and
            Path(metadata.get("checkpoint_path", "")).resolve() == checkpoint,
            "checkpoint counters/path differ from explicit invocation")
    expected_head = BASE.checked_head(args.expected_head)
    is_collection = expected_head == COLLECTION_TARGET_HEAD
    is_front_retention = expected_head == FRONT_RETENTION_TARGET_HEAD
    is_retention = expected_head in (
        RETENTION_TARGET_HEAD, FRONT_RETENTION_TARGET_HEAD,
        COLLECTION_TARGET_HEAD)
    require(expected_head in (
                OWNER_TARGET_HEAD, RETENTION_TARGET_HEAD,
                FRONT_RETENTION_TARGET_HEAD, COLLECTION_TARGET_HEAD) and
            args.checkpoint_runtime_head is None,
            "owner439 adapter accepts only exact f6d/72e/65a/collection512 runtimes")
    runtime = metadata.get("runtime_contract") or {}
    expected_runtime_sha = (FRONT_RETENTION_TARGET_RUNTIME_SHA
                            if is_front_retention else
                            RETENTION_TARGET_RUNTIME_SHA if is_retention else
                            OWNER_TARGET_RUNTIME_SHA)
    expected_contract_sha = (FRONT_RETENTION_TARGET_CONTRACT_SHA
                             if is_front_retention else
                             RETENTION_TARGET_CONTRACT_SHA if is_retention else
                             OWNER_TARGET_CONTRACT_SHA)
    require(runtime.get("experiment_id") == BASE.EXPERIMENT and
            runtime.get("source_git_commit") == expected_head and
            (is_collection or
             (runtime.get("runtime_content_sha256") == expected_runtime_sha and
              BASE.json_digest(runtime) == expected_contract_sha)) and
            (not is_collection or
             BASE.json_digest(runtime.get("files") or {}) ==
                runtime.get("runtime_content_sha256")),
            "checkpoint is not from a reviewed owner439/72e/front-retention439 runtime")
    require((manifest.get("runtime_contract") or {}) == runtime and
            proof.get("checkpoint_runtime_compatibility") is None,
            "owner439 evaluation must use the exact checkpoint runtime, not media compatibility")

    retention_revision = None
    front_retention_revision = None
    front_retention_auxiliary = None
    collection_revision = None
    lineage_counters = counters
    lineage_checkpoint = checkpoint
    if is_collection:
        collection_source, collection_source_runtime, collection_revision = \
            _collection_boundary(metadata, runtime, counters, checkpoint, args)
        source_counters = {key: collection_source[key] for key in BASE.COUNTERS}
        source_checkpoint = Path(
            collection_revision["source_checkpoint"]["checkpoint"])
        front_metadata, retention_runtime, front_retention_revision, \
            front_retention_auxiliary = _front_retention_boundary(
                collection_source, collection_source_runtime, source_counters,
                source_checkpoint, args)
        lineage_metadata, owner_runtime, retention_revision = _retention_boundary(
            front_metadata, retention_runtime, source_counters,
            source_checkpoint, args)
        # Historical revision totals stop at the bound 65a source. The outer
        # collection revision exclusively accounts for post-boundary updates.
        lineage_counters = source_counters
        lineage_checkpoint = source_checkpoint
    elif is_front_retention:
        _reject_collection_arguments(args)
        front_metadata, retention_runtime, front_retention_revision, \
            front_retention_auxiliary = _front_retention_boundary(
                metadata, runtime, counters, checkpoint, args)
        lineage_metadata, owner_runtime, retention_revision = _retention_boundary(
            front_metadata, retention_runtime, counters, checkpoint, args)
    elif is_retention:
        _reject_front_retention_arguments(args)
        _reject_collection_arguments(args)
        lineage_metadata, owner_runtime, retention_revision = _retention_boundary(
            metadata, runtime, counters, checkpoint, args)
    else:
        _reject_retention_arguments(args)
        _reject_front_retention_arguments(args)
        _reject_collection_arguments(args)
        lineage_metadata, owner_runtime = metadata, runtime
    publication = _validate_publication(args)
    receipt = lineage_metadata.get(OWNER_MIGRATION)
    factor = receipt.get(OWNER_FACTOR_KEY) if isinstance(receipt, dict) else None
    require(isinstance(factor, dict) and receipt.get("schema") == OWNER_SCHEMA and
            factor.get("schema") == OWNER_SCHEMA and
            receipt.get("source_checkpoint_sha256") == OWNER_SOURCE_SHA and
            receipt.get("source_manifest_sha256") == OWNER_SOURCE_MANIFEST_SHA and
            receipt.get("source_git_commit") == OWNER_SOURCE_HEAD and
            receipt.get("target_git_commit") == OWNER_TARGET_HEAD and
            receipt.get("target_runtime_content_sha256") == OWNER_TARGET_RUNTIME_SHA and
            receipt.get("target_contract_sha256") == OWNER_TARGET_CONTRACT_SHA,
            "checkpoint lacks the exact published owner439 boundary")
    plan = Path(receipt.get("plan_path", "")).resolve(strict=True)
    plan_sha = BASE.checked_sha(args.rear_owner_plan_sha256,
                                "--rear-owner-plan-sha256")
    require(plan.parent == OUTPUT_ROOT.resolve() and
            BASE.sha256(plan) == receipt.get("plan_sha256") == plan_sha == OWNER_PLAN_SHA and
            BASE.read_json(plan) == {key: value for key, value in receipt.items()
                                     if key not in ("plan_path", "plan_sha256")},
            "owner439 migration plan differs from its embedded receipt")
    published_metadata = BASE.read_json(publication["manifest"])
    require(published_metadata.get(OWNER_MIGRATION) == receipt and
            published_metadata.get("runtime_contract") == owner_runtime and
            published_metadata.get("policy_contract") == metadata.get("policy_contract"),
            "ordinary owner439 checkpoint no longer descends from the published boundary")
    source_runtime = reconstruct_owner_source_runtime(owner_runtime, receipt)
    source_policy = factor.get("source_policy_contract") or {}
    policy = metadata.get("policy_contract") or {}
    require(factor.get("target_policy_contract") == policy and
            policy.get("version") == OWNER_POLICY and
            policy.get("actor_class") == OWNER_ACTOR and
            policy.get("source_policy_version") == BASE.COOPERATIVE_PREP_POLICY_VERSION and
            policy.get("observation_layout") == OWNER_LAYOUT and
            policy.get("observation_dimension") == 439 and
            policy.get("preserved_observation_prefix_dimension") == 422 and
            policy.get("rear_owner_mode") == OWNER_MODE and
            policy.get("rear_owner_group") == OWNER_GROUP and
            policy.get("rear_owner_slice") == [422, 439] and
            policy.get("rear_owner_fields") == list(OWNER_FIELDS) and
            policy.get("rear_owner_scales") == list(OWNER_SCALES) and
            policy.get("sigma_kernel_observation_slice") == [0, 422] and
            policy.get("stochastic_kernel_change") is False and
            policy.get("raw_action_dimension") == 12 and
            policy.get("action_transform") ==
                "existing_single_mapper; suspended_dependent_FL_RL_servo_owner_uses_"
                "committed_FINAL_anchor_plus_bounded_request_change; wheels_and_explicit_"
                "stops_preserved; rear_completion_assists_OFF_FLassist_declared",
            "checkpoint lacks the exact owner439 policy/action contract")
    require(factor.get("revision_counter_origin") == OWNER_ORIGIN and
            factor.get("original_branch_origin") ==
                BASE.RECAPTURE_SOURCE_SELECTION["counters"] and
            factor.get("rear_owner_recovery_mode") == OWNER_MODE and
            factor.get("cooperative_preparation_revision") == OWNER_PROXY_REVISION and
            factor.get("actor_critic_mapping") == "old422_columns_exact_new17_zero" and
            factor.get("Adam_mapping") ==
                "all_old_moments_steps_options_exact_new17_first_weight_moment_columns_zero" and
            factor.get("old422_codec_preserved") is True and
            factor.get("old_rollout_inherited") is False and
            factor.get("physical_dynamics_changed") is False and
            factor.get("same_mdp_claimed") is False and
            all(factor.get("added_" + key) == 0 for key in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates")),
            "owner439 factor changes protected mapping/state or invents learning")
    require(lineage_metadata.get("runner_config") == factor.get("target_runner_config"),
            "owner439 runner differs from its published factor")
    preserved = factor.get("preserved_metadata_sha256") or {}
    for key, digest in preserved.items():
        if key.endswith(("_branch", "_migration")) or key == "checkpoint_output_routing":
            require(BASE.json_digest(lineage_metadata.get(key)) == digest,
                    "owner439 checkpoint changed protected source lineage: " + key)
    branch_root = (BASE.OUTPUT_ROOT / "branches" / BASE.RECAPTURE_BRANCH).resolve()
    route = lineage_metadata.get("checkpoint_output_routing")
    require(route == published_metadata.get("checkpoint_output_routing") and
            checkpoint.parent == branch_root / "checkpoints" / "history",
            "owner439 checkpoint escaped the isolated recapture branch")
    source_checkpoint = Path(receipt.get("source_checkpoint", "")).resolve(strict=True)
    source_manifest = source_checkpoint.with_name(
        source_checkpoint.stem + "_manifest.json").resolve(strict=True)
    require(BASE.sha256(source_checkpoint) == OWNER_SOURCE_SHA and
            BASE.sha256(source_manifest) == OWNER_SOURCE_MANIFEST_SHA,
            "owner439 learned225280 source bytes changed")
    owner_counts = {}
    for key in BASE.COUNTERS:
        require(lineage_counters[key] >= OWNER_ORIGIN[key],
                f"invalid owner439 {key} origin")
        owner_counts[key] = lineage_counters[key] - OWNER_ORIGIN[key]
    inherited = _inherited_422_identity(
        lineage_metadata, source_runtime, source_policy,
        lineage_counters, lineage_checkpoint)
    actor_hash = (proof.get("parameter_hashes") or {}).get("actor_parameter_sha256")
    require(isinstance(actor_hash, str) and
            metadata.get("actor_parameter_sha256") == actor_hash,
            "loaded actor differs from the owner439 checkpoint sidecar")
    identity = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "manifest": str(sidecar), "manifest_sha256": sidecar_sha,
        "actor_parameter_sha256": actor_hash, "policy_version": OWNER_POLICY,
        "observation_layout": OWNER_LAYOUT, "observation_dimension": 439,
        "lifetime_counters": counters, **inherited,
        "rear_owner_recovery_revision": {
            "schema": OWNER_SCHEMA, "mode": OWNER_MODE,
            "timing_revision": OWNER_TIMING_REVISION,
            "cooperative_proxy_revision": OWNER_PROXY_REVISION,
            "revision_counter_origin": OWNER_ORIGIN,
            "revision_branch_counts": owner_counts,
            "migration_plan": str(plan), "migration_plan_sha256": OWNER_PLAN_SHA,
            "publication": publication["path"],
            "publication_sha256": OWNER_PUBLICATION_SHA,
            "published_checkpoint": str(publication["checkpoint"]),
            "published_checkpoint_sha256": OWNER_PUBLISHED_CHECKPOINT_SHA,
            "published_manifest": str(publication["manifest"]),
            "published_manifest_sha256": OWNER_PUBLISHED_MANIFEST_SHA,
            "added_learning": {"policy_decisions": 0, "ppo_updates": 0,
                               "optimizer_steps": 0, "auxiliary_updates": 0}},
        "migration_added_learning": {"policy_decisions": 0, "ppo_updates": 0,
                                     "optimizer_steps": 0, "auxiliary_updates": 0},
        "this_run_auxiliary_updates": 0,
        "FL_capture_assist_enabled": True, "rear_task_assists_enabled": False,
        "runtime_head": expected_head, "checkpoint_runtime_head": expected_head,
        "evaluation_runtime_head": expected_head,
        "reviewed_media_only_runtime_compatibility": None}
    if retention_revision is not None:
        identity["rr_retention_reward_revision"] = retention_revision
    if front_retention_revision is not None:
        identity["front_retention439_runtime_revision"] = \
            front_retention_revision
        identity["front_retention439_auxiliary"] = front_retention_auxiliary
    if collection_revision is not None:
        identity["collection_horizon439_revision"] = collection_revision
    return identity


def _fixed_historical_n() -> dict[str, Any]:
    source = Path(BASE.DEFAULT_HISTORICAL_N).resolve(strict=True)
    run_manifest = source.parent / "run_manifest.json"
    manifest = source / "semantic_video_source_manifest.json"
    video = source / "actual_viewport_video.mp4"
    ledger = source / "viewport_frame_ledger.jsonl"
    require(BASE.sha256(manifest) == HISTORICAL_N_SOURCE_MANIFEST_SHA and
            BASE.sha256(run_manifest) == HISTORICAL_N_RUN_MANIFEST_SHA and
            BASE.sha256(video) == HISTORICAL_N_VIDEO_SHA and
            BASE.sha256(ledger) == HISTORICAL_N_LEDGER_SHA,
            "fixed historical N reference bytes changed")
    payload = BASE.read_json(manifest)
    require((payload.get("runtime_contract") or {}).get("source_git_commit") ==
                HISTORICAL_N_HEAD and
            payload.get("episode_physics_ticks") == HISTORICAL_N_TICKS,
            "fixed historical N identity/endpoint changed")
    return BASE.shared().sealed_source(source, candidate=False)


def detail_plan(rows: list[dict[str, Any]], step: int, success: bool) -> dict[str, Any]:
    suffix = "" if success else "_INCOMPLETE"
    rear = BASE.rr_reached(rows)
    rl = BASE.rl_reached(rows)
    rr_placed = any(item.get("rr_placed") is True for item in rows)
    if rear:
        start = next(index for index, item in enumerate(rows) if
            item["phase"] in ("P07", "P08", "P09", "P10", "P11", "P12", "P13") or
            item["rear_policy_timing"]["rr_carry_capture"] or
            item["rear_policy_timing"]["rr_support_handoff"])
        kind = "FL_TO_FR_RL_RECOVERY_ACTUAL_WINDOW"
        rr_label = ("RR CAPTURE OBSERVED" if rr_placed else
                    "RR CAPTURE INCOMPLETE")
        title = (f"DETAIL | CP{step} | FL to FR / RL RECOVERY | "
                 f"{rr_label} | {'RL WINDOW OBSERVED' if rl else 'RL NOT REACHED'} | "
                 f"{'SUCCESS' if success else 'INCOMPLETE'}")
        unavailable = None
        filename = (f"CP{step}_DET_RR_to_RL_detail{suffix}.mp4" if rl else
                    f"CP{step}_DET_RR_capture_detail_INCOMPLETE.mp4")
    else:
        start = max(0, len(rows) - 60 * BASE.FPS)
        kind = "RR_NOT_REACHED_PREDECESSOR_FAILURE"
        title = (f"DETAIL | CP{step} | FL to FR / RL RECOVERY | "
                 "RR NOT REACHED | PREDECESSOR FAILURE TAIL")
        unavailable = "THIS_EPISODE_DID_NOT_REACH_RR_WINDOW"
        filename = (f"CP{step}_DET_RR_NOT_REACHED_"
                    "PREDECESSOR_FAILURE_detail_INCOMPLETE.mp4")
    return {"kind": kind, "start": start, "end": len(rows),
        "filename": filename,
        "title": title, "requested_RR_detail_unavailable_reason": unavailable,
        "same_episode_actual_contiguous_tail": True,
        "RR_window_reached": rear, "RL_window_reached": rl,
        "RR_placed_history_observed": rr_placed,
        "RR_capture_or_task_success_inferred_from_phase": False}


def owner_status_lines(identity: dict[str, Any]) -> tuple[str, str]:
    """Public labels derived from checkpoint lineage, never manifest nominal_timing."""
    retention = identity.get("rr_retention_reward_revision")
    front_retention = identity.get("front_retention439_runtime_revision")
    front_aux = identity.get("front_retention439_auxiliary")
    collection = identity.get("collection_horizon439_revision")
    front_preservation = identity.get("front_preservation439_branch_identity")
    if front_preservation is not None:
        line_one = ("FL ASSIST ON | RR/RL TASK ASSIST OFF | "
                    "ISSUED-OWNER SUSPENSION/PROJECTION ON")
        counts = front_preservation["revision_branch_counts"]
        replay = identity["front_replay_counts"]
        line_two = (f"CP225280/f6d restored | branch +"
                    f"{counts['global_policy_decisions']}d/"
                    f"{counts['ppo_updates']}P/{counts['optimizer_steps']}A | "
                    f"replay-KL {replay['replay_row_exposures']} exposures "
                    "(not PPO samples/AUX steps)")
    elif collection is not None:
        line_one = ("FL ASSIST ON | RR/RL TASK ASSIST OFF | "
                    "ISSUED-OWNER SUSPENSION/PROJECTION ON")
        old = front_retention["revision_branch_counts"]
        new = collection["revision_branch_counts"]
        line_two = (f"OWNER439/reward72e | 65a +{old['global_policy_decisions']}d/"
                    f"{old['ppo_updates']}P/{old['optimizer_steps']}A | "
                    f"collection512 +{new['global_policy_decisions']}d/"
                    f"{new['ppo_updates']}P/{new['optimizer_steps']}A | "
                    f"AUX {front_aux['accepted_auxiliary_updates']}/"
                    f"{front_aux['attempted_auxiliary_optimizer_steps']} not-PPO")
    elif front_retention is not None:
        line_one = ("FL ASSIST ON | RR/RL TASK ASSIST OFF | "
                    "ISSUED-OWNER SUSPENSION/PROJECTION ON")
        counts = front_retention["revision_branch_counts"]
        line_two = (f"OWNER439 | edge-v4 | reward-only72e | acct65a "
                    f"+{counts['global_policy_decisions']}d/"
                    f"{counts['ppo_updates']}PPO/{counts['optimizer_steps']}Adam | "
                    f"front-ret AUX {front_aux['accepted_auxiliary_updates']}/"
                    f"{front_aux['attempted_auxiliary_optimizer_steps']} (not PPO)")
    elif retention is not None:
        line_one = ("FL ASSIST ON | RR/RL TASK ASSIST OFF | "
                    "ISSUED-OWNER SUSPENSION/PROJECTION ON")
        counts = retention["revision_branch_counts"]
        line_two = (f"OWNER439 issued-suspension-v1 | edge-v4 | reward-only72e boundary | "
                    f"learned +{counts['global_policy_decisions']}d/"
                    f"{counts['ppo_updates']}PPO/{counts['optimizer_steps']}Adam | AUX +0")
    else:
        line_one = ("FL ASSIST ON | REAR OWNER SUSPENSION/PROJECTION ON | "
                    "REAR CAPTURE/GEOMETRY COMPLETION OFF")
        owner = identity["rear_owner_recovery_revision"]
        counts = owner["revision_branch_counts"]
        line_two = (f"OWNER439 issued-suspension-v1 | edge-v4 | owner "
                    f"+{counts['global_policy_decisions']}d/"
                    f"{counts['ppo_updates']}PPO/{counts['optimizer_steps']}Adam | "
                    "inherited coop-v4 -> proxy-v5 | migration/AUX +0")
    return line_one, line_two


def owner_visible_labels(identity: dict[str, Any]) -> dict[str, Any]:
    """Receipt labels; the legacy source nominal_timing is never copied here."""
    labels = {"deterministic_residual_PPO": True,
        "front_FL_assist": "ON", "front_FL_assist_mode": BASE.FL_ASSIST_MODE,
        "rear_owner_suspension_projection": "ON", "rear_owner_mode": OWNER_MODE,
        "rear_capture_geometry_completion_assist": "OFF",
        "rear_capture_geometry_completion_assist_scope":
            "no rear capture or geometry-completion target assist",
        "policy_version": OWNER_POLICY,
        "observation_layout": OWNER_LAYOUT, "observation_dimension": 439,
        "owner_mode": OWNER_MODE, "timing_revision": OWNER_TIMING_REVISION,
        "training": "PPO + inherited limited AUX", "this_run_AUX_updates": 0}
    retention = identity.get("rr_retention_reward_revision")
    if retention is not None:
        labels.update({"RR_RL_task_assists": "OFF",
            "issued_owner_suspension_projection": "ON",
            "issued_owner_suspension_projection_role":
                "control/projection contribution; not RR/RL task-completion assist",
            "reward_revision": retention["reward_revision"],
            "reward_only_boundary": True,
            "source_manifest_nominal_timing_used_as_authority": False})
    front_revision = identity.get("front_retention439_runtime_revision")
    front_aux = identity.get("front_retention439_auxiliary")
    if front_revision is not None:
        labels.update({
            "front_retention439_runtime_identity": "accounting-only; added learning 0",
            "front_retention439_auxiliary_ledger": front_aux,
            "front_retention439_AUX_is_PPO": False,
            "front_retention439_AUX_is_physical_success": False,
            "training": ("PPO + inherited limited AUX + separately accounted "
                         "front-retention439 AUX"),
        })
    collection = identity.get("collection_horizon439_revision")
    if collection is not None:
        labels.update({
            "collection_horizon439": collection,
            "collection_profile": COLLECTION_PROFILE,
            "collection_steps_per_update": 512,
            "optimizer_steps_per_update": 20,
            "historical_128_counters_reinterpreted": False,
            "collection_boundary_added_learning": False,
            "training": ("PPO with explicit collection512 + inherited limited AUX + "
                         "separately accounted front-retention439 AUX"),
        })
    front_preservation = identity.get("front_preservation439_branch_identity")
    if front_preservation is not None:
        replay = identity["front_replay_counts"]
        labels.update({
            "front_preservation439_branch_identity": front_preservation,
            "front_baseline_origin":
                "restored exact f6d439 CP225280 learned state at zero-update boundary",
            "front_preservation_physical_success_proven_by_replay": False,
            "front_replay_regularization": "front-only conditional-Gaussian KL",
            "front_replay_counts": replay,
            "front_replay_rows_are_on_policy_samples": False,
            "front_replay_adds_separate_optimizer_steps": False,
            "front_replay_is_AUX": False,
            "legacy_inherited_source_auxiliary":
                identity.get("legacy_inherited_source_auxiliary"),
            "front_retention439_AUX32_inherited": False,
            "training": ("PPO + front-only replay KL in the same official Adam "
                         "minibatches + legacy source AUX lineage only"),
        })
    return labels


def owner_panel_lines(row: dict[str, Any], outcome: dict[str, Any],
                      identity: dict[str, Any]) -> list[str]:
    # Keep the validated base renderer bound before export() temporarily installs
    # this owner-specific wrapper on BASE.panel_lines.
    lines = list(BASE_PANEL_LINES(row, outcome, identity))
    lines[1], lines[2] = owner_status_lines(identity)
    return lines


def export(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).resolve(strict=True)
    destination = Path(args.destination).resolve()
    require(destination.is_relative_to(OUTPUT_ROOT.resolve()) and not destination.exists(),
            "destination must be a new directory in ppo_rl_recovery_learning_v1")
    original_identity = BASE.checkpoint_identity
    BASE.checkpoint_identity = checkpoint_identity
    try:
        candidate = BASE.sealed_source(source, args)
    finally:
        BASE.checkpoint_identity = original_identity
    require(candidate["diagnostic_partial"] is False,
            "owner439 RL completion export requires a sealed full episode, not capture abort")
    baseline = _fixed_historical_n()
    ffmpeg = BASE.shared().find_ffmpeg(
        candidate["capture"].get("full_decode", {}).get("ffmpeg_path"))
    _, ledger, source_validation = BASE.shared().checked_media(candidate, ffmpeg=ffmpeg)
    rows, selected = BASE.capture_rows(candidate, ledger)
    require(0 < len(rows) <= BASE.MAX_FRAMES,
            "source exceeds one continuous 200-second episode")
    outcome = BASE.terminal_evidence(candidate)
    identity = candidate["checkpoint"]
    step = identity["lifetime_counters"]["global_policy_decisions"]
    suffix = "" if outcome["success"] else "_INCOMPLETE"
    plan = detail_plan(rows, step, outcome["success"])
    destination.mkdir(parents=True)
    original_panel = BASE.panel_lines
    BASE.panel_lines = owner_panel_lines
    try:
        full = BASE.encode_full(
            candidate, rows,
            destination / f"CP{step}_DET_full_RL_completion_attempt{suffix}.mp4",
            outcome, identity, ffmpeg)
    finally:
        BASE.panel_lines = original_panel
    print("FULL_PLAYABLE_VALIDATED " + full["output"], flush=True)
    detail = BASE.encode_detail(Path(full["output"]), rows, plan, destination, ffmpeg)
    pair = BASE.encode_pair(
        baseline, {**candidate, "frame_count": len(rows)}, Path(full["output"]),
        destination / f"N_vs_CP{step}_DET_same_camera.mp4",
        step, outcome["result"], ffmpeg)
    evidence_path = destination / "selected_real_events.json"
    BASE.write_new_json(evidence_path, {
        "schema": "wlr50_clean.rl_recovery_video_selected_events.v1",
        "source": str(source),
        "selection": "actual phase/timing/contact transitions plus endpoint; no interpolation",
        "rows": selected, "full_per_tick_evidence": str(candidate["tick_path"])})
    retention = identity.get("rr_retention_reward_revision")
    front_retention = identity.get("front_retention439_runtime_revision")
    collection = identity.get("collection_horizon439_revision")
    front_preservation = identity.get("front_preservation439_branch_identity")
    receipt = {
        "schema": ("wlr50_clean.owner439_rl_recovery_video_export.v5"
                   if front_preservation is not None else
                   "wlr50_clean.owner439_rl_recovery_video_export.v4"
                   if collection is not None else
                   "wlr50_clean.owner439_rl_recovery_video_export.v3"
                   if front_retention is not None else
                   "wlr50_clean.owner439_rl_recovery_video_export.v2"
                   if retention is not None else
                   "wlr50_clean.owner439_rl_recovery_video_export.v1"),
        "source": str(source), "source_manifest": str(candidate["manifest_path"]),
        "source_manifest_sha256": BASE.sha256(candidate["manifest_path"]),
        "source_run_manifest_sha256": BASE.sha256(candidate["run_manifest_path"]),
        "source_video_validation": {key: source_validation.get(key) for key in (
            "sha256", "bytes", "valid", "full_decode", "frame_count", "fps",
            "resolution", "duration_s", "frame_pts_sha256",
            "decoded_frame_checksums_sha256", "black_like_frame_count",
            "timestamps_monotonic", "timestamps_continuous")},
        "checkpoint_identity": identity,
        "rear_owner_recovery_revision": identity["rear_owner_recovery_revision"],
        "experiment_id": BASE.EXPERIMENT, "control_method": BASE.CONTROL_METHOD,
        "visible_labels": owner_visible_labels(identity),
        "physical_result": outcome["result"],
        "physical_task_success": outcome["success"],
        "termination_reason": outcome["termination_reason"],
        "termination_source": outcome["termination_source"],
        "source_acceptance_error": outcome["source_acceptance_error"],
        "RR_window_reached": BASE.rr_reached(rows),
        "RR_placed_history_observed": plan["RR_placed_history_observed"],
        "RL_window_reached": BASE.rl_reached(rows),
        "requested_RR_detail_unavailable_reason":
            plan["requested_RR_detail_unavailable_reason"],
        "full_episode_continuous": True, "full_failure_tail_preserved": True,
        "normal_speed": True, "single_episode": True, "stitched": False,
        "speed_modified": False, "extra_intro_frames": 0,
        "historical_N": {"source": str(BASE.DEFAULT_HISTORICAL_N),
            "source_manifest_sha256": HISTORICAL_N_SOURCE_MANIFEST_SHA,
            "run_manifest_sha256": HISTORICAL_N_RUN_MANIFEST_SHA,
            "video_sha256": HISTORICAL_N_VIDEO_SHA,
            "ledger_sha256": HISTORICAL_N_LEDGER_SHA,
            "runtime_head": HISTORICAL_N_HEAD, "episode_physics_ticks": HISTORICAL_N_TICKS,
            "fresh_same_controller_B": False,
            "freeze_after_own_endpoint_is_physical_evidence": False},
        "full": full, "detail": detail, "historical_N_comparison": pair,
        "selected_real_events": str(evidence_path),
        "selected_real_events_sha256": BASE.sha256(evidence_path),
        "claims": {"owner439_migration_added_learning": False,
            "rear_owner_suspension_or_projection_active": True,
            "rear_capture_or_geometry_completion_assist_active": False,
            "front_FL_assist_is_policy_learning": False,
            "RR_window_or_phase_is_capture_success": False,
            "unreached_RR_detail_is_substituted_from_another_run": False,
            "historical_N_is_same_version_or_same_controller": False,
            "frozen_comparison_frames_are_physical_evidence": False,
            "failure_video_is_success": False},
        "code_binding": {"adapter_sha256": BASE.sha256(Path(__file__)),
            "base_exporter": str(BASE_EXPORTER),
            "base_exporter_sha256": BASE.sha256(BASE_EXPORTER),
            "shared_media_helper_sha256": BASE.sha256(BASE.SHARED_EXPORTER)}}
    if retention is not None:
        receipt["rr_retention_reward_revision"] = retention
        receipt["claims"].update({
            "reward_only72e_migration_added_learning": False,
            "reward_only72e_changed_policy_control_or_distribution": False,
            "post_reward_boundary_PPO_learning_present":
                retention["revision_branch_counts"]["ppo_updates"] > 0,
            "source_manifest_nominal_timing_copied_to_visible_labels": False,
        })
    if front_retention is not None:
        receipt["front_retention439_runtime_revision"] = front_retention
        receipt["front_retention439_auxiliary"] = \
            identity["front_retention439_auxiliary"]
        receipt["claims"].update({
            "front_retention439_runtime_identity_added_learning": False,
            "front_retention439_AUX_is_PPO": False,
            "front_retention439_AUX_is_physical_success": False,
            "front_retention439_AUX_is_teacher_deployment": False,
            "front_retention439_AUX_counts_merged_with_inherited_AUX": False,
        })
    if collection is not None:
        receipt["collection_horizon439_revision"] = collection
        receipt["claims"].update({
            "collection512_boundary_added_learning": False,
            "collection512_changed_reward_control_policy_or_distribution": False,
            "historical_128_counters_reinterpreted_as_collection512": False,
            "collection512_updates_are_counted_as_512_decisions_1_PPO_20_Adam": True,
            "front_retention439_AUX_counts_merged_with_collection_PPO": False,
        })
    if front_preservation is not None:
        receipt["front_preservation439_branch_identity"] = front_preservation
        receipt["front_replay_counts"] = identity["front_replay_counts"]
        receipt["legacy_inherited_source_auxiliary"] = \
            identity.get("legacy_inherited_source_auxiliary")
        receipt["claims"].update({
            "front_preservation_boundary_added_learning": False,
            "front_preservation_replay_proves_closed_loop_physical_retention": False,
            "front_replay_rows_count_as_on_policy_samples": False,
            "front_replay_adds_separate_optimizer_steps": False,
            "front_replay_is_AUX_or_PPO_credit": False,
            "front_replay_uses_same_official_PPO_Adam_minibatches": True,
            "legacy_source_AUX_is_new_front_preservation_AUX": False,
            "front_retention439_AUX32_inherited": False,
        })
        receipt["code_binding"]["front_preservation_adapter"] = \
            identity["front_preservation_export_adapter"]["path"]
        receipt["code_binding"]["front_preservation_adapter_sha256"] = \
            identity["front_preservation_export_adapter"]["sha256"]
    receipt_path = destination / "export_receipt.json"
    BASE.write_new_json(receipt_path, receipt)
    print(json.dumps({"physical_result": outcome["result"], "full": full["output"],
        "detail": detail["output"], "comparison": pair["output"],
        "receipt": str(receipt_path)}, indent=2))
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--source", type=Path, required=True)
    result.add_argument("--destination", type=Path, required=True)
    result.add_argument("--source-manifest-sha256", required=True)
    result.add_argument("--source-run-manifest-sha256", required=True)
    result.add_argument("--expected-head", required=True)
    result.add_argument("--checkpoint-runtime-head", default=None,
        help=argparse.SUPPRESS)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--checkpoint-manifest-sha256", required=True)
    result.add_argument("--expected-global-policy-decisions", type=int, required=True)
    result.add_argument("--expected-ppo-updates", type=int, required=True)
    result.add_argument("--expected-optimizer-steps", type=int, required=True)
    result.add_argument("--rear-owner-publication", type=Path, required=True)
    result.add_argument("--rear-owner-publication-sha256", required=True)
    result.add_argument("--rear-owner-plan-sha256", required=True)
    result.add_argument("--rr-retention-publication", type=Path, default=None)
    result.add_argument("--rr-retention-publication-sha256", default=None)
    result.add_argument("--rr-retention-plan-sha256", default=None)
    result.add_argument("--front-retention439-publication", type=Path, default=None)
    result.add_argument("--front-retention439-publication-sha256", default=None)
    result.add_argument("--front-retention439-plan-sha256", default=None)
    result.add_argument("--collection512-publication", type=Path, default=None)
    result.add_argument("--collection512-publication-sha256", default=None)
    result.set_defaults(diagnostic_capture_abort=False,
                        expected_new_auxiliary_updates=0,
                        historical_n_source=BASE.DEFAULT_HISTORICAL_N)
    return result


if __name__ == "__main__":
    export(parser().parse_args())

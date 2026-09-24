"""Optional finite AUX accounting; unchanged439 runtime identity and strict lineage.

No fit, sampler, controller, or CLI runner lives here. Historical migration and
branch objects are immutable. Only the new top-level ledger is appendable.
"""
from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path

from .semantic_rear_policy_timing_migration import COUNTERS, EXPERIMENT
from .semantic_rear_recapture_migration import BRANCH_NAME, CODE
from .semantic_rear_owner_profile import REAR_OWNER_POLICY, REAR_OWNER_OBSERVATION_LAYOUT

SCHEMA = "wlr50_clean.front_retention439_runtime_identity.v1"
FACTOR_KEY = "front_retention439_identity_factor"
IDENTITY = "front_retention439_runtime_identity"
LEDGER = "front_retention439_auxiliary"
LEDGER_SCHEMA = "wlr50_clean.front_retention439_auxiliary_ledger.v1"
FIT_SCHEMA = "wlr50_clean.finite_front_retention439_phase_columns_aux.v1.fit"
DATA_SCHEMA = "wlr50_clean.student_probe_v2_front_retention439_rows.v1"
SOURCE_HEAD = "72e63592bdf412d375abfda29b03cf456fbf4f7e"
ALLOWED = frozenset(CODE + name for name in (
    "semantic_front_retention439.py", "semantic_rear_policy_timing_migration.py", "semantic_training.py"))
PARAMETERS = ["actor.mlp.0.weight[:,[1,4,5,8]]"]
COLUMNS = [1, 4, 5, 8]
PPO_ZERO = ("PPO_decisions_added", "PPO_updates_added", "PPO_optimizer_steps_added")
PROBE_HASHES = {
    "source_run_manifest": "320ef912ee08fa7d02d8169df5536a99bce59fcec4be34e9fbfaf650df8cd20f",
    "source_decisions": "2a7e06b80fc813ccdd702450b93926c12f2437ce152aea6116de680f2f68ad79",
    "source_checkpoint": "e1c1d5e200f5d471fa4b32b4ac82ae41d7a8c42e13d0415b403318a6ca0dec74",
    "source_checkpoint_manifest": "4a3ac8bf48cf5b914cc7e7c293fa08702b18f86019e97a50d1b3f78f3063ac02"}
PHASE_ROWS = {1: range(2, 342), 4: range(347, 646), 5: range(646, 842), 8: range(844, 998)}


def require(value, message):
    if not value:
        raise ValueError(message)


def _hash(value):
    from .semantic_migration import digest
    return digest(value)


def _reference(binding):
    from .semantic_migration import file_sha
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}
            and isinstance(binding["sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", binding["sha256"]), "exact file reference required")
    path = Path(binding["path"]).resolve(strict=True)
    require(str(path) == binding["path"] and file_sha(path) == binding["sha256"], "bound file changed")
    return path


def _json_reference(binding):
    return json.loads(_reference(binding).read_text(encoding="utf-8"))


def _checkpoint(path, checkpoint_sha, manifest_sha):
    from .semantic_migration import checkpoint_metadata, file_sha
    path = Path(path).resolve(strict=True)
    sidecar = path.with_name(path.stem + "_manifest.json")
    require(file_sha(path) == checkpoint_sha and file_sha(sidecar) == manifest_sha,
            "explicit source checkpoint/sidecar changed")
    source = checkpoint_metadata(path)
    route = source.get("checkpoint_output_routing", {})
    branch = Path(route.get("output_root", "")).resolve()
    require(path.parent == branch / "checkpoints/history" and branch.name == BRANCH_NAME
            and branch.parent.name == "branches" and branch.parent.parent.name == "ppo_" + EXPERIMENT
            and route.get("main_latest_pointer_promotion") is False,
            "immutable source must remain in the existing exact branch")
    return source


def _checkpoint_reference(binding):
    require(isinstance(binding, dict) and set(binding) == {
        "checkpoint", "checkpoint_sha256", "manifest", "manifest_sha256"}, "exact checkpoint reference required")
    path = Path(binding["checkpoint"]).resolve(strict=True)
    require(binding["checkpoint"] == str(path)
            and binding["manifest"] == str(path.with_name(path.stem + "_manifest.json")), "checkpoint reference paths differ")
    return _checkpoint(path, binding["checkpoint_sha256"], binding["manifest_sha256"])


def _protected(source):
    from .semantic_rr_retention_migration import _preserved
    return {key: _hash(source[key]) for key in _preserved(source)}


def _reviewed_delta(old, new, root):
    from .semantic_migration import file_sha
    variable = {"files", "source_git_commit", "runtime_content_sha256"}
    require({k: v for k, v in old.items() if k not in variable}
            == {k: v for k, v in new.items() if k not in variable},
            "AUX accounting identity cannot change task, config, distribution or physical contract")
    changed = {p for p in old["files"].keys() | new["files"].keys()
               if old["files"].get(p) != new["files"].get(p)}
    require(changed == ALLOWED and CODE + "semantic_front_retention439.py" not in old["files"]
            and all(p in new["files"] for p in changed), "exact three accounting runtime paths required")
    require(all(file_sha(root / p) == sha for p, sha in new["files"].items()), "target runtime bytes changed")
    require(all(_hash(c["files"]) == c["runtime_content_sha256"] for c in (old, new)), "runtime inventory digest differs")
    return {p: {"before": old["files"].get(p), "after": new["files"][p]} for p in sorted(changed)}


def previous_contract(current, receipt):
    old = copy.deepcopy(current)
    require(receipt.get("target_contract_sha256") == _hash(current)
            and receipt.get("target_git_commit") == current.get("source_git_commit")
            and receipt.get("target_runtime_content_sha256") == current.get("runtime_content_sha256")
            and set(receipt.get("changed_file_hashes", {})) == ALLOWED,
            "front retention identity target/delta differs")
    for path, pair in receipt["changed_file_hashes"].items():
        require(old["files"].get(path) == pair["after"], "identity target file differs")
        if pair["before"] is None:
            old["files"].pop(path)
        else:
            old["files"][path] = pair["before"]
    old.update(source_git_commit=receipt["source_git_commit"],
               runtime_content_sha256=receipt["source_runtime_content_sha256"])
    require(old["source_git_commit"] == SOURCE_HEAD and _hash(old) == receipt["source_contract_sha256"]
            and _hash(old["files"]) == old["runtime_content_sha256"], "historical72e reconstruction differs")
    return old


def build_front_retention439_identity(checkpoint, current_contract, *, reason,
        expected_source_sha256, expected_manifest_sha256, project_root=None):
    from .semantic_migration import _contract, source_num_envs
    from .semantic_rr_retention_migration import validate_rr_retention_lineage, MAPPING, _policy
    checkpoint = Path(checkpoint).resolve(strict=True)
    source = _checkpoint(checkpoint, expected_source_sha256, expected_manifest_sha256)
    route = source["checkpoint_output_routing"]
    branch = Path(route["output_root"])
    old, new = _contract(source["runtime_contract"]), _contract(current_contract)
    validate_rr_retention_lineage(source, old, branch, checkpoint_output_routing=route)
    pointer = json.loads((branch / "checkpoints/checkpoint_last_pointer.json").read_text(encoding="utf-8"))
    require(pointer == {"checkpoint": str(checkpoint), "checkpoint_sha256": expected_source_sha256,
                        "manifest": str(checkpoint.with_name(checkpoint.stem + "_manifest.json")),
                        "manifest_sha256": expected_manifest_sha256}, "select actual latest complete72e checkpoint, not an older ancestor")
    last = source.get("last_update", {})
    require(isinstance(reason, str) and reason.strip() and IDENTITY not in source and LEDGER not in source
            and old["source_git_commit"] == SOURCE_HEAD and new["source_git_commit"] != SOURCE_HEAD
            and re.fullmatch("[0-9a-f]{40}", new["source_git_commit"])
            and source_num_envs(source) == 1 and source["policy_contract"] == _policy()
            and last.get("global_policy_decisions") == source["global_policy_decisions"]
            and last.get("ppo_update") == source["ppo_updates"]
            and last.get("actor_parameter_sha256_after") == source["actor_parameter_sha256"]
            and last.get("optimizer_learning_rate") == source["optimizer_learning_rate"]
            and last.get("optimizer_steps") == 20, "latest completed learned439 update required")
    root = Path(project_root or Path(__file__).resolve().parents[3])
    delta = _reviewed_delta(old, new, root)
    factor = dict(schema=SCHEMA, revision_counter_origin={k: source[k] for k in COUNTERS},
        source_stage_requested_decisions=copy.deepcopy(source["stage_requested_decisions"]),
        source_runner_config=copy.deepcopy(source["runner_config"]), target_runner_config=copy.deepcopy(source["runner_config"]),
        source_policy_contract=copy.deepcopy(source["policy_contract"]), target_policy_contract=copy.deepcopy(source["policy_contract"]),
        preserved_metadata_sha256=_protected(source), source_effective_learning_rate=source["optimizer_learning_rate"],
        parameter_mapping=MAPPING, observation_dimension=439, action_dimension=12, num_envs=1,
        same_mdp=True, old_rollout_inherited=False, added_policy_decisions=0,
        added_ppo_updates=0, added_optimizer_steps=0, added_auxiliary_updates=0)
    record = dict(schema=SCHEMA, reason=reason.strip(), source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=expected_source_sha256, source_manifest_sha256=expected_manifest_sha256,
        source_git_commit=old["source_git_commit"], target_git_commit=new["source_git_commit"],
        source_contract_sha256=_hash(old), target_contract_sha256=_hash(new),
        source_runtime_content_sha256=old["runtime_content_sha256"], target_runtime_content_sha256=new["runtime_content_sha256"],
        changed_file_hashes=delta, **{FACTOR_KEY: factor})
    require(previous_contract(new, record) == old, "identity inversion failed")
    return record


def validate_front_retention439_identity(checkpoint, contract, plan_path, *, project_root=None):
    from .semantic_migration import file_sha
    path = Path(plan_path).resolve(strict=True)
    supplied = json.loads(path.read_text(encoding="utf-8"))
    expected = build_front_retention439_identity(checkpoint, contract, reason=supplied.get("reason", ""),
        expected_source_sha256=supplied.get("source_checkpoint_sha256"),
        expected_manifest_sha256=supplied.get("source_manifest_sha256"), project_root=project_root)
    require(supplied == expected, "identity plan changed")
    return {**expected, "plan_path": str(path), "plan_sha256": file_sha(path)}


def _empty_ledger():
    return dict(schema=LEDGER_SCHEMA, events=[], accepted_auxiliary_updates_total=0,
                attempted_auxiliary_optimizer_steps_total=0)


def _data(binding):
    data = _json_reference(binding)
    require(data.get("receipt_content_sha256") == _hash({k:v for k,v in data.items()
                if k != "receipt_content_sha256"})
            and isinstance(data.get("selected_rows_content_sha256"), str)
            and re.fullmatch("[0-9a-f]{64}", data["selected_rows_content_sha256"]),
            "selected-row receipt content digest differs")
    require(data.get("schema") == DATA_SCHEMA and data.get("observation_dimension") == 439
            and data.get("action_dimension") == 12 and data.get("phase_columns") == COLUMNS
            and data.get("decision_index_cutoff_inclusive") == 997
            and data.get("target_semantics") == "executed_student_conditional_raw_request"
            and data.get("step_applied_action_validated_as") == "finite_physical_full12_not_raw_label"
            and data.get("synthetic_or_intervention_rows_used") is False
            and data.get("teacher_deployed") is False
            and data.get("whole_failed_trajectory_used_as_success_label") is False
            and data.get("targets_are_physical_success_labels") is False
            and type(data.get("PPO_credit")) is int and data["PPO_credit"] == 0
            and type(data.get("new_AUX_credit")) is int and data["new_AUX_credit"] == 0,
            "wrong bounded439 executed dataset semantics")
    for key, sha in PROBE_HASHES.items():
        require(data.get(key, {}).get("sha256") == sha, "unreviewed probe source: " + key)
        _reference(data[key])
    train, hold = data.get("train_row_ids"), data.get("holdout_row_ids")
    require(all(isinstance(ids, list) and ids and all(type(i) is int for i in ids)
                and len(set(ids)) == len(ids) for ids in (train, hold)), "unique explicit train/holdout row IDs required")
    require(set(train).isdisjoint(hold), "train/holdout leakage")
    allowed = set().union(*(set(rows) for rows in PHASE_ROWS.values()))
    require(set(train) | set(hold) <= allowed, "sparse phase or intervention/failure tail selected")
    partitions = [[sorted(set(ids).intersection(rows)) for rows in PHASE_ROWS.values()]
                  for ids in (train, hold)]
    counts = [len(window) for windows in partitions for window in windows]
    require(all(counts) and len(set(counts)) == 1 and all(
                all(b == a + 1 for a,b in zip(window,window[1:]))
                for windows in partitions for window in windows),
            "every learned phase requires balanced contiguous train and holdout windows")
    return data


def _state(metadata):
    return {k: metadata[k] for k in ("critic_parameter_sha256", "optimizer_state_sha256",
            "normalizer_state_sha256", "optimizer_learning_rate")} | {
            "training_rng_state_sha256": _hash(metadata["training_rng_state"])}


def _invariance_evidence(report):
    evidence = report.get("invariance_evidence")
    require(isinstance(evidence, dict) and evidence.get("schema") ==
            "wlr50_clean.front_retention439.real_P10_P12_synthetic_P13_invariance.v1",
            "explicit real/synthetic invariance evidence required")
    counts = evidence.get("actual_rows_by_phase")
    require(isinstance(counts, dict) and set(counts) == {"P10", "P11", "P12", "P13"}
            and all(type(counts[p]) is int and counts[p] > 0 for p in ("P10", "P11", "P12"))
            and type(counts["P13"]) is int and counts["P13"] == 0
            and evidence.get("real_P13_validation_claimed") is False
            and type(evidence.get("synthetic_P13_rows")) is int and evidence["synthetic_P13_rows"] == 3
            and evidence.get("synthetic_rows_used_for_fit") is False
            and evidence.get("synthetic_P13_origin") ==
                "first_real_row_per_P10_P12_with_only_phase_onehot_replaced_test_fixture"
            and evidence.get("selected_phase_columns") == COLUMNS
            and evidence.get("same_future_trajectory_claimed") is False
            and "same_input_P10_P13_full_Gaussian_bitwise_unchanged" not in report,
            "real P10-P12 / synthetic-only P13 evidence scope differs")
    for key in ("real_observations_float32_sha256", "synthetic_P13_observations_float32_sha256"):
        require(isinstance(evidence.get(key), str) and re.fullmatch("[0-9a-f]{64}", evidence[key]),
                "invariance observation matrix hash required")


def _event(source, source_binding, data_binding, report_binding, index):
    _data(data_binding)
    report = _json_reference(report_binding)
    accepted, attempted = report.get("accepted_auxiliary_updates"), report.get("attempted_auxiliary_optimizer_steps")
    require(report.get("schema") == FIT_SCHEMA and type(accepted) is int and type(attempted) is int
            and 0 < accepted <= attempted <= 32 and report.get("optimized_parameters") == PARAMETERS
            and report.get("optimized_scalar_count") == 1024
            and report.get("actor_parameter_sha256_before") == source["actor_parameter_sha256"]
            and isinstance(report.get("actor_parameter_sha256_after"), str)
            and re.fullmatch("[0-9a-f]{64}", report["actor_parameter_sha256_after"])
            and report["actor_parameter_sha256_after"] != source["actor_parameter_sha256"], "AUX report/source/whitelist mismatch")
    require(all(report.get(k) is True for k in ("real_P10_P12_full_Gaussian_bitwise_unchanged",
            "synthetic_P13_full_Gaussian_bitwise_unchanged", "zero_selected_phase_columns_P10_P13_algebra_verified",
            "nonselected_actor_state_unchanged", "fresh_PPO_rollout_required", "targets_were_executed_student_raw_requests"))
            and all(report.get(k) is False for k in ("targets_were_success_or_teacher_labels",
                "teacher_deployed", "physical_success_claimed", "same_future_trajectory_claimed"))
            and all(type(report.get(k)) is int and report[k] == 0 for k in PPO_ZERO)
            and report.get("protected_state_before") == report.get("protected_state_after") == _state(source),
            "AUX changed protected learned state or borrowed PPO credit")
    _invariance_evidence(report)
    require(report.get("data_receipt") == data_binding, "fit report not bound to selected dataset")
    budget = report.get("budget", {})
    require(set(budget) == {"max_attempts", "learning_rate", "maximum_train_request_shift_full12",
                "maximum_validation_request_shift_full12", "maximum_per_state_full_gaussian_kl", "maximum_abs_log_sigma_change"}
            and type(budget.get("max_attempts")) is int and attempted <= budget["max_attempts"] <= 32
            and type(budget.get("learning_rate")) in (int,float) and math.isfinite(budget["learning_rate"])
            and budget["learning_rate"] > 0, "finite fixed-LR AUX budget required")
    stop, rejected = report.get("stop_reason"), report.get("first_rejected_proposal_restored_and_stopped")
    require((stop == "first_rejected_proposal_restored_and_stopped" and rejected is True)
            or (stop in {"finite_budget_exhausted", "selected_student_raw_targets_already_match",
                "zero_sparse_gradient_no_optimizer_step"} and rejected is False),
            "finite stop/rejection receipt differs")
    for key in ("maximum_train_request_shift_full12", "maximum_validation_request_shift_full12"):
        values = budget.get(key)
        require(isinstance(values,list) and len(values) == 12 and all(type(v) in (float,int)
                and math.isfinite(v) and v > 0 for v in values), "twelve finite REQUEST trust bounds required")
    for key in ("maximum_per_state_full_gaussian_kl", "maximum_abs_log_sigma_change"):
        value = budget.get(key)
        require(type(value) in (int,float) and math.isfinite(value) and value > 0, "Gaussian trust bound required")
    return dict(event_index=index, kind="executed_P02_P05_P06_P09_retention_sparse_phase_columns_not_PPO",
        source_checkpoint=source_binding, source_ledger_sha256=_hash(source.get(LEDGER, _empty_ledger())),
        data_receipt=data_binding, fit_report=report_binding, accepted_auxiliary_updates=accepted,
        attempted_auxiliary_optimizer_steps=attempted, optimized_parameters=PARAMETERS,
        optimized_scalar_count=1024, actor_parameter_sha256_before=source["actor_parameter_sha256"],
        actor_parameter_sha256_after=report["actor_parameter_sha256_after"],
        PPO_counters_unchanged={k: source[k] for k in COUNTERS},
        PPO_decisions_added=0, PPO_updates_added=0, PPO_optimizer_steps_added=0,
        RR_capture_success_claimed=False, teacher_deployed=False, same_future_trajectory_claimed=False)


def validate_front_retention439_lineage(metadata, contract, output_root, *, checkpoint_output_routing=None):
    if COLLECTION_KEY in metadata:
        return validate_collection439_lineage(metadata, contract, output_root,
            checkpoint_output_routing=checkpoint_output_routing)
    from .semantic_rr_retention_migration import validate_rr_retention_lineage, _policy, MAPPING
    receipt = metadata.get(IDENTITY, {})
    factor = receipt.get(FACTOR_KEY, {})
    require(receipt.get("schema") == SCHEMA and factor.get("schema") == SCHEMA, "ledger requires independent runtime identity")
    old = previous_contract(contract, receipt)
    source = _checkpoint(receipt["source_checkpoint"], receipt["source_checkpoint_sha256"], receipt["source_manifest_sha256"])
    require(source["runtime_contract"] == old and factor.get("preserved_metadata_sha256") == _protected(source)
            and factor.get("revision_counter_origin") == {k: source[k] for k in COUNTERS}
            and factor.get("source_stage_requested_decisions") == source["stage_requested_decisions"]
            and factor.get("source_runner_config") == factor.get("target_runner_config") == source["runner_config"]
            and metadata.get("runner_config") == source["runner_config"]
            and metadata.get("policy_contract") == factor.get("source_policy_contract") == factor.get("target_policy_contract") == _policy()
            and factor.get("source_effective_learning_rate") == source["optimizer_learning_rate"]
            and factor.get("parameter_mapping") == MAPPING
            and (factor.get("observation_dimension"),factor.get("action_dimension"),factor.get("num_envs")) == (439,12,1)
            and factor.get("same_mdp") is True and factor.get("old_rollout_inherited") is False
            and all(type(factor.get(k)) is int and factor[k] == 0 for k in
                    ("added_policy_decisions","added_ppo_updates","added_optimizer_steps","added_auxiliary_updates")),
            "identity source/state contract changed")
    require(all(type(metadata.get(k)) is int and metadata[k] >= source[k] for k in COUNTERS), "counter rollback")
    updates = metadata["ppo_updates"] - source["ppo_updates"]
    require(metadata["global_policy_decisions"] - source["global_policy_decisions"] == 128 * updates
            and metadata["optimizer_steps"] - source["optimizer_steps"] == 20 * updates, "PPO counter increments disagree")
    require(all(type(metadata.get("stage_requested_decisions", {}).get(k)) is int
                and metadata["stage_requested_decisions"][k] >= value
                for k,value in source["stage_requested_decisions"].items()), "used stage budget rolled back")
    for key, sha in factor["preserved_metadata_sha256"].items():
        if key.endswith(("_branch", "_migration")) or key in (
                "checkpoint_output_routing", "runner_config", "policy_contract", "seed", "normalization"):
            require(_hash(metadata.get(key)) == sha, "historical receipt or branch changed: " + key)
    historical = {k: copy.deepcopy(v) for k,v in metadata.items() if k not in (IDENTITY, LEDGER)}
    historical["runtime_contract"] = old
    validate_rr_retention_lineage(historical, old, output_root, checkpoint_output_routing=checkpoint_output_routing)
    ledger = metadata.get(LEDGER, _empty_ledger())
    require(isinstance(ledger, dict) and ledger.get("schema") == LEDGER_SCHEMA and isinstance(ledger.get("events"), list), "invalid AUX ledger")
    prefix = _empty_ledger()
    for i,event in enumerate(ledger["events"],1):
        binding = event.get("source_checkpoint", {})
        parent = _checkpoint_reference(binding)
        parent_updates = parent["ppo_updates"] - source["ppo_updates"]
        require(parent.get(IDENTITY) == receipt and parent.get("runtime_contract") == contract
                and parent.get("checkpoint_output_routing") == metadata["checkpoint_output_routing"]
                and parent.get(LEDGER, _empty_ledger()) == prefix
                and all(type(parent.get(k)) is int and metadata[k] >= parent[k] >= source[k] for k in COUNTERS)
                and parent["global_policy_decisions"] - source["global_policy_decisions"] == 128 * parent_updates
                and parent["optimizer_steps"] - source["optimizer_steps"] == 20 * parent_updates
                and all(parent.get(k) == metadata.get(k) for k in factor["preserved_metadata_sha256"]
                        if k.endswith(("_branch", "_migration")) or k in (
                            "checkpoint_output_routing", "runner_config", "policy_contract", "seed", "normalization")),
                "AUX source/append chain/counters changed")
        require(event == _event(parent, binding, event.get("data_receipt"), event.get("fit_report"), i), "AUX event receipt changed")
        prefix["events"].append(copy.deepcopy(event))
        prefix["accepted_auxiliary_updates_total"] += event["accepted_auxiliary_updates"]
        prefix["attempted_auxiliary_optimizer_steps_total"] += event["attempted_auxiliary_optimizer_steps"]
    require(ledger == prefix, "AUX totals or append-only ledger changed")
    if not ledger["events"] and updates == 0:
        require(metadata.get("actor_parameter_sha256") == source["actor_parameter_sha256"]
                and _state(metadata) == _state(source), "identity without credited update must preserve learned state")
    if ledger["events"]:
        event = ledger["events"][-1]
        if {k: metadata[k] for k in COUNTERS} == event["PPO_counters_unchanged"]:
            report = _json_reference(event["fit_report"])
            require(metadata.get("actor_parameter_sha256") == event["actor_parameter_sha256_after"]
                    and _state(metadata) == report["protected_state_after"], "same-counter AUX publication state differs")
    return receipt


def append_front_retention439_event(infos, *, source_checkpoint, expected_source_sha256,
        expected_manifest_sha256, data_receipt, fit_report):
    source_checkpoint = Path(source_checkpoint).resolve(strict=True)
    source = _checkpoint(source_checkpoint, expected_source_sha256, expected_manifest_sha256)
    route = source["checkpoint_output_routing"]
    validate_front_retention439_lineage(source, source["runtime_contract"], Path(route["output_root"]), checkpoint_output_routing=route)
    require(all(infos.get(k) == value for k,value in source.items()
                if k not in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip", "resume_source_checkpoint")), "loaded AUX source infos differ")
    binding = dict(checkpoint=str(source_checkpoint), checkpoint_sha256=expected_source_sha256,
        manifest=str(source_checkpoint.with_name(source_checkpoint.stem + "_manifest.json")), manifest_sha256=expected_manifest_sha256)
    result = copy.deepcopy(infos)
    ledger = copy.deepcopy(source.get(LEDGER, _empty_ledger()))
    event = _event(source, binding, data_receipt, fit_report, len(ledger["events"])+1)
    ledger["events"].append(event)
    ledger["accepted_auxiliary_updates_total"] += event["accepted_auxiliary_updates"]
    ledger["attempted_auxiliary_optimizer_steps_total"] += event["attempted_auxiliary_optimizer_steps"]
    result[LEDGER] = ledger
    return result


def load_front_retention439_identity(runner, checkpoint, *, contract, seed, record):
    from .semantic_migration import checkpoint_metadata
    from .semantic_training import load_semantic_checkpoint, _runner_policy_contract
    from .semantic_rr_retention_migration import _verify_identity
    require(not any(k.endswith("_factor") and k != FACTOR_KEY and v is not None for k,v in record.items()), "mixed identity factors")
    verified = validate_front_retention439_identity(checkpoint, contract, record["plan_path"])
    source = checkpoint_metadata(Path(checkpoint))
    require(verified == dict(record) and seed == source["seed"]
            and _runner_policy_contract(runner) == source["policy_contract"], "identity seed/kernel/source changed")
    infos = load_semantic_checkpoint(runner, Path(checkpoint), contract=source["runtime_contract"], seed=seed)
    _verify_identity(runner, infos, verified[FACTOR_KEY])
    return {**infos, "runtime_contract": copy.deepcopy(contract), IDENTITY: verified,
            "old_rollout_inherited": False, "physical_env_state_saved": False,
            "resume_physics": "legal_reset_not_bitwise_continuation"}


def publish_front_retention439_checkpoint(checkpoint, contract, plan_path, output_checkpoint):
    from .semantic_migration import checkpoint_metadata, file_sha
    from .semantic_training import construct_semantic_runner, load_semantic_checkpoint, save_semantic_checkpoint
    from .semantic_rr_capture_migration import _shape_env
    from .semantic_rr_retention_migration import _verify_identity
    source = checkpoint_metadata(Path(checkpoint))
    record = validate_front_retention439_identity(checkpoint, contract, plan_path)
    route = source["checkpoint_output_routing"]
    destination = Path(output_checkpoint).resolve()
    require(destination.parent == Path(route["output_root"]) / "checkpoints/history"
            and not destination.exists() and not destination.with_name(destination.stem + "_manifest.json").exists(), "unique same-branch publication required")
    def make():
        device = source["runner_config"]["device"]
        return construct_semantic_runner(_shape_env(439,device),seed=source["seed"],device=device,
            policy_version=REAR_OWNER_POLICY,observation_layout=REAR_OWNER_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner = make()
    infos = load_front_retention439_identity(runner,checkpoint,contract=contract,seed=source["seed"],record=record)
    path, sidecar = save_semantic_checkpoint(runner,destination,infos)
    fresh = make()
    loaded = load_semantic_checkpoint(fresh,path,contract=contract,seed=source["seed"])
    _verify_identity(fresh,loaded,record[FACTOR_KEY])
    validate_front_retention439_lineage(loaded,contract,Path(route["output_root"]),checkpoint_output_routing=route)
    return dict(checkpoint=str(path),checkpoint_sha256=file_sha(path),manifest=str(sidecar),manifest_sha256=file_sha(sidecar),
        save_load_round_trip=True,latest_pointer_published=False,**{k:loaded[k] for k in COUNTERS},
        added_policy_decisions=0,added_ppo_updates=0,added_optimizer_steps=0,added_auxiliary_updates=0)


# One explicit collection boundary; historical identity/AUX receipts above stay intact.
COLLECTION_KEY = "collection_horizon439"
COLLECTION_SCHEMA = "wlr50_clean.collection_horizon439.v1"
COLLECTION_SOURCE_HEAD = "65a9255be6d9fd4e19590a48a3a650ae606b04f7"
COLLECTION_FILES = frozenset(CODE + name for name in (
    "semantic_return_profile.py", "semantic_training.py", "semantic_front_retention439.py",
    "semantic_cli.py", "semantic_video_cli.py", "semantic_policy_distribution.py"))


def collection_runner_options(metadata, *, experiment_id):
    """Select the explicit new constructor only; full lineage still loads normally."""
    from .semantic_return_profile import COLLECTION_PROFILE_KEY, COLLECTION_512, runner_collection_length
    config = metadata.get("runner_config", {})
    length = runner_collection_length(config)
    receipt = metadata.get(COLLECTION_KEY)
    if receipt is None:
        require(length == 128 and COLLECTION_PROFILE_KEY not in config,
                "unreceipted collection change")
        return {}
    require(experiment_id == EXPERIMENT and receipt.get("schema") == COLLECTION_SCHEMA
            and length == 512 and config.get(COLLECTION_PROFILE_KEY) == COLLECTION_512
            and receipt.get("target_runner_config") == config,
            "collection512 requires its exact experiment/receipt/config")
    return {"collection_profile": COLLECTION_512}


def _collection_counter_delta(metadata, source):
    """New segment uses real512/20 increments; never reinterpret historical128."""
    require(all(type(metadata.get(k)) is int and type(source.get(k)) is int
                and metadata[k] >= source[k] for k in COUNTERS), "collection counter rollback")
    updates = metadata["ppo_updates"] - source["ppo_updates"]
    require(metadata["global_policy_decisions"] - source["global_policy_decisions"] == 512 * updates
            and metadata["optimizer_steps"] - source["optimizer_steps"] == 20 * updates,
            "collection512 segment requires512 decisions and20 Adam steps per update")
    return updates


def _collection_target_config(source):
    from .semantic_return_profile import COLLECTION_PROFILE_KEY, COLLECTION_512, runner_return_profile
    config = copy.deepcopy(source["runner_config"])
    require(COLLECTION_PROFILE_KEY not in config
            and runner_return_profile(config, semantic_version="v3")["rollout_length"] == 128
            and config["algorithm"]["num_learning_epochs"] == 5
            and config["algorithm"]["num_mini_batches"] == 4,
            "collection boundary requires unchanged historical128/5epochs/4minibatches")
    config.update(num_steps_per_env=512)
    config[COLLECTION_PROFILE_KEY] = COLLECTION_512
    runner_return_profile(config, semantic_version="v3")
    return config


def _collection_frozen_metadata(source):
    keys = {IDENTITY, LEDGER, "checkpoint_output_routing", "policy_contract", "seed", "normalization",
            "new_mdp_origin_global_policy_decisions", "source_stage_requested_decisions"}
    keys |= {key for key in source if key != "resume_migration" and key.endswith(("_branch", "_migration"))}
    return {key: _hash(source[key]) for key in sorted(keys) if key in source}


def _collection_delta(old, new, *, project_root=None):
    from .semantic_migration import file_sha
    variable = {"files", "source_git_commit", "runtime_content_sha256"}
    require({k:v for k,v in old.items() if k not in variable}
            == {k:v for k,v in new.items() if k not in variable},
            "collection boundary cannot change reward, control, observation, distribution or topology")
    changed = {p for p in old["files"].keys() | new["files"].keys()
               if old["files"].get(p) != new["files"].get(p)}
    require(changed == COLLECTION_FILES and all(p in old["files"] and p in new["files"] for p in changed),
            "collection boundary requires exactly six reviewed existing runtime paths")
    require(old["source_git_commit"] == COLLECTION_SOURCE_HEAD
            and new["source_git_commit"] != COLLECTION_SOURCE_HEAD
            and re.fullmatch("[0-9a-f]{40}", new["source_git_commit"])
            and all(_hash(c["files"]) == c["runtime_content_sha256"] for c in (old,new)),
            "collection runtime source/digest differs")
    if project_root is not None:
        require(all(file_sha(Path(project_root) / p) == sha for p,sha in new["files"].items()),
                "collection target runtime bytes differ")
    return {p: {"before": old["files"][p], "after": new["files"][p]} for p in sorted(changed)}


def _collection_record(source, binding, contract, reason):
    from .semantic_migration import source_num_envs
    from .semantic_rr_retention_migration import _policy, MAPPING
    old = source["runtime_contract"]
    last = source.get("last_update", {})
    require(COLLECTION_KEY not in source and source_num_envs(source) == 1
            and source["policy_contract"] == _policy() and old["experiment_id"] == EXPERIMENT
            and isinstance(reason,str) and reason.strip()
            and last.get("global_policy_decisions") == source["global_policy_decisions"]
            and last.get("ppo_update") == source["ppo_updates"]
            and last.get("actor_parameter_sha256_after") == source["actor_parameter_sha256"]
            and last.get("optimizer_learning_rate") == source["optimizer_learning_rate"]
            and last.get("optimizer_steps") == 20,
            "collection source must be an actual complete learned65a update")
    return dict(schema=COLLECTION_SCHEMA, reason=reason.strip(), source_checkpoint=copy.deepcopy(binding),
        source_contract_sha256=_hash(old), target_contract_sha256=_hash(contract),
        changed_file_hashes=_collection_delta(old,contract),
        source_runner_config=copy.deepcopy(source["runner_config"]),
        target_runner_config=_collection_target_config(source),
        counter_origin={k:source[k] for k in COUNTERS},
        source_stage_requested_decisions=copy.deepcopy(source["stage_requested_decisions"]),
        frozen_historical_metadata_sha256=_collection_frozen_metadata(source),
        collection_steps_before=128, collection_steps_after=512, optimizer_steps_per_update=20,
        source_effective_learning_rate=source["optimizer_learning_rate"], parameter_mapping=MAPPING,
        same_mdp=True, return_estimator_changed=False, reward_changed=False,
        actor_input_changed=False, HISTORY_changed=False, raw_sample_likelihood_semantics_changed=False,
        old_rollout_inherited=False, fresh_rollout_required=True,
        added_policy_decisions=0, added_ppo_updates=0, added_optimizer_steps=0, added_auxiliary_updates=0)


def validate_collection439_lineage(metadata, contract, output_root, *, checkpoint_output_routing=None):
    receipt = metadata.get(COLLECTION_KEY, {})
    require(receipt.get("schema") == COLLECTION_SCHEMA, "collection receipt required")
    source = _checkpoint_reference(receipt.get("source_checkpoint", {}))
    require(COLLECTION_KEY not in source, "first512 boundary cannot rewrite or recursively relabel a segment")
    route = source["checkpoint_output_routing"]
    # Validate the frozen original128 source through every historical validator.
    validate_front_retention439_lineage(source, source["runtime_contract"], Path(route["output_root"]),
        checkpoint_output_routing=route)
    expected = _collection_record(source,receipt["source_checkpoint"],contract,receipt.get("reason",""))
    require(receipt == expected and metadata.get("runtime_contract") == contract
            and metadata.get("runner_config") == expected["target_runner_config"]
            and checkpoint_output_routing == route and metadata.get("checkpoint_output_routing") == route
            and Path(output_root).resolve() == Path(route["output_root"]).resolve(),
            "collection receipt/config/route differs")
    require(all(_hash(metadata.get(k)) == sha for k,sha in expected["frozen_historical_metadata_sha256"].items()),
            "historical receipts, AUX ledger, seed or policy changed")
    updates = _collection_counter_delta(metadata,source)
    require(all(type(metadata.get("stage_requested_decisions",{}).get(k)) is int
                and metadata["stage_requested_decisions"][k] >= value
                for k,value in source["stage_requested_decisions"].items()), "collection stage budget rolled back")
    # Old branch-count objects are derived totals, not immutable receipts.
    for key,value in metadata.items():
        if key.endswith("_branch") and isinstance(value,dict) and set(COUNTERS) <= set(value.get("counter_origin",{})):
            counts = key + "_counts"
            if counts in source:
                require(metadata.get(counts) == {k:metadata[k]-value["counter_origin"][k] for k in COUNTERS},
                        "actual historical branch totals differ: " + counts)
    if updates == 0:
        require(metadata.get("actor_parameter_sha256") == source["actor_parameter_sha256"]
                and _state(metadata) == _state(source)
                and metadata.get("last_update") == source["last_update"]
                and metadata.get("stage_requested_decisions") == source["stage_requested_decisions"],
                "zero-update collection publication changed learned state or used budget")
    else:
        last = metadata.get("last_update", {})
        require(last.get("global_policy_decisions") == metadata["global_policy_decisions"]
                and last.get("ppo_update") == metadata["ppo_updates"]
                and last.get("optimizer_steps") == 20
                and last.get("actor_parameter_sha256_after") == metadata.get("actor_parameter_sha256")
                and last.get("optimizer_learning_rate") == metadata.get("optimizer_learning_rate"),
                "collection checkpoint lacks its actual complete update")
    return receipt


def publish_collection512_checkpoint(checkpoint, contract, output_checkpoint, *, reason,
        expected_source_sha256, expected_manifest_sha256, project_root=None):
    """Explicit zero-update boundary, then reuse the ordinary train/eval loaders.

    Call only after the sole physical run has sealed. No teacher, fit, update,
    pointer promotion, hidden source selection, or partial-rollout reuse occurs.
    """
    from .semantic_migration import file_sha
    from .semantic_training import (construct_semantic_runner, load_semantic_checkpoint,
        save_semantic_checkpoint, parameter_hash, state_hash, _normalizers)
    from .semantic_rr_capture_migration import _shape_env
    from .semantic_return_profile import COLLECTION_512
    from .rl_library_wrapper import restore_training_rng_state, optimizer_learning_rate, capture_training_rng_state
    checkpoint = Path(checkpoint).resolve(strict=True)
    source = _checkpoint(checkpoint,expected_source_sha256,expected_manifest_sha256)
    route = source["checkpoint_output_routing"]
    branch = Path(route["output_root"])
    validate_front_retention439_lineage(source,source["runtime_contract"],branch,checkpoint_output_routing=route)
    binding = dict(checkpoint=str(checkpoint),checkpoint_sha256=expected_source_sha256,
        manifest=str(checkpoint.with_name(checkpoint.stem+"_manifest.json")),manifest_sha256=expected_manifest_sha256)
    require(json.loads((branch/"checkpoints/checkpoint_last_pointer.json").read_text(encoding="utf-8")) == binding,
            "select the actual latest complete65a source after the active block seals")
    _collection_delta(source["runtime_contract"],contract,
        project_root=Path(project_root or Path(__file__).resolve().parents[3]))
    receipt = _collection_record(source,binding,contract,reason)
    destination = Path(output_checkpoint).resolve()
    require(destination.parent == branch/"checkpoints/history" and not destination.exists()
            and not destination.with_name(destination.stem+"_manifest.json").exists(),
            "unique same-branch collection publication required")
    def make(marker):
        device = source["runner_config"]["device"]
        return construct_semantic_runner(_shape_env(439,device),seed=source["seed"],device=device,
            policy_version=REAR_OWNER_POLICY,observation_layout=REAR_OWNER_OBSERVATION_LAYOUT,
            initialize_actor=False,collection_profile=marker)[0]
    historical = make(None)  # Explicit historical128, no new collection marker.
    require(historical._semantic_runner_config == source["runner_config"], "source128 factory reconstruction differs")
    infos = load_semantic_checkpoint(historical,checkpoint,contract=source["runtime_contract"],seed=source["seed"])
    runner = make(COLLECTION_512)
    require(runner._semantic_runner_config == receipt["target_runner_config"], "target512 factory reconstruction differs")
    runner.alg.actor.load_state_dict(historical.alg.actor.state_dict(),strict=True)
    runner.alg.critic.load_state_dict(historical.alg.critic.state_dict(),strict=True)
    runner.alg.optimizer.load_state_dict(copy.deepcopy(historical.alg.optimizer.state_dict()))
    runner.alg.learning_rate = optimizer_learning_rate(historical)
    runner.current_learning_iteration = historical.current_learning_iteration
    require(runner.alg.storage.step == 0 and runner.alg.transition.actions is None,
            "collection512 requires newly allocated empty rollout and transition")
    restore_training_rng_state(source["training_rng_state"],expected_seed=source["seed"])
    def verify_identity(target):
        require(parameter_hash(target.alg.actor) == source["actor_parameter_sha256"]
                and parameter_hash(target.alg.critic) == source["critic_parameter_sha256"]
                and state_hash(target.alg.optimizer.state_dict()) == source["optimizer_state_sha256"]
                and state_hash(_normalizers(target)) == source["normalizer_state_sha256"]
                and optimizer_learning_rate(target) == source["optimizer_learning_rate"]
                and capture_training_rng_state(seed=source["seed"]) == source["training_rng_state"],
                "collection publication changed full learned/Adam/normalizer/LR/RNG state")
    verify_identity(runner)
    infos = {**infos,"runtime_contract":copy.deepcopy(contract),COLLECTION_KEY:receipt,
             "old_rollout_inherited":False,"physical_env_state_saved":False,
             "resume_physics":"legal_reset_not_bitwise_continuation"}
    path,sidecar = save_semantic_checkpoint(runner,destination,infos)
    fresh = make(COLLECTION_512)
    loaded = load_semantic_checkpoint(fresh,path,contract=contract,seed=source["seed"])
    verify_identity(fresh)
    validate_collection439_lineage(loaded,contract,branch,checkpoint_output_routing=route)
    return dict(checkpoint=str(path),checkpoint_sha256=file_sha(path),manifest=str(sidecar),
        manifest_sha256=file_sha(sidecar),save_load_round_trip=True,latest_pointer_published=False,
        collection_steps_per_env=512,**{k:loaded[k] for k in COUNTERS},
        added_policy_decisions=0,added_ppo_updates=0,added_optimizer_steps=0,added_auxiliary_updates=0)

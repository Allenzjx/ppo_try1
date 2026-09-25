"""Build the immutable CP225280 front conditional-Gaussian replay set.

Stdlib only.  This reads one sealed diagnostic, never imports a model, and
does not create PPO/AUX samples or checkpoints.  The output is an offline
regularizer reference, not a closed-loop success proof.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (HERE.parent / "staged_front_retention439" /
                 "data_contract.py")
DATASET_SCHEMA = "wlr50_clean.cp225280_front_replay.v1"
RECEIPT_SCHEMA = "wlr50_clean.cp225280_front_replay_receipt.v1"
CONFIG_SCHEMA = "wlr50_clean.front_replay_regularizer.v1"
FRONT_PHASES = ("P01", "P02", "P03", "P04", "P05", "P06")
ABUNDANT_PHASES = ("P02", "P05", "P06")
SCARCE_PHASES = ("P01", "P03", "P04")
MAX_ABUNDANT_PER_PHASE = 64
MAX_TOTAL_ROWS = 256
OBSERVATION_DIMENSION = 439
ACTION_DIMENSION = 12
SOURCE_HEAD = "b0ff99729b1df84c4648f9d53b1d372f1e9152af"
SOURCE_POLICY = "rear_owner_recovery_history_v1"
SOURCE_LAYOUT = "role422_rear_owner_recovery_v1"
F6D_PUBLICATION = (HERE.parent /
    "rear_owner_CP225280_gf6d1d2df8d87_publication.json")
F6D_PUBLICATION_SHA256 = (
    "161fd1c34400d5880f1ceea0d44bef0bfe8dac257dfbf40db172d8a7c5661267")
F6D_HEAD = "f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b"
F6D_CHECKPOINT_SHA256 = (
    "fbe28e3718a5796afc2271e7b10aa04017369593b5c62e0bde1973a20f68032f")
F6D_MANIFEST_SHA256 = (
    "f501b7a735c0aaa42be00114e09f80d46f7f009d7717aa416fc30ad3cc837d6d")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_cp225280_front_data_contract", CONTRACT_PATH)
    require(spec is not None and spec.loader is not None,
            "cannot load reviewed source contract")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_vector(value: Any, length: int, name: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length,
            f"{name} must have length {length}")
    result = [float(item) for item in value]
    require(all(math.isfinite(item) for item in result), name + " must be finite")
    return result


def phase_column(phase: str) -> int:
    require(phase in FRONT_PHASES, "row is not a front phase")
    return int(phase[1:]) - 1


def compact_front_row(row: dict[str, Any], expected_index: int) -> dict[str, Any]:
    require(row.get("decision_index") == expected_index,
            "decision indices must be contiguous and zero based")
    phase = row.get("phase")
    require(phase in FRONT_PHASES, "compact row must be P01-P06")
    observation = finite_vector(row.get("policy_observation_vector"),
                                OBSERVATION_DIMENSION, "observation")
    require(all(abs(value) <= 20.0 for value in observation),
            "observation violates its fixed clip")
    one_hot = observation[:13]
    require(all(value in (0.0, 1.0) for value in one_hot) and
            sum(one_hot) == 1.0 and one_hot[phase_column(phase)] == 1.0,
            "phase and observation one-hot differ")
    audit = row.get("original_student_request_audit")
    require(isinstance(audit, dict), "missing original request audit")
    mean = finite_vector(audit.get("conditional_mean_full12"),
                         ACTION_DIMENSION, "conditional mean")
    sigma = finite_vector(audit.get("effective_sigma_full12"),
                          ACTION_DIMENSION, "conditional sigma")
    log_std = finite_vector(audit.get("effective_log_std_full12"),
                            ACTION_DIMENSION, "effective log std")
    require(all(value > 0.0 for value in sigma),
            "conditional sigma must be positive")
    require(all(math.isclose(value, math.exp(log_value), rel_tol=2.0e-6,
                             abs_tol=2.0e-8)
                for value, log_value in zip(sigma, log_std)),
            "saved sigma and log std differ")
    raw = finite_vector(row.get("original_student_raw_full12"),
                        ACTION_DIMENSION, "student raw")
    selected = finite_vector(audit.get("selected_raw_full12"),
                             ACTION_DIMENSION, "selected raw")
    applied = finite_vector(row.get("applied_raw_full12"),
                            ACTION_DIMENSION, "applied raw")
    step = row.get("step_info") or {}
    step_raw = finite_vector(step.get("raw_policy_action_full12"),
                             ACTION_DIMENSION, "step raw")
    require(raw == mean == selected == applied == step_raw,
            "deterministic pre-intervention mean/raw receipts differ")
    require(audit.get("mode") == "deterministic_conditional_mean" and
            audit.get("policy_version") == SOURCE_POLICY and
            math.isclose(float(audit.get("rho")), 0.9, rel_tol=0.0,
                         abs_tol=1.0e-12) and
            math.isclose(float(audit.get("exploration_std_temperature")), 0.25,
                         rel_tol=0.0, abs_tol=1.0e-12),
            "conditional Gaussian/HISTORY identity differs")
    require(row.get("overridden_indices") == [] and
            row.get("probe_active") is False and
            row.get("teacher_actions_deployed") is False and
            row.get("diagnostic_rows_entered_in_on_policy_storage") is False and
            row.get("formal_deterministic_policy_result") is False and
            all(row.get(key) == 0 for key in (
                "new_PPO_samples", "new_PPO_updates", "new_optimizer_steps",
                "new_AUX_accepted_updates", "new_AUX_attempted_steps")),
            "selected row has intervention, teacher, or learning credit")
    return {"decision_index": expected_index, "phase": phase,
            "observation": observation, "reference_mean": mean,
            "reference_sigma": sigma}


def uniform_positions(count: int, quota: int) -> list[int]:
    """Integer-only inclusive uniform selection over one complete phase run."""
    require(type(count) is int and type(quota) is int and
            1 <= quota <= count, "invalid uniform selection bounds")
    if quota == 1:
        return [0]
    denominator = quota - 1
    result = [
        (index * (count - 1) + denominator // 2) // denominator
        for index in range(quota)
    ]
    require(len(result) == len(set(result)) and result[0] == 0 and
            result[-1] == count - 1, "uniform selection lost range coverage")
    return result


def select_rows(rows: Iterable[dict[str, Any]], *, cutoff_inclusive: int,
                expected_total_rows: int) -> dict[str, Any]:
    by_phase: dict[str, list[dict[str, Any]]] = {
        phase: [] for phase in FRONT_PHASES}
    all_phase_counts: dict[str, int] = {}
    total = 0
    for total, row in enumerate(rows, 1):
        index = total - 1
        require(row.get("decision_index") == index,
                "decision indices must be contiguous and zero based")
        phase = row.get("phase")
        all_phase_counts[phase] = all_phase_counts.get(phase, 0) + 1
        if index <= cutoff_inclusive and phase in FRONT_PHASES:
            by_phase[phase].append(compact_front_row(row, index))
        elif index <= cutoff_inclusive:
            # Every row before the intervention boundary is still checked for
            # absence of override/credit by the reviewed data contract below.
            require(row.get("overridden_indices") == [] and
                    row.get("probe_active") is False and
                    all(row.get(key) == 0 for key in (
                        "new_PPO_samples", "new_PPO_updates",
                        "new_optimizer_steps", "new_AUX_accepted_updates",
                        "new_AUX_attempted_steps")),
                    "pre-intervention non-front row has intervention or credit")
    require(total == expected_total_rows,
            "sealed decision row count differs")
    require(all(by_phase[phase] for phase in FRONT_PHASES),
            "sealed prefix does not cover every P01-P06 phase")

    selected: list[dict[str, Any]] = []
    phase_selection: dict[str, Any] = {}
    for phase in FRONT_PHASES:
        source = by_phase[phase]
        quota = min(MAX_ABUNDANT_PER_PHASE, len(source)) \
            if phase in ABUNDANT_PHASES else len(source)
        positions = uniform_positions(len(source), quota)
        phase_rows = [source[position] for position in positions]
        fit_ids: list[int] = []
        heldout_ids: list[int] = []
        for rank, item in enumerate(phase_rows):
            copy = dict(item)
            copy["partition"] = "training" if rank % 2 == 0 else "heldout"
            selected.append(copy)
            (fit_ids if rank % 2 == 0 else heldout_ids).append(
                copy["decision_index"])
        phase_selection[phase] = {
            "source_count": len(source),
            "source_first_decision_index": source[0]["decision_index"],
            "source_last_decision_index": source[-1]["decision_index"],
            "selected_count": quota,
            "selected_first_decision_index": phase_rows[0]["decision_index"],
            "selected_last_decision_index": phase_rows[-1]["decision_index"],
            "uniform_inclusive_full_phase_range": True,
            "training_row_ids": fit_ids, "heldout_row_ids": heldout_ids,
            "phase_specific_holdout_available": bool(heldout_ids),
        }
    require(len(selected) <= MAX_TOTAL_ROWS,
            "front replay exceeds its finite row budget")
    selected.sort(key=lambda item: item["decision_index"])
    training = [{key: value for key, value in row.items() if key != "partition"}
                for row in selected if row["partition"] == "training"]
    heldout = [{key: value for key, value in row.items() if key != "partition"}
               for row in selected if row["partition"] == "heldout"]
    train_ids = {row["decision_index"] for row in training}
    heldout_ids = {row["decision_index"] for row in heldout}
    require(train_ids.isdisjoint(heldout_ids) and
            len(training) + len(heldout) == len(selected),
            "training/heldout partitions overlap or lose rows")
    return {"training_rows": training, "heldout_rows": heldout,
            "phase_selection": phase_selection,
            "all_phase_counts": all_phase_counts}


def source_binding(contract: Any) -> dict[str, Any]:
    reviewed = contract.verify_reviewed_source_files()
    manifest_path = Path(reviewed["source_run_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = manifest.get("checkpoint_policy_contract") or {}
    proof = manifest.get("official_checkpoint_load_proof") or {}
    require(manifest.get("runtime_contract", {}).get("source_git_commit") ==
            SOURCE_HEAD and policy.get("version") == SOURCE_POLICY and
            policy.get("observation_dimension") == OBSERVATION_DIMENSION and
            policy.get("observation_layout") == SOURCE_LAYOUT and
            proof.get("policy_sampling_mode") ==
                "deterministic_conditional_mean",
            "probe manifest policy/HISTORY identity differs")
    evaluation = (manifest.get("physical_summary", {})
                  .get("physical_task_evaluation", {}))
    transfer = evaluation.get("transfer_roles", {})
    require(all((transfer.get(leg, {}).get("crossing_and_placement_evidence") or {})
                .get("placed") is True for leg in ("FR", "FL")),
            "source episode does not prove successful FR/FL placement")

    publication_path = F6D_PUBLICATION.resolve(strict=True)
    require(file_sha256(publication_path) == F6D_PUBLICATION_SHA256,
            "reviewed f6d publication bytes changed")
    publication = json.loads(publication_path.read_text(encoding="utf-8"))
    require(publication.get("checkpoint_sha256") == F6D_CHECKPOINT_SHA256 and
            publication.get("manifest_sha256") == F6D_MANIFEST_SHA256 and
            publication.get("global_policy_decisions") == 225280 and
            publication.get("ppo_updates") == 1725 and
            publication.get("optimizer_steps") == 34500 and
            publication.get("added_policy_decisions") == 0 and
            publication.get("added_ppo_updates") == 0 and
            publication.get("added_optimizer_steps") == 0 and
            publication.get("added_auxiliary_updates") == 0 and
            publication.get("save_load_round_trip") is True,
            "reviewed f6d zero-credit publication differs")
    return {
        **reviewed,
        "probe_runtime_head": SOURCE_HEAD,
        "policy_version": SOURCE_POLICY,
        "observation_layout": SOURCE_LAYOUT,
        "conditional_distribution": {
            "mean_field": "conditional_mean_full12",
            "sigma_field": "effective_sigma_full12",
            "log_std_field_validated": "effective_log_std_full12",
            "rho": 0.9, "exploration_std_temperature": 0.25,
            "deterministic_raw_equals_conditional_mean": True,
            "full_439_observation_includes_saved_HISTORY_context": True,
        },
        "front_progression": {
            "source_episode_reached_P06": True,
            "FR_crossed_and_placed": True, "FL_crossed_and_placed": True,
            "selected_rows_are_not_physical_success_labels": True,
        },
        "reviewed_f6d_tensor_identical_migration_target": {
            "runtime_head": F6D_HEAD,
            "publication": {"path": str(publication_path),
                            "sha256": F6D_PUBLICATION_SHA256},
            "checkpoint_sha256": F6D_CHECKPOINT_SHA256,
            "manifest_sha256": F6D_MANIFEST_SHA256,
            "counters": {"global_policy_decisions": 225280,
                         "ppo_updates": 1725, "optimizer_steps": 34500},
            "migration_added_learning": False,
            "tensor_identity_recomputed_by_this_stdlib_builder": False,
        },
    }


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL line {line_number}") from error
            require(isinstance(value, dict), "decision row is not an object")
            yield value


def write_new_json(path: Path, value: Any) -> None:
    require(not path.exists(), "immutable output already exists: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def build(output_directory: Path) -> dict[str, Any]:
    contract = load_contract()
    binding = source_binding(contract)
    decision_path = Path(binding["source_decisions"]["path"]).resolve(strict=True)
    selected = select_rows(
        iter_jsonl(decision_path),
        cutoff_inclusive=contract.CUTOFF_INCLUSIVE,
        expected_total_rows=contract.SOURCE_DECISION_ROWS)
    dataset = {"schema": DATASET_SCHEMA, "source_binding": binding,
               "training_rows": selected["training_rows"],
               "heldout_rows": selected["heldout_rows"]}
    dataset_path = output_directory / "CP225280_front_replay_dataset.json"
    write_new_json(dataset_path, dataset)
    dataset_sha = file_sha256(dataset_path)
    config = {"schema": CONFIG_SCHEMA,
              "dataset_path": str(dataset_path.resolve()),
              "dataset_sha256": dataset_sha,
              "coefficient": 1.0, "minibatch_size": 32}
    config_path = output_directory / "front_replay_regularizer_config.json"
    write_new_json(config_path, config)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "dataset": {"path": str(dataset_path.resolve()),
                    "sha256": dataset_sha},
        "regularizer_config": {"path": str(config_path.resolve()),
            "sha256": file_sha256(config_path)},
        "source_binding": binding,
        "selection": {
            "eligible_phases": list(FRONT_PHASES),
            "abundant_phase_quota": MAX_ABUNDANT_PER_PHASE,
            "scarce_phase_policy": "all_rows",
            "maximum_total_rows": MAX_TOTAL_ROWS,
            "method": "integer_uniform_inclusive_then_alternating_training_heldout",
            "phase_selection": selected["phase_selection"],
            "source_phase_counts_all_1368_rows": selected["all_phase_counts"],
            "training_count": len(selected["training_rows"]),
            "heldout_count": len(selected["heldout_rows"]),
            "total_count": (len(selected["training_rows"]) +
                            len(selected["heldout_rows"])),
            "partitions_disjoint": True,
            "full_phase_time_range_covered": True,
        },
        "semantics": {
            "target": "saved_old_policy_conditional_Gaussian_mean_and_sigma",
            "nominal_angles_or_physical_targets_used": False,
            "rear_phase_or_rear_success_labels_used": False,
            "post_intervention_rows_used": False,
            "failed_trajectory_tail_used": False,
            "replay_enters_on_policy_rollout_storage": False,
            "offline_replay_is_closed_loop_proof": False,
            "new_PPO_or_AUX_credit": 0,
        },
    }
    receipt["receipt_content_sha256"] = digest_bytes(canonical_bytes(receipt))
    receipt_path = output_directory / "CP225280_front_replay_receipt.json"
    write_new_json(receipt_path, receipt)
    return {"dataset": str(dataset_path), "dataset_sha256": dataset_sha,
            "config": str(config_path), "receipt": str(receipt_path),
            "training_rows": len(selected["training_rows"]),
            "heldout_rows": len(selected["heldout_rows"])}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--output-directory", type=Path, default=HERE)
    return result


if __name__ == "__main__":
    print(json.dumps(build(parser().parse_args().output_directory.resolve()),
                     indent=2))

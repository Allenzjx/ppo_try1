"""Stdlib-only admission for the dormant owner439 front-retention candidate.

This module never imports Torch and never writes a dataset.  It validates the
sealed diagnostic rows in memory and returns compact row selections plus a
receipt.  Numeric loading/fit remains a separate, explicitly authorized step.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "wlr50_clean.student_probe_v2_front_retention439_rows.v1"
PROBE_RUN = (ROOT / "runs" / "ppo_rr_rl_timing_policy_learning_v1" /
             "direction_probe" / "20260924_owner439_student_entry_v2")
RUN_MANIFEST_SHA256 = "320ef912ee08fa7d02d8169df5536a99bce59fcec4be34e9fbfaf650df8cd20f"
DECISIONS_SHA256 = "2a7e06b80fc813ccdd702450b93926c12f2437ce152aea6116de680f2f68ad79"
SOURCE_CHECKPOINT_SHA256 = "e1c1d5e200f5d471fa4b32b4ac82ae41d7a8c42e13d0415b403318a6ca0dec74"
SOURCE_MANIFEST_SHA256 = "4a3ac8bf48cf5b914cc7e7c293fa08702b18f86019e97a50d1b3f78f3063ac02"
SOURCE_DECISION_ROWS = 1368
CUTOFF_INCLUSIVE = 997
OBSERVATION_DIMENSION = 439
ACTION_DIMENSION = 12
TARGET_PHASES = ("P02", "P05", "P06", "P09")
PHASE_COLUMNS = (1, 4, 5, 8)
SPARSE_EXCLUDED_PHASES = ("P01", "P03", "P04", "P07", "P08")
INVARIANCE_PHASES = ("P10", "P11", "P12", "P13")
REAL_INVARIANCE_PHASES = ("P10", "P11", "P12")
TARGET_SEMANTICS = "executed_student_conditional_raw_request"
ZERO_KEYS = ("new_PPO_samples", "new_PPO_updates", "new_optimizer_steps",
             "new_AUX_accepted_updates", "new_AUX_attempted_steps")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _binding(path: Path, sha256: str) -> dict[str, str]:
    require(re.fullmatch(r"[0-9a-f]{64}", sha256) is not None,
            "binding requires lowercase SHA256")
    return {"path": str(path.resolve()), "sha256": sha256}


def reviewed_source_binding() -> dict[str, Any]:
    """Return immutable reviewed identities without reading their large files."""
    manifest = PROBE_RUN / "run_manifest.json"
    decisions = PROBE_RUN / "diagnostic_decisions.jsonl"
    # These paths are recorded by the sealed run; verification reads the
    # manifest and checks that it still names this exact checkpoint pair.
    checkpoint = (ROOT / "outputs" / "ppo_rr_rl_timing_policy_learning_v1" /
        "branches" / "ancestor220544_recapture_v2" / "checkpoints" / "history" /
        "checkpoint_rear_owner_CP225280_gb0ff99729b1d.pt")
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    return {"source_run_manifest": _binding(manifest, RUN_MANIFEST_SHA256),
        "source_decisions": _binding(decisions, DECISIONS_SHA256),
        "source_checkpoint": _binding(checkpoint, SOURCE_CHECKPOINT_SHA256),
        "source_checkpoint_manifest": _binding(sidecar, SOURCE_MANIFEST_SHA256),
        "source_decision_rows": SOURCE_DECISION_ROWS}


def validate_source_binding(value: dict[str, Any]) -> None:
    expected = reviewed_source_binding()
    require(value == expected, "probe-v2 source/checkpoint binding differs")


def verify_reviewed_source_files() -> dict[str, Any]:
    """Explicit future execution check; scans the large decisions file once."""
    binding = reviewed_source_binding()
    for key in ("source_run_manifest", "source_decisions", "source_checkpoint",
                "source_checkpoint_manifest"):
        item = binding[key]
        path = Path(item["path"]).resolve(strict=True)
        require(file_sha256(path) == item["sha256"], key + " bytes changed")
    manifest = json.loads(Path(binding["source_run_manifest"]["path"])
                          .read_text(encoding="utf-8"))
    proof = manifest.get("official_checkpoint_load_proof") or {}
    source = proof.get("source") or {}
    require(manifest.get("lifecycle") == "DIAGNOSTIC_SEALED" and
            manifest.get("natural_P01_student_prefix") is True and
            manifest.get("deterministic") is True and
            manifest.get("teacher_actions_deployed") is False and
            manifest.get("diagnostic_rows_entered_in_on_policy_storage") is False and
            manifest.get("new_PPO_samples") == manifest.get("new_PPO_updates") ==
                manifest.get("new_optimizer_steps") == 0 and
            manifest.get("new_AUX_accepted_updates") ==
                manifest.get("new_AUX_attempted_steps") == 0 and
            proof.get("checkpoint_loaded_and_verified") is True and
            proof.get("policy_sampling_mode") == "deterministic_conditional_mean" and
            source.get("checkpoint_sha256") == SOURCE_CHECKPOINT_SHA256 and
            source.get("manifest_sha256") == SOURCE_MANIFEST_SHA256,
            "sealed probe-v2 manifest no longer proves deterministic zero-credit data")
    return binding


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL row {line_number}") from error
            require(isinstance(value, dict), f"JSONL row {line_number} is not an object")
            yield value


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    require(isinstance(value, list) and len(value) == length,
            f"{name} must have length {length}")
    result = tuple(float(item) for item in value)
    require(all(math.isfinite(item) for item in result), name + " must be finite")
    return result


def _phase_column(phase: str) -> int:
    require(re.fullmatch(r"P(?:0[1-9]|1[0-3])", phase) is not None,
            "invalid semantic phase")
    return int(phase[1:]) - 1


def _observation(row: dict[str, Any], phase: str) -> tuple[float, ...]:
    value = _finite_vector(row.get("policy_observation_vector"),
                           OBSERVATION_DIMENSION, "policy observation")
    require(all(abs(item) <= 20.0 for item in value),
            "policy observation violates the schema clip")
    bits = value[:13]
    require(all(item in (0.0, 1.0) for item in bits) and sum(bits) == 1.0 and
            bits[_phase_column(phase)] == 1.0,
            "phase and exact one-hot policy observation differ")
    return value


def _compact_pre_intervention_row(row: dict[str, Any], expected_index: int) -> dict[str, Any]:
    require(row.get("decision_index") == expected_index,
            "probe decision indices must be contiguous zero-based")
    phase = row.get("phase")
    require(isinstance(phase, str), "probe row lacks phase")
    observation = _observation(row, phase)
    original = _finite_vector(row.get("original_student_raw_full12"),
                              ACTION_DIMENSION, "original student raw")
    applied = _finite_vector(row.get("applied_raw_full12"),
                             ACTION_DIMENSION, "applied raw")
    audit = row.get("original_student_request_audit") or {}
    step = row.get("step_info") or {}
    # This is the mapped physical-unit request, not a second raw Gaussian
    # sample. It may legitimately differ through caps, HISTORY and mapping.
    # Retain its finite-vector validation, but never turn it into a raw label.
    _finite_vector(step.get("applied_action_full12"), ACTION_DIMENSION,
                   "step physical applied action")
    require(original == applied and
            _finite_vector(audit.get("selected_raw_full12"), ACTION_DIMENSION,
                           "request-audit selected raw") == original and
            _finite_vector(step.get("raw_policy_action_full12"), ACTION_DIMENSION,
                           "step raw action") == original,
            "pre-intervention student raw/applied/request audit differ")
    require(row.get("overridden_indices") == [] and row.get("probe_active") is False and
            all(row.get(key) == 0 for key in ZERO_KEYS) and
            row.get("teacher_actions_deployed") is False and
            row.get("diagnostic_rows_entered_in_on_policy_storage") is False and
            row.get("formal_deterministic_policy_result") is False,
            "pre-intervention row has override, learning credit, teacher, or formal-result claim")
    return {"decision_index": expected_index, "phase": phase,
            "observation": observation, "target_raw": original}


def _runs(rows: list[dict[str, Any]], phase: str) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in rows:
        if row["phase"] == phase:
            if current and row["decision_index"] != current[-1]["decision_index"] + 1:
                result.append(current)
                current = []
            current.append(row)
        elif current:
            result.append(current)
            current = []
    if current:
        result.append(current)
    return result


def plan_probe_rows(rows: Iterable[dict[str, Any]], source_binding: dict[str, Any],
                    *, maximum_rows_per_phase_partition: int = 32,
                    minimum_rows_per_phase_partition: int = 4) -> dict[str, Any]:
    """Validate all rows and select balanced contiguous train/holdout windows.

    The zero-based 0..997 boundary is semantic.  File line numbers are never
    used as decision IDs, and every row at index 998 or later is excluded.
    """
    validate_source_binding(source_binding)
    require(type(maximum_rows_per_phase_partition) is int and
            type(minimum_rows_per_phase_partition) is int and
            1 <= minimum_rows_per_phase_partition <= maximum_rows_per_phase_partition <= 64,
            "finite balanced partition bounds required")
    eligible: list[dict[str, Any]] = []
    phase_counts = {phase: 0 for phase in TARGET_PHASES}
    excluded_sparse = {phase: 0 for phase in SPARSE_EXCLUDED_PHASES}
    post_cutoff = 0
    count = 0
    for count, row in enumerate(rows, 1):
        index = count - 1
        require(row.get("decision_index") == index,
                "probe decision indices must be contiguous zero-based")
        if index > CUTOFF_INCLUSIVE:
            post_cutoff += 1
            continue
        compact = _compact_pre_intervention_row(row, index)
        phase = compact["phase"]
        if phase in TARGET_PHASES:
            eligible.append(compact)
            phase_counts[phase] += 1
        elif phase in excluded_sparse:
            excluded_sparse[phase] += 1
        else:
            require(phase not in INVARIANCE_PHASES,
                    "P10-P13 probe rows cannot become fit targets")
    require(count == SOURCE_DECISION_ROWS and post_cutoff ==
            SOURCE_DECISION_ROWS - CUTOFF_INCLUSIVE - 1,
            "sealed probe must contain all rows while excluding every index >=998")
    selected_runs: dict[str, list[dict[str, Any]]] = {}
    for phase in TARGET_PHASES:
        candidates = _runs(eligible, phase)
        require(candidates, "missing target phase: " + phase)
        selected_runs[phase] = max(candidates, key=lambda value: (len(value),
                                                                  -value[0]["decision_index"]))
    quota = min(maximum_rows_per_phase_partition,
                min(len(value) // 2 for value in selected_runs.values()))
    require(quota >= minimum_rows_per_phase_partition,
            "each target phase needs complete contiguous train and holdout windows")
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    windows: dict[str, Any] = {}
    for phase in TARGET_PHASES:
        run = selected_runs[phase]
        train_window = run[:quota]
        holdout_window = run[-quota:]
        require(train_window[-1]["decision_index"] < holdout_window[0]["decision_index"] and
                all(b["decision_index"] == a["decision_index"] + 1
                    for values in (train_window, holdout_window)
                    for a, b in zip(values, values[1:])),
                "train/holdout must be disjoint complete contiguous windows")
        train.extend(train_window)
        holdout.extend(holdout_window)
        windows[phase] = {"source_event_start": run[0]["decision_index"],
            "source_event_end": run[-1]["decision_index"],
            "source_event_rows": len(run),
            "train_start": train_window[0]["decision_index"],
            "train_end": train_window[-1]["decision_index"],
            "holdout_start": holdout_window[0]["decision_index"],
            "holdout_end": holdout_window[-1]["decision_index"]}
    train_ids = [row["decision_index"] for row in train]
    holdout_ids = [row["decision_index"] for row in holdout]
    require(set(train_ids).isdisjoint(holdout_ids) and
            all(index <= CUTOFF_INCLUSIVE for index in train_ids + holdout_ids),
            "selected partitions overlap or cross the intervention boundary")
    selected_payload = [{"decision_index": row["decision_index"],
                         "phase": row["phase"],
                         "observation": row["observation"],
                         "target_raw": row["target_raw"]}
                        for row in train + holdout]
    receipt = {"schema": SCHEMA,
        **source_binding,
        "decision_index_semantics": "zero_based_field_not_file_line_number",
        "decision_index_cutoff_inclusive": CUTOFF_INCLUSIVE,
        "post_cutoff_rows_strictly_excluded": post_cutoff,
        "observation_dimension": OBSERVATION_DIMENSION,
        "action_dimension": ACTION_DIMENSION,
        "target_semantics": TARGET_SEMANTICS,
        "step_applied_action_validated_as": "finite_physical_full12_not_raw_label",
        "target_phases": list(TARGET_PHASES), "phase_columns": list(PHASE_COLUMNS),
        "sparse_phases_excluded_from_fit": list(SPARSE_EXCLUDED_PHASES),
        "train_row_ids": train_ids, "holdout_row_ids": holdout_ids,
        "rows_per_phase_per_partition": quota,
        "phase_event_windows": windows, "eligible_phase_counts": phase_counts,
        "excluded_sparse_phase_counts": excluded_sparse,
        "partitions_disjoint": True, "holdout_windows_contiguous": True,
        "synthetic_or_intervention_rows_used": False,
        "teacher_deployed": False, "PPO_credit": 0, "new_AUX_credit": 0,
        "whole_failed_trajectory_used_as_success_label": False,
        "targets_are_physical_success_labels": False,
        "P10_P12_real_same_input_full_Gaussian_invariance_required": True,
        "P13_synthetic_and_zero_phase_column_invariance_required": True,
        "real_P13_validation_required": False,
        "selected_rows_content_sha256": canonical_digest(selected_payload)}
    receipt["receipt_content_sha256"] = canonical_digest(receipt)
    return {"receipt": receipt, "train_rows": train, "holdout_rows": holdout}


def validate_invariance_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Admit supplied real P10--P12 rows, never invent a reached P13 state.

    The caller must retain their sealed source/row binding. This validator
    checks the supplied vectors, not physical provenance. P13 is checked
    separately by an explicitly synthetic zero-selected-column unit test.
    """
    compact: list[dict[str, Any]] = []
    for offset, row in enumerate(rows):
        phase = row.get("phase")
        require(phase in REAL_INVARIANCE_PHASES,
                "real invariance row must be available P10-P12; P13 is synthetic-only")
        observation = _observation(row, phase)
        require(all(observation[column] == 0.0 for column in PHASE_COLUMNS),
                "selected phase columns must be zero in P10-P13")
        compact.append({"row_id": row.get("row_id", f"invariance:{offset}"),
                        "phase": phase, "observation": observation})
    require(set(row["phase"] for row in compact) == set(REAL_INVARIANCE_PHASES),
            "real invariance set must cover available P10-P12 phases")
    row_ids = [row["row_id"] for row in compact]
    require(len(row_ids) == len(set(row_ids)), "invariance row IDs must be unique")
    return {"phases": list(REAL_INVARIANCE_PHASES), "row_ids": row_ids,
            "actual_rows_by_phase": {phase: sum(row["phase"] == phase for row in compact)
                                     for phase in INVARIANCE_PHASES},
            "real_P13_validation_claimed": False,
            "P13_validation_mode": "synthetic_unit_test_and_zero_selected_phase_columns_algebra",
            "rows_content_sha256": canonical_digest(compact), "rows": compact}

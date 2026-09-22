"""Bounded CPU-only audit for an ordinary receiving-x3 P12 first update.

This consumes already-written checkpoint/rollout/audit evidence.  It never
launches Isaac, runs a model forward, mutates an optimizer, or reads beyond the
first 128 complete learner records.  The successful-nominal prefix is read only
through the first complete ``policy_credit_start`` record.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

torch.set_num_threads(1)

from wlr50_clean.ppo.semantic_history_actor import cap_transition_request_history
from wlr50_clean.ppo.semantic_receiving_wheel_profile import RECEIVING_WHEEL_POLICY
from wlr50_clean.ppo.semantic_receiving_wheel_sigma import receiving_wheel_effective_log_std
from wlr50_clean.ppo.semantic_training import state_hash


SAMPLES = 128
OPTIMIZER_STEPS = 20
GATE_STAGES = (9, 10, 11)  # P10--P12, zero based.
WHEEL_CHANNELS = {"FL": 8, "FR": 9, "RL": 10, "RR": 11}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_tensor(value: Any) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32)


def max_abs(left: Iterable[float], right: Iterable[float]) -> float:
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def stats(values: Iterable[float]) -> dict[str, float]:
    rows = [float(value) for value in values]
    require(bool(rows), "cannot summarize an empty sequence")
    return {"min": min(rows), "mean": sum(rows) / len(rows), "max": max(rows)}


def parameter_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if name.startswith("obs_normalizer."):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def sidecar(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.stem + "_manifest.json")


def load_complete_prefix(path: Path, handoffs_needed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stream only through the handoffs needed by these 128 learner rows."""
    digest = hashlib.sha256()
    decisions = 0
    ticks = 0
    native_ticks = 0
    pending_result = None
    handoffs: list[dict[str, Any]] = []
    total_decisions = 0
    total_ticks = 0
    total_native_ticks = 0
    with path.open("rb") as stream:
        for line in stream:
            require(line.endswith(b"\n"), "prefix evidence ended with a partial record")
            digest.update(line)
            row = json.loads(line)
            kind = row.get("kind")
            if kind == "checkpoint_prefix_decision":
                require(row.get("policy_credit") is False, "prefix decision received PPO credit")
                require(all(float(v) == 0.0 for v in row["raw_policy_action_full12"]),
                        "successful-nominal prefix sampled a nonzero policy action")
                require(all(float(v) == 0.0 for v in row["projected_residual_full12"]),
                        "successful-nominal prefix applied a nonzero residual")
                audit = row["actuator_target_effect_audit"]
                require(audit["verified"] is True, "prefix native target audit failed")
                require(audit["setter_dispatch_targets_equal"] is True,
                        "prefix setter/dispatch target mismatch")
                require(audit["actual_mapping_matches_dispatch"] is True,
                        "prefix native mapping mismatch")
                per_tick = row["actuator_target_effect_audit_ticks"]
                require(all(tick["verified"] is True for tick in per_tick),
                        "prefix per-tick native audit failed")
                require(len(per_tick) == int(row["physics_ticks"]),
                        "prefix native tick ledger length mismatch")
                decisions += 1
                ticks += int(row["physics_ticks"])
                native_ticks += len(per_tick)
            elif kind == "checkpoint_prefix_result":
                require(pending_result is None, "duplicate prefix result before handoff")
                require(row["policy_credit"] is False, "prefix result received PPO credit")
                require(row["request"]["source"] == "successful_nominal",
                        "prefix was not real successful nominal N")
                require(row["request"]["target_phase"] == "P12", "prefix target was not P12")
                require(row["prefix_decisions"] == decisions, "prefix decision count mismatch")
                require(row["prefix_physics_ticks"] == ticks, "prefix physics tick count mismatch")
                require(decisions > 0 and native_ticks == ticks,
                        "prefix did not provide one native audit per physics tick")
                pending_result = row
            elif kind == "policy_credit_start":
                require(pending_result is not None, "policy-credit handoff has no prefix result")
                start = row["start"]
                provenance = start["prefix_policy_provenance"]
                require(provenance["source"] == "successful_nominal" and provenance["policy_credit"] is False,
                        "handoff provenance is not zero-credit successful nominal")
                require(all(float(v) == 0.0 for v in provenance["raw_action_full12"]),
                        "handoff provenance contains a nonzero policy action")
                require(start["requested_phase"] == "P12", "handoff request was not P12")
                if pending_result["accepted"]:
                    require(pending_result["miss"] is None, "accepted prefix contains a miss")
                    require(start["mode"] == "successful_nominal_initialized_suffix",
                            "accepted prefix has unexpected handoff mode")
                    require(start["from_P01_current_policy"] is False and start["actual_phase"] == "P12",
                            "accepted prefix did not hand off live P12")
                    require(start["requested_phase_still_active_at_credit"] is True,
                            "accepted P12 was stale at credit start")
                    require(start["target_first_observed_decision"]
                            == pending_result["target_first_observed_decision"],
                            "accepted handoff target-entry count mismatch")
                else:
                    require(pending_result["miss"] is not None, "rejected prefix lacks its physical miss")
                    require(start["mode"] == "fresh_P01_fallback"
                            and start["from_P01_current_policy"] is True
                            and start["actual_phase"] == "P01", "prefix miss did not use the legal fresh-P01 fallback")
                    require(start["prefix_miss"] == pending_result["miss"], "fallback miss record mismatch")
                handoffs.append({"result": pending_result, "start": start})
                total_decisions += decisions
                total_ticks += ticks
                total_native_ticks += native_ticks
                pending_result = None
                decisions = ticks = native_ticks = 0
                if len(handoffs) == handoffs_needed:
                    break
    require(len(handoffs) == handoffs_needed, "needed learner episode handoff is not completely written")
    return handoffs, {
        "complete_records_through_credit_start_sha256": digest.hexdigest(),
        "attempts_read": len(handoffs),
        "accepted_P12_attempts": sum(item["result"]["accepted"] for item in handoffs),
        "fresh_P01_fallbacks": sum(not item["result"]["accepted"] for item in handoffs),
        "decisions": total_decisions,
        "physics_ticks": total_ticks,
        "verified_native_ticks": total_native_ticks,
        "attempts": [{
            "index": index,
            "accepted": item["result"]["accepted"],
            "miss": item["result"]["miss"],
            "prefix_decisions": item["result"]["prefix_decisions"],
            "prefix_physics_ticks": item["result"]["prefix_physics_ticks"],
            "target_first_observed_decision": item["result"]["target_first_observed_decision"],
            "handoff_mode": item["start"]["mode"],
            "handoff_phase": item["start"]["actual_phase"],
            "handoff_tick": item["start"]["physics_tick"],
            "handoff_time_s": item["start"]["sim_time_s"],
            "remaining_task_time_s": item["start"]["remaining_task_time_s"],
        } for index, item in enumerate(handoffs)],
    }


def load_first_records(path: Path, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    consumed = 0
    with path.open("rb") as stream:
        for _ in range(count):
            line = stream.readline()
            require(bool(line) and line.endswith(b"\n"), f"fewer than {count} complete learner records")
            digest.update(line)
            consumed += len(line)
            rows.append(json.loads(line))
    return rows, {
        "complete_records": count,
        "consumed_bytes": consumed,
        "consumed_prefix_sha256": digest.hexdigest(),
        "whole_active_file_hashed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-run-decisions", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = args.run.resolve()
    source_path = args.source_checkpoint.resolve()
    target_path = args.target_checkpoint.resolve()
    output = args.output.resolve()
    require(run.is_relative_to((ROOT / "runs").resolve()), "run must be under this repository's runs directory")
    require(source_path.is_file() and target_path.is_file(), "source/target checkpoint is missing")
    require(output.is_relative_to(OUT.resolve()), "report must remain in this output tree")
    require(output.suffix.lower() == ".json" and not output.exists(), "report must be a new JSON path")

    source_manifest_path = sidecar(source_path)
    target_manifest_path = sidecar(target_path)
    require(source_manifest_path.is_file() and target_manifest_path.is_file(), "checkpoint sidecar is missing")
    source_manifest = read_json(source_manifest_path)
    target_manifest = read_json(target_manifest_path)
    require(sha256_file(source_path) == source_manifest["checkpoint_sha256"], "source checkpoint hash mismatch")
    require(sha256_file(target_path) == target_manifest["checkpoint_sha256"], "target checkpoint hash mismatch")
    require(target_manifest["source_run"] == str(run), "target checkpoint belongs to another run")

    started_path = run / "run_manifest.started.json"
    started = read_json(started_path)
    request = started["arguments"]
    require(Path(request["checkpoint"]).resolve() == source_path, "run did not resume the supplied source")
    require(request["decisions"] == args.expected_run_decisions, "run request was not the expected 512 decisions")
    require(request["stage"] == "phase_suffix" and request["from_phase"] == "P12", "run was not a P12 suffix")
    require(request["prefix_source"] == "successful_nominal", "run did not request real N prefix")
    require(request["teacher_offset_decisions"] == 0, "run requested a post-P12 teacher offset")
    require(request["resume_migration"] is None, "ordinary resume repeated a migration plan")
    require(request["new_mdp_warm_start"] is False, "ordinary resume requested a new-MDP warm start")
    require(request["policy_distribution_migration"] is False, "ordinary resume requested a distribution migration")
    require(request["_migration_record"] is None and request["_policy_migration_record"] is None,
            "ordinary resume contains a migration preflight record")
    require(request["_policy_version"] == RECEIVING_WHEEL_POLICY, "startup did not recognize receiving-x3")

    source = torch.load(source_path, map_location="cpu", weights_only=False)
    target = torch.load(target_path, map_location="cpu", weights_only=False)
    for label, payload, metadata in (("source", source, source_manifest), ("target", target, target_manifest)):
        require(all(metadata.get(key) == value for key, value in payload["infos"].items()),
                f"{label} embedded infos differ from sidecar")
        require(parameter_hash(payload["actor_state_dict"]) == metadata["actor_parameter_sha256"],
                f"{label} actor hash mismatch")
        require(parameter_hash(payload["critic_state_dict"]) == metadata["critic_parameter_sha256"],
                f"{label} critic hash mismatch")
        require(state_hash(payload["optimizer_state_dict"]) == metadata["optimizer_state_sha256"],
                f"{label} optimizer hash mismatch")
        require(all(not key.startswith("obs_normalizer.") for role in ("actor", "critic")
                    for key in payload[role + "_state_dict"]), f"{label} has unexpected learned normalizer")
        require(state_hash({"actor": {}, "critic": {}}) == metadata["normalizer_state_sha256"],
                f"{label} identity normalizer hash mismatch")

    require(source_manifest["policy_contract"] == target_manifest["policy_contract"],
            "ordinary resume changed policy contract")
    require(target_manifest["policy_contract"]["version"] == RECEIVING_WHEEL_POLICY,
            "target policy is not receiving-x3")
    require(source_manifest["runtime_contract"] == target_manifest["runtime_contract"],
            "ordinary resume changed runtime contract")
    require(source_manifest["receiving_wheel_sigma_migration"]
            == target_manifest["receiving_wheel_sigma_migration"],
            "persisted receiving migration receipt changed")
    require(source_manifest["training_quantity_budget_extension"]
            == target_manifest["training_quantity_budget_extension"],
            "persisted quantity-only receipt changed")
    ancestry = target_manifest["resume_ancestry"]
    expected_source_receipt = {
        "checkpoint": str(source_path),
        "checkpoint_sha256": source_manifest["checkpoint_sha256"],
        "manifest": str(source_manifest_path),
        "manifest_sha256": sha256_file(source_manifest_path),
    }
    require(ancestry["source_checkpoint"] == expected_source_receipt, "immediate source receipt mismatch")
    require(ancestry["source_actor_parameter_sha256"] == source_manifest["actor_parameter_sha256"],
            "resume actor lineage mismatch")
    require(ancestry["source_runtime_contract"] == source_manifest["runtime_contract"],
            "resume runtime lineage mismatch")
    require(ancestry["resume_migration"] is None, "ordinary target ancestry repeated the old migration")
    for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps"):
        require(ancestry["source_" + key] == source_manifest[key], f"resume {key} origin mismatch")
    expected_delta = {"global_policy_decisions": SAMPLES, "ppo_updates": 1,
                      "optimizer_steps": OPTIMIZER_STEPS}
    actual_delta = {key: target_manifest[key] - source_manifest[key] for key in expected_delta}
    require(actual_delta == expected_delta, "first-update lifetime counter delta mismatch")
    expected_stage = dict(source_manifest["stage_requested_decisions"])
    expected_stage["phase_suffix"] += SAMPLES
    require(target_manifest["stage_requested_decisions"] == expected_stage, "stage spending mismatch")
    for key in source_manifest:
        if key.endswith("_branch_counts"):
            delta = {field: target_manifest[key][field] - source_manifest[key][field]
                     for field in expected_delta}
            require(delta == expected_delta, f"{key} did not advance with ordinary PPO")
        if key.endswith("_branch"):
            require(target_manifest[key] == source_manifest[key], f"{key} metadata changed")
    ledger = target_manifest["task_conditioned_hip_wheel_branch"]["auxiliary_mean_learning"]
    require(ledger["accepted_auxiliary_updates_total"] == 7
            and ledger["attempted_auxiliary_optimizer_steps_total"] == 8,
            "LIMITED AUX 7/8 ledger changed")

    source_optimizer = source["optimizer_state_dict"]
    target_optimizer = target["optimizer_state_dict"]
    require(source_optimizer["state"].keys() == target_optimizer["state"].keys(),
            "Adam parameter-state keys changed")
    adam_step_deltas = []
    for key, before in source_optimizer["state"].items():
        after = target_optimizer["state"][key]
        require(before.keys() == after.keys() and {"step", "exp_avg", "exp_avg_sq"} <= set(before),
                "Adam state schema changed")
        for moment in ("exp_avg", "exp_avg_sq"):
            require(before[moment].shape == after[moment].shape
                    and bool(torch.isfinite(after[moment]).all()), "Adam moment shape/finite check failed")
        adam_step_deltas.append(float(after["step"]) - float(before["step"]))
    require(adam_step_deltas and set(adam_step_deltas) == {float(OPTIMIZER_STEPS)},
            "Adam steps did not advance by exactly 20")
    require(len(source_optimizer["param_groups"]) == len(target_optimizer["param_groups"]),
            "Adam parameter-group count changed")
    for before, after in zip(source_optimizer["param_groups"], target_optimizer["param_groups"]):
        require({k: v for k, v in before.items() if k != "lr"}
                == {k: v for k, v in after.items() if k != "lr"},
                "Adam non-LR parameter-group settings changed")
        require(float(before["lr"]) == float(source_manifest["optimizer_learning_rate"]),
                "source effective LR was not restored")
        require(float(after["lr"]) == float(target_manifest["optimizer_learning_rate"]),
                "target effective LR differs from saved optimizer")

    rows, learner_source = load_first_records(run / "residual_and_projection_audit.jsonl", SAMPLES)
    learner_episode_count = 1 + sum(bool(row["terminal"]) for row in rows[:-1])
    handoffs, prefix_summary = load_complete_prefix(
        run / "prefix_evidence.jsonl", learner_episode_count)

    update = target_manifest["last_update"]
    update_index = target_manifest["ppo_updates"]
    require(update["ppo_update"] == update_index, "target last-update index mismatch")
    require(update["global_policy_decisions"] == target_manifest["global_policy_decisions"],
            "target last-update decision count mismatch")
    require(update["optimizer_steps"] == OPTIMIZER_STEPS, "target update did not use 20 Adam steps")
    require(update["optimizer_learning_rate"] == target_manifest["optimizer_learning_rate"],
            "reported update LR differs from target optimizer LR")
    require(update["actor_parameter_sha256_before"] == source_manifest["actor_parameter_sha256"],
            "first-update actor-before hash is not the immediate source")
    require(update["actor_parameter_sha256_after"] == target_manifest["actor_parameter_sha256"],
            "first-update actor-after hash is not the target")

    rollout_path = run / "rollouts" / f"rollout_{update_index:06d}.pt"
    likelihood_path = run / "rollouts" / f"update_{update_index:06d}_likelihood.json"
    require(rollout_path.is_file() and likelihood_path.is_file(), "first rollout/update evidence is missing")
    rollout = torch.load(rollout_path, map_location="cpu", weights_only=False)
    require(tuple(rollout["actions"].shape) == (SAMPLES, 1, 12), "unexpected rollout action shape")
    require(tuple(rollout["observations"]["policy"].shape) == (SAMPLES, 1, 372),
            "unexpected rollout observation shape")
    require(rollout["policy_contract"] == target_manifest["policy_contract"], "rollout policy mismatch")
    require(rollout["runtime_contract"] == target_manifest["runtime_contract"], "rollout runtime mismatch")
    expected_checkpoint_curriculum = {
        **rollout["curriculum_epoch"],
        "changes_allowed_only_between_complete_rollout_updates": True,
    }
    require(expected_checkpoint_curriculum == target_manifest["curriculum_epoch"],
            "rollout/checkpoint curriculum epoch mismatch outside the checkpoint boundary flag")
    curriculum_prefix = rollout["curriculum_epoch"]["prefix_request"]
    require(curriculum_prefix["source"] == "successful_nominal"
            and curriculum_prefix["target_phase"] == "P12", "rollout prefix request mismatch")

    phases: Counter[str] = Counter()
    active_phases: Counter[str] = Counter()
    current_modes: Counter[str] = Counter()
    gate_active = 0
    pre_history_true = 0
    end_history_true = 0
    end_support = 0
    end_top = 0
    end_air = 0
    end_lift_valid = 0
    end_usable = 0
    native_ticks = 0
    request_sigma_error = 0.0
    other_ten_error = 0.0
    active_fr_rr_ratio_error = 0.0
    raw_dispatch_error = 0.0
    request_projection_error = 0.0
    final_composition_error = 0.0
    candidate_composition_error = 0.0
    target_sources: set[str] = set()
    phase_masks: set[tuple[int, ...]] = set()
    base_sigmas: dict[str, list[float]] = {name: [] for name in ("FR", "RR")}
    effective_sigmas: dict[str, list[float]] = {name: [] for name in ("FR", "RR")}
    wheel_response = {name: {field: [] for field in ("requested", "effective", "final_target", "measured_qd")}
                      for name in ("FR", "RR")}
    rr_positive_request_negative_qd: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    first_global = source_manifest["global_policy_decisions"] + 1
    previous_tick = None
    learner_episode = -1

    for index, row in enumerate(rows):
        require(row["global_policy_decision"] == first_global + index, "learner decision sequence mismatch")
        request_row = row["policy_request"]
        applied = row["applied_audit"]
        if index == 0 or bool(rows[index - 1]["terminal"]):
            learner_episode += 1
            previous_tick = int(handoffs[learner_episode]["start"]["physics_tick"])
        require(applied["physics_tick"] == previous_tick + applied["physics_ticks"],
                "learner native tick sequence is not continuous within its real episode")
        previous_tick = int(applied["physics_tick"])
        require(request_row["policy_version"] == RECEIVING_WHEEL_POLICY, "learner request used another policy")
        require(request_row["sampling_draws"] == 1, "request did not use exactly one Gaussian draw")
        require(request_row["extra_model_forwards"] == request_row["extra_random_draws"] == 0,
                "request used extra forward/RNG work")
        require(applied["prefix_teacher_data_in_ppo_storage"] is False
                and applied["prefix_checkpoint_policy_data_in_ppo_storage"] is False,
                "prefix data leaked into PPO storage")
        require(torch.equal(as_tensor(request_row["selected_raw_full12"]), rollout["actions"][index, 0]),
                "stored raw action differs from request")
        require(torch.equal(as_tensor(request_row["conditional_mean_full12"]),
                            rollout["distribution_params"][0][index, 0]), "stored old mean differs")
        require(torch.equal(as_tensor(request_row["effective_sigma_full12"]),
                            rollout["distribution_params"][1][index, 0]), "stored old sigma differs")
        require(torch.equal(as_tensor(row["old_distribution_mean_full12"]),
                            rollout["distribution_params"][0][index, 0]), "row old mean differs")
        require(torch.equal(as_tensor(row["old_distribution_std_full12"]),
                            rollout["distribution_params"][1][index, 0]), "row old sigma differs")
        require(request_row["selected_raw_log_probability"] == row["old_log_probability"]
                == rollout["actions_log_prob"][index, 0, 0].item(), "stored old log-probability differs")
        require(float(row["old_value"]) == rollout["values"][index, 0, 0].item(), "stored old value differs")
        require(float(row["reward"]) == rollout["rewards"][index, 0, 0].item(), "stored reward differs")
        require(bool(row["terminal"]) == bool(rollout["dones"][index, 0, 0].item()), "stored done differs")

        observation = rollout["observations"]["policy"][index]
        stage_index = int(observation[0, :13].argmax())
        history_placed = observation[0, 157].item() == 1.0
        active = stage_index in GATE_STAGES and history_placed
        require(request_row["stage_index"] == stage_index, "request stage differs from stored observation")
        require(request_row["RR_placed_history"] is history_placed, "request RR history differs from observation")
        require(request_row["receiving_continuation_active"] is active, "receiving gate differs from observation")
        multiplier = torch.ones(12)
        if active:
            multiplier[[WHEEL_CHANNELS["FR"], WHEEL_CHANNELS["RR"]]] = 3.0
        require(torch.equal(as_tensor(request_row["receiving_sigma_multiplier_full12"]), multiplier),
                "receiving multiplier is not conditional FR/RR x3")
        learned = as_tensor(request_row["learned_sigma_full12"])
        innovation = as_tensor(request_row["innovation_sigma_multiplier_full12"])
        base = learned * 0.25 * innovation
        effective = base * multiplier
        request_sigma_error = max(request_sigma_error,
                                  float((effective - as_tensor(request_row["effective_sigma_full12"])).abs().max()))
        torch.testing.assert_close(effective, as_tensor(request_row["effective_sigma_full12"]),
                                   rtol=2e-6, atol=1e-8)
        for channel in range(12):
            if channel not in (WHEEL_CHANNELS["FR"], WHEEL_CHANNELS["RR"]):
                other_ten_error = max(other_ten_error,
                                      abs(float(effective[channel]) - float(base[channel])))
        if active:
            gate_active += 1
            active_phases[f"P{stage_index + 1:02d}"] += 1
            for name in ("FR", "RR"):
                channel = WHEEL_CHANNELS[name]
                active_fr_rr_ratio_error = max(active_fr_rr_ratio_error,
                                               abs(float(effective[channel] / base[channel]) - 3.0))
                base_sigmas[name].append(float(base[channel]))
                effective_sigmas[name].append(float(effective[channel]))
        pre_history_true += int(history_placed)
        phases[applied["phase_id"]] += 1

        actuator = applied["actuator_target_effect_audit"]
        headroom = actuator["policy_headroom_evidence"]
        require(actuator["verified"] is True and actuator["setter_dispatch_targets_equal"] is True
                and actuator["actual_mapping_matches_dispatch"] is True, "learner native dispatch failed")
        require(all(tick["verified"] is True for tick in applied["actuator_target_effect_audit_ticks"]),
                "learner per-tick native dispatch failed")
        phase_masks.add(tuple(int(value) for value in actuator["phase_mask_full12"]))
        raw_dispatch_error = max(raw_dispatch_error,
                                 max_abs(row["raw_policy_action_full12"], actuator["raw_policy_action_full12"]))
        request_projection_error = max(request_projection_error,
                                       max_abs(applied["projected_residual_full12"],
                                               actuator["projected_residual_full12"]),
                                       max_abs(headroom["requested_policy_residual_full12"],
                                               actuator["projected_residual_full12"]))
        candidate = [float(n) + float(b) for n, b in zip(
            headroom["geometry_corrected_native_full12"],
            headroom["effective_combined_post_mapper_bias_full12"])]
        candidate_composition_error = max(candidate_composition_error,
                                          max_abs(headroom["candidate_native_target_before_final_slew_full12"],
                                                  candidate))
        # Servos retain the established final clamp/slew. Wheels have no later
        # mapper stage, so their final targets must equal baseline + residual.
        final_wheels = [float(n) + float(r) for n, r in zip(
            headroom["baseline_native_plus_controller_full12"][8:],
            headroom["effective_policy_residual_full12"][8:])]
        final_composition_error = max(final_composition_error,
                                      max_abs(applied["actual_drive_target_full12"][8:], final_wheels))
        native_ticks += len(applied["actuator_target_effect_audit_ticks"])
        target_sources.add(actuator["actual_target_source"])

        task = applied["semantic_task"]
        evaluator = task["physical_evaluator"]
        rr = evaluator["current_legs"]["RR"]
        if active:
            end_history_true += int(task["placed_history"]["RR"])
            end_support += int(rr["support"])
            end_top += int(rr["top_surface_contact"] and rr["bearing_verified"] and rr["support"])
            end_air += int(rr["air"])
            end_lift_valid += int(rr["current_lift_valid"])
            end_usable += int(task["rr_placed_currently_usable"])
            current_modes[rr["contact_mode"]] += 1
            measured = evaluator["measured_wheel_velocity_rad_s"]
            for name in ("FR", "RR"):
                channel = WHEEL_CHANNELS[name]
                physical_index = ("FL", "FR", "RL", "RR").index(name)
                wheel_response[name]["requested"].append(float(headroom["requested_policy_residual_full12"][channel]))
                wheel_response[name]["effective"].append(float(headroom["effective_policy_residual_full12"][channel]))
                wheel_response[name]["final_target"].append(float(applied["actual_drive_target_full12"][channel]))
                wheel_response[name]["measured_qd"].append(float(measured[physical_index]))
            if (wheel_response["RR"]["requested"][-1] > 0.0
                    and wheel_response["RR"]["measured_qd"][-1] < 0.0):
                rr_positive_request_negative_qd.append({
                    "decision": row["global_policy_decision"],
                    "tick": applied["physics_tick"],
                    "phase": applied["phase_id"],
                    "RR_requested_residual_rad_s": wheel_response["RR"]["requested"][-1],
                    "RR_effective_residual_rad_s": wheel_response["RR"]["effective"][-1],
                    "RR_final_target_rad_s": wheel_response["RR"]["final_target"][-1],
                    "RR_measured_endpoint_qd_rad_s": wheel_response["RR"]["measured_qd"][-1],
                    "RR_contact_mode": rr["contact_mode"],
                    "RR_support": rr["support"],
                    "RR_TOP": rr["top_surface_contact"] and rr["bearing_verified"] and rr["support"],
                    "RR_AIR": rr["air"],
                })
            if len(selected) < 1 or (not rr["support"] and all(s["end_RR_support"] for s in selected)):
                selected.append({
                    "decision": row["global_policy_decision"],
                    "tick": applied["physics_tick"],
                    "phase": applied["phase_id"],
                    "pre_action_RR_placed_history": history_placed,
                    "gate_active": active,
                    "end_RR_mode": rr["contact_mode"],
                    "end_RR_support": rr["support"],
                    "end_RR_TOP": rr["top_surface_contact"] and rr["bearing_verified"] and rr["support"],
                    "end_RR_AIR": rr["air"],
                    "FR_base_effective_sigma": [float(base[9]), float(effective[9])],
                    "RR_base_effective_sigma": [float(base[11]), float(effective[11])],
                })

    require(learner_episode + 1 == learner_episode_count, "learner episode/handoff accounting mismatch")
    require(other_ten_error == 0.0, "receiving profile changed one of the other ten sigmas")
    require(raw_dispatch_error == 0.0 and request_projection_error == 0.0,
            "request/projected/native audit identity failed")
    require(candidate_composition_error < 1e-10,
            "pre-slew candidate does not equal native target + combined bias")
    require(final_composition_error < 1e-10,
            "final wheel target does not equal mapped baseline + effective residual")

    likelihood = read_json(likelihood_path)
    require(likelihood["extra_model_forwards"] == likelihood["extra_random_draws"] == 0,
            "optimizer likelihood audit used extra forward/RNG work")
    require(len(likelihood["minibatches"]) == OPTIMIZER_STEPS, "optimizer did not emit 20 minibatches")
    coverage: Counter[int] = Counter()
    current_mean_error = 0.0
    current_sigma_error = 0.0
    current_logp_error = 0.0
    first_ratio_error = 0.0
    entropies = []
    for batch in likelihood["minibatches"]:
        require(all(isinstance(item, list) and len(item) == 1
                    for item in batch["rollout_flat_indices"]),
                "observation/action lookup is ambiguous; exact minibatch identity cannot be claimed")
        indices = [int(item[0]) for item in batch["rollout_flat_indices"]]
        require(all(0 <= item < SAMPLES for item in indices), "minibatch index is outside saved rollout")
        coverage.update(indices)
        observations = rollout["observations"]["policy"][indices, 0]
        current_mean = as_tensor(batch["current_conditional_mean"])
        current_sigma = as_tensor(batch["current_conditional_sigma"])
        history, _ = cap_transition_request_history(observations)
        expected_mean = 0.1 * as_tensor(batch["current_network_mean_full12"]) + 0.9 * history
        effective_log_sigma, _ = receiving_wheel_effective_log_std(
            as_tensor(batch["current_network_log_sigma_full12"]), observations, 0.25)
        expected_sigma = effective_log_sigma.exp()
        current_mean_error = max(current_mean_error, float((expected_mean - current_mean).abs().max()))
        current_sigma_error = max(current_sigma_error, float((expected_sigma - current_sigma).abs().max()))
        torch.testing.assert_close(current_mean, expected_mean, rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(current_sigma, expected_sigma, rtol=2e-6, atol=1e-8)
        require(batch["sigma_source"]
                == "current_official_Gaussian_cache_after_B_over_cap_and_receiving_FR_RR_sigma_x3",
                "optimizer used another sigma source")
        distribution = torch.distributions.Normal(current_mean, current_sigma)
        reconstructed_logp = distribution.log_prob(rollout["actions"][indices, 0]).sum(-1)
        optimization_logp = as_tensor(batch["optimization_log_probability"])
        current_logp_error = max(current_logp_error,
                                 float((reconstructed_logp - optimization_logp).abs().max()))
        torch.testing.assert_close(reconstructed_logp, optimization_logp, rtol=3e-6, atol=1e-5)
        require(torch.equal(as_tensor(batch["old_log_probability"]),
                            rollout["actions_log_prob"][indices, 0, 0]),
                "minibatch old log-probability is not the saved rollout value")
        require(torch.equal(as_tensor(batch["actual_advantage"]),
                            rollout["advantages"][indices, 0, 0]),
                "minibatch advantage is not the indexed saved rollout value")
        if batch["minibatch_index"] == 0:
            first_ratio_error = max(first_ratio_error, max(abs(float(v) - 1.0) for v in batch["ratio"]))
        entropies.append(float(distribution.entropy().sum(-1).mean()))
    require(coverage == Counter({index: 5 for index in range(SAMPLES)}),
            "minibatch coverage is not five epochs over every saved sample")
    reconstructed_entropy = sum(entropies) / len(entropies)
    require(math.isclose(reconstructed_entropy, float(update["entropy"]), rel_tol=0.0, abs_tol=2e-5),
            "CPU likelihood entropy differs from official update")

    result = {
        "schema": "wlr50_clean.receiving_p12_first128_readonly.v1",
        "status": "PASS",
        "scope": "bounded immutable CPU audit; no Isaac/model-forward/optimizer step; no physical success claim",
        "run": {
            "path": str(run),
            "started_manifest_sha256": sha256_file(started_path),
            "requested_policy_decisions": request["decisions"],
            "requested_stage": request["stage"],
            "requested_phase": request["from_phase"],
        },
        "source": {
            "path": str(source_path),
            "sha256": source_manifest["checkpoint_sha256"],
            "counters": {key: source_manifest[key] for key in expected_delta},
            "effective_learning_rate": source_manifest["optimizer_learning_rate"],
        },
        "target_first128": {
            "path": str(target_path),
            "sha256": target_manifest["checkpoint_sha256"],
            "counters": {key: target_manifest[key] for key in expected_delta},
            "counter_delta": actual_delta,
            "effective_learning_rate": target_manifest["optimizer_learning_rate"],
            "learner_source": learner_source,
        },
        "ordinary_resume": {
            "same_policy_contract": True,
            "same_runtime_contract": True,
            "immediate_source_receipt_exact": True,
            "persisted_receiving_migration_receipt_exact": True,
            "new_resume_migration": None,
            "no_repeated_migration": True,
            "identity_normalizer_both": True,
            "LIMITED_AUX_accepted_attempted": [7, 8],
            "all_inherited_branch_counter_deltas": expected_delta,
        },
        "successful_nominal_prefix": {
            **prefix_summary,
            "target_phase": "P12",
            "learner_credit": 0,
            "learner_episodes_in_first128": learner_episode_count,
        },
        "receiving_gate": {
            "learner_samples": SAMPLES,
            "phase_counts": dict(sorted(phases.items())),
            "pre_action_RR_placed_history_true": pre_history_true,
            "active_samples": gate_active,
            "active_profile_observed": gate_active > 0,
            "coverage_status": ("ACTIVE_FR_RR_X3_OBSERVED" if gate_active
                                else "NO_ACTIVE_SAMPLE_RECORDED_REPORT_ONLY"),
            "active_by_phase": dict(sorted(active_phases.items())),
            "condition": "stored pre-action stage in P10-P12 AND stored pre-action RR placed history",
            "FR_RR_multiplier_when_active": 3.0,
            "request_sigma_identity_max_abs_error": request_sigma_error,
            "active_FR_RR_ratio_max_abs_error": active_fr_rr_ratio_error,
            "other_ten_unchanged_max_abs_error": other_ten_error,
            "active_sigma": {
                name: ({"base": stats(base_sigmas[name]), "effective": stats(effective_sigmas[name])}
                       if base_sigmas[name] else None)
                for name in ("FR", "RR")
            },
            "post_action_RR_endpoint_only": {
                "historical_placed_true": end_history_true,
                "currently_usable_true": end_usable,
                "support_true": end_support,
                "TOP_true": end_top,
                "AIR_true": end_air,
                "current_lift_valid_true": end_lift_valid,
                "contact_modes": dict(sorted(current_modes.items())),
                "not_gate_input_or_continuous_8tick_claim": True,
            },
            "selected_endpoints": selected[:2],
        },
        "wheel_physical_response": {
            "scope": "receiving-gate-active samples; request/target and post-action 15Hz endpoint canonical qd",
            "canonical_wheel_order": ["FL", "FR", "RL", "RR"],
            "active_samples": gate_active,
            "FR_RR": {
                name: ({field: stats(values) for field, values in wheel_response[name].items()}
                       if gate_active else None)
                for name in ("FR", "RR")
            },
            "RR_positive_requested_count": sum(value > 0.0 for value in wheel_response["RR"]["requested"]),
            "RR_positive_requested_but_endpoint_qd_negative_count": len(rr_positive_request_negative_qd),
            "RR_positive_requested_and_endpoint_qd_nonnegative_count": sum(
                request_value > 0.0 and qd >= 0.0 for request_value, qd in zip(
                    wheel_response["RR"]["requested"], wheel_response["RR"]["measured_qd"])),
            "RR_positive_request_positive_final_target_but_endpoint_qd_negative_count": sum(
                row["RR_final_target_rad_s"] > 0.0 for row in rr_positive_request_negative_qd),
            "RR_positive_request_nonpositive_final_target_and_endpoint_qd_negative_count": sum(
                row["RR_final_target_rad_s"] <= 0.0 for row in rr_positive_request_negative_qd),
            "selected_positive_final_target_negative_endpoint_qd": [
                row for row in rr_positive_request_negative_qd if row["RR_final_target_rad_s"] > 0.0
            ][:3],
            "interpretation_limit": (
                "Measured qd is the action-interval endpoint response to the full12 target, nominal/controller, "
                "contact and whole-body dynamics; sign opposition is descriptive, not a mask or single-channel causal claim."),
        },
        "sampling_and_update": {
            "stored_raw_mean_sigma_old_logp_exact": True,
            "one_draw_no_extra_forward_or_rng": True,
            "minibatches": len(likelihood["minibatches"]),
            "each_saved_sample_seen": 5,
            "current_mean_max_abs_error": current_mean_error,
            "current_sigma_max_abs_error": current_sigma_error,
            "current_logp_CPU_reconstruction_max_abs_error": current_logp_error,
            "first_minibatch_ratio_max_error_from_one": first_ratio_error,
            "reported_entropy": update["entropy"],
            "CPU_reconstructed_entropy": reconstructed_entropy,
            "reported_KL": update["kl_mean"],
            "reported_clip_fraction": update["clip_fraction"],
        },
        "native_dispatch": {
            "all_first128_requests_and_ticks_verified": True,
            "native_physics_ticks": native_ticks,
            "observed_phase_masks": [list(mask) for mask in sorted(phase_masks)],
            "all_observed_phase_masks_full12": phase_masks == {(1,) * 12},
            "raw_request_max_abs_error": raw_dispatch_error,
            "requested_projected_residual_max_abs_error": request_projection_error,
            "final_target_composition_max_abs_error": final_composition_error,
            "pre_slew_candidate_composition_max_abs_error": candidate_composition_error,
            "actual_target_sources": sorted(target_sources),
        },
        "optimizer": {
            "parameter_state_count": len(adam_step_deltas),
            "all_Adam_step_deltas": OPTIMIZER_STEPS,
            "non_LR_group_settings_unchanged": True,
            "source_LR_restored": True,
            "target_LR_matches_saved_update": True,
            "adaptive_LR_was_not_assumed_constant": True,
            "actor_before_equals_immediate_source": True,
            "actor_after_equals_target": True,
        },
        "limitations": [
            "Current RR contact is the logged post-action endpoint; historical placed is the stored pre-action gate input.",
            "Endpoint counts do not claim every intermediate 120 Hz contact state or causal support improvement.",
            "The first-update audit validates ordinary resume, probability accounting and native dispatch, not task success.",
            "A legal fresh-P01 fallback or zero active receiving samples is reported, not promoted to a runtime/training failure.",
            "Wheel request-versus-measured-qdot sign is a same-interval endpoint diagnostic, not a single-channel counterfactual.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({
        "status": result["status"],
        "output": str(output),
        "source": result["source"]["counters"],
        "target": result["target_first128"]["counters"],
        "prefix_decisions": prefix_summary["decisions"],
        "receiving_gate_active_samples": gate_active,
        "active_endpoint_RR_support_TOP_AIR": [end_support, end_top, end_air],
        "minibatches": len(likelihood["minibatches"]),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

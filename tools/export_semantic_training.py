"""Offline export of checkpoint-bound optimized semantic transitions.

Standard library only. Does not import Torch, Isaac, or runtime project modules.
CSV native values are last-tick tensor targets (radians/rad/s), not logical
degree commands. Missing historical audit fields remain "unavailable".
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

FAMILIES = ("task_progress", "body_stability", "contact_motion_quality",
            "control_smoothness", "control_regularization")
CHANNELS = ("front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
            "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
            "front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle")
VECTORS = ("raw", "nominal", "projected_residual", "applied", "actual_drive",
           "native_last_tick", "native_counterfactual_last_tick", "native_delta_last_tick")
NA = "unavailable"
BASE_FIELDS = (
    "global_policy_decision", "ppo_update", "env_index", "num_envs", "phase", "end_phase",
    "episode_decision", "episode_physics_tick", "physics_ticks_this_decision", "sim_time_s",
    "source_head", "runtime_content_sha256", "reward_config_sha256", "reward_source_sha256",
    "task_spec_sha256", "supervisor_sha256", "source_run", "source_checkpoint",
    "source_checkpoint_sha256", "source_jsonl_line", "optimized",
    "old_log_probability", "old_value", "ppo_storage_reward", "environment_reward",
    "peer_bootstrap_credit", "reward_origin", "ppo_done", "physical_task_terminated",
    "external_peer_reset_truncation", "termination_interpretation", "termination_reason",
    "task_success", "all_tick_native_audit", "verified_native_tick_count",
    "native_effect_tick_count", "own_phase_effect_tick_count", "native_last_tick_verified",
    "native_last_tick_target_dtype", "no_in_episode_state_writes_verified",
    "in_episode_root_pose_writes", "in_episode_root_velocity_writes",
    "in_episode_force_or_impulse_writes", "in_episode_gravity_writes",
)
CSV_FIELDS = BASE_FIELDS + tuple("reward_" + family for family in FAMILIES) + tuple(
    f"{prefix}_{channel}" for prefix in VECTORS for channel in CHANNELS)
COVERAGE_FIELDS = ("source_head", "runtime_content_sha256", "reward_config_sha256",
    "task_spec_sha256", "phase", "optimized_policy_decisions", "physical_ticks",
    "physical_time_s", "ppo_done_count", "physical_task_terminal_count",
    "physical_failure_count", "peer_truncation_count", "training_success_count",
    "all_tick_audit_verified_decisions", "all_tick_audit_unavailable_decisions")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def record(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size}


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"nonfinite {label}")
    return value


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"invalid {label}")
    return value


def vector(value, label, *, optional=False):
    if value is None and optional:
        return [NA] * 12
    require(isinstance(value, (list, tuple)) and len(value) == 12, f"missing Full12 {label}")
    return [finite(x, label) for x in value]


def native_vector(audit, key):
    value = audit.get(key)
    if value is None:
        return [NA] * 12
    servo, wheel = value.get("servo_position_rad"), value.get("wheel_velocity_rad_s")
    require(isinstance(servo, list) and len(servo) == 8 and isinstance(wheel, list) and len(wheel) == 4,
            f"invalid native target vector {key}")
    return vector(servo + wheel, key)


def flag(value):
    return value if type(value) is bool else NA


def option(arguments, name):
    indices = [i for i, item in enumerate(arguments) if item == name]
    require(len(indices) <= 1, f"duplicate launcher argument {name}")
    if not indices:
        return None
    require(indices[0] + 1 < len(arguments), f"missing launcher argument value {name}")
    return arguments[indices[0] + 1]


def stable_jsonl(path):
    """Yield valid finite JSON; caller checks decision duplicates and gaps."""
    path = Path(path)
    before = path.stat()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid/truncated source JSONL at {path}:{line_no}") from exc
                yield line_no, value
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"source changed during offline export: {path}")


class Exporter:
    def __init__(self, project_root, checkpoint, initial_checkpoint):
        self.root = Path(project_root).resolve(strict=True)
        self.history = self.root / "outputs/ppo_semantic_v2/checkpoints/history"
        self.train_root = self.root / "runs/ppo_semantic_v2/train"
        self.inputs, self.metadata = {}, {}
        self.initial = self.load_checkpoint(initial_checkpoint)
        require(self.initial["global_policy_decisions"] == self.initial["ppo_updates"] ==
                self.initial["optimizer_steps"] == 0, "explicit initial checkpoint is not initial")
        self.latest = self.load_checkpoint(checkpoint)
        require(self.latest["global_policy_decisions"] > 0, "no optimized data in selected checkpoint")

    def capture(self, path):
        item = record(path)
        self.inputs[item["path"]] = item
        return item

    def load_checkpoint(self, path):
        path = Path(path).resolve(strict=True)
        require(path.is_relative_to(self.history.resolve()) and path.suffix == ".pt",
                "select an explicit immutable semantic history checkpoint, not checkpoint_last")
        if str(path) in self.metadata:
            return self.metadata[str(path)]
        sidecar = path.with_name(path.stem + "_manifest.json")
        data = read_json(sidecar)
        require(data.get("schema") == "wlr50_clean.semantic_checkpoint.v1" and
                Path(data.get("checkpoint_path", "")).resolve() == path and
                data.get("checkpoint_sha256") == sha(path) and data.get("save_load_round_trip") is True,
                "checkpoint/sidecar hash and actual recorded roundtrip disagree")
        contract = data["runtime_contract"]
        require(contract["runtime_content_sha256"] == digest(contract["files"]),
                "checkpoint runtime inventory digest mismatch")
        for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps"):
            integer(data[key], key)
        self.capture(path); self.capture(sidecar)
        self.metadata[str(path)] = data
        return data

    def source_run(self, child):
        run = Path(child["source_run"]).resolve(strict=True)
        require(run.is_relative_to(self.train_root.resolve()), "checkpoint source is not a semantic training run")
        started_path = run / "run_manifest.started.json"
        started = read_json(started_path)
        require(started.get("command") == "train" and started["runtime_contract"] == child["runtime_contract"],
                "checkpoint is not bound to the declared real training runtime")
        args = started["arguments"]
        require(Path(args["run_dir"]).resolve() == run and args["seed"] == child["seed"],
                "run launch identity/seed mismatch")
        launcher_path = run.with_name(run.name + "_launcher") / "arguments.json"
        launcher = read_json(launcher_path)
        require(option(launcher, "--run-dir") == str(run) and
                option(launcher, "--expected-head") == child["runtime_contract"]["source_git_commit"],
                "actual launcher does not bind the recorded training run")
        self.capture(started_path); self.capture(launcher_path)
        # A historical failure status is never rewritten as training success.
        final_path, stopped_path = run / "run_manifest.json", run / "external_stop.json"
        if final_path.exists():
            final = read_json(final_path)
            require(final.get("lifecycle") in ("SUCCEEDED", "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY", "FAILED"),
                    "run has not finalized; export only stable optimized prefixes")
            self.capture(final_path)
            status = final["lifecycle"]
        elif stopped_path.exists():
            stopped = read_json(stopped_path)
            require(stopped.get("lifecycle") == "EXTERNALLY_STOPPED_AFTER_VERIFIED_CHECKPOINT" and
                    stopped.get("partial_rollout_will_not_be_reused") is True,
                    "old run lacks explicit external-stop evidence")
            frontier = self.load_checkpoint(stopped["checkpoint"])
            require(frontier["checkpoint_sha256"] == stopped["checkpoint_sha256"] and
                    frontier["global_policy_decisions"] >= child["global_policy_decisions"],
                    "external-stop checkpoint does not cover selected immutable prefix")
            self.capture(stopped_path)
            status = stopped["lifecycle"]
        else:
            raise ValueError("training run is still active or lacks a finalized stop record")
        return run, started, launcher, status

    def ancestry(self):
        result, seen = [], set()
        child = self.latest
        while child["global_policy_decisions"]:
            path = child["checkpoint_path"]
            require(path not in seen, "checkpoint ancestry cycle")
            seen.add(path)
            run, started, launcher, status = self.source_run(child)
            resolved = started["arguments"].get("checkpoint")
            parent = self.initial if resolved is None else self.load_checkpoint(resolved)
            launch_arg = option(launcher, "--checkpoint")
            if resolved is None:
                require(launch_arg is None, "initial run unexpectedly declared a resume checkpoint")
            else:
                launch_path = Path(launch_arg).resolve() if launch_arg else None
                require(launch_path == Path(parent["checkpoint_path"]).resolve() or
                        launch_path == (self.history.parent / "checkpoint_last.pt").resolve(),
                        "launcher checkpoint differs from the immutable resolved run checkpoint")
            ancestry = child.get("resume_ancestry")
            method = "resolved_launch_argument_plus_checkpoint_hash_and_actor_update_chain"
            if ancestry is not None:
                source = ancestry["source_checkpoint"]
                require(Path(source["checkpoint"]).resolve() == Path(parent["checkpoint_path"]).resolve()
                        and source["checkpoint_sha256"] == parent["checkpoint_sha256"]
                        and source["manifest_sha256"] == sha(Path(parent["checkpoint_path"]).with_name(
                            Path(parent["checkpoint_path"]).stem + "_manifest.json"))
                        and ancestry["source_global_policy_decisions"] == parent["global_policy_decisions"]
                        and ancestry["source_ppo_updates"] == parent["ppo_updates"]
                        and ancestry["source_optimizer_steps"] == parent["optimizer_steps"]
                        and ancestry["source_actor_parameter_sha256"] == parent["actor_parameter_sha256"]
                        and ancestry["source_runtime_contract"] == parent["runtime_contract"],
                        "explicit checkpoint resume ancestry mismatch")
                migration = ancestry.get("resume_migration")
                if migration:
                    require(sha(migration["plan_path"]) == migration["plan_sha256"],
                            "historical migration plan changed")
                    self.capture(migration["plan_path"])
                method = "explicit_resume_ancestry_plus_resolved_launch_and_update_chain"
            require(parent["global_policy_decisions"] < child["global_policy_decisions"] and
                    parent["runtime_contract"]["frozen_A_files"] == child["runtime_contract"]["frozen_A_files"],
                    "non-increasing ancestry or changed frozen A evidence")
            result.append({"child": child, "parent": parent, "run": run, "started": started,
                           "status": status, "parent_resolution": method})
            child = parent
        require(child["checkpoint_sha256"] == self.initial["checkpoint_sha256"],
                "lineage did not terminate at explicitly selected initial weights")
        return list(reversed(result))

    def updates(self, segment):
        child, parent, run = segment["child"], segment["parent"], segment["run"]
        number = child.get("execution_topology", {}).get("num_envs", 1)
        require(number in (1, 8), "unreviewed number of environments")
        rollout = integer(child["runner_config"]["num_steps_per_env"], "rollout length", 1)
        require(rollout == 128, "exporter supports the actual fixed 128-step PPO rollout only")
        batch = rollout * number
        end, prior = child["global_policy_decisions"], parent["global_policy_decisions"]
        previous_actor, index, optimizer = parent["actor_parameter_sha256"], parent["ppo_updates"], 0
        selected, ignored = [], 0
        for _, row in stable_jsonl(run / "optimizer_updates.jsonl"):
            if row["global_policy_decisions"] > end:
                ignored += 1
                continue
            require(row["global_policy_decisions"] == prior + batch and row["ppo_update"] == index + 1,
                    "optimized ranges overlap or have gaps")
            require(row["actor_parameter_sha256_before"] == previous_actor and
                    row["actor_parameters_changed"] == (row["actor_parameter_sha256_before"] !=
                                                        row["actor_parameter_sha256_after"]),
                    "actor update ancestry is broken")
            require(row["finite_nonzero_gradient_observed"] is True and row["optimizer_steps"] > 0,
                    "selected update lacks real optimizer/gradient evidence")
            for key in ("gradient_norm_min", "gradient_norm_max", "kl_mean", "clip_fraction", "entropy", "value_loss"):
                finite(row[key], key)
            self.capture(run / "rollouts" / f"rollout_{row['ppo_update']:06d}.pt")
            selected.append({"start": prior + 1, "end": row["global_policy_decisions"], "row": row})
            prior, index, previous_actor = row["global_policy_decisions"], row["ppo_update"], row["actor_parameter_sha256_after"]
            optimizer += row["optimizer_steps"]
        require(selected and prior == end and index == child["ppo_updates"] and
                optimizer + parent["optimizer_steps"] == child["optimizer_steps"] and
                previous_actor == child["actor_parameter_sha256"] and selected[-1]["row"] == child["last_update"],
                "selected update ledger does not reach checkpoint counters/hash")
        self.capture(run / "optimizer_updates.jsonl")
        return selected, number, ignored


def project_row(row, *, segment, update, number, line):
    info, child = row["applied_audit"], segment["child"]
    contract = child["runtime_contract"]; files = contract["files"]
    raw = vector(row["raw_policy_action_full12"], "PPO raw")
    require(vector(info["raw_policy_action_full12"], "applied raw") == raw, "stored raw/action audit mismatch")
    reward = info.get("reward_breakdown", info.get("reward"))
    require(isinstance(reward, dict) and set(reward["families"]) == set(FAMILIES), "five reward families unavailable")
    native = info.get("actuator_target_effect_audit") or {}
    summary = info.get("actuator_target_effect_audit_summary") or {}
    peer = info.get("external_peer_reset_truncation")
    done = row["terminal"]
    require(type(done) is bool, "PPO done is not boolean")
    # Historical single-env runtime has no peer resets; infer ONLY this documented ABI.
    if peer is None and number == 1:
        peer, physical, interpretation = False, done, "historical_single_env_terminal_ABI"
    else:
        physical = info.get("task_terminated")
        interpretation = "explicit_vector_task_and_peer_flags"
        require(type(peer) is bool and type(physical) is bool, "missing vector terminal partition")
        require(done == (peer or physical) and not (peer and physical), "invalid terminal partition")
    reason = info.get("termination_reason")
    values = {
        "raw": raw, "nominal": vector(info["nominal_action_full12"], "nominal"),
        "projected_residual": vector(info["projected_residual_full12"], "residual"),
        "applied": vector(info["applied_action_full12"], "applied"),
        "actual_drive": vector(info.get("actual_drive_target_full12"), "drive", optional=True),
        "native_last_tick": native_vector(native, "actual_native_targets"),
        "native_counterfactual_last_tick": native_vector(native, "counterfactual_native_targets"),
        "native_delta_last_tick": native_vector(native, "native_target_delta"),
    }
    result = {
        "global_policy_decision": row["global_policy_decision"], "ppo_update": update["row"]["ppo_update"],
        "env_index": info.get("env_index", 0 if number == 1 else NA), "num_envs": number,
        "phase": info["phase_id"], "end_phase": info.get("end_phase_id", NA),
        "episode_decision": info.get("decision_count", NA), "episode_physics_tick": info.get("physics_tick", NA),
        "physics_ticks_this_decision": integer(info["physics_ticks"], "physical tick count", 1),
        "sim_time_s": finite(info["sim_time_s"], "simulation clock"),
        "source_head": contract["source_git_commit"], "runtime_content_sha256": contract["runtime_content_sha256"],
        "reward_config_sha256": files["configs/ppo_semantic_v2/reward_config.yaml"],
        "reward_source_sha256": files["src/wlr50_clean/ppo/semantic_reward.py"],
        "task_spec_sha256": files["configs/ppo_semantic_v2/stage_task_spec.yaml"],
        "supervisor_sha256": files["src/wlr50_clean/ppo/semantic_supervisor.py"],
        "source_run": str(segment["run"]), "source_checkpoint": child["checkpoint_path"],
        "source_checkpoint_sha256": child["checkpoint_sha256"], "source_jsonl_line": line, "optimized": True,
        "old_log_probability": finite(row["old_log_probability"], "old log probability"),
        "old_value": finite(row["old_value"], "old value"),
        "ppo_storage_reward": finite(row["reward"], "PPO reward"),
        "environment_reward": finite(info.get("environment_reward", reward["total"]), "environment reward"),
        "peer_bootstrap_credit": info.get("bootstrap_credit", 0.0 if number == 1 else NA),
        "reward_origin": "PPO_storage_reward_includes_peer_bootstrap" if peer else "physical_environment_reward",
        "ppo_done": done, "physical_task_terminated": physical, "external_peer_reset_truncation": peer,
        "termination_interpretation": interpretation, "termination_reason": reason,
        "task_success": flag(info.get("task_success")), "all_tick_native_audit": flag(summary.get("all_ticks_verified")),
        "verified_native_tick_count": summary.get("verified_tick_count", NA),
        "native_effect_tick_count": summary.get("actual_native_effect_tick_count", NA),
        "own_phase_effect_tick_count": summary.get("own_phase_request_effect_tick_count", NA),
        "native_last_tick_verified": flag(native.get("verified")), "native_last_tick_target_dtype": native.get("target_dtype", NA),
    }
    for name in BASE_FIELDS[BASE_FIELDS.index("no_in_episode_state_writes_verified"):]:
        result[name] = info.get(name, NA)
    for family in FAMILIES:
        result["reward_" + family] = finite(reward["families"][family], family)
    for prefix, vector_values in values.items():
        result.update({f"{prefix}_{channel}": value for channel, value in zip(CHANNELS, vector_values, strict=True)})
    require(result["env_index"] == (row["global_policy_decision"] - update["start"]) % number, "row index/global count mismatch")
    require(math.isclose(result["environment_reward"] + finite(result["peer_bootstrap_credit"], "bootstrap"),
                         result["ppo_storage_reward"], rel_tol=2e-6, abs_tol=2e-6), "reward/bootstrap accounting mismatch")
    return result


def export(project_root, checkpoint, initial_checkpoint, output_dir):
    exporter = Exporter(project_root, checkpoint, initial_checkpoint)
    output = Path(output_dir).resolve()
    export_root = exporter.root / "outputs/ppo_semantic_v2/metrics/training_exports"
    require(output.is_relative_to(export_root.resolve()) and output != export_root.resolve(),
            "use a new exclusive version directory inside semantic training_exports")
    output.mkdir(parents=True, exist_ok=False)
    try:
        segments = exporter.ancestry()
        coverage, ranges, global_seen = {}, [], 0
        with (output / "residual_and_projection_audit.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS); writer.writeheader()
            for segment in segments:
                updates, number, ignored_updates = exporter.updates(segment)
                start, end = updates[0]["start"], updates[-1]["end"]
                cursor, selected, excluded = 0, 0, 0
                for line, row in stable_jsonl(segment["run"] / "residual_and_projection_audit.jsonl"):
                    decision = integer(row["global_policy_decision"], "global decision", 1)
                    if decision > end:
                        excluded += 1
                        continue
                    require(decision >= start and decision == global_seen + 1,
                            "optimized audit has a duplicate, overlap, gap or pre-resume row")
                    while decision > updates[cursor]["end"]:
                        cursor += 1
                    flat = project_row(row, segment=segment, update=updates[cursor], number=number, line=line)
                    writer.writerow(flat); selected += 1; global_seen = decision
                    key = tuple(flat[name] for name in COVERAGE_FIELDS[:5])
                    if key not in coverage:
                        coverage[key] = dict(zip(COVERAGE_FIELDS[:5], key)) | dict.fromkeys(COVERAGE_FIELDS[5:], 0)
                    group = coverage[key]
                    group["optimized_policy_decisions"] += 1
                    group["physical_ticks"] += flat["physics_ticks_this_decision"]
                    group["physical_time_s"] += flat["physics_ticks_this_decision"] / segment["child"]["runtime_contract"]["physics_hz"]
                    group["ppo_done_count"] += flat["ppo_done"]
                    group["physical_task_terminal_count"] += flat["physical_task_terminated"]
                    group["physical_failure_count"] += flat["physical_task_terminated"] and flat["task_success"] is not True
                    group["peer_truncation_count"] += flat["external_peer_reset_truncation"]
                    group["training_success_count"] += flat["task_success"] is True
                    group["all_tick_audit_verified_decisions"] += flat["all_tick_native_audit"] is True
                    group["all_tick_audit_unavailable_decisions"] += flat["all_tick_native_audit"] == NA
                require(selected == end - start + 1, "selected audit length does not equal optimized checkpoint interval")
                exporter.capture(segment["run"] / "residual_and_projection_audit.jsonl")
                ranges.append({"source_run": str(segment["run"]), "source_head": segment["child"]["runtime_contract"]["source_git_commit"],
                    "first_optimized_decision": start, "last_optimized_decision": end, "selected_decisions": selected,
                    "excluded_rows_after_selected_checkpoint": excluded, "excluded_updates_after_checkpoint": ignored_updates,
                    "source_run_lifecycle": segment["status"], "num_envs": number,
                    "checkpoint": segment["child"]["checkpoint_path"], "checkpoint_sha256": segment["child"]["checkpoint_sha256"],
                    "parent_checkpoint": segment["parent"]["checkpoint_path"], "parent_resolution": segment["parent_resolution"],
                    "actual_optimizer_steps": sum(item["row"]["optimizer_steps"] for item in updates),
                    "actual_ppo_updates": len(updates), "optimizer_update_diagnostics": [item["row"] for item in updates]})
        require(global_seen == exporter.latest["global_policy_decisions"], "export total does not reach chosen checkpoint")
        semantic_versions = {key[:4] for key in coverage}
        for version in semantic_versions:
            for phase in (f"P{number:02d}" for number in range(1, 14)):
                key = (*version, phase)
                coverage.setdefault(key, dict(zip(COVERAGE_FIELDS[:5], key)) |
                                    dict.fromkeys(COVERAGE_FIELDS[5:], 0))
        with (output / "phase_coverage.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COVERAGE_FIELDS); writer.writeheader()
            writer.writerows(coverage[key] for key in sorted(coverage))
        # Detect an input mutated after parsing. Old .pt files are only hashed,
        # never loaded/re-saved or described as newly verified Torch roundtrips.
        require(all(sha(path) == item["sha256"] for path, item in exporter.inputs.items()), "source evidence changed during export")
        summary = {"schema": "wlr50_clean.semantic_training_export.v1", "status": "EXPORTED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "latest_checkpoint": record(checkpoint),
            "initial_checkpoint": record(initial_checkpoint), "optimized_policy_decisions": global_seen,
            "cumulative_ppo_updates": exporter.latest["ppo_updates"], "cumulative_optimizer_steps": exporter.latest["optimizer_steps"],
            "stage_requested_decisions": exporter.latest["stage_requested_decisions"],
            "actual_minus_requested_rounding": global_seen - sum(exporter.latest["stage_requested_decisions"].values()),
            "actor_parameter_sha256": exporter.latest["actor_parameter_sha256"],
            "optimizer_state_sha256": exporter.latest["optimizer_state_sha256"], "ancestry_ranges": ranges,
            "source_artifacts": list(exporter.inputs.values()),
            "outputs": [record(output / name) for name in ("residual_and_projection_audit.csv", "phase_coverage.csv")],
            "export_does_not_load_Torch_or_repeat_optimizer_verification": True,
            "checkpoint_bytes_hash_verified_and_update_ledger_tied_to_saved_actor_hashes": True,
            "missing_historical_native_audit_is_unavailable_not_verified": True,
            "phase_coverage_is_training_visitation_not_physical_task_success_or_current_head_evaluation": True,
            "physical_time_is_summed_per_environment_not_wall_time": True,
            "native_targets_are_last_tick_only": True, "native_servo_units": "radians", "native_wheel_units": "radians_per_second",
            "logical_servo_units": "degrees", "logical_wheel_units": "radians_per_second"}
        with (output / "training_manifest_summary.json").open("x", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
        return summary
    except BaseException as exc:
        with (output / "export_failed.json").open("x", encoding="utf-8") as stream:
            json.dump({"status": "FAILED_PARTIAL_OUTPUT_NOT_DELIVERY", "error": str(exc)}, stream, indent=2)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.project_root, args.checkpoint, args.initial_checkpoint, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()),
                      "optimized_policy_decisions": result["optimized_policy_decisions"]}))


if __name__ == "__main__":
    main()

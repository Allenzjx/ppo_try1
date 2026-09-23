"""Dormant sealed-run audit. Standard library only; never imports Torch/Isaac.

Execute once only after the run is sealed. This checks durable metadata,
completed optimizer/advantage journals and terminal summaries. It does not
replay physics, recompute Gaussian likelihood, or deserialize checkpoint tensors.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
HEAD = "d1871df37d6ea909657511d0e43e7435198f6ccd"
BRANCH = "ancestor220544_signed_wheel_v8"
SOURCE_COUNTS = dict(global_policy_decisions=220544, ppo_updates=1688, optimizer_steps=33760)
EXPECTED_RUN = "20260923T1034590877345Z_gd1871df37d6e_75f4fccea74c4c1bb78408b7e919b44b"
SOURCE_NAME = "checkpoint_rr_signed_wheel_v8_ancestor_step_000220544_gd1871df37d6e.pt"
PHASES = tuple(f"P{i:02d}" for i in range(1, 14))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    # Report-only local object digest; not a substitute for production bindings.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode("utf-8")).hexdigest()


def read(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def rows(path):
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            require(line.endswith("\n"), f"unsealed final journal line: {path}:{number}")
            require(bool(line.strip()), f"empty journal line: {path}:{number}")
            yield json.loads(line)


def counts(metadata):
    result = {key: metadata[key] for key in SOURCE_COUNTS}
    require(all(type(value) is int and value >= 0 for value in result.values()), "invalid counters")
    return result


def sidecar(checkpoint):
    return checkpoint.with_name(checkpoint.stem + "_manifest.json")


def verified_pair(checkpoint, manifest_path=None):
    checkpoint = Path(checkpoint).resolve(strict=True)
    manifest_path = Path(manifest_path or sidecar(checkpoint)).resolve(strict=True)
    metadata = read(manifest_path)
    require(Path(metadata["checkpoint_path"]).resolve() == checkpoint, "sidecar checkpoint path differs")
    require(metadata["checkpoint_sha256"] == sha(checkpoint), "checkpoint bytes differ from sidecar")
    require(metadata.get("save_load_round_trip") is True, "official checkpoint roundtrip absent")
    return metadata


def audit(run, source):
    run = Path(run).resolve(strict=True)
    source = Path(source).resolve(strict=True)
    require(run.name == EXPECTED_RUN and run.parent == ROOT / "runs/ppo_rr_capture_then_rl_transfer_v1/train",
            "audit is bound to the requested v8 training run")
    require(source == (OUT / "checkpoints/history" / SOURCE_NAME).resolve(), "wrong ancestor publication source")
    outer = read(run / "run_manifest.json")
    require(outer.get("lifecycle") in ("SUCCEEDED", "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY", "FAILED", "DIAGNOSTIC_FAILURE"),
            "run is not sealed; do not inspect active training")
    trained = read(run / "training_manifest.json")
    require(trained.get("lifecycle") in ("SUCCEEDED", "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY"),
            "complete training manifest is unavailable; do not invent partial update credit")
    require(trained["runtime_contract"]["source_git_commit"] == HEAD, "training runtime HEAD differs")
    require(trained["stage"] == "full_episode" and trained["num_envs"] == 1, "not the declared natural-P01 N1 block")
    require(trained["curriculum_epoch"].get("prefix_request") is None, "unexpected teacher/checkpoint prefix")
    require(trained.get("phase_suffix_curriculum_implemented") is False, "suffix curriculum unexpected")
    initial = verified_pair(source)
    require(counts(initial) == SOURCE_COUNTS, "ancestor source cannot borrow latest 640 decisions")
    require(initial["runtime_contract"] == trained["runtime_contract"], "source/train runtime differs")
    require(initial["policy_contract"]["observation_dimension"] == 410, "wrong observation contract")
    require(initial["rr_capture_transfer_branch_counts"] == dict.fromkeys(SOURCE_COUNTS, 0), "source RR branch is not zero-credit ancestor")
    v8 = initial["rr_signed_wheel_v8_migration"]
    require(v8["source_selection"]["source_role"] == "front_validated_ancestor_control_eval", "wrong published source role")
    require(v8["rr_signed_wheel_v8_factor"]["counter_origin"] == SOURCE_COUNTS, "v8 origin differs")

    completed = {}
    compact_updates = []
    optimizer_total = 0
    for index, update in enumerate(rows(run / "optimizer_updates.jsonl"), 1):
        number = SOURCE_COUNTS["ppo_updates"] + index
        require(update["ppo_update"] == number, "optimizer journal is duplicated or non-contiguous")
        require(update["global_policy_decisions"] == SOURCE_COUNTS["global_policy_decisions"] + 128 * index,
                "completed update does not correspond to a full128-row batch")
        require(update["optimizer_steps"] == 20, "actual minibatch step count differs from 5x4")
        lr = update["optimizer_learning_rate"]
        require(isinstance(lr, (int, float)) and not isinstance(lr, bool) and math.isfinite(lr) and lr > 0,
                "invalid actual learning rate")
        require(number not in completed, "duplicate completed optimizer update")
        completed[number] = update["global_policy_decisions"]
        optimizer_total += update["optimizer_steps"]
        compact_updates.append({key: update.get(key) for key in (
            "ppo_update", "global_policy_decisions", "optimizer_steps", "optimizer_learning_rate",
            "actor_parameters_changed", "finite_nonzero_gradient_observed")})
    require(bool(completed), "no completed optimizer update exists")

    phase_counts = Counter(dict.fromkeys(PHASES, 0))
    matched = set()
    ignored_preupdate = []
    terminal_samples = []
    phase_transitions = 0
    for row in rows(run / "advantage_audit.jsonl"):
        number = row["ppo_update_intended"]
        if number not in completed:
            ignored_preupdate.append(number)
            continue  # A pre-update record alone never earns optimizer/sample credit.
        require(number not in matched, "duplicate completed advantage audit")
        matched.add(number)
        require(row["sample_count"] == 128 and row["num_envs"] == 1 and row["rollout_steps"] == 128,
                "wrong completed rollout shape")
        require(row["last_global_policy_decision"] == completed[number]
                and row["first_global_policy_decision"] == completed[number] - 127, "rollout/global counter binding differs")
        require(row["teacher_prefix_samples_included"] is False, "prefix falsely included in PPO credit")
        require(row["optimizer_update_completed_by_this_record"] is False, "pre-update journal meaning changed")
        by_phase = row["by_request_phase"]
        require(set(by_phase).issubset(PHASES), "unknown requested phase")
        require(sum(item["sample_count"] for item in by_phase.values()) == 128, "phase counts do not sum to rollout")
        for phase, item in by_phase.items():
            require(type(item["sample_count"]) is int and item["sample_count"] >= 0, "invalid phase count")
            phase_counts[phase] += item["sample_count"]
        terminal_samples.extend(row["terminal_samples"])
        phase_transitions += len(row["ordinary_phase_change_samples"])
        require((run / "rollouts" / f"rollout_{number:06d}.pt").is_file(), "sealed rollout artifact missing")
        require((run / "rollouts" / f"update_{number:06d}_likelihood.json").is_file(), "native likelihood audit missing")
    require(matched == set(completed), "completed optimizer update lacks matched advantage journal")
    actual = len(completed) * 128
    require(sum(phase_counts.values()) == actual == trained["actual_policy_decisions"], "actual credited sample counts differ")
    require(trained["ppo_updates_this_run"] == len(completed)
            and trained["optimizer_steps_this_run"] == optimizer_total, "training manifest completed update totals differ")
    require(trained["global_policy_decisions"] == SOURCE_COUNTS["global_policy_decisions"] + actual,
            "training manifest source origin differs")
    require(trained["planned_requested_policy_decisions"] == trained["requested_policy_decisions"]
            + trained["unconsumed_requested_policy_decisions"], "planned/consumed budget does not reconcile")

    branch = (OUT / "branches" / BRANCH).resolve()
    pointer_path = branch / "checkpoints/checkpoint_last_pointer.json"
    pointer = read(pointer_path)
    target = Path(pointer["checkpoint"]).resolve(strict=True)
    target_manifest = Path(pointer["manifest"]).resolve(strict=True)
    require(target.parent == branch / "checkpoints/history", "checkpoint escaped isolated branch")
    require(target_manifest == sidecar(target), "pointer sidecar is not its immutable pair")
    require(pointer["checkpoint_sha256"] == sha(target) and pointer["manifest_sha256"] == sha(target_manifest),
            "branch pointer hash bindings differ")
    require(sha(branch / "checkpoints/checkpoint_last.pt") == pointer["checkpoint_sha256"], "branch convenience copy differs")
    require(sha(branch / "checkpoints/resume_state.json") == pointer["manifest_sha256"], "branch resume-state copy differs")
    final = verified_pair(target, target_manifest)
    require(Path(trained["checkpoints"][-1]["checkpoint"]).resolve() == target, "final train checkpoint and pointer disagree")
    expected_counts = {key: SOURCE_COUNTS[key] + added for key, added in zip(SOURCE_COUNTS, (actual, len(completed), optimizer_total))}
    require(counts(final) == expected_counts, "target counters do not come from the explicit ancestor plus actual updates")
    route = final["checkpoint_output_routing"]
    require(route["branch"] == BRANCH and Path(route["output_root"]).resolve() == branch
            and route["main_latest_pointer_promotion"] is False, "isolated branch metadata differs")
    require(route == trained["checkpoint_output_routing"], "manifest/checkpoint output routing differs")
    require(final["runtime_contract"] == initial["runtime_contract"] and final["policy_contract"] == initial["policy_contract"],
            "runtime or policy contract changed inside block")
    preserved = {}
    for key, value in initial.items():
        if ((key.endswith("_migration") and key != "resume_migration")
                or (key.endswith("_branch") and isinstance(value, dict) and "counter_origin" in value)):
            require(final.get(key) == value, "ancestry/origin/AUX object changed: " + key)
            preserved[key] = digest(value)
            if key.endswith("_branch"):
                expected = {counter: final[counter] - value["counter_origin"][counter] for counter in SOURCE_COUNTS}
                require(final.get(key + "_counts") == expected, "branch counters do not reconcile: " + key)
    require("rr_signed_wheel_v8_migration" in preserved and "rr_signed_contact_v7_migration" in preserved,
            "new or historical migration receipt missing")
    require(final["normalizer_state_sha256"] == initial["normalizer_state_sha256"], "Identity normalizer state changed")
    require(final["optimizer_learning_rate"] == compact_updates[-1]["optimizer_learning_rate"], "effective saved LR differs")
    require(bool(final.get("training_rng_state")), "full RNG metadata missing")

    episodes = []
    for summary in rows(run / "completed_episodes.jsonl"):
        info = summary["terminal_info"]
        require(summary["episode_index"] == len(episodes), "terminal episodes are not contiguous")
        require(info.get("terminal_bootstrap_allowed") is False, "real terminal was bootstrapped")
        require(summary["task_success"] == (summary["termination_reason"] == "SUCCESS"), "terminal success label contradicts task outcome")
        episodes.append({key: summary.get(key) for key in (
            "episode_index", "policy_decisions", "termination_reason", "task_success", "full_task_success", "duration_s")}
            | {"request_phase": info.get("phase_id"), "end_phase": info.get("end_phase_id"),
               "physics_tick": info.get("physics_tick")})
    require(len(episodes) == len(terminal_samples), "completed episodes differ from credited terminal transitions")
    require(len(episodes) == trained["telemetry"]["completed_episode_count"], "terminal summary count differs")
    partial = actual - sum(row["policy_decisions"] for row in episodes)
    require(partial >= 0, "terminal episodes exceed credited decisions")
    telemetry_phase = trained["telemetry"].get("core", {}).get("phase_decisions")
    if telemetry_phase is not None:
        require(Counter(telemetry_phase) == phase_counts, "independent core phase telemetry differs")

    # Report the separately retained parent pointer; no main checkpoint is written.
    main_pointer_path = OUT / "checkpoints/checkpoint_last_pointer.json"
    main = None
    if main_pointer_path.exists():
        main_pointer = read(main_pointer_path)
        main_metadata = read(main_pointer["manifest"])
        require(counts(main_metadata) == dict(global_policy_decisions=221184, ppo_updates=1693, optimizer_steps=33860),
                "separate latest-learned parent pointer changed unexpectedly")
        main = {"pointer": str(main_pointer_path), "checkpoint": main_pointer["checkpoint"],
                "checkpoint_sha256": main_pointer["checkpoint_sha256"], "counts": counts(main_metadata)}

    return {"schema": "wlr50_clean.v8_sealed_branch_training_audit.v1", "run": str(run),
        "outer_lifecycle": outer["lifecycle"], "outer_error": outer.get("error"),
        "training_lifecycle": trained["lifecycle"], "source_checkpoint": str(source),
        "source_checkpoint_sha256": initial["checkpoint_sha256"], "source_counts": SOURCE_COUNTS,
        "target_checkpoint": str(target), "target_checkpoint_sha256": pointer["checkpoint_sha256"],
        "target_counts": counts(final), "added": dict(policy_decisions=actual, ppo_updates=len(completed), optimizer_steps=optimizer_total),
        "planned_requested_policy_decisions": trained["planned_requested_policy_decisions"],
        "unconsumed_requested_policy_decisions": trained["unconsumed_requested_policy_decisions"],
        "phase_input_counts": dict(phase_counts), "ordinary_phase_transitions": phase_transitions,
        "ignored_uncompleted_preupdate_records": ignored_preupdate,
        "teacher_prefix_credit": 0, "completed_episodes": episodes,
        "last_sampling_partial": {"policy_decisions": partial, "is_task_failure_or_success": False,
                                  "present": partial > 0, "final_duration_not_inferred": True},
        "updates": compact_updates, "preserved_ancestry_object_digests": preserved,
        "official_save_reload_verified_in_sidecar": True, "checkpoint_bytes_match_sidecar": True,
        "actor_metadata_hash_changed": final["actor_parameter_sha256"] != initial["actor_parameter_sha256"],
        "optimizer_metadata_hash_changed": final["optimizer_state_sha256"] != initial["optimizer_state_sha256"],
        "full_RNG_metadata_present": True, "RNG_metadata_advanced": digest(final["training_rng_state"]) != digest(initial["training_rng_state"]),
        "normalizer_unchanged": True, "output_routing": route, "separate_parent_latest": main,
        "borrowed_latest_640_decisions": 0,
        "scope": "sealed metadata/journal audit; native likelihood artifacts exist, not independently recomputed; no tensor or physics replay",
        "training_lifecycle_success_is_not_physical_task_success": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=OUT / "checkpoints/history" / SOURCE_NAME)
    parser.add_argument("--report-name", default="v8_branch_training_audit")
    args = parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.report_name) is not None, "report name must be a basename")
    result = audit(args.run, args.source)
    json_path = OUT / (args.report_name + ".json")
    md_path = OUT / (args.report_name + ".md")
    require(not json_path.exists() and not md_path.exists(), "do not overwrite an existing audit")
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    lines = ["# V8 actual ancestor-branch training audit", "",
        f"Training lifecycle: {result['training_lifecycle']}; outer lifecycle: {result['outer_lifecycle']}.",
        f"Actual added: {result['added']}. Unconsumed requested decisions: {result['unconsumed_requested_policy_decisions']}.",
        f"Final counters: {result['target_counts']}; checkpoint: {result['target_checkpoint']}.",
        "", "| Requested input phase | Credited samples |", "| --- | ---: |"]
    lines.extend(f"| {phase} | {count} |" for phase, count in result["phase_input_counts"].items())
    lines += ["", f"Completed episodes: {len(result['completed_episodes'])}; trailing sampling partial decisions: {result['last_sampling_partial']['policy_decisions']}.",
        "A sampling partial is not a task outcome. Full-episode natural P01; prefix credit zero. No latest-branch640 credit borrowed.",
        "", "Terminal outcomes:"]
    lines.extend(f"- Episode {row['episode_index']}: {row['duration_s']}s, {row['policy_decisions']} decisions, {row['termination_reason']}, end phase {row['end_phase']}."
                 for row in result["completed_episodes"])
    lines += ["", "All inherited migration/origin/AUX objects retained exactly; Identity unchanged. Complete RNG metadata and official save/reload evidence retained. Actual LR by update is in JSON.",
        "This bounded audit verified checkpoint bytes/sidecar/pointer and completed journal counts. It did not independently recompute Gaussian likelihood or replay physics."]
    with md_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(json.dumps({"report": str(json_path), "markdown": str(md_path), "added": result["added"]}))


if __name__ == "__main__":
    main()

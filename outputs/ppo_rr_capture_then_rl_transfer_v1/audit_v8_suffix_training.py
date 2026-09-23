"""Dormant, one-pass sealed P07 nominal-prefix suffix audit; no Torch/Isaac.

The earlier natural-P01512 decisions are inherited, not repeated or relabeled.
Prefix physical evidence is initialization only, never PPO or full-policy success.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from audit_v8_branch_training import OUT, ROOT, HEAD, BRANCH, PHASES, require, read, rows, sha, digest, counts, sidecar, verified_pair

RUN_NAME = "20260923T1045146351727Z_gd1871df37d6e_0a956b4862894edb85aa7a239aa27f0f"
ANCESTOR = dict(global_policy_decisions=220544, ppo_updates=1688, optimizer_steps=33760)
SOURCE = dict(global_policy_decisions=221056, ppo_updates=1692, optimizer_steps=33840)
SOURCE_SHA = "0c614de0d3743824a5775b411eb581fd3458314dffe2e3fe60ed425f1b1f048c"
BRANCH_ROOT = OUT / "branches" / BRANCH
SOURCE_PATH = BRANCH_ROOT / "checkpoints/history/checkpoint_step_000221056.pt"


def check_prefix(run, contract):
    totals = Counter(); attempts = []; active = None
    expected_provenance = None
    for row in rows(run / "prefix_evidence.jsonl"):
        require(row.get("policy_credit") is False, "prefix record claims policy credit")
        kind = row["kind"]; totals[kind] += 1
        if kind == "checkpoint_prefix_bootstrap":
            require(row["physics_tick"] == 0, "prefix bootstrap is not natural t0")
        elif kind == "checkpoint_prefix_start":
            require(active is None, "overlapping prefix attempts")
            request = row["request"]
            require(request["source"] == "successful_nominal" and request["target_phase"] == "P07"
                    and request["teacher_offset_decisions"] == 0, "unexpected prefix curriculum")
            provenance = row["prefix_policy_provenance"]
            require(provenance["source"] == "successful_nominal" and provenance["policy_credit"] is False
                    and provenance["raw_action_full12"] == [0.0] * 12, "prefix is not explicit zero-residual nominal")
            require(provenance["runtime_content_sha256"] == contract["runtime_content_sha256"], "nominal prefix runtime binding differs")
            selected = contract["selected_configuration"]
            require(provenance["execution_profile_sha256"] == selected["execution_profile.yaml"]["sha256"]
                    and provenance["stage_task_spec_sha256"] == selected["stage_task_spec.yaml"]["sha256"], "prefix config binding differs")
            if expected_provenance is None: expected_provenance = provenance
            require(provenance == expected_provenance, "prefix provenance changed during training")
            active = {"decisions": 0, "ticks": 0, "last_phase": "P01"}
        elif kind == "checkpoint_prefix_decision":
            require(active is not None and "result" not in active, "prefix action outside current initialization")
            ticks = row["physics_ticks"]
            require(type(ticks) is int and 1 <= ticks <= 8 and row["physics_tick"] == active["ticks"] + ticks,
                    "prefix is not a continuous physical trajectory from t0")
            require(row["phase_id"] == active["last_phase"], "prefix phase sequence jumps")
            require(row["raw_policy_action_full12"] == [0.0] * 12, "nonzero policy action credited as nominal prefix")
            native = row["actuator_target_effect_audit_summary"]
            require(native["all_ticks_verified"] is True and native["verified_tick_count"] == ticks,
                    "prefix native execution proof missing")
            require(row["no_in_episode_state_writes_verified"] is True and row["full_task_success"] is False,
                    "prefix snapshot/write or full-policy success claim")
            active.update(decisions=active["decisions"] + 1, ticks=row["physics_tick"], last_phase=row["end_phase_id"])
            totals["physical_prefix_ticks"] += ticks
        elif kind == "checkpoint_prefix_result":
            require(active is not None and "result" not in active, "unpaired prefix result")
            require(row["prefix_decisions"] == active["decisions"] and row["prefix_physics_ticks"] == active["ticks"],
                    "prefix result counters differ from its actual actions")
            active["result"] = {key: row[key] for key in ("accepted", "miss", "prefix_decisions", "prefix_physics_ticks")}
        elif kind == "policy_credit_start":
            require(active is not None and "result" in active, "learner starts without finished prefix result")
            start = row["start"]; accepted = active["result"]["accepted"]
            require(start["requested_phase"] == "P07" and start["prefix_policy_provenance"] == expected_provenance,
                    "learner prefix handoff provenance differs")
            if accepted:
                require(start["actual_phase"] == "P07" and start["physics_tick"] == active["ticks"]
                        and start["from_P01_current_policy"] is False and start["requested_phase_still_active_at_credit"] is True,
                        "accepted nominal initialization is not a real continuous P07 handoff")
            else:
                require(start["actual_phase"] == "P01" and start["physics_tick"] == 0
                        and start["from_P01_current_policy"] is True, "missed prefix lacks truthful fresh-P01 fallback")
            attempts.append({**active["result"], "credit_start": {key: start.get(key) for key in (
                "mode", "actual_phase", "physics_tick", "sim_time_s", "from_P01_current_policy")}})
            active = None
        else:
            raise ValueError("unknown prefix evidence kind: " + kind)
    require(active is None, "prefix initialization was not sealed")
    return {"attempts": attempts, "actual_prefix_decisions": totals["checkpoint_prefix_decision"],
            "actual_prefix_physics_ticks": totals["physical_prefix_ticks"], "PPO_credit": 0,
            "nominal_provenance_digest": digest(expected_provenance), "record_counts": dict(totals)}


def audit(run):
    run = run.resolve(strict=True)
    require(run.name == RUN_NAME and run.parent == ROOT / "runs/ppo_rr_capture_then_rl_transfer_v1/train", "wrong suffix run")
    outer = read(run / "run_manifest.json")
    require(outer["lifecycle"] in ("SUCCEEDED", "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY", "FAILED", "DIAGNOSTIC_FAILURE"), "run still active")
    trained = read(run / "training_manifest.json")
    require(trained["lifecycle"] in ("SUCCEEDED", "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY"), "complete training manifest absent")
    initial = verified_pair(SOURCE_PATH)
    require(initial["checkpoint_sha256"] == SOURCE_SHA and counts(initial) == SOURCE, "wrong actual saved221056 source")
    contract = trained["runtime_contract"]
    require(contract == initial["runtime_contract"] and contract["source_git_commit"] == HEAD, "runtime changed")
    require(trained["stage"] == "phase_suffix" and trained["num_envs"] == 1, "not P07 N1 suffix training")
    request = trained["curriculum_epoch"]["prefix_request"]
    require(request["source"] == "successful_nominal" and request["target_phase"] == "P07"
            and request["teacher_offset_decisions"] == 0 and trained["phase_suffix_curriculum_implemented"] is True,
            "suffix manifest curriculum differs")
    require(initial["rr_capture_transfer_branch"]["counter_origin"] == ANCESTOR
            and initial["rr_capture_transfer_branch_counts"] == {k: SOURCE[k] - ANCESTOR[k] for k in ANCESTOR},
            "source did not retain the real earlier512/4/80")

    updates = []
    for index, row in enumerate(rows(run / "optimizer_updates.jsonl"), 1):
        require(row["ppo_update"] == SOURCE["ppo_updates"] + index
                and row["global_policy_decisions"] == SOURCE["global_policy_decisions"] + index * 128
                and row["optimizer_steps"] == 20, "non-contiguous actual128-row/20Adam update")
        require(math.isfinite(row["optimizer_learning_rate"]) and row["optimizer_learning_rate"] > 0, "invalid LR")
        updates.append({k: row.get(k) for k in ("ppo_update", "global_policy_decisions", "optimizer_steps", "optimizer_learning_rate", "actor_parameters_changed")})
    require(bool(updates), "no completed update")
    completed = {row["ppo_update"]: row["global_policy_decisions"] for row in updates}
    matched = set(); ignored = []; phases = Counter(dict.fromkeys(PHASES, 0)); terminals = []
    for row in rows(run / "advantage_audit.jsonl"):
        number = row["ppo_update_intended"]
        if number not in completed: ignored.append(number); continue
        require(number not in matched and row["sample_count"] == 128 and row["num_envs"] == 1,
                "duplicated or wrong-size completed advantage record")
        matched.add(number)
        require(row["last_global_policy_decision"] == completed[number]
                and row["first_global_policy_decision"] == completed[number] - 127
                and row["teacher_prefix_samples_included"] is False, "prefix data/counters entered learner storage")
        require(sum(v["sample_count"] for v in row["by_request_phase"].values()) == 128, "phase counts do not sum")
        for phase, value in row["by_request_phase"].items():
            require(phase in PHASES and type(value["sample_count"]) is int and value["sample_count"] >= 0, "invalid phase count")
            phases[phase] += value["sample_count"]
        terminals.extend(row["terminal_samples"])
        for name in (f"rollout_{number:06d}.pt", f"update_{number:06d}_likelihood.json"):
            require((run / "rollouts" / name).is_file(), "sealed rollout/native likelihood artifact missing")
    require(matched == set(completed), "completed update missing advantage record")
    added = dict(global_policy_decisions=128 * len(updates), ppo_updates=len(updates), optimizer_steps=sum(v["optimizer_steps"] for v in updates))
    require(sum(phases.values()) == trained["actual_policy_decisions"] == added["global_policy_decisions"], "learner actual count mismatch")
    require(trained["ppo_updates_this_run"] == added["ppo_updates"] and trained["optimizer_steps_this_run"] == added["optimizer_steps"], "manifest update totals mismatch")
    require(trained["planned_requested_policy_decisions"] == trained["requested_policy_decisions"] + trained["unconsumed_requested_policy_decisions"], "planned/consumed budget mismatch")
    prefix = check_prefix(run, contract)
    credit_core = trained["telemetry"]["core"]
    require(credit_core["prefix_behavior_decisions"] == prefix["actual_prefix_decisions"]
            and credit_core["prefix_physics_ticks"] == prefix["actual_prefix_physics_ticks"]
            and credit_core["decisions"] == added["global_policy_decisions"]
            and Counter(credit_core["phase_decisions"]) == phases, "prefix/learner telemetry does not separate actual credit")

    pointer = read(BRANCH_ROOT / "checkpoints/checkpoint_last_pointer.json")
    target = Path(pointer["checkpoint"]).resolve(strict=True)
    manifest = Path(pointer["manifest"]).resolve(strict=True)
    require(target.parent == (BRANCH_ROOT / "checkpoints/history").resolve() and manifest == sidecar(target), "target escaped branch")
    final = verified_pair(target, manifest)
    require(pointer["checkpoint_sha256"] == final["checkpoint_sha256"] and pointer["manifest_sha256"] == sha(manifest), "pointer binding differs")
    require(sha(BRANCH_ROOT / "checkpoints/checkpoint_last.pt") == pointer["checkpoint_sha256"]
            and sha(BRANCH_ROOT / "checkpoints/resume_state.json") == pointer["manifest_sha256"], "branch convenience copies differ")
    require(Path(trained["checkpoints"][-1]["checkpoint"]).resolve() == target, "training final pair differs")
    require(counts(final) == {k: SOURCE[k] + added[k] for k in SOURCE}, "target borrowed non-ancestor credit")
    require(final["checkpoint_output_routing"] == initial["checkpoint_output_routing"] == trained["checkpoint_output_routing"], "same-branch routing changed")
    require(final["runtime_contract"] == contract and final["policy_contract"] == initial["policy_contract"], "policy/runtime changed")
    require(final["normalizer_state_sha256"] == initial["normalizer_state_sha256"]
            and final["optimizer_learning_rate"] == updates[-1]["optimizer_learning_rate"]
            and bool(final["training_rng_state"]), "Identity/LR/RNG metadata invalid")
    preserved = {}
    for key, value in initial.items():
        if ((key.endswith("_migration") and key != "resume_migration")
                or (key.endswith("_branch") and isinstance(value, dict) and "counter_origin" in value)):
            require(final.get(key) == value, "changed ancestry/AUX: " + key); preserved[key] = digest(value)
            if key.endswith("_branch"):
                require(final[key + "_counts"] == {k: final[k] - value["counter_origin"][k] for k in SOURCE}, "branch credit differs")

    episodes = []
    for episode in rows(run / "completed_episodes.jsonl"):
        info = episode["terminal_info"]; start = info["curriculum_start"]
        require(episode["episode_index"] == len(episodes) and info["terminal_bootstrap_allowed"] is False, "invalid terminal journal")
        require(episode["task_success"] == (episode["termination_reason"] == "SUCCESS"), "task outcome mislabeled")
        require(episode["full_task_success"] == bool(episode["task_success"] and start["from_P01_current_policy"]), "suffix mislabeled full PPO success")
        episodes.append({k: episode[k] for k in ("episode_index", "policy_decisions", "duration_s", "termination_reason", "task_success", "full_task_success")}
            | {"end_phase": info.get("end_phase_id"), "scope": info.get("task_result_scope"), "credit_start": start})
    require(len(episodes) == len(terminals) == trained["telemetry"]["completed_episode_count"], "terminal counts differ")
    partial = added["global_policy_decisions"] - sum(row["policy_decisions"] for row in episodes)
    require(partial >= 0, "terminal learner decisions exceed actual count")
    parent = read(OUT / "checkpoints/checkpoint_last_pointer.json")
    parent_counts = counts(read(parent["manifest"]))
    require(parent_counts == dict(global_policy_decisions=221184, ppo_updates=1693, optimizer_steps=33860), "main latest pointer changed")
    return {"schema": "wlr50_clean.v8_sealed_nominal_P07_suffix_audit.v1", "run": str(run),
        "outer_lifecycle": outer["lifecycle"], "training_lifecycle": trained["lifecycle"],
        "source_checkpoint": str(SOURCE_PATH), "source_sha256": SOURCE_SHA, "source_counts": SOURCE,
        "target_checkpoint": str(target), "target_sha256": final["checkpoint_sha256"], "target_counts": counts(final),
        "actual_added_this_suffix": added, "ancestor_actual_total": {k: final[k] - ANCESTOR[k] for k in ANCESTOR},
        "earlier_natural_P01_block": {k: SOURCE[k] - ANCESTOR[k] for k in ANCESTOR},
        "planned_suffix_decisions": trained["planned_requested_policy_decisions"],
        "unconsumed_suffix_decisions": trained["unconsumed_requested_policy_decisions"],
        "phase_input_counts": dict(phases), "prefix": prefix, "ignored_uncompleted_preupdate_records": ignored,
        "completed_episodes": episodes, "sampling_partial_learner_decisions": partial,
        "partial_is_not_failure_or_success": True, "suffix_success_is_not_full_P01_PPO_success": True,
        "updates": updates, "ancestry_and_AUX_object_digests": preserved, "official_roundtrip": True,
        "normalizer_unchanged": True, "full_RNG_metadata_present": True,
        "actor_metadata_hash_changed": final["actor_parameter_sha256"] != initial["actor_parameter_sha256"],
        "optimizer_metadata_hash_changed": final["optimizer_state_sha256"] != initial["optimizer_state_sha256"],
        "main_latest_retained_counts": parent_counts, "borrowed_main_latest640_credit": 0,
        "scope": "sealed metadata/journal audit only; no tensor/physics replay or independently recomputed likelihood"}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report-name", default="v8_suffix_training_audit")
    args = parser.parse_args(); require(re.fullmatch(r"[A-Za-z0-9_-]+", args.report_name), "unsafe report basename")
    result = audit(args.run)
    paths = [OUT / (args.report_name + suffix) for suffix in (".json", ".md")]
    require(not any(path.exists() for path in paths), "do not overwrite prior audit")
    with paths[0].open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2, allow_nan=False)
    lines = ["# V8 actual P07 nominal-prefix suffix training", "",
        f"Lifecycle: {result['training_lifecycle']} (outer {result['outer_lifecycle']}).",
        f"Actual suffix additions: {result['actual_added_this_suffix']}; unconsumed requested decisions: {result['unconsumed_suffix_decisions']}.",
        f"Actual ancestor total including the earlier512: {result['ancestor_actual_total']}.",
        f"Physical nominal prefix: {result['prefix']['actual_prefix_decisions']} decisions / {result['prefix']['actual_prefix_physics_ticks']} ticks; PPO credit zero.",
        "", "| Learner requested phase | Samples |", "| --- | ---: |"]
    lines.extend(f"| {phase} | {count} |" for phase, count in result["phase_input_counts"].items())
    lines += ["", f"Completed learner episodes: {len(result['completed_episodes'])}; trailing sampling partial: {result['sampling_partial_learner_decisions']} learner decisions."]
    lines.extend(f"- Episode {e['episode_index']}: {e['duration_s']}s physical episode, {e['policy_decisions']} learner decisions, {e['termination_reason']}, end {e['end_phase']}, scope {e['scope']}." for e in result["completed_episodes"])
    lines += ["", "Nominal-prefix/suffix completion is not full natural-P01 PPO success. A sampling partial is not a task failure/success. All origins/migrations/AUX retained; main latest640 credit was not borrowed.",
              f"Verified final branch checkpoint: {result['target_checkpoint']}",
              "This bounded audit checks completed journals and checkpoint/sidecar/pointer metadata; it does not replay physics or independently recompute Gaussian likelihood."]
    with paths[1].open("x", encoding="utf-8") as stream: stream.write("\n".join(lines) + "\n")
    print(json.dumps({"report": str(paths[0]), "actual_added": result["actual_added_this_suffix"], "ancestor_actual_total": result["ancestor_actual_total"]}))


if __name__ == "__main__":
    main()

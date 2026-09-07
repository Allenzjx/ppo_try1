"""UNEXECUTED ARTIFACT-ONLY DRAFT: neutral A-outcome / C-success comparison.

Not imported by the runtime, not a training gate, and never starts Isaac.
Use only after live simulation exits. See render_outcome_comparison_README.md.
This first bounded helper supports a terminal A diagnostic and a validated
C success; it deliberately does not add a general failed-C publisher.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import stat
import subprocess

from wlr50_clean.ppo.semantic_cli import PROJECT_ROOT, runtime_contract
from wlr50_clean.ppo.semantic_video import (
    CAMERA, FPS, HZ, MAX_FRAMES, POST_TICKS, PRE_TICKS, ROLES, STRIDE, ZERO12,
    file_record, inside, require, validate_existing_settle_evidence,
    validate_semantic_video_source, validate_video_configuration,
)
from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator
from wlr50_clean.ppo.semantic_training import verified_native_effect
from wlr50_clean.evaluation.video_timeline import decode_frame_timeline, load_viewport_frame_ledger
from wlr50_clean.infrastructure.video_capture import find_ffmpeg, validate_mp4

DRAFT_STATUS = "UNEXECUTED_UNTESTED_ARTIFACT_HELPER"
WRITE_KEYS = ("in_episode_root_pose_writes", "in_episode_root_velocity_writes",
              "in_episode_force_or_impulse_writes", "in_episode_gravity_writes")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def managed_source(directory, role):
    root = Path(directory).resolve(strict=True)
    require(root.name == "source" and root.is_relative_to(PROJECT_ROOT / "runs/ppo_semantic_v3/video_eval"),
            "source must be a new managed v3 video capture")
    source = read_json(root / "semantic_video_source_manifest.json")
    run = read_json(root.parent / "run_manifest.json")
    require(run["result"] == source and run["runtime_contract"] == source["runtime_contract"],
            "capture/run manifest binding differs")
    args = run["arguments"]
    require(source["schema"] == "wlr50_clean.semantic_video_source.v1"
            and source["semantic_version"] == "v3" and source["role"] == role
            and source["mode"] == ROLES[role] and source["from_phase"] == "P01"
            and source["seed"] == 4001 and source["episode_count"] == 1
            and source["fresh_process_single_episode"] is True
            and source["optimizer_updates"] == 0 and source["improved_claim"] is False,
            "not a single fresh P01 v3 capture with neutral provenance")
    require(args["command"] == "eval" and args["semantic_version"] == "v3"
            and args["mode"] == ROLES[role] and args["num_envs"] == 1
            and args["from_phase"] == "P01" and args["teacher_offset_decisions"] == 0
            and args["seed"] == 4001 and args["headless"] is False
            and args["max_decisions"] == 3000 and args["decisions"] is None,
            "capture arguments differ from common-condition full episode")
    require(source["camera"] == {**CAMERA, "resolution": [1280, 720], "fps": FPS}
            and all(source[key] is False for key in ("stitched", "frame_interpolation", "speed_modified")),
            "camera or source timeline changed")
    reset = source["reset_evidence"]
    require(reset["reset_count"] == 1 and reset["reset_options"] == {}
            and reset["training_phase_snapshot"] is None,
            "video is not the first natural reset")
    configs = validate_video_configuration(source)
    contract = source["runtime_contract"]
    require(runtime_contract(expected_head=contract["source_git_commit"], semantic_version="v3") == contract,
            "current evaluator/frozen environment/runtime differs from capture")
    paths = {name: inside(root, record) for name, record in source["artifacts"].items()}
    roll = list(rows(paths["physical_video_roll_ticks.jsonl"]))
    pre = [row for row in roll if row["kind"] == "pre_action"]
    validate_existing_settle_evidence(source, pre)  # 180 existing settle, zero added ticks.
    return root, source, run, configs, paths, roll


def verify_source_timing(source, paths):
    endpoint = source["episode_physics_ticks"]
    post = source["performed_post_success_ticks"]
    require(type(endpoint) is int and 0 < endpoint <= 200 * HZ
            and type(post) is int and 0 <= post <= POST_TICKS,
            "invalid physical endpoint or context length")
    count = 1 + (PRE_TICKS + endpoint + post) // STRIDE
    require(2 <= count <= MAX_FRAMES, "complete source exceeds 200 seconds")
    ledger = load_viewport_frame_ledger(paths["viewport_frame_ledger.jsonl"])
    decoded = decode_frame_timeline(paths["actual_viewport_video.mp4"])
    require(len(ledger) == len(decoded) == count, "full source lost/added frames")
    for index, (record, frame) in enumerate(zip(ledger, decoded)):
        require(record.frame_index == frame.frame_index == index
                and record.sim_step == index * STRIDE
                and math.isclose(record.sim_time_s, index / FPS, abs_tol=1e-10)
                and math.isclose(frame.pts_s, index / FPS, abs_tol=1e-5),
                "source frame/PTS/physical clock gap or speed modification")
    # Last image may precede a partial terminal tick by at most seven ticks;
    # the ledger retains that truth instead of relabelling it as the endpoint.
    require(0 <= PRE_TICKS + endpoint + post - ledger[-1].sim_step < STRIDE,
            "source recording ended before the actual terminal interval")
    technical = validate_mp4(paths["actual_viewport_video.mp4"], expected_fps=FPS,
        expected_frame_count=count, maximum_duration_s=200., require_sane_container_duration=False)
    require(technical["valid"] is True, "source decode/H264/yuv420p validation failed")
    return count, technical


def verify_a_diagnostic(source, configs, paths, roll):
    require(source["success_candidate"] is False and source["diagnostic_only"] is True
            and source["performed_post_success_ticks"] == 0
            and source["source_acceptance_error"] ==
                "SemanticVideoError: episode did not meet common physical task",
            "A diagnostic is a recording/interface error or unsupported post-roll failure")
    require(len(roll) == PRE_TICKS, "failed A must not contain fabricated success post-roll")
    endpoint = source["episode_physics_ticks"]
    evaluator = TaskEvaluator(configs["task_spec_path"])
    count = 0
    for count, raw in enumerate(rows(paths["physical_observations.jsonl"]), 1):
        require(raw["physics_tick"] == count - 1, "A physical sensor stream gap")
        result = evaluator.observe(raw)
        require(not result["success"], "A diagnostic contains an unacknowledged task success")
        require(count - 1 == endpoint or result["termination_reason"] is None,
                "A source continued after an earlier physical terminal")
    require(count == endpoint + 1 and result == source["physical_episode"]["physical_task_evaluation"]
            and source["physical_episode"]["task_success"] is False,
            "A physical replay disagrees with its recorded result")
    count = 0
    for count, row in enumerate(rows(paths["native_tick_audit.jsonl"]), 1):
        audit = row["native_audit"]
        require(row["episode_physics_tick"] == count
                and tuple(audit["raw_policy_action_full12"]) == ZERO12
                and tuple(row["projected_residual_full12"]) == ZERO12
                and all(type(row[key]) is int and row[key] == 0 for key in WRITE_KEYS),
                "A native audit is nonzero, discontinuous, or contains a state write")
        verified_native_effect({"actuator_target_effect_audit": audit}, ZERO12)
    require(count == endpoint, "A native audit endpoint mismatch")
    decisions = list(rows(paths["video_policy_decisions.jsonl"]))
    require(len(decisions) == source["issued_policy_decisions"] and decisions,
            "A decision count differs")
    next_tick, completed = 0, 0
    for index, row in enumerate(decisions, 1):
        require(row["decision"] == index and row["start_tick"] == next_tick
                and 1 <= row["physics_ticks"] <= STRIDE
                and row["end_tick"] - row["start_tick"] == row["physics_ticks"]
                and tuple(row["raw_policy_action_full12"]) == ZERO12,
                "A decision ledger gap, clipped interval, or nonzero action")
        if row["environment_step_returned"]:
            completed += 1
        else:
            require(index == len(decisions) and row["stop_reason"] == result["termination_reason"],
                    "A interrupted interval is not the physical terminal")
        next_tick = row["end_tick"]
    last = decisions[-1]
    require(next_tick == endpoint and completed == source["completed_environment_steps"],
            "A recorded decisions do not reach its endpoint")
    partial = 0 if last["environment_step_returned"] else last["physics_ticks"]
    require(partial == source["interrupted_final_decision_ticks"], "A partial tick was hidden")
    if result["termination_reason"] is not None:
        physical_outcomes = {
            "TASK_FAILURE_BODY_COLLISION": "PHYSICAL FAILURE",
            "TASK_FAILURE_WHEEL_ONLY_CLIMB": "PHYSICAL FAILURE",
            "SAFETY_ABORT": "SAFETY ABORT",
            "INCOMPLETE_CONTROLLER_BLOCKED": "INCOMPLETE",
        }
        require(result["termination_reason"] in physical_outcomes,
                "unsupported physical result must not be labelled as a task failure")
        return physical_outcomes[result["termination_reason"]], result["termination_reason"]
    info = last.get("step_info", {})
    require(last["environment_step_returned"] and info.get("termination_reason")
            and (info.get("terminated") is True or info.get("truncated") is True),
            "A incomplete source has no genuine controller terminal; arbitrary crop is forbidden")
    return "INCOMPLETE", info["termination_reason"]


def verify_c_checkpoint(source, run):
    proof = source["checkpoint_load_provenance"]
    require(proof["checkpoint_loaded_and_verified"] is True
            and proof["official_load_semantic_checkpoint"] is True
            and proof["optimizer_updates"] == 0 and proof["video_seed"] == 4001,
            "C lacks deterministic saved-policy evaluation provenance")
    record = proof["source"]
    checkpoint = Path(record["checkpoint"]).resolve(strict=True)
    metadata_path = Path(record["manifest"]).resolve(strict=True)
    require(checkpoint.is_relative_to(PROJECT_ROOT / "outputs/ppo_semantic_v3/checkpoints/history")
            and Path(run["arguments"]["checkpoint"]).resolve(strict=True) == checkpoint
            and metadata_path == checkpoint.with_name(checkpoint.stem + "_manifest.json"),
            "C checkpoint path is not its bound immutable v3 source")
    require(file_record(checkpoint)["sha256"] == record["checkpoint_sha256"]
            and file_record(metadata_path)["sha256"] == record["manifest_sha256"],
            "C saved checkpoint or sidecar changed")
    metadata = read_json(metadata_path)
    require(metadata["save_load_round_trip"] is True
            and metadata["global_policy_decisions"] == proof["saved_global_policy_decisions"]
            and all(metadata[key] == value for key, value in proof["parameter_hashes"].items()),
            "C loaded state differs from the verified checkpoint")
    # The pinned semantic_video_cli checkpoint_loader calls stochastic_output=False.
    # The original full success validator also binds the unchanged runtime and
    # successful completion after its no-actor/critic/optimizer/normalizer-update assertion.


def validate(directory, role):
    root, source, run, configs, paths, roll = managed_source(directory, role)
    if source["success_candidate"]:
        validate_semantic_video_source(root, expected_role=role)
        outcome, reason = "TASK SUCCESS", "shared physical evaluator and stable post-roll"
    else:
        require(role == "A", "this bounded helper supports C success only; retain failed C raw files")
        outcome, reason = verify_a_diagnostic(source, configs, paths, roll)
    if role == "C":
        verify_c_checkpoint(source, run)
    count, technical = verify_source_timing(source, paths)
    return {"root": root, "source": source, "paths": paths, "count": count,
            "outcome": outcome, "reason": reason, "technical": technical}


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-source", required=True, type=Path)
    parser.add_argument("--c-source", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--acknowledge-untested-helper", action="store_true", required=True)
    args = parser.parse_args()
    a, c = validate(args.a_source, "A"), validate(args.c_source, "C")
    for key in ("runtime_contract", "camera", "seed", "semantic_version", "pre_action_source",
                "pre_action_ticks", "extra_pre_action_physics_ticks", "settle_capture_evidence"):
        require(a["source"][key] == c["source"][key], f"A/C condition differs: {key}")
    count = max(a["count"], c["count"])
    output = args.output_dir.absolute()
    for component in (output, *output.parents):
        if component.exists() or component.is_symlink():
            metadata = component.lstat()
            require(not stat.S_ISLNK(metadata.st_mode) and not
                    (getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT),
                    "output traverses a symlink/reparse path")
    output = output.resolve()
    require(output.is_relative_to(PROJECT_ROOT / "outputs/ppo_semantic_v3/videos")
            and not output.exists(), "choose a fresh directory below outputs/ppo_semantic_v3/videos")
    output.mkdir(parents=True, exist_ok=False)
    destination = output / "fsm_vs_ppo_outcomes.mp4"  # Never a success/improved filename.
    record = {"schema": "wlr50_clean.artifact_neutral_outcome_comparison.v1",
        "helper_review_status": DRAFT_STATUS, "helper": file_record(Path(__file__)),
        "sources": [{"manifest": file_record(item["root"] / "semantic_video_source_manifest.json"),
                     "outcome": item["outcome"], "reason": item["reason"],
                     "source_frame_count": item["count"], "technical": item["technical"]} for item in (a, c)],
        "improved_claim": False, "stability_superiority_claim": False, "optimizer_gate": False,
        "single_source_episode_per_side": True, "source_cropping": False,
        "speed_modified": False, "frame_interpolation": False,
        "shorter_source_suffix": "last recorded frame held; SOURCE ENDED label; not ongoing physics",
        "metric_comparison": "none; unequal episode durations do not establish stability superiority"}
    write_new(output / "comparison.started.json", record)
    branches = []
    for index, item in enumerate((a, c)):
        title = ("FSM A - " if index == 0 else "PPO C - ") + item["outcome"]
        branch = f"[{index}:v]setsar=1"
        if item["count"] < count:
            branch += f",tpad=stop_mode=clone:stop={count-item['count']}"
        branch += f",drawtext=text='{title}':x=28:y=26:fontcolor=white:fontsize=26"
        if item["count"] < count:
            branch += (",drawtext=text='SOURCE ENDED - last recorded frame held':x=28:y=64:"
                       f"fontcolor=white:fontsize=22:enable='gte(n,{item['count']})'")
        branches.append(branch + f"[side{index}]")
    filters = ";".join(branches + ["[side0][side1]hstack=inputs=2[out]"])
    command = [str(find_ffmpeg()), "-hide_banner", "-nostdin", "-v", "error", "-n"]
    for item in (a, c):
        command += ["-i", str(item["paths"]["actual_viewport_video.mp4"])]
    command += ["-filter_complex", filters, "-map", "[out]", "-an", "-sn", "-dn",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-fps_mode", "passthrough",
        "-movflags", "+faststart", str(destination)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
        require(result.returncode == 0, result.stderr[-2000:])
        validation = validate_mp4(destination, expected_width=2560, expected_height=720,
            expected_fps=FPS, expected_frame_count=count, maximum_duration_s=200.,
            require_sane_container_duration=True)
        require(validation["valid"] is True, "output full decode/PTS/container validation failed")
        for index, frame in enumerate(decode_frame_timeline(destination)):
            require(math.isclose(frame.pts_s, index/FPS, abs_tol=1e-5), "comparison changed playback timing")
        write_new(output / "comparison_manifest.json", {**record, "technical_status": "VALIDATED",
            "video": file_record(destination), "validation": validation, "ffmpeg_command": command})
    except BaseException as error:
        write_new(output / "comparison_failed.json", {**record, "technical_status": "FAILED",
            "error": f"{type(error).__name__}: {error}", "ffmpeg_command": command,
            "partial_output_preserved": destination.exists()})
        raise
    print(destination)


if __name__ == "__main__":
    main()

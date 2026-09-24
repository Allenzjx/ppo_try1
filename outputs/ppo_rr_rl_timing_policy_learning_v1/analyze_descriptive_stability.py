"""Generic sealed-checkpoint descriptive stability report.

The body-motion math and sealed height-diagnostic reader are reused from the
CP223616 helper.  This wrapper derives the candidate's public CP label from
the verified checkpoint/sidecar bound by the sealed video source; callers do
not supply a CP label or counter.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
LEGACY_PATH = HERE / "analyze_cp223616_descriptive_stability.py"
SPEC = importlib.util.spec_from_file_location("_descriptive_stability_math", LEGACY_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the outputs-only descriptive math helper")
math_helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(math_helper)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def checked_head(value: str) -> str:
    value = str(value).lower()
    require(len(value) == 40 and all(c in "0123456789abcdef" for c in value),
            "--expected-head must be one full lowercase Git commit")
    return value


def sealed_checkpoint_identity(source: Path, manifest_sha: str, run_sha: str,
                               expected_head: str) -> dict[str, Any]:
    source = source.resolve(strict=True)
    manifest_path = source / "semantic_video_source_manifest.json"
    run_path = source.parent / "run_manifest.json"
    require(manifest_path.is_file() and run_path.is_file(),
            "source/run manifests are missing")
    require(math_helper.sha256(manifest_path) ==
                math_helper.checked_sha(manifest_sha, "candidate manifest SHA") and
            math_helper.sha256(run_path) ==
                math_helper.checked_sha(run_sha, "candidate run SHA"),
            "sealed source/run manifest differs from explicit invocation")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = json.loads(run_path.read_text(encoding="utf-8"))
    require(bool(run.get("completed_at_utc")) and run.get("lifecycle") != "RUNNING" and
            manifest.get("schema") == "wlr50_clean.semantic_video_source.v1" and
            manifest.get("experiment_id") == "rr_rl_timing_policy_learning_v1" and
            manifest.get("role") == "C" and manifest.get("from_phase") == "P01" and
            manifest.get("episode_count") == 1 and
            manifest.get("optimizer_updates") == 0,
            "candidate is not one closed zero-update natural-P01 C evaluation")
    head = checked_head(expected_head)
    runtime = manifest.get("runtime_contract") or {}
    require(runtime.get("source_git_commit") == head,
            "candidate evaluation runtime differs from --expected-head")
    proof = manifest.get("checkpoint_load_provenance") or {}
    require(proof.get("checkpoint_loaded_and_verified") is True and
            manifest.get("policy_sampling_mode") == "deterministic_conditional_mean" and
            proof.get("stochastic_policy") in (None, False) and
            proof.get("policy_seed") is None,
            "candidate lacks deterministic verified checkpoint load provenance")
    binding = proof.get("source") or {}
    checkpoint = Path(binding.get("checkpoint", "")).resolve(strict=True)
    sidecar = Path(binding.get("manifest", "")).resolve(strict=True)
    expected_sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json").resolve(strict=True)
    require(sidecar == expected_sidecar,
            "bound checkpoint sidecar has a noncanonical path")
    checkpoint_sha = math_helper.sha256(checkpoint)
    sidecar_sha = math_helper.sha256(sidecar)
    require(checkpoint_sha == binding.get("checkpoint_sha256") and
            sidecar_sha == binding.get("manifest_sha256"),
            "checkpoint/sidecar bytes differ from the sealed load provenance")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    checkpoint_runtime = metadata.get("runtime_contract") or {}
    require(checkpoint_runtime.get("source_git_commit") == head and
            checkpoint_runtime.get("experiment_id") ==
                "rr_rl_timing_policy_learning_v1" and
            metadata.get("checkpoint_sha256") == checkpoint_sha and
            Path(metadata.get("checkpoint_path", "")).resolve() == checkpoint and
            metadata.get("save_load_round_trip") is True,
            "checkpoint sidecar lacks exact runtime/path/SHA/round-trip identity")
    counters = {}
    for name in ("global_policy_decisions", "ppo_updates", "optimizer_steps"):
        value = metadata.get(name)
        require(type(value) is int and value >= 0,
                f"checkpoint sidecar lacks valid {name}")
        counters[name] = value
    require(proof.get("saved_global_policy_decisions") ==
                counters["global_policy_decisions"],
            "source load proof and sidecar disagree on checkpoint step")
    actor_hash = (proof.get("parameter_hashes") or {}).get("actor_parameter_sha256")
    require(isinstance(actor_hash, str) and
            actor_hash == metadata.get("actor_parameter_sha256"),
            "source load proof and sidecar disagree on actor bytes")
    return {
        "display_label": f"CP{counters['global_policy_decisions']}",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_manifest": str(sidecar),
        "checkpoint_manifest_sha256": sidecar_sha,
        "runtime_head": head,
        "actor_parameter_sha256": actor_hash,
        "lifetime_counters": counters,
        "label_derived_from_verified_sealed_load_provenance": True,
    }


def main(args: argparse.Namespace) -> None:
    identity = sealed_checkpoint_identity(
        args.candidate_source, args.candidate_manifest_sha256,
        args.candidate_run_manifest_sha256, args.expected_head)
    candidate = math_helper.read_source(
        args.candidate_source, args.candidate_manifest_sha256,
        expected_experiment="rr_rl_timing_policy_learning_v1", expected_role="C")
    reference = math_helper.read_source(
        args.reference_source, args.reference_manifest_sha256,
        expected_experiment="task_conditioned_hip_wheel_v1", expected_role="B")
    candidate_label = identity["display_label"] + "_candidate"
    runs = {candidate_label: math_helper.public_run(candidate),
            "historical_N_reference": math_helper.public_run(reference)}
    result = {
        "schema": "wlr50_clean.descriptive_body_stability_two_sealed_runs.v2",
        "candidate_checkpoint_identity": identity,
        "runs": runs,
        "method": {
            "signals": "recorded actual base linear/angular velocity and base orientation quaternion",
            "quaternion_order": "wxyz",
            "roll_pitch_conversion": "normalized quaternion; standard world-referenced XYZ roll and pitch",
            "sampling": "sealed height_diagnostics; dominant 15 Hz (8 physics ticks)",
            "RMS": "left-sample piecewise-constant weighting to next diagnostic tick; exact physical endpoint duration; no interpolation",
            "extrema": "all recorded samples including exact zero-weight terminal sample",
            "RR_to_RL_window": "first recorded sample whose phase is P07 or later through the physical endpoint",
            "joint_command_variation": "N/A (not needed for this bounded body-motion table)",
        },
        "claims": {
            "stability_superiority_claimed": False,
            "causal_claimed": False,
            "time_normalized_ranking_claimed": False,
            "same_controller_or_version_claimed": False,
            "video_freeze_counted_as_physics": False,
            "different_task_outcomes_and_durations_retained": True,
        },
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    math_helper.write_new(output_json, json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    lines = [
        f"# {identity['display_label']} vs historical N: descriptive body-motion table", "",
        "These are separate sealed physical episodes with different task outcomes and durations. The table is descriptive only: it does **not** establish stability superiority, causality, or a time-normalized ranking. Historical N is not a fresh same-controller B. Video freeze is excluded.", "",
        "| Run / physical window | Outcome | Duration s | Samples | Linear speed RMS / max (m/s) | Angular speed RMS / max (rad/s) | Roll min..max / RMS0 (deg) | Pitch min..max / RMS0 (deg) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name, run in runs.items():
        outcome = "SUCCESS" if run["physical_task_success"] is True else "INCOMPLETE"
        for key in ("full_physical_episode", "RR_to_RL_window"):
            window = run[key]
            if window is None:
                lines.append(f"| {run_name} / {key} | {outcome} | N/A | 0 | N/A | N/A | N/A | N/A |")
                continue
            lin = window["body_linear_speed_m_s"]
            ang = window["body_angular_speed_rad_s"]
            roll = window["roll_deg"]
            pitch = window["pitch_deg"]
            fmt = math_helper.fmt
            lines.append(
                f"| {run_name} / {key} | {outcome} | {fmt(window['duration_s'])} | "
                f"{window['sample_count_including_zero_weight_endpoint']} | "
                f"{fmt(lin['rms'])} / {fmt(lin['max'])} | "
                f"{fmt(ang['rms'])} / {fmt(ang['max'])} | "
                f"{fmt(roll['min'])}..{fmt(roll['max'])} / {fmt(roll['rms_about_zero'])} | "
                f"{fmt(pitch['min'])}..{fmt(pitch['max'])} / {fmt(pitch['rms_about_zero'])} |")
    lines += [
        "", "## Sampling and source boundary", "",
        "Both runs use their sealed 15 Hz `height_diagnostics.jsonl` sensor stream (dominant stride: 8 physics ticks at 120 Hz). RMS uses a documented left-sample piecewise-constant weighting over the exact physical duration; it is an endpoint-weighted approximation, not a 120 Hz reconstruction or interpolation.", "",
        f"Candidate label `{identity['display_label']}` comes from the lifetime counter in the exact sidecar referenced and byte-verified by the sealed source's checkpoint load provenance; it is not caller-provided.", "",
        "Only the candidate run/source manifests, its checkpoint/sidecar, and each source's height-diagnostic JSONL are used and SHA-verified. Full paths and hashes are retained in the JSON report.", "",
        f"Machine-readable report: [{output_json.name}]({output_json.name})", "",
    ]
    math_helper.write_new(output_md, "\n".join(lines))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--candidate-source", type=Path, required=True)
    result.add_argument("--candidate-manifest-sha256", required=True)
    result.add_argument("--candidate-run-manifest-sha256", required=True)
    result.add_argument("--expected-head", required=True)
    result.add_argument("--reference-source", type=Path, required=True)
    result.add_argument("--reference-manifest-sha256", required=True)
    result.add_argument("--output-json", type=Path, required=True)
    result.add_argument("--output-md", type=Path, required=True)
    return result


if __name__ == "__main__":
    main(parser().parse_args())

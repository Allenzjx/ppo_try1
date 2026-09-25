"""NOTEXECUTED: one-shot real student-entry direction diagnostic.

The diagnostic starts from a natural P01 episode under one frozen checkpoint.
It computes the student's deterministic action at every decision, and—once a
strict P09 qualified/within-XY AIR entry is observed—replaces only three raw
latent components before calling the ordinary ``core.step``:

* FL knee: explicitly positive raw candidate;
* FR knee: explicitly positive raw candidate;
* RR hip: explicitly negative raw candidate.

RR knee and the other eight channels remain the current student's request.
The ordinary projector, mapper, caps, slew, single articulation write, and
native readback audit remain in the path.  This does *not* promise an absolute
joint target or exact RR-knee hold.  It is a finite exogenous diagnostic, not a
PPO sample, PPO update, AUX update, teacher, or learned-policy evaluation.

There are deliberately no top-level Torch/Isaac imports.  Root must review the
exact frozen checkpoint/runtime and CLI values before adding the explicit
``--execute-reviewed-real-diagnostic`` flag.  The default invocation refuses to
launch physics.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "src/wlr50_clean").is_dir())
SUPPORTED_EXPERIMENTS = {
    "rr_rl_timing_policy_learning_v1",
}
OVERRIDES = {1: "FL_knee_positive", 3: "FR_knee_positive", 6: "RR_hip_negative"}
ZERO_CREDIT = {
    "new_PPO_samples": 0,
    "new_PPO_updates": 0,
    "new_optimizer_steps": 0,
    "new_AUX_accepted_updates": 0,
    "new_AUX_attempted_steps": 0,
    "teacher_actions_deployed": False,
    "diagnostic_rows_entered_in_on_policy_storage": False,
    "formal_deterministic_policy_result": False,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exclusive_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def diagnostic_json_payload(value: Any, *, physical_json: Any, jsonable: Any) -> Any:
    """Serialize measured dataclasses before the generic tensor/list adapter.

    ``SemanticPhysicalObservation`` is a frozen slots dataclass.  The official
    physical recorder's ``physical_json`` is the authority for dataclasses,
    enums and explicit non-finite tokens; ``jsonable`` remains the final
    adapter for any already-materialized tensor-like policy audit values.
    """
    return jsonable(physical_json(value))


def apply_raw_candidate(
    original: Sequence[float], *, fl_knee_raw: float, fr_knee_raw: float,
    rr_hip_raw: float,
) -> tuple[float, ...]:
    """Return one explicit raw latent, without offsets or accumulation."""
    if len(original) != 12 or not all(math.isfinite(float(x)) for x in original):
        raise ValueError("finite Full12 original student request required")
    candidates = (float(fl_knee_raw), float(fr_knee_raw), float(rr_hip_raw))
    if not all(math.isfinite(x) and abs(x) <= 3.0 for x in candidates):
        raise ValueError("diagnostic raw candidates must be finite and within [-3,3]")
    if not (fl_knee_raw > 0.0 and fr_knee_raw > 0.0 and rr_hip_raw < 0.0):
        raise ValueError("required directions are FL knee+, FR knee+, RR hip-")
    result = [float(x) for x in original]
    result[1], result[3], result[6] = candidates
    return tuple(result)


def rr_air_entry_evidence(
    task: Mapping[str, Any], *, minimum_other_supports: int,
    force_noise_floor_n: float,
) -> dict[str, Any]:
    """Evaluate the bounded trigger from already-recorded task evidence only."""
    evaluator = task.get("physical_evaluator") or {}
    current = evaluator.get("current_legs") or {}
    history = evaluator.get("history") or {}
    rr = current.get("RR") or {}
    blockers: list[str] = []
    if task.get("phase_id", task.get("state_id")) not in (None, "P09"):
        blockers.append("phase_not_P09")
    if evaluator.get("valid") is False:
        blockers.append("physical_evaluator_invalid")
    if task.get("termination_reason"):
        blockers.append("task_already_terminal")
    if not rr.get("current_lift_valid", False):
        blockers.append("RR_not_currently_qualified")
    if not (history.get("active_lift") or {}).get("RR", False):
        blockers.append("RR_active_lift_history_absent")
    if not rr.get("within_top_xy", False):
        blockers.append("RR_not_within_top_xy")
    if not rr.get("within_lateral_span", False):
        blockers.append("RR_not_within_lateral_span")
    if not rr.get("air", False):
        blockers.append("RR_not_AIR")
    if rr.get("top_contact") or rr.get("ground_contact"):
        blockers.append("RR_already_in_wheel_contact")
    front = rr.get("front_distance_m")
    gap = rr.get("clearance_m")
    if front is None or not math.isfinite(float(front)) or float(front) < 0.0:
        blockers.append("RR_front_distance_not_nonnegative")
    if gap is None or not math.isfinite(float(gap)) or float(gap) <= 0.0:
        blockers.append("RR_positive_top_gap_absent")

    supports = []
    for leg in ("FR", "FL", "RL"):
        item = current.get(leg) or {}
        force = item.get("bearing_force_n")
        if (item.get("support") and item.get("bearing_verified")
                and force is not None and math.isfinite(float(force))
                and float(force) >= force_noise_floor_n):
            supports.append(leg)
    if "FL" not in supports:
        blockers.append("FL_real_support_absent")
    if len(supports) < minimum_other_supports:
        blockers.append("insufficient_other_real_supports")
    return {
        "eligible": not blockers,
        "blockers": blockers,
        "phase": task.get("phase_id", task.get("state_id")),
        "RR": deepcopy(rr),
        "supporting_other_legs": supports,
        "minimum_other_supports": minimum_other_supports,
        "force_noise_floor_n": force_noise_floor_n,
    }


def active_release_reason(
    task: Mapping[str, Any], evidence: Mapping[str, Any], *, elapsed_s: float,
    maximum_probe_seconds: float,
) -> str | None:
    """Return the first conservative one-shot release reason.

    Real RR contact is deliberately classified before the generic eligibility
    loss it also causes.  Every other lost trigger prerequisite—including
    current lift qualification and required support—is fail-closed.
    """
    rr = evidence["RR"]
    if (not rr.get("air", False) or rr.get("top_contact")
            or rr.get("ground_contact")):
        return "RR_real_contact_observed"
    if task.get("termination_reason"):
        return "task_terminal"
    if evidence.get("phase") != "P09":
        return "source_phase_left_P09"
    if not evidence.get("eligible", False):
        blockers = evidence.get("blockers") or ["unspecified_entry_evidence_loss"]
        return "entry_eligibility_lost:" + ",".join(str(item) for item in blockers)
    if elapsed_s >= maximum_probe_seconds:
        return "finite_duration_complete"
    return None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--checkpoint-manifest-sha256", required=True)
    result.add_argument("--expected-head", required=True)
    result.add_argument("--experiment-id", required=True, choices=sorted(SUPPORTED_EXPERIMENTS))
    result.add_argument("--semantic-version", default="v3", choices=("v3",))
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--fl-knee-raw", type=float, required=True)
    result.add_argument("--fr-knee-raw", type=float, required=True)
    result.add_argument("--rr-hip-raw", type=float, required=True)
    result.add_argument("--maximum-probe-seconds", type=float, default=5.0)
    result.add_argument("--execute-reviewed-real-diagnostic", action="store_true")
    return result


def _policy_args(args: argparse.Namespace, metadata: Mapping[str, Any]) -> None:
    policy = metadata["policy_contract"]
    args.seed = 4001
    args.stochastic_policy = False
    args.policy_seed = None
    args._policy_version = policy["version"]
    args._observation_layout = policy.get("observation_layout")
    args._migration_record = None


def _rear_assists_off(backend: Any, frame: Any) -> dict[str, Any]:
    receipt = {
        "rr_capture_assist_object": getattr(backend, "_rr_capture_assist", None) is not None,
        "nominal_geometry_advisory": getattr(backend, "_nominal_geometry_mode", None),
        "rr_carry_wheel_mode": getattr(backend, "_rr_carry_wheel_mode", "off"),
        "reported_rear_task_assist_disabled": frame.info.get("rear_task_assist_disabled"),
        "front_capture_assist_present": getattr(backend, "_capture_assist", None) is not None,
    }
    if (receipt["rr_capture_assist_object"]
            or receipt["nominal_geometry_advisory"] is not None
            or receipt["rr_carry_wheel_mode"] not in (None, "off")
            or receipt["reported_rear_task_assist_disabled"] is not True):
        raise ValueError("rear task assist must be explicitly OFF for this diagnostic")
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    # Validate the candidate before any runtime import or output creation.
    apply_raw_candidate((0.0,) * 12, fl_knee_raw=args.fl_knee_raw,
                        fr_knee_raw=args.fr_knee_raw, rr_hip_raw=args.rr_hip_raw)
    if not 4.0 <= args.maximum_probe_seconds <= 6.0:
        raise ValueError("finite diagnostic duration must be within [4,6] seconds")
    if not args.execute_reviewed_real_diagnostic:
        raise ValueError("NOTEXECUTED draft: explicit reviewed real-diagnostic flag required")
    if len(args.expected_head) != 40:
        raise ValueError("full frozen Git HEAD required")

    sys.path.insert(0, str(ROOT / "src"))
    checkpoint = args.checkpoint.resolve(strict=True)
    checkpoint_manifest = checkpoint.with_name(checkpoint.stem + "_manifest.json").resolve(strict=True)
    metadata = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    if (checkpoint.name == "checkpoint_last.pt"
            or sha256(checkpoint) != args.checkpoint_sha256
            or sha256(checkpoint_manifest) != args.checkpoint_manifest_sha256
            or metadata.get("checkpoint_sha256") != args.checkpoint_sha256
            or metadata.get("save_load_round_trip") is not True):
        raise ValueError("immutable round-tripped checkpoint binding required")
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    contract = runtime_contract(expected_head=args.expected_head,
                                semantic_version=args.semantic_version,
                                experiment_id=args.experiment_id)
    if metadata.get("runtime_contract") != contract:
        raise ValueError("diagnostic checkpoint must exactly match frozen runtime; no migration here")
    policy_contract = metadata.get("policy_contract")
    if (not isinstance(policy_contract, dict)
            or policy_contract.get("action_dimension", 12) != 12
            or type(policy_contract.get("observation_dimension")) is not int):
        raise ValueError("checkpoint policy contract is not a bound Full12 policy")
    _policy_args(args, metadata)

    run_dir = args.run_dir.resolve()
    allowed_root = (ROOT / "runs" / f"ppo_{args.experiment_id}" / "direction_probe").resolve()
    if not run_dir.is_relative_to(allowed_root) or run_dir == allowed_root:
        raise ValueError(f"run directory must be a new child of {allowed_root}")
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "schema": "outputs.student_entry_direction_probe.v1",
        "lifecycle": "RUNNING",
        **ZERO_CREDIT,
        "run_role": "NATURAL_P01_FROZEN_STUDENT_ONE_SHOT_RAW_DIRECTION_DIAGNOSTIC",
        "formal_policy_or_success_claim": False,
        "natural_P01_student_prefix": True,
        "successful_N_prefix_used": False,
        "state_injection_or_teleport": False,
        "candidate_semantics": (
            "absolute raw latent replacement at indices1/3/6; no per-tick accumulation; "
            "RR knee and other channels remain the student's current request; ordinary "
            "projector/mapper/caps/slew/write path remains authoritative"
        ),
        "candidate_raw": {
            "FL_knee": args.fl_knee_raw,
            "FR_knee": args.fr_knee_raw,
            "RR_hip": args.rr_hip_raw,
            "RR_knee": "student_request_unchanged_not_claimed_exact_final_hold",
        },
        "maximum_probe_seconds": args.maximum_probe_seconds,
        "seed": 4001,
        "deterministic": True,
        "experiment_id": args.experiment_id,
        "runtime_contract": contract,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": args.checkpoint_sha256,
        "checkpoint_manifest": str(checkpoint_manifest),
        "checkpoint_manifest_sha256": args.checkpoint_manifest_sha256,
        "checkpoint_policy_contract": policy_contract,
        "runner_sha256": sha256(Path(__file__)),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    exclusive_json(run_dir / "run_manifest.started.json", manifest)

    app = core = physical = unchanged = None
    lock = (ROOT / "runs/ppo_semantic_v2/.single_process.lock").open("r+b")
    try:
        import msvcrt
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        # Preserve the proven Windows native import order.
        import torch  # noqa: F401 - runtime-only import
        import tensordict  # noqa: F401 - runtime-only import
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=False).app
        app.update()
        from wlr50_clean.ppo.semantic_training import seed_training_rngs, jsonable
        from wlr50_clean.ppo.semantic_video_cli import build_video_core, checkpoint_loader
        from wlr50_clean.ppo.semantic_legacy_evaluation import (
            PhysicalEvaluationRecorder, physical_json)

        seed_training_rngs(4001)
        core = build_video_core(app, role="C", semantic_version=args.semantic_version,
                                experiment_id=args.experiment_id)
        observation = tuple(core.reset(seed=4001))
        if (core.frame.state_id != "P01" or core.frame.physics_tick != 0
                or len(observation) != policy_contract["observation_dimension"]):
            raise ValueError("natural P01 reset or checkpoint observation contract mismatch")
        rear_assist_receipt = _rear_assists_off(core.backend, core.frame)
        for phase in (f"P{number:02d}" for number in range(1, 14)):
            if tuple(core.projector.config.mask_for(phase)) != (1,) * 12:
                raise ValueError("diagnostic requires ordinary all12 student permission")
        action, load_proof, unchanged = checkpoint_loader(args, contract)(observation)
        manifest.update(
            official_checkpoint_load_proof=load_proof,
            rear_assist_receipt=rear_assist_receipt,
        )
        config = ROOT / "configs" / f"ppo_{args.experiment_id}"
        physical = PhysicalEvaluationRecorder(
            run_dir,
            task_spec_path=config / "stage_task_spec.yaml",
            quality_score_path=config / "quality_score.yaml",
        )
        physical.start(core.frame)

        probe = {
            "status": "WAIT",
            "start_time_s": None,
            "start_tick": None,
            "release_time_s": None,
            "release_tick": None,
            "release_reason": None,
            "trigger_evidence": None,
        }
        decision_context: dict[str, Any] = {}
        support_spec = core.backend._controller.supervisor.spec["support"]
        minimum_other_supports = int(support_spec["minimum_other_supports"])
        force_noise_floor_n = float(support_spec["force_noise_floor_n"])

        def release(reason: str) -> None:
            if probe["status"] == "ACTIVE":
                probe.update(status="RELEASED", release_time_s=core.frame.sim_time_s,
                             release_tick=core.frame.physics_tick, release_reason=reason)

        with ExitStack() as stack:
            ticks_stream = stack.enter_context(
                (run_dir / "diagnostic_physics.jsonl").open("x", encoding="utf-8"))
            decisions_stream = stack.enter_context(
                (run_dir / "diagnostic_decisions.jsonl").open("x", encoding="utf-8"))

            def emit(stream: Any, row: Mapping[str, Any]) -> None:
                payload = diagnostic_json_payload(
                    row, physical_json=physical_json, jsonable=jsonable)
                stream.write(json.dumps(payload, allow_nan=False) + "\n")
                stream.flush()

            def observe(before: Any, after: Any, projection: Any) -> None:
                physical.observe(before, after, projection)
                ack = after.info["atomic_ack"]
                audit = after.info["actuator_target_effect_audit"]
                if audit.get("verified") is not True or ack.get("articulation_writes_this_call") != 1:
                    raise RuntimeError("ordinary one-write native target/readback audit failed")
                emit(ticks_stream, {
                    **ZERO_CREDIT,
                    "episode_physics_tick": after.physics_tick,
                    "sim_time_s": after.sim_time_s,
                    "source_phase": before.state_id,
                    "resulting_phase": after.state_id,
                    "decision": decision_context,
                    "projected_residual_full12": projection.safe_projected_residual_full12,
                    "atomic_ACK": ack,
                    "native_readback_audit": audit,
                    "physical_evaluator": after.info["semantic_task"]["physical_evaluator"],
                    "raw_observation": after.info.get("raw_observation"),
                    "probe_state": deepcopy(probe),
                })

            core.tick_observer = observe
            while not core.done and core.frame.sim_time_s < 200.0 - 1e-10:
                original = tuple(action(observation, core.decision_count))
                request_audit = deepcopy(getattr(action, "last_request", None))
                task = deepcopy(core.frame.info["semantic_task"])
                task.setdefault("phase_id", core.frame.state_id)
                eligibility = rr_air_entry_evidence(
                    task,
                    minimum_other_supports=minimum_other_supports,
                    force_noise_floor_n=force_noise_floor_n,
                )
                if probe["status"] == "WAIT" and eligibility["eligible"]:
                    probe.update(status="ACTIVE", start_time_s=core.frame.sim_time_s,
                                 start_tick=core.frame.physics_tick,
                                 trigger_evidence=eligibility)
                if probe["status"] == "ACTIVE":
                    elapsed = core.frame.sim_time_s - probe["start_time_s"]
                    reason = active_release_reason(
                        task,
                        eligibility,
                        elapsed_s=elapsed,
                        maximum_probe_seconds=args.maximum_probe_seconds,
                    )
                    if reason is not None:
                        release(reason)
                active = probe["status"] == "ACTIVE"
                applied = (apply_raw_candidate(
                    original,
                    fl_knee_raw=args.fl_knee_raw,
                    fr_knee_raw=args.fr_knee_raw,
                    rr_hip_raw=args.rr_hip_raw,
                ) if active else original)
                decision_context = {
                    "decision_index": core.decision_count,
                    "decision_start_tick": core.frame.physics_tick,
                    "decision_start_time_s": core.frame.sim_time_s,
                    "phase": core.frame.state_id,
                    "policy_observation_vector": list(observation),
                    "original_student_raw_full12": list(original),
                    "original_student_request_audit": request_audit,
                    "applied_raw_full12": list(applied),
                    "overridden_indices": sorted(OVERRIDES) if active else [],
                    "override_semantics": OVERRIDES if active else {},
                    "eligibility": eligibility,
                    "probe_active": active,
                }
                start_tick = core.frame.physics_tick
                step = core.step(applied)
                observation = tuple(step.observation)
                emit(decisions_stream, {
                    **ZERO_CREDIT,
                    **decision_context,
                    "decision_end_tick": core.frame.physics_tick,
                    "decision_physics_ticks": core.frame.physics_tick - start_tick,
                    "step_info": step.info,
                    "probe_state_after_step": deepcopy(probe),
                })

        if probe["status"] == "ACTIVE":
            release("episode_or_global_deadline_ended_during_probe")

        unchanged()
        if (runtime_contract(expected_head=args.expected_head,
                             semantic_version=args.semantic_version,
                             experiment_id=args.experiment_id) != contract
                or sha256(checkpoint) != args.checkpoint_sha256
                or sha256(checkpoint_manifest) != args.checkpoint_manifest_sha256):
            raise RuntimeError("runtime or checkpoint changed during diagnostic")
        manifest.update(
            lifecycle="DIAGNOSTIC_SEALED",
            endpoint_tick=core.frame.physics_tick,
            physical_seconds=core.frame.sim_time_s,
            final_phase=core.frame.state_id,
            actual_task_termination_reason=core.frame.info["semantic_task"].get("termination_reason"),
            probe=probe,
            physical_summary=physical.summary(),
            frozen_learned_state_unchanged=True,
            same_episode_continued_to_natural_result=True,
        )
    except BaseException as exc:
        manifest.update(lifecycle="DIAGNOSTIC_ERROR", error=repr(exc),
                        traceback=traceback.format_exc())
        raise
    finally:
        if unchanged is not None:
            try:
                unchanged()
                manifest["frozen_learned_state_unchanged"] = True
            except BaseException as integrity_error:
                manifest.update(lifecycle="DIAGNOSTIC_ERROR",
                                frozen_learned_state_unchanged=False,
                                learned_state_verification_error=repr(integrity_error))
        if physical is not None:
            physical.close()
        if core is not None and core.frame is not None:
            manifest["last_episode_tick"] = core.frame.physics_tick
            manifest["last_simulation_time_s"] = core.frame.sim_time_s
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        exclusive_json(run_dir / "run_manifest.json", manifest)
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)
        lock.close()


if __name__ == "__main__":
    main()

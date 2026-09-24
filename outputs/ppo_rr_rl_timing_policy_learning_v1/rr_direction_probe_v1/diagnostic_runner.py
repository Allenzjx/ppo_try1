"""DRAFT / NOTEXECUTED: isolated JSONL-only real-physics RR direction diagnostic.

No top-level Torch/Isaac imports. Root must first run the separate native-audit
tests when Isaac is idle and review this file. This is not a formal C evaluator,
trainer, capture assist, or success video. No silent reset/retry after a miss.
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

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/wlr50_clean").is_dir())
EXPERIMENT = "rr_rl_timing_policy_learning_v1"
CONFIG = ROOT / "configs" / ("ppo_" + EXPERIMENT)
ZERO = (0.,) * 12
ATTRIBUTION = {
    "run_role": "DIAG_NprefixP07_frozen_policy_post_mapper_RR_INTERVENTION",
    "formal_C_or_pure_policy_success": False,
    "raw_policy_sample_executed_unmodified": False,
    "legacy_all12_policy_unmodified_and_effect_attribution_are_superseded": True,
    "supersession_scope": "policy_provenance_only_not_physical_readback_or_safety",
    "native_actual_readback_audit_required": True,
    "zero_policy_counterfactual_retains_same_exogenous_RR_target": True,
    "new_PPO_decisions": 0, "new_PPO_updates": 0, "new_optimizer_steps": 0,
    "rollout_samples_written": 0, "teacher_or_diagnostic_data_in_PPO_storage": False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def member(value, name):
    return value[name] if isinstance(value, dict) else getattr(value, name)


def source_owners(provider, servo_names):
    """Observe existing ordered channel ownership, without ticking source clocks.

    Later touched, non-retired layer wins exactly as the ordinary source merge.
    Include latest authored event for that channel; not the changing cursor or
    snapshot timestamp. A new event/owner conservatively releases the probe.
    """
    result = {}
    for index in (6, 7):
        owner = None
        for ordinal, layer in enumerate(provider._continuous_layers):
            sample = layer.get("sample")
            if (sample is None or index not in layer["touched"]
                    or index in layer.get("capture_retired_servo_indices", ())):
                continue
            motion = layer["motion"]
            groups = [(j, motion._scaled_source_tick(group.time_s))
                      for j, group in enumerate(motion.phase.atomic_groups)
                      if servo_names[index] in group.channels
                      and motion._scaled_source_tick(group.time_s) <= sample.tick_index]
            # Some inherited source channels have no explicit atomic event.
            # The owning layer remains real; do not invent an event timestamp.
            owner = (ordinal, layer["stage"], tuple(groups[-1]) if groups else None)
        result[index] = owner
    return tuple(result[i] for i in (6, 7))


def deterministic_proposal_log_density(request):
    """Density of the actual conditional mean, not a stochastic sample logp.

    Uses the official request audit's one-forward mean/std. No extra actor call
    or stale actor distribution is queried on the deterministic path.
    """
    raw, mean, sigma = (request[key] for key in
        ("selected_raw_full12", "conditional_mean_full12", "effective_sigma_full12"))
    if not (len(raw) == len(mean) == len(sigma) == 12):
        raise ValueError("actual Full12 Gaussian request receipt required")
    if any(not math.isfinite(s) or s <= 0. for s in sigma):
        raise ValueError("finite positive conditional standard deviations required")
    return sum(-.5*((x-m)/s)**2-math.log(s)-.5*math.log(2.*math.pi)
               for x, m, s in zip(raw, mean, sigma))


def capture_context(core, raw, log_density):
    """Current evaluator/source + adjacent real ACK; mapped N is NOT read here."""
    from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
    frame, backend = core.frame, core.backend
    adapter, ack = backend._adapter, frame.info["atomic_ack"]
    task, observation = frame.info["semantic_task"], frame.info["raw_observation"]
    ev = task["physical_evaluator"]
    if (ack["physics_tick"] != adapter._last_physics_tick
            or ack["write_count"] != adapter.write_count
            or backend._controller_frame.physics_tick != frame.physics_tick
            or ev["physics_tick"] != frame.physics_tick
            or not math.isclose(ev["simulation_time_s"], frame.sim_time_s, abs_tol=1e-9)):
        raise ValueError("physical/source/evaluator/adjacent ACK clocks disagree")
    controller = backend._controller
    if getattr(controller, "mode", None) in ("TEACHER", "TAKEOVER"):
        raise ValueError("frozen teacher/takeover is not successful-nominal prefix")
    provider = controller.nominal_provider
    owners = source_owners(provider, SERVO_ORDER)
    if frame.state_id == "P09" and any(owner is None for owner in owners):
        raise ValueError("actual RR source owner could not be established")
    support = controller.supervisor.spec["support"]
    actual = (tuple(float(member(member(observation, "joints")[name], "position_deg")) for name in SERVO_ORDER)
              + tuple(float(member(member(observation, "wheels")[name], "velocity_rad_s")) for name in WHEEL_ORDER))
    prior = ack["policy_headroom_evidence"]["effective_policy_residual_full12"]
    caps = tuple(a*b for a, b in zip(core.projector.config.scale_for(frame.state_id),
                                   core.projector.config.physical_residual_scale_full12))
    return dict(dispatch_tick=adapter._last_physics_tick+1,
        previous_ack_tick=ack["physics_tick"], episode_observation_tick=frame.physics_tick,
        sim_time_s=frame.sim_time_s, physics_dt_s=adapter.physics_dt_s,
        phase=frame.state_id, evaluation=deepcopy(ev),
        minimum_other_supports=support["minimum_other_supports"],
        force_noise_floor_n=support["force_noise_floor_n"],
        physical_abort=bool(task.get("termination_reason") or frame.info["termination_mapping"]["active_sources"]),
        previous_final_full12=tuple(ack["drive_target_full12"]), actual_full12=actual,
        previous_effective_residual_full12=tuple(prior), residual_caps_full12=caps,
        residual_rate_deg_s=backend.execution_profile["residual"]["servo_rate_deg_s"],
        final_slew_deg_per_tick=adapter.servo_target_mapper.maximum_delta_deg,
        rr_owner_ids=owners, policy_raw_full12=tuple(raw), policy_log_probability=log_density)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--checkpoint-manifest-sha256", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--execute-real-diagnostic", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.execute_real_diagnostic:
        raise ValueError("DRAFT is not an invocation; explicit reviewed real-diagnostic flag required")
    sys.path.insert(0, str(ROOT / "src"))
    cp = args.checkpoint.resolve(strict=True)
    meta_path = cp.with_name(cp.stem + "_manifest.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if (cp.name == "checkpoint_last.pt" or sha(cp) != args.checkpoint_sha256
            or sha(meta_path) != args.checkpoint_manifest_sha256
            or meta.get("checkpoint_sha256") != args.checkpoint_sha256
            or meta.get("save_load_round_trip") is not True):
        raise ValueError("explicit immutable checkpoint/manifest binding and round-trip proof required")
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    contract = runtime_contract(expected_head=args.expected_head, semantic_version="v3", experiment_id=EXPERIMENT)
    if meta["runtime_contract"] != contract:
        raise ValueError("checkpoint must exactly match current frozen422 runtime; no implicit migration")
    from wlr50_clean.ppo.semantic_p02_progress_profile import P02_PROGRESS_POLICY, P02_PROGRESS_OBSERVATION_LAYOUT
    policy_contract = meta["policy_contract"]
    if (policy_contract["version"] != P02_PROGRESS_POLICY
            or policy_contract["observation_dimension"] != 422
            or policy_contract["observation_layout"] != P02_PROGRESS_OBSERVATION_LAYOUT):
        raise ValueError("only current422 frozen checkpoint supported")
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to(ROOT / "runs" / ("ppo_"+EXPERIMENT)):
        raise ValueError("isolate diagnostic under this experiment's runs directory")
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_files = ("diagnostic_runner.py", "isolated_headroom_wrapper.py", "target_plan.py")
    manifest = dict(ATTRIBUTION, schema="wlr50_clean.rr_direction_probe.v1",
        lifecycle="RUNNING", source_HEAD=args.expected_head, runtime_contract=contract,
        checkpoint=str(cp), checkpoint_sha256=sha(cp), checkpoint_manifest_sha256=sha(meta_path),
        artifact_sha256={name: sha(Path(__file__).with_name(name)) for name in artifact_files},
        in_memory_runtime_override=True, immutable_source_bytes_do_not_imply_runtime_equivalence=True,
        seed=4001, deterministic=True, actual_prefix_decisions=0, actual_diagnostic_policy_decisions=0,
        maximum_task_seconds=200., maximum_probe_seconds=4., camera_capture=False,
        capture_status="JSONL_ONLY_NO_VIDEO_PRODUCED", source_N_changed=False,
        state_injection=False, reset_count=1, started_at_utc=datetime.now(timezone.utc).isoformat(),
        official_loader_may_allocate_empty_RSL_storage=True,
        storage_semantics="no act/process_env_step/compute_returns/update and no rollout writes")
    write(run_dir / "run_manifest.started.json", manifest)
    app = physical = core = unchanged = None
    lock = (ROOT / "runs/ppo_semantic_v2/.single_process.lock").open("r+b")
    wrapper = None
    try:
        import msvcrt
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        # Match the existing Windows native import order; nothing above loads a model.
        import torch
        import tensordict
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=False).app
        app.update()
        from wlr50_clean.ppo.semantic_video_cli import build_video_core, checkpoint_loader
        from wlr50_clean.ppo.semantic_training import seed_training_rngs, jsonable
        from wlr50_clean.ppo.semantic_legacy_evaluation import PhysicalEvaluationRecorder
        from wlr50_clean.ppo.semantic_checkpoint_prefix import CheckpointPolicyPrefixRequest, _provenance, sampling_label
        from wlr50_clean.ppo import semantic_headroom
        from isolated_headroom_wrapper import IsolatedHeadroomProbe
        seed_training_rngs(4001)
        core = build_video_core(app, role="C", semantic_version="v3", experiment_id=EXPERIMENT)
        observation = core.reset(seed=4001)
        backend = core.backend
        if (len(observation) != 422 or core.frame.state_id != "P01" or core.frame.physics_tick != 0
                or backend._rr_capture_assist is not None or backend._nominal_geometry_mode is not None
                or backend._rr_carry_wheel_mode != "off" or backend._capture_assist is None):
            raise ValueError("expected exact naturalP01/422/rearOFF/declaredFL runtime")
        for phase in (f"P{i:02d}" for i in range(1, 14)):
            if tuple(core.projector.config.mask_for(phase)) != (1,)*12:
                raise ValueError("diagnostic cannot change formal all12 permission")
        args.checkpoint = cp
        args.semantic_version, args.experiment_id, args.seed = "v3", EXPERIMENT, 4001
        args.stochastic_policy, args.policy_seed = False, None
        args._policy_version, args._observation_layout = P02_PROGRESS_POLICY, P02_PROGRESS_OBSERVATION_LAYOUT
        action, load_proof, unchanged = checkpoint_loader(args, contract)(observation)
        manifest["official_checkpoint_load_proof"] = load_proof
        prefix = CheckpointPolicyPrefixRequest(target_phase="P07", source="successful_nominal",
                                               maximum_prefix_decisions=3000)
        provenance = _provenance(dict(schema="wlr50_clean.successful_nominal_prefix.v1",
            source="successful_nominal", raw_action_full12=list(ZERO), policy_credit=False,
            execution_profile_sha256=sha(CONFIG/"execution_profile.yaml"),
            stage_task_spec_sha256=sha(CONFIG/"stage_task_spec.yaml"),
            runtime_content_sha256=contract["runtime_content_sha256"],
            interface_contract=dict(observation_dimension=422, observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT,
                                    action_dimension=12)))
        manifest.update(prefix_request=prefix.as_dict(), prefix_sampling=sampling_label(prefix),
            prefix_provenance=provenance, prefix_fallback_override="none_keep_single_physical_episode")
        physical = PhysicalEvaluationRecorder(run_dir, task_spec_path=CONFIG/"stage_task_spec.yaml",
                                             quality_score_path=CONFIG/"quality_score.yaml")
        physical.start(core.frame)
        current_proposal = {"raw": ZERO, "log_density": None, "request": None}
        wrapper = IsolatedHeadroomProbe(semantic_headroom.project_semantic_servo_headroom,
            read_committed_adapter_tick=lambda: backend._adapter._last_physics_tick,
            diagnostic_only=True, training_allowed=False)
        prefix_complete = False
        with ExitStack() as scope:
            ticks = scope.enter_context((run_dir/"diagnostic_physics.jsonl").open("x", encoding="utf-8"))
            decisions = scope.enter_context((run_dir/"diagnostic_decisions.jsonl").open("x", encoding="utf-8"))
            def emit(stream, row):
                stream.write(json.dumps(jsonable(row), allow_nan=False)+"\n")
                stream.flush()
            def observe(before, after, projection):
                physical.observe(before, after, projection)
                ack = after.info["atomic_ack"]
                audit = after.info["actuator_target_effect_audit"]
                if audit.get("verified") is not True or ack.get("articulation_writes_this_call") != 1:
                    raise RuntimeError("genuine physical native audit failed; never bypass it")
                terminal = bool(after.info["semantic_task"].get("termination_reason")
                                or after.info["termination_mapping"]["active_sources"])
                # A numerical/safety terminal must keep its raw physical tail;
                # do not demand a finite next-dispatch context for a nonexistent
                # next action. The recorder has already preserved invalid data.
                ctx = (None if not prefix_complete or terminal else
                       capture_context(core, current_proposal["raw"], current_proposal["log_density"]))
                emit(ticks, dict(ATTRIBUTION, episode_physics_tick=after.physics_tick,
                    sim_time_s=after.sim_time_s, source_phase=before.state_id, resulting_phase=after.state_id,
                    prefix_zero_residual=not prefix_complete, original_policy_proposal=current_proposal,
                    original_projected_residual_full12=projection.safe_projected_residual_full12,
                    atomic_ACK=ack, native_readback_audit=audit,
                    physical_evaluator=after.info["semantic_task"]["physical_evaluator"],
                    nominal_source_diagnostics=backend._controller.nominal_provider.nominal_suggestion_diagnostics,
                    real_context_for_next_dispatch=ctx,
                    diagnostic_plan=None if wrapper.pending is None else wrapper.pending.get("plan_receipt"),
                    original_request_HISTORY=core._history))
                if prefix_complete and not terminal:
                    wrapper.arm_from_observer(ctx)
            core.tick_observer = observe
            while not core.done and core.frame.sim_time_s < 200.-1e-10:
                if not prefix_complete and core.frame.state_id == prefix.target_phase:
                    prefix_complete = True
                    manifest["prefix_entry"] = dict(episode_physics_tick=core.frame.physics_tick,
                        simulation_time_s=core.frame.sim_time_s, phase=core.frame.state_id,
                        no_reset_or_HISTORY_reinitialization=True)
                    wrapper.arm_from_observer(capture_context(core, ZERO, None))
                    scope.enter_context(wrapper.installed_on(semantic_headroom))
                if not prefix_complete and int(core.frame.state_id[1:]) > 7:
                    # No teleport/fallback or undocumented later-stage start.
                    raise RuntimeError("P07 not observed at a decision boundary; diagnostic prefix missed")
                if prefix_complete:
                    raw = action(tuple(observation), core.decision_count)
                    request = deepcopy(action.last_request)
                    density = deterministic_proposal_log_density(request)
                    current_proposal.update(raw=raw, log_density=density, request=request,
                        log_density_semantics="original_conditional_mean_Gaussian_density_not_injected_action")
                    wrapper.bind_decision_proposal(raw, density)
                    manifest["actual_diagnostic_policy_decisions"] += 1
                else:
                    raw = ZERO
                    manifest["actual_prefix_decisions"] += 1
                start_tick = core.frame.physics_tick
                step = core.step(raw)
                observation = tuple(step.observation)
                emit(decisions, dict(ATTRIBUTION, start_tick=start_tick, end_tick=core.frame.physics_tick,
                    prefix_zero_residual=not prefix_complete, original_policy_proposal=current_proposal,
                    step_info=step.info, probe_state=wrapper.planner.state,
                    full_task_success_claim=False))
                # Neither first TOP nor release ends the loop. Normal task result
                # (including genuine safety/local failure) and 200s remain final.
        unchanged()
        if runtime_contract(expected_head=args.expected_head, semantic_version="v3", experiment_id=EXPERIMENT) != contract:
            raise RuntimeError("source/config changed during isolated diagnostic")
        if sha(cp) != args.checkpoint_sha256 or sha(meta_path) != args.checkpoint_manifest_sha256:
            raise RuntimeError("checkpoint binding changed during diagnostic")
        manifest.update(lifecycle="DIAGNOSTIC_SEALED", prefix_completed=prefix_complete,
            final_phase=core.frame.state_id, endpoint_tick=core.frame.physics_tick,
            physical_seconds=core.frame.sim_time_s,
            original_task_reason=core.frame.info["semantic_task"].get("termination_reason"),
            physical_summary=physical.summary(), probe_state=wrapper.planner.state,
            probe_anchor=wrapper.planner.anchor, probe_release_reason=wrapper.planner.release_reason,
            learned_state_unchanged=True, same_episode_continued_after_probe=True)
    except BaseException as exc:
        manifest.update(lifecycle="DIAGNOSTIC_ERROR", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if unchanged is not None:
            try:
                unchanged()
                manifest["learned_state_unchanged"] = True
            except BaseException as integrity_error:
                manifest.update(learned_state_unchanged=False,
                    learned_state_verification_error=repr(integrity_error), lifecycle="DIAGNOSTIC_ERROR")
        if physical is not None:
            physical.close()
        if core is not None and core.frame is not None:
            manifest["last_episode_tick"] = core.frame.physics_tick
            manifest["last_simulation_time_s"] = core.frame.sim_time_s
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write(run_dir/"run_manifest.json", manifest)
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)
        lock.close()


if __name__ == "__main__":
    main()

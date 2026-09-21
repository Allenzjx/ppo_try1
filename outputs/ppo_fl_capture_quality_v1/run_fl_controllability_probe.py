"""One natural-P01, frozen-CP178432 diagnostic; never trains/saves checkpoints.

This isolated harness loads against the checkpoint's source contract, then
separately records actual code hashes. It does NOT loosen the production CLI
or claim a source-runtime/full-task evaluation. Old RR six configs must match.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fl_probe_helpers import FiniteFLHold, live_fl_jacobian, trigger_ready


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def now():
    return datetime.now(timezone.utc).isoformat()


def supervisor_ast_scope(old_commit):
    name = "src/wlr50_clean/ppo/semantic_supervisor.py"
    before = subprocess.check_output(["git", "show", old_commit+":"+name], cwd=ROOT, text=True, encoding="utf-8")
    after = (ROOT/name).read_text(encoding="utf-8")
    def symbols(source):
        result = {}
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef):
                # Every non-method class statement is separately bound too.
                for i, child in enumerate(node.body):
                    key = node.name+"."+(child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else "__statement_"+str(i))
                    result[key] = ast.dump(child, include_attributes=False)
            else:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = node.name
                elif isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name):
                    key = node.targets[0].id
                else:
                    key = "__unchanged_statement_"+ast.dump(node, include_attributes=False)
                result[key] = ast.dump(node, include_attributes=False)
        return result
    a,b=symbols(before),symbols(after)
    changed = sorted(k for k in a.keys()|b.keys() if a.get(k)!=b.get(k))
    allowed = {"FL_CAPTURE_APPROACH_MODE", "CAPTURE_APPROACH_MODES", "_capture_approach_enabled",
               "TaskStageSupervisor.physical_potential", "TaskStageSupervisor._current_capture_progress"}
    if not set(changed) <= allowed:
        raise ValueError("old-RR diagnostic rejects non-soft-reward supervisor changes: "+repr(changed))
    return changed


def check_source(checkpoint, allowed):
    metadata = json.loads(checkpoint.with_name(checkpoint.stem+"_manifest.json").read_text())
    if metadata["global_policy_decisions"] != 178432 or sha(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("this diagnostic is specifically bound to preserved CP178432")
    source = metadata["runtime_contract"]
    selected = source["selected_configuration"]
    for row in selected.values():
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("old RR selected config changed: "+row["path"])
    # Broad new-experiment/CLI files are not invoked by this standalone loop.
    permitted = {"src/wlr50_clean/ppo/"+name for name in (
        "semantic_reward.py", "semantic_supervisor.py", "semantic_training.py",
        "semantic_migration.py", "semantic_cli.py", "semantic_video.py", "semantic_video_cli.py")}
    if not set(allowed) <= permitted:
        raise ValueError("diagnostic cannot allow mapper/actor/backend/env/observation/physics changes")
    actual, differences = {}, {}
    for name, old in source["files"].items():
        if not name.startswith("src/"):
            continue
        digest = sha(ROOT / name)
        actual[name] = digest
        if digest != old:
            differences[name] = {"source_sha256": old, "actual_sha256": digest}
            if name not in allowed:
                raise ValueError("unreviewed source difference: "+name)
    scope = supervisor_ast_scope(source["source_git_commit"])
    return metadata, dict(source_git_commit=source["source_git_commit"], supervisor_AST_changed_symbols=scope,
        actual_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        explicitly_reviewed_code_differences=differences, actual_source_hashes=actual,
        selected_configuration=selected, unchanged_execution_paths=True,
        source_runtime_equivalence_claim=False,
        loader_scope="official source-checkpoint integrity load; separate diagnostic actual-runtime receipt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("hip_minus", "hip_plus", "hip_knee_combo"), required=True)
    parser.add_argument("--hip-degrees", type=float, choices=(1.5, 2.0), default=1.5,
                        help="Bounded single-factor diagnostic amplitude, never a nominal/policy setting")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"outputs/ppo_residual_rr_fix_v1/checkpoints/history/checkpoint_step_000178432.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=4001)
    parser.add_argument("--max-seconds", type=float, default=30.)
    parser.add_argument("--allow-reviewed-runtime-file", action="append", default=[])
    parser.add_argument("--review-receipt", type=Path)
    args = parser.parse_args()
    if not 26 <= args.max_seconds <= 35:
        raise ValueError("bounded diagnostics require 26..35 seconds")
    metadata, runtime = check_source(args.checkpoint.resolve(), args.allow_reviewed_runtime_file)
    if runtime["explicitly_reviewed_code_differences"]:
        if args.review_receipt is None:
            raise ValueError("changed code requires an explicit root-reviewed hash receipt")
        review = json.loads(args.review_receipt.read_text())
        if (review.get("approved_for_old_RR_diagnostic") is not True or
                review.get("actual_source_hashes") != {k: v["actual_sha256"] for k,v in runtime["explicitly_reviewed_code_differences"].items()}):
            raise ValueError("review does not bind exact changed source files")
        runtime["review_receipt"] = str(args.review_receipt.resolve())
        runtime["review_receipt_sha256"] = sha(args.review_receipt)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="wlr50_clean.FL_controllability_diagnostic.v1", lifecycle="RUNNING",
        started_at_utc=now(), diagnostic_only=True, PPO_training_or_formal_success=False,
        optimizer_updates=0, training_policy_decisions=0, case=args.case, seed=args.seed,
        hip_offset_magnitude_deg=args.hip_degrees,
        diagnostic_harness_sha256=sha(__file__),
        diagnostic_helper_sha256=sha(Path(__file__).with_name("fl_probe_helpers.py")),
        runtime=runtime, checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
        intervention="finite selected-FL-channel physical residual hold; other channels current frozen-policy conditional mean",
        from_phase="P01", physical_state_injection=False, source_N_changed=False,
        ramp_decisions=8, hold_decisions=32, release_decisions=8, follow_decisions=12,
        maximum_simulation_seconds=args.max_seconds, normal_training_mask=None)
    write(args.run_dir/"run_manifest.started.json", manifest)
    app = recorder = None
    try:
        # Same proven Windows DLL order as the production entrypoint.
        import torch
        from tensordict import TensorDict
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=False).app
        app.update()
        from wlr50_clean.ppo.semantic_backend import SemanticIsaacBackend
        from wlr50_clean.ppo.semantic_env import SemanticEpisodeEnv
        from wlr50_clean.ppo.semantic_legacy_evaluation import PhysicalEvaluationRecorder
        from wlr50_clean.ppo.semantic_training import (seed_training_rngs, construct_semantic_runner,
            load_semantic_checkpoint, parameter_hash, state_hash, _normalizers, jsonable)
        config = ROOT/"configs/ppo_residual_rr_fix_v1"
        seed_training_rngs(args.seed)
        backend = SemanticIsaacBackend(app, audit_actuator_target_effect=True,
            execution_profile=config/"execution_profile.yaml", task_spec_path=config/"stage_task_spec.yaml")
        core = SemanticEpisodeEnv(backend, collect_trace=False, action_config=config/"execution_profile.yaml",
            reward_config_path=config/"reward_config.yaml", observation_schema_path=config/"observation_schema.json")
        observation = core.reset(seed=args.seed)
        if core.frame.physics_tick != 0 or core.frame.state_id != "P01":
            raise ValueError("diagnostic reset is not natural P01")
        class ObservationEnv:
            num_envs, num_actions = 1, 12
            cfg = {"evaluation": True, "semantic_version": "v3"}
            def get_observations(self):
                tensor = torch.tensor([observation], dtype=torch.float32, device=args.device)
                return TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[1], device=args.device)
        policy_contract = metadata["policy_contract"]
        runner, _ = construct_semantic_runner(ObservationEnv(), seed=metadata["seed"], device=args.device,
            initialize_actor=False, policy_version=policy_contract["version"], observation_layout=policy_contract["observation_layout"])
        infos = load_semantic_checkpoint(runner, args.checkpoint.resolve(), contract=metadata["runtime_contract"], seed=metadata["seed"])
        runner.alg.eval_mode()
        def hashes():
            return dict(actor=parameter_hash(runner.alg.actor), critic=parameter_hash(runner.alg.critic),
                        optimizer=state_hash(runner.alg.optimizer.state_dict()), normalizer=state_hash(_normalizers(runner)))
        saved_hashes = hashes()
        manifest.update(loaded_policy_contract=policy_contract, loaded_state_hashes=saved_hashes,
                        actual_resume_source=infos["resume_source_checkpoint"])
        recorder = PhysicalEvaluationRecorder(args.run_dir, task_spec_path=config/"stage_task_spec.yaml",
                                             quality_score_path=config/"quality_score.yaml")
        recorder.start(core.frame)
        core.tick_observer = recorder.observe
        offsets = {"hip_minus": (-args.hip_degrees, 0.), "hip_plus": (args.hip_degrees, 0.),
                   "hip_knee_combo": (-args.hip_degrees, 1.)}[args.case]
        probe = FiniteFLHold(offsets)
        decisions = 0
        with (args.run_dir/"probe_decisions.jsonl").open("x", encoding="utf-8") as stream:
            while not core.done and core.frame.sim_time_s < args.max_seconds-1e-10:
                tensor = torch.tensor([observation], dtype=torch.float32, device=args.device)
                inputs = TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[1], device=args.device)
                with torch.inference_mode():
                    baseline = tuple(runner.alg.actor(inputs, stochastic_output=False)[0].cpu().tolist())
                task = core.frame.info["semantic_task"]
                evaluator = task["physical_evaluator"]
                endpoint = backend._controller.nominal_provider.endpoint_issued
                if probe.anchor is None and trigger_ready(phase=core.frame.state_id, endpoint=endpoint,
                    evaluator=evaluator, tick=core.frame.physics_tick):
                    geometry = live_fl_jacobian(backend)
                    if not 1.5 < geometry["world_xyz_mm_per_canonical_deg"][2][0] < 4.0:
                        raise ValueError("live FL hip vertical Jacobian differs materially from the sealed estimate; no probe issued")
                    probe.start(tick=core.frame.physics_tick, projected_residual=core._history["previous_residual_full12"])
                    manifest["probe_entry"] = dict(tick=core.frame.physics_tick, geometry=geometry,
                        current_FL=evaluator["current_legs"]["FL"], source_endpoint=endpoint,
                        anchor_deg=probe.anchor, nominal=list(core.frame.nominal_action_full12),
                        observation=list(observation), previous_actual_ACK=jsonable(backend._adapter.last_ack))
                    write(args.run_dir/"probe_entry.json", manifest["probe_entry"])
                receipt, raw = None, baseline
                if probe.anchor is not None:
                    # Actual phase caps; no mask/history/slew override.
                    caps = tuple(a*b for a,b in zip(core.projector.config.scale_for(core.frame.state_id),
                                                       core.projector.config.physical_residual_scale_full12))
                    placed = evaluator["history"]["placed"]["FL"]
                    raw, receipt = probe.action(baseline, caps=caps,
                        capture_or_leave=placed or core.frame.state_id != "P05")
                before = core.frame.physics_tick
                step = core.step(raw)
                observation = tuple(step.observation)
                decisions += 1
                row = dict(start_tick=before, end_tick=core.frame.physics_tick, baseline_raw=baseline,
                    injected_raw=raw, intervention=receipt, step_info=jsonable(step.info))
                if probe.anchor is not None and probe.index in (8, 24, 40):
                    row["same_state_geometry_read_only"] = live_fl_jacobian(backend)
                stream.write(json.dumps(row, allow_nan=False)+"\n")
                stream.flush()
                if probe.complete:
                    break
        if hashes() != saved_hashes:
            raise RuntimeError("diagnostic changed learned or optimizer state")
        final = core.frame.info["semantic_task"]
        manifest.update(lifecycle="DIAGNOSTIC_SEALED", completed_at_utc=now(),
            actual_diagnostic_decisions=decisions, endpoint_tick=core.frame.physics_tick,
            episode_duration_s=core.frame.sim_time_s, final_phase=core.frame.state_id,
            natural_task_terminal=core.done, original_task_reason=final.get("termination_reason"),
            probe_started=probe.anchor is not None, probe_complete=probe.complete,
            external_diagnostic_time_limit=not core.done and not probe.complete,
            physical_summary=recorder.summary(), learned_state_unchanged=True)
    except BaseException as exc:
        manifest.update(lifecycle="DIAGNOSTIC_ERROR", completed_at_utc=now(), error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if recorder is not None:
            recorder.close()
        write(args.run_dir/"run_manifest.json", manifest)
        print(json.dumps({k: manifest.get(k) for k in ("lifecycle", "case", "endpoint_tick", "final_phase", "probe_started", "probe_complete", "original_task_reason")}), flush=True)
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    main()

"""Additive live video entry point; never called by the training CLI."""
from __future__ import annotations
import json
from pathlib import Path
import traceback
from datetime import datetime, timezone

from .semantic_cli import (parser, validate_request, runtime_contract,
                           _preflight_checkpoint, jsonable)
from .semantic_training import (construct_semantic_runner, load_semantic_checkpoint,
    seed_training_rngs, parameter_hash, state_hash, _normalizers, write_json)
from .semantic_video import (ROLES, require, capture_semantic_video,
                             validate_semantic_video_source, video_configuration)


def checkpoint_loader(args, contract):
    """Loads real saved state after refreshed tick-zero observation exists."""
    def load(observation):
        import torch
        from tensordict import TensorDict
        metadata = json.loads(args.checkpoint.with_name(args.checkpoint.stem+"_manifest.json").read_text())
        training_seed = int(metadata["seed"])
        class ObservationEnv:
            num_envs, num_actions = 1, 12
            cfg = {"evaluation": True, "semantic_version": args.semantic_version}
            def get_observations(self):
                tensor = torch.tensor([observation], dtype=torch.float32, device=args.device)
                return TensorDict({"policy": tensor, "critic": tensor.clone()},
                                  batch_size=[1], device=args.device)
        runner, _ = construct_semantic_runner(ObservationEnv(), seed=training_seed,
                                              device=args.device)
        infos = load_semantic_checkpoint(runner, args.checkpoint, contract=contract,
            seed=training_seed, migration=getattr(args,"_migration_record",None))
        runner.alg.eval_mode()
        def hashes():
            return {"actor_parameter_sha256": parameter_hash(runner.alg.actor),
                    "critic_parameter_sha256": parameter_hash(runner.alg.critic),
                    "optimizer_state_sha256": state_hash(runner.alg.optimizer.state_dict()),
                    "normalizer_state_sha256": state_hash(_normalizers(runner))}
        initial_hashes = hashes()
        def action(current, _decision):
            tensor = torch.tensor([current], dtype=torch.float32, device=args.device)
            inputs = TensorDict({"policy":tensor,"critic":tensor.clone()},
                                batch_size=[1], device=args.device)
            with torch.inference_mode():
                return tuple(runner.alg.actor(inputs,stochastic_output=False)[0].cpu().tolist())
        def unchanged():
            require(hashes()==initial_hashes, "video evaluation changed learned state")
        proof = {"checkpoint_loaded_and_verified":True,
                 "official_load_semantic_checkpoint":True,
                 "source":infos["resume_source_checkpoint"],
                 "saved_global_policy_decisions":infos["global_policy_decisions"],
                 "training_seed":training_seed, "video_seed":args.seed,
                 "parameter_hashes":initial_hashes, "optimizer_updates":0,
                 "migration":getattr(args,"_migration_record",None)}
        return action, proof, unchanged
    return load


def validate_video_args(args):
    validate_request(args)
    require(args.command=="eval" and args.seed==4001 and not args.headless,
            "video requires eval, locked seed 4001 and --no-headless")
    require(args.max_decisions==3000 and args.decisions is None,
            "video stops on common physical endpoint, not an arbitrary short window")
    require(args.num_envs == 1 and args.from_phase == "P01"
            and args.teacher_offset_decisions == 0 and not args.new_mdp_warm_start,
            "video requires one natural P01 episode without a prefix or warm start")
    return next(role for role,mode in ROLES.items() if mode==args.mode)


def build_video_core(app, *, role, semantic_version):
    """A control is unchanged; only B/C select semantic runtime configuration."""
    configs = video_configuration(semantic_version)
    if role == "A":
        from .isaac_fsm_backend import IsaacFSMBackend
        from .residual_direct_env import ResidualEpisodeEnv
        return ResidualEpisodeEnv(IsaacFSMBackend(app, audit_actuator_target_effect=True),
                                  collect_trace=False)
    from .semantic_backend import SemanticIsaacBackend
    from .semantic_env import SemanticEpisodeEnv
    backend_options = {"audit_actuator_target_effect": True}
    core_options = {"collect_trace": False}
    if semantic_version == "v3":
        backend_options.update(execution_profile=configs["execution_profile"],
                               task_spec_path=configs["task_spec_path"])
        core_options.update(action_config=configs["execution_profile"],
                            reward_config_path=configs["reward_config_path"],
                            observation_schema_path=configs["observation_schema_path"])
    return SemanticEpisodeEnv(SemanticIsaacBackend(app, **backend_options), **core_options)


def main(argv=None):
    args=parser().parse_args(argv)
    role=validate_video_args(args)
    contract_options = {"expected_head": args.expected_head,
                        "semantic_version": args.semantic_version}
    contract=runtime_contract(**contract_options)
    _preflight_checkpoint(args,contract)  # Strict contract before any native launch.
    args.run_dir.mkdir(parents=True,exist_ok=False)
    lifecycle={"schema":"wlr50_clean.semantic_video_run.v1","lifecycle":"RUNNING",
        "arguments":jsonable(vars(args)),"runtime_contract":contract,
        "started_at_utc":datetime.now(timezone.utc).isoformat(),"optimizer_updates":0}
    write_json(args.run_dir/"run_manifest.started.json",lifecycle)
    app=None
    try:
        # Preserve the proven Windows DLL import order.
        import torch
        import tensordict
        from isaaclab.app import AppLauncher
        app=AppLauncher(headless=False,enable_cameras=False).app
        app.update()
        seed_training_rngs(args.seed)
        core = build_video_core(app, role=role, semantic_version=args.semantic_version)
        source=args.run_dir/"source"
        result=capture_semantic_video(core,role=role,seed=args.seed,
            output_directory=source,contract=contract,
            policy_loader=checkpoint_loader(args,contract) if role=="C" else None,
            semantic_version=args.semantic_version)
        require(runtime_contract(**contract_options)==contract,
                "runtime changed during capture")
        if result["success_candidate"]:
            validate_semantic_video_source(source,expected_role=role)
            status="SUCCEEDED"
        else:
            status="DIAGNOSTIC_FAILURE"
        lifecycle.update(lifecycle=status,result=result,
                         completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.run_dir/"run_manifest.json",lifecycle)
        print(json.dumps({"semantic_video_run":str(args.run_dir),"lifecycle":status}),flush=True)
        return 0 if status=="SUCCEEDED" else 2
    except BaseException as exc:
        lifecycle.update(lifecycle="FAILED",error=str(exc),traceback=traceback.format_exc(),
                         completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.run_dir/"run_manifest.json",lifecycle)
        raise
    finally:
        # Native close may not return. All recorder/manifests finalized first.
        if app is not None:
            app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=="__main__":
    raise SystemExit(main())

"""One reviewed 372->389 capture-MDP boundary; no optimizer or mean reset.

The CPU publisher verifies source tensors, adds zero first-layer columns and
zero Adam moment columns, then uses the official saver and real reload. It
does not generate physical observations, actions, rollouts or learning credit.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess

from .semantic_p05_capture_profile import (
    P05_CAPTURE_POLICY, P05_CAPTURE_OBSERVATION_LAYOUT, P05_CAPTURE_OBSERVATION_DIM,
)

SCHEMA = "wlr50_clean.p05_capture_assist_migration.v1"
SOURCE_EXPERIMENT = "task_conditioned_hip_wheel_v1"
TARGET_EXPERIMENT = "p05_hip_only_continuation_v1"
REVIEWABLE_CODE = frozenset(f"src/wlr50_clean/ppo/{name}.py" for name in (
    "semantic_p05_capture_profile", "semantic_p05_capture_actor", "semantic_p05_capture_migration",
    "semantic_capture_assist", "semantic_observation", "semantic_policy_distribution",
    "semantic_training", "semantic_migration", "semantic_checkpoint_prefix_policy",
    "semantic_checkpoint_prefix", "semantic_cli", "semantic_video_cli", "semantic_video",
    "semantic_supervisor", "semantic_residual_adapter", "semantic_backend", "semantic_env",
    "actuator_target_effect",
)) | {"scripts/run_semantic_ppo.ps1", "scripts/run_semantic_video.ps1", "scripts/export_p05_capture_video.py"}


def build_p05_capture_migration(checkpoint, current_contract, *, reason, reviewed_code_sha256,
                                project_root=None):
    from .semantic_migration import PROJECT_ROOT, _contract, checkpoint_metadata, digest, file_sha, source_num_envs, _version_bytes
    from .semantic_policy_distribution import policy_contract, policy_version_from_metadata, CONFIG_NAMES
    from .semantic_receiving_wheel_profile import RECEIVING_WHEEL_POLICY
    from .semantic_observation import load_semantic_observation_schema
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    import yaml
    root = Path(project_root or PROJECT_ROOT).resolve()
    checkpoint = Path(checkpoint).resolve(strict=True)
    metadata = checkpoint_metadata(checkpoint)
    old,new = _contract(metadata["runtime_contract"]),_contract(current_contract)
    if (old.get("experiment_id") != SOURCE_EXPERIMENT or new.get("experiment_id") != TARGET_EXPERIMENT
            or policy_version_from_metadata(metadata) != RECEIVING_WHEEL_POLICY or source_num_envs(metadata) != 1
            or metadata["policy_contract"]["observation_layout"] != ROLE_OBSERVATION_LAYOUT):
        raise ValueError("P05 capture migration requires the existing receiving-wheel N1 role372 source")
    if not isinstance(reason,str) or not reason.strip():
        raise ValueError("P05 capture migration requires its explicit review reason")
    variable = {"source_git_commit","runtime_content_sha256","files","selected_configuration","experiment_id"}
    if {k:v for k,v in old.items() if k not in variable} != {k:v for k,v in new.items() if k not in variable}:
        raise ValueError("P05 migration cannot change frozen physics, runtime, rates, budgets or global horizon")
    head = subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
    if new["source_git_commit"] != head:
        raise ValueError("P05 target contract must name actual committed runtime HEAD")
    if set(old["files"])-set(new["files"]):
        raise ValueError("P05 migration cannot delete runtime or old baseline files")
    delta = sorted(p for p,h in new["files"].items() if old["files"].get(p) != h)
    selected = new.get("selected_configuration",{})
    if set(selected) != CONFIG_NAMES or set(old.get("selected_configuration",{})) != CONFIG_NAMES:
        raise ValueError("P05 migration requires six exact selected configuration bindings")
    new_configs = {f"configs/ppo_{TARGET_EXPERIMENT}/{name}" for name in CONFIG_NAMES}
    code_delta = set(delta)-new_configs
    if not code_delta <= REVIEWABLE_CODE or set(reviewed_code_sha256) != code_delta:
        raise ValueError("P05 migration code must be an exact reviewed bounded implementation inventory")
    for path in new["files"]:
        if file_sha(root/path) != new["files"][path]:
            raise ValueError("P05 target runtime file differs from its bound bytes: "+path)
    if any(reviewed_code_sha256[p] != new["files"][p] for p in code_delta):
        raise ValueError("P05 code review hashes differ from target bytes")
    for name in CONFIG_NAMES:
        row = selected[name]
        if row != {"path":f"configs/ppo_{TARGET_EXPERIMENT}/{name}","sha256":new["files"].get(f"configs/ppo_{TARGET_EXPERIMENT}/{name}")}:
            raise ValueError("P05 selected configuration namespace/hash differs")
    source_config = {}
    for name in CONFIG_NAMES:
        row = old["selected_configuration"][name]
        raw = _version_bytes(root,old,row["path"])
        if name in ("action_schema.json","quality_score.yaml","reward_config.yaml"):
            if raw != (root/selected[name]["path"]).read_bytes():
                raise ValueError("P05 first boundary preserves action capacity, quality/reward bytes: "+name)
        source_config[name] = json.loads(raw) if name.endswith(".json") else yaml.safe_load(raw)
    obs_path = root/selected["observation_schema.json"]["path"]
    schema = load_semantic_observation_schema(obs_path)
    target_obs = json.loads(obs_path.read_text(encoding="utf-8"))
    source_obs = source_config["observation_schema.json"]
    if (schema.observation_layout != P05_CAPTURE_OBSERVATION_LAYOUT
            or target_obs["feature_groups"][:-2] != source_obs["feature_groups"]
            or {k:v for k,v in target_obs.items() if k not in ("revision","capture_assist_features_version","feature_groups")}
            != {k:v for k,v in source_obs.items() if k not in ("revision","feature_groups")}):
        raise ValueError("P05 schema must preserve the complete old372 preprocessing and append explicit17")
    execution = yaml.safe_load((root/selected["execution_profile.yaml"]["path"]).read_text(encoding="utf-8"))
    if (execution.get("capture_assist_mode") != "p05_hip_only_continuation_v1"
            or {k:v for k,v in execution.items() if k not in ("revision","capture_assist_mode")} != {
            k:v for k,v in source_config["execution_profile.yaml"].items() if k != "revision"}):
        raise ValueError("P05 execution boundary may add capture_assist only, not caps/masks/gains/physics")
    spec = yaml.safe_load((root/selected["stage_task_spec.yaml"]["path"]).read_text(encoding="utf-8"))
    if (spec.get("capture_continuation_semantics") != "p05_hip_only_continuation_v1"
            or {k:v for k,v in spec.items() if k != "capture_continuation_semantics"}
            != source_config["stage_task_spec.yaml"]):
        raise ValueError("P05 pending mode must be explicit and preserve the global task horizon")
    factor = {"schema":"wlr50_clean.p05_capture_observation_append17.v1",
        "source_policy_contract":metadata["policy_contract"],
        "target_policy_contract":policy_contract(P05_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT),
        "source_observation_dimension":372,"target_observation_dimension":P05_CAPTURE_OBSERVATION_DIM,
        "source_observation_layout":ROLE_OBSERVATION_LAYOUT,"target_observation_layout":P05_CAPTURE_OBSERVATION_LAYOUT,
        "reviewed_code_sha256":dict(reviewed_code_sha256),
        "first_layer_parameter":"mlp.0.weight","appended_columns":[372,P05_CAPTURE_OBSERVATION_DIM],
        "parameter_mapping":"zero_append_actor_and_critic_first_layer_only",
        "optimizer_mapping":"preserve_Adam_steps_groups_LR_and_existing_moments_zero_append_input_columns",
        "source_effective_learning_rate":metadata["optimizer_learning_rate"],
        "normalizers":"preserve_Identity", "training_rng":"preserve_exact_source_state",
        "counter_origin":{k:metadata[k] for k in ("global_policy_decisions","ppo_updates","optimizer_steps")},
        "physical_mdp_changed":True,"physical_assets_or_capacity_changed":False,
        "ordinary_P05_local_timeout_done":False,"global_timeout_s":200.0,
        "added_policy_decisions":0,"added_ppo_updates":0,"added_optimizer_steps":0,"added_auxiliary_updates":0,
        "old_rollout_inherited":False,"physical_state_inherited":False}
    return {"schema":SCHEMA,"reason":reason.strip(),"source_checkpoint":str(checkpoint),
        "source_checkpoint_sha256":file_sha(checkpoint),
        "source_manifest_sha256":file_sha(checkpoint.with_name(checkpoint.stem+"_manifest.json")),
        "source_contract_sha256":digest(old),"target_contract_sha256":digest(new),
        "source_git_commit":old["source_git_commit"],"target_git_commit":new["source_git_commit"],
        "source_runtime_content_sha256":old["runtime_content_sha256"],"target_runtime_content_sha256":new["runtime_content_sha256"],
        "allowed_changed_files":delta,
        "changed_file_hashes":{p:{"before":old["files"].get(p),"after":new["files"][p]} for p in delta},
        "observation_dimension":P05_CAPTURE_OBSERVATION_DIM,"action_dimension":12,
        "discard_old_rollout_storage":True,"p05_capture_assist_factor":factor}


def validate_p05_capture_migration(checkpoint,current_contract,plan_path,*,project_root=None):
    from .semantic_migration import file_sha
    path = Path(plan_path).resolve(strict=True)
    supplied = json.loads(path.read_text(encoding="utf-8"))
    expected = build_p05_capture_migration(checkpoint,current_contract,reason=supplied.get("reason"),
        reviewed_code_sha256=supplied.get("p05_capture_assist_factor",{}).get("reviewed_code_sha256",{}),
        project_root=project_root)
    if supplied != expected:
        raise ValueError("P05 migration plan differs from immutable source and current runtime")
    return {**expected,"plan_path":str(path),"plan_sha256":file_sha(path)}


def zero_append_training_state(bundle, *, target_dimension=P05_CAPTURE_OBSERVATION_DIM):
    """Pure tensor mapping; no source mutation and no parameter-order guesses."""
    import torch
    if target_dimension != P05_CAPTURE_OBSERVATION_DIM:
        raise ValueError("only the explicit P05 append17 boundary is supported")
    result = copy.deepcopy(bundle)
    parameter_shapes = []
    for key in ("actor_state_dict","critic_state_dict"):
        state = result[key]
        expected = ["mlp.0.weight","mlp.0.bias","mlp.2.weight","mlp.2.bias","mlp.4.weight","mlp.4.bias"]
        if list(state) != expected or tuple(state["mlp.0.weight"].shape) != (256,372):
            raise ValueError("append17 requires the exact six-parameter Identity MLP state")
        parameter_shapes.extend(tuple(value.shape) for value in state.values())
        weight = state["mlp.0.weight"]
        state["mlp.0.weight"] = torch.cat((weight,weight.new_zeros((256,target_dimension-372))),dim=1)
    optimizer = result["optimizer_state_dict"]
    groups = optimizer["param_groups"]
    if len(groups) != 1 or groups[0]["params"] != list(range(12)) or set(optimizer["state"]) != set(range(12)):
        raise ValueError("append17 requires the verified official actor-then-critic Adam parameter IDs")
    for ident,shape in enumerate(parameter_shapes):
        row = optimizer["state"][ident]
        if set(row) != {"step","exp_avg","exp_avg_sq"} or row["step"].numel() != 1:
            raise ValueError("unsupported or incomplete Adam state")
        for name in ("exp_avg","exp_avg_sq"):
            value = row[name]
            if tuple(value.shape) != shape:
                raise ValueError("Adam parameter mapping differs from the source actor/critic order")
            if shape == (256,372):
                row[name] = torch.cat((value,value.new_zeros((256,target_dimension-372))),dim=1)
    return result


class _ObservationOnlyEnv:
    """CPU shape contract, never a simulator or source of training samples."""
    num_envs=1
    num_actions=12
    max_episode_length=3000
    device="cpu"
    def __init__(self,dimension):
        import torch
        self.dimension=dimension
        self.cfg={"semantic_version":"v3","reset_sampling":"P01_full_task_only_initial_version",
                  "migration_observation_only":True,"num_envs":1}
        self.episode_length_buf=torch.zeros(1,dtype=torch.long)
    def get_observations(self):
        import torch
        from tensordict import TensorDict
        x=torch.zeros(1,self.dimension);x[:,0]=1
        return TensorDict({"policy":x,"critic":x.clone()},batch_size=[1])
    def step(self,*args,**kwargs):
        raise RuntimeError("CPU migration never collects or credits transitions")


def load_p05_capture_migration(runner,checkpoint,*,contract,seed,record):
    import torch
    from .semantic_migration import checkpoint_metadata, continuation_topology
    from .semantic_training import (construct_semantic_runner,load_semantic_checkpoint,parameter_hash,state_hash,
        _normalizers,_runner_policy_contract)
    from .rl_library_wrapper import optimizer_learning_rate,restore_training_rng_state
    from .semantic_receiving_wheel_profile import RECEIVING_WHEEL_POLICY
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    verified=validate_p05_capture_migration(checkpoint,contract,record["plan_path"])
    if verified != dict(record):
        raise RuntimeError("P05 migration binding changed after preflight")
    metadata=checkpoint_metadata(Path(checkpoint))
    if (seed != metadata["seed"] or runner._semantic_policy_version != P05_CAPTURE_POLICY
            or _runner_policy_contract(runner) != verified["p05_capture_assist_factor"]["target_policy_contract"]
            or tuple(runner.alg.storage.actions.shape)!=(128,1,12)
            or any(tuple(runner.alg.storage.observations[k].shape)!=(128,1,P05_CAPTURE_OBSERVATION_DIM) for k in ("policy","critic"))
            or runner.alg.storage.step!=0 or runner.alg.transition.actions is not None):
        raise RuntimeError("P05 append requires source seed and fresh exact target N1 storage")
    source,_=construct_semantic_runner(_ObservationOnlyEnv(372),seed=seed,device="cpu",
        policy_version=RECEIVING_WHEEL_POLICY,observation_layout=ROLE_OBSERVATION_LAYOUT,initialize_actor=False)
    infos=load_semantic_checkpoint(source,Path(checkpoint),contract=metadata["runtime_contract"],seed=seed)
    bundle=torch.load(checkpoint,map_location="cpu",weights_only=False)
    mapped=zero_append_training_state(bundle)
    for role in ("actor","critic"):
        model=getattr(runner.alg,role)
        if type(model.obs_normalizer) is not torch.nn.Identity:
            raise RuntimeError("P05 capture append cannot change Identity normalization")
        model.load_state_dict(mapped[role+"_state_dict"],strict=True)
    runner.alg.optimizer.load_state_dict(mapped["optimizer_state_dict"])
    runner.alg.learning_rate=metadata["optimizer_learning_rate"]
    runner.current_learning_iteration=bundle["iter"]
    if optimizer_learning_rate(runner)!=metadata["optimizer_learning_rate"]:
        raise RuntimeError("P05 append changed actual source Adam LR")
    if state_hash(_normalizers(runner)) != metadata["normalizer_state_sha256"]:
        raise RuntimeError("P05 append changed Identity normalizer state")
    restore_training_rng_state(infos["training_rng_state"],expected_seed=seed)
    sampling="P01_full_task_only_initial_version"
    return {**infos,"runtime_contract":dict(contract),"runner_config":copy.deepcopy(runner._semantic_runner_config),
        "policy_contract":_runner_policy_contract(runner),"sampling":sampling,
        "execution_topology":continuation_topology(sampling,None,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT),
        "curriculum_epoch":{"reset_sampling":sampling,"prefix_request":None},
        "actor_parameter_sha256":parameter_hash(runner.alg.actor),"critic_parameter_sha256":parameter_hash(runner.alg.critic),
        "optimizer_state_sha256":state_hash(runner.alg.optimizer.state_dict()),
        "p05_capture_assist_migration":verified,
        "p05_capture_assist_branch":{"schema":"wlr50_clean.p05_capture_assist_branch.v1",
            "counter_origin":verified["p05_capture_assist_factor"]["counter_origin"],
            "source_checkpoint_sha256":verified["source_checkpoint_sha256"],"migration_added_updates":0},
        "old_rollout_inherited":False,"physical_env_state_saved":False,
        "resume_physics":"fresh_legal_P01_reset","control_assist_label":"PPO_PLUS_CAPTURE_ASSIST_PLUS_INHERITED_LIMITED_AUX"}


def publish_p05_capture_checkpoint(checkpoint,current_contract,plan_path,output_checkpoint):
    """CPU-only verified migration, official save and actual reload; zero credit."""
    from .semantic_migration import checkpoint_metadata
    from .semantic_training import construct_semantic_runner,save_semantic_checkpoint,load_semantic_checkpoint,state_hash,parameter_hash
    metadata=checkpoint_metadata(Path(checkpoint))
    verified=validate_p05_capture_migration(checkpoint,current_contract,plan_path)
    env=_ObservationOnlyEnv(P05_CAPTURE_OBSERVATION_DIM)
    runner,_=construct_semantic_runner(env,seed=metadata["seed"],device="cpu",
        policy_version=P05_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT,initialize_actor=False)
    infos=load_p05_capture_migration(runner,checkpoint,contract=current_contract,seed=metadata["seed"],record=verified)
    path,manifest=save_semantic_checkpoint(runner,Path(output_checkpoint),infos)
    fresh,_=construct_semantic_runner(_ObservationOnlyEnv(P05_CAPTURE_OBSERVATION_DIM),seed=metadata["seed"],device="cpu",
        policy_version=P05_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT,initialize_actor=False)
    loaded=load_semantic_checkpoint(fresh,path,contract=current_contract,seed=metadata["seed"])
    if (parameter_hash(fresh.alg.actor)!=parameter_hash(runner.alg.actor)
            or state_hash(fresh.alg.optimizer.state_dict())!=state_hash(runner.alg.optimizer.state_dict())):
        raise RuntimeError("P05 migrated checkpoint failed independent runner reload")
    return {"checkpoint":str(path),"manifest":str(manifest),"save_load_round_trip":True,
        "global_policy_decisions":loaded["global_policy_decisions"],"ppo_updates":loaded["ppo_updates"],
        "optimizer_steps":loaded["optimizer_steps"],"migration_added_policy_decisions":0,
        "migration_added_ppo_updates":0,"observation_dimension":P05_CAPTURE_OBSERVATION_DIM}

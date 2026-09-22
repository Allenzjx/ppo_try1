"""One same389 capture-feedback revision; exact tensors, fresh rollout only."""
from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import subprocess

from .semantic_p05_capture_profile import P05_CAPTURE_POLICY, P05_CAPTURE_OBSERVATION_LAYOUT

SCHEMA = "wlr50_clean.capture_feedback_semantics_same389.v1"
FACTOR_KEY = "capture_feedback_semantics_factor"
FEEDBACK_REVISION = "hold_to_air_progress_window_v2"
SOURCE_FEEDBACK_REVISION = "implicit_contact_gap_window_v1"
EXPERIMENT = "p05_hip_only_continuation_v1"
CAPTURE_CODE = "src/wlr50_clean/ppo/semantic_capture_assist.py"
MODULE = "src/wlr50_clean/ppo/semantic_capture_feedback_migration.py"
ALLOWED_FILES = frozenset({CAPTURE_CODE, MODULE,
    "src/wlr50_clean/ppo/semantic_migration.py", "src/wlr50_clean/ppo/semantic_training.py"})
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
PRESERVED = (*COUNTERS, "stage_requested_decisions", "new_mdp_origin_global_policy_decisions",
    "p05_capture_assist_branch", "task_conditioned_hip_wheel_branch")


def capture_feedback_factor(metadata, old, new, *, reason, reviewed_code_sha256):
    """Pure strict boundary check; the builder additionally verifies disk/git bytes."""
    from .semantic_policy_distribution import policy_contract, CONFIG_NAMES
    from .semantic_migration import source_num_envs
    canonical = policy_contract(P05_CAPTURE_POLICY, observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)
    if (not isinstance(reason, str) or not reason.strip()
            or metadata.get("semantic_version") != "v3" or source_num_envs(metadata) != 1
            or old.get("experiment_id") != EXPERIMENT or new.get("experiment_id") != EXPERIMENT
            or metadata.get("policy_contract") != canonical):
        raise ValueError("capture feedback requires an explicit same-profile v3 N1 P05 389 source")
    variable = {"files", "runtime_content_sha256", "source_git_commit"}
    if ({k:v for k,v in old.items() if k not in variable}
            != {k:v for k,v in new.items() if k not in variable}):
        raise ValueError("capture feedback cannot change any configuration, physics, rates, budgets or profile")
    if set(old["files"]) - set(new["files"]) or set(new["files"]) - set(old["files"]) - {MODULE}:
        raise ValueError("capture feedback may add only its dedicated migration module and delete nothing")
    delta = sorted(p for p in new["files"] if old["files"].get(p) != new["files"][p])
    if (CAPTURE_CODE not in delta or not set(delta) <= ALLOWED_FILES
            or dict(reviewed_code_sha256) != {p:new["files"][p] for p in delta}):
        raise ValueError("capture feedback requires exact reviewed code hashes, not a broad runtime waiver")
    selected = old.get("selected_configuration", {})
    if set(selected) != CONFIG_NAMES or any(
            row != {"path":f"configs/ppo_{EXPERIMENT}/{name}",
                    "sha256":old["files"].get(f"configs/ppo_{EXPERIMENT}/{name}")}
            for name,row in selected.items()):
        raise ValueError("capture feedback requires all six unchanged selected configuration bindings")
    if any(key not in metadata for key in PRESERVED):
        raise ValueError("capture feedback requires intact original P05/task/AUX lineage and counters")
    if any(type(metadata[k]) is not int or metadata[k] < 0 for k in COUNTERS):
        raise ValueError("capture feedback source counters are invalid")
    if metadata.get("capture_feedback_semantics_branch") is not None:
        raise ValueError("this one-time v1-to-v2 feedback boundary cannot be reapplied")
    rate = metadata.get("optimizer_learning_rate")
    if type(rate) not in (int,float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("capture feedback requires a valid source effective Adam LR")
    observation = {"source_policy_contract":canonical,"target_policy_contract":copy.deepcopy(canonical),
        "observation_layout":P05_CAPTURE_OBSERVATION_LAYOUT,"observation_dimension":389,
        "action_dimension":12,"num_envs":1,"parameter_mapping":"identity_all_parameters_and_buffers"}
    return {"schema":SCHEMA,"review_reason":reason.strip(),
        "source_feedback_revision":SOURCE_FEEDBACK_REVISION,"target_feedback_revision":FEEDBACK_REVISION,
        "reviewed_code_sha256":dict(reviewed_code_sha256),"observation_contract":observation,
        "source_effective_learning_rate":rate,"target_effective_learning_rate":rate,
        "preserved_source_metadata":{k:copy.deepcopy(metadata[k]) for k in PRESERVED},
        "counter_origin":{k:metadata[k] for k in COUNTERS},
        "controller_transition_semantics_changed":True,"same_mdp_claimed":False,
        "policy_kernel_changed":False,"reward_changed":False,"caps_changed":False,"sigma_changed":False,
        "physical_assets_changed":False,"parameter_mapping":"identity_all_parameters_and_buffers",
        "optimizer_mapping":"identity_all_Adam_moments_steps_groups_and_effective_LR",
        "normalizer_mapping":"identity_Identity","rng_mapping":"restore_exact_source_training_rng",
        "discard_old_rollout_storage":True,"physical_state_inherited":False,
        "added_policy_decisions":0,"added_ppo_updates":0,"added_optimizer_steps":0,"added_auxiliary_updates":0}


def _revision(raw):
    tree = ast.parse(raw)
    values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "CAPTURE_ASSIST_FEEDBACK_REVISION" for t in node.targets)]
    if len(values) > 1:
        raise ValueError("ambiguous capture feedback revision")
    return values[0] if values else None


def build_capture_feedback_migration(checkpoint, current_contract, *, reason, reviewed_code_sha256,
                                     project_root=None):
    from .semantic_migration import PROJECT_ROOT, checkpoint_metadata, _contract, _version_bytes, file_sha, digest
    root = Path(project_root or PROJECT_ROOT).resolve()
    checkpoint = Path(checkpoint).resolve(strict=True)
    metadata = checkpoint_metadata(checkpoint)
    old,new = _contract(metadata["runtime_contract"]),_contract(current_contract)
    factor = capture_feedback_factor(metadata,old,new,reason=reason,reviewed_code_sha256=reviewed_code_sha256)
    head = subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
    if new["source_git_commit"] != head:
        raise ValueError("capture feedback target must name actual committed runtime HEAD")
    for path,sha in new["files"].items():
        if file_sha(root/path) != sha:
            raise ValueError("capture feedback target runtime bytes differ: "+path)
    # All six configs remain byte-identical, including schema/action/reward and nominal.
    for binding in old["selected_configuration"].values():
        if _version_bytes(root,old,binding["path"]) != (root/binding["path"]).read_bytes():
            raise ValueError("capture feedback cannot alter configuration preprocessing or task rules")
    source = _version_bytes(root,old,CAPTURE_CODE)
    target = (root/CAPTURE_CODE).read_bytes()
    if _revision(source) is not None or _revision(target) != FEEDBACK_REVISION:
        raise ValueError("capture feedback requires exact implicit-v1 to explicit hold-to-air-v2 revision")
    delta = sorted(reviewed_code_sha256)
    return {"schema":SCHEMA,"reason":reason.strip(),"source_checkpoint":str(checkpoint),
        "source_checkpoint_sha256":file_sha(checkpoint),
        "source_manifest_sha256":file_sha(checkpoint.with_name(checkpoint.stem+"_manifest.json")),
        "source_contract_sha256":digest(old),"target_contract_sha256":digest(new),
        "source_git_commit":old["source_git_commit"],"target_git_commit":new["source_git_commit"],
        "source_runtime_content_sha256":old["runtime_content_sha256"],
        "target_runtime_content_sha256":new["runtime_content_sha256"],
        "allowed_changed_files":delta,
        "changed_file_hashes":{p:{"before":old["files"].get(p),"after":new["files"][p]} for p in delta},
        "observation_dimension":389,"action_dimension":12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget":True,
        "discard_old_rollout_storage":True,"physics_resume":"fresh_legal_P01_reset",FACTOR_KEY:factor}


def validate_capture_feedback_migration(checkpoint,current_contract,plan_path,*,project_root=None):
    from .semantic_migration import file_sha
    path = Path(plan_path).resolve(strict=True)
    supplied = json.loads(path.read_text(encoding="utf-8"))
    expected = build_capture_feedback_migration(checkpoint,current_contract,reason=supplied.get("reason"),
        reviewed_code_sha256=supplied.get(FACTOR_KEY,{}).get("reviewed_code_sha256",{}),project_root=project_root)
    if supplied != expected:
        raise ValueError("capture feedback plan differs from immutable source and target runtime")
    return {**expected,"plan_path":str(path),"plan_sha256":file_sha(path)}


def record_loaded_capture_feedback(runner,infos,verified):
    """Called only after strict official source load/hash/RNG verification."""
    import torch
    from .rl_library_wrapper import optimizer_learning_rate
    factor = verified[FACTOR_KEY]
    if (any(type(getattr(runner.alg,role).obs_normalizer) is not torch.nn.Identity for role in ("actor","critic"))
            or optimizer_learning_rate(runner) != factor["source_effective_learning_rate"]
            or any(infos.get(k) != v for k,v in factor["preserved_source_metadata"].items())
            or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None):
        raise RuntimeError("capture feedback changed source state or reused partial rollout")
    return {**infos,"capture_feedback_semantics_migration":copy.deepcopy(verified),
        "capture_feedback_semantics_branch":{"schema":SCHEMA,
            "feedback_revision":factor["target_feedback_revision"],
            "counter_origin":copy.deepcopy(factor["counter_origin"]),
            "source_checkpoint_sha256":verified["source_checkpoint_sha256"],"migration_added_updates":0}}


def capture_feedback_branch_counts(infos):
    """Independent feedback-v2 origin; never replace the original P05 origin."""
    result = dict(infos)
    if "capture_feedback_semantics_branch" not in result:
        return result
    for name in ("p05_capture_assist","capture_feedback_semantics"):
        origin = result[name+"_branch"]["counter_origin"]
        counts = {k:result[k]-origin[k] for k in COUNTERS}
        if any(type(v) is not int or v < 0 for v in counts.values()):
            raise RuntimeError("capture feedback lineage has invalid counter origin")
        result[name+"_branch_counts"] = counts
    return result


def publish_capture_feedback_checkpoint(checkpoint,current_contract,plan_path,output_checkpoint):
    """No simulator/credit; publish only after Isaac exits, on the source device.

    Source CUDA visibility and complete runner configuration remain unchanged.
    The existing strict loader verifies the source state plus the new factor.
    """
    from .semantic_p05_capture_migration import _ObservationOnlyEnv
    from .semantic_migration import checkpoint_metadata
    from .semantic_training import construct_semantic_runner,load_semantic_checkpoint,save_semantic_checkpoint
    metadata = checkpoint_metadata(Path(checkpoint))
    verified = validate_capture_feedback_migration(checkpoint,current_contract,plan_path)
    device = metadata["runner_config"]["device"]
    class SourceDeviceObservationOnlyEnv(_ObservationOnlyEnv):
        def __init__(self):
            super().__init__(389)
            self.device = device
            self.episode_length_buf = self.episode_length_buf.to(device)
        def get_observations(self):
            return super().get_observations().to(self.device)
    def make():
        return construct_semantic_runner(SourceDeviceObservationOnlyEnv(),seed=metadata["seed"],device=device,
            policy_version=P05_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner = make()
    infos = load_semantic_checkpoint(runner,Path(checkpoint),contract=current_contract,seed=metadata["seed"],migration=verified)
    infos.update(runtime_contract=dict(current_contract),resume_migration=verified,
        migration_publication_context="zero_update_no_environment; inherited training context describes source run",
        source_training_context={k:copy.deepcopy(metadata[k]) for k in (
            "stage","sampling","implemented_reset_sampling","phase_suffix_curriculum_implemented",
            "execution_topology","curriculum_epoch") if k in metadata},
        old_rollout_inherited=False)
    path,manifest = save_semantic_checkpoint(runner,Path(output_checkpoint),infos)
    fresh = make()
    loaded = load_semantic_checkpoint(fresh,path,contract=current_contract,seed=metadata["seed"])
    for key in ("actor_parameter_sha256","critic_parameter_sha256","optimizer_state_sha256",
                "normalizer_state_sha256","training_rng_state","runner_config",*PRESERVED):
        if loaded[key] != metadata[key]:
            raise RuntimeError("same389 publication changed source state: "+key)
    return {"checkpoint":str(path),"manifest":str(manifest),"save_load_round_trip":True,
        **{k:loaded[k] for k in COUNTERS},"migration_added_policy_decisions":0,"migration_added_ppo_updates":0,
        "migration_added_optimizer_steps":0,"migration_added_auxiliary_updates":0,
        "p05_capture_assist_branch_counts":loaded["p05_capture_assist_branch_counts"],
        "capture_feedback_semantics_branch_counts":loaded["capture_feedback_semantics_branch_counts"]}


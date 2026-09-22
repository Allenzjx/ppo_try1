"""One explicit RR workspace reward/scalar revision; identity state transfer.

No imports execute publication, optimization, or simulator operations.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import subprocess

from .semantic_p05_capture_profile import P05_CAPTURE_POLICY, P05_CAPTURE_OBSERVATION_LAYOUT

SCHEMA = "wlr50_clean.rr_postcross_workspace_same389.v1"
FACTOR_KEY = "rr_postcross_workspace_factor"
MODE_KEY = "rr_postcross_workspace_semantics"
MODE = "current_qualified_RR_over_top_receiver_retirement_v1"
EXPERIMENT = "p05_hip_only_continuation_v1"
SUPERVISOR = "src/wlr50_clean/ppo/semantic_supervisor.py"
MODULE = "src/wlr50_clean/ppo/semantic_rr_workspace_migration.py"
TASK_SPEC = f"configs/ppo_{EXPERIMENT}/stage_task_spec.yaml"
ALLOWED_FILES = frozenset({SUPERVISOR, MODULE, TASK_SPEC,
    "src/wlr50_clean/ppo/semantic_migration.py", "src/wlr50_clean/ppo/semantic_training.py",
    "src/wlr50_clean/ppo/semantic_capture_feedback_migration.py"})
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
REQUIRED = (*COUNTERS, "stage_requested_decisions", "new_mdp_origin_global_policy_decisions",
    "p05_capture_assist_branch", "p05_capture_assist_migration",
    "capture_feedback_semantics_branch", "capture_feedback_semantics_migration",
    "task_conditioned_hip_wheel_branch", "actor_parameter_sha256", "critic_parameter_sha256",
    "optimizer_state_sha256", "normalizer_state_sha256", "normalization", "training_rng_state",
    "optimizer_learning_rate", "runner_config", "policy_contract")


def preserved_keys(metadata):
    return tuple(sorted(set(REQUIRED) | {key for key in metadata
        if (key != "resume_migration" and key.endswith(("_branch", "_branch_counts", "_migration")))
        or key in ("source_stage_requested_decisions", "training_quantity_budget_extension")}))


def rr_workspace_factor(metadata, old, new, *, reason, reviewed_code_sha256,
                        source_task_spec, target_task_spec):
    from .semantic_policy_distribution import policy_contract, CONFIG_NAMES
    from .semantic_migration import source_num_envs, digest
    canonical = policy_contract(P05_CAPTURE_POLICY, observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)
    if (not isinstance(reason, str) or not reason.strip()
            or metadata.get("semantic_version") != "v3" or source_num_envs(metadata) != 1
            or old.get("experiment_id") != EXPERIMENT or new.get("experiment_id") != EXPERIMENT
            or metadata.get("policy_contract") != canonical):
        raise ValueError("RR workspace retirement requires the exact same389 P05 v3 N1 profile")
    variable = {"files", "runtime_content_sha256", "source_git_commit", "selected_configuration"}
    if ({k:v for k,v in old.items() if k not in variable}
            != {k:v for k,v in new.items() if k not in variable}):
        raise ValueError("RR workspace retirement cannot alter physics, rates, budgets or runtime profile")
    if set(old["files"]) - set(new["files"]) or set(new["files"]) - set(old["files"]) - {MODULE}:
        raise ValueError("RR workspace retirement may add only its dedicated module and delete nothing")
    delta = sorted(path for path in new["files"] if old["files"].get(path) != new["files"][path])
    if (not {SUPERVISOR, TASK_SPEC}.issubset(delta) or not set(delta) <= ALLOWED_FILES
            or dict(reviewed_code_sha256) != {path:new["files"][path] for path in delta}):
        raise ValueError("RR workspace retirement needs exact reviewed hashes within its narrow scope")
    for contract in (old, new):
        selected = contract.get("selected_configuration", {})
        if set(selected) != CONFIG_NAMES or any(binding != {
                "path":f"configs/ppo_{EXPERIMENT}/{name}",
                "sha256":contract["files"].get(f"configs/ppo_{EXPERIMENT}/{name}")}
                for name,binding in selected.items()):
            raise ValueError("RR workspace retirement requires intact six-config bindings")
    if any(old["selected_configuration"][name] != new["selected_configuration"][name]
           for name in CONFIG_NAMES if name != "stage_task_spec.yaml"):
        raise ValueError("only the explicit task-spec opt-in may change; all other configs must be identical")
    target_without_mode = dict(target_task_spec)
    if (MODE_KEY in source_task_spec or target_without_mode.pop(MODE_KEY, None) != MODE
            or target_without_mode != dict(source_task_spec)):
        raise ValueError("task spec must differ only by the exact RR workspace opt-in")
    if (any(key not in metadata for key in REQUIRED)
            or metadata["capture_feedback_semantics_branch"].get("feedback_revision")
                != "hold_to_air_progress_window_v2"
            or metadata.get("rr_postcross_workspace_branch") is not None):
        raise ValueError("RR workspace retirement requires intact P05/feedback-v2/AUX lineage and no prior application")
    if any(type(metadata[key]) is not int or metadata[key] < 0 for key in COUNTERS):
        raise ValueError("RR workspace source counters must be actual nonnegative integers")
    rate = metadata["optimizer_learning_rate"]
    if type(rate) not in (int,float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("RR workspace source needs its actual finite positive Adam LR")
    return {"schema":SCHEMA, "review_reason":reason.strip(), "target_semantics":MODE,
        "source_semantics":"RR_receiver_preparation_share_remains_active_after_crossing",
        "reviewed_code_sha256":dict(reviewed_code_sha256),
        "task_spec_change":{MODE_KEY:MODE}, "source_task_spec_sha256":digest(source_task_spec),
        "target_task_spec_sha256":digest(target_task_spec),
        "observation_contract":{"source_policy_contract":canonical,
            "target_policy_contract":copy.deepcopy(canonical), "observation_layout":P05_CAPTURE_OBSERVATION_LAYOUT,
            "observation_dimension":389, "action_dimension":12, "num_envs":1,
            "parameter_mapping":"identity_all_parameters_and_buffers"},
        "observation_semantics_changed":[{"index":17, "field":"task_progress_potential",
            "change":"same physical state may encode corrected RR post-cross workspace potential"}],
        "observation_shape_changed":False, "same_numeric_input_policy_mapping_preserved":True,
        "same_physical_state_action_equivalence_claimed":False,
        "reward_changed":True, "same_mdp_claimed":False, "physical_dynamics_changed":False,
        "policy_kernel_changed":False, "controller_predicates_changed":False,
        "nominal_changed":False, "capture_assist_changed":False, "caps_changed":False, "sigma_changed":False,
        "reward_coefficients_and_return_profile_changed":False,
        "preserved_metadata_sha256":{key:digest(metadata[key]) for key in preserved_keys(metadata)},
        "counter_origin":{key:metadata[key] for key in COUNTERS},
        "source_effective_learning_rate":rate, "target_effective_learning_rate":rate,
        "parameter_mapping":"identity_all_parameters_and_buffers",
        "optimizer_mapping":"identity_all_Adam_moments_steps_groups_and_effective_LR",
        "normalizer_mapping":"identity_Identity", "rng_mapping":"restore_exact_source_training_rng",
        "critic_semantics":"preserve_all_parameters_and_Adam; recalibrate_only_from_fresh_corrected_reward_data",
        "old_values_are_new_reward_ground_truth":False, "discard_old_rollout_storage":True,
        "physical_state_inherited":False, "added_policy_decisions":0, "added_ppo_updates":0,
        "added_optimizer_steps":0, "added_auxiliary_updates":0}


def build_rr_workspace_migration(checkpoint, current_contract, *, reason, reviewed_code_sha256,
                                 project_root=None):
    import yaml
    from .semantic_migration import PROJECT_ROOT, checkpoint_metadata, _contract, _version_bytes, file_sha, digest
    root = Path(project_root or PROJECT_ROOT).resolve(); checkpoint = Path(checkpoint).resolve(strict=True)
    metadata = checkpoint_metadata(checkpoint)
    old,new = _contract(metadata["runtime_contract"]),_contract(current_contract)
    source_task = yaml.safe_load(_version_bytes(root,old,TASK_SPEC))
    target_task = yaml.safe_load((root/TASK_SPEC).read_bytes())
    factor = rr_workspace_factor(metadata,old,new,reason=reason,reviewed_code_sha256=reviewed_code_sha256,
        source_task_spec=source_task,target_task_spec=target_task)
    head = subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()
    if new["source_git_commit"] != head or new["source_git_commit"] == old["source_git_commit"]:
        raise ValueError("RR workspace target must name a new actual committed runtime HEAD")
    for path,sha in new["files"].items():
        if file_sha(root/path) != sha:
            raise ValueError("RR workspace target runtime bytes differ: "+path)
    for name,binding in old["selected_configuration"].items():
        if name != "stage_task_spec.yaml" and _version_bytes(root,old,binding["path"]) != (root/binding["path"]).read_bytes():
            raise ValueError("RR workspace changed another configuration's exact bytes")
    delta = sorted(reviewed_code_sha256)
    return {"schema":SCHEMA, "reason":reason.strip(), "source_checkpoint":str(checkpoint),
        "source_checkpoint_sha256":file_sha(checkpoint),
        "source_manifest_sha256":file_sha(checkpoint.with_name(checkpoint.stem+"_manifest.json")),
        "source_contract_sha256":digest(old), "target_contract_sha256":digest(new),
        "source_git_commit":old["source_git_commit"], "target_git_commit":new["source_git_commit"],
        "source_runtime_content_sha256":old["runtime_content_sha256"],
        "target_runtime_content_sha256":new["runtime_content_sha256"], "allowed_changed_files":delta,
        "changed_file_hashes":{p:{"before":old["files"].get(p),"after":new["files"][p]} for p in delta},
        "observation_dimension":389, "action_dimension":12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget":True,
        "discard_old_rollout_storage":True, "physics_resume":"fresh_legal_P01_reset", FACTOR_KEY:factor}


def validate_rr_workspace_migration(checkpoint,current_contract,plan_path,*,project_root=None):
    from .semantic_migration import file_sha
    path = Path(plan_path).resolve(strict=True); supplied = json.loads(path.read_text(encoding="utf-8"))
    expected = build_rr_workspace_migration(checkpoint,current_contract,reason=supplied.get("reason"),
        reviewed_code_sha256=supplied.get(FACTOR_KEY,{}).get("reviewed_code_sha256",{}),project_root=project_root)
    if supplied != expected:
        raise ValueError("RR workspace plan differs from immutable source and target")
    return {**expected,"plan_path":str(path),"plan_sha256":file_sha(path)}


def record_loaded_rr_workspace(runner,infos,verified):
    import torch
    from .rl_library_wrapper import optimizer_learning_rate
    from .semantic_migration import digest
    factor = verified[FACTOR_KEY]
    if (any(type(getattr(runner.alg,role).obs_normalizer) is not torch.nn.Identity for role in ("actor","critic"))
            or optimizer_learning_rate(runner) != factor["source_effective_learning_rate"]
            or any(key not in infos or digest(infos[key]) != value
                   for key,value in factor["preserved_metadata_sha256"].items())
            or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None):
        raise RuntimeError("RR workspace transfer changed source state or reused partial rollout")
    return {**infos, "rr_postcross_workspace_migration":copy.deepcopy(verified),
        "rr_postcross_workspace_branch":{"schema":SCHEMA,"semantics":MODE,
            "counter_origin":copy.deepcopy(factor["counter_origin"]),
            "source_checkpoint_sha256":verified["source_checkpoint_sha256"],"migration_added_updates":0}}


def rr_workspace_branch_counts(infos):
    result = dict(infos)
    if "rr_postcross_workspace_branch" in result:
        origin = result["rr_postcross_workspace_branch"]["counter_origin"]
        counts = {key:result[key]-origin[key] for key in COUNTERS}
        if any(type(value) is not int or value < 0 for value in counts.values()):
            raise RuntimeError("RR workspace lineage has invalid counter origin")
        result["rr_postcross_workspace_branch_counts"] = counts
    return result


def publish_rr_workspace_checkpoint(checkpoint,current_contract,plan_path,output_checkpoint):
    from .semantic_capture_feedback_migration import _publish_same389_identity_checkpoint
    from .semantic_migration import checkpoint_metadata
    metadata = checkpoint_metadata(Path(checkpoint))
    return _publish_same389_identity_checkpoint(checkpoint,current_contract,plan_path,output_checkpoint,
        validate_migration=validate_rr_workspace_migration, preserved_keys=preserved_keys(metadata),
        branch_count_keys=("p05_capture_assist_branch_counts","capture_feedback_semantics_branch_counts",
                           "rr_postcross_workspace_branch_counts"))

"""UNAPPLIED strict same410 v11 completed-source P05 recovery migration.

Publication adds no PPO/AUX credit.
"""
from __future__ import annotations
import ast
import copy
import json
import math
from pathlib import Path
import re
import subprocess

from .semantic_rr_capture_profile import RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_LAYOUT
from .semantic_rr_capture_reserve_migration import preserved_keys as prior_keys
from .semantic_rr_capture_knee_migration import _literal

SCHEMA = "wlr50_clean.p05_completed_source_same410.v11"
FACTOR_KEY = "p05_completed_source_v11_factor"
MIGRATION = "p05_completed_source_v11_migration"
PRIOR = "rr_capture_reserve_v10_migration"
PRIOR_FACTOR = "rr_capture_reserve_v10_factor"
EXPERIMENT = "rr_capture_then_rl_transfer_v1"
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
SOURCE_HEAD = "99dff5fd366e9cad80899176bf8b008e02112daf"
SOURCE_SHA = "069a71f547427b69491ff749dccc14fcf403565d81b3c37db3c6ac9b1e739e55"
SOURCE_MANIFEST_SHA = "314f4e25189d895f24c51c590dbe3aa284c20cf9de7a049fad526bd13b2054e5"
SOURCE_AUX_SHA = "52aad294be62450ec44fb3c28b4f20a0b3cf9d54b3cb653c23fddad5047ac815"
SOURCE_COUNTS = dict(zip(COUNTERS,(222720,1705,34100)))
ANCESTOR_ORIGIN = dict(zip(COUNTERS,(220544,1688,33760)))
SOURCE_BRANCH = "ancestor220544_signed_wheel_v8"
SOURCE_REVISION = "rr_capture_progress_reserve_v10"
TARGET_REVISION = "rr_capture_p05_post_endpoint_recovery_v11"
FEEDBACK = "progress_earned_capture_reserve_incremental_v10"
SOURCE_MODE = "p05_preedge_same_air_recross_v2"
TARGET_MODE = "p05_completed_source_first_approach_recross_v3"
SUPERVISOR = "src/wlr50_clean/ppo/semantic_supervisor.py"
PROFILE = f"configs/ppo_{EXPERIMENT}/execution_profile.yaml"
TASK = f"configs/ppo_{EXPERIMENT}/stage_task_spec.yaml"
MODULE = "src/wlr50_clean/ppo/semantic_p05_completed_source_migration.py"
ALLOWED_FILES = frozenset({SUPERVISOR,TASK,PROFILE,MODULE,
    "src/wlr50_clean/ppo/semantic_cli.py","src/wlr50_clean/ppo/semantic_training.py",
    "src/wlr50_clean/ppo/semantic_migration.py"})


def registered_source(sha):
    if sha!=SOURCE_SHA:
        raise ValueError("v11 requires the actual latest sealed v10 CP222720, not its AUX parent or older source")
    return dict(checkpoint_sha256=SOURCE_SHA,manifest_sha256=SOURCE_MANIFEST_SHA,
        source_git_commit=SOURCE_HEAD,counters=copy.deepcopy(SOURCE_COUNTS),
        source_role="latest_sealed_v10_branch_PPO",checkpoint_output_branch=SOURCE_BRANCH,
        branch_origin=copy.deepcopy(ANCESTOR_ORIGIN),
        front_retention_auxiliary_sha256=SOURCE_AUX_SHA)


def preserved_keys(metadata):
    return tuple(sorted(set(prior_keys(metadata)) | {PRIOR, "checkpoint_output_routing"}))


def _source_branch(metadata):
    route = metadata.get("checkpoint_output_routing")
    selection = (metadata.get("rr_progress_handoff_v5_migration") or {}).get("source_selection")
    if (not isinstance(route, dict) or route.get("schema") != "wlr50_clean.checkpoint_output_routing.v1"
            or route.get("branch") != SOURCE_BRANCH or route.get("main_latest_pointer_promotion") is not False
            or not isinstance(route.get("output_root"), str)
            or Path(route["output_root"]).name != SOURCE_BRANCH
            or Path(route["output_root"]).parent.name != "branches"
            or not isinstance(selection, dict) or selection.get("source_role") != "front_validated_ancestor_control_eval"
            or selection.get("counters") != ANCESTOR_ORIGIN or route.get("source_selection") != selection
            or metadata.get("rr_capture_transfer_branch", {}).get("counter_origin") != ANCESTOR_ORIGIN):
        raise ValueError("v11 requires unchanged explicit ancestor branch routing and original220544 origin")
    return route


def p05_completed_source_factor(metadata, old, new, *, reason, reviewed_code_sha256,
        source_profile, target_profile, source_task, target_task, source_binding, expected_target_head):
    from .semantic_migration import digest, source_num_envs
    from .semantic_policy_distribution import CONFIG_NAMES, policy_contract
    if source_binding != registered_source(source_binding.get("checkpoint_sha256")):
        raise ValueError("v11 immutable source binding differs")
    canonical = policy_contract(RR_CAPTURE_POLICY, observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
    if (not isinstance(reason, str) or not reason.strip() or not isinstance(expected_target_head, str)
            or re.fullmatch("[0-9a-f]{40}", expected_target_head) is None or expected_target_head == SOURCE_HEAD
            or old.get("source_git_commit") != SOURCE_HEAD or new.get("source_git_commit") != expected_target_head
            or metadata.get("checkpoint_sha256") != source_binding["checkpoint_sha256"]
            or any(type(metadata.get(k)) is not int or metadata[k] != source_binding["counters"][k] for k in COUNTERS)
            or metadata.get("semantic_version") != "v3" or source_num_envs(metadata) != 1
            or metadata.get("policy_contract") != canonical
            or any(c.get("experiment_id") != EXPERIMENT for c in (old, new))):
        raise ValueError("v11 requires exact latest registered learned v10 source and unchanged RR410 N1 policy")
    variable = {"files", "runtime_content_sha256", "source_git_commit", "selected_configuration"}
    if {k:v for k,v in old.items() if k not in variable} != {k:v for k,v in new.items() if k not in variable}:
        raise ValueError("v11 cannot change namespace, physics, rates or training budgets")
    changed = {p for p,h in new["files"].items() if old["files"].get(p) != h}
    if (set(old["files"])-set(new["files"]) or set(new["files"])-set(old["files"]) != {MODULE}
            or changed != ALLOWED_FILES
            or dict(reviewed_code_sha256) != {p:new["files"][p] for p in sorted(changed)}):
        raise ValueError("v11 requires exactly seven reviewed paths; historical validators remain unchanged")
    for contract in (old, new):
        selected = contract["selected_configuration"]
        if set(selected) != CONFIG_NAMES or any(v != {"path":f"configs/ppo_{EXPERIMENT}/{k}",
                "sha256":contract["files"].get(f"configs/ppo_{EXPERIMENT}/{k}")} for k,v in selected.items()):
            raise ValueError("v11 exact configuration/file bindings required")
    if (any(old["selected_configuration"][k] != new["selected_configuration"][k]
            for k in CONFIG_NAMES-{"execution_profile.yaml","stage_task_spec.yaml"})
            or source_profile.get("revision") != SOURCE_REVISION
            or source_profile.get("rr_capture_feedback_revision") != FEEDBACK
            or source_profile.get("rr_capture_wheel_mode") != "rr_capture_support_forward_projection_v1"
            or target_profile != {**source_profile, "revision":TARGET_REVISION}):
        raise ValueError("v11 permits only profile revision; RR feedback, reward, caps and codec remain fixed")
    expected_task=copy.deepcopy(source_task)
    if expected_task.get("nominal",{}).get("p05_preedge_approach_recovery")!=SOURCE_MODE:
        raise ValueError("v11 source must use the frozen P05 same-AIR v2 mode")
    expected_task["nominal"]["p05_preedge_approach_recovery"]=TARGET_MODE
    if target_task!=expected_task:
        raise ValueError("v11 task spec may change only the P05 recovery opt-in; no timeout/budget/physics/reward delta")
    if MIGRATION in metadata or any(k not in metadata for k in preserved_keys(metadata)):
        raise ValueError("v11 requires complete source ancestry, without a repeated v11 migration")
    route = _source_branch(metadata)
    prior = metadata[PRIOR]
    from .semantic_rr_capture_reserve_migration import validate_v10_branch_receipt
    validate_v10_branch_receipt(metadata,old,route)
    ledger=metadata.get("rr_capture_transfer_branch",{}).get("front_retention_auxiliary")
    if (digest(ledger) if ledger is not None else None) != source_binding.get("front_retention_auxiliary_sha256"):
        raise ValueError("v11 selected source current-branch front AUX ledger differs")
    for key,value in metadata.items():
        if key.endswith("_branch") and isinstance(value,dict) and "counter_origin" in value:
            origin = value["counter_origin"]
            if (set(origin) != set(COUNTERS) or any(type(origin[k]) is not int or not 0 <= origin[k] <= metadata[k] for k in COUNTERS)
                    or metadata.get(key+"_counts") != {k:metadata[k]-origin[k] for k in COUNTERS}):
                raise ValueError("v11 source branch counts differ: "+key)
    rate = metadata["optimizer_learning_rate"]
    if type(rate) not in (int,float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("v11 source effective LR must be finite positive")
    return {"schema":SCHEMA, "review_reason":reason.strip(), "source_selection":copy.deepcopy(source_binding),
        "observation_contract":{"source_policy_contract":canonical,"target_policy_contract":copy.deepcopy(canonical),
            "observation_layout":RR_CAPTURE_OBSERVATION_LAYOUT,"observation_dimension":410,"action_dimension":12,
            "num_envs":1,"parameter_mapping":"identity_all_parameters_and_buffers"},
        "source_feedback_revision":FEEDBACK,"target_feedback_revision":FEEDBACK,
        "observation_shape_changed":False,"observation_codec_changed":False,"capture_search_budget_changed":False,
        "observation_semantics_changed":[],
        "existing_observed_state_controls_changed_transform":
            "P05 completed-source recovery may begin at age0 instead of waiting until age30; all measured/source gates and original upper deadline remain",
        "effective_execution_semantics":{"source_P05_mode":SOURCE_MODE,"target_P05_mode":TARGET_MODE,
            "lower_age_v3_s":0.,"upper_age_semantics":"original_P05_maximum_task_duration_plus_existing_extension",
            "old_v1_v2_modes_unchanged":True,"source_endpoint_then_subsequent_tick_required":True,
            "fresh_source_wheel_owner_blocks_recovery":True,"same_AIR_recross_and_current_support_gates_preserved":True,
            "wheel_amplitudes_joint_targets_timeouts_unchanged":True,
            "RR_assist_and_v9_wheel_bytes_unchanged":True,"raw_Gaussian_is_not_transformed_target":True},
        "checkpoint_output_routing":{"preserved_sha256":digest(route),"same_existing_branch_only":True,
            "historical_ancestor_origin":copy.deepcopy(ANCESTOR_ORIGIN),"new_source_origin":copy.deepcopy(source_binding["counters"]),
            "main_latest_pointer_promotion_authorized":False,"network_reset_or_borrowed_credit":False},
        "legacy_action_transform_is_current_execution_semantics":False,"controller_transition_semantics_changed":True,
        "same_mdp_claimed":False,"same_numeric_input_policy_mapping_preserved":True,
        "same_physical_state_action_equivalence_claimed":False,
        **dict.fromkeys(("policy_kernel_changed","sigma_changed","caps_changed","reward_changed",
                        "physical_dynamics_changed","physical_task_acceptance_rules_changed"),False),
        "reviewed_code_sha256":dict(reviewed_code_sha256),
        "preserved_metadata_sha256":{k:digest(metadata[k]) for k in preserved_keys(metadata)},
        "source_effective_learning_rate":rate,"target_effective_learning_rate":rate,
        "source_v10_receipt_sha256":digest(prior),"source_resume_migration_sha256":digest(metadata.get("resume_migration")),
        "counter_origin":copy.deepcopy(source_binding["counters"]),"existing_branch_origins_preserved":True,"creates_new_branch":False,
        "discard_old_rollout_storage":True,"old_rollout_is_new_MDP_onpolicy":False,"physical_state_inherited":False,
        "parameter_mapping":"identity_all_parameters_and_buffers","optimizer_mapping":"identity_full_Adam_moments_steps_groups_and_effective_LR",
        "normalizer_mapping":"identity_Identity","rng_mapping":"restore_exact_source_training_rng",
        "candidate_evaluation_only":False,"ancestor_training_requires_explicit_output_branch":True,
        "latest_branch_learned_weights_preserved":True,"main_latest_policy_equivalence_claimed":False,
        "latest_pointer_promotion_authorized":False,
        **dict.fromkeys(("added_policy_decisions","added_ppo_updates","added_optimizer_steps","added_auxiliary_updates"),0)}


def validate_supervisor_scope(source,target):
    """Review-bind exactly the new mode plus two recovery functions, not task acceptance."""
    for name in ("P05_PREEDGE_RECOVERY_MODE","P05_SAME_AIR_RECROSS_MODE","P05_FINITE_RECOVERY_TIMEOUT_MODE"):
        if _literal(source,name)!=_literal(target,name):
            raise ValueError("v11 must preserve historical P05 mode constants: "+name)
    if _literal(target,"P05_COMPLETED_SOURCE_RECOVERY_MODE")!=TARGET_MODE:
        raise ValueError("v11 completed-source public mode differs")
    old,new=ast.parse(source),ast.parse(target)
    def functions(tree):
        result={}
        for node in tree.body:
            if isinstance(node,ast.FunctionDef): result[node.name]=node
            elif isinstance(node,ast.ClassDef):
                result.update({node.name+"."+child.name:child for child in node.body if isinstance(child,ast.FunctionDef)})
        return result
    before,after=functions(old),functions(new)
    allowed={"_p05_preedge_recovery_enabled","TaskStageSupervisor._p05_preedge_recovery_status"}
    changed={name for name in before if name not in after
        or ast.dump(before[name],include_attributes=False)!=ast.dump(after[name],include_attributes=False)}
    if set(before)!=set(after) or changed!=allowed:
        raise ValueError("v11 supervisor changes must be exactly opt-in validation and P05 recovery status")
    for node in new.body:
        if isinstance(node,ast.ClassDef):
            node.body=[copy.deepcopy(before[node.name+"."+child.name]) if isinstance(child,ast.FunctionDef)
                and node.name+"."+child.name in allowed else child for child in node.body]
    new.body=[copy.deepcopy(before[node.name]) if isinstance(node,ast.FunctionDef)
        and node.name in allowed else node for node in new.body]
    marker=[node for node in new.body if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=="P05_COMPLETED_SOURCE_RECOVERY_MODE" for t in node.targets)]
    if len(marker)!=1: raise ValueError("v11 must add one public opt-in constant, no hidden state")
    new.body.remove(marker[0])
    if ast.dump(old,include_attributes=False)!=ast.dump(new,include_attributes=False):
        raise ValueError("v11 cannot change any other supervisor/evaluator/potential/timeout code")


def validate_v11_branch_receipt(metadata, contract, route):
    """Validate newest receipt without falling back to the inherited v10 receipt."""
    from .semantic_migration import digest
    receipt = metadata.get(MIGRATION)
    if not isinstance(receipt,dict):
        raise ValueError("v11 branch receipt missing or malformed")
    factor = receipt.get(FACTOR_KEY, {})
    if not isinstance(factor,dict):
        raise ValueError("v11 branch factor malformed")
    selection = registered_source(receipt.get("source_checkpoint_sha256"))
    if (receipt.get("schema") != SCHEMA or factor.get("schema") != SCHEMA
            or metadata.get("policy_contract",{}).get("observation_dimension") != 410
            or receipt.get("source_selection") != selection
            or receipt.get("source_manifest_sha256") != selection["manifest_sha256"]
            or receipt.get("source_git_commit") != SOURCE_HEAD
            or receipt.get("target_git_commit") != contract.get("source_git_commit")
            or receipt.get("target_contract_sha256") != digest(contract)
            or receipt.get("target_runtime_content_sha256") != contract.get("runtime_content_sha256")
            or factor.get("counter_origin") != selection["counters"] or factor.get("target_feedback_revision") != FEEDBACK
            or factor.get("source_selection") != selection
            or factor.get("preserved_metadata_sha256",{}).get("checkpoint_output_routing") != digest(route)
            or factor.get("source_v10_receipt_sha256") != digest(metadata.get(PRIOR))
            or _source_branch(metadata) != route
            or any(type(metadata.get(k)) is not int or metadata[k] < selection["counters"][k] for k in COUNTERS)):
        raise ValueError("v11 branch receipt/runtime/source counts or inherited routing differs")
    return receipt


def build_p05_completed_source_migration(checkpoint,current_contract,*,expected_source_sha256,
        expected_target_head,reason,reviewed_code_sha256,project_root=None):
    import yaml
    from .semantic_migration import PROJECT_ROOT,checkpoint_metadata,_contract,_version_bytes,file_sha,digest
    binding=registered_source(expected_source_sha256);root=Path(project_root or PROJECT_ROOT).resolve()
    checkpoint=Path(checkpoint).resolve(strict=True);manifest=checkpoint.with_name(checkpoint.stem+"_manifest.json")
    if file_sha(checkpoint)!=binding["checkpoint_sha256"] or file_sha(manifest)!=binding["manifest_sha256"]:
        raise ValueError("v11 source checkpoint/manifest differ from the registered learned pair")
    metadata=checkpoint_metadata(checkpoint);old,new=_contract(metadata["runtime_contract"]),_contract(current_contract)
    value=p05_completed_source_factor(metadata,old,new,reason=reason,reviewed_code_sha256=reviewed_code_sha256,
        source_profile=yaml.safe_load(_version_bytes(root,old,PROFILE)),target_profile=yaml.safe_load((root/PROFILE).read_bytes()),
        source_task=yaml.safe_load(_version_bytes(root,old,TASK)),target_task=yaml.safe_load((root/TASK).read_bytes()),
        source_binding=binding,expected_target_head=expected_target_head)
    route=metadata["checkpoint_output_routing"]
    if checkpoint.parent != (Path(route["output_root"])/"checkpoints/history").resolve():
        raise ValueError("v11 migration source must remain in its registered immutable branch history")
    head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(root),"status","--porcelain=v1","--untracked-files=all","--",
        "src/wlr50_clean","scripts","configs","artifacts/ppo_phase_v1_start","pyproject.toml"],text=True).strip()
    if head!=expected_target_head or dirty: raise ValueError("v11 target must be the explicit clean committed runtime")
    if any(file_sha(root/p)!=h for p,h in new["files"].items()): raise ValueError("v11 target runtime bytes differ")
    for name,selected in old["selected_configuration"].items():
        if name not in ("execution_profile.yaml","stage_task_spec.yaml") and _version_bytes(root,old,selected["path"])!=(root/selected["path"]).read_bytes():
            raise ValueError("v11 protected configuration bytes changed")
    validate_supervisor_scope(_version_bytes(root,old,SUPERVISOR),(root/SUPERVISOR).read_bytes())
    return {"schema":SCHEMA,"reason":reason.strip(),"source_checkpoint":str(checkpoint),
        "source_checkpoint_sha256":binding["checkpoint_sha256"],"source_manifest_sha256":binding["manifest_sha256"],
        "source_selection":binding,"source_git_commit":SOURCE_HEAD,"target_git_commit":expected_target_head,
        "source_contract_sha256":digest(old),"target_contract_sha256":digest(new),
        "source_runtime_content_sha256":old["runtime_content_sha256"],"target_runtime_content_sha256":new["runtime_content_sha256"],
        "allowed_changed_files":sorted(reviewed_code_sha256),"observation_dimension":410,"action_dimension":12,
        "discard_old_rollout_storage":True,"physics_resume":"fresh_legal_P01_reset",FACTOR_KEY:value}


def validate_p05_completed_source_migration(checkpoint,current_contract,plan_path,*,project_root=None):
    from .semantic_migration import file_sha
    path=Path(plan_path).resolve(strict=True);supplied=json.loads(path.read_text(encoding="utf-8"))
    expected=build_p05_completed_source_migration(checkpoint,current_contract,
        expected_source_sha256=supplied.get("source_checkpoint_sha256"),expected_target_head=supplied.get("target_git_commit"),
        reason=supplied.get("reason"),reviewed_code_sha256=supplied.get(FACTOR_KEY,{}).get("reviewed_code_sha256",{}),project_root=project_root)
    if supplied!=expected: raise ValueError("v11 plan differs from the learned source and reviewed target")
    return {**expected,"plan_path":str(path),"plan_sha256":file_sha(path)}


def record_loaded_p05_completed_source(runner,infos,verified):
    from .semantic_training import _verify_reviewed_same410_identity_state
    if verified.get("schema")!=SCHEMA or verified.get(FACTOR_KEY,{}).get("schema")!=SCHEMA or MIGRATION in infos:
        raise RuntimeError("v11 receipt is absent, ambiguous or repeated")
    _verify_reviewed_same410_identity_state(runner,infos,verified[FACTOR_KEY])
    return {**infos,MIGRATION:copy.deepcopy(verified)}

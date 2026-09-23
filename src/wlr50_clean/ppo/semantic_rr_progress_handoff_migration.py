"""Strict same410 v5 control revision from two separately pinned v4 states."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import re
import subprocess

from .semantic_rr_capture_profile import RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_LAYOUT
from .semantic_rr_carry_handoff_migration import preserved_keys as prior_keys
from .semantic_rr_capture_knee_migration import _literal

SCHEMA = "wlr50_clean.rr_progress_handoff_same410.v5"
FACTOR_KEY = "rr_progress_handoff_v5_factor"
MIGRATION = "rr_progress_handoff_v5_migration"
PRIOR = "rr_carry_handoff_v4_migration"
PRIOR_FACTOR = "rr_carry_handoff_v4_factor"
EXPERIMENT = "rr_capture_then_rl_transfer_v1"
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
SOURCE_REVISION = "rr_capture_then_rl_transfer_v4_support_forward_contact_handoff"
TARGET_REVISION = "rr_capture_progress_handoff_v5"
SOURCE_FEEDBACK = "window_peak_hip_then_knee_v3"
TARGET_FEEDBACK = "progress_reserve_captured_incremental_v4"
SEARCH_SEMANTICS = "hip20_knee20_then_single_progress_earned_near_top_knee12_total52_exposure44_no_recharge_public_mode6_credit_captured_targets_add_issued_N_and_requested_residual_deltas_RL_current_TOP_retirement"
MODULE = "src/wlr50_clean/ppo/semantic_rr_progress_handoff_migration.py"
ASSIST = "src/wlr50_clean/ppo/semantic_rr_capture_assist.py"
PROFILE = f"configs/ppo_{EXPERIMENT}/execution_profile.yaml"
BRIDGE = "src/wlr50_clean/ppo/semantic_rr_carry_handoff_migration.py"
BRIDGE_SHA256 = "ac4dce5300534bb01dfddd1667275a29ba4d28dd18d5518aa7caa7d4e6286a61"
ALLOWED_FILES = frozenset({MODULE, ASSIST, PROFILE,
    "src/wlr50_clean/ppo/semantic_residual_adapter.py",
    "src/wlr50_clean/ppo/actuator_target_effect.py",
    "src/wlr50_clean/ppo/semantic_rr_capture_context.py",
    "src/wlr50_clean/ppo/semantic_migration.py",
    "src/wlr50_clean/ppo/semantic_training.py"})
SOURCE_REGISTRY = {
    "2fed192f9c5141a40f769893d0266ec3b95c91d55a4b3091acc411cd99672f90": {
        "manifest_sha256":"23080b6fc2e3ac0b4bbc80114c6b4f0211fef7a3f4a15b68ee76ac4c74b39bd1",
        "source_git_commit":"632295bc49cf56e120db426f106a58ee7e78c3ff",
        "counters":dict(zip(COUNTERS,(221184,1693,33860))),
        "source_role":"latest_learned_continuation", "registry_bridge_required":True},
    "facb915397f06251bf3e9c4f32cc2f2389262ff4c36f20c3a97b051bbe046892": {
        "manifest_sha256":"a10975d12695745ddfe6873fcc6213c9fb7ac2443957664914af598f68ac29ec",
        "source_git_commit":"57ae41eb4b57f63d4258e79b6083053166cc3cb3",
        "counters":dict(zip(COUNTERS,(220544,1688,33760))),
        "source_role":"front_validated_ancestor_control_eval", "registry_bridge_required":False},
}


def registered_source(checkpoint_sha256):
    if checkpoint_sha256 not in SOURCE_REGISTRY:
        raise ValueError("v5 source must be one of the two explicitly registered v4 checkpoints")
    return {"checkpoint_sha256":checkpoint_sha256, **copy.deepcopy(SOURCE_REGISTRY[checkpoint_sha256])}


def preserved_keys(metadata):
    return tuple(sorted(set(prior_keys(metadata)) | {PRIOR}))


def rr_progress_handoff_factor(metadata, old, new, *, reason, reviewed_code_sha256,
        source_profile, target_profile, source_binding, expected_target_head):
    from .semantic_migration import digest, source_num_envs
    from .semantic_policy_distribution import policy_contract, CONFIG_NAMES
    if source_binding != registered_source(source_binding.get("checkpoint_sha256")):
        raise ValueError("v5 source registration was changed")
    canonical = policy_contract(RR_CAPTURE_POLICY, observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
    if (not isinstance(reason,str) or not reason.strip()
            or not isinstance(expected_target_head,str) or re.fullmatch("[0-9a-f]{40}",expected_target_head) is None
            or expected_target_head == source_binding["source_git_commit"]
            or old.get("source_git_commit") != source_binding["source_git_commit"]
            or new.get("source_git_commit") != expected_target_head
            or metadata.get("checkpoint_sha256") != source_binding["checkpoint_sha256"]
            or any(type(metadata.get(k)) is not int or metadata[k] != source_binding["counters"][k] for k in COUNTERS)
            or metadata.get("semantic_version") != "v3" or source_num_envs(metadata) != 1
            or metadata.get("policy_contract") != canonical
            or old.get("experiment_id") != EXPERIMENT or new.get("experiment_id") != EXPERIMENT):
        raise ValueError("v5 requires its exact source bytes/counts/HEAD and unchanged same410 N1 policy")
    variable={"files","runtime_content_sha256","source_git_commit","selected_configuration"}
    if {k:v for k,v in old.items() if k not in variable} != {k:v for k,v in new.items() if k not in variable}:
        raise ValueError("v5 cannot change runtime physics, rates, budgets or namespace")
    changed={p for p,h in new["files"].items() if old["files"].get(p)!=h}
    expected_files=ALLOWED_FILES | ({BRIDGE} if source_binding["registry_bridge_required"] else set())
    if (set(old["files"])-set(new["files"]) or set(new["files"])-set(old["files"])!={MODULE}
            or changed!=expected_files or dict(reviewed_code_sha256)!={p:new["files"][p] for p in sorted(changed)}
            or new["files"].get(BRIDGE)!=BRIDGE_SHA256):
        raise ValueError("v5 requires its exact reviewed delta and immutable 57ae registry bridge")
    for contract in (old,new):
        selected=contract["selected_configuration"]
        if set(selected)!=CONFIG_NAMES or any(binding!={"path":f"configs/ppo_{EXPERIMENT}/{name}",
                "sha256":contract["files"].get(f"configs/ppo_{EXPERIMENT}/{name}")}
                for name,binding in selected.items()):
            raise ValueError("v5 requires all six exact configuration bindings")
    if (any(old["selected_configuration"][k]!=new["selected_configuration"][k]
            for k in CONFIG_NAMES-{"execution_profile.yaml"})
            or source_profile.get("revision")!=SOURCE_REVISION
            or source_profile.get("rr_capture_feedback_revision")!=SOURCE_FEEDBACK
            or source_profile.get("rr_capture_wheel_mode")!="rr_capture_support_forward_projection_v1"
            or target_profile!={**source_profile,"revision":TARGET_REVISION,"rr_capture_feedback_revision":TARGET_FEEDBACK}):
        raise ValueError("v5 changes only execution revision/feedback marker; all other configuration is protected")
    if MIGRATION in metadata or any(k not in metadata for k in preserved_keys(metadata)):
        raise ValueError("v5 requires complete unrepeated source state and lineage")
    prior=metadata[PRIOR];prior_factor=prior.get(PRIOR_FACTOR,{}) if isinstance(prior,dict) else {}
    if (not isinstance(prior,dict) or prior.get("schema")!="wlr50_clean.rr_carry_handoff_same410.v4"
            or prior.get("target_git_commit")!=old["source_git_commit"]
            or prior.get("target_contract_sha256")!=digest(old)
            or prior.get("target_runtime_content_sha256")!=old["runtime_content_sha256"]
            or prior_factor.get("counter_origin")!=source_binding["counters"]):
        raise ValueError("source v4 receipt must bind its exact zero-update publication and source runtime")
    ancestor=source_binding["source_role"]=="front_validated_ancestor_control_eval"
    selected=prior.get("source_selection")
    if ((ancestor and (not isinstance(selected,dict) or selected.get("source_role")!=source_binding["source_role"]
                      or selected.get("counters")!=source_binding["counters"]))
            or (not ancestor and selected is not None)):
        raise ValueError("source v4 candidate role is inconsistent; newer learned credit cannot be borrowed")
    for key,value in metadata.items():
        if key.endswith("_branch") and isinstance(value,dict) and "counter_origin" in value:
            origin=value["counter_origin"]
            if (set(origin)!=set(COUNTERS) or any(type(origin[k]) is not int or not 0<=origin[k]<=metadata[k] for k in COUNTERS)
                    or metadata.get(key+"_counts")!={k:metadata[k]-origin[k] for k in COUNTERS}):
                raise ValueError("v5 source branch counts are inconsistent: "+key)
    rate=metadata["optimizer_learning_rate"]
    if type(rate) not in (int,float) or not math.isfinite(rate) or rate<=0:
        raise ValueError("v5 requires a positive finite source effective LR")
    return {"schema":SCHEMA,"review_reason":reason.strip(),"source_selection":copy.deepcopy(source_binding),
        "observation_contract":{"source_policy_contract":canonical,"target_policy_contract":copy.deepcopy(canonical),
            "observation_layout":RR_CAPTURE_OBSERVATION_LAYOUT,"observation_dimension":410,"action_dimension":12,
            "num_envs":1,"parameter_mapping":"identity_all_parameters_and_buffers"},
        "source_feedback_revision":SOURCE_FEEDBACK,"target_feedback_revision":TARGET_FEEDBACK,
        "observation_shape_changed":False,"observation_codec_changed":False,
        "observation_semantics_changed":["public mode6 DESCEND_PROGRESS and mode7 CAPTURED_FOLLOW",
            "existing travel/exposure ranges extend to52deg/44s with unchanged20/12 scales",
            "captured targets accumulate issued nominal/requested residual deltas from public HISTORY",
            "context recovery/transfer flags reflect live progress and captured follow"],
        "effective_execution_semantics":{"search_semantics":SEARCH_SEMANTICS,
            "earned_reserve_degrees":12,"earned_reserve_exposure_s":12,"maximum_total_travel_deg":52,
            "maximum_total_exposure_s":44,"reserve_recharges":False,"RL_air_releases_capture":False,
            "delta_history":"current_issued_N_minus_prior_issued_N_plus_current_projected_request_minus_prior_issued_request",
            "raw_Gaussian_is_not_transformed_target":True},
        "legacy_action_transform_is_current_execution_semantics":False,
        "controller_transition_semantics_changed":True,"same_mdp_claimed":False,
        "same_numeric_input_policy_mapping_preserved":True,"same_physical_state_action_equivalence_claimed":False,
        **dict.fromkeys(("policy_kernel_changed","sigma_changed","caps_changed","reward_changed",
                        "physical_dynamics_changed","physical_task_acceptance_rules_changed"),False),
        "reviewed_code_sha256":dict(reviewed_code_sha256),
        "preserved_metadata_sha256":{k:digest(metadata[k]) for k in preserved_keys(metadata)},
        "source_effective_learning_rate":rate,"target_effective_learning_rate":rate,
        "source_v4_receipt_sha256":digest(prior),"source_resume_migration_sha256":digest(metadata.get("resume_migration")),
        "counter_origin":{k:metadata[k] for k in COUNTERS},"existing_branch_origins_preserved":True,
        "creates_new_branch":False,"discard_old_rollout_storage":True,"old_rollout_is_new_MDP_onpolicy":False,
        "parameter_mapping":"identity_all_parameters_and_buffers",
        "optimizer_mapping":"identity_full_Adam_moments_steps_groups_and_effective_LR",
        "normalizer_mapping":"identity_Identity","rng_mapping":"restore_exact_source_training_rng",
        "physical_state_inherited":False,"latest_pointer_promotion_authorized":False,
        "candidate_evaluation_only":ancestor,"latest_learned_weights_preserved":not ancestor,
        "latest_learned_policy_equivalence_claimed":False,
        **dict.fromkeys(("added_policy_decisions","added_ppo_updates","added_optimizer_steps","added_auxiliary_updates"),0)}


def validate_assist_semantics(source,target):
    for key in ("RR_CAPTURE_ASSIST_FEATURE_NAMES","_SCALES","RR_CAPTURE_ASSIST_MODE",
                "RR_CAPTURE_ASSIST_SCHEMA","RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS"):
        if _literal(source,key)!=_literal(target,key):
            raise ValueError("v5 may not change the 14-field codec/window contract: "+key)
    modes=_literal(source,"_MODES")
    if (len(_literal(target,"RR_CAPTURE_ASSIST_FEATURE_NAMES"))!=14
            or modes!=("WAIT","DESCEND","HOLD","BLOCKED","RELEASE","RELEASED")
            or _literal(target,"_MODES")!=modes+("DESCEND_PROGRESS","CAPTURED_FOLLOW")
            or _literal(source,"RR_CAPTURE_FEEDBACK_REVISION")!=SOURCE_FEEDBACK
            or _literal(target,"RR_CAPTURE_FEEDBACK_REVISION")!=TARGET_FEEDBACK
            or _literal(target,"RR_CAPTURE_SEARCH_SEMANTICS")!=SEARCH_SEMANTICS):
        raise ValueError("v5 public modes and control semantics do not match the reviewed revision")


def build_rr_progress_handoff_migration(checkpoint,current_contract,*,expected_source_sha256,
        expected_target_head,reason,reviewed_code_sha256,project_root=None):
    import yaml
    from .semantic_migration import PROJECT_ROOT,checkpoint_metadata,_contract,_version_bytes,file_sha,digest
    binding=registered_source(expected_source_sha256)
    root=Path(project_root or PROJECT_ROOT).resolve();checkpoint=Path(checkpoint).resolve(strict=True)
    manifest=checkpoint.with_name(checkpoint.stem+"_manifest.json")
    if file_sha(checkpoint)!=binding["checkpoint_sha256"] or file_sha(manifest)!=binding["manifest_sha256"]:
        raise ValueError("v5 source checkpoint/manifest differ from the registered immutable pair")
    metadata=checkpoint_metadata(checkpoint);old,new=_contract(metadata["runtime_contract"]),_contract(current_contract)
    value=rr_progress_handoff_factor(metadata,old,new,reason=reason,reviewed_code_sha256=reviewed_code_sha256,
        source_profile=yaml.safe_load(_version_bytes(root,old,PROFILE)),
        target_profile=yaml.safe_load((root/PROFILE).read_bytes()),source_binding=binding,expected_target_head=expected_target_head)
    head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(root),"status","--porcelain=v1","--untracked-files=all","--",
        "src/wlr50_clean","scripts","configs","artifacts/ppo_phase_v1_start","pyproject.toml"],text=True).strip()
    if head!=expected_target_head or dirty:
        raise ValueError("v5 target must be the explicit clean committed runtime")
    if any(file_sha(root/p)!=h for p,h in new["files"].items()):
        raise ValueError("v5 target runtime bytes differ from its contract")
    for name,selected in old["selected_configuration"].items():
        if name!="execution_profile.yaml" and _version_bytes(root,old,selected["path"])!=(root/selected["path"]).read_bytes():
            raise ValueError("v5 protected configuration byte content changed")
    validate_assist_semantics(_version_bytes(root,old,ASSIST),(root/ASSIST).read_bytes())
    return {"schema":SCHEMA,"reason":reason.strip(),"source_checkpoint":str(checkpoint),
        "source_checkpoint_sha256":binding["checkpoint_sha256"],"source_manifest_sha256":binding["manifest_sha256"],
        "source_selection":binding,"source_git_commit":old["source_git_commit"],"target_git_commit":expected_target_head,
        "source_contract_sha256":digest(old),"target_contract_sha256":digest(new),
        "source_runtime_content_sha256":old["runtime_content_sha256"],"target_runtime_content_sha256":new["runtime_content_sha256"],
        "allowed_changed_files":sorted(reviewed_code_sha256),"observation_dimension":410,"action_dimension":12,
        "discard_old_rollout_storage":True,"physics_resume":"fresh_legal_P01_reset",FACTOR_KEY:value}


def validate_rr_progress_handoff_migration(checkpoint,current_contract,plan_path,*,project_root=None):
    from .semantic_migration import file_sha
    path=Path(plan_path).resolve(strict=True);supplied=json.loads(path.read_text(encoding="utf-8"))
    expected=build_rr_progress_handoff_migration(checkpoint,current_contract,
        expected_source_sha256=supplied.get("source_checkpoint_sha256"),expected_target_head=supplied.get("target_git_commit"),
        reason=supplied.get("reason"),reviewed_code_sha256=supplied.get(FACTOR_KEY,{}).get("reviewed_code_sha256",{}),project_root=project_root)
    if supplied!=expected: raise ValueError("v5 plan differs from the bound source and reviewed target")
    return {**expected,"plan_path":str(path),"plan_sha256":file_sha(path)}


def record_loaded_rr_progress_handoff(runner,infos,verified):
    from .semantic_training import _verify_reviewed_same410_identity_state
    if verified.get("schema")!=SCHEMA or verified.get(FACTOR_KEY,{}).get("schema")!=SCHEMA or MIGRATION in infos:
        raise RuntimeError("v5 receipt is absent, ambiguous or repeated")
    _verify_reviewed_same410_identity_state(runner,infos,verified[FACTOR_KEY])
    return {**infos,MIGRATION:copy.deepcopy(verified)}

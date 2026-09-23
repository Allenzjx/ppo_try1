"""Exact same410 v7 signed AIR capture permission; no training on migration."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import re
import subprocess

from .semantic_rr_capture_profile import RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_LAYOUT
from .semantic_rr_contact_onset_migration import preserved_keys as prior_keys
from .semantic_rr_capture_knee_migration import _literal

SCHEMA = "wlr50_clean.rr_signed_contact_same410.v7"
FACTOR_KEY = "rr_signed_contact_v7_factor"
MIGRATION = "rr_signed_contact_v7_migration"
PRIOR = "rr_contact_onset_v6_migration"
PRIOR_FACTOR = "rr_contact_onset_v6_factor"
EXPERIMENT = "rr_capture_then_rl_transfer_v1"
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
SOURCE_HEAD = "97c4367ee293cb4dd191944ef663bbda63a5b235"
SOURCE_REVISION = "rr_capture_contact_onset_v6"
TARGET_REVISION = "rr_capture_signed_contact_formation_v7"
SOURCE_FEEDBACK = "progress_reserve_contact_onset_incremental_v5"
TARGET_FEEDBACK = "signed_band_contact_formation_incremental_v6"
SEARCH_SEMANTICS = "hip20_knee20_progress_earned_signed_task_band_knee12_then_1deg_public_peak_gap_le1mm_total53_exposure45_no_recharge_AIR_not_contact_sensor_TOP_unchanged_captured_issued_N_request_deltas_RL_current_TOP_retirement"
ASSIST = "src/wlr50_clean/ppo/semantic_rr_capture_assist.py"
PROFILE = f"configs/ppo_{EXPERIMENT}/execution_profile.yaml"
MODULE = "src/wlr50_clean/ppo/semantic_rr_signed_contact_migration.py"
ALLOWED_FILES = frozenset({ASSIST, PROFILE, MODULE,
    "src/wlr50_clean/ppo/semantic_cli.py", "src/wlr50_clean/ppo/semantic_rr_capture_context.py",
    "src/wlr50_clean/ppo/semantic_training.py", "src/wlr50_clean/ppo/semantic_migration.py"})
SOURCE_REGISTRY = {
    "0a61fb608210828ff9b3edb95701a25dd8232f2a8e0c207179ea5ae6bd142e01": {
        "manifest_sha256":"5672a9446485d09da35aae818dc010fc128d2334165194c2cad773d74ac3fc4c",
        "counters":dict(zip(COUNTERS,(220544,1688,33760))),
        "source_role":"front_validated_ancestor_control_eval"},
    "297d68b5fab18270903bce708ad1dc0062145da3fe876fa88316c998128e879f": {
        "manifest_sha256":"8631c3587d2b7c33b087a7bf8093bc4b0f3967e102f970cc5b1772b943a345d1",
        "counters":dict(zip(COUNTERS,(221184,1693,33860))),
        "source_role":"latest_learned_continuation"},
}


def registered_source(sha):
    if sha not in SOURCE_REGISTRY:
        raise ValueError("v7 requires an explicitly registered published 97c4367 v6 checkpoint")
    return {"checkpoint_sha256":sha,"source_git_commit":SOURCE_HEAD,**copy.deepcopy(SOURCE_REGISTRY[sha])}


def preserved_keys(metadata):
    return tuple(sorted(set(prior_keys(metadata)) | {PRIOR}
        | ({"checkpoint_output_routing"} if "checkpoint_output_routing" in metadata else set())))


def rr_signed_contact_factor(metadata,old,new,*,reason,reviewed_code_sha256,
        source_profile,target_profile,source_binding,expected_target_head):
    from .semantic_migration import digest,source_num_envs
    from .semantic_policy_distribution import CONFIG_NAMES,policy_contract
    if source_binding!=registered_source(source_binding.get("checkpoint_sha256")):
        raise ValueError("v7 source binding differs from its immutable registration")
    canonical=policy_contract(RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
    if (not isinstance(reason,str) or not reason.strip()
            or not isinstance(expected_target_head,str) or re.fullmatch("[0-9a-f]{40}",expected_target_head) is None
            or expected_target_head==SOURCE_HEAD or old.get("source_git_commit")!=SOURCE_HEAD
            or new.get("source_git_commit")!=expected_target_head
            or metadata.get("checkpoint_sha256")!=source_binding["checkpoint_sha256"]
            or any(type(metadata.get(k)) is not int or metadata[k]!=source_binding["counters"][k] for k in COUNTERS)
            or metadata.get("semantic_version")!="v3" or source_num_envs(metadata)!=1
            or metadata.get("policy_contract")!=canonical
            or any(c.get("experiment_id")!=EXPERIMENT for c in (old,new))):
        raise ValueError("v7 requires exact source identity/counts and unchanged RR410 N1 policy")
    variable={"files","runtime_content_sha256","source_git_commit","selected_configuration"}
    if {k:v for k,v in old.items() if k not in variable}!={k:v for k,v in new.items() if k not in variable}:
        raise ValueError("v7 cannot change namespace, physics, rates or training budgets")
    changed={p for p,h in new["files"].items() if old["files"].get(p)!=h}
    if (set(old["files"])-set(new["files"]) or set(new["files"])-set(old["files"])!={MODULE}
            or changed!=ALLOWED_FILES or dict(reviewed_code_sha256)!={p:new["files"][p] for p in sorted(changed)}):
        raise ValueError("v7 requires the exact seven reviewed paths, no historical validator relaxation")
    for contract in (old,new):
        selected=contract["selected_configuration"]
        if set(selected)!=CONFIG_NAMES or any(v!={"path":f"configs/ppo_{EXPERIMENT}/{k}",
                "sha256":contract["files"].get(f"configs/ppo_{EXPERIMENT}/{k}")} for k,v in selected.items()):
            raise ValueError("v7 requires exact configuration/file bindings")
    if (any(old["selected_configuration"][k]!=new["selected_configuration"][k] for k in CONFIG_NAMES-{"execution_profile.yaml"})
            or source_profile.get("revision")!=SOURCE_REVISION
            or source_profile.get("rr_capture_feedback_revision")!=SOURCE_FEEDBACK
            or source_profile.get("rr_capture_wheel_mode")!="rr_capture_support_forward_projection_v1"
            or target_profile!={**source_profile,"revision":TARGET_REVISION,"rr_capture_feedback_revision":TARGET_FEEDBACK}):
        raise ValueError("v7 profile may change only revision/feedback; all other config semantics stay fixed")
    if MIGRATION in metadata or any(k not in metadata for k in preserved_keys(metadata)):
        raise ValueError("v7 requires complete, unrepeated full source state and ancestry")
    prior=metadata[PRIOR]
    if (not isinstance(prior,dict) or prior.get("schema")!="wlr50_clean.rr_contact_onset_same410.v6"
            or prior.get("target_git_commit")!=SOURCE_HEAD or prior.get("target_contract_sha256")!=digest(old)
            or prior.get("target_runtime_content_sha256")!=old["runtime_content_sha256"]
            or prior.get(PRIOR_FACTOR,{}).get("counter_origin")!=source_binding["counters"]
            or prior.get(PRIOR_FACTOR,{}).get("target_feedback_revision")!=SOURCE_FEEDBACK
            or prior.get("source_selection",{}).get("source_role")!=source_binding["source_role"]
            or prior.get("source_selection",{}).get("counters")!=source_binding["counters"]):
        raise ValueError("v7 source v6 receipt must bind its actual zero-update publication and distinct selection")
    for key,value in metadata.items():
        if key.endswith("_branch") and isinstance(value,dict) and "counter_origin" in value:
            origin=value["counter_origin"]
            if (set(origin)!=set(COUNTERS) or any(type(origin[k]) is not int or not 0<=origin[k]<=metadata[k] for k in COUNTERS)
                    or metadata.get(key+"_counts")!={k:metadata[k]-origin[k] for k in COUNTERS}):
                raise ValueError("v7 source branch counts differ: "+key)
    rate=metadata["optimizer_learning_rate"]
    if type(rate) not in (int,float) or not math.isfinite(rate) or rate<=0:
        raise ValueError("v7 source effective LR must be finite positive")
    ancestor=source_binding["source_role"]=="front_validated_ancestor_control_eval"
    return {"schema":SCHEMA,"review_reason":reason.strip(),"source_selection":copy.deepcopy(source_binding),
        "observation_contract":{"source_policy_contract":canonical,"target_policy_contract":copy.deepcopy(canonical),
            "observation_layout":RR_CAPTURE_OBSERVATION_LAYOUT,"observation_dimension":410,"action_dimension":12,
            "num_envs":1,"parameter_mapping":"identity_all_parameters_and_buffers"},
        "source_feedback_revision":SOURCE_FEEDBACK,"target_feedback_revision":TARGET_FEEDBACK,
        "observation_shape_changed":False,"observation_codec_changed":False,"capture_search_budget_changed":False,
        "observation_semantics_changed":["existing RR recovery permission now includes configured signed AIR capture band; no contact/support/placement inference"],
        "effective_execution_semantics":{"search_semantics":SEARCH_SEMANTICS,
            "configured_capture_band_m":[-.015,.025], "AIR_is_not_contact_or_bearing":True,
            "maximum_total_travel_deg":53,"maximum_total_exposure_s":45,"signed_contact_extra_deg":1,
            "signed_contact_extra_active_s":1,"required_public_progress_peak_maximum_m":.001,
            "budget_recharges":False,"sensor_TOP_confirmation_unchanged":True,"RL_air_releases_capture":False,
            "raw_Gaussian_is_not_transformed_target":True},
        "checkpoint_output_routing":{"optional_ancestor_only":True,"default_output_behavior_changed":False,
            "first_parent_source_requires_published_v7":True,"same_branch_resume_and_pointer_verified":True,
            "main_latest_pointer_promotion_authorized":False,"network_reset_or_borrowed_credit":False},
        "legacy_action_transform_is_current_execution_semantics":False,"controller_transition_semantics_changed":True,
        "same_mdp_claimed":False,"same_numeric_input_policy_mapping_preserved":True,
        "same_physical_state_action_equivalence_claimed":False,
        **dict.fromkeys(("policy_kernel_changed","sigma_changed","caps_changed","reward_changed",
                        "physical_dynamics_changed","physical_task_acceptance_rules_changed"),False),
        "reviewed_code_sha256":dict(reviewed_code_sha256),
        "preserved_metadata_sha256":{k:digest(metadata[k]) for k in preserved_keys(metadata)},
        "source_effective_learning_rate":rate,"target_effective_learning_rate":rate,
        "source_v6_receipt_sha256":digest(prior),"source_resume_migration_sha256":digest(metadata.get("resume_migration")),
        "counter_origin":{k:metadata[k] for k in COUNTERS},"existing_branch_origins_preserved":True,"creates_new_branch":False,
        "discard_old_rollout_storage":True,"old_rollout_is_new_MDP_onpolicy":False,"physical_state_inherited":False,
        "parameter_mapping":"identity_all_parameters_and_buffers","optimizer_mapping":"identity_full_Adam_moments_steps_groups_and_effective_LR",
        "normalizer_mapping":"identity_Identity","rng_mapping":"restore_exact_source_training_rng",
        "candidate_evaluation_only":False,"ancestor_training_requires_explicit_output_branch":ancestor,
        "latest_learned_weights_preserved":not ancestor,
        "latest_learned_policy_equivalence_claimed":False,"latest_pointer_promotion_authorized":False,
        **dict.fromkeys(("added_policy_decisions","added_ppo_updates","added_optimizer_steps","added_auxiliary_updates"),0)}


def validate_assist_semantics(source,target):
    from .semantic_rr_contact_onset_migration import SEARCH_SEMANTICS as previous_search
    for key in ("RR_CAPTURE_ASSIST_FEATURE_NAMES","_SCALES","_MODES","RR_CAPTURE_ASSIST_MODE",
                "RR_CAPTURE_ASSIST_SCHEMA","RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS"):
        if _literal(source,key)!=_literal(target,key):
            raise ValueError("v7 cannot change public field/mode/codec/window positions: "+key)
    if (len(_literal(target,"RR_CAPTURE_ASSIST_FEATURE_NAMES"))!=14
            or _literal(source,"RR_CAPTURE_FEEDBACK_REVISION")!=SOURCE_FEEDBACK
            or _literal(target,"RR_CAPTURE_FEEDBACK_REVISION")!=TARGET_FEEDBACK
            or _literal(source,"RR_CAPTURE_SEARCH_SEMANTICS")!=previous_search
            or _literal(target,"RR_CAPTURE_SEARCH_SEMANTICS")!=SEARCH_SEMANTICS):
        raise ValueError("v7 capture semantics do not match the finite reviewed contact-onset revision")


def validate_signed_context(source,spec):
    if (_literal(source,"RR_CAPTURE_GAP_MIN_M") != -.015 or _literal(source,"RR_CAPTURE_GAP_MAX_M") != .025
            or spec["geometry"]["top_gap_min_m"] != -.015 or spec["geometry"]["top_gap_max_m"] != .025):
        raise ValueError("v7 signed permission must use unchanged physical task geometry, not a new contact tolerance")


def build_rr_signed_contact_migration(checkpoint,current_contract,*,expected_source_sha256,
        expected_target_head,reason,reviewed_code_sha256,project_root=None):
    import yaml
    from .semantic_migration import PROJECT_ROOT,checkpoint_metadata,_contract,_version_bytes,file_sha,digest
    binding=registered_source(expected_source_sha256);root=Path(project_root or PROJECT_ROOT).resolve()
    checkpoint=Path(checkpoint).resolve(strict=True);manifest=checkpoint.with_name(checkpoint.stem+"_manifest.json")
    if file_sha(checkpoint)!=binding["checkpoint_sha256"] or file_sha(manifest)!=binding["manifest_sha256"]:
        raise ValueError("v7 source checkpoint/manifest differ from the registered immutable pair")
    metadata=checkpoint_metadata(checkpoint);old,new=_contract(metadata["runtime_contract"]),_contract(current_contract)
    value=rr_signed_contact_factor(metadata,old,new,reason=reason,reviewed_code_sha256=reviewed_code_sha256,
        source_profile=yaml.safe_load(_version_bytes(root,old,PROFILE)),target_profile=yaml.safe_load((root/PROFILE).read_bytes()),
        source_binding=binding,expected_target_head=expected_target_head)
    head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(root),"status","--porcelain=v1","--untracked-files=all","--",
        "src/wlr50_clean","scripts","configs","artifacts/ppo_phase_v1_start","pyproject.toml"],text=True).strip()
    if head!=expected_target_head or dirty: raise ValueError("v7 target must be the explicit clean committed runtime")
    if any(file_sha(root/p)!=h for p,h in new["files"].items()): raise ValueError("v7 target runtime bytes differ")
    for name,selected in old["selected_configuration"].items():
        if name!="execution_profile.yaml" and _version_bytes(root,old,selected["path"])!=(root/selected["path"]).read_bytes():
            raise ValueError("v7 protected configuration bytes changed")
    validate_assist_semantics(_version_bytes(root,old,ASSIST),(root/ASSIST).read_bytes())
    validate_signed_context((root/"src/wlr50_clean/ppo/semantic_rr_capture_context.py").read_bytes(),
        yaml.safe_load((root/f"configs/ppo_{EXPERIMENT}/stage_task_spec.yaml").read_bytes()))
    return {"schema":SCHEMA,"reason":reason.strip(),"source_checkpoint":str(checkpoint),
        "source_checkpoint_sha256":binding["checkpoint_sha256"],"source_manifest_sha256":binding["manifest_sha256"],
        "source_selection":binding,"source_git_commit":SOURCE_HEAD,"target_git_commit":expected_target_head,
        "source_contract_sha256":digest(old),"target_contract_sha256":digest(new),
        "source_runtime_content_sha256":old["runtime_content_sha256"],"target_runtime_content_sha256":new["runtime_content_sha256"],
        "allowed_changed_files":sorted(reviewed_code_sha256),"observation_dimension":410,"action_dimension":12,
        "discard_old_rollout_storage":True,"physics_resume":"fresh_legal_P01_reset",FACTOR_KEY:value}


def validate_rr_signed_contact_migration(checkpoint,current_contract,plan_path,*,project_root=None):
    from .semantic_migration import file_sha
    path=Path(plan_path).resolve(strict=True);supplied=json.loads(path.read_text(encoding="utf-8"))
    expected=build_rr_signed_contact_migration(checkpoint,current_contract,
        expected_source_sha256=supplied.get("source_checkpoint_sha256"),expected_target_head=supplied.get("target_git_commit"),
        reason=supplied.get("reason"),reviewed_code_sha256=supplied.get(FACTOR_KEY,{}).get("reviewed_code_sha256",{}),project_root=project_root)
    if supplied!=expected: raise ValueError("v7 plan differs from bound source and reviewed target")
    return {**expected,"plan_path":str(path),"plan_sha256":file_sha(path)}


def record_loaded_rr_signed_contact(runner,infos,verified):
    from .semantic_training import _verify_reviewed_same410_identity_state
    if verified.get("schema")!=SCHEMA or verified.get(FACTOR_KEY,{}).get("schema")!=SCHEMA or MIGRATION in infos:
        raise RuntimeError("v7 receipt is absent, ambiguous or repeated")
    _verify_reviewed_same410_identity_state(runner,infos,verified[FACTOR_KEY])
    return {**infos,MIGRATION:copy.deepcopy(verified)}


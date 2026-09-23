"""Strict same410 v9 P12 retention migration from the sealed learned branch.

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
from .semantic_rr_signed_wheel_migration import preserved_keys as prior_keys
from .semantic_rr_capture_knee_migration import _literal

SCHEMA = "wlr50_clean.rr_postcapture_wheel_same410.v9"
FACTOR_KEY = "rr_postcapture_wheel_v9_factor"
MIGRATION = "rr_postcapture_wheel_v9_migration"
PRIOR = "rr_signed_wheel_v8_migration"
PRIOR_FACTOR = "rr_signed_wheel_v8_factor"
EXPERIMENT = "rr_capture_then_rl_transfer_v1"
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
SOURCE_HEAD = "d1871df37d6ea909657511d0e43e7435198f6ccd"
SOURCE_SHA = "746c3abb9aa6bfced2c194a45b7c1c9db2cf543a25ab1c033108233bce1c685a"
SOURCE_MANIFEST_SHA = "8e4caab3cc7a1bf66c0b1e41617413c693405367efa68f627a461d99f51a629e"
SOURCE_COUNTS = dict(zip(COUNTERS, (221952, 1699, 33980)))
ANCESTOR_ORIGIN = dict(zip(COUNTERS, (220544, 1688, 33760)))
SOURCE_BRANCH = "ancestor220544_signed_wheel_v8"
SOURCE_REVISION = "rr_capture_signed_wheel_continuity_v8"
TARGET_REVISION = "rr_capture_postcapture_wheel_retention_v9"
FEEDBACK = "signed_band_contact_formation_incremental_v6"
SOURCE_WHEEL_SEMANTICS = "P09_post_source_committed_stop_current_RR_AIR_Q_cross_supported_FL_FR_RL_signed_task_band_nonnegative_depth_gap_floor_previous_FINAL_1p8_slew_TOP_release"
WHEEL_SEMANTICS = "P09_v8_unchanged_P12_RR_placed_RL_unplaced_post_authored_wheel_stop_current_TOP_or_qualified_signed_AIR_bearing_only_depth_floor_previous_FINAL_1p8_slew"
WHEEL = "src/wlr50_clean/ppo/semantic_rr_carry_wheel.py"
PROFILE = f"configs/ppo_{EXPERIMENT}/execution_profile.yaml"
MODULE = "src/wlr50_clean/ppo/semantic_rr_postcapture_wheel_migration.py"
ALLOWED_FILES = frozenset({WHEEL, PROFILE, MODULE,
    "src/wlr50_clean/ppo/semantic_cli.py", "src/wlr50_clean/ppo/semantic_training.py",
    "src/wlr50_clean/ppo/semantic_migration.py"})


def registered_source(sha):
    if sha != SOURCE_SHA:
        raise ValueError("v9 requires the sealed learned v8 branch checkpoint221952, not its ancestor or main latest")
    return dict(checkpoint_sha256=SOURCE_SHA, manifest_sha256=SOURCE_MANIFEST_SHA,
        source_git_commit=SOURCE_HEAD, counters=copy.deepcopy(SOURCE_COUNTS),
        source_role="learned_ancestor_branch_continuation", checkpoint_output_branch=SOURCE_BRANCH,
        branch_origin=copy.deepcopy(ANCESTOR_ORIGIN))


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
        raise ValueError("v9 requires unchanged explicit ancestor branch routing and original220544 origin")
    return route


def rr_postcapture_wheel_factor(metadata, old, new, *, reason, reviewed_code_sha256,
        source_profile, target_profile, source_binding, expected_target_head):
    from .semantic_migration import digest, source_num_envs
    from .semantic_policy_distribution import CONFIG_NAMES, policy_contract
    if source_binding != registered_source(source_binding.get("checkpoint_sha256")):
        raise ValueError("v9 immutable source binding differs")
    canonical = policy_contract(RR_CAPTURE_POLICY, observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
    if (not isinstance(reason, str) or not reason.strip() or not isinstance(expected_target_head, str)
            or re.fullmatch("[0-9a-f]{40}", expected_target_head) is None or expected_target_head == SOURCE_HEAD
            or old.get("source_git_commit") != SOURCE_HEAD or new.get("source_git_commit") != expected_target_head
            or metadata.get("checkpoint_sha256") != SOURCE_SHA
            or any(type(metadata.get(k)) is not int or metadata[k] != SOURCE_COUNTS[k] for k in COUNTERS)
            or metadata.get("semantic_version") != "v3" or source_num_envs(metadata) != 1
            or metadata.get("policy_contract") != canonical
            or any(c.get("experiment_id") != EXPERIMENT for c in (old, new))):
        raise ValueError("v9 requires exact learned source221952 and unchanged RR410 N1 policy")
    variable = {"files", "runtime_content_sha256", "source_git_commit", "selected_configuration"}
    if {k:v for k,v in old.items() if k not in variable} != {k:v for k,v in new.items() if k not in variable}:
        raise ValueError("v9 cannot change namespace, physics, rates or training budgets")
    changed = {p for p,h in new["files"].items() if old["files"].get(p) != h}
    if (set(old["files"])-set(new["files"]) or set(new["files"])-set(old["files"]) != {MODULE}
            or changed != ALLOWED_FILES
            or dict(reviewed_code_sha256) != {p:new["files"][p] for p in sorted(changed)}):
        raise ValueError("v9 requires exactly six reviewed paths; historical validators remain unchanged")
    for contract in (old, new):
        selected = contract["selected_configuration"]
        if set(selected) != CONFIG_NAMES or any(v != {"path":f"configs/ppo_{EXPERIMENT}/{k}",
                "sha256":contract["files"].get(f"configs/ppo_{EXPERIMENT}/{k}")} for k,v in selected.items()):
            raise ValueError("v9 exact configuration/file bindings required")
    if (any(old["selected_configuration"][k] != new["selected_configuration"][k]
            for k in CONFIG_NAMES-{"execution_profile.yaml"})
            or source_profile.get("revision") != SOURCE_REVISION
            or source_profile.get("rr_capture_feedback_revision") != FEEDBACK
            or source_profile.get("rr_capture_wheel_mode") != "rr_capture_support_forward_projection_v1"
            or target_profile != {**source_profile, "revision":TARGET_REVISION}):
        raise ValueError("v9 permits only profile revision; assist, task, reward, caps and codec remain fixed")
    if MIGRATION in metadata or any(k not in metadata for k in preserved_keys(metadata)):
        raise ValueError("v9 requires complete source ancestry, without a repeated v9 migration")
    route = _source_branch(metadata)
    prior = metadata[PRIOR]
    if (not isinstance(prior, dict) or prior.get("schema") != "wlr50_clean.rr_signed_wheel_same410.v8"
            or prior.get("target_git_commit") != SOURCE_HEAD or prior.get("target_contract_sha256") != digest(old)
            or prior.get("target_runtime_content_sha256") != old["runtime_content_sha256"]
            or prior.get(PRIOR_FACTOR, {}).get("counter_origin") != ANCESTOR_ORIGIN
            or prior.get(PRIOR_FACTOR, {}).get("target_feedback_revision") != FEEDBACK
            or prior.get("source_selection", {}).get("source_role") != "front_validated_ancestor_control_eval"
            or prior.get("source_selection", {}).get("counters") != ANCESTOR_ORIGIN):
        raise ValueError("v9 source v8 receipt must retain its original zero-update220544 publication, not rewrite it as221952")
    for key,value in metadata.items():
        if key.endswith("_branch") and isinstance(value,dict) and "counter_origin" in value:
            origin = value["counter_origin"]
            if (set(origin) != set(COUNTERS) or any(type(origin[k]) is not int or not 0 <= origin[k] <= metadata[k] for k in COUNTERS)
                    or metadata.get(key+"_counts") != {k:metadata[k]-origin[k] for k in COUNTERS}):
                raise ValueError("v9 source branch counts differ: "+key)
    rate = metadata["optimizer_learning_rate"]
    if type(rate) not in (int,float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("v9 source effective LR must be finite positive")
    return {"schema":SCHEMA, "review_reason":reason.strip(), "source_selection":copy.deepcopy(source_binding),
        "observation_contract":{"source_policy_contract":canonical,"target_policy_contract":copy.deepcopy(canonical),
            "observation_layout":RR_CAPTURE_OBSERVATION_LAYOUT,"observation_dimension":410,"action_dimension":12,
            "num_envs":1,"parameter_mapping":"identity_all_parameters_and_buffers"},
        "source_feedback_revision":FEEDBACK,"target_feedback_revision":FEEDBACK,
        "observation_shape_changed":False,"observation_codec_changed":False,"capture_search_budget_changed":False,
        "observation_semantics_changed":["X409 current envelope armed timing extends to explicitly observed P12 postcapture scope"],
        "effective_execution_semantics":{"source_wheel_semantics":SOURCE_WHEEL_SEMANTICS,"wheel_semantics":WHEEL_SEMANTICS,
            "P09_original_path_preserved":True,"P12_requires_its_own_adjacent_committed_authored_stop":True,
            "P12_nominal_feedback_plus_point3_is_permitted_not_a_new_source_owner":True,
            "P12_selected_wheels_require_measured_current_bearing_including_RR_only_when_bearing":True,
            "existing_slew_rad_s2":1.8,"existing_maximum_forward_floor_rad_s":.3,
            "RR_assist_budget_and_state_unchanged":True,"raw_Gaussian_is_not_transformed_target":True},
        "checkpoint_output_routing":{"preserved_sha256":digest(route),"same_existing_branch_only":True,
            "historical_ancestor_origin":copy.deepcopy(ANCESTOR_ORIGIN),"new_source_origin":copy.deepcopy(SOURCE_COUNTS),
            "main_latest_pointer_promotion_authorized":False,"network_reset_or_borrowed_credit":False},
        "legacy_action_transform_is_current_execution_semantics":False,"controller_transition_semantics_changed":True,
        "same_mdp_claimed":False,"same_numeric_input_policy_mapping_preserved":True,
        "same_physical_state_action_equivalence_claimed":False,
        **dict.fromkeys(("policy_kernel_changed","sigma_changed","caps_changed","reward_changed",
                        "physical_dynamics_changed","physical_task_acceptance_rules_changed"),False),
        "reviewed_code_sha256":dict(reviewed_code_sha256),
        "preserved_metadata_sha256":{k:digest(metadata[k]) for k in preserved_keys(metadata)},
        "source_effective_learning_rate":rate,"target_effective_learning_rate":rate,
        "source_v8_receipt_sha256":digest(prior),"source_resume_migration_sha256":digest(metadata.get("resume_migration")),
        "counter_origin":copy.deepcopy(SOURCE_COUNTS),"existing_branch_origins_preserved":True,"creates_new_branch":False,
        "discard_old_rollout_storage":True,"old_rollout_is_new_MDP_onpolicy":False,"physical_state_inherited":False,
        "parameter_mapping":"identity_all_parameters_and_buffers","optimizer_mapping":"identity_full_Adam_moments_steps_groups_and_effective_LR",
        "normalizer_mapping":"identity_Identity","rng_mapping":"restore_exact_source_training_rng",
        "candidate_evaluation_only":False,"ancestor_training_requires_explicit_output_branch":True,
        "latest_branch_learned_weights_preserved":True,"main_latest_policy_equivalence_claimed":False,
        "latest_pointer_promotion_authorized":False,
        **dict.fromkeys(("added_policy_decisions","added_ppo_updates","added_optimizer_steps","added_auxiliary_updates"),0)}


def validate_wheel_semantics(source, target):
    # Parameterized P12 branches share the old numerical primitives. Full wheel
    # bytes are review-bound; behavioral P09 regression is a test, not an AST
    # equivalence claim for the four deliberately parameterized functions.
    for key in ("MODE","CONTEXT_SCHEMA","EVIDENCE_SCHEMA","ACK_KEY","WHEEL_RATE_RAD_S2","SUPPORT_LEGS","SUPPORT_INDICES"):
        if _literal(source,key) != _literal(target,key):
            raise ValueError("v9 cannot change the original P09 wheel constants: "+key)
    if _literal(source,"SEMANTICS") != SOURCE_WHEEL_SEMANTICS or _literal(target,"SEMANTICS") != WHEEL_SEMANTICS:
        raise ValueError("v9 wheel semantics differ from the reviewed P12-only retention")
    old = {n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    new = {n.name:n for n in ast.parse(target).body if isinstance(n,ast.FunctionDef)}
    if set(old) != set(new):
        raise ValueError("v9 cannot add a hidden wheel state owner or replace the public wheel API")
    for name in ("_get", "_number", "_vector", "_tick", "_clip"):
        a,b = old[name],new[name]
        if ast.dump(a,include_attributes=False) != ast.dump(b,include_attributes=False):
            raise ValueError("v9 must retain the original wheel numeric primitive: "+name)
    for name in ("_source_evidence", "_committed_endpoint"):
        arguments = new[name].args
        defaults = {arg.arg:default for arg,default in zip(arguments.kwonlyargs,arguments.kw_defaults)}
        if "source_phase" not in defaults or ast.literal_eval(defaults["source_phase"]) != "P09":
            raise ValueError("v9 parameterized source proof must default to original P09")
    strings = {n.value for n in ast.walk(ast.parse(target)) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    required = {"P12_all4_authored_final_wheel_stop_not_full_RL_source_endpoint",
        "prior_committed_P12_wheel_stop_ACK_then_adjacent_tick_full_RL_source_may_continue",
        "previous_requested_wheels_stop_or_live_prior", "P12_RR_placed_RL_unplaced"}
    if not required <= strings:
        raise ValueError("v9 requires reviewed independent P12 authored wheel-stop evidence")


def validate_v9_branch_receipt(metadata, contract, route):
    """Validate newest receipt without falling back to the inherited v8 receipt."""
    from .semantic_migration import digest
    receipt = metadata.get(MIGRATION)
    if not isinstance(receipt,dict):
        raise ValueError("v9 branch receipt missing or malformed")
    factor = receipt.get(FACTOR_KEY, {})
    if not isinstance(factor,dict):
        raise ValueError("v9 branch factor malformed")
    selection = registered_source(receipt.get("source_checkpoint_sha256"))
    if (receipt.get("schema") != SCHEMA or factor.get("schema") != SCHEMA
            or metadata.get("policy_contract",{}).get("observation_dimension") != 410
            or receipt.get("source_selection") != selection
            or receipt.get("source_manifest_sha256") != selection["manifest_sha256"]
            or receipt.get("source_git_commit") != SOURCE_HEAD
            or receipt.get("target_git_commit") != contract.get("source_git_commit")
            or receipt.get("target_contract_sha256") != digest(contract)
            or receipt.get("target_runtime_content_sha256") != contract.get("runtime_content_sha256")
            or factor.get("counter_origin") != SOURCE_COUNTS or factor.get("target_feedback_revision") != FEEDBACK
            or factor.get("source_selection") != selection
            or factor.get("preserved_metadata_sha256",{}).get("checkpoint_output_routing") != digest(route)
            or factor.get("source_v8_receipt_sha256") != digest(metadata.get(PRIOR))
            or _source_branch(metadata) != route
            or any(type(metadata.get(k)) is not int or metadata[k] < SOURCE_COUNTS[k] for k in COUNTERS)):
        raise ValueError("v9 branch receipt/runtime/source counts or inherited routing differs")
    return receipt


def build_rr_postcapture_wheel_migration(checkpoint,current_contract,*,expected_source_sha256,
        expected_target_head,reason,reviewed_code_sha256,project_root=None):
    import yaml
    from .semantic_migration import PROJECT_ROOT,checkpoint_metadata,_contract,_version_bytes,file_sha,digest
    binding=registered_source(expected_source_sha256);root=Path(project_root or PROJECT_ROOT).resolve()
    checkpoint=Path(checkpoint).resolve(strict=True);manifest=checkpoint.with_name(checkpoint.stem+"_manifest.json")
    if file_sha(checkpoint)!=binding["checkpoint_sha256"] or file_sha(manifest)!=binding["manifest_sha256"]:
        raise ValueError("v9 source checkpoint/manifest differ from the registered learned pair")
    metadata=checkpoint_metadata(checkpoint);old,new=_contract(metadata["runtime_contract"]),_contract(current_contract)
    value=rr_postcapture_wheel_factor(metadata,old,new,reason=reason,reviewed_code_sha256=reviewed_code_sha256,
        source_profile=yaml.safe_load(_version_bytes(root,old,PROFILE)),target_profile=yaml.safe_load((root/PROFILE).read_bytes()),
        source_binding=binding,expected_target_head=expected_target_head)
    route=metadata["checkpoint_output_routing"]
    if checkpoint.parent != (Path(route["output_root"])/"checkpoints/history").resolve():
        raise ValueError("v9 migration source must remain in its registered immutable branch history")
    head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(root),"status","--porcelain=v1","--untracked-files=all","--",
        "src/wlr50_clean","scripts","configs","artifacts/ppo_phase_v1_start","pyproject.toml"],text=True).strip()
    if head!=expected_target_head or dirty: raise ValueError("v9 target must be the explicit clean committed runtime")
    if any(file_sha(root/p)!=h for p,h in new["files"].items()): raise ValueError("v9 target runtime bytes differ")
    for name,selected in old["selected_configuration"].items():
        if name!="execution_profile.yaml" and _version_bytes(root,old,selected["path"])!=(root/selected["path"]).read_bytes():
            raise ValueError("v9 protected configuration bytes changed")
    validate_wheel_semantics(_version_bytes(root,old,WHEEL),(root/WHEEL).read_bytes())
    return {"schema":SCHEMA,"reason":reason.strip(),"source_checkpoint":str(checkpoint),
        "source_checkpoint_sha256":binding["checkpoint_sha256"],"source_manifest_sha256":binding["manifest_sha256"],
        "source_selection":binding,"source_git_commit":SOURCE_HEAD,"target_git_commit":expected_target_head,
        "source_contract_sha256":digest(old),"target_contract_sha256":digest(new),
        "source_runtime_content_sha256":old["runtime_content_sha256"],"target_runtime_content_sha256":new["runtime_content_sha256"],
        "allowed_changed_files":sorted(reviewed_code_sha256),"observation_dimension":410,"action_dimension":12,
        "discard_old_rollout_storage":True,"physics_resume":"fresh_legal_P01_reset",FACTOR_KEY:value}


def validate_rr_postcapture_wheel_migration(checkpoint,current_contract,plan_path,*,project_root=None):
    from .semantic_migration import file_sha
    path=Path(plan_path).resolve(strict=True);supplied=json.loads(path.read_text(encoding="utf-8"))
    expected=build_rr_postcapture_wheel_migration(checkpoint,current_contract,
        expected_source_sha256=supplied.get("source_checkpoint_sha256"),expected_target_head=supplied.get("target_git_commit"),
        reason=supplied.get("reason"),reviewed_code_sha256=supplied.get(FACTOR_KEY,{}).get("reviewed_code_sha256",{}),project_root=project_root)
    if supplied!=expected: raise ValueError("v9 plan differs from the learned source and reviewed target")
    return {**expected,"plan_path":str(path),"plan_sha256":file_sha(path)}


def record_loaded_rr_postcapture_wheel(runner,infos,verified):
    from .semantic_training import _verify_reviewed_same410_identity_state
    if verified.get("schema")!=SCHEMA or verified.get(FACTOR_KEY,{}).get("schema")!=SCHEMA or MIGRATION in infos:
        raise RuntimeError("v9 receipt is absent, ambiguous or repeated")
    _verify_reviewed_same410_identity_state(runner,infos,verified[FACTOR_KEY])
    return {**infos,MIGRATION:copy.deepcopy(verified)}

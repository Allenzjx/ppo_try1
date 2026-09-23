"""UNAPPLIED strict same410 v10 earned-reserve migration draft.

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
from .semantic_rr_postcapture_wheel_migration import preserved_keys as prior_keys
from .semantic_rr_capture_knee_migration import _literal

SCHEMA = "wlr50_clean.rr_capture_reserve_same410.v10"
FACTOR_KEY = "rr_capture_reserve_v10_factor"
MIGRATION = "rr_capture_reserve_v10_migration"
PRIOR = "rr_postcapture_wheel_v9_migration"
PRIOR_FACTOR = "rr_postcapture_wheel_v9_factor"
EXPERIMENT = "rr_capture_then_rl_transfer_v1"
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
SOURCE_HEAD = "3edda51732f4ff85717fcb3491bf5c8c5766474d"
V9_PUBLICATION_COUNTS = dict(zip(COUNTERS, (221952, 1699, 33980)))
ANCESTOR_ORIGIN = dict(zip(COUNTERS, (220544, 1688, 33760)))
SOURCE_BRANCH = "ancestor220544_signed_wheel_v8"
SOURCE_REVISION = "rr_capture_postcapture_wheel_retention_v9"
TARGET_REVISION = "rr_capture_progress_reserve_v10"
SOURCE_FEEDBACK = "signed_band_contact_formation_incremental_v6"
TARGET_FEEDBACK = "progress_earned_capture_reserve_incremental_v10"
SOURCE_SEARCH = "hip20_knee20_progress_earned_signed_task_band_knee12_then_1deg_public_peak_gap_le1mm_total53_exposure45_no_recharge_AIR_not_contact_sensor_TOP_unchanged_captured_issued_N_request_deltas_RL_current_TOP_retirement"
TARGET_SEARCH = "hip20_knee20_progress_earned_current_XY_support_tracking_gap_ge_minus15mm_knee12_then_1deg_public_peak_gap_le1mm_total53_exposure45_no_recharge_AIR_not_contact_sensor_TOP_unchanged_captured_issued_N_request_deltas_RL_current_TOP_retirement"
ASSIST = "src/wlr50_clean/ppo/semantic_rr_capture_assist.py"
PROFILE = f"configs/ppo_{EXPERIMENT}/execution_profile.yaml"
MODULE = "src/wlr50_clean/ppo/semantic_rr_capture_reserve_migration.py"
ALLOWED_FILES = frozenset({ASSIST, PROFILE, MODULE,
    "src/wlr50_clean/ppo/semantic_cli.py", "src/wlr50_clean/ppo/semantic_training.py",
    "src/wlr50_clean/ppo/semantic_migration.py"})
# Intentionally empty while the real v9 block is active. At a legal boundary
# register the actual newest selected sealed PPO checkpoint, or its explicitly
# approved official AUX result, with exact sidecar/counters/optional ledger hash.
# Empty means no real source is currently authorized for this draft.
SOURCE_REGISTRY = {}


def registered_source(sha):
    if not isinstance(sha,str) or sha not in SOURCE_REGISTRY:
        raise ValueError("v10 requires an explicitly registered actual sealed v9 learned source; none inferred")
    value=copy.deepcopy(SOURCE_REGISTRY[sha])
    if (re.fullmatch("[0-9a-f]{64}",sha) is None
            or re.fullmatch("[0-9a-f]{64}",str(value.get("manifest_sha256",""))) is None
            or value.get("source_role") not in ("latest_sealed_v9_branch_PPO","latest_sealed_v9_branch_with_official_front_retention_AUX")
            or set(value.get("counters",{}))!=set(COUNTERS)):
        raise ValueError("v10 registration must bind real SHA/sidecar/role/counters")
    counts=value["counters"]
    if any(type(counts[k]) is not int for k in COUNTERS):
        raise ValueError("v10 source counters must be exact integers")
    delta={k:counts[k]-V9_PUBLICATION_COUNTS[k] for k in COUNTERS}
    if (delta["ppo_updates"]<1 or delta["global_policy_decisions"]!=128*delta["ppo_updates"]
            or delta["optimizer_steps"]!=20*delta["ppo_updates"]):
        raise ValueError("v10 source must preserve the actual newly sealed v9 PPO updates, never rollback")
    ledger=value.get("front_retention_auxiliary_sha256")
    if ledger is not None and (not isinstance(ledger,str) or re.fullmatch("[0-9a-f]{64}",ledger) is None):
        raise ValueError("v10 optional current-branch AUX ledger binding is malformed")
    if value["source_role"].endswith("_AUX") and ledger is None:
        raise ValueError("v10 official AUX source requires its complete new ledger digest")
    return dict(checkpoint_sha256=sha,source_git_commit=SOURCE_HEAD,
        checkpoint_output_branch=SOURCE_BRANCH,branch_origin=copy.deepcopy(ANCESTOR_ORIGIN),**value)


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
        raise ValueError("v10 requires unchanged explicit ancestor branch routing and original220544 origin")
    return route


def rr_capture_reserve_factor(metadata, old, new, *, reason, reviewed_code_sha256,
        source_profile, target_profile, source_binding, expected_target_head):
    from .semantic_migration import digest, source_num_envs
    from .semantic_policy_distribution import CONFIG_NAMES, policy_contract
    if source_binding != registered_source(source_binding.get("checkpoint_sha256")):
        raise ValueError("v10 immutable source binding differs")
    canonical = policy_contract(RR_CAPTURE_POLICY, observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
    if (not isinstance(reason, str) or not reason.strip() or not isinstance(expected_target_head, str)
            or re.fullmatch("[0-9a-f]{40}", expected_target_head) is None or expected_target_head == SOURCE_HEAD
            or old.get("source_git_commit") != SOURCE_HEAD or new.get("source_git_commit") != expected_target_head
            or metadata.get("checkpoint_sha256") != source_binding["checkpoint_sha256"]
            or any(type(metadata.get(k)) is not int or metadata[k] != source_binding["counters"][k] for k in COUNTERS)
            or metadata.get("semantic_version") != "v3" or source_num_envs(metadata) != 1
            or metadata.get("policy_contract") != canonical
            or any(c.get("experiment_id") != EXPERIMENT for c in (old, new))):
        raise ValueError("v10 requires exact latest registered learned v9 source and unchanged RR410 N1 policy")
    variable = {"files", "runtime_content_sha256", "source_git_commit", "selected_configuration"}
    if {k:v for k,v in old.items() if k not in variable} != {k:v for k,v in new.items() if k not in variable}:
        raise ValueError("v10 cannot change namespace, physics, rates or training budgets")
    changed = {p for p,h in new["files"].items() if old["files"].get(p) != h}
    if (set(old["files"])-set(new["files"]) or set(new["files"])-set(old["files"]) != {MODULE}
            or changed != ALLOWED_FILES
            or dict(reviewed_code_sha256) != {p:new["files"][p] for p in sorted(changed)}):
        raise ValueError("v10 requires exactly six reviewed paths; historical validators remain unchanged")
    for contract in (old, new):
        selected = contract["selected_configuration"]
        if set(selected) != CONFIG_NAMES or any(v != {"path":f"configs/ppo_{EXPERIMENT}/{k}",
                "sha256":contract["files"].get(f"configs/ppo_{EXPERIMENT}/{k}")} for k,v in selected.items()):
            raise ValueError("v10 exact configuration/file bindings required")
    if (any(old["selected_configuration"][k] != new["selected_configuration"][k]
            for k in CONFIG_NAMES-{"execution_profile.yaml"})
            or source_profile.get("revision") != SOURCE_REVISION
            or source_profile.get("rr_capture_feedback_revision") != SOURCE_FEEDBACK
            or source_profile.get("rr_capture_wheel_mode") != "rr_capture_support_forward_projection_v1"
            or target_profile != {**source_profile, "revision":TARGET_REVISION, "rr_capture_feedback_revision":TARGET_FEEDBACK}):
        raise ValueError("v10 permits only profile revision/feedback marker; task, reward, caps and codec remain fixed")
    if MIGRATION in metadata or any(k not in metadata for k in preserved_keys(metadata)):
        raise ValueError("v10 requires complete source ancestry, without a repeated v10 migration")
    route = _source_branch(metadata)
    prior = metadata[PRIOR]
    from .semantic_rr_postcapture_wheel_migration import validate_v9_branch_receipt
    validate_v9_branch_receipt(metadata,old,route)
    ledger=metadata.get("rr_capture_transfer_branch",{}).get("front_retention_auxiliary")
    if (digest(ledger) if ledger is not None else None) != source_binding.get("front_retention_auxiliary_sha256"):
        raise ValueError("v10 selected source current-branch front AUX ledger differs")
    for key,value in metadata.items():
        if key.endswith("_branch") and isinstance(value,dict) and "counter_origin" in value:
            origin = value["counter_origin"]
            if (set(origin) != set(COUNTERS) or any(type(origin[k]) is not int or not 0 <= origin[k] <= metadata[k] for k in COUNTERS)
                    or metadata.get(key+"_counts") != {k:metadata[k]-origin[k] for k in COUNTERS}):
                raise ValueError("v10 source branch counts differ: "+key)
    rate = metadata["optimizer_learning_rate"]
    if type(rate) not in (int,float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("v10 source effective LR must be finite positive")
    return {"schema":SCHEMA, "review_reason":reason.strip(), "source_selection":copy.deepcopy(source_binding),
        "observation_contract":{"source_policy_contract":canonical,"target_policy_contract":copy.deepcopy(canonical),
            "observation_layout":RR_CAPTURE_OBSERVATION_LAYOUT,"observation_dimension":410,"action_dimension":12,
            "num_envs":1,"parameter_mapping":"identity_all_parameters_and_buffers"},
        "source_feedback_revision":SOURCE_FEEDBACK,"target_feedback_revision":TARGET_FEEDBACK,
        "observation_shape_changed":False,"observation_codec_changed":False,"capture_search_budget_changed":False,
        "observation_semantics_changed":[],
        "existing_observed_state_controls_changed_transform":
            "Mode6 earned/continued reserve may act above25mm; the public gap/mode/travel fields and all budgets retain their definitions",
        "effective_execution_semantics":{"source_search_semantics":SOURCE_SEARCH,"search_semantics":TARGET_SEARCH,
            "removed_upper_gap_checks":["reserve_earned","reserve_continuation","mode6_snapshot_validation"],
            "signed_lower_gap_m":-.015,"sensor_TOP_upper_gap_rule_changed":False,
            "fresh_progress_m":.0002,"progress_window_active_s":2.,
            "total_maximum_travel_deg":53.,"total_maximum_active_exposure_s":45.,
            "reserve_budget_recharged":False,"current_XY_Q_cross_AIR_support_tracking_still_required":True,
            "v9_wheel_bytes_unchanged":True,"raw_Gaussian_is_not_transformed_target":True},
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
        "source_v9_receipt_sha256":digest(prior),"source_resume_migration_sha256":digest(metadata.get("resume_migration")),
        "counter_origin":copy.deepcopy(source_binding["counters"]),"existing_branch_origins_preserved":True,"creates_new_branch":False,
        "discard_old_rollout_storage":True,"old_rollout_is_new_MDP_onpolicy":False,"physical_state_inherited":False,
        "parameter_mapping":"identity_all_parameters_and_buffers","optimizer_mapping":"identity_full_Adam_moments_steps_groups_and_effective_LR",
        "normalizer_mapping":"identity_Identity","rng_mapping":"restore_exact_source_training_rng",
        "candidate_evaluation_only":False,"ancestor_training_requires_explicit_output_branch":True,
        "latest_branch_learned_weights_preserved":True,"main_latest_policy_equivalence_claimed":False,
        "latest_pointer_promotion_authorized":False,
        **dict.fromkeys(("added_policy_decisions","added_ppo_updates","added_optimizer_steps","added_auxiliary_updates"),0)}


def validate_assist_semantics(source,target):
    """Exactly the three reviewed upper-gap checks, public literals and docstrings."""
    if (_literal(source,"RR_CAPTURE_FEEDBACK_REVISION")!=SOURCE_FEEDBACK
            or _literal(target,"RR_CAPTURE_FEEDBACK_REVISION")!=TARGET_FEEDBACK
            or _literal(source,"RR_CAPTURE_SEARCH_SEMANTICS")!=SOURCE_SEARCH
            or _literal(target,"RR_CAPTURE_SEARCH_SEMANTICS")!=TARGET_SEARCH):
        raise ValueError("v10 assist feedback/search literals differ")
    counts=dict(import_upper=0,window_upper=0,gap_upper=0)
    old_window=ast.dump(ast.parse('not RR_CAPTURE_GAP_MIN_M <= state["window_start_gap_m"] <= RR_CAPTURE_GAP_MAX_M',mode="eval").body,include_attributes=False)
    old_gap=ast.dump(ast.parse('RR_CAPTURE_GAP_MIN_M <= gap <= RR_CAPTURE_GAP_MAX_M',mode="eval").body,include_attributes=False)
    replacements={SOURCE_FEEDBACK:TARGET_FEEDBACK,SOURCE_SEARCH:TARGET_SEARCH,
        "fresh_near_top_progress_required":"fresh_capture_progress_required",
        "RR capture assist DESCEND_PROGRESS lacks public fresh near-top credit":
            "RR capture assist DESCEND_PROGRESS lacks public fresh capture progress credit"}
    literal_counts={key:0 for key in replacements}
    class Expected(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.module=="semantic_rr_capture_context":
                aliases=[n for n in node.names if n.name!="RR_CAPTURE_GAP_MAX_M"]
                counts["import_upper"]+=len(node.names)-len(aliases);node.names=aliases
            return node
        def visit_UnaryOp(self,node):
            if ast.dump(node,include_attributes=False)==old_window:
                counts["window_upper"]+=1
                return ast.parse('state["window_start_gap_m"] < RR_CAPTURE_GAP_MIN_M',mode="eval").body
            return self.generic_visit(node)
        def visit_Compare(self,node):
            if ast.dump(node,include_attributes=False)==old_gap:
                counts["gap_upper"]+=1
                return ast.parse("gap >= RR_CAPTURE_GAP_MIN_M",mode="eval").body
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if isinstance(node.value,str) and node.value in replacements:
                literal_counts[node.value]+=1
                return ast.copy_location(ast.Constant(value=replacements[node.value]),node)
            return node
    class WithoutDocs(ast.NodeTransformer):
        def generic_visit(self,node):
            super().generic_visit(node)
            if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
                    node.body.pop(0)
            return node
    expected=WithoutDocs().visit(Expected().visit(ast.parse(source)))
    actual=WithoutDocs().visit(ast.parse(target))
    if (counts!=dict(import_upper=1,window_upper=1,gap_upper=2)
            or any(n!=1 for n in literal_counts.values())
            or ast.dump(expected,include_attributes=False)!=ast.dump(actual,include_attributes=False)):
        raise ValueError("v10 assist changed beyond exactly three upper checks/public literals/docs")


def validate_v10_branch_receipt(metadata, contract, route):
    """Validate newest receipt without falling back to the inherited v9 receipt."""
    from .semantic_migration import digest
    receipt = metadata.get(MIGRATION)
    if not isinstance(receipt,dict):
        raise ValueError("v10 branch receipt missing or malformed")
    factor = receipt.get(FACTOR_KEY, {})
    if not isinstance(factor,dict):
        raise ValueError("v10 branch factor malformed")
    selection = registered_source(receipt.get("source_checkpoint_sha256"))
    if (receipt.get("schema") != SCHEMA or factor.get("schema") != SCHEMA
            or metadata.get("policy_contract",{}).get("observation_dimension") != 410
            or receipt.get("source_selection") != selection
            or receipt.get("source_manifest_sha256") != selection["manifest_sha256"]
            or receipt.get("source_git_commit") != SOURCE_HEAD
            or receipt.get("target_git_commit") != contract.get("source_git_commit")
            or receipt.get("target_contract_sha256") != digest(contract)
            or receipt.get("target_runtime_content_sha256") != contract.get("runtime_content_sha256")
            or factor.get("counter_origin") != selection["counters"] or factor.get("target_feedback_revision") != TARGET_FEEDBACK
            or factor.get("source_selection") != selection
            or factor.get("preserved_metadata_sha256",{}).get("checkpoint_output_routing") != digest(route)
            or factor.get("source_v9_receipt_sha256") != digest(metadata.get(PRIOR))
            or _source_branch(metadata) != route
            or any(type(metadata.get(k)) is not int or metadata[k] < selection["counters"][k] for k in COUNTERS)):
        raise ValueError("v10 branch receipt/runtime/source counts or inherited routing differs")
    return receipt


def build_rr_capture_reserve_migration(checkpoint,current_contract,*,expected_source_sha256,
        expected_target_head,reason,reviewed_code_sha256,project_root=None):
    import yaml
    from .semantic_migration import PROJECT_ROOT,checkpoint_metadata,_contract,_version_bytes,file_sha,digest
    binding=registered_source(expected_source_sha256);root=Path(project_root or PROJECT_ROOT).resolve()
    checkpoint=Path(checkpoint).resolve(strict=True);manifest=checkpoint.with_name(checkpoint.stem+"_manifest.json")
    if file_sha(checkpoint)!=binding["checkpoint_sha256"] or file_sha(manifest)!=binding["manifest_sha256"]:
        raise ValueError("v10 source checkpoint/manifest differ from the registered learned pair")
    metadata=checkpoint_metadata(checkpoint);old,new=_contract(metadata["runtime_contract"]),_contract(current_contract)
    value=rr_capture_reserve_factor(metadata,old,new,reason=reason,reviewed_code_sha256=reviewed_code_sha256,
        source_profile=yaml.safe_load(_version_bytes(root,old,PROFILE)),target_profile=yaml.safe_load((root/PROFILE).read_bytes()),
        source_binding=binding,expected_target_head=expected_target_head)
    route=metadata["checkpoint_output_routing"]
    if checkpoint.parent != (Path(route["output_root"])/"checkpoints/history").resolve():
        raise ValueError("v10 migration source must remain in its registered immutable branch history")
    head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(root),"status","--porcelain=v1","--untracked-files=all","--",
        "src/wlr50_clean","scripts","configs","artifacts/ppo_phase_v1_start","pyproject.toml"],text=True).strip()
    if head!=expected_target_head or dirty: raise ValueError("v10 target must be the explicit clean committed runtime")
    if any(file_sha(root/p)!=h for p,h in new["files"].items()): raise ValueError("v10 target runtime bytes differ")
    for name,selected in old["selected_configuration"].items():
        if name!="execution_profile.yaml" and _version_bytes(root,old,selected["path"])!=(root/selected["path"]).read_bytes():
            raise ValueError("v10 protected configuration bytes changed")
    validate_assist_semantics(_version_bytes(root,old,ASSIST),(root/ASSIST).read_bytes())
    return {"schema":SCHEMA,"reason":reason.strip(),"source_checkpoint":str(checkpoint),
        "source_checkpoint_sha256":binding["checkpoint_sha256"],"source_manifest_sha256":binding["manifest_sha256"],
        "source_selection":binding,"source_git_commit":SOURCE_HEAD,"target_git_commit":expected_target_head,
        "source_contract_sha256":digest(old),"target_contract_sha256":digest(new),
        "source_runtime_content_sha256":old["runtime_content_sha256"],"target_runtime_content_sha256":new["runtime_content_sha256"],
        "allowed_changed_files":sorted(reviewed_code_sha256),"observation_dimension":410,"action_dimension":12,
        "discard_old_rollout_storage":True,"physics_resume":"fresh_legal_P01_reset",FACTOR_KEY:value}


def validate_rr_capture_reserve_migration(checkpoint,current_contract,plan_path,*,project_root=None):
    from .semantic_migration import file_sha
    path=Path(plan_path).resolve(strict=True);supplied=json.loads(path.read_text(encoding="utf-8"))
    expected=build_rr_capture_reserve_migration(checkpoint,current_contract,
        expected_source_sha256=supplied.get("source_checkpoint_sha256"),expected_target_head=supplied.get("target_git_commit"),
        reason=supplied.get("reason"),reviewed_code_sha256=supplied.get(FACTOR_KEY,{}).get("reviewed_code_sha256",{}),project_root=project_root)
    if supplied!=expected: raise ValueError("v10 plan differs from the learned source and reviewed target")
    return {**expected,"plan_path":str(path),"plan_sha256":file_sha(path)}


def record_loaded_rr_capture_reserve(runner,infos,verified):
    from .semantic_training import _verify_reviewed_same410_identity_state
    if verified.get("schema")!=SCHEMA or verified.get(FACTOR_KEY,{}).get("schema")!=SCHEMA or MIGRATION in infos:
        raise RuntimeError("v10 receipt is absent, ambiguous or repeated")
    _verify_reviewed_same410_identity_state(runner,infos,verified[FACTOR_KEY])
    return {**infos,MIGRATION:copy.deepcopy(verified)}

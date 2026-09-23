"""One reviewed RR410 knee v3-control revision; exact state, fresh rollout."""
from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import subprocess

from .semantic_rr_capture_profile import RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_LAYOUT

SCHEMA = "wlr50_clean.rr_capture_knee_same410.v3"
FACTOR_KEY = "rr_capture_knee_v3_factor"
MIGRATION = "rr_capture_knee_v3_migration"
FEEDBACK_REVISION = "window_peak_hip_then_knee_v3"
SOURCE_FEEDBACK_REVISION = "window_peak_progress_v2"
PRIOR_FEEDBACK_MIGRATION = "rr_capture_feedback_peak_v2_migration"
PRIOR_FEEDBACK_FACTOR = "rr_capture_feedback_peak_v2_factor"
WINDOW_REFERENCE_SEMANTICS = "public_window_peak_gap_reuses_existing_window_start_gap_scalar_upward_motion_never_resets_elapsed"
CAPTURE_SEARCH_SEMANTICS = "combined_travel_0_40_first20_negative_hip_then20_positive_knee_at1deg_per_s_knee_hold_is_dynamic_target_axis_from_travel_hip_elapsed_equals_total_elapsed_minus_max_travel_minus20_0"
SOURCE_HEAD = "26db2a1946e8357e4d3be2f53dabbf96d38edf97"
SOURCE_CHECKPOINT_SHA256 = "a32a4f01afbfbcda06ba59f7e439bb5ab9aa1dbefb3d8e3093e0dd86c827e85e"
EXPERIMENT = "rr_capture_then_rl_transfer_v1"
SOURCE_REVISION = "rr_capture_then_rl_transfer_v2_peak_gap_feedback"
TARGET_REVISION = "rr_capture_then_rl_transfer_v3_hip_then_knee_feedback"
ASSIST = "src/wlr50_clean/ppo/semantic_rr_capture_assist.py"
MODULE = "src/wlr50_clean/ppo/semantic_rr_capture_knee_migration.py"
PROFILE = f"configs/ppo_{EXPERIMENT}/execution_profile.yaml"
ALLOWED_FILES = frozenset({ASSIST, MODULE, PROFILE,
    "src/wlr50_clean/ppo/semantic_training.py",
    "src/wlr50_clean/ppo/semantic_migration.py"})
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
PRIOR_BRANCHES = ("p05_capture_assist", "capture_feedback_semantics", "rr_postcross_workspace",
    "rr_receiver_retirement_v2", "p05_preedge_approach_recovery", "rr_capture_transfer")


def preserved_keys(metadata):
    from .semantic_rr_workspace_migration import preserved_keys as previous_keys
    return tuple(sorted(set(previous_keys(metadata)) | {"seed", "execution_topology", PRIOR_FEEDBACK_MIGRATION} | {
        name + suffix for name in PRIOR_BRANCHES for suffix in ("_branch", "_branch_counts", "_migration")}
        | {k for k in metadata if k.startswith("new_mdp_") or "origin" in k
           or k in ("resume_ancestry", "policy_distribution_migration_evidence",
                    "observation_scale_compensation_evidence", "observation_append_evidence")}))


def rr_capture_knee_factor(metadata, old, new, *, reason, reviewed_code_sha256,
                               source_profile, target_profile):
    from .semantic_migration import source_num_envs, digest
    from .semantic_policy_distribution import policy_contract, CONFIG_NAMES
    canonical = policy_contract(RR_CAPTURE_POLICY, observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
    if (not isinstance(reason, str) or not reason.strip() or metadata.get("semantic_version") != "v3"
            or source_num_envs(metadata) != 1 or old.get("source_git_commit") != SOURCE_HEAD
            or old.get("experiment_id") != EXPERIMENT or new.get("experiment_id") != EXPERIMENT
            or metadata.get("policy_contract") != canonical or canonical["observation_dimension"] != 410):
        raise ValueError("RR knee v3 requires the exact 26db RR410 v3 N1 source, not a 389 append")
    variable = {"files", "runtime_content_sha256", "source_git_commit", "selected_configuration"}
    if ({k:v for k,v in old.items() if k not in variable}
            != {k:v for k,v in new.items() if k not in variable}):
        raise ValueError("RR knee v3 cannot alter runtime profile, physics, rates or budgets")
    if (MODULE in old["files"] or set(old["files"]) - set(new["files"])
            or set(new["files"]) - set(old["files"]) != {MODULE}):
        raise ValueError("RR knee v3 may add only its own module and delete no source files")
    delta = sorted(p for p in new["files"] if old["files"].get(p) != new["files"][p])
    if (set(delta) != ALLOWED_FILES
            or dict(reviewed_code_sha256) != {p:new["files"][p] for p in delta}):
        raise ValueError("RR knee v3 requires complete exact reviewed hashes within its finite scope")
    for contract in (old, new):
        selected = contract.get("selected_configuration", {})
        if set(selected) != CONFIG_NAMES or any(binding != {
                "path":f"configs/ppo_{EXPERIMENT}/{name}",
                "sha256":contract["files"].get(f"configs/ppo_{EXPERIMENT}/{name}")}
                for name,binding in selected.items()):
            raise ValueError("RR knee v3 requires all six selected configuration bindings")
    if any(old["selected_configuration"][name] != new["selected_configuration"][name]
           for name in CONFIG_NAMES if name != "execution_profile.yaml"):
        raise ValueError("RR knee v3 cannot alter the other five configuration bytes")
    if (not isinstance(source_profile, dict) or source_profile.get("revision") != SOURCE_REVISION
            or source_profile.get("rr_capture_feedback_revision") != SOURCE_FEEDBACK_REVISION
            or source_profile.get("rr_capture_assist_mode") != "rr_hip_only_capture_v1"
            or source_profile.get("rr_capture_wheel_mode") != "off"
            or target_profile != {**source_profile, "revision":TARGET_REVISION,
                                  "rr_capture_feedback_revision":FEEDBACK_REVISION}):
        raise ValueError("execution profile permits only its revision and explicit RR knee v3 opt-in")
    if MIGRATION in metadata or any(k.startswith("rr_capture_knee_v3_") for k in metadata):
        raise ValueError("RR knee v3 boundary cannot be repeated or replace prior lineage")
    if any(k not in metadata for k in preserved_keys(metadata)):
        raise ValueError("RR knee v3 requires full prior state, origins and AUX lineage")
    previous = metadata[PRIOR_FEEDBACK_MIGRATION]
    previous_factor = previous.get(PRIOR_FEEDBACK_FACTOR, {}) if isinstance(previous, dict) else {}
    if (not isinstance(previous, dict)
            or previous.get("schema") != "wlr50_clean.rr_capture_feedback_same410.v2"
            or previous.get("target_git_commit") != SOURCE_HEAD
            or previous.get("target_contract_sha256") != digest(old)
            or previous.get("target_runtime_content_sha256") != old["runtime_content_sha256"]
            or previous_factor.get("schema") != "wlr50_clean.rr_capture_feedback_same410.v2"
            or previous_factor.get("target_feedback_revision") != SOURCE_FEEDBACK_REVISION
            or previous_factor.get("observation_contract", {}).get("target_policy_contract") != canonical
            or previous_factor.get("counter_origin") != {k:metadata[k] for k in COUNTERS}):
        raise ValueError("RR knee v3 requires the intact zero-update 26db feedback-v2 publication receipt")
    if any(type(metadata[k]) is not int or metadata[k] < 0 for k in COUNTERS):
        raise ValueError("RR knee v3 source counters must be nonnegative integers")
    for name in PRIOR_BRANCHES:
        branch = metadata[name + "_branch"]
        origin = branch.get("counter_origin", {}) if isinstance(branch, dict) else {}
        if (set(origin) != set(COUNTERS) or not isinstance(metadata[name + "_migration"], dict)
                or any(type(origin[k]) is not int or not 0 <= origin[k] <= metadata[k] for k in COUNTERS)
                or metadata[name + "_branch_counts"] != {k:metadata[k]-origin[k] for k in COUNTERS}):
            raise ValueError("RR knee v3 requires intact independent origins and actual branch counts")
    if (metadata["rr_capture_transfer_branch"].get("schema") != "wlr50_clean.rr_capture_transfer_append.v1"
            or not isinstance(metadata["rr_postcross_workspace_branch"].get("front_rehearsal_auxiliary"), dict)
            or not isinstance(metadata["task_conditioned_hip_wheel_branch"].get("auxiliary_mean_learning"), dict)):
        raise ValueError("RR knee v3 requires the existing RR branch and both complete AUX objects")
    rate = metadata["optimizer_learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("RR knee v3 requires actual positive effective Adam LR")
    return {"schema":SCHEMA, "review_reason":reason.strip(),
        "source_feedback_revision":SOURCE_FEEDBACK_REVISION,
        "target_feedback_revision":FEEDBACK_REVISION, "window_reference_semantics":WINDOW_REFERENCE_SEMANTICS,
        "reviewed_code_sha256":dict(reviewed_code_sha256),
        "source_execution_profile_sha256":digest(source_profile), "target_execution_profile_sha256":digest(target_profile),
        "observation_contract":{"source_policy_contract":canonical, "target_policy_contract":copy.deepcopy(canonical),
            "observation_layout":RR_CAPTURE_OBSERVATION_LAYOUT, "observation_dimension":410,
            "action_dimension":12, "num_envs":1, "parameter_mapping":"identity_all_parameters_and_buffers"},
        "observation_shape_changed":False, "observation_codec_changed":False,
        "observation_semantics_changed":[
            "observation391 knee_hold_deg /180 is the current potentially moving knee capture target",
            "observation394 travel_used_deg /20 is combined hip-negative20 then knee-positive20 degree travel; range0..2",
            "observation395 descent_elapsed_s /12 is cumulative hip-plus-knee exposure up to32 seconds; range0..8/3"],
        "current_effective_execution_semantics":{
            "revision":FEEDBACK_REVISION,
            "capture_search_semantics":CAPTURE_SEARCH_SEMANTICS,
            "axis_order":"hip_negative_then_knee_positive_after_20_degree_commanded_hip_search_with_existing_actual_tracking_gate",
            "hip_travel_limit_deg":20., "knee_travel_limit_deg":20., "knee_rate_deg_s":1.,
            "combined_travel_limit_deg":40., "hip_exposure_limit_s":12., "total_exposure_limit_s":32.,
            "knee_used_deg":"max(travel_used_deg-20,0)",
            "hip_elapsed_s":"descent_elapsed_s-knee_used_deg_at_constant_knee_rate_1deg_per_s",
            "budget_refresh_allowed":False, "new_hidden_state":False,
            "tracking_cross_XY_other_two_support_peak_feedback_preserved":True,
            "FL_wheel_shaping":"off"},
        "legacy_action_transform_description":canonical["action_transform"],
        "legacy_action_transform_is_current_execution_semantics":False,
        "same_numeric_input_policy_mapping_preserved":True, "same_physical_state_action_equivalence_claimed":False,
        "controller_transition_semantics_changed":True, "capture_assist_changed":True, "same_mdp_claimed":False,
        "pre_cross_ownership_semantics":"qualified_P09_AIR_XY_hold_previous_final_only_no_descent_before_real_cross",
        "capture_owner_acquisition_changed":False, "pre_cross_descent_allowed":False,
        "policy_kernel_changed":False, "reward_changed":False, "nominal_changed":False,
        "caps_changed":False, "sigma_changed":False, "physical_dynamics_changed":False,
        "physical_task_acceptance_rules_changed":False,
        "preserved_metadata_sha256":{k:digest(metadata[k]) for k in preserved_keys(metadata)},
        "source_resume_migration_sha256":digest(metadata.get("resume_migration")),
        "source_feedback_v2_receipt_sha256":digest(previous),
        "counter_origin":{k:metadata[k] for k in COUNTERS},
        "source_effective_learning_rate":rate, "target_effective_learning_rate":rate,
        "parameter_mapping":"identity_all_parameters_and_buffers",
        "optimizer_mapping":"identity_all_Adam_moments_steps_groups_and_effective_LR",
        "normalizer_mapping":"identity_Identity", "rng_mapping":"restore_exact_source_training_rng",
        "old_rollout_is_new_MDP_onpolicy":False, "discard_old_rollout_storage":True,
        "physical_state_inherited":False, "existing_branch_origins_preserved":True, "creates_new_branch":False,
        **dict.fromkeys(("added_policy_decisions", "added_ppo_updates", "added_optimizer_steps", "added_auxiliary_updates"), 0)}


def _literal(raw, name):
    values = [ast.literal_eval(node.value) for node in ast.parse(raw).body
              if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)]
    if len(values) > 1:
        raise ValueError("ambiguous RR knee v3 constant: " + name)
    return values[0] if values else None


def _validate_assist_semantics(before, after):
    """Explicit changed meanings, unchanged public positions and numerical codec."""
    for name, source_value, target_value in (
            ("RR_CAPTURE_FEEDBACK_REVISION", SOURCE_FEEDBACK_REVISION, FEEDBACK_REVISION),
            ("RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS", WINDOW_REFERENCE_SEMANTICS, WINDOW_REFERENCE_SEMANTICS),
            ("RR_CAPTURE_SEARCH_SEMANTICS", None, CAPTURE_SEARCH_SEMANTICS)):
        if _literal(before, name) != source_value or _literal(after, name) != target_value:
            raise ValueError("RR knee v3 needs the exact peak-v2 to hip-then-knee-v3 constants")
    names = _literal(before, "RR_CAPTURE_ASSIST_FEATURE_NAMES")
    if (not isinstance(names, tuple) or len(names) != 14 or names[7] != "window_start_gap_m"
            or _literal(after, "RR_CAPTURE_ASSIST_FEATURE_NAMES") != names
            or _literal(before, "_SCALES") != _literal(after, "_SCALES")):
        raise ValueError("RR knee v3 must preserve all14 scalar names, positions and numerical codec")


def build_rr_capture_knee_migration(checkpoint, current_contract, *, reason, reviewed_code_sha256, project_root=None):
    import yaml
    from .semantic_migration import PROJECT_ROOT, checkpoint_metadata, _contract, _version_bytes, file_sha, digest
    root = Path(project_root or PROJECT_ROOT).resolve(); checkpoint = Path(checkpoint).resolve(strict=True)
    if file_sha(checkpoint) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("RR knee v3 requires the exact published CP220544 feedback-v2 checkpoint bytes")
    metadata = checkpoint_metadata(checkpoint)
    old, new = _contract(metadata["runtime_contract"]), _contract(current_contract)
    source_profile = yaml.safe_load(_version_bytes(root, old, PROFILE))
    target_profile = yaml.safe_load((root / PROFILE).read_bytes())
    factor = rr_capture_knee_factor(metadata, old, new, reason=reason, reviewed_code_sha256=reviewed_code_sha256,
        source_profile=source_profile, target_profile=target_profile)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if new["source_git_commit"] != head or head == old["source_git_commit"]:
        raise ValueError("RR knee v3 target must name a new actual committed runtime HEAD")
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all", "--",
        "src/wlr50_clean", "scripts", "configs", "artifacts/ppo_phase_v1_start", "pyproject.toml"],
        check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("RR knee v3 target runtime must be clean and committed")
    for path, sha in new["files"].items():
        if file_sha(root / path) != sha:
            raise ValueError("RR knee v3 target runtime bytes differ: " + path)
    for name, binding in old["selected_configuration"].items():
        if name != "execution_profile.yaml" and _version_bytes(root, old, binding["path"]) != (root / binding["path"]).read_bytes():
            raise ValueError("RR knee v3 changed a protected configuration's bytes")
    before, after = _version_bytes(root, old, ASSIST), (root / ASSIST).read_bytes()
    _validate_assist_semantics(before, after)
    return {"schema":SCHEMA, "reason":reason.strip(), "source_checkpoint":str(checkpoint),
        "source_checkpoint_sha256":file_sha(checkpoint),
        "source_manifest_sha256":file_sha(checkpoint.with_name(checkpoint.stem + "_manifest.json")),
        "source_contract_sha256":digest(old), "target_contract_sha256":digest(new),
        "source_git_commit":old["source_git_commit"], "target_git_commit":head,
        "source_runtime_content_sha256":old["runtime_content_sha256"], "target_runtime_content_sha256":new["runtime_content_sha256"],
        "allowed_changed_files":sorted(reviewed_code_sha256),
        "changed_file_hashes":{p:{"before":old["files"].get(p), "after":new["files"][p]} for p in sorted(reviewed_code_sha256)},
        "observation_dimension":410, "action_dimension":12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget":True,
        "discard_old_rollout_storage":True, "physics_resume":"fresh_legal_P01_reset", FACTOR_KEY:factor}


def validate_rr_capture_knee_migration(checkpoint, current_contract, plan_path, *, project_root=None):
    from .semantic_migration import file_sha
    path = Path(plan_path).resolve(strict=True); supplied = json.loads(path.read_text(encoding="utf-8"))
    expected = build_rr_capture_knee_migration(checkpoint, current_contract, reason=supplied.get("reason"),
        reviewed_code_sha256=supplied.get(FACTOR_KEY, {}).get("reviewed_code_sha256", {}), project_root=project_root)
    if supplied != expected:
        raise ValueError("RR knee v3 plan differs from immutable source and target runtime")
    return {**expected, "plan_path":str(path), "plan_sha256":file_sha(path)}


def record_loaded_rr_capture_knee(runner, infos, verified):
    """After the official full-state load; add only the reviewed receipt."""
    import torch
    from .semantic_migration import digest
    from .semantic_training import parameter_hash, state_hash, _normalizers
    from .rl_library_wrapper import optimizer_learning_rate, capture_training_rng_state
    factor = verified[FACTOR_KEY]
    if verified.get("schema") != SCHEMA or factor.get("schema") != SCHEMA or MIGRATION in infos:
        raise RuntimeError("RR knee v3 must be explicit and unrepeated")
    actual = {"actor_parameter_sha256":parameter_hash(runner.alg.actor),
        "critic_parameter_sha256":parameter_hash(runner.alg.critic),
        "optimizer_state_sha256":state_hash(runner.alg.optimizer.state_dict()),
        "normalizer_state_sha256":state_hash(_normalizers(runner)),
        "training_rng_state":capture_training_rng_state(seed=infos["seed"])}
    if (any(type(getattr(runner.alg, role).obs_normalizer) is not torch.nn.Identity for role in ("actor", "critic"))
            or optimizer_learning_rate(runner) != factor["source_effective_learning_rate"]
            or runner.alg.learning_rate != factor["source_effective_learning_rate"]
            or any(k not in infos or digest(infos[k]) != value for k, value in factor["preserved_metadata_sha256"].items())
            or any(value != infos[k] for k, value in actual.items())
            or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None
            or tuple(runner.alg.storage.actions.shape) != (128, 1, 12)
            or any(tuple(runner.alg.storage.observations[k].shape) != (128, 1, 410) for k in ("policy", "critic"))):
        raise RuntimeError("RR knee v3 altered exact source state, LR, RNG or inherited a rollout")
    return {**infos, MIGRATION:copy.deepcopy(verified)}


def publish_rr_capture_knee_checkpoint(checkpoint, current_contract, plan_path, output_checkpoint):
    """Official same410 save/exact reload, source device; never an optimizer step."""
    from .semantic_rr_capture_migration import _shape_env
    from .semantic_migration import checkpoint_metadata
    from .semantic_training import construct_semantic_runner, load_semantic_checkpoint, save_semantic_checkpoint
    checkpoint = Path(checkpoint).resolve(strict=True); output = Path(output_checkpoint).resolve()
    if output == checkpoint or output.exists() or output.with_name(output.stem + "_manifest.json").exists():
        raise FileExistsError("RR knee v3 publication must be a new immutable checkpoint")
    metadata = checkpoint_metadata(checkpoint)
    verified = validate_rr_capture_knee_migration(checkpoint, current_contract, plan_path)
    device = metadata["runner_config"]["device"]
    def make():
        return construct_semantic_runner(_shape_env(410, device), seed=metadata["seed"], device=device,
            policy_version=RR_CAPTURE_POLICY, observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT, initialize_actor=False)[0]
    runner = make()
    infos = load_semantic_checkpoint(runner, checkpoint, contract=current_contract, seed=metadata["seed"], migration=verified)
    if infos.get(MIGRATION) != verified:
        raise RuntimeError("official RR knee v3 loader did not record its reviewed receipt")
    infos.update(runtime_contract=dict(current_contract), resume_migration=verified, old_rollout_inherited=False,
        migration_publication_context="zero_update_same410_hip_then_knee_control_revision_no_environment",
        source_training_context={k:copy.deepcopy(metadata[k]) for k in ("stage", "sampling", "implemented_reset_sampling",
            "phase_suffix_curriculum_implemented", "execution_topology", "curriculum_epoch") if k in metadata})
    path, manifest = save_semantic_checkpoint(runner, output, infos)
    loaded = load_semantic_checkpoint(make(), path, contract=current_contract, seed=metadata["seed"])
    if any(loaded[k] != metadata[k] for k in preserved_keys(metadata)) or loaded.get(MIGRATION) != verified:
        raise RuntimeError("RR410 knee v3 publication changed saved state or existing lineage")
    return {"checkpoint":str(path), "manifest":str(manifest), "save_load_round_trip":True,
        "observation_dimension":410, **{k:loaded[k] for k in COUNTERS},
        **dict.fromkeys(("migration_added_policy_decisions", "migration_added_ppo_updates",
                        "migration_added_optimizer_steps", "migration_added_auxiliary_updates"), 0),
        "rr_capture_transfer_branch_counts":loaded["rr_capture_transfer_branch_counts"],
        "original_rr_branch_preserved":True, "feedback_revision":FEEDBACK_REVISION}

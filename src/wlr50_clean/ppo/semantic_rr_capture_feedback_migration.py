"""One reviewed RR410 feedback-control revision; exact state, fresh rollout."""
from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import subprocess

from .semantic_rr_capture_profile import RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_LAYOUT

SCHEMA = "wlr50_clean.rr_capture_feedback_same410.v2"
FACTOR_KEY = "rr_capture_feedback_peak_v2_factor"
MIGRATION = "rr_capture_feedback_peak_v2_migration"
FEEDBACK_REVISION = "window_peak_progress_v2"
WINDOW_REFERENCE_SEMANTICS = "public_window_peak_gap_reuses_existing_window_start_gap_scalar_upward_motion_never_resets_elapsed"
SOURCE_HEAD = "a54678ceef5868e419e55840cea34899854bc726"
EXPERIMENT = "rr_capture_then_rl_transfer_v1"
SOURCE_REVISION = "rr_capture_then_rl_transfer_v1_hip_only_direction_check"
TARGET_REVISION = "rr_capture_then_rl_transfer_v2_peak_gap_feedback"
ASSIST = "src/wlr50_clean/ppo/semantic_rr_capture_assist.py"
MODULE = "src/wlr50_clean/ppo/semantic_rr_capture_feedback_migration.py"
PROFILE = f"configs/ppo_{EXPERIMENT}/execution_profile.yaml"
ALLOWED_FILES = frozenset({ASSIST, MODULE, PROFILE,
    "src/wlr50_clean/ppo/semantic_backend.py", "src/wlr50_clean/ppo/semantic_training.py",
    "src/wlr50_clean/ppo/semantic_migration.py"})
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
PRIOR_BRANCHES = ("p05_capture_assist", "capture_feedback_semantics", "rr_postcross_workspace",
    "rr_receiver_retirement_v2", "p05_preedge_approach_recovery", "rr_capture_transfer")


def preserved_keys(metadata):
    from .semantic_rr_workspace_migration import preserved_keys as previous_keys
    return tuple(sorted(set(previous_keys(metadata)) | {"seed", "execution_topology"} | {
        name + suffix for name in PRIOR_BRANCHES for suffix in ("_branch", "_branch_counts", "_migration")}
        | {k for k in metadata if k.startswith("new_mdp_") or "origin" in k
           or k in ("resume_ancestry", "policy_distribution_migration_evidence",
                    "observation_scale_compensation_evidence", "observation_append_evidence")}))


def rr_capture_feedback_factor(metadata, old, new, *, reason, reviewed_code_sha256,
                               source_profile, target_profile):
    from .semantic_migration import source_num_envs, digest
    from .semantic_policy_distribution import policy_contract, CONFIG_NAMES
    canonical = policy_contract(RR_CAPTURE_POLICY, observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
    if (not isinstance(reason, str) or not reason.strip() or metadata.get("semantic_version") != "v3"
            or source_num_envs(metadata) != 1 or old.get("source_git_commit") != SOURCE_HEAD
            or old.get("experiment_id") != EXPERIMENT or new.get("experiment_id") != EXPERIMENT
            or metadata.get("policy_contract") != canonical or canonical["observation_dimension"] != 410):
        raise ValueError("RR feedback requires the exact a546 RR410 v3 N1 source, not a 389 append")
    variable = {"files", "runtime_content_sha256", "source_git_commit", "selected_configuration"}
    if ({k:v for k,v in old.items() if k not in variable}
            != {k:v for k,v in new.items() if k not in variable}):
        raise ValueError("RR feedback cannot alter runtime profile, physics, rates or budgets")
    if (MODULE in old["files"] or set(old["files"]) - set(new["files"])
            or set(new["files"]) - set(old["files"]) != {MODULE}):
        raise ValueError("RR feedback may add only its own module and delete no source files")
    delta = sorted(p for p in new["files"] if old["files"].get(p) != new["files"][p])
    if (not {ASSIST, MODULE, PROFILE}.issubset(delta) or not set(delta) <= ALLOWED_FILES
            or dict(reviewed_code_sha256) != {p:new["files"][p] for p in delta}):
        raise ValueError("RR feedback requires complete exact reviewed hashes within its finite scope")
    for contract in (old, new):
        selected = contract.get("selected_configuration", {})
        if set(selected) != CONFIG_NAMES or any(binding != {
                "path":f"configs/ppo_{EXPERIMENT}/{name}",
                "sha256":contract["files"].get(f"configs/ppo_{EXPERIMENT}/{name}")}
                for name,binding in selected.items()):
            raise ValueError("RR feedback requires all six selected configuration bindings")
    if any(old["selected_configuration"][name] != new["selected_configuration"][name]
           for name in CONFIG_NAMES if name != "execution_profile.yaml"):
        raise ValueError("RR feedback cannot alter the other five configuration bytes")
    if (not isinstance(source_profile, dict) or source_profile.get("revision") != SOURCE_REVISION
            or "rr_capture_feedback_revision" in source_profile
            or source_profile.get("rr_capture_assist_mode") != "rr_hip_only_capture_v1"
            or source_profile.get("rr_capture_wheel_mode") != "off"
            or target_profile != {**source_profile, "revision":TARGET_REVISION,
                                  "rr_capture_feedback_revision":FEEDBACK_REVISION}):
        raise ValueError("execution profile permits only its revision and explicit RR feedback opt-in")
    if MIGRATION in metadata or any(k.startswith("rr_capture_feedback_peak_v2_") for k in metadata):
        raise ValueError("RR feedback boundary cannot be repeated or replace prior lineage")
    if any(k not in metadata for k in preserved_keys(metadata)):
        raise ValueError("RR feedback requires full prior state, origins and AUX lineage")
    if any(type(metadata[k]) is not int or metadata[k] < 0 for k in COUNTERS):
        raise ValueError("RR feedback source counters must be nonnegative integers")
    for name in PRIOR_BRANCHES:
        branch = metadata[name + "_branch"]
        origin = branch.get("counter_origin", {}) if isinstance(branch, dict) else {}
        if (set(origin) != set(COUNTERS) or not isinstance(metadata[name + "_migration"], dict)
                or any(type(origin[k]) is not int or not 0 <= origin[k] <= metadata[k] for k in COUNTERS)
                or metadata[name + "_branch_counts"] != {k:metadata[k]-origin[k] for k in COUNTERS}):
            raise ValueError("RR feedback requires intact independent origins and actual branch counts")
    if (metadata["rr_capture_transfer_branch"].get("schema") != "wlr50_clean.rr_capture_transfer_append.v1"
            or not isinstance(metadata["rr_postcross_workspace_branch"].get("front_rehearsal_auxiliary"), dict)
            or not isinstance(metadata["task_conditioned_hip_wheel_branch"].get("auxiliary_mean_learning"), dict)):
        raise ValueError("RR feedback requires the existing RR branch and both complete AUX objects")
    rate = metadata["optimizer_learning_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
        raise ValueError("RR feedback requires actual positive effective Adam LR")
    return {"schema":SCHEMA, "review_reason":reason.strip(),
        "source_feedback_revision":"implicit_window_start_net_progress_v1",
        "target_feedback_revision":FEEDBACK_REVISION, "window_reference_semantics":WINDOW_REFERENCE_SEMANTICS,
        "reviewed_code_sha256":dict(reviewed_code_sha256),
        "source_execution_profile_sha256":digest(source_profile), "target_execution_profile_sha256":digest(target_profile),
        "observation_contract":{"source_policy_contract":canonical, "target_policy_contract":copy.deepcopy(canonical),
            "observation_layout":RR_CAPTURE_OBSERVATION_LAYOUT, "observation_dimension":410,
            "action_dimension":12, "num_envs":1, "parameter_mapping":"identity_all_parameters_and_buffers"},
        "observation_shape_changed":False, "observation_codec_changed":False,
        "observation_semantics_changed":["RR assist window_start_gap_m scalar index7 / observation396 becomes explicit window peak"],
        "same_numeric_input_policy_mapping_preserved":True, "same_physical_state_action_equivalence_claimed":False,
        "controller_transition_semantics_changed":True, "capture_assist_changed":True, "same_mdp_claimed":False,
        "pre_cross_ownership_semantics":"qualified_P09_AIR_XY_hold_previous_final_only_no_descent_before_real_cross",
        "capture_owner_acquisition_changed":True, "pre_cross_descent_allowed":False,
        "policy_kernel_changed":False, "reward_changed":False, "nominal_changed":False,
        "caps_changed":False, "sigma_changed":False, "physical_dynamics_changed":False,
        "physical_task_acceptance_rules_changed":False,
        "preserved_metadata_sha256":{k:digest(metadata[k]) for k in preserved_keys(metadata)},
        "source_resume_migration":copy.deepcopy(metadata.get("resume_migration")),
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
        raise ValueError("ambiguous RR feedback constant: " + name)
    return values[0] if values else None


def build_rr_capture_feedback_migration(checkpoint, current_contract, *, reason, reviewed_code_sha256, project_root=None):
    import yaml
    from .semantic_migration import PROJECT_ROOT, checkpoint_metadata, _contract, _version_bytes, file_sha, digest
    root = Path(project_root or PROJECT_ROOT).resolve(); checkpoint = Path(checkpoint).resolve(strict=True)
    metadata = checkpoint_metadata(checkpoint)
    old, new = _contract(metadata["runtime_contract"]), _contract(current_contract)
    source_profile = yaml.safe_load(_version_bytes(root, old, PROFILE))
    target_profile = yaml.safe_load((root / PROFILE).read_bytes())
    factor = rr_capture_feedback_factor(metadata, old, new, reason=reason, reviewed_code_sha256=reviewed_code_sha256,
        source_profile=source_profile, target_profile=target_profile)
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if new["source_git_commit"] != head or head == old["source_git_commit"]:
        raise ValueError("RR feedback target must name a new actual committed runtime HEAD")
    dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all", "--",
        "src/wlr50_clean", "scripts", "configs", "artifacts/ppo_phase_v1_start", "pyproject.toml"],
        check=True, capture_output=True, text=True).stdout.strip()
    if dirty:
        raise ValueError("RR feedback target runtime must be clean and committed")
    for path, sha in new["files"].items():
        if file_sha(root / path) != sha:
            raise ValueError("RR feedback target runtime bytes differ: " + path)
    for name, binding in old["selected_configuration"].items():
        if name != "execution_profile.yaml" and _version_bytes(root, old, binding["path"]) != (root / binding["path"]).read_bytes():
            raise ValueError("RR feedback changed a protected configuration's bytes")
    before, after = _version_bytes(root, old, ASSIST), (root / ASSIST).read_bytes()
    for name, value in (("RR_CAPTURE_FEEDBACK_REVISION", FEEDBACK_REVISION),
                        ("RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS", WINDOW_REFERENCE_SEMANTICS)):
        if _literal(before, name) is not None or _literal(after, name) != value:
            raise ValueError("RR feedback needs the exact implicit-v1 to explicit peak-v2 constants")
    names = _literal(before, "RR_CAPTURE_ASSIST_FEATURE_NAMES")
    if (not isinstance(names, tuple) or len(names) != 14 or names[7] != "window_start_gap_m"
            or _literal(after, "RR_CAPTURE_ASSIST_FEATURE_NAMES") != names
            or _literal(before, "_SCALES") != _literal(after, "_SCALES")):
        raise ValueError("RR feedback must preserve all14 scalar names, positions and numerical codec")
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


def validate_rr_capture_feedback_migration(checkpoint, current_contract, plan_path, *, project_root=None):
    from .semantic_migration import file_sha
    path = Path(plan_path).resolve(strict=True); supplied = json.loads(path.read_text(encoding="utf-8"))
    expected = build_rr_capture_feedback_migration(checkpoint, current_contract, reason=supplied.get("reason"),
        reviewed_code_sha256=supplied.get(FACTOR_KEY, {}).get("reviewed_code_sha256", {}), project_root=project_root)
    if supplied != expected:
        raise ValueError("RR feedback plan differs from immutable source and target runtime")
    return {**expected, "plan_path":str(path), "plan_sha256":file_sha(path)}


def record_loaded_rr_capture_feedback(runner, infos, verified):
    """After the official full-state load; add only the reviewed receipt."""
    import torch
    from .semantic_migration import digest
    from .semantic_training import parameter_hash, state_hash, _normalizers
    from .rl_library_wrapper import optimizer_learning_rate, capture_training_rng_state
    factor = verified[FACTOR_KEY]
    if verified.get("schema") != SCHEMA or factor.get("schema") != SCHEMA or MIGRATION in infos:
        raise RuntimeError("RR feedback must be explicit and unrepeated")
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
        raise RuntimeError("RR feedback altered exact source state, LR, RNG or inherited a rollout")
    return {**infos, MIGRATION:copy.deepcopy(verified)}


def publish_rr_capture_feedback_checkpoint(checkpoint, current_contract, plan_path, output_checkpoint):
    """Official same410 save/exact reload, source device; never an optimizer step."""
    from .semantic_rr_capture_migration import _shape_env
    from .semantic_migration import checkpoint_metadata
    from .semantic_training import construct_semantic_runner, load_semantic_checkpoint, save_semantic_checkpoint
    checkpoint = Path(checkpoint).resolve(strict=True); output = Path(output_checkpoint).resolve()
    if output == checkpoint or output.exists() or output.with_name(output.stem + "_manifest.json").exists():
        raise FileExistsError("RR feedback publication must be a new immutable checkpoint")
    metadata = checkpoint_metadata(checkpoint)
    verified = validate_rr_capture_feedback_migration(checkpoint, current_contract, plan_path)
    device = metadata["runner_config"]["device"]
    def make():
        return construct_semantic_runner(_shape_env(410, device), seed=metadata["seed"], device=device,
            policy_version=RR_CAPTURE_POLICY, observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT, initialize_actor=False)[0]
    runner = make()
    infos = load_semantic_checkpoint(runner, checkpoint, contract=current_contract, seed=metadata["seed"], migration=verified)
    if infos.get(MIGRATION) != verified:
        raise RuntimeError("official RR feedback loader did not record its reviewed receipt")
    infos.update(runtime_contract=dict(current_contract), resume_migration=verified, old_rollout_inherited=False,
        migration_publication_context="zero_update_same410_feedback_revision_no_environment",
        source_training_context={k:copy.deepcopy(metadata[k]) for k in ("stage", "sampling", "implemented_reset_sampling",
            "phase_suffix_curriculum_implemented", "execution_topology", "curriculum_epoch") if k in metadata})
    path, manifest = save_semantic_checkpoint(runner, output, infos)
    loaded = load_semantic_checkpoint(make(), path, contract=current_contract, seed=metadata["seed"])
    if any(loaded[k] != metadata[k] for k in preserved_keys(metadata)) or loaded.get(MIGRATION) != verified:
        raise RuntimeError("RR410 feedback publication changed saved state or existing lineage")
    return {"checkpoint":str(path), "manifest":str(manifest), "save_load_round_trip":True,
        "observation_dimension":410, **{k:loaded[k] for k in COUNTERS},
        **dict.fromkeys(("migration_added_policy_decisions", "migration_added_ppo_updates",
                        "migration_added_optimizer_steps", "migration_added_auxiliary_updates"), 0),
        "rr_capture_transfer_branch_counts":loaded["rr_capture_transfer_branch_counts"],
        "original_rr_branch_preserved":True, "feedback_revision":FEEDBACK_REVISION}

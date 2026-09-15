"""Narrow sampling boundary: synthetic Git only, no user checkpoint/Isaac."""
from __future__ import annotations
import copy
import json
import shutil
from types import SimpleNamespace
import pytest
from wlr50_clean.ppo import semantic_migration as m
from wlr50_clean.ppo.semantic_policy_distribution import HISTORY_POLICY, HISTORY_TEMPERED_POLICY
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_semantic_role_observation_migration import NAMES, PROTECTED, commit, git, source_metadata, write_json

CONFIG = "configs/ppo_fsm_reference_p09_stable_v2"

def bind(root, contract):
    contract["files"] = {p: m.file_sha(root / p) for p in contract["files"]}
    contract["runtime_content_sha256"] = m.digest(contract["files"])
    contract["selected_configuration"] = {n: {"path": f"{CONFIG}/{n}",
        "sha256": contract["files"][f"{CONFIG}/{n}"]} for n in NAMES}

@pytest.fixture(scope="module")
def prototype(tmp_path_factory):
    root = tmp_path_factory.mktemp("temperature_git")
    paths = m.EXPLORATION_TEMPERATURE_FILES | {f"{CONFIG}/{n}" for n in NAMES} | {PROTECTED}
    for path in paths:
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((m.PROJECT_ROOT / path).read_bytes())
    prefix = "src/wlr50_clean/ppo/"
    modules = {
        prefix+"semantic_history_actor.py": "LOCKED = 1\ndef history_conditioned_head():\n    return 9\nclass SemanticHistoryMLPModel:\n    pass\n",
        prefix+"semantic_policy_distribution.py": "LOCKED = 1\ndef policy_contract():\n    return 1\ndef configure_policy_distribution():\n    return 1\ndef supported_heteroscedastic_contract_version():\n    return 1\ndef policy_version_from_metadata():\n    return 1\n",
        prefix+"semantic_training.py": "LOCKED = 1\ndef semantic_runner_config():\n    return 1\ndef load_semantic_checkpoint():\n    return 1\ndef train_semantic():\n    return 9\n",
        prefix+"semantic_cli.py": "LOCKED = 1\ndef _preflight_checkpoint():\n    return 1\ndef _resolved_policy_version():\n    return 1\ndef train():\n    return 9\n",
    }
    for p, text in modules.items(): (root/p).write_text(text, encoding="utf-8")
    git(root, "init")
    git(root, "config", "user.name", "Temperature fixture")
    git(root, "config", "user.email", "temperature@example.invalid")
    old = {"source_git_commit": commit(root, "before temperature"), "files": dict.fromkeys(sorted(paths)),
        "semantic_version": "v3", "experiment_id": "fsm_reference_p09_stable_v2",
        "frozen_A_files": {"frozen": "a"*64}, "physics_hz": 120., "decision_hz": 15.,
        "task_timeout_s": 200., "timeout_bootstrap": False, "rsl_rl_version": "5.0.1",
        "local_runtime_versions": {"fixture": "CPU only"}}
    bind(root, old)
    for p, text in modules.items():
        text = text.replace("return 1", "return 2")
        if p.endswith("semantic_history_actor.py"):
            text += "class SemanticTemperedHistoryMLPModel:\n    pass\n"
        if p.endswith("semantic_training.py"):
            text += "def _validated_exploration_temperature_factor():\n    return 2\n"
        if p.endswith("semantic_policy_distribution.py"):
            text += "HISTORY_TEMPERED_POLICY = 'temperature'\nHISTORY_TEMPERED_ACTOR_CLASS = 'fixture'\n"
        (root/p).write_text(text, encoding="utf-8")
    new = copy.deepcopy(old)
    new["source_git_commit"] = commit(root, "temperature candidate")
    bind(root, new)
    return SimpleNamespace(root=root, old=old, new=new)

@pytest.fixture
def f(prototype, tmp_path):
    root = tmp_path / "case"
    shutil.copytree(prototype.root, root)
    checkpoint = root / "outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/source.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"opaque synthetic fixture, not torch loaded")
    metadata = source_metadata(checkpoint, prototype.old, layout=ROLE_OBSERVATION_LAYOUT)
    sidecar = checkpoint.with_name("source_manifest.json")
    write_json(sidecar, metadata)
    return SimpleNamespace(root=root, old=copy.deepcopy(prototype.old), new=copy.deepcopy(prototype.new),
        checkpoint=checkpoint, metadata=metadata, sidecar=sidecar)

def record(f, **options):
    delta = sorted(p for p in set(f.old["files"]) | set(f.new["files"])
        if f.old["files"].get(p) != f.new["files"].get(p))
    return m.build_migration_plan(f.checkpoint, f.new, allowed_changed_files=delta,
        reason="Sampling coverage candidate; not physical success or sigma-cause claim",
        exploration_temperature_review={"reason": "Reviewed fixed0.5 all-stochastic-path scale",
            "reviewed_code_sha256": {p: f.new["files"][p] for p in m.EXPLORATION_TEMPERATURE_FILES}},
        project_root=f.root, **options)

def test_bound_plan_roundtrip_preserves_all_learned_state(f):
    original = f.checkpoint.read_bytes(), f.sidecar.read_bytes()
    plan = record(f)
    factor = plan["exploration_temperature_factor"]
    assert factor["schema"] == m.EXPLORATION_TEMPERATURE_SCHEMA
    assert factor["source_policy_version"] == HISTORY_POLICY
    assert factor["target_policy_version"] == HISTORY_TEMPERED_POLICY
    assert factor["target_exploration_std_temperature"] == .5
    assert factor["kernel_changed"] and not factor["physical_mdp_changed"]
    assert not factor["nominal_control_changed"] and not factor["action_ranges_changed"]
    assert not factor["reward_changed"] and not factor["task_acceptance_changed"]
    assert factor["migration_added_updates"] == 0
    assert plan["preserve_actor_critic_optimizer_normalizer_rng_and_budget"]
    assert plan["discard_old_rollout_storage"] and plan["physics_resume"] == "fresh_legal_P01_reset"
    assert plan["observation_dimension"] == 372 and plan["action_dimension"] == 12
    path = f.root / "plan.json"
    write_json(path, plan)
    assert m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)["exploration_temperature_factor"] == factor
    assert original == (f.checkpoint.read_bytes(), f.sidecar.read_bytes())

@pytest.mark.parametrize("field", ["target_exploration_std_temperature", "target_policy_contract", "target_runner_config", "optimizer"])
def test_tampered_factor_rejected(f, field):
    plan = record(f)
    plan["exploration_temperature_factor"][field] = "tampered"
    path = f.root / "tampered.json"
    write_json(path, plan)
    with pytest.raises(ValueError, match="exactly bound"):
        m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)

@pytest.mark.parametrize("name", sorted(NAMES))
def test_no_configuration_byte_change_is_permitted(f, name):
    path = f.root / CONFIG / name
    path.write_bytes(path.read_bytes()+b"\n")
    bind(f.root, f.new)
    with pytest.raises(ValueError, match="configuration bindings|boundary|config bytes"):
        record(f)

@pytest.mark.parametrize("module,old_text,new_text", [
    ("semantic_history_actor", "return 9", "return 10"),
    ("semantic_training", "return 9", "return 10"),
    ("semantic_cli", "return 9", "return 10"),
    ("semantic_policy_distribution", "LOCKED = 1", "LOCKED = 2"),
])
def test_unchanged_mean_physics_and_training_regions_are_protected(f, module, old_text, new_text):
    path = f.root / f"src/wlr50_clean/ppo/{module}.py"
    path.write_text(path.read_text().replace(old_text,new_text), encoding="utf-8")
    bind(f.root, f.new)
    with pytest.raises(ValueError, match="outside reviewed"):
        record(f)

def test_added_physical_call_inside_reviewed_loader_is_rejected(f):
    path = f.root / "src/wlr50_clean/ppo/semantic_training.py"
    path.write_text(path.read_text().replace("def load_semantic_checkpoint():\n    return 2",
        "def load_semantic_checkpoint():\n    backend.step()\n    return 2"), encoding="utf-8")
    bind(f.root, f.new)
    with pytest.raises(ValueError, match="step calls"):
        record(f)

@pytest.mark.parametrize("field,value", [("physics_hz",60.),("task_timeout_s",201.),("experiment_id","all_stage_acceptance_v1")])
def test_physical_metadata_or_experiment_changes_rejected(f, field, value):
    f.new[field] = value
    with pytest.raises(ValueError): record(f)

@pytest.mark.parametrize("factor", ["height_recovery_review","body_reward_review","timing_review","video_review","execution_evidence","evaluator_review","prior_evidence","qualification_evidence"])
def test_other_migration_factors_cannot_mix(f, factor):
    with pytest.raises(ValueError, match="cannot mix"):
        record(f, **{factor:{}})

def test_source_runner_config_cannot_change_ppo_settings(f):
    f.metadata["runner_config"]["algorithm"]["gamma"] = .9
    write_json(f.sidecar, f.metadata)
    with pytest.raises(ValueError): record(f)

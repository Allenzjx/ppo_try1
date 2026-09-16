"""Final-stop owner-only migration: real temp Git and official CPU load, no Isaac."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import semantic_cli as cli, semantic_migration as m
from wlr50_clean.ppo.semantic_policy_distribution import HISTORY_TEMPERED_POLICY
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_semantic_role_observation_migration import NAMES, commit, git, source_metadata, write_json

CONFIG = "configs/ppo_task_first_recovery_v1"
ADAPTER = "src/wlr50_clean/ppo/semantic_residual_adapter.py"
GEOMETRY = "src/wlr50_clean/ppo/semantic_nominal_geometry.py"
PROTECTED = "src/wlr50_clean/ppo/semantic_reward.py"
HISTORY = "src/wlr50_clean/ppo/semantic_history_actor.py"


def bind(root, contract):
    contract["files"] = {p: m.file_sha(root / p) for p in contract["files"]}
    contract["runtime_content_sha256"] = m.digest(contract["files"])
    contract["selected_configuration"] = {n: {"path": f"{CONFIG}/{n}",
        "sha256": contract["files"][f"{CONFIG}/{n}"]} for n in NAMES}


@pytest.fixture(scope="module")
def prototype(tmp_path_factory):
    root = tmp_path_factory.mktemp("final_stop_git")
    paths = m.FINAL_STOP_HANDOFF_FILES | {f"{CONFIG}/{n}" for n in NAMES} | {PROTECTED, HISTORY, m.SUPERVISOR, ADAPTER, GEOMETRY}
    for relative in paths:
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((m.PROJECT_ROOT / relative).read_bytes())
    source_text = ("class TaskEvaluator:\n    def _all_stage_finish(self):\n        return False\n\n"
        "class NominalMotionProvider:\n    def _observe_final_stop_owner(self, task, observation):\n        return False\n"
        "    def evaluate(self):\n        return 1\n")
    (root / m.SUPERVISOR).write_text(source_text, encoding="utf-8")
    git(root, "init")
    git(root, "config", "user.name", "Final-stop boundary fixture")
    git(root, "config", "user.email", "composition@example.invalid")
    old = {"source_git_commit": commit(root, "source execution"), "files": dict.fromkeys(sorted(paths)),
        "semantic_version": "v3", "experiment_id": "task_first_recovery_v1",
        "frozen_A_files": {"frozen": "a" * 64}, "physics_hz": 120., "decision_hz": 15.,
        "task_timeout_s": 200., "timeout_bootstrap": False, "rsl_rl_version": "5.0.1",
        "local_runtime_versions": {"fixture": "CPU not Isaac"}}
    bind(root, old)
    changed_text = source_text.replace("def _observe_final_stop_owner(self, task, observation):\n        return False",
        "def _observe_final_stop_owner(self, task, observation):\n        return True")
    (root / m.SUPERVISOR).write_text(changed_text, encoding="utf-8")
    new = copy.deepcopy(old)
    new["source_git_commit"] = commit(root, "composition fix")
    bind(root, new)
    return SimpleNamespace(root=root, old=old, new=new)


@pytest.fixture
def f(prototype, tmp_path):
    root = tmp_path / "case"
    shutil.copytree(prototype.root, root)
    checkpoint = root / "outputs/ppo_task_first_recovery_v1/checkpoints/history/source.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"opaque metadata-only fixture")
    metadata = source_metadata(checkpoint, prototype.old, layout=ROLE_OBSERVATION_LAYOUT,
        policy=HISTORY_TEMPERED_POLICY)
    sidecar = checkpoint.with_name("source_manifest.json")
    write_json(sidecar, metadata)
    return SimpleNamespace(root=root, checkpoint=checkpoint, sidecar=sidecar, metadata=metadata,
        old=copy.deepcopy(prototype.old), new=copy.deepcopy(prototype.new))


def record(f, **extra):
    delta = sorted(p for p in f.old["files"].keys() | f.new["files"].keys()
        if f.old["files"].get(p) != f.new["files"].get(p))
    return m.build_migration_plan(f.checkpoint, f.new, allowed_changed_files=delta,
        reason="Acquire existing final-stop ownership at actual post-completion trigger",
        final_stop_handoff_review={"reason": "Only final-stop owner helper changed; evaluator and every other method protected",
            "reviewed_code_sha256": {p: f.new["files"][p] for p in delta}}, project_root=f.root, **extra)


def test_plan_is_execution_change_not_reward_only_and_roundtrip(f):
    original = f.checkpoint.read_bytes(), f.sidecar.read_bytes()
    plan = record(f)
    factor = plan["final_stop_handoff_factor"]
    assert factor["schema"] == m.FINAL_STOP_HANDOFF_SCHEMA
    assert factor["supervisor_scope"]["method"] == "NominalMotionProvider._observe_final_stop_owner"
    assert factor["supervisor_scope"]["evaluator_and_other_methods_unchanged"]
    assert factor["task_evaluator_changed"] is False and factor["fixed_post_completion_window_changed"] is False
    assert plan["target_runtime_content_sha256"] == f.new["runtime_content_sha256"]
    assert factor["action_execution_changed"] and factor["transition_execution_semantics_changed"]
    assert "physical_mdp_changed" not in factor and "task_first_reward_factor" not in plan
    assert not factor["reward_changed"] and factor["quality_epsilon"] == 0.
    assert not factor["physical_scene_changed"] and not factor["actuator_capability_changed"]
    assert factor["configuration_bindings"] == f.old["selected_configuration"] == f.new["selected_configuration"]
    assert factor["observation_contract"]["observation_dimension"] == 372
    assert plan["discard_old_rollout_storage"] and factor["migration_added_updates"] == 0
    path = f.root / "plan.json"
    write_json(path, plan)
    assert m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)["final_stop_handoff_factor"] == factor
    assert original == (f.checkpoint.read_bytes(), f.sidecar.read_bytes())


@pytest.mark.parametrize("kind", ["physics", "reward", "caps", "observation", "nominal", "history", "adapter", "geometry", "no_core_change", "missing_delta", "layout"])
def test_rejects_changes_outside_execution_repair(f, kind):
    if kind == "physics":
        f.new["physics_hz"] = 60.
    elif kind == "layout":
        f.metadata["policy_contract"]["rho"] = .8
        write_json(f.sidecar, f.metadata)
    elif kind == "no_core_change":
        f.new = copy.deepcopy(f.old)
    elif kind == "missing_delta":
        f.new["files"]["missing.py"] = "b" * 64
        f.new["runtime_content_sha256"] = m.digest(f.new["files"])
    else:
        relative = {"reward": PROTECTED, "caps": f"{CONFIG}/execution_profile.yaml",
            "observation": f"{CONFIG}/observation_schema.json", "nominal": m.SUPERVISOR, "history": HISTORY, "adapter": ADAPTER, "geometry": GEOMETRY}[kind]
        (f.root / relative).write_bytes((f.root / relative).read_bytes() + b"\n# not part of composition repair\n")
        bind(f.root, f.new)
    with pytest.raises(ValueError):
        record(f)


@pytest.mark.parametrize("other", ["task_first_reward_review", "exploration_temperature_review", "timing_review", "execution_evidence", "execution_composition_review"])
def test_factor_cannot_mix_other_boundaries(f, other):
    with pytest.raises(ValueError, match="cannot mix"):
        record(f, **{other: {}})


def test_receipt_tamper_and_target_hash_are_rejected(f):
    plan = record(f)
    plan["final_stop_handoff_factor"]["optimizer"] = "reset moments"
    path = f.root / "tamper.json"
    write_json(path, plan)
    with pytest.raises(ValueError, match="exactly bound"):
        m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)
    (f.root / m.SUPERVISOR).write_text("# changed after review\n", encoding="utf-8")
    with pytest.raises(ValueError, match="target bytes"):
        record(f)


def test_existing_cli_preflight_accepts_only_verified_same372_factor(f, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", f.root)
    path = f.root / "plan.json"
    write_json(path, record(f))
    original = m.validate_migration_plan
    monkeypatch.setattr(m, "validate_migration_plan", lambda cp, contract, p:
        original(cp, contract, p, project_root=f.root))
    args = cli.parser().parse_args(["train", "--semantic-version", "v3", "--experiment-id",
        "task_first_recovery_v1", "--run-dir", str(f.root / "runs/ppo_task_first_recovery_v1/train/new"),
        "--expected-head", f.new["source_git_commit"], "--checkpoint", str(f.checkpoint), "--resume-migration", str(path),
        "--stage", "full_episode", "--from-phase", "P01", "--decisions", "128", "--seed", "1001", "--device", "cpu"])
    cli.validate_request(args)
    cli._preflight_checkpoint(args, f.new)
    assert args._migration_record["final_stop_handoff_factor"]["schema"] == m.FINAL_STOP_HANDOFF_SCHEMA
    assert args._policy_version == HISTORY_TEMPERED_POLICY


def test_official_cpu_load_preserves_learned_state_rng_counters_and_fresh_storage(prototype, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from test_semantic_temperature_loader_cli import make_target
    from wlr50_clean.ppo import semantic_training as t
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        root = tmp_path / "real_cpu"
        shutil.copytree(prototype.root, root)
        t.seed_training_rngs(1001)
        source, source_env = make_target()
        result = t.train_semantic(source, source_env, run_dir=root / "old_run", output_root=root / "old_output",
            stage="full_episode", decisions=128, contract=prototype.old, seed=1001)
        metadata = json.loads(Path(result["checkpoints"][-1]["manifest"]).read_text())
        for key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"):
            metadata.pop(key)
        source.alg.learning_rate = 1e-5
        for group in source.alg.optimizer.param_groups:
            group.update(lr=1e-5, betas=(.87, .996), eps=2e-8)
        cp = root / "learned.pt"
        t.save_semantic_checkpoint(source, cp, metadata)
        saved = m.checkpoint_metadata(cp)
        assert source.alg.optimizer.state_dict()["state"]
        case = SimpleNamespace(root=root, old=prototype.old, new=prototype.new, checkpoint=cp)
        path = root / "plan.json"
        write_json(path, record(case))
        original = m.validate_migration_plan
        monkeypatch.setattr(m, "validate_migration_plan", lambda cp, c, p: original(cp, c, p, project_root=root))
        verified = m.validate_migration_plan(cp, prototype.new, path)
        target, env = make_target()
        loaded = t.load_semantic_checkpoint(target, cp, contract=prototype.new, seed=1001, migration=verified)
        for role in ("actor", "critic"):
            assert t.parameter_hash(getattr(source.alg, role)) == t.parameter_hash(getattr(target.alg, role))
        assert t.state_hash(target.alg.optimizer.state_dict()) == saved["optimizer_state_sha256"]
        assert t.state_hash(t._normalizers(target)) == saved["normalizer_state_sha256"]
        assert t.capture_training_rng_state(seed=1001) == saved["training_rng_state"]
        assert target.alg.learning_rate == t.optimizer_learning_rate(target) == 1e-5
        assert target.alg.storage.step == 0 and target.alg.transition.actions is None and env.core.calls == 0
        for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps", "stage_requested_decisions"):
            assert loaded[key] == saved[key]
        target.alg.storage.step = 1
        with pytest.raises(RuntimeError, match="partial old rollout"):
            t.load_semantic_checkpoint(target, cp, contract=prototype.new, seed=1001, migration=verified)
        target.alg.storage.step = 0
        continued = t.train_semantic(target, env, run_dir=root / "new_run", output_root=root / "new_output",
            stage="full_episode", decisions=128, contract=prototype.new, seed=1001, resume_infos=loaded)
        final, _ = make_target()
        infos = t.load_semantic_checkpoint(final, Path(continued["checkpoints"][-1]["checkpoint"]),
            contract=prototype.new, seed=1001)
        assert (infos["global_policy_decisions"], infos["ppo_updates"], infos["optimizer_steps"]) == (256, 2, 40)
        assert infos["resume_ancestry"]["resume_migration"]["final_stop_handoff_factor"]["schema"] == m.FINAL_STOP_HANDOFF_SCHEMA
    finally:
        torch.set_num_threads(original_threads)

@pytest.mark.parametrize("kind", ["evaluator", "another_nominal_method", "signature", "module_constant"])
def test_precise_method_scope_rejects_other_supervisor_edits(f, kind):
    file = f.root / m.SUPERVISOR
    text = file.read_text()
    if kind == "evaluator":
        text = text.replace("def _all_stage_finish(self):\n        return False",
                            "def _all_stage_finish(self):\n        return True")
    elif kind == "another_nominal_method":
        text = text.replace("return 1", "return 2")
    elif kind == "signature":
        text = text.replace("self, task, observation", "self, task, observation, bypass=True")
    else:
        text += "\nBYPASS = True\n"
    file.write_text(text, encoding="utf-8")
    bind(f.root, f.new)
    with pytest.raises(ValueError, match="outside the exact stop-owner method"):
        record(f)


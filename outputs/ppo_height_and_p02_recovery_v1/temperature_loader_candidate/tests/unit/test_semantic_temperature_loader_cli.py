"""Temperature loader boundary; synthetic CPU checkpoints only, never Isaac.

Small negative/preflight tests stub plan verification. The actual Adam test uses
the independent synthetic-Git builder/validator fixture, not a trusted fake plan.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import semantic_training as t, semantic_cli as cli, semantic_migration as m
from wlr50_clean.ppo.semantic_policy_distribution import HISTORY_POLICY, HISTORY_TEMPERED_POLICY, policy_contract
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_semantic_role_observation_migration import source_metadata, write_json
from test_semantic_exploration_temperature_migration import prototype, record


def temperature_factor(metadata):
    source = policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    target = policy_contract(HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    return {"source_policy_version": HISTORY_POLICY, "target_policy_version": HISTORY_TEMPERED_POLICY,
        "source_policy_contract": source, "target_policy_contract": target,
        "source_runner_config": copy.deepcopy(metadata["runner_config"]),
        "target_runner_config": t.semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
            policy_version=HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT),
        "observation_contract": {"source_policy_contract": source, "target_policy_contract": target,
            "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
            "action_dimension": 12, "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}}


@pytest.fixture
def opaque(tmp_path):
    cp = tmp_path / "source.pt"
    cp.write_bytes(b"opaque CPU preflight fixture; must never reach tensor loader")
    old, new = {"revision": "old-HISTORY"}, {"revision": "new-temperature"}
    meta = source_metadata(cp, old, layout=ROLE_OBSERVATION_LAYOUT)
    write_json(cp.with_name("source_manifest.json"), meta)
    plan = {"plan_path": str(tmp_path / "plan.json"), "observation_dimension": 372, "action_dimension": 12,
        "exploration_temperature_factor": temperature_factor(meta)}
    return SimpleNamespace(checkpoint=cp, metadata=meta, old=old, new=new, plan=plan)


def preflight_args(f):
    return SimpleNamespace(checkpoint=f.checkpoint, semantic_version="v3",
        experiment_id="fsm_reference_p09_stable_v2", command="train", num_envs=1,
        seed=1001, device="cpu", new_mdp_warm_start=False, resume_migration=Path(f.plan["plan_path"]),
        policy_distribution_migration=False, target_policy_version=None)


def test_cli_switches_only_after_verified_factor_and_resolves_explicit_role(opaque, monkeypatch):
    calls = []
    def verified(*args):
        calls.append(args)
        return copy.deepcopy(opaque.plan)
    monkeypatch.setattr(m, "validate_migration_plan", verified)
    args = preflight_args(opaque)
    cli._preflight_checkpoint(args, opaque.new)
    assert len(calls) == 1
    assert cli._resolved_policy_version(args) == HISTORY_TEMPERED_POLICY
    assert cli._resolved_policy_contract(args) == opaque.plan["exploration_temperature_factor"]["target_policy_contract"]
    assert args._migration_record == opaque.plan


@pytest.mark.parametrize("fault", ["source_config", "target_config", "tau", "source_contract", "target_contract",
    "source_version", "target_version", "observation", "N8", "mixed", "wrong_seed", "device"])
def test_cli_factor_tampering_never_selects_new_actor(opaque, monkeypatch, fault):
    plan = copy.deepcopy(opaque.plan)
    factor = plan["exploration_temperature_factor"]
    args = preflight_args(opaque)
    if fault == "source_config": factor["source_runner_config"]["algorithm"]["learning_rate"] = 1e-5
    elif fault == "target_config": factor["target_runner_config"]["algorithm"]["gamma"] = .95
    elif fault == "tau": factor["target_runner_config"]["actor"]["exploration_std_temperature"] = .7
    elif fault == "source_contract": factor["source_policy_contract"]["rho"] = .8
    elif fault == "target_contract": factor["target_policy_contract"]["rho"] = .8
    elif fault == "source_version": factor["source_policy_version"] = HISTORY_TEMPERED_POLICY
    elif fault == "target_version": factor["target_policy_version"] = HISTORY_POLICY
    elif fault == "observation": factor["observation_contract"]["observation_dimension"] = 324
    elif fault == "N8": args.num_envs = 8
    elif fault == "mixed": plan["height_recovery_factor"] = {}
    elif fault == "wrong_seed": args.seed = 1002
    elif fault == "device": args.device = "cuda:0"
    monkeypatch.setattr(m, "validate_migration_plan", lambda *a: copy.deepcopy(plan))
    with pytest.raises((RuntimeError, ValueError)):
        cli._preflight_checkpoint(args, opaque.new)
    if fault != "wrong_seed":
        assert args._policy_version == HISTORY_POLICY


@pytest.mark.parametrize("version,layout", [("v2", ROLE_OBSERVATION_LAYOUT), ("v3", None), ("v3", "unknown")])
def test_tempered_runner_rejects_non_v3_or_non_role_layout(version, layout):
    with pytest.raises(ValueError):
        t.semantic_runner_config(seed=1001, device="cpu", semantic_version=version,
            policy_version=HISTORY_TEMPERED_POLICY, observation_layout=layout)


def make_target():
    from test_semantic_role_append_training import Core372
    env = t.SemanticRslAdapter(Core372(), seed=1001, device="cpu")
    env.cfg["semantic_version"] = "v3"
    target, _ = t.construct_semantic_runner(env, seed=1001, device="cpu", initialize_actor=False,
        policy_version=HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    return target, env


@pytest.fixture
def cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield torch
    torch.set_num_threads(before)


@pytest.mark.parametrize("fault", ["missing_plan", "revalidated_changed", "old_actor", "target_config",
    "partial_storage", "pending_action", "mixed_execution", "mixed_instrumentation", "mixed_warm_start"])
def test_loader_rejects_before_tensor_load(opaque, monkeypatch, cpu, fault):
    target, env = make_target()
    supplied = copy.deepcopy(opaque.plan)
    validated = copy.deepcopy(supplied)
    kwargs = {"migration": supplied}
    if fault == "missing_plan": kwargs = {}
    elif fault == "revalidated_changed": validated["changed_after_preflight"] = True
    elif fault == "old_actor": target._semantic_policy_version = HISTORY_POLICY
    elif fault == "target_config": target._semantic_runner_config["actor"]["exploration_std_temperature"] = .9
    elif fault == "partial_storage": target.alg.storage.step = 1
    elif fault == "pending_action": target.alg.transition.actions = cpu.zeros(1, 12)
    elif fault == "mixed_execution": supplied["execution_factor"] = {}; validated = copy.deepcopy(supplied)
    elif fault == "mixed_instrumentation": supplied["instrumentation_observation_contract"] = {}; validated = copy.deepcopy(supplied)
    elif fault == "mixed_warm_start": kwargs["warm_start"] = {}
    monkeypatch.setattr(m, "validate_migration_plan", lambda *a: copy.deepcopy(validated))
    def forbidden(*a, **k):
        raise AssertionError("an invalid temperature request reached the tensor loader")
    monkeypatch.setattr(t, "load_checkpoint_round_trip", forbidden)
    with pytest.raises((RuntimeError, ValueError)):
        t.load_semantic_checkpoint(target, opaque.checkpoint, contract=opaque.new, seed=1001, **kwargs)
    assert env.core.calls == 0


def test_real_cpu_temperature_load_keeps_adam_rng_lr_and_continues(prototype, tmp_path, monkeypatch, cpu):
    from test_semantic_role_append_training import make
    root = tmp_path / "real_cpu"
    shutil.copytree(prototype.root, root)
    t.seed_training_rngs(1001)
    source, source_env = make(True)
    result = t.train_semantic(source, source_env, run_dir=root / "old_run", output_root=root / "old_output",
        stage="full_episode", decisions=128, contract=prototype.old, seed=1001)
    metadata = json.loads(Path(result["checkpoints"][-1]["manifest"]).read_text())
    for key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"): metadata.pop(key)
    source.alg.learning_rate = 1e-5
    for group in source.alg.optimizer.param_groups: group.update(lr=1e-5, betas=(.87, .996), eps=2e-8)
    cp = root / "learned.pt"
    t.save_semantic_checkpoint(source, cp, metadata)
    saved = m.checkpoint_metadata(cp)
    assert source.alg.optimizer.state_dict()["state"]
    f = SimpleNamespace(root=root, old=prototype.old, new=prototype.new, checkpoint=cp)
    plan = root / "plan.json"
    write_json(plan, record(f))
    original = m.validate_migration_plan
    monkeypatch.setattr(m, "validate_migration_plan", lambda cp, c, p: original(cp, c, p, project_root=root))
    verified = m.validate_migration_plan(cp, prototype.new, plan)
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
    continued = t.train_semantic(target, env, run_dir=root / "new_run", output_root=root / "new_output",
        stage="full_episode", decisions=128, contract=prototype.new, seed=1001, resume_infos=loaded)
    final, _ = make_target()
    infos = t.load_semantic_checkpoint(final, Path(continued["checkpoints"][-1]["checkpoint"]),
        contract=prototype.new, seed=1001)
    assert (infos["global_policy_decisions"], infos["ppo_updates"], infos["optimizer_steps"]) == (256, 2, 40)
    assert infos["policy_contract"] == policy_contract(HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    assert infos["resume_ancestry"]["resume_migration"]["exploration_temperature_factor"] == verified["exploration_temperature_factor"]

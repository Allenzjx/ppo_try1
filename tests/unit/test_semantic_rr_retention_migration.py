"""Same439 contract/identity candidates; run model checks only after Isaac exits."""
from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import pytest

from wlr50_clean.ppo import semantic_rr_retention_migration as m
from wlr50_clean.ppo.semantic_migration import digest, file_sha

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("WLR_SAME439_SOURCE_CHECKPOINT", str(ROOT /
    "outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/"
    "checkpoints/history/checkpoint_step_000226048.pt")))


def source_metadata():
    return json.loads(SOURCE.with_name(SOURCE.stem + "_manifest.json").read_text(encoding="utf-8"))


def target_contract(old):
    result = copy.deepcopy(old)
    result["source_git_commit"] = "e" * 40
    for path in m.ALLOWED:
        result["files"][path] = file_sha(ROOT / path)
    result["runtime_content_sha256"] = digest(result["files"])
    return result


def actual_plan():
    source = source_metadata()
    target = target_contract(source["runtime_contract"])
    record = m.build_rr_retention_migration(SOURCE, target, reason="focused actual-source contract test",
        expected_source_sha256=file_sha(SOURCE),
        expected_manifest_sha256=file_sha(SOURCE.with_name(SOURCE.stem + "_manifest.json")))
    return source, target, record


def test_actual_learned439_source_dynamic_counts_and_exact_parent():
    source, target, record = actual_plan()
    factor = record[m.FACTOR_KEY]
    assert factor["revision_counter_origin"] == {k: source[k] for k in m.COUNTERS}
    assert factor["revision_counter_origin"]["global_policy_decisions"] > 225280
    assert set(record["changed_file_hashes"]) == m.ALLOWED
    assert m._previous_contract(target, record) == source["runtime_contract"]
    assert factor["source_policy_contract"] == factor["target_policy_contract"] == source["policy_contract"]
    assert factor["source_runner_config"] == factor["target_runner_config"] == source["runner_config"]
    assert factor["source_effective_learning_rate"] == source["optimizer_learning_rate"]
    assert factor["old_rollout_inherited"] is False
    migrated = {**copy.deepcopy(source), "runtime_contract": target, m.MIGRATION: record}
    route = source["checkpoint_output_routing"]
    m.validate_rr_retention_lineage(migrated, target, Path(route["output_root"]), checkpoint_output_routing=route)
    assert migrated["rear_owner_recovery_migration"] == source["rear_owner_recovery_migration"]
    later = copy.deepcopy(migrated)
    for key, delta in zip(m.COUNTERS, (128, 1, 20), strict=True):
        later[key] += delta
    later["stage_requested_decisions"]["full_episode"] += 128
    from wlr50_clean.ppo.semantic_rear_policy_timing_migration import rear_policy_timing_branch_counts
    later = rear_policy_timing_branch_counts(later)
    later["actor_parameter_sha256"] = "a" * 64
    later["optimizer_state_sha256"] = "b" * 64
    m.validate_rr_retention_lineage(later, target, Path(route["output_root"]), checkpoint_output_routing=route)


@pytest.mark.parametrize("change", ["hash", "sidecar", "physics", "config", "kernel", "missing_path"])
def test_plan_rejects_unreviewed_change(change):
    source = source_metadata()
    target = target_contract(source["runtime_contract"])
    kwargs = dict(expected_source_sha256=file_sha(SOURCE),
        expected_manifest_sha256=file_sha(SOURCE.with_name(SOURCE.stem + "_manifest.json")))
    if change == "hash": kwargs["expected_source_sha256"] = "0" * 64
    elif change == "sidecar": kwargs["expected_manifest_sha256"] = "0" * 64
    elif change == "physics": target["physics_hz"] = 119.
    elif change == "config": target["training_budgets"]["phase_suffix"] += 128
    elif change == "kernel": target["files"][m.CODE + "semantic_rear_owner_actor.py"] = "0" * 64
    else: target["files"][m.CODE + "semantic_reward.py"] = source["runtime_contract"]["files"][m.CODE + "semantic_reward.py"]
    with pytest.raises(ValueError):
        m.build_rr_retention_migration(SOURCE, target, reason="reject mutation", **kwargs)


@pytest.mark.parametrize("change", ["parent", "origin", "route", "count", "stage_usage", "target"])
def test_lineage_fails_closed(change):
    source, target, record = actual_plan()
    row = {**copy.deepcopy(source), "runtime_contract": target, m.MIGRATION: copy.deepcopy(record)}
    if change == "parent": row["rear_owner_recovery_migration"]["reason"] += " changed"
    elif change == "origin": row[m.MIGRATION][m.FACTOR_KEY]["revision_counter_origin"]["ppo_updates"] -= 1
    elif change == "route": row["checkpoint_output_routing"]["branch"] = "wrong"
    elif change == "count": row["global_policy_decisions"] -= 1
    elif change == "stage_usage": row["stage_requested_decisions"]["phase_suffix"] -= 1
    else: target["source_git_commit"] = "0" * 40
    route = source["checkpoint_output_routing"]
    with pytest.raises(ValueError):
        m.validate_rr_retention_lineage(row, target, Path(route["output_root"]), checkpoint_output_routing=route)


def test_resume_schema_cli_namespace_and_save_wiring():
    # AST/read-only routing assertion: actual API/runner checks below complement this.
    import ast
    training = ast.parse((ROOT / (m.CODE + "semantic_training.py")).read_text(encoding="utf-8"))
    loader = next(n for n in training.body if isinstance(n, ast.FunctionDef) and n.name == "load_semantic_checkpoint")
    trainer = next(n for n in training.body if isinstance(n, ast.FunctionDef) and n.name == "train_semantic")
    assert m.FACTOR_KEY in {n.value for n in ast.walk(loader) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert m.MIGRATION in {n.value for n in ast.walk(trainer) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for filename, token in (("semantic_migration.py", m.SCHEMA),
            ("semantic_cli.py", m.FACTOR_KEY), ("semantic_rear_policy_timing_migration.py", m.MIGRATION)):
        assert token in (ROOT / (m.CODE + filename)).read_text(encoding="utf-8")


@pytest.mark.parametrize("phase", ["P01", "P07"])
def test_actual_cli_resume_admission_and_preflight_remain439(tmp_path, phase):
    from wlr50_clean.ppo import semantic_cli as cli
    source, target, record = actual_plan()
    plan = tmp_path / "same439_plan.json"
    plan.write_text(json.dumps(record), encoding="utf-8")
    runs_root = cli.version_paths("v3", experiment_id=m.EXPERIMENT)[0]
    args = cli.parser().parse_args(["train", "--run-dir", str(runs_root / "unit_no_physics" / phase),
        "--expected-head", target["source_git_commit"], "--semantic-version", "v3",
        "--experiment-id", m.EXPERIMENT, "--checkpoint", str(SOURCE),
        "--checkpoint-output-branch", m.BRANCH_NAME, "--resume-migration", str(plan),
        "--stage", "full_episode" if phase == "P01" else "phase_suffix",
        "--from-phase", phase, "--prefix-source", "frozen_fsm" if phase == "P01" else "successful_nominal",
        "--seed", str(source["seed"]), "--decisions", "128"])
    cli.validate_request(args)
    cli._preflight_checkpoint(args, target)
    assert args._policy_version == m.REAR_OWNER_POLICY
    assert args._observation_layout == m.REAR_OWNER_OBSERVATION_LAYOUT
    assert args._migration_record[m.FACTOR_KEY]["revision_counter_origin"] == {k: source[k] for k in m.COUNTERS}
    assert args._checkpoint_output_routing == source["checkpoint_output_routing"]
    assert not args.new_mdp_warm_start


@pytest.fixture
def cpu_state(tmp_path, monkeypatch):
    # Explicit CPU-only fixture; never instantiate while the physical runner is active.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    assert not torch.cuda.is_available(), "run this suite with CUDA_VISIBLE_DEVICES=-1"
    from wlr50_clean.ppo import semantic_training as t
    from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
    old_threads, old_rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    def make(env=None):
        return t.construct_semantic_runner(env or _shape_env(439, "cpu"), seed=1001, device="cpu",
            policy_version=m.REAR_OWNER_POLICY, observation_layout=m.REAR_OWNER_OBSERVATION_LAYOUT,
            initialize_actor=False)[0]
    runner = make()
    # Deliberately learned/nonzero new17 columns and populated full Adam; append-zero is invalid here.
    for role in ("actor", "critic"):
        with torch.no_grad():
            getattr(runner.alg, role).mlp[0].weight[:, 422:].fill_(.037)
    for parameter in list(runner.alg.actor.parameters()) + list(runner.alg.critic.parameters()):
        parameter.grad = torch.ones_like(parameter)
    runner.alg.optimizer.step(); runner.alg.optimizer.zero_grad()
    for group in runner.alg.optimizer.param_groups: group["lr"] = 2.25e-5
    runner.alg.learning_rate = 2.25e-5
    base = source_metadata()
    for key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"):
        base.pop(key, None)
    runner.current_learning_iteration = base["ppo_updates"]
    for state in runner.alg.optimizer.state.values(): state["step"].fill_(float(base["optimizer_steps"]))
    branch = tmp_path / ("ppo_" + m.EXPERIMENT) / "branches" / m.BRANCH_NAME
    base["checkpoint_output_routing"]["output_root"] = str(branch)
    path, sidecar = t.save_semantic_checkpoint(runner, branch / "checkpoints/history/source.pt", base)
    source = json.loads(sidecar.read_text(encoding="utf-8"))
    target = target_contract(source["runtime_contract"])
    factor = dict(schema=m.SCHEMA, source_policy_contract=source["policy_contract"], target_policy_contract=source["policy_contract"],
        source_runner_config=source["runner_config"], target_runner_config=source["runner_config"],
        source_effective_learning_rate=2.25e-5, parameter_mapping=m.MAPPING,
        preserved_metadata_sha256={k: digest(source[k]) for k in m._preserved(source)})
    record = dict(schema=m.SCHEMA, plan_path=str(tmp_path / "plan.json"), **{m.FACTOR_KEY: factor})
    # These mocks isolate model identity from the real-files/lineage tests above.
    monkeypatch.setattr(m, "validate_rr_retention_migration", lambda *a, **k: record)
    monkeypatch.setattr(m, "validate_rr_retention_lineage", lambda *a, **k: record)
    yield torch, t, runner, make, path, source, target, record, branch
    torch.set_rng_state(old_rng); torch.set_num_threads(old_threads)


def test_full_439_state_publication_and_independent_reload(cpu_state):
    torch, t, source_runner, make, path, source, target, record, branch = cpu_state
    output = branch / "checkpoints/history/same439_published.pt"
    receipt = m.publish_rr_retention_checkpoint(path, target, record["plan_path"], output)
    assert receipt["added_policy_decisions"] == receipt["added_ppo_updates"] == receipt["added_optimizer_steps"] == 0
    before, after = [torch.load(p, map_location="cpu", weights_only=False) for p in (path, output)]
    assert t.state_hash({k: v for k, v in before.items() if k != "infos"}) == t.state_hash({k: v for k, v in after.items() if k != "infos"})
    fresh = make()
    loaded = t.load_semantic_checkpoint(fresh, output, contract=target, seed=1001)
    m._verify_identity(fresh, loaded, record[m.FACTOR_KEY])
    for key in m._preserved(source): assert loaded[key] == source[key]
    assert loaded["sampling"] == "P01_full_task_only_initial_version"
    assert fresh.alg.storage.step == 0 and fresh.alg.transition.actions is None
    assert tuple(fresh.alg.storage.observations["policy"].shape) == (128, 1, 439)
    from tensordict import TensorDict
    x = torch.zeros(1, 439); x[:, 11] = 1.; x[:, 20] = .03; x[:, 412] = 1.
    obs = TensorDict({"policy": x, "critic": x.clone()}, batch_size=[1])
    with torch.no_grad():
        for role in ("actor", "critic"):
            assert torch.equal(getattr(source_runner.alg, role)(obs), getattr(fresh.alg, role)(obs))
        # Official RSL-RL deterministic forward does not instantiate the
        # sampling distribution. Compare the actual stochastic kernel with
        # the same RNG before asking it for a raw-action density.
        rng = torch.get_rng_state().clone()
        sample_before = source_runner.alg.actor(obs, stochastic_output=True)
        torch.set_rng_state(rng)
        sample_after = fresh.alg.actor(obs, stochastic_output=True)
        assert torch.equal(sample_before, sample_after)
        raw = torch.full((1, 12), .07)
        assert torch.equal(source_runner.alg.actor.get_output_log_prob(raw), fresh.alg.actor.get_output_log_prob(raw))
    assert not (branch / "checkpoints/checkpoint_last_pointer.json").exists()
    with pytest.raises(ValueError): m.publish_rr_retention_checkpoint(path, target, record["plan_path"], output)


@pytest.mark.parametrize("dirty", ["storage", "transition", "seed", "mixed_factor"])
def test_load_rejects_partial_rollout_and_mixed_boundary(cpu_state, dirty):
    torch, t, _, make, path, _, target, record, _ = cpu_state
    runner = make(); seed = 1001; attempt = copy.deepcopy(record)
    if dirty == "storage": runner.alg.storage.step = 1
    elif dirty == "transition": runner.alg.transition.actions = torch.zeros(1, 12)
    elif dirty == "seed": seed = 4001
    else: attempt["rear_owner_append439_factor"] = {"not": "same439"}
    with pytest.raises((ValueError, RuntimeError)):
        t.load_semantic_checkpoint(runner, path, contract=target, seed=seed, migration=attempt)


def test_one_fresh_synthetic_update_carries_both_receipts(cpu_state, tmp_path):
    torch, t, _, make, path, source, target, record, branch = cpu_state
    from test_semantic_rear_policy_training_audit import Synthetic419Core
    class Synthetic439Core(Synthetic419Core):
        def observation(self, raw=(0.,) * 12):
            return super().observation(raw) + (0.,) * 20
    env = t.SemanticRslAdapter(Synthetic439Core(), seed=1001, device="cpu")
    env.cfg["semantic_version"] = "v3"
    learner = make(env)
    infos = t.load_semantic_checkpoint(learner, path, contract=target, seed=1001, migration=record)
    run = tmp_path / "synthetic_only"
    result = t.train_semantic(learner, env, run_dir=run, output_root=branch,
        stage="full_episode", decisions=128, contract=target, seed=1001, resume_infos=infos,
        checkpoint_interval_updates=1, checkpoint_output_routing=source["checkpoint_output_routing"])
    assert (result["actual_policy_decisions"], result["ppo_updates_this_run"], result["optimizer_steps_this_run"]) == (128, 1, 20)
    saved = Path(result["checkpoints"][-1]["checkpoint"])
    manifest = json.loads(saved.with_name(saved.stem + "_manifest.json").read_text(encoding="utf-8"))
    assert manifest[m.MIGRATION] == record
    assert manifest["rear_owner_recovery_migration"] == source["rear_owner_recovery_migration"]
    for key, delta in zip(m.COUNTERS, (128, 1, 20), strict=True): assert manifest[key] == source[key] + delta
    fresh = make()
    t.load_semantic_checkpoint(fresh, saved, contract=target, seed=1001)
    assert t.state_hash(fresh.alg.optimizer.state_dict()) == manifest["optimizer_state_sha256"]
    rollout_path = next((run / "rollouts").glob("*.pt"))
    rollout = torch.load(rollout_path, map_location="cpu", weights_only=False)
    assert tuple(rollout["observations"]["policy"].shape) == (128, 1, 439)
    assert tuple(rollout["actions"].shape) == (128, 1, 12)

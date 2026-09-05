from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from wlr50_clean.ppo import semantic_cli as cli
from wlr50_clean.ppo.semantic_migration import build_migration_plan, validate_migration_plan, digest, file_sha, STAGE_SPEC
from wlr50_clean.ppo.semantic_training import (build_stop_request, load_semantic_checkpoint,
    save_semantic_checkpoint, semantic_runner_config, train_semantic, write_json, parameter_hash, state_hash)
from test_semantic_training import Core, make_runner


CODE = "src/wlr50_clean/ppo/semantic_training.py"


def contract(version, changed=CODE):
    files = {CODE: "1" * 64, "src/wlr50_clean/ppo/semantic_reward.py": "a" * 64,
             "configs/ppo_semantic_v2/observation_schema.json": "b" * 64}
    if version == 2:
        files[changed] = "2" * 64
    return {"source_git_commit": str(version) * 40, "files": files,
            "runtime_content_sha256": digest(files), "physics_hz": 120,
            "frozen_A_files": {"A": "f" * 64}, "version": "locked"}


def checkpoint(tmp_path, old=None):
    path = tmp_path / "source.pt"
    path.write_bytes(b"immutable fixture checkpoint")
    write_json(path.with_name("source_manifest.json"), {
        "schema": "wlr50_clean.semantic_checkpoint.v1", "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": file_sha(path), "save_load_round_trip": True,
        "runtime_contract": old or contract(1), "seed": 1001})
    return path


def test_migration_is_exact_file_and_checkpoint_bound(tmp_path):
    source = checkpoint(tmp_path)
    plan = build_migration_plan(source, contract(2), allowed_changed_files=[CODE], reason="reviewed instrumentation")
    path = tmp_path / "migration.json"
    write_json(path, plan)
    verified = validate_migration_plan(source, contract(2), path)
    assert verified["source_checkpoint_sha256"] == file_sha(source)
    plan["target_git_commit"] = "f" * 40
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="not exactly bound"):
        validate_migration_plan(source, contract(2), path)


@pytest.mark.parametrize("changed", ["src/wlr50_clean/ppo/semantic_reward.py", "src/wlr50_clean/ppo/semantic_backend.py",
                                     "configs/ppo_semantic_v2/observation_schema.json", "configs/ppo_semantic_v2/action_schema.json"])
def test_even_explicitly_declared_protected_changes_are_rejected(tmp_path, changed):
    with pytest.raises(ValueError, match="protected"):
        build_migration_plan(checkpoint(tmp_path), contract(2, changed), allowed_changed_files=[changed], reason="not authorized")


def test_undeclared_delta_and_frozen_runtime_change_rejected(tmp_path):
    source = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="undeclared"):
        build_migration_plan(source, contract(2), allowed_changed_files=[], reason="missing")
    target = contract(2)
    target["physics_hz"] = 60
    with pytest.raises(ValueError, match="frozen physics"):
        build_migration_plan(source, target, allowed_changed_files=[CODE], reason="invalid")


def test_exact_geometric_single_factor_and_no_second_change(tmp_path, monkeypatch):
    import wlr50_clean.ppo.semantic_migration as module
    from types import SimpleNamespace
    before = b"geometry:\n  approach_min_m: -0.18\n  approach_max_m: 0.10\n"
    target_file = tmp_path / STAGE_SPEC
    target_file.parent.mkdir(parents=True)
    target_file.write_bytes(before.replace(b"-0.18", b"-0.005"))
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=before))
    old, new = contract(1), contract(1)
    old["files"][STAGE_SPEC] = module.hashlib.sha256(before).hexdigest()
    new["files"][STAGE_SPEC] = file_sha(target_file)
    new["source_git_commit"] = "2" * 40
    for data in (old, new):
        data["runtime_content_sha256"] = digest(data["files"])
    source = checkpoint(tmp_path, old)
    plan = build_migration_plan(source, new, allowed_changed_files=[STAGE_SPEC], reason="one geometry factor", project_root=tmp_path)
    assert plan["geometric_factor"]["after"] == -0.005
    target_file.write_text(target_file.read_text().replace("0.10", "0.20"))
    new["files"][STAGE_SPEC] = file_sha(target_file)
    new["runtime_content_sha256"] = digest(new["files"])
    with pytest.raises(ValueError, match="only the declared"):
        build_migration_plan(source, new, allowed_changed_files=[STAGE_SPEC], reason="two changes", project_root=tmp_path)


def test_no_migration_flag_rejects_changed_contract_before_live(tmp_path):
    args = cli.parser().parse_args(["train", "--run-dir", str(tmp_path), "--expected-head", "2" * 40])
    args.checkpoint = checkpoint(tmp_path)
    with pytest.raises(ValueError, match="explicit reviewed"):
        cli._preflight_checkpoint(args, contract(2))


def test_real_optimizer_migration_preserves_all_learning_state_and_empty_storage(tmp_path):
    import torch
    class Core324(Core):
        def reset(self, *, seed=1001, options=None):
            return super().reset(seed=seed, options=options) + (0.0,) * 316
        def step(self, raw):
            result = super().step(raw)
            result.observation += (0.0,) * 316
            return result
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        old, new = contract(1), contract(2)
        runner, env = make_runner(Core324())
        train_semantic(runner, env, run_dir=tmp_path / "old", output_root=tmp_path / "outputs",
                       stage="smoke", decisions=128, contract=old, seed=1001)
        source = tmp_path / "outputs/checkpoints/history/checkpoint_step_000000128.pt"
        plan_file = tmp_path / "plan.json"
        write_json(plan_file, build_migration_plan(source, new, allowed_changed_files=[CODE], reason="reviewed"))
        record = validate_migration_plan(source, new, plan_file)
        actor, critic = parameter_hash(runner.alg.actor), parameter_hash(runner.alg.critic)
        optimizer = state_hash(runner.alg.optimizer.state_dict())
        expected_rng = torch.rand(8)
        fresh, fresh_env = make_runner(Core324())
        infos = load_semantic_checkpoint(fresh, source, contract=new, seed=1001, migration=record)
        assert (parameter_hash(fresh.alg.actor), parameter_hash(fresh.alg.critic)) == (actor, critic)
        assert state_hash(fresh.alg.optimizer.state_dict()) == optimizer
        assert torch.equal(torch.rand(8), expected_rng)
        assert fresh.alg.storage.step == 0 and fresh_env.total_decisions == 0
        assert infos["global_policy_decisions"] == 128 and infos["optimizer_steps"] == 20
        assert infos["stage_requested_decisions"]["smoke"] == 128
        assert infos["resume_migration"] == record
    finally:
        torch.set_num_threads(prior_threads)


def test_stop_during_second_rollout_waits_for_exact_256_boundary(tmp_path):
    import torch
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    run_dir = tmp_path / "run"
    current = contract(1)
    class RequestingCore(Core):
        def step(self, raw):
            result = super().step(raw)
            if self.calls == 130:
                write_json(run_dir / "stop_after_update.request.json", build_stop_request(run_dir, current["source_git_commit"], "review geometry"))
            return result
    try:
        runner, env = make_runner(RequestingCore())
        result = train_semantic(runner, env, run_dir=run_dir, output_root=tmp_path / "outputs",
                                stage="smoke", decisions=1000, contract=current, seed=1001)
        assert env.core.calls == 256
        assert result["lifecycle"] == "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY"
        assert result["requested_policy_decisions"] == 256 and result["unconsumed_requested_policy_decisions"] == 744
        assert result["optimizer_steps_this_run"] == 40
        final = tmp_path / "outputs/checkpoints/history/checkpoint_step_000000256_manifest.json"
        metadata = json.loads(final.read_text())
        assert metadata["save_load_round_trip"] is True
        assert metadata["stage_requested_decisions"]["smoke"] == 256
        assert metadata["stop_after_update"]["reason"] == "review geometry"
    finally:
        torch.set_num_threads(prior_threads)

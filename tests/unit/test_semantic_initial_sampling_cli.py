"""CLI publication metadata only; real official save/load, no native scene.

The already-reviewed migration loader and physical preparation are stubbed here.
The real CLI dictionary construction, topology and immutable checkpoint writer
are exercised, including embedded infos and the published JSON sidecar. These
tests do not claim to validate migration authorization or physical roll-ins.
"""
from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from test_semantic_policy_training import make
from wlr50_clean.ppo import semantic_backend, semantic_cli as cli, semantic_env
from wlr50_clean.ppo import semantic_migration as migration, semantic_prefix, semantic_training
from wlr50_clean.ppo.semantic_policy_distribution import STATE_DEPENDENT_POLICY, policy_contract


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def sampling_for(phase):
    if phase == "P01":
        return "P01_full_task_only_initial_version", None
    request = semantic_prefix.PrefixRequest(target_phase=phase)
    return semantic_prefix.sampling_label(request), request.as_dict()


@pytest.mark.parametrize("initial_kind", ["new_mdp", "policy_distribution"])
@pytest.mark.parametrize("source_phase,current_phase,old_field_present", [
    ("P06", "P01", True), ("P01", "P06", True), ("P06", "P01", False),
], ids=["P06-to-P01", "P01-to-P06", "legacy-field-absent"])
def test_initial_current_sampling_embedded_and_sidecar_without_rewriting_source(
        tmp_path, monkeypatch, initial_kind, source_phase, current_phase, old_field_present):
    semantic_training.seed_training_rngs(1001)
    runner, env, runner_config = make(STATE_DEPENDENT_POLICY)
    # Use nonzero mean AND log-std output weights so preservation is not merely
    # the equality of two freshly zero-initialized mean tensors.
    with torch.no_grad():
        runner.alg.actor.mlp[4].weight.add_(0.0125)
    source_sampling, source_prefix = sampling_for(source_phase)
    current_sampling, current_prefix = sampling_for(current_phase)
    source_infos = {
        "seed": 1001, "semantic_version": "v3", "runtime_contract": {"revision": "source"},
        "stage": "phase_suffix" if source_phase == "P06" else "full_episode",
        "sampling": source_sampling,
        "execution_topology": migration.continuation_topology(source_sampling, source_prefix),
        "curriculum_epoch": {"reset_sampling": source_sampling, "prefix_request": source_prefix},
        "global_policy_decisions": 68224, "ppo_updates": 498, "optimizer_steps": 9960,
        "stage_requested_decisions": {"smoke": 0, "full_episode": 25088, "phase_suffix": 33024},
        "new_mdp_origin_global_policy_decisions": 10112,
    }
    if old_field_present:
        source_infos["implemented_reset_sampling"] = source_sampling
    source = tmp_path / "source.pt"
    _, source_sidecar = semantic_training.save_semantic_checkpoint(runner, source, source_infos)
    source_bytes, sidecar_bytes = source.read_bytes(), source_sidecar.read_bytes()
    previous = torch.load(source, weights_only=False)["infos"]
    previous["resume_source_checkpoint"] = str(source)
    previous_before = copy.deepcopy(previous)
    state_before = {
        "actor": semantic_training.parameter_hash(runner.alg.actor),
        "critic": semantic_training.parameter_hash(runner.alg.critic),
        "optimizer": semantic_training.state_hash(runner.alg.optimizer.state_dict()),
        "normalizer": semantic_training.state_hash(semantic_training._normalizers(runner)),
    }
    env.cfg.update(reset_sampling=current_sampling, prefix_request=current_prefix)
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    args = cli.parser().parse_args([
        "train", "--run-dir", str(tmp_path / "run"), "--expected-head", "a" * 40,
        "--semantic-version", "v3", "--from-phase", current_phase,
        "--stage", "full_episode" if current_phase == "P01" else "phase_suffix",
        "--decisions", "128", "--checkpoint", str(source), "--device", "cpu",
    ])
    args.run_dir.mkdir()
    args._policy_version = STATE_DEPENDENT_POLICY
    args._migration_record = None
    args.new_mdp_warm_start = initial_kind == "new_mdp"
    args.policy_distribution_migration = initial_kind == "policy_distribution"
    args._warm_start_record = {"reviewed": "new-MDP fixture"} if args.new_mdp_warm_start else None
    args._policy_migration_record = {"reviewed": "policy fixture"} if args.policy_distribution_migration else None
    app_calls = []
    monkeypatch.setitem(sys.modules, "isaaclab.app", NS(AppLauncher=lambda **kw:
        NS(app=NS(update=lambda: app_calls.append("stub app update")))))
    monkeypatch.setattr(semantic_backend, "SemanticIsaacBackend", lambda *a, **kw: NS())
    monkeypatch.setattr(semantic_prefix, "PrefixSemanticIsaacBackend", lambda *a, **kw: NS())
    monkeypatch.setattr(semantic_env, "SemanticEpisodeEnv", lambda *a, **kw: NS())
    monkeypatch.setattr(cli, "SemanticRslAdapter", lambda *a, **kw: env)
    monkeypatch.setattr(semantic_prefix, "PrefixRslAdapter", lambda *a, **kw: env)
    constructs, loads, trains = [], [], []
    def construct(actual_env, **kwargs):
        constructs.append(kwargs)
        assert actual_env is env
        return runner, runner_config
    def load(actual_runner, checkpoint, **kwargs):
        loads.append(kwargs)
        assert actual_runner is runner and checkpoint == source
        return previous
    monkeypatch.setattr(cli, "construct_semantic_runner", construct)
    monkeypatch.setattr(cli, "load_semantic_checkpoint", load)
    monkeypatch.setattr(migration, "v3_warm_start_checkpoint_name", lambda record: "initial_new_mdp.pt")
    monkeypatch.setattr(cli, "policy_migration_checkpoint_name", lambda record: "initial_policy.pt")
    monkeypatch.setattr(migration, "warm_start_source_execution_profile", lambda *a, **kw: tmp_path / "old.yaml")
    monkeypatch.setattr(semantic_training, "compare_warm_start_action", lambda *a, **kw:
                        {"scope": "stubbed physical comparison; no simulated evidence"})
    monkeypatch.setattr(cli, "train_semantic", lambda *a, **kw: trains.append(kw) or {"not_optimized": True})
    # Keep the real immutable serializer/round-trip implementation, not a
    # capture-only mock of the infos argument.
    assert cli.save_semantic_checkpoint is semantic_training.save_semantic_checkpoint
    assert cli.dispatch_live(args, {"revision": "current"}) == {"not_optimized": True}
    name = "initial_new_mdp.pt" if args.new_mdp_warm_start else "initial_policy.pt"
    initial = cli.version_paths("v3")[1] / "checkpoints/history" / name
    embedded = torch.load(initial, weights_only=False)["infos"]
    sidecar = json.loads(initial.with_name(initial.stem + "_manifest.json").read_text())
    assert sidecar["save_load_round_trip"] is True
    for record in (embedded, sidecar):
        assert record["implemented_reset_sampling"] == record["sampling"] == current_sampling
        assert record["execution_topology"] == migration.continuation_topology(current_sampling, current_prefix)
        assert record["curriculum_epoch"] == {"reset_sampling": current_sampling, "prefix_request": current_prefix}
        for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps", "stage_requested_decisions",
                    "new_mdp_origin_global_policy_decisions", "resume_source_checkpoint"):
            assert record[key] == previous_before[key]
        assert record["policy_contract"] == policy_contract(STATE_DEPENDENT_POLICY)
        assert record["runner_config"] == runner_config
    assert source.read_bytes() == source_bytes and source_sidecar.read_bytes() == sidecar_bytes
    assert previous.get("implemented_reset_sampling") == previous_before.get("implemented_reset_sampling")
    assert previous["sampling"] == source_sampling
    allowed_additions = {"new_mdp_initial_action_comparison"} if args.new_mdp_warm_start else set()
    assert {k: v for k, v in previous.items() if k not in allowed_additions} == previous_before
    assert constructs == [{"seed": 1001, "device": "cpu", "policy_version": STATE_DEPENDENT_POLICY,
                           "initialize_actor": False}]
    assert loads == [{"contract": {"revision": "current"}, "seed": 1001, "migration": None,
                      "warm_start": args._warm_start_record, "policy_migration": args._policy_migration_record}]
    assert len(trains) == 1 and trains[0]["resume_infos"] is previous
    assert trains[0]["decisions"] == 128 and trains[0]["stage"] == args.stage
    assert app_calls == ["stub app update"] and runner.alg.storage.step == 0
    assert env.core.calls == 0
    assert state_before == {
        "actor": semantic_training.parameter_hash(runner.alg.actor),
        "critic": semantic_training.parameter_hash(runner.alg.critic),
        "optimizer": semantic_training.state_hash(runner.alg.optimizer.state_dict()),
        "normalizer": semantic_training.state_hash(semantic_training._normalizers(runner)),
    }

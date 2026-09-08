"""Reset-source plumbing only; no native scene or fake physical success."""
from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace as NS

import pytest

from wlr50_clean.ppo import semantic_cli as cli
from wlr50_clean.ppo.semantic_training import semantic_curriculum_epoch


def args_for(tmp_path, *extra):
    return cli.parser().parse_args([
        "train", "--run-dir", str(cli.PROJECT_ROOT / "runs/ppo_semantic_v3/train/prefix_cli_test"),
        "--expected-head", "a" * 40, "--semantic-version", "v3", "--stage", "phase_suffix",
        "--from-phase", "P06", "--prefix-source", "checkpoint_policy",
        "--checkpoint", str(tmp_path / "source.pt"), "--decisions", "128", *extra,
    ])


@pytest.mark.parametrize("change", [
    {"command": "eval"}, {"from_phase": "P01"}, {"stage": "full_episode"},
    {"semantic_version": "v2"}, {"num_envs": 8}, {"checkpoint": None},
    {"policy_distribution_migration": True},
])
def test_current_checkpoint_prefix_rejects_invalid_scope_before_native(tmp_path, change):
    args = args_for(tmp_path)
    for name, value in change.items():
        setattr(args, name, value)
    with pytest.raises(ValueError):
        cli.validate_request(args)


def test_original_prefix_is_still_parser_default(tmp_path):
    args = cli.parser().parse_args(["eval", "--run-dir", str(tmp_path), "--expected-head", "a" * 40])
    assert args.prefix_source == "frozen_fsm"


def test_request_validates_source_without_restarting_budget(tmp_path, monkeypatch):
    args = args_for(tmp_path)
    sidecar = args.checkpoint.with_name("source_manifest.json")
    sidecar.write_text(json.dumps({"semantic_version": "v3", "stage_requested_decisions": {
        "full_episode": 33280, "phase_suffix": 41600, "smoke": 0}}))
    monkeypatch.setattr(cli, "_resolved_checkpoint", lambda path, **kwargs: path)
    cli.validate_request(args)
    assert args.decisions == 128 and args.prefix_source == "checkpoint_policy"


def test_epoch_requires_bound_policy_and_copies_provenance():
    cfg = {"reset_sampling": "test-fixed-prefix", "prefix_request": {"source": "frozen_checkpoint_policy"},
           "prefix_policy_provenance": None}
    with pytest.raises(ValueError, match="installed"):
        semantic_curriculum_epoch(cfg)
    cfg["prefix_policy_provenance"] = {"checkpoint_sha256": "b" * 64, "nested": [1, 2]}
    epoch = semantic_curriculum_epoch(cfg)
    epoch["prefix_policy_provenance"]["nested"].append(3)
    assert cfg["prefix_policy_provenance"]["nested"] == [1, 2]
    cfg["prefix_policy_provenance"]["checkpoint_sha256"] = "c" * 64
    assert semantic_curriculum_epoch(cfg) != epoch
    assert semantic_curriculum_epoch({"reset_sampling": "original"}) == {
        "reset_sampling": "original", "prefix_request": None}


def test_checkpoint_prefix_topology_round_trip_keeps_original_defaults():
    from wlr50_clean.ppo.semantic_checkpoint_prefix import CheckpointPolicyPrefixRequest, sampling_label
    from wlr50_clean.ppo.semantic_migration import continuation_topology
    from wlr50_clean.ppo.semantic_prefix import PrefixRequest, sampling_label as teacher_label
    request = CheckpointPolicyPrefixRequest(target_phase="P06", teacher_offset_decisions=2)
    topology = continuation_topology(sampling_label(request), request.as_dict())
    assert topology["num_envs"] == 1 and topology["observation_dimension"] == 324
    assert topology["phase_suffix_curriculum_implemented"] is True
    altered = {**request.as_dict(), "extra_unbound_flag": True}
    with pytest.raises(ValueError):
        continuation_topology(sampling_label(request), altered)
    teacher = PrefixRequest(target_phase="P10")
    assert continuation_topology(teacher_label(teacher), teacher.as_dict())["reset_sampling"] == teacher_label(teacher)
    assert continuation_topology("P01_full_task_only_initial_version", None)["phase_suffix_curriculum_implemented"] is False


def test_dispatch_loads_actor_then_installs_prefix_without_teacher_backend(tmp_path, monkeypatch):
    from wlr50_clean.ppo import semantic_backend, semantic_env, semantic_prefix
    from wlr50_clean.ppo import semantic_checkpoint_prefix as prefix
    from wlr50_clean.ppo import semantic_checkpoint_prefix_policy as policy
    from wlr50_clean.ppo.semantic_policy_distribution import STATE_DEPENDENT_POLICY, policy_contract

    args = args_for(tmp_path, "--device", "cpu")
    args.run_dir = tmp_path / "run"
    args.run_dir.mkdir()
    args._policy_version = STATE_DEPENDENT_POLICY
    args._observation_layout = None  # This direct dispatch fixture represents verified old 324 preflight.
    args._migration_record = args._warm_start_record = args._policy_migration_record = None
    events = []
    monkeypatch.setitem(sys.modules, "isaaclab.app", NS(AppLauncher=lambda **kw: NS(app=NS(update=lambda: None))))
    monkeypatch.setattr(cli, "seed_training_rngs", lambda seed: None)
    monkeypatch.setattr(semantic_backend, "SemanticIsaacBackend", lambda *a, **kw: events.append("normal_backend") or NS())
    monkeypatch.setattr(semantic_prefix, "PrefixSemanticIsaacBackend", lambda *a, **kw: pytest.fail("teacher backend used"))
    monkeypatch.setattr(semantic_env, "SemanticEpisodeEnv", lambda *a, **kw: NS())
    source_actor = object()
    runner = NS(alg=NS(actor=source_actor))
    env = NS(cfg={"reset_sampling": "fixed-C-prefix", "prefix_request": {"source": "frozen_checkpoint_policy"}})
    def install(callback, provenance):
        assert events[-1] == "clone_loaded_actor"
        assert callback is fixed
        env.cfg["prefix_policy_provenance"] = copy.deepcopy(provenance)
        events.append("real_rollin_stub")
    env.install_prefix_policy = install
    monkeypatch.setattr(prefix, "CheckpointPolicyPrefixRslAdapter", lambda *a, **kw: events.append("bootstrap") or env)
    monkeypatch.setattr(cli, "construct_semantic_runner", lambda *a, **kw: (runner, {}))
    previous = {
        "resume_source_checkpoint": {"checkpoint": str(args.checkpoint), "checkpoint_sha256": "b" * 64},
        "actor_parameter_sha256": "c" * 64, "global_policy_decisions": 84992, "ppo_updates": 629,
        "runtime_contract": {"runtime_content_sha256": "d" * 64},
        "stage_requested_decisions": {"full_episode": 33280, "phase_suffix": 41600, "smoke": 0},
    }
    def load(*a, **kw):
        events.append("load_source")
        return previous
    monkeypatch.setattr(cli, "load_semantic_checkpoint", load)
    fixed = NS(provenance=None)
    def clone(actor, source):
        assert actor is source_actor and events[-1] == "load_source"
        assert source["source_global_policy_decisions"] == 84992
        assert source["source_ppo_updates"] == 629
        assert source["policy_contract"] == policy_contract(STATE_DEPENDENT_POLICY)
        fixed.provenance = source
        events.append("clone_loaded_actor")
        return fixed
    monkeypatch.setattr(policy, "build_frozen_checkpoint_prefix_policy", clone)
    def train(actual_runner, actual_env, **kwargs):
        assert actual_runner is runner and actual_env is env and runner.alg.actor is source_actor
        assert events[-1] == "real_rollin_stub"
        assert kwargs["decisions"] == 128 and kwargs["resume_infos"] is previous
        assert semantic_curriculum_epoch(env.cfg)["prefix_policy_provenance"] == fixed.provenance
        events.append("PPO_collection_stub")
        return {"scope": "plumbing only; no actual physics or optimization"}
    monkeypatch.setattr(cli, "train_semantic", train)
    cli.dispatch_live(args, {"current": True})
    assert events == ["normal_backend", "bootstrap", "load_source", "clone_loaded_actor", "real_rollin_stub", "PPO_collection_stub"]

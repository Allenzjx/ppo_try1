"""Policy-kernel selection is explicit and rejected before any native launch."""
from types import SimpleNamespace
from pathlib import Path

import pytest

from wlr50_clean.ppo import semantic_cli as cli
from wlr50_clean.ppo.semantic_policy_distribution import HISTORY_POLICY, STATE_DEPENDENT_POLICY
from wlr50_clean.ppo.semantic_training import semantic_runner_config


def request(**overrides):
    return SimpleNamespace(**{
        "target_policy_version": HISTORY_POLICY, "semantic_version": "v3", "num_envs": 1,
        "command": "train", "checkpoint": Path("not-opened.pt"), "new_mdp_warm_start": True,
        "resume_migration": None, "policy_distribution_migration": False,
        "prefix_source": "frozen_fsm", **overrides,
    })


def test_selector_parser_and_allowed_boundary():
    parsed = cli.parser().parse_args([
        "train", "--run-dir", "unused", "--expected-head", "a"*40,
        "--semantic-version", "v3", "--new-mdp-warm-start", "--checkpoint", "unused.pt",
        "--target-policy-version", HISTORY_POLICY,
    ])
    cli._validate_target_policy_request(parsed)
    assert parsed.target_policy_version == HISTORY_POLICY
    cli._validate_target_policy_request(request())


@pytest.mark.parametrize("change", [
    {"command": "eval"}, {"command": "smoke"}, {"command": "preflight"},
    {"new_mdp_warm_start": False}, {"checkpoint": None}, {"semantic_version": "v2"},
    {"num_envs": 8}, {"prefix_source": "checkpoint_policy"},
    {"policy_distribution_migration": True}, {"resume_migration": Path("old.json")},
    {"target_policy_version": STATE_DEPENDENT_POLICY},
])
def test_invalid_target_rejected_by_validate_and_preflight_before_source_read(change):
    args = request(**change)
    with pytest.raises(ValueError, match="target policy kernel"):
        cli.validate_request(args)
    with pytest.raises(ValueError, match="target policy kernel"):
        cli._preflight_checkpoint(args, {})


def test_absent_selector_does_not_silently_change_old_requests():
    for command in ("train", "eval", "smoke"):
        cli._validate_target_policy_request(request(target_policy_version=None, command=command))
    args = request()
    del args.target_policy_version
    cli._validate_target_policy_request(args)


def test_history_factory_rejects_v2_before_constructing_a_model():
    with pytest.raises(ValueError, match="history-conditioned policy"):
        semantic_runner_config(seed=1001, device="cpu", semantic_version="v2", policy_version=HISTORY_POLICY)


def test_preflight_sets_only_the_verified_explicit_target(monkeypatch, tmp_path):
    from wlr50_clean.ppo import semantic_migration as migration
    source = {"runtime_contract": {"old": True}, "seed": 1001}
    record = {"policy_kernel_transition": {"target_policy_version": HISTORY_POLICY}}
    seen = []
    monkeypatch.setattr(migration, "checkpoint_metadata", lambda checkpoint: source)
    monkeypatch.setattr(cli, "policy_version_from_metadata", lambda metadata: STATE_DEPENDENT_POLICY)
    def build(checkpoint, contract, **kwargs):
        seen.append(kwargs)
        return record
    monkeypatch.setattr(migration, "build_v3_warm_start_record", build)
    monkeypatch.setattr(migration, "v3_warm_start_checkpoint_name", lambda bound: "not-created.pt")
    monkeypatch.setattr(cli, "version_paths", lambda version: (tmp_path, tmp_path, tmp_path))
    args = request(seed=1001)
    cli._preflight_checkpoint(args, {"target": True})
    assert args._policy_version == HISTORY_POLICY and args._warm_start_record == record
    assert seen == [{"project_root": cli.PROJECT_ROOT, "target_policy_version": HISTORY_POLICY}]


def test_video_rejects_explicit_target_before_loading_checkpoint(tmp_path):
    from wlr50_clean.ppo.semantic_video_cli import validate_video_args
    args = cli.parser().parse_args([
        "eval", "--run-dir", str(tmp_path), "--expected-head", "a"*40,
        "--semantic-version", "v3", "--seed", "4001", "--no-headless",
        "--target-policy-version", HISTORY_POLICY,
    ])
    with pytest.raises(ValueError, match="target policy kernel"):
        validate_video_args(args)

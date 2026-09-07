"""Official CPU PPO migration checks; the small core is not physical evidence."""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from test_semantic_v3_continuation import Core324, contracts
from wlr50_clean.ppo import semantic_migration as migration
from wlr50_clean.ppo import semantic_policy_distribution as policies
from wlr50_clean.ppo import semantic_training as training


MODULE = "src/wlr50_clean/ppo/semantic_history_actor.py"
CHANGED = (
    MODULE, "src/wlr50_clean/ppo/semantic_policy_distribution.py",
    "src/wlr50_clean/ppo/semantic_training.py", "src/wlr50_clean/ppo/semantic_migration.py",
    "src/wlr50_clean/ppo/semantic_cli.py", "scripts/run_semantic_ppo.ps1",
    "src/wlr50_clean/ppo/semantic_checkpoint_prefix.py",
    "src/wlr50_clean/ppo/semantic_checkpoint_prefix_policy.py",
)
PHYSICAL = "src/wlr50_clean/ppo/action_projection.py"


class HistoryCore(Core324):
    """324 ABI with the actual previous issued raw action at the history slice."""

    def step(self, raw):
        result = super().step(raw)
        values = list(result.observation)
        values[195:207] = [max(-20.0, min(20.0, value)) for value in raw]
        result.observation = tuple(values)
        return result


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make(policy, *, initialize=False):
    env = training.SemanticRslAdapter(HistoryCore(), seed=1001, device="cpu")
    env.cfg["semantic_version"] = "v3"
    runner, _ = training.construct_semantic_runner(
        env, seed=1001, device="cpu", policy_version=policy, initialize_actor=initialize)
    assert all(parameter.device.type == "cpu" for parameter in runner.alg.actor.parameters())
    assert (runner.alg.gamma, runner.alg.lam) == (.9985, .99)
    return runner, env


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def inventory(root, contract):
    result = copy.deepcopy(contract)
    result["files"] = {name: migration.file_sha(root / name) for name in result["files"]}
    result["runtime_content_sha256"] = migration.digest(result["files"])
    result["source_git_commit"] = git(root, "rev-parse", "HEAD")
    return result


@pytest.fixture(scope="module")
def learned_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("history_migration")
    _, template = contracts(root)
    git(root, "init")
    git(root, "config", "core.autocrlf", "false")
    git(root, "config", "user.name", "CPU integration fixture")
    git(root, "config", "user.email", "cpu-fixture@example.invalid")
    current_bytes = {name: (migration.PROJECT_ROOT / name).read_bytes()
                     for name in (*CHANGED, PHYSICAL)}
    # Genuine Git source blobs differ from the target worktree. The harmless
    # comments are provenance fixtures, not a claim to execute historical code.
    for name in (*CHANGED[1:], PHYSICAL):
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(current_bytes[name] + (
            b"\n# previous CPU fixture revision\n" if name != PHYSICAL else b""))
        template["files"][name] = "pending"
    git(root, "add", ".")
    git(root, "commit", "-m", "source fixture")
    old = inventory(root, template)
    for name in CHANGED:
        (root / name).write_bytes(current_bytes[name])
    target_template = copy.deepcopy(template)
    target_template["files"][MODULE] = "pending"
    git(root, "add", ".")
    git(root, "commit", "-m", "history target fixture")
    new = inventory(root, target_template)

    training.seed_training_rngs(1001)
    runner, env = make(policies.STATE_DEPENDENT_POLICY, initialize=True)
    initial_std_head = runner.alg.actor.state_dict()["mlp.4.weight"][12:].clone()
    result = training.train_semantic(
        runner, env, run_dir=root / "source_run", output_root=root / "source_outputs",
        stage="full_episode", decisions=128, contract=old, seed=1001)
    assert result["optimizer_steps_this_run"] == 20
    assert runner.alg.optimizer.state_dict()["state"]
    assert not torch.equal(initial_std_head, runner.alg.actor.state_dict()["mlp.4.weight"][12:])
    previous = torch.load(result["checkpoints"][-1]["checkpoint"],
                          map_location="cpu", weights_only=False)["infos"]
    previous["new_mdp_origin_global_policy_decisions"] = 0
    # Prove the migration selects fresh 3e-5 Adam rather than inherited 1e-5.
    runner.alg.learning_rate = 1e-5
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = 1e-5
    path = root / "learned_source.pt"
    training.save_semantic_checkpoint(runner, path, previous)
    record = migration.build_v3_warm_start_record(
        path, new, project_root=root, target_policy_version=policies.HISTORY_POLICY)
    return SimpleNamespace(root=root, old=old, new=new, runner=runner, path=path,
        metadata=migration.checkpoint_metadata(path), record=record,
        source_bytes=path.read_bytes(),
        sidecar_bytes=path.with_name(path.stem + "_manifest.json").read_bytes())


def bind_real_validator(monkeypatch, source):
    original = migration.build_v3_warm_start_record
    monkeypatch.setattr(migration, "build_v3_warm_start_record",
        lambda checkpoint, contract, **kwargs: original(
            checkpoint, contract, project_root=source.root, **kwargs))


def test_learned_history_kernel_initial_save_exact_resume_and_official_update(learned_source, monkeypatch):
    source = learned_source
    bind_real_validator(monkeypatch, source)
    record = source.record
    kernel = record["policy_kernel_transition"]
    assert set(record["runtime_changed_files"]) == set(CHANGED)
    assert all(row["source_sha256"] == row["target_sha256"]
               for row in record["configuration_transition"].values())
    assert len(record["configuration_transition"]) == 6
    assert kernel["physical_mdp_changed"] is kernel["reward_changed"] is False
    assert kernel["parameter_layout_changed"] is kernel["old_rollout_inherited"] is False
    assert kernel["learned_parameters_preserved"] is True
    horizon = record["return_horizon_transition"]
    assert horizon["source"] == horizon["target"]
    assert (horizon["source"]["gamma"], horizon["source"]["lambda"]) == (.9985, .99)
    target, env = make(policies.HISTORY_POLICY)
    previous = training.load_semantic_checkpoint(
        target, source.path, contract=source.new, seed=1001, warm_start=record)
    assert target.alg.actor.state_dict().keys() == source.runner.alg.actor.state_dict().keys()
    for name, tensor in source.runner.alg.actor.state_dict().items():
        assert torch.equal(target.alg.actor.state_dict()[name], tensor), name
    assert training.parameter_hash(target.alg.critic) == training.parameter_hash(source.runner.alg.critic)
    assert training.state_hash(training._normalizers(target)) == source.metadata["normalizer_state_sha256"]
    assert all(not state for state in training._normalizers(target).values())
    assert training.capture_training_rng_state(seed=1001) == source.metadata["training_rng_state"]
    assert target.alg.optimizer.state_dict()["state"] == {}
    assert target.alg.learning_rate == 3e-5
    assert all(group["lr"] == 3e-5 for group in target.alg.optimizer.param_groups)
    assert target.alg.storage.step == 0 and target.alg.transition.actions is None
    assert tuple(target.alg.storage.actions.shape) == (128, 1, 12)
    assert previous["new_mdp_origin_global_policy_decisions"] == 0
    assert previous["stage_requested_decisions"] == {"smoke": 0, "phase_suffix": 0, "full_episode": 128}
    assert tuple(previous[name] for name in ("global_policy_decisions", "ppo_updates", "optimizer_steps")) == (128, 1, 20)

    comparison = training.compare_warm_start_policy_kernel(target, env, record=record)
    assert comparison["source_mean_full12"] != comparison["target_conditional_mean_full12"]
    assert comparison["source_to_target_conditional_kl"][0] > 0
    assert comparison["learned_weights_changed"] is comparison["rng_or_sampling_cache_changed"] is False
    with torch.inference_mode():
        observations = env.get_observations()
        old_head = source.runner.alg.actor.mlp(source.runner.alg.actor.get_latent(observations))
        assert comparison["shared_conditional_std_full12"] == old_head[..., 1, :].exp().tolist()
        assert comparison["shared_critic_value"] == source.runner.alg.critic(observations).tolist()
    assert env.core.calls == 0 and env.core.resets == 1
    assert training.capture_training_rng_state(seed=1001) == source.metadata["training_rng_state"]

    initial = source.root / migration.v3_warm_start_checkpoint_name(record)
    training.save_semantic_checkpoint(target, initial, {
        **previous, "runtime_contract": source.new, "stage": "initial",
        "new_mdp_initial_policy_kernel_comparison": comparison})
    embedded = torch.load(initial, map_location="cpu", weights_only=False)["infos"]
    metadata = migration.checkpoint_metadata(initial)
    assert all(metadata[key] == value for key, value in embedded.items())
    assert policies.policy_version_from_metadata(metadata) == policies.HISTORY_POLICY
    assert metadata["policy_contract"] == policies.policy_contract(policies.HISTORY_POLICY)
    assert metadata["new_mdp_warm_start"] == record
    assert metadata["actor_parameter_sha256"] == source.metadata["actor_parameter_sha256"]
    assert metadata["training_rng_state"] == source.metadata["training_rng_state"]
    assert source.path.read_bytes() == source.source_bytes
    assert source.path.with_name(source.path.stem + "_manifest.json").read_bytes() == source.sidecar_bytes

    resumed, resumed_env = make(policies.HISTORY_POLICY)
    restored = training.load_semantic_checkpoint(resumed, initial, contract=source.new, seed=1001)
    assert training.parameter_hash(resumed.alg.actor) == training.parameter_hash(target.alg.actor)
    assert training.parameter_hash(resumed.alg.critic) == training.parameter_hash(target.alg.critic)
    assert training.state_hash(resumed.alg.optimizer.state_dict()) == training.state_hash(target.alg.optimizer.state_dict())
    assert training.capture_training_rng_state(seed=1001) == metadata["training_rng_state"]
    assert resumed.alg.storage.step == 0 and resumed.alg.transition.actions is None
    result = training.train_semantic(resumed, resumed_env,
        run_dir=source.root / "target_run", output_root=source.root / "target_outputs",
        stage="full_episode", decisions=128, contract=source.new, seed=1001, resume_infos=restored)
    assert (result["global_policy_decisions"], result["ppo_updates_this_run"],
            result["optimizer_steps_this_run"]) == (256, 1, 20)
    assert result["finite_nonzero_gradient_observed"] is True
    assert result["actor_parameter_sha256_before"] != result["actor_parameter_sha256_after"]
    rollout = torch.load(source.root / "target_run/rollouts/rollout_000002.pt",
                         map_location="cpu", weights_only=False)
    rows = [json.loads(line) for line in
            (source.root / "target_run/residual_and_projection_audit.jsonl").read_text().splitlines()]
    assert len(rows) == 128
    for index, row in enumerate(rows):
        assert row["raw_policy_action_full12"] == rollout["actions"][index, 0].tolist()
        assert row["old_distribution_mean_full12"] == rollout["distribution_params"][0][index, 0].tolist()
        assert row["old_distribution_std_full12"] == rollout["distribution_params"][1][index, 0].tolist()
    expected_logprob = torch.distributions.Normal(*rollout["distribution_params"]).log_prob(rollout["actions"]).sum(-1)
    torch.testing.assert_close(expected_logprob, rollout["actions_log_prob"].squeeze(-1))
    observed = rollout["observations"]["policy"]
    for index in range(1, 128):
        expected = (torch.zeros(12) if rollout["dones"][index - 1, 0].item()
                    else rollout["actions"][index - 1, 0].clamp(-20, 20))
        assert torch.equal(observed[index, 0, 195:207], expected)
    final = migration.checkpoint_metadata(Path(result["checkpoints"][-1]["checkpoint"]))
    assert policies.policy_version_from_metadata(final) == policies.HISTORY_POLICY
    assert final["stage_requested_decisions"]["full_episode"] == 256
    assert (final["global_policy_decisions"], final["ppo_updates"], final["optimizer_steps"]) == (256, 2, 40)
    assert final["new_mdp_origin_global_policy_decisions"] == 0
    assert source.path.read_bytes() == source.source_bytes


def test_ordinary_cross_kernel_resume_is_rejected(learned_source):
    target, _ = make(policies.HISTORY_POLICY)
    before = training.parameter_hash(target.alg.actor)
    with pytest.raises(RuntimeError, match="policy distribution"):
        training.load_semantic_checkpoint(target, learned_source.path,
            contract=learned_source.old, seed=1001)
    assert training.parameter_hash(target.alg.actor) == before


@pytest.mark.parametrize("change", ["corrupt_record", "missing_selector"])
def test_explicit_kernel_record_is_required_and_reverified(learned_source, monkeypatch, change):
    source = learned_source
    if change == "missing_selector":
        record = migration.build_v3_warm_start_record(source.path, source.new, project_root=source.root)
        message = "policy kernel"
    else:
        record = copy.deepcopy(source.record)
        record["policy_kernel_transition"]["reward_changed"] = True
        message = "binding changed"
    bind_real_validator(monkeypatch, source)
    target, _ = make(policies.HISTORY_POLICY)
    with pytest.raises(RuntimeError, match=message):
        training.load_semantic_checkpoint(target, source.path, contract=source.new,
                                         seed=1001, warm_start=record)


def test_wrong_source_policy_and_missing_target_module_rejected(learned_source):
    source = learned_source
    legacy, _ = make(policies.LEGACY_POLICY, initialize=True)
    infos = {key: value for key, value in source.metadata.items()
             if key not in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip")}
    path = source.root / "wrong_policy_fixture.pt"
    training.save_semantic_checkpoint(legacy, path, infos)
    with pytest.raises(ValueError, match="v3 N1 heteroscedastic source"):
        migration.build_v3_warm_start_record(path, source.new, project_root=source.root,
                                             target_policy_version=policies.HISTORY_POLICY)
    missing = copy.deepcopy(source.new)
    missing["files"].pop(MODULE)
    missing["runtime_content_sha256"] = migration.digest(missing["files"])
    with pytest.raises(ValueError, match="non-whitelisted"):
        migration.build_v3_warm_start_record(source.path, missing, project_root=source.root,
                                             target_policy_version=policies.HISTORY_POLICY)


@pytest.mark.parametrize("relative", ["configs/ppo_semantic_v3/reward_config.yaml",
                                    "configs/ppo_semantic_v3/stage_task_spec.yaml", PHYSICAL])
def test_history_boundary_rejects_changed_reward_config_or_physical_bytes(learned_source, relative):
    source = learned_source
    path = source.root / relative
    before = path.read_bytes()
    try:
        path.write_bytes(before + b"\n# independent unapproved fixture delta\n")
        changed = inventory(source.root, source.new)
        with pytest.raises(ValueError, match="configuration|non-whitelisted"):
            migration.build_v3_warm_start_record(source.path, changed, project_root=source.root,
                                                 target_policy_version=policies.HISTORY_POLICY)
    finally:
        path.write_bytes(before)

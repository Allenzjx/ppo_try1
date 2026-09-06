from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from wlr50_clean.ppo import semantic_policy_distribution as policy
from wlr50_clean.ppo.semantic_migration import digest, file_sha, continuation_topology
from wlr50_clean.ppo.semantic_training import semantic_runner_config


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def metadata(version=policy.LEGACY_POLICY, *, semantic_version="v3"):
    result = {"seed": 1001, "semantic_version": semantic_version,
              "runner_config": semantic_runner_config(seed=1001, device="cpu",
                  semantic_version=semantic_version, policy_version=version)}
    if version != policy.LEGACY_POLICY:
        result["policy_contract"] = policy.policy_contract(version)
    return result


def actor(version):
    obs = TensorDict({"policy": torch.zeros(3, 324)}, batch_size=[3])
    cfg = copy.deepcopy(metadata(version)["runner_config"]["actor"])
    cfg.pop("class_name")
    return MLPModel(obs, {"actor": ["policy"]}, "actor", 12, **cfg)


def git(root, *arguments):
    return subprocess.run(["git", "-C", str(root), *arguments], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def migration_fixture(tmp_path):
    """Real git bytes and real sidecar hashing; no metadata validators mocked.

    The builder deliberately does not deserialize tensors. Official model tensor
    conversion is exercised separately below, rather than faking that API here.
    """
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init")
    git(root, "config", "user.email", "policy-test@example.invalid")
    git(root, "config", "user.name", "Policy test")
    git(root, "config", "core.autocrlf", "false")
    files = set(policy.ALLOWED_CHANGED_FILES) - {policy.MODULE_PATH}
    files |= {"src/wlr50_clean/ppo/semantic_reward.py", "src/wlr50_clean/ppo/semantic_supervisor.py",
              "configs/environment_lock.json"}
    files |= {f"configs/ppo_semantic_v3/{name}" for name in policy.CONFIG_NAMES}
    for relative in files:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"fixture source bytes: {relative}\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "source")
    old_files = {name: file_sha(root / name) for name in sorted(files)}
    old = {
        "schema": "wlr50_clean.semantic_runtime_contract.v1", "semantic_version": "v3",
        "source_git_commit": git(root, "rev-parse", "HEAD"), "files": old_files,
        "runtime_content_sha256": digest(old_files), "rsl_rl_version": "5.0.1",
        "local_runtime_versions": {"fixture": "CPU"}, "physics_hz": 120.0, "decision_hz": 15.0,
        "task_timeout_s": 200.0, "timeout_bootstrap": False,
        "training_budgets": {"smoke": 10000, "phase_suffix": 100000, "full_episode": 100000},
        "frozen_A_files": {"configs/environment_lock.json": old_files["configs/environment_lock.json"]},
        "selected_configuration": {name: {"path": f"configs/ppo_semantic_v3/{name}",
            "sha256": old_files[f"configs/ppo_semantic_v3/{name}"]} for name in policy.CONFIG_NAMES},
    }
    (root / "src/wlr50_clean/ppo/semantic_training.py").write_text("reviewed policy implementation\n", encoding="utf-8")
    (root / policy.MODULE_PATH).write_text("new policy module\n", encoding="utf-8")
    git(root, "add", ".")
    git(root, "commit", "-m", "policy architecture only")
    new_files = {name: file_sha(root / name) for name in sorted(files | {policy.MODULE_PATH})}
    new = {**copy.deepcopy(old), "source_git_commit": git(root, "rev-parse", "HEAD"),
           "files": new_files, "runtime_content_sha256": digest(new_files)}
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"opaque checkpoint bytes for the metadata-only validator")
    info = {**metadata(), "schema": "wlr50_clean.semantic_checkpoint.v1",
            "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": file_sha(checkpoint),
            "save_load_round_trip": True, "runtime_contract": old,
            "global_policy_decisions": 38272, "ppo_updates": 264, "optimizer_steps": 5280,
            "stage_requested_decisions": {"smoke": 0, "phase_suffix": 15360, "full_episode": 12800},
            "new_mdp_origin_global_policy_decisions": 10112,
            "optimizer_learning_rate": 1e-5, "normalization": policy.NORMALIZATION,
            "training_rng_state": {"schema": "wlr50_clean.training_rng_state.v1", "seed": 1001},
            "sampling": "P01_full_task_only_initial_version",
            "curriculum_epoch": {"reset_sampling": "P01_full_task_only_initial_version", "prefix_request": None},
            "execution_topology": continuation_topology("P01_full_task_only_initial_version", None),
            **{key: hashlib.sha256(key.encode()).hexdigest() for key in
               ("actor_parameter_sha256", "critic_parameter_sha256", "optimizer_state_sha256", "normalizer_state_sha256")}}
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    sidecar.write_text(json.dumps(info), encoding="utf-8")
    return root, checkpoint, sidecar, info, new


@pytest.mark.parametrize("version", [policy.LEGACY_POLICY, policy.STATE_DEPENDENT_POLICY])
def test_contract_and_inplace_configuration_only_change_two_fields(version):
    config = metadata()["runner_config"]
    config["actor"]["distribution_cfg"]["init_std"] = .173
    before = copy.deepcopy(config)
    policy.configure_policy_distribution(config, version)
    expected = copy.deepcopy(before)
    expected["actor"]["distribution_cfg"].update(
        class_name=policy.policy_contract(version)["distribution_class"],
        std_type=policy.policy_contract(version)["std_type"])
    assert config == expected
    first = policy.policy_contract(version)
    first["actor_hidden_dims"][0] = 1
    assert policy.policy_contract(version)["actor_hidden_dims"] == [256, 256]


@pytest.mark.parametrize("version", [policy.LEGACY_POLICY, policy.STATE_DEPENDENT_POLICY])
@pytest.mark.parametrize("semantic_version", ["v2", "v3"])
def test_complete_runner_config_validates_without_weak_fixture(version, semantic_version):
    info = metadata(version, semantic_version=semantic_version)
    assert policy.policy_version_from_metadata(info) == version
    if version == policy.LEGACY_POLICY:
        info["policy_contract"] = policy.policy_contract(version)
        assert policy.policy_version_from_metadata(info) == version


@pytest.mark.parametrize("change", ["missing_contract", "wrong_contract", "extra_config", "changed_gamma",
    "changed_normalizer", "changed_hidden", "changed_seed", "missing_field", "boolean_number", "policy_version"])
def test_rejects_incomplete_or_conflicting_metadata(change):
    info = metadata(policy.STATE_DEPENDENT_POLICY)
    if change == "missing_contract": info.pop("policy_contract")
    elif change == "wrong_contract": info["policy_contract"] = policy.policy_contract(policy.LEGACY_POLICY)
    elif change == "extra_config": info["runner_config"]["unreviewed"] = True
    elif change == "changed_gamma": info["runner_config"]["algorithm"]["gamma"] = .99
    elif change == "changed_normalizer": info["runner_config"]["actor"]["obs_normalization"] = True
    elif change == "changed_hidden": info["runner_config"]["actor"]["hidden_dims"] = [256, 128]
    elif change == "changed_seed": info["seed"] = 2001
    elif change == "missing_field": info["runner_config"].pop("save_interval")
    elif change == "boolean_number": info["runner_config"]["save_interval"] = True
    else: info["policy_version"] = policy.LEGACY_POLICY
    with pytest.raises(ValueError): policy.policy_version_from_metadata(info)


def test_legacy_missing_policy_contract_does_not_allow_other_distribution():
    info = metadata()
    info["runner_config"]["actor"]["distribution_cfg"]["std_type"] = "log"
    with pytest.raises(ValueError, match="unsupported policy"): policy.policy_version_from_metadata(info)


def test_real_official_model_mapping_preserves_mean_sigma_and_inputs(tmp_path):
    source, target = actor(policy.LEGACY_POLICY), actor(policy.STATE_DEPENDENT_POLICY)
    with torch.no_grad(): source.distribution.std_param.copy_(torch.linspace(.11, .22, 12))
    before_source = copy.deepcopy(source.state_dict())
    before_target = copy.deepcopy(target.state_dict())
    mapped, evidence = policy.map_gaussian_actor_state(source.state_dict(), target.state_dict())
    assert set(mapped) == set(target.state_dict())
    assert evidence["preserved_hidden_and_mean_weights_exact"]
    assert evidence["target_std_head_weights_zero"] and evidence["sigma_clipped"] is False
    assert evidence["removed_source_keys"] == ["distribution.std_param"]
    assert all(torch.equal(source.state_dict()[k], v) for k, v in before_source.items())
    assert all(torch.equal(target.state_dict()[k], v) for k, v in before_target.items())
    target.load_state_dict(mapped, strict=True)
    obs = TensorDict({"policy": torch.randn(19, 324)}, batch_size=[19])
    with torch.no_grad():
        torch.testing.assert_close(target(obs), source(obs), rtol=1e-6, atol=1e-7)
        source.distribution.update(source.mlp(obs["policy"]))
        target.distribution.update(target.mlp(obs["policy"]))
        torch.testing.assert_close(target.distribution.std, source.distribution.std, rtol=1e-6, atol=1e-8)
        assert float(target.distribution.kl_divergence(source.distribution.params, target.distribution.params).abs().max()) < 1e-5
        raw = torch.randn(19, 12)
        torch.testing.assert_close(target.get_output_log_prob(raw), source.get_output_log_prob(raw), rtol=1e-5, atol=2e-4)
    saved = tmp_path / "target.pt"
    torch.save(target.state_dict(), saved)
    fresh = actor(policy.STATE_DEPENDENT_POLICY)
    fresh.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    assert all(torch.equal(target.state_dict()[k], v) for k, v in fresh.state_dict().items())
    # The official reshape has no state key, but is present and yields 2x12.
    assert tuple(fresh.mlp(obs["policy"]).shape) == (19, 2, 12)
    assert isinstance(fresh.mlp[-1], torch.nn.Unflatten)


def test_official_log_std_half_receives_state_dependent_gradient_after_mapping():
    source, target = actor(policy.LEGACY_POLICY), actor(policy.STATE_DEPENDENT_POLICY)
    mapped, _ = policy.map_gaussian_actor_state(source.state_dict(), target.state_dict())
    target.load_state_dict(mapped, strict=True)
    obs = TensorDict({"policy": torch.randn(8, 324)}, batch_size=[8])
    target.distribution.update(target.mlp(obs["policy"]))
    actions = target.distribution.mean.detach() + torch.linspace(.01, .3, 8)[:, None]
    loss = -target.get_output_log_prob(actions).mean()
    loss.backward()
    gradient = target.mlp[4].weight.grad[12:]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    torch.optim.Adam(target.parameters(), lr=1e-5).step()
    target.distribution.update(target.mlp(obs["policy"]))
    assert torch.isfinite(target.distribution.std).all()
    assert not torch.equal(target.distribution.std[0], target.distribution.std[1])


@pytest.mark.parametrize("change", ["zero_sigma", "negative_sigma", "nan_sigma", "infinite_sigma",
    "extra_source", "extra_target", "missing_hidden", "wrong_shape", "wrong_dtype", "nan_hidden", "target_shape"])
def test_actor_mapping_rejects_unreviewed_tensor_changes(change):
    source = actor(policy.LEGACY_POLICY).state_dict()
    target = actor(policy.STATE_DEPENDENT_POLICY).state_dict()
    if change.endswith("sigma"):
        source["distribution.std_param"][3] = {"zero_sigma": 0., "negative_sigma": -.1,
            "nan_sigma": float("nan"), "infinite_sigma": float("inf")}[change]
    elif change == "extra_source": source["unexpected"] = torch.zeros(1)
    elif change == "extra_target": target["unexpected"] = torch.zeros(1)
    elif change == "missing_hidden": source.pop("mlp.2.bias")
    elif change == "wrong_shape": source["mlp.4.weight"] = torch.zeros(12, 128)
    elif change == "wrong_dtype": source["mlp.0.weight"] = source["mlp.0.weight"].double()
    elif change == "nan_hidden": source["mlp.2.weight"][0, 0] = float("nan")
    else: target["mlp.4.bias"] = torch.zeros(12)
    with pytest.raises(ValueError): policy.map_gaussian_actor_state(source, target)


def test_real_git_metadata_binding_preserves_effective_lr_and_lifetime(migration_fixture):
    root, source, sidecar, info, new = migration_fixture
    record = policy.build_policy_distribution_migration(source, new, project_root=root)
    assert record["physical_mdp_changed"] is False and record["reward_changed"] is False
    assert record["source_policy_version"] == policy.LEGACY_POLICY
    assert record["target_policy_version"] == policy.STATE_DEPENDENT_POLICY
    assert record["optimizer"]["initial_learning_rate"] == 1e-5
    assert info["runner_config"]["algorithm"]["learning_rate"] == 3e-5
    assert record["optimizer"]["old_moments_inherited"] is False
    assert record["source_global_policy_decisions"] == 38272
    assert record["source_ppo_updates"] == 264 and record["source_optimizer_steps"] == 5280
    assert record["new_mdp_origin_global_policy_decisions"] == 10112
    assert record["target_stage_requested_decisions"] == info["stage_requested_decisions"]
    assert record["changed_file_hashes"][policy.MODULE_PATH]["before"] is None
    assert record["old_rollout_buffer_inherited"] is False
    assert record["migration_added_policy_decisions"] == record["migration_added_ppo_updates"] == record["migration_added_optimizer_steps"] == 0
    assert record["source_manifest_sha256"] == file_sha(sidecar)
    assert policy.build_policy_distribution_migration(source, new, project_root=root) == record
    name = policy.policy_migration_checkpoint_name(record)
    assert Path(name).name == name and name.endswith(".pt") and "000038272" in name
    changed = copy.deepcopy(record)
    changed["source_checkpoint_sha256"] = "a" * 64
    assert policy.policy_migration_checkpoint_name(changed) != name
    changed = copy.deepcopy(record)
    changed["target_runtime_contract"]["source_git_commit"] = "b" * 40
    assert policy.policy_migration_checkpoint_name(changed) != name


@pytest.mark.parametrize("change", ["reward", "config", "delete", "extra_file", "effective_lr", "budget", "origin",
    "seed", "physical_rate", "target_bytes", "historical_bytes", "source_hash", "hetero_source", "non_v3"])
def test_migration_rejects_unbound_or_non_policy_changes(migration_fixture, change):
    root, source, sidecar, info, new = migration_fixture
    if change in ("reward", "config"):
        relative = ("src/wlr50_clean/ppo/semantic_reward.py" if change == "reward" else
                    "configs/ppo_semantic_v3/reward_config.yaml")
        (root / relative).write_text("unreviewed physical reward\n", encoding="utf-8")
        new["files"][relative] = file_sha(root / relative)
    elif change == "delete": new["files"].pop("src/wlr50_clean/ppo/semantic_video_cli.py")
    elif change == "extra_file": new["files"]["src/wlr50_clean/ppo/new_dynamics.py"] = "a" * 64
    elif change == "effective_lr": info["optimizer_learning_rate"] = float("nan")
    elif change == "budget": info["stage_requested_decisions"]["full_episode"] = 100001
    elif change == "origin": info["new_mdp_origin_global_policy_decisions"] = 40000
    elif change == "seed": info["training_rng_state"]["seed"] = 2001
    elif change == "physical_rate": new["decision_hz"] = 30.0
    elif change == "target_bytes": (root / policy.MODULE_PATH).write_text("changed after binding\n", encoding="utf-8")
    elif change == "historical_bytes":
        info["runtime_contract"]["files"]["src/wlr50_clean/ppo/semantic_training.py"] = "a" * 64
        info["runtime_contract"]["runtime_content_sha256"] = digest(info["runtime_contract"]["files"])
    elif change == "source_hash": source.write_bytes(b"corrupt checkpoint")
    elif change == "hetero_source":
        info.update(metadata(policy.STATE_DEPENDENT_POLICY))
    else:
        info.update(metadata(semantic_version="v2"))
    new["runtime_content_sha256"] = digest(new["files"])
    sidecar.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises((ValueError, subprocess.CalledProcessError, FileNotFoundError)):
        policy.build_policy_distribution_migration(source, new, project_root=root)


def test_pure_metadata_module_import_does_not_import_torch_in_fresh_process():
    import os
    import sys

    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(policy.PROJECT_ROOT / "src")
    result = subprocess.run([sys.executable, "-P", "-c",
        "import sys; import wlr50_clean.ppo.semantic_policy_distribution as p; "
        "assert 'torch' not in sys.modules; assert p.policy_contract(p.STATE_DEPENDENT_POLICY)['std_type']=='log'"],
        check=True, capture_output=True, text=True, env=environment)
    assert result.returncode == 0

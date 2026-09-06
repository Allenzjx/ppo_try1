"""Audited semantic PPO using the installed, unchanged official RSL-RL optimizer.

This is orchestration and instrumentation, not another PPO implementation.
The policy distribution is over raw Gaussian latent actions; projection never
replaces those samples or their old log probabilities in RSL storage.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from .rl_library_wrapper import (
    RSL_VERSION, assert_supported_rsl_runtime, build_rsl_runner_config,
    capture_training_rng_state, construct_runner, initialize_zero_mean_actor,
    load_checkpoint_round_trip, optimizer_learning_rate,
    restore_training_rng_state, seed_training_rngs, sha256_file,
)
from .semantic_policy_distribution import (
    LEGACY_POLICY, STATE_DEPENDENT_POLICY, configure_policy_distribution, policy_contract,
)

SEMANTIC_TRAINING_SCHEMA = "wlr50_clean.semantic_training.v1"
SEMANTIC_CHECKPOINT_SCHEMA = "wlr50_clean.semantic_checkpoint.v1"
STAGE_BUDGETS = {"smoke": 10_000, "phase_suffix": 100_000, "full_episode": 100_000}
ROLLOUT_LENGTH = 128


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        return
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unserializable semantic evidence: {type(value).__name__}")


def parameter_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.named_parameters()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def state_hash(value: Any) -> str:
    """Deterministic tensor-aware hash for optimizer and normalizer round trips."""
    digest = hashlib.sha256()
    def visit(item: Any) -> None:
        if hasattr(item, "detach"):
            tensor = item.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, Mapping):
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                digest.update(repr(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()


def semantic_runner_config(*, seed: int, device: str = "cuda:0", semantic_version: str = "v2",
                           policy_version: str = LEGACY_POLICY) -> dict[str, Any]:
    if semantic_version not in ("v2", "v3"):
        raise ValueError("unsupported semantic runtime version")
    # Same installed PPO/network hyperparameters; independent semantic identity.
    profile = SimpleNamespace(
        activation="elu", entropy_start=0.005, actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256), initial_action_std=0.15,
        rollout_length=ROLLOUT_LENGTH, update_epochs=5, num_minibatches=4,
        clip_ratio=0.20, gamma=0.995, lam=0.95, value_loss_coefficient=1.0,
        learning_rate=0.00003 if semantic_version == "v3" else 0.0003,
        max_grad_norm=1.0, schedule="adaptive", target_kl=0.01,
    )
    config = build_rsl_runner_config(profile, seed=seed, max_iterations=2000,
                                     experiment_name=f"wlr50_semantic_residual_{semantic_version}")
    config["device"] = device
    # Preserve the existing fixed-schema normalization contract. Its identity
    # normalizer state is still included in model checkpoints and verified.
    config["actor"]["obs_normalization"] = False
    config["critic"]["obs_normalization"] = False
    configure_policy_distribution(config, policy_version)
    return config


def verified_native_effect(info: Mapping[str, Any], raw: tuple[float, ...]) -> int:
    """Bind evidence to the sampled request; exact zero is valid training data."""
    native = info.get("actuator_target_effect_audit")
    required = ("verified", "actual_mapping_matches_dispatch", "setter_dispatch_targets_equal", "same_tick_counterfactual")
    if (not isinstance(native, Mapping) or native.get("schema") != "wlr50_clean.actuator_target_effect_audit.v1"
            or any(native.get(key) is not True for key in required)
            or tuple(native.get("raw_policy_action_full12", ())) != raw
            or native.get("target_dtype") != "torch.float32"):
        raise RuntimeError("semantic transition lacks verified same-tick native target audit")
    changed = native.get("changed_target_channel_count")
    if type(changed) is not int or not 0 <= changed <= 12:
        raise RuntimeError("semantic transition has invalid native target effect count")
    return changed


class SemanticRslAdapter:
    """One real core environment. Valid failed episodes remain on-policy data."""
    num_envs = 1
    num_actions = 12
    max_episode_length = 3000

    def __init__(self, core: Any, *, seed: int, device: str = "cuda:0") -> None:
        import torch
        self.core, self.seed, self.device = core, seed, device
        self.cfg = {"schema": "wlr50_clean.semantic_rsl_env.v1", "num_envs": 1,
                    "episode_timeout": "finite_horizon_task_terminal_no_bootstrap",
                    "reset_sampling": "P01_full_task_only_initial_version"}
        self.episode_length_buf = torch.zeros(1, dtype=torch.long, device=device)
        self.completed_episodes: list[dict[str, Any]] = []
        self.total_decisions = 0
        self._observation = tuple(core.reset(seed=seed))
        self.observation_dimension = len(self._observation)
        self.cfg["observation_dimension"] = self.observation_dimension

    def get_observations(self) -> Any:
        import torch
        from tensordict import TensorDict
        tensor = torch.tensor([self._observation], dtype=torch.float32, device=self.device)
        if tensor.shape != (1, self.observation_dimension) or not bool(torch.isfinite(tensor).all()):
            raise RuntimeError("semantic observation is non-finite or changed dimension")
        return TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[1], device=self.device)

    def step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, Any]]:
        import torch
        if actions.shape != (1, 12) or not bool(torch.isfinite(actions).all()):
            raise RuntimeError("semantic PPO must pass one finite raw Full12 action")
        raw = tuple(float(value) for value in actions[0].detach().cpu().tolist())
        step = self.core.step(raw)
        reward = float(step.reward)
        if not math.isfinite(reward):
            raise RuntimeError("semantic reward is non-finite")
        if bool(step.truncated):
            raise RuntimeError("semantic core returned external truncation; task timeout must be terminal")
        self.total_decisions += 1
        self.episode_length_buf += 1
        self._observation = tuple(step.observation)
        # Validate the true final observation before a reset can replace it.
        terminal_observation = self.get_observations()
        info = jsonable(dict(step.info))
        if "raw_policy_action_full12" not in info or tuple(info["raw_policy_action_full12"]) != raw:
            raise RuntimeError("semantic applied audit is not bound to the raw policy sample")
        verified_native_effect(info, raw)
        extras: dict[str, Any] = {"semantic_decisions": [info], "episode_summaries": [],
                                  "time_outs": torch.zeros(1, dtype=torch.bool, device=self.device)}
        done = bool(step.terminated)
        if done:
            summary = {"episode_index": len(self.completed_episodes), "seed": self.seed,
                       "policy_decisions": int(self.episode_length_buf.item()),
                       "termination_reason": info.get("termination_reason"),
                       "task_success": info.get("task_success") is True,
                       "task_outcome_label": info.get("task_outcome_label"),
                       "full_task_success": info.get("full_task_success", info.get("task_success")) is True,
                       "duration_s": float(self.core.frame.sim_time_s),
                       "terminal_info": info}
            self.completed_episodes.append(summary)
            extras["episode_summaries"] = [summary]
            extras["terminal_observation"] = terminal_observation
            self._observation = tuple(self.core.reset(seed=self.seed))
            self.episode_length_buf.zero_()
        return (self.get_observations(), torch.tensor([reward], device=self.device),
                torch.tensor([done], dtype=torch.bool, device=self.device), extras)

    def telemetry_summary(self) -> dict[str, Any]:
        return {"policy_decisions": self.total_decisions,
                "completed_episode_count": len(self.completed_episodes),
                "success_count": sum(row["task_success"] for row in self.completed_episodes),
                "core": jsonable(self.core.telemetry_summary()),
                "reset_sampling": "P01_full_task_only_initial_version"}


def construct_semantic_runner(env: Any, *, seed: int, device: str,
                              policy_version: str = LEGACY_POLICY,
                              initialize_actor: bool = True) -> tuple[Any, dict[str, Any]]:
    assert_supported_rsl_runtime()
    config = semantic_runner_config(seed=seed, device=device,
                                    semantic_version=env.cfg.get("semantic_version", "v2"),
                                    policy_version=policy_version)
    runner = construct_runner(env, config, log_dir=None)
    runner.logger.writer = None  # No hidden upstream automatic checkpoint writes.
    runner._semantic_policy_version = policy_version
    runner._semantic_version = env.cfg.get("semantic_version", "v2")
    runner._semantic_runner_config = copy.deepcopy(config)
    if initialize_actor:
        if policy_version == LEGACY_POLICY:
            initialize_zero_mean_actor(runner)
        else:
            # The official log-std half has its own initialization. Zeroing the
            # whole head would silently replace sigma=.15 with exp(0)=1.
            import torch
            linears = [m for m in runner.alg.actor.mlp.modules() if isinstance(m, torch.nn.Linear)]
            if not linears or linears[-1].out_features != 24:
                raise RuntimeError("state-dependent Full12 actor requires the official 24-row head")
            with torch.no_grad():
                linears[-1].weight[:12].zero_()
                linears[-1].bias[:12].zero_()
    return runner, config


def audited_ppo_update(runner: Any) -> dict[str, Any]:
    """Observe official minibatches and gradients without replacing PPO math."""
    import torch
    alg = runner.alg
    before = parameter_hash(alg.actor)
    generator = alg.storage.mini_batch_generator
    log_prob = alg.actor.get_output_log_prob
    kl_method = alg.actor.get_kl_divergence
    active: dict[str, Any] = {}
    gradients, clips, kls = [], [], []

    def batches(*args: Any, **kwargs: Any):
        for batch in generator(*args, **kwargs):
            active["batch"] = batch
            yield batch

    def observed_log_prob(raw: Any):
        result = log_prob(raw)
        batch = active["batch"]
        if not torch.equal(raw, batch.actions):
            raise RuntimeError("PPO likelihood did not use stored raw actions")
        with torch.no_grad():
            ratio = torch.exp(result - batch.old_actions_log_prob.squeeze(-1))
            clips.append(float(((ratio - 1).abs() > alg.clip_param).float().mean()))
        return result

    def observed_kl(*args: Any, **kwargs: Any):
        result = kl_method(*args, **kwargs)
        kls.append(float(result.detach().mean()))
        return result

    def optimizer_pre_step(optimizer: Any, args: Any, kwargs: Any):
        grads = [parameter.grad.detach() for group in optimizer.param_groups
                 for parameter in group["params"] if parameter.grad is not None]
        if not grads or any(not bool(torch.isfinite(grad).all()) for grad in grads):
            raise RuntimeError("PPO optimizer received missing/non-finite gradients")
        norm = math.sqrt(sum(float(grad.double().square().sum()) for grad in grads))
        gradients.append(norm)

    handle = alg.optimizer.register_step_pre_hook(optimizer_pre_step)
    try:
        alg.storage.mini_batch_generator = batches
        alg.actor.get_output_log_prob = observed_log_prob
        alg.actor.get_kl_divergence = observed_kl
        loss = alg.update()
    finally:
        handle.remove()
        alg.storage.mini_batch_generator = generator
        alg.actor.get_output_log_prob = log_prob
        alg.actor.get_kl_divergence = kl_method
    expected_steps = alg.num_learning_epochs * alg.num_mini_batches
    if len(gradients) != expected_steps or len(clips) != expected_steps or len(kls) != expected_steps:
        raise RuntimeError("official PPO minibatch instrumentation count mismatch")
    after = parameter_hash(alg.actor)
    values = [*gradients, *clips, *kls, *(float(value) for value in loss.values())]
    if any(not math.isfinite(value) for value in values):
        raise RuntimeError("non-finite PPO diagnostics")
    return {"optimizer_steps": len(gradients), "actor_parameter_sha256_before": before,
            "actor_parameter_sha256_after": after, "actor_parameters_changed": before != after,
            "finite_nonzero_gradient_observed": any(value > 0 for value in gradients),
            "gradient_norm_min": min(gradients), "gradient_norm_max": max(gradients),
            "kl_mean": sum(kls) / len(kls), "clip_fraction": sum(clips) / len(clips),
            "entropy": float(loss["entropy"]), "value_loss": float(loss["value"]),
            "surrogate_loss": float(loss["surrogate"]), "optimizer_learning_rate": optimizer_learning_rate(runner)}


def _normalizers(runner: Any) -> dict[str, Any]:
    return {role: getattr(runner.alg, role).obs_normalizer.state_dict() for role in ("actor", "critic")}


def save_semantic_checkpoint(runner: Any, checkpoint: Path, infos: Mapping[str, Any]) -> tuple[Path, Path]:
    """Publish immutable official state, then prove a real load restores it."""
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    metadata = {**dict(infos), "schema": SEMANTIC_CHECKPOINT_SCHEMA,
                "semantic_version": runner._semantic_version,
                "runner_config": copy.deepcopy(runner._semantic_runner_config),
                "optimizer_learning_rate": optimizer_learning_rate(runner),
                "actor_parameter_sha256": parameter_hash(runner.alg.actor),
                "critic_parameter_sha256": parameter_hash(runner.alg.critic),
                "optimizer_state_sha256": state_hash(runner.alg.optimizer.state_dict()),
                "normalizer_state_sha256": state_hash(_normalizers(runner)),
                "normalization": "fixed_versioned_observation_schema; identity_RSL_normalizer",
                "training_rng_state": capture_training_rng_state(seed=int(infos["seed"])),
                "physical_env_state_saved": False,
                "resume_physics": "legal_reset_not_bitwise_continuation"}
    if runner.alg.storage.observations["policy"].shape[-1] == 324:
        metadata["policy_contract"] = policy_contract(runner._semantic_policy_version)
    elif runner._semantic_policy_version != LEGACY_POLICY:
        raise RuntimeError("state-dependent semantic checkpoints require 324 observations")
    runner.save(str(checkpoint), infos=metadata)
    loaded = load_checkpoint_round_trip(runner, checkpoint)
    if dict(loaded) != metadata:
        raise RuntimeError("semantic checkpoint infos failed actual save/load round trip")
    if (parameter_hash(runner.alg.actor) != metadata["actor_parameter_sha256"]
            or parameter_hash(runner.alg.critic) != metadata["critic_parameter_sha256"]
            or state_hash(runner.alg.optimizer.state_dict()) != metadata["optimizer_state_sha256"]
            or state_hash(_normalizers(runner)) != metadata["normalizer_state_sha256"]):
        raise RuntimeError("semantic checkpoint model/optimizer/normalizer round trip changed state")
    restore_training_rng_state(metadata["training_rng_state"], expected_seed=int(infos["seed"]))
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    write_json(sidecar, {**metadata, "checkpoint_path": str(checkpoint.resolve()),
                         "checkpoint_sha256": sha256_file(checkpoint), "save_load_round_trip": True})
    return checkpoint, sidecar


def load_semantic_checkpoint(runner: Any, checkpoint: Path, *, contract: Mapping[str, Any], seed: int,
                             migration: Mapping[str, Any] | None = None,
                             warm_start: Mapping[str, Any] | None = None,
                             policy_migration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if policy_migration is not None:
        if migration is not None or warm_start is not None:
            raise ValueError("policy distribution, new-MDP and exact runtime migrations are separate operations")
        return _load_policy_distribution_migration(runner, checkpoint, contract=contract,
                                                   seed=seed, record=policy_migration)
    if warm_start is not None:
        if migration is not None:
            raise ValueError("new-MDP warm start and exact resume migration are distinct operations")
        return _load_v3_warm_start(runner, checkpoint, contract=contract, seed=seed, record=warm_start)
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    from .semantic_policy_distribution import policy_version_from_metadata
    if policy_version_from_metadata(metadata) != runner._semantic_policy_version:
        raise RuntimeError("checkpoint policy distribution differs from the constructed actor")
    expected_contract = dict(contract)
    from .semantic_migration import source_num_envs
    source_count = source_num_envs(metadata)
    target_count = int(runner.alg.storage.actions.shape[1])
    execution_factor = None if migration is None else migration.get("execution_factor")
    if execution_factor is not None and (execution_factor["source_num_envs"] != source_count or
            execution_factor["target_num_envs"] != target_count):
        raise RuntimeError("explicit topology plan does not match actual source/target storage")
    if source_count != target_count and (execution_factor is None or
            execution_factor["source_num_envs"] != source_count or
            execution_factor["target_num_envs"] != target_count):
        raise RuntimeError("PPO topology change requires explicit reviewed 1-to-8/8-to-1 migration")
    if runner.alg.storage.step != 0 or runner.alg.transition.actions is not None:
        raise RuntimeError("checkpoint loading cannot reuse a partial old rollout")
    if target_count == 8 and tuple(runner.alg.storage.actions.shape) != (128,8,12):
        raise RuntimeError("N8 requires fresh128x8x12 raw-action storage")
    if migration is not None:
        from .semantic_migration import validate_migration_plan
        verified = validate_migration_plan(checkpoint, contract, Path(migration["plan_path"]))
        if verified != dict(migration):
            raise RuntimeError("migration changed since pre-AppLauncher validation")
        expected_contract = metadata["runtime_contract"]
        if (runner.alg.storage.observations["policy"].shape[-1] != 324
                or runner.alg.storage.actions.shape[-1] != 12
                or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None):
            raise RuntimeError("migration requires fresh 324-observation/12-action storage with no old rollout")
        if metadata.get("runner_config") != semantic_runner_config(seed=seed, device=str(runner.device),
                semantic_version=metadata.get("semantic_version", "v2"),
                policy_version=runner._semantic_policy_version):
            raise RuntimeError("migration cannot change PPO hyperparameters or normalization")
    if (metadata.get("schema") != SEMANTIC_CHECKPOINT_SCHEMA
            or metadata.get("checkpoint_path") != str(checkpoint.resolve())
            or metadata.get("checkpoint_sha256") != sha256_file(checkpoint)
            or metadata.get("runtime_contract") != expected_contract
            or metadata.get("seed") != seed or metadata.get("save_load_round_trip") is not True):
        raise RuntimeError("semantic resume checkpoint version/hash/seed contract mismatch")
    infos = dict(load_checkpoint_round_trip(runner, checkpoint))
    if any(key not in metadata or metadata[key] != value for key, value in infos.items()):
        raise RuntimeError("semantic resume embedded infos differ from sidecar")
    for key, actual in (("actor_parameter_sha256", parameter_hash(runner.alg.actor)),
                        ("critic_parameter_sha256", parameter_hash(runner.alg.critic)),
                        ("optimizer_state_sha256", state_hash(runner.alg.optimizer.state_dict())),
                        ("normalizer_state_sha256", state_hash(_normalizers(runner)))):
        if infos.get(key) != actual:
            raise RuntimeError(f"semantic resume failed actual {key} verification")
    restore_training_rng_state(infos["training_rng_state"], expected_seed=seed)
    if migration is not None:
        infos = {**infos, "resume_migration": dict(migration)}
    infos = {**infos, "resume_source_checkpoint": {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": str(sidecar.resolve()), "manifest_sha256": sha256_file(sidecar)}}
    return infos


def _load_policy_distribution_migration(runner: Any, checkpoint: Path, *,
        contract: Mapping[str, Any], seed: int, record: Mapping[str, Any]) -> dict[str, Any]:
    """One explicit architecture boundary, not a new physical task/reward MDP."""
    import torch
    from .semantic_policy_distribution import (
        build_policy_distribution_migration, map_gaussian_actor_state,
    )
    verified = build_policy_distribution_migration(checkpoint, contract)
    if verified != dict(record):
        raise RuntimeError("policy migration binding changed after preflight")
    storage = runner.alg.storage
    if (runner._semantic_policy_version != STATE_DEPENDENT_POLICY
            or runner._semantic_version != "v3"
            or tuple(storage.actions.shape) != (128, 1, 12)
            or storage.observations["policy"].shape[-1] != 324
            or storage.step != 0 or runner.alg.transition.actions is not None):
        raise RuntimeError("policy migration requires fresh v3 N1 324/12 storage and the target actor")
    if runner._semantic_runner_config != semantic_runner_config(seed=seed, device=str(runner.device),
            semantic_version="v3", policy_version=STATE_DEPENDENT_POLICY):
        raise RuntimeError("policy migration cannot silently change PPO hyperparameters")
    # Construction reads the existing observation only: no reset, action,
    # physical snapshot restore or additional simulation step is performed.
    source, _ = construct_semantic_runner(runner.env, seed=seed, device=str(runner.device),
                                         policy_version=LEGACY_POLICY, initialize_actor=False)
    infos = load_semantic_checkpoint(source, checkpoint,
                                    contract=record["source_runtime_contract"], seed=seed)
    if (any(_normalizers(source).values()) or source.alg.actor.obs_normalization
            or source.alg.critic.obs_normalization):
        raise RuntimeError("policy migration requires the compatible identity normalizers")
    mapped, mapping_evidence = map_gaussian_actor_state(source.alg.actor.state_dict(),
                                                       runner.alg.actor.state_dict())
    runner.alg.actor.load_state_dict(mapped, strict=True)
    runner.alg.critic.load_state_dict(source.alg.critic.state_dict(), strict=True)
    if (parameter_hash(runner.alg.critic) != infos["critic_parameter_sha256"]
            or state_hash(_normalizers(runner)) != infos["normalizer_state_sha256"]):
        raise RuntimeError("policy migration failed exact critic/normalizer preservation")
    observations = runner.env.get_observations().to(runner.device)
    with torch.inference_mode():
        source_mean = source.alg.actor(observations, stochastic_output=False)
        target_mean = runner.alg.actor(observations, stochastic_output=False)
        source.alg.actor.distribution.update(source.alg.actor.mlp(source.alg.actor.get_latent(observations)))
        runner.alg.actor.distribution.update(runner.alg.actor.mlp(runner.alg.actor.get_latent(observations)))
        source_std, target_std = source.alg.actor.output_std, runner.alg.actor.output_std
        source_value, target_value = source.alg.critic(observations), runner.alg.critic(observations)
        for before, after in ((source_mean, target_mean), (source_std, target_std), (source_value, target_value)):
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
        kl = runner.alg.actor.get_kl_divergence(source.alg.actor.output_distribution_params,
                                               runner.alg.actor.output_distribution_params)
        if not bool(torch.isfinite(kl).all()) or float(kl.abs().max()) > 1e-6:
            raise RuntimeError("mapped initial policy distribution is not equivalent")
        comparison = {"scope": "same_current_observation_no_sampling_or_physics_step",
                      "observation_sha256": state_hash(observations["policy"]),
                      "mean_max_abs_difference": float((source_mean-target_mean).abs().max()),
                      "std_max_abs_difference": float((source_std-target_std).abs().max()),
                      "value_max_abs_difference": float((source_value-target_value).abs().max()),
                      "kl_max_abs": float(kl.abs().max()),
                      "bitwise_trajectory_equivalence_claimed": False}
    if type(source.alg.optimizer) is not torch.optim.Adam or len(source.alg.optimizer.param_groups) != 1:
        raise RuntimeError("policy migration supports only the verified single-group official Adam")
    actual_lr = optimizer_learning_rate(source)
    if actual_lr != record["optimizer"]["initial_learning_rate"]:
        raise RuntimeError("source effective learning rate differs from policy migration record")
    adam_options = {key: copy.deepcopy(source.alg.optimizer.param_groups[0][key])
                    for key in source.alg.optimizer.defaults if key != "lr"}
    runner.alg.optimizer = torch.optim.Adam(
        list(runner.alg.actor.parameters()) + list(runner.alg.critic.parameters()),
        lr=actual_lr, **adam_options)
    runner.alg.learning_rate = actual_lr
    runner.current_learning_iteration = int(infos["ppo_updates"])
    # Constructions consume RNG; comparisons above do not sample. Restore last
    # so the next action starts at the source RNG boundary, not constructor RNG.
    restore_training_rng_state(infos["training_rng_state"], expected_seed=seed)
    evidence = {"mapping": mapping_evidence, "same_observation_comparison": comparison,
                "source_actor_parameter_sha256": infos["actor_parameter_sha256"],
                "target_actor_parameter_sha256": parameter_hash(runner.alg.actor),
                "critic_parameter_sha256": parameter_hash(runner.alg.critic),
                "normalizer_state_sha256": state_hash(_normalizers(runner)),
                "optimizer_state": "fresh_Adam_no_old_moments",
                "optimizer_learning_rate": actual_lr,
                "optimizer_options": jsonable(adam_options),
                "rollout_step": storage.step, "physical_steps_added": 0,
                "policy_decisions_added": 0, "ppo_updates_added": 0, "optimizer_steps_added": 0,
                "source_rng_restored_after_construction_and_comparison": True}
    return {**infos, "semantic_version": "v3",
            "policy_contract": policy_contract(STATE_DEPENDENT_POLICY),
            "runner_config": copy.deepcopy(runner._semantic_runner_config),
            "policy_distribution_migration": dict(record),
            "policy_distribution_migration_evidence": evidence}


def _load_v3_warm_start(runner: Any, checkpoint: Path, *, contract: Mapping[str, Any],
                        seed: int, record: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse learned networks, explicitly discard old optimizer and rollout state."""
    import torch
    from .semantic_migration import build_v3_warm_start_record
    verified = build_v3_warm_start_record(checkpoint, contract)
    if verified != dict(record):
        raise RuntimeError("new-MDP checkpoint/configuration binding changed after preflight")
    storage = runner.alg.storage
    if (tuple(storage.actions.shape) != (128, 1, 12)
            or storage.observations["policy"].shape[-1] != 324
            or storage.step != 0 or runner.alg.transition.actions is not None):
        raise RuntimeError("v3 warm start requires fresh N1 324-observation/12-action rollout storage")
    expected_config = semantic_runner_config(seed=seed, device=str(runner.device), semantic_version="v3",
                                             policy_version=runner._semantic_policy_version)
    # Official RSL 5.0.1 consumes these factory-only entries during construction.
    for role in ("actor", "critic", "algorithm"):
        if type(getattr(runner.alg, role, runner.alg)).__name__ != expected_config[role].pop("class_name"):
            raise RuntimeError("v3 runner factory produced an unexpected official model/algorithm")
    expected_config["actor"]["distribution_cfg"].pop("class_name")
    expected_config["algorithm"].pop("share_cnn_encoders")
    if runner.cfg != expected_config:
        raise RuntimeError("v3 runner must use the explicit continuation configuration")
    infos = dict(load_checkpoint_round_trip(runner, checkpoint))
    from .semantic_migration import checkpoint_metadata
    metadata = checkpoint_metadata(checkpoint)
    if infos.get("seed") != seed or any(metadata.get(key) != value for key, value in infos.items()):
        raise RuntimeError("v3 warm start source embedded metadata/seed differs from verified sidecar")
    for key, actual in (("actor_parameter_sha256", parameter_hash(runner.alg.actor)),
                        ("critic_parameter_sha256", parameter_hash(runner.alg.critic)),
                        ("optimizer_state_sha256", state_hash(runner.alg.optimizer.state_dict())),
                        ("normalizer_state_sha256", state_hash(_normalizers(runner)))):
        if infos.get(key) != actual:
            raise RuntimeError(f"v3 warm start failed source {key} verification")
    if any(_normalizers(runner).values()):
        raise RuntimeError("v3 continuation requires verified identity RSL normalizers")
    # The temporary official restore above proves the saved state before this
    # deliberate new-MDP reset. No old Adam moment is used by any optimizer step.
    runner.alg.optimizer = torch.optim.Adam(
        list(runner.alg.actor.parameters()) + list(runner.alg.critic.parameters()), lr=3e-5)
    runner.alg.learning_rate = 3e-5
    restore_training_rng_state(infos["training_rng_state"], expected_seed=seed)
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    return {**infos, "semantic_version": "v3", "new_mdp_warm_start": dict(record),
            "source_stage_requested_decisions": dict(infos.get("stage_requested_decisions", {})),
            "stage_requested_decisions": dict(verified["target_stage_requested_decisions"]),
            "new_mdp_origin_global_policy_decisions": verified["new_mdp_origin_global_policy_decisions"],
            "resume_source_checkpoint": {"checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint), "manifest": str(sidecar.resolve()),
                "manifest_sha256": sha256_file(sidecar)}}


def compare_warm_start_action(runner: Any, env: Any, *, old_execution_profile: Path,
                              new_execution_profile: Path) -> dict[str, Any]:
    """Same current reset observation/nominal, fresh zero-history projectors only."""
    import torch
    from .semantic_backend import build_semantic_projector
    observations = env.get_observations().to(runner.device)
    frame = env.core.frame
    actor_training = runner.alg.actor.training
    try:
        runner.alg.actor.eval()
        with torch.inference_mode():
            raw = tuple(float(x) for x in runner.alg.actor(observations, stochastic_output=False)[0].cpu().tolist())
    finally:
        runner.alg.actor.train(actor_training)
    nominal = tuple(frame.nominal_action_full12)
    results = {}
    fields = ("bounded_residual_full12", "scaled_residual_full12", "masked_residual_full12",
              "rate_projected_residual_full12", "safe_projected_residual_full12", "applied_action_full12")
    for name, path in (("old", old_execution_profile), ("new", new_execution_profile)):
        result = build_semantic_projector(path).project(raw, state_id=frame.state_id,
            nominal_action_full12=nominal, reference_action_full12=nominal,
            reference_delta_full12=(0.0,) * 12, previous_projected_residual_full12=(0.0,) * 12,
            runtime_action_mask_full12=frame.action_mask_full12, safety=frame.safety_projection, dt_s=1/120)
        results[name] = {"execution_profile": str(path.resolve()), "sha256": sha256_file(path),
                         **{field: list(getattr(result, field)) for field in fields}}
    return {"schema": "wlr50_clean.semantic_new_mdp_same_state_action_comparison.v1",
            "phase_id": frame.state_id, "physics_tick": frame.physics_tick, "sim_time_s": frame.sim_time_s,
            "observation_sha256": state_hash(observations["policy"]),
            "observation_dimension": int(observations["policy"].shape[-1]),
            "raw_deterministic_actor_mean_full12": list(raw), "nominal_action_full12": list(nominal),
            "previous_residual_full12": [0.0] * 12, "projection_dt_s": 1/120, **results,
            "new_minus_old": {field: [a-b for a, b in zip(results["new"][field], results["old"][field])]
                              for field in fields},
            "scope": "same-state logical output/projected target, not native dispatch or subsequent trajectory",
            "actual_environment_or_bridge_history_modified": False, "optimizer_updates": 0}


def build_stop_request(run_dir: Path, source_git_commit: str, reason: str) -> dict[str, Any]:
    if not reason.strip() or len(source_git_commit) != 40:
        raise ValueError("stop request requires its pinned HEAD and a reason")
    return {"schema": "wlr50_clean.semantic_stop_after_update.v1", "run_dir": str(run_dir.resolve()),
            "source_git_commit": source_git_commit, "reason": reason.strip()}


def _stop_request(run_dir: Path, contract: Mapping[str, Any]) -> dict[str, Any] | None:
    path = run_dir / "stop_after_update.request.json"
    if not path.exists():
        return None
    supplied = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = build_stop_request(run_dir, str(contract.get("source_git_commit", "")), supplied.get("reason", ""))
    if supplied != expected:
        raise ValueError("stop request does not match this run and pinned runtime")
    return {**expected, "request_path": str(path.resolve()), "request_sha256": sha256_file(path)}


@contextmanager
def _training_failure_ledger(run_dir: Path, updates: list[Any], checkpoints: list[Any]):
    """An incomplete rollout/update is evidence, never a resumable PPO state."""
    try:
        yield
    except BaseException as exc:
        write_json(run_dir / "training_failure.json", {
            "schema": "wlr50_clean.semantic_training_failure.v1", "lifecycle": "FAILED",
            "error_type": type(exc).__name__, "error": str(exc),
            "completed_updates_this_run": len(updates),
            "last_completed_update": updates[-1] if updates else None,
            "last_verified_checkpoint": checkpoints[-1] if checkpoints else None,
            "partial_rollout_or_update_is_not_resumable": True,
            "resume_requires_explicit_verified_checkpoint_and_legal_reset": True,
        })
        raise


def _publish_last(checkpoint: Path, sidecar: Path, output_root: Path) -> None:
    directory = output_root / "checkpoints"
    for source, name in ((checkpoint, "checkpoint_last.pt"), (sidecar, "resume_state.json")):
        target = directory / name
        temporary = target.with_name(target.name + ".pending")
        with source.open("rb") as incoming, temporary.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, target)
    # This pointer names the immutable pair and avoids pretending its sidecar
    # was originally bound to the mutable convenience copy checkpoint_last.
    write_json(directory / "checkpoint_last_pointer.json", {
        "checkpoint": str(checkpoint.resolve()), "manifest": str(sidecar.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint), "manifest_sha256": sha256_file(sidecar),
    }, replace=True)


def train_semantic(runner: Any, env: SemanticRslAdapter, *, run_dir: Path,
                   output_root: Path, stage: str, decisions: int,
                   contract: Mapping[str, Any], seed: int, resume_infos: Mapping[str, Any] | None = None,
                   checkpoint_interval_updates: int = 10) -> dict[str, Any]:
    import torch
    if stage not in STAGE_BUDGETS or type(decisions) is not int or decisions < 1:
        raise ValueError("invalid semantic training stage/decision request")
    if checkpoint_interval_updates < 1:
        raise ValueError("checkpoint cadence must be positive")
    previous = dict(resume_infos or {})
    sampling = jsonable(env.cfg.get("reset_sampling", "P01_only"))
    prefix_request = jsonable(env.cfg.get("prefix_request"))
    semantic_version = env.cfg.get("semantic_version", "v2")
    stage_spent = {name: int(previous.get("stage_requested_decisions", {}).get(name, 0)) for name in STAGE_BUDGETS}
    if stage_spent[stage] + decisions > STAGE_BUDGETS[stage]:
        raise ValueError("additional request exceeds the remaining semantic stage budget")
    base_global = int(previous.get("global_policy_decisions", 0))
    base_updates = int(previous.get("ppo_updates", 0))
    base_optimizer = int(previous.get("optimizer_steps", 0))
    batch = int(runner.cfg["num_steps_per_env"]) * env.num_envs
    iterations = (decisions + batch - 1) // batch
    run_dir.mkdir(parents=True, exist_ok=True)
    rollout_dir = run_dir / "rollouts"
    rollout_dir.mkdir(exist_ok=False)
    updates, checkpoints = [], []
    stop_record = None
    initial_actor = parameter_hash(runner.alg.actor)
    started = time.perf_counter()
    gpu_probe = getattr(env, "gpu_probe", None)
    runner.alg.train_mode()
    obs = env.get_observations().to(runner.device)
    with _training_failure_ledger(run_dir, updates, checkpoints), \
         (run_dir / "residual_and_projection_audit.jsonl").open("x", encoding="utf-8") as audit_stream, \
         (run_dir / "optimizer_updates.jsonl").open("x", encoding="utf-8") as update_stream, \
         (run_dir / "completed_episodes.jsonl").open("x", encoding="utf-8") as episode_stream:
        for iteration in range(iterations):
            with torch.inference_mode():
                for tick in range(int(runner.cfg["num_steps_per_env"])):
                    if (jsonable(env.cfg.get("reset_sampling", "P01_only")) != sampling
                            or jsonable(env.cfg.get("prefix_request")) != prefix_request):
                        raise RuntimeError("curriculum must remain fixed throughout this on-policy epoch")
                    raw = runner.alg.act(obs)
                    sampled_raw = raw.detach().clone()
                    old_log_prob = runner.alg.transition.actions_log_prob.detach().clone()
                    old_value = runner.alg.transition.values.detach().clone()
                    old_mean, old_std = (value.detach().clone()
                                        for value in runner.alg.transition.distribution_params)
                    if (old_mean.shape != sampled_raw.shape or old_std.shape != sampled_raw.shape
                            or not bool(torch.isfinite(old_mean).all())
                            or not bool(torch.isfinite(old_std).all()) or not bool((old_std > 0).all())):
                        raise RuntimeError("invalid sampled raw policy distribution before physics step")
                    obs, rewards, dones, extras = env.step(raw.to(env.device))
                    if any(not bool(torch.isfinite(value).all()) for value in (sampled_raw, old_log_prob, old_value, rewards)):
                        raise RuntimeError("non-finite on-policy transition")
                    if bool(extras["time_outs"].any()):
                        raise RuntimeError("task finite-horizon termination must not bootstrap")
                    for index, info in enumerate(extras["semantic_decisions"]):
                        row = {"global_policy_decision": base_global + iteration * batch + tick * env.num_envs + index + 1,
                               "raw_policy_action_full12": sampled_raw[index].cpu().tolist(),
                               "old_distribution_mean_full12": old_mean[index].cpu().tolist(),
                               "old_distribution_std_full12": old_std[index].cpu().tolist(),
                               "old_log_probability": float(old_log_prob[index]), "old_value": float(old_value[index]),
                               "reward": float(rewards[index]), "terminal": bool(dones[index]),
                               "applied_audit": info}
                        audit_stream.write(json.dumps(row, allow_nan=False) + "\n")
                    for episode in extras["episode_summaries"]:
                        episode_stream.write(json.dumps(episode, allow_nan=False) + "\n")
                    obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                    runner.alg.process_env_step(obs, rewards, dones, extras)
                    if not torch.equal(runner.alg.storage.actions[tick], sampled_raw):
                        raise RuntimeError("stored PPO action differs from sampled raw latent")
                    if any(not torch.equal(saved[tick], expected) for saved, expected in
                           zip(runner.alg.storage.distribution_params, (old_mean, old_std))):
                        raise RuntimeError("stored PPO distribution differs from sampled mean/std")
                runner.alg.compute_returns(obs)
            if gpu_probe is not None:
                gpu_probe.sample("after_complete_rollout")
            storage = runner.alg.storage
            snapshot = {key: getattr(storage, key).detach().cpu().clone() for key in (
                "actions", "actions_log_prob", "values", "rewards", "dones", "returns", "advantages")}
            snapshot["observations"] = {key: value.detach().cpu().clone() for key, value in storage.observations.items()}
            snapshot["distribution_params"] = tuple(value.detach().cpu().clone() for value in storage.distribution_params)
            snapshot["schema"] = "wlr50_clean.semantic_on_policy_rollout.v1"
            snapshot["runtime_contract"] = dict(contract)
            if storage.observations["policy"].shape[-1] == 324:
                snapshot["policy_contract"] = policy_contract(runner._semantic_policy_version)
            torch.save(snapshot, rollout_dir / f"rollout_{base_updates + iteration + 1:06d}.pt")
            global_step = base_global + (iteration + 1) * batch
            runner.alg.entropy_coef = 0.005 + (0.001 - 0.005) * min(global_step / sum(STAGE_BUDGETS.values()), 1.0)
            update = audited_ppo_update(runner)
            if gpu_probe is not None:
                gpu_probe.sample("after_official_optimizer_update")
            update.update(ppo_update=base_updates + iteration + 1, global_policy_decisions=global_step)
            updates.append(update)
            update_stream.write(json.dumps(update, allow_nan=False) + "\n")
            update_stream.flush()
            audit_stream.flush()
            episode_stream.flush()
            runner.current_learning_iteration = base_updates + iteration + 1
            print(json.dumps({"semantic_ppo_update": update}, separators=(",", ":")), flush=True)
            stop_record = _stop_request(run_dir, contract)
            if stop_record is not None or iteration == 0 or (iteration + 1) % checkpoint_interval_updates == 0 or iteration + 1 == iterations:
                spent = dict(stage_spent)
                spent[stage] += min((iteration + 1) * batch, decisions)
                from .semantic_migration import topology
                infos = {"runtime_contract": dict(contract), "seed": seed, "stage": stage,
                         "execution_topology": topology(env.num_envs),
                         "phase_suffix_curriculum_implemented": prefix_request is not None,
                         "semantic_version": semantic_version,
                         "curriculum_epoch": {"reset_sampling": sampling, "prefix_request": prefix_request,
                                              "changes_allowed_only_between_complete_rollout_updates": True},
                         "implemented_reset_sampling": env.cfg.get("reset_sampling", "P01_only"),
                         "vector_smoke_evidence": env.cfg.get("vector_smoke_evidence"),
                         "global_policy_decisions": global_step, "ppo_updates": base_updates + iteration + 1,
                         "optimizer_steps": base_optimizer + sum(row["optimizer_steps"] for row in updates),
                         "stage_requested_decisions": spent, "source_run": str(run_dir.resolve()),
                         "sampling": sampling, "runner_config": copy.deepcopy(runner._semantic_runner_config),
                         "last_update": update}
                if semantic_version == "v3":
                    from .semantic_migration import continuation_topology
                    infos["execution_topology"] = continuation_topology(sampling, prefix_request)
                for key in ("new_mdp_warm_start", "new_mdp_origin_global_policy_decisions", "source_stage_requested_decisions",
                            "new_mdp_initial_action_comparison", "policy_distribution_migration",
                            "policy_distribution_migration_evidence"):
                    if key in previous:
                        infos[key] = previous[key]
                if previous:
                    infos["resume_ancestry"] = {
                        "source_global_policy_decisions": base_global, "source_ppo_updates": base_updates,
                        "source_optimizer_steps": base_optimizer,
                        "source_actor_parameter_sha256": previous["actor_parameter_sha256"],
                        "source_runtime_contract": previous["runtime_contract"],
                        "source_checkpoint": previous.get("resume_source_checkpoint"),
                        "resume_migration": previous.get("resume_migration"),
                    }
                if stop_record is not None:
                    infos["stop_after_update"] = stop_record
                checkpoint = output_root / "checkpoints" / "history" / f"checkpoint_step_{global_step:09d}.pt"
                pair = save_semantic_checkpoint(runner, checkpoint, infos)
                _publish_last(*pair, output_root)
                checkpoints.append({"checkpoint": str(pair[0]), "manifest": str(pair[1]), "global_policy_decisions": global_step})
            if stop_record is not None:
                break
    completed = len(updates)
    consumed = min(completed * batch, decisions)
    result = {"schema": SEMANTIC_TRAINING_SCHEMA, "stage": stage, "planned_requested_policy_decisions": decisions,
              "requested_policy_decisions": consumed, "unconsumed_requested_policy_decisions": decisions - consumed,
              "actual_policy_decisions": completed * batch, "rounding_overrun": completed * batch - consumed,
              "num_envs": env.num_envs, "policy_decisions_per_env": completed * batch // env.num_envs,
              "global_policy_decisions": base_global + completed * batch,
              "ppo_updates_this_run": len(updates), "optimizer_steps_this_run": sum(row["optimizer_steps"] for row in updates),
              "actor_parameter_sha256_before": initial_actor, "actor_parameter_sha256_after": parameter_hash(runner.alg.actor),
              "finite_nonzero_gradient_observed": any(row["finite_nonzero_gradient_observed"] for row in updates),
              "wall_time_s": time.perf_counter() - started, "telemetry": env.telemetry_summary(),
              "runtime_contract": dict(contract), "checkpoints": checkpoints, "training_success_is_not_task_success": True}
    result["lifecycle"] = "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY" if stop_record is not None else "SUCCEEDED"
    result["stop_after_update"] = stop_record
    result["phase_suffix_curriculum_implemented"] = prefix_request is not None
    result["semantic_version"] = semantic_version
    result["runner_config"] = copy.deepcopy(runner._semantic_runner_config)
    if runner.alg.storage.observations["policy"].shape[-1] == 324:
        result["policy_contract"] = policy_contract(runner._semantic_policy_version)
    result["curriculum_epoch"] = {"reset_sampling": sampling, "prefix_request": prefix_request}
    result["implemented_sampling"] = env.cfg.get("reset_sampling", "P01_only")
    if gpu_probe is not None:
        gpu_probe.sample("after_training_and_verified_checkpoint")
        result["gpu_measurements"] = gpu_probe.summary()
    write_json(run_dir / "training_manifest.json", result)
    return result

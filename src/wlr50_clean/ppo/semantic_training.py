"""Audited semantic PPO using the installed, unchanged official RSL-RL optimizer.

This is orchestration and instrumentation, not another PPO implementation.
The policy distribution is over raw Gaussian latent actions; projection never
replaces those samples or their old log probabilities in RSL storage.
"""
from __future__ import annotations
from .semantic_p02_progress_profile import P02_PROGRESS_POLICY, P02_PROGRESS_OBSERVATION_LAYOUT
from .semantic_rear_owner_profile import REAR_OWNER_POLICY, REAR_OWNER_OBSERVATION_LAYOUT
from .semantic_rear_cooperative_prep_profile import (
    COOPERATIVE_PREP_POLICY, COOPERATIVE_PREP_SIGMA_SEMANTICS)

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
from .semantic_receiving_wheel_profile import RECEIVING_WHEEL_POLICY, RECEIVING_WHEEL_SIGMA_SEMANTICS
from .semantic_p05_capture_profile import (P05_CAPTURE_POLICY, P05_CAPTURE_OBSERVATION_LAYOUT,
    P05_CAPTURE_OBSERVATION_DIM, P05_CAPTURE_HISTORY_SEMANTICS)
from .semantic_rr_capture_profile import (RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_LAYOUT,
    RR_CAPTURE_OBSERVATION_DIM, RR_ASSIST_START, RR_TASK_START)
from .semantic_rear_policy_timing_profile import (REAR_POLICY_TIMING_POLICY,
    REAR_POLICY_TIMING_OBSERVATION_LAYOUT, REAR_POLICY_TIMING_OBSERVATION_DIM,
    REAR_POLICY_TIMING_SIGMA_SEMANTICS)
from .semantic_return_profile import (
    RETURN_PROFILE, RUNNER_PROFILE_KEY, profile_parameters,
    reward_return_profile, runner_return_profile,
)

SEMANTIC_TRAINING_SCHEMA = "wlr50_clean.semantic_training.v1"
SEMANTIC_CHECKPOINT_SCHEMA = "wlr50_clean.semantic_checkpoint.v1"
STAGE_BUDGETS = {"smoke": 10_000, "phase_suffix": 100_000, "full_episode": 100_000}
ROLLOUT_LENGTH = 128


def training_quantity_budgets(experiment_id: str | None = None) -> dict[str, int]:
    """Explicit quantity ceiling; historical entropy horizon stays 210000."""
    budgets = dict(STAGE_BUDGETS)
    if experiment_id in ("task_conditioned_hip_wheel_v1","p05_hip_only_continuation_v1","rr_capture_then_rl_transfer_v1", "rr_rl_timing_policy_learning_v1"):
        budgets["full_episode"] = 131_072
    if experiment_id == "rr_rl_timing_policy_learning_v1":
        # Quantity authorization for further rear-heavy collection only. Keep
        # all earned counts and the historical entropy/learning-rate schedule.
        budgets["phase_suffix"] = 131_072
    return budgets


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
                           policy_version: str = LEGACY_POLICY,
                           return_profile: str | None = None,
                           observation_layout: str | None = None) -> dict[str, Any]:
    if semantic_version not in ("v2", "v3"):
        raise ValueError("unsupported semantic runtime version")
    from .semantic_policy_distribution import (HISTORY_POLICY, HISTORY_TEMPERED_POLICY,
        HISTORY_QUARTER_TEMPERED_POLICY, HISTORY_REQUEST_CAP_TRANSITION_POLICY,
        FR_KNEE_PHYSICAL_INNOVATION_POLICY, TASK_CONDITIONED_HIP_WHEEL_POLICY)
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    if policy_version in (HISTORY_POLICY, HISTORY_TEMPERED_POLICY, HISTORY_QUARTER_TEMPERED_POLICY,
                          HISTORY_REQUEST_CAP_TRANSITION_POLICY, FR_KNEE_PHYSICAL_INNOVATION_POLICY,
                          TASK_CONDITIONED_HIP_WHEEL_POLICY, RECEIVING_WHEEL_POLICY, P05_CAPTURE_POLICY, RR_CAPTURE_POLICY, REAR_POLICY_TIMING_POLICY, P02_PROGRESS_POLICY, COOPERATIVE_PREP_POLICY, REAR_OWNER_POLICY) and semantic_version != "v3":
        raise ValueError("history-conditioned policy requires the v3 semantic runtime")
    if policy_version in (HISTORY_TEMPERED_POLICY, HISTORY_QUARTER_TEMPERED_POLICY,
                          HISTORY_REQUEST_CAP_TRANSITION_POLICY, FR_KNEE_PHYSICAL_INNOVATION_POLICY,
                          TASK_CONDITIONED_HIP_WHEEL_POLICY, RECEIVING_WHEEL_POLICY) and observation_layout != ROLE_OBSERVATION_LAYOUT:
        raise ValueError("tempered history policy requires the explicit role372 observation layout")
    if policy_version == RR_CAPTURE_POLICY and observation_layout != RR_CAPTURE_OBSERVATION_LAYOUT:
        raise ValueError("RR capture actor requires its explicit appended-state layout")
    if policy_version in (P02_PROGRESS_POLICY, COOPERATIVE_PREP_POLICY) and observation_layout != P02_PROGRESS_OBSERVATION_LAYOUT:
        raise ValueError("P02 progress actor requires its explicit422 layout")
    if policy_version == REAR_OWNER_POLICY and observation_layout != REAR_OWNER_OBSERVATION_LAYOUT:
        raise ValueError("rear owner actor requires its explicit439 layout")
    if policy_version == REAR_POLICY_TIMING_POLICY and observation_layout != REAR_POLICY_TIMING_OBSERVATION_LAYOUT:
        raise ValueError("rear timing actor requires its explicit 419 layout")
    if policy_version == P05_CAPTURE_POLICY and observation_layout != P05_CAPTURE_OBSERVATION_LAYOUT:
        raise ValueError("P05 capture actor requires the explicit observable 389 layout")
    if return_profile is None:
        from .semantic_reward import load_semantic_reward_config
        path = Path(__file__).resolve().parents[3] / "configs" / f"ppo_semantic_{semantic_version}" / "reward_config.yaml"
        horizon = reward_return_profile(load_semantic_reward_config(path).values,
                                        semantic_version=semantic_version)
    else:
        # Explicit historical reconstruction for full source metadata validation;
        # the production runner constructor always resolves the current config.
        horizon = profile_parameters(return_profile, semantic_version=semantic_version)
    profile = SimpleNamespace(
        activation="elu", entropy_start=0.005, actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256), initial_action_std=0.15,
        rollout_length=ROLLOUT_LENGTH, update_epochs=5, num_minibatches=4,
        clip_ratio=0.20, gamma=horizon["gamma"], lam=horizon["lambda"], value_loss_coefficient=1.0,
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
    if observation_layout is None:
        configure_policy_distribution(config, policy_version)
    else:
        if semantic_version != "v3":
            raise ValueError("appended transfer observations require v3")
        configure_policy_distribution(config, policy_version, observation_layout=observation_layout)
    if horizon["version"] == RETURN_PROFILE:
        config[RUNNER_PROFILE_KEY] = RETURN_PROFILE
    return config


def assert_semantic_return_consistency(runner: Any, env: Any) -> dict[str, Any]:
    """Fail before credit/update if live PPO and the actual PBRS source disagree.

    Small algorithm-only CPU cores do not implement a physical reward calculator;
    they receive runner-only validation, never a claimed PBRS equality proof.
    """
    horizon = runner_return_profile(runner._semantic_runner_config,
                                     semantic_version=runner._semantic_version)
    live = runner_return_profile(runner.cfg, semantic_version=runner._semantic_version)
    if (live != horizon or isinstance(runner.alg.gamma, bool) or isinstance(runner.alg.lam, bool)
            or runner.alg.gamma != horizon["gamma"] or runner.alg.lam != horizon["lambda"]):
        raise RuntimeError("actual PPO gamma/lambda differs from the pinned return configuration")
    calculator = getattr(getattr(env, "core", None), "reward_calculator", None)
    if calculator is not None:
        reward_horizon = reward_return_profile(calculator.config.values,
                                               semantic_version=runner._semantic_version)
        if reward_horizon != horizon or calculator.config.gamma != runner.alg.gamma:
            raise RuntimeError("PPO and actual semantic PBRS return profiles must agree")
    elif hasattr(env, "gamma") and env.gamma != runner.alg.gamma:
        raise RuntimeError("vector reward and PPO gamma must agree")
    return horizon


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
        self._terminal_evidence_writer = None
        self._defer_terminal_reset = False
        self._pending_episode_reset = False
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
        if self._pending_episode_reset:
            raise RuntimeError("pending terminal reset must complete before the next policy action")
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
        reward_tensor = torch.tensor([reward], device=self.device)
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
            # A teacher-prefix reset may take minutes or fail. The collector's
            # sampled-action evidence must be durable before starting that reset,
            # The true final observation stays bound to this sampled transition;
            # the collector may defer reset only at a complete rollout tail.
            if self._terminal_evidence_writer is not None:
                self._terminal_evidence_writer(summary, terminal_observation, reward_tensor)
                extras["terminal_evidence_persisted_before_reset"] = True
            if self._defer_terminal_reset:
                # Only the collector's final rollout tick may request this.
                # Keep the real terminal observation readable, but do not run
                # an uncredited reset/prefix unless another rollout is needed.
                self._pending_episode_reset = True
                extras["terminal_reset_deferred"] = True
            else:
                self._observation = tuple(self.core.reset(seed=self.seed))
                self.episode_length_buf.zero_()
        return (self.get_observations(), reward_tensor,
                torch.tensor([done], dtype=torch.bool, device=self.device), extras)

    @property
    def episode_reset_pending(self) -> bool:
        return self._pending_episode_reset

    def reset_pending_episode(self) -> Any:
        """Explicit next-rollout seam; get_observations never resets physics."""
        if not self._pending_episode_reset:
            return self.get_observations()
        self._observation = tuple(self.core.reset(seed=self.seed))
        observations = self.get_observations()
        self.episode_length_buf.zero_()
        self._pending_episode_reset = False
        return observations

    def telemetry_summary(self) -> dict[str, Any]:
        return {"policy_decisions": self.total_decisions,
                "episode_reset_pending": self._pending_episode_reset,
                "completed_episode_count": len(self.completed_episodes),
                "success_count": sum(row["task_success"] for row in self.completed_episodes),
                "core": jsonable(self.core.telemetry_summary()),
                "reset_sampling": "P01_full_task_only_initial_version"}


def construct_semantic_runner(env: Any, *, seed: int, device: str,
                              policy_version: str = LEGACY_POLICY,
                              initialize_actor: bool = True,
                              observation_layout: str | None = None) -> tuple[Any, dict[str, Any]]:
    assert_supported_rsl_runtime()
    config = semantic_runner_config(seed=seed, device=device,
                                    semantic_version=env.cfg.get("semantic_version", "v2"),
                                    policy_version=policy_version, observation_layout=observation_layout)
    runner = construct_runner(env, config, log_dir=None)
    runner.logger.writer = None  # No hidden upstream automatic checkpoint writes.
    runner._semantic_policy_version = policy_version
    runner._semantic_observation_layout = observation_layout
    runner._semantic_version = env.cfg.get("semantic_version", "v2")
    runner._semantic_runner_config = copy.deepcopy(config)
    assert_semantic_return_consistency(runner, env)
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


def audited_ppo_update(runner: Any, *, likelihood_audit_path: Path | None = None) -> dict[str, Any]:
    """Observe official minibatches and gradients without replacing PPO math."""
    import torch
    from .semantic_policy_distribution import (HISTORY_REQUEST_CAP_TRANSITION_POLICY,
        FR_KNEE_PHYSICAL_INNOVATION_POLICY, TASK_CONDITIONED_HIP_WHEEL_POLICY)
    alg = runner.alg
    before = parameter_hash(alg.actor)
    generator = alg.storage.mini_batch_generator
    log_prob = alg.actor.get_output_log_prob
    kl_method = alg.actor.get_kl_divergence
    active: dict[str, Any] = {}
    gradients, clips, kls = [], [], []
    likelihood_rows = []
    sample_lookup = {}
    task_head_audit = (likelihood_audit_path is not None and
        getattr(runner, "_semantic_policy_version", None) in (TASK_CONDITIONED_HIP_WHEEL_POLICY, RECEIVING_WHEEL_POLICY, P05_CAPTURE_POLICY, RR_CAPTURE_POLICY, REAR_POLICY_TIMING_POLICY, P02_PROGRESS_POLICY, COOPERATIVE_PREP_POLICY, REAR_OWNER_POLICY))
    if likelihood_audit_path is not None:
        # Index immutable saved observations/raw samples, never the shuffled
        # neighbor or global history. No forward pass or RNG draw is added.
        obs_rows = alg.storage.observations["policy"].flatten(0, 1).detach().cpu()
        action_rows = alg.storage.actions.flatten(0, 1).detach().cpu()
        for index, (observation, action) in enumerate(zip(obs_rows, action_rows)):
            key = (observation.contiguous().numpy().tobytes(), action.contiguous().numpy().tobytes())
            sample_lookup.setdefault(key, []).append(index)

    def batches(*args: Any, **kwargs: Any):
        for batch in generator(*args, **kwargs):
            active["batch"] = batch
            if task_head_audit:
                active.pop("head", None)
                active.pop("head_gradient", None)
            yield batch

    def observed_head_gradient(gradient):
        # Return None: observe autograd's real derivative without replacing it.
        active["head_gradient"] = gradient.detach().clone()

    def observed_actor_head(_module, _inputs, output):
        if (output.shape[-2:] != (2, 12) or not output.requires_grad
                or "head" in active):
            raise RuntimeError("task head audit requires exactly one official differentiable minibatch head")
        active["head"] = output.detach().clone()
        output.register_hook(observed_head_gradient)

    def observed_log_prob(raw: Any):
        result = log_prob(raw)
        batch = active["batch"]
        if not torch.equal(raw, batch.actions):
            raise RuntimeError("PPO likelihood did not use stored raw actions")
        with torch.no_grad():
            ratio = torch.exp(result - batch.old_actions_log_prob.squeeze(-1))
            clips.append(float(((ratio - 1).abs() > alg.clip_param).float().mean()))
            if likelihood_audit_path is not None:
                indices = []
                for observation, action in zip(batch.observations["policy"].detach().cpu(), raw.detach().cpu()):
                    key = (observation.contiguous().numpy().tobytes(), action.contiguous().numpy().tobytes())
                    if key not in sample_lookup:
                        raise RuntimeError("minibatch observation/raw sample differs from saved rollout")
                    indices.append(sample_lookup[key])
                advantage = batch.advantages.squeeze(-1)
                unclipped = -advantage * ratio
                clipped = -advantage * ratio.clamp(1.0-alg.clip_param, 1.0+alg.clip_param)
                likelihood_rows.append({
                    "minibatch_index": len(likelihood_rows),
                    "rollout_flat_indices": indices,
                    "old_log_probability": jsonable(batch.old_actions_log_prob.squeeze(-1)),
                    "optimization_log_probability": jsonable(result),
                    "ratio": jsonable(ratio), "actual_advantage": jsonable(advantage),
                    "clipped_branch_strictly_active": jsonable(clipped > unclipped),
                    "current_conditional_mean": jsonable(alg.actor.output_distribution_params[0]),
                    "current_conditional_sigma": jsonable(alg.actor.output_distribution_params[1]),
                    "sigma_source": ("current_official_Gaussian_cache_after_shared_observed_cooperative_prep_TOTAL_rear_sigma"
                        if getattr(runner, "_semantic_policy_version", None) in (COOPERATIVE_PREP_POLICY, REAR_OWNER_POLICY)
                        else "current_official_Gaussian_cache_after_parent_receiving_and_observed_rear_local_sigma"
                        if getattr(runner, "_semantic_policy_version", None) in (REAR_POLICY_TIMING_POLICY, P02_PROGRESS_POLICY)
                        else "current_official_Gaussian_cache_after_B_over_cap_and_receiving_FR_RR_sigma_x3"
                        if getattr(runner, "_semantic_policy_version", None) in (RECEIVING_WHEEL_POLICY,P05_CAPTURE_POLICY,RR_CAPTURE_POLICY, REAR_POLICY_TIMING_POLICY, P02_PROGRESS_POLICY)
                        else "current_official_Gaussian_cache_after_current_observation_B_over_cap"
                        if getattr(runner, "_semantic_policy_version", None) == TASK_CONDITIONED_HIP_WHEEL_POLICY
                        else "current_official_Gaussian_cache_after_P06plus_FR_knee_24_over_112"
                        if getattr(runner, "_semantic_policy_version", None) == FR_KNEE_PHYSICAL_INNOVATION_POLICY
                        else "current_official_Gaussian_cache"),
                    "history_source": ("this_saved_observation_legacy372_plus_pending384_advanced386_no_fake_completion"
                        if getattr(runner,"_semantic_policy_version",None) in (P05_CAPTURE_POLICY,RR_CAPTURE_POLICY, REAR_POLICY_TIMING_POLICY, P02_PROGRESS_POLICY, COOPERATIVE_PREP_POLICY, REAR_OWNER_POLICY)
                        else "this_saved_observation_stage0_13_age20_completed158_171_raw195_207_request207_219_not_shuffled_neighbor"
                        if getattr(runner, "_semantic_policy_version", None) in
                        (HISTORY_REQUEST_CAP_TRANSITION_POLICY, FR_KNEE_PHYSICAL_INNOVATION_POLICY,
                         TASK_CONDITIONED_HIP_WHEEL_POLICY, RECEIVING_WHEEL_POLICY)
                        else "this_saved_observation_195_207_not_shuffled_neighbor"),
                })
                if task_head_audit:
                    head = active["head"]
                    likelihood_rows[-1].update(
                        current_network_mean_full12=jsonable(head[..., 0, :]),
                        current_network_log_sigma_full12=jsonable(head[..., 1, :]),
                        head_measurement="same_official_minibatch_forward_before_HISTORY_and_sigma_schedule")
                    if getattr(runner, "_semantic_policy_version", None) in (COOPERATIVE_PREP_POLICY, REAR_OWNER_POLICY):
                        from .semantic_rear_cooperative_prep_sigma import cooperative_prep_effective_log_std
                        checked_log_std, checked = cooperative_prep_effective_log_std(
                            head[..., 1, :], batch.observations["policy"][..., :422],
                            alg.actor.exploration_std_temperature)
                        if not torch.equal(checked_log_std.exp(), alg.actor.output_distribution_params[1]):
                            raise RuntimeError("cooperative current likelihood differs from the shared observed kernel")
                        likelihood_rows[-1].update(
                            cooperative_observed_rr_carry_capture=jsonable(checked['cooperative_observed_rr_carry_capture']),
                            cooperative_observed_rr_top_reachable=jsonable(checked['cooperative_observed_rr_top_reachable']),
                            cooperative_observed_rl_prep_transfer=jsonable(checked['cooperative_observed_rl_prep_transfer']),
                            cooperative_parent_receiving_continuation_active=jsonable(checked['cooperative_parent_receiving_continuation_active']),
                            cooperative_prep_allowed=jsonable(checked['cooperative_prep_allowed']),
                            cooperative_fl_wheel_extra_active=jsonable(checked['cooperative_fl_wheel_extra_active']),
                            rear_local_sigma_multiplier_full12=jsonable(checked['rear_local_sigma_multiplier_full12']),
                            cooperative_multiplier_reference=checked['cooperative_multiplier_reference'])
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
        if task_head_audit:
            derivative = active.get("head_gradient")
            if (derivative is None or not bool(torch.isfinite(derivative).all())
                    or len(likelihood_rows) != len(gradients)):
                raise RuntimeError("task head derivative does not match its official likelihood/optimizer step")
            head_layer = [m for m in alg.actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
            row_norm = (head_layer.weight.grad.detach().double().square().sum(-1)
                        + head_layer.bias.grad.detach().double().square()).sqrt()
            separate_norms = {}
            for role in ("actor", "critic"):
                separate_norms[role] = math.sqrt(sum(float(p.grad.detach().double().square().sum())
                    for p in getattr(alg, role).parameters() if p.grad is not None))
            likelihood_rows[-1].update(
                loss_gradient_wrt_network_mean_full12=jsonable(derivative[..., 0, :]),
                loss_gradient_wrt_network_log_sigma_full12=jsonable(derivative[..., 1, :]),
                actor_head_postclip_parameter_gradient_norm_by_row=jsonable(row_norm),
                separate_postclip_parameter_gradient_norms=separate_norms,
                gradient_semantics="actual_total_official_loss_output_derivative_before_parameter_clipping;parameter_norms_after_separate_actor_critic_clipping;not_an_isolated_sample_update_or_Adam_parameter_delta")

    handle = alg.optimizer.register_step_pre_hook(optimizer_pre_step)
    head_handle = alg.actor.mlp.register_forward_hook(observed_actor_head) if task_head_audit else None
    try:
        alg.storage.mini_batch_generator = batches
        alg.actor.get_output_log_prob = observed_log_prob
        alg.actor.get_kl_divergence = observed_kl
        loss = alg.update()
    finally:
        handle.remove()
        if head_handle is not None:
            head_handle.remove()
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
    if likelihood_audit_path is not None:
        write_json(likelihood_audit_path, {
            "schema": "wlr50_clean.task_recovery_minibatch_likelihood.v1",
            "source": "actual_official_PPO_log_prob_call_before_each_optimizer_step",
            "extra_model_forwards": 0, "extra_random_draws": 0,
            "sample_index_basis": "time_major_flattened_saved_rollout",
            "ambiguous_identical_observation_and_action_indices_retained": True,
            "clip_param": alg.clip_param, "minibatches": likelihood_rows})
    return {"optimizer_steps": len(gradients), "actor_parameter_sha256_before": before,
            "actor_parameter_sha256_after": after, "actor_parameters_changed": before != after,
            "finite_nonzero_gradient_observed": any(value > 0 for value in gradients),
            "gradient_norm_min": min(gradients), "gradient_norm_max": max(gradients),
            "kl_mean": sum(kls) / len(kls), "clip_fraction": sum(clips) / len(clips),
            "entropy": float(loss["entropy"]), "value_loss": float(loss["value"]),
            "surrogate_loss": float(loss["surrogate"]), "optimizer_learning_rate": optimizer_learning_rate(runner)}


def _normalizers(runner: Any) -> dict[str, Any]:
    return {role: getattr(runner.alg, role).obs_normalizer.state_dict() for role in ("actor", "critic")}


def _runner_policy_contract(runner: Any) -> dict[str, Any]:
    """Bind the explicit layout, not merely an arbitrary observed tensor width."""
    result = policy_contract(runner._semantic_policy_version,
        observation_layout=getattr(runner, "_semantic_observation_layout", None))
    if runner.alg.storage.observations["policy"].shape[-1] != result["observation_dimension"]:
        raise RuntimeError("runner observation width differs from its explicit policy layout")
    return result


def save_semantic_checkpoint(runner: Any, checkpoint: Path, infos: Mapping[str, Any]) -> tuple[Path, Path]:
    """Publish immutable official state, then prove a real load restores it."""
    if "rear_policy_timing_branch" in infos:
        from .semantic_rear_policy_timing_migration import rear_policy_timing_branch_counts
        infos = rear_policy_timing_branch_counts(infos)
    if "capture_feedback_semantics_branch" in infos:
        from .semantic_capture_feedback_migration import capture_feedback_branch_counts
        infos = capture_feedback_branch_counts(infos)
    if "rr_postcross_workspace_branch" in infos:
        from .semantic_rr_workspace_migration import rr_workspace_branch_counts
        infos = rr_workspace_branch_counts(infos)
    if "p05_preedge_approach_recovery_branch" in infos:
        from .semantic_p05_preedge_migration import p05_preedge_branch_counts
        infos = p05_preedge_branch_counts(infos)
    if "rr_capture_transfer_branch" in infos:
        from .semantic_rr_capture_migration import rr_capture_branch_counts
        infos = rr_capture_branch_counts(infos)
    assert_semantic_return_consistency(runner, runner.env)
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
    if runner.alg.storage.observations["policy"].shape[-1] in (324, 372, P05_CAPTURE_OBSERVATION_DIM, RR_CAPTURE_OBSERVATION_DIM, REAR_POLICY_TIMING_OBSERVATION_DIM, 422, 439):
        metadata["policy_contract"] = _runner_policy_contract(runner)
    elif runner._semantic_policy_version != LEGACY_POLICY:
        raise RuntimeError("state-dependent checkpoint has an unsupported observation layout")
    if "front_retention439_runtime_identity" in metadata or "front_retention439_auxiliary" in metadata:
        from .semantic_front_retention439 import validate_front_retention439_lineage
        route = metadata.get("checkpoint_output_routing", {})
        validate_front_retention439_lineage(metadata, metadata["runtime_contract"], Path(route.get("output_root", "")),
            checkpoint_output_routing=route)
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


def _validated_exploration_temperature_factor(metadata: Mapping[str, Any], record: Mapping[str, Any], *,
        semantic_version: str, seed: int, device: str, observation_layout: str | None) -> Mapping[str, Any] | None:
    """Validate only the reviewed same372 distribution boundary, after plan revalidation."""
    factor = record.get("exploration_temperature_factor")
    if factor is None:
        return None
    from .semantic_migration import source_num_envs
    from .semantic_policy_distribution import (HISTORY_POLICY, HISTORY_TEMPERED_POLICY,
        HISTORY_QUARTER_TEMPERED_POLICY, policy_version_from_metadata)
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    exclusive = ("execution_factor", "instrumentation_observation_contract", "video_instrumentation_factor",
        "nominal_timing_factor", "body_reward_factor", "task_first_reward_factor", "height_recovery_factor",
        "execution_composition_factor", "final_stop_handoff_factor", "rr_physical_acceptance_same372_factor",
        "fl_capture_quality_same372_factor")
    if any(record.get(key) is not None for key in exclusive):
        raise RuntimeError("exploration temperature migration cannot mix other migration factors")
    if (not isinstance(factor, Mapping) or semantic_version != "v3"
            or metadata.get("semantic_version") != "v3" or seed != metadata.get("seed")
            or observation_layout != ROLE_OBSERVATION_LAYOUT or source_num_envs(metadata) != 1
            or metadata.get("runner_config", {}).get("device") != device):
        raise RuntimeError("exploration temperature migration requires the exact v3 role372 N1 source")
    source_version = policy_version_from_metadata(metadata)
    if source_version == HISTORY_POLICY:
        target_version = HISTORY_TEMPERED_POLICY
    elif (source_version == HISTORY_TEMPERED_POLICY
            and metadata.get("runtime_contract", {}).get("experiment_id") == "task_first_recovery_v1"):
        target_version = HISTORY_QUARTER_TEMPERED_POLICY
    else:
        raise RuntimeError("exploration temperature migration has no reviewed source/target version pair")
    source_policy = policy_contract(source_version, observation_layout=observation_layout)
    target_policy = policy_contract(target_version, observation_layout=observation_layout)
    observation = {"source_policy_contract": source_policy, "target_policy_contract": target_policy,
        "observation_layout": observation_layout, "observation_dimension": 372, "action_dimension": 12,
        "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}
    if (metadata.get("policy_contract") != source_policy
            or factor.get("source_policy_contract") != source_policy
            or factor.get("target_policy_contract") != target_policy
            or factor.get("source_policy_version") != source_policy["version"]
            or factor.get("target_policy_version") != target_policy["version"]
            or factor.get("observation_contract") != observation):
        raise RuntimeError("exploration temperature migration lacks exact source/target policy and observation contracts")
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")["version"]
    configs = {version: semantic_runner_config(seed=seed, device=device, semantic_version="v3",
        policy_version=version, observation_layout=observation_layout, return_profile=horizon)
        for version in (source_version, target_version)}
    if (metadata["runner_config"] != configs[source_version]
            or factor.get("source_runner_config") != configs[source_version]
            or factor.get("target_runner_config") != configs[target_version]):
        raise RuntimeError("exploration temperature migration runner configuration differs from its exact contracts")
    return factor


def _validated_request_history_kernel_factor(metadata: Mapping[str, Any], record: Mapping[str, Any], *,
        semantic_version: str, seed: int, device: str, observation_layout: str | None) -> Mapping[str, Any] | None:
    """Validate the exclusive same372 mean-kernel boundary, not temperature."""
    factor = record.get("request_history_kernel_factor")
    if factor is None:
        return None
    from .semantic_migration import source_num_envs
    from .semantic_policy_distribution import (HISTORY_QUARTER_TEMPERED_POLICY,
        HISTORY_REQUEST_CAP_TRANSITION_POLICY, REQUEST_HISTORY_SEMANTICS, policy_version_from_metadata)
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    exclusive = ("execution_factor", "instrumentation_observation_contract", "video_instrumentation_factor",
        "nominal_timing_factor", "body_reward_factor", "task_first_reward_factor", "height_recovery_factor",
        "execution_composition_factor", "final_stop_handoff_factor", "rr_physical_acceptance_same372_factor",
        "fl_capture_quality_same372_factor", "exploration_temperature_factor")
    if any(record.get(key) is not None for key in exclusive):
        raise RuntimeError("request-history kernel migration cannot mix other migration factors")
    if (not isinstance(factor, Mapping) or semantic_version != "v3"
            or metadata.get("semantic_version") != "v3" or seed != metadata.get("seed")
            or observation_layout != ROLE_OBSERVATION_LAYOUT or source_num_envs(metadata) != 1
            or metadata.get("runner_config", {}).get("device") != device
            or metadata.get("runtime_contract", {}).get("experiment_id") != "fl_capture_quality_v1"
            or policy_version_from_metadata(metadata) != HISTORY_QUARTER_TEMPERED_POLICY
            or factor.get("schema") != "wlr50_clean.cap_transition_request_history_same372.v1"
            or factor.get("history_center_semantics") != REQUEST_HISTORY_SEMANTICS
            or metadata.get("optimizer_learning_rate") != 1e-5):
        raise RuntimeError("request-history migration requires exact old quarter role372 N1 FL source and LR1e-5")
    source_version, target_version = HISTORY_QUARTER_TEMPERED_POLICY, HISTORY_REQUEST_CAP_TRANSITION_POLICY
    source_policy = policy_contract(source_version, observation_layout=observation_layout)
    target_policy = policy_contract(target_version, observation_layout=observation_layout)
    observation = {"source_policy_contract": source_policy, "target_policy_contract": target_policy,
        "observation_layout": observation_layout, "observation_dimension": 372, "action_dimension": 12,
        "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}
    if (metadata.get("policy_contract") != source_policy
            or factor.get("source_policy_contract") != source_policy
            or factor.get("target_policy_contract") != target_policy
            or factor.get("source_policy_version") != source_version
            or factor.get("target_policy_version") != target_version
            or factor.get("observation_contract") != observation):
        raise RuntimeError("request-history migration lacks exact source/target policy and observation contracts")
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")["version"]
    configs = {version: semantic_runner_config(seed=seed, device=device, semantic_version="v3",
        policy_version=version, observation_layout=observation_layout, return_profile=horizon)
        for version in (source_version, target_version)}
    if (metadata["runner_config"] != configs[source_version]
            or factor.get("source_runner_config") != configs[source_version]
            or factor.get("target_runner_config") != configs[target_version]):
        raise RuntimeError("request-history migration runner differs from its exact contracts")
    return factor


def _validated_physical_innovation_sigma_factor(metadata: Mapping[str, Any], record: Mapping[str, Any], *,
        semantic_version: str, seed: int, device: str, observation_layout: str | None) -> Mapping[str, Any] | None:
    """Only the reviewed same-shape FR-knee sigma boundary; preserve source LR."""
    factor = record.get("physical_innovation_sigma_factor")
    if factor is None:
        return None
    from .semantic_migration import source_num_envs
    from .semantic_policy_distribution import (HISTORY_REQUEST_CAP_TRANSITION_POLICY,
        FR_KNEE_PHYSICAL_INNOVATION_POLICY, FR_KNEE_PHYSICAL_SIGMA_SEMANTICS, policy_version_from_metadata)
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    exclusive = ("execution_factor", "instrumentation_observation_contract", "video_instrumentation_factor",
        "nominal_timing_factor", "body_reward_factor", "task_first_reward_factor", "height_recovery_factor",
        "execution_composition_factor", "final_stop_handoff_factor", "rr_physical_acceptance_same372_factor",
        "fl_capture_quality_same372_factor", "exploration_temperature_factor", "request_history_kernel_factor")
    if any(record.get(key) is not None for key in exclusive):
        raise RuntimeError("physical innovation sigma migration cannot mix other migration factors")
    rate = metadata.get("optimizer_learning_rate")
    if (not isinstance(factor, Mapping) or semantic_version != "v3"
            or metadata.get("semantic_version") != "v3" or seed != metadata.get("seed")
            or observation_layout != ROLE_OBSERVATION_LAYOUT or source_num_envs(metadata) != 1
            or metadata.get("runner_config", {}).get("device") != device
            or metadata.get("runtime_contract", {}).get("experiment_id") != "fl_capture_quality_v1"
            or policy_version_from_metadata(metadata) != HISTORY_REQUEST_CAP_TRANSITION_POLICY
            or factor.get("schema") != "wlr50_clean.FR_knee_phase_physical_innovation_sigma.v1"
            or factor.get("sigma_scaling_semantics") != FR_KNEE_PHYSICAL_SIGMA_SEMANTICS
            or type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
            or factor.get("source_effective_learning_rate") != rate
            or factor.get("target_effective_learning_rate") != rate):
        raise RuntimeError("physical innovation sigma migration requires exact REQUEST role372 N1 FL source and preserved effective LR")
    source_version, target_version = HISTORY_REQUEST_CAP_TRANSITION_POLICY, FR_KNEE_PHYSICAL_INNOVATION_POLICY
    source_policy = policy_contract(source_version, observation_layout=observation_layout)
    target_policy = policy_contract(target_version, observation_layout=observation_layout)
    observation = {"source_policy_contract": source_policy, "target_policy_contract": target_policy,
        "observation_layout": observation_layout, "observation_dimension": 372, "action_dimension": 12,
        "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}
    if (metadata.get("policy_contract") != source_policy
            or factor.get("source_policy_contract") != source_policy
            or factor.get("target_policy_contract") != target_policy
            or factor.get("source_policy_version") != source_version
            or factor.get("target_policy_version") != target_version
            or factor.get("observation_contract") != observation):
        raise RuntimeError("physical innovation sigma migration lacks exact source/target policy and observation contracts")
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")["version"]
    configs = {version: semantic_runner_config(seed=seed, device=device, semantic_version="v3",
        policy_version=version, observation_layout=observation_layout, return_profile=horizon)
        for version in (source_version, target_version)}
    if (metadata["runner_config"] != configs[source_version]
            or factor.get("source_runner_config") != configs[source_version]
            or factor.get("target_runner_config") != configs[target_version]):
        raise RuntimeError("physical innovation sigma migration runner differs from its exact contracts")
    return factor


def _validated_task_conditioned_hip_wheel_factor(metadata: Mapping[str, Any], record: Mapping[str, Any], *,
        semantic_version: str, seed: int, device: str, observation_layout: str | None) -> Mapping[str, Any] | None:
    """Reviewed joint reward/potential and sigma boundary, identity learned state."""
    factor = record.get("task_conditioned_hip_wheel_factor")
    if factor is None:
        return None
    from .semantic_migration import source_num_envs
    from .semantic_policy_distribution import (FR_KNEE_PHYSICAL_INNOVATION_POLICY,
        TASK_CONDITIONED_HIP_WHEEL_POLICY, TASK_CONDITIONED_HIP_WHEEL_SIGMA_SEMANTICS, policy_version_from_metadata)
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    exclusive = ("execution_factor", "instrumentation_observation_contract", "video_instrumentation_factor",
        "nominal_timing_factor", "body_reward_factor", "task_first_reward_factor", "height_recovery_factor",
        "execution_composition_factor", "final_stop_handoff_factor", "rr_physical_acceptance_same372_factor",
        "fl_capture_quality_same372_factor", "exploration_temperature_factor", "request_history_kernel_factor",
        "physical_innovation_sigma_factor", "archive_only_exact_bytes_factor")
    if any(record.get(key) is not None for key in exclusive):
        raise RuntimeError("task-conditioned joint migration cannot mix separate migration factors")
    rate = metadata.get("optimizer_learning_rate")
    source_version, target_version = FR_KNEE_PHYSICAL_INNOVATION_POLICY, TASK_CONDITIONED_HIP_WHEEL_POLICY
    if (not isinstance(factor, Mapping) or semantic_version != "v3" or metadata.get("semantic_version") != "v3"
            or seed != metadata.get("seed") or observation_layout != ROLE_OBSERVATION_LAYOUT
            or source_num_envs(metadata) != 1 or metadata.get("runner_config", {}).get("device") != device
            or metadata.get("runtime_contract", {}).get("experiment_id") != "fl_capture_quality_v1"
            or policy_version_from_metadata(metadata) != source_version
            or factor.get("schema") != "wlr50_clean.task_conditioned_hip_wheel_same372.v1"
            or factor.get("sigma_scaling_semantics") != TASK_CONDITIONED_HIP_WHEEL_SIGMA_SEMANTICS
            or factor.get("branch_id") != "task_conditioned_hip_wheel_v1"
            or type(rate) not in (int,float) or not math.isfinite(rate) or rate <= 0
            or factor.get("source_effective_learning_rate") != rate or factor.get("target_effective_learning_rate") != rate):
        raise RuntimeError("task-conditioned continuation requires exact e735 role372 N1 FL source and preserved effective LR")
    source_policy = policy_contract(source_version, observation_layout=observation_layout)
    target_policy = policy_contract(target_version, observation_layout=observation_layout)
    observation = {"source_policy_contract":source_policy, "target_policy_contract":target_policy,
        "observation_layout":observation_layout, "observation_dimension":372, "action_dimension":12,
        "num_envs":1, "parameter_mapping":"identity_all_parameters_and_buffers"}
    if (metadata.get("policy_contract") != source_policy or factor.get("source_policy_contract") != source_policy
            or factor.get("target_policy_contract") != target_policy
            or factor.get("source_policy_version") != source_version or factor.get("target_policy_version") != target_version
            or factor.get("observation_contract") != observation):
        raise RuntimeError("task-conditioned migration lacks exact source/target policy and observation contracts")
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")["version"]
    configs = {version:semantic_runner_config(seed=seed,device=device,semantic_version="v3",
        policy_version=version,observation_layout=observation_layout,return_profile=horizon)
        for version in (source_version,target_version)}
    if (metadata["runner_config"] != configs[source_version] or factor.get("source_runner_config") != configs[source_version]
            or factor.get("target_runner_config") != configs[target_version]):
        raise RuntimeError("task-conditioned migration runner differs from exact unchanged PPO hyperparameter contracts")
    return factor


def _validated_receiving_wheel_sigma_factor(metadata: Mapping[str, Any], record: Mapping[str, Any], *,
        semantic_version: str, seed: int, device: str, observation_layout: str | None):
    factor = record.get("receiving_wheel_sigma_factor")
    if factor is None:
        return None
    from .semantic_policy_distribution import TASK_CONDITIONED_HIP_WHEEL_POLICY, policy_version_from_metadata
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    from .semantic_migration import source_num_envs
    if any(key.endswith("_factor") and key != "receiving_wheel_sigma_factor" and value is not None
           for key, value in record.items()):
        raise RuntimeError("receiving-wheel sigma migration cannot mix independent factors")
    source_version, target_version = TASK_CONDITIONED_HIP_WHEEL_POLICY, RECEIVING_WHEEL_POLICY
    rate = metadata.get("optimizer_learning_rate")
    if (not isinstance(factor, Mapping) or semantic_version != "v3" or metadata.get("semantic_version") != "v3"
            or seed != metadata.get("seed") or observation_layout != ROLE_OBSERVATION_LAYOUT
            or source_num_envs(metadata) != 1 or metadata.get("runner_config", {}).get("device") != device
            or metadata.get("runtime_contract", {}).get("experiment_id") != "task_conditioned_hip_wheel_v1"
            or policy_version_from_metadata(metadata) != source_version
            or factor.get("schema") != "wlr50_clean.receiving_wheel_sigma_same372.v1"
            or factor.get("sigma_scaling_semantics") != RECEIVING_WHEEL_SIGMA_SEMANTICS
            or factor.get("kernel_changed") is not True or factor.get("same_mdp_claimed") is not True
            or type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
            or factor.get("source_effective_learning_rate") != rate or factor.get("target_effective_learning_rate") != rate):
        raise RuntimeError("receiving-wheel sigma requires exact task profile/N1/source effective LR")
    source_policy = policy_contract(source_version, observation_layout=observation_layout)
    target_policy = policy_contract(target_version, observation_layout=observation_layout)
    observation = {"source_policy_contract": source_policy, "target_policy_contract": target_policy,
        "observation_layout": observation_layout, "observation_dimension": 372, "action_dimension": 12,
        "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}
    if (metadata.get("policy_contract") != source_policy or factor.get("source_policy_contract") != source_policy
            or factor.get("target_policy_contract") != target_policy or factor.get("observation_contract") != observation
            or factor.get("source_policy_version") != source_version or factor.get("target_policy_version") != target_version):
        raise RuntimeError("receiving-wheel sigma lacks exact source and target contracts")
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")["version"]
    configs = {version: semantic_runner_config(seed=seed, device=device, semantic_version="v3",
        policy_version=version, observation_layout=observation_layout, return_profile=horizon)
        for version in (source_version, target_version)}
    if (metadata["runner_config"] != configs[source_version] or factor.get("source_runner_config") != configs[source_version]
            or factor.get("target_runner_config") != configs[target_version]):
        raise RuntimeError("receiving-wheel sigma changes unrelated runner hyperparameters")
    return factor


def _verify_reviewed_same410_identity_state(runner, infos, factor):
    """Non-authorizing leaf: called only after a dedicated factor is verified."""
    import torch
    from .semantic_migration import digest
    from .rl_library_wrapper import optimizer_learning_rate, capture_training_rng_state
    actual = {
        "actor_parameter_sha256":parameter_hash(runner.alg.actor),
        "critic_parameter_sha256":parameter_hash(runner.alg.critic),
        "optimizer_state_sha256":state_hash(runner.alg.optimizer.state_dict()),
        "normalizer_state_sha256":state_hash(_normalizers(runner)),
        "training_rng_state":capture_training_rng_state(seed=infos["seed"])}
    if (any(type(getattr(runner.alg, role).obs_normalizer) is not torch.nn.Identity for role in ("actor","critic"))
            or optimizer_learning_rate(runner) != factor["source_effective_learning_rate"]
            or runner.alg.learning_rate != factor["source_effective_learning_rate"]
            or any(k not in infos or digest(infos[k]) != expected
                   for k,expected in factor["preserved_metadata_sha256"].items())
            or any(infos.get(k) != value for k,value in actual.items())
            or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None
            or tuple(runner.alg.storage.actions.shape) != (128,1,12)
            or any(tuple(runner.alg.storage.observations[k].shape) != (128,1,410) for k in ("policy","critic"))):
        raise RuntimeError("reviewed same410 load changed exact state/LR/RNG or inherited a rollout")


def load_semantic_checkpoint(runner: Any, checkpoint: Path, *, contract: Mapping[str, Any], seed: int,
                             migration: Mapping[str, Any] | None = None,
                             warm_start: Mapping[str, Any] | None = None,
                             policy_migration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if migration is not None and migration.get("front_retention439_identity_factor") is not None:
        if warm_start is not None or policy_migration is not None:
            raise ValueError("front retention accounting identity cannot mix another migration")
        from .semantic_front_retention439 import load_front_retention439_identity
        return load_front_retention439_identity(runner, checkpoint, contract=contract, seed=seed, record=migration)
    if migration is not None and migration.get("rr_retention_same439_factor") is not None:
        if warm_start is not None or policy_migration is not None:
            raise ValueError("same439 reward continuation cannot mix another migration")
        from .semantic_rr_retention_migration import load_rr_retention_migration
        return load_rr_retention_migration(runner, checkpoint, contract=contract, seed=seed, record=migration)
    if migration is not None and migration.get("rear_owner_append439_factor") is not None:
        if warm_start is not None or policy_migration is not None:
            raise ValueError("owner append cannot mix another migration")
        from .semantic_rear_owner_migration import load_rear_owner_migration
        return load_rear_owner_migration(runner, checkpoint, contract=contract, seed=seed, record=migration)
    if migration is not None and migration.get("cooperative_prep_same422_factor") is not None:
        if warm_start is not None or policy_migration is not None:
            raise ValueError("cooperative-prep migration cannot mix another boundary")
        from .semantic_cooperative_prep_migration import load_cooperative_prep_migration
        return load_cooperative_prep_migration(
            runner, checkpoint, contract=contract, seed=seed, record=migration)
    if migration is not None and migration.get("p02_progress_append_factor") is not None:
        if warm_start is not None or policy_migration is not None:
            raise ValueError("P02 append cannot mix another migration")
        from .semantic_p02_progress_migration import load_p02_progress_migration
        return load_p02_progress_migration(runner,checkpoint,contract=contract,seed=seed,record=migration)
    if migration is not None and migration.get("rear_live_swing_same419_factor") is not None:
        if warm_start is not None or policy_migration is not None:
            raise ValueError("same419 live-swing migration cannot mix another boundary")
        from .semantic_rear_live_swing_migration import load_rear_live_swing_migration
        return load_rear_live_swing_migration(runner, checkpoint, contract=contract, seed=seed, record=migration)
    if migration is not None and migration.get("rear_recapture_same419_factor") is not None:
        if warm_start is not None or policy_migration is not None:
            raise ValueError("same419 recapture migration cannot mix another boundary")
        from .semantic_rear_recapture_migration import load_rear_recapture_migration
        return load_rear_recapture_migration(runner, checkpoint, contract=contract, seed=seed, record=migration)
    if migration is not None and migration.get("rear_policy_timing_factor") is not None:
        if warm_start is not None or policy_migration is not None:
            raise ValueError("rear timing migration cannot mix another boundary")
        from .semantic_rear_policy_timing_migration import load_rear_policy_timing_migration
        return load_rear_policy_timing_migration(runner, checkpoint, contract=contract, seed=seed, record=migration)
    if migration is not None and migration.get("rr_capture_transfer_factor") is not None:
        if warm_start is not None or policy_migration is not None:
            raise ValueError("RR capture append cannot mix another migration")
        from .semantic_rr_capture_migration import load_rr_capture_migration
        return load_rr_capture_migration(runner,checkpoint,contract=contract,seed=seed,record=migration)
    if migration is not None and migration.get("p05_capture_assist_factor") is not None:
        if warm_start is not None or policy_migration is not None:
            raise ValueError("P05 capture boundary cannot mix another migration")
        from .semantic_p05_capture_migration import load_p05_capture_migration
        return load_p05_capture_migration(runner,checkpoint,contract=contract,seed=seed,record=migration)
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
    if "front_retention439_runtime_identity" in metadata or "front_retention439_auxiliary" in metadata:
        from .semantic_front_retention439 import validate_front_retention439_lineage
        route = metadata.get("checkpoint_output_routing", {})
        validate_front_retention439_lineage(metadata, metadata["runtime_contract"], Path(route.get("output_root", "")),
            checkpoint_output_routing=route)
    from .semantic_policy_distribution import policy_version_from_metadata
    verified = None
    if migration is not None and any(migration.get(key) is not None for key in (
            "exploration_temperature_factor", "request_history_kernel_factor", "physical_innovation_sigma_factor",
            "task_conditioned_hip_wheel_factor", "receiving_wheel_sigma_factor")):
        from .semantic_migration import validate_migration_plan
        verified = validate_migration_plan(checkpoint, contract, Path(migration["plan_path"]))
        if verified != dict(migration):
            raise RuntimeError("migration changed since pre-AppLauncher validation")
    temperature = _validated_exploration_temperature_factor(metadata, verified or {},
        semantic_version=runner._semantic_version, seed=seed, device=str(runner.device),
        observation_layout=getattr(runner, "_semantic_observation_layout", None))
    request_kernel = _validated_request_history_kernel_factor(metadata, verified or {},
        semantic_version=runner._semantic_version, seed=seed, device=str(runner.device),
        observation_layout=getattr(runner, "_semantic_observation_layout", None))
    physical_innovation = _validated_physical_innovation_sigma_factor(metadata, verified or {},
        semantic_version=runner._semantic_version, seed=seed, device=str(runner.device),
        observation_layout=getattr(runner, "_semantic_observation_layout", None))
    task_conditioned = _validated_task_conditioned_hip_wheel_factor(metadata, verified or {},
        semantic_version=runner._semantic_version, seed=seed, device=str(runner.device),
        observation_layout=getattr(runner, "_semantic_observation_layout", None))
    receiving_wheel = _validated_receiving_wheel_sigma_factor(metadata, verified or {},
        semantic_version=runner._semantic_version, seed=seed, device=str(runner.device),
        observation_layout=getattr(runner, "_semantic_observation_layout", None))
    kernel_boundary = temperature or request_kernel or physical_innovation or task_conditioned or receiving_wheel
    if kernel_boundary is not None:
        if (runner._semantic_policy_version != kernel_boundary["target_policy_contract"]["version"]
                or _runner_policy_contract(runner) != kernel_boundary["target_policy_contract"]
                or runner._semantic_runner_config != kernel_boundary["target_runner_config"]):
            raise RuntimeError("constructed kernel actor differs from the verified target configuration")
    else:
        if policy_version_from_metadata(metadata) != runner._semantic_policy_version:
            raise RuntimeError("checkpoint policy distribution differs from the constructed actor")
        if runner.alg.storage.observations["policy"].shape[-1] in (324, 372, P05_CAPTURE_OBSERVATION_DIM, RR_CAPTURE_OBSERVATION_DIM, REAR_POLICY_TIMING_OBSERVATION_DIM, 422, 439):
            if metadata.get("policy_contract") != _runner_policy_contract(runner):
                raise RuntimeError("checkpoint observation layout differs; explicit append migration is required")
    source_return = runner_return_profile(metadata["runner_config"],
                                         semantic_version=metadata.get("semantic_version", "v2"))
    if source_return != assert_semantic_return_consistency(runner, runner.env):
        raise RuntimeError("return-profile changes require explicit new-MDP warm start")
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
        if verified is None:
            from .semantic_migration import validate_migration_plan
            verified = validate_migration_plan(checkpoint, contract, Path(migration["plan_path"]))
            if verified != dict(migration):
                raise RuntimeError("migration changed since pre-AppLauncher validation")
        expected_contract = metadata["runtime_contract"]
        layout = getattr(runner, "_semantic_observation_layout", None)
        factor = verified.get("instrumentation_observation_contract")
        video_factor = (verified.get("video_instrumentation_factor") or {}).get("observation_contract")
        timing_factor = (verified.get("nominal_timing_factor") or {}).get("observation_contract")
        body_reward_factor = (verified.get("body_reward_factor") or {}).get("observation_contract")
        task_first_factor = (verified.get("task_first_reward_factor") or {}).get("observation_contract")
        composition_factor = (verified.get("execution_composition_factor") or {}).get("observation_contract")
        stop_handoff_factor = (verified.get("final_stop_handoff_factor") or {}).get("observation_contract")
        rr_acceptance_factor = (verified.get("rr_physical_acceptance_same372_factor") or {}).get("observation_contract")
        fl_quality_factor = (verified.get("fl_capture_quality_same372_factor") or {}).get("observation_contract")
        height_factor = (verified.get("height_recovery_factor") or {}).get("observation_contract")
        temperature_factor = None if temperature is None else temperature["observation_contract"]
        request_history_factor = None if request_kernel is None else request_kernel["observation_contract"]
        physical_innovation_factor = None if physical_innovation is None else physical_innovation["observation_contract"]
        task_conditioned_factor = None if task_conditioned is None else task_conditioned["observation_contract"]
        archive_factor = (verified.get("archive_only_exact_bytes_factor") or {}).get("observation_contract")
        budget_factor = (verified.get("training_quantity_budget_factor") or {}).get("observation_contract")
        receiving_factor = None if receiving_wheel is None else receiving_wheel["observation_contract"]
        capture_feedback_factor = (verified.get("capture_feedback_semantics_factor") or {}).get("observation_contract")
        rr_workspace_factor = (verified.get("rr_postcross_workspace_factor") or {}).get("observation_contract")
        p05_preedge_factor = (verified.get("p05_preedge_approach_recovery_factor") or {}).get("observation_contract")
        rr_capture_feedback_factor = (verified.get("rr_capture_feedback_peak_v2_factor") or {}).get("observation_contract")
        rr_capture_knee_factor = (verified.get("rr_capture_knee_v3_factor") or {}).get("observation_contract")
        rr_carry_handoff_factor = (verified.get("rr_carry_handoff_v4_factor") or {}).get("observation_contract")
        rr_progress_handoff_factor = (verified.get("rr_progress_handoff_v5_factor") or {}).get("observation_contract")
        rr_contact_onset_factor = (verified.get("rr_contact_onset_v6_factor") or {}).get("observation_contract")
        rr_signed_contact_factor = (verified.get("rr_signed_contact_v7_factor") or {}).get("observation_contract")
        rr_signed_wheel_factor = (verified.get("rr_signed_wheel_v8_factor") or {}).get("observation_contract")
        rr_postcapture_wheel_factor = (verified.get("rr_postcapture_wheel_v9_factor") or {}).get("observation_contract")
        rr_capture_reserve_factor = (verified.get("rr_capture_reserve_v10_factor") or {}).get("observation_contract")
        if sum(x is not None for x in (video_factor, timing_factor, body_reward_factor, task_first_factor, composition_factor, stop_handoff_factor, rr_acceptance_factor, fl_quality_factor, height_factor, temperature_factor, request_history_factor, physical_innovation_factor, task_conditioned_factor, archive_factor, budget_factor, receiving_factor, capture_feedback_factor, rr_workspace_factor, p05_preedge_factor, rr_capture_feedback_factor, rr_capture_knee_factor, rr_carry_handoff_factor, rr_progress_handoff_factor, rr_contact_onset_factor, rr_signed_contact_factor, rr_signed_wheel_factor, rr_postcapture_wheel_factor, rr_capture_reserve_factor)) > 1:
            raise RuntimeError("reviewed same-layout migration receipts must be exclusive")
        reviewed_factor = video_factor or timing_factor or body_reward_factor or task_first_factor or composition_factor or stop_handoff_factor or rr_acceptance_factor or fl_quality_factor or height_factor or temperature_factor or request_history_factor or physical_innovation_factor or task_conditioned_factor or archive_factor or budget_factor or receiving_factor or capture_feedback_factor or rr_workspace_factor or p05_preedge_factor or rr_capture_feedback_factor or rr_capture_knee_factor or rr_carry_handoff_factor or rr_progress_handoff_factor or rr_contact_onset_factor or rr_signed_contact_factor or rr_signed_wheel_factor or rr_postcapture_wheel_factor or rr_capture_reserve_factor
        if reviewed_factor is not None:
            if factor is not None:
                raise RuntimeError("reviewed control/video and instrumentation observation receipts must be exclusive")
            factor = reviewed_factor
            if (tuple(runner.alg.storage.actions.shape) != (ROLLOUT_LENGTH, 1, factor["action_dimension"])
                    or any(tuple(runner.alg.storage.observations[key].shape)
                           != (ROLLOUT_LENGTH, 1, factor["observation_dimension"])
                           for key in ("policy", "critic"))):
                raise RuntimeError("reviewed migration requires fresh verified N1 policy/critic/action storage")
        actual_contract = _runner_policy_contract(runner)
        if layout is not None and (factor is None
                or factor["source_policy_contract"] != metadata.get("policy_contract")
                or factor["target_policy_contract"] != actual_contract
                or factor["observation_layout"] != layout
                or factor["num_envs"] != target_count):
            raise RuntimeError("role-layout migration lacks exact verified source/target observation factor")
        if (runner.alg.storage.observations["policy"].shape[-1] != verified["observation_dimension"]
                or runner.alg.storage.actions.shape[-1] != verified["action_dimension"]
                or actual_contract["observation_dimension"] != verified["observation_dimension"]
                or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None):
            raise RuntimeError("migration requires verified-layout fresh storage with no old rollout")
        if kernel_boundary is None and metadata.get("runner_config") != semantic_runner_config(seed=seed, device=str(runner.device),
                semantic_version=metadata.get("semantic_version", "v2"),
                policy_version=runner._semantic_policy_version, observation_layout=layout):
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
    assert_semantic_return_consistency(runner, runner.env)
    if request_kernel is not None and (
            optimizer_learning_rate(runner) != 1e-5
            or optimizer_learning_rate(runner) != infos.get("optimizer_learning_rate")):
        raise RuntimeError("request-history migration changed source effective Adam learning rate")
    if physical_innovation is not None and (
            optimizer_learning_rate(runner) != infos.get("optimizer_learning_rate")
            or optimizer_learning_rate(runner) != physical_innovation["source_effective_learning_rate"]):
        raise RuntimeError("physical innovation sigma migration changed source effective Adam learning rate")
    if task_conditioned is not None and (
            optimizer_learning_rate(runner) != infos.get("optimizer_learning_rate")
            or optimizer_learning_rate(runner) != task_conditioned["source_effective_learning_rate"]):
        raise RuntimeError("task-conditioned migration changed source effective Adam learning rate")
    if migration is not None:
        infos = {**infos, "resume_migration": dict(migration)}
        if verified.get("capture_feedback_semantics_factor") is not None:
            from .semantic_capture_feedback_migration import record_loaded_capture_feedback
            infos = record_loaded_capture_feedback(runner,infos,verified)
        if verified.get("rr_postcross_workspace_factor") is not None:
            from .semantic_rr_workspace_migration import record_loaded_rr_workspace
            infos = record_loaded_rr_workspace(runner,infos,verified)
        if verified.get("p05_preedge_approach_recovery_factor") is not None:
            from .semantic_p05_preedge_migration import record_loaded_p05_preedge
            infos = record_loaded_p05_preedge(runner,infos,verified)
        if verified.get("rr_capture_feedback_peak_v2_factor") is not None:
            from .semantic_rr_capture_feedback_migration import record_loaded_rr_capture_feedback
            infos = record_loaded_rr_capture_feedback(runner,infos,verified)
        if verified.get("rr_capture_knee_v3_factor") is not None:
            from .semantic_rr_capture_knee_migration import record_loaded_rr_capture_knee
            infos = record_loaded_rr_capture_knee(runner,infos,verified)
        if verified.get("rr_carry_handoff_v4_factor") is not None:
            from .semantic_rr_carry_handoff_migration import record_loaded_rr_carry_handoff
            infos = record_loaded_rr_carry_handoff(runner,infos,verified)
        if verified.get("rr_progress_handoff_v5_factor") is not None:
            from .semantic_rr_progress_handoff_migration import record_loaded_rr_progress_handoff
            infos = record_loaded_rr_progress_handoff(runner,infos,verified)
        if verified.get("rr_contact_onset_v6_factor") is not None:
            from .semantic_rr_contact_onset_migration import record_loaded_rr_contact_onset
            infos = record_loaded_rr_contact_onset(runner,infos,verified)
        if verified.get("rr_signed_contact_v7_factor") is not None:
            from .semantic_rr_signed_contact_migration import record_loaded_rr_signed_contact
            infos = record_loaded_rr_signed_contact(runner,infos,verified)
        if verified.get("rr_signed_wheel_v8_factor") is not None:
            from .semantic_rr_signed_wheel_migration import record_loaded_rr_signed_wheel
            infos = record_loaded_rr_signed_wheel(runner,infos,verified)
        if verified.get("rr_postcapture_wheel_v9_factor") is not None:
            from .semantic_rr_postcapture_wheel_migration import record_loaded_rr_postcapture_wheel
            infos = record_loaded_rr_postcapture_wheel(runner,infos,verified)
        if verified.get("rr_capture_reserve_v10_factor") is not None:
            from .semantic_rr_capture_reserve_migration import record_loaded_rr_capture_reserve
            infos = record_loaded_rr_capture_reserve(runner,infos,verified)
        if receiving_wheel is not None:
            if (optimizer_learning_rate(runner) != infos.get("optimizer_learning_rate")
                    or optimizer_learning_rate(runner) != receiving_wheel["source_effective_learning_rate"]):
                raise RuntimeError("receiving-wheel sigma changed source effective Adam learning rate")
            preserved = {**receiving_wheel["preserved_training_state"],
                **receiving_wheel["preserved_branch_metadata"],
                **receiving_wheel["counter_origin"],
                "stage_requested_decisions": receiving_wheel["source_stage_requested_decisions"]}
            if any(infos.get(key) != value for key, value in preserved.items()):
                raise RuntimeError("receiving-wheel sigma changed source state, branch/aux lineage or counters")
            infos["receiving_wheel_sigma_migration"] = {
                "factor": dict(receiving_wheel), "plan_path": verified["plan_path"],
                "plan_sha256": verified["plan_sha256"],
                "source_checkpoint_sha256": verified["source_checkpoint_sha256"],
                "source_contract_sha256": verified["source_contract_sha256"],
                "target_contract_sha256": verified["target_contract_sha256"]}
        quantity = verified.get("training_quantity_budget_factor")
        if quantity is not None:
            if (optimizer_learning_rate(runner) != infos.get("optimizer_learning_rate")
                    or optimizer_learning_rate(runner) != quantity["source_effective_learning_rate"]):
                raise RuntimeError("quantity-only continuation changed source effective Adam learning rate")
            infos["training_quantity_budget_extension"] = {
                "factor": dict(quantity), "plan_path": verified["plan_path"],
                "plan_sha256": verified["plan_sha256"],
                "source_checkpoint_sha256": verified["source_checkpoint_sha256"],
                "source_contract_sha256": verified["source_contract_sha256"],
                "target_contract_sha256": verified["target_contract_sha256"]}
        if task_conditioned is not None:
            infos["task_conditioned_hip_wheel_branch"] = {
                "schema":"wlr50_clean.task_conditioned_hip_wheel_branch.v1",
                "branch_id":task_conditioned["branch_id"], "counter_origin":dict(task_conditioned["counter_origin"]),
                "source_checkpoint_sha256":verified["source_checkpoint_sha256"],
                "source_manifest_sha256":verified["source_manifest_sha256"], "migration_added_updates":0}
        fl_factor = verified.get("fl_capture_quality_same372_factor")
        if fl_factor is not None:
            if optimizer_learning_rate(runner) != infos.get("optimizer_learning_rate"):
                raise RuntimeError("FL quality continuation changed the source effective Adam learning rate")
            infos["fl_capture_quality_branch"] = {
                "schema": "wlr50_clean.fl_capture_quality_branch.v1",
                "branch_id": fl_factor["branch_id"], "counter_origin": dict(fl_factor["counter_origin"]),
                "source_checkpoint_sha256": verified["source_checkpoint_sha256"],
                "source_manifest_sha256": verified["source_manifest_sha256"], "migration_added_updates": 0}
        rr_factor = verified.get("rr_physical_acceptance_same372_factor")
        if rr_factor is not None:
            if optimizer_learning_rate(runner) != infos.get("optimizer_learning_rate"):
                raise RuntimeError("RR continuation changed the source effective Adam learning rate")
            infos["rr_task_branch"] = {
                "schema": "wlr50_clean.rr_task_continuation_branch.v1",
                "branch_id": rr_factor["branch_id"], "counter_origin": dict(rr_factor["counter_origin"]),
                "source_checkpoint_sha256": verified["source_checkpoint_sha256"],
                "source_manifest_sha256": verified["source_manifest_sha256"],
                "migration_added_updates": 0,
            }
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


def _compensate_observation_input_scales(runner: Any, transition: Mapping[str, Any],
                                        source_infos: Mapping[str, Any]) -> dict[str, Any]:
    """Compensate only reviewed fixed encoder columns after verified source load.

    For un-clipped source history, x_new=x_old/source_to_target_scale. Multiplying
    the two input-weight columns restores the old preactivation, up to floating
    point roundoff. Previously clipped history exposes new information; neither
    identical output over that expanded domain nor identical weights is claimed.
    """
    import torch
    source_scales = transition.get("source_scales")
    changed = source_scales == [4, 4]
    factors = [1.5, 1.5] if changed else [1.0, 1.0]
    if (transition.get("schema") != "wlr50_clean.transfer_roles_observation_scale_transition.v1"
            or transition.get("experiment_id") != "transfer_roles_v1"
            or transition.get("observation_dimension") != 324
            or transition.get("columns") != [210, 222]
            or transition.get("feature_groups") != ["previous_residual_full12", "previous_previous_residual_full12"]
            or transition.get("channel_index") != 3
            or source_scales not in ([4, 4], [6, 6])
            or transition.get("target_scales") != [6, 6]
            or transition.get("factors") != factors or transition.get("changed") is not changed
            or transition.get("first_layer_compensation") != "multiply_actor_and_critic_input_columns_by_target_over_source"):
        raise RuntimeError("unrecognized observation input-scale compensation")
    normalizer_hash = state_hash(_normalizers(runner))
    replacements = []
    for role in ("actor", "critic"):
        model = getattr(runner.alg, role)
        if (model.obs_normalization or any(_normalizers(runner).values())
                or normalizer_hash != source_infos["normalizer_state_sha256"]
                or parameter_hash(model) != source_infos[f"{role}_parameter_sha256"]):
            raise RuntimeError("input-scale compensation requires verified source weights and identity normalizers")
        first = next(((name, layer) for name, layer in model.mlp.named_modules()
                      if isinstance(layer, torch.nn.Linear)), None)
        if first is None or first[0] != "0" or first[1].in_features != 324:
            raise RuntimeError("input-scale compensation requires the existing first 324-input Linear")
        weight = first[1].weight.detach().clone()
        for column, factor in zip(transition["columns"], factors):
            weight[:, column] *= factor
        if not bool(torch.isfinite(weight).all()):
            raise RuntimeError("input-scale compensation produced nonfinite weights")
        replacements.append((first[1], weight))
    # Validate both models before changing either. There are no new parameters,
    # buffers, normalizer updates, observations, cache calls or random samples.
    if changed:
        with torch.no_grad():
            for layer, weight in replacements:
                layer.weight.copy_(weight)
    return {"schema": "wlr50_clean.observation_scale_compensation_evidence.v1",
            "transition": copy.deepcopy(dict(transition)),
            "source_actor_parameter_sha256": source_infos["actor_parameter_sha256"],
            "source_critic_parameter_sha256": source_infos["critic_parameter_sha256"],
            "compensated_actor_parameter_sha256": parameter_hash(runner.alg.actor),
            "compensated_critic_parameter_sha256": parameter_hash(runner.alg.critic),
            "normalizer_state_sha256": normalizer_hash,
            "first_layer_parameter": "mlp.0.weight", "compensation_applied": changed,
            "all_other_parameters_and_buffers_preserved": True,
            "previous_raw_history_195_207_and_policy_kernel_unchanged": True,
            "equivalence_scope": "same physical features with both source histories unclipped (abs <= 80); floating-point tolerance, not bitwise outputs; no expanded-domain or changed-task-feature equivalence claim"}


def _load_observation_append_source(runner: Any, checkpoint: Path, *, seed: int,
                                    record: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the old official runner, then append zero input columns only.

    The temporary source sees an immutable prefix of the current observation.
    It cannot reset or step physics, and no source rollout is inherited.
    """
    import torch
    from tensordict import TensorDict
    from .semantic_policy_distribution import HISTORY_POLICY
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    transition = record["observation_append_transition"]
    if (transition.get("schema") != "wlr50_clean.transfer_roles_observation_append_transition.v1"
            or transition.get("source_observation_dimension") != 324
            or transition.get("target_observation_dimension") != 372
            or transition.get("source_observation_layout") is not None
            or transition.get("target_observation_layout") != ROLE_OBSERVATION_LAYOUT
            or transition.get("first_layer_parameter") != "mlp.0.weight"
            or transition.get("appended_columns") != [324, 372]
            or runner._semantic_policy_version != HISTORY_POLICY
            or getattr(runner, "_semantic_observation_layout", None) != ROLE_OBSERVATION_LAYOUT
            or tuple(runner.alg.storage.actions.shape) != (128, 1, 12)
            or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None):
        raise RuntimeError("unrecognized or nonempty transfer-role input append")
    observations = runner.env.get_observations().to(runner.device)
    tensor = observations["policy"].detach().clone()
    if tuple(tensor.shape) != (1, 372) or not bool(torch.isfinite(tensor).all()):
        raise RuntimeError("role append requires one finite current372 observation")
    source_tensor = tensor[:, :324].clone()

    class SourceObservationOnly:
        num_envs, num_actions = 1, 12
        cfg = {"semantic_version": "v3", "observation_dimension": 324,
               "migration_observation_only": True}

        def get_observations(self):
            return TensorDict({"policy": source_tensor.clone(), "critic": source_tensor.clone()},
                              batch_size=[1], device=runner.device)

        def reset(self, *args, **kwargs):
            raise RuntimeError("observation append source must never reset physics")

        def step(self, *args, **kwargs):
            raise RuntimeError("observation append source must never step physics")

    source, _ = construct_semantic_runner(SourceObservationOnly(), seed=seed,
        device=str(runner.device), policy_version=HISTORY_POLICY, initialize_actor=False)
    infos = load_semantic_checkpoint(source, checkpoint,
        contract=record["source_runtime_contract"], seed=seed)
    if (transition.get("source_policy_contract") != _runner_policy_contract(source)
            or transition.get("target_policy_contract") != _runner_policy_contract(runner)
            or any(_normalizers(source).values()) or any(_normalizers(runner).values())
            or state_hash(_normalizers(runner)) != infos["normalizer_state_sha256"]
            or type(source.alg.optimizer) is not torch.optim.Adam):
        raise RuntimeError("role append source policy/identity normalizer/Adam differs")
    mapped_states = {}
    for role in ("actor", "critic"):
        old = getattr(source.alg, role).state_dict()
        target = getattr(runner.alg, role).state_dict()
        if old.keys() != target.keys():
            raise RuntimeError("role append cannot add model parameters except first-layer columns")
        mapped = {}
        for name, value in old.items():
            candidate = target[name]
            if value.dtype != candidate.dtype or not bool(torch.isfinite(value).all()):
                raise RuntimeError("role append source parameter dtype or finiteness differs")
            if name == "mlp.0.weight":
                if tuple(value.shape) != (256, 324) or tuple(candidate.shape) != (256, 372):
                    raise RuntimeError("role append requires the existing256-wide first Linear")
                mapped[name] = torch.zeros_like(candidate)
                mapped[name][:, :324].copy_(value)
            else:
                if value.shape != candidate.shape:
                    raise RuntimeError("role append changed a non-input model parameter")
                mapped[name] = value.detach().clone()
        if "mlp.0.weight" not in mapped:
            raise RuntimeError("role append has no reviewed first-layer parameter")
        mapped_states[role] = mapped
    # Validate both complete models before mutating either target.
    for role, state in mapped_states.items():
        getattr(runner.alg, role).load_state_dict(state, strict=True)
        actual = getattr(runner.alg, role).state_dict()
        if any(not torch.equal(actual[name], value) for name, value in state.items()):
            raise RuntimeError("role append changed a copied tensor")
    source_obs = source.env.get_observations()
    target_obs = TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[1], device=runner.device)
    with torch.inference_mode():
        source_mean = source.alg.actor(source_obs, stochastic_output=False)
        target_mean = runner.alg.actor(target_obs, stochastic_output=False)
        source_head = source.alg.actor.mlp(source.alg.actor.get_latent(source_obs))
        target_head = runner.alg.actor.mlp(runner.alg.actor.get_latent(target_obs))
        source_std, target_std = source_head[..., 1, :].exp(), target_head[..., 1, :].exp()
        source_value, target_value = source.alg.critic(source_obs), runner.alg.critic(target_obs)
        for before, after in ((source_mean, target_mean), (source_std, target_std), (source_value, target_value)):
            if not bool(torch.isfinite(before).all() & torch.isfinite(after).all()):
                raise RuntimeError("nonfinite initial input-append comparison")
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
        if not bool((source_std > 0).all() & (target_std > 0).all()):
            raise RuntimeError("role append produced invalid conditional standard deviation")
    runner.current_learning_iteration = int(infos["ppo_updates"])
    evidence = {"schema": "wlr50_clean.observation_append_evidence.v1",
        "transition": copy.deepcopy(dict(transition)),
        "source_actor_parameter_sha256": infos["actor_parameter_sha256"],
        "source_critic_parameter_sha256": infos["critic_parameter_sha256"],
        "appended_actor_parameter_sha256": parameter_hash(runner.alg.actor),
        "appended_critic_parameter_sha256": parameter_hash(runner.alg.critic),
        "source_optimizer_state_sha256_verified": infos["optimizer_state_sha256"],
        "normalizer_state_sha256": infos["normalizer_state_sha256"],
        "original324_columns_and_all_other_tensors_preserved": True,
        "appended48_columns_zero_initialized": True,
        "history_columns_195_207_and_rho_unchanged": True,
        "mean_max_abs_difference": float((source_mean-target_mean).abs().max()),
        "std_max_abs_difference": float((source_std-target_std).abs().max()),
        "value_max_abs_difference": float((source_value-target_value).abs().max()),
        "physical_steps_added": 0, "policy_decisions_added": 0, "ppo_updates_added": 0,
        "optimizer_steps_added": 0, "old_rollout_inherited": False,
        "optimizer_handling": "source_Adam_verified_then_fresh_Adam_3e-5_due_input_shape_change",
        "equivalence_scope": "same original324 features; floating-point tolerance, not bitwise trajectory equivalence or complete sliding-window Markov reconstruction"}
    return {**infos, "observation_append_evidence": evidence,
        "actor_parameter_sha256": evidence["appended_actor_parameter_sha256"],
        "critic_parameter_sha256": evidence["appended_critic_parameter_sha256"],
        "policy_contract": _runner_policy_contract(runner),
        "runner_config": copy.deepcopy(runner._semantic_runner_config)}


def _load_v3_warm_start(runner: Any, checkpoint: Path, *, contract: Mapping[str, Any],
                        seed: int, record: Mapping[str, Any]) -> dict[str, Any]:
    """Reuse learned networks, explicitly discard old optimizer and rollout state."""
    import torch
    from .semantic_migration import (
        build_v3_warm_start_record, checkpoint_metadata, SAME372_AUTHORITY_SCHEMA, ALL_STAGE_SCHEMA,
        FSM_REFERENCE_P09_SCHEMA, CAPTURE_HANDOFF_SAME372_SCHEMA,
    )
    kernel = record.get("policy_kernel_transition")
    options = {} if kernel is None else {"target_policy_version": kernel.get("target_policy_version")}
    verified = build_v3_warm_start_record(checkpoint, contract, **options)
    if verified != dict(record):
        raise RuntimeError("new-MDP checkpoint/configuration binding changed after preflight")
    from .semantic_policy_distribution import policy_version_from_metadata
    metadata = checkpoint_metadata(checkpoint)
    source_version = policy_version_from_metadata(metadata)
    target_version = source_version if kernel is None else kernel["target_policy_version"]
    if runner._semantic_policy_version != target_version:
        raise RuntimeError("warm-start target policy kernel differs from its explicit migration")
    storage = runner.alg.storage
    append_transition = record.get("observation_append_transition")
    same_layout = record.get("observation_same_layout_transition")
    if same_layout is not None:
        from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
        if (same_layout.get("schema") not in (SAME372_AUTHORITY_SCHEMA, ALL_STAGE_SCHEMA, FSM_REFERENCE_P09_SCHEMA, CAPTURE_HANDOFF_SAME372_SCHEMA)
                or same_layout.get("source_observation_dimension") != 372
                or same_layout.get("target_observation_dimension") != 372
                or same_layout.get("source_observation_layout") != ROLE_OBSERVATION_LAYOUT
                or same_layout.get("target_observation_layout") != ROLE_OBSERVATION_LAYOUT
                or same_layout.get("parameter_mapping") != "identity_all_parameters_and_buffers"
                or getattr(runner, "_semantic_observation_layout", None) != ROLE_OBSERVATION_LAYOUT
                or append_transition is not None or kernel is not None
                or record.get("observation_scale_transition") is not None
                or metadata["runner_config"] != runner._semantic_runner_config):
            raise RuntimeError("same372 authority must preserve the exact source runner, layout and parameter mapping")
    target_dimension = 372 if append_transition is not None or same_layout is not None else 324
    if (tuple(storage.actions.shape) != (128, 1, 12)
            or storage.observations["policy"].shape[-1] != target_dimension
            or storage.step != 0 or runner.alg.transition.actions is not None):
        raise RuntimeError("v3 warm start requires fresh N1 explicit-layout/12-action rollout storage")
    expected_config = semantic_runner_config(seed=seed, device=str(runner.device), semantic_version="v3",
        policy_version=runner._semantic_policy_version,
        observation_layout=getattr(runner, "_semantic_observation_layout", None))
    # Official RSL 5.0.1 consumes these factory-only entries during construction.
    from rsl_rl.utils import resolve_callable
    for role in ("actor", "critic", "algorithm"):
        if type(getattr(runner.alg, role, runner.alg)) is not resolve_callable(expected_config[role].pop("class_name")):
            raise RuntimeError("v3 runner factory produced an unexpected official model/algorithm")
    expected_config["actor"]["distribution_cfg"].pop("class_name")
    expected_config["algorithm"].pop("share_cnn_encoders")
    if runner.cfg != expected_config:
        raise RuntimeError("v3 runner must use the explicit continuation configuration")
    assert_semantic_return_consistency(runner, runner.env)
    if append_transition is not None:
        if kernel is not None or record.get("observation_scale_transition") is not None:
            raise RuntimeError("input append cannot combine policy-kernel or scale compensation")
        infos = _load_observation_append_source(runner, checkpoint, seed=seed, record=record)
    else:
        infos = dict(load_checkpoint_round_trip(runner, checkpoint))
    if infos.get("seed") != seed or (append_transition is None and any(metadata.get(key) != value for key, value in infos.items())):
        raise RuntimeError("v3 warm start source embedded metadata/seed differs from verified sidecar")
    for key, actual in (("actor_parameter_sha256", parameter_hash(runner.alg.actor)),
                        ("critic_parameter_sha256", parameter_hash(runner.alg.critic)),
                        ("optimizer_state_sha256", state_hash(runner.alg.optimizer.state_dict())),
                        ("normalizer_state_sha256", state_hash(_normalizers(runner)))):
        if append_transition is not None and key == "optimizer_state_sha256":
            continue  # Verified on the temporary source, never loaded into new-shaped target.
        if infos.get(key) != actual:
            raise RuntimeError(f"v3 warm start failed source {key} verification")
    if any(_normalizers(runner).values()):
        raise RuntimeError("v3 continuation requires verified identity RSL normalizers")
    scale_transition = verified.get("observation_scale_transition")
    if scale_transition is not None:
        scale_evidence = _compensate_observation_input_scales(runner, scale_transition, infos)
        infos = {**infos, "observation_scale_compensation_evidence": scale_evidence,
                 "actor_parameter_sha256": scale_evidence["compensated_actor_parameter_sha256"],
                 "critic_parameter_sha256": scale_evidence["compensated_critic_parameter_sha256"]}
    # The temporary official restore above proves the saved state before this
    # deliberate new-MDP reset. No old Adam moment is used by any optimizer step.
    learning_rate, adam_options = 3e-5, {}
    if same_layout is not None and same_layout.get("schema") in (FSM_REFERENCE_P09_SCHEMA, CAPTURE_HANDOFF_SAME372_SCHEMA):
        if type(runner.alg.optimizer) is not torch.optim.Adam or len(runner.alg.optimizer.param_groups) != 1:
            raise RuntimeError("source-LR same372 migration requires the verified single-group official Adam")
        learning_rate = optimizer_learning_rate(runner)
        if (learning_rate != metadata.get("optimizer_learning_rate")
                or learning_rate != record["optimizer"]["initial_learning_rate"]
                or record["optimizer"].get("learning_rate_policy") != "preserve_verified_source_effective_learning_rate"):
            raise RuntimeError("source-LR same372 effective Adam learning rate differs from its migration record")
        adam_options = {key: copy.deepcopy(runner.alg.optimizer.param_groups[0][key])
                        for key in runner.alg.optimizer.defaults if key != "lr"}
    runner.alg.optimizer = torch.optim.Adam(
        list(runner.alg.actor.parameters()) + list(runner.alg.critic.parameters()),
        lr=learning_rate, **adam_options)
    runner.alg.learning_rate = learning_rate
    assert_semantic_return_consistency(runner, runner.env)
    restore_training_rng_state(infos["training_rng_state"], expected_seed=seed)
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    if "policy_version" in infos:
        infos["policy_version"] = runner._semantic_policy_version
    return {**infos, "semantic_version": "v3", "new_mdp_warm_start": dict(record),
            "source_stage_requested_decisions": dict(infos.get("stage_requested_decisions", {})),
            "stage_requested_decisions": dict(verified["target_stage_requested_decisions"]),
            "new_mdp_origin_global_policy_decisions": verified["new_mdp_origin_global_policy_decisions"],
            "resume_source_checkpoint": {"checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": sha256_file(checkpoint), "manifest": str(sidecar.resolve()),
                "manifest_sha256": sha256_file(sidecar)}}


def compare_warm_start_policy_kernel(runner: Any, env: Any, *, record: Mapping[str, Any]) -> dict[str, Any]:
    """Same verified learned tensors/observation, not a physical rollout.

    This direct deterministic tensor calculation does not sample, mutate either
    distribution cache, or claim the target conditional mean matches the source.
    """
    import torch
    from .semantic_policy_distribution import HISTORY_POLICY, STATE_DEPENDENT_POLICY, policy_contract
    from .semantic_history_actor import SemanticHistoryMLPModel
    kernel = record.get("policy_kernel_transition", {})
    if (kernel.get("source_policy_version") != STATE_DEPENDENT_POLICY
            or kernel.get("target_policy_version") != HISTORY_POLICY
            or kernel.get("target_policy_contract") != policy_contract(HISTORY_POLICY)
            or type(runner.alg.actor) is not SemanticHistoryMLPModel):
        raise ValueError("same-state kernel comparison requires the explicit verified history migration")
    actor = runner.alg.actor
    cached = actor.distribution._distribution
    rng = capture_training_rng_state(seed=int(record.get("source_seed", env.seed)))
    observations = env.get_observations().to(runner.device)
    with torch.inference_mode():
        latent = actor.get_latent(observations)
        head = actor.mlp(latent)
        mu, sigma = head[..., 0, :], head[..., 1, :].exp()
        history = latent[..., 195:207].clamp(-20., 20.)
        target = actor(observations, stochastic_output=False)
        expected = .1 * mu + .9 * history
        if not torch.equal(target, expected):
            raise RuntimeError("actual history actor differs from its specified conditional mean")
        value = runner.alg.critic(observations)
        kl = ((mu-target).square()/(2.*sigma.square())).sum(-1)
    if (actor.distribution._distribution is not cached
            or capture_training_rng_state(seed=int(record.get("source_seed", env.seed))) != rng):
        raise RuntimeError("deterministic kernel comparison mutated sampling cache or RNG")
    if any(not bool(torch.isfinite(x).all()) for x in (mu, sigma, target, value, kl)) or not bool((sigma > 0).all()):
        raise RuntimeError("nonfinite initial policy-kernel comparison")
    return {"schema": "wlr50_clean.semantic_same_state_policy_kernel_comparison.v1",
            "source_policy_contract": kernel["source_policy_contract"],
            "target_policy_contract": kernel["target_policy_contract"],
            "observation_sha256": state_hash(observations["policy"]),
            "observation_dimension": 324, "source_mean_full12": mu.cpu().tolist(),
            "previous_raw_history_full12": history.cpu().tolist(),
            "target_conditional_mean_full12": target.cpu().tolist(),
            "shared_conditional_std_full12": sigma.cpu().tolist(),
            "shared_critic_value": value.cpu().tolist(),
            "source_to_target_conditional_kl": kl.cpu().tolist(),
            "learned_weights_changed": False, "rng_or_sampling_cache_changed": False,
            "scope": "same-observation conditional policy comparison; no native dispatch, trajectory or performance claim"}


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


def semantic_curriculum_epoch(config: Mapping[str, Any]) -> dict[str, Any]:
    """Bind reset behavior separately from the actual on-policy credit stream."""
    request = jsonable(config.get("prefix_request"))
    result = {"reset_sampling": jsonable(config.get("reset_sampling", "P01_only")),
              "prefix_request": request}
    provenance = config.get("prefix_policy_provenance")
    if isinstance(request, Mapping) and request.get("source") in ("frozen_checkpoint_policy", "successful_nominal"):
        if not isinstance(provenance, Mapping) or not provenance:
            raise ValueError("checkpoint-policy curriculum must be installed before training/publication")
    if provenance is not None:
        result["prefix_policy_provenance"] = jsonable(provenance)
    return result


@contextmanager
def _terminal_evidence_before_reset(env: Any, writer: Any, *, defer_terminal_reset: bool = False):
    """Install a collector-owned sink only for this single-environment step."""
    if not isinstance(env, SemanticRslAdapter):
        yield
        return
    if env._terminal_evidence_writer is not None or env._defer_terminal_reset:
        raise RuntimeError("terminal evidence writer is already installed")
    env._terminal_evidence_writer = writer
    env._defer_terminal_reset = defer_terminal_reset
    try:
        yield
    finally:
        env._terminal_evidence_writer = None
        env._defer_terminal_reset = False


def _supports_deferred_terminal_reset(runner: Any, env: Any) -> bool:
    """Limit this scheduling optimization to the existing stateless N=1 ABI.

    RSL processes next observations through normalizers/RND before storage.
    A terminal rather than reset observation is equivalent only for this
    supported identity-normalized, nonrecurrent, no-intrinsic-reward path.
    Other adapters/topologies retain their original autoreset behavior.
    """
    import torch
    if not isinstance(env, SemanticRslAdapter) or env.num_envs != 1:
        return False
    if getattr(runner.alg, "rnd", None) is not None:
        return False
    return all(getattr(model, "obs_normalization", None) is False
               and type(getattr(model, "obs_normalizer", None)) is torch.nn.Identity
               and getattr(model, "is_recurrent", None) is False
               for model in (runner.alg.actor, runner.alg.critic))


def audited_history_policy_request(actor, observation, action_call, *, stochastic: bool):
    """Observe the one real actor MLP call; add neither a forward nor a draw.

    Request tanh is dimensionless. Per-phase physical scaling, mapper/slew and
    actual actuator response belong to the same decision's applied/native audit.
    """
    import torch
    from .semantic_history_actor import (SemanticQuarterTemperedHistoryMLPModel,
        SemanticCapTransitionQuarterHistoryMLPModel, cap_transition_request_history,
        SemanticFRKneePhysicalInnovationHistoryMLPModel, physical_innovation_effective_log_std,
        SemanticTaskConditionedHipWheelHistoryMLPModel, task_conditioned_effective_log_std,
        history_conditioned_head, HISTORY_START, HISTORY_STOP, HISTORY_RHO)
    from .semantic_policy_distribution import (HISTORY_REQUEST_CAP_TRANSITION_POLICY,
        REQUEST_HISTORY_SEMANTICS, FR_KNEE_PHYSICAL_INNOVATION_POLICY, FR_KNEE_PHYSICAL_SIGMA_SEMANTICS,
        TASK_CONDITIONED_HIP_WHEEL_POLICY, TASK_CONDITIONED_HIP_WHEEL_SIGMA_SEMANTICS)
    from .semantic_receiving_wheel_sigma import (
        SemanticReceivingWheelSigmaHistoryMLPModel, receiving_wheel_effective_log_std)
    from .semantic_p05_capture_actor import SemanticP05CaptureHistoryMLPModel, p05_capture_request_history
    from .semantic_rr_capture_actor import SemanticRRCaptureHistoryMLPModel
    from .semantic_rear_policy_timing_actor import (
        SemanticRearPolicyTimingHistoryMLPModel, rear_policy_timing_effective_log_std)
    from .semantic_p02_progress_actor import SemanticP02ProgressHistoryMLPModel, p02_progress_effective_log_std
    from .semantic_rear_cooperative_prep_actor import SemanticRearCooperativePrepHistoryMLPModel
    from .semantic_rear_owner_actor import SemanticRearOwnerRecoveryHistoryMLPModel, rear_owner_effective_log_std
    from .semantic_rear_cooperative_prep_sigma import cooperative_prep_effective_log_std
    if type(actor) not in (SemanticQuarterTemperedHistoryMLPModel, SemanticCapTransitionQuarterHistoryMLPModel,
                          SemanticFRKneePhysicalInnovationHistoryMLPModel, SemanticTaskConditionedHipWheelHistoryMLPModel,
                          SemanticReceivingWheelSigmaHistoryMLPModel, SemanticP05CaptureHistoryMLPModel, SemanticRRCaptureHistoryMLPModel, SemanticRearPolicyTimingHistoryMLPModel, SemanticP02ProgressHistoryMLPModel, SemanticRearCooperativePrepHistoryMLPModel, SemanticRearOwnerRecoveryHistoryMLPModel):
        raise ValueError("request audit requires an exact supported quarter HISTORY actor")
    heads = []
    handle = actor.mlp.register_forward_hook(lambda _module, _inputs, output: heads.append(output.detach().clone()))
    try:
        raw = action_call()
    finally:
        handle.remove()
    if len(heads) != 1 or tuple(heads[0].shape) != (1, 2, 12):
        raise RuntimeError("request audit did not observe exactly one actual N1 actor head")
    head = heads[0]
    history = observation["policy"][..., HISTORY_START:HISTORY_STOP]
    center, request_evidence = history, None
    if type(actor) in (SemanticCapTransitionQuarterHistoryMLPModel, SemanticFRKneePhysicalInnovationHistoryMLPModel,
                      SemanticTaskConditionedHipWheelHistoryMLPModel, SemanticReceivingWheelSigmaHistoryMLPModel):
        center, request_evidence = cap_transition_request_history(observation["policy"])
    if type(actor) in (SemanticP05CaptureHistoryMLPModel,SemanticRRCaptureHistoryMLPModel, SemanticRearPolicyTimingHistoryMLPModel, SemanticP02ProgressHistoryMLPModel, SemanticRearCooperativePrepHistoryMLPModel, SemanticRearOwnerRecoveryHistoryMLPModel):
        center, request_evidence = p05_capture_request_history(observation["policy"][...,:389])
    conditional = history_conditioned_head(head, center, HISTORY_RHO)
    sigma_multiplier = None
    task_sigma_evidence = None
    effective_log_std = head[..., 1, :] + math.log(actor.exploration_std_temperature)
    if type(actor) is SemanticFRKneePhysicalInnovationHistoryMLPModel:
        effective_log_std, sigma_multiplier = physical_innovation_effective_log_std(
            head[..., 1, :], observation["policy"], actor.exploration_std_temperature)
    if type(actor) is SemanticTaskConditionedHipWheelHistoryMLPModel:
        effective_log_std, task_sigma_evidence = task_conditioned_effective_log_std(
            head[...,1,:], observation["policy"], actor.exploration_std_temperature)
    if type(actor) is SemanticReceivingWheelSigmaHistoryMLPModel:
        effective_log_std, task_sigma_evidence = receiving_wheel_effective_log_std(
            head[...,1,:], observation["policy"], actor.exploration_std_temperature)
    if type(actor) in (SemanticP05CaptureHistoryMLPModel,SemanticRRCaptureHistoryMLPModel, SemanticRearPolicyTimingHistoryMLPModel, SemanticP02ProgressHistoryMLPModel, SemanticRearCooperativePrepHistoryMLPModel, SemanticRearOwnerRecoveryHistoryMLPModel):
        effective_log_std, task_sigma_evidence = receiving_wheel_effective_log_std(
            head[...,1,:], observation["policy"][...,:372], actor.exploration_std_temperature)
    effective_std = effective_log_std.exp()
    if type(actor) in (SemanticRearPolicyTimingHistoryMLPModel, SemanticP02ProgressHistoryMLPModel, SemanticRearCooperativePrepHistoryMLPModel, SemanticRearOwnerRecoveryHistoryMLPModel):
        sigma_kernel = (rear_owner_effective_log_std if type(actor) is SemanticRearOwnerRecoveryHistoryMLPModel
                        else cooperative_prep_effective_log_std if type(actor) is SemanticRearCooperativePrepHistoryMLPModel
                        else p02_progress_effective_log_std if type(actor) is SemanticP02ProgressHistoryMLPModel
                        else rear_policy_timing_effective_log_std)
        effective_log_std, task_sigma_evidence = sigma_kernel(
            head[...,1,:], observation["policy"], actor.exploration_std_temperature)
        effective_std = effective_log_std.exp()
    if stochastic:
        mean, std = actor.output_distribution_params
        if not torch.equal(mean, conditional[..., 0, :]) or not torch.equal(std, effective_std):
            raise RuntimeError("sampled distribution disagrees with the actual head and current history")
    elif not torch.equal(raw, conditional[..., 0, :]):
        raise RuntimeError("deterministic action differs from the actual conditional mean")
    vector = lambda value: value[0].detach().cpu().tolist()
    record = {"schema": "wlr50_clean.actual_history_policy_request.v1",
        "mode": "training_style_conditional_gaussian" if stochastic else "deterministic_conditional_mean",
        "base_mean_full12": vector(head[..., 0, :]),
        "conditional_mean_full12": vector(conditional[..., 0, :]),
        "previous_raw_from_current_observation_full12": vector(history),
        "learned_sigma_full12": vector(head[..., 1, :].exp()),
        "effective_sigma_full12": vector(effective_std), "rho": HISTORY_RHO,
        "exploration_std_temperature": actor.exploration_std_temperature,
        "selected_raw_full12": vector(raw), "selected_tanh_full12": vector(raw.tanh()),
        "sampling_draws": 1 if stochastic else 0, "extra_model_forwards": 0, "extra_random_draws": 0,
        "physical_scale_and_execution_source": "same_decision_applied_audit_and_native_tick_audit"}
    if stochastic:
        record["selected_raw_log_probability"] = float(actor.get_output_log_prob(raw)[0])
    if request_evidence is not None:
        record.update(schema="wlr50_clean.actual_cap_transition_history_policy_request.v1",
            policy_version=(FR_KNEE_PHYSICAL_INNOVATION_POLICY if sigma_multiplier is not None
                            else HISTORY_REQUEST_CAP_TRANSITION_POLICY),
            history_center_semantics=REQUEST_HISTORY_SEMANTICS,
            history_center_full12=vector(center),
            cap_transition_gate_full12=vector(request_evidence["gate_full12"]),
            stage_index=int(request_evidence["stage_index"][0]),
            encoded_stage_age=float(request_evidence["encoded_stage_age"][0]),
            predecessor_completed=bool(request_evidence["predecessor_completed"][0]),
            current_cap_full12=vector(request_evidence["current_cap_full12"]),
            predecessor_cap_full12=vector(request_evidence["predecessor_cap_full12"]),
            previous_filtered_request_full12=vector(request_evidence["previous_filtered_request_full12"]),
            request_history_scope="previous_filtered_REQUEST_not_effective_final_drive_or_actual_motion")
    if sigma_multiplier is not None:
        record.update(schema="wlr50_clean.actual_physical_innovation_history_policy_request.v1",
            sigma_scaling_semantics=FR_KNEE_PHYSICAL_SIGMA_SEMANTICS,
            innovation_sigma_multiplier_full12=vector(sigma_multiplier),
            sigma_scaling_gate_full12=vector(sigma_multiplier != 1),
            effective_log_std_full12=vector(effective_log_std))
    if task_sigma_evidence is not None:
        record.update(schema="wlr50_clean.actual_task_conditioned_hip_wheel_policy_request.v1",
            policy_version=TASK_CONDITIONED_HIP_WHEEL_POLICY,
            sigma_scaling_semantics=TASK_CONDITIONED_HIP_WHEEL_SIGMA_SEMANTICS,
            effective_log_std_full12=vector(effective_log_std),
            physical_equivalent_B_full12=vector(task_sigma_evidence["physical_equivalent_B_full12"]),
            innovation_sigma_multiplier_full12=vector(task_sigma_evidence["innovation_sigma_multiplier_full12"]),
            task_state_weights={key:float(value[0]) for key,value in task_sigma_evidence["task_state_weights"].items()},
            front_support_proxy_not_exact_TOP=bool(task_sigma_evidence["front_support_proxy_not_exact_TOP"][0]),
            current_RR_qualification=bool(task_sigma_evidence["current_RR_qualification"][0]),
            current_rear_front_distance_m=float(task_sigma_evidence["current_rear_front_distance_m"][0]),
            task_state_audit_topology="one_actual_N1_request_not_a_batched_N8_summary")
    if type(actor) in (SemanticReceivingWheelSigmaHistoryMLPModel,SemanticP05CaptureHistoryMLPModel,SemanticRRCaptureHistoryMLPModel, SemanticRearPolicyTimingHistoryMLPModel, SemanticP02ProgressHistoryMLPModel, SemanticRearCooperativePrepHistoryMLPModel, SemanticRearOwnerRecoveryHistoryMLPModel):
        record.update(schema="wlr50_clean.actual_receiving_wheel_sigma_policy_request.v1",
            policy_version=RECEIVING_WHEEL_POLICY, sigma_scaling_semantics=RECEIVING_WHEEL_SIGMA_SEMANTICS,
            receiving_continuation_active=bool(task_sigma_evidence["receiving_continuation_active"][0]),
            RR_placed_history=bool(task_sigma_evidence["RR_placed_history"][0]),
            receiving_sigma_gate_full12=vector(task_sigma_evidence["receiving_sigma_gate_full12"]),
            receiving_sigma_multiplier_full12=vector(task_sigma_evidence["receiving_sigma_multiplier_full12"]),
            effective_innovation_sigma_multiplier_full12=vector(task_sigma_evidence["effective_innovation_sigma_multiplier_full12"]),
            receiving_state_semantics="historical_RR_placed_continuation_or_recovery_not_current_bearing")
    if type(actor) in (SemanticP05CaptureHistoryMLPModel,SemanticRRCaptureHistoryMLPModel, SemanticRearPolicyTimingHistoryMLPModel, SemanticP02ProgressHistoryMLPModel, SemanticRearCooperativePrepHistoryMLPModel, SemanticRearOwnerRecoveryHistoryMLPModel):
        record.update(schema="wlr50_clean.actual_p05_capture_assist_policy_request.v1",
            policy_version=P05_CAPTURE_POLICY,history_center_semantics=P05_CAPTURE_HISTORY_SEMANTICS,
            capture_assist_observed_features=vector(observation["policy"][...,372:384]),
            capture_continuation_observed_features=vector(observation["policy"][...,384:389]),
            pending_scheduler_handoff=bool(request_evidence["pending_scheduler_handoff"][0]),
            physical_predecessor_completed=bool(request_evidence["physical_predecessor_completed"][0]),
            transformed_actuator_targets_are_not_policy_samples=True)
    if type(actor) in (SemanticRRCaptureHistoryMLPModel, SemanticRearPolicyTimingHistoryMLPModel, SemanticP02ProgressHistoryMLPModel, SemanticRearCooperativePrepHistoryMLPModel, SemanticRearOwnerRecoveryHistoryMLPModel):
        record.update(schema="wlr50_clean.actual_rr_capture_transfer_policy_request.v1",policy_version=RR_CAPTURE_POLICY,
            rr_capture_assist_observed_features=vector(observation["policy"][...,RR_ASSIST_START:RR_TASK_START]),
            rr_capture_transfer_observed_features=vector(observation["policy"][...,RR_TASK_START:RR_CAPTURE_OBSERVATION_DIM]))
    if type(actor) in (SemanticRearPolicyTimingHistoryMLPModel, SemanticP02ProgressHistoryMLPModel, SemanticRearCooperativePrepHistoryMLPModel, SemanticRearOwnerRecoveryHistoryMLPModel):
        record.update(schema="wlr50_clean.actual_rear_policy_timing_request.v1",
            policy_version=REAR_POLICY_TIMING_POLICY, sigma_scaling_semantics=REAR_POLICY_TIMING_SIGMA_SEMANTICS,
            rear_policy_timing_observed_features=vector(observation["policy"][...,410:419]),
            rear_local_sigma_multiplier_full12=vector(task_sigma_evidence["rear_local_sigma_multiplier_full12"]),
            rear_task_assists_enabled=False)
    if type(actor) in (SemanticP02ProgressHistoryMLPModel, SemanticRearCooperativePrepHistoryMLPModel, SemanticRearOwnerRecoveryHistoryMLPModel):
        record.update(schema='wlr50_clean.actual_p02_progress_request.v1',policy_version=P02_PROGRESS_POLICY,
            p02_progress_observed_features=vector(observation['policy'][...,419:422]),
            sigma_kernel_observation_slice=[0,419])
    if type(actor) in (SemanticRearCooperativePrepHistoryMLPModel, SemanticRearOwnerRecoveryHistoryMLPModel):
        record.update(schema='wlr50_clean.actual_rear_cooperative_prep_request.v1',
            policy_version=COOPERATIVE_PREP_POLICY, sigma_scaling_semantics=COOPERATIVE_PREP_SIGMA_SEMANTICS,
            sigma_kernel_observation_slice=[0,422],
            cooperative_sigma_mode=task_sigma_evidence['cooperative_sigma_mode'],
            cooperative_observed_rr_carry_capture=bool(task_sigma_evidence['cooperative_observed_rr_carry_capture'][0]),
            cooperative_observed_rr_top_reachable=bool(task_sigma_evidence['cooperative_observed_rr_top_reachable'][0]),
            cooperative_observed_rl_prep_transfer=bool(task_sigma_evidence['cooperative_observed_rl_prep_transfer'][0]),
            cooperative_parent_receiving_continuation_active=bool(task_sigma_evidence['cooperative_parent_receiving_continuation_active'][0]),
            cooperative_prep_allowed=bool(task_sigma_evidence['cooperative_prep_allowed'][0]),
            cooperative_fl_wheel_extra_active=bool(task_sigma_evidence['cooperative_fl_wheel_extra_active'][0]),
            cooperative_multiplier_reference=task_sigma_evidence['cooperative_multiplier_reference'],
            cooperative_support_transfer_permission_modified=False,
            cooperative_target_or_mean_modified=False)
    if type(actor) is SemanticRearOwnerRecoveryHistoryMLPModel:
        record.update(schema='wlr50_clean.actual_rear_owner_recovery_request.v1',
            policy_version=REAR_OWNER_POLICY,
            rear_owner_observed_features=vector(observation['policy'][...,422:439]),
            owner_transform='explicit_issued_owner_suspension_not_rear_task_teacher')
    return raw, record


def _semantic_decision_audit_row(*, global_decision: int, index: int,
                                 sampled_raw: Any, old_mean: Any, old_std: Any,
                                 old_log_prob: Any, old_value: Any,
                                 reward: Any, terminal: bool, info: Any) -> dict[str, Any]:
    return {"global_policy_decision": global_decision,
            "raw_policy_action_full12": sampled_raw[index].cpu().tolist(),
            "old_distribution_mean_full12": old_mean[index].cpu().tolist(),
            "old_distribution_std_full12": old_std[index].cpu().tolist(),
            "old_log_probability": float(old_log_prob[index]), "old_value": float(old_value[index]),
            "reward": float(reward), "terminal": terminal, "applied_audit": info}


def _rollout_advantage_audit(snapshot: Mapping[str, Any], requests: Any, *,
                             first_global_decision: int, ppo_update: int,
                             gamma: float, gae_lambda: float,
                             normalize_advantage_per_mini_batch: bool) -> dict[str, Any]:
    """Read the existing post-returns CPU snapshot; no forward or storage writes."""
    import torch
    names = ("returns", "values", "advantages", "rewards", "dones")
    data = {key: snapshot[key] for key in names}
    shape = tuple(data["returns"].shape)
    if (len(shape) != 3 or shape[-1] != 1 or shape[0] < 1 or shape[1] < 1
            or any(tuple(value.shape) != shape or value.device.type != "cpu" for value in data.values())):
        raise ValueError("advantage audit requires the existing [T,N,1] CPU rollout snapshot")
    steps, environments, _ = shape
    if len(requests) != steps or any(len(row) != environments for row in requests):
        raise ValueError("actual decision request mapping differs from rollout topology")
    flat_requests = [row for tick in requests for row in tick]
    phases = [row.get("phase_id") for row in flat_requests]
    if any(phase not in tuple(f"P{i:02d}" for i in range(1, 14)) for phase in phases):
        raise ValueError("advantage audit requires each actual request phase, not curriculum phase")
    # Subtract in storage dtype exactly as official PPO does before normalizing.
    # This is raw GAE reconstructed from saved returns and old values, not a
    # fresh critic estimate, episode return, or normalized advantage rescaling.
    metrics = {"raw_gae_returns_minus_old_values": data["returns"]-data["values"],
               "stored_advantages": data["advantages"], "old_values": data["values"],
               "returns": data["returns"], "rewards": data["rewards"]}
    metrics = {key: value.detach().reshape(-1) for key, value in metrics.items()}
    dones = data["dones"].detach().reshape(-1).bool()

    def stats(value):
        finite_mask = torch.isfinite(value)
        finite = value[finite_mask].double()
        return {"count": value.numel(), "finite_count": finite.numel(),
                "nonfinite_count": int((~finite_mask).sum()),
                "minimum": float(finite.min()) if finite.numel() else None,
                "maximum": float(finite.max()) if finite.numel() else None,
                "mean": float(finite.mean()) if finite.numel() else None,
                "std_population": float(finite.std(unbiased=False)) if finite.numel() else None,
                "positive_count": int((finite > 0).sum()),
                "negative_count": int((finite < 0).sum()), "zero_count": int((finite == 0).sum())}

    def summary(indices):
        index = torch.tensor(indices, dtype=torch.long)
        return {"sample_count": len(indices), "terminal_count": int(dones[index].sum()),
                **{key: stats(value[index]) for key, value in metrics.items()}}

    terminal_rows, phase_changes = [], []
    for index, row in enumerate(flat_requests):
        ending, count = row.get("physics_tick"), row.get("physics_ticks")
        start = ending-count if type(ending) is int and type(count) is int and 1 <= count <= 8 else None
        coordinate = {"global_policy_decision": first_global_decision+index,
                      "rollout_step": index//environments, "environment_index": index%environments,
                      "request_phase": phases[index], "end_phase": row.get("end_phase_id"),
                      "episode_start_tick": start, "episode_end_tick": ending,
                      "executed_physics_ticks": count, "sim_time_s": row.get("sim_time_s"),
                      "terminal": bool(dones[index]),
                      "termination_reason": row.get("termination_reason"),
                      "terminal_bootstrap_allowed": row.get("terminal_bootstrap_allowed")}
        if dones[index]:
            terminal_rows.append({**coordinate, **{key: (float(value[index])
                if torch.isfinite(value[index]) else None) for key, value in metrics.items()}})
        if not bool(dones[index]) and row.get("end_phase_id") not in (None, phases[index]):
            phase_changes.append(coordinate)
    return {"schema": "wlr50_clean.semantic_preupdate_advantage_audit.v1",
            "ppo_update_intended": ppo_update, "collection_status": "complete_rollout_pre_update",
            "optimizer_update_completed_by_this_record": False,
            "first_global_policy_decision": first_global_decision,
            "last_global_policy_decision": first_global_decision+steps*environments-1,
            "rollout_steps": steps, "num_envs": environments,
            "sample_count": steps*environments, "teacher_prefix_samples_included": False,
            "phase_mapping": "actual_decision_request_phase_in_storage_time_env_order",
            "gamma": gamma, "lambda": gae_lambda,
            "discount_clock": "once_per_issued_policy_action_even_for_short_terminal_interval",
            "stored_advantage_semantics": ("raw_GAE_before_official_minibatch_normalization"
                if normalize_advantage_per_mini_batch else "official_whole_rollout_standardized_GAE"),
            "std_statistic": "population_std_for_reporting_only; official_normalization_unchanged",
            "overall": summary(list(range(steps*environments))),
            "by_request_phase": {phase: summary([i for i, value in enumerate(phases) if value == phase])
                                 for phase in sorted(set(phases))},
            "terminal_samples": terminal_rows, "ordinary_phase_change_samples": phase_changes,
            "tail_bootstrap": {"terminal_env_indices": [i for i in range(environments) if bool(data["dones"][-1, i, 0])],
                "nonterminal_env_indices": [i for i in range(environments) if not bool(data["dones"][-1, i, 0])],
                "value_source": "existing_official_compute_returns_last_values; not re-evaluated or logged here",
                "last_values_measured_separately": None}}

def train_semantic(runner: Any, env: SemanticRslAdapter, *, run_dir: Path,
                   output_root: Path, stage: str, decisions: int,
                   contract: Mapping[str, Any], seed: int, resume_infos: Mapping[str, Any] | None = None,
                   checkpoint_interval_updates: int = 10,
                   checkpoint_output_routing: Mapping[str, Any] | None = None) -> dict[str, Any]:
    import torch
    if stage not in STAGE_BUDGETS or type(decisions) is not int or decisions < 1:
        raise ValueError("invalid semantic training stage/decision request")
    if checkpoint_interval_updates < 1:
        raise ValueError("checkpoint cadence must be positive")
    previous = dict(resume_infos or {})
    rear_timing = "rear_policy_timing_migration" in previous
    if rear_timing:
        from .semantic_rear_policy_timing_migration import validate_rear_policy_namespace
        validate_rear_policy_namespace(previous, contract, output_root,
            checkpoint_output_routing=checkpoint_output_routing)
        if "rear_recapture_migration" in previous and checkpoint_output_routing is None:
            raise ValueError("published rear419 recapture ancestor training requires its explicit output branch")
    inherited_routing = previous.get("checkpoint_output_routing")
    capture_reserve = "rr_capture_reserve_v10_migration" in previous
    if not rear_timing and capture_reserve and checkpoint_output_routing is None:
        raise ValueError("v10 learned continuation requires its explicit output routing branch")
    postcapture_wheel = "rr_postcapture_wheel_v9_migration" in previous
    if not rear_timing and postcapture_wheel and checkpoint_output_routing is None:
        raise ValueError("v9 learned continuation requires its explicit output routing branch")
    wheel_signed = "rr_signed_wheel_v8_migration" in previous
    signed = "rr_signed_contact_v7_migration" in previous
    contact_receipt = previous.get("rr_signed_wheel_v8_migration" if wheel_signed else "rr_signed_contact_v7_migration" if signed else "rr_contact_onset_v6_migration") or {}
    contact_schema, contact_factor, contact_feedback = (
        ("wlr50_clean.rr_signed_wheel_same410.v8", "rr_signed_wheel_v8_factor", "signed_band_contact_formation_incremental_v6") if wheel_signed else
        ("wlr50_clean.rr_signed_contact_same410.v7", "rr_signed_contact_v7_factor", "signed_band_contact_formation_incremental_v6") if signed else
        ("wlr50_clean.rr_contact_onset_same410.v6", "rr_contact_onset_v6_factor", "progress_reserve_contact_onset_incremental_v5"))
    if (not rear_timing and checkpoint_output_routing is None and any((previous.get(key) or {}).get("source_selection", {}).get(
            "source_role") == "front_validated_ancestor_control_eval" for key in (
                "rr_contact_onset_v6_migration", "rr_signed_contact_v7_migration", "rr_signed_wheel_v8_migration"))):
        raise ValueError("published ancestor training requires its explicit output routing branch")
    if checkpoint_output_routing is not None:
        from .semantic_migration import digest as metadata_digest
        route = jsonable(checkpoint_output_routing)
        selection = route.get("source_selection")
        if rear_timing:
            validate_rear_policy_namespace(previous, contract, output_root,checkpoint_output_routing=route)
        elif capture_reserve:
            from .semantic_rr_capture_reserve_migration import validate_v10_branch_receipt
            if route.get("output_root") != str(output_root.resolve()) or inherited_routing != route:
                raise ValueError("v10 destination must remain the inherited output branch")
            validate_v10_branch_receipt(previous, contract, route)
        elif postcapture_wheel:
            from .semantic_rr_postcapture_wheel_migration import validate_v9_branch_receipt
            if route.get("output_root") != str(output_root.resolve()) or inherited_routing != route:
                raise ValueError("v9 destination must remain the inherited output branch")
            validate_v9_branch_receipt(previous, contract, route)
        elif (route.get("schema") != "wlr50_clean.checkpoint_output_routing.v1"
                or route.get("output_root") != str(output_root.resolve())
                or output_root.resolve().parent.name != "branches"
                or output_root.resolve().name != route.get("branch")
                or route.get("main_latest_pointer_promotion") is not False
                or not isinstance(selection, dict)
                or selection != (previous.get("rr_progress_handoff_v5_migration") or {}).get("source_selection")
                or selection.get("source_role") != "front_validated_ancestor_control_eval"
                or contact_receipt.get("schema") != contact_schema
                or contact_receipt.get("target_contract_sha256") != metadata_digest(contract)
                or contact_receipt.get("target_git_commit") != contract.get("source_git_commit")
                or contact_receipt.get("target_runtime_content_sha256") != contract.get("runtime_content_sha256")
                or contact_receipt.get("source_selection", {}).get("source_role") != selection.get("source_role")
                or contact_receipt.get(contact_factor, {}).get("counter_origin") != selection.get("counters")
                or contact_receipt.get(contact_factor, {}).get("target_feedback_revision") != contact_feedback
                or (inherited_routing is not None and inherited_routing != route)):
            raise ValueError("checkpoint output routing differs from the loaded ancestor state or explicit destination")
    else:
        route = None
        if inherited_routing is not None:
            raise ValueError("branch checkpoint continuation requires its explicit output routing")
    assert_semantic_return_consistency(runner, env)
    sampling = jsonable(env.cfg.get("reset_sampling", "P01_only"))
    prefix_request = jsonable(env.cfg.get("prefix_request"))
    curriculum = semantic_curriculum_epoch(env.cfg)
    defer_tail_reset = _supports_deferred_terminal_reset(runner, env)
    semantic_version = env.cfg.get("semantic_version", "v2")
    budgets = training_quantity_budgets(contract.get("experiment_id"))
    declared_budgets = contract.get("training_budgets", budgets)
    if (declared_budgets != budgets or any(type(value) is not int for value in declared_budgets.values())
            or (contract.get("experiment_id") in ("task_conditioned_hip_wheel_v1","p05_hip_only_continuation_v1","rr_capture_then_rl_transfer_v1", "rr_rl_timing_policy_learning_v1")
                and "training_budgets" not in contract)):
        raise ValueError("training quantity budgets differ from the explicit experiment declaration")
    stage_spent = {name: int(previous.get("stage_requested_decisions", {}).get(name, 0)) for name in STAGE_BUDGETS}
    if stage_spent[stage] + decisions > budgets[stage]:
        raise ValueError("additional request exceeds the remaining semantic stage budget")
    base_global = int(previous.get("global_policy_decisions", 0))
    base_updates = int(previous.get("ppo_updates", 0))
    base_optimizer = int(previous.get("optimizer_steps", 0))
    batch = int(runner.cfg["num_steps_per_env"]) * env.num_envs
    iterations = (decisions + batch - 1) // batch
    if (route is not None or rear_timing) and any((output_root / "checkpoints/history" /
            f"checkpoint_step_{base_global+(i+1)*batch:09d}.pt").exists() for i in range(iterations)):
        raise FileExistsError("branch continuation would collide with an existing checkpoint; no optimizer step started")
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
         (run_dir / "completed_episodes.jsonl").open("x", encoding="utf-8") as episode_stream, \
         (run_dir / "advantage_audit.jsonl").open("x", encoding="utf-8") as advantage_stream:
        for iteration in range(iterations):
            rollout_requests = []
            with torch.inference_mode():
                if defer_tail_reset and env.episode_reset_pending:
                    if semantic_curriculum_epoch(env.cfg) != curriculum:
                        raise RuntimeError("curriculum changed before deferred terminal reset")
                    # No next policy sample may be drawn from terminal obs.
                    # Previous complete update/checkpoint is already durable.
                    obs = env.reset_pending_episode().to(runner.device)
                    if semantic_curriculum_epoch(env.cfg) != curriculum:
                        raise RuntimeError("curriculum changed during deferred terminal reset")
                    assert_semantic_return_consistency(runner, env)
                for tick in range(int(runner.cfg["num_steps_per_env"])):
                    if semantic_curriculum_epoch(env.cfg) != curriculum:
                        raise RuntimeError("curriculum must remain fixed throughout this on-policy epoch")
                    assert_semantic_return_consistency(runner, env)
                    policy_request = None
                    if contract.get("experiment_id") in ("fl_capture_quality_v1", "task_conditioned_hip_wheel_v1", "p05_hip_only_continuation_v1", "rr_capture_then_rl_transfer_v1", "rr_rl_timing_policy_learning_v1"):
                        raw, policy_request = audited_history_policy_request(
                            runner.alg.actor, obs, lambda: runner.alg.act(obs), stochastic=True)
                    else:
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
                    terminal_persisted = False

                    def persist_terminal(summary: Any, final_observation: Any, terminal_reward: Any) -> None:
                        nonlocal terminal_persisted
                        if terminal_persisted:
                            raise RuntimeError("duplicate terminal evidence callback for one policy decision")
                        if any(not bool(torch.isfinite(value).all())
                               for value in (sampled_raw, old_log_prob, old_value, terminal_reward)):
                            raise RuntimeError("non-finite on-policy terminal transition")
                        row = _semantic_decision_audit_row(
                            global_decision=base_global + iteration * batch + tick + 1, index=0,
                            sampled_raw=sampled_raw, old_mean=old_mean, old_std=old_std,
                            old_log_prob=old_log_prob, old_value=old_value,
                            reward=terminal_reward[0], terminal=True, info=summary["terminal_info"])
                        if policy_request is not None:
                            row["policy_request"] = policy_request
                        row["terminal_observation"] = {
                            key: jsonable(value) for key, value in final_observation.items()}
                        # Serialize both before either write; fsync both before
                        # allowing the potentially long or failing next prefix.
                        audit_line = json.dumps(row, allow_nan=False) + "\n"
                        episode_line = json.dumps(summary, allow_nan=False) + "\n"
                        audit_stream.write(audit_line)
                        episode_stream.write(episode_line)
                        for stream in (audit_stream, episode_stream):
                            stream.flush()
                            os.fsync(stream.fileno())
                        terminal_persisted = True

                    with _terminal_evidence_before_reset(
                            env, persist_terminal,
                            defer_terminal_reset=(defer_tail_reset
                                and tick == int(runner.cfg["num_steps_per_env"]) - 1)):
                        obs, rewards, dones, extras = env.step(raw.to(env.device))
                    if semantic_curriculum_epoch(env.cfg) != curriculum:
                        raise RuntimeError("curriculum changed during a physical step/reset of the on-policy epoch")
                    assert_semantic_return_consistency(runner, env)
                    if any(not bool(torch.isfinite(value).all()) for value in (sampled_raw, old_log_prob, old_value, rewards)):
                        raise RuntimeError("non-finite on-policy transition")
                    if bool(extras["time_outs"].any()):
                        raise RuntimeError("task finite-horizon termination must not bootstrap")
                    # Actual requests from this issued step only; teacher roll-in
                    # and the configured reset/curriculum phase are not PPO rows.
                    rollout_requests.append([{key: info.get(key) for key in (
                        "phase_id", "end_phase_id", "physics_tick", "physics_ticks", "sim_time_s",
                        "termination_reason", "terminal_bootstrap_allowed")}
                        for info in extras["semantic_decisions"]])
                    for index, info in enumerate(extras["semantic_decisions"]):
                        if terminal_persisted:
                            continue
                        row = _semantic_decision_audit_row(
                            global_decision=base_global + iteration * batch + tick * env.num_envs + index + 1,
                            index=index, sampled_raw=sampled_raw, old_mean=old_mean, old_std=old_std,
                            old_log_prob=old_log_prob, old_value=old_value,
                            reward=rewards[index], terminal=bool(dones[index]), info=info)
                        if policy_request is not None:
                            row["policy_request"] = policy_request
                        audit_stream.write(json.dumps(row, allow_nan=False) + "\n")
                    for episode in extras["episode_summaries"]:
                        if not terminal_persisted:
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
            snapshot["curriculum_epoch"] = copy.deepcopy(curriculum)
            if storage.observations["policy"].shape[-1] in (324, 372, P05_CAPTURE_OBSERVATION_DIM, RR_CAPTURE_OBSERVATION_DIM, REAR_POLICY_TIMING_OBSERVATION_DIM, 422, 439):
                snapshot["policy_contract"] = _runner_policy_contract(runner)
            advantage_row = _rollout_advantage_audit(snapshot, rollout_requests,
                first_global_decision=base_global+iteration*batch+1,
                ppo_update=base_updates+iteration+1, gamma=runner.alg.gamma,
                gae_lambda=runner.alg.lam,
                normalize_advantage_per_mini_batch=runner.alg.normalize_advantage_per_mini_batch)
            advantage_stream.write(json.dumps(advantage_row, allow_nan=False) + "\n")
            advantage_stream.flush()
            os.fsync(advantage_stream.fileno())
            torch.save(snapshot, rollout_dir / f"rollout_{base_updates + iteration + 1:06d}.pt")
            global_step = base_global + (iteration + 1) * batch
            runner.alg.entropy_coef = 0.005 + (0.001 - 0.005) * min(global_step / sum(STAGE_BUDGETS.values()), 1.0)
            if contract.get("experiment_id") in ("task_first_recovery_v1", "residual_rr_fix_v1", "fl_capture_quality_v1", "task_conditioned_hip_wheel_v1", "p05_hip_only_continuation_v1", "rr_capture_then_rl_transfer_v1", "rr_rl_timing_policy_learning_v1"):
                update = audited_ppo_update(runner, likelihood_audit_path=rollout_dir /
                    f"update_{base_updates + iteration + 1:06d}_likelihood.json")
            else:
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
            if (stop_record is not None or iteration == 0
                    or (iteration + 1) % checkpoint_interval_updates == 0 or iteration + 1 == iterations
                    or (defer_tail_reset and env.episode_reset_pending)):
                spent = dict(stage_spent)
                spent[stage] += min((iteration + 1) * batch, decisions)
                from .semantic_migration import topology
                infos = {"runtime_contract": dict(contract), "seed": seed, "stage": stage,
                         "execution_topology": topology(env.num_envs,
                             observation_layout=getattr(runner, "_semantic_observation_layout", None)),
                         "phase_suffix_curriculum_implemented": prefix_request is not None,
                         "semantic_version": semantic_version,
                         "curriculum_epoch": {**copy.deepcopy(curriculum),
                                              "changes_allowed_only_between_complete_rollout_updates": True},
                         "implemented_reset_sampling": env.cfg.get("reset_sampling", "P01_only"),
                         "vector_smoke_evidence": env.cfg.get("vector_smoke_evidence"),
                         "global_policy_decisions": global_step, "ppo_updates": base_updates + iteration + 1,
                         "optimizer_steps": base_optimizer + sum(row["optimizer_steps"] for row in updates),
                         "stage_requested_decisions": spent, "source_run": str(run_dir.resolve()),
                         "sampling": sampling, "runner_config": copy.deepcopy(runner._semantic_runner_config),
                         "last_update": update}
                if route is not None:
                    infos["checkpoint_output_routing"] = copy.deepcopy(route)
                if semantic_version == "v3":
                    from .semantic_migration import continuation_topology
                    infos["execution_topology"] = continuation_topology(sampling, prefix_request,
                        observation_layout=getattr(runner, "_semantic_observation_layout", None))
                for key in ("front_retention439_runtime_identity", "front_retention439_auxiliary", "rr_retention_reward_migration", "rear_owner_recovery_migration", "cooperative_prep_migration", "p02_progress_migration", "rear_live_swing_migration", "rear_recapture_migration", "rear_policy_timing_migration", "rear_policy_timing_branch",
                            "new_mdp_warm_start", "new_mdp_origin_global_policy_decisions", "source_stage_requested_decisions",
                            "new_mdp_initial_action_comparison", "policy_distribution_migration",
                            "policy_distribution_migration_evidence", "new_mdp_initial_policy_kernel_comparison",
                            "observation_scale_compensation_evidence", "observation_append_evidence",
                            "task_recovery_branch", "rr_task_branch", "fl_capture_quality_branch",
                            "task_conditioned_hip_wheel_branch", "training_quantity_budget_extension",
                            "receiving_wheel_sigma_migration", "p05_capture_assist_migration", "p05_capture_assist_branch",
                            "rr_capture_transfer_migration", "rr_capture_transfer_branch",
                            "rr_capture_feedback_peak_v2_migration",
                            "rr_capture_knee_v3_migration",
                            "rr_carry_handoff_v4_migration",
                            "rr_progress_handoff_v5_migration",
                            "rr_contact_onset_v6_migration",
                            "rr_signed_contact_v7_migration",
                            "rr_signed_wheel_v8_migration",
                            "rr_postcapture_wheel_v9_migration",
                            "rr_capture_reserve_v10_migration",
                            "capture_feedback_semantics_migration", "capture_feedback_semantics_branch",
                            "rr_postcross_workspace_migration", "rr_postcross_workspace_branch",
                            "rr_receiver_retirement_v2_migration", "rr_receiver_retirement_v2_branch",
                            "p05_preedge_approach_recovery_migration", "p05_preedge_approach_recovery_branch"):
                    if key in previous:
                        infos[key] = previous[key]
                if "task_recovery_branch" in infos:
                    origin = infos["task_recovery_branch"]["counter_origin"]
                    infos["task_recovery_branch_counts"] = {
                        key: int(infos[key]) - int(origin[key])
                        for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps")}
                if "rr_task_branch" in infos:
                    origin = infos["rr_task_branch"]["counter_origin"]
                    infos["rr_task_branch_counts"] = {
                        key: int(infos[key]) - int(origin[key])
                        for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps")}
                if "fl_capture_quality_branch" in infos:
                    origin = infos["fl_capture_quality_branch"]["counter_origin"]
                    infos["fl_capture_quality_branch_counts"] = {
                        key: int(infos[key]) - int(origin[key])
                        for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps")}
                if "task_conditioned_hip_wheel_branch" in infos:
                    origin = infos["task_conditioned_hip_wheel_branch"]["counter_origin"]
                    infos["task_conditioned_hip_wheel_branch_counts"] = {
                        key:int(infos[key])-int(origin[key])
                        for key in ("global_policy_decisions","ppo_updates","optimizer_steps")}
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
    if route is not None:
        result["checkpoint_output_routing"] = copy.deepcopy(route)
    result["runner_config"] = copy.deepcopy(runner._semantic_runner_config)
    if runner.alg.storage.observations["policy"].shape[-1] in (324, 372, P05_CAPTURE_OBSERVATION_DIM, RR_CAPTURE_OBSERVATION_DIM, REAR_POLICY_TIMING_OBSERVATION_DIM, 422, 439):
        result["policy_contract"] = _runner_policy_contract(runner)
    result["curriculum_epoch"] = copy.deepcopy(curriculum)
    result["implemented_sampling"] = env.cfg.get("reset_sampling", "P01_only")
    if gpu_probe is not None:
        gpu_probe.sample("after_training_and_verified_checkpoint")
        result["gpu_measurements"] = gpu_probe.summary()
    write_json(run_dir / "training_manifest.json", result)
    return result

"""Versioned, observation-conditioned Gaussian mean; no private action history.

The existing normalized observation supplies the clipped previous raw action.
Sigma remains the learned *conditional innovation* standard deviation. This is
not a claim of stationary marginal variance or of improved physical control.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from rsl_rl.utils import unpad_trajectories
from .semantic_transfer_roles import (
    ROLE_OBSERVATION_LAYOUT, ROLE_OBSERVATION_BASE_DIM, ROLE_OBSERVATION_DIM,
)

HISTORY_RHO = 0.9
HISTORY_START = 195
HISTORY_STOP = 207
HISTORY_CLIP = 20.0


def history_conditioned_head(
    head: torch.Tensor, history: torch.Tensor, rho: float = HISTORY_RHO,
) -> torch.Tensor:
    """Return a same-layout [..., 2, 12] head, leaving log sigma unchanged.

    ``history`` is the already schema-clipped observation feature, not a cache or
    reconstructed projected action. Reject out-of-contract inputs rather than
    silently adding a second observation transformation. Rho zero returns the
    original tensor, including signed zeros and its exact sampling/RNG path.
    """
    if isinstance(rho, bool) or not isinstance(rho, Real):
        raise ValueError("rho must be a finite real in [0,1)")
    try:
        ratio = float(rho)
    except (OverflowError, ValueError) as error:
        raise ValueError("rho must be a finite real in [0,1)") from error
    if not math.isfinite(ratio) or not 0 <= ratio < 1:
        raise ValueError("rho must be a finite real in [0,1)")
    if (not isinstance(head, torch.Tensor) or not isinstance(history, torch.Tensor)
            or not head.is_floating_point() or not history.is_floating_point()
            or head.ndim < 2 or tuple(head.shape[-2:]) != (2, 12)
            or tuple(history.shape) != (*head.shape[:-2], 12)
            or history.dtype != head.dtype or history.device != head.device):
        raise ValueError("history/head require matching floating [...,12]/[...,2,12] tensors")
    if not bool(torch.isfinite(head).all()) or not bool(torch.isfinite(history).all()):
        raise ValueError("history/head must be finite")
    sigma = head[..., 1, :].exp()
    if not bool((torch.isfinite(sigma) & (sigma > 0)).all()):
        raise ValueError("learned conditional sigma must be finite and strictly positive without clipping")
    if not bool((history.abs() <= HISTORY_CLIP).all()):
        raise ValueError("previous raw history must already be schema-clipped to +/-20")
    if ratio == 0:
        return head
    mean = (1.0 - ratio) * head[..., 0, :] + ratio * history
    if not bool(torch.isfinite(mean).all()):
        raise ValueError("conditional mean is not representable")
    return torch.stack((mean, head[..., 1, :]), dim=-2)


class SemanticHistoryMLPModel(MLPModel):
    """Official MLP layout with a fixed, stateless conditional output kernel."""

    def __init__(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str,
        output_dim: int, hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "elu", obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        observation_layout: str | None = None,
    ) -> None:
        if observation_layout is not None and (
                type(observation_layout) is not str or observation_layout != ROLE_OBSERVATION_LAYOUT):
            raise ValueError("history actor observation layout is unsupported")
        dimension = ROLE_OBSERVATION_BASE_DIM if observation_layout is None else ROLE_OBSERVATION_DIM
        if (obs_normalization is not False or obs_set != "actor" or output_dim != 12
                or obs_groups.get(obs_set) != ["policy"]
                or "policy" not in obs or obs["policy"].ndim != 2
                or obs["policy"].shape[-1] != dimension
                or not isinstance(distribution_cfg, dict)
                or distribution_cfg.get("class_name") != "HeteroscedasticGaussianDistribution"
                or distribution_cfg.get("std_type") != "log"):
            raise ValueError("history actor requires its explicit identity-normalized policy layout and log-Gaussian Full12")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims,
                         activation, obs_normalization, distribution_cfg)
        if type(self.distribution) is not HeteroscedasticGaussianDistribution:
            raise ValueError("history actor requires the official heteroscedastic distribution")
        # Preserve the legacy actor's attribute layout as well as its state_dict.
        # This immutable configuration label is not action history or a buffer.
        if observation_layout is not None:
            self.observation_layout = observation_layout

    def forward(
        self, obs: TensorDict, masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None, stochastic_output: bool = False,
    ) -> torch.Tensor:
        # Match the official nonrecurrent padded-minibatch path exactly.
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        expected_dimension = (ROLE_OBSERVATION_BASE_DIM if getattr(self, "observation_layout", None) is None
                              else ROLE_OBSERVATION_DIM)
        if (getattr(self, "observation_layout", None) not in (None, ROLE_OBSERVATION_LAYOUT)
                or self.obs_dim != expected_dimension or latent.shape[-1] != expected_dimension):
            raise ValueError("history actor observation differs from its explicit layout")
        head = history_conditioned_head(
            self.mlp(latent), latent[..., HISTORY_START:HISTORY_STOP], HISTORY_RHO)
        if stochastic_output:
            self.distribution.update(head)
            return self.distribution.sample()
        # Like official deterministic inference, do not update distribution cache.
        return self.distribution.deterministic_output(head)

    def as_jit(self) -> torch.nn.Module:
        raise NotImplementedError("history-conditioned JIT export is not supported; base export drops history")

    def as_onnx(self, verbose: bool = False) -> torch.nn.Module:
        raise NotImplementedError("history-conditioned ONNX export is not supported; base export drops history")


class SemanticTemperedHistoryMLPModel(SemanticHistoryMLPModel):
    """Same learned state and mean; versioned half-temperature innovation only.

    The exact temperature is explicit constructor configuration, not a learned
    parameter, mutable action history, sampler wrapper, or observation feature.
    The official distribution cache supplies sampling AND every PPO likelihood.
    """

    def __init__(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str,
        output_dim: int, hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "elu", obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        observation_layout: str | None = None,
        exploration_std_temperature: float | None = None,
    ) -> None:
        if (type(exploration_std_temperature) is not float
                or exploration_std_temperature != 0.5
                or observation_layout != ROLE_OBSERVATION_LAYOUT):
            raise ValueError("tempered history requires explicit temperature 0.5 and the 372 role layout")
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims,
                         activation, obs_normalization, distribution_cfg, observation_layout)

    @property
    def exploration_std_temperature(self) -> float:
        return 0.5

    def forward(
        self, obs: TensorDict, masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None, stochastic_output: bool = False,
    ) -> torch.Tensor:
        if not stochastic_output:
            # Keep frozen deterministic output, cache and RNG exactly on v1.
            return super().forward(obs, masks, hidden_state, stochastic_output=False)
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        if (getattr(self, "observation_layout", None) != ROLE_OBSERVATION_LAYOUT
                or self.obs_dim != ROLE_OBSERVATION_DIM or latent.shape[-1] != ROLE_OBSERVATION_DIM):
            raise ValueError("tempered history actor observation differs from its explicit 372 layout")
        head = history_conditioned_head(
            self.mlp(latent), latent[..., HISTORY_START:HISTORY_STOP], HISTORY_RHO)
        effective_log_std = head[..., 1, :] + math.log(self.exploration_std_temperature)
        effective_std = effective_log_std.exp()
        if not bool((torch.isfinite(effective_std) & (effective_std > 0)).all()):
            raise ValueError("effective conditional sigma must be finite and strictly positive without clipping")
        effective_head = torch.stack((head[..., 0, :], effective_log_std), dim=-2)
        self.distribution.update(effective_head)
        return self.distribution.sample()


class SemanticQuarterTemperedHistoryMLPModel(SemanticTemperedHistoryMLPModel):
    """Explicit quarter-temperature version, sharing the existing Gaussian path.

    State-dict topology, conditional mean, history rho and learned sigma head
    remain unchanged. Only effective innovation sigma is halved relative to the
    half-temperature version; this is not a deterministic-control improvement.
    """

    def __init__(
        self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str,
        output_dim: int, hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "elu", obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        observation_layout: str | None = None,
        exploration_std_temperature: float | None = None,
    ) -> None:
        if (type(exploration_std_temperature) is not float
                or exploration_std_temperature != 0.25
                or observation_layout != ROLE_OBSERVATION_LAYOUT):
            raise ValueError("quarter-tempered history requires explicit temperature 0.25 and the 372 role layout")
        # Bypass only the immutable 0.5 constructor gate. Inherit its common
        # forward path so sampling and all PPO likelihoods use the same sigma.
        SemanticHistoryMLPModel.__init__(
            self, obs, obs_groups, obs_set, output_dim, hidden_dims,
            activation, obs_normalization, distribution_cfg, observation_layout)

    @property
    def exploration_std_temperature(self) -> float:
        return 0.25


def cap_transition_request_history(latent: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pure current-observation center; never consume a mutable handoff flag.

    The saved physical quantity is the previous filtered REQUEST, not the
    post-headroom/final-slew actuator command or measured joint displacement.
    """
    from .semantic_policy_distribution import REQUEST_HISTORY_CAPS, REQUEST_HISTORY_SCALES
    if (not isinstance(latent, torch.Tensor) or not latent.is_floating_point()
            or latent.ndim < 2 or latent.shape[-1] != ROLE_OBSERVATION_DIM
            or not bool(torch.isfinite(latent).all()) or not bool((latent.abs() <= HISTORY_CLIP).all())):
        raise ValueError("request history requires finite fixed-scale-clipped 372 observations")
    stage_bits, completed, age = latent[..., :13], latent[..., 158:171], latent[..., 20]
    if (not bool(((stage_bits == 0) | (stage_bits == 1)).all())
            or not bool((stage_bits.sum(-1) == 1).all())
            or not bool(((completed == 0) | (completed == 1)).all())
            or not bool((age >= 0).all())):
        raise ValueError("request history stage/completed/age encoding is invalid")
    stage = stage_bits.argmax(-1)
    predecessor = (stage - 1).clamp(min=0)
    caps = latent.new_tensor(REQUEST_HISTORY_CAPS)
    current_cap, previous_cap = caps[stage], caps[predecessor]
    predecessor_completed = completed.gather(-1, predecessor.unsqueeze(-1)).squeeze(-1) == 1
    first = (age == 0) & (stage > 0) & predecessor_completed
    gate = first.unsqueeze(-1) & (current_cap > previous_cap)
    previous_request = latent[..., 207:219] * latent.new_tensor(REQUEST_HISTORY_SCALES)
    raw_history = latent[..., HISTORY_START:HISTORY_STOP]
    center = raw_history
    if bool(gate.any()):
        # Valid old-cap values are strictly interior in every enlarged cap.
        # Only allow float32 fixed-scale decode roundoff, never silently clip.
        valid = previous_request.abs() <= previous_cap + 1e-5
        ratio = torch.where(gate, previous_request/current_cap, torch.zeros_like(raw_history))
        if not bool((~gate | valid).all()) or not bool((ratio.abs() < 1).all()):
            raise ValueError("request history violates its predecessor physical request cap")
        center = torch.where(gate, torch.atanh(ratio), raw_history)
    return center, {"gate_full12": gate, "stage_index": stage, "encoded_stage_age": age,
        "predecessor_completed": predecessor_completed, "current_cap_full12": current_cap,
        "predecessor_cap_full12": previous_cap, "previous_filtered_request_full12": previous_request}


class SemanticCapTransitionQuarterHistoryMLPModel(SemanticQuarterTemperedHistoryMLPModel):
    """Same parameters and sigma; explicit observable entry-history mean kernel."""

    def forward(self, obs: TensorDict, masks: torch.Tensor | None = None,
                hidden_state: HiddenState = None, stochastic_output: bool = False) -> torch.Tensor:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        if (getattr(self, "observation_layout", None) != ROLE_OBSERVATION_LAYOUT
                or self.obs_dim != ROLE_OBSERVATION_DIM or latent.shape[-1] != ROLE_OBSERVATION_DIM):
            raise ValueError("request history actor differs from its explicit 372 layout")
        history, _ = cap_transition_request_history(latent)
        head = history_conditioned_head(self.mlp(latent), history, HISTORY_RHO)
        if not stochastic_output:
            return self.distribution.deterministic_output(head)
        log_std = head[..., 1, :] + math.log(self.exploration_std_temperature)
        sigma = log_std.exp()
        if not bool((torch.isfinite(sigma) & (sigma > 0)).all()):
            raise ValueError("effective conditional sigma must be finite and strictly positive without clipping")
        self.distribution.update(torch.stack((head[..., 0, :], log_std), dim=-2))
        return self.distribution.sample()


def physical_innovation_effective_log_std(
    head_log_std: torch.Tensor, latent: torch.Tensor, temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One phase/channel sigma rule shared by the actor and request evidence.

    This changes a raw Gaussian's innovation, not its mean, sampled action,
    physical cap or mutable history. Phase validation is the existing pure
    current-observation contract; it does not advance an actor or consume RNG.
    """
    _, evidence = cap_transition_request_history(latent)
    if (type(temperature) is not float or temperature != 0.25
            or not isinstance(head_log_std, torch.Tensor)
            or head_log_std.shape != latent.shape[:-1] + (12,)
            or head_log_std.device != latent.device or head_log_std.dtype != latent.dtype):
        raise ValueError("physical innovation sigma requires the quarter full12 current-observation head")
    log_std = head_log_std + math.log(temperature)
    gate = ((evidence["stage_index"] >= 5).unsqueeze(-1)
            & (torch.arange(12, device=latent.device) == 3))
    log_std = torch.where(gate, log_std + math.log(24.0 / 112.0), log_std)
    sigma = log_std.exp()
    if not bool((torch.isfinite(sigma) & (sigma > 0)).all()):
        raise ValueError("physical innovation sigma must remain finite and strictly positive without clipping")
    multiplier = torch.where(gate, torch.full_like(log_std, 24.0 / 112.0), torch.ones_like(log_std))
    return log_std, multiplier


class SemanticFRKneePhysicalInnovationHistoryMLPModel(SemanticCapTransitionQuarterHistoryMLPModel):
    """Same REQUEST-history mean; only P06+ FR-knee innovation sigma is scaled."""

    def forward(self, obs: TensorDict, masks: torch.Tensor | None = None,
                hidden_state: HiddenState = None, stochastic_output: bool = False) -> torch.Tensor:
        if not stochastic_output:
            return super().forward(obs, masks, hidden_state, stochastic_output=False)
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        if (getattr(self, "observation_layout", None) != ROLE_OBSERVATION_LAYOUT
                or self.obs_dim != ROLE_OBSERVATION_DIM or latent.shape[-1] != ROLE_OBSERVATION_DIM):
            raise ValueError("physical innovation actor requires the original explicit 372 observation layout")
        history, _ = cap_transition_request_history(latent)
        head = history_conditioned_head(self.mlp(latent), history, HISTORY_RHO)
        log_std, _ = physical_innovation_effective_log_std(
            head[..., 1, :], latent, self.exploration_std_temperature)
        self.distribution.update(torch.stack((head[..., 0, :], log_std), dim=-2))
        return self.distribution.sample()


def task_conditioned_physical_scales(latent: torch.Tensor):
    """Pure observable exploration schedule; contact proxies are not acceptance.

    B is a physical equivalent innovation scale, NOT actual sigma, a target,
    capacity or a sign preference. No hidden timer or new action history.
    """
    from .semantic_policy_distribution import TASK_CONDITIONED_PHYSICAL_B_TABLE
    _, evidence = cap_transition_request_history(latent)
    for start, end in ((131,139), (146,158)):
        bits = latent[...,start:end]
        if not bool(((bits == 0) | (bits == 1)).all()):
            raise ValueError("task sigma requires binary observed contact/history flags")
    stage, cap = evidence["stage_index"], evidence["current_cap_full12"]
    base = cap.clone()
    base[...,3] = torch.where(stage >= 5, 24., base[...,3])
    weights = {}
    no_fr = (latent[...,133] == 0) & (latent[...,134] == 0)
    no_fl = (latent[...,131] == 0) & (latent[...,132] == 0)
    fr = ((stage <= 1) & (latent[...,155] == 0) & (latent[...,147] == 1)
          & no_fr & (latent[...,33] >= 2))
    weights["FR_air_approach_proxy"] = fr * (latent[...,24]/.015).clamp(0.,1.)
    fl = ((stage == 4) & (latent[...,150] == 1) & (latent[...,154] == 0) & no_fl)
    weights["FL_crossed_air_pending"] = fl * (latent[...,21]/.003).clamp(0.,1.)

    def front_support_proxy(i):
        goal, geom, pair, force = 21+3*i, 99+3*i, 131+2*i, 123+2*i
        # Obstacle-pair reaction near top is NOT classified TOP or current Y ROI.
        return ((latent[...,pair] == 0) & (latent[...,pair+1] == 1)
                & (latent[...,force+1]*100. >= .2)
                & (latent[...,goal] >= -.015) & (latent[...,goal] <= .025)
                & (latent[...,goal+1] >= -.005) & (latent[...,geom+1] <= .005))

    fronts_placed = (latent[...,154] == 1) & (latent[...,155] == 1)
    support_proxy = front_support_proxy(0) & front_support_proxy(1)
    pending = latent[...,157] == 0
    p06 = (stage == 5) & fronts_placed & pending
    rolling = p06 & support_proxy
    weights["P06_front_support_rolling_proxy"] = rolling.to(latent.dtype)
    weights["P06_front_support_recovery_proxy"] = (p06 & ~support_proxy).to(latent.dtype)
    # Any rear leg approaching the preparation region releases rolling preference.
    rear_x = torch.maximum(latent[...,28], latent[...,31])
    rear_proximity = ((rear_x+.27)/.05).clamp(0.,1.)
    prepare = ((stage >= 6) & (stage <= 8) & pending).to(latent.dtype)
    weights["RR_preparation"] = torch.maximum(prepare, p06*rear_proximity)
    valid_rr = ((stage >= 5) & (stage <= 8) & pending & (latent[...,149] == 1))
    weights["RR_current_valid_lift"] = valid_rr.to(latent.dtype)
    B = base
    for name, weight in weights.items():
        target = latent.new_tensor(TASK_CONDITIONED_PHYSICAL_B_TABLE[name])
        if target.shape != (12,) or not bool((torch.isfinite(target) & (target > 0)).all()):
            raise ValueError("task sigma table must preserve twelve positive finite scales")
        B = torch.lerp(B, target, weight.unsqueeze(-1))
    return B, cap, {"task_state_weights": weights, "stage_index":stage,
        "front_support_proxy_not_exact_TOP":support_proxy,
        "current_RR_qualification":latent[...,149], "current_rear_front_distance_m":rear_x}


def task_conditioned_effective_log_std(head_log_std: torch.Tensor, latent: torch.Tensor,
                                     temperature: float):
    """The one sigma kernel used by official sampling/likelihood and evidence."""
    B, cap, evidence = task_conditioned_physical_scales(latent)
    if (type(temperature) is not float or temperature != .25
            or not isinstance(head_log_std, torch.Tensor)
            or head_log_std.shape != latent.shape[:-1]+(12,)
            or head_log_std.device != latent.device or head_log_std.dtype != latent.dtype):
        raise ValueError("task sigma requires matching quarter full12 current-observation head")
    multiplier = B/cap
    log_std = head_log_std + math.log(temperature) + multiplier.log()
    sigma = log_std.exp()
    if not bool((torch.isfinite(sigma) & (sigma > 0)).all()):
        raise ValueError("task-conditioned Gaussian sigma must remain positive finite without clipping")
    return log_std, {**evidence, "physical_equivalent_B_full12":B,
                    "current_cap_full12":cap, "innovation_sigma_multiplier_full12":multiplier}


class SemanticTaskConditionedHipWheelHistoryMLPModel(SemanticCapTransitionQuarterHistoryMLPModel):
    """Same REQUEST-history mean/rho/cap; observable positive full12 innovation."""

    def forward(self, obs: TensorDict, masks: torch.Tensor | None = None,
                hidden_state: HiddenState = None, stochastic_output: bool = False) -> torch.Tensor:
        if not stochastic_output:
            return super().forward(obs,masks,hidden_state,stochastic_output=False)
        obs = unpad_trajectories(obs,masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs,masks,hidden_state)
        if (getattr(self,"observation_layout",None) != ROLE_OBSERVATION_LAYOUT
                or self.obs_dim != ROLE_OBSERVATION_DIM or latent.shape[-1] != ROLE_OBSERVATION_DIM):
            raise ValueError("task sigma actor requires the unchanged explicit 372 layout")
        history,_ = cap_transition_request_history(latent)
        head = history_conditioned_head(self.mlp(latent),history,HISTORY_RHO)
        log_std,_ = task_conditioned_effective_log_std(head[...,1,:],latent,self.exploration_std_temperature)
        self.distribution.update(torch.stack((head[...,0,:],log_std),dim=-2))
        return self.distribution.sample()

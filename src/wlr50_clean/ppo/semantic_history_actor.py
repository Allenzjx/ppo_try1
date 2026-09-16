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

"""Frozen CP225280 prior plus an observable, RR-window raw Gaussian correction.

This module does not choose the physical activation event, maintain a gate,
step an environment, or own an optimizer. The environment supplies a latched
gate in the observation. Only active observations may enter stochastic PPO.
One combined raw head passes through the existing HISTORY kernel exactly once.
"""
from __future__ import annotations

import copy
import hashlib
import math
import re
from collections.abc import Mapping, Sequence

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from rsl_rl.utils import unpad_trajectories

from .semantic_history_actor import HISTORY_RHO, history_conditioned_head
from .semantic_p05_capture_actor import p05_capture_request_history
from .semantic_rear_owner_actor import (
    SemanticRearOwnerRecoveryHistoryMLPModel, validate_rear_owner_latent,
)
from .semantic_rear_owner_profile import REAR_OWNER_OBSERVATION_LAYOUT


LEGACY_POLICY_VERSION = "frozen_cp225280_rr_capture_local_history_v1"
LEGACY_OBSERVATION_LAYOUT = "role439_rr_capture_local_v1"
LEGACY_OBSERVATION_DIMENSION = 447
POLICY_VERSION = "frozen_cp225280_rr_capture_local_history_v2"
OBSERVATION_LAYOUT = "role439_rr_capture_local_v2"
PRIOR_DIMENSION = 439
OBSERVATION_DIMENSION = 448
LEGACY_CAPTURE_FIELDS = (
    "active", "activation_age_norm", "entry_rr_hip_norm", "entry_rr_knee_norm",
    "entry_gap_norm", "current_top_contact", "current_top_bearing", "capture_hold_progress",
)
CAPTURE_FIELDS = LEGACY_CAPTURE_FIELDS + ("current_attempt_capture_eligible",)
# Raw Gaussian innovation sigma, AFTER the mean's HISTORY kernel. No hidden
# quarter temperature, cooperative multipliers, or second additive noise draw.
# These modest defaults are construction settings, not proven physical tuning.
DEFAULT_CAPTURE_STD = (.04, .05, .03, .03, .04, .05, .20, .20, .05, .05, .05, .05)
IDENTITY_LOCAL_MEAN_COORDINATES = (1.,) * 12


def checked_local_mean_coordinates(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 12
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
            or any(value[i] != 1. for i in (0, 1, 2, 3, 4, 5, 8, 9, 10, 11))
            or value[6] != value[7] or value[6] not in (1., 10.)):
        raise ValueError("only identity or explicitly versioned RR mean coordinate gain10 is supported")
    return tuple(float(x) for x in value)


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Named tensor hash; includes parameters and any registered buffers."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError("prior state must contain only named tensors")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _checked_hash(value):
    if value is not None and (not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None):
        raise ValueError("expected prior hash must be a lowercase SHA256")
    return value


def _checked_std(value: Sequence[float]) -> tuple[float, ...]:
    if (not isinstance(value, (tuple, list)) or len(value) != 12
            or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in value)):
        raise ValueError("capture innovation sigma requires twelve finite positive raw standard deviations")
    return tuple(float(x) for x in value)


def validate_capture_local_latent(latent, *, legacy447_migration_only=False):
    if type(legacy447_migration_only) is not bool:
        raise ValueError("legacy447 migration flag must be an explicit boolean")
    dimension = LEGACY_OBSERVATION_DIMENSION if legacy447_migration_only else OBSERVATION_DIMENSION
    if (not isinstance(latent, torch.Tensor) or not latent.is_floating_point()
            or latent.ndim < 2 or latent.shape[-1] != dimension
            or not bool(torch.isfinite(latent).all()) or not bool((latent.abs() <= 20.).all())):
        raise ValueError(f"capture local actor requires finite schema-clipped{dimension} observations")
    validate_rear_owner_latent(latent[..., :PRIOR_DIMENSION])
    context = latent[..., PRIOR_DIMENSION:]
    boolean_indices = (0, 5, 6) if legacy447_migration_only else (0, 5, 6, 8)
    for index in boolean_indices:
        if not bool(((context[..., index] == 0) | (context[..., index] == 1)).all()):
            raise ValueError("capture activation, contact/bearing and current-attempt eligibility must be boolean features")
    for index in (1, 7):
        if not bool(((context[..., index] >= 0) & (context[..., index] <= 1)).all()):
            raise ValueError("capture age and hold progress must be normalized to [0,1]")
    if not bool((context[..., 6] <= context[..., 5]).all()):
        raise ValueError("current TOP bearing cannot be true without current TOP contact")
    # Eligibility is observed task history, not another action gate. An active
    # recovering policy may see eligibility0 after GROUND, and actual TOP can
    # exist without a qualified lift attempt. Do not erase its control or fake
    # eligibility here; the physical task owns requalification and success.
    return context[..., 0].bool()


class SemanticRRCaptureLocalHistoryMLPModel(MLPModel):
    """Official RSL model API; ``mlp`` is ONLY the trainable local head.

    The complete frozen prior is a registered child and travels in state_dict.
    Call load_frozen_prior_state once for initial publication; ordinary strict
    load_state_dict restores a complete published composite without reinitializing
    local parameters. Neither call imports a checkpoint or changes the environment.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(64, 64),
                 activation="elu", obs_normalization=False, distribution_cfg=None,
                 observation_layout=OBSERVATION_LAYOUT, initial_capture_std=DEFAULT_CAPTURE_STD,
                 expected_prior_state_sha256=None, legacy447_migration_only=False,
                 local_mean_coordinate_gain_full12=IDENTITY_LOCAL_MEAN_COORDINATES):
        if type(legacy447_migration_only) is not bool:
            raise ValueError("legacy447 migration flag must be an explicit boolean")
        self.local_mean_coordinate_gain_full12 = checked_local_mean_coordinates(local_mean_coordinate_gain_full12)
        if legacy447_migration_only and self.local_mean_coordinate_gain_full12 != IDENTITY_LOCAL_MEAN_COORDINATES:
            raise ValueError("legacy447 migration actor must keep identity parameter coordinates")
        dimension = LEGACY_OBSERVATION_DIMENSION if legacy447_migration_only else OBSERVATION_DIMENSION
        layout = LEGACY_OBSERVATION_LAYOUT if legacy447_migration_only else OBSERVATION_LAYOUT
        if (observation_layout != layout or obs_set != "actor" or output_dim != 12
                or obs_groups.get("actor") != ["policy"] or obs_normalization is not False
                or obs["policy"].ndim != 2 or obs["policy"].shape[-1] != dimension
                or not isinstance(distribution_cfg, dict)
                or distribution_cfg.get("class_name") != "HeteroscedasticGaussianDistribution"
                or distribution_cfg.get("std_type") != "log"):
            raise ValueError(f"local RR actor requires explicit{dimension}/full12/Identity/log-Gaussian configuration")
        self.legacy447_migration_only = legacy447_migration_only
        self.observation_dimension = dimension
        self.policy_version = LEGACY_POLICY_VERSION if legacy447_migration_only else POLICY_VERSION
        self.initial_capture_std = _checked_std(initial_capture_std)
        self.expected_prior_state_sha256 = _checked_hash(expected_prior_state_sha256)
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, copy.deepcopy(distribution_cfg))
        if type(self.distribution) is not HeteroscedasticGaussianDistribution:
            raise ValueError("local RR actor requires the official heteroscedastic Gaussian")
        self.observation_layout = observation_layout
        # Preserve the source topology and source normalization/kernel configuration.
        self.frozen_prior = SemanticRearOwnerRecoveryHistoryMLPModel(
            {"policy": obs["policy"][..., :PRIOR_DIMENSION]}, {"actor": ["policy"]}, "actor", 12,
            hidden_dims=(256, 256), activation="elu", obs_normalization=False,
            distribution_cfg=copy.deepcopy(distribution_cfg),
            observation_layout=REAR_OWNER_OBSERVATION_LAYOUT, exploration_std_temperature=.25)
        self._prior_anchor = None
        self._last_forward_evidence = None  # detached audit only, never read to select an action
        self._freeze_prior()
        last = [module for module in self.mlp.modules() if isinstance(module, torch.nn.Linear)][-1]
        if last.out_features != 24:
            raise ValueError("local head must contain mean12 and log-standard-deviation12")
        with torch.no_grad():
            last.weight.zero_()
            last.bias[:12].zero_()
            last.bias[12:].copy_(last.bias.new_tensor(self.initial_capture_std).log())

    def _freeze_prior(self):
        self.frozen_prior.requires_grad_(False)
        self.frozen_prior.eval()
        for parameter in self.frozen_prior.parameters():
            parameter.grad = None

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, "frozen_prior"):
            self._freeze_prior()
        return self

    def update_normalization(self, obs):
        # PPO calls this after every credited transition. The source and local
        # normalizers are Identity; deliberately do not delegate to the prior.
        if self.obs_normalization or type(self.obs_normalizer) is not torch.nn.Identity:
            raise RuntimeError("capture local normalization changed")
        if (self.frozen_prior.obs_normalization
                or type(self.frozen_prior.obs_normalizer) is not torch.nn.Identity):
            raise RuntimeError("frozen prior normalization changed")

    def trainable_parameters(self):
        """Caller builds a NEW optimizer from these plus its independent critic."""
        prior_ids = {id(parameter) for parameter in self.frozen_prior.parameters()}
        result = tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
        if not result or any(id(parameter) in prior_ids for parameter in result):
            raise RuntimeError("local optimizer parameter isolation failed")
        return result

    def _remember_prior(self):
        self._freeze_prior()
        state = self.frozen_prior.state_dict()
        actual_hash = tensor_state_sha256(state)
        if self.expected_prior_state_sha256 is not None and actual_hash != self.expected_prior_state_sha256:
            raise ValueError("frozen prior does not match the declared immutable state hash")
        self.expected_prior_state_sha256 = actual_hash
        self._prior_anchor = {name: value.detach().cpu().clone() for name, value in state.items()}
        return actual_hash

    def load_frozen_prior_state(self, state, *, expected_sha256=None):
        if self._prior_anchor is not None:
            raise RuntimeError("frozen prior already bound; do not replace it or reset the local module")
        expected = _checked_hash(expected_sha256)
        if expected is not None:
            if self.expected_prior_state_sha256 not in (None, expected):
                raise ValueError("prior constructor and load hashes disagree")
            self.expected_prior_state_sha256 = expected
        actual = tensor_state_sha256(state)
        if self.expected_prior_state_sha256 not in (None, actual):
            raise ValueError("source state differs from the declared frozen prior")
        self.frozen_prior.load_state_dict(state, strict=True)
        return self._remember_prior()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict is not True or assign is not False:
            raise ValueError("composite restore must be strict and preserve optimizer parameter objects")
        source = {name[len("frozen_prior."):]: value for name, value in state_dict.items()
                  if name.startswith("frozen_prior.")}
        if not source:
            raise ValueError("composite checkpoint is missing its frozen prior")
        actual = tensor_state_sha256(source)
        if self.expected_prior_state_sha256 not in (None, actual):
            raise ValueError("composite checkpoint attempts to replace the frozen prior")
        result = super().load_state_dict(state_dict, strict=True, assign=False)
        self._remember_prior()
        self._last_forward_evidence = None
        return result

    def assert_frozen_state(self, optimizer=None):
        if self._prior_anchor is None:
            raise RuntimeError("load the verified CP225280 prior before inference/training")
        self.update_normalization(None)
        if self.frozen_prior.training:
            raise RuntimeError("frozen prior left permanent eval mode")
        current = self.frozen_prior.state_dict()
        if current.keys() != self._prior_anchor.keys() or any(
                not torch.equal(value.detach().cpu(), self._prior_anchor[name])
                for name, value in current.items()):
            raise RuntimeError("frozen prior parameter/buffer changed")
        parameters = tuple(self.frozen_prior.parameters())
        if any(parameter.requires_grad or parameter.grad is not None for parameter in parameters):
            raise RuntimeError("frozen prior received trainable state or gradients")
        if optimizer is not None:
            prior_ids = {id(parameter) for parameter in parameters}
            if any(id(parameter) in prior_ids for group in optimizer.param_groups for parameter in group["params"]):
                raise RuntimeError("frozen parameters must not be present in the local optimizer")
        return self.expected_prior_state_sha256

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if self.legacy447_migration_only and stochastic_output:
            raise ValueError("legacy447 migration-only actor rejects stochastic/PPO forward")
        if self._prior_anchor is None:
            raise RuntimeError("load the verified CP225280 prior before inference/training")
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        active = validate_capture_local_latent(latent, legacy447_migration_only=self.legacy447_migration_only)
        if stochastic_output and not bool(active.all()):
            raise ValueError("inactive frozen prefix is deterministic and may not enter PPO Gaussian storage")
        if (self.frozen_prior.training or any(p.requires_grad for p in self.frozen_prior.parameters())
                or self.frozen_prior.obs_normalization
                or type(self.frozen_prior.obs_normalizer) is not torch.nn.Identity):
            raise RuntimeError("prior freeze/normalization contract violated")
        prior_latent = latent[..., :PRIOR_DIMENSION]
        with torch.no_grad():
            prior_head = self.frozen_prior.mlp(self.frozen_prior.obs_normalizer(prior_latent))
        center, _ = p05_capture_request_history(prior_latent[..., :389])
        # No local forward, stochastic draw, or distribution-cache update in an
        # all-inactive prefix. For mixed deterministic batches, where selects
        # the literal prior head on inactive rows, not a second HISTORY output.
        if bool(active.any()):
            local_head = self.mlp(latent)
            local_mean = local_head[..., 0, :]
            # Fixed serialized PARAMETER coordinates only; action/HISTORY units
            # and all log-std rows remain exactly the existing contract.
            if self.local_mean_coordinate_gain_full12 != IDENTITY_LOCAL_MEAN_COORDINATES:
                local_mean = torch.cat((local_mean[..., :6],
                    local_mean[..., 6:8] * self.local_mean_coordinate_gain_full12[6],
                    local_mean[..., 8:]), dim=-1)
            applied_local = torch.where(active.unsqueeze(-1), local_mean, torch.zeros_like(local_mean))
            combined_mean = torch.where(active.unsqueeze(-1), prior_head[..., 0, :] + local_mean,
                                        prior_head[..., 0, :])
            head_log_std = torch.where(active.unsqueeze(-1), local_head[..., 1, :], prior_head[..., 1, :])
            head = torch.stack((combined_mean, head_log_std), dim=-2)
        else:
            local_mean = torch.zeros_like(prior_head[..., 0, :])
            applied_local = local_mean
            head = prior_head
        conditional = history_conditioned_head(head, center, HISTORY_RHO)
        # This separate counterfactual is audit arithmetic, not an executed
        # second filter, shadow action history, or independent nominal rollout.
        prior_conditional = (1. - HISTORY_RHO) * prior_head[..., 0, :] + HISTORY_RHO * center
        self._last_forward_evidence = {
            "local_mean_coordinate_gain_full12": self.local_mean_coordinate_gain_full12,
            "active": active.detach().clone(), "capture_context": latent[..., PRIOR_DIMENSION:].detach().clone(),
            "prior_raw_mean": prior_head[..., 0, :].detach().clone(),
            "local_raw_mean_delta": local_mean.detach().clone(),
            "applied_local_raw_mean_delta": applied_local.detach().clone(),
            "combined_raw_mean": head[..., 0, :].detach().clone(),
            "history_center": center.detach().clone(),
            "prior_same_observation_conditional_mean": prior_conditional.detach().clone(),
            "conditional_mean": conditional[..., 0, :].detach().clone(),
            "head_log_std": conditional[..., 1, :].detach().clone(),
            "local_forward_count": int(bool(active.any())),
        }
        if stochastic_output:
            self.distribution.update(conditional)
            return self.distribution.sample()
        # Match official deterministic inference: leave the Gaussian cache and RNG alone.
        return self.distribution.deterministic_output(conditional)

    def as_jit(self):
        raise NotImplementedError("composite prior/gate/HISTORY export is not implemented")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("composite prior/gate/HISTORY export is not implemented")


def audited_capture_local_policy_request(actor, observation, action_call, *, stochastic):
    """Observe the actual one forward/draw, not a second reconstructed policy.

    Training passes ``lambda: alg.act(obs)``. Prefix/evaluation passes the same
    composite's deterministic call. This helper never supplies PPO credit or
    writes storage and never computes a fictitious prefix Gaussian likelihood.
    """
    if type(actor) is not SemanticRRCaptureLocalHistoryMLPModel:
        raise ValueError("capture request audit requires its explicit composite actor")
    if actor.legacy447_migration_only:
        raise ValueError("legacy447 migration-only actor cannot enter a production/PPO request audit")
    seen = []

    def observed(_module, _inputs, _output):
        seen.append(actor._last_forward_evidence)

    handle = actor.register_forward_hook(observed)
    try:
        raw = action_call()
    finally:
        handle.remove()
    if (len(seen) != 1 or tuple(raw.shape) != (1, 12)
            or observation["policy"].shape != (1, OBSERVATION_DIMENSION)):
        raise RuntimeError("capture request audit requires exactly one N1 composite forward")
    evidence = seen[0]
    if not torch.equal(evidence["capture_context"], observation["policy"][..., PRIOR_DIMENSION:]):
        raise RuntimeError("capture audit observation differs from the actual forward")
    active = bool(evidence["active"][0])
    vector = lambda value: value[0].detach().cpu().tolist()
    record = dict(schema="wlr50_clean.actual_rr_capture_local_request.v2", policy_version=POLICY_VERSION,
        mode="active_combined_raw_Gaussian" if stochastic else "deterministic_combined_conditional_mean",
        capture_active=active, capture_context=dict(zip(CAPTURE_FIELDS, vector(evidence["capture_context"]))),
        prior_raw_mean_full12=vector(evidence["prior_raw_mean"]),
        local_raw_mean_delta_full12=vector(evidence["local_raw_mean_delta"]),
        applied_local_raw_mean_delta_full12=vector(evidence["applied_local_raw_mean_delta"]),
        local_mean_coordinate_gain_full12=list(actor.local_mean_coordinate_gain_full12),
        combined_raw_mean_full12=vector(evidence["combined_raw_mean"]),
        history_center_full12=vector(evidence["history_center"]), rho=HISTORY_RHO,
        prior_same_observation_conditional_mean_full12=vector(evidence["prior_same_observation_conditional_mean"]),
        conditional_mean_full12=vector(evidence["conditional_mean"]),
        selected_raw_full12=vector(raw), selected_tanh_full12=vector(torch.tanh(raw)),
        active_conditional_std_full12=vector(evidence["head_log_std"].exp()) if active else None,
        selected_raw_log_probability=None, sampling_draws=int(stochastic), extra_random_draws=0,
        extra_model_forwards=0, prior_network_forwards=1, local_network_forwards=evidence["local_forward_count"],
        history_kernel_applications=1, transformed_targets_are_not_policy_samples=True,
        prefix_excluded_from_new_PPO_credit=not active,
        frozen_prior_state_sha256=actor.expected_prior_state_sha256,
        sigma_semantics="declared_local_raw_conditional_std_no_prior_noise_or_extra_temperature",
        physical_scale_and_execution_source="same_decision_projector_mapper_headroom_and_native_dispatch_audit")
    if stochastic:
        mean, std = actor.output_distribution_params
        if not active or not torch.equal(mean, evidence["conditional_mean"]) or not torch.equal(
                std, evidence["head_log_std"].exp()):
            raise RuntimeError("actual composite Gaussian cache differs from the sampled mean/std")
        record["selected_raw_log_probability"] = float(actor.get_output_log_prob(raw)[0].detach().cpu())
    elif not torch.equal(raw, evidence["conditional_mean"]):
        raise RuntimeError("deterministic composite request differs from its actual conditional mean")
    return raw, record

"""An independent, checkpoint-frozen deterministic policy for reset-only roll-in.

The caller has already verified and loaded the checkpoint. This module checks
the loaded actor hash, not the checkpoint file again. It neither owns an
optimizer nor calls the training actor, including for a comparison forward pass.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from numbers import Real
import re
from typing import Any, Mapping

from .semantic_policy_distribution import (
    HISTORY_POLICY, supported_heteroscedastic_contract_version,
)
from .semantic_receiving_wheel_profile import RECEIVING_WHEEL_POLICY
from .semantic_p05_capture_profile import P05_CAPTURE_POLICY
from .semantic_rr_capture_profile import RR_CAPTURE_POLICY
from .semantic_p02_progress_profile import P02_PROGRESS_POLICY
from .semantic_p02_progress_actor import SemanticP02ProgressHistoryMLPModel
from .semantic_rear_cooperative_prep_profile import COOPERATIVE_PREP_POLICY
from .semantic_rear_cooperative_prep_actor import SemanticRearCooperativePrepHistoryMLPModel
from .semantic_rear_policy_timing_profile import REAR_POLICY_TIMING_POLICY
from .semantic_rear_policy_timing_actor import SemanticRearPolicyTimingHistoryMLPModel


def _tensor_hash(items: Any) -> str:
    """Same named-parameter byte convention as semantic_training.parameter_hash."""
    digest = hashlib.sha256()
    for name, value in sorted(items):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _task_conditioned_source_record(record):
    from pathlib import Path
    from .semantic_policy_distribution import (TASK_CONDITIONED_HIP_WHEEL_POLICY,
        FR_KNEE_PHYSICAL_INNOVATION_POLICY)
    target, source = record["policy_contract"], record.get("source_policy_contract")
    source_version = supported_heteroscedastic_contract_version(source)
    runtime_hash, effective_hash = record.get("source_runtime_content_sha256"), record.get("effective_runtime_content_sha256")
    if (record.get("effective_policy_contract") != target or runtime_hash is None
            or not isinstance(effective_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", effective_hash)):
        raise ValueError("task prefix requires explicit source and effective kernel/runtime provenance")
    archive = record.get("archive_only_exact_bytes_migration")
    task = record.get("task_conditioned_hip_wheel_migration")
    receiving = record.get("receiving_wheel_sigma_migration")
    if (sum(x is not None for x in (archive, task, receiving)) > 1
            or record.get("physical_innovation_sigma_migration") is not None
            or record.get("request_history_kernel_migration") is not None):
        raise ValueError("task prefix may not mix independent migration provenance")
    if archive is not None:
        migration, factor_key = archive, "archive_only_exact_bytes_factor"
        schema = "wlr50_clean.archive_only_exact_bytes_same372.v1"
        if (source != target or source_version not in (FR_KNEE_PHYSICAL_INNOVATION_POLICY, TASK_CONDITIONED_HIP_WHEEL_POLICY)
                or effective_hash != runtime_hash):
            raise ValueError("archive-only prefix may not change policy or runtime bytes")
    elif task is not None:
        migration, factor_key = task, "task_conditioned_hip_wheel_factor"
        schema = "wlr50_clean.task_conditioned_hip_wheel_same372.v1"
        if source_version != FR_KNEE_PHYSICAL_INNOVATION_POLICY or target["version"] != TASK_CONDITIONED_HIP_WHEEL_POLICY:
            raise ValueError("task prefix requires the exact FR-knee to task-conditioned policy boundary")
    elif receiving is not None:
        migration, factor_key = receiving, "receiving_wheel_sigma_factor"
        schema = "wlr50_clean.receiving_wheel_sigma_same372.v1"
        if source_version != TASK_CONDITIONED_HIP_WHEEL_POLICY or target["version"] != RECEIVING_WHEEL_POLICY:
            raise ValueError("receiving prefix requires the exact task to receiving-wheel sigma boundary")
    else:
        if (source != target or target["version"] not in (TASK_CONDITIONED_HIP_WHEEL_POLICY, RECEIVING_WHEEL_POLICY)
                or effective_hash != runtime_hash):
            raise ValueError("task prefix is not a verified migration or exact target checkpoint resume")
        return
    keys = {"plan_path", "plan_sha256", "source_checkpoint_sha256",
        "source_runtime_content_sha256", "target_runtime_content_sha256"}
    if not isinstance(migration, dict) or set(migration) != keys:
        raise ValueError("task prefix requires its exact immutable migration binding")
    path = Path(migration["plan_path"]).resolve(strict=True)
    raw = path.read_bytes()
    plan = json.loads(raw)
    factor = plan.get(factor_key, {})
    if (hashlib.sha256(raw).hexdigest() != migration["plan_sha256"]
            or migration["source_checkpoint_sha256"] != record["checkpoint_sha256"]
            or migration["source_runtime_content_sha256"] != runtime_hash
            or migration["target_runtime_content_sha256"] != effective_hash
            or plan.get("source_checkpoint") != record["checkpoint_path"]
            or plan.get("source_checkpoint_sha256") != record["checkpoint_sha256"]
            or plan.get("source_runtime_content_sha256") != runtime_hash
            or plan.get("target_runtime_content_sha256") != effective_hash
            or factor.get("schema") != schema or factor.get("source_policy_contract") != source
            or factor.get("target_policy_contract") != target):
        raise ValueError("task prefix plan differs from source checkpoint and effective kernel")
    if archive is not None and (factor.get("runtime_bytes_identical") is not True
            or plan.get("allowed_changed_files") != [] or plan.get("changed_file_hashes") != {}):
        raise ValueError("archive-only prefix plan contains runtime changes")


def _source_record(source: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        raise ValueError("verified source checkpoint must be a JSON mapping")
    try:
        record = json.loads(json.dumps(dict(source), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("verified source checkpoint must be finite JSON") from error
    if not isinstance(record.get("checkpoint_path"), str) or not record["checkpoint_path"].strip():
        raise ValueError("source checkpoint_path is required")
    for key in ("checkpoint_sha256", "actor_parameter_sha256"):
        if not isinstance(record.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", record[key]):
            raise ValueError(f"source {key} must be a lowercase SHA256")
    for key in ("source_global_policy_decisions", "source_ppo_updates"):
        if type(record.get(key)) is not int or record[key] < 0:
            raise ValueError(f"source {key} must be a nonnegative integer")
    try:
        supported_heteroscedastic_contract_version(record.get("policy_contract"))
    except ValueError as error:
        raise ValueError("checkpoint prefix requires the exact supported heteroscedastic policy contract") from error
    runtime_hash = record.get("source_runtime_content_sha256")
    if runtime_hash is not None and (
        not isinstance(runtime_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", runtime_hash)
    ):
        raise ValueError("source_runtime_content_sha256 must be a lowercase SHA256")
    from .semantic_policy_distribution import (HISTORY_REQUEST_CAP_TRANSITION_POLICY,
        HISTORY_QUARTER_TEMPERED_POLICY, FR_KNEE_PHYSICAL_INNOVATION_POLICY, TASK_CONDITIONED_HIP_WHEEL_POLICY)
    if record["policy_contract"]["version"] in (P05_CAPTURE_POLICY,RR_CAPTURE_POLICY,REAR_POLICY_TIMING_POLICY,P02_PROGRESS_POLICY,COOPERATIVE_PREP_POLICY):
        # Prefixes begin only from the already saved/reloaded migrated model,
        # never by pretending an old source hash names an appended actor.
        if (record.get("source_policy_contract") != record["policy_contract"]
                or record.get("effective_policy_contract") != record["policy_contract"]
                or record.get("effective_runtime_content_sha256") != runtime_hash
                or runtime_hash is None):
            raise ValueError("capture prefix requires an exact saved checkpoint/layout/runtime, not implicit remapping")
        return record
    if (record["policy_contract"]["version"] in (TASK_CONDITIONED_HIP_WHEEL_POLICY, RECEIVING_WHEEL_POLICY)
            or record.get("archive_only_exact_bytes_migration") is not None):
        _task_conditioned_source_record(record)
        return record
    if record["policy_contract"]["version"] in (HISTORY_REQUEST_CAP_TRANSITION_POLICY, FR_KNEE_PHYSICAL_INNOVATION_POLICY):
        target = record["policy_contract"]
        physical_innovation = target["version"] == FR_KNEE_PHYSICAL_INNOVATION_POLICY
        source_kernel = HISTORY_REQUEST_CAP_TRANSITION_POLICY if physical_innovation else HISTORY_QUARTER_TEMPERED_POLICY
        migration_key = "physical_innovation_sigma_migration" if physical_innovation else "request_history_kernel_migration"
        factor_key = "physical_innovation_sigma_factor" if physical_innovation else "request_history_kernel_factor"
        factor_schema = ("wlr50_clean.FR_knee_phase_physical_innovation_sigma.v1" if physical_innovation
                         else "wlr50_clean.cap_transition_request_history_same372.v1")
        source = record.get("source_policy_contract")
        source_version = supported_heteroscedastic_contract_version(source)
        effective_hash = record.get("effective_runtime_content_sha256")
        if (record.get("effective_policy_contract") != target
                or not isinstance(effective_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", effective_hash)
                or runtime_hash is None):
            raise ValueError("request-history prefix requires explicit effective kernel/runtime provenance")
        migration = record.get(migration_key)
        other_key = "request_history_kernel_migration" if physical_innovation else "physical_innovation_sigma_migration"
        if record.get(other_key) is not None:
            raise ValueError("checkpoint prefix may not mix policy kernel migration provenance")
        if source_version == source_kernel:
            from pathlib import Path
            expected_keys = {"plan_path", "plan_sha256", "source_checkpoint_sha256",
                "source_runtime_content_sha256", "target_runtime_content_sha256"}
            if not isinstance(migration, dict) or set(migration) != expected_keys:
                raise ValueError("old checkpoint/new prefix kernel requires its exact verified plan binding")
            path = Path(migration["plan_path"]).resolve(strict=True)
            raw = path.read_bytes()
            if (hashlib.sha256(raw).hexdigest() != migration["plan_sha256"]
                    or migration["source_checkpoint_sha256"] != record["checkpoint_sha256"]
                    or migration["source_runtime_content_sha256"] != runtime_hash
                    or migration["target_runtime_content_sha256"] != effective_hash):
                raise ValueError("request-history prefix migration hash/runtime provenance differs")
            plan = json.loads(raw)
            factor = plan.get(factor_key, {})
            if (plan.get("source_checkpoint") != record["checkpoint_path"]
                    or plan.get("source_checkpoint_sha256") != record["checkpoint_sha256"]
                    or plan.get("source_runtime_content_sha256") != runtime_hash
                    or plan.get("target_runtime_content_sha256") != effective_hash
                    or factor.get("schema") != factor_schema
                    or factor.get("source_policy_contract") != source
                    or factor.get("target_policy_contract") != target):
                raise ValueError("request-history prefix does not match source weights and target plan kernel")
        elif (source_version != target["version"] or source != target
                or migration is not None or effective_hash != runtime_hash):
            raise ValueError("request-history prefix exact resume contract differs")
    return record


class FrozenCheckpointPrefixPolicy:
    """Frozen private actor; observations are already semantic-schema normalized.

    Its copied RSL normalizer is still used exactly as in C evaluation. No extra
    normalization, clipping, tanh, or stochastic distribution update is applied.
    """

    def __init__(self, actor: Any, source_checkpoint: Mapping[str, Any]) -> None:
        import torch
        from rsl_rl.models import MLPModel
        from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
        from .semantic_history_actor import (SemanticHistoryMLPModel,
            SemanticTemperedHistoryMLPModel, SemanticQuarterTemperedHistoryMLPModel,
            SemanticCapTransitionQuarterHistoryMLPModel, SemanticFRKneePhysicalInnovationHistoryMLPModel,
            SemanticTaskConditionedHipWheelHistoryMLPModel)
        from .semantic_receiving_wheel_sigma import SemanticReceivingWheelSigmaHistoryMLPModel
        from .semantic_p05_capture_actor import SemanticP05CaptureHistoryMLPModel
        from .semantic_rr_capture_actor import SemanticRRCaptureHistoryMLPModel
        from .semantic_policy_distribution import (HISTORY_TEMPERED_POLICY, HISTORY_QUARTER_TEMPERED_POLICY,
            HISTORY_REQUEST_CAP_TRANSITION_POLICY, FR_KNEE_PHYSICAL_INNOVATION_POLICY, TASK_CONDITIONED_HIP_WHEEL_POLICY)

        record = _source_record(source_checkpoint)
        version = supported_heteroscedastic_contract_version(record["policy_contract"])
        dimension = record["policy_contract"]["observation_dimension"]
        layout = record["policy_contract"].get("observation_layout")
        history_classes = {HISTORY_POLICY: SemanticHistoryMLPModel,
            HISTORY_TEMPERED_POLICY: SemanticTemperedHistoryMLPModel,
            HISTORY_QUARTER_TEMPERED_POLICY: SemanticQuarterTemperedHistoryMLPModel,
            HISTORY_REQUEST_CAP_TRANSITION_POLICY: SemanticCapTransitionQuarterHistoryMLPModel,
            FR_KNEE_PHYSICAL_INNOVATION_POLICY: SemanticFRKneePhysicalInnovationHistoryMLPModel,
            TASK_CONDITIONED_HIP_WHEEL_POLICY: SemanticTaskConditionedHipWheelHistoryMLPModel,
            RECEIVING_WHEEL_POLICY: SemanticReceivingWheelSigmaHistoryMLPModel,
            P05_CAPTURE_POLICY:SemanticP05CaptureHistoryMLPModel,
            RR_CAPTURE_POLICY:SemanticRRCaptureHistoryMLPModel,
            REAR_POLICY_TIMING_POLICY:SemanticRearPolicyTimingHistoryMLPModel,
            P02_PROGRESS_POLICY:SemanticP02ProgressHistoryMLPModel,
            COOPERATIVE_PREP_POLICY:SemanticRearCooperativePrepHistoryMLPModel}
        expected_class = history_classes.get(version, MLPModel)
        if not isinstance(actor, torch.nn.Module) or getattr(actor, "is_recurrent", False):
            raise ValueError("checkpoint prefix requires a nonrecurrent torch actor")
        # At the same input layout both kernels have identical learned tensors.
        # A parameter hash therefore cannot distinguish their inference laws.
        # Reject subclasses and per-instance kernel replacements, rather than
        # relabeling a legacy mean as a history-conditioned mean (or vice versa).
        if (type(actor) is not expected_class
                or getattr(actor.forward, "__func__", None) is not expected_class.forward
                or getattr(actor.get_latent, "__func__", None) is not expected_class.get_latent):
            raise ValueError("checkpoint prefix actor kernel/class disagrees with its policy contract")
        distribution = getattr(actor, "distribution", None)
        if (type(distribution) is not HeteroscedasticGaussianDistribution
                or getattr(distribution.deterministic_output, "__func__", None)
                is not HeteroscedasticGaussianDistribution.deterministic_output
                or distribution.std_type != "log" or distribution.output_dim != 12
                or getattr(actor, "obs_dim", None) != dimension
                or getattr(actor, "observation_layout", None) != layout
                or list(getattr(actor, "obs_groups", ())) != ["policy"]
                or not isinstance(getattr(actor, "obs_normalizer", None), torch.nn.Module)):
            raise ValueError("actor does not implement its declared observation layout and Full12 heteroscedastic RSL interface")
        if version in history_classes and (
                actor.obs_normalization is not False
                or type(actor.obs_normalizer) is not torch.nn.Identity):
            raise ValueError("history prefix requires the identity-normalized stored raw-history kernel")
        parameters = dict(actor.named_parameters())
        if not parameters or any(value.dtype != torch.float32 for value in parameters.values()):
            raise ValueError("checkpoint prefix requires float32 actor parameters")
        device = next(iter(parameters.values())).device
        source_tensors = {**parameters, **dict(actor.named_buffers())}
        if device.type not in ("cpu", "cuda") or any(t.device != device for t in source_tensors.values()):
            raise ValueError("checkpoint prefix actor must have one CPU or CUDA device")
        source_hash = _tensor_hash(parameters.items())
        if source_hash != record["actor_parameter_sha256"]:
            raise ValueError("loaded actor hash differs from the verified source checkpoint")

        # Official RSL caches a Normal whose loc/scale can be nonleaf training
        # tensors. Exclude only this known ephemeral sampling cache via deepcopy
        # memo: never clear/mutate it on the source, or reconstruct/reinitialize
        # the network. Unknown noncopyable state fails explicitly.
        cached = getattr(distribution, "_distribution", None)
        if cached is not None and not isinstance(cached, torch.distributions.Normal):
            raise ValueError("unsupported RSL distribution cache")
        try:
            frozen = copy.deepcopy(actor, {id(cached): None} if cached is not None else {})
        except Exception as error:
            raise ValueError("actor cannot be independently copied without resetting learned state") from error
        if {id(module) for module in actor.modules()} & {id(module) for module in frozen.modules()}:
            raise ValueError("frozen prefix actor shares a source module")
        copied_tensors = {**dict(frozen.named_parameters()), **dict(frozen.named_buffers())}
        if copied_tensors.keys() != source_tensors.keys():
            raise ValueError("frozen prefix copy changed actor parameters or buffers")
        source_storages = {t.untyped_storage().data_ptr() for t in source_tensors.values() if t.numel()}
        for name, original in source_tensors.items():
            copied = copied_tensors[name]
            if (copied.device != original.device or copied.dtype != original.dtype
                    or copied.shape != original.shape
                    or (copied.numel() and copied.untyped_storage().data_ptr() in source_storages)
                    or not torch.equal(copied, original)):
                raise ValueError(f"frozen prefix copy is not independent and equal: {name}")
        if _tensor_hash(frozen.named_parameters()) != source_hash:
            raise ValueError("frozen prefix parameter bytes changed")
        buffer_hash = _tensor_hash(actor.named_buffers())
        if _tensor_hash(frozen.named_buffers()) != buffer_hash:
            raise ValueError("frozen prefix normalizer/buffer bytes changed")
        frozen.eval()
        frozen.requires_grad_(False)
        self._actor = frozen
        self._device = device
        self._observation_dimension = dimension
        self._provenance = {
            **record,
            "prefix_policy_schema": "wlr50_clean.frozen_checkpoint_prefix_policy.v1",
            "frozen_actor_parameter_sha256": source_hash,
            "actor_buffer_sha256": buffer_hash,
            "independent_parameter_and_buffer_storage_verified": True,
            "distribution_cache_copied": False,
            "inference": "TensorDict policy+critic; stochastic_output=False; no projection",
            "frozen_for_entire_training_block": True,
        }

    @property
    def provenance(self) -> dict[str, Any]:
        return copy.deepcopy(self._provenance)

    def __call__(self, observation: tuple[float, ...]) -> tuple[float, ...]:
        import torch
        from tensordict import TensorDict

        if not isinstance(observation, (tuple, list)) or len(observation) != self._observation_dimension:
            raise ValueError(f"checkpoint prefix observation must have exactly {self._observation_dimension} values")
        if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in observation):
            raise ValueError("checkpoint prefix observation must contain finite real values")
        with torch.inference_mode():
            tensor = torch.tensor([observation], dtype=torch.float32, device=self._device)
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("checkpoint prefix observation is outside finite float32 range")
            observations = TensorDict({"policy": tensor, "critic": tensor.clone()},
                                      batch_size=[1], device=self._device)
            action = self._actor(observations, stochastic_output=False)
            if not isinstance(action, torch.Tensor) or tuple(action.shape) != (1, 12):
                raise ValueError("checkpoint prefix actor must return a (1, 12) tensor")
            result = tuple(float(v) for v in action[0].cpu().tolist())
        if not all(math.isfinite(v) for v in result):
            raise ValueError("checkpoint prefix actor returned nonfinite raw latent actions")
        return result


def build_frozen_checkpoint_prefix_policy(
    actor: Any, source_checkpoint: Mapping[str, Any],
) -> FrozenCheckpointPrefixPolicy:
    return FrozenCheckpointPrefixPolicy(actor, source_checkpoint)

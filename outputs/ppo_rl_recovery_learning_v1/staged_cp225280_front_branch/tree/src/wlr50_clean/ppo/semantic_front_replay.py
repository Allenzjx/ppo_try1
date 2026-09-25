"""Declared front-only reference KL in the existing PPO optimizer steps.

Replay is not interaction data and never enters PPO storage or its likelihood.
Its actor gradient is added during backward, before the native actor clipping.
The critic, stored raw samples/advantages/log probabilities and Adam step count
are unchanged. No teacher is deployed in physics. This is a training objective,
not proof of closed-loop retention.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "wlr50_clean.front_replay_regularizer.v1"
DATA_SCHEMA = "wlr50_clean.cp225280_front_replay.v1"
VERSION = "cp225280_front_gaussian_kl_v1"
FRONT_PHASES = {f"P{i:02d}" for i in range(1, 7)}


def validate_replay_spec(spec):
    if (spec.get("schema") != SCHEMA or spec.get("version", VERSION) != VERSION
            or spec.get("coefficient") != 1.0 or spec.get("minibatch_size") != 32):
        raise ValueError("front replay requires its explicit fixed objective")
    path = Path(spec["dataset_path"])
    if not path.is_absolute() or not path.is_file():
        raise ValueError("front replay requires an existing absolute dataset")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != spec.get("dataset_sha256"):
        raise ValueError("front replay dataset bytes changed")
    data = json.loads(payload)
    if data.get("schema") != DATA_SCHEMA:
        raise ValueError("unsupported front replay dataset")
    partitions = []
    for key in ("training_rows", "heldout_rows"):
        rows = data.get(key)
        if not isinstance(rows, list) or not rows:
            raise ValueError("front replay requires nonempty fit and heldout rows")
        ids = set()
        for row in rows:
            phase = row.get("phase")
            index = row.get("decision_index")
            if phase not in FRONT_PHASES or type(index) is not int or index < 0 or index in ids:
                raise ValueError("front replay phase/index is invalid")
            ids.add(index)
            for field, dimension in (("observation", 439), ("reference_mean", 12), ("reference_sigma", 12)):
                value = row.get(field)
                if (not isinstance(value, list) or len(value) != dimension
                        or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
                    raise ValueError("nonfinite/wrong-width replay " + field)
            obs = row["observation"]
            if (any(x not in (0, 1) for x in obs[:13]) or sum(obs[:13]) != 1
                    or obs[int(phase[1:]) - 1] != 1 or any(abs(x) > 20 for x in obs)
                    or any(x <= 0 for x in row["reference_sigma"])):
                raise ValueError("replay observation phase/clip or Gaussian scale invalid")
        partitions.append(ids)
    if partitions[0] & partitions[1] or sum(map(len, partitions)) > 256:
        raise ValueError("replay partitions overlap or exceed the bounded dataset")
    return data


def current_front_gaussian(actor, observation):
    """Pure shared-kernel forward: no distribution-cache write or RNG draw.

    Calling the MLP's forward explicitly keeps the existing official PPO head
    hook scoped to its on-policy minibatch. Replay forwards are separately
    counted. Child modules and the exact deployed HISTORY/sigma math are used.
    """
    from .semantic_history_actor import history_conditioned_head, HISTORY_RHO
    from .semantic_p05_capture_actor import p05_capture_request_history
    from .semantic_rear_owner_actor import validate_rear_owner_latent, rear_owner_effective_log_std
    validate_rear_owner_latent(observation)
    if actor.obs_normalization or actor.is_recurrent:
        raise ValueError("front replay supports only the existing Identity439 actor")
    center, _ = p05_capture_request_history(observation[..., :389])
    head = history_conditioned_head(actor.mlp.forward(observation), center, HISTORY_RHO)
    log_sigma, _ = rear_owner_effective_log_std(head[..., 1, :], observation,
                                               actor.exploration_std_temperature)
    return head[..., 0, :], log_sigma


def reference_kl(mean, log_sigma, reference_mean, reference_sigma):
    """KL(reference || current), summed over the twelve raw Gaussian channels."""
    return (log_sigma - reference_sigma.log()
            + (reference_sigma.square() + (reference_mean - mean).square())
            / (2 * (2 * log_sigma).exp()) - 0.5).sum(-1)


@contextmanager
def add_actor_gradient(parameters, gradients):
    """Exact gradient sum before the optimizer's existing clipping/step.

    The reference gradients are computed before these hooks are installed.
    No parameter, .grad, optimizer state or unrelated global function is edited.
    """
    handles, calls = [], [0] * len(parameters)
    try:
        for index, (parameter, addition) in enumerate(zip(parameters, gradients)):
            if addition is None:
                continue
            def hook(gradient, i=index, extra=addition.detach()):
                calls[i] += 1
                return gradient + extra
            handles.append(parameter.register_hook(hook))
        yield calls
        if any(calls[i] != 1 for i, addition in enumerate(gradients) if addition is not None):
            raise RuntimeError("front replay gradient was not consumed once by the PPO backward")
    finally:
        for handle in handles:
            handle.remove()


class FrontReplayRegularizer:
    def __init__(self, spec, *, device):
        import torch
        self.spec = dict(spec)
        data = validate_replay_spec(spec)
        self.rows = {key: data[key] for key in ("training_rows", "heldout_rows")}
        self.tensors = {key: tuple(torch.tensor([row[field] for row in rows],
                                               dtype=torch.float32, device=device)
                                  for field in ("observation", "reference_mean", "reference_sigma"))
                        for key, rows in self.rows.items()}
        self.minibatches = []

    def evaluate(self, actor, partition):
        import torch
        observations, means, sigmas = self.tensors[partition]
        with torch.no_grad():
            current_mean, current_log_sigma = current_front_gaussian(actor, observations)
            values = reference_kl(current_mean, current_log_sigma, means, sigmas)
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("nonfinite front-retention evaluation")
        return {"rows": len(values), "mean_kl": float(values.mean()), "max_kl": float(values.max()),
                "phase_counts": dict(Counter(row["phase"] for row in self.rows[partition]))}

    @contextmanager
    def minibatch(self, actor, *, global_minibatch_index):
        import torch
        rows = self.rows["training_rows"]
        size = self.spec["minibatch_size"]
        indices = [(global_minibatch_index * size + offset) % len(rows) for offset in range(size)]
        observation, reference_mean, reference_sigma = (item[indices] for item in self.tensors["training_rows"])
        mean, log_sigma = current_front_gaussian(actor, observation)
        kl = reference_kl(mean, log_sigma, reference_mean, reference_sigma).mean()
        parameters = tuple(parameter for parameter in actor.parameters() if parameter.requires_grad)
        gradients = torch.autograd.grad(self.spec["coefficient"] * kl, parameters, allow_unused=True)
        if (not bool(torch.isfinite(kl))
                or any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients)):
            raise RuntimeError("nonfinite front replay objective/gradient")
        row = {"global_minibatch_index": global_minibatch_index, "dataset_indices": indices,
               "source_decision_indices": [rows[i]["decision_index"] for i in indices],
               "phase_counts": dict(Counter(rows[i]["phase"] for i in indices)),
               "kl_reference_to_current": float(kl.detach()), "coefficient": self.spec["coefficient"],
               "unclipped_replay_actor_gradient_norm": math.sqrt(sum(float(g.detach().double().square().sum())
                                                                        for g in gradients if g is not None)),
               "on_policy_samples_added": 0, "separate_optimizer_steps": 0,
               "extra_actor_forwards": 1, "extra_random_draws": 0}
        with add_actor_gradient(parameters, gradients) as calls:
            yield row
        row["gradient_consumed_once"] = all(calls[i] == 1 for i, gradient in enumerate(gradients) if gradient is not None)
        self.minibatches.append(row)

    def begin_update(self, actor):
        self.minibatches = []
        return {key: self.evaluate(actor, key) for key in self.rows}

    def finish_update(self, actor, before, expected_minibatches):
        if len(self.minibatches) != expected_minibatches:
            raise RuntimeError("front replay did not enter every actual PPO optimizer step")
        counts = Counter()
        for row in self.minibatches:
            counts.update(row["phase_counts"])
        return {"schema": SCHEMA, "spec": self.spec, "before": before,
                "after": {key: self.evaluate(actor, key) for key in self.rows},
                "minibatches": self.minibatches, "actual_replay_row_exposures": sum(counts.values()),
                "actual_replay_phase_exposures": dict(counts), "on_policy_samples_added": 0,
                "separate_auxiliary_optimizer_steps": 0, "realtime_teacher_deployed": False,
                "gradient_semantics": "PPO_actor_gradient_plus_reference_KL_gradient_before_native_actor_norm_clip_same_Adam_step",
                "physical_front_retention_proven": False}

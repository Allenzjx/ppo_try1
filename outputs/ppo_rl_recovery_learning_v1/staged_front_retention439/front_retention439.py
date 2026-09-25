"""Dormant finite owner439 front-retention AUX kernel; never runs on import.

The only mutable leaf is ``actor.mlp.0.weight[:, [1,4,5,8]]``: the exact
P02/P05/P06/P09 one-hot columns.  Targets are the student's own deterministic
raw requests that were actually applied before probe-v2 intervention.  They are
not PPO samples, teacher actions, task successes, nominal commands, or FINAL
actuator targets.

Torch and production modules are imported lazily by numeric entry points.  The
stdlib contract and source tests can therefore run while simulation is active.
This file has no CLI, checkpoint writer, publisher, or import-time fit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "wlr50_clean.finite_front_retention439_phase_columns_aux.v1"
FIT_SCHEMA = SCHEMA + ".fit"
INSPECTION_SCHEMA = SCHEMA + ".inspection"
PHASE_COLUMNS = (1, 4, 5, 8)
TARGET_PHASES = ("P02", "P05", "P06", "P09")
INVARIANCE_PHASES = ("P10", "P11", "P12", "P13")
REAL_INVARIANCE_PHASES = ("P10", "P11", "P12")
INVARIANCE_SCHEMA = "wlr50_clean.front_retention439.real_P10_P12_synthetic_P13_invariance.v1"
OPTIMIZED_PARAMETERS = ("actor.mlp.0.weight[:,[1,4,5,8]]",)
OPTIMIZED_SCALAR_COUNT = 256 * len(PHASE_COLUMNS)
CHANNELS = ("FL_hip", "FL_knee", "FR_hip", "FR_knee", "RL_hip", "RL_knee",
            "RR_hip", "RR_knee", "FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel")
UNITS = ("deg",) * 8 + ("rad/s",) * 4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Budget:
    """One reviewed constant-LR budget.  No default and no LR sweep."""
    max_attempts: int
    learning_rate: float
    maximum_train_request_shift_full12: tuple[float, ...]
    maximum_validation_request_shift_full12: tuple[float, ...]
    maximum_per_state_full_gaussian_kl: float
    maximum_abs_log_sigma_change: float

    def validate(self) -> None:
        require(type(self.max_attempts) is int and 1 <= self.max_attempts <= 32,
                "finite AUX budget requires 1..32 total attempts")
        for name in ("learning_rate", "maximum_per_state_full_gaussian_kl",
                     "maximum_abs_log_sigma_change"):
            value = getattr(self, name)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                    "budget requires positive finite " + name)
        for name in ("maximum_train_request_shift_full12",
                     "maximum_validation_request_shift_full12"):
            values = getattr(self, name)
            require(isinstance(values, (tuple, list)) and len(values) == 12 and
                    all(type(value) in (int, float) and math.isfinite(value) and value > 0
                        for value in values),
                    "budget requires twelve positive finite REQUEST bounds: " + name)


def validate_fit_report(report: dict[str, Any], *, publishable: bool = False) -> None:
    """Stdlib ledger-facing validation; it does not trust the fit implementation."""
    require(isinstance(report, dict) and report.get("schema") == FIT_SCHEMA and
            report.get("optimized_parameters") == list(OPTIMIZED_PARAMETERS) and
            report.get("optimized_scalar_count") == OPTIMIZED_SCALAR_COUNT,
            "front-retention439 report has wrong schema or sparse whitelist")
    accepted = report.get("accepted_auxiliary_updates")
    attempted = report.get("attempted_auxiliary_optimizer_steps")
    require(type(accepted) is int and type(attempted) is int and
            0 <= accepted <= attempted <= 32 and (accepted > 0 if publishable else True),
            "invalid finite AUX accepted/attempted counts")
    budget_payload = report.get("budget")
    require(isinstance(budget_payload, dict) and
            set(budget_payload) == {"max_attempts", "learning_rate",
                "maximum_train_request_shift_full12",
                "maximum_validation_request_shift_full12",
                "maximum_per_state_full_gaussian_kl",
                "maximum_abs_log_sigma_change"},
            "fit report requires the exact reviewed finite budget")
    Budget(**budget_payload).validate()
    require(attempted <= budget_payload["max_attempts"],
            "attempted AUX steps exceed reviewed finite budget")
    stop_reason = report.get("stop_reason")
    rejected = report.get("first_rejected_proposal_restored_and_stopped")
    allowed_nonrejection = {"finite_budget_exhausted",
        "selected_student_raw_targets_already_match",
        "zero_sparse_gradient_no_optimizer_step"}
    require((stop_reason == "first_rejected_proposal_restored_and_stopped" and
             rejected is True) or
            (stop_reason in allowed_nonrejection and rejected is False),
            "fit stop/rejection receipt differs")
    require(all(report.get(key) == 0 for key in
                ("PPO_decisions_added", "PPO_updates_added",
                 "PPO_optimizer_steps_added")) and
            report.get("real_P10_P12_full_Gaussian_bitwise_unchanged") is True and
            report.get("synthetic_P13_full_Gaussian_bitwise_unchanged") is True and
            report.get("zero_selected_phase_columns_P10_P13_algebra_verified") is True and
            report.get("nonselected_actor_state_unchanged") is True and
            report.get("fresh_PPO_rollout_required") is True and
            report.get("targets_were_executed_student_raw_requests") is True and
            report.get("targets_were_success_or_teacher_labels") is False and
            report.get("teacher_deployed") is False and
            report.get("physical_success_claimed") is False,
            "front-retention439 report changes credit or evidence semantics")
    before = report.get("protected_state_before")
    after = report.get("protected_state_after")
    keys = {"critic_parameter_sha256", "optimizer_state_sha256",
            "normalizer_state_sha256", "training_rng_state_sha256",
            "optimizer_learning_rate"}
    require(isinstance(before, dict) and set(before) == keys and before == after,
            "critic/PPO-Adam/LR/RNG/Identity state changed")
    require(all(isinstance(before[key], str) and
                re.fullmatch(r"[0-9a-f]{64}", before[key]) is not None
                for key in keys - {"optimizer_learning_rate"}) and
            type(before["optimizer_learning_rate"]) in (int, float) and
            math.isfinite(before["optimizer_learning_rate"]) and
            isinstance(report.get("actor_parameter_sha256_before"), str) and
            re.fullmatch(r"[0-9a-f]{64}", report["actor_parameter_sha256_before"]) is not None and
            isinstance(report.get("actor_parameter_sha256_after"), str) and
            re.fullmatch(r"[0-9a-f]{64}", report["actor_parameter_sha256_after"]) is not None,
            "fit report hashes or protected learning rate differ")
    _reference_binding(report.get("data_receipt"), verify_file=False)
    validate_invariance_evidence(report.get("invariance_evidence"))
    require("same_input_P10_P13_full_Gaussian_bitwise_unchanged" not in report,
            "legacy mixed actual/synthetic invariance claim is not accepted")
    if accepted:
        require(report.get("actor_parameter_sha256_before") !=
                report.get("actor_parameter_sha256_after"),
                "accepted AUX must change the sparse actor leaf")


def validate_invariance_evidence(evidence: Any) -> None:
    """Explicitly distinguish existing real rows from synthetic P13 coverage."""
    require(isinstance(evidence, dict) and evidence.get("schema") == INVARIANCE_SCHEMA,
            "explicit real/synthetic invariance evidence required")
    counts = evidence.get("actual_rows_by_phase")
    require(isinstance(counts, dict) and set(counts) == set(INVARIANCE_PHASES)
            and all(type(counts[p]) is int and counts[p] > 0 for p in REAL_INVARIANCE_PHASES)
            and type(counts["P13"]) is int and counts["P13"] == 0
            and evidence.get("real_P13_validation_claimed") is False
            and type(evidence.get("synthetic_P13_rows")) is int and evidence["synthetic_P13_rows"] == 3
            and evidence.get("synthetic_rows_used_for_fit") is False
            and evidence.get("synthetic_P13_origin") == "first_real_row_per_P10_P12_with_only_phase_onehot_replaced_test_fixture"
            and evidence.get("selected_phase_columns") == list(PHASE_COLUMNS)
            and evidence.get("same_future_trajectory_claimed") is False,
            "real P10-P12 / synthetic-only P13 evidence scope differs")
    for key in ("real_observations_float32_sha256", "synthetic_P13_observations_float32_sha256"):
        require(isinstance(evidence.get(key), str) and
                re.fullmatch(r"[0-9a-f]{64}", evidence[key]) is not None,
                "invariance observation matrix hash required")


def _numeric_dependencies() -> dict[str, Any]:
    """Explicit future numeric boundary; never called by stdlib tests/import."""
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    import torch
    from tensordict import TensorDict
    from wlr50_clean.ppo import semantic_training as training
    from wlr50_clean.ppo.semantic_history_actor import HISTORY_RHO, history_conditioned_head
    from wlr50_clean.ppo.semantic_migration import digest
    from wlr50_clean.ppo.semantic_p05_capture_actor import p05_capture_request_history
    from wlr50_clean.ppo.semantic_rear_owner_actor import (
        SemanticRearOwnerRecoveryHistoryMLPModel, rear_owner_effective_log_std)
    return {"torch": torch, "TensorDict": TensorDict, "training": training,
        "HISTORY_RHO": HISTORY_RHO, "history_conditioned_head": history_conditioned_head,
        "digest": digest, "p05_capture_request_history": p05_capture_request_history,
        "Actor": SemanticRearOwnerRecoveryHistoryMLPModel,
        "effective_log_std": rear_owner_effective_log_std}


def _reference_binding(binding: Any, *, verify_file: bool) -> dict[str, str]:
    """Validate the small selected-row receipt reference, never the large source log."""
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"} and
            isinstance(binding.get("path"), str) and
            isinstance(binding.get("sha256"), str) and
            re.fullmatch(r"[0-9a-f]{64}", binding["sha256"]) is not None,
            "data receipt requires exact {path,sha256} binding")
    if verify_file:
        path = Path(binding["path"]).resolve(strict=True)
        require(str(path) == binding["path"], "data receipt path must be resolved and exact")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == binding["sha256"], "selected-row data receipt changed")
    return {"path": binding["path"], "sha256": binding["sha256"]}


def _validate_observation(x: Any, deps: dict[str, Any]) -> None:
    torch = deps["torch"]
    require(isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.ndim == 2 and
            x.shape[1] == 439 and 1 <= len(x) <= 4096 and
            x.device.type in ("cpu", "cuda") and bool(torch.isfinite(x).all()) and
            bool((x.abs() <= 20).all()),
            "observations require bounded source-device float32 N x 439")
    bits = x[:, :13]
    require(bool(((bits == 0) | (bits == 1)).all()) and
            bool((bits.sum(-1) == 1).all()),
            "phase features must be exact P01-P13 one-hot")


def tensors(observations: Any, *, device: str = "cpu", deps: dict[str, Any] | None = None) -> Any:
    deps = deps or _numeric_dependencies()
    torch = deps["torch"]
    x = torch.as_tensor(observations, dtype=torch.float32, device=device)
    _validate_observation(x, deps)
    return deps["TensorDict"]({"policy": x, "critic": x.clone()},
                              batch_size=[len(x)], device=x.device)


def first_layer(actor: Any, deps: dict[str, Any] | None = None) -> Any:
    deps = deps or _numeric_dependencies()
    torch = deps["torch"]
    require(type(actor) is deps["Actor"] and actor.obs_dim == 439 and
            not actor.is_recurrent and actor.exploration_std_temperature == .25 and
            deps["HISTORY_RHO"] == .9 and type(actor.obs_normalizer) is torch.nn.Identity,
            "exact owner439/full12/Identity/quarter-HISTORY actor required")
    require(len(actor.mlp) == 6 and type(actor.mlp[0]) is torch.nn.Linear and
            type(actor.mlp[1]) is torch.nn.ELU and type(actor.mlp[2]) is torch.nn.Linear and
            type(actor.mlp[3]) is torch.nn.ELU and type(actor.mlp[4]) is torch.nn.Linear and
            type(actor.mlp[5]) is torch.nn.Unflatten and
            tuple(actor.mlp[0].weight.shape) == (256, 439) and
            tuple(actor.mlp[2].weight.shape) == (256, 256) and
            tuple(actor.mlp[4].weight.shape) == (24, 256),
            "unexpected canonical owner439 256/256 mean+log-sigma MLP")
    layer = actor.mlp[0]
    require(dict(actor.named_parameters()).get("mlp.0.weight") is layer.weight and
            all(parameter.device == layer.weight.device and parameter.dtype == torch.float32
                for parameter in actor.parameters()),
            "actor parameter binding/device differs")
    return layer


def distribution(actor: Any, observations: Any, leaf: Any | None = None,
                 deps: dict[str, Any] | None = None) -> dict[str, Any]:
    """Pure official mean/log-sigma kernel; no cache, draw, or RNG mutation."""
    deps = deps or _numeric_dependencies()
    torch = deps["torch"]
    layer = first_layer(actor, deps)
    latent = observations["policy"]
    _validate_observation(latent, deps)
    require(latent.device == layer.weight.device and
            torch.equal(actor.get_latent(observations), latent),
            "Identity latent or source device differs")
    center, evidence = deps["p05_capture_request_history"](latent[:, :389])
    if leaf is None:
        raw_head = actor.mlp(latent)
    else:
        require(tuple(leaf.shape) == (256, 4) and leaf.device == layer.weight.device and
                leaf.dtype == layer.weight.dtype,
                "temporary leaf must be the exact sparse 256x4 columns")
        columns = torch.tensor(PHASE_COLUMNS, dtype=torch.long, device=layer.weight.device)
        weight = layer.weight.detach().index_copy(1, columns, leaf)
        raw_head = torch.func.functional_call(actor.mlp, {"0.weight": weight}, (latent,))
    head = deps["history_conditioned_head"](raw_head, center, deps["HISTORY_RHO"])
    log_sigma, _ = deps["effective_log_std"](
        head[:, 1, :], latent, actor.exploration_std_temperature)
    mean = head[:, 0, :]
    caps = evidence["current_cap_full12"]
    return {"mean": mean, "log_sigma": log_sigma, "sigma": log_sigma.exp(),
            "request": caps * mean.tanh(), "caps": caps,
            "network_mean": raw_head[:, 0, :], "history": center}


def _targets(values: Any, rows: int, device: Any, deps: dict[str, Any]) -> Any:
    torch = deps["torch"]
    value = torch.as_tensor(values, dtype=torch.float32, device=device)
    require(tuple(value.shape) == (rows, 12) and bool(torch.isfinite(value).all()),
            "targets must be actual applied deterministic student raw12")
    return value


def _dataset(actor: Any, train_obs: Any, train_targets: Any, holdout_obs: Any,
             holdout_targets: Any, invariance_obs: Any,
             deps: dict[str, Any]) -> tuple[Any, Any]:
    layer = first_layer(actor, deps)
    for observations in (train_obs, holdout_obs, invariance_obs):
        _validate_observation(observations["policy"], deps)
        require(observations["policy"].device == layer.weight.device,
                "all data and actor must share the source device")
    tx, hx, ix = (value["policy"] for value in
                  (train_obs, holdout_obs, invariance_obs))
    torch = deps["torch"]
    selected = torch.tensor(PHASE_COLUMNS, dtype=torch.long, device=tx.device)
    train_phase = tx[:, :13].argmax(-1)
    holdout_phase = hx[:, :13].argmax(-1)
    train_counts = [int((train_phase == column).sum()) for column in PHASE_COLUMNS]
    holdout_counts = [int((holdout_phase == column).sum()) for column in PHASE_COLUMNS]
    require(bool((tx.index_select(1, selected).sum(-1) == 1).all()) and
            bool((hx.index_select(1, selected).sum(-1) == 1).all()) and
            set(train_phase.tolist()) == set(PHASE_COLUMNS) and
            set(holdout_phase.tolist()) == set(PHASE_COLUMNS) and
            len(set(train_counts)) == len(set(holdout_counts)) == 1 and
            train_counts == holdout_counts,
            "train/holdout must be balanced P02/P05/P06/P09 only")
    require(bool((ix.index_select(1, selected) == 0).all()) and
            set(ix[:, :13].argmax(-1).tolist()) == {9, 10, 11},
            "same-input actual invariance requires available P10-P12; P13 is synthetic-only")
    train_bytes = {row.numpy().tobytes() for row in tx.detach().cpu()}
    require(not any(row.numpy().tobytes() in train_bytes for row in hx.detach().cpu()),
            "train and complete holdout observations must be disjoint")
    return (_targets(train_targets, len(tx), tx.device, deps),
            _targets(holdout_targets, len(hx), hx.device, deps))


def _detached(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item.detach().clone() for key, item in value.items()}


def _exact_gaussian(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return all(before[key].equal(after[key]) for key in
               ("mean", "log_sigma", "sigma", "request"))


def _synthetic_p13_invariance(invariance_obs: Any, deps: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Synthetic branch unit fixture, NEVER a claimed reached robot state.

    Change only the phase one-hot in one existing row of each P10/P11/P12.
    These three vectors enter invariance tests only, not targets or fitting.
    All P13 claims remain same-input algebra/numerics, never physical coverage.
    """
    torch = deps["torch"]
    x = invariance_obs["policy"]
    phase = x[:, :13].argmax(-1)
    require(set(phase.tolist()) == {9, 10, 11}, "synthetic fixture needs available real P10-P12")
    indices = [(phase == column).nonzero()[0, 0] for column in (9, 10, 11)]
    synthetic = x[torch.stack(indices)].detach().clone()
    synthetic[:, :13] = 0.
    synthetic[:, 12] = 1.
    evidence = dict(schema=INVARIANCE_SCHEMA,
        actual_rows_by_phase={name: int((phase == column).sum())
                              for column, name in enumerate(INVARIANCE_PHASES, 9)},
        real_P13_validation_claimed=False, synthetic_P13_rows=3,
        synthetic_P13_origin="first_real_row_per_P10_P12_with_only_phase_onehot_replaced_test_fixture",
        synthetic_rows_used_for_fit=False, selected_phase_columns=list(PHASE_COLUMNS),
        real_observations_float32_sha256=hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        synthetic_P13_observations_float32_sha256=hashlib.sha256(synthetic.cpu().contiguous().numpy().tobytes()).hexdigest(),
        same_future_trajectory_claimed=False)
    validate_invariance_evidence(evidence)
    return tensors(synthetic, device=str(x.device), deps=deps), evidence


def gaussian_change(before: dict[str, Any], after: dict[str, Any],
                    deps: dict[str, Any] | None = None) -> dict[str, Any]:
    deps = deps or _numeric_dependencies()
    torch = deps["torch"]
    old_sigma, new_sigma = before["sigma"].double(), after["sigma"].double()
    log_ratio = (new_sigma / old_sigma).log()
    delta = after["mean"].double() - before["mean"].double()
    forward = (log_ratio + .5 * (torch.expm1(-2 * log_ratio) +
                                (delta / new_sigma).square())).sum(-1).clamp_min(0)
    reverse = (-log_ratio + .5 * (torch.expm1(2 * log_ratio) +
                                 (delta / old_sigma).square())).sum(-1).clamp_min(0)
    return {"kl_original_to_candidate": forward,
            "kl_candidate_to_original": reverse,
            "log_sigma_delta": after["log_sigma"] - before["log_sigma"],
            "raw_mean_delta": after["mean"] - before["mean"],
            "requested_residual_delta": after["request"] - before["request"]}


def inspect(actor: Any, train_obs: Any, train_targets: Any, holdout_obs: Any,
            holdout_targets: Any, invariance_obs: Any, *, seed: int) -> dict[str, Any]:
    """Future legal-boundary read-only gradient inspection; performs no SGD step."""
    deps = _numeric_dependencies()
    torch, training = deps["torch"], deps["training"]
    target, held_target = _dataset(actor, train_obs, train_targets, holdout_obs,
                                   holdout_targets, invariance_obs, deps)
    actor_hash = training.parameter_hash(actor)
    rng = training.capture_training_rng_state(seed=seed)
    base, held, invariant = (_detached(distribution(actor, value, deps=deps))
                             for value in (train_obs, holdout_obs, invariance_obs))
    _, invariance_evidence = _synthetic_p13_invariance(invariance_obs, deps)
    leaf = torch.nn.Parameter(first_layer(actor, deps).weight[:, PHASE_COLUMNS]
                              .detach().clone())
    candidate = distribution(actor, train_obs, leaf, deps)
    loss = .5 * (candidate["mean"] - target).square().mean()
    gradient, = torch.autograd.grad(loss, (leaf,))
    require(bool(torch.isfinite(gradient).all()) and
            training.parameter_hash(actor) == actor_hash and
            training.capture_training_rng_state(seed=seed) == rng,
            "inspection changed actor/RNG or produced nonfinite gradient")
    return {"schema": INSPECTION_SCHEMA, "automatic_aux_enabled": False,
        "optimized_parameters": list(OPTIMIZED_PARAMETERS),
        "optimized_scalar_count": OPTIMIZED_SCALAR_COUNT,
        "initial_train_raw_half_MSE": float(loss),
        "initial_holdout_raw_half_MSE": float(
            .5 * (held["mean"] - held_target).square().mean()),
        "selected_gradient_l2": float(gradient.norm()),
        "selected_gradient_by_phase_column_l2":
            gradient.norm(dim=0).detach().cpu().tolist(),
        "invariance_baseline_rows": len(invariant["mean"]),
        "invariance_evidence": invariance_evidence,
        "optimizer_steps_performed": 0, "PPO_credit": 0,
        "targets_were_executed_student_raw_requests": True,
        "targets_were_success_or_teacher_labels": False,
        "physical_success_claimed": False}


def _protected_state(runner: Any, seed: int, deps: dict[str, Any]) -> dict[str, Any]:
    training = deps["training"]
    rng = training.capture_training_rng_state(seed=seed)
    return {"critic_parameter_sha256": training.parameter_hash(runner.alg.critic),
        "optimizer_state_sha256": training.state_hash(runner.alg.optimizer.state_dict()),
        "normalizer_state_sha256": training.state_hash(training._normalizers(runner)),
        "training_rng_state_sha256": deps["digest"](rng),
        "optimizer_learning_rate": training.optimizer_learning_rate(runner)}


def fit(runner: Any, train_obs: Any, train_targets: Any, holdout_obs: Any,
        holdout_targets: Any, invariance_obs: Any, *, seed: int, budget: Budget,
        data_receipt: dict[str, str], authorized: bool = False) -> dict[str, Any]:
    """Finite independent SGD; first rejected proposal is restored and stops.

    This function does not save or publish.  Only the last accepted sparse leaf
    is copied into the actor after all checks.  PPO Adam is never stepped.
    """
    require(authorized is True, "front-retention439 fit requires explicit authorization")
    budget.validate()
    data_binding = _reference_binding(data_receipt, verify_file=True)
    deps = _numeric_dependencies()
    torch, training = deps["torch"], deps["training"]
    require(runner.alg.storage.step == 0 and runner.alg.transition.actions is None,
            "AUX requires a legal checkpoint boundary and fresh PPO rollout")
    actor = runner.alg.actor
    target, held_target = _dataset(actor, train_obs, train_targets, holdout_obs,
                                   holdout_targets, invariance_obs, deps)
    layer = first_layer(actor, deps)
    original_state = {key: value.detach().clone()
                      for key, value in actor.state_dict().items()}
    original_gradients = {key: None if parameter.grad is None else parameter.grad.detach().clone()
                          for key, parameter in actor.named_parameters()}
    actor_before = training.parameter_hash(actor)
    protected_before = _protected_state(runner, seed, deps)
    base, held, invariant = (_detached(distribution(actor, value, deps=deps))
                             for value in (train_obs, holdout_obs, invariance_obs))
    synthetic_p13_obs, invariance_evidence = _synthetic_p13_invariance(invariance_obs, deps)
    synthetic_invariant = _detached(distribution(actor, synthetic_p13_obs, deps=deps))
    leaf = torch.nn.Parameter(layer.weight[:, PHASE_COLUMNS].detach().clone())
    optimizer = torch.optim.SGD([leaf], lr=budget.learning_rate,
                                momentum=0.0, weight_decay=0.0)
    records: list[dict[str, Any]] = []
    accepted = attempted = 0
    stop_reason = "finite_budget_exhausted"
    rejected_restored = False
    for attempt in range(1, budget.max_attempts + 1):
        current = distribution(actor, train_obs, leaf, deps)
        loss = .5 * (current["mean"] - target).square().mean()
        require(bool(torch.isfinite(loss)), "nonfinite auxiliary loss")
        if float(loss) < 1e-14:
            stop_reason = "selected_student_raw_targets_already_match"
            break
        gradient, = torch.autograd.grad(loss, (leaf,))
        require(bool(torch.isfinite(gradient).all()), "nonfinite sparse auxiliary gradient")
        if not bool(gradient.any()):
            stop_reason = "zero_sparse_gradient_no_optimizer_step"
            break
        prior_leaf = leaf.detach().clone()
        leaf.grad = gradient
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        attempted += 1
        rejection: list[str] = []
        with torch.no_grad():
            candidate = distribution(actor, train_obs, leaf, deps)
            held_candidate = distribution(actor, holdout_obs, leaf, deps)
            invariant_candidate = distribution(actor, invariance_obs, leaf, deps)
            synthetic_candidate = distribution(actor, synthetic_p13_obs, leaf, deps)
            train_change = gaussian_change(base, candidate, deps)
            holdout_change = gaussian_change(held, held_candidate, deps)
            next_loss = .5 * (candidate["mean"] - target).square().mean()
            if torch.equal(leaf, prior_leaf):
                rejection.append("numerically_unchanged_sparse_proposal")
            if not all(bool(torch.isfinite(value).all()) for payload in
                       (candidate, held_candidate, train_change, holdout_change)
                       for value in payload.values()):
                rejection.append("nonfinite_candidate")
            if float(next_loss) > float(loss):
                rejection.append("training_raw_target_loss_increased")
            for name, change, limit in (
                    ("train", train_change, budget.maximum_train_request_shift_full12),
                    ("holdout", holdout_change,
                     budget.maximum_validation_request_shift_full12)):
                if bool((change["requested_residual_delta"].abs().amax(0) >
                         change["requested_residual_delta"].new_tensor(limit)).any()):
                    rejection.append(name + "_REQUEST_bound")
                if max(float(change["kl_original_to_candidate"].max()),
                       float(change["kl_candidate_to_original"].max())) > \
                        budget.maximum_per_state_full_gaussian_kl:
                    rejection.append(name + "_bidirectional_full_Gaussian_KL_bound")
                if float(change["log_sigma_delta"].abs().max()) > \
                        budget.maximum_abs_log_sigma_change:
                    rejection.append(name + "_log_sigma_bound")
            if not _exact_gaussian(invariant, invariant_candidate):
                rejection.append("real_P10_P12_same_input_full_Gaussian_changed")
            if not _exact_gaussian(synthetic_invariant, synthetic_candidate):
                rejection.append("synthetic_P13_same_input_full_Gaussian_changed")
        record = {"attempt": attempt, "accepted": not rejection,
                  "rejection_reasons": rejection,
                  "train_raw_half_MSE_before": float(loss),
                  "train_raw_half_MSE_after": float(next_loss),
                  "actual_movement_from_original": {
                      name: {
                          "per_channel_max_abs_REQUEST_delta_full12":
                              change["requested_residual_delta"].abs().amax(0).detach().cpu().tolist(),
                          "maximum_KL_original_to_candidate": float(change["kl_original_to_candidate"].max()),
                          "maximum_KL_candidate_to_original": float(change["kl_candidate_to_original"].max()),
                          "maximum_abs_log_sigma_delta": float(change["log_sigma_delta"].abs().max())}
                      for name, change in (("train", train_change), ("holdout", holdout_change))}}
        records.append(record)
        if rejection:
            with torch.no_grad():
                leaf.copy_(prior_leaf)
            require(torch.equal(leaf, prior_leaf), "rejected sparse leaf restore failed")
            rejected_restored = True
            stop_reason = "first_rejected_proposal_restored_and_stopped"
            break
        accepted += 1
    with torch.no_grad():
        if accepted:
            columns = torch.tensor(PHASE_COLUMNS, dtype=torch.long, device=layer.weight.device)
            layer.weight.index_copy_(1, columns, leaf)
    selected = set(PHASE_COLUMNS)
    nonselected_unchanged = True
    for name, value in actor.state_dict().items():
        before = original_state[name]
        if name == "mlp.0.weight":
            keep = [index for index in range(value.shape[1]) if index not in selected]
            nonselected_unchanged &= torch.equal(value[:, keep], before[:, keep])
        else:
            nonselected_unchanged &= torch.equal(value, before)
    require(nonselected_unchanged, "nonselected actor state changed")
    for name, parameter in actor.named_parameters():
        expected = original_gradients[name]
        require((expected is None and parameter.grad is None) or
                (expected is not None and parameter.grad is not None and
                 torch.equal(expected, parameter.grad)),
                "existing actor gradients changed")
    final_invariant = distribution(actor, invariance_obs, deps=deps)
    invariant_equal = _exact_gaussian(invariant, final_invariant)
    synthetic_equal = _exact_gaussian(synthetic_invariant, distribution(actor, synthetic_p13_obs, deps=deps))
    # The exact first_layer contract includes Identity and the complete MLP;
    # with all other parameters untouched, delta_W @ same_x == 0 for every
    # finite same-input P10-P13 x whose selected one-hot columns are zero.
    algebra_verified = nonselected_unchanged and bool(torch.isfinite(layer.weight).all())
    protected_after = _protected_state(runner, seed, deps)
    require(invariant_equal and synthetic_equal and algebra_verified and protected_after == protected_before,
            "actual/synthetic same-input Gaussian, sparse algebra, or protected PPO state changed")
    actor_after = training.parameter_hash(actor)
    require((accepted == 0 and actor_after == actor_before) or
            (accepted > 0 and actor_after != actor_before),
            "accepted sparse AUX/actor hash relationship differs")
    report = {"schema": FIT_SCHEMA, "budget": asdict(budget),
        "data_receipt": data_binding,
        "optimized_parameters": list(OPTIMIZED_PARAMETERS),
        "optimized_scalar_count": OPTIMIZED_SCALAR_COUNT,
        "accepted_auxiliary_updates": accepted,
        "attempted_auxiliary_optimizer_steps": attempted,
        "stop_reason": stop_reason,
        "first_rejected_proposal_restored_and_stopped": rejected_restored,
        "actor_parameter_sha256_before": actor_before,
        "actor_parameter_sha256_after": actor_after,
        "protected_state_before": protected_before,
        "protected_state_after": protected_after,
        "nonselected_actor_state_unchanged": nonselected_unchanged,
        "real_P10_P12_full_Gaussian_bitwise_unchanged": invariant_equal,
        "synthetic_P13_full_Gaussian_bitwise_unchanged": synthetic_equal,
        "zero_selected_phase_columns_P10_P13_algebra_verified": algebra_verified,
        "invariance_evidence": invariance_evidence,
        "fresh_PPO_rollout_required": True,
        "targets_were_executed_student_raw_requests": True,
        "targets_were_success_or_teacher_labels": False,
        "teacher_deployed": False, "physical_success_claimed": False,
        "PPO_decisions_added": 0, "PPO_updates_added": 0,
        "PPO_optimizer_steps_added": 0,
        "attempt_records": records,
        "same_future_trajectory_claimed": False}
    validate_fit_report(report)
    return report

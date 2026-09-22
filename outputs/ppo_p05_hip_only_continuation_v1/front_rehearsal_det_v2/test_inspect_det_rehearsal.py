"""Synthetic CPU interface checks; no real checkpoint fit or physical credit."""
import copy
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import inspect_det_rehearsal as m
from wlr50_clean.ppo.semantic_p05_capture_migration import _ObservationOnlyEnv


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    rng = torch.get_rng_state(); threads = torch.get_num_threads()
    torch.set_num_threads(1); torch.manual_seed(619)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_rng_state(rng); torch.set_num_threads(threads)


def runner():
    return m.training.construct_semantic_runner(_ObservationOnlyEnv(389), seed=1001, device="cpu",
        initialize_actor=False, policy_version=m.P05_CAPTURE_POLICY,
        observation_layout=m.P05_CAPTURE_OBSERVATION_LAYOUT)[0]


def states(actor):
    x = torch.zeros(6, 389); x[:, 1] = 1; x[:, 20] = .01
    x[:, 30] = torch.linspace(-.1, .1, 6)
    x[:, 195:207] = torch.linspace(-.2, .2, 72).reshape(6, 12)
    v = x.clone(); v[:, 30] += .031
    p01 = x[:2].clone(); p01[:, :13] = 0; p01[:, 0] = 1
    ix = x[:4].clone(); ix[:, :13] = 0
    ix[torch.arange(4), torch.tensor([2, 3, 4, 5])] = 1
    with torch.no_grad():
        target = m.kernel.distribution(actor, m._tensor(x))["mean"] + .015
        vt = m.kernel.distribution(actor, m._tensor(v))["mean"] + .015
    receipt = {"deterministic_source": {"observation_provenance": {
        "direct389_tensor_was_saved_by_source": False}, "synthetic_fixture_only": True},
        "current_MDP_training_admission_claimed": False}
    return {"train_observations": x, "train_raw_targets": target,
            "validation_observations": v, "validation_raw_targets": vt,
            "p01_observations": p01, "invariance_observations": ix, "receipt": receipt}


def candidate():
    x = np.zeros((254, 389), dtype=np.float32); x[:, 1] = 1
    x[:, 30] = np.linspace(-.1, .1, 254)
    arrays = {"X389_reconstructed": x, "raw12_actual_deterministic": np.zeros((254, 12), dtype=np.float32),
              "source_conditional_mean12": np.zeros((254, 12), dtype=np.float32),
              "source_effective_sigma12": np.ones((254, 12), dtype=np.float32),
              "source_history_center12": np.zeros((254, 12), dtype=np.float32),
              "source_decision": np.arange(3, 257), "input_tick": np.arange(16, 2041, 8),
              "suggested_train_indices": np.arange(0, 254, 3),
              "suggested_validation_indices": np.arange(1, 254, 3), "source_only_indices": np.arange(2, 254, 3)}
    errors = dict.fromkeys(("conditional_mean", "base_mean", "history_center", "learned_sigma",
                           "effective_sigma", "effective_log_std", "effective_innovation_sigma_multiplier"), 0.)
    checks = [{"candidate_index": i, "decision": i + 3, "start_tick": 16 + 8 * i, "end_tick": 24 + 8 * i,
               "reconstructed_float32_observation_sha256": hashlib.sha256(x[i].tobytes()).hexdigest(),
               "history_and_assist_exact": True, "all12_permission_no_assist_each_physics_tick": True,
               "errors_max_abs": copy.deepcopy(errors)} for i in range(254)]
    manifest = {"schema": "wlr50_clean.det_front_reconstruction_candidate.v1", "status": "CANDIDATE_ONLY_NOT_ADMITTED_TO_LEARNING",
        "observation_provenance": {"direct389_tensor_was_saved_by_source": False,
            "reconstructed_from_explicit_synchronized_fields": True, "missing_fields_guessed_or_filled": False,
            "original_CUDA_vs_CPU_is_not_bitwise_proof": True, "matching12_outputs_alone_cannot_prove389_input_identity": True},
        "dataset": {"rows": 254, "phase": "P02", "shapes": {key: list(a.shape) for key, a in arrays.items()}},
        "checks": {"numeric_atol_fixed_before_replay": 1e-6, "all_history_center_exact": True,
            "all_previous_raw_and_assist_features_exact": True, "all_raw_labels_exactly_recorded_mu_and_executed_raw": True,
            "all12_permission_and_no_assist_in_2032_actual_physics_ticks": True, "maximum_absolute_errors": errors},
        "learning": {"AUX_updates": 0, "PPO_decisions_added": 0, "PPO_updates_added": 0,
            "checkpoint_written": False, "current_policy_compatibility_or_training_admission_claimed": False,
            "full_task_success_claimed": False}}
    return manifest, arrays, checks


def test_frozen_kernel_hash_and_no_mutating_entry_points():
    assert m.sha(m.FROZEN_KERNEL) == m.FROZEN_KERNEL_SHA256
    assert not hasattr(m, "fit") and not hasattr(m, "execute") and not hasattr(m, "Budget")
    source = inspect.getsource(m)
    assert "kernel.fit(" not in source and "save_semantic_checkpoint(" not in source
    assert 'parser.add_argument("--checkpoint", required=True' in source
    assert 'parser.add_argument("--expected-checkpoint-sha256", required=True' in source
    assert 'parser.add_argument("--expected-policy-decisions", required=True' in source


def test_only_P02_gradient_and_protected_entire_Gaussian(monkeypatch):
    run = runner(); data = states(run.alg.actor)
    actor_hash = m.training.parameter_hash(run.alg.actor)
    rng = m.training.capture_training_rng_state(seed=1001)
    def forbid(*args, **kwargs): raise AssertionError("inspection invoked optimizer/fit")
    monkeypatch.setattr(torch.optim.SGD, "step", forbid)
    monkeypatch.setattr(torch.optim.Adam, "step", forbid)
    monkeypatch.setattr(m.kernel, "fit", forbid)
    result = m.inspect_actor(run.alg.actor, data)
    assert result["P01_gradient_column_exact_zero"]
    assert result["p02_inspection"]["selected_gradient_by_phase_column_l2"][0] == 0
    assert result["p02_inspection"]["selected_gradient_by_phase_column_l2"][1] > 0
    assert all(result["protected_full_Gaussian_algebraic_probe_bitwise_equal"].values())
    assert all(result["same_input_protected_JVP_exact_zero"].values())
    assert result["actual_P03plus_preservation_phase_ids"] == [3, 4, 5, 6]
    assert result["P02_mean_and_log_sigma_may_both_change"]
    assert result["p02_inspection"]["P01_P02_mean_and_sigma_may_both_change"] is False
    assert result["reconstruction_receipt"] == data["receipt"]
    assert result["optimizer_steps_performed"] == result["auxiliary_updates_added"] == result["PPO_updates_added"] == 0
    assert not result["checkpoint_written"] and not result["teacher_deployed"] and not result["physical_success_claimed"]
    assert m.training.parameter_hash(run.alg.actor) == actor_hash
    assert m.training.capture_training_rng_state(seed=1001) == rng
    assert all(p.grad is None for p in run.alg.actor.parameters())


@pytest.mark.parametrize("bad", ["P01_train", "P03_validation", "P02_protection", "P01_missing",
                                  "direct_exact", "admission", "same_rows"])
def test_protection_and_source_scope_rejected(bad):
    run = runner(); data = states(run.alg.actor)
    if bad in ("P01_train", "P03_validation"):
        key, phase = ("train_observations", 0) if bad == "P01_train" else ("validation_observations", 2)
        data[key][0, :13] = 0; data[key][0, phase] = 1
    elif bad == "P02_protection": data["invariance_observations"][0, :13] = 0; data["invariance_observations"][0, 1] = 1
    elif bad == "P01_missing": data["p01_observations"] = torch.zeros(0, 389)
    elif bad == "direct_exact": data["receipt"]["deterministic_source"]["observation_provenance"]["direct389_tensor_was_saved_by_source"] = True
    elif bad == "admission": data["receipt"]["current_MDP_training_admission_claimed"] = True
    elif bad == "same_rows": data["validation_observations"] = data["train_observations"]
    with pytest.raises(ValueError): m.inspect_actor(run.alg.actor, data)


def test_complete_candidate_array_verification():
    manifest, arrays, checks = candidate()
    m.validate_candidate_arrays(manifest, arrays, checks)


@pytest.mark.parametrize("bad", ["truncated", "wrong_phase", "changed_raw", "row_hash", "source_tick",
                                  "large_error", "nan_error", "changed_maximum", "relaxed_tolerance",
                                  "direct_saved", "guessed_field", "fit_claim", "altered_split"])
def test_candidate_bad_reconstruction_or_false_claim_rejected(bad):
    manifest, arrays, checks = candidate()
    if bad == "truncated": checks.pop()
    elif bad == "wrong_phase": arrays["X389_reconstructed"][0, :2] = [1, 0]
    elif bad == "changed_raw": arrays["raw12_actual_deterministic"][0, 0] = .1
    elif bad == "row_hash": checks[0]["reconstructed_float32_observation_sha256"] = "wrong"
    elif bad == "source_tick": arrays["input_tick"][0] += 1
    elif bad == "large_error": checks[0]["errors_max_abs"]["conditional_mean"] = 1.1e-6
    elif bad == "nan_error": checks[0]["errors_max_abs"]["conditional_mean"] = float("nan")
    elif bad == "changed_maximum": manifest["checks"]["maximum_absolute_errors"]["base_mean"] = 1e-7
    elif bad == "relaxed_tolerance": manifest["checks"]["numeric_atol_fixed_before_replay"] = 1e-3
    elif bad == "direct_saved": manifest["observation_provenance"]["direct389_tensor_was_saved_by_source"] = True
    elif bad == "guessed_field": manifest["observation_provenance"]["missing_fields_guessed_or_filled"] = True
    elif bad == "fit_claim": manifest["learning"]["AUX_updates"] = 1
    elif bad == "altered_split": arrays["suggested_train_indices"][0] = 1
    with pytest.raises(ValueError): m.validate_candidate_arrays(manifest, arrays, checks)


def test_actor_copy_inspection_restores_rng_without_loading_Adam(monkeypatch):
    run = runner(); data = states(run.alg.actor)
    metadata = {"seed": 1001, "runner_config": copy.deepcopy(run._semantic_runner_config),
                "policy_contract": m.training._runner_policy_contract(run),
                "actor_parameter_sha256": m.training.parameter_hash(run.alg.actor),
                "optimizer_state_sha256": m.training.state_hash(run.alg.optimizer.state_dict()),
                "global_policy_decisions": 512, "ppo_updates": 4, "optimizer_steps": 80}
    payload = {"infos": copy.deepcopy(metadata), "actor_state_dict": copy.deepcopy(run.alg.actor.state_dict()),
               "optimizer_state_dict": copy.deepcopy(run.alg.optimizer.state_dict())}
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    def forbid(*args, **kwargs): raise AssertionError("inspection loaded/stepped PPO Adam")
    monkeypatch.setattr(torch.optim.Adam, "step", forbid)
    monkeypatch.setattr(torch.optim.Adam, "load_state_dict", forbid)
    rng = m.training.capture_training_rng_state(seed=1001)
    result = m.cpu_inspection(Path("synthetic_readonly_no_file.pt"), metadata, data)
    assert m.training.capture_training_rng_state(seed=1001) == rng
    assert result["source_PPO_counters_unchanged"]["global_policy_decisions"] == 512
    assert result["auxiliary_updates_added"] == 0


def test_actual_data_loader_only_without_inspecting_current_actor(monkeypatch):
    # Root supplied this exact newly saved source and authorized the CPU window.
    # This tests only data/provenance assembly, not its current actor or a fit.
    checkpoint = m.OUT / "checkpoints/history/checkpoint_step_000211968.pt"
    require_sha = "5d0658331204c2bc79d71fd223582637e68dbb986ce4bf57596648f02c77d881"
    metadata = m.semantic_migration.checkpoint_metadata(checkpoint)
    assert metadata["checkpoint_sha256"] == require_sha
    assert metadata["global_policy_decisions"] == 211968
    contract = m.semantic_cli.runtime_contract(
        expected_head="5fd88852bf20c94cd74405c791a13a9fd9e0a3d8", semantic_version="v3",
        experiment_id="p05_hip_only_continuation_v1")
    assert metadata["runtime_contract"] == contract
    def forbid(*args, **kwargs): raise AssertionError("data-only check attempted current actor inspection/optimization")
    monkeypatch.setattr(m, "cpu_inspection", forbid)
    monkeypatch.setattr(m.kernel, "inspect", forbid)
    monkeypatch.setattr(m.kernel, "fit", forbid)
    monkeypatch.setattr(torch.optim.SGD, "step", forbid)
    monkeypatch.setattr(torch.optim.Adam, "step", forbid)
    data = m.load_validated_data(m.DATA_DIR / "candidate_manifest.json", metadata, contract)
    assert data["train_observations"].shape == data["validation_observations"].shape == (85, 389)
    assert data["p01_observations"].shape == (2, 389)
    assert data["invariance_observations"].shape == (13, 389)
    assert data["receipt"]["current_MDP_training_admission_claimed"] is False
    assert data["receipt"]["deterministic_source"]["observation_provenance"]["direct389_tensor_was_saved_by_source"] is False
    assert data["receipt"]["protection_holdouts"]["used_for_training_or_validation_targets"] is False

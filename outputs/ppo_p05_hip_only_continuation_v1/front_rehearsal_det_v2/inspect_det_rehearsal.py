"""Read-only P02 deterministic-source inspection; no fit/execute/save API.

The source demonstrations use field-reconstructed inputs, not directly saved
exact389 vectors. Their reconstruction receipt is retained in full. This tool
never chooses a checkpoint, deploys a teacher, or modifies a policy parameter.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "outputs/ppo_p05_hip_only_continuation_v1"
FROZEN_KERNEL = OUT / "front_rehearsal_v1/front_rehearsal.py"
FROZEN_KERNEL_SHA256 = "d72ff9dd9de8f2e302d2d41d9660a70ba1c880eeb2505b6cd8db26da37895321"
FROZEN_HOLDOUT = OUT / "front_rehearsal_v1/reviewed_data.py"
FROZEN_HOLDOUT_SHA256 = "3ee15f8ce3bb6c1a88ce9204298cfaa8d6b8b4c162349ee0c459b0071ce3c0dd"
DEMONSTRATION_CHECKPOINT_SHA256 = "a4eb243ce07ad4a9ed1cf2f7f9a22e951181bdb6e0716361b8eaeb173acfd5b6"
DATA_DIR = OUT / "det_front_rehearsal_data_v1"
DATA_READER = DATA_DIR / "read_candidate.py"
DATA_READER_SHA256 = "b336419401b5a34354cef070cdc095133343233c6af1df15cddebeea045c490d"
CANDIDATE_MANIFEST_SHA256 = "2cc855dffc6608d3118e5fb38ee76799d63beb4bf775130c6b5339423fa7c8b4"
SCHEMA = "wlr50_clean.deterministic_front_rehearsal_readonly_inspection.v2"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_kernel():
    require(sha(FROZEN_KERNEL) == FROZEN_KERNEL_SHA256, "frozen v1 kernel bytes changed")
    name = "frozen_front_rehearsal_v1_for_det_readonly"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, FROZEN_KERNEL)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


kernel = load_frozen_kernel()
torch = kernel.torch
TensorDict = kernel.TensorDict
training = kernel.training
from wlr50_clean.ppo import semantic_cli, semantic_migration
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_POLICY, P05_CAPTURE_OBSERVATION_LAYOUT


def _tensor(values):
    return kernel.tensors(values, device="cpu")


def _zero_response(response):
    return all(not bool(torch.tensor(response[key]).any()) for key in (
        "raw_mean_per_unit_lr_full12", "log_sigma_per_unit_lr_full12",
        "requested_residual_per_unit_lr_full12"))


def _exact(d1, d2):
    return all(torch.equal(d1[key], d2[key]) for key in ("mean", "sigma", "log_sigma", "request"))


def inspect_actor(actor, data):
    """Same-input P02-only derivative check on an explicitly supplied actor.

    Only the frozen implementation supplies gradients/JVP. A fixed temporary
    P02-column algebraic perturbation is NOT fitted, copied, saved, or sampled.
    It checks protected rows beyond the trivial unchanged live-actor result.
    """
    require(not torch.cuda.is_available(), "read-only inspection requires CUDA_VISIBLE_DEVICES=-1")
    require(data["receipt"]["deterministic_source"]["observation_provenance"]["direct389_tensor_was_saved_by_source"] is False
            and data["receipt"]["current_MDP_training_admission_claimed"] is False,
            "reconstructed data must not be labeled direct-exact389 or admitted to training")
    require(next(actor.parameters()).device.type == "cpu", "inspection actor must be an explicit CPU copy")
    tx, vx, ix, p01 = (_tensor(data[key]) for key in (
        "train_observations", "validation_observations", "invariance_observations", "p01_observations"))
    require(bool((tx["policy"][:, 1] == 1).all()) and bool((vx["policy"][:, 1] == 1).all()),
            "deterministic rehearsal inspection is P02-only")
    require(bool((p01["policy"][:, 0] == 1).all()), "separate real P01 preservation rows required")
    require(bool((ix["policy"][:, :2] == 0).all()), "separate real P03+ preservation rows required")
    original = training.parameter_hash(actor)
    rng = torch.get_rng_state().clone()
    raw = kernel.inspect(actor, tx, data["train_raw_targets"], vx,
                         data["validation_raw_targets"], ix)
    gradient = torch.tensor(raw["selected_gradient_full256x2"], dtype=torch.float32)
    require(tuple(gradient.shape) == (256, 2) and not bool(gradient[:, 0].any()),
            "P02-only objective leaked a gradient into the protected P01 column")
    leaf = kernel.first_layer(actor).weight[:, :2].detach().clone()
    p01_response = kernel._first_gradient_response(actor, p01, leaf, gradient)
    other_response = kernel._first_gradient_response(actor, ix, leaf, gradient)
    require(_zero_response(p01_response) and _zero_response(other_response),
            "P02-only initial-gradient JVP changed protected same-input Gaussian")
    # Explicitly synthetic algebraic perturbation, never an optimization step.
    probe = leaf.clone()
    probe[:, 1] += torch.linspace(-.03125, .03125, 256)
    synthetic = ix["policy"][:1].repeat(12, 1).clone()
    phase_ids = [0, *range(2, 13)]
    synthetic[:, :13] = 0
    synthetic[torch.arange(12), torch.tensor(phase_ids)] = 1
    protected_probes = _tensor(synthetic)
    with torch.no_grad():
        exact = {name: _exact(kernel.distribution(actor, obs), kernel.distribution(actor, obs, probe))
                 for name, obs in (("real_P01", p01), ("real_P03plus", ix),
                                   ("synthetic_P01_P03_to_P13", protected_probes))}
        require(all(exact.values()), "P02-only column algebraic probe changed protected full Gaussian")
        p01_distribution = kernel._summary(kernel.distribution(actor, p01))
    require(training.parameter_hash(actor) == original and torch.equal(torch.get_rng_state(), rng),
            "read-only inspection changed live actor or CPU RNG")
    # The reused generic kernel supports two columns, but this dataset creates
    # an exactly zero P01 derivative. Do not inherit its broader change claim.
    raw["optimized_parameter_scope"] = "none optimized; derivative scope actor.mlp.0.weight[:,1] (256 P02 scalars)"
    raw["P01_P02_mean_and_sigma_may_both_change"] = False
    return {"schema": SCHEMA, "inspection_only": True, "automatic_aux_enabled": False,
            "input_provenance": "field_reconstructed389_with_source_actor_numeric_validation_not_direct_exact389",
            "current_MDP_training_admission_claimed": False,
            "current_actor_errors_are_same_numeric_input_only": True,
            "reconstruction_receipt": deepcopy(data["receipt"]),
            "frozen_kernel": {"path": str(FROZEN_KERNEL), "sha256": FROZEN_KERNEL_SHA256},
            "p02_inspection": raw,
            "permitted_future_derivative_scope_not_executed": "actor.mlp.0.weight[:,1] (256 scalars)",
            "P02_mean_and_log_sigma_may_both_change": True,
            "P01_gradient_column_exact_zero": True,
            "same_input_protected_JVP_exact_zero": {"P01": True, "P03plus": True},
            "protected_full_Gaussian_algebraic_probe_bitwise_equal": exact,
            "algebraic_probe_semantics": "fixed temporary P02-column perturbation; no fit/copy/save; synthetic phase probes are not physical coverage",
            "actual_P01_preservation_row_count": len(p01["policy"]),
            "actual_P03plus_preservation_phase_ids": raw["actual_invariance_phase_ids"],
            "P01_distribution": p01_distribution,
            "P01_negative_initial_gradient_response": p01_response,
            "P03plus_negative_initial_gradient_response": other_response,
            "same_input_invariance_is_not_future_trajectory_invariance": True,
            "PPO_decisions_added": 0, "PPO_updates_added": 0, "PPO_optimizer_steps_added": 0,
            "auxiliary_updates_added": 0, "optimizer_steps_performed": 0,
            "teacher_deployed": False, "physical_success_claimed": False,
            "checkpoint_written": False, "execution_or_budget_interface_available": False}


def observation_runner(observation, metadata):
    class ObservationOnly:
        num_envs, num_actions = 1, 12
        cfg = {"evaluation": True, "semantic_version": "v3"}
        def get_observations(self):
            x = torch.as_tensor(observation, dtype=torch.float32, device="cpu").reshape(1, 389)
            return TensorDict({"policy": x, "critic": x.clone()}, batch_size=[1], device="cpu")
    runner, _ = training.construct_semantic_runner(ObservationOnly(), seed=metadata["seed"], device="cpu",
        initialize_actor=False, policy_version=P05_CAPTURE_POLICY, observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)
    expected = deepcopy(metadata["runner_config"])
    expected["device"] = "cpu"
    require(runner._semantic_runner_config == expected, "inspection runner differs beyond explicit CPU-copy device")
    require(training._runner_policy_contract(runner) == metadata["policy_contract"], "current389 policy contract mismatch")
    return runner


def cpu_inspection(checkpoint, metadata, data):
    """Actor-copy inspection, not an official relocated training resume."""
    require(not torch.cuda.is_available(), "CPU inspection requires CUDA_VISIBLE_DEVICES=-1")
    original_rng = training.capture_training_rng_state(seed=metadata["seed"])
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        expected = {k: v for k, v in metadata.items() if k not in (
            "checkpoint_path", "checkpoint_sha256", "save_load_round_trip")}
        require(payload["infos"] == expected
                and training.state_hash(payload["optimizer_state_dict"]) == metadata["optimizer_state_sha256"],
                "current source checkpoint metadata or Adam integrity mismatch")
        runner = observation_runner(data["train_observations"][0], metadata)
        runner.alg.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        require(training.parameter_hash(runner.alg.actor) == metadata["actor_parameter_sha256"],
                "explicit current source actor hash mismatch")
        runner.alg.actor.eval()
        result = inspect_actor(runner.alg.actor, data)
        return {**result, "load_semantics": "CPU actor-state copy only; PPO optimizer not loaded or stepped",
                "source_actor_parameter_sha256": metadata["actor_parameter_sha256"],
                "source_PPO_counters_unchanged": {k: metadata[k] for k in (
                    "global_policy_decisions", "ppo_updates", "optimizer_steps")}}
    finally:
        training.restore_training_rng_state(original_rng, expected_seed=metadata["seed"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-policy-decisions", required=True, type=int)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    require(not torch.cuda.is_available(), "inspection requires CUDA_VISIBLE_DEVICES=-1")
    checkpoint = args.checkpoint.resolve(strict=True)
    metadata = semantic_migration.checkpoint_metadata(checkpoint)
    require(metadata["checkpoint_sha256"] == args.expected_checkpoint_sha256
            and metadata["global_policy_decisions"] == args.expected_policy_decisions,
            "explicit current checkpoint SHA/counter differs; no old-source fallback")
    contract = semantic_cli.runtime_contract(expected_head=args.expected_head, semantic_version="v3",
                                            experiment_id="p05_hip_only_continuation_v1")
    require(metadata["runtime_contract"] == contract and metadata["policy_contract"]["version"] == P05_CAPTURE_POLICY,
            "explicit current checkpoint/runtime/policy mismatch")
    data = load_validated_data(args.data, metadata, contract)
    report_path = args.report.resolve()
    require(report_path.is_relative_to(HERE.resolve()) and not report_path.exists(),
            "inspection report requires a new path in the independent det_v2 output directory")
    result = cpu_inspection(checkpoint, metadata, data)
    result["binding"] = {"source_checkpoint": str(checkpoint), "source_sha256": metadata["checkpoint_sha256"],
                         "source_manifest_sha256": sha(checkpoint.with_name(checkpoint.stem + "_manifest.json")),
                         "inspection_helper_sha256": sha(__file__), "data": str(args.data.resolve()),
                         "data_sha256": sha(args.data)}
    training.write_json(report_path, result)
    print(json.dumps({"report": str(report_path), "mode": "inspection_only", "optimizer_steps_performed": 0}))
    return result


def validate_candidate_arrays(manifest, arrays, checks):
    """Independent byte/row checks, not a new 389 reconstruction or admission."""
    import numpy as np
    require(manifest["schema"] == "wlr50_clean.det_front_reconstruction_candidate.v1"
            and manifest["status"] == "CANDIDATE_ONLY_NOT_ADMITTED_TO_LEARNING", "wrong reconstruction candidate schema/status")
    provenance = manifest["observation_provenance"]
    require(provenance["direct389_tensor_was_saved_by_source"] is False
            and provenance["reconstructed_from_explicit_synchronized_fields"] is True
            and provenance["missing_fields_guessed_or_filled"] is False
            and provenance["original_CUDA_vs_CPU_is_not_bitwise_proof"] is True
            and provenance["matching12_outputs_alone_cannot_prove389_input_identity"] is True,
            "missing or overstated reconstruction provenance")
    require(manifest["dataset"]["rows"] == 254 and manifest["dataset"]["phase"] == "P02"
            and manifest["checks"]["numeric_atol_fixed_before_replay"] == 1e-6,
            "reviewed full P02 window or fixed reconstruction tolerance changed")
    for flag in ("all_history_center_exact", "all_previous_raw_and_assist_features_exact",
                 "all_raw_labels_exactly_recorded_mu_and_executed_raw", "all12_permission_and_no_assist_in_2032_actual_physics_ticks"):
        require(manifest["checks"][flag] is True, "source row verification missing: " + flag)
    require(manifest["learning"]["AUX_updates"] == manifest["learning"]["PPO_decisions_added"]
            == manifest["learning"]["PPO_updates_added"] == 0
            and manifest["learning"]["checkpoint_written"] is False
            and manifest["learning"]["current_policy_compatibility_or_training_admission_claimed"] is False
            and manifest["learning"]["full_task_success_claimed"] is False, "candidate cannot contain learning/success claims")
    shapes = {"X389_reconstructed": (254, 389), "raw12_actual_deterministic": (254, 12),
              "source_conditional_mean12": (254, 12), "source_effective_sigma12": (254, 12),
              "source_history_center12": (254, 12)}
    for key, shape in shapes.items():
        value = arrays[key]
        require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(),
                "candidate array must preserve finite float32 source shape: " + key)
        require(manifest["dataset"]["shapes"][key] == list(shape), "manifest shape differs: " + key)
    x, target = arrays["X389_reconstructed"], arrays["raw12_actual_deterministic"]
    require(np.array_equal(target, arrays["source_conditional_mean12"]), "target is not the recorded executed deterministic raw mean")
    require((arrays["source_effective_sigma12"] > 0).all() and (x[:, 1] == 1).all()
            and (x[:, :13].sum(-1) == 1).all(), "candidate sigma or P02 phase invalid")
    require(np.array_equal(arrays["source_decision"], np.arange(3, 257))
            and np.array_equal(arrays["input_tick"], np.arange(16, 2041, 8)), "source P02 temporal indices changed")
    for key, expected in (("suggested_train_indices", np.arange(0, 254, 3)),
                          ("suggested_validation_indices", np.arange(1, 254, 3)),
                          ("source_only_indices", np.arange(2, 254, 3))):
        require(np.array_equal(arrays[key], expected), "fixed suggested split changed: " + key)
    require(len(checks) == 254, "all254 reconstruction row checks are required")
    error_keys = {"conditional_mean", "base_mean", "history_center", "learned_sigma", "effective_sigma",
                  "effective_log_std", "effective_innovation_sigma_multiplier"}
    for i, row in enumerate(checks):
        require(row["candidate_index"] == i and row["decision"] == i + 3
                and row["start_tick"] == 16 + i * 8 and row["end_tick"] == 24 + i * 8,
                "reconstruction row/check temporal mismatch")
        require(hashlib.sha256(x[i].tobytes()).hexdigest() == row["reconstructed_float32_observation_sha256"],
                "reconstructed observation bytes differ from validated source row")
        require(row["history_and_assist_exact"] is True and row["all12_permission_no_assist_each_physics_tick"] is True,
                "row lost history/assist/permission verification")
        errors = row["errors_max_abs"]
        require(set(errors) == error_keys and all(type(v) in (float, int) and math.isfinite(v)
                and 0 <= v <= 1e-6 for v in errors.values()) and errors["history_center"] == 0,
                "reconstruction row has nonfinite/excess errors or nonexact history")
    require(manifest["checks"]["maximum_absolute_errors"] == {
        key: max(row["errors_max_abs"][key] for row in checks) for key in error_keys},
        "aggregate reconstruction errors differ from all row checks")


def load_protected_holdouts(metadata, contract):
    require(sha(FROZEN_HOLDOUT) == FROZEN_HOLDOUT_SHA256, "frozen direct-observation holdout loader changed")
    name = "frozen_block03_holdouts_for_det_v2"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, FROZEN_HOLDOUT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    legacy = sys.modules[name].load_reviewed_data(metadata, contract)
    front = legacy["train_observations"]
    p01 = front[front[:, 0] == 1].clone()
    ix = legacy["invariance_observations"].clone()
    require(p01.shape == (2, 389) and ix.shape == (13, 389), "reviewed protection-only selection changed")
    receipt = {"source_kind": "separate_block03_directly_saved389_protection_only_not_positive_targets",
               "P01_source_indices": [0, 1], "P03_P06_source_indices": legacy["receipt"]["groups"]["invariance"]["indices"],
               "P01_rows": 2, "P03_P06_rows": 13, "used_for_training_or_validation_targets": False,
               "real_P07_P13_coverage_claimed": False, "loader_sha256": FROZEN_HOLDOUT_SHA256,
               "P01_float32_sha256": sys.modules[name].tensor_sha(p01),
               "P03_P06_float32_sha256": sys.modules[name].tensor_sha(ix),
               "full_source_validation_receipt": legacy["receipt"]}
    return p01, ix, receipt


def load_validated_data(path, metadata, contract):
    import numpy as np
    path = Path(path).resolve(strict=True)
    require(path == (DATA_DIR / "candidate_manifest.json").resolve()
            and sha(path) == CANDIDATE_MANIFEST_SHA256,
            "--data must name the reviewed in-namespace candidate manifest; no candidate substitution")
    require(sha(DATA_READER) == DATA_READER_SHA256, "independent read-only reconstruction validator changed")
    name = "frozen_det_source_candidate_reader_for_det_v2"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, DATA_READER)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    arrays, manifest = sys.modules[name].load_candidate(path)
    source = manifest["source_checkpoint"]
    require(source["sha256"] == DEMONSTRATION_CHECKPOINT_SHA256 and sha(source["path"]) == source["sha256"],
            "deterministic demonstration checkpoint is not the reviewed CP201728")
    old_meta = semantic_migration.checkpoint_metadata(Path(source["path"]))
    require(old_meta["actor_parameter_sha256"] == source["actor_sha256"]
            and sha(Path(source["path"]).with_name(Path(source["path"]).stem + "_manifest.json")) == source["manifest_sha256"],
            "demonstration actor/manifest binding changed")
    checks = [json.loads(line) for line in Path(manifest["checks"]["path"]).read_text(encoding="utf-8").splitlines()]
    validate_candidate_arrays(manifest, arrays, checks)
    counts = {"video_policy_decisions.jsonl": 409, "physical_observations.jsonl": 2049,
              "capture_assist_ticks.jsonl": 2048, "native_tick_audit.jsonl": 2048}
    require(len(manifest["source_prefixes"]) == 4, "all four bounded source prefixes required")
    seen = set()
    for receipt in manifest["source_prefixes"]:
        source_path = Path(receipt["path"])
        require(source_path.name in counts and source_path.name not in seen
                and receipt["rows_read"] == counts[source_path.name] and receipt["whole_file_rehashed"] is False,
                "source prefix scope changed")
        seen.add(source_path.name)
        digest = hashlib.sha256()
        with source_path.open("rb") as stream:
            for index in range(receipt["rows_read"]):
                line = next(stream)
                digest.update(line)
                if source_path.name == "video_policy_decisions.jsonl" and 2 <= index <= 255:
                    row = json.loads(line); i = index - 2; request = row["policy_request"]
                    require(hashlib.sha256(line).hexdigest() == checks[i]["source_decision_line_sha256"]
                            and row["request_phase"] == "P02" and request["mode"] == "deterministic_conditional_mean"
                            and request["sampling_draws"] == request["extra_random_draws"] == 0,
                            "source deterministic decision provenance changed")
                    for key, value in (("raw12_actual_deterministic", row["raw_policy_action_full12"]),
                                       ("source_conditional_mean12", request["conditional_mean_full12"]),
                                       ("source_effective_sigma12", request["effective_sigma_full12"]),
                                       ("source_history_center12", request["history_center_full12"])):
                        require(np.array_equal(arrays[key][i], np.asarray(value, dtype=np.float32)),
                                "candidate differs from recorded actual source: " + key)
        require(digest.hexdigest() == receipt["prefix_bytes_sha256"], "bounded source-prefix bytes changed")
    p01, ix, protection_receipt = load_protected_holdouts(metadata, contract)
    train, valid = arrays["suggested_train_indices"], arrays["suggested_validation_indices"]
    return {"train_observations": arrays["X389_reconstructed"][train],
            "train_raw_targets": arrays["raw12_actual_deterministic"][train],
            "validation_observations": arrays["X389_reconstructed"][valid],
            "validation_raw_targets": arrays["raw12_actual_deterministic"][valid],
            "p01_observations": p01, "invariance_observations": ix,
            "receipt": {"deterministic_source": manifest, "protection_holdouts": protection_receipt,
                        "candidate_manifest_sha256": sha(path), "current_MDP_training_admission_claimed": False,
                        "candidate_validator_sha256": DATA_READER_SHA256,
                        "numeric_input_comparison_only_not_direct_exact389": True,
                        "train_indices": train.tolist(), "validation_indices": valid.tolist(),
                        "source_only_indices": arrays["source_only_indices"].tolist()}}


if __name__ == "__main__":
    main()

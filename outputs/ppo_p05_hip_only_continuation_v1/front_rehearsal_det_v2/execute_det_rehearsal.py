"""Explicitly admitted, finite deterministic P02 AUX; never automatic.

This is a wrapper around the immutable v1 finite SGD kernel and official
checkpoint save/reload. It adds no optimizer rule, automatic budget or retry.
The historical deterministic labels were actually executed raw actions equal
to the recorded conditional means; reconstructed inputs are not direct389.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import inspect_det_rehearsal as inspection

HERE, OUT, ROOT = inspection.HERE, inspection.OUT, inspection.ROOT
kernel, training, torch = inspection.kernel, inspection.training, inspection.torch
require, sha = inspection.require, inspection.sha
READONLY_SHA256 = "f688c3a7c4424fb7aba500f99e8a362aed8c91b57cabbd72fdb3d475b5a42d80"
SCHEMA = "wlr50_clean.finite_reconstructed_deterministic_P02_aux.v2"
LEDGER_SCHEMA = "wlr50_clean.front_rehearsal_auxiliary.v1"
LEDGER_KEY = "front_rehearsal_auxiliary"
PARAMETERS = ["actor.mlp.0.weight[:,1]"]
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
ADMISSION_PATH = OUT / "det_P02_current_semantics_admission.json"
ADMISSION_SHA256 = "08a44852a26b98aaddaded707cb6f52ebd76f25e14cee69ce510e851f8c7915b"
ADMISSION_AUDIT_PATH = OUT / "audit_det_P02_current_semantics.py"
ADMISSION_AUDIT_SHA256 = "c3f28334e31cca93d24ec8aff68ff39384ef033591226a50fd42021d08d463c1"
ADMISSION_SCHEMA = "wlr50_clean.det_front_current_semantics_admission.v1"
ADMISSION_RESULT = "PASS_BOUNDED_P02_HISTORICAL_SUPERVISED_DATA_SEMANTICS_ONLY"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def json_value(value):
    return json.loads(json.dumps(value, allow_nan=False))


def helper_hashes():
    require(sha(HERE / "inspect_det_rehearsal.py") == READONLY_SHA256, "bound readonly inspector changed")
    require(sha(inspection.FROZEN_KERNEL) == inspection.FROZEN_KERNEL_SHA256, "immutable v1 kernel changed")
    require(sha(inspection.FROZEN_HOLDOUT) == inspection.FROZEN_HOLDOUT_SHA256, "immutable direct holdout loader changed")
    require(sha(inspection.DATA_READER) == inspection.DATA_READER_SHA256, "immutable reconstruction reader changed")
    require(sha(ADMISSION_AUDIT_PATH) == ADMISSION_AUDIT_SHA256, "reviewed current-semantics audit code changed")
    return {"execute_det_rehearsal.py": sha(__file__), "inspect_det_rehearsal.py": READONLY_SHA256,
            "front_rehearsal_v1/front_rehearsal.py": inspection.FROZEN_KERNEL_SHA256,
            "front_rehearsal_v1/reviewed_data.py": inspection.FROZEN_HOLDOUT_SHA256,
            "det_front_rehearsal_data_v1/read_candidate.py": inspection.DATA_READER_SHA256,
            "audit_det_P02_current_semantics.py": ADMISSION_AUDIT_SHA256}


def source_identity(checkpoint, metadata):
    checkpoint = Path(checkpoint).resolve()
    return {"path": str(checkpoint), "sha256": metadata["checkpoint_sha256"],
            "manifest_sha256": sha(checkpoint.with_name(checkpoint.stem + "_manifest.json")),
            "PPO_counters": {key: metadata[key] for key in COUNTERS},
            "runtime_contract_sha256": digest(metadata["runtime_contract"])}


def verify_inspection(path, source, data):
    result = read(path)
    bind = result["binding"]
    require(result["schema"] == inspection.SCHEMA and result["inspection_only"] is True
            and result["auxiliary_updates_added"] == result["optimizer_steps_performed"] == 0,
            "execution requires a prior genuine readonly inspection")
    require(bind["source_checkpoint"] == source["path"] and bind["source_sha256"] == source["sha256"]
            and bind["source_manifest_sha256"] == source["manifest_sha256"]
            and result["source_PPO_counters_unchanged"] == source["PPO_counters"]
            and bind["inspection_helper_sha256"] == READONLY_SHA256
            and bind["data_sha256"] == data["receipt"]["candidate_manifest_sha256"],
            "stale or mismatched source/data/inspection binding")
    require(result["reconstruction_receipt"] == data["receipt"]
            and result["current_MDP_training_admission_claimed"] is False
            and result["P01_gradient_column_exact_zero"] is True
            and all(result["protected_full_Gaussian_algebraic_probe_bitwise_equal"].values()),
            "readonly proof or honest reconstruction receipt differs")
    return {"path": str(Path(path).resolve()), "sha256": sha(path), "schema": result["schema"]}


def budget_from_file(path, expected_binding):
    envelope = read(path)
    require(envelope.get("schema") == "wlr50_clean.explicit_det_P02_aux_budget.v2"
            and envelope.get("binding") == expected_binding,
            "explicit budget must bind this source, data, inspection, admission and helper set")
    require(set(envelope) == {"schema", "binding", "budget"}, "unreviewed automatic budget options")
    budget = kernel.Budget(**envelope["budget"])
    budget.validate()
    return budget, {"path": str(Path(path).resolve()), "sha256": sha(path), "envelope": envelope}


def _tensor(values, device):
    return kernel.tensors(values, device=device)


def _exact(before, after):
    return all(torch.equal(before[key], after[key]) for key in ("mean", "sigma", "log_sigma", "request"))


def _fit_p02(runner, data, *, budget, authorized=False):
    """Private scope guard; public CLI must first validate external admission."""
    require(authorized is True, "no admitted/authorized P02 AUX execution")
    actor = runner.alg.actor
    device = next(actor.parameters()).device
    tx, vx, ix, p01 = (_tensor(data[key], device) for key in (
        "train_observations", "validation_observations", "invariance_observations", "p01_observations"))
    require(bool((tx["policy"][:, 1] == 1).all()) and bool((vx["policy"][:, 1] == 1).all())
            and bool((p01["policy"][:, 0] == 1).all()) and bool((ix["policy"][:, :2] == 0).all()),
            "P02-only positives and separate protected P01/P03+ rows required")
    saved = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    leaf = kernel.first_layer(actor).weight[:, :2].detach().clone().requires_grad_(True)
    td = kernel.distribution(actor, tx, leaf)
    target = torch.as_tensor(data["train_raw_targets"], dtype=leaf.dtype, device=device)
    gradient, = torch.autograd.grad(.5 * (td["mean"] - target).square().mean(), (leaf,))
    require(not bool(gradient[:, 0].any()), "P02-only objective generated a P01-column gradient")
    synthetic = ix["policy"][:1].repeat(12, 1).clone()
    synthetic[:, :13] = 0
    synthetic[torch.arange(12, device=device), torch.tensor([0, *range(2, 13)], device=device)] = 1
    probes = _tensor(synthetic, device)
    with torch.no_grad():
        protected = {name: {k: v.detach().clone() for k, v in kernel.distribution(actor, obs).items()}
                     for name, obs in (("real_P01", p01), ("real_P03plus", ix), ("synthetic_P01_P03_P13", probes))}
    try:
        raw_report = kernel.fit(runner, tx, target, vx, data["validation_raw_targets"], ix,
                                budget=budget, authorized=True)
        for key, value in actor.state_dict().items():
            if key == "mlp.0.weight":
                require(torch.equal(value[:, 0], saved[key][:, 0]) and torch.equal(value[:, 2:], saved[key][:, 2:]),
                        "AUX changed a protected first-layer column")
            else:
                require(torch.equal(value, saved[key]), "AUX changed an unselected actor tensor")
        with torch.no_grad():
            proof = {name: _exact(protected[name], kernel.distribution(actor, obs)) for name, obs in
                     (("real_P01", p01), ("real_P03plus", ix), ("synthetic_P01_P03_P13", probes))}
        require(all(proof.values()), "AUX changed a protected same-input entire Gaussian")
        changed = int(torch.count_nonzero(actor.mlp[0].weight[:, 1] - saved["mlp.0.weight"][:, 1]))
        require(0 <= changed <= 256 and (raw_report["accepted_auxiliary_updates"] == 0 or changed > 0),
                "accepted P02 AUX requires a real whitelisted parameter change")
        return {"schema": SCHEMA, "accepted_auxiliary_updates": raw_report["accepted_auxiliary_updates"],
                "attempted_auxiliary_optimizer_steps": raw_report["attempted_auxiliary_optimizer_steps"],
                "stop_reason": raw_report["stop_reason"], "optimized_parameters": PARAMETERS.copy(),
                "optimized_scalar_count": 256, "actually_changed_scalar_count": changed,
                "kernel_temporary_leaf_scope": "512 scalars, P01 column remains exactly unchanged; only P02 column256 may change",
                "P01_initial_gradient_exact_zero": True, "protected_entire_Gaussian_bitwise_equal": proof,
                "P02_mean_and_log_sigma_may_both_change": True, "MDP_or_control_changed": False,
                "kernel_or_sigma_rule_changed": False, "training_rng_preserved": True,
                "fresh_PPO_rollout_required": True, "teacher_deployed": False, "physical_success_claimed": False,
                "PPO_decisions_added": 0, "PPO_updates_added": 0, "PPO_optimizer_steps_added": 0,
                "actor_parameter_sha256_before": raw_report["actor_parameter_sha256_before"],
                "actor_parameter_sha256_after": raw_report["actor_parameter_sha256_after"],
                "source_observations_directly_saved389": False,
                "target_semantics": "actual_executed_deterministic_raw_equal_to_recorded_mu_not_unexecuted_mean_or_final_target",
                "same_input_invariance_is_not_future_trajectory_invariance": True,
                "nested_kernel_scope_is_capability_not_this_execution_scope": True,
                "frozen_kernel_fit_report": raw_report}
    except BaseException:
        actor.load_state_dict(saved, strict=True)
        raise


def append_event(infos, *, report, binding, data_receipt, budget_receipt):
    require(report["schema"] == SCHEMA and report["optimized_parameters"] == PARAMETERS
            and report["optimized_scalar_count"] == 256 and report["actually_changed_scalar_count"] > 0,
            "invalid P02-only fit scope")
    accepted, attempts = report["accepted_auxiliary_updates"], report["attempted_auxiliary_optimizer_steps"]
    require(type(accepted) is int and type(attempts) is int and 0 < accepted <= attempts <= 32,
            "no real accepted finite AUX updates to publish")
    require(all(report[key] == 0 for key in ("PPO_decisions_added", "PPO_updates_added", "PPO_optimizer_steps_added"))
            and report["teacher_deployed"] is False and report["physical_success_claimed"] is False
            and report["source_observations_directly_saved389"] is False, "false fit credit/provenance")
    result = deepcopy(infos)
    branch = result["rr_postcross_workspace_branch"]
    require(branch["schema"] == "wlr50_clean.rr_postcross_workspace_same389.v1"
            and branch["semantics"] == "current_qualified_RR_over_top_receiver_retirement_v1"
            and branch["migration_added_updates"] == 0, "wrong current RR branch lineage")
    ledger = branch[LEDGER_KEY]
    require(ledger["schema"] == LEDGER_SCHEMA and len(ledger["events"]) >= 1
            and [event["event_index"] for event in ledger["events"]] == list(range(1, len(ledger["events"]) + 1)),
            "original ordered front AUX lineage missing")
    old_events = deepcopy(ledger["events"])
    event = {"event_index": len(old_events) + 1,
        "kind": "finite_supervised_reconstructed_deterministic_P02_raw_actions_not_PPO",
        "source_checkpoint": deepcopy(binding["source_checkpoint"]), "helper_sha256": deepcopy(binding["helpers"]),
        "binding": deepcopy(binding), "data_receipt": deepcopy(data_receipt),
        "fit_report": deepcopy(report), "fit_report_sha256": digest(report),
        "budget": deepcopy(budget_receipt["envelope"]["budget"]), "budget_receipt": deepcopy(budget_receipt),
        "optimized_parameters": PARAMETERS.copy(), "phase_scope": ["P02"], "mean_and_log_sigma_may_change": True,
        "source_observations_directly_saved389": False,
        "target_semantics": report["target_semantics"], "same_input_P01_P03plus_preserved": True,
        "physical_success_claimed": False, "teacher_deployed": False,
        "PPO_counters_unchanged": {key: infos[key] for key in COUNTERS},
        "stage_requested_decisions_unchanged": deepcopy(infos["stage_requested_decisions"]),
        "PPO_decisions_added": 0, "PPO_updates_added": 0, "PPO_optimizer_steps_added": 0}
    ledger["events"].append(json_value(event))
    ledger["accepted_auxiliary_updates_total"] = sum(e["fit_report"]["accepted_auxiliary_updates"] for e in ledger["events"])
    ledger["attempted_auxiliary_optimizer_steps_total"] = sum(e["fit_report"]["attempted_auxiliary_optimizer_steps"] for e in ledger["events"])
    require(ledger["events"][:-1] == old_events, "historical front AUX events changed")
    restored = deepcopy(result); restored["rr_postcross_workspace_branch"] = deepcopy(infos["rr_postcross_workspace_branch"])
    require(restored == infos, "historical metadata outside current ledger changed")
    previous_branch = deepcopy(branch); previous_branch[LEDGER_KEY] = deepcopy(infos["rr_postcross_workspace_branch"][LEDGER_KEY])
    require(previous_branch == infos["rr_postcross_workspace_branch"], "original RR branch keys changed")
    return result


def observation_runner(observation, metadata, *, device):
    class ObservationOnly:
        num_envs, num_actions = 1, 12
        cfg = {"evaluation": True, "semantic_version": "v3"}
        def get_observations(self):
            x = torch.as_tensor(observation, dtype=torch.float32, device=device).reshape(1, 389)
            return inspection.TensorDict({"policy": x, "critic": x.clone()}, batch_size=[1], device=device)
    runner, _ = training.construct_semantic_runner(ObservationOnly(), seed=metadata["seed"], device=device,
        initialize_actor=False, policy_version=inspection.P05_CAPTURE_POLICY,
        observation_layout=inspection.P05_CAPTURE_OBSERVATION_LAYOUT)
    require(runner._semantic_runner_config == metadata["runner_config"]
            and training._runner_policy_contract(runner) == metadata["policy_contract"],
            "execution must retain exact source-device runner/policy config")
    return runner


def _new_path(path, *, checkpoint=False):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT.resolve()) and not path.exists(), "use a new output path in the experiment namespace")
    if checkpoint:
        require(path.suffix == ".pt" and path.name.startswith("checkpoint_aux_detP02_")
                and not path.with_name(path.stem + "_manifest.json").exists(), "use a unique explicit detP02 AUX checkpoint")
    return path


def _protected_saved_state(runner, infos):
    for key, actual in (("critic_parameter_sha256", training.parameter_hash(runner.alg.critic)),
                        ("optimizer_state_sha256", training.state_hash(runner.alg.optimizer.state_dict())),
                        ("normalizer_state_sha256", training.state_hash(training._normalizers(runner)))):
        require(actual == infos[key], "protected source training state changed: " + key)
    require(runner._semantic_runner_config == infos["runner_config"]
            and training.optimizer_learning_rate(runner) == infos["optimizer_learning_rate"]
            and training.capture_training_rng_state(seed=infos["seed"]) == infos["training_rng_state"],
            "source device/config/LR/full CPU-CUDA RNG changed")


def save_candidate(runner, path, infos, *, data, report, binding, budget_receipt):
    path = _new_path(path, checkpoint=True)
    require(runner.alg.storage.step == 0 and runner.alg.transition.actions is None, "save requires fresh empty rollout")
    require(infos["actor_parameter_sha256"] == report["actor_parameter_sha256_before"]
            and training.parameter_hash(runner.alg.actor) == report["actor_parameter_sha256_after"], "actor report binding differs")
    _protected_saved_state(runner, infos)
    payload = append_event(infos, report=report, binding=binding, data_receipt=data["receipt"], budget_receipt=budget_receipt)
    payload["stage"] = "auxiliary_reconstructed_deterministic_P02_not_PPO"
    checkpoint, manifest = training.save_semantic_checkpoint(runner, path, payload)
    metadata = inspection.semantic_migration.checkpoint_metadata(checkpoint)
    fresh = observation_runner(data["train_observations"][0], metadata, device=metadata["runner_config"]["device"])
    loaded = training.load_semantic_checkpoint(fresh, checkpoint, contract=metadata["runtime_contract"], seed=metadata["seed"])
    require(all(loaded[key] == infos[key] for key in COUNTERS)
            and fresh.alg.storage.step == 0 and fresh.alg.transition.actions is None,
            "independent reload changed PPO counters/fresh storage")
    require(loaded["rr_postcross_workspace_branch"] == payload["rr_postcross_workspace_branch"], "independent AUX ledger reload differs")
    for key in infos:
        if key.endswith(("_branch", "_migration")) and key != "rr_postcross_workspace_branch":
            require(loaded[key] == infos[key], "historical branch/migration changed: " + key)
    require(training.parameter_hash(fresh.alg.actor) == report["actor_parameter_sha256_after"], "independent actor reload differs")
    _protected_saved_state(fresh, infos)
    return {"path": str(checkpoint), "manifest": str(manifest), "sha256": sha(checkpoint),
            "independent_official_reload_verified": True, "source_device_preserved": metadata["runner_config"]["device"],
            "latest_pointer_published": False, "normal_PPO_save_carry_after_this_real_AUX_not_yet_verified": True}


def validated_admission(path, metadata, contract, inspection_path, data_path):
    path = Path(path).resolve(strict=True)
    require(path == ADMISSION_PATH.resolve() and sha(path) == ADMISSION_SHA256
            and sha(ADMISSION_AUDIT_PATH) == ADMISSION_AUDIT_SHA256,
            "execution needs the independently reviewed immutable admission proof")
    receipt = read(path)
    candidate = read(data_path)
    require(receipt["schema"] == ADMISSION_SCHEMA and receipt["result"] == ADMISSION_RESULT
            and receipt["rows_checked"] == 254, "no explicit bounded P02 data-semantics admission")
    require(receipt["current_checkpoint"] == metadata["checkpoint_path"]
            and receipt["current_checkpoint_sha256"] == metadata["checkpoint_sha256"]
            and receipt["current_runtime"] == contract["source_git_commit"]
            and receipt["candidate_manifest_sha256"] == sha(data_path)
            and receipt["candidate_npz_sha256"] == candidate["dataset"]["sha256"]
            and receipt["source_checkpoint_sha256"] == candidate["source_checkpoint"]["sha256"],
            "admission applies to different current checkpoint/runtime or reconstructed dataset")
    for key in ("both_immutable_plans_and_pure_strict_factors_validated", "policy_kernel_action_caps_sigma_contract_equal",
                "normalization_semantics_equal", "capture_numeric_encoder_AST_equal", "physical_runtime_profile_equal",
                "all_float32_X17_exact", "no_data_or_existing_helper_mutation"):
        require(receipt[key] is True, "missing exact current-semantics proof: " + key)
    require(receipt["maximum_source_phi_abs_error"] == receipt["maximum_old_current_phi_abs_difference"]
            == receipt["RR_retirement_active_inputs"] == receipt["capture_HOLD_to_AIR_eligible_inputs"]
            == receipt["AUX_updates"] == receipt["PPO_updates"] == 0,
            "unadmitted source/current semantic difference or invented learning credit")
    require(receipt["migration_plan_sha256"]["RR_potential"] == metadata["rr_postcross_workspace_migration"]["plan_sha256"],
            "admission RR migration proof differs from current immutable lineage")
    rows = receipt["row_checks"]
    require(len(rows) == 254 and all(row["candidate_index"] == i and row["source_decision"] == i + 3
            and row["input_tick"] == 16 + 8 * i and row["X17_float32_exact"] is True
            and row["source_phi_abs_error"] == row["old_current_phi_abs_difference"] == 0
            and row["feedback_hold_to_air_change_can_trigger"] is False
            and row["RR_retirement_can_trigger"] is False for i, row in enumerate(rows)),
            "admission does not cover every unchanged P02 input")
    return {"path": str(path), "sha256": ADMISSION_SHA256, "schema": ADMISSION_SCHEMA,
            "result": ADMISSION_RESULT, "rows_checked": 254,
            "current_checkpoint_sha256": receipt["current_checkpoint_sha256"],
            "candidate_manifest_sha256": receipt["candidate_manifest_sha256"],
            "candidate_npz_sha256": receipt["candidate_npz_sha256"],
            "audit_script": {"path": str(ADMISSION_AUDIT_PATH), "sha256": ADMISSION_AUDIT_SHA256},
            "inspection_receipt": {"path": str(Path(inspection_path).resolve()), "sha256": sha(inspection_path)},
            "restrictions": deepcopy(receipt["restrictions"]),
            "admission_itself_authorizes_optimization": False}


def prepare_binding(checkpoint, metadata, contract, *, inspection_path, admission_path, data_path, expected_head):
    """Read-only binding assembly; no budget values, optimizer or checkpoint write."""
    require(metadata["runtime_contract"] == contract and contract["source_git_commit"] == expected_head,
            "source/runtime/head binding mismatch")
    data = inspection.load_validated_data(data_path, metadata, contract)
    source = source_identity(checkpoint, metadata)
    reviewed_inspection = verify_inspection(inspection_path, source, data)
    admission = validated_admission(admission_path, metadata, contract, inspection_path, data_path)
    binding = {"source_checkpoint": source, "helpers": helper_hashes(), "inspection": reviewed_inspection,
               "data_receipt_sha256": digest(data["receipt"]), "data_manifest_sha256": sha(data_path),
               "admission": admission, "expected_head": expected_head}
    return binding, data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "inspection-receipt", "admission", "data", "budget", "aux-checkpoint", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-policy-decisions", required=True, type=int)
    parser.add_argument("--execute-aux", action="store_true")
    args = parser.parse_args(argv)
    require(args.execute_aux is True, "explicit --execute-aux authorization required; no automatic fit")
    report_path, target = _new_path(args.report), _new_path(args.aux_checkpoint, checkpoint=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    metadata = inspection.semantic_migration.checkpoint_metadata(checkpoint)
    require(metadata["checkpoint_sha256"] == args.expected_checkpoint_sha256
            and metadata["global_policy_decisions"] == args.expected_policy_decisions, "explicit source SHA/counter mismatch")
    contract = inspection.semantic_cli.runtime_contract(expected_head=args.expected_head, semantic_version="v3",
                                                       experiment_id="p05_hip_only_continuation_v1")
    require(metadata["runtime_contract"] == contract, "source/current runtime contract mismatch")
    binding, data = prepare_binding(checkpoint, metadata, contract, inspection_path=args.inspection_receipt,
                                   admission_path=args.admission, data_path=args.data, expected_head=args.expected_head)
    budget, budget_receipt = budget_from_file(args.budget, binding)
    device = metadata["runner_config"]["device"]
    if str(device).startswith("cuda"):
        require(torch.cuda.is_available(), "actual source CUDA visibility required; CPU relocation forbidden")
    aliases = [OUT / "checkpoints" / name for name in (
        "checkpoint_last_pointer.json", "resume_state.json", "checkpoint_last.pt", "checkpoint_last_manifest.json")]
    alias_before = {str(path): sha(path) if path.exists() else None for path in aliases}
    runner = observation_runner(data["train_observations"][0], metadata, device=device)
    infos = training.load_semantic_checkpoint(runner, checkpoint, contract=contract, seed=metadata["seed"])
    fit_report = _fit_p02(runner, data, budget=budget, authorized=True)
    result = {"schema": SCHEMA + ".execution", "binding": binding, "budget_receipt": budget_receipt,
              "fit_report": fit_report, "auxiliary_checkpoint": None, "teacher_deployed": False,
              "physical_success_claimed": False, "PPO_decisions_added": 0, "PPO_updates_added": 0,
              "PPO_optimizer_steps_added": 0}
    if fit_report["accepted_auxiliary_updates"] > 0:
        result["auxiliary_checkpoint"] = save_candidate(runner, target, infos, data=data, report=fit_report,
                                                         binding=binding, budget_receipt=budget_receipt)
    require({str(path): sha(path) if path.exists() else None for path in aliases} == alias_before,
            "convenience latest-pointer artifacts changed unexpectedly")
    result["latest_pointer_artifacts_unchanged"] = True
    training.write_json(report_path, result)
    print(json.dumps({"report": str(report_path), "accepted_AUX": fit_report["accepted_auxiliary_updates"],
                      "attempted_AUX": fit_report["attempted_auxiliary_optimizer_steps"], "PPO_added": 0}))
    return result


if __name__ == "__main__":
    main()

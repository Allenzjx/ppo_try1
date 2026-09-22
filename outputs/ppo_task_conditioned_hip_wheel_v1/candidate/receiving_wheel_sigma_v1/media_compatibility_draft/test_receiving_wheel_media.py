from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import receiving_wheel_provenance as provenance
import receiving_wheel_media as media


def boundary():
    ledger = {
        "schema": "wlr50_clean.auxiliary_mean_learning_ledger.v1",
        "training_lineage_label": "PPO_plus_explicit_finite_auxiliary_mean_supervision",
        "events": [{}],
        "accepted_auxiliary_updates_total": 7,
        "attempted_auxiliary_optimizer_steps_total": 8,
    }
    old_runtime = {
        "experiment_id": "task_conditioned_hip_wheel_v1",
        "files": {"old.py": "a"},
        "runtime_content_sha256": "old",
        "source_git_commit": "old",
        "physics_dt": 1 / 240,
        "training_budgets": {"full_episode": 131072},
    }
    new_runtime = copy.deepcopy(old_runtime)
    new_runtime.update(files={"new.py": "b"}, runtime_content_sha256="new",
                       source_git_commit="new")
    source_policy = {"version": provenance.SOURCE_POLICY, "observation_dimension": 372,
                     "raw_action_dimension": 12}
    target_policy = {**source_policy, "version": provenance.TARGET_POLICY,
                     "sigma_scaling_semantics": provenance.SIGMA_SEMANTICS}
    source_runner = {"actor": {"class_name": "parent"}, "num_steps_per_env": 128}
    target_runner = {"actor": {"class_name": "receiving"}, "num_steps_per_env": 128}
    state = {
        "actor_parameter_sha256": "actor", "critic_parameter_sha256": "critic",
        "optimizer_state_sha256": "adam", "normalizer_state_sha256": "norm",
        "training_rng_state": {"cpu": "rng"},
    }
    source = {
        **state, "runtime_contract": old_runtime, "policy_contract": source_policy,
        "runner_config": source_runner, "checkpoint_sha256": "source-cp",
        provenance.AUX_BRANCH: {provenance.AUX_LEDGER: ledger},
    }
    factor = {
        "schema": provenance.FACTOR_SCHEMA,
        "source_policy_version": provenance.SOURCE_POLICY,
        "target_policy_version": provenance.TARGET_POLICY,
        "source_policy_contract": source_policy,
        "target_policy_contract": target_policy,
        "source_runner_config": source_runner,
        "target_runner_config": target_runner,
        "sigma_scaling_semantics": provenance.SIGMA_SEMANTICS,
        "receiving_continuation_gate": {
            "phase_indices": [9, 10, 11],
            "RR_placed_history_observation_index": 157,
            "RR_placed_history_value": 1,
            "channel_indices": [9, 11],
            "channel_names": ["FR_wheel", "RR_wheel"],
            "conditional_sigma_multiplier": 3.0,
            "otherwise_multiplier": 1.0,
            "current_RR_support_required": False,
            "current_RR_lift_required": False,
            "P13_unchanged": True,
        },
        "observation_contract": {
            "observation_dimension": 372, "action_dimension": 12, "num_envs": 1,
            "parameter_mapping": "identity_all_parameters_and_buffers",
        },
        "preserved_training_state": state,
        "preserved_auxiliary_ledger_sha256": provenance.digest(ledger),
        "kernel_changed": True, "same_mdp_claimed": True,
        "original_branch_origin_preserved": True, "auxiliary_updates_added": 0,
        **{key: False for key in provenance.FALSE_CLAIMS},
    }
    verified = {
        "receiving_wheel_sigma_factor": factor,
        "plan_path": "plan.json", "plan_sha256": "plan",
        "source_checkpoint_sha256": "source-cp",
        "source_contract_sha256": "old-contract",
        "target_contract_sha256": "new-contract",
    }
    target = {
        "runtime_contract": new_runtime, "policy_contract": target_policy,
        "runner_config": target_runner,
        provenance.AUX_BRANCH: copy.deepcopy(source[provenance.AUX_BRANCH]),
    }
    target[provenance.MIGRATION_KEY] = {
        "factor": factor, "plan_path": "plan.json", "plan_sha256": "plan",
        "source_checkpoint_sha256": "source-cp",
        "source_contract_sha256": "old-contract",
        "target_contract_sha256": "new-contract",
    }
    return source, target, verified


def test_accepts_only_exact_sigma_boundary_and_saved_record():
    source, target, verified = boundary()
    result = provenance.validate_saved_record(target, source, verified)
    assert result["gate"]["phase_indices"] == [9, 10, 11]
    assert result["gate"]["channel_indices"] == [9, 11]
    assert result["deterministic_mean_changed"] is False
    assert result["auxiliary_updates_added"] == 0


@pytest.mark.parametrize("mutation", [
    lambda f: f.update(target_policy_version="arbitrary_policy"),
    lambda f: f["receiving_continuation_gate"].update(phase_indices=[1, 2, 3]),
    lambda f: f["receiving_continuation_gate"].update(channel_indices=[0, 1]),
    lambda f: f.update(deterministic_mean_changed=True),
    lambda f: f.update(auxiliary_updates_added=1),
])
def test_rejects_profile_widening(mutation):
    source, target, verified = boundary()
    mutation(verified["receiving_wheel_sigma_factor"])
    target[provenance.MIGRATION_KEY]["factor"] = verified["receiving_wheel_sigma_factor"]
    with pytest.raises(ValueError):
        provenance.validate_saved_record(target, source, verified)


def test_rejects_mixed_factor_and_aux_lineage_change():
    source, target, verified = boundary()
    verified["training_quantity_budget_factor"] = {"unexpected": True}
    with pytest.raises(ValueError, match="mixes another"):
        provenance.validate_saved_record(target, source, verified)
    source, target, verified = boundary()
    target[provenance.AUX_BRANCH][provenance.AUX_LEDGER][
        "attempted_auxiliary_optimizer_steps_total"] = 9
    with pytest.raises(ValueError, match="AUX"):
        provenance.validate_saved_record(target, source, verified)


def test_rejects_receipt_not_identical_to_official_result():
    source, target, verified = boundary()
    target[provenance.MIGRATION_KEY]["plan_sha256"] = "stale"
    with pytest.raises(ValueError, match="official plan result"):
        provenance.validate_saved_record(target, source, verified)


def test_target_ancestry_requires_actual_fresh_ppo_predecessor(tmp_path):
    source, current, verified = boundary()
    source_path = tmp_path / "source.pt"
    current_path = tmp_path / "current.pt"
    source_path.write_bytes(b"source")
    current_path.write_bytes(b"current")
    source_sidecar = tmp_path / "source_manifest.json"
    source_sidecar.write_text("{}", encoding="utf-8")
    source.update(global_policy_decisions=196608, ppo_updates=1501,
                  optimizer_steps=30020, stage_requested_decisions={"full_episode": 12000})
    current.update(global_policy_decisions=196736, ppo_updates=1502,
                   optimizer_steps=30040, stage_requested_decisions={"full_episode": 12128},
                   checkpoint_sha256="target-cp")
    current["resume_ancestry"] = {
        "source_global_policy_decisions": 196608,
        "source_ppo_updates": 1501,
        "source_optimizer_steps": 30020,
        "source_actor_parameter_sha256": source["actor_parameter_sha256"],
        "source_runtime_contract": source["runtime_contract"],
        "source_checkpoint": {
            "checkpoint": str(source_path.resolve()),
            "checkpoint_sha256": source["checkpoint_sha256"],
            "manifest": str(source_sidecar.resolve()),
            "manifest_sha256": provenance.sha(source_sidecar),
        },
        "resume_migration": verified,
    }

    class Official:
        @staticmethod
        def checkpoint_metadata(path):
            assert Path(path) == source_path.resolve()
            return source

    rows = provenance._target_ancestry(
        current_path, current, source_path.resolve(), source, verified, Official,
    )
    assert [row["sha256"] for row in rows] == ["target-cp", "source-cp"]
    current["resume_ancestry"]["source_optimizer_steps"] -= 1
    with pytest.raises(ValueError, match="predecessor"):
        provenance._target_ancestry(
            current_path, current, source_path.resolve(), source, verified, Official,
        )


def test_two_hop_target_ancestry_has_migration_only_on_boundary_hop(tmp_path):
    source, boundary_target, verified = boundary()
    source_path = tmp_path / "source.pt"
    boundary_path = tmp_path / "boundary_target.pt"
    continued_path = tmp_path / "continued_target.pt"
    for path, value in ((source_path, b"source"), (boundary_path, b"boundary"),
                        (continued_path, b"continued")):
        path.write_bytes(value)
        path.with_name(path.stem + "_manifest.json").write_text("{}", encoding="utf-8")
    source.update(global_policy_decisions=196608, ppo_updates=1501,
                  optimizer_steps=30020, stage_requested_decisions={"full_episode": 12000})
    boundary_target.update(global_policy_decisions=196736, ppo_updates=1502,
                           optimizer_steps=30040,
                           stage_requested_decisions={"full_episode": 12128},
                           checkpoint_sha256="boundary-cp",
                           actor_parameter_sha256="boundary-actor")

    def ancestry(parent_path, parent, migration):
        sidecar = parent_path.with_name(parent_path.stem + "_manifest.json")
        return {
            "source_global_policy_decisions": parent["global_policy_decisions"],
            "source_ppo_updates": parent["ppo_updates"],
            "source_optimizer_steps": parent["optimizer_steps"],
            "source_actor_parameter_sha256": parent["actor_parameter_sha256"],
            "source_runtime_contract": parent["runtime_contract"],
            "source_checkpoint": {
                "checkpoint": str(parent_path.resolve()),
                "checkpoint_sha256": parent["checkpoint_sha256"],
                "manifest": str(sidecar.resolve()),
                "manifest_sha256": provenance.sha(sidecar),
            },
            "resume_migration": migration,
        }

    boundary_target["resume_ancestry"] = ancestry(source_path, source, verified)
    continued = copy.deepcopy(boundary_target)
    continued.update(global_policy_decisions=196864, ppo_updates=1503,
                     optimizer_steps=30060,
                     stage_requested_decisions={"full_episode": 12256},
                     checkpoint_sha256="continued-cp",
                     actor_parameter_sha256="continued-actor")
    continued["resume_ancestry"] = ancestry(boundary_path, boundary_target, None)
    records = {source_path.resolve(): source, boundary_path.resolve(): boundary_target}

    class Official:
        @staticmethod
        def checkpoint_metadata(path):
            return records[Path(path)]

    rows = provenance._target_ancestry(
        continued_path, continued, source_path.resolve(), source, verified, Official,
    )
    assert [row["sha256"] for row in rows] == [
        "continued-cp", "boundary-cp", "source-cp",
    ]
    boundary_target["resume_ancestry"]["resume_migration"] = None
    with pytest.raises(ValueError, match="predecessor and migration"):
        provenance._target_ancestry(
            continued_path, continued, source_path.resolve(), source, verified, Official,
        )


def test_pair_uses_distinct_archived_quantity_boundary():
    source_runtime = {"experiment_id": "task_conditioned_hip_wheel_v1", "version": "quantity-source"}
    quantity_target = {"experiment_id": "task_conditioned_hip_wheel_v1", "version": "quantity-target"}
    source_eval, target_eval = {"execution_profile": {"sha256": "old"}}, {
        "execution_profile": {"sha256": "quantity"}}
    entry = {"seed": 4001, "kind": "natural_P01"}
    identity = {
        "sigma_source_checkpoint_identity": {"runtime_contract": quantity_target},
        "historical_quantity_boundary": {
            "schema": "wlr50_clean.historical_quantity_aux_posthoc.v1",
            "posthoc_only_no_training": True,
            "quantity_source_runtime_contract": source_runtime,
            "quantity_target_runtime_contract": quantity_target,
            "source_evaluation_configuration": source_eval,
            "target_evaluation_configuration": target_eval,
            "historical_runtime": {"detached_head": "97ecd305"},
            "quantity_plan": "sealed-plan.json", "quantity_plan_sha256": "plan-hash",
        },
    }
    b = {"receipt": {"role": "B", "mode": "N_plus_zero"}, "manifest": {
        "experiment_id": "task_conditioned_hip_wheel_v1", "camera": {"view": "whole"},
        "seed": 4001, "natural_reset_proof": {"entry": entry},
        "runtime_contract": source_runtime, "evaluation_configuration": source_eval}}
    c = {"receipt": {"role": "C", "mode": "deterministic_conditional_mean"}, "manifest": {
        "experiment_id": "task_conditioned_hip_wheel_v1", "camera": {"view": "whole"},
        "seed": 4001, "natural_reset_proof": {"entry": entry},
        "runtime_contract": {"version": "sigma-target"},
        "evaluation_configuration": target_eval}}
    result = media._archived_quantity_control(b, c, identity)
    assert result["same_control_via_archived_verified_quantity_only_boundary"] is True
    assert result["current_sigma_runtime_verified_separately"] is True
    c["manifest"]["evaluation_configuration"] = source_eval
    with pytest.raises(ValueError, match="archived quantity boundary"):
        media._archived_quantity_control(b, c, identity)


def test_lazy_adapters_construct_without_training_encoding_or_checkpoint_load():
    historical = Path("explicit-historical-root-not-read-until-formal-use")
    adapter = media.make_adapter(Path("not-read-until-formal-use.json"), historical)
    assert callable(adapter.export) and callable(adapter.pair) and callable(adapter.modes)
    summarize = media.make_training_summarizer(
        Path("not-read-until-formal-use.json"), historical)
    assert callable(summarize)

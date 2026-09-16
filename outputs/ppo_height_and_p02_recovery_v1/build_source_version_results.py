"""Output-only scalar version/result inventory; no active-run reads or CSV."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TIMING = HERE.parent / "ppo_timing_task_priority_v1"
FIELDS = "run_label lifecycle intervention_label source_git_commit runtime_content_sha256 seed training_seed physics_hz decision_hz environment_lock_sha256 asset_sha256 scene_factory_sha256 execution_profile_sha256 observation_schema_sha256 action_schema_sha256 reward_config_sha256 reward_implementation_sha256 nominal_provider_sha256 nominal_geometry_sha256 task_spec_sha256 checkpoint_decisions checkpoint_ppo_updates checkpoint_optimizer_steps evaluation_optimizer_updates on_policy_training_samples physical_ticks physical_duration_s terminal_phase original_controller_result physical_result physical_task_success result_provenance formal_C0 formal_P02_evidence stability_superiority_claim media_path media_receipt_path source_run source_manifest source_checkpoint_manifest runtime_versions_json frozen_A_files_json initial_equality_evidence comparability_note known_limit planned_decisions actual_added_decisions actual_added_PPO_updates migration_plan target_contract_sha256_not_runtime_sha256".split()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def row(label):
    value = dict.fromkeys(FIELDS)
    value.update(run_label=label, stability_superiority_claim=False)
    return value


def media_row(label, path, note):
    receipt = load(path)
    source_path = Path(receipt["source_manifest"])
    source = load(source_path)
    run = load(source_path.parent.parent / "run_manifest.json")
    assert run.get("completed_at_utc") and run["runtime_contract"] == source["runtime_contract"] == receipt["runtime_contract"]
    runtime = source["runtime_contract"]
    files, cfg, frozen = runtime["files"], source["evaluation_configuration"], runtime["frozen_A_files"]
    proof = source.get("checkpoint_load_provenance") or {}
    cp_path = (proof.get("source") or {}).get("manifest")
    cp = load(cp_path) if cp_path else {}
    value = row(label)
    value.update(lifecycle=run["lifecycle"], intervention_label=(source.get("diagnostic_intervention") or {}).get("mode"),
        source_git_commit=runtime["source_git_commit"], runtime_content_sha256=runtime["runtime_content_sha256"],
        seed=source["seed"], training_seed=proof.get("training_seed"), physics_hz=runtime["physics_hz"], decision_hz=runtime["decision_hz"],
        environment_lock_sha256=frozen.get("configs/environment_lock.json"), scene_factory_sha256=files.get("src/wlr50_clean/infrastructure/scene_factory.py"),
        runtime_versions_json=json.dumps(runtime["local_runtime_versions"], sort_keys=True), frozen_A_files_json=json.dumps(frozen, sort_keys=True),
        checkpoint_decisions=proof.get("saved_global_policy_decisions"), evaluation_optimizer_updates=source["optimizer_updates"],
        on_policy_training_samples=source.get("on_policy_training_samples", 0), physical_ticks=source["episode_physics_ticks"],
        physical_duration_s=source["episode_physics_ticks"]/runtime["physics_hz"], terminal_phase=receipt["terminal_phase"],
        physical_result=receipt["physical_result"], physical_task_success=source["physical_task_success"],
        formal_C0=receipt.get("formal_C0", False), formal_P02_evidence=source.get("formal_P02_evidence"),
        media_path=receipt["output"], media_receipt_path=str(path.resolve()), source_run=str(source_path.parent.parent),
        source_manifest=str(source_path), source_checkpoint_manifest=cp_path, result_provenance="Finalized source + existing verified media receipt; no success relabeling",
        comparability_note=note, known_limit="Environment/config hashes bind declared content; they do not by themselves prove equal trajectories or measured motor torque.")
    for key, original in (("execution_profile_sha256", "execution_profile"), ("observation_schema_sha256", "observation_schema_path"),
        ("action_schema_sha256", "action_schema_path"), ("reward_config_sha256", "reward_config_path"), ("task_spec_sha256", "task_spec_path")):
        value[key] = cfg[original]["sha256"]
    for key, file in (("reward_implementation_sha256", "semantic_reward.py"), ("nominal_provider_sha256", "semantic_supervisor.py"), ("nominal_geometry_sha256", "semantic_nominal_geometry.py")):
        value[key] = files.get("src/wlr50_clean/ppo/" + file)
    # Checkpoint counters are intentionally filled only from known actual sidecar fields.
    value["checkpoint_ppo_updates"] = cp.get("ppo_updates")
    value["checkpoint_optimizer_steps"] = cp.get("optimizer_steps")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Never overwrite result snapshots"
    identity = load(TIMING / "reference_identity.json")
    a_path = Path(identity["A_physical_source"]) / "trial_manifest.json"
    a, adjudication = load(a_path), load(identity["A_existing_physical_reclassification"])["selected_success_trial"]
    first = row("A_FROZEN_EXISTING_REFERENCE")
    first.update(lifecycle="EXISTING_FROZEN_REFERENCE", physics_hz=a["physics_hz"], decision_hz=a["decision_hz"],
        environment_lock_sha256=adjudication["environment_hash"], asset_sha256=adjudication["robot_asset_hash"],
        physical_ticks=a["control_steps"], physical_duration_s=a["control_steps"]/a["physics_hz"], terminal_phase="P13",
        original_controller_result=a["result"], physical_result="EXISTING_PHYSICAL_READJUDICATION_SUCCESS",
        physical_task_success=adjudication["P01_P13_complete"] and adjudication["physical_traversal_complete"],
        result_provenance=identity["A_existing_physical_reclassification"], formal_C0=False, formal_P02_evidence=False,
        media_path=identity["A_video"], media_receipt_path=identity["A_publication_receipt"], source_run=str(a_path.parent), source_manifest=str(a_path),
        comparability_note=identity["A_classification_boundary"], known_limit="No modern runtime digest/HEAD or seed retained in trial manifest. Original historical conformance failure remains preserved; not a newly rerun A.")
    rows = [first,
        media_row("B2_ZERO_OLD_GEOMETRY", TIMING / "videos/zero_residual_after.media.json", "Old B2 vs old C: same nominal/physics/initial arrays but explicitly reviewed reward-only runtime delta; not identical runtime/reward/actor."),
        media_row("C168192_OLD_UNMASKED", TIMING / "videos/ppo_after.media.json", "Unmasked frozen CP168192 natural P01 evaluation; physical P02 safety abort, not successful PPO. Runtime differs from new height candidate."),
        media_row("B_FL_MINUS4_HEIGHT_V1", HERE / "videos/zero_FLminus4_incomplete.media.json", "New live-collider bounded geometry AND FL minus4 differ from B2. Equal initial arrays do not isolate FL effect."),
        media_row("D_WHEELS4_MASKED_ONLY", HERE / "videos/p02_wheels4_masked_diagnostic_only_16s.media.json", "All wheel policy raw8..11 zero from first decision, other8 closed-loop with actual masked HISTORY. Not formal all12 C0 or a same-runtime replicate of old C.")]
    rows[1]["initial_equality_evidence"] = rows[2]["initial_equality_evidence"] = str((TIMING / "videos/zero_vs_ppo_after.media.json").resolve())
    rows[3]["initial_equality_evidence"] = str((HERE / "B_HEIGHT_FL4.aggregate.json").resolve())
    rows[3]["original_controller_result"] = "INCOMPLETE_CONTROLLER_BLOCKED"
    rows[4]["initial_equality_evidence"] = str((HERE / "p02_wheels4_completed_comparison.json").resolve())
    rows[4]["original_controller_result"] = "DIAGNOSTIC_BOUNDED_WINDOW"
    plan_path = HERE / "checkpoint168192_RL3_late_entry_advantage_migration.json"
    plan = load(plan_path)
    active = row("TRAIN_A9C_P01_2048_IN_PROGRESS")
    active.update(lifecycle="IN_PROGRESS", source_git_commit=plan["target_git_commit"], planned_decisions=2048,
        migration_plan=str(plan_path.resolve()), target_contract_sha256_not_runtime_sha256=plan["target_contract_sha256"],
        checkpoint_decisions=168192, checkpoint_ppo_updates=1279, checkpoint_optimizer_steps=25580,
        source_checkpoint_manifest=str(ROOT / "outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000168192_manifest.json"),
        result_provenance="Parent reported real P01 training launch; no active run/stream read and no completed counter claimed",
        comparability_note="RL minus3 / FL0, live-collider geometry, guarded P09 late atomic entry plus read-only GAE audit. Await actual formal all12 evaluation and RL3 zero on same runtime.",
        known_limit="All actual added counters, end state, runtime digest, seed and outcome remain null pending finalized evidence; planned 2048 is not completed work.")
    rows.append(active)
    assert all(set(value) == set(FIELDS) and all(not isinstance(v, (dict, list)) for v in value.values()) for value in rows)
    payload = dict(schema="wlr50_clean.source_version_result_scalar_rows.v1", rows=rows, columns=FIELDS,
        active_training_stream_read=False, CSV_written=False, old_results_overwritten=False,
        contract="Seeds, runtime content, physical/config provenance, reward and intervention are separate columns. null means unknown/not yet finalized, never zero. No stability superiority or new PPO success is claimed.")
    with args.output.open("x", encoding="utf-8") as stream: json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(rows), "columns": len(FIELDS), "active_training_counts_claimed": False}))


if __name__ == "__main__": main()

"""Root-bound zero-update control-semantic migration; never simulates or fits."""
import argparse
import json
from pathlib import Path

from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import checkpoint_metadata, file_sha
from wlr50_clean.ppo.semantic_p05_preedge_migration import (
    SCHEMA, build_p05_preedge_migration, publish_p05_preedge_checkpoint,
)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
SOURCE = OUT / "checkpoints/history/checkpoint_step_000216448.pt"
SOURCE_SHA = "8ec7784a9078ae7e9bb5f1d7298a4aa655c0d979ddb906e8a4d577bf8079a8f4"
TARGET = OUT / "checkpoints/history/checkpoint_preedge_recovery_step_000216448.pt"
PLAN = HERE / "p05_preedge_actual_migration_plan.json"
REPORT = HERE / "p05_preedge_actual_migration.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    if any(p.exists() for p in (TARGET, TARGET.with_name(TARGET.stem + "_manifest.json"), PLAN, REPORT)):
        raise FileExistsError("Immutable migration outputs already exist; inspect instead of overwriting")
    if file_sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Root-selected CP216448 differs")
    source = checkpoint_metadata(SOURCE)
    counters = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
    if tuple(source[k] for k in counters) != (216448, 1656, 33120):
        raise ValueError("Actual source counters differ")
    ledger = source["rr_postcross_workspace_branch"]["front_rehearsal_auxiliary"]
    last = ledger["events"][-1]
    if (len(ledger["events"]) != 4 or last["event_index"] != 4
            or last["fit_report"]["accepted_auxiliary_updates"] != 7
            or last["fit_report"]["attempted_auxiliary_optimizer_steps"] != 8):
        raise ValueError("Four-event AUX lineage changed; no event5 is authorized here")
    contract = runtime_contract(expected_head=args.expected_head, semantic_version="v3",
                                experiment_id="p05_hip_only_continuation_v1")
    before = source["runtime_contract"]["files"]
    changes = {p: h for p, h in contract["files"].items() if before.get(p) != h}
    expected = {
        "configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml",
        "src/wlr50_clean/ppo/semantic_supervisor.py",
        "src/wlr50_clean/ppo/semantic_p05_preedge_migration.py",
        "src/wlr50_clean/ppo/semantic_migration.py",
        "src/wlr50_clean/ppo/semantic_training.py",
    }
    if set(changes) != expected or set(before) - set(contract["files"]):
        raise ValueError("Unreviewed production change at migration boundary")
    plan = build_p05_preedge_migration(SOURCE, contract,
        reason="Finite measured P05 pre-edge wheel approach after its local deadline; preserve actual cross/contact/placement and later-stage gates, all residual channels and original source stops.",
        reviewed_code_sha256=changes)
    if plan["schema"] != SCHEMA:
        raise ValueError("Expected dedicated same389 control-MDP migration")
    PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = publish_p05_preedge_checkpoint(SOURCE, contract, PLAN, TARGET)
    target = checkpoint_metadata(TARGET)
    for key in counters + ("actor_parameter_sha256", "critic_parameter_sha256", "optimizer_state_sha256",
                            "normalizer_state_sha256", "optimizer_learning_rate", "training_rng_state",
                            "runner_config"):
        if target[key] != source[key]:
            raise ValueError(f"Identity migration changed {key}")
    if target["rr_postcross_workspace_branch"] != source["rr_postcross_workspace_branch"]:
        raise ValueError("Inherited complete RR/AUX branch changed")
    result.update(source_checkpoint=str(SOURCE), source_checkpoint_sha256=SOURCE_SHA,
                  target_checkpoint_sha256=file_sha(TARGET), target_head=args.expected_head,
                  old_rollout_discarded=True, physical_state_inherited=False,
                  PPO_decisions_added=0, PPO_updates_added=0, AUX_updates_added=0)
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"target": str(TARGET), "sha256": file_sha(TARGET), "report": str(REPORT),
                      "decisions": target[counters[0]], "PPO_updates": target[counters[1]],
                      "Adam_steps": target[counters[2]], "migration_added_updates": 0}))


if __name__ == "__main__":
    main()

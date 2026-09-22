"""One root-selected zero-update identity migration; never trains or simulates."""
import json
from pathlib import Path

from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import checkpoint_metadata, file_sha
from wlr50_clean.ppo.semantic_rr_workspace_migration import (
    build_rr_workspace_migration, publish_rr_workspace_checkpoint, SCHEMA_V2,
)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
HEAD = "336b7c56d2f048d44c866b2357d33e1eb1f21dce"
SOURCE = OUT / "checkpoints/history/checkpoint_aux_meanrr_step_000214400_v4.pt"
SOURCE_SHA = "9987a4df3b4a2afaeddac794aa97b650e6133fe7ec89e8484cf59084cda69b6a"
TARGET = OUT / "checkpoints/history/checkpoint_rr_receiver_v2_step_000214400.pt"
PLAN = HERE / "rr_receiver_v2_migration_plan.json"
REPORT = HERE / "rr_receiver_v2_actual_migration.json"


def main():
    if any(path.exists() for path in (TARGET, TARGET.with_name(TARGET.stem + "_manifest.json"), PLAN, REPORT)):
        raise FileExistsError("The selected migration is immutable and cannot be retried into existing outputs")
    if file_sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Root-selected actual RR auxiliary checkpoint differs")
    metadata = checkpoint_metadata(SOURCE)
    if tuple(metadata[k] for k in ("global_policy_decisions", "ppo_updates", "optimizer_steps")) != (214400, 1640, 32800):
        raise ValueError("Actual source counters differ")
    ledger = metadata["rr_postcross_workspace_branch"]["front_rehearsal_auxiliary"]
    event = ledger["events"][-1]
    if (len(ledger["events"]) != 4 or event["event_index"] != 4
            or event["fit_report"]["accepted_auxiliary_updates"] != 7
            or event["fit_report"]["attempted_auxiliary_optimizer_steps"] != 8):
        raise ValueError("The actual separately counted RR7/8 event was not preserved")
    contract = runtime_contract(expected_head=HEAD, semantic_version="v3", experiment_id="p05_hip_only_continuation_v1")
    before = metadata["runtime_contract"]["files"]
    changes = {path: sha for path, sha in contract["files"].items() if before.get(path) != sha}
    expected = {
        "configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml",
        "src/wlr50_clean/ppo/semantic_supervisor.py",
        "src/wlr50_clean/ppo/semantic_rr_workspace_migration.py",
        "src/wlr50_clean/ppo/semantic_migration.py",
        "src/wlr50_clean/ppo/semantic_training.py",
    }
    if set(changes) != expected:
        raise ValueError("Unreviewed production change at the selected boundary")
    plan = build_rr_workspace_migration(SOURCE, contract,
        reason="Receiver preparation remains retired during an established legal RR capture attempt despite transient loss of current body-support qualification; physical qualification and controls unchanged.",
        reviewed_code_sha256=changes)
    if plan["schema"] != SCHEMA_V2:
        raise ValueError("Unexpected migration semantics")
    PLAN.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = publish_rr_workspace_checkpoint(SOURCE, contract, PLAN, TARGET)
    result.update(source_checkpoint=str(SOURCE), source_checkpoint_sha256=SOURCE_SHA,
        target_checkpoint_sha256=file_sha(TARGET), target_head=HEAD,
        old_rollout_discarded=True, physical_state_inherited=False,
        new_RR_auxiliary_accepted=7, new_RR_auxiliary_attempted=8,
        inherited_front_auxiliary_accepted=96, inherited_historical_auxiliary_accepted=7)
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

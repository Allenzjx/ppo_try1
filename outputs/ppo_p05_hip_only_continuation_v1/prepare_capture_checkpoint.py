"""Explicit one-time state-preserving boundary; no simulator or PPO credit."""
import argparse
import json
from pathlib import Path

from wlr50_clean.ppo.semantic_cli import runtime_contract, PROJECT_ROOT
from wlr50_clean.ppo.semantic_migration import checkpoint_metadata
from wlr50_clean.ppo.semantic_p05_capture_migration import build_p05_capture_migration, publish_p05_capture_checkpoint
from wlr50_clean.ppo.semantic_training import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    root = PROJECT_ROOT / "outputs/ppo_p05_hip_only_continuation_v1"
    contract = runtime_contract(expected_head=args.expected_head, semantic_version="v3",
                                experiment_id="p05_hip_only_continuation_v1")
    source = checkpoint_metadata(args.checkpoint)
    changed = {p:h for p,h in contract["files"].items()
        if source["runtime_contract"]["files"].get(p) != h
        and not p.startswith("configs/ppo_p05_hip_only_continuation_v1/")}
    record = build_p05_capture_migration(args.checkpoint, contract,
        reason="User-authorized observable hip-only capture assist and nonterminal P05 pending continuation; preserve all old weights, Adam, Identity, RNG, action capacity and reward; fresh rollout.",
        reviewed_code_sha256=changed, project_root=PROJECT_ROOT)
    tag = args.expected_head[:12]
    plan = root / f"capture_migration_plan_g{tag}.json"
    write_json(plan, record)
    write_json(root / f"capture_runtime_contract_g{tag}.json", contract)
    destination = root / "checkpoints/history" / f"checkpoint_capture_migrated_step_{source['global_policy_decisions']:09d}.pt"
    receipt = publish_p05_capture_checkpoint(args.checkpoint, contract, plan, destination)
    write_json(root / f"capture_migration_receipt_g{tag}.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()

"""Create one explicit sigma-only plan after safe, committed adoption.

Metadata only: no runner, policy forward, checkpoint mutation or simulation.
The production builder validates the entire actual source/target boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ppo_task_conditioned_hip_wheel_v1"
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    from wlr50_clean.ppo.semantic_migration import (
        build_migration_plan, validate_migration_plan, checkpoint_metadata,
    )
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    if (not checkpoint.is_relative_to(OUT / "checkpoints/history")
            or not output.is_relative_to(OUT) or output.exists()):
        raise ValueError("use an existing immutable branch checkpoint and a new isolated plan path")
    seal = json.loads((OUT / "candidate/receiving_wheel_sigma_v1/SEALED_FILES.json").read_text(encoding="utf-8"))
    expected = seal["candidate_file_sha256"]
    actual = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in expected}
    if actual != expected:
        raise ValueError("adopted runtime bytes differ from the independently reviewed candidate")
    source = checkpoint_metadata(checkpoint)
    target = runtime_contract(expected_head=args.expected_head, semantic_version="v3",
                              experiment_id="task_conditioned_hip_wheel_v1")
    reason = ("Bounded receiving-continuation exploration after recurrent actual RR post-placement retreat; "
              "P10-P12 and historical RR placed only, FR/RR wheel conditional sigma x3. "
              "Same physical MDP, no mean/HISTORY/cap/reward/control change; no physical success claimed.")
    plan = build_migration_plan(checkpoint, target, allowed_changed_files=sorted(expected), reason=reason,
        receiving_wheel_sigma_review={"reason": reason, "reviewed_code_sha256": expected},
        project_root=ROOT)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
    verified = validate_migration_plan(checkpoint, target, output, project_root=ROOT)
    print(json.dumps({"plan": str(output), "plan_sha256": verified["plan_sha256"],
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_counters": {k: source[k] for k in ("global_policy_decisions", "ppo_updates", "optimizer_steps")},
        "source_effective_learning_rate": source["optimizer_learning_rate"],
        "target_policy": plan["receiving_wheel_sigma_factor"]["target_policy_version"],
        "target_head": target["source_git_commit"], "same_physical_MDP": True,
        "changed_stochastic_kernel": True, "migration_learning_credit": 0}, indent=2))


if __name__ == "__main__":
    main()

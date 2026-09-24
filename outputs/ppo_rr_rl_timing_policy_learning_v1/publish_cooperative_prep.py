"""Publish the reviewed same422 cooperative-prep boundary; no learning/pointer."""
import argparse
import json
import subprocess
from pathlib import Path

from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_cooperative_prep_migration import (
    BRANCH_NAME, EXPERIMENT, REVISION_ORIGIN, SOURCE_MANIFEST_SHA, SOURCE_SHA,
    build_cooperative_prep_migration, publish_cooperative_prep_checkpoint,
)
from wlr50_clean.ppo.semantic_training import write_json


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head:
        raise ValueError("explicit frozen target HEAD differs from actual HEAD")
    contract = runtime_contract(
        expected_head=head, semantic_version="v3", experiment_id=EXPERIMENT)
    output = ROOT / "outputs" / ("ppo_" + EXPERIMENT)
    branch = output / "branches" / BRANCH_NAME
    source = branch / "checkpoints/history/checkpoint_step_000223232.pt"
    label = "cooperative_prep_CP223232_g" + head[:12]
    plan_path = output / (label + "_migration.json")
    target = branch / "checkpoints/history" / ("checkpoint_" + label + ".pt")
    receipt_path = output / (label + "_publication.json")
    sidecar = target.with_name(target.stem + "_manifest.json")
    if any(path.exists() for path in (plan_path, target, sidecar, receipt_path)):
        raise FileExistsError("unique cooperative-prep publication already exists")
    plan = build_cooperative_prep_migration(
        source, contract,
        reason=("Completed d7e learned422 course; same422 full-state identity; "
                "observable cooperative preparation sigma and bounded measured "
                "workspace/counterroll reward telemetry; fresh rollout; zero migration credit"))
    if not args.publish:
        print(json.dumps({"validated": True, "publish": False,
            "source": str(source), "source_sha256": SOURCE_SHA,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA,
            "source_counters": REVISION_ORIGIN, "target": str(target)}))
        return
    write_json(plan_path, plan)
    receipt = publish_cooperative_prep_checkpoint(
        source, contract, plan_path, target)
    write_json(receipt_path, receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()


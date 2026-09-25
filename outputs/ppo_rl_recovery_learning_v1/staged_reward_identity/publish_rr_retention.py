"""Outputs-only publisher; run only at an approved idle/frozen production boundary."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output-checkpoint", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--publish", action="store_true")
    args = p.parse_args()
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    from wlr50_clean.ppo.semantic_rr_retention_migration import (
        EXPERIMENT, build_rr_retention_migration, publish_rr_retention_checkpoint)
    from wlr50_clean.ppo.semantic_training import write_json
    contract = runtime_contract(expected_head=args.expected_head, semantic_version="v3", experiment_id=EXPERIMENT)
    record = build_rr_retention_migration(args.checkpoint, contract, reason=args.reason,
        expected_source_sha256=args.source_sha256, expected_manifest_sha256=args.manifest_sha256)
    targets = (args.output_checkpoint, args.output_checkpoint.with_name(args.output_checkpoint.stem + "_manifest.json"),
               args.plan, args.receipt)
    if len({p.resolve() for p in targets}) != len(targets) or any(p.exists() for p in targets):
        raise FileExistsError("publication artifacts must be distinct new paths; preserve every old artifact")
    if not args.publish:
        print(json.dumps(dict(validated=True, source=str(args.checkpoint.resolve()),
            revision_counter_origin=record["rr_retention_same439_factor"]["revision_counter_origin"],
            changed_paths=sorted(record["changed_file_hashes"]))))
        return
    write_json(args.plan, record)
    receipt = publish_rr_retention_checkpoint(args.checkpoint, contract, args.plan, args.output_checkpoint)
    write_json(args.receipt, receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

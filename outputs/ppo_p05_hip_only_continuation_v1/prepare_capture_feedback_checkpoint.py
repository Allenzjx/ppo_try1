"""Explicit same389 feedback-v2 publisher. Run only after runtime commit/freeze.

Default invocation validates/binds the plan without publishing. Add --publish
only after Isaac exits, with source CUDA visibility intact. No pointer updates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


SOURCE_SHA256 = "8e4a72db49c01faee9749db6752daab3db2428a89ce67995c34e515f2526bff8"
REASON = (
    "Confirmed HOLD-to-air recontact progress-window repair; same389 observations, "
    "raw Gaussian policy, reward, nominal, caps, sigma and physical assets. Preserve all "
    "compatible actor/critic tensors, Adam state and actual LR, Identity, RNG, original "
    "P05/task/AUX lineage and lifetime counters; independent feedback-v2 origin; fresh rollout."
)


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, default=project / "outputs" /
        "ppo_p05_hip_only_continuation_v1/checkpoints/history/checkpoint_step_000203776.pt")
    parser.add_argument("--expected-source-sha256", default=SOURCE_SHA256)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()

    from wlr50_clean.ppo.semantic_cli import PROJECT_ROOT, runtime_contract
    from wlr50_clean.ppo.semantic_migration import checkpoint_metadata, validate_migration_plan
    from wlr50_clean.ppo.semantic_capture_feedback_migration import (
        build_capture_feedback_migration, publish_capture_feedback_checkpoint,
    )
    from wlr50_clean.ppo.semantic_training import write_json

    if project != PROJECT_ROOT.resolve():
        raise RuntimeError("driver project differs from the imported production runtime")
    checkpoint = args.checkpoint.resolve(strict=True)
    metadata = checkpoint_metadata(checkpoint)
    if metadata["checkpoint_sha256"] != args.expected_source_sha256:
        raise RuntimeError("source checkpoint differs from the explicitly selected immutable SHA")
    head = subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    # This rejects dirty runtime/config files and validates frozen A before writing a plan.
    contract = runtime_contract(expected_head=head, semantic_version="v3",
                                experiment_id="p05_hip_only_continuation_v1")
    old = metadata["runtime_contract"]
    reviewed = {path:sha for path,sha in contract["files"].items() if old["files"].get(path) != sha}
    plan = build_capture_feedback_migration(checkpoint, contract, reason=REASON,
        reviewed_code_sha256=reviewed, project_root=project)
    tag = f"step_{metadata['global_policy_decisions']:09d}_g{head[:12]}"
    artifacts = project / "outputs/ppo_p05_hip_only_continuation_v1" / f"capture_feedback_v2_{tag}"
    artifacts.mkdir(parents=True, exist_ok=True)

    def write_once(path, value):
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != value:
                raise RuntimeError("immutable migration artifact already differs: " + str(path))
        else:
            write_json(path, value)

    plan_path = artifacts / "migration_plan.json"
    write_once(plan_path, plan)
    write_once(artifacts / "runtime_contract.json", contract)
    verified = validate_migration_plan(checkpoint, contract, plan_path, project_root=project)
    destination = project / "outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history" / (
        f"checkpoint_capture_feedback_v2_{tag}.pt")
    summary = {"source_checkpoint":str(checkpoint), "source_checkpoint_sha256":metadata["checkpoint_sha256"],
        "target_head":head, "plan":str(plan_path), "plan_sha256":verified["plan_sha256"],
        "destination_checkpoint":str(destination), "source_device":metadata["runner_config"]["device"],
        "source_global_policy_decisions":metadata["global_policy_decisions"],
        "source_ppo_updates":metadata["ppo_updates"], "source_optimizer_steps":metadata["optimizer_steps"],
        "source_effective_learning_rate":metadata["optimizer_learning_rate"],
        "publication_requested":args.publish, "pointers_updated":False}
    if args.publish:
        if destination.exists() or destination.with_name(destination.stem+"_manifest.json").exists():
            raise FileExistsError("target already exists; inspect and resume it, do not republish: "+str(destination))
        import torch
        expected_cuda = metadata["training_rng_state"]["torch_cuda_device_count"]
        if torch.cuda.device_count() != expected_cuda:
            raise RuntimeError("source CUDA RNG visibility differs; restore original visible devices before publication")
        receipt = publish_capture_feedback_checkpoint(checkpoint, contract, plan_path, destination)
        write_once(artifacts / "publication_receipt.json", receipt)
        summary["publication"] = receipt
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

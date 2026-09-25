"""One explicit latest-state identity and one fixed finite AUX application.

Separate from PPO storage/optimizer; no simulator or deployed teacher. This
script never promotes latest/best and never repeats a rejected learning rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE / "staged_front_retention439"))
from inspect_front_retention439 import BRANCH, make_runner, prepare_inputs
import data_contract as data
import front_retention439 as kernel


def main():
    from wlr50_clean.ppo import semantic_training as training
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    from wlr50_clean.ppo.semantic_migration import checkpoint_metadata
    from wlr50_clean.ppo.semantic_front_retention439 import (
        build_front_retention439_identity, publish_front_retention439_checkpoint,
        append_front_retention439_event, LEDGER)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("mode", choices=("identity", "fit"))
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--identity-publication")
    args = parser.parse_args()
    contract = runtime_contract(expected_head=args.expected_head, semantic_version="v3", experiment_id="rr_rl_timing_policy_learning_v1")
    label = "front_retention439_CP229120_g" + args.expected_head[:12]
    if args.mode == "identity":
        pointer = json.loads((BRANCH / "checkpoints/checkpoint_last_pointer.json").read_text())
        source = Path(pointer["checkpoint"])
        metadata = checkpoint_metadata(source)
        if metadata["global_policy_decisions"] != 229120:
            raise RuntimeError("reviewed latest source changed; do not silently fit another model")
        plan_path = HERE / (label + "_identity.json")
        publication = HERE / (label + "_identity_publication.json")
        target = BRANCH / "checkpoints/history" / ("checkpoint_" + label + "_identity.pt")
        if any(p.exists() for p in (plan_path, publication, target)):
            raise FileExistsError("identity already attempted; preserve immutable evidence")
        plan = build_front_retention439_identity(source, contract,
            reason="Latest CP229120 retained in full; independent finite pre-intervention student-prefix AUX accounting; no controller, distribution, reward or physical change",
            expected_source_sha256=pointer["checkpoint_sha256"], expected_manifest_sha256=pointer["manifest_sha256"])
        training.write_json(plan_path, plan)
        receipt = publish_front_retention439_checkpoint(source, contract, plan_path, target)
        training.write_json(publication, receipt)
        print(json.dumps({"publication": str(publication), **receipt}))
        return
    if not args.identity_publication:
        raise ValueError("fit requires the explicit sealed identity publication")
    publication = Path(args.identity_publication).resolve(strict=True)
    identity = json.loads(publication.read_text())
    source = Path(identity["checkpoint"])
    sidecar = Path(identity["manifest"])
    if data.file_sha256(source) != identity["checkpoint_sha256"] or data.file_sha256(sidecar) != identity["manifest_sha256"]:
        raise RuntimeError("identity checkpoint bytes changed")
    metadata = checkpoint_metadata(source)
    if metadata["runtime_contract"] != contract or metadata["global_policy_decisions"] != 229120:
        raise RuntimeError("fit runtime/counter scope changed")
    destination = BRANCH / "checkpoints/history" / ("checkpoint_" + label + "_aux.pt")
    paths = {name: HERE / (label + "_" + name + ".json") for name in ("data", "budget", "fit", "aux_publication")}
    actor_recovery = HERE / (label + "_accepted_actor_RECOVERY_ONLY.pt")
    if any(p.exists() for p in (*paths.values(), destination, actor_recovery)):
        raise FileExistsError("fixed fit already attempted; no search/retry or overwrite")
    # Fixed once, justified by prior phase-column retention response and this
    # source's read-only gradient scale. Trust limits are NOT task-success gates.
    budget = kernel.Budget(max_attempts=32, learning_rate=1000.,
        maximum_train_request_shift_full12=(.25,)*8+(.005,)*4,
        maximum_validation_request_shift_full12=(.25,)*8+(.005,)*4,
        maximum_per_state_full_gaussian_kl=.10, maximum_abs_log_sigma_change=.125)
    from dataclasses import asdict
    training.write_json(paths["budget"], {"budget": asdict(budget),
        "rationale": "SINGLE_FIT_BUDGET_RECOMMENDATION.md; source gradient norm 1.0891219972e-5; fixed LR, first rejected proposal restores/stops; no physical improvement claim",
        "source_identity_publication": {"path": str(publication), "sha256": data.file_sha256(publication)}})
    runner = make_runner(metadata)
    infos = training.load_semantic_checkpoint(runner, source, contract=contract, seed=metadata["seed"])
    selected, train, hold, invariant = prepare_inputs(str(runner.device))
    training.write_json(paths["data"], selected["receipt"])
    data_binding = {"path": str(paths["data"].resolve()), "sha256": data.file_sha256(paths["data"])}
    report = kernel.fit(runner, train, [r["target_raw"] for r in selected["train_rows"]],
        hold, [r["target_raw"] for r in selected["holdout_rows"]], invariant,
        seed=metadata["seed"], budget=budget, data_receipt=data_binding, authorized=True)
    report.update(invariance_provenance=selected["invariance_provenance"],
        budget_receipt={"path": str(paths["budget"].resolve()), "sha256": data.file_sha256(paths["budget"])})
    if report["accepted_auxiliary_updates"]:
        # Retain the accepted leaf even if a later first-time lineage/save
        # validation fails. This is NOT a deployable checkpoint or PPO update.
        import torch
        torch.save(runner.alg.actor.state_dict(), actor_recovery)
        report["accepted_actor_recovery_only"] = {"path": str(actor_recovery.resolve()),
            "sha256": data.file_sha256(actor_recovery), "deployable_checkpoint": False}
    training.write_json(paths["fit"], report)
    if report["accepted_auxiliary_updates"] == 0:
        print(json.dumps({"fit_report": str(paths["fit"]), "accepted_auxiliary_updates": 0,
            "attempted_auxiliary_optimizer_steps": report["attempted_auxiliary_optimizer_steps"], "checkpoint_published": False}))
        return
    new_infos = append_front_retention439_event(infos, source_checkpoint=source,
        expected_source_sha256=identity["checkpoint_sha256"], expected_manifest_sha256=identity["manifest_sha256"],
        data_receipt=data_binding, fit_report={"path": str(paths["fit"].resolve()), "sha256": data.file_sha256(paths["fit"])})
    checkpoint, manifest = training.save_semantic_checkpoint(runner, destination, new_infos)
    fresh = make_runner(metadata)
    loaded = training.load_semantic_checkpoint(fresh, checkpoint, contract=contract, seed=metadata["seed"])
    if training.parameter_hash(fresh.alg.actor) != report["actor_parameter_sha256_after"] or loaded[LEDGER] != new_infos[LEDGER]:
        raise RuntimeError("independent AUX reload changed actor or ledger")
    receipt = dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=data.file_sha256(checkpoint),
        manifest=str(manifest.resolve()), manifest_sha256=data.file_sha256(manifest),
        save_load_round_trip=True, independent_official_reload=True, latest_pointer_published=False,
        global_policy_decisions=loaded["global_policy_decisions"], ppo_updates=loaded["ppo_updates"], optimizer_steps=loaded["optimizer_steps"],
        accepted_auxiliary_updates=report["accepted_auxiliary_updates"], attempted_auxiliary_optimizer_steps=report["attempted_auxiliary_optimizer_steps"],
        added_policy_decisions=0, added_ppo_updates=0, added_ppo_optimizer_steps=0,
        fit_report=str(paths["fit"]), physical_success_claimed=False, teacher_deployed=False)
    training.write_json(paths["aux_publication"], receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()

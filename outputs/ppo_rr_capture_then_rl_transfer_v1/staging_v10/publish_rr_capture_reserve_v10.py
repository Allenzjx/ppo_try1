"""Zero-update v10 publication in the existing learned branch; no pointer promotion.

Run only after staging is installed, tested, reviewed and committed, and the
sole Isaac process has exited. Default is validation only, not publication.
Actual publication keeps the source CUDA device/visibility and full RNG.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess

from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import checkpoint_metadata, file_sha
from wlr50_clean.ppo.semantic_rr_capture_reserve_migration import (
    MIGRATION, COUNTERS, SOURCE_BRANCH, preserved_keys,
    build_rr_capture_reserve_migration, validate_rr_capture_reserve_migration,
    validate_v10_branch_receipt,
)
from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
from wlr50_clean.ppo.semantic_rr_capture_profile import RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_training import construct_semantic_runner, load_semantic_checkpoint, save_semantic_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-head",required=True)
    parser.add_argument("--publish",action="store_true")
    args=parser.parse_args()
    staging=Path(__file__).resolve().parent
    out=staging.parent
    root=out.parents[1]
    head=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if head!=args.expected_head:
        raise ValueError("explicit committed target HEAD differs")
    branch=out/"branches"/SOURCE_BRANCH
    source=branch/"checkpoints/history/checkpoint_aux_frontretention410_CP222592_01.pt"
    source_sha="8aaf255c118fa9199467b8e4016ad6d8ac235d28a31f25335423c45e93e5d86e"
    metadata=checkpoint_metadata(source)
    contract=runtime_contract(expected_head=head,semantic_version="v3",experiment_id="rr_capture_then_rl_transfer_v1")
    old=metadata["runtime_contract"]
    review={p:h for p,h in contract["files"].items() if old["files"].get(p)!=h}
    plan=build_rr_capture_reserve_migration(source,contract,expected_source_sha256=source_sha,
        expected_target_head=head,reviewed_code_sha256=review,
        reason="Remove only the three assist +25mm reserve-permission ceilings while preserving all physical/contact "
            "requirements and the existing53deg/45s budget. Preserve exact latest learned v9 CP222592 plus separately "
            "recorded32/32 front-retention AUX, all old state/lineage and v9 wheel. "
            "Same410 control-semantic migration, fresh rollout, zero learning and no pointer promotion.")
    step=metadata["global_policy_decisions"]
    stem=f"CP{step}_RR410_capture_reserve_v10_g{head[:12]}"
    target=branch/f"checkpoints/history/checkpoint_rr_capture_reserve_v10_step_{step:09d}_g{head[:12]}.pt"
    plan_path=staging/(stem+"_migration.json")
    receipt_path=staging/(stem+"_publication.json")
    if not args.publish:
        print(json.dumps(dict(validated_only=True,target_checkpoint=str(target),target_head=head,
            source_selection=plan["source_selection"],no_pointer_promotion=True)))
        return
    for path in (target,target.with_name(target.stem+"_manifest.json"),plan_path,receipt_path):
        if path.exists(): raise FileExistsError(path)
    pointers=[out/"checkpoints/checkpoint_last.pt",branch/"checkpoints/checkpoint_last.pt",
        out/"checkpoints/resume_state.json",branch/"checkpoints/resume_state.json"]
    pointer_hashes={str(p):file_sha(p) if p.exists() else None for p in pointers}
    with plan_path.open("x",encoding="utf-8") as stream:
        json.dump(plan,stream,indent=2,allow_nan=False)
    verified=validate_rr_capture_reserve_migration(source,contract,plan_path)
    device=metadata["runner_config"]["device"]
    def make():
        return construct_semantic_runner(_shape_env(410,device),seed=metadata["seed"],device=device,
            policy_version=RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner=make()
    infos=load_semantic_checkpoint(runner,source,contract=contract,seed=metadata["seed"],migration=verified)
    if infos.get(MIGRATION)!=verified: raise RuntimeError("v10 migration receipt missing")
    infos.update(runtime_contract=copy.deepcopy(contract),resume_migration=verified,old_rollout_inherited=False,
        migration_publication_context="zero_update_same410_v10_learned_branch_full_state_no_environment")
    saved,manifest=save_semantic_checkpoint(runner,target,infos)
    loaded=load_semantic_checkpoint(make(),saved,contract=contract,seed=metadata["seed"])
    if any(loaded[k]!=metadata[k] for k in preserved_keys(metadata)) or loaded[MIGRATION]!=verified:
        raise RuntimeError("official fresh reload changed full state or ancestry")
    for key in ("actor_parameter_sha256","critic_parameter_sha256","optimizer_state_sha256",
            "training_rng_state","runner_config","optimizer_learning_rate",*COUNTERS):
        if loaded[key]!=metadata[key]:
            raise RuntimeError("official fresh reload changed source field: "+key)
    validate_v10_branch_receipt(loaded,contract,metadata["checkpoint_output_routing"])
    after_hashes={str(p):file_sha(p) if p.exists() else None for p in pointers}
    if pointer_hashes!=after_hashes: raise RuntimeError("zero-update publication changed a checkpoint pointer")
    receipt=dict(checkpoint=str(saved),manifest=str(manifest),checkpoint_sha256=file_sha(saved),
        source_checkpoint=str(source),source_checkpoint_sha256=file_sha(source),target_git_commit=head,
        save_load_round_trip=True,exact_same410_source_state_preserved=True,
        **{k:loaded[k] for k in COUNTERS},rr_capture_transfer_branch_counts=loaded["rr_capture_transfer_branch_counts"],
        migration_added_policy_decisions=0,migration_added_ppo_updates=0,migration_added_optimizer_steps=0,
        migration_added_auxiliary_updates=0,old_rollout_inherited=False,no_pointer_promotion=True,
        checkpoint_output_branch=SOURCE_BRANCH,actual_effective_learning_rate=loaded["optimizer_learning_rate"],
        source_selection=plan["source_selection"],physical_evaluation="NOT_YET_EVALUATED",
        control_version="v10_progress_earned_existing_reserve",
        front_retention_auxiliary_counts=dict(accepted=32,attempted=32),source_aux_checkpoint_physically_evaluated=False)
    with receipt_path.open("x",encoding="utf-8") as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps(receipt,allow_nan=False))


if __name__=="__main__": main()

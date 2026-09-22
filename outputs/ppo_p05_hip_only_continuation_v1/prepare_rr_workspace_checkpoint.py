"""Explicit plan/publish driver. Requires selected real boundary checkpoint/SHA.

Use only after adopting/committing the reviewed runtime. --publish additionally
requires Isaac to have exited and original source CUDA visibility. No pointer
updates, default source, counter fabrication, or on-policy optimization.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

REASON = (
    "Narrow opt-in RR receiver-workspace preparation retirement after currently qualified "
    "legal crossing. Existing top_gap_min_m tolerance; current loss restores original role. "
    "Preserve controller predicates, nominal, FL assist, physical assets, all389 feature "
    "positions, policy kernel/caps/sigma and every actor/critic/Adam/LR/Identity/RNG value. "
    "Reward potential and observation index17 numerical meaning change explicitly; old "
    "values are not new-objective truth. Fresh rollout, actual independent counter origin, "
    "zero migration learning credit; preserve P05, feedback-v2 and prior AUX lineage."
)


def main():
    project=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--expected-source-sha256',required=True)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    from wlr50_clean.ppo.semantic_cli import PROJECT_ROOT,runtime_contract
    from wlr50_clean.ppo.semantic_migration import checkpoint_metadata,validate_migration_plan
    from wlr50_clean.ppo.semantic_rr_workspace_migration import build_rr_workspace_migration,publish_rr_workspace_checkpoint
    from wlr50_clean.ppo.semantic_training import write_json
    if project!=PROJECT_ROOT.resolve():raise RuntimeError('driver/runtime project mismatch')
    checkpoint=args.checkpoint.resolve(strict=True);metadata=checkpoint_metadata(checkpoint)
    if metadata['checkpoint_sha256']!=args.expected_source_sha256:
        raise RuntimeError('selected real source checkpoint SHA differs')
    head=subprocess.run(['git','-C',str(project),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    contract=runtime_contract(expected_head=head,semantic_version='v3',experiment_id='p05_hip_only_continuation_v1')
    old=metadata['runtime_contract']
    reviewed={path:sha for path,sha in contract['files'].items() if old['files'].get(path)!=sha}
    plan=build_rr_workspace_migration(checkpoint,contract,reason=REASON,reviewed_code_sha256=reviewed,project_root=project)
    tag=f"step_{metadata['global_policy_decisions']:09d}_g{head[:12]}"
    folder=project/'outputs/ppo_p05_hip_only_continuation_v1'/f'rr_postcross_workspace_v1_{tag}'
    folder.mkdir(parents=True,exist_ok=True)
    def write_once(path,value):
        if path.exists():
            if json.loads(path.read_text(encoding='utf8'))!=value:raise RuntimeError('existing immutable artifact differs: '+str(path))
        else:write_json(path,value)
    plan_path=folder/'migration_plan.json'
    write_once(plan_path,plan);write_once(folder/'runtime_contract.json',contract)
    verified=validate_migration_plan(checkpoint,contract,plan_path,project_root=project)
    destination=project/'outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history'/f'checkpoint_rr_postcross_workspace_v1_{tag}.pt'
    summary={'source_checkpoint':str(checkpoint),'source_checkpoint_sha256':metadata['checkpoint_sha256'],
        'actual_counter_origin':{key:metadata[key] for key in ('global_policy_decisions','ppo_updates','optimizer_steps')},
        'target_head':head,'plan':str(plan_path),'plan_sha256':verified['plan_sha256'],
        'destination':str(destination),'source_device':metadata['runner_config']['device'],
        'publication_requested':args.publish,'pointers_updated':False}
    if args.publish:
        if destination.exists() or destination.with_name(destination.stem+'_manifest.json').exists():
            raise FileExistsError('target exists; inspect/reload instead of republishing')
        import torch
        if torch.cuda.device_count()!=metadata['training_rng_state']['torch_cuda_device_count']:
            raise RuntimeError('original source CUDA RNG visibility is required')
        receipt=publish_rr_workspace_checkpoint(checkpoint,contract,plan_path,destination)
        write_once(folder/'publication_receipt.json',receipt);summary['publication']=receipt
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

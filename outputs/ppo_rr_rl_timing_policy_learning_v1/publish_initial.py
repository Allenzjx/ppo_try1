"""Explicit official source-device publication; no simulation, update or pointer."""
import argparse
import json
from pathlib import Path
import subprocess
from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_training import write_json
from wlr50_clean.ppo.semantic_rear_policy_timing_migration import (
    build_rear_policy_timing_migration,publish_rear_policy_timing_checkpoint,EXPERIMENT)

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/ppo_rr_capture_then_rl_transfer_v1/checkpoints/history/checkpoint_rr_signed_contact_v7_ancestor_step_000220544_g60abc00957c0.pt'

def main():
    p=argparse.ArgumentParser();p.add_argument('--publish',action='store_true');args=p.parse_args()
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    contract=runtime_contract(expected_head=head,semantic_version='v3',experiment_id=EXPERIMENT)
    directory=ROOT/'outputs'/('ppo_'+EXPERIMENT)
    plan=directory/('initial_migration_g'+head[:12]+'.json')
    value=build_rear_policy_timing_migration(SOURCE,contract,
        reason='Explicit verified front/RR ancestor selection; policy-owned rear and visible source timing; preserve full learned state; new419 fresh rollout')
    if not args.publish:
        print(json.dumps({'validated':True,'source':str(SOURCE),'target_head':head,'publishing':False}))
        return
    write_json(plan,value)
    target=directory/'checkpoints/history'/('checkpoint_initial_rear_policy_step_000220544_g'+head[:12]+'.pt')
    result=publish_rear_policy_timing_checkpoint(SOURCE,contract,plan,target)
    write_json(directory/('initial_publication_g'+head[:12]+'.json'),result)
    print(json.dumps(result))

if __name__=='__main__':main()

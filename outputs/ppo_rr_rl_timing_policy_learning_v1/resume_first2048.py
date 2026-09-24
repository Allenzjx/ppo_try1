"""Recover the first mixed course without repeating its real manual P01 block.

Run with base Python (stdlib only), never the env_isaaclab parent interpreter.
The production scheduler and all child training commands remain unchanged.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from wlr50_clean.ppo import semantic_curriculum as curriculum

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENT='rr_rl_timing_policy_learning_v1'
COURSES=ROOT/'runs'/('ppo_'+EXPERIMENT)/'curriculum'
FAILED=COURSES/'first2048_gfa4b98ed506e/curriculum.json'
ADOPTED=ROOT/'runs'/('ppo_'+EXPERIMENT)/'train/20260923T1908118345669Z_gfa4b98ed506e_1e77c2dd353745ebb86708c5095a34c9'
RECOVERED=COURSES/'first2048_recovered_gfa4b98ed506e'

def prepare(failed_path=FAILED,adopted=ADOPTED):
    failed=curriculum.read(failed_path)
    if (failed.get('lifecycle')!='FAILED' or failed.get('completed_blocks')!=[]
            or any(failed.get('actual_counts',{}).values())
            or failed.get('execute') is not True):
        raise ValueError('original course must remain failed with no credited block')
    configuration=Path(failed['configuration']);source=Path(failed['source_checkpoint'])
    if (curriculum.sha(configuration)!=failed['configuration_sha256']
            or curriculum.sha(source)!=failed['source_checkpoint_sha256']):
        raise ValueError('original bound configuration or immutable source changed')
    blocks=curriculum.plan_blocks(curriculum.read(configuration),2048)
    if blocks!=failed['planned_blocks']:
        raise ValueError('recovery must execute the unchanged bound three-block plan')
    evidence=curriculum.sealed_result(adopted,source,blocks[0],failed['expected_head'])
    if (evidence['lifecycle']!='SUCCEEDED' or evidence['actual_counts']!=dict(
            global_policy_decisions=512,ppo_updates=4,optimizer_steps=80)
            or evidence['unused_requested_decisions']!=0):
        raise ValueError('manual P01 block must first seal exactly512/4/80; partial work is not adopted')
    return failed,blocks,evidence

def execute_recovery(*,run_dir=RECOVERED):
    failed,blocks,evidence=prepare()
    run_dir=Path(run_dir).resolve()
    if not run_dir.is_relative_to(COURSES.resolve()) or run_dir.exists():
        raise ValueError('recovery requires a new independent curriculum directory')
    original_sha=curriculum.sha(FAILED);adopted_manifest_sha=curriculum.sha(ADOPTED/'run_manifest.json')
    calls=0
    def launcher(command,**options):
        nonlocal calls
        index=calls;calls+=1
        def arg(key):return command[command.index(key)+1]
        if index==0:
            if (arg('-FromPhase')!='P01' or arg('-Decisions')!='512'
                    or Path(arg('-Checkpoint')).resolve()!=Path(failed['source_checkpoint']).resolve()
                    or arg('-ExpectedHead')!=failed['expected_head']):
                raise ValueError('adoption hook received a different first block')
            curriculum.sealed_result(ADOPTED,Path(failed['source_checkpoint']),blocks[0],failed['expected_head'])
            curriculum.write(run_dir/'recovery_binding.json',dict(
                initial_failed_ledger=str(FAILED),initial_failed_ledger_sha256=original_sha,
                adopted_run=str(ADOPTED),adopted_run_manifest_sha256=adopted_manifest_sha,
                adopted_counts=evidence['actual_counts'],adoption_launched_physics=False,
                future_blocks=[row['from_phase'] for row in blocks[1:]],
                prior_P01_counts_in_course_are_not_newly_executed_by_this_invocation=True))
            curriculum.write(run_dir/'launch_00_adopted.json',dict(
                action='ADOPTED_ALREADY_EXECUTED_SEALED_RUN_NO_LAUNCH',run_dir=str(ADOPTED),
                run_manifest_sha256=adopted_manifest_sha,actual_counts=evidence['actual_counts'],
                newly_executed_decisions=0,optimizer_calls_by_adoption=0))
            return SimpleNamespace(stdout='ADOPTED already-sealed real P01 block\n'+str(ADOPTED),stderr='',returncode=0)
        if index not in (1,2) or arg('-FromPhase')!=blocks[index]['from_phase']:
            raise ValueError('recovery cannot repeat or add another training block')
        path=run_dir/f'launch_{index:02d}_process.json'
        try:
            result=subprocess.run(command,**options)
        except subprocess.CalledProcessError as error:
            curriculum.write(path,dict(action='ACTUAL_CHILD_PROCESS_FAILED',command=command,
                returncode=error.returncode,stdout=error.stdout,stderr=error.stderr))
            raise
        curriculum.write(path,dict(action='ACTUAL_CHILD_PROCESS_RETURNED',command=command,
            returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        return result
    try:
        return curriculum.run_curriculum(configuration_path=failed['configuration'],
            source=failed['source_checkpoint'],source_sha256=failed['source_checkpoint_sha256'],
            expected_head=failed['expected_head'],decisions=2048,run_dir=run_dir,seed=1001,
            execute=True,launcher=launcher)
    finally:
        if curriculum.sha(FAILED)!=original_sha:
            raise RuntimeError('original failed ledger unexpectedly changed')
        ledger_path=run_dir/'curriculum.json'
        if ledger_path.exists():
            ledger=curriculum.read(ledger_path)
            runs=[row['run_dir'] for row in ledger.get('completed_blocks',[])]
            if len(runs)!=len(set(runs)):
                raise RuntimeError('duplicate physical run credited in recovered course')
            ledger['recovery_provenance']=dict(initial_failed_ledger=str(FAILED),
                initial_failed_ledger_sha256=original_sha,adopted_run=str(ADOPTED),
                adopted_run_manifest_sha256=adopted_manifest_sha,
                adopted_counts=evidence['actual_counts'],adoption_launched_physics=False,
                adopted_run_counted_exactly_once=runs.count(str(ADOPTED))==1,
                course_counts_include_previously_executed_P01=True,
                this_invocation_new_counts={k:sum(row['actual_counts'][k] for row in ledger.get('completed_blocks',[])
                    if Path(row['run_dir']).resolve()!=ADOPTED.resolve()) for k in curriculum.COUNTERS})
            curriculum.write(ledger_path,ledger)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
    p.add_argument('--run-dir',type=Path,default=RECOVERED);args=p.parse_args()
    if not args.execute:
        try:
            _,_,evidence=prepare()
        except (FileNotFoundError,ValueError) as error:
            print(json.dumps(dict(ready=False,reason=str(error),execute=False)))
            return
        print(json.dumps(dict(ready=True,adopted_run=str(ADOPTED),adopted_counts=evidence['actual_counts'],execute=False)))
        return
    result=execute_recovery(run_dir=args.run_dir)
    print(json.dumps(dict(lifecycle=result['lifecycle'],actual_counts=result['actual_counts'])))

if __name__=='__main__':main()

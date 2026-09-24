"""Run a bounded, serial existing-CLI course; never simulate or optimize here."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

from wlr50_clean.ppo.semantic_curriculum import plan_blocks, read, sha, write

ROOT = Path(__file__).resolve().parents[2]
NAMESPACE = 'ppo_rr_rl_timing_policy_learning_v1'
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')
PWSH = Path(r'C:\Users\kskzz\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--decisions', type=int, default=1024,
        help='Whole-update learner budget consumed by the existing curriculum plan')
    parser.add_argument('--order', choices=('rear_first','natural_first'), default='rear_first')
    args = parser.parse_args()
    config_path = ROOT/'configs'/NAMESPACE/'curriculum_plan.json'
    config = read(config_path)
    blocks = plan_blocks(config, args.decisions)
    # Allocation always comes from the bound configuration. Explicit ordering
    # does not turn a zero-credit prefix into front optimization coverage.
    order = ('P07','P10','P01') if args.order=='rear_first' else ('P01','P07','P10')
    blocks.sort(key=lambda b: order.index(b['from_phase']))
    source = args.source.resolve(strict=True)
    source_meta = read(source.with_name(source.stem+'_manifest.json'))
    if (sha(source) != args.source_sha256
            or source_meta['runtime_contract']['source_git_commit'] != args.expected_head
            or source_meta['runtime_contract']['selected_configuration']['curriculum_plan.json']['sha256'] != sha(config_path)
            or not source_meta.get('save_load_round_trip')):
        raise ValueError('exact published source/configuration binding required')
    base = (ROOT/'outputs'/NAMESPACE).resolve()
    branch_root = base/'branches'/args.branch
    if (branch_root.resolve() != branch_root or branch_root.parent != base/'branches'
            or not PWSH.is_file()):
        raise ValueError('explicit safe branch and existing PowerShell7 required')
    args.run_dir.mkdir(parents=True, exist_ok=False)
    ledger = dict(schema='wlr50_clean.rear_recapture_branch_course.v1',
        source_checkpoint=str(source), source_sha256=args.source_sha256,
        source_counters={k:source_meta[k] for k in COUNTERS},
        runtime_head=args.expected_head, output_branch=args.branch,
        configuration=str(config_path), configuration_sha256=sha(config_path),
        requested_policy_decisions=args.decisions,
        planned_blocks=blocks, execution_order=list(order),
        order_reason=('rear recapture first; natural P01 learning last' if args.order=='rear_first'
            else 'long natural P01 learner trajectory first; requested entry is not actual phase coverage'),
        completed_blocks=[], actual_counts=dict.fromkeys(COUNTERS, 0),
        actual_phase_samples={f'P{i:02d}':0 for i in range(1,14)},
        main_pointer_promotion=False, lifecycle='RUNNING')
    ledger_path=args.run_dir/'curriculum.json'
    write(ledger_path,ledger)
    try:
        for block in blocks:
            if (args.run_dir/'stop_before_next_block.request.json').exists():
                ledger['lifecycle']='STOPPED_BETWEEN_SEALED_BLOCKS'
                break
            old=read(source.with_name(source.stem+'_manifest.json'))
            command=[str(PWSH),'-NoProfile','-File',str(ROOT/'scripts/run_semantic_ppo.ps1'),
                '-Command','train','-ExpectedHead',args.expected_head,'-SemanticVersion','v3',
                '-ExperimentId','rr_rl_timing_policy_learning_v1','-Stage',block['stage'],
                '-Decisions',str(block['decisions']),'-FromPhase',block['from_phase'],
                '-PrefixSource',block['prefix_source'],'-Checkpoint',str(source),
                '-CheckpointOutputBranch',args.branch,'-Seed','1001','-NumEnvs','1',
                '-CheckpointIntervalUpdates','1']
            ledger['active_block']=block;ledger['active_command']=command
            write(ledger_path,ledger)
            child=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
            run=Path(child.stdout.strip().splitlines()[-1]).resolve(strict=True)
            if not run.is_relative_to((ROOT/'runs'/NAMESPACE).resolve()):
                raise ValueError('child run outside experiment')
            manifest=read(run/'run_manifest.json'); result=manifest['result']; request=manifest['arguments']
            if (manifest['lifecycle'] not in ('SUCCEEDED','STOPPED_AT_VERIFIED_UPDATE_BOUNDARY')
                    or request['checkpoint_output_branch']!=args.branch
                    or Path(request['checkpoint']).resolve()!=source or request['from_phase']!=block['from_phase']
                    or request['prefix_source']!=block['prefix_source'] or request['decisions']!=block['decisions']
                    or manifest['runtime_contract']['source_git_commit']!=args.expected_head):
                raise ValueError('child must seal exact requested complete block')
            updates=[json.loads(s) for s in (run/'optimizer_updates.jsonl').read_text().splitlines() if s]
            completed={row['ppo_update'] for row in updates}
            actual=result['actual_policy_decisions']
            if (type(actual) is not int or not 0 < actual <= block['decisions'] or actual % 128
                    or manifest['lifecycle']=='SUCCEEDED' and actual!=block['decisions']):
                raise ValueError('child did not seal a valid full-update learner budget')
            expected=actual//128
            if (len(completed)!=expected or len(updates)!=expected
                    or any(row['optimizer_steps']!=20 for row in updates)
                    or result['ppo_updates_this_run']!=expected
                    or result['optimizer_steps_this_run']!=20*expected):
                raise ValueError('completed real optimizer counts differ')
            phases=Counter();audited=set()
            with (run/'advantage_audit.jsonl').open() as stream:
                for line in stream:
                    row=json.loads(line)
                    if row['ppo_update_intended'] not in completed:continue
                    if row['ppo_update_intended'] in audited or row['teacher_prefix_samples_included']:
                        raise ValueError('duplicate audit or prefix credited to PPO')
                    audited.add(row['ppo_update_intended'])
                    phases.update({p:v['sample_count'] for p,v in row['by_request_phase'].items()})
            if audited!=completed or sum(phases.values())!=actual:
                raise ValueError('actual optimized stage coverage mismatch')
            target=Path(result['checkpoints'][-1]['checkpoint']).resolve(strict=True)
            if not target.is_relative_to(branch_root/'checkpoints/history'):
                raise ValueError('checkpoint escaped isolated branch')
            new=read(target.with_name(target.stem+'_manifest.json'))
            delta=dict(zip(COUNTERS,(actual,expected,20*expected)))
            if (not new['save_load_round_trip'] or new['checkpoint_sha256']!=sha(target)
                    or any(new[k]-old[k]!=v for k,v in delta.items())):
                raise ValueError('saved checkpoint counts or integrity mismatch')
            evidence=dict(run_dir=str(run),from_phase=block['from_phase'],prefix_source=block['prefix_source'],
                checkpoint=str(target),checkpoint_sha256=sha(target),actual_counts=delta,
                actual_phase_samples=dict(phases),prefix_policy_credit=False,
                lifecycle=manifest['lifecycle'],unused_requested_decisions=block['decisions']-actual,
                task_success_not_inferred_from_budget_completion=True)
            ledger['completed_blocks'].append(evidence)
            for k,v in delta.items():ledger['actual_counts'][k]+=v
            for k,v in phases.items():ledger['actual_phase_samples'][k]+=v
            source=target;ledger['latest_checkpoint']=str(source)
            write(ledger_path,ledger)
            print(json.dumps(evidence),flush=True)
            if manifest['lifecycle']=='STOPPED_AT_VERIFIED_UPDATE_BOUNDARY':
                ledger['lifecycle']='STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'
                break
        else:ledger['lifecycle']='SUCCEEDED'
    except BaseException as exc:
        ledger['lifecycle']='FAILED';ledger['error']=str(exc)
        if isinstance(exc,subprocess.CalledProcessError):
            ledger['child_stdout_tail']=(exc.stdout or '')[-2000:]
            ledger['child_stderr_tail']=(exc.stderr or '')[-4000:]
        raise
    finally:write(ledger_path,ledger)
    print(json.dumps(dict(lifecycle=ledger['lifecycle'],actual_counts=ledger['actual_counts'])))


if __name__=='__main__':main()

"""Small, real update-boundary curriculum over the existing single-Isaac CLI.

No simulation, Torch, or optimizer is implemented here. Each child is the normal
audited train command; the next child receives its sealed immutable checkpoint.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

EXPERIMENT = 'rr_rl_timing_policy_learning_v1'
SCHEMA = 'wlr50_clean.rear_timing_curriculum.v1'
ROOT = Path(__file__).resolve().parents[3]
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.pending')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
    temporary.replace(path)

def plan_blocks(configuration, decisions):
    """Allocate complete updates, largest-remainder rounding, explicit order."""
    if (configuration.get('schema') != SCHEMA or configuration.get('experiment_id') != EXPERIMENT
            or configuration.get('rollout_length') != 128 or configuration.get('num_envs') != 1
            or type(decisions) is not int or decisions < 384 or decisions % 128):
        raise ValueError('curriculum requires N1 whole128 updates and at least one per entry')
    rows = configuration.get('entries', [])
    if len(rows) != 3 or [r.get('from_phase') for r in rows] != ['P01', 'P07', 'P10']:
        raise ValueError('explicit P01/P07/P10 course required')
    weights = [r.get('weight') for r in rows]
    if any(type(w) not in (int, float) or not 0 < w <= 1 for w in weights) or abs(sum(weights)-1) > 1e-12:
        raise ValueError('positive curriculum weights must sum to one')
    updates = decisions // 128
    exact = [updates*w for w in weights]
    allocation = [int(x) for x in exact]
    for i in sorted(range(3), key=lambda i: (-(exact[i]-allocation[i]), i))[:updates-sum(allocation)]:
        allocation[i] += 1
    if min(allocation) < 1:
        raise ValueError('requested budget does not cover every configured entry')
    blocks = []
    for i, (row, count) in enumerate(zip(rows, allocation)):
        phase = row['from_phase']
        prefix = row.get('prefix_source')
        if prefix not in ('checkpoint_policy', 'successful_nominal') and not (phase == 'P01' and prefix == 'frozen_fsm'):
            raise ValueError('suffix requires a real continuous nominal or frozen-policy prefix')
        if phase == 'P01' and prefix != 'frozen_fsm':
            raise ValueError('natural P01 has no prefix')
        blocks.append(dict(index=i, from_phase=phase, prefix_source=prefix,
            stage='full_episode' if phase == 'P01' else 'phase_suffix',
            decisions=count*128, planned_updates=count, prefix_policy_credit=False))
    return blocks

def sealed_result(run, source, block, expected_head):
    """Count only optimizer-completed rows, never a pending pre-update audit."""
    run, source = Path(run), Path(source)
    manifest = read(run/'run_manifest.json')
    if manifest.get('lifecycle') not in ('SUCCEEDED', 'STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'):
        raise ValueError('curriculum child is not sealed at a verified update boundary')
    args, result = manifest['arguments'], manifest['result']
    if (manifest['runtime_contract']['source_git_commit'] != expected_head
            or args['experiment_id'] != EXPERIMENT or args['from_phase'] != block['from_phase']
            or args['prefix_source'] != block['prefix_source'] or args['decisions'] != block['decisions']
            or Path(args['checkpoint']).resolve() != source.resolve()):
        raise ValueError('child invocation differs from the selected curriculum block')
    old = read(source.with_name(source.stem+'_manifest.json'))
    updates = [json.loads(line) for line in (run/'optimizer_updates.jsonl').read_text().splitlines() if line]
    completed = {r['ppo_update'] for r in updates}
    actual = result['actual_policy_decisions']
    if (len(completed) != len(updates) or actual != 128*len(updates)
            or actual > block['decisions'] or result['ppo_updates_this_run'] != len(updates)
            or result['optimizer_steps_this_run'] != 20*len(updates)
            or any(r['optimizer_steps'] != 20 for r in updates)):
        raise ValueError('completed update credit is inconsistent')
    phases = Counter({f'P{i:02d}':0 for i in range(1,14)})
    audited = set()
    with (run/'advantage_audit.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['ppo_update_intended'] not in completed:
                continue
            if row['ppo_update_intended'] in audited or row['teacher_prefix_samples_included']:
                raise ValueError('duplicate completed audit or credited teacher prefix')
            audited.add(row['ppo_update_intended'])
            phases.update({p:v['sample_count'] for p,v in row['by_request_phase'].items()})
    if audited != completed or sum(phases.values()) != actual:
        raise ValueError('actual optimized phase sample count differs')
    target = Path(result['checkpoints'][-1]['checkpoint']).resolve(strict=True)
    namespace = ROOT/'outputs'/('ppo_'+EXPERIMENT)/'checkpoints/history'
    if not target.is_relative_to(namespace.resolve()):
        raise ValueError('child checkpoint escaped the new experiment')
    latest = read(target.with_name(target.stem+'_manifest.json'))
    delta = dict(zip(COUNTERS,(actual,len(updates),20*len(updates))))
    if (not latest.get('save_load_round_trip') or latest['checkpoint_sha256'] != sha(target)
            or any(latest[k]-old[k] != v for k,v in delta.items())):
        raise ValueError('sealed checkpoint/counter/roundtrip mismatch')
    return dict(run_dir=str(run), lifecycle=manifest['lifecycle'], actual_counts=delta,
        actual_phase_samples=dict(phases), checkpoint=str(target), checkpoint_sha256=sha(target),
        prefix_policy_credit=False, physical_entry_success_not_inferred_from_requested_phase=True,
        unused_requested_decisions=block['decisions']-actual)

def run_curriculum(*, configuration_path, source, source_sha256, expected_head, decisions,
                   run_dir, seed=1001, execute=False, launcher=None):
    configuration_path, source, run_dir = map(Path, (configuration_path,source,run_dir))
    configuration = read(configuration_path)
    blocks = plan_blocks(configuration, decisions)
    if sha(source) != source_sha256:
        raise ValueError('explicit initial checkpoint SHA mismatch')
    if not source.resolve().is_relative_to((ROOT/'outputs'/('ppo_'+EXPERIMENT)/'checkpoints/history').resolve()):
        raise ValueError('curriculum starts from a published current-experiment checkpoint')
    source_meta = read(source.with_name(source.stem+'_manifest.json'))
    if (source_meta['runtime_contract']['source_git_commit'] != expected_head
            or source_meta['runtime_contract']['experiment_id'] != EXPERIMENT
            or not source_meta.get('rear_policy_timing_migration')):
        raise ValueError('curriculum source lacks the explicit current runtime migration')
    config_entry = source_meta['runtime_contract']['selected_configuration'].get(configuration_path.name,{})
    if config_entry.get('sha256') != sha(configuration_path):
        raise ValueError('curriculum configuration was not bound by the source runtime')
    run_dir.mkdir(parents=True, exist_ok=False)
    ledger = dict(schema=SCHEMA, expected_head=expected_head, configuration=str(configuration_path.resolve()),
        configuration_sha256=sha(configuration_path), source_checkpoint=str(source.resolve()),
        source_checkpoint_sha256=source_sha256, planned_blocks=blocks, completed_blocks=[],
        execute=execute, lifecycle='PLANNED', actual_counts=dict.fromkeys(COUNTERS,0),
        actual_phase_samples={f'P{i:02d}':0 for i in range(1,14)})
    write(run_dir/'curriculum.json',ledger)
    if not execute:
        return ledger
    launcher = launcher or subprocess.run
    try:
        for block in blocks:
            if (run_dir/'stop_before_next_block.request.json').exists():
                ledger['lifecycle']='STOPPED_BETWEEN_SEALED_BLOCKS'
                break
            command=['powershell','-NoProfile','-File',str(ROOT/'scripts/run_semantic_ppo.ps1'),
                '-Command','train','-ExpectedHead',expected_head,'-SemanticVersion','v3',
                '-ExperimentId',EXPERIMENT,'-Stage',block['stage'],'-Decisions',str(block['decisions']),
                '-FromPhase',block['from_phase'],'-PrefixSource',block['prefix_source'],
                '-Checkpoint',str(source.resolve()),'-Seed',str(seed),'-NumEnvs','1',
                '-CheckpointIntervalUpdates','1']
            ledger['lifecycle']='RUNNING'; ledger['active_block']=block; ledger['active_command']=command
            write(run_dir/'curriculum.json',ledger)
            child=launcher(command,check=True,capture_output=True,text=True)
            run=Path(child.stdout.strip().splitlines()[-1])
            evidence=sealed_result(run,source,block,expected_head)
            ledger['completed_blocks'].append(evidence)
            for key,value in evidence['actual_counts'].items():ledger['actual_counts'][key]+=value
            for key,value in evidence['actual_phase_samples'].items():ledger['actual_phase_samples'][key]+=value
            source=Path(evidence['checkpoint']);ledger['latest_checkpoint']=str(source)
            write(run_dir/'curriculum.json',ledger)
            if evidence['lifecycle']!='SUCCEEDED':
                ledger['lifecycle']='STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'
                break
        else:
            ledger['lifecycle']='SUCCEEDED'
    except BaseException as error:
        ledger['lifecycle']='FAILED';ledger['error']=str(error)
        raise
    finally:
        write(run_dir/'curriculum.json',ledger)
    return ledger

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',type=Path,default=ROOT/'configs'/('ppo_'+EXPERIMENT)/'curriculum_plan.json')
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--decisions',type=int,default=2048)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=1001)
    parser.add_argument('--execute',action='store_true')
    args=vars(parser.parse_args());args['configuration_path']=args.pop('configuration')
    result=run_curriculum(**args)
    print(json.dumps({'lifecycle':result['lifecycle'],'actual_counts':result['actual_counts']}))

if __name__=='__main__':main()

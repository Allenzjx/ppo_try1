"""Resume only the two unrun blocks of one sealed interrupted course.

Base-Python/stdlib orchestrator. No simulation, Torch, checkpoint mutation,
optimizer, old-ledger rewrite, or automatic retry is implemented here.
Default is read-only inspection; --execute explicitly launches normal CLI.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from wlr50_clean.ppo.semantic_curriculum import COUNTERS, plan_blocks, read, sha, write

EXPERIMENT = 'rr_rl_timing_policy_learning_v1'
NAMESPACE = 'ppo_'+EXPERIMENT
HEAD = '49eb23163a6e20bc56301dbafb59b137ecebce66'
BRANCH = 'ancestor220544_recapture_v2'
BASE = ROOT/'outputs'/NAMESPACE
BRANCH_ROOT = BASE/'branches'/BRANCH
COURSES = ROOT/'runs'/NAMESPACE/'curriculum'
ORIGINAL = COURSES/'cooperative2048_g49eb23163a6e/curriculum.json'
DEFAULT_RUN = COURSES/'cooperative2048_continued_after_video_g49eb23163a6e'
PHASES = tuple(f'P{i:02d}' for i in range(1,14))


def require(value, message):
    if not value:
        raise ValueError(message)


def command_argument(command, key):
    require(command.count(key) == 1, 'one explicit '+key+' required')
    return command[command.index(key)+1]


def remaining_plan(prior, configuration):
    planned = plan_blocks(configuration, 2048)
    planned.sort(key=lambda b: ('P07','P10','P01').index(b['from_phase']))
    require(prior.get('lifecycle') == 'STOPPED_BETWEEN_SEALED_BLOCKS'
        and prior.get('runtime_head') == HEAD and prior.get('output_branch') == BRANCH
        and prior.get('planned_blocks') == planned
        and prior.get('requested_policy_decisions') == 2048
        and prior.get('main_pointer_promotion') is False,
        'exact stopped 2048 cooperative course required')
    completed = prior.get('completed_blocks', [])
    require(len(completed) == 1 and completed[0].get('from_phase') == 'P07'
        and completed[0].get('prefix_source') == 'successful_nominal'
        and completed[0].get('lifecycle') == 'SUCCEEDED'
        and completed[0].get('prefix_policy_credit') is False
        and completed[0].get('unused_requested_decisions') == 0
        and completed[0].get('actual_counts') == dict(zip(COUNTERS,(384,3,60)))
        and prior.get('actual_counts') == completed[0]['actual_counts'],
        'exactly one already executed P07 384/3/60 block must precede continuation')
    require([(b['from_phase'],b['decisions']) for b in planned[1:]] == [('P10',384),('P01',1280)],
        'remaining plan must be P10 384 then natural P01 1280')
    return planned[1:]


def checkpoint_info(path, expected_sha, head, config_sha):
    path = Path(path).resolve(strict=True)
    require(path.parent == (BRANCH_ROOT/'checkpoints/history').resolve(), 'checkpoint escaped branch history')
    digest = sha(path)
    require(digest == expected_sha, 'explicit checkpoint SHA differs')
    sidecar = path.with_name(path.stem+'_manifest.json')
    metadata = read(sidecar)  # Never print/copy its large inherited manifests.
    contract = metadata['runtime_contract']; route = metadata.get('checkpoint_output_routing') or {}
    require(metadata.get('save_load_round_trip') is True and metadata.get('checkpoint_sha256') == digest
        and contract.get('source_git_commit') == head and contract.get('experiment_id') == EXPERIMENT
        and contract['selected_configuration']['curriculum_plan.json']['sha256'] == config_sha
        and route.get('branch') == BRANCH and Path(route['output_root']).resolve() == BRANCH_ROOT.resolve()
        and route.get('main_latest_pointer_promotion') is False,
        'source/runtime/configuration/roundtrip/branch binding failed')
    counters = {key:metadata[key] for key in COUNTERS}
    require(all(type(x) is int and x >= 0 for x in counters.values()), 'invalid saved counters')
    return dict(checkpoint=str(path), checkpoint_sha256=digest,
        manifest=str(sidecar), manifest_sha256=sha(sidecar), counters=counters,
        save_load_round_trip=True, output_branch=BRANCH)


def verify_branch_pointer(info):
    pointer = read(BRANCH_ROOT/'checkpoints/checkpoint_last_pointer.json')
    require(Path(pointer['checkpoint']).resolve() == Path(info['checkpoint'])
        and pointer['checkpoint_sha256'] == info['checkpoint_sha256']
        and Path(pointer['manifest']).resolve() == Path(info['manifest'])
        and pointer['manifest_sha256'] == info['manifest_sha256'],
        'branch pointer is not the exact last sealed checkpoint; do not roll it back')


def sealed_block(run, source_info, block, head, config_sha):
    run = Path(run).resolve(strict=True)
    require(run.parent == (ROOT/'runs'/NAMESPACE/'train').resolve(), 'foreign child run')
    manifest_path = run/'run_manifest.json'
    manifest = read(manifest_path); args = manifest['arguments']; result = manifest['result']
    require(manifest.get('lifecycle') in ('SUCCEEDED','STOPPED_AT_VERIFIED_UPDATE_BOUNDARY')
        and manifest['runtime_contract']['source_git_commit'] == head
        and args['experiment_id'] == EXPERIMENT and args['checkpoint_output_branch'] == BRANCH
        and Path(args['checkpoint']).resolve() == Path(source_info['checkpoint'])
        and args['stage'] == block['stage'] and args['from_phase'] == block['from_phase']
        and args['prefix_source'] == block['prefix_source'] and args['decisions'] == block['decisions']
        and args['seed'] == 1001 and args['num_envs'] == 1
        and args['teacher_offset_decisions'] == 0,
        'sealed child differs from the exact requested branch course block')
    with (run/'optimizer_updates.jsonl').open(encoding='utf-8') as stream:
        updates = [json.loads(line) for line in stream if line.strip()]
    completed = {u['ppo_update'] for u in updates}; actual = result['actual_policy_decisions']
    require(type(actual) is int and actual > 0 and actual == 128*len(updates)
        and actual <= block['decisions'] and len(completed) == len(updates)
        and completed == set(range(source_info['counters']['ppo_updates']+1,
                                  source_info['counters']['ppo_updates']+1+len(updates)))
        and all(u['optimizer_steps'] == 20 for u in updates)
        and result['ppo_updates_this_run'] == len(updates)
        and result['optimizer_steps_this_run'] == 20*len(updates)
        and (manifest['lifecycle'] != 'SUCCEEDED' or actual == block['decisions']),
        'only complete actual optimizer updates receive credit')
    phases = Counter(dict.fromkeys(PHASES,0)); audited = set()
    with (run/'advantage_audit.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line); update = row['ppo_update_intended']
            if update not in completed: continue
            require(update not in audited and not row['teacher_prefix_samples_included'],
                    'duplicate completed audit or prefix learning credit')
            audited.add(update)
            amounts = {phase:value['sample_count'] for phase,value in row['by_request_phase'].items()}
            require(set(amounts) <= set(PHASES) and all(type(n) is int and n >= 0 for n in amounts.values()),
                    'invalid actual optimized phase distribution')
            phases.update(amounts)
    require(audited == completed and sum(phases.values()) == actual, 'phase evidence/count mismatch')
    target = Path(result['checkpoints'][-1]['checkpoint']).resolve(strict=True)
    # Bind actual output, never predict the next CP name or reuse a stale pointer.
    info = checkpoint_info(target, sha(target), head, config_sha)
    delta = dict(zip(COUNTERS,(actual,len(updates),20*len(updates))))
    require(all(info['counters'][k]-source_info['counters'][k] == v for k,v in delta.items()),
            'actual source/target counter delta mismatch')
    return dict(run_dir=str(run), run_manifest_sha256=sha(manifest_path),
        from_phase=block['from_phase'], prefix_source=block['prefix_source'],
        actual_counts=delta, actual_phase_samples=dict(phases), checkpoint=info,
        lifecycle=manifest['lifecycle'], unused_requested_decisions=block['decisions']-actual,
        prefix_policy_credit=False, requested_entry_is_not_actual_phase_coverage=True,
        task_success_not_inferred_from_budget_completion=True)


def prepare(args):
    prior = read(ORIGINAL); configuration = Path(prior['configuration']).resolve(strict=True)
    require(configuration == (ROOT/'configs'/NAMESPACE/'curriculum_plan.json').resolve()
        and sha(configuration) == prior['configuration_sha256'] and args.expected_head == HEAD,
        'unchanged frozen curriculum and explicit current HEAD required')
    blocks = remaining_plan(prior, read(configuration))
    stop = ORIGINAL.parent/'stop_before_next_block.request.json'
    request = read(stop)
    require(request.get('request_scope') == 'between_complete_blocks_only'
        and request.get('remaining_training_is_not_cancelled') is True,
        'original pause was not the explicit between-block video pause')
    source = checkpoint_info(args.source, args.source_sha256, HEAD, prior['configuration_sha256'])
    require(Path(prior['latest_checkpoint']).resolve() == Path(source['checkpoint'])
        and prior['completed_blocks'][0]['checkpoint_sha256'] == source['checkpoint_sha256'],
        'continuation must start from the actual P07 last saved checkpoint')
    verify_branch_pointer(source)
    first_source = checkpoint_info(prior['source_checkpoint'], prior['source_sha256'], HEAD,
                                  prior['configuration_sha256'])
    first = sealed_block(prior['completed_blocks'][0]['run_dir'], first_source,
                         prior['planned_blocks'][0], HEAD, prior['configuration_sha256'])
    require(first['checkpoint'] == source and first['actual_counts'] == prior['actual_counts']
        and first['actual_phase_samples'] == prior['actual_phase_samples'],
        'original P07 adopted evidence differs from its sealed actual run')
    template = prior['active_command']
    for key, value in [('-Command','train'),('-ExpectedHead',HEAD),('-ExperimentId',EXPERIMENT),
        ('-CheckpointOutputBranch',BRANCH),('-Seed','1001'),('-NumEnvs','1'),('-CheckpointIntervalUpdates','1')]:
        require(command_argument(template,key) == value, 'previous actual CLI template differs: '+key)
    require(Path(template[0]).is_file() and Path(command_argument(template,'-File')).resolve()
        == (ROOT/'scripts/run_semantic_ppo.ps1').resolve(), 'normal PowerShell7 wrapper required')
    return prior, blocks, source, first, template, {
        str(ORIGINAL): sha(ORIGINAL), str(stop):sha(stop),
        str(BASE/'checkpoints/checkpoint_last_pointer.json'):sha(BASE/'checkpoints/checkpoint_last_pointer.json')}


def block_command(template, block, source):
    result = list(template)
    for key,value in [('-Stage',block['stage']),('-Decisions',str(block['decisions'])),
        ('-FromPhase',block['from_phase']),('-PrefixSource',block['prefix_source']),
        ('-Checkpoint',source['checkpoint'])]:
        command_argument(result,key)
        result[result.index(key)+1] = value
    return result


def run(args):
    require('isaac' not in sys.executable.lower(), 'use base Python; child wrapper owns the sole Isaac resource')
    prior, blocks, source, adopted, template, preserved = prepare(args)
    if not args.execute:
        return dict(ready=True, execute=False, source=source, remaining_blocks=blocks,
                    previously_executed_counts=adopted['actual_counts'], this_invocation_new_counts=dict.fromkeys(COUNTERS,0))
    destination = args.run_dir.resolve()
    require(destination.parent == COURSES.resolve() and not destination.exists(), 'new independent continuation directory required')
    destination.mkdir()
    ledger = dict(schema='wlr50_clean.cooperative_course_continuation.v1', lifecycle='RUNNING',
        original_ledger=str(ORIGINAL), preserved_files=preserved, runtime_head=HEAD, output_branch=BRANCH,
        source_checkpoint=source, planned_remaining_blocks=blocks,
        previous_completed_block=adopted, adoption_launches=0, completed_blocks=[],
        this_invocation_new_counts=dict.fromkeys(COUNTERS,0),
        actual_counts=dict.fromkeys(COUNTERS,0),
        actual_phase_samples=dict.fromkeys(PHASES,0),
        course_counts_including_previous=dict(adopted['actual_counts']),
        course_phase_samples_including_previous=dict(adopted['actual_phase_samples']),
        main_pointer_promotion=False, requested_remaining_decisions=1664)
    path = destination/'curriculum.json'; write(path,ledger)
    try:
        for index,block in enumerate(blocks):
            if (destination/'stop_before_next_block.request.json').exists():
                ledger['lifecycle']='STOPPED_BETWEEN_SEALED_BLOCKS'; break
            verify_branch_pointer(source)
            command = block_command(template,block,source)
            ledger.update(active_block=block,active_command=command); write(path,ledger)
            try:
                # The unchanged wrapper enforces clean/pinned runtime and the
                # existing OS-level single-Isaac lock/busy guard.
                child = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,check=True)
            except subprocess.CalledProcessError as exc:
                write(destination/f'launch_{index:02d}.json',dict(command=command,
                    returncode=exc.returncode,stdout=exc.stdout,stderr=exc.stderr)); raise
            write(destination/f'launch_{index:02d}.json',dict(command=command,
                returncode=child.returncode,stdout=child.stdout,stderr=child.stderr))
            run_dir = Path(child.stdout.strip().splitlines()[-1])
            evidence = sealed_block(run_dir, source, block, HEAD, prior['configuration_sha256'])
            require(str(run_dir.resolve()) != adopted['run_dir']
                and all(evidence['run_dir'] != row['run_dir'] for row in ledger['completed_blocks']),
                'one actual run cannot receive duplicate course credit')
            verify_branch_pointer(evidence['checkpoint'])
            ledger['completed_blocks'].append(evidence)
            for key,value in evidence['actual_counts'].items():
                ledger['this_invocation_new_counts'][key] += value
                ledger['actual_counts'][key] += value
                ledger['course_counts_including_previous'][key] += value
            for phase,value in evidence['actual_phase_samples'].items():
                ledger['actual_phase_samples'][phase] += value
                ledger['course_phase_samples_including_previous'][phase] += value
            source = evidence['checkpoint']; ledger['latest_checkpoint'] = source
            write(path,ledger); print(json.dumps(evidence),flush=True)
            if evidence['lifecycle'] != 'SUCCEEDED':
                ledger['lifecycle']='STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'; break
        else: ledger['lifecycle']='SUCCEEDED'
    except BaseException as exc:
        ledger['lifecycle']='FAILED'; ledger['error']=f'{type(exc).__name__}: {exc}'; raise
    finally:
        ledger['unconsumed_requested_decisions'] = 1664-ledger['actual_counts']['global_policy_decisions']
        ledger['preserved_files_unchanged'] = all(sha(Path(file)) == digest for file,digest in preserved.items())
        if not ledger['preserved_files_unchanged']:
            ledger['lifecycle']='FAILED'
            ledger['error']='original ledger/stop file/main pointer changed'
        write(path,ledger)
        require(ledger['preserved_files_unchanged'], 'original ledger/stop file/main pointer changed')
    return {k:ledger[k] for k in ('lifecycle','actual_counts','course_counts_including_previous','unconsumed_requested_decisions')}


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--run-dir',type=Path,default=DEFAULT_RUN)
    args=parser.parse_args(); print(json.dumps(run(args),indent=2))


if __name__=='__main__': main()

"""Read only the first accepted prefix and first credited policy row; no monitor."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    run = args.run.resolve(strict=True)
    manifest = json.loads((run/'run_manifest.started.json').read_text())
    arguments = manifest['arguments']
    prefix, digest = [], hashlib.sha256()
    with (run/'prefix_evidence.jsonl').open('rb') as stream:
        for _ in range(1805):
            line = stream.readline()
            if not line or not line.endswith(b'\n'): raise ValueError('first prefix not complete')
            digest.update(line)
            row = json.loads(line)
            prefix.append(row)
            if row['kind'] == 'policy_credit_start': break
        else: raise ValueError('bounded first-prefix scan exceeded declared limit')
    with (run/'residual_and_projection_audit.jsonl').open('rb') as stream:
        first_line = stream.readline()
        first = json.loads(first_line)
    steps = [r for r in prefix if r['kind'] == 'checkpoint_prefix_decision']
    result = next(r for r in prefix if r['kind'] == 'checkpoint_prefix_result')
    provenance = next(r for r in prefix if r['kind'] == 'checkpoint_prefix_start')['prefix_policy_provenance']
    start = prefix[-1]['start']
    cp = Path(provenance['checkpoint_path'])
    metadata = json.loads(cp.with_name(cp.stem+'_manifest.json').read_text())
    checkpoint_hash = hashlib.file_digest(cp.open('rb'), 'sha256').hexdigest()
    history = first['policy_request']['previous_raw_from_current_observation_full12']
    expected = [max(-20., min(20., v)) for v in steps[-1]['raw_policy_action_full12']]
    info = first['applied_audit']
    checks = {
        'actual_command_P04_checkpoint_policy': arguments['prefix_source'] == 'checkpoint_policy' and arguments['from_phase'] == 'P04',
        'training_seed_retained': arguments['seed'] == metadata['seed'] == 1001,
        'actual_source_is_CP178944': metadata['global_policy_decisions'] == provenance['source_global_policy_decisions'] == 178944,
        'checkpoint_path_and_sha_match': Path(arguments['checkpoint']).resolve() == cp.resolve() and checkpoint_hash == metadata['checkpoint_sha256'] == provenance['checkpoint_sha256'],
        'actor_hash_matches_checkpoint': metadata['actor_parameter_sha256'] == provenance['actor_parameter_sha256'] == provenance['frozen_actor_parameter_sha256'],
        'exact_policy_contract_matches_checkpoint': provenance['policy_contract'] == metadata['policy_contract'],
        'runtime_matches_checkpoint': provenance['source_runtime_content_sha256'] == metadata['runtime_contract']['runtime_content_sha256'] == manifest['runtime_contract']['runtime_content_sha256'],
        'frozen_independent_actor': provenance['frozen_for_entire_training_block'] is True and provenance['independent_parameter_and_buffer_storage_verified'] is True,
        'deterministic_prefix_not_nominal': provenance['inference'] == 'TensorDict policy+critic; stochastic_output=False; no projection' and all(any(v != 0 for v in r['raw_policy_action_full12']) for r in steps),
        'natural_P01_first_prefix_step': steps[0]['phase_id'] == 'P01' and steps[0]['physics_tick']-steps[0]['physics_ticks'] == 0,
        'accepted_without_fallback': result['accepted'] is True and result['miss'] is None and start['mode'] == 'checkpoint_policy_initialized_suffix',
        'actual_P04_credit_entry': start['actual_phase'] == start['requested_phase'] == 'P04' and start['requested_phase_still_active_at_credit'] is True,
        'all_prefix_rows_without_credit': all(r['policy_credit'] is False for r in prefix),
        'prefix_samples_excluded': info['prefix_checkpoint_policy_data_in_ppo_storage'] is False and first['global_policy_decision'] == metadata['global_policy_decisions']+1,
        'raw_history_continues_exactly': history == expected,
        'first_policy_action_contiguous_ticks': info['physics_tick']-info['physics_ticks'] == start['physics_tick'] == steps[-1]['physics_tick'],
        'all_prefix_steps_native_verified': all(r['actuator_target_effect_audit_summary']['all_ticks_verified'] for r in steps),
        'no_prefix_state_writes': all(r['no_in_episode_state_writes_verified'] is True for r in steps),
    }
    if not all(checks.values()): raise RuntimeError(checks)
    receipt = {'schema':'wlr50_clean.first_checkpoint_policy_prefix_audit.v1', 'run':str(run),
        'scope':'first_completed_prefix_and_first_policy_row_only',
        'source_prefix_raw_lines_sha256':digest.hexdigest(), 'prefix_rows_read':len(prefix),
        'first_policy_raw_line_sha256':hashlib.sha256(first_line).hexdigest(),
        'checks':checks, 'all_checks_passed':True,
        'checkpoint':str(cp), 'checkpoint_sha256':checkpoint_hash,
        'prefix_decisions':len(steps), 'prefix_physics_ticks':sum(r['physics_ticks'] for r in steps),
        'prefix_phase_decisions':dict(Counter(r['phase_id'] for r in steps)),
        'accepted_start':start, 'last_prefix_raw_full12':steps[-1]['raw_policy_action_full12'],
        'first_policy_previous_raw_full12':history,
        'first_policy_global_decision':first['global_policy_decision'],
        'first_policy_request_phase':info['phase_id'], 'first_policy_end_phase':info['end_phase_id'],
        'first_policy_start_tick':info['physics_tick']-info['physics_ticks'], 'first_policy_end_tick':info['physics_tick'],
        'prefix_optimizer_credit':0, 'P01_P02_quality_credit_from_prefix':0,
        'training_completion_or_later_coverage_claimed':False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({k:receipt[k] for k in ('all_checks_passed','prefix_decisions','prefix_physics_ticks',
        'prefix_phase_decisions','first_policy_global_decision','first_policy_request_phase','first_policy_end_phase',
        'first_policy_start_tick','first_policy_end_tick','prefix_optimizer_credit')},indent=2))


if __name__ == '__main__': main()

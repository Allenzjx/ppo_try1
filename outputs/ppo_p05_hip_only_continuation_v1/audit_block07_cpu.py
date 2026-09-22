"""Bounded CPU audit of sealed block07; no fit, GPU, runtime or checkpoint writes."""
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import types

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1156115839619Z_g5fd88852bf20_a9df5b5d15aa4500a30f12933b1c36fb'
SOURCE = OUT / 'checkpoints/history/checkpoint_aux_detP02_step_000211968_v2.pt'
SOURCE_SHA = '27bc7cbdd98775c4602057d7776dd3becc9eef9b42ec62f4691d2dae7bb06d55'
FINAL_SHA = '8e5961e3e78260f12124447bff72a7459b7323d2da8f83cc467a59ddc9290e54'


def main():
    # Reuse the unchanged bounded auditor. This run deliberately ended at a
    # verified update boundary, so extend only its lifecycle guard in memory.
    # Keep the actual lifecycle string in the result; never relabel it success.
    path = OUT / 'audit_block05_cpu.py'
    code = path.read_text(encoding='utf-8')
    old = "if manifest.get('lifecycle')!='SUCCEEDED':"
    assert code.count(old) == 1
    code = code.replace(old, "if manifest.get('lifecycle') not in ('SUCCEEDED', 'STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'):")
    audit = types.ModuleType('block07_reused_bounded_auditor')
    audit.__file__ = str(path)
    exec(compile(code, str(path), 'exec'), audit.__dict__)
    audit.RUN, audit.SOURCE, audit.SOURCE_SHA = RUN, SOURCE, SOURCE_SHA
    manifest = audit.read(RUN / 'training_manifest.json')
    assert manifest['lifecycle'] == 'STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'
    assert manifest['stop_after_update']['run_dir'] == str(RUN)
    request_path = Path(manifest['stop_after_update']['request_path'])
    assert audit.sha(request_path) == manifest['stop_after_update']['request_sha256']
    assert manifest['planned_requested_policy_decisions'] == 2048
    assert manifest['actual_policy_decisions'] == 1408
    assert manifest['unconsumed_requested_policy_decisions'] == 640
    original_lines = audit.lines
    extra = Counter()
    fr_per_episode = []

    def checked_lines(name):
        for row in original_lines(name):
            if name == 'residual_and_projection_audit.jsonl':
                applied = row['applied_audit']
                if applied['decision_count'] == 1:
                    fr_per_episode.append(Counter())
                ev = applied['semantic_task']['physical_evaluator']
                fr = ev['current_legs']['FR']
                legal_top = (fr['top_contact'] and fr['top_surface_contact']
                    and fr['contact_surface'] == 'TOP' and fr['within_top_xy']
                    and fr['within_lateral_span'] and not fr['ground_contact'])
                flags = {'FR_qualified_history': ev['history']['active_lift']['FR'],
                    'FR_crossed_history': ev['history']['front_edge_crossed']['FR'],
                    'FR_placed_history': ev['history']['placed']['FR'],
                    'FR_current_exact_legal_TOP': legal_top,
                    'FR_current_exact_legal_TOP_bearing_support': legal_top and fr['support'] and fr['bearing_verified'],
                    'FR_current_ground_contact': fr['ground_contact'], 'FR_current_AIR': fr['air']}
                extra.update({key: int(value) for key, value in flags.items()})
                fr_per_episode[-1].update({key: int(value) for key, value in flags.items()})
                ticks = applied['actuator_target_effect_audit_ticks']
                assert len(ticks) == applied['physics_ticks']
                for tick in ticks:
                    assert tick['verified']
                    extra['actual_native_verified_ticks'] += 1
                native = applied['actuator_target_effect_audit']
                assert native['actual_mapping_matches_dispatch'] and native['phase_mask_full12'] == [1] * 12
                extra['actual_all12_permission_endpoints'] += 1
                extra['observed_active_assist_inputs'] += row['policy_request']['capture_assist_observed_features'][0] != 0
                extra['valid_physical_endpoints'] += ev['valid'] is True
            yield row

    audit.lines = checked_lines
    captured = io.StringIO()
    with redirect_stdout(captured):
        audit.main()
    result = json.loads(captured.getvalue())
    assert result['checkpoint_sha256'] == FINAL_SHA
    assert result['new_counts'] == {'policy_decisions': 1408, 'ppo_updates': 11, 'optimizer_steps': 220}
    assert result['lifetime_counts'] == {'global_policy_decisions': 213376, 'ppo_updates': 1632, 'optimizer_steps': 32640}
    sm = audit.read(SOURCE.with_name(SOURCE.stem + '_manifest.json'))
    final = Path(result['checkpoint'])
    fm = audit.read(final.with_name(final.stem + '_manifest.json'))
    ledger = fm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert ledger == sm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert ledger['accepted_auxiliary_updates_total'] == ledger['attempted_auxiliary_optimizer_steps_total'] == 64
    assert [event['event_index'] for event in ledger['events']] == [1, 2]
    older = fm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert older == sm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert older['accepted_auxiliary_updates_total'] == 7 and older['attempted_auxiliary_optimizer_steps_total'] == 8
    assert extra['actual_native_verified_ticks'] == result['execution_counts']['physics_ticks']
    assert extra['actual_all12_permission_endpoints'] == 1408
    rng = fm['training_rng_state']
    source_rng = sm['training_rng_state']
    assert set(rng) == set(source_rng)
    assert rng['schema'] == source_rng['schema'] == 'wlr50_clean.training_rng_state.v1'
    assert rng['seed'] == source_rng['seed'] == fm['seed'] == sm['seed']
    assert rng['torch_cuda_device_count'] == source_rng['torch_cuda_device_count']
    assert len(rng['torch_cuda']) == rng['torch_cuda_device_count']
    assert all(rng[key] for key in ('python_random', 'numpy_random', 'torch_cpu'))
    # CPU audit verifies serialized/embedded RNG state, but does not restore CUDA
    # RNG or require it equal the source after real stochastic PPO learning.
    from wlr50_clean.ppo.semantic_training import state_hash
    result.update(schema='wlr50_clean.p05_capture_block07_sealed_audit.v1',
        stop_after_update=manifest['stop_after_update'], source_counts={key: sm[key] for key in audit.COUNTERS},
        execution_front_detail=dict(extra),
        front_rehearsal_AUX_totals={key: ledger[key] for key in ('accepted_auxiliary_updates_total', 'attempted_auxiliary_optimizer_steps_total')},
        front_rehearsal_AUX_full_ledger_exact_source_carry=True,
        training_RNG={'seed': rng['seed'], 'CUDA_device_count': rng['torch_cuda_device_count'],
            'source_sha256': state_hash(source_rng), 'final_sha256': state_hash(rng),
            'exact_embedded_sidecar_match': True, 'all_source_state_kinds_retained': True,
            'source_equal_required': False, 'CUDA_restore_performed_by_this_CPU_audit': False},
        unconsumed_640_decisions_receive_no_credit=True)
    for episode, front in zip(result['episodes'], fr_per_episode, strict=True):
        episode['FR_physical_endpoint_counts'] = dict(front)
        episode['first_uncompleted_task'] = episode['last_phase'] if not episode['full_task_success'] else None
        episode['ended_episode_not_task_success'] = episode['terminal'] and not episode['full_task_success']
    result['actual_full_task_success_episode_count'] = sum(e['full_task_success'] for e in result['episodes'])
    result['first_terminal_front_failure'] = next(({
        key: e[key] for key in ('episode_index', 'last_global_decision', 'last_phase', 'termination_reason', 'duration_s')}
        for e in result['episodes'] if e['terminal'] and not e['full_task_success']), None)
    (OUT / 'block07_training_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    phases = ', '.join(f'{key}={value}' for key, value in result['request_phase_counts'].items())
    episode_rows = '\n'.join(f"| {e['episode_index']} | {e['policy_decisions']} | {e['duration_s']:.6f} | {'ended' if e['terminal'] else 'partial'} | {e['last_phase']} | {e['termination_reason'] or 'sampling boundary; done remains false'} |" for e in result['episodes'])
    text = f'''# Block07 sealed real PPO audit

**Verified update-boundary stop**, not successful task completion. Actual source is AUX event2 checkpoint `checkpoint_aux_detP02_step_000211968_v2.pt` (SHA `{SOURCE_SHA}`). Final is `checkpoint_step_000213376.pt` (SHA `{FINAL_SHA}`).

- Actual new credit: **1408 decisions / 11 PPO updates / 220 Adam steps**; total **213376 / 1632 / 32640**. Planned 2048; **640 unconsumed decisions receive no credit**. Stop-request hash and run binding verified.
- Natural P01, no prefix. Input counts: {phases}. Sum 1408.
- {result['completed_episode_count']} ended episodes, {result['partial_episode_count']} nonterminal sampling-boundary partial episode; complete-task successes {result['actual_full_task_success_episode_count']}.
- {result['execution_counts']['physics_ticks']} actual ticks, all compact native records verified; all 1408 detailed endpoints map correctly with all 12 residual permissions. Assist-owned endpoints {result['execution_counts']['assist_owned_endpoints']}; active-assist inputs {extra['observed_active_assist_inputs']}. Compact tick records do not independently store masks/owners.
- FR endpoint history: qualified {extra['FR_qualified_history']}, crossed {extra['FR_crossed_history']}, placed {extra['FR_placed_history']}; actual legal TOP {extra['FR_current_exact_legal_TOP']}, verified TOP bearing/support {extra['FR_current_exact_legal_TOP_bearing_support']}. RR retirement learner predicate/consumed/effective-change counts: {result['RR_retirement_input_counts']}.
- All 11 updates have finite nonzero gradients and changed actor parameters; each of 12 Adam states advanced exactly 220. LR {result['effective_LR']}, Identity unchanged. Every raw12/mean/std/logp/reward/done and 389 input matches sealed storage; each real sample used 5 times. CPU logp maximum difference {result['independent_CPU_logp_max_error']:.9g}.
- Both front AUX events and the whole **64/64** ledger remain exactly equal to source; old **7/8** ledger and all three origins/migration dictionaries are preserved. This block adds **0 AUX**.
- Saved RNG retains Python/NumPy/CPU and {rng['torch_cuda_device_count']} CUDA states, correct seed, and exact embedded/sidecar equality. Real training advances RNG; source equality is not required. No GPU RNG restore was attempted in this CPU audit. Actual normal save/reload is recorded true.

## Episodes

| Episode | Credited decisions | Seconds | Status | First unfinished task | Result |
| --- | ---: | ---: | --- | --- | --- |
{episode_rows}

Evidence: `{RUN}`; sealed rollouts 1622–1632, matching optimizer and likelihood records, synchronous physical logs, actual source/final checkpoints. The final partial trajectory is retained with `done=false`, not converted into failure or success. Earlier local successes or AUX data do not substitute for current physical outcomes. No other runs were scanned.

Auditor reuses unchanged block05 checks, extending only its lifecycle guard in memory to accept the actual verified-boundary status. No production/frozen-helper edit, optimization, GPU, simulator or checkpoint write. CPU helper exits after report creation.
'''
    (OUT / 'block07_training_audit.md').write_text(text, encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'lifecycle': result['lifecycle'], 'new_counts': result['new_counts'],
        'lifetime_counts': result['lifetime_counts'], 'phases': result['request_phase_counts'],
        'episodes': [{key: e[key] for key in ('episode_index', 'policy_decisions', 'duration_s', 'terminal', 'last_phase', 'termination_reason')} for e in result['episodes']],
        'front': dict(extra), 'RR_retirement_inputs': result['RR_retirement_input_counts'],
        'RNG': result['training_RNG'], 'report': str(OUT / 'block07_training_audit.md')}, indent=2))


if __name__ == '__main__':
    main()

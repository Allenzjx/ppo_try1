"""Outputs-only sealed block06 audit, reusing the existing bounded auditor.

No simulator, GPU, fit, checkpoint publication or production/helper edits.
"""
from collections import Counter
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1041563379791Z_g5fd88852bf20_087b67f789494ca695e6d94de78c5623'
SOURCE = OUT / 'checkpoints/history/checkpoint_aux_frontrehearsal_step_000209920_v1.pt'
SOURCE_SHA = '3c45e7325210487431ebe8b5d1cbd92b480734e3d53829b04429869c4773e3e8'
FINAL_SHA = '5d0658331204c2bc79d71fd223582637e68dbb986ce4bf57596648f02c77d881'


def main():
    spec = importlib.util.spec_from_file_location('sealed_block_auditor', OUT / 'audit_block05_cpu.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.RUN, audit.SOURCE, audit.SOURCE_SHA = RUN, SOURCE, SOURCE_SHA
    original_lines = audit.lines
    extra = Counter()
    fr_per_episode = []

    def lines_with_front_and_per_tick_checks(name):
        for row in original_lines(name):
            if name == 'residual_and_projection_audit.jsonl':
                applied = row['applied_audit']
                if applied['decision_count'] == 1:
                    fr_per_episode.append(Counter())
                ev = applied['semantic_task']['physical_evaluator']
                fr = ev['current_legs']['FR']
                require_top = (fr['top_contact'] and fr['top_surface_contact']
                    and fr['contact_surface'] == 'TOP' and fr['within_top_xy']
                    and fr['within_lateral_span'] and not fr['ground_contact'])
                flags = {'FR_qualified_history': ev['history']['active_lift']['FR'],
                    'FR_crossed_history': ev['history']['front_edge_crossed']['FR'],
                    'FR_placed_history': ev['history']['placed']['FR'],
                    'FR_current_exact_legal_TOP': require_top,
                    'FR_current_exact_legal_TOP_bearing_support': require_top and fr['support'] and fr['bearing_verified'],
                    'FR_current_ground_contact': fr['ground_contact'],
                    'FR_current_AIR': fr['air']}
                extra.update({key: int(value) for key, value in flags.items()})
                fr_per_episode[-1].update({key: int(value) for key, value in flags.items()})
                ticks = applied['actuator_target_effect_audit_ticks']
                assert len(ticks) == applied['physics_ticks']
                for item in ticks:
                    assert item['verified']
                    extra['actual_native_verified_ticks'] += 1
                endpoint = applied['actuator_target_effect_audit']
                assert endpoint['actual_mapping_matches_dispatch'] and endpoint['phase_mask_full12'] == [1] * 12
                extra['actual_all12_permission_endpoints'] += 1
                extra['observed_active_assist_inputs'] += row['policy_request']['capture_assist_observed_features'][0] != 0
                extra['valid_physical_endpoints'] += ev['valid'] is True
            yield row

    audit.lines = lines_with_front_and_per_tick_checks
    captured = io.StringIO()
    with redirect_stdout(captured):
        audit.main()
    result = json.loads(captured.getvalue())
    assert result['checkpoint_sha256'] == FINAL_SHA
    assert result['new_counts'] == {'policy_decisions': 2048, 'ppo_updates': 16, 'optimizer_steps': 320}
    assert result['lifetime_counts'] == {'global_policy_decisions': 211968, 'ppo_updates': 1621, 'optimizer_steps': 32420}
    sm = audit.read(SOURCE.with_name(SOURCE.stem + '_manifest.json'))
    final = Path(result['checkpoint'])
    fm = audit.read(final.with_name(final.stem + '_manifest.json'))
    newer = fm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    older = fm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert newer == sm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert older == sm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert newer['accepted_auxiliary_updates_total'] == newer['attempted_auxiliary_optimizer_steps_total'] == 32
    assert older['accepted_auxiliary_updates_total'] == 7 and older['attempted_auxiliary_optimizer_steps_total'] == 8
    assert extra['actual_native_verified_ticks'] == result['execution_counts']['physics_ticks']
    assert extra['actual_all12_permission_endpoints'] == result['new_counts']['policy_decisions']
    result['schema'] = 'wlr50_clean.p05_capture_block06_sealed_audit.v1'
    result['execution_front_detail'] = dict(extra)
    result['front_rehearsal_AUX_totals'] = {key: newer[key] for key in (
        'accepted_auxiliary_updates_total', 'attempted_auxiliary_optimizer_steps_total')}
    result['front_rehearsal_AUX_full_ledger_exact_source_carry'] = True
    result['source_counts'] = {key: sm[key] for key in audit.COUNTERS}
    for episode, fr_counts in zip(result['episodes'], fr_per_episode, strict=True):
        episode['FR_physical_endpoint_counts'] = dict(fr_counts)
        episode['first_uncompleted_task'] = episode['last_phase'] if not episode['full_task_success'] else None
        episode['closed_episode_not_task_success'] = episode['terminal'] and not episode['full_task_success']
    result['notes_block06_vs_earlier'] = [
        'The 32/32 finite front rehearsal is prior to this run and remains exactly carried; this run adds zero AUX.',
        'Block05 before AUX also stayed in P01/P02. Compare explicit phase/contact counts, not SUCCEEDED lifecycle.',
        'Earlier block03 local FR/FL and rear capture are historical observations, not evidence this current checkpoint retained them.',
        'The separate live deterministic evaluation is not included in these stochastic on-policy training counts.']
    (OUT / 'block06_training_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    phase_line = ', '.join(f'{key}={value}' for key, value in result['request_phase_counts'].items())
    episodes_text = '\n'.join(
        f"| {e['episode_index']} | {e['policy_decisions']} | {e['duration_s']:.6f} | {'ended' if e['terminal'] else 'partial'} | {e['last_phase']} | {e['termination_reason'] or 'collection boundary, not terminal'} |"
        for e in result['episodes'])
    md = f'''# Block06 sealed genuine PPO audit

Training lifecycle **SUCCEEDED**, not physical task success. Source is the actual CP209920 AUX candidate (`{SOURCE_SHA}`), not pre-AUX CP209920. Final checkpoint is `checkpoint_step_000211968.pt`, SHA256 `{FINAL_SHA}`.

- Real new credit: **2048 decisions / 16 PPO updates / 320 Adam steps**; cumulative **211968 / 1621 / 32420**. Requested2048, unconsumed{result['unconsumed_requested_policy_decisions']}.
- Natural P01, no prefix. Credited input counts: {phase_line}. Sum2048.
- Actual physical ticks: {result['execution_counts']['physics_ticks']}; every compact native-tick record is verified. All2048 detailed decision endpoints pass mapping/all12-mask checks; assist-owned endpoints {result['execution_counts']['assist_owned_endpoints']}, active-assist input observations {extra['observed_active_assist_inputs']}. The compact per-tick records do not separately persist mask/assist fields, so no independent per-tick mask/owner count is invented.
- FR endpoint coverage: qualified history {extra['FR_qualified_history']}, crossed history {extra['FR_crossed_history']}, placed history {extra['FR_placed_history']}, current exact legal TOP {extra['FR_current_exact_legal_TOP']}, verified TOP bearing/support {extra['FR_current_exact_legal_TOP_bearing_support']}. History is not current contact.
- RR retirement learner inputs: predicate {result['RR_retirement_input_counts'].get('predicate_active',0)}, workspace share consumed {result['RR_retirement_input_counts'].get('workspace_share_consumed',0)}, nonzero actual potential change {result['RR_retirement_input_counts'].get('effective_potential_difference',0)}. Mere configured reward semantics is not physical coverage.
- Front rehearsal ledger **32 accepted/32 attempted** and earlier limited AUX **7/8** are fully equal to their source dictionaries. This PPO run adds **0 AUX**; all three original counter origins and historical migration dictionaries remain intact.
- Every update has finite nonzero gradients and changed actor parameters. All12 Adam states advance exactly320; LR {result['effective_LR']}, Identity preserved, final ordinary save/reload recorded true. Exact sealed raw12/mean/std/logp/reward/done and389 input checks pass; each original sample used5 times. CPU Normal logp max difference {result['independent_CPU_logp_max_error']:.9g}.

## Actual episodes

{result['completed_episode_count']} ended episodes and {result['partial_episode_count']} collection-boundary partial episode(s); ended does not mean successful.

| Episode | Decisions | Seconds | Status | First unfinished stage | Result |
| --- | ---: | ---: | --- | --- | --- |
{episodes_text}

This block's current front/rear physical coverage is the table and counts above; earlier block03 local front/rear successes cannot substitute for it. The prior AUX32/32 is a lineage fact, not proof that natural deterministic P02 has recovered. The separate current deterministic evaluation remains authoritative for that question.

Evidence: `{RUN}`;16 sealed rollouts1606–1621, matching likelihood ledgers/optimizer records, synchronous execution stream, completed episodes, actual source/final checkpoints. JSON records per-rollout, per-episode and all three origin details. CPU-only audit, no simulator/GPU/optimization/checkpoint writes or edits to bound v1 helpers. Helper exited after report creation.
'''
    (OUT / 'block06_training_audit.md').write_text(md, encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'new_counts': result['new_counts'], 'input_phases': result['request_phase_counts'],
        'episodes': [{key: e[key] for key in ('episode_index','policy_decisions','duration_s','terminal','last_phase','termination_reason')} for e in result['episodes']],
        'execution_front_detail': dict(extra), 'rr_retirement': result['RR_retirement_input_counts'],
        'new_AUX_added': 0, 'front_AUX_carry': '32/32 exact', 'old_AUX_carry': '7/8 exact',
        'report': str(OUT / 'block06_training_audit.md')}, indent=2))


if __name__ == '__main__':
    main()

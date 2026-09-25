"""Bounded stdlib evidence: first opportunity only; no model or simulation imports."""
import json
import math
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
snapshot = json.loads((OUT / 'train_gain10_first_opportunity_snapshot.json').read_text(encoding='utf-8'))
source = Path(snapshot['sources']['decisions']['path'])
boundary = snapshot['sources']['decisions']['snapshot_bytes']
rows = []
prefix_count = 0
with source.open('rb') as stream:
    while stream.tell() < boundary:
        start = stream.tell()
        line = stream.readline(boundary - start)
        if not line.endswith(b'\n'):
            break
        row = json.loads(line)
        if row['kind'] == 'frozen_prior_prefix':
            assert row['PPO_credit'] == 0 and not rows
            prefix_count += 1
        else:
            assert row['kind'] == 'activated_on_policy' and row['PPO_credit'] == 1
            row['_byte_range'] = [start, stream.tell()]
            rows.append(row)
            if row['step_info']['local_task_success']:
                break
    read_end = stream.tell()
assert len(rows) == 143 and prefix_count == 998
metrics = [r['step_info']['rr_capture_local']['metrics'] for r in rows]
top = [n for n, m in enumerate(metrics) if m['current_top_contact']]
assert top == list(range(134, 143))
assert all(m['current_attempt_capture_eligible'] and not m['ground_contact'] for m in metrics)
assert all(metrics[n]['current_top_bearing'] and metrics[n]['consecutive_top_samples'] == 2 + 8 * (n - 134) for n in top)
last = rows[-1]['step_info']
assert last['local_task_success'] and last['local_reward']['terminated'] and not last['full_task_success']
assert all(not r['step_info']['local_reward']['terminated'] for r in rows[:-1])

def compact(index):
    row = rows[index]
    info = row['step_info']
    local = info['rr_capture_local']
    audit = info['actuator_target_effect_audit']
    policy = row['policy_request']
    return {
        'episode_row': index, 'global_decision': row['global_decision'], 'byte_range': row['_byte_range'],
        'phase': [info['phase_id'], info['end_phase_id']], 'metrics': local['metrics'], 'hold_s': local['hold_elapsed_s'],
        'source_rr_deg': info['nominal_action_full12'][6:8], 'mapped_N_rr_deg': audit['native_drive_target_full12'][6:8],
        'generic_bias_rr_deg': audit['controller_drive_bias_full12'][6:8],
        'conditional_mean_rr': policy['conditional_mean_full12'][6:8], 'selected_raw_rr': policy['selected_raw_full12'][6:8],
        'raw_local_head_rr': policy['local_raw_mean_delta_full12'][6:8],
        'applied_local_raw_mean_rr': policy['applied_local_raw_mean_delta_full12'][6:8],
        'coordinate_gain_rr': policy['local_mean_coordinate_gain_full12'][6:8],
        'requested_residual_rr_deg': audit['policy_headroom_evidence']['requested_policy_residual_full12'][6:8],
        'effective_residual_rr_deg': audit['policy_headroom_evidence']['effective_policy_residual_full12'][6:8],
        'FINAL_rr_deg': info['actual_drive_target_full12'][6:8], 'old_logp': row['old_logp'][0],
        'tracking_servo_names': audit['tracking_reference_evidence'].get('tracking_servo_names'),
        'deferred_source': info['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2'],
    }

late = [v for r in rows for v in r['step_info']['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']['layers']]
pending = [v for r in rows for v in r['step_info']['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']['p12_pending_start']]
assert late and all(not v['pending_event_consumed'] and v['deferred_source_tracking_servo_names'] == [] for v in late)
assert pending and all(v['status'] == 'local_RR_capture_holding_new_P12_start' and not v['rl_current_swing'] for v in pending)
assert all(r['policy_request']['selected_raw_full12'] == r['step_info']['raw_policy_action_full12']
           and r['step_info']['actuator_target_effect_audit']['all12_policy_channels_unmodified_at_actuator']
           and r['policy_request']['history_kernel_applications'] == 1
           and r['policy_request']['sampling_draws'] == 1
           and r['policy_request']['extra_random_draws'] == 0
           and len(r['observation']) == 448
           and math.isfinite(r['old_logp'][0])
           and r['old_logp'][0] == r['policy_request']['selected_raw_log_probability'] for r in rows)
history = last['semantic_task']['history']
rr_events = [e for e in history['lift_attempt_events'] if e.get('leg') == 'RR']
assert not any('ground' in str(e).lower() and e['physics_tick'] >= 6995 for e in rr_events)
descent_begin = top[0]
while descent_begin > 0 and metrics[descent_begin]['gap_m'] < metrics[descent_begin - 1]['gap_m'] - 1e-6:
    descent_begin -= 1
assert descent_begin == 129
selected = [compact(n) for n in (0, 120, 129, 131, 133, 134, 135, 136, 142)]
report = {
    'schema': 'wlr50_clean.gain10_first_success_readonly.v1', 'source': str(source),
    'snapshot_byte_boundary': boundary, 'actual_read_end': read_end,
    'scope': 'First episode only, through terminal9128; no later prefix read. Decision endpoints plus stored native observer counters/history, not an independent120Hz rescan.',
    'mode': 'stochastic training, function-equivalent CP228864/AUX64 coordinate migration; before update8',
    'counts': {'prefix_credit0': prefix_count, 'actual_PPO_rows': len(rows), 'new_optimizer_updates_at_snapshot': snapshot['completed_updates_observed'],
               'phase_rows': dict(Counter(r['step_info']['phase_id'] for r in rows)), 'new_AUX_steps': 0},
    'local_success': True, 'full_task_success': False, 'terminal': last['local_reward'],
    'activation': rows[0]['before_local'], 'events': history['event_ticks'], 'RR_attempt_events': rr_events,
    'same_attempt': {'all143_current_eligible': True, 'GROUND_endpoints': 0, 'GROUND_revocations_after_qualified': 0,
                    'TOP_endpoint_ticks': [metrics[n]['tick'] for n in top],
                    'consecutive_TOP_counts': [metrics[n]['consecutive_top_samples'] for n in top],
                    'native_TOP_start_inferred_from_counter': 9063, 'last_native_TOP_tick': 9128,
                    'hold_s': last['rr_capture_local']['hold_elapsed_s'], 'last_bearing_force_n': metrics[-1]['bearing_force_n'],
                    'last_load_fraction': metrics[-1]['load_fraction'], 'load_fraction_valid': metrics[-1]['load_fraction_valid']},
    'last_strict_descent': {'start_endpoint_index': descent_begin, 'start_tick': metrics[descent_begin]['tick'],
                           'end_tick': metrics[top[0]]['tick'], 'action_rows': list(range(descent_begin + 1, top[0] + 1)),
                           'gap_mm': [m['gap_m'] * 1000 for m in metrics[descent_begin:top[0]+1]],
                           'not_all143_a_success_trajectory_label': True},
    'source_rules': {'late_five_channels_unconsumed_all_rows': True, 'carrier_tracking_inherited_empty': True,
                     'P12_new_unload_pending_all_receipts': True, 'all12_student_raw_unchanged': True,
                     'HISTORY_once': True, 'no_phase_change_terminal_before_hold': True},
    'selected_rows': selected,
    'limits': ['Success precedes update8; not gain10 optimizer benefit, deterministic ability, or fullRR/RL traversal.',
               'Contact/hold confirmed through stored native counters plus actualforce; no independent raw120Hz rescan.',
               'Actual movement reflects sampled policy, prior/HISTORY, source/mapper, generictracking and physicalcoupling; no singleaxis causal attribution.',
               'No AUXfit, teacherlabel, modelchange, productionchange or physics run performed by this analysis.'],
}
(OUT / 'train_gain10_first_success_readonly.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
def pair(v):
    return '[' + ', '.join(f'{x:+.3f}' for x in v) + ']'
lines = [
    '# gain10 首次机会：随机 RR capture+hold 事实', '',
    f'固定快照 {boundary:,} bytes，仅读至第一回合终点 byte {read_end:,}。998 前缀 credit0；143 个真实 PPO 样本（P09=135、P10=1、P11=1、P12=6）。这是与0ff功能等价迁移后的 CP228864/AUX64 随机采样，发生在 update8 之前：不是 gain10 更新收益、确定性成功或完整 RR/RL 越障。', '',
    '## 同次真实接触与保持', '',
    'RR qualified6995 → crossed/gate7979 → placed9064。首次TOP端点9064已有2个连续native样本，随后每8tick计数增加8，至9128为66；据记录计数可推断TOP9063–9128连续保持。hold=0.541667s，末11.394116N、load_fraction=0.408725（valid=true）、eligible=true。143端点无GROUND/无bearing丢失，历史无此次qualified之后GROUND撤资格。没有另行扫描原始120Hz日志。', '',
    '仅9128/76.066667s提交 RR_CAPTURE_HOLD_SUCCESS；此前P09→P10→P11→P12均不终止。local=true、full=false。', '',
    '## 最后一次下降及接续（RR顺序 hip,knee，度）', '',
    '| tick / phase | gap mm / hold s | source / mappedN / generic bias | mean / selected raw | FINAL / actual |',
    '|---|---|---|---|---|',
]
for x in selected:
    m = x['metrics']
    lines.append(f"| {m['tick']} / {'→'.join(x['phase'])} | {m['gap_m']*1000:+.3f} / {x['hold_s']:.3f} | {pair(x['source_rr_deg'])} / {pair(x['mapped_N_rr_deg'])} / {pair(x['generic_bias_rr_deg'])} | {pair(x['conditional_mean_rr'])} / {pair(x['selected_raw_rr'])} | {pair(x['FINAL_rr_deg'])} / {pair(m['actual_rr_hip_knee_deg'])} |")
lines += ['',
    '早先8952的1.363mm近台面并没有接触，随后回升至9024的22.122mm。末次严格端点下降为9024→9032→9040→9048→9056→9064：22.122→21.123→11.970→6.610→4.689→−0.520mm，共5次决策/0.333333s；不是整143拍持续下降，也不能把143拍都当成功动作。', '',
    '触地前这5拍 sourceRR[−6.9,−37.8]、mappedN[−8.15,−39.05]、generic RR bias[0,0]固定，实际 hip 逐步转负；knee并非每拍正向，最后三拍才明显展开。首次TOP9064仍是同一P09源。9072才执行合法P10 knee：source−29.3、mapped−32.1、generic+0.75，FINAL从−35.136到−25.136；9080 mapped knee−27.2继续接续。因此初接触不是P10源动作事先造成，后续保持也不能全部归于raw policy。', '',
    'P09 late五通道全143行未消费，carrier从previous_sample继承空tracking，没有偷偷重新启动延迟full12跟踪。P12收到的全部pending-start记录仍为local_RR_capture_holding_new_P12_start，rl_current_swing=false；没有启动新的RL强卸载。合法P10 RR knee与P11接续仍可执行。所选逐拍source/mapper、headroom与tracking名称保存在JSON。', '',
    '所有143行保留448观测、原始sample及其old logp，selected==issued，12通道不被隐藏覆盖，一次采样/一次HISTORY。主任务只得到随机局部捕获保持证据；没有新增AUX拟合、标签或控制修复。',
]
(OUT / 'train_gain10_first_success_readonly.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps({'output': str(OUT/'train_gain10_first_success_readonly.json'), 'counts': report['counts'], 'same_attempt': report['same_attempt'], 'last_descent': report['last_strict_descent']}, ensure_ascii=False))

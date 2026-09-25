"""Stdlib, fixed-byte first opportunity only; never follows the active writer."""
from pathlib import Path
from collections import Counter
import json

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/ppo_rr_capture_first_cp225280_v1/train_after_aux64_fresh512_0ff03ea/decisions.jsonl'
BOUNDARY=60748586  # captured with open/seek(END)/tell, not directory stat
rows=[]; prefix=0; next_prefix=None
with SOURCE.open('rb') as stream:
    while stream.tell()<BOUNDARY:
        line=stream.readline(BOUNDARY-stream.tell())
        if not line.endswith(b'\n'): break
        row=json.loads(line)
        if row.get('kind')=='frozen_prior_prefix':
            assert row['PPO_credit']==0
            if rows:
                next_prefix=row['local']['metrics']; break
            prefix+=1
        elif row.get('kind')=='activated_on_policy':
            assert row['PPO_credit']==1
            rows.append(row)
        else: raise ValueError('unexpected row kind')
    read_end=stream.tell()
assert next_prefix is not None and len(rows)==423
last=rows[-1]['step_info']; semantic=last['semantic_task']
assert last['local_reward']['terminated'] and not last['local_task_success']

def summary(index):
    row=rows[index]; info=row['step_info']; local=info['rr_capture_local']; metric=local['metrics']
    audit=info['actuator_target_effect_audit']; task=info['semantic_task']; physical=task['physical_evaluator']
    policy=row['policy_request']; legs=physical['current_legs']; deferred=task['nominal_provider_diagnostics']['rr_local_deferred_late_v2']
    context=physical.get('transfer_roles',{}).get('RR',{}).get('transfer_direction_context',{})
    bounds=physical.get('body_traversal_geometry',{})
    return dict(index=index,global_decision=row['global_decision'],tick=metric['tick'],time_s=metric['time_s'],
        phase=[info['phase_id'],info['end_phase_id']],RR=metric,hold_s=local['hold_elapsed_s'],
        RR_front_mm=legs['RR']['front_distance_m']*1000,
        source_full12=info['nominal_action_full12'],mapped_N_full12=audit['native_drive_target_full12'],
        generic_bias_full12=audit['controller_drive_bias_full12'],
        policy_conditional_mean_full12=policy['conditional_mean_full12'],
        policy_local_raw_mean_full12=policy['local_raw_mean_delta_full12'],
        policy_sample_raw_full12=policy['selected_raw_full12'],old_logp=row['old_logp'][0],
        residual_request_full12=audit['policy_headroom_evidence']['requested_policy_residual_full12'],
        effective_residual_full12=audit['policy_headroom_evidence']['effective_policy_residual_full12'],
        FINAL_full12=info['actual_drive_target_full12'],
        actual_wheel_canonical_rad_s=physical['measured_wheel_velocity_rad_s'],
        body_base_x_relative_front_m=physical['goal_features']['body_forward_m'],
        body_world_displacement_m=context.get('body_world_displacement_m'),
        body_displacement_reference_tick=context.get('reference_tick'),
        mass_weighted_COM_world_m=context.get('mass_weighted_com_position_w_m'),
        body_bounds_world_m={k:bounds.get(k) for k in ('minimum_w_m','maximum_w_m')},
        support={leg:{k:legs[leg].get(k) for k in ('air','ground_contact','top_surface_contact','support','bearing_force_n','load_fraction_valid')} for leg in ('FL','FR','RL','RR')},
        tracking=audit['tracking_reference_evidence'].get('tracking_servo_names'),
        deferred_layers=deferred['layers'],p12_pending_start=deferred['p12_pending_start'])

table=[summary(i) for i in (80,82,83,84,85,86,94,422)]
metrics=[r['step_info']['rr_capture_local']['metrics'] for r in rows]
ground=[m for m in metrics if m['ground_contact']]
revocations=[e for e in semantic['history']['lift_attempt_events']
             if e.get('leg')=='RR' and e.get('event')=='current_lift_revoked_ground' and e['physics_tick']>=7979]
max_hold=max(max(r['before_local']['hold_elapsed_s'],r['step_info']['rr_capture_local']['hold_elapsed_s']) for r in rows)
evidence=dict(schema='wlr50_clean.after_aux64_first_opportunity_snapshot.v1',source=str(SOURCE),
    snapshot_byte_boundary=BOUNDARY,first_episode_plus_next_prefix_read_end=read_end,
    counts=dict(first_opportunity_new_PPO_rows=len(rows),prefix_PPO_credit0=prefix,
        global_decision_first=rows[0]['global_decision'],global_decision_last=rows[-1]['global_decision'],
        completed_PPO_updates_in_first_opportunity=0,remaining_rows_to_first512_update=89,
        phase_rows=dict(Counter(r['step_info']['phase_id'] for r in rows))),
    terminal=dict(tick=metrics[-1]['tick'],time_s=metrics[-1]['time_s'],phase=metrics[-1]['phase_id'],
        reason=last['termination_reason'],source=semantic['termination_source'],local_success=last['local_task_success'],
        full_success=last['full_task_success'],local_reward=last['local_reward'],last_metrics=metrics[-1]),
    TOP_endpoint_ticks=[m['tick'] for m in metrics if m['current_top_contact']],
    max_stored_native_hold_s=max_hold,max_stored_consecutive_TOP_samples=max(m['consecutive_top_samples'] for m in metrics),
    exact_all_native_tick_max_hold_s=None,
    hold_scope='Stored native-observer values at decision endpoints/before_local; not independently logged each120Hz observer tick. Two TOP samples at8664 imply contact began8663; next endpoint8672 is not bearing. Exact subdecision maximum unavailable; do not call8.33ms the exact all-tick maximum.',
    first_ground_endpoint=ground[0] if ground else None,
    RR_ground_revocation_native_history_events=revocations,
    revoked_endpoint_count=sum(not m['current_attempt_capture_eligible'] for m in metrics),
    ground_endpoint_count=len(ground),next_prefix=next_prefix,
    invariants=dict(selected_raw_equals_issued=all(r['policy_request']['selected_raw_full12']==r['step_info']['raw_policy_action_full12'] for r in rows),
        all12_unmodified=all(r['step_info']['actuator_target_effect_audit']['all12_policy_channels_unmodified_at_actuator'] for r in rows),
        pending_late_unconsumed=all(not layer['pending_event_consumed'] for r in rows for layer in r['step_info']['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']['layers']),
        local_gate_stays_active=all(r['step_info']['rr_capture_local']['active'] for r in rows)),
    selected_rows=table,
    limits=['This is stochastic post-AUX sampling before its first fresh PPO optimizer update, not DET/video/full success.',
        'SOURCE/mappedN/bias/request/FINAL are same saved decision receipts; they do not isolate floating-body physical causality.',
        'Wheel mean/raw are dimensionless Gaussian values; source/FINAL/actual wheels are canonical rad/s in FL,FR,RL,RR order.',
        'body_base_x_relative_front is actual base x minus front; world COM is distinct. No endpoint full base XYZ is invented.'])
stem='train_after_aux64_first_opportunity_readonly'
target=OUT/(stem+'.json')
with target.open('x',encoding='utf-8') as f: json.dump(evidence,f,indent=2,allow_nan=False)
def arr(values): return '['+', '.join(f'{v:+.3f}' for v in values)+']'
lines=['# AUX64 后第一真实机会：未形成保持','',
    '固定字节快照：60,748,586 bytes；只扫描首机会并在下一条 credit0 前缀处停止。423 个新 PPO-credit1 样本，998 个前缀 credit0；首机会未满512，未执行本块 optimizer，尚余89个样本。',
    '', '终止：11364 tick / 94.700s / P10，INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED；local/full success 均 false。',
    '', '唯一 TOP 决策端点8664 / 72.200s，22.527N，stored native hold=0.008333s、连续TOP samples=2。已保存 before/end 值的最大hold也是0.008333s，不是独立120Hz全区间最大：精确子拍最大未保存，N/A。8663是由连续计数推得的起点；8672端点已无承载。',
    '', '| tick / s | phase | RR source / mappedN | request / bias | FINAL / actual hip,knee deg | gap / front mm | TOP / force / hold |',
    '|---|---|---|---|---|---|---|']
for r in table:
    m=r['RR']; lines.append(f"| {r['tick']} / {r['time_s']:.3f} | {'→'.join(r['phase'])} | {arr(r['source_full12'][6:8])} / {arr(r['mapped_N_full12'][6:8])} | {arr(r['residual_request_full12'][6:8])} / {arr(r['generic_bias_full12'][6:8])} | {arr(r['FINAL_full12'][6:8])} / {arr(m['actual_rr_hip_knee_deg'])} | {m['gap_m']*1000:+.3f} / {r['RR_front_mm']:+.3f} | {m['current_top_contact']} / {m['bearing_force_n']:.3f} / {r['hold_s']:.6f} |")
lines += ['', '首个已见差异：8664→8672失载时仍在合法XY。P10 source knee−37.8→−29.3、mappedN−39.05→−32.1，generic tracking bias +0.75；policy膝请求反而19.679→17.299。FINAL膝−19.371→−14.051（+5.320°展开），实际−24.476→−20.907。该拍不是膝重新内折造成的旧模式；P10原动作、采样/跟踪与全身反馈同时变化，不能据此单轴归因。下一端点8680才失去合法XY。',
    '', 'P09五通道late始终未消费；本机会仅到P10，不能称P12 source强卸载。FL/FR/RL source保持不等于其policy不动。',
    '', '## 同一窗口四轮与body证据', '', '轮顺序FL,FR,RL,RR；mean/raw无量纲，N/FINAL/actual为canonical rad/s。', '',
    '| tick | N | policy conditional mean / sampled raw | FINAL / actual | base x−front m / world CoM m |', '|---|---|---|---|---|']
for r in table[2:6]:
    lines.append(f"| {r['tick']} | {arr(r['source_full12'][8:12])} | {arr(r['policy_conditional_mean_full12'][8:12])} / {arr(r['policy_sample_raw_full12'][8:12])} | {arr(r['FINAL_full12'][8:12])} / {arr(r['actual_wheel_canonical_rad_s'])} | {r['body_base_x_relative_front_m']:+.6f} / {arr(r['mass_weighted_COM_world_m'])} |")
lines += ['', '明确stop使N四轮0后，policy仍令FL最终速度持续负向；不是mask。真实轮速已在同表区分，不把轮端位移当自转。完整全12 source/mean/raw/request/FINAL、支撑力、world body bounds与固定短窗body displacement见JSON，未虚构base完整XYZ。',
    '', 'RR确实掉回GROUND：native history `semantic_task.history.lift_attempt_events` 首个 `current_lift_revoked_ground`=8737 / 72.808333s；首个决策GROUND端点8744 / 72.866667s，current_attempt_capture_eligible=false。历史cross/placed保留，local gate持续active；未沿用它们冒充当前资格。后续ground撤资格见JSON；本机会无第二TOP端点或local_success。',
    '', '第一未完成条件仍是当前合格TOP承载连续0.5s。此次短TOP是随机采样进展，不是确定性成功或新PPO更新收益。']
with (OUT/(stem+'.md')).open('x',encoding='utf-8') as f: f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(output=str(target),counts=evidence['counts'],terminal=evidence['terminal']['reason'],TOP_ticks=evidence['TOP_endpoint_ticks'],max_stored_hold_s=max_hold)))

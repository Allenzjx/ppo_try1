"""Narrow stdlib summary; native boolean counts already verified once below."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/ppo_rr_capture_first_cp225280_v1/eval_CP228864_aux64_DET_0ff03ea/source'
manifest=json.loads((SOURCE/'source_manifest.json').read_text())
parent=json.loads((SOURCE.parent/'run_manifest.json').read_text())
assert parent['lifecycle']=='COMPLETE' and parent['result']==manifest
assert manifest['mode']=='DETERMINISTIC_COMPOSITE_POLICY' and not manifest['diagnostic_intervention']
assert manifest['PPO_updates']==0 and manifest['error'] is None
base=json.loads((OUT/'CP228864_aux64_DET_sealed_actualmetrics.json').read_text())
rows=[json.loads(line) for line in (SOURCE/'video_policy_decisions.jsonl').open('rb')]
active=[r for r in rows if r['policy_request']['capture_active'] is True]
assert len(active)==370 and len(rows)==1368
entry=base['current']['gate_entry']; final_entry=entry['entry_final_rr_deg']; actual_entry=entry['rr']['actual_rr_deg']
def summary(r):
    info=r['step_info']; m=info['rr_capture_local']['metrics']; a=info['actuator_target_effect_audit']; p=r['policy_request']
    return dict(tick=m['tick'],time_s=m['time_s'],RR=m,
        source_full12=info['nominal_action_full12'],mapped_N_full12=a['native_drive_target_full12'],
        generic_bias_full12=a['controller_drive_bias_full12'],FINAL_full12=info['actual_drive_target_full12'],
        selected_raw_full12=p['selected_raw_full12'],conditional_mean_full12=p['conditional_mean_full12'],
        local_raw_mean_delta_full12=p['local_raw_mean_delta_full12'],
        request_full12=a['policy_headroom_evidence']['requested_policy_residual_full12'],
        effective_full12=a['policy_headroom_evidence']['effective_policy_residual_full12'],
        actual_wheel_canonical_rad_s=info['semantic_task']['physical_evaluator']['measured_wheel_velocity_rad_s'],
        tracking=a['tracking_reference_evidence']['tracking_servo_names'])
samples=[summary(r) for r in active]
minimum=min(samples,key=lambda r:r['RR']['gap_m'])
leave=next(r for r in samples if r['FINAL_full12'][7]>-58+1e-8)
source_stop=next(summary(r) for r in active if any(864 in x['source_stop_event_ticks'] for x in r['step_info']['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']['layers']))
composite_stop=next(r for r in samples if r['source_full12'][8:]==[0]*4)
ranges={}
for j,name in enumerate(('hip','knee')):
    final=[r['FINAL_full12'][6+j] for r in samples]; actual=[r['RR']['actual_rr_hip_knee_deg'][j] for r in samples]
    ranges[name]=dict(FINAL_minmax_deg=[min(final),max(final)],actual_minmax_deg=[min(actual),max(actual)],
        FINAL_delta_from_logged_gate_FINAL_minmax_deg=[min(final)-final_entry[j],max(final)-final_entry[j]],
        actual_delta_from_first_gate_visible_actual_minmax_deg=[min(actual)-actual_entry[j],max(actual)-actual_entry[j]])
events=rows[-1]['step_info']['semantic_task']['history']['event_ticks']
reference_events={'active_lift':{'FR':24,'FL':2774,'RR':6995},'front_edge_crossed':{'FR':2751,'FL':4269,'RR':7979},'placed':{'FR':2761,'FL':5165}}
assert events==reference_events
control=manifest['control_contributions']
assert control['rear_owner_projection'] is False and control['rr_capture_assist_mode'] is None
assert control['nominal_geometry_advisory'] is None and control['rr_capture_wheel_mode']=='off'
assert control['AUX_updates_during_evaluation']==0 and control['local_auxiliary_optimizer_steps']==64
report=dict(schema='wlr50_clean.CP228864_aux64_DET_result_readonly.v1',source=str(SOURCE),
    mode='DET finite RR AUX training lineage; rear helpers OFF; FL hip-only assist ON',
    checkpoint=manifest['checkpoint_load_provenance'],counts=manifest['counts'],new_PPO_credit=0,new_AUX_steps=0,
    physical_duration_s=10940/120,actual_ticks=10940,decision_and_encoded_frame_count=1368,
    local_success=False,full_success=False,error=None,
    terminal_reason=rows[-1]['step_info']['termination_reason'],
    terminal_source=rows[-1]['step_info']['semantic_task']['termination_source'],
    first_unfinished_task='RR current qualified TOP capture, then0.5s usable bearing hold',
    prefix=base['current']['prefix'],event_ticks=events,reference_event_ticks=reference_events,event_ticks_equal=True,
    gate=dict(activation_tick=7979,gate_FINAL=final_entry,first_gate_visible_tick=7984,first_gate_visible_actual=actual_entry),
    active_request_rows=370,gate_visible_plus_active_endpoint_rows=371,ranges=ranges,
    FINAL_knee_exact_minus58_count=sum(r['FINAL_full12'][7]==-58 for r in samples),
    first_FINAL_knee_leaves_minus58=leave,minimum_active_decision_gap=minimum,latest=samples[-1],
    selected=[samples[0],leave,next(r for r in samples if r['tick']==8104),minimum,samples[-1]],
    source864_first_seen=source_stop,composite_N_first_zero=composite_stop,
    native_RR_contact_boolean_scan=dict(file=str(SOURCE/'capture_assist_ticks.jsonl'),
        total_rows=10940,consecutive_episode_ticks=[1,10940],missing_top_surface_contact_boolean=0,
        RR_top_surface_contact_true=0,since_activation_rows=2962,since_activation_TOP_true=0,since_activation_ground_true=0,
        scope='One completed stdlib streaming pass checked only RR booleans/tick continuity; no native joint/range minimum inferred.'),
    active_invariants=dict(no_actual_RR_tracking=all(r['tracking']==[] for r in samples),
        RR_source_constant=all(r['source_full12'][6:8]==[-6.9,-37.8] for r in samples),
        RR_mapped_N_constant=all(r['mapped_N_full12'][6:8]==[-8.15,-39.05] for r in samples),
        generic_RR_bias_zero=all(r['generic_bias_full12'][6:8]==[0,0] for r in samples),
        FL_FINAL_negative_all370=all(r['FINAL_full12'][8]<0 for r in samples),
        FL_actual_canonical_negative_all370=all(r['actual_wheel_canonical_rad_s'][0]<0 for r in samples)),
    limits=['Ranges/minimum gap are370 active-policy decision endpoints, not every120Hz measurement.',
        '998 prefix requests include final gate-detection step; gate-visible7984 is not credited local action.',
        'Terminal current_attempt_capture_eligible=false accompanies live=false; active native ground count0, not a ground revocation.',
        'FiniteAUX64 and priorPPO training exist; evaluation adds no training. Source stop event is not compositeN stop.'])
with (OUT/'CP228864_aux64_DET_result_readonly.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
def arr(x):return '['+', '.join(f'{v:+.3f}' for v in x)+']'
lines=['# CP228864 AUX64 确定性评估：前缀保留，RR膝脱离裁剪但仍未捕获','',
    '已封存：10940tick / 91.166667s / 1368帧，error=null；INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED，local/full success均false。标签为DET、finite RR AUX training lineage、rear helpers OFF（原FL hip-only assist ON）。本次新增PPO/AUX/训练credit全部0。',
    '', '前缀998请求严格local0、一次HISTORY、无local forward；与接受的CP225280 anchor对齐998行，raw/FINAL最大差均0、missing0。FR lift24/cross2751/place2761；FL lift2774/cross4269/place5165；RR lift6995/cross7979，全部同tick。',
    '', '## 本次真实RR动作（不是旧“370拍都−58”）','',
    '激活gate7979的FINAL=[+7.485,−58.000]°；首gate可见7984 actual=[+7.223,−57.760]°。以下范围仅370个真正active-policy决策端点，不包含前缀最后一拍。',
    '', '| RR | FINAL范围 deg | actual范围 deg | FINAL相对gate范围 deg | actual相对7984范围 deg |','|---|---|---|---|---|']
for name,v in ranges.items():lines.append(f"| {name} | {arr(v['FINAL_minmax_deg'])} | {arr(v['actual_minmax_deg'])} | {arr(v['FINAL_delta_from_logged_gate_FINAL_minmax_deg'])} | {arr(v['actual_delta_from_first_gate_visible_actual_minmax_deg'])} |")
lines += ['', '仅4/370个FINAL knee仍等于−58°；首次脱离为8024/66.866667s（−57.924482°），约67.53s已到−57.25°附近。展开确实执行，但整个FINAL膝变化范围仅约1.30°，不能把脱离裁剪等同已完成下降。',
    '', '| tick / s | RR FINAL / actual deg | gap mm | raw conditional mean hip,knee | local raw head hip,knee |','|---|---|---|---|---|']
for r in report['selected']:
    lines.append(f"| {r['tick']} / {r['time_s']:.6f} | {arr(r['FINAL_full12'][6:8])} / {arr(r['RR']['actual_rr_hip_knee_deg'])} | {r['RR']['gap_m']*1000:.6f} | {arr(r['conditional_mean_full12'][6:8])} | {arr(r['local_raw_mean_delta_full12'][6:8])} |")
lines += ['', 'active最小gap=55.230760mm（8224/68.533333s），末57.408820mm。source RR=[−6.9,−37.8]、mappedN=[−8.15,−39.05]全程恒定，generic RR bias=0、actual tracking列表空，P09 late保持pending。实际均值/动作已有变化，但未把轮端从高空推进到TOP；不单轴归因。',
    '', '## 真实TOP与停止/FL反转','',
    '仅对封存capture_assist_ticks做一次完整布尔扫描：10940行、tick1…10940连续，RR top_surface_contact=true为0，字段缺失0。gate7979起2962个native tick也无GROUND。因此不是仅凭15Hz漏看TOP；本次确无记录TOP，hold=0。末资格false伴随任务live=false，不解释成GROUND撤资格。',
    '', 'P09 source864 stop首次见于8200/68.333s，但同拍合成N四轮仍[+.3,+.3,+.3,+.3]；8920/74.333s合成N才全0，不能把source stop与最终合成stop混用。FL在370个active端点FINAL和实测canonical速度均负；尾N四轮0，FINAL=[−1.005577,−.038372,+.039282,−.059130]，实测=[−1.060134,−.037648,+.060978,−.059827] rad/s（FL,FR,RL,RR）。这是策略残差持续反向，不是缺失nominal或mask证据。',
    '', '第一未完成任务：RR真实顶部捕获及可用承载0.5s。不是动作角度仍完全锁死，也不是phase标签或无碰撞就成功。新视频由主任务导出，本报告不替代目视QA或声明全程成功。',
    '', '详细来源：`CP228864_aux64_DET_sealed_actualmetrics.json`（既有分析器）与 `CP228864_aux64_DET_result_readonly.json`（本次范围/计数）。']
with (OUT/'CP228864_aux64_DET_result_readonly.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(output=str(OUT/'CP228864_aux64_DET_result_readonly.json'),knee58_count=report['FINAL_knee_exact_minus58_count'],min_gap_mm=minimum['RR']['gap_m']*1000,ranges=ranges)))

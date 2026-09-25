"""Stdlib sealed decision analysis + single narrow native RR boolean scan."""
import json
from pathlib import Path
import analyze_actual_metrics as actual

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'runs/ppo_rr_capture_first_cp225280_v1/eval_CP229376_gain10_aux64_DET_0d0f894/source'
manifest=json.loads((SOURCE/'source_manifest.json').read_text(encoding='utf-8'))
parent=json.loads((SOURCE.parent/'run_manifest.json').read_text(encoding='utf-8'))
assert parent['lifecycle']=='COMPLETE' and parent['result']==manifest
assert manifest['mode']=='DETERMINISTIC_COMPOSITE_POLICY' and not manifest['diagnostic_intervention']
assert manifest['error'] is None and manifest['PPO_updates']==0
prefix=json.loads((OUT/'CP229376_DET_RR_activation_live.json').read_text(encoding='utf-8'))
assert prefix['prefix']['checked_rows']==998 and prefix['prefix']['raw_full12_max_abs_delta']==0 and prefix['prefix']['FINAL_full12_max_abs_delta']==0
samples=[]; last_info=None
original_reader=actual.decisions

def observed_reader(path,live,status):
    global last_info
    for row in original_reader(path,live,status):
        info=row['step_info']; last_info=info
        if row['policy_request']['capture_active'] is True:
            audit=info['actuator_target_effect_audit']; p=row['policy_request']; m=info['rr_capture_local']['metrics']
            deferred=info['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']
            samples.append({'tick':m['tick'],'time_s':m['time_s'],'RR':m,
                'source_full12':info['nominal_action_full12'],'mapped_N_full12':audit['native_drive_target_full12'],
                'generic_bias_full12':audit['controller_drive_bias_full12'],'FINAL_full12':info['actual_drive_target_full12'],
                'selected_raw_full12':p['selected_raw_full12'],'conditional_mean_full12':p['conditional_mean_full12'],
                'local_raw_mean_delta_full12':p['local_raw_mean_delta_full12'],
                'applied_local_raw_mean_delta_full12':p['applied_local_raw_mean_delta_full12'],
                'local_mean_coordinate_gain_full12':p['local_mean_coordinate_gain_full12'],
                'request_full12':audit['policy_headroom_evidence']['requested_policy_residual_full12'],
                'effective_full12':audit['policy_headroom_evidence']['effective_policy_residual_full12'],
                'actual_wheel_canonical_rad_s':info['semantic_task']['physical_evaluator']['measured_wheel_velocity_rad_s'],
                'tracking':audit['tracking_reference_evidence']['tracking_servo_names'],
                'late_unconsumed':all(not x['pending_event_consumed'] for x in deferred['layers']),
                'source_stop864_seen':any(864 in x['source_stop_event_ticks'] for x in deferred['layers'])})
        yield row

actual.decisions=observed_reader
current,_=actual.analyze(SOURCE/'video_policy_decisions.jsonl')
actual.decisions=original_reader
assert len(samples)==370 and current['source']['complete_rows']==1368
base={'schema':'wlr50_clean.CP229376_sealed_actualmetrics.v1','current':current,
      'accepted_prefix_comparison_reused_from':'CP229376_DET_RR_activation_live.json',
      'accepted_prefix_comparison':prefix['prefix'],'accepted_event_comparison':prefix['events'],
      'note':'analyze_actual_metrics reused with one decision pass; original CP225280 reference not rescanned.'}
(OUT/'CP229376_gain10_aux64_DET_sealed_actualmetrics.json').write_text(json.dumps(base,indent=2,allow_nan=False)+'\n',encoding='utf-8')
entry=current['gate_entry']; final_entry=entry['entry_final_rr_deg']; actual_entry=entry['rr']['actual_rr_deg']
minimum=min(samples,key=lambda r:r['RR']['gap_m']); tail=samples[-1]
ranges={}
for j,name in enumerate(('hip','knee')):
    f=[r['FINAL_full12'][6+j] for r in samples]; a=[r['RR']['actual_rr_hip_knee_deg'][j] for r in samples]
    ranges[name]={'FINAL_minmax_deg':[min(f),max(f)],'actual_minmax_deg':[min(a),max(a)],
        'FINAL_delta_from_gate_minmax_deg':[min(f)-final_entry[j],max(f)-final_entry[j]],
        'actual_delta_from_first_gate_visible_minmax_deg':[min(a)-actual_entry[j],max(a)-actual_entry[j]],
        'terminal_actual_delta_from_first_gate_visible_deg':a[-1]-actual_entry[j]}

native={'file':str(SOURCE/'capture_assist_ticks.jsonl'),'rows':0,'RR_TOP_true':0,'RR_TOP_missing':0,
        'since_gate_rows':0,'since_gate_TOP_true':0,'since_gate_GROUND_true':0,'since_gate_GROUND_missing':0,
        'rear_task_assist_disabled_true':0,'unexpected_RR_assist_active':0}
with (SOURCE/'capture_assist_ticks.jsonl').open('rb') as stream:
    for line in stream:
        if not line.endswith(b'\n'): raise ValueError('Sealed native log incomplete')
        row=json.loads(line); native['rows']+=1
        assert row['episode_physics_tick']==native['rows']
        rr=row['current_legs']['RR']; top=rr.get('top_surface_contact'); ground=rr.get('ground_contact')
        native['RR_TOP_true']+=int(top is True); native['RR_TOP_missing']+=int(type(top) is not bool)
        native['rear_task_assist_disabled_true']+=int(row.get('rear_task_assist_disabled') is True)
        native['unexpected_RR_assist_active']+=int(row.get('rr_capture_assist',{}).get('active') is True)
        if row['episode_physics_tick']>=7979:
            native['since_gate_rows']+=1; native['since_gate_TOP_true']+=int(top is True)
            native['since_gate_GROUND_true']+=int(ground is True); native['since_gate_GROUND_missing']+=int(type(ground) is not bool)
assert native['rows']==10940
native['consecutive_ticks']=[1,10940]
native['scope']='One streaming pass of exact RR contact booleans and helperflags/tick continuity; native joint extrema not computed.'
control=manifest['control_contributions']
assert control['rear_owner_projection'] is False and control['rr_capture_assist_mode'] is None
assert control['nominal_geometry_advisory'] is None and control['rr_capture_wheel_mode']=='off'
assert control['local_auxiliary_optimizer_steps']==64 and control['AUX_updates_during_evaluation']==0
old=json.loads((OUT/'CP228864_aux64_DET_result_readonly.json').read_text(encoding='utf-8'))
old_min=old['minimum_active_decision_gap']['RR']['gap_m']*1000
old_end=old['latest']['RR']['gap_m']*1000
stop=next((r for r in samples if r['source_stop864_seen']),None)
composite_stop=next((r for r in samples if r['source_full12'][8:]==[0]*4),None)
report={'schema':'wlr50_clean.CP229376_gain10_aux64_DET_result_readonly.v1','source':str(SOURCE),
    'mode':'DET finite RR AUX64 training lineage, gain10 coordinates; rear helpers OFF, FL hip-only assist ON',
    'checkpoint':manifest['checkpoint_load_provenance'],'training_counts':manifest['counts'],'evaluation_new_PPO_credit':0,'evaluation_new_AUX_credit':0,
    'actual_ticks':10940,'physical_duration_s':10940/120,'decision_and_frame_count':1368,'error':None,
    'local_success':last_info['local_task_success'],'full_success':last_info['full_task_success'],
    'termination_reason':last_info['termination_reason'],'termination_source':last_info['semantic_task']['termination_source'],
    'first_unfinished_task':'RR real TOP capture and0.5s usable same-attempt bearing hold',
    'prefix_evidence_file':'CP229376_DET_RR_activation_live.json','prefix':prefix['prefix'],'event_comparison':prefix['events'],
    'final_event_ticks':last_info['semantic_task']['history']['event_ticks'],
    'gate':{'activation_tick':7979,'FINAL_RR_deg':final_entry,'first_visible_tick':7984,'actual_RR_deg':actual_entry},
    'active_request_rows':370,'ranges':ranges,
    'FINAL_knee_exact_minus58_count':sum(s['FINAL_full12'][7]==-58 for s in samples),
    'minimum_active_decision_gap':minimum,'latest':tail,'selected':[samples[0],minimum,tail],
    'native_RR_boolean_scan':native,'source864_first_seen':stop,'composite_N_first_zero':composite_stop,
    'active_invariants':{'RR_source_constant':all(s['source_full12'][6:8]==[-6.9,-37.8] for s in samples),
        'RR_mapped_N_constant':all(s['mapped_N_full12'][6:8]==[-8.15,-39.05] for s in samples),
        'generic_RR_bias_zero':all(s['generic_bias_full12'][6:8]==[0,0] for s in samples),
        'all_actual_tracking_empty':all(s['tracking']==[] for s in samples),'P09_late_unconsumed':all(s['late_unconsumed'] for s in samples),
        'FL_FINAL_negative_endpoints':sum(s['FINAL_full12'][8]<0 for s in samples),
        'FL_actual_canonical_negative_endpoints':sum(s['actual_wheel_canonical_rad_s'][0]<0 for s in samples)},
    'comparison_with_CP228864':{'old_min_gap_mm':old_min,'new_min_gap_mm':minimum['RR']['gap_m']*1000,
        'min_gap_delta_mm':minimum['RR']['gap_m']*1000-old_min,'old_final_gap_mm':old_end,'new_final_gap_mm':tail['RR']['gap_m']*1000,
        'final_gap_delta_mm':tail['RR']['gap_m']*1000-old_end,'both_actual_TOP_count':0,
        'causal_limit':'Observation comparison across evaluated checkpoints, not isolation of coordinate gain, update, knee or whole-body causal effects.'},
    'limits':['Ranges/minimum are370 active decision endpoints; only RRcontact booleans were scanned at120Hz.',
              '998 prefix equality already independently verified; this report does not rescan all historicalreference.',
              'Terminal eligibility may becomefalse with live=false; GROUND interpretation uses native boolean/history, not terminal flagalone.',
              'No new model/production changes, physics, videoexport or PPO/AUX training performed by analyzer.']}
stem='CP229376_gain10_aux64_DET_result_readonly'
(OUT/(stem+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def arr(values): return '['+', '.join(f'{v:+.3f}' for v in values)+']'
lines=['# CP229376 gain10 / AUX64 确定性结果','',
    '封存91.166667s /10940tick /1368帧，error=null；local/full均false，INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED。这是有限RR AUX64训练谱系的DET，后腿helpers OFF、原FL assist ON；本次评估不增加PPO或AUX。', '',
    '前缀直接引用 `CP229376_DET_RR_activation_live.json`：998拍raw/FINAL对CP225280最大差0；FRplaced2761、FLplaced5165、RRqualified6995/cross7979完全一致，不再重扫历史参考。', '',
    '## RR实际变化（370个active决策端点）','',
    'gate FINAL=[+7.485144,−58.000000]°；首gate可见7984 actual=[+7.223447,−57.760433]°。', '',
    '| RR | FINAL范围° | actual范围° | actual相对7984范围° | actual末尾相对7984° |','|---|---|---|---|---|']
for name,r in ranges.items():
    lines.append(f"| {name} | {arr(r['FINAL_minmax_deg'])} | {arr(r['actual_minmax_deg'])} | {arr(r['actual_delta_from_first_gate_visible_minmax_deg'])} | {r['terminal_actual_delta_from_first_gate_visible_deg']:+.6f} |")
lines+=['', '| tick / 秒 | FINAL / actual RR° | gap mm | conditional raw / applied local raw mean |','|---|---|---|---|']
for s in report['selected']:
    lines.append(f"| {s['tick']} / {s['time_s']:.6f} | {arr(s['FINAL_full12'][6:8])} / {arr(s['RR']['actual_rr_hip_knee_deg'])} | {s['RR']['gap_m']*1000:.6f} | {arr(s['conditional_mean_full12'][6:8])} / {arr(s['applied_local_raw_mean_delta_full12'][6:8])} |")
lines+=['',
    f"最小gap {minimum['RR']['gap_m']*1000:.6f}mm（tick{minimum['tick']}），末gap {tail['RR']['gap_m']*1000:.6f}mm；FINAL knee仍等于−58仅{report['FINAL_knee_exact_minus58_count']}/370。实际关节确有改变，但仍未取得真实台面接触。",
    '',f"完整native布尔扫描{native['rows']}行、tick1…10940连续：RR TOP=true {native['RR_TOP_true']}，字段缺失{native['RR_TOP_missing']}；gate起{native['since_gate_rows']}行TOP={native['since_gate_TOP_true']}、GROUND={native['since_gate_GROUND_true']}。rear_task_assist_disabled为true {native['rear_task_assist_disabled_true']}行，RR assist active={native['unexpected_RR_assist_active']}。所以无TOP并非只从15Hz端点推测；此扫描未计算120Hz关节/间隙极值。", '',
    'active source RR=[−6.9,−37.8]、mappedN=[−8.15,−39.05]恒定，generic RR bias=0，actual tracking列表空，P09 late未消费。', '',
    f"FL末尾 nominal={tail['source_full12'][8]:+.6f}、FINAL={tail['FINAL_full12'][8]:+.6f}、实测canonical={tail['actual_wheel_canonical_rad_s'][0]:+.6f}rad/s；370拍中FINAL负速{report['active_invariants']['FL_FINAL_negative_endpoints']}拍、实测负速{report['active_invariants']['FL_actual_canonical_negative_endpoints']}拍。源864stop首次端点{stop['tick'] if stop else 'N/A'}，composite N四轮全0首次{composite_stop['tick'] if composite_stop else 'N/A'}；两者不可混为同一停止。", '',
    f"相对旧CP228864：最小gap {old_min:.6f}→{minimum['RR']['gap_m']*1000:.6f}mm；末gap {old_end:.6f}→{tail['RR']['gap_m']*1000:.6f}mm。间隙数值改善，但两者均无TOP。这是两个已评估checkpoint的观测差异，不证明gain10、某次更新或单一关节是唯一原因。", '',
    '第一未完成任务仍是RR真实顶部捕获和同次有效承载保持0.5s。没有完成RR捕获，更不能称RL或整段成功。视频由主线程交付；本报告未导出视频、未改生产、未执行训练。']
(OUT/(stem+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'outputs':stem,'min_gap_mm':minimum['RR']['gap_m']*1000,'min_tick':minimum['tick'],
                  'final_gap_mm':tail['RR']['gap_m']*1000,'ranges':ranges,'native':native,
                  'FL_tail_FINAL_actual':[tail['FINAL_full12'][8],tail['actual_wheel_canonical_rad_s'][0]],
                  'invariants':report['active_invariants']}))

"""Bounded stdlib first-opportunity attribution; no next-prefix or tensor read."""
import json
import math
from pathlib import Path
from collections import Counter

OUT=Path(__file__).resolve().parent
snap=json.loads((OUT/'continuous2048_first_opportunity_snapshot.json').read_text(encoding='utf-8'))
path=Path(snap['sources']['decisions']['path']); boundary=snap['sources']['decisions']['snapshot_bytes']
rows=[]; prefixes=0
with path.open('rb') as stream:
    while stream.tell()<boundary:
        start=stream.tell(); line=stream.readline(boundary-start)
        if not line.endswith(b'\n'): raise ValueError('Incomplete required opportunity snapshot')
        r=json.loads(line)
        if r['kind']=='frozen_prior_prefix':
            assert not rows and r['PPO_credit']==0
            prefixes+=1
        else:
            assert r['kind']=='activated_on_policy' and r['PPO_credit']==1
            r['_byte_range']=[start,stream.tell()]; rows.append(r)
            if r['step_info']['local_task_success']: break
    end=stream.tell()
assert len(rows)==87 and prefixes==998 and snap['completed_updates_observed']==0
metrics=[r['step_info']['rr_capture_local']['metrics'] for r in rows]
first_top=next(n for n,m in enumerate(metrics) if m['current_top_contact'])
assert first_top==79
assert all(m['current_attempt_capture_eligible'] and not m['ground_contact'] for m in metrics)
assert all(m['current_top_contact'] and m['current_top_bearing'] and m['within_top_xy'] and m['consecutive_top_samples']==7+8*(n-79)
           for n,m in enumerate(metrics) if n>=79)
last=rows[-1]['step_info']; history=last['semantic_task']['history']
assert last['local_reward']['terminated'] and not last['full_task_success']
assert all(not r['step_info']['local_reward']['terminated'] for r in rows[:-1])
descent_start=first_top
while descent_start>0 and metrics[descent_start]['gap_m']<metrics[descent_start-1]['gap_m']-1e-6:
    descent_start-=1
assert descent_start==70 and all(m['within_top_xy'] for m in metrics[70:80])
minimum=min(range(len(rows)),key=lambda n:metrics[n]['gap_m'])

def sample(n):
    r=rows[n]; i=r['step_info']; a=i['actuator_target_effect_audit']; p=r['policy_request']; ev=i['semantic_task']['physical_evaluator']
    rr=ev['current_legs']['RR']; d=i['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']
    return {'index':n,'global_decision':r['global_decision'],'byte_range':r['_byte_range'],'phase':[i['phase_id'],i['end_phase_id']],
        'metrics':metrics[n],'hold_s':i['rr_capture_local']['hold_elapsed_s'],
        'contact_geometry':{k:rr.get(k) for k in ('front_distance_m','clearance_m','ground_relative_lift_m','top_geometry','within_top_xy','within_lateral_span',
            'top_xy_outside_distance_m','air','ground_contact','contact_surface','contact_mode','obstacle_pair_active','top_contact','top_surface_contact',
            'bearing_force_n','support','active_attempt','current_lift_valid','lift_established','placed_on_top')},
        'source_RR_deg':i['nominal_action_full12'][6:8],'mapped_N_RR_deg':a['native_drive_target_full12'][6:8],
        'generic_bias_RR_deg':a['controller_drive_bias_full12'][6:8],
        'conditional_mean_RR':p['conditional_mean_full12'][6:8],'raw_local_head_RR':p['local_raw_mean_delta_full12'][6:8],
        'applied_local_raw_mean_RR':p['applied_local_raw_mean_delta_full12'][6:8],'selected_raw_RR':p['selected_raw_full12'][6:8],
        'request_RR_deg':a['policy_headroom_evidence']['requested_policy_residual_full12'][6:8],
        'effective_RR_deg':a['policy_headroom_evidence']['effective_policy_residual_full12'][6:8],
        'FINAL_RR_deg':i['actual_drive_target_full12'][6:8],
        'tracking':a['tracking_reference_evidence']['tracking_servo_names'],'deferred_source':d,
        'termination_reason':i['termination_reason'],'task_outcome_label':i['task_outcome_label']}

late=[x for r in rows for x in r['step_info']['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']['layers']]
pending=[x for r in rows for x in r['step_info']['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']['p12_pending_start']]
assert late and all(not x['pending_event_consumed'] and x['deferred_source_tracking_servo_names']==[] for x in late)
assert pending and all(x['status']=='local_RR_capture_holding_new_P12_start' and not x['rl_current_swing'] for x in pending)
assert all(len(r['observation'])==448 and math.isfinite(r['old_logp'][0]) and r['old_logp'][0]==r['policy_request']['selected_raw_log_probability']
           and r['policy_request']['selected_raw_full12']==r['step_info']['raw_policy_action_full12']
           and r['policy_request']['history_kernel_applications']==1 and r['policy_request']['sampling_draws']==1
           and r['step_info']['actuator_target_effect_audit']['all12_policy_channels_unmodified_at_actuator'] for r in rows)
rr_events=[e for e in history['lift_attempt_events'] if e.get('leg')=='RR']
assert not any('ground' in str(e).lower() and e['physics_tick']>=6995 for e in rr_events)
anomalies=[n for n,r in enumerate(rows) if r['step_info']['semantic_task']['physical_evaluator']['current_legs']['RR']['obstacle_pair_active'] and not metrics[n]['current_top_contact']]
top_gaps=[m['gap_m']*1000 for m in metrics[first_top:]]
report={'schema':'wlr50_clean.continuous2048_first_success_readonly.v1','source':str(path),
    'snapshot_bytes':boundary,'actual_read_end':end,'next_prefix_read':False,
    'counts':{'first_prefix_credit0':prefixes,'first_opportunity_PPO_samples':87,'new_PPO_updates_at_snapshot':0,'new_AUX_steps':0,
        'phases':dict(Counter(r['step_info']['phase_id'] for r in rows))},
    'mode':'CP229376 gain10/AUX64 existing checkpoint stochastic sampling before next PPO update; rear helpers OFF',
    'same_attempt':{'events':history['event_ticks'],'RR_lift_events':rr_events,'all87_eligible':True,'GROUND_endpoints':0,
        'GROUND_revocations_after_qualification':0,'first_TOP_endpoint':8624,'native_TOP_start_inferred':8618,
        'TOP_endpoint_ticks':[m['tick'] for m in metrics[first_top:]],'continuous_native_counts':[m['consecutive_top_samples'] for m in metrics[first_top:]],
        'terminal_tick':8680,'hold_s':last['rr_capture_local']['hold_elapsed_s'],'force_n':metrics[-1]['bearing_force_n'],
        'load_fraction':metrics[-1]['load_fraction'],'load_fraction_valid':metrics[-1]['load_fraction_valid']},
    'local_success':True,'full_success':False,'terminal':last['local_reward'],
    'selected_rows':[sample(n) for n in (29,70,74,78,79,80,81,86)],
    'last_legal_descent':{'start_tick':metrics[70]['tick'],'end_tick':metrics[79]['tick'],'action_rows':list(range(71,80)),
        'gap_mm':[m['gap_m']*1000 for m in metrics[70:80]],'all_within_top_xy':True},
    'minimum_gap':sample(minimum),'pre_capture_obstacle_contacts_not_awarded_local_TOP':[sample(n) for n in anomalies],
    'valid_TOP_gap_mm_minmax':[min(top_gaps),max(top_gaps)],
    'P09_late_all_unconsumed':True,'previous_source_tracking_inherited_empty':True,'P12_new_unload_pending':True,
    'original_sample_likelihood_and_HISTORY_verified_rows':87,
    'limits':['Only first episode decision endpoints and stored nativeobserver counters/history; no nativefile rescan.',
        'Negative height relative infinite topplane outsideXY is not proof of obstaclepenetration or validtouchdown.',
        'Earlier outsideXY TOP/ambiguous paircontacts occurred but localcontact/hold/placed were not credited there.',
        'Cannot infer no unobserved intradecision geometricintersection or all-link penetration from endpoints alone.',
        'This is preupdate stochastic localsuccess, not DET/fullsuccess/newupdate benefit, no AUXlabel or fitter created.']}
stem='continuous2048_first_success_readonly'
(OUT/(stem+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def pair(x):return '['+', '.join(f'{v:+.3f}' for v in x)+']'
lines=['# CP229376 连续训练首机会：真实局部捕获，尚未发生新更新','',
    f'固定快照{boundary:,} bytes，仅读至首回合终点byte{end:,}，没有读取第二前缀。998前缀credit0，87个active真实PPO样本（P09=80/P10=1/P11=1/P12=5）。这是既有CP229376 gain10/AUX64的随机采样能力，不是本run新PPO收益、DET或完整越障；没有新增AUX。','',
    'RR qualified6995/cross7979；首次有效TOP端点8624，连续native计数7→15→…→63，推断从8618至8680连续。末hold0.516667s、13.282523N、load_fraction0.490504（valid=true）。87端点eligible均true、GROUND均false，history无本次qualified后的GROUND撤资格。仅8680/72.333333s按保持完成提交local success；full=false。以上native连续性来自存储observer计数，未重扫120Hz日志。','',
    '| tick / phase | gap mm / hold s | source / mappedN / generic | mean / applied local mean / sample | FINAL / actual RR hip,knee° |','|---|---|---|---|---|']
for r in report['selected_rows']:
    m=r['metrics']; lines.append(f"| {m['tick']} / {'→'.join(r['phase'])} | {m['gap_m']*1000:+.3f} / {r['hold_s']:.3f} | {pair(r['source_RR_deg'])} / {pair(r['mapped_N_RR_deg'])} / {pair(r['generic_bias_RR_deg'])} | {pair(r['conditional_mean_RR'])} / {pair(r['applied_local_raw_mean_RR'])} / {pair(r['selected_raw_RR'])} | {pair(r['FINAL_RR_deg'])} / {pair(m['actual_rr_hip_knee_deg'])} |")
lines+=['',
    '最后有效下降是8552→8624、9次决策/0.6s：gap62.506→55.295→46.111→34.814→25.668→17.890→11.710→6.084→1.077→−0.034mm，全段合法XY。首次TOP8624仍用P09 RR source[−6.9,−37.8]/mapped[−8.15,−39.05]、generic[0,0]。8632才执行P10 knee展开接续；因此P10并非初触地的前因，但对随后的保持有nominal/mapper贡献，不能全算网络独立动作。P11接续继续，P09 late五通道始终未消费、previous tracking为空，新P12仍pending且rl_current_swing=false。','',
    '## −32.611mm不是此次有效TOP的穿透深度','',
    '该最小值在8224：front=−49.677mm、within_top_xy=false（距合法XY44.677mm），AIR、0N、无障碍pair、GROUND=false，轮底仍高于ground17.421mm。它是前缘外轮底相对台面水平面的负高度，不能据此断言轮子穿入台体；同样不能当下降完成。此前历史cross并没有把这个失去合法XY的状态当成已placed。','',
    '确有其他中间接触：8168/8176在XY外，原contact_surface=TOP，载荷约1.248/0.371N；8256/8264/8272为OBSTACLE_AMBIGUOUS。它们的local current_top_contact=false、hold=0，未计作本地有效TOP/placed；本次抬升谱系资格仍有效，不能混为撤资格。也不能删去这些接触把整87拍描述为干净成功轨迹。','',
    f"真正有效TOP8624…8680一直合法XY、实际sensor承载；该保持段gap仅在[{min(top_gaps):+.6f},{max(top_gaps):+.6f}]mm，非−32.6mm。此窗口没有记录GROUND、身体碰撞或纯wheel失败终止。端点数据不足以证明所有link在每个中间物理tick绝无几何相交，但没有证据将这次合法传感器保持判为假成功。",'',
    '87行原sample/old logp、448观测和一次HISTORY核验通过，selected=issued、全12通道无隐藏覆盖。下降/捕获/保持有sample与nominal接续共同贡献；只报告一次随机RR局部成功，不生成成功AUX标签、不拟合、不改生产。']
(OUT/(stem+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'output':stem,'counts':report['counts'],'same_attempt':report['same_attempt'],'min_gap_index':minimum,
    'pre_capture_nonqualified_contact_indices':anomalies,'valid_TOP_gap_mm_minmax':report['valid_TOP_gap_mm_minmax']}))

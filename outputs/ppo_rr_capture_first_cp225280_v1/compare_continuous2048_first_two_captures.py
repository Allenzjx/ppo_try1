"""Stdlib fixed snapshot; terminate immediately after second local terminal."""
import json
from collections import Counter
from pathlib import Path

OUT=Path(__file__).resolve().parent
snap=json.loads((OUT/'continuous2048_second_opportunity_snapshot.json').read_text(encoding='utf-8'))
source=Path(snap['sources']['decisions']['path']); boundary=snap['sources']['decisions']['snapshot_bytes']
episodes=[[]]; prefixes=[0]; terminals=0
with source.open('rb') as stream:
    while stream.tell()<boundary:
        start=stream.tell(); line=stream.readline(boundary-start)
        if not line.endswith(b'\n'):raise ValueError('Required completed opportunity unavailable')
        row=json.loads(line)
        if row['kind']=='frozen_prior_prefix':
            if episodes[-1]:episodes.append([]);prefixes.append(0)
            assert row['PPO_credit']==0
            prefixes[-1]+=1
        else:
            assert row['kind']=='activated_on_policy' and row['PPO_credit']==1
            row['_byte_range']=[start,stream.tell()]; episodes[-1].append(row)
            if row['step_info']['local_task_success']:
                terminals+=1
                if terminals==2:break
    end=stream.tell()
assert [len(g) for g in episodes]==[87,294] and prefixes==[998,997]
assert snap['completed_updates_observed']==0

def sample(row):
    i=row['step_info']; e=i['semantic_task']['physical_evaluator']; m=i['rr_capture_local']['metrics']
    c=e['transfer_roles']['RR']['transfer_direction_context']; a=i['actuator_target_effect_audit']; p=row['policy_request']
    return {'global_decision':row['global_decision'],'byte_range':row['_byte_range'],'tick':m['tick'],
        'phase':[i['phase_id'],i['end_phase_id']],'RR':m,'hold_s':i['rr_capture_local']['hold_elapsed_s'],
        'source_RR_deg':i['nominal_action_full12'][6:8],'mappedN_RR_deg':a['native_drive_target_full12'][6:8],
        'generic_RR_bias_deg':a['controller_drive_bias_full12'][6:8],'FINAL_RR_deg':i['actual_drive_target_full12'][6:8],
        'conditional_mean_RR':p['conditional_mean_full12'][6:8],'selected_raw_RR':p['selected_raw_full12'][6:8],
        'applied_local_mean_RR':p['applied_local_raw_mean_delta_full12'][6:8],
        'nominal_wheels_canonical_rad_s':i['nominal_action_full12'][8:12],
        'FINAL_wheels_canonical_rad_s':i['actual_drive_target_full12'][8:12],
        'actual_wheels_canonical_rad_s':e['measured_wheel_velocity_rad_s'],
        'COM_world_m':c['mass_weighted_com_position_w_m'],'COM_velocity_world_m_s':c['mass_weighted_com_velocity_w_m_s'],
        'body_x_relative_front_m':e['goal_features']['body_forward_m'],
        'body_linear_speed_m_s':e['goal_features']['body_linear_speed_m_s'],
        'body_angular_speed_rad_s':e['goal_features']['body_angular_speed_rad_s'],
        'body_bounds_world_m':e['body_traversal_geometry'],
        'support':{leg:{k:e['current_legs'][leg].get(k) for k in ('air','ground_contact','top_surface_contact','within_top_xy',
            'support','bearing_force_n','load_fraction','load_fraction_valid','clearance_m','front_distance_m')} for leg in ('FL','FR','RL','RR')},
        'physics_valid':e['valid'],'run_validity':e['run_validity'],'physical_evidence_status':e['physical_evidence_status'],
        'physical_termination_reason':e['termination_reason'],'local_termination_reason':i['termination_reason'],
        'deferred_source':i['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']}

summaries=[]
for epn,rows in enumerate(episodes,1):
    metrics=[r['step_info']['rr_capture_local']['metrics'] for r in rows]
    first=next(n for n,m in enumerate(metrics) if m['current_top_contact']); start=first
    while start>0 and metrics[start]['gap_m']<metrics[start-1]['gap_m']-1e-6:start-=1
    assert all(m['within_top_xy'] for m in metrics[start:first+1])
    assert all(m['current_attempt_capture_eligible'] and not m['ground_contact'] for m in metrics)
    assert all(m['current_top_contact'] and m['current_top_bearing'] and m['within_top_xy'] for m in metrics[first:])
    count0=metrics[first]['consecutive_top_samples']
    assert all(m['consecutive_top_samples']==count0+8*(j-first) for j,m in enumerate(metrics) if j>=first)
    late=[x for r in rows for x in r['step_info']['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']['layers']]
    pending=[x for r in rows for x in r['step_info']['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']['p12_pending_start']]
    assert all(not x['pending_event_consumed'] and x['deferred_source_tracking_servo_names']==[] for x in late)
    assert all(x['status']=='local_RR_capture_holding_new_P12_start' and not x['rl_current_swing'] for x in pending)
    states=[sample(r) for r in rows[start:]]
    touchdown=sample(rows[first]); last=sample(rows[-1]); first_p10=sample(next(r for r in rows if r['step_info']['phase_id']=='P10'))
    summary={'episode':epn,'actual_PPO_rows':len(rows),'prefix_credit0':prefixes[epn-1],
        'phase_counts':dict(Counter(r['step_info']['phase_id'] for r in rows)),
        'history_events':rows[-1]['step_info']['semantic_task']['history']['event_ticks'],
        'all_endpoint_eligible':True,'ground_endpoint_count':0,'local_success':True,'full_success':False,
        'first_TOP':touchdown,'first_P10_command':first_p10,'terminal':last,
        'inferred_native_TOP_start_tick':metrics[first]['tick']-count0+1,
        'last_native_TOP_tick':metrics[-1]['tick'],'TOP_counter_sequence':[m['consecutive_top_samples'] for m in metrics[first:]],
        'last_legal_descent_start':sample(rows[start]),'last_legal_descent_before_TOP':sample(rows[first-1]),
        'late_unconsumed':True,'new_P12_pending_receipt_count':len(pending),'new_P12_started':False,
        'window_body_COM':{'COM_z_minmax_m':[min(s['COM_world_m'][2] for s in states),max(s['COM_world_m'][2] for s in states)],
            'body_AABB_bottom_z_minmax_m':[min(s['body_bounds_world_m']['minimum_w_m'][2] for s in states),max(s['body_bounds_world_m']['minimum_w_m'][2] for s in states)],
            'COM_delta_firstTOP_to_terminal_m':[b-a for a,b in zip(touchdown['COM_world_m'],last['COM_world_m'])],
            'body_AABB_bottom_z_delta_firstTOP_to_terminal_m':last['body_bounds_world_m']['minimum_w_m'][2]-touchdown['body_bounds_world_m']['minimum_w_m'][2],
            'body_max_linear_speed_m_s':max(s['body_linear_speed_m_s'] for s in states),
            'body_max_angular_speed_rad_s':max(s['body_angular_speed_rad_s'] for s in states)},
        'whole_episode_physical_valid_endpoints':sum(r['step_info']['semantic_task']['physical_evaluator']['valid'] is True for r in rows),
        'whole_episode_run_validity_values':sorted(set(r['step_info']['semantic_task']['physical_evaluator']['run_validity'] for r in rows)),
        'whole_episode_physical_termination_reasons':sorted({r['step_info']['semantic_task']['physical_evaluator']['termination_reason'] for r in rows if r['step_info']['semantic_task']['physical_evaluator']['termination_reason']}),
        'selected_rows':[sample(rows[start]),sample(rows[first-1]),touchdown,first_p10,last]}
    summaries.append(summary)

report={'schema':'wlr50_clean.continuous2048_first_two_capture_comparison.v1','source':str(source),
    'snapshot_byte_boundary':boundary,'actual_read_end':end,'read_third_prefix':False,
    'new_on_policy_samples':381,'prefix_credit0':[998,997],'new_PPO_updates_at_snapshot':0,'new_AUX_steps':0,
    'checkpoint':'CP229376 gain10/AUX64, before any optimizer in this continuous2048 run',
    'episodes':summaries,
    'limits':['Local RR successes only; episode2 terminalP11 does not mean RLunload/fulltaskcomplete.',
        'Contact continuity from storednativeobserver counters; no fullnative120Hz rescan or proof of all-link collisionfree geometry.',
        'AABB lowerz is a bodyboundingbox diagnostic, not baseorigin, COM, rigidhipmount or terrainclearance.',
        'Different sampled trajectories permit different RR configurations; no fixedhipangle successlabel and no unique causalexplanation.',
        'No AUXlabels/fitting, runtimechanges, physics, Torch/PXR or live-third-prefix reads.']}
stem='continuous2048_first_two_capture_comparison'
(OUT/(stem+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def vec(a,n=3):return '['+', '.join(f'{x:+.{n}f}' for x in a)+']'
lines=['# 前两次随机RR成功：不同关节/全身状态均形成实际接触','',
    f'固定{boundary:,}字节边界，仅读到第二terminal byte{end:,}，没有读取第三前缀。87+294=381个真实PPO样本；前缀998+997均credit0。本run尚无optimizer，两次均属已保存CP229376 gain10/AUX64的随机轨迹，不是新更新收益或确定性成功。','',
    '| episode / event | tick | actual RR hip,knee° | gap mm / RR N / hold s | COM世界xyz m | body AABB最低z m |','|---|---|---|---|---|---|']
for ep in summaries:
    for label,key in [('last descent start','last_legal_descent_start'),('first TOP','first_TOP'),('terminal','terminal')]:
        s=ep[key];m=s['RR'];lines.append(f"| {ep['episode']} / {label} | {s['tick']} | {vec(m['actual_rr_hip_knee_deg'])} | {m['gap_m']*1000:+.3f} / {m['bearing_force_n']:.3f} / {s['hold_s']:.3f} | {vec(s['COM_world_m'],6)} | {s['body_bounds_world_m']['minimum_w_m'][2]:.6f} |")
lines+=['','第一段TOP端点8624→8680、计数7→63、hold0.516667s；第二段TOP端点10272→10328、计数5→61、hold0.5s（推断native10268起），末13.6653N。两段当前资格持续有效、GROUND端点均0，TOP保持期间合法XY且bearing=true。第一尾actual hip−16.409°，第二尾+4.251°，并不存在必须hip≈−20°才成功的实测结论。','',
    '## 源接续与四轮（顺序FL,FR,RL,RR；canonical rad/s）','',
    '| episode / event | wheel N | FINAL | actual | FL,FR,RL,RR载荷 N |','|---|---|---|---|---|']
for ep in summaries:
    for label,key in [('first TOP','first_TOP'),('terminal','terminal')]:
        s=ep[key];loads=[s['support'][leg]['bearing_force_n'] for leg in ('FL','FR','RL','RR')]
        lines.append(f"| {ep['episode']} / {label} | {vec(s['nominal_wheels_canonical_rad_s'])} | {vec(s['FINAL_wheels_canonical_rad_s'])} | {vec(s['actual_wheels_canonical_rad_s'])} | {vec(loads)} |")
lines+=['',
    '第二次10272首次TOP仍为P09 RR source[−6.9,−37.8]/mapped[−8.15,−39.05]/generic[0,0]；10280才P10 knee source−29.3/mapped−32.1/generic+0.75。这与第一次首次TOP8624、P10在8632的先后相同：初触地先形成，合法nominal展开参与保持，不能把整段归为纯policy独立发现。逐拍mean/sample/FINAL在JSON保存。', '',
    '两次P09 late五通道均未消费、previous-source tracking继承空；第一次P12仍pending，第二次停在P11，根本没有新P12启动，未拿RR捕获成功冒充RL完成。轮速不相同且FL仍反转，说明不同闭环全身状态下都能局部捕获；不据两条轨迹反推某种轮速/固定hip唯一正确。','',
    '## 身体降低与安全证据边界','']
for ep in summaries:
    w=ep['window_body_COM']
    lines.append(f"episode{ep['episode']}最后下降至保持窗口：COM z范围{vec(w['COM_z_minmax_m'],6)}m，body AABB最低z范围{vec(w['body_AABB_bottom_z_minmax_m'],6)}m；firstTOP→terminal COM位移{vec(w['COM_delta_firstTOP_to_terminal_m'],6)}m，AABB最低z变化{w['body_AABB_bottom_z_delta_firstTOP_to_terminal_m']:+.6f}m。")
lines+=['',
    '第二次approach确有身体/CoM下降，随后在接触保持期间回升；不能简单称整个身体塌低。两回合所有active端点physical valid=true、run_validity=VALID，physical evaluator未记录安全终止原因，最终只是RR局部成功。世界AABB最低z不是髋安装点高度，也不是精确机身离障间隙；这里没有四个一致髋安装点实测，不用关节角替代。没有明确塌低/碰撞/跌倒/硬限/NaN违规证据，但也不把端点统计当120Hz全link穿透或动态稳定性的完整证明。','',
    '这里证明的是传感器和同次资格认可的两种真实局部捕获状态，不解释唯一因果、不创建AUX标签、不重新运行实验。']
(OUT/(stem+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'output':stem,'new_PPO_rows':381,'updates':0,'summaries':[{'episode':s['episode'],'descent_start':s['last_legal_descent_start']['tick'],
    'firstTOP':s['first_TOP']['tick'],'P10':s['first_P10_command']['tick'],'terminal_actual':s['terminal']['RR']['actual_rr_hip_knee_deg'],
    'TOP_force':s['terminal']['RR']['bearing_force_n'],'body_COM':s['window_body_COM'],'physical_reasons':s['whole_episode_physical_termination_reasons'],
    'valid_endpoints':s['whole_episode_physical_valid_endpoints'],'pendingP12':s['new_P12_pending_receipt_count']} for s in summaries]}))

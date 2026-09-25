"""Only episode9's located byte window; stdlib, no later episode/native scan."""
import json
from collections import Counter
from pathlib import Path

OUT=Path(__file__).resolve().parent
snap=json.loads((OUT/'continuous2048_ninth_opportunity_snapshot.json').read_text(encoding='utf-8'))
source=Path(snap['sources']['decisions']['path']); boundary=snap['sources']['decisions']['snapshot_bytes']
# Located once by episode-boundary kind markers; prior episodes were not analyzed.
START,END=173109593,227159151
assert END<=boundary
rows=[]
with source.open('rb') as stream:
    stream.seek(START)
    while stream.tell()<END:
        start=stream.tell();line=stream.readline(END-start)
        if not line.endswith(b'\n'):raise ValueError('Truncated targeted episode window')
        row=json.loads(line); assert row['kind']=='activated_on_policy' and row['PPO_credit']==1
        row['_byte_range']=[start,stream.tell()];rows.append(row)
    assert stream.tell()==END
assert len(rows)==372
metrics=[r['step_info']['rr_capture_local']['metrics'] for r in rows]
assert metrics[0]['tick']==7984 and metrics[-1]['tick']==10948
last=rows[-1]['step_info']; history=last['semantic_task']['history']
assert history['event_ticks']['placed']['RR']==8218 and not last['local_task_success']
assert all(not m['current_top_contact'] and not m['current_top_bearing'] for m in metrics)
assert all(r['step_info']['rr_capture_local']['hold_elapsed_s']==0 for r in rows)

def sample(n):
    r=rows[n];i=r['step_info'];p=r['policy_request'];a=i['actuator_target_effect_audit']
    rr=i['semantic_task']['physical_evaluator']['current_legs']['RR']
    return {'index':n,'global_decision':r['global_decision'],'byte_range':r['_byte_range'],
        'phase':[i['phase_id'],i['end_phase_id']],'metrics':metrics[n],
        'leg_geometry_contact':{k:rr.get(k) for k in ('front_distance_m','top_xy_outside_distance_m','within_top_xy','top_geometry',
            'contact_surface','contact_mode','top_surface_contact','top_contact','obstacle_pair_active','ground_contact','air','support',
            'bearing_force_n','active_attempt','current_lift_valid','lift_established','consecutive_air_samples','free_air_reference_tick',
            'unsupported_free_lift_m','consecutive_free_air_samples','ground_relative_lift_m','placed_on_top')},
        'source_RR_deg':i['nominal_action_full12'][6:8],'mappedN_RR_deg':a['native_drive_target_full12'][6:8],
        'generic_RR_bias_deg':a['controller_drive_bias_full12'][6:8],
        'conditional_mean_RR':p['conditional_mean_full12'][6:8],'selected_raw_RR':p['selected_raw_full12'][6:8],
        'requested_RR_deg':a['policy_headroom_evidence']['requested_policy_residual_full12'][6:8],
        'effective_RR_deg':a['policy_headroom_evidence']['effective_policy_residual_full12'][6:8],
        'FINAL_RR_deg':i['actual_drive_target_full12'][6:8],
        'transition':[{'from_stage':x.get('from_stage'),'to_stage':x.get('to_stage'),'tick':x.get('physics_tick'),
                       'reason':x.get('reason'),'completion_values':x.get('completion_values')} for x in i['stage_transition_evidence']],
        'deferred':i['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']}

first_placed=next(n for n,m in enumerate(metrics) if m['placed'])
first_ground=next(n for n,m in enumerate(metrics) if m['ground_contact'])
first_p10=next(n for n,r in enumerate(rows) if r['step_info']['phase_id']=='P10')
first_reeligible=next((n for n in range(first_ground+1,len(rows)) if metrics[n]['current_attempt_capture_eligible']),None)
rr_events=[e for e in history['lift_attempt_events'] if e.get('leg')=='RR']
revokes=[e['physics_tick'] for e in rr_events if e['event']=='current_lift_revoked_ground']
assert revokes[0]==8414 and first_ground==54 and not metrics[first_ground]['current_attempt_capture_eligible']
late=[x for r in rows for x in r['step_info']['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']['layers']]
pending=[x for r in rows for x in r['step_info']['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']['p12_pending_start']]
assert all(not x['pending_event_consumed'] for x in late) and pending==[]
outside_pairs=[{'tick':m['tick'],'front_mm':r['step_info']['semantic_task']['physical_evaluator']['current_legs']['RR']['front_distance_m']*1000,
                'surface':r['step_info']['semantic_task']['physical_evaluator']['current_legs']['RR']['contact_surface'],
                'bearing_force_n':m['bearing_force_n'],'ground':m['ground_contact']}
               for r,m in zip(rows,metrics) if not m['within_top_xy'] and r['step_info']['semantic_task']['physical_evaluator']['current_legs']['RR']['obstacle_pair_active']]
report={'schema':'wlr50_clean.continuous2048_ninth_incomplete_readonly.v1','source':str(source),
    'fixed_snapshot_boundary':boundary,'analyzed_byte_range':[START,END],'episode':9,'read_episode10':False,
    'actual_PPO_rows':372,'phase_counts':dict(Counter(r['step_info']['phase_id'] for r in rows)),
    'history_events':history['event_ticks'],'RR_attempt_events':rr_events,
    'current_TOP_endpoint_count':0,'current_bearing_endpoint_count':0,'max_logged_hold_s':0,
    'local_success':False,'full_success':False,'terminal':last['local_reward'],
    'first_history_placed_visible':sample(first_placed),'first_P10_action':sample(first_p10),'first_GROUND_endpoint':sample(first_ground),
    'GROUND_revoke_native_ticks':revokes,'GROUND_endpoint_count':sum(m['ground_contact'] for m in metrics),
    'eligible_endpoint_count_after_firstGROUND':sum(m['current_attempt_capture_eligible'] for m in metrics[first_ground:]),
    'first_reeligible_after_GROUND':sample(first_reeligible) if first_reeligible is not None else None,
    'last':sample(len(rows)-1),'selected_rows':[sample(n) for n in (29,30,32,33,34,53,54,371)],
    'historical_placement_contact_inference':{'placed_native_tick':8218,'minimum_top_samples':2,
        'preceding_endpoint8216_TOP':False,'following_endpoint8224_TOP':False,
        'native_TOP_loaded_ticks_inferred_from_placement':[8217,8218],
        'endpoint8224_consecutive_AIR_samples':5,'endpoint8224_free_air_reference_tick':8219,
        'basis':'Selected TaskEvaluator loaded=top_surface and notground and withinXY; placement needs2consecutive loaded native samples. Historical event implies brief native TOP, not0.5s local bearing hold.',
        'exact_native_force_and_peak_local_hold':'Unavailable in this endpoint-only snapshot; not reconstructed from history.'},
    'outside_XY_obstacle_pair_endpoint_count':len(outside_pairs),'outside_XY_obstacle_pair_first_rows':outside_pairs[:5],
    'pending_late_unconsumed':True,'new_P12_started':False,
    'code_evidence':{'semantic_supervisor.py':'loaded predicate lines972–975; topcount andplaced lines1020–1022',
        'semantic_rr_capture_local_task.py':'currentTOP/bearing lines124–131; captureeligibility144–145; continuoushold andsuccess217–230',
        'configs/ppo_rr_capture_first_cp225280_v1/stage_task_spec.yaml':'history.minimum_top_samples=2 at84'},
    'limits':['No raw120Hz force/contact rescan; briefnative contact is code-and-event inference, not actual nativeforce reconstruction.',
        'P10 follows historical placement but initialcontact was already lost beforeP10; no unique causalclaim for laterGROUND.',
        'Ordinaryedgecontacts or absentnewqualification are not automatically purewheelviolations.',
        'No productionchange/newexperiment/Torch/PXR/AUXlabels; only episode9 outcomes described.']}
stem='continuous2048_ninth_incomplete_readonly'
(OUT/(stem+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def pair(x):return '['+', '.join(f'{v:+.3f}' for v in x)+']'
lines=['# 第9机会：短暂历史placed，不是当前捕获成功','',
    f'固定快照上界{boundary:,} bytes；只解析第9机会byte[{START:,},{END:,})，372个active样本，未读第10前缀。末10948/91.233333s/P11为INCOMPLETE_CONTROLLER_BLOCKED，local/full均false；372端点合法TOP/bearing均0、记录hold均0。','',
    '## 历史placed从哪里来','',
    'RR历史placed记录在8218，首次在8224决策端可见。但8224已AIR/0N、consecutiveAIR=5、free_air_reference_tick=8219。所选物理判定器只在top_surface接触、非GROUND且合法XY时累计TOP；minimum_top_samples=2。因此历史事件结合前后端点支持8217–8218至少两次短暂native TOP接触，随后即丢失。它不是凭P11标签伪造的历史，但也不是0.5s可用承载。该endpoint日志没有这两tick实际force或hold峰值，不能把max_logged_hold=0升级成物理上绝无瞬时触碰。','',
    '| tick / phase | gap / front mm | TOP / ground / eligible | source / mappedN / generic RR° | FINAL / actual RR° |','|---|---|---|---|---|']
for r in report['selected_rows']:
    m=r['metrics'];c=r['leg_geometry_contact']
    lines.append(f"| {m['tick']} / {'→'.join(r['phase'])} | {m['gap_m']*1000:+.3f} / {c['front_distance_m']*1000:+.3f} | {m['current_top_contact']} / {m['ground_contact']} / {m['current_attempt_capture_eligible']} | {pair(r['source_RR_deg'])} / {pair(r['mappedN_RR_deg'])} / {pair(r['generic_RR_bias_deg'])} | {pair(r['FINAL_RR_deg'])} / {pair(m['actual_rr_hip_knee_deg'])} |")
lines+=['',
    'P09→P10在8240发生，completion_values仅placed_RR=1。8216→8224→8232→8240期间RR source[−6.9,−37.8]、mappedN[−8.15,−39.05]和generic0未变，但FINAL knee从−21.566依次回折至−25.566/−29.566/−33.566°；这是本次sample/执行限速后的实际请求变化。首次接触丢失在P10以前，不能归为P10提前展开造成。', '',
    '第一拍P10到8248：source knee−29.3、mapped−34.6（本回合不是旧例的−32.1）、generic+0.75，FINAL knee−31.866，比上一拍展开1.700°，actual仍因跟踪/物理滞后进一步回折至−30.001°。8256进入P11接续source/mapped knee−27.2，FINAL展开到−21.866°，此时轮心已退至前缘−18.472mm、失去合法XY。动作和载荷有耦合，本表不能指定某个单轴为以后掉地的唯一原因。','',
    '## 第一次GROUND与资格保护','',
    '首次native current_lift_revoked_ground=8414；最近决策端8408仍非GROUND、eligible=true，8416已GROUND、eligible=false。该时刻比第一P10动作晚约1.4s，RR在XY外，gap≈−51.198mm，并有OBSTACLE_AMBIGUOUS边缘/立面pair。掉地后继续保留active任务和历史cross/placed，但历史不替代当前资格或支撑。', '',
    f"本回合GROUND端点共{report['GROUND_endpoint_count']}，首次GROUND后eligible=true端点{report['eligible_endpoint_count_after_firstGROUND']}。首个重新eligible端点8552：free-air reference8536、连续AIR16tick、实测unsupported_free_lift=11.929mm，current_lift_valid/lift_established均true，但仍XY外、TOPfalse/0N。这不同于只沿用旧placed，不能说掉地后资格一直为false，也不把单个initial_clearance事件自动当新qualified。末当前TOPfalse/0N、XY区域外75.595mm、hold0，故没有本地成功。P09 late始终未消费，未启动P12新卸载。", '',
    '边缘短接触允许连续处理，不等同跳过主动抬升的纯wheel爬升；本次有既有真实RR资格，最后是普通任务未完成，不追加未被观测证明的纯wheel违规。局部成功仍要求当前合法TOP+真实bearing+本次有效抬升谱系+连续0.5s，阶段P11和旧placed都不能单独满足。', '',
    '窄代码依据：semantic_supervisor.py的loaded/TOP计数与placed；semantic_rr_capture_local_task.py的currentTOP/current-attempt和连续hold；当前task_spec的minimum_top_samples=2。未重扫全仓库或原始native日志，未修改运行/生产或创建AUX标签。']
(OUT/(stem+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'output':stem,'rows':372,'first_placed':metrics[first_placed]['tick'],'P10':metrics[first_p10]['tick'],
    'first_GROUND':metrics[first_ground]['tick'],'revoke_native':revokes[0],'ground_endpoints':report['GROUND_endpoint_count'],
    'eligible_after_ground':report['eligible_endpoint_count_after_firstGROUND'],'outside_pairs':len(outside_pairs),'terminal':last['termination_reason']}))

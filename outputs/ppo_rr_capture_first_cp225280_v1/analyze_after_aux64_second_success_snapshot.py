"""Stdlib fixed-byte evidence for opportunity2; not a teacher/AUX fitter."""
from pathlib import Path
from collections import Counter
import json
import math

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/ppo_rr_capture_first_cp225280_v1/train_after_aux64_fresh512_0ff03ea/decisions.jsonl'
BOUNDARY=68784689
groups=[[]]; prefixes=[0]; next_prefix=None
with SOURCE.open('rb') as stream:
    while stream.tell()<BOUNDARY:
        start=stream.tell(); line=stream.readline(BOUNDARY-start)
        if not line.endswith(b'\n'): break
        row=json.loads(line)
        if row['kind']=='frozen_prior_prefix':
            assert row['PPO_credit']==0
            if groups[-1]:
                if len(groups)==2:
                    next_prefix=row['local']['metrics']; break
                groups.append([]); prefixes.append(0)
            prefixes[-1]+=1
        else:
            assert row['kind']=='activated_on_policy' and row['PPO_credit']==1
            row['_snapshot_byte_start']=start; row['_snapshot_byte_end']=stream.tell()
            groups[-1].append(row)
    read_end=stream.tell()
rows=groups[1]
assert [len(g) for g in groups]==[423,46] and next_prefix is not None
metrics=[r['step_info']['rr_capture_local']['metrics'] for r in rows]
top=[i for i,m in enumerate(metrics) if m['current_top_contact']]
assert top==list(range(37,46))
assert all(m['current_attempt_capture_eligible'] and not m['ground_contact'] for m in metrics)
assert all(metrics[i]['consecutive_top_samples']==3+8*(i-37) for i in top)
last=rows[-1]['step_info']; task=last['semantic_task']
assert last['local_task_success'] and last['local_reward']['terminated'] and not last['full_task_success']

def summarize(index):
    r=rows[index]; i=r['step_info']; local=i['rr_capture_local']; m=local['metrics']
    a=i['actuator_target_effect_audit']; p=r['policy_request']; s=i['semantic_task']
    physical=s['physical_evaluator']; legs=physical['current_legs']
    deferred=s['nominal_provider_diagnostics']['rr_local_deferred_late_v2']
    context=physical.get('transfer_roles',{}).get('RR',{}).get('transfer_direction_context',{})
    return dict(index=index,global_decision=r['global_decision'],tick=m['tick'],time_s=m['time_s'],
        phase=[i['phase_id'],i['end_phase_id']],RR=m,hold_s=local['hold_elapsed_s'],
        RR_front_mm=legs['RR']['front_distance_m']*1000,
        source_full12=i['nominal_action_full12'],mapped_N_full12=a['native_drive_target_full12'],
        generic_bias_full12=a['controller_drive_bias_full12'],
        prior_conditional_mean_full12=p['prior_same_observation_conditional_mean_full12'],
        conditional_mean_full12=p['conditional_mean_full12'],raw_local_head_full12=p['local_raw_mean_delta_full12'],
        selected_raw_full12=p['selected_raw_full12'],old_logp=r['old_logp'][0],
        residual_request_full12=a['policy_headroom_evidence']['requested_policy_residual_full12'],
        effective_residual_full12=a['policy_headroom_evidence']['effective_policy_residual_full12'],
        FINAL_full12=i['actual_drive_target_full12'],actual_wheels_canonical_rad_s=physical['measured_wheel_velocity_rad_s'],
        support={leg:{k:legs[leg].get(k) for k in ('air','ground_contact','top_surface_contact','support','bearing_force_n','load_fraction_valid')} for leg in ('FL','FR','RL','RR')},
        body_base_x_relative_front_m=physical['goal_features']['body_forward_m'],
        world_COM_m=context.get('mass_weighted_com_position_w_m'),
        body_world_displacement_m=context.get('body_world_displacement_m'),body_reference_tick=context.get('reference_tick'),
        tracking=a['tracking_reference_evidence'].get('tracking_servo_names'),deferred_source=deferred)

selected=[summarize(i) for i in (0,23,34,35,36,37,38,40,45)]
candidate=rows[24:46]
assert all(r['before_local']['metrics']['gap_m']>r['step_info']['rr_capture_local']['metrics']['gap_m']+1e-6 for r in rows[24:38])
def sample_ok(r):
    p=r['policy_request']; a=r['step_info']['actuator_target_effect_audit']
    return (len(r['observation'])==448 and len(p['selected_raw_full12'])==12 and
        math.isfinite(r['old_logp'][0]) and r['old_logp'][0]==p['selected_raw_log_probability'] and
        p['sampling_draws']==1 and p['history_kernel_applications']==1 and p['extra_random_draws']==0 and
        p['selected_raw_full12']==r['step_info']['raw_policy_action_full12'] and a['all12_policy_channels_unmodified_at_actuator'])
assert all(sample_ok(r) for r in rows)
late=[x for r in rows for x in r['step_info']['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']['layers']]
pending=[x for r in rows for x in r['step_info']['semantic_task']['nominal_provider_diagnostics']['rr_local_deferred_late_v2']['p12_pending_start']]
assert late and all(not x['pending_event_consumed'] and x['deferred_source_tracking_servo_names']==[] for x in late)
assert pending and all(x['status']=='local_RR_capture_holding_new_P12_start' and not x['rl_current_swing'] for x in pending)
events=[e for e in task['history']['lift_attempt_events'] if e.get('leg')=='RR' and e['physics_tick']>6000]
report=dict(schema='wlr50_clean.after_aux64_second_success_snapshot.v1',source=str(SOURCE),
    snapshot_byte_boundary=BOUNDARY,read_end=read_end,opportunity=2,
    counts=dict(new_second_opportunity_PPO_rows=46,first_opportunity_rows=423,total_new_PPO_rows=469,
        prefix_credit0=prefixes,completed_fresh_PPO_updates=0,remaining_to512=43,
        phase_rows=dict(Counter(r['step_info']['phase_id'] for r in rows))),
    local_success=True,full_task_success=False,mode='stochastic_training_after_AUX64_before_update7',
    activation=rows[0]['before_local'],terminal=last['rr_capture_local'],terminal_reward=last['local_reward'],
    history_event_ticks=task['history']['event_ticks'],RR_qualification_events=events,
    TOP_endpoint_ticks=[metrics[i]['tick'] for i in top],
    same_attempt_continuity=dict(all46_current_eligible=True,ground_endpoints=0,ground_revocations_after_qualification=0,
        consecutive_TOP_samples=[metrics[i]['consecutive_top_samples'] for i in top],
        native_contact_start_tick_inferred=8278,terminal_hold_s=.55,
        evidence='Stored native counters3→67 increase8 per8ticks and observed hold0.016667→0.55 support uninterrupted same-attempt TOP/bearing; not an independent120Hz log rescan.'),
    source_rules=dict(P09_late_unconsumed=True,carrier_tracking_inherited_empty=True,
        P10_RR_knee_allowed=True,P12_new_unload_pending=True,P12_pending_receipts=pending,
        phase_transition_is_not_terminal=True),
    raw_execution=dict(all46_observation448_raw12_logp_real=True,HISTORY_applications=1,
        sampling_draws=1,extra_random_draws=0,all12_unmodified=True),selected_rows=selected,
    optional_future_AUX_evidence_only=dict(approved_for_fit=False,fit_performed=False,
        selected_episode_indices=list(range(24,46)),global_range=[candidate[0]['global_decision'],candidate[-1]['global_decision']],
        start_observation_tick=8168,end_tick=8344,duration_s=176/120,
        strict_gap_descent_actions=14,subsequent_contact_hold_actions=8,
        rows=[dict(index=24+j,global_decision=r['global_decision'],byte_start=r['_snapshot_byte_start'],byte_end=r['_snapshot_byte_end'],
                   observation_dimension=len(r['observation']),raw_action_dimension=12,old_logp=r['old_logp'][0],verified=sample_ok(r)) for j,r in enumerate(candidate)],
        boundaries='Exclude episode rows0–23 including long high-gap/knee-clipped AIR. Rows24–37 execute measured descent; rows38–45 maintain real TOP. Not46 universal success labels.',
        control_compatibility='Same0ff tracking+deferred-source version. Includes legitimate P10 RRk source expansion and generic+.75 at8288 plus P11/held P12. Any later source change requires new compatibility review; raw sample is not an isolated pure-policy cause.'),
    next_prefix=next_prefix,
    limits=['Local RR capture success only; not DET, full traversal or update7 learning benefit.',
        'No new AUX fit/data publication/optimizer/simulator was run; raw samples retain original likelihoods, not teacher likelihood.',
        'Actual RR joint tracking can differ strongly from FINAL while contact is sustained; do not label joint target as measured pose.'])
stem='train_after_aux64_second_success_readonly'
with (OUT/(stem+'.json')).open('x',encoding='utf-8') as f: json.dump(report,f,indent=2,allow_nan=False)
def arr(values,d=3): return '['+', '.join(f'{v:+.{d}f}' for v in values)+']'
lines=['# AUX64 后第二机会：真实 RR capture+hold 局部成功','',
    '固定快照68,784,689 bytes；第二机会46行（global228776–228821），本块累计469行、尚未optimizer，余43行。已见第三真实前缀开始，未中断rollout。此为AUX64训练谱系下随机采样进展，非确定性/全程成功、非update7收益。',
    '', '## 同一次有效抬升与真实保持', '',
    'RR qualified6994 → cross/gate7975 → placed8279。首次TOP端点8280已有3个连续样本；之后8288…8344计数11…67，每8tick增8。native hold0.016667→0.55s，尾10.053564N、load_fraction0.330140（valid=true）、当前资格true、GROUND=false。46行无GROUND，history无此次抬升后的GROUND撤资格。计数/hold支持8278→8344连续真实承载；未另读取120Hz逐tick日志。',
    '', '终止是RR_CAPTURE_HOLD_SUCCESS，local success=true、full=false；普通P09→P10→P11→P12变化先前不终止，最终才按0.5s保持提交局部终止。',
    '', '| tick / phase | gap mm / hold s | RR raw mean / sample | source / mappedN / generic bias | FINAL / actual hip,knee deg |',
    '|---|---|---|---|---|']
for r in selected:
    lines.append(f"| {r['tick']} / {'→'.join(r['phase'])} | {r['RR']['gap_m']*1000:+.3f} / {r['hold_s']:.3f} | {arr(r['conditional_mean_full12'][6:8])} / {arr(r['selected_raw_full12'][6:8])} | {arr(r['source_full12'][6:8])} / {arr(r['mapped_N_full12'][6:8])} / {arr(r['generic_bias_full12'][6:8])} | {arr(r['FINAL_full12'][6:8])} / {arr(r['RR']['actual_rr_hip_knee_deg'])} |")
lines += ['', 'P09 late五通道整段未消费，deferred carrier tracking一直继承空；合法P10 RR knee继续执行（8288 generic bias+0.75）；P12从8304进入时仍holding_new_P12_start、rl_current_swing=false，尾亦未启动新RL卸载。P12标签不是RL完成。',
    '', '首观测RR actual[+7.223,−57.757]°；尾[+5.310,−32.910]°。同输入local raw head、冻结prior条件mean与组合mean已在JSON分别保留。触地前sample膝连续偏于该拍mean上方，但保持段并非每拍同符号；P10源展开、HISTORY、其他腿与负载也在共同作用，不把成功全归AUX或RR两通道。尾FINAL[+10.336,−9.858]与actual差别明显，不能称目标已被机械精确跟踪。',
    '', '## 四轮与实际支撑窗口','', '轮顺序FL,FR,RL,RR；mean/raw无量纲，其余轮速canonical rad/s。', '',
    '| tick | N | mean / sample | FINAL / actual | FL,FR,RL,RR bearing N / base x−front m |','|---|---|---|---|---|']
for r in [selected[i] for i in (4,5,6,7,8)]:
    loads=[r['support'][leg]['bearing_force_n'] for leg in ('FL','FR','RL','RR')]
    lines.append(f"| {r['tick']} | {arr(r['source_full12'][8:12])} | {arr(r['conditional_mean_full12'][8:12])} / {arr(r['selected_raw_full12'][8:12])} | {arr(r['FINAL_full12'][8:12])} / {arr(r['actual_wheels_canonical_rad_s'])} | {arr(loads)} / {r['body_base_x_relative_front_m']:+.6f} |")
lines += ['', '接触种类与AIR/ground/top支撑真值、CoM world和body固定短窗位移见JSON；AIR不计承载。FL负轮速仍存在，此次局部成功不证明持续反转无害。',
    '', '## 仅供以后决策的短有效片段（未拟合）','',
    '候选是第二机会row24–45/global228800–228821：从before tick8168至8344，共1.466667s/22行。前14拍gap由约69.958mm持续降到初TOP，后8拍保持真实承载；448观测、原12维sample、old logp、一次HISTORY/真实下发均可得。JSON列出固定文件byte地址。不要复制row0–23高gap/膝裁剪AIR，也不要把46行全部标为成功动作。',
    '', '该短段包含当前0ff正确source/mapper与P10展开、P11接续、P12 pending，未来若改source不能直接跨版本蒸馏。这里只界定实测可用证据，没有生成teacher动作、执行AUX或改模型；仍需独立有限学习决策。']
with (OUT/(stem+'.md')).open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(output=str(OUT/(stem+'.json')),counts=report['counts'],success=True,hold=.55,candidate_rows=22)))

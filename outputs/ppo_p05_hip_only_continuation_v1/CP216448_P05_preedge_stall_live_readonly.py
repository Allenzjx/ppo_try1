"""Fixed completed 28--38 s window only; never forwards or controls simulation."""
import hashlib
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/20260922T1521094182530Z_g336b7c56d2f0_3cdf838115e44829b66ad9585ad92ba7/source'
with (SOURCE/'capture_assist_ticks.jsonl').open('rb') as f:
    lines=list(itertools.islice(f,3358,4560))
assert len(lines)==1202 and all(line.endswith(b'\n') for line in lines)
rows=[json.loads(x) for x in lines]
assert [r['episode_physics_tick'] for r in rows]==list(range(3359,4561))
assert all(r['phase']=='P05' for r in rows)
leg_order=['FL','FR','RL','RR']
wheel_order=['front_left_ankle','front_right_ankle','rear_left_ankle','rear_right_ankle']
def stat(v):return dict(minimum=min(v),maximum=max(v),mean=sum(v)/len(v))
def select(r):
    d=r['dispatch'];c=d['capture_assist_evidence'];legs=r['current_legs']
    native=d['native_drive_target_full12'][8:];req=d['independent_policy_residual_requested_full12'][8:];final=d['drive_target_full12'][8:]
    return dict(tick=r['episode_physics_tick'],sim_time_s=r['sim_time_s'],dispatch_physics_tick=d['physics_tick'],
        nominal_after_frame_wheels=r['nominal_full12'][8:],actual_dispatch_mapped_nominal_wheels=native,
        raw_Gaussian_policy_wheels=None,policy_REQUEST_after_projector_wheels=req,
        actual_permission_mask_wheels=r['policy_permission_mask_full12'][8:],mask_object='residual permission; not nominal ownership or final actuator selection',
        effective_same_ACK_wheel_addition=[a-b for a,b in zip(final,native)],final_wheel_target=final,
        measured_canonical_wheel_rad_s=[r['wheels'][name]['velocity_rad_s'] for name in wheel_order],
        native_joint_ids=None,measured_native_joint_rad_s=None,live_source_atomic_group_id=None,
        last_dispatch_source='saved atomic_ack dispatch; production AtomicRobotAdapter final Full12 setter',
        contact={l:{k:legs[l].get(k) for k in ['air','ground_contact','top_surface_contact','obstacle_pair_active','support','bearing_force_n','bearing_verified']} for l in leg_order},
        FL={k:legs['FL'].get(k) for k in ['front_distance_m','clearance_m','within_top_xy','within_lateral_span','air','top_contact','consecutive_top_samples']},
        FL_target_hip_knee_deg=d['drive_target_full12'][:2],FL_actual_hip_knee_deg=r['actual_full12'][:2],
        body_position_w_m=r['body']['position_w_m'],body_velocity_w_m_s=r['body']['linear_velocity_w_m_s'],
        support_signed_margin_m=r['support']['signed_margin_m'],support_projection_inside=r['support']['projection_inside'],
        source_flags={k:c['context'][k] for k in ['source_unfold_dispatched','source_endpoint_issued','crossed_FL','qualified_FL','physical_valid','other_support_count']},
        capture_assist={k:r['capture_assist'][k] for k in ['mode_name','active','owners']},
        continuation=r['capture_continuation'])
window=rows[1:]
assert all(r['policy_permission_mask_full12']==[1]*12 for r in window)
assert all(r['capture_assist']['mode_name']=='WAIT' and not r['capture_assist']['active'] for r in window)
assert all(r['dispatch']['capture_assist_evidence']['assist_correction_full12']==[0]*12 for r in window)
errors=[];timing_errors=[]
for before,r in zip(rows,rows[1:]):
    d=r['dispatch'];n=d['native_drive_target_full12'][8:];req=d['independent_policy_residual_requested_full12'][8:];target=d['drive_target_full12'][8:]
    errors.extend(abs(a-b-c) for a,b,c in zip(target,n,req))
    timing_errors.extend(abs(a-b) for a,b in zip(n,before['nominal_full12'][8:]))
assert max(errors)<1e-12 and max(timing_errors)==0
changes=[]
for before,r in zip(rows,rows[1:]):
    old=(before['nominal_full12'][8:],before['dispatch']['native_drive_target_full12'][8:])
    new=(r['nominal_full12'][8:],r['dispatch']['native_drive_target_full12'][8:])
    if old!=new:changes.append(select(r))
source_path=ROOT/'configs/recording_motion_contract.json'
source_contract=json.loads(source_path.read_text())
p05=next(p for p in source_contract['phases'] if p['state_id']=='P05')
wheel_waypoints=[{k:w.get(k) for k in ['time_s','full12','changed_channels','atomic_channels','source_events','source_commands','atomic_batch_id']}
    for w in p05['waypoints'] if any('wheel' in command for command in w.get('source_commands',[]))]
steady=[r for r in window if r['episode_physics_tick']>=3706]
snapshots=[select(r) for r in window if r['episode_physics_tick'] in [3360,3487,3488,3489,3568,3569,3570,3706,4090,4560]]
snapshot_by_tick={s['tick']:s for s in snapshots}
# These streams were initially buffered, but became readable during the fixed
# window review. Read only the requested completed ticks, never extend the window.
if (SOURCE/'native_tick_audit.jsonl').stat().st_size:
    with (SOURCE/'native_tick_audit.jsonl').open('rb') as f:
        for index,line in enumerate(itertools.islice(f,4560),1):
            if index not in snapshot_by_tick:continue
            n=json.loads(line);s=snapshot_by_tick[index];a=n['native_audit']
            assert n['episode_physics_tick']==index and n['source_phase_id']=='P05'
            assert n['nominal_full12'][8:]==s['actual_dispatch_mapped_nominal_wheels']
            assert n['projected_residual_full12'][8:]==s['policy_REQUEST_after_projector_wheels']
            s.update(raw_Gaussian_policy_wheels=a['raw_policy_action_full12'][8:],
                native_actuator_wheel_target_rad_s=a['actual_native_targets']['wheel_velocity_rad_s'],
                last_dispatch_source=a['actual_target_source'],actual_mapping_matches_dispatch=a['actual_mapping_matches_dispatch'],
                setter_dispatch_targets_equal=a['setter_dispatch_targets_equal'],native_audit_verified=a['verified'])
if (SOURCE/'video_policy_decisions.jsonl').stat().st_size:
    wanted={(s['tick']-1)//8+1 for s in snapshots}
    with (SOURCE/'video_policy_decisions.jsonl').open('rb') as f:
        for index,line in enumerate(itertools.islice(f,570),1):
            if index not in wanted:continue
            d=json.loads(line);q=d['policy_request'];diag=d['step_info']['semantic_task']['nominal_provider_diagnostics']
            for s in snapshots:
                if not d['start_tick']<s['tick']<=d['end_tick']:continue
                assert s['raw_Gaussian_policy_wheels']==d['raw_policy_action_full12'][8:]
                s.update(policy_decision=d['decision'],raw_current_cap_wheels=q['current_cap_full12'][8:],
                    conditional_mean_wheels=q['conditional_mean_full12'][8:],
                    sample_draws=q['sampling_draws'],nominal_ownership_diagnostic=dict(
                        current_final_stop_owner_active=diag['final_stop_owner']['active'],
                        P06_layer_present=diag['capture_continuation']['P06_layer_present'],
                        policy_residual_restricted=diag['capture_continuation']['policy_residual_restricted']))
bytick={r['episode_physics_tick']:r for r in rows}
result=dict(schema='wlr50_clean.CP216448_P05_preedge_stall_live_readonly.v1',source=str(SOURCE),checkpoint_label='CP216448 deterministic; provided active run association',
    ticks=[3360,4560],one_preceding_tick=3359,time_s=[28,38],complete_selected_tick_count=1201,
    selected_1202_lines_sha256=hashlib.sha256(b''.join(lines)).hexdigest(),
    canonical_leg_order=leg_order,wheel_units='rad/s; already canonical reported velocities, not inferred native-axis readback',
    raw_policy_and_native_audit_files_at_read={name:(SOURCE/name).stat().st_size for name in ['video_policy_decisions.jsonl','native_tick_audit.jsonl','physical_observations.jsonl']},
    raw_request_limit='Capture evidence has physical REQUEST; later-flushed native/decision streams supply actual raw/cap for selected same-tick snapshots. No arctanh or independent nominal reconstruction used. Numeric native joint IDs/readback are not in these selected artifacts.',
    source_contract_path=str(source_path),source_contract_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
    configured_P05_wheel_waypoints=wheel_waypoints,configured_P05_endpoint_full12=p05['end_full12'],
    configured_stop_live_match_limit='Exact current nominal/dispatch zero transition is saved. Specific live atomic group/batch owner metadata is not in capture_assist_ticks; fixed source P05 contract explicitly authors the same four-wheel stop.',
    first_nominal_after_frame_zero_tick=next(r['episode_physics_tick'] for r in window if r['nominal_full12'][8:]==[0]*4),
    first_actual_dispatch_nominal_zero_tick=next(r['episode_physics_tick'] for r in window if r['dispatch']['native_drive_target_full12'][8:]==[0]*4),
    first_source_endpoint_issued_context_tick=next(r['episode_physics_tick'] for r in window if r['dispatch']['capture_assist_evidence']['context']['source_endpoint_issued']),
    first_source_endpoint_previously_dispatched_context_tick=next(r['episode_physics_tick'] for r in window if r['dispatch']['capture_assist_evidence']['context']['source_unfold_dispatched']),
    max_same_ACK_final_minus_mapped_nominal_minus_REQUEST_error=max(errors),max_dispatch_N_minus_previous_after_frame_N_error=max(timing_errors),
    all1201_actual12_residual_permission_one=True,all1201_assist_WAIT_no_correction=True,
    all1201_FL_crossed_false=all(not r['dispatch']['capture_assist_evidence']['context']['crossed_FL'] for r in window),
    all1201_pending_and_allow_false=all(not r['fl_capture_pending'] and not r['capture_continuation']['allow_capture_continuation'] for r in window),
    steady_30_883_to38=dict(n=len(steady),FL_front_distance_m=stat([r['current_legs']['FL']['front_distance_m'] for r in steady]),
        FL_gap_m=stat([r['current_legs']['FL']['clearance_m'] for r in steady]),
        all_FL_AIR=all(r['current_legs']['FL']['air'] for r in steady),all_lateral_legal=all(r['current_legs']['FL']['within_lateral_span'] for r in steady),
        all_current_physical_valid=all(r['dispatch']['capture_assist_evidence']['context']['physical_valid'] for r in steady),
        other_support_counts=sorted({r['dispatch']['capture_assist_evidence']['context']['other_support_count'] for r in steady}),
        support_margin_m=stat([r['support']['signed_margin_m'] for r in steady if r['support']['signed_margin_m'] is not None]),
        body_position_net_change_m=[b-a for a,b in zip(steady[0]['body']['position_w_m'],steady[-1]['body']['position_w_m'])],
        wheel_target_stats={leg:stat([r['dispatch']['drive_target_full12'][8+i] for r in steady]) for i,leg in enumerate(leg_order)},
        wheel_actual_stats={leg:stat([r['actual_full12'][8+i] for r in steady]) for i,leg in enumerate(leg_order)}),
    snapshots=snapshots,
    limitations=['Current-window-only diagnosis, not the final video outcome.',
        'Zero nominal is distinct from final zero; source stop does not disable residual authority.',
        'Mask, raw, REQUEST, final target and actual canonical velocity are separately reported; native joint IDs/velocities missing, not zero.',
        'Support and positive vertical gap do not certify a new forward path or future successful capture; no new controller authorized or tested.',
        'FR/RL tracking discrepancy under contact shows mechanical coupling/load response, not its uniquely identified physical cause.'])
print(json.dumps(result,indent=2,allow_nan=False))

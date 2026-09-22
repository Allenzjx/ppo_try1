"""After-seal finite FL diagnostic analysis only; never launch a simulator."""
import hashlib
import json
from pathlib import Path
from rr_probe_readonly import ROOT, lines, stats, compact_physical

OUT=Path(__file__).resolve().parent
PROBE=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/diagnostics/FL_hip_minus1_knee_plus1_CP199168_20260921_01'
BASE=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T1429210894242Z_g649ccd906421_0c5fc0498bbd4acca26ca3dde5d10193/source'
FIELDS=('base','joints','wheels','contacts','center_of_mass','body_bounds_w_m','obstacle','actual_full12')

def differences(a,b,path=''):
    if type(a)!=type(b):return [dict(path=path,probe=a,control=b)]
    if isinstance(a,dict):
        return [d for k in sorted(set(a)|set(b)) for d in differences(a.get(k),b.get(k),path+'.'+k)]
    if isinstance(a,list):
        if len(a)!=len(b):return [dict(path=path,probe_length=len(a),control_length=len(b))]
        return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+f'[{i}]')]
    return [] if a==b else [dict(path=path,probe=a,control=b)]

def compare_prefix(end):
    digests=[hashlib.sha256(),hashlib.sha256()];first=None;count=0;mismatch_count=0
    for a,b in zip(lines(PROBE/'physical_observations.jsonl'),lines(BASE/'physical_observations.jsonl')):
        assert a['physics_tick']==b['physics_tick']==count
        assert a['simulation_time_s']==b['simulation_time_s']
        aa={k:a[k] for k in FIELDS};bb={k:b[k] for k in FIELDS}
        for digest,value in zip(digests,(aa,bb)):
            digest.update(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        changed=[k for k in FIELDS if aa[k]!=bb[k]]
        if changed:
            mismatch_count+=1
            if first is None:first=dict(tick=count,fields=changed,first_differences=differences(aa,bb)[:8])
        if count==end:break
        count+=1
    assert count==end
    return dict(ticks=[0,end],samples=end+1,fields=list(FIELDS),exact_equal=first is None,
        mismatching_ticks=mismatch_count,first_difference=first,
        canonical_physical_subset_sha256=dict(probe=digests[0].hexdigest(),control=digests[1].hexdigest()),
        scope='selected actual physical fields at every identical tick/time; not assumed from headless/video labels')

def physics(path,end):
    rows=[]
    for r in lines(path/'physical_observations.jsonl'):
        t=r['physics_tick'];assert t<=end
        c=compact_physical(r)
        c['FL_actual']=[r['joints'][n]['position_deg'] for n in ('front_left_hip','front_left_knee')]
        c['FL_target']=[r['joints'][n]['command_deg'] for n in ('front_left_hip','front_left_knee')]
        rows.append(c)
        if t==end:break
    assert [r['tick'] for r in rows]==list(range(end+1))
    return rows

def current_leg(c):
    return {k:c.get(k) for k in ('air','ground_contact','top_surface_contact','bearing_verified','support',
        'bearing_force_n','within_top_xy','front_distance_m','clearance_m')}

def current_summary(rs):
    if not rs:return None
    return dict(endpoint_ticks=[rs[0]['end'],rs[-1]['end']],samples=len(rs),
        FL_TOP_verified_bearing=sum(r['FL']['top_surface_contact'] and r['FL']['bearing_verified'] and r['FL']['support'] for r in rs),
        FL_AIR=sum(r['FL']['air'] for r in rs),FL_ground=sum(r['FL']['ground_contact'] for r in rs),
        FL_gap_m=stats(r['FL']['clearance_m'] for r in rs),
        body_forward_change_m=rs[-1]['body_forward_m']-rs[0]['body_forward_m'],last_FL=rs[-1]['FL'])

def window(raw,lo,hi):
    if lo>hi or lo>=len(raw):return None
    hi=min(hi,len(raw)-1);rs=raw[lo:hi+1]
    return dict(ticks=[lo,hi],samples=len(rs),FL_gap_m=stats(r['gap']['FL'] for r in rs),
        FL_hip_actual_deg=stats(r['FL_actual'][0] for r in rs),FL_knee_actual_deg=stats(r['FL_actual'][1] for r in rs),
        body_collider_min_world_z_m=stats(r['body_min_z'] for r in rs),
        conservative_AABB_distance_m=stats(r['separation'] for r in rs),
        actual_pair_contact_samples={leg:{kind:sum(r['contacts'][leg][kind]['active'] for r in rs)
            for kind in ('ground','obstacle')} for leg in ('FL','FR','RL','RR')},
        FL_obstacle_pair_samples=sum(r['contacts']['FL']['obstacle']['active'] for r in rs),
        FL_ground_pair_samples=sum(r['contacts']['FL']['ground']['active'] for r in rs))

def main():
    # Refuse an active run; a real sealed error remains an error, never success.
    m=json.loads((PROBE/'run_manifest.json').read_text())
    assert m['lifecycle'] in ('DIAGNOSTIC_SEALED','DIAGNOSTIC_ERROR')
    bm=json.loads((BASE/'semantic_video_source_manifest.json').read_text())
    end=m.get('endpoint_tick',m.get('last_core_frame_tick'))
    assert isinstance(end,int),'sealed manifest lacks an authoritative endpoint'
    entry_path=PROBE/'probe_entry.json';entry=json.loads(entry_path.read_text()) if entry_path.exists() else None
    pre_entry=None if entry is None else entry['pre_action_receipt']
    start=None if pre_entry is None else pre_entry['start_tick']
    same_runtime=bm['runtime_contract']==m['runtime_contract']
    same_state=bm['checkpoint_load_provenance']['parameter_hashes']==m['checkpoint_load_provenance']['parameter_hashes']
    prefix=None if start is None else compare_prefix(start)
    rows=[];active=[];pending={};checks=[];last_history=None
    for r in lines(PROBE/'probe_decisions.jsonl'):
        if r['record_kind']=='pre_action':
            hashed={k:v for k,v in r.items() if k!='pre_action_receipt_sha256'}
            assert hashlib.sha256(json.dumps(hashed,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==r['pre_action_receipt_sha256']
            pending[r['decision']]=r;continue
        if r['record_kind']!='step_result':continue
        p=pending.pop(r['decision']);assert p['pre_action_receipt_sha256']==r['pre_action_receipt_sha256']
        s=r['step_info'];task=s['semantic_task'];ev=task['physical_evaluator'];a=s['actuator_target_effect_audit'];h=a['policy_headroom_evidence']
        intervention=p.get('intervention') or {};index=intervention.get('index')
        label='before' if start is None or p['start_tick']<start else 'after_probe'
        if index is not None:
            release=intervention['release_age']
            label=('ramp' if index<12 else 'hold') if release is None else ('release' if index-release<12 else 'follow')
            active.append(dict(start=p['start_tick'],end=r['end_tick'],index=index,label=label,
                requested_selected_deg=intervention['requested_selected_deg'],release_age=release,capture_age=intervention['capture_age']))
        checks.append(dict(other10_zero=all(v==0 for v in p['manual_raw_delta_full12'][2:]),
            masks_one=p['combined_residual_permission_mask_full12']==[1]*12,
            no_credit=all(p[k]==0 for k in ('new_PPO_decisions','new_PPO_updates','new_optimizer_steps','new_auxiliary_updates','positive_auxiliary_labels_emitted')),
            no_manual_likelihood=p['manually_issued_action_policy_log_probability'] is None,
            live_H_equals_ACK=(p['actual_live_history']['previous_residual_full12']==
                p['previous_actual_ACK'].get('independent_policy_residual_requested_full12')) if p['start_tick'] else None,
            clipped=h['clipped_servo_indices'],request_effective_delta=max(abs(x-y) for x,y in
                zip(h['requested_policy_residual_full12'][:2],h['effective_policy_residual_full12'][:2])),
            dispatch_verified=a['verified'] and a['setter_dispatch_targets_equal'] and a['actual_mapping_matches_dispatch']))
        g=r['four_hip_geometry'];assert g['clock_unchanged'] and g['frame_clock_aligned'] and g['physics_tick']==r['end_tick']
        rows.append(dict(start=p['start_tick'],end=r['end_tick'],phase=s['phase_id'],end_phase=s['end_phase_id'],label=label,
            FL=current_leg(ev['current_legs']['FL']),body_forward_m=ev['goal_features']['body_forward_m'],
            N=s['nominal_action_full12'][:2],mapped=h['baseline_native_plus_controller_full12'][:2],
            REQUEST=h['requested_policy_residual_full12'][:2],effective=h['effective_policy_residual_full12'][:2],
            final=s['actual_drive_target_full12'][:2],hip_mount_z_m={leg:g['hip_mount_world_m'][leg]['value'][2]
                if g['hip_mount_world_m'][leg]['value'] is not None else None for leg in ('FL','FR','RL','RR')}))
        last_history=task['history']
    raw=physics(PROBE,end)
    # Baseline has a real finite endpoint; overlap only, never extend it with zeros.
    control_end=bm['physical_episode']['physical_task_evaluation']['physics_tick']
    overlap=min(end,control_end);control=physics(BASE,overlap)
    ranges={label:(min(r['start'] for r in rows if r['label']==label)+1,max(r['end'] for r in rows if r['label']==label))
            for label in ('ramp','hold','release','follow','after_probe') if any(r['label']==label for r in rows)}
    placed=None if last_history is None else last_history['event_ticks']['placed'].get('FL')
    first_contact=next((r['tick'] for r in raw if start is not None and r['tick']>start and r['contacts']['FL']['obstacle']['active']),None)
    release_start=next((r['start'] for r in rows if r['label']=='release'),None)
    selected={r['end'] for r in rows if r['label'] in ranges and (r['start']+1,r['end'])==ranges[r['label']]}
    for label in ranges:
        rs=[r for r in rows if r['label']==label];selected.update((rs[0]['end'],rs[-1]['end']))
    if placed is not None:
        after=next((r for r in rows if r['end']>=placed),None)
        if after:selected.add(after['end'])
    examples=[{**r,'actual_FL':raw[r['end']]['FL_actual'],'body_min_z_m':raw[r['end']]['body_min_z']} for r in rows if r['end'] in selected]
    result=dict(schema='CP199168.sealed_FL_hip_knee_diagnostic_readonly.v1',probe=str(PROBE),control=str(BASE),
        sealed={k:m.get(k) for k in ('lifecycle','endpoint_tick','final_phase','original_task_reason','probe_complete','external_budget_stop','learned_state_unchanged','actual_diagnostic_decisions_completed')},
        same_CP_parameter_hashes=same_state,same_runtime=same_runtime,preintervention_physical_prefix=prefix,
        direct_entry=None if pre_entry is None else dict(tick=start,anchor=pre_entry['intervention']['anchor_full12'][:2],
            ACK_REQUEST=pre_entry['previous_actual_ACK']['independent_policy_residual_requested_full12'][:2],
            live_HISTORY=pre_entry['actual_live_history']['previous_residual_full12'][:2],
            nominal_selected=pre_entry['pre_source_nominal_full12'][:2],receipt_sha256=pre_entry['pre_action_receipt_sha256']),
        primary_outcome=dict(first_FL_obstacle_pair_contact_tick=first_contact,FL_qualified_placed_tick=placed,
            P06_current_retention=current_summary([r for r in rows if r['phase']=='P06']),
            after_release_current_retention=None if release_start is None else current_summary([r for r in rows if r['start']>=release_start+96])),
        timeline_by_label={label:dict(ticks=list(bounds),decisions=sum(r['label']==label for r in rows)) for label,bounds in ranges.items()},
        physical_windows={label:dict(probe=window(raw,lo,hi),control_same_elapsed=window(control,lo,hi),
            comparison='matched elapsed time only after intervention; states/policy feedback differ, not same-state counterfactual')
            for label,(lo,hi) in ranges.items()},selected_same_dispatch_examples=examples,
        checks=dict(paired_completed_decisions=len(rows),unpaired_preactions=list(pending),
            other10_manual_delta_zero=all(c['other10_zero'] for c in checks),masks_all_one=all(c['masks_one'] for c in checks),
            all_dispatch_verified=all(c['dispatch_verified'] for c in checks),no_PPO_AUX_credit=all(c['no_credit'] for c in checks),
            no_manual_policy_likelihood=all(c['no_manual_likelihood'] for c in checks),
            all_noninitial_live_REQUEST_H_matches_ACK=all(c['live_H_equals_ACK'] for c in checks if c['live_H_equals_ACK'] is not None),
            selected_clip_decisions=sum(any(i in (0,1) for i in c['clipped']) for c in checks),
            selected_REQUEST_effective_max_difference=max((c['request_effective_delta'] for c in checks),default=None)),
        events=None if last_history is None else last_history['event_ticks'],
        endpoint=None if not rows else rows[-1],new_PPO_decisions=0,new_PPO_updates=0,new_auxiliary_updates=0,
        limitations='No capture inferred from minimum gap. Historical placed not current support. Same elapsed post-intervention rows are separate trajectories; headless/video prefix equality is measured explicitly. Manual intervention not a learned-policy success or positive auxiliary label.')
    with (OUT/'CP199168_FL_hip_knee_evidence.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('sealed','same_CP_parameter_hashes','same_runtime','preintervention_physical_prefix','direct_entry','primary_outcome','timeline_by_label','checks','events')},indent=2))

if __name__=='__main__':main()

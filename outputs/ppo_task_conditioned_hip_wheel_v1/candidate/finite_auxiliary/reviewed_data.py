"""Read a sealed physical review and its original pre/post byte ranges only."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import struct
from finite_auxiliary_mean import require, LABEL_SCOPE


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]


def validate_pair(pre,post,row,*,training):
    require(pre.get('record_kind')=='pre_action' and post.get('record_kind')=='step_result'
        and post.get('environment_step_returned') is True, 'not a complete real pre/post pair')
    require(pre['decision']==post['decision']==row['decision'] and pre['start_tick']==post['start_tick']==row['start_tick']
        and post['end_tick']==row['end_tick']==row['start_tick']+8, 'pre/post decision/tick join differs')
    copy=dict(pre);digest=copy.pop('pre_action_receipt_sha256')
    require(hashlib.sha256(json.dumps(copy,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        ==digest==post['pre_action_receipt_sha256']==row['pre_action_receipt_sha256'], 'pre-action receipt hash mismatch')
    obs=pre['actor_input_float32_372'];encoded=pre['observation_encoded_372']
    require(len(obs)==len(encoded)==372 and all(math.isfinite(v) and abs(v)<=20 for v in obs)
        and obs==[f32(v) for v in encoded], 'original encoded372/float32 inputs differ')
    require(hashlib.sha256(struct.pack('<372f',*obs)).hexdigest()
        ==pre['actor_input_float32_le_sha256']==row['actor_input_float32_le_sha256'], 'actor input hash differs')
    for name,group in pre['observation_history_groups'].items():
        scales=group['scale'];scales=[scales]*12 if isinstance(scales,(int,float)) else scales
        expected=[f32(max(-20,min(20,x/s))) for x,s in zip(pre['actual_live_history'][name],scales,strict=True)]
        require(obs[group['start']:group['stop_exclusive']]==expected, 'real HISTORY encoding mismatch: '+name)
    history,ack=pre['actual_live_history'],pre['previous_actual_ACK']
    if 'independent_policy_residual_requested_full12' in ack:
        require(history['previous_residual_full12']==ack['independent_policy_residual_requested_full12'],
            'previous REQUEST history differs from actual ACK')
    else:
        require(pre['decision']==pre['start_tick']==0 and row['initial_reset_observation'] is True
            and history['previous_residual_full12']==[0.]*12, 'missing previous REQUEST outside actual initial reset')
    require(history['previous_applied_full12']==ack['drive_target_full12']
        and post['step_info']['no_in_episode_state_writes_verified'] is True,
        'previous actual target differs from ACK or state-write proof is missing')
    q=pre['original_policy_request_same_live_state']
    baseline,issued=pre['policy_baseline_raw_full12'],pre['manually_selected_raw_full12']
    require(baseline==q['conditional_mean_full12']==q['selected_raw_full12']
        and issued==post['step_info']['raw_policy_action_full12'] and len(issued)==12
        and all(math.isfinite(v) for v in issued), 'issued action is not original logged manual/raw action')
    require(issued[1:]==baseline[1:] and pre['manual_override_selector_full12'][1:]==[0]*11,
        'diagnostic changed or supervised another action channel')
    require(all(pre[name]==[1.]*12 for name in ('phase_residual_permission_mask_full12',
        'runtime_residual_permission_mask_full12','safety_residual_permission_mask_full12',
        'combined_residual_permission_mask_full12')), 'residual permission is not full12')
    require(q['previous_raw_from_current_observation_full12']==obs[195:207], 'actor HISTORY binding differs')
    require(pre['diagnostic_only'] is True and all(pre[k]==0 for k in ('new_PPO_decisions','new_PPO_updates','new_optimizer_steps')),
        'diagnostic data must have zero PPO credit')
    if training:
        require(pre['phase']=='P05' and row['eligible_for_gap_approach'] is True and row['reason']==[]
            and row['capture_label'] is False and bool(row['checks']) and all(v is True for v in row['checks'].values()),
            'unreviewed or falsely capture-labeled sample')
        intervention=pre['intervention'];index=row['intervention_index']
        require(intervention['case']=='FL_minus6' and intervention['index']==index and 0<=index<75
            and intervention['delta_deg']=={'0':-6.0} and intervention['capture_age'] is None
            and intervention['release_age'] is None and intervention['not_added_recursively_to_HISTORY'] is True
            and row['segment']==('ramp' if index<12 else 'hold'), 'changed diagnostic intervention or window segment')
        require(row['selected_hip_raw_issued']==issued[0], 'review labels differ from actual issued FL action')
        ev=post['step_info']['semantic_task']['physical_evaluator'];fl=ev['current_legs']['FL']
        require(ev['valid'] is True and not ev.get('termination_reason') and fl['air'] is True
            and fl['within_top_xy'] is True and fl['clearance_m']>0 and not fl['top_contact'],
            'post sample is not valid legal AIR approach')
        require(abs(1000*fl['clearance_m']-row['FL_gap_pre_post_mm'][1])<1e-8, 'reviewed gap differs from the original post')
    else:
        require(pre['phase']!='P05' and row['auxiliary_label'] is False
            and issued==baseline and pre['manual_override_selector_full12']==[0]*12,
            'holdout must be real non-target unmodified-policy data')
    return obs,issued[0]


def load_reviewed_selection(path):
    """No physics inference and no state rewriting; keep the entire reviewed window."""
    path=Path(path).resolve(strict=True);selection=json.loads(path.read_text(encoding='utf-8'))
    require(selection.get('schema')=='wlr50_clean.FL_minus6_real_AIR_gap_approach_selection.v1'
        and selection.get('label_scope')==LABEL_SCOPE and selection.get('training_executed') is False
        and selection.get('other_action_channels_supervised') is False, 'unsupported reviewed selection')
    source=selection['source_files'];run=Path(selection['run_dir']).resolve(strict=True)
    require(set(source)=={'run_manifest.json','probe_decisions.jsonl','physical_observations.jsonl',
        'native_tick_audit.jsonl','probe_entry.json'}, 'review lacks exact source bindings')
    for name,record in source.items():
        file=Path(record['path']).resolve(strict=True)
        require(file==run/name and sha(file)==record['sha256'], 'sealed diagnostic source hash/path changed: '+name)
    manifest=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
    require(manifest.get('lifecycle')=='DIAGNOSTIC_SEALED' and manifest.get('error') is None
        and manifest.get('training_data_eligible') is False and manifest.get('auxiliary_updates') is False,
        'diagnostic is incomplete, failed at interface, or already treated as training')
    rows=selection['rows'];holdout=selection['holdout_non_P05'];indices=selection['decision_indices']
    entry=json.loads((run/'probe_entry.json').read_text(encoding='utf-8'))['pre_action_receipt']
    require(len(rows)==len(indices)==75 and indices==list(range(entry['decision'],entry['decision']+75))
        and [r['intervention_index'] for r in rows]==list(range(75))
        and rows[0]['start_tick']==entry['start_tick'] and rows[-1]['end_tick']==entry['start_tick']+600
        and [r['decision'] for r in rows]==indices and 1<=len(holdout)<=128
        and not set(indices)&{r['decision'] for r in holdout}, 'selection must retain one whole contiguous window and separate holdout')
    result=[];holds=[];targets=[]
    with (run/'probe_decisions.jsonl').open('rb') as stream:
        for group,is_training in ((rows,True),(holdout,False)):
            for row in group:
                pair=[]
                for kind in ('pre','post'):
                    location=row[kind]
                    require(type(location['byte_offset']) is int and location['byte_offset']>=0
                        and type(location['byte_length']) is int and 0<location['byte_length']<=2_000_000,
                        'invalid bounded source byte range')
                    stream.seek(location['byte_offset']);raw=stream.read(location['byte_length'])
                    require(raw.endswith(b'\n'),'source byte range is not a complete line')
                    pair.append(json.loads(raw))
                obs,target=validate_pair(*pair,row,training=is_training)
                if is_training:result.append(obs);targets.append(target)
                else:holds.append(obs)
    receipt={'selection_path':str(path),'selection_sha256':sha(path),'source_files':source,
        'diagnostic_checkpoint_sha256':manifest['checkpoint_sha256'],'diagnostic_runtime_contract':manifest['runtime_contract'],
        'diagnostic_policy_contract':manifest['policy_contract'],'decision_indices':indices,
        'holdout_decision_indices':[r['decision'] for r in holdout], 'label_scope':LABEL_SCOPE,
        'capture_labels':False,'release_tail_retained_in_source':True,'window_summary':selection['window_summary'],
        'PPO_credit':0,'other_channels_supervised':False}
    return result,targets,holds,receipt

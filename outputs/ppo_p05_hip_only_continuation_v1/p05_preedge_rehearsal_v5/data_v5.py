"""DATA ONLY: direct same-v2 P05 approach, excluding every assisted action.

No fit, budget, actor construction, checkpoint write, or live observation/raw
reconstruction. A later assisted FL placement is continuation evidence only.
"""
from collections import Counter
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent;OUT=HERE.parent;ROOT=OUT.parents[1]
MANIFEST=HERE/'data_manifest.json';DATA=HERE/'candidate_data.npz'
SOURCE=OUT/'checkpoints/history/checkpoint_step_000216448.pt'
SOURCE_SHA='8ec7784a9078ae7e9bb5f1d7298a4aa655c0d979ddb906e8a4d577bf8079a8f4'
RUN=ROOT/'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1428131775302Z_g336b7c56d2f0_23b26a4c40a24a178d190def03381413'
LIVE=ROOT/'runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/20260922T1521094182530Z_g336b7c56d2f0_3cdf838115e44829b66ad9585ad92ba7'
SCHEMA='wlr50_clean.P05_preedge_unassisted_actual_raw_data.v5'
COUNTERS=('global_policy_decisions','ppo_updates','optimizer_steps')


def require(value,message):
    if not bool(value):raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def binding(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def wait_uninitialized(snapshot):
    return snapshot.get('mode')==0 and snapshot.get('initialized')==0 and snapshot.get('mode_name')=='WAIT'


def unassisted_entire_interval(input_snapshot,evidence):
    # initialized latches until episode reset; it never returns to0 within an
    # episode. End initialized0 + adjacent startWAIT rules out hidden N1 owners.
    return (wait_uninitialized(input_snapshot) and wait_uninitialized(evidence['state_before'])
        and wait_uninitialized(evidence['state_after']) and evidence['owner_indices']==[]
        and evidence['candidate_before_assist_full12']==evidence['candidate_after_assist_full12'])


def stats(values):
    values=list(values)
    return {'min':min(values),'max':max(values),'mean':sum(values)/len(values)} if values else None


def physical_state(ev):
    from wlr50_clean.infrastructure.command_batch import servo_limits_deg
    joints={}
    for role in ev['transfer_roles'].values():
        for name,margin in role['receiver_workspace_state']['joint_range_margin_deg'].items():
            low,high=servo_limits_deg(name);actual=low+margin['negative_deg']
            require(abs(actual-(high-margin['positive_deg']))<1e-9,'actual joint margin inversion differs')
            joints[name]=actual
    fl=ev['current_legs']['FL']
    return {'tick':ev['physics_tick'],'time_s':ev['physics_tick']/120,
        'FL_gap_mm':1000*fl['clearance_m'],'FL_front_mm':1000*fl['front_distance_m'],
        'FL_lift_history':bool(ev['history']['active_lift']['FL']),'FL_crossed_history':bool(ev['history']['front_edge_crossed']['FL']),
        'FL_placed_history':bool(ev['history']['placed']['FL']),'FL_legal_XY':bool(fl['within_top_xy'] and fl['within_lateral_span']),
        'body_collider_min_z_mm':1000*ev['body_traversal_geometry']['minimum_w_m'][2],
        'actual_joints_deg':joints,'support':{leg:bool(r['support'] and r['bearing_verified']) for leg,r in ev['current_legs'].items()},
        'contact_surface':{leg:r['contact_surface'] for leg,r in ev['current_legs'].items()}}


def physical_summary(rows):
    return {'rows':len(rows),'time_s':stats(r['time_s'] for r in rows),'FL_gap_mm':stats(r['FL_gap_mm'] for r in rows),
        'FL_front_mm':stats(r['FL_front_mm'] for r in rows),'body_collider_min_z_mm':stats(r['body_collider_min_z_mm'] for r in rows),
        'FL_lift_history':sum(r['FL_lift_history'] for r in rows),'FL_crossed_history':sum(r['FL_crossed_history'] for r in rows),
        'support_counts':{leg:sum(r['support'][leg] for r in rows) for leg in ('FL','FR','RL','RR')},
        'actual_joints_deg':{key:stats(r['actual_joints_deg'][key] for r in rows) for key in rows[0]['actual_joints_deg']} if rows else {}}


def split_by_episode(episodes):
    train=[];validation=[]
    for ep in sorted(set(episodes.tolist())):
        ids=np.flatnonzero(episodes==ep).tolist()
        for offset,index in enumerate(ids):
            (validation if offset%3==1 else train).append(index)
    return np.asarray(sorted(train),dtype=np.int64),np.asarray(sorted(validation),dtype=np.int64)


def check_tensor_row(batch,offset,row):
    x=batch['observations']['policy'][offset,0];a=row['applied_audit'];native=a['actuator_target_effect_audit'];p=row['policy_request']
    require(x.shape==(389,) and x.dtype==torch.float32 and torch.isfinite(x).all()
        and torch.equal(x,batch['observations']['critic'][offset,0]),'direct389 invalid')
    require(int(x[:13].argmax())==int(a['phase_id'][1:])-1,'source phase differs')
    raw=batch['actions'][offset,0];mean,std=(t[offset,0] for t in batch['distribution_params']);logp=batch['actions_log_prob'][offset,0].reshape(())
    for tensor,key in ((raw,'raw_policy_action_full12'),(mean,'old_distribution_mean_full12'),(std,'old_distribution_std_full12'),(logp,'old_log_probability')):
        require(torch.equal(tensor,torch.tensor(row[key],dtype=tensor.dtype)),'actual raw/distribution source mismatch')
    err=float((torch.distributions.Normal(mean,std).log_prob(raw).sum()-logp).abs())
    require(err<=1e-5,'actual sampled likelihood differs')
    require(p['sampling_draws']==1 and p['extra_random_draws']==0 and row['raw_policy_action_full12']==p['selected_raw_full12']==native['raw_policy_action_full12'],'not original raw draw')
    require(native['verified'] and native['actual_mapping_matches_dispatch'] and native['phase_mask_full12']==[1]*12
        and a['actuator_target_effect_audit_summary']['all_ticks_verified'] and a['no_in_episode_state_writes_verified'],'native all12 execution invalid')
    require(a['semantic_task']['physical_evaluator']['valid'] and a['semantic_task']['physical_evaluator']['termination_reason'] is None
        and not row['terminal'],'physical invalid/terminal selected input')
    require(np.float32(a['reward_breakdown']['potential_before'])==x[17].item(),'same-v2 directX17 mismatch')
    for begin,end,key in ((372,384,'capture_assist_observed_features'),(384,389,'capture_continuation_observed_features')):
        require(torch.equal(x[begin:end],torch.tensor(p[key],dtype=torch.float32)),'observable assist/continuation mismatch')
    return x,raw,mean,std,logp,err


def validate_arrays(a):
    x=a['observations389'];n=len(x)
    require(n>0 and x.shape==(n,389) and a['actual_raw12'].shape==(n,12),'candidate shape invalid')
    for key in ('observations389','actual_raw12','source_mean12','source_sigma12','source_logp','protection_observations'):
        require(a[key].dtype==np.float32 and np.isfinite(a[key]).all(),'nonfinite/nonfloat32 '+key)
    require(np.all(x[:,4]==1) and np.all(x[:,:13].sum(-1)==1),'positive rows must be P05 only')
    require(np.all(x[:,372:384]==0),'positive action input must be uninitialized WAIT')
    require(np.all(a['source_sigma12']>0) and not np.any(np.all(a['actual_raw12']==a['source_mean12'],axis=1)),'stochastic raw replaced with stored mean')
    train,val=split_by_episode(a['episode_indices'])
    require(np.array_equal(train,a['train_indices']) and np.array_equal(val,a['validation_indices']),'fixed same-episode interleave changed')
    require(not set(train)&set(val) and len(train)+len(val)==n,'split drops or overlaps rows')
    p=a['protection_observations']
    require(p.shape==(8,389) and set(p[:,:13].argmax(-1)+1)=={4,6,7,8,9,10,11,12} and np.all(p[:,4]==0),'nonP05 real protection invalid')


def live_coverage(source_states):
    started=read(LIVE/'run_manifest.started.json')
    require(Path(started['arguments']['checkpoint']).resolve()==SOURCE.resolve(),'wrong live evaluated checkpoint')
    path=LIVE/'source/video_policy_decisions.jsonl';rows=[];line_hash=hashlib.sha256();read_bytes=0;read_lines=0
    # Exactly one filesystem snapshot, stop at first endpoint beyond38s.
    with path.open('rb') as stream:
        stream.seek(0,2);size=stream.tell();stream.seek(0)
        while stream.tell()<size:
            line=stream.readline()
            if not line.endswith(b'\n') or stream.tell()>size:break
            record=json.loads(line);a=record['step_info'];a=a[-1] if isinstance(a,list) else a
            read_bytes+=len(line);read_lines+=1;line_hash.update(line)
            if a['sim_time_s']>38:break
            if a['sim_time_s']<28:continue
            require(a['phase_id']=='P05','fixed live window left P05; do not silently widen')
            ev=a['semantic_task']['physical_evaluator'];require(ev['valid'] and ev['termination_reason'] is None,'invalid live state')
            state=physical_state(ev);native=a['actuator_target_effect_audit'];assist=native['capture_assist_evidence']
            state.update(decision=record['decision'],source_nominal_wheel_target=a['nominal_action_full12'][8:],
                logged_raw_wheel_request=record['raw_policy_action_full12'][8:],final_wheel_target=a['actual_drive_target_full12'][8:],
                assist_initialized=assist['state_after']['initialized'],assist_owners=assist['owner_indices'])
            rows.append(state)
    require(rows,'no complete fixed live28..38s records available; no polling')
    current=rows[-1]
    near=[r for r in source_states if abs(r['FL_front_mm']-current['FL_front_mm'])<=10 and abs(r['FL_gap_mm']-current['FL_gap_mm'])<=10]
    snapshots=[]
    for time in (28.,29.0666666667,30.,34.,38.):
        chosen=min(rows,key=lambda r:abs(r['time_s']-time))
        if chosen not in snapshots:snapshots.append(chosen)
    return {'source':str(path),'fixed_endpoint_window_s':[28,38],'snapshot_file_bytes_available':size,
        'bounded_prefix_bytes_read':read_bytes,'bounded_prefix_lines_read':read_lines,'bounded_prefix_sha256':line_hash.hexdigest(),
        'last_available_time_s':current['time_s'],'no_future_polling_or_observation_reconstruction':True,
        'direct_logged_physical_states_only_not_actor_X389_or_AUX_labels':True,
        'live_summary':physical_summary(rows),'source_all_selected_summary':physical_summary(source_states),
        'source_last3s_each_episode_before_assist_summary':physical_summary([r for r in source_states
            if r['time_s']>=max(s['time_s'] for s in source_states if s['episode_index']==r['episode_index'])-3]),
        'source_near_last_live_FL_front_and_gap_within10mm':physical_summary(near),
        'representative_live_states':snapshots,'live_source_nominal_allwheel_zero_rows':sum(all(v==0 for v in r['source_nominal_wheel_target']) for r in rows),
        'live_final_wheel_any_nonzero_rows':sum(any(abs(v)>1e-6 for v in r['final_wheel_target']) for r in rows),
        'live_assist_ever_initialized_in_window':any(r['assist_initialized'] for r in rows),
        'mask_bug_or_single_wheel_cause_claimed':False,'old_action_current_success_or_causal_fix_claimed':False}


def build():
    require(not torch.cuda.is_available(),'CPU-only preparation');torch.set_num_threads(1)
    require(not MANIFEST.exists() and not DATA.exists(),'do not overwrite candidate')
    smpath=SOURCE.with_name(SOURCE.stem+'_manifest.json');sm=read(smpath)
    require(sha(SOURCE)==sm['checkpoint_sha256']==SOURCE_SHA and tuple(sm[k] for k in COUNTERS)==(216448,1656,33120),'source binding mismatch')
    sealed=read(RUN/'training_manifest.json');require(sealed['lifecycle']=='SUCCEEDED' and sealed['actual_policy_decisions']==2048,'source run not sealed')
    updates={r['ppo_update']:r for r in map(json.loads,(RUN/'optimizer_updates.jsonl').read_text().splitlines())}
    batches={u:torch.load(RUN/f'rollouts/rollout_{u:06}.pt',map_location='cpu',weights_only=False) for u in range(1641,1657)}
    require(all(b['runtime_contract']==sm['runtime_contract'] and b['policy_contract']==sm['policy_contract'] for b in batches.values()),'source data not exact same v2 codec/kernel')
    candidates=[];states=[];records=[];protection={};episodes=[];prior=None;exclusions=Counter();bindings={}
    with (RUN/'residual_and_projection_audit.jsonl').open('rb') as stream:
        for index,line in enumerate(itertools.islice(stream,2048)):
            row=json.loads(line);a=row['applied_audit'];ev=a['semantic_task']['physical_evaluator'];native=a['actuator_target_effect_audit'];assist=native['capture_assist_evidence']
            if a['decision_count']==1:
                episodes.append({'episode_index':len(episodes),'P05_all_rows':0,'candidate_rows':0,'first_P06_input_tick':None,
                    'first_FL_crossed_endpoint':None,'first_FL_placed_endpoint':None,'excluded_assisted_or_not_provably_unassisted':0})
                prior=None
            ep=episodes[-1];epno=ep['episode_index'];update=1641+index//128;offset=index%128
            for flag,target in (('front_edge_crossed','first_FL_crossed_endpoint'),('placed','first_FL_placed_endpoint')):
                if ev['history'][flag]['FL'] and ep[target] is None:
                    ep[target]={'event_tick':ev['history']['event_ticks'][flag]['FL'],'endpoint_tick':a['physics_tick'],
                        'global_decision':row['global_policy_decision'],'assist_owner_indices':assist['owner_indices'],
                        'assist_initialized':assist['state_after']['initialized'],'FL_current_TOP':ev['current_legs']['FL']['top_contact'],
                        'FL_current_surface':ev['current_legs']['FL']['contact_surface']}
            ep['FL_lift_tick']=ev['history']['event_ticks']['active_lift'].get('FL')
            if a['phase_id']=='P06' and ep['first_P06_input_tick'] is None:ep['first_P06_input_tick']=a['physics_tick']-a['physics_ticks']
            ep.update(later_episode_terminal=bool(row['terminal']),later_episode_result=a['termination_reason'],last_phase=a['end_phase_id'])
            if a['phase_id']=='P05':
                ep['P05_all_rows']+=1
                require(prior is not None,'P05 has no recorded exact preceding input state')
                previous_native=prior['applied_audit']['actuator_target_effect_audit'];input_assist=previous_native['capture_assist_evidence']['state_after']
                if not unassisted_entire_interval(input_assist,assist):
                    ep['excluded_assisted_or_not_provably_unassisted']+=1;exclusions['assisted_or_initialized_interval']+=1
                else:
                    x,raw,mean,std,logp,err=check_tensor_row(batches[update],offset,row)
                    input_ev=prior['applied_audit']['semantic_task']['physical_evaluator'];tick=a['physics_tick']-a['physics_ticks']
                    require(tick==input_ev['physics_tick'] and input_ev['valid'] and input_ev['termination_reason'] is None and not prior['terminal'],'invalid/discontinuous source input')
                    require(bool((x[372:384]==0).all()),'input wait state is not allzero')
                    state=physical_state(input_ev);state['episode_index']=epno;states.append(state)
                    candidates.append((x,raw,mean,std,logp));ep['candidate_rows']+=1
                    records.append({'source_index':index,'episode_index':epno,'global_decision':row['global_policy_decision'],'input_tick':tick,
                        'rollout_update':update,'rollout_offset':offset,'source_audit_row_sha256':hashlib.sha256(line).hexdigest(),
                        'whole_N1_unassisted_by_monotonic_uninitialized_WAIT':True,'owners':[],
                        'FL_front_mm':state['FL_front_mm'],'FL_gap_mm':state['FL_gap_mm'],'FL_crossed_history':state['FL_crossed_history'],
                        'source_logp_recompute_error':err,'nominal_wheel_target':a['nominal_action_full12'][8:],
                        'final_wheel_target':a['actual_drive_target_full12'][8:]})
            elif a['phase_id'] not in protection and not row['terminal'] and ev['valid'] and ev['termination_reason'] is None:
                x,*_=check_tensor_row(batches[update],offset,row)
                protection[a['phase_id']]={'observation':x,'record':{'phase':a['phase_id'],'source_index':index,
                    'episode_index':epno,'global_decision':row['global_policy_decision'],'rollout_update':update,'rollout_offset':offset,
                    'original_direct_current389':True,'action_label':False}}
            prior=row
    require(len(episodes)==3 and all(ep['first_FL_crossed_endpoint'] and ep['first_FL_placed_endpoint'] and ep['first_P06_input_tick'] for ep in episodes),'not all selected episodes have real local approach/capture/continuation')
    require(sum(ep['P05_all_rows'] for ep in episodes)==590,'unexpected source P05 population')
    require(sum(ep['candidate_rows'] for ep in episodes)==len(candidates),'selection bookkeeping mismatch')
    for ep in episodes:
        ep['local_qualification']='actual FL crossing and later placed plus actual P06 continuation; capture may be assisted'
        ep['later_episode_failure_not_relabelled_success']=True
    selected_updates=sorted({r['rollout_update'] for r in records}|{r['record']['rollout_update'] for r in protection.values()})
    for update in selected_updates:
        before=214400+(update-1641)*128
        cp=OUT/f'checkpoints/history/checkpoint_step_{before:09}.pt' if update!=1641 else OUT/'checkpoints/history/checkpoint_rr_receiver_v2_step_000214400.pt'
        meta=read(cp.with_name(cp.stem+'_manifest.json'))
        require(sha(cp)==meta['checkpoint_sha256'] and meta['actor_parameter_sha256']==updates[update]['actor_parameter_sha256_before'],'collecting model binding differs')
        require(meta['runtime_contract']==sm['runtime_contract'] and len(meta['training_rng_state']['torch_cuda'])==1,'collecting semantics/RNG differ')
        bindings[str(update)]={'rollout':binding(RUN/f'rollouts/rollout_{update:06}.pt'),'collection_checkpoint':binding(cp),
            'source_actor_sha256':meta['actor_parameter_sha256'],'boundary_full_RNG_sha256':digest(meta['training_rng_state']),
            'per_action_RNG_replay_claimed':False}
    keys=('observations389','actual_raw12','source_mean12','source_sigma12','source_logp')
    arrays={key:torch.stack([r[i] for r in candidates]).numpy() for i,key in enumerate(keys)}
    arrays['episode_indices']=np.array([r['episode_index'] for r in records],dtype=np.int64)
    arrays['source_indices']=np.array([r['source_index'] for r in records],dtype=np.int64)
    arrays['input_ticks']=np.array([r['input_tick'] for r in records],dtype=np.int64)
    arrays['train_indices'],arrays['validation_indices']=split_by_episode(arrays['episode_indices'])
    order=sorted(protection);arrays['protection_observations']=torch.stack([protection[p]['observation'] for p in order]).numpy()
    validate_arrays(arrays)
    coverage=live_coverage(states)
    np.savez_compressed(DATA,**arrays)
    ledger=sm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    manifest={'schema':SCHEMA,'status':'DATA_CANDIDATE_AND_P05_COLUMN_INSPECTION_ONLY_NOT_AUX_AUTHORIZATION',
        'source_checkpoint':binding(SOURCE),'source_manifest':binding(smpath),'source_actor_sha256':sm['actor_parameter_sha256'],
        'runtime_head':sm['runtime_contract']['source_git_commit'],'runtime_contract_sha256':digest(sm['runtime_contract']),
        'policy_contract_sha256':digest(sm['policy_contract']),'source_PPO_counts':{k:sm[k] for k in COUNTERS},
        'current_aux_ledger':{'entire_four_event_object_sha256':digest(ledger),'mixed_accepted_attempted':[103,104],
            'front':[96,96],'RR':[7,8],'older_separate':[7,8]},'dataset':binding(DATA),'loader':binding(Path(__file__)),
        'source_run':str(RUN),'source_original_direct389':True,'X17_remapping_performed':False,
        'source_same_v2_runtime_and_policy_contract':True,'actual_raw12_labels_not_source_mu_or_transformed_target':True,
        'rows':len(candidates),'train_rows':len(arrays['train_indices']),'validation_rows':len(arrays['validation_indices']),
        'selection':'All valid P05 actions of three locally qualified trajectories with whole-N1 uninitialized WAIT; no value/action/outcome cherry-picking within that scope',
        'split':'Within each episode eligible source-order offsets0,2 mod3 train;1 validation; temporally correlated same episodes',
        'exclusions':dict(exclusions),'episode_local_qualification':episodes,'rows_provenance':records,'collecting_model_bindings':bindings,
        'source_physical_state_summary':physical_summary(states),'protection_rows':[protection[p]['record'] for p in order],
        'actual_protection_phases':order,'missing_actual_protection_phases':['P01','P02','P03','P05','P13'],
        'protection_targets_not_provided':True,'P05_column_future_inspection_supported':True,
        'P05_column_note':'Selected first-layer W[:,4] is zero-active outside onehotP05 for identical inputs; withinP05 it may alter both mu and sigma. Not future trajectory preservation.',
        'coverage':coverage,'new_reconstruction_performed':False,'AUX_authorized':False,'fit_executed':False,'budget_selected':False,
        'PPO_or_AUX_credit_added':0,'checkpoint_written':False,'physical_success_claimed':False,
        'limitations':['Three correlated local approaches; later FL placement can be assisted and is not used as unassisted action target.',
            'Whole episodes later fail or remain partial; only actual local approach + continuation qualifies.',
            'P05-column updates affect all12 policy outputs including log-sigma; no movement improvement is proven.',
            'Current live28..38s physical comparisons are not reconstructed actor observations or candidate training labels.',
            'Source and live raw values are logged values only; no guessed reset state or raw action.',
            'Genuine source wheelstop remains unchanged; nonzero final residual targets do not establish a mask failure.']}
    MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest


def load_reviewed_data(metadata,contract):
    from wlr50_clean.ppo.semantic_migration import _contract
    m=read(MANIFEST)
    require(m['schema']==SCHEMA and not m['AUX_authorized'] and not m['fit_executed'],'not inspection-only candidate')
    for key in ('source_manifest','dataset','loader'):
        require(sha(m[key]['path'])==m[key]['sha256'],'immutable candidate binding changed:'+key)
    reference=read(m['source_manifest']['path'])
    require(metadata['checkpoint_sha256']==SOURCE_SHA and sha(SOURCE)==SOURCE_SHA,'explicit currentCP216448 only')
    require(_contract(contract)==metadata['runtime_contract']==reference['runtime_contract'] and digest(metadata['runtime_contract'])==m['runtime_contract_sha256'],'current v2 runtime differs')
    require(metadata['actor_parameter_sha256']==m['source_actor_sha256'] and {k:metadata[k] for k in COUNTERS}==m['source_PPO_counts'],'source weights/counters differ')
    require(metadata['policy_contract']==reference['policy_contract'] and metadata['normalization']==reference['normalization'],'policy/normalization differs')
    require(digest(metadata['rr_postcross_workspace_branch']['front_rehearsal_auxiliary'])==m['current_aux_ledger']['entire_four_event_object_sha256'],'four-event AUX lineage differs')
    with np.load(DATA,allow_pickle=False) as loaded:a={key:loaded[key].copy() for key in loaded.files}
    validate_arrays(a);train,val=a['train_indices'],a['validation_indices'];t=lambda x:torch.from_numpy(x.copy())
    receipt={key:m[key] for key in ('schema','status','source_checkpoint','source_actor_sha256','runtime_head','runtime_contract_sha256',
        'source_PPO_counts','current_aux_ledger','source_original_direct389','X17_remapping_performed','rows','train_rows','validation_rows',
        'selection','split','exclusions','episode_local_qualification','actual_protection_phases','missing_actual_protection_phases','limitations')}
    receipt.update(manifest=binding(MANIFEST),dataset=m['dataset'],loader=m['loader'],AUX_authorized=False,
        optimizer_steps_performed=0,PPO_or_AUX_credit_added=0,current_checkpoint_exactly_bound=True)
    return {'train_observations':t(a['observations389'][train]),'train_raw_targets':t(a['actual_raw12'][train]),
        'validation_observations':t(a['observations389'][val]),'validation_raw_targets':t(a['actual_raw12'][val]),
        'protection_observations':t(a['protection_observations']),'receipt':receipt}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',action='store_true');args=parser.parse_args()
    require(not torch.cuda.is_available(),'CPU-only');torch.set_num_threads(1)
    if args.build:build()
    metadata=read(SOURCE.with_name(SOURCE.stem+'_manifest.json'));metadata['checkpoint_path']=str(SOURCE)
    data=load_reviewed_data(metadata,metadata['runtime_contract']);m=read(MANIFEST)
    report={'result':'PASS_DATA_ONLY','source_SHA':SOURCE_SHA,'rows':m['rows'],'train_rows':m['train_rows'],'validation_rows':m['validation_rows'],
        'episodes':m['episode_local_qualification'],'exclusions':m['exclusions'],'source_summary':m['source_physical_state_summary'],
        'coverage':m['coverage'],'manifest':binding(MANIFEST),'dataset':binding(DATA),'fit_executed':False,'AUX_authorized':False}
    (HERE/'data_admission_readonly.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('result','rows','train_rows','validation_rows','episodes','exclusions')},indent=2))


if __name__=='__main__':main()

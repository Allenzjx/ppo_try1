"""Export one finalized N+0 episode and its measured camera/home evidence.

Output-only. No Isaac, policy forward, optimization, source edits, or run polling.
"""
from __future__ import annotations
import argparse
from collections import deque
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('sealed_zero_base_media',ROOT/'outputs/ppo_rr_video_diagnosis_v1/diagnostic_media.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
from wlr50_clean.ppo.semantic_video import camera_for_experiment


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open(encoding='utf8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)


def find_named(value,name):
    if isinstance(value,dict):
        if name in value:yield value[name]
        for item in value.values():yield from find_named(item,name)
    elif isinstance(value,list):
        for item in value:yield from find_named(item,name)


def norm(vector):
    if not isinstance(vector,(list,tuple)) or len(vector)!=3:return None
    if not all(type(x) in (int,float) and math.isfinite(x) for x in vector):return None
    return math.sqrt(sum(x*x for x in vector))


def sealed_source(source):
    source=Path(source).resolve(strict=True)
    run_path=source.parent/'run_manifest.json'
    base.require(run_path.is_file(),'Run has no sealed manifest; active capture must not be processed')
    run=base.read_json(run_path)
    base.require(bool(run.get('completed_at_utc')) and run.get('lifecycle')!='RUNNING',
        'Run is not finalized; do not inspect active partial logs')
    manifest_path=source/'semantic_video_source_manifest.json'
    manifest=base.read_json(manifest_path)
    base.require(manifest.get('experiment_id')=='non_residual_refine_v1'
        and manifest.get('role')=='B' and manifest.get('mode')=='semantic_prior_eval'
        and manifest.get('checkpoint_load_provenance') is None
        and manifest.get('diagnostic_intervention') is None,'Not isolated N+0 refinement')
    base.require(run['runtime_contract']==manifest['runtime_contract'],'Run/source runtime differs')
    base.require(manifest['camera']=={**camera_for_experiment('non_residual_refine_v1'),
        'resolution':[1280,720],'fps':15},'New review camera differs')
    return manifest,{'path':str(run_path),'sha256':sha(run_path),'lifecycle':run['lifecycle'],
        'source_manifest':str(manifest_path),'source_manifest_sha256':sha(manifest_path)}


def aggregate(source,manifest):
    decision_count=0; owners=[]; home=None; last_decision=None
    for row in rows(source/'video_policy_decisions.jsonl'):
        action=row.get('raw_policy_action_full12')
        base.require(isinstance(action,list) and len(action)==12 and all(v==0 for v in action),
            'N+0 label requires all twelve actual issued policy channels to be zero')
        decision_count+=1; last_decision=row
        for owner in find_named(row.get('step_info') or {},'final_stop_owner'):
            if not isinstance(owner,dict):continue
            if owner.get('active'):owners.append({'decision':row.get('decision'),
                'end_tick':row.get('end_tick'),'owner':owner})
            entry=(owner.get('home_recovery') or {}).get('entry')
            if isinstance(entry,dict) and home is None:
                home={'first_saved_decision':row.get('decision'),'first_saved_end_tick':row.get('end_tick'),**entry}
    native_count=0; phases={}; native_tail=deque(maxlen=241)
    for row in rows(source/'native_tick_audit.jsonl'):
        tick=row['episode_physics_tick']; native=row['native_audit']
        base.require(tick==native_count+1,'Native tick discontinuity')
        raw=native.get('raw_policy_action_full12'); projected=row.get('projected_residual_full12')
        base.require(len(raw)==len(projected)==12 and all(v==0 for v in raw+projected),
            'N+0 label requires every dispatched raw/projected residual channel zero')
        native_count+=1; phase=row['source_phase_id']
        if phase not in phases:phases[phase]={'first_tick':tick,'last_tick':tick,'ticks':0}
        phases[phase]['last_tick']=tick; phases[phase]['ticks']+=1
        native_tail.append({'tick':tick,'phase':phase,'nominal_full12':row['nominal_full12'],
            'mapped_nominal_full12':native.get('native_drive_target_full12'),
            'actual_native_targets':native.get('actual_native_targets')})
    base.require(native_count==manifest['episode_physics_ticks'],'Source/native endpoint differs')
    physical_count=0; terminal=deque(maxlen=241)
    for row in rows(source/'physical_observations.jsonl'):
        base.require(row['physics_tick']==physical_count,'Physical tick discontinuity')
        physical_count+=1; body=row['base']
        terminal.append({'tick':row['physics_tick'],'time_s':row['simulation_time_s'],
            'actual_full12':row['actual_full12'],'sensor_commanded_full12':row['commanded_full12'],
            'body_position_w_m':body['position_w_m'],
            'body_linear_velocity_w_m_s':body['linear_velocity_w_m_s'],
            'body_angular_velocity_w_rad_s':body['angular_velocity_w_rad_s'],
            'body_speed_m_s':norm(body['linear_velocity_w_m_s']),
            'body_angular_speed_rad_s':norm(body['angular_velocity_w_rad_s']),
            'contact_class_by_body':{k:v.get('contact_class') for k,v in row['contacts'].items()},
            'body_collision':row.get('body_collision')})
    base.require(physical_count==native_count+1,'Physical endpoint missing')
    last=terminal[-1]
    home_target=None if home is None else home.get('target_servo_deg')
    error=None if home_target is None else [actual-target for actual,target in zip(last['actual_full12'][:8],home_target)]
    result={'schema':'wlr50_clean.sealed_zero_review_aggregate.v1','source':str(source),
        'label':'N+0 - non-residual nominal baseline','trained_ppo_claim':False,
        'all_issued_and_dispatched_policy_and_projected_residuals_zero':True,
        'decision_count':decision_count,'native_tick_count':native_count,'physical_row_count':physical_count,
        'phase_samples':phases,'source_home_entry':home,'source_home_target_servo_deg':home_target,
        'terminal_actual_servo_deg':last['actual_full12'][:8],'terminal_actual_minus_source_home_deg':error,
        'maximum_abs_terminal_home_error_deg':None if error is None else max(map(abs,error)),
        'terminal_physical':last,'terminal_native':native_tail[-1],
        'last_two_seconds_physical':list(terminal),'last_two_seconds_native':list(native_tail),
        'first_active_stop_owner':None if not owners else owners[0],
        'last_active_stop_owner':None if not owners else owners[-1],
        'last_recorded_decision':{k:last_decision.get(k) for k in ('decision','request_phase','start_tick','end_tick','physics_ticks','environment_step_returned')},
        'source_physical_episode':manifest['physical_episode'],
        'source_success_candidate':manifest['success_candidate'],
        'source_acceptance_error':manifest.get('source_acceptance_error'),
        'camera':manifest['camera'],'manual_camera_QA_pending':True,
        'limitations':['Source home request is not proof of reaching home.',
            'Only observed physical outcome is reported; no new acceptance threshold is created.',
            'CPU aggregation does not certify no occlusion or independently re-adjudicate RR mechanism.']}
    return result


def keyframes(manifest,summary,ledger):
    requested=[('start',ledger[0].sim_step)]
    events=manifest['physical_episode']['physical_task_evaluation'].get('history',{}).get('event_ticks',{})
    for leg in ('FR','FL','RR','RL'):
        for event in ('active_lift','front_edge_crossed','placed'):
            tick=(events.get(event) or {}).get(leg)
            if type(tick) is int:requested.append((f'{leg}_{event}',tick))
    home=summary.get('source_home_entry')
    if home and type(home.get('entry_observation_tick')) is int:
        requested.append(('home_request_entry',home['entry_observation_tick']))
    if 'P13' in summary['phase_samples']:
        requested.append(('P13_begin',summary['phase_samples']['P13']['first_tick']))
    requested.append(('terminal',ledger[-1].sim_step))
    selected=[]
    for label,tick in requested:
        index=next((i for i,row in enumerate(ledger) if row.sim_step>=tick),len(ledger)-1)
        selected.append({'label':label,'requested_event_tick':tick,'source_frame':index,
            'actual_frame_tick':ledger[index].sim_step,'offset_from_event_ticks':ledger[index].sim_step-tick})
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--name',required=True,help='Unique output folder name under video_review_v1')
    args=parser.parse_args()
    base.require(args.name and all(c.isalnum() or c in '_-' for c in args.name),'Unsafe output name')
    source=args.source.resolve(strict=True)
    manifest,binding=sealed_source(source)  # No large-file reads before this.
    destination=OUT/args.name
    base.require(not destination.exists(),'Output folder already exists; never overwrite')
    summary=aggregate(source,manifest)
    destination.mkdir()
    summary.update(binding)
    base.write_new_json(destination/'zero_review_aggregate.json',summary)
    evaluation=manifest['physical_episode']['physical_task_evaluation']
    print(json.dumps({'sealed_zero_compact':{
        'aggregate':str(destination/'zero_review_aggregate.json'),
        'ticks':manifest['episode_physics_ticks'],
        'duration_s':manifest['episode_physics_ticks']/120,
        'source_acceptance_error':manifest.get('source_acceptance_error'),
        'evaluation':{key:evaluation.get(key) for key in ('success','termination_reason','termination_source',
            'final_region_valid','final_support_available','final_controlled','task_completed_controlled',
            'post_completion_observation_started','post_completion_observation_complete','post_completion_loss_observed')},
        'home_entry':summary['source_home_entry'],
        'source_home_target_servo_deg':summary['source_home_target_servo_deg'],
        'terminal_actual_servo_deg':summary['terminal_actual_servo_deg'],
        'terminal_actual_minus_source_home_deg':summary['terminal_actual_minus_source_home_deg'],
        'maximum_abs_terminal_home_error_deg':summary['maximum_abs_terminal_home_error_deg'],
        'body_speed_m_s':summary['terminal_physical']['body_speed_m_s'],
        'body_angular_speed_rad_s':summary['terminal_physical']['body_angular_speed_rad_s']
    }},indent=2),flush=True)
    full=destination/'zero_Nplus0_full_native.mp4'
    # Bound subprocess decoding even in the reused decoder implementation.
    original=subprocess.run
    def bounded(command,*pos,**kw):
        if isinstance(command,(list,tuple)) and 'ffmpeg' in Path(command[0]).name.lower():
            command=[command[0],'-threads','2','-filter_threads','1','-filter_complex_threads','1',*command[1:]]
        return original(command,*pos,**kw)
    subprocess.run=bounded
    try:
        base.export(source,full,'B0')
        media=base.read_json(full.with_suffix('.media.json'))
        labelled=destination/'zero_Nplus0_full_review.mp4'
        caption=f"N + 0 | NON-RESIDUAL | recorded result {media['physical_result']} | 1x - 15 fps"
        filters=("pad=1280:780:0:60:black,drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':"
            f"text='{caption}':x=16:y=17:fontcolor=white:fontsize=23")
        ffmpeg=base.find_ffmpeg()
        command=[str(ffmpeg),'-hide_banner','-nostdin','-v','error','-n','-i',str(full),'-vf',filters,
            '-an','-sn','-dn','-frames:v',str(media['frame_count']),'-r','15','-fps_mode','cfr',
            '-c:v','libx264','-crf','18','-preset','veryfast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(labelled)]
        base.run(command)
        validation=base.validate_mp4(labelled,expected_frame_count=media['frame_count'],expected_fps=15,
            expected_width=1280,expected_height=780,maximum_duration_s=200,require_sane_container_duration=True)
        base.require(validation['valid'],'Labelled full source failed decode/PTS validation')
        decoded=base.decode_frame_timeline(labelled)
        base.require(all(abs(frame.pts_s-i/15)<1e-5 for i,frame in enumerate(decoded)),'Labelled PTS differs')
        ledger=base.load_viewport_frame_ledger(source/'viewport_frame_ledger.jsonl')
        selected=keyframes(manifest,summary,ledger)
        frame_ids=sorted({x['source_frame'] for x in selected})
        expression='+'.join(f'eq(n\\,{index})' for index in frame_ids)
        base.run([str(ffmpeg),'-hide_banner','-nostdin','-v','error','-n','-i',str(labelled),'-vf',f'select={expression}',
            '-fps_mode','passthrough','-frames:v',str(len(frame_ids)),'-threads','1',str(destination/'camera_key_%02d.png')])
        image_map={frame:str(destination/f'camera_key_{index+1:02d}.png') for index,frame in enumerate(frame_ids)}
        for row in selected:row['path']=image_map[row['source_frame']]
        base.require(all(Path(path).is_file() for path in image_map.values()),'Missing keyframe')
        receipt={'schema':'wlr50_clean.sealed_zero_video_review.v1',**binding,'output':str(labelled),
            'native_full_receipt':str(full.with_suffix('.media.json')),'aggregate':str(destination/'zero_review_aggregate.json'),
            'label':caption,'physical_result':media['physical_result'],'frame_count':media['frame_count'],
            'physical_duration_s':media['physical_duration_s'],'media_duration_s':media['frame_count']/15,
            'normal_speed':True,'speed_modified':False,'stitched':False,'interpolated':False,'new_physics_steps':0,
            'robot_view_cropped':False,'annotation_only_added_margin_px':60,
            'validation':base.compact_validation(validation),'camera_keyframes':selected,
            'manual_camera_QA_pending':True,'command':command}
        base.write_new_json(labelled.with_suffix('.media.json'),receipt)
        print(json.dumps({k:receipt[k] for k in ('output','physical_result','frame_count','physical_duration_s','camera_keyframes')},indent=2))
    finally:subprocess.run=original


if __name__=='__main__':main()

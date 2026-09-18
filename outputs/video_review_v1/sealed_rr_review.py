"""Sealed-only full-P01 RR review media. Output-only; never runs a policy/Isaac.

Reuses native remux/decode/ledger validation. Bottom panels are data-only and do
not crop the original robot view. Missing semantic records stay N/A; no forward
fill of measurements or semantic flags. Comparison is descriptive, not a claim
that the zero and RR-fixed task controllers have identical semantics.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('rr_review_base',ROOT/'outputs/ppo_rr_video_diagnosis_v1/diagnostic_media.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
from wlr50_clean.ppo.semantic_video import camera_for_experiment

LEGS=('FL','FR','RL','RR')
NAMES=('front_left_ankle','front_right_ankle','rear_left_ankle','rear_right_ankle')
SIGNS=(-1.,1.,-1.,1.)
COUNTERS=('global_policy_decisions','ppo_updates','optimizer_steps')
FAMILY={'B': 'non_residual_refine_v1','C': 'residual_rr_fix_v1'}
PANEL_HEIGHT=160


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def sealed_source(source,role):
    source=Path(source).resolve(strict=True)
    run_path=source.parent/'run_manifest.json'
    base.require(run_path.is_file(),'Missing sealed run manifest')
    run=base.read_json(run_path)
    base.require(bool(run.get('completed_at_utc')) and run.get('lifecycle')!='RUNNING',
        'Run not finalized; do not read active large logs')
    manifest=base.read_json(source/'semantic_video_source_manifest.json')
    base.require(manifest.get('role')==role and manifest.get('experiment_id')==FAMILY[role],
        'Wrong isolated experiment or role')
    base.require(manifest.get('diagnostic_intervention') is None,'Intervention is not formal all-channel PPO')
    base.require(manifest.get('from_phase')=='P01' and manifest.get('optimizer_updates')==0
        and manifest.get('fresh_process_single_episode') is True and manifest.get('episode_count')==1,
        'Only one natural full-P01 evaluation is supported')
    base.require(run['runtime_contract']==manifest['runtime_contract'],'Run/source runtime mismatch')
    expected={**camera_for_experiment(FAMILY[role]),'resolution':[1280,720],'fps':15}
    base.require(manifest['camera']==expected,'Not the declared common review camera')
    return manifest


def learning_counts(metadata,proof):
    base.require(proof.get('checkpoint_loaded_and_verified') is True,'No verified official load')
    base.require(metadata['global_policy_decisions']==proof.get('saved_global_policy_decisions'),
        'Loaded checkpoint count differs')
    base.require(metadata.get('actor_parameter_sha256')==proof.get('parameter_hashes',{}).get('actor_parameter_sha256'),
        'Loaded actor differs')
    base.require(not str(metadata.get('stage','')).startswith('initial')
        and not Path(metadata['checkpoint_path']).name.startswith('checkpoint_initial_'),
        'Initialization is not learned continuation')
    branch=metadata.get('rr_task_branch') or {}; origin=branch.get('counter_origin') or {}
    counts=metadata.get('rr_task_branch_counts') or {}
    base.require(branch.get('branch_id')=='residual_rr_fix_v1','Not RR continuation branch')
    base.require(all(type(counts.get(k)) is int and counts[k]>0 and type(origin.get(k)) is int
        and metadata[k]-origin[k]==counts[k] for k in COUNTERS),'No positive verified RR branch learning')
    return {'branch_id':branch['branch_id'],'origin':origin,'actual_added_counts':counts,
        'initialization_is_training':False,'learning_is_full_task_success':False}


def learned_source(manifest):
    proof=manifest.get('checkpoint_load_provenance') or {}
    binding=proof.get('source') or {}; path=Path(binding.get('manifest',''))
    base.require(path.is_file() and path.parent.name=='history' and sha(path)==binding.get('manifest_sha256'),
        'Missing immutable checkpoint manifest binding')
    metadata=base.read_json(path)
    base.require(Path(metadata['checkpoint_path']).resolve()==Path(binding['checkpoint']).resolve()
        and metadata['checkpoint_sha256']==binding['checkpoint_sha256'],'Checkpoint binding mismatch')
    base.require(metadata['runtime_contract']==manifest['runtime_contract'],'Checkpoint/source runtime differs')
    evidence=learning_counts(metadata,proof)
    return {**evidence,'manifest':str(path),'manifest_sha256':binding['manifest_sha256'],
        'checkpoint':binding['checkpoint'],'checkpoint_sha256':binding['checkpoint_sha256']}


def checked_rows(source,manifest,name):
    digest=hashlib.sha256()
    with (source/name).open('rb') as stream:
        for line,raw in enumerate(stream,1):
            digest.update(raw)
            if raw.strip():yield line,json.loads(raw)
    base.require(digest.hexdigest()==manifest['artifacts'][name]['sha256'],f'{name} source hash mismatch')


def selected_rows(source,manifest,name,key,wanted,check=None):
    result={}
    for line,row in checked_rows(source,manifest,name):
        if check is not None:check(row)
        tick=row[key]
        if tick in wanted:
            base.require(tick not in result,f'{name}: duplicated frame tick')
            result[tick]=(line,row)
    base.require(set(result)==wanted,f'{name}: missing exact samples; no forward fill')
    return result


def semantic_sample(task,tick):
    evaluator=task.get('physical_evaluator') or {}
    exact=evaluator.get('physics_tick')==tick
    rr=(evaluator.get('current_legs') or {}).get('RR',{}) if exact else {}
    layers=((task.get('nominal_provider_diagnostics') or {}).get('source_partial_order') or {}).get('layers',[])
    # Retired layers may keep stale diagnostic records. Keep only an explicit,
    # most recent observed source check and display that check's own tick.
    applicable=[x for x in layers if type(x.get('observation_tick')) is int and x['observation_tick']<=tick]
    latest=max(applicable,key=lambda x:(x['observation_tick'],x['stage'])) if applicable else None
    if any(type(x.get('observation_tick')) is int and x['observation_tick']>tick for x in layers):
        raise RuntimeError('Source readiness from the future cannot annotate this frame')
    return {'evaluation_exact_tick':exact,'RR_gap_m':rr.get('clearance_m'),
        'RR_free_gain_m':rr.get('unsupported_free_lift_m'),'RR_current_lift':rr.get('current_lift_valid'),
        'RR_air':rr.get('air'),'RR_contact_mode':rr.get('contact_mode'),
        'source_readiness':latest,'semantic_tick':evaluator.get('physics_tick')}


def frame_data(source,manifest,ledger):
    wanted={f.sim_step for f in ledger}; tasks={}; decisions={}
    for line,row in checked_rows(source,manifest,'video_policy_decisions.jsonl'):
        tick=row['end_tick']; raw=row['raw_policy_action_full12']
        if manifest['role']=='B':base.require(len(raw)==12 and all(v==0 for v in raw),'Zero had nonzero raw action')
        if tick in wanted:
            base.require(tick not in decisions,'Duplicate decision endpoint')
            decisions[tick]={'line':line,'request_phase':row['request_phase'],
                'environment_step_returned':row['environment_step_returned']}
            tasks[tick]=(row.get('step_info') or {}).get('semantic_task') or {}
    # Interrupted final decision has no step_info. The independently recorded
    # terminal evaluator can supply final physical flags, not source scheduling.
    terminal=manifest['episode_physics_ticks']; final=manifest['physical_episode']['physical_task_evaluation']
    if terminal in wanted and not tasks.get(terminal):tasks[terminal]={'physical_evaluator':final}
    physical=selected_rows(source,manifest,'physical_observations.jsonl','physics_tick',wanted)
    def check_native(row):
        audit=row['native_audit']
        if manifest['role']=='C':
            base.require(len(audit['phase_mask_full12'])==12 and all(v==1 for v in audit['phase_mask_full12']),
                'Formal FULL12 label requires all actual dispatched residual permissions enabled')
        else:
            base.require(all(x==0 for x in audit['raw_policy_action_full12']+row['projected_residual_full12']),
                'N+0 label requires all native ticks to have zero residual')
    native=selected_rows(source,manifest,'native_tick_audit.jsonl','episode_physics_tick',wanted,check_native)
    height=selected_rows(source,manifest,'height_diagnostics.jsonl','physics_tick',wanted)
    startup_path=source/'height_diagnostics_startup.json'
    base.require(sha(startup_path)==manifest['artifacts'][startup_path.name]['sha256'],'Startup hash mismatch')
    startup=base.read_json(startup_path); names=startup['joint_names_native_order']['value']
    indices=[names.index(n) for n in NAMES]
    result=[]
    for frame in ledger:
        tick=frame.sim_step; pl,p=physical[tick]; nl,n=native[tick]; hl,h=height[tick]
        audit=n['native_audit']; readback=h['joint_velocity_native_rad_s']
        base.require(audit['verified'] and audit['actual_mapping_matches_dispatch'] and h['clock_unchanged'],
            'Unverified joint readback or dispatch')
        base.require(tuple(audit['canonical_order'][8:])==NAMES and readback['source']=='robot.data.joint_vel',
            'Wheel joint order/provenance differs')
        measured=[readback['value'][i] for i in indices]
        canonical=[p['wheels'][name]['velocity_rad_s'] for name in NAMES]
        base.require(all(math.isfinite(x) for x in measured+canonical),'Unknown measured qd must not become zero')
        base.require(all(math.isclose(a,b*s,rel_tol=1e-8,abs_tol=1e-8) for a,b,s in zip(measured,canonical,SIGNS)),
            'Native/canonical angular velocity mismatch')
        if manifest['role']=='B':
            base.require(all(x==0 for x in audit['raw_policy_action_full12']+n['projected_residual_full12']),
                'Zero had nonzero dispatched residual')
        result.append({'source_frame_index':frame.frame_index,'actual_physics_tick':tick,
            'actual_sim_time_s':frame.sim_time_s,'phase':n['source_phase_id'],
            'measured_native_qd_rad_s':measured,'measured_canonical_qd_rad_s':canonical,
            'target_native_rad_s':audit['actual_native_targets']['wheel_velocity_rad_s'],
            **semantic_sample(tasks.get(tick,{ }),tick),
            'lines':{'physical':pl,'native':nl,'height':hl,'decision':decisions.get(tick)},
            'mask_full12':audit.get('phase_mask_full12')})
    return result


def value(v,unit=1):
    return 'N/A' if v is None else f'{v*unit:+.2f}'


def bool_text(v):return 'N/A' if v is None else str(v).lower()


def panel_lines(row,caption,last=False):
    source=row['source_readiness']
    if source is None:wait='Source check: N/A (no recorded applicable source layer)'
    else:
        wait=(f"N check@{source['observation_tick']} {source['stage']} {source.get('status','N/A')} | "
            f"wait={source.get('wait_reason') or 'none'}")
    return [caption,
        f"1x / 15fps | t={row['actual_sim_time_s']:.3f}s | tick {row['actual_physics_tick']} | {row['phase']}"
        + (' | END - actual run endpoint' if last else ''),
        'Measured wheel joint qd [native rad/s]:  '+ '   '.join(f'{leg} {qd:+.3f}' for leg,qd in zip(LEGS,row['measured_native_qd_rad_s'])),
        f"RR gap {value(row['RR_gap_m'],1000)} mm | free AIR gain {value(row['RR_free_gain_m'],1000)} mm | "
        f"current lift {bool_text(row['RR_current_lift'])} | {row['RR_contact_mode'] or 'N/A'}",
        wait,
        'FL front left | FR front right | RL rear left | RR rear right. qd is rotation, not wheel-center motion.']


def validate_video(output,count,width,height):
    validation=base.validate_mp4(output,expected_fps=15,expected_frame_count=count,expected_width=width,
        expected_height=height,maximum_duration_s=200,require_sane_container_duration=True)
    base.require(validation['valid'],'Full decode/PTS validation failed')
    frames=base.decode_frame_timeline(output)
    base.require(len(frames)==count and all(abs(f.pts_s-i/15)<1e-5 for i,f in enumerate(frames)),
        'Output retimed or lost frames')
    return base.compact_validation(validation)


def keyframe_requests(manifest,rows):
    events=manifest['physical_episode']['physical_task_evaluation'].get('history',{}).get('event_ticks',{})
    requested=[('start',rows[0]['actual_physics_tick'])]
    for leg in ('FR','FL','RR','RL'):
        for event in ('active_lift','front_edge_crossed','placed'):
            tick=(events.get(event) or {}).get(leg)
            if type(tick) is int:requested.append((f'{leg}_{event}',tick))
    p13=[row for row in rows if row['phase']=='P13']
    if p13:
        requested.extend([('P13_first_frame',p13[0]['actual_physics_tick']),
            ('P13_final_half_second',max(p13[0]['actual_physics_tick'],rows[-1]['actual_physics_tick']-60))])
    requested.append(('terminal',rows[-1]['actual_physics_tick']))
    selected=[]
    for label,tick in requested:
        index=next((i for i,row in enumerate(rows) if row['actual_physics_tick']>=tick),len(rows)-1)
        selected.append({'label':label,'event_tick':tick,'frame_index':index,
            'actual_frame_tick':rows[index]['actual_physics_tick']})
    return selected


def annotate(receipt_path,output,learning=None):
    receipt=base.read_json(receipt_path); source=Path(receipt['source_manifest']).parent
    role='B' if receipt['label']=='B0' else 'C'
    manifest=sealed_source(source,role)
    base.require(receipt['media_validity']=='COMPLETE_PLAYABLE' and receipt['decoded_frame_pts_key_sequence_equal']
        and not receipt['speed_modified'] and not receipt['stitched'],'Not native whole-run media')
    base.require(sha(receipt['output'])==receipt['output_validation']['sha256']
        and sha(source/'semantic_video_source_manifest.json')==receipt['source_manifest_sha256'],'Input binding changed')
    ledger=base.load_viewport_frame_ledger(source/'viewport_frame_ledger.jsonl')
    base.require(sha(source/'viewport_frame_ledger.jsonl')==manifest['artifacts']['viewport_frame_ledger.jsonl']['sha256'],
        'Ledger hash mismatch')
    base.require(len(ledger)==receipt['frame_count'] and ledger[0].sim_step==8,'Unexpected frame coverage')
    rows=frame_data(source,manifest,ledger); count=len(rows)
    caption=(f"PPO FULL12 CP{manifest['checkpoint_load_provenance']['saved_global_policy_decisions']}" if role=='C'
        else 'N + 0 / non-residual nominal baseline')+f" | recorded {receipt['physical_result']}"
    base.require(not output.exists() and not output.with_suffix('.media.json').exists(),'Do not overwrite media')
    panels=output.with_suffix('.panels'); panels.mkdir(exist_ok=False)
    font_path=Path('C:/Windows/Fonts/consola.ttf'); fonts={size:ImageFont.truetype(str(font_path),size) for size in (16,18,20)}
    for i,row in enumerate(rows):
        panel=Image.new('RGB',(1280,PANEL_HEIGHT),'black'); draw=ImageDraw.Draw(panel)
        for j,text in enumerate(panel_lines(row,caption,last=i==count-1)):
            font=fonts[16 if j==5 else 18 if j==4 else 20]
            base.require(draw.textlength(text,font=font)<=1248,'Panel text exceeds margin; never cover/crop robot')
            draw.text((16,5+25*j),text,font=font,fill='#ffff80' if j==0 else '#a0ffa0' if j==2 else 'white')
        panel.save(panels/f'panel_{i:06d}.png')
    data=output.with_suffix('.frame_data.json')
    base.write_new_json(data,{'schema':'wlr50_clean.sealed_rr_frame_data.v1','source_manifest':receipt['source_manifest'],
        'source_manifest_sha256':receipt['source_manifest_sha256'],'wheel_order':LEGS,'native_signs_from_canonical':SIGNS,
        'rows':rows,'no_forwardfilled_measurements':True,'source_check_tick_displayed_separately':True})
    command=[str(base.find_ffmpeg()),'-hide_banner','-nostdin','-v','error','-n','-i',receipt['output'],
        '-framerate','15','-start_number','0','-i',str(panels/'panel_%06d.png'),'-filter_complex',
        '[0:v]setpts=PTS-STARTPTS[v];[1:v]format=yuv420p,setpts=PTS-STARTPTS[p];[v][p]vstack=inputs=2[out]',
        '-map','[out]','-an','-frames:v',str(count),'-r','15','-fps_mode','cfr','-c:v','libx264','-preset','veryfast',
        '-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output)]
    base.run(command); validation=validate_video(output,count,1280,720+PANEL_HEIGHT)
    selected=keyframe_requests(manifest,rows); indices=sorted({x['frame_index'] for x in selected})
    pattern=output.with_name(output.stem+'_key_%02d.png')
    expression='+'.join(f'eq(n\\,{index})' for index in indices)
    base.run([str(base.find_ffmpeg()),'-hide_banner','-nostdin','-v','error','-n','-i',str(output),
        '-vf','select='+expression,'-fps_mode','passthrough','-frames:v',str(len(indices)),'-threads','1',str(pattern)])
    paths={index:output.with_name(output.stem+f'_key_{j+1:02d}.png') for j,index in enumerate(indices)}
    for row in selected:
        row['path']=str(paths[row['frame_index']]); base.require(paths[row['frame_index']].is_file(),'Missing camera keyframe')
    result={'schema':'wlr50_clean.sealed_rr_overlay.v1','output':str(output),'source_manifest':receipt['source_manifest'],
        'source_manifest_sha256':receipt['source_manifest_sha256'],'native_receipt':str(receipt_path),
        'native_receipt_sha256':sha(receipt_path),'role':role,'caption':caption,'physical_result':receipt['physical_result'],
        'frame_count':count,'normal_speed':True,'first_actual_tick':ledger[0].sim_step,'last_actual_tick':ledger[-1].sim_step,
        'learning':learning,'frame_data':str(data),'frame_data_sha256':sha(data),'original_view_uncropped':True,
        'annotation_margin_px':PANEL_HEIGHT,'unknowns_are_not_zero':True,'new_physics_steps':0,
        'validation':validation,'camera_keyframes':selected,'manual_camera_QA_pending':True,'command':command,
        'previews':base.previews(output,count,base.find_ffmpeg())}
    base.write_new_json(output.with_suffix('.media.json'),result)
    return result,rows


def failure_window(rows):
    end=rows[-1]['actual_physics_tick']; phase=rows[-1]['phase']
    first=next(r['actual_physics_tick'] for r in rows if r['phase']==phase)
    # Preserve at least 12s context and the final-phase entry when within 30s.
    start=max(rows[0]['actual_physics_tick'],end-3600,min(first,end-1440))
    index=max(len(rows)-450,next(i for i,r in enumerate(rows) if r['actual_physics_tick']>=start))
    return index,len(rows)


def failure_excerpt(overlay,rows,output):
    base.require(overlay['physical_result']!='SUCCESS','Success has no failure excerpt')
    start,end=failure_window(rows); count=end-start
    command=[str(base.find_ffmpeg()),'-hide_banner','-nostdin','-v','error','-n','-i',overlay['output'],
        '-vf',f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS',
        '-an','-frames:v',str(count),'-r','15','-fps_mode','cfr','-c:v','libx264','-preset','veryfast',
        '-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output)]
    base.run(command); validation=validate_video(output,count,1280,720+PANEL_HEIGHT)
    result={'schema':'wlr50_clean.sealed_rr_failure_excerpt.v1','output':str(output),'full_attempt':overlay['output'],
        'source_frame_interval_half_open':[start,end],'actual_tick_endpoints':[rows[start]['actual_physics_tick'],rows[-1]['actual_physics_tick']],
        'normal_speed':True,'full_attempt_preserved':True,'same_single_episode':True,'new_physics_steps':0,
        'not_a_separate_task_attempt':True,'physical_result':overlay['physical_result'],
        'validation':validation,'command':command,'previews':base.previews(output,count,base.find_ffmpeg())}
    base.write_new_json(output.with_suffix('.media.json'),result)
    return result


def stage_difference(zero,ppo):
    expected=json.loads(json.dumps(zero))
    base.require(zero['nominal']['final_stop_owner']=='source_home_after_physical_stop_v2','Need new sealed v2 zero')
    base.require(zero['p09_lift_semantics']=='functional_lift_edge_v2'
        and 'rr_carry_source_semantics' not in zero['nominal'],'Unexpected zero RR semantics')
    expected.update(revision='residual_rr_fix_v1',p09_lift_semantics='functional_free_air_lift_v3')
    expected['nominal']['rr_carry_source_semantics']='current_free_lift_before_pending_knee_and_roll_v1'
    base.require(ppo==expected,'Unreviewed zero/PPO stage configuration difference')
    return {'same_nominal_source_schedule_claimed':False,'same_task_acceptance_claimed':False,
        'RR_current_free_air_qualification_and_source_readiness_differ':True,'same_terminal_home_v2':True}


def recorded_config(manifest,name):
    binding=manifest['runtime_contract']['selected_configuration'][name]
    raw=subprocess.run(['git','show',manifest['runtime_contract']['source_git_commit']+':'+binding['path']],
        cwd=ROOT,capture_output=True,check=True).stdout
    base.require(hashlib.sha256(raw).hexdigest()==binding['sha256'],'Committed selected config hash differs')
    return yaml.safe_load(raw)


def comparison_contract(left,right):
    base.require(left['camera']==right['camera'] and left['seed']==right['seed'],'Camera or seed differs')
    a,b=left['runtime_contract'],right['runtime_contract']
    base.require(a['source_git_commit']==b['source_git_commit'] and a['files']==b['files'],
        'Builds differ; needs explicit review, not silent comparison waiver')
    for name in ('action_schema.json','execution_profile.yaml','observation_schema.json','quality_score.yaml','reward_config.yaml'):
        base.require(a['selected_configuration'][name]['sha256']==b['selected_configuration'][name]['sha256'],
            f'Protected config differs: {name}')
    return stage_difference(recorded_config(left,'stage_task_spec.yaml'),recorded_config(right,'stage_task_spec.yaml'))


def compare(zero,ppo,output):
    left=base.read_json(zero['source_manifest']); right=base.read_json(ppo['source_manifest'])
    differences=comparison_contract(left,right)
    base.require(zero['first_actual_tick']==ppo['first_actual_tick']==8,'No inferred alignment')
    count=max(zero['frame_count'],ppo['frame_count']); base.require(count<=3000,'Comparison exceeds 200s')
    font='C\\:/Windows/Fonts/arial.ttf'; filters=[]
    for index,item in enumerate((zero,ppo)):
        base.require(sha(item['output'])==item['validation']['sha256'],'Comparison input changed')
        filters.append(f'[{index}:v]setpts=PTS-STARTPTS,scale=960:660,pad=960:696:0:0:black,'
            f"tpad=stop_mode=clone:stop={count-item['frame_count']},setpts=N/(15*TB),"
            f"drawtext=fontfile='{font}':text='END - frozen last frame, not new simulation':fontsize=21:fontcolor=yellow:x=14:y=668:"
            f"enable='gte(n,{item['frame_count']})'[v{index}]")
    filters.append('[v0][v1]hstack=inputs=2:shortest=1,format=yuv420p[out]')
    command=[str(base.find_ffmpeg()),'-hide_banner','-nostdin','-v','error','-n','-i',zero['output'],'-i',ppo['output'],
        '-filter_complex',';'.join(filters),'-map','[out]','-an','-frames:v',str(count),'-r','15','-fps_mode','cfr',
        '-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output)]
    base.run(command); validation=validate_video(output,count,1920,696)
    result={'schema':'wlr50_clean.sealed_rr_same_camera_comparison.v1','output':str(output),
        'inputs':[zero['output'],ppo['output']],'source_manifests':[zero['source_manifest'],ppo['source_manifest']],
        'config_difference_disclosure':differences,'alignment':'Same elapsed P01 time, no phase synchronization',
        'same_camera':True,'same_code_build':True,'same_scene_actuator_and_action_config':True,
        'identical_initial_measured_state_claimed':False,'identical_task_controller_claimed':False,
        'normal_speed':True,'new_physics_steps':0,'freeze_is_physics':False,
        'freeze_added_frames':{'zero':count-zero['frame_count'],'PPO':count-ppo['frame_count']},
        'validation':validation,'command':command,'manual_camera_QA_pending':True,
        'previews':base.previews(output,count,base.find_ffmpeg())}
    base.write_new_json(output.with_suffix('.media.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--name',required=True)
    parser.add_argument('--zero-native-receipt',type=Path,help='Optional native B0 receipt from sealed_zero_review.py')
    args=parser.parse_args(); source=args.source.resolve(strict=True)
    base.require(args.name and all(c.isalnum() or c in '_-' for c in args.name),'Unsafe output name')
    manifest=sealed_source(source,'C'); learning=learned_source(manifest)
    destination=OUT/args.name; base.require(not destination.exists(),'Never overwrite output directory')
    destination.mkdir(); native=destination/'PPO_full_native.mp4'
    base.export(source,native,'C0')
    overlay,rows=annotate(native.with_suffix('.media.json'),destination/'PPO_full_actual_wheels_RR.mp4',learning)
    result={'full':overlay['output'],'native':str(native),'source_manifest':str(source/'semantic_video_source_manifest.json'),
        'physical_result':overlay['physical_result'],'learning':learning,'failure_excerpt':None,'comparison':None}
    if overlay['physical_result']!='SUCCESS':
        clip=failure_excerpt(overlay,rows,destination/'PPO_latest_failure_context.mp4'); result['failure_excerpt']=clip['output']
    if args.zero_native_receipt:
        zero,zrows=annotate(args.zero_native_receipt,destination/'zero_full_actual_wheels_RR.mp4')
        pair=compare(zero,overlay,destination/'zero_vs_PPO_same_camera.mp4'); result['comparison']=pair['output']
    base.write_new_json(destination/'review_export_receipt.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    original=subprocess.run
    def bounded(command,*args,**kw):
        if isinstance(command,(list,tuple)) and 'ffmpeg' in Path(command[0]).name.lower():
            command=[command[0],'-threads','2','-filter_threads','1','-filter_complex_threads','1',*command[1:]]
        return original(command,*args,**kw)
    subprocess.run=bounded
    try:main()
    finally:subprocess.run=original

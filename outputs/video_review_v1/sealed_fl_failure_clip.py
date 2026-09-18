"""Same-frame FL capture-failure variant; original media and source untouched."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
from PIL import Image,ImageDraw,ImageFont

spec=importlib.util.spec_from_file_location('fl_clip_reuse',Path(__file__).with_name('sealed_rr_review.py'))
review=importlib.util.module_from_spec(spec); spec.loader.exec_module(review)
base=review.base


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--folder',type=Path,required=True)
    args=parser.parse_args(); folder=args.folder.resolve(strict=True)
    base.require(folder.is_relative_to(review.OUT),'Output-only folder required')
    full=base.read_json(folder/'PPO_full_native.media.json'); source=Path(full['source_manifest']).parent
    manifest=review.sealed_source(source,'C')
    base.require(full['physical_result']!='SUCCESS' and full['terminal_phase']=='P05','This variant is only recorded P05 incomplete')
    frame_file=folder/'PPO_full_actual_wheels_RR.frame_data.json'
    frames=base.read_json(frame_file)['rows']; end=len(frames); start=max(0,end-450); selected=frames[start:]
    wanted={r['actual_physics_tick'] for r in selected}; tasks={}
    for line,row in review.checked_rows(source,manifest,'video_policy_decisions.jsonl'):
        if row['end_tick'] in wanted:tasks[row['end_tick']]={'line':line,'task':(row.get('step_info') or {}).get('semantic_task') or {}}
    physical=review.selected_rows(source,manifest,'physical_observations.jsonl','physics_tick',wanted)
    terminal=manifest['episode_physics_ticks']
    if not (tasks.get(terminal) or {}).get('task'):
        tasks[terminal]={'line':None,'task':{'physical_evaluator':manifest['physical_episode']['physical_task_evaluation']}}
    output=folder/'PPO_P05_FL_capture_failure_actual_30s.mp4'; panels=output.with_suffix('.panels')
    base.require(not output.exists() and not output.with_suffix('.media.json').exists(),'Never overwrite')
    base.require(review.sha(full['output'])==full['output_validation']['sha256'],'Native media changed')
    panels.mkdir(exist_ok=False); fonts={n:ImageFont.truetype('C:/Windows/Fonts/consola.ttf',n) for n in (18,20)}
    rows=[]
    cp=manifest['checkpoint_load_provenance']['saved_global_policy_decisions']
    for i,row in enumerate(selected):
        tick=row['actual_physics_tick']; pl,p=physical[tick]; task=(tasks.get(tick) or {}).get('task') or {}
        ev=task.get('physical_evaluator') or {}; exact=ev.get('physics_tick')==tick
        fl=(ev.get('current_legs') or {}).get('FL',{}) if exact else {}
        contact=p['contacts']['front_left_wheel']; hip=p['joints']['front_left_hip']; knee=p['joints']['front_left_knee']
        observed_gap=p['wheels']['front_left_ankle']['bottom_w_m'][2]-p['obstacle']['top_z_m']
        if fl.get('clearance_m') is not None:base.require(abs(observed_gap-fl['clearance_m'])<1e-8,'FL gap clock/geometry mismatch')
        data={'output_frame':i,'source_frame':start+i,'actual_physics_tick':tick,'time_s':row['actual_sim_time_s'],
            'phase':row['phase'],'FL_gap_m':observed_gap,'FL_front_distance_m':fl.get('front_distance_m'),
            'FL_contact_class':contact['contact_class'],'FL_ground_contact':contact['ground']['active'],
            'FL_obstacle_contact':contact['obstacle']['active'],'FL_bearing_force_N':fl.get('bearing_force_n'),
            'FL_bearing_verified':fl.get('bearing_verified'),'FL_support':fl.get('support'),'FL_top_contact':fl.get('top_contact'),
            'FL_actual_hip_deg':hip['position_deg'],'FL_target_hip_deg':hip['command_deg'],
            'FL_actual_knee_deg':knee['position_deg'],'FL_target_knee_deg':knee['command_deg'],
            'native_wheel_qd_rad_s':row['measured_native_qd_rad_s'],'canonical_wheel_qd_rad_s':row['measured_canonical_qd_rad_s'],
            'evaluator_exact_tick':exact,'physical_line':pl,'decision_line':(tasks.get(tick) or {}).get('line'),
            'termination_source':task.get('termination_source'),'termination_reason':task.get('termination_reason'),
            'stage_age_s':task.get('stage_age_s')}
        rows.append(data)
        lines=[f'CP{cp} FULL12 | P05 FL not placed | EXCERPT - normal speed / 15 fps',
            f"t={data['time_s']:.3f}s | tick {tick} | {data['phase']}"+(' | END - actual run endpoint' if i==len(selected)-1 else ''),
            f"FL(front-left) gap={review.value(data['FL_gap_m'],1000)} mm | front={review.value(data['FL_front_distance_m'],1000)} mm | {data['FL_contact_class']}",
            f"FL bearing={review.value(data['FL_bearing_force_N'])} N verified={review.bool_text(data['FL_bearing_verified'])} | top={review.bool_text(data['FL_top_contact'])} support={review.bool_text(data['FL_support'])}",
            f"FL hip target/actual {hip['command_deg']:+.2f}/{hip['position_deg']:+.2f} deg | knee {knee['command_deg']:+.2f}/{knee['position_deg']:+.2f} deg",
            'Measured wheel joint qd [native rad/s]: '+'   '.join(f'{leg} {qd:+.3f}' for leg,qd in zip(review.LEGS,data['native_wheel_qd_rad_s']))]
        panel=Image.new('RGB',(1280,160),'black'); draw=ImageDraw.Draw(panel)
        for j,text in enumerate(lines):
            font=fonts[18 if j==3 else 20]
            base.require(draw.textlength(text,font=font)<1248,'FL caption overflows external panel')
            draw.text((16,5+25*j),text,font=font,fill='#ffff80' if j==0 else '#a0ffa0' if j==5 else 'white')
        panel.save(panels/f'panel_{i:06d}.png')
    data_path=output.with_suffix('.frame_data.json')
    base.write_new_json(data_path,{'schema':'wlr50_clean.sealed_FL_capture_failure_frames.v1',
        'source_manifest':full['source_manifest'],'source_manifest_sha256':full['source_manifest_sha256'],
        'source_wheel_data_sha256':review.sha(frame_file),'rows':rows,'no_forward_fill':True})
    command=[str(base.find_ffmpeg()),'-hide_banner','-nostdin','-v','error','-n','-i',full['output'],
        '-framerate','15','-start_number','0','-i',str(panels/'panel_%06d.png'),'-filter_complex',
        f'[0:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS[v];[1:v]format=yuv420p,setpts=PTS-STARTPTS[p];[v][p]vstack=inputs=2[out]',
        '-map','[out]','-an','-frames:v',str(len(selected)),'-r','15','-fps_mode','cfr','-c:v','libx264',
        '-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output)]
    base.run(command); validation=review.validate_video(output,len(selected),1280,880)
    result={'schema':'wlr50_clean.sealed_FL_capture_failure_media.v1','output':str(output),'checkpoint_decisions':cp,
        'source_manifest':full['source_manifest'],'source_manifest_sha256':full['source_manifest_sha256'],
        'source_frame_interval_half_open':[start,end],'actual_tick_endpoints':[rows[0]['actual_physics_tick'],rows[-1]['actual_physics_tick']],
        'frame_count':len(selected),'duration_s':len(selected)/15,'normal_speed':True,'new_physics_steps':0,
        'full_attempt_retained':full['output'],'physical_result':full['physical_result'],'first_incomplete_task':'P05 FL placement',
        'RR_phase_reached':False,'RR_success_claimed':False,'crop_or_zoom':False,'robot_pixels_uncropped':True,
        'overlay_outside_original_view':True,'no_guessed_image_leg_coordinates':True,
        'same_tick_measured_native_wheel_qd':True,'evaluator_unknowns_stay_NA':True,
        'terminal':rows[-1],'frame_data':str(data_path),'frame_data_sha256':review.sha(data_path),
        'validation':validation,'command':command,'previews':base.previews(output,len(selected),base.find_ffmpeg())}
    base.write_new_json(output.with_suffix('.media.json'),result)
    print({k:result[k] for k in ('output','frame_count','duration_s','actual_tick_endpoints','terminal')})


if __name__=='__main__':
    original=subprocess.run
    def bounded(command,*args,**kw):
        if isinstance(command,(list,tuple)) and 'ffmpeg' in Path(command[0]).name.lower():
            command=[command[0],'-threads','2','-filter_threads','1','-filter_complex_threads','1',*command[1:]]
        return original(command,*args,**kw)
    subprocess.run=bounded
    try:main()
    finally:subprocess.run=original

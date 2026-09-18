"""Source-bound normal-speed historical RR review; no new physical evaluation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('review_base',ROOT/'outputs/ppo_rr_video_diagnosis_v1/diagnostic_media.py')
base=importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
OUT=Path(__file__).resolve().parent
DIAG=ROOT/'outputs/diagnostics_v1/residual_RR_diagnosis.json'
FULL=ROOT/'outputs/ppo_task_first_recovery_v1/videos/ppo_task_recovery_cp177152_full_success.media.json'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stamp(seconds):
    # ASS has 10 ms granularity. Floor boundaries so the new row is active at
    # its exact 15 Hz frame PTS; rounding upward would show the preceding tick.
    centiseconds=math.floor(seconds*100+1e-8)
    hours,rest=divmod(centiseconds,360000); minutes,rest=divmod(rest,6000); secs,cs=divmod(rest,100)
    return f'{hours}:{minutes:02d}:{secs:02d}.{cs:02d}'


def main():
    diag=json.loads(DIAG.read_text(encoding='utf8'))['runs']['CP177152']
    full=json.loads(FULL.read_text(encoding='utf8'))
    source=Path(diag['source']); manifest=source/'semantic_video_source_manifest.json'
    assert str(manifest)==full['source_manifest'] and sha(manifest)==full['source_manifest_sha256']
    input_path=Path(full['output'])
    assert sha(input_path)==full['output_validation']['sha256']
    ledger=base.load_viewport_frame_ledger(source/'viewport_frame_ledger.jsonl')
    # Includes the first genuine AIR lift and later contact-assisted-looking
    # ascent, rather than presenting only the latter as the entire attempt.
    selected=[(i,row) for i,row in enumerate(ledger) if 4352<=row.sim_step<=5440]
    indices=[i for i,_ in selected]
    assert indices==list(range(indices[0],indices[-1]+1))
    by_tick={row['tick']:row for row in diag['rows']}
    rows=[by_tick[row.sim_step] for _,row in selected]
    assert all(rows[i]['tick']-rows[i-1]['tick']==8 for i in range(1,len(rows)))
    output=OUT/'historical_cp177152_RR_contact_review_native_tick.mp4'
    subtitle=output.with_suffix('.ass')
    receipt=output.with_suffix('.media.json')
    assert not output.exists() and not subtitle.exists() and not receipt.exists()
    header='''[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 840
WrapStyle: 2
[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Review,Arial,23,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,18,18,10,1
[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
'''
    lines=[]
    for i,row in enumerate(rows):
        text=(r'{\c&H00FFFF&}CP177152 - HISTORICALLY ACCEPTED / RR UNDER REVIEW{\c&HFFFFFF&}'
            r'\N'+f"1x | source t={row['time_s']:.3f}s | tick {row['tick']} | {row['phase']} | not latest CP177792"
            r'\N'+f"RR wheel joint qd={row['RR_qd']:+.3f} rad/s | obstacle contact={row['RR_obstacle']} | surface={row['RR_surface']}"
            r'\N'+f"RR bottom-to-top gap={1000*row['RR_gap']:+.2f} mm | air={row['RR_air']} | no new success claim")
        lines.append(f'Dialogue: 0,{stamp(i/15)},{stamp((i+1)/15)},Review,,0,0,0,,{text}\n')
    subtitle.write_text(header+''.join(lines),encoding='utf8')
    escaped=str(subtitle).replace('\\','/').replace(':',r'\:')
    filters=(f'trim=start_frame={indices[0]}:end_frame={indices[-1]+1},setpts=PTS-STARTPTS,'
             f"pad=1280:840:0:120:black,ass='{escaped}'")
    ffmpeg=base.find_ffmpeg()
    command=[str(ffmpeg),'-threads','2','-filter_threads','1','-hide_banner','-nostdin','-v','error','-n',
        '-i',str(input_path),'-vf',filters,'-an','-sn','-dn','-frames:v',str(len(rows)),
        '-c:v','libx264','-crf','18','-preset','veryfast','-threads','2','-pix_fmt','yuv420p',
        '-r','15','-fps_mode','cfr','-movflags','+faststart',str(output)]
    base.run(command)
    validation=base.validate_mp4(output,expected_fps=15,expected_frame_count=len(rows),expected_width=1280,
        expected_height=840,maximum_duration_s=200,require_sane_container_duration=True)
    assert validation['valid']
    timeline=base.decode_frame_timeline(output)
    assert len(timeline)==len(rows) and all(abs(f.pts_s-i/15)<1e-5 for i,f in enumerate(timeline))
    assert sha(input_path)==full['output_validation']['sha256']
    data={'schema':'wlr50_clean.historical_rr_review_clip.v1','output':str(output),
        'source_full_receipt':str(FULL),'source_full_receipt_sha256':sha(FULL),
        'source_manifest':str(manifest),'source_manifest_sha256':sha(manifest),
        'diagnosis':str(DIAG),'diagnosis_sha256':sha(DIAG),
        'source_frame_indices_inclusive':[indices[0],indices[-1]],
        'actual_endpoint_ticks':[rows[0]['tick'],rows[-1]['tick']],
        'frame_tick_bindings':[{'frame':i,'source_frame':pair[0],'tick':row['tick'],
            'RR_qd_rad_s':row['RR_qd'],'RR_surface':row['RR_surface'],'RR_gap_m':row['RR_gap']}
            for i,(pair,row) in enumerate(zip(selected,rows))],
        'frame_count':len(rows),'normal_speed':True,'speed_modified':False,'interpolated':False,
        'stitched':False,'cropped_robot_view':False,'new_physics_steps':0,
        'current_acceptance':'RR_ACCEPTANCE_UNDER_REVIEW','historical_label_preserved':full['physical_result'],
        'new_success_claim':False,'latest_candidate_evaluation':False,
        'first_real_air_lift_retained':True,
        'causality_limitation':'Powered obstacle-contact ascent is shown, not isolated proof of wheel motive force or a claim that RR never lifted.',
        'diagnostic_surface_ambiguity':'OBSTACLE_AMBIGUOUS is exact obstacle contact without trustworthy wall/top location, not proof of wall contact.',
        'original_camera_limitation':'Historical camera retained; no synthetic new view.',
        'validation':base.compact_validation(validation),'command':command,
        'previews':base.previews(output,len(rows),ffmpeg)}
    base.write_new_json(receipt,data)
    print(json.dumps({k:data[k] for k in ('output','actual_endpoint_ticks','frame_count','current_acceptance','previews')},indent=2))


if __name__=='__main__':main()

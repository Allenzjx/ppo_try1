"""Frame-ledger wheel-rotation overlay on an already validated real video.

Output-only. Native measured ankle qd is not wheel-center translation. No camera
change, no physics, no phase replay, no retiming, and no success reclassification.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('height_media_reuse', HERE / 'height_media.py')
height = importlib.util.module_from_spec(spec)
spec.loader.exec_module(height)
base = height.base
LEGS = ('FL', 'FR', 'RL', 'RR')
LONG = ('front left', 'front right', 'rear left', 'rear right')
NAMES = ('front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle')
SIGNS = (-1., 1., -1., 1.)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def family_name(family, receipt, source):
    intervention = source.get('diagnostic_intervention')
    proof = source.get('checkpoint_load_provenance') or {}
    if family == 'ZERO':
        base.require(source['role'] == 'B' and not intervention and not proof
                     and receipt.get('base_media_role') == 'B0', 'Not verified unmasked zero')
        return 'ZERO residual | nominal control | full run ' + receipt['physical_result']
    if family == 'PPO_FULL12':
        base.require(source['role'] == 'C' and not intervention and receipt.get('formal_C0') is True
                     and proof.get('checkpoint_loaded_and_verified') is True, 'Not formal all12 C')
        return f"PPO FULL12 | checkpoint {proof['saved_global_policy_decisions']} | {receipt['physical_result']}"
    base.require(family == 'MASKED_WHEELS4' and isinstance(intervention, dict)
                 and intervention.get('masked_raw_indices') == [8, 9, 10, 11]
                 and intervention.get('mode') == 'DIAGNOSTIC_WHEELS4_RAW_OFF_FROM_FIRST_DECISION'
                 and receipt.get('formal_C0') is False, 'Not explicitly masked wheels4 diagnostic')
    return 'MASK DIAGNOSTIC ONLY | raw wheels4 OFF | not formal PPO success'


def data_rows(source, manifest, name, tick_key, wanted):
    expected = manifest['artifacts'][name]['sha256']
    digest, result = hashlib.sha256(), {}
    with (source / name).open('rb') as stream:
        for line_no, raw in enumerate(stream, 1):
            digest.update(raw)
            if not raw.strip():
                continue
            row = json.loads(raw)
            tick = row[tick_key]
            if tick in wanted:
                base.require(tick not in result, 'Duplicate actual frame tick')
                result[tick] = (line_no, row)
    base.require(digest.hexdigest() == expected, f'{name} does not match source artifact')
    base.require(set(result) == wanted, f'{name} missing actual frame samples; never forward-fill')
    return result


def ass_time(frame):
    # floor-centiseconds place every frame's event before its exact 15Hz PTS,
    # but after the preceding frame's PTS. No frame-data interpolation.
    cs = frame * 100 // 15
    return f'{cs // 360000}:{cs // 6000 % 60:02}:{cs // 100 % 60:02}.{cs % 100:02}'


def text_event(start, end, x, y, size, text, color='FFFFFF'):
    base.require(not any(c in text for c in '{}\\\r'), 'Unsafe ASS text')
    text = text.replace('\n', r'\N')
    return f'Dialogue: 0,{ass_time(start)},{ass_time(end)},Default,,0,0,0,,{{\\an7\\pos({x},{y})\\fs{size}\\c&H{color}&}}{text}\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--media-receipt', type=Path)
    parser.add_argument('--family', choices=('ZERO', 'PPO_FULL12', 'MASKED_WHEELS4'))
    parser.add_argument('--window', choices=('full', 'p01-p03'), default='full')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        for i in range(3000):
            low, high = i * 100 // 15 / 100, (i + 1) * 100 // 15 / 100
            base.require(low <= i / 15 < high, 'Subtitle clock cannot cover actual frame')
        try:
            family_name('PPO_FULL12', {'formal_C0': True}, {'role': 'C', 'diagnostic_intervention': {'mask': True}})
        except RuntimeError:
            print(json.dumps({'self_test': 'PASS', 'mask_cannot_be_formal': True, 'frame_clock_3000_verified': True}))
            return
        raise AssertionError('Mask mislabeled formal')
    base.require(args.media_receipt and args.family and args.output, 'receipt/family/new output required')
    receipt = base.read_json(args.media_receipt)
    source_manifest = Path(receipt['source_manifest'])
    source, manifest = source_manifest.parent, base.read_json(source_manifest)
    original, output = Path(receipt['output']), args.output.resolve()
    base.require(not output.exists() and not output.with_suffix('.media.json').exists(), 'Never overwrite outputs')
    base.require(receipt['media_validity'] == 'COMPLETE_PLAYABLE' and receipt['decoded_frame_pts_key_sequence_equal']
                 and not receipt['speed_modified'] and not receipt['stitched'], 'Not verified unretimed source media')
    base.require(sha(original) == receipt['output_validation']['sha256']
                 and sha(source_manifest) == receipt['source_manifest_sha256'], 'Media/source receipt changed')
    base.require(manifest['runtime_contract'] == receipt['runtime_contract']
                 and manifest['optimizer_updates'] == 0 and manifest['from_phase'] == 'P01', 'Source binding mismatch')
    caption = family_name(args.family, receipt, manifest)
    base.require(args.window == 'full' or args.family == 'ZERO', 'Formal C and diagnostic must retain their entire actual run')
    ledger = base.load_viewport_frame_ledger(source / 'viewport_frame_ledger.jsonl')
    base.require(sha(source / 'viewport_frame_ledger.jsonl') == manifest['artifacts']['viewport_frame_ledger.jsonl']['sha256'], 'Ledger changed')
    base.require(len(ledger) == receipt['frame_count'] and ledger[0].sim_step == 8, 'Unexpected source frame coverage')
    cut = manifest['episode_physics_ticks']
    if args.window == 'p01-p03':
        transitions_path = source / 'stage_transition_evidence.jsonl'
        base.require(sha(transitions_path) == manifest['artifacts']['stage_transition_evidence.jsonl']['sha256'], 'Transition evidence changed')
        transitions = [json.loads(line) for line in transitions_path.read_text(encoding='utf8').splitlines()]
        candidates = [r['physics_tick'] for r in transitions if r.get('from_stage') == 'P03' and r.get('to_stage') == 'P04']
        base.require(len(candidates) == 1, 'Unique real P03 exit required for excerpt')
        cut = candidates[0]
    ledger = [f for f in ledger if f.sim_step <= cut]
    base.require(len(ledger) >= 2 and len(ledger) / 15 <= 200, 'Invalid contiguous normal-speed cut')
    wanted = {f.sim_step for f in ledger}
    physical = data_rows(source, manifest, 'physical_observations.jsonl', 'physics_tick', wanted)
    native = data_rows(source, manifest, 'native_tick_audit.jsonl', 'episode_physics_tick', wanted)
    heights = data_rows(source, manifest, 'height_diagnostics.jsonl', 'physics_tick', wanted)
    startup_path = source / 'height_diagnostics_startup.json'
    base.require(sha(startup_path) == manifest['artifacts']['height_diagnostics_startup.json']['sha256'], 'Startup changed')
    startup = base.read_json(startup_path)
    native_names = startup['joint_names_native_order']['value']
    indices = [native_names.index(name) for name in NAMES]
    command_source = base.PROJECT / 'src/wlr50_clean/infrastructure/command_batch.py'
    base.require(sha(command_source) == manifest['runtime_contract']['files']['src/wlr50_clean/infrastructure/command_batch.py'], 'Sign mapping source differs from this run')
    from wlr50_clean.infrastructure.command_batch import WHEEL_FORWARD_SIGN
    base.require(tuple(WHEEL_FORWARD_SIGN[n] for n in NAMES) == SIGNS, 'Unexpected canonical/native signs')
    rows, bindings = [], {}
    for frame in ledger:
        tick = frame.sim_step
        pl, p = physical[tick]; nl, n = native[tick]; hl, h = heights[tick]
        audit, actual = n['native_audit'], h['joint_velocity_native_rad_s']
        base.require(audit['verified'] and audit['actual_mapping_matches_dispatch'] and h['clock_unchanged'], 'Unverified actual readback/dispatch')
        base.require(tuple(audit['canonical_order'][8:]) == NAMES and actual['source'] == 'robot.data.joint_vel', 'Wrong angular velocity provenance')
        measured = [actual['value'][i] for i in indices]
        targets = audit['actual_native_targets']['wheel_velocity_rad_s']
        canonical = [p['wheels'][name]['velocity_rad_s'] for name in NAMES]
        for j, name in enumerate(NAMES):
            base.require(all(math.isfinite(float(v)) for v in (measured[j], targets[j], canonical[j])), 'Unknown wheel sample cannot become zero')
            base.require(math.isclose(measured[j], canonical[j] * SIGNS[j], abs_tol=1e-8, rel_tol=1e-8), 'Native/canonical measured qd differ')
            wheel = p['wheels'][name]
            body = name.replace('_ankle', '_wheel')
            base.require(wheel['name'] == name and wheel['body_name'] == body and body in wheel['geometry_source'], 'Wheel name/prim binding missing')
            bindings[LEGS[j]] = {'joint': name, 'body': body, 'geometry_source': wheel['geometry_source'], 'native_index': indices[j], 'canonical_to_native_sign': SIGNS[j]}
        rows.append({'frame_index': frame.frame_index, 'output_pts_s': frame.frame_index / 15, 'actual_physics_tick': tick,
            'actual_sim_time_s': frame.sim_time_s, 'native_source_phase': n['source_phase_id'],
            'target_native_rad_s': targets, 'measured_native_qd_rad_s': measured, 'measured_canonical_qd_rad_s': canonical,
            'physical_line': pl, 'native_line': nl, 'height_line': hl})
    count = len(rows)
    ass_path, data_path = output.with_suffix('.ass'), output.with_suffix('.wheel_data.json')
    for dest in (ass_path, data_path): base.require(not dest.exists(), 'Derivative already exists')
    script = '[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 960\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: Default,Consolas,22,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'
    for i, row in enumerate(rows):
        status = ('RUN ENDED - ' + receipt['physical_result']) if i == count-1 and args.window == 'full' else ('EXCERPT END - full run retained' if i == count-1 else 'LIVE actual frame')
        panel = [caption,
            f"1x / 15fps | tick {row['actual_physics_tick']} | sim {row['actual_sim_time_s']:.3f}s | {row['native_source_phase']} | {status}",
            '', ''.join(f'{leg} {label:<20}' for leg,label in zip(LEGS,LONG)),
            ''.join(f"target {v:+8.4f} rad/s   " for v in row['target_native_rad_s']),
            ''.join(f"actual {v:+8.4f} rad/s   " for v in row['measured_native_qd_rad_s']),
            '', 'NATIVE axis signs. canonical -> native: FL -1, FR +1, RL -1, RR +1.',
            'Measured ankle angular qd, NOT translation. First encoded frame = actual tick8.']
        script += text_event(i,i+1,16,728,20,'\n'.join(panel))
    output.parent.mkdir(parents=True,exist_ok=True)
    with ass_path.open('x',encoding='utf8') as stream: stream.write(script)
    # libass on this installed FFmpeg produced intermittent missing glyphs.
    # Deterministic data-only PNG panels avoid that renderer. No source frame is edited.
    panel_dir = output.with_suffix('.panels')
    panel_dir.mkdir(exist_ok=False)
    font_path = Path('C:/Windows/Fonts/consola.ttf')
    base.require(font_path.is_file(), 'Reviewed monospace font missing')
    fonts = {size: ImageFont.truetype(str(font_path),size) for size in (17,18,20,22)}
    for i,row in enumerate(rows):
        panel = Image.new('RGB',(1280,240),'black')
        draw = ImageDraw.Draw(panel)
        draw.text((16,8),caption,font=fonts[22],fill='white')
        status = ('RUN ENDED - ' + receipt['physical_result']) if i == count-1 and args.window == 'full' else ('EXCERPT END - full run retained' if i == count-1 else 'LIVE actual frame')
        draw.text((16,42),f"1x / 15fps | tick {row['actual_physics_tick']} | sim {row['actual_sim_time_s']:.3f}s | {row['native_source_phase']} | {status}",font=fonts[20],fill='white')
        for j,(leg,label) in enumerate(zip(LEGS,LONG)):
            x=16+j*316
            draw.text((x,78),f'{leg} {label}',font=fonts[22],fill='#ffff80')
            draw.text((x,112),f"target {row['target_native_rad_s'][j]:+8.4f}",font=fonts[22],fill='white')
            draw.text((x,144),f"actual {row['measured_native_qd_rad_s'][j]:+8.4f}",font=fonts[22],fill='#a0ffa0')
        draw.text((16,183),'NATIVE joint-axis signs. canonical -> native: FL -1, FR +1, RL -1, RR +1.',font=fonts[18],fill='white')
        draw.text((16,212),'Measured ankle angular qd [rad/s], NOT wheel-center translation. First encoded frame = actual tick8.',font=fonts[17],fill='white')
        panel.save(panel_dir/f'panel_{i:06d}.png')
    base.write_new_json(data_path, {'schema':'wlr50_clean.exact_frame_wheel_overlay.v1','family':args.family,
        'source_manifest':str(source_manifest),'input_media_receipt':str(args.media_receipt.resolve()),'wheel_order':LEGS,
        'wheel_bindings':bindings,'rows':rows,'readback_source':'height_diagnostics robot.data.joint_vel; exact same-frame physical and native audit cross-check'})
    ffmpeg = base.find_ffmpeg()
    command = [str(ffmpeg),'-hide_banner','-nostdin','-v','error','-n','-i',str(original),
        '-framerate','15','-start_number','0','-i',str(panel_dir/'panel_%06d.png'),'-an',
        '-filter_complex',f'[0:v]trim=end_frame={count},setpts=PTS-STARTPTS[v];[1:v]format=yuv420p,setpts=PTS-STARTPTS[p];[v][p]vstack=inputs=2[out]',
        '-map','[out]',
        '-frames:v',str(count),'-r','15','-fps_mode','cfr','-c:v','libx264','-preset','veryfast','-crf','18',
        '-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output)]
    base.run(command)
    validation = base.validate_mp4(output,ffmpeg=ffmpeg,expected_width=1280,expected_height=960,expected_frame_count=count)
    base.require(validation['valid'], 'Overlay failed complete media validation')
    decoded = base.decode_frame_timeline(output,ffmpeg=ffmpeg)
    base.require(len(decoded)==count and all(abs(f.pts_s-i/15)<1e-5 for i,f in enumerate(decoded)), 'Overlay changed frame cadence')
    result = {'schema':'wlr50_clean.wheel_overlay_media.v1','output':str(output),'family':args.family,'caption':caption,
        'input_media_receipt':str(args.media_receipt.resolve()),'input_media_sha256':receipt['output_validation']['sha256'],
        'source_head':manifest['runtime_contract']['source_git_commit'],'runtime_sha256':manifest['runtime_contract']['runtime_content_sha256'],
        'source_manifest':str(source_manifest),'diagnostic_intervention':manifest.get('diagnostic_intervention'),
        'physical_task_success_of_full_source':manifest['physical_task_success'],'full_source_result':receipt['physical_result'],
        'window':args.window,'excerpt_is_not_a_separate_task_success':True,'first_actual_tick':ledger[0].sim_step,'last_actual_tick':ledger[-1].sim_step,
        'frame_count':count,'normal_speed':True,'fps':15,'freeze_frames_added':0,'full_C_failure_endpoint_retained':args.family=='PPO_FULL12',
        'original_video_preserved':True,'original_1280x720_pixels_not_cropped_or_scaled':True,'data_panel_added_below_original_view':True,
        'reencoded_for_text':True,'no_AI_images':True,'no_interpolated_or_forwardfilled_measurements':True,
        'data':str(data_path),'data_sha256':sha(data_path),'ass_unused_debug_text':str(ass_path),
        'renderer':'Pillow deterministic data-only PNG panels + FFmpeg vstack; ASS not used',
        'panel_directory':str(panel_dir),'font_path':str(font_path),'font_sha256':sha(font_path),
        'output_validation':base.compact_validation(validation),'previews':base.previews(output,count,ffmpeg)}
    base.write_new_json(output.with_suffix('.media.json'),result)
    print(json.dumps({k:result[k] for k in ('output','family','frame_count','first_actual_tick','last_actual_tick','previews')}))


if __name__ == '__main__':
    original_run = subprocess.run
    def bounded_run(command,*args,**kwargs):
        if isinstance(command,(list,tuple)) and 'ffmpeg' in Path(command[0]).name.lower():
            command=[command[0],'-threads','2','-filter_threads','1','-filter_complex_threads','1',*command[1:]]
        return original_run(command,*args,**kwargs)
    subprocess.run=bounded_run
    try: main()
    finally: subprocess.run=original_run

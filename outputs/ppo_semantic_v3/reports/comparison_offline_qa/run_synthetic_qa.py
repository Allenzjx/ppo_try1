"""SYNTHETIC_QA only: AST-extracted renderer, no runtime/Torch/Isaac imports."""
import ast
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'render_outcome_comparison.py'
FFMPEG = Path(r'C:\Users\kskzz\miniconda3\envs\env_isaaclab\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe')
START = time.monotonic()
RECEIPT = {'schema': 'SYNTHETIC_QA.comparison.v1', 'real_robot_video': False,
           'task_success_evidence': False, 'scope': 'exact renderer AST and selected rejection predicates; not full provenance/CLI acceptance',
           'source_script_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'checks': {}}


def run(command, *, binary=False):
    assert time.monotonic() - START < 110, 'bounded QA time expired'
    # Decoder, software encoder, simple and complex filters each use one thread.
    bounded = [command[0], '-threads', '1', '-filter_threads', '1', '-filter_complex_threads', '1']
    for part in command[1:]:
        if part == '-i': bounded += ['-threads', '1']
        bounded.append(part)
    if command[-1] not in ('-', 'NUL'):
        bounded[-1:-1] = ['-threads', '1']
    result = subprocess.run(bounded, capture_output=True, text=not binary, timeout=35)
    if result.returncode:
        err = result.stderr.decode(errors='replace') if binary else result.stderr
        raise RuntimeError(err[-3000:])
    return result


def require(condition, message):
    if not condition: raise ValueError(message)


def decode(path, crop=None):
    vf = ([] if crop is None else ['-vf', crop])
    result = run([str(FFMPEG), '-hide_banner', '-nostdin', '-v', 'error', '-i', str(path),
                  *vf, '-map', '0:v:0', '-an', '-sn', '-dn', '-pix_fmt', 'rgb24',
                  '-fps_mode', 'passthrough', '-f', 'rawvideo', '-'], binary=True)
    return result.stdout


try:
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    first = next(i for i, node in enumerate(main.body)
                 if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'branches' for t in node.targets))
    last = next(i for i in range(first, len(main.body)) if isinstance(main.body[i], ast.Try))
    render_ast = ast.Module(body=main.body[first:last], type_ignores=[])
    # Exact source rendering expressions; only thread/resource flags added at execution.
    for label, frames in [('A', 6), ('C', 12)]:
        target = HERE / f'SYNTHETIC_QA_{label}_{frames}frames.mp4'
        run([str(FFMPEG), '-hide_banner', '-nostdin', '-v', 'error', '-n',
             '-f', 'lavfi', '-i', 'testsrc2=size=1280x720:rate=15',
             '-f', 'lavfi', '-i', 'sine=frequency=440:sample_rate=8000',
             '-vf', f"drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':text='SYNTHETIC_QA {label} - NOT A ROBOT VIDEO':x=30:y=650:fontsize=30:fontcolor=white",
             '-frames:v', str(frames), '-t', str(frames/15), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
             '-preset', 'ultrafast', '-c:a', 'aac', str(target)])
    a = {'count': 6, 'outcome': 'INCOMPLETE', 'paths': {'actual_viewport_video.mp4': HERE/'SYNTHETIC_QA_A_6frames.mp4'}}
    # Deliberately neutral synthetic label; no success caption is manufactured.
    c = {'count': 12, 'outcome': 'SYNTHETIC QA ONLY', 'paths': {'actual_viewport_video.mp4': HERE/'SYNTHETIC_QA_C_12frames.mp4'}}
    destination = HERE/'SYNTHETIC_QA_comparison_NOT_ROBOT.mp4'
    env = {'a': a, 'c': c, 'count': 12, 'destination': destination, 'find_ffmpeg': lambda: FFMPEG}
    exec(compile(render_ast, str(SOURCE), 'exec'), env)
    RECEIPT['exact_script_filter'] = env['filters']
    RECEIPT['unmodified_script_command'] = [str(v) for v in env['command']]
    run(env['command'])
    probe = run([str(FFMPEG), '-hide_banner', '-nostdin', '-i', str(destination),
                 '-vf', 'showinfo', '-map', '0:v:0', '-an', '-fps_mode', 'passthrough', '-f', 'null', 'NUL'])
    log = probe.stderr
    pts = [float(x) for x in re.findall(r'\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:\s*([-+0-9.eE]+)', log)]
    require(len(pts) == 12 and all(abs(p-i/15)<1e-5 for i,p in enumerate(pts)), 'changed frame count or PTS')
    require('2560x720' in log and '15 fps' in log and 'yuv420p' in log, 'wrong raster/FPS/format')
    input_section = log.split('Stream mapping:')[0]
    require('Audio:' not in input_section, 'output contains audio')
    require('Duration: 00:00:00.80' in input_section, 'unexpected output container duration')
    # Lossy re-encoding prevents whole-frame byte equality. Compare lower content
    # regions untouched by comparison labels, preserving all source edges/raster.
    def frames(path, crop):
        data = decode(path, crop)
        size = 1280*600*3
        require(len(data)%size == 0, 'partial decoded RGB frame')
        return [data[i:i+size] for i in range(0,len(data),size)]
    src_a, src_c = frames(a['paths']['actual_viewport_video.mp4'], 'crop=1280:600:0:120'), frames(c['paths']['actual_viewport_video.mp4'], 'crop=1280:600:0:120')
    out_a, out_c = frames(destination, 'crop=1280:600:0:120'), frames(destination, 'crop=1280:600:1280:120')
    # Sample every 97th byte: bounded CPU/memory, full width and bottom retained.
    mae = lambda x,y: sum(abs(x[i]-y[i]) for i in range(0,len(x),97))/len(range(0,len(x),97))
    errors_a = [mae(out_a[i], src_a[min(i,5)]) for i in range(12)]
    errors_c = [mae(out_c[i], src_c[i]) for i in range(12)]
    require(max(errors_a+errors_c)<3, 'source alignment/freeze content differs')
    require("enable='gte(n,6)'" in env['filters'] and 'SOURCE ENDED - last recorded frame held' in env['filters'], 'missing exact ended label boundary')
    RECEIPT['checks'].update(frame_count=12, fps=15, duration_s=.8, raster=[2560,720], audio_tracks=0,
        source_counts=[6,12], a_final_frame_held_at_output_indices=list(range(6,12)),
        source_content_sampled_mae_a=errors_a, source_content_sampled_mae_c=errors_c,
        no_crop_scale_interpolation_or_rate_filter=True, neutral_label_filter_present=True)
    # Exact production path/type rejection, extracted without top-level imports.
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('managed_source','verify_source_timing')]
    rejection_env = {'Path':Path, 'PROJECT_ROOT':SOURCE.parents[3], 'require':require, 'HZ':120, 'POST_TICKS':120}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(SOURCE),'exec'),rejection_env)
    rejected=[]
    for label, fn in [('unmanaged synthetic source',lambda:rejection_env['managed_source'](HERE,'A')),
                      ('bool endpoint',lambda:rejection_env['verify_source_timing']({'episode_physics_ticks':True,'performed_post_success_ticks':0},{})),
                      ('negative endpoint',lambda:rejection_env['verify_source_timing']({'episode_physics_ticks':-1,'performed_post_success_ticks':0},{}))]:
        try: fn()
        except ValueError: rejected.append(label)
        else: raise AssertionError('invalid input accepted: '+label)
    # Check -n actually refuses an existing destination, preserving source/output.
    existing_size = destination.stat().st_size
    existing_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    try:
        refusal = run(env['command'])
        # This bundled FFmpeg emits refusal text but returns 0 for -n.
        require('already exists' in refusal.stderr and 'Exiting' in refusal.stderr,
                'existing destination did not produce an explicit refusal')
    except RuntimeError:
        pass
    rejected.append('existing destination refused by -n; byte hash unchanged')
    require(destination.stat().st_size == existing_size, 'existing output changed')
    require(hashlib.sha256(destination.read_bytes()).hexdigest() == existing_hash, 'existing output bytes changed')
    RECEIPT['checks']['rejected_inputs']=rejected
    RECEIPT['status']='LIMITED_SYNTHETIC_RENDER_QA_PASSED'
except Exception as exc:
    RECEIPT['status']='QA_BLOCKED_OR_FAILED'
    RECEIPT['error']=f'{type(exc).__name__}: {exc}'
finally:
    RECEIPT['elapsed_s']=time.monotonic()-START
    RECEIPT['directory_bytes']=sum(p.stat().st_size for p in HERE.iterdir() if p.is_file())
    require(RECEIPT['directory_bytes']<100_000_000,'artifact budget exceeded')
    (HERE/'SYNTHETIC_QA_receipt_final.json').write_text(json.dumps(RECEIPT,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(RECEIPT))

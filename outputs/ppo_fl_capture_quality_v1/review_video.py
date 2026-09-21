"""Small sealed-only FL review; reuse existing native remux/decoding checks.

Panels stream into ffmpeg: no directory of thousands of repeated panel files.
No physical replay, policy forward, time warping, camera crop or interpolation.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('sealed_review', ROOT/'outputs/video_review_v1/sealed_rr_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
base = review.base
require = base.require
USED_SOURCE_ARTIFACTS = (
    'actual_viewport_video.mp4', 'viewport_buffer_video_manifest.json',
    'viewport_frame_ledger.jsonl', 'video_policy_decisions.jsonl',
    'physical_observations.jsonl', 'native_tick_audit.jsonl',
    'height_diagnostics.jsonl', 'height_diagnostics_startup.json',
)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checkpoint_identity(proof):
    """Bind a C capture to the exact persisted checkpoint and manifest."""
    require(proof.get('checkpoint_loaded_and_verified') is True,
            'C source has no verified official checkpoint load')
    decisions = proof.get('saved_global_policy_decisions')
    require(type(decisions) is int and decisions > 178432,
            'C source is not a post-178432 learned checkpoint')
    binding = proof.get('source') or {}
    checkpoint = Path(binding.get('checkpoint', '')).resolve()
    manifest_path = Path(binding.get('manifest', '')).resolve()
    require(checkpoint.is_file() and manifest_path.is_file(),
            'C checkpoint or checkpoint manifest is missing')
    require(sha256(checkpoint) == binding.get('checkpoint_sha256')
            and sha256(manifest_path) == binding.get('manifest_sha256'),
            'C checkpoint binding changed after capture')
    metadata = base.read_json(manifest_path)
    require(Path(metadata.get('checkpoint_path', '')).resolve() == checkpoint
            and metadata.get('checkpoint_sha256') == binding['checkpoint_sha256'],
            'Checkpoint manifest does not bind the loaded checkpoint')
    require(metadata.get('global_policy_decisions') == decisions,
            'Capture and checkpoint manifest decision counts differ')
    actor_hash = (proof.get('parameter_hashes') or {}).get('actor_parameter_sha256')
    require(bool(actor_hash) and metadata.get('actor_parameter_sha256') == actor_hash,
            'Capture and checkpoint manifest actor hashes differ')
    branch = metadata.get('fl_capture_quality_branch') or {}
    origin = branch.get('counter_origin') or {}
    counts = metadata.get('fl_capture_quality_branch_counts') or {}
    counter_names = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')
    require(branch.get('branch_id') == 'fl_capture_quality_v1'
            and all(type(origin.get(name)) is int and type(counts.get(name)) is int
                    and type(metadata.get(name)) is int
                    and counts[name] > 0
                    and metadata.get(name)-origin[name] == counts[name]
                    for name in counter_names),
            'C checkpoint has no positive verified FL-quality optimizer branch')
    return {
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': binding['checkpoint_sha256'],
        'manifest': str(manifest_path),
        'manifest_sha256': binding['manifest_sha256'],
        'saved_global_policy_decisions': decisions,
        'actor_parameter_sha256': actor_hash,
        'fl_capture_quality_branch_counts': {name: counts[name] for name in counter_names},
    }


def sealed_source(source):
    """Return a sealed input context; no condition is a removable assert."""
    source = Path(source).resolve(strict=True)
    run_path = source.parent/'run_manifest.json'
    require(run_path.is_file(), 'Missing parent run manifest')
    run = base.read_json(run_path)
    require(bool(run.get('completed_at_utc')) and run.get('lifecycle') != 'RUNNING',
            'Source run is not sealed')
    manifest_path = source/'semantic_video_source_manifest.json'
    require(manifest_path.is_file(), 'Missing semantic video source manifest')
    manifest = base.read_json(manifest_path)
    require(manifest.get('experiment_id') == 'fl_capture_quality_v1',
            'Wrong experiment family')
    require(manifest.get('from_phase') == 'P01' and manifest.get('episode_count') == 1,
            'Review requires one natural P01 episode')
    require(manifest.get('fresh_process_single_episode') is True
            and manifest.get('optimizer_updates') == 0,
            'Review source is not a fresh evaluation-only process')
    require(manifest.get('diagnostic_intervention') is None,
            'Diagnostic intervention cannot be formal review media')
    require(run.get('runtime_contract') == manifest.get('runtime_contract'),
            'Run/source runtime contract mismatch')
    role = manifest.get('role')
    require(role in ('B', 'C'), 'Only same-version B_control or C is supported')
    mode = manifest.get('policy_sampling_mode') or manifest.get('sampling_mode')
    checkpoint = None
    if role == 'C':
        proof = manifest.get('checkpoint_load_provenance') or {}
        checkpoint = checkpoint_identity(proof)
        mode = mode or proof.get('policy_sampling_mode') or proof.get('sampling_mode')
        require(mode in ('deterministic_conditional_mean',
                         'training_style_conditional_gaussian'),
                'C sampling mode must be explicit')
        if mode == 'training_style_conditional_gaussian':
            require(proof.get('stochastic_policy') is True
                    and type(proof.get('policy_seed')) is int,
                    'Stochastic C requires an explicit independent policy seed')
        else:
            require(proof.get('stochastic_policy') in (None, False)
                    and proof.get('policy_seed') is None,
                    'Deterministic C must not carry a stochastic sampling seed')
    else:
        mode = 'N_plus_zero'
    return {'source': source, 'run': run, 'manifest': manifest, 'role': role,
            'mode': mode, 'checkpoint': checkpoint,
            'source_manifest_sha256': sha256(manifest_path)}


def export(source, destination):
    context = sealed_source(source)
    source, manifest = context['source'], context['manifest']
    role, mode = context['role'], context['mode']
    proof = manifest.get('checkpoint_load_provenance') or {}
    destination.mkdir(parents=True, exist_ok=False)
    native = destination/'full_native.mp4'
    base.export(source, native, 'B0' if role == 'B' else 'C0')
    receipt = base.read_json(native.with_suffix('.media.json'))
    ledger = base.load_viewport_frame_ledger(source/'viewport_frame_ledger.jsonl')
    frames = review.frame_data(source, manifest, ledger)
    require(receipt['frame_count'] == len(frames) == len(ledger),
            'Native receipt, frame evidence and viewport ledger counts differ')
    require(0 < receipt['physical_duration_s'] <= 200,
            'Full episode is empty or exceeds 200 seconds')
    tasks = {}
    for _, row in review.checked_rows(source, manifest, 'video_policy_decisions.jsonl'):
        task = (row.get('step_info') or {}).get('semantic_task') or {}
        tasks[row['end_tick']] = task.get('physical_evaluator') or {}
    final = manifest['physical_episode']['physical_task_evaluation']
    tasks[manifest['episode_physics_ticks']] = final
    for row in frames:
        ev = tasks.get(row['actual_physics_tick'], {})
        exact = ev.get('physics_tick') == row['actual_physics_tick']
        row['FL'] = ev.get('current_legs', {}).get('FL', {}) if exact else {}
        row['FR'] = ev.get('current_legs', {}).get('FR', {}) if exact else {}
        row['placed'] = ev.get('history', {}).get('placed', {}) if exact else {}
    success = manifest['physical_task_success'] is True
    stem = ('B_control_Nplus0' if role == 'B' else 'ppo_smooth_exploration_stochastic_labeled'
            if mode == 'training_style_conditional_gaussian'
            else 'ppo_latest_full_episode')
    output = destination/(stem + ('.mp4' if success else '_incomplete.mp4'))
    label = ('B CONTROL / N + 0' if role == 'B' else 'PPO FULL12 CP' + str(proof['saved_global_policy_decisions']))
    label += ' | ' + ('STOCHASTIC' if mode == 'training_style_conditional_gaussian' else 'DETERMINISTIC' if role == 'C' else 'ZERO')
    label += ' | ' + receipt['physical_result']
    count = len(frames)
    command = [str(base.find_ffmpeg()), '-hide_banner', '-nostdin', '-v', 'error', '-n',
        '-threads', '2', '-filter_threads', '1', '-filter_complex_threads', '1', '-i', str(native),
        '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1280x150', '-framerate', '15', '-i', 'pipe:0',
        '-filter_complex', '[0:v]setpts=PTS-STARTPTS[v];[1:v]setpts=PTS-STARTPTS[p];[v][p]vstack=inputs=2[out]',
        '-map', '[out]', '-an', '-frames:v', str(count), '-r', '15', '-fps_mode', 'cfr',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-threads', '2', '-movflags', '+faststart', str(output)]
    font = ImageFont.truetype('C:/Windows/Fonts/consola.ttf', 19)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for row in frames:
            panel = Image.new('RGB', (1280, 150), 'black')
            draw = ImageDraw.Draw(panel)
            fl, fr = row['FL'], row['FR']
            fmt = lambda x, s=1.: 'N/A' if x is None else f'{x*s:+.3f}'
            lines = [label,
                f"1x / 15fps | t={row['actual_sim_time_s']:.3f}s | {row['phase']} | original whole-body view retained",
                'Actual native wheel qd rad/s: ' + '  '.join(f'{leg} {v:+.3f}' for leg, v in zip(review.LEGS, row['measured_native_qd_rad_s'])),
                f"FL gap {fmt(fl.get('clearance_m'),1000)} mm | {'AIR' if fl.get('air') is True else fl.get('contact_surface','N/A')} | bearing {fmt(fl.get('bearing_force_n'))} N | placed {row['placed'].get('FL','N/A')}",
                f"FR gap {fmt(fr.get('clearance_m'),1000)} mm | placed {row['placed'].get('FR','N/A')} | FL/FR front left/right; RL/RR rear left/right",
                'Wheel qd is joint rotation, not endpoint translation. AIR wheel rotation does not establish traction.']
            for i, line in enumerate(lines):
                require(draw.textlength(line, font=font) <= 1248,
                        'Panel text exceeds the external annotation margin: '+line)
                draw.text((16, 2+24*i), line, font=font, fill='#ffff80' if i == 0 else 'white')
            require(process.poll() is None, 'ffmpeg exited before all data panels were streamed')
            process.stdin.write(panel.tobytes())
        process.stdin.close()
        error = process.stderr.read().decode(errors='replace')
        require(process.wait() == 0, 'Panel encode failed: '+error[-2500:])
    finally:
        if process.poll() is None:
            process.terminate()
    validation = review.validate_video(output, count, 1280, 870)
    evidence_path = destination/'frame_evidence.json'
    base.write_new_json(evidence_path, {
        'schema': 'wlr50_clean.fl_capture_quality_frame_evidence.v1',
        'frames': frames, 'source': str(source),
        'source_manifest': str(source/'semantic_video_source_manifest.json'),
        'source_manifest_sha256': context['source_manifest_sha256'],
        'no_forward_fill': True,
    })
    artifact_bindings = {}
    for name in USED_SOURCE_ARTIFACTS:
        require(name in manifest['artifacts'], 'Source manifest lacks required artifact '+name)
        artifact_bindings[name] = manifest['artifacts'][name]['sha256']
    result = {'schema': 'wlr50_clean.fl_capture_quality_review.v1',
        'output': str(output), 'source_manifest': str(source/'semantic_video_source_manifest.json'),
        'source_manifest_sha256': context['source_manifest_sha256'],
        'native_receipt': str(native.with_suffix('.media.json')), 'role': role, 'mode': mode,
        'checkpoint_identity': context['checkpoint'],
        'physical_result': receipt['physical_result'], 'physical_duration_s': manifest['episode_physics_ticks']/120.,
        'physical_task_success': success,
        'frame_count': count, 'first_tick': ledger[0].sim_step, 'last_tick': ledger[-1].sim_step,
        'normal_speed': True, 'speed_modified': False, 'stitched': False,
        'full_episode_preserved': True, 'single_episode': True, 'diagnostic_intervention': None,
        'source_git_commit': manifest['runtime_contract']['source_git_commit'],
        'selected_configuration': manifest['runtime_contract']['selected_configuration'],
        'evaluation_configuration': manifest['evaluation_configuration'],
        'used_source_artifact_sha256': artifact_bindings,
        'frame_evidence': str(evidence_path),
        'frame_evidence_sha256': sha256(evidence_path),
        'physical_events': final.get('history', {}).get('event_ticks', {}),
        'validation': base.compact_validation(validation),
        'previews': base.previews(output, count, base.find_ffmpeg())}
    base.write_new_json(output.with_suffix('.media.json'), result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    export(args.source, args.destination)

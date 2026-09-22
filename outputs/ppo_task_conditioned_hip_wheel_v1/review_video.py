"""Small sealed-only task-conditioned review; old review rules stay untouched.

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
OUT = Path(__file__).resolve().parent
EXPERIMENT = 'task_conditioned_hip_wheel_v1'
POLICY_VERSION = 'task_conditioned_hip_wheel_sigma_v1'
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
    require(proof.get('checkpoint_loaded_and_verified') is True
            and proof.get('official_load_semantic_checkpoint') is True
            and proof.get('policy_version') == POLICY_VERSION
            and proof.get('optimizer_updates') == 0,
            'C source has no verified official checkpoint load')
    decisions = proof.get('saved_global_policy_decisions')
    require(type(decisions) is int and decisions > 0,
            'C source has no saved positive decision count')
    binding = proof.get('source') or {}
    checkpoint = Path(binding.get('checkpoint', '')).resolve()
    manifest_path = Path(binding.get('manifest', '')).resolve()
    require(checkpoint.is_file() and manifest_path.is_file(),
            'C checkpoint or checkpoint manifest is missing')
    require(checkpoint.is_relative_to((OUT/'checkpoints/history').resolve())
            and checkpoint.parent == manifest_path.parent
            and not checkpoint.name.startswith('checkpoint_initial_'),
            'C must use a learned checkpoint from this isolated task branch')
    require(sha256(checkpoint) == binding.get('checkpoint_sha256')
            and sha256(manifest_path) == binding.get('manifest_sha256'),
            'C checkpoint binding changed after capture')
    metadata = base.read_json(manifest_path)
    require(Path(metadata.get('checkpoint_path', '')).resolve() == checkpoint
            and metadata.get('checkpoint_sha256') == binding['checkpoint_sha256'],
            'Checkpoint manifest does not bind the loaded checkpoint')
    require(metadata.get('global_policy_decisions') == decisions,
            'Capture and checkpoint manifest decision counts differ')
    require(metadata.get('save_load_round_trip') is True
            and metadata.get('runtime_contract', {}).get('experiment_id') == EXPERIMENT
            and metadata.get('policy_contract', {}).get('version') == POLICY_VERSION
            and metadata.get('policy_contract') == proof.get('policy_contract'),
            'C checkpoint has the wrong experiment/distribution or no verified roundtrip')
    actor_hash = (proof.get('parameter_hashes') or {}).get('actor_parameter_sha256')
    require(bool(actor_hash) and metadata.get('actor_parameter_sha256') == actor_hash,
            'Capture and checkpoint manifest actor hashes differ')
    branch = metadata.get('task_conditioned_hip_wheel_branch') or {}
    origin = branch.get('counter_origin') or {}
    counts = metadata.get('task_conditioned_hip_wheel_branch_counts') or {}
    counter_names = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')
    require(branch.get('branch_id') == EXPERIMENT
            and all(type(origin.get(name)) is int and type(counts.get(name)) is int
                    and type(metadata.get(name)) is int
                    and counts[name] > 0
                    and metadata.get(name)-origin[name] == counts[name]
                    for name in counter_names),
            'C checkpoint has no positive verified task-conditioned optimizer branch')
    return {
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': binding['checkpoint_sha256'],
        'manifest': str(manifest_path),
        'manifest_sha256': binding['manifest_sha256'],
        'saved_global_policy_decisions': decisions,
        'actor_parameter_sha256': actor_hash,
        'task_conditioned_hip_wheel_branch_counts': {name: counts[name] for name in counter_names},
        'policy_version': POLICY_VERSION, 'runtime_contract': metadata['runtime_contract'],
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
    require(manifest.get('experiment_id') == EXPERIMENT,
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
    require(manifest.get('runtime_contract', {}).get('experiment_id') == EXPERIMENT
            and manifest.get('seed') == 4001,
            'Review requires the isolated task runtime and declared video scene seed')
    role = manifest.get('role')
    require(role in ('B', 'C'), 'Only same-version B_control or C is supported')
    mode = manifest.get('policy_sampling_mode') or manifest.get('sampling_mode')
    checkpoint = None
    if role == 'C':
        proof = manifest.get('checkpoint_load_provenance') or {}
        checkpoint = checkpoint_identity(proof)
        require(checkpoint['runtime_contract'] == manifest.get('runtime_contract'),
                'Formal new-branch review requires exact saved checkpoint/evaluation runtime')
        mode = mode or proof.get('policy_sampling_mode') or proof.get('sampling_mode')
        require(mode in ('deterministic_conditional_mean',
                         'training_style_conditional_gaussian'),
                'C sampling mode must be explicit')
        require(mode == proof.get('policy_sampling_mode'),
                'Source mode and actual checkpoint sampling proof disagree')
        if mode == 'training_style_conditional_gaussian':
            require(proof.get('stochastic_policy') is True
                    and type(proof.get('policy_seed')) is int
                    and 0 <= proof['policy_seed'] <= 2147483647,
                    'Stochastic C requires an explicit independent policy seed')
        else:
            require(proof.get('stochastic_policy') in (None, False)
                    and proof.get('policy_seed') is None,
                    'Deterministic C must not carry a stochastic sampling seed')
    else:
        require(manifest.get('checkpoint_load_provenance') is None,
                'B must not load a learned checkpoint')
        mode = 'N_plus_zero'
    return {'source': source, 'run': run, 'manifest': manifest, 'role': role,
            'mode': mode, 'checkpoint': checkpoint,
            'source_manifest_sha256': sha256(manifest_path)}


def output_name(role, mode, proof):
    if role == 'B':
        return 'B_current_Nplus0_P01_full.mp4'
    cp = proof['saved_global_policy_decisions']
    return (f'CP{cp}_stochastic_P01_full_seed{proof["policy_seed"]}.mp4'
            if mode == 'training_style_conditional_gaussian'
            else f'CP{cp}_deterministic_P01_full.mp4')


def export(source, destination):
    context = sealed_source(source)
    source, manifest = context['source'], context['manifest']
    role, mode = context['role'], context['mode']
    proof = manifest.get('checkpoint_load_provenance') or {}
    destination = Path(destination).resolve()
    require(destination.is_relative_to(OUT) and destination != OUT,
            'Derived media must stay inside a new isolated task output directory')
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
    output = destination/output_name(role, mode, proof)
    label = ('B CONTROL / N + 0' if role == 'B' else 'PPO FULL12 CP' + str(proof['saved_global_policy_decisions']))
    label += ' | ' + ('STOCHASTIC' if mode == 'training_style_conditional_gaussian' else 'DETERMINISTIC' if role == 'C' else 'ZERO')
    if mode == 'training_style_conditional_gaussian':
        label += ' seed ' + str(proof['policy_seed'])
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
        'schema': 'wlr50_clean.task_conditioned_hip_wheel_frame_evidence.v1',
        'frames': frames, 'source': str(source),
        'source_manifest': str(source/'semantic_video_source_manifest.json'),
        'source_manifest_sha256': context['source_manifest_sha256'],
        'no_forward_fill': True,
    })
    artifact_bindings = {}
    for name in USED_SOURCE_ARTIFACTS:
        require(name in manifest['artifacts'], 'Source manifest lacks required artifact '+name)
        artifact_bindings[name] = manifest['artifacts'][name]['sha256']
    result = {'schema': 'wlr50_clean.task_conditioned_hip_wheel_review.v1',
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
        'experiment_id': EXPERIMENT,
        'neutral_full_filename_does_not_claim_task_success': True,
        'missing_unsynchronized_body_geometry_or_hip_overlay': None,
        'validation': base.compact_validation(validation),
        'previews': base.previews(output, count, base.find_ffmpeg())}
    base.write_new_json(output.with_suffix('.media.json'), result)
    print(json.dumps(result, indent=2))
    return result


def checked_review(path):
    """Rebind already exported new-family media before organizing comparisons."""
    path = Path(path).resolve(strict=True)
    receipt = base.read_json(path)
    require(receipt.get('schema') == 'wlr50_clean.task_conditioned_hip_wheel_review.v1',
            'Pair input must be the new task-conditioned review, not historical media')
    output = Path(receipt.get('output', '')).resolve(strict=True)
    source_manifest = Path(receipt.get('source_manifest', '')).resolve(strict=True)
    require(sha256(output) == receipt.get('validation', {}).get('sha256')
            and sha256(source_manifest) == receipt.get('source_manifest_sha256'),
            'Reviewed video/source manifest changed')
    require(receipt.get('normal_speed') is True and receipt.get('speed_modified') is False
            and receipt.get('stitched') is False and receipt.get('single_episode') is True
            and receipt.get('full_episode_preserved') is True and receipt.get('validation', {}).get('valid') is True,
            'Not a validated normal-speed complete single attempt')
    context = sealed_source(source_manifest.parent)
    require(context['role'] == receipt.get('role') and context['mode'] == receipt.get('mode')
            and context['checkpoint'] == receipt.get('checkpoint_identity'),
            'Review identity no longer matches its sealed source')
    evidence_path = Path(receipt.get('frame_evidence', '')).resolve(strict=True)
    require(sha256(evidence_path) == receipt.get('frame_evidence_sha256'), 'Frame evidence changed')
    evidence = base.read_json(evidence_path)
    require(evidence.get('schema') == 'wlr50_clean.task_conditioned_hip_wheel_frame_evidence.v1'
            and evidence.get('no_forward_fill') is True
            and evidence.get('source_manifest_sha256') == receipt['source_manifest_sha256'],
            'Frame evidence belongs to a different source or lacks exact timestamps')
    rows = evidence.get('frames') or []
    require(len(rows) == receipt.get('frame_count') and len(rows) > 0
            and [r['source_frame_index'] for r in rows] == list(range(len(rows)))
            and all(b['actual_physics_tick'] > a['actual_physics_tick'] for a,b in zip(rows,rows[1:])),
            'Full frame evidence is not contiguous or monotonically timed')
    bindings = receipt.get('used_source_artifact_sha256') or {}
    require(set(bindings) == set(USED_SOURCE_ARTIFACTS), 'Incomplete used-source artifact bindings')
    for name, expected in bindings.items():
        require(sha256(source_manifest.parent/name) == expected, 'Source artifact changed: '+name)
    return {'receipt_path':path, 'receipt':receipt, 'context':context,
            'manifest':context['manifest'], 'rows':rows}


def strict_common_capture(left, right):
    a,b = left['manifest'],right['manifest']
    require(a.get('experiment_id') == b.get('experiment_id') == EXPERIMENT, 'Pair experiment mismatch')
    for key in ('runtime_contract','evaluation_configuration','camera'):
        require(a.get(key) is not None and a[key] == b.get(key), 'Pair '+key+' differs')
    require(a.get('seed') == b.get('seed') == 4001, 'Pair scene/reset seeds differ')
    entry = a.get('natural_reset_proof', {}).get('entry')
    require(entry is not None and entry == b.get('natural_reset_proof', {}).get('entry'),
            'Pair natural reset entry differs')
    return {'same_runtime_contract':a['runtime_contract'], 'same_camera':a['camera'],
            'same_evaluation_configuration':a['evaluation_configuration'], 'same_scene_seed':4001,
            'same_measured_initial_state_claimed':False, 'single_pair_is_not_statistical_evidence':True}


def strict_policy_modes(deterministic, stochastic):
    require(deterministic['receipt'].get('role') == stochastic['receipt'].get('role') == 'C'
            and deterministic['receipt'].get('mode') == 'deterministic_conditional_mean'
            and stochastic['receipt'].get('mode') == 'training_style_conditional_gaussian',
            'Policy-mode inputs must be deterministic then labeled stochastic C')
    common = strict_common_capture(deterministic, stochastic)
    identity = deterministic['receipt'].get('checkpoint_identity')
    require(identity is not None and identity == stochastic['receipt'].get('checkpoint_identity'),
            'Different checkpoints must not be presented as a same-model mode comparison')
    return {**common, 'same_checkpoint':identity,
            'stochastic_policy_seed':stochastic['manifest']['checkpoint_load_provenance']['policy_seed'],
            'stochastic_is_not_deterministic_success':True}


def verify_modes(deterministic_path, stochastic_path, output):
    deterministic, stochastic = checked_review(deterministic_path), checked_review(stochastic_path)
    result = {'schema':'wlr50_clean.task_conditioned_policy_mode_pair.v1',
              **strict_policy_modes(deterministic, stochastic),
              'deterministic_receipt':str(deterministic['receipt_path']),
              'stochastic_receipt':str(stochastic['receipt_path'])}
    output = Path(output).resolve()
    require(output.is_relative_to(OUT), 'Mode receipt must remain under isolated outputs')
    base.write_new_json(output,result)
    return result


def build_pair(b_path, deterministic_path, destination):
    b,c = checked_review(b_path),checked_review(deterministic_path)
    require(b['receipt'].get('role') == 'B' and b['receipt'].get('mode') == 'N_plus_zero'
            and c['receipt'].get('role') == 'C' and c['receipt'].get('mode') == 'deterministic_conditional_mean',
            'N comparison requires current same-version zero B and deterministic C')
    common = strict_common_capture(b,c)
    destination = Path(destination).resolve()
    require(destination.is_relative_to(OUT) and destination != OUT, 'Pair must use new isolated directory')
    destination.mkdir(parents=True,exist_ok=False)
    # Reuse only the established encoder, not its old-family acceptance gates.
    # This function preserves all input frames and labels every shorter-side freeze.
    spec = importlib.util.spec_from_file_location('old_pair_encoder_only',ROOT/'outputs/ppo_fl_capture_quality_v1/paired_event_media.py')
    encoder = importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder)
    cp = c['receipt']['checkpoint_identity']['saved_global_policy_decisions']
    full = encoder._comparison_video(b,c,destination/f'N_vs_CP{cp}_deterministic.mp4',
        left_end=b['receipt']['frame_count'],right_end=c['receipt']['frame_count'],
        kind='full_attempt_same_elapsed_P01_not_event_timewarped')
    result = {'schema':'wlr50_clean.task_conditioned_control_pair.v1',
              'B_receipt':str(b['receipt_path']), 'deterministic_C_receipt':str(c['receipt_path']),
              'strict_same_version_contract':common, 'full_episode':full,
              'B_physical_result':b['receipt']['physical_result'], 'C_physical_result':c['receipt']['physical_result'],
              'B_physical_duration_s':b['receipt']['physical_duration_s'],
              'C_physical_duration_s':c['receipt']['physical_duration_s'],
              'quality_improvement_claim':None, 'task_success_not_inferred_from_media':True,
              'historical_N_ref_not_relabelled_as_current_B':True}
    base.write_new_json(destination/'pair_receipt.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    single = commands.add_parser('export')
    single.add_argument('--source',type=Path,required=True)
    single.add_argument('--destination',type=Path,required=True)
    modes = commands.add_parser('verify-modes')
    modes.add_argument('--deterministic-receipt',type=Path,required=True)
    modes.add_argument('--stochastic-receipt',type=Path,required=True)
    modes.add_argument('--output',type=Path,required=True)
    pair = commands.add_parser('pair')
    pair.add_argument('--b-receipt',type=Path,required=True)
    pair.add_argument('--deterministic-receipt',type=Path,required=True)
    pair.add_argument('--destination',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'export':
        export(args.source,args.destination)
    elif args.command == 'verify-modes':
        print(json.dumps(verify_modes(args.deterministic_receipt,args.stochastic_receipt,args.output),indent=2))
    else:
        print(json.dumps(build_pair(args.b_receipt,args.deterministic_receipt,args.destination),indent=2))

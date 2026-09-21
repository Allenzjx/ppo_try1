"""Build bounded offline B/C comparisons and contiguous FL event clips.

Inputs are receipts produced by ``review_video.py`` from already sealed runs.
This helper never starts Isaac or a policy.  It rejects cross-version B/C
comparisons, labels stochastic evidence separately, preserves full attempts,
and never removes time from inside an event interval.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fl_review_video', OUT/'review_video.py')
review_video = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review_video)
base = review_video.base
require = base.require
FPS = 15
quality_spec = importlib.util.spec_from_file_location('fl_event_quality', OUT/'event_quality.py')
event_quality = importlib.util.module_from_spec(quality_spec)
quality_spec.loader.exec_module(event_quality)


def read_json(path):
    return base.read_json(Path(path))


def checked_review(receipt_path):
    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = read_json(receipt_path)
    require(receipt.get('schema') == 'wlr50_clean.fl_capture_quality_review.v1',
            'Input is not an FL-quality review receipt')
    output = Path(receipt.get('output', '')).resolve()
    source_manifest = Path(receipt.get('source_manifest', '')).resolve()
    require(output.is_file() and source_manifest.is_file(),
            'Review video or source manifest is missing')
    require(review_video.sha256(output) == receipt.get('validation', {}).get('sha256'),
            'Review video changed after validation')
    require(review_video.sha256(source_manifest) == receipt.get('source_manifest_sha256'),
            'Source manifest changed after review export')
    require(receipt.get('normal_speed') is True
            and receipt.get('speed_modified') is False
            and receipt.get('stitched') is False
            and receipt.get('single_episode') is True
            and receipt.get('full_episode_preserved') is True
            and receipt.get('validation', {}).get('valid') is True,
            'Input is not a valid full, normal-speed single episode')
    context = review_video.sealed_source(source_manifest.parent)
    require(context['role'] == receipt.get('role')
            and context['mode'] == receipt.get('mode')
            and context['checkpoint'] == receipt.get('checkpoint_identity'),
            'Review receipt no longer agrees with its sealed source')
    evidence_path = Path(receipt.get('frame_evidence', '')).resolve()
    require(evidence_path.is_file()
            and review_video.sha256(evidence_path) == receipt.get('frame_evidence_sha256'),
            'Frame evidence changed after review export')
    bindings = receipt.get('used_source_artifact_sha256') or {}
    require(set(bindings) == set(review_video.USED_SOURCE_ARTIFACTS),
            'Review receipt lacks exact source artifact bindings')
    source = source_manifest.parent
    for name, expected in bindings.items():
        require(review_video.sha256(source/name) == expected,
                'Sealed source artifact changed after review: '+name)
    evidence = read_json(evidence_path)
    require(evidence.get('schema') == 'wlr50_clean.fl_capture_quality_frame_evidence.v1'
            and evidence.get('source_manifest_sha256') == receipt['source_manifest_sha256']
            and evidence.get('no_forward_fill') is True,
            'Frame evidence is missing or belongs to another source')
    rows = evidence.get('frames') or []
    require(len(rows) == receipt['frame_count'] > 0,
            'Frame evidence count differs from media')
    require([row['source_frame_index'] for row in rows] == list(range(len(rows))),
            'Frame evidence indices are not contiguous')
    require(all(b['actual_physics_tick'] > a['actual_physics_tick']
                for a, b in zip(rows, rows[1:])),
            'Frame evidence ticks are not strictly increasing')
    return {'receipt_path': receipt_path, 'receipt': receipt, 'context': context,
            'manifest': context['manifest'], 'rows': rows}


def _strict_common_capture(left, right):
    a, b = left['manifest'], right['manifest']
    require(a['experiment_id'] == b['experiment_id'] == 'fl_capture_quality_v1',
            'Pair is not from the FL-quality experiment')
    require(a['seed'] == b['seed'] == 4001, 'Pair reset seeds differ')
    require(a['camera'] == b['camera'], 'Pair cameras differ')
    require(a['runtime_contract'] == b['runtime_contract'],
            'Pair runtime/source configuration is not byte-identical by contract')
    require(a['evaluation_configuration'] == b['evaluation_configuration'],
            'Pair evaluation configuration hashes differ')
    require(a.get('natural_reset_proof', {}).get('entry')
            == b.get('natural_reset_proof', {}).get('entry'),
            'Pair logical reset entries differ')
    return {
        'same_source_git_commit': a['runtime_contract']['source_git_commit'],
        'same_runtime_content_sha256': a['runtime_contract']['runtime_content_sha256'],
        'same_selected_configuration': a['runtime_contract']['selected_configuration'],
        'same_evaluation_configuration': a['evaluation_configuration'],
        'same_seed': a['seed'], 'same_camera': a['camera'],
        'identical_initial_measured_state_claimed': False,
    }


def strict_control_pair(b_control, c_policy):
    require(b_control['receipt']['role'] == 'B'
            and b_control['receipt']['mode'] == 'N_plus_zero',
            'Left input must be same-version B_control N+0')
    require(c_policy['receipt']['role'] == 'C'
            and c_policy['receipt']['mode'] == 'deterministic_conditional_mean',
            'Main B/C comparison requires deterministic C')
    return _strict_common_capture(b_control, c_policy)


def strict_policy_mode_pair(deterministic, stochastic):
    require(deterministic['receipt']['role'] == stochastic['receipt']['role'] == 'C',
            'Policy-mode pair must contain two C evaluations')
    require(deterministic['receipt']['mode'] == 'deterministic_conditional_mean'
            and stochastic['receipt']['mode'] == 'training_style_conditional_gaussian',
            'Policy-mode pair must be deterministic then explicitly stochastic')
    common = _strict_common_capture(deterministic, stochastic)
    require(deterministic['receipt']['checkpoint_identity']
            == stochastic['receipt']['checkpoint_identity'],
            'Deterministic/stochastic videos do not use the same checkpoint')
    return {**common,
            'same_checkpoint': deterministic['receipt']['checkpoint_identity'],
            'stochastic_policy_seed':
                stochastic['manifest']['checkpoint_load_provenance']['policy_seed'],
            'stochastic_is_separate_labeled_evidence': True,
            'stochastic_is_not_used_for_deterministic_quality_claim': True}


def _new_directory(path):
    destination = Path(path).resolve()
    require(destination.is_relative_to(OUT.resolve()),
            'All derived media must remain under outputs/ppo_fl_capture_quality_v1')
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _event_frame(rows, tick):
    require(type(tick) is int, 'Physical event tick is unavailable')
    return next((i for i, row in enumerate(rows)
                 if row['actual_physics_tick'] >= tick), len(rows)-1)


def fl_clip_interval(rows, events):
    """Select one contiguous P04/P05-through-P06 (or failure-tail) interval."""
    require(bool(rows), 'FL event clip has no frames')
    placed_tick = events.get('placed', {}).get('FL')
    p06 = next((i for i, row in enumerate(rows)
                if row['phase'] == 'P06'
                and type(placed_tick) is int
                and row['actual_physics_tick'] >= placed_tick), None)
    genuine = type(placed_tick) is int and p06 is not None
    start = next((i for i, row in enumerate(rows)
                  if row['phase'] in ('P04', 'P05')), 0)
    if genuine:
        end = next((i for i, row in enumerate(rows[p06:], p06)
                    if row['phase'] not in ('P05', 'P06')), len(rows))
        end = max(p06+1, end)
    else:
        end = len(rows)  # retain the complete failure stall/tail
    return start, end, genuine, placed_tick


def _quality_pair(b_path, c_path, b_control, c_policy):
    require(__debug__, 'Quality recomputation must not run with Python -O')
    b_path, c_path = Path(b_path).resolve(strict=True), Path(c_path).resolve(strict=True)
    b, c = read_json(b_path), read_json(c_path)
    require(Path(b['source']).resolve() == Path(b_control['context']['source']).resolve()
            and Path(c['source']).resolve() == Path(c_policy['context']['source']).resolve(),
            'Quality records do not belong to the paired sources')
    # Recompute from the now re-hashed sealed artifacts.  A hand-edited metric
    # JSON cannot become the basis for a quality claim.
    b_recomputed = event_quality.analyze(b_control['context']['source'])
    c_recomputed = event_quality.analyze(c_policy['context']['source'])
    fields = ('experiment_id', 'role', 'sampling_mode', 'full_task_success',
              'physical_duration_s', 'events_first_observed', 'windows',
              'task_result_not_inferred_from_quality', 'missing_stages_are_not_zero')
    require(all(b.get(key) == b_recomputed.get(key) for key in fields)
            and all(c.get(key) == c_recomputed.get(key) for key in fields),
            'Quality JSON differs from independent event-aligned recomputation')
    key = 'FR_preparation_to_capture'
    bw, cw = b['windows'][key], c['windows'][key]
    require(bw['complete'] is True and cw['complete'] is True,
            'P01/P02 comparison requires real FR placement in both episodes')
    return {
        'protocol': str(OUT/'QUALITY_PROTOCOL.md'), 'window': key,
        'B_quality_record': str(b_path), 'B_quality_sha256': review_video.sha256(b_path),
        'C_quality_record': str(c_path), 'C_quality_sha256': review_video.sha256(c_path),
        'primary_metric': 'combined_chassis_roll_pitch_euler_rate_rms_rad_s',
        'B_control': bw, 'C_deterministic': cw,
        'C_minus_B_rate_rms_rad_s': cw['rate_rms_rad_s']-bw['rate_rms_rad_s'],
        'primary_metric_improved': cw['rate_rms_rad_s'] < bw['rate_rms_rad_s'],
        'duration_disclosed': True, 'one_pair_is_not_robustness': True,
    }


def _comparison_video(left, right, output, *, left_end, right_end, kind):
    require(1 < left_end <= left['receipt']['frame_count']
            and 1 < right_end <= right['receipt']['frame_count'],
            'Invalid contiguous comparison frame interval')
    count = max(left_end, right_end)
    require(count/FPS <= 200, 'Comparison exceeds 200 seconds')
    filters = []
    for index, (item, end) in enumerate(((left, left_end), (right, right_end))):
        # Infinite clone padding is bounded by -frames:v below.  It never claims
        # new physics and avoids off-by-one duration behavior across ffmpeg builds.
        filters.append(
            f'[{index}:v]trim=start_frame=0:end_frame={end},setpts=PTS-STARTPTS,'
            'scale=960:652,pad=960:688:0:0:black,'
            f'tpad=stop_mode=clone:stop=-1,setpts=N/(15*TB),drawtext='
            "fontfile='C\\:/Windows/Fonts/arial.ttf':"
            "text='RUN WINDOW ENDED - FROZEN FRAME':fontcolor=yellow:fontsize=18:x=12:y=662:"
            f"enable='gte(n,{end})'[v{index}]"
        )
    filters.append('[v0][v1]hstack=inputs=2:shortest=1,format=yuv420p[out]')
    command = [str(base.find_ffmpeg()), '-hide_banner', '-nostdin', '-v', 'error', '-n',
        '-threads', '2', '-filter_threads', '1', '-filter_complex_threads', '1',
        '-i', left['receipt']['output'], '-i', right['receipt']['output'],
        '-filter_complex', ';'.join(filters), '-map', '[out]', '-an',
        '-frames:v', str(count), '-r', '15', '-fps_mode', 'cfr', '-c:v', 'libx264',
        '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', '-threads', '2',
        '-movflags', '+faststart', str(output)]
    base.run(command)
    validation = review_video.review.validate_video(output, count, 1920, 688)
    return {'output': str(output), 'kind': kind, 'frame_count': count,
        'source_frame_intervals_half_open': [[0, left_end], [0, right_end]],
        'freeze_added_frames': [count-left_end, count-right_end],
        'freeze_is_physical_evidence': False, 'normal_speed': True,
        'speed_modified': False, 'temporal_stitched': False,
        'spatial_side_by_side': True, 'same_elapsed_P01_origin': True,
        'phase_or_event_time_warping': False, 'validation': validation,
        'command': command, 'previews': base.previews(output, count, base.find_ffmpeg())}


def build_pair(b_receipt, c_receipt, b_quality, c_quality, destination):
    destination = _new_directory(destination)
    b_control, c_policy = checked_review(b_receipt), checked_review(c_receipt)
    common = strict_control_pair(b_control, c_policy)
    quality = _quality_pair(b_quality, c_quality, b_control, c_policy)
    complete = (b_control['receipt']['physical_task_success'] is True
                and c_policy['receipt']['physical_task_success'] is True)
    full_name = 'nominal_vs_ppo_latest.mp4' if complete else 'nominal_vs_ppo_latest_incomplete.mp4'
    full = _comparison_video(b_control, c_policy, destination/full_name,
        left_end=b_control['receipt']['frame_count'], right_end=c_policy['receipt']['frame_count'],
        kind='full_episode_same_elapsed_P01')
    b_fr = b_control['receipt']['physical_events'].get('placed', {}).get('FR')
    c_fr = c_policy['receipt']['physical_events'].get('placed', {}).get('FR')
    b_end, c_end = _event_frame(b_control['rows'], b_fr)+1, _event_frame(c_policy['rows'], c_fr)+1
    front = _comparison_video(b_control, c_policy,
        destination/'ppo_P01_P02_stability_vs_nominal.mp4',
        left_end=b_end, right_end=c_end, kind='P01_to_real_FR_placement')
    result = {'schema': 'wlr50_clean.fl_capture_quality_pair.v1',
        'B_control_receipt': str(b_control['receipt_path']),
        'C_deterministic_receipt': str(c_policy['receipt_path']),
        'strict_same_version_contract': common, 'quality': quality,
        'full_episode': full, 'front_event_window': front,
        'N_ref_not_used_for_same_controller_claim': True,
        'task_success_not_inferred_from_quality': True}
    base.write_new_json(destination/'pair_receipt.json', result)
    return result


def build_fl_clip(c_receipt, destination, *, stochastic=False):
    destination = _new_directory(destination)
    item = checked_review(c_receipt)
    expected_mode = ('training_style_conditional_gaussian' if stochastic
                     else 'deterministic_conditional_mean')
    require(item['receipt']['role'] == 'C'
            and item['receipt']['mode'] == expected_mode,
            'P05 clip must match its explicit deterministic/stochastic command')
    rows = item['rows']
    start, end, genuine, placed_tick = fl_clip_interval(
        rows, item['receipt']['physical_events'])
    count = end-start
    require(count >= 2 and count/FPS <= 200, 'FL event clip interval is invalid')
    stem = ('ppo_P05_capture_stochastic_after_training' if stochastic
            else 'ppo_P05_capture_after_training')
    name = stem + ('.mp4' if genuine else '_incomplete.mp4')
    output = destination/name
    command = [str(base.find_ffmpeg()), '-hide_banner', '-nostdin', '-v', 'error', '-n',
        '-threads', '2', '-filter_threads', '1', '-filter_complex_threads', '1',
        '-i', item['receipt']['output'], '-vf',
        f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS', '-an',
        '-frames:v', str(count), '-r', '15', '-fps_mode', 'cfr', '-c:v', 'libx264',
        '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '2',
        '-movflags', '+faststart', str(output)]
    base.run(command)
    validation = review_video.review.validate_video(output, count, 1280, 870)
    result = {'schema': 'wlr50_clean.fl_capture_event_clip.v1', 'output': str(output),
        'source_review_receipt': str(item['receipt_path']),
        'policy_sampling_mode': expected_mode,
        'stochastic_is_not_deterministic_task_success': stochastic,
        'physical_result': item['receipt']['physical_result'],
        'genuine_FL_placed_event_tick': placed_tick,
        'P06_observed_after_placement': genuine,
        'success_semantic_name_used': genuine,
        'source_frame_interval_half_open': [start, end],
        'actual_tick_endpoints': [rows[start]['actual_physics_tick'], rows[end-1]['actual_physics_tick']],
        'contiguous_same_episode_window': True, 'internal_stalls_removed': False,
        'full_episode_preserved_at': item['receipt']['output'],
        'normal_speed': True, 'speed_modified': False, 'stitched': False,
        'validation': validation, 'command': command,
        'previews': base.previews(output, count, base.find_ffmpeg())}
    base.write_new_json(destination/'FL_event_clip_receipt.json', result)
    return result


def verify_modes(deterministic_receipt, stochastic_receipt, output):
    deterministic = checked_review(deterministic_receipt)
    stochastic = checked_review(stochastic_receipt)
    result = {'schema': 'wlr50_clean.fl_capture_quality_policy_mode_pair.v1',
        **strict_policy_mode_pair(deterministic, stochastic),
        'deterministic_receipt': str(deterministic['receipt_path']),
        'stochastic_receipt': str(stochastic['receipt_path'])}
    output = Path(output).resolve()
    require(output.is_relative_to(OUT.resolve()), 'Mode receipt must stay under output root')
    base.write_new_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    pair = sub.add_parser('pair')
    pair.add_argument('--b-receipt', type=Path, required=True)
    pair.add_argument('--c-receipt', type=Path, required=True)
    pair.add_argument('--b-quality', type=Path, required=True)
    pair.add_argument('--c-quality', type=Path, required=True)
    pair.add_argument('--destination', type=Path, required=True)
    for command in ('fl-clip', 'stochastic-fl-clip'):
        clip = sub.add_parser(command)
        clip.add_argument('--c-receipt', type=Path, required=True)
        clip.add_argument('--destination', type=Path, required=True)
    modes = sub.add_parser('verify-policy-modes')
    modes.add_argument('--deterministic-receipt', type=Path, required=True)
    modes.add_argument('--stochastic-receipt', type=Path, required=True)
    modes.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'pair':
        result = build_pair(args.b_receipt, args.c_receipt, args.b_quality,
                            args.c_quality, args.destination)
    elif args.command in ('fl-clip', 'stochastic-fl-clip'):
        result = build_fl_clip(args.c_receipt, args.destination,
                              stochastic=args.command == 'stochastic-fl-clip')
    else:
        result = verify_modes(args.deterministic_receipt, args.stochastic_receipt,
                              args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

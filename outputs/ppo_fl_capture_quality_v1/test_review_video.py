from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


review = load('tested_fl_review', HERE/'review_video.py')
media = load('tested_fl_paired_media', HERE/'paired_event_media.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_checkpoint_identity_rechecks_both_persisted_files(tmp_path):
    checkpoint = tmp_path/'checkpoint.pt'
    checkpoint.write_bytes(b'checkpoint')
    actor = 'a'*64
    manifest = tmp_path/'checkpoint_manifest.json'
    manifest.write_text(json.dumps({
        'checkpoint_path': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
        'global_policy_decisions': 180480, 'ppo_updates': 1360, 'optimizer_steps': 27200,
        'actor_parameter_sha256': actor,
        'fl_capture_quality_branch': {'branch_id': 'fl_capture_quality_v1',
            'counter_origin': {'global_policy_decisions': 178432,
                               'ppo_updates': 1359, 'optimizer_steps': 27180}},
        'fl_capture_quality_branch_counts': {'global_policy_decisions': 2048,
            'ppo_updates': 1, 'optimizer_steps': 20},
    }), encoding='utf-8')
    proof = {'checkpoint_loaded_and_verified': True,
        'saved_global_policy_decisions': 180480,
        'parameter_hashes': {'actor_parameter_sha256': actor},
        'source': {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
                   'manifest': str(manifest), 'manifest_sha256': sha(manifest)}}
    result = review.checkpoint_identity(proof)
    assert result['saved_global_policy_decisions'] == 180480
    checkpoint.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='binding changed'):
        review.checkpoint_identity(proof)


def test_sealed_source_does_not_depend_on_python_asserts(tmp_path):
    run = tmp_path/'run'
    source = run/'source'
    source.mkdir(parents=True)
    contract = {'source_git_commit': '0'*40, 'selected_configuration': {},
                'runtime_content_sha256': '1'*64}
    (run/'run_manifest.json').write_text(json.dumps({
        'completed_at_utc': '2026-09-17T00:00:00Z', 'lifecycle': 'FAILED',
        'runtime_contract': contract}), encoding='utf-8')
    manifest = {'experiment_id': 'fl_capture_quality_v1', 'from_phase': 'P01',
        'episode_count': 1, 'fresh_process_single_episode': True,
        'optimizer_updates': 0, 'diagnostic_intervention': None,
        'runtime_contract': contract, 'role': 'B'}
    (source/'semantic_video_source_manifest.json').write_text(
        json.dumps(manifest), encoding='utf-8')
    assert review.sealed_source(source)['mode'] == 'N_plus_zero'
    manifest['from_phase'] = 'P05'
    (source/'semantic_video_source_manifest.json').write_text(
        json.dumps(manifest), encoding='utf-8')
    with pytest.raises(RuntimeError, match='natural P01'):
        review.sealed_source(source)


def capture_manifest():
    return {'experiment_id': 'fl_capture_quality_v1', 'seed': 4001,
        'camera': {'eye': [1, 2, 3]}, 'natural_reset_proof': {'entry': 'P01'},
        'evaluation_configuration': {'task': {'sha256': 'a'*64}},
        'runtime_contract': {'source_git_commit': 'b'*40,
            'runtime_content_sha256': 'c'*64,
            'selected_configuration': {'task': {'sha256': 'a'*64}}}}


def reviewed(role, mode, checkpoint=None):
    return {'receipt': {'role': role, 'mode': mode,
                        'checkpoint_identity': checkpoint},
            'manifest': capture_manifest()}


def test_pair_requires_exact_same_runtime_and_evaluation_configuration():
    left = reviewed('B', 'N_plus_zero')
    right = reviewed('C', 'deterministic_conditional_mean', {'checkpoint': 'x'})
    result = media.strict_control_pair(left, right)
    assert result['same_source_git_commit'] == 'b'*40
    changed = copy.deepcopy(right)
    changed['manifest']['runtime_contract']['source_git_commit'] = 'd'*40
    with pytest.raises(RuntimeError, match='not byte-identical'):
        media.strict_control_pair(left, changed)
    changed = copy.deepcopy(right)
    changed['manifest']['evaluation_configuration']['task']['sha256'] = 'e'*64
    with pytest.raises(RuntimeError, match='configuration hashes differ'):
        media.strict_control_pair(left, changed)


def test_stochastic_evidence_must_use_the_exact_deterministic_checkpoint():
    identity = {'checkpoint': 'x', 'checkpoint_sha256': 'a'*64}
    deterministic = reviewed('C', 'deterministic_conditional_mean', identity)
    stochastic = reviewed('C', 'training_style_conditional_gaussian', identity)
    stochastic['manifest']['checkpoint_load_provenance'] = {'policy_seed': 1234}
    result = media.strict_policy_mode_pair(deterministic, stochastic)
    assert result['same_checkpoint'] == identity
    assert result['stochastic_is_not_used_for_deterministic_quality_claim'] is True
    stochastic['receipt']['checkpoint_identity'] = {'checkpoint': 'other'}
    with pytest.raises(RuntimeError, match='same checkpoint'):
        media.strict_policy_mode_pair(deterministic, stochastic)


def test_fl_clip_preserves_all_stalls_and_requires_placed_then_p06():
    rows = [
        {'phase': 'P03', 'actual_physics_tick': 8},
        {'phase': 'P04', 'actual_physics_tick': 16},
        {'phase': 'P05', 'actual_physics_tick': 24},
        {'phase': 'P05', 'actual_physics_tick': 32},
        {'phase': 'P06', 'actual_physics_tick': 40},
        {'phase': 'P06', 'actual_physics_tick': 48},
        {'phase': 'P07', 'actual_physics_tick': 56},
    ]
    assert media.fl_clip_interval(rows, {'placed': {'FL': 32}}) == (1, 6, True, 32)
    # A 3 mm AIR near-miss has no placed event: keep through the real endpoint.
    assert media.fl_clip_interval(rows[:4], {'placed': {}}) == (1, 4, False, None)
    # A placed label without physically observed P06 is not success-qualified.
    assert media.fl_clip_interval(rows[:4], {'placed': {'FL': 32}}) == (1, 4, False, 32)


def test_checked_review_rehashes_frame_evidence_and_all_used_source_artifacts(
        tmp_path, monkeypatch):
    source = tmp_path/'run'/'source'
    source.mkdir(parents=True)
    manifest_path = source/'semantic_video_source_manifest.json'
    manifest_path.write_text('{}', encoding='utf-8')
    bindings = {}
    for name in review.USED_SOURCE_ARTIFACTS:
        path = source/name
        path.write_bytes(('sealed-'+name).encode())
        bindings[name] = sha(path)
    output = tmp_path/'review'/'full.mp4'
    output.parent.mkdir()
    output.write_bytes(b'review video')
    evidence = output.parent/'frame_evidence.json'
    evidence.write_text(json.dumps({'schema': 'wlr50_clean.fl_capture_quality_frame_evidence.v1',
        'source_manifest_sha256': sha(manifest_path), 'no_forward_fill': True,
        'frames': [
            {'source_frame_index': 0, 'actual_physics_tick': 8},
            {'source_frame_index': 1, 'actual_physics_tick': 16},
        ]}), encoding='utf-8')
    identity = {'checkpoint': 'cp'}
    receipt = {'schema': 'wlr50_clean.fl_capture_quality_review.v1',
        'output': str(output), 'source_manifest': str(manifest_path),
        'source_manifest_sha256': sha(manifest_path),
        'validation': {'sha256': sha(output), 'valid': True},
        'normal_speed': True, 'speed_modified': False, 'stitched': False,
        'single_episode': True, 'full_episode_preserved': True,
        'role': 'C', 'mode': 'deterministic_conditional_mean',
        'checkpoint_identity': identity, 'frame_count': 2,
        'frame_evidence': str(evidence), 'frame_evidence_sha256': sha(evidence),
        'used_source_artifact_sha256': bindings}
    receipt_path = output.with_suffix('.media.json')
    receipt_path.write_text(json.dumps(receipt), encoding='utf-8')
    context = {'source': source, 'role': 'C', 'mode': receipt['mode'],
        'checkpoint': identity, 'manifest': {'experiment_id': 'fl_capture_quality_v1'}}
    monkeypatch.setattr(media.review_video, 'sealed_source', lambda _: context)
    assert len(media.checked_review(receipt_path)['rows']) == 2
    (source/'physical_observations.jsonl').write_bytes(b'tampered')
    with pytest.raises(RuntimeError, match='changed after review'):
        media.checked_review(receipt_path)


def _test_video(path, frames):
    ffmpeg = str(review.base.find_ffmpeg())
    result = subprocess.run([ffmpeg, '-hide_banner', '-nostdin', '-v', 'error', '-n',
        '-f', 'lavfi', '-i', 'testsrc=size=1280x870:rate=15', '-frames:v', str(frames),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-threads', '2',
        '-movflags', '+faststart', str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_side_by_side_padding_is_bounded_and_explicit(tmp_path):
    left, right, output = tmp_path/'left.mp4', tmp_path/'right.mp4', tmp_path/'pair.mp4'
    _test_video(left, 6)
    _test_video(right, 4)
    result = media._comparison_video(
        {'receipt': {'output': str(left), 'frame_count': 6}},
        {'receipt': {'output': str(right), 'frame_count': 4}},
        output, left_end=6, right_end=4, kind='fixture')
    assert result['frame_count'] == 6
    assert result['freeze_added_frames'] == [0, 2]
    assert result['validation']['valid'] is True
    assert result['speed_modified'] is False
    assert result['phase_or_event_time_warping'] is False

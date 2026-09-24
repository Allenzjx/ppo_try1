"""Synthetic read-only writer integration; no Isaac or actual stage claims."""
import copy
import json
import math

import pytest

from wlr50_clean.ppo import semantic_height_diagnostics as diag
from wlr50_clean.ppo.semantic_hip_mount_geometry import resolve_hip_mounts
from test_semantic_hip_mount_geometry import fixture
from test_semantic_height_diagnostics import setup_live


def resolved_mounts():
    provider, usd, physics, _, _ = fixture()
    return resolve_hip_mounts(provider, usd=usd, usd_physics=physics)


def measured_frame(frame, tick, *, q=(1., 0., 0., 0.), p=(0., 0., 2.)):
    value = frame(tick)
    raw = value.info['raw_observation']
    raw.update(simulation_time_s=tick/120., bodies={'base_link': {
        'name': 'base_link', 'position_w_m': p, 'orientation_wxyz': q}})
    return value


def test_four_mount_startup_and_each_sample_use_observed_parent(setup_live, monkeypatch, tmp_path):
    recorder, frame, original, calls, robot = setup_live
    resolved = resolved_mounts()
    monkeypatch.setattr(diag, 'resolve_hip_mounts', lambda provider: copy.deepcopy(resolved))
    recorder.start(measured_frame(frame, 0))
    angle = .3
    current = measured_frame(frame, 8, q=(math.cos(angle/2), math.sin(angle/2), 0., 0.))
    # Deliberately unlike the already sampled raw pose: the new metric must
    # not combine current getter values with an earlier observation clock.
    robot.data.body_link_pos_w[0, 0] = [10., 20., 30.]
    recorder.sample(current)
    receipt = recorder.close()
    startup = json.loads((tmp_path/'height_diagnostics_startup.json').read_text())
    assert startup['same_rigid_body_hip_mount_definitions'] == resolved
    rows = [json.loads(x) for x in (tmp_path/'height_diagnostics.jsonl').read_text().splitlines()]
    assert [x['physics_tick'] for x in rows] == [0, 8]
    first, second = [x['same_rigid_body_hip_mount_geometry']['value'] for x in rows]
    assert first['FR_world_z_m'] == 2.
    assert second['left_minus_right_mean_z_m'] == pytest.approx(2*math.sin(angle))
    assert second['parent_pose']['position_w_m'] == [0., 0., 2.]
    assert second['physics_tick'] == second['parent_pose']['observation_tick'] == 8
    assert second['source_verified'] and second['physics_writes'] == 0
    assert not second['reward_or_target_or_permission']
    assert all(x['clock_unchanged'] for x in rows)
    assert original._body_local_points == {}
    assert receipt['stream_closed'] and not receipt['errors']


def test_failed_actual_resolution_writes_NA_not_URDF_or_zero(setup_live, monkeypatch, tmp_path):
    recorder, frame, _, _, _ = setup_live
    failed = resolved_mounts()
    failed.update(source_verified=False, mounts=None, reason='synthetic wrong parent')
    monkeypatch.setattr(diag, 'resolve_hip_mounts', lambda provider: failed)
    recorder.start(measured_frame(frame, 0))
    receipt = recorder.close()
    row = json.loads((tmp_path/'height_diagnostics.jsonl').read_text())
    value = row['same_rigid_body_hip_mount_geometry']['value']
    assert value['valid'] is False and value['source_verified'] is False
    assert value['mount_world_m'] is None and value['left_minus_right_mean_z_m'] is None
    assert value['FR_world_z_m'] is None and value['reason']
    assert receipt['stream_closed'] and not receipt['errors']


def test_missing_observed_pose_does_not_suppress_other_diagnostics(setup_live, monkeypatch, tmp_path):
    recorder, frame, _, _, _ = setup_live
    monkeypatch.setattr(diag, 'resolve_hip_mounts', lambda provider: resolved_mounts())
    recorder.start(frame(0))
    receipt = recorder.close()
    row = json.loads((tmp_path/'height_diagnostics.jsonl').read_text())
    value = row['same_rigid_body_hip_mount_geometry']['value']
    assert value['valid'] is False and value['mount_world_m'] is None
    assert row['rr_hip_mount_w_m']['value'] == pytest.approx([.9, 1.95, .32])
    assert row['joint_position_native_rad']['value'] == [.5]
    assert receipt['rows'] == 1 and receipt['stream_closed']

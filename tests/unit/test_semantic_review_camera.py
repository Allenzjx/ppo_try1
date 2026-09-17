"""Viewport-only opt-in wiring; not a rendered visibility certificate."""
from types import SimpleNamespace as NS
import pytest

from wlr50_clean.ppo import semantic_video as video
from wlr50_clean.ppo.isaac_fsm_backend import IsaacFSMBackend, IsaacFSMBackendError


@pytest.mark.parametrize('experiment', [None, 'fsm_reference_p09_stable_v2', 'task_first_recovery_v1'])
def test_historical_camera_bytes_remain_selected(experiment):
    assert video.camera_for_experiment(experiment) == {
        'eye_m': [1.45, -1.25, .8], 'target_m': [.45, 0., .12]}


def test_review_B_C_share_fixed_camera_without_mutating_global():
    assert video.camera_for_experiment('non_residual_refine_v1') == video.camera_for_experiment('residual_rr_fix_v1')
    selected = video.camera_for_experiment('non_residual_refine_v1')
    selected['eye_m'][0] = 999
    assert video.REVIEW_CAMERA['eye_m'][0] == 1.85
    assert video.CAMERA['eye_m'][0] == 1.45


def test_fresh_camera_configuration_and_application_are_viewport_only():
    backend = IsaacFSMBackend()
    requested = video.camera_for_experiment('non_residual_refine_v1')
    backend.configure_video_camera(**requested)
    requested['eye_m'][0] = 999
    calls = []
    backend._scene = NS(sim=NS(set_camera_view=lambda **kw: calls.append(kw)))
    physics = {'root': [1, 2, 3], 'obstacle': {'z': .08}, 'dt': 1/120}
    source = {**physics, 'camera': video.CAMERA}
    result = backend._apply_video_camera_override(source)
    assert calls == [{'eye': [1.85, -1.65, 1.15], 'target': [.70, -.15, .15]}]
    assert result['camera'] == video.REVIEW_CAMERA
    assert source['camera'] == video.CAMERA
    assert {k: result[k] for k in physics} == physics
    assert backend._episode_tick == backend._reset_count == 0
    assert backend._controller is backend._adapter is backend._reader is None
    assert backend._last_atomic_ack is None


def test_non_video_backend_does_not_apply_camera_or_require_scene():
    backend = IsaacFSMBackend()
    source = {'camera': video.CAMERA, 'physics': 'unchanged'}
    assert backend._apply_video_camera_override(source) == source


@pytest.mark.parametrize('eye,target', [([1,2], [0,0,0]), ([float('nan'),0,1],[0,0,0]),
    ([1,2,3],[1,2,3]), ([True,0,1],[0,0,0]), ([1,2,3],[float('inf'),0,0])])
def test_invalid_camera_fails_before_scene(eye,target):
    backend = IsaacFSMBackend()
    with pytest.raises(IsaacFSMBackendError):
        backend.configure_video_camera(eye_m=eye,target_m=target)
    assert backend._video_camera_override is None


@pytest.mark.parametrize('state', ['scene', 'reset'])
def test_active_or_used_backend_cannot_change_view(state):
    backend = IsaacFSMBackend()
    if state == 'scene': backend._scene = object()
    else: backend._reset_count = 1
    with pytest.raises(IsaacFSMBackendError, match='first reset'):
        backend.configure_video_camera(**video.REVIEW_CAMERA)

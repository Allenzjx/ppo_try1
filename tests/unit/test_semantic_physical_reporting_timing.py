"""Reporting-only wiring and observed-window tests; no simulator or model load."""
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_semantic_legacy_evaluation import audited_frame, PROJECTION
from wlr50_clean.ppo.semantic_legacy_evaluation import PhysicalEvaluationRecorder

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / 'configs/ppo_all_stage_acceptance_v1/stage_task_spec.yaml'


@pytest.mark.parametrize('enabled', [False, True])
def test_recorder_explicit_finish_optin_uses_real_evaluator(tmp_path, enabled):
    recorder = PhysicalEvaluationRecorder(tmp_path, task_spec_path=SPEC,
                                         finish_recovery_enabled=enabled)
    try:
        assert recorder.evaluator._finish_recovery_enabled is enabled
        recorder.start(audited_frame(0))
        recorder.observe(audited_frame(0), audited_frame(1), PROJECTION)
        result = recorder.summary()
        assert result['finish_recovery_enabled'] is enabled
        assert result['evaluation_timing_schema'] == 'wlr50_clean.physical_evaluation_timing.v2'
        assert result['evaluation_timing_version'] == (
            'finish_settle_pending_speed_only_v1' if enabled else 'original_task_spec_timing')
        assert ('finish_settle_pending' in result['physical_task_evaluation']) is enabled
    finally:
        recorder.close()


def test_recorder_legacy_default_and_invalid_flag(tmp_path):
    recorder = PhysicalEvaluationRecorder(tmp_path)
    try:
        assert recorder.finish_recovery_enabled is False
        assert recorder.evaluator._finish_recovery_enabled is False
    finally:
        recorder.close()
    with pytest.raises(ValueError):
        PhysicalEvaluationRecorder(tmp_path, task_spec_path=SPEC, finish_recovery_enabled='true')


@pytest.mark.parametrize('final_success', [False, True])
def test_transient_success_never_freezes_later_physical_metrics(tmp_path, final_success):
    recorder = PhysicalEvaluationRecorder(tmp_path)
    class ObservedSequence:
        def __init__(self):
            self.sequence = iter([False, True, False, final_success])
        def observe(self, raw):
            self.snapshot = dict(success=next(self.sequence), history={}, goal_features={})
            return self.snapshot
    recorder.evaluator = ObservedSequence()
    try:
        recorder.start(audited_frame(0))
        for tick in range(1, 4):
            recorder.observe(audited_frame(tick-1), audited_frame(tick), PROJECTION)
        result = recorder.summary()
        assert result['task_success'] is final_success
        assert result['first_observed_success_time_s'] == pytest.approx(1/120)
        assert result['physical_task_duration_s'] == pytest.approx(3/120)
        assert result['observed_physics_ticks'] == 3
        assert result['quality_metrics']['global']['physics_ticks'] == 3
        assert result['quality_metrics']['global']['duration_s'] == pytest.approx(3/120)
        assert result['metric_coverage']['version'] == 'all_observed_physical_intervals_v2'
        assert not result['metric_coverage']['video_only_padding_included']
    finally:
        recorder.close()
    with (tmp_path/'physics_quality_metrics.csv').open(newline='') as stream:
        assert len(list(csv.DictReader(stream))) == 3


@pytest.mark.parametrize('routed,finish', [(False, None), (True, None),
                                         (True, {'enabled': False}), (True, {'enabled': True})])
def test_actual_evaluate_passes_selected_route_setting_without_render_or_torch(tmp_path, monkeypatch,
                                                                            routed, finish):
    from wlr50_clean.ppo import semantic_rr_capture_local as route_module
    from wlr50_clean.ppo import semantic_legacy_evaluation as report_module
    from wlr50_clean.ppo import semantic_rr_mean_coordinates as coordinates
    from wlr50_clean.infrastructure import video_capture
    # evaluate imports Torch, but this test stops at report construction before
    # any inference/model/RNG access. No Isaac imports or physical steps occur.
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    monkeypatch.setattr(video_capture, 'ActiveViewportVideoRecorder', lambda _: object())
    monkeypatch.setattr(coordinates, 'assert_coordinate_binding', lambda *args: None)
    monkeypatch.setattr(route_module, 'auxiliary_events', lambda *args: [])
    monkeypatch.setattr(route_module, 'settings', lambda: {})
    cfg = {} if finish is None else {'finish_recovery': finish}
    if not routed:
        monkeypatch.setattr(route_module, 'settings', lambda: cfg)
    received = {}
    class StopAtReport(Exception):
        pass
    def recorder(path, **kwargs):
        received.update(kwargs)
        raise StopAtReport
    monkeypatch.setattr(report_module, 'PhysicalEvaluationRecorder', recorder)
    core = SimpleNamespace(backend=SimpleNamespace(configure_video_camera=lambda **kwargs: None),
                           reset=lambda **kwargs: None)
    route = (dict(validate_runner=lambda *args: None, auxiliary_ledger=[], settings=cfg,
                  config_dir=ROOT/'configs/ppo_rr_capture_first_cp225280_v1',
                  tensor_observation=None, request=None) if routed else None)
    with pytest.raises(StopAtReport):
        route_module.evaluate(core, object(), {}, {}, {}, tmp_path, route=route)
    assert received['finish_recovery_enabled'] is bool(finish and finish['enabled'])
    assert received['task_spec_path'].name == 'stage_task_spec.yaml'

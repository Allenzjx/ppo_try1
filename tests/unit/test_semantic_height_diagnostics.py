"""Synthetic CPU diagnostics checks; no contact/physical success claims."""
import json
import math
from types import SimpleNamespace as NS

import numpy as np
import pytest

from wlr50_clean.ppo import semantic_height_diagnostics as diag
from wlr50_clean.ppo import semantic_video as video


def test_nonfinite_unknown_are_not_zero():
    result = diag.read_value(lambda: np.array([[float("nan"), float("inf"), 0.]]), "fixture")
    assert result["value"] == [[None, None, 0.]]
    assert len(result["nonfinite_entries"]) == 2
    assert diag.read_value(lambda: (_ for _ in ()).throw(AttributeError("unsupported")), "fixture")["value"] is None
    assert diag.read_value(lambda: None, "fixture")["reason"]


def test_lowest_is_actual_rotated_point_not_center_xy():
    points = diag._body_local_point_array([[-2., 0., 0.], [1., 0., -1.]])
    quat = [math.cos(math.pi / 4), 0., math.sin(math.pi / 4), 0.]
    value = diag.lowest_collider_point(points, [10., 20., 3.], quat)
    assert value["body_local_point_m"] == [1., 0., -1.]
    assert value["world_point_m"] == pytest.approx([9., 20., 2.])
    assert value["world_point_m"][0] != 10.


@pytest.fixture
def setup_live(monkeypatch, tmp_path):
    bodies = [diag.BASE_BODY, *diag.WHEEL_BODIES]
    positions = np.array([[[1., 2., .3] for _ in bodies]])
    quats = np.array([[[1., 0., 0., 0.] for _ in bodies]])
    calls = []
    class Provider:
        def __init__(self, stage, *, robot_prim_path):
            self.stage, self.robot_prim_path = stage, robot_prim_path
            self._body_local_points, self._collider_paths = {}, {}
        def collision_bounds(self, body, **kwargs):
            calls.append(body)
            self._body_local_points[body] = ((-.1, 0., -.02), (.2, .01, .03))
            self._collider_paths[body] = ("/robot/" + body + "/collision",)
    original = Provider(object(), robot_prim_path="/robot")
    monkeypatch.setattr(diag, "UsdCollisionBoundsProvider", Provider)
    monkeypatch.setattr(diag, "resolve_rr_mount", lambda _: {
        "parent_body_name": diag.BASE_BODY, "local_pos0_m": [-.1, -.05, .02]})
    getters = {method: (lambda method=method: (calls.append(method) or np.array([[1., 2.]])))
               for method in diag.GETTERS.values()}
    actuator = NS(joint_names=["rear_right_knee"], is_implicit_model=True,
                  computed_effort=np.array([[9.]]), applied_effort=np.array([[5.]]))
    robot = NS(body_names=bodies, joint_names=["rear_right_knee"], root_physx_view=NS(**getters),
        data=NS(body_link_pos_w=positions, body_link_quat_w=quats,
                joint_pos=np.array([[.5]]), joint_vel=np.array([[-.2]])), actuators={"leg": actuator})
    backend = NS(_adapter=NS(robot=robot), _reader=NS(geometry_backend=NS(provider=original)),
                 _episode_tick=0, _controller=NS(physics_tick=1), _reset_count=1)
    def frame(tick):
        backend._episode_tick, backend._controller.physics_tick = tick, tick + 1
        return NS(physics_tick=tick, state_id="P09", info={"raw_observation": {
            "physics_tick": tick, "body_bounds_w_m": {"base_link": {"minimum_m": [9., 9., 9.]}},
            "geometry_pose_aware": False, "joints": {}, "base": {}, "center_of_mass": {}}})
    return diag.HeightDiagnostics(tmp_path, backend), frame, original, calls, robot


def test_tick0_grid_exact_terminal_independent_bounds_and_one_startup(setup_live, tmp_path):
    recorder, frame, original, calls, robot = setup_live
    recorder.start(frame(0))
    for tick in range(1, 12):
        recorder.sample(frame(tick), terminal=tick == 11)
    receipt = recorder.close(frame(11))
    rows = [json.loads(x) for x in (tmp_path / "height_diagnostics.jsonl").read_text().splitlines()]
    assert [r["physics_tick"] for r in rows] == [0, 8, 11]
    assert rows[-1]["terminal_sample"] is True
    assert rows[0]["body_collision_minimum_z_w_m"] == pytest.approx(.28)
    assert rows[0]["rr_hip_mount_w_m"]["value"] == pytest.approx([.9, 1.95, .32])
    assert rows[0]["recorded_evaluator_body_bounds"]["value"]["base_link"]["minimum_m"] == [9., 9., 9.]
    assert all(row["clock_unchanged"] for row in rows)
    assert original._body_local_points == {}  # No shared geometry cache changes.
    assert calls.count("get_dof_stiffnesses") == 1
    assert calls.count("base_link") == 1  # Immutable local asset cached, not a world extent.
    assert receipt["rows"] == 3 and receipt["stream_closed"] and not receipt["errors"]
    startup = json.loads((tmp_path / "height_diagnostics_startup.json").read_text())
    assert startup["clock_before"] == startup["clock_after"]
    assert startup["clock_before"]["controller_next_tick"] == 1
    effort = rows[-1]["actuator_effort_estimates"]["value"]["leg"]
    assert effort["computed_effort_Nm"]["source"] == "implicit_PD_estimate_before_clip"
    assert effort["measured_physx_drive_torque_Nm"] is None


def test_rotation_does_not_reuse_world_extent(setup_live, tmp_path):
    recorder, frame, _, _, robot = setup_live
    recorder.start(frame(0))
    robot.data.body_link_quat_w[0, 0] = [math.cos(math.pi / 4), 0., math.sin(math.pi / 4), 0.]
    recorder.sample(frame(8))
    recorder.close()
    rows = [json.loads(x) for x in (tmp_path / "height_diagnostics.jsonl").read_text().splitlines()]
    assert rows[1]["body_collision_minimum_z_w_m"] == pytest.approx(.1)
    assert rows[1]["body_collision_minimum_z_w_m"] != rows[0]["body_collision_minimum_z_w_m"]


def test_missing_everything_still_closes_with_unknowns(tmp_path):
    recorder = diag.HeightDiagnostics(tmp_path, NS())
    frame = NS(physics_tick=0, state_id="P01", info={})
    recorder.start(frame)
    receipt = recorder.close(frame)
    row = json.loads((tmp_path / "height_diagnostics.jsonl").read_text())
    assert row["body_collision_minimum_z_w_m"] is None
    assert row["rr_hip_mount_w_m"]["value"] is None
    assert row["joint_position_native_rad"]["value"] is None
    assert receipt["stream_closed"] and receipt["errors"]


def test_controller_terminal_non_grid_retained_by_close(setup_live, tmp_path):
    recorder, frame, _, _, _ = setup_live
    recorder.start(frame(0))
    recorder.sample(frame(8))
    recorder.sample(frame(9))
    receipt = recorder.close(frame(9))
    rows = [json.loads(x) for x in (tmp_path / "height_diagnostics.jsonl").read_text().splitlines()]
    assert [r["physics_tick"] for r in rows] == [0, 8, 9]
    assert receipt["last_sample_tick"] == 9


def test_observer_diagnostic_before_terminal_render_failure(monkeypatch):
    events = []
    physical = NS(observe=lambda *_: events.append("metrics"),
                  evaluator=NS(snapshot={"success": False, "termination_reason": "SAFETY_ABORT"}))
    heights = NS(sample=lambda frame, **kw: events.append(("diagnostic", frame.physics_tick, kw)))
    monkeypatch.setattr(video, "capture_task_interval_frame", lambda *_: (_ for _ in ()).throw(RuntimeError("encoder")))
    observer = video.EndpointObserver(physical, None, None, task_window=True, height_diagnostics=heights)
    with pytest.raises(RuntimeError, match="encoder"):
        observer(NS(physics_tick=2), NS(physics_tick=3), None)
    assert events == ["metrics", ("diagnostic", 3, {"terminal": True})]


@pytest.mark.parametrize("options", [{"terminal_encoder_failure": True}, {"done_only": True}])
def test_capture_finally_closes_and_binds_diagnostic_on_failures(tmp_path, monkeypatch, options):
    from test_semantic_video_fsm_reference_v2 import _capture_tick23_fixture
    capture = _capture_tick23_fixture(tmp_path, monkeypatch, success=False, **options)
    receipt = capture.result["height_diagnostics"]
    assert receipt["stream_closed"] and receipt["last_sample_tick"] == 23
    assert "height_diagnostics.jsonl" in capture.result["artifacts"]
    assert "height_diagnostics_startup.json" in capture.result["artifacts"]
    rows = [json.loads(x) for x in (capture.source / "height_diagnostics.jsonl").read_text().splitlines()]
    assert [r["physics_tick"] for r in rows] == [0, 8, 16, 23]
    assert rows[-1]["terminal_sample"]

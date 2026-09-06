"""No-physics wiring checks for the shared B/C nominal geometry path."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.ppo.isaac_fsm_backend import IsaacFSMBackend
from wlr50_clean.ppo.semantic_backend import SemanticIsaacBackend, load_execution_profile
from wlr50_clean.ppo.semantic_nominal_geometry import MODE
from wlr50_clean.ppo.semantic_residual_adapter import SemanticActuationDispatch

ROOT = Path(__file__).resolve().parents[2]
ZERO = (0.,) * 12


@pytest.mark.parametrize("prefix_mode,handoff,enabled,expect_capture", [
    (None, None, True, True),
    ("READY", 9, True, True),
    ("READY", 10, True, False),
    ("TAKEOVER", 9, True, False),
    ("TEACHER", None, True, False),
    (None, None, False, False),
])
def test_geometry_context_is_not_injected_into_teacher_or_exact_handoff(
        monkeypatch, prefix_mode, handoff, enabled, expect_capture):
    backend = SemanticIsaacBackend.__new__(SemanticIsaacBackend)
    backend._semantic_actuation_plan = object()
    backend._nominal_geometry_mode = MODE if enabled else None
    backend._nominal_geometry_margin_m = .015
    backend._controller_frame = SimpleNamespace(physics_tick=10, state_id="P09")
    backend._raw_observation = object()
    controller = SimpleNamespace(task_snapshot={"fixture": "actual source task"})
    if prefix_mode is not None:
        controller.mode = prefix_mode
        controller._handoff_tick = handoff
    backend._controller = controller
    calls, writes = [], []
    context = {"fixture": "one same-state capture"}
    def capture(**kwargs):
        calls.append(kwargs)
        return context
    monkeypatch.setattr("wlr50_clean.ppo.semantic_nominal_geometry.capture_nominal_geometry_context", capture)
    def atomic(self, adapter, command, **kwargs):
        writes.append((adapter, command, kwargs))
        return {"single_underlying_dispatch": True}
    monkeypatch.setattr(IsaacFSMBackend, "_atomic_apply", atomic)
    original_adapter = object()
    result = backend._atomic_apply(original_adapter, ZERO, physics_tick=190,
        tracking_servo_names=(), drive_feedback_bias_full12=ZERO)
    assert result == {"single_underlying_dispatch": True}
    assert len(writes) == 1
    assert isinstance(writes[0][0], SemanticActuationDispatch)
    assert writes[0][0].adapter is original_adapter
    assert len(calls) == int(expect_capture)
    assert writes[0][0].nominal_geometry_context is (context if expect_capture else None)
    if expect_capture:
        assert calls[0] == {
            "adapter": original_adapter, "observation": backend._raw_observation,
            "source_frame": backend._controller_frame,
            "task_snapshot": controller.task_snapshot,
            "clearance_margin_m": .015, "physics_tick": 190,
        }


def test_reset_without_actuation_plan_does_not_capture_or_wrap(monkeypatch):
    backend = SemanticIsaacBackend.__new__(SemanticIsaacBackend)
    backend._semantic_actuation_plan = None
    backend._nominal_geometry_mode = MODE
    calls = []
    def atomic(self, adapter, command, **kwargs):
        calls.append(adapter)
        return {}
    monkeypatch.setattr(IsaacFSMBackend, "_atomic_apply", atomic)
    def unexpected(**kwargs):
        pytest.fail("reset must not request a task geometry context")
    monkeypatch.setattr("wlr50_clean.ppo.semantic_nominal_geometry.capture_nominal_geometry_context", unexpected)
    adapter = object()
    backend._atomic_apply(adapter, ZERO, physics_tick=0,
        tracking_servo_names=(), drive_feedback_bias_full12=ZERO)
    assert calls == [adapter]


def test_versioned_option_only_on_current_v3_profile():
    old = load_execution_profile(ROOT/"configs/ppo_semantic_v2/execution_profile.yaml")
    new = load_execution_profile(ROOT/"configs/ppo_semantic_v3/execution_profile.yaml")
    assert old.get("nominal_geometry_advisory") is None
    assert new["nominal_geometry_advisory"] == MODE
    assert new["residual"]["composition"] == "independent_post_mapper_residual.v1"


@pytest.mark.parametrize("mode,composition", [
    ("unknown_geometry_model", "independent_post_mapper_residual.v1"),
    (MODE, None),
])
def test_unknown_version_or_wrong_composition_rejected(tmp_path, mode, composition):
    source = ROOT/"configs/ppo_semantic_v2/execution_profile.yaml"
    profile = yaml.safe_load(source.read_text(encoding="utf-8"))
    profile["nominal_geometry_advisory"] = mode
    profile["residual"]["composition"] = composition
    path = tmp_path/"profile.yaml"
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    with pytest.raises(ValueError):
        load_execution_profile(path)

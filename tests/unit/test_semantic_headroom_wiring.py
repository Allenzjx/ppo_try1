"""Request projection and profile wiring; these are not physical task successes."""
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from wlr50_clean.ppo.action_projection import ActionProjector, ActionProjectionError, SafetyProjection
from wlr50_clean.ppo.semantic_backend import build_semantic_projector, load_execution_profile
from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE, project_semantic_servo_headroom
from wlr50_clean.ppo.phase_action_masks_v2 import PhaseTransitionBridge

ROOT = Path(__file__).resolve().parents[2]
V3 = ROOT / "configs/ppo_semantic_v3/execution_profile.yaml"
ZERO = (0.,) * 12


def context(phase="P09"):
    nominal = ZERO[:7] + (-37.8,) + ZERO[8:]
    return dict(state_id=phase, nominal_action_full12=nominal,
                reference_action_full12=nominal, reference_delta_full12=ZERO)


def test_actual_logical_clip_counterexample_defers_only_servo_margin():
    current = build_semantic_projector(V3)
    legacy = ActionProjector(replace(current.config, policy_headroom_mode=None))
    raw = ZERO[:7] + (-.855614,) + ZERO[8:]
    previous = ZERO[:7] + (-20.2,) + ZERO[8:]
    rows = [p.project(raw, previous_projected_residual_full12=previous, **context())
            for p in (legacy, current)]
    assert rows[0].safe_projected_residual_full12[7] == pytest.approx(-20.2)
    assert rows[1].safe_projected_residual_full12[7] == pytest.approx(-20.7)
    assert rows[1].applied_action_full12[7] == pytest.approx(-58.5)  # transport, not drive
    native = ZERO[:7] + (-27.8,) + ZERO[8:]
    evidence = project_semantic_servo_headroom(native, ZERO, rows[1].safe_projected_residual_full12)
    assert evidence["effective_policy_residual_full12"][7] == pytest.approx(-20.7)
    assert evidence["candidate_native_target_before_final_slew_full12"][7] == pytest.approx(-48.5)


@pytest.mark.parametrize("phase", [f"P{i:02}" for i in range(1,14)])
def test_caps_request_slew_masks_and_zero_remain(phase):
    p = build_semantic_projector(V3)
    row = p.project((1.e6,) * 12, **context(phase))
    assert row.safe_projected_residual_full12 == pytest.approx((.5,)*8 + (.015,)*4)
    assert p.project(ZERO, **context(phase)).zero_residual_fast_path
    row = p.project((1.,)*12, runtime_action_mask_full12=(0,)*12, **context(phase))
    assert row.safe_projected_residual_full12 == ZERO
    row = p.project((1.,)*12, safety=SafetyProjection(body_collision_detected=True), **context(phase))
    assert row.safe_projected_residual_full12 == ZERO


@pytest.mark.parametrize("pair", [("P07","P08"),("P08","P09"),("P09","P10"),
                                  ("P10","P11"),("P11","P12"),("P12","P13")])
def test_handoff_keeps_request_history_even_outside_logical_reserve(pair):
    bridge = PhaseTransitionBridge(build_semantic_projector(V3))
    previous = ZERO[:7] + (-24.,) + (.05,)*4
    bridge.reset(state_id=pair[0], projected_residual_full12=previous)
    out = bridge.project_tick(ZERO, **context(pair[1]))
    assert out.transition_metric.handoff_hold_used
    assert out.projection.safe_projected_residual_full12 == pytest.approx(previous)


def test_wheel_logical_bound_unchanged_and_nonfinite_rejected():
    p = build_semantic_projector(V3)
    kw = context()
    kw["nominal_action_full12"] = ZERO[:8] + (2.09,)*4
    row = p.project((1.,)*12, **kw)
    assert row.safe_projected_residual_full12[8:] == pytest.approx((2.0943951023931953-2.09,)*4)
    kw["nominal_action_full12"] = (float("nan"),) + ZERO[1:]
    with pytest.raises(ActionProjectionError):
        p.project(ZERO, **kw)


@pytest.mark.parametrize("mode,composition", [("unknown", "independent_post_mapper_residual.v1"),
                                                (HEADROOM_MODE, None)])
def test_profile_rejects_unimplemented_mode_and_wrong_composition(tmp_path, mode, composition):
    profile = yaml.safe_load(V3.read_text(encoding="utf-8"))
    profile["residual"].update(policy_headroom_mode=mode, composition=composition)
    path = tmp_path / "profile.yaml"
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    with pytest.raises(ValueError):
        load_execution_profile(path)


@pytest.mark.parametrize("index", range(8))
def test_constructor_cannot_silently_widen_reserve(index):
    config = build_semantic_projector(V3).config
    bounds = list(config.safety_limits_full12)
    lo, hi = bounds[index]
    bounds[index] = (lo-.01, hi)
    with pytest.raises(ActionProjectionError, match="2-degree"):
        ActionProjector(replace(config, safety_limits_full12=tuple(bounds)))


def test_old_profile_keeps_old_mode_and_vector_rejects_new_mode_before_scene():
    from wlr50_clean.ppo.semantic_vector_backend import SemanticVectorIsaacBackend
    assert load_execution_profile(ROOT/"configs/ppo_semantic_v2/execution_profile.yaml")["residual"].get("policy_headroom_mode") is None
    with pytest.raises(ValueError, match="audited N=1"):
        SemanticVectorIsaacBackend(None, execution_profile=V3)

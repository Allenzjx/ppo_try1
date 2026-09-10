"""Same-state nominal target tests; no claim of physical clearance guarantees."""
from copy import deepcopy
from pathlib import Path

import pytest

from test_semantic_nominal_geometry_context import fixture
from wlr50_clean.ppo.semantic_nominal_geometry import (
    CONTEXT_SCHEMA, EVIDENCE_SCHEMA, MODE, FUNCTIONAL_RR_MODE, NominalGeometryError,
    capture_nominal_geometry_context, correct_nominal_geometry,
)
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider, load_task_spec
from wlr50_clean.reference.motion_contract import load_motion_contract

ROOT = Path(__file__).resolve().parents[2]


def capture(f, *, gain=.010, distance=-.30, mode=FUNCTIONAL_RR_MODE):
    f.current.update(ground_relative_lift_m=gain, front_distance_m=distance)
    return capture_nominal_geometry_context(adapter=f.adapter, observation=f.observation,
        source_frame=f.source, task_snapshot=f.task, clearance_margin_m=.015,
        physics_tick=f.dispatch_tick, mode=mode, minimum_lift_gain_m=.008, workspace_min_m=-.22)


def correct(f, context):
    native = [0.] * 12
    native[f.indices[0]] = -20.
    native[8:] = [.1, -.2, .3, -.4]
    adjusted, evidence = correct_nominal_geometry(adapter=f.adapter, native_full12=native,
        controller_bias_full12=(0.,) * 12, context=context)
    return tuple(native), adjusted, evidence


def test_far_rr_ten_mm_ground_lift_allows_two_mm_nominal_descent_below_platform():
    f = fixture(clearance=-.04, canonical_zero=True)
    context = capture(f)
    assert context["clearance_margin_m"] == pytest.approx(-.042)
    assert context["clearance_floor_semantics"] == "measured_ground_reference_existing_lift_scale"
    _, _, evidence = correct(f, context)
    assert evidence["projection"]["available_descent_m"] == pytest.approx(.002)
    assert evidence["context"]["clearance_m"] < 0.
    assert not evidence["physical_motion_guaranteed"]


def test_near_rr_below_top_does_not_get_nominal_clock_forced_descent():
    f = fixture(clearance=-.04, canonical_zero=True)
    context = capture(f, distance=-.10)
    assert context["clearance_margin_m"] == pytest.approx(context["clearance_m"])
    assert context["clearance_floor_semantics"] == "current_below_top_or_surface_floor_before_cross"
    _, _, evidence = correct(f, context)
    assert evidence["projection"]["available_descent_m"] == 0.
    assert evidence["projection"]["mathematical_contract_verified"]
    assert evidence["projection"]["corrected_linear_z_m"] >= -1e-12


def test_edge_context_keeps_projection_constraints_and_never_consumes_residual():
    f = fixture(clearance=-.04, canonical_zero=True)
    f.current.update(air=False, obstacle_pair_active=True, contact_mode="FRONT_WALL")
    context = capture(f, distance=-.10)
    native, adjusted, evidence = correct(f, context)
    assert context["mode"] == FUNCTIONAL_RR_MODE
    assert evidence["schema"] == EVIDENCE_SCHEMA
    assert evidence["projection"]["mathematical_contract_verified"]
    assert not evidence["current_policy_residual_used"]
    assert adjusted[8:] == native[8:]
    assert all(adjusted[i] == native[i] for i in range(12) if i not in f.indices)
    with pytest.raises(TypeError):
        correct_nominal_geometry(adapter=f.adapter, native_full12=native,
            controller_bias_full12=(0.,)*12, context=context, current_policy_residual_full12=(.2,)*12)


def test_missing_measured_ground_reference_bypasses_without_inventing_floor_or_reading_articulation():
    f = fixture(clearance=-.04)
    f.adapter.robot = None
    assert capture(f, gain=None) is None
    assert not f.calls


def test_p12_retains_exact_original_platform_floor_and_targets_in_new_experiment_mode():
    old = fixture(phase="P12", clearance=.035, canonical_zero=True)
    new = fixture(phase="P12", clearance=.035, canonical_zero=True)
    old_context = capture(old, mode=MODE)
    new_context = capture(new)
    assert new_context["clearance_margin_m"] == old_context["clearance_margin_m"] == .015
    assert new_context["clearance_floor_semantics"] == "existing_above_top_margin"
    _, old_target, old_evidence = correct(old, old_context)
    _, new_target, new_evidence = correct(new, new_context)
    assert old_target == new_target
    assert old_evidence["projection"] == new_evidence["projection"]


def test_functional_mode_keeps_same_tick_geometry_schema_and_stale_clock_rejection():
    f = fixture(clearance=-.04)
    context = capture(f)
    assert context["schema"] == CONTEXT_SCHEMA
    assert context["source_control_tick"] == 240 and context["dispatch_physics_tick"] == 391
    assert context["source_sim_time_s"] == 2.
    f.task["physical_evaluator"]["physics_tick"] = 239
    with pytest.raises(NominalGeometryError, match="clock"):
        capture(f)


def test_functional_far_air_carry_below_top_has_no_timer_or_blind_edge_wheel_override():
    spec = load_task_spec(ROOT / "configs/ppo_fsm_reference_p09_stable_v2/stage_task_spec.yaml")
    contract = load_motion_contract(ROOT / "configs/recording_motion_contract.json")
    task = dict(stage_id="P09", termination_reason=None, physical_evaluator=dict(
        valid=True, termination_reason=None, history={"active_lift": {"RR": True}},
        current_legs={"RR": dict(current_lift_valid=True, motion_continuation_allowed=True,
            air=True, ground_contact=False, within_lateral_span=True,
            front_distance_m=-.30, clearance_m=-.04, air_duration_s=2/120.)}))
    short = NominalMotionProvider(contract, spec=spec)
    short_request, _ = short._continuous_advisory(task)
    assert short_request[8:] == (.3,)*4
    assert short.nominal_suggestion_diagnostics["rr_carry_continuation"]["added_rolling_suggestion"]
    long_task = deepcopy(task)
    long_task["physical_evaluator"]["current_legs"]["RR"]["air_duration_s"] = 1.5
    long = NominalMotionProvider(contract, spec=spec)
    assert long._continuous_advisory(long_task)[0] == short_request
    edge_task = deepcopy(task)
    edge_task["physical_evaluator"]["current_legs"]["RR"].update(
        air=False, contact_mode="FRONT_WALL", obstacle_pair_active=True)
    edge = NominalMotionProvider(contract, spec=spec)
    edge_request, _ = edge._continuous_advisory(edge_task)
    diagnostic = edge.nominal_suggestion_diagnostics["rr_carry_continuation"]
    assert not diagnostic["added_rolling_suggestion"]
    assert diagnostic["source_joint_owners_continued"] and not diagnostic["fixed_lift_timer_gate"]
    assert not diagnostic["above_top_15mm_action_gate"]
    assert edge_request[:8] == short_request[:8]

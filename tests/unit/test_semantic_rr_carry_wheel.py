"""Pure controller/source-provenance checks; these are not physics success."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.fsm.motion_executor import MotionExecutor
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER, WHEEL_VELOCITY_LIMIT_RAD_S
from wlr50_clean.ppo.semantic_rr_carry_wheel import (
    ACK_KEY, MODE, build_rr_carry_wheel_context, project_rr_carry_wheels,
)
from wlr50_clean.reference.motion_contract import load_motion_contract

ROOT = Path(__file__).resolve().parents[2]
SPEC = yaml.safe_load((ROOT / "configs/ppo_rr_capture_then_rl_transfer_v1/stage_task_spec.yaml").read_text())
CONTRACT = load_motion_contract(ROOT / "configs/recording_motion_contract.json")


def fixture():
    """Real compact P09 source clock, synthetic current physical sensor values."""
    def support():
        return dict(support=True, bearing_verified=True, bearing_force_n=1., air=False,
            ground_contact=True, top_surface_contact=False, contact_surface="GROUND")
    legs = {leg: support() for leg in ("FL", "FR", "RL", "RR")}
    legs["RR"].update(support=False, bearing_force_n=0., air=True,
        ground_contact=False, contact_surface="NONE", obstacle_pair_active=False, within_top_xy=True,
        within_lateral_span=True, current_lift_valid=True, clearance_m=.03, front_distance_m=.005)
    history = {key: {leg: False for leg in legs} for key in ("active_lift", "front_edge_crossed", "placed")}
    history["active_lift"]["RR"] = history["front_edge_crossed"]["RR"] = True
    task = dict(stage_id="P09", termination_reason=None, physical_evaluator=dict(
        physics_tick=1000, valid=True, termination_reason=None, current_legs=legs, history=history))
    obs = dict(physics_tick=1000, obstacle=dict(front_x_m=1., back_x_m=2.))
    frame = SimpleNamespace(state_id="P09", physics_tick=1000)
    motion = MotionExecutor()
    motion.start_phase_at_endpoint(CONTRACT.phase("P09"))
    layer = dict(stage="P09", motion=motion, sample=motion.tick(), advanced_this_tick=True)
    provider = SimpleNamespace(spec=deepcopy(SPEC), _continuous_layers=[layer])
    kwargs = dict(task=task, observation=obs, source_frame=frame, nominal_provider=provider,
        physics_tick=1179, source_ack=None, previous_write_count=40, mode=MODE)
    first = build_rr_carry_wheel_context(**kwargs)
    assert not first["envelope_active"]
    candidate = [float(i) for i in range(8)] + [-.93, .06, -.02, -.08]
    # A synthetic committed one-write ACK is explicitly not an Isaac result.
    kwargs["source_ack"] = dict(physics_tick=1179, write_count=41,
        articulation_writes_this_call=1, requested_full12=list(layer["sample"].full12),
        drive_target_full12=candidate[:], **{ACK_KEY: dict(source_evidence=first["source_evidence"])})
    kwargs["previous_write_count"] = 41
    kwargs["physics_tick"] += 1
    frame.physics_tick += 1
    obs["physics_tick"] += 1
    task["physical_evaluator"]["physics_tick"] += 1
    layer["sample"] = motion.tick()
    return kwargs, candidate


def context(kwargs):
    return build_rr_carry_wheel_context(**kwargs)


def project(kwargs, candidate):
    return project_rr_carry_wheels(candidate, context=context(kwargs),
        previous_final_wheel_rad_s=kwargs["source_ack"]["drive_target_full12"][8:], physics_dt_s=1/120.)


def test_real_P09_endpoint_then_next_source_tick_arms_without_measured_wheel_stop():
    kwargs, candidate = fixture()
    result = project(kwargs, candidate)
    assert result["envelope_active"] and result["context"]["source_commit_verified"]
    assert result["selected_indices"] == [8, 9, 10]
    assert result["desired_before_slew_full12"][8:11] == [.3, .3, .3]
    assert result["output_full12"][8:11] == pytest.approx([-.915, .075, -.005])
    assert result["output_full12"][:8] == candidate[:8]
    assert result["output_full12"][11] == candidate[11]
    assert result["previous_final_wheel_rad_s"][0] < 0.  # Source stop != applied/actual stop.
    assert result["source_evidence"]["sample_tick"] > result["source_evidence"]["last_wheel_stop_tick"]


@pytest.mark.parametrize("distance,gap,expected", [(.005,.015,1.), (.0325,.015,.5),
    (.005,.0075,.5), (.0325,.0075,.25), (.06,.03,0.), (.08,.03,0.), (.005,0.,0.)])
def test_existing_depth_and_gap_scales_taper_but_zero_gain_keeps_scoped_zero_floor(distance, gap, expected):
    kwargs, candidate = fixture()
    kwargs["task"]["physical_evaluator"]["current_legs"]["RR"].update(front_distance_m=distance, clearance_m=gap)
    result = project(kwargs, candidate)
    assert result["context"]["gain"] == pytest.approx(expected)
    assert result["context"]["action"] == "forward_floor"
    assert result["desired_before_slew_full12"][8] == pytest.approx(.3*expected)
    assert result["output_full12"][8] == pytest.approx(-.915)


def test_already_forward_policy_can_exceed_advice_and_inputs_are_not_mutated():
    kwargs, candidate = fixture()
    candidate[8:11] = [.4, .5, .6]
    kwargs["source_ack"]["drive_target_full12"] = candidate[:]
    ctx = context(kwargs)
    before = deepcopy((ctx, candidate))
    result = project_rr_carry_wheels(candidate, context=ctx, previous_final_wheel_rad_s=candidate[8:])
    assert (ctx, candidate) == before
    assert result["envelope_active"] and not result["actual_projection_changed"]
    assert result["output_full12"] == candidate


def test_only_current_bearing_FL_and_FR_receive_floor_AIR_RL_and_RR_pass_through():
    kwargs, candidate = fixture()
    kwargs["task"]["physical_evaluator"]["current_legs"]["RL"].update(air=True, support=False)
    result = project(kwargs, candidate)
    assert result["selected_indices"] == [8, 9]
    assert result["output_full12"][10:] == candidate[10:]


@pytest.mark.parametrize("bad", ["no_FL", "one_support", "unverified_FL", "below_force_floor",
    "ground_RR", "wall_RR", "invalid", "terminal", "no_cross", "RL_placed", "P10", "P13"])
def test_safety_phase_and_real_support_bypass_never_rewrite_leg_or_wheel_source(bad):
    kwargs, candidate = fixture()
    task = kwargs["task"]
    ev, legs = task["physical_evaluator"], task["physical_evaluator"]["current_legs"]
    if bad == "no_FL": legs["FL"]["support"] = False
    elif bad == "one_support": legs["FR"]["support"] = legs["RL"]["support"] = False
    elif bad == "unverified_FL": legs["FL"]["bearing_verified"] = False
    elif bad == "below_force_floor": legs["FL"]["bearing_force_n"] = SPEC["support"]["force_noise_floor_n"]-.001
    elif bad == "ground_RR": legs["RR"]["ground_contact"] = True
    elif bad == "wall_RR": legs["RR"]["contact_surface"] = "FRONT_WALL"
    elif bad == "invalid": ev["valid"] = False
    elif bad == "terminal": ev["termination_reason"] = "BODY_COLLISION"
    elif bad == "no_cross": ev["history"]["front_edge_crossed"]["RR"] = False
    elif bad == "RL_placed": ev["history"]["placed"]["RL"] = True
    else: task["stage_id"] = kwargs["source_frame"].state_id = bad
    result = project(kwargs, candidate)
    assert not result["envelope_active"]
    assert result["output_full12"] == candidate


@pytest.mark.parametrize("bad", ["missing_ACK", "missing_source_commit", "stale_ACK", "wrong_write_count",
    "two_writes", "nonzero_nominal", "stale_source", "not_endpoint", "source_not_advanced"])
def test_nominal_zero_without_exact_committed_endpoint_and_adjacent_ACK_is_not_proof(bad):
    kwargs, candidate = fixture()
    ack = kwargs["source_ack"]
    if bad == "missing_ACK": kwargs["source_ack"] = None
    elif bad == "missing_source_commit": del ack[ACK_KEY]
    elif bad == "stale_ACK": ack["physics_tick"] -= 1
    elif bad == "wrong_write_count": ack["write_count"] -= 1
    elif bad == "two_writes": ack["articulation_writes_this_call"] = 2
    elif bad == "nonzero_nominal": ack["requested_full12"][8] = -.1
    elif bad == "stale_source": ack[ACK_KEY]["source_evidence"]["source_control_tick"] -= 1
    elif bad == "not_endpoint": ack[ACK_KEY]["source_evidence"]["endpoint_issued"] = False
    elif bad == "source_not_advanced": kwargs["nominal_provider"]._continuous_layers[0]["advanced_this_tick"] = False
    ctx = context(kwargs)
    result = project_rr_carry_wheels(candidate, context=ctx, previous_final_wheel_rad_s=None)
    assert not result["envelope_active"] and result["output_full12"] == candidate


def test_fresh_explicit_source_stop_or_other_wheel_owner_wins():
    kwargs, candidate = fixture()
    layer = kwargs["nominal_provider"]._continuous_layers[0]
    source_stop = max((g for g in layer["motion"].phase.atomic_groups if set(g.channels)&set(WHEEL_ORDER)), key=lambda g:g.time_s)
    fresh = dict(stage="P10", sample=replace(layer["sample"], atomic_groups=(source_stop,)), advanced_this_tick=True)
    kwargs["nominal_provider"]._continuous_layers.append(fresh)
    result = project(kwargs, candidate)
    assert not result["envelope_active"] and result["output_full12"] == candidate
    assert result["source_evidence"]["fresh_wheel_owners"]


@pytest.mark.parametrize("kind", ["TOP", "lost_Q", "lost_XY", "lost_lateral"])
def test_P09_exit_slews_to_original_candidate_without_zero_or_positive_floor(kind):
    kwargs, candidate = fixture()
    rr = kwargs["task"]["physical_evaluator"]["current_legs"]["RR"]
    if kind == "TOP": rr.update(air=False, top_surface_contact=True, obstacle_pair_active=True,
        contact_surface="TOP", clearance_m=-.001)
    else: rr[{"lost_Q":"current_lift_valid", "lost_XY":"within_top_xy", "lost_lateral":"within_lateral_span"}[kind]] = False
    kwargs["source_ack"]["drive_target_full12"][8:11] = [.2, .2, .2]
    result = project(kwargs, candidate)
    assert result["envelope_active"] and result["context"]["action"] == "release_slew"
    assert result["desired_before_slew_full12"] == candidate
    assert result["output_full12"][8:11] == pytest.approx([.185, .185, .185])
    assert result["current_TOP_has_forward_floor"] is False


def test_default_off_is_exact_identity_even_with_valid_RR_geometry():
    kwargs, candidate = fixture()
    del kwargs["mode"]
    result = project(kwargs, candidate)
    assert result["output_full12"] == candidate and not result["envelope_active"]


def test_actual_and_zero_current_policy_counterfactual_share_context_and_previous_FINAL():
    kwargs, actual = fixture()
    zero = actual[:8] + [0.]*4
    ctx, previous = context(kwargs), kwargs["source_ack"]["drive_target_full12"][8:]
    a = project_rr_carry_wheels(actual, context=ctx, previous_final_wheel_rad_s=previous)
    z = project_rr_carry_wheels(zero, context=ctx, previous_final_wheel_rad_s=previous)
    assert a["output_full12"][8:11] == z["output_full12"][8:11]
    assert a["output_full12"][11] != z["output_full12"][11]
    assert actual[8] == -.93 and a["raw_policy_and_log_probability_unchanged"]


@pytest.mark.parametrize("bad", ["stale_observation", "nonfinite_scale", "changed_rate", "different_previous",
    "wrong_dt", "hard_limit", "missing_RR_flag"])
def test_contract_errors_cannot_silently_arm_or_change_execution(bad):
    kwargs, candidate = fixture()
    if bad == "missing_RR_flag":
        del kwargs["task"]["physical_evaluator"]["current_legs"]["RR"]["within_top_xy"]
        assert not context(kwargs)["envelope_active"]
        return
    if bad == "stale_observation": kwargs["observation"]["physics_tick"] -= 1
    elif bad == "nonfinite_scale": kwargs["nominal_provider"].spec["geometry"]["workspace_max_m"] = float("nan")
    elif bad == "changed_rate": kwargs["wheel_rate_rad_s2"] = 3.
    with pytest.raises(ValueError):
        ctx = context(kwargs)
        previous = kwargs["source_ack"]["drive_target_full12"][8:]
        if bad == "different_previous": previous = [.1]*4
        if bad == "hard_limit": candidate[8] = WHEEL_VELOCITY_LIMIT_RAD_S+.001
        project_rr_carry_wheels(candidate, context=ctx, previous_final_wheel_rad_s=previous,
            physics_dt_s=1/60. if bad == "wrong_dt" else 1/120.)


@pytest.mark.parametrize("changes", [dict(obstacle_pair_active=True), dict(obstacle_pair_active=None),
    dict(contact_surface="TOP"), dict(top_surface_contact=True), dict(air=False),
    dict(air=False,contact_surface="TOP",top_surface_contact=True,obstacle_pair_active=False),
    dict(air=False,contact_surface="TOP",top_surface_contact=False,obstacle_pair_active=True)])
def test_unknown_or_contradictory_RR_contact_never_enables_floor_or_release(changes):
    kwargs, candidate = fixture()
    kwargs["task"]["physical_evaluator"]["current_legs"]["RR"].update(changes)
    result = project(kwargs, candidate)
    assert not result["envelope_active"] and result["output_full12"] == candidate
    assert "consistent_current_RR_contact_class" in result["context"]["reasons"]


def test_measured_platform_back_plane_bounds_advice_before_the_back_edge():
    kwargs, candidate = fixture()
    kwargs["observation"]["obstacle"]["back_x_m"] = 1.04
    kwargs["task"]["physical_evaluator"]["current_legs"]["RR"]["front_distance_m"] = .035
    result = project(kwargs, candidate)
    assert result["context"]["geometry_scales"]["depth_taper_m"] == pytest.approx(.035)
    assert result["context"]["gain"] == pytest.approx(0., abs=1e-12)
    assert result["desired_before_slew_full12"][8] == pytest.approx(0., abs=1e-12)


def test_too_narrow_measured_platform_yields_without_changing_scene_or_physics():
    kwargs, candidate = fixture()
    kwargs["observation"]["obstacle"]["back_x_m"] = 1.008
    result = project(kwargs, candidate)
    assert not result["envelope_active"] and result["output_full12"] == candidate
    assert result["context"]["reasons"] == ["insufficient_measured_platform_depth_for_existing_taper"]


@pytest.mark.parametrize("value", [None, float("nan"), .9, 1.])
def test_invalid_measured_platform_planes_are_not_replaced_with_assumed_width(value):
    kwargs, _ = fixture()
    kwargs["observation"]["obstacle"]["back_x_m"] = value
    with pytest.raises(ValueError):
        context(kwargs)

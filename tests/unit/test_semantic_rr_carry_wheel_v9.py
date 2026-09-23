"""P12 wheel-subchannel completion, not full-source or physical-success tests.

Prepared while physics is active: not executed. The imports intentionally use
the normal package; run against an explicitly selected candidate overlay or
after authorized adoption, never silently substitute the live runtime.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.fsm.motion_executor import MotionExecutor
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.ppo.semantic_rr_carry_wheel import (
    ACK_KEY, MODE, SEMANTICS, build_rr_carry_wheel_context, project_rr_carry_wheels,
)
from wlr50_clean.reference.motion_contract import load_motion_contract

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "configs/recording_motion_contract.json").is_file())
SPEC = yaml.safe_load((ROOT / "configs/ppo_rr_capture_then_rl_transfer_v1/stage_task_spec.yaml").read_text())
CONTRACT = load_motion_contract(ROOT / "configs/recording_motion_contract.json")


def fixture(phase="P12", *, current_source_tick=321, top=True):
    def support():
        return dict(support=True, bearing_verified=True, bearing_force_n=1., air=False,
            ground_contact=True, top_surface_contact=False, contact_surface="GROUND")
    legs = {leg: support() for leg in ("FL", "FR", "RL", "RR")}
    legs["RR"].update(ground_contact=False, obstacle_pair_active=top,
        air=not top, support=top, bearing_force_n=1. if top else 0.,
        top_surface_contact=top, contact_surface="TOP" if top else "NONE",
        within_top_xy=True, within_lateral_span=True, current_lift_valid=True,
        clearance_m=-.001 if top else .015, front_distance_m=.02)
    history = {key: {leg: False for leg in legs}
               for key in ("active_lift", "front_edge_crossed", "placed")}
    for key in history:
        history[key]["RR"] = True
    task = dict(stage_id=phase, termination_reason=None, physical_evaluator=dict(
        physics_tick=1000, valid=True, termination_reason=None, current_legs=legs, history=history))
    frame = SimpleNamespace(state_id=phase, physics_tick=1000)
    obs = dict(physics_tick=1000, obstacle=dict(front_x_m=1., back_x_m=2.))
    motion = MotionExecutor()
    motion.start_phase(CONTRACT.phase(phase))
    sample = None
    for _ in range(current_source_tick):
        sample = motion.tick()  # Previous sample index = current_source_tick-1.
    assert sample.tick_index == current_source_tick-1
    layer = dict(stage=phase, motion=motion, sample=sample, advanced_this_tick=True)
    provider = SimpleNamespace(spec=deepcopy(SPEC), _continuous_layers=[layer])
    kwargs = dict(task=task, observation=obs, source_frame=frame, nominal_provider=provider,
        physics_tick=1180, source_ack=None, previous_write_count=40, mode=MODE)
    before = build_rr_carry_wheel_context(**kwargs)
    candidate = list(map(float, range(8)))+[-.93, .06, -.02, -.08]
    kwargs["source_ack"] = dict(physics_tick=1180, write_count=41,
        articulation_writes_this_call=1, requested_full12=list(sample.full12),
        drive_target_full12=candidate[:], **{ACK_KEY: dict(source_evidence=before["source_evidence"])})
    kwargs["physics_tick"] += 1
    kwargs["previous_write_count"] = 41
    frame.physics_tick += 1
    obs["physics_tick"] += 1
    task["physical_evaluator"]["physics_tick"] += 1
    layer["sample"] = motion.tick()
    return kwargs, candidate


def project(kwargs, candidate):
    ctx = build_rr_carry_wheel_context(**kwargs)
    return project_rr_carry_wheels(candidate, context=ctx,
        previous_final_wheel_rad_s=kwargs["source_ack"]["drive_target_full12"][8:])


def test_old_P09_committed_ACK_never_arms_P12():
    kwargs, candidate = fixture()
    prior = kwargs["source_ack"][ACK_KEY]["source_evidence"]
    prior.update(phase="P09", source_phase="P09", endpoint_issued=True)
    result = project(kwargs, candidate)
    assert not result["envelope_active"]
    assert "previous_committed_source_phase" in result["context"]["source_commit_reasons"]
    assert result["output_full12"] == candidate


def test_real_P12_wheel_stop_and_next_ACK_arms_before_full_RL_endpoint():
    kwargs, candidate = fixture()
    result = project(kwargs, candidate)
    s = result["source_evidence"]
    assert s["source_phase"] == "P12" and s["sample_tick"] == 321
    assert s["last_wheel_stop_tick"] == 320 and s["endpoint_tick"] == 560
    assert s["wheel_stop_endpoint_issued"] and s["authored_reverse_pulse_verified"]
    assert not s["endpoint_issued"] and not s["source_full_endpoint_issued"]
    assert not result["context"]["full_source_endpoint_required"]
    assert result["envelope_active"] and result["selected_indices"] == [8, 9, 10, 11]
    assert result["current_TOP_has_forward_floor"]
    assert result["context"]["gain"] == pytest.approx((.06-.02)/(.06-.005))
    assert result["output_full12"][8:] == pytest.approx([-.915, .075, -.005, -.065])
    assert result["output_full12"][:8] == candidate[:8]


@pytest.mark.parametrize("tick", [57, 100, 319, 320])
def test_authored_negative_pulse_and_fresh_stop_win(tick):
    kwargs, candidate = fixture(current_source_tick=tick)
    result = project(kwargs, candidate)
    assert not result["envelope_active"] and result["output_full12"] == candidate
    if tick == 320:
        assert result["source_evidence"]["fresh_wheel_owners"]


def test_any_fresh_source_wheel_owner_wins_after_P12_stop():
    kwargs, candidate = fixture()
    layer = kwargs["nominal_provider"]._continuous_layers[0]
    group = layer["motion"].phase.atomic_groups[-1]
    kwargs["nominal_provider"]._continuous_layers.append(dict(stage="P13", advanced_this_tick=True,
        sample=replace(layer["sample"], atomic_groups=(group,))))
    assert not project(kwargs, candidate)["envelope_active"]


def test_live_nominal_positive_approach_after_stop_is_not_a_new_source_pulse():
    kwargs, candidate = fixture(current_source_tick=322)
    kwargs["source_ack"]["requested_full12"][8:] = [.3]*4
    result = project(kwargs, candidate)
    assert result["envelope_active"] and result["context"]["source_commit_verified"]
    assert result["source_evidence"]["source_wheel_target_rad_s"] == [0.]*4


def test_stop_tick_itself_requires_actual_zero_nominal_not_positive_advice():
    kwargs, candidate = fixture()  # Previous ACK is the authored stop's tick320.
    kwargs["source_ack"]["requested_full12"][8:] = [.3]*4
    assert not project(kwargs, candidate)["envelope_active"]


@pytest.mark.parametrize("bad", ["no_receipt", "stale_ACK", "wrong_write", "two_writes", "wrong_stop",
    "wrong_sample", "negative_N", "arbitrary_N", "missing_pulse_proof", "missing_stop_proof"])
def test_P12_stop_proof_does_not_accept_nominal_zero_alone(bad):
    kwargs, candidate = fixture()
    ack = kwargs["source_ack"]
    previous = ack[ACK_KEY]["source_evidence"]
    if bad == "no_receipt": del ack[ACK_KEY]
    elif bad == "stale_ACK": ack["physics_tick"] -= 1
    elif bad == "wrong_write": ack["write_count"] -= 1
    elif bad == "two_writes": ack["articulation_writes_this_call"] = 2
    elif bad == "wrong_stop": previous["last_wheel_stop_tick"] -= 1
    elif bad == "wrong_sample": previous["sample_tick"] -= 1
    elif bad == "negative_N": ack["requested_full12"][8:] = [-.3]*4
    elif bad == "arbitrary_N": ack["requested_full12"][8] = .2
    elif bad == "missing_pulse_proof": previous["authored_reverse_pulse_verified"] = False
    elif bad == "missing_stop_proof": previous["wheel_stop_endpoint_issued"] = False
    result = project(kwargs, candidate)
    assert not result["envelope_active"] and result["output_full12"] == candidate


def test_only_current_bearing_wheels_including_RR_receive_retention_floor():
    kwargs, candidate = fixture()
    legs = kwargs["task"]["physical_evaluator"]["current_legs"]
    for leg in ("FR", "RL"):
        legs[leg].update(support=False, air=True, ground_contact=False)
    result = project(kwargs, candidate)
    assert result["selected_indices"] == [8, 11]  # FL+RR allowed; not a stability assertion.
    assert result["output_full12"][9:11] == candidate[9:11]
    assert not result["added_forward_is_measured_traction"]


@pytest.mark.parametrize("gap", [.015, 0., -.00001, -.015])
def test_qualified_AIR_never_becomes_RR_bearing_or_receives_RR_wheel_floor(gap):
    kwargs, candidate = fixture(top=False)
    rr = kwargs["task"]["physical_evaluator"]["current_legs"]["RR"]
    rr["clearance_m"] = gap
    result = project(kwargs, candidate)
    assert result["envelope_active"] and result["context"]["action"] == "forward_floor"
    assert result["selected_indices"] == [8, 9, 10]
    assert result["output_full12"][11] == candidate[11]
    assert not result["current_TOP_has_forward_floor"] and not rr["support"]


def test_TOP_depth_gain_does_not_vanish_at_contact_and_deep_floor_stays_zero():
    kwargs, candidate = fixture()
    rr = kwargs["task"]["physical_evaluator"]["current_legs"]["RR"]
    for gap in (-.001, 0., .001):
        rr["clearance_m"] = gap
        assert project(kwargs, candidate)["context"]["gain"] > .7
    rr["front_distance_m"] = .06
    result = project(kwargs, candidate)
    assert result["context"]["gain"] == 0. and result["context"]["action"] == "forward_floor"
    assert result["desired_before_slew_full12"][8] == 0.


@pytest.mark.parametrize("bad", ["no_FL", "one_support", "RR_ground", "RR_wall", "RR_unknown",
    "unsafe", "RR_not_placed", "RL_placed", "P10", "P11", "P13"])
def test_support_safety_history_and_phase_loss_yield_without_hidden_latch(bad):
    kwargs, candidate = fixture()
    task = kwargs["task"]; ev = task["physical_evaluator"]; legs = ev["current_legs"]
    if bad == "no_FL": legs["FL"]["support"] = False
    elif bad == "one_support":
        for leg in ("FR", "RL", "RR"): legs[leg]["support"] = False
    elif bad == "RR_ground": legs["RR"]["ground_contact"] = True
    elif bad == "RR_wall": legs["RR"]["contact_surface"] = "FRONT_WALL"
    elif bad == "RR_unknown": legs["RR"]["obstacle_pair_active"] = None
    elif bad == "unsafe": ev["termination_reason"] = "BODY_COLLISION"
    elif bad == "RR_not_placed": ev["history"]["placed"]["RR"] = False
    elif bad == "RL_placed": ev["history"]["placed"]["RL"] = True
    else: task["stage_id"] = kwargs["source_frame"].state_id = bad
    result = project(kwargs, candidate)
    assert not result["envelope_active"] and result["output_full12"] == candidate


@pytest.mark.parametrize("bad", ["XY", "lateral", "low_gap", "high_gap", "AIR_Q"])
def test_lost_legal_capture_releases_at_same_previous_FINAL_rate(bad):
    kwargs, candidate = fixture(top=bad != "AIR_Q")
    rr = kwargs["task"]["physical_evaluator"]["current_legs"]["RR"]
    if bad == "XY": rr["within_top_xy"] = False
    elif bad == "lateral": rr["within_lateral_span"] = False
    elif bad == "low_gap": rr["clearance_m"] = -.015001
    elif bad == "high_gap": rr["clearance_m"] = .025001
    else: rr["current_lift_valid"] = False
    kwargs["source_ack"]["drive_target_full12"][8:] = [.2]*4
    result = project(kwargs, candidate)
    assert result["context"]["action"] == "release_slew"
    assert result["desired_before_slew_full12"] == candidate
    assert result["output_full12"][8] == pytest.approx(.185)


def test_actual_and_zero_residual_share_prestate_without_mutating_raw12_or_joint_channels():
    kwargs, actual = fixture()
    ctx = build_rr_carry_wheel_context(**kwargs)
    previous = kwargs["source_ack"]["drive_target_full12"][8:]
    zero = actual[:8]+[0.]*4
    before = deepcopy((kwargs["task"], ctx, actual, zero))
    a = project_rr_carry_wheels(actual, context=ctx, previous_final_wheel_rad_s=previous)
    z = project_rr_carry_wheels(zero, context=ctx, previous_final_wheel_rad_s=previous)
    assert a["output_full12"][8:] == z["output_full12"][8:]
    assert a["output_full12"][:8] == actual[:8] and z["output_full12"][:8] == zero[:8]
    assert (kwargs["task"], ctx, actual, zero) == before
    assert a["raw_policy_and_log_probability_unchanged"]
    assert SEMANTICS.startswith("P09_v8_unchanged_P12_")


def test_default_off_is_identity_in_P12():
    kwargs, candidate = fixture()
    kwargs["mode"] = "off"
    assert project(kwargs, candidate)["output_full12"] == candidate


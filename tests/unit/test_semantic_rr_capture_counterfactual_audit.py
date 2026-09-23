"""CPU synthetic ACK/float32 audit; no physical or learning-success evidence."""
from copy import deepcopy
from contextlib import nullcontext
from unittest.mock import patch

import pytest
torch = pytest.importorskip("torch")

from test_actuator_target_effect import _adapter
from test_semantic_rr_capture_assist import context
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.ppo.actuator_target_effect import ActuatorTargetEffectError, build_actuator_target_effect_audit
from wlr50_clean.ppo.isaac_fsm_backend import build_residual_actuation_plan
from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE
from wlr50_clean.ppo.semantic_residual_adapter import apply_semantic_residual
from wlr50_clean.ppo.semantic_rr_capture_assist import RRHipOnlyCaptureAssist
from wlr50_clean.ppo.semantic_tracking_reference import MODE, capture_tracking_reference_context

ZERO = (0.,)*12


def vector(hip, knee):
    return ZERO[:6]+(hip,knee)+ZERO[8:]


def dispatch_case(*, mode="follow", prior_request=(.4,.2), request=(.5,.3),
                  nominal=(.2,.1), geometry=False, with_wheel=False):
    adapter = _adapter()
    adapter.apply_full12(ZERO, physics_tick=0, tracking_servo_names=(), drive_feedback_bias_full12=ZERO)
    for tick in range(1,25):
        apply_semantic_residual(adapter,ZERO,physics_tick=tick,tracking_servo_names=(),
            controller_bias_full12=ZERO,projected_residual_full12=vector(*prior_request),
            policy_headroom_mode=HEADROOM_MODE,tracking_reference_mode=MODE,tracking_reference_bootstrap_tick=1)
    previous = tuple(adapter._final_drive_servo_deg[name] for name in SERVO_ORDER)
    rr = RRHipOnlyCaptureAssist()
    # Explicit synthetic public captured-state fixture, not fabricated robot data.
    rr.state.update(mode=7.,initialized=1.,contact_seen=1.,hip_entry_deg=previous[6],
        hip_target_deg=previous[6],knee_hold_deg=previous[7],hold_elapsed_s=.1)
    rr.last_tick = 24
    if mode == "release":
        rr.state.update(mode=4.,retired=1.,release_fraction=.25)
    elif mode == "wait":
        rr = RRHipOnlyCaptureAssist()
    elif mode == "unsafe":
        rr.state.update(mode=3.,blocked_reason=1.)
    before_snapshot = rr.snapshot()
    before_tracking = capture_tracking_reference_context(adapter,physics_tick=25,bootstrap_physics_tick=1)
    before_tracking["source_tracking_servo_names"] = []
    nominal, residual = vector(*nominal), vector(*request)
    plan = build_residual_actuation_plan(tuple(a+b for a,b in zip(nominal,residual)),
        frozen_nominal_full12=nominal,drive_feedback_bias_full12=ZERO,normal_drive_bias_full12=ZERO)
    ctx = context(rr,25,stage_id="P10",air=False,top_surface_contact=True,
        obstacle_pair_active=True,current_top_bearing=True,gap_m=0.,
        rl_placed_current_top_support=mode=="release",physical_valid=mode!="unsafe")
    extra = {}
    geometry_scope = nullcontext()
    if geometry:
        def geometry_adjust(*, native_full12, **kwargs):
            changed = list(native_full12)
            changed[7] += .4
            return tuple(changed), {"synthetic_single_rear_pair":True}
        # Isolate the existing geometry output seam; no FK/physics claim.
        geometry_scope = patch("wlr50_clean.ppo.semantic_nominal_geometry.correct_nominal_geometry", geometry_adjust)
        extra["nominal_geometry_context"] = {"dispatch_physics_tick":25,"canonical_servo_indices":[6,7]}
    previous_wheels = tuple(adapter.last_ack["drive_target_full12"][8:])
    wheel_context = None
    if with_wheel:
        from test_semantic_rr_carry_dispatch import context as wheel_fixture
        wheel_context = wheel_fixture(25,previous_wheels,active=False)
        extra["rr_carry_wheel_context"] = wheel_context
    with geometry_scope:
        ack = apply_semantic_residual(adapter,nominal,physics_tick=25,tracking_servo_names=(),
            controller_bias_full12=ZERO,projected_residual_full12=plan.projected_residual_full12,
            policy_headroom_mode=HEADROOM_MODE,tracking_reference_mode=MODE,tracking_reference_bootstrap_tick=1,
            rr_capture_assist=rr,rr_capture_assist_context=ctx,**extra)
    args = dict(adapter=adapter,actuation=plan,raw_ack=ack,previous_final_drive_servo_deg=previous,
        source_phase_id="P10",policy_request=None,policy_headroom_mode=HEADROOM_MODE,
        tracking_reference_mode=MODE,tracking_reference_context=deepcopy(before_tracking))
    if with_wheel:
        args.update(rr_carry_wheel_context=deepcopy(wheel_context),previous_final_drive_wheel_rad_s=previous_wheels)
    return args,rr,before_snapshot


def test_follow_replays_zero_minus_prior_request_not_zero_delta_or_actual_after():
    args,rr,before = dispatch_case()
    after = rr.snapshot()
    audit = build_actuator_target_effect_audit(**args)
    replay = audit["rr_capture_assist_branch_replays"]
    actual,zero = replay["actual"],replay["zero_current_policy"]
    assert actual["issued_inputs"]["issued_nominal_delta_rr_deg"] == [.2,.1]
    assert actual["issued_inputs"]["issued_requested_residual_delta_rr_deg"] == pytest.approx([.1,.1])
    assert zero["issued_inputs"]["issued_requested_residual_delta_rr_deg"] == [-.4,-.2]
    assert zero["state_after"]["hip_target_deg"] == pytest.approx(before["hip_target_deg"]+.2-.4)
    assert actual["state_after"] != zero["state_after"]
    assert audit["changed_channels_full12"][6:8] == [True,True]
    assert audit["rr_capture_issued_inputs_independently_verified"]
    assert rr.snapshot() == after and args["adapter"].write_count == 26


def test_release_uses_three_own_candidates_from_same_prestate_not_actual_after():
    args,rr,before = dispatch_case(mode="release",geometry=True)
    audit = build_actuator_target_effect_audit(**args)
    replay = audit["rr_capture_assist_branch_replays"]
    assert set(replay) == {"actual","zero_current_policy","zero_current_policy_without_geometry"}
    candidates = [replay[name]["issued_inputs"]["release_candidate_rr_deg"][1] for name in replay]
    assert len(set(candidates)) == 3
    fractions = [row["state_after"]["release_fraction"] for row in replay.values()]
    assert fractions == pytest.approx([.25+(1/120)/.75]*3)
    assert fractions[0] < .28  # no repeated mutation of one assist instance
    after_knee = [row["state_after"]["knee_hold_deg"] for row in replay.values()]
    assert len(set(after_knee)) == 3
    assert audit["nominal_geometry_changed_channels_full12"][7]
    assert audit["changed_channels_full12"][7]
    assert rr.snapshot() == replay["actual"]["state_after"]


@pytest.mark.parametrize("key",["issued_nominal_delta_rr_deg","issued_requested_residual_delta_rr_deg","release_candidate_rr_deg"])
@pytest.mark.parametrize("corruption",["missing","wrong","bool","nan"])
def test_all_issued_context_fields_fail_closed_even_in_wait(key,corruption):
    args,_,_ = dispatch_case(mode="wait")
    ctx = args["raw_ack"]["rr_capture_assist_evidence"]["context"]
    if corruption == "missing":
        del ctx[key]
    else:
        ctx[key][0] = {"wrong":9.,"bool":True,"nan":float("nan")}[corruption]
    with pytest.raises(ActuatorTargetEffectError,match="RR"):
        build_actuator_target_effect_audit(**args)


@pytest.mark.parametrize("corruption",["missing_precontext","wrong_previous_request","wrong_previous_nominal","stale_before","missing_initialized_before_tick","forged_increment"])
def test_independent_prestate_and_increment_receipt_are_not_trusted(corruption):
    args,_,_ = dispatch_case()
    if corruption == "missing_precontext":
        args["tracking_reference_context"] = None
    elif corruption == "wrong_previous_request":
        args["tracking_reference_context"]["previous_requested_full12"][6] += 1.
    elif corruption == "wrong_previous_nominal":
        args["tracking_reference_context"]["mapper_pre_state"]["requested_servo_deg"][6] += 1.
    elif corruption == "stale_before":
        args["raw_ack"]["rr_capture_assist_evidence"]["state_before"]["last_dispatch_physics_tick"] -= 1
    elif corruption == "missing_initialized_before_tick":
        args["raw_ack"]["rr_capture_assist_evidence"]["state_before"]["last_dispatch_physics_tick"] = None
    else:
        args["raw_ack"]["rr_capture_assist_evidence"]["captured_incremental_target"]["nominal_delta_rr_deg"][0] += 1.
    with pytest.raises(ActuatorTargetEffectError):
        build_actuator_target_effect_audit(**args)


def test_zero_current_request_and_support_wheel_share_same_prestate_no_effect():
    args,_,_ = dispatch_case(request=(0.,0.),with_wheel=True)
    audit = build_actuator_target_effect_audit(**args)
    replay = audit["rr_capture_assist_branch_replays"]
    assert replay["actual"]["state_after"] == replay["zero_current_policy"]["state_after"]
    assert not any(audit["changed_channels_full12"])
    assert audit["rr_carry_wheel_context_and_previous_FINAL_independently_verified"]


def test_unsafe_captured_pending_targets_differ_but_actual_previous_final_is_held():
    args,_,_ = dispatch_case(mode="unsafe")
    audit = build_actuator_target_effect_audit(**args)
    replay = audit["rr_capture_assist_branch_replays"]
    assert replay["actual"]["state_after"] != replay["zero_current_policy"]["state_after"]
    assert replay["actual"]["state_after"]["mode_name"] == "BLOCKED"
    assert audit["actual_native_targets"] == audit["counterfactual_native_targets"]
    assert args["raw_ack"]["drive_target_full12"][6:8] == list(args["previous_final_drive_servo_deg"][6:8])


def test_audit_does_not_advance_live_mapper_or_assist_or_dispatch(monkeypatch):
    args,rr,_ = dispatch_case()
    adapter = args["adapter"]
    snapshot,mapper,events = rr.snapshot(),deepcopy(adapter.servo_target_mapper.__dict__),list(adapter.robot.events)
    def forbidden(*a,**k):
        pytest.fail("audit mutated live control/dispatch")
    monkeypatch.setattr(rr,"advance",forbidden)
    monkeypatch.setattr(adapter.servo_target_mapper,"advance",forbidden)
    monkeypatch.setattr(adapter.robot,"write_data_to_sim",forbidden)
    build_actuator_target_effect_audit(**args)
    assert rr.snapshot() == snapshot and adapter.robot.events == events
    assert {k:v for k,v in adapter.servo_target_mapper.__dict__.items() if k!="advance"} == mapper


def test_forged_actual_float32_targets_still_fail_closed():
    args,_,_ = dispatch_case()
    adapter = args["adapter"]
    j = adapter.joint_map.servo_ids[7]
    adapter.robot.data.joint_pos_target[0,j] += .01
    adapter.robot._joint_pos_target_sim[0,j] += .01
    with pytest.raises(ActuatorTargetEffectError,match="actual dispatch"):
        build_actuator_target_effect_audit(**args)

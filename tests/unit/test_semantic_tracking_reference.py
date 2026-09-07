"""Pure CPU reference/ACK tests; no Torch, GPU, simulator or physical write."""
import copy
import json
import math
import struct
from types import SimpleNamespace as NS

import pytest

from wlr50_clean.infrastructure.command_batch import FULL12_ORDER, SERVO_ORDER, SERVO_COMMAND_SIGN, servo_limits_deg
from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapper
from wlr50_clean.ppo.semantic_tracking_reference import (
    MODE, EVIDENCE_SCHEMA, SemanticTrackingReferenceError,
    capture_tracking_reference_context, build_tracking_reference,
)


class Matrix:
    def __init__(self, row):
        self.row = list(row)
        self.shape = (1, len(row))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return Matrix([self.row[i] for i in key[1]])
        assert key == 0
        return self.row


def fixture(*, tick=20, request=None, nominal=None, canonical=None):
    standing = {n: 11.+i for i,n in enumerate(SERVO_ORDER)}
    nominal = [0.]*8 if nominal is None else list(nominal)
    canonical = nominal if canonical is None else list(canonical)
    q = [math.radians(standing[n]+SERVO_COMMAND_SIGN[n]*v) for n,v in zip(SERVO_ORDER,canonical)]
    mapper = ServoTargetMapper(standing)
    mapper._feedback_tick = tick
    mapper._requested.update(zip(SERVO_ORDER,nominal))
    mapper._applied.update(zip(SERVO_ORDER,nominal))
    ack = {"schema":"wlr50_clean.atomic_full12_ack.v1", "physics_tick":tick-1,
           "write_count":tick, "articulation_writes_this_call":1, "canonical_order":list(FULL12_ORDER),
           "requested_full12":nominal+[0.]*4, "applied_full12":nominal+[0.]*4,
           "native_drive_target_full12":nominal+[0.]*4, "drive_target_full12":nominal+[0.]*4,
           "drive_feedback_bias_requested_full12":[0.]*12, "servo_native_drive_command_deg":nominal,
           "servo_tracking_compensation_deg":[0.]*8, "servo_nominal_target_reached":[True]*8,
           "servo_tracking_active":[False]*8, "tracking_servo_names":[],
           "servo_tracking_feedback_sample_tick":tick-1}
    if request is not None:
        ack.update(semantic_residual_composition="independent_post_mapper_residual.v1",
            tracking_reference_mode=MODE,
            tracking_reference_evidence={"schema":EVIDENCE_SCHEMA,"mode":MODE,
                "dispatch_physics_tick":tick-1,"mapper_feedback_tick":tick-1},
            independent_policy_residual_requested_full12=list(request),
            bounded_controller_bias_requested_full12=[0.]*12,
            drive_feedback_bias_requested_full12=list(request),
            policy_headroom_evidence={"effective_policy_residual_full12":[0.]*12})
    return NS(last_ack=ack, _last_physics_tick=tick-1, write_count=tick,
        standing_pose_deg=standing, servo_target_mapper=mapper,
        joint_map=NS(servo_ids=tuple(range(8))), robot=NS(data=NS(joint_pos=Matrix(q))))


def one(i, value, size=12):
    result=[0.]*size; result[i]=value; return result


def build(adapter, *, nominal=None, names=SERVO_ORDER, bootstrap=None):
    return build_tracking_reference(capture_tracking_reference_context(adapter,
        physics_tick=adapter._last_physics_tick+1, bootstrap_physics_tick=bootstrap),
        requested_command_deg=nominal or list(adapter.servo_target_mapper._requested.values()),
        tracking_servo_names=names)


def test_explicit_bootstrap_original_ack_is_read_only_json_and_not_implicit():
    a=fixture(tick=1)
    before=copy.deepcopy(a.__dict__)
    with pytest.raises(SemanticTrackingReferenceError): build(a)
    result=build(a,bootstrap=1)
    assert result["previous_request_source"]=="explicit_original_zero_bootstrap"
    assert result["previous_requested_full12"]==[0.]*12
    assert json.loads(json.dumps(result,allow_nan=False))==result
    assert a.last_ack==before["last_ack"]
    assert a.servo_target_mapper.__dict__==before["servo_target_mapper"].__dict__
    assert a.robot.data.joint_pos.row==before["robot"].data.joint_pos.row
    assert result["mapper_advances"]==result["articulation_writes"]==0


@pytest.mark.parametrize("i",range(8))
@pytest.mark.parametrize("r",[-3.7,3.7])
def test_all_signs_requested_reference_even_if_previous_effective_was_zero(i,r):
    a=fixture(request=one(i,r))
    result=build(a)
    q=a.robot.data.joint_pos.row[i]
    assert result["previous_requested_full12"][i]==r
    assert result["mapper_computational_reference_rad"][i]==q-SERVO_COMMAND_SIGN[SERVO_ORDER[i]]*math.radians(r)
    row=result["channels"][i]
    assert row["reference_used"] and row["desired_reference_c1_deg"]==pytest.approx(math.copysign(10.,r))
    assert all(result["mapper_computational_reference_rad"][j]==a.robot.data.joint_pos.row[j] for j in range(8) if j!=i)


@pytest.mark.parametrize("i",range(8))
@pytest.mark.parametrize("side",[-1,1])
def test_reserve_bounds_with_original_baseline_exception_and_actual_frozen_advance(i,side):
    lo,hi=servo_limits_deg(SERVO_ORDER[i]); n=hi-5 if side>0 else lo+5
    nom=one(i,n,8); a=fixture(request=one(i,side*3.7),nominal=nom)
    evidence=build(a)
    row=evidence["channels"][i]
    assert row["bounded_desired_c1_deg"]==pytest.approx(side*3.)
    assert abs(row["reconstruction_error_deg"])<1e-12
    mapped=a.servo_target_mapper.advance(nom,evidence["mapper_computational_reference_rad"],tracking_servo_names=SERVO_ORDER)
    assert abs(mapped.tracking_compensation_deg[i])<=1.25
    assert lo+2-1e-12<=mapped.applied_drive_command_deg[i]<=hi-2+1e-12
    # Original same-state load error already needs 10 degrees: preserve the exception.
    q=list(nom); q[i]-=side*2
    b=fixture(request=one(i,side*3.7),nominal=nom,canonical=q)
    second=build(b)["channels"][i]
    assert second["desired_original_c0_deg"]==pytest.approx(side*10)
    assert second["bounded_desired_c1_deg"]==pytest.approx(side*10)


@pytest.mark.parametrize("i",range(8))
@pytest.mark.parametrize("edge",["lower","upper"])
def test_hard_boundary_outward_requested_correction_is_zero(i,edge):
    lo,hi=servo_limits_deg(SERVO_ORDER[i]); n=lo if edge=="lower" else hi
    a=fixture(request=one(i,-20 if edge=="lower" else 20),nominal=one(i,n,8))
    result=build(a)
    assert result["channels"][i]["bounded_desired_c1_deg"]==pytest.approx(0.,abs=1e-12)


@pytest.mark.parametrize("case",["zero","gain0","inactive","nonsample","changed","not_reached","ended","retiring"])
def test_ineligible_path_exact_q_and_frozen_state_transition(case):
    a=fixture(tick=21 if case=="nonsample" else 20,request=one(0,0. if case=="zero" else 3.7))
    mapper=a.servo_target_mapper; nom=[0.]*8; names=SERVO_ORDER
    if case=="gain0": mapper.tracking_gain=0.
    if case in ("inactive","ended","retiring"): names=[]
    if case=="changed": nom[0]=5.
    if case=="not_reached": mapper._nominal_reached[SERVO_ORDER[0]]=False; a.last_ack["servo_nominal_target_reached"][0]=False
    if case in ("ended","retiring"):
        mapper._tracking_active[SERVO_ORDER[0]]=True; a.last_ack["servo_tracking_active"][0]=True
        mapper._compensation[SERVO_ORDER[0]]=5.; mapper._applied[SERVO_ORDER[0]]=5.
        a.last_ack["servo_tracking_compensation_deg"][0]=5.; a.last_ack["servo_native_drive_command_deg"][0]=5.
    if case=="retiring": mapper._retiring_stale_bias[SERVO_ORDER[0]]=True
    actual=list(a.robot.data.joint_pos.row); original=copy.deepcopy(mapper)
    result=build(a,nominal=nom,names=names)
    assert result["mapper_computational_reference_rad"]==actual
    assert not result["channels"][0]["reference_used"]
    assert mapper.advance(nom,actual,tracking_servo_names=names)==original.advance(nom,result["mapper_computational_reference_rad"],tracking_servo_names=names)


def test_signed_zero_gain_and_phase_label_do_not_change_reference():
    a=fixture(request=[-0.]*12)
    a.robot.data.joint_pos.row[0]=-0.
    before=build(a)
    a.phase_id="P13"
    after=build(a)
    assert before==after
    assert struct.pack("d",after["mapper_computational_reference_rad"][0])==struct.pack("d",-0.)


def test_current_zero_is_not_an_input_and_previous_request_still_used():
    a=fixture(request=one(0,3.7))
    a.current_policy_residual_full12=[0.]*12
    assert build(a)["channels"][0]["reference_used"]


@pytest.mark.parametrize("case",["noack","wrong_schema","missing_mode","missing_receipt","bad_receipt_tick","bad_receipt_feedback",
    "nan_request","bool_request","bad_combined","stale_tick","fraction_tick","bool_tick","feedback","write_count","mapping",
    "negative_gain","nan_q","bad_ids","batch2"])
def test_capture_fail_closed(case):
    a=fixture(request=one(0,3.7)); tick=20
    if case=="noack": a.last_ack=None
    elif case=="wrong_schema": a.last_ack["schema"]="bad"
    elif case=="missing_mode": del a.last_ack["tracking_reference_mode"]
    elif case=="missing_receipt": del a.last_ack["tracking_reference_evidence"]
    elif case=="bad_receipt_tick": a.last_ack["tracking_reference_evidence"]["dispatch_physics_tick"]=18
    elif case=="bad_receipt_feedback": a.last_ack["tracking_reference_evidence"]["mapper_feedback_tick"]=True
    elif case=="nan_request": a.last_ack["independent_policy_residual_requested_full12"][0]=float("nan")
    elif case=="bool_request": a.last_ack["independent_policy_residual_requested_full12"][0]=True
    elif case=="bad_combined": a.last_ack["drive_feedback_bias_requested_full12"][0]=0.
    elif case=="stale_tick": tick=21
    elif case=="fraction_tick": tick=20.0
    elif case=="bool_tick": tick=True
    elif case=="feedback": a.servo_target_mapper._feedback_tick=21
    elif case=="write_count": a.write_count=19
    elif case=="mapping": a.last_ack["servo_native_drive_command_deg"][0]=1.
    elif case=="negative_gain": a.servo_target_mapper.tracking_gain=-1.
    elif case=="nan_q": a.robot.data.joint_pos.row[0]=float("nan")
    elif case=="bad_ids": a.joint_map.servo_ids=(0,)*8
    elif case=="batch2": a.robot.data.joint_pos.shape=(2,8)
    with pytest.raises(SemanticTrackingReferenceError): capture_tracking_reference_context(a,physics_tick=tick)


@pytest.mark.parametrize("case",["tick","nonzero","tracking","compensation","semantic_field","clock"])
def test_original_zero_bootstrap_cannot_be_used_as_missing_history_fallback(case):
    a=fixture(tick=20); bootstrap=20
    if case=="tick": bootstrap=19
    elif case=="nonzero": a.last_ack["requested_full12"][0]=1.
    elif case=="tracking": a.last_ack["tracking_servo_names"]=[SERVO_ORDER[0]]
    elif case=="compensation": a.servo_target_mapper._compensation[SERVO_ORDER[0]]=1.; a.last_ack["servo_tracking_compensation_deg"][0]=1.
    elif case=="semantic_field": a.last_ack["semantic_residual_composition"]="independent_post_mapper_residual.v1"
    elif case=="clock": a.servo_target_mapper._feedback_tick=19
    with pytest.raises(SemanticTrackingReferenceError): build(a,bootstrap=bootstrap)


@pytest.mark.parametrize("bad",[True,float("nan"),float("inf"),"1"])
def test_invalid_nominal_and_context_numbers(bad):
    a=fixture(request=one(0,3.7)); context=capture_tracking_reference_context(a,physics_tick=20)
    with pytest.raises(SemanticTrackingReferenceError): build_tracking_reference(context,requested_command_deg=one(0,bad,8),tracking_servo_names=SERVO_ORDER)
    context["tracking_gain"]=bad
    with pytest.raises(SemanticTrackingReferenceError): build_tracking_reference(context,requested_command_deg=[0.]*8,tracking_servo_names=SERVO_ORDER)


def test_old_compensation_is_not_instantly_relocated_to_reserve():
    a=fixture(request=one(0,3.7),nominal=one(0,130.,8))
    m=a.servo_target_mapper; m._compensation[SERVO_ORDER[0]]=5.; m._applied[SERVO_ORDER[0]]=135.
    a.last_ack["servo_tracking_compensation_deg"][0]=5.; a.last_ack["servo_native_drive_command_deg"][0]=135.
    evidence=build(a)
    result=m.advance(one(0,130.,8),evidence["mapper_computational_reference_rad"],tracking_servo_names=SERVO_ORDER)
    assert evidence["channels"][0]["bounded_desired_c1_deg"]==pytest.approx(3.)
    assert result.tracking_compensation_deg[0]==3.75  # Old slew, not instant desired=3.
    assert result.applied_drive_command_deg[0]==133.75
    assert "old_slew" in evidence["reserve_scope"]

"""Bounded pure-math checks; no Torch, simulator, model, or optimizer."""
import math
import random
import struct
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from wlr50_clean.ppo.semantic_headroom import (
    HEADROOM_MODE, SemanticHeadroomError, project_semantic_servo_headroom,
    validate_semantic_servo_headroom_config,
)
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg

ZERO = (0.0,)*12


def one(index, value):
    values=list(ZERO); values[index]=value
    return values


def bits(value):
    return struct.pack(">d", value)


def test_actual_base_config_margin_and_frozen_hard_limits():
    config=yaml.safe_load((PROJECT_ROOT/"configs/ppo_action_projection.yaml").read_text())
    validate_semantic_servo_headroom_config(HEADROOM_MODE,config["joint_safety_margin_deg"])
    result=project_semantic_servo_headroom(ZERO,ZERO,ZERO)
    assert result["servo_safety_limits_deg"]==[[-133.,133.],[-58.,208.]]*4
    assert result["servo_hard_limits_deg"]==[list(servo_limits_deg(n)) for n in SERVO_ORDER]
    assert config["absolute_action_limits"]["hip_deg"]==[-135.,135.]
    assert config["absolute_action_limits"]["knee_deg"]==[-60.,210.]


def test_real_static_counterexample_uses_mapper_output_not_logical_nominal():
    logical_nominal=-37.8  # Not an input to the post-mapper helper.
    result=project_semantic_servo_headroom(one(7,-27.8),ZERO,one(7,-24.98))
    assert -58-logical_nominal==pytest.approx(-20.2)
    assert result["effective_policy_residual_full12"][7]==-24.98
    assert result["candidate_native_target_before_final_slew_full12"][7]==pytest.approx(-52.78)
    assert result["clipped_servo_indices"]==[]


def test_reverse_boundary_cannot_spend_nominal_headroom():
    result=project_semantic_servo_headroom(one(7,-58),ZERO,one(7,-24.98))
    assert result["effective_policy_residual_full12"][7]==0
    assert result["candidate_native_target_before_final_slew_full12"][7]==-58
    assert result["clipped_servo_indices"]==[7]


@pytest.mark.parametrize("policy_request,effective,target",[(0.,0.,-59.),(-0.,-0.,-59.),(.5,.5,-58.5),(-.5,0.,-59.)])
def test_outside_reserve_never_relocates_baseline(policy_request,effective,target):
    result=project_semantic_servo_headroom(one(7,-59),ZERO,one(7,policy_request))
    assert bits(result["effective_policy_residual_full12"][7])==bits(effective)
    assert result["candidate_native_target_before_final_slew_full12"][7]==target
    assert result["policy_residual_intervals_servo_deg"][7]==[0.,267.]
    assert result["baseline_outside_reserved_servo_indices"]==[7]


def test_geometry_corrected_baseline_not_uncorrected_or_previous_slew_target():
    corrected=project_semantic_servo_headroom(one(7,-40),ZERO,one(7,-10))
    uncorrected=project_semantic_servo_headroom(one(7,-58),ZERO,one(7,-10))
    assert corrected["effective_policy_residual_full12"][7]==-10
    assert uncorrected["effective_policy_residual_full12"][7]==0
    # A different previous final target cannot be supplied through this ABI.
    with pytest.raises(TypeError):
        project_semantic_servo_headroom(one(7,-58),ZERO,one(7,-10),previous_slew_target=one(7,-20))
    with pytest.raises(TypeError):
        project_semantic_servo_headroom(one(7,-58),ZERO,one(7,-10),logical_nominal=one(7,-37.8))


def test_bounded_controller_is_in_headroom_but_residual_not_capped_to_ten():
    result=project_semantic_servo_headroom(one(7,-50),one(7,10),one(7,-30))
    assert result["baseline_native_plus_controller_full12"][7]==-40
    assert result["effective_policy_residual_full12"][7]==-18
    assert result["effective_combined_post_mapper_bias_full12"][7]==-8
    assert result["candidate_native_target_before_final_slew_full12"][7]==-58


@pytest.mark.parametrize("index",range(8))
@pytest.mark.parametrize("side",["lower","upper"])
def test_each_servo_bound_and_one_degree_reserve_violation(index,side):
    hard_lo,hard_hi=servo_limits_deg(SERVO_ORDER[index]); lo,hi=hard_lo+2,hard_hi-2
    boundary=lo if side=="lower" else hi
    outward=-.25 if side=="lower" else .25
    result=project_semantic_servo_headroom(one(index,boundary),ZERO,one(index,outward))
    assert result["effective_policy_residual_full12"][index]==0
    assert result["candidate_native_target_before_final_slew_full12"][index]==boundary
    inner=project_semantic_servo_headroom(one(index,boundary),ZERO,one(index,-outward))
    assert inner["effective_policy_residual_full12"][index]==-outward
    outside=boundary+(-1 if side=="lower" else 1)
    partial=project_semantic_servo_headroom(one(index,outside),ZERO,one(index,-outward))
    assert partial["candidate_native_target_before_final_slew_full12"][index]==outside-outward


@pytest.mark.parametrize("value",[0.,-0.,1e-9,-1e-9,1e-15,-1e-15,5e-324,-5e-324])
def test_signed_zero_and_tiny_inside_residual_bits_are_preserved(value):
    result=project_semantic_servo_headroom([30.]*12,ZERO,[value]*12)
    assert all(bits(x)==bits(value) for x in result["effective_policy_residual_full12"])
    if value==0:
        assert result["candidate_native_target_before_final_slew_full12"]==[30.]*12
    assert result["clipped_servo_indices"]==[]


@pytest.mark.parametrize("value",[-1e-9,-1e-15,-5e-324])
def test_tiny_outward_residual_outside_reserve_does_not_relocate_baseline(value):
    result=project_semantic_servo_headroom(one(7,-59),ZERO,one(7,value))
    assert result["effective_policy_residual_full12"][7]==0
    assert result["candidate_native_target_before_final_slew_full12"][7]==-59


def test_signed_zero_native_is_not_rewritten_by_zero_offsets():
    result=project_semantic_servo_headroom([-0.]*12,[0.]*12,[-0.]*12)
    assert all(bits(x)==bits(-0.) for x in result["baseline_native_plus_controller_full12"])
    assert all(bits(x)==bits(-0.) for x in result["candidate_native_target_before_final_slew_full12"][:8])


def test_wheel_signed_zero_composition_matches_legacy_operations():
    for native in (0.,-0.):
        for controller in (0.,-0.):
            for residual in (0.,-0.):
                result=project_semantic_servo_headroom([native]*12,[controller]*12,[residual]*12)
                for i in range(8,12):
                    assert bits(result["effective_policy_residual_full12"][i])==bits(residual)
                    assert bits(result["effective_combined_post_mapper_bias_full12"][i])==bits(controller+residual)
                    assert bits(result["candidate_native_target_before_final_slew_full12"][i])==bits(native+(controller+residual))


def test_wheel_residuals_are_unchanged_even_beyond_hard_target_limits():
    native=list(ZERO); native[8:]=[2.,-2.,1.,-1.]
    requests=list(ZERO);requests[8:]=[100.,-100.,-0.,1e-300]
    result=project_semantic_servo_headroom(native,ZERO,requests)
    assert result["effective_policy_residual_full12"][8:]==requests[8:]
    assert result["candidate_native_target_before_final_slew_full12"][8:10]==[102.,-102.]
    assert bits(result["effective_policy_residual_full12"][10])==bits(-0.)
    assert result["wheel_residual_unchanged"] is True


@pytest.mark.parametrize("seed",[7,101,1001,20260906])
def test_bounded_random_finite_properties(seed):
    rng=random.Random(seed)
    for _ in range(250):
        native=[rng.uniform(-300,300) for _ in range(12)]
        controller=[rng.uniform(-10,10) for _ in range(8)]+[rng.uniform(-2,2) for _ in range(4)]
        request=[rng.uniform(-600,600) for _ in range(12)]
        saved=(tuple(native),tuple(controller),tuple(request))
        result=project_semantic_servo_headroom(native,controller,request)
        for i in range(8):
            base=native[i]+controller[i];lo,hi=result["servo_safety_limits_deg"][i]
            interval=(min(0.,lo-base),max(0.,hi-base))
            eff=result["effective_policy_residual_full12"][i]
            assert result["policy_residual_intervals_servo_deg"][i]==list(interval)
            assert eff==max(interval[0],min(interval[1],request[i]))
            assert abs(eff)<=abs(request[i])
            assert eff==0 or math.copysign(1,eff)==math.copysign(1,request[i])
            # The reachable envelope includes a legacy baseline outside reserve.
            target=result["candidate_native_target_before_final_slew_full12"][i]
            assert min(lo,base)-1e-10<=target<=max(hi,base)+1e-10
            if base<lo: assert eff>=0
            if base>hi: assert eff<=0
        assert result["effective_policy_residual_full12"][8:]==request[8:]
        assert saved==(tuple(native),tuple(controller),tuple(request))


@pytest.mark.parametrize("argument",range(3))
@pytest.mark.parametrize("bad",[None,[],[0.]*11,[0.]*13,"0"*12,{str(i):0 for i in range(12)}])
def test_bad_shape_rejected(argument,bad):
    args=[ZERO,ZERO,ZERO];args[argument]=bad
    with pytest.raises(SemanticHeadroomError):project_semantic_servo_headroom(*args)


@pytest.mark.parametrize("argument",range(3))
@pytest.mark.parametrize("bad",[float("nan"),float("inf"),-float("inf"),True,"1",complex(1,0)])
def test_bad_elements_rejected(argument,bad):
    args=[ZERO,ZERO,ZERO];args[argument]=one(7,bad)
    with pytest.raises(SemanticHeadroomError):project_semantic_servo_headroom(*args)


@pytest.mark.parametrize("index,value",[(0,10.01),(7,-10.01),(8,2.1),(11,-2.1)])
def test_unbounded_controller_rejected(index,value):
    with pytest.raises(SemanticHeadroomError):
        project_semantic_servo_headroom(ZERO,one(index,value),ZERO)


def test_composition_overflow_rejected_not_silently_clamped():
    with pytest.raises(SemanticHeadroomError):
        project_semantic_servo_headroom(one(8,1e308),ZERO,one(8,1e308))


@pytest.mark.parametrize("mode",[None,"","other",True,1])
def test_unknown_mode_rejected(mode):
    with pytest.raises(SemanticHeadroomError):validate_semantic_servo_headroom_config(mode,{"hip":2.,"knee":2.})


@pytest.mark.parametrize("margins",[None,{}, {"hip":2.}, {"hip":1.99,"knee":2.},
    {"hip":2.,"knee":2.01},{"hip":True,"knee":2.},{"hip":"2","knee":2.},
    {"hip":float("nan"),"knee":2.}])
def test_incompatible_margin_rejected(margins):
    with pytest.raises(SemanticHeadroomError):validate_semantic_servo_headroom_config(HEADROOM_MODE,margins)

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_actuator_target_effect import _adapter
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg, WHEEL_VELOCITY_LIMIT_RAD_S
from wlr50_clean.infrastructure.robot_adapter import RobotAdapterError
from wlr50_clean.ppo.actuator_target_effect import build_actuator_target_effect_audit, actuator_target_audit_request
from wlr50_clean.ppo.isaac_fsm_backend import build_residual_actuation_plan, IsaacFSMBackend
from wlr50_clean.ppo.semantic_residual_adapter import apply_semantic_residual, SemanticActuationDispatch
from wlr50_clean.ppo.semantic_backend import SemanticIsaacBackend

ZERO=(0.,)*12


def plan(residual, controller=ZERO, nominal=ZERO):
    return build_residual_actuation_plan(tuple(a+b for a,b in zip(nominal,residual)),
        frozen_nominal_full12=nominal,drive_feedback_bias_full12=controller,normal_drive_bias_full12=ZERO)


def dispatch(adapter, actuation, tick):
    return IsaacFSMBackend._atomic_apply(None,SemanticActuationDispatch(adapter,actuation),
        actuation.frozen_nominal_full12,physics_tick=tick,tracking_servo_names=(),
        drive_feedback_bias_full12=actuation.combined_post_mapper_bias_full12)


@pytest.mark.parametrize("residual",[(0.,)*12,(.15,)*8+(.03,)*4,(40.,-40.)*4+(3.,)*4])
def test_native_composition_preserves_mapper_and_same_tick_audit(residual,monkeypatch):
    adapter=_adapter()
    nominal=_adapter()
    controller=(2.,-2.)*4+(.1,)*4
    actuation=plan(residual,controller)
    advance=adapter.servo_target_mapper.advance
    calls=[]
    def counted(*args,**kwargs):
        calls.append(args[0])
        return advance(*args,**kwargs)
    monkeypatch.setattr(adapter.servo_target_mapper,"advance",counted)
    for tick in range(1,49):
        previous=tuple(adapter._final_drive_servo_deg.values())
        ack=dispatch(adapter,actuation,tick)
        reference=nominal.apply_full12(ZERO,physics_tick=tick,tracking_servo_names=(),drive_feedback_bias_full12=controller)
        assert ack["native_drive_target_full12"]==reference["native_drive_target_full12"]
        assert ack["bounded_controller_bias_requested_full12"]==list(controller)
        assert ack["independent_policy_residual_requested_full12"]==list(residual)
        assert ack["applied_full12"]==list(ZERO)
        assert ack["articulation_writes_this_call"]==1 and ack["write_count"]==tick
        for name,before,after in zip(SERVO_ORDER,previous,ack["drive_target_full12"]):
            lower,upper=servo_limits_deg(name)
            assert lower<=after<=upper
            assert abs(after-before)<=adapter.servo_target_mapper.maximum_delta_deg+1e-12
        assert max(abs(x) for x in ack["drive_target_full12"][8:])<=WHEEL_VELOCITY_LIMIT_RAD_S
        audit=build_actuator_target_effect_audit(adapter=adapter,actuation=actuation,raw_ack=ack,
            previous_final_drive_servo_deg=previous,source_phase_id="P06",
            policy_request=actuator_target_audit_request("P06",(.5,)*12,(1,)*12))
        assert audit["verified"] and audit["setter_dispatch_targets_equal"]
        if not any(residual):
            assert audit["changed_target_channel_count"]==0
            assert ack["drive_target_full12"]==reference["drive_target_full12"]
            assert torch.equal(adapter.robot._joint_pos_target_sim,nominal.robot._joint_pos_target_sim)
            assert torch.equal(adapter.robot._joint_vel_target_sim,nominal.robot._joint_vel_target_sim)
    assert len(calls)==48 and calls==[ZERO[:8]]*48
    assert adapter.robot.events==["position.setter","velocity.setter","dispatch"]*48
    if max(abs(x) for x in residual)>10:
        assert max(abs(x) for x in ack["drive_feedback_bias_realized_full12"][:8])>10
        assert audit["changed_target_channel_count"]>0


@pytest.mark.parametrize("residual",[ZERO,(50.,)*12])
def test_controller_correction_cap_stays_frozen_even_when_policy_cancels(residual):
    adapter=_adapter()
    with pytest.raises(RobotAdapterError,match="bounded mapper envelope"):
        apply_semantic_residual(adapter,ZERO,physics_tick=1,tracking_servo_names=(),
            controller_bias_full12=(10.01,)+ZERO[1:],projected_residual_full12=residual)
    assert adapter.write_count==0 and not adapter.robot.events
    with pytest.raises(RobotAdapterError,match="bounded mapper envelope"):
        adapter.apply_full12(ZERO,physics_tick=1,drive_feedback_bias_full12=(40.,)*12)


def test_rejected_plan_never_advances_or_dispatches():
    adapter=_adapter()
    bound=SemanticActuationDispatch(adapter,plan((40.,)*12))
    with pytest.raises(RobotAdapterError,match="differs"):
        bound.apply_full12((1.,)*12,physics_tick=1,tracking_servo_names=(),drive_feedback_bias_full12=(40.,)*12)
    with pytest.raises(RobotAdapterError,match="finite"):
        apply_semantic_residual(adapter,ZERO,physics_tick=1,tracking_servo_names=(),
            controller_bias_full12=ZERO,projected_residual_full12=(float("nan"),)*12)
    assert adapter.write_count==0 and not adapter.robot.events


def test_zero_current_residual_after_large_history_has_zero_counterfactual_effect():
    adapter=_adapter()
    for tick in range(1,30):dispatch(adapter,plan((40.,)*8+ZERO[8:]),tick)
    before=tuple(adapter._final_drive_servo_deg.values())
    actuation=plan(ZERO)
    ack=dispatch(adapter,actuation,30)
    assert ack["drive_target_full12"][:8]!=list(ZERO[:8])
    audit=build_actuator_target_effect_audit(adapter=adapter,actuation=actuation,raw_ack=ack,
        previous_final_drive_servo_deg=before,source_phase_id="P07",
        policy_request=actuator_target_audit_request("P07",ZERO,(1,)*12))
    assert audit["changed_target_channel_count"]==0


def test_v2_does_not_opt_in_and_v3_explicitly_does():
    from wlr50_clean.ppo.semantic_cli import version_paths
    old=SemanticIsaacBackend()
    new=SemanticIsaacBackend(execution_profile=version_paths("v3")[2]/"execution_profile.yaml")
    assert old._independent_policy_residual is False
    assert new._independent_policy_residual is True


def test_new_range_only_uses_one_reset_and_crosses_old_envelope(monkeypatch,tmp_path):
    from test_semantic_prefix import real_prefix_runtime
    from wlr50_clean.ppo import semantic_workspace_probe as probe
    from wlr50_clean.ppo.semantic_cli import version_paths
    runtime,backend=real_prefix_runtime(monkeypatch,"P06")
    # The shared physical fixture defaults to v2. Explicitly select the same
    # v3 execution factor used by the live probe, without touching its mapper.
    configured=SemanticIsaacBackend(execution_profile=version_paths("v3")[2]/"execution_profile.yaml")
    backend._independent_policy_residual=configured._independent_policy_residual
    args=SimpleNamespace(run_dir=tmp_path,seed=1001,from_phase="P06",teacher_offset_decisions=0,
        decisions=32,raw_magnitude=.5,segments="new_range")
    result=probe.run_segments(None,args,{"offline_fixture":True},backend=backend)
    assert result["independent_physical_resets"]==runtime.reset_scene_count==1
    assert result["same_raw_old_new"] is False
    assert result["optimized_policy_decisions"]==0
    segment=result["segments"][0]
    assert segment["segment"]=="new_range" and segment["actual_response_decisions"]==32
    assert segment["response_physics_ticks"]==256
    assert segment["actual_native_effect_ticks"]>0
    assert backend._last_atomic_ack["controller_bias_original_envelope_verified"]
    assert max(abs(x) for x in backend._last_atomic_ack["independent_policy_residual_requested_full12"][:8])>10


def test_probe_segment_cli_defaults_and_rejects_unknown():
    from wlr50_clean.ppo.semantic_workspace_probe import parser
    base=["--run-dir","unused","--expected-head","unused"]
    assert parser().parse_args(base).segments=="all"
    assert parser().parse_args(base+["--segments","new_range"]).segments=="new_range"
    with pytest.raises(SystemExit):parser().parse_args(base+["--segments","old_range"])

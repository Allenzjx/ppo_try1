"""Steer-focused CPU seams only; no Isaac or PPO training credit."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/unit"))
from test_actuator_target_effect import _adapter
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER, WHEEL_FORWARD_SIGN
from wlr50_clean.ppo.semantic_backend import build_semantic_projector
from wlr50_clean.ppo.semantic_residual_adapter import apply_semantic_residual
from wlr50_clean.ppo.phase_action_masks_v2 import PhaseTransitionBridge

ZERO = (0.,) * 12
N = (0.,) * 8 + (.30, .22, -.17, .41)
PROFILE = ROOT / "configs/ppo_fsm_reference_p09_stable_v2/execution_profile.yaml"

def project(projector, raw, *, phase="P02", mask=(1,)*12, previous=ZERO):
    return projector.project(raw, state_id=phase, nominal_action_full12=N,
        reference_action_full12=N, reference_delta_full12=ZERO,
        runtime_action_mask_full12=mask, previous_projected_residual_full12=previous)

@pytest.mark.parametrize("phase", ["P01", "P02", "P03"])
@pytest.mark.parametrize("previous", [ZERO, (0.,)*8+(.2,)*4])
def test_residual_permission_mask_never_masks_nominal(phase, previous):
    out = project(build_semantic_projector(PROFILE), (.7,)*12,
        phase=phase, mask=(1,)*8+(0,)*4, previous=previous)
    assert out.effective_action_mask_full12[8:] == (0,)*4
    assert out.safe_projected_residual_full12[8:] == (0.,)*4
    assert out.applied_action_full12[8:] == N[8:]

@pytest.mark.parametrize("channel", range(4))
@pytest.mark.parametrize("delta", [-.13, .13])
def test_one_residual_writes_only_its_wheel_on_reversed_joint_indices(channel, delta):
    adapter = _adapter()  # Deliberately reversed articulation joint ordering.
    residual = list(ZERO)
    residual[8+channel] = delta
    ack = apply_semantic_residual(adapter, N, physics_tick=1, tracking_servo_names=(),
        controller_bias_full12=ZERO, projected_residual_full12=residual)
    expected = list(N[8:]); expected[channel] += delta
    native = [value * WHEEL_FORWARD_SIGN[name] for name,value in zip(WHEEL_ORDER, expected)]
    assert ack["native_drive_target_full12"][8:] == list(N[8:])
    assert ack["drive_target_full12"][8:] == pytest.approx(expected)
    assert ack["wheel_joint_ids"] == list(adapter.joint_map.wheel_ids)
    assert adapter.robot._joint_vel_target_sim[0, list(adapter.joint_map.wheel_ids)].tolist() == pytest.approx(native)
    assert adapter.robot.events == ["position.setter", "velocity.setter", "dispatch"]
    assert ack["articulation_writes_this_call"] == 1

def test_zero_residual_retains_all_four_nominal_targets():
    adapter = _adapter()
    ack = apply_semantic_residual(adapter, N, physics_tick=1, tracking_servo_names=(),
        controller_bias_full12=ZERO, projected_residual_full12=ZERO)
    assert ack["drive_target_full12"][8:] == list(N[8:])
    assert all(value != 0 for value in ack["wheel_target_physical_rad_s"])

@pytest.mark.parametrize("pair", [("P01","P02"),("P02","P03")])
def test_phase_handoff_preserves_nominal_and_residual_history(pair):
    bridge = PhaseTransitionBridge(build_semantic_projector(PROFILE))
    previous = (0.,)*8 + (.08,-.07,.06,-.05)
    bridge.reset(state_id=pair[0], projected_residual_full12=previous,
        applied_action_full12=tuple(a+b for a,b in zip(N,previous)))
    out = bridge.project_tick(ZERO, state_id=pair[1], nominal_action_full12=N,
        reference_action_full12=N, reference_delta_full12=ZERO)
    assert out.projection.safe_projected_residual_full12 == pytest.approx(previous)
    assert out.projection.applied_action_full12[8:] == pytest.approx(tuple(a+b for a,b in zip(N[8:],previous[8:])))
    bridge.reset(state_id="P01", applied_action_full12=N)
    assert bridge.previous_projected_residual_full12 == ZERO
    assert bridge.projector.config.mask_for("P02") == (1,)*12

def test_policy_cancellation_zero_target_and_measured_velocity_are_distinct():
    adapter = _adapter()
    # Existing physical velocity is not changed by staging a new zero target.
    ids = list(adapter.joint_map.wheel_ids)
    adapter.robot.data.joint_vel[0,ids] = torch.tensor([.4,-.3,.2,-.1])
    residual = (0.,)*8 + (-N[8],0.,0.,0.)
    ack = apply_semantic_residual(adapter, N, physics_tick=1, tracking_servo_names=(),
        controller_bias_full12=ZERO, projected_residual_full12=residual)
    assert ack["native_drive_target_full12"][8] == .3
    assert ack["independent_policy_residual_requested_full12"][8] == -.3
    assert ack["drive_target_full12"][8] == 0.
    assert float(adapter.robot.data.joint_vel[0,ids[0]]) == pytest.approx(.4)
    assert adapter.get_actual_full12()[8] == pytest.approx(-.4)

def diagnostic_module():
    spec = importlib.util.spec_from_file_location("wheel_diagnostic_isolation_fixture", Path(__file__).with_name("run_p02_mask_diagnostic.py"))
    module = importlib.util.module_from_spec(spec)
    before = list(sys.path)
    try: spec.loader.exec_module(module)
    finally: sys.path[:] = before
    return module

def test_diagnostic_wrapper_restores_cli_functions_on_exception(monkeypatch, tmp_path):
    diagnostic = diagnostic_module()
    cli = diagnostic.cli
    names = ("build_video_core", "checkpoint_loader", "capture_semantic_video", "write_json")
    original = {name:getattr(cli,name) for name in names}
    args = SimpleNamespace(semantic_version="v3", from_phase="P01", num_envs=1,
        expected_head="0"*40, checkpoint=tmp_path/"unused.pt", resume_migration=None,
        run_dir=tmp_path/"never_started")
    monkeypatch.setattr(cli, "parser", lambda:SimpleNamespace(parse_args=lambda _:args))
    monkeypatch.setattr(cli, "validate_video_args", lambda _:"C")
    class ExpectedStop(Exception): pass
    def stopped(_):
        assert all(getattr(cli,name) is not original[name] for name in names)
        raise ExpectedStop()
    monkeypatch.setattr(cli,"main",stopped)
    before_environment = dict(os.environ)
    with pytest.raises(ExpectedStop): diagnostic.main(["--diagnostic-mask","wheels4"])
    assert all(getattr(cli,name) is original[name] for name in names)
    assert dict(os.environ) == before_environment
    assert not args.run_dir.exists()
    actor_raw = tuple(.1*(i+1) for i in range(12))
    assert diagnostic.dispatch_raw(actor_raw,(8,9,10,11))[8:] == (0.,)*4
    assert diagnostic.dispatch_raw(actor_raw,()) == actor_raw
    assert build_semantic_projector(PROFILE).config.mask_for("P02") == (1,)*12

"""DEFERRED: CPU Torch/real mapper/native buffers, never Isaac.

Do NOT execute or collect this during an Isaac run. Root must explicitly set
RR_DIAG_NATIVE_TESTS_IDLE=1 after its resource audit. This file is not evidence
of physical capture. Its evaluator geometry is a synthetic test fixture.
"""
import os
from pathlib import Path
import sys

import pytest

if os.environ.get("RR_DIAG_NATIVE_TESTS_IDLE") != "1":
    pytest.skip("deferred until root confirms no active Isaac", allow_module_level=True)

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/wlr50_clean").is_dir())
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/unit"))

import torch
from test_semantic_tracking_reference_dispatch import adapter, context as tracking_context, dispatch, audit
from test_semantic_residual_adapter import plan
from test_target_plan import context as probe_context
from isolated_headroom_wrapper import IsolatedHeadroomProbe
from wlr50_clean.ppo import semantic_headroom
from wlr50_clean.ppo.actuator_target_effect import ActuatorTargetEffectError
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER

ZERO = (0.,) * 12


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def prepared():
    a = adapter()
    residual = (.1,) * 8 + (.05, -.06, .07, -.08)
    p = plan(residual)
    # Normal real frozen mapper + tracking dispatch, not fabricated last_ACK.
    dispatch(a, p, 1)
    c = probe_context(tick=2)
    c.update(previous_final_full12=tuple(a.last_ack["drive_target_full12"]),
        previous_effective_residual_full12=tuple(a.last_ack["policy_headroom_evidence"]["effective_policy_residual_full12"]),
        actual_full12=tuple(a.last_ack["drive_target_full12"]),
        final_slew_deg_per_tick=a.servo_target_mapper.maximum_delta_deg)
    c["residual_caps_full12"][6:8] = [24., 36.]
    w = IsolatedHeadroomProbe(semantic_headroom.project_semantic_servo_headroom,
        read_committed_adapter_tick=lambda: a._last_physics_tick, diagnostic_only=True)
    w.arm_from_observer(c)
    ctx = tracking_context(a, 2)
    ctx["source_tracking_servo_names"] = []
    previous = tuple(a._final_drive_servo_deg[name] for name in SERVO_ORDER)
    return a, p, c, w, ctx, previous


def test_actual_dispatch_independent_auditor_and_float32_readback_remain_enabled():
    a, p, c, w, ctx, previous = prepared()
    writes, mapper_ticks = a.write_count, a.servo_target_mapper.feedback_tick
    with w.installed_on(semantic_headroom):
        ack = dispatch(a, p, 2)
        proof = audit(a, p, ack, previous, ctx)
        assert proof["verified"] and proof["setter_dispatch_targets_equal"]
        assert proof["actual_mapping_matches_dispatch"]
        assert w.plan_calls == 1  # dispatch only; actual/zero audit did not advance
        assert a.write_count == writes+1
        assert a.servo_target_mapper.feedback_tick == mapper_ticks+1
        diagnostic = ack["policy_headroom_evidence"]["explicit_RR_direction_diagnostic"]
        assert diagnostic["new_PPO_decisions"] == diagnostic["new_PPO_updates"] == 0
        assert diagnostic["raw_policy_sample_was_executed_unmodified"] is False
        assert ack["independent_policy_residual_requested_full12"] == list(p.projected_residual_full12)
        # A no-policy counterfactual retains the same exogenous RR target.
        assert proof["actual_native_targets"]["servo_position_rad"][6:8] == (
            proof["counterfactual_native_targets"]["servo_position_rad"][6:8])
        # No expected-result bypass: actual setters and simulator buffers match.
        assert torch.equal(a.robot.data.joint_pos_target, a.robot._joint_pos_target_sim)
        assert torch.equal(a.robot.data.joint_vel_target, a.robot._joint_vel_target_sim)
        assert all(t.device.type == "cpu" for t in
                   (a.robot._joint_pos_target_sim, a.robot._joint_vel_target_sim))


def test_real_auditor_rejects_corrupted_dispatched_tensor_despite_diagnostic_wrapper():
    a, p, c, w, ctx, previous = prepared()
    with w.installed_on(semantic_headroom):
        ack = dispatch(a, p, 2)
        a.robot._joint_pos_target_sim[0, a.joint_map.servo_ids[7]] += .1
        with pytest.raises(ActuatorTargetEffectError):
            audit(a, p, ack, previous, ctx)


def test_real_auditor_rejects_forged_headroom_and_wrapper_restores_after_failure():
    a, p, c, w, ctx, previous = prepared()
    original = semantic_headroom.project_semantic_servo_headroom
    with w.installed_on(semantic_headroom):
        ack = dispatch(a, p, 2)
        ack["policy_headroom_evidence"]["explicit_RR_direction_diagnostic"]["dispatch_tick"] = 999
        with pytest.raises(ActuatorTargetEffectError, match="headroom evidence"):
            audit(a, p, ack, previous, ctx)
    assert semantic_headroom.project_semantic_servo_headroom is original


def test_preedge_wait_is_exact_original_dispatch_not_hidden_selected_channel_hold():
    a, p, c, w, ctx, previous = prepared()
    reference, rp, rc, rw, rctx, rprevious = prepared()
    w.pending["context"]["evaluation"]["current_legs"]["RR"]["within_top_xy"] = False
    with w.installed_on(semantic_headroom):
        ack = dispatch(a, p, 2)
        assert audit(a, p, ack, previous, ctx)["verified"]
    other = dispatch(reference, rp, 2)
    assert ack == other
    assert w.planner.state == "WAIT" and w.planner.anchor is None
    assert torch.equal(a.robot._joint_pos_target_sim, reference.robot._joint_pos_target_sim)
    assert torch.equal(a.robot._joint_vel_target_sim, reference.robot._joint_vel_target_sim)

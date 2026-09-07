"""Transfer-role range checks on the real CPU dispatch path, not physics.

The existing tensor-buffer fixture skips live USD only. Mapper, requested
tracking reference, headroom, final hard/slew and float32 audit are production.
Synthetic held load error is not evidence that an articulation follows a target.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import yaml

from test_actuator_target_effect import _adapter
from test_semantic_residual_adapter import plan
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
from wlr50_clean.ppo.actuator_target_effect import (
    actuator_target_audit_request, build_actuator_target_effect_audit,
)
from wlr50_clean.ppo.isaac_fsm_backend import IsaacFSMBackend
from wlr50_clean.ppo.semantic_backend import build_semantic_projector
from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE, project_semantic_servo_headroom
from wlr50_clean.ppo.semantic_residual_adapter import SemanticActuationDispatch
from wlr50_clean.ppo.semantic_tracking_reference import MODE, capture_tracking_reference_context
from wlr50_clean.ppo.semantic_observation import load_semantic_observation_schema
from wlr50_clean.ppo.phase_action_masks_v2 import PhaseTransitionBridge

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "configs/ppo_semantic_v3/execution_profile.yaml"
SCHEMA = ROOT / "configs/ppo_semantic_v3/observation_schema.json"
ZERO = (0.,) * 12
PHASES = tuple(f"P{i:02}" for i in range(1, 14))


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    patch = pytest.MonkeyPatch()
    patch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert not torch.cuda.is_initialized()
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU target-buffer tests must not initialize CUDA")
    patch.setattr(torch.cuda, "_lazy_init", forbidden)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
        patch.undo()


def full(value, index):
    return tuple(value if i == index else 0. for i in range(12))


def project(projector, raw, previous, *, phase, nominal):
    return projector.project(raw, state_id=phase, nominal_action_full12=nominal,
        reference_action_full12=nominal, reference_delta_full12=ZERO,
        previous_projected_residual_full12=previous,
        runtime_action_mask_full12=(1,) * 12, dt_s=1/120.)


def adapter():
    a = _adapter()
    a.apply_full12(ZERO, physics_tick=0, tracking_servo_names=(),
                   drive_feedback_bias_full12=ZERO)
    return a


def step(a, residual, nominal, tick, *, controller=ZERO, tracking=(), raw=ZERO,
         phase="P06", audit=False):
    p = plan(residual, nominal=nominal, controller=controller)
    before = tuple(a._final_drive_servo_deg.values())
    context = capture_tracking_reference_context(a, physics_tick=tick, bootstrap_physics_tick=1)
    context["source_tracking_servo_names"] = list(tracking)
    counts = a.write_count, a.servo_target_mapper.feedback_tick
    ack = IsaacFSMBackend._atomic_apply(None, SemanticActuationDispatch(
        a, p, policy_headroom_mode=HEADROOM_MODE, tracking_reference_mode=MODE,
        tracking_reference_bootstrap_tick=1), p.frozen_nominal_full12,
        physics_tick=tick, tracking_servo_names=tracking,
        drive_feedback_bias_full12=p.combined_post_mapper_bias_full12)
    assert (a.write_count, a.servo_target_mapper.feedback_tick) == tuple(x+1 for x in counts)
    assert ack["articulation_writes_this_call"] == 1
    assert ack["motion_start_skew_s"] == 0.
    assert ack["applied_full12"] == list(nominal)
    for i, name in enumerate(SERVO_ORDER):
        lo, hi = servo_limits_deg(name)
        assert lo <= ack["drive_target_full12"][i] <= hi
        assert abs(ack["drive_target_full12"][i]-before[i]) <= 1.25+1e-12
    if audit:
        effect = build_actuator_target_effect_audit(adapter=a, actuation=p, raw_ack=ack,
            previous_final_drive_servo_deg=before, source_phase_id=phase,
            policy_request=actuator_target_audit_request(phase, raw, (1,) * 12),
            policy_headroom_mode=HEADROOM_MODE, tracking_reference_mode=MODE,
            tracking_reference_context=context)
        assert effect["verified"] and effect["actual_mapping_matches_dispatch"]
        assert effect["setter_dispatch_targets_equal"]
        assert effect["target_dtype"] == "torch.float32"
        assert (a.write_count, a.servo_target_mapper.feedback_tick) == tuple(x+1 for x in counts)
    return ack


def test_only_selected_caps_expand_and_all_twelve_remain_open():
    cfg = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    rows = cfg["residual"]["phase_caps_full12"]
    assert tuple(rows) == PHASES
    for phase in PHASES:
        expected = ([18,24,18,24,12,18,12,18,.6,.6,.6,.6] if phase < "P06"
                    else [24,36,24,112,24,36,24,36,1.2,1.2,.6,.6])
        assert rows[phase] == expected
    assert all(all(b >= a for a,b in zip(rows[p], rows[q])) for p,q in zip(PHASES, PHASES[1:]))
    assert cfg["physics_hz"] == 120 and cfg["decision_hz"] == 15
    assert cfg["residual"]["servo_rate_deg_s"] == 60
    assert cfg["residual"]["wheel_rate_rad_s2"] == 1.8
    assert cfg["residual"]["allowed_channels"] == "all_12_in_every_phase"
    projector = build_semantic_projector(PROFILE)
    for phase in PHASES:
        out = project(projector, (.2,) * 12, ZERO, phase=phase, nominal=ZERO)
        assert all(out.safe_projected_residual_full12)


@pytest.mark.parametrize("phase", ["P01", "P05"])
@pytest.mark.parametrize("index", range(8, 12))
@pytest.mark.parametrize("desired", [-.02, 0., .02])
def test_early_pending_capture_wheel_can_cancel_and_reverse_with_finite_latent(phase, index, desired):
    nominal = full(.3, index)
    raw = full(math.atanh((desired-.3)/.6), index)
    assert math.isfinite(raw[index]) and abs(raw[index]) < 1.
    a, projector, previous = adapter(), build_semantic_projector(PROFILE), ZERO
    ticks = math.ceil(abs(desired-.3)/.015)
    for tick in range(1, ticks+1):
        out = project(projector, raw, previous, phase=phase, nominal=nominal)
        request = out.safe_projected_residual_full12
        assert abs(request[index]-previous[index]) <= .015+1e-14
        ack = step(a, request, nominal, tick, raw=raw, phase=phase, audit=tick == ticks)
        previous = request
    assert ack["drive_target_full12"][index] == pytest.approx(desired, abs=1e-14)
    sign = (-1., 1., -1., 1.)[index-8]
    expected = torch.tensor(sign*desired, dtype=torch.float32).item()
    actual = a.robot._joint_vel_target_sim[0, a.joint_map.wheel_ids[index-8]].item()
    assert actual == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize("nominal_knee", [31.1, 45.9])
@pytest.mark.parametrize("controller_knee", [-10., 0., 10.])
def test_fr_knee_candidate_reachable_after_real_mapper_plus_bias_and_both_slews(nominal_knee, controller_knee):
    a, nominal = adapter(), full(nominal_knee, 3)
    controller = full(controller_knee, 3)
    # Real mapper develops +10 from a held synthetic measured load error.
    # After the authored segment ends, nonconverged measured q retains this
    # legal bias. No mapper state or physical buffer is overwritten by the test.
    for tick in range(1, 101):
        ack = step(a, ZERO, nominal, tick, controller=controller, tracking=(SERVO_ORDER[3],))
    assert ack["servo_tracking_compensation_deg"][3] == 10.
    baseline = nominal_knee+10.+controller_knee
    desired, cap = -40., 112.
    residual = desired-baseline
    raw = full(math.atanh(residual/cap), 3)
    assert abs(residual) < cap and math.isfinite(raw[3])
    projector, previous = build_semantic_projector(PROFILE), ZERO
    ticks = math.ceil(abs(residual)/.5)
    for i in range(1, ticks+1):
        out = project(projector, raw, previous, phase="P06", nominal=nominal)
        request = out.safe_projected_residual_full12
        assert abs(request[3]-previous[3]) <= .5+1e-13
        ack = step(a, request, nominal, 100+i, controller=controller, raw=raw, audit=i == ticks)
        previous = request
    assert ack["native_drive_target_full12"][3] == pytest.approx(nominal_knee+10., abs=1e-13)
    assert ack["policy_headroom_evidence"]["effective_policy_residual_full12"][3] == pytest.approx(residual, abs=1e-12)
    assert ack["drive_target_full12"][3] == pytest.approx(desired, abs=1e-12)
    assert all(value == 0 for i,value in enumerate(ack["drive_target_full12"]) if i != 3)
    actual = a.robot._joint_pos_target_sim[0, a.joint_map.servo_ids[3]].item()
    assert actual == torch.tensor(.75+math.radians(desired), dtype=torch.float32).item()
    # FR native sign is +1; -40 is relative to standing, not articulation -40.
    assert ack["servo_target_physical_rad"][3] == pytest.approx(.75+math.radians(-40.), abs=1e-14)
    assert ticks <= 212  # Worst case: 105.9/60 seconds rounded to 120 Hz ticks.


def test_96_counterexample_and_original_hard_reserve_not_relaxed():
    native, controller = full(55.9, 3), full(10., 3)
    old = project_semantic_servo_headroom(native, controller, full(-96., 3))
    assert old["candidate_native_target_before_final_slew_full12"][3] == pytest.approx(-30.1)
    assert -105.9 < -96.  # Desired -40 is outside this old requested range.
    for value, expected in [(-112., -46.1), (112., 177.9)]:
        result = project_semantic_servo_headroom(native, controller, full(value, 3))
        assert result["candidate_native_target_before_final_slew_full12"][3] == pytest.approx(expected)
        assert result["servo_safety_limits_deg"][3] == [-58., 208.]
    assert servo_limits_deg(SERVO_ORDER[3]) == (-60., 210.)
    clipped = project_semantic_servo_headroom(full(0., 3), ZERO, full(-112., 3))
    assert clipped["effective_policy_residual_full12"][3] == -58.


def test_large_fr_request_crosses_phase_without_reset_and_slews_on_withdrawal():
    bridge = PhaseTransitionBridge(build_semantic_projector(PROFILE))
    history = full(-105.9, 3)
    bridge.reset(state_id="P06", projected_residual_full12=history, applied_action_full12=history)
    args = dict(state_id="P07", nominal_action_full12=ZERO, reference_action_full12=ZERO,
                reference_delta_full12=ZERO, dt_s=1/120.)
    held = bridge.project_tick(ZERO, **args)
    assert held.transition_metric.handoff_hold_used
    assert held.projection.safe_projected_residual_full12 == history
    assert held.transition_metric.phase_scale_clipped_channel_indices == ()
    next_tick = bridge.project_tick(ZERO, **args)
    assert next_tick.projection.safe_projected_residual_full12[3] == pytest.approx(-105.4)


def test_both_fr_requested_history_columns_remain_observable_without_changing_raw_slice():
    schema = load_semantic_observation_schema(SCHEMA)
    assert schema.dimension == 324 and schema.clip == 20.
    groups = {row["name"]: (0.,) * row["size"] for row in schema.groups}
    offsets, cursor = {}, 0
    for row in schema.groups:
        offsets[row["name"]] = cursor
        cursor += row["size"]
    assert offsets["previous_raw_full12"] == 195
    for name, expected in [("previous_residual_full12", 210), ("previous_previous_residual_full12", 222)]:
        row = next(row for row in schema.groups if row["name"] == name)
        assert row["scale"] == [4,4,4,6,4,4,4,4,.12,.12,.12,.12]
        assert offsets[name]+3 == expected
        for sign in [-1., 1.]:
            groups[name] = full(sign*112., 3)
            encoded = schema.encode(groups)
            assert encoded[expected] == sign*112./6.
            assert abs(encoded[expected]) < 20.
            assert encoded[expected]*6 == pytest.approx(sign*112., abs=1e-14)
    # Compatible old-domain input-column conversion only; full model/RNG
    # migration is tested separately by its owner, not asserted by this scalar.
    assert (36./4.)*2. == (36./6.)*(2.*1.5)


@pytest.mark.parametrize("phase", ["P01", "P05", "P06", "P13"])
def test_zero_policy_uses_original_frozen_dispatch_and_has_no_target_effect(phase):
    a, original = adapter(), adapter()
    nominal = (1.,2.,3.,4.,5.,6.,7.,8.,.3,.3,.3,.3)
    projection = project(build_semantic_projector(PROFILE), ZERO, ZERO, phase=phase, nominal=nominal)
    assert projection.zero_residual_fast_path
    ack = step(a, projection.safe_projected_residual_full12, nominal, 1, phase=phase, audit=True)
    old = original.apply_full12(nominal, physics_tick=1, tracking_servo_names=(), drive_feedback_bias_full12=ZERO)
    assert ack["drive_target_full12"] == old["drive_target_full12"]
    assert torch.equal(a.robot._joint_pos_target_sim, original.robot._joint_pos_target_sim)
    assert torch.equal(a.robot._joint_vel_target_sim, original.robot._joint_vel_target_sim)

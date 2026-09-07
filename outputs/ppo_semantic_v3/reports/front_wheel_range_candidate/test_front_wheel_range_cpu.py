"""Unwired 1.2/1.2/.6/.6 candidate: actual CPU projector/adapter, no physics."""
from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pytest
import torch
import yaml

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests/unit"))
from test_actuator_target_effect import _adapter
from test_semantic_residual_adapter import dispatch, plan
from test_semantic_observation_reward_env import _frame
from wlr50_clean.infrastructure.command_batch import WHEEL_VELOCITY_LIMIT_RAD_S as LIMIT
from wlr50_clean.ppo.actuator_target_effect import actuator_target_audit_request, build_actuator_target_effect_audit
from wlr50_clean.ppo.semantic_backend import build_semantic_projector
from wlr50_clean.ppo.phase_action_masks_v2 import PhaseTransitionBridge
from wlr50_clean.ppo.semantic_observation import (
    HISTORY_GROUPS, SemanticObservationBuilder, load_semantic_observation_schema,
)

ZERO = (0.,) * 12
PHASES = tuple(f"P{i:02}" for i in range(1, 14))
PROFILE = ROOT / "configs/ppo_semantic_v3/execution_profile.yaml"
SCHEMA = ROOT / "configs/ppo_semantic_v3/observation_schema.json"
SIGN = (-1., 1., -1., 1.)
REQUIREMENTS = (("P07", 9, -.63), ("P09", 8, -1.07),
                ("P13", 9, -.72), ("P13", 8, 1.09))


@pytest.fixture(scope="session", autouse=True)
def cpu_only_and_source_bytes_unchanged():
    originals = {path: path.read_bytes() for path in (PROFILE, SCHEMA)}
    assert not torch.cuda.is_initialized()
    patch = pytest.MonkeyPatch()
    def forbidden(*args, **kwargs):
        raise AssertionError("candidate CPU tests must not initialize CUDA")
    patch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    assert torch.get_num_threads() == torch.get_num_interop_threads() == 1
    assert not torch.cuda.is_initialized()
    assert not any(name in sys.modules for name in ("isaaclab.app", "isaacsim", "omni.kit.app"))
    assert all(path.read_bytes() == data for path, data in originals.items())
    patch.undo()


@pytest.fixture(scope="session")
def profiles(tmp_path_factory):
    original = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    candidate = copy.deepcopy(original)
    for phase in PHASES:
        assert original["residual"]["phase_caps_full12"][phase][8:] == ([.3] * 4 if phase < "P06" else [.6] * 4)
        if phase >= "P06":
            candidate["residual"]["phase_caps_full12"][phase][8:10] = [1.2, 1.2]
    folder = tmp_path_factory.mktemp("profiles")
    old, new = folder / "original_v3.yaml", folder / "candidate.yaml"
    old.write_bytes(PROFILE.read_bytes())
    new.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
    return old, new, original, candidate


def one(index, value):
    vector = list(ZERO)
    vector[index] = value
    return tuple(vector)


def project(projector, raw, *, phase="P06", nominal=ZERO, previous=ZERO):
    return projector.project(raw, state_id=phase, nominal_action_full12=nominal,
        reference_action_full12=nominal, reference_delta_full12=ZERO,
        previous_projected_residual_full12=previous, runtime_action_mask_full12=(1,) * 12,
        dt_s=1/120.)


def settle(projector, raw, *, phase="P06", nominal=ZERO, count=200):
    previous = ZERO
    for _ in range(count):
        out = project(projector, raw, phase=phase, nominal=nominal, previous=previous)
        assert all(abs(a-b) <= .015 + 2e-14 for a, b in zip(out.safe_projected_residual_full12[8:], previous[8:]))
        previous = out.safe_projected_residual_full12
    return out


def actual_native(out, *, phase="P06", nominal=ZERO, controller=ZERO):
    adapter = _adapter()  # Real mapper/adapter, CPU tensors and reversed joint IDs.
    mapper = adapter.servo_target_mapper.advance
    calls = []
    def counted(*args, **kwargs):
        calls.append(args)
        return mapper(*args, **kwargs)
    adapter.servo_target_mapper.advance = counted
    actuation = plan(out.safe_projected_residual_full12, controller=controller, nominal=nominal)
    previous = tuple(adapter._final_drive_servo_deg.values())
    ack = dispatch(adapter, actuation, 1)
    after = tuple(adapter._final_drive_servo_deg.values())
    targets = adapter.robot._joint_vel_target_sim.clone()
    events = list(adapter.robot.events)
    audit = build_actuator_target_effect_audit(adapter=adapter, actuation=actuation, raw_ack=ack,
        previous_final_drive_servo_deg=previous, source_phase_id=phase,
        policy_request=actuator_target_audit_request(phase, out.raw_residual_full12, (1,) * 12))
    assert len(calls) == adapter.write_count == ack["articulation_writes_this_call"] == 1
    assert adapter.robot.events == events == ["position.setter", "velocity.setter", "dispatch"]
    assert tuple(adapter._final_drive_servo_deg.values()) == after
    assert torch.equal(targets, adapter.robot._joint_vel_target_sim)
    assert audit["verified"] and audit["actual_mapping_matches_dispatch"] and audit["setter_dispatch_targets_equal"]
    expected = tuple(max(-LIMIT, min(LIMIT, n+c+r)) for n, c, r in
                     zip(nominal[8:], controller[8:], out.safe_projected_residual_full12[8:]))
    native = torch.tensor([s*x for s, x in zip(SIGN, expected)], dtype=torch.float32).tolist()
    assert audit["actual_native_targets"]["wheel_velocity_rad_s"] == native
    assert ack["bounded_controller_bias_requested_full12"] == list(controller)
    assert ack["applied_full12"] == list(nominal)
    assert audit["target_dtype"] == "torch.float32"
    return audit, ack


def test_candidate_changes_only_sixteen_front_cap_entries(profiles):
    _, _, old, new = profiles
    restored = copy.deepcopy(new)
    changed = []
    for phase in PHASES:
        for index, (a, b) in enumerate(zip(old["residual"]["phase_caps_full12"][phase], new["residual"]["phase_caps_full12"][phase])):
            if a != b:
                changed.append((phase, index, a, b))
                restored["residual"]["phase_caps_full12"][phase][index] = a
    assert changed == [(phase, index, .6, 1.2) for phase in PHASES[5:] for index in (8, 9)]
    assert restored == old
    assert new["residual"]["wheel_rate_rad_s2"] == 1.8


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("raw_value", [-.7, .7])
def test_all_phase_exact_expression_and_unchanged_channels(profiles, phase, raw_value):
    old_path, new_path, old, new = profiles
    raw = (raw_value,) * 12
    a, b = [project(build_semantic_projector(path), raw, phase=phase) for path in (old_path, new_path)]
    for out, config in ((a, old), (b, new)):
        expected = tuple(math.tanh(raw_value)*cap for cap in config["residual"]["phase_caps_full12"][phase])
        assert out.scaled_residual_full12 == pytest.approx(expected, rel=1e-14, abs=1e-14)
        assert out.effective_action_mask_full12 == (1,) * 12
    assert b.scaled_residual_full12[:8] == a.scaled_residual_full12[:8]
    assert b.scaled_residual_full12[10:] == a.scaled_residual_full12[10:]
    factor = 1. if phase < "P06" else 2.
    assert b.scaled_residual_full12[8:10] == tuple(factor*x for x in a.scaled_residual_full12[8:10])
    assert a.safe_projected_residual_full12 == b.safe_projected_residual_full12  # First tick is slew-limited, not proof of equal cap.


@pytest.mark.parametrize("phase,index,nominal_value", REQUIREMENTS)
@pytest.mark.parametrize("desired_net", [-.02, 0., .02])
def test_old_cannot_new_can_cancel_with_finite_raw_and_signed_margin(profiles, phase, index, nominal_value, desired_net):
    old_path, new_path, _, _ = profiles
    nominal = one(index, nominal_value)
    required = desired_net - nominal_value
    assert abs(required) < 1.2
    raw = one(index, math.atanh(required/1.2))
    assert math.isfinite(raw[index]) and abs(raw[index]) < 2.
    new = settle(build_semantic_projector(new_path), raw, phase=phase, nominal=nominal)
    assert new.safe_projected_residual_full12[index] == pytest.approx(required, rel=0, abs=5e-14)
    assert new.applied_action_full12[index] == pytest.approx(desired_net, rel=0, abs=5e-14)
    audit, _ = actual_native(new, phase=phase, nominal=nominal)
    assert audit["changed_channels_full12"][index]
    assert audit["actual_native_targets"]["wheel_velocity_rad_s"][index-8] == pytest.approx(SIGN[index-8]*desired_net, rel=0, abs=2e-9)
    # Best opposing old request, not merely the same smaller raw as the new one.
    old = settle(build_semantic_projector(old_path), one(index, math.copysign(20., -nominal_value)), phase=phase, nominal=nominal)
    nearest = math.copysign(abs(nominal_value)-.6, nominal_value)
    assert old.applied_action_full12[index] == pytest.approx(nearest, rel=0, abs=5e-14)
    assert abs(old.applied_action_full12[index]-desired_net) >= abs(nominal_value)-.6-.02-5e-14
    assert abs(old.applied_action_full12[index]) >= .03-5e-14
    actual_native(old, phase=phase, nominal=nominal)


@pytest.mark.parametrize("index,nominal_value", [(8, -1.07), (8, 1.09)])
@pytest.mark.parametrize("controller_value", [0., .05])
def test_same_direction_large_request_keeps_absolute_intersection(profiles, index, nominal_value, controller_value):
    direction = math.copysign(1., nominal_value)
    nominal = one(index, nominal_value)
    out = settle(build_semantic_projector(profiles[1]), one(index, direction*20.), nominal=nominal)
    assert out.scaled_residual_full12[index] == pytest.approx(direction*1.2, rel=0, abs=1e-14)
    assert abs(nominal_value)+1.2 > LIMIT
    assert out.safe_projected_residual_full12[index] == pytest.approx(direction*(LIMIT-abs(nominal_value)), abs=1e-14)
    assert out.applied_action_full12[index] == direction*LIMIT
    audit, ack = actual_native(out, nominal=nominal, controller=one(index, direction*controller_value))
    assert ack["drive_target_full12"][index] == direction*LIMIT
    assert abs(audit["actual_native_targets"]["wheel_velocity_rad_s"][index-8]) == torch.tensor(LIMIT, dtype=torch.float32).item()


@pytest.mark.parametrize("direction", [-1., 1.])
def test_per_tick_slew_grows_to_new_front_bound_then_decays_without_zero_jump(profiles, direction):
    projector = build_semantic_projector(profiles[1])
    previous = ZERO
    caps = (1.2, 1.2, .6, .6)
    for tick in range(1, 101):
        out = project(projector, ZERO[:8]+(20.*direction,)*4, previous=previous)
        expected = tuple(direction*min(cap, tick*.015) for cap in caps)
        assert out.safe_projected_residual_full12[8:] == pytest.approx(expected, rel=0, abs=2e-13)
        assert all(abs(a-b) <= .015+2e-14 for a, b in zip(out.safe_projected_residual_full12[8:], previous[8:]))
        previous = out.safe_projected_residual_full12
    decay = project(projector, ZERO, previous=previous)
    assert not decay.zero_residual_fast_path
    assert decay.safe_projected_residual_full12[8:] == pytest.approx(tuple(direction*(cap-.015) for cap in caps), abs=2e-13)
    assert all(decay.safe_projected_residual_full12[8:])


@pytest.mark.parametrize("before,after", list(zip(PHASES[4:-1], PHASES[5:])))
@pytest.mark.parametrize("direction", [-1., 1.])
def test_p05_to_p06_and_p07_through_p13_bridge_preserves_feasible_history(profiles, before, after, direction):
    bridge = PhaseTransitionBridge(build_semantic_projector(profiles[1]))
    wheels = (.25,)*4 if before == "P05" else (1.0, .9, .4, .3)
    history = tuple(direction*x for x in ((.125,)*8+wheels))
    bridge.reset(state_id=before, projected_residual_full12=history, applied_action_full12=history)
    kwargs = dict(state_id=after, nominal_action_full12=ZERO, reference_action_full12=ZERO,
                  reference_delta_full12=ZERO, dt_s=1/120.)
    held = bridge.project_tick(ZERO, **kwargs)
    assert held.transition_metric.handoff_hold_used
    assert held.projection.safe_projected_residual_full12 == pytest.approx(history, rel=0, abs=2e-13)
    assert held.transition_metric.phase_scale_clipped_channel_indices == ()
    assert held.transition_metric.forbidden_channel_indices == ()
    assert held.transition_metric.applied_action_jump_full12 == pytest.approx(ZERO, rel=0, abs=2e-13)
    decayed = bridge.project_tick(ZERO, **kwargs)
    expected = tuple(direction*(x-.015) for x in wheels)
    assert decayed.projection.safe_projected_residual_full12[8:] == pytest.approx(expected, rel=0, abs=2e-13)
    assert all(decayed.projection.safe_projected_residual_full12[8:])
    assert bridge.state_id == after


@pytest.mark.parametrize("which", [0, 1])
def test_original_point_three_cancel_preserved_and_candidate_uses_per_wheel_raw(profiles, which):
    nominal = ZERO[:8]+(.3,)*4
    caps = (.6,)*4 if which == 0 else (1.2, 1.2, .6, .6)
    raw = ZERO[:8]+tuple(-math.atanh(.3/cap) for cap in caps)
    if which == 0:
        assert raw[8:] == (-math.atanh(.5),)*4  # Original test condition remains exact.
    out = settle(build_semantic_projector(profiles[which]), raw, nominal=nominal)
    assert out.safe_projected_residual_full12[8:] == pytest.approx((-.3,)*4, rel=0, abs=5e-14)
    audit, _ = actual_native(out, nominal=nominal)
    assert audit["actual_native_targets"]["wheel_velocity_rad_s"] == pytest.approx((0.,)*4, rel=0, abs=5e-14)
    assert all(audit["changed_channels_full12"][8:])


def test_old_uniform_cancel_raw_does_not_silently_keep_same_new_physical_output(profiles):
    nominal = ZERO[:8]+(.3,)*4
    raw = ZERO[:8]+(-math.atanh(.5),)*4
    old, new = [settle(build_semantic_projector(path), raw, nominal=nominal) for path in profiles[:2]]
    assert old.applied_action_full12[8:] == pytest.approx((0.,)*4, rel=0, abs=5e-14)
    assert new.applied_action_full12[8:] == pytest.approx((-.3, -.3, 0., 0.), rel=0, abs=5e-14)
    a, _ = actual_native(old, nominal=nominal)
    b, _ = actual_native(new, nominal=nominal)
    assert a["actual_native_targets"] != b["actual_native_targets"]


@pytest.mark.parametrize("raw_value", [-.9, -.2, .2, .9])
def test_same_raw_new_front_physical_output_changes_after_slew_rear_does_not(profiles, raw_value):
    raw = ZERO[:8]+(raw_value,)*4
    old, new = [settle(build_semantic_projector(path), raw) for path in profiles[:2]]
    assert new.applied_action_full12[8:10] == tuple(2*x for x in old.applied_action_full12[8:10])
    assert new.applied_action_full12[10:] == old.applied_action_full12[10:]
    a, _ = actual_native(old)
    b, _ = actual_native(new)
    assert b["actual_native_targets"]["wheel_velocity_rad_s"][:2] != a["actual_native_targets"]["wheel_velocity_rad_s"][:2]
    assert b["actual_native_targets"]["wheel_velocity_rad_s"][2:] == a["actual_native_targets"]["wheel_velocity_rad_s"][2:]


@pytest.mark.parametrize("direction", [-1., 1.])
def test_existing_324_history_scale_represents_new_caps_without_clipping_or_overflow(profiles, direction):
    schema = load_semantic_observation_schema(SCHEMA)
    assert schema.dimension == 324 and schema.clip == 20.
    builder = SemanticObservationBuilder(schema)
    residual = tuple(direction*x for x in (24., 36., 24., 36., 24., 36., 24., 36., 1.2, 1.2, .6, .6))
    applied = ZERO[:8]+(direction*LIMIT,)*4
    history = dict.fromkeys(HISTORY_GROUPS, ZERO)
    for key in ("previous_residual_full12", "previous_previous_residual_full12"):
        history[key] = residual
    for key in ("previous_applied_full12", "previous_previous_applied_full12"):
        history[key] = applied
    built = builder.build(_frame(stage="P13"), history)
    encoded = schema.encode(built.groups)
    assert len(encoded) == 324 and torch.isfinite(torch.tensor(encoded, dtype=torch.float32)).all()
    start = 0
    for spec in schema.groups:
        end = start+spec["size"]
        if spec["name"] in ("previous_residual_full12", "previous_previous_residual_full12"):
            assert spec["scale"][8:] == [.12]*4  # Do not silently renormalize old actor input.
            assert encoded[start+8:end] == pytest.approx(tuple(direction*x for x in (10.,10.,5.,5.)), rel=0, abs=2e-14)
            assert max(abs(x) for x in encoded[start:end]) == pytest.approx(10.)
            assert max(abs(x) for x in encoded[start:end]) < schema.clip
            assert tuple(v*s for v, s in zip(encoded[start:end], spec["scale"])) == pytest.approx(residual, rel=0, abs=2e-14)
        if spec["name"] in ("previous_applied_full12", "previous_previous_applied_full12"):
            assert all(abs(x) < 1. for x in encoded[start+8:end])
        start = end


@pytest.mark.parametrize("fault", ["short", "zero", "nan", "infinity", "shrink", "over_span", "order"])
def test_candidate_loader_keeps_existing_bad_configuration_rejections(profiles, tmp_path, fault):
    config = copy.deepcopy(profiles[3])
    rows = config["residual"]["phase_caps_full12"]
    if fault == "short": rows["P06"].pop()
    elif fault == "zero": rows["P06"][8] = 0.
    elif fault == "nan": rows["P06"][8] = math.nan
    elif fault == "infinity": rows["P06"][8] = math.inf
    elif fault == "shrink": rows["P07"][8] = .6
    elif fault == "over_span": rows["P06"][8] = 2*LIMIT+.1
    elif fault == "order": config["residual"]["phase_caps_full12"] = dict(reversed(list(rows.items())))
    path = tmp_path/f"bad_{fault}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        build_semantic_projector(path)

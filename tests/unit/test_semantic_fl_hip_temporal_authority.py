"""FL-hip advisory/request authority, using the production CPU control chain.

These tests do not simulate an articulation. The tensor-buffer fixture retains
the real mapper, tracking feedback, post-mapper headroom, final slew, signs and
float32 dispatch audit. Nominal + requested residual is NOT a native target or
a guarantee of physical cancellation. No observation-layout change is needed.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path

import pytest
import torch
import yaml

from test_semantic_residual_adapter import plan
from test_semantic_transfer_residual_authority import adapter, full, project, step
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
from wlr50_clean.ppo.actuator_target_effect import (
    ActuatorTargetEffectError, actuator_target_audit_request,
    build_actuator_target_effect_audit,
)
from wlr50_clean.ppo.phase_action_masks_v2 import PhaseTransitionBridge
from wlr50_clean.ppo.semantic_backend import build_semantic_projector
from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider, load_task_spec
from wlr50_clean.ppo.semantic_tracking_reference import MODE, capture_tracking_reference_context
from wlr50_clean.reference.motion_contract import load_motion_contract


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/ppo_semantic_v3/stage_task_spec.yaml"
PROFILE = ROOT / "configs/ppo_semantic_v3/execution_profile.yaml"
KEY = "phase_servo_rate_overrides_deg_s"
PHASES = tuple(f"P{i:02}" for i in range(1, 14))
LATE = PHASES[5:]
ZERO = (0.,) * 12
DT = 1 / 120.


@pytest.fixture(scope="module", autouse=True)
def cpu_only():
    patch = pytest.MonkeyPatch()
    patch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert not torch.cuda.is_initialized()

    def forbidden(*args, **kwargs):
        raise AssertionError("target-buffer test must not initialize CUDA")

    patch.setattr(torch.cuda, "_lazy_init", forbidden)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)
        patch.undo()


@pytest.fixture(scope="module")
def contract():
    return load_motion_contract(ROOT / "configs/recording_motion_contract.json")


def task_spec(candidate):
    spec = load_task_spec(SPEC)
    assert spec["nominal"]["servo_handoff_rate_deg_s"] == 150.
    spec["nominal"].pop(KEY, None)
    if candidate:
        spec["nominal"][KEY] = {p: {"front_left_hip": 60.} for p in LATE}
    return spec


@pytest.fixture
def profiles(tmp_path):
    base = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    paths = {}
    for candidate, cap in ((False, 24.), (True, 32.)):
        cfg = deepcopy(base)
        for phase in LATE:
            cfg["residual"]["phase_caps_full12"][phase][0] = cap
        path = tmp_path / ("candidate.yaml" if candidate else "legacy.yaml")
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        paths[candidate] = path
    return paths


def provider(contract, candidate, phase="P07", seed=None):
    return NominalMotionProvider.from_handoff(
        contract, spec=task_spec(candidate), stage_id=phase,
        nominal_full12=contract.phase(phase).start_full12 if seed is None else seed,
        tracking_servo_names=(),
    )


def context(phase, nominal=ZERO):
    return dict(state_id=phase, nominal_action_full12=nominal,
                reference_action_full12=nominal, reference_delta_full12=ZERO,
                runtime_action_mask_full12=(1,) * 12, dt_s=DT)


def timeline(contract, profile, *, candidate, raw_value, ticks=80):
    """Known short handoff schedule, not a replay or task-success fixture."""
    nominal_provider = provider(contract, candidate)
    bridge = PhaseTransitionBridge(build_semantic_projector(profile))
    bridge.reset(state_id="P07", projected_residual_full12=ZERO)
    rows, previous_nominal, previous_request = [], nominal_provider.nominal_full12, ZERO
    held_raw = None
    for tick in range(1, ticks + 1):
        phase = "P07" if tick <= 8 else "P08" if tick <= 24 else "P09"
        # The same finite policy sample is available only at 15 Hz; its request
        # is filtered at every 120 Hz tick. Phase boundaries do not resample it.
        if (tick - 1) % 8 == 0:
            held_raw = full(raw_value, 0)
        nominal = nominal_provider.evaluate({"stage_id": phase, "termination_reason": None})
        result = bridge.project_tick(held_raw, **context(phase, nominal))
        request = result.projection.safe_projected_residual_full12
        metric = result.transition_metric
        held = metric is not None and metric.handoff_hold_used
        assert abs(request[0] - previous_request[0]) <= .5 + 1.e-12
        nominal_limit = .5 if candidate else 1.25
        assert abs(nominal[0] - previous_nominal[0]) <= nominal_limit + 1.e-12
        if held:
            assert request == previous_request
            assert nominal == previous_nominal
        rows.append(dict(tick=tick, phase=phase, nominal=nominal, request=request,
                         held=held, tracking=nominal_provider.tracking_servo_names))
        previous_nominal, previous_request = nominal, request
    return rows, nominal_provider


def audited_step(a, request, nominal, tick, *, raw=ZERO, controller=ZERO, tracking=()):
    """Reuse the real dispatch helper and retain the pre-dispatch audit state."""
    previous = tuple(a._final_drive_servo_deg.values())
    reference = capture_tracking_reference_context(a, physics_tick=tick, bootstrap_physics_tick=1)
    reference["source_tracking_servo_names"] = list(tracking)
    ack = step(a, request, nominal, tick, raw=raw, controller=controller, tracking=tracking)
    inputs = dict(adapter=a, actuation=plan(request, nominal=nominal, controller=controller),
                  raw_ack=ack, previous_final_drive_servo_deg=previous,
                  source_phase_id="P07", policy_request=actuator_target_audit_request("P07", raw, (1,) * 12),
                  policy_headroom_mode=HEADROOM_MODE, tracking_reference_mode=MODE,
                  tracking_reference_context=reference)
    counts = a.write_count, a.servo_target_mapper.feedback_tick, len(a.robot.events)
    effect = build_actuator_target_effect_audit(**inputs)
    assert (a.write_count, a.servo_target_mapper.feedback_tick, len(a.robot.events)) == counts
    assert effect["verified"] and effect["actual_mapping_matches_dispatch"]
    assert effect["setter_dispatch_targets_equal"] and effect["target_dtype"] == "torch.float32"
    return ack, effect, inputs


def test_published_config_changes_only_late_fl_hip_rate_and_cap():
    spec = load_task_spec(SPEC)
    cfg = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    assert spec["nominal"]["servo_handoff_rate_deg_s"] == 150.
    assert spec["nominal"][KEY] == {p: {"front_left_hip": 60.} for p in LATE}
    for phase in PHASES:
        expected = ([18, 24, 18, 24, 12, 18, 12, 18, .6, .6, .6, .6]
                    if phase < "P06" else [32, 36, 24, 112, 24, 36, 24, 36, 1.2, 1.2, .6, .6])
        assert cfg["residual"]["phase_caps_full12"][phase] == expected
    assert (cfg["physics_hz"], cfg["decision_hz"]) == (120, 15)
    assert cfg["residual"]["servo_rate_deg_s"] == 60.
    assert cfg["residual"]["wheel_rate_rad_s2"] == 1.8
    assert spec["nominal"]["wheel_handoff_rate_rad_s2"] == 3.


@pytest.mark.parametrize("phase", PHASES)
def test_provider_selects_only_current_phase_fl_override_with_legacy_absence(contract, phase):
    authored = [w.full12 for w in contract.phase(phase).waypoints if w.time_s == 0.][-1]
    seed = list(authored)
    seed[0] -= 12.
    legacy, candidate = provider(contract, False, phase, seed), provider(contract, True, phase, seed)
    old, new = legacy.evaluate(phase), candidate.evaluate(phase)
    assert old[0] - seed[0] == pytest.approx(1.25)
    assert new[0] - seed[0] == pytest.approx(.5 if phase in LATE else 1.25)
    assert new[1:] == old[1:]
    assert legacy.servo_rate_limit_deg_s == candidate.servo_rate_limit_deg_s == 150.
    if phase in LATE:
        assert candidate._servo_rate_overrides[phase] == (60.,) + (150.,) * 7


@pytest.mark.parametrize("overrides", [
    {"P99": {"front_left_hip": 60.}},
    {"P07": {"front_left_ankle": 60.}},
    {"P07": {"not_a_servo": 60.}},
    {"P07": {"front_left_hip": True}},
    {"P07": {"front_left_hip": float("nan")}},
    {"P07": {"front_left_hip": float("inf")}},
    {"P07": {"front_left_hip": 0.}},
    {"P07": {"front_left_hip": -1.}},
    {"P07": {"front_left_hip": 150.0001}},
], ids=["unknown_phase", "wheel_name", "unknown_servo", "bool", "nan", "inf", "zero", "negative", "above_default"])
def test_invalid_rate_overrides_fail_at_provider_construction(contract, overrides):
    spec = task_spec(False)
    spec["nominal"][KEY] = overrides
    with pytest.raises(ValueError):
        NominalMotionProvider(contract, spec=spec)


def test_validated_rate_cache_is_detached_from_input_mapping(contract):
    spec = task_spec(True)
    nominal = NominalMotionProvider(contract, spec=spec)
    spec["nominal"][KEY]["P07"]["front_left_hip"] = 1.
    assert nominal._servo_rate_overrides["P07"] == (60.,) + (150.,) * 7


def test_old_150_24_has_temporal_and_endpoint_expression_gap(contract, profiles):
    rows, _ = timeline(contract, profiles[False], candidate=False, raw_value=-20.)
    assert rows[-1]["nominal"][0] == pytest.approx(49.2)
    assert rows[-1]["request"][0] == pytest.approx(-24.)
    gaps = [r["nominal"][0] + r["request"][0] - 22.8 for r in rows]
    assert max(gaps) > 10.
    assert gaps[-1] == pytest.approx(2.4)
    assert [r["tick"] for r in rows if r["held"]] == [9, 25]
    # This is an upstream expression counterexample, not a native/physics claim.
    assert all(abs(r["request"][0]) <= 24. for r in rows)


@pytest.mark.parametrize("net_change", [0., -1.])
def test_new_60_32_finite_latent_can_cancel_or_oppose_bounded_source(contract, profiles, net_change):
    requested = net_change - 26.4
    raw = math.atanh(requested / 32.)
    assert math.isfinite(raw) and abs(requested) < 32.
    rows, _ = timeline(contract, profiles[True], candidate=True, raw_value=raw)
    assert rows[-1]["nominal"][0] == pytest.approx(49.2)
    assert rows[-1]["request"][0] == pytest.approx(requested)
    assert rows[-1]["nominal"][0] + rows[-1]["request"][0] - 22.8 == pytest.approx(net_change)
    holds = sum(r["held"] for r in rows)
    assert holds == 2
    active_ticks = math.ceil(abs(requested) / .5)
    assert rows[active_ticks + holds - 1]["request"][0] == pytest.approx(requested)
    old, _ = timeline(contract, profiles[False], candidate=False, raw_value=-20.)
    assert max(r["nominal"][0] + r["request"][0] - 22.8 for r in rows) < max(
        r["nominal"][0] + r["request"][0] - 22.8 for r in old)
    assert [r["nominal"][1:] for r in rows] == [r["nominal"][1:] for r in old]


def test_raw_zero_still_has_finite_fl_nominal_motion_and_source_layer_continuity(contract, profiles):
    rows, nominal = timeline(contract, profiles[True], candidate=True, raw_value=0.)
    assert all(r["request"] == ZERO for r in rows)
    assert rows[-1]["nominal"][0] - 22.8 == pytest.approx(26.4)
    assert [layer["stage"] for layer in nominal._continuous_layers] == ["P07", "P08", "P09"]
    assert [layer["ticks"] for layer in nominal._continuous_layers] == [80, 72, 56]
    # P07 suffix did not fabricate an earlier P06 layer/teacher command queue.
    assert not nominal.nominal_suggestion_diagnostics["p06_wheel_tail"]["layer_present"]


def test_later_real_wheel_owner_and_other_channel_rates_are_unchanged(contract, profiles):
    old, _ = timeline(contract, profiles[False], candidate=False, raw_value=0.)
    new, _ = timeline(contract, profiles[True], candidate=True, raw_value=0.)
    assert all(a["nominal"][1:] == b["nominal"][1:] for a, b in zip(old, new))
    # Actual finite P07 owner includes the negative FR wheel suggestion. This
    # revision must neither freeze all channels nor synthesize a +.3 takeover.
    assert min(r["nominal"][9] for r in new) < 0.
    previous = contract.phase("P07").start_full12
    for row in new:
        for i in range(1, 12):
            assert abs(row["nominal"][i] - previous[i]) <= (1.25 if i < 8 else .025) + 1.e-12
        previous = row["nominal"]


@pytest.mark.parametrize("source,target", list(zip(PHASES[4:-1], PHASES[5:])))
def test_bridge_preserves_fl_request_across_every_late_handoff(profiles, source, target):
    bridge = PhaseTransitionBridge(build_semantic_projector(profiles[True]))
    history = full(-16. if source == "P05" else -26.4, 0)
    bridge.reset(state_id=source, projected_residual_full12=history, applied_action_full12=history)
    first = bridge.project_tick(ZERO, **context(target))
    assert first.transition_metric.handoff_hold_used
    assert first.transition_metric.phase_scale_clipped_channel_indices == ()
    assert first.transition_metric.carried_projected_residual_full12 == history
    # Holding goes through the production inverse-tanh/tanh pair; -26.4 has
    # a one-ULP round-trip difference. This is not bitwise history reset.
    held = first.projection.safe_projected_residual_full12
    assert held[0] == pytest.approx(history[0], rel=0., abs=2 * math.ulp(history[0]))
    assert held[1:] == history[1:]
    second = bridge.project_tick(ZERO, **context(target))
    assert second.projection.safe_projected_residual_full12[0] == pytest.approx(history[0] + .5)


@pytest.mark.parametrize("phase", ["P05", "P06", "P07", "P13"])
def test_projector_preserves_all_other_channels_and_both_request_slews(profiles, phase):
    old, new = (build_semantic_projector(profiles[x]) for x in (False, True))
    previous_old = previous_new = ZERO
    for _ in range(80):
        a = project(old, (.7,) * 12, previous_old, phase=phase, nominal=ZERO)
        b = project(new, (.7,) * 12, previous_new, phase=phase, nominal=ZERO)
        assert a.safe_projected_residual_full12[1:] == b.safe_projected_residual_full12[1:]
        for i, (before, after) in enumerate(zip(previous_new, b.safe_projected_residual_full12)):
            assert abs(after - before) <= (.5 if i < 8 else .015) + 1.e-12
        previous_old, previous_new = a.safe_projected_residual_full12, b.safe_projected_residual_full12
    if phase == "P05":
        assert previous_old == previous_new
    else:
        assert previous_new[0] == pytest.approx(math.tanh(.7) * 32.)
        assert previous_old[0] == pytest.approx(math.tanh(.7) * 24.)


def test_provider_projector_bridge_targets_reach_real_float32_audit(contract, profiles):
    raw = math.atanh(-26.4 / 32.)
    rows, _ = timeline(contract, profiles[True], candidate=True, raw_value=raw)
    a = adapter()
    measured = a.robot.data.joint_pos.clone()
    for tick, row in enumerate(rows, 1):
        step(a, row["request"], row["nominal"], tick, raw=full(raw, 0), phase=row["phase"],
             tracking=row["tracking"], audit=tick in (8, 9, 24, 25, 55, 80))
    assert a.write_count == a.servo_target_mapper.feedback_tick == 81  # Includes one bootstrap write.
    assert torch.equal(measured, a.robot.data.joint_pos)
    assert "update" not in a.robot.events  # Target writes, never a physics advance.


def test_fl_mapper_feedback_and_controller_bias_remain_separate_from_request(profiles):
    a, nominal, controller = adapter(), full(49.2, 0), full(3., 0)
    for tick in range(1, 101):
        ack = step(a, ZERO, nominal, tick, controller=controller, tracking=(SERVO_ORDER[0],))
    assert ack["servo_tracking_compensation_deg"][0] == 10.
    raw, previous = full(math.atanh(-26.4 / 32.), 0), ZERO
    projector = build_semantic_projector(profiles[True])
    for tick in range(101, 161):
        request = project(projector, raw, previous, phase="P07", nominal=nominal).safe_projected_residual_full12
        ack, effect, _ = audited_step(a, request, nominal, tick, raw=raw, controller=controller)
        previous = request
    assert ack["native_drive_target_full12"][0] == pytest.approx(59.2)
    assert ack["policy_headroom_evidence"]["effective_policy_residual_full12"][0] == pytest.approx(-26.4)
    assert ack["drive_target_full12"][0] == pytest.approx(35.8)
    assert effect["changed_target_channel_count"] == 1
    native = a.robot._joint_pos_target_sim[0, a.joint_map.servo_ids[0]].item()
    assert native == torch.tensor(.75 + math.radians(35.8), dtype=torch.float32).item()


def test_larger_request_does_not_relax_post_mapper_reserve_or_final_slew():
    a, nominal, controller = adapter(), full(130., 0), full(3., 0)
    for tick in range(1, 141):
        step(a, ZERO, nominal, tick, controller=controller)
    outward, no_effect, _ = audited_step(a, full(32., 0), nominal, 141, controller=controller)
    assert servo_limits_deg(SERVO_ORDER[0]) == (-135., 135.)
    assert outward["policy_headroom_evidence"]["servo_safety_limits_deg"][0] == [-133., 133.]
    assert outward["policy_headroom_evidence"]["effective_policy_residual_full12"][0] == 0.
    assert no_effect["changed_target_channel_count"] == 0
    inward, _, _ = audited_step(a, full(-32., 0), nominal, 142, controller=controller)
    assert inward["policy_headroom_evidence"]["effective_policy_residual_full12"][0] == -32.
    assert inward["drive_target_full12"][0] == pytest.approx(131.75)


@pytest.mark.parametrize("requested_value", [0., -0., 1.e-12, -1.e-12])
def test_float32_zero_and_subquantum_request_are_not_claimed_as_target_effect(requested_value):
    _, effect, _ = audited_step(adapter(), full(requested_value, 0), ZERO, 1)
    assert effect["changed_target_channel_count"] == 0
    assert effect["actual_native_targets"] == effect["counterfactual_native_targets"]


def test_final_slew_can_hide_request_even_when_headroom_accepts_it():
    ack, effect, _ = audited_step(adapter(), full(.5, 0), ZERO, 1, controller=full(1.25, 0))
    assert ack["policy_headroom_evidence"]["effective_policy_residual_full12"][0] == .5
    assert ack["drive_target_full12"][0] == 1.25
    assert effect["changed_target_channel_count"] == 0


@pytest.mark.parametrize("tamper", ["dispatch", "both", "dtype", "shape", "ids", "nan"])
def test_real_dispatch_audit_stays_fail_closed_after_candidate_request(tamper):
    a = adapter()
    _, _, inputs = audited_step(a, full(.5, 0), ZERO, 1)
    joint = a.joint_map.servo_ids[0]
    if tamper in ("dispatch", "both"):
        a.robot._joint_pos_target_sim[0, joint] += .1
        if tamper == "both":
            a.robot.data.joint_pos_target[0, joint] += .1
    elif tamper == "dtype":
        a.robot._joint_pos_target_sim = a.robot._joint_pos_target_sim.to(torch.float64)
    elif tamper == "shape":
        a.robot._joint_pos_target_sim = a.robot._joint_pos_target_sim.squeeze(0)
    elif tamper == "ids":
        ids = list(a.joint_map.servo_ids)
        ids[0] = ids[1]
        a.joint_map = replace(a.joint_map, servo_ids=tuple(ids))
    else:
        a.robot._joint_pos_target_sim[0, joint] = float("nan")
    counts = a.write_count, a.servo_target_mapper.feedback_tick, len(a.robot.events)
    with pytest.raises(ActuatorTargetEffectError):
        build_actuator_target_effect_audit(**inputs)
    assert (a.write_count, a.servo_target_mapper.feedback_tick, len(a.robot.events)) == counts

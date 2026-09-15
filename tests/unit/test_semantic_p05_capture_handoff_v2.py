"""Measured capture / source-clock / native CPU regression, not task success.

The source P05 stop is genuinely emitted before the seam.  A real evaluator
earns FL Q/C/P from contiguous synthetic measurements, then the real supervisor
hands off on its next 15 Hz boundary.  No production deadline or pose is changed.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from test_semantic_all_stage_physical_acceptance import new_observation, set_leg, early_top_place
from test_semantic_supervisor import advance
from test_semantic_tracking_reference_dispatch import adapter, checked_step, cpu_only
from wlr50_clean.fsm.motion_executor import MotionExecutor
from wlr50_clean.fsm.state_spec import load_fsm_spec
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.ppo.isaac_fsm_backend import build_residual_actuation_plan
from wlr50_clean.ppo.phase_action_masks_v2 import PhaseTransitionBridge
from wlr50_clean.ppo.semantic_backend import build_semantic_projector
from wlr50_clean.ppo.semantic_supervisor import (
    NominalMotionProvider, SemanticObservationError, TaskEvaluator,
    TaskStageSupervisor, load_task_spec,
)
from wlr50_clean.reference.motion_contract import load_motion_contract

ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs/ppo_fsm_reference_p09_stable_v2"
V1 = "current_FL_capture_wheel_continuation_v1"
V2 = "current_FL_capture_wheel_continuation_to_handoff_v2"
ZERO = (0.,) * 12
ROLL = (.3,) * 4


@pytest.fixture(scope="module")
def resources():
    return (load_fsm_spec(ROOT / "configs/fsm_states.yaml"),
            load_motion_contract(ROOT / "configs/recording_motion_contract.json"))


def configured(mode=V2):
    value = load_task_spec(CFG / "stage_task_spec.yaml")
    value["nominal"]["p05_pending_capture"] = mode
    return value


def measured():
    obs = new_observation()
    obs["center_of_mass"] = dict(valid=True, position_w_m=[.30, 0., .10],
        velocity_w_m_s=[0., 0., 0.], total_mass_kg=3.,
        included_bodies=["synthetic full-body measurement"])
    for name, xy in zip(WHEEL_ORDER, ((.4, .12), (.4, -.12), (.20, .12), (.20, -.12))):
        obs["wheels"][name]["center_w_m"][:2] = xy
        obs["wheels"][name]["bottom_w_m"][:2] = xy
    return obs


def step(state, *, surface=None):
    p, sup, obs = state
    obs = advance(obs)
    if surface is not None:
        set_leg(obs, "FL", x=.53, bottom=.05 if surface == "TOP" else .075,
                surface=surface)
    task = sup.observe_and_update(obs)
    nominal = p.evaluate(task, obs)
    return (p, sup, obs), task, nominal


@pytest.fixture(scope="module")
def mature_sources(resources):
    """Each legacy/v2 mode really runs the finite source to its old zero tail."""
    fsm, contract = resources
    result = {}
    for mode in (None, V1, V2):
        spec = configured(mode)
        ev = TaskEvaluator(spec=spec)
        obs = measured(); ev.observe(obs)
        obs = early_top_place(ev, obs, "FR")
        for bottom, hip, x in ((.012, 1.5, .4), (.020, 3., .4), (.075, 4., .53)):
            obs = advance(obs)
            set_leg(obs, "FL", x=x, bottom=bottom, hip=hip)
            ev.observe(obs)
        assert ev.snapshot["history"]["active_lift"]["FL"]
        assert ev.snapshot["history"]["front_edge_crossed"]["FL"]
        assert not ev.snapshot["history"]["placed"]["FL"]
        sup = TaskStageSupervisor(CFG / "stage_task_spec.yaml", evaluator=ev, initial_stage_id="P05")
        sup.spec = deepcopy(spec)
        p = NominalMotionProvider(contract, spec=spec, fsm_spec=fsm)
        state = (p, sup, obs)
        while state[2]["physics_tick"] < 1200:
            state, task, nominal = step(state)
        assert task["stage_id"] == "P05" and task["pending_capture"]
        assert task["termination_reason"] is None
        assert p._continuous_layers[-1]["sample"].endpoint_issued
        assert p._continuous_layers[-1]["sample"].full12[8:] == (0.,) * 4
        assert nominal[8:] == ((0.,) * 4 if mode is None else ROLL)
        result[mode] = state
    return result


def clone_state(state):
    # Frozen source specifications contain mappingproxy fields; preserve those
    # immutable inputs while copying every live evaluator/provider/history state.
    p = state[0]
    immutable = (p.contract, *p.contract.phases, p._reference_fsm_spec,
                 *p._reference_fsm_spec.states)
    return deepcopy(state, {id(value): value for value in immutable})


def captured(mature_sources, *, offset=1, mode=V2):
    state = clone_state(mature_sources[mode])
    capture_tick = 1208 + offset
    while state[2]["physics_tick"] < capture_tick - 2:
        state, _, _ = step(state)
    state, first, first_nominal = step(state, surface="TOP")
    assert not first["history"]["placed"]["FL"]
    assert first["pending_capture"] and first["stage_id"] == "P05"
    assert first_nominal[8:] == ((0.,) * 4 if mode is None else ROLL)
    state, task, nominal = step(state, surface="TOP")
    assert task["history"]["placed"]["FL"]
    assert task["physical_evaluator"]["current_legs"]["FL"]["bearing_verified"]
    assert state[2]["physics_tick"] == capture_tick
    return state, task, nominal


@pytest.mark.parametrize("offset", range(1, 8))
def test_real_capture_remains_rolling_for_only_remaining_decision_ticks(mature_sources, offset):
    state, task, nominal = captured(mature_sources, offset=offset)
    p, sup, obs = state
    assert task["stage_id"] == "P05" and nominal[8:] == ROLL
    start_ticks = p._continuous_layers[-1]["ticks"]
    start_clock = p._continuous_layers[-1]["motion"]._tick_index
    count = 0
    while state[2]["physics_tick"] < 1216:
        state, task, nominal = step(state)
        count += 1
        assert nominal[8:] == ROLL
        assert task["termination_reason"] is None and not task["success"]
        assert task["stage_id"] == ("P06" if state[2]["physics_tick"] == 1216 else "P05")
    assert count == 8 - offset
    assert [layer["stage"] for layer in p._continuous_layers] == ["P05", "P06"]
    assert p._continuous_layers[0]["ticks"] == start_ticks + count
    assert p._continuous_layers[0]["motion"]._tick_index == start_clock + count
    assert p._continuous_layers[1]["ticks"] == 1
    assert p._continuous_layers[1]["sample"].tick_index == 0
    assert task["transition_evidence"][-1]["physics_tick"] == 1216
    assert task["completed_stage_ids"] == ["P05"]


def test_capture_at_decision_boundary_uses_real_p06_tick_zero_once(mature_sources):
    state, task, nominal = captured(mature_sources, offset=0)
    p = state[0]
    assert task["stage_id"] == "P06" and nominal[8:] == ROLL
    new = p._continuous_layers[-1]
    assert new["ticks"] == 1 and new["sample"].tick_index == 0
    assert any(set(group.channels).intersection(WHEEL_ORDER) for group in new["sample"].atomic_groups)
    state, _, nominal = step(state)
    assert nominal[8:] == ROLL and new["sample"].tick_index == 1
    assert not any(set(group.channels).intersection(WHEEL_ORDER) for group in new["sample"].atomic_groups)


@pytest.mark.parametrize("mode", [None, V1])
def test_legacy_modes_keep_the_original_old_stop_gap(mature_sources, mode):
    state, task, nominal = captured(mature_sources, mode=mode)
    assert task["stage_id"] == "P05" and nominal[8:] == (0.,) * 4
    while state[2]["physics_tick"] < 1216:
        state, task, nominal = step(state)
    assert task["stage_id"] == "P06" and nominal[8:] == ROLL


@pytest.mark.parametrize("change", ["air", "ground", "outside_xy", "no_top", "no_support",
    "unknown_bearing", "unknown_air", "one_other_support", "bad_gap", "beyond_approach",
    "not_placed", "no_previous_roll", "terminal", "invalid"])
def test_historical_placement_is_not_current_verified_capture_or_start_authority(mature_sources, change):
    state, task, _ = captured(mature_sources)
    p, _, obs = state
    obs = advance(obs)
    task = deepcopy(task)
    ev = task["physical_evaluator"]
    ev.update(physics_tick=obs["physics_tick"], simulation_time_s=obs["simulation_time_s"])
    leg = ev["current_legs"]["FL"]
    assert task["transfer_roles"]["FL"]["pending_capture"] is False
    if change == "air": leg["air"] = True
    elif change == "ground": leg["ground_contact"] = True
    elif change == "outside_xy": leg["within_top_xy"] = False
    elif change == "no_top": leg["top_contact"] = False
    elif change == "no_support": leg["support"] = False
    elif change == "unknown_bearing": leg["bearing_verified"] = None
    elif change == "unknown_air": leg["air"] = None
    elif change == "one_other_support": task["transfer_roles"]["FL"]["observed_support_contacts"] = ["FL", "FR"]
    elif change == "bad_gap": leg["clearance_m"] = -.016
    elif change == "beyond_approach": leg["front_distance_m"] = .10
    elif change == "not_placed": ev["history"]["placed"]["FL"] = False
    elif change == "no_previous_roll": p.nominal_full12 = p.nominal_full12[:8] + (0.,) * 4
    elif change == "terminal": task["termination_reason"] = "BODY_COLLISION"
    elif change == "invalid": ev["valid"] = False
    if change == "invalid":
        clocks = [x["motion"]._tick_index for x in p._continuous_layers]
        with pytest.raises(SemanticObservationError): p.evaluate(task, obs)
        assert [x["motion"]._tick_index for x in p._continuous_layers] == clocks
    else:
        assert p.evaluate(task, obs)[8:] == (0.,) * 4


def stop_layer(p):
    """Prepare an actual frozen P05 source immediately before its authored stop."""
    phase = p.contract.phase("P05")
    stop_tick = max(round(group.time_s * p.physics_hz) for group in phase.atomic_groups
                    if set(group.channels).intersection(WHEEL_ORDER))
    motion = MotionExecutor(physics_hz=p.physics_hz, initial_full12=phase.start_full12)
    p._start_source_motion(motion, phase)
    for _ in range(stop_tick): sample = motion.tick()
    assert sample.full12[8:] == ROLL
    return dict(stage="P05", motion=motion, last=sample.full12, touched=set(range(8, 12)),
                sample=sample, ticks=stop_tick)


@pytest.mark.parametrize("owner", ["current", "older"])
def test_fresh_wheel_stop_in_any_actual_continuous_layer_has_precedence(mature_sources, owner):
    state, _, _ = captured(mature_sources)
    p = state[0]
    fresh = stop_layer(p)
    if owner == "current": p._continuous_layers[-1] = fresh
    else: p._continuous_layers.insert(0, fresh)
    state, task, nominal = step(state)
    assert task["stage_id"] == "P05" and task["history"]["placed"]["FL"]
    assert any(set(group.channels).intersection(WHEEL_ORDER) for group in fresh["sample"].atomic_groups)
    assert fresh["sample"].full12[8:] == (0.,) * 4
    assert nominal[8:] == (0.,) * 4
    # The bridge cannot reactivate itself on the next tick after this true stop.
    state, _, nominal = step(state)
    assert nominal[8:] == (0.,) * 4


def test_missed_handoff_boundary_cannot_extend_bridge_or_restart_on_next_tick(mature_sources):
    state, task, _ = captured(mature_sources, offset=7)
    p, _, obs = state
    for tick in (1216, 1217):
        task = deepcopy(task)
        obs = advance(obs)
        ev = task["physical_evaluator"]
        ev.update(physics_tick=tick, simulation_time_s=tick / 120.)
        # Fault-injection scheduler seam: P05 remains despite an expired boundary.
        assert p.evaluate(task, obs)[8:] == (0.,) * 4


@pytest.mark.parametrize("bad", ["unknown", True, 1])
def test_unknown_capture_mode_rejected_by_provider(resources, bad):
    fsm, contract = resources
    with pytest.raises(ValueError, match="capture-wheel"):
        NominalMotionProvider(contract, spec=configured(bad), fsm_spec=fsm)


@pytest.mark.parametrize("missing", ["reference", "inheritance", "physical"])
def test_v2_requires_current_continuous_source_dependencies(resources, missing):
    fsm, contract = resources
    spec = configured()
    # Isolate capture handoff's dependency contract from other opt-in owners.
    spec["nominal"].pop("final_stop_owner", None)
    if missing == "reference": spec.pop("reference_nominal_semantics")
    elif missing == "inheritance": spec["nominal"]["continuous_channel_inheritance"] = False
    else: spec.pop("physical_acceptance_version")
    with pytest.raises(ValueError, match="capture handoff"):
        NominalMotionProvider(contract, spec=spec, fsm_spec=fsm)


def test_nonzero_residual_and_native_mapper_history_survive_actual_capture_handoff(mature_sources):
    state = clone_state(mature_sources[V2])
    p = state[0]
    native = adapter()
    bridge = PhaseTransitionBridge(build_semantic_projector(CFG / "execution_profile.yaml"))
    previous_ack = previous_projected = None
    crossed = False
    for dispatch_tick in range(1, 25):
        source_tick = state[2]["physics_tick"] + 1
        state, task, nominal = step(state, surface="TOP" if source_tick >= 1208 else None)
        phase = task["stage_id"]
        before = bridge.previous_projected_residual_full12
        raw = ((-.2,) * 12 if phase == "P06" else (.2,) * 12)
        result = bridge.project_tick(raw, state_id=phase, nominal_action_full12=nominal,
            reference_action_full12=nominal, reference_delta_full12=ZERO)
        projected = result.projection.safe_projected_residual_full12
        actuation = build_residual_actuation_plan(result.projection.applied_action_full12,
            frozen_nominal_full12=nominal, drive_feedback_bias_full12=p.normal_drive_bias_full12,
            normal_drive_bias_full12=p.normal_drive_bias_full12)
        events_before = len(native.robot.events)
        ctx, _, ack, _ = checked_step(native, actuation, dispatch_tick,
            tracking=p.tracking_servo_names, phase=phase)
        assert native.robot.events[events_before:] == ["position.setter", "velocity.setter", "dispatch"]
        assert native.servo_target_mapper.feedback_tick == dispatch_tick + 1
        assert native.write_count == dispatch_tick + 1
        assert nominal[8:] == ROLL
        if previous_ack is not None:
            assert ctx["previous_ack_physics_tick"] == dispatch_tick - 1
            assert ctx["previous_requested_full12"] == previous_ack["independent_policy_residual_requested_full12"]
        if result.transition_metric is not None:
            crossed = True
            assert (result.transition_metric.from_state_id, result.transition_metric.to_state_id) == ("P05", "P06")
            assert source_tick == 1216
            assert result.transition_metric.handoff_hold_used
            assert projected == pytest.approx(previous_projected, abs=1e-12)
            assert projected == pytest.approx(before, abs=1e-12)
            assert all(value != 0. for value in projected)
            assert result.transition_metric.forbidden_channel_indices == ()
            assert result.transition_metric.phase_scale_clipped_channel_indices == ()
            assert p._continuous_layers[-1]["ticks"] == 1
        assert task["termination_reason"] is None
        previous_ack, previous_projected = ack, projected
    assert crossed and bridge.state_id == "P06"

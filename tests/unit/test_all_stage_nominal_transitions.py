"""Pure CPU nominal ownership/authority tests; no simulated physical success."""
from copy import deepcopy
import math
from pathlib import Path

import pytest
import yaml

from test_semantic_supervisor import observation, place
from wlr50_clean.ppo.semantic_backend import build_semantic_projector
from wlr50_clean.ppo.phase_action_masks_v2 import PhaseTransitionBridge
from wlr50_clean.ppo.semantic_supervisor import (
    LEG_ORDER, PHASE_IDS, NominalMotionProvider, SemanticObservationError, TaskEvaluator,
)
from wlr50_clean.reference.motion_contract import load_motion_contract


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "configs/ppo_semantic_v3"
NEW = ROOT / "configs/ppo_all_stage_acceptance_v1"
ZERO = (0.,)*12


@pytest.fixture(scope="module")
def contract():
    return load_motion_contract(ROOT / "configs/recording_motion_contract.json")


def spec(enabled=True):
    # Isolate this nominal factor from root's concurrent acceptance/timeout work.
    result = yaml.safe_load((OLD / "stage_task_spec.yaml").read_text(encoding="utf-8"))
    if enabled:
        result["physical_acceptance_version"] = "all_stage_v1"
    return result


def provider(contract, phase="P01", enabled=True):
    return NominalMotionProvider.from_handoff(contract, spec=spec(enabled), stage_id=phase,
        nominal_full12=contract.phase(phase).start_full12, tracking_servo_names=())


def task(phase, tick, *, placed=(), distance=-.30, terminal=None):
    # Scheduler unit input, not a synthetic physical rollout or success verdict.
    legs = {leg: {"front_distance_m": distance, "within_lateral_span": True,
                  "clearance_m": .001, "within_top_xy": False, "ground_contact": True,
                  "top_contact": False, "air": False} for leg in LEG_ORDER}
    ev = {"valid": True, "physics_tick": tick, "simulation_time_s": tick/120.,
          "termination_reason": terminal, "current_legs": legs,
          "history": {"placed": {leg: leg in placed for leg in LEG_ORDER},
                      "active_lift": {leg: False for leg in LEG_ORDER}}}
    return {"stage_id": phase, "termination_reason": terminal, "physical_evaluator": ev}


def step(p, phase, tick, **kwargs):
    item = task(phase, tick, **kwargs)
    return p.evaluate(item, {"physics_tick": tick, "simulation_time_s": tick/120.})


def state(p):
    return deepcopy((p.nominal_full12, p.tracking_servo_names, p.state_id,
                     p._source_motion._tick_index, p._continuous_layers,
                     p.nominal_suggestion_diagnostics))


@pytest.mark.parametrize("phase,leg,prefix_ticks", [
    ("P02", "FR", 8), ("P05", "FL", 64), ("P09", "RR", 32), ("P12", "RL", 16),
])
def test_capture_retires_old_pair_even_its_not_yet_touched_channel(contract, phase, leg, prefix_ticks):
    p = provider(contract, phase)
    for tick in range(prefix_ticks):
        step(p, phase, tick)
    indices = (2*LEG_ORDER.index(leg), 2*LEG_ORDER.index(leg)+1)
    held = tuple(p.nominal_full12[i] for i in indices)
    end = round(contract.phase(phase).active_duration_s*120)+40
    for tick in range(prefix_ticks, end):
        command = step(p, phase, tick, placed=(leg,))
        assert tuple(command[i] for i in indices) == held
    layer = p._continuous_layers[0]
    assert set(indices) <= layer["capture_retired_servo_indices"]
    assert layer["ticks"] == end  # Source clock still advances, without old owner writes.
    assert layer["sample"].endpoint_issued
    assert tuple(layer["sample"].full12[i] for i in indices) != held
    assert not set(p.tracking_servo_names) & {
        ("front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
         "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee")[i] for i in indices}
    record = p.nominal_suggestion_diagnostics["capture_owner_hold"]["captures"][leg]
    assert record["observed_capture_tick"] == prefix_ticks
    assert record["held_nominal_servo_deg"] == held


def test_real_evaluator_placement_drives_hold_without_mutating_history(contract):
    ev = TaskEvaluator()
    obs = observation()
    ev.observe(obs)
    p = provider(contract, "P02")
    p.evaluate({"stage_id": "P02", "termination_reason": None, "physical_evaluator": ev.snapshot}, obs)
    held = p.nominal_full12[2:4]
    obs = place(ev, obs, "FR")  # Actual evaluator earns Q/C/P from complete sensor fixtures.
    before = deepcopy(ev.snapshot)
    result = p.evaluate({"stage_id": "P03", "termination_reason": None, "physical_evaluator": ev.snapshot}, obs)
    assert result[2:4] == held
    assert ev.snapshot == before and before["history"]["placed"]["FR"]


def advanced_task(phase, tick, leg, *, surface="TOP", placed=True):
    result = task(phase, tick, placed=(leg,) if placed else ())
    result["physical_evaluator"]["current_legs"][leg].update(
        within_top_xy=True, ground_contact=False, top_contact=surface == "TOP",
        air=surface == "AIR", clearance_m=-.001 if surface == "TOP" else 0.)
    return result


@pytest.mark.parametrize("phase,leg", [("P02", "FR"), ("P05", "FL"), ("P09", "RR"), ("P12", "RL")])
@pytest.mark.parametrize("surface", ["TOP", "AIR"])
def test_current_advanced_placement_suppresses_only_new_swing_pair(contract, phase, leg, surface):
    p, control = provider(contract, phase), provider(contract, phase)
    indices = (2*LEG_ORDER.index(leg), 2*LEG_ORDER.index(leg)+1)
    held = tuple(p.nominal_full12[i] for i in indices)
    end = round(contract.phase(phase).active_duration_s*120)+40
    for tick in range(end):
        item = advanced_task(phase, tick, leg, surface=surface)
        before = deepcopy(item)
        actual = p.evaluate(item)
        baseline = control.evaluate(advanced_task(phase, tick, leg, surface=surface, placed=False))
        assert item == before
        assert tuple(actual[i] for i in indices) == held
        assert all(actual[i] == baseline[i] for i in range(12) if i not in indices)
    layer = p._continuous_layers[0]
    assert layer["capture_retired_servo_indices"] == set(indices)
    assert layer["ticks"] == end and layer["sample"].endpoint_issued
    assert tuple(layer["sample"].full12[i] for i in indices) != held
    rows = p.nominal_suggestion_diagnostics["capture_owner_hold"]["suppressed_new_swing_layers"]
    assert len(rows) == 1 and rows[0]["leg"] == leg and rows[0]["source_physics_tick"] == 0
    assert rows[0]["held_nominal_servo_deg"] == held


@pytest.mark.parametrize("case", ["ground", "outside", "below_platform_air", "wall", "not_placed", "legacy"])
def test_history_or_invalid_current_region_cannot_suppress_new_swing(contract, case):
    p = provider(contract, "P12", enabled=case != "legacy")
    start = p.nominal_full12[4:6]
    for tick in range(64):
        item = advanced_task("P12", tick, "RL", surface="AIR", placed=case != "not_placed")
        current = item["physical_evaluator"]["current_legs"]["RL"]
        if case == "ground": current.update(ground_contact=True, air=False)
        elif case == "outside": current["within_top_xy"] = False
        elif case == "below_platform_air": current["clearance_m"] = -.001
        elif case == "wall": current.update(air=False, top_contact=False)
        result = p.evaluate(item)
    assert result[4:6] != start
    assert not p._continuous_layers[0].get("capture_retired_servo_indices")


@pytest.mark.parametrize("swing,target,leg", [("P02", "P11", "FR"), ("P05", "P07", "FL"), ("P12", "P13", "RL")])
def test_suppressed_swing_does_not_block_later_receiving_or_recovery_owner(contract, swing, target, leg):
    p = provider(contract, swing)
    for tick in range(16):
        p.evaluate(advanced_task(swing, tick, leg))
    indices = (2*LEG_ORDER.index(leg), 2*LEG_ORDER.index(leg)+1)
    held = tuple(p.nominal_full12[i] for i in indices)
    first = p.evaluate(advanced_task(target, 16, leg))
    assert tuple(first[i] for i in indices) == held  # Ordinary handoff is still held once.
    changed = False
    end = round(contract.phase(target).active_duration_s*120)+100
    for tick in range(17, 17+end):
        actual = p.evaluate(advanced_task(target, tick, leg))
        changed |= tuple(actual[i] for i in indices) != held
    assert changed and not p._continuous_layers[-1].get("capture_retired_servo_indices")


@pytest.mark.parametrize("bad", ["missing_current", "integer_flag", "nan_clearance", "bool_clearance"])
def test_new_captured_swing_needs_current_evidence_before_source_clock(contract, bad):
    p = provider(contract, "P09")
    item = advanced_task("P09", 0, "RR")
    current = item["physical_evaluator"]["current_legs"]["RR"]
    if bad == "missing_current": item["physical_evaluator"].pop("current_legs")
    elif bad == "integer_flag": current["within_top_xy"] = 1
    elif bad == "nan_clearance": current["clearance_m"] = math.nan
    elif bad == "bool_clearance": current["clearance_m"] = True
    before = state(p)
    with pytest.raises(SemanticObservationError):
        p.evaluate(item)
    after = state(p)
    assert after[:4] == before[:4] and after[5] == before[5] and after[4] == []


def test_new_owner_can_reopen_fl_after_p05_capture_and_p06_hold(contract):
    p = provider(contract, "P05")
    for tick in range(64):
        step(p, "P05", tick)
    held = p.nominal_full12[:2]
    assert step(p, "P06", 64, placed=("FL",))[:2] == held
    for tick in range(65, 90):
        assert step(p, "P06", tick, placed=("FL",))[:2] == held
    assert step(p, "P07", 90, placed=("FL",))[:2] == held
    result = step(p, "P07", 91, placed=("FL",))
    assert result[0] != held[0] and abs(result[0]-held[0]) <= .5+1e-12
    assert result[1] == held[1]
    assert "capture_retired_servo_indices" not in p._continuous_layers[-1]


def test_prefix_existing_capture_does_not_lock_new_owner_or_p13_recovery(contract):
    p = provider(contract, "P13")
    start = p.nominal_full12
    end = round(contract.phase("P13").active_duration_s*120)
    for tick in range(end+100):
        command = step(p, "P13", tick, placed=LEG_ORDER)
    assert command == ZERO and start != ZERO
    assert not p._continuous_layers[0].get("capture_retired_servo_indices")
    assert all(v["retired_preexisting_layers"] == () for v in
               p.nominal_suggestion_diagnostics["capture_owner_hold"]["captures"].values())


def test_p12_early_capture_does_not_permanently_lock_later_p13_owner(contract):
    p = provider(contract, "P12")
    for tick in range(16):
        step(p, "P12", tick)
    step(p, "P12", 16, placed=("RL",))
    held = p.nominal_full12[4:6]
    assert step(p, "P13", 17, placed=("RL",))[4:6] == held
    end = round(contract.phase("P13").active_duration_s*120)
    for tick in range(18, 18+end+100):
        result = step(p, "P13", tick, placed=("RL",))
    assert result[4:6] == (0., 0.) and held != (0., 0.)


def test_old_mode_keeps_original_timed_trajectory_and_no_new_receipt(contract):
    old = provider(contract, "P05", enabled=False)
    for tick in range(1250):
        result = step(old, "P05", tick, placed=("FL",) if tick >= 64 else ())
    assert result[:2] == contract.phase("P05").end_full12[:2]
    assert "capture_owner_hold" not in old.nominal_suggestion_diagnostics
    assert not old._continuous_layers[0].get("capture_retired_servo_indices")


@pytest.mark.parametrize("phase", PHASE_IDS)
def test_no_capture_far_geometry_retains_old_path_all_phases(contract, phase):
    new, old = provider(contract, phase), provider(contract, phase, enabled=False)
    for tick in range(72):
        assert step(new, phase, tick) == step(old, phase, tick)
        assert new.tracking_servo_names == old.tracking_servo_names
        assert new.endpoint_issued == old.endpoint_issued


@pytest.mark.parametrize("source,target", list(zip(PHASE_IDS[:-1], PHASE_IDS[1:])))
def test_all_twelve_handoffs_hold_nominal_and_request_without_layer_reset(contract, source, target):
    p = provider(contract, source)
    bridge = PhaseTransitionBridge(build_semantic_projector(NEW / "execution_profile.yaml"))
    bridge.reset(state_id=source, applied_action_full12=p.nominal_full12)
    for tick in range(16):
        nominal = step(p, source, tick)
        out = bridge.project_tick((.2,)*12, state_id=source,
            nominal_action_full12=nominal, reference_action_full12=nominal, reference_delta_full12=ZERO)
    old_nominal, old_tracking = p.nominal_full12, p.tracking_servo_names
    old_request = out.projection.safe_projected_residual_full12
    next_nominal = step(p, target, 16)
    assert next_nominal == old_nominal and p.tracking_servo_names == old_tracking
    out = bridge.project_tick((-.2,)*12, state_id=target,
        nominal_action_full12=next_nominal, reference_action_full12=next_nominal, reference_delta_full12=ZERO)
    assert out.projection.safe_projected_residual_full12 == pytest.approx(old_request, abs=1e-12)
    assert out.transition_metric.handoff_hold_used
    assert [x["ticks"] for x in p._continuous_layers] == [17, 1]
    assert out.transition_metric.clipped_phase_scale_excess_residual_full12 == ZERO


def test_p06_gain_recovers_before_and_after_finite_source_endpoint(contract):
    new, old = provider(contract, "P06"), provider(contract, "P06", enabled=False)
    end = round(contract.phase("P06").active_duration_s*120)
    for tick in range(end+70):
        distance = -.215 if (16 <= tick < 32 or end+16 <= tick < end+32) else -.30
        before = new.nominal_full12
        actual = step(new, "P06", tick, distance=distance)
        legacy = step(old, "P06", tick, distance=distance)
        assert all(abs(a-b) <= .025+1e-12 for a, b in zip(actual[8:], before[8:]))
        diag = new.nominal_suggestion_diagnostics["p06_rolling_retirement"]
        assert 0. <= diag["wheel_gain"] <= 1.
        if tick in (47, end+47):
            assert actual[8:] == (.3,)*4 and legacy[8:] == (0.,)*4
            assert diag["peak_fraction"] == 1. and diag["wheel_gain"] == 1.
    assert new._continuous_layers[0]["last"][8:] == (0.,)*4


def test_p06_recovery_cannot_override_later_fr_wheel_owner(contract):
    p = provider(contract, "P06")
    for tick in range(20):
        step(p, "P06", tick, distance=-.215)
    step(p, "P07", 20, distance=-.30)
    for tick in range(21, 120):
        result = step(p, "P07", tick, distance=-.30)
    assert result[9] == -.63  # Not -.63 + .3, and not re-zeroed by P06 peak.
    assert tuple(result[i] for i in (8, 10, 11)) == (.3,)*3


def test_terminal_cannot_reactivate_tail_or_earn_capture(contract):
    p = provider(contract, "P06")
    for tick in range(20):
        step(p, "P06", tick, distance=-.215)
    result = step(p, "P06", 20, distance=-.30, placed=LEG_ORDER, terminal="FALL")
    assert result[8:] == (0.,)*4
    assert p.nominal_suggestion_diagnostics["capture_owner_hold"]["captures"] == {}
    assert p.nominal_suggestion_diagnostics["p06_wheel_tail"]["status"] == "terminal_no_tail"


@pytest.mark.parametrize("bad", ["bool_tick", "nan_time", "missing_placed", "integer_flag", "invalid", "stale"])
def test_invalid_capture_measurement_rejects_before_any_nominal_clock_mutation(contract, bad):
    p = provider(contract, "P05")
    step(p, "P05", 0)
    before = state(p)
    current = task("P05", 1)
    ev = current["physical_evaluator"]
    if bad == "bool_tick": ev["physics_tick"] = True
    elif bad == "nan_time": ev["simulation_time_s"] = math.nan
    elif bad == "missing_placed": ev["history"].pop("placed")
    elif bad == "integer_flag": ev["history"]["placed"]["FR"] = 1
    elif bad == "invalid": ev["valid"] = False
    elif bad == "stale": ev.update(physics_tick=0, simulation_time_s=0.)
    with pytest.raises(SemanticObservationError):
        p.evaluate(current, {"physics_tick": 1, "simulation_time_s": 1/120.})
    after = state(p)
    # MotionExecutor has identity equality; compare its serialized scalar state separately.
    assert after[:4] == before[:4]
    assert after[5] == before[5]
    assert [x["ticks"] for x in after[4]] == [x["ticks"] for x in before[4]]


def test_new_execution_profile_changes_only_approved_caps_and_revision():
    old = yaml.safe_load((OLD / "execution_profile.yaml").read_text(encoding="utf-8"))
    new = yaml.safe_load((NEW / "execution_profile.yaml").read_text(encoding="utf-8"))
    assert new["revision"] != old["revision"]
    expected = deepcopy(old)
    expected["revision"] = new["revision"]
    for index, phase in enumerate(PHASE_IDS, start=1):
        if index >= 3:
            expected["residual"]["phase_caps_full12"][phase][10] = 1.
            if index <= 5:
                expected["residual"]["phase_caps_full12"][phase][8] = 1.
    assert new == expected
    built = build_semantic_projector(NEW / "execution_profile.yaml")
    assert all(built.config.mask_for(phase) == (1,)*12 for phase in PHASE_IDS)


@pytest.mark.parametrize("channel,nominal", [(8, -.79), (10, .61)])
@pytest.mark.parametrize("target", [-.02, 0., .02])
def test_p03_finite_latent_can_cancel_and_reverse_only_with_new_authority(channel, nominal, target):
    old = build_semantic_projector(OLD / "execution_profile.yaml")
    new = build_semantic_projector(NEW / "execution_profile.yaml")
    command = list(ZERO); command[channel] = nominal
    required = target-nominal
    old_cap = old.config.scale_for("P03")[channel]*old.config.physical_residual_scale_full12[channel]
    if target == 0. or target*nominal < 0.:
        assert abs(required) > old_cap
    raw = list(ZERO); raw[channel] = math.atanh(required)
    history = ZERO
    for _ in range(80):
        out = new.project(raw, state_id="P03", nominal_action_full12=command,
            reference_action_full12=command, reference_delta_full12=ZERO,
            previous_projected_residual_full12=history, dt_s=1/120.)
        assert abs(out.safe_projected_residual_full12[channel]-history[channel]) <= .015+1e-12
        history = out.safe_projected_residual_full12
    assert out.applied_action_full12[channel] == pytest.approx(target, abs=1e-12)
    assert all(out.applied_action_full12[i] == 0. for i in range(12) if i != channel)
    assert math.isfinite(raw[channel])

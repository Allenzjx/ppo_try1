"""CPU source ownership checks; no synthetic physics or success claims."""
from copy import deepcopy
from dataclasses import replace

import pytest

from test_semantic_source_partial_order_v1 import contract, layer, make_provider as legacy_provider
from test_semantic_rr_top_continuation import top_input
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.ppo.semantic_supervisor import (
    NominalMotionProvider, P09_FREE_AIR_LIFT_MODE, RR_CARRY_SOURCE_MODE,
)


def make_provider(contract):
    spec = deepcopy(legacy_provider(contract, "P09").spec)
    spec["p09_lift_semantics"] = P09_FREE_AIR_LIFT_MODE
    spec["nominal"]["rr_carry_source_semantics"] = RR_CARRY_SOURCE_MODE
    return NominalMotionProvider.from_handoff(contract, spec=spec, stage_id="P09",
        nominal_full12=contract.phase("P09").start_full12, tracking_servo_names=())


def issue(p, contract, tick, *, qualified=True, air=True, gap=.05, vz=0., within_top=False,
          change=None):
    item, raw = top_input(contract, tick)
    rr = item["physical_evaluator"]["current_legs"]["RR"]
    rr.update(current_lift_valid=qualified, air=air, within_top_xy=within_top,
        clearance_m=gap, wheel_bottom_vz_m_s=vz, wheel_bottom_vz_observation_tick=tick)
    if within_top:
        rr.update(front_distance_m=.02, top_geometry=not air)
    if air:
        rr.update(contact_surface="NONE", top_surface_contact=False, obstacle_pair_active=False,
                  bearing_force_n=0., support=False)
    if change: change(item)
    before = deepcopy((item, raw))
    command = p.evaluate(item, raw)
    assert (item, raw) == before
    return command


def at_pending(contract, *, knee=True, index=0):
    p = make_provider(contract)
    issue(p, contract, 0)
    source = layer(p, "P09")
    time = p._rr_carry_knee_source_times[index] if knee else p._rr_carry_roll_source_time
    # Only CPU source state is positioned; this is not a physical state restore.
    source["ticks"] = source["motion"]._tick_index = source["motion"]._scaled_source_tick(time)
    return p, source


def test_events_come_from_contract_not_fixed_ticks_or_count(contract):
    original = contract.phase("P09")
    shift = .1
    shifted = replace(original,
        waypoints=tuple(replace(w, time_s=w.time_s+shift) for w in original.waypoints),
        atomic_groups=tuple(replace(g, time_s=g.time_s+shift) for g in original.atomic_groups))
    class ShiftedContract:
        def __getattr__(self, name): return getattr(contract, name)
        def phase(self, name): return shifted if name == "P09" else contract.phase(name)
    first, second = make_provider(contract), make_provider(ShiftedContract())
    assert second._rr_carry_knee_source_times == pytest.approx(tuple(t+shift for t in first._rr_carry_knee_source_times))
    assert second._rr_carry_roll_source_time == pytest.approx(first._rr_carry_roll_source_time+shift)


@pytest.mark.parametrize("qualified,gap,vz", [(False,.1,1.), (True,.009,-.14), (True,-.01,-.01)])
def test_pending_knee_waits_without_consuming_clock_replaying_pose_or_catching_up(contract, qualified, gap, vz):
    p, source = at_pending(contract)
    target, tick, index = p.nominal_full12, source["ticks"], source["sample"].tick_index
    tracking, emitted = p.tracking_servo_names, source["motion"].source_atomic_emitted
    for wall_tick in range(1, 5):
        assert issue(p, contract, wall_tick, qualified=qualified, gap=gap, vz=vz) == target
        assert source["ticks"] == source["motion"]._tick_index == tick
        assert source["sample"].tick_index == index and p.tracking_servo_names == tracking
        assert source["motion"].source_atomic_emitted == emitted
    issue(p, contract, 5, gap=.02, vz=.01)
    assert source["ticks"] == tick+1 and source["sample"].tick_index == tick


@pytest.mark.parametrize("gap,vz,top", [(.02,-.2,False), (.009,.001,False), (.009,0.,False), (.009,-.1,True)])
def test_high_or_non_decreasing_air_and_actual_platform_are_not_static_pose_gates(contract, gap, vz, top):
    p, source = at_pending(contract)
    tick = source["ticks"]
    issue(p, contract, 1, gap=gap, vz=vz, within_top=top)
    assert source["ticks"] == tick+1


@pytest.mark.parametrize("qualified,air,surface", [(False,True,"NONE"), (False,False,"TOP"), (True,False,"OBSTACLE_AMBIGUOUS")])
def test_xy_alone_cannot_bypass_current_lift_or_legal_drop_region(contract, qualified, air, surface):
    p, source = at_pending(contract)
    tick = source["ticks"]
    issue(p, contract, 1, qualified=qualified, air=air, gap=-.05, vz=-.1, within_top=True,
          change=lambda t: t["physical_evaluator"]["current_legs"]["RR"].update(
              contact_surface=surface))
    assert source["ticks"] == tick


def test_xy_air_below_top_still_obeys_low_clearance_descent_guard(contract):
    p, source = at_pending(contract)
    tick = source["ticks"]
    issue(p, contract, 1, gap=-.05, vz=-.1, within_top=True)
    assert source["ticks"] == tick


def test_stale_evaluator_cannot_consume_pending_event_even_if_vz_tick_matches_it(contract):
    p, source = at_pending(contract, knee=False)
    item, raw = top_input(contract, 1)
    item["physical_evaluator"]["current_legs"]["RR"].update(wheel_bottom_vz_m_s=0., wheel_bottom_vz_observation_tick=1)
    raw["physics_tick"] = 2
    raw["simulation_time_s"] = 2/120
    result = p._rr_pending_carry_readiness(source, item, raw)
    assert not result["ready"] and not result["current_evidence_fresh"]


def test_low_clearance_unknown_or_stale_derivative_is_not_fabricated_zero(contract):
    for change in (None, lambda t: t["physical_evaluator"]["current_legs"]["RR"].update(
            wheel_bottom_vz_observation_tick=-1)):
        p, source = at_pending(contract)
        tick = source["ticks"]
        issue(p, contract, 1, gap=.009, vz=None if change is None else .1, change=change)
        assert source["ticks"] == tick
        assert source["sequence_diagnostic"]["current_free_lift_source_readiness"]["RR_bottom_vz_m_s"] is None


def test_no_pending_knee_event_does_not_freeze_normal_forwarding_or_landing(contract):
    p, source = at_pending(contract)
    source["ticks"] += 1
    source["motion"]._tick_index += 1
    tick = source["ticks"]
    issue(p, contract, 1, qualified=False, gap=-.007, vz=-.1)
    assert source["ticks"] == tick+1


def test_pending_readiness_diagnostic_is_cleared_on_next_non_event_tick(contract):
    p, source = at_pending(contract)
    issue(p, contract, 1, gap=.02)
    assert source["sequence_diagnostic"]["current_free_lift_source_readiness"]["ready"]
    issue(p, contract, 2, gap=.02)
    assert source["sequence_diagnostic"]["observation_tick"] == 2
    assert source["sequence_diagnostic"]["current_free_lift_source_readiness"] is None


@pytest.mark.parametrize("gap", [-.012008445891588275, 0., .025])
def test_first_rolling_group_accepts_qualified_air_even_with_negative_topgap(contract, gap):
    p, source = at_pending(contract, knee=False)
    tick = source["ticks"]
    command = issue(p, contract, 1, gap=gap, vz=-.05)
    assert source["ticks"] == tick+1 and command[8:] == (.3,)*4
    assert len(source["sample"].atomic_groups) == 1
    assert set(source["sample"].atomic_groups[0].channels) == set(WHEEL_ORDER)


def test_first_rolling_accepts_current_verified_top_without_endpoint_prerequisite(contract):
    p, source = at_pending(contract, knee=False)
    tick = source["ticks"]
    command = issue(p, contract, 1, air=False, gap=-.012)
    assert source["ticks"] == tick+1 and command[8:] == (.3,)*4
    assert not source["sample"].endpoint_issued


@pytest.mark.parametrize("fault", ["no_q", "ground", "edge", "unknown_bearing", "only_one_support", "terminal"])
def test_first_rolling_rejects_missing_current_evidence_and_unknown_edge(contract, fault):
    p, source = at_pending(contract, knee=False)
    tick = source["ticks"]
    def change(t):
        ev = t["physical_evaluator"]; rr = ev["current_legs"]["RR"]
        if fault == "no_q": rr["current_lift_valid"] = False
        elif fault == "ground": rr["ground_contact"] = True
        elif fault == "edge": rr["contact_surface"] = "OBSTACLE_AMBIGUOUS"
        elif fault == "unknown_bearing": rr["bearing_verified"] = False
        elif fault == "terminal": ev["termination_reason"] = "SAFETY_ABORT"
        else:
            for leg in ("FL", "RL"): ev["current_legs"][leg]["support"] = False
    issue(p, contract, 1, air=False, gap=-.012, change=change)
    assert source["ticks"] == tick


def test_explicit_stop_still_owns_all_authored_wheels_when_q_is_lost(contract):
    p, source = at_pending(contract, knee=False)
    assert issue(p, contract, 1)[8:] == (.3,)*4
    stop_time = next(w.time_s for w in contract.phase("P09").waypoints
        if w.time_s > p._rr_carry_roll_source_time and set(w.atomic_channels) == set(WHEEL_ORDER)
        and w.full12[8:] == (0.,)*4)
    stop_tick = source["motion"]._scaled_source_tick(stop_time)
    source["ticks"] = source["motion"]._tick_index = stop_tick
    out = issue(p, contract, 2, qualified=False, gap=-.05, change=lambda t:
        t["physical_evaluator"]["current_legs"]["RR"].update(ground_contact=True))
    assert out[8:] == (0.,)*4 and source["sample"].tick_index == stop_tick
    assert set(range(8,12)) <= source["touched"]


@pytest.mark.parametrize("fault", ["old_mode", "unknown_marker", "no_sequence", "no_inheritance"])
def test_source_marker_rejects_incompatible_modes(contract, fault):
    p = make_provider(contract); spec = deepcopy(p.spec)
    if fault == "old_mode": spec["p09_lift_semantics"] = "functional_lift_edge_v2"
    elif fault == "unknown_marker": spec["nominal"]["rr_carry_source_semantics"] = "wrong"
    elif fault == "no_sequence": spec["nominal"].pop("sequence_semantics")
    else: spec["nominal"]["continuous_channel_inheritance"] = False
    with pytest.raises(ValueError):
        NominalMotionProvider(contract, spec=spec)


def test_new_qualification_without_explicit_source_marker_keeps_old_source_path(contract):
    p = make_provider(contract); spec = deepcopy(p.spec)
    spec["nominal"].pop("rr_carry_source_semantics")
    old = NominalMotionProvider(contract, spec=spec)
    assert old._rr_carry_source_mode is None and old._rr_carry_knee_source_times == ()

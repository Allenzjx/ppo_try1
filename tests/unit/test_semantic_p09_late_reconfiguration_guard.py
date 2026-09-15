"""CPU scheduling checks only; fixtures do not establish physical success."""
from copy import deepcopy

import pytest

from test_semantic_rr_top_continuation import at_source_tick, top_input
from test_semantic_source_partial_order_v1 import contract, layer
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER


def send(p, contract, tick, *, air=False, crossed=False, change=None):
    item, raw = top_input(contract, tick)
    rr = item["physical_evaluator"]["current_legs"]["RR"]
    # Rounded measured A/B geometry: no P/capture required; RR joints unchanged.
    rr.update(front_distance_m=.096715 if crossed else -.031451,
              clearance_m=.013059 if air else -.012102, within_top_xy=crossed)
    if air:
        rr.update(air=True, support=False, bearing_force_n=0., contact_surface="NONE",
                  top_surface_contact=False, obstacle_pair_active=False, top_geometry=False)
    else:
        rr["top_geometry"] = crossed
    if change:
        change(item)
    before = deepcopy((item, raw))
    out = p.evaluate(item, raw)
    assert (item, raw) == before
    return out, item


def before_group(contract):
    p = at_source_tick(contract, 639)
    for tick in range(639, 648):
        send(p, contract, tick)
    return p


def test_verified_source_is_stop640_then_late648_with_original_atomic_channels(contract):
    p = before_group(contract)
    stop, group = p._p09_late_source
    assert round(stop*120) == 640 and round(group*120) == 648
    phase = contract.phase("P09")
    atomic = next(g for g in phase.atomic_groups if g.source_full12_atomic)
    assert set(atomic.required_runtime_channels) == set(SERVO_ORDER+WHEEL_ORDER)
    waypoint = next(w for w in phase.waypoints if w.time_s == atomic.time_s)
    assert set(waypoint.changed_channels) == {
        "front_left_hip", "front_left_knee", "rear_left_hip", "rear_left_knee", "front_left_ankle"}


def test_corner_holds_source_clock_and_current_owners_but_can_create_crossing(contract):
    p = at_source_tick(contract, 639)
    for tick in range(639, 648):
        out, _ = send(p, contract, tick)
        if tick == 640:
            assert out[8:] == (0.,)*4
            assert layer(p, "P09")["sample"].atomic_groups
            assert not p.nominal_suggestion_diagnostics["rr_carry_continuation"]["added_rolling_suggestion"]
    source = layer(p, "P09")
    held = out[:8]
    prior_tracking = p.tracking_servo_names
    emitted = source["motion"].source_atomic_emitted
    for tick in range(648, 748):
        out, item = send(p, contract, tick)
        assert source["ticks"] == source["motion"]._tick_index == 648
        assert source["sample"].tick_index == 647
        assert source["motion"].source_atomic_emitted == emitted
        assert out[:8] == held and p.tracking_servo_names == prior_tracking
        assert out[8:] == (.3,)*4
        assert not item["physical_evaluator"]["history"]["placed"]["RR"]
        assert not p.endpoint_issued
        assert p.nominal_suggestion_diagnostics["rr_carry_continuation"]["late_reconfiguration_waiting"]
    out, item = send(p, contract, 748, air=True, crossed=True)
    assert source["sample"].tick_index == 648 and source["ticks"] == 649
    assert source["motion"].source_atomic_emitted == emitted+1
    assert source["sequence_diagnostic"]["late_group_start_tick"] == 748
    assert len(source["sample"].atomic_groups) == 1
    assert source["sample"].atomic_groups[0].source_full12_atomic
    assert tuple(out[i] for i in (0, 1, 4, 5, 8)) == tuple(source["sample"].full12[i] for i in (0, 1, 4, 5, 8))
    assert out[8:] == (-1.07, 0., 0., 0.)
    assert not item["physical_evaluator"]["history"]["placed"]["RR"]
    assert item["physical_evaluator"]["current_legs"]["RR"]["clearance_m"] < .015
    send(p, contract, 749)  # New unsafe geometry cannot rewind/repeat an issued group.
    assert source["sample"].tick_index == 649 and source["motion"].source_atomic_emitted == emitted+1


@pytest.mark.parametrize("air,gap", [(True, 0.), (True, .013059), (True, .1), (False, -.015), (False, .025)])
def test_legal_current_over_platform_entry_does_not_require_full_capture_or_fixed_hover(contract, air, gap):
    p = before_group(contract)
    _, item = send(p, contract, 648, air=air, crossed=True, change=lambda t:
        t["physical_evaluator"]["current_legs"]["RR"].update(clearance_m=gap))
    assert layer(p, "P09")["sample"].tick_index == 648
    assert not item["physical_evaluator"]["history"]["placed"]["RR"]


@pytest.mark.parametrize("key,value", [
    ("front_distance_m", -.00001), ("within_top_xy", False), ("within_lateral_span", False),
    ("ground_contact", True), ("current_lift_valid", False), ("motion_continuation_allowed", False),
    ("clearance_m", -.00001), ("clearance_m", float("nan")), ("front_distance_m", None),
])
def test_air_entry_rejects_current_invalid_geometry_or_q(contract, key, value):
    p = before_group(contract)
    send(p, contract, 648, air=True, crossed=True, change=lambda t:
        t["physical_evaluator"]["current_legs"]["RR"].update({key: value}))
    assert layer(p, "P09")["ticks"] == 648


@pytest.mark.parametrize("key,value", [
    ("contact_surface", "FRONT_WALL"), ("top_geometry", False), ("top_surface_contact", False),
    ("obstacle_pair_active", False), ("bearing_verified", False), ("bearing_force_n", .199),
    ("clearance_m", -.01501), ("clearance_m", .02501),
])
def test_top_entry_requires_current_legal_surface_and_bearing(contract, key, value):
    p = before_group(contract)
    send(p, contract, 648, crossed=True, change=lambda t:
        t["physical_evaluator"]["current_legs"]["RR"].update({key: value}))
    assert layer(p, "P09")["ticks"] == 648


def test_pending_air_feedback_can_cover_last_five_mm_without_waiting_for_contact(contract):
    p = before_group(contract)
    out, _ = send(p, contract, 648, air=True, change=lambda t:
        t["physical_evaluator"]["current_legs"]["RR"].update(front_distance_m=-.002))
    assert layer(p, "P09")["ticks"] == 648 and out[8:] == (.3,)*4


@pytest.mark.parametrize("fault", ["only_one", "wall_only", "unknown_bearing"])
def test_two_current_other_supports_not_rr_or_unknown_load_required(contract, fault):
    p = before_group(contract)
    def change(t):
        ev = t["physical_evaluator"]
        ev["physical_evidence_status"] = "CONTACT_BEARING_UNVERIFIED"
        for leg in ("FL", "RL"):
            row = ev["current_legs"][leg]
            if fault == "only_one": row["support"] = False
            elif fault == "wall_only": row.update(ground_contact=False, top_surface_contact=False)
            else: row["bearing_verified"] = False
    out, _ = send(p, contract, 648, air=True, crossed=True, change=change)
    assert layer(p, "P09")["ticks"] == 648 and out[8:] == (0.,)*4


@pytest.mark.parametrize("fault", ["task_terminal", "evaluator_terminal", "unknown_sensor"])
def test_terminal_or_unverified_sample_cannot_unlock_or_create_pending_carry(contract, fault):
    p = before_group(contract)
    def change(t):
        if fault == "task_terminal": t["termination_reason"] = "INCOMPLETE_CONTROLLER_BLOCKED"
        elif fault == "evaluator_terminal": t["physical_evaluator"]["termination_reason"] = "SAFETY_ABORT"
        else: t["physical_evaluator"]["physical_evidence_status"] = "UNVERIFIED_SENSOR"
    out, _ = send(p, contract, 648, air=True, crossed=True, change=change)
    assert layer(p, "P09")["ticks"] == 648 and out[8:] == (0.,)*4

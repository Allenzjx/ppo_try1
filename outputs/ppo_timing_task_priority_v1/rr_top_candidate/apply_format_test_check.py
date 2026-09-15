"""CPU-only RR TOP-corner continuation; no physical-success claim."""
from copy import deepcopy

import pytest

from test_semantic_source_partial_order_v1 import contract, current_input, layer, make_provider
from wlr50_clean.ppo.semantic_supervisor import SemanticObservationError


def top_input(contract, tick):
    item, raw = current_input(contract, "P09", tick, continuing=True,
                              supports=("FL", "FR", "RL"))
    ev = item["physical_evaluator"]
    ev["physical_evidence_status"] = "VERIFIED"
    ev["history"]["active_lift"]["RR"] = True
    for leg, row in ev["current_legs"].items():
        row.update(bearing_verified=True, bearing_force_n=8.,
                   top_surface_contact=False, obstacle_pair_active=False)
        if leg != "RR":
            row.update(ground_contact=True, air=False, support=True)
    ev["current_legs"]["RR"].update(active_attempt=True, air=False, ground_contact=False,
        contact_surface="TOP", contact_mode="TOP", top_surface_contact=True,
        obstacle_pair_active=True, bearing_force_n=2.35, support=True,
        front_distance_m=-.0295, clearance_m=-.012, within_lateral_span=True,
        within_top_xy=False, top_geometry=False, top_contact=False)
    return item, raw


def at_source_tick(contract, source_tick):
    p = make_provider(contract, "P09")
    item, raw = top_input(contract, 0)
    p.evaluate(item, raw)
    source = layer(p, "P09")
    # Jump only the synthetic source executor in this CPU fixture. No state
    # snapshot, physical simulator, recorded run or source command is changed.
    source["ticks"] = source["motion"]._tick_index = source_tick
    return p


def source_endpoint(contract):
    p = make_provider(contract, "P09")
    item, raw = top_input(contract, 0)
    p.evaluate(item, raw)
    source = layer(p, "P09")
    return source["motion"]._endpoint_tick(source["motion"].phase)


def dispatch(p, contract, tick=240, change=None):
    item, raw = top_input(contract, tick)
    if change:
        change(item)
    before = deepcopy((item, raw))
    out = p.evaluate(item, raw)
    assert (item, raw) == before
    return out, p.nominal_suggestion_diagnostics["rr_carry_continuation"], item


def test_real_top_corner_before_center_cross_can_receive_only_existing_wheel_suggestion(contract):
    p = at_source_tick(contract, source_endpoint(contract)+1)
    out, diag, item = dispatch(p, contract)
    rr = item["physical_evaluator"]["current_legs"]["RR"]
    assert not rr["top_contact"] and not rr["top_geometry"] and not rr["within_top_xy"]
    assert rr["front_distance_m"] < 0. and rr["clearance_m"] < 0.
    assert out[8:] == (.3,)*4
    assert diag["reason"] == "current_TOP_corner_continuation"
    assert diag["current_TOP_continuation_eligible"] and not diag["ordered_source_wheel_owner"]
    assert item["physical_evaluator"]["history"]["placed"]["RR"] is False


def test_finite_source_and_fresh_authored_endpoint_stop_keep_priority(contract):
    p = at_source_tick(contract, source_endpoint(contract)-1)
    _, before, _ = dispatch(p, contract, 240)
    assert not before["current_TOP_continuation_eligible"]
    assert before["ordered_source_wheel_owner"] and not before["added_rolling_suggestion"]
    endpoint, fresh, _ = dispatch(p, contract, 241)
    assert layer(p, "P09")["sample"].endpoint_issued
    assert fresh["current_TOP_continuation_eligible"] and fresh["ordered_source_wheel_owner"]
    assert endpoint[8:] == (0.,)*4
    after, clear, _ = dispatch(p, contract, 242)
    assert clear["added_rolling_suggestion"] and not clear["ordered_source_wheel_owner"]
    assert after[:8] == endpoint[:8]  # No servo entry restore or amplitude change.
    assert after[8:] == (.3,)*4


@pytest.mark.parametrize("surface", ["FRONT_WALL", "EDGE", "UNKNOWN", "NONE", None])
def test_non_top_contact_never_enables_new_branch_even_with_stale_true_lift_bits(contract, surface):
    p = at_source_tick(contract, source_endpoint(contract)+1)
    out, diag, _ = dispatch(p, contract, change=lambda t:
        t["physical_evaluator"]["current_legs"]["RR"].update(contact_surface=surface))
    assert not diag["current_TOP_continuation_eligible"]
    assert not diag["added_rolling_suggestion"] and out[8:] == (0.,)*4


@pytest.mark.parametrize("key,value", [
    ("active_attempt", False), ("current_lift_valid", False),
    ("motion_continuation_allowed", False), ("ground_contact", True),
    ("within_lateral_span", False), ("top_surface_contact", False),
    ("obstacle_pair_active", False), ("bearing_verified", False),
    ("bearing_force_n", .1999), ("bearing_force_n", None), ("support", False),
    ("clearance_m", -.01501), ("clearance_m", .02501), ("clearance_m", float("nan")),
    ("front_distance_m", -.22001), ("front_distance_m", .1),
])
def test_current_invalid_ground_roi_gap_or_bearing_rejects_top_continuation(contract, key, value):
    p = at_source_tick(contract, source_endpoint(contract)+1)
    out, diag, _ = dispatch(p, contract, change=lambda t:
        t["physical_evaluator"]["current_legs"]["RR"].update({key:value}))
    assert not diag["current_TOP_continuation_eligible"]
    assert not diag["added_rolling_suggestion"] and out[8:] == (0.,)*4


@pytest.mark.parametrize("distance", [-.004, 0., .02])
def test_top_branch_does_not_stop_at_old_minus5mm_air_approach_threshold(contract, distance):
    p = at_source_tick(contract, source_endpoint(contract)+1)
    out, diag, _ = dispatch(p, contract, change=lambda t:
        t["physical_evaluator"]["current_legs"]["RR"].update(front_distance_m=distance))
    assert diag["added_rolling_suggestion"] and out[8:] == (.3,)*4


@pytest.mark.parametrize("kind", ["unearned", "placed", "task_terminal", "evaluator_terminal"])
def test_history_and_termination_do_not_award_or_extend_crossing(contract, kind):
    p = at_source_tick(contract, source_endpoint(contract)+1)
    def change(t):
        ev=t["physical_evaluator"]
        if kind == "unearned": ev["history"]["active_lift"]["RR"] = False
        elif kind == "placed": ev["history"]["placed"]["RR"] = True
        elif kind == "task_terminal": t["termination_reason"] = "TASK_FAILURE_BODY_COLLISION"
        else: ev["termination_reason"] = "SAFETY_ABORT"
    out, diag, _ = dispatch(p, contract, change=change)
    assert not diag["current_TOP_continuation_eligible"] and not diag["added_rolling_suggestion"]
    assert out[8:] == (0.,)*4


@pytest.mark.parametrize("fault", ["only_one_other", "unverified_bearing", "wall_reactions_only", "missing_force"])
def test_rr_itself_or_unknown_normalization_cannot_supply_other_supports(contract, fault):
    p = at_source_tick(contract, source_endpoint(contract)+1)
    def change(t):
        ev=t["physical_evaluator"]
        ev["physical_evidence_status"]="CONTACT_BEARING_UNVERIFIED"
        for leg in ("FL","RL"):
            row=ev["current_legs"][leg]
            row.update(load_fraction=0.,load_fraction_valid=False)
            if fault=="only_one_other": row["support"]=False
            elif fault=="unverified_bearing": row["bearing_verified"]=False
            elif fault=="wall_reactions_only": row.update(ground_contact=False,top_surface_contact=False)
            else: row["bearing_force_n"]=None
    out, diag, _ = dispatch(p, contract, change=change)
    assert not diag["current_TOP_continuation_eligible"] and out[8:] == (0.,)*4


def test_independent_two_supports_remain_valid_if_fourth_leg_normalization_is_unknown(contract):
    p = at_source_tick(contract, source_endpoint(contract)+1)
    def change(t):
        ev=t["physical_evaluator"]
        ev["physical_evidence_status"]="CONTACT_BEARING_UNVERIFIED"
        ev["current_legs"]["RL"]["bearing_verified"]=False
        for row in ev["current_legs"].values():
            row.update(load_fraction=0.,load_fraction_valid=False)
    out, diag, _ = dispatch(p, contract, change=change)
    assert diag["added_rolling_suggestion"] and out[8:] == (.3,)*4


@pytest.mark.parametrize("status", ["UNVERIFIED_SENSOR", "NONFINITE", None])
def test_invalid_sensor_status_cannot_create_top_continuation(contract, status):
    p = at_source_tick(contract, source_endpoint(contract)+1)
    out, diag, _ = dispatch(p, contract, change=lambda t:
        t["physical_evaluator"].update(physical_evidence_status=status))
    assert not diag["added_rolling_suggestion"] and out[8:] == (0.,)*4


def test_invalid_evaluator_is_rejected_by_existing_capture_validation(contract):
    p = at_source_tick(contract, source_endpoint(contract)+1)
    item, raw = top_input(contract,240)
    item["physical_evaluator"]["valid"]=False
    assert not p._rr_top_continuation_allowed(item)
    with pytest.raises(SemanticObservationError,match="valid current evaluator"):
        p.evaluate(item,raw)


def test_missing_p09_layer_is_not_treated_as_an_already_executed_suffix(contract):
    p = make_provider(contract,"P09")
    item, _ = top_input(contract,240)
    assert not p._rr_top_continuation_allowed(item)

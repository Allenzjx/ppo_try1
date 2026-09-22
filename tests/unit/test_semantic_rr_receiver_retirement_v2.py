"""Receiver-only soft-credit tests; synthetic snapshots are not physics evidence."""
from copy import deepcopy
from pathlib import Path

import pytest

from wlr50_clean.ppo import semantic_supervisor as s


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml"
WEIGHT = .85 / 4 * .1 * .5


def fixture(mode):
    spec = s.load_task_spec(SPEC)
    assert spec["rr_postcross_workspace_semantics"] == s.RR_WORKSPACE_RETIREMENT_MODE_V2
    spec["rr_postcross_workspace_semantics"] = mode
    snapshot = {
        "valid": True, "termination_reason": None,
        "history": {"active_lift": {"RR": True}, "front_edge_crossed": {"RR": True}},
        "current_legs": {"RR": {
            "current_lift_valid": True, "lift_established": True,
            "motion_continuation_allowed": True, "body_control_evidence": True,
            "ground_relative_lift_m": .06300224191249629,
            "ground_contact": False, "within_top_xy": True, "within_lateral_span": True,
            "front_distance_m": .14, "air": True, "clearance_m": .012916069011434272,
            "top_surface_contact": False, "top_contact": False,
        }},
        "transfer_roles": {"RR": {"workspace_progress": .3803756506975887}},
    }
    return spec, snapshot


def retired(spec, snapshot, leg="RR"):
    before = deepcopy(snapshot)
    result = s._current_rr_receiver_preparation_retired(spec, leg, snapshot)
    assert snapshot == before
    return result


def workspace(spec, snapshot):
    supervisor = object.__new__(s.TaskStageSupervisor)
    supervisor.spec = spec
    return supervisor._workspace_potential_progress("RR", snapshot)


def test_v1_keeps_original_currentQ_dependency():
    spec, snapshot = fixture(s.RR_WORKSPACE_RETIREMENT_MODE)
    # v1 did not require these new soft-only inputs; preserve that old meaning.
    for key in ("lift_established", "motion_continuation_allowed", "ground_relative_lift_m"):
        snapshot["current_legs"]["RR"].pop(key)
    assert retired(spec, snapshot)
    snapshot["current_legs"]["RR"]["current_lift_valid"] = False
    assert not retired(spec, snapshot)


def test_8704_to_8712_receiver_share_only_counterexample():
    # Recorded scalar values, with a minimal synthetic surrounding snapshot.
    # This is algebraic Phi checking, not replay, new GAE or a physical rollout.
    old, a = fixture(s.RR_WORKSPACE_RETIREMENT_MODE)
    new = deepcopy(old)
    new["rr_postcross_workspace_semantics"] = s.RR_WORKSPACE_RETIREMENT_MODE_V2
    b = deepcopy(a)
    b["current_legs"]["RR"].update(current_lift_valid=False, body_control_evidence=False,
        clearance_m=.003976173645021605, ground_relative_lift_m=.05406234654608362)
    b["transfer_roles"]["RR"]["workspace_progress"] = .3084242123196141
    assert retired(old, a) and retired(new, a)
    assert not retired(old, b) and retired(new, b)
    assert workspace(new, a) == workspace(old, a)
    phi_difference = .85 / 4 * .1 * (workspace(new, b) - workspace(old, b))
    assert phi_difference == pytest.approx(.007347992744104101)
    assert 0 < phi_difference <= WEIGHT
    assert 5 * .9985 * phi_difference == pytest.approx(.036684853774939725)
    # The separate current-lift credit is not repaired or fabricated.
    sup = object.__new__(s.TaskStageSupervisor)
    sup.spec = new
    assert sup._current_lift_credit("RR", b) == 0.


def test_v2_currentQ_true_states_unchanged_and_top_capture_not_fake_support():
    old, snapshot = fixture(s.RR_WORKSPACE_RETIREMENT_MODE)
    new = deepcopy(old)
    new["rr_postcross_workspace_semantics"] = s.RR_WORKSPACE_RETIREMENT_MODE_V2
    assert workspace(new, snapshot) == workspace(old, snapshot)
    snapshot["current_legs"]["RR"].update(air=False, top_surface_contact=True,
        top_contact=True, current_lift_valid=False, body_control_evidence=False,
        clearance_m=-.0002)
    assert retired(new, snapshot)
    assert not snapshot["current_legs"]["RR"]["current_lift_valid"]


@pytest.mark.parametrize("change", [
    {"ground_contact": True}, {"ground_contact": False, "lift_established": False},
    {"motion_continuation_allowed": False}, {"within_top_xy": False},
    {"within_lateral_span": False}, {"front_distance_m": -.00001},
    {"ground_relative_lift_m": .002999}, {"clearance_m": -.015001},
    {"air": False, "top_surface_contact": False, "top_contact": False},
    {"air": False, "top_surface_contact": True, "top_contact": False},
])
def test_v2_current_invalidity_cannot_keep_receiver_credit(change):
    spec, snapshot = fixture(s.RR_WORKSPACE_RETIREMENT_MODE_V2)
    snapshot["current_legs"]["RR"].update(change)
    assert not retired(spec, snapshot)


@pytest.mark.parametrize("key", ["ground_relative_lift_m", "front_distance_m", "clearance_m",
                                 "lift_established", "motion_continuation_allowed",
                                 "within_top_xy", "within_lateral_span", "ground_contact", "air"])
def test_v2_missing_required_current_evidence_is_not_credit(key):
    spec, snapshot = fixture(s.RR_WORKSPACE_RETIREMENT_MODE_V2)
    snapshot["current_legs"]["RR"].pop(key)
    assert not retired(spec, snapshot)


@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf"), "0.05"])
def test_v2_ground_relative_measurement_must_be_finite_numeric(bad):
    spec, snapshot = fixture(s.RR_WORKSPACE_RETIREMENT_MODE_V2)
    snapshot["current_legs"]["RR"]["ground_relative_lift_m"] = bad
    # Avoid NaN equality in the no-mutation assertion.
    assert not s._current_rr_receiver_preparation_retired(spec, "RR", snapshot)


@pytest.mark.parametrize("kind", ["invalid", "unsafe", "noQ", "noCross", "otherLeg", "disabled"])
def test_v2_history_validity_and_leg_scope(kind):
    spec, snapshot = fixture(s.RR_WORKSPACE_RETIREMENT_MODE_V2)
    leg = "RR"
    if kind == "invalid": snapshot["valid"] = False
    elif kind == "unsafe": snapshot["termination_reason"] = "BODY_COLLISION"
    elif kind == "noQ": snapshot["history"]["active_lift"]["RR"] = False
    elif kind == "noCross": snapshot["history"]["front_edge_crossed"]["RR"] = False
    elif kind == "otherLeg": leg = "FL"
    else: spec.pop("rr_postcross_workspace_semantics")
    assert not retired(spec, snapshot, leg)


def test_v2_existing_threshold_no_new_fixed_clearance():
    spec, snapshot = fixture(s.RR_WORKSPACE_RETIREMENT_MODE_V2)
    snapshot["current_legs"]["RR"]["ground_relative_lift_m"] = spec["history"]["minimum_initial_clearance_gain_m"]
    assert retired(spec, snapshot)
    snapshot["current_legs"]["RR"]["clearance_m"] = spec["geometry"]["top_gap_min_m"]
    assert retired(spec, snapshot)

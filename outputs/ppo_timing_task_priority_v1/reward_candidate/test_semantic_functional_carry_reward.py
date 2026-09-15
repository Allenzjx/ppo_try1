"""Reward-only carry floor: fixtures use existing TaskEvaluator snapshot fields.

No Isaac, optimizer, actor checkpoint, observation/schema or production config
mutation. The floor changes three body costs only, not physical task acceptance.
"""
from copy import deepcopy
from dataclasses import replace

import pytest

from test_semantic_transfer_role_reward import frame, reward
from wlr50_clean.ppo.semantic_reward import (
    CARRY_BODY_ALLOWANCE_MODE, SemanticRewardCalculator,
    _current_functional_body_allowance, _validate_carry_body_allowance,
    load_semantic_reward_config,
)

ORDER = ("FR", "FL", "RR", "RL")
CFG = load_semantic_reward_config().path.parent.parent / "ppo_fsm_reference_p09_stable_v2/reward_config.yaml"


def calculator(enabled=True):
    config = load_semantic_reward_config(CFG)
    values = dict(config.values)
    if enabled:
        values.update(carry_body_allowance=CARRY_BODY_ALLOWANCE_MODE, capture_settle_window_s=.5)
    return SemanticRewardCalculator(replace(config, values=values))


def carrying(leg="FR", *, tick=240, fraction=0., stage="P02"):
    built = frame(tick, fraction=fraction, stage=stage)
    legs = {name: dict(support=True, bearing_verified=True, bearing_force_n=10.,
        ground_contact=True, top_contact=False, top_surface_contact=False, top_geometry=False,
        obstacle_pair_active=False, air=False, within_lateral_span=True,
        within_top_xy=False, front_distance_m=-.1, clearance_m=-.05,
        active_attempt=False, load_fraction=0., load_fraction_valid=False) for name in ORDER}
    legs[leg].update(support=False, bearing_force_n=0., ground_contact=False, air=True,
        clearance_m=.02, active_attempt=True, current_lift_valid=leg == "RR",
        motion_continuation_allowed=True)
    if leg == "RR":
        legs[leg]["clearance_m"] = -.03  # Functional lift need not already be above the platform.
    # One non-target pair has unavailable bearing classification. The producer
    # therefore marks ALL normalized fractions unavailable, while two other
    # legs still have independent verified bearing force/contact evidence.
    legs["RR" if leg != "RR" else "RL"]["bearing_verified"] = False
    history = {key: dict.fromkeys(ORDER, False) for key in ("active_lift", "front_edge_crossed", "placed")}
    history["active_lift"][leg] = True
    for prior in ORDER[:ORDER.index(leg)]:
        history["placed"][prior] = True
    history["event_ticks"] = {"active_lift": {leg: tick-100}, "front_edge_crossed": {}, "placed": {}}
    task = deepcopy(built.task)
    task["physical_evaluator"] = dict(valid=True, physical_evidence_status="CONTACT_BEARING_UNVERIFIED",
        termination_reason=None, physics_tick=tick, simulation_time_s=tick/120,
        current_legs=legs, history=history)
    # No instantaneous motion/CoM/load-drop evidence is supplied or required.
    return replace(built, task=task)


def captured(leg="FR", *, tick=240, event_tick=240, fraction=0., stage="P03"):
    built = carrying(leg, tick=tick, fraction=fraction, stage=stage)
    ev = built.task["physical_evaluator"]
    ev["history"]["placed"][leg] = True
    ev["history"]["front_edge_crossed"][leg] = True
    ev["history"]["event_ticks"]["placed"][leg] = event_tick
    ev["current_legs"][leg].update(support=True, bearing_verified=True, bearing_force_n=10.,
        air=False, top_contact=True, top_surface_contact=True, obstacle_pair_active=True,
        within_top_xy=True, top_geometry=True, front_distance_m=.01, clearance_m=0.)
    return built


def compare(built):
    before = frame(0)
    old = reward(before, built, calculator=calculator(False))
    new = reward(before, built, calculator=calculator())
    return old, new


@pytest.mark.parametrize("leg", ORDER)
def test_current_earned_functional_carry_keeps_all_three_body_allowances_without_motion(leg):
    built = carrying(leg)
    snapshot = deepcopy(built)
    old, new = compare(built)
    for name in ("gravity_attitude", "euler_rate", "angular_acceleration"):
        assert new["cost_components"][name] == pytest.approx(old["cost_components"][name]*.2)
    assert new["families"]["body_stability"] == pytest.approx(old["families"]["body_stability"]*.2)
    for family in old["families"]:
        if family != "body_stability":
            assert new["families"][family] == old["families"][family]
    for key in ("potential_shaping", "terminal_event", "elapsed_physics_s"):
        assert new[key] == old[key]
    assert built == snapshot  # Fixed 372/HISTORY/task input is only consumed.


@pytest.mark.parametrize("stage", ["P01", "P02", "P03", "P07", "P08", "P09", "P10", "P13"])
def test_carry_allowance_is_phase_independent(stage):
    assert _current_functional_body_allowance(carrying(stage=stage).task, .5) == 1.


def test_earned_current_top_contact_before_capture_is_allowed_without_air_requirement():
    built = carrying()
    built.task["physical_evaluator"]["current_legs"]["FR"].update(
        air=False, obstacle_pair_active=True, top_contact=True,
        top_surface_contact=True, within_top_xy=True, top_geometry=True, clearance_m=0.)
    assert _current_functional_body_allowance(built.task, .5) == 1.


def test_rr_current_edge_permission_is_consumed_but_stale_lift_is_not():
    built = carrying("RR")
    rr = built.task["physical_evaluator"]["current_legs"]["RR"]
    rr.update(air=False, obstacle_pair_active=True, contact_mode="FRONT_WALL")
    assert _current_functional_body_allowance(built.task, .5) == 1.
    rr["current_lift_valid"] = False
    assert _current_functional_body_allowance(built.task, .5) == 0.
    rr.update(current_lift_valid=True, motion_continuation_allowed=False)
    assert _current_functional_body_allowance(built.task, .5) == 0.


@pytest.mark.parametrize("status", ["UNVERIFIED_SENSOR", "NONFINITE", None])
def test_unverified_sensor_or_nonfinite_status_is_never_a_carry_allowance(status):
    built = carrying()
    built.task["physical_evaluator"]["physical_evidence_status"] = status
    assert _current_functional_body_allowance(built.task, .5) == 0.


@pytest.mark.parametrize("placed", [False, True])
@pytest.mark.parametrize("clearance, geometry, expected", [
    (-.001, True, 1.), (-.0151, True, 0.), (.0251, True, 0.),
    (0., False, 0.), (None, True, 0.), (float("nan"), True, 0.),
])
def test_top_boolean_alone_does_not_override_current_contact_geometry(placed, clearance, geometry, expected):
    built = captured()
    ev = built.task["physical_evaluator"]
    ev["history"]["placed"]["FR"] = placed
    ev["current_legs"]["FR"].update(clearance_m=clearance, top_geometry=geometry)
    assert _current_functional_body_allowance(built.task, .5) == expected


@pytest.mark.parametrize("fault", [
    "unearned", "inactive_attempt", "grounded", "lateral_lost", "front_roi_lost",
    "insufficient_clearance", "air_contact_conflict", "missing_actual_support",
    "unverified_bearing", "wall_only_support", "invalid_evaluator", "safety_abort",
])
def test_stale_air_and_missing_physical_evidence_cannot_create_body_floor(fault):
    built = carrying()
    ev = built.task["physical_evaluator"]
    row = ev["current_legs"]["FR"]
    if fault == "unearned":
        ev["history"]["active_lift"]["FR"] = False
    elif fault == "inactive_attempt":
        row["active_attempt"] = False
    elif fault == "grounded":
        row["ground_contact"] = True
    elif fault == "lateral_lost":
        row["within_lateral_span"] = False
    elif fault == "front_roi_lost":
        row["front_distance_m"] = -.41
    elif fault == "insufficient_clearance":
        row["clearance_m"] = .0149
    elif fault == "air_contact_conflict":
        row["obstacle_pair_active"] = True
    elif fault == "missing_actual_support":
        for leg in ("FL", "RL"):
            ev["current_legs"][leg]["support"] = False
    elif fault == "unverified_bearing":
        for leg in ("FL", "RL"):
            ev["current_legs"][leg]["bearing_verified"] = False
    elif fault == "wall_only_support":
        for leg in ("FL", "RL"):
            ev["current_legs"][leg].update(ground_contact=False, obstacle_pair_active=True)
    elif fault == "invalid_evaluator":
        ev["valid"] = False
    else:
        ev["termination_reason"] = "SAFETY_ABORT"
    assert _current_functional_body_allowance(built.task, .5) == 0.
    old, new = compare(built)
    assert new == old


def test_unknown_normalized_load_is_neither_support_nor_a_reason_to_discard_verified_forces():
    built = carrying()
    ev = built.task["physical_evaluator"]
    # Producer-consistent: _all_stage_finish marks CONTACT_BEARING_UNVERIFIED
    # when ANY leg prevents normalized load validity, including an airborne
    # target or unrelated fourth leg; separately verified force supports remain.
    assert ev["physical_evidence_status"] == "CONTACT_BEARING_UNVERIFIED"
    assert all(row["load_fraction_valid"] is False for row in ev["current_legs"].values())
    ev["current_legs"]["RR"]["bearing_verified"] = False
    # Target FR and fourth leg RR are not counted; FL and RL are two real supports.
    assert _current_functional_body_allowance(built.task, .5) == 1.
    for leg in ("FL", "RL"):
        ev["current_legs"][leg].update(bearing_force_n=None, bearing_verified=False)
    assert _current_functional_body_allowance(built.task, .5) == 0.


@pytest.mark.parametrize("age_ticks, expected", [(0, 1.), (15, .75), (30, .5), (59, 1/60), (60, 0.), (90, 0.)])
def test_real_capture_event_tick_restores_cost_continuously_without_a_phase_timer(age_ticks, expected):
    built = captured(tick=240+age_ticks, event_tick=240, stage="P13")
    assert _current_functional_body_allowance(built.task, .5) == pytest.approx(expected)
    old, new = compare(built)
    weight = 1.-.8*expected
    for name in ("gravity_attitude", "euler_rate", "angular_acceleration"):
        assert new["cost_components"][name] == pytest.approx(old["cost_components"][name]*weight)


@pytest.mark.parametrize("fault", ["missing_event", "future_event", "float_event", "lost_top", "grounded"])
def test_capture_settle_requires_real_past_event_and_current_capture(fault):
    built = captured(tick=270, event_tick=240)
    ev = built.task["physical_evaluator"]
    if fault == "missing_event":
        ev["history"]["event_ticks"]["placed"].clear()
    elif fault == "future_event":
        ev["history"]["event_ticks"]["placed"]["FR"] = 271
    elif fault == "float_event":
        ev["history"]["event_ticks"]["placed"]["FR"] = 240.
    elif fault == "lost_top":
        ev["current_legs"]["FR"].update(air=True, top_contact=False, top_surface_contact=False)
    else:
        ev["current_legs"]["FR"]["ground_contact"] = True
    assert _current_functional_body_allowance(built.task, .5) == 0.


def test_successor_carry_does_not_get_interrupted_by_predecessor_settle_expiry():
    built = carrying("FL", tick=300)
    ev = built.task["physical_evaluator"]
    ev["history"]["active_lift"]["FR"] = True
    ev["history"]["front_edge_crossed"]["FR"] = True
    ev["history"]["event_ticks"]["placed"]["FR"] = 240
    ev["current_legs"]["FR"].update(active_attempt=True, ground_contact=False,
        top_contact=True, top_surface_contact=True, obstacle_pair_active=True,
        within_top_xy=True, top_geometry=True, front_distance_m=.01, clearance_m=0.)
    assert _current_functional_body_allowance(built.task, .5) == 1.


@pytest.mark.parametrize("fraction", [0., .25, .5, .75, 1.])
def test_only_body_floor_changes_contact_keeps_original_fraction(fraction):
    built = carrying(fraction=fraction)
    metrics = deepcopy(built.metrics)
    for wheel in metrics["wheels"]:
        wheel["slip_speed"] = .4
    built = replace(built, metrics=metrics)
    old, new = compare(built)
    assert old["families"]["contact_motion_quality"] < 0.
    assert new["families"]["contact_motion_quality"] == old["families"]["contact_motion_quality"]
    assert new["cost_components"]["contact_quality"] == old["cost_components"]["contact_quality"]
    if fraction == 1.:
        assert new == old  # In particular, the recorded C0 full-f case is unchanged.


@pytest.mark.parametrize("values", [
    {"capture_settle_window_s": .5},
    {"carry_body_allowance": "typo", "capture_settle_window_s": .5},
    {"carry_body_allowance": CARRY_BODY_ALLOWANCE_MODE},
    {"carry_body_allowance": CARRY_BODY_ALLOWANCE_MODE, "capture_settle_window_s": 0.},
    {"carry_body_allowance": CARRY_BODY_ALLOWANCE_MODE, "capture_settle_window_s": .51},
    {"carry_body_allowance": CARRY_BODY_ALLOWANCE_MODE, "capture_settle_window_s": True},
])
def test_mode_and_window_are_validated_without_relaxing_other_configuration(values):
    with pytest.raises((TypeError, ValueError)):
        _validate_carry_body_allowance(values)


def test_legacy_configuration_and_task_remain_exact():
    _validate_carry_body_allowance({})
    _validate_carry_body_allowance({"carry_body_allowance": CARRY_BODY_ALLOWANCE_MODE, "capture_settle_window_s": .5})
    before, after = frame(), frame(1)
    assert reward(before, after, calculator=calculator()) == reward(before, after, calculator=calculator(False))

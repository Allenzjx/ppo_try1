"""Bounded synthetic RR transform tests; never physical-success evidence."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.ppo.semantic_rr_capture_assist import (
    RR_CAPTURE_ASSIST_MODE, RR_CAPTURE_ASSIST_FEATURE_NAMES,
    RRHipOnlyCaptureAssist, apply_rr_capture_assist_snapshot,
    rr_capture_assist_features, rr_capture_assist_context,
    validate_rr_capture_assist_snapshot,
)

DT = 1. / 120.
ZERO = (0.,) * 12
PREVIOUS = ZERO[:6] + (10., -8.) + ZERO[8:]


def context(assist, tick, **changes):
    initialized = assist.state["initialized"]
    return {"dispatch_physics_tick": tick, "stage_id": "P09", "physical_valid": True,
        "within_top_xy": True, "other_support_count": 2,
        "qualified_RR": True, "crossed_RR": True, "rl_qualified_lift": False,
        "rl_placed_current_top_support": False,
        "issued_nominal_delta_rr_deg": (0., 0.),
        "issued_requested_residual_delta_rr_deg": (0., 0.),
        "release_candidate_rr_deg": (9., -7.),
        "air": True, "ground_contact": False, "top_surface_contact": False,
        "obstacle_pair_active": False, "current_top_bearing": False, "gap_m": .02,
        "hip_actual_deg": assist.state["hip_target_deg"] if initialized else 10.,
        "knee_actual_deg": assist.state["knee_hold_deg"] if initialized else -8., **changes}


def step(assist, tick, previous=None, **changes):
    if previous is None:
        previous = ZERO[:6] + ((assist.state["hip_target_deg"], assist.state["knee_hold_deg"])
                              if assist.state["initialized"] else (10., -8.)) + ZERO[8:]
    return assist.advance(context=context(assist, tick, **changes),
        previous_final_full12=previous, physics_dt_s=DT)


def top():
    return dict(air=False, top_surface_contact=True, obstacle_pair_active=True,
                current_top_bearing=True, gap_m=0.)


def test_public_14_features_and_fresh_identity():
    assist = RRHipOnlyCaptureAssist()
    assert len(RR_CAPTURE_ASSIST_FEATURE_NAMES) == 14
    assert rr_capture_assist_features(assist.snapshot()) == (0.,) * 14
    assert assist.snapshot()["version"] == RR_CAPTURE_ASSIST_MODE == "rr_hip_only_capture_v1"
    assert apply_rr_capture_assist_snapshot(tuple(range(12)), assist.snapshot()) == tuple(range(12))


@pytest.mark.parametrize("change", [dict(stage_id="P08"), dict(stage_id="P10"),
    dict(qualified_RR=False), dict(within_top_xy=False),
    dict(physical_valid=False), dict(other_support_count=1), dict(air=False),
    dict(air=False, obstacle_pair_active=True), dict(rl_placed_current_top_support=True)])
def test_first_acquisition_requires_true_P09_qualified_air_candidate(change):
    assist = RRHipOnlyCaptureAssist()
    step(assist, 1, **change)
    assert assist.snapshot()["mode_name"] == "WAIT"
    assert apply_rr_capture_assist_snapshot(PREVIOUS, assist.snapshot()) == PREVIOUS


def test_anchor_previous_final_then_finite_negative_hip_and_exact_knee_other10():
    assist = RRHipOnlyCaptureAssist()
    step(assist, 1, previous=PREVIOUS)
    assert assist.state["hip_target_deg"] == 10. and assist.state["knee_hold_deg"] == -8.
    assert assist.state["travel_used_deg"] == 0.
    step(assist, 2)
    assert assist.state["hip_target_deg"] == pytest.approx(10. - 2. * DT)
    candidate = tuple(float(i + 20) for i in range(12))
    transformed = apply_rr_capture_assist_snapshot(candidate, assist.snapshot())
    assert transformed[7] == -8.
    assert all(transformed[i] == candidate[i] for i in range(12) if i not in (6, 7))
    step(assist, 3, gap_m=.002)
    assert assist.state["hip_target_deg"] == pytest.approx(10. - 3. * DT)


@pytest.mark.parametrize("changes,reason", [
    (dict(physical_valid=False), 1.), (dict(within_top_xy=False), 2.),
    (dict(other_support_count=1), 3.), (dict(hip_actual_deg=20.), 6.),
    (dict(knee_actual_deg=0.), 6.), (dict(air=False), 7.), (dict(qualified_RR=False), 9.)])
def test_block_holds_target_and_recovery_does_not_reset_budget(changes, reason):
    assist = RRHipOnlyCaptureAssist()
    step(assist, 1); step(assist, 2)
    before = deepcopy(assist.state)
    for tick in range(3, 16):
        step(assist, tick, **changes)
    assert assist.snapshot()["mode_name"] == "BLOCKED"
    assert assist.state["blocked_reason"] == reason
    for key in ("hip_target_deg", "knee_hold_deg", "travel_used_deg", "descent_elapsed_s", "window_elapsed_s"):
        assert assist.state[key] == before[key]
    step(assist, 16)
    assert assist.state["travel_used_deg"] > before["travel_used_deg"]


def test_flat_or_wrong_direction_stops_and_real_gap_improvement_can_resume():
    assist = RRHipOnlyCaptureAssist(); step(assist, 1)
    for tick in range(2, 350):
        step(assist, tick, gap_m=.02 + tick * .000001)
    held = deepcopy(assist.state)
    assert held["blocked_reason"] == 4.
    assert held["travel_used_deg"] == pytest.approx(4.)
    step(assist, 350, gap_m=.02025)  # less than0.2mm below the public local peak
    assert assist.state["hip_target_deg"] == held["hip_target_deg"]
    step(assist, 351, gap_m=.019)
    assert assist.state["hip_target_deg"] < held["hip_target_deg"]
    assert assist.state["descent_elapsed_s"] > held["descent_elapsed_s"]


@pytest.mark.parametrize("contact", [dict(top_surface_contact=True, obstacle_pair_active=True),
    dict(obstacle_pair_active=True), dict(ground_contact=True)])
def test_any_real_contact_stops_pressing_without_fabricating_bearing(contact):
    assist = RRHipOnlyCaptureAssist(); step(assist, 1); step(assist, 2)
    before = deepcopy(assist.state)
    step(assist, 3, air=False, **contact)
    assert assist.snapshot()["mode_name"] == "HOLD"
    assert assist.state["hip_target_deg"] == before["hip_target_deg"]
    assert assist.state["contact_seen"] == float(bool(contact.get("top_surface_contact")))
    step(assist, 4, stage_id="P10", air=False, **contact)
    assert assist.snapshot()["mode_name"] in ("HOLD", "BLOCKED")  # no invented current bearing


def test_real_bearing_follow_retains_capture_target_not_previous_FINAL_or_absolute_candidate():
    assist = RRHipOnlyCaptureAssist(); step(assist, 1); step(assist, 2, **top())
    previous = ZERO[:6] + (9.8, -8.1) + ZERO[8:]
    candidate = ZERO[:6] + (9., -7.) + (.3,) * 4
    step(assist, 3, previous=previous, stage_id="P10", **top())
    assert assist.snapshot()["mode_name"] == "CAPTURED_FOLLOW"
    assert apply_rr_capture_assist_snapshot(candidate, assist.snapshot())[6:8] == (10., -8.)
    values = []
    for tick in range(4, 94):
        step(assist, tick, stage_id="P10", **top())
        values.append(apply_rr_capture_assist_snapshot(candidate, assist.snapshot())[6])
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert values == [10.] * 90
    assert assist.snapshot()["mode_name"] == "CAPTURED_FOLLOW"


@pytest.mark.parametrize("long_contact", [False, True])
def test_true_bearing_loss_keeps_integrator_and_spends_only_remaining_feedback(long_contact):
    assist = RRHipOnlyCaptureAssist(); step(assist, 1); step(assist, 2)
    spent = assist.state["travel_used_deg"]
    step(assist, 3, stage_id="P10", **top())
    end = 94 if long_contact else 6
    for tick in range(4, end):
        step(assist, tick, stage_id="P10", **top())
    previous = ZERO[:6] + (9.2, -7.5) + ZERO[8:]
    pending = assist.state["hip_target_deg"]
    step(assist, end, previous=previous, stage_id="P11", gap_m=.01)
    assert assist.state["hip_target_deg"] == pytest.approx(pending - 2*DT)
    assert assist.state["knee_hold_deg"] == -8.
    assert assist.state["release_fraction"] == 0.
    assert assist.state["travel_used_deg"] == pytest.approx(spent + 2*DT)
    step(assist, end + 1, stage_id="P11", gap_m=.009)
    assert assist.state["hip_target_deg"] == pytest.approx(pending - 4*DT)
    assert assist.state["hip_entry_deg"] == 10.


def test_RL_lift_does_not_retire_and_real_placement_release_cannot_reacquire():
    assist = RRHipOnlyCaptureAssist(); step(assist, 1); step(assist, 2)
    step(assist, 3, rl_qualified_lift=True)
    assert assist.state["retired"] == 0.
    step(assist, 4, stage_id="P13", rl_placed_current_top_support=True, **top())
    assert assist.state["retired"] == 1. and assist.snapshot()["mode_name"] == "RELEASE"
    for tick in range(5, 110):
        step(assist, tick, stage_id="P13", rl_placed_current_top_support=True, **top())
    assert assist.snapshot()["mode_name"] == "RELEASED"
    before = deepcopy(assist.state)
    step(assist, 110, stage_id="P09")
    assert assist.state == before  # even a malformed phase regression grants no new budget/owner


def test_P12_P13_do_not_start_new_capture_or_claim_bearing():
    assist = RRHipOnlyCaptureAssist()
    for tick in range(1, 110):
        step(assist, tick, stage_id="P12")
    assert assist.snapshot()["mode_name"] == "WAIT"
    before = deepcopy(assist.state)
    step(assist, 110, stage_id="P13")
    assert assist.state == before


def test_contact_losses_and_above_band_progress_cannot_replenish_total_reserve():
    assist = RRHipOnlyCaptureAssist(); step(assist, 1)
    # v10 allows these real, tracked peak drops to earn the original reserve
    # above25 mm. Contact pauses still cannot recharge its cumulative budget.
    for tick in range(2, 7000):
        if tick % 7 == 0:
            step(assist, tick, air=False, obstacle_pair_active=True, gap_m=.1)
        else:
            step(assist, tick, gap_m=.1 - (tick % 7) * .0003)
    assert assist.state["travel_used_deg"] == pytest.approx(52.)
    assert assist.state["descent_elapsed_s"] == pytest.approx(42.)
    assert assist.state["hip_target_deg"] == pytest.approx(-10.)
    assert assist.state["knee_hold_deg"] == pytest.approx(24.)
    assert not assist.state["contact_seen"]  # wall/unknown contact is not TOP
    budget = (assist.state["travel_used_deg"], assist.state["descent_elapsed_s"])
    step(assist, 7000, gap_m=.09)
    assert assist.snapshot()["mode_name"] == "BLOCKED"
    step(assist, 7001, **top())
    step(assist, 7002, gap_m=.01)
    step(assist, 7003, gap_m=.0097)  # fresh drop, but no <=1 mm terminal credit
    assert assist.snapshot()["mode_name"] == "BLOCKED"
    assert (assist.state["travel_used_deg"], assist.state["descent_elapsed_s"]) == budget


def test_twelve_second_exposure_limit_is_independent_of_contact_and_progress():
    assist = RRHipOnlyCaptureAssist(); step(assist, 1, gap_m=.002)
    for tick in range(2, 1700):
        if tick % 8 == 0:
            step(assist, tick, air=False, obstacle_pair_active=True, gap_m=.002)
        else:
            step(assist, tick, gap_m=.002 - (tick % 8) * .0001)
    assert assist.state["descent_elapsed_s"] == pytest.approx(12.)
    assert assist.state["travel_used_deg"] == pytest.approx(12.)
    step(assist, 1700, gap_m=.001)
    assert assist.state["blocked_reason"] == 8.


def test_original_two_degree_lower_reserve_is_not_relaxed():
    assist = RRHipOnlyCaptureAssist()
    step(assist, 1, previous=ZERO[:6] + (-132.99, -8.) + ZERO[8:], hip_actual_deg=-132.99)
    step(assist, 2)
    assert assist.state["hip_target_deg"] == -133.
    step(assist, 3)
    assert assist.state["blocked_reason"] == 5.


def test_strict_snapshot_replay_is_same_without_current_policy_sample():
    assist = RRHipOnlyCaptureAssist(); step(assist, 1); step(assist, 2)
    before = assist.snapshot(); replay = RRHipOnlyCaptureAssist.from_snapshot(before)
    expected = step(assist, 3, gap_m=.019)
    actual = step(replay, 3, gap_m=.019)
    assert expected == actual
    frozen = replay.snapshot()
    one = apply_rr_capture_assist_snapshot((100.,) * 12, frozen)
    two = apply_rr_capture_assist_snapshot(ZERO, frozen)
    assert one[6:8] == two[6:8] and replay.snapshot() == frozen


@pytest.mark.parametrize("key,value", [("mode", .5), ("mode", True), ("retired", 2.),
    ("hip_target_deg", float("nan")), ("travel_used_deg", 20.1),
    ("version", "wrong"), ("owners", ["wrong", "wrong"]), ("active", 1),
    ("feature_names", [])])
def test_snapshot_tampering_fails_closed(key, value):
    assist = RRHipOnlyCaptureAssist(); step(assist, 1)
    snapshot = assist.snapshot(); snapshot[key] = value
    with pytest.raises(ValueError):
        validate_rr_capture_assist_snapshot(snapshot)


@pytest.mark.parametrize("change", [dict(dispatch_physics_tick=4), dict(stage_id="P00"),
    dict(air=1), dict(gap_m=float("nan")), dict(knee_actual_deg=None),
    dict(current_top_bearing=True)])
def test_invalid_advance_is_transactional(change):
    assist = RRHipOnlyCaptureAssist(); step(assist, 1)
    before = assist.snapshot()
    with pytest.raises(ValueError):
        assist.advance(context=context(assist, 2, **change), previous_final_full12=PREVIOUS, physics_dt_s=DT)
    assert assist.snapshot() == before


def test_context_uses_current_verified_load_not_historical_placement():
    support = dict(support=True, bearing_verified=True, bearing_force_n=10., air=False,
                   ground_contact=True, top_surface_contact=False)
    rr = dict(within_top_xy=True, clearance_m=.05, air=True, ground_contact=False, current_lift_valid=True,
              top_surface_contact=False, obstacle_pair_active=False, contact_surface="NONE",
              top_contact=False, support=False, bearing_verified=True, bearing_force_n=0.)
    task = {"physical_evaluator": dict(valid=True, physics_tick=12,
        current_legs={"FL": dict(support), "FR": dict(support), "RL": dict(support), "RR": rr},
        history={"active_lift": {"RR": True, "RL": False}, "front_edge_crossed": {"RR": True}, "placed": {"RR": True}})}
    joints = {SERVO_ORDER[6]: {"position_deg": 10.}, SERVO_ORDER[7]: {"position_deg": -8.}}
    args = dict(task=task, observation={"joints": joints}, source_frame=SimpleNamespace(state_id="P10"),
                physics_tick=13, previous_ack={}, support_spec={"force_noise_floor_n": 5.})
    c = rr_capture_assist_context(**args)
    assert c["other_support_count"] == 3 and not c["current_top_bearing"]
    assert "placed_RR" not in c
    rr.update(air=False, contact_surface="TOP", top_contact=True, top_surface_contact=True,
              obstacle_pair_active=True, support=True, bearing_force_n=10.)
    assert rr_capture_assist_context(**args)["current_top_bearing"]
    rr["bearing_verified"] = False
    assert not rr_capture_assist_context(**args)["current_top_bearing"]
    task["physical_evaluator"]["current_legs"]["FR"]["bearing_force_n"] = 0.
    assert rr_capture_assist_context(**args)["other_support_count"] == 2


@pytest.mark.parametrize("air,ground", [(False, True), (True, False)])
def test_old_lift_history_after_reground_cannot_initialize_without_current_valid_lift(air, ground):
    bearing = dict(support=True, bearing_verified=True, bearing_force_n=10., air=False,
                   ground_contact=True, top_surface_contact=False)
    rr = dict(within_top_xy=True, clearance_m=.05, air=air, ground_contact=ground,
              top_surface_contact=False, obstacle_pair_active=False, current_lift_valid=False)
    task = {"physical_evaluator": dict(valid=True,
        current_legs={"FL": dict(bearing), "FR": dict(bearing), "RL": dict(bearing), "RR": rr},
        history={"active_lift": {"RR": True}, "front_edge_crossed": {"RR": True}, "placed": {"RR": True}})}
    args = dict(task=task, observation={"joints": {
        SERVO_ORDER[6]: {"position_deg": 10.}, SERVO_ORDER[7]: {"position_deg": -8.}}},
        source_frame=SimpleNamespace(state_id="P09"), physics_tick=1,
        support_spec={"force_noise_floor_n": 5.})
    before = deepcopy(task)
    c = rr_capture_assist_context(**args)
    assert not c["qualified_RR"] and c["crossed_RR"]
    assist = RRHipOnlyCaptureAssist()
    assist.advance(context=c, previous_final_full12=PREVIOUS, physics_dt_s=DT)
    assert assist.snapshot()["mode_name"] == "WAIT"
    assert task == before


def test_current_bearing_follow_does_not_need_air_lift_predicate_to_remain_true():
    assist = RRHipOnlyCaptureAssist(); step(assist, 1)
    step(assist, 2, stage_id="P10", qualified_RR=False, **top())
    assert assist.snapshot()["mode_name"] == "CAPTURED_FOLLOW"


def test_mixed_GROUND_TOP_cannot_be_current_TOP_bearing_or_start_release():
    support = dict(support=True, bearing_verified=True, bearing_force_n=10., air=False,
                   ground_contact=True, top_surface_contact=False)
    rr = dict(support, within_top_xy=True, clearance_m=0., current_lift_valid=True,
              top_surface_contact=True, obstacle_pair_active=True, contact_surface="TOP", top_contact=True)
    task = {"physical_evaluator": dict(valid=True,
        current_legs={"FL": dict(support), "FR": dict(support), "RL": dict(support), "RR": rr},
        history={"active_lift": {"RR": True}, "front_edge_crossed": {"RR": True}})}
    c = rr_capture_assist_context(task=task, observation={"joints": {
        SERVO_ORDER[6]: {"position_deg": 10.}, SERVO_ORDER[7]: {"position_deg": -8.}}},
        source_frame=SimpleNamespace(state_id="P10"), physics_tick=2,
        support_spec={"force_noise_floor_n": 5.})
    assert not c["current_top_bearing"]
    assist = RRHipOnlyCaptureAssist(); step(assist, 1)
    assist.advance(context=c, previous_final_full12=PREVIOUS, physics_dt_s=DT)
    assert assist.snapshot()["mode_name"] == "HOLD"
    with pytest.raises(ValueError, match="contradicts"):
        step(assist, 3, stage_id="P10", ground_contact=True, **top())


@pytest.mark.parametrize("state_change,reason", [
    ({"travel_used_deg": 40. - DT, "descent_elapsed_s": 30. - DT}, 5.),
    ({"travel_used_deg": 12. - DT, "descent_elapsed_s": 12. - DT}, 8.),
    ({"window_elapsed_s": 2. - DT}, 4.),
    ({"hip_target_deg": -133. + DT}, 5.),
])
def test_final_budget_consuming_step_immediately_advertises_BLOCKED(state_change, reason):
    assist = RRHipOnlyCaptureAssist(); step(assist, 1)
    assist.state.update(state_change)
    receipt = step(assist, 2)
    assert receipt["state_after"]["mode_name"] == "BLOCKED"
    assert receipt["state_after"]["blocked_reason"] == reason
    assert not (receipt["state_after"]["mode_name"] == "DESCEND")
    target = assist.state["hip_target_deg"]
    step(assist, 3)
    assert assist.state["hip_target_deg"] == target


@pytest.mark.parametrize("key,value", [("travel_used_deg", 20.), ("descent_elapsed_s", 12.),
    ("window_elapsed_s", 2.), ("hip_target_deg", -133.), ("retired", 1.)])
def test_forged_DESCEND_snapshot_cannot_hide_exhaustion_or_retirement(key, value):
    assist = RRHipOnlyCaptureAssist(); step(assist, 1)
    snapshot = assist.snapshot(); snapshot[key] = value
    with pytest.raises(ValueError, match="DESCEND"):
        validate_rr_capture_assist_snapshot(snapshot)


def test_reacquisition_with_consumed_total_budget_never_advertises_DESCEND():
    assist = RRHipOnlyCaptureAssist(); step(assist, 1)
    assist.state.update(mode=3., blocked_reason=5., travel_used_deg=40., descent_elapsed_s=30.)
    step(assist, 2, stage_id="P10", **top())
    for tick in range(3, 93):
        step(assist, tick, stage_id="P10", **top())
    assert assist.snapshot()["mode_name"] == "CAPTURED_FOLLOW"
    step(assist, 93, stage_id="P11")
    assert assist.snapshot()["mode_name"] == "BLOCKED"
    assert assist.state["blocked_reason"] == 10. and assist.state["travel_used_deg"] == 40.

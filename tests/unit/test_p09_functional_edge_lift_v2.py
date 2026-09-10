"""Contiguous measured counterexamples for functional RR lift, not hover time."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from test_semantic_supervisor import advance
from test_semantic_all_stage_physical_acceptance import (
    new_spec, new_observation, set_leg, early_top_place,
)
from wlr50_clean.infrastructure.command_batch import servo_limits_deg
from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator, TaskStageSupervisor, load_task_spec
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_DIM

ROOT = Path(__file__).resolve().parents[2]


def spec(*, roles=False):
    result = load_task_spec(ROOT / "configs/ppo_all_stage_acceptance_v1/stage_task_spec.yaml")
    result["p09_lift_semantics"] = "functional_lift_edge_v2"
    if not roles:
        result.pop("transfer_roles", None)
    return result


def start(*, roles=False, front_placed=False):
    ev = TaskEvaluator(spec=spec(roles=roles))
    obs = new_observation()
    obs["center_of_mass"] = dict(valid=True, position_w_m=[.30, 0., .10],
        velocity_w_m_s=[0., 0., 0.], total_mass_kg=3., included_bodies=["synthetic"])
    ev.observe(obs)
    if front_placed:
        for leg in ("FR", "FL"):
            obs = early_top_place(ev, obs, leg)
    return ev, obs


def lift(ev, obs, *, surface="AIR"):
    # RR's own hip/knee never need to move: the other legs can lift its wheel.
    initial = obs["joints"]["front_left_hip"]["position_deg"]
    for i, bottom in enumerate((.004, .012), 1):
        obs = advance(obs)
        set_leg(obs, "RR", bottom=bottom, surface=surface)
        obs["joints"]["front_left_hip"].update(position_deg=initial+1.5*i,
                                                 command_deg=initial+1.5*i+.6)
        ev.observe(obs)
    return obs


def supervisor(ev, tmp_path, phase):
    path = tmp_path / "functional.yaml"
    path.write_text(yaml.safe_dump(ev.spec, sort_keys=False), encoding="utf-8")
    return TaskStageSupervisor(path, evaluator=ev, initial_stage_id=phase)


@pytest.mark.parametrize("later_ticks", [0, 17, 145])
def test_short_or_long_supported_air_does_not_have_a_one_second_boundary(later_ticks):
    ev, obs = start()
    obs = lift(ev, obs)
    for _ in range(later_ticks):
        obs = advance(obs)
        ev.observe(obs)
    rr = ev.snapshot["current_legs"]["RR"]
    assert rr["initial_lift_observed"] and rr["lift_established"] and rr["current_lift_valid"]
    assert rr["motion_continuation_allowed"] and rr["contact_mode"] == "AIR"
    assert rr["edge_contact_duration_s"] == 0.
    assert rr["durations_are_diagnostic_only"]
    assert not ev.snapshot["history"]["placed"]["RR"]


def test_other_joint_motion_can_establish_rr_lift_without_rr_joint_change():
    ev, obs = start()
    obs = lift(ev, obs)
    assert obs["joints"]["rear_right_hip"]["position_deg"] == 0.
    assert obs["joints"]["rear_right_knee"]["position_deg"] == 0.
    assert ev.snapshot["current_legs"]["RR"]["current_lift_valid"]
    assert ev.snapshot["history"]["active_lift"]["RR"]


@pytest.mark.parametrize("surface", ["WALL", "AMBIGUOUS"])
def test_active_edge_lift_can_establish_without_being_relabelled_air(surface):
    ev, obs = start()
    obs = lift(ev, obs, surface=surface)
    rr = ev.snapshot["current_legs"]["RR"]
    assert rr["lift_established"] and rr["current_lift_valid"]
    assert rr["motion_continuation_allowed"] and not rr["air"]
    assert rr["contact_mode"] == ("FRONT_WALL" if surface == "WALL" else "OBSTACLE_AMBIGUOUS")
    assert rr["air_duration_s"] == 0. and rr["edge_contact_duration_s"] > 0.
    assert not rr["front_edge_crossed"] and not rr["placed_on_top"]


def test_light_edge_contact_preserves_attempt_and_long_stuck_state_is_not_time_qualified():
    ev, obs = start()
    obs = lift(ev, obs)
    first_q = ev.snapshot["history"]["event_ticks"]["active_lift"]["RR"]
    obs = advance(obs)
    set_leg(obs, "RR", surface="WALL", bottom=.013)
    rr = ev.observe(obs)["current_legs"]["RR"]
    assert rr["current_lift_valid"] and rr["lift_established"] and not rr["air"]
    for _ in range(145):
        obs = advance(obs)
        rr = ev.observe(obs)["current_legs"]["RR"]
    assert rr["edge_contact_duration_s"] > 1.
    assert rr["lift_established"] and not rr["current_lift_valid"]
    assert not rr["edge_adjustment_response_observed"]
    assert rr["motion_continuation_allowed"]  # Allow bounded probing, no parked wait gate.
    assert ev.snapshot["history"]["event_ticks"]["active_lift"]["RR"] == first_q
    assert not ev.snapshot["history"]["placed"]["RR"]


def test_single_frame_dropout_bounce_and_same_tick_cache_do_not_establish():
    ev, obs = start()
    obs = advance(obs)
    set_leg(obs, "RR", bottom=.012)
    obs["joints"]["front_left_hip"].update(position_deg=3., command_deg=3.6)
    ev.observe(obs)
    for _ in range(10):
        snap = ev.observe(deepcopy(obs))
    assert not snap["current_legs"]["RR"]["lift_established"]
    assert snap["current_legs"]["RR"]["consecutive_air_samples"] == 1
    obs = advance(obs)
    set_leg(obs, "RR", bottom=0.)
    assert not ev.observe(obs)["current_legs"]["RR"]["lift_established"]


def test_no_joint_demand_or_measured_response_never_qualifies_by_waiting():
    ev, obs = start()
    for i in range(150):
        obs = advance(obs)
        set_leg(obs, "RR", bottom=.012, surface="WALL")
        obs["wheels"]["rear_right_ankle"]["command_rad_s"] = .3
        rr = ev.observe(obs)["current_legs"]["RR"]
    assert not rr["lift_established"] and not rr["current_lift_valid"]
    assert rr["motion_continuation_allowed"] and rr["edge_contact_duration_s"] > 1.


def test_p08_edge_lift_passes_to_p09_with_same_attempt_and_existing_observation_slot(tmp_path):
    ev, obs = start(front_placed=True)
    obs = lift(ev, obs, surface="WALL")
    original_q = ev.snapshot["history"]["event_ticks"]["active_lift"]["RR"]
    sup = supervisor(ev, tmp_path, "P08")
    for _ in range(8):
        obs = advance(obs)
        task = sup.observe_and_update(obs)
        if task["stage_id"] == "P09":
            break
    assert task["stage_id"] == "P09" and task["termination_reason"] is None
    assert task["active_lift_history"]["RR"]
    assert task["history"]["event_ticks"]["active_lift"]["RR"] == original_q
    assert ROLE_OBSERVATION_DIM == 372
    assert len(task["active_lift_history"]) == 4
    assert not task["history"]["placed"]["RR"]


def test_ground_revokes_current_attempt_and_fresh_retry_is_allowed():
    ev, obs = start()
    obs = lift(ev, obs)
    obs = advance(obs)
    set_leg(obs, "RR", bottom=0., surface="GROUND")
    rr = ev.observe(obs)["current_legs"]["RR"]
    assert not rr["lift_established"] and not rr["current_lift_valid"]
    assert not ev.snapshot["history"]["active_lift"]["RR"]
    obs = lift(ev, obs, surface="WALL")
    assert ev.snapshot["current_legs"]["RR"]["current_lift_valid"]


def test_real_early_top_placement_needs_no_air_hold_and_p10_needs_current_usability(tmp_path):
    ev, obs = start(front_placed=True)
    obs = early_top_place(ev, obs, "RR")
    sup = supervisor(ev, tmp_path, "P09")
    assert sup.predicate("placed_RR", ev.snapshot) == 1.
    assert sup.entry_report("P10")["valid"]
    assert ev.snapshot["current_legs"]["RR"]["air_duration_s"] == 0.
    obs = advance(obs)
    set_leg(obs, "RR", x=.4, bottom=0., surface="GROUND")
    ev.observe(obs)
    assert ev.snapshot["history"]["front_edge_crossed"]["RR"]
    assert ev.snapshot["history"]["placed"]["RR"]
    assert not ev.snapshot["current_legs"]["RR"]["current_lift_valid"]
    assert sup.predicate("placed_RR", ev.snapshot) < 1.
    assert "placed_RR" in sup.entry_report("P10")["reasons"]


def test_unknown_load_preserves_independent_geometry_and_motion_not_false_unloading():
    ev, obs = start(roles=True)
    for i in range(1, 25):
        obs = advance(obs)
        obs["center_of_mass"]["position_w_m"][0] += .0002
        obs["center_of_mass"]["velocity_w_m_s"] = [.024, 0., 0.]
        obs["joints"]["front_left_hip"].update(position_deg=.2*i, command_deg=.2*i+.6)
        if i >= 18:
            set_leg(obs, "RR", bottom=.004, surface="AMBIGUOUS")
        snap = ev.observe(obs)
    role = snap["transfer_roles"]["RR"]
    assert role["valid"] and not role["normalized_load_valid"] and not role["load_change_valid"]
    assert role["transfer_direction_context"]["reference_tick"] == 0
    assert role["transfer_direction_context"]["com_toward_receiver_m"] > 0.
    assert role["transfer_direction_context"]["load_fraction_change"] is None
    assert role["motion_fraction"] > 0.
    assert not role["transfer_ready"] and role["transfer_progress"] == 0.


@pytest.mark.parametrize("fault", ["fall", "actual_joint_limit", "body_collision"])
def test_physical_safety_and_actual_joint_limits_remain_authoritative(fault):
    ev, obs = start()
    obs = lift(ev, obs)
    obs = advance(obs)
    if fault == "fall":
        obs["imu"]["projected_gravity_b"] = [0., 0., -.2]
    elif fault == "actual_joint_limit":
        obs["joints"]["rear_right_knee"]["position_deg"] = servo_limits_deg("rear_right_knee")[1]+.01
        obs["joints"]["rear_right_knee"]["command_deg"] = 0.
    else:
        obs["body_collision"]["detected"] = True
    snap = ev.observe(obs)
    assert snap["termination_reason"] is not None and not snap["success"]
    assert not snap["current_legs"]["RR"]["current_lift_valid"]
    assert not snap["current_legs"]["RR"]["motion_continuation_allowed"]
    if fault == "actual_joint_limit":
        assert snap["termination_source"] == "HARD_JOINT_LIMIT"


def test_positive_wheel_only_wall_ascent_is_not_legal_active_lift():
    ev, obs = start()
    for x, bottom in ((.47, .01), (.48, .025), (.501, .049)):
        obs = advance(obs)
        set_leg(obs, "RR", x=x, bottom=bottom, surface="WALL")
        obs["wheels"]["rear_right_ankle"]["command_rad_s"] = .3
        snap = ev.observe(obs)
    assert snap["termination_reason"] == "TASK_FAILURE_WHEEL_ONLY_CLIMB"
    assert not snap["history"]["placed"]["RR"]

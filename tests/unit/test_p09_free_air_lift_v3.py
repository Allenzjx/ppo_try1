"""Measured-input RR v3 component checks, not a new physical-success claim."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from test_p09_functional_edge_lift_v2 import spec as old_spec, lift
from test_semantic_all_stage_physical_acceptance import new_observation, set_leg
from test_semantic_supervisor import advance
from wlr50_clean.ppo.semantic_supervisor import (
    TaskEvaluator, TaskStageSupervisor, P09_FREE_AIR_LIFT_MODE, load_task_spec, SemanticObservationError,
)
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_DIM


def start(roles=False):
    spec = old_spec(roles=roles)
    spec["p09_lift_semantics"] = P09_FREE_AIR_LIFT_MODE
    ev, obs = TaskEvaluator(spec=spec), new_observation()
    obs["center_of_mass"] = dict(valid=True, position_w_m=[.30, 0., .10],
        velocity_w_m_s=[0., 0., 0.], total_mass_kg=3., included_bodies=["synthetic"])
    ev.observe(obs)
    return ev, obs


def move(ev, obs, bottom, surface="AIR"):
    obs = advance(obs)
    set_leg(obs, "RR", bottom=bottom, surface=surface)
    q = obs["joints"]["front_left_hip"]["position_deg"] + 1.5
    obs["joints"]["front_left_hip"].update(position_deg=q, command_deg=q+.6)
    return obs, ev.observe(obs)["current_legs"]["RR"]


def test_current_free_air_3mm_initial_then_8mm_qualified_without_rr_joint_motion():
    ev, obs = start()
    obs, rr = move(ev, obs, .0035)
    assert not rr["initial_now"]  # Existing two-sample rejection is retained.
    obs, rr = move(ev, obs, .004)
    assert rr["initial_now"] and not rr["current_lift_valid"]
    obs, rr = move(ev, obs, .009)
    assert rr["lift_established_now"] and rr["current_lift_valid"]
    assert rr["unsupported_free_lift_m"] == pytest.approx(.009)
    assert rr["ground_relative_lift_m"] == pytest.approx(.009)
    assert obs["joints"]["rear_right_hip"]["position_deg"] == 0.
    assert obs["joints"]["rear_right_knee"]["position_deg"] == 0.


@pytest.mark.parametrize("surface", ["WALL", "AMBIGUOUS", "TOP"])
def test_contact_rise_is_neither_new_initial_nor_new_qualification(surface):
    ev, obs = start()
    obs = lift(ev, obs, surface=surface)
    rr = ev.snapshot["current_legs"]["RR"]
    assert not rr["initial_lift_observed"] and not rr["lift_established"]
    assert not rr["current_lift_valid"] and not ev.snapshot["history"]["active_lift"]["RR"]
    assert rr["unsupported_free_lift_m"] is None
    assert rr["ground_relative_lift_m"] == pytest.approx(.012)
    assert rr["motion_continuation_allowed"]


def test_ground_revokes_then_wall_rise_and_late_air_cannot_reuse_old_height():
    ev, obs = start()
    obs = lift(ev, obs)
    assert ev.snapshot["current_legs"]["RR"]["current_lift_valid"]
    obs, rr = move(ev, obs, 0., "GROUND")
    assert not rr["current_lift_valid"] and not ev.snapshot["history"]["active_lift"]["RR"]
    for z in (.008, .025, .045):
        obs, rr = move(ev, obs, z, "AMBIGUOUS")
    for z in (.0452, .0456):
        obs, rr = move(ev, obs, z)
    assert rr["ground_relative_lift_m"] > .04
    assert rr["unsupported_free_lift_m"] == pytest.approx(.0006)
    assert not rr["initial_now"] and not rr["current_lift_valid"]
    obs, rr = move(ev, obs, .054)
    assert rr["current_lift_valid"]  # Genuine new unsupported rise can recover.


@pytest.mark.parametrize("surface", ["WALL", "AMBIGUOUS", "TOP"])
def test_already_earned_lift_can_continue_through_real_light_contact(surface):
    ev, obs = start()
    obs = lift(ev, obs)
    obs, rr = move(ev, obs, .013, surface)
    assert rr["current_lift_valid"] and rr["lift_established"]
    assert not rr["lift_established_now"] and rr["unsupported_free_lift_m"] is None
    assert ev.snapshot["termination_reason"] is None


def test_other_supports_do_not_require_fl_contact_and_unknown_is_not_unloaded():
    ev, obs = start(roles=True)
    set_leg(obs, "FL", bottom=.07)
    obs = lift(ev, obs)
    rr = ev.snapshot["current_legs"]["RR"]
    assert rr["current_lift_valid"] and rr["observed_other_support_contacts"] == ("FR", "RL")
    assert ev.snapshot["current_legs"]["FL"]["bearing_force_n"] == 0.
    obs, rr = move(ev, obs, .013, "AMBIGUOUS")
    assert not rr["load_fraction_valid"]
    assert ev.snapshot["transfer_roles"]["RR"]["valid"]
    assert not ev.snapshot["transfer_roles"]["RR"]["normalized_load_valid"]


def test_bottom_derivative_is_adjacent_physical_geometry_not_axle_speed_or_target():
    ev, obs = start()
    assert ev.snapshot["current_legs"]["RR"]["wheel_bottom_vz_m_s"] is None
    obs["wheels"]["rear_right_ankle"].update(velocity_rad_s=7., command_rad_s=-3.)
    obs, rr = move(ev, obs, .004)
    assert rr["wheel_bottom_vz_m_s"] == pytest.approx(.48)
    assert rr["wheel_bottom_vz_interval_s"] == pytest.approx(1/120)
    assert ev.observe(deepcopy(obs))["current_legs"]["RR"] == rr
    obs = advance(advance(obs))
    with pytest.raises(SemanticObservationError, match="contiguous"):
        ev.observe(obs)  # Existing public API rejects a physical history hole.
    rr = ev._observe_functional_rr(now=obs["simulation_time_s"], tick=obs["physics_tick"],
        bottom=(.3,0.,.004), center=(.3,0.,.054), ground=False, obstacle_contact=False,
        top_surface=False, surface="NONE", active_response=False,
        current_other_legs={leg: row for leg,row in ev.snapshot["current_legs"].items() if leg != "RR"})
    assert rr["wheel_bottom_vz_m_s"] is None and rr["unsupported_free_lift_m"] is None


def test_v3_snapshot_and_existing_observation_slots_keep_current_qualification(tmp_path):
    ev, obs = start()
    obs = lift(ev, obs)
    path = tmp_path / "v3.yaml"
    path.write_text(yaml.safe_dump(ev.spec), encoding="utf8")
    assert load_task_spec(path)["p09_lift_semantics"] == P09_FREE_AIR_LIFT_MODE
    sup = TaskStageSupervisor(path, evaluator=ev, initial_stage_id="P08")
    task = sup.observe_and_update(obs)
    assert task["p09_lift_semantics"] == P09_FREE_AIR_LIFT_MODE
    assert task["active_lift_history"]["RR"] and ROLE_OBSERVATION_DIM == 372
    assert sup.predicate("lifted_RR", ev.snapshot) == 1.
    assert sup._current_lift_credit("RR", ev.snapshot) > 0.


@pytest.mark.parametrize("fault", ["body_collision", "one_other_support", "unknown_other_bearing"])
def test_current_body_and_verified_other_supports_remain_required(fault):
    ev, obs = start()
    if fault == "body_collision": obs["body_collision"]["detected"] = True
    elif fault == "one_other_support":
        for leg in ("FL", "RL"): set_leg(obs, leg, bottom=.06)
    else:
        for leg in ("FL", "RL"): set_leg(obs, leg, bottom=.02, surface="AMBIGUOUS")
    obs = lift(ev, obs)
    assert not ev.snapshot["current_legs"]["RR"]["current_lift_valid"]


def test_backend_existing_geometry_accepts_explicit_free_air_mode_without_geometry_changes(tmp_path):
    from test_isaac_fsm_backend import FakeRuntime
    from wlr50_clean.ppo.semantic_backend import SemanticIsaacBackend
    root = Path(__file__).resolve().parents[2]
    cfg = root / "configs/ppo_task_first_recovery_v1"
    spec = load_task_spec(cfg / "stage_task_spec.yaml")
    spec["p09_lift_semantics"] = P09_FREE_AIR_LIFT_MODE
    path = tmp_path / "backend_v3.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf8")
    backend = SemanticIsaacBackend(dependencies=FakeRuntime().dependencies(),
        execution_profile=cfg / "execution_profile.yaml", task_spec_path=path)
    assert backend._functional_geometry_parameters["minimum_lift_gain_m"] == .008
    assert backend._functional_geometry_parameters["mode"] == "contact_aware_bounded_rr_nominal_v3"

"""RR timeout integration with contiguous synthetic sensing, never Isaac credit."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_semantic_all_stage_physical_acceptance import early_top_place, set_leg
from test_semantic_p05_capture_handoff_v2 import measured
from test_semantic_supervisor import advance
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.ppo.semantic_rr_capture_assist import (
    RRHipOnlyCaptureAssist, rr_capture_assist_context,
)
from wlr50_clean.ppo.semantic_supervisor import (
    TaskEvaluator, TaskStageSupervisor, load_task_spec,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/ppo_rr_capture_then_rl_transfer_v1/stage_task_spec.yaml"


@pytest.fixture
def airborne_rr():
    spec = load_task_spec(SPEC)
    ev, obs = TaskEvaluator(spec=spec), measured()
    ev.observe(obs)
    for leg in ("FR", "FL"):
        obs = early_top_place(ev, obs, leg)
    # Earn RR Q/C through the real evaluator. Other-joint motion provides
    # whole-body actuation evidence; RR's own joints need not change first.
    for bottom, x in ((.004, .2), (.012, .2), (.075, .53)):
        obs = advance(obs)
        set_leg(obs, "RR", bottom=bottom, x=x)
        fl = obs["joints"]["front_left_hip"]
        fl.update(position_deg=fl["position_deg"] + 1.5,
                  command_deg=fl["position_deg"] + 2.1)
        ev.observe(obs)
    assert ev.snapshot["current_legs"]["RR"]["current_lift_valid"]
    assert ev.snapshot["history"]["front_edge_crossed"]["RR"]
    assert not ev.snapshot["history"]["placed"]["RR"]
    assert ev.snapshot["history"]["placed"]["FL"]  # No separate FL pending waiver.
    sup = TaskStageSupervisor(SPEC, evaluator=ev, initial_stage_id="P09")
    task = sup.observe_and_update(obs)
    assert task["stage_id"] == "P09" and not task["fl_capture_pending"]
    return sup, obs


def feedback(sup, obs, tick, mode="DESCEND"):
    """Make an actual production assist snapshot, without an actuator/simulator."""
    assist = RRHipOnlyCaptureAssist()
    previous = tuple(obs["joints"][name]["position_deg"] for name in SERVO_ORDER) + (0.,) * 4
    context = rr_capture_assist_context(task=sup.snapshot, observation=obs,
        source_frame=SimpleNamespace(state_id="P09"), physics_tick=tick - (mode == "BLOCKED"),
        support_spec=sup.spec["support"])
    assist.advance(context=context, previous_final_full12=previous, physics_dt_s=1. / 120.)
    if mode == "BLOCKED":
        context = {**context, "dispatch_physics_tick": tick, "other_support_count": 1}
        assist.advance(context=context, previous_final_full12=previous, physics_dt_s=1. / 120.)
    assert assist.snapshot()["mode_name"] == mode
    return {"episode_observation_tick": tick, "state": assist.snapshot()}


def at_local_deadline(sup, obs, *, mode="DESCEND", tick_offset=0, absent=False,
                      episode_age=100., sensor_fault=None):
    next_obs = advance(obs)
    sup.rr_capture_feedback = (None if absent else feedback(sup, obs, next_obs["physics_tick"], mode))
    if sup.rr_capture_feedback is not None:
        sup.rr_capture_feedback["episode_observation_tick"] += tick_offset
    # Exercise deadline arithmetic without fabricating thousands of sensor
    # samples; the evaluator still receives the next adjacent physical tick.
    now = next_obs["simulation_time_s"]
    sup.stage_started_s = now - 45.  # Beyond nominal30 + maximum extension10.
    sup.episode_started_s = now - episode_age
    if sensor_fault == "invalid":
        next_obs["geometry_pose_aware"] = False
    elif sensor_fault == "body_collision":
        next_obs["body_collision"]["detected"] = True
        next_obs["contacts"]["base_link"]["obstacle"].update(
            active=True, normal_force_n=1., force_w_n=[1., 0., 0.])
    elif sensor_fault == "outside_xy":
        set_leg(next_obs, "RR", x=.3, bottom=.075)
    return sup.observe_and_update(next_obs)


def test_fresh_descend_waives_only_local_p09_deadline_without_awarding_capture(airborne_rr):
    sup, obs = airborne_rr
    task = at_local_deadline(sup, obs)
    context = task["rr_capture_continuation"]
    assert context["committed_feedback_matches_current_tick"]
    assert context["rr_top_reachable"] and context["rr_capture_recovery_allowed"]
    assert context["local_warning_only"]
    assert task["local_timeout"]["nominal_limit_exceeded"]
    assert not task["local_timeout"]["local_episode_terminal_enabled"]
    assert task["local_timeout"]["classification"] == "scheduler_warning_not_episode_end"
    assert task["termination_reason"] is None and task["stage_id"] == "P09"
    assert not task["success"] and not task["placed_history"]["RR"]
    assert task["completed_stage_ids"] == []


@pytest.mark.parametrize("kind", ["missing", "stale", "future", "blocked", "outside_xy"])
def test_uncommitted_stale_blocked_or_currently_illegal_feedback_cannot_waive_deadline(airborne_rr, kind):
    sup, obs = airborne_rr
    task = at_local_deadline(sup, obs, absent=kind == "missing",
        tick_offset=-1 if kind == "stale" else 1 if kind == "future" else 0,
        mode="BLOCKED" if kind == "blocked" else "DESCEND",
        sensor_fault="outside_xy" if kind == "outside_xy" else None)
    context = task["rr_capture_continuation"]
    assert context["committed_feedback_matches_current_tick"] is (kind in ("blocked", "outside_xy"))
    assert not context["rr_capture_recovery_allowed"] and not context["local_warning_only"]
    assert task["local_timeout"]["local_episode_terminal_enabled"]
    assert task["termination_reason"] == "INCOMPLETE_CONTROLLER_BLOCKED"
    assert task["termination_source"].startswith("LOCAL_")


def test_current_unverified_sensor_cannot_borrow_last_valid_recovery_context(airborne_rr):
    sup, obs = airborne_rr
    task = at_local_deadline(sup, obs, sensor_fault="invalid")
    assert not task["physical_evaluator"]["valid"]
    assert not task["rr_capture_continuation"]["rr_capture_recovery_allowed"]
    assert not task["rr_capture_continuation"]["local_warning_only"]
    assert task["termination_reason"] == "INFRASTRUCTURE_ERROR"
    assert task["termination_source"] == "UNVERIFIED_SENSOR"


def test_global_200_second_deadline_wins_even_with_fresh_valid_descent(airborne_rr):
    sup, obs = airborne_rr
    task = at_local_deadline(sup, obs, episode_age=200.)
    assert task["rr_capture_continuation"]["committed_feedback_matches_current_tick"]
    assert task["rr_capture_continuation"]["rr_top_reachable"]
    assert task["termination_reason"] == "INCOMPLETE_CONTROLLER_BLOCKED"
    assert task["termination_source"] == "GLOBAL_FINITE_TASK_DEADLINE"
    assert not task["success"]


def test_measured_body_collision_wins_over_fresh_descent_and_deadlines(airborne_rr):
    sup, obs = airborne_rr
    task = at_local_deadline(sup, obs, episode_age=200., sensor_fault="body_collision")
    assert task["termination_reason"] == "TASK_FAILURE_BODY_COLLISION"
    assert task["termination_source"] == "BODY_CONTACT"
    assert not task["rr_capture_continuation"]["rr_capture_recovery_allowed"]
    assert not task["rr_capture_continuation"]["local_warning_only"]
    assert not task["success"]

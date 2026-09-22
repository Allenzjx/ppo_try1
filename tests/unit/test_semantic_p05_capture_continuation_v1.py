"""Opt-in pending scheduling tests; synthetic sensors are not Isaac success."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from test_semantic_all_stage_physical_acceptance import early_top_place, set_leg
from test_semantic_p05_capture_handoff_v2 import measured
from test_semantic_supervisor import advance
from wlr50_clean.fsm.state_spec import load_fsm_spec
from wlr50_clean.ppo.semantic_supervisor import (
    CAPTURE_CONTINUATION_MODE, LEG_ORDER, NominalMotionProvider, TaskEvaluator,
    TaskStageSupervisor, load_task_spec, placement_predecessors_satisfied,
)
from wlr50_clean.reference.motion_contract import load_motion_contract

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "configs/ppo_task_conditioned_hip_wheel_v1/stage_task_spec.yaml"


def pending_setup(tmp_path, *, enabled=True):
    spec = load_task_spec(BASE)
    if enabled:
        spec["capture_continuation_semantics"] = CAPTURE_CONTINUATION_MODE
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    ev = TaskEvaluator(spec=spec)
    obs = measured()
    ev.observe(obs)
    obs = early_top_place(ev, obs, "FR")
    for bottom, hip, x in ((.012, 1.5, .4), (.020, 3., .4), (.075, 4., .53)):
        obs = advance(obs)
        set_leg(obs, "FL", x=x, bottom=bottom, hip=hip)
        ev.observe(obs)
    assert ev.snapshot["history"]["front_edge_crossed"]["FL"]
    assert not ev.snapshot["history"]["placed"]["FL"]
    sup = TaskStageSupervisor(path, evaluator=ev, initial_stage_id="P05")
    sup.observe_and_update(obs)
    return sup, obs


def boundary(sup, obs, *, stage_age=30.):
    while obs["physics_tick"] % 8 or sup._last_observation_tick == obs["physics_tick"]:
        obs = advance(obs)
        if obs["physics_tick"] % 8:
            sup.observe_and_update(obs)
    sup.stage_started_s = obs["simulation_time_s"] - stage_age
    return obs, sup.observe_and_update(obs)


def test_pending_handoff_changes_schedule_without_placement_or_completion(tmp_path):
    sup, obs = pending_setup(tmp_path)
    obs, task = boundary(sup, obs)
    assert task["stage_id"] == "P06" and task["entry_valid"]
    assert task["fl_capture_pending"] and task["allow_capture_continuation"]
    assert task["pending_capture"]  # Does not disappear when active_leg becomes RR.
    assert task["capture_continuation"]["submode"] == "P06_CAPTURE_PENDING"
    assert task["completed_stage_ids"] == []
    assert not task["placed_history"]["FL"] and not task["fl_contact_observed"]
    event = task["transition_evidence"][-1]
    assert event["scheduler_transition"] and not event["physical_completion_awarded"]
    assert event["completion_values"]["placed_FL"] < 1.
    assert task["termination_reason"] is None and not task["success"]
    assert task["local_timeout"]["classification"] == "scheduler_warning_not_episode_end"
    assert task["p05_local_deadline_warning"]
    assert task["capture_pending_elapsed_s"] == pytest.approx(
        (obs["physics_tick"]-task["history"]["event_ticks"]["front_edge_crossed"]["FL"])/120.)
    assert task["capture_pending_elapsed_s"] > 0.
    assert not task["local_timeout"]["timer_inputs_observable_in_existing372"]
    assert task["local_timeout"]["current_schema_observable"]
    assert task["task_progress_potential"] == sup.physical_potential(task["physical_evaluator"])


@pytest.mark.parametrize("fault", ["outside_xy", "one_support", "ground", "no_lift"])
def test_unsafe_or_unqualified_capture_path_waits_without_local_episode_end(tmp_path, fault):
    sup, obs = pending_setup(tmp_path)
    if fault == "outside_xy":
        obs["wheels"]["front_left_ankle"]["center_w_m"][1] = 2.
        obs["wheels"]["front_left_ankle"]["bottom_w_m"][1] = 2.
    elif fault == "one_support":
        set_leg(obs, "RR", bottom=.001)
        set_leg(obs, "RL", bottom=.001)
    elif fault == "ground":
        set_leg(obs, "FL", bottom=0., surface="GROUND")
    else:
        sup.evaluator._history["active_lift"]["FL"] = False
        sup.evaluator._attempt_active["FL"] = False
        sup.evaluator._samples["FL"].clear()
    obs, task = boundary(sup, obs, stage_age=45.)
    assert task["stage_id"] == "P05"
    assert not task["allow_capture_continuation"]
    assert task["termination_reason"] is None and not task["success"]
    assert not task["local_timeout"]["local_episode_terminal_enabled"]


def test_old_mode_keeps_original_local_timeout(tmp_path):
    sup, obs = pending_setup(tmp_path, enabled=False)
    _, task = boundary(sup, obs, stage_age=45.)
    assert task["stage_id"] == "P05"
    assert task["termination_reason"] == "INCOMPLETE_CONTROLLER_BLOCKED"
    assert task["termination_source"].startswith("LOCAL_")
    assert "capture_continuation" not in task


@pytest.mark.parametrize("phase", ["P06", "P09", "P11", "P13"])
def test_lost_downstream_permission_waits_without_false_support_or_local_done(tmp_path, phase):
    sup, obs = pending_setup(tmp_path)
    sup.stage_id = phase
    set_leg(obs, "RL", bottom=.001)
    set_leg(obs, "RR", bottom=.001)
    obs, task = boundary(sup, obs, stage_age=100.)
    assert task["stage_id"] == phase and task["fl_capture_pending"]
    assert not task["allow_capture_continuation"] and not task["entry_valid"]
    assert not task["physical_evaluator"]["current_legs"]["FL"]["support"]
    assert task["termination_reason"] is None and not task["success"]
    assert not task["local_timeout"]["local_episode_terminal_enabled"]


def test_global_deadline_and_hard_safety_are_not_suppressed(tmp_path):
    sup, obs = pending_setup(tmp_path)
    obs, task = boundary(sup, obs)
    assert task["stage_id"] == "P06"
    sup.episode_started_s = obs["simulation_time_s"] - 200.
    task = sup.observe_and_update(advance(obs))
    assert task["termination_source"] == "GLOBAL_FINITE_TASK_DEADLINE"
    assert task["termination_reason"] == "INCOMPLETE_CONTROLLER_BLOCKED"
    sup, obs = pending_setup(tmp_path)
    obs = advance(obs)
    obs["body_collision"]["detected"] = True
    obs["contacts"]["base_link"]["obstacle"].update(active=True, force_w_n=[1., 0., 0.])
    task = sup.observe_and_update(obs)
    assert task["termination_source"] == "BODY_CONTACT"
    assert task["termination_reason"] == "TASK_FAILURE_BODY_COLLISION"


def test_late_real_contact_records_old_task_once_without_rewind(tmp_path):
    sup, obs = pending_setup(tmp_path)
    obs, task = boundary(sup, obs)
    for count in range(1, 4):
        obs = advance(obs)
        set_leg(obs, "FL", x=.53, bottom=.05, surface="TOP")
        task = sup.observe_and_update(obs)
        assert task["stage_id"] == "P06"
        assert task["fl_contact_observed"]
        assert task["placed_history"]["FL"] is (count >= 2)
    assert task["completed_stage_ids"] == ["P05"]
    assert not task["fl_capture_pending"]
    late = [event for event in task["transition_evidence"]
            if event.get("event_kind") == "pending_capture_physically_completed"]
    assert len(late) == 1 and not late[0]["scheduler_transition"]
    assert late[0]["physical_completion_awarded"]
    assert late[0]["physics_tick"] == task["history"]["event_ticks"]["placed"]["FL"]


@pytest.mark.parametrize("phase", ["P06", "P07", "P08", "P09", "P10", "P11", "P13"])
def test_later_entry_waives_only_FL_history_not_other_task_predicates(tmp_path, phase):
    sup, _ = pending_setup(tmp_path)
    ev = deepcopy(sup.evaluator.snapshot)
    # These represent separate evaluator measurements solely for entry wiring.
    for leg in ("RR", "RL"):
        ev["history"]["placed"][leg] = True
        ev["history"]["front_edge_crossed"][leg] = True
        ev["current_legs"][leg].update(current_lift_valid=True, lift_established=True,
            within_top_xy=True, ground_contact=False, top_contact=True, top_surface_contact=True)
    report = sup.entry_report(phase, ev)
    assert report["valid"] and report["continuation_waived_conditions"] == ["placed_FL"]
    assert report["values"]["placed_FL"] < 1. and not report["actual_conditions_satisfied"]
    assert not ev["history"]["placed"]["FL"]
    ev["history"]["placed"]["FR"] = False
    assert not sup.entry_report(phase, ev)["valid"]
    if phase in ("P10", "P11", "P13"):
        ev["history"]["placed"]["FR"] = True
        ev["history"]["placed"]["RR"] = False
        assert not sup.entry_report(phase, ev)["valid"]


def test_rear_progress_credit_can_follow_real_motion_but_RR_FIRST_remains(tmp_path):
    sup, obs = pending_setup(tmp_path)
    history = sup.evaluator.snapshot["history"]
    assert placement_predecessors_satisfied(sup.spec, history, "RR")
    assert not placement_predecessors_satisfied(sup.spec, history, "RL")
    old = deepcopy(sup.spec)
    old.pop("capture_continuation_semantics")
    assert not placement_predecessors_satisfied(old, history, "RR")
    # Real RL Q/C/P still cannot jump the RR placement requirement.
    for values in ((.012, 1.5, .4, "AIR"), (.049, 3., .498, "TOP"), (.049, 3., .501, "TOP")):
        obs = advance(obs)
        bottom, hip, x, surface = values
        set_leg(obs, "RL", x=x, bottom=bottom, hip=hip, surface=surface)
        ev = sup.evaluator.observe(obs)
    assert ev["termination_source"] == "RR_FIRST_ORDER"
    assert not ev["history"]["placed"]["RL"]


def test_pending_P13_is_not_full_task_success(tmp_path):
    sup, obs = pending_setup(tmp_path)
    sup.stage_id = "P13"
    sup.stage_started_s = -100.
    task = sup.observe_and_update(advance(obs))
    assert task["stage_id"] == "P13" and task["fl_capture_pending"]
    assert task["termination_reason"] is None
    assert not task["physical_evaluator"]["traversal_task_complete"] and not task["success"]


def test_P06_owner_starts_real_clock_and_pauses_only_its_unsafe_pending_contribution(tmp_path):
    sup, obs = pending_setup(tmp_path)
    fsm = load_fsm_spec(ROOT / "configs/fsm_states.yaml")
    contract = load_motion_contract(ROOT / "configs/recording_motion_contract.json")
    provider = NominalMotionProvider(contract, spec=sup.spec, fsm_spec=fsm)
    provider.evaluate(sup.snapshot, obs)
    obs, task = boundary(sup, obs)
    nominal = provider.evaluate(task, obs)
    layer = provider._continuous_layers[-1]
    assert layer["stage"] == "P06" and layer["ticks"] == 1 and layer["sample"].tick_index == 0
    assert nominal[8:] == (.3,) * 4
    assert provider.nominal_suggestion_diagnostics["capture_continuation"]["P06_wheel_contribution_enabled"]
    obs = advance(obs)
    set_leg(obs, "RL", bottom=.001)
    set_leg(obs, "RR", bottom=.001)
    task = sup.observe_and_update(obs)
    assert task["fl_capture_pending"] and not task["allow_capture_continuation"]
    nominal = provider.evaluate(task, obs)
    assert layer["ticks"] == 1 and nominal[8:] == (0.,) * 4
    obs = advance(obs)
    set_leg(obs, "RL", bottom=0., surface="GROUND")
    set_leg(obs, "RR", bottom=0., surface="GROUND")
    task = sup.observe_and_update(obs)
    nominal = provider.evaluate(task, obs)
    assert task["allow_capture_continuation"]
    assert layer["ticks"] == 2 and layer["sample"].tick_index == 1
    assert nominal[8:] == (.3,) * 4
    assert not provider.nominal_suggestion_diagnostics["capture_continuation"]["source_clock_catch_up"]


def test_unknown_or_incomplete_opt_in_rejected(tmp_path):
    sup, _ = pending_setup(tmp_path)
    for value in ("unversioned", True):
        spec = deepcopy(sup.spec)
        spec["capture_continuation_semantics"] = value
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError, match="capture continuation"):
            load_task_spec(path)

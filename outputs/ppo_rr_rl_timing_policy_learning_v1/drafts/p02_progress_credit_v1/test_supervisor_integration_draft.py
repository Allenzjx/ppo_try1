"""OUTPUTS-ONLY integration draft; real evaluator on synthetic sensors, not Isaac success."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from test_semantic_all_stage_physical_acceptance import new_observation, set_leg
from test_semantic_supervisor import advance
from wlr50_clean.ppo.semantic_supervisor import (
    TaskEvaluator, TaskStageSupervisor, load_task_spec, _P02ProgressCredit,
    P02_PROGRESS_CREDIT_MODE,
)

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "configs/ppo_rr_rl_timing_policy_learning_v1").is_dir())
CONFIG = ROOT / "configs/ppo_rr_rl_timing_policy_learning_v1/stage_task_spec.yaml"


def setup(tmp_path, *, enabled=True):
    spec = load_task_spec(CONFIG)
    if enabled:
        spec["p02_progress_credit_mode"] = P02_PROGRESS_CREDIT_MODE
    else:
        spec.pop("p02_progress_credit_mode", None)
    path = tmp_path / ("credit.yaml" if enabled else "legacy.yaml")
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    ev = TaskEvaluator(spec=spec)
    sup = TaskStageSupervisor(path, evaluator=ev, initial_stage_id="P02")
    obs = new_observation()
    obs["center_of_mass"] = dict(valid=True, position_w_m=[.30, 0., .10],
        velocity_w_m_s=[0., 0., 0.], total_mass_kg=3., included_bodies=["synthetic"])
    return sup, obs


def motion(sup, obs, *, x=None, bottom=.07, hip=4.5, surface="AIR"):
    obs = advance(obs)
    set_leg(obs, "FR", x=x, bottom=bottom, hip=hip, surface=surface)
    return obs, sup.observe_and_update(obs)


def ready(tmp_path, *, enabled=True):
    sup, obs = setup(tmp_path, enabled=enabled)
    initial = sup.observe_and_update(obs)
    if enabled:
        assert initial["p02_progress_credit"]["credit_m"] == 0.
        assert initial["p02_progress_credit"]["elapsed_physics_s"] == 0.
    for bottom, hip in ((.004, 1.5), (.012, 3.), (.07, 4.5)):
        obs, task = motion(sup, obs, bottom=bottom, hip=hip)
    assert task["history"]["active_lift"]["FR"]
    for i in range(1, 13):
        obs, task = motion(sup, obs, x=.4 + .001 * i)
    if enabled:
        assert task["p02_progress_credit"]["current_eligible"]
        assert task["p02_progress_credit"]["credit_fraction"] == 1.
    return sup, obs, task


def exhausted_tick(sup, obs, **kwargs):
    # Isolate the local-budget condition without inventing sensor history.
    sup.stage_started_s = obs["simulation_time_s"] - 23.
    return motion(sup, obs, **kwargs)


def test_real_air_progress_continues_same_old_budget_but_legacy_does_not(tmp_path):
    for enabled in (True, False):
        sup, obs, _ = ready(tmp_path, enabled=enabled)
        obs, task = exhausted_tick(sup, obs, x=.4121)
        assert task["stage_id"] == "P02" and not task["success"]
        assert not task["placed_history"]["FR"]
        if enabled:
            assert task["termination_reason"] is None
            assert task["p02_progress_credit"]["local_deadline_suppressed"]
            assert task["p02_progress_credit"]["accepted_approach_lower_m"] == -.010
        else:
            assert task["termination_source"] == "LOCAL_BOUNDED_RECOVERY_EXHAUSTED"


def test_duplicate_snapshot_idempotent_and_actual_physics_dt(tmp_path):
    sup, obs, task = ready(tmp_path)
    same = sup.observe_and_update(deepcopy(obs))
    assert same == task
    credit = task["p02_progress_credit"]["credit_m"]
    obs, later = motion(sup, obs, x=.412)
    assert later["p02_progress_credit"]["elapsed_physics_s"] == pytest.approx(1 / 120)
    assert later["p02_progress_credit"]["credit_m"] == pytest.approx(credit - .00125 / 120)
    # The ledger itself also tolerates repeated same-tick requests without charge.
    again = sup._p02_progress_credit.observe(obs, later["physical_evaluator"],
        stage_id="P02", goal_values=later["completion_values"], entry_valid=True,
        episode_age_s=obs["simulation_time_s"], physical_terminal=None)
    assert again["credit_m"] == later["p02_progress_credit"]["credit_m"]


def test_stall_or_old_space_oscillation_cannot_renew_earned_extension(tmp_path):
    sup, obs, _ = ready(tmp_path)
    sup.stage_started_s = obs["simulation_time_s"] - 23.
    for i in range(800):
        obs, task = motion(sup, obs, x=.412 - .003 * (i % 2))
        if task["termination_reason"]:
            break
    assert 715 <= i <= 721
    assert task["p02_progress_credit"]["credit_fraction"] == 0.
    assert task["termination_source"] == "LOCAL_BOUNDED_RECOVERY_EXHAUSTED"


@pytest.mark.parametrize("fault", ["ground", "low_clearance", "lateral", "one_support", "sensor"])
def test_current_contact_geometry_support_and_sensor_gates_override_credit(tmp_path, fault):
    sup, obs, _ = ready(tmp_path)
    sup.stage_started_s = obs["simulation_time_s"] - 23.
    obs = advance(obs)
    set_leg(obs, "FR", x=.4121, bottom=.07, hip=4.5)
    if fault == "ground":
        set_leg(obs, "FR", bottom=0., surface="GROUND")
    elif fault == "low_clearance":
        set_leg(obs, "FR", bottom=.052)
    elif fault == "lateral":
        for key in ("center_w_m", "bottom_w_m"):
            obs["wheels"]["front_right_ankle"][key][1] = 2.
    elif fault == "one_support":
        for leg in ("FL", "RL"):
            set_leg(obs, leg, bottom=.01, surface="AIR")
    else:
        obs["contacts"]["front_left_wheel"]["ground"]["pair_verified"] = False
    task = sup.observe_and_update(obs)
    assert not task["p02_progress_credit"]["current_eligible"]
    assert not task["p02_progress_credit"]["local_deadline_suppressed"]
    assert task["termination_reason"] is not None


def test_ground_then_air_does_not_use_crossed_old_lift_event(tmp_path):
    sup, obs, task = ready(tmp_path)
    ev = sup.evaluator
    ledger = _P02ProgressCredit(sup.spec)
    # Use the real evaluator directly so a genuine crossing does not intentionally
    # move the scheduler to P03; no current-leg qualification keys are fabricated.
    for data in (dict(x=.501, bottom=.07), dict(x=.501, bottom=.049, surface="TOP"),
                 dict(x=.501, bottom=.049, surface="TOP"), dict(bottom=0., surface="GROUND"),
                 dict(bottom=.052, surface="AIR")):
        obs = advance(obs)
        set_leg(obs, "FR", hip=4.5, **data)
        ev.observe(obs)
    snap = ev.snapshot
    assert snap["history"]["active_lift"]["FR"] and snap["history"]["front_edge_crossed"]["FR"]
    assert snap["current_legs"]["FR"]["air"]
    values = {k: sup.predicate(k, snap) for k in ("lifted_FR", "clear_FR", "approach_FR")}
    result = ledger.observe(obs, snap, stage_id="P02", goal_values=values,
        entry_valid=True, episode_age_s=obs["simulation_time_s"], physical_terminal=None)
    assert not result["checks"]["same_air_qualified_lift"]
    assert not result["current_eligible"]


@pytest.mark.parametrize("fault", ["global200", "body_collision"])
def test_global_and_real_safety_take_precedence(tmp_path, fault):
    sup, obs, _ = ready(tmp_path)
    if fault == "global200":
        sup.episode_started_s = obs["simulation_time_s"] - 200.
    else:
        obs["contacts"]["base_link"]["obstacle"].update(active=True,
            normal_force_n=3., force_w_n=[3., 0., 0.])
        obs["body_collision"]["detected"] = True
    obs, task = exhausted_tick(sup, obs, x=.4121)
    assert task["termination_reason"] is not None
    assert not task["p02_progress_credit"]["continuation_allowed"]
    if fault == "global200":
        assert task["termination_source"] == "GLOBAL_FINITE_TASK_DEADLINE"


def test_unchanged_approach_band_and_next_decision_handoff(tmp_path):
    sup, obs, _ = ready(tmp_path)
    sup.stage_started_s = obs["simulation_time_s"] - 23.
    while (obs["physics_tick"] + 1) % 8 == 0:
        obs, _ = motion(sup, obs, x=.412)
    obs, task = motion(sup, obs, x=.4901)
    assert task["completion_values"]["approach_FR"] == 1.
    assert task["p02_progress_credit"]["completed_handoff_pending"]
    assert task["termination_reason"] is None
    while task["stage_id"] == "P02":
        obs, task = motion(sup, obs, x=.4901)
    assert task["stage_id"] == "P03"
    assert task["termination_reason"] is None
    assert not task["placed_history"]["FR"]
    assert tuple(task["p02_progress_credit"][k] for k in
                 ("best_remaining_m", "credit_fraction", "current_eligible")) == (0., 0., False)


def test_bad_mode_is_not_silently_legacy(tmp_path):
    spec = load_task_spec(CONFIG)
    spec["p02_progress_credit_mode"] = "wrong"
    path = tmp_path / "wrong.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown P02"):
        load_task_spec(path)


def test_real_top_completion_pending_does_not_require_air_or_remaining_credit(tmp_path):
    sup, obs, _ = ready(tmp_path)
    sup.stage_started_s = obs["simulation_time_s"] - 23.
    while obs["physics_tick"] % 8:
        obs, _ = motion(sup, obs, x=.412)
    obs, task = motion(sup, obs, x=.501)
    assert task["p02_progress_credit"]["completed_handoff_pending"]
    # Isolate already-complete, decision-pending behavior with an empty ledger.
    sup._p02_progress_credit.credit_m = 0.
    obs, task = motion(sup, obs, x=.501, bottom=.049, surface="TOP")
    assert task["physical_evaluator"]["current_legs"]["FR"]["top_contact"]
    assert not task["p02_progress_credit"]["current_eligible"]
    assert task["p02_progress_credit"]["credit_fraction"] == 0.
    assert task["p02_progress_credit"]["completed_handoff_pending"]
    assert task["termination_reason"] is None
    while task["stage_id"] == "P02":
        obs, task = motion(sup, obs, x=.501, bottom=.049, surface="TOP")
    assert task["stage_id"] == "P03" and task["termination_reason"] is None


def test_fresh_real_lift_after_ground_before_crossing_can_earn_again(tmp_path):
    sup, obs, task = ready(tmp_path)
    old_event = task["history"]["event_ticks"]["active_lift"]["FR"]
    obs, task = motion(sup, obs, x=.412, bottom=0., surface="GROUND")
    assert not task["history"]["active_lift"]["FR"]
    for bottom, hip in ((.004, 6.), (.012, 7.5), (.07, 9.)):
        obs, task = motion(sup, obs, x=.412, bottom=bottom, hip=hip)
    assert task["history"]["active_lift"]["FR"]
    assert task["history"]["event_ticks"]["active_lift"]["FR"] == old_event  # First historical event remains.
    assert task["p02_progress_credit"]["current_eligible"]
    obs, task = motion(sup, obs, x=.413, hip=9.)
    assert task["p02_progress_credit"]["earned_progress_m"] > 0.
    assert not task["placed_history"]["FR"]

"""Same-AIR P05 recross and finite timeout; synthetic CPU evidence, not Isaac."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from test_semantic_all_stage_physical_acceptance import early_top_place, set_leg
from test_semantic_p05_capture_handoff_v2 import measured
from test_semantic_p05_preedge_recovery import (
    KEY, ROLL, clone, mature, status, task,
)
from test_semantic_supervisor import advance
from wlr50_clean.ppo.semantic_supervisor import (
    NominalMotionProvider, P05_FINITE_RECOVERY_TIMEOUT_MODE,
    P05_PREEDGE_RECOVERY_MODE, P05_SAME_AIR_RECROSS_MODE,
    TaskEvaluator, TaskStageSupervisor, _p05_preedge_recovery_enabled,
    load_task_spec,
)


ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs/ppo_rr_capture_then_rl_transfer_v1/stage_task_spec.yaml"
TIMEOUT_KEY = "p05_finite_recovery_timeout_semantics"


def configured():
    spec = load_task_spec(CFG)
    assert spec["nominal"][KEY] == P05_SAME_AIR_RECROSS_MODE
    assert spec[TIMEOUT_KEY] == P05_FINITE_RECOVERY_TIMEOUT_MODE
    return spec


@pytest.fixture(scope="module")
def mature_v2(mature):
    old = mature[0]
    provider = NominalMotionProvider(old.contract, spec=configured(),
                                    fsm_spec=old._reference_fsm_spec)
    for tick in range(3600):
        command = provider.evaluate(task(tick, tick / 120.))
    assert provider.endpoint_issued and command[8:] == (0.,) * 4
    return provider


def recross_task(tick=3600, age=30.):
    """An isolated predicate input; actual evaluator provenance is tested below."""
    value = task(tick, age)
    ev = value["physical_evaluator"]
    ev["history"]["front_edge_crossed"]["FL"] = True
    ev["history"]["event_ticks"] = {"front_edge_crossed": {"FL": tick - 60}}
    ev["current_legs"]["FL"].update(active_attempt=True, consecutive_air_samples=61)
    return value


def test_new_modes_are_paired_and_adopted_not_silently_enabled_for_v1():
    value = configured()
    assert _p05_preedge_recovery_enabled(value)
    value.pop(TIMEOUT_KEY)
    with pytest.raises(ValueError, match="explicitly paired"):
        _p05_preedge_recovery_enabled(value)
    value = configured(); value["nominal"][KEY] = P05_PREEDGE_RECOVERY_MODE
    with pytest.raises(ValueError, match="explicitly paired"):
        _p05_preedge_recovery_enabled(value)


@pytest.mark.parametrize("age", [29.999, 30., 39.999, 40.])
def test_first_approach_matches_existing_mode_and_absolute_window(mature, mature_v2, age):
    item = task(age=age)
    old, new = status(mature[0], item), status(mature_v2, item)
    assert new["eligible"] == old["eligible"] == (30. <= age < 40.)
    assert new["same_air_recross"]["branch"] == "first_approach"
    old_provider, new_provider = clone(mature[0]), clone(mature_v2)
    assert new_provider.evaluate(deepcopy(item)) == old_provider.evaluate(deepcopy(item))
    assert new_provider.tracking_servo_names == old_provider.tracking_servo_names
    assert new_provider.normal_drive_bias_full12 == old_provider.normal_drive_bias_full12


def measured_recross(cross_surface):
    """Contiguous measured sequence earns cross at100, observes current tick160.

    A TOP cross at100 followed by AIR at101 must not borrow that crossing for
    same-AIR recovery, even though the old active attempt remains true.
    """
    ev, obs = TaskEvaluator(spec=configured()), measured()
    ev.observe(obs)
    obs = early_top_place(ev, obs, "FR")
    while obs["physics_tick"] < 97:
        obs = advance(obs); ev.observe(obs)
    obs = advance(obs); set_leg(obs, "FL", x=.4, bottom=.012, hip=1.5); ev.observe(obs)
    obs = advance(obs)
    set_leg(obs, "FL", x=.49, bottom=.049, surface="TOP", hip=3.)
    ev.observe(obs)
    assert ev.snapshot["history"]["active_lift"]["FL"]
    assert not ev.snapshot["history"]["front_edge_crossed"]["FL"]
    obs = advance(obs)
    set_leg(obs, "FL", x=.53 if cross_surface == "AIR" else .501,
            bottom=.075 if cross_surface == "AIR" else .049,
            surface=cross_surface, hip=4.)
    ev.observe(obs)
    while obs["physics_tick"] < 160:
        obs = advance(obs); set_leg(obs, "FL", x=.482, bottom=.0569); ev.observe(obs)
    assert ev.snapshot["history"]["event_ticks"]["front_edge_crossed"]["FL"] == 100
    assert not ev.snapshot["history"]["placed"]["FL"]
    assert ev.snapshot["current_legs"]["FL"]["active_attempt"]
    return ev, obs


@pytest.mark.parametrize("cross_surface,air_count,eligible", [("AIR", 61, True), ("TOP", 60, False)])
def test_real_evaluator_air_streak_must_include_cross_tick(mature_v2, cross_surface, air_count, eligible):
    ev, _ = measured_recross(cross_surface)
    assert ev.snapshot["current_legs"]["FL"]["consecutive_air_samples"] == air_count
    item = task(160, 30.); item["physical_evaluator"] = ev.snapshot
    # Only deadline arithmetic is isolated; sensing and event counters above
    # are from adjacent real evaluator calls, not fabricated history fields.
    info = status(mature_v2, item, previous_tick=159)
    assert info["eligible"] is eligible
    recross = info["same_air_recross"]
    assert recross["cross_event_tick"] == 100 and recross["consecutive_air_samples"] == air_count
    assert recross["same_air_span_valid"] is eligible
    assert recross["air_start_tick"] == (100 if eligible else 101)


def test_real_reground_does_not_borrow_surviving_cross_or_attempt(mature_v2):
    ev, obs = measured_recross("AIR")
    obs = advance(obs); set_leg(obs, "FL", x=.482, bottom=0., surface="GROUND"); ev.observe(obs)
    obs = advance(obs); set_leg(obs, "FL", x=.482, bottom=.0569); ev.observe(obs)
    current = ev.snapshot["current_legs"]["FL"]
    assert ev.snapshot["history"]["front_edge_crossed"]["FL"] and current["active_attempt"]
    assert current["consecutive_air_samples"] == 1
    item = task(162, 30.); item["physical_evaluator"] = ev.snapshot
    info = status(mature_v2, item, previous_tick=161)
    assert not info["eligible"] and not info["same_air_recross"]["same_air_span_valid"]


@pytest.mark.parametrize("field,bad", [
    ("count", True), ("count", 61.), ("count", "61"), ("count", None),
    ("count", 0), ("count", 162), ("cross", True), ("cross", 100.),
    ("cross", "100"), ("cross", None), ("cross", -1), ("cross", 161),
    ("active_attempt", False), ("active_attempt", 1),
])
def test_recross_requires_strict_finite_history_and_attempt(mature_v2, field, bad):
    item = recross_task(160)
    ev = item["physical_evaluator"]
    if field == "cross": ev["history"]["event_ticks"]["front_edge_crossed"]["FL"] = bad
    else: ev["current_legs"]["FL"]["consecutive_air_samples" if field == "count" else field] = bad
    assert not status(mature_v2, item, previous_tick=159)["eligible"]


@pytest.mark.parametrize("age,expected", [(29.999, False), (30., True), (39.999, True), (40., False)])
def test_recross_does_not_renew_original_window(mature_v2, age, expected):
    assert status(mature_v2, recross_task(age=age))["eligible"] is expected


@pytest.mark.parametrize("field,bad", [("within_top_xy", True), ("ground_contact", True),
    ("air", False), ("clearance_m", 0.), ("front_distance_m", -.005),
    ("front_distance_m", -.10001)])
def test_recross_does_not_widen_capture_or_geometry(mature_v2, field, bad):
    item = recross_task(); item["physical_evaluator"]["current_legs"]["FL"][field] = bad
    assert not status(mature_v2, item)["eligible"]


def test_recross_intercept_only_advises_wheels_without_history_or_other_owner_change(mature_v2):
    enabled, disabled = clone(mature_v2), clone(mature_v2)
    disabled._p05_preedge_recovery = False
    item = recross_task(); original = deepcopy(item)
    expected, actual = disabled.evaluate(deepcopy(item)), enabled.evaluate(item)
    assert actual[:8] == expected[:8] and actual[8:] == ROLL and expected[8:] == (0.,) * 4
    assert enabled.tracking_servo_names == disabled.tracking_servo_names
    assert enabled.normal_drive_bias_full12 == disabled.normal_drive_bias_full12
    assert enabled._continuous_layers[-1]["sample"] == disabled._continuous_layers[-1]["sample"]
    assert enabled._source_motion._tick_index == disabled._source_motion._tick_index
    assert item == original and enabled.state_id == "P05"
    info = enabled.nominal_suggestion_diagnostics[KEY]
    assert info["eligible"] and not info["policy_residual_restricted"]
    assert not info["phase_or_capture_credit_awarded"] and not info["recovery_clock_reset_or_latch"]


def test_fresh_source_stop_still_owns_wheels_in_recross(mature, mature_v2, monkeypatch):
    provider, stop = clone(mature_v2), mature[1]
    original = provider._continuous_advisory
    def with_stop(*args, **kwargs):
        proposed, tracking = original(*args, **kwargs)
        layer = provider._continuous_layers[-1]
        layer["sample"] = replace(layer["sample"], atomic_groups=stop.atomic_groups)
        return proposed[:8] + (0.,) * 4, tracking
    monkeypatch.setattr(provider, "_continuous_advisory", with_stop)
    assert provider.evaluate(recross_task())[8:] == (0.,) * 4
    info = provider.nominal_suggestion_diagnostics[KEY]
    assert not info["eligible"] and info["fresh_source_wheel_owners"]
    assert "fresh_source_wheel_owner_absent" in info["reasons"]


@pytest.fixture
def airborne_fl():
    ev, obs = TaskEvaluator(spec=configured()), measured()
    ev.observe(obs); obs = early_top_place(ev, obs, "FR")
    for bottom, hip, x in ((.012, 1.5, .4), (.020, 3., .4), (.075, 4., .53)):
        obs = advance(obs); set_leg(obs, "FL", x=x, bottom=bottom, hip=hip); ev.observe(obs)
    assert obs["physics_tick"] == 6 and ev.snapshot["history"]["front_edge_crossed"]["FL"]
    assert not ev.snapshot["history"]["placed"]["FL"]
    sup = TaskStageSupervisor(CFG, evaluator=ev, initial_stage_id="P05")
    sup.observe_and_update(obs)
    return sup, obs


def at_deadline(sup, obs, *, age=40., episode_age=100., path=True, fault=None):
    obs = advance(obs)
    if not path: set_leg(obs, "FL", x=.482, bottom=.0569)
    if fault == "collision":
        obs["body_collision"]["detected"] = True
        obs["contacts"]["base_link"]["obstacle"].update(active=True, normal_force_n=1., force_w_n=[1., 0., 0.])
    elif fault == "invalid": obs["geometry_pose_aware"] = False
    sup.stage_started_s = obs["simulation_time_s"] - age
    sup.episode_started_s = obs["simulation_time_s"] - episode_age
    return sup.observe_and_update(obs), obs


@pytest.mark.parametrize("age,ends", [(39.999, False), (40., True), (45., True)])
def test_no_current_path_is_finite_after_original_window(airborne_fl, age, ends):
    task_now, _ = at_deadline(*airborne_fl, age=age, path=False)
    assert not task_now["allow_capture_continuation"]
    assert task_now["local_timeout"]["local_episode_terminal_enabled"] is ends
    assert (task_now["termination_reason"] == "INCOMPLETE_CONTROLLER_BLOCKED") is ends
    if ends: assert task_now["termination_source"].startswith("LOCAL_")
    assert not task_now["success"] and not task_now["placed_history"]["FL"]


def test_current_legal_capture_survives_and_hands_over_without_fake_placement(airborne_fl):
    sup, obs = airborne_fl
    task_now, obs = at_deadline(sup, obs, age=45.)
    assert task_now["stage_id"] == "P05" and task_now["allow_capture_continuation"]
    assert task_now["termination_reason"] is None
    assert not task_now["local_timeout"]["local_episode_terminal_enabled"]
    task_now = sup.observe_and_update(advance(obs))
    assert task_now["stage_id"] == "P06" and task_now["termination_reason"] is None
    assert not task_now["placed_history"]["FL"] and task_now["completed_stage_ids"] == []


def test_actual_completed_goal_waits_only_for_next_decision_handoff(airborne_fl):
    sup, obs = airborne_fl
    obs = advance(obs); sup.observe_and_update(obs)  # Tick7, still before deadline.
    obs = advance(obs); set_leg(obs, "FL", x=.53, bottom=.049, surface="TOP")
    first = sup.observe_and_update(obs)
    assert obs["physics_tick"] == 8 and not first["placed_history"]["FL"]
    obs = advance(obs)
    sup.stage_started_s = obs["simulation_time_s"] - 40.
    sup.episode_started_s = obs["simulation_time_s"] - 100.
    captured = sup.observe_and_update(obs)
    assert captured["placed_history"]["FL"] and captured["stage_id"] == "P05"
    assert captured["termination_reason"] is None and not captured["allow_capture_continuation"]
    assert not captured["local_timeout"]["local_episode_terminal_enabled"]
    while obs["physics_tick"] < 16:
        obs = advance(obs); captured = sup.observe_and_update(obs)
        assert captured["termination_reason"] is None
    assert captured["stage_id"] == "P06" and captured["completed_stage_ids"] == ["P05"]


def test_completed_goal_without_valid_entry_does_not_get_handoff_exemption():
    ev, obs = TaskEvaluator(spec=configured()), measured()
    ev.observe(obs)
    obs = early_top_place(ev, obs, "FL")  # No FR placement: P05 entry is illegal.
    sup = TaskStageSupervisor(CFG, evaluator=ev, initial_stage_id="P05")
    sup.observe_and_update(obs)
    task_now, obs = at_deadline(sup, obs, age=40.)
    assert obs["physics_tick"] % 8 != 0
    assert task_now["placed_history"]["FL"] and not task_now["placed_history"]["FR"]
    assert task_now["stage_id"] == "P05" and not task_now["allow_capture_continuation"]
    assert task_now["local_timeout"]["local_episode_terminal_enabled"]
    assert task_now["termination_reason"] == "INCOMPLETE_CONTROLLER_BLOCKED"
    assert task_now["termination_source"].startswith("LOCAL_")


@pytest.mark.parametrize("fault,reason,source", [
    (None, "INCOMPLETE_CONTROLLER_BLOCKED", "GLOBAL_FINITE_TASK_DEADLINE"),
    ("collision", "TASK_FAILURE_BODY_COLLISION", "BODY_CONTACT"),
    ("invalid", "INFRASTRUCTURE_ERROR", "UNVERIFIED_SENSOR"),
])
def test_physical_failure_and_global_200_win_over_current_capture(airborne_fl, fault, reason, source):
    task_now, _ = at_deadline(*airborne_fl, age=45., episode_age=200., fault=fault)
    assert task_now["termination_reason"] == reason and task_now["termination_source"] == source
    assert not task_now["success"]

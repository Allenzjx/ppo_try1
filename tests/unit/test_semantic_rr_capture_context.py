"""Current-sensor rear facts; synthetic counterexamples are not task success."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.ppo.semantic_rr_capture_assist import RRHipOnlyCaptureAssist, rr_capture_assist_context
from wlr50_clean.ppo.semantic_rr_capture_context import rr_capture_transfer_context
from wlr50_clean.ppo.semantic_transfer_roles import DIAGONAL, validate_config


ROOT = Path(__file__).resolve().parents[2]
SUPPORT_SPEC = yaml.safe_load((ROOT / "configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml").read_text())["support"]
FORCE_FLOOR = SUPPORT_SPEC["force_noise_floor_n"]
FACTS = ("rr_lift_carry", "rr_top_reachable", "rr_top_contact", "rr_current_bearing",
         "rl_transfer_ready", "rr_capture_recovery_allowed", "fl_wheel_guidance_active")


def measured():
    def supported():
        return dict(support=True, bearing_verified=True, bearing_force_n=2., air=False,
                    ground_contact=True, top_surface_contact=False, top_contact=False,
                    obstacle_pair_active=False, contact_surface="GROUND")
    legs = {leg: supported() for leg in ("FL", "FR", "RL", "RR")}
    legs["RR"].update(support=False, bearing_force_n=0., air=True, ground_contact=False,
                      contact_surface="NONE", current_lift_valid=True, lift_established=True,
                      motion_continuation_allowed=True, within_top_xy=True,
                      within_lateral_span=True, clearance_m=.020)
    hist = {key: {leg: False for leg in legs}
            for key in ("active_lift", "front_edge_crossed", "placed")}
    hist["active_lift"]["RR"] = hist["front_edge_crossed"]["RR"] = True
    ev = dict(valid=True, termination_reason=None, physics_tick=100, current_legs=legs,
              history=hist, transfer_roles={leg: dict(preparation_ready=False) for leg in legs})
    task = dict(physical_evaluator=ev, termination_reason=None)
    obs = dict(joints={name: dict(position_deg=0.) for name in SERVO_ORDER})
    return task, obs


def facts(task, observation, **kwargs):
    return rr_capture_transfer_context(task=task, observation=observation,
                                       support_spec=SUPPORT_SPEC, **kwargs)


def assist_state(mode="DESCEND"):
    assist = RRHipOnlyCaptureAssist()
    modes = ("WAIT", "DESCEND", "HOLD", "BLOCKED", "RELEASE", "RELEASED")
    assist.state.update(mode=float(modes.index(mode)), initialized=float(mode != "WAIT"),
                        release_fraction=float(mode == "RELEASED"))
    return assist.snapshot()


def top_rr(task, *, support=True):
    row = task["physical_evaluator"]["current_legs"]["RR"]
    row.update(air=False, ground_contact=False, top_contact=True, top_surface_contact=True,
               obstacle_pair_active=True, contact_surface="TOP", support=support,
               bearing_force_n=2. if support else .05, clearance_m=-.001)


def test_air_top_candidate_is_not_contact_support_or_rl_permission():
    task, obs = measured()
    result = facts(task, obs)
    assert all(type(result[name]) is bool for name in FACTS)
    assert result["rr_lift_carry"] and result["rr_top_reachable"]
    assert not result["rr_top_contact"] and not result["rr_current_bearing"]
    assert not result["rl_transfer_ready"] and not result["air_is_support"]
    assert set(result["other_measured_supports"]) == {"FL", "FR", "RL"}


@pytest.mark.parametrize("placed", [False, True])
def test_historic_placement_does_not_make_current_air_bearing(placed):
    task, obs = measured()
    task["physical_evaluator"]["history"]["placed"]["RR"] = placed
    task["physical_evaluator"]["transfer_roles"]["RL"]["preparation_ready"] = True
    result = facts(task, obs)
    assert result["rr_history_placed"] is placed
    assert not result["rr_current_bearing"] and not result["rl_transfer_ready"]


def test_ground_rejects_old_qualification_as_carry_or_first_capture_candidate():
    task, obs = measured()
    task["physical_evaluator"]["current_legs"]["RR"].update(
        ground_contact=True, air=False, support=True, bearing_force_n=2.)
    result = facts(task, obs, assist_snapshot=assist_state())
    for name in ("rr_lift_carry", "rr_top_reachable", "rr_current_bearing",
                 "rr_capture_recovery_allowed"):
        assert not result[name]


@pytest.mark.parametrize("field", ["current_lift_valid", "within_top_xy", "within_lateral_span"])
def test_current_capture_requirements_cannot_be_replaced_by_history(field):
    task, obs = measured()
    task["physical_evaluator"]["current_legs"]["RR"][field] = False
    task["physical_evaluator"]["history"]["placed"]["RR"] = True
    assert not facts(task, obs)["rr_top_reachable"]


def test_top_contact_can_exist_before_verified_bearing():
    task, obs = measured()
    top_rr(task, support=False)
    result = facts(task, obs)
    assert result["rr_top_contact"]
    assert not result["rr_current_bearing"] and not result["rl_transfer_ready"]


def test_real_current_top_bearing_does_not_need_fresh_air_qualification():
    task, obs = measured()
    top_rr(task)
    task["physical_evaluator"]["current_legs"]["RR"]["current_lift_valid"] = False
    result = facts(task, obs)
    assert result["rr_current_bearing"] and result["rr_top_reachable"]


@pytest.mark.parametrize("kind", ["below_top_air", "wall_contact"])
def test_xy_alone_does_not_make_below_top_or_wall_a_reachable_capture(kind):
    task, obs = measured()
    rr = task["physical_evaluator"]["current_legs"]["RR"]
    if kind == "below_top_air":
        rr["clearance_m"] = -.2
    else:
        rr.update(air=False, obstacle_pair_active=True, contact_surface="FRONT",
                  top_contact=False, top_surface_contact=False)
    assert not facts(task, obs)["rr_top_reachable"]


@pytest.mark.parametrize("ready", [False, True])
def test_rl_permission_requires_current_rr_bearing_and_the_rl_role(ready):
    task, obs = measured()
    top_rr(task)
    roles = task["physical_evaluator"]["transfer_roles"]
    roles["RR"]["preparation_ready"] = True
    roles["RL"]["preparation_ready"] = ready
    result = facts(task, obs)
    assert result["rr_current_bearing"]
    assert result["rl_transfer_ready"] is ready


@pytest.mark.parametrize("field,value", [("support", False), ("bearing_verified", False),
    ("air", True), ("top_surface_contact", False), ("top_contact", False),
    ("ground_contact", True)])
def test_inconsistent_or_unverified_contact_does_not_grant_current_rr_bearing(field, value):
    task, obs = measured()
    top_rr(task)
    task["physical_evaluator"]["current_legs"]["RR"][field] = value
    assert not facts(task, obs)["rr_current_bearing"]


@pytest.mark.parametrize("remaining", [1, 2, 3])
def test_reachable_counts_only_current_verified_other_supports(remaining):
    task, obs = measured()
    legs = task["physical_evaluator"]["current_legs"]
    for leg in ("FL", "FR", "RL")[remaining:]:
        legs[leg].update(air=True, support=False, ground_contact=False, bearing_force_n=0.)
        task["physical_evaluator"]["history"]["placed"][leg] = True
    result = facts(task, obs)
    assert len(result["other_measured_supports"]) == remaining
    assert result["rr_top_reachable"] is (remaining >= 2)


@pytest.mark.parametrize("mode", ["WAIT", "DESCEND", "HOLD", "BLOCKED", "RELEASE", "RELEASED"])
def test_only_current_descend_can_claim_local_recovery(mode):
    task, obs = measured()
    result = facts(task, obs, assist_snapshot=assist_state(mode))
    assert result["rr_capture_recovery_allowed"] is (mode == "DESCEND")


@pytest.mark.parametrize("reason", ["retired", "invalid", "physical_abort", "task_abort", "outside"])
def test_old_descend_does_not_override_current_ineligibility(reason):
    task, obs = measured()
    ev = task["physical_evaluator"]
    snapshot = assist_state("RELEASED" if reason == "retired" else "DESCEND")
    if reason == "invalid":
        ev["valid"] = False
    elif reason == "physical_abort":
        ev["termination_reason"] = "BODY_COLLISION"
    elif reason == "task_abort":
        task["termination_reason"] = "GLOBAL_TIME_LIMIT"
    elif reason == "outside":
        ev["current_legs"]["RR"]["within_top_xy"] = False
    assert not facts(task, obs, assist_snapshot=snapshot)["rr_capture_recovery_allowed"]


def test_wheel_guidance_off_is_default_and_context_has_no_mutations():
    task, obs = measured()
    before = deepcopy((task, obs))
    default = facts(task, obs)
    assert not default["fl_wheel_guidance_active"]
    assert default == facts(task, obs, wheel_mode="off")
    assert (task, obs) == before
    with pytest.raises(ValueError, match="unknown RR carry wheel"):
        facts(task, obs, wheel_mode="unreviewed")


def test_explicit_guidance_still_requires_real_fl_support():
    task, obs = measured()
    assert facts(task, obs, wheel_mode="rr_carry_forward_v1")["fl_wheel_guidance_active"]
    task["physical_evaluator"]["current_legs"]["FL"].update(
        air=True, support=False, ground_contact=False, bearing_force_n=0.)
    assert not facts(task, obs, wheel_mode="rr_carry_forward_v1")["fl_wheel_guidance_active"]


@pytest.mark.parametrize("experiment", ["ppo_p05_hip_only_continuation_v1",
                                        "ppo_rr_capture_then_rl_transfer_v1"])
def test_existing_diagonal_receiver_and_bridge_roles_are_not_swapped(experiment):
    spec = yaml.safe_load((ROOT / "configs" / experiment / "stage_task_spec.yaml").read_text())
    validate_config(spec)
    mapping = spec["transfer_roles"]["mapping"]
    assert DIAGONAL["RR"] == mapping["RR"]["diagonal_receiving_side"] == "FL"
    assert DIAGONAL["RL"] == mapping["RL"]["diagonal_receiving_side"] == "FR"
    assert set(mapping["RR"]["preferred_bridge_contacts"]) == {"FR", "RL"}
    assert set(mapping["RL"]["preferred_bridge_contacts"]) == {"FL", "RR"}


@pytest.mark.parametrize("fraction,support", [(0., False), (.95, False), (1., True), (10., True)])
def test_valid_evaluator_support_agrees_with_assist_noise_floor(fraction, support):
    task, obs = measured()
    top_rr(task)
    rr = task["physical_evaluator"]["current_legs"]["RR"]
    rr.update(bearing_force_n=fraction * FORCE_FLOOR, support=support)
    task_fact = facts(task, obs)
    assist_fact = rr_capture_assist_context(task=task, observation=obs,
        source_frame=SimpleNamespace(state_id="P09"), physics_tick=101,
        support_spec=SUPPORT_SPEC)
    assert task_fact["rr_current_bearing"] == assist_fact["current_top_bearing"] == support


@pytest.mark.parametrize("force", [None, float("nan"), float("inf"), True, 0.])
def test_claimed_support_without_verified_finite_force_fails_both_contexts(force):
    task, obs = measured()
    top_rr(task)
    task["physical_evaluator"]["current_legs"]["RR"]["bearing_force_n"] = force
    task_fact = facts(task, obs)
    assist_fact = rr_capture_assist_context(task=task, observation=obs,
        source_frame=SimpleNamespace(state_id="P09"), physics_tick=101,
        support_spec=SUPPORT_SPEC)
    assert not task_fact["rr_current_bearing"]
    assert not assist_fact["current_top_bearing"]


@pytest.mark.parametrize("field,value", [("clearance_m", float("nan")),
                                         ("hip_actual", float("inf"))])
def test_nonfinite_physical_measurement_is_rejected(field, value):
    task, obs = measured()
    if field == "hip_actual":
        obs["joints"][SERVO_ORDER[6]]["position_deg"] = value
    else:
        task["physical_evaluator"]["current_legs"]["RR"][field] = value
    with pytest.raises(ValueError, match="finite"):
        facts(task, obs)

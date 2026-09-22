"""Finite P05 nominal recovery: synthetic CPU wiring, never physical success."""
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path

import pytest

from wlr50_clean.fsm import state_spec
from wlr50_clean.fsm.state_spec import load_fsm_spec
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.ppo.semantic_supervisor import (
    LEG_ORDER, NominalMotionProvider, P05_PREEDGE_RECOVERY_MODE,
    _capture_continuation_status, _p05_preedge_recovery_enabled, load_task_spec,
)
from wlr50_clean.reference.motion_contract import load_motion_contract

ROOT = Path(state_spec.__file__).resolve().parents[3]
CFG = ROOT / "configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml"
KEY = "p05_preedge_approach_recovery"
ROLL = (.3,) * 4


def configured(enabled):
    spec = load_task_spec(CFG)
    spec["nominal"].pop(KEY, None)
    if enabled:
        spec["nominal"][KEY] = P05_PREEDGE_RECOVERY_MODE
    return spec


def task(tick=3600, age=30.):
    legs = {leg: dict(support=True, bearing_verified=True, air=False,
        ground_contact=True, top_surface_contact=False, top_contact=False,
        obstacle_pair_active=False, within_top_xy=False, within_lateral_span=True,
        clearance_m=.0077, front_distance_m=-.034) for leg in LEG_ORDER}
    legs["FL"].update(support=False, air=True, ground_contact=False)
    legs["FR"].update(ground_contact=False, top_surface_contact=True,
                      top_contact=True, obstacle_pair_active=True, within_top_xy=True)
    ev = dict(valid=True, physics_tick=tick, simulation_time_s=tick/120.,
        termination_reason=None, physical_evidence_status="VERIFIED", current_legs=legs,
        history=dict(placed={leg: leg == "FR" for leg in LEG_ORDER},
            active_lift={leg: leg in ("FR", "FL") for leg in LEG_ORDER},
            front_edge_crossed={leg: leg == "FR" for leg in LEG_ORDER}))
    return dict(stage_id="P05", stage_elapsed_s=age, physical_evaluator=ev,
        termination_reason=None, transfer_roles={}, fl_capture_pending=False,
        allow_capture_continuation=False,
        p05_local_deadline_warning=type(age) in (int, float) and age >= 30.)


@pytest.fixture(scope="module")
def mature():
    contract = load_motion_contract(ROOT / "configs/recording_motion_contract.json")
    fsm = load_fsm_spec(ROOT / "configs/fsm_states.yaml")
    provider = NominalMotionProvider(contract, spec=configured(True), fsm_spec=fsm)
    last_stop = None
    for tick in range(3600):
        command = provider.evaluate(task(tick, tick/120.))
        sample = provider._continuous_layers[-1]["sample"]
        if sample.atomic_groups and sample.full12[8:] == (0.,)*4:
            if any(set(group.channels).intersection(WHEEL_ORDER) for group in sample.atomic_groups):
                last_stop = sample
    assert provider.endpoint_issued and command[8:] == (0.,)*4 and last_stop is not None
    return provider, last_stop


def status(provider, value=None, *, previous_tick=3599, previous_endpoint=True):
    return provider._p05_preedge_recovery_status(task() if value is None else value,
        prior_source_endpoint_issued=previous_endpoint, prior_observation_tick=previous_tick)


def clone(provider):
    # The immutable parsed source contains mappingproxy guards.
    return deepcopy(provider, {id(provider.contract): provider.contract,
        id(provider._reference_fsm_spec): provider._reference_fsm_spec})


def test_opt_in_is_explicit_and_old_mode_default_off():
    assert not _p05_preedge_recovery_enabled(configured(False))
    assert _p05_preedge_recovery_enabled(configured(True))
    value = configured(True)
    value["nominal"][KEY] = "unknown"
    with pytest.raises(ValueError):
        _p05_preedge_recovery_enabled(value)
    value = configured(True)
    value.pop("capture_continuation_semantics")
    with pytest.raises(ValueError):
        _p05_preedge_recovery_enabled(value)


@pytest.mark.parametrize("phase", [f"P{i:02}" for i in range(1, 14)])
def test_only_p05(mature, phase):
    provider, _ = mature
    value = task(); value["stage_id"] = phase
    assert status(provider, value)["eligible"] is (phase == "P05")


@pytest.mark.parametrize("age,eligible", [(29.999, False), (30., True),
    (39.999, True), (40., False), (100., False), (math.nan, False), (True, False), (None, False)])
def test_absolute_window_never_renews(mature, age, eligible):
    provider, _ = mature
    assert status(provider, task(age=age))["eligible"] is eligible


@pytest.mark.parametrize("key,value", [
    ("clearance_m", 0.), ("clearance_m", -.001), ("clearance_m", math.nan),
    ("clearance_m", None), ("clearance_m", True), ("front_distance_m", -.10001),
    ("front_distance_m", .00501), ("front_distance_m", math.nan),
    ("within_lateral_span", False), ("air", False), ("ground_contact", True),
    ("top_surface_contact", True), ("obstacle_pair_active", True)])
def test_current_geometry_or_contact_rejects(mature, key, value):
    provider, _ = mature
    item = task(); item["physical_evaluator"]["current_legs"]["FL"][key] = value
    assert not status(provider, item)["eligible"]


@pytest.mark.parametrize("distance", [-.1, -.005, 0., .005])
def test_preedge_boundaries_have_no_minus5mm_dead_zone(mature, distance):
    provider, _ = mature
    item = task(); item["physical_evaluator"]["current_legs"]["FL"]["front_distance_m"] = distance
    assert status(provider, item)["eligible"]


@pytest.mark.parametrize("field,leg,value", [("active_lift", "FL", False),
    ("placed", "FR", False), ("placed", "FL", True), ("front_edge_crossed", "FL", True)])
def test_history_is_not_fabricated(mature, field, leg, value):
    provider, _ = mature
    item = task(); item["physical_evaluator"]["history"][field][leg] = value
    assert not status(provider, item)["eligible"]


@pytest.mark.parametrize("fault", ["one_support", "unverified", "airborne_support",
    "invalid", "task_abort", "eval_abort", "unverified_evidence", "missing_FL"])
def test_safety_support_and_missing_evidence(mature, fault):
    provider, _ = mature
    item = task(); ev = item["physical_evaluator"]
    if fault == "one_support":
        for leg in ("RL", "RR"): ev["current_legs"][leg]["support"] = False
    elif fault in ("unverified", "airborne_support"):
        for leg in ("RL", "RR"):
            ev["current_legs"][leg]["bearing_verified" if fault == "unverified" else "air"] = fault != "unverified"
    elif fault == "invalid": ev["valid"] = False
    elif fault == "task_abort": item["termination_reason"] = "TASK_FAILURE_BODY_COLLISION"
    elif fault == "eval_abort": ev["termination_reason"] = "TASK_FAILURE_BODY_COLLISION"
    elif fault == "unverified_evidence": ev["physical_evidence_status"] = "UNVERIFIED"
    else: ev["current_legs"].pop("FL")
    assert not status(provider, item)["eligible"]


@pytest.mark.parametrize("previous_tick", [None, True, 3600, 3601, 3598])
def test_missing_nonincreasing_or_skipped_tick_is_not_endpoint_evidence(mature, previous_tick):
    provider, _ = mature
    assert not status(provider, previous_tick=previous_tick)["eligible"]


def test_endpoint_needs_prior_issue_and_explicitly_is_not_ack_verification(mature):
    provider, _ = mature
    assert not status(provider, previous_endpoint=False)["eligible"]
    value = clone(provider); value.endpoint_issued = False
    assert not status(value)["eligible"]
    info = status(provider)
    assert info["eligible"] and not info["independent_ack_verified"]
    assert info["absolute_window_s"] == (30., 40.) and info["source_tick_consecutive"]


def test_nominal_intercept_changes_only_wheels_no_phase_or_source_replay(mature):
    enabled = clone(mature[0]); disabled = clone(enabled)
    disabled._p05_preedge_recovery = False
    item = task(); original = deepcopy(item)
    expected, actual = disabled.evaluate(deepcopy(item)), enabled.evaluate(item)
    assert actual[:8] == expected[:8] and actual[8:] == ROLL and expected[8:] == (0.,)*4
    assert enabled.tracking_servo_names == disabled.tracking_servo_names
    assert enabled.normal_drive_bias_full12 == disabled.normal_drive_bias_full12
    assert item == original and enabled.state_id == "P05"
    assert [layer["stage"] for layer in enabled._continuous_layers] == ["P05"]
    assert enabled._source_motion._tick_index == disabled._source_motion._tick_index
    assert enabled._continuous_layers[-1]["sample"] == disabled._continuous_layers[-1]["sample"]
    assert not _capture_continuation_status(enabled.spec, item["physical_evaluator"], "P05")["allow_capture_continuation"]
    info = enabled.nominal_suggestion_diagnostics[KEY]
    assert info["eligible"] and not info["policy_residual_restricted"]
    assert not info["phase_or_capture_credit_awarded"] and not info["recovery_clock_reset_or_latch"]


def test_fresh_authored_stop_wins_at_interception(mature, monkeypatch):
    provider, authored_stop = clone(mature[0]), mature[1]
    original = provider._continuous_advisory
    def with_stop(*args, **kwargs):
        proposed, tracking = original(*args, **kwargs)
        layer = provider._continuous_layers[-1]
        layer["sample"] = replace(layer["sample"], atomic_groups=authored_stop.atomic_groups)
        return proposed[:8] + (0.,)*4, tracking
    monkeypatch.setattr(provider, "_continuous_advisory", with_stop)
    assert provider.evaluate(task())[8:] == (0.,)*4
    info = provider.nominal_suggestion_diagnostics[KEY]
    assert not info["eligible"] and info["fresh_source_wheel_owners"]
    assert "fresh_source_wheel_owner_absent" in info["reasons"]


def test_loss_and_expiry_remove_advice_without_renewal(mature):
    provider = clone(mature[0])
    assert provider.evaluate(task())[8:] == ROLL
    value = task(3601, 30. + 1/120.)
    value["physical_evaluator"]["current_legs"]["FL"]["clearance_m"] = 0.
    assert provider.evaluate(value)[8:] == (0.,)*4
    assert provider.evaluate(task(3602, 30. + 2/120.))[8:] == ROLL
    assert provider.evaluate(task(3603, 40.))[8:] == (0.,)*4
    assert provider.evaluate(task(3604, 40. + 1/120.))[8:] == (0.,)*4
    assert provider._continuous_layers[-1]["ticks"] == 3605


def test_postcross_existing_capture_branch_remains_separate(mature):
    enabled = clone(mature[0]); disabled = clone(enabled)
    disabled._p05_preedge_recovery = False
    value = task()
    value["physical_evaluator"]["history"]["front_edge_crossed"]["FL"] = True
    value["physical_evaluator"]["current_legs"]["FL"]["within_top_xy"] = True
    value["transfer_roles"] = {"FL": {"pending_capture": True, "observed_support_contacts": ("FR", "RL", "RR")}}
    assert enabled.evaluate(deepcopy(value)) == disabled.evaluate(deepcopy(value))
    assert not enabled.nominal_suggestion_diagnostics[KEY]["eligible"]

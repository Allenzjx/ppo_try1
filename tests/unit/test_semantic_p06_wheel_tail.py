"""P06 owner-local finite tail, not physical success or a new dispatch gate."""
from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path

import pytest
import yaml

from test_semantic_supervisor import observation, leg_state
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.semantic_supervisor import (
    DEFAULT_TASK_SPEC_PATH, NominalMotionProvider, P06_TAIL_MODE,
    SemanticObservationError, TaskEvaluator, TaskStageSupervisor, load_task_spec,
)
from wlr50_clean.reference.motion_contract import load_motion_contract


ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs/ppo_semantic_v3"
SPEC = CFG / "stage_task_spec.yaml"
ZERO = (0.,)*12


@pytest.fixture(scope="module")
def contract():
    return load_motion_contract(ROOT / "configs/recording_motion_contract.json")


def _spec(enabled=True):
    spec = load_task_spec(SPEC)
    if enabled:
        spec["nominal"]["p06_wheel_tail_semantics"] = P06_TAIL_MODE
    else:
        spec["nominal"].pop("p06_wheel_tail_semantics", None)
    return spec


def _provider(contract, enabled=True):
    return NominalMotionProvider(contract, spec=_spec(enabled))


def _measured(tick=0, phase="P06", *, rl=-.30, rr=None, lateral=True):
    obs = observation(tick)
    for joint in obs["joints"].values():
        joint["command_deg"] = 0.
    for leg, distance in (("RL", rl), ("RR", rl if rr is None else rr)):
        leg_state(obs, leg, x=obs["obstacle"]["front_x_m"]+distance)
    if not lateral:
        for key in ("center_w_m", "bottom_w_m"):
            obs["wheels"][WHEEL_ORDER[2]][key][1] = 2.
    ev = TaskEvaluator(spec=_spec()).observe(obs)
    assert ev["valid"] and ev["termination_reason"] is None
    return {"stage_id": phase, "termination_reason": None, "physical_evaluator": ev}, obs


def _run(p, count, *, tick=0, phase="P06", **geometry):
    # Repeated complete stationary sensor geometry, freshly stamped each tick.
    # This is a nominal-scheduler test, not a fabricated improving trajectory.
    task, obs = _measured(tick, phase, **geometry)
    for current in range(tick, tick+count):
        obs.update(physics_tick=current, simulation_time_s=current/120.)
        task["physical_evaluator"].update(physics_tick=current, simulation_time_s=current/120.)
        previous = p.nominal_full12
        command = p.evaluate(task, obs)
        assert all(abs(a-b) <= .025+1e-12 for a, b in zip(command[8:], previous[8:]))
    return command, task, obs


def _tail(p):
    return p.nominal_suggestion_diagnostics["p06_wheel_tail"]


def _retirement(p):
    return p.nominal_suggestion_diagnostics["p06_rolling_retirement"]


def _layer(p):
    return next(layer for layer in p._continuous_layers if layer["stage"] == "P06")


def _state(p):
    return (p.nominal_full12, p.tracking_servo_names, p.state_id,
            p._source_motion._tick_index, deepcopy(p.nominal_suggestion_diagnostics),
            [(layer["stage"], layer["ticks"], layer["motion"]._tick_index,
              layer["last"], layer["sample"], set(layer["touched"]),
              layer.get("rolling_retirement_peak")) for layer in p._continuous_layers])


@pytest.fixture(scope="module")
def before_endpoint(contract):
    p = _provider(contract)
    end = round(contract.phase("P06").active_duration_s*contract.physics_hz)
    _run(p, end)
    assert _layer(p)["sample"].tick_index == end-1
    assert not _layer(p)["sample"].endpoint_issued
    assert p.nominal_full12[8:] == (.3,)*4
    return p, end


def test_actual_source_three_waypoint_timing_and_tail_only_at_original_endpoint(contract):
    phase = contract.phase("P06")
    assert len(phase.waypoints) == 3
    assert [row.time_s for row in phase.waypoints[:2]] == [0., 0.]
    assert phase.start_full12[8:] == phase.end_full12[8:] == (0.,)*4
    assert phase.waypoints[1].full12[8:] == (.3,)*4
    end = round(phase.active_duration_s*120)
    assert end == 3064
    new, old = _provider(contract), _provider(contract, False)
    task, obs = _measured()
    for tick in range(end+26):
        obs.update(physics_tick=tick, simulation_time_s=tick/120.)
        task["physical_evaluator"].update(physics_tick=tick, simulation_time_s=tick/120.)
        current, legacy = new.evaluate(task, obs), old.evaluate(task, obs)
        assert current[:8] == legacy[:8]
        assert new.tracking_servo_names == old.tracking_servo_names
        assert new.elapsed_s == old.elapsed_s
        assert new.endpoint_issued == old.endpoint_issued
        left, right = _layer(new), _layer(old)
        for field in ("last", "sample", "touched", "ticks"):
            assert left[field] == right[field]
        assert left["motion"]._tick_index == right["motion"]._tick_index == tick+1
        if tick < end:
            assert current == legacy
            assert not _tail(new)["finite_source_tail_replaced"]
        else:
            assert current[8:] == (.3,)*4
            assert _tail(new)["source_endpoint_issued"]
            assert _tail(new)["finite_source_tail_replaced"]
            assert left["sample"].full12[8:] == left["last"][8:] == (0.,)*4
    assert old.nominal_full12[8:] == pytest.approx((0.,)*4)
    assert _retirement(new)["peak_fraction"] == 0.
    assert TaskStageSupervisor(SPEC).predicate("rear_approach", task["physical_evaluator"]) < 1.
    assert not task["physical_evaluator"]["success"]


@pytest.mark.parametrize("mode", ["absent", None])
def test_absent_and_none_optin_preserve_old_zero_tail_and_v2(contract, mode):
    spec = _spec(False)
    if mode is None:
        spec["nominal"]["p06_wheel_tail_semantics"] = None
    p = NominalMotionProvider(contract, spec=spec)
    command, _, _ = _run(p, 3090)
    assert command[8:] == pytest.approx((0.,)*4)
    assert p.endpoint_issued and _retirement(p)["wheel_gain"] == 1.
    assert "p06_wheel_tail" not in p.nominal_suggestion_diagnostics
    v2 = NominalMotionProvider(contract, spec=load_task_spec(DEFAULT_TASK_SPEC_PATH))
    assert not v2.nominal_suggestion_diagnostics
    for _ in range(3090):
        command = v2.evaluate("P06")
    assert command[8:] == pytest.approx((0.,)*4)


def test_current_recorded_far_workspace_stays_uncompleted_across_source_stop(before_endpoint):
    p, end = deepcopy(before_endpoint)
    task, obs = _measured(end, rl=-.350562, rr=-.256393)
    before = deepcopy(task)
    command = p.evaluate(task, obs)
    assert command[8:] == (.3,)*4 and p.endpoint_issued
    assert _retirement(p)["wheel_gain"] == 1.
    assert task == before  # Recorded magnitudes are a same-state test, not a rollout.
    assert not task["physical_evaluator"]["success"]


def test_partial_full_retirement_and_retreat_do_not_reactivate_tail(before_endpoint):
    p, tick = deepcopy(before_endpoint)
    _run(p, 20, tick=tick, rl=-.2175)
    assert p.nominal_full12[8:] == pytest.approx((.15,)*4)
    assert _retirement(p)["peak_fraction"] == pytest.approx(.5)
    assert _tail(p)["finite_source_tail_replaced"]
    _run(p, 20, tick=tick+20, rl=-.215)
    assert p.nominal_full12[8:] == pytest.approx((0.,)*4)
    assert not _tail(p)["finite_source_tail_replaced"] and _tail(p)["status"] == "retired"
    _run(p, 20, tick=tick+40, rl=-.4)
    assert _retirement(p)["peak_fraction"] == pytest.approx(1.)
    assert p.nominal_full12[8:] == pytest.approx((0.,)*4)


@pytest.mark.parametrize("rl,rr,lateral", [(-.3, -.2, True), (-.21, -.3, True), (-.21, -.21, False)])
def test_lagging_rear_or_invalid_lateral_cannot_earn_new_retirement(before_endpoint, rl, rr, lateral):
    p, tick = deepcopy(before_endpoint)
    _run(p, 20, tick=tick, rl=rl, rr=rr, lateral=lateral)
    assert _retirement(p)["peak_fraction"] == 0.
    assert p.nominal_full12[8:] == (.3,)*4


@pytest.mark.parametrize("before_source_expiry", [True, False])
def test_early_handoff_keeps_layer_clock_and_tail_uses_p06_not_current_endpoint(contract, before_source_expiry):
    p = _provider(contract)
    end = round(contract.phase("P06").active_duration_s*120)
    count = end-2 if before_source_expiry else end+2
    _run(p, count)
    layer = _layer(p)
    incoming, tracking = p.nominal_full12, p.tracking_servo_names
    task, obs = _measured(count, "P08")  # Real P08 has no new wheel owner.
    assert p.evaluate(task, obs) == incoming
    assert p.tracking_servo_names == tracking
    _run(p, 30, tick=count+1, phase="P08")
    assert _layer(p) is layer and len(p._continuous_layers) == 2
    assert layer["ticks"] == count+31
    assert _tail(p)["source_endpoint_issued"] and _tail(p)["finite_source_tail_replaced"]
    assert p.nominal_full12[8:] == (.3,)*4


def test_later_real_p07_reverse_and_stop_own_only_fr_then_p09_owns_all(before_endpoint):
    p, tick = deepcopy(before_endpoint)
    _run(p, 2, tick=tick)
    tick += 2
    _run(p, 100, tick=tick, phase="P07")
    assert p.nominal_full12[9] == pytest.approx(-.63)
    assert tuple(p.nominal_full12[i] for i in (8, 10, 11)) == (.3,)*3
    _run(p, 120, tick=tick+100, phase="P07")
    assert p.nominal_full12[9] == pytest.approx(0.)
    assert tuple(p.nominal_full12[i] for i in (8, 10, 11)) == (.3,)*3
    tick += 220
    _run(p, 280, tick=tick, phase="P09")
    assert p.nominal_full12[8:] == pytest.approx((.3,)*4)
    # The final FL=-1.07 segment also needs the unchanged 3 rad/s^2
    # provider slew to settle after the source issues its zero endpoint.
    _run(p, 650, tick=tick+280, phase="P09")
    assert p.nominal_full12[8:] == pytest.approx((0.,)*4)
    assert _tail(p)["finite_source_tail_replaced"]  # Layer contribution, not final command!
    assert _layer(p)["last"][8:] == (0.,)*4


def test_p13_endpoint_stop_has_priority_over_still_active_p06_tail(before_endpoint, contract):
    p, tick = deepcopy(before_endpoint)
    _run(p, 1, tick=tick)
    end13 = round(contract.phase("P13").active_duration_s*120)
    _run(p, end13+80, tick=tick+1, phase="P13")
    assert p.endpoint_issued
    assert p.nominal_full12[8:] == pytest.approx((0.,)*4)
    assert _tail(p)["source_endpoint_issued"]
    assert _tail(p)["finite_source_tail_replaced"]


def test_qualified_carry_override_still_wins_after_complete_retirement(before_endpoint):
    from test_semantic_continuous_v3 import live, whole_body_lift
    p, tick = deepcopy(before_endpoint)
    _run(p, 20, tick=tick, rl=-.21)
    ev, obs = TaskEvaluator(SPEC), live()
    for leg in ("RR", "RL"):
        leg_state(obs, leg, x=.3)
    ev.observe(obs)
    obs = whole_body_lift(ev, obs, top_clearance=.08)
    assert ev.snapshot["history"]["active_lift"]["RR"]
    for _ in range(18):
        obs["physics_tick"] += 1
        obs["simulation_time_s"] = obs["physics_tick"]/120.
        command = p.evaluate({"stage_id": "P09", "termination_reason": None,
                              "physical_evaluator": ev.observe(obs)}, obs)
    assert _retirement(p)["wheel_gain"] == 0.
    assert command[8:] == pytest.approx((.3,)*4)


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("field", ["task", "evaluator"])
def test_terminal_first_tail_tick_advances_original_source_but_never_earns_or_activates(before_endpoint, partial, field):
    p, tick = deepcopy(before_endpoint)
    if partial:
        # Real measured peak is obtained before terminal, not assigned manually.
        p.evaluate(*_measured(tick, rl=-.2175))
        tick += 1
    old_nominal, old_peak = p.nominal_full12, _retirement(p)["peak_fraction"]
    previous_ticks = _layer(p)["ticks"]
    task, obs = _measured(tick, rl=-.21)
    if field == "task":
        task["termination_reason"] = "FALL"
    else:
        task["physical_evaluator"]["termination_reason"] = "BODY_COLLISION"
    preserved = deepcopy(task)
    command = p.evaluate(task, obs)
    assert _layer(p)["ticks"] == previous_ticks+1
    assert _layer(p)["sample"].endpoint_issued
    assert _retirement(p)["peak_fraction"] == old_peak
    assert _retirement(p)["status"] == "terminal_no_new_retirement"
    assert _tail(p)["status"] == "terminal_no_tail"
    assert not _tail(p)["finite_source_tail_replaced"]
    assert command[8:] == pytest.approx(tuple(max(0., value-.025) for value in old_nominal[8:]))
    assert task == preserved


@pytest.mark.parametrize("bad", ["missing", "invalid", "nan", "stale", "lateral_type"])
def test_invalid_tail_measurement_fails_before_clock_or_nominal_mutation(before_endpoint, bad):
    p, tick = deepcopy(before_endpoint)
    task, obs = _measured(tick)
    if bad == "missing":
        task.pop("physical_evaluator")
    elif bad == "invalid":
        task["physical_evaluator"]["valid"] = False
    elif bad == "nan":
        task["physical_evaluator"]["current_legs"]["RR"]["front_distance_m"] = float("nan")
    elif bad == "stale":
        obs["physics_tick"] += 1
    else:
        task["physical_evaluator"]["current_legs"]["RL"]["within_lateral_span"] = 1
    saved = _state(p)
    with pytest.raises(SemanticObservationError):
        p.evaluate(task, obs)
    assert _state(p) == saved


@pytest.mark.parametrize("phase", ["P09", "P10", "P12"])
def test_direct_teacher_handoff_cannot_invent_p06_layer_or_tail(contract, phase):
    incoming = ZERO[:8]+(.13,)*4
    p = NominalMotionProvider.from_handoff(contract, spec=_spec(), stage_id=phase,
        nominal_full12=incoming, tracking_servo_names=())
    command = p.evaluate(*_measured(80, phase))
    assert command[8:] == incoming[8:]
    assert [layer["stage"] for layer in p._continuous_layers] == [phase]
    assert not _tail(p)["layer_present"] and not _tail(p)["finite_source_tail_replaced"]
    assert _tail(p)["status"] == "not_applicable_no_P06_layer"
    assert _retirement(p)["peak_fraction"] is None


def test_direct_p06_handoff_starts_current_source_and_slews_actual_receipt(contract):
    p = NominalMotionProvider.from_handoff(contract, spec=_spec(), stage_id="P06",
        nominal_full12=ZERO[:8]+(.13,)*4, tracking_servo_names=())
    command = p.evaluate(*_measured(80))
    assert command[8:] == pytest.approx((.155,)*4)
    assert _layer(p)["sample"].tick_index == 0
    assert not _tail(p)["source_endpoint_issued"]
    assert _retirement(p)["source_observation_tick"] == 80


def test_tail_diagnostics_are_copied_and_label_pre_owner_pre_slew(before_endpoint):
    p, tick = deepcopy(before_endpoint)
    p.evaluate(*_measured(tick))
    diag = p.nominal_suggestion_diagnostics
    tail = diag["p06_wheel_tail"]
    assert tail["enabled"] and tail["source_rolling_rad_s"] == (.3,)*4
    assert "before_later_owners_and_slew_not_applied_target" in tail["timing"]
    tail["finite_source_tail_replaced"] = False
    assert _tail(p)["finite_source_tail_replaced"]


def test_real_projected_negative_residual_can_cancel_tail_with_one_frozen_dispatch(before_endpoint, monkeypatch):
    from test_actuator_target_effect import _adapter
    from test_semantic_residual_adapter import dispatch, plan
    from wlr50_clean.ppo.actuator_target_effect import actuator_target_audit_request, build_actuator_target_effect_audit
    from wlr50_clean.ppo.semantic_backend import build_semantic_projector
    p, tick = deepcopy(before_endpoint)
    nominal = p.evaluate(*_measured(tick))
    # P06 front/rear wheel caps are 1.2/.6; request the same -.3 rad/s
    # cancellation on every wheel without assuming identical action scales.
    raw = ZERO[:8]+(-math.atanh(.25),)*2+(-math.atanh(.5),)*2
    projector = build_semantic_projector(CFG / "execution_profile.yaml")
    previous_residual = ZERO
    for _ in range(200):
        projection = projector.project(raw, state_id="P06", nominal_action_full12=nominal,
            reference_action_full12=nominal, reference_delta_full12=ZERO, dt_s=1/120.,
            previous_projected_residual_full12=previous_residual)
        previous_residual = projection.safe_projected_residual_full12
    adapter = _adapter()
    advance = adapter.servo_target_mapper.advance
    calls = []
    def counted(*args, **kwargs):
        calls.append(args[0])
        return advance(*args, **kwargs)
    monkeypatch.setattr(adapter.servo_target_mapper, "advance", counted)
    previous = tuple(adapter._final_drive_servo_deg.values())
    actuation = plan(projection.safe_projected_residual_full12, nominal=nominal)
    ack = dispatch(adapter, actuation, 1)
    audit = build_actuator_target_effect_audit(adapter=adapter, actuation=actuation, raw_ack=ack,
        previous_final_drive_servo_deg=previous, source_phase_id="P06",
        policy_request=actuator_target_audit_request("P06", raw, (1,)*12))
    assert len(calls) == 1 and ack["articulation_writes_this_call"] == 1
    assert adapter.write_count == 1 and audit["verified"]
    assert all(audit["changed_channels_full12"][8:])
    assert projection.safe_projected_residual_full12[8:] == pytest.approx((-.3,)*4, abs=1e-6)
    assert audit["actual_native_targets"]["wheel_velocity_rad_s"] == pytest.approx((0.,)*4, abs=1e-6)


@pytest.mark.parametrize("bad", ["unknown", "no_retirement", "not_continuous"])
def test_loader_rejects_mode_without_existing_retirement_contract(tmp_path, bad):
    spec = _spec()
    if bad == "unknown":
        spec["nominal"]["p06_wheel_tail_semantics"] = "always_push"
    elif bad == "no_retirement":
        spec["nominal"].pop("p06_rolling_retirement")
    else:
        spec["nominal"]["continuous_channel_inheritance"] = False
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_task_spec(path)


@pytest.mark.parametrize("bad", ["amplitude", "negative", "extra", "anchor", "endpoint",
    "servo", "onset", "duration", "atomic", "kind"])
def test_constructor_rejects_unreviewed_source_structures_but_absent_mode_stays_compatible(contract, bad):
    phase = contract.phase("P06")
    rows = list(phase.waypoints)
    if bad in ("amplitude", "negative"):
        rows[1] = replace(rows[1], full12=rows[1].full12[:8]+((.4 if bad == "amplitude" else -.3),)*4)
    elif bad == "extra":
        rows.insert(2, rows[1])
    elif bad == "anchor":
        rows[0] = replace(rows[0], full12=rows[0].full12[:8]+(.1,)*4)
    elif bad == "endpoint":
        rows[2] = replace(rows[2], full12=rows[2].full12[:8]+(.1,)*4)
    elif bad == "servo":
        rows[1] = replace(rows[1], full12=(rows[1].full12[0]+1.,)+rows[1].full12[1:])
    elif bad == "onset":
        rows[1] = replace(rows[1], time_s=.1)
    elif bad == "duration":
        phase = replace(phase, active_duration_s=phase.active_duration_s+1.)
    elif bad == "atomic":
        rows[1] = replace(rows[1], atomic_channels=(SERVO_ORDER[0],)+WHEEL_ORDER)
    else:
        rows[0] = replace(rows[0], kind="reference_waypoint")
    phase = replace(phase, waypoints=tuple(rows))
    modified = replace(contract, phases=tuple(phase if p.state_id == "P06" else p for p in contract.phases))
    with pytest.raises(ValueError, match="P06 wheel tail"):
        _provider(modified)
    assert "p06_wheel_tail" not in _provider(modified, False).nominal_suggestion_diagnostics


def test_source_amplitude_is_bound_to_verified_p01_tuple_not_hardcoded_constant(contract):
    rolling = (.2,)*4
    phases = []
    for phase in contract.phases:
        if phase.state_id in ("P01", "P06"):
            rows = tuple(replace(row, full12=row.full12[:8]+rolling)
                if any(row.full12[8:]) else row for row in phase.waypoints)
            phase = replace(phase, waypoints=rows)
        phases.append(phase)
    spec = _spec()
    spec["nominal"]["approach_wheel_prior_rad_s"] = rolling
    p = NominalMotionProvider(replace(contract, phases=tuple(phases)), spec=spec)
    assert _tail(p)["source_rolling_rad_s"] == rolling

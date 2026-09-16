"""Directed command-composition checks, not a substitute for contact physics."""
from copy import deepcopy
import math

import pytest

from test_actuator_target_effect import _adapter
from test_semantic_nominal_geometry_dispatch import saturated_history
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
from wlr50_clean.infrastructure.robot_adapter import RobotAdapterError
from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE
from wlr50_clean.ppo.semantic_nominal_geometry import (
    CONTEXT_SCHEMA, MODE, FUNCTIONAL_RR_MODE, BOUNDED_RR_MODE, correct_nominal_geometry,
)
from wlr50_clean.ppo.semantic_residual_adapter import apply_semantic_residual

ZERO = (0.,)*12


def vector(index, value):
    row = list(ZERO)
    row[index] = value
    return tuple(row)


def context(adapter, tick, mode=BOUNDED_RR_MODE, phase="P09", lag=False):
    indices = (6, 7) if phase == "P09" else (4, 5)
    return {"schema": CONTEXT_SCHEMA, "mode": mode, "source_phase_id": phase,
        "dispatch_physics_tick": tick, "canonical_servo_indices": indices,
        "physical_q_rad": tuple(math.radians(adapter.standing_pose_deg[SERVO_ORDER[i]]
                                             + (80. if lag else 0.)) for i in indices),
        "jacobian_x_m_per_rad": (0., 0.), "jacobian_z_m_per_rad": (0., 1.),
        "clearance_m": .015, "clearance_margin_m": .015,
        "place_xy": False, "ground_contact": False}


def dispatch(adapter, tick, residual=ZERO, nominal=ZERO, geometry=None, controller=ZERO):
    return apply_semantic_residual(adapter, nominal, physics_tick=tick,
        tracking_servo_names=(), controller_bias_full12=controller,
        projected_residual_full12=residual, nominal_geometry_context=geometry,
        policy_headroom_mode=HEADROOM_MODE)


@pytest.mark.parametrize("index,value", [(6, 2.56), (6, -2.56), (7, 1.786), (7, -1.786)])
def test_constant_rr_offset_is_not_reintegrated_at_each_physics_tick(index, value):
    adapter = _adapter()
    for tick in range(1, 121):
        previous = tuple(adapter._final_drive_servo_deg.values())
        ack = dispatch(adapter, tick, vector(index, value), geometry=context(adapter, tick, lag=True))
        assert ack["nominal_command_servo_deg"] == pytest.approx((0.,)*8, abs=1e-12)
        assert ack["geometry_adjusted_native_full12"] == pytest.approx(ZERO, abs=1e-12)
        assert abs(ack["drive_target_full12"][index]-previous[index]) <= 1.25+1e-12
        assert max(abs(v) for v in ack["drive_target_full12"][:8]) <= abs(value)+1e-12
    assert ack["drive_target_full12"][index] == pytest.approx(value)
    assert adapter.write_count == adapter.servo_target_mapper.feedback_tick == 120
    assert len(adapter.robot.events) == 360


@pytest.mark.parametrize("mode", [MODE, FUNCTIONAL_RR_MODE, BOUNDED_RR_MODE])
@pytest.mark.parametrize("phase", ["P09", "P12"])
def test_every_geometry_mode_uses_explicit_nominal_history_not_actual_final(mode, phase):
    adapter = _adapter()
    i = 6 if phase == "P09" else 4
    for tick in range(1, 17):
        ack = dispatch(adapter, tick, vector(i, 8.))
    assert adapter._final_drive_servo_deg[SERVO_ORDER[i]] == 8.
    ctx = context(adapter, 17, mode, phase)
    actual_before = deepcopy(adapter._final_drive_servo_deg)
    adjusted, evidence = correct_nominal_geometry(adapter=adapter, native_full12=ZERO,
        controller_bias_full12=ZERO, context=ctx, nominal_previous_servo_deg=(0.,)*8)
    assert adjusted == pytest.approx(ZERO, abs=1e-12)
    assert evidence["nominal_previous_servo_deg"] == [0.]*8
    assert adapter._final_drive_servo_deg == actual_before
    ack = dispatch(adapter, 17, vector(i, 8.), geometry=ctx)
    assert ack["nominal_command_servo_deg"] == pytest.approx([0.]*8, abs=1e-12)
    assert ack["drive_target_full12"][i] == pytest.approx(8.)


@pytest.mark.parametrize("value", [1000., -1000.])
def test_headroom_and_final_slew_remain_actual_while_nominal_stays_policy_free(value):
    adapter = _adapter()
    for tick in range(1, 151):
        old = tuple(adapter._final_drive_servo_deg.values())
        ack = dispatch(adapter, tick, vector(7, value), geometry=context(adapter, tick, lag=True))
        assert ack["nominal_command_servo_deg"] == pytest.approx([0.]*8, abs=1e-12)
        assert abs(ack["drive_target_full12"][7]-old[7]) <= 1.25+1e-12
        lo, hi = servo_limits_deg(SERVO_ORDER[7])
        assert lo <= ack["drive_target_full12"][7] <= hi
    assert ack["policy_headroom_evidence"]["effective_policy_residual_full12"][7] != value
    assert abs(ack["drive_target_full12"][7]) > 10.


def test_phase_context_exit_withdrawal_and_source_wheel_stop_do_not_reset_nominal_history():
    adapter = _adapter()
    wheel_source = ZERO[:8]+(.3,)*4
    for tick in range(1, 33):
        ack = dispatch(adapter, tick, vector(6, 20.), nominal=wheel_source)
    before = tuple(ack["drive_target_full12"][:8])
    assert before[6] == 20.
    for tick in range(33, 66):
        # P09 entry, P10 geometry exit, P12 entry, P13 stop are not resets.
        geom = context(adapter, tick, phase="P09" if tick < 40 else "P12", lag=True) if tick in (33, 50) else None
        nominal = wheel_source if tick < 60 else ZERO
        ack = dispatch(adapter, tick, nominal=nominal, geometry=geom)
        assert ack["nominal_history_initialization"] == "previous_semantic_nominal_command"
        assert ack["nominal_command_servo_deg"] == pytest.approx([0.]*8, abs=1e-12)
        assert abs(ack["drive_target_full12"][6]-before[6]) <= 1.25+1e-12
        assert ack["drive_target_full12"][8:] == list(nominal[8:])
        before = tuple(ack["drive_target_full12"][:8])
    assert before == pytest.approx((0.,)*8, abs=1e-12)


def test_nominal_only_fastpath_retains_controller_slew_and_zero_identity():
    semantic, original = _adapter(), _adapter()
    for tick in range(1, 49):
        nominal = (25., -15.)*4+(.3, -.2, .1, 0.) if tick < 30 else ZERO
        controller = (2., -2.)*4+(.02,)*4 if tick < 30 else ZERO
        actual = dispatch(semantic, tick, nominal=nominal, controller=controller)
        old = original.apply_full12(nominal, physics_tick=tick,
            drive_feedback_bias_full12=controller)
        assert actual["drive_target_full12"] == old["drive_target_full12"]
        assert actual["nominal_command_servo_deg"] == old["drive_target_full12"][:8]


def test_cancelled_nominal_history_is_not_replaced_by_actual_zero_in_geometry():
    adapter = saturated_history()
    assert adapter._final_drive_servo_deg[SERVO_ORDER[6]] == 0.
    assert adapter._semantic_nominal_command_history["servo_deg"][6] == 20.
    ack = dispatch(adapter, 21, nominal=vector(6, 20.), geometry=context(adapter, 21, lag=True))
    assert ack["nominal_geometry_evidence"]["nominal_previous_servo_deg"][6] == 20.
    assert ack["nominal_command_servo_deg"][6] == pytest.approx(20.)
    assert ack["drive_target_full12"][6] == 1.25


@pytest.mark.parametrize("tamper", ["missing", "nan", "stale"])
def test_trace_loss_or_corruption_cannot_bootstrap_from_a_policy_history(tamper):
    adapter = _adapter()
    dispatch(adapter, 1, vector(6, 2.))
    if tamper == "missing":
        del adapter._semantic_nominal_command_history
    elif tamper == "nan":
        adapter._semantic_nominal_command_history["servo_deg"] = (float("nan"),)*8
    else:
        adapter._semantic_nominal_command_history["write_count"] = 0
    with pytest.raises(RobotAdapterError, match="nominal command history"):
        dispatch(adapter, 2)
    assert adapter.write_count == adapter.servo_target_mapper.feedback_tick == 1


def test_settled_original_zero_initializes_once_and_a_fresh_reset_has_no_policy_trace():
    adapter = _adapter()
    for tick in range(180):
        adapter.apply_full12(ZERO, physics_tick=tick)
    ack = dispatch(adapter, 180, vector(6, 8.))
    assert ack["nominal_history_initialization"] == "settled_original_zero_ack"
    reset = _adapter()
    ack = dispatch(reset, 1)
    assert ack["nominal_history_initialization"] == "fresh_adapter_zero"
    assert ack["nominal_command_servo_deg"] == [0.]*8


def test_foreign_dispatch_is_not_silently_absorbed_as_a_nominal_history_reset():
    adapter = _adapter()
    dispatch(adapter, 1, vector(6, 4.))
    adapter.apply_full12(ZERO, physics_tick=2)
    with pytest.raises(RobotAdapterError, match="not adjacent"):
        dispatch(adapter, 3)
    assert adapter.write_count == adapter.servo_target_mapper.feedback_tick == 2

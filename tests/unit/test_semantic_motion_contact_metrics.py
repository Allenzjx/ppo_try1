from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_semantic_observation_reward_env import _frame, _raw
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.ppo.semantic_metrics import (
    SemanticMetricsAccumulator, WHEEL_RADIUS_ESTIMATE_M,
)
from wlr50_clean.sensing.contact_classifier import SENSED_BODIES
from wlr50_clean.sensing.observation import Observation, PairContactObservation

PROJECTION = SimpleNamespace(safe_projected_residual_full12=(0.,)*12)


def frame(tick, *, phase="P01", substage="TRANSFER", speed=0., com_speed=0., contact=True,
          vertical=0., force=25., wheel_speed=0., obstacle=False):
    raw = _raw(tick,joint=0.)
    base = replace(raw.base,linear_velocity_w_m_s=(speed,0.,0.))
    com = replace(raw.center_of_mass,velocity_w_m_s=(com_speed,0.,0.),total_mass_kg=10.)
    bodies,contacts,wheels = dict(raw.bodies),dict(raw.contacts),dict(raw.wheels)
    for name in WHEEL_ORDER:
        body = wheels[name].body_name
        wheels[name] = replace(wheels[name],velocity_rad_s=wheel_speed)
        bodies[body] = replace(bodies[body],linear_velocity_w_m_s=(speed,0.,vertical))
        old = contacts[body]
        def pair(template, active):
            return replace(template,active=active,normal_force_n=force if active else 0.,
                force_w_n=(0.,0.,force if active else 0.),active_history=(True,False,True))
        contacts[body] = replace(old,ground=pair(old.ground,contact and not obstacle),
                                obstacle=pair(old.obstacle,contact and obstacle))
    raw = replace(raw,base=base,center_of_mass=com,bodies=bodies,contacts=contacts,wheels=wheels)
    result = _frame(tick,stage=phase,raw=raw)
    result.info["semantic_task"]["substage"] = substage
    result.info["atomic_ack"] = {"drive_target_full12":(0.,)*12}
    return result


def diagnostics(metrics, *, phase=None, substage=None):
    report = metrics.summary()
    if substage:
        return report["transfer_capture_split"][phase][substage]["motion_contact_diagnostics"]
    return report["global"]["motion_contact_diagnostics"] if phase is None else report["phases"][phase]["motion_contact_diagnostics"]


def test_actual_dataclasses_linear_and_com_momentum_trends_use_physical_dt():
    before = frame(0,speed=.1,com_speed=.2)
    middle = frame(1,speed=.2,com_speed=.3)
    after = frame(2,speed=.4,com_speed=.5)
    assert isinstance(before.info["raw_observation"],Observation)
    assert isinstance(before.info["raw_observation"].contacts[next(iter(before.info["raw_observation"].contacts))].ground,PairContactObservation)
    metrics = SemanticMetricsAccumulator()
    metrics.observe(before,middle,PROJECTION)
    metrics.observe(middle,after,PROJECTION)
    data = diagnostics(metrics)
    assert data["body_linear_velocity_world_x_m_s"]["mean"] == pytest.approx(.3)
    assert data["com_velocity_world_x_m_s"]["mean"] == pytest.approx(.4)
    assert data["com_linear_momentum_world_x_kg_m_s"]["mean"] == pytest.approx(4.)
    assert data["com_linear_momentum_change_world_x_kg_m_s"]["sum"] == pytest.approx(3.)
    assert data["body_linear_velocity_change_world_x_m_s"]["sum"] == pytest.approx(.3)
    assert data["FL_contact_normal_force_n"]["integral"] == pytest.approx(50/120)
    assert data["FL_contact_force_world_z_n"]["integral"] == pytest.approx(50/120)
    assert data["wheel_contact_count"]["mean"] == data["wheel_exact_active_pair_count"]["mean"] == 4


def test_transfer_capture_phase_splits_and_old_fixed_score_are_unchanged():
    metrics = SemanticMetricsAccumulator()
    a,b,c = frame(0,speed=.1),frame(1,speed=.2,substage="CAPTURE"),frame(2,speed=.4,phase="P02")
    metrics.observe(a,b,PROJECTION)
    metrics.observe(b,c,PROJECTION)
    assert diagnostics(metrics,phase="P01",substage="TRANSFER")["body_linear_speed_m_s"]["mean"] == pytest.approx(.2)
    assert diagnostics(metrics,phase="P01",substage="CAPTURE")["body_linear_speed_m_s"]["mean"] == pytest.approx(.4)
    report = metrics.summary()
    assert report["phases"]["P01"]["quality_score"] == 0.
    assert report["fixed_quality_score"] is None
    assert report["phases"]["P02"]["motion_contact_diagnostics"] == {}


def test_touchdown_descent_rebound_and_exact_chatter_count_edges_once():
    metrics = SemanticMetricsAccumulator()
    a,b,c = frame(0,contact=False,vertical=-.4),frame(1,contact=True,force=60.),frame(2,contact=False,vertical=.3)
    metrics.observe(a,b,PROJECTION)
    metrics.observe(b,c,PROJECTION)
    data = diagnostics(metrics)
    assert data["FL_touchdown_event_count"]["sum"] == 1
    assert data["FL_touchdown_descent_speed_m_s"]["mean"] == pytest.approx(.4)
    assert data["FL_touchdown_normal_force_n"]["peak_abs"] == 60.
    assert data["FL_post_touchdown_rebound_event_count"]["sum"] == 1
    assert data["FL_post_touchdown_rebound_upward_speed_m_s"]["mean"] == pytest.approx(.3)
    # active_history is deliberately alternating in every fixture; it must not
    # inflate true chronological edge counts through overlapping windows.
    assert data["FL_exact_pair_toggle_event_count"]["sum"] == 2
    assert data["FL_wheel_contact_toggle_event_count"]["sum"] == 2


def test_ground_to_obstacle_pair_transfer_is_not_a_new_wheel_touchdown():
    metrics = SemanticMetricsAccumulator()
    metrics.observe(frame(0),frame(1,obstacle=True),PROJECTION)
    data = diagnostics(metrics)
    assert data["FL_exact_pair_toggle_event_count"]["sum"] == 2
    assert data["FL_wheel_contact_toggle_event_count"]["sum"] == 0
    assert data["FL_touchdown_event_count"]["sum"] == 0
    assert data["FL_touchdown_descent_speed_m_s"]["available"] is False


def test_slip_is_conditional_labeled_estimate_and_normal_rolling_can_be_zero():
    metrics = SemanticMetricsAccumulator()
    speed = 2*WHEEL_RADIUS_ESTIMATE_M
    metrics.observe(frame(0,speed=speed,wheel_speed=2),frame(1,speed=speed,wheel_speed=2),PROJECTION)
    data = diagnostics(metrics)
    assert data["FL_contact_rolling_slip_estimate_abs_m_s"]["mean"] == pytest.approx(0.,abs=1e-15)
    assert "Not ground-truth" in metrics.summary()["diagnostic_notes"]["slip"]
    metrics.reset()
    metrics.observe(frame(0,contact=False,wheel_speed=20),frame(1,contact=False,wheel_speed=20),PROJECTION)
    data = diagnostics(metrics)
    assert data["FL_contact_rolling_slip_estimate_abs_m_s"]["available"] is False
    assert data["FL_exact_active_pair_count"]["mean"] == 0.
    assert data["FL_contact_normal_force_n"]["integral"] == 0.


def test_missing_or_unverified_measurements_are_null_not_false_zero():
    before,after = frame(0),frame(1)
    raw = after.info["raw_observation"]
    contacts = dict(raw.contacts)
    body = raw.wheels[WHEEL_ORDER[0]].body_name
    contacts[body] = replace(contacts[body],ground=replace(contacts[body].ground,pair_verified=False))
    after.info["raw_observation"] = replace(raw,contacts=contacts,center_of_mass=replace(raw.center_of_mass,valid=False))
    metrics = SemanticMetricsAccumulator()
    metrics.observe(before,after,PROJECTION)
    data = diagnostics(metrics)
    assert data["FL_contact_normal_force_n"]["mean"] is None
    assert data["FL_touchdown_event_count"]["sum"] is None
    assert data["com_linear_momentum_world_x_kg_m_s"]["available"] is False
    assert data["FR_contact_normal_force_n"]["mean"] == 25.
    # This fixture supplies only four wheels, not the complete thirteen bodies.
    assert data["all_body_exact_active_pair_count"]["mean"] is None
    json.dumps(metrics.summary(),allow_nan=False)


def test_full_verified_body_pair_bank_count_and_support_impulse_not_badness():
    before,after = frame(0),frame(1)
    raw = after.info["raw_observation"]
    contacts = dict(raw.contacts)
    template = next(iter(contacts.values()))
    for body in SENSED_BODIES:
        if body not in contacts:
            contacts[body] = replace(template,body_name=body,
                ground=replace(template.ground,sensor_body=body,active=False,normal_force_n=0.,force_w_n=(0.,0.,0.)),
                obstacle=replace(template.obstacle,sensor_body=body,active=False,normal_force_n=0.,force_w_n=(0.,0.,0.)))
    after.info["raw_observation"] = replace(raw,contacts=contacts)
    metrics = SemanticMetricsAccumulator()
    metrics.observe(before,after,PROJECTION)
    data = diagnostics(metrics)
    assert data["all_body_exact_active_pair_count"]["mean"] == 4
    assert data["all_body_exact_normal_force_n"]["integral"] == pytest.approx(100/120)
    assert "not a universal badness cost" in metrics.summary()["diagnostic_notes"]["force_impulse"]
    assert metrics.summary()["phases"]["P01"]["quality_score"] == 0.


def test_reset_clears_rebound_history_and_nonfinite_terminal_skip_is_explicit():
    metrics = SemanticMetricsAccumulator()
    metrics.observe(frame(0,contact=False),frame(1),PROJECTION)
    metrics.reset()
    metrics.observe(frame(0),frame(1,contact=False,vertical=.3),PROJECTION)
    assert diagnostics(metrics)["FL_post_touchdown_rebound_event_count"]["sum"] == 0
    bad = frame(2)
    bad.info["raw_observation"] = replace(bad.info["raw_observation"],all_finite=False)
    metrics.observe(frame(1,contact=False),bad,PROJECTION)
    report = metrics.summary()
    assert report["nonfinite_terminal_ticks_skipped"] == 1
    assert report["global"]["physics_ticks"] == 1

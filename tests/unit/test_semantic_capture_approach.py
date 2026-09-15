"""Post-cross soft capture progress; no new placement or controller gate.

Sensor sequences below exercise the actual task evaluator, not Isaac. Isolated
snapshot counterfactuals test the exact potential dependency, not a physical
trajectory prediction or an alternative source of hard event history.
"""
from copy import deepcopy
import math
from pathlib import Path

import pytest
import yaml

from test_semantic_supervisor import advance, leg_state, observation
from test_semantic_observation_reward_env import _built, _frame, _sample
from wlr50_clean.ppo.semantic_supervisor import (
    CAPTURE_APPROACH_MODE, CAPTURE_RETENTION_MODE, DEFAULT_TASK_SPEC_PATH,
    LEG_ORDER, PHASE_IDS, PLACEMENT_PREDECESSORS, SemanticObservationError,
    TaskEvaluator, TaskStageSupervisor, load_task_spec,
)
from wlr50_clean.ppo.semantic_reward import (
    FAMILIES, SemanticRewardCalculator, load_semantic_reward_config,
)


ROOT = Path(__file__).resolve().parents[2]
CFG = ROOT / "configs/ppo_semantic_v3"
SPEC = CFG / "stage_task_spec.yaml"
MODE_KEY = "capture_approach_semantics"
WEIGHT = .85 / 4 * .2


def _spec(enabled=True):
    spec = load_task_spec(SPEC)
    # Isolate the earlier capture-factor revision from the later replacement
    # of its old low-load preparation term (covered by transfer-role tests).
    spec.pop('transfer_roles', None)
    if enabled:
        spec[MODE_KEY] = CAPTURE_APPROACH_MODE
    else:
        spec.pop(MODE_KEY, None)
    return spec


def _step(ev, obs, leg, **changes):
    obs = advance(obs)
    leg_state(obs, leg, **changes)
    snapshot = ev.observe(obs)
    assert snapshot["valid"] and snapshot["termination_reason"] is None
    return obs, snapshot


def _qualified(leg="RR", *, crossed=True, spec=None):
    """Earn real Q, then C, with predecessor placement in the real evaluator."""
    ev = TaskEvaluator(spec=deepcopy(spec) if spec is not None else _spec())
    obs = observation()
    for joint in obs["joints"].values():
        joint["command_deg"] = 0.
    ev.observe(obs)
    frames = [deepcopy(obs)]
    for prior in PLACEMENT_PREDECESSORS[leg]:
        for data in ({"bottom": .012, "air": True, "hip": 1.5},
                     {"bottom": .020, "air": True, "hip": 3.},
                     {"x": .53, "bottom": .075, "air": True, "hip": 4.},
                     {"x": .53, "bottom": .05, "top": True},
                     {"x": .53, "bottom": .05, "top": True}):
            obs, _ = _step(ev, obs, prior, **data)
            frames.append(deepcopy(obs))
        assert ev.snapshot["history"]["placed"][prior]
    for height, hip in ((.012, 1.5), (.020, 3.), (.075, 4.)):
        obs, _ = _step(ev, obs, leg, x=.4, bottom=height, hip=hip, air=True)
        frames.append(deepcopy(obs))
    assert ev.snapshot["history"]["active_lift"][leg]
    assert not ev.snapshot["history"]["front_edge_crossed"][leg]
    if crossed:
        obs, _ = _step(ev, obs, leg, x=.53, bottom=.075, air=True)
        frames.append(deepcopy(obs))
        ticks = ev.snapshot["history"]["event_ticks"]
        assert ticks["active_lift"][leg] < ticks["front_edge_crossed"][leg]
    assert not ev.snapshot["history"]["placed"][leg]
    return ev, obs, frames


def _supervisors(ev):
    return (_supervisor(_spec(), evaluator=ev),
            _supervisor(_spec(False), evaluator=ev))


def _supervisor(spec, *, evaluator=None, initial_stage_id="P01"):
    # Production constructor takes a path, not a spec kwarg. The variants are
    # copies of the loaded valid v3 spec differing only in this optional mode.
    sup = TaskStageSupervisor(SPEC, evaluator=evaluator, initial_stage_id=initial_stage_id)
    sup.spec = deepcopy(spec)
    if evaluator is None:
        sup.evaluator = TaskEvaluator(spec=spec)
    return sup


def _approach(outside, clearance, spec):
    tolerance = spec["geometry"]["top_gap_max_m"]
    return max(0., min(1., 1. - outside / .25)) * tolerance / (tolerance + abs(clearance))


def _counterfactual(snapshot, leg="RR", **fields):
    snap = deepcopy(snapshot)
    snap["current_legs"][leg].update(fields)
    return snap


@pytest.mark.parametrize("leg", LEG_ORDER)
def test_real_qualified_crossed_air_gap_sweep_uses_exact_existing_half_capture_share(leg):
    ev, obs, _ = _qualified(leg)
    new, old = _supervisors(ev)
    history = deepcopy(ev.snapshot["history"])
    phis = []
    for gap in (.025, .015, .005, .001, .0004):
        obs, snap = _step(ev, obs, leg, x=.53, bottom=.05+gap, air=True)
        current = snap["current_legs"][leg]
        assert current["air"] and current["consecutive_top_samples"] == 0
        assert current["load_fraction"] == 0. and current["top_xy_outside_distance_m"] == 0.
        assert snap["history"] == history and not snap["success"]
        expected = .5 * _approach(0., gap, new.spec)
        phis.append(new.physical_potential(snap))
        assert phis[-1] - old.physical_potential(snap) == pytest.approx(WEIGHT * expected)
        assert new.predicate(f"placed_{leg}", snap) == old.predicate(f"placed_{leg}", snap) < 1.
    assert all(a < b for a, b in zip(phis, phis[1:]))


@pytest.mark.parametrize("gap", [.0004, .001, .005, .015, .025, .1])
def test_signed_gap_symmetry_and_no_downward_reward_beyond_surface(gap):
    ev, _, _ = _qualified()
    new, old = _supervisors(ev)
    positive = _counterfactual(ev.snapshot, clearance_m=gap)
    negative = _counterfactual(ev.snapshot, clearance_m=-gap)
    assert new.physical_potential(positive) == new.physical_potential(negative)
    assert new.physical_potential(negative)-old.physical_potential(negative) == pytest.approx(
        WEIGHT*.5*.025/(.025+gap))
    surface = _counterfactual(ev.snapshot, clearance_m=0.)
    assert new.physical_potential(surface) > new.physical_potential(negative)


@pytest.mark.parametrize("outside", [0., 1e-9, .005, .125, .249999999, .25, .5])
def test_current_xy_outside_continuously_reduces_only_post_cross_approach(outside):
    ev, _, _ = _qualified()
    new, old = _supervisors(ev)
    snap = _counterfactual(ev.snapshot, top_xy_outside_distance_m=outside, clearance_m=.005)
    before = deepcopy(snap)
    assert new.physical_potential(snap)-old.physical_potential(snap) == pytest.approx(
        WEIGHT*.5*_approach(outside, .005, new.spec))
    assert snap == before


def test_real_back_edge_geometry_not_front_distance_alone_controls_approach():
    ev, obs, _ = _qualified()
    new, old = _supervisors(ev)
    # Already crossed, still AIR. The real measured XY distance detects a
    # point outside the platform's rear edge despite its positive front gap.
    obs, snap = _step(ev, obs, "RR", x=2.02, bottom=.055, air=True)
    outside = .02-new.spec["geometry"]["xy_measurement_tolerance_m"]
    assert snap["current_legs"]["RR"]["top_xy_outside_distance_m"] == pytest.approx(outside)
    assert not snap["current_legs"]["RR"]["within_top_xy"]
    assert new.physical_potential(snap)-old.physical_potential(snap) == pytest.approx(
        WEIGHT*.5*_approach(outside, .005, new.spec))
    assert not snap["history"]["placed"]["RR"]


def test_real_contact_samples_supply_separate_half_and_only_original_second_sample_places():
    ev, obs, _ = _qualified()
    new, old = _supervisors(ev)
    obs, air = _step(ev, obs, "RR", x=.53, bottom=.05, air=True)
    assert new.physical_potential(air)-old.physical_potential(air) == pytest.approx(WEIGHT*.5)
    assert not air["history"]["placed"]["RR"]
    obs, one = _step(ev, obs, "RR", x=.53, bottom=.05, top=True)
    assert one["current_legs"]["RR"]["consecutive_top_samples"] == 1
    assert not one["history"]["placed"]["RR"]
    # New capture=.5+.5*(1/2)=.75; old capture=1/2. The same revision
    # preserves completed post-cross unloading instead of taxing first contact.
    earned_unload = .85/4*.1*(1.-old.predicate("load_ready_RR", one))
    assert new.physical_potential(one)-old.physical_potential(one) == pytest.approx(WEIGHT*.25+earned_unload)
    obs, placed = _step(ev, obs, "RR", x=.53, bottom=.05, top=True)
    assert placed["history"]["placed"]["RR"]
    assert new.predicate("placed_RR", placed) == old.predicate("placed_RR", placed) == 1.
    assert new.physical_potential(placed) == old.physical_potential(placed)


def test_first_real_loaded_top_contact_cannot_lose_previously_completed_unload_progress():
    ev, obs, _ = _qualified()
    sup, _ = _supervisors(ev)
    obs, air = _step(ev, obs, "RR", x=.53, bottom=.05, air=True)
    assert sup.predicate("load_ready_RR", air) == 1.
    next_obs = advance(obs)
    leg_state(next_obs, "RR", x=.53, bottom=.05, top=True)
    # Actual evaluator load fraction, not injected current_legs: the other
    # three verified supports remain, while this first TOP contact bears load.
    contact = next_obs["contacts"]["rear_right_wheel"]["obstacle"]
    contact["normal_force_n"] = 2100.
    top = ev.observe(next_obs)
    assert top["valid"] and top["termination_reason"] is None
    assert top["history"]["active_lift"]["RR"] and top["history"]["front_edge_crossed"]["RR"]
    assert not top["history"]["placed"]["RR"]
    assert top["current_legs"]["RR"]["consecutive_top_samples"] == 1
    assert sup.predicate("load_ready_RR", top) < .02
    assert sup.physical_potential(top) > sup.physical_potential(air)


@pytest.mark.parametrize("qualified,crossed", [(False, False), (True, False), (False, True), (True, True)])
def test_unload_is_completed_only_after_both_hard_events_without_altering_load_predicate(qualified, crossed):
    ev, _, _ = _qualified()
    new, old = _supervisors(ev)
    base = deepcopy(ev.snapshot)
    base["history"]["active_lift"]["RR"] = qualified
    base["history"]["front_edge_crossed"]["RR"] = crossed
    base["current_legs"]["RR"].update(top_xy_outside_distance_m=.25,
        consecutive_top_samples=0, load_fraction=0.)
    loaded = _counterfactual(base, load_fraction=1.)
    before = deepcopy(loaded)
    assert new.predicate("load_ready_RR", loaded) == old.predicate("load_ready_RR", loaded) == 0.
    assert new.predicate("load_ready_RR", base) == old.predicate("load_ready_RR", base) == 1.
    if qualified and crossed:
        assert new.physical_potential(loaded) == new.physical_potential(base)
        assert new.physical_potential(loaded)-old.physical_potential(loaded) == pytest.approx(.85/4*.1)
    else:
        assert new.physical_potential(loaded)-new.physical_potential(base) == pytest.approx(-.85/4*.1)
    assert loaded == before


@pytest.mark.parametrize("leg", LEG_ORDER)
def test_placed_air_uses_original_current_retention_not_preplacement_surface_penalty(leg):
    ev, obs, _ = _qualified(leg)
    for _ in range(2):
        obs, _ = _step(ev, obs, leg, x=.53, bottom=.05, top=True)
    new, old = _supervisors(ev)
    obs, high = _step(ev, obs, leg, x=.53, bottom=.55, air=True)
    assert high["current_legs"][leg]["load_fraction"] == 0.
    assert high["history"]["placed"][leg]
    # The real contact loss also redistributes other legs' load fractions.
    # Compare the identical current state, not two different total-body states.
    assert new._current_capture_retention(leg, high) == old._current_capture_retention(leg, high) == 1.
    reference = new.physical_potential(high)
    assert reference == old.physical_potential(high)
    for outside, gap in ((.1, .1), (0., -.05)):
        snap = _counterfactual(high, leg, top_xy_outside_distance_m=outside, clearance_m=gap)
        assert new.physical_potential(snap) == old.physical_potential(snap) < reference
        assert new.predicate(f"placed_{leg}", snap) == 1.


def test_no_cross_has_no_capture_proximity_incentive_to_descend_early():
    ev, obs, _ = _qualified(crossed=False)
    new, old = _supervisors(ev)
    for gap in (.025, .015, .005, .001, .0004):
        obs, snap = _step(ev, obs, "RR", x=.4, bottom=.05+gap, air=True)
        assert snap["history"]["active_lift"]["RR"]
        assert not snap["history"]["front_edge_crossed"]["RR"]
        assert new.physical_potential(snap) == old.physical_potential(snap)
        assert not snap["history"]["placed"]["RR"]


@pytest.mark.parametrize("qualified,crossed", [(False, False), (True, False), (False, True)])
def test_missing_either_required_history_bit_cannot_earn_capture(qualified, crossed):
    ev, _, _ = _qualified()
    new, _ = _supervisors(ev)
    snap = deepcopy(ev.snapshot)
    snap["history"]["active_lift"]["RR"] = qualified
    snap["history"]["front_edge_crossed"]["RR"] = crossed
    # Deliberately inconsistent direct snapshots test the dependency, not an
    # assertion that real TaskEvaluator can generate cross without lift.
    off = _counterfactual(snap, clearance_m=.025, top_xy_outside_distance_m=.25,
                         consecutive_top_samples=0)
    on = _counterfactual(snap, clearance_m=.025, top_xy_outside_distance_m=0.,
                        consecutive_top_samples=2)
    assert new.physical_potential(on) == new.physical_potential(off)


def test_unplaced_rl_stays_predecessor_gated_even_with_inconsistent_perfect_capture_snapshot():
    ev, _, _ = _qualified("RR")
    new, old = _supervisors(ev)
    snap = deepcopy(ev.snapshot)
    snap["history"]["active_lift"]["RL"] = snap["history"]["front_edge_crossed"]["RL"] = True
    snap["current_legs"]["RL"].update(clearance_m=0., top_xy_outside_distance_m=0., consecutive_top_samples=100)
    assert not snap["history"]["placed"]["RR"]
    # Remove the independent RR approach contribution to isolate RL.
    snap["current_legs"]["RR"]["top_xy_outside_distance_m"] = .25
    assert new.physical_potential(snap) == old.physical_potential(snap)


def test_real_evaluator_history_stage_predicates_and_done_are_identical_to_optout():
    ev, obs, frames = _qualified("FL")
    for gap, top in ((.015, False), (.0004, False), (0., True), (0., True)):
        obs, _ = _step(ev, obs, "FL", x=.53, bottom=.05+gap, air=not top, top=top)
        frames.append(deepcopy(obs))
    new = _supervisor(_spec())
    old = _supervisor(_spec(False))
    for raw in frames:
        current, reference = new.observe_and_update(raw), old.observe_and_update(raw)
        for key in ("stage_id", "completed_stage_ids", "success", "termination_reason",
                    "entry_valid", "completion_values", "history", "transition_evidence", "phase_progress"):
            assert current[key] == reference[key]
        assert current["physical_evaluator"] == reference["physical_evaluator"]
    assert new.evaluator.snapshot["history"]["placed"]["FL"]


def _reward(a, b, *, terminal=None):
    before, after = _built(_frame(phi=a)), _built(_frame(1, phi=b))
    return SemanticRewardCalculator(load_semantic_reward_config(CFG/"reward_config.yaml")).evaluate(
        before, after, [_sample(before, after)], termination_reason=terminal, task_success=False)


def test_real_reward_single_pbrs_term_hover_and_closed_surface_cycle_cannot_farm_reward():
    ev, obs, _ = _qualified()
    sup, _ = _supervisors(ev)
    phis = []
    for gap in (.025, .0004, .025):
        obs, snap = _step(ev, obs, "RR", x=.53, bottom=.05+gap, air=True)
        phis.append(sup.physical_potential(snap))
    a, b, c = phis
    assert b > a and c == a
    static, approach = _reward(a, a), _reward(a, b)
    gamma = load_semantic_reward_config(ROOT/'configs/ppo_semantic_v3/reward_config.yaml').gamma
    assert static["potential_shaping"] == pytest.approx(5*(gamma*a-a))
    assert static["potential_shaping"] < 0.
    assert approach["potential_shaping"] == pytest.approx(5*(gamma*b-a))
    assert approach["total"]-static["total"] == pytest.approx(5*gamma*(b-a))
    assert tuple(approach["families"]) == FAMILIES
    for family in FAMILIES[1:]:
        assert approach["families"][family] == static["families"][family]
    assert approach["potential_shaping"]+gamma*_reward(b, c)["potential_shaping"] <= 0.
    failure = _reward(b, b, terminal="BODY_COLLISION")
    assert failure["potential_after"] == 0.
    assert failure["potential_shaping"] == pytest.approx(-5*b)


def test_hover_does_not_accumulate_contact_or_events_and_phi_is_phase_label_independent():
    ev, obs, _ = _qualified()
    sup, _ = _supervisors(ev)
    obs, snap = _step(ev, obs, "RR", x=.53, bottom=.0504, air=True)
    before, phi = deepcopy(snap["history"]), sup.physical_potential(snap)
    for _ in range(32):
        obs, snap = _step(ev, obs, "RR", x=.53, bottom=.0504, air=True)
        assert snap["history"] == before and not snap["history"]["placed"]["RR"]
        assert snap["current_legs"]["RR"]["consecutive_top_samples"] == 0
        assert sup.physical_potential(snap) == phi
    for phase in PHASE_IDS:
        assert _supervisor(_spec(), evaluator=ev, initial_stage_id=phase).physical_potential(snap) == phi
    assert 0. <= phi <= 1.


@pytest.mark.parametrize("mode", ["absent", None])
def test_absent_or_none_mode_keeps_old_count_based_capture_and_needs_no_new_xy_value(mode):
    spec = _spec(False)
    if mode is None:
        spec[MODE_KEY] = None
    ev, _, _ = _qualified(spec=spec)
    old = _supervisor(spec, evaluator=ev)
    snap = deepcopy(ev.snapshot)
    snap["current_legs"]["RR"].pop("top_xy_outside_distance_m")
    baseline = old.physical_potential(snap)
    snap["current_legs"]["RR"]["consecutive_top_samples"] = 1
    assert old.physical_potential(snap)-baseline == pytest.approx(WEIGHT*.5)
    snap["current_legs"]["RR"]["consecutive_top_samples"] = 100
    assert old.physical_potential(snap)-baseline == pytest.approx(WEIGHT)
    assert MODE_KEY not in load_task_spec(DEFAULT_TASK_SPEC_PATH)


@pytest.mark.parametrize("field,bad", [("top_xy_outside_distance_m", x) for x in
        (-.01, math.nan, math.inf, -math.inf, None, True)] + [("clearance_m", math.nan)])
def test_active_new_branch_rejects_malformed_current_geometry(field, bad):
    ev, _, _ = _qualified()
    new, _ = _supervisors(ev)
    with pytest.raises((SemanticObservationError, ValueError)):
        new.physical_potential(_counterfactual(ev.snapshot, **{field: bad}))


@pytest.mark.parametrize("bad", ["unknown", "no_retention", "wrong_retention", "no_global", "zero_gap", "negative_gap", "nan_gap"])
def test_loader_validates_optin_and_existing_dependency_scales(tmp_path, bad):
    spec = _spec()
    if bad == "unknown": spec[MODE_KEY] = "unreviewed"
    if bad == "no_retention": spec.pop("capture_retention_semantics")
    if bad == "wrong_retention": spec["capture_retention_semantics"] = "wrong"
    if bad == "no_global": spec["potential_definition"] = "phase_local"
    if bad == "zero_gap": spec["geometry"]["top_gap_max_m"] = 0.
    if bad == "negative_gap": spec["geometry"]["top_gap_max_m"] = -.025
    if bad == "nan_gap": spec["geometry"]["top_gap_max_m"] = math.nan
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError): load_task_spec(path)


def test_actual_actor_schema_stays_372_and_changes_only_existing_phi_scalar():
    from wlr50_clean.ppo.semantic_observation import HISTORY_GROUPS, SemanticObservationBuilder, load_semantic_observation_schema
    ev, _, _ = _qualified()
    new, old = _supervisors(ev)
    new_phi, old_phi = new.physical_potential(ev.snapshot), old.physical_potential(ev.snapshot)
    schema = load_semantic_observation_schema(CFG/"observation_schema.json")
    history = dict.fromkeys(HISTORY_GROUPS, (0.,)*12)
    a = SemanticObservationBuilder(schema).build(_frame(phi=old_phi), history)
    b = SemanticObservationBuilder(schema).build(_frame(phi=new_phi), history)
    assert schema.dimension == 372
    assert len(schema.encode(a.groups)) == len(schema.encode(b.groups)) == 372
    changed = [key for key in a.groups if a.groups[key] != b.groups[key]]
    assert changed == ["task_progress"]
    assert a.groups["task_progress"][0] == b.groups["task_progress"][0]
    assert b.groups["task_progress"][1]-a.groups["task_progress"][1] == pytest.approx(new_phi-old_phi)

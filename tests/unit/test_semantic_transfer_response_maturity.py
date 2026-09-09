"""Real legacy measurement/encoder counterexamples for opt-in role clocks.

Isolate the tracker from the separately changing all-stage contact evaluator:
its current-leg inputs come from the real legacy TaskEvaluator unless an
unavailable-load flag is explicitly injected. Synthetic contiguous sensor
observations, not simulator or all-stage detector acceptance evidence.
"""
from copy import deepcopy

import pytest

from test_semantic_supervisor import advance, leg_state, place
from test_semantic_transfer_roles import SPEC, measured, transfer
from wlr50_clean.infrastructure.command_batch import WHEEL_ORDER
from wlr50_clean.ppo.semantic_observation import transfer_role_observation_features
from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator, load_task_spec
from wlr50_clean.ppo.semantic_transfer_roles import (
    ALL_STAGE_ACCEPTANCE_VERSION, LEGS, ROLE_OBSERVATION_DIM,
    ROLE_OBSERVATION_FIELDS, TransferRoleTracker,
)


def evaluator(*, enabled=True, explicit_none=False):
    spec = deepcopy(load_task_spec(SPEC))
    # Pin both cases explicitly even when the production default opts in later.
    spec.pop("physical_acceptance_version", None)
    if explicit_none:
        spec["physical_acceptance_version"] = None
    ev = TaskEvaluator(spec=spec)
    if enabled:
        tracker_spec = deepcopy(spec)
        tracker_spec["physical_acceptance_version"] = ALL_STAGE_ACCEPTANCE_VERSION
        ev._transfer_tracker = TransferRoleTracker(tracker_spec)
    raw = measured()
    ev.observe(raw)
    return ev, raw


def prepare_space(ev, raw, leg):
    raw, role = transfer(ev, raw, leg, move_com=False, move_receiver=True, n=35)
    assert role["preparation_ready"]
    assert role["transfer_progress"] == 0.
    assert role["transfer_direction_context"]["com_toward_receiver_m"] == 0.
    return raw, role


def new_drop(ev, raw, leg):
    raw = advance(raw)
    wheel = raw["wheels"][WHEEL_ORDER[LEGS.index(leg)]]
    raw["contacts"][wheel["body_name"]]["ground"]["normal_force_n"] = 3.
    return raw, ev.observe(raw)["transfer_roles"][leg]


@pytest.mark.parametrize("leg", LEGS)
def test_mature_space_cannot_supply_age_to_first_new_load_drop(leg):
    ev, raw = evaluator()
    raw, prepared = prepare_space(ev, raw, leg)
    assert prepared["transfer_direction_context"]["continued_response_duration_s"] == 0.
    raw, role = new_drop(ev, raw, leg)
    context = role["transfer_direction_context"]
    assert context["load_fraction_change"] >= ev.spec["transfer_roles"]["load_change_scale"]
    assert context["preparation_response_duration_s"] > context["minimum_evidence_s"]
    assert context["continued_response_duration_s"] == 0.
    assert role["preparation_ready"] and role["preparation_progress"] == 1.
    assert not role["transfer_ready"] and role["transfer_progress"] == 0.
    assert ev.snapshot["current_legs"][leg]["ground_contact"]
    assert not ev.snapshot["current_legs"][leg]["air"]
    assert not ev.snapshot["history"]["active_lift"][leg]


@pytest.mark.parametrize("leg", LEGS)
def test_new_transfer_response_matures_after_its_own_short_continuation(leg):
    ev, raw = evaluator()
    raw, _ = prepare_space(ev, raw, leg)
    raw, _ = new_drop(ev, raw, leg)
    drop_tick = raw["physics_tick"]
    for elapsed_ticks in range(1, 9):
        raw = advance(raw)
        role = ev.observe(raw)["transfer_roles"][leg]
        assert role["transfer_direction_context"]["continued_response_duration_s"] == pytest.approx(elapsed_ticks/120.)
        assert role["transfer_ready"] is (elapsed_ticks == 8)
    assert raw["physics_tick"] == drop_tick+8
    assert role["transfer_progress"] == pytest.approx(1.)
    assert role["observed_support_contacts"] == list(LEGS)
    # Neither own-leg lift nor a compulsory AIR receiver has been introduced.
    assert not ev.snapshot["current_legs"][leg]["initial_clearance"]
    assert not role["receiver_workspace_state"]["receiver_air"]


def test_legacy_unmarked_and_explicit_none_keep_shared_age_and_old_dictionary():
    snapshots = []
    for explicit_none in (False, True):
        ev, raw = evaluator(enabled=False, explicit_none=explicit_none)
        raw, _ = prepare_space(ev, raw, "RR")
        raw, role = new_drop(ev, raw, "RR")
        assert role["transfer_ready"]  # Exact old counterexample retained only for legacy.
        assert "preparation_response_duration_s" not in role["transfer_direction_context"]
        assert "response_maturity_semantics" not in role["transfer_direction_context"]
        snapshots.append(ev.snapshot)
    assert snapshots[0] == snapshots[1]


def test_mature_real_directional_response_can_support_a_later_load_drop():
    ev, raw = evaluator()
    raw, prepared = transfer(ev, raw, "RR", move_com=True, move_receiver=False, n=35)
    assert prepared["transfer_direction_context"]["continued_response_duration_s"] > ev.spec["transfer_roles"]["minimum_evidence_s"]
    raw, role = new_drop(ev, raw, "RR")
    assert role["transfer_ready"]
    assert role["transfer_direction_context"]["com_toward_receiver_m"] > 0.


def test_same_legacy_physical_lift_cross_placement_is_unchanged_by_clock_optin():
    records = []
    for enabled in (False, True):
        ev, raw = evaluator(enabled=enabled)
        raw = place(ev, raw, "FR")
        raw = place(ev, raw, "FL")
        raw = place(ev, raw, "RR")
        snapshot = ev.snapshot
        assert all(snapshot["history"]["placed"][leg] for leg in ("FR", "FL", "RR"))
        records.append({key: snapshot[key] for key in
                        ("history", "current_legs", "termination_reason", "success")})
    assert records[0] == records[1]


def test_space_response_without_actuation_does_not_mature_either_clock():
    ev, raw = evaluator()
    receiver = raw["wheels"]["front_left_ankle"]
    for _ in range(35):
        raw = advance(raw)
        receiver = raw["wheels"]["front_left_ankle"]
        for key in ("center_w_m", "bottom_w_m"):
            receiver[key][0] -= .0002
        role = ev.observe(raw)["transfer_roles"]["RR"]
    assert role["receiver_workspace_state"]["radial_contraction_m"] > .001
    assert not role["transfer_direction_context"]["actuated_response"]
    assert not role["preparation_ready"] and not role["transfer_ready"]
    assert ev._transfer_tracker.response_since["RR"] is None
    assert ev._transfer_tracker.transfer_response_since["RR"] is None


def test_actual_support_interruption_revokes_both_clocks_without_erasing_history():
    ev, raw = evaluator()
    raw, _ = prepare_space(ev, raw, "RR")
    raw, _ = new_drop(ev, raw, "RR")
    before_history = deepcopy(ev.snapshot["history"])
    raw = advance(raw)
    for leg in ("FL", "FR"):
        leg_state(raw, leg, air=True, bottom=.001)
    role = ev.observe(raw)["transfer_roles"]["RR"]
    assert role["observed_support_contacts"] == ["RL", "RR"]
    assert not role["preparation_ready"] and not role["transfer_ready"]
    assert ev._transfer_tracker.response_since["RR"] is None
    assert ev._transfer_tracker.transfer_response_since["RR"] is None
    assert ev.snapshot["history"] == before_history


def test_invalid_geometry_clears_evidence_not_fabricates_missing_measurements():
    ev, raw = evaluator()
    raw, _ = prepare_space(ev, raw, "RR")
    raw, _ = new_drop(ev, raw, "RR")
    tracker = ev._transfer_tracker
    snapshot = ev.snapshot
    frozen = deepcopy(snapshot)
    raw["center_of_mass"]["valid"] = False
    result = tracker.observe(raw, snapshot)
    assert all(not row["valid"] for row in result.values())
    assert not tracker.samples
    assert all(value is None for value in tracker.response_since.values())
    assert all(value is None for value in tracker.transfer_response_since.values())
    assert snapshot == frozen
    assert transfer_role_observation_features({"transfer_roles": result}) == (0.,)*48


@pytest.mark.parametrize("invalid_leg", LEGS)
def test_unknown_normalized_bearing_load_cannot_mature_or_reuse_transfer(invalid_leg):
    ev, raw = evaluator()
    raw, _ = prepare_space(ev, raw, "RR")
    raw, _ = new_drop(ev, raw, "RR")
    tracker = ev._transfer_tracker
    snapshot = ev.snapshot
    before_history = deepcopy(snapshot["history"])
    snapshot["current_legs"][invalid_leg].update(load_fraction_valid=False, load_fraction=0.)
    result = tracker.observe(raw, snapshot)
    assert all(not role["valid"] and not role["transfer_ready"] for role in result.values())
    assert all(role["reason"] == "normalized bearing load is unavailable" for role in result.values())
    assert not tracker.samples
    assert all(value is None for value in tracker.transfer_response_since.values())
    assert all(value is None for value in tracker.response_since.values())
    assert snapshot["history"] == before_history
    # Once evidence is available again the old mature physical window is gone.
    raw = advance(raw)
    role = ev.observe(raw)["transfer_roles"]["RR"]
    assert role["valid"] and not role["transfer_ready"]
    assert role["transfer_direction_context"]["continued_response_duration_s"] == 0.


def test_legacy_does_not_reinterpret_new_optional_load_validity_metadata():
    ev, raw = evaluator(enabled=False)
    raw, _ = prepare_space(ev, raw, "RR")
    raw, _ = new_drop(ev, raw, "RR")
    left, right = TransferRoleTracker(ev.spec), TransferRoleTracker(ev.spec)
    snapshot = ev.snapshot
    annotated = deepcopy(snapshot)
    for row in annotated["current_legs"].values():
        row["load_fraction_valid"] = False
    assert left.observe(raw, snapshot) == right.observe(raw, annotated)


def test_phase_labels_do_not_restart_physical_attempt_or_alter_qcp():
    ev, raw = evaluator()
    raw, _ = prepare_space(ev, raw, "RR")
    raw, _ = new_drop(ev, raw, "RR")
    before = deepcopy(ev.snapshot["history"])
    start = ev._transfer_tracker.transfer_response_since["RR"]
    # Tracker consumes physical evidence only; it has no phase reset callback.
    reference = TransferRoleTracker(ev.spec)
    changed_labels = TransferRoleTracker(ev.spec)
    for sample in ev._transfer_tracker.samples:
        assert "stage_id" not in sample
    for tick, phase in enumerate(("P07", "P08", "P09", "P09"), start=1):
        raw = advance(raw)
        evaluation = ev.observe(raw)
        same = deepcopy(evaluation)
        same["stage_id"] = phase
        assert reference.observe(raw, evaluation) == changed_labels.observe(raw, same)
        assert ev._transfer_tracker.transfer_response_since["RR"] == start
        assert evaluation["transfer_roles"]["RR"]["transfer_direction_context"]["continued_response_duration_s"] == pytest.approx(tick/120.)
    assert ev.snapshot["history"] == before


def test_existing_372_slots_separately_expose_preparation_and_transfer_maturity():
    ev, raw = evaluator()
    raw, _ = prepare_space(ev, raw, "RR")
    raw, first = new_drop(ev, raw, "RR")
    encoded = transfer_role_observation_features(ev.snapshot)
    start = 12*LEGS.index("RR")
    assert ROLE_OBSERVATION_DIM == 372 and len(encoded) == 48
    assert ROLE_OBSERVATION_FIELDS[10] == "continued_response_fraction"
    assert encoded[start+2] == 1. and encoded[start+5] == 1.
    assert encoded[start+3] == 0. and encoded[start+6] == 0. and encoded[start+10] == 0.
    for _ in range(4):
        raw = advance(raw)
        ev.observe(raw)
    half = transfer_role_observation_features(ev.snapshot)
    assert half[start+10] == pytest.approx(.5)
    assert half[start+2] == first["preparation_progress"] == 1.
    assert half[start+3] == pytest.approx(.5)


@pytest.mark.parametrize("version", ["all_stage_typo", True, 1, {}])
def test_unknown_acceptance_version_is_not_silently_legacy(version):
    ev, _ = evaluator()
    spec = deepcopy(ev.spec)
    spec["physical_acceptance_version"] = version
    with pytest.raises(ValueError, match="physical acceptance version"):
        TransferRoleTracker(spec)

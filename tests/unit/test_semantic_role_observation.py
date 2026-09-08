"""Narrow 324-prefix/48-role encoding tests; no simulator or policy training."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import struct

import pytest

from test_semantic_observation_reward_env import _frame, _raw, _sample, ZERO12
from wlr50_clean.ppo.semantic_observation import (
    HISTORY_GROUPS, SemanticObservationBuilder, SemanticObservationError,
    load_semantic_observation_schema, transfer_role_observation_features,
)
from wlr50_clean.ppo.semantic_reward import SemanticRewardCalculator, load_semantic_reward_config
from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator, TaskStageSupervisor, load_task_spec
from wlr50_clean.ppo.semantic_transfer_roles import (
    MODE, LEGS, ROLE_OBSERVATION_LAYOUT, ROLE_OBSERVATION_GROUP,
    ROLE_OBSERVATION_BASE_DIM, ROLE_OBSERVATION_DIM, ROLE_OBSERVATION_FIELDS,
)

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "configs/ppo_semantic_v3/observation_schema.json"
SPEC = ROOT / "configs/ppo_semantic_v3/stage_task_spec.yaml"


def _role(index=0):
    return {
        "valid": True, "workspace_progress": .1 + .1*index,
        "preparation_progress": .2, "transfer_progress": .3, "motion_fraction": .4,
        "preparation_ready": False, "transfer_ready": True,
        "transfer_direction_context": {
            "fixed_direction_world": (-.6, .8, 0.),
            "short_support_continuity_fraction": .75,
            "minimum_evidence_s": 1/15,
            "continued_response_duration_s": 1/30, "window_s": .5,
        },
    }


def _legacy_v3_schema():
    # Start from the unchanged v2 324 layout, independently applying only the
    # already-reviewed FR-knee scales. Do not derive expected columns from 372.
    legacy = load_semantic_observation_schema()
    groups = deepcopy(legacy.groups)
    for group in groups:
        if group["name"] in ("previous_residual_full12", "previous_previous_residual_full12"):
            group["scale"][3] = 6
    return replace(legacy, groups=groups)


def _build(frame, schema=None):
    schema = schema or load_semantic_observation_schema(SCHEMA)
    return SemanticObservationBuilder(schema).build(frame, dict.fromkeys(HISTORY_GROUPS, ZERO12))


def test_production_optin_has_exact_shared_layout_and_unmodified_324_groups():
    schema = load_semantic_observation_schema(SCHEMA)
    assert ROLE_OBSERVATION_LAYOUT == "diagonal_transfer_state_v1"
    assert ROLE_OBSERVATION_GROUP == "transfer_role_context_full48"
    assert LEGS == ("FL", "FR", "RL", "RR")
    assert ROLE_OBSERVATION_FIELDS == (
        "valid", "workspace_progress", "preparation_progress", "transfer_progress",
        "motion_fraction", "preparation_ready", "transfer_ready", "fixed_direction_world_x",
        "fixed_direction_world_y", "short_support_continuity_fraction",
        "continued_response_fraction", "window_evidence_fraction",
    )
    assert schema.transfer_role_features_version == ROLE_OBSERVATION_LAYOUT
    assert schema.dimension == ROLE_OBSERVATION_DIM == 372
    assert _legacy_v3_schema().dimension == ROLE_OBSERVATION_BASE_DIM == 324
    assert schema.groups[:-1] == _legacy_v3_schema().groups
    assert schema.groups[-1] == {"name": ROLE_OBSERVATION_GROUP, "size": 48, "scale": 1.0}


def test_all_372_columns_and_two_step_324_prefix_are_byte_identical():
    schema, legacy = load_semantic_observation_schema(SCHEMA), _legacy_v3_schema()
    new_builder, old_builder = SemanticObservationBuilder(schema), SemanticObservationBuilder(legacy)
    history = {key: tuple(float(i) for i in range(12)) for key in HISTORY_GROUPS}
    for tick in (0, 8):
        frame = _frame(tick, raw=_raw(tick, pitch=.12 + tick*.001, joint=12.))
        frame.info["semantic_task"]["transfer_roles"] = {leg: _role(i) for i, leg in enumerate(LEGS)}
        new = schema.encode(new_builder.build(frame, history).groups)
        old = legacy.encode(old_builder.build(frame, history).groups)
        assert struct.pack("!324d", *new[:324]) == struct.pack("!324d", *old)
        for index, leg in enumerate(LEGS):
            start = 324 + 12*index
            assert new[start:start+12] == pytest.approx(
                (1., .1+.1*index, .2, .3, .4, 0., 1., -.6, .8, .75, .5, 1.))
        assert len(new) == 372


def test_legacy_schema_ignores_optional_role_metadata_exactly():
    schema = load_semantic_observation_schema()
    assert schema.dimension == 324 and schema.transfer_role_features_version is None
    before = _frame()
    after = deepcopy(before)
    after.info["semantic_task"]["transfer_roles"] = "unused_by_legacy_encoder"
    assert schema.encode(_build(before, schema).groups) == schema.encode(_build(after, schema).groups)


@pytest.mark.parametrize("case", ["unknown", "unmarked", "wrong_size", "wrong_scale", "bool_scale", "reordered", "extra_key", "wrong_base"])
def test_malformed_or_unversioned_append_schema_is_rejected(tmp_path, case):
    data = json.loads(SCHEMA.read_text(encoding="utf-8"))
    if case == "unknown": data["transfer_role_features_version"] = "unknown"
    elif case == "unmarked": del data["transfer_role_features_version"]
    elif case == "wrong_size": data["feature_groups"][-1]["size"] = 47
    elif case == "wrong_scale": data["feature_groups"][-1]["scale"] = 2.
    elif case == "bool_scale": data["feature_groups"][-1]["scale"] = True
    elif case == "reordered": data["feature_groups"][-2:] = reversed(data["feature_groups"][-2:])
    elif case == "extra_key": data["feature_groups"][-1]["unexpected"] = True
    elif case == "wrong_base": data["feature_groups"][0]["size"] += 1
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SemanticObservationError):
        load_semantic_observation_schema(path)


def test_missing_optional_and_explicit_invalid_rows_have_unavailable_zero_fillers():
    assert transfer_role_observation_features({}) == (0.,)*48
    rows = {"FL": {"valid": False, "motion_fraction": math.nan}, "FR": _role(1)}
    encoded = transfer_role_observation_features({"transfer_roles": rows})
    assert encoded[:12] == (0.,)*12
    assert encoded[12] == 1. and encoded[16] == .4
    assert encoded[24:] == (0.,)*24
    # Existing 324-only complete-physics fixtures can explicitly lack role rows.
    assert _build(_frame()).groups[ROLE_OBSERVATION_GROUP] == (0.,)*48


@pytest.mark.parametrize("key,value", [
    ("valid", 1), ("workspace_progress", math.nan), ("workspace_progress", math.inf),
    ("workspace_progress", -.01), ("transfer_progress", 1.01), ("motion_fraction", True),
    ("preparation_ready", 1), ("transfer_ready", "true"),
])
def test_declared_valid_role_malformed_values_fail_closed(key, value):
    row = _role(); row[key] = value
    with pytest.raises(SemanticObservationError):
        transfer_role_observation_features({"transfer_roles": {"RR": row}})


@pytest.mark.parametrize("key,value", [
    ("fixed_direction_world", (math.nan, 0., 0.)),
    ("fixed_direction_world", (0., 0.)),
    ("fixed_direction_world", (0., 0., .1)),
    ("fixed_direction_world", (1.01, 0., 0.)),
    ("short_support_continuity_fraction", 1.1),
    ("minimum_evidence_s", 0.), ("minimum_evidence_s", math.inf),
    ("continued_response_duration_s", -.1), ("window_s", math.nan),
])
def test_declared_valid_context_malformed_values_fail_closed(key, value):
    row = _role(); row["transfer_direction_context"][key] = value
    with pytest.raises(SemanticObservationError):
        transfer_role_observation_features({"transfer_roles": {"FL": row}})


@pytest.mark.parametrize("rows", [[], {"extra_leg": _role()}, {"FL": {}}, {"FL": 1}])
def test_invalid_mapping_is_not_silently_interpreted_as_absence(rows):
    with pytest.raises(SemanticObservationError):
        transfer_role_observation_features({"transfer_roles": rows})


def test_valid_row_missing_required_fields_is_not_silently_zeroed():
    for key in ("motion_fraction", "transfer_direction_context"):
        row = _role(); del row[key]
        with pytest.raises(SemanticObservationError):
            transfer_role_observation_features({"transfer_roles": {"FL": row}})
    row = _role(); del row["transfer_direction_context"]["minimum_evidence_s"]
    with pytest.raises(SemanticObservationError):
        transfer_role_observation_features({"transfer_roles": {"FL": row}})


def test_maturity_clips_at_existing_denominator_without_overflow_or_absolute_time_feature():
    row = _role()
    context = row["transfer_direction_context"]
    context.update(continued_response_duration_s=0., window_s=1/15)
    first = transfer_role_observation_features({"transfer_roles": {"FL": row}})
    assert first[10:12] == (0., 1.)
    context.update(minimum_evidence_s=5e-324, continued_response_duration_s=1e308, window_s=1e308)
    assert transfer_role_observation_features({"transfer_roles": {"FL": row}})[10:12] == (1., 1.)


def test_actual_evaluator_and_supervisor_provide_config_bound_context_without_new_gates():
    supervisor = TaskStageSupervisor(SPEC)
    schema = load_semantic_observation_schema(SCHEMA)
    for tick in range(9):
        raw = _raw(tick, joint=0.)
        task = supervisor.observe_and_update(raw)
        frame = _frame(tick, raw=raw)
        frame.info["semantic_task"] = task
        frame.state_id = task["stage_id"]
        history = deepcopy(task["history"])
        built = _build(frame, schema)
        assert task["history"] == history
        assert task["termination_reason"] is None
        assert len(schema.encode(built.groups)) == 372
        for index, leg in enumerate(LEGS):
            role = task["transfer_roles"][leg]
            context = role["transfer_direction_context"]
            assert context["minimum_evidence_s"] == load_task_spec(SPEC)["transfer_roles"]["minimum_evidence_s"]
            assert built.groups[ROLE_OBSERVATION_GROUP][index*12] == 1.
            assert built.groups[ROLE_OBSERVATION_GROUP][index*12+4] == role["motion_fraction"]


def test_same_324_consumer_state_can_expose_different_role_motion_and_readiness():
    # A synthetic consumer-state counterexample, not a claim of a new physical
    # rollout: after hard Q+C, unplaced FL transfer credit is retired to 1.
    # Different role-window response can still change motion weighting without
    # changing physical Phi, current physical features, or P05 goal progress.
    evaluator = TaskEvaluator(SPEC)
    ev = deepcopy(evaluator.observe(_raw(0, pitch=.2)))
    ev["history"]["placed"]["FR"] = True
    ev["history"]["active_lift"]["FL"] = True
    ev["history"]["front_edge_crossed"]["FL"] = True
    ev["current_legs"]["FL"].update(air=True, ground_contact=False, initial_clearance=True)
    ev["transfer_roles"] = {leg: _role(i) for i, leg in enumerate(LEGS)}
    other = deepcopy(ev)
    ev["transfer_roles"]["FL"].update(motion_fraction=0., preparation_ready=False, transfer_ready=False)
    other["transfer_roles"]["FL"].update(motion_fraction=1., preparation_ready=True, transfer_ready=True)
    supervisor = TaskStageSupervisor(SPEC)
    phi = supervisor.physical_potential(ev)
    assert supervisor.physical_potential(other) == phi
    schema = load_semantic_observation_schema(SCHEMA)
    states = []
    for evaluation in (ev, other):
        frame = _frame(8, stage="P05", phi=phi, raw=_raw(8, pitch=.2))
        task = frame.info["semantic_task"]
        task.update(transfer_roles=evaluation["transfer_roles"], transfer_roles_version=MODE,
                    physical_transfer_fraction=evaluation["transfer_roles"]["FL"]["motion_fraction"])
        for name, key in (("active_lift_history", "active_lift"),
                          ("front_edge_crossed_history", "front_edge_crossed"), ("placed_history", "placed")):
            task[name] = deepcopy(evaluation["history"][key])
        states.append(_build(frame, schema))
    left, right = (schema.encode(state.groups) for state in states)
    assert left[:324] == right[:324]
    assert (left[328], left[329], left[330]) == (0., 0., 0.)
    assert (right[328], right[329], right[330]) == (1., 1., 1.)
    config = load_semantic_reward_config(ROOT / "configs/ppo_semantic_v3/reward_config.yaml")
    before = _build(_frame(7, stage="P05", phi=phi, raw=_raw(7, pitch=.2)), schema)
    results = [SemanticRewardCalculator(config).evaluate(before, state, [_sample(before, state)],
               termination_reason=None, task_success=False) for state in states]
    assert results[0]["families"]["task_progress"] == results[1]["families"]["task_progress"]
    assert results[0]["families"]["body_stability"] < results[1]["families"]["body_stability"]


def test_phase_change_does_not_reset_supplied_role_window_or_rewrite_existing_features():
    states = []
    for stage in ("P07", "P08", "P09"):
        frame = _frame(80, stage=stage)
        frame.info["semantic_task"]["transfer_roles"] = {"RR": _role()}
        states.append(_build(frame).groups[ROLE_OBSERVATION_GROUP])
    assert states[0] == states[1] == states[2]

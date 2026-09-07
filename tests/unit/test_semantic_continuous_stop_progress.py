"""Four-type stop shaping through real evaluator, reward and encoder APIs.

Sensor sequences earn placement history; scalar counterfactuals test only the
soft dependency and do not claim a physical trajectory or learned success.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from test_semantic_supervisor import advance, leg_state, observation, place
from test_semantic_observation_reward_env import _built, _frame, _sample
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.semantic_observation import (
    GOAL_KEYS, HISTORY_GROUPS, SemanticObservationBuilder,
    load_semantic_observation_schema,
)
from wlr50_clean.ppo.semantic_reward import (
    FAMILIES, SemanticRewardCalculator, load_semantic_reward_config,
)
from wlr50_clean.ppo.semantic_supervisor import (
    DEFAULT_TASK_SPEC_PATH, GOAL_FEATURE_KEYS, LEG_ORDER, PHASE_IDS,
    STOP_PROGRESS_MODE, SemanticObservationError, TaskEvaluator,
    TaskStageSupervisor, load_task_spec,
)


CONFIG = Path(__file__).resolve().parents[2] / "configs/ppo_semantic_v3"
SPEC = CONFIG / "stage_task_spec.yaml"
PAYLOAD = ("measured_wheel_velocity_rad_s", "applied_wheel_command_rad_s",
           "stop_progress_wheel_order")
TOLERANCES = ("maximum_wheel_speed_rad_s", "maximum_commanded_wheel_speed_rad_s",
              "maximum_body_linear_speed_m_s", "maximum_body_angular_speed_rad_s")
THRESHOLDS = (.25, .02, .05, .30)


def _path(tmp_path, spec, name="task.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return path


def _placed_entry(path=SPEC):
    evaluator = TaskEvaluator(path)
    obs = observation()
    for joint in obs["joints"].values():
        joint["command_deg"] = 0.
    evaluator.observe(obs)
    for leg in ("FR", "FL", "RR", "RL"):
        obs = place(evaluator, obs, leg)
    assert all(evaluator.snapshot["history"]["placed"].values())
    assert not evaluator.snapshot["success"]
    obs = advance(obs)
    obs["base"]["position_w_m"][0] = .9
    for leg in LEG_ORDER:
        leg_state(obs, leg, x=.8, bottom=.05, top=True, hip=0.)
    return evaluator, obs


def _rates(obs, speeds=(0.,)*4, commands=(0.,)*4, linear=0., angular=0.):
    obs = deepcopy(obs)
    for name, speed, command in zip(WHEEL_ORDER, speeds, commands, strict=True):
        obs["wheels"][name].update(velocity_rad_s=speed, command_rad_s=command)
    obs["base"]["linear_velocity_w_m_s"] = [linear, 0., 0.]
    obs["base"]["angular_velocity_w_rad_s"] = [0., 0., angular]
    return obs


def _case(evaluator, obs, *, path=SPEC, **rates):
    ev = deepcopy(evaluator)
    snapshot = ev.observe(_rates(obs, **rates))
    sup = TaskStageSupervisor(path, evaluator=ev, initial_stage_id="P13")
    return snapshot, sup.predicate("whole_task_success", snapshot), sup.physical_potential(snapshot)


def _q(value, tolerance):
    return tolerance / (tolerance + abs(value))


@pytest.fixture(params=[pytest.param(.995, id="legacy_v3_gamma_0995"),
                       pytest.param(.9985, id="current_v3_gamma_09985")])
def matched_reward_config(request, tmp_path):
    config = load_semantic_reward_config(CONFIG / "reward_config.yaml")
    if request.param == .995:
        # Keep the original v3 quality costs and PBRS weight while explicitly
        # restoring its legacy, unmarked return profile (not the v2 costs).
        values = deepcopy(dict(config.values))
        values.pop("return_profile")
        values["gamma"] = .995
        config = load_semantic_reward_config(_path(tmp_path, values, "legacy_v3_reward.yaml"))
    assert config.gamma == request.param
    assert config.values["potential_weight"] == 5.
    return config


def _reward(a, b, *, terminal=None, success=False, config=None):
    before, after = _built(_frame(phi=a)), _built(_frame(1, phi=b))
    calculator = SemanticRewardCalculator(
        config if config is not None else load_semantic_reward_config(CONFIG / "reward_config.yaml"))
    return calculator.evaluate(before, after, [_sample(before, after)],
                               termination_reason=terminal, task_success=success)


def test_actual_evaluator_preserves_signed_order_clock_source_and_hard_maxima():
    ev, obs = _placed_entry()
    speeds, commands = (.1, -.2, .3, -.4), (-.01, .02, -.03, .04)
    snapshot, _, _ = _case(ev, obs, speeds=speeds, commands=commands)
    assert snapshot[PAYLOAD[0]] == speeds
    assert snapshot[PAYLOAD[1]] == commands
    assert snapshot[PAYLOAD[2]] == WHEEL_ORDER
    assert all(isinstance(snapshot[key], tuple) for key in PAYLOAD)
    assert snapshot["goal_features"]["maximum_wheel_speed_rad_s"] == .4
    assert snapshot["maximum_commanded_wheel_speed_rad_s"] == .04
    assert not snapshot["final_controlled"]
    assert snapshot["physics_tick"] == obs["physics_tick"]
    assert snapshot["simulation_time_s"] == obs["simulation_time_s"]
    assert snapshot["source"] == "current_episode_live_joint_geometry_exact_contact_history"
    assert tuple(snapshot["goal_features"]) == GOAL_FEATURE_KEYS
    assert len(GOAL_FEATURE_KEYS) == 17  # No per-wheel diagnostic actor features.
    with pytest.raises(TypeError):
        snapshot[PAYLOAD[0]][0] = 999.


def test_json_round_trip_lists_preserve_finish_and_global_potential():
    ev, obs = _placed_entry()
    snapshot = ev.observe(_rates(obs, speeds=(.1, -.4, .3, -.2),
        commands=(-.03, .04, -.01, .02), linear=.08, angular=.4))
    restored = json.loads(json.dumps(snapshot, sort_keys=True, allow_nan=False))
    assert all(isinstance(restored[key], list) for key in PAYLOAD)
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    assert sup.predicate("whole_task_success", restored) == sup.predicate("whole_task_success", snapshot)
    assert sup.physical_potential(restored) == sup.physical_potential(snapshot)


@pytest.mark.parametrize("kind,tolerance", zip(("speeds", "commands", "linear", "angular"), THRESHOLDS))
def test_each_type_has_strict_above_and_below_threshold_progress(kind, tolerance):
    ev, obs = _placed_entry()
    values = []
    for scale in (4., 2., 1., .5, 0.):
        value = scale*tolerance
        snapshot, finish, phi = _case(ev, obs, **{kind: (value,)*4 if kind in ("speeds", "commands") else value})
        q = 1./(1.+scale)
        control = (3.+q)/4.
        assert finish == pytest.approx(.4+.3+.2*control)
        assert phi == pytest.approx(.85+.15*finish)
        assert not snapshot["success"]
        assert snapshot["final_controlled"] is (scale <= 1.)
        values.append(phi)
    assert all(a < b for a, b in zip(values, values[1:]))
    assert all(a > b for a, b in zip(reversed(values), list(reversed(values))[1:]))


def test_exact_threshold_is_continuous_half_progress_not_new_hard_zero():
    ev, obs = _placed_entry()
    scores = []
    for factor in (1.-1e-8, 1., 1.+1e-8):
        snapshot, finish, _ = _case(ev, obs, speeds=(.25*factor,)*4,
            commands=(.02*factor,)*4, linear=.05*factor, angular=.30*factor)
        scores.append(finish)
        assert snapshot["final_controlled"] is (factor <= 1.)
        assert not snapshot["success"]
    assert scores[1] == pytest.approx(.8)  # all four q(T)=.5
    assert 0. < scores[0]-scores[1] < 1e-8
    assert 0. < scores[1]-scores[2] < 1e-8


@pytest.mark.parametrize("kind,tolerance", [("speeds", .25), ("commands", .02)])
@pytest.mark.parametrize("index", range(4))
def test_nonmaximum_wheel_changes_only_its_existing_one_sixteenth_control_share(kind, tolerance, index):
    ev, obs = _placed_entry()
    before = [4*tolerance]*4
    before[index] = 2*tolerance
    after = list(before)
    after[index] = tolerance
    a, _, pa = _case(ev, obs, **{kind: tuple(before)})
    b, _, pb = _case(ev, obs, **{kind: tuple(after)})
    maximum = "maximum_commanded_wheel_speed_rad_s" if kind == "commands" else "maximum_wheel_speed_rad_s"
    assert (a[maximum] if kind == "commands" else a["goal_features"][maximum]) == 4*tolerance
    assert (b[maximum] if kind == "commands" else b["goal_features"][maximum]) == 4*tolerance
    assert pb-pa == pytest.approx(.001875*(_q(tolerance, tolerance)-_q(2*tolerance, tolerance)))
    assert not a["final_controlled"] and not b["final_controlled"]
    assert a["history"] == b["history"]


def test_four_types_have_equal_weight_and_no_duplicate_legacy_command_term(tmp_path):
    ev, obs = _placed_entry()
    base = _case(ev, obs)[2]
    losses = [base-_case(ev, obs, **{kind: (t,)*4 if kind in ("speeds", "commands") else t})[2]
              for kind, t in zip(("speeds", "commands", "linear", "angular"), THRESHOLDS)]
    assert losses == pytest.approx([.0075*.5]*4)
    # The old command opt-in may remain declared, but is not a fifth term.
    spec = load_task_spec(SPEC)
    spec["final"].pop("stop_command_progress", None)
    path = _path(tmp_path, spec)
    assert _case(ev, obs, commands=(.08,)*4, path=path)[1:] == pytest.approx(
        _case(ev, obs, commands=(.08,)*4)[1:])


def test_signs_and_wheel_permutation_preserve_soft_aggregate():
    ev, obs = _placed_entry()
    speeds, commands = (.1, -.4, .3, -.2), (-.03, .04, -.01, .02)
    value = _case(ev, obs, speeds=speeds, commands=commands, linear=.08, angular=.4)[1:]
    mirrored = _case(ev, obs, speeds=tuple(-x for x in speeds), commands=tuple(-x for x in commands), linear=-.08, angular=-.4)[1:]
    reordered = _case(ev, obs, speeds=tuple(reversed(speeds)), commands=tuple(reversed(commands)), linear=.08, angular=.4)[1:]
    assert value == pytest.approx(mirrored)
    assert value == pytest.approx(reordered)


def test_phase_label_independence_finish_history_activation_and_preserved_coefficients():
    ev, obs = _placed_entry()
    snap, _, phi = _case(ev, obs, commands=(.08,)*4)
    assert [TaskStageSupervisor(SPEC, evaluator=ev, initial_stage_id=p).physical_potential(snap)
            for p in PHASE_IDS] == [phi]*13
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    base = sup.predicate("whole_task_success", snap)
    saved = deepcopy(snap)
    half_settled = deepcopy(snap)
    half_settled["final_stable_for_s"] = .25
    assert sup.predicate("whole_task_success", half_settled)-base == pytest.approx(.1*.5)
    less_history = deepcopy(snap)
    less_history["history"]["placed"]["RL"] = False
    assert sup.predicate("whole_task_success", less_history)-base == pytest.approx(-.4/4)
    half_forward = deepcopy(snap)
    half_forward["goal_features"]["body_forward_m"] = .075
    assert sup.predicate("whole_task_success", half_forward)-base == pytest.approx(-.3*.5)
    slow = deepcopy(less_history)
    fast = deepcopy(less_history)
    slow[PAYLOAD[1]], fast[PAYLOAD[1]] = (0.,)*4, (.6,)*4
    assert sup.physical_potential(slow) == sup.physical_potential(fast)
    assert snap == saved  # Soft queries do not mutate earned history.
    almost = deepcopy(snap)
    almost[PAYLOAD[0]], almost[PAYLOAD[1]] = (0.,)*4, (0.,)*4
    almost["final_stable_for_s"] = .49
    assert sup.predicate("whole_task_success", almost) == .99
    assert sup.physical_potential(almost) == pytest.approx(.9985)


def test_real_reward_uses_only_existing_single_pbrs_term_and_can_be_negative_while_improving(matched_reward_config):
    ev, obs = _placed_entry()
    a = _case(ev, obs, commands=(.08,)*4)[2]
    b = _case(ev, obs, commands=(.079,)*4)[2]
    assert b > a
    gamma = matched_reward_config.gamma
    baseline = _reward(a, a, config=matched_reward_config)
    improving = _reward(a, b, config=matched_reward_config)
    assert tuple(improving["families"]) == FAMILIES and len(FAMILIES) == 5
    assert improving["potential_shaping"] == pytest.approx(5*(gamma*b-a))
    assert improving["potential_shaping"] < 0.
    assert improving["total"] < 0.
    assert improving["terminal_event"] == 0.
    assert improving["total"]-baseline["total"] == pytest.approx(5*gamma*(b-a))
    for family in FAMILIES[1:]:
        assert improving["families"][family] == baseline["families"][family]


@pytest.mark.parametrize("reason,success", [("SUCCESS", True), ("BODY_COLLISION", False),
    ("INCOMPLETE_CONTROLLER_BLOCKED", False), ("TASK_TIMEOUT", False)])
def test_real_terminal_reward_absorbs_phi_and_never_bootstraps(reason, success):
    ev, obs = _placed_entry()
    phi = _case(ev, obs, commands=(.04,)*4)[2]
    result = _reward(phi, 1., terminal=reason, success=success)
    assert result["potential_after"] == 0.
    assert result["potential_shaping"] == pytest.approx(-5*phi)
    assert result["terminal_bootstrap_allowed"] is False
    assert result["terminal_event"] == (40. if success else -40.)


def test_hovering_is_not_a_bonus_and_terminal_discounted_shaping_telescopes(matched_reward_config):
    ev, obs = _placed_entry()
    phis = [_case(ev, obs, commands=(c,)*4)[2] for c in (.16, .08, .02)]
    gamma = matched_reward_config.gamma
    hover = _reward(phis[0], phis[0], config=matched_reward_config)["potential_shaping"]
    assert hover == pytest.approx(5*(gamma-1.)*phis[0])
    assert hover < 0.
    if gamma == .995:
        assert hover == pytest.approx(-5*.005*phis[0])
    terms = [_reward(phis[0], phis[1], config=matched_reward_config)["potential_shaping"],
             _reward(phis[1], phis[2], config=matched_reward_config)["potential_shaping"],
             _reward(phis[2], 1., terminal="SUCCESS", success=True,
                     config=matched_reward_config)["potential_shaping"]]
    assert sum(gamma**i*term for i, term in enumerate(terms)) == pytest.approx(-5*phis[0])


@pytest.mark.parametrize("failure", ["speed", "command", "linear", "angular", "region", "support", "collision", "fall", "joint_limit"])
def test_soft_control_cannot_bypass_any_existing_hard_requirement(failure):
    ev, obs = _placed_entry()
    if failure in ("speed", "command"):
        field = "velocity_rad_s" if failure == "speed" else "command_rad_s"
        obs["wheels"][WHEEL_ORDER[0]][field] = .251 if failure == "speed" else .0201
    elif failure == "linear":
        obs["base"]["linear_velocity_w_m_s"] = [.051, 0., 0.]
    elif failure == "angular":
        obs["base"]["angular_velocity_w_rad_s"] = [0., 0., .301]
    elif failure == "region":
        obs["base"]["position_w_m"][1] = 2.
    elif failure == "support":
        for leg in LEG_ORDER:
            leg_state(obs, leg, x=.8, bottom=.05, air=True)
    elif failure == "collision":
        obs["body_collision"]["detected"] = True
    elif failure == "fall":
        obs["imu"]["projected_gravity_b"] = [0., 0., 1.]
    else:
        obs["joints"][SERVO_ORDER[0]]["position_deg"] = 999.
    for _ in range(62):
        snap = ev.observe(obs)
        assert not snap["success"] and snap["final_stable_for_s"] == 0.
        obs = advance(obs)
    if failure in ("speed", "command", "linear", "angular"):
        assert not snap["final_controlled"]


def test_missing_earned_history_and_zero_speed_do_not_complete():
    ev, obs = TaskEvaluator(SPEC), observation()
    for joint in obs["joints"].values():
        joint["command_deg"] = 0.
    for _ in range(62):
        snap = ev.observe(obs)
        assert snap["final_controlled"]
        assert not any(snap["history"]["placed"].values())
        assert not snap["success"]
        obs = advance(obs)


def test_exact_existing_rate_thresholds_complete_after_same_continuous_half_second():
    ev, obs = _placed_entry()
    obs = _rates(obs, speeds=(.25,)*4, commands=(.02,)*4, linear=.05, angular=.30)
    sup = TaskStageSupervisor(SPEC, evaluator=ev)
    for i in range(61):
        snap = ev.observe(obs)
        assert snap["final_controlled"]
        assert snap["success"] is (i == 60)
        if i == 0:
            assert sup.predicate("whole_task_success", snap) == pytest.approx(.8)
        obs = advance(obs)
    assert snap["final_stable_for_s"] == pytest.approx(.5)
    assert sup.predicate("whole_task_success", snap) == 1.


@pytest.mark.parametrize("field", PAYLOAD[:2])
@pytest.mark.parametrize("bad", [None, (), (0.,)*3, (0.,)*5, (0., 0., 0., float("nan")),
    (0., 0., 0., float("inf")), (0., 0., 0., True)])
def test_missing_malformed_or_nonfinite_new_snapshot_vectors_fail_explicitly(field, bad):
    ev, obs = _placed_entry()
    snap = ev.observe(obs)
    if bad is None:
        snap.pop(field)
    else:
        snap[field] = bad
    with pytest.raises(SemanticObservationError):
        TaskStageSupervisor(SPEC, evaluator=ev).predicate("whole_task_success", snap)


@pytest.mark.parametrize("order", [None, tuple(reversed(WHEEL_ORDER)), ("FL", "FR", "RL", "RR")])
def test_missing_or_wrong_measured_wheel_order_is_rejected(order):
    ev, obs = _placed_entry()
    snap = ev.observe(obs)
    snap[PAYLOAD[2]] = order
    with pytest.raises(SemanticObservationError, match="order"):
        TaskStageSupervisor(SPEC, evaluator=ev).predicate("whole_task_success", snap)


@pytest.mark.parametrize("field", ["velocity_rad_s", "command_rad_s"])
@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf")])
def test_live_measurements_cannot_be_replaced_by_zero(field, bad):
    ev, obs = _placed_entry()
    obs["wheels"][WHEEL_ORDER[0]][field] = bad
    with pytest.raises(SemanticObservationError, match="finite"):
        ev.observe(obs)


@pytest.mark.parametrize("key", TOLERANCES)
@pytest.mark.parametrize("bad", [0., -.01, float("nan"), float("inf"), True])
def test_loader_rejects_nonpositive_or_nonfinite_existing_tolerances(tmp_path, key, bad):
    spec = load_task_spec(SPEC)
    spec["final"][key] = bad
    with pytest.raises(ValueError):
        load_task_spec(_path(tmp_path, spec))


@pytest.mark.parametrize("change", ["unknown", "missing_global", "missing_physical_stop"])
def test_loader_requires_explicit_known_mode_and_compatible_progress(tmp_path, change):
    spec = load_task_spec(SPEC)
    if change == "unknown":
        spec["final"]["stop_progress_semantics"] = "guess_zero"
    elif change == "missing_global":
        spec.pop("potential_definition")
        spec.pop("preparation_credit_semantics", None)
        spec.pop("capture_retention_semantics", None)
        # Isolate this older stop-mode dependency error from newer progress modes.
        spec.pop("capture_approach_semantics", None)
        spec.pop("workspace_potential_semantics", None)
    else:
        spec["final"].pop("stop_pose_semantics")
    with pytest.raises(ValueError, match="continuous stop"):
        load_task_spec(_path(tmp_path, spec))


def test_absent_mode_preserves_old_v3_formulas_and_v2_payload_home_gate(tmp_path):
    assert STOP_PROGRESS_MODE == "per_wheel_four_type_threshold_ratio_v1"
    spec = load_task_spec(SPEC)
    assert spec["final"]["stop_progress_semantics"] == STOP_PROGRESS_MODE
    spec["final"].pop("stop_progress_semantics")
    reciprocal = _path(tmp_path, spec, "old_reciprocal.yaml")
    ev, obs = _placed_entry(reciprocal)
    a, finish, phi = _case(ev, obs, path=reciprocal, commands=(.16,)*4)
    assert finish == pytest.approx(.85625) and phi == pytest.approx(.9784375)
    assert not any(key in a for key in PAYLOAD)
    spec["final"].pop("stop_command_progress")
    legacy = _path(tmp_path, spec, "old_no_command.yaml")
    ev, obs = _placed_entry(legacy)
    assert _case(ev, obs, path=legacy, commands=(.16,)*4)[1] == pytest.approx(.9)
    ev, obs = _placed_entry(DEFAULT_TASK_SPEC_PATH)
    obs["joints"][SERVO_ORDER[0]]["position_deg"] = 20.
    a, finish, _ = _case(ev, obs, path=DEFAULT_TASK_SPEC_PATH, commands=(.01,)*4)
    assert not a["final_controlled"] and finish == pytest.approx(.85)
    assert not any(key in a for key in PAYLOAD)


def test_new_diagnostics_do_not_expand_324_observation_or_change_other_encoder_groups(tmp_path):
    spec = load_task_spec(SPEC)
    spec["final"].pop("stop_progress_semantics")
    old_path = _path(tmp_path, spec)
    built = []
    schema = load_semantic_observation_schema(CONFIG / "observation_schema.json")
    for path in (old_path, SPEC):
        ev, obs = _placed_entry(path)
        obs = _rates(obs, speeds=(.5,)*4, commands=(.08,)*4, linear=.1, angular=.6)
        sup = TaskStageSupervisor(path, evaluator=ev, initial_stage_id="P13")
        task = sup.observe_and_update(obs)
        frame = _frame(obs["physics_tick"], stage="P13")
        frame.info["semantic_task"] = task
        data = SemanticObservationBuilder(schema).build(frame, dict.fromkeys(HISTORY_GROUPS, (0.,)*12))
        assert tuple(task["goal_features"]) == GOAL_FEATURE_KEYS == GOAL_KEYS
        assert schema.dimension == len(schema.encode(data.groups)) == 324
        assert len(frame.nominal_action_full12) == 12
        built.append(data.groups)
    assert built[0]["task_progress"] != built[1]["task_progress"]
    for name in built[0]:
        if name != "task_progress":
            assert built[0][name] == built[1][name], name

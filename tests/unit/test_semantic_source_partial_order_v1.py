"""CPU-only synthetic source scheduling checks; no simulated physical success."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from test_all_stage_nominal_transitions import task
from test_semantic_supervisor import observation
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider
from wlr50_clean.reference.motion_contract import load_motion_contract

ROOT = Path(__file__).resolve().parents[2]
MODE = "source_partial_order_physical_ready_v1"


@pytest.fixture(scope="module")
def contract():
    return load_motion_contract(ROOT / "configs/recording_motion_contract.json")


def make_provider(contract, phase="P07", *, enabled=True):
    spec = yaml.safe_load((ROOT / "configs/ppo_fsm_reference_p09_stable_v2/stage_task_spec.yaml").read_text())
    assert spec["nominal"]["sequence_semantics"] == MODE
    if not enabled:
        spec["nominal"].pop("sequence_semantics")
    return NominalMotionProvider.from_handoff(contract, spec=spec, stage_id=phase,
        nominal_full12=contract.phase(phase).start_full12, tracking_servo_names=())


def current_input(contract, phase, tick, *, space=False, fr_motion=0., rl_motion=0.,
                  rr_load=.35, load_valid=True, continuing=False, supports=("FR", "RL")):
    """Explicit synthetic sensor/supervisor inputs, not inferred physical motion."""
    item, raw = task(phase, tick), observation(tick)
    base_q = contract.phase("P07").start_full12[:8]
    for name, value in zip(SERVO_ORDER, base_q):
        raw["joints"][name]["position_deg"] = value
    raw["joints"]["front_right_knee"]["position_deg"] -= fr_motion
    raw["joints"]["rear_left_hip"]["position_deg"] += rl_motion
    legs = item["physical_evaluator"]["current_legs"]
    for leg, row in legs.items():
        row.update(support=leg in supports, bearing_verified=True, load_fraction_valid=True,
                   load_fraction=.3, current_lift_valid=False, motion_continuation_allowed=False)
    if space:
        legs["FL"].update(air=True, ground_contact=False, support=False, clearance_m=.004,
                          load_fraction=0.)
        raw["wheels"][WHEEL_ORDER[0]]["center_w_m"][2] += .054
    legs["RR"].update(load_fraction=rr_load, load_fraction_valid=load_valid,
                      current_lift_valid=continuing, motion_continuation_allowed=continuing,
                      ground_contact=not continuing, air=continuing)
    return item, raw


def issue(p, contract, phase, tick, **kwargs):
    item, raw = current_input(contract, phase, tick, **kwargs)
    before = deepcopy((item, raw))
    command = p.evaluate(item, raw)
    assert (item, raw) == before  # Scheduler must not fabricate readiness/history.
    return command


def layer(p, phase):
    return next(value for value in p._continuous_layers if value["stage"] == phase)


def enter_fast_chain(p, contract):
    for tick in range(8):
        issue(p, contract, "P07", tick)
    for tick in range(8, 16):
        issue(p, contract, "P08", tick)
    issue(p, contract, "P09", 16)


def test_quick_semantic_handoffs_do_not_start_pending_source_clocks(contract):
    p = make_provider(contract)
    enter_fast_chain(p, contract)
    for tick in range(17, 90):
        issue(p, contract, "P09", tick)
    assert layer(p, "P07")["ticks"] == 40
    for phase in ("P08", "P09"):
        pending = layer(p, phase)
        assert pending["ticks"] == pending["motion"]._tick_index == 0
        assert pending["sample"] is None and pending["touched"] == set()
        assert pending["sequence_diagnostic"]["status"] == "pending"
        assert "actual_start_tick" not in pending["sequence_diagnostic"]
    assert p.elapsed_s == 0. and not p.endpoint_issued


def test_fl_actual_space_releases_whole_fr_atomic_group_without_catchup(contract):
    p = make_provider(contract)
    for tick in range(40):
        held = issue(p, contract, "P07", tick)
    source = layer(p, "P07")
    atomic_count = source["motion"].source_atomic_emitted
    for tick in range(40, 93):
        assert issue(p, contract, "P07", tick) == held
        assert source["ticks"] == source["motion"]._tick_index == 40
        assert source["sample"].tick_index == 39
        assert source["sequence_diagnostic"]["observation_tick"] == tick
        assert source["sequence_diagnostic"]["wait_reason"] == "FL_measured_space_before_FR_atomic_group"
        assert source["motion"].source_atomic_emitted == atomic_count
    actual = issue(p, contract, "P07", 93, space=True)
    assert source["ticks"] == 41 and source["sample"].tick_index == 40
    assert source["sequence_diagnostic"]["fr_group_start_tick"] == 93
    groups = source["sample"].atomic_groups
    assert len(groups) == 1 and {"front_right_knee", "front_right_ankle"} <= set(groups[0].channels)
    assert actual[3] != held[3] and actual[9] == -.63 and held[9] == 0.
    assert source["motion"].source_atomic_emitted == atomic_count + 1


@pytest.mark.parametrize("ready_kind", ["source_not_ended", "no_fr_response", "fr_not_support", "space_lost"])
def test_p08_needs_completed_p07_and_current_space_fr_response(contract, ready_kind):
    p = make_provider(contract)
    enter_fast_chain(p, contract)
    for tick in range(17, 190 if ready_kind == "source_not_ended" else 230):
        issue(p, contract, "P09", tick, space=True)
    source = layer(p, "P07")
    assert bool(source["sample"].endpoint_issued) == (ready_kind != "source_not_ended")
    tick = 230
    options = dict(space=True, fr_motion=3.)
    if ready_kind == "no_fr_response": options["fr_motion"] = 0.
    if ready_kind == "fr_not_support": options["supports"] = ("FL", "RL", "RR")
    if ready_kind == "space_lost": options["space"] = False
    issue(p, contract, "P09", tick, **options)
    assert layer(p, "P08")["ticks"] == 0


def start_p08_after_delayed_fr_response(p, contract):
    enter_fast_chain(p, contract)
    for tick in range(17, 230):
        issue(p, contract, "P09", tick, space=True)
    assert layer(p, "P07")["sample"].endpoint_issued and layer(p, "P08")["ticks"] == 0
    issue(p, contract, "P09", 230, space=True, fr_motion=3.)
    assist = layer(p, "P08")
    assert assist["ticks"] == 1 and assist["sample"].tick_index == 0
    assert assist["sample"].elapsed_s == 0. and not assist["sample"].endpoint_issued
    assert assist["sequence_diagnostic"]["actual_start_tick"] == 230
    assert layer(p, "P09")["ticks"] == 0


def test_delayed_p08_starts_at_local_zero_then_p09_waits_for_rl_and_rr_response(contract):
    p = make_provider(contract)
    start_p08_after_delayed_fr_response(p, contract)
    issue(p, contract, "P09", 231, space=True, fr_motion=3., rl_motion=3., rr_load=.1)
    assert layer(p, "P09")["ticks"] == 0  # Response alone cannot skip the P08 finite source.
    for tick in range(232, 280):
        issue(p, contract, "P09", tick, space=True, fr_motion=3.)
    assist, swing = layer(p, "P08"), layer(p, "P09")
    assert assist["sample"].endpoint_issued and swing["ticks"] == 0
    issue(p, contract, "P09", 280, space=True, fr_motion=3., rr_load=.1)
    assert swing["ticks"] == 0  # Unload alone does not manufacture RL response.
    issue(p, contract, "P09", 281, space=True, fr_motion=3., rl_motion=3.)
    assert swing["ticks"] == 0  # RL motion without RR unload is insufficient.
    issue(p, contract, "P09", 282, space=True, fr_motion=3., rl_motion=3., rr_load=.1, load_valid=False)
    assert swing["ticks"] == 0  # Unknown load is not zero.
    issue(p, contract, "P09", 283, space=True, fr_motion=3., rl_motion=3., rr_load=.1, supports=("FR", "RR"))
    assert swing["ticks"] == 0  # RR itself is not a second OTHER support.
    issue(p, contract, "P09", 284, space=True, fr_motion=3., rl_motion=3., rr_load=.1)
    assert swing["ticks"] == 1 and swing["sample"].tick_index == 0
    assert swing["sequence_diagnostic"]["actual_start_tick"] == 284


def test_current_valid_whole_body_rr_lift_continues_without_own_joint_motion(contract):
    p = make_provider(contract)
    enter_fast_chain(p, contract)
    before = current_input(contract, "P09", 17)[1]["joints"]
    item, raw = current_input(contract, "P09", 17, continuing=True)
    assert raw["joints"] == before  # No own RR or other joint displacement is invented.
    p.evaluate(item, raw)
    assert layer(p, "P08")["sample"].tick_index == layer(p, "P09")["sample"].tick_index == 0
    assert layer(p, "P08")["sequence_diagnostic"]["RR_current_continuation"]
    assert layer(p, "P09")["sequence_diagnostic"]["RR_current_continuation"]


def test_authored_p07_wheel_stop_owns_all_four_wheels_after_p06(contract):
    p = make_provider(contract, "P06")
    for tick in range(8):
        previous = issue(p, contract, "P06", tick)
    assert previous[8:] == (.3,) * 4
    assert issue(p, contract, "P07", 8)[8:] == previous[8:]  # FL can start; phase transition is not a wheel stop.
    stop_seen = False
    for tick in range(9, 190):
        actual = issue(p, contract, "P07", tick, space=True)
        current = layer(p, "P07")
        if current["sample"].tick_index == 40:
            assert actual[9] == -.63 and tuple(actual[i] for i in (8, 10, 11)) == (.3,) * 3
        if current["sample"].tick_index == 160:
            assert actual[8:] == (0.,) * 4
            assert set(range(8, 12)) <= current["touched"]
            assert len(current["sample"].atomic_groups) == 1
            assert set(current["sample"].atomic_groups[0].channels) == set(WHEEL_ORDER)
            stop_seen = True
        if current["sample"].tick_index > 160:
            assert actual[8:] == (0.,) * 4  # P06 cannot resurrect its old wheel owner.
    assert stop_seen


def test_old_mode_keeps_fast_layer_clocks_and_no_new_sequence_receipt(contract):
    old = make_provider(contract, enabled=False)
    enter_fast_chain(old, contract)
    for tick in range(17, 70):
        issue(old, contract, "P09", tick)
    assert [layer(old, phase)["ticks"] for phase in ("P07", "P08", "P09")] == [70, 62, 54]
    assert "source_partial_order" not in old.nominal_suggestion_diagnostics


def test_p09_extra_carry_feedback_cannot_override_p07_atomic_pulse_or_stop(contract):
    p = make_provider(contract)
    enter_fast_chain(p, contract)
    for tick in range(17, 180):
        item, raw = current_input(contract, "P09", tick, continuing=True)
        item["physical_evaluator"]["current_legs"]["RR"].update(
            front_distance_m=-.30, clearance_m=-.04)
        actual = p.evaluate(item, raw)
        local = layer(p, "P07")["sample"].tick_index
        if 40 <= local < 160:
            assert actual[9] == -.63
        elif local >= 160:
            assert actual[8:] == (0.,) * 4
        assert p.nominal_suggestion_diagnostics["rr_carry_continuation"]["ordered_source_wheel_owner"]


@pytest.mark.parametrize("phase", ["P01", "P02", "P03", "P04", "P05", "P06", "P10", "P11", "P12", "P13"])
def test_sequence_mode_does_not_change_unrelated_phase_paths(contract, phase):
    new, old = make_provider(contract, phase), make_provider(contract, phase, enabled=False)
    for tick in range(60):
        assert issue(new, contract, phase, tick) == issue(old, contract, phase, tick)
        assert new.tracking_servo_names == old.tracking_servo_names

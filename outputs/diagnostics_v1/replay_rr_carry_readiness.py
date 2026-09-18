"""Replay proposed readiness predicates on saved inputs only; never simulate holds."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "residual_RR_diagnosis.json").read_text(encoding="utf8"))
out = {
    "schema": "wlr50_clean.rr_carry_predicate_replay.v1",
    "scope": "independent original-input predicate checks, not counterfactual dynamics or new success",
    "production_changed": False,
    "existing_clearance_threshold_m": .015,
    "source_P09_knee_waypoint_times_s": [.6666666666666288, .7333333333332916,
        .7999999999999545, .8666666666666174, .9333333333332803, .9999999999999432],
    "source_P09_positive_fourwheel_atomic_time_s": 1.9333333333332234,
    "runs": {},
}
for name, start in (("CP177152", 4600), ("retained_zero", 5408)):
    run = data["runs"][name]
    rows = {r["tick"]: r for r in run["rows"]}
    checks = []
    for tick in range(start + 1, start + 234):
        previous, current = rows[tick - 1], rows[tick]
        knee_down = current["N"][7] < previous["N"][7] - 1e-9
        wheel_start = all(v == 0. for v in previous["N"][8:]) and all(v > 0. for v in current["N"][8:])
        if not (knee_down or wheel_start):
            continue
        saved = previous["same_tick_decision_diagnostic"]
        # Source events align to saved exact decision-end evaluator inputs.
        assert saved is not None, (name, tick)
        rr = saved["RR_current"]
        valid = rr["current_lift_valid"] is True
        airborne = previous["RR_air"]
        before_platform = not rr["within_top_xy"]
        falling = previous["RR_bottom_velocity_finite_difference_m_s"] < 0.
        knee_hold = bool(knee_down and valid and airborne and before_platform
                         and previous["RR_gap"] <= .015 and falling)
        new_roll_ready = bool(valid and airborne and not previous["RR_ground"])
        checks.append({
            "dispatch_tick": tick, "observation_tick": tick - 1,
            "source_elapsed_s": (tick - 1 - start) / 120,
            "kind": "negative_knee_waypoint" if knee_down else "positive_fourwheel_atomic_group",
            "old_N_RR_hip_knee": previous["N"][6:8], "new_N_RR_hip_knee": current["N"][6:8],
            "old_N_wheels": previous["N"][8:], "new_N_wheels": current["N"][8:],
            "RR_air": airborne, "RR_ground": previous["RR_ground"],
            "RR_current_lift_valid_original_exact_snapshot": valid,
            "RR_within_top_xy": rr["within_top_xy"],
            "RR_gap_m": previous["RR_gap"],
            "RR_bottom_vz_finite_difference_m_s": previous["RR_bottom_velocity_finite_difference_m_s"],
            "proposed_knee_waypoint_hold": knee_hold,
            "proposed_new_rolling_ready": new_roll_ready if wheel_start else None,
            "other_bearing_N": {leg: previous["bearing_by_leg"][leg] for leg in ("FL", "FR", "RL")},
        })
    holds = [r for r in checks if r["proposed_knee_waypoint_hold"]]
    wheels = [r for r in checks if r["kind"] == "positive_fourwheel_atomic_group"]
    assert len(wheels) == 1
    out["runs"][name] = {"sealed_source": run["source"], "P09_source_start_tick": start,
        "checks": checks, "first_would_hold_tick": holds[0]["dispatch_tick"] if holds else None,
        "knee_checks": sum(r["kind"] == "negative_knee_waypoint" for r in checks),
        "knee_holds_on_original_trace": len(holds),
        "positive_wheel_group_allowed_on_original_trace": wheels[0]["proposed_new_rolling_ready"]}
assert out["runs"]["retained_zero"]["knee_holds_on_original_trace"] == 0
assert out["runs"]["retained_zero"]["positive_wheel_group_allowed_on_original_trace"]
assert out["runs"]["CP177152"]["first_would_hold_tick"] == 4697
assert not out["runs"]["CP177152"]["positive_wheel_group_allowed_on_original_trace"]
with (HERE / "rr_carry_predicate_replay.json").open("x", encoding="utf8") as stream:
    json.dump(out, stream, indent=2, ensure_ascii=False)
print(json.dumps({name: {k: v for k, v in run.items() if k != "checks"}
                  for name, run in out["runs"].items()}, indent=2))

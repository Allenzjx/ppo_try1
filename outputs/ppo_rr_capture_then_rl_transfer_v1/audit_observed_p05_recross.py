"""Counterfactual guard check on the sealed P05 window, NOT new physics."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.ppo.semantic_supervisor import NominalMotionProvider, load_task_spec

SOURCE = ROOT / "runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation/20260923T0139060677019Z_g047e8a78be61_aeef4282d54c4e66bb6798e26d17d1d5/source"
DESTINATION = Path(__file__).with_name("P05_observed_window_guard_audit.json")


def main():
    provider = object.__new__(NominalMotionProvider)
    provider.spec = load_task_spec(ROOT / "configs/ppo_rr_capture_then_rl_transfer_v1/stage_task_spec.yaml")
    provider.physics_hz = 120.
    requested_ages = iter((30.1, 32., 35.4, 39.8))
    next_age = next(requested_ages)
    selected = []
    with (SOURCE / "video_policy_decisions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            task = row["step_info"]["semantic_task"]
            if task["stage_id"] != "P05" or task["stage_elapsed_s"] < next_age:
                continue
            old = task["nominal_provider_diagnostics"]["p05_preedge_approach_recovery"]
            assert not old["fresh_source_wheel_owners"], "Cannot substitute an absent owner for real fresh source events"
            assert old["source_tick_consecutive"] and old["source_endpoint_issued"]
            # This is only a pure predicate evaluation, not evaluate()/clock
            # advancement, mapper replay or an actuator command. These source
            # facts are copied from the sealed original diagnostic.
            provider._continuous_layers = []
            provider._approach_wheel_prior = tuple(old["wheel_prior_rad_s"])
            provider.endpoint_issued = old["source_endpoint_issued"]
            provider.elapsed_s = old["source_elapsed_s"]
            new = provider._p05_preedge_recovery_status(task,
                prior_source_endpoint_issued=old["prior_source_endpoint_issued"],
                prior_observation_tick=old["prior_source_observation_tick"])
            assert not old["eligible"] and old["reasons"] == ["not_crossed_or_placed_FL"]
            assert new["eligible"] and new["same_air_recross"]["branch"] == "same_air_recross"
            fl = task["physical_evaluator"]["current_legs"]["FL"]
            selected.append({"episode_tick": task["physical_evaluator"]["physics_tick"],
                "stage_elapsed_s": task["stage_elapsed_s"],
                "FL_gap_m": fl["clearance_m"], "FL_front_distance_m": fl["front_distance_m"],
                "FL_placed": task["placed_history"]["FL"],
                "old_rejected_by": old["reasons"], "new_predicate": new,
                "new_nominal_wheel_advice_if_executed_rad_s": new["wheel_prior_rad_s"],
                "new_advice_was_physically_executed_in_this_source": False})
            next_age = next(requested_ages, None)
            if next_age is None:
                break
    assert len(selected) == 4
    result = {"schema": "wlr50_clean.observed_p05_recross_guard_audit.v1",
        "source": str(SOURCE), "selected": selected,
        "new_physics_steps": 0, "policy_updates": 0,
        "counterfactual_nominal_permission_only": True,
        "policy_mapper_tracking_or_physical_success_proven": False}
    with DESTINATION.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(DESTINATION), "selected_real_ticks": [r["episode_tick"] for r in selected],
        "old_rejected_new_guard_eligible": len(selected), "new_physics_steps": 0}))


if __name__ == "__main__":
    main()

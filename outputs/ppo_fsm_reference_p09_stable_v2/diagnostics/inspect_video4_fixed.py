"""One fixed completed-video policy-ledger pass; no simulation/model loading."""
import json
import math
from collections import Counter
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260910T1310293487954Z_g7db0d17f398d_9f7a4dce1be54856b99818002f74a91d"
SOURCE = RUN / "source"
manifest = json.loads((SOURCE / "semantic_video_source_manifest.json").read_text(encoding="utf-8"))
caps = yaml.safe_load((ROOT / "configs/ppo_fsm_reference_p09_stable_v2/execution_profile.yaml").read_text(encoding="utf-8"))["residual"]["phase_caps_full12"]
counts = Counter()
contact = Counter()
after_cross_contact = Counter()
reasons = Counter()
transitions = []
end_ticks = []
native_ticks = []
native_verified = native_effect = own_effect = 0
residual_formula_max = 0.0
last_p05_rows_equal_formula = 0
no_writes = True
first_cross_decision = None
for line in (SOURCE / "video_policy_decisions.jsonl").open(encoding="utf-8"):
    row = json.loads(line)
    counts[row["request_phase"]] += 1
    assert row["decision"] == sum(counts.values())
    assert row["start_tick"] == (end_ticks[-1] if end_ticks else 0)
    assert row["end_tick"] - row["start_tick"] == row["physics_ticks"]
    end_ticks.append(row["end_tick"])
    info = row.get("step_info", {})
    task = info.get("semantic_task", {})
    if info:
        fl = task["physical_evaluator"]["current_legs"]["FL"]
        mode = "TOP" if fl["top_contact"] else "GROUND" if fl["ground_contact"] else "AIR" if fl["air"] else "OTHER"
        contact[mode] += 1
        if task["front_edge_crossed_history"]["FL"]:
            after_cross_contact[mode] += 1
            first_cross_decision = first_cross_decision or {"decision": row["decision"], "end_tick": row["end_tick"], "phase": row["request_phase"]}
        if info.get("termination_reason"):
            reasons[info["termination_reason"]] += 1
        transitions.extend(info.get("stage_transition_evidence", []))
        no_writes = no_writes and info.get("no_in_episode_state_writes_verified") is True
        for tick in info["actuator_target_effect_audit_ticks"]:
            native_ticks.append(tick["episode_physics_tick"])
            native_verified += tick["verified"] is True
            native_effect += tick["actual_native_effect"] is True
            own_effect += tick["own_phase_request_effect"] is True
        expected = [math.tanh(value)*cap for value, cap in zip(info["raw_policy_action_full12"], caps[row["request_phase"]])]
        difference = max(abs(a-b) for a,b in zip(expected, info["projected_residual_full12"]))
        residual_formula_max = max(residual_formula_max, difference)
        if row["request_phase"] == "P05" and difference < 1e-10:
            last_p05_rows_equal_formula += 1
    last = row

assert len(end_ticks) == 749 and end_ticks[-1] == 5988
assert native_ticks == list(range(1, 5989))
last_info = last["step_info"]
last_task = last_info["semantic_task"]
last_eval = last_task["physical_evaluator"]
native = last_info["actuator_target_effect_audit"]
quality = manifest["physical_episode"]["quality_metrics"]
out = {
    "run_id": RUN.name, "decisions": len(end_ticks), "phase_counts": dict(sorted(counts.items())),
    "native_tick_count": len(native_ticks), "native_ticks_contiguous_1_to_5988": True,
    "native_verified": native_verified, "native_effect": native_effect, "own_phase_native_effect": own_effect,
    "no_in_episode_state_writes": no_writes, "termination_reason_counts": dict(reasons),
    "fl_decision_endpoint_contact_counts": dict(contact), "fl_after_cross_decision_endpoint_contact_counts": dict(after_cross_contact),
    "fl_first_cross_decision_endpoint": first_cross_decision,
    "stage_transitions": [{k:t.get(k) for k in ("from_stage", "to_stage", "physics_tick", "simulation_time_s", "reason")} for t in transitions],
    "last_decision": {k:last.get(k) for k in ("decision", "request_phase", "start_tick", "end_tick", "physics_ticks", "environment_step_returned")},
    "last_task": {k:last_task.get(k) for k in ("stage_id", "completed_stage_ids", "termination_reason", "termination_source", "local_timeout", "stage_age_s", "completion_values", "entry_valid", "entry_reasons", "stall_diagnostic")},
    "last_FL": last_eval["current_legs"]["FL"],
    "last_history": last_eval["history"],
    "last_command": {k:last_info[k] for k in ("raw_policy_action_full12", "nominal_action_full12", "projected_residual_full12", "actual_drive_target_full12")},
    "last_native_audit": {k:native.get(k) for k in ("verified", "phase_mask_full12", "changed_target_channel_count", "native_drive_target_full12", "actual_mapping_matches_dispatch", "setter_dispatch_targets_equal", "same_tick_counterfactual")},
    "last_residual_equals_tanh_cap_max_error": max(abs(math.tanh(a)*b-c) for a,b,c in zip(last_info["raw_policy_action_full12"], caps["P05"], last_info["projected_residual_full12"])),
    "P05_decision_endpoints_residual_equals_tanh_cap": last_p05_rows_equal_formula,
    "all_decision_endpoint_max_difference_from_instantaneous_tanh_cap": residual_formula_max,
    "quality_partial_only": {k:quality["global"][k] for k in ("duration_s", "physics_ticks", "roll_rms_rad", "pitch_rms_rad", "roll_rate_rms_rad_s", "pitch_rate_rms_rad_s")},
    "all_phases_sampled": quality["all_phases_sampled"],
}
print(json.dumps(out, indent=2))

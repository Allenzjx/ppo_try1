"""One completed video5 ledger pass, scalar summaries only; no live-run access."""
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260910T1411052557957Z_g7db0d17f398d_fda5b92927ef45d7aa2b048e98015a8e"
SOURCE = RUN / "source"
manifest = json.loads((SOURCE / "semantic_video_source_manifest.json").read_text(encoding="utf-8"))
capture = json.loads((SOURCE / "viewport_buffer_video_manifest.json").read_text(encoding="utf-8"))
lifecycle = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
caps = yaml.safe_load((ROOT / "configs/ppo_fsm_reference_p09_stable_v2/execution_profile.yaml").read_text(encoding="utf-8"))["residual"]["phase_caps_full12"]
counts, contact, after_contact, reasons = Counter(), Counter(), Counter(), Counter()
transitions, end_ticks, native_ticks, post_cross = [], [], [], []
native_verified = native_effect = own_effect = all_masks_open = p05_formula_equal = 0
p05_formula_max = p05_final_composition_max = 0.0
no_writes = True
any_FL_placed = False
for line in (SOURCE / "video_policy_decisions.jsonl").open(encoding="utf-8"):
    row = json.loads(line)
    counts[row["request_phase"]] += 1
    assert row["decision"] == sum(counts.values())
    assert row["start_tick"] == (end_ticks[-1] if end_ticks else 0)
    assert row["end_tick"] - row["start_tick"] == row["physics_ticks"]
    end_ticks.append(row["end_tick"])
    info = row["step_info"]
    task = info["semantic_task"]
    ev = task["physical_evaluator"]
    fl = ev["current_legs"]["FL"]
    mode = "TOP" if fl["top_contact"] else "GROUND" if fl["ground_contact"] else "AIR" if fl["air"] else "OTHER"
    contact[mode] += 1
    any_FL_placed |= bool(task["placed_history"]["FL"])
    if task["front_edge_crossed_history"]["FL"]:
        after_contact[mode] += 1
        post_cross.append({"decision":row["decision"], "tick":row["end_tick"], "clearance_m":fl["clearance_m"],
                           "front_distance_m":fl["front_distance_m"], "bearing_force_n":fl["bearing_force_n"],
                           "support":fl["support"]})
    if info.get("termination_reason"):
        reasons[info["termination_reason"]] += 1
    transitions.extend(info.get("stage_transition_evidence", []))
    no_writes &= info.get("no_in_episode_state_writes_verified") is True
    for tick in info["actuator_target_effect_audit_ticks"]:
        native_ticks.append(tick["episode_physics_tick"])
        native_verified += tick["verified"] is True
        native_effect += tick["actual_native_effect"] is True
        own_effect += tick["own_phase_request_effect"] is True
    native = info["actuator_target_effect_audit"]
    all_masks_open += native["phase_mask_full12"] == [1]*12
    if row["request_phase"] == "P05":
        error = max(abs(math.tanh(a)*b-c) for a,b,c in zip(info["raw_policy_action_full12"], caps["P05"], info["projected_residual_full12"]))
        p05_formula_equal += error < 1e-10
        p05_formula_max = max(p05_formula_max, error)
        composition = max(abs(a+b-c) for a,b,c in zip(native["native_drive_target_full12"], info["projected_residual_full12"], info["actual_drive_target_full12"]))
        p05_final_composition_max = max(p05_final_composition_max, composition)
    last = row

assert len(end_ticks) == 800 and end_ticks[-1] == 6396
assert native_ticks == list(range(1,6397))
last_info, load = last["step_info"], manifest["checkpoint_load_provenance"]
task, native = last_info["semantic_task"], last_info["actuator_target_effect_audit"]
history = task["physical_evaluator"]["history"]
quality = manifest["physical_episode"]["quality_metrics"]
physical = manifest["physical_episode"]["physical_task_evaluation"]
out = {
  "checked_at_utc":datetime.now(timezone.utc).isoformat(), "run_id":RUN.name,
  "lifecycle":lifecycle["lifecycle"], "completed_at_utc":lifecycle["completed_at_utc"],
  "load":{k:load.get(k) for k in ("checkpoint_loaded_and_verified", "official_load_semantic_checkpoint", "saved_global_policy_decisions", "source", "migration", "observation_dimension", "video_seed", "optimizer_updates")},
  "natural_reset":manifest["natural_reset_proof"]["entry"],
  "reset_metadata":{k:manifest["natural_reset_proof"]["reset_metadata"].get(k) for k in ("reset_count", "reset_options", "reset_prime_tick_count", "training_phase_snapshot")},
  "deterministic_output":load["policy_contract"]["deterministic_output"],
  "decisions":len(end_ticks), "physics_ticks":len(native_ticks), "phase_counts":{f"P{i:02d}":counts[f"P{i:02d}"] for i in range(1,14)},
  "native_contiguous":True, "native_verified":native_verified, "native_effect":native_effect, "native_own_phase_effect":own_effect,
  "no_state_writes":no_writes, "terminal_reasons":dict(reasons),
  "FL_events":[event for event in history["lift_attempt_events"] if event["leg"]=="FL"],
  "FL_event_ticks":{kind:legs.get("FL") for kind,legs in history["event_ticks"].items()},
  "any_FL_placed":any_FL_placed,
  "FL_endpoint_contact":dict(contact), "FL_postC_endpoint_contact":dict(after_contact),
  "FL_postC_first":post_cross[0], "FL_postC_min_clearance":min(post_cross,key=lambda v:v["clearance_m"]),
  "FL_postC_max_clearance":max(post_cross,key=lambda v:v["clearance_m"]),
  "FL_postC_bearing_max_n":max(v["bearing_force_n"] for v in post_cross),
  "FL_terminal":task["physical_evaluator"]["current_legs"]["FL"],
  "last_task":{k:task.get(k) for k in ("stage_id", "completed_stage_ids", "termination_reason", "termination_source", "local_timeout", "stage_age_s", "completion_values", "stall_diagnostic")},
  "transitions":[{k:t.get(k) for k in ("from_stage", "to_stage", "physics_tick")} for t in transitions],
  "last_decision":{k:last[k] for k in ("decision", "start_tick", "end_tick", "physics_ticks", "environment_step_returned")},
  "last_commands":{k:last_info[k] for k in ("raw_policy_action_full12", "nominal_action_full12", "projected_residual_full12", "actual_drive_target_full12")},
  "last_post_mapper_base_full12":native["native_drive_target_full12"],
  "open_12_channel_mask_decision_endpoints":all_masks_open,
  "P05_exact_tanh_cap_endpoint_count":p05_formula_equal, "P05_tanh_cap_max_error":p05_formula_max,
  "P05_post_mapper_plus_residual_vs_final_target_max_error":p05_final_composition_max,
  "physical_status":{k:physical.get(k) for k in ("valid", "run_validity", "physical_evidence_status", "success", "termination_reason", "termination_source")},
  "partial_quality":{k:quality["global"][k] for k in ("duration_s", "physics_ticks", "roll_rms_rad", "pitch_rms_rad", "roll_rate_rms_rad_s", "pitch_rate_rms_rad_s")},
  "all_phases_sampled":quality["all_phases_sampled"],
  "capture_metadata":{k:capture.get(k) for k in ("valid", "frame_count", "frame_ledger_complete", "one_callback_per_render", "encoder_finalized_before_app_close")},
  "recorded_decode_metadata_not_rerun":{k:capture["full_decode"].get(k) for k in ("valid", "full_decode", "bytes", "sha256", "duration_s", "container_duration_s", "container_duration_valid", "fps", "frame_count", "unique_frame_checksums", "black_like_frame_count")},
  "task_interval_window":{k:manifest["task_interval_window"].get(k) for k in ("encoded_duration_s", "terminal_frame_display_quantization_s")},
  "source_acceptance_error":manifest["source_acceptance_error"],
  "scope":"One completed policy-ledger scan plus small source/capture/run metadata; no physical/native stream scan, model hash, replay, remux or active-N0 access. Initial checkpoint-load hash is not an end-of-evaluation unchanged-model proof."
}
print(json.dumps(out,indent=2))

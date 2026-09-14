"""Bounded one-pass video6 diagnostic: only completed policy ledger and small manifests."""
from __future__ import annotations
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260910T1528556872035Z_g7db0d17f398d_a5db22d8e4b545969459f6ea059a6c87"
SOURCE = RUN / "source"
OUT = ROOT / "outputs/ppo_fsm_reference_p09_stable_v2/video6_151936_p06_diagnosis.json"
manifest = json.loads((SOURCE / "semantic_video_source_manifest.json").read_text(encoding="utf-8"))
capture = json.loads((SOURCE / "viewport_buffer_video_manifest.json").read_text(encoding="utf-8"))
lifecycle = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
caps = yaml.safe_load((ROOT / "configs/ppo_fsm_reference_p09_stable_v2/execution_profile.yaml").read_text(encoding="utf-8"))["residual"]["phase_caps_full12"]
counts, native, masks, formula_counts = Counter(), Counter(), Counter(), Counter()
formula_max, residual_absmax = Counter(), {}
states, transitions, bridges = [], [], []
last_tick = 0
no_writes = True
leg_fields = ("air","top_contact","ground_contact","support","contact_mode","surface_class",
              "bearing_verified","bearing_force_n","load_fraction","load_fraction_valid","front_distance_m",
              "clearance_m","ground_relative_lift_m","initial_lift_observed","lift_established",
              "current_lift_valid","motion_continuation_ready","lift_evidence_invalid_reason",
              "body_control_evidence","within_top_region","consecutive_air_ticks","air_duration_s")
for line in (SOURCE / "video_policy_decisions.jsonl").open(encoding="utf-8"):
    row = json.loads(line)
    phase = row["request_phase"]
    counts[phase] += 1
    assert row["decision"] == sum(counts.values())
    assert row["start_tick"] == last_tick
    assert row["end_tick"]-last_tick == row["physics_ticks"]
    last_tick = row["end_tick"]
    info = row["step_info"]
    task = info["semantic_task"]
    ev = task["physical_evaluator"]
    no_writes &= info.get("no_in_episode_state_writes_verified") is True
    for tick in info["actuator_target_effect_audit_ticks"]:
        native["ticks"] += 1
        assert tick["episode_physics_tick"] == native["ticks"]
        native["verified"] += tick["verified"] is True
        native["effect"] += tick["actual_native_effect"] is True
        native["own_phase_effect"] += tick["own_phase_request_effect"] is True
    a = info["actuator_target_effect_audit"]
    masks[phase] += a["phase_mask_full12"] == [1]*12
    raw = info["raw_policy_action_full12"]
    projected = info["projected_residual_full12"]
    err = max(abs(math.tanh(x)*cap-value) for x,cap,value in zip(raw,caps[phase],projected))
    formula_max[phase] = max(formula_max[phase],err)
    formula_counts[phase] += err < 1e-10
    residual_absmax[phase] = [max(old,abs(new)) for old,new in zip(residual_absmax.get(phase,[0.0]*12),projected)]
    compact = {"decision":row["decision"],"tick":row["end_tick"],"time_s":row["end_tick"]/120,
       "request_phase":phase,"returned_phase":task["stage_id"],
       "terminal":info.get("termination_reason"),
       "physical_evidence_status":ev.get("physical_evidence_status"),
       "FL":{k:ev["current_legs"]["FL"].get(k) for k in leg_fields},
       "RR":{k:ev["current_legs"]["RR"].get(k) for k in leg_fields},
       "FL_placed_history":task["placed_history"]["FL"],
       "RR_placed_history":task["placed_history"]["RR"],
       "FL_C_history":task["front_edge_crossed_history"]["FL"],
       "RR_C_history":task["front_edge_crossed_history"]["RR"],
       "commands":{k:info[k] for k in ("nominal_action_full12","projected_residual_full12","actual_drive_target_full12")},
       "post_mapper_base_full12":a["native_drive_target_full12"],
       "mask":a["phase_mask_full12"]}
    states.append(compact)
    transitions.extend(info.get("stage_transition_evidence",[]))
    for key,value in info.items():
        if "bridge" in key or "handoff" in key:
            if value:
                bridges.append({"decision":row["decision"],"tick":row["end_tick"],"key":key,"value":value})
    last = row
assert len(states)==1176 and last_tick==9408 and native["ticks"]==9408
info = last["step_info"]
task = info["semantic_task"]
history = task["physical_evaluator"]["history"]
events = [e for e in history["lift_attempt_events"] if e["leg"] in ("FL","RR")]
def mode(s,leg):
    v=s[leg]
    return (v["contact_mode"],v["air"],v["top_contact"],v["ground_contact"],v["support"],
            v["bearing_verified"],v["load_fraction_valid"],v["current_lift_valid"])
mode_changes = {}
for leg in ("FL","RR"):
    mode_changes[leg] = [s for i,s in enumerate(states) if i==0 or mode(s,leg)!=mode(states[i-1],leg)]
selected_ticks = {states[0]["tick"],states[-1]["tick"]}
for e in events:
    if isinstance(e.get("physics_tick"),(float,int)):
        t=e["physics_tick"]
        for s in states:
            if s["tick"]>=t:
                selected_ticks.add(s["tick"])
                if s["decision"]>1: selected_ticks.add(states[s["decision"]-2]["tick"])
                break
for kind,legs in history["event_ticks"].items():
    for leg in ("FL","RR"):
        t=legs.get(leg)
        if isinstance(t,(float,int)):
            for s in states:
                if s["tick"]>=t:
                    selected_ticks.add(s["tick"])
                    if s["decision"]>1: selected_ticks.add(states[s["decision"]-2]["tick"])
                    break
for s in states:
    if s["returned_phase"]!=s["request_phase"]:
        selected_ticks.add(s["tick"])
        if s["decision"]<len(states): selected_ticks.add(states[s["decision"]]["tick"])
p06=[s for s in states if s["request_phase"]=="P06"]
postP=[s for s in states if s["FL_placed_history"]]
quality=manifest["physical_episode"]["quality_metrics"]
physical=manifest["physical_episode"]["physical_task_evaluation"]
out = {
 "schema":"wlr50_clean.fixed_video6_diagnostic.v1","checked_at_utc":datetime.now(timezone.utc).isoformat(),
 "run_id":RUN.name,"lifecycle":lifecycle["lifecycle"],"completed_at_utc":lifecycle["completed_at_utc"],
 "physical_task_success":manifest["physical_task_success"],"diagnostic_only":manifest["diagnostic_only"],
 "optimizer_updates":manifest["optimizer_updates"],
 "load":manifest["checkpoint_load_provenance"],
 "natural_reset":manifest["natural_reset_proof"],
 "entry_scope":{k:manifest.get(k) for k in ("from_phase","pre_action_ticks","pre_action_source","extra_pre_action_physics_ticks","episode_count","fresh_process_single_episode","performed_post_success_ticks")},
 "decisions":len(states),"physics_ticks":last_tick,"duration_s":last_tick/120,
 "phase_counts":{f"P{i:02d}":counts[f"P{i:02d}"] for i in range(1,14)},
 "native_contiguous":True,"native":dict(native),"no_in_episode_state_writes":no_writes,
 "open_12_channel_mask_counts":dict(masks),"tanh_cap_formula_equal_counts":dict(formula_counts),
 "tanh_cap_max_absolute_error":dict(formula_max),"residual_absmax_full12_byphase":residual_absmax,
 "leg_events":events,"event_ticks":{k:{leg:v.get(leg) for leg in ("FL","RR")} for k,v in history["event_ticks"].items()},
 "leg_mode_changes":mode_changes,
 "selected_event_decision_endpoints":[s for s in states if s["tick"] in selected_ticks],
 "P06_first":p06[0],"P06_last":p06[-1],
 "P06_leg_mode_counts":{leg:dict(Counter(str(mode(s,leg)) for s in p06)) for leg in ("FL","RR")},
 "post_FL_placement_first":postP[0] if postP else None,
 "stage_transitions":transitions,
 "bridge_records":bridges,
 "terminal_legs":{leg:task["physical_evaluator"]["current_legs"][leg] for leg in ("FL","RR")},
 "terminal_task":{k:task.get(k) for k in ("stage_id","completed_stage_ids","termination_reason","termination_source","local_timeout","stage_age_s","completion_values","stall_diagnostic")},
 "terminal_info":{k:info.get(k) for k in ("termination_reason","done","terminated","truncated","time_outs","terminal_bootstrap_allowed")},
 "physical_status":{k:physical.get(k) for k in ("valid","run_validity","physical_evidence_status","success","task_violations","safety_aborts")},
 "partial_quality":{k:quality["global"].get(k) for k in ("duration_s","physics_ticks","roll_rms_rad","pitch_rms_rad","roll_rate_rms_rad_s","pitch_rate_rms_rad_s")},
 "all_phases_sampled":quality["all_phases_sampled"],
 "capture":{k:capture.get(k) for k in ("valid","status","frame_count","frame_ledger_complete","one_callback_per_render","encoder_finalized_before_app_close","render_observer_only")},
 "recorded_decode_metadata":capture["full_decode"],
 "source_acceptance_error":manifest["source_acceptance_error"],
 "limits":["One completed policy ledger pass; no separate physical/native large-stream scan or model-byte hash.",
           "Decision endpoints are sampled at 8 physics ticks; event ticks can be exact history without exact-tick force telemetry.",
           "Initial load parameter hashes are not end-of-evaluation unchanged-model proof; failure at physical success requirement precedes check_model().",
           "Partial trajectory quality is not successful full-course stability, and unknown bearing is not BODY_COLLISION or wheel-only evidence."]
}
with OUT.open("x",encoding="utf-8") as f: json.dump(out,f,indent=2)
print(json.dumps({"output":str(OUT),"phase_counts":out["phase_counts"],"native":out["native"],
 "events":events,"event_ticks":out["event_ticks"],"mode_change_count":{k:len(v) for k,v in mode_changes.items()},
 "terminal_task":out["terminal_task"],"terminal_legs":out["terminal_legs"],
 "P06_first":out["P06_first"],"P06_last":out["P06_last"],
 "partial_quality":out["partial_quality"],"physical_status":out["physical_status"],
 "bridge_record_count":len(bridges)},indent=2))


"""Sealed P05 baseline facts, no policy execution, mapper replay or simulator."""
import json
from pathlib import Path
from rr_probe_readonly import ROOT, lines, stats, compact_physical
from wlr50_clean.reference.motion_contract import load_motion_contract
from wlr50_clean.fsm.state_spec import load_fsm_spec
from wlr50_clean.fsm.motion_executor import MotionExecutor

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T0801290800396Z_gee5a9651591d_a6d92def5d74491580cccd936a1edf18/source"
ds = list(lines(SOURCE / "video_policy_decisions.jsonl"))
by_end = {d["end_tick"]: d for d in ds}
ns = [r for r in lines(SOURCE / "native_tick_audit.jsonl") if r["source_phase_id"] == "P05"]
contract = load_motion_contract(ROOT / "configs/recording_motion_contract.json")
phase = contract.phase("P05")
state = load_fsm_spec(ROOT / "configs/fsm_states.yaml").state("P05")
assert state.normal_correction_domain == "none" and state.normal_time_scale == 1.0
# Reproduce only the finite source clock, NEVER a mapper or a nominal counterfactual.
clock = MotionExecutor(physics_hz=contract.physics_hz, servo_rate_limit_deg_s=contract.servo_rate_limit_deg_s)
clock.start_phase(phase, time_scale=state.normal_time_scale)
for source_index in range(2000):
    if clock.tick().endpoint_issued:
        break
else:
    raise AssertionError("No finite P05 source endpoint")
first_source_tick = ns[0]["episode_physics_tick"]
endpoint_tick = first_source_tick + source_index
assert [r["episode_physics_tick"] for r in ns] == list(range(first_source_tick, ds[-1]["end_tick"] + 1))
def eligible(d):
    task = d["step_info"]["semantic_task"]; ev = task["physical_evaluator"]; fl = ev["current_legs"]["FL"]
    return (d["end_tick"] >= endpoint_tick and task["stage_id"] == "P05" and ev["valid"]
            and not ev["termination_reason"] and ev["history"]["front_edge_crossed"]["FL"]
            and not ev["history"]["placed"]["FL"] and fl["air"] and fl["within_top_xy"])
entry = next(d["end_tick"] for d in ds if eligible(d))
ticks = {entry, 6200, ds[-1]["end_tick"]}
raw = {r["physics_tick"]: r for r in lines(SOURCE / "physical_observations.jsonl") if r["physics_tick"] >= entry}
height = {r["physics_tick"]: r for r in lines(SOURCE / "height_diagnostics.jsonl") if r["physics_tick"] in ticks}
startup = json.loads((SOURCE / "height_diagnostics_startup.json").read_text())
names = startup["joint_names_native_order"]["value"]
wheel_names = ["front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle"]
samples = []
for n in ns:
    tick = n["episode_physics_tick"]
    if tick not in ticks:
        continue
    audit = n["native_audit"]; head = audit["policy_headroom_evidence"]; r = raw[tick]
    d = by_end[tick]; ev = d["step_info"]["semantic_task"]["physical_evaluator"]; compact = compact_physical(r)
    qd = height[tick]["joint_velocity_native_rad_s"]["value"]
    if len(qd) == 1 and isinstance(qd[0], list):
        qd = qd[0]
    assert len(qd) == len(names)
    assert audit["setter_dispatch_targets_equal"] and audit["actual_mapping_matches_dispatch"]
    samples.append(dict(tick=tick, simulation_time_s=r["simulation_time_s"], native_command_clock_tick=audit["physics_tick"],
        source_N_FL_hip_knee_deg=n["nominal_full12"][:2],
        mapped_baseline_FL_hip_knee_deg=head["baseline_native_plus_controller_full12"][:2],
        REQUEST_FL_hip_knee_deg=n["projected_residual_full12"][:2],
        effective_FL_hip_knee_deg=head["effective_policy_residual_full12"][:2],
        final_FL_hip_knee_deg=[r["joints"][name]["command_deg"] for name in ("front_left_hip", "front_left_knee")],
        actual_FL_hip_knee_deg=[r["joints"][name]["position_deg"] for name in ("front_left_hip", "front_left_knee")],
        actual_FL_hip_knee_velocity_deg_s=[r["joints"][name]["velocity_deg_s"] for name in ("front_left_hip", "front_left_knee")],
        residual_mask=audit["phase_mask_full12"], clipped_servo_indices=head["clipped_servo_indices"],
        source_N_four_wheels_rad_s=n["nominal_full12"][8:],
        final_four_wheels_canonical_rad_s=[r["wheels"][name]["command_rad_s"] for name in wheel_names],
        final_four_wheels_native_rad_s=audit["actual_native_targets"]["wheel_velocity_rad_s"],
        measured_four_wheels_native_rad_s=[qd[names.index(name)] for name in wheel_names],
        actual_target_source=audit["actual_target_source"], verified_dispatch=True,
        FL_current={k:ev["current_legs"]["FL"][k] for k in ("air", "within_top_xy", "clearance_m", "front_distance_m", "bearing_force_n", "contact_surface")},
        body_collider_minimum_world_z_m=height[tick]["body_collision_minimum_z_w_m"],
        conservative_body_obstacle_AABB_separation_m=compact["separation"]))
post = [n for n in ns if n["episode_physics_tick"] >= entry]
end = ds[-1]["step_info"]; final_ev = end["semantic_task"]["physical_evaluator"]
result = dict(schema="CP189952.sealed_P05_baseline_readonly.v1", source=str(SOURCE),
    outcome=dict(final_tick=ds[-1]["end_tick"], semantic_termination=end["termination_reason"],
        semantic_termination_source=end["semantic_task"]["termination_source"],
        physical_valid=final_ev["valid"], physical_termination=final_ev["termination_reason"],
        physical_success=final_ev["success"], FL_events={k:v.get("FL") for k,v in final_ev["history"]["event_ticks"].items()}),
    finite_trigger=dict(direct_endpoint_flag_logged=None, first_P05_source_tick=first_source_tick,
        endpoint_source_index=source_index, inferred_first_endpoint_physics_tick=endpoint_tick,
        first_decision_boundary_with_all_predicates=entry,
        provenance="P05 has no sequence pause; contiguous native source ticks + unchanged finite source MotionExecutor clock, confirmed source final FL22.8/-13.4 appears at endpoint. Not a logged endpoint flag or mapper replay.",
        last_prior_decision_tick=max(t for t in by_end if t < entry)),
    axis_semantics=dict(wheel_order=wheel_names, canonical_to_native_wheel_sign=[-1,1,-1,1],
        observations="post-physics exact episode tick; native command clock separately retained (+179); native actuator buffers are targets, not measured velocities"),
    same_dispatch_samples=samples,
    entry_to_terminal=dict(ticks=[entry,ds[-1]["end_tick"]],
        REQUEST_FL_hip_deg=stats(n["projected_residual_full12"][0] for n in post),
        effective_FL_hip_deg=stats(n["native_audit"]["policy_headroom_evidence"]["effective_policy_residual_full12"][0] for n in post),
        negative_REQUEST_FL_hip_ticks=sum(n["projected_residual_full12"][0]<0 for n in post),
        FL_hip_headroom_clip_ticks=sum(0 in n["native_audit"]["policy_headroom_evidence"]["clipped_servo_indices"] for n in post),
        actual_FL_hip_deg=stats(r["joints"]["front_left_hip"]["position_deg"] for r in raw.values()),
        actual_FL_hip_velocity_deg_s=stats(r["joints"]["front_left_hip"]["velocity_deg_s"] for r in raw.values()),
        FL_gap_m=stats(compact_physical(r)["gap"]["FL"] for r in raw.values())),
    limitations="No FL-minus3 intervention ran in this baseline. Positive hip residual is distinct from negative measured hip velocity during source lowering. No separate nominal replay was subtracted. No causal claim for capture or future intervention; termination null is not success.")
target = OUT / "CP189952_P05_readonly.json"
with target.open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps(result, ensure_ascii=False, indent=2))

"""One fixed completed real window; no policy forward, optimizer or physics."""
from pathlib import Path
import itertools
import json

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / "runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/20260922T1521094182530Z_g336b7c56d2f0_3cdf838115e44829b66ad9585ad92ba7/source"
NEW = ROOT / "runs/ppo_p05_hip_only_continuation_v1/validation/20260922T1641475169869Z_g6ac7b553d792_13764630c7a0468285bacb6aa3607ee8"
FIRST, LAST = 5992, 6160
LEGS = ("FL", "FR", "RL", "RR")
def window(path, first_line, stop_line, key):
    rows = {}
    with path.open("rb") as stream:
        for line in itertools.islice(stream, first_line, stop_line):
            if not line.endswith(b"\n"): break
            row = json.loads(line); rows[row[key]] = row
    return rows
old_native = window(OLD / "native_tick_audit.jsonl", FIRST-1, LAST, "episode_physics_tick")
new_native = window(NEW / "native_tick_audit.jsonl", FIRST-1, LAST, "episode_physics_tick")
assert sorted(old_native) == sorted(new_native) == list(range(FIRST,LAST+1))
old_phys = window(OLD / "physical_observations.jsonl", FIRST, LAST+1, "physics_tick")
new_phys = window(NEW / "physical_observations.jsonl", FIRST, LAST+1, "physics_tick")
new_dec = window(NEW / "residual_and_projection_audit.jsonl", FIRST//8-1, LAST//8, "physics_tick")
assert sorted(new_dec) == list(range(FIRST, LAST+1, 8))
old_dec = {}
with (OLD / "video_policy_decisions.jsonl").open("rb") as stream:
    for line in itertools.islice(stream, FIRST//8-1, LAST//8):
        row = json.loads(line)["step_info"]; old_dec[row["physics_tick"]] = row

def maximum_delta(a,b):
    return max(abs(x-y) for x,y in zip(a,b,strict=True))
def first_difference(extract):
    return next((tick for tick in new_native if extract(new_native[tick]) != extract(old_native[tick])), None)
def physical(tick):
    row = new_phys.get(tick)
    if row is None: return dict(status="not_yet_available", measured_canonical_qd=None, final_canonical_target=None)
    return dict(status="saved_actual_measurement", measured_canonical_qd=row["actual_full12"][8:],
        final_canonical_target=row["commanded_full12"][8:],
        body_position_w_m=row["base"]["position_w_m"],
        body_linear_velocity_w_m_s=row["base"]["linear_velocity_w_m_s"])

checks = dict(mask_all12_one=True, native_audit_verified=True, setter_and_mapping_verified=True)
for tick,row in new_native.items():
    audit=row["native_audit"]
    checks["mask_all12_one"] &= audit["phase_mask_full12"] == [1]*12
    checks["native_audit_verified"] &= audit["verified"] is True
    checks["setter_and_mapping_verified"] &= audit["setter_dispatch_targets_equal"] is True and audit["actual_mapping_matches_dispatch"] is True
assert all(checks.values())
snapshots=[]
for tick in (5992,5999,6000,6001,6008,6144,6160):
    row=new_native[tick];audit=row["native_audit"];p=physical(tick)
    canonical=[a+b for a,b in zip(audit["native_drive_target_full12"][8:],audit["combined_post_mapper_bias_full12"][8:],strict=True)]
    if p["status"] == "saved_actual_measurement":
        assert maximum_delta(canonical,p["final_canonical_target"]) < 1e-12
    native=audit["actual_native_targets"]["wheel_velocity_rad_s"]
    assert maximum_delta(native,[-canonical[0],canonical[1],-canonical[2],canonical[3]]) < 1e-7
    snapshot=dict(tick=tick,sim_time_s=tick/120.,phase=row["source_phase_id"],
        original_P05_source_held_wheels=[0.,0.,0.,0.],
        actual_nominal_input_wheels=row["nominal_full12"][8:],
        actual_mapped_nominal_wheels=audit["native_drive_target_full12"][8:],
        raw_Gaussian_policy_wheels=audit["raw_policy_action_full12"][8:],
        requested_residual_rad_s=row["projected_residual_full12"][8:],
        residual_mask=audit["phase_mask_full12"][8:],mask_object="PPO residual permission, not nominal or final-target mask",
        effective_same_dispatch_residual_rad_s=audit["combined_post_mapper_bias_full12"][8:],
        canonical_final_target_from_same_dispatch=canonical,
        canonical_target_source="matched saved physical commanded_full12" if p["status"]=="saved_actual_measurement" else "same audited mapped nominal plus combined bias, not measured velocity",
        native_final_joint_targets=native,native_joint_ids=None,native_measured_qd=None,
        native_axis_relation_to_canonical=[-1,1,-1,1],
        last_writer=audit["actual_target_source"],measurement=p,
        nominal_owner="finite_P05_preedge_recovery_advice" if tick>=6001 else "original_P05_held_explicit_stop",
        source_owner_evidence="nominal input plus saved decision diagnostics and production schedule; no per-tick atomic-owner ID serialized")
    if tick in new_dec:
        task=new_dec[tick]["semantic_task"];fl=task["physical_evaluator"]["current_legs"]["FL"]
        snapshot.update(FL_front_m=fl["front_distance_m"],FL_gap_m=fl["clearance_m"],
            FL_air=fl["air"],FL_crossed=task["front_edge_crossed_history"]["FL"],
            FL_placed=task["placed_history"]["FL"],
            contacts={leg:{k:task["physical_evaluator"]["current_legs"][leg].get(k) for k in
                ("air","ground_contact","top_surface_contact","support","bearing_verified")} for leg in LEGS},
            after_frame_recovery_diagnostic=task["nominal_provider_diagnostics"]["p05_preedge_approach_recovery"])
        assert maximum_delta(canonical,new_dec[tick]["actual_drive_target_full12"][8:]) < 1e-12
    snapshots.append(snapshot)

def progress(dec,phys):
    before,after=(dec[t]["semantic_task"]["physical_evaluator"]["current_legs"]["FL"] for t in (6000,6160))
    body_delta=None
    if all(t in phys for t in (6000,6160)):
        body_delta=[b-a for a,b in zip(phys[6000]["base"]["position_w_m"],phys[6160]["base"]["position_w_m"],strict=True)]
    return dict(start_tick=6000,end_tick=6160,FL_front_start_m=before["front_distance_m"],
        FL_front_end_m=after["front_distance_m"],
        FL_front_change_m=after["front_distance_m"]-before["front_distance_m"],
        FL_gap_start_m=before["clearance_m"],FL_gap_end_m=after["clearance_m"],
        body_position_change_m=body_delta)
gates={tick:row["semantic_task"]["nominal_provider_diagnostics"]["p05_preedge_approach_recovery"] for tick,row in new_dec.items()}
report=dict(schema="wlr50_clean.p05_preedge_actual_activation.v1",old_source=str(OLD),new_source=str(NEW),
    fixed_ticks=[FIRST,LAST],tick_count=len(new_native),canonical_wheel_order=LEGS,wheel_units="rad/s except unitless raw Gaussian",
    measured_new_ticks_available=len(new_phys),measured_old_ticks_available=len(old_phys),
    first_saved_after_frame_recovery_eligible_tick=next(t for t,v in gates.items() if v["eligible"]),
    first_different_actual_dispatch_nominal_tick=first_difference(lambda r:r["nominal_full12"][8:]),
    first_different_mapped_nominal_tick=first_difference(lambda r:r["native_audit"]["native_drive_target_full12"][8:]),
    first_different_raw_policy_tick=first_difference(lambda r:r["native_audit"]["raw_policy_action_full12"]),
    first_different_REQUEST_tick=first_difference(lambda r:r["projected_residual_full12"]),
    first_different_native_final_target_tick=first_difference(lambda r:r["native_audit"]["actual_native_targets"]["wheel_velocity_rad_s"]),
    first_different_measured_actual_full12_tick=next((t for t in new_phys if t in old_phys and new_phys[t]["actual_full12"]!=old_phys[t]["actual_full12"]),None),
    gate_eligible_decision_endpoints=sum(v["eligible"] for v in gates.values()),
    total_decision_endpoints=len(gates),fresh_wheel_owners_at_saved_decision_endpoints={str(t):v["fresh_source_wheel_owners"] for t,v in gates.items()},
    checks=checks,new_progress=progress(new_dec,new_phys),old_progress=progress(old_dec,old_phys),
    snapshots=snapshots,new_optimizer_updates_in_this_eval=0,
    attribution="new finite recovery nominal causes the first control difference; later frozen-policy closed-loop response is not new PPO learning",
    limits=["Fixed completed window only; no task-success or long-term stability claim.",
        "Targets are separately identified from measured angular velocities; numeric native IDs and native measured velocities are not in these saved rows.",
        "A changed raw request after recovery may reflect the changed observation with unchanged weights, not an optimizer update."])
print(json.dumps(report,indent=2,allow_nan=False))

"""Bounded read-only P06 audit; writes only a new output JSON, never runs Isaac."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from rr_probe_readonly import lines, stats, servo_limits_deg


def variation(values):
    values = list(values)
    delta = [b-a for a, b in zip(values, values[1:])]
    signs = [1 if x > 0 else -1 for x in delta if abs(x) >= .05]
    return dict(**stats(values), net_change=values[-1]-values[0],
                sampled_total_variation=sum(map(abs, delta)),
                reversals_excluding_steps_below_0_05_deg=sum(a != b for a, b in zip(signs, signs[1:])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for row in lines(args.run / "residual_and_projection_audit.jsonl"):
        tick = row["applied_audit"]["physics_tick"]
        if tick > 4000:
            break
        if tick >= 2800:
            rows.append(row)
        if tick == 4000:
            break
    samples = [r["applied_audit"] for r in rows]
    assert [s["physics_tick"] for s in samples] == list(range(2800, 4001, 8))
    assert all(s["phase_id"] == s["end_phase_id"] == "P06" for s in samples)
    actions = samples[1:]
    physical = [s["semantic_task"]["physical_evaluator"] for s in samples]
    natives = [s["actuator_target_effect_audit"] for s in actions]
    heads = [n["policy_headroom_evidence"] for n in natives]
    nominal = [s["semantic_task"]["nominal_provider_diagnostics"] for s in actions]
    step_audit = [t for s in actions for t in s["actuator_target_effect_audit_ticks"]]
    assert [t["episode_physics_tick"] for t in step_audit] == list(range(2801, 4001))
    wheel_order = ["FL", "FR", "RL", "RR"]
    assert len(set(tuple(e["stop_progress_wheel_order"]) for e in physical)) == 1
    wheels = {}
    for i, leg in enumerate(wheel_order):
        j = 8+i
        final = [s["actual_drive_target_full12"][j] for s in actions]
        actual = [e["measured_wheel_velocity_rad_s"][i] for e in physical[1:]]
        effective = [h["effective_policy_residual_full12"][j] for h in heads]
        mapped = [h["baseline_native_plus_controller_full12"][j] for h in heads]
        native_target = [n["actual_native_targets"]["wheel_velocity_rad_s"][i] for n in natives]
        sign = (-1, 1, -1, 1)[i]
        wheels[leg] = dict(canonical_channel=j, native_forward_sign=sign,
            source_P06_rolling_rad_s=stats(d["p06_wheel_tail"]["source_rolling_rad_s"][i] for d in nominal),
            nominal_rad_s=stats(s["nominal_action_full12"][j] for s in actions),
            same_dispatch_mapped_baseline_rad_s=stats(mapped),
            policy_raw_unitless=stats(r["raw_policy_action_full12"][j] for r in rows[1:]),
            residual_permission_mask_values=sorted(set(n["phase_mask_full12"][j] for n in natives)),
            projected_REQUEST_rad_s=stats(s["projected_residual_full12"][j] for s in actions),
            effective_same_dispatch_residual_rad_s=stats(effective),
            final_canonical_target_rad_s=stats(final),
            native_target_buffer_rad_s=stats(native_target),
            measured_canonical_velocity_rad_s=stats(actual),
            measured_raw_native_velocity_rad_s=None,
            raw_native_absence_reason="Training audit logs canonical physical readback only; native target buffer is not measured qd.",
            zero_final_target_count=sum(abs(x) <= 1e-6 for x in final),
            reverse_final_target_count=sum(x < -1e-6 for x in final),
            near_zero_actual_count_0_01_rad_s=sum(abs(x) < .01 for x in actual),
            reverse_actual_count=sum(x < -1e-6 for x in actual),
            cancellation_below_half_nominal_count=sum(x < .15 for x in final),
            final_minus_mapped_minus_effective_max_abs=max(abs(x-b-r) for x, b, r in zip(final, mapped, effective)),
            target_native_sign_mapping_max_abs=max(abs(x-sign*y) for x, y in zip(native_target, final)),
            same_endpoint_actual_minus_target_RMS_rad_s=math.sqrt(sum((x-y)**2 for x, y in zip(actual, final))/len(final)))
    contacts = {}
    for leg in wheel_order:
        current = [e["current_legs"][leg] for e in physical]
        modes = ["AIR" if c["air"] else
                 (("GROUND+" if c["ground_contact"] else "")+
                  (c["contact_surface"] if c["obstacle_pair_active"] else "")).rstrip("+")
                 for c in current]
        fd = [c["front_distance_m"] for c in current]
        contacts[leg] = dict(samples=len(current), mode_counts=dict(Counter(modes)),
            mode_transition_count=sum(a != b for a, b in zip(modes, modes[1:])),
            mode_transitions=[dict(tick=samples[k]["physics_tick"], previous=modes[k-1], current=modes[k])
                              for k in range(1,len(modes)) if modes[k] != modes[k-1]],
            current_support_count=sum(c["support"] for c in current),
            ground_pair_contact_count=sum(c["ground_contact"] for c in current),
            obstacle_pair_contact_count=sum(c["obstacle_pair_active"] for c in current),
            actual_TOP_verified_bearing_count=sum(c["top_surface_contact"] and c["bearing_verified"] and c["support"] for c in current),
            bearing_force_n=stats(c["bearing_force_n"] for c in current),
            wheel_center_front_distance_m=stats(fd), wheel_center_forward_displacement_m=fd[-1]-fd[0],
            wheel_bottom_gap_above_obstacle_top_m=stats(c["clearance_m"] for c in current),
            placed_history_count=sum(s["semantic_task"]["history"]["placed"][leg] for s in samples))
    positions = {}
    margins = {}
    for e in physical:
        for role in e["transfer_roles"].values():
            for name, values in role["receiver_workspace_state"]["joint_range_margin_deg"].items():
                lo, hi = servo_limits_deg(name)
                value = values["negative_deg"]+lo
                assert abs(hi-value-values["positive_deg"]) < 1e-8
                positions.setdefault(name, []).append(value)
                margins.setdefault(name, []).append(min(values.values()))
    assert all(len(v) == 151 for v in positions.values())
    geometry = [g for s in actions for g in s["reward_breakdown"]["task_space_quality_sample_audit"]]
    assert [round(g["sim_time_s"]*120) for g in geometry] == list(range(2801, 4001))
    assert all(g["eligible"] and g["valid"] for g in geometry)
    body = [e["goal_features"]["body_forward_m"] for e in physical]
    rear = [max(e["current_legs"][leg]["front_distance_m"] for leg in ("RL", "RR")) for e in physical]
    com = [e["transfer_roles"]["RR"]["transfer_direction_context"]["mass_weighted_com_position_w_m"] for e in physical]
    first_global, last_global = rows[0]["global_policy_decision"], rows[-1]["global_policy_decision"]
    updates = [dict(ppo_update=u["ppo_update"], global_policy_decisions=u["global_policy_decisions"], actor_parameters_changed=u["actor_parameters_changed"])
               for u in lines(args.run / "optimizer_updates.jsonl") if first_global <= u["global_policy_decisions"] < last_global]
    result = dict(schema="training_P06_wheel_readonly.v1", source_run=str(args.run),
        classification="stochastic_training_rollout_with_updates_NOT_fixed_checkpoint_evaluation_or_causal_traction_experiment",
        window=dict(episode_tick_endpoints=[2800,4000], sim_time_s=[2800/120,4000/120], duration_s=10.,
            physical_endpoint_samples=151, action_intervals=150, action_interval_end_ticks=[2808,4000],
            verified_dispatch_physics_ticks=1200, global_decision_endpoints=[first_global,last_global], optimizer_updates=updates,
            native_command_tick_minus_episode_poststep_tick=sorted(set(t["command_physics_tick"]-t["episode_physics_tick"] for t in step_audit)),
            tick_alignment="episode readback is after physics; native command tick has separate reset/settle offset and refers to preceding dispatch, never same clock label"),
        wheels=wheels,
        dispatch=dict(all_full12_masks_one=all(n["phase_mask_full12"] == [1]*12 for n in natives),
            all_final_native_verified=all(n["verified"] and n["setter_dispatch_targets_equal"] and n["actual_mapping_matches_dispatch"] for n in natives),
            all_1200_tick_dispatches_verified=all(t["verified"] for t in step_audit),
            handoff_hold_ticks=sum(t["handoff_hold_used"] for t in step_audit),
            clipped_servo_indices=sorted(set(i for h in heads for i in h["clipped_servo_indices"])),
            native_buffer_read_sources=sorted(set(n["actual_target_source"] for n in natives)),
            source_owner_evidence="P06 source layer present; live measured retirement wheel_gain1; finite source before endpoint; captured FR/FL servo owners held, later cooperation not blocked",
            source_tail_statuses=sorted(set(d["p06_wheel_tail"]["status"] for d in nominal)),
            source_wheel_gains=sorted(set(d["p06_wheel_tail"]["wheel_gain"] for d in nominal)),
            retirement_wheel_gains=sorted(set(d["p06_rolling_retirement"]["wheel_gain"] for d in nominal)),
            stop_owner_active_count=sum(d["final_stop_owner"]["active"] for d in nominal),
            raw_joint_ids=None, complete_per_channel_owner_ledger=None,
            owner_evidence_limit="Named source layer and verified last dispatcher/buffer are logged, not numeric joint IDs or a full source-event owner ledger."),
        body_and_progress=dict(body_forward_from_obstacle_front_m=stats(body), body_forward_displacement_m=body[-1]-body[0],
            nearest_rear_center_from_front_m=stats(rear), nearest_rear_center_forward_displacement_m=rear[-1]-rear[0],
            mass_weighted_CoM_world_displacement_m=[com[-1][i]-com[0][i] for i in range(3)],
            collider_min_world_z_m_120Hz=stats(g["body_collider_minimum_w_m"][2] for g in geometry),
            conservative_body_obstacle_AABB_separation_m_120Hz=stats(g["separation_lower_bound_m"] for g in geometry),
            geometry_quality_cost_sum=sum(g["weighted_geometry_cost"] for g in geometry),
            hip_mount_world_z_m=None, hip_absence_reason="No four_hip_geometry records in this training stream; not inferred from base height."),
        current_contact_and_wheel_end_motion=contacts,
        linkage=dict(nominal_servo_unique_vectors=[list(v) for v in sorted(set(tuple(s["nominal_action_full12"][:8]) for s in actions))],
            measured_positions_deg={name:variation(v) for name,v in positions.items()},
            minimum_to_either_joint_limit_deg={name:min(v) for name,v in margins.items()},
            position_source="Saved measured joint_range_margin_deg + verified physical lower limit; validated against positive margin, not reconstructed from targets.",
            limitation="15Hz sampled total variation/reversals show back-and-forth joint motion, not exact 120Hz path length, periodic gait, work, or propulsion attribution."),
        conclusions=["All four wheel targets and measured canonical rotations are present; no RR-only dispatch or nominal deletion by residual mask in this window.",
            "FL historical capture is not current load bearing; its airborne rotation cannot establish ground traction.",
            "Positive body and rear-wheel-center motion coexist with source rolling commands and residual joint oscillations. No torque/work/slip or wheel-off controlled comparison: main propulsion fraction remains unproven."])
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

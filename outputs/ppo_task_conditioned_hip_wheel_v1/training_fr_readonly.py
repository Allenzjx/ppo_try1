"""Bounded FR audit from a live training log's already-complete decisions."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
from rr_probe_readonly import ROOT, B, lines, stats


def load_fr(path,kind):
    selected=[]
    for row in lines(path):
        if kind=="video" and "step_info" not in row:continue
        info=row["applied_audit"] if kind=="training" else row["step_info"]
        selected.append((row,info))
        placed=info["semantic_task"]["history"]["event_ticks"]["placed"].get("FR")
        if placed is not None:break
    else:raise ValueError("FR capture not yet in a complete written decision")
    return selected,placed


def front_metrics(selected,placed):
    audit=[a for _,s in selected for a in s["reward_breakdown"].get("front_quality_sample_audit",[]) if a["sim_time_s"]*120<=placed+1e-6]
    ticks=[round(a["sim_time_s"]*120) for a in audit]
    assert ticks==list(range(1,max(ticks)+1)),"FR quality prefix is not a contiguous120Hz stream"
    duration=sum(a["dt_s"] for a in audit)
    return dict(tick_interval=[1,ticks[-1]],physics_intervals=len(audit),duration_s=duration,
        phase_counts=dict(Counter(a["phase"] for a in audit)),
        rate_RMS_rad_s=math.sqrt(sum(.5*sum(v*v for v in a["roll_pitch_rate_rad_s"])*a["dt_s"] for a in audit)/duration),
        roll_RMS_rad=math.sqrt(sum(a["roll_pitch_rad"][0]**2*a["dt_s"] for a in audit)/duration),
        pitch_RMS_rad=math.sqrt(sum(a["roll_pitch_rad"][1]**2*a["dt_s"] for a in audit)/duration),
        tilt_peak_rad=max(math.hypot(*a["roll_pitch_rad"]) for a in audit),
        weighted_front_cost=sum(a["weighted_quality_cost"] for a in audit),
        beta_per_s=stats(a["effective_beta_per_s"] for a in audit))


def sampled_fr(selected,placed):
    decisions=[s for _,s in selected if s["physics_tick"]<=placed]
    legs=[s["semantic_task"]["physical_evaluator"]["current_legs"]["FR"] for s in decisions]
    safe=next((s["physics_tick"] for s,r in zip(decisions,legs) if r["air"] and r["clearance_m"]>=.015),None)
    cross=selected[-1][1]["semantic_task"]["history"]["event_ticks"]["front_edge_crossed"]["FR"]
    gaps=[r["clearance_m"] for s,r in zip(decisions,legs) if safe is not None and safe<=s["physics_tick"]<cross]
    return dict(decision_samples=len(decisions),last_sample_tick=decisions[-1]["physics_tick"],
        body_collider_min_world_z_m_15Hz=stats(s["semantic_task"]["physical_evaluator"]["body_traversal_geometry"]["minimum_w_m"][2] for s in decisions),
        first_sampled_AIR_with_15mm_top_gap_tick=safe,FR_top_gap_after_first_safe_before_cross_m_15Hz=stats(gaps),
        FR_top_gap_all_samples_m_15Hz=stats(r["clearance_m"] for r in legs),
        missing_exact_capture_tick_in_decision_stream=placed%8!=0,
        exact_120Hz_full_FR_gap=None,hip_world_z=None,
        absence_reason="matched decision-endpoint slice only; no120Hz gap interpolation or reconstructed hip height; B dense raw stream exists separately")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();training,placed=load_fr(args.run/"residual_and_projection_audit.jsonl","training")
    reference,bplaced=load_fr(B/"video_policy_decisions.jsonl","video")
    front=front_metrics(training,placed); bfront=front_metrics(reference,bplaced)
    spaces=[a for _,s in training for a in s["reward_breakdown"]["task_space_quality_sample_audit"] if a["sim_time_s"]*120<=placed+1e-6]
    eligible=[a for a in spaces if a["eligible"]]
    assert all(a["valid"] and not a["terminal_measurement_omitted"] for a in eligible),"missing eligible geometry"
    assert [round(a["sim_time_s"]*120) for a in spaces]==list(range(1,placed+1))
    equations=[]
    for _,s in training:
        r=s["reward_breakdown"];c=r["cost_components"]
        if r.get("front_quality_sample_audit"):
            assert "front_weighted_tilt_cost" in c and "front_weighted_rate_cost" in c
        if any(a["eligible"] for a in r["task_space_quality_sample_audit"]):
            assert "task_space_weighted_geometry_cost" in c
        equations.append(abs(r["families"]["body_stability"]+c.get("front_weighted_tilt_cost",0.)+c.get("front_weighted_rate_cost",0.)+c.get("task_space_weighted_geometry_cost",0.)))
    config_check=max(abs(a["weighted_geometry_cost"]-.03*a["dt_s"]*min(1.,max(0.,(.02-a["separation_lower_bound_m"])/.02))**2) for a in eligible)
    bgeom=[]
    for raw in lines(B/"physical_observations.jsonl"):
        tick=raw["physics_tick"]
        if tick>bfront["tick_interval"][1]:break
        if tick==0:continue
        bounds=raw["body_bounds_w_m"]["base_link"];ob=raw["obstacle"]
        ol=(ob["front_x_m"],ob["right_y_m"],ob["bottom_z_m"]);oh=(ob["back_x_m"],ob["left_y_m"],ob["top_z_m"])
        distance=math.sqrt(sum(max(a-y,x-b,0.)**2 for a,b,x,y in zip(bounds["minimum_m"],bounds["maximum_m"],ol,oh)))
        bgeom.append((bounds["minimum_m"][2],distance))
    audit_end=training[-1][1]["physics_tick"];end_global=training[-1][0]["global_policy_decision"]
    updates=[r for r in lines(args.run/"optimizer_updates.jsonl") if r["global_policy_decisions"]<end_global]
    result=dict(schema="training_FR_quality_readonly.v1",source_run=str(args.run),
        classification="stochastic_learning_rollout_with_in_episode_optimizer_update_NOT_fixed_checkpoint_evaluation",
        action_audit=dict(decisions=len(training),sampling_modes=sorted(set(r["policy_request"]["mode"] for r,_ in training)),
            all_full12_masks_one=all(s["actuator_target_effect_audit"]["phase_mask_full12"]==[1]*12 for _,s in training),
            all_native_verified=all(s["actuator_target_effect_audit"]["verified"] for _,s in training),
            nonzero_projected_REQUEST_counts_full12=[sum(abs(s["projected_residual_full12"][i])>1e-12 for _,s in training) for i in range(12)]),
        global_decision_span=[training[0][0]["global_policy_decision"],end_global],
        FR_events=training[-1][1]["semantic_task"]["history"]["event_ticks"],FR_placed_tick=placed,FR_placed_time_s=placed/120,
        last_complete_decision_read_tick=audit_end,extra_decision_ticks_after_capture=audit_end-placed,
        optimizer_updates_before_FR_capture=[dict(ppo_update=u["ppo_update"],global_policy_decisions=u["global_policy_decisions"],
            actor_parameters_changed=u["actor_parameters_changed"]) for u in updates],
        all_consumed_objective_profiles=sorted(set(s["reward_breakdown"]["objective_profile"] for _,s in training)),
        all_consumed_quality_epsilons=sorted(set(s["reward_breakdown"]["quality_epsilon"] for _,s in training)),
        physical_front_P01_P02=front,
        geometry_P01_P02=dict(eligible_120Hz_samples=len(eligible),all_valid=True,invalid_eligible_count=0,
            body_collider_min_world_z_m=stats(a["body_collider_minimum_w_m"][2] for a in eligible),
            conservative_obstacle_separation_lower_bound_m=stats(a["separation_lower_bound_m"] for a in eligible),
            margin_m=.020,beta_per_s=stats(a["effective_beta_per_s"] for a in eligible),
            nonzero_geometry_cost_samples=sum(a["raw_geometry_cost"]>0 for a in eligible),
            weighted_geometry_cost=sum(a["weighted_geometry_cost"] for a in eligible),
            expected_weighted_formula_max_error=config_check),
        emitted_geometry_audit_samples_through_capture=len(spaces),
        outside_quality_phase_count=sum(not a["eligible"] for a in spaces),
        outside_quality_phase_reason_counts=dict(Counter(a["reason"] for a in spaces if not a["eligible"])),
        full_FR_120Hz_rate_RMS=None,full_FR_120Hz_geometry_minimum=None,
        full_FR_missing_reason="P03 has no front tilt/rate audit and is outside task_space_quality phases; do not fill its29 physical ticks with zeros",
        sampled_full_FR=sampled_fr(training,placed),
        max_body_reward_decomposition_error=max(equations),
        body_family_reward_through_last_complete_decision=sum(s["reward_breakdown"]["families"]["body_stability"] for _,s in training),
        reference_B=dict(path=str(B),classification="fixed nominal zero residual independent full success retained",
            FR_events=reference[-1][1]["semantic_task"]["history"]["event_ticks"],FR_placed_tick=bplaced,FR_placed_time_s=bplaced/120,
            physical_front_P01_P02=bfront,
            geometry_P01_P02=dict(body_collider_min_world_z_m=stats(a for a,b in bgeom),conservative_obstacle_separation_lower_bound_m=stats(b for a,b in bgeom)),
            sampled_full_FR=sampled_fr(reference,bplaced)),
        differences=dict(FR_placement_delay_s=(placed-bplaced)/120,
            P01_P02_rate_RMS_fraction=front["rate_RMS_rad_s"]/bfront["rate_RMS_rad_s"]-1,
            P01_P02_peak_tilt_fraction=front["tilt_peak_rad"]/bfront["tilt_peak_rad"]-1),
        not_full_task_success_claim=True,no_fixed_checkpoint_improvement_claim=True)
    with args.output.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ("global_decision_span","FR_placed_tick","optimizer_updates_before_FR_capture","physical_front_P01_P02","geometry_P01_P02","differences","max_body_reward_decomposition_error","body_family_reward_through_last_complete_decision")},indent=2))


if __name__=="__main__":main()

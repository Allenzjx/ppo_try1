"""Fixed block13 audit/update analysis; stdlib only, no Torch/Isaac or physical stream."""
from __future__ import annotations
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/"runs/ppo_fsm_reference_p09_stable_v2/train/20260910T1451368221058Z_g7db0d17f398d_19bb6607a9384f08b00d341ba21060b7"
OUT=ROOT/"outputs/ppo_fsm_reference_p09_stable_v2/block13_reward_execution_learning_diagnosis.json"
PHASES=("P05","P06","P09")
CHANNELS=("FL_hip_deg","FL_knee_deg","FR_hip_deg","FR_knee_deg","RL_hip_deg","RL_knee_deg","RR_hip_deg","RR_knee_deg","FL_wheel_rad_s","FR_wheel_rad_s","RL_wheel_rad_s","RR_wheel_rad_s")
CAPS={"P05":[18,24,18,24,12,18,12,18,1,.6,1,.6],
      "P06":[32,36,24,112,24,36,24,36,1.2,1.2,1,.6],
      "P09":[32,36,24,112,24,36,24,36,1.2,1.2,1,.6]}
FAMILIES=("task_progress","body_stability","contact_motion_quality","control_smoothness","control_regularization")
WEIGHTS=dict(zip(FAMILIES,(1,.4,.2,.1,0)))
gamma=.9985
def stats(values):
    xs=sorted(values)
    if not xs:return {"n":0}
    def q(f):
        x=(len(xs)-1)*f
        lo=math.floor(x);hi=math.ceil(x)
        return xs[lo]+(xs[hi]-xs[lo])*(x-lo)
    return {"n":len(xs),"sum":sum(xs),"mean":sum(xs)/len(xs),"min":xs[0],
            "p05":q(.05),"median":q(.5),"p95":q(.95),"max":xs[-1]}
def f32(x):return struct.unpack("f",struct.pack("f",x))[0]
def record_stats(rows):
    if not rows:return {"n":0}
    names=("reward","task_progress","PBRS","terminal_event","time_cost","body_stability",
           "contact_motion_quality","control_smoothness","control_regularization",
           "delta_phi","phi_after","gravity_attitude","euler_rate","angular_acceleration",
           "actual_first","actual_second","nominal_first_diagnostic","residual_first_diagnostic",
           "contact_chatter_diagnostic","rebound_cost_integral")
    out={name:stats([r[name] for r in rows]) for name in names}
    out["seconds"]=sum(r["seconds"] for r in rows)
    out["positive_PBRS_decisions"]=sum(r["PBRS"]>0 for r in rows)
    out["positive_PBRS_but_negative_total_decisions"]=sum(r["PBRS"]>0 and r["reward"]<0 for r in rows)
    out["positive_delta_phi_but_nonpositive_PBRS"]=sum(r["delta_phi"]>0 and r["PBRS"]<=0 for r in rows)
    out["positive_PBRS_sum"]=sum(max(0,r["PBRS"]) for r in rows)
    out["negative_PBRS_magnitude_sum"]=sum(max(0,-r["PBRS"]) for r in rows)
    out["body_cost_magnitude_to_positive_PBRS_sum"]=(sum(-r["body_stability"] for r in rows)/out["positive_PBRS_sum"] if out["positive_PBRS_sum"] else None)
    out["all_nonprogress_cost_to_positive_PBRS_sum"]=(sum(-r["body_stability"]-r["contact_motion_quality"]-r["control_smoothness"]-r["time_cost"] for r in rows)/out["positive_PBRS_sum"] if out["positive_PBRS_sum"] else None)
    return out

counts=Counter(); ep=0; last=None; prev_projected=[0.]*12
checks=Counter(); maxerrors=Counter(); boundaries=[]; terminals=[]; phase_rows=defaultdict(list)
execution={p:{"n":0,"channels":{name:Counter() for name in CHANNELS},"decision_counts":Counter(),"native_ticks":Counter(),"examples_unmet":[]} for p in PHASES}
native_total=Counter(); rows_total=0; all_ep_rewards=[]; current_ep=[]
for line in (RUN/"residual_and_projection_audit.jsonl").open(encoding="utf-8"):
    row=json.loads(line); rows_total+=1
    assert row["global_policy_decision"]==149888+rows_total
    a=row["applied_audit"]; p=a["phase_id"]; counts[p]+=1
    b=a["reward_breakdown"]; fam=b["families"]; comp=b["cost_components"]
    terminal=row["terminal"]
    assert terminal == bool(a["termination_reason"])
    if last is not None and not last["terminal"]:
        err=abs(b["potential_before"]-last["applied_audit"]["reward_breakdown"]["potential_after"])
        maxerrors["nonterminal_phi_continuity"]=max(maxerrors["nonterminal_phi_continuity"],err)
    expected_pbrs=5*(gamma*b["potential_after"]-b["potential_before"])
    errors={
      "PBRS_formula":abs(b["potential_shaping"]-expected_pbrs),
      "task_family":abs(fam["task_progress"]-(b["potential_shaping"]+b["terminal_event"]-.02*b["elapsed_physics_s"])),
      "weighted_family_sum":abs(sum(fam.values())-b["total"]),
      "audit_double_reward":abs((a["reward"]["total"] if isinstance(a["reward"],dict) else a["reward"])-b["total"]),
      "collector_float32_reward":abs(row["reward"]-f32(b["total"])),
      "body_component_identity":abs(fam["body_stability"]+.4*(comp["gravity_attitude"]+comp["euler_rate"]+comp["angular_acceleration"])/3),
      "applied_only_smoothness_identity":abs(fam["control_smoothness"]+.1*(comp["actual_drive_first_difference"]+comp["actual_drive_second_difference"])/2),
      "contact_component_identity":abs(fam["contact_motion_quality"]+.2*comp["contact_quality"]),
      "actual_tick_duration":abs(b["elapsed_physics_s"]-a["physics_ticks"]/120)}
    for name,value in errors.items():
        maxerrors[name]=max(maxerrors[name],value)
        checks[name+"_within_1e-10"]+=value<1e-10
    for name in FAMILIES:
        maxerrors["family_weight_identity"]=max(maxerrors["family_weight_identity"],abs(fam[name]-b["unweighted_families"][name]*WEIGHTS[name]))
    checks["raw_matches_applied_audit"]+=row["raw_policy_action_full12"]==a["raw_policy_action_full12"]
    joint_logp=sum(-.5*((x-mu)/sd)**2-math.log(sd)-.5*math.log(2*math.pi)
                  for x,mu,sd in zip(row["raw_policy_action_full12"],row["old_distribution_mean_full12"],row["old_distribution_std_full12"]))
    maxerrors["old_joint_gaussian_logp_float64_vs_logged_float32"]=max(maxerrors["old_joint_gaussian_logp_float64_vs_logged_float32"],abs(joint_logp-row["old_log_probability"]))
    checks["finite_raw_mean_std_logp_reward"]+=all(math.isfinite(x) for xs in (row["raw_policy_action_full12"],row["old_distribution_mean_full12"],row["old_distribution_std_full12"],[row["old_log_probability"],row["reward"]]) for x in xs)
    checks["positive_std"]+=all(x>0 for x in row["old_distribution_std_full12"])
    checks["timeout_false"]+=a["time_outs"] is False
    checks["bootstrap_matches_nonterminal"]+=a["terminal_bootstrap_allowed"] is (not terminal)
    checks["reward_bootstrap_matches_nonterminal"]+=b["terminal_bootstrap_allowed"] is (not terminal)
    checks["terminal_phi_zero"]+=terminal and b["potential_after"]==0
    checks["nonpositive_nonprogress_families"]+=all(fam[k]<=0 for k in FAMILIES[1:])
    checks["no_regularization"]+=fam["control_regularization"]==0
    checks["no_in_episode_state_writes"]+=a["no_in_episode_state_writes_verified"] is True
    native=a["actuator_target_effect_audit"]
    checks["all12_mask_open"]+=native["phase_mask_full12"]==[1]*12
    for tick in a["actuator_target_effect_audit_ticks"]:
        native_total["ticks"]+=1
        native_total["verified"]+=tick["verified"] is True
        native_total["effect"]+=tick["actual_native_effect"] is True
        native_total["own_phase"]+=tick["own_phase_request_effect"] is True
    compact={"global":row["global_policy_decision"],"episode":ep,"phase":p,"end_phase":a["end_phase_id"],
        "tick":a["physics_tick"],"terminal":terminal,"reward":row["reward"],"task_progress":fam["task_progress"],
        "PBRS":b["potential_shaping"],"terminal_event":b["terminal_event"],
        "time_cost":-.02*b["elapsed_physics_s"],"seconds":b["elapsed_physics_s"],
        "delta_phi":b["potential_after"]-b["potential_before"],"phi_after":b["potential_after"],
        **{name:fam[name] for name in FAMILIES[1:]},
        "gravity_attitude":comp["gravity_attitude"],"euler_rate":comp["euler_rate"],"angular_acceleration":comp["angular_acceleration"],
        "actual_first":comp["actual_drive_first_difference"],"actual_second":comp["actual_drive_second_difference"],
        "nominal_first_diagnostic":comp["nominal_first_difference"],"residual_first_diagnostic":comp["residual_first_difference"],
        "contact_chatter_diagnostic":comp.get("contact_chatter_diagnostic",0),"rebound_cost_integral":comp.get("confirmed_post_touchdown_rebound",0)}
    current_ep.append(compact)
    if p in PHASES:
        phase_rows[p].append(compact)
        ex=execution[p]; ex["n"]+=1
        raw=row["raw_policy_action_full12"]
        target=[math.tanh(x)*cap for x,cap in zip(raw,CAPS[p])]
        requested=a["projected_residual_full12"]
        headroom=native.get("policy_headroom_evidence",{})
        effective=headroom.get("effective_policy_residual_full12")
        candidate=headroom.get("candidate_native_target_before_final_slew_full12")
        actual=a["actual_drive_target_full12"]
        hold=sum(tick["handoff_hold_used"] is True for tick in a["actuator_target_effect_audit_ticks"])
        available=max(0,a["physics_ticks"]-hold)/120
        predicted=[old+max(-rate*available,min(rate*available,new-old)) for old,new,rate in zip(prev_projected,target,[60]*8+[1.8]*4)]
        flags=Counter()
        for i,name in enumerate(CHANNELS):
            c=ex["channels"][name]; c["samples"]+=1
            c["tanh_abs_ge_0.95"]+=abs(math.tanh(raw[i]))>=.95
            c["tanh_abs_ge_0.99"]+=abs(math.tanh(raw[i]))>=.99
            c["filtered_abs_cap_fraction_ge_0.95"]+=abs(requested[i])/CAPS[p][i]>=.95
            unmet=abs(target[i]-requested[i])>1e-7
            c["requested_vs_filtered_diff"]+=unmet; flags["unmet"]+=unmet
            rate_consistent=unmet and abs(predicted[i]-requested[i])<1e-7
            c["unmet_consistent_with_rate_and_handoff_hold"]+=rate_consistent
            flags["rate_consistent"]+=rate_consistent
            if effective is not None:
                clipped=abs(effective[i]-requested[i])>1e-7
                c["headroom_effective_diff"]+=clipped; flags["headroom"]+=clipped
            if candidate is not None:
                finaldiff=abs(candidate[i]-actual[i])>1e-7
                c["candidate_vs_final_target_diff"]+=finaldiff; flags["finaldiff"]+=finaldiff
        ex["decision_counts"]["requested_vs_filtered_diff"]+=flags["unmet"]>0
        ex["decision_counts"]["rate_consistent_unmet"]+=flags["rate_consistent"]>0
        ex["decision_counts"]["headroom_effective_diff"]+=flags["headroom"]>0
        ex["decision_counts"]["candidate_vs_final_target_diff"]+=flags["finaldiff"]>0
        ex["decision_counts"]["headroom_receipt_available"]+=effective is not None
        ex["decision_counts"]["final_candidate_receipt_available"]+=candidate is not None
        ex["decision_counts"]["handoff_hold_tick_in_decision"]+=hold>0
        ex["decision_counts"]["headroom_clipped_servo_indices_nonempty"]+=bool(headroom.get("clipped_servo_indices",[]))
        for tick in a["actuator_target_effect_audit_ticks"]:
            ex["native_ticks"]["ticks"]+=1
            for label,key in (("verified","verified"),("effect","actual_native_effect"),("own_phase","own_phase_request_effect")):
                ex["native_ticks"][label]+=tick[key] is True
        if flags["unmet"] and len(ex["examples_unmet"])<2:
            ex["examples_unmet"].append({"global":row["global_policy_decision"],"tick":a["physics_tick"],
                "raw":raw,"tanh_cap_target":target,"filtered":requested,"effective":effective,
                "actual_target":actual,"handoff_hold_ticks":hold,"all_rate_prediction_max_error":max(abs(x-y) for x,y in zip(predicted,requested))})
    if p!=a["end_phase_id"]:
        boundaries.append({"global":row["global_policy_decision"],"episode":ep,"tick":a["physics_tick"],"from":p,"to":a["end_phase_id"],
            "done":terminal,"bootstrap_allowed":a["terminal_bootstrap_allowed"],"time_outs":a["time_outs"],
            "phi_before":b["potential_before"],"phi_after":b["potential_after"],"PBRS":b["potential_shaping"],
            "task_progress":fam["task_progress"],"terminal_event":b["terminal_event"]})
    if terminal:
        terminals.append({**compact,"reason":a["termination_reason"]})
        all_ep_rewards.append(current_ep);current_ep=[];ep+=1;prev_projected=[0.]*12
    else: prev_projected=a["projected_residual_full12"]
    last=row
assert rows_total==2048
if current_ep:all_ep_rewards.append(current_ep)
updates=[json.loads(line) for line in (RUN/"optimizer_updates.jsonl").open()]
assert len(updates)==16
assert [r["global_policy_decisions"] for r in updates]==list(range(150016,151937,128))
assert [r["ppo_update"] for r in updates]==list(range(1137,1153))
telescopes=[]
for epi,rows in enumerate(all_ep_rewards):
    before=rows[0]["phi_after"]-rows[0]["delta_phi"]
    discounted=sum(gamma**i*r["PBRS"] for i,r in enumerate(rows))
    expected=5*(gamma**len(rows)*rows[-1]["phi_after"]-before)
    telescopes.append({"episode":epi,"decisions":len(rows),"terminal":rows[-1]["terminal"],
       "discounted_PBRS_sum":discounted,"telescoping_expected":expected,"absolute_error":abs(discounted-expected)})
result={"schema":"wlr50_clean.fixed_block13_learning_audit.v1","checked_at_utc":datetime.now(timezone.utc).isoformat(),
 "source_run":RUN.name,"source_head":"7db0d17f398d393ce026b6990bd2566d53366407",
 "scope":"completed block13 2048 policy audits and 16 update rows; no rollout tensor, checkpoint/model bytes, physical stream, active data or CSV",
 "sampled_and_update_covered_decisions":rows_total,"global_range":[149889,151936],
 "ppo_updates":16,"optimizer_steps":sum(r["optimizer_steps"] for r in updates),
 "phase_counts":dict(counts),"native_total":dict(native_total),"checks":dict(checks),"max_errors":dict(maxerrors),
 "phase_reward_scales":{p:{"all":record_stats(rows),"nonterminal":record_stats([r for r in rows if not r["terminal"]]),
    "terminal":record_stats([r for r in rows if r["terminal"]]),
    "positive_progress_negative_total_examples":[r for r in rows if not r["terminal"] and r["PBRS"]>0 and r["reward"]<0][:3]}
    for p,rows in phase_rows.items()},
 "reward_configuration":{"gamma":gamma,"potential_weight":5.,"failure_cost":40.,"time_cost_per_s":.02,
    "family_weights":WEIGHTS,"no_reference_imitation":True,"no_phase_transition_bonus":True,"no_residual_magnitude_cost":True,
    "contact_chatter_is_diagnostic":True,"positive_PBRS_means_discounted_Phi_growth_not_just_positive_delta":True},
 "ordinary_phase_boundaries":boundaries,"terminals":terminals,"episode_PBRS_telescoping":telescopes,
 "residual_execution":execution,
 "optimizer_update_statistics":{key:stats([r[key] for r in updates]) for key in
    ("optimizer_learning_rate","kl_mean","clip_fraction","entropy","value_loss","surrogate_loss","gradient_norm_min","gradient_norm_max")},
 "all_updates_actor_changed":all(r["actor_parameters_changed"] for r in updates),
 "all_updates_finite_nonzero_gradient":all(r["finite_nonzero_gradient_observed"] for r in updates),
 "limitations":["tanh >=.95/.99 are descriptive saturation thresholds, not control/success gates.",
 "Requested-minus-filtered is directly measured; the rate+handoff formula is only a consistency classification, not a replay of every physical projector tick.",
 "Headroom and final target differences use recorded end-tick receipts; final target difference may include final clamp/slew, not measured joint tracking.",
 "Audit reward/raw/logprob/done checks and successful optimizer guards are not a fresh independent tensor-storage/GAE reconstruction; no Torch/rollout loading was requested.",
 "No causal counterfactual proves which transient motion was necessary; positive task shaping offset by stability/smoothness cost is a tradeoff, not alone a reward-sign defect.",
 "This is old-v1 natural-P01 stochastic training with mid-episode updates, not a fixed-policy full success or the current v2-control learning distribution."]}
with OUT.open("x",encoding="utf-8") as f:json.dump(result,f,indent=2)
print(json.dumps({"output":str(OUT),"phase_counts":dict(counts),"checks":dict(checks),"max_errors":dict(maxerrors),
 "reward_scales":{p:{"n":len(rows),"nonterminal":{k:stats([r[k] for r in rows if not r["terminal"]]) for k in ("reward","task_progress","PBRS","body_stability","contact_motion_quality","control_smoothness")},
 "terminals":[{"reason":r["reason"],"reward":r["reward"],"PBRS":r["PBRS"]} for r in terminals if r["phase"]==p]}
 for p,rows in phase_rows.items()},
 "phase_boundaries":len(boundaries),"terminals":len(terminals),"execution":{p:{"n":v["n"],"decisions":dict(v["decision_counts"]),"channels":v["channels"]} for p,v in execution.items()},
 "max_telescoping_error":max(t["absolute_error"] for t in telescopes),"optimizer":result["optimizer_update_statistics"]},indent=2))

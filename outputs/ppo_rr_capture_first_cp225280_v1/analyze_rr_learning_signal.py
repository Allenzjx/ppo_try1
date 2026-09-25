"""Completed-block RR PPO evidence, never a controller or an optimizer.

Importing this file is standard-library only. Tensor/model imports are behind
the explicit --isaac-stopped acknowledgement and completed-block checks.
Same-observation checkpoint comparisons are counterfactual network outputs,
NOT physical replay, success evidence, or an executable action target.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs/ppo_rr_capture_first_cp225280_v1/train_first2048_ecf205e"
RR = {"hip": 6, "knee": 7}


def stats(values):
    values = list(values)
    if not values:
        return {"n": 0}
    if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
        raise ValueError("non-finite/non-numeric metric")
    return dict(n=len(values), min=min(values), mean=statistics.fmean(values),
                median=statistics.median(values), max=max(values),
                positive=sum(v > 0 for v in values), negative=sum(v < 0 for v in values))


def physical_bucket(before, after, seen_top):
    """Endpoint classification, not reconstruction of all native contacts."""
    if after["current_top_contact"] and not before["current_top_contact"]:
        return "TOP_reacquisition" if seen_top else "first_TOP_endpoint"
    if before["current_top_bearing"] and not after["current_top_bearing"]:
        return "bearing_drop_endpoint"
    if after["current_top_bearing"]:
        return "TOP_bearing_hold"
    if after["current_top_contact"]:
        return "TOP_contact_not_bearing"
    if after["ground_contact"]:
        return "GROUND_not_RR_success"
    if after["free_air"]:
        if seen_top:
            return "AIR_recapture_after_TOP"
        if after["within_top_xy"] and before["gap_m"] - after["gap_m"] > 1e-6:
            return "AIR_legal_gap_decreasing"
        return "AIR_other"
    return "other_contact_or_unknown"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_rows(path):
    with path.open("rb") as stream:
        for number, line in enumerate(stream, 1):
            if not line.endswith(b"\n"):
                raise ValueError(f"{path}:{number}: incomplete row; sealed runs only")
            yield json.loads(line)


def collect(run, expected_updates, *, allow_live_suffix=False, update_offset=0):
    updates = list(json_rows(run / "updates.jsonl"))
    if allow_live_suffix:
        updates = updates[:expected_updates]
    if len(updates) != expected_updates:
        raise ValueError("expected complete update block is not sealed yet")
    samples, prefix = [], 0
    episode, last_tick, seen_top = 0, None, False
    for row in json_rows(run / "decisions.jsonl"):
        if row["kind"] == "frozen_prior_prefix":
            if row["PPO_credit"] != 0:
                raise ValueError("prefix entered PPO")
            prefix += 1
            tick = row["local"]["metrics"]["tick"]
            if last_tick is None or tick < last_tick:
                episode += 1
                seen_top = False
            last_tick = tick
            continue
        if row["kind"] != "activated_on_policy" or row["PPO_credit"] != 1:
            raise ValueError("unexpected diagnostic/AUX/uncredited active row")
        info, policy = row["step_info"], row["policy_request"]
        if policy.get("independent_diagnostic"):
            raise ValueError("diagnostic is not on-policy")
        before, after = row["before_local"]["metrics"], info["rr_capture_local"]["metrics"]
        if not row["before_local"]["active"] or not policy["capture_active"]:
            raise ValueError("inactive observation in PPO")
        bucket = physical_bucket(before, after, seen_top)
        seen_top = seen_top or before["current_top_contact"] or after["current_top_contact"]
        last_tick = after["tick"]
        audit = info["actuator_target_effect_audit"]
        if not audit["verified"] or not info["actuator_target_effect_audit_summary"]["all_ticks_verified"]:
            raise ValueError("missing verified native action receipt")
        headroom = audit["policy_headroom_evidence"]
        samples.append(dict(index=len(samples), episode=episode, tick=after["tick"],
            time_s=after["time_s"], phase=info["phase_id"], bucket=bucket,
            observation=row["observation"], policy=policy,
            old_logp=float(row["old_logp"][0]), before=before, after=after,
            hold_s=info["rr_capture_local"]["hold_elapsed_s"],
            local_success=info["local_task_success"], local_reward=info["local_reward"],
            final=info["actual_drive_target_full12"], headroom=headroom,
            native_nominal=audit["native_drive_target_full12"],
            controller_bias=audit["controller_drive_bias_full12"],
            source_nominal=info["nominal_action_full12"],
            tracking=audit.get("tracking_reference_evidence", {}),
            source_deferred=info.get("semantic_task", {}).get("nominal_provider_diagnostics", {}).get("rr_local_deferred_late_v2", {})))
        if allow_live_suffix and len(samples) == expected_updates * 512:
            break  # Already sealed decisions only; do not follow current rollout.
    if len(samples) != 512 * expected_updates:
        raise ValueError("credited decisions do not match complete512 updates")
    for index, update in enumerate(updates, 1):
        if update["counts"]["local_ppo_updates"] != index + update_offset:
            raise ValueError("only explicitly offset contiguous local update blocks supported")
        if update["counts"]["local_policy_decisions"] != (index + update_offset) * 512:
            raise ValueError("update count mismatch")
    return samples, updates, prefix


def gaussian_kl_parts(old_mean, old_std, new_mean, new_std):
    """KL(old || current), decomposed into mean and scale, no tensor import."""
    values=(old_mean,old_std,new_mean,new_std)
    if not all(math.isfinite(x) for x in values) or old_std <= 0 or new_std <= 0:
        raise ValueError("finite Gaussian means and positive std required")
    ratio=old_std/new_std
    mean=(old_mean-new_mean)**2/(2*new_std**2)
    scale=.5*(ratio*ratio-1)-math.log(ratio)
    return dict(mean=mean,scale=scale,total=mean+scale)


def channel_kl_report(run,samples,updates, *, update_offset=0):
    """Actual minibatch Gaussian cache vs collection Gaussian, JSON only."""
    groups={"RR_hip_knee":(6,7),"other10":(0,1,2,3,4,5,8,9,10,11),
            "other6_leg_joints":(0,1,2,3,4,5),"wheels4":(8,9,10,11)}
    reports=[]
    for update_number,update in enumerate(updates,1):
        rows=samples[(update_number-1)*512:update_number*512]
        path=run/"rollouts"/f"likelihood_{update_number + update_offset:04d}.json"
        likelihood=json.loads(path.read_text())
        batches=[]
        for batch in likelihood["minibatches"]:
            identities=batch["rollout_flat_indices"]
            means=batch["current_conditional_mean"]
            sigmas=batch["current_conditional_sigma"]
            if len(identities)!=len(means) or len(means)!=len(sigmas):
                raise ValueError("minibatch shape mismatch")
            parts=[]
            for identities_,mu,sigma in zip(identities,means,sigmas):
                if len(identities_)!=1 or not 0 <= identities_[0] < len(rows):
                    raise ValueError("ambiguous or missing rollout identity")
                row=rows[identities_[0]]["policy"]
                parts.append([gaussian_kl_parts(a,b,c,d) for a,b,c,d in zip(
                    row["conditional_mean_full12"],row["active_conditional_std_full12"],mu,sigma)])
                if len(parts[-1])!=12:
                    raise ValueError("full12 KL expected")
            channels=[{part:statistics.fmean(p[j][part] for p in parts) for part in ("mean","scale","total")} for j in range(12)]
            total=sum(c["total"] for c in channels)
            batches.append(dict(index=batch["minibatch_index"],samples=len(parts),
                channels=channels,total_kl=total,
                groups={name:{part:sum(channels[j][part] for j in indices) for part in ("mean","scale","total")} for name,indices in groups.items()},
                reconstructed_adaptive_action="decrease" if total > .02 else "increase" if 0 < total < .005 else "unchanged",
                near_zero_float32_scheduler_ambiguity=abs(total)<1e-7))
        if len(batches)!=update["optimizer_steps"]:
            raise ValueError("KL minibatch / Adam step mismatch")
        averages={name:{part:statistics.fmean(b["groups"][name][part] for b in batches) for part in ("mean","scale","total")} for name in groups}
        total=statistics.fmean(b["total_kl"] for b in batches)
        error=abs(total-update["kl_mean"])
        if error > 1e-5:
            raise ValueError(f"JSON Gaussian KL differs from official audited KL: {error}")
        reports.append(dict(update=update_number + update_offset,official_kl_mean=update["kl_mean"],
            reconstructed_kl_mean=total,max_aggregate_float_error=error,
            actual_final_learning_rate=update["optimizer_learning_rate"],
            groups=averages,RR_fraction=averages["RR_hip_knee"]["total"]/total if total else None,
            channels=[{part:statistics.fmean(b["channels"][j][part] for b in batches) for part in ("mean","scale","total")} for j in range(12)],
            batches=batches))
    return dict(schema="wlr50_clean.RR_channel_KL_readonly.v1",updates=reports,
        group_semantics="RR is channels6/7; other10 is disjoint; other6 joints and wheels4 partition other10.",
        scheduler="Official global sum12 KL; >.02 divides LR by1.5 down to1e-5; 0<KL<.005 multiplies1.5 up to.01. JSON reconstructed near-zero values do not establish exact float32 scheduling.",
        causal_limit="Channel contribution explains the measured global KL. It does not prove a hypothetical separate-group schedule would improve physical RR placement.")


def metric_summary(rows):
    result = dict(samples=len(rows), episodes=sorted({r["episode"] for r in rows}),
                  local_success_endpoints=sum(r["local_success"] for r in rows))
    for key in ("reward", "return", "value", "advantage", "raw_gae"):
        result[key] = stats(r[key] for r in rows)
    result["gap_descent_mm"] = stats(1000 * (r["before"]["gap_m"]-r["after"]["gap_m"]) for r in rows)
    result["native_observer_hold_s"] = stats(r["hold_s"] for r in rows)
    result["RR"] = {}
    for joint, channel in RR.items():
        metrics = {}
        for label, field in (("selected_raw", "selected_raw_full12"),
                             ("conditional_mean", "conditional_mean_full12"),
                             ("conditional_std", "active_conditional_std_full12"),
                             ("local_head_mean_delta", "local_raw_mean_delta_full12")):
            metrics[label] = stats(r["policy"][field][channel] for r in rows)
        metrics["tanh_abs_ge_0_95"] = sum(abs(math.tanh(r["policy"]["selected_raw_full12"][channel])) >= .95 for r in rows)
        metrics["headroom_clipped_last_dispatch"] = sum(channel in r["headroom"]["clipped_servo_indices"] for r in rows)
        metrics["requested_residual_deg"] = stats(r["headroom"]["requested_policy_residual_full12"][channel] for r in rows)
        metrics["effective_residual_deg"] = stats(r["headroom"]["effective_policy_residual_full12"][channel] for r in rows)
        metrics["controller_bias_deg"] = stats(r["controller_bias"][channel] for r in rows)
        metrics["final_deg"] = stats(r["final"][channel] for r in rows)
        metrics["actual_deg"] = stats(r["after"]["actual_rr_hip_knee_deg"][channel-6] for r in rows)
        metrics["actual_delta_over_decision_deg"] = stats(r["after"]["actual_rr_hip_knee_deg"][channel-6]-r["before"]["actual_rr_hip_knee_deg"][channel-6] for r in rows)
        # Score direction at collection distribution only, not clipped-PPO's
        # actual multi-epoch gradient or causal physical action attribution.
        metrics["unclipped_collection_mean_score_times_advantage"] = stats(
            r["advantage"] * (r["policy"]["selected_raw_full12"][channel]-r["policy"]["conditional_mean_full12"][channel]) /
            r["policy"]["active_conditional_std_full12"][channel]**2 for r in rows)
        result["RR"][joint] = metrics
    return result


def close_tensor(torch, a, b, label, tolerance=1e-5):
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError(label + " shape/nonfinite mismatch")
    error = float((a-b).abs().max()) if a.numel() else 0.
    if error > tolerance:
        raise ValueError(f"{label} max mismatch {error} > {tolerance}")
    return error


def analyze(run, *, expected_updates=4, isaac_stopped=False):
    if not isaac_stopped:
        raise ValueError("Wait for explicit Isaac exit confirmation; require --isaac-stopped")
    samples, updates, prefix = collect(run, expected_updates)
    # No Torch, RSL, model, PXR or Isaac imports before this point.
    import torch
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / "src"))
    from wlr50_clean.ppo.semantic_rr_capture_local_actor import SemanticRRCaptureLocalHistoryMLPModel
    source_files, checks, all_observations = [], [], []
    for update_number in range(1, expected_updates + 1):
        path = run / "rollouts" / f"rollout_{update_number:04d}.pt"
        storage = torch.load(path, map_location="cpu", weights_only=False)
        rows = samples[(update_number-1)*512:update_number*512]
        obs = storage["observations"]["policy"].reshape(512, 447)
        all_observations.append(obs)
        check = dict(update=update_number)
        comparisons = dict(
            observations=(obs, [r["observation"] for r in rows]),
            raw_actions=(storage["actions"].reshape(512,12), [r["policy"]["selected_raw_full12"] for r in rows]),
            old_logp=(storage["actions_log_prob"].reshape(512), [r["old_logp"] for r in rows]),
            stored_mean=(storage["distribution_params"][0].reshape(512,12), [r["policy"]["conditional_mean_full12"] for r in rows]),
            stored_std=(storage["distribution_params"][1].reshape(512,12), [r["policy"]["active_conditional_std_full12"] for r in rows]),
            rewards=(storage["rewards"].reshape(512), [r["local_reward"]["reward"] for r in rows]),
            dones=(storage["dones"].reshape(512).float(), [float(r["local_reward"]["terminated"]) for r in rows]))
        for label, (actual, expected) in comparisons.items():
            check[label+"_max_abs_error"] = close_tensor(torch, actual, torch.tensor(expected,dtype=actual.dtype), label)
        mean, std = storage["distribution_params"]
        recomputed_logp = torch.distributions.Normal(mean,std).log_prob(storage["actions"]).sum(-1).reshape(512)
        check["recomputed_gaussian_logp_max_abs_error"] = close_tensor(torch, recomputed_logp,
            storage["actions_log_prob"].reshape(512), "Gaussian logp", 1e-4)
        raw_gae = (storage["returns"]-storage["values"]).reshape(512)
        expected_adv = (raw_gae-raw_gae.mean())/(raw_gae.std()+1e-8)
        check["whole_rollout_normalized_advantage_max_abs_error"] = close_tensor(torch,
            expected_adv,storage["advantages"].reshape(512),"advantage normalization")
        for i,row in enumerate(rows):
            for target,key in (("reward","rewards"),("return","returns"),("value","values"),("advantage","advantages")):
                row[target] = float(storage[key].reshape(512)[i])
            row["raw_gae"] = float(raw_gae[i])
        checks.append(check)
        source_files.append(dict(path=str(path),sha256=sha(path)))
    observations = torch.cat(all_observations)
    outputs, checkpoints = [], []
    for update_number in range(expected_updates+1):
        count = update_number*512
        path = OUTPUT/"checkpoints/history"/f"checkpoint_CP{225280+count}_local{count:06d}.pt"
        sidecar = path.with_name(path.stem+"_manifest.json")
        metadata = json.loads(sidecar.read_text())
        checksum = sha(path)
        if metadata["checkpoint_sha256"] != checksum or not metadata["save_load_round_trip"]:
            raise ValueError("checkpoint checksum/roundtrip mismatch")
        data = torch.load(path,map_location="cpu",weights_only=False)
        cfg = dict(metadata["runner_config"]["actor"])
        cfg.pop("class_name")
        if "legacy447_migration_only" in inspect.signature(SemanticRRCaptureLocalHistoryMLPModel.__init__).parameters:
            # Historical447 read-only analysis after the production448 version
            # is installed. Explicit old layout still comes from its sidecar;
            # this route only uses deterministic forward, never sampling/PPO.
            cfg["legacy447_migration_only"] = True
        actor = SemanticRRCaptureLocalHistoryMLPModel({"policy":observations[:1]},
            {"actor":["policy"]},"actor",12,**cfg)
        actor.load_state_dict(data["actor_state_dict"],strict=True)
        actor.eval()
        with torch.inference_mode():
            conditional = actor({"policy":observations},stochastic_output=False).clone()
            evidence = actor._last_forward_evidence
            local = evidence["local_raw_mean_delta"].clone()
            std = evidence["head_log_std"].exp().clone()
        actor.assert_frozen_state()
        outputs.append(dict(conditional=conditional,local=local,std=std))
        if update_number < expected_updates:
            start,end = count,count+512
            logged = torch.tensor([r["policy"]["conditional_mean_full12"] for r in samples[start:end]])
            checks[update_number]["same_checkpoint_cpu_mean_vs_collection_max_abs_error"] = close_tensor(torch,
                conditional[start:end],logged,"actual collection checkpoint reconstructed mean",1e-4)
        checkpoints.append(dict(update=update_number,path=str(path),sha256=checksum,
            manifest_sha256=sha(sidecar),counts=metadata["counts"],
            prior_hash=actor.expected_prior_state_sha256))
        del actor,data
    if len({x["prior_hash"] for x in checkpoints}) != 1:
        raise ValueError("frozen prior drift across checkpoints")
    groups = defaultdict(list)
    for row in samples:
        groups[row["bucket"]].append(row)
    fixed_comparisons = []
    for update_number in range(1,expected_updates+1):
        comparison = dict(update=update_number,own_collection_rows={},fixed_all_block_cohort={})
        for cohort_name,cohort in (("own_collection_rows",samples[(update_number-1)*512:update_number*512]),
                                   ("fixed_all_block_cohort",samples)):
            indices_by_bucket=defaultdict(list)
            for row in cohort:
                indices_by_bucket[row["bucket"]].append(row["index"])
            for bucket,indices in indices_by_bucket.items():
                pair = {}
                for joint,channel in RR.items():
                    old,new = outputs[update_number-1],outputs[update_number]
                    shift = new["conditional"][indices,channel]-old["conditional"][indices,channel]
                    pair[joint] = dict(conditional_mean_shift=stats(shift.tolist()),
                        local_head_mean_after=stats(new["local"][indices,channel].tolist()),
                        conditional_std_after=stats(new["std"][indices,channel].tolist()),
                        mean_shift_matches_user_candidate_fraction=float((shift < 0 if joint=="hip" else shift > 0).float().mean()),
                        note="Candidate direction only; hip decrease is not a necessary or sufficient physical success condition.")
                comparison[cohort_name][bucket]=dict(n=len(indices),RR=pair)
        fixed_comparisons.append(comparison)
    return dict(schema="wlr50_clean.rr_completed_block_learning_signal.v1",run=str(run),
        counts=dict(on_policy=len(samples),prefix_credit0=prefix,updates=len(updates),
                    optimizer_steps=sum(x["optimizer_steps"] for x in updates),AUX=0),
        scope=["Read-only CPU deserialization after explicit Isaac exit acknowledgement.",
               "No optimizer, model write, policy deployment, physics run, or teacher action.",
               "Buckets use decision endpoints; native hold duration is the logged native observer result.",
               "Headroom clipping is last dispatch of each decision, not all eight ticks.",
               "Saved advantages are whole-rollout normalized; raw GAE = returns - values.",
               "Same-observation mean shifts isolate parameter effects, not rollout distribution differences.",
               "Counterfactual fixed-state output is not a fresh deterministic closed-loop evaluation.",
               "Score-times-advantage is collection-distribution diagnostic, not actual clipped optimizer gradient.",
               "No positive advantage relabeling. No gradient-based capability claim. No AUX recommendation without physical evidence."],
        verification=checks,source_rollouts=source_files,checkpoints=checkpoints,
        aggregate=metric_summary(samples),physical_buckets={k:metric_summary(v) for k,v in groups.items()},
        per_update=[dict(update=i+1,metrics=metric_summary(samples[i*512:(i+1)*512]),
                         optimizer_receipt=updates[i]) for i in range(expected_updates)],
        fixed_state_mean_changes=fixed_comparisons,
        channel_KL=channel_kl_report(run,samples,updates),
        contact_event_rows=[{k:r[k] for k in ("index","episode","tick","time_s","phase","bucket","before","after","hold_s","local_success","reward","return","value","advantage","raw_gae","policy","final","controller_bias")} for r in samples if r["bucket"] in ("first_TOP_endpoint","TOP_reacquisition","bearing_drop_endpoint") or r["local_success"]])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,default=DEFAULT_RUN)
    parser.add_argument("--expected-updates",type=int,default=4)
    parser.add_argument("--isaac-stopped",action="store_true",help="Explicit acknowledgement from parent that Isaac exited; never use while live.")
    parser.add_argument("--json-kl-only",action="store_true",help="Standard-library analysis of already completed updates only; never loads Torch or models.")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.json_kl_only:
        samples,updates,prefix=collect(args.run.resolve(),args.expected_updates,allow_live_suffix=True)
        result=channel_kl_report(args.run.resolve(),samples,updates)
        result["counts"]=dict(on_policy=len(samples),prefix_credit0=prefix,updates=len(updates))
    else:
        result=analyze(args.run.resolve(),expected_updates=args.expected_updates,isaac_stopped=args.isaac_stopped)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output.resolve()),counts=result["counts"],
                          buckets={k:v["samples"] for k,v in result.get("physical_buckets",{}).items()})))


if __name__=="__main__":
    main()

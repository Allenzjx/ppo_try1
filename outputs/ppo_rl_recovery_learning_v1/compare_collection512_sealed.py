"""Bounded sealed512 versus sealed72e P12 episode1; no live polling/model forwards.

Default stdlib-only. --cpu-saved-tensors explicitly permits CPU tensor snapshot
reading AFTER Isaac exits; it never loads a policy checkpoint or computes values.
Writes only new report files; this module does nothing when imported.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import importlib.util
from itertools import islice
import json
import math
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / "runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0835078527430Z_g72e63592bdf4_eb09a77b75b0420997a80291feb869d8"


def helpers():
    spec = importlib.util.spec_from_file_location("sealed_coverage_helpers", OUT/"inspect_course.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats(values):
    values = list(values)
    return dict(n=len(values),mean=sum(values)/len(values),minimum=min(values),maximum=max(values),
                positive=sum(x>0 for x in values),negative=sum(x<0 for x in values)) if values else dict(n=0)


def snapshot(path, allow_cpu):
    if not allow_cpu:
        return None
    import torch  # Explicit offline flag only. Never import at module load.
    value = torch.load(path,map_location="cpu",weights_only=True)
    assert all(value[k].device.type == "cpu" for k in ("actions","actions_log_prob","returns","values"))
    return value


def analyze(run, *, old, allow_cpu, h):
    run = run.resolve(strict=True)
    manifest = h.small_json(run/"run_manifest.json")
    train = h.small_json(run/"training_manifest.json")
    h.require(manifest["lifecycle"] in ("SUCCEEDED","STOPPED_AT_VERIFIED_UPDATE_BOUNDARY"),
              "run must be sealed; do not analyze a live or failed partial buffer")
    length = train["runner_config"]["num_steps_per_env"]
    h.require(length == (128 if old else 512), "unexpected collection identity")
    if not old:
        h.require(train["actual_policy_decisions"] == 512 and train["ppo_updates_this_run"] == 1,
                  "this bounded comparison expects the first sealed512/one-update block only")
    updates = list(islice(h.rows(run/"optimizer_updates.jsonl"),4 if old else 1))
    audits = list(islice(h.rows(run/"advantage_audit.jsonl"),len(updates)))
    h.require(len(updates) == (4 if old else 1), "required complete updates not present")
    context_n = length * len(updates)
    data = list(islice(h.rows(run/"residual_and_projection_audit.jsonl"),context_n))
    h.require(len(data) == context_n == 512, "exact first512 completed context rows required")
    base = data[0]["global_policy_decision"]
    h.require([x["global_policy_decision"] for x in data] == list(range(base,base+512)),
              "noncontiguous learner rows")
    support = h.bound_support(manifest["runtime_contract"])
    terminal_indices = [i for i,row in enumerate(data) if row["terminal"]]
    first_end = terminal_indices[0]+1 if terminal_indices else 512
    if old:
        h.require(first_end == 452 and data[451]["global_policy_decision"] == 227012,
                  "old comparison must remain the known first452 episode")
    bounds, facts = [], []
    for row in data:
        a = row["applied_audit"]; task = a["semantic_task"]; ev = task["physical_evaluator"]
        legs = ev["current_legs"]
        flags, _ = h.physical_windows(task,support)
        if h.qualified(legs["RL"]): flags.add("RL_current_qualified")
        if legs["RR"].get("ground_contact"): flags.add("RR_current_ground")
        if h.bearing(legs["RR"],support["force_noise_floor_n"],top=True): flags.add("RR_legal_TOP_bearing")
        facts.append(dict(global_decision=row["global_policy_decision"],tick=a["physics_tick"],
            sim_time_s=a["sim_time_s"],request_phase=a["phase_id"],end_phase=a["end_phase_id"],
            terminal=row["terminal"],termination_reason=a.get("termination_reason"),
            reward=row["reward"],old_value=row["old_value"],old_logp=row["old_log_probability"],
            normalized_advantage=None,raw_GAE=None,stored_return=None,
            groups=sorted(flags),RR_contact=legs["RR"].get("contact_mode"),
            RR_bearing_force_n=legs["RR"].get("bearing_force_n"),
            RL_current_qualified=h.qualified(legs["RL"]),RL_qualified_tick=legs["RL"].get("current_lift_qualified_tick"),
            RL_bearing_force_n=legs["RL"].get("bearing_force_n"),
            RL_crossed=ev["history"]["front_edge_crossed"]["RL"],RL_placed=ev["history"]["placed"]["RL"],
            teacher_prefix_excluded=(a.get("prefix_teacher_data_in_ppo_storage") is False
                                    and a.get("prefix_checkpoint_policy_data_in_ppo_storage") is False)))
    for block,(update,audit) in enumerate(zip(updates,audits)):
        first,last = block*length,(block+1)*length
        rows = data[first:last]
        h.require(update["optimizer_steps"] == 20 and audit["sample_count"] == length
                  and update["ppo_update"] == audit["ppo_update_intended"]
                  and audit["first_global_policy_decision"] == rows[0]["global_policy_decision"]
                  and audit["last_global_policy_decision"] == update["global_policy_decisions"] == rows[-1]["global_policy_decision"]
                  and audit["teacher_prefix_samples_included"] is False,
                  "only complete actual matching updates earn credit")
        exposed = Counter(); max_logp_error = 0.; mb_count = 0
        for mb in h.minibatches(run/"rollouts"/f"update_{update['ppo_update']:06d}_likelihood.json"):
            mb_count += 1
            for j,indices in enumerate(mb["rollout_flat_indices"]):
                h.require(len(indices) == 1, "ambiguous raw sample identity")
                i = indices[0]; row = rows[i]; fact = facts[first+i]
                h.require(h.scalar(mb["old_log_probability"][j]) == row["old_log_probability"],
                          "old raw log probability mismatch")
                adv = h.scalar(mb["actual_advantage"][j])
                h.require(fact["normalized_advantage"] in (None,adv), "stored advantage changed between epochs")
                fact["normalized_advantage"] = adv
                err = abs(h.gaussian_logp(row["raw_policy_action_full12"],mb["current_conditional_mean"][j],
                    mb["current_conditional_sigma"][j])-h.scalar(mb["optimization_log_probability"][j]))
                max_logp_error = max(max_logp_error,err); exposed[i] += 1
        h.require(mb_count == 20 and exposed == Counter({i:5 for i in range(length)})
                  and max_logp_error < 1e-4, "raw/index/density/five-exposure audit failed")
        saved = snapshot(run/"rollouts"/f"rollout_{update['ppo_update']:06d}.pt",allow_cpu)
        if saved is not None:
            h.require(tuple(saved["actions"].shape) == (length,1,12), "saved collection shape differs")
            raw_gae = saved["returns"]-saved["values"]  # Actual stored Float32 subtraction.
            for i,row in enumerate(rows):
                h.require(saved["actions"][i,0].tolist() == row["raw_policy_action_full12"]
                    and saved["actions_log_prob"][i,0].item() == row["old_log_probability"]
                    and saved["rewards"][i,0].item() == row["reward"]
                    and saved["values"][i,0].item() == row["old_value"]
                    and saved["advantages"][i,0].item() == facts[first+i]["normalized_advantage"],
                    "saved tensor and issued-row/minibatch identities differ")
                facts[first+i].update(raw_GAE=raw_gae[i,0].item(),stored_return=saved["returns"][i,0].item())
        # Terminal raw GAE is already individually available in the stdlib JSON audit.
        for terminal in audit["terminal_samples"]:
            i = terminal["global_policy_decision"]-base
            facts[i].update(raw_GAE=terminal["raw_gae_returns_minus_old_values"],
                            stored_return=terminal["returns"])
        bounds.append(dict(update=update["ppo_update"],first_local=first+1,last_local=last,
            terminal_local_decisions=[i+1 for i in range(first,last) if data[i]["terminal"]],
            terminal_samples=audit["terminal_samples"],phase_samples=dict(Counter(r["applied_audit"]["phase_id"] for r in rows)),
            gamma=audit["gamma"],lambda_=audit["lambda"],overall=audit["overall"],
            tail_bootstrap=audit["tail_bootstrap"],teacher_prefix_samples_included=False,
            five_exposures_verified=True,raw_tensor_equality_checked=saved is not None,
            maximum_current_density_error=max_logp_error,optimizer_steps=update["optimizer_steps"],
            value_loss=update["value_loss"],kl_mean=update["kl_mean"],clip_fraction=update["clip_fraction"]))
    def scope(items):
        groups = defaultdict(list)
        for fact in items:
            for name in ("all",*fact["groups"]): groups[name].append(fact)
        return dict(samples=len(items),phase_samples=dict(Counter(x["request_phase"] for x in items)),
            terminal_count=sum(x["terminal"] for x in items),
            physical_windows={name:dict(samples=len(group),reward=stats(x["reward"] for x in group),
                old_value=stats(x["old_value"] for x in group),
                normalized_advantage=stats(x["normalized_advantage"] for x in group),
                raw_GAE=stats(x["raw_GAE"] for x in group if x["raw_GAE"] is not None),
                raw_GAE_scope="all saved rows" if allow_cpu else "terminal rows only; do not invert normalization")
                for name,group in sorted(groups.items())})
    first_facts = facts[:first_end]
    selected = {0,first_end-1}
    for name in ("RL_current_qualified","qualified_RL_edge_recovery","RL_qualified_AIR_capture_region",
                 "RL_actual_TOP_bearing","RR_current_ground","RR_legal_TOP_bearing"):
        indices = [i for i,f in enumerate(first_facts) if name in f["groups"]]
        if indices: selected.update((indices[0],indices[-1]))
    selected.update(i for i,f in enumerate(first_facts) if f["terminal"])
    handoff = data[0]["applied_audit"].get("curriculum_start",{})
    return dict(run=str(run),source_checkpoint=manifest["arguments"].get("checkpoint"),
        runtime=manifest["runtime_contract"]["source_git_commit"],collection_length=length,
        total_context_samples=512,first_episode_samples=first_end,
        first_episode_terminal_observed=bool(terminal_indices),
        first_episode_terminal_in_first_rollout=(bool(terminal_indices) and terminal_indices[0]<length),
        learner_handoff=handoff,first_episode=scope(first_facts),
        whole_new512_block=None if old else scope(facts),
        RL_qualification_ticks_in_first_episode=sorted({x["RL_qualified_tick"] for x in first_facts if type(x["RL_qualified_tick"]) is int}),
        inherited_RL_qualification_ticks=sorted({x["RL_qualified_tick"] for x in first_facts
            if type(x["RL_qualified_tick"]) is int and x["RL_qualified_tick"]<=handoff.get("physics_tick",-1)}),
        first_episode_terminal=facts[first_end-1] if terminal_indices else None,
        complete_updates=bounds,selected_first_episode_rows=[first_facts[i] for i in sorted(selected)],
        ordinary_phase_changes=sum(x["request_phase"]!=x["end_phase"] and not x["terminal"] for x in facts),
        all_prefix_storage_flags_excluded=all(x["teacher_prefix_excluded"] for x in facts),
        old_fourth_rollout_scope="Its60 next-episode rows are read only to bind complete-update likelihood/tensors; first-episode metrics exclude them" if old else None)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-run",type=Path,required=True)
    parser.add_argument("--cpu-saved-tensors",action="store_true",
        help="explicit offline CPU snapshot access only after sole Isaac exits; never loads model checkpoints")
    args=parser.parse_args(argv)
    h=helpers()
    old=analyze(OLD,old=True,allow_cpu=args.cpu_saved_tensors,h=h)
    new=analyze(args.new_run,old=False,allow_cpu=args.cpu_saved_tensors,h=h)
    h.require((old["complete_updates"][0]["gamma"],old["complete_updates"][0]["lambda_"])
              == (new["complete_updates"][0]["gamma"],new["complete_updates"][0]["lambda_"]),
              "return estimator changed; not the intended comparison")
    report=dict(schema="readonly.sealed512_vs72e_P12episode1.v1",
        no_model_checkpoint_load=True,no_model_forward=True,no_production_changes=True,
        tensor_snapshot_CPU_access=args.cpu_saved_tensors,old=old,new=new,
        interpretation=[
            "This is an unequal-policy/unequal-training-history comparison, not a controlled causal rollout-length ablation.",
            "Old72e first452 ends in fourth128 rollout; its firstthree tails bootstrap. The sealed fourth update is now read, unlike the older pre-seal report.",
            "Actual new terminal coverage is reported, not assumed from512: a block with no terminal has not gained observed terminal credit.",
            "Positive GAE/standardized advantage is not physical success or proof a policy learned to retreat; minibatch gradients and clipping are shared.",
            "FR-axis positive projection with RL fraction decline is not verified lateral transfer or absolute-force unload.",
            "No old128 blocks are concatenated into a counterfactual on-policy512; same gamma/lambda leaves GAE half-life about4s.",
            "Prefix-owned qualification/contact is separated from current learner state; suffix progress is not naturalP01 success.",
            "Default per-row nonterminal rawGAE is unavailable; full exact saved values require explicit offline CPU snapshot flag."])
    print(json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2))


if __name__ == "__main__":
    main()

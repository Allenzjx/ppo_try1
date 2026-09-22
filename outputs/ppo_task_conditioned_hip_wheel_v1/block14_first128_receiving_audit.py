"""Read-only audit of one sealed actual rollout/update; no forward/optimizer/Isaac."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import torch
torch.set_num_threads(1)
from wlr50_clean.ppo.semantic_training import state_hash
from wlr50_clean.ppo.semantic_history_actor import cap_transition_request_history
from wlr50_clean.ppo.semantic_receiving_wheel_sigma import receiving_wheel_effective_log_std
from wlr50_clean.ppo.semantic_receiving_wheel_profile import RECEIVING_WHEEL_POLICY
from wlr50_clean.ppo.semantic_migration import validate_migration_plan, digest

OUT = ROOT / "outputs/ppo_task_conditioned_hip_wheel_v1"
RUN = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T1355291518150Z_g649ccd906421_aac78369d59846409cb2d4039083e87a"
SOURCE = OUT / "checkpoints/history/checkpoint_step_000197120.pt"
TARGET = OUT / "checkpoints/history/checkpoint_step_000197248.pt"
PLAN = OUT / "checkpoint197120_receiving_sigma_migration.json"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding="utf-8"))
def tensor(value):return torch.tensor(value,dtype=torch.float32)


def parameter_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        if name.startswith("obs_normalizer."):continue
        v=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def main():
    target_manifest=TARGET.with_name(TARGET.stem+"_manifest.json")
    if not target_manifest.is_file():
        print(json.dumps({"status":"PENDING_IMMUTABLE_FIRST_CHECKPOINT","actual_credit_reported":0}));return
    sm=read(SOURCE.with_name(SOURCE.stem+"_manifest.json"));tm=read(target_manifest)
    assert tm["source_run"]==str(RUN.resolve())
    assert sha(SOURCE)==sm["checkpoint_sha256"] and sha(TARGET)==tm["checkpoint_sha256"]
    source=torch.load(SOURCE,map_location="cpu",weights_only=False)
    target=torch.load(TARGET,map_location="cpu",weights_only=False)
    for payload,metadata in ((source,sm),(target,tm)):
        assert all(metadata[k]==v for k,v in payload["infos"].items())
        assert parameter_hash(payload["actor_state_dict"])==metadata["actor_parameter_sha256"]
        assert parameter_hash(payload["critic_state_dict"])==metadata["critic_parameter_sha256"]
        assert state_hash(payload["optimizer_state_dict"])==metadata["optimizer_state_sha256"]
        assert all(not key.startswith("obs_normalizer.") for role in ("actor","critic")
                   for key in payload[role+"_state_dict"])
        assert state_hash({"actor":{},"critic":{}})==metadata["normalizer_state_sha256"]
    verified=validate_migration_plan(SOURCE,tm["runtime_contract"],PLAN,project_root=ROOT)
    factor=verified["receiving_wheel_sigma_factor"]
    receipt=tm["receiving_wheel_sigma_migration"]
    expected_receipt={"factor":factor,**{k:verified[k] for k in ("plan_path","plan_sha256",
        "source_checkpoint_sha256","source_contract_sha256","target_contract_sha256")}}
    assert receipt==expected_receipt
    assert tm["policy_contract"]==factor["target_policy_contract"] and tm["policy_contract"]["version"]==RECEIVING_WHEEL_POLICY
    assert sm["policy_contract"]==factor["source_policy_contract"]
    assert factor["preserved_training_state"]=={k:sm[k] for k in factor["preserved_training_state"]}
    assert factor["source_effective_learning_rate"]==factor["target_effective_learning_rate"]==sm["optimizer_learning_rate"]==1e-5
    for key in ("global_policy_decisions","ppo_updates","optimizer_steps"):
        assert factor["counter_origin"][key]==sm[key]
    assert [tm[k]-sm[k] for k in ("global_policy_decisions","ppo_updates","optimizer_steps")]==[128,1,20]
    assert tm["task_conditioned_hip_wheel_branch"]==sm["task_conditioned_hip_wheel_branch"]
    assert tm["training_quantity_budget_extension"]==sm["training_quantity_budget_extension"]
    assert tm["stage_requested_decisions"]=={**sm["stage_requested_decisions"],
        "full_episode":sm["stage_requested_decisions"]["full_episode"]+128}
    for key in sm:
        if key.endswith("_branch_counts"):
            assert [tm[key][field]-sm[key][field] for field in
                    ("global_policy_decisions","ppo_updates","optimizer_steps")]==[128,1,20]
    ledger=tm["task_conditioned_hip_wheel_branch"]["auxiliary_mean_learning"]
    assert digest(ledger)==factor["preserved_auxiliary_ledger_sha256"]
    assert ledger["accepted_auxiliary_updates_total"]==7 and ledger["attempted_auxiliary_optimizer_steps_total"]==8
    before_opt,after_opt=source["optimizer_state_dict"],target["optimizer_state_dict"]
    assert before_opt["state"].keys()==after_opt["state"].keys()
    adam_steps=[]
    for key,before in before_opt["state"].items():
        after=after_opt["state"][key]
        assert before.keys()==after.keys()
        assert set(before)>={"step","exp_avg","exp_avg_sq"}
        assert float(after["step"])-float(before["step"])==20
        adam_steps.append([float(before["step"]),float(after["step"])])
    assert len(before_opt["param_groups"])==len(after_opt["param_groups"])
    for before,after in zip(before_opt["param_groups"],after_opt["param_groups"]):
        assert {k:v for k,v in before.items() if k!="lr"}=={k:v for k,v in after.items() if k!="lr"}
        assert before["lr"]==sm["optimizer_learning_rate"] and after["lr"]==tm["optimizer_learning_rate"]
    rows=[]
    with (RUN/"residual_and_projection_audit.jsonl").open(encoding="utf-8") as stream:
        for _ in range(128):rows.append(json.loads(next(stream)))
    update=tm["last_update"]  # Immutable checkpoint evidence even if a live JSONL buffer is not visible yet.
    assert update["ppo_update"]==1506 and update["global_policy_decisions"]==197248
    assert update["actor_parameter_sha256_before"]==sm["actor_parameter_sha256"]
    assert update["actor_parameter_sha256_after"]==tm["actor_parameter_sha256"]
    rollout=RUN/"rollouts/rollout_001506.pt"
    saved=torch.load(rollout,map_location="cpu",weights_only=False)
    assert saved["policy_contract"]==tm["policy_contract"] and saved["runtime_contract"]==tm["runtime_contract"]
    assert tuple(saved["actions"].shape)==(128,1,12) and tuple(saved["observations"]["policy"].shape)==(128,1,372)
    assert saved["curriculum_epoch"]["prefix_request"] is None
    assert saved["curriculum_epoch"]["reset_sampling"]=="P01_full_task_only_initial_version"
    phases=Counter();active=0;physics_ticks=0;verified_ticks=0;native_mask_paths={}
    request_sigma_error=0.; current_sigma_error=0.; current_mu_error=0.; current_logp_error=0.
    def masks(value,path=""):
        if isinstance(value,dict):
            for key,item in value.items():
                current=path+"."+key
                if "mask" in key and isinstance(item,(list,tuple)) and len(item)==12:
                    native_mask_paths.setdefault(current,set()).add(tuple(item))
                elif key!="semantic_task":masks(item,current)
    for i,row in enumerate(rows):
        assert row["global_policy_decision"]==197121+i
        req,applied=row["policy_request"],row["applied_audit"]
        assert req["policy_version"]==RECEIVING_WHEEL_POLICY and req["sampling_draws"]==1
        assert req["extra_model_forwards"]==req["extra_random_draws"]==0
        assert torch.equal(tensor(req["selected_raw_full12"]),saved["actions"][i,0])
        assert torch.equal(tensor(req["conditional_mean_full12"]),saved["distribution_params"][0][i,0])
        assert torch.equal(tensor(req["effective_sigma_full12"]),saved["distribution_params"][1][i,0])
        assert req["selected_raw_log_probability"]==row["old_log_probability"]==saved["actions_log_prob"][i,0,0].item()
        x=saved["observations"]["policy"][i]
        stage=int(x[0,:13].argmax());gate=stage in (9,10,11) and x[0,157].item()==1
        assert req["stage_index"]==stage and req["receiving_continuation_active"]==gate
        expected=torch.ones(12)
        if gate:expected[[9,11]]=3;active+=1
        assert torch.equal(tensor(req["receiving_sigma_multiplier_full12"]),expected)
        assert bool((torch.isfinite(saved["distribution_params"][1][i]) & (saved["distribution_params"][1][i]>0)).all())
        inferred=tensor(req["learned_sigma_full12"])*.25*tensor(req["innovation_sigma_multiplier_full12"])*expected
        error=(inferred-tensor(req["effective_sigma_full12"])).abs().max().item()
        request_sigma_error=max(request_sigma_error,error)
        torch.testing.assert_close(inferred,tensor(req["effective_sigma_full12"]),rtol=2e-6,atol=1e-8)
        phases[applied["phase_id"]]+=1;physics_ticks+=applied["physics_ticks"]
        verified_ticks+=applied["actuator_target_effect_audit_summary"]["verified_tick_count"]
        masks(applied["actuator_target_effect_audit"],"native")
    likelihood_path=RUN/"rollouts/update_001506_likelihood.json"
    likelihood=read(likelihood_path)
    assert likelihood["extra_model_forwards"]==likelihood["extra_random_draws"]==0 and len(likelihood["minibatches"])==20
    entropies=[];coverage=Counter();first_ratio_error=0.
    for batch in likelihood["minibatches"]:
        indices=[matches[0] for matches in batch["rollout_flat_indices"]]
        coverage.update(indices)
        x=saved["observations"]["policy"][indices,0]
        mean,sigma=tensor(batch["current_conditional_mean"]),tensor(batch["current_conditional_sigma"])
        history,_=cap_transition_request_history(x)
        expected_mu=.1*tensor(batch["current_network_mean_full12"])+.9*history
        logs,_=receiving_wheel_effective_log_std(tensor(batch["current_network_log_sigma_full12"]),x,.25)
        current_mu_error=max(current_mu_error,(expected_mu-mean).abs().max().item())
        current_sigma_error=max(current_sigma_error,(logs.exp()-sigma).abs().max().item())
        torch.testing.assert_close(mean,expected_mu,rtol=2e-6,atol=1e-7)
        torch.testing.assert_close(sigma,logs.exp(),rtol=2e-6,atol=1e-8)
        assert batch["sigma_source"]=="current_official_Gaussian_cache_after_B_over_cap_and_receiving_FR_RR_sigma_x3"
        gaussian=torch.distributions.Normal(mean,sigma)
        lp=gaussian.log_prob(saved["actions"][indices,0]).sum(-1)
        actual=tensor(batch["optimization_log_probability"])
        current_logp_error=max(current_logp_error,(lp-actual).abs().max().item())
        torch.testing.assert_close(lp,actual,rtol=3e-6,atol=1e-5)
        old_lp=saved["actions_log_prob"][indices,0,0]
        assert torch.equal(tensor(batch["old_log_probability"]),old_lp)
        if batch["minibatch_index"]==0:first_ratio_error=max(abs(v-1) for v in batch["ratio"])
        entropies.append(float(gaussian.entropy().sum(-1).mean()))
    assert coverage==Counter({i:5 for i in range(128)})
    entropy=sum(entropies)/len(entropies)
    assert abs(entropy-update["entropy"])<2e-5
    result={"schema":"wlr50_clean.block14_first128_receiving_readonly_audit.v1","status":"PASS",
        "scope":"one sealed actual rollout/update; no model forward, optimization, simulation or file mutation",
        "source":{"path":str(SOURCE),"sha256":sm["checkpoint_sha256"],"policy_version":sm["policy_contract"]["version"]},
        "checkpoint":{"path":str(TARGET),"sha256":tm["checkpoint_sha256"],"policy_version":RECEIVING_WHEEL_POLICY,
            "global_policy_decisions":tm["global_policy_decisions"],"ppo_updates":tm["ppo_updates"],"optimizer_steps":tm["optimizer_steps"]},
        "migration":{"plan":str(PLAN),"sha256":sha(PLAN),"official_revalidation_passed":True,"persisted_receipt_exact":True,
            "all_source_saved_tensor_hashes_verified":True,"source_actual_Adam_LR":sm["optimizer_learning_rate"],
            "post_update_Adam_LR":tm["optimizer_learning_rate"],"Adam_groups":len(before_opt["param_groups"]),
            "Adam_parameter_states":len(adam_steps),"Adam_steps_each_increment":20,
            "pre_update_actor_hash_equals_source":True,"Identity_state_unchanged":True,"AUX_accepted_attempted":[7,8],
            "original_branch_quantity_receipt_preserved":True,
            "stage_spending_after":tm["stage_requested_decisions"],
            "all_inherited_branch_counters_increment":[128,1,20],
            "restore_evidence_limit":"Full actor/critic/Adam/Identity hashes and RNG restore are enforced by live loader before sampling; first-update actor-before hash and surviving Adam steps independently corroborate. No separate pre-update Adam/RNG memory snapshot was saved; no GPU RNG replay was performed.",
            "fresh_rollout_evidence":"loader requires step=0/transition.actions=None; first snapshot is 128 new decisions 197121..197248 with new profile/runtime and natural-P01 state, no source rollout imported"},
        "actual_new_credit":{"policy_decisions":128,"PPO_updates":1,"optimizer_steps":20,"auxiliary_updates":0,
            "phase_counts":{f"P{i:02d}":phases[f"P{i:02d}"] for i in range(1,14)},"terminal_decisions":sum(bool(r["terminal"]) for r in rows)},
        "sampling":{"rows":128,"stored_raw_mean_sigma_logp_exact":True,"full12_positive_sigma":True,
            "receiving_x3_active_samples":active,"request_sigma_identity_max_abs_error":request_sigma_error,
            "one_draw_no_extra_forward_or_rng":True,"native_physics_ticks":physics_ticks,"verified_native_ticks":verified_ticks,
            "observed_mask_values":{k:[list(row) for row in sorted(v)] for k,v in native_mask_paths.items()}},
        "actual_update":{"minibatches":20,"each_saved_sample_seen":5,"current_mean_max_abs_error":current_mu_error,
            "current_sigma_max_abs_error":current_sigma_error,"current_logp_CPU_reconstruction_max_abs_error":current_logp_error,
            "first_minibatch_ratio_max_error_from_one":first_ratio_error,"reported_entropy":update["entropy"],
            "CPU_same_sigma_entropy":entropy,"mean_KL":update["kl_mean"],"clip_fraction":update["clip_fraction"]},
        "limits":["This is migration/probability/credit evidence, not physical success or improved receiving support.",
                  "If receiving_x3_active_samples is zero, this first batch verifies the unchanged branch of the new profile only; actual P10-P12 activation remains to be observed."]}
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()

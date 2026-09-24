"""Focused candidates. Numerical tests must wait for the Isaac-free boundary."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from wlr50_clean.ppo import semantic_front_retention439 as m
from wlr50_clean.ppo.semantic_migration import digest, file_sha

ROOT = Path(__file__).resolve().parents[2]


def source_path():
    explicit = os.environ.get("WLR_FRONT439_SOURCE_CHECKPOINT")
    if explicit:
        return Path(explicit)
    pointer = ROOT / "outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/checkpoint_last_pointer.json"
    return Path(json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"])


def source_metadata():
    source = source_path()
    result = json.loads(source.with_name(source.stem + "_manifest.json").read_text(encoding="utf-8"))
    if result["runtime_contract"]["source_git_commit"] != m.SOURCE_HEAD:
        pytest.skip("set WLR_FRONT439_SOURCE_CHECKPOINT to the actual selected completed72e source; never guess an ancestor")
    return result


def target_contract(old):
    new = copy.deepcopy(old)
    new["source_git_commit"] = "e" * 40
    for path in m.ALLOWED:
        new["files"][path] = file_sha(ROOT / path)
    new["runtime_content_sha256"] = digest(new["files"])
    return new


def actual_plan():
    path = source_path()
    source = source_metadata()
    target = target_contract(source["runtime_contract"])
    record = m.build_front_retention439_identity(path,target,reason="focused same439 accounting identity",
        expected_source_sha256=file_sha(path),expected_manifest_sha256=file_sha(path.with_name(path.stem+"_manifest.json")))
    return source,target,record


def test_old_without_ledger_and_new_identity_preserve_historical_receipts():
    from wlr50_clean.ppo.semantic_rear_policy_timing_migration import validate_rear_policy_namespace
    source,target,record = actual_plan()
    route = source["checkpoint_output_routing"]
    validate_rear_policy_namespace(source,source["runtime_contract"],Path(route["output_root"]),checkpoint_output_routing=route)
    candidate = {**copy.deepcopy(source),"runtime_contract":target,m.IDENTITY:record}
    validate_rear_policy_namespace(candidate,target,Path(route["output_root"]),checkpoint_output_routing=route)
    assert m.previous_contract(target,record) == source["runtime_contract"]
    assert set(record["changed_file_hashes"]) == m.ALLOWED
    assert record[m.FACTOR_KEY]["revision_counter_origin"] == {k:source[k] for k in m.COUNTERS}
    assert candidate["rr_retention_reward_migration"] == source["rr_retention_reward_migration"]
    assert candidate["rear_owner_recovery_migration"] == source["rear_owner_recovery_migration"]
    assert candidate["rear_policy_timing_branch"] == source["rear_policy_timing_branch"]
    assert m.LEDGER not in candidate


@pytest.mark.parametrize("change",["source_sha","manifest_sha","config","reward","missing_runtime_path"])
def test_identity_rejects_source_and_runtime_tampering(change):
    path = source_path(); source = source_metadata(); target = target_contract(source["runtime_contract"])
    kwargs = dict(expected_source_sha256=file_sha(path),expected_manifest_sha256=file_sha(path.with_name(path.stem+"_manifest.json")))
    if change == "source_sha": kwargs["expected_source_sha256"] = "0"*64
    elif change == "manifest_sha": kwargs["expected_manifest_sha256"] = "0"*64
    elif change == "config": target["training_budgets"]["phase_suffix"] += 128
    elif change == "reward": target["files"][m.CODE+"semantic_reward.py"] = "0"*64
    else: target["files"][m.CODE+"semantic_training.py"] = source["runtime_contract"]["files"][m.CODE+"semantic_training.py"]
    with pytest.raises(ValueError):
        m.build_front_retention439_identity(path,target,reason="reject",**kwargs)


@pytest.mark.parametrize("change",["counter","adam_count","branch","old_owner","old_reward","origin","ledger_without_identity","uncredited_actor","uncredited_adam"])
def test_lineage_counter_branch_and_receipt_mutations_fail(change):
    source,target,record = actual_plan(); route = source["checkpoint_output_routing"]
    row = {**copy.deepcopy(source),"runtime_contract":target,m.IDENTITY:copy.deepcopy(record)}
    if change == "counter": row["global_policy_decisions"] -= 128
    elif change == "adam_count": row["optimizer_steps"] += 1
    elif change == "branch": row["checkpoint_output_routing"]["branch"] = "wrong"
    elif change == "old_owner": row["rear_owner_recovery_migration"]["reason"] += "changed"
    elif change == "old_reward": row["rr_retention_reward_migration"]["reason"] += "changed"
    elif change == "origin": row[m.IDENTITY][m.FACTOR_KEY]["revision_counter_origin"]["ppo_updates"] -= 1
    elif change == "uncredited_actor": row["actor_parameter_sha256"]="0"*64
    elif change == "uncredited_adam": row["optimizer_state_sha256"]="0"*64
    else: row.pop(m.IDENTITY);row[m.LEDGER] = m._empty_ledger()
    with pytest.raises(ValueError):
        m.validate_front_retention439_lineage(row,target,Path(route["output_root"]),checkpoint_output_routing=route)


def write_bound(path,value):
    path.write_text(json.dumps(value,sort_keys=True),encoding="utf-8")
    return {"path":str(path.resolve()),"sha256":file_sha(path)}


@pytest.fixture
def evidence(tmp_path,monkeypatch):
    # Synthetic references explicitly test accounting only, not physical outcomes.
    bindings = {}
    for key in m.PROBE_HASHES:
        bindings[key] = write_bound(tmp_path/(key+".json"),{"fixture":key})
    monkeypatch.setattr(m,"PROBE_HASHES",{k:v["sha256"] for k,v in bindings.items()})
    data = dict(schema=m.DATA_SCHEMA,observation_dimension=439,action_dimension=12,
        phase_columns=m.COLUMNS,decision_index_cutoff_inclusive=997,
        target_semantics="executed_student_conditional_raw_request",synthetic_or_intervention_rows_used=False,
        step_applied_action_validated_as="finite_physical_full12_not_raw_label",
        teacher_deployed=False,whole_failed_trajectory_used_as_success_label=False,targets_are_physical_success_labels=False,
        PPO_credit=0,new_AUX_credit=0,selected_rows_content_sha256="6"*64,
        train_row_ids=[2,347,646,844],holdout_row_ids=[10,355,660,854],**bindings)
    data["receipt_content_sha256"]=digest(data)
    source = dict(actor_parameter_sha256="1"*64,critic_parameter_sha256="2"*64,
        optimizer_state_sha256="3"*64,normalizer_state_sha256="4"*64,
        optimizer_learning_rate=1e-5,training_rng_state={"fixture":True},
        global_policy_decisions=234496,ppo_updates=1797,optimizer_steps=35940)
    data_binding = write_bound(tmp_path/"data.json",data)
    report = dict(schema=m.FIT_SCHEMA,accepted_auxiliary_updates=1,attempted_auxiliary_optimizer_steps=2,
        optimized_parameters=m.PARAMETERS,optimized_scalar_count=1024,
        actor_parameter_sha256_before=source["actor_parameter_sha256"],actor_parameter_sha256_after="5"*64,
        protected_state_before=m._state(source),protected_state_after=m._state(source),
        real_P10_P12_full_Gaussian_bitwise_unchanged=True,
        synthetic_P13_full_Gaussian_bitwise_unchanged=True,
        zero_selected_phase_columns_P10_P13_algebra_verified=True,nonselected_actor_state_unchanged=True,
        invariance_evidence=dict(schema="wlr50_clean.front_retention439.real_P10_P12_synthetic_P13_invariance.v1",
            actual_rows_by_phase={"P10":1,"P11":1,"P12":1,"P13":0},
            real_P13_validation_claimed=False,synthetic_P13_rows=3,
            synthetic_P13_origin="first_real_row_per_P10_P12_with_only_phase_onehot_replaced_test_fixture",
            synthetic_rows_used_for_fit=False,selected_phase_columns=[1,4,5,8],
            real_observations_float32_sha256="a"*64,synthetic_P13_observations_float32_sha256="b"*64,
            same_future_trajectory_claimed=False),
        fresh_PPO_rollout_required=True,first_rejected_proposal_restored_and_stopped=True,
        stop_reason="first_rejected_proposal_restored_and_stopped",targets_were_executed_student_raw_requests=True,
        targets_were_success_or_teacher_labels=False,teacher_deployed=False,physical_success_claimed=False,
        same_future_trajectory_claimed=False,
        PPO_decisions_added=0,PPO_updates_added=0,PPO_optimizer_steps_added=0,data_receipt=data_binding,
        budget=dict(max_attempts=2,learning_rate=.001,maximum_train_request_shift_full12=[1.]*12,
            maximum_validation_request_shift_full12=[1.]*12,maximum_per_state_full_gaussian_kl=.01,
            maximum_abs_log_sigma_change=.02))
    report_binding = write_bound(tmp_path/"fit.json",report)
    return source,data,data_binding,report,report_binding


def test_exact_sparse_aux_event_has_separate_credit(evidence):
    source,_,data,_,report = evidence
    result = m._event(source,{"fixture":"parent"},data,report,1)
    assert result["optimized_scalar_count"] == 1024
    assert all(result[k] == 0 for k in m.PPO_ZERO)
    assert result["accepted_auxiliary_updates"] == 1
    assert result["attempted_auxiliary_optimizer_steps"] == 2
    assert result["RR_capture_success_claimed"] is False
    assert result["kind"] == "executed_P02_P05_P06_P09_retention_sparse_phase_columns_not_PPO"


@pytest.mark.parametrize("reason",["finite_budget_exhausted","selected_student_raw_targets_already_match","zero_sparse_gradient_no_optimizer_step"])
def test_finite_successful_fit_need_not_have_rejected_proposal(evidence,reason):
    source,_,data,report,binding=evidence
    report["stop_reason"]=reason
    report["first_rejected_proposal_restored_and_stopped"]=False
    binding=write_bound(Path(binding["path"]),report)
    assert m._event(source,{"fixture":"parent"},data,binding,1)["accepted_auxiliary_updates"]==1


@pytest.mark.parametrize("change",["cutoff","intervention","leak","sparse_phase","source","file_bytes","digest","imbalanced","teacher"])
def test_data_receipt_rejects_cutoff_partition_and_source_tampering(evidence,change):
    _,data,binding,_,_ = evidence
    if change == "cutoff": data["decision_index_cutoff_inclusive"] = 998
    elif change == "intervention": data["train_row_ids"].append(998)
    elif change == "leak": data["holdout_row_ids"].append(2)
    elif change == "sparse_phase": data["train_row_ids"].append(842)
    elif change == "source": data["source_decisions"]["sha256"] = "0"*64
    elif change == "digest": data["receipt_content_sha256"]="0"*64
    elif change == "imbalanced": data["train_row_ids"].append(3)
    elif change == "teacher": data["teacher_deployed"]=True
    else:
        Path(data["source_decisions"]["path"]).write_text("changed",encoding="utf-8")
    if change != "digest":
        data["receipt_content_sha256"]=digest({k:v for k,v in data.items() if k!="receipt_content_sha256"})
    binding = write_bound(Path(binding["path"]),data)
    with pytest.raises(ValueError): m._data(binding)


@pytest.mark.parametrize("change",["scope","critic","ppo_credit","counter_type","data_binding","missing_invariance","budget","stop","teacher","future_claim"])
def test_fit_receipt_cannot_expand_authority_or_hide_mutation(evidence,change):
    source,_,data,report,binding = evidence
    if change == "scope": report["optimized_parameters"] = ["actor.mlp.0.weight[:,0:9]"]
    elif change == "critic": report["protected_state_after"]["critic_parameter_sha256"] = "9"*64
    elif change == "ppo_credit": report["PPO_updates_added"] = 1
    elif change == "counter_type": report["accepted_auxiliary_updates"] = True
    elif change == "data_binding": report["data_receipt"] = {**data,"sha256":"0"*64}
    elif change == "missing_invariance": report["real_P10_P12_full_Gaussian_bitwise_unchanged"] = False
    elif change == "stop": report["first_rejected_proposal_restored_and_stopped"] = False
    elif change == "teacher": report["teacher_deployed"] = True
    elif change == "future_claim": report["same_future_trajectory_claimed"] = True
    else: report["budget"]["max_attempts"] = 33
    binding = write_bound(Path(binding["path"]),report)
    with pytest.raises(ValueError): m._event(source,{"fixture":"parent"},data,binding,1)


@pytest.mark.parametrize("key,value", [
    ("actual_rows_by_phase", {"P10":1,"P11":1,"P12":1,"P13":1}),
    ("real_P13_validation_claimed", True), ("synthetic_P13_rows", 0),
    ("synthetic_rows_used_for_fit", True), ("selected_phase_columns", list(range(9))),
])
def test_invariance_receipt_does_not_claim_synthetic_P13_as_real(evidence,key,value):
    source,_,data,report,binding = evidence
    report["invariance_evidence"][key] = value
    binding = write_bound(Path(binding["path"]),report)
    with pytest.raises(ValueError): m._event(source,{"fixture":"parent"},data,binding,1)


def test_checkpoint_event_binding_rejects_sidecar_or_source_changes(tmp_path,monkeypatch):
    checkpoint=tmp_path/"source.pt";checkpoint.write_bytes(b"synthetic checkpoint binding only")
    sidecar=checkpoint.with_name(checkpoint.stem+"_manifest.json")
    sidecar.write_text("{}",encoding="utf-8")
    binding=dict(checkpoint=str(checkpoint.resolve()),checkpoint_sha256=file_sha(checkpoint),
        manifest=str(sidecar.resolve()),manifest_sha256=file_sha(sidecar))
    def checked(path,cp_sha,manifest_sha):
        if cp_sha!=file_sha(path) or manifest_sha!=file_sha(sidecar): raise ValueError("changed")
        return {"synthetic":True}
    monkeypatch.setattr(m,"_checkpoint",checked)
    assert m._checkpoint_reference(binding)=={"synthetic":True}
    for key in ("checkpoint_sha256","manifest_sha256","manifest"):
        bad={**binding,key: "0"*64 if key.endswith("sha256") else str(tmp_path/"other.json")}
        with pytest.raises(ValueError): m._checkpoint_reference(bad)


def test_runtime_wiring_does_not_modify_cli_or_old_receipts():
    import ast
    tree = ast.parse((ROOT/(m.CODE+"semantic_training.py")).read_text(encoding="utf-8"))
    functions = {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    strings = lambda node:{n.value for n in ast.walk(node) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    assert m.FACTOR_KEY in strings(functions["load_semantic_checkpoint"])
    for function in ("save_semantic_checkpoint","load_semantic_checkpoint","train_semantic"):
        assert m.IDENTITY in strings(functions[function]) and m.LEDGER in strings(functions[function])
    assert len(m.ALLOWED) == 3
    assert m.CODE+"semantic_cli.py" not in m.ALLOWED
    assert m.CODE+"semantic_rr_retention_migration.py" not in m.ALLOWED


@pytest.fixture
def cpu_state(tmp_path,monkeypatch):
    # Numeric identity/carry tests intentionally isolate ancestral GPU receipts.
    # The real unmodified receipt/source tests above cover that separate boundary.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","-1")
    torch = pytest.importorskip("torch");pytest.importorskip("rsl_rl")
    assert not torch.cuda.is_available(), "run numerical tests only after Isaac exits"
    from wlr50_clean.ppo import semantic_training as t
    from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
    old_threads,old_rng = torch.get_num_threads(),torch.get_rng_state()
    torch.set_num_threads(1)
    def make(env=None):
        return t.construct_semantic_runner(env or _shape_env(439,"cpu"),seed=1001,device="cpu",
            policy_version=m.REAR_OWNER_POLICY,observation_layout=m.REAR_OWNER_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner = make()
    for role in ("actor","critic"):
        with torch.no_grad(): getattr(runner.alg,role).mlp[0].weight[:,422:].fill_(.043)
    for parameter in list(runner.alg.actor.parameters())+list(runner.alg.critic.parameters()): parameter.grad=torch.ones_like(parameter)
    runner.alg.optimizer.step();runner.alg.optimizer.zero_grad()
    for group in runner.alg.optimizer.param_groups: group["lr"] = 2.25e-5
    runner.alg.learning_rate = 2.25e-5
    base = source_metadata()
    for key in ("checkpoint_path","checkpoint_sha256","save_load_round_trip"): base.pop(key,None)
    runner.current_learning_iteration = base["ppo_updates"]
    for state in runner.alg.optimizer.state.values(): state["step"].fill_(float(base["optimizer_steps"]))
    branch = tmp_path/("ppo_"+m.EXPERIMENT)/"branches"/m.BRANCH_NAME
    base["checkpoint_output_routing"]["output_root"] = str(branch)
    path,sidecar = t.save_semantic_checkpoint(runner,branch/"checkpoints/history/source.pt",base)
    source = json.loads(sidecar.read_text(encoding="utf-8"));target=target_contract(source["runtime_contract"])
    factor = dict(source_policy_contract=source["policy_contract"],target_policy_contract=source["policy_contract"],
        target_runner_config=source["runner_config"],source_effective_learning_rate=source["optimizer_learning_rate"],
        preserved_metadata_sha256=m._protected(source))
    record = dict(schema=m.SCHEMA,plan_path=str(tmp_path/"plan.json"),**{m.FACTOR_KEY:factor})
    monkeypatch.setattr(m,"validate_front_retention439_identity",lambda *a,**k:record)
    calls=[]
    monkeypatch.setattr(m,"validate_front_retention439_lineage",lambda *a,**k:calls.append(a[0].get(m.LEDGER)) or record)
    yield torch,t,make,path,source,target,record,branch,calls
    torch.set_rng_state(old_rng);torch.set_num_threads(old_threads)


def test_complete439_identity_publication_and_official_reload(cpu_state):
    torch,t,make,path,source,target,record,branch,calls=cpu_state
    output=branch/"checkpoints/history/identity.pt"
    receipt=m.publish_front_retention439_checkpoint(path,target,record["plan_path"],output)
    before,after=[torch.load(p,map_location="cpu",weights_only=False) for p in (path,output)]
    assert t.state_hash({k:v for k,v in before.items() if k!="infos"})==t.state_hash({k:v for k,v in after.items() if k!="infos"})
    fresh=make();loaded=t.load_semantic_checkpoint(fresh,output,contract=target,seed=1001)
    assert loaded["rr_retention_reward_migration"]==source["rr_retention_reward_migration"]
    assert loaded["rear_owner_recovery_migration"]==source["rear_owner_recovery_migration"]
    assert all(receipt[k]==0 for k in ("added_policy_decisions","added_ppo_updates","added_optimizer_steps","added_auxiliary_updates"))
    assert fresh.alg.storage.step==0 and fresh.alg.transition.actions is None
    assert calls and not (branch/"checkpoints/checkpoint_last_pointer.json").exists()


def test_sparse_P10_P13_full_gaussian_and_raw_logp_synthetic_unit_only(cpu_state):
    """All rows here are synthetic; this test never claims a reached P13 state."""
    import importlib.util
    import sys
    torch,t,make,*_ = cpu_state
    path=ROOT/"outputs/ppo_rl_recovery_learning_v1/staged_front_retention439/front_retention439.py"
    spec=importlib.util.spec_from_file_location("front439_numeric_unit_kernel",path)
    kernel=importlib.util.module_from_spec(spec);sys.modules[spec.name]=kernel;spec.loader.exec_module(kernel)
    deps=kernel._numeric_dependencies()
    actor=make().alg.actor
    x=torch.zeros(3,439)
    for row,phase in enumerate((9,10,11)):
        x[row,phase]=1.;x[row,20]=.03;x[row,158:158+phase]=1.
    observations=kernel.tensors(x,deps=deps)
    p13,evidence=kernel._synthetic_p13_invariance(observations,deps)
    assert evidence["actual_rows_by_phase"]["P13"]==0
    assert evidence["synthetic_P13_rows"]==3 and evidence["real_P13_validation_claimed"] is False
    first=kernel.first_layer(actor,deps)
    original=t.parameter_hash(actor)
    # Independent finite leaf perturbation: no fit, SGD, or optimizer step.
    leaf=first.weight[:,kernel.PHASE_COLUMNS].detach().clone()+.125
    for supplied in (observations,p13):
        before=kernel.distribution(actor,supplied,deps=deps)
        after=kernel.distribution(actor,supplied,leaf,deps)
        assert kernel._exact_gaussian(before,after)
        raw=before["mean"].detach()+.137*before["sigma"].detach()
        logp_before=torch.distributions.Normal(before["mean"],before["sigma"]).log_prob(raw).sum(-1)
        logp_after=torch.distributions.Normal(after["mean"],after["sigma"]).log_prob(raw).sum(-1)
        assert torch.equal(logp_before,logp_after)
    assert t.parameter_hash(actor)==original
    # Counterexample: selected phase input is nonzero, so there is no promise.
    selected=x[:1].clone();selected[:,:13]=0.;selected[:,1]=1.
    selected=kernel.tensors(selected,deps=deps)
    assert not kernel._exact_gaussian(kernel.distribution(actor,selected,deps=deps),
                                      kernel.distribution(actor,selected,leaf,deps))
    # Non-Identity normalization invalidates the zero-column argument.
    normalizer=actor.obs_normalizer
    try:
        actor.obs_normalizer=torch.nn.LayerNorm(439)
        with pytest.raises(ValueError,match="Identity"):
            kernel.first_layer(actor,deps)
    finally:
        actor.obs_normalizer=normalizer



def test_aux_then_one_ordinary_ppo_update_validates_and_carries_ledger(cpu_state,tmp_path):
    torch,t,make,path,source,target,record,branch,calls=cpu_state
    from test_semantic_rear_policy_training_audit import Synthetic419Core
    class Synthetic439Core(Synthetic419Core):
        def observation(self,raw=(0.,)*12): return super().observation(raw)+(0.,)*20
    env=t.SemanticRslAdapter(Synthetic439Core(),seed=1001,device="cpu");env.cfg["semantic_version"]="v3"
    learner=make(env)
    infos=t.load_semantic_checkpoint(learner,path,contract=target,seed=1001,migration=record)
    old_actor=t.parameter_hash(learner.alg.actor);old_adam=t.state_hash(learner.alg.optimizer.state_dict())
    with torch.no_grad(): learner.alg.actor.mlp[0].weight[:,1].add_(.0003)
    # Synthetic accepted AUX receipt; numeric carry is isolated from evidence parser.
    ledger={"schema":m.LEDGER_SCHEMA,"events":[{"event_index":1,"fixture":"accepted sparse AUX"}],
        "accepted_auxiliary_updates_total":1,"attempted_auxiliary_optimizer_steps_total":1}
    infos[m.LEDGER]=ledger
    aux_path,_=t.save_semantic_checkpoint(learner,branch/"checkpoints/history/aux.pt",infos)
    assert t.parameter_hash(learner.alg.actor)!=old_actor and t.state_hash(learner.alg.optimizer.state_dict())==old_adam
    infos=t.load_semantic_checkpoint(learner,aux_path,contract=target,seed=1001)
    result=t.train_semantic(learner,env,run_dir=tmp_path/"one_update",output_root=branch,
        stage="full_episode",decisions=128,contract=target,seed=1001,resume_infos=infos,
        checkpoint_interval_updates=1,checkpoint_output_routing=source["checkpoint_output_routing"])
    assert (result["actual_policy_decisions"],result["ppo_updates_this_run"],result["optimizer_steps_this_run"])==(128,1,20)
    saved=Path(result["checkpoints"][-1]["checkpoint"])
    manifest=json.loads(saved.with_name(saved.stem+"_manifest.json").read_text(encoding="utf-8"))
    assert manifest[m.LEDGER]==ledger and manifest[m.IDENTITY]==record
    assert manifest["rr_retention_reward_migration"]==source["rr_retention_reward_migration"]
    for key,delta in zip(m.COUNTERS,(128,1,20),strict=True): assert manifest[key]==source[key]+delta
    assert sum(value==ledger for value in calls)>=3
    fresh=make();loaded=t.load_semantic_checkpoint(fresh,saved,contract=target,seed=1001)
    assert loaded[m.LEDGER]==ledger

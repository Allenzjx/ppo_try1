"""Bounded CPU synthetic v9 migration/routing; never real learning evidence."""
from copy import deepcopy
import ast
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
import yaml

from test_semantic_rr_signed_wheel_migration import boundary as v8_boundary
from test_semantic_rr_capture_knee_migration import git, write
from wlr50_clean.ppo import semantic_rr_postcapture_wheel_migration as m
from wlr50_clean.ppo.semantic_migration import digest, PROJECT_ROOT


def bind_prior(meta,old):
    meta[m.PRIOR]={"schema":"wlr50_clean.rr_signed_wheel_same410.v8",
        "target_git_commit":old["source_git_commit"],"target_contract_sha256":digest(old),
        "target_runtime_content_sha256":old["runtime_content_sha256"],
        "source_selection":{"source_role":"front_validated_ancestor_control_eval","counters":deepcopy(m.ANCESTOR_ORIGIN)},
        m.PRIOR_FACTOR:{"counter_origin":deepcopy(m.ANCESTOR_ORIGIN),"target_feedback_revision":m.FEEDBACK}}


def boundary(base=None):
    meta,_,old,_=v8_boundary(True)
    old=deepcopy(old);old["source_git_commit"]=m.SOURCE_HEAD
    for p in m.ALLOWED_FILES-{m.MODULE}: old["files"].setdefault(p,"a"*64)
    old["runtime_content_sha256"]=digest(old["files"])
    for selected in old["selected_configuration"].values(): selected["sha256"]=old["files"][selected["path"]]
    new=deepcopy(old);new["source_git_commit"]="f"*40
    new["files"].update({p:"9"*64 for p in m.ALLOWED_FILES})
    new["runtime_content_sha256"]=digest(new["files"])
    for selected in new["selected_configuration"].values(): selected["sha256"]=new["files"][selected["path"]]
    meta.update(m.SOURCE_COUNTS)
    meta.update(runtime_contract=old,checkpoint_sha256=m.SOURCE_SHA)
    selection={"source_role":"front_validated_ancestor_control_eval","counters":deepcopy(m.ANCESTOR_ORIGIN)}
    meta["rr_progress_handoff_v5_migration"]["source_selection"]=selection
    meta["rr_capture_transfer_branch"]["counter_origin"]=deepcopy(m.ANCESTOR_ORIGIN)
    meta["checkpoint_output_routing"]={"schema":"wlr50_clean.checkpoint_output_routing.v1",
        "branch":m.SOURCE_BRANCH,"output_root":str((Path(base or "synthetic")/"branches"/m.SOURCE_BRANCH).resolve()),
        "main_latest_pointer_promotion":False,"source_selection":deepcopy(selection)}
    for k,v in list(meta.items()):
        if k.endswith("_branch") and isinstance(v,dict) and "counter_origin" in v:
            meta[k+"_counts"]={c:meta[c]-v["counter_origin"][c] for c in m.COUNTERS}
    bind_prior(meta,old)
    profile={"revision":m.SOURCE_REVISION,"rr_capture_feedback_revision":m.FEEDBACK,
        "rr_capture_wheel_mode":"rr_capture_support_forward_projection_v1","unchanged":{"caps":True}}
    return meta,old,new,dict(reason="synthetic v9 learned branch continuation",source_binding=m.registered_source(m.SOURCE_SHA),
        source_profile=profile,target_profile={**profile,"revision":m.TARGET_REVISION},
        expected_target_head=new["source_git_commit"],reviewed_code_sha256={p:new["files"][p] for p in m.ALLOWED_FILES})


def bind_current(meta,old,new,args):
    factor=m.rr_postcapture_wheel_factor(meta,old,new,**args)
    value=deepcopy(meta);value["runtime_contract"]=deepcopy(new)
    value[m.MIGRATION]={"schema":m.SCHEMA,"source_checkpoint_sha256":m.SOURCE_SHA,
        "source_manifest_sha256":m.SOURCE_MANIFEST_SHA,"source_git_commit":m.SOURCE_HEAD,
        "source_selection":m.registered_source(m.SOURCE_SHA),"target_git_commit":new["source_git_commit"],
        "target_contract_sha256":digest(new),"target_runtime_content_sha256":new["runtime_content_sha256"],m.FACTOR_KEY:factor}
    return value


def test_learned_source_credit_and_old_ancestor_receipt_remain_distinct():
    meta,old,new,args=boundary();before=deepcopy(meta)
    factor=m.rr_postcapture_wheel_factor(meta,old,new,**args)
    assert meta==before
    assert factor["counter_origin"]==dict(global_policy_decisions=221952,ppo_updates=1699,optimizer_steps=33980)
    assert meta[m.PRIOR][m.PRIOR_FACTOR]["counter_origin"]==m.ANCESTOR_ORIGIN
    assert meta["rr_capture_transfer_branch_counts"]==dict(global_policy_decisions=1408,ppo_updates=11,optimizer_steps=220)
    assert all(factor["preserved_metadata_sha256"][k]==digest(meta[k]) for k in m.preserved_keys(meta))
    assert not factor["creates_new_branch"] and not factor["same_mdp_claimed"]
    assert all(factor[k]==0 for k in ("added_policy_decisions","added_ppo_updates","added_optimizer_steps","added_auxiliary_updates"))
    assert factor["observation_shape_changed"] is False and "X409" in factor["observation_semantics_changed"][0]
    assert len(args["reviewed_code_sha256"])==6


@pytest.mark.parametrize("bad",["old_zero_update_source","main_latest","source_manifest","borrow_main640","counts",
    "source_head","target_head","prior_origin_rewritten","prior_runtime","old_receipt_removed","route_removed",
    "foreign_branch","route_selection","prior_branch_origin","branch_counts","repeat","profile_extra","reward",
    "assist_changed","old_validator_changed","extra_file","missing_wheel","review","LR"])
def test_fail_closed_source_scope_and_ancestry(bad):
    meta,old,new,args=boundary()
    if bad in ("old_zero_update_source","main_latest"): args["source_binding"]["checkpoint_sha256"]="0"*64
    elif bad=="source_manifest": args["source_binding"]["manifest_sha256"]="0"*64
    elif bad=="borrow_main640": meta["global_policy_decisions"]+=640
    elif bad=="counts": meta["optimizer_steps"]+=1
    elif bad=="source_head": old["source_git_commit"]="0"*40
    elif bad=="target_head": args["expected_target_head"]=None
    elif bad=="prior_origin_rewritten": meta[m.PRIOR][m.PRIOR_FACTOR]["counter_origin"]=deepcopy(m.SOURCE_COUNTS)
    elif bad=="prior_runtime": meta[m.PRIOR]["target_contract_sha256"]="0"*64
    elif bad=="old_receipt_removed": meta.pop("rr_capture_knee_v3_migration")
    elif bad=="route_removed": meta.pop("checkpoint_output_routing")
    elif bad=="foreign_branch": meta["checkpoint_output_routing"]["branch"]="other"
    elif bad=="route_selection": meta["checkpoint_output_routing"]["source_selection"]["source_role"]="latest_learned_continuation"
    elif bad=="prior_branch_origin": meta["rr_capture_transfer_branch"]["counter_origin"]=deepcopy(m.SOURCE_COUNTS)
    elif bad=="branch_counts": meta["rr_capture_transfer_branch_counts"]["ppo_updates"]+=1
    elif bad=="repeat": meta[m.MIGRATION]={}
    elif bad=="profile_extra": args["target_profile"]["rr_capture_feedback_revision"]="other"
    elif bad=="reward": new["selected_configuration"]["reward_config.yaml"]["sha256"]="0"*64
    elif bad in ("assist_changed","old_validator_changed","extra_file"):
        path="src/wlr50_clean/ppo/"+{"assist_changed":"semantic_rr_capture_assist.py",
            "old_validator_changed":"semantic_rr_signed_wheel_migration.py","extra_file":"unreviewed.py"}[bad]
        new["files"][path]=args["reviewed_code_sha256"][path]="0"*64
    elif bad=="missing_wheel": new["files"][m.WHEEL]=old["files"][m.WHEEL]
    elif bad=="review": args["reviewed_code_sha256"].pop(m.WHEEL)
    else: meta["optimizer_learning_rate"]=float("nan")
    with pytest.raises(ValueError): m.rr_postcapture_wheel_factor(meta,old,new,**args)


def wheel_pair():
    source=subprocess.check_output(["git","-C",str(PROJECT_ROOT),"show",m.SOURCE_HEAD+":"+m.WHEEL])
    target=(PROJECT_ROOT/m.WHEEL).read_bytes()
    return source,target


@pytest.mark.parametrize("bad",[None,"semantics","slew","default_phase","sourceproof","numeric_helper"])
def test_reviewed_wheel_constants_and_public_P12_evidence(bad):
    source,target=wheel_pair();target=target.decode()
    if bad=="semantics": target=target.replace(m.WHEEL_SEMANTICS,"unreviewed")
    elif bad=="slew": target=target.replace("WHEEL_RATE_RAD_S2 = 1.8","WHEEL_RATE_RAD_S2 = 2.0")
    elif bad=="default_phase": target=target.replace('source_phase="P09"','source_phase="P12"')
    elif bad=="sourceproof": target=target.replace("P12_all4_authored_final_wheel_stop_not_full_RL_source_endpoint","unreviewed")
    elif bad=="numeric_helper": target=target.replace("return max(lo, min(hi, value))","return min(hi, value)")
    if bad:
        with pytest.raises(ValueError): m.validate_wheel_semantics(source,target)
    else: m.validate_wheel_semantics(source,target)


@pytest.mark.parametrize("phase",["P01","P07"])
def test_v9_same_branch_natural_and_suffix_route_and_newest_receipt_priority(tmp_path,phase):
    from wlr50_clean.ppo import semantic_cli as cli,semantic_training as training
    meta,old,new,kw=boundary(tmp_path);current=bind_current(meta,old,new,kw)
    branch=Path(current["checkpoint_output_routing"]["output_root"])
    source=branch/"checkpoints/history/unique_v9.pt"
    write(source,"synthetic routing only")
    manifest=source.with_name(source.stem+"_manifest.json");write(manifest,json.dumps(current))
    args=SimpleNamespace(checkpoint=source,checkpoint_output_branch=m.SOURCE_BRANCH,
        command="train",from_phase=phase,prefix_source="checkpoint_policy" if phase!="P01" else None)
    cli._bind_checkpoint_output_routing(args,tmp_path,branch,branch)
    assert args._checkpoint_output_routing==current["checkpoint_output_routing"]
    # No flag direct API rejects before inspecting a runner or stepping any optimizer.
    with pytest.raises(ValueError,match="explicit output routing"):
        training.train_semantic(None,None,run_dir=tmp_path/"no_run",output_root=branch,stage="full_episode",
            decisions=128,contract=new,seed=1001,resume_infos=current)
    current[m.MIGRATION]={};write(manifest,json.dumps(current))
    with pytest.raises(ValueError): cli._bind_checkpoint_output_routing(args,tmp_path,branch,branch)


@pytest.mark.parametrize("bad",["rewritten_origin","stale_runtime","route","counts","factor","missing_current"])
def test_v9_branch_receipt_never_falls_back_to_valid_old_receipt(bad):
    meta,old,new,kw=boundary();current=bind_current(meta,old,new,kw);route=deepcopy(current["checkpoint_output_routing"])
    if bad=="rewritten_origin": current[m.MIGRATION][m.FACTOR_KEY]["counter_origin"]=deepcopy(m.ANCESTOR_ORIGIN)
    elif bad=="stale_runtime": current[m.MIGRATION]["target_contract_sha256"]=digest(old)
    elif bad=="route": route["branch"]="other"
    elif bad=="counts": current["global_policy_decisions"]-=1
    elif bad=="factor": current[m.MIGRATION][m.FACTOR_KEY]=None
    else: current[m.MIGRATION]={}
    with pytest.raises(ValueError): m.validate_v9_branch_receipt(current,new,route)


def test_official_identity_roundtrip_from_learned_branch_and_normal_save_carry(tmp_path,monkeypatch):
    torch=pytest.importorskip("torch");pytest.importorskip("rsl_rl")
    from wlr50_clean.ppo import semantic_training as training,semantic_migration as shared
    from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
    assert not torch.cuda.is_available()
    source_wheel,target_wheel=wheel_pair()
    meta,old,new,args=boundary(tmp_path)
    root=tmp_path/"git";root.mkdir();git(root,"init")
    git(root,"config","user.email","test@example.invalid");git(root,"config","user.name","Synthetic test")
    git(root,"config","core.autocrlf","false")
    paths=set(old["files"])
    for path in paths:
        write(root/path,yaml.safe_dump(args["source_profile"]) if path==m.PROFILE else
            source_wheel.decode() if path==m.WHEEL else "# synthetic old runtime\n")
    git(root,"add",".");git(root,"commit","-m","source")
    def contract(template):
        value=deepcopy(template);value["source_git_commit"]=git(root,"rev-parse","HEAD")
        value["files"]={p:shared.file_sha(root/p) for p in paths};value["runtime_content_sha256"]=digest(value["files"])
        for selected in value["selected_configuration"].values(): selected["sha256"]=value["files"][selected["path"]]
        return value
    old=contract(old);paths.add(m.MODULE)
    for path in m.ALLOWED_FILES:
        write(root/path,yaml.safe_dump(args["target_profile"]) if path==m.PROFILE else
            target_wheel.decode() if path==m.WHEEL else "# synthetic target runtime\n")
    git(root,"add",".");git(root,"commit","-m","target");new=contract(new)
    monkeypatch.setattr(m,"SOURCE_HEAD",old["source_git_commit"])
    meta["runtime_contract"]=old;bind_prior(meta,old)
    def make():
        return training.construct_semantic_runner(_shape_env(410,"cpu"),seed=1001,device="cpu",
            policy_version=m.RR_CAPTURE_POLICY,observation_layout=m.RR_CAPTURE_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner=make();runner.alg.learning_rate=1e-5
    for group in runner.alg.optimizer.param_groups:
        group["lr"]=1e-5
        for p in group["params"]:
            runner.alg.optimizer.state[p]={"step":torch.tensor(33980.),"exp_avg":torch.full_like(p,.001),"exp_avg_sq":torch.full_like(p,.002)}
    meta.pop("checkpoint_sha256")
    source_path=Path(meta["checkpoint_output_routing"]["output_root"])/"checkpoints/history/checkpoint_step_000221952.pt"
    source,manifest=training.save_semantic_checkpoint(runner,source_path,meta)
    original=shared.checkpoint_metadata(source)
    monkeypatch.setattr(m,"SOURCE_SHA",shared.file_sha(source));monkeypatch.setattr(m,"SOURCE_MANIFEST_SHA",shared.file_sha(manifest))
    plan=m.build_rr_postcapture_wheel_migration(source,new,expected_source_sha256=m.SOURCE_SHA,
        expected_target_head=new["source_git_commit"],reason="synthetic official v9",project_root=root,
        reviewed_code_sha256={p:h for p,h in new["files"].items() if old["files"].get(p)!=h})
    path=tmp_path/"plan.json";write(path,json.dumps(plan))
    validate=shared.validate_migration_plan
    monkeypatch.setattr(shared,"validate_migration_plan",lambda c,co,p,**kw:validate(c,co,p,project_root=root))
    verified=shared.validate_migration_plan(source,new,path)
    migrated=make();loaded=training.load_semantic_checkpoint(migrated,source,contract=new,seed=1001,migration=verified)
    loaded.update(runtime_contract=new,old_rollout_inherited=False)
    saved,_=training.save_semantic_checkpoint(migrated,source.parent/"unique_v9.pt",loaded)
    fresh=make();again=training.load_semantic_checkpoint(fresh,saved,contract=new,seed=1001)
    assert again[m.MIGRATION]==verified and all(again[k]==original[k] for k in m.preserved_keys(original))
    assert {k:again[k] for k in m.COUNTERS}==m.SOURCE_COUNTS
    assert again["rr_capture_transfer_branch_counts"]==dict(global_policy_decisions=1408,ppo_updates=11,optimizer_steps=220)
    assert fresh.alg.storage.step==0 and fresh.alg.transition.actions is None
    m.validate_v9_branch_receipt(again,new,again["checkpoint_output_routing"])
    observations=_shape_env(410,"cpu").get_observations()
    observations["policy"][...,0]=0.;observations["policy"][...,11]=1.;observations["policy"][...,409]=1.
    observations["critic"]=observations["policy"].clone()
    with torch.no_grad():
        runner.alg.actor(observations,stochastic_output=True);before=tuple(v.clone() for v in runner.alg.actor.output_distribution_params)
        fresh.alg.actor(observations,stochastic_output=True)
        assert all(torch.equal(a,b) for a,b in zip(before,fresh.alg.actor.output_distribution_params))
        assert torch.equal(runner.alg.critic(observations),fresh.alg.critic(observations))
    assert training.state_hash(fresh.alg.optimizer.state_dict())==training.state_hash(runner.alg.optimizer.state_dict())
    carry=[ast.literal_eval(n.iter) for n in ast.walk(ast.parse(inspect.getsource(training.train_semantic)))
        if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple)
        and any(isinstance(e,ast.Constant) and e.value=="new_mdp_warm_start" for e in n.iter.elts)]
    assert len(carry)==1 and m.MIGRATION in carry[0] and m.PRIOR in carry[0]
    # Metadata-only ordinary-save fixture, explicitly not an actual PPO update.
    later=deepcopy(again);later.update({k:again[k]+v for k,v in zip(m.COUNTERS,(128,1,20))})
    later_path,_=training.save_semantic_checkpoint(fresh,source.parent/"later_synthetic.pt",later)
    final=training.load_semantic_checkpoint(make(),later_path,contract=new,seed=1001)
    assert final[m.MIGRATION]==verified and final[m.PRIOR]==original[m.PRIOR]
    assert final["rr_postcross_workspace_branch"]==original["rr_postcross_workspace_branch"]
    assert final["checkpoint_output_routing"]==original["checkpoint_output_routing"]
    m.validate_v9_branch_receipt(final,new,final["checkpoint_output_routing"])

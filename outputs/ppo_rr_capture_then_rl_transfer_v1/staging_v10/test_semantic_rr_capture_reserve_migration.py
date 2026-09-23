"""UNAPPLIED bounded synthetic v10 tests; no actual source or learning credit."""
from copy import deepcopy
import ast
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from test_semantic_rr_postcapture_wheel_migration import boundary as boundary9,bind_current as bind9
from test_semantic_rr_capture_knee_migration import git,write
from wlr50_clean.ppo import semantic_rr_capture_reserve_migration as m
from wlr50_clean.ppo.semantic_migration import digest,PROJECT_ROOT


def boundary(monkeypatch,base=None,aux=False):
    meta,older,old,kw=boundary9(base)
    old["source_git_commit"]=m.SOURCE_HEAD;kw["expected_target_head"]=m.SOURCE_HEAD
    meta=bind9(meta,older,old,kw)
    counts={k:m.V9_PUBLICATION_COUNTS[k]+n for k,n in zip(m.COUNTERS,(128,1,20))}
    meta.update(counts);meta["checkpoint_sha256"]="a"*64
    if aux:
        meta["rr_capture_transfer_branch"]["front_retention_auxiliary"]={
            "schema":"synthetic_current_branch_AUX","events":[{"event_index":1,"accepted":2,"attempted":3}]}
    for key,value in list(meta.items()):
        if key.endswith("_branch") and isinstance(value,dict) and "counter_origin" in value:
            meta[key+"_counts"]={k:meta[k]-value["counter_origin"][k] for k in m.COUNTERS}
    registry={"a"*64:dict(manifest_sha256="b"*64,counters=counts,
        source_role="latest_sealed_v9_branch_with_official_front_retention_AUX" if aux else "latest_sealed_v9_branch_PPO",
        front_retention_auxiliary_sha256=digest(meta["rr_capture_transfer_branch"]["front_retention_auxiliary"]) if aux else None)}
    monkeypatch.setattr(m,"SOURCE_REGISTRY",registry)
    new=deepcopy(old);new["source_git_commit"]="e"*40
    new["files"].update({path:"f"*64 for path in m.ALLOWED_FILES})
    new["runtime_content_sha256"]=digest(new["files"])
    for selected in new["selected_configuration"].values(): selected["sha256"]=new["files"][selected["path"]]
    profile={"revision":m.SOURCE_REVISION,"rr_capture_feedback_revision":m.SOURCE_FEEDBACK,
        "rr_capture_wheel_mode":"rr_capture_support_forward_projection_v1"}
    args=dict(reason="synthetic latest v9 preservation",reviewed_code_sha256={p:new["files"][p] for p in m.ALLOWED_FILES},
        source_profile=profile,target_profile={**profile,"revision":m.TARGET_REVISION,"rr_capture_feedback_revision":m.TARGET_FEEDBACK},
        source_binding=m.registered_source("a"*64),expected_target_head=new["source_git_commit"])
    return meta,old,new,args


def current(meta,old,new,args):
    factor=m.rr_capture_reserve_factor(meta,old,new,**args)
    value=deepcopy(meta);value["runtime_contract"]=new
    selected=args["source_binding"]
    value[m.MIGRATION]={"schema":m.SCHEMA,"source_checkpoint_sha256":selected["checkpoint_sha256"],
        "source_manifest_sha256":selected["manifest_sha256"],"source_git_commit":m.SOURCE_HEAD,
        "source_selection":selected,"target_git_commit":new["source_git_commit"],"target_contract_sha256":digest(new),
        "target_runtime_content_sha256":new["runtime_content_sha256"],m.FACTOR_KEY:factor}
    return value


@pytest.mark.parametrize("aux",[False,True])
def test_preserve_latest_actual_counter_origin_and_all_AUX(monkeypatch,aux):
    meta,old,new,args=boundary(monkeypatch,aux=aux);before=deepcopy(meta)
    factor=m.rr_capture_reserve_factor(meta,old,new,**args)
    assert meta==before and factor["counter_origin"]==args["source_binding"]["counters"]
    assert all(factor["preserved_metadata_sha256"][k]==digest(meta[k]) for k in m.preserved_keys(meta))
    assert factor["effective_execution_semantics"]["total_maximum_travel_deg"]==53.
    assert factor["effective_execution_semantics"]["total_maximum_active_exposure_s"]==45.
    assert factor["observation_semantics_changed"]==[] and not factor["creates_new_branch"]
    assert all(factor[k]==0 for k in ("added_policy_decisions","added_ppo_updates","added_optimizer_steps","added_auxiliary_updates"))
    assert len(m.ALLOWED_FILES)==6


@pytest.mark.parametrize("bad",["unknown_source","manifest","rollback","borrow640","AUXremoved","AUXchanged","oldreceipt",
    "reward","wheel","context","budget","missingpath","review","repeated","LR"])
def test_reject_unregistered_or_expanded_delta(monkeypatch,bad):
    meta,old,new,args=boundary(monkeypatch,aux=True)
    if bad=="unknown_source": args["source_binding"]["checkpoint_sha256"]="0"*64
    elif bad=="manifest": args["source_binding"]["manifest_sha256"]="0"*64
    elif bad=="rollback": m.SOURCE_REGISTRY["a"*64]["counters"]=deepcopy(m.V9_PUBLICATION_COUNTS)
    elif bad=="borrow640": meta["global_policy_decisions"]+=640
    elif bad=="AUXremoved": meta["rr_capture_transfer_branch"].pop("front_retention_auxiliary")
    elif bad=="AUXchanged": meta["rr_capture_transfer_branch"]["front_retention_auxiliary"]["events"][0]["accepted"]+=1
    elif bad=="oldreceipt": meta.pop(m.PRIOR)
    elif bad in ("reward","wheel","context"):
        path={"reward":"configs/ppo_rr_capture_then_rl_transfer_v1/reward_config.yaml",
            "wheel":"src/wlr50_clean/ppo/semantic_rr_carry_wheel.py",
            "context":"src/wlr50_clean/ppo/semantic_rr_capture_context.py"}[bad]
        new["files"][path]=args["reviewed_code_sha256"][path]="1"*64
    elif bad=="budget": args["target_profile"]["new_budget"]=99
    elif bad=="missingpath": new["files"][m.ASSIST]=old["files"][m.ASSIST]
    elif bad=="review": args["reviewed_code_sha256"].pop(m.ASSIST)
    elif bad=="repeated": meta[m.MIGRATION]={}
    else: meta["optimizer_learning_rate"]=float("nan")
    with pytest.raises(ValueError): m.rr_capture_reserve_factor(meta,old,new,**args)


def assist_pair():
    old=subprocess.check_output(["git","-C",str(PROJECT_ROOT),"show",m.SOURCE_HEAD+":"+m.ASSIST])
    return old,(PROJECT_ROOT/m.ASSIST).read_bytes()


@pytest.mark.parametrize("bad",[None,"partial","budget","scale","lower"])
def test_exact_three_upper_checks_no_added_budget(bad):
    source,target=assist_pair();target=target.decode()
    if bad=="partial":
        target=target.replace('state["window_start_gap_m"] < RR_CAPTURE_GAP_MIN_M',
            'not RR_CAPTURE_GAP_MIN_M <= state["window_start_gap_m"] <= .025')
    elif bad=="budget": target=target.replace("53.", "54.")
    elif bad=="scale": target=target.replace("180., 180., 180., 20.", "180., 180., 180., 21.")
    elif bad=="lower": target=target.replace("gap >= RR_CAPTURE_GAP_MIN_M","gap >= -.02",1)
    if bad:
        # Mutations must actually alter the candidate, never vacuous negatives.
        assert target.encode()!=assist_pair()[1]
        with pytest.raises(ValueError): m.validate_assist_semantics(source,target)
    else: m.validate_assist_semantics(source,target)


@pytest.mark.parametrize("phase",["P01","P07"])
def test_newest_v10_route_no_fallback_and_no_main_pointer(monkeypatch,tmp_path,phase):
    from wlr50_clean.ppo import semantic_cli as cli,semantic_training as training
    meta,old,new,args=boundary(monkeypatch,tmp_path,aux=True);value=current(meta,old,new,args)
    branch=Path(value["checkpoint_output_routing"]["output_root"])
    cp=branch/"checkpoints/history/unique_v10.pt";write(cp,"synthetic only")
    manifest=cp.with_name(cp.stem+"_manifest.json");write(manifest,json.dumps(value))
    request=SimpleNamespace(checkpoint=cp,checkpoint_output_branch=m.SOURCE_BRANCH,command="train",
        from_phase=phase,prefix_source="checkpoint_policy" if phase!="P01" else None)
    cli._bind_checkpoint_output_routing(request,tmp_path,branch,branch)
    assert request._checkpoint_output_routing==value["checkpoint_output_routing"]
    with pytest.raises(ValueError,match="explicit output routing"):
        training.train_semantic(None,None,run_dir=tmp_path/"no_run",output_root=branch,stage="full_episode",
            decisions=128,contract=new,seed=1001,resume_infos=value)
    value[m.MIGRATION]={};write(manifest,json.dumps(value))
    with pytest.raises(ValueError): cli._bind_checkpoint_output_routing(request,tmp_path,branch,branch)


def test_official_full_state_identity_roundtrip_and_save_carry(monkeypatch,tmp_path):
    torch=pytest.importorskip("torch");pytest.importorskip("rsl_rl")
    from wlr50_clean.ppo import semantic_training as training,semantic_migration as shared
    from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
    assert not torch.cuda.is_available()
    source_assist,target_assist=assist_pair()
    meta,old,new,args=boundary(monkeypatch,tmp_path,aux=True)
    root=tmp_path/"git";root.mkdir();git(root,"init")
    git(root,"config","user.email","test@example.invalid");git(root,"config","user.name","Synthetic test")
    git(root,"config","core.autocrlf","false")
    paths=set(old["files"])
    for path in paths:
        write(root/path,yaml.safe_dump(args["source_profile"]) if path==m.PROFILE else
            source_assist.decode() if path==m.ASSIST else "# synthetic old runtime\n")
    git(root,"add",".");git(root,"commit","-m","source")
    def contract(template):
        value=deepcopy(template);value["source_git_commit"]=git(root,"rev-parse","HEAD")
        value["files"]={p:shared.file_sha(root/p) for p in paths};value["runtime_content_sha256"]=digest(value["files"])
        for selected in value["selected_configuration"].values(): selected["sha256"]=value["files"][selected["path"]]
        return value
    old=contract(old);paths.add(m.MODULE)
    for path in m.ALLOWED_FILES:
        write(root/path,yaml.safe_dump(args["target_profile"]) if path==m.PROFILE else
            target_assist.decode() if path==m.ASSIST else "# synthetic target runtime\n")
    git(root,"add",".");git(root,"commit","-m","target");new=contract(new)
    monkeypatch.setattr(m,"SOURCE_HEAD",old["source_git_commit"])
    meta["runtime_contract"]=old
    meta[m.PRIOR].update(target_git_commit=old["source_git_commit"],target_contract_sha256=digest(old),
        target_runtime_content_sha256=old["runtime_content_sha256"])
    def make():
        return training.construct_semantic_runner(_shape_env(410,"cpu"),seed=1001,device="cpu",
            policy_version=m.RR_CAPTURE_POLICY,observation_layout=m.RR_CAPTURE_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner=make();runner.alg.learning_rate=1e-5
    for group in runner.alg.optimizer.param_groups:
        group["lr"]=1e-5
        for p in group["params"]:
            runner.alg.optimizer.state[p]={"step":torch.tensor(float(meta["optimizer_steps"])),
                "exp_avg":torch.full_like(p,.001),"exp_avg_sq":torch.full_like(p,.002)}
    meta.pop("checkpoint_sha256")
    source,sidecar=training.save_semantic_checkpoint(runner,Path(meta["checkpoint_output_routing"]["output_root"])/"checkpoints/history/synthetic_latest_AUX.pt",meta)
    original=shared.checkpoint_metadata(source)
    registered=deepcopy(m.SOURCE_REGISTRY["a"*64]);registered["manifest_sha256"]=shared.file_sha(sidecar)
    monkeypatch.setattr(m,"SOURCE_REGISTRY",{shared.file_sha(source):registered})
    plan=m.build_rr_capture_reserve_migration(source,new,expected_source_sha256=shared.file_sha(source),
        expected_target_head=new["source_git_commit"],reason="synthetic full-state v10",project_root=root,
        reviewed_code_sha256={p:h for p,h in new["files"].items() if old["files"].get(p)!=h})
    plan_path=tmp_path/"plan.json";write(plan_path,json.dumps(plan))
    validate=shared.validate_migration_plan
    monkeypatch.setattr(shared,"validate_migration_plan",lambda c,co,p,**kw:validate(c,co,p,project_root=root))
    verified=shared.validate_migration_plan(source,new,plan_path)
    migrated=make();loaded=training.load_semantic_checkpoint(migrated,source,contract=new,seed=1001,migration=verified)
    loaded.update(runtime_contract=new,old_rollout_inherited=False)
    saved,_=training.save_semantic_checkpoint(migrated,source.parent/"unique_v10.pt",loaded)
    fresh=make();again=training.load_semantic_checkpoint(fresh,saved,contract=new,seed=1001)
    assert again[m.MIGRATION]==verified and all(again[k]==original[k] for k in m.preserved_keys(original))
    assert {k:again[k] for k in m.COUNTERS}==registered["counters"]
    assert fresh.alg.storage.step==0 and fresh.alg.transition.actions is None
    assert training.state_hash(fresh.alg.actor.state_dict())==training.state_hash(runner.alg.actor.state_dict())
    assert training.state_hash(fresh.alg.critic.state_dict())==training.state_hash(runner.alg.critic.state_dict())
    assert training.state_hash(fresh.alg.optimizer.state_dict())==training.state_hash(runner.alg.optimizer.state_dict())
    assert again["training_rng_state"]==original["training_rng_state"]
    m.validate_v10_branch_receipt(again,new,again["checkpoint_output_routing"])
    carry=[ast.literal_eval(n.iter) for n in ast.walk(ast.parse(inspect.getsource(training.train_semantic)))
        if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple)
        and any(isinstance(e,ast.Constant) and e.value=="new_mdp_warm_start" for e in n.iter.elts)]
    assert len(carry)==1 and m.MIGRATION in carry[0] and m.PRIOR in carry[0]
    # Save-carry fixture only; this is not a real PPO update.
    later=deepcopy(again);later.update({k:again[k]+n for k,n in zip(m.COUNTERS,(128,1,20))})
    path,_=training.save_semantic_checkpoint(fresh,source.parent/"later_synthetic.pt",later)
    final=training.load_semantic_checkpoint(make(),path,contract=new,seed=1001)
    assert final[m.MIGRATION]==verified and all(final[k]==again[k] for k in
        (m.PRIOR,"rr_capture_transfer_branch","rr_postcross_workspace_branch","checkpoint_output_routing"))
    m.validate_v10_branch_receipt(final,new,final["checkpoint_output_routing"])

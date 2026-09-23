"""CPU synthetic v6 full-state migration; not physical or learning evidence."""
from copy import deepcopy
import ast
import inspect
import json
import pytest
import yaml
from test_semantic_rr_progress_handoff_migration import boundary as prior_boundary
from test_semantic_rr_capture_knee_migration import git,write
from wlr50_clean.ppo import semantic_rr_contact_onset_migration as m
from wlr50_clean.ppo import semantic_rr_progress_handoff_migration as m5
from wlr50_clean.ppo.semantic_migration import digest


def boundary(ancestor=False):
    meta,_,old,_=prior_boundary(ancestor)
    binding=m.registered_source(list(m.SOURCE_REGISTRY)[0 if ancestor else 1])
    old=deepcopy(old);old["source_git_commit"]=m.SOURCE_HEAD
    for path in m.ALLOWED_FILES-{m.MODULE}: old["files"].setdefault(path,"a"*64)
    old["runtime_content_sha256"]=digest(old["files"])
    for selected in old["selected_configuration"].values(): selected["sha256"]=old["files"][selected["path"]]
    new=deepcopy(old);new["source_git_commit"]="f"*40
    new["files"].update({p:"e"*64 for p in m.ALLOWED_FILES})
    new["runtime_content_sha256"]=digest(new["files"])
    for selected in new["selected_configuration"].values(): selected["sha256"]=new["files"][selected["path"]]
    meta.update(binding["counters"]);meta.update(runtime_contract=old,checkpoint_sha256=binding["checkpoint_sha256"])
    for key,value in list(meta.items()):
        if key.endswith("_branch") and isinstance(value,dict) and "counter_origin" in value:
            meta[key+"_counts"]={k:meta[k]-value["counter_origin"][k] for k in m.COUNTERS}
    bind_prior(meta,old,binding)
    source_profile={"revision":m.SOURCE_REVISION,"rr_capture_feedback_revision":m.SOURCE_FEEDBACK,
        "rr_capture_wheel_mode":"rr_capture_support_forward_projection_v1","other":{"all12":True}}
    args=dict(reason="synthetic exact same410 v6 control migration",source_binding=binding,
        source_profile=source_profile,target_profile={**source_profile,"revision":m.TARGET_REVISION,
            "rr_capture_feedback_revision":m.TARGET_FEEDBACK},expected_target_head=new["source_git_commit"],
        reviewed_code_sha256={p:h for p,h in new["files"].items() if old["files"].get(p)!=h})
    return meta,old,new,args


def bind_prior(meta,old,binding):
    meta[m.PRIOR]={"schema":"wlr50_clean.rr_progress_handoff_same410.v5",
        "target_git_commit":old["source_git_commit"],"target_contract_sha256":digest(old),
        "target_runtime_content_sha256":old["runtime_content_sha256"],
        "source_selection":deepcopy(binding),
        m.PRIOR_FACTOR:{"counter_origin":deepcopy(binding["counters"]),"target_feedback_revision":m.SOURCE_FEEDBACK}}


@pytest.mark.parametrize("ancestor",[False,True])
def test_two_distinct_sources_keep_exact_state_origins_receipts_and_aux(ancestor):
    meta,old,new,args=boundary(ancestor);before=deepcopy(meta)
    value=m.rr_contact_onset_factor(meta,old,new,**args)
    assert meta==before and value["counter_origin"]==args["source_binding"]["counters"]
    assert value["candidate_evaluation_only"] is False
    assert value["ancestor_training_requires_explicit_output_branch"]==ancestor
    assert not value["latest_pointer_promotion_authorized"]
    assert value["source_v5_receipt_sha256"]==digest(meta[m.PRIOR])
    assert value["observation_contract"]["source_policy_contract"]==value["observation_contract"]["target_policy_contract"]
    assert all(value["preserved_metadata_sha256"][k]==digest(meta[k]) for k in m.preserved_keys(meta))
    assert not value["creates_new_branch"] and not value["same_mdp_claimed"]
    assert value["added_policy_decisions"]==value["added_ppo_updates"]==value["added_optimizer_steps"]==0
    assert len(args["reviewed_code_sha256"])==8


@pytest.mark.parametrize("bad",["unregistered","binding","counters","head","repeat","receipt","origin",
    "role","branch_counts","missing_lineage","LR","rate","codec","old_validator","extra_backend","extra_wheel",
    "deleted","existing_module","review","profile_extra","config","wrong_feedback","receipt_feedback","missing_video"])
def test_strict_source_and_review_rejections(bad):
    meta,old,new,args=boundary(True)
    if bad=="unregistered": args["source_binding"]["checkpoint_sha256"]="0"*64
    elif bad=="binding": args["source_binding"]["manifest_sha256"]="0"*64
    elif bad=="counters": meta["ppo_updates"]+=1
    elif bad=="head": args["expected_target_head"]=None
    elif bad=="repeat": meta[m.MIGRATION]={}
    elif bad=="receipt": meta[m.PRIOR]["target_runtime_content_sha256"]="0"*64
    elif bad=="origin": meta[m.PRIOR][m.PRIOR_FACTOR]["counter_origin"]["ppo_updates"]-=1
    elif bad=="role": meta[m.PRIOR].pop("source_selection")
    elif bad=="receipt_feedback": meta[m.PRIOR][m.PRIOR_FACTOR]["target_feedback_revision"]="wrong"
    elif bad=="branch_counts": meta["rr_capture_transfer_branch_counts"]["ppo_updates"]+=1
    elif bad=="missing_lineage": meta.pop("rr_capture_knee_v3_migration")
    elif bad=="LR": meta["optimizer_learning_rate"]=float("nan")
    elif bad=="rate": new["physics_hz"]=240
    elif bad=="codec": meta["policy_contract"]["observation_dimension"]=389
    elif bad in ("old_validator","extra_backend","extra_wheel"):
        path="src/wlr50_clean/ppo/"+({"old_validator":"semantic_rr_progress_handoff_migration.py",
            "extra_backend":"semantic_backend.py","extra_wheel":"semantic_rr_carry_wheel.py"}[bad])
        new["files"][path]=args["reviewed_code_sha256"][path]="0"*64
    elif bad=="deleted": new["files"].pop(m.ASSIST)
    elif bad=="existing_module": old["files"][m.MODULE]="0"*64
    elif bad=="review": args["reviewed_code_sha256"].pop(m.ASSIST)
    elif bad=="missing_video": new["files"]["scripts/run_semantic_video.ps1"]=old["files"]["scripts/run_semantic_video.ps1"]
    elif bad=="profile_extra": args["target_profile"]["other"]={"all12":False}
    elif bad=="config": new["selected_configuration"]["stage_task_spec.yaml"]["sha256"]="0"*64
    else: args["target_profile"]["rr_capture_feedback_revision"]="unknown"
    with pytest.raises(ValueError): m.rr_contact_onset_factor(meta,old,new,**args)


def assist_literals(target=False):
    from wlr50_clean.ppo.semantic_rr_capture_assist import RR_CAPTURE_ASSIST_FEATURE_NAMES,_SCALES
    return {"RR_CAPTURE_ASSIST_FEATURE_NAMES":RR_CAPTURE_ASSIST_FEATURE_NAMES,"_SCALES":_SCALES,
        "RR_CAPTURE_ASSIST_MODE":"rr_hip_only_capture_v1","RR_CAPTURE_ASSIST_SCHEMA":"wlr50_clean.rr_capture_assist_state.v1",
        "RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS":"unchanged window",
        "RR_CAPTURE_FEEDBACK_REVISION":m.TARGET_FEEDBACK if target else m.SOURCE_FEEDBACK,
        "RR_CAPTURE_SEARCH_SEMANTICS":m.SEARCH_SEMANTICS if target else m5.SEARCH_SEMANTICS,
        "_MODES":("WAIT","DESCEND","HOLD","BLOCKED","RELEASE","RELEASED","DESCEND_PROGRESS","CAPTURED_FOLLOW")}


def code(values): return "\n".join(f"{k} = {v!r}" for k,v in values.items())+"\n"


@pytest.mark.parametrize("bad",[None,"scale","features","window","modes","search","feedback"])
def test_public_same410_numeric_positions_and_modes_unchanged(bad):
    source=assist_literals();target=assist_literals(True)
    key={"scale":"_SCALES","features":"RR_CAPTURE_ASSIST_FEATURE_NAMES","window":"RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS",
         "modes":"_MODES","search":"RR_CAPTURE_SEARCH_SEMANTICS","feedback":"RR_CAPTURE_FEEDBACK_REVISION"}.get(bad)
    if bad:
        target[key]="unreviewed"
        with pytest.raises(ValueError): m.validate_assist_semantics(code(source),code(target))
    else: m.validate_assist_semantics(code(source),code(target))


@pytest.mark.parametrize("ancestor",[False,True])
def test_generic_official_identity_save_reload_and_later_receipt_carry(tmp_path,monkeypatch,ancestor):
    torch=pytest.importorskip("torch");pytest.importorskip("rsl_rl")
    from wlr50_clean.ppo import semantic_training as training,semantic_migration as shared
    from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
    assert not torch.cuda.is_available()
    meta,old,new,args=boundary(ancestor)
    root=tmp_path/"git";root.mkdir();git(root,"init")
    git(root,"config","user.email","test@example.invalid");git(root,"config","user.name","Synthetic test")
    git(root,"config","core.autocrlf","false")
    paths=set(old["files"])
    for path in paths:
        text=yaml.safe_dump(args["source_profile"]) if path==m.PROFILE else code(assist_literals()) if path==m.ASSIST else "# synthetic source\n"
        write(root/path,text)
    git(root,"add",".");git(root,"commit","-m","source")
    def contract(template):
        result=deepcopy(template);result["source_git_commit"]=git(root,"rev-parse","HEAD")
        result["files"]={p:shared.file_sha(root/p) for p in paths};result["runtime_content_sha256"]=digest(result["files"])
        for selected in result["selected_configuration"].values(): selected["sha256"]=result["files"][selected["path"]]
        return result
    old=contract(old);paths.add(m.MODULE)
    for path in m.ALLOWED_FILES:
        write(root/path,yaml.safe_dump(args["target_profile"]) if path==m.PROFILE else
              code(assist_literals(True)) if path==m.ASSIST else "# synthetic target\n")
    git(root,"add",".");git(root,"commit","-m","target");new=contract(new)
    binding=deepcopy(args["source_binding"]);binding["source_git_commit"]=old["source_git_commit"]
    monkeypatch.setattr(m,"SOURCE_HEAD",old["source_git_commit"])
    meta["runtime_contract"]=old;bind_prior(meta,old,binding)
    def make():
        return training.construct_semantic_runner(_shape_env(410,"cpu"),seed=1001,device="cpu",
            policy_version=m.RR_CAPTURE_POLICY,observation_layout=m.RR_CAPTURE_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner=make();runner.alg.learning_rate=1e-5
    for group in runner.alg.optimizer.param_groups:
        group["lr"]=1e-5
        for p in group["params"]:
            runner.alg.optimizer.state[p]={"step":torch.tensor(9.),"exp_avg":torch.full_like(p,.001),"exp_avg_sq":torch.full_like(p,.002)}
    # Sidecar-only self hash must not be embedded into the checkpoint it hashes.
    meta.pop("checkpoint_sha256")
    source,manifest=training.save_semantic_checkpoint(runner,tmp_path/"source.pt",meta)
    original=shared.checkpoint_metadata(source);sha=shared.file_sha(source)
    binding.update(checkpoint_sha256=sha,manifest_sha256=shared.file_sha(manifest))
    monkeypatch.setattr(m,"SOURCE_REGISTRY",{sha:{k:v for k,v in binding.items() if k!="checkpoint_sha256"}})
    review={p:h for p,h in new["files"].items() if old["files"].get(p)!=h}
    plan=m.build_rr_contact_onset_migration(source,new,expected_source_sha256=sha,
        expected_target_head=new["source_git_commit"],reason="synthetic official v6 migration",reviewed_code_sha256=review,project_root=root)
    plan_path=tmp_path/"plan.json";write(plan_path,json.dumps(plan))
    validate=shared.validate_migration_plan
    monkeypatch.setattr(shared,"validate_migration_plan",lambda c,co,p,**kw:validate(c,co,p,project_root=root))
    verified=shared.validate_migration_plan(source,new,plan_path)
    altered=deepcopy(plan);altered["source_selection"]["source_role"]="opposite"
    bad=tmp_path/"tampered.json";write(bad,json.dumps(altered))
    with pytest.raises(ValueError): shared.validate_migration_plan(source,new,bad)
    migrated=make();loaded=training.load_semantic_checkpoint(migrated,source,contract=new,seed=1001,migration=verified)
    loaded.update(runtime_contract=new,old_rollout_inherited=False)
    saved,_=training.save_semantic_checkpoint(migrated,tmp_path/"unique_v6.pt",loaded)
    fresh=make();again=training.load_semantic_checkpoint(fresh,saved,contract=new,seed=1001)
    assert again[m.MIGRATION]==verified and all(again[k]==original[k] for k in m.preserved_keys(original))
    assert fresh.alg.storage.step==0 and fresh.alg.transition.actions is None
    observations=_shape_env(410,"cpu").get_observations()
    observations["policy"][...,0]=0.;observations["policy"][...,8]=1.
    observations["policy"][...,389]=7/5;observations["policy"][...,394]=53/20;observations["policy"][...,395]=45/12
    observations["critic"]=observations["policy"].clone()
    with torch.no_grad():
        runner.alg.actor(observations,stochastic_output=True);before=tuple(x.clone() for x in runner.alg.actor.output_distribution_params)
        fresh.alg.actor(observations,stochastic_output=True)
        assert all(torch.equal(a,b) for a,b in zip(before,fresh.alg.actor.output_distribution_params))
        assert torch.equal(runner.alg.critic(observations),fresh.alg.critic(observations))
    carry=[ast.literal_eval(n.iter) for n in ast.walk(ast.parse(inspect.getsource(training.train_semantic)))
        if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple)
        and any(isinstance(e,ast.Constant) and e.value=="new_mdp_warm_start" for e in n.iter.elts)]
    assert len(carry)==1 and m.MIGRATION in carry[0] and m.PRIOR in carry[0]
    # Synthetic ordinary-save counter fixture only, not a real PPO update.
    later=deepcopy(again);later.update({k:again[k]+delta for k,delta in zip(m.COUNTERS,(128,1,20))})
    path,_=training.save_semantic_checkpoint(fresh,tmp_path/"later_synthetic.pt",later)
    last=training.load_semantic_checkpoint(make(),path,contract=new,seed=1001)
    assert last[m.MIGRATION]==verified and last[m.PRIOR]==original[m.PRIOR]
    assert last["rr_postcross_workspace_branch"]==original["rr_postcross_workspace_branch"]
    assert last["rr_capture_transfer_branch"]==original["rr_capture_transfer_branch"]



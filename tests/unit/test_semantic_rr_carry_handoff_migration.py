"""Focused strict same410 v4 factor and official identity/carry fixtures.
No real checkpoint, simulator, optimizer update or historical result is modified.
"""
from copy import deepcopy
from pathlib import Path
import ast
import inspect
import json
import pytest
import yaml
from test_semantic_rr_capture_knee_migration import boundary as knee_boundary
from test_semantic_rr_capture_knee_migration import git, write
from wlr50_clean.ppo import semantic_rr_carry_handoff_migration as m
from wlr50_clean.ppo.semantic_migration import digest


@pytest.fixture
def scope(monkeypatch):
    # A deliberately finite synthetic reviewed implementation, not a real v4 review.
    monkeypatch.setattr(m, "SOURCE_HEAD", "4" * 40)
    monkeypatch.setattr(m, "REVIEWED_BEHAVIOR_FILES", m.CANDIDATE_BEHAVIOR)
    monkeypatch.setattr(m, "TARGET_WHEEL_MODE", "synthetic_reviewed_forward_v1")
    monkeypatch.setattr(m, "REVIEWED_TARGET_LITERALS", {m.WHEEL: {"MODE": "synthetic_reviewed_forward_v1"}})


def boundary():
    meta = knee_boundary()[0]
    source_profile = {"revision":m.SOURCE_REVISION, "rr_capture_wheel_mode":"off",
        "rr_capture_feedback_revision":"window_peak_hip_then_knee_v3",
        "residual":{"must_preserve_all12":True}}
    source_task = {"geometry":{"must_preserve":True}, "reward_and_acceptance":"unchanged"}
    old = deepcopy(meta["runtime_contract"])
    old["source_git_commit"] = "4" * 40
    for path in m.scope() - {m.MODULE, m.WHEEL}:
        old["files"].setdefault(path, "a" * 64)
    old["runtime_content_sha256"] = digest(old["files"])
    for binding in old["selected_configuration"].values():
        binding["sha256"] = old["files"][binding["path"]]
    new = deepcopy(old); new["source_git_commit"] = "5" * 40
    new["files"].update({p:"d" * 64 for p in m.scope()})
    new["runtime_content_sha256"] = digest(new["files"])
    for binding in new["selected_configuration"].values():
        binding["sha256"] = new["files"][binding["path"]]
    meta["runtime_contract"] = old
    # Explicitly learned source: counts exceed the v3 publication origin.
    meta[m.PRIOR] = {"schema":"wlr50_clean.rr_capture_knee_same410.v3",
        "target_git_commit":old["source_git_commit"], "target_contract_sha256":digest(old),
        "target_runtime_content_sha256":old["runtime_content_sha256"],
        m.PRIOR_FACTOR:{"target_feedback_revision":"window_peak_hip_then_knee_v3",
            "counter_origin":{k:meta[k]-delta for k,delta in zip(m.COUNTERS,(128,1,20))}}}
    args = dict(reason="synthetic explicit v4 boundary", reviewed_code_sha256={p:new["files"][p] for p in m.scope()},
        source_profile=source_profile, target_profile={**source_profile,"revision":m.TARGET_REVISION,"rr_capture_wheel_mode":m.TARGET_WHEEL_MODE},
        source_task=source_task, target_task={**source_task,m.HANDOFF_KEY:m.HANDOFF_MODE},
        expected_source_head=old["source_git_commit"], expected_target_head=new["source_git_commit"])
    return meta,old,new,args


def test_unfinalized_draft_scope_is_not_runnable(monkeypatch):
    monkeypatch.setattr(m,"REVIEWED_BEHAVIOR_FILES",None)
    with pytest.raises(ValueError,match="not yet"):
        m.scope()


def test_learned_same410_source_is_allowed_without_zero_update_origin_lie(scope):
    meta,old,new,args=boundary(); original=deepcopy(meta)
    value=m.factor(meta,old,new,**args)
    assert meta == original and value["counter_origin"] == {k:meta[k] for k in m.COUNTERS}
    assert not value["creates_new_branch"] and not value["same_mdp_claimed"]
    assert value["observation_contract"]["source_policy_contract"] == value["observation_contract"]["target_policy_contract"]
    assert value["added_policy_decisions"] == value["added_ppo_updates"] == value["added_optimizer_steps"] == value["added_auxiliary_updates"] == 0
    assert value["source_v3_receipt_sha256"] == digest(meta[m.PRIOR])
    assert all(value["preserved_metadata_sha256"][k] == digest(meta[k]) for k in m.preserved_keys(meta))
    assert "source_selection" not in value and "candidate_evaluation_only" not in value


def test_source_registration_is_exact_and_independent():
    learned=m.registered_source(m.SOURCE_CHECKPOINT_SHA256)
    ancestor=m.registered_source(m.ANCESTOR_CHECKPOINT_SHA256)
    assert learned["manifest_sha256"]==m.SOURCE_MANIFEST_SHA256
    assert tuple(learned["counters"].values())==(221184,1693,33860)
    assert ancestor["manifest_sha256"]==m.ANCESTOR_MANIFEST_SHA256
    assert tuple(ancestor["counters"].values())==(220544,1688,33760)
    assert ancestor["source_role"]==m.ANCESTOR_ROLE
    with pytest.raises(ValueError,match="two explicitly registered"):
        m.registered_source("9"*64)


@pytest.mark.parametrize("bad",[None,"role","count","origin","sha","rr_credit"])
def test_ancestor_selection_has_zero_new_credit_not_latest_claim(scope,monkeypatch,bad):
    meta,old,new,args=boundary()
    monkeypatch.setattr(m,"ANCESTOR_COUNTERS",tuple(meta[k] for k in m.COUNTERS))
    meta["checkpoint_sha256"]=m.ANCESTOR_CHECKPOINT_SHA256
    meta[m.PRIOR][m.PRIOR_FACTOR]["counter_origin"]={k:meta[k] for k in m.COUNTERS}
    selection=m.registered_source(m.ANCESTOR_CHECKPOINT_SHA256)
    if bad=="role": selection["source_role"]="latest_learned_continuation"
    elif bad=="count": selection["counters"]["ppo_updates"]+=1
    elif bad=="origin": meta[m.PRIOR][m.PRIOR_FACTOR]["counter_origin"]["ppo_updates"]-=1
    elif bad=="sha": meta["checkpoint_sha256"]=m.SOURCE_CHECKPOINT_SHA256
    elif bad=="rr_credit":
        meta["rr_capture_transfer_branch"]["counter_origin"]["ppo_updates"]-=1
        meta["rr_capture_transfer_branch_counts"]["ppo_updates"]=1
    if bad:
        with pytest.raises(ValueError): m.factor(meta,old,new,**args,source_selection=selection)
    else:
        value=m.factor(meta,old,new,**args,source_selection=selection)
        assert value["source_selection"]==selection and value["candidate_evaluation_only"]
        assert not value["latest_learned_policy_equivalence_claimed"]
        assert not value["latest_pointer_promotion_authorized"]
        assert value["added_policy_decisions"]==value["added_ppo_updates"]==value["added_optimizer_steps"]==0


@pytest.mark.parametrize("bad", ["repeat","source_head","target_head","source_sha_not_a_plan_argument",
    "prior_runtime","future_origin","branch_counts","extra_path","missing_path","new_hidden_module",
    "assist_change","reward_config","task_extra","profile_extra","wheel_off","literal_scope_missing"])
def test_reject_unreviewed_state_or_scope(scope,bad):
    meta,old,new,args=boundary()
    if bad=="repeat": meta[m.MIGRATION]={}
    elif bad=="source_head": args["expected_source_head"]="6"*40
    elif bad=="target_head": args["expected_target_head"]=None
    elif bad=="source_sha_not_a_plan_argument": args["reviewed_code_sha256"]["unreviewed"]="0"*64
    elif bad=="prior_runtime": meta[m.PRIOR]["target_contract_sha256"]="0"*64
    elif bad=="future_origin": meta[m.PRIOR][m.PRIOR_FACTOR]["counter_origin"]["ppo_updates"]=meta["ppo_updates"]+1
    elif bad=="branch_counts": meta["rr_capture_transfer_branch_counts"]["ppo_updates"]+=1
    elif bad in ("extra_path","assist_change","reward_config","new_hidden_module"):
        path={"extra_path":"src/wlr50_clean/ppo/semantic_reward.py",
            "assist_change":"src/wlr50_clean/ppo/semantic_rr_capture_assist.py",
            "reward_config":f"configs/ppo_{m.EXPERIMENT}/reward_config.yaml",
            "new_hidden_module":"src/wlr50_clean/ppo/hidden_state.py"}[bad]
        new["files"][path]=args["reviewed_code_sha256"][path]="9"*64
    elif bad=="missing_path":
        path=next(iter(m.HOOKS));new["files"][path]=old["files"][path];args["reviewed_code_sha256"].pop(path)
    elif bad=="task_extra": args["target_task"]["geometry"]={"changed":True}
    elif bad=="profile_extra": args["target_profile"]["residual"]={"changed":True}
    elif bad=="wheel_off": args["target_profile"]["rr_capture_wheel_mode"]="off"
    else: args["reviewed_code_sha256"]={}
    with pytest.raises(ValueError): m.factor(meta,old,new,**args)


def test_source_sha_is_mandatory_and_exact_before_checkpoint_loading(scope,tmp_path):
    path=tmp_path/"unrelated.pt";path.write_bytes(b"not a checkpoint")
    for bound in (None,"","short","0"*64):
        with pytest.raises(ValueError):
            m.build_rr_carry_handoff_migration(path,{},expected_source_sha256=bound,
                expected_source_head="4"*40,expected_target_head="5"*40,
                reason="synthetic",reviewed_code_sha256={})


def test_identity_leaf_official_save_reload_and_receipt_carry(scope,tmp_path):
    torch=pytest.importorskip("torch");pytest.importorskip("rsl_rl")
    from wlr50_clean.ppo import semantic_training as training
    from wlr50_clean.ppo.semantic_migration import checkpoint_metadata
    from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
    assert not torch.cuda.is_available()
    def make():
        return training.construct_semantic_runner(_shape_env(410,"cpu"),seed=1001,device="cpu",
            policy_version=m.RR_CAPTURE_POLICY,observation_layout=m.RR_CAPTURE_OBSERVATION_LAYOUT,
            initialize_actor=False)[0]
    runner=make();meta,old,new,args=boundary()
    runner.alg.learning_rate=2.25e-5
    for group in runner.alg.optimizer.param_groups:
        group["lr"]=2.25e-5
        for p in group["params"]:
            runner.alg.optimizer.state[p]={"step":torch.tensor(9.),"exp_avg":torch.full_like(p,.001),
                                           "exp_avg_sq":torch.full_like(p,.002)}
    source,_=training.save_semantic_checkpoint(runner,tmp_path/"synthetic_source.pt",meta)
    sm=checkpoint_metadata(source);fresh=make()
    infos=training.load_semantic_checkpoint(fresh,source,contract=old,seed=1001)
    value=m.factor(sm,old,new,**args)
    verified={"schema":m.SCHEMA,m.FACTOR_KEY:value}
    loaded=m.record_loaded_rr_carry_handoff(fresh,infos,verified)
    assert all(loaded[k]==sm[k] for k in m.preserved_keys(sm))
    assert loaded[m.MIGRATION]==verified
    # Isolate official save/reload and carry, NOT a real migration-builder publication.
    target,_=training.save_semantic_checkpoint(fresh,tmp_path/"synthetic_receipt_carry.pt",loaded)
    again=training.load_semantic_checkpoint(make(),target,contract=old,seed=1001)
    assert again[m.MIGRATION]==verified and all(again[k]==sm[k] for k in m.preserved_keys(sm))
    carry=[ast.literal_eval(n.iter) for n in ast.walk(ast.parse(inspect.getsource(training.train_semantic)))
        if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple)
        and any(isinstance(x,ast.Constant) and x.value=="new_mdp_warm_start" for x in n.iter.elts)]
    assert len(carry)==1 and m.MIGRATION in carry[0]
    restored=make();same=training.load_semantic_checkpoint(restored,source,contract=old,seed=1001)
    restored.alg.storage.step=1
    with pytest.raises(RuntimeError,match="rollout"):
        m.record_loaded_rr_carry_handoff(restored,same,verified)


@pytest.mark.parametrize("ancestor",[False,True])
def test_generic_plan_official_load_publish_fresh_reload(scope,tmp_path,monkeypatch,ancestor):
    """Exercise the actual generic route, not a substituted validator result."""
    torch=pytest.importorskip("torch");pytest.importorskip("rsl_rl")
    from wlr50_clean.ppo import semantic_training as training, semantic_migration as shared
    from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
    assert not torch.cuda.is_available()
    meta,old,new,args=boundary()
    root=tmp_path/"synthetic_git";root.mkdir()
    git(root,"init");git(root,"config","user.email","test@example.invalid")
    git(root,"config","user.name","Synthetic CPU test");git(root,"config","core.autocrlf","false")
    paths=set(old["files"])
    for path in paths:
        content=(yaml.safe_dump(args["source_profile"]) if path==m.PROFILE else
                 yaml.safe_dump(args["source_task"]) if path==m.TASK else "# synthetic source\n")
        write(root/path,content)
    git(root,"add",".");git(root,"commit","-m","synthetic source")
    def bind(template):
        value=deepcopy(template)
        value["source_git_commit"]=git(root,"rev-parse","HEAD")
        value["files"]={p:shared.file_sha(root/p) for p in paths}
        value["runtime_content_sha256"]=digest(value["files"])
        for binding in value["selected_configuration"].values():
            binding["sha256"]=value["files"][binding["path"]]
        return value
    old=bind(old);monkeypatch.setattr(m,"SOURCE_HEAD",old["source_git_commit"])
    paths.update((m.MODULE,m.WHEEL))
    for path in m.scope():
        content=(yaml.safe_dump(args["target_profile"]) if path==m.PROFILE else
                 yaml.safe_dump(args["target_task"]) if path==m.TASK else
                 "\n".join(f"{k} = {v!r}" for k,v in m.REVIEWED_TARGET_LITERALS[path].items())+"\n"
                 if path in m.REVIEWED_TARGET_LITERALS else "# synthetic reviewed target\n")
        write(root/path,content)
    git(root,"add",".");git(root,"commit","-m","synthetic target")
    new=bind(new);meta["runtime_contract"]=old
    meta[m.PRIOR].update(target_git_commit=old["source_git_commit"],target_contract_sha256=digest(old),
                        target_runtime_content_sha256=old["runtime_content_sha256"])
    if ancestor:
        meta[m.PRIOR][m.PRIOR_FACTOR]["counter_origin"]={k:meta[k] for k in m.COUNTERS}
    def make():
        return training.construct_semantic_runner(_shape_env(410,"cpu"),seed=1001,device="cpu",
            policy_version=m.RR_CAPTURE_POLICY,observation_layout=m.RR_CAPTURE_OBSERVATION_LAYOUT,
            initialize_actor=False)[0]
    runner=make();runner.alg.learning_rate=1e-5
    for group in runner.alg.optimizer.param_groups:
        group["lr"]=1e-5
        for parameter in group["params"]:
            runner.alg.optimizer.state[parameter]={"step":torch.tensor(9.),
                "exp_avg":torch.full_like(parameter,.001),"exp_avg_sq":torch.full_like(parameter,.002)}
    source,manifest=training.save_semantic_checkpoint(runner,tmp_path/"source.pt",meta)
    original=shared.checkpoint_metadata(source)
    prefix="ANCESTOR" if ancestor else "SOURCE"
    monkeypatch.setattr(m,prefix+"_CHECKPOINT_SHA256",shared.file_sha(source))
    monkeypatch.setattr(m,prefix+"_MANIFEST_SHA256",shared.file_sha(manifest))
    monkeypatch.setattr(m,prefix+"_COUNTERS",tuple(meta[k] for k in m.COUNTERS))
    kwargs=dict(expected_source_sha256=shared.file_sha(source),expected_source_head=old["source_git_commit"],
        expected_target_head=new["source_git_commit"],reason="synthetic exact v4 publication",
        reviewed_code_sha256={p:new["files"][p] for p in m.scope()},project_root=root)
    plan=m.build_rr_carry_handoff_migration(source,new,**kwargs)
    assert ("source_selection" in plan)==ancestor
    assert ("source_selection" in plan[m.FACTOR_KEY])==ancestor
    path=tmp_path/"plan.json";write(path,json.dumps(plan))
    original_validator=shared.validate_migration_plan
    monkeypatch.setattr(shared,"validate_migration_plan",lambda c,contract,p,**kw:
        original_validator(c,contract,p,project_root=root))
    verified=shared.validate_migration_plan(source,new,path)
    bad=deepcopy(plan);bad[m.FACTOR_KEY]["same_mdp_claimed"]=True
    bad_path=tmp_path/"tampered.json";write(bad_path,json.dumps(bad))
    with pytest.raises(ValueError,match="differs"):
        shared.validate_migration_plan(source,new,bad_path)
    # Valid hashes but false declared semantics still fail before loading.
    wheel=root/m.WHEEL;bound=wheel.read_bytes()
    write(wheel,"MODE = 'wrong'\n")
    with pytest.raises(ValueError):
        m.build_rr_carry_handoff_migration(source,new,**kwargs)
    wheel.write_bytes(bound)
    migrated=make()
    loaded=training.load_semantic_checkpoint(migrated,source,contract=new,seed=1001,migration=verified)
    assert loaded[m.MIGRATION]==verified
    assert all(loaded[k]==original[k] for k in m.preserved_keys(original))
    loaded.update(runtime_contract=new,old_rollout_inherited=False)
    target,_=training.save_semantic_checkpoint(migrated,tmp_path/"target_unique.pt",loaded)
    fresh=make();again=training.load_semantic_checkpoint(fresh,target,contract=new,seed=1001)
    assert again[m.MIGRATION]==verified and again["resume_migration"]==verified
    assert all(again[k]==original[k] for k in m.preserved_keys(original))
    assert fresh.alg.storage.step==0 and fresh.alg.transition.actions is None
    probes=_shape_env(410,"cpu").get_observations()
    probes["policy"][...,0]=0.;probes["policy"][...,8]=1.;probes["policy"][...,409]=1.
    probes["critic"]=probes["policy"].clone()
    with torch.no_grad():
        runner.alg.actor(probes,stochastic_output=True)
        before=tuple(x.clone() for x in runner.alg.actor.output_distribution_params)
        fresh.alg.actor(probes,stochastic_output=True)
        assert all(torch.equal(a,b) for a,b in zip(before,fresh.alg.actor.output_distribution_params))
        assert torch.equal(runner.alg.critic(probes),fresh.alg.critic(probes))

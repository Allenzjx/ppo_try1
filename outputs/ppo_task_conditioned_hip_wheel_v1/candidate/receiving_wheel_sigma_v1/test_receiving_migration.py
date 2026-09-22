"""One bounded full-state CPU migration fixture; no production or physical credit."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch
from receiving_test_bootstrap import ROOT, HERE, MODULES
from test_receiving_integration import runner, minimal_contract, OLD, NEW

m, t, cli = (MODULES[k] for k in ("semantic_migration","semantic_training","semantic_cli"))
CP = ROOT / "outputs/ppo_task_conditioned_hip_wheel_v1/checkpoints/history/checkpoint_step_000196608.pt"
REVISION = "b"*40
CODE = m.RECEIVING_WHEEL_SIGMA_NEW_FILES | m.RECEIVING_WHEEL_SIGMA_EXISTING_FILES


def make_fixture(tmp_path, monkeypatch, checkpoint=None):
    cp=Path(checkpoint or CP).resolve()
    metadata=m.checkpoint_metadata(cp)
    old,new=copy.deepcopy(metadata["runtime_contract"]),copy.deepcopy(metadata["runtime_contract"])
    root=tmp_path/"CPU_NON_DEPLOYABLE_RUNTIME"
    real_bytes,real_run=m._version_bytes,subprocess.run
    for relative in old["files"].keys() | m.RECEIVING_WHEEL_SIGMA_NEW_FILES:
        content=(HERE/relative).read_bytes() if relative in CODE else real_bytes(ROOT,old,relative,prefer_worktree=True)
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        new["files"][relative]=hashlib.sha256(content).hexdigest()
    new["source_git_commit"]=REVISION
    new["runtime_content_sha256"]=m.digest(new["files"])
    def version_bytes(project,contract,relative,*,prefer_worktree=False):
        if contract["source_git_commit"]==old["source_git_commit"]:
            return real_bytes(ROOT,contract,relative,prefer_worktree=True)
        return real_bytes(project,contract,relative,prefer_worktree=prefer_worktree)
    def run(command,*args,**kwargs):
        if command==["git","-C",str(root),"rev-parse","HEAD"]:return SimpleNamespace(stdout=REVISION)
        return real_run(command,*args,**kwargs)
    monkeypatch.setattr(m,"_version_bytes",version_bytes)
    monkeypatch.setattr(m.subprocess,"run",run)
    return SimpleNamespace(checkpoint=cp,metadata=metadata,old=old,new=new,project_root=root)


def plan(f,**options):
    return m.build_migration_plan(f.checkpoint,f.new,allowed_changed_files=sorted(CODE),
        reason="CPU synthetic future runtime only; not a deployable reviewed plan",
        receiving_wheel_sigma_review={"reason":"CPU test of explicit sigma-only boundary",
            "reviewed_code_sha256":{p:f.new["files"][p] for p in sorted(CODE)}},
        project_root=f.project_root,**options)


def test_actual_source_metadata_and_target_scope_with_bounded_negative_cases(tmp_path,monkeypatch):
    f=make_fixture(tmp_path,monkeypatch); result=plan(f)
    factor=result["receiving_wheel_sigma_factor"]
    assert factor["source_policy_version"]==OLD and factor["target_policy_version"]==NEW
    assert factor["kernel_changed"] and factor["same_mdp_claimed"] and not factor["new_mdp"]
    assert factor["deterministic_mean_changed"] is factor["history_kernel_changed"] is False
    assert all(row["bytes_identical"] for row in factor["configuration_bindings"].values())
    assert factor["preserved_branch_metadata"]["task_conditioned_hip_wheel_branch"]==f.metadata["task_conditioned_hip_wheel_branch"]
    assert factor["source_effective_learning_rate"]==f.metadata["optimizer_learning_rate"]
    path=tmp_path/"CPU_only_plan.json";path.write_text(json.dumps(result),encoding="utf-8")
    assert m.validate_migration_plan(f.checkpoint,f.new,path,project_root=f.project_root)["receiving_wheel_sigma_factor"]==factor
    for field,value in (("new_mdp",True),("old_rollout_inherited",True),("migration_added_updates",1),
                        ("source_stage_requested_decisions",{}),("preserved_auxiliary_ledger_sha256","0"*64)):
        changed=copy.deepcopy(result);changed["receiving_wheel_sigma_factor"][field]=value
        path.write_text(json.dumps(changed),encoding="utf-8")
        with pytest.raises(ValueError,match="exactly bound"):
            m.validate_migration_plan(f.checkpoint,f.new,path,project_root=f.project_root)
    for field,value in (("experiment_id","fl_capture_quality_v1"),("decision_hz",30.),
                        ("training_budgets",{"smoke":10000,"phase_suffix":100000,"full_episode":131073})):
        original=f.new[field];f.new[field]=value
        with pytest.raises(ValueError):plan(f)
        f.new[field]=original
    with pytest.raises(ValueError):plan(f,archive_only_exact_bytes_review={"reason":"mixed"})
    real_metadata=m.checkpoint_metadata
    for field in ("branch_count","aux_count","lr"):
        altered=copy.deepcopy(f.metadata)
        if field=="branch_count":altered["task_conditioned_hip_wheel_branch_counts"]["ppo_updates"]+=1
        if field=="aux_count":altered["task_conditioned_hip_wheel_branch"]["auxiliary_mean_learning"]["accepted_auxiliary_updates_total"]=8
        if field=="lr":altered["optimizer_learning_rate"]=0
        monkeypatch.setattr(m,"checkpoint_metadata",lambda _:altered)
        with pytest.raises(ValueError):plan(f)
    monkeypatch.setattr(m,"checkpoint_metadata",real_metadata)
    # Old migration helpers and budget/reward/control logic are outside this extension.
    before=ast.parse((ROOT/"src/wlr50_clean/ppo/semantic_migration.py").read_text(encoding="utf-8"))
    after=ast.parse((HERE/"src/wlr50_clean/ppo/semantic_migration.py").read_text(encoding="utf-8"))
    nodes={n.name:ast.dump(n,include_attributes=False) for n in after.body if isinstance(n,ast.FunctionDef)}
    for node in before.body:
        if isinstance(node,ast.FunctionDef) and node.name not in ("build_migration_plan","validate_migration_plan"):
            assert nodes[node.name]==ast.dump(node,include_attributes=False),node.name


def test_official_fullstate_migrate_fresh128_update_save_exact_resume(tmp_path,monkeypatch):
    rng,threads=torch.get_rng_state(),torch.get_num_threads();torch.set_num_threads(1)
    try:
        t.seed_training_rngs(1001)
        source,env=runner(OLD)
        first=t.train_semantic(source,env,run_dir=tmp_path/"cpu_source",output_root=tmp_path/"cpu_source_out",
            stage="full_episode",decisions=128,contract=minimal_contract(),seed=1001)
        real=m.checkpoint_metadata(CP)
        infos=copy.deepcopy(real)
        for key in ("checkpoint_path","checkpoint_sha256","save_load_round_trip"):infos.pop(key)
        infos["CPU_SYNTHETIC_MODEL_WITH_REFERENCE_METADATA_NOT_ADOPTABLE"]=True
        source.alg.learning_rate=2.25e-5
        for group in source.alg.optimizer.param_groups:group.update(lr=2.25e-5,betas=(.87,.996),eps=2e-8)
        source_cp=tmp_path/"cpu_source_state.pt";t.save_semantic_checkpoint(source,source_cp,infos)
        source_meta=m.checkpoint_metadata(source_cp)
        f=make_fixture(tmp_path,monkeypatch,checkpoint=source_cp)
        supplied=plan(f);plan_path=tmp_path/"CPU_non_deployable_sigma_plan.json"
        plan_path.write_text(json.dumps(supplied),encoding="utf-8")
        validate=m.validate_migration_plan
        monkeypatch.setattr(m,"validate_migration_plan",lambda cp,contract,path:
            validate(cp,contract,path,project_root=f.project_root))
        verified=m.validate_migration_plan(source_cp,f.new,plan_path)
        monkeypatch.setattr(cli,"_request_paths",lambda args:(tmp_path,tmp_path,f.project_root/"configs/ppo_task_conditioned_hip_wheel_v1"))
        args=SimpleNamespace(checkpoint=source_cp,semantic_version="v3",experiment_id="task_conditioned_hip_wheel_v1",
            command="train",num_envs=1,stage="full_episode",from_phase="P01",teacher_offset_decisions=0,
            prefix_source="frozen_fsm",seed=1001,device="cpu",new_mdp_warm_start=False,
            resume_migration=plan_path,policy_distribution_migration=False,target_policy_version=None)
        cli._preflight_checkpoint(args,f.new)
        assert args._policy_version==NEW
        for fields in ({"command":"eval"},{"num_envs":8},{"from_phase":"P12","stage":"phase_suffix"},
                       {"teacher_offset_decisions":35},{"prefix_source":"successful_nominal"}):
            bad=copy.copy(args)
            for key,value in fields.items():setattr(bad,key,value)
            with pytest.raises(ValueError,match="receiving-wheel sigma first"):
                cli._preflight_checkpoint(bad,f.new)
        for fault in ("missing_plan","partial","pending"):
            rejected,_=runner()
            if fault=="partial":rejected.alg.storage.step=1
            if fault=="pending":rejected.alg.transition.actions=torch.zeros(1,12)
            with pytest.raises((ValueError,RuntimeError)):
                t.load_semantic_checkpoint(rejected,source_cp,contract=f.new,seed=1001,
                    migration=None if fault=="missing_plan" else verified)
        target,newenv=runner()
        loaded=t.load_semantic_checkpoint(target,source_cp,contract=f.new,seed=1001,migration=verified)
        assert target.alg.storage.step==0 and target.alg.transition.actions is None and newenv.core.calls==0
        assert tuple(target.alg.storage.actions.shape)==(128,1,12)
        assert tuple(target.alg.storage.observations["policy"].shape)==(128,1,372)
        assert target.alg.learning_rate==t.optimizer_learning_rate(target)==2.25e-5
        assert all(g["lr"]==2.25e-5 and g["betas"]==(.87,.996) and g["eps"]==2e-8 for g in target.alg.optimizer.param_groups)
        assert t.parameter_hash(target.alg.actor)==source_meta["actor_parameter_sha256"]
        assert t.parameter_hash(target.alg.critic)==source_meta["critic_parameter_sha256"]
        assert t.state_hash(target.alg.optimizer.state_dict())==source_meta["optimizer_state_sha256"]
        assert t.state_hash(t._normalizers(target))==source_meta["normalizer_state_sha256"]
        assert t.capture_training_rng_state(seed=1001)==source_meta["training_rng_state"]
        observation=newenv.get_observations(); saved_rng=torch.get_rng_state()
        assert torch.equal(target.alg.actor(observation),source.alg.actor(observation))
        assert torch.equal(saved_rng,torch.get_rng_state())
        for key in (*verified["receiving_wheel_sigma_factor"]["preserved_branch_metadata"],
                    "global_policy_decisions","ppo_updates","optimizer_steps","stage_requested_decisions"):
            assert loaded[key]==source_meta[key]
        follow=t.train_semantic(target,newenv,run_dir=tmp_path/"cpu_target",output_root=tmp_path/"cpu_target_out",
            stage="full_episode",decisions=128,contract=f.new,seed=1001,resume_infos=loaded)
        final_cp=Path(follow["checkpoints"][-1]["checkpoint"])
        final,finalenv=runner(); finalinfo=t.load_semantic_checkpoint(final,final_cp,contract=f.new,seed=1001)
        assert finalinfo["global_policy_decisions"]==source_meta["global_policy_decisions"]+128
        assert finalinfo["ppo_updates"]==source_meta["ppo_updates"]+1
        assert finalinfo["optimizer_steps"]==source_meta["optimizer_steps"]+20
        assert finalinfo["task_conditioned_hip_wheel_branch"]==source_meta["task_conditioned_hip_wheel_branch"]
        assert finalinfo["receiving_wheel_sigma_migration"]==loaded["receiving_wheel_sigma_migration"]
        assert finalinfo["training_quantity_budget_extension"]==loaded["training_quantity_budget_extension"]
        assert t.state_hash(final.alg.optimizer.state_dict())==t.state_hash(target.alg.optimizer.state_dict())
        args.checkpoint=final_cp;args.resume_migration=None
        cli._preflight_checkpoint(args,f.new)
        assert args._policy_version==NEW and finalenv.core.calls==0
        provenance={"checkpoint_path":str(final_cp.resolve()),"checkpoint_sha256":t.sha256_file(final_cp),
            "actor_parameter_sha256":finalinfo["actor_parameter_sha256"],
            "source_global_policy_decisions":finalinfo["global_policy_decisions"],
            "source_ppo_updates":finalinfo["ppo_updates"],"policy_contract":finalinfo["policy_contract"],
            "source_runtime_content_sha256":f.new["runtime_content_sha256"],
            **cli._request_history_prefix_provenance(args,f.new,finalinfo)}
        prefix_module=MODULES["semantic_checkpoint_prefix_policy"]
        frozen=prefix_module.build_frozen_checkpoint_prefix_policy(final.alg.actor,provenance)
        observation=finalenv.get_observations(); saved_rng=torch.get_rng_state()
        expected=final.alg.actor(observation)[0].tolist()
        assert list(frozen(tuple(observation["policy"][0].tolist())))==expected
        assert torch.equal(saved_rng,torch.get_rng_state())
        bad=copy.deepcopy(provenance);bad["source_policy_contract"]["version"]=OLD
        with pytest.raises(ValueError):prefix_module.build_frozen_checkpoint_prefix_policy(final.alg.actor,bad)
    finally:
        torch.set_rng_state(rng);torch.set_num_threads(threads)

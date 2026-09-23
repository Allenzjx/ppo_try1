"""Synthetic CPU identity/branch tests, never robot-training credit."""
import copy
import json
from pathlib import Path
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from wlr50_clean.ppo import semantic_training as t
from wlr50_clean.ppo import semantic_rear_recapture_migration as m
from wlr50_clean.ppo import semantic_rear_policy_timing_migration as initial
from wlr50_clean.ppo.semantic_migration import digest, continuation_topology
from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
from wlr50_clean.ppo.semantic_rear_policy_timing_profile import (
    REAR_POLICY_TIMING_POLICY as POLICY, REAR_POLICY_TIMING_OBSERVATION_LAYOUT as LAYOUT)
from test_semantic_rear_policy_training_audit import Synthetic419Core


def make(env=None):
    return t.construct_semantic_runner(env or _shape_env(419,"cpu"),
        seed=1001,device="cpu",initialize_actor=False,policy_version=POLICY,observation_layout=LAYOUT)[0]


def fixture(tmp_path, monkeypatch):
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES=-1"
    torch.set_num_threads(1)
    runner = make()
    for parameter in list(runner.alg.actor.parameters()) + list(runner.alg.critic.parameters()):
        parameter.grad = torch.ones_like(parameter)
    runner.alg.optimizer.step(); runner.alg.optimizer.zero_grad()
    for state in runner.alg.optimizer.state.values():
        state["step"].fill_(33760.)
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = 2.25e-5
    runner.alg.learning_rate = 2.25e-5
    runner.current_learning_iteration = 1688
    counts = dict(initial.SOURCE_COUNTS)  # Synthetic fixture only.
    old = dict(experiment_id=m.EXPERIMENT,source_git_commit=m.SOURCE_HEAD,semantic_version="v3",
        training_budgets=t.training_quantity_budgets(m.EXPERIMENT))
    new = {**old,"source_git_commit":"b"*40}
    ancestor = dict(schema=initial.SCHEMA,target_contract_sha256=digest(old),
        source_checkpoint_sha256=initial.SOURCE_SHA)
    infos = {**counts,"seed":1001,"runtime_contract":old,"semantic_version":"v3",
        "stage_requested_decisions":dict(smoke=0,full_episode=128,phase_suffix=0),
        "sampling":"P01_full_task_only_initial_version",
        "rear_policy_timing_migration":ancestor,
        "rear_policy_timing_branch":{"schema":initial.SCHEMA,"counter_origin":initial.SOURCE_COUNTS,
            "source_checkpoint_sha256":initial.SOURCE_SHA},
        "rr_postcross_workspace_branch":{"counter_origin":initial.SOURCE_COUNTS,
            "front_retention_auxiliary":{"accepted":32,"attempted":32}},
        "p05_capture_assist_migration":{"immutable":{"accepted":7,"attempted":8}},
        "execution_topology":continuation_topology("P01_full_task_only_initial_version",None,observation_layout=LAYOUT)}
    cp, side = t.save_semantic_checkpoint(runner,tmp_path/"source.pt",infos)
    metadata = json.loads(side.read_text())
    factor = dict(schema=m.SCHEMA,source_policy_contract=metadata["policy_contract"],
        source_selection=copy.deepcopy(m.SOURCE_SELECTION),
        target_policy_contract=metadata["policy_contract"],target_mode=m.TARGET_MODE,
        revision_counter_origin=counts,original_branch_origin=initial.SOURCE_COUNTS,
        preserved_metadata_sha256={k:digest(metadata[k]) for k in m._preserved(metadata)},
        source_effective_learning_rate=2.25e-5,
        added_policy_decisions=0,added_ppo_updates=0,added_optimizer_steps=0,added_auxiliary_updates=0)
    record = dict(schema=m.SCHEMA,plan_path=str(tmp_path/"plan.json"),source_git_commit=m.SOURCE_HEAD,
        source_checkpoint_sha256=m.INITIAL_SOURCE_SHA,source_manifest_sha256=m.INITIAL_MANIFEST_SHA,
        target_git_commit=new["source_git_commit"],target_runtime_content_sha256=new.get("runtime_content_sha256"),
        source_selection=copy.deepcopy(m.SOURCE_SELECTION),
        source_contract_sha256=digest(old),target_contract_sha256=digest(new),**{m.FACTOR_KEY:factor})
    # Isolate identity load/save wiring; source/target plan checks have separate negative tests.
    monkeypatch.setattr(m,"validate_rear_recapture_migration",lambda *a,**k:record)
    return cp,metadata,new,record


def test_same419_identity_publish_freshload_and_ordinary_ppo_carry(tmp_path,monkeypatch):
    cp,metadata,new,record=fixture(tmp_path,monkeypatch)
    outroot=tmp_path/("ppo_"+m.EXPERIMENT)
    output=outroot/"checkpoints/history/published.pt"
    result=m.publish_rear_recapture_checkpoint(cp,new,tmp_path/"plan.json",output)
    assert result["save_load_round_trip"] and result["added_ppo_updates"] == 0
    before=torch.load(cp,map_location="cpu",weights_only=False)
    after=torch.load(output,map_location="cpu",weights_only=False)
    assert t.state_hash({k:v for k,v in before.items() if k!="infos"}) == t.state_hash({k:v for k,v in after.items() if k!="infos"})
    current=json.loads(output.with_name(output.stem+"_manifest.json").read_text())
    for key in m._preserved(metadata):
        assert current[key] == metadata[key]
    assert current[m.MIGRATION] == record
    assert current["optimizer_learning_rate"] == 2.25e-5
    initial.validate_rear_policy_namespace(current,new,outroot)
    env=t.SemanticRslAdapter(Synthetic419Core(),seed=1001,device="cpu")
    env.cfg["semantic_version"]="v3"
    learner=make(env)
    loaded=t.load_semantic_checkpoint(learner,output,contract=new,seed=1001)
    run=tmp_path/"synthetic_run"
    branch=outroot/"branches"/m.BRANCH_NAME
    route=initial.build_rear_policy_output_routing(loaded,new,branch)
    parent_pointer=outroot/"checkpoints/checkpoint_last_pointer.json"
    assert not parent_pointer.exists()
    with pytest.raises(ValueError,match="explicit output branch"):
        t.train_semantic(learner,env,run_dir=run,output_root=outroot,stage="full_episode",
            decisions=128,contract=new,seed=1001,resume_infos=loaded)
    trained=t.train_semantic(learner,env,run_dir=run,output_root=branch,stage="full_episode",
        decisions=128,contract=new,seed=1001,resume_infos=loaded,checkpoint_interval_updates=1,
        checkpoint_output_routing=route)
    assert (trained["actual_policy_decisions"],trained["ppo_updates_this_run"],
        trained["optimizer_steps_this_run"]) == (128,1,20)
    finalpath=Path(trained["checkpoints"][-1]["checkpoint"])
    final=json.loads(finalpath.with_name(finalpath.stem+"_manifest.json").read_text())
    assert final[m.MIGRATION] == current[m.MIGRATION]
    assert final["rear_policy_timing_migration"] == current["rear_policy_timing_migration"]
    assert final["rr_postcross_workspace_branch"] == current["rr_postcross_workspace_branch"]
    assert final["p05_capture_assist_migration"] == current["p05_capture_assist_migration"]
    assert tuple(final[k] for k in m.COUNTERS) == (220672,1689,33780)
    assert final["rear_policy_timing_branch"]["counter_origin"] == initial.SOURCE_COUNTS
    assert final["save_load_round_trip"]
    assert not parent_pointer.exists()
    assert final["checkpoint_output_routing"] == route
    initial.validate_rear_policy_namespace(final,new,branch,checkpoint_output_routing=route)
    restored=make()
    t.load_semantic_checkpoint(restored,finalpath,contract=new,seed=1001)
    assert t.state_hash(restored.alg.optimizer.state_dict()) == final["optimizer_state_sha256"]
    assert Path(trained["checkpoints"][-1]["checkpoint"]).is_relative_to(branch)
    with pytest.raises(FileExistsError):
        t.train_semantic(learner,env,run_dir=tmp_path/"collision",output_root=branch,stage="full_episode",
            decisions=128,contract=new,seed=1001,resume_infos=loaded,checkpoint_output_routing=route)
    rollout=torch.load(run/"rollouts/rollout_001689.pt",weights_only=False)
    assert rollout["observations"]["policy"].shape == (128,1,419)


def test_lineage_rejects_old_runtime_or_tampered_parent(tmp_path,monkeypatch):
    cp,metadata,new,record=fixture(tmp_path,monkeypatch)
    amended={**metadata,"runtime_contract":new,m.MIGRATION:record}
    output=tmp_path/("ppo_"+m.EXPERIMENT)
    initial.validate_rear_policy_namespace(amended,new,output)
    with pytest.raises(ValueError):
        initial.validate_rear_policy_namespace(amended,metadata["runtime_contract"],output)
    with pytest.raises(ValueError):
        initial.validate_rear_policy_namespace(metadata,new,output)
    broken=copy.deepcopy(amended)
    broken["rear_policy_timing_migration"]["source_checkpoint_sha256"]="0"*64
    with pytest.raises(ValueError):
        initial.validate_rear_policy_namespace(broken,new,output)
    broken=copy.deepcopy(amended)
    broken[m.MIGRATION][m.FACTOR_KEY]["added_ppo_updates"]=1
    with pytest.raises(ValueError):
        initial.validate_rear_policy_namespace(broken,new,output)


def test_scope_does_not_allow_network_reward_or_physics_rewrites():
    for path in ("semantic_rear_policy_timing_actor.py","semantic_rear_policy_timing_profile.py",
            "semantic_observation.py","semantic_reward.py","semantic_residual_adapter.py"):
        assert m.CODE+path not in m.ALLOWED
    assert m.CONFIG+"reward_config.yaml" not in m.ALLOWED
    assert m.CONFIG+"observation_schema.json" not in m.ALLOWED


def test_media_delta_requires_exact_independent_review(monkeypatch):
    pins={path:dict(before="a"*64,after="b"*64) for path in m.MEDIA_REVIEW}
    old={"files":{p:v["before"] for p,v in pins.items()}}
    new={"files":{p:v["after"] for p,v in pins.items()}}
    monkeypatch.setattr(m,"MEDIA_REVIEW",pins)
    assert m._reviewed_media_delta(old,old) is None
    receipt=m._reviewed_media_delta(old,new)
    assert receipt["changed_file_hashes"] == pins
    assert receipt["task_control_changed_by_media_revision"] is False
    bad=copy.deepcopy(new);bad["files"][next(iter(pins))]="c"*64
    with pytest.raises(ValueError):m._reviewed_media_delta(old,bad)
    partial=copy.deepcopy(new);partial["files"][next(iter(pins))]="a"*64
    with pytest.raises(ValueError):m._reviewed_media_delta(old,partial)
    monkeypatch.setattr(m,"MEDIA_REVIEW",{p:{**v,"after":None} for p,v in pins.items()})
    with pytest.raises(ValueError):m._reviewed_media_delta(old,new)

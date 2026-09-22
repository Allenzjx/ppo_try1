"""CPU/synthetic same389 feedback migration. Not physical-success evidence."""
from copy import deepcopy
import json
from pathlib import Path
import pytest

from wlr50_clean.ppo.semantic_capture_feedback_migration import (
    CAPTURE_CODE,MODULE,SCHEMA,FACTOR_KEY,FEEDBACK_REVISION,EXPERIMENT,
    capture_feedback_factor,capture_feedback_branch_counts,_revision,
)
from wlr50_clean.ppo.semantic_policy_distribution import policy_contract,CONFIG_NAMES
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_POLICY,P05_CAPTURE_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_training import semantic_runner_config
from wlr50_clean.ppo.semantic_migration import continuation_topology


def boundary():
    canonical=policy_contract(P05_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)
    paths={f"configs/ppo_{EXPERIMENT}/{name}":"a"*64 for name in CONFIG_NAMES}
    selected={name:{"path":f"configs/ppo_{EXPERIMENT}/{name}","sha256":"a"*64} for name in CONFIG_NAMES}
    old={"experiment_id":EXPERIMENT,"semantic_version":"v3","selected_configuration":selected,
         "files":{**paths,CAPTURE_CODE:"b"*64},"source_git_commit":"a"*40,"runtime_content_sha256":"c"*64}
    new={**deepcopy(old),"files":{**old["files"],CAPTURE_CODE:"d"*64,MODULE:"e"*64},
         "source_git_commit":"f"*40,"runtime_content_sha256":"f"*64}
    sampling="P01_full_task_only_initial_version"
    meta={"semantic_version":"v3","runtime_contract":old,"policy_contract":canonical,"seed":1001,
          "runner_config":semantic_runner_config(seed=1001,device="cpu",semantic_version="v3",
              policy_version=P05_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT),
          "sampling":sampling,
          "execution_topology":continuation_topology(sampling,None,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT),
          "global_policy_decisions":203776,"ppo_updates":1557,
          "optimizer_steps":31140,"optimizer_learning_rate":1e-5,
          "stage_requested_decisions":{"full_episode":4096,"phase_suffix":2048,"smoke":0},
          "new_mdp_origin_global_policy_decisions":10112,
          "p05_capture_assist_branch":{"counter_origin":{"global_policy_decisions":199680,"ppo_updates":1525,"optimizer_steps":30500}},
          "task_conditioned_hip_wheel_branch":{"auxiliary_mean_learning":{"accepted":7,"attempted":8}}}
    review={CAPTURE_CODE:"d"*64,MODULE:"e"*64}
    return meta,old,new,review


def test_exact_same389_contract_and_independent_counter_origins():
    meta,old,new,review=boundary()
    factor=capture_feedback_factor(meta,old,new,reason="reviewed HOLD-to-AIR window fix",reviewed_code_sha256=review)
    assert factor["observation_contract"]["source_policy_contract"]==meta["policy_contract"]
    assert factor["observation_contract"]["source_policy_contract"]==factor["observation_contract"]["target_policy_contract"]
    assert factor["target_effective_learning_rate"]==1e-5 and factor["discard_old_rollout_storage"]
    state={**meta,"capture_feedback_semantics_branch":{"counter_origin":factor["counter_origin"]}}
    counted=capture_feedback_branch_counts(state)
    assert counted["p05_capture_assist_branch_counts"]["ppo_updates"]==32
    assert counted["capture_feedback_semantics_branch_counts"]["ppo_updates"]==0
    continued=capture_feedback_branch_counts({**state,"global_policy_decisions":203904,"ppo_updates":1558,"optimizer_steps":31160})
    assert continued["capture_feedback_semantics_branch_counts"]=={"global_policy_decisions":128,"ppo_updates":1,"optimizer_steps":20}
    assert state["p05_capture_assist_branch"]==meta["p05_capture_assist_branch"]


@pytest.mark.parametrize("corruption",["config","protected_code","missing_review","policy","repeat","missing_aux"])
def test_migration_rejects_unreviewed_or_nonidentity_boundary(corruption):
    meta,old,new,review=boundary()
    if corruption=="config":new["selected_configuration"]["reward_config.yaml"]["sha256"]="9"*64
    elif corruption=="protected_code":new["files"]["src/wlr50_clean/ppo/semantic_reward.py"]="9"*64
    elif corruption=="missing_review":review.pop(CAPTURE_CODE)
    elif corruption=="policy":meta["policy_contract"]["exploration_std_temperature"]=1.0
    elif corruption=="repeat":meta["capture_feedback_semantics_branch"]={}
    else:meta.pop("task_conditioned_hip_wheel_branch")
    with pytest.raises(ValueError):capture_feedback_factor(meta,old,new,reason="bounded fix",reviewed_code_sha256=review)


def test_feedback_revision_is_nonnumeric_and_not_a_new_observation_feature():
    assert _revision(b'CAPTURE_ASSIST_MODE = "p05_hip_only_continuation_v1"') is None
    assert _revision(f'CAPTURE_ASSIST_FEEDBACK_REVISION = "{FEEDBACK_REVISION}"')==FEEDBACK_REVISION
    from wlr50_clean.ppo.semantic_capture_assist import HipOnlyCaptureAssist,capture_assist_features
    snapshot=HipOnlyCaptureAssist().snapshot();before=capture_assist_features(snapshot)
    snapshot["feedback_revision"]=FEEDBACK_REVISION
    assert capture_assist_features(snapshot)==before and len(before)==12


def test_actual_same389_state_publish_and_reload_preserves_Adam_rng_and_lineage(tmp_path,monkeypatch):
    torch=pytest.importorskip("torch");pytest.importorskip("rsl_rl")
    from wlr50_clean.ppo import semantic_capture_feedback_migration as migration
    from wlr50_clean.ppo.semantic_training import construct_semantic_runner,save_semantic_checkpoint,load_semantic_checkpoint
    from wlr50_clean.ppo.semantic_p05_capture_migration import _ObservationOnlyEnv
    from wlr50_clean.ppo.semantic_migration import checkpoint_metadata
    def make():return construct_semantic_runner(_ObservationOnlyEnv(389),seed=1001,device="cpu",
        policy_version=P05_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner=make()
    for parameter in (*runner.alg.actor.parameters(),*runner.alg.critic.parameters()):parameter.grad=torch.randn_like(parameter)*.01
    runner.alg.optimizer.step();runner.alg.optimizer.zero_grad()
    runner.alg.optimizer.param_groups[0]["lr"]=1e-5;runner.alg.learning_rate=1e-5
    meta,old,new,review=boundary();meta.update(seed=1001,sampling="P01_full_task_only_initial_version")
    source,_=save_semantic_checkpoint(runner,tmp_path/"source.pt",meta)
    source_meta=checkpoint_metadata(source)
    factor=capture_feedback_factor(source_meta,old,new,reason="synthetic identity boundary",reviewed_code_sha256=review)
    record={"schema":SCHEMA,"plan_path":str(tmp_path/"mock_plan.json"),"source_checkpoint_sha256":source_meta["checkpoint_sha256"],
        "observation_dimension":389,"action_dimension":12,FACTOR_KEY:factor}
    monkeypatch.setattr(migration,"validate_capture_feedback_migration",lambda *a,**k:deepcopy(record))
    from wlr50_clean.ppo import semantic_migration as shared_migration
    monkeypatch.setattr(shared_migration,"validate_migration_plan",lambda *a,**k:deepcopy(record))
    receipt=migration.publish_capture_feedback_checkpoint(source,new,tmp_path/"mock_plan.json",tmp_path/"target.pt")
    target=checkpoint_metadata(Path(receipt["checkpoint"]))
    for key in ("actor_parameter_sha256","critic_parameter_sha256","optimizer_state_sha256","normalizer_state_sha256","training_rng_state","runner_config"):
        assert target[key]==source_meta[key]
    assert target["capture_feedback_semantics_branch_counts"]["ppo_updates"]==0
    # Ordinary later save must keep two independent origins, not overwrite P05.
    fresh=make();loaded=load_semantic_checkpoint(fresh,Path(receipt["checkpoint"]),contract=new,seed=1001)
    loaded.update(global_policy_decisions=203904,ppo_updates=1558,optimizer_steps=31160)
    later,_=save_semantic_checkpoint(fresh,tmp_path/"later_fixture.pt",loaded)
    later_meta=checkpoint_metadata(later)
    assert later_meta["capture_feedback_semantics_migration"]==record
    assert later_meta["capture_feedback_semantics_branch_counts"]["ppo_updates"]==1
    assert later_meta["p05_capture_assist_branch_counts"]["ppo_updates"]==33

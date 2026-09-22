"""CPU/synthetic append17 tests. None of these are physical task success."""
from copy import deepcopy
from pathlib import Path
import json

import pytest
torch=pytest.importorskip("torch")
TensorDict=pytest.importorskip("tensordict").TensorDict
pytest.importorskip("rsl_rl")

from wlr50_clean.ppo.semantic_p05_capture_profile import (
    P05_CAPTURE_POLICY,P05_CAPTURE_OBSERVATION_LAYOUT,P05_CAPTURE_OBSERVATION_DIM,
)
from wlr50_clean.ppo.semantic_p05_capture_actor import SemanticP05CaptureHistoryMLPModel,p05_capture_request_history
from wlr50_clean.ppo.semantic_p05_capture_migration import zero_append_training_state,_ObservationOnlyEnv
from wlr50_clean.ppo.semantic_receiving_wheel_profile import RECEIVING_WHEEL_POLICY
from wlr50_clean.ppo.semantic_receiving_wheel_sigma import SemanticReceivingWheelSigmaHistoryMLPModel
from wlr50_clean.ppo.semantic_training import (construct_semantic_runner,parameter_hash,state_hash,
    audited_history_policy_request,save_semantic_checkpoint,load_semantic_checkpoint)
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_observation import (load_semantic_observation_schema,
    SemanticObservationBuilder,HISTORY_GROUPS,semantic_task,SemanticObservationError)
from wlr50_clean.ppo.semantic_capture_assist import HipOnlyCaptureAssist

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def cpu_threads_and_rng():
    old=torch.get_rng_state();threads=torch.get_num_threads()
    torch.manual_seed(617);torch.set_num_threads(1)
    yield
    torch.set_rng_state(old);torch.set_num_threads(threads)


def make(dimension):
    return construct_semantic_runner(_ObservationOnlyEnv(dimension),seed=1001,device="cpu",
        policy_version=P05_CAPTURE_POLICY if dimension==389 else RECEIVING_WHEEL_POLICY,
        observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT if dimension==389 else ROLE_OBSERVATION_LAYOUT,
        initialize_actor=False)[0]


def source_bundle():
    runner=make(372)
    for parameter in [*runner.alg.actor.parameters(),*runner.alg.critic.parameters()]:
        parameter.grad=torch.randn_like(parameter)*.01
    runner.alg.optimizer.step();runner.alg.optimizer.zero_grad()
    runner.alg.learning_rate=2.25e-5;runner.alg.optimizer.param_groups[0]["lr"]=2.25e-5
    return runner,{"actor_state_dict":runner.alg.actor.state_dict(),"critic_state_dict":runner.alg.critic.state_dict(),
        "optimizer_state_dict":runner.alg.optimizer.state_dict(),"iter":7,"infos":{"fixture_only":True}}


def test_zero_extend_preserves_all_parameters_Adam_steps_LR_and_source_bytes():
    source,bundle=source_bundle();before=state_hash(bundle)
    mapped=zero_append_training_state(bundle)
    assert state_hash(bundle)==before
    target=make(389)
    for role in ("actor","critic"):
        getattr(target.alg,role).load_state_dict(mapped[role+"_state_dict"],strict=True)
        for key,old in bundle[role+"_state_dict"].items():
            new=mapped[role+"_state_dict"][key]
            if key=="mlp.0.weight":
                assert torch.equal(new[:,:372],old)
                assert torch.count_nonzero(new[:,372:])==0
            else: assert torch.equal(new,old)
    target.alg.optimizer.load_state_dict(mapped["optimizer_state_dict"])
    assert mapped["optimizer_state_dict"]["param_groups"]==bundle["optimizer_state_dict"]["param_groups"]
    for ident,old in bundle["optimizer_state_dict"]["state"].items():
        for key,value in old.items():
            new=mapped["optimizer_state_dict"]["state"][ident][key]
            if ident in (0,6) and key!="step":
                assert torch.equal(value,new[:,:372]) and torch.count_nonzero(new[:,372:])==0
            else: assert torch.equal(value,new)
    assert target.alg.optimizer.param_groups[0]["lr"]==2.25e-5


@pytest.mark.parametrize("phase",range(13))
def test_padded_distribution_continuity_and_true_raw_likelihood(phase):
    source,bundle=source_bundle();target=make(389)
    target.alg.actor.load_state_dict(zero_append_training_state(bundle)["actor_state_dict"])
    x=torch.zeros(1,372);x[:,phase]=1;x[:,20]=.01;x[:,158:158+phase]=1
    x[:,195:207]=torch.linspace(-.2,.2,12);x[:,157]=float(phase>9)
    extra=torch.linspace(-.4,.4,17).reshape(1,17)
    # New scheduling flags are booleans, not arbitrary actor test noise.
    extra[:,12:16]=0
    old=TensorDict({"policy":x,"critic":x.clone()},batch_size=[1])
    newx=torch.cat((x,extra),dim=-1)
    new=TensorDict({"policy":newx,"critic":newx.clone()},batch_size=[1])
    with torch.no_grad():
        expected=source.alg.actor(old)
        actual,audit=audited_history_policy_request(target.alg.actor,new,lambda:target.alg.actor(new),stochastic=False)
        torch.testing.assert_close(actual,expected,rtol=2e-6,atol=2e-7)
        source.alg.actor(old,stochastic_output=True)
        raw,audit=audited_history_policy_request(target.alg.actor,new,
            lambda:target.alg.actor(new,stochastic_output=True),stochastic=True)
        torch.testing.assert_close(target.alg.actor.output_std,source.alg.actor.output_std,rtol=2e-6,atol=2e-7)
        expected_lp=torch.distributions.Normal(target.alg.actor.output_mean,target.alg.actor.output_std).log_prob(raw).sum(-1)
        assert torch.equal(target.alg.actor.get_output_log_prob(raw),expected_lp)
        assert audit["policy_version"]==P05_CAPTURE_POLICY and audit["sampling_draws"]==1
        assert audit["selected_raw_log_probability"]==expected_lp.item()
        assert audit["transformed_actuator_targets_are_not_policy_samples"]


def test_pending_handoff_preserves_previous_REQUEST_without_faking_completed_state():
    x=torch.zeros(1,389);x[:,5]=1;x[:,158:162]=1
    x[:,384]=1;x[:,386]=1;x[:,207:219]=.05
    before=x.clone();center,evidence=p05_capture_request_history(x)
    assert torch.equal(x,before) and x[0,162]==0
    assert evidence["pending_scheduler_handoff"].item()
    assert not evidence["physical_predecessor_completed"].item()
    gate=evidence["gate_full12"]
    expected=evidence["previous_filtered_request_full12"]/evidence["current_cap_full12"]
    torch.testing.assert_close(center.tanh()[gate],expected[gate])


def test_new_schema_exposes_assist_and_pending_but_retains_actual_completion_subset():
    from test_semantic_observation_reward_env import _frame,ZERO12
    schema=load_semantic_observation_schema(ROOT/"configs/ppo_p05_hip_only_continuation_v1/observation_schema.json")
    old=load_semantic_observation_schema(ROOT/"configs/ppo_task_conditioned_hip_wheel_v1/observation_schema.json")
    assert schema.observation_layout==P05_CAPTURE_OBSERVATION_LAYOUT and schema.dimension==389
    assert schema.groups[:-2]==old.groups
    frame=_frame(stage="P07")
    task=frame.info["semantic_task"]
    task["completed_stage_ids"].remove("P05")
    task.update(capture_continuation={"mode":"p05_hip_only_continuation_v1","scheduler_advanced_pending":True},
        fl_capture_pending=True,allow_capture_continuation=True,p05_local_deadline_warning=True,capture_pending_elapsed_s=8.)
    state=HipOnlyCaptureAssist()
    frame.info["capture_assist"]=state.snapshot()
    groups=SemanticObservationBuilder(schema).build(frame,dict.fromkeys(HISTORY_GROUPS,ZERO12)).groups
    values=schema.encode(groups)
    assert values[162]==0 and values[154]==0
    assert values[372:384]==(0.,)*12
    assert values[384:]==(1.,1.,1.,1.,.04)
    task.pop("capture_continuation")
    with pytest.raises(SemanticObservationError): semantic_task(frame)


def test_loader_save_reload_mapping_with_preserved_source_state(tmp_path,monkeypatch):
    from wlr50_clean.ppo import semantic_p05_capture_migration as migration
    from wlr50_clean.ppo.semantic_migration import continuation_topology
    from wlr50_clean.ppo.semantic_policy_distribution import policy_contract
    source,_=source_bundle()
    source_infos={"seed":1001,"runtime_contract":{"fixture":"source"},"global_policy_decisions":128,
        "ppo_updates":1,"optimizer_steps":20,"sampling":"P01_full_task_only_initial_version",
        "execution_topology":continuation_topology("P01_full_task_only_initial_version",None,observation_layout=ROLE_OBSERVATION_LAYOUT),
        "stage_requested_decisions":{"full_episode":128,"phase_suffix":0,"smoke":0},
        "auxiliary_fixture":{"accepted":7,"attempted":8}}
    cp,_=save_semantic_checkpoint(source,tmp_path/"source.pt",source_infos)
    record={"plan_path":str(tmp_path/"not_file_mock_plan.json"),"source_checkpoint_sha256":"fixture",
        "p05_capture_assist_factor":{"target_policy_contract":policy_contract(P05_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT),
            "counter_origin":{"global_policy_decisions":128,"ppo_updates":1,"optimizer_steps":20}}}
    monkeypatch.setattr(migration,"validate_p05_capture_migration",lambda *a,**k:record)
    target=make(389)
    infos=load_semantic_checkpoint(target,cp,contract={"fixture":"target"},seed=1001,migration=record)
    assert infos["auxiliary_fixture"]==source_infos["auxiliary_fixture"]
    assert infos["global_policy_decisions"]==128 and infos["ppo_updates"]==1
    assert target.alg.optimizer.param_groups[0]["lr"]==2.25e-5
    assert target.alg.storage.step==0 and target.alg.transition.actions is None
    out,_=save_semantic_checkpoint(target,tmp_path/"target.pt",infos)
    fresh=make(389)
    final=load_semantic_checkpoint(fresh,out,contract={"fixture":"target"},seed=1001)
    assert parameter_hash(fresh.alg.actor)==parameter_hash(target.alg.actor)
    assert state_hash(fresh.alg.optimizer.state_dict())==state_hash(target.alg.optimizer.state_dict())
    assert final["optimizer_steps"]==20

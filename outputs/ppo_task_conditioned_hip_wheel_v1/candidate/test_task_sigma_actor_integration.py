"""Output-only candidate overlay. CPU/synthetic evidence, zero real credit.

The shared bootstrap overlays the complete reviewed candidate dependency graph.
No global env edits, production file writes, or physical execution occur.
"""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from candidate_bootstrap import CANDIDATE
from wlr50_clean.ppo import semantic_history_actor as a, semantic_policy_distribution as p, semantic_training as t
assert a is CANDIDATE['semantic_history_actor'] and t is CANDIDATE['semantic_training']
NEW=p.TASK_CONDITIONED_HIP_WHEEL_POLICY
OLD=p.FR_KNEE_PHYSICAL_INNOVATION_POLICY
LAYOUT='diagonal_transfer_state_v1'


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    state,threads=torch.get_rng_state(),torch.get_num_threads()
    torch.manual_seed(219);torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    yield
    torch.set_rng_state(state);torch.set_num_threads(threads)


def values(phase):
    x=torch.zeros(1,372);x[0,phase]=1;x[0,158:158+phase]=1;x[0,20]=.01
    x[0,195:207]=torch.linspace(-.2,.2,12)
    if phase<=1:x[0,147]=1;x[0,24]=.015;x[0,33]=3
    if phase==4:x[0,150]=1;x[0,21]=.026
    if phase==5:
        x[0,154:156]=1;x[0,132]=x[0,134]=1;x[0,124]=x[0,126]=.1
        x[0,22]=x[0,25]=.1;x[0,100]=x[0,103]=-.1;x[0,28]=x[0,31]=-.3
    if phase==8:x[0,149]=1
    return x


def observation(phase):
    x=values(phase)
    return TensorDict({'policy':x,'critic':x.clone()},batch_size=[1])


def actor(cls,obs):
    return cls(obs,{'actor':['policy']},'actor',12,observation_layout=LAYOUT,
        exploration_std_temperature=.25,
        distribution_cfg={'class_name':'HeteroscedasticGaussianDistribution','std_type':'log','init_std':.15})


@pytest.mark.parametrize('phase',range(13))
def test_actual_actor_one_head_one_Gaussian_full12_shared_audit_and_deterministic_exact(phase):
    obs=observation(phase)
    old=actor(a.SemanticFRKneePhysicalInnovationHistoryMLPModel,obs)
    new=actor(a.SemanticTaskConditionedHipWheelHistoryMLPModel,obs)
    new.load_state_dict(old.state_dict(),strict=True)
    state=torch.get_rng_state(); calls=[]
    hook=new.mlp.register_forward_hook(lambda *_:calls.append(1))
    raw,audit=t.audited_history_policy_request(new,obs,lambda:new(obs,stochastic_output=True),stochastic=True)
    hook.remove();after=torch.get_rng_state()
    assert len(calls)==1 and audit['sampling_draws']==1 and audit['extra_model_forwards']==0
    torch.set_rng_state(state)
    normal=torch.distributions.Normal(new.output_mean,new.output_std)
    assert torch.equal(raw,normal.sample()) and torch.equal(after,torch.get_rng_state())
    assert torch.equal(new.get_output_log_prob(raw),normal.log_prob(raw).sum(-1))
    assert torch.equal(new.output_entropy,normal.entropy().sum(-1))
    assert bool((new.output_std>0).all()) and audit['policy_version']==NEW
    assert torch.equal(torch.tensor(audit['effective_sigma_full12']),new.output_std[0])
    assert audit['rho']==.9 and len(audit['physical_equivalent_B_full12'])==12
    cache=[v.clone() for v in new.output_distribution_params];state=torch.get_rng_state()
    fixed,diag=t.audited_history_policy_request(new,obs,lambda:new(obs),stochastic=False)
    assert torch.equal(fixed,old(obs)) and torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(x,y) for x,y in zip(cache,new.output_distribution_params))
    assert diag['sampling_draws']==0 and diag['task_state_weights']==audit['task_state_weights']
    assert set(dict(new.named_parameters()))==set(dict(old.named_parameters()))
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())


def test_supported_complete_policy_contract_and_config_tamper_rejected():
    c=p.policy_contract(NEW,observation_layout=LAYOUT)
    cfg=t.semantic_runner_config(seed=1001,device='cpu',semantic_version='v3',policy_version=NEW,observation_layout=LAYOUT)
    metadata={'runner_config':cfg,'seed':1001,'semantic_version':'v3','policy_contract':c,'policy_version':NEW}
    assert p.policy_version_from_metadata(metadata)==NEW
    assert p.supported_heteroscedastic_contract_version(c)==NEW
    assert c['rho']==.9 and c['directional_bias'] is False
    for key in ('rho','physical_equivalent_B_full12','sigma_scaling_semantics'):
        bad=copy.deepcopy(metadata);bad['policy_contract'][key]='bad'
        with pytest.raises(ValueError):p.policy_version_from_metadata(bad)


@pytest.mark.parametrize('near_leg',[None,28,31])
def test_actual_P06_AIR_recovery_and_either_rear_near_exits_rolling_preference(near_leg):
    obs=observation(5);x=obs['policy'];x[0,132]=0
    if near_leg is not None:x[0,near_leg]=-.22
    model=actor(a.SemanticTaskConditionedHipWheelHistoryMLPModel,obs)
    _,audit=t.audited_history_policy_request(model,obs,lambda:model(obs,stochastic_output=True),stochastic=True)
    label='P06_front_support_recovery_proxy' if near_leg is None else 'RR_preparation'
    assert audit['physical_equivalent_B_full12']==torch.tensor(p.TASK_CONDITIONED_PHYSICAL_B_TABLE[label]).tolist()
    assert audit['task_state_weights']['P06_front_support_recovery_proxy']==1
    assert audit['task_state_weights']['P06_front_support_rolling_proxy']==0
    assert audit['task_state_weights']['RR_preparation']==(0 if near_leg is None else 1)
    assert audit['task_state_audit_topology']=='one_actual_N1_request_not_a_batched_N8_summary'
    assert all(v>0 for v in audit['effective_sigma_full12'])
    assert audit['current_cap_full12']==torch.tensor(p.REQUEST_HISTORY_CAPS[5]).tolist()


class CPUCore:
    def __init__(self):self.calls=0;self.frame=SimpleNamespace(sim_time_s=0.)
    def observation(self,raw=None):
        phase=(1,4,5,8)[min(self.tick//4,3)]
        x=values(phase)[0];x[20]=.01+self.tick/3000
        if raw is not None:x[195:207]=torch.tensor(raw).clamp(-19.,19.)
        return tuple(x.tolist())
    def reset(self,*,seed=1001,options=None):
        self.tick=0;self.frame.sim_time_s=0.;return self.observation()
    def step(self,raw):
        phase=(1,4,5,8)[min(self.tick//4,3)];self.tick+=1;self.calls+=1;self.frame.sim_time_s=self.tick/15
        end=self.tick==17
        return SimpleNamespace(observation=self.observation(raw),reward=1+raw[0]-.1*raw[1]**2,
            terminated=end,truncated=False,info={'phase_id':f'P{phase+1:02d}','raw_policy_action_full12':raw,
            'applied_action_full12':tuple(.1*v for v in raw),'task_success':False,
            'termination_reason':'CPU_SYNTHETIC_ONLY' if end else None,
            'actuator_target_effect_audit':{'schema':'wlr50_clean.actuator_target_effect_audit.v1',
                'verified':True,'actual_mapping_matches_dispatch':True,'setter_dispatch_targets_equal':True,
                'same_tick_counterfactual':True,'raw_policy_action_full12':raw,'target_dtype':'torch.float32',
                'changed_target_channel_count':12}})
    def telemetry_summary(self):return {'CPU_SYNTHETIC_NOT_PHYSICS':True,'calls':self.calls}


def runner(version=NEW):
    env=t.SemanticRslAdapter(CPUCore(),seed=1001,device='cpu');env.cfg['semantic_version']='v3'
    model,_=t.construct_semantic_runner(env,seed=1001,device='cpu',initialize_actor=False,
        policy_version=version,observation_layout=LAYOUT)
    return model,env


def test_official_PPO_saved_actual_sigma_likelihood_update_save_load_and_continue(tmp_path):
    t.seed_training_rngs(1001)
    model,env=runner()
    source=ROOT/'outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000185856_manifest.json'
    contract=copy.deepcopy(json.loads(source.read_text())['runtime_contract'])
    contract['experiment_id']='task_conditioned_hip_wheel_v1'
    contract['CPU_SYNTHETIC_ONLY_NOT_ADOPTABLE_RUNTIME']=True
    result=t.train_semantic(model,env,run_dir=tmp_path/'run1',output_root=tmp_path/'out1',stage='full_episode',
        decisions=128,contract=contract,seed=1001)
    assert result['ppo_updates_this_run']==1 and result['optimizer_steps_this_run']==20
    rows=[json.loads(line) for line in (tmp_path/'run1/residual_and_projection_audit.jsonl').read_text().splitlines()]
    storage=torch.load(tmp_path/'run1/rollouts/rollout_000001.pt',map_location='cpu',weights_only=False)
    assert len(rows)==128
    for i,row in enumerate(rows):
        request=row['policy_request']
        assert request['policy_version']==NEW
        assert torch.equal(torch.tensor(request['selected_raw_full12']),storage['actions'][i,0])
        assert torch.equal(torch.tensor(request['effective_sigma_full12']),storage['distribution_params'][1][i,0])
        assert request['selected_raw_log_probability']==storage['actions_log_prob'][i,0,0].item()
    hooks=json.loads((tmp_path/'run1/rollouts/update_000001_likelihood.json').read_text())
    # Read the real official update hooks, not a second optimizer replay.
    assert hooks['minibatches'][0]['sigma_source']=='current_official_Gaussian_cache_after_current_observation_B_over_cap'
    assert max(abs(v-1) for v in hooks['minibatches'][0]['ratio'])<1e-5
    model.alg.learning_rate=2.3e-5
    for group in model.alg.optimizer.param_groups:group.update(lr=2.3e-5,betas=(.87,.996),eps=2e-8)
    metadata=json.loads(Path(result['checkpoints'][-1]['manifest']).read_text())
    for key in ('checkpoint_path','checkpoint_sha256','save_load_round_trip'):metadata.pop(key)
    cp=tmp_path/'same_version_nondefault_Adam.pt';t.save_semantic_checkpoint(model,cp,metadata)
    fresh,newenv=runner();loaded=t.load_semantic_checkpoint(fresh,cp,contract=contract,seed=1001)
    assert newenv.core.calls==0 and fresh.alg.storage.step==0 and fresh.alg.transition.actions is None
    assert fresh.alg.learning_rate==t.optimizer_learning_rate(fresh)==2.3e-5
    assert t.parameter_hash(fresh.alg.actor)==t.parameter_hash(model.alg.actor)
    assert t.state_hash(fresh.alg.optimizer.state_dict())==t.state_hash(model.alg.optimizer.state_dict())
    assert t.state_hash(t._normalizers(fresh))==t.state_hash(t._normalizers(model))
    assert t.capture_training_rng_state(seed=1001)==loaded['training_rng_state']
    continuation=t.train_semantic(fresh,newenv,run_dir=tmp_path/'run2',output_root=tmp_path/'out2',stage='full_episode',
        decisions=128,contract=contract,seed=1001,resume_infos=loaded)
    assert (continuation['global_policy_decisions'],continuation['ppo_updates_this_run'])==(256,1)


def test_actual_minibatch_head_hooks_are_read_only_and_align_with_official_update(tmp_path):
    """Same real RSL minibatches, CPU synthetic data only; no physics credit."""
    t.seed_training_rngs(1001)
    model, env = runner()

    def collect():
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(128):
                raw = model.alg.act(obs)
                obs, rewards, dones, extras = env.step(raw)
                model.alg.process_env_step(obs, rewards, dones, extras)
            model.alg.compute_returns(obs)

    collect()
    t.audited_ppo_update(model)  # Give both branches an actual nonempty Adam state.
    collect()
    alg = model.alg
    source = {role: copy.deepcopy(getattr(alg, role).state_dict())
              for role in ('actor', 'critic', 'optimizer')}
    storage, lr = copy.deepcopy(alg.storage), alg.learning_rate
    obs_rows = storage.observations['policy'].flatten(0, 1).clone()
    before_rng = t.capture_training_rng_state(seed=1001)
    no_audit_forwards = []
    counter = alg.actor.mlp.register_forward_hook(lambda *_: no_audit_forwards.append(1))
    try:
        disabled = t.audited_ppo_update(model)
    finally:
        counter.remove()
    disabled_hashes = {role: t.state_hash(getattr(alg, role).state_dict())
                       for role in ('actor', 'critic', 'optimizer')}
    disabled_rng = t.capture_training_rng_state(seed=1001)
    disabled_lr = alg.learning_rate
    for role in ('actor', 'critic', 'optimizer'):
        getattr(alg, role).load_state_dict(source[role])
    alg.storage, alg.learning_rate = copy.deepcopy(storage), lr
    t.restore_training_rng_state(before_rng, expected_seed=1001)

    observed, derivatives, parameter_norms = [], [], []
    def inspect_head(_module, _inputs, output):
        observed.append(output.detach().clone())
        def inspect_derivative(gradient):
            derivatives.append(gradient.detach().clone())
            # Deliberately implicit None, as in the candidate's read-only hook.
        output.register_hook(inspect_derivative)
    def inspect_step(_optimizer, _args, _kwargs):
        layer = [v for v in alg.actor.mlp.modules() if isinstance(v, torch.nn.Linear)][-1]
        rows = (layer.weight.grad.detach().double().square().sum(-1)
                + layer.bias.grad.detach().double().square()).sqrt()
        separate = {role: sum(float(v.grad.detach().double().square().sum())
                    for v in getattr(alg, role).parameters() if v.grad is not None) ** .5
                    for role in ('actor', 'critic')}
        parameter_norms.append((rows, separate))
    head_handle = alg.actor.mlp.register_forward_hook(inspect_head)
    step_handle = alg.optimizer.register_step_pre_hook(inspect_step)
    audit_path = tmp_path/'actual_minibatch_head.json'
    try:
        enabled = t.audited_ppo_update(model, likelihood_audit_path=audit_path)
    finally:
        head_handle.remove()
        step_handle.remove()
    assert enabled == disabled
    assert {role: t.state_hash(getattr(alg, role).state_dict())
            for role in ('actor', 'critic', 'optimizer')} == disabled_hashes
    assert alg.learning_rate == disabled_lr
    assert t.capture_training_rng_state(seed=1001) == disabled_rng
    assert len(no_audit_forwards) == len(observed) == len(derivatives) == len(parameter_norms) == 20
    assert not alg.actor.mlp._forward_hooks and not alg.optimizer._optimizer_step_pre_hooks
    evidence = json.loads(audit_path.read_text())
    assert evidence['extra_model_forwards'] == evidence['extra_random_draws'] == 0
    assert len(evidence['minibatches']) == 20
    for i, row in enumerate(evidence['minibatches']):
        assert row['minibatch_index'] == i
        assert torch.equal(torch.tensor(row['current_network_mean_full12']), observed[i][..., 0, :])
        assert torch.equal(torch.tensor(row['current_network_log_sigma_full12']), observed[i][..., 1, :])
        assert torch.equal(torch.tensor(row['loss_gradient_wrt_network_mean_full12']), derivatives[i][..., 0, :])
        assert torch.equal(torch.tensor(row['loss_gradient_wrt_network_log_sigma_full12']), derivatives[i][..., 1, :])
        assert torch.equal(torch.tensor(row['actor_head_postclip_parameter_gradient_norm_by_row'],
                                      dtype=torch.float64), parameter_norms[i][0])
        assert row['separate_postclip_parameter_gradient_norms'] == parameter_norms[i][1]
        saved_obs = obs_rows[[indices[0] for indices in row['rollout_flat_indices']]]
        history, _ = a.cap_transition_request_history(saved_obs)
        conditioned = a.history_conditioned_head(observed[i], history, a.HISTORY_RHO)
        log_sigma, _ = a.task_conditioned_effective_log_std(
            observed[i][..., 1, :], saved_obs, alg.actor.exploration_std_temperature)
        assert torch.equal(torch.tensor(row['current_conditional_mean']), conditioned[..., 0, :])
        assert torch.equal(torch.tensor(row['current_conditional_sigma']), log_sigma.exp())

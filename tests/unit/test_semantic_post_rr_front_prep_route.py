"""Cold CPU only; actual sealed sources, temporary checkpoint, no Isaac credit."""
import copy
import json
from pathlib import Path

import pytest
import torch

from wlr50_clean.ppo import semantic_post_rr_front_prep as route
from wlr50_clean.ppo.rl_library_wrapper import (
    capture_training_rng_state, restore_training_rng_state, seed_training_rngs,
)
from wlr50_clean.ppo.semantic_training import audited_ppo_update, state_hash


@pytest.fixture(autouse=True)
def preserve_cpu_rng():
    before = capture_training_rng_state(seed=1001)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    seed_training_rngs(1001)
    yield
    restore_training_rng_state(before,expected_seed=1001)
    torch.set_num_threads(threads)


def observation(active=False,index=0):
    dataset = route.ROOT/'outputs/ppo_rl_recovery_learning_v1/staged_cp225280_front_branch/CP225280_front_replay_dataset.json'
    values = json.loads(dataset.read_text())['training_rows'][0]['observation'] + [0.]*48
    assert len(values) == 487
    values[439] = 1.  # The accepted learned capture branch, not only the225280 prior.
    values[447] = 1.
    values[465] = float(active)
    values[466] = float(active)
    values[470] = index/10000.
    return route.tensor_observation(values,'cpu')


def runtime():
    return dict(experiment_id=route.NAME,local_contract=route.settings(),
                source_git_commit='0'*40,test_only='synthetic_not_physical_not_formal_PPO')


def initialized():
    runner = route.make_runner('cpu',1001)
    lineage,counts = route.initialize_from_accepted(runner)
    return runner,lineage,counts


def test_actual_accepted_composite_critic_migration_optimizer_and_prefix():
    runner,lineage,counts = initialized()
    actor = runner.alg.actor
    anchor = actor.frozen_anchor
    assert lineage['accepted_anchor']['counts']['local_policy_decisions'] == 6656
    assert lineage['latest_preserved_critic_source']['counts']['local_policy_decisions'] == 7168
    assert lineage['old512_is_new_RL_T1'] is False
    assert lineage['old_Adam_loaded_into_new_optimizer'] is False
    assert lineage['old_RR2_AUX_package_used'] is False
    assert counts['post_rr_policy_decisions'] == counts['post_rr_ppo_updates'] == counts['auxiliary_updates'] == 0
    assert runner.alg.learning_rate == 1.e-5
    assert not runner.alg.optimizer.state
    ids = {id(p) for g in runner.alg.optimizer.param_groups for p in g['params']}
    assert ids == {id(p) for p in actor.trainable_parameters()} | {id(p) for p in runner.alg.critic.parameters()}
    assert not ids & {id(p) for p in anchor.parameters()}
    source = torch.load(route.source_metadata('critic')[0],map_location='cpu',weights_only=False)
    for key,value in runner.alg.critic.state_dict().items():
        expected = source['critic_state_dict'][key]
        if key == lineage['critic_expanded_input']:
            assert torch.equal(value[:,:465],expected)
            assert torch.count_nonzero(value[:,465:]) == 0
        else:
            assert torch.equal(value,expected)
    runner.alg.train_mode()
    assert not anchor.training
    assert isinstance(actor.obs_normalizer,torch.nn.Identity)
    assert isinstance(anchor.obs_normalizer,torch.nn.Identity)
    assert isinstance(runner.alg.critic.obs_normalizer,torch.nn.Identity)
    obs = observation(False)
    before = capture_training_rng_state(seed=1001)
    with torch.inference_mode():
        raw,audit = route.request(runner,obs,stochastic=False)
        expected = anchor({'policy':obs['policy'][...,:465]},stochastic_output=False)
    assert torch.equal(raw,expected)
    assert capture_training_rng_state(seed=1001) == before
    assert audit['prefix_excluded_from_new_PPO_credit'] and audit['sampling_draws'] == 0
    assert runner.alg.storage.step == 0 and runner.alg.transition.actions is None
    with torch.inference_mode(),pytest.raises(ValueError,match='no post-RR noise'):
        route.request(runner,obs,stochastic=True)
    actor.assert_frozen_state(runner.alg.optimizer)


@pytest.mark.parametrize('terminal_tail',[False,True])
def test_actual_512_official_update_whole_anchor_freeze_and_temporary_reload(tmp_path,monkeypatch,terminal_tail):
    runner,lineage,counts = initialized()
    runner.alg.train_mode()
    frozen = state_hash(runner.alg.actor.frozen_anchor.state_dict())
    original_head = state_hash(runner.alg.actor.mlp.state_dict())
    for index in range(512):
        obs,nxt = observation(True,index),observation(True,index+1)
        with torch.inference_mode():
            action,audit = route.request(runner,obs,stochastic=True)
            logp = runner.alg.transition.actions_log_prob.clone()
            mean,sigma = runner.alg.actor.output_distribution_params
            expected = torch.distributions.Normal(mean,sigma).log_prob(action).sum(-1)
            assert torch.equal(logp,expected)
            assert audit['history_kernel_applications'] == 1
            assert audit['gain10_applied_again'] is False
            reward = torch.tensor([-.02+.003*(index%7)])
            done = torch.tensor([terminal_tail and index==511])
            runner.alg.process_env_step(nxt,reward,done,{'time_outs':torch.zeros_like(done)})
        assert torch.equal(runner.alg.storage.actions[index],action)
        assert torch.equal(runner.alg.storage.actions_log_prob[index].view(-1),logp.view(-1))
    with torch.inference_mode():
        value = runner.alg.critic(nxt).flatten()[0]
        runner.alg.compute_returns(nxt)
    assert torch.allclose(runner.alg.storage.returns[-1,0,0],
                          reward[0]+(0. if terminal_tail else runner.alg.gamma*value),atol=2.e-6,rtol=0)
    assert (runner.alg.storage.observations['policy'][...,465] == 1.).all()
    report = audited_ppo_update(runner,likelihood_audit_path=tmp_path/'synthetic_likelihood.json')
    assert report['optimizer_steps'] == 20 and report['actor_parameters_changed']
    assert state_hash(runner.alg.actor.frozen_anchor.state_dict()) == frozen
    assert state_hash(runner.alg.actor.mlp.state_dict()) != original_head
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)
    assert runner.alg.storage.step == 0
    counts.update(post_rr_policy_decisions=512,post_rr_ppo_updates=1,post_rr_optimizer_steps=20)
    monkeypatch.setattr(route,'OUTPUT',tmp_path/'artifacts')
    prior_state = state_hash(runner.alg.save())
    prior_rng = capture_training_rng_state(seed=1001)
    pointer = route.save(runner,runtime(),lineage,counts,source_run=tmp_path)
    assert Path(pointer['checkpoint']).is_relative_to(tmp_path)
    assert state_hash(runner.alg.save()) == prior_state
    assert capture_training_rng_state(seed=1001) == prior_rng
    expected_random = torch.rand(8)
    metadata = json.loads(Path(pointer['manifest']).read_text())
    resumed = route.make_runner('cpu',1001,saved_configuration=metadata['runner_config'])
    saved_lineage,saved_counts = route.load(resumed,pointer['checkpoint'],runtime())
    assert torch.equal(torch.rand(8),expected_random)
    assert saved_lineage == lineage and saved_counts == counts
    assert state_hash(resumed.alg.save()) == prior_state
    resumed.alg.actor.assert_frozen_state(resumed.alg.optimizer)
    with torch.inference_mode():
        assert torch.equal(resumed.alg.actor(nxt),runner.alg.actor(nxt))
    with pytest.raises(ValueError,match='contract/hash/ledger'):
        route.load(resumed,pointer['checkpoint'],dict(runtime(),experiment_id='old_route'))


def test_critic_expansion_does_not_silently_reset_existing_weights():
    source = {'layer':torch.arange(930.).reshape(2,465),'bias':torch.tensor([2.,3.])}
    target = {'layer':torch.zeros(2,487),'bias':torch.zeros(2)}
    expanded,key = route.expanded_critic_state(source,target)
    assert key == 'layer' and torch.equal(expanded[key][:,:465],source[key])
    assert torch.count_nonzero(expanded[key][:,465:]) == 0
    assert torch.equal(expanded['bias'],source['bias'])
    with pytest.raises(ValueError,match='unexpected critic shape'):
        route.expanded_critic_state(source,dict(target,layer=torch.zeros(3,487)))

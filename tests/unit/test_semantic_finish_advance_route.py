"""Cold CPU route integration; synthetic data never counts as physical PPO.

Uses the immutable04 CP232960 source. All generated updates/checkpoints are in
pytest temporary directories; the real outputs pointer is never modified.
"""
import copy
import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_semantic_finish_advance_actor import recorded490, source_data
from wlr50_clean.ppo import semantic_finish_advance as route
from wlr50_clean.ppo.semantic_finish_advance_task import FINISH_FIELDS
from wlr50_clean.ppo.semantic_training import state_hash
from wlr50_clean.ppo.rl_library_wrapper import capture_training_rng_state, restore_training_rng_state


def test_native_extension_import_order_matches_accepted_isaac_entrypoint():
    # Prevent the actual startup-only Windows C-extension failure from recurring.
    tree = ast.parse(inspect.getsource(route.main))
    imports = {alias.name: node.lineno for node in ast.walk(tree)
               if isinstance(node, ast.Import) for alias in node.names}
    app_import = next(node.lineno for node in ast.walk(tree)
                      if isinstance(node, ast.ImportFrom) and node.module == 'isaaclab.app')
    assert imports['torch'] < app_import and imports['tensordict'] < app_import


@pytest.fixture(autouse=True)
def preserve_rng():
    before = capture_training_rng_state(seed=1001)
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    restore_training_rng_state(before,expected_seed=1001)
    torch.set_num_threads(threads)


def runtime():
    return dict(experiment_id=route.NAME,local_contract=route.settings(),source_git_commit='0'*40,
                test_only='synthetic_CPU_not_physical_PPO')


def observation(index=0,active=True):
    data = recorded490()[index%32].tolist()+[0.]*41
    data[490] = float(active)
    for i in range(12):
        data[490+FINISH_FIELDS.index('finish_request_ref_ratio_'+str(i))] = -.35+.05*i if active else 0.
    data[490+FINISH_FIELDS.index('finish_advance_progress_norm')] = (index%128)/128
    return data


def initialized():
    runner = route.make_runner('cpu',1001)
    lineage,counts = route.initialize_from_accepted(runner)
    return runner,lineage,counts


def test_actual04_init_critic_Adam_inherited_frozen490_RNG_and_no_prefix_storage():
    runner,lineage,counts = initialized()
    saved = source_data(); actor = runner.alg.actor
    assert all(v==0 for v in counts.values())
    assert lineage['accepted_complete_sha256'] == route.settings()['accepted_checkpoint_sha256']
    assert lineage['critic_Adam_inherited'] and lineage['new_finish_actor_Adam_initialized']
    assert lineage['old_rollout_reused'] is False and lineage['old_gain10_applied_again'] is False
    assert runner.alg.learning_rate == saved['infos']['learning_rate'] == 1.e-5
    assert capture_training_rng_state(seed=1001) == saved['infos']['training_rng']
    assert state_hash(actor.frozen_accepted.state_dict()) == state_hash(saved['actor_state_dict'])
    names = list(runner.alg.critic.named_parameters())
    old_opt = saved['optimizer_state_dict']
    ids = [i for g in old_opt['param_groups'] for i in g['params']]
    for (name,param),old_id in zip(names,ids[-len(names):],strict=True):
        expected = saved['critic_state_dict'][name]
        actual = param.detach()
        if actual.shape != expected.shape:
            assert actual.shape[1] == 531 and expected.shape[1] == 490
            assert torch.equal(actual[:,:490],expected) and not torch.count_nonzero(actual[:,490:])
        else:
            assert torch.equal(actual,expected)
        state = runner.alg.optimizer.state[param]
        for key,value in old_opt['state'][old_id].items():
            got = state[key]
            if torch.is_tensor(value) and got.shape != value.shape:
                assert torch.equal(got[:,:490],value) and not torch.count_nonzero(got[:,490:])
            elif torch.is_tensor(value):
                assert torch.equal(got,value)
            else:
                assert got == value
    assert all(p not in runner.alg.optimizer.state for p in actor.trainable_parameters())
    assert all(not torch.is_inference(v) for state in runner.alg.optimizer.state.values()
               for v in state.values() if torch.is_tensor(v))
    assert not any(torch.is_inference(p) for p in runner.alg.critic.parameters())
    actor.assert_frozen_state(runner.alg.optimizer)
    before = capture_training_rng_state(seed=1001)
    obs = route.tensor_observation(observation(active=False),'cpu')
    with torch.inference_mode():
        raw,audit = route.request(runner,obs,stochastic=False)
        expected = actor.frozen_accepted({'policy':obs['policy'][...,:490]},stochastic_output=False)
    assert torch.equal(raw,expected) and capture_training_rng_state(seed=1001) == before
    assert audit['prefix_excluded_from_new_PPO_credit'] and audit['sampling_draws']==0
    assert runner.alg.storage.step == 0 and runner.alg.transition.actions is None
    assert runner.alg.gamma == .9985 and runner.alg.lam == .99
    assert runner.alg.num_learning_epochs == 5 and runner.alg.num_mini_batches == 4
    assert runner.alg.storage.num_transitions_per_env == 512


class SyntheticCore:
    """Finite unit-test values + explicit synthetic ACK, never a physical result."""
    def __init__(self,terminal_tail):
        self.index=0; self.done=False; self.terminal_tail=terminal_tail
        self.task=SimpleNamespace(finish_active=True,snapshot=self.snapshot)
        self.frame=SimpleNamespace(physics_tick=0,sim_time_s=76.,state_id='P13')
        self.observation=observation(0)
    def snapshot(self):
        mode='ADVANCE_FOR_HOME_CLEARANCE' if self.index<256 else 'HOME_RECOVERY' if self.index<384 else 'FINAL_SETTLE'
        return dict(requested_finish={'mode':mode},synthetic_only=True)
    def step(self,raw):
        self.index+=1; self.observation=observation(self.index)
        self.frame=SimpleNamespace(physics_tick=self.index*8,sim_time_s=76.+self.index/15,state_id='P13')
        self.done=bool(self.terminal_tail and self.index==512)
        native=dict(schema='wlr50_clean.actuator_target_effect_audit.v1',verified=True,
            actual_mapping_matches_dispatch=True,setter_dispatch_targets_equal=True,
            same_tick_counterfactual=True,raw_policy_action_full12=list(raw),target_dtype='torch.float32',
            changed_target_channel_count=12,synthetic_unit_test_only=True)
        return SimpleNamespace(observation=self.observation,reward=-.02+.003*((self.index-1)%7),
            terminated=self.done,info=dict(phase_id='P13',applied_raw_full12=list(raw),
                actuator_target_effect_audit=native,full_task_success=False,synthetic_unit_test_only=True))


@pytest.mark.parametrize('terminal_tail',[False,True])
def test_actual_train_loop512_official20_raw_logp_bootstrap_and_temporary_strict_reload(tmp_path,monkeypatch,terminal_tail):
    runner,lineage,counts = initialized()
    actor = runner.alg.actor
    frozen = state_hash(actor.frozen_accepted.state_dict())
    head_before = state_hash(actor.mlp.state_dict())
    critic_steps = {name:float(runner.alg.optimizer.state[p]['step']) for name,p in runner.alg.critic.named_parameters()}
    run=tmp_path/'synthetic_run';run.mkdir()
    monkeypatch.setattr(route,'OUTPUT',tmp_path/'artifacts')
    core = SyntheticCore(terminal_tail)
    pointer=route.train(core,runner,runtime(),lineage,counts,run,512)
    assert Path(pointer['checkpoint']).is_relative_to(tmp_path)
    assert counts['finish_policy_decisions']==512 and counts['finish_ppo_updates']==1
    assert counts['finish_optimizer_steps']==20 and counts['auxiliary_updates']==0
    assert counts['prefix_decisions']==0 and counts['completed_episodes']==int(terminal_tail)
    reports=[json.loads(s) for s in (run/'updates.jsonl').read_text().splitlines()]
    assert len(reports)==1 and reports[0]['optimizer_steps']==20
    assert reports[0]['actual_phase_counts']=={'P13':512}
    assert reports[0]['actual_finish_mode_counts']=={
        'ADVANCE_FOR_HOME_CLEARANCE':256,'HOME_RECOVERY':128,'FINAL_SETTLE':128}
    assert reports[0]['actor_parameters_changed']
    assert state_hash(actor.frozen_accepted.state_dict())==frozen
    assert state_hash(actor.mlp.state_dict())!=head_before
    actor.assert_frozen_state(runner.alg.optimizer)
    assert runner.alg.storage.step==0 and runner.alg.transition.actions is None
    assert all(float(runner.alg.optimizer.state[p]['step'])==20 for p in actor.trainable_parameters())
    assert all(float(runner.alg.optimizer.state[p]['step'])==critic_steps[name]+20
               for name,p in runner.alg.critic.named_parameters())
    roll=torch.load(run/'rollouts/rollout_0001.pt',map_location='cpu',weights_only=False)
    decisions=[json.loads(s) for s in (run/'decisions.jsonl').read_text().splitlines()]
    assert len(decisions)==512 and all(r['PPO_credit']==1 for r in decisions)
    for i,row in enumerate(decisions):
        raw=torch.tensor(row['policy_request']['selected_raw_full12'])
        assert torch.equal(roll['actions'][i,0],raw)
        assert torch.equal(roll['actions_log_prob'][i,0].view(-1),torch.tensor(row['old_logp']).view(-1))
        mean=torch.tensor(row['policy_request']['conditional_mean_full12'],dtype=torch.float64)
        std=torch.tensor(row['policy_request']['active_conditional_std_full12'],dtype=torch.float64)
        same_dtype=torch.distributions.Normal(mean.float(),std.float()).log_prob(raw).sum()
        assert torch.equal(same_dtype,roll['actions_log_prob'][i,0,0])
        manual=torch.distributions.Normal(mean,std).log_prob(raw.double()).sum()
        # Cross-dtype reduction is not bitwise: float64 recomputation of this
        # stored float32 Gaussian differs by at most 4.617e-6 in the fixed case.
        assert torch.allclose(manual,roll['actions_log_prob'][i,0,0].double(),atol=6.e-6,rtol=0.)
    assert torch.isfinite(roll['advantages']).all() and torch.isfinite(roll['returns']).all()
    if terminal_tail:
        assert bool(roll['dones'][-1,0,0])
        # Official RSL computes (reward - value) + value. Compare that exact
        # float32 computation, not an unrealistically cancellation-free r.
        terminal_return=(roll['rewards'][-1]-roll['values'][-1])+roll['values'][-1]
        assert torch.equal(roll['returns'][-1],terminal_return)
        assert torch.allclose(roll['returns'][-1],roll['rewards'][-1],atol=2.e-6,rtol=0.)
    else:
        # Compute the PRE-update bootstrap from the immutable source critic;
        # fresh columns were zero and the old learned critic was preserved.
        saved=source_data(); old=route.accepted.make_runner('cpu',1001,saved_configuration=saved['infos']['runner_config'])
        route.accepted.load(old,route.ROOT/route.settings()['accepted_checkpoint'],saved['infos']['runtime_contract'])
        with torch.inference_mode():
            val=old.alg.critic(route.accepted.tensor_observation(core.observation[:490],'cpu')).view(-1)[0]
        assert torch.allclose(roll['returns'][-1,0,0],roll['rewards'][-1,0,0]+.9985*val,atol=2.e-5,rtol=0.)
    # Reproduce exact save-time RNG, independent of the extra bootstrap audit.
    meta=json.loads(Path(pointer['manifest']).read_text())
    restored=route.make_runner('cpu',1001,saved_configuration=meta['runner_config'])
    restored_lineage,restored_counts=route.load(restored,pointer['checkpoint'],runtime())
    assert restored_lineage==lineage and restored_counts==counts
    assert state_hash(restored.alg.save())==state_hash(runner.alg.save())
    assert capture_training_rng_state(seed=1001)==meta['training_rng']
    with pytest.raises(ValueError,match='contract/hash/ledger'):
        route.load(restored,pointer['checkpoint'],dict(runtime(),experiment_id='wrong_route'))


def test_evaluate_route_exposes_all_reused_wrapper_keys(monkeypatch,tmp_path):
    captured={}
    def old_evaluate(core,runner,runtime,lineage,counts,run,diagnostic,*,route):
        captured.update(route)
        return {'test_only':True}
    monkeypatch.setattr(route.old,'evaluate',old_evaluate)
    result=route.evaluate(None,None,{},dict(historical_AUX_updates=96),{},tmp_path)
    assert result['test_only']
    assert captured['validate_runner'] is route.validate_runner
    assert captured['tensor_observation'] is route.tensor_observation and captured['request'] is route.request
    cfg=captured['settings']
    for key in ('version','observation_dimension','local_mean_coordinate_gain_full12',
                'capture_source_dispatch','source_tracking_owner_revision','rr_authorized_assist','finish_recovery'):
        assert key in cfg
    assert cfg['observation_dimension']==531 and cfg['finish_recovery']['enabled']
    assert captured['auxiliary_ledger']==[]
    assert captured['control_contributions']['public_finish_module']
    assert captured['config_dir']==route.ROOT/route.settings()['accepted_control_config']

# Adaptation draft only: install under tests/unit after production modules exist.
"""Bounded CPU runner integration. No production paths, GPU, Isaac, or PPO credit.

CUDA RNG is an opaque CPU byte bank in these tests: exact serialization is
checked, but no CUDA RNG execution/restoration is claimed. CPU/NumPy/Python RNG,
model/Adam computations, official PPO update and checkpoint IO are actual.
"""
import copy
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/wlr50_clean').is_dir())
CANDIDATE_CONFIG = ROOT/'outputs/ppo_finish_advance_then_home_v1/staged_B/local_training.candidate.json'
sys.path.insert(0,str(ROOT/'tests/unit'))
from test_semantic_finish_advance_actor import recorded490
from wlr50_clean.ppo.semantic_training import state_hash
from wlr50_clean.ppo import semantic_fl_forward as production_route
from wlr50_clean.ppo.rl_library_wrapper import capture_training_rng_state, restore_training_rng_state, _decode_torch_rng_state


@pytest.fixture
def route(monkeypatch,tmp_path):
    cfg=json.loads(CANDIDATE_CONFIG.read_text())
    # Structural regression fixture remains the physically successful A0 even
    # if the eventual B deployment selects a later validated A checkpoint.
    cfg.update(
        accepted_A_checkpoint='outputs/ppo_finish_advance_then_home_v1/checkpoints/history/checkpoint_CP232960_finish000000_gc039c74281c1.pt',
        accepted_A_sha256='c5f2d8ae756296f9392e71955ea0bacd41397cd8983b3a98fbb1dea75566b845',
        accepted_A_manifest_sha256='825b6b70e3ba754fa613fb18a49ea722bbbad4b45f5c40d0f0b5d02944238d37',
        accepted_A_evaluation_manifest='runs/ppo_finish_advance_then_home_v1/A0_fixed_DET_c039c74/source/source_manifest.json',
        accepted_A_evaluation_sha256='e5fb6e8d8d52d1c836f858a5e8a6d22725d7f47082099c60f49961601951bde8',
        accepted_A_selection='immutable successful A0 structural CPU fixture, not deployment selection',
        historical_decision_origin=232960)
    assert cfg['observation_dimension']==535
    source=ROOT/cfg['accepted_A_checkpoint']
    meta=json.loads(source.with_name(source.stem+'_manifest.json').read_text())
    bank=[_decode_torch_rng_state(value,label='test-only opaque CUDA state') for value in meta['training_rng']['torch_cuda']]
    def set_bank(values): bank[:]=[value.detach().cpu().clone() for value in values]
    monkeypatch.setattr(torch.cuda,'is_available',lambda:bool(bank))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:len(bank))
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[v.clone() for v in bank])
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',set_bank)
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda *a,**k:None)
    def no_cuda(*a,**k): raise AssertionError('GPU initialization forbidden in CPU test')
    monkeypatch.setattr(torch.cuda,'init',no_cuda)
    monkeypatch.setattr(torch.cuda,'_lazy_init',no_cuda)
    threads=torch.get_num_threads();torch.set_num_threads(1)
    original_rng=capture_training_rng_state(seed=1001)
    route=production_route
    monkeypatch.setattr(route,'settings',lambda:copy.deepcopy(cfg))
    monkeypatch.setattr(route,'OUTPUT',tmp_path/'artifacts')
    route.test_metadata=meta
    yield route
    restore_training_rng_state(original_rng,expected_seed=1001)
    torch.set_num_threads(threads)


def runtime(route):
    return dict(experiment_id=route.NAME,local_contract=route.settings(),source_git_commit='0'*40,
                test_only='synthetic_CPU_and_opaque_CUDA_RNG_not_physical_PPO')


def observation(index=0,active=True):
    values=recorded490()[index%32].tolist()+[0.]*41+[float(active),.8,1.,1.]
    assert len(values)==535
    return values


def initialize(route):
    runner=route.make_runner('cpu')
    lineage,counts=route.initialize_from_A(runner)
    return runner,lineage,counts


def test_actual_A0_migration_preserves_critic_Adam_LR_RNG_frozen_prefix_and_source(route):
    cfg=route.settings(); source=ROOT/cfg['accepted_A_checkpoint']
    before_sha=route.sha(source)
    runner,lineage,counts=initialize(route)
    old=torch.load(source,map_location='cpu',weights_only=False)
    assert before_sha==cfg['accepted_A_sha256']==route.sha(source)
    assert not any(counts.values()) and lineage['new_actor_parameter_count']==2
    assert lineage['critic_Adam_inherited'] and lineage['new_actor_Adam_empty']
    assert lineage['old_rollout_reused'] is False
    assert runner.alg.learning_rate==old['infos']['learning_rate']
    assert capture_training_rng_state(seed=1001)==old['infos']['training_rng']
    actor=runner.alg.actor
    assert state_hash(actor.frozen_A.state_dict())==state_hash(old['actor_state_dict'])
    names=list(runner.alg.critic.named_parameters())
    ids=old['optimizer_state_dict']['param_groups'][0]['params'][-len(names):]
    for (name,p),oid in zip(names,ids,strict=True):
        value=old['critic_state_dict'][name]
        if p.shape!=value.shape:
            assert p.shape[1]==535 and value.shape[1]==531
            assert torch.equal(p[:,:531],value) and not torch.count_nonzero(p[:,531:])
        else: assert torch.equal(p,value)
        for key,v in old['optimizer_state_dict']['state'][oid].items():
            got=runner.alg.optimizer.state[p][key]
            if torch.is_tensor(v) and v.shape!=got.shape:
                assert torch.equal(got[:,:531],v) and not torch.count_nonzero(got[:,531:])
            elif torch.is_tensor(v): assert torch.equal(got,v)
            else: assert got==v
    assert all(p not in runner.alg.optimizer.state for p in actor.trainable_parameters())
    actor.assert_frozen_state(runner.alg.optimizer)
    obs=route.tensor_observation(observation(active=False),'cpu')
    rng=capture_training_rng_state(seed=1001)
    with torch.inference_mode():
        raw,audit=route.request(runner,obs,stochastic=False)
        accepted=actor.frozen_A({'policy':obs['policy'][:,:531]},stochastic_output=False)
    assert torch.equal(raw,accepted) and audit['sampling_draws']==0
    assert capture_training_rng_state(seed=1001)==rng
    assert runner.alg.storage.step==0 and runner.alg.transition.actions is None
    assert all(not torch.is_inference(v) for state in runner.alg.optimizer.state.values()
               for v in state.values() if torch.is_tensor(v))


def test_zero_learning_strict_B_save_reload_preserves_all_states_and_rejects_wrong_contract(route,tmp_path):
    runner,lineage,counts=initialize(route)
    before=state_hash(runner.alg.save()); rng=capture_training_rng_state(seed=1001)
    pointer=route.save(runner,runtime(route),lineage,counts,source_run=tmp_path,publish_pointer=False)
    assert not (route.OUTPUT/'checkpoints/checkpoint_last_pointer.json').exists()
    assert Path(pointer['checkpoint']).is_relative_to(tmp_path)
    meta=json.loads(Path(pointer['manifest']).read_text())
    restored=route.make_runner('cpu',saved_configuration=meta['runner_config'])
    recovered_lineage,recovered_counts=route.load(restored,pointer['checkpoint'],runtime(route))
    assert recovered_lineage==lineage and recovered_counts==counts
    assert state_hash(restored.alg.save())==before
    assert capture_training_rng_state(seed=1001)==rng==meta['training_rng']
    assert restored.alg.learning_rate==meta['learning_rate']
    with pytest.raises(ValueError,match='contract/hash/ledger'):
        route.load(restored,pointer['checkpoint'],dict(runtime(route),experiment_id='other'))


class SyntheticCore:
    """Simple full512 active train arm or short prefix/inactive/terminal arm."""
    def __init__(self,*,all_active=False):
        self.all_active=all_active;self.resets=0
        self.reset()
    def reset(self,seed=1001):
        self.resets+=1;self.index=0;self.done=False
        self._sync()
        return self.observation
    def _sync(self):
        active=self.all_active or 2<=self.index<4
        self.fl_window={'active':active,'test_only':True}
        self.frame=SimpleNamespace(physics_tick=self.index*8,sim_time_s=self.index/15.,
                                  state_id='P06' if active else 'P05' if self.index<2 else 'P07')
        self.observation=observation(self.index,active)
    def step(self,raw):
        phase=self.frame.state_id
        self.index+=1
        self.done=(self.index==512 if self.all_active else self.index==6)
        self._sync()
        audit=dict(schema='wlr50_clean.actuator_target_effect_audit.v1',verified=True,
            actual_mapping_matches_dispatch=True,setter_dispatch_targets_equal=True,
            same_tick_counterfactual=True,raw_policy_action_full12=list(raw),target_dtype='torch.float32',
            changed_target_channel_count=12,synthetic_unit_test_only=True)
        return SimpleNamespace(observation=self.observation,reward=-.01+.002*(self.index%5),
            terminated=self.done,info=dict(phase_id=phase,applied_raw_full12=list(raw),
                actuator_target_effect_audit=audit,full_task_success=False,synthetic_unit_test_only=True))


def test_bounded_prefix_then_active_inactive_terminal_is_contiguous_without_fake_done(route):
    runner,lineage,counts=initialize(route);core=SyntheticCore();stream=io.StringIO()
    route.prefix(core,runner,stream,counts)
    assert counts['prefix_decisions']==2 and counts['opportunities']==1
    assert runner.alg.storage.step==0 and runner.alg.transition.actions is None
    rows=[json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(rows)==2 and all(r['actor_credit']==r['critic_credit']==0 for r in rows)
    flags=[];done=[]
    for _ in range(4):
        obs=route.tensor_observation(core.observation,'cpu')
        with torch.no_grad():
            raw,audit=route.request(runner,obs,stochastic=True)
            step=core.step(raw[0].tolist())
            nxt=route.tensor_observation(step.observation,'cpu')
            runner.alg.process_env_step(nxt,torch.tensor([step.reward]),torch.tensor([step.terminated]),{})
        flags.append(audit['active']);done.append(step.terminated)
    assert flags==[True,True,False,False] and done==[False,False,False,True]
    assert runner.alg.storage.step==4
    assert runner.alg.storage.distribution_params[2][:4,0,0].tolist()==[1.,1.,0.,0.]
    assert runner.alg.storage.actions.shape[-1]==12
    assert runner.alg.storage.actions_log_prob[2:4].eq(0.).all()
    assert runner.alg.storage.dones[:4,0,0].tolist()==[0,0,0,1]


def test_actual_train512_raw_storage_scalar_likelihood20steps_and_tmp_checkpoint(route,tmp_path):
    runner,lineage,counts=initialize(route)
    actor=runner.alg.actor;frozen=state_hash(actor.frozen_A.state_dict())
    prior_critic_steps={name:float(runner.alg.optimizer.state[p]['step']) for name,p in runner.alg.critic.named_parameters()}
    run=tmp_path/'actual_runner_synthetic512';run.mkdir()
    core=SyntheticCore(all_active=True)
    pointer=route.train(core,runner,runtime(route),lineage,counts,run,512)
    assert counts['fl_policy_decisions']==counts['continuous_storage_decisions']==512
    assert counts['critic_only_decisions']==0 and counts['episodes']==1
    assert counts['ppo_updates']==1 and counts['optimizer_steps']==counts['actor_optimizer_steps']==20
    assert counts['auxiliary_updates']==0
    assert all(float(runner.alg.optimizer.state[p]['step'])==20 for p in actor.trainable_parameters())
    assert all(float(runner.alg.optimizer.state[p]['step'])==prior_critic_steps[name]+20
               for name,p in runner.alg.critic.named_parameters())
    actor.assert_frozen_state(runner.alg.optimizer)
    assert state_hash(actor.frozen_A.state_dict())==frozen
    assert runner.alg.storage.step==0 and runner.alg.transition.actions is None
    rows=[json.loads(line) for line in (run/'decisions.jsonl').read_text().splitlines()]
    roll=torch.load(run/'rollouts/rollout_0001.pt',map_location='cpu',weights_only=False)
    assert len(rows)==512 and all(r['actor_credit']==r['critic_credit']==1 for r in rows)
    assert roll['actions'].shape==(512,1,12)
    assert all(p.shape==(512,1,1) for p in roll['distribution_params'])
    for i,row in enumerate(rows):
        raw=torch.tensor(row['policy_request']['selected_raw_full12'])
        assert torch.equal(roll['actions'][i,0],raw)
        assert torch.equal(roll['actions_log_prob'][i,0].view(-1),torch.tensor(row['old_logp']).view(-1))
        mu,sigma,_=roll['distribution_params']
        expected=torch.distributions.Normal(mu[i,0,0],sigma[i,0,0]).log_prob(raw[8])
        assert torch.equal(expected,roll['actions_log_prob'][i,0,0])
    assert not roll['dones'][:-1].any() and bool(roll['dones'][-1,0,0])
    terminal=(roll['rewards'][-1]-roll['values'][-1])+roll['values'][-1]
    assert torch.equal(roll['returns'][-1],terminal)
    report=json.loads((run/'updates.jsonl').read_text())
    assert report['total_continuous_storage_rows']==report['stochastic_FL_rows']==512
    assert report['actual_phase_counts']=={'P06':512}
    assert not report['ordinary_phase_done'] and not report['skipped_off_window_time']
    meta=json.loads(Path(pointer['manifest']).read_text())
    restored=route.make_runner('cpu',saved_configuration=meta['runner_config'])
    restored_lineage,restored_counts=route.load(restored,pointer['checkpoint'],runtime(route))
    assert restored_lineage==lineage and restored_counts==counts
    assert state_hash(restored.alg.save())==state_hash(runner.alg.save())
    assert capture_training_rng_state(seed=1001)==meta['training_rng']
    assert route.sha(ROOT/route.settings()['accepted_A_checkpoint'])==route.settings()['accepted_A_sha256']

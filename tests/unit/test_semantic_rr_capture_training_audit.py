"""One actual CPU PPO update on synthetic ABI data, not robot task evidence."""
from collections import Counter
import json
from pathlib import Path

import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('rsl_rl')

from test_semantic_training import Core
from wlr50_clean.ppo import semantic_training as training
from wlr50_clean.ppo.semantic_rr_capture_profile import (
    RR_CAPTURE_POLICY,RR_CAPTURE_OBSERVATION_LAYOUT,RR_ASSIST_START,RR_TASK_START)


class Synthetic410Core(Core):
    """Reuse the strict synthetic native-ACK ABI; no physics or real contacts."""
    def observation(self,raw=(0.,)*12):
        row=[0.]*410
        phase=8 if self.tick<4 else 9
        row[phase]=1.;row[18]=self.tick/3000.;row[20]=self.tick/3000.
        row[158:158+phase]=[1.]*phase
        row[195:207]=raw
        row[207:219]=[.01*x for x in raw]
        # Explicit synthetic observed state, never alleged captured physical state.
        row[RR_ASSIST_START:RR_TASK_START]=[i/100. for i in range(14)]
        row[RR_TASK_START:410]=[1.,1.,0.,0.,0.,1.,0.]
        return tuple(row)

    def reset(self,*,seed=1001,options=None):
        super().reset(seed=seed,options=options)
        return self.observation()

    def step(self,raw):
        phase='P09' if self.tick<4 else 'P10'
        result=super().step(raw)
        result.observation=self.observation(raw)
        result.info.update(phase_id=phase,end_phase_id='P09' if self.tick<4 else 'P10',
            physics_tick=self.tick*8,physics_ticks=8,sim_time_s=self.frame.sim_time_s,
            terminal_bootstrap_allowed=not result.terminated)
        return result


def test_rr410_actual_train_route_emits_request_and_every_minibatch_likelihood(tmp_path):
    assert not torch.cuda.is_available(),'run this bounded regression with CUDA_VISIBLE_DEVICES=-1'
    rng=torch.get_rng_state();threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        def make():
            env=training.SemanticRslAdapter(Synthetic410Core(),seed=1001,device='cpu')
            env.cfg['semantic_version']='v3'
            runner,_=training.construct_semantic_runner(env,seed=1001,device='cpu',initialize_actor=False,
                policy_version=RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
            return runner,env
        runner,env=make()
        contract={'experiment_id':'rr_capture_then_rl_transfer_v1','semantic_version':'v3',
            'training_budgets':training.training_quantity_budgets('rr_capture_then_rl_transfer_v1'),
            'evidence':'synthetic CPU ABI only; no real simulator or training credit'}
        result=training.train_semantic(runner,env,run_dir=tmp_path/'run',output_root=tmp_path/'output',
            stage='full_episode',decisions=128,contract=contract,seed=1001,checkpoint_interval_updates=1)
        assert result['actual_policy_decisions']==128 and result['ppo_updates_this_run']==1
        assert result['optimizer_steps_this_run']==20 and result['finite_nonzero_gradient_observed']
        assert result['actor_parameter_sha256_before']!=result['actor_parameter_sha256_after']
        rollout=torch.load(tmp_path/'run/rollouts/rollout_000001.pt',map_location='cpu',weights_only=False)
        assert rollout['observations']['policy'].shape==(128,1,410)
        rows=[json.loads(line) for line in (tmp_path/'run/residual_and_projection_audit.jsonl').read_text().splitlines()]
        assert len(rows)==128 and any(row['terminal'] for row in rows)
        for i,row in enumerate(rows):
            request=row['policy_request'];x=rollout['observations']['policy'][i,0]
            assert request['policy_version']==RR_CAPTURE_POLICY
            assert request['sampling_draws']==1
            assert request['extra_random_draws']==request['extra_model_forwards']==0
            assert request['rr_capture_assist_observed_features']==x[RR_ASSIST_START:RR_TASK_START].tolist()
            assert request['rr_capture_transfer_observed_features']==x[RR_TASK_START:410].tolist()
            assert request['selected_raw_full12']==row['raw_policy_action_full12']==rollout['actions'][i,0].tolist()
            assert row['raw_policy_action_full12']!=row['applied_audit']['applied_action_full12']
            assert request['selected_raw_log_probability']==row['old_log_probability']==rollout['actions_log_prob'][i,0].item()
            assert request['conditional_mean_full12']==row['old_distribution_mean_full12']==rollout['distribution_params'][0][i,0].tolist()
            assert request['effective_sigma_full12']==row['old_distribution_std_full12']==rollout['distribution_params'][1][i,0].tolist()
            assert row['reward']==rollout['rewards'][i,0].item()
            assert row['terminal']==bool(rollout['dones'][i,0].item())
            assert request['transformed_actuator_targets_are_not_policy_samples']
        audit=json.loads((tmp_path/'run/rollouts/update_000001_likelihood.json').read_text())
        assert audit['extra_random_draws']==audit['extra_model_forwards']==0
        assert len(audit['minibatches'])==20
        exposure=Counter()
        for batch in audit['minibatches']:
            for j,indices in enumerate(batch['rollout_flat_indices']):
                assert len(indices)==1
                exposure[indices[0]]+=1
                assert batch['old_log_probability'][j]==rollout['actions_log_prob'][indices[0],0].item()
            for key in ('current_network_mean_full12','current_network_log_sigma_full12',
                        'loss_gradient_wrt_network_mean_full12','loss_gradient_wrt_network_log_sigma_full12'):
                assert torch.tensor(batch[key]).shape==(32,12)
        assert exposure==Counter({i:5 for i in range(128)})
        checkpoint=Path(result['checkpoints'][0]['checkpoint'])
        fresh,_=make()
        loaded=training.load_semantic_checkpoint(fresh,checkpoint,contract=contract,seed=1001)
        assert (loaded['global_policy_decisions'],loaded['ppo_updates'],loaded['optimizer_steps'])==(128,1,20)
        assert training.parameter_hash(fresh.alg.actor)==training.parameter_hash(runner.alg.actor)
        assert training.state_hash(fresh.alg.optimizer.state_dict())==training.state_hash(runner.alg.optimizer.state_dict())
        assert all(float(row['step'])==20. for row in fresh.alg.optimizer.state_dict()['state'].values())
    finally:
        torch.set_rng_state(rng);torch.set_num_threads(threads)

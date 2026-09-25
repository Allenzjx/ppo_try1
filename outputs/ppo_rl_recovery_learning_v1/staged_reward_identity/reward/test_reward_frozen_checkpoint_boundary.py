"""DEFERRED real frozen439 CP test: run only after the sole Isaac stops.

No Torch/model import occurs unless both explicit environment keys are supplied.
This is inference/immutability verification, zero PPO/AUX credit, no physics.
RR_REWARD_NO_ISAAC=confirmed
RR_REWARD_FROZEN_CP=<actual latest immutable .pt file, not a mutable pointer>
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import unittest


class DeferredFrozenCheckpoint(unittest.TestCase):
    def test_reward_only_has_no_direct_effect_on_frozen_actor_critic(self):
        checkpoint=os.environ.get('RR_REWARD_FROZEN_CP')
        if os.environ.get('RR_REWARD_NO_ISAAC')!='confirmed' or not checkpoint:
            self.skipTest('DEFERRED: real CP inference forbidden alongside active Isaac; explicit boundary required')
        # Deliberately deferred below the opt-in, not module-level imports.
        import torch
        from tensordict import TensorDict
        from wlr50_clean.ppo import semantic_training as training
        from wlr50_clean.ppo import semantic_reward as production_reward
        from wlr50_clean.ppo.semantic_observation import load_semantic_observation_schema
        from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
        from wlr50_clean.ppo.semantic_rear_owner_profile import REAR_OWNER_POLICY,REAR_OWNER_OBSERVATION_LAYOUT
        from test_reward_only_stdlib import ROOT,task,ground,frame,samples
        self.assertEqual(production_reward.RR_RETENTION_REWARD_ONLY_MODE,'rr_recapture_retention_reward_only_v1',
            'Run this test against the reviewed, applied production module after the legal boundary.')
        path=Path(checkpoint).resolve(strict=True)
        self.assertNotEqual(path.name,'checkpoint_last.pt','use immutable source')
        manifest=json.loads(path.with_name(path.stem+'_manifest.json').read_text())
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),manifest['checkpoint_sha256'])
        original_rng=torch.get_rng_state();original_threads=torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            payload=torch.load(path,map_location='cpu',weights_only=False)
            self.assertEqual(payload['actor_state_dict']['mlp.0.weight'].shape[1],439)
            runner,_=training.construct_semantic_runner(_shape_env(439,'cpu'),seed=1001,device='cpu',
                initialize_actor=False,policy_version=REAR_OWNER_POLICY,observation_layout=REAR_OWNER_OBSERVATION_LAYOUT)
            actor,critic=runner.alg.actor,runner.alg.critic
            actor.load_state_dict(payload['actor_state_dict'],strict=True)
            critic.load_state_dict(payload['critic_state_dict'],strict=True)
            actor.eval();critic.eval()
            hashes=[training.parameter_hash(model) for model in (actor,critic)]
            previous,_=frame(task());current,_=frame(task(dict(ground(),top_xy_outside_distance_m=.01)))
            schema=load_semantic_observation_schema(ROOT/'configs/ppo_rr_rl_timing_policy_learning_v1/observation_schema.json')
            config=production_reward.load_semantic_reward_config(ROOT/'configs/ppo_rr_rl_timing_policy_learning_v1/reward_config.yaml')
            previous.metrics['sim_time_s']=58.266666666666666
            current.metrics['sim_time_s']=58.333333333333336
            frozen_frame=deepcopy(previous)
            encoded_before=schema.encode(previous.groups)
            def obs(values):
                x=torch.tensor([values],dtype=torch.float32)
                return TensorDict({'policy':x,'critic':x.clone()},batch_size=[1])
            with torch.no_grad():
                for stochastic in (False,True):
                    before_obs=obs(encoded_before);rng=torch.get_rng_state().clone()
                    raw_before,audit_before=training.audited_history_policy_request(actor,before_obs,
                        lambda:actor(before_obs,stochastic_output=stochastic),stochastic=stochastic)
                    value_before=critic(before_obs).clone()
                    reward=production_reward.SemanticRewardCalculator(config).evaluate(
                        previous,current,samples(previous,current),termination_reason=None,task_success=False)
                    self.assertNotEqual(reward['potential_before'],previous.task['task_progress_potential'])
                    self.assertEqual(previous,frozen_frame)
                    encoded_after=schema.encode(previous.groups)
                    self.assertEqual(encoded_before,encoded_after)
                    after_obs=obs(encoded_after);torch.set_rng_state(rng)
                    raw_after,audit_after=training.audited_history_policy_request(actor,after_obs,
                        lambda:actor(after_obs,stochastic_output=stochastic),stochastic=stochastic)
                    self.assertTrue(torch.equal(raw_before,raw_after))
                    self.assertTrue(torch.equal(value_before,critic(after_obs)))
                    keys=('conditional_mean_full12','effective_sigma_full12','selected_raw_full12')
                    if stochastic:
                        keys += ('selected_raw_log_probability',)
                    else:
                        self.assertNotIn('selected_raw_log_probability',audit_before)
                        self.assertNotIn('selected_raw_log_probability',audit_after)
                    for key in keys:
                        self.assertEqual(audit_before[key],audit_after[key])
                    self.assertEqual(hashes,[training.parameter_hash(model) for model in (actor,critic)])
        finally:
            torch.set_rng_state(original_rng);torch.set_num_threads(original_threads)


if __name__=='__main__':unittest.main(verbosity=2)

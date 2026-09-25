"""Dry/synthetic analyzer checks only: never read current run or import Torch."""
import ast
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import analyze_rr_learning_signal_update8 as review


def pair():
    counts=dict(local_policy_decisions=3584,local_ppo_updates=7,local_optimizer_steps=140,
        task_v2_policy_decisions=1536,task_v2_ppo_updates=3,auxiliary_updates=64,prefix_decisions=100)
    runtime=dict(runtime_content_sha256='a'*64,local_contract={review.GAIN_KEY:review.GAIN,'rollout_length':512})
    config=dict(num_steps_per_env=512,algorithm=dict(num_learning_epochs=5,num_mini_batches=4),
                actor={review.GAIN_KEY:review.GAIN,'observation_layout':review.previous.OBSERVATION_LAYOUT})
    event=dict(source_head='0ff03eafeeb75ba8505a99478cea2a6243cf93f5',destination_head=review.HEAD,
        source_gain=[1.]*12,target_gain=review.GAIN,destination_runtime_sha256='a'*64,
        source_counts=copy.deepcopy(counts),destination_counts=copy.deepcopy(counts),new_policy_decisions=0,new_PPO_updates=0,
        new_PPO_Adam_steps=0,new_AUX_steps=0)
    before=dict(counts=counts,runner_config=config,runtime_contract=runtime,
        local_auxiliary_events=[dict(actual_optimizer_steps=64,accepted_steps=64)],
        **{review.LEDGER_KEY:[event]})
    after=copy.deepcopy(before)
    after['counts'].update(local_policy_decisions=4096,local_ppo_updates=8,local_optimizer_steps=160,
        task_v2_policy_decisions=2048,task_v2_ppo_updates=4,prefix_decisions=125)
    return before,after


class Update8Tests(unittest.TestCase):
    def test_explicit_exit_before_any_source_or_tensor(self):
        with patch.object(review,'sealed_inputs',side_effect=AssertionError('must not read run')):
            with self.assertRaisesRegex(ValueError,'Explicit Isaac exit'):review.analyze()
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pxr',sys.modules)

    def test_unsealed_refused_before_started_or_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            with patch.object(review,'RUN',run):
                with self.assertRaisesRegex(ValueError,'not sealed COMPLETE'):review.sealed_inputs(run)
                (run/'run_manifest.json').write_text(json.dumps(dict(lifecycle='RUNNING',mode='train')))
                with self.assertRaisesRegex(ValueError,'sealed COMPLETE'):review.sealed_inputs(run)

    def test_actual_deltas_and_unchanged_coordinates_aux(self):
        before,after=pair(); result=review.validate_pair(before,after)
        self.assertEqual(result['delta']['prefix_decisions'],25)
        self.assertEqual((result['rows'],result['absolute_update'],result['update_offset']),(512,8,7))
        for mutation in ('AUX','coordinates','gain','partial','Adam','config'):
            a,b=copy.deepcopy(before),copy.deepcopy(after)
            if mutation=='AUX':b['counts']['auxiliary_updates']=128
            elif mutation=='coordinates':b[review.LEDGER_KEY][0]['new_AUX_steps']=1
            elif mutation=='gain':a['runner_config']['actor'][review.GAIN_KEY]=[1.]*12
            elif mutation=='partial':b['counts']['local_policy_decisions']-=1
            elif mutation=='Adam':b['counts']['local_optimizer_steps']+=1
            else:b['runner_config']['actor']['obs_normalization']=True
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):review.validate_pair(a,b)

    def test_logged_local_is_already_gain_applied_and_one_history(self):
        p=dict(local_mean_coordinate_gain_full12=review.GAIN,rho=.9,extra_model_forwards=0,
            sampling_draws=1,history_kernel_applications=1,extra_random_draws=0,
            selected_raw_log_probability=.25,prior_raw_mean_full12=[.2]*12,
            local_raw_mean_delta_full12=[.03]*12,applied_local_raw_mean_delta_full12=[.03]*12,
            combined_raw_mean_full12=[.23]*12,history_center_full12=[.4]*12,
            conditional_mean_full12=[.383]*12,selected_raw_full12=[.4]*12,
            selected_tanh_full12=[math.tanh(.4)]*12)
        row=dict(policy=p,old_logp=.25,observation=[0.]*448)
        self.assertLess(review.validate_gain_rows([row])['maximum_raw_HISTORY_tanh_error'],1e-6)
        for mutation in ('gain','double_gain','extra_history'):
            bad=copy.deepcopy(row)
            if mutation=='gain':bad['policy'][review.GAIN_KEY]=[1.]*12
            elif mutation=='double_gain':bad['policy']['conditional_mean_full12'][7]=.41
            else:bad['policy']['history_kernel_applications']=2
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):review.validate_gain_rows([bad])

    def test_update_number_and_actual_minibatch_exposures(self):
        rows=[dict(old_logp=.01*i,phase='P09') for i in range(512)]
        accounting=review.validate_pair(*pair())
        batches=[dict(rollout_flat_indices=[[i] for i in range(start,start+128)],
            old_log_probability=[rows[i]['old_logp'] for i in range(start,start+128)])
            for _ in range(5) for start in range(0,512,128)]
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);(run/'rollouts').mkdir();path=run/'rollouts/likelihood_0008.json'
            path.write_text(json.dumps(dict(minibatches=batches)))
            result=review.minibatch_evidence(run,rows,dict(optimizer_steps=20),accounting)
            self.assertEqual(result['row_exposures'],2560)
            batches[-1]['old_log_probability'][-1]+=1
            path.write_text(json.dumps(dict(minibatches=batches)))
            with self.assertRaisesRegex(ValueError,'likelihood mismatch'):
                review.minibatch_evidence(run,rows,dict(optimizer_steps=20),accounting)

    def test_dry_syntax_and_no_tensor_import(self):
        source=Path(review.__file__).read_text();ast.parse(source)
        self.assertIn('cfg = dict(meta[\'runner_config\'][\'actor\'])',source)
        self.assertNotIn('cfg["legacy447_migration_only"] = True',source)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pxr',sys.modules)


if __name__=='__main__':unittest.main()

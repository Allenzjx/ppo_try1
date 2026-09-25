"""Small synthetic stdlib tests; never reads the active training run."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import analyze_rr_learning_signal_update7 as review


def metadata(decisions, updates, adam, task_decisions, task_updates):
    return dict(counts=dict(local_policy_decisions=decisions,local_ppo_updates=updates,
        local_optimizer_steps=adam,task_v2_policy_decisions=task_decisions,
        task_v2_ppo_updates=task_updates,auxiliary_updates=64),
        local_auxiliary_events=[dict(actual_optimizer_steps=64,accepted_steps=64,PPO_credit=0)])


def row(index, before_top=False, after_top=False, success=False, hold=0., episode=1):
    def metrics(top):
        return dict(current_top_contact=top,current_top_bearing=top,free_air=not top,
            within_top_xy=True,current_attempt_capture_eligible=True,ground_contact=False,
            placed=top,crossed=True,gap_m=0. if top else .04)
    return dict(index=index,episode=episode,tick=(index+1)*8,phase='P09',
        before=metrics(before_top),after=metrics(after_top),local_success=success,hold_s=hold,
        local_reward=dict(terminated=success),old_logp=.01*index)


class Update7Tests(unittest.TestCase):
    def test_explicit_exit_required_before_read_or_tensor(self):
        with patch.object(review,'sealed_inputs',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'Explicit Isaac exit'):
                review.analyze()
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('pxr',sys.modules)

    def test_unsealed_refused_before_started_or_checkpoint_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            with patch.object(review,'RUN',run):
                with self.assertRaisesRegex(ValueError,'not sealed COMPLETE'):
                    review.sealed_inputs(run)
                (run/'run_manifest.json').write_text(json.dumps(dict(lifecycle='RUNNING',mode='train')))
                with self.assertRaisesRegex(ValueError,'sealed COMPLETE'):
                    review.sealed_inputs(run)

    def test_exact_update_and_aux_ledger_remain_separate(self):
        before=metadata(3072,6,120,1024,2)
        after=metadata(3584,7,140,1536,3)
        review.validate_pair(before,after)
        for mutate in (
            lambda m:m['counts'].update(auxiliary_updates=128),
            lambda m:m['counts'].update(local_policy_decisions=4096),
            lambda m:m['counts'].update(local_optimizer_steps=141),
            lambda m:m['local_auxiliary_events'][0].update(accepted_steps=63),
            lambda m:m.pop('local_auxiliary_events'),
        ):
            bad=copy.deepcopy(after); mutate(bad)
            with self.assertRaises(ValueError): review.validate_pair(before,bad)

    def test_no_success_or_top_is_valid_not_first41_assertion(self):
        rows=[row(i) for i in range(512)]
        groups=review.physical_cohorts(rows)
        self.assertEqual(len(groups['all512']),512)
        self.assertEqual(len(groups['request_AIR']),512)
        self.assertEqual(len(groups['local_success']),0)
        summary=review.contact_summary(rows,groups)
        self.assertIsNone(summary['episodes'][0]['first_TOP_endpoint'])
        self.assertEqual(summary['episodes'][0]['success_endpoint_ticks'],[])
        self.assertIn('no independent120Hz',summary['evidence_scope'])

    def test_top_drop_reacquire_and_new_episode_remain_distinct(self):
        rows=[row(0,after_top=True,hold=.02),row(1,True,False),
              row(2,after_top=True,hold=.03),row(3,True,True,True,.5),
              row(4,after_top=True,episode=2)]
        groups=review.physical_cohorts(rows)
        self.assertEqual([r['index'] for r in groups['first_TOP_transition']],[0,4])
        self.assertEqual([r['index'] for r in groups['bearing_drop_transition']],[1])
        self.assertEqual([r['index'] for r in groups['TOP_reacquisition']],[2])
        self.assertEqual([r['index'] for r in groups['local_success']],[3])
        self.assertEqual(review.contact_summary(rows,groups)['episodes'][0]['maximum_native_observer_hold_s'],.5)

    def test_historical_placed_cannot_supply_success(self):
        good=row(0,True,True,True,.5)
        review.physical_cohorts([good])
        for field,value in (('current_attempt_capture_eligible',False),('current_top_bearing',False),
                            ('ground_contact',True),('within_top_xy',False)):
            bad=copy.deepcopy(good); bad['after'][field]=value
            with self.assertRaisesRegex(ValueError,'current eligible TOP hold'):
                review.physical_cohorts([bad])
        bad=copy.deepcopy(good); bad['hold_s']=.49
        with self.assertRaises(ValueError): review.physical_cohorts([bad])

    def test_all512_real_minibatch_exposures_and_logp(self):
        rows=[row(i) for i in range(512)]
        rows[256]['phase']='P10'
        batches=[dict(rollout_flat_indices=[[i] for i in range(start,start+128)],
                      old_log_probability=[rows[i]['old_logp'] for i in range(start,start+128)])
                 for _ in range(5) for start in range(0,512,128)]
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder); (run/'rollouts').mkdir()
            path=run/'rollouts/likelihood_0007.json'
            path.write_text(json.dumps(dict(minibatches=batches)))
            result=review.minibatch_evidence(run,rows,dict(optimizer_steps=20))
            self.assertEqual(result['row_exposures'],2560)
            self.assertEqual(result['phase_rows'],{'P09':511,'P10':1})
            self.assertEqual(result['phase_row_exposures']['P10'],5)
            batches[-1]['old_log_probability'][-1]+=1
            path.write_text(json.dumps(dict(minibatches=batches)))
            with self.assertRaisesRegex(ValueError,'likelihood mismatch'):
                review.minibatch_evidence(run,rows,dict(optimizer_steps=20))


if __name__=='__main__': unittest.main()

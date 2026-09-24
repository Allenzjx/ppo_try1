"""Pure scheduler preparation checks, zero subprocess/physics/learning calls."""
import copy
import importlib.util
from pathlib import Path
import unittest

PATH=Path(__file__).with_name('resume_cooperative2048.py')
SPEC=importlib.util.spec_from_file_location('continuation_tested',PATH)
runner=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runner)


class ContinuationPlanTests(unittest.TestCase):
    def setUp(self):
        self.prior=runner.read(runner.ORIGINAL)
        self.config=runner.read(self.prior['configuration'])
    def test_remaining_exact_P10_then_naturalP01(self):
        before=copy.deepcopy(self.prior)
        blocks=runner.remaining_plan(self.prior,self.config)
        self.assertEqual([(b['from_phase'],b['decisions']) for b in blocks],[('P10',384),('P01',1280)])
        self.assertEqual(sum(b['planned_updates'] for b in blocks),13)
        self.assertTrue(all(b['prefix_policy_credit'] is False for b in blocks))
        self.assertEqual(self.prior,before)
    def test_running_original_cannot_resume(self):
        self.prior['lifecycle']='RUNNING'
        with self.assertRaises(ValueError):runner.remaining_plan(self.prior,self.config)
    def test_incomplete_prior_block_not_adopted_as384(self):
        self.prior['completed_blocks'][0]['actual_counts']['global_policy_decisions']=256
        with self.assertRaises(ValueError):runner.remaining_plan(self.prior,self.config)
    def test_altered_config_not_reallocated_silently(self):
        self.config['entries'][0]['weight']=.5
        self.config['entries'][1]['weight']=.25
        self.config['entries'][2]['weight']=.25
        with self.assertRaises(ValueError):runner.remaining_plan(self.prior,self.config)
    def test_duplicate_prior_block_refused(self):
        self.prior['completed_blocks']*=2
        with self.assertRaises(ValueError):runner.remaining_plan(self.prior,self.config)
    def test_branch_or_source_role_not_generalized(self):
        self.prior['output_branch']='../other'
        with self.assertRaises(ValueError):runner.remaining_plan(self.prior,self.config)
    def test_actual_next_source_used_not_original_ancestor(self):
        blocks=runner.remaining_plan(self.prior,self.config)
        template=copy.deepcopy(self.prior['active_command'])
        command=runner.block_command(template,blocks[1],{'checkpoint':'actual_P10_sealed_checkpoint.pt'})
        self.assertEqual(runner.command_argument(command,'-Checkpoint'),'actual_P10_sealed_checkpoint.pt')
        self.assertEqual(runner.command_argument(command,'-FromPhase'),'P01')
        self.assertEqual(runner.command_argument(command,'-Stage'),'full_episode')
        self.assertEqual(runner.command_argument(command,'-Decisions'),'1280')
        self.assertEqual(runner.command_argument(command,'-PrefixSource'),'frozen_fsm')
        self.assertEqual(runner.command_argument(command,'-CheckpointOutputBranch'),runner.BRANCH)
        self.assertEqual(template,self.prior['active_command'])
    def test_duplicate_CLI_argument_refused(self):
        with self.assertRaises(ValueError):runner.command_argument(['-Seed','1001','-Seed','2'],'-Seed')


if __name__=='__main__':unittest.main()

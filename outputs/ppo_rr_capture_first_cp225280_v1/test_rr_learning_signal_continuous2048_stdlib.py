"""Synthetic closed-chain/dry checks only; never reads the active run."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import analyze_rr_learning_signal_continuous2048 as review


class ContinuousReviewTests(unittest.TestCase):
    def test_exit_guard_before_source_reads(self):
        with patch.object(review,'sealed_inputs',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'Explicit Isaac exit'):review.analyze()
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pxr',sys.modules)

    def test_no_manifest_is_not_completed_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);(run/'some_expected_checkpoint.pt').write_bytes(b'not completion evidence')
            with patch.object(review,'RUN',run):
                with self.assertRaisesRegex(ValueError,'not sealed COMPLETE'):review.sealed_inputs(run)
                (run/'run_manifest.json').write_text(json.dumps(dict(lifecycle='RUNNING',mode='train')))
                with self.assertRaisesRegex(ValueError,'real COMPLETE'):review.sealed_inputs(run)

    def fixture(self,folder):
        run=Path(folder)/'actual_run';run.mkdir()
        history=Path(folder)/'history';history.mkdir()
        runtime=dict(source_git_commit=review.prior.HEAD,local_contract={review.prior.GAIN_KEY:review.prior.GAIN})
        updates=[]
        for number in range(9,13):
            # Deliberately arbitrary display names: only metadata counts+run bind.
            cp=history/f'not_a_predicted_CP_{number}_g{review.prior.HEAD[:12]}.pt'
            cp.write_bytes(f'synthetic actor checkpoint {number}'.encode())
            counts=dict(local_ppo_updates=number,local_policy_decisions=512*number)
            metadata=dict(schema='wlr50_clean.frozen_prior_rr_capture_checkpoint.v2',
                runtime_contract=runtime,source_run=str(run),checkpoint=str(cp),checkpoint_sha256=review.common.sha(cp),
                counts=counts,rollout_empty=True,save_load_round_trip=True,front_FL_assist=True,rear_task_assists=False)
            cp.with_name(cp.stem+'_manifest.json').write_text(json.dumps(metadata))
            updates.append(dict(counts=copy.deepcopy(counts)))
        return run,history,runtime,updates

    def test_real_update_counts_select_ordered_unique_sidecars(self):
        with tempfile.TemporaryDirectory() as folder:
            run,history,runtime,updates=self.fixture(folder)
            records=review.find_update_records(run,runtime,updates,history)
            self.assertEqual([x[2]['counts']['local_ppo_updates'] for x in records],[9,10,11,12])
            self.assertTrue(all('not_a_predicted_CP' in x[0].name for x in records))
            first=records[0]
            duplicate=history/f'duplicate_g{review.prior.HEAD[:12]}_manifest.json'
            duplicate.write_text(first[1].read_text())
            with self.assertRaisesRegex(ValueError,'exactly one'):
                review.find_update_records(run,runtime,updates,history)

    def test_checkpoint_tamper_and_missing_actual_update_rejected(self):
        for mutation in ('tamper','wrong_run','wrong_counts'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as folder:
                run,history,runtime,updates=self.fixture(folder)
                records=review.find_update_records(run,runtime,updates,history);cp,path,metadata=records[1]
                if mutation=='tamper':cp.write_bytes(b'modified after seal')
                else:
                    if mutation=='wrong_run':metadata['source_run']=str(run/'other')
                    else:metadata['counts']['local_policy_decisions']-=1
                    path.write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):review.find_update_records(run,runtime,updates,history)

    def test_storage_indices_are_local_without_resetting_episode_history(self):
        samples=[dict(index=i,episode=2 if i>=400 else 1,tick=8*i,marker=i) for i in range(2048)]
        for update in range(4):
            rows=review.rollout_rows(samples,update)
            self.assertEqual([r['index'] for r in rows],list(range(512)))
            self.assertEqual(rows[0]['marker'],update*512)
            self.assertEqual(rows[-1]['marker'],(update+1)*512-1)
        self.assertEqual(review.rollout_rows(samples,1)[0]['episode'],2)
        self.assertEqual(samples[512]['index'],512)

    def physical_rows(self,n=1024):
        def metrics(tick,top):
            return dict(tick=tick,current_top_contact=top,current_top_bearing=top,free_air=not top,
                within_top_xy=True,current_attempt_capture_eligible=True,ground_contact=False,
                placed=top,crossed=True,gap_m=0. if top else .025)
        return [dict(index=i,episode=1,tick=(i+1)*8,phase='P09' if i<512 else 'P10',
            before=metrics(i*8,i-1 in (100,512)),after=metrics((i+1)*8,i in (100,512)),
            local_success=False,hold_s=.1 if i in (100,512) else 0.,local_reward={'terminated':False})
            for i in range(n)]

    def test_top_reacquisition_memory_survives_update_boundary(self):
        samples=self.physical_rows()
        slices=[review.rollout_rows(samples,i) for i in range(2)]
        whole,windows=review.cohort_windows(samples,slices)
        self.assertEqual(len(whole['all_on_policy']),1024)
        self.assertNotIn('all512',whole)
        self.assertEqual([r['index'] for r in whole['first_TOP_transition']],[100])
        self.assertEqual([r['index'] for r in whole['TOP_reacquisition']],[512])
        self.assertEqual(windows[1]['first_TOP_transition'],[])
        self.assertEqual([r['index'] for r in windows[1]['TOP_reacquisition']],[0])
        self.assertIs(windows[1]['TOP_reacquisition'][0],slices[1][0])
        self.assertEqual(len(windows[1]['all512']),512)

    def test_live_episode_phase_change_and_final_budget_are_not_done(self):
        samples=self.physical_rows();accounting=[{'absolute_update':9},{'absolute_update':10}]
        boundaries=review.rollout_boundaries(samples,accounting)
        self.assertEqual(boundaries[0]['relation'],'same_live_episode_continues_with_next_collector')
        self.assertEqual(boundaries[1]['relation'],'nonterminal_final_budget_tail')
        self.assertTrue(all(not b['actual_terminal'] and b['bootstrap_required_by_nonterminal'] for b in boundaries))
        self.assertTrue(all(not b['bootstrap_recurrence_independently_recomputed'] for b in boundaries))
        # A true terminal and real next-prefix episode are distinct from updates.
        samples[511]['local_reward']['terminated']=True
        for row in samples[512:]:row['episode']=2
        self.assertEqual(review.rollout_boundaries(samples,accounting)[0]['relation'],'new_real_prefix_after_actual_terminal')
        samples[-1]['local_reward']['terminated']=True
        self.assertEqual(review.rollout_boundaries(samples,accounting)[1]['relation'],'actual_terminal_at_budget_boundary')
        samples[511]['local_reward']['terminated']=False
        with self.assertRaisesRegex(ValueError,'unexpectedly reset'):
            review.rollout_boundaries(samples,accounting)

    def test_actual_report_expression_uses_each_successive_collector(self):
        tree=ast.parse(Path(review.__file__).read_text())
        analyze=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='analyze')
        expression=next(node.value for node in analyze.body if isinstance(node,ast.Assign) and
            any(isinstance(target,ast.Name) and target.id=='reports' for target in node.targets))
        state=dict(run='synthetic',slices=[['row9'],['row10'],['row11'],['row12']],
            records=['CP8','CP9','CP10','CP11','CP12'],accounting=[9,10,11,12],checks=[{}]*4,window_groups=[{}]*4,
            tensor_pair=lambda run,rows,pair,account,checks,groups:(pair,rows,account))
        actual=eval(compile(ast.Expression(expression),'<actual report pairing>','eval'),state)
        self.assertEqual([item[0] for item in actual],[['CP8','CP9'],['CP9','CP10'],['CP10','CP11'],['CP11','CP12']])

    def test_dry_tensor_path_pairs_each_update_and_normalizes_per512(self):
        source=Path(review.__file__).read_text();ast.parse(source)
        self.assertIn('records[i:i+2]',source)
        self.assertIn('per512_normalized_GAE',source)
        self.assertIn('storage[\'advantages\'].reshape(512)',source)
        self.assertNotIn('reshape(2048)',source)
        self.assertNotIn('run_manifest.json\').write',source)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('pxr',sys.modules)


if __name__=='__main__':unittest.main()

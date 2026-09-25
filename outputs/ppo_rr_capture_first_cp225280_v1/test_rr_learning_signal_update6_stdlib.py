"""Prepared analyzer guards/statistics only: never import tensors or run a model."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import analyze_rr_learning_signal_update6 as m


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def sealed_fixture(root):
    run = root/'run'
    run.mkdir()
    runtime = dict(source_git_commit=m.HEAD,
        experiment_id='ppo_rr_capture_first_cp225280_v1',local_contract=dict(
            source_tracking_owner_revision=m.REVISION,observation_dimension=448,
            rollout_length=512,capture_source_dispatch='rr_local_defer_p09_late_and_new_p12_until_terminal_v2'))
    checkpoints = []
    for index, values in enumerate(((2560,5,100,512,1,0),(3072,6,120,1024,2,0))):
        path = root/f'checkpoint_{index}.pt'
        path.write_bytes(f'synthetic sealed bytes {index}'.encode())
        sidecar = path.with_name(path.stem+'_manifest.json')
        counts = dict(zip(('local_policy_decisions','local_ppo_updates','local_optimizer_steps',
            'task_v2_policy_decisions','task_v2_ppo_updates','auxiliary_updates'),values))
        meta = dict(schema='wlr50_clean.frozen_prior_rr_capture_checkpoint.v2',runtime_contract=runtime,
            checkpoint_sha256=m.common.sha(path),rollout_empty=True,save_load_round_trip=True,
            runner_config=dict(actor=dict(observation_layout=m.OBSERVATION_LAYOUT)),counts=counts)
        write_json(sidecar,meta)
        checkpoints.append((path,sidecar,meta))
    write_json(run/'run_manifest.started.json',dict(checkpoint=str(checkpoints[0][0]),runtime_contract=runtime))
    manifest = dict(lifecycle='COMPLETE',mode='train',result=dict(checkpoint=str(checkpoints[1][0]),
        checkpoint_sha256=m.common.sha(checkpoints[1][0]),manifest_sha256=m.common.sha(checkpoints[1][1])))
    write_json(run/'run_manifest.json',manifest)
    return run, checkpoints, manifest


def cohort_rows():
    rows = []
    for index in range(512):
        before = dict(free_air=True,within_top_xy=True,current_attempt_capture_eligible=True,
            gap_m=.03,actual_rr_hip_knee_deg=[0.,-30.])
        after = dict(before,current_top_contact=False,current_top_bearing=False,
            ground_contact=False,placed=False,crossed=True,gap_m=.02,
            actual_rr_hip_knee_deg=[-1.,-29.])
        rows.append(dict(index=index,episode=1 if index<41 else 2,tick=7992+8*index,
            local_success=False,hold_s=0.,before=before,after=after))
    rows[40].update(local_success=True,hold_s=.5)
    rows[40]['after'].update(current_top_contact=True,current_top_bearing=True,placed=True)
    return rows


def tracking_row():
    return dict(index=0,episode=1,tick=7992,phase='P09',source_nominal=[0.]*12,
        native_nominal=[0.]*12,controller_bias=[0.]*12,
        after=dict(actual_rr_hip_knee_deg=[1.,-58.]),
        source_deferred=dict(layers=[dict(pending_event_tick=648,pending_event_consumed=False,
            deferred_tracking_source='previous_sample_before_unconsumed_late',
            deferred_source_tracking_servo_names=[])]),
        tracking=dict(tracking_servo_names=[],mapper_pre_state=dict(tracking_active=[False]*8),
            previous_ack_physics_tick=1,reference_semantics='previous_ACK_filtered_REQUEST_not_effective_or_realized_residual'))


class Update6Tests(unittest.TestCase):
    def test_stdlib_import_and_explicit_exit_guard(self):
        with self.assertRaisesRegex(ValueError,'Isaac exit'):
            m.analyze(Path('never_read_this_path'))
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('pxr',sys.modules)
        self.assertFalse(any(k.startswith('wlr50_clean') for k in sys.modules))

    def test_unsealed_update6_rejected_even_with_exit_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError,'not sealed COMPLETE'):
                m.analyze(Path(directory),isaac_stopped=True)
        self.assertNotIn('torch',sys.modules)

    def test_exact_sealed_metadata_allowed_without_tensor_load(self):
        with tempfile.TemporaryDirectory() as directory:
            run, records, _ = sealed_fixture(Path(directory))
            with patch.object(m,'BEFORE_SHA',m.common.sha(records[0][0])):
                actual, offset = m.sealed_inputs(run)
            self.assertEqual(offset,5)
            self.assertEqual(actual[1][2]['counts']['local_ppo_updates'],6)

    def test_wrong_head_or_old_tracking_revision_rejected(self):
        for field,value in (('source_git_commit','5f8487b'),('source_tracking_owner_revision',None)):
            with tempfile.TemporaryDirectory() as directory:
                run, _, _ = sealed_fixture(Path(directory))
                started = json.loads((run/'run_manifest.started.json').read_text())
                target = started['runtime_contract'] if field=='source_git_commit' else started['runtime_contract']['local_contract']
                target[field] = value
                write_json(run/'run_manifest.started.json',started)
                with self.assertRaisesRegex(ValueError,'exact1e'):
                    m.sealed_inputs(run)

    def test_wrong_source_checkpoint_or_incomplete_update_counts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run, records, manifest = sealed_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError,'exact rebound'):
                m.sealed_inputs(run)
            records[1][2]['counts']['local_policy_decisions'] = 3071
            write_json(records[1][1],records[1][2])
            manifest['result']['manifest_sha256'] = m.common.sha(records[1][1])
            write_json(run/'run_manifest.json',manifest)
            with patch.object(m,'BEFORE_SHA',m.common.sha(records[0][0])):
                with self.assertRaisesRegex(ValueError,'update5 to6'):
                    m.sealed_inputs(run)

    def test_source_tensor_bytes_tamper_rejected_without_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            run, records, _ = sealed_fixture(Path(directory))
            records[1][0].write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'metadata/hash'):
                m.sealed_inputs(run)

    def test_success41_partition_is_not_41_positive_labels(self):
        result = m.success_cohorts(cohort_rows())
        self.assertEqual(len(result['first_successful_episode41']),41)
        self.assertEqual(len(result['other471']),471)
        self.assertEqual(sum(r['local_success'] for r in result['first_successful_episode41']),1)

    def test_success_cohort_rejects_wrong_count_and_revoked_attempt(self):
        for mutation in ('count','ground','qualification','hold'):
            rows=cohort_rows()
            if mutation=='count':rows.pop()
            elif mutation=='ground':rows[40]['after']['ground_contact']=True
            elif mutation=='qualification':rows[40]['after']['current_attempt_capture_eligible']=False
            else:rows[40]['hold_s']=.49
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                m.success_cohorts(rows)

    def test_tracking_carrier_empty_not_confused_with_legal_P10_tracking(self):
        first=tracking_row(); second=copy.deepcopy(first)
        second.update(index=1,tick=8264,phase='P10')
        second['tracking']['tracking_servo_names']=['rear_right_knee']
        second['source_nominal'][7]=-29.3
        second['controller_bias'][7]=.75
        result=m.tracking_summary([first,second])
        self.assertFalse(result['old_bug_report_consulted'])
        self.assertEqual(result['pending_with_actual_RR_tracking'],1)
        self.assertEqual(result['transitions'][1]['inherited_carrier_tracking'],[[]])
        self.assertEqual(result['transitions'][1]['actual_tracking'],['rear_right_knee'])

    def test_missing_or_old_tracking_receipt_not_inferred_as_fixed(self):
        for field,value in (('deferred_tracking_source','late_group'),('deferred_source_tracking_servo_names',None)):
            row=tracking_row(); row['source_deferred']['layers'][0][field]=value
            with self.assertRaisesRegex(ValueError,'inheritance evidence'):
                m.tracking_summary([row])

    def test_headroom_and_physical_direction_do_not_force_advantage_sign(self):
        row=cohort_rows()[0]
        row.update(raw_gae=-2.,final=[0.]*7+[-58.]+[0.]*4,
            headroom=dict(clipped_servo_indices=[7],requested_policy_residual_full12=[0.]*7+[-5.]+[0.]*4,
                effective_policy_residual_full12=[0.]*7+[-3.]+[0.]*4))
        result=m.knee_response([row])['headroom_clipped']
        self.assertEqual(result['negative_raw_gae'],1)
        self.assertEqual(result['request_minus_effective_deg']['mean'],-2.)
        self.assertEqual(result['actual_knee_delta_deg']['mean'],1.)
        self.assertGreater(result['gap_descent_mm']['mean'],0.)

    def test_raw_sample_vs_issued_and_prefix_credit_checked(self):
        row=dict(kind='activated_on_policy',PPO_credit=1,policy_request=dict(selected_raw_full12=[0.]*12),
            step_info=dict(raw_policy_action_full12=[0.]*12,actuator_target_effect_audit=dict(
                all12_policy_channels_unmodified_at_actuator=True,phase_mask_full12=[1]*12)))
        self.assertEqual(m.validate_execution_rows([row],1)['rows'],1)
        row['step_info']['raw_policy_action_full12'][7]=1.
        with self.assertRaisesRegex(ValueError,'sample/issued'):
            m.validate_execution_rows([row],1)
        with self.assertRaisesRegex(ValueError,'prefix'):
            m.validate_execution_rows([dict(kind='frozen_prior_prefix',PPO_credit=1)],0)

    def test_actual_minibatch_old_logp_and_five_epoch_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            run=Path(directory); (run/'rollouts').mkdir()
            rows=[dict(old_logp=float(i)/100) for i in range(512)]
            batches=[]
            for batch in range(20):
                ids=list(range((batch%4)*128,(batch%4+1)*128))
                batches.append(dict(rollout_flat_indices=[[i] for i in ids],
                    old_log_probability=[rows[i]['old_logp'] for i in ids]))
            path=run/'rollouts/likelihood_0006.json'; write_json(path,dict(minibatches=batches))
            result=m.validate_minibatch_old_logp(run,rows,dict(optimizer_steps=20))
            self.assertEqual(result['actual_row_exposures'],2560)
            batches[0]['old_log_probability'][0]=99.
            write_json(path,dict(minibatches=batches))
            with self.assertRaisesRegex(ValueError,'old_logp differs'):
                m.validate_minibatch_old_logp(run,rows,dict(optimizer_steps=20))


if __name__=='__main__':
    unittest.main(verbosity=2)

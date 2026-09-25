"""Bounded metadata tests only: no torch, model, simulation, or production edits.

The historical source-lineage call is explicitly mocked: these tests do NOT
claim a real checkpoint load or tensor/Adam identity round-trip.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location('front_preservation_candidate',
    HERE / 'tree/src/wlr50_clean/ppo/semantic_front_preservation.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class FrontPreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='front-preservation-stdlib-')
        self.addCleanup(self.tmp.cleanup)
        self.dataset = Path(self.tmp.name) / 'dataset.json'
        self.dataset.write_text(json.dumps({'fixture': 'not physical training data', 'training_rows': [
            {'phase': 'P01', 'decision_index': 0}, {'phase': 'P05', 'decision_index': 1}]}), encoding='utf-8')
        self.replay = dict(schema=m.REPLAY_SCHEMA, version='cp225280_front_gaussian_kl_v1',
            dataset_path=str(self.dataset), dataset_sha256=m.file_sha(self.dataset),
            coefficient=1.0, minibatch_size=32)
        self.binding = m._source_binding(ROOT)
        self.source = json.loads(Path(self.binding['manifest']).read_text(encoding='utf-8'))
        latest = ROOT / 'outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000231168_manifest.json'
        self.contract = json.loads(latest.read_text(encoding='utf-8'))['runtime_contract']
        self.contract['source_git_commit'] = '1' * 40
        for name in ('semantic_front_preservation.py', 'semantic_front_replay.py'):
            self.contract['files'][m.CODE + name] = '2' * 64
        self.contract['runtime_content_sha256'] = m.digest(self.contract['files'])
        self.plan = m._make_plan(self.source, self.binding, self.contract, 'authorized independent continuation',
                                 self.replay, ROOT, verify_bytes=False)
        self.metadata = copy.deepcopy(self.source)
        self.metadata.update(runtime_contract=copy.deepcopy(self.contract),
            runner_config=copy.deepcopy(self.plan['target_runner_config']),
            checkpoint_output_routing=copy.deepcopy(self.plan['checkpoint_output_routing']))
        self.metadata[m.IDENTITY] = copy.deepcopy(self.plan)
        self.metadata = m.front_preservation_branch_counts(self.metadata)
        self.metadata[m.REPLAY_COUNTS_KEY] = m.expected_front_replay_counts(0)
        self.source_mock = patch.object(m, '_source_checkpoint', return_value=(self.source, self.binding)).start()
        self.lineage_mock = patch.object(m, '_validate_source_lineage', return_value=None).start()
        self.addCleanup(patch.stopall)

    def validate(self, metadata=None, contract=None):
        value = metadata or self.metadata
        return m.validate_front_preservation_lineage(value, contract or self.contract, m._branch(ROOT),
            checkpoint_output_routing=value['checkpoint_output_routing'])

    def descendant(self):
        candidate = copy.deepcopy(self.metadata)
        candidate.update(global_policy_decisions=225792, ppo_updates=1726, optimizer_steps=34520)
        candidate['actor_parameter_sha256'] = '4' * 64
        batches = [dict(global_minibatch_index=34500+i, dataset_indices=[0, 1]*16,
            source_decision_indices=[0, 1]*16, phase_counts={'P01': 16, 'P05': 16},
            gradient_consumed_once=True, coefficient=1.0, on_policy_samples_added=0,
            separate_optimizer_steps=0, extra_actor_forwards=1, extra_random_draws=0,
            kl_reference_to_current=.001, unclipped_replay_actor_gradient_norm=.01) for i in range(20)]
        report = dict(schema=m.REPLAY_SCHEMA, spec=copy.deepcopy(self.replay), minibatches=batches,
            actual_replay_row_exposures=640, actual_replay_phase_exposures={'P01': 320, 'P05': 320},
            on_policy_samples_added=0, separate_auxiliary_optimizer_steps=0, realtime_teacher_deployed=False)
        candidate['last_update'] = dict(global_policy_decisions=225792, ppo_update=1726,
            optimizer_steps=20, actor_parameter_sha256_after=candidate['actor_parameter_sha256'],
            optimizer_learning_rate=candidate['optimizer_learning_rate'], front_replay_regularization=report)
        candidate = m.front_preservation_branch_counts(candidate)
        candidate[m.REPLAY_COUNTS_KEY] = m.expected_front_replay_counts(1)
        return candidate

    def test_zero_credit_identity_keeps_original_full_manifest_and_history(self):
        self.validate()
        self.assertEqual(self.plan['source_manifest'], self.source)
        self.assertEqual(self.metadata['rear_policy_timing_branch'], self.source['rear_policy_timing_branch'])
        self.assertEqual(self.metadata[m.COUNTS_KEY], dict.fromkeys(m.COUNTERS, 0))
        self.assertFalse(self.plan['old_rollout_inherited'])
        self.assertEqual(self.plan['target_runner_config']['num_steps_per_env'], 512)

    def test_only_explicit_sibling_route(self):
        self.assertEqual(m.build_front_preservation_output_routing(self.metadata, self.contract, m._branch(ROOT)),
                         self.plan['checkpoint_output_routing'])
        with self.assertRaises(ValueError):
            m.build_front_preservation_output_routing(self.metadata, self.contract, m._branch(ROOT, 'unrelated'))

    def test_constructor_options_are_explicit_439_512(self):
        self.assertEqual(m.front_preservation_collection_options(self.metadata, experiment_id=m.EXPERIMENT),
                         {'collection_profile': m.COLLECTION_512})
        self.metadata['runner_config']['num_steps_per_env'] = 128
        with self.assertRaises(ValueError):
            m.front_preservation_collection_options(self.metadata, experiment_id=m.EXPERIMENT)

    def test_reject_modified_N_mapper_config_or_physics(self):
        for path in ('src/wlr50_clean/ppo/semantic_residual_adapter.py',
                     'configs/ppo_rr_rl_timing_policy_learning_v1/execution_profile.yaml',
                     'unreviewed/robot_asset.usd'):
            candidate = copy.deepcopy(self.contract)
            candidate['files'][path] = '3' * 64
            candidate['runtime_content_sha256'] = m.digest(candidate['files'])
            with self.assertRaises(ValueError):
                m._runtime_delta(self.source['runtime_contract'], candidate, ROOT, verify_bytes=False)

    def test_reject_unreviewed_reward(self):
        candidate = copy.deepcopy(self.contract)
        candidate['files'][m.CODE + 'semantic_reward.py'] = '3' * 64
        candidate['runtime_content_sha256'] = m.digest(candidate['files'])
        with self.assertRaises(ValueError):
            m._runtime_delta(self.source['runtime_contract'], candidate, ROOT, verify_bytes=False)

    def test_reject_changed_source_snapshot_or_historical_receipt(self):
        self.metadata[m.IDENTITY]['source_manifest']['global_policy_decisions'] += 1
        with self.assertRaises(ValueError): self.validate()
        self.metadata[m.IDENTITY] = copy.deepcopy(self.plan)
        self.metadata['rear_owner_recovery_migration']['reason'] = 'tampered'
        with self.assertRaises(ValueError): self.validate()

    def test_reject_borrowed_65a_AUX_or512_lineage(self):
        for key in m.FORBIDDEN_LATER_LINEAGE:
            candidate = copy.deepcopy(self.metadata)
            candidate[key] = {}
            with self.assertRaises(ValueError): self.validate(candidate)

    def test_replay_binding_is_mandatory_and_immutable(self):
        self.metadata[m.IDENTITY]['front_replay_spec']['coefficient'] = .1
        with self.assertRaises(ValueError): self.validate()
        self.metadata[m.IDENTITY] = copy.deepcopy(self.plan)
        self.dataset.write_text('changed fixture', encoding='utf-8')
        with self.assertRaises(ValueError): self.validate()

    def test_zero_update_cannot_reset_Adam_LR_or_weights(self):
        for key in ('actor_parameter_sha256', 'optimizer_state_sha256', 'optimizer_learning_rate'):
            candidate = copy.deepcopy(self.metadata)
            candidate[key] = 'wrong' if key.endswith('sha256') else 3e-5
            with self.assertRaises(ValueError): self.validate(candidate)

    def test_valid_complete512_descendant_and_reject_128_or_extra_Adam(self):
        candidate = self.descendant()
        self.validate(candidate)
        for key, bad in (('global_policy_decisions', 225408), ('optimizer_steps', 34521)):
            invalid = copy.deepcopy(candidate)
            invalid[key] = bad
            with self.assertRaises(ValueError): self.validate(invalid)

    def test_new_counter_origin_does_not_borrow_latest_counts(self):
        self.metadata[m.IDENTITY]['counter_origin']['global_policy_decisions'] = 231168
        with self.assertRaises(ValueError): self.validate()

    def test_replay_counts_cannot_add_PPO_credit_or_extra_optimizer(self):
        for key in ('on_policy_samples_added', 'separate_auxiliary_optimizer_steps', 'replay_row_exposures'):
            candidate = copy.deepcopy(self.metadata)
            candidate[m.REPLAY_COUNTS_KEY][key] = 1
            with self.assertRaises(ValueError): self.validate(candidate)

    def test_report_cannot_replace_actual_minibatch_evidence_with_expected_totals(self):
        for key, bad in [('gradient_consumed_once', False), ('phase_counts', {'P12': 32}),
                         ('source_decision_indices', [999]*32), ('extra_random_draws', 1)]:
            candidate = self.descendant()
            candidate['last_update']['front_replay_regularization']['minibatches'][0][key] = bad
            with self.assertRaises(ValueError): self.validate(candidate)
        candidate = self.descendant()
        candidate['last_update']['front_replay_regularization']['minibatches'].pop()
        with self.assertRaises(ValueError): self.validate(candidate)

    def test_no_torch_or_simulation_imported(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('pxr', sys.modules)
        self.assertNotIn('isaacsim', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)

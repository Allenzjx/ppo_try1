"""CPU-only synthetic logging tests: never imports/launches Isaac or trains."""
import copy
import ast
import json
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction_probe as old
import direction_probe_preaction_candidate as candidate


def fixture():
    schema = json.loads((candidate.CONFIG/'observation_schema.json').read_text())
    offsets = candidate.schema_offsets(schema)
    history = {name: tuple((i+1)*.01 for i in range(12)) for name in candidate.HISTORY_NAMES}
    obs = [0.] * 372
    obs[4] = 1.
    for name, values in history.items():
        lo, hi, scale = offsets[name]
        scales = [scale]*12 if isinstance(scale, (int, float)) else scale
        obs[lo:hi] = [max(-schema['clip'], min(schema['clip'], x/s)) for x, s in zip(values, scales)]
    raw = tuple(candidate.f32(.005*(i+1)) for i in range(12))
    caps = (18., 24., 12., 112., 12., 18., 12., 18., 1.2, 1.2, 1., .6)
    lo, hi, _ = offsets['previous_raw_full12']
    request = dict(mode='deterministic_conditional_mean', policy_version=candidate.POLICY,
        sampling_draws=0, extra_model_forwards=0, extra_random_draws=0,
        stage_index=4,
        selected_raw_full12=raw, conditional_mean_full12=raw,
        previous_raw_from_current_observation_full12=tuple(candidate.f32(x) for x in obs[lo:hi]),
        history_center_full12=tuple(candidate.f32(x) for x in obs[lo:hi]), current_cap_full12=caps)
    lo, hi, scales = offsets['previous_residual_full12']
    request['previous_filtered_request_full12'] = tuple(candidate.f32(candidate.f32(x)*candidate.f32(s))
                                                       for x, s in zip(obs[lo:hi], scales))
    return dict(decision=17, tick=2800, sim_time=2800/120, phase='P05', observation=obs,
        schema=schema, history=history, request=request, baseline=raw, injected=raw,
        caps=caps, phase_mask=(1,)*12, runtime_mask=(1,)*12, safety_mask=(1,)*12,
        intervention=None, nominal=(22.8, -12.15)+(0.,)*6+(.3,)*4,
        previous_ack={'tick': 2799, 'not_final_action': 'previous only'})


class ReceiptTests(unittest.TestCase):
    def test_true_pre_input_snapshot_and_no_mutation(self):
        args = fixture()
        original = copy.deepcopy(args)
        receipt = candidate.pre_action_receipt(**args)
        self.assertEqual(args, original)
        self.assertEqual(len(receipt['actor_input_float32_372']), 372)
        self.assertEqual(receipt['actual_live_history'], args['history'])
        self.assertEqual(receipt['manual_override_selector_full12'], (0,)*12)
        self.assertEqual(receipt['manual_raw_delta_full12'], (0.,)*12)
        args['history']['previous_raw_full12'] = (9.,)*12
        self.assertNotEqual(receipt['actual_live_history'], args['history'])
        json.dumps(receipt, allow_nan=False)

    def test_history_mismatch_is_rejected(self):
        args = fixture()
        args['observation'][195] += .1
        with self.assertRaisesRegex(ValueError, 'live history'):
            candidate.pre_action_receipt(**args)

    def test_all_observation_values_required_and_finite(self):
        for bad in ([0.]*371, [0.]*371+[float('nan')]):
            args = fixture(); args['observation'] = bad
            with self.assertRaises(ValueError): candidate.pre_action_receipt(**args)

    def test_policy_mean_mismatch_is_rejected(self):
        args = fixture(); args['request']['conditional_mean_full12'] = (1.,)*12
        with self.assertRaisesRegex(ValueError, 'conditional mean'):
            candidate.pre_action_receipt(**args)

    def test_phase_and_filtered_history_binding(self):
        for key, value in [('stage_index', 5), ('previous_filtered_request_full12', (0.,)*12)]:
            args = fixture(); args['request'][key] = value
            with self.assertRaises(ValueError): candidate.pre_action_receipt(**args)

    def test_stochastic_or_extra_forward_is_rejected(self):
        for key, value in [('sampling_draws', 1), ('extra_model_forwards', 1),
                           ('extra_random_draws', 1), ('policy_version', 'old')]:
            args = fixture(); args['request'][key] = value
            with self.assertRaises(ValueError): candidate.pre_action_receipt(**args)

    def test_foreign_channel_intervention_is_rejected(self):
        args = fixture(); raw = list(args['injected']); raw[4] += .1; args['injected'] = raw
        with self.assertRaisesRegex(ValueError, 'unselected channel'):
            candidate.pre_action_receipt(**args)

    def test_mask_roles_and_units_do_not_erase_nominal(self):
        args = fixture(); args['phase_mask'] = (0,)*12
        receipt = candidate.pre_action_receipt(**args)
        self.assertEqual(receipt['pre_source_nominal_full12'], args['nominal'])
        self.assertEqual(receipt['combined_residual_permission_mask_full12'], (0.,)*12)
        self.assertNotEqual(receipt['candidate_physical_before_permission_and_slew_full12'], (0.,)*12)
        self.assertEqual(receipt['physical_units_full12'], ('deg',)*8+('rad/s',)*4)
        self.assertIn('not final targets', receipt['actual_execution_source'])

    def test_caps_or_masks_mismatch_is_rejected(self):
        for key, value in [('caps', (1.,)*12), ('runtime_mask', (.5,)*12)]:
            args = fixture(); args[key] = value
            with self.assertRaises(ValueError): candidate.pre_action_receipt(**args)

    def test_unchanged_finite_implementation_amplitude_and_release(self):
        self.assertIs(candidate.FiniteDirection, old.FiniteDirection)
        self.assertIs(candidate.trigger, old.trigger)
        args = fixture(); probe = candidate.FiniteDirection('FL_minus3')
        anchor = tuple(c*math.tanh(x) for c, x in zip(args['caps'], args['baseline']))
        probe.start(anchor)
        ev = {'history': {'placed': dict(FL=False, FR=False, RR=False, RL=False)}}
        records = []
        for _ in range(118):
            raw, intervention = probe.apply(args['baseline'], args['caps'], 'P05', ev)
            args.update(injected=raw, intervention=intervention)
            records.append(candidate.pre_action_receipt(**args))
        self.assertAlmostEqual(records[11]['candidate_physical_after_intervention_before_permission_and_slew_full12'][0], anchor[0]-3., places=12)
        self.assertEqual(probe.release_age, 75)
        self.assertTrue(probe.complete)
        self.assertEqual(records[-1]['manually_selected_raw_full12'], args['baseline'])
        self.assertTrue(all(r['manually_selected_raw_full12'][1:] == args['baseline'][1:] for r in records))
        self.assertTrue(all(r['new_PPO_decisions'] == r['new_PPO_updates'] == 0 for r in records))

    def test_intervened_state_mean_is_not_untreated_trajectory(self):
        receipt = candidate.pre_action_receipt(**fixture())
        self.assertIn('not an untreated deterministic trajectory', receipt['counterfactual_scope'])


class BindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cp = candidate.OUT/'checkpoints/history/checkpoint_step_000187904.pt'
        cls.metadata = json.loads(cls.cp.with_name(cls.cp.stem+'_manifest.json').read_text())

    def test_real_sealed_metadata_interface(self):
        candidate.validate_checkpoint_metadata(self.metadata, self.cp, self.metadata['runtime_contract'])

    def test_wrong_experiment_or_layout_or_untrained_branch_rejected(self):
        for target, key, value in [('runtime_contract', 'experiment_id', 'fl_capture_quality_v1'),
                                  ('policy_contract', 'observation_layout', 'wrong'),
                                  ('task_conditioned_hip_wheel_branch_counts', 'ppo_updates', 0)]:
            metadata = copy.deepcopy(self.metadata); metadata[target][key] = value
            with self.assertRaises(ValueError):
                candidate.validate_checkpoint_metadata(metadata, self.cp, metadata['runtime_contract'])

    def test_implicit_archive_or_migration_is_rejected(self):
        current = copy.deepcopy(self.metadata['runtime_contract']); current['source_git_commit'] = 'f'*40
        with self.assertRaisesRegex(ValueError, 'exact current runtime'):
            candidate.validate_checkpoint_metadata(self.metadata, self.cp, current)


class ExceptionSealTests(unittest.TestCase):
    def test_cached_summary_and_actual_observed_endpoint_survive_to_manifest(self):
        class Recorder:
            _last_frame = SimpleNamespace(physics_tick=5046, sim_time_s=42.05, state_id='P09')
            calls = 0
            def summary(self):
                self.calls += 1
                if self.calls > 1: raise FileExistsError('exclusive CSV already exists')
                return {'observed_physics_ticks': 5046,
                        'physical_task_evaluation': {'termination_reason': 'TASK_FAILURE_BODY_COLLISION'}}
        physical, manifest = Recorder(), {}
        args = dict(physical=physical, decision=630, receipt_hash='pre-hash', start_tick=5040,
                    last_core_frame_tick=5045, exception=RuntimeError('original step error'))
        row = candidate.seal_step_exception(manifest, **args)
        repeated = candidate.seal_step_exception(manifest, **args)
        self.assertEqual(physical.calls, 1)
        self.assertIs(row['physical_summary'], manifest['physical_summary'])
        self.assertIs(repeated['physical_summary'], row['physical_summary'])
        self.assertEqual(manifest['endpoint_tick'], 5046)
        self.assertEqual(manifest['last_core_frame_tick'], 5045)
        self.assertEqual(manifest['endpoint_sim_time_s'], 42.05)
        self.assertFalse(manifest['environment_step_returned'])
        self.assertNotIn('reward', row)
        self.assertNotIn('step_info', row)
        json.dumps(manifest, allow_nan=False)

    def test_summary_failure_does_not_replace_original_step_error_or_retry(self):
        class Recorder:
            _last_frame = SimpleNamespace(physics_tick=8, sim_time_s=8/120, state_id='P02')
            calls = 0
            def summary(self):
                self.calls += 1
                raise FileExistsError('summary write failed')
        physical, manifest = Recorder(), {}
        original = RuntimeError('original physical step error')
        args = dict(physical=physical, decision=0, receipt_hash='pre', start_tick=0,
                    last_core_frame_tick=7, exception=original)
        row = candidate.seal_step_exception(manifest, **args)
        candidate.seal_step_exception(manifest, **args)
        self.assertEqual(physical.calls, 1)
        self.assertEqual(row['exception'], repr(original))
        self.assertIn('summary write failed', manifest['physical_summary_error'])
        self.assertIsNone(manifest['physical_summary'])
        self.assertEqual(manifest['endpoint_tick'], 8)

    def test_windows_preload_order_is_explicit_before_AppLauncher(self):
        tree = ast.parse(Path(candidate.__file__).read_text())
        main = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'main')
        torch_line = next(x.lineno for x in ast.walk(main) if isinstance(x, ast.Import)
                          and any(y.name == 'torch' for y in x.names))
        tensor_line = next(x.lineno for x in ast.walk(main) if isinstance(x, ast.ImportFrom)
                           and x.module == 'tensordict')
        app_line = next(x.lineno for x in ast.walk(main) if isinstance(x, ast.ImportFrom)
                        and x.module == 'isaaclab.app')
        self.assertLess(torch_line, tensor_line)
        self.assertLess(tensor_line, app_line)


if __name__ == '__main__':
    unittest.main()

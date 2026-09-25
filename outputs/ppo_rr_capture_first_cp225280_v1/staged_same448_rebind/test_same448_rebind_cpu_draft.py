"""Cold-only real-tensor rehearsal. NOT run during Isaac; never publishes a CP.

CPU actor/critic/Adam tensors, real saved state, and the route's strict load.
RNG verification deliberately includes saved CUDA RNG if present. This is not
an Isaac run or a PPO update. A temporary fixture changes ONLY runner device to
CPU and its self-consistent hashes; it must never be treated as a deployable CP.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from test_same448_rebind_stdlib import HERE, ROOT, NAME, staged_namespace, runtime_pair

AUTHORIZED = '--isaac-stopped' in sys.argv
if AUTHORIZED:
    sys.argv.remove('--isaac-stopped')


@unittest.skipUnless(AUTHORIZED, 'requires explicit --isaac-stopped cold boundary')
class Same448RealTensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Imports intentionally delayed until the explicit safe-boundary flag.
        sys.path.insert(0, str(ROOT/'src'))
        import torch
        from wlr50_clean.ppo import semantic_rr_capture_local as route
        from wlr50_clean.ppo.semantic_training import state_hash
        cls.torch, cls.route, cls.state_hash = torch, route, staticmethod(state_hash)
        pointer = json.loads((HERE.parent/'checkpoints/checkpoint_last_pointer.json').read_text())
        metadata = json.loads(Path(pointer['manifest']).read_text())
        if metadata['runtime_contract']['source_git_commit'] != staged_namespace()['SAME448_REBIND_SOURCE_HEAD']:
            receipt = metadata['local_control_rebinds'][-1]
            metadata = json.loads(Path(receipt['source_manifest']).read_text())
        cls.source_metadata = metadata
        cls.source = Path(metadata['checkpoint'])
        cls.source_payload = torch.load(cls.source, map_location='cpu', weights_only=False)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='same448-rebind-test-')
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'CPU_TEST_FIXTURE_NOT_A_PUBLISHED_CHECKPOINT.pt'
        self.runner = self.route.make_runner('cpu', 1001)
        self.data = copy.deepcopy(self.source_payload)
        # A CPU-only tensor fixture keeps all learned values/moments/step/LR.
        self.data['infos']['runner_config']['device'] = 'cpu'
        self.torch.save(self.data, self.path)
        self.meta = dict(copy.deepcopy(self.data['infos']), checkpoint=str(self.path),
            checkpoint_sha256=self.route.sha(self.path), save_load_round_trip=True)
        self.sidecar = self.path.with_name(self.path.stem+'_manifest.json')
        self.sidecar.write_text(json.dumps(self.meta, sort_keys=True), encoding='utf-8')
        self.ns = staged_namespace()
        self.ns.update(sha=self.route.sha, settings=self.route.settings, load=self.route.load)
        self.old, self.new = runtime_pair()
        self.assertEqual(self.old, self.meta['runtime_contract'])
        self.before_source = self.route.sha(self.source)
        self.optimizer_id = id(self.runner.alg.optimizer)

    def invoke(self, **kwargs):
        arguments = dict(checkpoint_sha256=self.route.sha(self.path),
                         manifest_sha256=self.route.sha(self.sidecar))
        arguments.update(kwargs)
        return self.ns['rebind_checkpoint'](self.runner, self.path, self.new, **arguments)

    def test_real_state_adam_lr_rng_and_lineage_unchanged(self):
        prior, counts = self.invoke()
        self.assertEqual(id(self.runner.alg.optimizer), self.optimizer_id)
        self.assertEqual(counts, self.meta['counts'])
        self.assertEqual(prior, self.meta['prior'])
        self.assertEqual(self.runner.local_migration, self.meta.get('local_migration'))
        self.assertEqual(self.runner.alg.learning_rate, self.meta['learning_rate'])
        actual = self.runner.alg.save()
        for key, expected in self.meta['state_hashes'].items():
            self.assertEqual(self.state_hash(actual[key]), expected, key)
        self.assertEqual(self.runner.alg.storage.step, 0)
        self.assertIsNone(self.runner.alg.transition.actions)
        self.assertEqual(self.runner.local_control_rebinds[-1]['new_optimizer_steps'], 0)
        self.assertEqual(self.route.sha(self.source), self.before_source)

    def test_wrong_sealed_sha_rejected_before_load(self):
        with patch.dict(self.ns, load=lambda *a: self.fail('must not load rejected input')):
            with self.assertRaises(ValueError):
                self.invoke(checkpoint_sha256='0'*64)

    def test_incomplete_source_rollout_rejected(self):
        self.data['infos']['rollout_empty'] = False
        self.torch.save(self.data, self.path)
        self.meta.update(rollout_empty=False, checkpoint_sha256=self.route.sha(self.path))
        self.sidecar.write_text(json.dumps(self.meta), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.invoke()

    def test_mismatched_embedded_infos_rejected(self):
        self.meta['counts']['local_successes'] += 1
        self.sidecar.write_text(json.dumps(self.meta), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.invoke()

    def test_real_tensor_corruption_rejected(self):
        key = next(k for k,v in self.data['actor_state_dict'].items()
                   if self.torch.is_tensor(v) and v.is_floating_point() and v.numel())
        self.data['actor_state_dict'][key].view(-1)[0] += 0.01
        self.torch.save(self.data, self.path)
        self.meta['checkpoint_sha256'] = self.route.sha(self.path)
        self.sidecar.write_text(json.dumps(self.meta), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.invoke()

    def test_actual_adam_lr_inconsistency_rejected(self):
        self.data['optimizer_state_dict']['param_groups'][0]['lr'] *= 2.
        self.data['infos']['state_hashes']['optimizer_state_dict'] = self.state_hash(
            self.data['optimizer_state_dict'])
        self.torch.save(self.data, self.path)
        self.meta = dict(copy.deepcopy(self.data['infos']), checkpoint=str(self.path),
            checkpoint_sha256=self.route.sha(self.path), save_load_round_trip=True)
        self.sidecar.write_text(json.dumps(self.meta), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.invoke()


if __name__ == '__main__':
    unittest.main(verbosity=2)


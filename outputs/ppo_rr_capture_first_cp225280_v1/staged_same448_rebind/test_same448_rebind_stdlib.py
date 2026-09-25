"""Stdlib-only checks for the staged same448 patch; never import Torch/Isaac."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ROUTE = ROOT / 'src/wlr50_clean/ppo/semantic_rr_capture_local.py'
PATCH = HERE / 'route_rebind.apply_patch.txt'
NAME = 'ppo_rr_capture_first_cp225280_v1'
SCHEMA = 'wlr50_clean.frozen_prior_rr_capture_checkpoint.v2'


def staged_source():
    original = ROUTE.read_text(encoding='utf-8')
    lines = PATCH.read_text(encoding='utf-8').splitlines()
    chunks, chunk = [], None
    for line in lines:
        if line.startswith('@@'):
            if chunk is not None:
                chunks.append(chunk)
            chunk = []
        elif line.startswith('***'):
            continue
        elif chunk is not None and line[:1] in (' ', '+', '-'):
            chunk.append(line)
    if chunk:
        chunks.append(chunk)
    candidate = original
    for chunk in chunks:
        old = '\n'.join(s[1:] for s in chunk if s[:1] in (' ', '-')) + '\n'
        new = '\n'.join(s[1:] for s in chunk if s[:1] in (' ', '+')) + '\n'
        if candidate.count(old) != 1:
            raise AssertionError('staged hunk is not a unique match: ' + old[:160])
        candidate = candidate.replace(old, new, 1)
    ast.parse(candidate)
    return original, candidate


def staged_namespace():
    text = PATCH.read_text(encoding='utf-8')
    start = text.index("+SAME448_REBIND_SOURCE_HEAD = ")
    end = text.index("+def migrate_checkpoint(runner, path):", start)
    code = '\n'.join(line[1:] for line in text[start:end].splitlines() if line.startswith('+'))
    namespace = dict(__name__='staged_rebind', __package__='wlr50_clean.ppo',
        copy=copy, hashlib=hashlib, json=json, math=math, Path=Path, NAME=NAME, SCHEMA=SCHEMA)
    exec(compile(code, str(PATCH), 'exec'), namespace)
    return namespace


def runtime_pair():
    pointer = json.loads((HERE.parent/'checkpoints/checkpoint_last_pointer.json').read_text())
    metadata = json.loads(Path(pointer['manifest']).read_text())
    source = copy.deepcopy(metadata['runtime_contract'])
    if source['source_git_commit'] != '5f8487b76f27eb165a329a0b6f3096f54ebb4d48':
        # Fixtures remain tied to the actual archived pre-rebind runtime.
        receipt = metadata.get('local_control_rebinds', [])[-1]
        metadata = json.loads(Path(receipt['source_manifest']).read_text())
        source = copy.deepcopy(metadata['runtime_contract'])
    target = copy.deepcopy(source)
    target['source_git_commit'] = '1' * 40
    target['local_contract']['source_tracking_owner_revision'] = 'pending_source_tracking_inheritance_v1'
    for path in staged_namespace()['SAME448_REBIND_FILES']:
        target['files'][path] = hashlib.sha256(('test-only candidate/' + path).encode()).hexdigest()
    target['selected_configuration']['local_training.json']['sha256'] = target['files'][
        'configs/' + NAME + '/local_training.json']
    refresh_inventory(target)
    return source, target


def refresh_inventory(runtime):
    runtime['runtime_content_sha256'] = hashlib.sha256(json.dumps(
        runtime['files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Same448ContractTests(unittest.TestCase):
    def setUp(self):
        self.ns = staged_namespace()
        self.old, self.new = runtime_pair()
        self.validate = self.ns['validate_same448_rebind_runtime']

    def test_exact_declared_repair(self):
        result = self.validate(self.old, self.new)
        self.assertEqual(set(result), self.ns['SAME448_REBIND_FILES'])

    def test_wrong_source_head(self):
        self.old['source_git_commit'] = '2' * 40
        with self.assertRaises(ValueError):
            self.validate(self.old, self.new)

    def test_same_destination_head(self):
        self.new['source_git_commit'] = self.old['source_git_commit']
        with self.assertRaises(ValueError):
            self.validate(self.old, self.new)

    def test_unrelated_source_change(self):
        path = next(p for p in self.new['files'] if p not in self.ns['SAME448_REBIND_FILES'])
        self.new['files'][path] = 'c' * 64
        refresh_inventory(self.new)
        with self.assertRaises(ValueError):
            self.validate(self.old, self.new)

    def test_added_runtime_file(self):
        self.new['files']['src/not_allowed.py'] = 'c' * 64
        refresh_inventory(self.new)
        with self.assertRaises(ValueError):
            self.validate(self.old, self.new)

    def test_inventory_digest_mismatch(self):
        self.new['runtime_content_sha256'] = 'd' * 64
        with self.assertRaises(ValueError):
            self.validate(self.old, self.new)

    def test_physics_library_changes(self):
        for key in ('physics_hz', 'local_runtime_versions', 'rsl_rl_version'):
            candidate = copy.deepcopy(self.new)
            candidate[key] = 'NOT_THE_ORIGINAL'
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(self.old, candidate)

    def test_config_algorithm_changes(self):
        for key, value in (('learning_rate', .009), ('history_composition', 'two_HISTORY'),
                           ('local_features', []), ('active_initial_sigma_full12', [1.] * 12),
                           ('observation_dimension', 447), ('capture_source_dispatch', 'wrong')):
            candidate = copy.deepcopy(self.new)
            candidate['local_contract'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(self.old, candidate)

    def test_selected_hash_and_path_unbound(self):
        for key, value in (('sha256', 'f' * 64), ('path', 'somewhere/else.json')):
            candidate = copy.deepcopy(self.new)
            candidate['selected_configuration']['local_training.json'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(self.old, candidate)

    def test_revision_must_be_new_exact_field(self):
        self.new['local_contract']['source_tracking_owner_revision'] = 'different'
        with self.assertRaises(ValueError):
            self.validate(self.old, self.new)

    def test_source_bugfix_must_actually_change(self):
        p = 'src/wlr50_clean/ppo/semantic_rr_capture_deferred_late.py'
        self.new['files'][p] = self.old['files'][p]
        refresh_inventory(self.new)
        with self.assertRaises(ValueError):
            self.validate(self.old, self.new)

    def test_patch_parses_and_preserves_ordinary_load_guard(self):
        original, candidate = staged_source()
        self.assertIn("metadata['runtime_contract'] != runtime", candidate)
        self.assertIn("('initialize','migrate','rebind')", candidate)
        self.assertIn("prior,counts=rebind_checkpoint", candidate)
        self.assertEqual(original.count("metadata['runtime_contract'] != runtime"),
                         candidate.count("metadata['runtime_contract'] != runtime"))

    def test_rebind_never_constructs_optimizer_or_changes_learning_state(self):
        _, candidate = staged_source()
        function = next(n for n in ast.parse(candidate).body
            if isinstance(n, ast.FunctionDef) and n.name == 'rebind_checkpoint')
        text = ast.unparse(function)
        self.assertNotIn('optim.Adam(', text)
        self.assertNotIn('initialize_prior(', text)
        self.assertNotIn('migrate_checkpoint(', text)
        self.assertNotIn('.step(', text)
        self.assertNotIn('.backward(', text)
        self.assertIn('parameter_ids', text)
        self.assertIn('capture_training_rng_state', text)

    def test_no_torch_or_isaac_imported_by_stdlib_suite(self):
        import sys
        self.assertFalse(any(n in sys.modules for n in ('torch', 'pxr', 'isaaclab')))


if __name__ == '__main__':
    unittest.main(verbosity=2)


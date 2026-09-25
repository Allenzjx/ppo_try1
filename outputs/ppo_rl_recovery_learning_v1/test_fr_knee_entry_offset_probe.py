"""Small stdlib-only one-channel/provenance tests; no Torch/Isaac import."""
import ast
from copy import deepcopy
import math
from pathlib import Path
import sys
import unittest

import fr_knee_entry_offset_probe as probe


def evidence(tick=120, age=.5, valid=True):
    return dict(phase='P05', tick=tick, stage_age_s=age, eligible=valid,
                blockers=[] if valid else ['FR_current_bearing_absent'])


class CandidateTests(unittest.TestCase):
    def choose(self, p, tick, *, raw=None, filtered=-6., valid=True, age=.5):
        raw = tuple(.1*i for i in range(12)) if raw is None else raw
        return p.choose(raw, tick=tick, cap=24., filtered_request=filtered,
                        evidence=evidence(tick, age, valid))

    def test_fixed_entry_offset_only_one_channel(self):
        p = probe.EntryOffset()
        original = tuple(.1*i for i in range(12))
        applied, changed, first = self.choose(p, 120, raw=original)
        self.assertEqual(changed, [3])
        self.assertAlmostEqual(24.*math.tanh(applied[3]), -6.)
        self.assertEqual([applied[i] for i in range(12) if i != 3], [original[i] for i in range(12) if i != 3])
        for tick in (240, 248, 300, 352):
            applied, _, row = self.choose(p, tick, filtered=-10.)
            self.assertAlmostEqual(24.*math.tanh(applied[3]), -10.)
            self.assertEqual(p.state['entry_filtered_request_deg'], -6.)
            self.assertEqual(row['fixed_entry_delta_deg'], -4.)

    def test_zero_delta_exact_identity_and_both_signs(self):
        for raw_value in (-.4, 0., .4):
            raw = tuple(raw_value if i == 3 else .1*i for i in range(12))
            current = 24.*math.tanh(raw_value)
            self.assertIs(probe.replace_fr_request(raw, 24., current), raw)
            for delta in (-4., 4.):
                out = probe.replace_fr_request(raw, 24., current+delta)
                self.assertAlmostEqual(24.*math.tanh(out[3])-current, delta)
                self.assertEqual([out[i] for i in range(12) if i != 3], [raw[i] for i in range(12) if i != 3])

    def test_ramp_and_release_smooth_to_current_student(self):
        p = probe.EntryOffset()
        raw = (0.,)*12
        self.choose(p, 120, raw=raw)
        applied, _, _ = self.choose(p, 180, raw=raw)
        self.assertAlmostEqual(24.*math.tanh(applied[3]), -8.)
        applied, _, row = self.choose(p, 420, raw=raw)
        self.assertEqual(row['mode'], 'BLEND_TO_CURRENT_STUDENT')
        self.assertAlmostEqual(24.*math.tanh(applied[3]), -5.)
        applied, changed, _ = self.choose(p, 480, raw=raw)
        self.assertEqual(applied, raw)
        self.assertEqual(changed, [])
        self.assertEqual(p.state['status'], 'RELEASED')

    def test_early_loss_latched_no_retrigger(self):
        p = probe.EntryOffset()
        self.choose(p, 120)
        p.observe(121, evidence(121, valid=False))
        self.assertEqual(p.state['status'], 'RELEASE_PENDING')
        raw = (.1,)*12
        applied, changed, _ = self.choose(p, 128, raw=raw)
        self.assertEqual((applied, changed), (raw, []))
        self.assertEqual(p.state['release_requested_tick'], 121)
        self.assertEqual(p.state['release_effective_tick'], 128)
        self.choose(p, 256)
        self.assertEqual(p.state['start_tick'], 120)

    def test_unreachable_fixed_entry_fails_closed(self):
        p = probe.EntryOffset()
        raw = (0.,)*12
        applied, changed, _ = self.choose(p, 120, raw=raw, filtered=-22.)
        self.assertEqual((applied, changed), (raw, []))
        self.assertEqual(p.state['status'], 'NOT_TRIGGERED')

    def test_no_late_or_early_trigger(self):
        p = probe.EntryOffset()
        self.choose(p, 80, age=.4)
        self.assertEqual(p.state['status'], 'WAIT')
        self.choose(p, 130, age=.8)
        self.assertEqual(p.state['status'], 'NOT_TRIGGERED')

    def test_current_bearing_not_history(self):
        legs = {'FR': dict(air=False, support=True, bearing_verified=True, bearing_force_n=12.,
                          top_contact=True, within_top_xy=True, within_lateral_span=True, ground_contact=False),
                'RL': dict(air=False, support=True, bearing_verified=True, bearing_force_n=14.)}
        task = dict(stage_id='P05', stage_age_s=.5333,
                    physical_evaluator=dict(valid=True, physics_tick=64, current_legs=legs))
        self.assertTrue(probe.eligibility(task, phase='P05', tick=64, force_floor=.5)['eligible'])
        lost = deepcopy(task)
        lost['physical_evaluator']['current_legs']['FR'].update(air=True, bearing_force_n=0.)
        lost['physical_evaluator']['history'] = dict(placed={'FR': True})
        self.assertFalse(probe.eligibility(lost, phase='P05', tick=64, force_floor=.5)['eligible'])
        self.assertFalse(probe.eligibility(task, phase='P05', tick=65, force_floor=.5)['eligible'])

    def test_default_refuses_execution_and_import_is_stdlib(self):
        with self.assertRaisesRegex(ValueError, 'NOT EXECUTED'):
            probe.main(['--run-dir', 'not-created'])
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('pxr', sys.modules)

    def test_explicit_current_collection_loader_and_bound_budget(self):
        tree = ast.parse(Path(probe.__file__).read_text(encoding='utf-8'))
        names = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn('_checkpoint_collection_options', names)
        self.assertIn('checkpoint_loader', names)
        self.assertNotIn('construct_semantic_runner', names)
        self.assertNotIn('fit', names)
        self.assertNotIn('load_front_retention439_identity', names)
        self.assertTrue(all(v == 0 for k, v in probe.ZERO.items() if k.startswith('new_')))


if __name__ == '__main__':
    unittest.main()

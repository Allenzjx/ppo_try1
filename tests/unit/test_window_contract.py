# Adaptation draft only: install under tests/unit after production modules exist.
"""Stdlib-only tests; safe while the sole Isaac process runs."""
import ast
import dataclasses
import math
from pathlib import Path
import unittest
from wlr50_clean.ppo.semantic_fl_forward_window import WindowFacts, select_window, scalar_logp, scalar_entropy, scalar_kl


ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/wlr50_clean').is_dir())


def ready(**changes):
    base=WindowFacts(tick=6000,phase='P06',front_left_placed=True,front_right_placed=True,
        owner_fl_nominal_rad_s=.3,owner_explicit_stop=False,safety_stop=False,
        accepted_recoil_or_reverse_pulse=False,winning_owner_is_p06=True,
        p06_rolling_gain=1.,native_ticks_to_atomic_stop=2000)
    return dataclasses.replace(base,**changes)


class WindowTests(unittest.TestCase):
    def test_full_held_positive_owner_is_permitted(self):
        self.assertEqual(select_window(ready()),'P06_FORWARD')

    def test_stop_recoil_and_safety_take_precedence(self):
        for name in ('owner_explicit_stop','safety_stop','accepted_recoil_or_reverse_pulse'):
            with self.subTest(name=name):
                self.assertEqual(select_window(ready(**{name:True})),'OFF')

    def test_authoritative_owner_capture_and_retirement_required(self):
        cases=({'front_left_placed':False},{'front_right_placed':False},{'winning_owner_is_p06':False},
               {'pending_fl_capture':True},{'p06_rolling_gain':.99},{'owner_fl_nominal_rad_s':-.3},
               {'p06_endpoint_issued':True})
        for case in cases:
            with self.subTest(case=case): self.assertEqual(select_window(ready(**case)),'OFF')

    def test_known_atomic_stop_horizon_excludes_entire_eight_tick_interval(self):
        for ticks in (None,-1,0,1,7,8):
            self.assertEqual(select_window(ready(native_ticks_to_atomic_stop=ticks)),'OFF')
        self.assertEqual(select_window(ready(native_ticks_to_atomic_stop=9)),'P06_FORWARD')

    def test_no_prep_or_other_phase_policy(self):
        for phase in ('P05','P07','P09','P10','P11','P12','P13'):
            self.assertEqual(select_window(ready(phase=phase)),'OFF')

    def test_scalar_gaussian_and_inactive_density(self):
        self.assertAlmostEqual(scalar_logp(.2,.2,.3),-math.log(.3)-.5*math.log(2*math.pi))
        self.assertEqual(scalar_kl(.1,.03,.1,.03),0.)
        self.assertGreater(scalar_kl(.1,.03,.2,.04),0.)
        self.assertTrue(math.isfinite(scalar_entropy(.03)))
        self.assertEqual(scalar_logp(99.,.2,.03,False),0.)
        self.assertEqual(scalar_entropy(.03,False),0.)
        self.assertEqual(scalar_kl(.1,.03,.2,.04,old_active=False,new_active=False),0.)
        with self.assertRaises(ValueError): scalar_kl(.1,.03,.2,.04,new_active=False)
        with self.assertRaises(ValueError): scalar_logp(0.,0.,0.)

    def test_actor_source_parses_without_importing_torch(self):
        tree=ast.parse((ROOT/'src/wlr50_clean/ppo/semantic_fl_forward_actor.py').read_text())
        names={node.name for node in tree.body if isinstance(node,ast.ClassDef)}
        self.assertEqual(names,{'WindowFLDistribution','ConstantFLHead','WindowFLActor'})


if __name__=='__main__': unittest.main()

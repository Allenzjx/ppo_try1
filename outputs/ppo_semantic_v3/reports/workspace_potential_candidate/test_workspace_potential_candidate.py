"""Reusable stdlib-only draft; AST-reading is NOT live supervisor integration.

Run this file directly for a JSON receipt on stdout. It writes no files, imports
no project runtime modules, and never patches the production module or class.
"""
from __future__ import annotations

import ast
import copy
from decimal import Decimal, localcontext
import io
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping
import unittest

from workspace_potential_candidate import interval_distance_progress as progress

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE = PROJECT_ROOT / "src/wlr50_clean/ppo/semantic_supervisor.py"
ACTUAL = (
    ("RR_first", -.532472255957309, .44446636674462875),
    ("RL_first", -.5413483663843905, .4375614156071736),
    ("RR_closest", -.4928732051895892, .47812738828212126),
    ("RL_closest", -.5309575333641803, .44566653468525047),
    ("RR_terminal", -.8239279951350724, .29276473124698943),
    ("RL_terminal", -.8555664386050736, .2823051880712586),
)


def original_predicate():
    """Compile just the unchanged source predicate/_clip, not module imports."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    supervisor = next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="TaskStageSupervisor")
    predicate = next(x for x in supervisor.body if isinstance(x,ast.FunctionDef) and x.name=="predicate")
    clip = next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="_clip")
    namespace = {"Mapping":Mapping,"Any":Any}
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(clip),copy.deepcopy(predicate)],type_ignores=[]))
    exec(compile(module,str(SOURCE),"exec"),namespace)
    return namespace["predicate"]


PREDICATE = original_predicate()


def old_workspace(x, lateral=True):
    legs={leg:{"front_distance_m":x,"within_lateral_span":lateral,"support":True} for leg in ("FR","FL","RR","RL")}
    obj=SimpleNamespace(spec={"geometry":{"workspace_min_m":-.22,"workspace_max_m":.06},
                              "support":{"minimum_other_supports":2}})
    return PREDICATE(obj,"workspace_RR",{"history":{},"current_legs":legs})


def decimal_expected(x, lower, upper, scale):
    with localcontext() as ctx:
        ctx.prec=1600
        dx,dl,du,ds=map(Decimal.from_float,map(float,(x,lower,upper,scale)))
        distance=max(dl-dx,Decimal(0),dx-du)
        return float(ds/(ds+distance))


class WorkspacePotentialCandidateTests(unittest.TestCase):
    def test_actual_six_recorded_positions_old_flat_new_ordered(self):
        for label,x,expected in ACTUAL:
            with self.subTest(label=label):
                self.assertEqual(old_workspace(x),0.)
                self.assertAlmostEqual(progress(x,-.22,.06,True),expected,places=15)
        for leg in ("RR","RL"):
            rows={name:progress(x,-.22,.06,True) for name,x,_ in ACTUAL if name.startswith(leg)}
            self.assertLess(rows[leg+"_terminal"],rows[leg+"_first"])
            self.assertLess(rows[leg+"_first"],rows[leg+"_closest"])

    def test_interval_and_both_boundaries_equal_one(self):
        for x in (-.22,-.1,0.,.06):
            self.assertEqual(progress(x,-.22,.06,True),1.)
            self.assertEqual(old_workspace(x),1.)

    def test_old_minus_point47_cutoff_now_continuous(self):
        points=[-.470001,-.47,-.469999]
        values=[progress(x,-.22,.06,True) for x in points]
        self.assertLess(values[0],values[1]); self.assertLess(values[1],values[2])
        self.assertAlmostEqual(values[1],.5,places=15)
        self.assertLess(max(values)-min(values),3e-6)
        self.assertEqual(old_workspace(points[0]),0.)

    def test_left_right_equal_distance_symmetry(self):
        for distance in (.001,.05,.25,1.,100.):
            self.assertAlmostEqual(progress(-.22-distance,-.22,.06,True),progress(.06+distance,-.22,.06,True),places=15)

    def test_distance_monotone_and_far_nonzero(self):
        values=[progress(-.22-d,-.22,.06,True) for d in (0.,.001,.25,.5,1.,10.,1e6)]
        self.assertTrue(all(a>b for a,b in zip(values,values[1:])))
        self.assertGreater(values[-1],0.)

    def test_lateral_false_zero_without_skipping_validation(self):
        self.assertEqual(progress(0.,-.22,.06,False),0.)
        self.assertEqual(progress(-1.,-.22,.06,False),0.)
        with self.assertRaises(ValueError): progress(float("nan"),-.22,.06,False)
        with self.assertRaises(ValueError): progress(0.,.06,-.22,False)

    def test_strict_numeric_validation(self):
        for bad in (True,False,float("nan"),float("inf"),-float("inf"),"0",None):
            for field in range(4):
                values=[0.,-.22,.06,.25];values[field]=bad
                with self.subTest(bad=repr(bad),field=field), self.assertRaises(ValueError):
                    progress(values[0],values[1],values[2],True,values[3])

    def test_strict_span_bool(self):
        for bad in (0,1,None,"True",[],float("nan")):
            with self.subTest(bad=repr(bad)),self.assertRaises(ValueError): progress(0.,-.22,.06,bad)

    def test_interval_and_scale_validation(self):
        for lo,hi in ((0.,0.),(.1,-.1)):
            with self.assertRaises(ValueError): progress(0.,lo,hi,True)
        for scale in (0.,-0.,-.25):
            with self.assertRaises(ValueError): progress(0.,-.22,.06,True,scale)

    def test_finite_distance_denominator_overflow_is_avoided(self):
        f=sys.float_info.max
        observed=progress(-f,0.,1.,True,f)
        self.assertEqual(observed,.5)
        self.assertEqual(observed,decimal_expected(-f,0.,1.,f))

    def test_finite_endpoint_subtraction_overflow_is_avoided(self):
        f=sys.float_info.max
        for x,lo,hi in ((-f,f/2,f),(f,-f,-f/2)):
            observed=progress(x,lo,hi,True,f)
            self.assertAlmostEqual(observed,.4,places=15)
            self.assertEqual(observed,decimal_expected(x,lo,hi,f))

    def test_subnormal_scale_and_distance_remain_usable(self):
        tiny=math.ulp(0.)
        self.assertEqual(progress(-tiny,0.,1.,True,tiny),.5)
        self.assertEqual(progress(-tiny,0.,1.,True,2*tiny),2/3)

    def test_true_extreme_underflow_is_zero_not_error(self):
        tiny=math.ulp(0.); f=sys.float_info.max
        for lo,hi in ((0.,1.),(f/2,f)):
            result=progress(-f,lo,hi,True,tiny)
            self.assertEqual(result,0.)
            self.assertEqual(result,decimal_expected(-f,lo,hi,tiny))

    def test_soft_near_boundary_rounding_must_not_replace_hard_predicate(self):
        x=math.nextafter(-.22,-math.inf)
        self.assertLess(x,-.22)
        self.assertEqual(progress(x,-.22,.06,True),1.)  # Permitted soft rounding, not completion.
        self.assertLess(old_workspace(x),1.)
        self.assertLess(old_workspace(-.221),1.)
        self.assertLess(old_workspace(.061),1.)

    def test_normal_finite_range_matches_high_precision_reference(self):
        for x in (-1e100,-100.,-.531,-.47,-.221,.061,1.,1e100):
            actual=progress(x,-.22,.06,True)
            expected=decimal_expected(x,-.22,.06,.25)
            self.assertTrue(math.isfinite(actual) and 0.<=actual<=1.)
            self.assertLessEqual(abs(actual-expected),max(2*math.ulp(expected),math.ulp(0.)))

    def test_pbrs_stationary_soft_progress_is_not_positive(self):
        phi=.02125*progress(-.7,-.22,.06,True)
        self.assertLess(5*(.995*phi-phi),0.)
        closer=.02125*progress(-.6,-.22,.06,True)
        farther=.02125*progress(-.8,-.22,.06,True)
        self.assertGreater(5*(.995*closer-phi),5*(.995*farther-phi))
        self.assertEqual(5*(.995*0.-phi),-5*phi)  # Unchanged terminal convention.

    def test_no_phase_history_or_runtime_modules(self):
        self.assertNotIn("torch",sys.modules)
        self.assertNotIn("wlr50_clean.ppo.semantic_supervisor",sys.modules)
        self.assertFalse(any(name.startswith(("isaaclab","omni.isaac")) for name in sys.modules))


if __name__=="__main__":
    started=time.perf_counter(); stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(WorkspacePotentialCandidateTests))
    print(json.dumps({"scope":"UNWIRED candidate only; AST predicate branch, not complete supervisor integration",
        "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "skipped":len(result.skipped),"passed":result.wasSuccessful(),
        "wall_seconds":time.perf_counter()-started,
        "actual_positions":[{"label":label,"x":x,"old_workspace":old_workspace(x),"candidate":progress(x,-.22,.06,True)} for label,x,_ in ACTUAL],
        "test_output":stream.getvalue()},ensure_ascii=False,allow_nan=False,indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)

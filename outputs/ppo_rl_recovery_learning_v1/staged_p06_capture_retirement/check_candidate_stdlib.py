"""Load only the isolated stdlib capture helper; never Torch/PXR/Isaac."""
from pathlib import Path
import importlib.util
import json
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "src"))
NAME = "wlr50_clean.ppo.semantic_capture_assist"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load("legacy_capture_for_stdlib_comparison", ROOT / "src/wlr50_clean/ppo/semantic_capture_assist.py")
candidate = load(NAME, HERE / "candidate/src/wlr50_clean/ppo/semantic_capture_assist.py")
tests = load("isolated_p06_capture_retirement_tests", HERE / "test_semantic_p06_capture_retirement.py")
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
if not result.wasSuccessful():
    raise SystemExit(1)

# Exact old/new numeric traces in scopes that the candidate must not alter.
traces = []
for label, stage, placed in (("P05_initial_then_exhausted", "P05", False),
                             ("P06_unplaced_pending", "P06", False),
                             ("P07_existing_release", "P07", True)):
    old, new = legacy.HipOnlyCaptureAssist(), candidate.HipOnlyCaptureAssist()
    if stage == "P07":
        old.state.update(tests.exhausted().state)
        new.state.update(tests.exhausted().state)
    for tick in range(1, 901):
        # Initial P05/P06 may contact and reopen, without historical placement.
        contact = stage != "P07" and tick in (20, 21, 50)
        ctx = tests.context(tick, stage_id=stage, placed_FL=placed,
            current_FL_air=not contact, current_FL_support=contact,
            top_surface_contact=contact, obstacle_pair_active=contact,
            hip_actual_deg=old.state["hip_target_deg"])
        kwargs = dict(context=ctx, previous_final_full12=tests.FINAL, physics_dt_s=tests.DT)
        a, b = old.advance(**kwargs), new.advance(**kwargs)
        assert old.state == new.state, (label, tick, old.state, new.state)
        for side in ("state_before", "state_after"):
            b[side].pop("retirement_revision")
        assert a == b, (label, tick)
        assert legacy.apply_capture_assist_snapshot(tests.CANDIDATE, old.snapshot()) == \
            candidate.apply_capture_assist_snapshot(tests.CANDIDATE, new.snapshot())
    traces.append({"name": label, "ticks": 900, "numeric_state_receipts_and_applied_targets_equal": True})

forbidden = sorted(name for name in sys.modules
                   if name == "torch" or name.startswith(("torch.", "pxr", "isaac", "omni")))
assert not forbidden, forbidden
summary = {"normal_unit_tests_passed": result.testsRun,
           "legacy_trace_comparisons": traces,
           "forbidden_imports": forbidden, "production_changed": False,
           "physical_validation": "not_run", "model_loaded": False}
(HERE / "stdlib_test_result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))

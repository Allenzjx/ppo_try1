"""Output-only baseline comparison and in-memory patch generation; no file writes."""
import ast
from copy import deepcopy
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))


def load(name, path):
    item = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(item)
    sys.modules[name] = module
    item.loader.exec_module(module)
    return module


old = load("wlr50_clean.ppo.preedge_old_review", ROOT / "src/wlr50_clean/ppo/semantic_supervisor.py")
new = load("wlr50_clean.ppo.preedge_new_review", HERE / "src/wlr50_clean/ppo/semantic_supervisor.py")
sys.modules["wlr50_clean.ppo.semantic_supervisor"] = new
fixture = load("preedge_fixture", HERE / "tests/unit/test_semantic_p05_preedge_recovery.py")
spec = fixture.configured(False)
contract = fixture.load_motion_contract(ROOT / "configs/recording_motion_contract.json")
fsm = fixture.load_fsm_spec(ROOT / "configs/fsm_states.yaml")
a, b = (module.NominalMotionProvider(contract, spec=deepcopy(spec), fsm_spec=fsm) for module in (old, new))
for tick in range(5000):
    task = fixture.task(tick, tick/120.)
    assert a.evaluate(deepcopy(task)) == b.evaluate(deepcopy(task)), tick
    assert a.tracking_servo_names == b.tracking_servo_names, tick
    assert a.normal_drive_bias_full12 == b.normal_drive_bias_full12, tick
    assert a.nominal_suggestion_diagnostics == b.nominal_suggestion_diagnostics, tick
    assert a._continuous_layers[-1]["sample"] == b._continuous_layers[-1]["sample"], tick

old_tree = ast.parse((ROOT / "src/wlr50_clean/ppo/semantic_supervisor.py").read_text())
new_tree = ast.parse((HERE / "src/wlr50_clean/ppo/semantic_supervisor.py").read_text())
allowed = {"load_task_spec", "NominalMotionProvider.__init__",
           "NominalMotionProvider.nominal_suggestion_diagnostics", "NominalMotionProvider.evaluate"}
added = {"_p05_preedge_recovery_enabled", "NominalMotionProvider._p05_preedge_recovery_status"}
def stripped(tree):
    def prune(node, prefix=""):
        kept = []
        for child in node.body:
            name = prefix + getattr(child, "name", "")
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and name in allowed | added:
                continue
            if isinstance(child, ast.ClassDef):
                prune(child, name + ".")
            if isinstance(child, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "P05_PREEDGE_RECOVERY_MODE" for x in child.targets):
                continue
            kept.append(child)
        node.body = kept
    prune(tree)
    return ast.dump(tree, include_attributes=False)
assert stripped(old_tree) == stripped(new_tree)
old_spec = old.load_task_spec(ROOT / "configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml")
new_spec = new.load_task_spec(HERE / "configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml")
assert new_spec["nominal"].pop(fixture.KEY) == new.P05_PREEDGE_RECOVERY_MODE
assert new_spec == old_spec

paths = ["src/wlr50_clean/ppo/semantic_supervisor.py",
         "configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml",
         "tests/unit/test_semantic_p05_preedge_recovery.py"]
patch = ["*** Begin Patch\n"]
for path in paths:
    target = HERE / path
    text = target.read_text(encoding="utf-8")
    source = ROOT / path
    if not source.exists():
        patch.append(f"*** Add File: {path}\n")
        patch.extend("+" + line + "\n" for line in text.splitlines())
    else:
        before = source.read_text(encoding="utf-8")
        lines = list(difflib.unified_diff(before.splitlines(keepends=True), text.splitlines(keepends=True), n=3))
        patch.append(f"*** Update File: {path}\n")
        for line in lines[2:]:
            patch.append("@@\n" if line.startswith("@@") else line)
        # Apply each generated context hunk in memory; never touch production.
        result, offset = before.splitlines(), 0
        chunks, current = [], None
        for line in lines[2:]:
            if line.startswith("@@"):
                current = []; chunks.append(current)
            else:
                current.append(line.rstrip("\n"))
        for chunk in chunks:
            left = [line[1:] for line in chunk if line[0] in " -"]
            right = [line[1:] for line in chunk if line[0] in " +"]
            matches = [i for i in range(offset, len(result)-len(left)+1) if result[i:i+len(left)] == left]
            assert len(matches) == 1, (path, len(matches))
            at = matches[0]; result[at:at+len(left)] = right; offset = at + len(right)
        assert result == text.splitlines(), path
patch.append("*** End Patch\n")
report = dict(default_off_equal_physics_ticks=5000, default_off_commands_tracking_bias_diagnostics_equal=True,
    unchanged_ast_except=sorted(allowed | added | {"P05_PREEDGE_RECOVERY_MODE"}),
    unchanged_evaluator_supervisor_physical_potential_continuous_scheduler=True,
    only_config_change="nominal.p05_preedge_approach_recovery=p05_preedge_approach_recovery_v1",
    in_memory_apply_patch_roundtrip=True, production_applied=False, physical_success_claimed=False,
    files={path: hashlib.sha256((HERE / path).read_bytes()).hexdigest() for path in paths})
print(json.dumps({"report": report, "patch": "".join(patch)}))

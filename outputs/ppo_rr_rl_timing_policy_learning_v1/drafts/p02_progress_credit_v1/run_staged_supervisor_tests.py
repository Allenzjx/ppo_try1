"""Apply draft hunks in memory only; forbid Torch/Isaac and run pure pytest."""
import importlib.abc
import pathlib
import sys
import types

HERE = pathlib.Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "src/wlr50_clean").is_dir())


class ForbiddenImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch", "isaaclab", "isaacsim", "omni"}:
            raise RuntimeError("pure staging audit forbids " + fullname)
        return None


sys.meta_path.insert(0, ForbiddenImports())
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/unit")]
source_path = ROOT / "src/wlr50_clean/ppo/semantic_supervisor.py"
source = source_path.read_text(encoding="utf-8")
patch = (HERE / "production.patch").read_text(encoding="utf-8")
for chunk in patch.split("@@\n")[1:]:
    old, new = [], []
    for line in chunk.splitlines(keepends=True):
        if line.startswith("***"):
            break
        if line.startswith(" "):
            old.append(line[1:]); new.append(line[1:])
        elif line.startswith("-"):
            old.append(line[1:])
        elif line.startswith("+"):
            new.append(line[1:])
        else:
            raise RuntimeError("unexpected patch line " + repr(line))
    before, after = "".join(old), "".join(new)
    if source.count(before) != 1:
        raise RuntimeError("draft hunk must match exactly once: " + before[:180])
    source = source.replace(before, after, 1)

name = "wlr50_clean.ppo.semantic_supervisor"
module = types.ModuleType(name)
module.__file__ = str(source_path)
module.__package__ = "wlr50_clean.ppo"
sys.modules[name] = module
exec(compile(source, str(source_path), "exec"), module.__dict__)
import pytest

raise SystemExit(pytest.main([str(HERE / "test_supervisor_integration_draft.py"), "-q", "-p", "no:cacheprovider"]))

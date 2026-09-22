"""One process-local candidate runtime, with separately retained real old fixtures.

This never writes or imports candidate files into another process. The few
production modules deriving config paths from __file__ retain that root anchor;
__spec__.origin and code object filenames always identify the actual candidate.
"""
from __future__ import annotations
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE / "src/wlr50_clean/ppo"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests/unit"))

# Freeze genuine production objects before any candidate canonical import. These
# references intentionally remain old even after the candidate runtime is loaded.
import wlr50_clean.ppo as package
OLD = {name: importlib.import_module("wlr50_clean.ppo." + name) for name in
       ("semantic_reward", "semantic_supervisor", "semantic_observation")}
OLD_FIXTURES = importlib.import_module("test_semantic_observation_reward_env")

# Clear the whole PPO dependency subgraph, not only the three actor modules:
# otherwise semantic_training can retain an old migration function by import order.
for full in tuple(sys.modules):
    if full.startswith("wlr50_clean.ppo."):
        del sys.modules[full]
for name, value in tuple(vars(package).items()):
    if getattr(value, "__name__", "").startswith("wlr50_clean.ppo."):
        delattr(package, name)

_ANCHORS = {"semantic_training", "semantic_supervisor", "semantic_observation",
            "semantic_migration", "semantic_cli"}


class _CandidateLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        name = module.__name__.rsplit(".", 1)[-1]
        if name in _ANCHORS:
            module.__file__ = str(ROOT / "src/wlr50_clean/ppo" / (name + ".py"))
        super().exec_module(module)


class _CandidateFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("wlr50_clean.ppo."):
            return None
        source = SOURCE / (fullname.rsplit(".", 1)[-1] + ".py")
        if not source.is_file():
            return None
        return importlib.util.spec_from_file_location(fullname, source,
            loader=_CandidateLoader(fullname, str(source)))


sys.meta_path.insert(0, _CandidateFinder())
CANDIDATE = {path.stem: importlib.import_module("wlr50_clean.ppo." + path.stem)
             for path in sorted(SOURCE.glob("*.py"))}


def provenance():
    return {name: {"actual_code_file": module.__spec__.origin,
                   "module_file_config_anchor": module.__file__,
                   "candidate": Path(module.__spec__.origin).resolve() == SOURCE / (name + ".py")}
            for name, module in CANDIDATE.items()}


assert all(row["candidate"] for row in provenance().values())
assert all(Path(module.__spec__.origin).resolve().is_relative_to(ROOT / "src")
           for module in OLD.values())

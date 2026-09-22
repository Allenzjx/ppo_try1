"""Output-only overlay; not included in the production patch."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
name = "wlr50_clean.ppo.semantic_supervisor"
spec = importlib.util.spec_from_file_location(name, HERE / "src/wlr50_clean/ppo/semantic_supervisor.py")
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)

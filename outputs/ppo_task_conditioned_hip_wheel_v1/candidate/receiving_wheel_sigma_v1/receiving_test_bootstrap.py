"""Output-only test package path: all seven candidate modules or fail closed."""
from pathlib import Path
import importlib
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / "src/wlr50_clean/ppo"
NAMES = ("semantic_receiving_wheel_profile", "semantic_receiving_wheel_sigma",
         "semantic_migration", "semantic_policy_distribution", "semantic_training",
         "semantic_cli", "semantic_checkpoint_prefix_policy")
sys.path.insert(0, str(ROOT / "src"))
import wlr50_clean.ppo as package
if any("wlr50_clean.ppo." + name in sys.modules for name in NAMES):
    raise RuntimeError("run integration tests in a fresh process, without a partial package overlay")
package.__path__.insert(0, str(SOURCE))
MODULES = {name: importlib.import_module("wlr50_clean.ppo." + name) for name in NAMES}
assert all(Path(module.__file__).resolve() == (SOURCE / (name + ".py")).resolve()
           for name, module in MODULES.items())

"""Run output-mirror tests on CPU; no actual checkpoint or native simulator."""
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
root = here.parents[1]
mirror = here / "temperature_combined_candidate"
sys.path.insert(0, str(root / "src"))
sys.path.insert(0, str(root / "tests/unit"))
sys.path.insert(0, str(mirror / "tests/unit"))
import wlr50_clean.ppo as package
package.__path__.insert(0, str(mirror / "src/wlr50_clean/ppo"))
import pytest

selected = sys.argv[1:] or [
    str(mirror / "tests/unit/test_semantic_temperature_loader_cli.py"),
    str(mirror / "tests/unit/test_semantic_exploration_temperature_migration.py")]
raise SystemExit(pytest.main(["-o", "addopts=", "-q", *selected]))

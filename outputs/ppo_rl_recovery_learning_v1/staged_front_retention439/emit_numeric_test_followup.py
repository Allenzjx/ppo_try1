"""Outputs-only one-test follow-up, independent of already applied runtime."""
import ast
import difflib
from pathlib import Path

here = Path(__file__).resolve().parent
relative = "tests/unit/test_semantic_front_retention439.py"
source = (here / "test_semantic_front_retention439.py").read_text(encoding="utf-8")
tree = ast.parse(source)
node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and
            item.name == "test_sparse_P10_P13_full_gaussian_and_raw_logp_synthetic_unit_only")
function = "".join(source.splitlines(True)[node.lineno-1:node.end_lineno])
before = (here.parents[2] / relative).read_text(encoding="utf-8")
anchor = "def test_aux_then_one_ordinary_ppo_update_validates_and_carries_ledger(cpu_state,tmp_path):\n"
assert before.count(anchor) == 1 and node.name not in before
after = before.replace(anchor, function + "\n\n\n" + anchor)
ast.parse(after)
diff = list(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                               fromfile="a/" + relative, tofile="b/" + relative))
(here / "additional_numeric_test.patch").write_text("".join(diff), encoding="utf-8", newline="\n")
apply = ["*** Begin Patch\n", "*** Update File: " + relative + "\n"]
apply += ["@@\n" if line.startswith("@@") else line for line in diff[2:]]
apply += ["*** End Patch\n"]
(here / "additional_numeric_test.apply_patch").write_text("".join(apply), encoding="utf-8", newline="\n")
print("one isolated synthetic full-Gaussian/raw-logp test follow-up emitted; production untouched")

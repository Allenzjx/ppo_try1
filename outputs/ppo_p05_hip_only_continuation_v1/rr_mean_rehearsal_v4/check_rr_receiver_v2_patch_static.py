"""Read-only parser/application/AST check; no candidate imports, tests, fit or writes."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
PATCH = Path(__file__).with_name("rr_receiver_v2_production.patch")
HEAD = "5fd88852bf20c94cd74405c791a13a9fd9e0a3d8"


def parse_patch():
    lines = PATCH.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "*** Begin Patch" and lines[-1] == "*** End Patch"
    sections, current = [], None
    for line in lines[1:-1]:
        if line.startswith(("*** Update File: ", "*** Add File: ")):
            kind, path = line[4:].split(" File: ", 1)
            current = {"kind": kind, "path": path, "lines": []}
            sections.append(current)
        else:
            assert current is not None
            current["lines"].append(line)
    return sections


def apply_in_memory(section):
    path = ROOT / section["path"]
    if section["kind"] == "Add":
        assert not path.exists(), "production add target already exists"
        assert all(line.startswith("+") for line in section["lines"])
        return None, "\n".join(line[1:] for line in section["lines"])+"\n"
    before = path.read_text(encoding="utf-8")
    after, hunks, hunk = before, [], None
    for line in section["lines"]:
        if line == "@@":
            hunk = []
            hunks.append(hunk)
        else:
            assert hunk is not None and line[:1] in ("+", "-", " ")
            hunk.append(line)
    for hunk in hunks:
        old = "\n".join(line[1:] for line in hunk if line[:1] in ("-", " "))+"\n"
        new = "\n".join(line[1:] for line in hunk if line[:1] in ("+", " "))+"\n"
        assert after.count(old) == 1, (section["path"], "nonunique/missing complete-line hunk", old)
        after = after.replace(old, new, 1)
    return before, after


def pruned(source, functions=(), names=()):
    module = ast.parse(source)
    module.body = [node for node in module.body if not (
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in functions
        or isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in names for target in node.targets))]
    return ast.dump(module, include_attributes=False)


def main():
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == HEAD
    rows = {}
    for section in parse_patch():
        path = section["path"]
        before, after = apply_in_memory(section)
        if path.endswith(".py"):
            ast.parse(after, filename=path)
        if path.endswith("/semantic_supervisor.py"):
            assert pruned(before,
                ("_rr_workspace_retirement_enabled", "_current_rr_receiver_preparation_retired")) == pruned(after,
                ("_rr_workspace_retirement_enabled", "_current_rr_receiver_preparation_retired"),
                ("RR_WORKSPACE_RETIREMENT_MODE_V2",))
            # Both whole classes are included above: TaskEvaluator / nominal unchanged.
        elif path.endswith("/semantic_migration.py"):
            assert pruned(before, ("validate_migration_plan",)) == pruned(after, ("validate_migration_plan",))
        elif path.endswith("/semantic_training.py"):
            old_tree, new_tree = ast.parse(before), ast.parse(after)
            removed = []
            for node in ast.walk(new_tree):
                if isinstance(node, ast.Tuple):
                    keep = []
                    for value in node.elts:
                        if isinstance(value, ast.Constant) and value.value in (
                            "rr_receiver_retirement_v2_migration", "rr_receiver_retirement_v2_branch"):
                            removed.append(value.value)
                        else:
                            keep.append(value)
                    node.elts = keep
            assert sorted(removed) == ["rr_receiver_retirement_v2_branch", "rr_receiver_retirement_v2_migration"]
            assert ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False)
        elif path.endswith("/stage_task_spec.yaml"):
            assert before.replace(
                "rr_postcross_workspace_semantics: current_qualified_RR_over_top_receiver_retirement_v1",
                "rr_postcross_workspace_semantics: established_RR_over_top_receiver_retirement_v2") == after
        current_bytes = None if before is None else (ROOT/path).read_bytes()
        rows[path] = {
            "syntax_or_exact_config_change": True,
            "before_sha256": None if current_bytes is None else hashlib.sha256(current_bytes).hexdigest(),
            "candidate_lf_sha256": hashlib.sha256(after.encode("utf-8")).hexdigest(),
            "candidate_lines": len(after.splitlines()),
        }
    changed = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", "HEAD", "--",
        *[p for p,row in rows.items() if row["before_sha256"] is not None]], check=True,
        capture_output=True, text=True).stdout.strip()
    assert not changed, ("preexisting production/test changes", changed)
    print(json.dumps({
        "schema": "wlr50_clean.rr_receiver_v2_patch_static.v1", "base_head": head,
        "patch_sha256": hashlib.sha256(PATCH.read_bytes()).hexdigest(), "target_count": len(rows),
        "production_edited": False, "imports_of_candidate": 0, "tests_executed": 0,
        "fit_steps": 0, "Isaac_started": False, "checks": rows,
    }, indent=2))


if __name__ == "__main__":
    main()

"""Mechanical outputs-only patch assembly and AST validation; never imports models."""
from __future__ import annotations
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPECTED_HEAD = "f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b"


def routing_targets():
    rows = (HERE / "routing.patch").read_text(encoding="utf-8").splitlines()
    result, path, hunks, hunk = {}, None, [], []
    def flush_hunk():
        nonlocal hunk
        if hunk: hunks.append(hunk)
        hunk = []
    def flush_file():
        nonlocal hunks
        if path is None: return
        flush_hunk()
        value = (ROOT / path).read_text(encoding="utf-8")
        for chunk in hunks:
            before = "\n".join(line[1:] for line in chunk if line[0] in " -") + "\n"
            after = "\n".join(line[1:] for line in chunk if line[0] in " +") + "\n"
            if value.count(before) != 1:
                raise ValueError(f"routing anchor no longer unique: {path}: {before[:120]!r}")
            value = value.replace(before, after, 1)
        result[path] = value
        hunks = []
    for line in rows:
        if line.startswith("*** Update File: "):
            flush_file(); path = line.removeprefix("*** Update File: ")
        elif line.startswith("@@"): flush_hunk()
        elif line.startswith("***"): continue
        elif line and line[0] in " +-": hunk.append(line)
        else: raise ValueError("unsupported routing patch syntax")
    flush_file()
    return result


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--reward-candidate", type=Path)
    args = p.parse_args()
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if head != EXPECTED_HEAD: raise ValueError("staging base HEAD changed; review rather than silently rebase")
    targets = routing_targets()
    targets["src/wlr50_clean/ppo/semantic_rr_retention_migration.py"] = (HERE / "semantic_rr_retention_migration.py").read_text(encoding="utf-8")
    targets["tests/unit/test_semantic_rr_retention_migration.py"] = (HERE / "test_semantic_rr_retention_migration.py").read_text(encoding="utf-8")
    if args.reward_candidate:
        if not args.reward_candidate.resolve().is_relative_to(HERE):
            raise ValueError("reward candidate must be inside the isolated staging directory")
        targets["src/wlr50_clean/ppo/semantic_reward.py"] = args.reward_candidate.read_text(encoding="utf-8")
    patches, apply_sections, inventory = [], [], {}
    for path, value in sorted(targets.items()):
        ast.parse(value, filename=path)
        production = ROOT / path
        old = production.read_text(encoding="utf-8") if production.exists() else ""
        if old == value: raise ValueError("expected reviewed delta missing: " + path)
        delta = list(difflib.unified_diff(old.splitlines(keepends=True), value.splitlines(keepends=True),
            fromfile="a/" + path if production.exists() else "/dev/null", tofile="b/" + path))
        patches.append("".join(delta))
        if production.exists():
            body = "".join("@@\n" if line.startswith("@@") else line for line in delta[2:])
            apply_sections.append("*** Update File: " + path + "\n" + body)
        else:
            apply_sections.append("*** Add File: " + path + "\n" + "".join("+" + line for line in value.splitlines(keepends=True)))
        destination = HERE / "tree" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(value, encoding="utf-8", newline="\n")
        inventory[path] = dict(before=hashlib.sha256(production.read_bytes()).hexdigest() if production.exists() else None,
            after=hashlib.sha256(destination.read_bytes()).hexdigest())
    label = "complete_reward_identity" if args.reward_candidate else "migration_only"
    patch = HERE / (label + ".patch")
    patch.write_text("".join(patches), encoding="utf-8", newline="\n")
    apply_patch = HERE / (label + ".apply_patch")
    apply_patch.write_text("*** Begin Patch\n" + "".join(apply_sections) + "*** End Patch\n", encoding="utf-8", newline="\n")
    checked = subprocess.run(["git", "-C", str(ROOT), "apply", "--check", str(patch)], check=True,
        capture_output=True, text=True)
    result = dict(base_head=head, label=label, runtime_paths=[k for k in inventory if k.startswith("src/")],
        files=inventory, ast_only=True, git_apply_check=True, production_changed=False,
        model_imports=False, tests_executed=False, reward_included=bool(args.reward_candidate),
        patch=str(patch), patch_sha256=hashlib.sha256(patch.read_bytes()).hexdigest(),
        apply_patch=str(apply_patch), apply_patch_sha256=hashlib.sha256(apply_patch.read_bytes()).hexdigest())
    (HERE / (label + "_manifest.json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()

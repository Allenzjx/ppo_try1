"""Stdlib-only mechanical staging. Never writes a production path."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = "src/wlr50_clean/ppo/"


def replace_once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new)


def main():
    assert HERE.is_relative_to(ROOT / "outputs/ppo_rl_recovery_learning_v1")
    candidates = {CODE + "semantic_front_retention439.py": (HERE / "semantic_front_retention439.py").read_text(encoding="utf-8")}
    name = CODE + "semantic_rear_policy_timing_migration.py"
    source = (ROOT / name).read_text(encoding="utf-8")
    old = "    if metadata.get('rr_retention_reward_migration') is not None:\n"
    new = """    if 'front_retention439_runtime_identity' in metadata or 'front_retention439_auxiliary' in metadata:
        from .semantic_front_retention439 import validate_front_retention439_lineage
        validate_front_retention439_lineage(metadata,contract,output_root,
            checkpoint_output_routing=checkpoint_output_routing)
        return
""" + old
    candidates[name] = replace_once(source, old, new)
    name = CODE + "semantic_training.py"
    source = (ROOT / name).read_text(encoding="utf-8")
    old = '    if migration is not None and migration.get("rr_retention_same439_factor") is not None:\n'
    source = replace_once(source, old, """    if migration is not None and migration.get("front_retention439_identity_factor") is not None:
        if warm_start is not None or policy_migration is not None:
            raise ValueError("front retention accounting identity cannot mix another migration")
        from .semantic_front_retention439 import load_front_retention439_identity
        return load_front_retention439_identity(runner, checkpoint, contract=contract, seed=seed, record=migration)
""" + old)
    old = '    metadata = json.loads(sidecar.read_text(encoding="utf-8"))\n'
    source = replace_once(source, old, old + """    if "front_retention439_runtime_identity" in metadata or "front_retention439_auxiliary" in metadata:
        from .semantic_front_retention439 import validate_front_retention439_lineage
        route = metadata.get("checkpoint_output_routing", {})
        validate_front_retention439_lineage(metadata, metadata["runtime_contract"], Path(route.get("output_root", "")),
            checkpoint_output_routing=route)
""")
    old = '    runner.save(str(checkpoint), infos=metadata)\n'
    source = replace_once(source, old, """    if "front_retention439_runtime_identity" in metadata or "front_retention439_auxiliary" in metadata:
        from .semantic_front_retention439 import validate_front_retention439_lineage
        route = metadata.get("checkpoint_output_routing", {})
        validate_front_retention439_lineage(metadata, metadata["runtime_contract"], Path(route.get("output_root", "")),
            checkpoint_output_routing=route)
""" + old)
    old = '                for key in ("rr_retention_reward_migration", "rear_owner_recovery_migration",'
    source = replace_once(source, old, '                for key in ("front_retention439_runtime_identity", "front_retention439_auxiliary", "rr_retention_reward_migration", "rear_owner_recovery_migration",')
    candidates[name] = source
    tests = "tests/unit/test_semantic_front_retention439.py"
    candidates[tests] = (HERE / "test_semantic_front_retention439.py").read_text(encoding="utf-8")
    standard, apply = [], ["*** Begin Patch\n"]
    records = {}
    for relative, target in candidates.items():
        ast.parse(target, filename=relative)
        path = ROOT / relative
        before = path.read_text(encoding="utf-8") if path.exists() else ""
        staged = HERE / "tree" / relative
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(target, encoding="utf-8", newline="\n")
        lines = list(difflib.unified_diff(before.splitlines(True), target.splitlines(True),
            fromfile="a/" + relative if path.exists() else "/dev/null", tofile="b/" + relative))
        standard.extend(lines)
        if path.exists():
            apply.append("*** Update File: " + relative + "\n")
            apply.extend("@@\n" if line.startswith("@@") else line for line in lines[2:])
        else:
            apply.append("*** Add File: " + relative + "\n")
            apply.extend("+" + line for line in target.splitlines(True))
        records[relative] = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None,
                             "candidate_sha256": hashlib.sha256(staged.read_bytes()).hexdigest()}
    apply.append("*** End Patch\n")
    (HERE / "candidate.patch").write_text("".join(standard), encoding="utf-8", newline="\n")
    (HERE / "candidate.apply_patch").write_text("".join(apply), encoding="utf-8", newline="\n")
    (HERE / "candidate_manifest.json").write_text(json.dumps({"status": "UNAPPLIED_NOT_MODEL_TESTED",
        "runtime_paths": sorted(k for k in records if k.startswith(CODE)), "files": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "STAGED_AST_PASS_NO_IMPORT_OR_MODEL", "runtime_paths": 3, "test_files": 1}))


if __name__ == "__main__":
    main()

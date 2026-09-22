"""Generate unapplied apply_patch-compatible routes/module/test patches only."""
from pathlib import Path
import difflib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
MODULE='src/wlr50_clean/ppo/semantic_p05_preedge_migration.py'


def replacement(text,before,after):
    assert text.count(before)==1,(before[:120],text.count(before))
    return text.replace(before,after,1)


def patched_sources():
    path='src/wlr50_clean/ppo/semantic_migration.py';text=(ROOT/path).read_text(encoding='utf-8')
    text=replacement(text,
        '    if supplied.get("schema") in ("wlr50_clean.rr_postcross_workspace_same389.v1",',
        '    if supplied.get("schema") == "wlr50_clean.p05_preedge_approach_recovery_same389.v1":\n'
        '        from .semantic_p05_preedge_migration import validate_p05_preedge_migration\n'
        '        return validate_p05_preedge_migration(checkpoint,current_contract,path,project_root=project_root)\n'
        '    if supplied.get("schema") in ("wlr50_clean.rr_postcross_workspace_same389.v1",')
    result={path:text}
    path='src/wlr50_clean/ppo/semantic_training.py';text=(ROOT/path).read_text(encoding='utf-8')
    text=replacement(text,
        '        infos = rr_workspace_branch_counts(infos)\n    assert_semantic_return_consistency(runner, runner.env)',
        '        infos = rr_workspace_branch_counts(infos)\n'
        '    if "p05_preedge_approach_recovery_branch" in infos:\n'
        '        from .semantic_p05_preedge_migration import p05_preedge_branch_counts\n'
        '        infos = p05_preedge_branch_counts(infos)\n'
        '    assert_semantic_return_consistency(runner, runner.env)')
    text=replacement(text,
        '        rr_workspace_factor = (verified.get("rr_postcross_workspace_factor") or {}).get("observation_contract")',
        '        rr_workspace_factor = (verified.get("rr_postcross_workspace_factor") or {}).get("observation_contract")\n'
        '        p05_preedge_factor = (verified.get("p05_preedge_approach_recovery_factor") or {}).get("observation_contract")')
    text=replacement(text,
        'receiving_factor, capture_feedback_factor, rr_workspace_factor)) > 1:',
        'receiving_factor, capture_feedback_factor, rr_workspace_factor, p05_preedge_factor)) > 1:')
    text=replacement(text,
        'or receiving_factor or capture_feedback_factor or rr_workspace_factor\n',
        'or receiving_factor or capture_feedback_factor or rr_workspace_factor or p05_preedge_factor\n')
    text=replacement(text,
        '            infos = record_loaded_rr_workspace(runner,infos,verified)\n        if receiving_wheel is not None:',
        '            infos = record_loaded_rr_workspace(runner,infos,verified)\n'
        '        if verified.get("p05_preedge_approach_recovery_factor") is not None:\n'
        '            from .semantic_p05_preedge_migration import record_loaded_p05_preedge\n'
        '            infos = record_loaded_p05_preedge(runner,infos,verified)\n'
        '        if receiving_wheel is not None:')
    text=replacement(text,
        '                            "rr_receiver_retirement_v2_migration", "rr_receiver_retirement_v2_branch"):',
        '                            "rr_receiver_retirement_v2_migration", "rr_receiver_retirement_v2_branch",\n'
        '                            "p05_preedge_approach_recovery_migration", "p05_preedge_approach_recovery_branch"):')
    result[path]=text
    return result


def patch_add(path,source):
    return ['*** Add File: '+path]+['+'+line for line in source.splitlines()]


def main():
    rows=['*** Begin Patch']
    rows+=patch_add(MODULE,(HERE/'semantic_p05_preedge_migration.py').read_text(encoding='utf-8'))
    for path,text in patched_sources().items():
        diff=list(difflib.unified_diff((ROOT/path).read_text(encoding='utf-8').splitlines(),text.splitlines(),n=3))[2:]
        rows.append('*** Update File: '+path)
        rows.extend('@@' if line.startswith('@@') else line for line in diff)
    rows.append('*** End Patch')
    (HERE/'migration.patch').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    test=HERE/'test_semantic_p05_preedge_migration.py'
    if test.exists():
        rows=['*** Begin Patch']+patch_add('tests/unit/test_semantic_p05_preedge_migration.py',test.read_text(encoding='utf-8'))+['*** End Patch']
        (HERE/'tests.patch').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    print('Unapplied migration.patch/tests.patch regenerated; no runtime files changed.')


if __name__=='__main__':main()

"""Generate a review-only unified patch; never apply it or mutate production."""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODULES = ('semantic_history_actor','semantic_policy_distribution','semantic_training',
    'semantic_migration','semantic_cli','semantic_checkpoint_prefix_policy')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--modules',nargs='+',choices=MODULES,default=list(MODULES))
    parser.add_argument('--name',default='candidate_unified')
    args=parser.parse_args()
    if Path(args.name).name != args.name or not args.name.replace('_','').isalnum():
        raise ValueError('output name must be a plain alphanumeric/underscore stem')
    paths=[f'src/wlr50_clean/ppo/{module}.py' for module in args.modules]
    subprocess.run(['git','diff','--quiet','HEAD','--',*paths],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!='f2e552406ea74a5aae2803347d1c8bebe261910d':
        raise RuntimeError('review patch baseline changed; do not silently rebase')
    output=HERE/(args.name+'.patch')
    receipt_path=HERE/(args.name+'_receipt.json')
    if output.exists() or receipt_path.exists():
        raise FileExistsError('review artifacts are immutable; choose a new name')
    diff=[]
    bindings={}
    for path in paths:
        original,candidate=ROOT/path,HERE/path
        before,after=original.read_bytes(),candidate.read_bytes()
        bindings[path]={'production_sha256':hashlib.sha256(before).hexdigest(),
            'candidate_sha256':hashlib.sha256(after).hexdigest()}
        diff.extend(difflib.unified_diff(before.decode().replace('\r\n','\n').splitlines(keepends=True),
            after.decode().replace('\r\n','\n').splitlines(keepends=True),
            fromfile='a/'+path,tofile='b/'+path,n=3))
    output.write_text(''.join(diff),encoding='utf8',newline='\n')
    checked=subprocess.run(['git','apply','--check',str(output)],cwd=ROOT,text=True,capture_output=True)
    receipt={'schema':'wlr50_clean.output_only_candidate_patch.v1','baseline_head':head,
        'patch':str(output),'patch_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'bindings':bindings,'git_apply_check_exit_code':checked.returncode,
        'git_apply_check_output':checked.stdout+checked.stderr,
        'production_modified':False,'patch_applied':False,'physics_started':False,
        'real_policy_decisions_added':0,'real_optimizer_steps_added':0}
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'patch':str(output),'sha256':receipt['patch_sha256'],
        'apply_check_exit_code':checked.returncode,'production_modified':False},indent=2))
    if checked.returncode:
        raise RuntimeError(checked.stdout+checked.stderr)


if __name__=='__main__':
    main()

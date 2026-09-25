"""Mechanical isolated single-file rewrite; refuses source drift/overwrites."""
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REL = 'src/wlr50_clean/ppo/semantic_reward.py'
SOURCE_SHA = 'b2ed7047870a776c6c0102576300559565fe18ee4581d601690df42428ecabde'


def once(text, old, new):
    assert text.count(old) == 1, f'unique source anchor required: {old[:80]}'
    return text.replace(old,new,1)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--revision',default='v2',choices=('v2','v3','v4'))
    args=parser.parse_args()
    src=(ROOT/REL).read_bytes()
    assert hashlib.sha256(src).hexdigest()==SOURCE_SHA, 'frozen reward source drift'
    original=src.decode('utf-8')
    assert '\r\n' not in original, 'mechanical patch expects current LF source'
    additions=(HERE/'semantic_reward_additions.py.txt').read_text(encoding='utf-8')
    updated=once(original,'@dataclass(frozen=True)\nclass SemanticRewardConfig:',
        additions+'@dataclass(frozen=True)\nclass SemanticRewardConfig:')
    updated=once(updated,'        self.config = config or load_semantic_reward_config()\n        self.reset()',
        '        self.config = config or load_semantic_reward_config()\n'
        '        self._rr_retention_binding = _rr_retention_reward_binding(self.config)\n        self.reset()')
    updated=once(updated,
        '        phi_before = finite(previous.task["task_progress_potential"],"potential before")\n'
        '        phi_after = 0.0 if termination_reason else finite(current.task["task_progress_potential"],"potential after")',
        '        phi_before, phi_after, rr_potential_audit = _reward_only_potential_pair(\n'
        '            previous, current, termination_reason, self._rr_retention_binding)')
    updated=once(updated,'        if front_quality:\n            result["front_quality_sample_audit"]',
        '        if rr_potential_audit is not None:\n'
        '            result["potential_semantics"] = RR_RETENTION_REWARD_ONLY_MODE\n'
        '            result["rr_retention_reward_only"] = rr_potential_audit\n'
        '        if front_quality:\n            result["front_quality_sample_audit"]')
    ast.parse(updated)
    output=HERE/args.revision
    output.mkdir(exist_ok=True)
    target=output/'semantic_reward.py'
    patch=output/'semantic_reward.patch'
    receipt=output/'reward_patch_identity.json'
    assert not any(p.exists() for p in (target,patch,receipt)), 'preserve staged artifacts'
    out=updated.encode('utf-8')
    diff=''.join(difflib.unified_diff(original.splitlines(True),updated.splitlines(True),
        fromfile='a/'+REL,tofile='b/'+REL))
    target.write_bytes(out)
    patch.write_text(diff,encoding='utf-8',newline='\n')
    identity=dict(relative_path=REL,source_sha256=SOURCE_SHA,target_sha256=hashlib.sha256(out).hexdigest(),
        patch_sha256=hashlib.sha256(diff.encode()).hexdigest(),mode='rr_recapture_retention_reward_only_v1',
        changes_production=False,configuration_files_changed=[],observation_or_control_changed=False,
        binding='cooperative v5; exact immutable adjacent stage spec scales loaded once at calculator construction')
    receipt.write_text(json.dumps(identity,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(identity,indent=2))


if __name__=='__main__': main()

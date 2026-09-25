"""Mechanical Add File patch, outputs only. Do not import the unit test here."""
import ast
import difflib
from pathlib import Path

HERE=Path(__file__).resolve().parent
REL='tests/unit/test_semantic_rr_retention_reward_only.py'
source=(HERE/REL).read_text(encoding='utf-8')
ast.parse(source)
unified=''.join(difflib.unified_diff([],source.splitlines(True),fromfile='/dev/null',tofile='b/'+REL))
codex='*** Begin Patch\n*** Add File: '+REL+'\n'+''.join('+'+line for line in source.splitlines(True))+'*** End Patch\n'
for name,text in [('reward_unit_tests_v2.patch',unified),('reward_unit_tests_v2.apply_patch',codex)]:
    path=HERE/name
    assert not path.exists(), 'preserve staged patch'
    path.write_text(text,encoding='utf-8',newline='\n')
print('Unit test AST parsed; normal production imports intentionally NOT executed while Isaac is active.')

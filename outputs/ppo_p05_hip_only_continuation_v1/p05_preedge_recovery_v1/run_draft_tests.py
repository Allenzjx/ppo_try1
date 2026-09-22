"""CPU process-only installed-source simulation; production remains untouched."""
import importlib
import linecache
from pathlib import Path
import sys
import types

from build_candidate_patches import HERE,ROOT,MODULE,patched_sources


def main():
    import torch
    assert not torch.cuda.is_available();torch.set_num_threads(1)
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'tests/unit'))
    replacements=patched_sources()
    modules={p:importlib.import_module('wlr50_clean.ppo.'+Path(p).stem) for p in replacements}
    for path,text in replacements.items():
        filename=str(ROOT/path);linecache.cache[filename]=(len(text),None,text.splitlines(True),filename)
        exec(compile(text,filename,'exec'),modules[path].__dict__)
    name='wlr50_clean.ppo.semantic_p05_preedge_migration'
    module=types.ModuleType(name);module.__package__='wlr50_clean.ppo';module.__file__=str(HERE/'semantic_p05_preedge_migration.py')
    sys.modules[name]=module
    exec(compile(Path(module.__file__).read_text(encoding='utf-8'),module.__file__,'exec'),module.__dict__)
    import pytest
    raise SystemExit(pytest.main([str(HERE/'test_semantic_p05_preedge_migration.py'),'-q','-p','no:cacheprovider']))


if __name__=='__main__':main()

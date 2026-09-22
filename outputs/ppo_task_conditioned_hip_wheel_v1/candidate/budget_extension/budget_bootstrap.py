"""Process-local candidate imports; no production writes or native simulation."""
from pathlib import Path
import importlib.abc
import importlib.machinery
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = HERE / 'src/wlr50_clean/ppo'
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests/unit'))


class CandidateLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        module.__file__ = str(ROOT / 'src/wlr50_clean/ppo' / (module.__name__.rsplit('.', 1)[-1] + '.py'))
        super().exec_module(module)


class CandidateFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('wlr50_clean.ppo.'):
            source = SOURCE / (fullname.rsplit('.', 1)[-1] + '.py')
            if source.is_file():
                return importlib.util.spec_from_file_location(fullname, source,
                    loader=CandidateLoader(fullname, str(source)))
        return None


assert not any(n.startswith('wlr50_clean.ppo.') for n in sys.modules), 'bootstrap must run before PPO imports'
sys.meta_path.insert(0, CandidateFinder())

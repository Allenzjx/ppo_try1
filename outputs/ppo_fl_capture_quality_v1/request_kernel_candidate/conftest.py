"""Process-local CPU candidate overlay, never a production-file modification."""
from pathlib import Path
import sys
import wlr50_clean.ppo

CANDIDATE = Path(__file__).resolve().parent / 'src/wlr50_clean/ppo'
for name in ('semantic_history_actor','semantic_policy_distribution','semantic_training',
             'semantic_migration','semantic_cli','semantic_checkpoint_prefix_policy'):
    if 'wlr50_clean.ppo.'+name in sys.modules:
        raise RuntimeError('candidate tests require a fresh process before policy modules import')
wlr50_clean.ppo.__path__.insert(0, str(CANDIDATE))


def pytest_configure(config):
    # Only resource routing for copied __file__: preserve the actual loader and
    # actual read-only old reward bytes. No copied config or changed semantics.
    from wlr50_clean.ppo import semantic_reward
    original = semantic_reward.load_semantic_reward_config
    candidate_root = Path(__file__).resolve().parent
    repository = candidate_root.parents[2]
    missing = candidate_root/'configs/ppo_semantic_v3/reward_config.yaml'
    actual = repository/'configs/ppo_semantic_v3/reward_config.yaml'

    def mapped_resource_load(*args, **kwargs):
        values = list(args)
        if values and Path(values[0]).resolve() == missing:
            values[0] = actual
        return original(*values, **kwargs)

    semantic_reward.load_semantic_reward_config = mapped_resource_load

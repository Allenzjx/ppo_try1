"""Cold CPU tests: actual sealed source migration, no new physical credit."""
import json
from pathlib import Path
import pytest
import torch
from wlr50_clean.ppo import semantic_rr_capture_local as route
from wlr50_clean.ppo.rl_library_wrapper import capture_training_rng_state
from wlr50_clean.ppo.semantic_training import state_hash


def test_actual_complete_checkpoint_migration_and_reload(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    source = route.OUTPUT/'checkpoints/history/checkpoint_CP227328_local002048.pt'
    original_sha = route.sha(source)
    runner = route.make_runner('cpu',1001)
    prior,counts = route.migrate_checkpoint(runner,source)
    assert route.sha(source) == original_sha == route.LEGACY_CHECKPOINT_SHA
    assert counts['local_policy_decisions']==2048
    assert counts['local_ppo_updates']==4 and counts['local_optimizer_steps']==80
    assert counts['task_v2_policy_decisions']==counts['task_v2_ppo_updates']==0
    assert counts['prefix_decisions']==4986 and counts['auxiliary_updates']==0
    assert runner.alg.learning_rate == pytest.approx(.0000225)
    assert runner.alg.optimizer.param_groups[0]['lr']==runner.alg.learning_rate
    assert max(runner.local_migration['actual_observation_comparison_max_abs'].values()) < 2e-6
    assert runner.alg.storage.step==0 and runner.alg.transition.actions is None
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)
    metadata=json.loads(source.with_name(source.stem+'_manifest.json').read_text())
    assert capture_training_rng_state(seed=1001)==metadata['training_rng']
    with pytest.raises(ValueError,match='contract/hash mismatch'):
        route.load(runner,source,{'test_only':True})
    monkeypatch.setattr(route,'OUTPUT',tmp_path/'temporary_publication')
    expected=state_hash(runner.alg.save())
    pointer=route.save(runner,{'test_only':True},prior,counts,source_run=tmp_path)
    restored=route.make_runner('cpu',1001)
    p,c=route.load(restored,pointer['checkpoint'],{'test_only':True})
    assert p==prior and c==counts
    assert restored.local_migration==runner.local_migration
    assert state_hash(restored.alg.save())==expected
    assert capture_training_rng_state(seed=1001)==metadata['training_rng']
    assert Path(pointer['checkpoint']).name.endswith('_lineage448_v2.pt')

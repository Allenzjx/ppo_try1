"""Cold actual CP232960 migration; only temporary saves, no physical/PPO credit."""
import copy
import hashlib
import json

import pytest
import torch
from tensordict import TensorDict

from wlr50_clean.ppo import semantic_post_rr_front_prep as route
from wlr50_clean.ppo.rl_library_wrapper import capture_training_rng_state,restore_training_rng_state
from wlr50_clean.ppo.semantic_training import state_hash


SOURCE = route.ROOT/'outputs/ppo_post_rr_front_pair_rl_v1/checkpoints/history/checkpoint_CP232960_postRR000512_g475eb1f4c572.pt'


@pytest.fixture(autouse=True)
def preserve_rng():
    old=capture_training_rng_state(seed=1001)
    threads=torch.get_num_threads();torch.set_num_threads(1)
    yield
    restore_training_rng_state(old,expected_seed=1001)
    torch.set_num_threads(threads)


def metadata():
    return json.loads(SOURCE.with_name(SOURCE.stem+'_manifest.json').read_text())


def candidate_runtime():
    runtime=copy.deepcopy(metadata()['runtime_contract'])
    runtime.update(source_git_commit='f'*40,local_contract=route.settings())
    for name in route.FINISH_CHANGED_FILES:
        runtime['files'][name]=route.sha(route.ROOT/name)
    runtime['runtime_content_sha256']=hashlib.sha256(json.dumps(runtime['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    local=route.CONFIG/'local_training.json'
    runtime['selected_configuration']['local_training.json']['sha256']=route.sha(local)
    return runtime


def loaded_source():
    meta=metadata()
    runner=route.make_runner('cpu',saved_configuration=meta['runner_config'],legacy_finish_source=meta)
    route._load_checkpoint(runner,SOURCE,meta['runtime_contract'],expected_settings=meta['runtime_contract']['local_contract'])
    return runner


def tensor_inputs(dimension,*,inactive=False):
    rollout=torch.load(route.ROOT/'runs/ppo_post_rr_front_pair_rl_v1/RL_T1_train512_475eb1f/rollouts/rollout_0001.pt',
                       map_location='cpu',weights_only=False)
    x=rollout['observations']['policy'].reshape(-1,487)[[0,10,44,142,303,419,464,511]].clone()
    if inactive:
        x[:,465:]=0.  # Counterfactual inactive routing test, not new physical data.
    if dimension==490:
        x=torch.cat((x,torch.zeros(len(x),3)),dim=-1)
    return TensorDict({'policy':x,'critic':x.clone()},batch_size=[len(x)])


def test_actual_actor_critic_Adam_RNG_zero_column_migration_and_strict_reload(tmp_path,monkeypatch):
    old=loaded_source()
    source_hash=state_hash(old.alg.save())
    current=route.make_runner('cpu')
    runtime=candidate_runtime()
    lineage,counts=route.migrate_finish(current,SOURCE,runtime)
    assert counts==metadata()['counts']
    assert current.alg.learning_rate==metadata()['learning_rate']==1.e-5
    assert capture_training_rng_state(seed=1001)==metadata()['training_rng']
    assert current.alg.storage.step==0 and current.alg.transition.actions is None
    event=current.finish_migrations[-1]
    assert event['new_PPO_decisions']==event['new_Adam_steps']==event['new_AUX_updates']==0
    assert event['old_state_exact_after_zero_column_collapse'] and event['all_old_Adam_steps_preserved']
    assert len(event['expanded_model_parameters'])==len(event['expanded_Adam_moments'])==2
    planned,_=route.expand_finish_training_state(old,current)
    assert state_hash(planned)==state_hash(current.alg.save())
    assert state_hash(old.alg.save())==source_hash
    assert state_hash(current.alg.actor.frozen_anchor.state_dict())==state_hash(old.alg.actor.frozen_anchor.state_dict())
    for inactive in (False,True):
        before=tensor_inputs(487,inactive=inactive)
        after=tensor_inputs(490,inactive=inactive)
        rng=capture_training_rng_state(seed=1001)
        with torch.inference_mode():
            original=old.alg.actor(before,stochastic_output=False)
            migrated=current.alg.actor(after,stochastic_output=False)
            old_v=old.alg.critic(before)
            new_v=current.alg.critic(after)
        if inactive:
            assert torch.equal(original,migrated)
        else:
            assert torch.allclose(original,migrated,atol=2.e-6,rtol=0.)
        assert torch.allclose(old_v,new_v,atol=2.e-5,rtol=0.)
        assert capture_training_rng_state(seed=1001)==rng
    # Existing490 observations can reveal the finish state without altering the
    # just-migrated mean: those columns are exactly zero, not hidden controller state.
    after=tensor_inputs(490)
    with torch.inference_mode():
        baseline=current.alg.actor(after,stochastic_output=False)
        after['policy'][:,-3:]=torch.tensor([1.,1.,.7])
        assert torch.equal(current.alg.actor(after,stochastic_output=False),baseline)
    monkeypatch.setattr(route,'OUTPUT',tmp_path/'checkpoints')
    complete=state_hash(current.alg.save())
    pointer=route.save(current,runtime,lineage,counts,source_run=tmp_path)
    expected=torch.rand(5)
    saved=json.loads(open(pointer['manifest']).read())
    restored=route.make_runner('cpu',saved_configuration=saved['runner_config'])
    restored_lineage,restored_counts=route.load(restored,pointer['checkpoint'],runtime)
    assert torch.equal(torch.rand(5),expected)
    assert restored_lineage==lineage and restored_counts==counts
    assert restored.finish_migrations==current.finish_migrations
    assert state_hash(restored.alg.save())==complete
    restored.alg.actor.assert_frozen_state(restored.alg.optimizer)


def test_ordinary_load_stays_strict_and_rejects_legacy_or_unrelated_changes():
    current=route.make_runner('cpu')
    runtime=candidate_runtime()
    with pytest.raises(ValueError,match='contract/hash/ledger'):
        route.load(current,SOURCE,runtime)
    changed=copy.deepcopy(runtime)
    changed['local_contract']['gamma']=.95
    with pytest.raises(ValueError,match='task/reward/profile'):
        route.validate_finish_migration_runtime(metadata()['runtime_contract'],changed)
    changed=copy.deepcopy(runtime)
    changed['files']['src/wlr50_clean/assets/unrelated_physics.py']='0'*64
    with pytest.raises(ValueError,match='unrelated runtime'):
        route.validate_finish_migration_runtime(metadata()['runtime_contract'],changed)
    wrong=copy.deepcopy(metadata());wrong['checkpoint_sha256']='0'*64
    with pytest.raises(ValueError,match='pinned complete487'):
        route.make_runner('cpu',saved_configuration=wrong['runner_config'],legacy_finish_source=wrong)


def test_expansion_rejects_unrelated_parameter_shapes_and_missing_Adam_state():
    old=loaded_source();new=route.make_runner('cpu')
    source=old.alg.actor.state_dict();target=new.alg.actor.state_dict()
    target=copy.deepcopy(target)
    target['mlp.2.weight']=target['mlp.2.weight'][:1]
    with pytest.raises(ValueError,match='unexpected finish model shape'):
        route._zero_expand_state(source,target)
    old.alg.optimizer.state.clear()
    with pytest.raises(ValueError,match='missing actual trained Adam state'):
        route.expand_finish_training_state(old,new)

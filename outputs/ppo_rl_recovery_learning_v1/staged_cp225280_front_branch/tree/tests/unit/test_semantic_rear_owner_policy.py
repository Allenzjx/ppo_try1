"""Bounded CPU439 append/likelihood/full-state tests; zero robot learning credit."""
import copy
import json
from pathlib import Path
import pytest
import torch
from tensordict import TensorDict
from wlr50_clean.ppo import semantic_training as t
from wlr50_clean.ppo import semantic_rear_owner_migration as m
from wlr50_clean.ppo.semantic_rear_owner_profile import *
from wlr50_clean.ppo.semantic_rear_owner_actor import rear_owner_effective_log_std
from wlr50_clean.ppo.semantic_rear_cooperative_prep_sigma import cooperative_prep_effective_log_std
from wlr50_clean.ppo.semantic_rear_cooperative_prep_profile import COOPERATIVE_PREP_POLICY, COOPERATIVE_PREP_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
from wlr50_clean.ppo.semantic_migration import digest, file_sha

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000225280.pt'


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(state); torch.set_num_threads(threads)


def make(width):
    return t.construct_semantic_runner(_shape_env(width,'cpu'),seed=1001,device='cpu',initialize_actor=False,
        policy_version=COOPERATIVE_PREP_POLICY if width == 422 else REAR_OWNER_POLICY,
        observation_layout=COOPERATIVE_PREP_OBSERVATION_LAYOUT if width == 422 else REAR_OWNER_OBSERVATION_LAYOUT)[0]


def obs(x): return TensorDict({'policy':x,'critic':x.clone()},batch_size=[x.shape[0]])


def public(active=False):
    return dict(anchor_final_deg=[-12.,-40.,20.,5.] if active else [0.]*4,
        anchor_request_deg=[-2.,-8.,3.,4.] if active else [0.]*4,
        active=[active]*4, winning_late_owner=[False]*4, rl_edge_recovery_permitted=active)


@pytest.mark.parametrize('phase',[1,8,11])
def test_zero_append_mean_critic_and_shared_sigma(phase):
    source, target = make(422), make(439)
    for role in ('actor','critic'):
        state = copy.deepcopy(getattr(source.alg,role).state_dict())
        state['mlp.0.weight'] = torch.cat((state['mlp.0.weight'],torch.zeros(256,17)),dim=1)
        getattr(target.alg,role).load_state_dict(state)
    x = torch.zeros(2,439); x[:,phase] = 1.; x[:,20] = .02; x[:,158:158+phase] = 1.
    x[:,195:207] = torch.linspace(-.3,.3,12)
    if phase == 1: x[:,419:422] = torch.tensor([.03,.8,1.])
    if phase == 8: x[:,410] = x[:,404] = 1.
    if phase == 11: x[:,412] = 1.
    x[:,422:] = torch.tensor(rear_owner_features(public(True)))
    with torch.no_grad():
        for role in ('actor','critic'):
            torch.testing.assert_close(getattr(source.alg,role)(obs(x[:,:422])),getattr(target.alg,role)(obs(x)),atol=3e-7,rtol=3e-6)
        head = target.alg.actor.mlp(x)[...,1,:]
        assert torch.equal(rear_owner_effective_log_std(head,x)[0],cooperative_prep_effective_log_std(head,x[:,:422])[0])
    assert list(target.alg.actor.state_dict()) == list(source.alg.actor.state_dict())


@pytest.mark.parametrize('stochastic',[False,True])
def test_actual_raw_request_audit_and_frozen_prefix(stochastic):
    from wlr50_clean.ppo.semantic_checkpoint_prefix_policy import FrozenCheckpointPrefixPolicy
    runner = make(439); actor = runner.alg.actor
    x = torch.zeros(1,439); x[:,11] = 1.; x[:,20] = .03; x[:,412] = 1.
    x[:,422:] = torch.tensor(rear_owner_features(public(True)))
    with torch.no_grad():
        raw, receipt = t.audited_history_policy_request(actor,obs(x),lambda:actor(obs(x),stochastic_output=stochastic),stochastic=stochastic)
    assert receipt['policy_version'] == REAR_OWNER_POLICY
    assert receipt['rear_owner_observed_features'] == x[0,422:].tolist()
    assert receipt['sampling_draws'] == int(stochastic) and receipt['extra_model_forwards'] == 0
    assert receipt['selected_raw_full12'] == raw[0].tolist()
    policy = t._runner_policy_contract(runner)
    record = dict(checkpoint_path='synthetic.pt',checkpoint_sha256='a'*64,actor_parameter_sha256=t.parameter_hash(actor),
        source_global_policy_decisions=0,source_ppo_updates=0,policy_contract=policy,source_policy_contract=policy,
        effective_policy_contract=policy,source_runtime_content_sha256='b'*64,effective_runtime_content_sha256='b'*64)
    prefix = FrozenCheckpointPrefixPolicy(actor,record)
    assert len(prefix(tuple(x[0].tolist()))) == 12


def test_public_pending_release_and_inactive_anchors():
    assert len(rear_owner_features(public(True))) == 17 # active old dispatch + new owner false is legal.
    for key, value in [('active',[1]*4),('anchor_final_deg',[float('nan')]*4),('rl_edge_recovery_permitted',1)]:
        row = public(); row[key] = value
        with pytest.raises(ValueError): rear_owner_features(row)
    row = public(); row['anchor_final_deg'][0] = 1.
    with pytest.raises(ValueError): rear_owner_features(row)


def test_schema_exact_old422_prefix_and_missing_marker(tmp_path):
    from wlr50_clean.ppo.semantic_observation import load_semantic_observation_schema
    data = json.loads((ROOT/'configs/ppo_rr_rl_timing_policy_learning_v1/observation_schema.json').read_text())
    path = tmp_path/'schema.json'; path.write_text(json.dumps(data))
    schema = load_semantic_observation_schema(path)
    assert schema.dimension == 439 and schema.observation_layout == REAR_OWNER_OBSERVATION_LAYOUT
    old = json.loads(__import__('subprocess').check_output(['git','show',m.SOURCE_HEAD+':configs/ppo_rr_rl_timing_policy_learning_v1/observation_schema.json'],cwd=ROOT,text=True))
    assert list(schema.groups[:-1]) == old['feature_groups']
    data.pop('rear_owner_recovery_features_version'); path.write_text(json.dumps(data))
    with pytest.raises(ValueError): load_semantic_observation_schema(path)


def test_actual_source_build_and_precise_ancestry_negative(monkeypatch):
    # This is the historical422->f6d439 migration, not a migration to whatever
    # runtime happens to be checked out now (e.g. the later512 collector).
    import hashlib
    from wlr50_clean.ppo import semantic_migration as migration_module
    source = json.loads(SOURCE.with_name(SOURCE.stem+'_manifest.json').read_text())
    historical_sidecar = SOURCE.with_name('checkpoint_rear_owner_CP225280_gf6d1d2df8d87_manifest.json')
    published = json.loads(historical_sidecar.read_text())
    current = copy.deepcopy(published['runtime_contract'])
    target_head = 'f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b'
    assert current['source_git_commit'] == target_head
    assert file_sha(historical_sidecar) == 'f501b7a735c0aaa42be00114e09f80d46f7f009d7717aa416fc30ad3cc837d6d'
    actual_file_sha, historical_hashes = migration_module.file_sha, {}

    def historical_target_file_sha(path):
        # Mock only the runtime-byte accessor. Each returned hash is calculated
        # from actual immutable git bytes with the existing manifest-matched
        # LF/CRLF reconstruction. Checkpoints/manifests and paths outside that inventory
        # retain normal live-file hashing. No lineage validator is mocked.
        path = Path(path).resolve()
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            return actual_file_sha(path)
        if relative not in current['files']:
            return actual_file_sha(path)
        if relative not in historical_hashes:
            # An unchanged file may retain historical mixed Windows line ends.
            # Accept its actual bytes only when the published hash matches.
            blob = (path.read_bytes() if actual_file_sha(path) == current['files'][relative]
                    else migration_module._version_bytes(ROOT, current, relative, prefer_worktree=False))
            measured = hashlib.sha256(blob).hexdigest()
            assert measured == current['files'][relative]
            historical_hashes[relative] = measured
        return historical_hashes[relative]

    monkeypatch.setattr(migration_module, 'file_sha', historical_target_file_sha)
    record = m.build_rear_owner_migration(SOURCE,current,reason='historical f6d target contract test, not publication')
    factor = record[m.FACTOR_KEY]
    assert m._previous_contract(current,record) == source['runtime_contract']
    assert set(historical_hashes) == set(current['files'])
    migrated = copy.deepcopy(source)
    migrated.update(runtime_contract=current,policy_contract=factor['target_policy_contract'],runner_config=factor['target_runner_config'],**{m.MIGRATION:record})
    route = source['checkpoint_output_routing']
    assert m.validate_rear_owner_lineage(migrated,current,Path(route['output_root']),checkpoint_output_routing=route) == record
    bad = copy.deepcopy(migrated); bad['cooperative_prep_migration']['reason'] += ' changed'
    with pytest.raises(ValueError): m.validate_rear_owner_lineage(bad,current,Path(route['output_root']),checkpoint_output_routing=route)
    with pytest.raises(ValueError): m.validate_rear_owner_lineage(migrated,current,ROOT/'outputs/wrong',checkpoint_output_routing=route)


def test_full_adam_official_append_save_reload_and_receipt_carry(tmp_path,monkeypatch):
    source = make(422)
    for p in list(source.alg.actor.parameters())+list(source.alg.critic.parameters()): p.grad = torch.ones_like(p)
    source.alg.optimizer.step(); source.alg.optimizer.zero_grad()
    for state in source.alg.optimizer.state.values(): state['step'].fill_(34500.)
    source.alg.learning_rate = 1e-5
    for group in source.alg.optimizer.param_groups: group['lr'] = 1e-5
    source.current_learning_iteration = 1725
    base = json.loads(SOURCE.with_name(SOURCE.stem+'_manifest.json').read_text())
    for key in ('checkpoint_path','checkpoint_sha256','save_load_round_trip'): base.pop(key,None)
    source_path, side = t.save_semantic_checkpoint(source,tmp_path/'synthetic_source.pt',base)
    old = json.loads(side.read_text()); current = copy.deepcopy(old['runtime_contract'])
    current.update(source_git_commit='f'*40,runtime_content_sha256='e'*64)
    target = make(439)
    record = dict(schema=m.SCHEMA,plan_path=str(tmp_path/'synthetic_plan.json'),
        **{m.FACTOR_KEY:dict(schema=m.SCHEMA,source_policy_contract=old['policy_contract'],target_policy_contract=t._runner_policy_contract(target),
            source_runner_config=source._semantic_runner_config,target_runner_config=target._semantic_runner_config,
            source_effective_learning_rate=1e-5,preserved_metadata_sha256={k:digest(old[k]) for k in m._preserved(old)})})
    monkeypatch.setattr(m,'validate_rear_owner_migration',lambda *a,**kw:record)
    mapped = m.zero_append_rear_owner_training_state(torch.load(source_path,weights_only=False))
    infos = t.load_semantic_checkpoint(target,source_path,contract=current,seed=1001,migration=record)
    for role in ('actor','critic'):
        state = getattr(target.alg,role).state_dict()
        assert t.state_hash(state) == t.state_hash(mapped[role+'_state_dict'])
        assert torch.equal(state['mlp.0.weight'][:,:422],getattr(source.alg,role).state_dict()['mlp.0.weight'])
        assert torch.count_nonzero(state['mlp.0.weight'][:,422:]) == 0
    assert t.state_hash(target.alg.optimizer.state_dict()) == t.state_hash(mapped['optimizer_state_dict'])
    for ident in (0,6):
        for key in ('exp_avg','exp_avg_sq'): assert torch.count_nonzero(mapped['optimizer_state_dict']['state'][ident][key][:,422:]) == 0
    assert t.capture_training_rng_state(seed=1001) == old['training_rng_state']
    assert target.alg.storage.step == 0 and target.alg.storage.observations['policy'].shape == (128,1,439)
    saved, manifest = t.save_semantic_checkpoint(target,tmp_path/'synthetic_migrated.pt',infos)
    fresh = make(439); loaded = t.load_semantic_checkpoint(fresh,saved,contract=current,seed=1001)
    for key in m._preserved(old): assert loaded[key] == old[key]
    assert t.state_hash(fresh.alg.optimizer.state_dict()) == t.state_hash(mapped['optimizer_state_dict'])
    again, side = t.save_semantic_checkpoint(fresh,tmp_path/'synthetic_normal_save.pt',loaded)
    carried = json.loads(side.read_text())
    assert carried[m.MIGRATION] == record and carried['cooperative_prep_migration'] == old['cooperative_prep_migration']
    assert {k:carried[k] for k in m.COUNTERS} == m.REVISION_ORIGIN


def test_one_synthetic439_update_raw_likelihood_and_five_exposures(tmp_path):
    from collections import Counter
    from test_semantic_rear_policy_training_audit import Synthetic419Core
    class Synthetic439Core(Synthetic419Core):
        def observation(self,raw=(0.,)*12): return super().observation(raw)+(0.,)*20
    env = t.SemanticRslAdapter(Synthetic439Core(),seed=1001,device='cpu'); env.cfg['semantic_version'] = 'v3'
    runner,_ = t.construct_semantic_runner(env,seed=1001,device='cpu',initialize_actor=False,
        policy_version=REAR_OWNER_POLICY,observation_layout=REAR_OWNER_OBSERVATION_LAYOUT)
    contract = dict(experiment_id=m.EXPERIMENT,semantic_version='v3',training_budgets=t.training_quantity_budgets(m.EXPERIMENT),evidence='synthetic CPU only')
    result = t.train_semantic(runner,env,run_dir=tmp_path/'run',output_root=tmp_path/'out',stage='full_episode',
        decisions=128,contract=contract,seed=1001,checkpoint_interval_updates=1)
    assert (result['actual_policy_decisions'],result['ppo_updates_this_run'],result['optimizer_steps_this_run']) == (128,1,20)
    rollout = torch.load(tmp_path/'run/rollouts/rollout_000001.pt',map_location='cpu',weights_only=False)
    assert rollout['observations']['policy'].shape == (128,1,439)
    rows = [json.loads(line) for line in (tmp_path/'run/residual_and_projection_audit.jsonl').read_text().splitlines()]
    for i,row in enumerate(rows):
        request = row['policy_request']
        assert request['rear_owner_observed_features'] == rollout['observations']['policy'][i,0,422:].tolist()
        assert request['selected_raw_full12'] == rollout['actions'][i,0].tolist() == row['raw_policy_action_full12']
        assert request['selected_raw_log_probability'] == rollout['actions_log_prob'][i,0].item() == row['old_log_probability']
        assert request['effective_sigma_full12'] == rollout['distribution_params'][1][i,0].tolist()
    audit = next(json.loads(path.read_text()) for path in (tmp_path/'run').rglob('*likelihood*.json'))
    assert len(audit['minibatches']) == 20
    exposure = Counter(indices[0] for batch in audit['minibatches'] for indices in batch['rollout_flat_indices'])
    assert exposure == Counter({i:5 for i in range(128)})

"""Task-first .5 -> .25 sampling-only boundary; synthetic Git and real CPU PPO.

No user checkpoint and no Isaac. Success-route migration validation is real;
only the project-root routing is adapted to the isolated Git fixture.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import semantic_migration as m, semantic_training as t, semantic_cli as cli
from wlr50_clean.ppo.semantic_policy_distribution import (
    HISTORY_TEMPERED_POLICY, HISTORY_QUARTER_TEMPERED_POLICY, policy_contract,
)
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_semantic_role_observation_migration import NAMES, PROTECTED, commit, git, source_metadata, write_json

CONFIG = 'configs/ppo_task_first_recovery_v1'


def bind(root, contract):
    contract['files'] = {p: m.file_sha(root / p) for p in contract['files']}
    contract['runtime_content_sha256'] = m.digest(contract['files'])
    contract['selected_configuration'] = {n: {'path': f'{CONFIG}/{n}',
        'sha256': contract['files'][f'{CONFIG}/{n}']} for n in NAMES}


@pytest.fixture(scope='module')
def quarter_prototype(tmp_path_factory):
    root = tmp_path_factory.mktemp('quarter_temperature_git')
    paths = m.EXPLORATION_TEMPERATURE_FILES | {f'{CONFIG}/{n}' for n in NAMES} | {PROTECTED}
    for path in paths:
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((m.PROJECT_ROOT / path).read_bytes())
    prefix = 'src/wlr50_clean/ppo/'
    modules = {
        prefix+'semantic_history_actor.py': 'LOCKED = 1\ndef history_conditioned_head():\n    return 9\nclass SemanticHistoryMLPModel:\n    pass\nclass SemanticTemperedHistoryMLPModel:\n    def half_kernel(self):\n        return 0.5\n',
        prefix+'semantic_policy_distribution.py': "LOCKED = 1\nHISTORY_TEMPERED_POLICY = 'half'\nHISTORY_TEMPERED_ACTOR_CLASS = 'halfclass'\ndef policy_contract():\n    return 1\ndef configure_policy_distribution():\n    return 1\ndef supported_heteroscedastic_contract_version():\n    return 1\ndef policy_version_from_metadata():\n    return 1\n",
        prefix+'semantic_training.py': 'LOCKED = 1\ndef semantic_runner_config():\n    return 1\ndef _validated_exploration_temperature_factor():\n    return 1\ndef load_semantic_checkpoint():\n    return 9\ndef train_semantic():\n    return 9\n',
        prefix+'semantic_cli.py': 'LOCKED = 1\ndef _preflight_checkpoint():\n    return 1\ndef _resolved_policy_version():\n    return 1\ndef train():\n    return 9\n',
    }
    for path, text in modules.items():
        (root/path).write_text(text, encoding='utf8')
    git(root, 'init')
    git(root, 'config', 'user.name', 'Quarter CPU fixture')
    git(root, 'config', 'user.email', 'quarter@example.invalid')
    old = {'source_git_commit': commit(root, 'unchanged half-temperature source'),
        'files': dict.fromkeys(sorted(paths)), 'semantic_version': 'v3', 'experiment_id': 'task_first_recovery_v1',
        'frozen_A_files': {'frozen': 'a'*64}, 'physics_hz': 120., 'decision_hz': 15.,
        'task_timeout_s': 200., 'timeout_bootstrap': False, 'rsl_rl_version': '5.0.1',
        'local_runtime_versions': {'fixture': 'CPU only'}}
    bind(root, old)
    for path, text in modules.items():
        if path.endswith('semantic_history_actor.py'):
            text += 'class SemanticQuarterTemperedHistoryMLPModel:\n    pass\n'
        elif path.endswith('semantic_policy_distribution.py'):
            text = text.replace('return 1', 'return 2')
            text += "HISTORY_QUARTER_TEMPERED_POLICY = 'quarter'\nHISTORY_QUARTER_TEMPERED_ACTOR_CLASS = 'quarterclass'\n"
        elif path.endswith('semantic_training.py'):
            text = text.replace('return 1', 'return 2')
        # CLI, old .5 actor and raw HISTORY stay byte-identical.
        (root/path).write_text(text, encoding='utf8')
    new = copy.deepcopy(old)
    new['source_git_commit'] = commit(root, 'quarter-only candidate')
    bind(root, new)
    return SimpleNamespace(root=root, old=old, new=new)


@pytest.fixture
def q(quarter_prototype, tmp_path):
    root = tmp_path/'case'
    shutil.copytree(quarter_prototype.root, root)
    checkpoint = root/'outputs/ppo_task_first_recovery_v1/checkpoints/history/source.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'opaque synthetic sidecar fixture, never tensor-loaded')
    metadata = source_metadata(checkpoint, quarter_prototype.old,
        layout=ROLE_OBSERVATION_LAYOUT, policy=HISTORY_TEMPERED_POLICY)
    sidecar = checkpoint.with_name('source_manifest.json')
    write_json(sidecar, metadata)
    return SimpleNamespace(root=root, old=copy.deepcopy(quarter_prototype.old), new=copy.deepcopy(quarter_prototype.new),
        checkpoint=checkpoint, metadata=metadata, sidecar=sidecar)


def record(f, **options):
    delta = sorted(p for p in f.old['files'] if f.old['files'][p] != f.new['files'][p])
    return m.build_migration_plan(f.checkpoint, f.new, allowed_changed_files=delta,
        reason='Independent .5 to .25 task-first sampling candidate, not task success',
        exploration_temperature_review={'reason': 'Quarter scale applies to all stochastic consumers only',
            'reviewed_code_sha256': {p:f.new['files'][p] for p in m.EXPLORATION_TEMPERATURE_FILES}},
        project_root=f.root, **options)


def test_quarter_plan_real_git_roundtrip_and_unchanged_task_config(q):
    original = q.checkpoint.read_bytes(), q.sidecar.read_bytes()
    plan = record(q)
    factor = plan['exploration_temperature_factor']
    assert (factor['source_policy_version'], factor['target_policy_version']) == (HISTORY_TEMPERED_POLICY, HISTORY_QUARTER_TEMPERED_POLICY)
    assert (factor['source_exploration_std_temperature'], factor['target_exploration_std_temperature']) == (.5, .25)
    assert factor['deterministic_same_weights_same_observation'] == 'exact_original_conditional_mean_path'
    assert factor['migration_added_updates'] == 0
    assert plan['preserve_actor_critic_optimizer_normalizer_rng_and_budget'] and plan['discard_old_rollout_storage']
    assert all(row['source_sha256'] == row['target_sha256'] for row in factor['configuration_bindings'].values())
    assert all(not factor[k] for k in ('physical_mdp_changed','nominal_control_changed','reward_changed','task_acceptance_changed','action_ranges_changed'))
    path = q.root/'quarter.json'
    write_json(path, plan)
    assert m.validate_migration_plan(q.checkpoint, q.new, path, project_root=q.root)['exploration_temperature_factor'] == factor
    assert original == (q.checkpoint.read_bytes(), q.sidecar.read_bytes())


@pytest.mark.parametrize('field', ['target_exploration_std_temperature','target_policy_contract','target_runner_config','optimizer'])
def test_quarter_tampered_plan_rejected(q, field):
    plan = record(q)
    plan['exploration_temperature_factor'][field] = 'tampered'
    path = q.root/'tampered.json'
    write_json(path, plan)
    with pytest.raises(ValueError, match='exactly bound'):
        m.validate_migration_plan(q.checkpoint, q.new, path, project_root=q.root)


@pytest.mark.parametrize('name', ['reward_config.yaml','observation_schema.json','execution_profile.yaml'])
def test_quarter_cannot_mix_reward_history_or_execution_config(q, name):
    path = q.root/CONFIG/name
    path.write_bytes(path.read_bytes()+b'\n')
    bind(q.root, q.new)
    with pytest.raises(ValueError, match='configuration bindings|config bytes|boundary'):
        record(q)


@pytest.mark.parametrize('module,old,new', [
    ('semantic_history_actor','return 0.5','return 0.6'),
    ('semantic_history_actor','return 9','return 10'),
    ('semantic_training','def load_semantic_checkpoint():\n    return 9','def load_semantic_checkpoint():\n    return 10'),
    ('semantic_cli','return 1','return 2'),
])
def test_old_half_history_loader_and_cli_regions_cannot_change(q, module, old, new):
    path = q.root/f'src/wlr50_clean/ppo/{module}.py'
    path.write_text(path.read_text().replace(old,new),encoding='utf8')
    bind(q.root,q.new)
    with pytest.raises(ValueError,match='outside reviewed|unchanged CLI'):
        record(q)


def test_quarter_wrong_namespace_rejected(q):
    q.new['experiment_id'] = 'fsm_reference_p09_stable_v2'
    with pytest.raises(ValueError,match='same-experiment'):
        record(q)


@pytest.mark.parametrize('factor', ['task_first_reward_review','execution_composition_review','final_stop_handoff_review','prior_evidence'])
def test_quarter_cannot_mix_another_migration(q, factor):
    with pytest.raises(ValueError,match='cannot mix'):
        record(q, **{factor:{}})


@pytest.fixture
def cpu(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('rsl_rl')
    count, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    yield torch
    torch.set_rng_state(rng)
    torch.set_num_threads(count)


def make_runner(version):
    from test_semantic_role_append_training import Core372
    env = t.SemanticRslAdapter(Core372(), seed=1001, device='cpu')
    env.cfg['semantic_version'] = 'v3'
    runner,_ = t.construct_semantic_runner(env, seed=1001, device='cpu', initialize_actor=False,
        policy_version=version, observation_layout=ROLE_OBSERVATION_LAYOUT)
    return runner,env


def real_plan(f, monkeypatch):
    path=f.root/'quarter_real.json'
    write_json(path,record(f))
    original=m.validate_migration_plan
    monkeypatch.setattr(m,'validate_migration_plan',lambda cp,c,p:original(cp,c,p,project_root=f.root))
    return m.validate_migration_plan(f.checkpoint,f.new,path)


def test_unchanged_cli_selects_quarter_only_from_real_verified_plan(q,monkeypatch):
    verified=real_plan(q,monkeypatch)
    args=SimpleNamespace(checkpoint=q.checkpoint,semantic_version='v3',
        experiment_id='task_first_recovery_v1',command='train',num_envs=1,
        seed=1001,device='cpu',new_mdp_warm_start=False,resume_migration=Path(verified['plan_path']),
        policy_distribution_migration=False,target_policy_version=None)
    cli._preflight_checkpoint(args,q.new)
    assert cli._resolved_policy_version(args)==HISTORY_QUARTER_TEMPERED_POLICY
    assert cli._resolved_policy_contract(args)==verified['exploration_temperature_factor']['target_policy_contract']
    assert args._migration_record==verified


@pytest.mark.parametrize('fault',['partial_storage','pending_action','wrong_actor'])
def test_quarter_loader_rejects_invalid_state_before_loading_tensors(q,monkeypatch,cpu,fault):
    verified=real_plan(q,monkeypatch)
    target,env=make_runner(HISTORY_QUARTER_TEMPERED_POLICY)
    if fault=='partial_storage': target.alg.storage.step=1
    elif fault=='pending_action': target.alg.transition.actions=cpu.zeros(1,12)
    else: target._semantic_policy_version=HISTORY_TEMPERED_POLICY
    def forbidden(*args,**kwargs):
        raise AssertionError('invalid quarter request reached tensor loader')
    monkeypatch.setattr(t,'load_checkpoint_round_trip',forbidden)
    with pytest.raises((RuntimeError,ValueError)):
        t.load_semantic_checkpoint(target,q.checkpoint,contract=q.new,seed=1001,migration=verified)
    assert env.core.calls==0


def test_real_cpu_quarter_load_preserves_mean_adam_identity_rng_counts_then_updates_reloads(quarter_prototype,tmp_path,monkeypatch,cpu):
    from tensordict import TensorDict
    root=tmp_path/'actual_cpu'
    shutil.copytree(quarter_prototype.root,root)
    t.seed_training_rngs(1001)
    source,source_env=make_runner(HISTORY_TEMPERED_POLICY)
    trained=t.train_semantic(source,source_env,run_dir=root/'old_run',output_root=root/'old_output',
        stage='full_episode',decisions=128,contract=quarter_prototype.old,seed=1001)
    metadata=json.loads(Path(trained['checkpoints'][-1]['manifest']).read_text())
    for key in ('checkpoint_path','checkpoint_sha256','save_load_round_trip'): metadata.pop(key)
    source.alg.learning_rate=1e-5
    for group in source.alg.optimizer.param_groups: group.update(lr=1e-5,betas=(.87,.996),eps=2e-8)
    checkpoint=root/'learned_half.pt'
    t.save_semantic_checkpoint(source,checkpoint,metadata)
    saved=m.checkpoint_metadata(checkpoint)
    assert source.alg.optimizer.state_dict()['state']
    f=SimpleNamespace(root=root,old=quarter_prototype.old,new=quarter_prototype.new,checkpoint=checkpoint)
    verified=real_plan(f,monkeypatch)
    target,env=make_runner(HISTORY_QUARTER_TEMPERED_POLICY)
    loaded=t.load_semantic_checkpoint(target,checkpoint,contract=f.new,seed=1001,migration=verified)
    for role in ('actor','critic'):
        assert t.parameter_hash(getattr(source.alg,role))==t.parameter_hash(getattr(target.alg,role))
    assert t.state_hash(target.alg.optimizer.state_dict())==saved['optimizer_state_sha256']
    assert t.state_hash(t._normalizers(target))==saved['normalizer_state_sha256']
    assert target.alg.actor.obs_normalizer.state_dict()==target.alg.critic.obs_normalizer.state_dict()=={}
    assert t.capture_training_rng_state(seed=1001)==saved['training_rng_state']
    assert target.alg.learning_rate==t.optimizer_learning_rate(target)==1e-5
    assert target.alg.storage.step==0 and target.alg.transition.actions is None and env.core.calls==0
    for key in ('global_policy_decisions','ppo_updates','optimizer_steps','stage_requested_decisions'):
        assert loaded[key]==saved[key]
    values=cpu.randn(7,372,generator=cpu.Generator().manual_seed(771))
    obs=TensorDict({'policy':values,'critic':values},batch_size=[7])
    rng=cpu.get_rng_state()
    assert cpu.equal(source.alg.actor(obs),target.alg.actor(obs))
    assert cpu.equal(cpu.get_rng_state(),rng)
    continued=t.train_semantic(target,env,run_dir=root/'new_run',output_root=root/'new_output',
        stage='full_episode',decisions=128,contract=f.new,seed=1001,resume_infos=loaded)
    final,_=make_runner(HISTORY_QUARTER_TEMPERED_POLICY)
    info=t.load_semantic_checkpoint(final,Path(continued['checkpoints'][-1]['checkpoint']),contract=f.new,seed=1001)
    assert (info['global_policy_decisions'],info['ppo_updates'],info['optimizer_steps'])==(256,2,40)
    assert info['policy_contract']==policy_contract(HISTORY_QUARTER_TEMPERED_POLICY,observation_layout=ROLE_OBSERVATION_LAYOUT)
    assert info['resume_ancestry']['resume_migration']['exploration_temperature_factor']==verified['exploration_temperature_factor']
    assert t.parameter_hash(final.alg.actor)==t.parameter_hash(target.alg.actor)
    assert t.state_hash(final.alg.optimizer.state_dict())==t.state_hash(target.alg.optimizer.state_dict())

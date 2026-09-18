"""Bounded FL reward boundary and actual CPU PPO/save/load; never Isaac."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace as NS

import pytest
import yaml

from wlr50_clean.ppo import semantic_cli as cli, semantic_migration as m, semantic_training as t
from wlr50_clean.ppo import semantic_video as video, semantic_video_cli as vc
from wlr50_clean.ppo.semantic_policy_distribution import HISTORY_QUARTER_TEMPERED_POLICY as POLICY
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_semantic_role_observation_migration import NAMES, commit, git, source_metadata, write_json
from test_semantic_quarter_temperature_migration import make_runner

SOURCE, TARGET = 'configs/ppo_residual_rr_fix_v1', 'configs/ppo_fl_capture_quality_v1'
FACTOR = 'fl_capture_quality_same372_factor'
PROTECTED = {f'src/wlr50_clean/ppo/{name}.py' for name in (
    'semantic_history_actor', 'semantic_policy_distribution', 'semantic_nominal_geometry', 'semantic_residual_adapter')}


def bind(root, contract):
    contract['files'] = {p: m.file_sha(root/p) for p in contract['files']}
    contract['runtime_content_sha256'] = m.digest(contract['files'])
    namespace = f"configs/ppo_{contract['experiment_id']}"
    contract['selected_configuration'] = {n: {'path': f'{namespace}/{n}',
        'sha256': contract['files'][f'{namespace}/{n}']} for n in NAMES}


@pytest.fixture(scope='module')
def prototype(tmp_path_factory):
    root = tmp_path_factory.mktemp('fl_quality_git')
    paths = m.FL_CAPTURE_QUALITY_FILES | PROTECTED | {f'{SOURCE}/{n}' for n in NAMES}
    for p in paths:
        destination = root/p
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((m.PROJECT_ROOT/p).read_bytes())
    git(root, 'init')
    git(root, 'config', 'user.name', 'FL quality CPU fixture')
    git(root, 'config', 'user.email', 'fl-quality@example.invalid')
    old = {'source_git_commit': commit(root, 'source RR profile'), 'files': dict.fromkeys(sorted(paths)),
        'semantic_version': 'v3', 'experiment_id': 'residual_rr_fix_v1',
        'frozen_A_files': {'frozen': 'a'*64}, 'physics_hz': 120., 'decision_hz': 15.,
        'task_timeout_s': 200., 'timeout_bootstrap': False, 'rsl_rl_version': '5.0.1',
        'local_runtime_versions': {'fixture': 'CPU only; not physical evidence'}}
    bind(root, old)
    for n in NAMES:
        destination = root/TARGET/n
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((m.PROJECT_ROOT/TARGET/n).read_bytes())
    for p in (m.SUPERVISOR, m.BODY_REWARD_CODE):
        with (root/p).open('a', encoding='utf-8') as stream:
            stream.write('\n# Explicit reviewed target fixture; old profile remains opt-in.\n')
    new = copy.deepcopy(old)
    new['experiment_id'] = 'fl_capture_quality_v1'
    new['files'].update(dict.fromkeys(f'{TARGET}/{n}' for n in NAMES))
    new['source_git_commit'] = commit(root, 'FL reward profile')
    bind(root, new)
    return NS(root=root, old=old, new=new)


@pytest.fixture
def f(prototype, tmp_path):
    root = tmp_path/'case'
    shutil.copytree(prototype.root, root)
    checkpoint = root/'outputs/ppo_residual_rr_fix_v1/checkpoints/history/source.pt'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'metadata-only fixture; actual CPU test below')
    metadata = source_metadata(checkpoint, prototype.old, layout=ROLE_OBSERVATION_LAYOUT, policy=POLICY)
    sidecar = checkpoint.with_name('source_manifest.json')
    write_json(sidecar, metadata)
    return NS(root=root, old=copy.deepcopy(prototype.old), new=copy.deepcopy(prototype.new),
        checkpoint=checkpoint, sidecar=sidecar, metadata=metadata)


def record(f, **options):
    delta = sorted(p for p in f.old['files'].keys() | f.new['files'].keys()
        if f.old['files'].get(p) != f.new['files'].get(p))
    return m.build_migration_plan(f.checkpoint, f.new, allowed_changed_files=delta,
        reason='FL fine-gap physical potential and bounded nonzero front body quality',
        fl_capture_quality_review={'reason': 'Same quarter HISTORY and control; new reward and potential feature semantics',
            'reviewed_code_sha256': {p: f.new['files'][p] for p in delta if p in m.FL_CAPTURE_QUALITY_FILES}},
        project_root=f.root, **options)


def verify(f, monkeypatch):
    path = f.root/'fl_plan.json'
    write_json(path, record(f))
    original = m.validate_migration_plan
    monkeypatch.setattr(m, 'validate_migration_plan', lambda cp, c, p:
        original(cp, c, p, project_root=f.root))
    return m.validate_migration_plan(f.checkpoint, f.new, path)


def test_paths_configs_and_roundtrip(f):
    for name in set(NAMES)-{'stage_task_spec.yaml', 'reward_config.yaml'}:
        assert (m.PROJECT_ROOT/SOURCE/name).read_bytes() == (m.PROJECT_ROOT/TARGET/name).read_bytes()
    assert [p.name for p in cli.version_paths('v3', experiment_id='fl_capture_quality_v1')] == ['ppo_fl_capture_quality_v1']*3
    assert video.camera_for_experiment('fl_capture_quality_v1') == video.camera_for_experiment('non_residual_refine_v1')
    assert video.task_interval_receipt(5907, experiment_id='fl_capture_quality_v1')['frame_count'] == 739
    plan = record(f)
    factor = plan[FACTOR]
    assert factor['reward_changed'] and factor['observation_semantics_changed'] and not factor['same_mdp_claimed']
    assert not any(factor[k] for k in ('task_acceptance_changed', 'nominal_control_changed',
        'physical_scene_changed', 'action_execution_changed', 'kernel_changed', 'observation_layout_changed'))
    assert factor['supervisor_scope']['protected_ast_identical']
    assert factor['source_runner_config'] == factor['target_runner_config'] == f.metadata['runner_config']
    assert plan['discard_old_rollout_storage'] and factor['migration_added_updates'] == 0
    path = f.root/'plan.json'
    write_json(path, plan)
    assert m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)[FACTOR] == factor
    plan[FACTOR]['kernel_changed'] = True
    write_json(path, plan)
    with pytest.raises(ValueError, match='exactly bound'):
        m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)


@pytest.mark.parametrize('fault', ['namespace', 'physics', 'temperature', 'runner', 'stage', 'reward', 'caps', 'actor', 'nominal', 'mixed'])
def test_rejects_mixed_or_unreviewed_boundary(f, fault):
    if fault == 'namespace': f.new['experiment_id'] = 'task_first_recovery_v1'
    elif fault == 'physics': f.new['physics_hz'] = 60.
    elif fault in ('temperature', 'runner'):
        if fault == 'temperature': f.metadata['policy_contract']['exploration_std_temperature'] = .5
        else: f.metadata['runner_config']['algorithm']['clip_param'] = .3
        write_json(f.sidecar, f.metadata)
    elif fault == 'mixed':
        with pytest.raises(ValueError, match='mix'):
            record(f, rr_physical_acceptance_review={'reason': 'not allowed'})
        return
    else:
        path = {'stage': f'{TARGET}/stage_task_spec.yaml', 'reward': f'{TARGET}/reward_config.yaml',
            'caps': f'{TARGET}/execution_profile.yaml', 'actor': 'src/wlr50_clean/ppo/semantic_history_actor.py',
            'nominal': m.SUPERVISOR}[fault]
        text = (f.root/path).read_text()
        if fault == 'stage': text = text.replace('functional_free_air_lift_v3', 'functional_lift_edge_v2')
        elif fault == 'reward': text = text.replace('quality_epsilon: 0.03', 'quality_epsilon: 0.0')
        elif fault == 'nominal': text += '\ndef unreviewed_control_change():\n    return 1\n'
        else: text += '\n# unrelated protected byte change\n'
        (f.root/path).write_text(text)
        bind(f.root, f.new)
    with pytest.raises(ValueError):
        record(f)


@pytest.fixture
def cpu(monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('rsl_rl')
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    yield torch
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


@pytest.mark.parametrize('stochastic', [False, True])
def test_actual_head_audit_adds_no_forward_rng_or_action_change(cpu, stochastic):
    runner, env = make_runner(POLICY)
    obs = env.get_observations()
    rng = cpu.get_rng_state()
    with cpu.inference_mode():
        expected = runner.alg.actor(obs, stochastic_output=stochastic)
        expected_rng = cpu.get_rng_state()
        cpu.set_rng_state(rng)
        actual, trace = t.audited_history_policy_request(runner.alg.actor, obs,
            lambda: runner.alg.actor(obs, stochastic_output=stochastic), stochastic=stochastic)
    assert cpu.equal(actual, expected) and cpu.equal(cpu.get_rng_state(), expected_rng)
    assert trace['extra_random_draws'] == trace['extra_model_forwards'] == 0
    assert trace['selected_raw_full12'] == actual[0].tolist()
    assert trace['exploration_std_temperature'] == .25 and trace['rho'] == .9
    assert len(runner.alg.actor.mlp._forward_hooks) == 0


def test_official_cpu_migration_update_save_reload_and_both_video_modes(prototype, tmp_path, monkeypatch, cpu):
    root = tmp_path/'cpu'
    shutil.copytree(prototype.root, root)
    t.seed_training_rngs(1001)
    source, source_env = make_runner(POLICY)
    trained = t.train_semantic(source, source_env, run_dir=root/'source_run', output_root=root/'source_output',
        stage='full_episode', decisions=128, contract=prototype.old, seed=1001)
    metadata = json.loads(Path(trained['checkpoints'][-1]['manifest']).read_text())
    for key in ('checkpoint_path', 'checkpoint_sha256', 'save_load_round_trip'): metadata.pop(key)
    metadata['rr_task_branch'] = {'counter_origin': {'global_policy_decisions': 0, 'ppo_updates': 0, 'optimizer_steps': 0}}
    source.alg.learning_rate = 1e-5
    for group in source.alg.optimizer.param_groups: group.update(lr=1e-5, betas=(.87, .996), eps=2e-8)
    cp = root/'outputs/ppo_residual_rr_fix_v1/checkpoints/history/learned.pt'
    t.save_semantic_checkpoint(source, cp, metadata)
    saved = m.checkpoint_metadata(cp)
    f = NS(root=root, old=prototype.old, new=prototype.new, checkpoint=cp)
    verified = verify(f, monkeypatch)
    target, env = make_runner(POLICY)
    loaded = t.load_semantic_checkpoint(target, cp, contract=f.new, seed=1001, migration=verified)
    for role in ('actor', 'critic'):
        assert t.parameter_hash(getattr(target.alg, role)) == saved[f'{role}_parameter_sha256']
    assert t.state_hash(target.alg.optimizer.state_dict()) == saved['optimizer_state_sha256']
    assert t.state_hash(t._normalizers(target)) == saved['normalizer_state_sha256']
    assert t.capture_training_rng_state(seed=1001) == saved['training_rng_state']
    assert t.optimizer_learning_rate(target) == target.alg.learning_rate == 1e-5
    assert target.alg.storage.step == 0 and target.alg.transition.actions is None and env.core.calls == 0
    assert loaded['fl_capture_quality_branch']['counter_origin'] == {'global_policy_decisions': 128, 'ppo_updates': 1, 'optimizer_steps': 20}
    for fault in ('partial', 'pending'):
        target.alg.storage.step = int(fault == 'partial')
        target.alg.transition.actions = cpu.zeros(1, 12) if fault == 'pending' else None
        with pytest.raises(RuntimeError, match='partial old rollout'):
            t.load_semantic_checkpoint(target, cp, contract=f.new, seed=1001, migration=verified)
    target.alg.storage.step, target.alg.transition.actions = 0, None
    result = t.train_semantic(target, env, run_dir=root/'target_run', output_root=root/'outputs/ppo_fl_capture_quality_v1',
        stage='full_episode', decisions=128, contract=f.new, seed=1001, resume_infos=loaded)
    final_cp = Path(result['checkpoints'][-1]['checkpoint'])
    final, _ = make_runner(POLICY)
    info = t.load_semantic_checkpoint(final, final_cp, contract=f.new, seed=1001)
    assert (info['global_policy_decisions'], info['ppo_updates'], info['optimizer_steps']) == (256, 2, 40)
    assert info['fl_capture_quality_branch_counts'] == {'global_policy_decisions': 128, 'ppo_updates': 1, 'optimizer_steps': 20}
    assert info['rr_task_branch'] == saved['rr_task_branch']
    assert info['resume_ancestry']['resume_migration'][FACTOR] == verified[FACTOR]
    assert t.parameter_hash(final.alg.actor) == t.parameter_hash(target.alg.actor) != saved['actor_parameter_sha256']
    assert t.state_hash(final.alg.optimizer.state_dict()) == t.state_hash(target.alg.optimizer.state_dict())
    assert list((root/'target_run/rollouts').glob('update_000002_likelihood.json'))
    audit = [json.loads(x) for x in (root/'target_run/residual_and_projection_audit.jsonl').read_text().splitlines()]
    assert len(audit) == 128 and all(x['policy_request']['selected_raw_full12'] == x['raw_policy_action_full12'] for x in audit)
    monkeypatch.setattr(cli, 'PROJECT_ROOT', root)
    base = ['eval', '--semantic-version', 'v3', '--experiment-id', 'fl_capture_quality_v1',
        '--run-dir', str(root/'runs/ppo_fl_capture_quality_v1/video_eval/test'), '--expected-head', f.new['source_git_commit'],
        '--checkpoint', str(final_cp), '--seed', '4001', '--device', 'cpu', '--mode', 'semantic_residual_eval', '--no-headless']
    outputs = []
    for stochastic in (False, True, True):
        args = vc.video_parser().parse_args(base + (['--stochastic-policy', '--policy-seed', '4101'] if stochastic else []))
        assert vc.validate_video_args(args) == 'C'
        cli._preflight_checkpoint(args, f.new)
        action, proof, unchanged = vc.checkpoint_loader(args, f.new)((0.,)*372)
        outputs.append(action((0.,)*372, 0))
        unchanged()
        assert proof['stochastic_policy'] == stochastic and proof['optimizer_updates'] == 0
        assert action.last_request['selected_raw_full12'] == list(outputs[-1])
    assert outputs[0] != outputs[1] == outputs[2]
    for extras in (['--stochastic-policy'], ['--policy-seed', '4101']):
        with pytest.raises(RuntimeError, match='stochastic video'):
            vc.validate_video_args(vc.video_parser().parse_args(base+extras))

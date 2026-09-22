"""CPU-only quantity tests. Synthetic samples never carry physical PPO credit."""
from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from budget_bootstrap import HERE, ROOT
from wlr50_clean.ppo import semantic_cli as cli, semantic_training as training

EXPERIMENT = 'task_conditioned_hip_wheel_v1'
OLD = {'smoke':10000, 'phase_suffix':100000, 'full_episode':100000}
NEW = {**OLD, 'full_episode':131072}


class CPUCore:
    """Synthetic ABI, deliberately not evidence of contact or native execution."""
    def __init__(self):
        self.calls = 0
        self.frame = SimpleNamespace(sim_time_s=0.)
    def observation(self, raw=None):
        obs = [0.]*372; obs[0] = 1.; obs[20] = .01+self.tick/3000
        obs[158:163] = [1.]*5
        if raw is not None:obs[195:207] = [max(-19., min(19., v)) for v in raw]
        return tuple(obs)
    def reset(self, *, seed=1001, options=None):
        self.tick = 0; self.frame.sim_time_s = 0.
        return self.observation()
    def step(self, raw):
        self.tick += 1; self.calls += 1; self.frame.sim_time_s = self.tick/15
        done = self.tick == 17
        return SimpleNamespace(observation=self.observation(raw),reward=1+raw[0]-.1*raw[1]**2,
            terminated=done,truncated=False,info={'phase_id':'P01','raw_policy_action_full12':raw,
            'applied_action_full12':tuple(.1*v for v in raw),'task_success':False,
            'termination_reason':'CPU_TEST_DOUBLE_NOT_PHYSICS' if done else None,
            'actuator_target_effect_audit':{'schema':'wlr50_clean.actuator_target_effect_audit.v1',
                'verified':True,'actual_mapping_matches_dispatch':True,'setter_dispatch_targets_equal':True,
                'same_tick_counterfactual':True,'raw_policy_action_full12':raw,'target_dtype':'torch.float32',
                'changed_target_channel_count':12}})
    def telemetry_summary(self):return {'synthetic_CPU_only':True,'calls':self.calls}


def task_runner():
    from wlr50_clean.ppo.semantic_policy_distribution import TASK_CONDITIONED_HIP_WHEEL_POLICY
    env = training.SemanticRslAdapter(CPUCore(),seed=1001,device='cpu')
    env.cfg['semantic_version'] = 'v3'
    runner,_ = training.construct_semantic_runner(env,seed=1001,device='cpu',initialize_actor=False,
        policy_version=TASK_CONDITIONED_HIP_WHEEL_POLICY,observation_layout='diagonal_transfer_state_v1')
    return runner,env


def test_only_explicit_task_ceiling_changes_and_return_is_not_mutable_global():
    assert training.STAGE_BUDGETS == OLD
    for name in (None, 'fl_capture_quality_v1', 'transfer_roles_v1', 'unknown'):
        assert training.training_quantity_budgets(name) == OLD
    actual = training.training_quantity_budgets(EXPERIMENT)
    assert actual == NEW
    actual['full_episode'] = 0
    assert training.training_quantity_budgets(EXPERIMENT) == NEW
    assert training.STAGE_BUDGETS == OLD


def entropy_node(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)
        and n.targets[0].attr == 'entropy_coef')


def test_entropy_schedule_node_and_every_integer_step_to_extended_limit_unchanged():
    old = entropy_node(ROOT/'src/wlr50_clean/ppo/semantic_training.py')
    new = entropy_node(HERE/'src/wlr50_clean/ppo/semantic_training.py')
    assert ast.dump(old, include_attributes=False) == ast.dump(new, include_attributes=False)
    a, b = (compile(ast.Expression(n.value), '<entropy-only>', 'eval') for n in (old,new))
    assert sum(training.STAGE_BUDGETS.values()) == 210000
    for step in range(260001):
        assert eval(a, {'global_step':step,'STAGE_BUDGETS':OLD}) == eval(b, {'global_step':step,'STAGE_BUDGETS':training.STAGE_BUDGETS})


def test_profile_only_explicit_budget_delta():
    import yaml
    path = Path('configs/ppo_task_conditioned_hip_wheel_v1/execution_profile.yaml')
    old = yaml.safe_load((ROOT/path).read_text()); new = yaml.safe_load((HERE/path).read_text())
    expected = copy.deepcopy(old); expected['training_budgets'] = NEW
    assert new == expected and old['training_budgets'] == OLD


def test_unsupported_experiment_remains_rejected():
    with pytest.raises(ValueError):
        cli.runtime_contract(expected_head='0'*40,semantic_version='v3',experiment_id='undeclared_budget_override')


@pytest.mark.parametrize('declared',[
    OLD, {**NEW,'full_episode':131073}, {**NEW,'full_episode':131072.0},
    {**NEW,'phase_suffix':100001}, {**NEW,'unused_stage':1}, {},
])
def test_actual_runtime_profile_guard_rejects_non_declared_budget(tmp_path,declared):
    # Execute the exact isolated guard AST, not a replacement check or git/hash
    # waiver. The full runtime/migration hash binding has independent tests.
    import yaml
    tree = ast.parse((HERE/'src/wlr50_clean/ppo/semantic_cli.py').read_text())
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == 'runtime_contract')
    guard = next(n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
        and any(isinstance(c,ast.Constant) and c.value == EXPERIMENT for c in n.test.comparators))
    code = compile(ast.Module(body=[guard],type_ignores=[]),'<actual runtime profile guard>','exec')
    path = tmp_path/'execution_profile.yaml'
    path.write_text(yaml.safe_dump({'training_budgets':declared}))
    with pytest.raises(ValueError,match='execution profile quantity budget'):
        exec(code,{'experiment_id':EXPERIMENT,'config_root':tmp_path,'contract':{'training_budgets':NEW}})
    path.write_text(yaml.safe_dump({'training_budgets':NEW}))
    exec(code,{'experiment_id':EXPERIMENT,'config_root':tmp_path,'contract':{'training_budgets':NEW}})


def test_real_saved_CP190464_cpu_readonly_state_and_metadata_hashes():
    """Not exact CPU-runner loading: source device remains honestly cuda:0."""
    import torch
    checkpoint = ROOT/'outputs/ppo_task_conditioned_hip_wheel_v1/checkpoints/history/checkpoint_step_000190464.pt'
    metadata = json.loads(checkpoint.with_name(checkpoint.stem+'_manifest.json').read_text())
    before_rng = torch.get_rng_state().clone()
    payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert torch.equal(before_rng,torch.get_rng_state())
    assert training.sha256_file(checkpoint) == metadata['checkpoint_sha256']
    assert metadata['save_load_round_trip'] and metadata['runner_config']['device'] == 'cuda:0'
    for role in ('actor','critic'):
        state = payload[role+'_state_dict']
        assert set(state) == {f'mlp.{layer}.{part}' for layer in (0,2,4) for part in ('weight','bias')}
        assert training.parameter_hash(SimpleNamespace(named_parameters=lambda:state.items())) == metadata[role+'_parameter_sha256']
        assert all(bool(torch.isfinite(value).all()) for value in state.values())
    optimizer = payload['optimizer_state_dict']
    assert training.state_hash(optimizer) == metadata['optimizer_state_sha256']
    assert training.state_hash({'actor':{},'critic':{}}) == metadata['normalizer_state_sha256']
    assert len(optimizer['state']) == 12
    for state in optimizer['state'].values():
        assert set(state) == {'step','exp_avg','exp_avg_sq'}
        assert all(bool(torch.isfinite(value).all()) for value in state.values())
        # Adam's existing per-parameter step is not the lifetime update ledger;
        # earlier recorded architecture boundaries predate this candidate.
        assert 0 < float(state['step']) <= metadata['optimizer_steps']
    assert {float(state['step']) for state in optimizer['state'].values()} == {5980.0}
    assert all(group['lr'] == metadata['optimizer_learning_rate'] for group in optimizer['param_groups'])
    assert payload['iter'] == metadata['ppo_updates']
    assert all(metadata[key] == value for key,value in payload['infos'].items())
    assert metadata['stage_requested_decisions']['full_episode'] == 98432
    assert metadata['runtime_contract']['training_budgets'] == OLD


def fake_cli(tmp_path, monkeypatch, experiment, spent, requested):
    monkeypatch.setattr(cli,'PROJECT_ROOT',tmp_path)
    namespace = 'ppo_'+experiment
    cp = tmp_path/f'outputs/{namespace}/checkpoints/history/source.pt'
    cp.parent.mkdir(parents=True, exist_ok=True); cp.write_bytes(b'path/metadata only; not a tensor checkpoint')
    cp.with_name(cp.stem+'_manifest.json').write_text(json.dumps({'semantic_version':'v3',
        'stage_requested_decisions':{'smoke':0,'phase_suffix':81920,'full_episode':spent}}))
    args = cli.parser().parse_args(['train','--semantic-version','v3','--experiment-id',experiment,
        '--run-dir',str(tmp_path/f'runs/{namespace}/train/cpu'),'--expected-head','a'*40,
        '--checkpoint',str(cp),'--stage','full_episode','--decisions',str(requested)])
    return args


def test_cli_old_cap_does_not_reset_and_new_cap_accepts_lifetime_spent(tmp_path,monkeypatch):
    args = fake_cli(tmp_path,monkeypatch,EXPERIMENT,99968,2048)
    cli.validate_request(args)
    assert args.decisions == 2048 and not args.new_mdp_warm_start and args.from_phase == 'P01'
    args.decisions = 31104; cli.validate_request(args)
    args.decisions = 31105
    with pytest.raises(ValueError,match='remaining'):cli.validate_request(args)
    old = fake_cli(tmp_path,monkeypatch,'fl_capture_quality_v1',99968,128)
    with pytest.raises(ValueError,match='remaining'):cli.validate_request(old)


@pytest.mark.parametrize('contract',[
    {'experiment_id':EXPERIMENT},
    {'experiment_id':EXPERIMENT,'training_budgets':OLD},
    {'experiment_id':EXPERIMENT,'training_budgets':{**NEW,'smoke':10000.0}},
    {'training_budgets':NEW},
])
def test_direct_train_rejects_unbound_or_wrong_caps_before_sample(tmp_path,contract):
    import torch
    from test_semantic_training import make_runner
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        runner, env = make_runner()
        with pytest.raises(ValueError,match='quantity budgets'):
            training.train_semantic(runner,env,run_dir=tmp_path/'r',output_root=tmp_path/'o',
                stage='full_episode',decisions=128,contract=contract,seed=1001)
        assert env.core.calls == 0
    finally:torch.set_num_threads(threads)


def test_direct_train_new_ceiling_does_not_reset_spent_or_branch_origin(tmp_path):
    import torch
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        training.seed_training_rngs(1001)
        runner, env = task_runner()
        origin = {'global_policy_decisions':185856,'ppo_updates':1417,'optimizer_steps':28340}
        previous = {'global_policy_decisions':192000,'ppo_updates':1465,'optimizer_steps':29300,
            'stage_requested_decisions':{'smoke':0,'phase_suffix':81920,'full_episode':99968},
            'actor_parameter_sha256':training.parameter_hash(runner.alg.actor),'runtime_contract':{'cpu_fixture':True},
            'new_mdp_origin_global_policy_decisions':10112,
            'task_conditioned_hip_wheel_branch':{'counter_origin':origin,'branch_id':EXPERIMENT},
            'training_quantity_budget_extension':{'CPU_receipt_fixture_only':True}}
        contract = {'experiment_id':EXPERIMENT,'training_budgets':NEW}
        result = training.train_semantic(runner,env,run_dir=tmp_path/'run',output_root=tmp_path/'output',
            stage='full_episode',decisions=128,contract=contract,seed=1001,resume_infos=previous)
        metadata = json.loads(Path(result['checkpoints'][-1]['manifest']).read_text())
        assert metadata['global_policy_decisions'] == 192128
        assert metadata['stage_requested_decisions'] == {**previous['stage_requested_decisions'],'full_episode':100096}
        assert metadata['task_conditioned_hip_wheel_branch']['counter_origin'] == origin
        assert metadata['task_conditioned_hip_wheel_branch_counts']['global_policy_decisions'] == 6272
        assert metadata['new_mdp_origin_global_policy_decisions'] == 10112
        assert metadata['training_quantity_budget_extension'] == previous['training_quantity_budget_extension']
        assert metadata['runtime_contract']['training_budgets'] == NEW
        assert runner.alg.entropy_coef == .005+(.001-.005)*min(192128/210000,1.)
    finally:torch.set_num_threads(threads)

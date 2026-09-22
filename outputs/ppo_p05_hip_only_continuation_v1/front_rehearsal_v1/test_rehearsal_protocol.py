"""CPU-only source checks and synthetic checkpoint protocol; no real AUX fit."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import front_rehearsal as kernel
import rehearsal_cli as cli
import reviewed_data as data
from test_front_rehearsal import runner, states, budget


@pytest.fixture(autouse=True)
def cpu_only():
    assert not torch.cuda.is_available(), 'test process must have CUDA_VISIBLE_DEVICES=-1'
    old = torch.get_rng_state(); threads = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_rng_state(old); torch.set_num_threads(threads)


def test_fixed_selection_no_overlap_and_no_failure_positive():
    train, validation, hold = data.selection_indices()
    assert train == [0, 1, *range(2, 280, 3)] and validation == list(range(3, 280, 3))
    assert len(train) == 95 and len(validation) == 93 and len(hold) == 13
    assert min(hold) == 280 and max(hold) == 464
    assert not set(train) & set(validation) and max(train + validation) < 280


@pytest.fixture(scope='module')
def real_source():
    cp = cli.OUT / 'checkpoints/history/checkpoint_step_000209920.pt'
    metadata = cli.semantic_migration.checkpoint_metadata(cp)
    return data.load_reviewed_data(metadata, metadata['runtime_contract'])


def test_exact_real_data_only_read_no_optimization(real_source):
    d = real_source; receipt = d['receipt']
    assert d['train_observations'].shape == (95, 389)
    assert d['validation_raw_targets'].shape == (93, 12)
    assert d['invariance_observations'].shape == (13, 389)
    assert receipt['potential_semantics_verification']['full_adjacent_evaluator_recomputed_inputs'] == 464
    assert receipt['potential_semantics_verification']['all_stored_float32_index17_unchanged']
    assert receipt['invariance_holdout_phase_counts'] == {'P03': 4, 'P04': 1, 'P05': 4, 'P06': 4}
    assert receipt['local_qualification']['FR_placed']['physics_tick'] == 2268
    copy = deepcopy(receipt); content_sha = copy.pop('receipt_content_sha256')
    assert content_sha == data.digest(copy)
    assert receipt['front_raw_minus_recorded_mean_max_abs'] > .2


@pytest.mark.parametrize('corruption', ['invalid_evaluator', 'termination', 'mask', 'raw', 'wrong_phase', 'assist'])
def test_source_integrity_negative_cases(corruption):
    with (data.RUN / 'residual_and_projection_audit.jsonl').open(encoding='utf-8') as stream:
        row = json.loads(next(stream))
    sealed = torch.load(data.RUN / 'rollouts/rollout_001558.pt', map_location='cpu', weights_only=False)
    obs = sealed['observations']['policy'][0, 0]
    action, mu, std, lp = sealed['actions'][0, 0], sealed['distribution_params'][0][0, 0], sealed['distribution_params'][1][0, 0], sealed['actions_log_prob'][0, 0]
    a = row['applied_audit']
    if corruption == 'invalid_evaluator': a['semantic_task']['physical_evaluator']['valid'] = False
    elif corruption == 'termination': a['semantic_task']['physical_evaluator']['termination_reason'] = 'BODY_COLLISION'
    elif corruption == 'mask': a['actuator_target_effect_audit']['phase_mask_full12'][0] = 0
    elif corruption == 'raw': row['raw_policy_action_full12'][0] += .01
    elif corruption == 'wrong_phase': a['phase_id'] = 'P05'
    elif corruption == 'assist': a['actuator_target_effect_audit']['capture_assist_evidence']['owner_indices'] = [0, 1]
    with pytest.raises(ValueError): data.validate_tensor_row(obs, action, mu, std, lp, row, index=0)


def synthetic_metadata():
    origin = dict(global_policy_decisions=0, ppo_updates=0, optimizer_steps=0)
    return {'seed': 1001, 'runtime_contract': {'synthetic_CPU_fixture_not_real_physics': True},
        'global_policy_decisions': 128, 'ppo_updates': 1, 'optimizer_steps': 20,
        'stage_requested_decisions': {'smoke': 0, 'full_episode': 128, 'phase_suffix': 0},
        'stage': 'full_episode', 'sampling': 'P01_full_task_only_initial_version',
        'execution_topology': cli.semantic_migration.continuation_topology('P01_full_task_only_initial_version', None,
            observation_layout=cli.P05_CAPTURE_OBSERVATION_LAYOUT),
        'new_mdp_origin_global_policy_decisions': 0,
        'task_conditioned_hip_wheel_branch': {'counter_origin': deepcopy(origin), 'branch_id': 'task_conditioned_hip_wheel_v1',
            'auxiliary_mean_learning': {'schema': 'historical_fixture', 'accepted': 7, 'attempted': 8}},
        'p05_capture_assist_branch': {'counter_origin': deepcopy(origin)},
        'capture_feedback_semantics_branch': {'counter_origin': deepcopy(origin)},
        'capture_feedback_semantics_migration': {'unchanged_historical_fixture': True},
        'p05_capture_assist_migration': {'unchanged_historical_fixture': True},
        'rr_postcross_workspace_migration': {'unchanged_historical_fixture': True},
        'rr_postcross_workspace_branch': {'schema': 'wlr50_clean.rr_postcross_workspace_same389.v1',
            'semantics': 'current_qualified_RR_over_top_receiver_retirement_v1', 'counter_origin': deepcopy(origin),
            'migration_added_updates': 0, 'source_checkpoint_sha256': 'a' * 64}}


def test_ledger_only_current_branch_no_legacy_rewrite():
    infos = synthetic_metadata(); before = deepcopy(infos)
    report = {'accepted_auxiliary_updates': 1, 'attempted_auxiliary_optimizer_steps': 2}
    out = cli.append_ledger(infos, report=report, data_receipt={'synthetic': True},
        source_checkpoint={'path': 'synthetic', 'sha256': 'b' * 64, 'manifest_sha256': 'c' * 64}, helper_sha256={}, budget={})
    assert infos == before
    ledger = out['rr_postcross_workspace_branch'][cli.LEDGER_KEY]
    assert ledger['events'][0]['event_index'] == 1 and ledger['events'][0]['mean_and_log_sigma_may_change']
    assert ledger['events'][0]['fit_report_sha256'] == data.digest(report)
    assert out['task_conditioned_hip_wheel_branch'] == before['task_conditioned_hip_wheel_branch']
    again = cli.append_ledger(out, report=report, data_receipt={}, source_checkpoint={}, helper_sha256={}, budget={})
    assert again['rr_postcross_workspace_branch'][cli.LEDGER_KEY]['accepted_auxiliary_updates_total'] == 2
    assert again['rr_postcross_workspace_branch'][cli.LEDGER_KEY]['attempted_auxiliary_optimizer_steps_total'] == 4
    with pytest.raises(ValueError): cli.append_ledger(infos, report={'accepted_auxiliary_updates': 0},
        data_receipt={}, source_checkpoint={}, helper_sha256={}, budget={})


class CPUCore:
    def __init__(self): self.calls = 0; self.frame = SimpleNamespace(sim_time_s=0.)
    def observation(self, raw=None):
        obs = [0.] * 389; obs[0] = 1.; obs[20] = .01 + self.tick / 3000
        if raw is not None: obs[195:207] = [max(-19., min(19., v)) for v in raw]
        return tuple(obs)
    def reset(self, *, seed=1001, options=None):
        self.tick = 0; self.frame.sim_time_s = 0.; return self.observation()
    def step(self, raw):
        self.tick += 1; self.calls += 1; self.frame.sim_time_s = self.tick / 15
        return SimpleNamespace(observation=self.observation(raw), reward=1 + raw[0] - .1 * raw[1] ** 2,
            terminated=self.tick == 17, truncated=False, info={'phase_id': 'P01', 'raw_policy_action_full12': raw,
                'applied_action_full12': tuple(.1 * v for v in raw), 'task_success': False,
                'termination_reason': 'CPU_FIXTURE_NOT_PHYSICS' if self.tick == 17 else None,
                'actuator_target_effect_audit': {'schema': 'wlr50_clean.actuator_target_effect_audit.v1',
                    'verified': True, 'actual_mapping_matches_dispatch': True, 'setter_dispatch_targets_equal': True,
                    'same_tick_counterfactual': True, 'raw_policy_action_full12': raw, 'target_dtype': 'torch.float32',
                    'changed_target_channel_count': 12}})
    def telemetry_summary(self): return {'synthetic_CPU_only': True, 'calls': self.calls}


def test_synthetic_aux_independent_reload_then_normal_PPO_carry(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, 'OUT', tmp_path)
    run = runner(); values = states(run)
    source, _ = cli.training.save_semantic_checkpoint(run, tmp_path / 'synthetic_source.pt', synthetic_metadata())
    metadata = cli.semantic_migration.checkpoint_metadata(source)
    loaded = cli.training.load_semantic_checkpoint(run, source, contract=metadata['runtime_contract'], seed=1001)
    fit = kernel.fit(run, *values, budget=budget(), authorized=True)
    assert fit['accepted_auxiliary_updates'] > 0
    synthetic_data = {'train_observations': values[0]['policy'], 'receipt': {'synthetic_fixture_only': True}}
    result = cli.save_auxiliary_checkpoint(run, tmp_path / 'checkpoint_aux_frontrehearsal_SYNTHETIC.pt', loaded,
        report=fit, data=synthetic_data, source_checkpoint={'path': str(source), 'sha256': data.sha(source)},
        helper_sha256={'fixture': 'f' * 64}, budget=asdict(budget()))
    assert result['independent_official_reload_verified'] and not result['latest_pointer_published']
    aux_metadata = cli.semantic_migration.checkpoint_metadata(Path(result['path']))
    ledger = deepcopy(aux_metadata['rr_postcross_workspace_branch'][cli.LEDGER_KEY])
    env = cli.training.SemanticRslAdapter(CPUCore(), seed=1001, device='cpu'); env.cfg['semantic_version'] = 'v3'
    fresh, _ = cli.training.construct_semantic_runner(env, seed=1001, device='cpu', initialize_actor=False,
        policy_version=cli.P05_CAPTURE_POLICY, observation_layout=cli.P05_CAPTURE_OBSERVATION_LAYOUT)
    resume = cli.training.load_semantic_checkpoint(fresh, Path(result['path']), contract=metadata['runtime_contract'], seed=1001)
    later = cli.training.train_semantic(fresh, env, run_dir=tmp_path / 'synthetic_ppo', output_root=tmp_path / 'synthetic_out',
        stage='full_episode', decisions=128, contract=metadata['runtime_contract'], seed=1001, resume_infos=resume)
    final = cli.semantic_migration.checkpoint_metadata(Path(later['checkpoints'][-1]['checkpoint']))
    assert final['rr_postcross_workspace_branch'][cli.LEDGER_KEY] == ledger
    assert final['task_conditioned_hip_wheel_branch'] == metadata['task_conditioned_hip_wheel_branch']
    assert final['p05_capture_assist_migration'] == metadata['p05_capture_assist_migration']
    assert final['capture_feedback_semantics_migration'] == metadata['capture_feedback_semantics_migration']
    assert final['rr_postcross_workspace_migration'] == metadata['rr_postcross_workspace_migration']
    assert tuple(final[k] for k in cli.COUNTERS) == (256, 2, 40)

"""Official CPU synthetic state migration; never physical task/PPO evidence."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from budget_bootstrap import ROOT
from test_budget_quantity import CPUCore, task_runner, EXPERIMENT
from test_budget_migration import CP, make_budget_fixture, plan
from wlr50_clean.ppo import semantic_training as t, semantic_migration as m, semantic_cli as cli


def test_official_CPU_full_state_quantity_load_update_save_exact_resume(tmp_path,monkeypatch):
    import torch
    saved_rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    try:
        t.seed_training_rngs(1001)
        source, env = task_runner()
        first = t.train_semantic(source,env,run_dir=tmp_path/'source_cpu_run',output_root=tmp_path/'source_cpu_output',
            stage='full_episode',decisions=128,contract={'CPU_synthetic_no_physical_credit':True},seed=1001)
        infos = json.loads(Path(first['checkpoints'][-1]['manifest']).read_text())
        for key in ('checkpoint_path','checkpoint_sha256','save_load_round_trip'):infos.pop(key)
        real = m.checkpoint_metadata(CP)
        infos['runtime_contract'] = copy.deepcopy(real['runtime_contract'])
        infos['new_mdp_origin_global_policy_decisions'] = 0
        zero_origin = {'global_policy_decisions':0,'ppo_updates':0,'optimizer_steps':0}
        infos['task_conditioned_hip_wheel_branch'] = {
            'schema':'wlr50_clean.task_conditioned_hip_wheel_branch.v1', 'branch_id':EXPERIMENT,
            'counter_origin':zero_origin,'CPU_synthetic_branch_only':True}
        infos['task_conditioned_hip_wheel_branch_counts'] = {k:infos[k] for k in zero_origin}
        infos['rr_task_branch'] = {'counter_origin':zero_origin,'CPU_synthetic_older_branch_only':True}
        infos['rr_task_branch_counts'] = {k:infos[k] for k in zero_origin}
        source.alg.learning_rate = 2.3e-5
        for group in source.alg.optimizer.param_groups:
            group.update(lr=2.3e-5,betas=(.87,.996),eps=2e-8)
        checkpoint = tmp_path/'CPU_synthetic_source.pt'
        t.save_semantic_checkpoint(source,checkpoint,infos)
        original = m.checkpoint_metadata(checkpoint)
        f = make_budget_fixture(tmp_path,monkeypatch,checkpoint=checkpoint)
        supplied = plan(f)
        plan_path = tmp_path/'CPU_non_deployable_budget_plan.json'
        plan_path.write_text(json.dumps(supplied),encoding='utf-8')
        actual_validate = m.validate_migration_plan
        monkeypatch.setattr(m,'validate_migration_plan',lambda cp,c,p:
            actual_validate(cp,c,p,project_root=f.project_root))
        verified = m.validate_migration_plan(checkpoint,f.new,plan_path)
        monkeypatch.setattr(cli,'_request_paths',lambda _:
            (tmp_path,tmp_path,f.project_root/'configs/ppo_task_conditioned_hip_wheel_v1'))
        args = SimpleNamespace(checkpoint=checkpoint,semantic_version='v3',experiment_id=EXPERIMENT,
            command='train',num_envs=1,stage='full_episode',from_phase='P01',teacher_offset_decisions=0,
            prefix_source='frozen_fsm',seed=1001,device='cpu',new_mdp_warm_start=False,
            resume_migration=plan_path,policy_distribution_migration=False,target_policy_version=None)
        cli._preflight_checkpoint(args,f.new)
        for changed in ({'command':'eval'},{'from_phase':'P09','stage':'phase_suffix'},
                        {'teacher_offset_decisions':35},{'num_envs':8},{'prefix_source':'successful_nominal'}):
            bad = copy.copy(args)
            for key,value in changed.items():setattr(bad,key,value)
            with pytest.raises(ValueError,match='quantity-only migration first'):
                cli._preflight_checkpoint(bad,f.new)
        for fault in ('no_plan','partial','pending'):
            invalid,_ = task_runner()
            if fault == 'partial':invalid.alg.storage.step = 1
            if fault == 'pending':invalid.alg.transition.actions = torch.zeros(1,12)
            with pytest.raises((RuntimeError,ValueError)):
                t.load_semantic_checkpoint(invalid,checkpoint,contract=f.new,seed=1001,
                    migration=None if fault=='no_plan' else verified)
        target,newenv = task_runner()
        loaded = t.load_semantic_checkpoint(target,checkpoint,contract=f.new,seed=1001,migration=verified)
        assert newenv.core.calls == 0 and target.alg.storage.step == 0 and target.alg.transition.actions is None
        assert target.alg.learning_rate == t.optimizer_learning_rate(target) == 2.3e-5
        assert all(g['lr']==2.3e-5 and g['betas']==(.87,.996) and g['eps']==2e-8 for g in target.alg.optimizer.param_groups)
        assert t.state_hash(target.alg.optimizer.state_dict()) == original['optimizer_state_sha256']
        assert t.state_hash(t._normalizers(target)) == original['normalizer_state_sha256']
        assert t.parameter_hash(target.alg.actor) == original['actor_parameter_sha256']
        assert t.parameter_hash(target.alg.critic) == original['critic_parameter_sha256']
        assert t.capture_training_rng_state(seed=1001) == original['training_rng_state']
        for key in ('global_policy_decisions','ppo_updates','optimizer_steps','stage_requested_decisions',
            'new_mdp_origin_global_policy_decisions','task_conditioned_hip_wheel_branch',
            'task_conditioned_hip_wheel_branch_counts','rr_task_branch','rr_task_branch_counts'):
            assert loaded[key] == original[key]
        assert loaded['training_quantity_budget_extension']['factor']['kernel_changed'] is False
        assert loaded['training_quantity_budget_extension']['factor']['new_mdp'] is False
        continued = t.train_semantic(target,newenv,run_dir=tmp_path/'target_cpu_run',output_root=tmp_path/'target_cpu_output',
            stage='full_episode',decisions=128,contract=f.new,seed=1001,resume_infos=loaded)
        finalcp = Path(continued['checkpoints'][-1]['checkpoint'])
        final, finalenv = task_runner()
        finalinfo = t.load_semantic_checkpoint(final,finalcp,contract=f.new,seed=1001)
        assert (finalinfo['global_policy_decisions'],finalinfo['ppo_updates'],finalinfo['optimizer_steps']) == (256,2,40)
        assert finalinfo['stage_requested_decisions'] == {'smoke':0,'phase_suffix':0,'full_episode':256}
        assert finalinfo['task_conditioned_hip_wheel_branch'] == original['task_conditioned_hip_wheel_branch']
        assert finalinfo['task_conditioned_hip_wheel_branch_counts'] == {'global_policy_decisions':256,'ppo_updates':2,'optimizer_steps':40}
        assert finalinfo['training_quantity_budget_extension'] == loaded['training_quantity_budget_extension']
        assert t.state_hash(final.alg.optimizer.state_dict()) == t.state_hash(target.alg.optimizer.state_dict())
        assert t.parameter_hash(final.alg.actor) == t.parameter_hash(target.alg.actor)
        assert t.capture_training_rng_state(seed=1001) == finalinfo['training_rng_state']
        args.checkpoint = finalcp; args.resume_migration = None
        cli._preflight_checkpoint(args,f.new)
        assert finalenv.core.calls == 0
    finally:
        torch.set_rng_state(saved_rng);torch.set_num_threads(threads)

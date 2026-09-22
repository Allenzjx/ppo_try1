"""CPU-only candidate boundary tests; temporary contracts are not adoptable plans."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
from budget_bootstrap import HERE, ROOT
from wlr50_clean.ppo import semantic_migration as m

CP = ROOT / 'outputs/ppo_task_conditioned_hip_wheel_v1/checkpoints/history/checkpoint_step_000190976.pt'
REVISION = 'b' * 40
PROFILE = 'configs/ppo_task_conditioned_hip_wheel_v1/execution_profile.yaml'
CODE = {f'src/wlr50_clean/ppo/{n}.py' for n in ('semantic_cli','semantic_training','semantic_migration')}


def make_budget_fixture(tmp_path, monkeypatch, checkpoint=None):
    """Real immutable source; actual candidate bytes in a synthetic future runtime."""
    cp = Path(checkpoint or CP).resolve()
    metadata = m.checkpoint_metadata(cp)
    old, new = copy.deepcopy(metadata['runtime_contract']), copy.deepcopy(metadata['runtime_contract'])
    project_root = tmp_path / 'quantity_runtime'
    original_version_bytes, original_run = m._version_bytes, subprocess.run
    data = {}
    for relative in old['files']:
        source = original_version_bytes(ROOT, old, relative, prefer_worktree=True)
        value = (HERE/relative).read_bytes() if relative in CODE | {PROFILE} else source
        target = project_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
        data[relative] = value
        new['files'][relative] = hashlib.sha256(value).hexdigest()
    new['source_git_commit'] = REVISION
    new['training_budgets'] = {**old['training_budgets'], 'full_episode':131072}
    for row in new['selected_configuration'].values():
        row['sha256'] = new['files'][row['path']]
    new['runtime_content_sha256'] = m.digest(new['files'])
    def version_bytes(root, contract, relative, *, prefer_worktree=False):
        if contract['source_git_commit'] == old['source_git_commit']:
            return original_version_bytes(ROOT, contract, relative, prefer_worktree=True)
        return original_version_bytes(root, contract, relative, prefer_worktree=prefer_worktree)
    def run(command, *a, **kw):
        if command == ['git','-C',str(project_root),'rev-parse','HEAD']:
            return SimpleNamespace(stdout=REVISION)
        return original_run(command,*a,**kw)
    monkeypatch.setattr(m,'_version_bytes',version_bytes)
    monkeypatch.setattr(m.subprocess,'run',run)
    return SimpleNamespace(old=old,new=new,metadata=metadata,checkpoint=cp,project_root=project_root,data=data)


@pytest.fixture
def budget_fixture(tmp_path,monkeypatch):
    return make_budget_fixture(tmp_path,monkeypatch)


def plan(f, **options):
    delta=sorted(p for p in f.old['files'].keys()|f.new['files'].keys() if f.old['files'].get(p)!=f.new['files'].get(p))
    review={'reason':'CPU-only exact quantity extension, not a committed production plan',
            'reviewed_code_sha256':{p:f.new['files'][p] for p in sorted(CODE)}}
    return m.build_migration_plan(f.checkpoint,f.new,allowed_changed_files=delta,
        reason='CPU-only synthetic future HEAD; no physical or optimizer credit',
        training_quantity_budget_review=review,project_root=f.project_root,**options)


def mutate(f,relative,transform):
    value=transform((f.project_root/relative).read_bytes())
    assert value != (f.project_root/relative).read_bytes(), 'counterexample must really mutate bytes'
    (f.project_root/relative).write_bytes(value)
    f.data[relative]=value
    f.new['files'][relative]=hashlib.sha256(value).hexdigest()
    for row in f.new['selected_configuration'].values():
        row['sha256']=f.new['files'][row['path']]
    f.new['runtime_content_sha256']=m.digest(f.new['files'])


def test_actual_source_exact_quantity_boundary(budget_fixture,tmp_path):
    f=budget_fixture; result=plan(f);factor=result['training_quantity_budget_factor']
    assert factor['schema']=='wlr50_clean.training_quantity_budget_same372.v1'
    assert factor['source_policy_contract']==factor['target_policy_contract']==f.metadata['policy_contract']
    assert factor['source_runner_config']==factor['target_runner_config']==f.metadata['runner_config']
    assert factor['source_effective_learning_rate']==factor['target_effective_learning_rate']==f.metadata['optimizer_learning_rate']
    assert factor['source_training_budgets']=={'smoke':10000,'phase_suffix':100000,'full_episode':100000}
    assert factor['target_training_budgets']=={'smoke':10000,'phase_suffix':100000,'full_episode':131072}
    assert factor['source_stage_requested_decisions']==factor['target_stage_requested_decisions']==f.metadata['stage_requested_decisions']
    assert factor['preserved_branch_metadata']['task_conditioned_hip_wheel_branch']==f.metadata['task_conditioned_hip_wheel_branch']
    assert factor['target_remaining_full_episode']-factor['source_remaining_full_episode']==31072
    assert factor['same_mdp_claimed'] and not factor['new_mdp'] and not factor['kernel_changed'] and not factor['reward_changed']
    assert not factor['lifetime_stage_budget_counters_reset'] and factor['entropy_schedule_denominator']==210000
    assert factor['old_rollout_inherited'] is False and factor['migration_added_updates']==factor['migration_added_policy_decisions']==0
    assert sum(x['bytes_identical'] for x in factor['configuration_bindings'].values())==5
    path=tmp_path/'CPU_non_adoptable_plan.json';path.write_text(json.dumps(result),encoding='utf-8')
    verified=m.validate_migration_plan(f.checkpoint,f.new,path,project_root=f.project_root)
    assert verified['training_quantity_budget_factor']==factor


@pytest.mark.parametrize('fault',[
    'namespace','wrong_head','same_head','full_cap','smoke_cap','boolean_cap','metadata_field',
    'missing_config','wrong_policy','zero_lr','bad_origin','bad_count','reset_stage','already_extended','budget_origin','spent_over_global',
    'profile_control','profile_suffix','protected_config','entropy','loss','constant','migration_old_route','mixed',
])
def test_rejects_unrelated_or_reset_changes(budget_fixture,monkeypatch,fault):
    f=budget_fixture
    if fault=='namespace': f.new['experiment_id']='fl_capture_quality_v1'
    elif fault=='wrong_head':f.new['source_git_commit']='c'*40
    elif fault=='same_head':f.new['source_git_commit']=f.old['source_git_commit']
    elif fault=='full_cap':f.new['training_budgets']['full_episode']=131073
    elif fault=='smoke_cap':f.new['training_budgets']['smoke']=10001
    elif fault=='boolean_cap':f.new['training_budgets']['full_episode']=True
    elif fault=='metadata_field':f.new['decision_hz']=30
    elif fault=='missing_config':f.new['selected_configuration'].pop('quality_score.yaml')
    elif fault in ('wrong_policy','zero_lr','bad_origin','bad_count','reset_stage','already_extended','budget_origin','spent_over_global'):
        metadata=copy.deepcopy(f.metadata)
        if fault=='wrong_policy':metadata['policy_contract']['version']='request_history_FR_knee_P06plus_physical_innovation_sigma_v1'
        elif fault=='zero_lr':metadata['optimizer_learning_rate']=0
        elif fault=='bad_origin':metadata['task_conditioned_hip_wheel_branch']['counter_origin']['ppo_updates']=metadata['ppo_updates']+1
        elif fault=='bad_count':metadata['task_conditioned_hip_wheel_branch_counts']['ppo_updates']=0
        elif fault=='reset_stage':metadata['stage_requested_decisions']['full_episode']=-1
        elif fault=='already_extended':metadata['training_quantity_budget_extension']={'prior':True}
        elif fault=='budget_origin':metadata['new_mdp_origin_global_policy_decisions']=True
        elif fault=='spent_over_global':metadata['new_mdp_origin_global_policy_decisions']=metadata['global_policy_decisions']
        monkeypatch.setattr(m,'checkpoint_metadata',lambda _:metadata)
    elif fault=='profile_control':mutate(f,PROFILE,lambda b:b.replace(b'servo_rate_deg_s: 60.0',b'servo_rate_deg_s: 61.0'))
    elif fault=='profile_suffix':mutate(f,PROFILE,lambda b:b.replace(b'  phase_suffix: 100000',b'  phase_suffix: 100001'))
    elif fault=='protected_config':mutate(f,'configs/ppo_task_conditioned_hip_wheel_v1/reward_config.yaml',lambda b:b+b'\n# Even comment-only protected bytes are rejected.\n')
    elif fault=='entropy':mutate(f,'src/wlr50_clean/ppo/semantic_training.py',lambda b:b.replace(b'sum(STAGE_BUDGETS.values())',b'sum(budgets.values())'))
    elif fault=='loss':mutate(f,'src/wlr50_clean/ppo/semantic_training.py',lambda b:b.replace(b'runner.alg.entropy_coef = 0.005',b'runner.alg.entropy_coef = 0.006'))
    elif fault=='constant':mutate(f,'src/wlr50_clean/ppo/semantic_training.py',lambda b:b.replace(b'ROLLOUT_LENGTH = 128',b'ROLLOUT_LENGTH = 256'))
    elif fault=='migration_old_route':mutate(f,'src/wlr50_clean/ppo/semantic_migration.py',lambda b:b.replace(b'requires a versioned exact supported v3 N1 source',b'requires an unreviewed source',1))
    with pytest.raises((ValueError,KeyError)):
        plan(f,**({'archive_only_exact_bytes_review':{'reason':'mixed'}} if fault=='mixed' else {}))


def test_revalidation_rejects_counter_or_mdp_receipt_relabel(budget_fixture,tmp_path):
    f=budget_fixture;result=plan(f)
    for key,value in [('new_mdp',True),('entropy_schedule_denominator',231072),('source_stage_requested_decisions',{'full_episode':0})]:
        altered=copy.deepcopy(result);altered['training_quantity_budget_factor'][key]=value
        path=tmp_path/f'{key}.json';path.write_text(json.dumps(altered),encoding='utf-8')
        with pytest.raises(ValueError,match='exactly bound'):
            m.validate_migration_plan(f.checkpoint,f.new,path,project_root=f.project_root)


def test_all_old_migration_helpers_are_unchanged():
    old=ast.parse((ROOT/'src/wlr50_clean/ppo/semantic_migration.py').read_text(encoding='utf-8'))
    new=ast.parse((HERE/'src/wlr50_clean/ppo/semantic_migration.py').read_text(encoding='utf-8'))
    lookup={n.name:ast.dump(n,include_attributes=False) for n in new.body if isinstance(n,ast.FunctionDef)}
    for node in old.body:
        if isinstance(node,ast.FunctionDef) and node.name not in ('build_migration_plan','validate_migration_plan'):
            assert lookup[node.name]==ast.dump(node,include_attributes=False),node.name

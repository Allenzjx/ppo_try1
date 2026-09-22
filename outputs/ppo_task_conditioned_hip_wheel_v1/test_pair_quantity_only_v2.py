"""Pure metadata wiring tests; official quantity rules stay in production validator."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
import test_pair_quantity_only as old

PATH=Path(__file__).with_name('pair_quantity_only_v2.py')
spec=importlib.util.spec_from_file_location('quantity_v2_test',PATH)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(old,'p',p)
    f=old.fixture.__wrapped__(tmp_path,monkeypatch)
    names={'action_schema.json':'action_schema_path','execution_profile.yaml':'execution_profile',
        'observation_schema.json':'observation_schema_path','quality_score.yaml':'quality_score_path',
        'reward_config.yaml':'reward_config_path','stage_task_spec.yaml':'task_spec_path'}
    data={};bindings={};rows=[]
    for side,contract in (('source',f['old']),('target',f['new'])):
        contract['files']={};view={}
        for name,key in names.items():
            relative='configs/ppo_task_conditioned_hip_wheel_v1/'+name
            raw=(b'full_episode: 100000\n' if side=='source' else b'full_episode: 131072\n') if name=='execution_profile.yaml' else name.encode()
            digest=hashlib.sha256(raw).hexdigest();data[side,relative]=raw;contract['files'][relative]=digest
            bindings.setdefault(name,{'bytes_identical':name!='execution_profile.yaml'})[side]={'path':relative,'sha256':digest}
            view[key]={'path':str((p.ROOT/relative).resolve()),'sha256':digest,'bytes':len(raw)}
        rows.append(view)
    f['b']['manifest']['evaluation_configuration'],f['c']['manifest']['evaluation_configuration']=rows
    f['factor']['configuration_bindings']=bindings
    for key,contract in (('source_contract_sha256',f['old']),('target_contract_sha256',f['new'])):
        f['raw'][key]=f['verified'][key]=f['extension'][key]=old.m.digest(contract)
    f['plan_path'].write_text(json.dumps(f['raw']),encoding='utf-8')
    f['verified']['plan_sha256']=f['extension']['plan_sha256']=p.r.sha256(f['plan_path'])
    def version_bytes(root,contract,relative,*,prefer_worktree=False):
        side='source' if contract==f['old'] else 'target'
        value=data[side,relative]
        assert hashlib.sha256(value).hexdigest()==contract['files'][relative]
        return value
    monkeypatch.setattr(old.m,'_version_bytes',version_bytes)
    f['save']();return f


def test_bound_profile_only_after_actual_validator_call(fixture,monkeypatch):
    f=fixture;original=p.verified_evaluation_configuration
    def after_validation(*args):
        assert len(f['calls'])==1
        return original(*args)
    monkeypatch.setattr(p,'verified_evaluation_configuration',after_validation)
    result=p.strict_quantity_capture(f['b'],f['c'])
    assert result['same_evaluation_configuration'] is False
    assert result['same_control_configuration_via_verified_quantity_factor'] is True
    assert result['same_runtime_contract'] is False
    assert result['source_B_evaluation_configuration']!=result['target_C_evaluation_configuration']


@pytest.mark.parametrize('fault',['A','stochastic','camera','evaluation_configuration','seed','reset','experiment',
    'missing_persisted','manifest_only','tampered_plan','wrong_saved_factor','extra_aux_factor',
    'kernel','reward','task','nominal','execution','new_mdp','same_mdp','quantity','B_runtime','C_runtime','aux_checkpoint_provenance'])
def test_all_previous_nonquantity_refusals_still_apply(fixture,fault):
    old.test_rejects_any_nonquantity_or_unbound_condition(fixture,fault)


@pytest.mark.parametrize('fault',['profile_hash','profile_path','profile_bytes','other_hash','extra_field','missing_config','bindings','nested_aux','empty_aux','ancestor_aux'])
def test_rejects_unbound_config_delta_and_nested_aux(fixture,fault):
    f=fixture;e=f['c']['manifest']['evaluation_configuration']
    if fault=='profile_hash':e['execution_profile']['sha256']='x'*64
    elif fault=='profile_path':e['execution_profile']['path']=str(p.ROOT/'different.yaml')
    elif fault=='profile_bytes':e['execution_profile']['bytes']+=1
    elif fault=='other_hash':e['reward_config_path']['sha256']='x'*64
    elif fault=='extra_field':e['not_reviewed']=True
    elif fault=='missing_config':e.pop('action_schema_path')
    elif fault=='bindings':f['factor']['configuration_bindings'].pop('reward_config.yaml');f['save']()
    elif fault in ('nested_aux','empty_aux'):
        f['infos']['branch']={'auxiliary_mean_learning':{'steps':1} if fault=='nested_aux' else None};f['save']()
    elif fault=='ancestor_aux':f['infos']['resume_ancestry']=[{'branch':{'aux_unreviewed':{}}}];f['save']()
    with pytest.raises((RuntimeError,ValueError,KeyError,AssertionError)):
        p.strict_quantity_capture(f['b'],f['c'])


def test_old_pair_and_review_unchanged():
    assert p.r.sha256(p.OUT/'pair_quantity_only.py')=='2b9fb2311f96f4cf2c206b29f6a2e6b92d13a8de219116f96e2ad02934feee5a'
    assert p.r.sha256(p.OUT/'review_video.py')=='1a7e718de692f73c5aa683037474126e2bb7efc39683c864bbe12dab90d5f20b'

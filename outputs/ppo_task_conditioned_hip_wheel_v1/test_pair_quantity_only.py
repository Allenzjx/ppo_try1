"""Pure metadata/persistence gate tests; no real review, encoder, model or Isaac.

The official quantity validator itself is covered by budget_extension's 27 tests.
Here a call-recording test double isolates the small pair gate's obligations.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest

PATH=Path(__file__).with_name('pair_quantity_only.py')
spec=importlib.util.spec_from_file_location('quantity_pair_test',PATH)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
sys.path.insert(0,str(p.ROOT/'src'))
from wlr50_clean.ppo import semantic_migration as m


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    import torch
    source=tmp_path/'SYNTHETIC_source.pt';source.write_bytes(b'NOT_A_MODEL')
    checkpoint=tmp_path/'SYNTHETIC_C.pt'
    plan_path=tmp_path/'SYNTHETIC_plan.json'
    manifest=tmp_path/'SYNTHETIC_C_manifest.json'
    old={'experiment_id':p.r.EXPERIMENT,'source_git_commit':'a'*40,'training_budgets':{'full_episode':100000}}
    new={**old,'source_git_commit':'b'*40,'training_budgets':{'full_episode':131072}}
    factor={'schema':'wlr50_clean.training_quantity_budget_same372.v1','training_quantity_only':True,
        'same_mdp_claimed':True,**{k:False for k in ('kernel_changed','reward_changed','task_acceptance_changed',
            'nominal_control_changed','action_execution_changed','new_mdp')}}
    raw={'source_checkpoint':str(source),'training_quantity_budget_factor':factor,'geometric_factor':None,
        'source_checkpoint_sha256':p.r.sha256(source),'source_contract_sha256':m.digest(old),
        'target_contract_sha256':m.digest(new)}
    plan_path.write_text(json.dumps(raw),encoding='utf-8')
    verified={**raw,'plan_path':str(plan_path),'plan_sha256':p.r.sha256(plan_path)}
    extension={'factor':factor,**{k:verified[k] for k in ('plan_path','plan_sha256','source_checkpoint_sha256',
        'source_contract_sha256','target_contract_sha256')}}
    infos={'runtime_contract':new,'training_quantity_budget_extension':extension,'global_policy_decisions':256}
    source_metadata={'runtime_contract':old,'global_policy_decisions':128,'checkpoint_path':str(source),
        'checkpoint_sha256':p.r.sha256(source),'save_load_round_trip':True}
    identity={'checkpoint':str(checkpoint),'manifest':str(manifest),'runtime_contract':new,
        'checkpoint_sha256':None,'saved_global_policy_decisions':256}
    common={'experiment_id':p.r.EXPERIMENT,'camera':{'name':'TEST_CAMERA'},'evaluation_configuration':{'test':True},
        'seed':4001,'natural_reset_proof':{'entry':'TEST_P01'}}
    b={'manifest':{**copy.deepcopy(common),'runtime_contract':old},
        'receipt':{'role':'B','mode':'N_plus_zero'}}
    c={'manifest':{**copy.deepcopy(common),'runtime_contract':new},
        'receipt':{'role':'C','mode':'deterministic_conditional_mean','checkpoint_identity':identity}}
    def save():
        torch.save({'infos':infos},checkpoint)
        identity['checkpoint_sha256']=p.r.sha256(checkpoint)
        manifest.write_text(json.dumps({**infos,'checkpoint_path':str(checkpoint),
            'checkpoint_sha256':identity['checkpoint_sha256'],'save_load_round_trip':True}),encoding='utf-8')
    calls=[]
    def validate(cp,target,path,*,project_root):
        calls.append((cp,target,path,project_root))
        assert cp==source and target==new and path==plan_path and project_root==p.ROOT
        return copy.deepcopy(verified)
    monkeypatch.setattr(m,'validate_migration_plan',validate)
    monkeypatch.setattr(m,'checkpoint_metadata',lambda cp:copy.deepcopy(source_metadata))
    save()
    return locals()


def test_exact_persisted_quantity_only_gate_calls_official_validator(fixture):
    f=fixture;result=p.strict_quantity_capture(f['b'],f['c'])
    assert len(f['calls'])==1
    assert result['same_control_via_verified_quantity_only_boundary'] is True
    assert result['same_runtime_contract'] is False
    assert result['old_B_not_relabelled_as_new_runtime_B'] is True
    assert result['same_measured_initial_state_claimed'] is False


@pytest.mark.parametrize('fault',['A','stochastic','camera','evaluation_configuration','seed','reset','experiment',
    'missing_persisted','manifest_only','tampered_plan','wrong_saved_factor','extra_aux_factor',
    'kernel','reward','task','nominal','execution','new_mdp','same_mdp','quantity',
    'B_runtime','C_runtime','aux_checkpoint_provenance'])
def test_rejects_any_nonquantity_or_unbound_condition(fixture,fault):
    f=fixture;b,c=f['b'],f['c'];factor=f['factor']
    save=True
    if fault=='A':b['receipt']['role']='A'
    elif fault=='stochastic':c['receipt']['mode']='training_style_conditional_gaussian'
    elif fault in ('camera','evaluation_configuration','seed'):c['manifest'][fault]='different'
    elif fault=='reset':c['manifest']['natural_reset_proof']['entry']='not_P01'
    elif fault=='experiment':c['manifest']['experiment_id']='fl_capture_quality_v1'
    elif fault=='missing_persisted':f['infos'].pop('training_quantity_budget_extension')
    elif fault=='manifest_only':
        import torch
        torch.save({'infos':{'runtime_contract':f['new']}},f['checkpoint']);save=False
    elif fault=='tampered_plan':f['plan_path'].write_text('{}',encoding='utf-8')
    elif fault=='wrong_saved_factor':f['infos']['training_quantity_budget_extension']=copy.deepcopy(f['extension']);f['infos']['training_quantity_budget_extension']['factor']['new_mdp']=True
    elif fault=='extra_aux_factor':f['verified']['auxiliary_factor']={'updates':1}
    elif fault in ('kernel','reward','task','nominal','execution','new_mdp'):
        key={'kernel':'kernel_changed','reward':'reward_changed','task':'task_acceptance_changed',
            'nominal':'nominal_control_changed','execution':'action_execution_changed','new_mdp':'new_mdp'}[fault]
        factor[key]=True
    elif fault=='same_mdp':factor['same_mdp_claimed']=False
    elif fault=='quantity':factor['training_quantity_only']=False
    elif fault=='B_runtime':b['manifest']['runtime_contract']={**f['old'],'source_git_commit':'x'*40}
    elif fault=='C_runtime':c['receipt']['checkpoint_identity']['runtime_contract']={'different':True}
    elif fault=='aux_checkpoint_provenance':f['infos']['auxiliary_training_steps']=1
    if save:f['save']()
    with pytest.raises((RuntimeError,ValueError,KeyError)):
        p.strict_quantity_capture(b,c)


def test_old_review_and_encoder_remain_unchanged():
    assert p.r.sha256(p.OUT/'review_video.py')=='1a7e718de692f73c5aa683037474126e2bb7efc39683c864bbe12dab90d5f20b'
    encoder=(p.ROOT/'outputs/ppo_fl_capture_quality_v1/paired_event_media.py').read_text(encoding='utf-8')
    assert "RUN WINDOW ENDED - FROZEN FRAME" in encoder
    assert "'freeze_is_physical_evidence': False" in encoder
    assert len(PATH.read_text(encoding='utf-8').splitlines())<=150

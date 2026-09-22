"""Pure gate tests; no checkpoint forward, simulator or video encoder."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

PATH=Path(__file__).with_name('pure_ppo_media_guard.py')
spec=importlib.util.spec_from_file_location('candidate_aux_method_guard',PATH)
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def test_no_auxiliary_provenance_is_pure():
    infos={'task_conditioned_hip_wheel_branch':{'counter_origin':{'ppo_updates':1417}},
           'task_conditioned_hip_wheel_branch_counts':{'ppo_updates':52}}
    assert g.require_pure_infos(infos)['method']=='pure_PPO'
    assert g.require_pure_infos(infos)['auxiliary_learning_claimed'] is False


@pytest.mark.parametrize('ledger',[None,{},[],0,False,{'steps':0},{'steps':3,'claimed_PPO_updates':0}])
def test_nested_ledger_never_silently_becomes_pure(ledger):
    infos={'task_conditioned_hip_wheel_branch':{'auxiliary_mean_learning':ledger}}
    assert g.auxiliary_paths(infos)==['$.task_conditioned_hip_wheel_branch.auxiliary_mean_learning']
    with pytest.raises((RuntimeError,ValueError),match='dedicated validated PPO'):
        g.require_pure_infos(infos)


@pytest.mark.parametrize('infos',[
    {'aux_training_steps':1},
    {'resume_ancestry':[{'branch':{'auxiliary_mean_learning':{'steps':1}}}]},
    {'training_quantity_budget_extension':{'factor':{'preserved_branch_metadata':{'branch':{'auxiliary_mean_learning':{}}}}}},
    {'branch':{'unknown_auxiliary_method':'not_reviewed'}},
    {'branch':{'AUX':{'steps':1}}},
])
def test_unknown_or_inherited_auxiliary_provenance_rejected(infos):
    with pytest.raises((RuntimeError,ValueError),match='auxiliary provenance'):
        g.require_pure_infos(infos)


def persisted(tmp_path, infos):
    import torch
    cp=tmp_path/'SYNTHETIC_METADATA_ONLY.pt';torch.save({'infos':infos},cp)
    manifest=tmp_path/'SYNTHETIC_METADATA_ONLY_manifest.json'
    manifest.write_text(json.dumps({**infos,'checkpoint_path':str(cp),
        'checkpoint_sha256':g.r.sha256(cp),'save_load_round_trip':True}),encoding='utf-8')
    return {'role':'C','checkpoint':{'checkpoint':str(cp),'manifest':str(manifest)}}


def test_real_embedded_infos_checked_not_only_visible_branch_counts(tmp_path):
    infos={'task_conditioned_hip_wheel_branch':{'counter_origin':{},'auxiliary_mean_learning':{'steps':2}},
           'task_conditioned_hip_wheel_branch_counts':{'ppo_updates':52}}
    context=persisted(tmp_path,infos)
    with pytest.raises((RuntimeError,ValueError),match='dedicated validated PPO'):
        g.checkpoint_method(context)
    side=Path(context['checkpoint']['manifest']);meta=json.loads(side.read_text())
    meta['task_conditioned_hip_wheel_branch'].pop('auxiliary_mean_learning')
    side.write_text(json.dumps(meta),encoding='utf-8')
    with pytest.raises((RuntimeError,ValueError),match='sidecar disagree'):
        g.checkpoint_method(context)


def test_pure_export_reuses_existing_export_only_after_gate(tmp_path,monkeypatch):
    context=persisted(tmp_path,{'task_conditioned_hip_wheel_branch':{}})
    calls=[]
    monkeypatch.setattr(g.r,'sealed_source',lambda source:context)
    monkeypatch.setattr(g.r,'export',lambda source,destination:calls.append((source,destination)) or {'pure':True})
    assert g.export_pure('source','destination')=={'pure':True}
    assert calls==[('source','destination')]


def test_aux_export_rejected_before_any_export_or_directory(tmp_path,monkeypatch):
    context=persisted(tmp_path,{'branch':{'auxiliary_mean_learning':{}}})
    monkeypatch.setattr(g.r,'sealed_source',lambda source:context)
    monkeypatch.setattr(g.r,'export',lambda *a:pytest.fail('aux must never reach pure encoder'))
    destination=tmp_path/'should_not_exist'
    with pytest.raises((RuntimeError,ValueError),match='Pure PPO media refused'):
        g.export_pure('source',destination)
    assert not destination.exists()


def test_aux_quantity_pair_rejected_before_existing_top_level_only_gate(tmp_path,monkeypatch):
    context=persisted(tmp_path,{'task_conditioned_hip_wheel_branch':{'auxiliary_mean_learning':{'steps':1}}})
    monkeypatch.setattr(g.r,'checked_review',lambda path:{'context':{'role':'B'} if path=='b' else context})
    monkeypatch.setattr(g,'module',lambda *a:pytest.fail('aux must not reach old quantity pair'))
    with pytest.raises((RuntimeError,ValueError),match='Pure PPO media refused'):
        g.quantity_pair_pure('b','c',tmp_path/'not_created')


def test_pure_pair_still_requires_original_official_quantity_validation(tmp_path,monkeypatch):
    context=persisted(tmp_path,{'branch':{}});calls=[]
    monkeypatch.setattr(g.r,'checked_review',lambda path:{'context':{'role':'B'} if path=='b' else context})
    def fail_after_call(*args):
        calls.append(args);raise ValueError('original official quantity validation still rejects')
    monkeypatch.setattr(g,'module',lambda *a:SimpleNamespace(build_pair=fail_after_call))
    with pytest.raises(ValueError,match='official quantity'):
        g.quantity_pair_pure('b','c','destination')
    assert calls==[('b','c','destination')]


def test_existing_media_modules_untouched_and_wrapper_small():
    assert g.r.sha256(g.OUT/'review_video.py')=='1a7e718de692f73c5aa683037474126e2bb7efc39683c864bbe12dab90d5f20b'
    assert g.r.sha256(g.OUT/'pair_quantity_only.py')=='2b9fb2311f96f4cf2c206b29f6a2e6b92d13a8de219116f96e2ad02934feee5a'
    assert len(PATH.read_text(encoding='utf-8').splitlines())<150

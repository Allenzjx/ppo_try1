"""CPU synthetic source/data/ledger tests; never run against a real optimizer."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

import front_retention as kernel
import reviewed_data_410 as data
import retention_cli as cli
from test_front_retention import runner,states,budget


def observation(phase=1,tick=0):
    x=torch.zeros(389);x[phase-1]=1.;x[18]=tick/24000.
    return x


def endpoint():
    rr={'current_lift_valid':False,'obstacle_pair_active':False,'top_surface_contact':False,
        'contact_surface':'GROUND','ground_contact':True,'within_top_xy':False,
        'clearance_m':-.11,'front_distance_m':-.4}
    ev={'valid':True,'termination_reason':None,'physics_tick':8,'current_legs':{'RR':rr},
        'history':{key:{'RR':False} for key in ('active_lift','front_edge_crossed','placed')}}
    return {'applied_audit':{'decision_count':1,'semantic_task':{'physical_evaluator':ev,'termination_reason':None}}}


def test_reset_zero_tail_proof_does_not_invent_reset_frame():
    result=data.prove_zero_tail_input(observation(),{},None,index=0)
    assert result['tail21_zero'] and not result['full_reset_frame_reconstructed']


def test_unobserved_reset_excluded_but_measured_P01_row_retained():
    from types import SimpleNamespace
    original=([0,1,*range(2,280,3)],list(range(3,280,3)),[280,281,282,283,284,285,286,287,288,461,462,463,464])
    legacy=SimpleNamespace(selection_indices=lambda:deepcopy(original))
    train,validation,protection=data.selection_indices(legacy)
    assert (len(train),len(validation),len(protection))==(94,93,13)
    assert 0 not in train and train[0]==1 and original[0][0]==0
    assert validation==original[1] and protection==original[2]
    with pytest.raises(ValueError):
        data.selection_indices(SimpleNamespace(selection_indices=lambda:([1],[],[])))


@pytest.mark.parametrize('bad',['active_lift','crossed','placed','obstacle_pair','later_phase','reset_time','RR_TOP'])
def test_unsafe_zero_padding_rejected(bad):
    x=observation();row={};previous=None;index=0
    if bad in ('active_lift','crossed','placed','obstacle_pair'):
        x[{'active_lift':149,'crossed':153,'placed':157,'obstacle_pair':138}[bad]]=1.
    elif bad=='later_phase': x=observation(9)
    elif bad=='reset_time': x[18]=.01
    else:
        previous=endpoint();previous['applied_audit']['semantic_task']['physical_evaluator']['current_legs']['RR']['contact_surface']='TOP'
        row={'applied_audit':{'decision_count':2}};x=observation(1,8);index=1
    with pytest.raises(ValueError): data.prove_zero_tail_input(x,row,previous,index=index)


def test_adjacent_true_state_needed_even_though_phase_is_front():
    previous=endpoint();row={'applied_audit':{'decision_count':2}}
    proof=data.prove_zero_tail_input(observation(2,8),row,previous,index=1)
    assert proof['tail21_zero'] and proof['RR_contact_surface']=='GROUND'
    with pytest.raises(ValueError): data.prove_zero_tail_input(observation(2,16),row,previous,index=1)


def synthetic_metadata(tmp_path):
    # Reuse the narrowly reviewed same410 official migration fixture, not a
    # fabricated runtime waiver or a real checkpoint's confidential payload.
    tests=data.ROOT/'tests/unit'
    if str(tests) not in sys.path: sys.path.insert(0,str(tests))
    from test_semantic_rr_postcapture_wheel_migration import boundary,bind_current
    meta,old,new,args=boundary(tmp_path)
    new['source_git_commit']=cli.FROZEN_HEAD;args['expected_target_head']=cli.FROZEN_HEAD
    return bind_current(meta,old,new,args)


def test_ledger_appends_only_current_RR_branch(tmp_path):
    infos=synthetic_metadata(tmp_path);before=deepcopy(infos)
    report={'accepted_auxiliary_updates':2,'attempted_auxiliary_optimizer_steps':3}
    result=cli.append_ledger(infos,report=report,report_binding={'path':'synthetic_fit.json','sha256':'a'*64},
        data_receipt={'receipt_content_sha256':'b'*64},source_binding={'sha256':'c'*64},helper_sha256={},budget={})
    ledger=result['rr_capture_transfer_branch'][cli.LEDGER_KEY]
    assert ledger['accepted_auxiliary_updates_total']==2 and ledger['attempted_auxiliary_optimizer_steps_total']==3
    assert ledger['events'][0]['event_index']==1 and ledger['events'][0]['PPO_updates_added']==0
    assert infos==before and result['rr_postcross_workspace_branch']==before['rr_postcross_workspace_branch']
    for key in before:
        if key!='rr_capture_transfer_branch': assert result[key]==before[key]
    with pytest.raises(ValueError): cli.append_ledger(infos,report={**report,'accepted_auxiliary_updates':0},
        report_binding={},data_receipt={},source_binding={},helper_sha256={},budget={})


def test_actual_official_CPU_save_fresh_reload_and_branch_identity(tmp_path,monkeypatch):
    assert not torch.cuda.is_available()
    infos=synthetic_metadata(tmp_path);run=runner();tx,raw,vx,vraw,ix=states(run)
    branch=Path(infos['checkpoint_output_routing']['output_root'])
    monkeypatch.setattr(cli,'BRANCH',branch);monkeypatch.setattr(cli,'OUT',tmp_path)
    infos.pop('checkpoint_sha256',None)
    source,sidecar=kernel.training.save_semantic_checkpoint(run,branch/'checkpoints/history/synthetic_source.pt',infos)
    source_meta=cli.semantic_migration.checkpoint_metadata(source)
    # Source lookup uses explicit hashes even when the runtime/branch match.
    binding=cli.validate_source(source,source_meta,source_meta['runtime_contract'],expected_sha=data.sha(source),
        expected_manifest_sha=data.sha(sidecar))
    with pytest.raises(ValueError): cli.validate_source(source,source_meta,source_meta['runtime_contract'],expected_sha='0'*64,
        expected_manifest_sha=data.sha(sidecar))
    fresh=cli.observation_runner(tx['policy'][0],source_meta,device='cpu')
    loaded=kernel.training.load_semantic_checkpoint(fresh,source,contract=source_meta['runtime_contract'],seed=source_meta['seed'])
    # The test alone uses synthetic bounded SGD, never a production checkpoint.
    report=kernel.fit(fresh,tx,raw,vx,vraw,ix,budget=budget(max_attempts=1),authorized=True)
    assert report['accepted_auxiliary_updates']==1
    report_path=tmp_path/'synthetic_fit.json';kernel.training.write_json(report_path,report)
    candidate=branch/'checkpoints/history/checkpoint_aux_frontretention410_synthetic.pt'
    result=cli.save_auxiliary_checkpoint(fresh,candidate,loaded,report=report,
        report_binding={'path':str(report_path),'sha256':data.sha(report_path),'content_sha256':data.digest(report)},
        data={'train_observations':tx['policy'],'receipt':{'receipt_content_sha256':'d'*64}},
        source_binding=binding,budget={})
    assert result['independent_official_reload_verified'] and not result['latest_pointer_published']
    target=cli.semantic_migration.checkpoint_metadata(candidate)
    assert all(target[k]==source_meta[k] for k in cli.COUNTERS)
    assert target['rr_postcross_workspace_branch']==source_meta['rr_postcross_workspace_branch']
    assert target['rr_postcapture_wheel_v9_migration']==source_meta['rr_postcapture_wheel_v9_migration']
    assert target['checkpoint_output_routing']==source_meta['checkpoint_output_routing']

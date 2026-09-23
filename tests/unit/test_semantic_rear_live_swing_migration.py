"""Bounded synthetic same419 state/lineage tests; zero robot-training credit."""
import copy
import json
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from wlr50_clean.ppo import semantic_training as t
from wlr50_clean.ppo import semantic_rear_live_swing_migration as m
from wlr50_clean.ppo import semantic_rear_recapture_migration as parent
from wlr50_clean.ppo import semantic_rear_policy_timing_migration as initial
from wlr50_clean.ppo.semantic_migration import digest,continuation_topology
from test_semantic_rear_recapture_migration import make,Synthetic419Core,LAYOUT
from test_semantic_rear_checkpoint_output_branch import metadata as ancestor_metadata


def lineage(tmp_path):
    meta=ancestor_metadata()
    old=meta['runtime_contract']
    old.update(source_git_commit=m.SOURCE_HEAD,semantic_version='v3',
        training_budgets=t.training_quantity_budgets(m.EXPERIMENT),
        files={'unchanged.py':'a'*64,'timing.py':'b'*64},
        selected_configuration={'execution_profile.yaml':{'path':'profile','sha256':'c'*64}})
    meta[parent.MIGRATION].update(target_git_commit=m.SOURCE_HEAD,
        target_contract_sha256=digest(old),target_runtime_content_sha256=old['runtime_content_sha256'])
    meta.update(m.REVISION_ORIGIN)
    meta['rear_policy_timing_branch_counts']={k:meta[k]-m.SOURCE_COUNTS[k] for k in m.COUNTERS}
    branch=tmp_path/('ppo_'+m.EXPERIMENT)/'branches'/m.BRANCH_NAME
    meta['checkpoint_output_routing']=dict(schema='wlr50_clean.checkpoint_output_routing.v1',
        branch=m.BRANCH_NAME,output_root=str(branch.resolve()),main_latest_pointer_promotion=False,
        source_selection=copy.deepcopy(parent.SOURCE_SELECTION))
    new=copy.deepcopy(old)
    new.update(source_git_commit='d'*40,runtime_content_sha256='e'*64)
    new['files'].update({'timing.py':'f'*64,'new.py':'1'*64})
    new['selected_configuration']['execution_profile.yaml']['sha256']='2'*64
    record=dict(schema=m.SCHEMA,source_checkpoint_sha256=m.SOURCE_SHA,
        source_manifest_sha256=m.SOURCE_MANIFEST_SHA,source_git_commit=m.SOURCE_HEAD,
        source_contract_sha256=digest(old),target_contract_sha256=digest(new),
        target_git_commit=new['source_git_commit'],source_runtime_content_sha256=old['runtime_content_sha256'],
        target_runtime_content_sha256=new['runtime_content_sha256'],
        source_changed_configuration=copy.deepcopy(old['selected_configuration']),
        changed_file_hashes={'timing.py':dict(before='b'*64,after='f'*64),'new.py':dict(before=None,after='1'*64)},
        plan_path=str(tmp_path/'synthetic_plan.json'),
        **{m.FACTOR_KEY:dict(schema=m.SCHEMA,source_mode=m.SOURCE_MODE,target_mode=m.TARGET_MODE,
            source_policy_contract=meta['policy_contract'],target_policy_contract=meta['policy_contract'],
            revision_counter_origin=m.REVISION_ORIGIN,original_branch_origin=m.SOURCE_COUNTS,
            preserved_metadata_sha256={k:digest(meta[k]) for k in (parent.MIGRATION,'checkpoint_output_routing')},
            added_policy_decisions=0,added_ppo_updates=0,added_optimizer_steps=0,added_auxiliary_updates=0)})
    return meta,new,record,branch


def test_exact_predecessor_reconstruction_and_branch_lineage(tmp_path):
    meta,new,record,branch=lineage(tmp_path)
    assert m._previous_contract(new,record)==meta['runtime_contract']
    initial.validate_rear_policy_namespace(meta,meta['runtime_contract'],branch,
        checkpoint_output_routing=meta['checkpoint_output_routing'])
    current={**meta,'runtime_contract':new,m.MIGRATION:record}
    initial.validate_rear_policy_namespace(current,new,branch,checkpoint_output_routing=meta['checkpoint_output_routing'])
    assert current[parent.MIGRATION]==meta[parent.MIGRATION]
    with pytest.raises(ValueError):
        initial.validate_rear_policy_namespace(current,new,branch.parent.parent)


@pytest.mark.parametrize('bad',('source_sha','sidecar_sha','source_head','target_head','contract',
    'parent','borrowed_route','reset_to_initial','origin','added_credit','policy','config_before'))
def test_strict_successor_rejects_lineage_drift(tmp_path,bad):
    meta,new,record,branch=lineage(tmp_path)
    current={**meta,'runtime_contract':new,m.MIGRATION:record}
    factor=record[m.FACTOR_KEY]
    if bad=='source_sha':record['source_checkpoint_sha256']='0'*64
    elif bad=='sidecar_sha':record['source_manifest_sha256']='0'*64
    elif bad=='source_head':record['source_git_commit']='0'*40
    elif bad=='target_head':record['target_git_commit']='0'*40
    elif bad=='contract':new['files']['unchanged.py']='0'*64
    elif bad=='parent':current[parent.MIGRATION]['target_contract_sha256']='0'*64
    elif bad=='borrowed_route':current['checkpoint_output_routing']['branch']='foreign'
    elif bad=='reset_to_initial':current.update(m.SOURCE_COUNTS)
    elif bad=='origin':factor['revision_counter_origin']={**m.REVISION_ORIGIN,'ppo_updates':1688}
    elif bad=='added_credit':factor['added_ppo_updates']=1
    elif bad=='policy':factor['target_policy_contract']={'observation_dimension':419}
    elif bad=='config_before':record['source_changed_configuration']['execution_profile.yaml']['sha256']='0'*64
    with pytest.raises(ValueError):
        initial.validate_rear_policy_namespace(current,new,branch,checkpoint_output_routing=current['checkpoint_output_routing'])


def test_review_guard_and_wrong_registered_source_fail_before_load(tmp_path,monkeypatch):
    monkeypatch.setattr(m,'CONTROL_REVIEW_COMPLETE',False)
    with pytest.raises(ValueError,match='reviewed'):
        m.build_rear_live_swing_migration(tmp_path/'missing.pt',{},reason='synthetic')
    monkeypatch.setattr(m,'CONTROL_REVIEW_COMPLETE',True)
    cp=tmp_path/'wrong.pt';cp.write_bytes(b'synthetic not actual checkpoint')
    with pytest.raises(ValueError,match='registered'):
        m.build_rear_live_swing_migration(cp,{},reason='synthetic',project_root=tmp_path)


def test_populated_state_official_publish_reload_and_one_real_algorithm_update(tmp_path,monkeypatch):
    assert not torch.cuda.is_available(),'run with CUDA_VISIBLE_DEVICES=-1'
    torch.set_num_threads(1)
    meta,new,record,branch=lineage(tmp_path)
    runner=make()
    for parameter in list(runner.alg.actor.parameters())+list(runner.alg.critic.parameters()):
        parameter.grad=torch.ones_like(parameter)
    runner.alg.optimizer.step();runner.alg.optimizer.zero_grad()
    for state in runner.alg.optimizer.state.values():state['step'].fill_(33920.)
    for group in runner.alg.optimizer.param_groups:group['lr']=2.25e-5
    runner.alg.learning_rate=2.25e-5;runner.current_learning_iteration=1696
    meta.update(sampling='P01_full_task_only_initial_version',
        execution_topology=continuation_topology('P01_full_task_only_initial_version',None,observation_layout=LAYOUT),
        rr_postcross_workspace_branch={'counter_origin':m.SOURCE_COUNTS,'old_AUX':{'accepted':103,'attempted':104}},
        p05_capture_assist_migration={'old_AUX':{'accepted':7,'attempted':8}})
    cp,side=t.save_semantic_checkpoint(runner,branch/'checkpoints/history/source.pt',meta)
    source=json.loads(side.read_text())
    record[m.FACTOR_KEY].update(source_effective_learning_rate=2.25e-5,
        preserved_metadata_sha256={k:digest(source[k]) for k in (*m._preserved(source),'checkpoint_output_routing')})
    monkeypatch.setattr(m,'validate_rear_live_swing_migration',lambda *a,**kw:record)
    parent_pointer=branch.parent.parent/'checkpoints/checkpoint_last_pointer.json'
    own_pointer=branch/'checkpoints/checkpoint_last_pointer.json'
    assert not parent_pointer.exists() and not own_pointer.exists()
    output=branch/'checkpoints/history/published.pt'
    result=m.publish_rear_live_swing_checkpoint(cp,new,tmp_path/'synthetic_plan.json',output)
    before=torch.load(cp,weights_only=False);after=torch.load(output,weights_only=False)
    assert t.state_hash({k:v for k,v in before.items() if k!='infos'})==t.state_hash({k:v for k,v in after.items() if k!='infos'})
    assert result['save_load_round_trip'] and result['added_ppo_updates']==0
    assert not parent_pointer.exists() and not own_pointer.exists()
    current=json.loads(output.with_name(output.stem+'_manifest.json').read_text())
    for key in (*m._preserved(source),'checkpoint_output_routing'):assert current[key]==source[key]
    env=t.SemanticRslAdapter(Synthetic419Core(),seed=1001,device='cpu');env.cfg['semantic_version']='v3'
    learner=make(env);loaded=t.load_semantic_checkpoint(learner,output,contract=new,seed=1001)
    result=t.train_semantic(learner,env,run_dir=tmp_path/'synthetic_run',output_root=branch,
        stage='full_episode',decisions=128,contract=new,seed=1001,resume_infos=loaded,
        checkpoint_output_routing=loaded['checkpoint_output_routing'],checkpoint_interval_updates=1)
    assert tuple(result[k] for k in ('actual_policy_decisions','ppo_updates_this_run','optimizer_steps_this_run'))==(128,1,20)
    finalpath=Path(result['checkpoints'][-1]['checkpoint'])
    final=json.loads(finalpath.with_name(finalpath.stem+'_manifest.json').read_text())
    assert tuple(final[k] for k in m.COUNTERS)==(221696,1697,33940)
    assert final['rear_policy_timing_branch_counts']==dict(zip(m.COUNTERS,(1152,9,180)))
    for key in (m.MIGRATION,parent.MIGRATION,initial.MIGRATION,'rr_postcross_workspace_branch','p05_capture_assist_migration','checkpoint_output_routing'):
        assert final[key]==current[key]
    assert final['save_load_round_trip'] and not parent_pointer.exists() and own_pointer.exists()
    initial.validate_rear_policy_namespace(final,new,branch,checkpoint_output_routing=final['checkpoint_output_routing'])
    fresh=make();reloaded=t.load_semantic_checkpoint(fresh,finalpath,contract=new,seed=1001)
    assert t.state_hash(fresh.alg.optimizer.state_dict())==final['optimizer_state_sha256']
    assert reloaded['training_rng_state']==final['training_rng_state']

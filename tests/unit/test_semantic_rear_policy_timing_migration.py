"""Synthetic official state roundtrip; never counted as real PPO/Isaac credit."""
import copy
import json
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from wlr50_clean.ppo import semantic_training as t
from wlr50_clean.ppo import semantic_rear_policy_timing_migration as m
from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
from wlr50_clean.ppo.semantic_rr_capture_profile import RR_CAPTURE_POLICY,RR_CAPTURE_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_rear_policy_timing_profile import REAR_POLICY_TIMING_POLICY,REAR_POLICY_TIMING_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_migration import digest,continuation_topology

def make(width):
    return t.construct_semantic_runner(_shape_env(width,'cpu'),seed=1001,device='cpu',initialize_actor=False,
        policy_version=RR_CAPTURE_POLICY if width==410 else REAR_POLICY_TIMING_POLICY,
        observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT if width==410 else REAR_POLICY_TIMING_OBSERVATION_LAYOUT)[0]

def test_zero_append_complete_state_and_official_roundtrip(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    source=make(410)
    for p in list(source.alg.actor.parameters())+list(source.alg.critic.parameters()):p.grad=torch.ones_like(p)
    source.alg.optimizer.step();source.alg.optimizer.zero_grad()
    for s in source.alg.optimizer.state.values():s['step'].fill_(33760.)
    for g in source.alg.optimizer.param_groups:g['lr']=1e-5
    source.alg.learning_rate=1e-5;source.current_learning_iteration=1688
    old={'schema':'synthetic_cpu_no_physical_credit','experiment_id':'rr_capture_then_rl_transfer_v1'}
    base={**m.SOURCE_COUNTS,'seed':1001,'runtime_contract':old,'semantic_version':'v3',
        'stage_requested_decisions':dict(smoke=0,full_episode=114688,phase_suffix=95744),
        'historical_branch':{'counter_origin':{'global_policy_decisions':1},'AUX':{'accepted':7,'attempted':8}},
        'rr_signed_contact_v7_migration':{'immutable':'synthetic old receipt'},'sampling':'P01_full_task_only_initial_version',
        'execution_topology':continuation_topology('P01_full_task_only_initial_version',None,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)}
    cp,side=t.save_semantic_checkpoint(source,tmp_path/'source.pt',base)
    metadata=json.loads(side.read_text());new={'experiment_id':m.EXPERIMENT,'source_git_commit':'a'*40}
    target=make(419)
    record={'schema':m.SCHEMA,'plan_path':str(tmp_path/'plan.json'),'source_checkpoint_sha256':m.SOURCE_SHA,
        'target_contract_sha256':digest(new),m.FACTOR_KEY:{
            'target_policy_contract':t._runner_policy_contract(target),'source_role':'explicit_front_and_RR_validated_ancestor_continuation',
            'preserved_metadata_sha256':{k:digest(metadata[k]) for k in m.preserved_keys(metadata)}}}
    monkeypatch.setattr(m,'validate_rear_policy_timing_migration',lambda *a,**kw:record)
    mapped=m.zero_append_rear_policy_training_state(torch.load(cp,weights_only=False))
    original=torch.load(cp,weights_only=False)
    for role in ('actor','critic'):
        for name,v in original[role+'_state_dict'].items():
            n=mapped[role+'_state_dict'][name]
            assert torch.equal(v,n[:,:410] if name=='mlp.0.weight' else n)
            if name=='mlp.0.weight':assert torch.count_nonzero(n[:,410:])==0
    for i,state in original['optimizer_state_dict']['state'].items():
        for k,v in state.items():
            n=mapped['optimizer_state_dict']['state'][i][k]
            assert torch.equal(v,n[:,:410] if i in (0,6) and k!='step' else n)
    output=tmp_path/('ppo_'+m.EXPERIMENT)/'checkpoints/history/published.pt'
    result=m.publish_rear_policy_timing_checkpoint(cp,new,tmp_path/'plan.json',output)
    assert result['save_load_round_trip'] and result['rear_policy_timing_branch_counts']==dict.fromkeys(m.COUNTERS,0)
    published=torch.load(output,weights_only=False)
    assert t.state_hash({k:v for k,v in mapped.items() if k!='infos'})==t.state_hash({k:v for k,v in published.items() if k!='infos'})
    out=json.loads(output.with_name(output.stem+'_manifest.json').read_text())
    assert out['historical_branch']==metadata['historical_branch']
    assert out['training_rng_state']==metadata['training_rng_state']
    assert out['optimizer_learning_rate']==1e-5
    m.validate_rear_policy_namespace(out,new,output.parents[2])
    with pytest.raises(ValueError):m.validate_rear_policy_namespace(out,new,tmp_path/'old_output')

def test_mapping_rejects_partial_or_wrong_width():
    with pytest.raises((KeyError,ValueError)):m.zero_append_rear_policy_training_state({})

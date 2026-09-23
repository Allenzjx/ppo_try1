import copy
import json
from pathlib import Path
import pytest
from wlr50_clean.ppo.semantic_curriculum import plan_blocks, SCHEMA, EXPERIMENT

def config():
    return dict(schema=SCHEMA,experiment_id=EXPERIMENT,rollout_length=128,num_envs=1,entries=[
        dict(from_phase='P01',weight=.25,prefix_source='frozen_fsm'),
        dict(from_phase='P07',weight=.5,prefix_source='successful_nominal'),
        dict(from_phase='P10',weight=.25,prefix_source='checkpoint_policy')])

def test_real_plan_quantities_and_prefix_credit():
    plan=plan_blocks(config(),2048)
    assert [b['decisions'] for b in plan]==[512,1024,512]
    assert [b['stage'] for b in plan]==['full_episode','phase_suffix','phase_suffix']
    assert all(not b['prefix_policy_credit'] for b in plan)

def test_config_weights_are_consumed_not_cosmetic():
    c=config()
    for row,w in zip(c['entries'],(.5,.25,.25)):row['weight']=w
    assert [b['decisions'] for b in plan_blocks(c,2048)]==[1024,512,512]

@pytest.mark.parametrize('budget',[0,127,129,256,2049])
def test_no_partial_rollout(budget):
    with pytest.raises(ValueError):plan_blocks(config(),budget)

def test_suffix_cannot_teleport_or_borrow_credit():
    c=config();c['entries'][1]['prefix_source']='snapshot'
    with pytest.raises(ValueError):plan_blocks(c,2048)

def test_committed_curriculum_config_is_real():
    root=Path(__file__).resolve().parents[2]
    c=json.loads((root/'configs/ppo_rr_rl_timing_policy_learning_v1/curriculum_plan.json').read_text())
    assert [b['decisions'] for b in plan_blocks(c,4096)]==[1024,2048,1024]

def test_scheduler_really_invokes_three_entries_and_credits_only_sealed_updates(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from wlr50_clean.ppo import semantic_curriculum as c
    monkeypatch.setattr(c,'ROOT',tmp_path)
    directory=tmp_path/'outputs'/('ppo_'+EXPERIMENT)/'checkpoints/history'
    directory.mkdir(parents=True)
    source=directory/'source.pt';source.write_bytes(b'synthetic-no-weights')
    cfg=tmp_path/'curriculum_plan.json';cfg.write_text(json.dumps(config()))
    head='a'*40
    contract=dict(source_git_commit=head,experiment_id=EXPERIMENT,
        selected_configuration={cfg.name:{'sha256':c.sha(cfg)}})
    metadata=dict(runtime_contract=contract,rear_policy_timing_migration={'synthetic':True},
        global_policy_decisions=220544,ppo_updates=1688,optimizer_steps=33760)
    source.with_name('source_manifest.json').write_text(json.dumps(metadata))
    calls=[]
    def child(command,**kwargs):
        calls.append(command);arg=lambda name:command[command.index(name)+1]
        n=int(arg('-Decisions'));oldcp=Path(arg('-Checkpoint'))
        old=json.loads(oldcp.with_name(oldcp.stem+'_manifest.json').read_text())
        phase=arg('-FromPhase');run=tmp_path/f'run{len(calls)}';run.mkdir()
        updates=[dict(ppo_update=old['ppo_updates']+i+1,optimizer_steps=20) for i in range(n//128)]
        (run/'optimizer_updates.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in updates))
        (run/'advantage_audit.jsonl').write_text(''.join(json.dumps(dict(
            ppo_update_intended=x['ppo_update'],teacher_prefix_samples_included=False,
            by_request_phase={phase:{'sample_count':128}}))+'\n' for x in updates))
        target=directory/f'cp{len(calls)}.pt';target.write_bytes(b'synthetic-new-state')
        new={**old,'global_policy_decisions':old['global_policy_decisions']+n,
            'ppo_updates':old['ppo_updates']+n//128,'optimizer_steps':old['optimizer_steps']+n//128*20,
            'checkpoint_sha256':c.sha(target),'save_load_round_trip':True}
        target.with_name(target.stem+'_manifest.json').write_text(json.dumps(new))
        manifest=dict(lifecycle='SUCCEEDED',runtime_contract=contract,
            arguments=dict(experiment_id=EXPERIMENT,from_phase=phase,prefix_source=arg('-PrefixSource'),
                decisions=n,checkpoint=str(oldcp)),result=dict(actual_policy_decisions=n,
                ppo_updates_this_run=n//128,optimizer_steps_this_run=n//128*20,
                checkpoints=[{'checkpoint':str(target)}]))
        (run/'run_manifest.json').write_text(json.dumps(manifest))
        return SimpleNamespace(stdout=str(run))
    result=c.run_curriculum(configuration_path=cfg,source=source,source_sha256=c.sha(source),
        expected_head=head,decisions=2048,run_dir=tmp_path/'curriculum',execute=True,launcher=child)
    assert len(calls)==3 and [x[x.index('-FromPhase')+1] for x in calls]==['P01','P07','P10']
    assert result['actual_counts']==dict(global_policy_decisions=2048,ppo_updates=16,optimizer_steps=320)
    assert [result['actual_phase_samples'][p] for p in ('P01','P07','P10')]==[512,1024,512]
    assert all(not x['prefix_policy_credit'] for x in result['completed_blocks'])

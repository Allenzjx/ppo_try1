"""Read ONLY first two already-completed PPO updates; stdlib/no model imports."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import inspect_course as coverage

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0731401970292Z_g72e63592bdf4_8abca665b0f64439bdd233a9d189e513'


def completed_prefix(path,n):
    digest=hashlib.sha256(); result=[]
    with path.open('rb') as f:
        for _ in range(n):
            line=f.readline(1_000_001)
            if not line.endswith(b'\n') or len(line)>1_000_000:
                return None,None
            digest.update(line);result.append(json.loads(line))
        size=f.tell()
    return result,dict(path=str(path),rows=n,bytes=size,sha256=digest.hexdigest())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--updates',type=int,choices=(1,2),default=2)
    requested=parser.parse_args().updates
    count=requested*128
    path=RUN/'optimizer_updates.jsonl'
    if not path.exists():
        print('No complete optimizer-update evidence yet; prefix has zero PPO credit.');return
    updates,us=completed_prefix(path,requested)
    if updates is None:
        print('Fewer than requested complete optimizer-update records; no live rollout interpreted.');return
    assert [u['ppo_update'] for u in updates]==[1732,1733][:requested]
    assert [u['global_policy_decisions'] for u in updates]==[226176,226304][:requested]
    assert all(u['optimizer_steps']==20 and u['actor_parameters_changed'] for u in updates)
    samples,source=completed_prefix(RUN/'residual_and_projection_audit.jsonl',count)
    assert samples is not None
    assert [r['global_policy_decision'] for r in samples]==list(range(226049,226049+count))
    started=coverage.small_json(RUN/'run_manifest.started.json')
    support=coverage.bound_support(started['runtime_contract'])
    rows=[];ep=1;prior_tick=None
    for r in samples:
        a=r['applied_audit'];t=a['semantic_task'];ev=t['physical_evaluator'];tick=a['physics_tick']
        if prior_tick is not None and tick<prior_tick:ep+=1
        prior_tick=tick
        flags,physical=coverage.physical_windows(t,support)
        n=a['actuator_target_effect_audit'];p=r['policy_request']
        assert n['phase_mask_full12']==[1]*12 and not p['rear_task_assists_enabled']
        rows.append(dict(global_decision=r['global_policy_decision'],episode=ep,tick=tick,time_s=a['sim_time_s'],
            phase=a['end_phase_id'],terminal=r['terminal'],reason=a['termination_reason'],
            RR_placed=ev['history']['placed']['RR'],RL_placed=ev['history']['placed']['RL'],
            RR_current_qualified=ev['current_legs']['RR']['current_lift_valid'],
            RL_current_qualified=ev['current_legs']['RL']['current_lift_valid'],
            RR_ground=ev['current_legs']['RR']['ground_contact'],RL_ground=ev['current_legs']['RL']['ground_contact'],
            RR_contact_mode=ev['current_legs']['RR'].get('contact_mode'),
            RL_contact_mode=ev['current_legs']['RL'].get('contact_mode'),
            RL_current_qualified_tick=ev['current_legs']['RL'].get('current_lift_qualified_tick'),
            history_events=ev['history']['event_ticks'],windows=sorted(flags),physical=physical,
            owner_active=p['rear_owner_observed_features'][8:12],
            potential_semantics=a.get('reward_breakdown',{}).get('potential_semantics'),
            reward_only=a.get('reward_breakdown',{}).get('rr_retention_reward_only')))
    counts=Counter(w for r in rows for w in r['windows']);episodes=[]
    for en in range(1,ep+1):
        sub=[r for r in rows if r['episode']==en]
        episodes.append(dict(episode=en,count=len(sub),ticks=[sub[0]['tick'],sub[-1]['tick']],
            phases=dict(Counter(r['phase'] for r in sub)),terminal=sub[-1]['terminal'],reason=sub[-1]['reason'],
            RR_placed_ground=sum(r['RR_placed'] and r['RR_ground'] for r in sub),
            RR_current_bearing=sum(r['physical'].get('RR_actual_bearing') is True for r in sub),
            RL_current_qualified=sum(r['RL_current_qualified'] for r in sub),
            coverage=dict(Counter(w for r in sub for w in r['windows'])),
            first=sub[0],last=sub[-1]))
    prefix_result=None;handoff=None;prefix_sha=hashlib.sha256()
    with (RUN/'prefix_evidence.jsonl').open('rb') as f:
        for line in f:
            prefix_sha.update(line);r=json.loads(line)
            assert r['policy_credit'] is False
            if r['kind']=='checkpoint_prefix_result':prefix_result=r
            if r['kind']=='policy_credit_start':handoff=r['start'];break
    assert prefix_result is not None and handoff is not None and handoff['physics_tick']==6136
    assert rows[0]['tick']==6144
    qualified_rows=[r for r in rows if r['RL_current_qualified']]
    first_qualified=qualified_rows[0] if qualified_rows else None
    first_lost=next((r for r in rows if first_qualified and r['tick']>first_qualified['tick']
        and not r['RL_current_qualified']),None)
    result=dict(schema='readonly.72e_completed_prefix.v1',run=str(RUN),runtime_head=started['runtime_contract']['source_git_commit'],
        completed_update_evidence=updates,source_prefix=source,update_prefix=us,
        credit=dict(new_policy_decisions=count,PPO_updates=requested,Adam_steps=20*requested,prefix_decisions=0,partial_rollout=0),
        semantics=['Decision-end physical-window counts, not duration-normalized success rates.',
            'Only optimizer-confirmed requested prefix rows; no unfinished/live tail contributes.',
            'Successful nominal prefix is not policy learning or natural-P01 success.',
            coverage.FR_PROJECTION_SEMANTICS,
            'First-two-update coverage does not prove deterministic improvement or causal learning from reward-only revision.'],
        physical_windows={k:counts[k] for k in coverage.WINDOWS},episodes=episodes,
        first_episode_handoff=dict(prefix_result=prefix_result,handoff=handoff,
            evidence_prefix_sha256=prefix_sha.hexdigest(),first_learner_endpoint_tick=rows[0]['tick'],
            RR_placement_event_tick=rows[0]['history_events'].get('placed',{}).get('RR'),
            RR_placement_credit='successful nominal prefix, NOT learner first placement',
            first_RL_qualified_after_handoff=first_qualified,
            first_RL_qualification_loss=first_lost,
            note='Post-handoff occurrence is student-controlled suffix evidence, not proof that reward/PPO update caused the action; first rollout used frozen CP226048 weights.'),
        RR_placed_ground_count=sum(r['RR_placed'] and r['RR_ground'] for r in rows),
        RL_qualified_count=sum(r['RL_current_qualified'] for r in rows))
    target=OUT/f'72e_first{count}_completed_readonly.json'
    assert not target.exists(),'preserve prior completed snapshot'
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(target),credit=result['credit'],physical_windows=result['physical_windows'],
        episodes=[{k:v for k,v in e.items() if k not in ('first','last')} for e in episodes]),indent=2))


if __name__=='__main__':main()

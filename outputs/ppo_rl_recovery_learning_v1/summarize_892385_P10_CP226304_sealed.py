"""Bounded stdlib audit AFTER CP226304 P10 course seals; no live polling.

Reuses the completed512 likelihood/physical definitions. At most two official
512 updates are allowed; counts come from actual receipts, not requested1024.
No tensor/checkpoint model deserialization or robot imports are performed.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUN=ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T1717242165088Z_g892385cba8a7_afdc21482bbc480a9692fb63fe8ad858'


def helpers():
    spec=importlib.util.spec_from_file_location('sealed_front_branch_course',OUT/'summarize_892385_P07_sealed.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analyze(run):
    h=helpers()
    result=h.analyze(run,expected_source=226304,maximum_updates=2,segment_only=True)
    result['schema']='outputs.bounded_892385_P10_CP226304_sealed.v1'
    prior=json.loads((OUT/'892385_P07_CP225792_512_coverage.json').read_text(encoding='utf-8'))
    old=prior['combined_branch_1024']; now=result['summary']
    if prior['checkpoint']['global_policy_decisions']!=226304:
        raise ValueError('prior report is not the sealed CP226304 source')
    new_qual=sum(len(e['events']['RL']['fresh_qualification_ticks']) for e in result['episodes'])
    new_capture=sum(e['events']['RR']['event_credit'].get('placed')=='student' for e in result['episodes'])
    # Do not call first ground->TOP first capture a recapture; helper separately
    # records AIR recontact and recovery after a measured post-capture ground.
    result['current_course_events']=dict(
        student_RR_first_capture_events=new_capture,
        student_RL_fresh_qualification_events=new_qual,
        student_RL_qualified_endpoints=sum(e['current_contact_counts'].get('RL_student_qualified_endpoints',0) for e in result['episodes']),
        inherited_or_unresolved_RL_qualified_endpoints=sum(e['current_contact_counts'].get('RL_inherited_or_unresolved_qualified_endpoints',0) for e in result['episodes']),
        student_RL_cross_events=sum(e['events']['RL']['event_credit'].get('front_edge_crossed')=='student' for e in result['episodes']),
        student_RL_placed_events=sum(e['events']['RL']['event_credit'].get('placed')=='student' for e in result['episodes']))
    result['combined_branch_actual']=dict(branch_origin=225280,
        actual_policy_decisions=old['actual_policy_decisions']+now['optimized_decisions'],
        ppo_updates=old['ppo_updates']+now['ppo_updates'],optimizer_steps=old['optimizer_steps']+now['optimizer_steps'],
        request_phase_counts={p:old['request_phase_counts'][p]+n for p,n in now['actual_request_phase_counts'].items()},
        endpoint_physical_windows={p:old['endpoint_physical_windows'][p]+n for p,n in now['endpoint_physical_window_counts'].items()},
        prefix_decisions_excluded=old['prefix_decisions']+now['prefix'].get('decisions',0),
        prefix_physics_ticks_excluded=old['prefix_physics_ticks']+now['prefix'].get('physics_ticks',0),
        prefix_PPO_credit=0,replay_exposures=old['replay_exposures']+result['actual_replay_exposures'],
        fresh_student_RR_capture_events=old['fresh_student_RR_capture_events']+new_capture,
        fresh_student_RL_qualification_events=old['fresh_student_RL_qualification_events']+new_qual)
    result['notes']=['Training lifecycle success is not physical task success.',
        'Teacher prefix forms real states but has zero PPO credit; no snapshot teleport is inferred.',
        'Fresh versus inherited events are attributed against each actual handoff tick.',
        'RR TOP recovery/phase labels do not themselves prove a renewed qualified traversal.',
        'Each optimized raw index has five likelihood exposures verified from official JSON receipts; no tensor/model loaded.',
        'Offline front replay is counted separately; heldout KL is not physical front-retention proof.',
        'CP226304 formal video report remains unchanged; new checkpoint is not evaluated by this coverage report.']
    return result


def markdown(x):
    s=x['summary'];ck=x['checkpoint'];tot=x['combined_branch_actual']
    lines=['# 892385 P10 course from CP226304 — actual sealed coverage','',
        f"Run `{s['run']}`. Actual {s['optimized_decisions']} learner decisions / {s['ppo_updates']} PPO updates / {s['optimizer_steps']} official Adam steps.",
        f"Saved CP{ck['global_policy_decisions']} / PPO{ck['ppo_updates']} / Adam{ck['optimizer_steps']}; roundtrip true.",
        f"Request phases: `{s['actual_request_phase_counts']}`.",
        f"Prefix excluded: `{s['prefix']}` (zero credit).",'',
        '| Episode | Learner n | Handoff | End phase/time | Terminal cause | RR placement ownership | RL fresh qualifications |',
        '| --- | ---: | --- | --- | --- | --- | --- |']
    for e in x['episodes']:
        rr=e['events']['RR']; reason=e['physical_reason'] or e['termination_reason'] or 'nonterminal budget tail'
        lines.append(f"| {e['episode_index']} | {e['student_decisions']} | {e['handoff_tick']} / {e['handoff_s']:.6f}s | {e['endpoint_phase']} / {e['endpoint_s']:.6f}s | {reason} | {rr['event_credit'].get('placed','none')} {rr['history_event_ticks'].get('placed','')} | {e['events']['RL']['fresh_qualification_ticks']} |")
    lines += ['',f"Current-course event counts: `{x['current_course_events']}`.",'',
        '| Physical endpoint window (nonexclusive) | Samples |','| --- | ---: |']
    lines += [f'| {k} | {n} |' for k,n in s['endpoint_physical_window_counts'].items()]
    lines += ['',x['physical_proxy_semantics'],'',
        '| Episode | RR legal TOP endpoints | TOP losses | AIR->TOP recontacts | Post-ground TOP restoration |',
        '| --- | ---: | --- | --- | --- |']
    for e in x['episodes']:
        lines.append(f"| {e['episode_index']} | {e['current_contact_counts'].get('RR_legal_TOP_bearing_endpoints',0)} | {e['RR_TOP_loss_endpoint_ticks']} | {[v['tick'] for v in e['RR_legal_TOP_recontact_without_reground']]} | {[v['tick'] for v in e['RR_legal_TOP_restore_after_ground']]} |")
    lines += ['', '## Official optimization and separate replay','',
        f"Five official likelihood exposures per optimized raw index; max independent old Gaussian logp error {s['maximum_old_raw_logp_error']:.9g}. "
        f"Actual replay exposures={x['actual_replay_exposures']}; added on-policy samples=0, separate AUX Adam=0."]
    for r in x['replay_by_update']:
        lines.append(f"- Completed PPO{r['ppo_update']} / global{r['global_policy_decisions']}: {r['actual_replay_row_exposures']} replay exposures; heldout after `{r['after']['heldout_rows']}`.")
    lines += ['',f"Collection actor semantics: {x['collection_actor_semantics']}",
        f"Checkpoint `{ck['checkpoint']}`; SHA `{ck['sha256']}`; sidecar SHA `{ck['sidecar_sha256']}`.",'',
        '## Branch actual totals','',json.dumps(tot,ensure_ascii=False),'']
    lines += ['- '+s for s in x['notes']]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--report',default='892385_P10_CP226304_course_coverage')
    args=parser.parse_args()
    if Path(args.report).name!=args.report:raise ValueError('report must be basename')
    paths=[OUT/(args.report+ext) for ext in ('.json','.md')]
    if any(p.exists() for p in paths):raise FileExistsError('preserve existing reports')
    report=analyze(args.run.resolve(strict=True))
    for path,text in zip(paths,(json.dumps(report,ensure_ascii=False,indent=2)+'\n',markdown(report))):
        with path.open('x',encoding='utf-8') as stream:stream.write(text)
    print(json.dumps({'reports':[str(p) for p in paths],'actual_decisions':report['summary']['optimized_decisions'],
        'PPO_updates':report['summary']['ppo_updates'],'checkpoint':report['checkpoint']},ensure_ascii=False))


if __name__=='__main__':main()

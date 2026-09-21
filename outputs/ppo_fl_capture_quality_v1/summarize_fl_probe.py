"""Read a sealed bounded probe; never reads active runs or claims PPO success."""
import argparse
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def window(path,start,stop):
    with path.open(encoding='utf-8') as stream:
        yield from (json.loads(line) for line in itertools.islice(stream,start,stop))


def stats(values):
    v=list(values)
    return {'min':min(v),'max':max(v),'mean':sum(v)/len(v)} if v else None


def summarize(directory):
    m=json.loads((directory/'run_manifest.json').read_text())
    if m.get('lifecycle') not in ('DIAGNOSTIC_SEALED','DIAGNOSTIC_ERROR') or not m.get('completed_at_utc'):
        raise ValueError('only sealed diagnostic runs may be analyzed')
    decisions=list(rows(directory/'probe_decisions.jsonl'))
    entry=m.get('probe_entry')
    start=max(1,entry['tick']-16) if entry else max(1,m.get('endpoint_tick',16)-16)
    end=m.get('endpoint_tick',decisions[-1]['end_tick'] if decisions else 0)
    if end-start>1200:
        raise ValueError('diagnostic post-entry window unexpectedly exceeds ten seconds')
    phases={}
    for d in decisions:
        label=(d.get('intervention') or {}).get('state','pre')
        for tick in range(max(start,d['start_tick']+1),d['end_tick']+1):
            phases[tick]=label
    samples=[]
    for physical,native in zip(window(directory/'physical_observations.jsonl',start,end+1),
                               window(directory/'native_tick_audit.jsonl',start-1,end),strict=True):
        tick=physical['physics_tick']
        if tick!=native['episode_physics_tick']:
            raise ValueError('physical/native tick misalignment')
        audit=native['native_audit']; w=physical['wheels']['front_left_ankle']
        joints=[physical['joints']['front_left_'+name] for name in ('hip','knee')]
        contact=physical['contacts']['front_left_wheel']
        reference=audit['tracking_reference_evidence']
        support={}
        for leg,name in (('FR','front_right'),('RL','rear_left'),('RR','rear_right')):
            c=physical['contacts'][name+'_wheel']
            support[leg]={key:{f:c[key][f] for f in ('active','pair_verified','normal_force_n')} for key in ('ground','obstacle')}
        headroom=audit.get('policy_headroom_evidence',{})
        samples.append(dict(tick=tick,stage=native['source_phase_id'],probe_state=phases[tick],
            FL_N_deg=native['nominal_full12'][:2],FL_mapped_N_deg=audit['native_drive_target_full12'][:2],
            FL_controller_deg=audit['controller_drive_bias_full12'][:2],
            FL_requested_projected_residual_deg=native['projected_residual_full12'][:2],
            FL_effective_headroom_residual_deg=headroom.get('effective_policy_residual_full12',[])[:2],
            FL_target_deg=[j['command_deg'] for j in joints],FL_actual_deg=[j['position_deg'] for j in joints],
            FL_tracking_error_deg=[j['command_deg']-j['position_deg'] for j in joints],
            FL_actual_velocity_deg_s=[j['velocity_deg_s'] for j in joints],
            FL_mapper_pre_compensation_deg=reference['mapper_pre_state']['tracking_compensation_deg'][:2],
            FL_feedback=reference['channels'][:2],FL_raw=audit['raw_policy_action_full12'][:2],
            FL_gap_m=w['bottom_w_m'][2]-physical['obstacle']['top_z_m'],
            FL_front_distance_m=w['center_w_m'][0]-physical['obstacle']['front_x_m'],
            FL_contact_class=contact['contact_class'],
            FL_ground=contact['ground'],FL_obstacle=contact['obstacle'],other_contacts=support,
            body=physical['base'],CoM=physical['center_of_mass'],
            mask=audit['phase_mask_full12'],dispatch_verified=audit['verified'],
            same_tick_effect=audit['actual_mapping_matches_dispatch'],
            full12_target=[physical['joints'][name]["command_deg"] for name in (
                'front_left_hip','front_left_knee','front_right_hip','front_right_knee',
                'rear_left_hip','rear_left_knee','rear_right_hip','rear_right_knee')]+
                [physical['wheels'][name+'_ankle']['command_rad_s'] for name in ('front_left','front_right','rear_left','rear_right')]))
    groups=defaultdict(list)
    for r in samples:groups[r['probe_state']].append(r)
    aggregate={}
    for key,rs in groups.items():
        aggregate[key]=dict(ticks=[rs[0]['tick'],rs[-1]['tick']],duration_s=len(rs)/120,
            FL_gap_mm=stats(r['FL_gap_m']*1000 for r in rs),
            FL_target_deg=[stats(r['FL_target_deg'][i] for r in rs) for i in range(2)],
            FL_actual_deg=[stats(r['FL_actual_deg'][i] for r in rs) for i in range(2)],
            FL_projected_residual_deg=[stats(r['FL_requested_projected_residual_deg'][i] for r in rs) for i in range(2)],
            FL_abs_tracking_error_deg=[stats(abs(r['FL_tracking_error_deg'][i]) for r in rs) for i in range(2)],
            active_FL_obstacle_samples=sum(r['FL_obstacle']['active'] for r in rs),
            active_FL_ground_samples=sum(r['FL_ground']['active'] for r in rs),
            contact_classes=dict(Counter(r['FL_contact_class'] for r in rs)),
            stages=dict(Counter(r['stage'] for r in rs)),
            mapper_compensation_deg=[stats(r['FL_mapper_pre_compensation_deg'][i] for r in rs) for i in range(2)],
            first=rs[0],last=rs[-1])
    tasks=[dict(tick=d['end_tick'],phase=d['step_info']['semantic_task']['stage_id'],
        history=d['step_info']['semantic_task']['physical_evaluator']['history'],
        current_FL=d['step_info']['semantic_task']['physical_evaluator']['current_legs']['FL'],
        other_support={k:v['support'] for k,v in d['step_info']['semantic_task']['physical_evaluator']['current_legs'].items() if k!='FL'},
        intervention=d['intervention']) for d in decisions if d['end_tick']>=start]
    first_contact=next((r['tick'] for r in samples if r['FL_obstacle']['active']),None)
    first_placed=next((t['tick'] for t in tasks if t['history']['placed']['FL']),None)
    first_p06=next((t['tick'] for t in tasks if t['phase']=='P06'),None)
    report=dict(schema='wlr50_clean.sealed_FL_probe_summary.v1',source=str(directory.resolve()),
        lifecycle=m['lifecycle'],case=m['case'],diagnostic_only=True,optimizer_updates=0,
        probe_started=bool(entry),probe_complete=m.get('probe_complete',False),
        natural_task_terminal=m.get('natural_task_terminal'),original_task_reason=m.get('original_task_reason'),
        bounded_window=[start,end],first_obstacle_contact_tick=first_contact,
        first_placed_decision_tick=first_placed,first_P06_decision_tick=first_p06,
        final_task=tasks[-1] if tasks else None,groups=aggregate,actual_physics_rows=samples,decision_snapshots=tasks,
        caveats=['External controllability intervention, not trained-PPO or formal full success.',
            'Other ten policy channels react on actual observations; not a rigid-base isolated response.',
            'Mapper pre-state, filtered request, headroom-effective offset, final target and physical response are separate.',
            'P06 entry and current contact retention are reported separately from historical placement.'])
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output-stem',type=Path,required=True)
    a=p.parse_args();report=summarize(a.run)
    with a.output_stem.with_suffix('.json').open('x',encoding='utf-8') as s:json.dump(report,s,indent=2,allow_nan=False)
    lines=[f"# {report['case']} — sealed external FL diagnostic",'',
        f"Started: {report['probe_started']}; complete: {report['probe_complete']}; original terminal: {report['original_task_reason']}.",
        f"First obstacle contact: {report['first_obstacle_contact_tick']}; placed decision: {report['first_placed_decision_tick']}; P06 decision: {report['first_P06_decision_tick']}.",'',
        '| Window | ticks | FL gap min/mean/max mm | hip target mean / actual mean deg | obstacle-contact samples |',
        '|---|---|---|---|---|']
    for k,v in report['groups'].items():
        g=v['FL_gap_mm'];lines.append(f"| {k} | {v['ticks']} | {g['min']:.3f} / {g['mean']:.3f} / {g['max']:.3f} | {v['FL_target_deg'][0]['mean']:.4f} / {v['FL_actual_deg'][0]['mean']:.4f} | {v['active_FL_obstacle_samples']} |")
    lines.extend(['',*report['caveats'],'','Full raw execution chain and final current support are in the paired JSON.'])
    with a.output_stem.with_suffix('.md').open('x',encoding='utf-8') as s:s.write('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ('case','probe_started','probe_complete','first_obstacle_contact_tick','first_placed_decision_tick','first_P06_decision_tick')}))


if __name__=='__main__':main()

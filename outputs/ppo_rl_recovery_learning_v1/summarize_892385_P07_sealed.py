"""One sealed 512 P07 block; stdlib JSON receipts only, never Torch/models.

No live polling. Reuses inspect_course physical/likelihood definitions unchanged.
Run only after the parent confirms the complete update/save boundary. Outputs
are new report artifacts; existing reports are never overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
from itertools import islice
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T1527536734451Z_g892385cba8a7_f00e6fb48d074b26a2a98898505632e6'


def helper():
    spec = importlib.util.spec_from_file_location('sealed_course', OUT/'inspect_course.py')
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    # New replay receipts contain a nested same-name field. Select TOP LEVEL,
    # never the old regex's first occurrence; do not modify the shared helper.
    def official_minibatches(path):
        with path.open(encoding='utf-8-sig') as stream:
            document = json.load(stream)
        yield from document['minibatches']
    h.minibatches = official_minibatches
    return h


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def sidecar_receipt(path):
    """Read only unique top-level scalar receipts from the pretty JSON sidecar.

    This source's duplicated ancestry is 162 MB; do not deserialize that whole
    object beside Isaac. Two-space root scalar lines are unambiguous in the
    verified artifact format. Fail on missing/duplicate/type mismatch.
    """
    keys={'checkpoint_sha256', 'save_load_round_trip', 'global_policy_decisions',
          'ppo_updates', 'optimizer_steps'}
    result={}
    pattern=re.compile(r'^  "([^"]+)": (.*?)(?:,)?\s*$')
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            match=pattern.match(line)
            if match and match[1] in keys:
                if match[1] in result:
                    raise ValueError('duplicate top-level checkpoint receipt')
                value=match[2].rstrip().removesuffix(',')
                result[match[1]]=json.loads(value)
    if set(result) != keys:
        raise ValueError('missing root checkpoint receipt; unsupported formatting')
    return result


def analyze(run, *, expected_source=225792, maximum_updates=1, segment_only=False):
    h = helper()
    # Read the small seal FIRST, before any learner, prefix or saved-update file.
    manifest = h.small_json(run/'run_manifest.json')
    h.require(manifest.get('completed_at_utc') and manifest['lifecycle'] in
              ('SUCCEEDED', 'STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'), 'not a sealed successful training lifecycle')
    train = h.small_json(run/'training_manifest.json')
    sample_n = train['actual_policy_decisions']
    update_n = train['ppo_updates_this_run']
    h.require(train['runner_config']['num_steps_per_env'] == 512 and
              1 <= update_n <= maximum_updates and sample_n == 512 * update_n,
              'only actual complete 512 updates within the requested bound are credited')
    h.require(train['runtime_contract']['source_git_commit'].startswith('892385'), 'unexpected frozen runtime')
    summary = h.inspect_run(run)
    data = list(islice(h.rows(run/'residual_and_projection_audit.jsonl'), sample_n))
    h.require(len(data) == sample_n, 'missing optimized rows')
    h.require(data[0]['global_policy_decision'] == expected_source+1, 'not the requested source continuation')
    handoffs = [r['start'] for r in h.rows(run/'prefix_evidence.jsonl') if r.get('kind') == 'policy_credit_start']
    ends = [i+1 for i, row in enumerate(data) if row['terminal']]
    if not ends or ends[-1] < sample_n:
        ends.append(sample_n)
    h.require(len(handoffs) == len(ends), 'cannot align actual prefix and student episode')
    complete = list(h.rows(run/'completed_episodes.jsonl', optional=True))
    h.require(len(complete) == sum(r['terminal'] for r in data), 'completed episode receipt mismatch')
    episodes, begin = [], 0
    floor = summary['bound_support']['force_noise_floor_n']
    for index, end in enumerate(ends):
        block = data[begin:end]
        handoff = handoffs[index]
        handoff_tick = handoff['physics_tick']
        last = block[-1]['applied_audit']
        task = last['semantic_task']; ev = task['physical_evaluator']
        counts = Counter(); contacts = Counter(); events = {}
        rr_top_previous = None; rr_ever_top = False; rr_regrounded_after_top = False
        rr_capture_seen = False
        top_loss_ticks, restore_after_ground, recontact_without_ground = [], [], []
        qualified_ticks = {'RR': set(), 'RL': set()}
        event_ticks = {'RR': {}, 'RL': {}}
        for row in block:
            a = row['applied_audit']; t = a['semantic_task']; e = t['physical_evaluator']
            tick = a['physics_tick']; legs = e['current_legs']; history = e['history']
            flags, _ = h.physical_windows(t, summary['bound_support']); counts.update(flags)
            rr = legs['RR']; rl = legs['RL']
            top = h.bearing(rr, floor, top=True)
            contacts['RR_legal_TOP_bearing_endpoints'] += int(top)
            contacts['RR_ground_endpoints'] += int(rr.get('ground_contact') is True)
            contacts['RR_AIR_endpoints'] += int(rr.get('air') is True)
            placed_tick = history.get('event_ticks', {}).get('placed', {}).get('RR')
            rr_capture_seen |= isinstance(placed_tick, int) and placed_tick > handoff_tick
            if rr_top_previous is True and not top:
                top_loss_ticks.append(tick)
            if top and rr_top_previous is False and rr_ever_top:
                target = restore_after_ground if rr_regrounded_after_top else recontact_without_ground
                target.append(dict(tick=tick, current_lift_valid=rr.get('current_lift_valid'),
                    bearing_force_n=rr.get('bearing_force_n'), is_renewed_task_placement_claim=False))
                rr_regrounded_after_top = False
            if rr_capture_seen and not top:
                contacts['RR_nonbearing_after_student_capture_endpoints'] += 1
            rr_regrounded_after_top |= rr_ever_top and rr.get('ground_contact') is True
            rr_ever_top |= top
            rr_top_previous = top
            for leg in ('RR', 'RL'):
                current = legs[leg]
                for kind, byleg in history.get('event_ticks', {}).items():
                    value = byleg.get(leg)
                    if isinstance(value, int):
                        event_ticks[leg][kind] = value
                for event in history.get('lift_attempt_events', []):
                    if event.get('leg') == leg and event.get('event') == 'qualified_measured_upward_lift':
                        qualified_ticks[leg].add(event['physics_tick'])
                if h.qualified(current):
                    qt = current.get('current_lift_qualified_tick')
                    if not isinstance(qt, int):
                        qt = history.get('event_ticks', {}).get('active_lift', {}).get(leg)
                    key = 'student_qualified_endpoints' if isinstance(qt, int) and qt > handoff_tick else 'inherited_or_unresolved_qualified_endpoints'
                    contacts[leg+'_'+key] += 1
        for leg in ('RR', 'RL'):
            events[leg] = dict(history_event_ticks=event_ticks[leg],
                event_credit={kind: ('student' if tick > handoff_tick else 'prefix') for kind,tick in event_ticks[leg].items()},
                fresh_qualification_ticks=sorted(t for t in qualified_ticks[leg] if t > handoff_tick),
                inherited_qualification_ticks=sorted(t for t in qualified_ticks[leg] if t <= handoff_tick))
        terminal = block[-1]['terminal']
        if terminal:
            receipt = complete[index]
            h.require(receipt['policy_decisions'] == len(block) and receipt['terminal_info']['physics_tick'] == last['physics_tick'], 'terminal receipt mismatched')
        legs_compact = {leg:{k:val.get(k) for k in ('contact_mode','top_contact','ground_contact','within_top_xy',
            'current_lift_valid','bearing_force_n','clearance_m','front_distance_m')} for leg,val in ev['current_legs'].items()}
        safety = None
        if terminal and last['termination_reason'] == 'HARD_JOINT_LIMIT':
            obs = block[-1]['terminal_observation']['policy'][0]
            audit = last['actuator_target_effect_audit']
            safety = dict(reason=ev.get('reason'), actual_servo_canonical_deg=[v*90 for v in obs[38:46]],
                actual_quantization='439 fixed-scale float32 terminal observation; not native raw sensor',
                final_servo_canonical_deg=last['actual_drive_target_full12'][:8],
                actual_native_target_rad=audit['actual_native_targets']['servo_position_rad'],
                hard_limits_deg=audit['policy_headroom_evidence']['servo_hard_limits_deg'],
                mapping_matches_dispatch=audit['actual_mapping_matches_dispatch'])
        episodes.append(dict(episode_index=index, student_decisions=len(block),
            global_range=[block[0]['global_policy_decision'],block[-1]['global_policy_decision']],
            handoff_tick=handoff_tick,handoff_s=handoff['sim_time_s'],prefix_credit=0,
            from_current_policy_P01=handoff['from_P01_current_policy'],
            request_phase_counts=dict(Counter(r['applied_audit']['phase_id'] for r in block)),
            terminal=terminal,endpoint_tick=last['physics_tick'],endpoint_s=last['sim_time_s'],endpoint_phase=last['end_phase_id'],
            termination_reason=last.get('termination_reason'),termination_source=task.get('termination_source'),
            physical_reason=ev.get('reason'),safety=safety,events=events,current_contact_counts=dict(contacts),
            RR_TOP_loss_endpoint_ticks=top_loss_ticks,RR_legal_TOP_restore_after_ground=restore_after_ground,
            RR_legal_TOP_recontact_without_reground=recontact_without_ground,
            physical_windows={key:counts[key] for key in h.WINDOWS},endpoint_legs=legs_compact))
        begin = end
    updates = list(h.rows(run/'optimizer_updates.jsonl'))
    h.require(len(updates) == update_n, 'completed update receipts disagree')
    replays = []
    for update in updates:
        replay = update['front_replay_regularization']
        h.require(update['optimizer_steps'] == 20 and replay['actual_replay_row_exposures'] == 640,
                  'actual per-update Adam/replay counts differ from 20/640')
        h.require(replay['on_policy_samples_added'] == 0 and replay['separate_auxiliary_optimizer_steps'] == 0,
                  'replay credit semantics changed')
        h.require(len(replay['minibatches']) == 20 and all(x['gradient_consumed_once'] for x in replay['minibatches']),
                  'replay gradient receipt inconsistent')
        replays.append(dict(ppo_update=update['ppo_update'],global_policy_decisions=update['global_policy_decisions'],
            **{k:v for k,v in replay.items() if k not in ('minibatches','spec')}))
    ck = train['checkpoints'][-1]
    side = sidecar_receipt(Path(ck['manifest']))
    checkpoint_hash = sha(Path(ck['checkpoint'])); sidecar_hash = sha(Path(ck['manifest']))
    h.require(checkpoint_hash == side['checkpoint_sha256'] and side['save_load_round_trip'] is True,
              'saved checkpoint hash/reload mismatch')
    h.require(side['global_policy_decisions'] == summary['final_global_policy_decisions'], 'checkpoint counter mismatch')
    if segment_only:
        return dict(schema='outputs.bounded_892385_sealed_segment.v1',summary=summary,episodes=episodes,
            replay_by_update=replays,actual_replay_exposures=sum(r['actual_replay_row_exposures'] for r in replays),
            checkpoint={**ck,'sha256':checkpoint_hash,'sidecar_sha256':sidecar_hash,
                'ppo_updates':side['ppo_updates'],'optimizer_steps':side['optimizer_steps'],'roundtrip':True},
            source_checkpoint_decisions=expected_source,planned_maximum_updates=maximum_updates,
            collection_actor_semantics='each sample uses the actor active before its consuming update; later rollout can use prior completed update',
            physical_proxy_semantics=h.FR_PROJECTION_SEMANTICS)
    h.require(update_n == 1 and expected_source == 225792, 'legacy P07 combined-report path is one exact known block')
    prior = h.small_json(OUT/'892385_P12_front_preserved_512_coverage.json')
    h.require(prior['final_global_policy_decisions'] == 225792 and prior['optimized_decisions'] == 512,
              'prior sealed P12 report identity differs')
    combined = dict(source_reports=['892385_P12_front_preserved_512_coverage.json',
        'this sealed P07 block'], branch_origin=225280, actual_policy_decisions=1024,
        ppo_updates=prior['ppo_updates']+summary['ppo_updates'],
        optimizer_steps=prior['optimizer_steps']+summary['optimizer_steps'],
        request_phase_counts={p:prior['actual_request_phase_counts'][p]+summary['actual_request_phase_counts'][p]
            for p in h.PHASES},
        endpoint_physical_windows={p:prior['endpoint_physical_window_counts'][p]+summary['endpoint_physical_window_counts'][p]
            for p in h.WINDOWS},
        prefix_decisions=prior['prefix']['decisions']+summary['prefix']['decisions'],
        prefix_physics_ticks=prior['prefix']['physics_ticks']+summary['prefix']['physics_ticks'],prefix_credit=0,
        fresh_student_RR_capture_events=sum(e['events']['RR']['event_credit'].get('placed')=='student' for e in episodes),
        fresh_student_RL_qualification_events=prior['physical_claims']['learner_RL_fresh_qualification_count']+
            sum(len(e['events']['RL']['fresh_qualification_ticks']) for e in episodes),
        fresh_student_RL_qualified_endpoints=prior['physical_claims']['learner_RL_qualified_endpoints']+
            sum(e['current_contact_counts'].get('RL_student_qualified_endpoints',0) for e in episodes),
        prefix_inherited_RL_qualified_endpoints=prior['physical_claims']['prefix_inherited_RL_qualified_endpoints']+
            sum(e['current_contact_counts'].get('RL_inherited_or_unresolved_qualified_endpoints',0) for e in episodes),
        RL_cross_or_placed_events=0,full_task_successes=0,
        replay_exposures=prior['replay']['actual_replay_row_exposures']+replay['actual_replay_row_exposures'],
        separate_AUX_Adam_steps=0,pre_update_actor_trajectories_not_new_checkpoint_validation=True)
    return dict(schema='outputs.bounded_892385_P07_sealed.v1', summary=summary,episodes=episodes,
        combined_branch_1024=combined,
        replay={k:v for k,v in replay.items() if k not in ('minibatches','spec')},
        checkpoint={**ck, 'sha256':checkpoint_hash,'sidecar_sha256':sidecar_hash,
            'ppo_updates':side['ppo_updates'],'optimizer_steps':side['optimizer_steps'],'roundtrip':True},
        statements=['Lifecycle completion is not physical task success.',
            'All trajectories were collected by pre-update actor, not proof of new checkpoint gains.',
            h.FR_PROJECTION_SEMANTICS,
            'Legal TOP restoration after GROUND is not automatically renewed valid traversal.',
            'Official raw-sample association uses saved index/old-logp receipts; no tensor/model loaded.',
            'Heldout KL is distribution retention, not physical front-success evidence.'])


def markdown(report):
    s=report['summary']; ck=report['checkpoint']; r=report['replay']
    text=['# 892385 P07 continuous student block — sealed', '',
        f"Run `{s['run']}`; lifecycle {s['lifecycle']} is training completion, not task success.", '',
        f"Actual {s['optimized_decisions']} decisions / {s['ppo_updates']} PPO / {s['optimizer_steps']} Adam. "
        f"Final CP{ck['global_policy_decisions']} / PPO{ck['ppo_updates']} / Adam{ck['optimizer_steps']}.",
        f"Request phases: `{s['actual_request_phase_counts']}`.",
        f"Teacher prefix: `{s['prefix']}`; zero learning credit.", '',
        '| Episode | Student n | Handoff tick | Endpoint | Terminal | RR learner events | RL fresh qualifications |',
        '| --- | ---: | ---: | --- | --- | --- | --- |']
    for e in report['episodes']:
        rr=e['events']['RR']; fresh={k:v for k,v in rr['history_event_ticks'].items() if rr['event_credit'][k]=='student'}
        reason=e['physical_reason'] or e['termination_reason'] or 'nonterminal budget tail'
        text.append(f"| {e['episode_index']} | {e['student_decisions']} | {e['handoff_tick']} | {e['endpoint_phase']} {e['endpoint_s']:.6f}s | {reason} | {fresh} | {e['events']['RL']['fresh_qualification_ticks']} |")
    text += ['', 'Physical endpoint windows (nonexclusive):', '', '| Window | Samples |', '| --- | ---: |']
    text += [f'| {k} | {v} |' for k,v in s['endpoint_physical_window_counts'].items()]
    text += ['', f"Official likelihood: each optimized raw index appears five times; {s['optimizer_steps']} minibatches. "
        f"Old raw Gaussian max independent error {s['maximum_old_raw_logp_error']:.9g}.",
        f"Replay {r['actual_replay_row_exposures']} repeated offline exposures, added PPO samples=0, separate AUX Adam=0. "
        f"Heldout after KL: `{r['after']['heldout_rows']}`.", '',
        f"Checkpoint `{ck['checkpoint']}`; SHA `{ck['sha256']}`; sidecar `{ck['sidecar_sha256']}`; roundtrip true.", '',
        'Detailed fresh/prefix event ownership, current TOP losses, terminal safety joint/actual/target and final legs are in JSON.', '']
    text += ['- '+x for x in report['statements']]
    text += ['', '## RR contact continuity and combined branch credit', '',
        'The two student RR placements occur in episodes 1 and 4, after their real P07 prefix handoffs. '
        'They are not teacher-produced captures, but both subsequently lose usable TOP support. '
        'Initial GROUND -> first TOP is first capture, never labelled post-capture recapture.', '',
        '| Episode | RR legal TOP endpoints | TOP loss ticks | TOP recontact without reground | TOP restore after post-capture ground |',
        '| --- | ---: | --- | --- | --- |']
    for e in report['episodes']:
        text.append(f"| {e['episode_index']} | {e['current_contact_counts'].get('RR_legal_TOP_bearing_endpoints',0)} | {e['RR_TOP_loss_endpoint_ticks']} | {[x['tick'] for x in e['RR_legal_TOP_recontact_without_reground']]} | {[x['tick'] for x in e['RR_legal_TOP_restore_after_ground']]} |")
    c=report['combined_branch_1024']
    text += ['', f"Combined with sealed P12 block: {c['actual_policy_decisions']} new decisions / {c['ppo_updates']} PPO / "
        f"{c['optimizer_steps']} official Adam; {c['replay_exposures']} offline replay exposures, no extra PPO samples or separate AUX Adam.",
        f"Prefix excluded: {c['prefix_decisions']} decisions / {c['prefix_physics_ticks']} physics ticks.",
        f"Combined request phases: `{c['request_phase_counts']}`.",
        f"Combined physical windows: `{c['endpoint_physical_windows']}`.",
        f"Fresh student RR captures={c['fresh_student_RR_capture_events']}; RL fresh qualifications={c['fresh_student_RL_qualification_events']} "
        f"({c['fresh_student_RL_qualified_endpoints']} endpoints), plus {c['prefix_inherited_RL_qualified_endpoints']} inherited prefix-qualified endpoints. "
        'RL cross/place and full task success remain 0. P07 block itself has zero RL qualified endpoints despite 42 P12-labelled samples.',
        'New CP226304 physical evaluation is separate; these samples were collected before its update.']
    return '\n'.join(text)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=RUN)
    parser.add_argument('--report',default='892385_P07_CP225792_512_coverage')
    args=parser.parse_args()
    if Path(args.report).name != args.report:
        raise ValueError('report must be a basename')
    paths=[OUT/(args.report+ext) for ext in ('.json','.md')]
    if any(p.exists() for p in paths):
        raise FileExistsError('preserve prior report; choose a new basename')
    report=analyze(args.run.resolve(strict=True))
    for path,content in zip(paths,(json.dumps(report,ensure_ascii=False,indent=2)+'\n',markdown(report))):
        with path.open('x',encoding='utf-8') as stream:
            stream.write(content)
    print(json.dumps({'reports':[str(p) for p in paths],
        'optimized':report['summary']['optimized_decisions'],'episodes':len(report['episodes']),
        'checkpoint':report['checkpoint']},ensure_ascii=False))


if __name__ == '__main__':
    main()

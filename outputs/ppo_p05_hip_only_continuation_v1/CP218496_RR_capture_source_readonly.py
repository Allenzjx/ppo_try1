"""Read fixed sealed-video source/control/physical snapshots; no model or sim."""
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from wlr50_clean.ppo.semantic_observation import _rpy, _quaternion

SOURCE = ROOT/'runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/20260922T1818481605380Z_g6ac7b553d792_d3e1bceb3e8e462ca7eddd06eb20a21f/source'
POINTS = (6000,6400,6656,6657,6658,6661,6662,6663,6664,6680,6800,6878,6879,6880,7000,9744)


def read_sparse(name, key, selected, *, offset):
    result = {}
    with (SOURCE/name).open(encoding='utf-8') as stream:
        for index, line in enumerate(itertools.islice(stream, max(selected)-offset+1)):
            tick = index+offset
            if tick in selected:
                row = json.loads(line)
                assert row[key] == tick
                result[tick] = row
    return result


def main():
    native = read_sparse('native_tick_audit.jsonl','episode_physics_tick',set(POINTS)|set(range(6650,6882)),offset=1)
    physics = read_sparse('physical_observations.jsonl','physics_tick',set(POINTS),offset=0)
    decisions = {}
    wanted = {6000,6400,6656,6664,6680,6800,6880,7000,9744,9748}
    with (SOURCE/'video_policy_decisions.jsonl').open(encoding='utf-8') as stream:
        for index,line in enumerate(stream):
            if (index+1)*8 not in wanted and (index+1)*8 != 9752:
                continue
            row=json.loads(line)
            decisions[row['end_tick']]=row
    snapshots=[]
    for tick in POINTS:
        row=native[tick];a=row['native_audit'];h=a['policy_headroom_evidence'];p=physics[tick]
        assert a['verified'] and a['actual_mapping_matches_dispatch'] and a['setter_dispatch_targets_equal']
        q=p['actual_full12'];final=p['commanded_full12'];rr=p['wheels']['rear_right_ankle'];o=p['obstacle']
        item=dict(tick=tick,time_s=tick/120,source_phase=row['source_phase_id'],
            nominal=row['nominal_full12'],mapped_nominal=a['native_drive_target_full12'],
            geometry_baseline=h['geometry_corrected_native_full12'],
            requested_residual=h['requested_policy_residual_full12'],
            effective_residual=h['effective_policy_residual_full12'],final=final,actual=q,
            headroom_clipped_indices=h['clipped_servo_indices'],mask=a['phase_mask_full12'],
            RR_gap_mm=1000*(rr['bottom_w_m'][2]-o['top_z_m']),
            RR_front_mm=1000*(rr['center_w_m'][0]-o['front_x_m']),
            body_z_mm=1000*p['base']['position_w_m'][2],
            body_pitch_deg=math.degrees(_rpy(_quaternion(p['base']['orientation_wxyz']))[1]))
        if tick in decisions:
            task=decisions[tick]['step_info']['semantic_task'];d=task['nominal_provider_diagnostics']
            item.update(request_raw=decisions[tick]['raw_policy_action_full12'],
                task_stage=task['stage_id'],termination=task['termination_reason'],
                source_P09=next(x for x in d['source_partial_order']['layers'] if x['stage']=='P09'),
                rr_carry=d['rr_carry_continuation'],height_recovery=d['height_recovery'],
                RR_contact={k:task['physical_evaluator']['current_legs']['RR'].get(k) for k in
                    ('current_lift_valid','within_top_xy','air','ground_contact','top_surface_contact','contact_surface')})
        snapshots.append(item)
    changes=[]
    for tick in range(6651,6882):
        old,new=native[tick-1]['nominal_full12'],native[tick]['nominal_full12']
        if old!=new:
            changes.append(dict(dispatch_tick=tick,changed_indices=[i for i in range(12) if old[i]!=new[i]],before=old,after=new))
    geometry_changes=[]
    for tick in range(6651,6882):
        def offset(t):
            a=native[t]['native_audit'];g=a['policy_headroom_evidence']['geometry_corrected_native_full12']
            return [g[i]-a['native_drive_target_full12'][i] for i in (6,7)]
        if offset(tick)!=offset(tick-1):
            geometry_changes.append(dict(dispatch_tick=tick,previous_RR_geometry_offset=offset(tick-1),RR_geometry_offset=offset(tick)))
    last=decisions[9748]['step_info']['semantic_task']
    print(json.dumps(dict(source=str(SOURCE),scope='16 sparse snapshots plus 232 native ticks to locate first changes',
        nominal_changes=changes,RR_geometry_changes=geometry_changes,snapshots=snapshots,
        last_task=dict(stage=last['stage_id'],termination=last['termination_reason'],source=last['termination_source'],
            local_timeout=last['local_timeout'],history_events=last['physical_evaluator']['history']['event_ticks'],
            nominal_partial_order=last['nominal_provider_diagnostics']['source_partial_order']),
        order=['FLhip','FLknee','FRhip','FRknee','RLhip','RLknee','RRhip','RRknee','FLwheel','FRwheel','RLwheel','RRwheel'],
        no_model_forward=True,no_optimizer=True,no_simulation=True),separators=(',',':'),allow_nan=False))


if __name__=='__main__':
    main()

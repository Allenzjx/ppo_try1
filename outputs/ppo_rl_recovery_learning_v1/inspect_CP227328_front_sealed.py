"""CP227328 front comparison; REFUSE active runs before reading any JSONL.

Stdlib-only output-side wrapper. No models, simulation or video decoding.
Uses exact existing FR+6s analysis and bounded FL capture/P06 endpoints.
Prints JSON; publication by caller through apply_patch after review.
"""
import argparse
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T1842131087811Z_g892385cba8a7_c074d1ebec6947cc830fb3e0ec86dcec/source'


def analyze(source):
    seal = json.loads((source.parent/'run_manifest.json').read_text(encoding='utf-8-sig'))
    if not seal.get('completed_at_utc') or seal.get('lifecycle') == 'RUNNING':
        raise ValueError('Wait for parent-confirmed natural seal; no active JSONL read')
    meta = json.loads((source/'semantic_video_source_manifest.json').read_text(encoding='utf-8-sig'))
    if meta['from_phase'] != 'P01' or meta['policy_sampling_mode'] != 'deterministic_conditional_mean':
        raise ValueError('Expected natural P01 deterministic evaluation')
    spec = importlib.util.spec_from_file_location('previous_front_window',OUT/'inspect_CP226304_front_window.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    front = helper.read_window(meta['episode_physics_ticks'],source=source,checkpoint=227328,sealed=True)
    prior = json.loads((OUT/'CP226304_DET_FRplaced_front_space_6s.json').read_text(encoding='utf-8'))
    front['comparison_prior'].insert(0,dict(label='CP226304',FR_placed_tick=prior['FR_placed_tick'],endpoints=prior['endpoints']))
    points=[]; previous_mode=None; placed=None; first_p06=None; previous_cross=False; events={}
    lower=front['FR_placed_tick']+720
    for row in helper.complete_rows(source/'video_policy_decisions.jsonl'):
        tick=row['end_tick']
        if tick<lower: continue
        step=row['step_info'];ev=step['semantic_task']['physical_evaluator']; events=ev['history']['event_ticks']
        assist=step['actuator_target_effect_audit']['capture_assist_evidence']; mode=assist['state_after']['mode']
        new_placed=events['placed'].get('FL'); crossed=bool(events['front_edge_crossed'].get('FL'))
        p06=step['phase_id']=='P06' and first_p06 is None
        if p06: first_p06=tick
        if mode!=previous_mode or new_placed!=placed or crossed!=previous_cross or p06 or (placed and tick>=placed+120):
            leg=ev['current_legs']['FL'];a=step['actuator_target_effect_audit'];tracking=a['tracking_reference_evidence']
            points.append(dict(tick=tick,sim_s=tick/120,request_phase=step['phase_id'],end_phase=step['end_phase_id'],
                assist_mode=mode,assist_correction=assist['assist_correction_full12'],
                FL_final=step['actual_drive_target_full12'][:2],
                FL_actual_previous_physics_tick=[step['nominal_action_full12'][i]-tracking['channels'][i]['current_actual_canonical_error_deg'] for i in (0,1)],
                FL={k:leg.get(k) for k in ('top_contact','within_top_xy','bearing_verified','bearing_force_n','air','ground_contact','clearance_m','front_distance_m')},
                wheel_final=step['actual_drive_target_full12'][8:]))
        previous_mode,previous_cross,placed=mode,crossed,new_placed
        if placed and tick>=placed+120: break
    front['FL_capture_P06']=dict(events=events,FL_placed_tick=placed,P06_first_request_endpoint=first_p06,selected=points,
        last_read_tick=tick,normal_declared_FL_assist=meta.get('capture_assist_enabled_in_training_and_evaluation'),
        rear_task_assist=meta.get('rear_task_assist'),
        pure_policy_capture_claim=False,comparison='prior accepted/CP225792/CP226304 reports retain their separate source identities')
    front['source_seal']=dict(lifecycle=seal['lifecycle'],completed_at_utc=seal['completed_at_utc'],
        issued_policy_decisions=meta['issued_policy_decisions'],completed_environment_steps=meta['completed_environment_steps'],
        episode_physics_ticks=meta['episode_physics_ticks'],physical_task_success=meta['physical_task_success'],
        source_acceptance_error=meta.get('source_acceptance_error'),control_method=meta.get('control_method'),
        checkpoint_load_provenance=meta['checkpoint_load_provenance'])
    front['branch_coverage_reference']='CP227328_branch_physical_coverage_recheck.json'
    front['scopes']=[item for item in front['scopes'] if not item.startswith('An unobserved future event')]
    front['scopes'].append('Run is sealed; this front-window comparison does not substitute for the full-episode physical result.')
    front['legacy_mount_policy']='CP225280 mount heights remain N/A if same-run verified rigid mount geometry was absent; never reconstructed from joint angles'
    return front


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    args=parser.parse_args()
    print(json.dumps(analyze(args.source),ensure_ascii=False))

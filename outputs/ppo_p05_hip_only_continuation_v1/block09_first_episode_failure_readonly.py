"""Read exactly block09's sealed first episode; print evidence, never simulate/write."""
from pathlib import Path
import hashlib
import itertools
import json

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1428131775302Z_g336b7c56d2f0_23b26a4c40a24a178d190def03381413'
with (RUN / 'residual_and_projection_audit.jsonl').open('rb') as stream:
    raw_lines = list(itertools.islice(stream, 484))
rows = [json.loads(line) for line in raw_lines]
assert len(rows) == 484 and rows[-1]['terminal'] and not any(r['terminal'] for r in rows[:-1])
assert rows[-1]['global_policy_decision'] == 214884
with (RUN / 'completed_episodes.jsonl').open('rb') as stream:
    episode_bytes = stream.readline()
episode = json.loads(episode_bytes)
last = rows[-1]['applied_audit']
ev = last['semantic_task']['physical_evaluator']
assert ev['physics_tick'] == 5982 and ev['termination_reason'] == 'TASK_FAILURE_BODY_COLLISION'

def compact_sample(sample):
    low, high = sample['body_collider_minimum_w_m'], sample['body_collider_maximum_w_m']
    front, back, left, right, bottom, top = sample['obstacle_planes_world_m']
    overlaps = [min(high[i], (back,left,top)[i]) - max(low[i],(front,right,bottom)[i]) for i in range(3)]
    return dict(tick=round(sample['sim_time_s'] * 120), sim_time_s=sample['sim_time_s'], phase=sample['phase'],
                separation_lower_bound_m=sample['separation_lower_bound_m'], raw_geometry_cost=sample['raw_geometry_cost'],
                body_minimum_w_m=low, body_maximum_w_m=high, obstacle_planes_world_m=sample['obstacle_planes_world_m'],
                aabb_intersection_depth_m=max(0.,min(overlaps)))

samples = [s for r in rows for s in r['applied_audit']['reward']['task_space_quality_sample_audit'] if s.get('valid') and s.get('eligible')]
def first_sample(predicate):
    return next((compact_sample(s) for s in samples if predicate(s)), None)

mode_changes, phase_changes = [], []
old_mode = None
for i,r in enumerate(rows):
    a = r['applied_audit']; assist = a['actuator_target_effect_audit']['capture_assist_evidence']
    state = assist['state_after']; mode = state['mode_name']
    if mode != old_mode:
        mode_changes.append(dict(first_saved_endpoint_tick=a['physics_tick'], previous_saved_endpoint_tick=rows[i-1]['applied_audit']['physics_tick'] if i else None,
            source_observation_tick=assist['context']['source_observation_tick'], before_mode=assist['state_before']['mode_name'], mode=mode,
            stage=assist['context']['stage_id'], active=state['active'], release_fraction=state['release_fraction'],
            owners=state['owners'], gap_m=assist['context']['gap_m']))
        old_mode=mode
    for t in a['stage_transition_evidence']:
        phase_changes.append({k:t[k] for k in ['from_stage','to_stage','physics_tick','sim_time_s']})

def endpoint(r):
    a=r['applied_audit']; e=a['semantic_task']['physical_evaluator']; g=e['goal_features']
    keys=['clearance_m','front_distance_m','within_top_xy','ground_contact','top_surface_contact','support','bearing_force_n','contact_mode','ground_relative_lift_m','lift_established','current_lift_valid']
    return dict(global_policy_decision=r['global_policy_decision'],tick=a['physics_tick'],time_s=a['sim_time_s'],phase=a['phase_id'],terminal=r['terminal'],
        body_bounds=e['body_traversal_geometry'], body_linear_speed_m_s=g['body_linear_speed_m_s'],body_angular_speed_rad_s=g['body_angular_speed_rad_s'],
        legs={l:{k:e['current_legs'][l].get(k) for k in keys} for l in ['FL','FR','RL','RR']})

assist=last['actuator_target_effect_audit']['capture_assist_evidence']
result=dict(schema='wlr50_clean.block09_first_episode_failure_readonly.v1',
    scope='Only first 484 immutable learner records plus first completed-episode line; no replay, forward, fit, simulation or production edit.',
    run=str(RUN), bounded_prefix_sha256=hashlib.sha256(b''.join(raw_lines)).hexdigest(), completed_episode_line_sha256=hashlib.sha256(episode_bytes).hexdigest(),
    episode={k:episode.get(k) for k in ['episode_index','seed','policy_decisions','duration_s','termination_reason','task_success','full_task_success']},
    curriculum_start=last['curriculum_start'], terminal_global_policy_decision=214884, terminal_physics_tick=5982, terminal_physics_ticks_in_decision=last['physics_ticks'],
    adjudication={k:ev.get(k) for k in ['valid','run_validity','physical_evidence_status','termination_reason','termination_source','reason']},
    exact_body_pair_contract=dict(sensor_prim='/World/WLRRobot/base_link',other_prim='/World/Obstacle',body_role='BODY',
        criterion='verified exact active pair AND (at least 2 active physics samples OR pose-aware AABB intersection depth >= 0.001 m)',
        observed_pair_force_n=None, observed_contact_point_w_m=None, observed_subcollider_path=None, observed_first_contact_tick=None,
        observed_detector_reason=None, observed_pair_streak=None,
        interpretation='Failure is authoritative BODY_CONTACT, not the standalone reward geometry proxy. Raw detector fields and collider-child/contact-point identity are not persisted in these learner logs.',
        branch_inference='Terminal current-pose AABB depth is below 1 mm; given the unchanged detector, persistence is the compatible branch. This is a code-plus-recorded-geometry inference, not a saved raw pair/streak measurement.'),
    first_positive_soft_geometry_deficit_in_recorded_learner_samples=first_sample(lambda s:s['raw_geometry_cost']>0),
    first_positive_soft_geometry_deficit_in_P09=first_sample(lambda s:s['phase']=='P09' and s['raw_geometry_cost']>0),
    first_recorded_positive_AABB_intersection=next((compact_sample(s) for s in samples if compact_sample(s)['aabb_intersection_depth_m']>0),None),
    terminal_six_physics_geometry=[compact_sample(s) for s in last['reward']['task_space_quality_sample_audit']],
    preceding_endpoints=[endpoint(r) for r in rows if r['applied_audit']['physics_tick'] in [5944,5960,5976,5982]],
    lift_and_placement_history=ev['history'], phase_transitions=phase_changes,
    capture_assist_endpoint_mode_changes=mode_changes,
    terminal_capture_assist={k:assist.get(k) for k in ['owner_indices','assist_correction_full12','policy_request_unchanged','nominal_history_receives_assist']},
    terminal_capture_assist_state={k:assist['state_after'][k] for k in ['mode_name','active','owners','release_fraction']},
    terminal_drive=dict(canonical_order=last['actuator_target_effect_audit']['canonical_order'],
        units='first eight degrees; last four rad/s, canonical FL/FR/RL/RR order',
        actual_drive_target_full12=last['actual_drive_target_full12'],
        projected_residual_full12=last['projected_residual_full12'],
        measured_wheel_velocity_rad_s=ev['measured_wheel_velocity_rad_s'],
        effect_verified=last['actuator_target_effect_audit']['verified'],
        last_tick_native_effect=last['actuator_target_effect_audit_ticks'][-1]),
    terminal_reward={k:last['reward'][k] for k in ['total','potential_before','potential_after','potential_shaping','terminal_event','elapsed_physics_s','terminal_bootstrap_allowed']},
    limits=['No raw pair force or contact point/subcollider is retained here; cannot independently reconstruct the first microscopic contact.',
        'AABB is a conservative collider aggregate, not exact mesh penetration or exact contact location.',
        'FL/FR historical placement is not current support; RR terminal absolute wheel-bottom z and relative free-lift gain differ.',
        'Assist mode onset is bounded by neighboring saved endpoints unless a same-tick before/after transition is present.',
        'Temporal association does not prove that capture-assist release caused the later collision.'])
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

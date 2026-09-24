"""Only preserved successful N source; selected rear windows. No runtime imports/writes."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/video_eval/prior_B/20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source'
CURRENT_STARTUP=ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T0056424060031Z_g49eb23163a6e_c9d15196204e4b9aa94d71b1f7e309bd/source/height_diagnostics_startup.json'
def take(d,keys):return {k:d.get(k) for k in keys}
manifest=json.loads((SOURCE/'semantic_video_source_manifest.json').read_text())
old_start=json.loads((SOURCE/'height_diagnostics_startup.json').read_text())
new_start=json.loads(CURRENT_STARTUP.read_text())
assert manifest['physical_task_success'] and manifest['physical_episode']['physical_task_duration_s']==73.80833333333334
assert manifest['policy_sampling_mode']=='nominal_without_learned_residual'
events=[]
with (SOURCE/'stage_transition_evidence.jsonl').open() as f:
    for line in f:
        r=json.loads(line)
        if 5400<=r['physics_tick']<=6728:
            events.append(dict(tick=r['physics_tick'],time_s=r['sim_time_s'],from_stage=r['from_stage'],to_stage=r['to_stage'],
                event_ticks=r['physical_history']['event_ticks']))
selected={6000,6120,6152,6160,6168,6176,6200,6248,6256,6320,6664,6728}
decisions={}; first_xy=None; first_air_xy=None
with (SOURCE/'video_policy_decisions.jsonl').open() as f:
    for line in f:
        r=json.loads(line);a=r['step_info'];tick=a['physics_tick']
        if tick<5400:continue
        if tick>6736:break
        t=a['semantic_task'];ev=t['physical_evaluator'];rr=ev['current_legs']['RR']
        if first_xy is None and rr['within_top_xy'] and ev['history']['active_lift']['RR']:
            first_xy=tick;selected.add(tick)
        if first_air_xy is None and rr['within_top_xy'] and rr['air'] and ev['history']['active_lift']['RR']:
            first_air_xy=tick;selected.add(tick)
        if tick not in selected:continue
        assert all(x==0. for x in r['raw_policy_action_full12'])
        legfields=('air','top_contact','ground_contact','within_top_xy','within_lateral_span','support',
            'bearing_verified','bearing_force_n','load_fraction','front_distance_m','clearance_m','current_lift_valid','contact_mode')
        role=ev['transfer_roles']['RL']
        decisions[tick]=dict(tick=tick,time_s=a['sim_time_s'],request_phase=a['phase_id'],
            source_nominal_full12=a['nominal_action_full12'],final_target_full12=a['actual_drive_target_full12'],
            legs={k:take(ev['current_legs'][k],legfields) for k in ('FR','FL','RR','RL')},
            history=take(ev['history'],('active_lift','front_edge_crossed','placed')),
            RL_receiver_role=take(role,('target_swing_leg','diagonal_receiving_side','preferred_bridge_contacts',
                'observed_support_contacts','preferred_bridge_observed','preparation_ready','transfer_ready')),
            CoM_to_FR_logged_short_window=take(role['transfer_direction_context'],('reference_tick','window_s','fixed_direction_world',
                'com_world_displacement_m','receiver_world_displacement_m','com_toward_receiver_m','mass_weighted_com_position_w_m',
                'mass_weighted_com_velocity_w_m_s','two_contact_static_stability_proven')),
            nominal_source_layers=[take(layer,('stage','source_ticks','status','wait_reason','late_group_start_tick'))
                for layer in t['nominal_provider_diagnostics']['source_partial_order']['layers'] if layer['stage'] in ('P09','P12')])
observed=set()
with (SOURCE/'physical_observations.jsonl').open() as f:
    for line in f:
        r=json.loads(line);tick=r['physics_tick']
        if tick>max(selected):break
        if tick not in decisions:continue
        assert r['all_finite']
        d=decisions[tick]; observed.add(tick)
        d['actual_canonical_full12']=r['actual_full12']
        d['physical_commanded_full12']=r['commanded_full12']
        assert max(abs(a-b) for a,b in zip(r['commanded_full12'],d['final_target_full12']))<1e-9
        d['joints']={k:take(r['joints'][k],('position_deg','velocity_deg_s','command_deg','error_deg')) for k in
            ('front_right_hip','front_right_knee','front_left_knee','rear_right_hip','rear_right_knee')}
        d['wheel_canonical']={k:take(v,('velocity_rad_s','command_rad_s','center_w_m','bottom_w_m')) for k,v in r['wheels'].items()}
        d['physical_contacts']={k:dict(contact_class=v['contact_class'],
            ground_active=v['ground']['active'],ground_normal_force_n=v['ground']['normal_force_n'],
            obstacle_active=v['obstacle']['active'],obstacle_normal_force_n=v['obstacle']['normal_force_n'],
            ground_pair_verified=v['ground']['pair_verified'],obstacle_pair_verified=v['obstacle']['pair_verified'])
            for k,v in r['contacts'].items() if k.endswith('_wheel')}
        d['base_link_pose']=take(r['bodies']['base_link'],('position_w_m','orientation_wxyz'))
        d['mass_weighted_CoM']=r['center_of_mass']
assert observed==set(decisions)
with (SOURCE/'height_diagnostics.jsonl').open() as f:
    for line in f:
        r=json.loads(line);tick=r['physics_tick']
        if tick>max(selected):break
        if tick in decisions:
            decisions[tick]['RR_hip_mount_world_logged']=r['rr_hip_mount_w_m']
            decisions[tick]['body_collision_minimum_z_w_m']=r['body_collision_minimum_z_w_m']
result=dict(schema='readonly.successful_N_RR_RL_physical_reference.v1',source_dir=str(SOURCE),
    source_git=manifest['runtime_contract']['source_git_commit'],source_role='preserved_successful_N_ref_B_no_learned_residual',
    physical_task_success=True,physical_task_duration_s=73.80833333333334,optimizer_updates=0,
    source_event_window=events,first_RR_withinXY_qualified_endpoint=first_xy,
    first_RR_withinXY_AIR_endpoint_in_inspected_window=first_air_xy,
    first_air_xy_warning='This first measured AIR+XY endpoint is after historical RR placement. Do not invent a pre-contact high-hover counterpart.',
    rows=[decisions[t] for t in sorted(decisions)],
    mount_reconstruction=dict(status='N/A_for_four_mounts',reason='N_ref source binds environment_lock hash but records no robot USD content SHA; old startup has RR only. Current immutable startup proves four body0/localPos0 definitions but explicitly delegates asset hash binding. Same asset path or same RR point is not proof of all four old definitions.',
        N_ref_environment_lock_sha256=manifest['runtime_contract']['files']['configs/environment_lock.json'],
        N_ref_RR_definition=old_start['rr_hip_mount_definition'],
        current_immutable_startup=str(CURRENT_STARTUP),current_four_mounts_source_verified=new_start['same_rigid_body_hip_mount_definitions']['source_verified'],
        same_RR_localPos0=old_start['rr_hip_mount_definition']['value']['local_pos0_m']==new_start['same_rigid_body_hip_mount_definitions']['mounts']['RR']['local_pos0_m'],
        four_mount_world_heights=None,recorded_RR_mount_only_in_rows=True),
    scope='Only named successful_N source plus current immutable height startup. No growingDET logs, Torch, Isaac, production edits, Recording rescan, or new rollout. Absolute canonical angles; source nominal != mapped/final != actual. Contact class is historical source classification, not re-adjudication with current evaluator.')
print(json.dumps(result,separators=(',',':')))

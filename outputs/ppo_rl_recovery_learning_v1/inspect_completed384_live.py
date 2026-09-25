"""One bounded stdlib-only snapshot of three completed updates; no live tail read."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import inspect_course as coverage

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7'
FIRST, LAST = 225281, 225664
COUNT = LAST-FIRST+1
INDICES = (0, 1, 4, 5)
NAMES = ('FL_hip','FL_knee','FR_hip','FR_knee','RL_hip','RL_knee','RR_hip','RR_knee')


def bounded_prefix(path, count):
    digest = hashlib.sha256()
    result = []
    with path.open('rb') as stream:
        for _ in range(count):
            line = stream.readline(1_000_001)
            assert len(line) <= 1_000_000 and line.endswith(b'\n'), 'missing complete bounded row'
            digest.update(line)
            result.append(json.loads(line))
        nbytes = stream.tell()
    return result, dict(path=str(path), rows=count, byte_prefix=nbytes, sha256=digest.hexdigest())


def span(rows, fn):
    values = [fn(row) for row in rows]
    values = [x for x in values if type(x) in (float,int)]
    return [min(values),max(values)] if values else None


def main():
    targets = [OUT/'completed384_current_RR_recapture_readonly.json',
               OUT/'completed384_current_RR_recapture_readonly.md']
    assert not any(p.exists() for p in targets), 'preserve existing report'
    updates, update_source = bounded_prefix(RUN/'optimizer_updates.jsonl',3)
    assert [r['ppo_update'] for r in updates] == [1726,1727,1728]
    assert updates[-1]['global_policy_decisions'] == LAST
    assert all(r['optimizer_steps']==20 and r['actor_parameters_changed'] for r in updates)
    original, source = bounded_prefix(RUN/'residual_and_projection_audit.jsonl',COUNT)
    assert [r['global_policy_decision'] for r in original] == list(range(FIRST,LAST+1))
    started = coverage.small_json(RUN/'run_manifest.started.json')
    support = coverage.bound_support(started['runtime_contract'])
    compact = []; episode=1; last_tick=None
    for row in original:
        a=row['applied_audit']; t=a['semantic_task']; ev=t['physical_evaluator']
        tick=a['physics_tick']
        if last_tick is not None and tick<last_tick: episode+=1
        last_tick=tick
        n=a['actuator_target_effect_audit']; tracking=n['tracking_reference_evidence']
        # This is the independently logged pre-last-dispatch measurement, not
        # a 15 Hz endpoint encoder invented from the final target.
        q=[c['nominal_deg']-c['current_actual_canonical_error_deg'] for c in tracking['channels']]
        own=row['policy_request']['rear_owner_observed_features']
        assert len(own)==17 and n['phase_mask_full12']==[1]*12
        assert row['policy_request']['rear_task_assists_enabled'] is False
        flags, physical=coverage.physical_windows(t,support)
        dep=[x for x in t['nominal_provider_diagnostics']['source_partial_order']['layers'] if x['stage']=='P12']
        legs=ev['current_legs']
        prep=t['cooperative_preparation']
        capture=t['rr_capture_continuation']
        entry=dict(global_policy_decision=row['global_policy_decision'],episode=episode,
            tick=tick,sim_time_s=a['sim_time_s'],request_phase=a['phase_id'],end_phase=a['end_phase_id'],
            terminal=row['terminal'],termination_reason=a['termination_reason'],reward=row['reward'],
            potential=t['task_progress_potential'],physical_windows=sorted(flags),physical=physical,
            RR={k:legs['RR'].get(k) for k in ('ground_contact','air','top_contact','bearing_verified','bearing_force_n',
                'within_top_xy','current_lift_valid','clearance_m','front_distance_m','contact_mode')},
            RL={k:legs['RL'].get(k) for k in ('ground_contact','air','top_contact','bearing_verified','bearing_force_n',
                'within_top_xy','current_lift_valid','clearance_m','front_distance_m','contact_mode')},
            RR_placed_history=ev['history']['placed']['RR'],RL_placed_history=ev['history']['placed']['RL'],
            prep_relevant=prep['relevant'],prep_status=prep.get('measurement_status'),
            rr_capture_context={k:capture[k] for k in ('rr_lift_carry','rr_top_reachable','rr_top_contact',
                'rr_current_bearing','rl_transfer_ready','rr_capture_recovery_allowed')},
            rear_timing_at_policy_input=row['policy_request']['rear_policy_timing_observed_features'],
            rr_context_at_policy_input=row['policy_request']['rr_capture_transfer_observed_features'],
            cooperative_prep_allowed_at_policy_input=row['policy_request']['cooperative_prep_allowed'],
            p12_source=dep[-1] if dep else None,
            owner_at_policy_input=dict(H_final_deg=[v*180 for v in own[:4]],
                r0_request_deg=[v*180 for v in own[4:8]],active=own[8:12],winning_owner=own[12:16],edge=own[16]),
            raw_mean_full12=row['old_distribution_mean_full12'],raw_sample_full12=row['raw_policy_action_full12'],
            projected_request_full12=a['projected_residual_full12'],
            final_drive_full12=a['actual_drive_target_full12'],
            mapped_nominal_before_residual_full12=n['native_drive_target_full12'],
            actual_q_before_last_dispatch_deg=q,actual_q_clock='pre-last-dispatch tracking measurement; endpoint minus one physical tick',
            H_minus_actual_q_deg=[own[j]*180-q[i] if own[j+8] else None for j,i in enumerate(INDICES)],
            final_minus_actual_q_deg=[a['actual_drive_target_full12'][i]-q[i] for i in range(8)],
            owner_transition_receipt_saved_in_native_audit='rear_owner_recovery_evidence' in n,
            measured_wheels_canonical_rad_s=ev['measured_wheel_velocity_rad_s'])
        compact.append(entry)
    counts=Counter(w for r in compact for w in r['physical_windows'])
    episodes=[]
    for ep in range(1,episode+1):
        sub=[r for r in compact if r['episode']==ep]
        episodes.append(dict(episode=ep,decisions=len(sub),global_range=[sub[0]['global_policy_decision'],sub[-1]['global_policy_decision']],
            ticks=[sub[0]['tick'],sub[-1]['tick']],phases=dict(Counter(r['request_phase'] for r in sub)),
            terminal=sub[-1]['terminal'],reason=sub[-1]['termination_reason'],
            windows=dict(Counter(w for r in sub for w in r['physical_windows']))))
    third=[r for r in compact if r['episode']==3]
    ground=[r for r in third if r['RR_placed_history'] and r['RR']['ground_contact'] and not r['RL']['current_lift_valid']]
    recapture=[r for r in third if r['RR_placed_history'] and not r['rr_capture_context']['rr_current_bearing'] and not r['RL']['current_lift_valid']]
    event_rows=[]; prev=None
    for r in third:
        signature=(r['request_phase'],r['end_phase'],r['RR_placed_history'],r['RR']['contact_mode'],
            r['RR']['current_lift_valid'],r['rr_capture_context']['rr_current_bearing'],r['prep_relevant'],
            tuple(r['owner_at_policy_input']['active']),r['RL']['current_lift_valid'])
        if signature!=prev or r is third[-1]: event_rows.append(r)
        prev=signature
    projections=[r for r in compact if coverage.FR_PROJECTION_WINDOW in r['physical_windows']]
    result=dict(schema='readonly.completed_384_recapture.v1',run=str(RUN),completed_updates=updates,
        source_prefix=source,update_prefix=update_source,credit=dict(decisions=384,PPO_updates=3,Adam_steps=60,
        teacher_prefix_decisions=0,uncompleted_tail=0),limitations=[
            'Only first384 optimized decisions are consumed. No live unfinished tail, tensors, Torch, Isaac imports, or production mutation.',
            'Episode3 remains nonterminal in this prefix; no claim about its later outcome.',
            'Owner anchors/flags are policy-input time; q is endpoint minus one tick. Anchors are held, but this is not a simultaneous input-q sample.',
            'Native independent audit omits owner transition receipt: absence is N/A, not inactive; public17 is actual logged input evidence.',
            'Mapped nominal native_drive_target_full12 is NOT final command; final uses actual_drive_target_full12.',
            coverage.FR_PROJECTION_SEMANTICS],
        endpoint_windows={k:counts[k] for k in coverage.WINDOWS},episodes=episodes,
        positive_projection_diagnostic=dict(count=len(projections),
            world_CoM_dx_m=span(projections,lambda r:r['physical']['FR_window_CoM_world_displacement_xyz_m'][0]),
            world_CoM_dy_m=span(projections,lambda r:r['physical']['FR_window_CoM_world_displacement_xyz_m'][1]),
            world_CoM_dz_m=span(projections,lambda r:r['physical']['FR_window_CoM_world_displacement_xyz_m'][2]),
            world_body_dx_m=span(projections,lambda r:r['physical']['FR_window_body_world_displacement_xyz_m'][0]),
            world_body_dy_m=span(projections,lambda r:r['physical']['FR_window_body_world_displacement_xyz_m'][1]),
            RL_bearing_force_n=span(projections,lambda r:r['RL']['bearing_force_n']),
            RL_current_qualified_count=sum(r['RL']['current_lift_valid'] for r in projections),
            examples=projections[:2]),
        third_episode_recapture=dict(no_RR_bearing_RL_unqualified_count=len(recapture),
            RR_ground_RL_unqualified_count=len(ground),first_ground=ground[0] if ground else None,
            prep_relevant_count=sum(r['prep_relevant'] for r in ground),
            input_rr_carry_capture_count=sum(r['rear_timing_at_policy_input'][0]==1 for r in ground),
            input_all4_owner_active_count=sum(r['owner_at_policy_input']['active']==[1]*4 for r in ground),
            p12_dependency_wait_count=sum(r['p12_source'] is not None and r['p12_source']['status']=='holding_RL_joint_lane' for r in ground),
            reward_range=span(ground,lambda r:r['reward']),reward_sum=sum(r['reward'] for r in ground),
            potential_range=span(ground,lambda r:r['potential']),
            H_minus_q_ranges_deg={NAMES[i]:span(ground,lambda r,j=j:r['H_minus_actual_q_deg'][j]) for j,i in enumerate(INDICES)},
            final_minus_q_ranges_deg={NAMES[i]:span(ground,lambda r,i=i:r['final_minus_actual_q_deg'][i]) for i in range(8)},
            final_drive_ranges={NAMES[i]:span(ground,lambda r,i=i:r['final_drive_full12'][i]) for i in range(8)},
            raw_mean_ranges={NAMES[i]:span(ground,lambda r,i=i:r['raw_mean_full12'][i]) for i in range(8)},
            event_rows=event_rows,last=third[-1]),
        code_facts=[
            'semantic_rear_policy_timing.public_timing: current loss after RR placed sets rr_carry_capture even at P12, not hidden phase-only work.',
            'semantic_cooperative_preparation.measure_preparation: RR ground or outside topXY sets relevant=False, removing FL-range/RL-space proxy contributions.',
            'semantic_supervisor.physical_potential: history-placed RR bypasses workspace/lift/carry and gets .8+.2*current retention.',
            'semantic_supervisor._current_capture_retention: RR illegal/ground while RL not actual swing returns0; recapture direction outside legal region is not shaped by that retention share.',
            'RL without current RR support and no qualified AIR receives only .1*workspace; no false unload/lift progress from history placed RR.',
            'All12 actual mask stays1. Owner17 suspension changes dependent FL/RL composition, not raw Gaussian samples. Wheels, FR, RR remain ordinary compose/limit lanes.',
            'Observation includes actual joints, wheel geometry/velocities/forces and current explicit rr_carry_capture, dependency waits and owner17; no observation-only explanation that P12 must be RL transfer.'])
    with targets[0].open('x',encoding='utf-8') as f: json.dump(result,f,indent=2,allow_nan=False)
    lines=['# Completed384: current RR recapture read-only evidence','',
        'Only 225281–225664 / updates1726–1728 /60 Adam steps. Prefix and unfinished tail receive no credit.',
        '', '| Physical endpoint window | Count |','| --- | ---: |']
    lines += [f'| {k} | {counts[k]} |' for k in coverage.WINDOWS]
    lines += ['',coverage.FR_PROJECTION_SEMANTICS,'',
        f'Third episode RR-ground / unqualified-RL samples: {len(ground)}. Cooperative relevant: {sum(r["prep_relevant"] for r in ground)}.',
        'RR recapture remains an explicit actor task and all12 are open. Strong dependent source lanes pause; changing policy requests can still move the suspended FL/RL targets. This is not a source-mask block on RR/FR/wheels.',
        'However, after prior RR placement, illegal/ground RR has zero current-retention progress and skips workspace/lift/carry shaping. FL-range/RL-space shaping also disappears when cooperative relevance becomes false. Physical potential is not phase-only, but this recovery region is locally sparse in RR recovery guidance.',
        'Observed owner H can remain different from actual q: freezing an already committed target does not stop residual tracking/inertia immediately. Numeric H/q/final/request separation and exact clocks are in JSON.',
        'No production changes or intervention. No attribution of success to projection or learning; episode3 is still unfinished in this bounded prefix.',
        '', '## Source facts',''] + ['- '+x for x in result['code_facts']]
    with targets[1].open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('endpoint_windows','episodes','positive_projection_diagnostic')},indent=2)[:7000])
    print(json.dumps({'json':str(targets[0]),'markdown':str(targets[1]),'ground_count':len(ground)}))


if __name__=='__main__': main()

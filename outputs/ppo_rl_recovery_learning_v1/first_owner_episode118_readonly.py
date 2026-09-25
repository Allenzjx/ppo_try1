"""One fixed, completed 118-decision episode; stdlib only, no live-tail reads."""
from collections import Counter
from itertools import islice
import hashlib
import json
import math
from pathlib import Path
import runpy

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7'
DEST = OUT / 'first_owner_episode118_readonly'
util = runpy.run_path(str(OUT / 'inspect_course.py'))
with (RUN / 'residual_and_projection_audit.jsonl').open(encoding='utf-8-sig') as stream:
    lines = list(islice(stream, 118))
assert len(lines) == 118
rows = [json.loads(line) for line in lines]
assert [r['global_policy_decision'] for r in rows] == list(range(225281, 225399))
assert [r['global_policy_decision'] for r in rows if r['terminal']] == [225398]
manifest = util['small_json'](RUN / 'run_manifest.started.json')
support = util['bound_support'](manifest['runtime_contract'])
floor = support['force_noise_floor_n']
joint_names = ('front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee',
               'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee')
owner_indices = (0, 1, 4, 5)


def actual_q(task, headroom):
    margins = {}
    for role in task['transfer_roles'].values():
        margins.update(role['receiver_workspace_state']['joint_range_margin_deg'])
    result = []
    for i, name in enumerate(joint_names):
        low, high = headroom['servo_hard_limits_deg'][i]
        q = low + margins[name]['negative_deg']
        assert abs(q - (high - margins[name]['positive_deg'])) < 1e-8
        result.append(q)
    return result


def compact_leg(leg):
    fields = ('air', 'ground_contact', 'contact_surface', 'top_contact', 'top_surface_contact',
              'obstacle_pair_active', 'support', 'bearing_verified', 'bearing_force_n',
              'clearance_m', 'front_distance_m', 'within_top_xy', 'within_lateral_span',
              'current_lift_valid', 'motion_continuation_allowed', 'active_attempt',
              'initial_clearance_seen', 'current_lift_qualified_tick', 'consecutive_top_samples')
    return {k: leg.get(k) for k in fields}


compact, windows, phases, owner_counts, checks = [], Counter(), Counter(), Counter(), Counter()
max_logp_error = 0.0
previous = None
for row in rows:
    a = row['applied_audit']; task = a['semantic_task']; ev = task['physical_evaluator']
    native = a['actuator_target_effect_audit']; h = native['policy_headroom_evidence']
    p = row['policy_request']; features = p['rear_owner_observed_features']
    assert len(features) == 17 and native['verified'] and native['phase_mask_full12'] == [1] * 12
    assert p['selected_raw_full12'] == row['raw_policy_action_full12']
    assert p['conditional_mean_full12'] == row['old_distribution_mean_full12']
    assert p['effective_sigma_full12'] == row['old_distribution_std_full12']
    assert p['selected_raw_log_probability'] == row['old_log_probability']
    assert p['sampling_draws'] == 1 and p['rear_task_assists_enabled'] is False
    assert a['prefix_teacher_data_in_ppo_storage'] is False
    assert a['prefix_checkpoint_policy_data_in_ppo_storage'] is False
    assert a['no_in_episode_state_writes_verified'] is True
    max_logp_error = max(max_logp_error, abs(util['gaussian_logp'](
        row['raw_policy_action_full12'], row['old_distribution_mean_full12'],
        row['old_distribution_std_full12']) - row['old_log_probability']))
    for tick in a['actuator_target_effect_audit_ticks']:
        assert tick['verified'] is True
        checks['verified_physics_ticks'] += 1
    checks['native_owner_receipt_present'] += 'rear_owner_recovery_evidence' in native
    checks['FL_capture_assist_owned_endpoint'] += any(native['capture_assist_owned_channels_full12'])
    flags, evidence = util['physical_windows'](task, support)
    windows.update(flags); phases[a['phase_id']] += 1
    q = actual_q(task, h)
    start = a['physics_tick'] - a['physics_ticks']
    dt = None if previous is None else (a['physics_tick'] - previous['endpoint_tick']) / 120
    if previous is not None:
        assert previous['endpoint_tick'] == start
    provider = task['nominal_provider_diagnostics']
    layers = provider['source_partial_order']['layers']
    selected_layers = {x['stage']: {k: x.get(k) for k in (
        'source_ticks', 'status', 'wait_reason', 'late_group_start_tick', 'late_group_source_tick',
        'rr_top_contact', 'rr_current_bearing', 'support_transfer_permitted', 'rl_current_swing',
        'rl_edge_recovery_permitted')} for x in layers if x['stage'] in ('P09', 'P12')}
    item = dict(global_policy_decision=row['global_policy_decision'], input_tick=start,
        endpoint_tick=a['physics_tick'], time_s=a['sim_time_s'], input_phase=a['phase_id'], end_phase=a['end_phase_id'],
        terminal=bool(row['terminal']), source_phase_last_dispatch=native['source_phase_id'],
        mapped_N_full12=native['native_drive_target_full12'],
        raw_gaussian_full12=row['raw_policy_action_full12'], mean_full12=row['old_distribution_mean_full12'],
        sigma_full12=row['old_distribution_std_full12'],
        requested_residual_full12=h['requested_policy_residual_full12'],
        headroom_effective_residual_full12=h['effective_policy_residual_full12'],
        candidate_before_owner_and_final_slew_full12=h['candidate_native_target_before_final_slew_full12'],
        final_target_full12=a['actual_drive_target_full12'], actual_joint_deg=q,
        actual_joint_source='recorded current joint_range_margin_deg plus recorded hard lower bound; upper-bound identity cross-checked',
        hard_joint_limits_deg=h['servo_hard_limits_deg'],
        endpoint_target_minus_actual_deg=[a['actual_drive_target_full12'][i]-q[i] for i in range(8)],
        finite_difference_q_velocity_deg_s=(None if previous is None else [(q[i]-previous['actual_joint_deg'][i])/dt for i in range(8)]),
        velocity_semantics='endpoint finite difference, NOT measured joint velocity or actuator force',
        clipped_servo_indices=h['clipped_servo_indices'],
        owner_input=dict(anchor_final_deg=[x*180 for x in features[:4]],
            anchor_request_deg=[x*180 for x in features[4:8]], active=[bool(x) for x in features[8:12]],
            winning_owner=[bool(x) for x in features[12:16]], edge_recovery=bool(features[16])),
        input_anchor_minus_actual_deg=(None if previous is None else [features[j]*180-previous['actual_joint_deg'][i]
            if features[8+j] else None for j,i in enumerate(owner_indices)]),
        legs={leg:compact_leg(ev['current_legs'][leg]) for leg in ('FL','FR','RL','RR')},
        RR_actual_TOP_bearing=util['bearing'](ev['current_legs']['RR'],floor,top=True),
        history_RR_placed=ev['history']['placed']['RR'], history_RL_placed=ev['history']['placed']['RL'],
        layers=selected_layers, physical_windows=sorted(flags), reward=row['reward'], old_value=row['old_value'])
    for j,i in enumerate(owner_indices):
        owner_counts[f'active_channel_{i}'] += bool(features[8+j])
        owner_counts[f'winning_channel_{i}'] += bool(features[12+j])
    compact.append(item); previous = item

first_bearing = next((x for x in compact if x['RR_actual_TOP_bearing']), None)
dropouts = []
for before, after in zip(compact,compact[1:]):
    if before['RR_actual_TOP_bearing'] and not after['RR_actual_TOP_bearing']:
        reacquired = next((x for x in compact if x['endpoint_tick'] > after['endpoint_tick'] and x['RR_actual_TOP_bearing']), None)
        dropouts.append(dict(previous_tick=before['endpoint_tick'], endpoint_tick=after['endpoint_tick'],
            terminal=after['terminal'], current_RR=after['legs']['RR'], owner_input=after['owner_input'],
            reacquired_sample_tick=reacquired['endpoint_tick'] if reacquired else None,
            current_final_FLhip_FLknee_RLhip_RLknee=[after['final_target_full12'][i] for i in owner_indices],
            current_actual_FLhip_FLknee_RLhip_RLknee=[after['actual_joint_deg'][i] for i in owner_indices],
            final_minus_actual_FLhip_FLknee_RLhip_RLknee=[after['endpoint_target_minus_actual_deg'][i] for i in owner_indices],
            actual_anchor=None, explanation='No active/winning owner in sampled input; these are actual endpoint FINAL minus q, NOT a hypothetical H activation.'))
late = next((x for x in compact if x['layers'].get('P09',{}).get('late_group_start_tick') is not None),None)


def saturation(index):
    selected = [x for x in compact if index in x['clipped_servo_indices']]
    lower = [x for x in selected if x['requested_residual_full12'][index] < x['headroom_effective_residual_full12'][index]]
    exact = [x for x in lower if abs(x['final_target_full12'][index]+58)<1e-8]
    def bounds(seq, key):
        values = [x[key][index] for x in seq]
        return [min(values),max(values)] if values else None
    return dict(total_rows=len(compact), headroom_clip_rows=len(selected), lower_clip_rows=len(lower),
        lower_clip_fraction=len(lower)/len(compact), final_minus58_rows=len(exact),
        first_lower_clip_decision=lower[0]['global_policy_decision'] if lower else None,
        lower_clipped_raw_range=bounds(lower,'raw_gaussian_full12'),
        lower_clipped_mean_range=bounds(lower,'mean_full12'), lower_clipped_sigma_range=bounds(lower,'sigma_full12'),
        lower_clipped_request_deg_range=bounds(lower,'requested_residual_full12'),
        lower_clipped_final_deg_range=bounds(lower,'final_target_full12'),
        raw_samples_mapped_to_minus58_range=bounds(exact,'raw_gaussian_full12'),
        minus58_unique_raw_samples=len({x['raw_gaussian_full12'][index] for x in exact}),
        Gaussian_exit_raw_threshold=None, Gaussian_exit_probability=None,
        reason_no_probability='The physical-tick headroom baseline follows the single actual mapper and current history; the 8-tick filtered/rate-limited request and changed physical response are not an invertible single frozen raw-to-final map in these decision-end receipts. No unverified threshold/probability is fabricated.')

terminal = rows[-1]['applied_audit']; ev = terminal['semantic_task']['physical_evaluator']
selected_indices = sorted(set([0, len(compact)-1] + list(range(max(0,len(compact)-12),len(compact))) +
    [i for i,x in enumerate(compact) if (x is first_bearing or x is late or (i and x['end_phase']!=compact[i-1]['end_phase']))]))
report = dict(schema='bounded_first_owner_episode_readonly.v1', run=str(RUN),
    source_file='residual_and_projection_audit.jsonl', fixed_rows=118,
    fixed_118_lines_sha256=hashlib.sha256(''.join(lines).encode('utf-8')).hexdigest(),
    source_head=manifest['runtime_contract']['source_git_commit'], scope='collected genuine learner episode only; no optimizer-completion credit asserted by this audit',
    global_decision_range=[225281,225398], request_phase_counts=dict(phases), support_binding=support,
    prefix=rows[0]['applied_audit']['curriculum_start'],
    prefix_physics_ticks=rows[0]['applied_audit']['curriculum_start']['physics_tick'],
    learner_physics_ticks=sum(x['applied_audit']['physics_ticks'] for x in rows),
    outcome=dict(terminal=True, reason=terminal['termination_reason'], task_scope=terminal['task_result_scope'],
        task_success=terminal['task_success'], full_task_success=terminal['full_task_success'],
        episode_end_tick=terminal['physics_tick'], physical_episode_duration_s=terminal['episode_elapsed_s'],
        evaluator_reason=ev['reason'], termination_source=ev['termination_source']),
    physical_window_endpoint_counts=dict(windows), native_checks=dict(checks), owner_input_counts=dict(owner_counts),
    raw_mean_sigma_logp_matches=118, maximum_float_Gaussian_logp_recompute_error=max_logp_error,
    event_ticks=ev['history']['event_ticks'],
    first_sampled_RR_TOP_bearing=first_bearing, sampled_TOP_bearing_loss_transitions=dropouts,
    first_sample_with_late_source=late,
    owner_H_actual_finding='No sampled pre-action active owner or nonzero anchor in all 118 rows. No concrete active H-versus-actual dropout entrance can be measured in this episode. Per-physics owner receipt is validated internally but omitted from the returned JSON audit, so subdecision transient activations cannot be excluded.',
    owner_design_scope='With constant request after an active entrance the code keeps H=previous FINAL, not actual q. Thus it stops additional stale-N displacement but is not guaranteed to remove an existing servo position error; this is a conditional code property, not demonstrated active behavior in this episode.',
    saturation={joint_names[i]:saturation(i) for i in (1,7)},
    selected_control_rows=[compact[i] for i in selected_indices],
    limits=['15 Hz endpoint contact/owner sampling cannot rule out subdecision changes.',
        'No measured joint velocity/force is present in these rows; endpoint finite difference does not isolate inertia from load, servo response, or contact coupling.',
        'Native mapped N is taken from the one actual mapper receipt, never final minus raw or an independently replayed mapper.',
        'This is P07 suffix training with successful_nominal prefix; front placements in the prefix are not student policy success.',
        'A safety terminal is a true failed task episode, not a budget boundary or complete traversal.'])
json_path=DEST.with_suffix('.json'); md_path=DEST.with_suffix('.md')
if md_path.exists() and not json_path.exists():
    raise RuntimeError('refuse to overwrite an unbound prior note')
if json_path.exists():
    assert json.loads(json_path.read_text(encoding='utf-8'))['fixed_118_lines_sha256'] == report['fixed_118_lines_sha256']
tail=compact[-1]; lines_md=[
    '# First completed owner-recovery episode: bounded read-only audit','',
    f'- Exact first 118 rows, global 225281–225398, {report["learner_physics_ticks"]} learner physics ticks; not a completed-update claim.',
    f'- P07 successful-nominal prefix ends tick 5160 / 43 s; request phases: {dict(phases)}. Not natural-P01 policy success.',
    '- True safety failure: FL knee HARD_JOINT_LIMIT, tick 6103 / 50.858333 s, P12. Full task success=false.',
    f'- RR qualified/crossed/placed ticks: {ev["history"]["event_ticks"]["active_lift"]["RR"]} / {ev["history"]["event_ticks"]["front_edge_crossed"]["RR"]} / {ev["history"]["event_ticks"]["placed"]["RR"]}. RL qualified/crossed/placed remain absent; initial clearance is not qualified swing.',
    f'- First sampled real RR TOP+bearing tick {first_bearing["endpoint_tick"] if first_bearing else None}; P09 late source starts at {late["layers"]["P09"]["late_group_start_tick"] if late else None}. Sampled post-TOP loss transitions: {len(dropouts)}.',
    f'- Both sampled losses ({[x["endpoint_tick"] for x in dropouts]}) precede late-owner acquisition; corresponding sampled winning/active owner bits are false. Reacquired TOP samples: {[x["reacquired_sample_tick"] for x in dropouts]}. They do not test suspension of an already-issued late target.',
    f'- Terminal RR TOP {tail["legs"]["RR"]["bearing_force_n"]:.6f} N; FL TOP {tail["legs"]["FL"]["bearing_force_n"]:.6f} N. Terminal safety invalidates controller permission; that is not evidence RR physically lost bearing.',
    '', '## Last FL knee control path (degrees)', '',
    '| End tick | Actual mapped N | Gaussian raw | Requested residual | Effective residual | FINAL | Actual q | FINAL−q | Δq/Δt °/s |',
    '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in compact[-10:]:
    lines_md.append(f'| {x["endpoint_tick"]} | {x["mapped_N_full12"][1]:.3f} | {x["raw_gaussian_full12"][1]:.4f} | {x["requested_residual_full12"][1]:.3f} | {x["headroom_effective_residual_full12"][1]:.3f} | {x["final_target_full12"][1]:.3f} | {x["actual_joint_deg"][1]:.3f} | {x["endpoint_target_minus_actual_deg"][1]:.3f} | {x["finite_difference_q_velocity_deg_s"][1]:.3f} |')
lines_md += ['', 'Actual q is recovered from logged current joint hard-limit margin, cross-checked against both bounds. Δq/Δt is only endpoint finite difference, not measured velocity/force.',
    '', '## Saturation and owner limits','',
    '| Joint | Lower headroom clipped | FINAL −58° | Clipped raw range | Mean range | Effective σ range |',
    '|---|---:|---:|---|---|---|']
for name,s in report['saturation'].items():
    lines_md.append(f'| {name} | {s["lower_clip_rows"]}/118 | {s["final_minus58_rows"]}/118 | {s["lower_clipped_raw_range"]} | {s["lower_clipped_mean_range"]} | {s["lower_clipped_sigma_range"]} |')
lines_md += ['', report['owner_H_actual_finding'], '', report['owner_design_scope'], '',
    'The FL target is protected to −58°, but actual reaches −60.011567° (hard lower bound −60°). This proves target projection alone did not prevent this physical crossing; it does not isolate dynamics, contact coupling or inertia. Neither a fixed measured-q exit nor more noise is justified by this episode alone.', '',
    'The actual N contribution changes −12.15→−41.4° after late source starts while RR has current TOP bearing; the residual remains about −33 to −34°. Thus this is combined source/residual demand, not evidence of an absent residual channel. From tick 6088 onward FINAL−q is positive while q continues negative: the final target is already restorative relative to measured position; past acceleration and physical load cannot be separated here.', '',
    'Exact Gaussian probability of exiting the lower headroom region is not recoverable as a single raw threshold from these filtered, rate-limited, mapper-dependent endpoint receipts. Observed clipping rates and original mean/σ are reported without inventing an inverse.', '',
    f'Raw sample/μ/σ/logp bindings and all-1 mask/native verification pass for 118 rows and {checks["verified_physics_ticks"]} physics ticks. No rear task teacher or prefix samples credited. Per-tick owner JSON evidence is missing despite internal verification; active public input count is zero, not proof of zero subdecision activity.', '',
    'Physical-window counts (overlapping endpoint categories): '+str(dict(windows))+'.', '',
    'JSON contains selected exact control rows, frozen support threshold binding, event ticks, source timing and evidence limits. No model, simulator, production file or reward changed.']
json_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
md_path.write_text('\n'.join(lines_md)+'\n',encoding='utf-8')
print(json.dumps({k:report[k] for k in ('request_phase_counts','outcome','physical_window_endpoint_counts','owner_input_counts','saturation')},indent=2))
print(str(md_path))

"""Bounded stdlib analysis of two sealed student captures; no runtime imports."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T1527536734451Z_g892385cba8a7_f00e6fb48d074b26a2a98898505632e6'
REPORT = json.loads((OUT / '892385_P07_CP225792_512_coverage.json').read_text())

def read_rows():
    with (RUN / 'residual_and_projection_audit.jsonl').open('rb') as stream:
        stream.seek(0, 2)
        extent = stream.tell()
        stream.seek(0)
        while stream.tell() < extent:
            line = stream.readline()
            if not line.endswith(b'\n'):
                break
            row = json.loads(line)
            if any(a <= row['global_policy_decision'] <= b for a, b in
                   (REPORT['episodes'][e]['global_range'] for e in (1, 4))):
                yield row

rows = list(read_rows())
byglobal = {r['global_policy_decision']: r for r in rows}

def decode(row):
    a, p = row['applied_audit'], row['policy_request']
    q = a['actuator_target_effect_audit']
    t, h, track = a['semantic_task'], q['policy_headroom_evidence'], q['tracking_reference_evidence']
    physical = t['physical_evaluator']
    legs = physical.get('current_legs', {})
    tick = a['physics_tick']
    nxt = byglobal.get(row['global_policy_decision'] + 1)
    endpoint_owner = None
    if not row['terminal'] and nxt and nxt['applied_audit']['physics_tick'] - nxt['applied_audit']['physics_ticks'] == tick:
        endpoint_owner = nxt['policy_request']['rear_owner_observed_features']
    actual = [c['nominal_deg'] - c['current_actual_canonical_error_deg'] for c in track['channels']]
    # Same-tick evidence's error is (nominal physical - actual physical)/axis sign.
    # Thus nominal - error returns canonical sensor degrees without assuming mirror signs.
    current_legs = {leg: {k: v.get(k) for k in ('contact_mode', 'top_contact', 'within_top_xy',
        'ground_contact', 'air', 'bearing_force_n', 'clearance_m', 'front_distance_m',
        'consecutive_air_samples', 'consecutive_top_samples', 'free_air_reference_tick',
        'wheel_bottom_vz_m_s')} for leg, v in legs.items()}
    layers = t['nominal_provider_diagnostics']['source_partial_order']['layers']
    candidate = h['candidate_native_target_before_final_slew_full12']
    final = a['actual_drive_target_full12']
    return dict(global_policy_decision=row['global_policy_decision'], endpoint_tick=tick,
        sensor_servo_episode_tick=tick-1, dispatch_native_tick=q['physics_tick'],
        phase=a['phase_id']+'->'+a['end_phase_id'], terminal=row['terminal'],
        physical_reason=physical.get('reason'), legs=current_legs,
        source_nominal_full12=a['nominal_action_full12'],
        mapped_nominal_full12=q['native_drive_target_full12'],
        raw_policy_full12=row['raw_policy_action_full12'], conditional_mean_full12=row['old_distribution_mean_full12'],
        sigma_full12=row['old_distribution_std_full12'],
        filtered_request_full12=h['requested_policy_residual_full12'],
        effective_headroom_full12=h['effective_policy_residual_full12'],
        headroom_clipped_indices=h['clipped_servo_indices'],
        headroom_candidate_full12=candidate, final_full12=final,
        final_minus_headroom_candidate_full12=[f-c for f,c in zip(final,candidate)],
        actual_servo_canonical_deg_pre_step=actual,
        actual_wheel_canonical_rad_s=physical.get('measured_wheel_velocity_rad_s'),
        actual_wheel_native_rad_s=None,
        wheel_readback_semantics='physical evaluator speeds come from canonical actual_full12; do not apply axis signs a second time; native sensor speeds not separately stored here',
        owner_input17=p['rear_owner_observed_features'], endpoint_owner17_via_next_input=endpoint_owner,
        endpoint_owner_semantics='next input at exact endpoint: anchors/active from last committed dispatch; winning flags current context; no native owner receipt',
        source_layers=[{k:v.get(k) for k in ('stage','source_ticks','actual_start_tick','wait_reason','status')} for v in layers],
        capture_hold=t['nominal_provider_diagnostics']['capture_owner_hold'],
        rr_currently_usable=t.get('rr_placed_currently_usable'),
        cooperative=t.get('cooperative_preparation'),
        transition_evidence=a.get('stage_transition_evidence'),
        reward=row['reward'], source_clock_tick_offset=q['physics_tick']-(tick-1),
        mapping_verified=q['actual_mapping_matches_dispatch'])

episodes = []
for e in (1, 4):
    lo, hi = REPORT['episodes'][e]['global_range']
    rr = [decode(r) for r in rows if lo <= r['global_policy_decision'] <= hi]
    placement = REPORT['episodes'][e]['events']['RR']['history_event_ticks']['placed']
    losses=[]
    previously=False
    for r in rr:
        leg=r['legs'].get('RR',{})
        usable=bool(leg.get('top_contact') and leg.get('within_top_xy') and (leg.get('bearing_force_n') or 0)>.2)
        if r['endpoint_tick']>=placement and not usable and previously:
            losses.append(r['endpoint_tick'])
        previously=usable
    saturated=[r for r in rr if abs(r['final_full12'][3]+58)<1e-8]
    first_loss=losses[0]
    selected=[r for r in rr if placement-25<=r['endpoint_tick']<=first_loss+32
              or r['endpoint_tick'] in losses[1:]
              or (saturated and r['endpoint_tick'] in (saturated[0]['endpoint_tick'],saturated[-1]['endpoint_tick']))
              or r['terminal']]
    episodes.append(dict(episode_index=e,placement_tick=placement,first_observed_usable_loss=first_loss,
        usable_loss_endpoint_ticks=losses,fr_knee_saturated_endpoint_count=len(saturated),
        first_fr_knee_saturation_tick=saturated[0]['endpoint_tick'] if saturated else None,
        last_fr_knee_saturation_tick=saturated[-1]['endpoint_tick'] if saturated else None,
        all_endpoint_rows=rr,selected_rows=selected))

result=dict(schema='wlr50_clean.readonly_two_student_RR_capture_support_loss.v1',run=str(RUN),
    scope='Only episode1/4 audit rows from the sealed P07 block; referenced P12 and DET reports read separately; no Torch/models/physics run',
    caveats=['Sensor servo evidence is exact canonical reconstruction at endpoint-1; force/geometry is endpoint.',
             'No complete 120Hz contact stream in this training run. Consecutive-air counters can localize AIR onset, but earlier low-force TOP loss is not excluded between endpoints.',
             'Public owner17 is decoded from same-row input, or next row input aligned to endpoint. Native owner receipt is absent.',
             'Final minus headroom candidate is the aggregate later transform; it is not automatically current policy effect or owner-only effect.',
             'Multiple source/policy/body/contact variables change together; no physical single-channel counterfactual performed.'],
    episodes=episodes)
(OUT/'892385_two_student_RR_support_loss_readonly.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
for e in episodes:
    print('EPISODE',e['episode_index'],'placement',e['placement_tick'],'losses',e['usable_loss_endpoint_ticks'],
          'FRk saturation',e['first_fr_knee_saturation_tick'],e['last_fr_knee_saturation_tick'],e['fr_knee_saturated_endpoint_count'])
    for r in e['selected_rows']:
        rr=r['legs'].get('RR',{})
        print(json.dumps(dict(tick=r['endpoint_tick'],phase=r['phase'],rr=[rr.get('contact_mode'),rr.get('bearing_force_n'),rr.get('clearance_m'),rr.get('front_distance_m'),rr.get('consecutive_air_samples')],
            N=[round(v,3) for v in r['source_nominal_full12'][:8]],mapped=[round(v,3) for v in r['mapped_nominal_full12'][:8]],
            request=[round(v,3) for v in r['filtered_request_full12'][:8]],final=[round(v,3) for v in r['final_full12'][:8]],
            actual=[round(v,3) for v in r['actual_servo_canonical_deg_pre_step']],
            owner=r['endpoint_owner17_via_next_input'][8:] if r['endpoint_owner17_via_next_input'] else None,
            later=[round(v,3) for v in r['final_minus_headroom_candidate_full12'][:8]],
            wheels=[round(v,3) for v in r['final_full12'][8:]],layers=r['source_layers']),ensure_ascii=False))

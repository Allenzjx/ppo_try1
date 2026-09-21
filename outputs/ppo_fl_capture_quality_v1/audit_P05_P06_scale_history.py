"""One sealed episode's handoff only; no actor forward, optimizer, or simulator."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/ppo_fl_capture_quality_v1/train/20260918T0412242524179Z_gf2e552406ea7_541c9a6d6c8e436d811fa7bcd64a885d'
OUT = Path(__file__).with_name('P05_P06_scale_history_audit.json')


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    digest = hashlib.sha256()
    rows = []
    with (RUN / 'residual_and_projection_audit.jsonl').open('rb') as stream:
        for line in itertools.islice(stream, 282):
            digest.update(line)
            rows.append(json.loads(line))
    assert len(rows) == 282 and rows[-1]['applied_audit']['physics_tick'] == 2249
    assert rows[-1]['terminal'] is True
    cfgpath = ROOT / 'configs/ppo_fl_capture_quality_v1/execution_profile.yaml'
    cfg = yaml.safe_load(cfgpath.read_text())['residual']
    caps = cfg['phase_caps_full12']
    selected = [r for r in rows if 2112 <= r['applied_audit']['physics_tick'] <= 2249]
    last = next(r for r in selected if r['applied_audit']['physics_tick'] == 2160)
    first = next(r for r in selected if r['applied_audit']['physics_tick'] == 2168)
    assert last['applied_audit']['phase_id'] == 'P05'
    assert first['applied_audit']['phase_id'] == 'P06'
    assert first['policy_request']['previous_raw_from_current_observation_full12'] == last['raw_policy_action_full12']
    same_raw = []
    order = first['applied_audit']['actuator_target_effect_audit']['canonical_order']
    for index, name in enumerate(order):
        previous = last['raw_policy_action_full12'][index]
        current = first['raw_policy_action_full12'][index]
        old_cap, new_cap = caps['P05'][index], caps['P06'][index]
        same_raw.append({'channel': name, 'unit': 'deg' if index < 8 else 'rad/s',
            'P05_cap': old_cap, 'P06_cap': new_cap, 'cap_ratio': new_cap / old_cap,
            'last_P05_raw': previous, 'first_P06_raw': current,
            'last_raw_old_cap_unfiltered': old_cap * math.tanh(previous),
            'same_last_raw_new_cap_unfiltered': new_cap * math.tanh(previous),
            'first_new_raw_new_cap_unfiltered': new_cap * math.tanh(current),
            'scale_only_increment': (new_cap-old_cap) * math.tanh(previous),
            'raw_change_increment_at_new_cap': new_cap * (math.tanh(current)-math.tanh(previous)),
            'last_P05_filtered_request': last['applied_audit']['projected_residual_full12'][index],
            'first_P06_filtered_request': first['applied_audit']['projected_residual_full12'][index]})
    compact = []
    for row in selected:
        audit = row['applied_audit']
        request = row['policy_request']
        native = audit['actuator_target_effect_audit']
        headroom = native['policy_headroom_evidence']
        index = 3
        previous = request['previous_raw_from_current_observation_full12'][index]
        raw = request['selected_raw_full12'][index]
        conditional = request['conditional_mean_full12'][index]
        cap = caps[audit['phase_id']][index]
        compact.append({'end_tick':audit['physics_tick'], 'start_tick':audit['physics_tick']-audit['physics_ticks'],
            'phase':audit['phase_id'], 'end_phase':audit['end_phase_id'], 'global_decision':row['global_policy_decision'],
            'FR_knee':{
                'previous_raw':previous, 'base_mean':request['base_mean_full12'][index],
                'conditional_mean':conditional, 'effective_sigma':request['effective_sigma_full12'][index],
                'selected_raw':raw, 'innovation':raw-conditional,
                'innovation_sigma':(raw-conditional)/request['effective_sigma_full12'][index],
                'phase_cap_deg':cap, 'unfiltered_desired_deg':cap*math.tanh(raw),
                'filtered_request_deg':audit['projected_residual_full12'][index],
                'effective_headroom_residual_deg':headroom['effective_policy_residual_full12'][index],
                'nominal_mapped_plus_controller_deg':headroom['baseline_native_plus_controller_full12'][index],
                'native_axis_target_rad':native['actual_native_targets']['servo_position_rad'][index],
                'pre_dispatch_measured_native_axis_rad':native['tracking_reference_evidence']['actual_measured_physical_rad'][index],
                'final_drive_canonical_deg':audit['actual_drive_target_full12'][index],
                'last_tick_final_drive_delta_deg':audit['actual_drive_target_full12'][index]-native['previous_final_drive_servo_deg'][index],
                'final_slew_limit_deg_per_tick':native['tracking_reference_evidence']['maximum_delta_deg'],
                'final_slew_modified_target':abs(audit['actual_drive_target_full12'][index]-headroom['candidate_native_target_before_final_slew_full12'][index])>1e-9,
                'headroom_clipped':index in headroom['clipped_servo_indices'],
                'tracking_reference_used':native['tracking_reference_evidence']['channels'][index]['reference_used'],
            },
            'transition':audit['phase_transition_action_jump'],
            'light_tick_trace':audit['actuator_target_effect_audit_ticks'] if audit['phase_transition_action_jump'] else [],
            'terminal':row['terminal']})
    p06 = [r for r in compact if r['phase']=='P06']
    i = 3
    oldphysical = last['applied_audit']['projected_residual_full12'][i]
    newcap = caps['P06'][i]
    physical_history_latent = math.atanh(oldphysical/newcap)
    first_req = first['policy_request']
    hypothetical_cond = .1*first_req['base_mean_full12'][i]+.9*physical_history_latent
    receipt = {'schema':'wlr50_clean.bounded_P05_P06_scale_history_audit.v1',
        'source_run':str(RUN), 'audit_first282_lines_sha256':digest.hexdigest(),
        'execution_profile_sha256':hashlib.sha256(cfgpath.read_bytes()).hexdigest(),
        'scope':{'rows_read':282,'selected_decisions':len(compact),'new_forwards':0,'optimizer_steps':0,
            'physics_started':False,'production_changed':False,'all_values_actual_except_labeled_algebra':True},
        'same_raw_scale_algebra':same_raw, 'FR_knee_decision_window':compact,
        'P06_request_minus_previous_endpoint_deltas_deg':[r['FR_knee']['filtered_request_deg']-(last['applied_audit']['projected_residual_full12'][i] if j==0 else p06[j-1]['FR_knee']['filtered_request_deg']) for j,r in enumerate(p06)],
        'all_P06_raw_history_exact_previous_sample':all(rows[k]['policy_request']['previous_raw_from_current_observation_full12']==rows[k-1]['raw_policy_action_full12'] for k in range(270,282)),
        'mean_formula_max_abs_error':max(abs(r['policy_request']['conditional_mean_full12'][i]-(.1*r['policy_request']['base_mean_full12'][i]+.9*r['policy_request']['previous_raw_from_current_observation_full12'][i])) for r in selected),
        'first_transition':first['applied_audit']['phase_transition_action_jump'][0],
        'physical_history_center_algebra_not_run':{'FR_knee_previous_filtered_request_deg':oldphysical,
            'P06_cap_deg':newcap,'exact_inverse_tanh_center':physical_history_latent,
            'current_conditional_mean':first_req['conditional_mean_full12'][i],
            'hypothetical_conditional_mean_same_base_rho':hypothetical_cond,
            'hypothetical_mean_physical_request_deg':newcap*math.tanh(hypothetical_cond),
            'not_a_sample_or_closed_loop_prediction':True},
        'limitations':['Unfiltered values and decomposition are algebra over actual saved raw, not separately executed commands.',
            'The native pre-dispatch measured joint value is before the final physics step, not a post-step readback.',
            'Full per-tick native targets were not logged in this training run; light tick verification and decision-end targets are used.',
            'No single-joint causality of body collision is established.']}
    OUT.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(OUT), 'FR_knee_scale':same_raw[3],
        'FR_knee_window':[{k:v for k,v in r.items() if k not in ('transition','light_tick_trace')} for r in compact],
        'P06_deltas':receipt['P06_request_minus_previous_endpoint_deltas_deg'],
        'physical_history_hypothesis':receipt['physical_history_center_algebra_not_run']},indent=2))


if __name__ == '__main__':
    main()

"""Bounded stdlib read of the already identified first41 success, no live tail."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE/'tracking_fixed_first_stochastic_capture_readonly.json'


def analyze():
    evidence = json.loads(EVIDENCE.read_text())
    source = evidence['fixed_snapshot']['decisions']
    with Path(source['path']).open('rb') as stream:
        data = stream.read(source['snapshot_bytes'])
    if len(data) != source['snapshot_bytes'] or not data.endswith(b'\n'):
        raise ValueError('fixed successful snapshot boundary changed')
    rows = [json.loads(s) for s in data.splitlines()]
    active = [r for r in rows if r.get('kind') == 'activated_on_policy']
    if len(active) != 41:
        raise ValueError('exactly the identified first41 credited samples required')
    result = []
    errors = dict(composed_conditional_mean=0., inverse_history_target=0.,
                  selected_vs_issued=0.,stored_oldlogp_vs_request=0.)
    for i,r in enumerate(active,1):
        p,info = r['policy_request'],r['step_info']
        before,after = r['before_local']['metrics'],info['rr_capture_local']['metrics']
        if (len(r['observation']) != 448 or not all(math.isfinite(x) for x in r['observation'])
                or r['PPO_credit'] != 1 or p.get('independent_diagnostic')
                or p['history_kernel_applications'] != 1 or p['sampling_draws'] != 1
                or p['extra_random_draws'] != 0):
            raise ValueError('unexpected observation/action sampling provenance')
        raw,mean = p['selected_raw_full12'],p['conditional_mean_full12']
        prior,local,center,rho = p['prior_raw_mean_full12'],p['local_raw_mean_delta_full12'],p['history_center_full12'],p['rho']
        if type(rho) not in (float,int) or not 0 <= rho < 1:
            raise ValueError('unexpected scalar HISTORY rho')
        target = [(a-rho*h)/(1-rho)-n for a,h,n in zip(raw,center,prior)]
        expected = [(1-rho)*(n+l)+rho*h for n,l,h in zip(prior,local,center)]
        rebuilt = [(1-rho)*(n+t)+rho*h for n,t,h in zip(prior,target,center)]
        errors['composed_conditional_mean'] = max(errors['composed_conditional_mean'],max(abs(a-b) for a,b in zip(expected,mean)))
        errors['inverse_history_target'] = max(errors['inverse_history_target'],max(abs(a-b) for a,b in zip(rebuilt,raw)))
        errors['selected_vs_issued'] = max(errors['selected_vs_issued'],max(abs(a-b) for a,b in zip(raw,info['raw_policy_action_full12'])))
        errors['stored_oldlogp_vs_request'] = max(errors['stored_oldlogp_vs_request'],abs(r['old_logp'][0]-p['selected_raw_log_probability']))
        audit = info['actuator_target_effect_audit']
        local_snap = info['rr_capture_local']
        result.append(dict(index1=i,global_decision=r['global_decision'],
            start_tick=before['tick'],end_tick=after['tick'],
            phase=[info['phase_id'],info['end_phase_id']],
            gap_before_mm=1000*before['gap_m'],gap_after_mm=1000*after['gap_m'],
            actual_before_RR=before['actual_rr_hip_knee_deg'],actual_after_RR=after['actual_rr_hip_knee_deg'],
            current_attempt_eligible=before['current_attempt_capture_eligible'],
            legal_xy_before=before['within_top_xy'],legal_xy_after=after['within_top_xy'],
            free_air_before=before['free_air'],ground_after=after['ground_contact'],
            TOP=after['current_top_contact'],bearing=after['current_top_bearing'],
            force_N=after['bearing_force_n'],hold_s=local_snap['hold_elapsed_s'],
            rho=rho,raw_full12=raw,conditional_mean_full12=mean,conditional_sigma_full12=p['active_conditional_std_full12'],
            prior_head_full12=prior,history_center_full12=center,current_local_head_full12=local,
            inverse_history_local_target_full12=target,
            standardized_raw_innovation_full12=[(a-m)/s for a,m,s in zip(raw,mean,p['active_conditional_std_full12'])],
            source_N_RR=info['nominal_action_full12'][6:8],
            mapped_N_RR=audit['native_drive_target_full12'][6:8],
            generic_RR_bias=audit['controller_drive_bias_full12'][6:8],
            requested_RR=audit['policy_headroom_evidence']['requested_policy_residual_full12'][6:8],
            effective_RR=audit['policy_headroom_evidence']['effective_policy_residual_full12'][6:8],
            final_RR=info['actual_drive_target_full12'][6:8],
            clipped_servo_indices=audit['policy_headroom_evidence']['clipped_servo_indices'],
            local_success=info['local_task_success'],terminal=info['local_reward']['terminated']))
    if not result[-1]['local_success'] or not result[-1]['terminal']:
        raise ValueError('identified trajectory lacks its actual successful terminal')
    return dict(schema='wlr50_clean.current448_success_aux_readonly.v1',
        snapshot=dict(path=source['path'],bytes=len(data),sha256=hashlib.sha256(data).hexdigest()),
        source_evidence=dict(path=str(EVIDENCE),sha256=hashlib.sha256(EVIDENCE.read_bytes()).hexdigest()),
        analysis_only=True,new_PPO_updates_at_snapshot=0,new_AUX_updates=0,
        no_teacher_dataset_created=True,math_checks_max_abs=errors,rows=result)


if __name__=='__main__':
    output = HERE/'current448_success_aux_readonly.json'
    if output.exists():
        raise FileExistsError(output)
    result = analyze()
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    for r in result['rows']:
        print(json.dumps({k:r[k] for k in ('index1','start_tick','end_tick','gap_before_mm','gap_after_mm','TOP','hold_s','source_N_RR','mapped_N_RR','generic_RR_bias','clipped_servo_indices')}))
    print(json.dumps(result['math_checks_max_abs']))


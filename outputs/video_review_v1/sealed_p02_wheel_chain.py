"""Bounded sealed P01-tail/P02/P03-entry wheel chain; no policy or simulation.

Uses actual native audit's same-pre-tick zero-current-policy branch, never an
independently recomputed nominal or a different trajectory as the policy effect.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import yaml

spec=importlib.util.spec_from_file_location('p02_media_reuse',Path(__file__).with_name('sealed_rr_review.py'))
review=importlib.util.module_from_spec(spec); spec.loader.exec_module(review)
base=review.base
LEGS=review.LEGS; NAMES=review.NAMES; SIGNS=review.SIGNS


def bounded_rows(path,include):
    """Record exactly what was read; deliberately do not hash an entire large log."""
    digest=hashlib.sha256(); result=[]; byte_count=0; line_count=0
    before=path.stat()
    with path.open('rb') as stream:
        for line,raw in enumerate(stream,1):
            digest.update(raw); byte_count+=len(raw); line_count=line
            if not raw.strip():continue
            row=json.loads(raw)
            if not include(row):break
            result.append((line,row))
    after=path.stat()
    base.require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Sealed log changed during bounded read')
    return result,{'path':str(path),'source_file_bytes':before.st_size,'read_prefix_bytes':byte_count,
        'read_prefix_lines':line_count,'read_prefix_sha256':digest.hexdigest(),
        'whole_file_hashed':byte_count==before.st_size,'partial_prefix_hash_is_not_manifest_artifact_hash':True}


def native_prefix(source):
    first_p03=None
    def include(row):
        nonlocal first_p03
        phase=row['source_phase_id']; tick=row['episode_physics_tick']
        if phase=='P03' and first_p03 is None:first_p03=tick
        return phase in ('P01','P02','P03') and (first_p03 is None or tick<first_p03+16)
    rows,binding=bounded_rows(source/'native_tick_audit.jsonl',include)
    return {r['episode_physics_tick']:(line,r) for line,r in rows},binding


def list_stats(values):
    return None if not values else {'n':len(values),'min':min(values),'max':max(values),'mean':statistics.fmean(values)}


def sample(native,physical,height,indices,decision,cap):
    tick=native['episode_physics_tick']; audit=native['native_audit']
    base.require(tuple(audit['canonical_order'][8:])==NAMES,'Unexpected Full12 wheel order')
    actual_native=audit['actual_native_targets']['wheel_velocity_rad_s']
    zero_native=audit['counterfactual_native_targets']['wheel_velocity_rad_s']
    delta_native=audit['native_target_delta']['wheel_velocity_rad_s']
    final=[x*s for x,s in zip(actual_native,SIGNS)]
    zero=[x*s for x,s in zip(zero_native,SIGNS)]
    effect=[x*s for x,s in zip(delta_native,SIGNS)]
    qd=[physical['wheels'][n]['velocity_rad_s'] for n in NAMES]
    native_qd=None
    if height is not None:
        base.require(height['physics_tick']==tick and height['clock_unchanged'],'Wrong readback clock')
        base.require(height['joint_velocity_native_rad_s']['source']=='robot.data.joint_vel','Wrong native qd provenance')
        native_qd=[height['joint_velocity_native_rad_s']['value'][i] for i in indices]
        base.require(all(math.isclose(n,c*s,abs_tol=1e-8,rel_tol=1e-8) for n,c,s in zip(native_qd,qd,SIGNS)),
            'Native qd/canonical sign mismatch')
    headroom=audit.get('policy_headroom_evidence') or {}
    nominal=native['nominal_full12'][8:]; mapped=audit['native_drive_target_full12'][8:]
    projected=native['projected_residual_full12'][8:]; controller=audit['controller_drive_bias_full12'][8:]
    effective=(headroom.get('effective_policy_residual_full12') or audit['projected_residual_full12'])[8:]
    task=((decision or {}).get('step_info') or {}).get('semantic_task') or {}
    contacts=[physical['contacts'][n.replace('_ankle','_wheel')] for n in NAMES]
    return {'tick':tick,'simulation_time_s':physical['simulation_time_s'],'phase':native['source_phase_id'],
        'N_canonical_rad_s':nominal,'mapped_N_canonical_rad_s':mapped,
        'raw_policy_latent':audit['raw_policy_action_full12'][8:],'configured_cap_rad_s':cap,
        'actual_residual_permission_mask':audit['phase_mask_full12'][8:],
        'mask_object':'PPO additive residual permission; not nominal or actuator-write selection',
        'raw_tanh_times_cap_rad_s':None if cap is None else [math.tanh(x)*s for x,s in zip(audit['raw_policy_action_full12'][8:],cap)],
        'transformed_diagnostic_is_not_filtered_policy_effect':True,
        'projected_filtered_residual_rad_s':projected,'headroom_effective_requested_residual_rad_s':effective,
        'controller_bias_rad_s':controller,'zero_current_policy_target_canonical_rad_s':zero,
        'same_tick_effective_target_delta_canonical_rad_s':effect,
        'same_tick_effective_target_delta_native_rad_s':delta_native,
        'final_target_canonical_rad_s':final,'final_target_native_rad_s':actual_native,
        'measured_qd_canonical_rad_s':qd,'measured_qd_native_rad_s':native_qd,
        'native_qd_available':native_qd is not None,
        'sensor_command_canonical_rad_s':[physical['wheels'][n]['command_rad_s'] for n in NAMES],
        'contact_class':[c['contact_class'] for c in contacts],
        'ground_contact':[c['ground']['active'] for c in contacts],
        'ground_normal_force_N':[c['ground']['normal_force_n'] for c in contacts],
        'obstacle_contact':[c['obstacle']['active'] for c in contacts],
        'RR_knee_target_deg':physical['joints']['rear_right_knee']['command_deg'],
        'RR_knee_actual_deg':physical['joints']['rear_right_knee']['position_deg'],
        'FR_gap_m':physical['wheels']['front_right_ankle']['bottom_w_m'][2]-physical['obstacle']['top_z_m'],
        'body_origin_z_m':physical['base']['position_w_m'][2],
        'body_linear_velocity_m_s':physical['base']['linear_velocity_w_m_s'],
        'dispatch_flags':{key:audit.get(key) for key in ('verified','setter_dispatch_targets_equal','actual_mapping_matches_dispatch','same_tick_counterfactual')},
        'counterfactual_scope':audit.get('counterfactual_scope'),'last_write_evidence':audit.get('actual_target_source'),
        'source_provider_diagnostics_at_decision_endpoint':task.get('nominal_provider_diagnostics'),
        'unique_channel_owner_not_recorded':True,
        'arithmetic_unclipped_sum_residual_error':max(abs(f-(m+c+p)) for f,m,c,p in zip(final,mapped,controller,effective)),
        'audit_delta_float32_subtraction_error':max(abs(e-(f-z)) for e,f,z in zip(effect,final,zero))}


def window(samples,start,end):
    selected=[r for t,r in samples.items() if start<=t<=end]; output=[]
    for j,leg in enumerate(LEGS):
        positive=[r for r in selected if r['N_canonical_rad_s'][j]>0]
        output.append({'leg':leg,'positive_nominal_ticks':len(positive),
            'final_target_abs_below_0_05_ticks':sum(abs(r['final_target_canonical_rad_s'][j])<.05 for r in positive),
            'final_target_below_half_N_ticks':sum(r['final_target_canonical_rad_s'][j]<.5*r['N_canonical_rad_s'][j] for r in positive),
            'same_tick_policy_effect_negative_ticks':sum(r['same_tick_effective_target_delta_canonical_rad_s'][j]<0 for r in positive),
            'measured_qd_abs_below_0_05_ticks':sum(abs(r['measured_qd_canonical_rad_s'][j])<.05 for r in positive),
            **{key:list_stats([r[key][j] for r in positive]) for key in ('N_canonical_rad_s','raw_policy_latent',
                'projected_filtered_residual_rad_s','same_tick_effective_target_delta_canonical_rad_s',
                'final_target_canonical_rad_s','measured_qd_canonical_rad_s')}})
    return {'start_tick':start,'end_tick':end,'actual_ticks':len(selected),'by_wheel':output,
        'thresholds_are_descriptive_not_acceptance':True}


def explain(row,j):
    nominal=row['N_canonical_rad_s'][j]; mask=row['actual_residual_permission_mask'][j]
    mapped=row['mapped_N_canonical_rad_s'][j]; final=row['final_target_canonical_rad_s'][j]
    zero=row['zero_current_policy_target_canonical_rad_s'][j]; effect=row['same_tick_effective_target_delta_canonical_rad_s'][j]
    qd=row['measured_qd_canonical_rad_s'][j]
    facts=[]
    if mask==0:facts.append('residual permission off; nominal must be inspected separately')
    if abs(nominal-mapped)>1e-6:facts.append('source N and actual mapped N differ')
    if nominal>0 and zero>0 and effect<0 and final<.5*zero:
        facts.append('same-tick policy effect reduces baseline target below half')
    if abs(final)<.05:facts.append('near-zero final target')
    if abs(qd)<.05:facts.append('near-zero measured rotation')
    if abs(final)>=.05 and abs(qd)<.05:facts.append('target present but weak measured tracking; cause not isolated')
    return facts or ['no listed cancellation/near-zero predicate; not proof of good traction']


def first_observations(samples):
    result={}
    for j,leg in enumerate(LEGS):
        tests={
            'residual_permission_off':lambda r:r['actual_residual_permission_mask'][j]==0,
            'mapped_N_differs_from_source_N':lambda r:abs(r['N_canonical_rad_s'][j]-r['mapped_N_canonical_rad_s'][j])>1e-6,
            'policy_reduces_positive_same_state_baseline_below_half':lambda r:
                r['N_canonical_rad_s'][j]>0 and r['zero_current_policy_target_canonical_rad_s'][j]>0
                and r['same_tick_effective_target_delta_canonical_rad_s'][j]<0
                and r['final_target_canonical_rad_s'][j]<.5*r['zero_current_policy_target_canonical_rad_s'][j],
            'final_target_near_zero_while_N_positive':lambda r:
                r['N_canonical_rad_s'][j]>0 and abs(r['final_target_canonical_rad_s'][j])<.05,
            'measured_rotation_near_zero_while_target_present':lambda r:
                abs(r['final_target_canonical_rad_s'][j])>=.05 and abs(r['measured_qd_canonical_rad_s'][j])<.05}
        result[leg]={}
        for label,predicate in tests.items():
            found=next((r for _,r in sorted(samples.items()) if predicate(r)),None)
            result[leg][label]=None if found is None else {k:found[k] for k in ('tick','simulation_time_s','phase')}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True); parser.add_argument('--name',required=True)
    parser.add_argument('--smoke',action='store_true',help='Old sealed-source parser smoke; never latest-C evidence')
    args=parser.parse_args(); source=args.source.resolve(strict=True)
    run=base.read_json(source.parent/'run_manifest.json')
    base.require(run.get('completed_at_utc') and run.get('lifecycle')!='RUNNING','Only finalized sources; no active log reads')
    manifest_path=source/'semantic_video_source_manifest.json'; manifest=base.read_json(manifest_path)
    base.require(args.smoke or (manifest.get('experiment_id')=='residual_rr_fix_v1' and manifest.get('role')=='C'
        and manifest.get('from_phase')=='P01' and manifest.get('diagnostic_intervention') is None),'Need formal natural P01 RR source')
    base.require(args.name and all(c.isalnum() or c in '_-' for c in args.name),'Unsafe new output name')
    destination=review.OUT/args.name; base.require(not destination.exists(),'Never overwrite'); destination.mkdir()
    native,nb=native_prefix(source); base.require(native,'No bounded front-window samples')
    end=max(native); bindings={'native':nb}
    others={}
    for name,key in [('physical_observations.jsonl','physics_tick'),('height_diagnostics.jsonl','physics_tick'),('video_policy_decisions.jsonl','end_tick')]:
        values,binding=bounded_rows(source/name,lambda r,key=key:r[key]<=end)
        others[name]={r[key]:r for _,r in values}; bindings[name]=binding
    startup_path=source/'height_diagnostics_startup.json'; startup=base.read_json(startup_path)
    base.require(review.sha(startup_path)==manifest['artifacts'][startup_path.name]['sha256'],'Startup binding differs')
    names=startup['joint_names_native_order']['value']; indices=[names.index(n) for n in NAMES]
    config=review.recorded_config(manifest,'execution_profile.yaml')
    caps=config['residual'].get('phase_caps_full12') or {}
    samples={t:sample(r,others['physical_observations.jsonl'][t],others['height_diagnostics.jsonl'].get(t),indices,
        others['video_policy_decisions.jsonl'].get(t),None if r['source_phase_id'] not in caps else caps[r['source_phase_id']][8:])
        for t,(_,r) in native.items()}
    p02=[t for t,r in samples.items() if r['phase']=='P02']; p03=[t for t,r in samples.items() if r['phase']=='P03']
    start=max(1,min(p02)-24) if p02 else 1
    selected=sorted({t for t in (start,min(p02) if p02 else None,400,max(p02) if p02 else None,min(p03) if p03 else None,end) if t in samples})
    evaluation=manifest['physical_episode']['physical_task_evaluation']
    result={'schema':'wlr50_clean.sealed_P02_wheel_chain.v1','source':str(source),'source_manifest_sha256':review.sha(manifest_path),
        'smoke_only':args.smoke,'latest_candidate_evidence':not args.smoke,'new_physics_steps':0,'optimizer_updates':0,
        'checkpoint_decisions':(manifest.get('checkpoint_load_provenance') or {}).get('saved_global_policy_decisions'),
        'recorded_full_result':{k:evaluation.get(k) for k in ('success','termination_reason','termination_source')},
        'wheel_order':LEGS,'native_joint_indices':indices,'canonical_to_native_signs':SIGNS,
        'bounded_reads':bindings,'config_binding':manifest['runtime_contract']['selected_configuration']['execution_profile.yaml'],
        'window':{'P01_tail_start':start,'P02_first':min(p02) if p02 else None,'P02_last':max(p02) if p02 else None,
            'P03_first':min(p03) if p03 else None,'last_read_included_tick':end},
        'tick400':samples.get(400),'representative_samples':[samples[t] for t in selected],
        'chain':{'all12_permission_one':all(r['native_audit']['phase_mask_full12']==[1]*12 for _,r in native.values()),
            'dispatch_verified_every_bounded_tick':all(all(r['dispatch_flags'].values()) for r in samples.values()),
            'same_pre_tick_counterfactual_scope_every_tick':all(r['counterfactual_scope']=='same_pre_tick_state_without_current_ppo_residual' for r in samples.values()),
            'max_unclipped_sum_error_rad_s':max(r['arithmetic_unclipped_sum_residual_error'] for r in samples.values()),
            'max_same_tick_float32_delta_error_rad_s':max(r['audit_delta_float32_subtraction_error'] for r in samples.values()),
            'all_one_mask_is_not_tracking_or_task_success':True},
        'P01_tail_P02_P03_entry':window(samples,start,end),
        'P02_only':window(samples,min(p02),max(p02)) if p02 else None,
        'first_observed_layer_conditions':first_observations(samples),
        'tick400_channel_facts':None if 400 not in samples else {leg:explain(samples[400],j) for j,leg in enumerate(LEGS)},
        'owner_scope':'Full held N is actual recorded value. No unique source owner or wheel-specific final writer ID is logged; changed_channels does not reassign owner. Native staged/dispatched audit verifies its named write buffers, not an independent motor-force causal proof.',
        'history_scope':'Projected/filtered residual and same-pre-tick zero-current-policy target delta are both retained. Raw tanh*cap is diagnostic only; no independent N recalculation, history reset, or different-run subtraction is called the current policy effect.',
        'limitations':['Near-zero measured qd does not prove masking; nonzero qd does not prove traction.',
            'No isolated driver/load/contact causality experiment.', 'P02 evidence is not full obstacle success.']}
    base.write_new_json(destination/'p02_wheel_chain.json',result)
    lines=['# Sealed P02 four-wheel chain', '',f"Source: `{source}`",'',f"Smoke-only: {args.smoke}; recorded checkpoint decisions: {result['checkpoint_decisions']}. No new physics/optimizer.",'']
    row=samples.get(400)
    if row:
        lines+=['## Actual tick400 (3.333333s)', '', '| Leg | N | mapped N | raw latent | mask | filtered residual | same-state target effect | final canonical / native | measured canonical / native | contact |',
            '|---|---:|---:|---:|---:|---:|---:|---|---|---|']
        for j,leg in enumerate(LEGS):
            native_qd='N/A' if row['measured_qd_native_rad_s'] is None else f"{row['measured_qd_native_rad_s'][j]:+.5f}"
            lines.append(f"| {leg} | {row['N_canonical_rad_s'][j]:+.5f} | {row['mapped_N_canonical_rad_s'][j]:+.5f} | {row['raw_policy_latent'][j]:+.5f} | {row['actual_residual_permission_mask'][j]} | {row['projected_filtered_residual_rad_s'][j]:+.5f} | {row['same_tick_effective_target_delta_canonical_rad_s'][j]:+.5f} | {row['final_target_canonical_rad_s'][j]:+.5f} / {row['final_target_native_rad_s'][j]:+.5f} | {row['measured_qd_canonical_rad_s'][j]:+.5f} / {native_qd} | {row['contact_class'][j]} |")
        lines+=['', 'All velocities/corrections are rad/s; raw is dimensionless. Mask acts on additive residual, not N.',
            f"RR knee target/actual: {row['RR_knee_target_deg']:+.5f}/{row['RR_knee_actual_deg']:+.5f}deg; FR gap {row['FR_gap_m']:+.6f}m; body origin z {row['body_origin_z_m']:.6f}m.",'']
        lines.extend(f"- {leg}: {'; '.join(result['tick400_channel_facts'][leg])}." for leg in LEGS)
    else:lines+=['Tick400 is not in this actual front-stage window; no invented row.']
    lines+=['','## Window and evidence limits','',json.dumps(result['window']), '',result['owner_scope'],'',result['history_scope'],
        '', 'Detailed P02 and handoff-window per-wheel counts, masks, bound source-prefix digests, measured values and actual dispatch checks are in `p02_wheel_chain.json`. Descriptive 0.05/half-N comparisons are not new task thresholds.']
    (destination/'p02_wheel_chain.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps({'output':str(destination),'smoke_only':args.smoke,'window':result['window'],'chain':result['chain']},indent=2))


if __name__=='__main__':main()

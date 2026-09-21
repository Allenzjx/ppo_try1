"""Bounded first prefix + first86 collected rows; no network/storage recomputation."""
from pathlib import Path
import json
from audit_first_completed_update import read_prefix

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
RUN=ROOT/'runs/ppo_fl_capture_quality_v1/train/20260918T0909562623513Z_g3a50657a96c9_d0f934ba1eae46958f7c8b880fdac987'
prefix=[]
with (RUN/'prefix_evidence.jsonl').open() as f:
    for _ in range(410):
        line=f.readline()
        assert line and line.endswith('\n')
        row=json.loads(line);prefix.append(row)
        if row['kind']=='policy_credit_start':break
    else:raise AssertionError('first accepted prefix missing')
steps=[x for x in prefix if x['kind']=='checkpoint_prefix_decision']
result=next(x for x in prefix if x['kind']=='checkpoint_prefix_result')
provenance=next(x for x in prefix if x['kind']=='checkpoint_prefix_start')['prefix_policy_provenance']
start=prefix[-1]['start']
rows,digest=read_prefix(RUN/'residual_and_projection_audit.jsonl',86)
assert rows[-1]['applied_audit']['physics_tick']==3896
with (RUN/'optimizer_updates.jsonl').open() as f:
    line=f.readline()
    first_update=json.loads(line) if line and line.endswith('\n') else None
metadata=json.loads(Path(provenance['checkpoint_path']).with_name('checkpoint_step_000183552_manifest.json').read_text())
capture=rows[67]
assert capture['applied_audit']['semantic_task']['history']['event_ticks']['placed']['FL']==3748
checks=dict(prefix_accepted_no_fallback=result['accepted'] is True and result['miss'] is None,
    prefix_401_decisions_to_P05_tick3208=len(steps)==401 and result['prefix_decisions']==401 and start['actual_phase']=='P05' and start['physics_tick']==3208,
    prefix_all_credit_false=all(x['policy_credit'] is False for x in prefix),
    prefix_storage_excluded=all(x['applied_audit']['prefix_checkpoint_policy_data_in_ppo_storage'] is False for x in rows),
    first_real_credit_183553=rows[0]['global_policy_decision']==metadata['global_policy_decisions']+1==183553,
    exact_raw_history_continuity=rows[0]['policy_request']['previous_raw_from_current_observation_full12']==steps[-1]['raw_policy_action_full12'],
    source_checkpoint_and_actor_provenance_match=metadata['checkpoint_sha256']==provenance['checkpoint_sha256'] and metadata['actor_parameter_sha256']==provenance['actor_parameter_sha256']==provenance['frozen_actor_parameter_sha256'],
    collected_before_first128_update=capture['global_policy_decision']-metadata['global_policy_decisions']==68<128,
    actual_first_optimizer_after_capture=first_update is not None and first_update['ppo_update']==1400 and first_update['global_policy_decisions']==183680>capture['global_policy_decision'],
    first_optimizer_before_actor_hash_is_source=first_update is not None and first_update['actor_parameter_sha256_before']==provenance['actor_parameter_sha256'])
assert all(checks.values()),checks
samples=[]
for i in (0,66,67,85):
    r=rows[i];a=r['applied_audit'];q=r['policy_request'];e=a['semantic_task']['physical_evaluator']
    n=a['actuator_target_effect_audit'];h=n['policy_headroom_evidence'];tr=n['tracking_reference_evidence']
    samples.append(dict(credited_sample=i+1,global_policy_decision=r['global_policy_decision'],end_tick=a['physics_tick'],
        phase=a['phase_id'],end_phase=a['end_phase_id'],reward=r['reward'],terminal=r['terminal'],
        FL_hip_knee=dict(base_mean=q['base_mean_full12'][:2],conditional_mean=q['conditional_mean_full12'][:2],raw_sample=q['selected_raw_full12'][:2],
            filtered_request_deg=a['projected_residual_full12'][:2],effective_residual_deg=h['effective_policy_residual_full12'][:2],
            nominal_deg=a['nominal_action_full12'][:2],mapped_baseline_deg=h['baseline_native_plus_controller_full12'][:2],
            final_deg=a['actual_drive_target_full12'][:2],
            measured_native_before_last_dispatch_rad=tr['actual_measured_physical_rad'][:2],
            derived_measured_canonical_before_last_dispatch_deg=[tr['channels'][j]['nominal_deg']-tr['channels'][j]['current_actual_canonical_error_deg'] for j in (0,1)],
            canonical_derivation='logged nominal minus logged current_actual_canonical_error; pre-dispatch, not post-step actual',
            actual_post_step_deg=None),
        contacts={leg:{k:e['current_legs'][leg][k] for k in ('air','top_contact','ground_contact','support','bearing_force_n','clearance_m','front_distance_m')}
            for leg in ('FL','FR','RL','RR')},
        historical_FL_placed=e['history']['placed']['FL'],FL_placed_tick=e['history']['event_ticks']['placed'].get('FL'),
        all_physical_ticks_verified=a['actuator_target_effect_audit_summary']['all_ticks_verified'],
        endpoint_dispatch_verified=n['verified'] and n['setter_dispatch_targets_equal'] and n['actual_mapping_matches_dispatch'],
        mask=n['phase_mask_full12'],clipped_servo_indices=h['clipped_servo_indices'],
        sample_mode=q['mode'],extra_model_forwards=q['extra_model_forwards'],extra_random_draws=q['extra_random_draws']))
reference=json.loads((OUT/'CP183552_P05_diagnosis.json').read_text())
ref=next(s for s in reference['selected_same_tick_chain'] if s['tick']==3208)
record=dict(schema='wlr50_clean.block12_first_capture.v1',run=str(RUN),read_boundary='first completed prefix + first86 learner rows through tick3896 + first optimizer receipt only',
    learner_first86_lines_sha256=digest,checks=checks,source_checkpoint=provenance['checkpoint_path'],source_checkpoint_sha256=provenance['checkpoint_sha256'],
    prefix_result=result,prefix_credit=0,prefix_end_tick=3208,first_learner_global=183553,
    capture=dict(physics_tick=3748,time_s=3748/120,within_credited_sample=68,global_policy_decision=183620,P06_entry_tick=3752,new_optimizer_updates_before_capture=0),
    first_optimizer_receipt={k:first_update[k] for k in ('ppo_update','global_policy_decisions','optimizer_steps','actor_parameter_sha256_before','actor_parameter_sha256_after')},
    previous_fixed_eval_entry=dict(seed=4001,tick=3208,FL_gap_mm=ref['FL_gap_mm'],FL_front_mm=ref['FL_front_mm'],FL_contact=ref['FL_actual_contact'],
        prefix_last_raw_equals_eval_raw=steps[-1]['raw_policy_action_full12'][:2]==[ref['channels'][x]['raw'] for x in ('hip','knee')],
        prefix_last_final_equals_eval_final=steps[-1]['actual_drive_target_full12'][:2]==[ref['channels'][x]['final_deg'] for x in ('hip','knee')]),
    samples=samples,missing=['The prefix outcome does not contain FL geometry at tick3208. Actual first learner endpoint tick3216 is recorded; 24.990mm is from the separately sealed same-CP deterministic eval, not fabricated prefix sensing.',
        'Training logs here retain native joint measurement before last dispatch, not post-step canonical actual q. It is explicitly separate from endpoint contact/gap.'],
    limits=['Random closed-loop recovery from an unplaced P05 entry, not learned mean improvement or full-task success.',
        'No single hip causal attribution; all12 channels and body/support evolve.',
        'FL later AIR at tick3896 despite historical placed; no continuous support claim.',
        'First optimizer subsequently completed while this bounded read ran; capture chronologically preceded it.'],
    new_forward_optimizer_simulation_cuda_production_write=False)
with (OUT/'block12_first_capture.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,allow_nan=False)
print(json.dumps({'checks':checks,'capture':record['capture'],'reference':record['previous_fixed_eval_entry']},indent=2))

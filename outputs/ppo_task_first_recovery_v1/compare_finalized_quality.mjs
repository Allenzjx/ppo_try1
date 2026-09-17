// Compact JSON-manifest comparison only. Never read active logs or CSV.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const metrics=['roll_rms_rad','pitch_rms_rad','roll_rate_p95_rad_s','pitch_rate_p95_rad_s',
 'body_x_omega_p95_rad_s','body_y_omega_p95_rad_s','angular_acceleration_p95_rad_s2',
 'applied_servo_rate_rms_deg_s','applied_servo_acceleration_p95_deg_s2',
 'applied_wheel_rate_rms_rad_s2','applied_wheel_acceleration_p95_rad_s3'];
const controlFiles=['semantic_backend.py','semantic_supervisor.py','semantic_residual_adapter.py',
 'semantic_nominal_geometry.py','semantic_height_recovery.py','semantic_tracking_reference.py',
 'semantic_headroom.py','semantic_metrics.py'];
const pick=(o,keys)=>Object.fromEntries(keys.map(k=>[k,o?.[k]??null]));
const finite=x=>typeof x==='number'&&Number.isFinite(x);
function metricSnapshot(q){return {physics_ticks:q.physics_ticks,duration_s:q.duration_s,
 sampled:q.sampled,...pick(q,metrics)};}
export function extractFinalized(run){
 const source=path.join(path.resolve(run),'run_manifest.json');
 const m=JSON.parse(fs.readFileSync(source,'utf8'));
 assert(m.completed_at_utc&&['SUCCEEDED','DIAGNOSTIC_FAILURE'].includes(m.lifecycle),'Only finalized video run manifests');
 const r=m.result,p=r.physical_episode,f=p.physical_task_evaluation,q=p.quality_metrics;
 assert(q.schema==='wlr50_clean.semantic_quality_metrics.v1');
 assert(q.comparison_window==='actual_physics_ticks_excluding_video_padding');
 assert(q.global.physics_ticks===r.episode_physics_ticks-q.nonfinite_terminal_ticks_skipped);
 const legs=['FL','FR','RL','RR'];
 const contact={};
 for(const leg of legs){contact[leg]={};
  for(const suffix of ['touchdown_event_count','touchdown_descent_speed_m_s','touchdown_normal_force_n','contact_rolling_slip_estimate_abs_m_s']){
   contact[leg][suffix]=pick(q.global.motion_contact_diagnostics?.[leg+'_'+suffix],
    ['available','valid_ticks','missing_or_inapplicable_ticks','valid_duration_s','mean','rms','p95_abs','peak_abs','sum']);
  }
 }
 return {source,completed_at_utc:m.completed_at_utc,lifecycle:m.lifecycle,role:r.role,seed:r.seed,
  task_success:f.success,task_reason:f.reason,terminal_reason:f.termination_reason,
  actual_physics_ticks:r.episode_physics_ticks,physical_task_duration_s:p.physical_task_duration_s,
  global:metricSnapshot(q.global),phase_coverage:Object.fromEntries(Object.entries(q.phases).map(([phase,v])=>[phase,metricSnapshot(v)])),
  contact,final_body_collider_minimum_world_z_m:f.body_traversal_geometry?.valid?f.body_traversal_geometry.minimum_w_m[2]:null,
  whole_trajectory_body_height_minimum:null,body_height_scope:'Final physical evaluator rotated collider min only; no whole-trajectory height minimum stored in this compact manifest summary.',
  final_current_leg_evidence:Object.fromEntries(legs.map(leg=>[leg,pick(f.current_legs?.[leg],['support','bearing_verified','bearing_force_n','air','top_contact','ground_contact','current_lift_valid'])])),
  actual_event_ticks:f.history?.event_ticks??null,diagnostic_notes:q.diagnostic_notes,
  fixed_quality_score:q.fixed_quality_score,score_not_task_success:q.score_is_not_success,
  contract:{git:r.runtime_contract.source_git_commit,
   frozen_A:r.runtime_contract.frozen_A_files,
   control_files:Object.fromEntries(controlFiles.map(file=>[file,r.runtime_contract.files['src/wlr50_clean/ppo/'+file]??null])),
   configurations:Object.fromEntries(Object.entries(r.evaluation_configuration).filter(([k])=>k!=='reward_config_path').map(([k,v])=>[k,v.sha256]))}};
}
export function compareFinalized(zeroRun,ppoRun){
 const zero=extractFinalized(zeroRun),ppo=extractFinalized(ppoRun);
 const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
 const matching={seed:zero.seed===ppo.seed,frozen_physics_and_A_files:equal(zero.contract.frozen_A,ppo.contract.frozen_A),
  successful_N_mapping_and_metrics_files:equal(zero.contract.control_files,ppo.contract.control_files),
  non_reward_configuration:equal(zero.contract.configurations,ppo.contract.configurations)};
 const taskMatched=zero.task_success===true&&ppo.task_success===true&&Object.values(matching).every(Boolean);
 return {schema:'wlr50_clean.task_first_finalized_quality_comparison.v1',
  matching,task_ability_matched_for_quality_comparison:taskMatched,
  scope:'Each run entire real natural episode, no video freeze/padding. Unequal durations retained; no fabricated aligned prefix.',
  rows:metrics.map(metric=>({metric,zero:zero.global[metric],ppo:ppo.global[metric],
   relative_change_percent:taskMatched&&finite(zero.global[metric])&&finite(ppo.global[metric])&&zero.global[metric]!==0
    ?100*(ppo.global[metric]/zero.global[metric]-1):null})),
  zero,ppo,stability_superiority_claimed:false,
  claim_limit:'Lower full-window motion in an incomplete/unsafe run is not stability improvement. Even when both succeed, per-metric differences are descriptive one-episode evidence, not robust superiority.',
  contact_limit:'Touchdown speed is wheel-body pre-contact downward speed, not resolved point impact. Force includes support/propulsion and is not automatically harmful impulse. Slip is rolling proxy, not measured ground-truth tangential slip.'};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const args=process.argv.slice(2);assert(args.length===1||args.length===2);
 console.log(JSON.stringify(args.length===1?extractFinalized(args[0]):compareFinalized(args[0],args[1]),null,2));
}

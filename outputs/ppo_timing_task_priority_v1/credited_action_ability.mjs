// Read-only credited-decision endpoint statistics. Prints JSON; never writes run data.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
const ORDER=['front_left_hip','front_left_knee','front_right_hip','front_right_knee',
 'rear_left_hip','rear_left_knee','rear_right_hip','rear_right_knee',
 'front_left_ankle','front_right_ankle','rear_left_ankle','rear_right_ankle'];
const stat=()=>({observed_count:Array(12).fill(0),nonzero_count:Array(12).fill(0),
 positive_count:Array(12).fill(0),negative_count:Array(12).fill(0),
 min:Array(12).fill(null),max:Array(12).fill(null)});
function add(s,v) {
 for(let i=0;i<12;i++) if(typeof v?.[i]==='number'&&Number.isFinite(v[i])) {
  const x=v[i];s.observed_count[i]++;s.nonzero_count[i]+=Number(x!==0);
  s.positive_count[i]+=Number(x>0);s.negative_count[i]+=Number(x<0);
  s.min[i]=s.min[i]===null?x:Math.min(s.min[i],x);
  s.max[i]=s.max[i]===null?x:Math.max(s.max[i],x);
 }
}
function group() {
 return {credited_decisions:0,terminal_decisions:0,returned_physics_ticks:0,
  endpoint_effect_verified_count:0,endpoint_request_matches_start_phase_count:0,
  endpoint_source_phase_differs_count:0,endpoint_effect_missing_or_unverified_count:0,
  physics_tick_audit_summary:{physics_ticks:0,verified_tick_count:0,
   actual_native_effect_tick_count:0,own_phase_request_effect_tick_count:0,
   all_ticks_verified_decisions:0},
  raw_policy_latent:stat(),projected_residual_deg_or_rad_s:stat(),
  effective_post_headroom_residual_deg_or_rad_s:stat(),
  policy_same_prestate_native_effect_rad_or_rad_s:stat(),
  policy_same_prestate_canonical_target_effect_deg_or_rad_s:stat(),
  nominal_geometry_native_effect_rad_or_rad_s:stat(),
  geometry_adjustment_before_final_slew_deg_or_rad_s:stat(),
  headroom:{observed_decisions:0,clipped_count_by_servo:Array(8).fill(0),
   baseline_outside_reserve_count_by_servo:Array(8).fill(0),
   interval_lower_min_deg:Array(8).fill(null),interval_upper_max_deg:Array(8).fill(null)},
  mask:{observed_decisions:0,zero_count_by_channel:Array(12).fill(0)},
  explicit_saturation_or_ratelimit_flags:null};
}
function nativeArray(x) {
 if(!Array.isArray(x?.servo_position_rad)||x.servo_position_rad.length!==8||
    !Array.isArray(x?.wheel_velocity_rad_s)||x.wheel_velocity_rad_s.length!==4)return null;
 return [...x.servo_position_rad,...x.wheel_velocity_rad_s];
}
function readSigns(project) {
 const source=path.join(project,'src/wlr50_clean/infrastructure/command_batch.py');
 const data=fs.readFileSync(source,'utf8');
 const get=(name,names)=>{
  const body=data.match(new RegExp(name+':[^\\n]+MappingProxyType\\(\\{([\\s\\S]*?)\\}\\)'))?.[1];
  if(!body)throw Error('Missing physical sign declaration: '+name);
  return names.map(n=>{
   const v=body.match(new RegExp('"'+n+'":\\s*(-?1\\.0)'))?.[1];
   if(v===undefined)throw Error('Unknown physical sign: '+n);return Number(v);
  });
 };
 return {servo:get('SERVO_COMMAND_SIGN',ORDER.slice(0,8)),
  wheel:get('WHEEL_FORWARD_SIGN',ORDER.slice(8)),
  source,sha256:crypto.createHash('sha256').update(data).digest('hex')};
}
function feed(g,r,signs) {
 const a=r.applied_audit,e=a.actuator_target_effect_audit;
 g.credited_decisions++;g.terminal_decisions+=Number(r.terminal===true);
 g.returned_physics_ticks+=a.physics_ticks??0;
 add(g.raw_policy_latent,r.raw_policy_action_full12);
 add(g.projected_residual_deg_or_rad_s,a.projected_residual_full12);
 const summary=a.actuator_target_effect_audit_summary;
 if(summary) {
  for(const k of ['physics_ticks','verified_tick_count','actual_native_effect_tick_count','own_phase_request_effect_tick_count'])
   g.physics_tick_audit_summary[k]+=summary[k]??0;
  g.physics_tick_audit_summary.all_ticks_verified_decisions+=Number(summary.all_ticks_verified===true);
 }
 const verified=e?.verified===true&&e.same_tick_counterfactual===true&&
  e.setter_dispatch_targets_equal===true&&e.actual_mapping_matches_dispatch===true&&
  e.counterfactual_scope==='same_pre_tick_state_without_current_ppo_residual';
 if(!verified){g.endpoint_effect_missing_or_unverified_count++;return;}
 if(JSON.stringify(e.canonical_order)!==JSON.stringify(ORDER))throw Error('Unexpected native channel order');
 g.endpoint_effect_verified_count++;
 g.endpoint_request_matches_start_phase_count+=Number(e.policy_request_phase===a.phase_id);
 g.endpoint_source_phase_differs_count+=Number(e.source_phase_id!==a.phase_id);
 const delta=nativeArray(e.native_target_delta);
 add(g.policy_same_prestate_native_effect_rad_or_rad_s,delta);
 add(g.policy_same_prestate_canonical_target_effect_deg_or_rad_s,delta?.map((x,i)=>
  i<8?x*180/Math.PI*signs.servo[i]:x/signs.wheel[i-8]));
 add(g.nominal_geometry_native_effect_rad_or_rad_s,nativeArray(e.nominal_geometry_native_target_delta));
 add(g.geometry_adjustment_before_final_slew_deg_or_rad_s,e.nominal_geometry_adjustment_full12);
 const h=e.policy_headroom_evidence;
 if(h) {
  g.headroom.observed_decisions++;
  add(g.effective_post_headroom_residual_deg_or_rad_s,h.effective_policy_residual_full12);
  for(const i of h.clipped_servo_indices??[])g.headroom.clipped_count_by_servo[i]++;
  for(const i of h.baseline_outside_reserved_servo_indices??[])g.headroom.baseline_outside_reserve_count_by_servo[i]++;
  for(let i=0;i<8;i++) {
   const pair=h.policy_residual_intervals_servo_deg?.[i];
   if(pair?.length!==2)continue;
   g.headroom.interval_lower_min_deg[i]=g.headroom.interval_lower_min_deg[i]===null?pair[0]:Math.min(g.headroom.interval_lower_min_deg[i],pair[0]);
   g.headroom.interval_upper_max_deg[i]=g.headroom.interval_upper_max_deg[i]===null?pair[1]:Math.max(g.headroom.interval_upper_max_deg[i],pair[1]);
  }
 }
 if(Array.isArray(e.phase_mask_full12)&&e.phase_mask_full12.length===12){
  g.mask.observed_decisions++;e.phase_mask_full12.forEach((v,i)=>g.mask.zero_count_by_channel[i]+=Number(v===0));
 }
}
export async function aggregate(runPath) {
 const run=path.resolve(runPath),manifestPath=path.join(run,'training_manifest.json');
 const m=JSON.parse(fs.readFileSync(manifestPath,'utf8'));
 if(!['SUCCEEDED','STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'].includes(m.lifecycle))throw Error('Only finalized complete-update training manifests are accepted; this says nothing about task success.');
 if(m.lifecycle==='STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'&&!m.stop_after_update)throw Error('Missing verified boundary stop receipt');
 if(!Number.isInteger(m.actual_policy_decisions)||!Number.isInteger(m.global_policy_decisions))throw Error('Missing completed credit counters');
 const project=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..'),signs=readSigns(project);
 const first=m.global_policy_decisions-m.actual_policy_decisions+1;
 const total=group(),phases={},sourceCounts={};let count=0;
 const input=path.join(run,'residual_and_projection_audit.jsonl');
 const lines=readline.createInterface({input:fs.createReadStream(input),crlfDelay:Infinity});
 for await(const line of lines) {
  if(!line.trim())continue;const r=JSON.parse(line),a=r.applied_audit;
  if(r.global_policy_decision!==first+count)throw Error('Noncontiguous or uncredited policy row');
  if(!/^P(0[1-9]|1[0-3])$/.test(a?.phase_id))throw Error('Unknown request phase');
  if(JSON.stringify(r.raw_policy_action_full12)!==JSON.stringify(a.raw_policy_action_full12))throw Error('Stored/applied raw mismatch');
  count++;phases[a.phase_id]??=group();feed(total,r,signs);feed(phases[a.phase_id],r,signs);
  const source=a.actuator_target_effect_audit?.source_phase_id??'UNKNOWN';sourceCounts[source]=(sourceCounts[source]??0)+1;
 }
 if(count!==m.actual_policy_decisions)throw Error('Credited row count differs from finalized manifest');
 for(const [p,n] of Object.entries(m.telemetry?.core?.phase_decisions??{}))
  if((phases[p]?.credited_decisions??0)!==n)throw Error('Phase count differs from manifest: '+p);
 const rr={};
 for(const [p,g] of Object.entries({ALL:total,...phases})){
  const pick=s=>({observed:s.observed_count[7],positive:s.positive_count[7],negative:s.negative_count[7],
   nonzero:s.nonzero_count[7],min:s.min[7],max:s.max[7]});
  rr[p]={raw:pick(g.raw_policy_latent),projected:pick(g.projected_residual_deg_or_rad_s),
   effective_post_headroom:pick(g.effective_post_headroom_residual_deg_or_rad_s),
   actual_dispatched_target_effect_canonical_deg:pick(g.policy_same_prestate_canonical_target_effect_deg_or_rad_s)};
 }
 return {schema:'wlr50_clean.credited_policy_control_authority_summary.v1',
  sources:{run,manifest:manifestPath,credited_audit:input,physical_coordinate_mapping:signs},
  scope:{teacher_prefix_excluded:true,credit_source:'Only residual_and_projection_audit rows with contiguous global_policy_decision, checked against finalized training manifest; no prefix/native stream is read.',
   grouping:'Policy request start phase applied_audit.phase_id; cross-phase endpoint source is counted separately.',
   endpoint_scope:'Raw is one sampled raw per credited decision. Projected/effective/native effect are the final audited dispatch endpoint of that decision, not all eight 120Hz channel arrays.',
   effect_scope:'Actual float32 dispatched actuator target minus same-pre-tick zero-current-policy target, sharing N/geometry/controller/history. Not a separate zero-residual trajectory and not measured q motion.',
   geometry_scope:'Reported separately only when explicit nominal_geometry_* fields exist. Zero observed_count means absent metadata, never filled with invented zero.',
   nonzero_rule:'Exact numeric nonzero (no invented significance threshold). Native target values are float32 readback differences.',
   saturation_scope:'Only explicit headroom clipped/baseline-outside metadata counted. No projection cap/rate-limit saturation flags exist in inspected decision schema; null, not inferred from magnitude.',
   physics_tick_summaries:'Existing all-tick summary counts may cover full interval, but do not expose per-channel signed deltas for interior ticks.',
   no_production_changes:true,no_policy_load:true,no_probe:true,no_active_run_read:true},
  manifest:{lifecycle:m.lifecycle,stage:m.stage,implemented_sampling:m.implemented_sampling,
   curriculum_epoch:m.curriculum_epoch,actual_policy_decisions:m.actual_policy_decisions,
   global_policy_decisions:m.global_policy_decisions,ppo_updates_this_run:m.ppo_updates_this_run,
   optimizer_steps_this_run:m.optimizer_steps_this_run??null,
   prefix_behavior_decisions_excluded:m.telemetry?.core?.prefix_behavior_decisions??0,
   prefix_physics_ticks_excluded:m.telemetry?.core?.prefix_physics_ticks??0,
   checkpoint:m.checkpoints?.at(-1)??null,phase_decisions:m.telemetry?.core?.phase_decisions??null,
   terminations:m.telemetry?.core?.terminations??null,success_count:m.telemetry?.success_count??null},
  canonical_order:ORDER,endpoint_source_phase_counts:sourceCounts,total,by_request_phase:phases,RR_knee_focus:rr};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 if(process.argv.length!==3)throw Error('Usage: node credited_action_ability.mjs FINALIZED_RUN_DIR');
 console.log(JSON.stringify(await aggregate(process.argv[2])));
}

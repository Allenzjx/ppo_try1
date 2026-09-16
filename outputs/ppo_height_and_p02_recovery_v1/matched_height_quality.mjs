// Completed physical/height JSONL only; source full-window quality is retained separately.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {calibratedEuler} from '../ppo_timing_task_priority_v1/stability_metrics.mjs';
const here=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(here,'../..');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,''));
const finite=x=>Number.isFinite(x)?x:null;
const difference=(a,b)=>Number.isFinite(a)&&Number.isFinite(b)?a-b:null;
async function lines(file,consume){const r=readline.createInterface({input:fs.createReadStream(file),crlfDelay:Infinity});let n=0;
 for await(const line of r)if(line.trim())consume(JSON.parse(line.replace(/^\uFEFF/,'')),++n);return n;}
function stat(){return{samples:0,sum_squares:0,min:null,min_tick:null,max:null,max_tick:null};}
function add(s,x,t){if(!Number.isFinite(x))return;s.samples++;s.sum_squares+=x*x;
 if(s.min===null||x<s.min){s.min=x;s.min_tick=t;}if(s.max===null||x>s.max){s.max=x;s.max_tick=t;}}
function finish(s){return{valid_samples:s.samples,rms:s.samples?Math.sqrt(s.sum_squares/s.samples):null,min:s.min,min_tick:s.min_tick,max:s.max,max_tick:s.max_tick};}
function window(end){return{end,physical_rows:0,height_rows:0,metrics:{},end_physical:null,end_height:null};}
function apply(w,t,values,type){if(t>w.end)return;w[type+'_rows']++;for(const[k,v]of Object.entries(values)){w.metrics[k]??=stat();add(w.metrics[k],v,t);}
 if(t===w.end)w['end_'+type]={physics_tick:t,...values};}
function completed(argument,role){const source=fs.realpathSync(path.resolve(root,argument));assert.equal(path.basename(source),'source');
 const final=read(path.join(source,'semantic_video_source_manifest.json')),run=read(path.join(source,'../run_manifest.json'));
 assert(run.completed_at_utc&&['SUCCEEDED','DIAGNOSTIC_FAILURE'].includes(run.lifecycle));assert.equal(final.role,role);
 assert.equal(final.optimizer_updates,0);assert(final.diagnostic_intervention==null);assert.deepEqual(final.runtime_contract,run.runtime_contract);
 const selected=final.runtime_contract.selected_configuration['observation_schema.json'],file=path.resolve(root,selected.path),bytes=fs.readFileSync(file);
 assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),selected.sha256);
 const calibration=JSON.parse(bytes.toString('utf8').replace(/^\uFEFF/,''));assert.equal(calibration.fixed_chassis_to_body_wxyz.length,4);
 const end=final.episode_physics_ticks;assert(Number.isInteger(end)&&end>0&&end<=24000);
 assert.equal(final.runtime_contract.physics_hz,120);assert(final.height_diagnostics.stream_closed===true);
 return{source,final,run,end,calibration,calibration_path:file,calibration_sha256:selected.sha256};}
async function measure(r,commonEnd){const full=window(r.end),common=window(commonEnd);let last=-1,priorRPY=null,priorOmega=null;
 await lines(path.join(r.source,'physical_observations.jsonl'),p=>{const t=p.physics_tick;assert.equal(t,last+1);assert(t<=r.end);
  assert(Math.abs(p.simulation_time_s-t/120)<1e-8);const q=calibratedEuler(p.base.orientation_wxyz,r.calibration.fixed_chassis_to_body_wxyz);
  const omega=p.base.angular_velocity_w_rad_s,validOmega=Array.isArray(omega)&&omega.length===3&&omega.every(Number.isFinite);
  const rate=q&&priorRPY?q.slice(0,2).map((v,i)=>Math.atan2(Math.sin(v-priorRPY[i]),Math.cos(v-priorRPY[i]))*120):null;
  const v={calibrated_roll_rad:q?.[0]??null,calibrated_pitch_rad:q?.[1]??null,derived_Euler_roll_rate_rad_s:rate?.[0]??null,
   derived_Euler_pitch_rate_rad_s:rate?.[1]??null,measured_world_angular_speed_rad_s:validOmega?Math.hypot(...omega):null,
   derived_world_angular_acceleration_rad_s2:validOmega&&priorOmega?Math.hypot(...omega.map((v,i)=>(v-priorOmega[i])*120)):null,
   base_origin_world_z_m:finite(p.base.position_w_m?.[2]),valid_CoM_world_z_m:p.center_of_mass?.valid===true?finite(p.center_of_mass.position_w_m?.[2]):null,
   recorded_body_collider_min_world_z_m:finite(p.body_bounds_w_m?.base_link?.minimum_m?.[2]),
   RR_center_minus_obstacle_front_m:difference(p.wheels?.rear_right_ankle?.center_w_m?.[0],p.obstacle?.front_x_m),
   RR_bottom_minus_obstacle_top_m:difference(p.wheels?.rear_right_ankle?.bottom_w_m?.[2],p.obstacle?.top_z_m),
   RR_knee_actual_minus_final_deg:difference(p.joints?.rear_right_knee?.position_deg,p.joints?.rear_right_knee?.command_deg),
   RR_hip_actual_minus_final_deg:difference(p.joints?.rear_right_hip?.position_deg,p.joints?.rear_right_hip?.command_deg)};
  apply(full,t,v,'physical');apply(common,t,v,'physical');last=t;priorRPY=q;priorOmega=validOmega?omega:null;});
 assert.equal(last,r.end);assert.equal(full.physical_rows,r.end+1);assert.equal(common.physical_rows,commonEnd+1);
 const expected=[...Array(Math.floor(r.end/8)+1)].map((_,i)=>8*i);if(r.end%8)expected.push(r.end);let index=0;
 await lines(path.join(r.source,'height_diagnostics.jsonl'),h=>{const t=h.physics_tick;assert.equal(t,expected[index++]);assert.equal(h.clock_unchanged,true);
  const v={independent_body_collider_min_world_z_m:finite(h.body_collision_minimum_z_w_m),
   actual_USD_RR_mount_world_z_m:finite(h.rr_hip_mount_w_m?.value?.[2]),
   independent_RR_wheel_bottom_world_z_m:finite(h.fresh_collider_bounds?.rear_right_wheel?.value?.minimum_m?.[2])};
  apply(full,t,v,'height');apply(common,t,v,'height');});assert.equal(index,expected.length);
 function result(w){return{inclusive_physical_tick_range:[0,w.end],duration_s:w.end/120,physical_observation_count:w.physical_rows,
  exact_height_sample_count:w.height_rows,metrics:Object.fromEntries(Object.entries(w.metrics).map(([k,v])=>[k,finish(v)])),
  endpoint_physical:w.end_physical,endpoint_exact_height:w.end_height};}
 return{role:r.final.role,source:r.source,source_git_commit:r.final.runtime_contract.source_git_commit,runtime_content_sha256:r.final.runtime_contract.runtime_content_sha256,
  seed:r.final.seed,task_success:r.final.physical_task_success,physical_evaluator_termination_reason:r.final.physical_episode.physical_task_evaluation.termination_reason,
  physical_evaluator_termination_source:r.final.physical_episode.physical_task_evaluation.termination_source,
  calibration:{path:r.calibration_path,sha256:r.calibration_sha256,fixed_chassis_to_body_wxyz:r.calibration.fixed_chassis_to_body_wxyz},
  source_quality_metrics_full_window_unchanged:r.final.physical_episode.quality_metrics,
  full_physical_window_quality:result(full),matched_common_prefix_quality:result(common)};}
async function main(){const a=process.argv.slice(2),o={};for(let i=0;i<a.length;i+=2){assert(['--b-source','--c-source','--output'].includes(a[i])&&a[i+1]);assert(!(a[i]in o));o[a[i]]=a[i+1];}
 assert.equal(Object.keys(o).length,3);const output=path.resolve(root,o['--output']),relative=path.relative(here,output);assert(relative&&!relative.startsWith('..')&&!path.isAbsolute(relative)&&!fs.existsSync(output));
 const b=completed(o['--b-source'],'B'),c=completed(o['--c-source'],'C');assert.deepEqual(b.final.runtime_contract,c.final.runtime_contract);
 assert.deepEqual(b.final.evaluation_configuration,c.final.evaluation_configuration);assert.equal(b.final.seed,c.final.seed);
 const commonEnd=Math.min(b.end,c.end),runs=[];for(const r of[b,c])runs.push(await measure(r,commonEnd));
 const payload={schema:'wlr50_clean.matched_height_quality.v1',same_recorded_runtime:true,matched_common_prefix_ticks:[0,commonEnd],runs,
  definitions:{physical:'Real inclusive 120Hz observations, no video padding or action/force forward fill.',height:'Independent current USD/collider diagnostics at exact tick0/every8/terminal only; per-metric counts show this lower rate; no interpolation.',
   statistics:'RMS=sqrt(mean(square)) of finite recorded/derived values; min/max retain actual ticks; null is unknown.',
   attitude:'Reused calibratedEuler: normalized(q_world_body * fixed_chassis_to_body), ZYX radians. Euler rates are shortest-angle differences, not measured angular velocity.',
   geometry:'Base origin is NOT body clearance. RR front is wheel link-center x minus obstacle front, gap is recorded wheel bottom z minus obstacle top; mount is actual USD localPos0 with live parenttransform.',
   source_score:'Untouched source quality_metrics describes its own full physical window and scoring convention, not the common-prefix window.'},
  no_stability_superiority_claim:true,interpretation:'Common-prefix and unequal complete windows are separate. Lower motion/score can coexist with collapse, safety abort or incomplete traversal. Frozen video end frames never enter these metrics; runtime equality is not itself initial-state proof.'};
 fs.writeFileSync(output,JSON.stringify(payload,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({output,matched_end_tick:commonEnd,rows:runs.map(r=>({role:r.role,end:r.full_physical_window_quality.inclusive_physical_tick_range[1]}))}));}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))main().catch(e=>{console.error(e.stack);process.exitCode=1;});

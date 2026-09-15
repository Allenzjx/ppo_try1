// Read-only physical metrics; one new output JSON, no Python/Isaac/CSV/native-audit reread.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {fileURLToPath} from 'node:url';

const project=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const resolve=p=>path.resolve(project,p);
const read=p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,''));
const require=(test,msg)=>{if(!test)throw Error(msg);};
const vector=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
const finite=v=>Number.isFinite(v)?v:null;
const pick=(o,keys)=>Object.fromEntries(keys.map(k=>[k,o?.[k]??null]));
const normalize=q=>vector(q,4)&&Math.hypot(...q)>1e-12?q.map(v=>v/Math.hypot(...q)):null;

export function euler(q) {
  const n=normalize(q);if(!n)return null;
  const[w,x,y,z]=n;
  return [Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
    Math.asin(Math.max(-1,Math.min(1,2*(w*y-z*x)))),Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))];
}
export function calibratedEuler(q,fixed) {
  const n=normalize(q),f=normalize(fixed);if(!n||!f)return null;
  const[w,x,y,z]=n,[a,b,c,d]=f;
  return euler([w*a-x*b-y*c-z*d,w*b+x*a+y*d-z*c,w*c-x*d+y*a+z*b,w*d+x*c-y*b+z*a]);
}
class Metric {
  constructor(){this.n=0;this.sq=0;this.max=null;this.peakTick=null;this.signedPeak=null;this.min=null;this.minTick=null;}
  add(v,tick){if(!Number.isFinite(v))return;this.n++;this.sq+=v*v;
    if(this.max===null||Math.abs(v)>this.max){this.max=Math.abs(v);this.peakTick=tick;this.signedPeak=v;}
    if(this.min===null||v<this.min){this.min=v;this.minTick=tick;}}
  rms(){return{valid_samples:this.n,rms:this.n?Math.sqrt(this.sq/this.n):null,max_abs:this.max,max_abs_tick:this.peakTick};}
  minimum(){return{valid_samples:this.n,min:this.min,min_tick:this.minTick};}
  peak(){return{valid_samples:this.n,max_abs:this.max,signed_error_at_peak:this.signedPeak,physics_tick:this.peakTick};}
}
function accumulator(end) {
  const fields=['roll','pitch','rollRate','pitchRate','omegaX','omegaY','omegaZ','omegaNorm','baseZ','comZ','rrHip','rrKnee'];
  return {end,rows:0,nonfiniteFlagRows:0,dataQualityRows:0,nearEulerSingularityRows:0,
    metrics:Object.fromEntries(fields.map(k=>[k,new Metric()])),start:null,last:null};
}
function add(acc,p,rpy,rates) {
  const tick=p.physics_tick;if(tick>acc.end)return;
  const m=acc.metrics;acc.rows++;
  if(p.all_finite!==true)acc.nonfiniteFlagRows++;
  if(Array.isArray(p.data_quality)&&p.data_quality.length)acc.dataQualityRows++;
  if(rpy&&Math.abs(Math.cos(rpy[1]))<.1)acc.nearEulerSingularityRows++;
  m.roll.add(rpy?.[0],tick);m.pitch.add(rpy?.[1],tick);
  m.rollRate.add(rates?.[0],tick);m.pitchRate.add(rates?.[1],tick);
  const omega=p.base.angular_velocity_w_rad_s;
  if(vector(omega,3)){for(const[i,k]of['omegaX','omegaY','omegaZ'].entries())m[k].add(omega[i],tick);m.omegaNorm.add(Math.hypot(...omega),tick);}
  const baseZ=finite(p.base.position_w_m?.[2]);
  const comZ=p.center_of_mass?.valid===true?finite(p.center_of_mass.position_w_m?.[2]):null;
  m.baseZ.add(baseZ,tick);m.comZ.add(comZ,tick);
  for(const[joint,key]of [['hip','rrHip'],['knee','rrKnee']]){
    const j=p.joints[`rear_right_${joint}`];
    const value=Number.isFinite(j?.position_deg)&&Number.isFinite(j?.command_deg)?j.position_deg-j.command_deg:null;
    m[key].add(value,tick);
  }
  const point={physics_tick:tick,simulation_time_s:p.simulation_time_s,
    calibrated_roll_pitch_rad:rpy?.slice(0,2)??null,base_origin_z_world_m:baseZ,valid_CoM_z_world_m:comZ};
  if(acc.start===null)acc.start=point;acc.last=point;
}
function finish(a){const m=a.metrics;return{
  first_tick:0,last_tick:a.end,duration_s:a.end/120,physical_sample_count:a.rows,
  calibrated_body_roll_pitch_rad:{roll:m.roll.rms(),pitch:m.pitch.rms()},
  derived_calibrated_Euler_rate_rad_s:{roll:m.rollRate.rms(),pitch:m.pitchRate.rms()},
  measured_body_angular_velocity_world_rad_s:{x:m.omegaX.rms(),y:m.omegaY.rms(),z:m.omegaZ.rms(),norm:m.omegaNorm.rms()},
  minimum_height_world_m:{base_origin:m.baseZ.minimum(),valid_mass_weighted_CoM:m.comZ.minimum()},
  RR_actual_minus_final_command_deg:{hip:m.rrHip.peak(),knee:m.rrKnee.peak()},
  start:a.start,end:a.last,quality:{all_finite_flag_not_true_rows:a.nonfiniteFlagRows,
    data_quality_nonempty_rows:a.dataQualityRows,near_Euler_gimbal_singularity_rows:a.nearEulerSingularityRows}};}

function loadCompleted(runArgument,summaryArgument,role) {
  let run=fs.realpathSync(resolve(runArgument));if(path.basename(run)==='source')run=path.dirname(run);
  const manifest=read(path.join(run,'run_manifest.json')),source=path.join(run,'source');
  require(Boolean(manifest.completed_at_utc)&&['SUCCEEDED','DIAGNOSTIC_FAILURE'].includes(manifest.lifecycle),
    'Only explicitly completed evaluations are accepted');
  const final=read(path.join(source,'semantic_video_source_manifest.json'));
  require(final.role===role&&final.optimizer_updates===0&&manifest.optimizer_updates===0,
    'Expected a completed zero-update B/C evaluation');
  const end=final.episode_physics_ticks;
  require(Number.isInteger(end)&&end>0&&end<=24000,'Invalid completed <=200s physical endpoint');
  require(final.runtime_contract.runtime_content_sha256===manifest.runtime_contract.runtime_content_sha256,
    'Source/run runtime digest mismatch');
  const summaryPath=fs.realpathSync(resolve(summaryArgument)),summary=read(summaryPath);
  require(fs.realpathSync(summary.source)===source&&summary.end_tick===end
    &&summary.runtime_content_sha256===manifest.runtime_contract.runtime_content_sha256
    &&summary.source_git_commit===manifest.runtime_contract.source_git_commit
    &&summary.exact_full_stream_join_complete===true&&summary.native_ticks===end
    &&summary.physical_rows===end+1&&summary.native_verified_ticks===end,
    'Completed full-rate native/physical summary is not bound to this run');
  const calibration=summary.attitude_calibration;
  const selected=manifest.runtime_contract.selected_configuration?.['observation_schema.json'];
  require(calibration?.status==='recorded_configuration_sha256_verified'&&selected
    &&calibration.sha256===selected.sha256&&path.resolve(calibration.path)===resolve(selected.path)
    &&normalize(calibration.fixed_chassis_to_body_wxyz),'Missing or mismatched recorded calibration; do not assume identity');
  return{run,source,manifest,final,end,summary,summaryPath,calibration};
}

async function measure(run,commonEnd) {
  const full=accumulator(run.end),common=accumulator(commonEnd);let previousRpy=null,previousTick=-1;
  const lines=readline.createInterface({input:fs.createReadStream(path.join(run.source,'physical_observations.jsonl')),crlfDelay:Infinity});
  for await(const line of lines){if(!line.trim())continue;const p=JSON.parse(line.replace(/^\uFEFF/,''));
    const tick=p.physics_tick;
    require(Number.isInteger(tick)&&tick===previousTick+1&&tick<=run.end
      &&Number.isFinite(p.simulation_time_s)&&Math.abs(p.simulation_time_s-tick/120)<1e-8
      &&p.base&&p.joints,'Physical schema/clock gap, duplicate or post-terminal row');
    const rpy=calibratedEuler(p.base.orientation_wxyz,run.calibration.fixed_chassis_to_body_wxyz);
    const rates=rpy&&previousRpy?rpy.slice(0,2).map((v,i)=>Math.atan2(Math.sin(v-previousRpy[i]),Math.cos(v-previousRpy[i]))*120):null;
    add(full,p,rpy,rates);add(common,p,rpy,rates);previousRpy=rpy;previousTick=tick;
  }
  require(previousTick===run.end&&full.rows===run.end+1&&common.rows===commonEnd+1,'Incomplete physical stream');
  const fullResult=finish(full);
  for(const joint of ['hip','knee']){
    const a=fullResult.RR_actual_minus_final_command_deg[joint],b=run.summary.RR_tracking_peaks_over_full_run?.[joint];
    require(b&&a.physics_tick===b.physics_tick&&Math.abs(a.signed_error_at_peak-b.signed_error_deg)<1e-8,
      `RR ${joint} full physical tracking maximum disagrees with completed native/physical extractor`);
  }
  const final=run.final,evaluation=final.physical_episode?.physical_task_evaluation??{};
  return{role:final.role,run:run.run,source:run.source,source_git_commit:run.manifest.runtime_contract.source_git_commit,
    runtime_content_sha256:run.manifest.runtime_contract.runtime_content_sha256,
    calibration:{...run.calibration,verification:'Reused existing SHA-verified completed extractor calibration, bound to recorded run configuration'},
    native_evidence:{summary:run.summaryPath,native_ticks:run.summary.native_ticks,
      verified_ticks:run.summary.native_verified_ticks,full_rate_RR_peaks_recomputed_and_matched:true,
      native_stream_reread:false},
    task:{lifecycle:run.manifest.lifecycle,physical_task_success:final.physical_task_success,
      ...pick(evaluation,['success','task_completed_controlled','traversal_task_complete','termination_reason','termination_source','physics_tick','simulation_time_s']),
      last_phase_from_recorded_transition:run.summary.stage_transitions?.at(-1)?.to_stage??final.from_phase,
      source_acceptance_error:final.source_acceptance_error??null,
      checkpoint_decisions:run.summary.checkpoint_decisions,optimizer_updates:0,
      issued_policy_decisions:final.issued_policy_decisions,returned_environment_steps:final.completed_environment_steps,
      interrupted_final_decision_ticks:final.interrupted_final_decision_ticks},
    full_episode:fullResult,matched_common_window:finish(common)};
}

async function main(){const args=process.argv.slice(2),o={};for(let i=0;i<args.length;i+=2){
  require(['--b-run','--c-run','--b-summary','--c-summary','--output'].includes(args[i])&&args[i+1],
    'Usage: node stability_metrics.mjs --b-run RUN --c-run RUN --b-summary SUMMARY --c-summary SUMMARY --output NEW_JSON');
  require(!(args[i]in o),'Duplicate argument');o[args[i]]=args[i+1];}
  require(Object.keys(o).length===5,'All five options are required');
  const dest=resolve(o['--output']),allowed=resolve('outputs/ppo_timing_task_priority_v1'),rel=path.relative(allowed,dest);
  require(rel&&!rel.startsWith('..')&&!path.isAbsolute(rel)&&dest.endsWith('.json')&&!fs.existsSync(dest),
    'Output must be a new JSON under outputs/ppo_timing_task_priority_v1');
  const b=loadCompleted(o['--b-run'],o['--b-summary'],'B'),c=loadCompleted(o['--c-run'],o['--c-summary'],'C');
  const commonEnd=Math.min(b.end,c.end),runs=[];
  for(const run of[b,c])runs.push(await measure(run,commonEnd));
  const report={schema:'wlr50_clean.paired_physical_stability_metrics.v1',created_at_utc:new Date().toISOString(),
    scope:'Explicit completed runs only; full 120Hz physical stream plus reused, bound full-rate native/physical summary; no new native audit read or policy sampling',
    same_recorded_runtime:b.manifest.runtime_contract.runtime_content_sha256===c.manifest.runtime_contract.runtime_content_sha256,
    matched_window:{inclusive_tick_range:[0,commonEnd],duration_s:commonEnd/120,samples_per_run:commonEnd+1},
    definitions:{RMS:'sqrt(mean(value squared)) over valid inclusive 120Hz observations, with per-metric sample counts',
      attitude:'Derived calibrated ZYX roll/pitch from normalized(q_world_body * fixed_chassis_to_body); radians, not raw quaternion components',
      angle_rate:'Derived shortest-angle first difference of consecutive calibrated Euler roll/pitch times 120; ticks1..end; not measured angular velocity',
      measured_angular_velocity:'Recorded base.angular_velocity_w_rad_s in WORLD axes; x/y are not Euler roll/pitch rates',
      height:'Recorded base origin or valid mass-weighted CoM world Z; base-origin Z is not body-mesh obstacle clearance',
      tracking:'Measured RR joint position_deg minus final command_deg, same physical row; derived error, not torque or a causal attribution',
      sample_scope:'Physics observations, not PPO decisions; no invented terminal reward or 15Hz forward fill'},
    interpretation:'No stability superiority claim: shorter, more horizontal or lower-variance motion can coexist with body/CoM collapse, safety abort or unfinished tasks. Compare common time first, retain both complete endpoints and task outcomes; equal runtime alone does not prove identical initial state.',
    runs};
  fs.writeFileSync(dest,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({output:dest,matched_end_tick:commonEnd,roles:runs.map(r=>({role:r.role,end_tick:r.full_episode.last_tick,
    task_success:r.task.physical_task_success,termination:r.task.termination_reason,source:r.task.termination_source,
    matched_roll_rms_rad:r.matched_common_window.calibrated_body_roll_pitch_rad.roll.rms,
    matched_pitch_rms_rad:r.matched_common_window.calibrated_body_roll_pitch_rad.pitch.rms,
    matched_CoM_min_m:r.matched_common_window.minimum_height_world_m.valid_mass_weighted_CoM.min}))}));
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))
  main().catch(e=>{console.error(e.stack??String(e));process.exitCode=1;});

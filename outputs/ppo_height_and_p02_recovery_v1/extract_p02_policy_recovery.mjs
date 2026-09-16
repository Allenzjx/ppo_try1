// Explicit completed formal-C source only. Old completed C is permitted only as a labelled fixture.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {firstRemainingTask} from '../ppo_timing_task_priority_v1/build_latest_results.mjs';
const here=path.dirname(fileURLToPath(import.meta.url)),project=path.resolve(here,'../..');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,''));
const real=p=>fs.realpathSync(path.resolve(project,p));
const vec=(v,n)=>Array.isArray(v)&&v.length===n&&v.every(Number.isFinite);
const pick=(o,keys)=>Object.fromEntries(keys.map(k=>[k,o?.[k]??null]));
const writeKeys=['in_episode_root_pose_writes','in_episode_root_velocity_writes','in_episode_force_or_impulse_writes','in_episode_gravity_writes'];
async function rows(file,consume){let lineNumber=0;
 const input=readline.createInterface({input:fs.createReadStream(file),crlfDelay:Infinity});
 for await(const line of input)if(line.trim())consume(JSON.parse(line.replace(/^\uFEFF/,'')),++lineNumber);
 return lineNumber;
}
function metric(){return{samples:0,min:null,min_tick:null,max:null,max_tick:null,max_abs:null,max_abs_tick:null,signed_at_max_abs:null};}
function add(m,x,t){if(!Number.isFinite(x))return;m.samples++;
 if(m.min===null||x<m.min){m.min=x;m.min_tick=t;}if(m.max===null||x>m.max){m.max=x;m.max_tick=t;}
 if(m.max_abs===null||Math.abs(x)>m.max_abs){m.max_abs=Math.abs(x);m.max_abs_tick=t;m.signed_at_max_abs=x;}}

export async function extract({source:sourceArg,checkpoint:checkpointArg,expectedHead,purpose}){
 assert(['fixture','formal'].includes(purpose));
 const source=real(sourceArg),runDir=path.dirname(source);
 assert.equal(path.basename(source),'source');assert.equal(path.basename(path.dirname(runDir)),'validation');
 const paths=Object.fromEntries(['run_manifest','semantic_video_source_manifest','physical_observations',
  'native_tick_audit','video_policy_decisions','stage_transition_evidence','viewport_frame_ledger'].map(name=>
   [name,name==='run_manifest'?path.join(runDir,name+'.json'):path.join(source,name+
     (name==='semantic_video_source_manifest'?'.json':'.jsonl'))]));
 const run=read(paths.run_manifest),final=read(paths.semantic_video_source_manifest);
 assert(['SUCCEEDED','DIAGNOSTIC_FAILURE'].includes(run.lifecycle)&&run.completed_at_utc,'Run is not completed');
 assert.equal(final.role,'C');assert.equal(final.mode,'semantic_residual_eval');
 assert.deepEqual(final.runtime_contract,run.runtime_contract);assert.equal(run.runtime_contract.source_git_commit,expectedHead);
 for(const o of [run,final]){assert.equal(o.optimizer_updates,0);assert(o.diagnostic_intervention==null);}
 const args=run.arguments;
 assert.equal(args.command,'eval');assert.equal(args.mode,'semantic_residual_eval');assert.equal(args.from_phase,'P01');
 assert.equal(args.num_envs,1);assert.equal(args.teacher_offset_decisions,0);assert.equal(args.new_mdp_warm_start,false);
 assert.equal(args.policy_distribution_migration,false);assert.equal(args.max_decisions,3000);assert.equal(args.decisions,null);
 if(purpose==='formal')assert('src/wlr50_clean/ppo/semantic_height_recovery.py' in final.runtime_contract.files,
  'Old C fixture cannot be labelled as this round formal evidence');
 const proof=final.checkpoint_load_provenance,cpPath=real(checkpointArg);
 assert.equal(proof?.checkpoint_loaded_and_verified,true);assert.equal(proof.official_load_semantic_checkpoint,true);
 assert.equal(proof.optimizer_updates,0);assert(proof.diagnostic_intervention==null);assert.equal(real(proof.source.checkpoint),cpPath);
 assert.equal(real(args.checkpoint),cpPath);const cpManifest=real(proof.source.manifest),cp=read(cpManifest);
 assert.equal(cpManifest,cpPath.replace(/\.pt$/,'_manifest.json'));assert.equal(cp.save_load_round_trip,true);
 assert.equal(real(cp.checkpoint_path),cpPath);assert.equal(proof.source.checkpoint_sha256,cp.checkpoint_sha256);
 assert.equal(proof.saved_global_policy_decisions,cp.global_policy_decisions);
 for(const k of ['actor_parameter_sha256','critic_parameter_sha256','optimizer_state_sha256','normalizer_state_sha256'])
  assert.equal(proof.parameter_hashes[k],cp[k]);
 assert.equal(proof.observation_dimension,372);assert.equal(proof.policy_contract.raw_action_dimension,12);
 assert.equal(proof.policy_contract.deterministic_output,'conditional_mean');assert.equal(proof.policy_contract.rho,.9);
 assert.deepEqual(proof.policy_contract,cp.policy_contract);
 const reset=final.natural_reset_proof,restoration=reset?.reset_metadata?.phase_snapshot_restoration;
 assert.equal(reset?.schema,'wlr50_clean.current_video_natural_reset.v1');assert.equal(reset.role,'C');
 assert.deepEqual(reset.entry,{decision_count:0,done:false,physics_tick:0,state_id:'P01'});
 assert.equal(reset.reset_metadata.execution_mode,'semantic_B_or_C');assert.equal(restoration?.mode,'semantic_natural_P01');
 assert.equal(restoration.requested_phase,'P01');assert.equal(restoration.historical_state_equality_required,false);
 assert.equal(restoration.policy_credit_excludes_settle,true);
 for(const k of ['snapshot_path','state_sha256','file_sha256','source_tick','physical_state'])assert(restoration[k]==null);
 for(const k of ['pre_action_ticks','extra_pre_action_physics_ticks','performed_post_success_ticks'])assert.equal(final[k],0);
 assert.equal(final.from_phase,'P01');assert.equal(final.episode_count,1);assert.equal(final.fresh_process_single_episode,true);
 for(const k of ['stitched','speed_modified','frame_interpolation'])assert.equal(final[k],false);
 const end=final.episode_physics_ticks,physical=final.physical_episode?.physical_task_evaluation;
 assert(Number.isInteger(end)&&end>0&&end<=24000);assert.equal(physical.physics_tick,end);
 const decisions=[],phaseCounts={};let returned=0,lastEnd=0;
 const p02Endpoints={samples:0,clearance_m:metric(),front_distance_m:metric(),current_air_samples:0,
  qualified_history_samples:0,scope:'Actual P02-request decision endpoints, not full-rate extrema or an automatic pass'};
 await rows(paths.video_policy_decisions,(r,line)=>{
  assert.equal(r.decision,decisions.length+1);assert.equal(r.start_tick,lastEnd);
  assert(Number.isInteger(r.end_tick)&&r.end_tick>r.start_tick&&r.end_tick<=end);
  assert.equal(r.physics_ticks,r.end_tick-r.start_tick);assert(vec(r.raw_policy_action_full12,12));
  assert(typeof r.environment_step_returned==='boolean');
  if(r.environment_step_returned){returned++;assert.deepEqual(r.step_info.raw_policy_action_full12,r.raw_policy_action_full12);
   assert(r.step_info.prefix_teacher_data_in_ppo_storage!==true);}
  else assert.equal(r.end_tick,end,'Only the actual terminal decision may be unreturned');
  phaseCounts[r.request_phase]=(phaseCounts[r.request_phase]??0)+1;
  if(r.request_phase==='P02'&&r.environment_step_returned){
   const ev=r.step_info.semantic_task?.physical_evaluator,fr=ev?.current_legs?.FR;
   assert(fr&&ev.history,'Actual P02 endpoint lacks current physical FR evidence');
   if(fr){p02Endpoints.samples++;add(p02Endpoints.clearance_m,fr.clearance_m,r.end_tick);
    add(p02Endpoints.front_distance_m,fr.front_distance_m,r.end_tick);
    if(fr.air===true)p02Endpoints.current_air_samples++;
    if(ev.history?.active_lift?.FR===true)p02Endpoints.qualified_history_samples++;}
  }
  decisions.push({line,...pick(r,['decision','request_phase','start_tick','end_tick','physics_ticks','environment_step_returned']),
   returned_termination_reason:r.environment_step_returned?(r.step_info.termination_reason??null):null,
   returned_end_phase:r.environment_step_returned?(r.step_info.end_phase_id??null):null,
   raw:r.raw_policy_action_full12});lastEnd=r.end_tick;
 });
 assert.equal(lastEnd,end);assert.equal(decisions.length,final.issued_policy_decisions);assert.equal(returned,final.completed_environment_steps);
 const last=decisions.at(-1);assert.equal(last.environment_step_returned?0:last.physics_ticks,final.interrupted_final_decision_ticks);
 let nativeTick=0,decisionIndex=0,nonzeroNativeTicks=0;const effects=Array.from({length:12},()=>metric()),rawMetrics=Array.from({length:12},()=>metric());
 await rows(paths.native_tick_audit,(r,line)=>{
  const tick=r.episode_physics_tick,a=r.native_audit;assert.equal(tick,++nativeTick);assert(tick<=end);
  while(tick>decisions[decisionIndex].end_tick)decisionIndex++;
  const d=decisions[decisionIndex];assert(tick>d.start_tick);
  assert.equal(a?.verified,true);assert.equal(a.setter_dispatch_targets_equal,true);assert.equal(a.actual_mapping_matches_dispatch,true);
  assert.equal(a.same_tick_counterfactual,true);assert.deepEqual(a.phase_mask_full12,Array(12).fill(1));
  assert.deepEqual(a.raw_policy_action_full12,d.raw);assert.equal(a.policy_request_phase,d.request_phase);
  writeKeys.forEach(k=>assert.equal(r[k],0,`In-episode state write: ${k}`));
  assert(vec(a.native_target_delta?.servo_position_rad,8)&&vec(a.native_target_delta?.wheel_velocity_rad_s,4));
  const values=[...a.native_target_delta.servo_position_rad,...a.native_target_delta.wheel_velocity_rad_s];
  assert(Array.isArray(a.changed_channels_full12)&&a.changed_channels_full12.length===12
   &&a.changed_channels_full12.every(x=>typeof x==='boolean'));
  const changed=a.changed_channels_full12.filter(x=>x===true).length;
  assert.equal(changed,a.changed_target_channel_count);if(changed>0)nonzeroNativeTicks++;
  values.forEach((v,i)=>add(effects[i],v,tick));d.raw.forEach((v,i)=>add(rawMetrics[i],v,tick));
 });
 assert.equal(nativeTick,end);
 const rr={hip:{q_deg:metric(),target_deg:metric(),actual_minus_target_deg:metric()},knee:{q_deg:metric(),target_deg:metric(),actual_minus_target_deg:metric()}};
 const frontPrefixEnd=physical.history?.event_ticks?.placed?.FR??end;
 assert(Number.isInteger(frontPrefixEnd)&&frontPrefixEnd>=0&&frontPrefixEnd<=end);
 const rrFrontPrefix={hip:{q_deg:metric(),target_deg:metric(),actual_minus_target_deg:metric()},
  knee:{q_deg:metric(),target_deg:metric(),actual_minus_target_deg:metric()}};
 let physicalTick=-1,terminalJoint=null;
 await rows(paths.physical_observations,r=>{
  assert.equal(r.physics_tick,++physicalTick);assert(physicalTick<=end);assert(Math.abs(r.simulation_time_s-physicalTick/120)<1e-8);
  for(const joint of ['hip','knee']){const v=r.joints?.[`rear_right_${joint}`];assert(v);
   add(rr[joint].q_deg,v.position_deg,physicalTick);add(rr[joint].target_deg,v.command_deg,physicalTick);
   add(rr[joint].actual_minus_target_deg,Number.isFinite(v.position_deg)&&Number.isFinite(v.command_deg)?v.position_deg-v.command_deg:null,physicalTick);
   if(physicalTick<=frontPrefixEnd){add(rrFrontPrefix[joint].q_deg,v.position_deg,physicalTick);
    add(rrFrontPrefix[joint].target_deg,v.command_deg,physicalTick);
    add(rrFrontPrefix[joint].actual_minus_target_deg,v.position_deg-v.command_deg,physicalTick);}}
  if(physicalTick===end)terminalJoint=Object.fromEntries(['hip','knee'].map(j=>[j,pick(r.joints[`rear_right_${j}`],['position_deg','command_deg','velocity_deg_s'])]));
 });
 assert.equal(physicalTick,end);
 const transitions=[];
 await rows(paths.stage_transition_evidence,(r,line)=>{assert(Number.isInteger(r.physics_tick)&&r.physics_tick<=end);
  if(r.from_stage!==r.to_stage)transitions.push({...pick(r,['physics_tick','from_stage','to_stage']),source_line:line});});
 let frames=0,lastFrame=0;
 await rows(paths.viewport_frame_ledger,r=>{assert.equal(r.encoded_frame_index,frames++);
  assert(r.sim_step>lastFrame&&r.sim_step<=end);assert.equal(r.sim_step,Math.min(frames*8,end));lastFrame=r.sim_step;});
 assert.equal(lastFrame,end);assert.equal(frames,Math.ceil(end/8));
 const interval=final.task_interval_window;assert.equal(interval.endpoint_episode_tick,end);assert.equal(interval.last_frame_episode_tick,end);
 assert.equal(interval.frame_count,frames);
 assert.equal(interval.extra_physics_ticks,0);assert.equal(interval.extra_pre_frames,0);assert.equal(interval.extra_post_frames,0);
 const remaining=firstRemainingTask({physical_task_success:final.physical_task_success},physical,transitions.at(-1)?.to_stage??'P01');
 const events=(physical.history?.lift_attempt_events??[]).filter(e=>e.leg==='FR');
 const fr={events,history_event_ticks:Object.fromEntries(['active_lift','front_edge_crossed','placed'].map(k=>[k,physical.history?.event_ticks?.[k]?.FR??null])),
  terminal_IQCP:remaining.leg_events.FR,P02_decision_endpoints:p02Endpoints,
  P02_to_P03:transitions.filter(t=>t.from_stage==='P02'&&t.to_stage==='P03')};
 return{schema:'wlr50_clean.p02_policy_recovery_evidence.v1',created_at_utc:new Date().toISOString(),purpose,
  fixture_not_current_round:purpose==='fixture',fixture_result_must_not_be_used_as_current_success:purpose==='fixture',
  source_paths:{source,run:runDir,checkpoint:cpPath,checkpoint_manifest:cpManifest,...paths},
  runtime:{source_git_commit:expectedHead,runtime_content_sha256:final.runtime_contract.runtime_content_sha256},
  checkpoint:{saved_global_policy_decisions:cp.global_policy_decisions,ppo_updates:cp.ppo_updates,optimizer_steps:cp.optimizer_steps,
   checkpoint_sha256:cp.checkpoint_sha256,save_load_round_trip_recorded:true,actual_eval_official_reload_verified:true,
   parameter_hashes:proof.parameter_hashes,normalizer_frozen_by_official_eval_contract:true,optimizer_updates:0,
   independent_tensor_reload_performed_by_extractor:false},
  policy_execution:{natural_P01_no_snapshot_verified:true,full12_raw_and_native_mask_verified:true,
   teacher_prefix_present:false,teacher_source_default_argument_is_not_teacher_execution:true,
   deterministic_conditional_mean_contract:proof.policy_contract.deterministic_output,
   issued_decisions:decisions.length,returned_decisions:returned,request_phase_counts:phaseCounts,
   native_verified_ticks:nativeTick,native_nonzero_actual_target_effect_ticks:nonzeroNativeTicks,
   native_nonzero_effect_observed:nonzeroNativeTicks>0,
   raw_latent_per_channel:rawMetrics,native_actual_minus_same_tick_zero_policy_target:effects,
   native_effect_units:'first8 physical servo rad; last4 physical wheel rad/s; target effect, not measured torque or guaranteed body movement',
   last_decision:pick(last,['line','decision','request_phase','start_tick','end_tick','physics_ticks','environment_step_returned',
    'returned_termination_reason','returned_end_phase']),
   unreturned_last_ticks_do_not_invent_a_policy_reward:final.interrupted_final_decision_ticks},
  FR:fr,stage_transitions:transitions,RR:{full_rate:rr,terminal:terminalJoint,
   natural_P01_through_FR_placement_or_actual_endpoint:{end_tick:frontPrefixEnd,full_rate:rrFrontPrefix,
    scope:'Separates early uncontrolled descent from any later commanded rear-leg swing; not an automatic success verdict'}},
  endpoint:{physics_tick:end,duration_s:end/120,physical_rows:end+1,native_rows:nativeTick,
   video_frames:frames,last_video_tick:lastFrame,continuous_to_actual_endpoint_verified:true,
   coverage_proof_scope:'Recorded physical/native/decision/frame-ledger continuity; video codec/decodability belongs to the separate media receipt',
   ...pick(physical,['termination_reason','termination_source','traversal_task_complete','task_completed_controlled']),
   last_returned_decision_termination_reason:last.returned_termination_reason,
   source_acceptance_error:final.source_acceptance_error,physical_task_success:final.physical_task_success,
   terminal_hard_joint_limit:physical.termination_source==='HARD_JOINT_LIMIT',first_remaining_task:remaining},
  result_boundaries:{P02_to_P03_observed:fr.P02_to_P03.length>0,
   current_round_task_success:purpose==='formal'?final.physical_task_success:null,
   automatic_P02_recovery_success_claim:false,
   interpretation:'I/Q history, present lift, edge crossing, placement and complete traversal remain distinct. Native target effect is not contact/actuator causality. A completed extraction or valid failed video is not a successful policy.'}};
}

async function main(){const argv=process.argv.slice(2),options={};
 for(let i=0;i<argv.length;i+=2){assert(['--source','--checkpoint','--expected-head','--purpose','--output'].includes(argv[i])&&argv[i+1]);
  assert(!(argv[i]in options));options[argv[i]]=argv[i+1];}assert.equal(Object.keys(options).length,5);
 const dest=path.resolve(project,options['--output']),rel=path.relative(here,dest);
 assert(rel&&!rel.startsWith('..')&&!path.isAbsolute(rel)&&dest.endsWith('.json')&&!fs.existsSync(dest));
 const result=await extract({source:options['--source'],checkpoint:options['--checkpoint'],expectedHead:options['--expected-head'],purpose:options['--purpose']});
 fs.writeFileSync(dest,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
 console.log(JSON.stringify({output:dest,purpose:result.purpose,checkpoint:result.checkpoint.saved_global_policy_decisions,
  end_tick:result.endpoint.physics_tick,P02_to_P03:result.result_boundaries.P02_to_P03_observed,
  hard_joint_limit:result.endpoint.terminal_hard_joint_limit,task_success:result.result_boundaries.current_round_task_success}));
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))
 main().catch(e=>{console.error(e.stack);process.exitCode=1;});

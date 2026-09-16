// Output-only adapter of analyze_training_action_distribution.mjs; sealed runs only.
// Usage: node THIS.mjs --receipt COMPLETED_SUMMARIZE_RECEIPT.json --output NEW.json
// Self-test: node THIS.mjs --self-test (two synthetic decisions; no files/models/runs).
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';

const here=path.dirname(fileURLToPath(import.meta.url)),project=path.resolve(here,'../..');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,''));
const resolve=p=>path.resolve(project,p);
const OLD='history_conditioned_heteroscedastic_log_v1',NEW='history_conditioned_heteroscedastic_log_temperature_v1';
const OLD_CLASS='wlr50_clean.ppo.semantic_history_actor:SemanticHistoryMLPModel';
const NEW_CLASS='wlr50_clean.ppo.semantic_history_actor:SemanticTemperedHistoryMLPModel';
const LAYOUT='diagonal_transfer_state_v1';
const names=['front_left_hip','front_left_knee','front_right_hip','front_right_knee',
 'rear_left_hip','rear_left_knee','rear_right_hip','rear_right_knee',
 'front_left_ankle','front_right_ankle','rear_left_ankle','rear_right_ankle'];
const phaseNames=Array.from({length:13},(_,i)=>`P${String(i+1).padStart(2,'0')}`);
const vector=v=>Array.isArray(v)&&v.length===12&&v.every(Number.isFinite);
const countPhases=()=>Object.fromEntries(phaseNames.map(p=>[p,0]));

function expectedContract(version,role){
 const result={schema:'wlr50_clean.semantic_policy_distribution.v1',version,
  distribution_class:'HeteroscedasticGaussianDistribution',std_type:'log',observation_dimension:324,
  raw_action_dimension:12,actor_hidden_dims:[256,256],activation:'elu',state_dependent_std:true,
  normalization:'fixed_versioned_observation_schema; identity_RSL_normalizer',
  raw_action_semantics:'unbounded_Gaussian_latent_before_existing_tanh_projection',actor_class:OLD_CLASS,
  history_feature:'previous_raw_full12',history_slice:[195,207],history_clip:20,rho:.9,
  conditional_mean:'(1-rho)*base_mean+rho*clipped_previous_raw',
  conditional_std:'unchanged_learned_sigma_as_innovation; no_stationary_rescaling',
  history_state:'stored_observation_only; no_actor_mutable_history',deterministic_output:'conditional_mean',
  export_support:'JIT_and_ONNX_rejected_until_explicitly_supported'};
 if(role)Object.assign(result,{observation_dimension:372,observation_layout:LAYOUT,base_observation_dimension:324,
  role_observation_group:'transfer_role_context_full48',role_observation_slice:[324,372],
  role_observation_leg_order:['FL','FR','RL','RR'],role_observation_fields:['valid','workspace_progress',
   'preparation_progress','transfer_progress','motion_fraction','preparation_ready','transfer_ready',
   'fixed_direction_world_x','fixed_direction_world_y','short_support_continuity_fraction',
   'continued_response_fraction','window_evidence_fraction']});
 if(version===NEW)Object.assign(result,{actor_class:NEW_CLASS,exploration_std_temperature:.5,
  conditional_std:'0.5*learned_sigma_as_innovation; no_stationary_rescaling',
  effective_log_std:'learned_log_std+log(0.5)',
  temperature_scope:'all_stochastic_sample_logprob_entropy_KL_calls; deterministic_mean_unchanged'});
 return result;
}

export function temperatureFromContract(contract){
 assert(contract&&[OLD,NEW].includes(contract.version),'Unsupported or absent saved policy contract');
 const role=contract.observation_layout===LAYOUT;
 assert(contract.version!==NEW||role,'Tempered contract must declare ROLE372');
 assert.deepEqual(contract,expectedContract(contract.version,role),'Saved contract is not the exact supported version');
 return contract.version===OLD?1:.5;
}

// Online population summaries: constant memory, no second audit pass/quantile sort.
const emptyStat=()=>({n:0,mean:0,m2:0,square_sum:0,min:Infinity,max:-Infinity,nonzero_count:0});
function add(s,x){assert(Number.isFinite(x));s.n++;const d=x-s.mean;s.mean+=d/s.n;s.m2+=d*(x-s.mean);
 s.square_sum+=x*x;s.min=Math.min(s.min,x);s.max=Math.max(s.max,x);if(x!==0)s.nonzero_count++;}
const finish=s=>s.n?{n:s.n,min:s.min,max:s.max,mean:s.mean,population_std:Math.sqrt(Math.max(0,s.m2/s.n)),
 rms:Math.sqrt(s.square_sum/s.n),nonzero_count:s.nonzero_count}:null;
const fields=['conditional_mean','recorded_effective_sigma','derived_learned_sigma','raw',
 'innovation_z','projected_residual_endpoint','native_target_effect_endpoint'];
const groups=()=>Object.fromEntries(phaseNames.map(p=>[p,{samples:0,physics_ticks:0,verified_ticks:0,
 actual_native_effect_ticks:0,mask_all12_decisions:0,terminal_request_phase_count:0,
 terminal_end_phase_count:0,channels:names.map(()=>Object.fromEntries(fields.map(k=>[k,emptyStat()])))}]));

export function accumulate(byPhase,row,tau){
 assert(tau===1||tau===.5);const a=row.applied_audit;
 assert(a&&a.phase_id in byPhase&&a.end_phase_id in byPhase);
 assert.equal(typeof row.terminal,'boolean','Terminal status must be actual recorded boolean');
 assert.notEqual(a.prefix_teacher_data_in_ppo_storage,true,'Teacher data cannot be counted as policy decisions');
 const mu=row.old_distribution_mean_full12,sigma=row.old_distribution_std_full12,raw=row.raw_policy_action_full12;
 assert(vector(mu)&&vector(sigma)&&vector(raw)&&sigma.every(s=>s>0),'Missing/invalid recorded pre-action distribution');
 assert.deepEqual(a.raw_policy_action_full12,raw);
 const native=a.actuator_target_effect_audit,summary=a.actuator_target_effect_audit_summary;
 assert(native?.verified===true&&summary?.all_ticks_verified===true);
 assert.deepEqual(native.raw_policy_action_full12,raw);
 assert.deepEqual(native.phase_mask_full12,Array(12).fill(1),'All12 enabled proof absent');
 assert(Number.isInteger(a.physics_ticks)&&a.physics_ticks>0);
 assert.equal(summary.physics_ticks,a.physics_ticks);assert.equal(summary.verified_tick_count,a.physics_ticks);
 assert(Number.isInteger(summary.actual_native_effect_tick_count)&&summary.actual_native_effect_tick_count>=0
  &&summary.actual_native_effect_tick_count<=a.physics_ticks);
 const delta=[...native.native_target_delta.servo_position_rad,...native.native_target_delta.wheel_velocity_rad_s];
 assert(vector(delta)&&vector(a.projected_residual_full12));
 const g=byPhase[a.phase_id];g.samples++;g.physics_ticks+=a.physics_ticks;
 g.verified_ticks+=summary.verified_tick_count;g.actual_native_effect_ticks+=summary.actual_native_effect_tick_count;
 g.mask_all12_decisions++;
 for(let i=0;i<12;i++)for(const[k,v]of Object.entries({conditional_mean:mu[i],recorded_effective_sigma:sigma[i],
  derived_learned_sigma:sigma[i]/tau,raw:raw[i],innovation_z:(raw[i]-mu[i])/sigma[i],
  projected_residual_endpoint:a.projected_residual_full12[i],native_target_effect_endpoint:delta[i]}))add(g.channels[i][k],v);
 if(row.terminal){assert.equal(a.terminal_bootstrap_allowed,false);
  g.terminal_request_phase_count++;byPhase[a.end_phase_id].terminal_end_phase_count++;}
 return row.terminal?{global_policy_decision:row.global_policy_decision,request_phase:a.phase_id,
  end_phase:a.end_phase_id,endpoint_tick:a.physics_tick,interval_ticks:a.physics_ticks,
  termination_reason:a.termination_reason??null,task_success:a.task_success??null,
  full_task_success:a.full_task_success??null,terminal_bootstrap_allowed:false}:null;
}

const finished=byPhase=>Object.fromEntries(Object.entries(byPhase).map(([p,g])=>[p,{...g,
 channels:g.channels.map((c,i)=>({index:i,channel:names[i],...Object.fromEntries(fields.map(k=>[k,finish(c[k])]))}))}]));

export async function analyze(receiptPath){
 receiptPath=resolve(receiptPath);const receipt=read(receiptPath);
 assert(['wlr50_clean.height_round_completed_training.v1','wlr50_clean.completed_training_summary.v1'].includes(receipt.schema));
 assert(Array.isArray(receipt.runs)&&receipt.runs.length&&receipt.aggregate?.runs_are_sequential_checkpoint_chain===true);
 assert.equal(new Set(receipt.runs.map(r=>resolve(r.run))).size,receipt.runs.length);
 const runs=[],all=groups(),terminals=[],phaseCounts=countPhases();let total=0;
 for(const [index,r]of receipt.runs.entries()){
  assert(['SUCCEEDED','STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'].includes(r.lifecycle)&&r.completed_at_utc);
  const run=fs.realpathSync(resolve(r.run)),manifest=read(path.join(run,'run_manifest.json'));
  assert.equal(manifest.lifecycle,r.lifecycle);assert(manifest.completed_at_utc);assert.equal(manifest.arguments?.command,'train');
  const savedPath=fs.realpathSync(resolve(r.saved.manifest)),saved=read(savedPath),tau=temperatureFromContract(saved.policy_contract);
  assert.equal(fs.realpathSync(resolve(saved.source_run)),run);
  assert.equal(saved.global_policy_decisions,r.saved.global_policy_decisions);
  assert.equal(saved.save_load_round_trip,true);
  assert.deepEqual(saved.policy_contract,r.configuration.policy_contract);
  assert.equal(saved.runner_config.actor.class_name,saved.policy_contract.actor_class);
  assert.equal(saved.runner_config.actor.distribution_cfg.class_name,'HeteroscedasticGaussianDistribution');
  assert.equal(saved.runner_config.actor.distribution_cfg.std_type,'log');
  assert.equal(saved.runner_config.actor.obs_normalization,false);
  if(tau===.5)assert.equal(saved.runner_config.actor.exploration_std_temperature,.5);
  else assert(!('exploration_std_temperature' in saved.runner_config.actor));
  assert.equal(r.sampling.teacher_prefix_exclusion_verified,true);
  if(index)assert.equal(r.source.global_policy_decisions,receipt.runs[index-1].saved.global_policy_decisions);
  const audit=path.join(run,'residual_and_projection_audit.jsonl'),local=groups(),terminalRows=[];
  let rows=0,previous=r.source.global_policy_decisions;
  for await(const line of readline.createInterface({input:fs.createReadStream(audit),crlfDelay:Infinity})){
   if(!line.trim())continue;const row=JSON.parse(line);
   assert.equal(row.global_policy_decision,++previous);rows++;total++;
   if(r.configuration.prefix_request)assert.equal(row.applied_audit?.prefix_teacher_data_in_ppo_storage,false);
   const terminal=accumulate(local,row,tau);accumulate(all,row,tau);
   phaseCounts[row.applied_audit.phase_id]++;
   if(terminal){terminalRows.push(terminal);terminals.push({run_index:index,...terminal});}
  }
  assert.equal(rows,r.increments.policy_decisions);assert.equal(previous,r.saved.global_policy_decisions);
  for(const p of phaseNames)assert.equal(local[p].samples,r.sampling.phase_policy_decisions[p]??0);
  assert.equal(terminalRows.length,r.sampling.terminal_decisions.length);
  assert.deepEqual(terminalRows.map(t=>t.global_policy_decision),r.sampling.terminal_decisions.map(t=>t.global_policy_decision));
  runs.push({run,audit,saved_manifest:savedPath,source_checkpoint_decisions:r.source.global_policy_decisions,
   saved_checkpoint_decisions:r.saved.global_policy_decisions,policy_version:saved.policy_contract.version,
   saved_source_git_commit:saved.runtime_contract?.source_git_commit??null,
   exploration_std_temperature:tau,audit_streaming_passes:1,rows,phases:finished(local),terminals:terminalRows});
 }
 assert.equal(total,receipt.aggregate.added_policy_decisions);
 for(const p of phaseNames)assert.equal(phaseCounts[p],receipt.aggregate.phase_policy_decisions[p]??0);
 return{schema:'wlr50_clean.completed_training_temperature_distribution.v1',created_at_utc:new Date().toISOString(),
  completed_training_receipt:receiptPath,receipt_purpose:receipt.purpose??null,
  fixture_not_current_round:receipt.fixture_not_current_round??null,runs,
  aggregate:{samples:total,phase_samples:phaseCounts,phases:finished(all),actual_terminal_count:terminals.length,
   P02_terminal_request_phase_count:all.P02.terminal_request_phase_count,P02_terminal_end_phase_count:all.P02.terminal_end_phase_count,
   temperatures_present:[...new Set(runs.map(r=>r.exploration_std_temperature))],terminals,
   all12_enabled_and_native_audit_verified:true,all12_scope:'Every credited decision had all12 mask; summaries verified every actual physics tick. Nonzero endpoint raw/projected/native counts are separately reported per channel, not inferred from enabled masks.'},
  definitions:{samples:'Credited policy decisions assigned to actual request phase, not 120Hz ticks or requested curriculum phase.',
   conditional_mean:'Recorded pre-action conditional mean including observation HISTORY; no model forward.',
   recorded_effective_sigma:'Recorded old_distribution_std_full12 actually used by sample/logprob. Not init_std or marginal action variability.',
   derived_learned_sigma:'DERIVED ONLY: recorded_effective_sigma / exact saved-contract temperature. Not an independently read head. exp(log_sigma+log(tau)) rounding means this may differ slightly from independently exponentiating learned_log_sigma.',
   innovation_z:'(recorded raw - recorded conditional mean) / recorded effective sigma at the same decision.',
   native_target_effect_endpoint:'Last-tick same-prestate policy target effect: servo radians then wheel rad/s; not nominal geometry, measured torque, tracking or all-tick extrema.',
   terminal:'Actual row.terminal only, with recorded false bootstrap; nonterminal budget tails are not episodes.',
   aggregate:'Descriptive across evolving states/checkpoints; mixed temperatures remain separately identified per run. Not a causal improvement or normality test.'},
  changes:{production:false,checkpoint_tensor_loaded:false,model_forward:false,simulator_launched:false,checkpoint_rehashed:false}};
}

function selfTest(){
 for(const[version,tau]of[[OLD,1],[NEW,.5]]){
  const c=expectedContract(version,true);assert.equal(temperatureFromContract(c),tau);
  const bad=structuredClone(c);bad.exploration_std_temperature=.25;
  assert.throws(()=>temperatureFromContract(bad));
 }
 const g=groups(),make=(terminal,phase)=>({global_policy_decision:terminal?2:1,terminal,
  old_distribution_mean_full12:Array(12).fill(.25),old_distribution_std_full12:Array(12).fill(.1),
  raw_policy_action_full12:Array(12).fill(.45),applied_audit:{phase_id:phase,end_phase_id:phase,
   raw_policy_action_full12:Array(12).fill(.45),projected_residual_full12:Array(12).fill(.02),
   physics_ticks:8,physics_tick:terminal?16:8,terminal_bootstrap_allowed:!terminal,termination_reason:terminal?'FALL':null,
   actuator_target_effect_audit:{verified:true,raw_policy_action_full12:Array(12).fill(.45),phase_mask_full12:Array(12).fill(1),
    native_target_delta:{servo_position_rad:Array(8).fill(.001),wheel_velocity_rad_s:Array(4).fill(.01)}},
   actuator_target_effect_audit_summary:{all_ticks_verified:true,physics_ticks:8,verified_tick_count:8,actual_native_effect_tick_count:8}}});
 const first=make(false,'P01'),second=make(true,'P02');
 assert.equal(accumulate(g,first,1),null);assert.equal(accumulate(g,second,.5).termination_reason,'FALL');
 const done=finished(g);assert.equal(done.P01.channels[0].derived_learned_sigma.mean,.1);
 assert.equal(done.P02.channels[0].derived_learned_sigma.mean,.2);
 assert.equal(done.P02.channels[0].innovation_z.mean,2);assert.equal(done.P02.terminal_end_phase_count,1);
 const masked=structuredClone(second);masked.applied_audit.actuator_target_effect_audit.phase_mask_full12[0]=0;
 assert.throws(()=>accumulate(groups(),masked,.5));
 const teacher=structuredClone(second);teacher.applied_audit.prefix_teacher_data_in_ppo_storage=true;
 assert.throws(()=>accumulate(groups(),teacher,.5));
 return{self_test:true,synthetic_decision_rows:2,real_audits_read:0,model_forward:false,result:'passed'};
}

async function main(){
 const args=process.argv.slice(2);
 if(args.length===1&&args[0]==='--self-test'){console.log(JSON.stringify(selfTest()));return;}
 assert.equal(args.length,4,'--receipt COMPLETED_JSON --output NEW_OUTPUT_JSON');
 const options={};for(let i=0;i<args.length;i+=2){assert(['--receipt','--output'].includes(args[i])&&!options[args[i]]);options[args[i]]=args[i+1];}
 assert(options['--receipt']&&options['--output']);
 const output=resolve(options['--output']),relative=path.relative(here,output);
 assert(relative&&!relative.startsWith('..')&&!path.isAbsolute(relative)&&output.endsWith('.json')&&!fs.existsSync(output));
 const result=await analyze(options['--receipt']);
 fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
 console.log(JSON.stringify({output,samples:result.aggregate.samples,temperatures:result.aggregate.temperatures_present,
  terminal_count:result.aggregate.actual_terminal_count,P02_terminal_count:result.aggregate.P02_terminal_end_phase_count}));
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))
 main().catch(e=>{console.error(e.stack);process.exitCode=1;});

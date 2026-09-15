// Output-only receipt for explicitly named completed runs; no torch, Isaac or checkpoint bytes.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import {fileURLToPath} from 'node:url';

const project = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const phases = Array.from({length:13}, (_,i)=>`P${String(i+1).padStart(2,'0')}`);
const phaseCounts = ()=>Object.fromEntries(phases.map(p=>[p,0]));
const read = p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,''));
const resolve = p=>path.resolve(project,p);
const pick = (o,keys)=>Object.fromEntries(keys.map(k=>[k,o?.[k]??null]));
const require = (condition,message)=>{if(!condition) throw Error(message);};
const sum = (xs,key)=>xs.reduce((n,x)=>n+(x[key]??0),0);
const range = xs=>xs.length?{min:Math.min(...xs),max:Math.max(...xs)}:null;
const countKeys = ['global_policy_decisions','ppo_updates','optimizer_steps'];

async function rows(file,consume) {
  const lines=readline.createInterface({input:fs.createReadStream(file),crlfDelay:Infinity});
  let n=0;
  for await(const line of lines) if(line.trim()) {consume(JSON.parse(line),n);n++;}
  return n;
}

export async function summarize(runArgument) {
  const run=fs.realpathSync(resolve(runArgument));
  const lifecycle=read(path.join(run,'run_manifest.json'));
  require(['SUCCEEDED','STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'].includes(lifecycle.lifecycle)
    && Boolean(lifecycle.completed_at_utc),'Only a finalized successful-execution or verified-update-boundary run is accepted');
  const args=lifecycle.arguments;
  require(args?.command==='train','Not a training run');
  const train=read(path.join(run,'training_manifest.json'));
  require(train.lifecycle===lifecycle.lifecycle,'Lifecycle/training status mismatch');
  for(const k of ['actual_policy_decisions','ppo_updates_this_run','optimizer_steps_this_run'])
    require(train[k]===lifecycle.result?.[k],`Run result disagrees with training manifest: ${k}`);
  const last=train.checkpoints?.at(-1);
  require(last,'Completed training has no saved checkpoint record');
  const savedPath=fs.realpathSync(resolve(last.manifest));
  const saved=read(savedPath);
  const sourcePath=fs.realpathSync(resolve(saved.resume_ancestry?.source_checkpoint?.manifest
    ??args.checkpoint.replace(/\.pt$/,'_manifest.json')));
  const source=read(sourcePath);
  const deltas=Object.fromEntries(countKeys.map(k=>[k,saved[k]-source[k]]));
  require(deltas.global_policy_decisions===train.actual_policy_decisions
    && deltas.ppo_updates===train.ppo_updates_this_run
    && deltas.optimizer_steps===train.optimizer_steps_this_run,'Saved/source/manifest count deltas disagree');
  require(saved.source_run&&fs.realpathSync(saved.source_run)===run,'Saved sidecar belongs to another run');
  require(saved.runtime_contract.runtime_content_sha256===train.runtime_contract.runtime_content_sha256
    && lifecycle.runtime_contract.runtime_content_sha256===train.runtime_contract.runtime_content_sha256,
    'Saved/run/training runtime digests disagree');
  const counts=phaseCounts(),teacher={true:0,false:0,absent_or_null:0},terminals=[];
  let ticks=0,first=null,lastRow=null,prior=source.global_policy_decisions;
  const auditRows=await rows(path.join(run,'residual_and_projection_audit.jsonl'),row=>{
    const a=row.applied_audit;
    require(a&&row.global_policy_decision===prior+1,'Audit credit order is missing, repeated or discontinuous');
    prior=row.global_policy_decision;
    if(first===null) first=prior;
    counts[a.phase_id]=(counts[a.phase_id]??0)+1;
    require(Number.isInteger(a.physics_ticks)&&a.physics_ticks>0,'Audit physics tick count is invalid');
    ticks+=a.physics_ticks;
    const flag=a.prefix_teacher_data_in_ppo_storage;
    teacher[flag===true?'true':flag===false?'false':'absent_or_null']++;
    lastRow={global_policy_decision:prior,phase:a.phase_id,end_phase:a.end_phase_id,
      physics_tick:a.physics_tick,sim_time_s:a.sim_time_s,terminal:row.terminal,
      terminal_bootstrap_allowed:a.terminal_bootstrap_allowed,task_result_scope:a.task_result_scope??'unknown_not_recorded'};
    if(row.terminal) terminals.push({...lastRow,termination_reason:a.termination_reason,
      task_success:a.task_success,full_task_success:a.full_task_success});
  });
  require(auditRows===train.actual_policy_decisions&&prior===saved.global_policy_decisions,
    'Credited audit rows do not match optimized checkpoint count');
  const updates=[];
  await rows(path.join(run,'optimizer_updates.jsonl'),r=>updates.push(pick(r,[
    'ppo_update','global_policy_decisions','optimizer_steps','optimizer_learning_rate',
    'actor_parameters_changed','finite_nonzero_gradient_observed','gradient_norm_min','gradient_norm_max',
    'kl_mean','clip_fraction','entropy','value_loss','surrogate_loss'])));
  require(updates.length===deltas.ppo_updates&&sum(updates,'optimizer_steps')===deltas.optimizer_steps,
    'Recorded optimizer updates disagree with checkpoint counts');
  const episodes=[];
  await rows(path.join(run,'completed_episodes.jsonl'),r=>episodes.push(pick(r,[
    'episode_index','seed','policy_decisions','termination_reason','task_success',
    'task_outcome_label','full_task_success','duration_s'])));
  const cfg=saved.runner_config,core=train.telemetry?.core??{},prefix=saved.curriculum_epoch?.prefix_request;
  const telemetryPhases=core.phase_decisions??{};
  const phaseMatch=Object.keys({...counts,...telemetryPhases}).every(k=>(counts[k]??0)===(telemetryPhases[k]??0));
  require(phaseMatch,'Credited phase audit differs from curriculum telemetry');
  const teacherDecisions=core.prefix_behavior_decisions??(prefix?null:0);
  const teacherTicks=core.prefix_physics_ticks??(prefix?null:0);
  return {
    run,lifecycle:lifecycle.lifecycle,completed_at_utc:lifecycle.completed_at_utc,
    configuration:{...pick(args,['semantic_version','experiment_id','stage','from_phase','prefix_source',
      'teacher_offset_decisions','seed','num_envs','device','checkpoint_interval_updates']),
      implemented_sampling:train.implemented_sampling,prefix_request:prefix??null,
      physics_hz:train.runtime_contract.physics_hz,decision_hz:train.runtime_contract.decision_hz,
      rollout_steps_per_env:cfg.num_steps_per_env,rollout_policy_decisions:cfg.num_steps_per_env*train.num_envs,
      observation_dimension:saved.execution_topology?.observation_dimension,
      action_dimension:saved.execution_topology?.action_dimension,
      policy_contract:saved.policy_contract,
      actor:pick(cfg.actor,['class_name','hidden_dims','activation','obs_normalization','observation_layout','distribution_cfg']),
      critic:pick(cfg.critic,['class_name','hidden_dims','activation','obs_normalization']),
      optimizer:pick(cfg.algorithm,['optimizer','schedule','learning_rate','gamma','lam','num_learning_epochs',
        'num_mini_batches','clip_param','max_grad_norm']),
      initial_learning_rate_is_configured_default_not_restored_effective_lr:true},
    request:{planned:train.planned_requested_policy_decisions,consumed:train.requested_policy_decisions,
      unconsumed:train.unconsumed_requested_policy_decisions,rounding_overrun:train.rounding_overrun,
      stage_requested_before:source.stage_requested_decisions,stage_requested_after:saved.stage_requested_decisions,
      stage_requested_delta:Object.fromEntries(Object.keys(saved.stage_requested_decisions).map(k=>
        [k,saved.stage_requested_decisions[k]-(source.stage_requested_decisions?.[k]??0)]))},
    source:{checkpoint:source.checkpoint_path??args.checkpoint,manifest:sourcePath,...pick(source,countKeys),
      effective_learning_rate:source.optimizer_learning_rate,actor_parameter_sha256:source.actor_parameter_sha256,
      checkpoint_sha256:source.checkpoint_sha256},
    saved:{checkpoint:last.checkpoint,manifest:savedPath,...pick(saved,countKeys),
      ...pick(saved,['actor_parameter_sha256','critic_parameter_sha256','optimizer_state_sha256',
        'normalizer_state_sha256','checkpoint_sha256','save_load_round_trip']),
      effective_learning_rate:saved.optimizer_learning_rate,
      source_git_commit:saved.runtime_contract.source_git_commit,
      runtime_content_sha256:saved.runtime_contract.runtime_content_sha256},
    increments:{policy_decisions:deltas.global_policy_decisions,ppo_updates:deltas.ppo_updates,
      optimizer_steps:deltas.optimizer_steps},
    sampling:{phase_policy_decisions:counts,phase_assignment:'Applied audit phase_id at the decision request; requested curriculum phase is not a sample count',
      credited_rows:auditRows,credited_physics_ticks:ticks,
      first_global_policy_decision:first,last_global_policy_decision:prior,
      teacher_storage_flags:teacher,teacher_prefix_exclusion_verified:teacher.true===0&&(!prefix||teacher.false===auditRows),
      teacher_prefix_decisions_excluded:teacherDecisions,
      teacher_prefix_physics_ticks_excluded:teacherTicks,prefix_attempts:core.prefix_attempts??[],
      all_physical_decisions_including_prefix:core.physical_core_including_prefix?.decisions??auditRows,
      phase_counts_match_credited_telemetry:phaseMatch,terminal_decisions:terminals,last_credited_decision:lastRow},
    updates,gradient_summary:{all_updates_finite_nonzero:updates.every(r=>r.finite_nonzero_gradient_observed===true),
      all_updates_actor_changed:updates.every(r=>r.actor_parameters_changed===true),
      optimizer_gradient_norm_range:range(updates.flatMap(r=>[r.gradient_norm_min,r.gradient_norm_max]).filter(Number.isFinite)),
      effective_learning_rate_range:range(updates.map(r=>r.optimizer_learning_rate).filter(Number.isFinite)),
      gradient_measurement:'Existing optimizer pre-hook norm over all non-null optimizer parameter gradients'},
    state_handling:{normalization:saved.normalization,
      normalizer_hash_unchanged:source.normalizer_state_sha256===saved.normalizer_state_sha256,
      actor_obs_normalization:cfg.actor.obs_normalization,critic_obs_normalization:cfg.critic.obs_normalization,
      resume_physics:saved.resume_physics,physical_env_state_saved:saved.physical_env_state_saved,
      fresh_rollout_required_by_migration:saved.resume_ancestry?.resume_migration?.discard_old_rollout_storage??null,
      migration_schema:saved.resume_ancestry?.resume_migration?.schema??null,
      historical_warm_start_metadata_is_not_this_run_reset:true},
    task_results:{training_success_is_not_task_success:train.training_success_is_not_task_success,
      completed_episodes:episodes,success_count_scope:train.telemetry?.success_count_scope??(prefix?'unknown_not_recorded':'fresh_P01_no_prefix'),
      current_policy_full_P01_success_count:(!prefix||train.telemetry?.success_count_scope==='fresh_P01_current_policy_only')
        ?train.telemetry?.success_count??null:null,
      teacher_initialized_task_success_count:train.telemetry?.teacher_initialized_task_success_count??0,
      nonterminal_tail_is_not_a_completed_episode:lastRow?.terminal===false},
    stop_after_update:train.stop_after_update??null,
    verification:'Metadata plus credited audit/optimizer/episode records only. Recorded save/load proof is not independently rerun; no checkpoint tensor load or byte rehash.'
  };
}

async function main() {
  const argv=process.argv.slice(2),runs=[];let output;
  for(let i=0;i<argv.length;i+=2) {
    require(argv[i+1]!==undefined,'Every argument needs a value');
    if(argv[i]==='--run') runs.push(argv[i+1]);
    else if(argv[i]==='--output') output=argv[i+1];
    else throw Error(`Unknown argument: ${argv[i]}`);
  }
  require(runs.length>0&&output,'Usage: node summarize_completed_training.mjs --run <completed run> [--run <next run>] --output <new outputs JSON>');
  const dest=resolve(output),allowed=resolve('outputs/ppo_timing_task_priority_v1');
  const relative=path.relative(allowed,dest);
  require(relative&&!relative.startsWith('..')&&!path.isAbsolute(relative)&&dest.endsWith('.json')&&!fs.existsSync(dest),
    'Output must be a new JSON inside outputs/ppo_timing_task_priority_v1');
  const results=[];for(const run of runs) results.push(await summarize(run));
  require(new Set(results.map(r=>r.run)).size===results.length,'Duplicate runs would double count training');
  const aggregatePhases=phaseCounts();for(const r of results) for(const[p,n]of Object.entries(r.sampling.phase_policy_decisions))
    aggregatePhases[p]=(aggregatePhases[p]??0)+n;
  const linked=results.every((r,i)=>i===0||countKeys.every(k=>r.source[k]===results[i-1].saved[k])
    &&r.source.actor_parameter_sha256===results[i-1].saved.actor_parameter_sha256
    &&r.source.checkpoint_sha256===results[i-1].saved.checkpoint_sha256);
  require(linked,'Requested runs are not one sequential checkpoint chain; summarize unrelated runs separately');
  const result={schema:'wlr50_clean.completed_training_summary.v1',created_at_utc:new Date().toISOString(),runs:results,
    aggregate:{runs_are_sequential_checkpoint_chain:linked,
      added_policy_decisions:sum(results.map(r=>r.increments),'policy_decisions'),
      added_ppo_updates:sum(results.map(r=>r.increments),'ppo_updates'),
      added_optimizer_steps:sum(results.map(r=>r.increments),'optimizer_steps'),
      phase_policy_decisions:aggregatePhases,
      teacher_prefix_decisions_excluded:results.every(r=>r.sampling.teacher_prefix_decisions_excluded!==null)
        ?sum(results.map(r=>r.sampling),'teacher_prefix_decisions_excluded'):null,
      source_counts:pick(results[0].source,countKeys),final_counts:pick(results.at(-1).saved,countKeys),
      full_task_success_claimed:false}};
  fs.writeFileSync(dest,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({output:dest,...result.aggregate}));
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))
  main().catch(e=>{console.error(e.stack??String(e));process.exitCode=1;});

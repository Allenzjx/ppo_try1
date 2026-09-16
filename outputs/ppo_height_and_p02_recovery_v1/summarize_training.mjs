// Thin output adapter: reuse the completed-run accounting, never load checkpoint tensors.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {summarize} from '../ppo_timing_task_priority_v1/summarize_completed_training.mjs';
const here=path.dirname(fileURLToPath(import.meta.url)),project=path.resolve(here,'../..');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,''));
const counts=['global_policy_decisions','ppo_updates','optimizer_steps'];
const pick=(o,keys)=>Object.fromEntries(keys.map(k=>[k,o?.[k]??null]));

export async function collect(runs,purpose){
 assert(['fixture','formal'].includes(purpose));
 const receipts=[];
 for(const run of runs){
  const r=await summarize(run),source=read(r.source.manifest),saved=read(r.saved.manifest);
  for(const item of [source,saved])for(const k of ['checkpoint_sha256','actor_parameter_sha256',
   'critic_parameter_sha256','optimizer_state_sha256','normalizer_state_sha256'])assert.match(item[k],/^[a-f0-9]{64}$/);
  assert.equal(saved.save_load_round_trip,true);
  assert.equal(saved.resume_ancestry?.source_checkpoint?.checkpoint_sha256,source.checkpoint_sha256);
  assert.equal(path.resolve(saved.checkpoint_path),path.resolve(r.saved.checkpoint));
  assert.equal(path.basename(saved.checkpoint_path),`checkpoint_step_${String(saved.global_policy_decisions).padStart(9,'0')}.pt`);
  assert.equal(path.basename(path.dirname(saved.checkpoint_path)),'history');
  if(purpose==='formal')assert('src/wlr50_clean/ppo/semantic_height_recovery.py' in saved.runtime_contract.files,
    'A previous-round fixture cannot become current formal training');
  const migration=saved.resume_ancestry?.resume_migration;
  r.state_handling={...r.state_handling,
   source_checkpoint_binding_verified:true,
   source_optimizer_state_sha256:source.optimizer_state_sha256??null,
   saved_optimizer_state_sha256:saved.optimizer_state_sha256??null,
   source_rng_state_recorded:!!source.training_rng_state&&Object.keys(source.training_rng_state).length>0,
   saved_rng_state_recorded:!!saved.training_rng_state&&Object.keys(saved.training_rng_state).length>0,
   rng_equality_across_training_is_not_expected:true,
   migration_preserves_all_learned_state:migration?.preserve_actor_critic_optimizer_normalizer_rng_and_budget??null,
   migration_factor_schema:migration?.height_recovery_factor?.schema??null,
   save_load_round_trip_recorded:saved.save_load_round_trip,
   evidence_limit:'Recorded immutable sidecars, successful official resume path and completed accounting. No independent tensor/Adam/RNG reload; saved Adam/RNG may legitimately change during learning.'};
  receipts.push(r);
 }
 assert(receipts.length>0&&new Set(receipts.map(r=>r.run)).size===receipts.length);
 receipts.forEach((r,i)=>{if(i){counts.forEach(k=>assert.equal(r.source[k],receipts[i-1].saved[k]));
  assert.equal(r.source.checkpoint_sha256,receipts[i-1].saved.checkpoint_sha256);}});
 const phaseCounts=Object.fromEntries(Array.from({length:13},(_,i)=>[`P${String(i+1).padStart(2,'0')}`,0]));
 receipts.forEach(r=>Object.entries(r.sampling.phase_policy_decisions).forEach(([k,n])=>{
  assert(k in phaseCounts&&Number.isInteger(n)&&n>=0);phaseCounts[k]+=n;}));
 const sum=(group,key)=>receipts.reduce((n,r)=>n+r[group][key],0),last=receipts.at(-1);
 return{schema:'wlr50_clean.height_round_completed_training.v1',created_at_utc:new Date().toISOString(),
  purpose,fixture_not_current_round:purpose==='fixture',full_task_success_claimed:false,runs:receipts,
  aggregate:{runs_are_sequential_checkpoint_chain:true,added_policy_decisions:sum('increments','policy_decisions'),
   added_ppo_updates:sum('increments','ppo_updates'),added_optimizer_steps:sum('increments','optimizer_steps'),
   phase_policy_decisions:phaseCounts,teacher_prefix_exclusion_verified:receipts.every(r=>r.sampling.teacher_prefix_exclusion_verified),
   teacher_prefix_decisions_excluded:receipts.every(r=>Number.isInteger(r.sampling.teacher_prefix_decisions_excluded))
    ?sum('sampling','teacher_prefix_decisions_excluded'):null,
   source_counts:pick(receipts[0].source,counts),final_counts:pick(last.saved,counts)},
  latest_checkpoint_in_explicit_completed_chain:last.saved,
  latest_scope:'Latest only within the supplied sequential completed runs; no mutable global pointer read'};
}

async function main(){
 const args=process.argv.slice(2),runs=[];let output,purpose;
 for(let i=0;i<args.length;i+=2){assert(args[i+1]);if(args[i]==='--run')runs.push(args[i+1]);
  else if(args[i]==='--output'){assert(!output);output=args[i+1];}
  else if(args[i]==='--purpose'){assert(!purpose);purpose=args[i+1];}else throw Error(`Unknown option ${args[i]}`);}
 assert(output&&purpose&&runs.length,'--run RUN [--run RUN] --purpose fixture|formal --output NEW_JSON');
 const dest=path.resolve(project,output),rel=path.relative(here,dest);
 assert(rel&&!rel.startsWith('..')&&!path.isAbsolute(rel)&&dest.endsWith('.json')&&!fs.existsSync(dest));
 const result=await collect(runs,purpose);
 fs.writeFileSync(dest,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
 console.log(JSON.stringify({output:dest,purpose,...result.aggregate}));
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))
 main().catch(e=>{console.error(e.stack);process.exitCode=1;});

// Output-only: refuse unfinished runs, reuse existing complete-run accounting.
// Emits JSON to stdout so the caller persists it with apply_patch.
// Run once per completed run, not repeatedly against an active large audit.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {summarize} from '../ppo_timing_task_priority_v1/summarize_completed_training.mjs';

function executionBoundary(saved){
 let current=saved;
 // Follow only the already-linked metadata ancestry, not a directory scan or tensor load.
 for(let depth=0;depth<32;depth++){
  const migration=current.resume_ancestry?.resume_migration;
  if(migration?.rr_physical_acceptance_same372_factor
    ||migration?.exploration_temperature_factor?.target_exploration_std_temperature===0.25
    ||migration?.final_stop_handoff_factor||migration?.execution_composition_factor)return migration;
  const link=current.resume_ancestry?.source_checkpoint;
  if(!link?.manifest)return null;
  const parent=JSON.parse(fs.readFileSync(link.manifest,'utf8'));
  assert.equal(parent.checkpoint_sha256,link.checkpoint_sha256);
  assert.deepEqual(parent.runtime_contract,current.resume_ancestry.source_runtime_contract);
  current=parent;
 }
 throw Error('Reviewed execution boundary not found in bounded recorded ancestry');
}

export async function completedReceipt(run, branchId) {
 assert.match(branchId,/^[a-z0-9_]+$/);
 const r=await summarize(run);
 assert(['task_first_recovery_v1','residual_rr_fix_v1'].includes(r.configuration.experiment_id));
 const saved=JSON.parse(fs.readFileSync(r.saved.manifest,'utf8'));
 assert.equal(saved.save_load_round_trip,true);
 const directMigration=saved.resume_ancestry?.resume_migration;
 const migration=executionBoundary(saved)??directMigration;
 const composition=migration?.execution_composition_factor;
 const finalStop=migration?.final_stop_handoff_factor;
 const quarter=migration?.exploration_temperature_factor;
 const rr=migration?.rr_physical_acceptance_same372_factor;
 if(rr){
  assert.equal(r.configuration.experiment_id,'residual_rr_fix_v1');
  assert.equal(branchId,'residual_rr_fix_v1');
  assert.equal(rr.schema,'wlr50_clean.rr_physical_acceptance_same372_continuation.v1');
  assert.equal(quarter,undefined);assert.equal(composition,undefined);assert.equal(finalStop,undefined);
  assert.equal(migration.source_checkpoint_sha256,'e4552bb18ef223fb51da31a616cf57cf3291ba1a108b08e7da40c07bee0db0ba');
  assert.equal(migration.source_manifest_sha256,'9f78b80a8128c3a44c4b13c3e6ea4fbf47e01b491eff8a62c73f0aae50dd1b0c');
  assert.equal(migration.target_git_commit,saved.runtime_contract.source_git_commit);
  assert.equal(migration.target_runtime_content_sha256,saved.runtime_contract.runtime_content_sha256);
  assert.equal(rr.quality_epsilon,0);assert.equal(rr.kernel_changed,false);
  assert.equal(rr.action_ranges_changed,false);assert.equal(rr.nominal_geometry_changed,false);
  assert.equal(rr.reward_code_and_weights_changed,false);assert.equal(rr.task_reward_event_semantics_changed,true);
  assert.equal(rr.task_acceptance_changed,true);assert.equal(rr.action_execution_changed,true);
  assert.equal(rr.observation_layout_changed,false);assert(rr.observation_semantics_changed.length>0);
  assert.equal(rr.p09_lift_semantics,'functional_free_air_lift_v3');
  assert.equal(rr.rr_carry_source_semantics,'current_free_lift_before_pending_knee_and_roll_v1');
  assert.equal(rr.final_stop_owner,'source_home_after_physical_stop_v2');
  const ob=rr.observation_contract;
  assert.deepEqual(ob.source_policy_contract,ob.target_policy_contract);
  assert.deepEqual(saved.policy_contract,ob.target_policy_contract);
  assert.equal(saved.policy_contract.exploration_std_temperature,.25);assert.equal(saved.policy_contract.rho,.9);
  assert.equal(ob.observation_dimension,372);assert.equal(ob.action_dimension,12);
  for(const [file,hash]of Object.entries(rr.reviewed_code_sha256)){
   assert.equal(saved.runtime_contract.files[file],hash);
   assert.equal(migration.changed_file_hashes[file].after,hash);
  }
  assert.equal(Object.keys(rr.configuration_bindings).length,6);
  for(const [name,binding]of Object.entries(rr.configuration_bindings)){
   assert.deepEqual(saved.runtime_contract.selected_configuration[name],binding.target);
   if(name!=='stage_task_spec.yaml')assert.equal(binding.source.sha256,binding.target.sha256);
  }
  assert.deepEqual(rr.counter_origin,{global_policy_decisions:177792,ppo_updates:1354,optimizer_steps:27080});
  assert.deepEqual(saved.rr_task_branch.counter_origin,rr.counter_origin);
  assert.equal(saved.rr_task_branch.branch_id,branchId);
  for(const [key,origin]of Object.entries(rr.counter_origin))assert.equal(saved.rr_task_branch_counts[key],saved[key]-origin);
 }else if(quarter){
  assert.equal(branchId,'p06_quarter_temperature_v1');
  assert.equal(quarter.schema,'wlr50_clean.history372_innovation_temperature_continuation.v1');
  assert.equal(composition,undefined);assert.equal(finalStop,undefined);
  assert.equal(migration.source_checkpoint_sha256,'8ffb446aeabd973057a8240e99f5ce6d7ba91610de79affe3ca552b6826fc30e');
  assert.equal(migration.target_git_commit,'fc14a68c037abd69e24f49b1524457fd6854543a');
  assert.equal(saved.runtime_contract.source_git_commit,migration.target_git_commit);
  assert.equal(saved.runtime_contract.runtime_content_sha256,'bfa1e1471cc0e645ae33e873439a242142dde5adceb70a87bcc419152815c48c');
  const changed=['semantic_history_actor','semantic_migration','semantic_policy_distribution','semantic_training'].map(n=>'src/wlr50_clean/ppo/'+n+'.py');
  assert.deepEqual([...migration.allowed_changed_files].sort(),changed.sort());
  assert.deepEqual(Object.keys(quarter.reviewed_code_sha256).sort(),[...changed,'src/wlr50_clean/ppo/semantic_cli.py'].sort());
  for(const [file,hash]of Object.entries(quarter.reviewed_code_sha256)){
   assert.equal(saved.runtime_contract.files[file],hash);
   if(changed.includes(file))assert.equal(migration.changed_file_hashes[file].after,hash);
  }
  assert.equal(quarter.source_exploration_std_temperature,.5);
  assert.equal(quarter.target_exploration_std_temperature,.25);
  assert.equal(quarter.source_policy_contract.rho,.9);assert.equal(quarter.target_policy_contract.rho,.9);
  assert.deepEqual(saved.policy_contract,quarter.target_policy_contract);
  assert.equal(quarter.kernel_changed,true);assert.equal(quarter.nominal_control_changed,false);
  assert.equal(quarter.reward_changed,false);assert.equal(quarter.task_acceptance_changed,false);
  assert.equal(quarter.action_ranges_changed,false);assert.deepEqual(quarter.observation_semantics_changed,[]);
  assert.equal(quarter.deterministic_same_weights_same_observation,'exact_original_conditional_mean_path');
  for(const [name,binding]of Object.entries(quarter.configuration_bindings)){
   assert.equal(binding.source_sha256,binding.target_sha256);
   assert.deepEqual(saved.runtime_contract.selected_configuration[name],{path:binding.path,sha256:binding.target_sha256});
  }
  assert.equal(Object.keys(quarter.configuration_bindings).length,6);
 }else if(finalStop){
  assert.equal(finalStop.schema,'wlr50_clean.final_stop_handoff_same372_fix.v1');
  assert.equal(composition,undefined);
  assert.equal(migration.target_git_commit,saved.runtime_contract.source_git_commit);
  assert.equal(migration.target_runtime_content_sha256,saved.runtime_contract.runtime_content_sha256);
  const allowed=['semantic_supervisor','semantic_migration','semantic_training'].map(n=>'src/wlr50_clean/ppo/'+n+'.py');
  assert(migration.allowed_changed_files.includes('src/wlr50_clean/ppo/semantic_supervisor.py'));
  assert(migration.allowed_changed_files.every(p=>allowed.includes(p)));
  assert.deepEqual(Object.keys(finalStop.reviewed_code_sha256).sort(),[...migration.allowed_changed_files].sort());
  for(const [file,hash]of Object.entries(finalStop.reviewed_code_sha256)){
   assert.equal(saved.runtime_contract.files[file],hash);
   assert.equal(migration.changed_file_hashes[file].after,hash);
  }
  assert.equal(finalStop.supervisor_scope.method,'NominalMotionProvider._observe_final_stop_owner');
  assert.equal(finalStop.supervisor_scope.evaluator_and_other_methods_unchanged,true);
  assert.equal(finalStop.task_evaluator_changed,false);
  assert.equal(finalStop.fixed_post_completion_window_changed,false);
  assert.equal(finalStop.residual_composition_changed,false);
  assert.equal(finalStop.action_execution_changed,true);
  assert.equal(finalStop.reward_changed,false);
  assert.equal(finalStop.quality_epsilon,0);
  assert.deepEqual(finalStop.configuration_bindings,saved.runtime_contract.selected_configuration);
 }else if(saved.runtime_contract.source_git_commit!=='b0438f66ec635c31ea6afaf3b475d0ca19bd9604'){
  assert.equal(saved.runtime_contract.source_git_commit,'ad0c1328f1772c440755f3b6a6622c35e464396e');
  assert.equal(composition?.schema,'wlr50_clean.independent_post_mapper_residual_same372_fix.v1');
  assert.equal(migration.target_git_commit,saved.runtime_contract.source_git_commit);
  assert.equal(saved.runtime_contract.runtime_content_sha256,'4fbe469af3433096d241932e293caa9fcc9838abc02412d48d9a5bded8052b67');
  for(const [file,hash]of Object.entries(composition.reviewed_code_sha256)){
   assert.equal(saved.runtime_contract.files[file],hash);
   assert.equal(migration.changed_file_hashes[file].after,hash);
  }
  assert.equal(composition.action_execution_changed,true);
  assert.equal(composition.reward_changed,false);
  assert.equal(composition.quality_epsilon,0);
  assert.deepEqual(composition.configuration_bindings,saved.runtime_contract.selected_configuration);
 }
 return {schema:'wlr50_clean.task_first_completed_training_receipt.v1',
  created_at_utc:new Date().toISOString(),branch_id:branchId,
  branch_added_counts:{...r.increments},
  ancestry_source_counts:{global_policy_decisions:r.source.global_policy_decisions,
   ppo_updates:r.source.ppo_updates,optimizer_steps:r.source.optimizer_steps},
  latest_checkpoint_counts:{global_policy_decisions:r.saved.global_policy_decisions,
   ppo_updates:r.saved.ppo_updates,optimizer_steps:r.saved.optimizer_steps},
  ...(rr?{rr_branch_origin:rr.counter_origin,rr_branch_total_added_counts:saved.rr_task_branch_counts}:{}),
  reward_objective:'task_first_recovery_v1',quality_epsilon:0,
  preservation:{successful_N_source_unchanged:!rr,recorded_FSM_source_unchanged:true,physical_scene_and_actuator_capability_unchanged:true,
   action_execution_changed_at_migration_boundary:!!(directMigration?.execution_composition_factor||directMigration?.final_stop_handoff_factor||directMigration?.rr_physical_acceptance_same372_factor),
   stochastic_kernel_changed_at_migration_boundary:!!directMigration?.exploration_temperature_factor,
   reviewed_execution_boundary:rr?.schema??quarter?.schema??finalStop?.schema??composition?.schema??null,
   exploration_std_temperature:r.configuration.policy_contract.exploration_std_temperature,
   same_weights_deterministic_mean_changed_at_temperature_boundary:quarter?false:null,
   physical_MDP_unchanged_claimed:(rr||composition||finalStop)?false:null,full12_capacity_unchanged:true,
   old_rollout_not_inherited:directMigration?.discard_old_rollout_storage??null,
   verified_state_evidence:'sidecars and completed run receipts; no new checkpoint tensor load or hash pass'},
  training_is_not_deterministic_M1_M2_M3_proof:true,
  receipt:r};
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 const args=process.argv.slice(2);
 assert(args.length===4&&args[0]==='--run'&&args[2]==='--branch',
  'Usage: node summarize_recovery_training.mjs --run COMPLETED_RUN --branch BRANCH_ID');
 console.log(JSON.stringify(await completedReceipt(args[1],args[3]),null,2));
}

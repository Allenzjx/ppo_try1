// Small output-helper tests. Immutable OLD data only; no active runs or file mutation.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {extract} from './extract_p02_policy_recovery.mjs';
import {collect} from './summarize_training.mjs';
const source='runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260915T0011185025896Z_gdbc492f25c0a_ccb65744dd8f48dd94fbf44d3a1285b8/source';
const checkpoint='outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000168192.pt';
const expectedHead='dbc492f25c0a3f9713b8477b9ef4341159625a41';
const options={source,checkpoint,expectedHead,purpose:'fixture'};
const result=await extract(options);
assert.equal(result.purpose,'fixture');assert.equal(result.result_boundaries.current_round_task_success,null);
assert.equal(result.checkpoint.saved_global_policy_decisions,168192);assert.equal(result.endpoint.physics_tick,887);
assert.equal(result.endpoint.terminal_hard_joint_limit,true);assert.equal(result.result_boundaries.P02_to_P03_observed,false);
assert.equal(result.endpoint.first_remaining_task.leg,'FR');assert.equal(result.endpoint.first_remaining_task.next_event,'C');
assert.equal(result.policy_execution.unreturned_last_ticks_do_not_invent_a_policy_reward,7);
assert.equal(result.policy_execution.native_verified_ticks,887);assert(result.policy_execution.native_nonzero_actual_target_effect_ticks>0);
assert.equal(result.FR.history_event_ticks.active_lift,26);
const oldSummary=JSON.parse(fs.readFileSync(new URL('../ppo_timing_task_priority_v1/C_AFTER_FULL.summary.json',import.meta.url),'utf8'));
for(const joint of ['hip','knee']){
 const actual=result.RR.full_rate[joint].actual_minus_target_deg,prior=oldSummary.RR_tracking_peaks_over_full_run[joint];
 assert.equal(actual.max_abs_tick,prior.physics_tick);
 assert(Math.abs(actual.signed_at_max_abs-prior.signed_error_deg)<1e-8);
}
const training=await collect(['runs/ppo_fsm_reference_p09_stable_v2/train/20260914T2332031251091Z_gdbc492f25c0a_5f07ea0fc12a4428b87ec5191c70d209'],'fixture');
assert.equal(training.aggregate.added_policy_decisions,512);assert.equal(training.aggregate.added_ppo_updates,4);
assert.equal(training.aggregate.added_optimizer_steps,80);assert.equal(training.aggregate.teacher_prefix_decisions_excluded,0);
assert.equal(training.runs[0].state_handling.source_rng_state_recorded,true);
assert.equal(training.runs[0].state_handling.saved_rng_state_recorded,true);
await assert.rejects(extract({...options,purpose:'formal'}),/Old C fixture/);
await assert.rejects(extract({...options,expectedHead:'0'.repeat(40)}));
await assert.rejects(extract({...options,checkpoint:checkpoint.replace('168192','167936')}));
// Inject a read-only in-memory uncompleted-manifest counterexample; no disk writes.
const original=fs.readFileSync;
try{
 fs.readFileSync=function(file,...rest){const raw=original.call(this,file,...rest);
  if(String(file).endsWith('run_manifest.json')){const row=JSON.parse(String(raw));
   row.completed_at_utc=null;return JSON.stringify(row);}return raw;};
 await assert.rejects(extract(options),/Run is not completed/);
}finally{fs.readFileSync=original;}
console.log(JSON.stringify({fixture_only:true,positive_old_C_verified:true,
 completed_training_reuse_verified:'old512/4/80, not current round',RR_tracking_peaks_match_existing_extractor:true,
 negative_cases_passed:['old_as_formal','wrong_head','wrong_checkpoint','uncompleted_manifest'],
 no_production_or_original_evidence_writes:true}));

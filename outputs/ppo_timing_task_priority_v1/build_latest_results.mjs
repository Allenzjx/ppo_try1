// Assemble completed evidence only; not a task-success adjudicator.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url)), project=path.resolve(here,'../..');
const read=p=>JSON.parse(fs.readFileSync(path.resolve(here,p),'utf8').replace(/^\uFEFF/,''));
const load=p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,''));
const real=p=>fs.realpathSync(path.resolve(project,p));
const samePath=(a,b)=>assert.equal(real(a),real(b));
const hashes=['checkpoint_sha256','actor_parameter_sha256','critic_parameter_sha256','optimizer_state_sha256','normalizer_state_sha256'];
const counters=['global_policy_decisions','ppo_updates','optimizer_steps'];
const tri=v=>typeof v==='boolean'?v:null;

export function loadEvaluation(summary,role){
 const source=real(summary.source),run=load(path.join(path.dirname(source),'run_manifest.json'));
 assert(['SUCCEEDED','DIAGNOSTIC_FAILURE'].includes(run.lifecycle)&&run.completed_at_utc);
 const final=load(path.join(source,'semantic_video_source_manifest.json'));
 assert.equal(summary.lifecycle,run.lifecycle);assert.equal(summary.completed_at_utc,run.completed_at_utc);
 assert.equal(final.role,role);assert.equal(final.mode,role==='B'?'semantic_prior_eval':'semantic_residual_eval');
 assert.equal(run.optimizer_updates,0);assert.equal(final.optimizer_updates,0);assert.deepEqual(final.runtime_contract,run.runtime_contract);
 assert.equal(summary.source_git_commit,final.runtime_contract.source_git_commit);assert.equal(summary.runtime_content_sha256,final.runtime_contract.runtime_content_sha256);
 assert.equal(summary.end_tick,final.episode_physics_ticks);assert.equal(summary.native_verified_ticks,summary.end_tick);
 assert.equal(summary.physical_task_success,final.physical_task_success);assert.equal(typeof final.physical_task_success,'boolean');
 assert.equal(final.from_phase,'P01');assert.equal(final.episode_count,1);assert.equal(final.fresh_process_single_episode,true);
 assert.equal(final.pre_action_ticks,0);assert.equal(final.extra_pre_action_physics_ticks,0);
 const proof=final.natural_reset_proof,reset=proof?.reset_metadata,restoration=reset?.phase_snapshot_restoration;
 assert.equal(proof?.schema,'wlr50_clean.current_video_natural_reset.v1');assert.equal(proof.role,role);
 assert.deepEqual(proof.entry,{decision_count:0,done:false,physics_tick:0,state_id:'P01'});
 assert.equal(reset.execution_mode,'semantic_B_or_C');assert.equal(restoration?.mode,'semantic_natural_P01');
 assert.equal(restoration.requested_phase,'P01');assert.equal(restoration.historical_state_equality_required,false);assert.equal(restoration.policy_credit_excludes_settle,true);
 for(const k of ['snapshot_path','state_sha256','file_sha256','source_tick','physical_state'])assert(restoration[k]==null);
 const physical=final.physical_episode?.physical_task_evaluation;assert(physical&&physical.physics_tick===summary.end_tick);
 assert.equal(summary.physical_evaluator_termination_reason,physical.termination_reason??null);assert.equal(summary.physical_evaluator_termination_source,physical.termination_source??null);
 return{summary,source,run,final,physical};
}

export function firstRemainingTask(summary,p,terminalPhase){
 const events=Array.isArray(p.history?.lift_attempt_events)?p.history.lift_attempt_events:[];
 const legs=Object.fromEntries(['FR','FL','RR','RL'].map(leg=>{const c=p.current_legs?.[leg]??{};
  const initial=events.some(e=>e.leg===leg&&e.event==='whole_body_initial_clearance');
  return[leg,{I_initial_lift_observed:initial?true:tri(c.initial_lift_observed),
   I_evidence:initial?'recorded whole_body_initial_clearance event':typeof c.initial_lift_observed==='boolean'?'explicit current leg field':'unknown_not_recorded',
   Q_qualified_lift_history:tri(p.history?.active_lift?.[leg]),Q_established_current_attempt:tri(c.lift_established),current_lift_valid:tri(c.current_lift_valid),
   C_front_edge_crossed:tri(p.history?.front_edge_crossed?.[leg]),P_placed:tri(p.history?.placed?.[leg]),
   current_contact_surface:c.contact_surface??null,current_support:tri(c.support),current_bearing_verified:tri(c.bearing_verified)}];}));
 const base={terminal_phase:terminalPhase??'unknown',leg_order:['FR','FL','RR','RL'],leg_events:legs,
  physical_termination_reason:p.termination_reason??null,physical_termination_source:p.termination_source??null,
  qualification_scope:'Historical I/Q/C/P and current lift/support remain separate; missing evidence is unknown, not false'};
 if(summary.physical_task_success===true)return{...base,status:'complete',leg:null,next_event:null,task:'Common complete physical task accepted'};
 const keys={I:'I_initial_lift_observed',Q:'Q_qualified_lift_history',C:'C_front_edge_crossed',P:'P_placed'};
 const names={I:'initial measured lift with task preparation',Q:'qualified controllable lift',C:'cross the obstacle front edge',P:'controlled placement on top'};
 for(const leg of base.leg_order){const state=legs[leg];if(state.P_placed===true)continue;
  const next=state.C_front_edge_crossed===true?'P':state.Q_qualified_lift_history===true?'C':state.I_initial_lift_observed===true?'Q':'I';
  const value=state[keys[next]],unknown=value===null;
  return{...base,status:unknown?'unknown':'not_completed',leg,next_event:next,
   task:unknown?`${leg}: ${names[next]} evidence unknown`:`${leg}: ${names[next]} not completed`,next_event_value:value,placement_value:state.P_placed,
   additional_current_requirement:state.current_lift_valid===false&&state.Q_qualified_lift_history===true
    ?'Previously qualified lift is not currently valid; restore current controllable carry as needed before crossing/placement':null};}
 const traversal=tri(p.traversal_task_complete),controlled=tri(p.task_completed_controlled);
 return{...base,status:traversal===false||controlled===false?'not_completed':'unknown',leg:null,next_event:'whole_body_completion',
  task:traversal===false?'Whole-body traversal not completed':controlled===false?'Controlled final completion not established':'Final whole-body/post-completion acceptance unknown',
  traversal_task_complete:traversal,task_completed_controlled:controlled};
}

export function verifyBundle(training,metrics,pair,B,C,{requireRewardOnlyDelta=true}={}){
 assert(training.aggregate.runs_are_sequential_checkpoint_chain&&training.runs.length);
 const latest=training.runs.at(-1).saved,first=training.runs[0].source;
 counters.forEach(k=>{assert.equal(training.aggregate.final_counts[k],latest[k]);assert.equal(training.aggregate.source_counts[k],first[k]);});
 for(const[k,add]of [['global_policy_decisions','policy_decisions'],['ppo_updates','ppo_updates'],['optimizer_steps','optimizer_steps']]){
  assert.equal(latest[k]-first[k],training.runs.reduce((n,r)=>n+r.increments[add],0));assert.equal(training.aggregate[`added_${add}`],latest[k]-first[k]);}
 training.runs.forEach((r,i)=>{assert(['SUCCEEDED','STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'].includes(r.lifecycle));
  assert.equal(r.request.planned,r.request.consumed+r.request.unconsumed);assert.equal(r.increments.policy_decisions,r.request.consumed+r.request.rounding_overrun);
  if(i){counters.forEach(k=>assert.equal(r.source[k],training.runs[i-1].saved[k]));assert.equal(r.source.checkpoint_sha256,training.runs[i-1].saved.checkpoint_sha256);}});
 const checkpoint=real(latest.checkpoint),manifest=real(latest.manifest),history=real(path.join(project,'outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history'));
 assert.equal(path.dirname(checkpoint),history);assert.equal(path.basename(checkpoint),`checkpoint_step_${String(latest.global_policy_decisions).padStart(9,'0')}.pt`);
 assert.equal(manifest,checkpoint.replace(/\.pt$/,'_manifest.json'));const cp=load(manifest);samePath(cp.checkpoint_path,checkpoint);
 counters.forEach(k=>assert.equal(cp[k],latest[k]));hashes.forEach(k=>{assert.match(cp[k],/^[a-f0-9]{64}$/);assert.equal(cp[k],latest[k]);});assert.equal(cp.save_load_round_trip,true);
 const b=loadEvaluation(B,'B'),c=loadEvaluation(C,'C'),proof=c.final.checkpoint_load_provenance;
 assert.equal(b.final.checkpoint_load_provenance,null);assert.equal(proof?.checkpoint_loaded_and_verified,true);assert.equal(proof.official_load_semantic_checkpoint,true);assert.equal(proof.optimizer_updates,0);
 assert.equal(proof.saved_global_policy_decisions,cp.global_policy_decisions);assert.equal(C.checkpoint_decisions,cp.global_policy_decisions);
 samePath(proof.source.checkpoint,checkpoint);samePath(proof.source.manifest,manifest);assert.equal(proof.source.checkpoint_sha256,cp.checkpoint_sha256);assert.match(proof.source.manifest_sha256,/^[a-f0-9]{64}$/);
 assert.equal(proof.training_seed,cp.seed);for(const k of hashes.filter(k=>k!=='checkpoint_sha256'))assert.equal(proof.parameter_hashes[k],cp[k]);
 assert.equal(proof.observation_dimension,372);assert.equal(proof.policy_contract.raw_action_dimension,12);
 assert(pair.after_repair&&pair.initial_physical_state_equality?.all_initial_arrays_contacts_and_validity_exact_equal);assert.equal(pair.media_validity,'COMPLETE_PLAYABLE');assert.equal(pair.sources.length,2);
 if(requireRewardOnlyDelta){assert.equal(pair.runtime_match_policy,'explicit strict reward-only delta');assert.equal(pair.reward_only_runtime_delta_audit?.all_other_runtime_files_selected_configs_and_metadata_exact_equal,true);}
 else{assert.equal(pair.runtime_match_policy,'exact runtime/evaluation match');assert.equal(pair.reward_only_runtime_delta_audit,null);}
 const individuals=pair.sources.map(p=>load(real(p))),evaluations=[b,c];
 evaluations.forEach((ev,i)=>{const item=individuals[i],side=i===0?'left':'right',expectedManifest=path.join(ev.source,'semantic_video_source_manifest.json');
  samePath(item.source_manifest,expectedManifest);samePath(item.source,path.join(ev.source,'actual_viewport_video.mp4'));assert.equal(item.base_media_role,i===0?'B0':'C0');assert.equal(item.after_repair,true);assert.equal(item.media_validity,'COMPLETE_PLAYABLE');
  assert.equal(item.physical_ticks,ev.summary.end_tick);assert.equal(item.physical_task_success,ev.final.physical_task_success);assert.deepEqual(item.runtime_contract,ev.final.runtime_contract);
  assert.deepEqual(item.checkpoint_load_provenance,ev.final.checkpoint_load_provenance);assert.deepEqual(pair.paired_checkpoint_load_provenance[side],ev.final.checkpoint_load_provenance);
  assert.deepEqual(pair.paired_runtime_bindings[side],{source_git_commit:ev.summary.source_git_commit,runtime_content_sha256:ev.summary.runtime_content_sha256});
  const initial=pair.initial_physical_state_equality.evidence[i];samePath(initial.source_manifest,expectedManifest);assert.match(item.source_manifest_sha256,/^[a-f0-9]{64}$/);assert.equal(initial.source_manifest_sha256,item.source_manifest_sha256);
  assert.equal(item.terminal_phase,ev.summary.stage_transitions.at(-1)?.to_stage??'P01');});
 assert.equal(metrics.runs.length,2);const end=Math.min(B.end_tick,C.end_tick);assert.deepEqual(metrics.matched_window.inclusive_tick_range,[0,end]);assert.equal(metrics.matched_window.samples_per_run,end+1);
 evaluations.forEach((ev,i)=>{const m=metrics.runs[i];assert.equal(m.role,i===0?'B':'C');samePath(m.source,ev.source);assert.equal(m.source_git_commit,ev.summary.source_git_commit);assert.equal(m.runtime_content_sha256,ev.summary.runtime_content_sha256);
  assert.equal(m.full_episode.last_tick,ev.summary.end_tick);assert.equal(m.matched_common_window.last_tick,end);assert.equal(m.task.physical_task_success,ev.final.physical_task_success);assert.equal(m.task.checkpoint_decisions,ev.summary.checkpoint_decisions);
  assert.equal(m.task.termination_reason,ev.physical.termination_reason??null);assert.equal(m.task.termination_source,ev.physical.termination_source??null);});
 return{checkpoint,manifest,cp,b,c,individuals,first_remaining_task:{B:firstRemainingTask(B,b.physical,individuals[0].terminal_phase),C:firstRemainingTask(C,c.physical,individuals[1].terminal_phase)}};
}

function main(){
const training=read('training_total_actual.json'), metrics=read('final_stability_metrics.json');
const reference=read('reference_identity.json'), pair=read('videos/zero_vs_ppo_after.media.json');
const B=read('B_AFTER_FULL.summary.json'),C=read('C_AFTER_FULL.summary.json');
const checked=verifyBundle(training,metrics,pair,B,C),{checkpoint,manifest,cp}=checked;
const physicalB=checked.b.physical,physicalC=checked.c.physical;
const taskState=(s,p)=>({stage:s.stage_transitions.at(-1)?.to_stage??'P01',
 current_legs:Object.fromEntries(['FR','FL','RR','RL'].map(leg=>[leg,Object.fromEntries([
 'front_distance_m','clearance_m','ground_contact','contact_surface','current_lift_valid','bearing_force_n'
 ].map(k=>[k,p.current_legs?.[leg]?.[k]??null]))])),
 active_lift:p.history?.active_lift??null,front_edge_crossed:p.history?.front_edge_crossed??null,placed:p.history?.placed??null});
const compact=s=>({source:s.source,head:s.source_git_commit,runtime_sha:s.runtime_content_sha256,
 checkpoint_decisions:s.checkpoint_decisions,end_tick:s.end_tick,duration_s:s.duration_s,
 physical_task_success:s.physical_task_success,physical_evaluator_termination_reason:s.physical_evaluator_termination_reason,
 physical_evaluator_termination_source:s.physical_evaluator_termination_source,decision_phase_counts:s.decision_request_phase_counts,
 native_verified_ticks:s.native_verified_ticks,RR_events:s.RR_events,
 source_activation_events:s.source_activation_events,tracking_peaks:s.RR_tracking_peaks_over_full_run});
const payload={schema:'wlr50_clean.timing_task_priority_results.v1',completed_at_utc:new Date().toISOString(),
 request_scope:'Bounded timing/control repair, new zero/frozen/learned P01 attempts and finite real continuation; incomplete is not success',
 reference_identity:'reference_identity.json',original_A_preserved:true,A_original_and_physical_reclassification:reference.A_classification_boundary,
 code_commits:{source_partial_order:'a89ba82af0618ebaa1829f91d9f01b3e80b47cd2',TOP_continuation:B.source_git_commit,body_reward:C.source_git_commit},
 production_files:['src/wlr50_clean/ppo/semantic_supervisor.py','configs/ppo_fsm_reference_p09_stable_v2/stage_task_spec.yaml',
 'src/wlr50_clean/ppo/semantic_reward.py','configs/ppo_fsm_reference_p09_stable_v2/reward_config.yaml',
 'src/wlr50_clean/ppo/semantic_migration.py','src/wlr50_clean/ppo/semantic_training.py'].map(p=>path.join(project,p)),
 unchanged:['frozen A','robot/scene/physics/actuator gains and limits','HISTORY372/12 architecture','120Hz physics/15Hz policy',
 'all12 phase residual channels and existing caps','task acceptance and safety','controller/mapper histories at ordinary phase changes'],
 default_RL_assist_retained:true,RR_contact_impulse_backup_attempted:false,
 first_timing_pair:{B:compact(read('B_TIMING1_AFTER.summary.json')),C:compact(read('C_TIMING1_AFTER.summary.json')),
  B_task_terminal:'INCOMPLETE_CONTROLLER_BLOCKED at physics_tick9281, distinct from null physical safety termination',
  videos:['videos/zero_residual_timing_candidate1.mp4','videos/ppo_timing_only_after.mp4','videos/zero_vs_ppo_timing_candidate1.mp4']},
 latest_pair:{B:{...compact(B),terminal_task_state:taskState(B,physicalB)},C:{...compact(C),terminal_task_state:taskState(C,physicalC)},same_nominal:true,different_reward_versions:true,
  exact_measured_initial_state_match:true,natural_P01_whole_episode_proofs_verified:true,comparison_receipt:'videos/zero_vs_ppo_after.media.json'},
 training:training.aggregate,actual_training_details:'training_total_actual.json',
 training_runs:training.runs.map(r=>({run:r.run,lifecycle:r.lifecycle,request:r.request,increments:r.increments,task_results:r.task_results,stop_after_update:r.stop_after_update})),
 latest_checkpoint:checkpoint,latest_checkpoint_manifest:manifest,checkpoint_qualified_as_best:false,
 latest_checkpoint_binding:{...Object.fromEntries(hashes.map(k=>[k,cp[k]])),optimizer_steps:cp.optimizer_steps,
  eval_load_source:checked.c.final.checkpoint_load_provenance.source,verification:'Actual evaluation officially reloaded and verified this checkpoint; this output assembler cross-checks its recorded sidecar/load/media SHA and parameter hashes without an additional tensor reload or byte rehash'},
 migration:{source_checkpoint:reference.checkpoint_before_training,plan:'checkpoint167424_body_reward_migration.json',
  optimizer:'complete Adam and effective LR retained',normalizers:'identity state retained',old_rollout:'discarded; new on-policy data collected'},
 metrics:'final_stability_metrics.json',matched_window:metrics.matched_window,
 first_remaining_task:checked.first_remaining_task,
 attribution_limits:['B1 first timing repair earned RR lift but not crossing/placement',
 'B2 TOP continuation advanced about10mm but existing fixed-base nominal geometry reached knee limit',
 'C1 early target/actual departure is distinct from B2 zero-policy geometry correction',
 'The body allowance conflict is confirmed, but C1 returned P02 weights were already protected',
 'Finite continuation is real learning; no isolated old-reward training counterfactual or stability-superiority claim'],
 both_latest_runs_successful:B.physical_task_success===true&&C.physical_task_success===true,stability_superiority_claim:false,
 outputs:['videos/zero_residual_after.mp4','videos/ppo_after.mp4','videos/zero_vs_ppo_after.mp4','sequence_before_after.csv',
 'rr_target_actual_and_task_trace.csv','reward_conflict_and_changes.md','reference_identity.json','RECOVERY.md']};
const dest=path.join(here,'latest_run_results.json');
fs.writeFileSync(dest,JSON.stringify(payload,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({output:dest,checkpoint,training:training.aggregate,latest_B:compact(B),latest_C:compact(C)}));
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))main();

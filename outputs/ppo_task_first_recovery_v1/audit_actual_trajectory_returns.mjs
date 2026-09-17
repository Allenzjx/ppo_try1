// Read-only reward accounting of complete existing real trajectories.
// No simulator, policy, critic, source rewrite, or fabricated continuation.
import fs from 'node:fs';
import readline from 'node:readline';
import path from 'node:path';
const root = process.cwd();
const gamma = .9985, potentialWeight = 5, timeCost = .02;
const base = 'runs/ppo_fsm_reference_p09_stable_v2';
const sources = [
 ['successful_zero', `${base}/video_eval/prior_B/20260915T0525295619176Z_g4a58c0190ef7_615f8fe3cfc64cff81f73ac6a0d302e6/source/video_policy_decisions.jsonl`, 'video'],
 ['C174592_early_hard_limit', `${base}/video_eval/validation/20260915T0710472104000Z_g4a58c0190ef7_253ae24ad12c4830ad3e3ae5316ac659/source/video_policy_decisions.jsonl`, 'video'],
 ['sampled_training_172544_to_174592', `${base}/train/20260915T0633266376940Z_g4a58c0190ef7_d19a81607e6044a281a010de29469635/residual_and_projection_audit.jsonl`, 'training'],
];
function fresh(){return {n:0,dt:0,old:0,new:0,potential:0,time:0,event:0,families:{},phase:{},first:null,last:null,qualityUndiscounted:0,maxFormulaError:0,maxContinuityError:0,frCapture:null,derivedEndpoint:null,qualityUnknownBound:0};}
function summarize(s, terminal) {
 const first=s.first, last=s.last;
 const telescope=potentialWeight*(gamma**s.n*last.reward.potential_after-first.reward.potential_before);
 return {decisions:s.n, duration_s:s.dt, start_tick:first.physics_tick-first.physics_ticks,
  end_tick:last.physics_tick, first_global_decision:first.global??null,last_global_decision:last.global??null,
  policy_update_range_before_sampling:first.global == null ? null : [1313+Math.floor((first.global-172545)/128),1313+Math.floor((last.global-172545)/128)],
  terminal:terminal,termination_reason:last.termination_reason,terminal_bootstrap_allowed:last.reward.terminal_bootstrap_allowed,
  phase_counts:s.phase, discounted_old_total:s.derivedEndpoint?null:s.old,
  discounted_old_total_bounds:[s.old-s.qualityUnknownBound,s.old],discounted_task_first_total:s.new,
  discounted_old_quality:s.derivedEndpoint?null:s.old-s.new,
  discounted_old_quality_bounds:[s.old-s.new-s.qualityUnknownBound,s.old-s.new],
  undiscounted_old_quality:s.derivedEndpoint?null:s.qualityUndiscounted,
  discounted_families:s.families,discounted_potential:s.potential,discounted_time:s.time,
  discounted_terminal_event:s.event,terminal_event_undiscounted:last.reward.terminal_event,
  terminal_event_discount:gamma**(s.n-1),potential_initial:first.reward.potential_before,
  potential_final:last.reward.potential_after,potential_telescoping_expected:telescope,
  potential_telescoping_abs_error:Math.abs(s.potential-telescope),
  max_reward_formula_abs_error:s.maxFormulaError,max_potential_continuity_error:s.maxContinuityError,
  FR_capture_prefix:s.frCapture,
  observed_task_event_ticks:last.semantic_task?.physical_evaluator?.history?.event_ticks??null,
  derived_video_terminal_accounting:s.derivedEndpoint,
  nonterminal_tail_caution:terminal?null:'Observed finite prefix only. No bootstrap/value or terminal event fabricated; not a full-episode return.'};
}
const output={schema:'wlr50_clean.task_first_actual_trajectory_returns.v1',
 convention:'G=sum(gamma**t*r_t), one gamma per issued policy decision; actual physical dt costs; all real task/safety terminals zero Phi_next/no bootstrap; ordinary nonterminal tails retain Phi_next and are NOT completed returns.',
 new_objective_revaluation:'Same actually observed trajectory, existing task_progress family retained, four quality family weights zero. This is offline objective accounting, not a new on-policy rollout/critic/GAE or a simulated counterfactual trajectory.',
 gamma,potential_weight:potentialWeight,time_cost_per_s:timeCost,sources:[]};
for (const [name,relative,kind] of sources){
 let state=fresh(), episodes=[];
 const absolute=path.join(root,relative);
 for await(const line of readline.createInterface({input:fs.createReadStream(absolute),crlfDelay:Infinity})){
  if(!line.trim())continue;
  const raw=JSON.parse(line);
  let a=kind==='video'?raw.step_info:raw.applied_audit;
  if(!a?.reward){
   if(kind!=='video'||raw.environment_step_returned!==false||!state.last)throw new Error(`Missing saved actual reward in ${name}`);
   const manifestPath=path.join(path.dirname(absolute),'semantic_video_source_manifest.json');
   const manifest=JSON.parse(fs.readFileSync(manifestPath,'utf8'));
   const physical=manifest.physical_episode.physical_task_evaluation;
   const reason=physical.success?'SUCCESS':physical.termination_reason;
   if(physical.physics_tick!==raw.end_tick||reason!==raw.stop_reason)throw new Error('Video terminal physical mismatch');
   const dt=raw.physics_ticks/120,phi=state.last.reward.potential_after;
   const event=physical.success?40:-40, shaping=-potentialWeight*phi;
   const task=shaping+event-timeCost*dt;
   a={physics_tick:raw.end_tick,physics_ticks:raw.physics_ticks,phase_id:raw.request_phase,
    termination_reason:reason,semantic_task:{physical_evaluator:physical},
    reward:{total:task,families:{task_progress:task},potential_before:phi,potential_after:0,
     potential_shaping:shaping,terminal_event:event,elapsed_physics_s:dt,terminal_bootstrap_allowed:false}};
   state.qualityUnknownBound=gamma**state.n*.7*dt;
   state.derivedEndpoint={source:manifestPath,actual_stop_reason:reason,actual_final_tick:raw.end_tick,
    stored_reward:false,explanation:'Observer exited before env reward returned. Only task terminal contribution revalued exactly from actual physical terminal, previous stored Phi, and actual partial dt. Missing old quality integral is bounded, not invented.',
    missing_quality_undiscounted_bounds:[-.7*dt,0],task_first_terminal_reward:task};
  }
  a.global=raw.global_policy_decision;
  const r=a.reward,d=gamma**state.n;
  if(state.first===null)state.first=a;
  if(state.last)state.maxContinuityError=Math.max(state.maxContinuityError,Math.abs(state.last.reward.potential_after-r.potential_before));
  const expected=potentialWeight*(gamma*r.potential_after-r.potential_before)+r.terminal_event-timeCost*r.elapsed_physics_s;
  state.maxFormulaError=Math.max(state.maxFormulaError,Math.abs(expected-r.families.task_progress));
  state.old+=d*r.total;state.new+=d*r.families.task_progress;
  state.potential+=d*r.potential_shaping;state.time-=d*timeCost*r.elapsed_physics_s;state.event+=d*r.terminal_event;
  state.qualityUndiscounted+=r.total-r.families.task_progress;
  for(const [f,v] of Object.entries(r.families))state.families[f]=(state.families[f]??0)+d*v;
  state.phase[a.phase_id]=(state.phase[a.phase_id]??0)+1;
  state.n++;state.dt+=r.elapsed_physics_s;state.last=a;
  if(!state.frCapture && a.semantic_task?.physical_evaluator?.history?.placed?.FR){
   state.frCapture={through_decision:state.n,end_tick:a.physics_tick,elapsed_s:state.dt,
    global_policy_decision:a.global??null,discounted_old_prefix:state.old,discounted_task_first_prefix:state.new,
    phi_after:r.potential_after,terminal:false,bootstrap_value:'not recomputed; this prefix is not a completed-episode return'};
  }
  if(a.termination_reason){episodes.push(summarize(state,true));state=fresh();}
 }
 if(state.n)episodes.push(summarize(state,false));
 output.sources.push({name,source:absolute,episodes});
}
console.log(JSON.stringify(output,null,2));

// One supplementary pass over a completed new-objective run. No model/tensor load.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import assert from 'node:assert/strict';
const run=path.resolve(process.argv[2]);
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const train=read(path.join(run,'training_manifest.json'));
assert(['SUCCEEDED','STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'].includes(train.lifecycle));
const sums={},episodes=[];let count=0,full12=0,eps0=0,native=0,ticks=0,realEffect=0,episode=[];
const families=['body_stability','contact_motion_quality','control_smoothness','control_regularization'];
const wheels=['FL','FR','RL','RR'];
const wheelRows=Object.fromEntries(wheels.map(w=>[w,{nominal_positive_samples:0,final_reversed:0,final_near_zero:0,final_enhanced:0,final_sum:0}]));
function finish(rows,terminal){
 const last=rows.at(-1),a=last.applied_audit,r=a.reward;
 return {decisions:rows.length,first_global_decision:rows[0].global_policy_decision,last_global_decision:last.global_policy_decision,
  duration_s:rows.reduce((s,x)=>s+x.applied_audit.reward.elapsed_physics_s,0),terminal,
  end_tick:a.physics_tick,end_phase:a.end_phase_id,reason:a.termination_reason,
  bootstrap_allowed:r.terminal_bootstrap_allowed,
  real_event_ticks:a.semantic_task.physical_evaluator.history.event_ticks,
  discounted_observed_reward:rows.reduce((s,x,i)=>s+Math.pow(.9985,i)*x.applied_audit.reward.total,0),
  is_complete_return:terminal,nonterminal_caution:terminal?null:'Observed prefix only; no terminal or critic bootstrap fabricated.'};
}
for await(const line of readline.createInterface({input:fs.createReadStream(path.join(run,'residual_and_projection_audit.jsonl')),crlfDelay:Infinity})){
 if(!line.trim())continue;
 const row=JSON.parse(line),a=row.applied_audit,r=a.reward;
 count++;episode.push(row);
 full12+=Number(a.actuator_target_effect_audit.phase_mask_full12.length===12&&a.actuator_target_effect_audit.phase_mask_full12.every(x=>x===1));
 eps0+=Number(r.objective_profile==='task_first_recovery_v1'&&r.quality_epsilon===0&&families.every(k=>r.families[k]===0)&&r.total===r.families.task_progress);
 native+=a.actuator_target_effect_audit_summary.verified_tick_count;
 ticks+=a.physics_ticks;realEffect+=a.actuator_target_effect_audit_summary.actual_native_effect_tick_count;
 const s=sums[a.phase_id]??={count:0,total:0,potential:0,time:0,terminal:0,positive:0,negative:0};
 s.count++;s.total+=r.total;s.potential+=r.potential_shaping;s.time-=.02*r.elapsed_physics_s;s.terminal+=r.terminal_event;s.positive+=Number(r.total>0);s.negative+=Number(r.total<0);
 if(a.phase_id==='P02')for(let j=0;j<4;j++){
  const n=a.nominal_action_full12[8+j],f=a.actual_drive_target_full12[8+j],w=wheelRows[wheels[j]];
  if(n>.05){w.nominal_positive_samples++;w.final_reversed+=Number(f<0);w.final_near_zero+=Number(Math.abs(f)<.05);w.final_enhanced+=Number(f>n);w.final_sum+=f;}
 }
 if(row.terminal){episodes.push(finish(episode,true));episode=[];}
}
if(episode.length)episodes.push(finish(episode,false));
assert.equal(count,train.actual_policy_decisions);
const gae={};let ordinaryTransitions=0,ordinaryDone=0,terminalCount=0;
function addStat(dst,src){
 for(const k of ['count','positive_count','negative_count','zero_count','nonfinite_count'])dst[k]=(dst[k]??0)+src[k];
 dst.sum=(dst.sum??0)+src.mean*src.count;
 dst.minimum=Math.min(dst.minimum??Infinity,src.minimum);dst.maximum=Math.max(dst.maximum??-Infinity,src.maximum);
}
for(const line of fs.readFileSync(path.join(run,'advantage_audit.jsonl'),'utf8').trim().split('\n')){
 const row=JSON.parse(line);terminalCount+=row.terminal_samples.length;
 for(const x of row.ordinary_phase_change_samples){ordinaryTransitions++;ordinaryDone+=Number(x.done===true||x.terminal===true);}
 for(const [phase,v]of Object.entries(row.by_request_phase)){
  gae[phase]??={};for(const key of ['raw_gae_returns_minus_old_values','stored_advantages','old_values','returns']){gae[phase][key]??={};addStat(gae[phase][key],v[key]);}
 }
}
for(const groups of Object.values(gae))for(const g of Object.values(groups)){g.mean=g.sum/g.count;delete g.sum;}
for(const w of Object.values(wheelRows)){w.final_mean=w.nominal_positive_samples?w.final_sum/w.nominal_positive_samples:null;delete w.final_sum;}
console.log(JSON.stringify({schema:'wlr50_clean.task_first_completed_reward_signal.v1',run,
 objective_runtime:{decisions:count,epsilon0_task_only_decisions:eps0,full12_permission_decisions:full12,
  actual_physics_ticks:ticks,verified_native_dispatch_ticks:native,actual_native_effect_ticks:realEffect},
 by_request_phase:sums,episodes,original_GAE_summary:gae,
 phase_handoffs:{ordinary_phase_changes:ordinaryTransitions,incorrect_done_observed:ordinaryDone,real_terminal_samples:terminalCount},
 p02_positive_nominal_final_targets:wheelRows,
 diagnostic_thresholds:'N>.05, |final|<.05 rad/s are descriptive bins, not a new controller/reward gate; wheel final vectors are canonical forward-positive. This is sampled closed-loop training, not formal deterministic behavior.',
 evidence_limits:'Supplementary pass after complete receipt. No recalculated critic/GAE, no model forward/update, no formal milestone or stability claim.'},null,2));

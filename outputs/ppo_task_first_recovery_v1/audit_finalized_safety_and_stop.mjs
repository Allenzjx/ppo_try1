// Sealed-run, read-only physical safety and stop-owner extraction. No new physics.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import assert from 'node:assert/strict';
const run=path.resolve(process.argv[2]);
const manifest=JSON.parse(fs.readFileSync(path.join(run,'run_manifest.json'),'utf8'));
assert(manifest.completed_at_utc);
const result=manifest.result, evaluation=result.physical_episode.physical_task_evaluation;
const each=async(name,fn)=>{for await(const line of readline.createInterface({input:fs.createReadStream(path.join(run,'source',name)),crlfDelay:Infinity}))if(line.trim())fn(JSON.parse(line));};
let observations=0,bodyCollisions=0,bodyPairs=0,nonfinite=0,hardLimits=0,minMargin=Infinity,lastPhysical;
await each('physical_observations.jsonl',row=>{
 observations++;bodyCollisions+=Number(row.body_collision.detected);bodyPairs+=Number(row.body_collision.real_pair_active);nonfinite+=Number(row.all_finite!==true);
 for(const [name,joint]of Object.entries(row.joints)){
  const range=name.endsWith('_hip')?[-135,135]:[-60,210];
  const margin=Math.min(joint.position_deg-range[0],range[1]-joint.position_deg);
  minMargin=Math.min(minMargin,margin);hardLimits+=Number(margin<0);
 }
 lastPhysical=row;
});
assert.equal(observations,result.episode_physics_ticks+1);
let ownerFirst=null,ownerLast=null,ownerActiveRecords=0;
await each('video_policy_decisions.jsonl',row=>{
 const owner=row.step_info?.semantic_task?.nominal_provider_diagnostics?.final_stop_owner;
 if(!owner?.active)return;
 ownerActiveRecords++;
 const entry={decision:row.decision,start_tick:row.start_tick,end_tick:row.end_tick,owner};
 ownerFirst??=entry;ownerLast=entry;
});
let firstP13ZeroNominal=null,lastNative=null;const phaseTicks={};
await each('native_tick_audit.jsonl',row=>{
 phaseTicks[row.source_phase_id]=(phaseTicks[row.source_phase_id]??0)+1;
 if(row.source_phase_id==='P13'&&row.nominal_full12.slice(8).every(v=>v===0))firstP13ZeroNominal??={episode_tick:row.episode_physics_tick,nominal:row.nominal_full12};
 lastNative=row;
});
const current=Object.fromEntries(Object.entries(evaluation).filter(([key])=>!['history','transfer_roles','current_legs','body_traversal_geometry'].includes(key)));
console.log(JSON.stringify({schema:'wlr50_clean.finalized_safety_and_stop_audit.v1',run,
 physical_safety:{observations_including_tick0:observations,body_collision_observations:bodyCollisions,exact_body_obstacle_pair_active_observations:bodyPairs,nonfinite_observations:nonfinite,actual_joint_hard_limit_violations:hardLimits,minimum_actual_joint_limit_margin_deg:minMargin},
 phase_physical_ticks:phaseTicks,final_task:current,first_P13_zero_nominal:firstP13ZeroNominal,
 owner_first_record:ownerFirst,owner_last_record:ownerLast,owner_active_returned_records:ownerActiveRecords,
 final_command:{canonical_order:lastNative.native_audit.canonical_order,nominal_full12:lastNative.nominal_full12,policy_effective_residual_full12:lastNative.projected_residual_full12,phase_mask_full12:lastNative.native_audit.phase_mask_full12,raw_policy_action_full12:lastNative.native_audit.raw_policy_action_full12},
 final_base:lastPhysical.base,final_joints:lastPhysical.joints,
 claim_scope:'This sealed episode only. Current controlled-task acceptance is not exact-zero wheel speed, historical-home recovery, deployment safety or generalized stability.'}));

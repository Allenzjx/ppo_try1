// Read-only, bounded post-finalization evidence extraction. No simulation or mutations.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import assert from 'node:assert/strict';
const run=path.resolve(process.argv[2]);
const m=JSON.parse(fs.readFileSync(path.join(run,'run_manifest.json'),'utf8'));
assert(m.completed_at_utc&&['SUCCEEDED','DIAGNOSTIC_FAILURE'].includes(m.lifecycle));
const r=m.result, f=r.physical_episode.physical_task_evaluation;
const read=async(file,fn)=>{for await(const line of readline.createInterface({input:fs.createReadStream(path.join(run,'source',file)),crlfDelay:Infinity})){if(line.trim())fn(JSON.parse(line));}};
const legs={FL:'front_left',FR:'front_right',RL:'rear_left',RR:'rear_right'};
const wanted=new Set([0]);
// Capture this sealed run's real event ticks and terminal state, never assume old event timing.
for(const events of Object.values(f.history?.event_ticks??{}))for(const tick of Object.values(events)){
 wanted.add(tick);if(tick+1<=r.episode_physics_ticks)wanted.add(tick+1);
}
wanted.add(r.episode_physics_ticks);
const physical=[];
await read('physical_observations.jsonl',x=>{if(!wanted.has(x.physics_tick))return;physical.push({
 tick:x.physics_tick,time_s:x.simulation_time_s,base_position_w_m:x.base.position_w_m,
 base_velocity_w_m_s:x.base.linear_velocity_w_m_s,body_collision:x.body_collision,
 body_collider_bounds:x.body_bounds_w_m.base_link,legs:Object.fromEntries(Object.entries(legs).map(([leg,n])=>{
 const w=x.wheels[n+'_ankle'],c=x.contacts[n+'_wheel'];
 return [leg,{center_w_m:w.center_w_m,bottom_w_m:w.bottom_w_m,
 clearance_above_top_m:w.bottom_w_m[2]-x.obstacle.top_z_m,
 center_front_offset_m:w.center_w_m[0]-x.obstacle.front_x_m,
 command_rad_s:w.command_rad_s,velocity_rad_s:w.velocity_rad_s,contact_class:c.contact_class,
 obstacle_contact:{active:c.obstacle.active,pair_verified:c.obstacle.pair_verified,normal_force_n:c.obstacle.normal_force_n,force_w_n:c.obstacle.force_w_n,contact_point_w_m:c.obstacle.contact_point_w_m},
 ground_contact:{active:c.ground.active,normal_force_n:c.ground.normal_force_n}}];})),
 RR_knee:x.joints.rear_right_knee});});
let ticks=0, full12=0, verified=0,effect=0,zeroRaw=0,forbidden=0;
await read('native_tick_audit.jsonl',x=>{const a=x.native_audit;ticks++;full12+=Number(a.phase_mask_full12?.length===12&&a.phase_mask_full12.every(v=>v===1));verified+=Number(a.verified===true);effect+=Number(a.changed_target_channel_count>0);zeroRaw+=Number(a.raw_policy_action_full12.every(v=>v===0));for(const k of ['in_episode_root_pose_writes','in_episode_root_velocity_writes','in_episode_force_or_impulse_writes','in_episode_gravity_writes'])forbidden+=x[k]??0;});
let decisions=0,returned=0,epsilon0=0,last=null;const phase={};
await read('video_policy_decisions.jsonl',x=>{decisions++;phase[x.request_phase]=(phase[x.request_phase]??0)+1;returned+=Number(x.environment_step_returned);epsilon0+=Number(x.step_info?.reward_breakdown?.quality_epsilon===0&&x.step_info?.reward_breakdown?.objective_profile==='task_first_recovery_v1');last=x;});
assert(ticks===r.episode_physics_ticks&&full12===ticks&&verified===ticks&&effect===ticks&&forbidden===0);
console.log(JSON.stringify({schema:'wlr50_clean.finalized_task_recovery_evaluation.v1',run,lifecycle:m.lifecycle,completed_at_utc:m.completed_at_utc,
 checkpoint_load_provenance:r.checkpoint_load_provenance,seed:r.seed,role:r.role,mode:r.mode,from_phase:r.from_phase,
 episode_count:r.episode_count,optimizer_updates:r.optimizer_updates,natural_entry:r.natural_reset_proof,pre_action_ticks:r.pre_action_ticks,extra_pre_action_physics_ticks:r.extra_pre_action_physics_ticks,
 task_success:f.success,physical_evaluation_valid:f.valid,ticks:r.episode_physics_ticks,duration_s:r.physical_episode.physical_task_duration_s,
 termination_reason:f.termination_reason,reason:f.reason,full12_permission_ticks:full12,native_verified_ticks:verified,learning_effect_ticks:effect,all_zero_raw_ticks:zeroRaw,forbidden_state_write_counter_sum:forbidden,
 issued_policy_decisions:decisions,returned_environment_steps:returned,epsilon0_task_only_reward_records:epsilon0,phase_requested_decisions:phase,
 actual_event_ticks:f.history?.event_ticks,current_legs:f.current_legs,final_body_traversal_geometry:f.body_traversal_geometry,
 selected_actual_physical_observations:physical,last_decision:{decision:last.decision,request_phase:last.request_phase,start_tick:last.start_tick,end_tick:last.end_tick,environment_step_returned:last.environment_step_returned,termination_reason:last.step_info?.termination_reason},
 source_video:path.join(run,'source','actual_viewport_video.mp4'),teacher_prefix_decisions:0,
 teacher_evidence:'Natural entry at P01/tick0; pre-action ticks0; official deterministic checkpoint load; every actual tick full12/native verified/nonzero raw, no forbidden state writes.'},null,process.argv.includes('--compact')?0:2));

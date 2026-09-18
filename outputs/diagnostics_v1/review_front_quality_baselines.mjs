// Read only the two named sealed runs, stopping both streams after the front window.
// Output-only descriptive diagnostics; no task reclassification, PPO or reward changes.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const sources=[
 {label:'zero_refine_v1_sealed',run:'runs/ppo_non_residual_refine_v1/video_eval/prior_B/20260917T0343245466199Z_g3d231897c91c_0e58ddffca37441089dd17f34f4ed5d8',review:'Final one-second controlled-stop check failed; not the protected older successful zero.'},
 {label:'historical_CP177152',run:'runs/ppo_task_first_recovery_v1/video_eval/validation/20260916T0550249947408Z_g00050a2b1452_086805789a6e491392ef4656c7708c2c',review:'Historical evaluator success retained; RR_ACCEPTANCE_UNDER_REVIEW, not latest CP177792 or new strict RR success.'},
];
const legName={FR:'front_right',FL:'front_left',RL:'rear_left',RR:'rear_right'};
const norm=v=>Math.hypot(...v), dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const q95=a=>{const s=[...a].sort((a,b)=>a-b);return s[Math.max(0,Math.ceil(s.length*.95)-1)];};
function stats(values){
 const a=values.filter(Number.isFinite);
 return a.length?{n:a.length,min:Math.min(...a),max:Math.max(...a),mean:a.reduce((s,x)=>s+x,0)/a.length,
  rms:Math.sqrt(a.reduce((s,x)=>s+x*x,0)/a.length),abs_p95:q95(a.map(Math.abs)),peak_to_peak:Math.max(...a)-Math.min(...a)}:null;
}
function rpy(q){
 const length=norm(q),[w,x,y,z]=q.map(v=>v/length);
 return [Math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),Math.asin(Math.max(-1,Math.min(1,2*(w*y-z*x)))),Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))];
}
const wheel=(row,leg)=>row.wheels[`${legName[leg]}_ankle`];
const contact=(row,leg)=>row.contacts[`${legName[leg]}_wheel`];
function measuredTopPair(row,leg){
 // Same existing physical_contact_surface TOP test, not history. Fixed old spec: tolerance .005m, force floor .2N.
 const pair=contact(row,leg).obstacle,p=pair.contact_point_w_m,o=row.obstacle,f=pair.force_w_n;
 const up=Math.max(0,f[2]),h=Math.hypot(f[0],f[1]);
 return pair.pair_verified===true&&pair.active===true&&Array.isArray(p)&&p.every(Number.isFinite)
  &&p[0]>=o.front_x_m-.005&&p[0]<=o.back_x_m+.005&&p[1]>=o.right_y_m-.005&&p[1]<=o.left_y_m+.005
  &&Math.abs(p[2]-o.top_z_m)<=.005&&up>=.2&&up>h;
}
async function prefixRows(file,lastTick,tickOf,consume){
 let n=0;const stream=fs.createReadStream(file),lines=readline.createInterface({input:stream,crlfDelay:Infinity});
 try {for await(const line of lines){if(!line.trim())continue;const row=JSON.parse(line);if(tickOf(row)>lastTick)break;consume(row);n++;}}
 finally{lines.close();stream.destroy();}
 return n;
}
function physicalWindow(rows,start,end){
 const a=rows.filter(r=>r.physics_tick>=start&&r.physics_tick<=end);
 assert.equal(a.length,end-start+1);
 const first=a[0],last=a.at(-1),att=a.map(r=>rpy(r.base.orientation_wxyz));
 const delta=sub(wheel(first,'RL').center_w_m,first.center_of_mass.position_w_m).slice(0,2);
 const direction=delta.map(x=>x/norm(delta));
 const c0=first.center_of_mass.position_w_m;
 const toward=a.map(r=>dot(sub(r.center_of_mass.position_w_m,c0).slice(0,2),direction));
 const distances=a.map(r=>norm(sub(wheel(r,'RL').center_w_m,r.center_of_mass.position_w_m).slice(0,2)));
 const both=a.filter(r=>measuredTopPair(r,'FR')&&measuredTopPair(r,'FL'));
 const rate=[[],[]];
 for(let i=1;i<a.length;i++)for(let axis=0;axis<2;axis++){
  const d=att[i][axis]-att[i-1][axis];rate[axis].push(Math.atan2(Math.sin(d),Math.cos(d))/(a[i].simulation_time_s-a[i-1].simulation_time_s));
 }
 return {start_tick:start,end_tick:end,start_s:first.simulation_time_s,end_s:last.simulation_time_s,
  duration_s:last.simulation_time_s-first.simulation_time_s,physical_observations:a.length,
  attitude:{roll_rad:stats(att.map(q=>q[0])),pitch_rad:stats(att.map(q=>q[1])),
   roll_rate_rad_s:stats(rate[0]),pitch_rate_rad_s:stats(rate[1]),body_angular_speed_rad_s:stats(a.map(r=>norm(r.base.angular_velocity_w_rad_s)))},
  com_to_RL:{fixed_direction_xy_at_window_start:direction,
   com_world_displacement_m:sub(last.center_of_mass.position_w_m,c0),
   RL_world_displacement_m:sub(wheel(last,'RL').center_w_m,wheel(first,'RL').center_w_m),
   com_toward_initial_RL_direction_m:stats(toward),end_projection_m:toward.at(-1),
   instantaneous_com_to_RL_xy_distance_m:stats(distances),distance_change_m:distances.at(-1)-distances[0]},
  front_geometry:{signed_FR_minus_FL_wheel_center_z_m:stats(a.map(r=>wheel(r,'FR').center_w_m[2]-wheel(r,'FL').center_w_m[2])),
   FR_bottom_minus_obstacle_top_m:stats(a.map(r=>wheel(r,'FR').bottom_w_m[2]-r.obstacle.top_z_m)),
   FL_bottom_minus_obstacle_top_m:stats(a.map(r=>wheel(r,'FL').bottom_w_m[2]-r.obstacle.top_z_m)),
   simultaneous_measured_top_bearing_rows:both.length,total_rows:a.length,
   signed_FR_minus_FL_actual_top_contact_point_z_m:stats(both.map(r=>contact(r,'FR').obstacle.contact_point_w_m[2]-contact(r,'FL').obstacle.contact_point_w_m[2])),
   both_top_bearing_wheel_center_z_difference_m:stats(both.map(r=>wheel(r,'FR').center_w_m[2]-wheel(r,'FL').center_w_m[2]))},
  quality_flags:{all_physical_rows_finite:a.every(r=>r.all_finite),all_com_valid:a.every(r=>r.center_of_mass.valid),
   all_front_wheel_geometry_verified:a.every(r=>wheel(r,'FR').geometry_verified&&wheel(r,'FL').geometry_verified),
   body_collision_rows:a.filter(r=>r.body_collision.detected).length}};
}
function recordedWindow(rows,start,end){
 const a=rows.filter(r=>r.end_tick>=start&&r.end_tick<=end);
 const legs=a.map(r=>r.step_info.semantic_task.physical_evaluator.current_legs);
 const role=a.map(r=>r.step_info.semantic_task.transfer_roles.FR);
 return {returned_decision_samples:a.length,first_tick:a[0]?.end_tick??null,last_tick:a.at(-1)?.end_tick??null,
  scope:'Existing recorded evaluator/transfer diagnostics at returned decision endpoints; not interpolated physics or causal proof.',
  FR_air_samples:legs.filter(l=>l.FR.air).length,
  RL_current_bearing_support_samples:legs.filter(l=>l.RL.bearing_verified&&l.RL.support&&l.RL.bearing_force_n>=.2).length,
  RL_air_samples:legs.filter(l=>l.RL.air).length,
  RL_bearing_force_n:stats(legs.map(l=>l.RL.bearing_force_n)),RL_load_fraction:stats(legs.map(l=>l.RL.load_fraction_valid?l.RL.load_fraction:null)),
  both_front_recorded_TOP_bearing_samples:legs.filter(l=>['FR','FL'].every(k=>l[k].contact_surface==='TOP'&&l[k].top_surface_contact&&l[k].bearing_verified&&l[k].bearing_force_n>=.2)).length,
  recorded_FR_role_local_window_com_toward_RL_m:stats(role.filter(r=>r.valid).map(r=>r.transfer_direction_context?.com_toward_receiver_m)),
  recorded_FR_role_local_window_FR_load_drop:stats(role.filter(r=>r.valid).map(r=>r.transfer_direction_context?.load_fraction_change))};
}
const result=[];
for(const item of sources){
 const run=path.join(root,item.run),manifest=JSON.parse(fs.readFileSync(path.join(run,'run_manifest.json'),'utf8'));
 assert(manifest.completed_at_utc,'Must be sealed, never active');
 const final=manifest.result.physical_episode.physical_task_evaluation,ticks=final.history.event_ticks;
 const placed=Math.max(ticks.placed.FR,ticks.placed.FL);
 assert(Number.isInteger(placed));
 const lastTick=placed+120,physical=[],decisions=[];
 await prefixRows(path.join(run,'source/physical_observations.jsonl'),lastTick,r=>r.physics_tick,r=>physical.push(r));
 await prefixRows(path.join(run,'source/video_policy_decisions.jsonl'),lastTick,r=>r.end_tick,r=>decisions.push(r));
 const ranges={FR_initial_1s:[0,120],FR_task_until_first_placement:[0,ticks.placed.FR],front_pair_first_placement_plus_1s:[placed,placed+120]};
 result.push({label:item.label,run,source_git_commit:manifest.runtime_contract.source_git_commit,
  runtime_content_sha256:manifest.runtime_contract.runtime_content_sha256,completed_at_utc:manifest.completed_at_utc,
  historical_run_result:{lifecycle:manifest.lifecycle,success:final.success,termination_reason:final.termination_reason,interpretation:item.review},
  recorded_event_ticks:{FR:{qualified:ticks.active_lift.FR,crossed:ticks.front_edge_crossed.FR,placed:ticks.placed.FR},
   FL:{qualified:ticks.active_lift.FL,crossed:ticks.front_edge_crossed.FL,placed:ticks.placed.FL}},
  physical_prefix_rows_read:physical.length,recorded_decision_prefix_rows_read:decisions.length,
  windows:Object.fromEntries(Object.entries(ranges).map(([name,[a,b]])=>[name,{...physicalWindow(physical,a,b),recorded:recordedWindow(decisions,a,b)}]))});
}
console.log(JSON.stringify({schema:'wlr50_clean.front_quality_baseline_review.v1',created_at_utc:new Date().toISOString(),
 scope:'Two named historical sealed trajectories; no current zero-v2/live run, no CP177792 evaluation, no optimizer updates.',
 methods:{physical_hz:120,decision_hz:15,quaternion:'Existing fixed chassis-to-body identity; normalize measured wxyz, standard roll/pitch; no episode levelling.',
  stats:'Finite values only; abs_p95 uses nearest-rank. Inclusive endpoint samples; derivative samples exclude first endpoint.',
  windows:'Same initial 1s and same first-front-pair-placement +1s. Full FR-to-placement windows have unequal durations and are descriptive only.',
  placement_vs_support:'Window starts at historical physical placement events, but contemporaneous bearing is independently counted. Historical placed is not current TOP contact.',
  top_bearing:'Existing measured TOP contact test: verified active obstacle pair; actual contact point within top rectangle ±.005m and |z-top|<=.005m; upward reaction>=.2N and greater than horizontal reaction.',
  wheel_center_vs_contact:'Wheel centers, pose-aware collider bottoms and actual contact points are separate recorded quantities. A center-z mismatch is not a top-plane-height mismatch; common obstacle top remains z=.05m.',
  com:'Mass-weighted measured CoM; window-start fixed geometric RL direction and recorded local transfer diagnostics are motion correlations, not proof RL was loaded or of an optimal/causal CoM target.',
  transfer_load_field:'Existing FR-role load_fraction_change means FR load fraction at local-window start minus current FR load fraction, not an increase in RL loading. RL bearing force/load are reported independently.'},
 claims:{quality_reward_changed:false,training_added_decisions:0,stability_superiority:false,optimal_com_inferred:false,
  RR_task_validity_solved:false,quality_tuning_deferred_until_RR_task_valid:true},runs:result},null,2));

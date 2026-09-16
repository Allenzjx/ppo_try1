// Bounded, read-only source inspection. Reuses the completed full-stream aggregate proof.
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
const [runArg, aggregateArg, outputArg] = process.argv.slice(2);
assert(runArg && aggregateArg && outputArg, 'run aggregate new-output required');
const source = path.resolve(runArg, 'source'), output = path.resolve(outputArg);
const read = async file => JSON.parse(await fs.readFile(file, 'utf8'));
const final = await read(path.join(source, 'semantic_video_source_manifest.json'));
const run = await read(path.join(source, '..', 'run_manifest.json'));
const aggregate = await read(aggregateArg), end = final.episode_physics_ticks;
assert(run.completed_at_utc && final.role === 'B' && final.optimizer_updates === 0);
assert.equal(path.resolve(aggregate.source), source);
assert.equal(aggregate.exact_native_physical_join_ticks, end);
assert.equal(aggregate.source_runtime_sha256, final.runtime_contract.runtime_content_sha256);
async function tail(name, cap) {
  const file = await fs.open(path.join(source, name), 'r');
  try {
    const {size} = await file.stat(), start = Math.max(0, size - cap);
    const data = Buffer.alloc(size - start); await file.read(data, 0, data.length, start);
    const text = data.toString('utf8');
    return (start ? text.slice(text.indexOf('\n') + 1) : text).trim().split('\n').map(JSON.parse);
  } finally { await file.close(); }
}
const decisions = await tail('video_policy_decisions.jsonl', 8 * 1024 ** 2);
const owners = decisions.map(r => ({decision_line:r.decision, decision_end_tick:r.end_tick,
  owner:r.step_info?.semantic_task?.nominal_provider_diagnostics?.final_stop_owner})).filter(r => r.owner);
const first = owners.find(r => r.owner.active), entry = first?.owner.entry;
assert(entry && owners.some(r => r.decision_end_tick < entry.entry_observation_tick && !r.owner.active), 'acquisition not bracketed');
const pre = entry.entry_observation_tick, from = pre - 1;
const native = (await tail('native_tick_audit.jsonl', 8 * 1024 ** 2)).filter(r => r.episode_physics_tick >= from);
const physical = (await tail('physical_observations.jsonl', 12 * 1024 ** 2)).filter(r => r.physics_tick >= from);
for (const [rows, field] of [[native,'episode_physics_tick'],[physical,'physics_tick']]) {
  assert.equal(rows.length,end-from+1); rows.forEach((r,i)=>assert.equal(r[field],from+i));
}
const actual = new Map(physical.map(r=>[r.physics_tick,r]));
const window = native.filter(r => r.episode_physics_tick >= pre);
assert(window.every(r=>r.nominal_full12.every((v,i)=>v===entry.issued_nominal_full12[i])));
assert(window.every(r=>r.native_audit.actual_native_targets.wheel_velocity_rad_s.every(v=>v===0)));
assert(window.every(r=>r.native_audit.verified && r.native_audit.raw_policy_action_full12.every(v=>v===0)));
const norm = vector => Math.hypot(...vector);
const samples = [...new Set([from,pre,pre+1,8745,end])].filter(t=>t>=from&&t<=end).map(t=>{
  const n=native[t-from], p=actual.get(t);
  return {post_step_tick:t,simulation_time_s:t/120,native_source_line:t,physical_source_line:t+1,
    nominal_full12_deg_then_rad_s:n.nominal_full12,
    drive_full12_deg_then_rad_s:n.native_audit.native_drive_target_full12,
    actual_full12_deg_then_rad_s:p.actual_full12,
    body_linear_speed_m_s:norm(p.base.linear_velocity_w_m_s),
    body_angular_speed_rad_s:norm(p.base.angular_velocity_w_rad_s),body_collision:p.body_collision.detected};
});
const ev=final.physical_episode.physical_task_evaluation, quality=final.physical_episode.quality_metrics;
const result={schema:'wlr50_clean.final_stop_owner_live_evidence.v1',source,
  source_head:final.runtime_contract.source_git_commit,runtime_sha256:aggregate.source_runtime_sha256,
  reused_full_stream_aggregate:path.resolve(aggregateArg),tail_caps_bytes:{decision:8388608,native:8388608,physical:12582912},
  clock_definition:'entry_observation_tick is the real pre-step state; first held-next-dispatch effect is entry+1. Nominal stop already existed at entry. First decision observation is separately recorded; native/physical rows are post-step.',
  owner_entry:entry,first_owner_observed_decision:first,last_owner_observed_decision:owners.at(-1),
  owner_next_dispatch_post_step_tick:pre+1,exact_native_physical_window:[from,end],
  issued_nominal_held_from_entry_through_terminal:true,actual_native_wheels_zero_entire_window:true,
  full12_raw_zero_entire_window:true,all12_native_audit_verified_entire_window:true,
  source_clock_still_advances:owners.at(-1).owner.source_layer_ticks.P13>first.owner.source_layer_ticks.P13,
  observed_samples:samples,history_event_ticks:ev.history.event_ticks,
  terminal:Object.fromEntries(['physics_tick','success','termination_reason','termination_source','final_region_valid','final_controlled','final_support_available','task_completed_controlled','post_completion_observation_s','post_completion_elapsed_s','post_completion_observation_complete','post_completion_loss_observed'].map(k=>[k,ev[k]??null])),
  source_quality:{comparison_window:quality.comparison_window,fixed_quality_score:quality.fixed_quality_score,
    global:Object.fromEntries(['duration_s','physics_ticks','roll_rms_rad','pitch_rms_rad','roll_rate_rms_rad_s','pitch_rate_rms_rad_s','angular_acceleration_rms_rad_s2'].map(k=>[k,quality.global[k]??null]))},
  evidence_limits:['Nominal held does not mean physical q or final mapped servo targets are fixed. Existing mapper/feedback remains active.','Zero success is not learned PPO success. Tau is not used by zero.','Native/physical source-line identities reuse the validated complete aggregate cadence; owner entry is literal recorded data, not inferred from video.']};
await fs.writeFile(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({output,entry_tick:pre,first_owner_decision_end:first.decision_end_tick,terminal:result.terminal,tail_rows:{native:native.length,physical:physical.length},quality:result.source_quality}));

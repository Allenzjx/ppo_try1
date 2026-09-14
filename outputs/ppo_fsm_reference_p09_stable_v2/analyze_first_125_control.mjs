// Report-only, no production imports, no Torch/Isaac, no output file writes.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const root = path.resolve(import.meta.dirname, '../..');
const run = '20260910T0542011958077Z_g28609010db4e_b88e6f85c2e1480ea33d2fbcf28d6139';
const source = path.join(root, 'runs/ppo_fsm_reference_p09_stable_v2/train', run, 'residual_and_projection_audit.jsonl');
const fd = fs.openSync(source, 'r');
let bytes = Buffer.alloc(0), cut = -1, count = 0, cursor = 0;
try {
  while (count < 125) {
    const chunk = Buffer.alloc(65536);
    const n = fs.readSync(fd, chunk, 0, chunk.length, null);
    if (!n) throw new Error('Fewer than 125 complete audit lines');
    bytes = Buffer.concat([bytes, chunk.subarray(0, n)]);
    for (; cursor < bytes.length; cursor++) if (bytes[cursor] === 10 && ++count === 125) { cut = cursor + 1; break; }
  }
} finally { fs.closeSync(fd); }
// At most one 64-KiB read buffer beyond the fixed cutoff; no later line parsed.
const exact = bytes.subarray(0, cut);
const rows = exact.toString('utf8').trimEnd().split('\n').map(JSON.parse);
if (rows.length !== 125 || rows[0].global_policy_decision !== 141057 || rows[124].global_policy_decision !== 141181) throw new Error('Unexpected fixed credit range');
const manifest = JSON.parse(fs.readFileSync(path.join(path.dirname(source), 'run_manifest.started.json'), 'utf8'));
if (!manifest.runtime_contract.source_git_commit.startsWith('28609010db4e')) throw new Error('Unexpected production revision');
const profilePath = path.join(root, 'configs/ppo_fsm_reference_p09_stable_v2/execution_profile.yaml');
const profileBytes = fs.readFileSync(profilePath);
if (crypto.createHash('sha256').update(profileBytes).digest('hex') !== manifest.runtime_contract.selected_configuration['execution_profile.yaml'].sha256) throw new Error('Live config differs from manifest');
const profile = profileBytes.toString('utf8');
const caps = Object.fromEntries([...profile.matchAll(/^    (P\d\d): (\[[^\n]+\])/gm)].map(m => [m[1], JSON.parse(m[2])]));
const names = rows[0].applied_audit.actuator_target_effect_audit.canonical_order;
const signs = [1,1,1,1,-1,-1,-1,-1,-1,1,-1,1];
const finite = x => typeof x === 'number' && Number.isFinite(x);
const round = x => Math.round(x * 1e6) / 1e6;
const stats = values => {
  const v = values.filter(finite);
  return {n:v.length, missing:values.length-v.length, min:v.length ? round(Math.min(...v)):null,
    max:v.length ? round(Math.max(...v)):null, mean_abs:v.length ? round(v.reduce((s,x)=>s+Math.abs(x),0)/v.length):null};
};
let episode = 0, prev = null;
const data = rows.map((r, idx) => {
  if (r.global_policy_decision !== 141057+idx) throw new Error('Non-contiguous policy credit');
  const a = r.applied_audit, n = a.actuator_target_effect_audit, ev = a.semantic_task.physical_evaluator;
  if (!n.verified || !n.setter_dispatch_targets_equal || !n.actual_mapping_matches_dispatch || !n.same_tick_counterfactual) throw new Error('Unverified actual target effect');
  if (a.prefix_teacher_data_in_ppo_storage !== false) throw new Error('Teacher credit contamination or unknown flag');
  const actual = Array(12).fill(null);
  for (const role of Object.values(a.semantic_task.transfer_roles)) {
    for (const [name, m] of Object.entries(role.receiver_workspace_state.joint_range_margin_deg)) {
      const j = names.indexOf(name), lim = n.policy_headroom_evidence.servo_hard_limits_deg[j];
      if (j < 0 || !lim || !finite(m.negative_deg) || !finite(m.positive_deg)) continue;
      const q = m.negative_deg + lim[0];
      if (Math.abs(q - (lim[1]-m.positive_deg)) > 1e-8) throw new Error('Inconsistent actual joint margin reconstruction');
      actual[j] = q;
    }
  }
  ev.measured_wheel_velocity_rad_s.forEach((x,i) => actual[8+i] = finite(x)?x:null);
  const d = {g:r.global_policy_decision, line:idx+1, episode, phase:a.phase_id, end_phase:a.end_phase_id,
    tick:a.physics_tick, time:a.sim_time_s, terminal:r.terminal, reason:a.termination_reason,
    raw:r.raw_policy_action_full12, desired:r.raw_policy_action_full12.map((x,i)=>Math.tanh(x)*caps[a.phase_id][i]),
    projected:a.projected_residual_full12, final:a.actual_drive_target_full12,
    effective:n.policy_headroom_evidence.effective_policy_residual_full12,
    candidate:n.policy_headroom_evidence.candidate_native_target_before_final_slew_full12,
    mask:n.phase_mask_full12, caps:caps[a.phase_id], actual,
    native_effect:[...n.native_target_delta.servo_position_rad.map((x,i)=>x*180/Math.PI*signs[i]),
      ...n.native_target_delta.wheel_velocity_rad_s.map((x,i)=>x*signs[8+i])],
    native_changed:n.changed_channels_full12,
    headroom_clipped:n.policy_headroom_evidence.clipped_servo_indices,
    joint_change:actual.map((x,i)=>prev && finite(x) && finite(prev.actual[i]) ? x-prev.actual[i] : null),
    error:a.actual_drive_target_full12.map((x,i)=>finite(actual[i])?x-actual[i]:null),
    predecessor:prev?.g ?? null, physical_ticks:a.physics_ticks,
    tick_evidence:a.actuator_target_effect_audit_summary,
    handoff: a.phase_transition_action_jump.length > 0,
  };
  prev = r.terminal ? null : d;
  if (r.terminal) episode++;
  return d;
});
function aggregate(selected) {
  return {decisions:selected.length, executed_ticks:selected.reduce((s,d)=>s+d.physical_ticks,0),
    handoff_decision_count:selected.filter(d=>d.handoff).length,
    channels:names.map((name,j)=>({name, unit:j<8?'deg':'rad/s',
      raw:stats(selected.map(d=>d.raw[j])), tanh_scaled_request:stats(selected.map(d=>d.desired[j])),
      projected:stats(selected.map(d=>d.projected[j])), effective_post_headroom:stats(selected.map(d=>d.effective[j])),
      final_target:stats(selected.map(d=>d.final[j])), native_same_tick_effect:stats(selected.map(d=>d.native_effect[j])),
      actual_measured:stats(selected.map(d=>d.actual[j])), observed_change_from_prior_decision:stats(selected.map(d=>d.joint_change[j])),
      final_minus_actual_error:stats(selected.map(d=>d.error[j])),
      mask_closed_count:selected.filter(d=>d.mask[j]===0).length,
      transformed_request_not_reached_count:selected.filter(d=>Math.abs(d.desired[j]-d.projected[j])>1e-8).length,
      headroom_clipped_count:selected.filter(d=>d.headroom_clipped.includes(j)).length,
      effective_differs_requested_count:selected.filter(d=>Math.abs(d.effective[j]-d.projected[j])>1e-8).length,
      final_slew_or_clamp_active_count:selected.filter(d=>Math.abs(d.candidate[j]-d.final[j])>1e-8).length,
      final_candidate_outside_servo_hard_bounds_count:j<8?selected.filter(d=>d.candidate[j]<(j%2?-60:-135)-1e-8||d.candidate[j]>(j%2?210:135)+1e-8).length:null,
      native_target_changed_count:selected.filter(d=>d.native_changed[j]).length,
      projected_max_abs_cap_fraction:round(Math.max(...selected.map(d=>Math.abs(d.projected[j])/d.caps[j]))),
    }))};
}
const summary = {schema:'report_only.fixed_first_125_actual_control.v1',run,source, source_complete_lines:125,
  exact_prefix_bytes:cut, exact_prefix_sha256:crypto.createHash('sha256').update(exact).digest('hex'),
  read_at_utc:new Date().toISOString(), first_global_policy_decision:141057,last_global_policy_decision:141181,
  phase_counts:Object.fromEntries(['P06','P07','P08','P09'].map(p=>[p,data.filter(d=>d.phase===p).length])),
  terminal_rows:data.filter(d=>d.terminal).map(d=>({line:d.line,g:d.g,episode:d.episode,phase:d.phase,tick:d.tick,time:d.time,reason:d.reason})),
  native_unverified_tick_rows:data.filter(d=>!d.tick_evidence.all_ticks_verified).length,
  all:aggregate(data),P09:aggregate(data.filter(d=>d.phase==='P09')),
  P09_rows:data.filter(d=>d.phase==='P09').map(({raw,desired,projected,effective,final,actual,native_effect,error,predecessor,joint_change,g,line,tick,time,terminal,reason})=>({g,line,tick,time,terminal,reason,raw,desired,projected,effective,final,actual,native_effect,error,predecessor,joint_change})),
  interpretation:{sample_granularity:'Each aggregate channel rate uses last-dispatch / decision-end audits, not a claim about all per-physics-tick rates.',
    not_reached:'tanh(raw)*configured cap differs from final projected residual: aggregate transport/slew/handoff/bounds effect, not automatically hard clipping.',
    final_limit:'Candidate is pre-final-slew; if inside hard bounds, candidate-final difference is final slew, not a narrowed exploration cap.',
    actual:'Servo q reconstructed from independently consistent current measured hard-limit margins; wheel speed is current measured canonical observation. First decision of each episode excluded from observed change; no cross-reset subtraction.',
    effect:'Same-pre-tick without-current-residual target counterfactual; proves native target authority, not a separately simulated body-motion counterfactual.',
    causality:'Actual response contains nominal, residual, history, contact and inertia. Do not attribute all motion to current residual or call native target effect task success.',
    decision_credit:'Only 125 saved policy rows; teacher prefix omitted, no optimizer/checkpoint claim.'}};
process.stdout.write(JSON.stringify(summary,null,2)+'\n');

// Output-only extraction of recorded completed-video rewards. No simulation,
// controller replay, checkpoint load, CSV authoring, or production mutation.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import assert from 'node:assert/strict';

const args = Object.fromEntries(process.argv.slice(2).reduce((out, value, index, all) => {
  if (index % 2 === 0) {
    assert(value.startsWith('--') && all[index + 1], 'arguments require --name value');
    out.push([value.slice(2), all[index + 1]]);
  }
  return out;
}, []));
assert(Object.keys(args).every(k => ['run', 'label', 'output-prefix', 'max-seconds', 'max-tick'].includes(k)));
const run = path.resolve(args.run);
const label = args.label;
assert(/^[A-Za-z0-9_-]+$/.test(label));
assert(!(args['max-seconds'] && args['max-tick']), 'choose seconds or ticks, not both');
const maxTick = args['max-tick'] ? Number(args['max-tick']) : Number(args['max-seconds'] ?? 20) * 120;
assert(maxTick > 0 && maxTick <= 2400 && Number.isInteger(maxTick));
const maxSeconds = maxTick / 120;
const prefix = path.resolve(args['output-prefix']);
const root = path.dirname(path.dirname(path.dirname(prefix)));
assert(prefix.startsWith(path.resolve(root, 'outputs', 'ppo_rr_video_diagnosis_v1') + path.sep), 'output must stay in diagnostic directory');
const outputs = [prefix + '.json', prefix + '.md'];
assert(outputs.every(p => !fs.existsSync(p)), 'refusing to replace an existing result');

const manifestPath = path.join(run, 'run_manifest.json');
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
assert(['SUCCEEDED', 'DIAGNOSTIC_FAILURE'].includes(manifest.lifecycle), 'only finalized video runs');
assert(manifest.arguments.stage === 'full_episode' && manifest.arguments.seed === 4001);
const decisionPath = path.join(run, 'source', 'video_policy_decisions.jsonl');
const sourceBefore = fs.statSync(decisionPath);
const selected = [], missingReward = [], excludedCrossingLimit = [], events = new Map(), transitions = new Map();
const validation = {maximum_family_sum_error: 0, maximum_component_reconstruction_error: 0,
  maximum_elapsed_tick_error: 0, reward_aliases_identical: true};
const familyNames = ['task_progress', 'body_stability', 'contact_motion_quality', 'control_smoothness', 'control_regularization'];
const finite = x => { assert(typeof x === 'number' && Number.isFinite(x), 'required finite scalar'); return x; };
const addEvent = e => {
  if (e.leg !== 'FR' || !Number.isInteger(e.physics_tick) || e.physics_tick > maxTick) return;
  const record = {leg: 'FR', event: e.event, physics_tick: e.physics_tick, time_s: e.physics_tick / 120};
  events.set(`${record.event}:${record.physics_tick}`, record);
};
const stream = fs.createReadStream(decisionPath, {encoding: 'utf8', highWaterMark: 65536});
const reader = readline.createInterface({input: stream, crlfDelay: Infinity});
let parsedRows = 0, parsedUtf8Bytes = 0, previousEnd = 0, firstBeyond = null;
try {
  for await (const line of reader) {
    if (!line.trim()) continue;
    const row = JSON.parse(line);
    parsedRows++;
    parsedUtf8Bytes += Buffer.byteLength(line, 'utf8');
    assert(row.start_tick === previousEnd && row.end_tick > row.start_tick, 'video decisions must be contiguous');
    previousEnd = row.end_tick;
    if (row.start_tick >= maxTick) { firstBeyond = {decision: row.decision, start_tick: row.start_tick, end_tick: row.end_tick}; break; }
    if (row.end_tick > maxTick) { excludedCrossingLimit.push({decision: row.decision, start_tick: row.start_tick, end_tick: row.end_tick}); break; }
    if (row.environment_step_returned !== true) {
      missingReward.push({decision: row.decision, start_tick: row.start_tick, end_tick: row.end_tick,
        reason: row.stop_reason ?? null, logged_reward_available: false});
      continue;
    }
    const info = row.step_info, reward = info.reward, task = info.semantic_task;
    assert.deepStrictEqual(reward, info.reward_breakdown, 'reward aliases disagree');
    assert(info.physics_ticks === row.end_tick - row.start_tick);
    const duration = finite(reward.elapsed_physics_s);
    const d = reward.cost_components;
    const components = {
      potential_shaping: finite(reward.potential_shaping), terminal_event: finite(reward.terminal_event),
      elapsed_time: -0.02 * duration,
      body_attitude: -(0.4 / 3) * finite(d.gravity_attitude),
      body_euler_rates: -(0.4 / 3) * finite(d.euler_rate),
      body_angular_acceleration: -(0.4 / 3) * finite(d.angular_acceleration),
      contact_quality: -0.2 * finite(d.contact_quality),
      applied_first_difference: -0.05 * finite(d.actual_drive_first_difference),
      applied_second_difference: -0.05 * finite(d.actual_drive_second_difference),
      residual_regularization: finite(reward.families.control_regularization),
    };
    const familySum = familyNames.reduce((s, k) => s + finite(reward.families[k]), 0);
    const componentSum = Object.values(components).reduce((s, x) => s + x, 0);
    validation.maximum_family_sum_error = Math.max(validation.maximum_family_sum_error, Math.abs(familySum - finite(reward.total)));
    validation.maximum_component_reconstruction_error = Math.max(validation.maximum_component_reconstruction_error, Math.abs(componentSum - reward.total));
    validation.maximum_elapsed_tick_error = Math.max(validation.maximum_elapsed_tick_error, Math.abs(duration - info.physics_ticks / 120));
    assert(Math.abs(componentSum - reward.total) < 1e-10, 'logged reward does not match this reviewed coefficient set');
    const fraction = finite(task.physical_transfer_fraction);
    assert(fraction >= 0 && fraction <= 1);
    const evaluator = task.physical_evaluator, fr = evaluator.current_legs.FR, role = task.transfer_roles.FR;
    const history = evaluator.history ?? task.history;
    for (const e of history.lift_attempt_events ?? []) addEvent(e);
    for (const [key, name] of [['active_lift', 'first_history_Q'], ['front_edge_crossed', 'front_edge_crossed'], ['placed', 'placed']]) {
      const tick = history.event_ticks?.[key]?.FR ?? task.history?.event_ticks?.[key]?.FR;
      if (Number.isInteger(tick)) addEvent({leg: 'FR', event: name, physics_tick: tick});
    }
    for (const tr of info.stage_transition_evidence ?? []) {
      transitions.set(`${tr.physics_tick}:${tr.from_stage}:${tr.to_stage}`, {tick: tr.physics_tick,
        from_phase: tr.from_stage, to_phase: tr.to_stage, time_s: tr.sim_time_s});
    }
    const headroom = info.actuator_target_effect_audit?.policy_headroom_evidence;
    const clips = headroom?.clipped_servo_indices;
    assert(clips === undefined || Array.isArray(clips));
    selected.push({decision: row.decision, start_tick: row.start_tick, end_tick: row.end_tick,
      request_phase: row.request_phase, end_phase: info.end_phase_id, duration_s: duration,
      families: Object.fromEntries(familyNames.map(k => [k, reward.families[k]])), total: reward.total,
      components, cost_components_integrated: d, potential_before: reward.potential_before, potential_after: reward.potential_after,
      endpoint: {transfer_fraction: fraction, body_contact_weight: 1 - 0.8 * fraction,
        FR_role_motion_fraction: finite(role.motion_fraction), FR_preparation_progress: finite(role.preparation_progress),
        FR_transfer_progress: finite(role.transfer_progress), FR_I: fr.initial_clearance,
        FR_Q_history: history.active_lift.FR, FR_C: history.front_edge_crossed.FR, FR_P: history.placed.FR,
        FR_air: fr.air, FR_ground: fr.ground_contact, FR_top: fr.top_contact,
        FR_front_distance_m: finite(fr.front_distance_m), FR_top_gap_m: finite(fr.clearance_m),
        FR_normalized_load_valid: fr.load_fraction_valid, FR_role_valid: role.valid,
        FR_actuated_response: role.transfer_direction_context?.actuated_response ?? null,
        FR_short_support_continuity: role.transfer_direction_context?.short_support_continuity_fraction ?? null,
        FR_observed_supports: role.observed_support_contacts ?? [],
        body_angular_velocity_world_rad_s: role.transfer_direction_context?.body_angular_velocity_w_rad_s ?? null,
        observed_body_linear_speed_m_s: evaluator.goal_features?.body_linear_speed_m_s ?? null,
        observed_body_angular_speed_rad_s: evaluator.goal_features?.body_angular_speed_rad_s ?? null,
        headroom_clipped_servo_indices: clips ?? null,
        headroom_baseline_outside_reserved_servo_indices: headroom?.baseline_outside_reserved_servo_indices ?? null},
      nominal_full12: info.nominal_action_full12, residual_full12: info.projected_residual_full12,
      actual_drive_full12: info.actual_drive_target_full12,
      raw_residual_nonzero: row.raw_policy_action_full12.some(x => x !== 0),
      terminal_reason: info.termination_reason ?? null,
    });
  }
} finally { reader.close(); stream.destroy(); }
const sourceAfter = fs.statSync(decisionPath);
assert(sourceBefore.size === sourceAfter.size && sourceBefore.mtimeMs === sourceAfter.mtimeMs, 'completed source changed during read');
assert(selected.length > 0);

const sumObject = (rows, key) => Object.fromEntries(Object.keys(rows[0]?.[key] ?? {}).map(name =>
  [name, rows.reduce((s, row) => s + row[key][name], 0)]));
const range = values => values.length ? {min: Math.min(...values), max: Math.max(...values)} : null;
function summarize(name, rows, membership) {
  const duration = rows.reduce((s, row) => s + row.duration_s, 0);
  const costs = sumObject(rows, 'cost_components_integrated');
  return {name, membership, decisions: rows.map(row => row.decision), decision_count: rows.length,
    covered_ticks: rows.reduce((s, row) => s + row.end_tick - row.start_tick, 0), duration_s: duration,
    start_tick: rows[0]?.start_tick ?? null, end_tick: rows.at(-1)?.end_tick ?? null,
    reward_total: rows.reduce((s, row) => s + row.total, 0), families: sumObject(rows, 'families'),
    signed_components: sumObject(rows, 'components'), cost_components_integrated: costs,
    endpoint_transfer_fraction: range(rows.map(row => row.endpoint.transfer_fraction)),
    endpoint_body_contact_weight: range(rows.map(row => row.endpoint.body_contact_weight)),
    endpoint_FR_motion_fraction: range(rows.map(row => row.endpoint.FR_role_motion_fraction)),
    endpoint_FR_front_distance_m: range(rows.map(row => row.endpoint.FR_front_distance_m)),
    endpoint_FR_top_gap_m: range(rows.map(row => row.endpoint.FR_top_gap_m)),
    endpoint_FR_air_count: rows.filter(row => row.endpoint.FR_air).length,
    endpoint_FR_initial_count: rows.filter(row => row.endpoint.FR_I).length,
    endpoint_FR_Q_history_count: rows.filter(row => row.endpoint.FR_Q_history).length,
    endpoint_FR_placed_count: rows.filter(row => row.endpoint.FR_P).length,
    endpoint_FR_invalid_normalized_load_count: rows.filter(row => row.endpoint.FR_normalized_load_valid === false).length,
    endpoint_headroom_recorded_count: rows.filter(row => row.endpoint.headroom_clipped_servo_indices !== null).length,
    endpoint_headroom_clipped_decisions: rows.filter(row => row.endpoint.headroom_clipped_servo_indices?.length).map(row => row.decision),
    raw_residual_nonzero_decisions: rows.filter(row => row.raw_residual_nonzero).length,
    actual_motor_torque_saturation: null,
    reward_clipping_channel_saturation_counts: null,
    weighted_cost_mean_per_second: duration ? Object.fromEntries(['gravity_attitude', 'euler_rate', 'angular_acceleration', 'contact_quality', 'actual_drive_first_difference', 'actual_drive_second_difference'].map(k => [k, costs[k] / duration])) : null,
    handoff_spanning_decisions: rows.filter(row => row.request_phase !== row.end_phase).map(row =>
      ({decision: row.decision, start_tick: row.start_tick, end_tick: row.end_tick, request_phase: row.request_phase, end_phase: row.end_phase})),
  };
}
const windows = [summarize('all_returned_decisions_ending_by_bound', selected, {kind: 'complete_decisions', requested_ticks: [0, maxTick]})];
for (const phase of ['P01', 'P02', 'P03']) {
  windows.push(summarize(`request_${phase}`, selected.filter(row => row.request_phase === phase),
    {kind: 'request_phase_complete_decisions', request_phase: phase, within_ticks: [0, maxTick], not_exact_physical_phase_integral: true}));
}
const eventList = [...events.values()].sort((a, b) => a.physics_tick - b.physics_tick || a.event.localeCompare(b.event));
for (const e of eventList) {
  e.containing_decision = selected.find(row => row.start_tick < e.physics_tick && row.end_tick >= e.physics_tick)?.decision ?? null;
  if (e.event === 'first_history_Q') continue; // same event also has exact attempt Q record
  const lo = Math.max(0, e.physics_tick - 60), hi = Math.min(maxTick, e.physics_tick + 60);
  const intersecting = selected.filter(row => row.start_tick < hi && row.end_tick > lo);
  const contained = intersecting.filter(row => row.start_tick >= lo && row.end_tick <= hi);
  windows.push(summarize(`FR_${e.event}_${e.physics_tick}_plus_minus_0p5s_complete_only`, contained,
    {kind: 'fully_contained_complete_decisions', requested_ticks: [lo, hi], event: e,
      excluded_boundary_decisions: intersecting.filter(row => !contained.includes(row)).map(row =>
        ({decision: row.decision, start_tick: row.start_tick, end_tick: row.end_tick}))}));
}
const crossing = eventList.find(e => e.event === 'front_edge_crossed');
const placement = eventList.find(e => e.event === 'placed');
if (crossing && placement) {
  const lo = crossing.physics_tick, hi = placement.physics_tick;
  const rows = selected.filter(row => row.start_tick < hi && row.end_tick >= lo);
  windows.push(summarize('FR_cross_to_place_whole_decision_envelope', rows,
    {kind: 'whole_decision_envelope', requested_physical_event_ticks: [lo, hi],
      interval_padding_not_event_exact: true}));
}
const result = {schema: 'wlr50_clean.recorded_reward_window_extraction.v1', label,
  created_at_utc: new Date().toISOString(), run, lifecycle: manifest.lifecycle,
  runtime_contract: {source_git_commit: manifest.runtime_contract.source_git_commit,
    runtime_content_sha256: manifest.runtime_contract.runtime_content_sha256},
  source: {path: decisionPath, full_file_size_bytes: sourceBefore.size,
    parsed_rows_including_one_beyond_limit: parsedRows, parsed_utf8_content_bytes_excluding_line_endings: parsedUtf8Bytes,
    no_full_file_hash_repeated: true, source_size_and_mtime_unchanged: true},
  bound: {maximum_seconds: maxSeconds, maximum_tick: maxTick, first_beyond_bound: firstBeyond,
    excluded_crossing_limit: excludedCrossingLimit, missing_reward_intervals: missingReward},
  validation, events: eventList, phase_transitions: [...transitions.values()].sort((a, b) => a.tick - b.tick),
  windows, selected_decision_evidence: selected,
  limitations: ['No actor/critic/optimizer updates and no learned-motive inference.',
    'Transfer fraction, physical state and headroom evidence are decision endpoints, not a 120 Hz series.',
    'Cost components are logged 120 Hz integrated weighted/clipped costs, not raw RMS or unweighted physical rates.',
    'Complete-decision phase allocation is by request phase; handoff-spanning decisions are explicit.',
    'A physical-endpoint interrupted final step has no recorded reward; no terminal reward is fabricated.',
    'Motor torque/force saturation and per-channel reward clipping counts are not present and remain null.'],
};
const fmt = x => typeof x === 'number' ? x.toFixed(9) : String(x);
const lines = [`# ${label}: recorded reward prefix, up to ${maxSeconds} s`, '',
  `Source: \`${decisionPath}\`. Lifecycle: ${manifest.lifecycle}. No new simulation, checkpoint load, production changes or learning.`, '',
  `Read ${selected.length} returned decisions, ending at tick ${selected.at(-1).end_tick} (${fmt(selected.at(-1).end_tick / 120)} s). One subsequent row may be parsed only to close the prefix bound.`, '',
  '## Signed recorded totals', '',
  '| Window | Decisions | Seconds | Potential | Time | Body attitude | Euler rates | Angular acceleration | Contact | Applied 1st | Applied 2nd | Terminal | Total |',
  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'];
for (const w of windows) {
  const c = w.signed_components;
  lines.push(`| ${w.name} | ${w.decision_count} | ${fmt(w.duration_s)} | ${fmt(c.potential_shaping ?? 0)} | ${fmt(c.elapsed_time ?? 0)} | ${fmt(c.body_attitude ?? 0)} | ${fmt(c.body_euler_rates ?? 0)} | ${fmt(c.body_angular_acceleration ?? 0)} | ${fmt(c.contact_quality ?? 0)} | ${fmt(c.applied_first_difference ?? 0)} | ${fmt(c.applied_second_difference ?? 0)} | ${fmt(c.terminal_event ?? 0)} | ${fmt(w.reward_total)} |`);
}
lines.push('', 'Windows overlap: never sum table rows into a new episode total. P01/P02/P03 rows use the request phase; JSON identifies each handoff-spanning decision. Event windows include only fully contained decisions and list excluded boundaries. The C-to-P envelope includes padding, not an exact event-time integral.', '',
  '## Event and weight evidence', '');
for (const e of eventList) lines.push(`- FR ${e.event}: tick ${e.physics_tick}, ${fmt(e.time_s)} s, containing decision ${e.containing_decision}.`);
for (const w of windows.filter(w => w.decision_count)) lines.push(`- ${w.name}: endpoint f ${fmt(w.endpoint_transfer_fraction.min)}–${fmt(w.endpoint_transfer_fraction.max)}, weight ${fmt(w.endpoint_body_contact_weight.min)}–${fmt(w.endpoint_body_contact_weight.max)}; recorded endpoint headroom clipping ${w.endpoint_headroom_clipped_decisions.length}/${w.endpoint_headroom_recorded_count}; FR normalized-load-invalid endpoints ${w.endpoint_FR_invalid_normalized_load_count}/${w.decision_count}.`);
lines.push('', '## Verification and limits', '',
  `Maximum reward reconstruction error ${validation.maximum_component_reconstruction_error}; family-sum error ${validation.maximum_family_sum_error}; elapsed-tick error ${validation.maximum_elapsed_tick_error}. Reward/reward_breakdown aliases are identical.`, '',
  ...result.limitations.map(x => `- ${x}`), '',
  `Unreturned/missing-reward prefix intervals: ${JSON.stringify(missingReward)}. No full-episode reward or full task-success claim. JSON contains exact decision membership, endpoint evidence, signed integrals, normalized weighted cost means, and boundary exclusions.`, '');
fs.writeFileSync(outputs[0], JSON.stringify(result, null, 2) + '\n', {flag: 'wx', encoding: 'utf8'});
fs.writeFileSync(outputs[1], lines.join('\n'), {flag: 'wx', encoding: 'utf8'});
console.log(JSON.stringify({outputs, label, selected_decisions: selected.length,
  last_tick: selected.at(-1).end_tick, validation, events: eventList, total: windows[0]}));

/**
 * Flat scientific CSV from completed NEW-experiment actual train/eval decisions.
 * Run only after the parent has marked this CSV artifact operation exactly once.
 * Use the bundled Node executable and a node_modules junction in this directory
 * targeting the load_workspace_dependencies bundled node_modules directory.
 * No Isaac/Torch/checkpoint loading, source mutation, or CSV fabrication occurs.
 *
 * node build_p09_timeline.mjs --run <completed run> [--run <another completed run>]
 *   [--output <new-experiment-output-path/p09_evidence_timeline.csv>]
 * Output is CSV ONLY, plus bounded authoring QA PNG/JSON support files. No XLSX.
 */
import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import readline from 'node:readline';
import { createHash } from 'node:crypto';
import { Workbook } from '@oai/artifact-tool';

const outputRoot = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(outputRoot, '../..');
const runRoot = path.join(projectRoot, 'runs', 'ppo_fsm_reference_p09_stable_v2');
const experimentId = 'fsm_reference_p09_stable_v2';
const channels = ['FL_hip', 'FL_knee', 'FR_hip', 'FR_knee', 'RL_hip', 'RL_knee',
  'RR_hip', 'RR_knee', 'FL_wheel', 'FR_wheel', 'RL_wheel', 'RR_wheel'];
const servoNames = ['front_left_hip', 'front_left_knee', 'front_right_hip', 'front_right_knee',
  'rear_left_hip', 'rear_left_knee', 'rear_right_hip', 'rear_right_knee'];
const wheelNames = ['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle'];
// Frozen production canonical readback: command_batch.py SERVO_COMMAND_SIGN /
// HIP_LIMIT_DEG / KNEE_LIMIT_DEG. These transform measurements, not action bounds.
const servoSigns = [1, 1, 1, 1, -1, -1, -1, -1];
const servoLower = [-135, -60, -135, -60, -135, -60, -135, -60];
const servoUpper = [135, 210, 135, 210, 135, 210, 135, 210];
const args = process.argv.slice(2), runs = [];
let output = path.join(outputRoot, 'p09_evidence_timeline.csv');
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--run' && args[i + 1]) runs.push(path.resolve(args[++i]));
  else if (args[i] === '--output' && args[i + 1]) output = path.resolve(args[++i]);
  else throw new Error('Use --run <completed new-namespace train/eval run> [--run ...] [--output <fresh CSV path>]');
}
const assert = (ok, message) => { if (!ok) throw new Error(message); };
const inside = (root, target) => {
  const rel = path.relative(root, target);
  return !!rel && !path.isAbsolute(rel) && rel !== '..' && !rel.startsWith(`..${path.sep}`);
};
const finite = value => typeof value === 'number' && Number.isFinite(value) ? value : null;
const boolean = value => typeof value === 'boolean' ? value : null;
const string = value => typeof value === 'string' ? value : null;
const vector = (value, size) => Array.from({ length: size }, (_, i) => finite(Array.isArray(value) ? value[i] : null));
const complete = values => values.every(v => v !== null);
const scalar = value => value === undefined ? null : value;
const equal = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
const json = async file => JSON.parse(await fs.readFile(file, 'utf8'));
assert(runs.length > 0, 'Explicit completed run input required; never export an empty placeholder timeline');
assert(inside(outputRoot, output) && path.extname(output).toLowerCase() === '.csv', 'CSV must stay in this new output namespace');
assert(new Set(runs.map(v => v.toLowerCase())).size === runs.length, 'Duplicate run inputs would duplicate decisions');
try { await fs.access(output); throw new Error('CSV already exists; preserve it and choose a fresh snapshot path'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }

const rows = [], sourceReceipts = [];
const prefix = ['source_file', 'source_line_1based', 'source_run', 'source_command', 'source_mode', 'source_cli_mode_argument',
  'source_manifest', 'source_lifecycle', 'source_git_head', 'run_episode_index', 'episode_index_source',
  'run_decision_index', 'episode_decision_count', 'global_policy_decision', 'on_policy_sample_valid',
  'credited_policy_decision', 'optimized_policy_decision', 'saved_policy_decision', 'zero_residual_eval',
  'deterministic_eval', 'teacher_prefix_credit', 'phase_before', 'phase_after', 'substage_after',
  'decision_end_physics_tick', 'decision_end_sim_time_s', 'executed_physics_ticks',
  'physical_source_file', 'physical_source_line_1based', 'physical_source_episode_index',
  'physical_source_tick', 'physical_source_sim_time_s', 'physical_exact_tick_match',
  'physical_decision_join_valid', 'physical_join_status', 'physical_source_all_finite',
  'physical_direct_servo_count', 'physical_direct_wheel_count', 'physical_canonical_crosscheck_count',
  'physical_servo_margin_crosscheck_count', 'physical_measurement_units',
  'last_dispatch_native_tick', 'last_pre_dispatch_measurement_episode_tick',
  'state_values_timing', 'target_values_timing', 'terminal', 'termination_reason', 'termination_source',
  'task_success', 'full_task_success', 'terminal_bootstrap_allowed', 'reward', 'old_log_probability',
  'raw_action_distribution_valid', 'native_all_ticks_verified', 'native_effect_ticks', 'root_writes_verified_absent',
  'rr_contact_mode', 'rr_air', 'rr_ground_contact', 'rr_obstacle_pair_active', 'rr_top_surface_contact',
  'rr_nontop_obstacle_contact', 'rr_contact_surface', 'rr_contact_mode_available',
  'rr_initial_lift_observed', 'rr_lift_established', 'rr_current_lift_valid', 'rr_motion_continuation_allowed',
  'rr_qualification_history_Q', 'rr_crossed_history_C', 'rr_placed_history_P', 'rr_placed_currently_usable',
  'rr_current_actor_qualification_bit', 'rr_front_distance_m', 'rr_top_clearance_m', 'rr_ground_relative_lift_m',
  'rr_recent_wheel_displacement_m', 'rr_edge_adjustment_response_observed', 'rr_body_control_evidence',
  'rr_air_duration_s_diagnostic', 'rr_edge_contact_duration_s_diagnostic', 'duration_is_acceptance_gate',
  'rr_load_fraction', 'rr_load_fraction_valid', 'rr_raw_encoded_load_fraction', 'rr_contact_reaction_force_n',
  'rr_bearing_component_n', 'rr_bearing_verified', 'rr_motion_continuation_reason',
  'rr_nominal_added_rolling_suggestion', 'rr_nominal_carry_reason', 'rr_nominal_geometry_status',
  'rr_nominal_geometry_floor_semantics', 'rr_nominal_geometry_descent_allowance_m',
  'rr_hip_execution_notes', 'rr_knee_execution_notes', 'rr_current_joint_measurement_source',
  'rr_current_joint_measurement_valid', 'body_linear_speed_m_s', 'body_linear_velocity_x_m_s',
  'body_linear_velocity_y_m_s', 'body_linear_velocity_z_m_s', 'body_linear_vector_available',
  'body_linear_velocity_frame', 'body_roll_rad', 'body_pitch_rad', 'body_attitude_available',
  'body_attitude_source', 'body_attitude_calibration_file', 'body_angular_velocity_x_rad_s',
  'body_angular_velocity_y_rad_s', 'body_angular_velocity_z_rad_s', 'body_angular_vector_available',
  'body_angular_speed_rad_s', 'body_angular_velocity_frame', 'body_vector_source', 'com_x_m', 'com_y_m', 'com_z_m',
  'com_velocity_x_m_s', 'com_velocity_y_m_s', 'com_velocity_z_m_s', 'com_measurement_available'];
const groups = [
  ['raw_latent', 12, () => ''],
  ['nominal_request', 12, i => i < 8 ? '_deg' : '_rad_s'],
  ['projected_residual_request', 12, i => i < 8 ? '_deg' : '_rad_s'],
  ['effective_post_mapper_residual', 12, i => i < 8 ? '_deg' : '_rad_s'],
  ['bridge_applied_request', 12, i => i < 8 ? '_deg' : '_rad_s'],
  ['final_drive_target', 12, i => i < 8 ? '_deg' : '_rad_s'],
  ['actual_native_target', 12, i => i < 8 ? '_rad' : '_rad_s'],
  ['native_target_delta_without_current_residual', 12, i => i < 8 ? '_rad' : '_rad_s'],
  ['current_measured_joint_or_wheel', 12, i => i < 8 ? '_deg' : '_rad_s'],
  ['pre_dispatch_measured_servo', 8, () => '_deg'],
  ['phase_mask', 12, () => ''],
];
const header = [...prefix, ...groups.flatMap(([name, count, unit]) =>
  [`${name}_complete`, ...channels.slice(0, count).map((channel, i) => `${name}_${channel}${unit(i)}`)])];

function currentJointMeasurements(task) {
  // Role diagnostics contain current MEASURED q minus each immutable limit.
  // Invert both margins and cross-check rather than pretending target == q.
  const values = Array(8).fill(null);
  for (const role of Object.values(task.transfer_roles ?? {})) {
    const margins = role.receiver_workspace_state?.joint_range_margin_deg;
    if (role.valid !== true || !margins) continue;
    for (let i = 0; i < 8; i++) {
      const m = margins[servoNames[i]], low = finite(m?.negative_deg), high = finite(m?.positive_deg);
      if (low === null || high === null) continue;
      const fromLow = low + servoLower[i], fromHigh = servoUpper[i] - high;
      assert(Math.abs(fromLow - fromHigh) <= 1e-8, `Inconsistent measured joint margins: ${servoNames[i]}`);
      assert(values[i] === null || Math.abs(values[i] - fromLow) <= 1e-8, 'Conflicting current measured joint sources');
      values[i] = fromLow;
    }
  }
  return values;
}

async function physicalObservationJoin(run, enabled) {
  // Streaming merge by (physical episode, exact tick), never nearest-neighbour.
  // A physical episode changes only on a real reset to tick 0. Initial tick 0
  // and intervening physics samples are not policy decisions or CSV rows.
  const source = enabled ? path.join(run, 'physical_observations.jsonl') : null;
  let before = null;
  if (source) {
    try { before = await fs.stat(source); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  const stream = before ? createReadStream(source) : null;
  const reader = stream ? readline.createInterface({ input: stream, crlfDelay: Infinity }) : null;
  const iterator = reader?.[Symbol.asyncIterator]();
  let pending = null, eof = false, line = 0, epoch = 0, lastTick = null;
  let matched = 0, missing = 0, invalidClock = 0;
  async function advance() {
    while (!eof) {
      const next = await iterator.next();
      if (next.done) { eof = true; pending = null; return; }
      line++;
      if (!next.value.trim()) continue;
      const observation = JSON.parse(next.value), tick = observation.physics_tick;
      assert(Number.isInteger(tick) && tick >= 0, `Physical clock invalid at ${source}:${line}`);
      if (lastTick !== null && tick < lastTick) {
        assert(tick === 0, `Physical stream moved backwards without a tick-0 reset at ${source}:${line}`);
        epoch++;
      } else assert(tick !== lastTick, `Ambiguous duplicate physical tick at ${source}:${line}`);
      lastTick = tick;
      pending = { observation, source, line, episode: epoch, tick };
      return;
    }
  }
  return {
    async lookup(episode, tick, time) {
      if (!iterator) {
        missing++;
        return { source: before ? source : null, matched: false, valid: false,
          status: enabled ? 'physical_stream_missing' : 'training_has_no_physical_join' };
      }
      if (!pending && !eof) await advance();
      while (pending && (pending.episode < episode || (pending.episode === episode && pending.tick < tick)))
        await advance();
      if (!pending || pending.episode !== episode || pending.tick !== tick) {
        missing++;
        // Do not expose the adjacent record as a measured source for this row.
        return { source, matched: false, valid: false, status: 'exact_episode_tick_missing' };
      }
      const o = pending.observation;
      const sameTime = finite(o.simulation_time_s) !== null && Math.abs(o.simulation_time_s - time) <= 1e-8;
      const sameDt = finite(o.physics_dt_s) !== null && Math.abs(o.physics_dt_s - 1 / 120) <= 1e-12;
      const schema = o.schema === 'wlr50_clean.live_observation.v1';
      const valid = sameTime && sameDt && schema;
      matched++;
      if (!valid) invalidClock++;
      return { ...pending, matched: true, valid,
        status: !schema ? 'unsupported_physical_schema' : !sameTime ? 'simulation_time_mismatch'
          : !sameDt ? 'physics_dt_mismatch' : 'exact_episode_tick_and_clock' };
    },
    async finish() {
      reader?.close(); stream?.destroy();
      if (before) {
        const after = await fs.stat(source);
        assert(before.size === after.size && before.mtimeMs === after.mtimeMs,
          'Physical stream changed while joining completed decisions; do not export a torn view');
      }
      return { source: before ? source : null, fileBytes: before?.size ?? null,
        physicalLinesConsumed: line, exactMatchedDecisions: matched, missingDecisions: missing,
        invalidClockDecisions: invalidClock, nearestSampleUsed: false,
        episodeJoin: 'tick_zero_reset_order_matched_to_actual_terminal_order' };
    },
  };
}

function joinedMeasurements(join, marginJoints, evaluatorWheels, fixedChassisQuaternion) {
  const eligible = join.valid && join.observation?.all_finite === true;
  const o = eligible ? join.observation : null;
  const joints = [...marginJoints], wheels = [...evaluatorWheels];
  const jointSources = marginJoints.map(v => v === null ? null : 'current_role_measured_q_margin_inverse_both_limits_crosschecked');
  let servoCount = 0, wheelCount = 0, canonicalChecks = 0, marginChecks = 0;
  const actual = vector(o?.actual_full12, 12);
  for (let i = 0; i < 12; i++) {
    const name = i < 8 ? servoNames[i] : wheelNames[i - 8];
    const named = i < 8 ? o?.joints?.[name] : o?.wheels?.[name];
    const direct = named?.name === name ? finite(i < 8 ? named.position_deg : named.velocity_rad_s) : null;
    if (direct === null) continue;
    // sensor_reader.py writes already-canonical position_deg / velocity_rad_s
    // from actual_full12. No physical-radian or rear-leg sign conversion here.
    if (actual[i] !== null) {
      assert(Math.abs(direct - actual[i]) <= 1e-6, `Named physical measurement disagrees with canonical actual_full12: ${name}`);
      canonicalChecks++;
    }
    const derived = i < 8 ? marginJoints[i] : evaluatorWheels[i - 8];
    if (derived !== null) {
      assert(Math.abs(direct - derived) <= 1e-6, `Exact-tick physical measurement disagrees with evaluator: ${name}`);
      if (i < 8) marginChecks++;
    }
    if (i < 8) { joints[i] = direct; jointSources[i] = 'exact_tick_physical_joints_position_deg_canonical'; servoCount++; }
    else { wheels[i - 8] = direct; wheelCount++; }
  }
  const base = o?.base?.name === 'base_link' ? o.base : null;
  const linear = vector(base?.linear_velocity_w_m_s, 3), angular = vector(base?.angular_velocity_w_rad_s, 3);
  const q = vector(base?.orientation_wxyz, 4), calibration = vector(fixedChassisQuaternion, 4);
  let roll = null, pitch = null;
  if (complete(q) && complete(calibration)) {
    const qn = Math.hypot(...q), cn = Math.hypot(...calibration);
    if (Number.isFinite(qn) && Number.isFinite(cn) && qn >= 1e-12 && cn >= 1e-12) {
      const [w, x, y, z] = q.map(v => v / qn), [a, b, c, d] = calibration.map(v => v / cn);
      // Same q_world_body * fixed_chassis_to_body convention as semantic_observation.py.
      const product = [w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
        w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a];
      const productNorm = Math.hypot(...product);
      const [cw, cx, cy, cz] = product.map(v => v / productNorm);
      roll = Math.atan2(2 * (cw*cx + cy*cz), 1 - 2 * (cx*cx + cy*cy));
      pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (cw*cy - cz*cx))));
    }
  }
  return { joints, wheels, jointSources, linear, angular, roll, pitch,
    servoCount, wheelCount, canonicalChecks, marginChecks };
}

function executionNotes(index, info, native, headroom) {
  const notes = [];
  const raw = finite(info.raw_policy_action_full12?.[index]);
  const projected = finite(info.projected_residual_full12?.[index]);
  if (native.phase_mask_full12?.[index] === 0) notes.push('phase_mask_disabled');
  if (headroom.clipped_servo_indices?.includes(index)) notes.push('post_mapper_reserved_headroom_clipped');
  const candidate = finite(headroom.candidate_native_target_before_final_slew_full12?.[index]);
  const final = finite(info.actual_drive_target_full12?.[index]);
  if (candidate !== null && final !== null && Math.abs(candidate - final) > 1e-9)
    notes.push('final_bounded_target_differs_from_candidate');
  if (raw === 0 && projected !== null && projected !== 0) notes.push('filtered_residual_decay_after_raw_zero');
  const delta = finite(native.native_target_delta?.servo_position_rad?.[index]);
  if (raw !== null && raw !== 0 && delta === 0 && native.verified === true)
    notes.push('no_same_tick_native_target_effect_from_current_residual');
  if (native.nominal_geometry_adjustment_full12?.[index]) notes.push('nominal_geometry_adjustment');
  return notes.length ? notes.join('; ') : null;
}

for (const run of runs) {
  assert(inside(runRoot, run), 'Only NEW namespace runs are allowed');
  const manifestFile = path.join(run, 'run_manifest.json'), manifest = await json(manifestFile);
  const a = manifest.arguments ?? {};
  assert(a.experiment_id === experimentId && a.num_envs === 1, 'Exact new experiment and N1 required');
  assert(['train', 'eval'].includes(manifest.command ?? a.command), 'Only actual train/eval; no smoke, teacher-prefix or recording inputs');
  assert(['SUCCEEDED', 'STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'].includes(manifest.lifecycle),
    'Require completed manifest; no active, partial or failed run gets silently credited');
  const train = (manifest.command ?? a.command) === 'train';
  const result = await json(path.join(run, train ? 'training_manifest.json' : 'evaluation_manifest.json'));
  assert(!train || result.num_envs === 1, 'N1 training manifest required');
  assert(train || result.optimizer_updates_during_evaluation === 0, 'Evaluation must not update optimizer');
  assert(train || result.mode !== 'interface_smoke', 'Interface probes cannot become task evaluation data');
  const expected = train ? result.actual_policy_decisions : result.policy_decisions;
  assert(Number.isInteger(expected) && expected > 0, 'Completed actual decision count required');
  const source = path.join(run, 'residual_and_projection_audit.jsonl');
  const beforeStat = await fs.stat(source), rowStart = rows.length;
  let attitudeCalibrationFile = null, fixedChassisQuaternion = null;
  if (!train) {
    const calibrationRecord = manifest.runtime_contract?.selected_configuration?.['observation_schema.json'];
    if (calibrationRecord?.path && calibrationRecord?.sha256) {
      attitudeCalibrationFile = path.resolve(projectRoot, calibrationRecord.path);
      assert(inside(projectRoot, attitudeCalibrationFile), 'Attitude calibration must be a recorded project configuration');
      const calibrationBytes = await fs.readFile(attitudeCalibrationFile);
      assert(createHash('sha256').update(calibrationBytes).digest('hex') === calibrationRecord.sha256,
        'Observation schema differs from completed run; do not invent attitude axes');
      fixedChassisQuaternion = vector(JSON.parse(calibrationBytes.toString('utf8')).fixed_chassis_to_body_wxyz, 4);
      assert(complete(fixedChassisQuaternion) && Math.hypot(...fixedChassisQuaternion) > 0,
        'Recorded fixed chassis quaternion is invalid');
    }
  }
  const physicalJoin = await physicalObservationJoin(run, !train);
  const globalEnd = train ? result.global_policy_decisions : null;
  const globalStart = train ? globalEnd - expected : null;
  const savedEnd = train ? Math.max(...(result.checkpoints ?? []).map(v => v.global_policy_decisions)) : null;
  assert(!train || (Number.isInteger(globalEnd) && Number.isInteger(savedEnd) && savedEnd >= globalEnd),
    'Completed train block must be saved before it is exported as saved learning');
  let sourceLine = 0, count = 0, episode = 0;
  const seen = new Set();
  for await (const line of readline.createInterface({ input: createReadStream(source), crlfDelay: Infinity })) {
    sourceLine++;
    if (!line.trim()) continue;
    const wrapper = JSON.parse(line), info = train ? wrapper.applied_audit : wrapper;
    assert(info?.schema === 'wlr50_clean.semantic_decision.v2', `Invalid actual decision at ${source}:${sourceLine}`);
    assert(Object.hasOwn(info, 'termination_reason') && (info.termination_reason === null || typeof info.termination_reason === 'string'),
      'Missing terminal evidence is not a nonterminal or terminal observation');
    assert(Number.isInteger(info.physics_tick) && info.physics_tick > 0 && Number.isInteger(info.physics_ticks)
      && info.physics_ticks >= 1 && info.physics_ticks <= 8 && finite(info.sim_time_s) !== null,
      'Actual decision requires its measured physical clock and executed tick count');
    const task = info.semantic_task ?? {}, ev = task.physical_evaluator ?? {}, rr = ev.current_legs?.RR ?? {};
    const hist = ev.history ?? task.history ?? {}, native = info.actuator_target_effect_audit ?? {};
    const headroom = native.policy_headroom_evidence ?? {}, tracking = native.tracking_reference_evidence ?? {};
    const geometry = native.nominal_geometry_evidence ?? {}, carry = task.nominal_provider_diagnostics?.rr_carry_continuation ?? {};
    const ctx = task.transfer_roles?.RR?.transfer_direction_context ?? ev.transfer_roles?.RR?.transfer_direction_context ?? {};
    const raw = vector(info.raw_policy_action_full12, 12), means = vector(wrapper.old_distribution_mean_full12, 12);
    const stds = vector(wrapper.old_distribution_std_full12, 12);
    const distributionValid = train ? complete(means) && complete(stds) && stds.every(v => v > 0)
      && finite(wrapper.old_log_probability) !== null : null;
    assert(complete(raw), 'Actual sampled/evaluated raw Full12 must be finite');
    if (train) {
      assert(distributionValid && equal(raw, vector(wrapper.raw_policy_action_full12, 12)), 'Sampled latent/distribution binding invalid');
      assert(wrapper.global_policy_decision === globalStart + count + 1, 'Training decision credit sequence mismatch');
    }
    const terminal = train ? boolean(wrapper.terminal) : info.termination_reason !== null;
    assert(terminal !== null && terminal === (info.termination_reason !== null), 'Terminal audit binding mismatch');
    const key = `${episode}:${info.decision_count}:${info.physics_tick}`;
    assert(!seen.has(key), 'Duplicate actual decision; terminal sink must not be counted twice');
    seen.add(key);
    const joined = await physicalJoin.lookup(episode, info.physics_tick, info.sim_time_s);
    const physical = joinedMeasurements(joined, currentJointMeasurements(task), vector(ev.measured_wheel_velocity_rad_s, 4),
      fixedChassisQuaternion);
    const endJoints = physical.joints, measuredRad = vector(tracking.actual_measured_physical_rad, 8);
    const standing = vector(tracking.standing_pose_deg, 8);
    const preJoints = measuredRad.map((v, i) => v === null || standing[i] === null ? null : servoSigns[i] * (v * 180 / Math.PI - standing[i]));
    const contextOmega = vector(ctx.body_angular_velocity_w_rad_s, 3);
    if (complete(physical.angular) && complete(contextOmega))
      assert(physical.angular.every((v, i) => Math.abs(v - contextOmega[i]) <= 1e-6),
        'Exact-tick physical body angular velocity differs from current evaluator');
    const omega = complete(physical.angular) ? physical.angular : contextOmega;
    const linearSpeed = complete(physical.linear) ? Math.hypot(...physical.linear) : finite(ev.goal_features?.body_linear_speed_m_s);
    const angularSpeed = complete(physical.angular) ? Math.hypot(...physical.angular) : finite(ev.goal_features?.body_angular_speed_rad_s);
    if (complete(physical.linear) && finite(ev.goal_features?.body_linear_speed_m_s) !== null)
      assert(Math.abs(linearSpeed - ev.goal_features.body_linear_speed_m_s) <= 1e-6,
        'Exact-tick physical body linear speed differs from current evaluator');
    if (complete(physical.angular) && finite(ev.goal_features?.body_angular_speed_rad_s) !== null)
      assert(Math.abs(angularSpeed - ev.goal_features.body_angular_speed_rad_s) <= 1e-6,
        'Exact-tick physical body angular speed differs from current evaluator');
    const com = vector(ctx.mass_weighted_com_position_w_m, 3), comV = vector(ctx.mass_weighted_com_velocity_w_m_s, 3);
    const nativeTargets = [...vector(native.actual_native_targets?.servo_position_rad, 8), ...vector(native.actual_native_targets?.wheel_velocity_rad_s, 4)];
    const nativeDelta = [...vector(native.native_target_delta?.servo_position_rad, 8), ...vector(native.native_target_delta?.wheel_velocity_rad_s, 4)];
    const loadValid = boolean(rr.load_fraction_valid);
    const edge = [rr.obstacle_pair_active, rr.ground_contact, rr.top_surface_contact].every(v => typeof v === 'boolean')
      ? rr.obstacle_pair_active && !rr.ground_contact && !rr.top_surface_contact : null;
    const zeroEval = !train && a.mode === 'semantic_prior_eval';
    if (zeroEval) assert(raw.every(v => v === 0), 'Zero-residual evaluation contains a nonzero raw action');
    const record = {
      source_file: source, source_line_1based: sourceLine, source_run: path.basename(run),
      source_command: train ? 'train' : 'eval', source_mode: train ? 'semantic_residual_train' : a.mode,
      source_cli_mode_argument: a.mode, source_manifest: manifestFile,
      source_lifecycle: manifest.lifecycle, source_git_head: manifest.runtime_contract?.source_git_commit,
      run_episode_index: episode, episode_index_source: 'contiguous_actual_terminal_order_no_empty_prefix_episode',
      run_decision_index: count + 1, episode_decision_count: info.decision_count,
      global_policy_decision: train ? wrapper.global_policy_decision : null,
      on_policy_sample_valid: train ? true : false, credited_policy_decision: train,
      optimized_policy_decision: train, saved_policy_decision: train, zero_residual_eval: zeroEval,
      deterministic_eval: train ? null : boolean(result.deterministic_policy), teacher_prefix_credit: false,
      phase_before: info.phase_id, phase_after: info.end_phase_id, substage_after: task.substage,
      decision_end_physics_tick: info.physics_tick, decision_end_sim_time_s: info.sim_time_s,
      executed_physics_ticks: info.physics_ticks,
      physical_source_file: joined.source, physical_source_line_1based: joined.matched ? joined.line : null,
      physical_source_episode_index: joined.matched ? joined.episode : null,
      physical_source_tick: joined.matched ? joined.tick : null,
      physical_source_sim_time_s: joined.matched ? finite(joined.observation.simulation_time_s) : null,
      physical_exact_tick_match: joined.matched, physical_decision_join_valid: joined.valid,
      physical_join_status: joined.status, physical_source_all_finite: boolean(joined.observation?.all_finite),
      physical_direct_servo_count: physical.servoCount, physical_direct_wheel_count: physical.wheelCount,
      physical_canonical_crosscheck_count: physical.canonicalChecks, physical_servo_margin_crosscheck_count: physical.marginChecks,
      physical_measurement_units: physical.servoCount + physical.wheelCount > 0
        ? 'sensor_reader_canonical_position_deg_and_wheel_velocity_rad_s_no_second_sign_conversion' : null,
      last_dispatch_native_tick: native.physics_tick,
      last_pre_dispatch_measurement_episode_tick: complete(preJoints) ? info.physics_tick - 1 : null,
      state_values_timing: joined.valid && joined.observation?.all_finite === true
        ? 'current_evaluator_and_exact_tick_physical_decision_end' : 'current_evaluator_at_decision_end',
      target_values_timing: 'last_native_dispatch_of_this_decision_not_actual_motion', terminal,
      termination_reason: info.termination_reason, termination_source: task.termination_source ?? ev.termination_source,
      task_success: boolean(info.task_success), full_task_success: boolean(info.full_task_success),
      terminal_bootstrap_allowed: boolean(info.terminal_bootstrap_allowed),
      reward: finite(train ? wrapper.reward : info.reward?.total), old_log_probability: train ? finite(wrapper.old_log_probability) : null,
      raw_action_distribution_valid: distributionValid,
      native_all_ticks_verified: boolean(info.actuator_target_effect_audit_summary?.all_ticks_verified),
      native_effect_ticks: finite(info.actuator_target_effect_audit_summary?.actual_native_effect_tick_count),
      root_writes_verified_absent: boolean(info.no_in_episode_state_writes_verified),
      rr_contact_mode: string(rr.contact_mode), rr_air: boolean(rr.air), rr_ground_contact: boolean(rr.ground_contact),
      rr_obstacle_pair_active: boolean(rr.obstacle_pair_active), rr_top_surface_contact: boolean(rr.top_surface_contact),
      rr_nontop_obstacle_contact: edge, rr_contact_surface: string(rr.contact_surface), rr_contact_mode_available: typeof rr.contact_mode === 'string',
      rr_initial_lift_observed: boolean(rr.initial_lift_observed), rr_lift_established: boolean(rr.lift_established),
      rr_current_lift_valid: boolean(rr.current_lift_valid), rr_motion_continuation_allowed: boolean(rr.motion_continuation_allowed),
      rr_qualification_history_Q: boolean(hist.active_lift?.RR), rr_crossed_history_C: boolean(hist.front_edge_crossed?.RR),
      rr_placed_history_P: boolean(hist.placed?.RR), rr_placed_currently_usable: boolean(task.rr_placed_currently_usable),
      rr_current_actor_qualification_bit: boolean(task.active_lift_history?.RR),
      rr_front_distance_m: finite(rr.front_distance_m), rr_top_clearance_m: finite(rr.clearance_m),
      rr_ground_relative_lift_m: finite(rr.ground_relative_lift_m), rr_recent_wheel_displacement_m: finite(rr.recent_wheel_displacement_m),
      rr_edge_adjustment_response_observed: boolean(rr.edge_adjustment_response_observed), rr_body_control_evidence: boolean(rr.body_control_evidence),
      rr_air_duration_s_diagnostic: finite(rr.air_duration_s), rr_edge_contact_duration_s_diagnostic: finite(rr.edge_contact_duration_s),
      duration_is_acceptance_gate: rr.durations_are_diagnostic_only === true ? false : null,
      rr_load_fraction: loadValid === true ? finite(rr.load_fraction) : null, rr_load_fraction_valid: loadValid,
      rr_raw_encoded_load_fraction: finite(rr.load_fraction), rr_contact_reaction_force_n: finite(rr.contact_reaction_force_n),
      rr_bearing_component_n: finite(rr.bearing_force_n), rr_bearing_verified: boolean(rr.bearing_verified),
      rr_motion_continuation_reason: string(rr.motion_continuation_reason),
      rr_nominal_added_rolling_suggestion: boolean(carry.added_rolling_suggestion), rr_nominal_carry_reason: string(carry.reason),
      rr_nominal_geometry_status: string(geometry.status), rr_nominal_geometry_floor_semantics: string(geometry.context?.clearance_floor_semantics),
      rr_nominal_geometry_descent_allowance_m: finite(geometry.projection?.available_descent_m),
      rr_hip_execution_notes: executionNotes(6, info, native, headroom), rr_knee_execution_notes: executionNotes(7, info, native, headroom),
      rr_current_joint_measurement_source: endJoints[6] !== null && endJoints[7] !== null
        ? [...new Set(physical.jointSources.slice(6, 8))].join('; ') : null,
      rr_current_joint_measurement_valid: endJoints[6] !== null && endJoints[7] !== null,
      body_linear_speed_m_s: linearSpeed,
      body_linear_velocity_x_m_s: physical.linear[0], body_linear_velocity_y_m_s: physical.linear[1], body_linear_velocity_z_m_s: physical.linear[2],
      body_linear_vector_available: complete(physical.linear), body_linear_velocity_frame: complete(physical.linear) ? 'world' : null,
      body_roll_rad: physical.roll, body_pitch_rad: physical.pitch,
      body_attitude_available: physical.roll !== null && physical.pitch !== null,
      body_attitude_source: physical.roll !== null && physical.pitch !== null
        ? 'exact_tick_base_orientation_wxyz_times_recorded_fixed_chassis_to_body_ZYX_roll_pitch_relative_world' : null,
      body_attitude_calibration_file: physical.roll !== null && physical.pitch !== null ? attitudeCalibrationFile : null,
      body_angular_velocity_x_rad_s: omega[0], body_angular_velocity_y_rad_s: omega[1], body_angular_velocity_z_rad_s: omega[2],
      body_angular_vector_available: complete(omega), body_angular_speed_rad_s: angularSpeed,
      body_angular_velocity_frame: complete(omega) ? 'world' : null,
      body_vector_source: [complete(physical.linear) ? 'exact_tick_physical_base_linear_velocity_w_m_s' : null,
        complete(physical.angular) ? 'exact_tick_physical_base_angular_velocity_w_rad_s'
          : complete(contextOmega) ? 'current_RR_transfer_direction_context_angular_velocity_only' : null].filter(Boolean).join('; ') || null,
      com_x_m: com[0], com_y_m: com[1], com_z_m: com[2], com_velocity_x_m_s: comV[0], com_velocity_y_m_s: comV[1], com_velocity_z_m_s: comV[2],
      com_measurement_available: complete(com) && complete(comV),
    };
    const arrays = [raw, vector(info.nominal_action_full12, 12), vector(info.projected_residual_full12, 12),
      vector(headroom.effective_policy_residual_full12, 12), vector(info.applied_action_full12, 12),
      vector(info.actual_drive_target_full12, 12), nativeTargets, nativeDelta,
      [...endJoints, ...physical.wheels], preJoints, vector(native.phase_mask_full12, 12)];
    const flat = [...prefix.map(k => scalar(record[k])), ...arrays.flatMap(v => [complete(v), ...v])];
    assert(flat.length === header.length, 'Timeline columns lost alignment');
    assert(flat.every(v => v === null || ['number', 'string', 'boolean'].includes(typeof v)), 'Only scalar scientific table values are allowed');
    assert(!flat.some(v => typeof v === 'string' && v.startsWith('=')), 'Unexpected formula-like source text; do not execute it as a formula');
    rows.push(flat); count++;
    if (terminal) episode++;
  }
  const afterStat = await fs.stat(source);
  assert(beforeStat.size === afterStat.size && beforeStat.mtimeMs === afterStat.mtimeMs, 'Run audit changed during read; stop instead of exporting a torn live view');
  assert(count === expected, `Actual audit row count ${count} != completed manifest ${expected}`);
  const physicalReceipt = await physicalJoin.finish();
  sourceReceipts.push({ run, source, command: train ? 'train' : 'eval', actualDecisionRows: count,
    firstOutputDataRow: rowStart + 2, lastOutputDataRow: rows.length + 1,
    terminalEpisodes: episode, globalStartExclusive: globalStart, globalEndInclusive: globalEnd,
    teacherPrefixRowsRead: 0, bytesRead: beforeStat.size, physicalObservationJoin: physicalReceipt });
}
assert(rows.length > 0, 'No actual decisions: do not emit empty CSV');

// Required artifact authoring/verification. Typed, unmerged, one record per row.
const workbook = Workbook.create(), sheet = workbook.worksheets.add('Timeline');
sheet.getRangeByIndexes(0, 0, 1, header.length).values = [header];
const chunk = 256;
for (let offset = 0; offset < rows.length; offset += chunk) {
  const values = rows.slice(offset, offset + chunk);
  sheet.getRangeByIndexes(offset + 1, 0, values.length, header.length).values = values;
}
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
const populated = sheet.getRangeByIndexes(0, 0, rows.length + 1, header.length);
populated.format = { font: { name: 'Arial', size: 10 }, columnWidth: 22, rowHeight: 24, verticalAlignment: 'center' };
sheet.getRangeByIndexes(0, 0, 1, header.length).format = { font: { name: 'Arial', size: 10, bold: true },
  wrapText: true, rowHeight: 84, horizontalAlignment: 'center', fill: '#E8EEF4' };
workbook.recalculate();
const col = index => {
  let s = '', n = index + 1;
  while (n) { s = String.fromCharCode(65 + (n - 1) % 26) + s; n = Math.floor((n - 1) / 26); }
  return s;
};
const samples = [...new Set([0, Math.floor(rows.length / 2), rows.length - 1,
  rows.findIndex(r => r[header.indexOf('phase_before')] === 'P09'),
  rows.findIndex(r => r[header.indexOf('terminal')] === true)].filter(v => v >= 0))];
const inspections = [];
for (const index of samples) {
  const range = sheet.getRangeByIndexes(index + 1, 0, 1, header.length).values[0];
  assert(equal(range.map(scalar), rows[index]), `Authored grid mismatch at data row ${index + 1}`);
  inspections.push(await workbook.inspect({ kind: 'table', range: `Timeline!U${index + 2}:BC${index + 2}`,
    include: 'values,formulas', tableMaxRows: 1, tableMaxCols: 35, maxChars: 2500 }));
}
const errors = await workbook.inspect({ kind: 'match',
  searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',
  options: { useRegex: true, maxResults: 20 }, summary: 'final scalar timeline error scan' });
// Each immutable CSV snapshot owns its own QA evidence; later exports must
// not overwrite the first check's rendered image, receipt or review notes.
const qaDir = path.join(path.dirname(output), `${path.basename(output, '.csv')}_authoring_qa`);
await fs.mkdir(qaDir, { recursive: true });
const previewStart = header.indexOf('rr_contact_mode'), previewEnd = header.indexOf('rr_contact_mode_available');
const png = await workbook.render({ sheetName: 'Timeline',
  range: `${col(previewStart)}1:${col(previewEnd)}${Math.min(rows.length + 1, 8)}`, scale: 1, format: 'png' });
await fs.writeFile(path.join(qaDir, 'contact_columns.png'), new Uint8Array(await png.arrayBuffer()));
// Public docs have no CSV exporter. Serialize VERIFIED authored sheet values,
// retaining numbers/booleans and null blanks. No parallel XLSX is exported.
const quote = value => value === null || value === undefined ? '' : typeof value === 'number' ? String(value)
  : typeof value === 'boolean' ? String(value) : `"${String(value).replaceAll('"', '""')}"`;
const csvLines = [header.map(quote).join(',')];
for (let offset = 0; offset < rows.length; offset += chunk) {
  const size = Math.min(chunk, rows.length - offset);
  const authored = sheet.getRangeByIndexes(offset + 1, 0, size, header.length).values;
  for (let i = 0; i < size; i++) {
    assert(equal(authored[i].map(scalar), rows[offset + i]), `Authored scalar mismatch at data row ${offset + i + 1}`);
    csvLines.push(authored[i].map(quote).join(','));
  }
}
await fs.mkdir(path.dirname(output), { recursive: true });
await fs.writeFile(output, csvLines.join('\r\n') + '\r\n', { encoding: 'utf8', flag: 'wx' });
const receipt = { output, rows: rows.length, columns: header.length, sourceReceipts,
  teacherPrefixCredit: false, missingValuesRemainBlank: true, fixedRRHoldAcceptance: false,
  scalarGridVerified: true, engineRecalculated: true, inspections, errorScan: errors,
  visualReviewRequired: path.join(qaDir, 'contact_columns.png'),
  unavailableFields: ['body_roll_rad', 'body_pitch_rad', 'body_linear_velocity_x_m_s', 'body_linear_velocity_y_m_s', 'body_linear_velocity_z_m_s']
    .filter(field => rows.every(row => row[header.indexOf(field)] === null)),
  physicalJoinPolicy: 'completed eval only; exact episode+decision_end_tick+time+dt; no nearest or previous sample imputation',
  timingNote: 'Current q prefers exact-tick physical canonical joint positions crosschecked against actual_full12 and measured role margins; margins remain the explicit fallback. Missing physical body attitude/linear vectors stay blank. Pre-dispatch q is separate; no command or CoM is used as body measurement.' };
await fs.writeFile(path.join(qaDir, 'receipt.json'), JSON.stringify(receipt, null, 2) + '\n', 'utf8');
console.log(JSON.stringify({ output, rows: rows.length, columns: header.length, receipt: path.join(qaDir, 'receipt.json'),
  visualReviewRequired: receipt.visualReviewRequired }));

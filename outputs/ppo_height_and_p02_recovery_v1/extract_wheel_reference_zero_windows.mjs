// Output-only, fixed sealed sources; one bounded prefix pass per large stream.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import assert from 'node:assert/strict';

const root = path.resolve(import.meta.dirname, '../..');
const outdir = import.meta.dirname;
const A = 'C:/robotics_sim/wlr_robot/fsm_50mm_recording_shaped_clean_v1/runs/trial_043_20260902_clean_v010';
const B = path.join(root, 'runs/ppo_fsm_reference_p09_stable_v2/video_eval/prior_B/20260915T0525295619176Z_g4a58c0190ef7_615f8fe3cfc64cff81f73ac6a0d302e6/source');
const W = ['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle'];
const LEGS = ['FL', 'FR', 'RL', 'RR'];
const stats = xs => ({ min: Math.min(...xs), max: Math.max(...xs), mean: xs.reduce((a, b) => a + b, 0) / xs.length });
const vstats = xs => W.map((_, i) => stats(xs.map(x => x[i])));
const vector = x => { assert(Array.isArray(x) && x.length === 4 && x.every(Number.isFinite)); return x; };
const w4 = x => vector(x.slice(8));
const tally = xs => xs.reduce((a, x) => (a[String(x)] = (a[String(x)] ?? 0) + 1, a), {});
const json = p => JSON.parse(fs.readFileSync(p, 'utf8'));
const transitions = p => fs.readFileSync(p, 'utf8').trim().split(/\r?\n/).map(JSON.parse);
const reads = [];
async function prefix(file, tickKey, max, consume) {
  const stream = fs.createReadStream(file);
  const lines = readline.createInterface({ input: stream, crlfDelay: Infinity });
  let lineNo = 0, previous = -1, used = 0;
  try {
    for await (const line of lines) {
      lineNo++;
      const row = JSON.parse(line), tick = row[tickKey];
      assert(Number.isInteger(tick) && tick > previous, `${file}: monotonic ${tickKey}`);
      previous = tick;
      if (tick > max) break;
      consume(row, lineNo); used++;
    }
  } finally { lines.close(); stream.destroy(); }
  reads.push({ file, max_included_tick: max, rows_used: used, rows_parsed_including_one_stop_lookahead: lineNo });
}
function physical(p, line) {
  return { tick: p.physics_tick, line, qd: W.map(w => p.wheels[w].velocity_rad_s), command: W.map(w => p.wheels[w].command_rad_s),
    contact: W.map(w => {
      const c = p.contacts[p.wheels[w].body_name];
      assert(c && c.ground && c.obstacle);
      return { body: p.wheels[w].body_name, class: c.contact_class,
        ground: { active: c.ground.active, verified: c.ground.pair_verified, normal_n: c.ground.normal_force_n },
        obstacle: { active: c.obstacle.active, verified: c.obstacle.pair_verified, normal_n: c.obstacle.normal_force_n } };
    }) };
}
const aCommands = new Map(), aPhysical = new Map(), bNative = new Map(), bPhysical = new Map(), bDecisions = [];
// A P04 begins command 1816; B P04 starts after physical 1504. Include eight B handoff ticks.
await prefix(path.join(A, 'full12_commands_120hz.jsonl'), 'control_physics_tick', 1815, (v, line) => {
  aCommands.set(v.control_physics_tick + 1, { tick: v.control_physics_tick + 1, command_tick: v.control_physics_tick, line,
    stage: v.state_id, lifecycle: v.lifecycle, motion_tick: v.motion_tick_index,
    source_nominal: w4(v.nominal_full12), nominal: w4(v.full12), mapped: w4(v.native_drive_target_full12),
    command: w4(v.commanded_full12), native_physical_target: vector(v.atomic_ack.wheel_target_physical_rad_s),
    residual: w4(v.residual_full12), normal_bias: w4(v.normal_drive_bias_full12),
    feedback_bias: w4(v.drive_feedback_bias_realized_full12), atomic_source_event: v.atomic_source_event,
    feedback_kind: v.drive_feedback.kind, feedback_active: v.drive_feedback.active });
});
await prefix(path.join(A, 'observation_120hz.jsonl'), 'physics_tick', 1816, (v, line) => aPhysical.set(v.physics_tick, physical(v, line)));
await prefix(path.join(B, 'native_tick_audit.jsonl'), 'episode_physics_tick', 1512, (v, line) => {
  const a = v.native_audit;
  assert(a.verified && a.setter_dispatch_targets_equal && a.actual_mapping_matches_dispatch);
  assert(v.projected_residual_full12.length === 12 && v.projected_residual_full12.every(x => x === 0));
  assert(a.native_target_delta.servo_position_rad.length === 8 && a.native_target_delta.servo_position_rad.every(x => x === 0));
  bNative.set(v.episode_physics_tick, { tick: v.episode_physics_tick, command_tick: v.episode_physics_tick - 1, line,
    stage: v.source_phase_id, nominal: w4(v.nominal_full12), mapped: w4(a.native_drive_target_full12),
    native_physical_target: vector(a.actual_native_targets.wheel_velocity_rad_s), residual: w4(v.projected_residual_full12),
    raw_full12: a.raw_policy_action_full12, mask_full12: a.phase_mask_full12,
    native_delta: vector(a.native_target_delta.wheel_velocity_rad_s), normal_bias: w4(a.controller_drive_bias_full12) });
});
await prefix(path.join(B, 'physical_observations.jsonl'), 'physics_tick', 1512, (v, line) => bPhysical.set(v.physics_tick, physical(v, line)));
await prefix(path.join(B, 'video_policy_decisions.jsonl'), 'end_tick', 1512, v => {
  const task = v.step_info?.semantic_task;
  assert(task && v.environment_step_returned === true);
  bDecisions.push({ decision: v.decision, start_tick: v.start_tick, end_tick: v.end_tick, request_phase: v.request_phase,
    end_phase: task.stage_id, nominal: w4(v.step_info.nominal_action_full12),
    task_stage_elapsed_s_not_source_clock: task.stage_elapsed_s,
    diagnostic_keys: Object.keys(task.nominal_provider_diagnostics),
    recorded_partial_order_layers: task.nominal_provider_diagnostics.source_partial_order.layers,
    capture_owner_hold: task.nominal_provider_diagnostics.capture_owner_hold,
    FR: Object.fromEntries(['contact_mode', 'air', 'support', 'bearing_verified', 'bearing_force_n', 'clearance_m', 'front_distance_m'].map(k => [k, task.physical_evaluator.current_legs.FR[k]])) });
});
function join(cmds, phys, label) {
  return [...cmds.values()].map(c => {
    const p = phys.get(c.tick); assert(p, `${label} missing post-step observation ${c.tick}`);
    if (c.command) assert.deepEqual(c.command, p.command, `${label} pre/post command mismatch ${c.tick}`);
    for (let i = 0; i < 4; i++) assert(Math.abs(c.mapped[i] - p.command[i]) < 1e-7, `${label} native/canonical command mismatch ${c.tick}`);
    return { ...c, measured: p };
  });
}
const ar = join(aCommands, aPhysical, 'A'), br = join(bNative, bPhysical, 'B');
assert(br.every(r => r.raw_full12.every(v => v === 0) && r.residual.every(v => v === 0) && r.native_delta.every(v => v === 0)));
assert(br.every(r => r.mask_full12.every(v => v === 1)));
function summarize(rows, label) {
  assert(rows.length);
  const first = rows[0], last = rows.at(-1);
  return { label, post_observation_ticks_inclusive: [first.tick, last.tick], samples_120hz: rows.length,
    command_ticks_inclusive: [first.command_tick, last.command_tick], task_stage_counts: tally(rows.map(r => r.stage)),
    source_nominal_rad_s: first.source_nominal === undefined ? null : vstats(rows.map(r => r.source_nominal)),
    nominal_rad_s: vstats(rows.map(r => r.nominal)), mapped_target_rad_s: vstats(rows.map(r => r.mapped)),
    final_canonical_command_rad_s: vstats(rows.map(r => r.measured.command)), native_physical_target_rad_s: vstats(rows.map(r => r.native_physical_target)),
    actual_canonical_qd_rad_s: vstats(rows.map(r => r.measured.qd)),
    contact: W.map((_, i) => ({ leg: LEGS[i], class_counts: tally(rows.map(r => r.measured.contact[i].class)),
      ...Object.fromEntries(['ground', 'obstacle'].map(s => [s, {
        active_samples: rows.filter(r => r.measured.contact[i][s].active === true).length,
        pair_verified_samples: rows.filter(r => r.measured.contact[i][s].verified === true).length,
        normal_force_n: stats(rows.map(r => r.measured.contact[i][s].normal_n)) }])) })),
    feedback_active_samples: first.feedback_active === undefined ? null : rows.filter(r => r.feedback_active).length,
    normal_bias_rad_s: vstats(rows.map(r => r.normal_bias)),
    source_lines: { command_or_native: [first.line, last.line], physical: [first.measured.line, last.measured.line] } };
}
function segments(rows) {
  const chunks = [];
  for (const r of rows) {
    const key = JSON.stringify([r.stage, r.source_nominal, r.nominal, r.mapped, r.native_physical_target]);
    if (chunks.at(-1)?.key !== key) chunks.push({ key, rows: [] });
    chunks.at(-1).rows.push(r);
  }
  return chunks.map((c, i) => summarize(c.rows, `command_segment_${i + 1}`));
}
const contract = json(path.join(root, 'configs/recording_motion_contract.json'));
const source = contract.phases.slice(0, 3).map(p => ({ stage: p.state_id, start_wheels_rad_s: w4(p.start_full12), end_wheels_rad_s: w4(p.end_full12),
  source_recording_steps: p.reference_steps, active_duration_s: p.active_duration_s,
  waypoints: p.waypoints.map(w => ({ time_s: w.time_s, source_events: w.source_events, full_wheels_rad_s: w4(w.full12),
    changed_wheel_channels: w.changed_channels.filter(n => W.includes(n)), atomic_wheel_channels: (w.atomic_channels ?? []).filter(n => W.includes(n)) })) }));
const aWindows = [
  summarize(ar.filter(r => r.command_tick >= 1584 && r.command_tick <= 1591), 'P01_last_rolling_8_ticks'),
  summarize(ar.filter(r => r.command_tick >= 1592 && r.command_tick <= 1599), 'P01_explicit_stop_8_ticks'),
  summarize(ar.filter(r => r.stage === 'P02'), 'P02_all'),
  summarize(ar.filter(r => r.stage === 'P03'), 'P03_all') ];
const bWindows = [
  summarize(br.filter(r => r.stage === 'P01'), 'P01_all_16_ticks'),
  summarize(br.filter(r => r.stage === 'P02'), 'P02_all'),
  summarize(br.filter(r => r.tick >= 1465 && r.tick <= 1472), 'P02_last_8_ticks'),
  summarize(br.filter(r => r.stage === 'P03'), 'P03_all'),
  summarize(br.filter(r => r.stage === 'P04'), 'P03_to_P04_8_tick_handoff') ];
const result = {
  schema: 'wlr50_clean.readonly_wheel_reference_zero_windows.v1', wheel_order: LEGS, units: 'rad/s except contact force N',
  scope: 'Only sealed A prefix through P03 and sealed successful zero prefix through first P04 handoff. No simulation, policy update, re-adjudication, full Recording scan, or checkpoint load.',
  sources: { A, B, contract: path.join(root, 'configs/recording_motion_contract.json'), successful_zero_aggregate: path.join(outdir, 'B_HEIGHT_FINAL_STOP.aggregate.json') },
  zero_terminal_preserved: json(path.join(outdir, 'B_HEIGHT_FINAL_STOP.aggregate.json')).terminal,
  clock_semantics: { A: 'command control_physics_tick=t precedes observation physics_tick=t+1; joined post-step t+1', B: 'native episode_physics_tick=t joins physical physics_tick=t (post-step). source_phase_id is task dispatch phase, NOT sole source-layer owner.' },
  source_contract: source,
  ownership_interpretation: {
    A: 'The command stream nominal_full12 already contains successful-FSM logical correction, not raw Recording: P03 RL=0.51911 in both nominal_full12 and full12. Raw source_contract is separately 0.61; 0.61*(1-0.149)=0.51911. One current motion stage; correction is not PPO.',
    B_P02: 'P01 changed-channel owner continues across early P02 task transition. P02 may additionally suggest verified P01 four-wheel rolling via _approach_assist_required. Existing stream does not distinguish these coincident same-value contributors at every tick; source_partial_order.layers is empty for P01-P03.',
    B_P03: 'P03 FL/RL changed owners replace those two channels. FR/RR retain prior 0.3 until an authored wheel stop or later owner. Do not copy raw P03 snapshot zero into untouched channels.',
    proof_limit: 'Per-tick owner attribution is code-supported interpretation, not a newly recorded owner field. Full N, native target, measured qd and exact-pair contact are direct data.' },
  production_evidence: ['semantic_supervisor.py:_start_source_motion', 'semantic_supervisor.py:_continuous_advisory', 'semantic_supervisor.py:_approach_assist_required', 'semantic_supervisor.py:tick', 'configs/fsm_states.yaml:P03 normal_correction_fractions RL wheel=-0.149'],
  A: { transitions: transitions(path.join(A, 'state_transitions.jsonl')).filter(v => ['P01', 'P02', 'P03'].includes(v.state_id)).map(v => ({ time_s: v.sim_time_s, stage: v.state_id, from: v.from_lifecycle, to: v.to_lifecycle })), windows: aWindows, command_segments: segments(ar) },
  B: { transitions: transitions(path.join(B, 'stage_transition_evidence.jsonl')).filter(v => v.physics_tick <= 1512).map(v => ({ tick: v.physics_tick, from: v.from_stage, to: v.to_stage })), windows: bWindows, command_segments: segments(br),
    decision_evidence: bDecisions.filter(v => [16, 24, 40, 48, 1464, 1472, 1480, 1488, 1504, 1512].includes(v.end_tick)),
    zero_raw_projected_and_same_tick_native_effect_all_channels_in_scanned_prefix: true, all12_phase_mask_open_in_scanned_prefix: true },
  reads,
  limitations: ['Measured wheel qd is load-dependent and may differ across wheels or from target; not a requirement for four-wheel equal-speed motion.', 'AIR contact is not load-bearing. Pair verification is reported separately from activity.', 'These unequal task/time windows describe each valid chain, not matched-condition causal proof or PPO learning.', 'A original run status is preserved; this helper does not re-label A terminal success.', 'Successful zero completion is preserved, not made pending on this audit.']
};
// Keep terminal identity compact; all full leg details remain in source aggregate.
result.zero_terminal_preserved = Object.fromEntries(['tick','duration_s','phase','task_success','evaluation_termination_reason','source_acceptance_error'].map(k => [k, result.zero_terminal_preserved[k]]));
fs.writeFileSync(path.join(outdir, 'wheel_reference_zero_windows.json'), JSON.stringify(result, null, 2) + '\n');
const vec = x => '[' + x.map(v => Number(v.toFixed(5))).join(', ') + ']';
const target = stats4 => stats4.every(v => v.min === v.max) ? vec(stats4.map(v => v.min)) : `${vec(stats4.map(v => v.min))}..${vec(stats4.map(v => v.max))}`;
let md = '# 原 FSM A / 成功 zero：P01–P03 四轮完整通路\n\n';
md += '四轮顺序 FL/FR/RL/RR，单位 rad/s；下面是物理 120 Hz 样本，不是 policy decisions。成功 zero 保持 SUCCEEDED（8857 ticks / 73.80833333333334 s），没有修改生产或重新定义成功。\n\n';
md += '| 运行 / 窗口 | 后步观测 ticks | N（A 为修正后 Full12） | 最终下发 | 实测 qd 均值 | 实测 qd 最小..最大 |\n|---|---:|---|---|---|---|\n';
for (const [label, windows] of [['A',aWindows],['zero',bWindows]]) for (const w of windows) md += `| ${label} ${w.label} | ${w.post_observation_ticks_inclusive.join('..')} | ${target(w.nominal_rad_s)} | ${target(w.final_canonical_command_rad_s)} | ${vec(w.actual_canonical_qd_rad_s.map(v=>v.mean))} | ${vec(w.actual_canonical_qd_rad_s.map(v=>v.min))}..${vec(w.actual_canonical_qd_rad_s.map(v=>v.max))} |\n`;
md += '\n## 源值、所有权与物理结果\n\n';
md += '- Recording P01 从 0.33333333333333215 s 起完整四轮为 [0.3,0.3,0.3,0.3]，13.266666666666424 s 有显式四轮零。P02 所有完整快照四轮为零，但 changed_channels 无轮不能据此抹掉跨阶段正在执行的旧 owner。\n';
md += '- A 的 P01→P02 在命令 tick1600，P02→P03 在1664；P02 实际最终四轮零。成功 zero 的任务标签在物理 tick16 就进入 P02，P01 源建议继续，P02 含四轮 0.3 的真实连续下发；不能把两者相同任务标签当相同源时刻。\n';
md += '- Recording P03 全快照为 [-0.79,0,0.61,0]；原成功 FSM 配置既有 RL −0.149 逻辑幅度修正得到 0.51911。A 的 FR/RR 为零；zero 的 P03 FL/RL 接管，但 FR/RR 继续先前 0.3，因此实际 N 为 [-0.79,0.3,0.51911,0.3]。这是差速与继承，不是四轮必须同速。\n';
md += '- A 日志 nominal_full12 字段本身已含上述逻辑修正，不能误称它是未经修正的 Recording；原始完整快照另存 JSON source_contract。A P03 的 1.2 s 显式零在后步1809生效；zero 在1504已完成该任务阶段，不能用相同 local phase time 强迫复现 A 的 stop。\n';
md += '- JSON command_segments 保留所有完整四轮 source（A）、N、mapped、native physical targets；每窗口另列逐轮 ground/obstacle active、pair_verified 计数、法向力范围和均值。实测 qd 不等于命令，AIR 不算承载；不据单轮速度大小推断打滑或接触故障。\n';
md += '- 本版旧日志没有记录 P01/P02 各源层逐拍 owner。代码允许 P01 held 与 P02 物理 approach assist 贡献同一 0.3；不能从 N 值唯一反推当拍是哪一路。已明确保留这个观测缺口，未伪造 owner 字段。\n';
md += '- A 命令 t→后步观测 t+1；zero native t→同 tick 物理观测 t。JSON 已严格核对命令和观测一致，不能使用错误同拍拼接。zero 已扫描前缀的全12 raw、projected 和反事实 native effect 均为零，mask 全12开放；本对照不代表 PPO 学会。\n';
md += '\n## 实测接触小表\n\n下面是 exact body-pair contact 的活跃样本数，四轮顺序仍 FL/FR/RL/RR；各窗口所有 ground/obstacle pair_verified 均为 true。OBSTACLE 本身不等于已取得任务 TOP 放置信用。\n\n';
md += '| 窗口 | 样本数 | ground 活跃数 | obstacle 活跃数 | ground 平均法向力 N | obstacle 平均法向力 N |\n|---|---:|---|---|---|---|\n';
for (const [label, windows] of [['A',aWindows],['zero',bWindows]]) for (const w of windows.filter(w => /P02_all|P03_all/.test(w.label))) {
  assert(w.contact.every(c => c.ground.pair_verified_samples === w.samples_120hz && c.obstacle.pair_verified_samples === w.samples_120hz));
  md += `| ${label} ${w.label} | ${w.samples_120hz} | ${vec(w.contact.map(c => c.ground.active_samples))} | ${vec(w.contact.map(c => c.obstacle.active_samples))} | ${vec(w.contact.map(c => c.ground.normal_force_n.mean))} | ${vec(w.contact.map(c => c.obstacle.normal_force_n.mean))} |\n`;
}
md += '\n两者 P02 的 FR 均是全窗 AIR；zero FR 仍实测转动接近目标0.3，但地面/障碍物载荷均为0，不可虚构该腿承载。zero P03 FR 从AIR进入真实障碍物接触；此表不单独替代现有任务判定。\n';
md += '\n原始路径、各窗口原始行号、有限读取边界、所有接触与完整数值见同名 JSON。没有扫描整个 Recording，没有重跑 A，没有读活跃仿真流。\n';
fs.writeFileSync(path.join(outdir, 'wheel_reference_zero_windows.md'), md);
console.log(JSON.stringify({ output: path.join(outdir, 'wheel_reference_zero_windows.json'), A_segments: result.A.command_segments.length, B_segments: result.B.command_segments.length, A_windows: aWindows.length, B_windows: bWindows.length, reads }, null, 2));

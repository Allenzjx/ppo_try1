# P05 recross：安全边界后的最小 patch plan（未应用）

2026-09-22。复核现生产函数后的精确建议，补充并收窄 `P05_recross_recovery_review.md`。当前唯一 Isaac 录像不受影响；本轮仅读取源码并写此输出文件，没有运行 Python、测试、FFmpeg、仿真或编辑 src/config/tests。按主代理随后提出的缩减方案，**不增加 FL feedback、backend 接线或预算 schema**；合法当前 capture 已覆盖需要继续的下降路径。

## 1. 建议显式版本与范围

待主代理确认的命名（不是已采用配置）：

```yaml
nominal:
  p05_preedge_approach_recovery: p05_preedge_same_air_recross_v2
p05_finite_recovery_timeout_semantics: current_capture_or_completed_handoff_after_original_window_v1
```

两个值作为一对严格验证；原 `p05_preedge_approach_recovery_v1` 和未启用模式保持旧行为。此次变化只涉及 P05 同一 AIR 尝试回撤后的 wheel advice，以及原恢复窗口耗尽后的 P05 局部豁免，不更改 Q/C/P、RR assist、reward、mean/sigma/caps、P06+、物理或 source 记录。

## 2. stateless recross 的字段均由现 evaluator 真实提供

`TaskEvaluator.snapshot` 合并 `history.event_ticks`；`observe` 每 physics tick 更新 `current_legs.FL.consecutive_air_samples`，并写 `active_attempt`。普通阶段切换不清这些历史。

| 用途 | 实际字段 |
|---|---|
| 当前 episode tick | `task.physical_evaluator.physics_tick` |
| 第一次真实 FL cross tick | `task.physical_evaluator.history.event_ticks.front_edge_crossed.FL` |
| 当前连续无 ground/obstacle pair 的 AIR 样本数 | `task.physical_evaluator.current_legs.FL.consecutive_air_samples` |
| 当前 attempt、AIR、接触、几何 | `current_legs.FL.active_attempt/air/ground_contact/obstacle_pair_active/top_surface_contact/within_top_xy/within_lateral_span/clearance_m/front_distance_m` |
| 不得授予或改写的历史 | `history.active_lift.FL/front_edge_crossed.FL/placed.FL` |

AIR counter 在任一真实 ground 或 obstacle pair active 时归零。cross tick 在历史首次 cross 时记录，之后不会因回撤重写；故 `air_start = tick - air_count + 1 <= cross_tick <= tick` 可保守证明当前 AIR 段包含该 cross。要求三项为严格整数、非 bool，`1<=air_count<=tick+1`、`air_start>=0`。缺失/未来/非有限/错误类型一律不允许 recross。

不能只加 `active_attempt=true`：FL 已 cross 后的 GROUND 不在现有 attempt 清空条件内，旧 attempt/Q 可能保留。保留该字段作必要证据，同时用 AIR streak 排除跨过→落地→再 AIR 的旧史借用。此窄分支也不接纳 cross 后轻触 obstacle 再 AIR；该类恢复本轮不扩展。

## 3. 具体函数与最小插入点

### `semantic_supervisor.py`

1. 常量区/`_p05_preedge_recovery_enabled()`：认可上述 v2 与配套 timeout 值，拒绝未知或错配；保存 mode 而不改变旧布尔 opt-in 行为。
2. `NominalMotionProvider._p05_preedge_recovery_status()`：仅 v2 把 `not_crossed_or_placed_FL` 拆为 `unplaced_FL` 和 `first_approach_or_same_air_recross`。first 分支仍为旧 `cross is False`；recross 分支要求 history cross=true、active_attempt=true、同一 AIR streak，以及当前在前缘之外 `within_top_xy is False && front < -xy_tolerance`。其余原 checks 原样保留。
3. 诊断返回增加 branch、air_count、air_start_tick、cross_tick、same_air_span_valid，不修改 evaluator/history。`evaluate()` 的原 wheel-only 插入点继续位于 source advice 后、mapper/residual/dispatch 前，不移动优先级或重执行 source。
4. `TaskStageSupervisor.observe_and_update()`：在原 `local_warning_only` 构造处，仅 v2 且当前 P05、`age>=window_end` 时替换原无条件 P05 term。使用该函数已经计算的 `continuation`、`entry`、`goal_values`；不新增 feedback/state，也不再推进 assist/provider。P06+ FL pending term 和 P09 RR recovery term 不变。现有安全→global deadline→local deadline顺序不变。

恢复窗口仍由原 spec 计算：`start=P05.maximum_task_duration`、`end=start+local_timeout_policy.maximum_extension_s`，即 `[30,40)`。recross、再次后退、暂时失去净空都不续期。原距离范围为 [−100,+5] mm、gap>0；不新增 20 mm 硬门。

## 4. 无需新增 FL feedback 的 finite failure 判据

反例核查结论：在**有效 P05 入口**（当前 physical_valid + FR 已 placed）中，安全有效的 FL AIR 下降需要历史 Q/cross、尚未 placed、当前合法 TOP XY/lateral、真实其他支撑及 gap 在原捕获区域。它们已经满足 `_capture_continuation_status().allow_capture_continuation`；无需在此再给 DESCEND 第二张许可。若 FL 已 placed，P05 completion goal 已满足，须保护到下一 15 Hz 正常 handoff。

字面 mode=DESCEND 并不严格蕴含 allow=true：assist 内部旧 mode 可以遇到当前 gap 低于 `top_gap_min_m`、当前资格/入口不合法或刚变坏的接触。然而这些不是需要保留的**当前安全有效 capture**反例，不能仅因 mode 名称为 DESCEND而豁免任务时限。预算耗尽但当前几何仍合法时，允许已有 P05→P06 捕获接管，而不是在P05继续无限等待；原调度在下一个 `tick%8==0` 执行接管。

建议仅替换 P05 term，伪代码为：

```text
completed_handoff_pending = entry.valid
    and evaluation.valid and self.termination_reason is None
    and all(value >= 1 for value in goal_values.values())
    and current_tick % 8 != 0

P05_warning = age < (P05.maximum_task_duration + maximum_extension_s)
    or continuation.allow_capture_continuation
    or completed_handoff_pending
```

goal 等待分支必须带 `entry.valid`，避免无效入口只凭旧 placed 无限豁免。若当前已到 handoff tick且真实完成/入口合法，前面的正常 transition 已经把 stage 改为 P06，不需新增 P05例外。

`allow_capture_continuation` 必须使用原函数的**当前** XY/接触/支撑结果，不是历史 pending。age>=40 且上述后两项均 false，原有限 local deadline arithmetic 自然落到 `INCOMPLETE_CONTROLLER_BLOCKED` + 原 `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`/`LOCAL_TASK_DEADLINE`。不覆盖既有 physical failure，global200始终先于local，也不修改P06+的现行规则。这只关闭 P05窗口耗尽后的无路径豁免，不把到30秒直接设为失败。

### 明确不需要修改的接线

`semantic_backend._atomic_apply()` 的 ACK已经有 `capture_assist_evidence.state_after`，`_make_frame()` 也已把 FL snapshot放进actor info；本次无需新增 `fl_capture_feedback`、提交tick字段或纯预算validator。该方案避免重复复制20°/2°/2s/0.2mm/3°参数，FL/RR assist算法完全保留。

## 5. actor 已观测什么，尚未单独编码什么

当前410布局原样包含：FL assist 12 状态位于 `[372:384)`；原 FL scheduling 5 项 `[384:389)`，其中已有 `allow_capture_continuation`。stage elapsed 是现 `task_times` 原始有限值；完成placement、phase、当前hip/gap也在现观测。新的有限 P05判断全部复用这些当前任务结果，不需要actor新维度或新增动作历史状态。FL assist的窗口/hip目标状态依旧正常可见，但本次不让timeout单独消费它们。

AIR streak精确长度与首次cross tick没有各自专用actor字段；历史cross、当前接触、cross后elapsed和本次effective nominal已在现观测中。不得称410因本改动成为完整Markov状态；本方案没有新增mutable计时/latch，只用已有传感历史决定有限 nominal建议。若以后要求显式新增其编码，应另行版本化，不能偷偷占用旧列。

## 6. 应在当前录像结束后运行的少量定向例（现在未运行）

- v1完整回归；v2 first approach与原控制逐位相同；同AIR的cross→退至−18mm允许；tick160/cross100/count61通过，count60拒绝。
- cross→GROUND→AIR、cross→obstacle touch→AIR、缺失/伪类型/未来cross、placed=true均不放行；所有旧gap/横向/支撑/终止反例保持。
- 29.999/30/39.999/40边界；旧 source endpoint时序、fresh explicit stop优先；仅wheel prior改变，其他source/servo/HISTORY不变。
- age>=40：当前合法AIR capture/真实TOP capture保留下一handoff；真实placed在非8倍tick保留到下一边界；entry无效不能仅凭placed取得无限豁免。
- 不读取任何assist mode字符串：即使旧snapshot写DESCEND，只要当前allow=false、无完成handoff且窗口耗尽仍finite incomplete；安全有效DESCEND的当前几何可通过原allow，不被误终止。
- 200秒及真实body/safety仍优先；同拍重复调用不再推进状态；P06+与P09新RR恢复结果逐项保持旧规则。

预计行为修改只需 supervisor 和一个 task spec（另有必要严格迁移接线）；**backend也无需修改**。可以完全不改 actor、FL/RR assist算法、codec、reward、分布或 reference资产。新nominal/终止语义必须明确新control版本、保留最新兼容全状态和所有谱系、fresh rollout；不以此方案或合成测试声称真实越障成功。

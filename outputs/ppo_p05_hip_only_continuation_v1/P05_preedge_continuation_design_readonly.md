# P05 pre-edge continuation：有界设计审查，未实施

建议：可以在新显式版本中复用现有 **P06-class wheel suggestion**，不必仅等待 policy 自行补足。但最小、风险较低的路径是**保持 P05，增加有限 pre-edge nominal recovery owner**，而不是把现有 `allow_capture_continuation` 一处改成更宽的 OR。合法源 `wheel stop` 不是 bug；首次问题仍是当前轮腿策略未在有限 source 前送窗口内完成 approach。

## 推荐的窄候选

新增独立 opt-in mode（建议名 `p05_preedge_approach_recovery_v1`，尚非已部署接口），保留现有 capture-continuation mode/5 个观测位的原义。仅在以下条件同时满足时，为当前 P05 nominal 的四轮提供已有 `[.3,.3,.3,.3] rad/s` 建议：

- P05，局部原期限已到，原有限 source endpoint 已实际 dispatch；FR 历史 placed、FL 历史 qualified lift、FL 尚未 cross/placed。
- 当前 evaluator/safety 有效且无终止；FL 当前 AIR、无 GROUND/obstacle pair、lateral 合法、有限实测保守 gap > 0。不能使用允许低于台面 15 mm 的 `top_gap_min_m` 作为 pre-edge 前送净空。
- 当前其他真实 bearing support 数量至少使用原 `minimum_other_supports=2`，不指定固定两腿、不要求固定 CoM 或静止，也不增加力阈值。
- 有限前缘距离窗口。一个可审查的初始边界是复用现有 approach 尺度，`-approach_max_m <= front_distance <= xy_measurement_tolerance_m`，即约 [-100,+5] mm，并仍要求历史 cross=false。**不要在 -5 mm 的 within-XY 边界就撤销前送**，否则会重新留下尚未产生真实 cross 的死区。此范围是待实测的 advisory 适用条件，不改变 cross/placed 验收。
- 原局部期限后的绝对有限窗口，例如 `[30,40) s` 的 P05 stage age：复用配置中 10 s 上限，但不能复用当前 progress-weighted allowance 的数值（未 placed 时 P05 progress=0，那会得到零恢复窗口）。窗口不因抖动、失载、重新满足条件而续期。

现成 `NominalMotionProvider.evaluate` 已有 P05 wheel-overlap 合成点；直接在该点增加独立 pre-edge 许可，并继续走现有映射/合成/物理限制。原 P05 servo/source 时钟、nominal/residual/HISTORY 分离不变；不创建 P06/P07 source layer、不改变完成列表。fresh authored wheel event（尤其 explicit stop）继续优先。条件失效即停止**这份恢复建议**，而非屏蔽 policy；所有 12 residual 通道继续有效。

真实 cross 后退出新 pre-edge 许可并接回原 post-cross recovery；真实 placed 后走原 P05→P06 接续。窗口耗尽只记录 recovery exhausted，不能假成功、不能重启十秒计时，也不能恢复 P05 局部 deadline 的 episode 终止；全球 200 s 与真实安全条件仍负责终止。该候选仅创造一次有限的学习/恢复机会，不保证改善：负 residual 仍可能抵消 +.3 的建议。

## 为什么不能只放宽现有 allow

| 消费点 | 直接放宽的外溢风险／最小方案处理 |
|---|---|
| `_capture_continuation_status` → `entry_report` | 同一 allow 会 waiver P06–P13 的 `placed_FL`；新 pre-edge 许可不能进入这条链。 |
| `observe_and_update` 的 pending_handoff/takeover | 到期限可直接前移 scheduler；P06 又可能因 rear_approach/takeover 转 P07。保持 P05 避免新增 RR 提前 source。 |
| `_sequence_permission(P06)`、`_continuous_advisory` | 该 allow 同时控制 P06 源时钟和 wheel contribution；不要令 pre-edge 标志隐式打开后继层。 |
| `NominalMotionProvider.evaluate` | 已有 P05 wheel-only overlap、fresh stop precedence、真实捕获到 15 Hz handoff 的桥接；复用这里，不改原 Recording。 |
| `placement_predecessors_satisfied` | 旧 mode 已有独立 rear progress eligibility 语义。新 nominal mode 不应顺带改它，不能把 new pre-edge 许可授予 RR Q/C/P 或新奖励资格。 |
| `SemanticCaptureAssist.update` | WAIT 要求 FL cross+withinXY；保留，不将“前送许可”伪装为“已可垂直捕获”。 |

现有 global physical Phi 已给真实 FL lift/carry/水平推进信用；`_current_capture_progress` 仍要真实 cross 才给捕获接近。维持这些资格、reward 系数与 evaluator 不变。本项是控制/调度恢复，不是 reward 修正。

## 389 观测、恢复与有限文件面

最小方案不改 389 维、policy kernel、rho、sigma、caps 或 actor 类。`task_times.stage_elapsed_s` 在 index 20，scale=200（30–40 s 为 .15–.20，未饱和）；nominal12 已可见。pre-edge 时原五位可保持 `[pending=0, allow=0, scheduler_advanced=0, warning=1, elapsed=0]`；窗口来自已有阶段时间，当前合法性来自已有几何/接触，实际恢复建议从 nominal 可见。不要新增隐藏 first-trigger/retry latch；若需要新的隐藏状态或重解释五位，就不再属于此最小方案，必须另做观察合同迁移。

潜在实现面仅：`semantic_supervisor.py` 的独立 mode 验证/纯状态许可/`NominalMotionProvider.evaluate` 合成及诊断；task-spec opt-in 字段；专用 same389 控制迁移模块与 `semantic_migration.py` 路由、`semantic_training.py` 正常 save/carry 接点；定向 tests。**不修改 evaluator、Recording/fsm source、capture-assist 动作、mapper/adapter、reward、actor/distribution/observation schema。** 具体行数/文件数需以最终候选 diff 为准，本报告不授权应用。

必须在活动 episode 和完整 update 安全边界后迁移：真实最新 CP，完整 actor/critic/Adam（包括实际 adaptive LR）/Identity/RNG/累计计数保留，旧 rollout 清空。这是 `controller_transition_semantics_changed=true`、`same_mdp_claimed=false`，不是旧 RR reward-only factor 或 metadata-only 续接；新增独立 receipt/branch，不覆写 RR v1/v2 branches、全部 origins、front_rehearsal_auxiliary 四事件或旧 task auxiliary 7/8。可复用现有 strict same389 publisher 的 full-state 验证能力，不复用不相符的旧许可。

后续 prefix 使用已在目标 runtime 正式保存/重载的 checkpoint；不能把旧 CP 直接配新 nominal 标作 exact resume。保留 `_request_history_prefix_provenance` / `_source_record` 的现有严格来源校验，不为此恢复建议扩权或放宽。

N_ref 原件不动；新 B/C 都使用同一新 nominal recovery（不按 residual==0 分支）。改善若来自恢复建议必须披露为控制版本变化，不能全部称为学到的 PPO 收益；旧 B 不自动构成“同控制版本”证明。

## 必要正反例（实施前定向，不是成功门槛）

正例：有效 pre-edge AIR/positive gap/两真实其他支撑/期限与有限窗口/source endpoint 俱满足 → 仅恢复 wheel nominal；原 stop 先已执行，residual 仍能加减。cross→原 post-cross、placed→P06 均不重置 mapper/H。

反例：未 qualified、GROUND/墙接触、gap<=0、越侧界、过远、缺测/非有限、当前安全终止、支撑不足、尚未到期限、窗口耗尽、fresh authored stop → 无新恢复建议。禁止新完成/placed/cross credit、禁止 pre-edge 许可放行 P07/RR 源。窗口失效再恢复不能续期；policy 抵消是允许行为而非执行错误。CPU 检查同 raw 概率/全12通道、严格源迁移和 save/load carry；之后仅做有限真实验证，不能把该建议当已证物理成功。

# P05 连续 AIR 回撤：最小 recovery 修订建议（未实施）

2026-09-22；只读当前生产代码与父任务提供的 live 测量。未改 runtime/config/test，未启动 CPU 分析 helper、Isaac 或优化。当前录像必须自然封存；以下不是对 live episode 的修改或成功判定。

后续收窄说明：第3节的独立FL反馈建议已被 `P05_recross_patch_plan.md` 的更小方案取代。合法当前下降已由现有 `allow_capture_continuation` 覆盖，已完成等待交接单独保护；选定方案不增加FL feedback、不改backend。以下保留为原始审查过程，不作为最终实施范围。

## 1. 当前拒绝原因与字段

父任务已测：P05、FL gap 约 +6.9 mm、front 约 −18 mm、连续 AIR、Q/cross 历史均 true、未 placed、源 endpoint 已写、其他三腿有支撑。`NominalMotionProvider._p05_preedge_recovery_status()` 的 `not_crossed_or_placed_FL` 要求历史 cross=false，因此拒绝；已有 `_capture_continuation_status()` 又要求当前 `within_top_xy=true` 才允许后续捕获。两分支之间，历史已 cross 而当前退到前缘之外的合法 AIR 状态没有重新前送建议。原 source stop 本身不是错误。

原 preedge 条件必须保持：P05；stage age `[30,40)`，不续期；当前有效无终止；FR 历史 placement；FL 未 placed、当前 AIR 且无 ground/obstacle/TOP contact；gap>0；lateral 合法；front 在 `[-approach_max_m, xy_measurement_tolerance_m]`；至少两条 verified 其他支撑；源 endpoint 已发且后一连续 tick；fresh source wheel owner 优先。现 spec 的距离区间实际为 **[−100,+5] mm**，不是 20 mm。−18 mm 是此次状态量级，不新增 20 mm 门。

字段路径均已存在：

- `task.stage_elapsed_s`、`task.stage_id`。
- `task.physical_evaluator.physics_tick`。
- `task.physical_evaluator.history.active_lift.FL`、`.front_edge_crossed.FL`、`.placed.FL`。
- `task.physical_evaluator.history.event_ticks.front_edge_crossed.FL`。
- `task.physical_evaluator.current_legs.FL.active_attempt`、`.consecutive_air_samples`、`.air`、`.ground_contact`、`.obstacle_pair_active`、`.top_surface_contact`、`.within_top_xy`、`.within_lateral_span`、`.clearance_m`、`.front_distance_m`。

## 2. 仅添加 stateless recross 分支

保留旧 v1 分支与所有其他 checks；新显式 mode 中把历史条件拆成 `unplaced_FL` 与 `first_approach_or_same_air_recross`：

```text
first_approach := history.cross_FL is False
air_start_tick := current_tick - consecutive_air_samples + 1
same_air_recross := history.cross_FL is True
    and current_FL.active_attempt is True
    and current_tick/cross_tick/air_count 均为严格整数（非 bool）
    and 1 <= air_count <= current_tick + 1
    and 0 <= air_start_tick <= cross_tick <= current_tick
    and 当前确实在前缘之外（within_top_xy=false，front < -xy_tolerance）
eligible := 所有原当前几何/支撑/时间/源 owner checks
    and history.placed_FL is False
    and (first_approach or same_air_recross)
```

到达当前 TOP XY 后，不再消费 recross 分支，继续使用已有真实捕获/下一阶段接管；绝不以旧 cross 代替当前 XY，也不擦除或重造 cross/placed。输出仍仅是原 wheel prior，保持 12 通道 policy、servo、mapper/HISTORY、限速与物理硬限。建议诊断新增 `branch=first_approach|same_air_recross`、air_count/start_tick/cross_tick 及拒绝原因；不加 latch、stage 或 AUX。

**为何 active_attempt 单独不够：** evaluator 只在 `GROUND && (!history_cross || leg==RR)` 清空 attempt。FL 已 cross 后发生 GROUND，旧 `active_attempt`/Q 可能仍为 true；它不是独立的当前连续尝试证明。因此此窄修订要求当前 AIR streak 覆盖原 cross 事件。这是保守范围：cross 后轻触 TOP/壁面再 AIR，即使未来可能可恢复，也不在首个 stateless recross 分支中借旧史放行。

## 3. 原有限窗口耗尽后的最小终止建议

当前 `TaskStageSupervisor.observe_and_update()` 的 `local_warning_only` 对所有 P05 无条件 true；没有当前 legal capture、没有有效恢复时，也会一直豁免局部终止直到 global200。这个事实独立于 policy 是否主动后退。

最小改动限新 mode 的 P05：保留原 `[30,40)` 恢复机会，不因短暂前缘状态而提前重建 deadline；**age>=40 后**，只有以下现行有效路径继续局部 warning：

1. 当前 `allow_capture_continuation=true` 的真实合法捕获路径，等待原 15 Hz handoff；这不授予 placed。
2. 当前合法 capture 几何/支撑下，已提交的 FL assist 正在有效 `DESCEND`，且尚有原旅行/余量/进展窗口预算，非 BLOCKED/stale/retired。不能仅由历史 cross 或任意字符串 DESCEND 推断。

若两者都不成立且原有限局部额度已耗尽，回到既有 `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`（原 allowance 为零时保留 `LOCAL_TASK_DEADLINE`）路径；属于任务未完成，不是 body collision，也不是人为成功。全局 200 s 与真实硬安全始终先于局部豁免。P06+ 现有 FL continuation 与新 P09 RR recovery 语义本次不泛化修改。

接线注意：supervisor 目前没有已提交 FL assist feedback，只有新 RR feedback。FL snapshot 已存在于 backend `frame.info['capture_assist']`/现有 actor 编码；如果 timeout 要消费它，应沿现有 RR 一次派发→提交 snapshot 的模式，复用同拍完整 FL snapshot 与 tick，而不是在 supervisor 再 `advance` assist 或假造 mode。不能让 provider 为了判断 timeout 再执行一次源时钟。

## 4. 定向正反例与版本范围

- 正例：旧 first-approach v1 行为不变；当前 +6.9 mm/−18 mm、同 AIR streak 覆盖 cross、未 placed，在原窗口内允许原 wheel prior。
- 连续性边界：tick160/cross100/air_count61 可用；air_count60 起点101，拒绝；缺失/非整数/未来 cross、负计数均拒绝。
- 历史漏洞：cross→GROUND→AIR，即使 Q/active_attempt 仍 true，也拒绝借旧史。placed=true 永不走 recross。
- 当前物理反例：gap<=0、壁/TOP/ground contact、横向越界、支持不足、无效/终止均拒绝；当前合法 XY 回交既有 capture，不重写 history。
- 时间/owner：39.999 可用、40 不可续；再退回不重置 age；fresh stop 胜出；未发 endpoint/跳 tick 拒绝；servo/phase/mapper/HISTORY 不变。
- timeout：age>=40 无合法路径或仅 stale/BLOCKED/耗尽 DESCEND 时未完成终止；同拍有效 DESCEND 与合法 capture 仍可继续；global200/真实碰撞仍优先。

建议以新 preedge mode/control 版本显式采用，不改变 v1；属于 nominal 与终止语义变化，应保留兼容权重、完整训练状态和谱系，清空旧 rollout 后重新采集。旧成功 N 未进入该回撤分支的 trace 只能证明分支不激活，不能代替新版本 B 实测。此建议不保证 PPO 被修好，也不把前腿重新优化作为后腿诊断的新门禁。

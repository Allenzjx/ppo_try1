# RR 保持之后：最小 FR 接收 / RL 接续候选

2026-09-23；仅静态源码与已封存报告分析。当前 v9 真实采样未被修改；本工作新增 **0 仿真、0 actor forward、0 PPO/AUX**。这是 v9 若能保住 RR、但 RL 仍不能卸载时的后续选择，不是已实施或已验证动作。

## 事实与假设

- v8 首个课程：P10/P11/P12 入口 RR 都实际 TOP 承载；首次失载发生在 P12 合法 −0.3 四轮脉冲期间。因此只加阶段入口承载门，不能阻止该失载。后续再 TOP 后仍持续 FL 负残差、退回前缘是 v9 当前处理的独立窗口。
- 成功 N：FR knee 实测约 +30°是**此前持续构型**，P11 新源主要为 FR hip；同期 N 的 FL knee 从约 −12°走向 −31°。C 的 FR knee 约 −20…−35°、FL knee 长时 −58°是实际差异，**不是“FR 必须全为正”或“N 已证明 FL 正向回收有效”**。
- N 的 FR 在 P11/P12 早期可以 AIR；FL/RR 同时实际承载。FR 接收空间不等于 FR 已接收体重，两点承载不等于静态稳定证明。C 的 FL 负向余量不足是可信诊断线索，向较不负方向回收能否改善支撑力臂仍待实测。

## 最窄可执行设计：先分离测量，再决定是否建立 lane

```text
RR 当前 TOP 承载 → 小量 FR/FL 接收空间诊断 → 实际支撑与行程响应
        └─失载→ RR 连续 recapture；不删除 placed，不冒充当前 support
若确需接续修订：RL 未起摆节点独立许可 → RL 当前 AIR 后连续 carry / touchdown
```

下一次有限诊断由**同一真实连续前缀**到 RR 当前 TOP，保留 v9、wheel 脉冲/stop、RR captured-follow、原 all12 策略。先独立测 FR knee 较不负的小量、限速增量，另测 FL knee 正向有限回收；相对当时 **FINAL** 定义一次有界目标，不每拍累计、不跳到 N 的 +30°/−31°。原 P11 FR hip 保留并单列混杂。方向有利且支撑可用再测协调，不先自动整套角度表。诊断接管须公开 owner、预算、退出/applied action，不算网络自发成功。

若确认“新 RL 节点在重构前已被消费”才加最小 **P12 RL 两关节 owner lane**：

1. P12 通道 4/5 独立消费源游标；原其他通道时钟继续。首个 RL knee 在源 t=0，必须在首拍前检查，不能只查 wheel atomic_groups 或后续 hip。等待持有当前有效 owner 请求，不恢复历史入口；恢复只发下一个未消费节点，不追赶或重播。源正常 bias/tracking 必须跟对应 lane 时钟。
2. 新地面起摆需要当前同拍有效 RR TOP/pair/XY/bearing，且 FR/FL 至少一条真实支撑；**不把 RL 本身计作卸载后的桥接**。现 `rl_transfer_ready` 的 other_supports 包含 RL，不能直接作为这条许可。FR AIR + FL/RR 承载应允许；不增加固定角度、三点模板、静止时间或正 support-margin 门。
3. RR 失载只阻止依赖它的**尚未发出的新卸载**；RR 捕获、FR/FL 有益准备、wheel pulse/stop 继续。RL 已真实 AIR 后不能冻结全部 carry/下降；用当前尝试而非历史 active_lift 选择连续落脚分支。若需要该分支的记忆，必须显式公开。仅暂停 source 不限制 policy/全身运动造成的卸载，不得声称已全面约束 RL。
4. 最终 Full12 原子写入不变。RL 几何辅助只拥有 P12 的 4/5，在 ground/within_top_xy 时退出；它不是 FR/FL 准备。新辅助如确需采用，应只在 RR 捕获后的局部 owner 生效，接触丢失时平滑让权，不另写 actuator。

## 两项可证伪量，而不是角度相似度验收

**① 真 CoM 固定 FR 方向响应。** 诊断开始固定 `d0 = normalize((FR_center0 − CoM0).xy)`，测 `dot(CoM(t) − CoM0, d0)`，并列 base/yaw、FR 轮心位移、RL 载荷/净空。只有脚移近/body 转向，没有 CoM 正向进展或 RL 减载，就不能宣称完成转移。现 role tracker 是滑动窗锚点；固定锚点另列诊断，不改奖励。

**② 有效行程—接触响应。** 同步 FL/FR 的 N、raw residual、请求、裁剪/整形、FINAL、actual/双向余量，及 RR/FL TOP 力、RR front/gap、RL force/air。FL actual 较不负、负向行程增加且 RR/FL 支撑可继续利用，才算候选改善；只有 target 变/仍裁剪，或以 RR 掉地换余量，均证伪。瞬时失载与持续失载分开。

## 观测与实现边界

410 已含实际关节/轮速、接触力、CoM、48 个角色字段、nominal/mapper/HISTORY、14 个 RR assist 状态和 7 个 RR 任务位；足以报告上述当前响应，**不足以观察新独立暂停的 RL 源状态**。`source_partial_order.layers` 目前仅日志 P07–P09，日志新增 P12 也不等于 actor 已看到。

若 lane 被采用，最小新增公开类别为：`rl_consumed_source_tick`（未创建哨兵/明确 endpoint）、`p12_main_source_tick`（wheel/其他 owner 实际时钟）、`rl_lane_mode`（未启用/等待/推进/空中接续/退休及当拍许可）。若许可可由这些值和已有当前事实唯一确定，不重复加位；否则单列实际许可。事件 index/owner 可由公开游标和不可变源表导出，不另藏 latch。若另加持续 FR/FL 辅助，还须公开其累计偏移/保留目标/剩余预算；不能塞进原 14+7 位或只写日志。

因此 owner-lane 需新 410→410+K schema，保留旧列/完整 Adam、LR、Identity、RNG、所有谱系，新列与对应矩为零，清空未完成 rollout；不改 raw12 Gaussian likelihood。两项小诊断先决定是否值得这次迁移，不把迁移或诊断全程成功变成继续 PPO 的新门禁。

依据：supervisor 的 continuous nominal/sequence permission/source bias，rr_capture_context、transfer_roles、nominal_geometry、observation 与 residual_adapter；已指定 v8/N/v9 三份报告。迁移边界见仓库根 `POST_CAPTURE_PUBLIC_STATE_MIGRATION_OPTIONS.md`，不改旧 receipt。

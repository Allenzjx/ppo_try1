# RR capture → RL transfer：最小实施接口与后续时序设计

2026-09-22。只读设计；本文件没有修改生产、运行仿真或拟合模型。基线为 f48cd2f（父任务已核 runtime 与 6ac7 相同）。首版范围已按主代理决策收窄：RR hip-only 实际方向验证 + 当前任务 context；不先改 P09 late 原组、不改 reward、不启用 FL wheel guidance。下面明确区分首版接口与待实测后的后续候选。

## 1. 已有事实，不把相关性说成唯一原因

复用 `outputs/ppo_p05_hip_only_continuation_v1/CP218496_RR_capture_source_readonly.{json,md}`，未重扫 Recording。

| 证据 | 已确认的含义 |
|---|---|
| CP218496：6657 进入 TOP XY 容差；6658 原 RR geometry +10°/+10° 建议退出；6662 RR cross；6663 P09 late 首次派发 | geometry 退出与 whole-body 重配置相近但不是同一事件，最终 target 仍经过 slew；不能把 gap 上升全归因于其中一个。 |
| RR gap：6656 为 22.516 mm，6680 为 57.101 mm，9744 为 69.700 mm；9744 RR final `[10.619,-53.301]°`，actual `[10.340,-53.085]°` | 悬空时目标基本被实际跟踪，不是已发现的 actuator 输出丢失。policy RR hip 约 +16.6°、knee 约 −16° 的作用确实存在。 |
| P09 late 真正改变 FL hip/knee、RL hip/knee、FL wheel；RR pair 仅重发原值 −6.9°/−37.8° | RL pair 的变化不能仅因腿名就认作提前卸载；可能参与 RR 捕获。整组等 RR placed/bearing 可能切断捕获路径。 |
| accepted N_ref：RR cross+placed=6155；P10 entry=6160；P10 RR knee 第一节点=6161 | 成功 N 在 P10 前已 capture；不能称 P10 当前门为已证死锁。原 Recording 也先有 RR TOP_LOADED，再展开 P10。 |

P11 实际是 FR hip 序列 `3.7→5.8→12.2→10.1→9→6.9→5.8→3.7°`；P12 才先展开 RL knee、再改变 RL hip。今后只拦 RL 两关节会漏掉更早的接收侧转移动作。

## 2. 首版最小 context 与真实性边界

主代理已定 `frame.info['rr_capture_transfer_context']` 的 7 个布尔字段，profile `role389_rr_capture_transfer_v1`：旧 389 + RR assist 14 + 当前任务 7 = 410。建议同一 pre-dispatch 物理快照形成 context，任务判定、assist、日志均复用它，禁止分别从不同 tick 取值。

| 字段 | 建议语义与不能推断的事 |
|---|---|
| `rr_lift_carry` | 当前 attempt 已建立有效 lift，历史真实 Q 保留、当前无 GROUND，并满足已有连续运动合法条件；不要求每拍重新证明 RR 自身关节先变化。不是 placed。 |
| `rr_top_reachable` | 当前 over-TOP/横向/净空等捕获几何条件满足的操作性 proxy；和 current-Q/body-control 瞬时波动分开。不是完整 IK 可达证明，也不是接触或承载。 |
| `rr_top_contact` | 当前真实 TOP 表面接触且区域有效；壁面触碰或只有历史 placed 不算。可以先接触、后形成承载。 |
| `rr_current_bearing` | 当前 TOP contact + 现有 verified/support/force-noise 规则通过，非 AIR、非 GROUND。复用真实判定阈值，不新增任意载荷百分比。 |
| `rl_transfer_ready` | 当前 RR bearing 与已有接收/桥接支持、空间和连续性条件共同成立；用于后续新卸载许可，不替代 RL 当前摆腿后的回落恢复。 |
| `rr_capture_recovery_allowed` | 当前安全/几何/其他实测支撑合法，且真实 assist 尚有预算、追踪与有限进展窗口有效；仅有 mode=DESCEND 不足以续命。 |
| `fl_wheel_guidance_active` | 首版明确 false。不能因为字段存在就宣称 wheel 投影已实施。 |

特别注意：`semantic_supervisor._current_rr_placement_usable()` 允许历史 placed 后当前 AIR，不能直接作为 `rr_current_bearing`。未知/缺失接触或无效观测应有明确 invalid reason，不能当作零载荷测量值，也不能凭历史升级为 bearing。

建议纯 helper 内部还返回日志用证据：observation/dispatch tick、当前 gap/front/lateral/ground、TOP 与 bearing 来源、实际其他支撑列表、history Q/cross/placed 分列、实际 hip/knee 和双向余量。它们可以复用已有观测字段；不能据 7 bool 声称所有 contact/source 内部状态均完全 Markov。

RR assist 应每个真实 physics tick 仅 `advance` 一次；纯 `snapshot/apply` 用于本次 actuator 与审计重算。保留 12 维原始采样与同分布 logp；若 assist 在执行层拥有 RR hip，必须如实标注该通道最终 owner/有效 policy 权限，不以 mask 全 1 声称物理控制完全未受约束。

### 局部 timeout 接线

主代理拟把上一实际派发 assist snapshot 传入 supervisor，可避免 supervisor 再推进 assist。需同时带派发 tick；只接受明确相邻 tick 的已提交快照，并用当前安全/接触/几何再次核验。stale DESCEND、BLOCKED、预算耗尽、相同目标无进展、无效观测不能继续关闭局部失败。14 个 assist features 必须包含所有影响动作/timeout 的模式、目标/锚点、累计幅度或剩余预算、进展窗口/释放状态。global 200 s、机身碰障及独立硬安全不变。普通 phase change 不设置 done。

## 3. 后续最小 sequence 修订：按功能 owner，不按整阶段冻结

下表是待首版实测后才采用的设计，不是当前已实现的行为。

| owner / 通道 | 捕获前后建议 |
|---|---|
| P09 现有 capture-side late 原子组：FL 0/1、RL 4/5、FL wheel 8（其余为保持值） | 首版保持已有 current-over-TOP 门与源时钟。不能把整个组延后至 bearing，更不能丢弃源 stop。未来拆组会改变原子语义，须显式版本化和逐事件消费记录。 |
| 新 RR hip capture owner：6 | 在合法 AIR 捕获区作有限、测量反馈的 hip-only 搜索；从上一实际 final target 连续接入。真实接触停止下探、受控保持/释放；失去安全或耗尽预算不靠重置窗口无限重试。 |
| 无害的 FR/FL 空间预备：2/3、0/1 | reachable 时可尝试，不要求先 placed。FR 可能需收起/正向适配，FL 负端 knee 可能需正向回收余量；动作方向必须靠当前几何与实测响应，不锁固定正角，也不能仅按通道名判定无害。 |
| 后续明确 RL-transfer owner：P11 FR hip 2；P12 新 RL 卸载/起摆 4/5 | 新卸载依赖 RR 当前 bearing，而非历史 placed。若许可丢失，暂停尚未执行的卸载事件，不推进其消费游标；其他捕获、支撑、stop 继续。RL 已 AIR 后的必要落脚恢复不能整段被拦。 |
| wheel 8–11 | N 源保持 + policy residual + 原 mapper/限幅；不强制四轮等速或全程非零。fresh explicit stop 取得写明通道的所有权，恢复建议不得复活已停止源命令。 |
| P10 RR knee 7 | 只有 hip-only 实测显示有效下探不足、且跟踪/余量证据支持时，才考虑有限耦合或借用原三节点。不能因悬空自动提前整个 P10/P11；若借用，真正进入 P10 不重播已消费节点。 |

暂停“新卸载”不是冻结整个 source clock。否则后续合法 stop 会被饿死；按通道分拆若仍共用整组 consumed 位，也会丢掉延后通道或重复已执行通道。任何新增游标/释放状态须纳入新观测与 checkpoint/reset 语义，而非隐式 latch。

RR geometry 退出候选也应单独辨识：已有 +10°/+10° 建议在进入 XY 时消失，不能永久保留来掩盖失效。优先使新 capture owner 从上一实际 final 接入并用既有 slew/有限释放连续接管；不要叠加两套补偿或跳回旧入口。若之后改 geometry blend，单列 source/mapped/G/assist/policy/final/actual 和退出原因，先证明确实改善捕获，再归因。

## 4. 真实载荷转移与观测范围

角色不变：RR → 接收 FL / 桥接 FR+RL；RL → 接收 FR / 桥接 FL+RR。CoM 向接收侧移动与接收脚当前承载分开。固定方向应为该次转移开始时 `d_start = normalize((receiver_start - massCOM_start).xy)`，进展用 `dot(massCOM_now-massCOM_start,d_start)`；接收脚自身移动另记，不计入 CoM 进展。方向只是量测，不硬编码唯一运动路线。

现有 `TransferRoleTracker` 方向来自滑动窗口最早样本，窗口推进会重新锚定；它虽然已经分开 CoM 与接收脚位移，但不等于上述固定起点。首版若只新增 RR assist14+bool7、未实现固定 anchor，就应明确该项尚未完成。后续 anchor 若影响动作/奖励，必须显式保存/编码 start tick、COM/receiver 起点与方向，跨普通 phase/contact flicker 保持；只有真实 attempt 失效/新尝试才重建，不能用会在短暂接触时重置的 free-air reference 冒充稳定 attempt ID。

新 410 是控制/观测语义迁移，不能标为旧 same389/no-op。保留兼容权重、完整 Adam/LR/Identity/RNG、所有 counter/origin/AUX ledger；追加输入列与 optimizer shape 的准确处理由严格迁移实现证明，旧 rollout 清空，新状态重新采集。不能因控制修订便声称 PPO 网络已改善。

## 5. 最小检查与官方来源核对

CPU 定向正反例：reachable+AIR 不升级为 contact/bearing；历史 placed+AIR 不准新卸载；wall/ground/unknown contact 不算 TOP；接触先于 bearing 合法；相邻 snapshot 可用而 stale/budget-exhausted 无 timeout 豁免；stop 不被暂停饿死；RR capture 与 RL 新卸载 owner 分离；释放无跳变；关闭新 mode 旧前段行为保持。首版真实验证重点是 hip 搜索方向、gap/竖直速度/实际跟踪/关节余量、触碰→bearing→释放，不要求先重复五连成功。

- [Keep Rollin’ — Whole-Body Motion Control and Planning for Wheeled Quadrupedal Robots](https://arxiv.org/abs/1809.03557)：标题匹配。借鉴 wheel、腿、机身和滚动约束联合管理；并不支持把轮速非零直接等同牵引，也不提供本项目固定角度答案。
- [Perceptive Locomotion through Nonlinear Model Predictive Control](https://arxiv.org/abs/2208.08373)：标题匹配。借鉴把落脚几何可行区与全身动力学分开检查；不移植其 NMPC、地形流水线或求解器。
- [OCS2 SwingTrajectoryPlanner.cpp](https://github.com/leggedrobotics/ocs2/blob/main/ocs2_robotic_examples/ocs2_legged_robot/src/foot_planner/SwingTrajectoryPlanner.cpp)：`update()` 根据 mode schedule 构造 lift-off/touchdown 高度/速度样条，`extractContactFlags()` 从计划模式取 flags。这是计划接触，不是实测 bearing 检测。借鉴 touchdown 参考与接触模式职责分离，不照搬按时间宣告实际落脚。
- [OCS2 SwitchedModelReferenceManager.cpp](https://github.com/leggedrobotics/ocs2/blob/main/ocs2_robotic_examples/ocs2_legged_robot/src/reference_manager/SwitchedModelReferenceManager.cpp)：`modifyReferences()` 获取 gait mode schedule 后更新 swing planner；`getContactFlags()` 仍来自计划模式。不能引用它来证明已经检查真实接触。上述两文件头为 BSD-3-Clause 条款；本设计没有复制代码。
- [Residual Policy Learning](https://arxiv.org/abs/1812.06298)：是 Tom Silver 等人的该标题，不是同名近似的另一篇工作。只借鉴已有控制器与学习残差的互补分工；辅助控制表现改善与 PPO 自身学习改善须分别记账。

结论：先保持 P09 原捕获协作组，验证有限 RR hip 捕获反馈，再依据当前 bearing 接入后续 RL 新卸载许可。现有事实不支持直接宣判 P10 调度有错，也不保证新 assist 能解决整个越障任务。

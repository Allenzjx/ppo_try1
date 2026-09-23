# RR → RL 新卸载许可：有限只读设计

2026-09-23；v7 唯一 Isaac 运行期间仅阅读少量源码、既有 N_ref 报告和冻结 contract 的 P12 小段。没有 Python、测试、仿真、actor forward、生产修改或新轨迹采集。**这是待物理结果后选择的控制设计，不是已实施修复。**

## 结论与边界

当前确实没有 P10/P11/P12 的 source current-bearing 门。建议保留历史任务连续性，另对 **P12 尚未开始/仍在地面的新 RL 起摆 source owner** 建立当前承载许可。不能直接把整个 P12 layer 暂停：它与四轮有限 pulse/stop 共用一个时钟。要满足“只暂停新 RL 卸载，捕获跟随、FR/FL 准备与 wheel 继续”，至少要显式区分 RL 两关节消费游标；当前 410 观测没有这个游标，不能隐藏新增状态后仍宣称同布局无语义变化。

这是 source 许可，不是“保证 RL 永不卸载”：保留全部 12 个 policy residual 和其他全身动作时，policy/身体运动仍可能主动使 RL 离地。也不能从膝角正负给 waypoint 作已证的物理升降分类。

## 当前事实与精确接入点

| 文件/位置 | 已有行为 | 最小候选处置 |
|---|---|---|
| `semantic_supervisor.py:_current_rr_placement_usable`（约186） | 历史 placed/cross + 当前合法 TOP，或 current-Q 合法 AIR，可继续 placed_RR 任务谓词。 | **不改**，不把历史 placed 伪称当前 bearing；不令 RR 失载抹掉任务历史。 |
| `_sequence_permission`（2059；2073早退） | P10/11/12 不在现 P07–P09 序列许可中，直接 true。 | 新许可不能写成 `P10/P11/P12 and not bearing => False`；只对新 RL owner 的消费生效。 |
| `_continuous_nominal`：layer 创建（约2347）与循环（2403–2494） | 一个 `MotionExecutor.tick()` 同时推进该层全部通道；拒绝时保留整个旧 sample。`touched` 累计，后层覆盖前层。 | 若采用，P12 分离 RL 4/5 的独立游标与原其他通道游标；不推进被拒 RL 游标，不追赶、不丢事件，也不能用旧 Full12 重盖其他10通道。保留一个实际 Full12 派发。 |
| `MotionExecutor.tick`（`fsm/motion_executor.py:226`） | 根据 source tick 选持有 waypoint，并发出该 tick 的 atomic groups；每次调用递增。 | 不改公共 executor；候选 source-view/owner 分拆必须保持原有限节点/比例/相对 RL 时序，而非重建动作。 |
| `_source_normal_bias`（约1955）、`_continuous_nominal` tracking/bias（2450–2493） | normal bias 的终止随 sample.elapsed；tracking 随 sample owner。 | RL 延后则仅 RL bias/tracking 必须跟其游标；不能让主 wheel 时钟提前耗尽 RL 的有限 bias，也不能在暂停时恢复旧入口。 |
| `semantic_rr_capture_context.py:verified_current_support/rr_capture_transfer_context` | 已有严格 RR 当前 TOP/pair/XY/bearing 与 `rl_transfer_ready`。后者 `other_supports` 包含 RL。 | 复用同一 current-support 判据；新 RL 卸载检查排除 RL 自身，不能只信 `rl_transfer_ready`。 |
| `semantic_observation.py:363–419`、`semantic_rr_capture_profile.py` | 410 包含 stage age、当前 N/mapper/history、14 assist 状态、7 task bool；**无 P12 延后 source 游标**。 | 显式新观测/版本，见下文；不是复用一个无关 bool 隐藏时钟。 |

**P12 tick0 注意事项：** 冻结 `configs/recording_motion_contract.json:25342–25440` 先有 entry，再在 source t=0 出现 `rear_left_knee` 19.4→22.6，随后 t=.066667 到31.1。这些单关节节点记录在 `waypoints[].atomic_channels/changed_channels`。该 phase 的 `atomic_groups`（约27000）仅有 t=.466667 与2.666667 的四轮组。只查 `atomic_groups` 会漏掉第一次 RL 动作；许可必须在 P12 创建后的首个消费前检查。四轮组必须完整保留同拍原子性；不要按膝角符号过滤节点。

## 候选许可与动作分类

1. 只在 P12 未捕获 RL 的新起摆 source lane 检查；P09 late 仍有帮助 RR 捕获的 RL/FL 全身动作，**不反向把它们全列为 RL 卸载而冻结**。P10 RR knee / RR assist captured-follow、P11 FR 空间准备、既有 FL owner、wheel pulse/stop 各自继续。不得用 P11 阶段名推断 FR 已承载。
2. 许可需要同一当前 observation/task tick、valid/no physical/task abort；RR 必须严格当前 verified TOP bearing、pair 和当前合法 XY/noGROUND，不接受 history placed、合法 AIR、weak raw contact 或接近台面代替。要求 FR/FL **至少一个**按现 force floor 验证的当前支撑；RR 本身是卸 RL 后另一个支撑。无需 FR 必须承载、固定 FL+RR 模板、固定 load fraction 或新增静态稳定余量门。
3. `rl_transfer_ready` 可记录为已有准备证据，不宜未经实证再叠一个新的必过 hard gate：它包含角色准备窗，且其 other-support 计数可能只有 RL 本人。最窄修订是 current RR bearing + 非 RL 桥接证据，不再更改准备/完成谓词。
4. 等待时持有此前实际生效的 RL nominal owner 请求，mapper/HISTORY/residual 照常，不回写历史入口。源 wheel clock 继续且显式 stop 优先。再次许可时从下一个未消费 RL 节点继续，不能按墙钟跳过未执行节点，也不能一次追赶多节点。
5. RL 已真实 AIR 后不能因 RR 失载就冻结其全部摆腿/落脚恢复。当前 generic RL 没有 RR 特有 `current_lift_valid/motion_continuation_allowed` 字段，不能误用 `.get()` 造成永远 false。可据现 `air/noGROUND/active_attempt/consecutive_air_samples/initial_clearance/history event_ticks` 区分当前尝试；历史 `active_lift` 本身不够。若需明确“一次起摆已经发出”的状态，应由公开 lane cursor 推导，不能另加未观测 started latch。任何新的继续/暂停分支先用既有实际事件作回放反例，不能冻结必要安全恢复。
6. 不修改 RR/ RL placed/cross 判定、阶段完成/奖励/200s安全上限，不因等待补信用或重置局部时限。没有当前合法执行路径时仍由现有限任务结束，而非无限等 RR bearing。

## 观测、版本与最小实现范围

当前 `source_partial_order.layers` 只记录 P07–P09（约1913），即使给 JSON 加上 P12 cursor，也**不等于 actor 已观测**。stage_elapsed 是墙钟，不是被暂停的 source cursor；同一 current N 可出现在源 holds 的多个时刻，不能从 N 唯一反推下一事件。

若采用 owner-lane：最少新增公开 RL source cursor（含未创建哨兵、明确120Hz tick/scale、endpoint语义）与当前 lane permission/continuation 状态。若实现还引入独立主 source cursor、消费位或 latch，也必须逐项公开/能从公开数据确定；先冻结实际状态机再确定最终维数，不预先保证“仅加2维就完全 Markov”。不能挪用 RR assist14 既有字段。source-view 分拆须验证未暂停时逐拍与原 P12 完全相同；分拆只改新 RL lane，不让原 stage endpoint 冒充 RL endpoint，或让后续阶段重播 RL。

因此这不是当前410普通 exact-resume开关。需要新 nominal/观测版本、明确 controller/MDP migration、兼容权重映射、完整 Adam/LR/Identity/RNG/全部 AUX 与 origins 保留、清空旧未完成 rollout；不清空网络，不重做既有迁移。改动范围围绕 supervisor owner 调度、context/观测和其严格迁移；**本报告不实现，也不要求立刻为未发生的物理问题扩版本。**

## 定向反例（待选方案后才测试）

- RR history placed + 当前 AIR：不发新 RL unload；P10 RR跟随、P11 FR准备、四轮有限 pulse/stop 同原时钟继续。
- RR当前TOP bearing + FL bearing + FR AIR：许可；不能因FR未接触拒绝。RR + RL 两脚而 FR/FL无支撑：不能把 RL 自身当卸载后的桥接。
- RR旧placed但GROUND/墙/weak raw pair/XY失效/stale/invalid：拒绝新卸载，不造TOP；原物理安全终止优先。
- P12首拍 t=0 RL knee事件可被等待，四轮 t=.466667 /2.666667 各只消费一次；长hold后恢复无追赶、无相同节点重播。始终许可时，原全部 source/最终owner/bias/tracking逐拍等价。
- RL已真实起摆/当前AIR，RR瞬时失载：不一刀切冻结落脚所需动作；GROUND旧史不得错误走“已在空中”捷径。
- 暂停跨普通阶段或 RL 实际提前捕获：没有旧RL重放，无 source hold覆盖新合法owner；all12 policy/raw Gaussian/logp不变。
- 保存/重载/reset/prefix时 cursor/permission语义一致；actor实际编码与日志同拍，未添加暗游标；课程前缀0学习信用。

## 已有物理依据，不是新方案成功证明

复用 `N_P10_P12_actual_vs_v5_RR_precontact.{md,json}`：accepted N 的 RR placed6155，P10/11/12 entry6160/6168/6176；6168、6176 FR 为 AIR，FL/RR已真实障碍接触、RL地面；6248 FR与RL都AIR，FL/RR有约14N当前接触。它反驳“FR必须先始终承载才容许准备/起摆”的固定模板，但两点接触不等于已证明静态稳定。报告源 raw OBSTACLE contact 与严格 TOP 判据仍应分列。现 v7 是否实际发生需该门处理的 RR→RL状态，等本次物理封存事实决定。

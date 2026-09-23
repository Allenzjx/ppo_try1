# RR capture → RL handoff：有限静态审查

仅静态读取当前代码、`KEY_WINDOW_OWNERS.md`、`DESIGN_sequence.md` 与既有 CP218496 源审查；当前唯一录像仍运行，本次没有改生产/配置/tests，也没有 Python、测试、编码或仿真。以下是下一合法边界的建议，不是本次 RR 已接触的实测结果。

## 已确认的结构风险

当前 RR assist 在 P10/P11 检测到真实 current TOP bearing 后，将上一 **FINAL** RR hip/knee 作为释放起点；但释放终点不是捕获保持姿态，而是每拍重新计算的 `candidate = mapped/geometry N + effective residual`。两个通道一起用 `(1−α)·capture_anchor + α·candidate(t)`，α 在 0.75 s 内到 1。原限速保留，接触丢失时会重新锚定上一 FINAL，累计预算不重置；这些避免瞬间回跳，却不能保证释放终点仍保有支撑。

**P10 source 与释放存在明确时序错配风险。** 合同的 RR knee 节点在源时间 0、0.066667、0.133333 s；成功 FSM `normal_time_scale=0.875`，即约第 0、7、14 个物理 tick，末点约 0.116667 s。P10/P11 不在 `_sequence_permission` 的 P07–P09 gate 内，assist 不暂停 source 消费。若释放与 P10 首拍同起，末节点被消费时 α 只有约 0.156：其早期动作大部分仍被 knee hold 衰减，后面主要释放向已经走完的 endpoint，而非原三节点原时序。RR hip 没有 P10 新源动作，却也同时退出捕获锚点、回到 nominal+policy。

这不是“重复写入”或已经证明的掉脚原因，而是**有限释放保证连续性、不保证目标可承载**的真实结构风险。旧 CP218496 曾有 nominal+policy 的悬空偏置；不能因此断言本次同符号必然抬脚，也不能只延长 0.75 s 就声称解决了目标问题。当前 `rl_transfer_ready` 是观测/context，不是已经安装的 P11/P12 source owner gate；历史 `placed_RR`/`_current_rr_placement_usable` 也不是当前 RR bearing。

## 后续最小动作分工

| 动作 owner | 最小处理建议与边界 |
|---|---|
| P09 late：FL 0/1、RL 4/5、FL wheel 8 的原 full12 原子组 | 保留当前合法 over-TOP AIR/TOP gate 与协作。成功 N 原组在 RR placed 后启动，旧 PPO 则在 crossed/AIR 启动；这说明入口不同，不证明整个组应等 bearing。RL pair 可能仍参与 RR 捕获，不能仅按腿名推迟它。没有进一步实测前不拆组。 |
| RR capture 6/7 与 P10 RR knee | 捕获锚点应作为真正的接收起点；RR hip 不因 P10 标签自动回到旧 candidate。P10 knee 是 RR 支撑/后续空间的双用途动作，不是可整段丢弃的“RL 动作”。若真实视频证实释放损失承载，优先把 hip 保持与 knee 支撑调整分开交接：当前 bearing 允许有限、实测跟踪的 knee 接续；承载丢失暂停尚未消费节点并保留捕获恢复，不补发过期节点。接收参考从当前 FINAL 连续建立，不重新套历史绝对入口；方向/幅度须看真实 gap、bearing、跟踪和余量，不能靠角度符号判定。 |
| P11 FR hip 接收侧转移、P12 **新** RL 卸载/起摆 | 这些才是可以等待当前 RR bearing/真实 transfer readiness 的后继 owner。不要用历史 placed 放行新卸载；也不要把已经 AIR 的 RL 必要落脚恢复一起冻结。第一真实 RL qualified lift 后 RR 局部捕获层仍按现规则退出，不无限重新接管。 |
| wheel 8–11 与 source stop | 不冻结整个 source clock。保留原保持值和写明通道的 fresh stop 所有权，P09 late 前四轮 stop、late 后 FL stop 都不能被等待或恢复建议复活。所有通道仍经过唯一 full12 原子派发、原映射/限速/硬限；不强制四轮常非零或等速。 |

最小后续优先级是**先在真正到达 RR capture 的录像中辨认 hold→release→P10 节点→bearing loss 的先后**，再决定是否只修 RR 6/7 接收参考与后继新卸载 owner；不先重排整个 P09 late，也不提前整段 P10/P11。若新增延后节点/独立释放状态，应明确 source 事件消费游标和可观测状态，不能假称源节点已原样执行或沿用旧观测 no-op 迁移。

定位：`semantic_rr_capture_assist.py` 的 `_start_release/_release_step/advance`；`semantic_residual_adapter.py` 的 RR candidate/owner 组合；`semantic_supervisor.py` 的 `_sequence_permission`、`_rr_late_reconfiguration_ready`、`_continuous_advisory`；`configs/recording_motion_contract.json` 的 P10/P11 节点及 `configs/fsm_states.yaml` 的对应 time scale。

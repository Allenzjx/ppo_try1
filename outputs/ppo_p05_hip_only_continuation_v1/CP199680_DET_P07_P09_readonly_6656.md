# 新确定性运行 P07–P09 有界只读核对

运行：`20260922T0349194627694Z_g0001c3138b0b_608ca7ebf2074e70af33a21f8423fb00`。
运行代码 HEAD：`0001c3138b0b38a7278edc288b8ddc7444815770`。
只读范围：该运行 source 的已完整写出记录，终点 episode tick **6656 / 55.466667 s**。不是最终整回合验收；未改运行代码、配置、权重或物理状态，未另启 Isaac。

## 结论

不是“RR 从未抬起”，也未发现旧 `placed_FL` 入口锁或 actuator 派发损坏。RR 确实获得自由 AIR 抬升，再于越沿前重新接地。随后 P09 的待执行 knee 源事件因当前低净空下降／已接地而保持，policy 残差仍在执行。这个有物理依据的 source 等待是实际停滞的一部分，但现有证据不证明判定器错误，也不支持取消保护；无需因此阻止计划中的自然 P01 PPO 采样更新。

## 实际事件与源动作时序

| 事件 | episode tick | 秒 | 证据 |
|---|---:|---:|---|
| FL 真实放置 | 3789 | 31.575000 | evaluator placed_FL；不是 pending 放行 |
| 进入 P06 | 3792 | 31.600000 | 真实完成 P05 后正常切换 |
| P07 标签／source 启动 | 5080 | 42.333333 | source tick 1；entry_valid=true |
| P08 标签 | 5088 | 42.400000 | source 等待 P07 实际动作完成 |
| P09 标签 | 5096 | 42.466667 | entry_valid=true；源动作未补发过期 waypoint |
| P07 的 FR atomic group 启动 | 5120 | 42.666667 | `fr_group_start_tick` |
| P08 source 真正启动 | 5280 | 44.000000 | `actual_start_tick`，source tick 1 |
| P09 source 真正启动 | 5328 | 44.400000 | `actual_start_tick`，source tick 1 |
| RR 初始有效自由离地 | 5361 | 44.675000 | unsupported free lift 3.287901 mm |
| RR 合格抬升 | 5368 | 44.733333 | unsupported free lift 8.026413 mm |
| RR 自由抬升峰值 | 5389 | 44.908333 | 25.571358 mm；仍低于台面 |
| 第一次 P09 待执行 knee 等待 | 5408 | 45.066667 | source tick 固定 80；当前抬升仍有效，但 gap −45.763212 mm、底部 vz −0.281787 m/s |
| RR 重新接地 | 5410 | 45.083333 | 真实 ground_contact；撤销当前抬升资格，尚无 RR crossing/placement |

P07/P08/P09 的 entry 均有效；source 真正启动具有先后关系，不是标签连续跳转后丢掉前驱动作。5408 的等待使用新鲜同 tick 测量、两个实际其他承载支撑；5410 后等待理由变为 `current_free_lift_before_pending_knee`。至 6656 仍 source tick 80，RR 当前 ground=true。未把 FL 悬空当作承载。

FL 首个放置后的 AIR 样本为 3794，但 **4000 又有 TOP 承载**，不能把 3794 起全部称为连续悬空。4400 时 AIR/gap 9.525482 mm；5080 时 AIR/gap 38.165456 mm。补全器在 5171 已 `RELEASED`，故 RR 抬升窗口不是补全器继续直接拥有 RR 通道。

## 实际动作响应，而非仅 ACK 成功

角度为现有统一逻辑单位 degree；轮速为原日志逻辑单位 rad/s，未把轮端位移当成自转或牵引。

| tick | RR hip 最终 target / actual | RR knee 最终 target / actual | RR 台面 gap mm | RR 前缘距离 mm |
|---:|---:|---:|---:|---:|
| 5328 | 10.406875 / 10.351628 | −12.740263 / −12.461084 | −50.900969 | −167.611745 |
| 5368 | 33.778086 / 23.717727 | −12.677542 / −12.580665 | −42.881719 | −183.887121 |
| 5384 | 46.658672 / 38.359812 | −12.713824 / −12.647489 | −26.418872 | −210.384485 |
| 5408 | 57.322178 / 52.134384 | −12.794031 / −12.777645 | −45.763212 | −227.767285 |
| 5410 | 58.600466 / 53.172341 | −12.831273 / −12.982948 | −50.179171 | −228.330400 |
| 6656 | 65.422820 / 65.784972 | −12.515004 / −11.805843 | −50.749541 | −221.964088 |

RR hip actual 明显响应，不是 RR actuator 被屏蔽；在这一连续过程中 RR 轮端却先升后降、向后远离前沿。结合全身姿态与其他关节，才可进一步识别原因；这些同期数据不是单关节因果实验。

6656 四轮按 **FL / FR / RL / RR**：

- 同拍 source N wheel：`[0, 0, 0, 0]`，有限 source/ordered wheel owner 保持；不是 nominal 丢失。
- 最终下发：`[-0.250577, +0.075541, +0.122829, -0.142095]`。
- 实测角速度：`[-0.250833, -0.069993, +0.052651, -0.049751]`。

四个最终目标都非零；目标与实际不完全同号/同幅，不能据此宣称四轮都有相同牵引。尤其 FL 此时 AIR，没有地面牵引证据。

## 接线核对与数据边界

5080–6656 共 **1577 个 native tick**：`verified`、setter/dispatch 一致、实际映射一致、独立 previous ACK 核对、assist 状态转换重建均 1577/1577；residual permission mask 全 1，RR hip/knee REQUEST 非零亦 1577/1577。记录内根位姿／根速度／外力冲量／重力直接写入计数全 0。上述证明执行链及实际响应存在，不是任务成功证明。

`capture_assist_ticks.dispatch.independent_policy_residual_effective_full12` 在本次读取窗口为 null，**未补造 effective residual**。该日志的当前 nominal 与已完成 dispatch 的 nominal 可处不同记录时刻，例如 tick 5368 的当前 nominal RR hip 45°，同期 native audit 的实际源 nominal 为 33.4°；因此本报告不将两者差值算作 PPO 当拍修正。最终 target 使用 dispatch 记录，actual 使用物理测量；真实 source nominal 用 native audit / completed decision。

依据仅为本 run 的 `stage_transition_evidence.jsonl`、`video_policy_decisions.jsonl`、`capture_assist_ticks.jsonl`、`native_tick_audit.jsonl`，以及冻结运行版本 `semantic_supervisor.py` 中 `_sequence_permission`、`_rr_pending_carry_readiness`、`_continuous_advisory`。原始记录未改写，不把正在增长文件作最终封存或整片验收。

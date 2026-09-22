# CP218496：RR 越沿后未放置的源动作核对

**P09 晚期动作没有丢失或卡在 completion gate；P10 膝展开确实尚未调度，但不能称作已证调度 bug。** 已成功 N_ref 和原 Recording 的真实事件都显示：先 RR 放置，再执行 P10。当前结果更直接呈现了不同 residual/全身构型下的悬空，尚不能归因于单关节或保证某个名义改动会修复。

仅分析完整视频 source `181848…d3e1bceb…` 的 16 个关键拍及 232 拍小窗口；复用早先 `P09_late_source_9840_10560_readonly.md` 的已有机制说明。未改生产、forward、优化或仿真。此次自然结束于 tick **9748 / 81.2333 s**，P09 `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`；RR cross=6662、未 placed。不是此前零 callback 的录像失败 run。

## 首处接管与实际控制

- tick **6657**：RR center front=−4.981 mm，进入已有 XY 容差。**6658** 已有 geometry advice 的 RR hip/knee +10° 撤回；source nominal 与当拍 residual 都未变，最终 target 经原限速逐步响应。这是既有 `within_top_xy` 退出规则，不是 PPO 当拍自行减去 10°。
- tick **6662**：RR 实测 cross，gap=23.664 mm、front=+0.596 mm，满足现有 late gate；**6663** 立即派发原 P09 late full12 组。它将 FL hip/knee `38.6/−13.4→−18.5/−31.4`、RL `28.2/0→15.4/19.4`、wheels `.3/.3/.3/.3→−1.07/0/0/0`。RR nominal 自身仍 `−6.9/−37.8`。
- tick **6879**：原 FL wheel 脉冲 stop 正确下发；之后四轮 N=0，但 PPO wheel residual 仍开放。P09 source 时钟继续，无等待 RR placed 才执行 late 的循环；终止末拍出现的 `no_live_physical_readiness` 是 task 已终止的结果，不是先前阻塞原因。

下表 RR 数值为 hip/knee（度）。`mapped N` 是日志中真实 mapper baseline，不用独立重算的 N 差冒充 policy；瞬态 final 另受既有限速／反馈影响。

| tick | mapped N | 请求 residual | 有效 residual | 最终 target | actual | RR gap mm |
|---:|---|---|---|---|---|---:|
| 6656 | −9.400 / −39.050 | +16.545 / −16.601 | 同请求 | +17.145 / −45.651¹ | +16.822 / −45.457 | 22.516 |
| 6664 | −9.400 / −39.050 | +16.557 / −16.621 | 同请求 | +8.407 / −54.421² | +14.459 / −47.838 | 25.112 |
| 6680 | −9.400 / −41.550 | +16.572 / −16.462 | +16.572 / −16.450 | +7.172 / −58.000 | +8.174 / −54.981 | 57.101 |
| 6800 | −6.066 / −37.347 | +16.430 / −15.806 | 同请求 | +10.363 / −53.153 | +9.594 / −53.489 | 74.320 |
| 9744 | −6.022 / −37.347 | +16.641 / −15.954 | 同请求 | +10.619 / −53.301 | +10.340 / −53.085 | 69.700 |

¹ 该拍仍有 geometry baseline +10/+10；² 撤回后的 final slew 尚未收敛。

末段 actual 紧跟 final，并非 RR 指令未到执行器；所有所选拍 mask 均为全 1，native 映射/最后派发目标验证通过。policy RR hip 正 residual 抵消了负 nominal，knee 负 residual 使最终膝更负。与此同时 FL knee 长期到保留限位 −58°，RL hip 也有负 residual；不能把整条动作链归因为 RR 一维。body z 从6656的73.813 mm升至6680的81.392 mm，随后回到9744的75.466 mm，而 RR gap仍约69.7 mm；这不是仅由机身整体抬高能解释的差值。完整四腿/wheel/几何表在 JSON。

## P10 的确未发，但不是已证错误

P09 completion 是真实 `placed_RR`。尚未 placed 时不会建立 P10 layer；原 P10 唯一 changed channel 是 RR knee：`−37.8→−34.6→−29.3→−27.2°`。因此这组后续动作本次确实尚待调度；“P09 late full12 重发 RR 原值”不能被误报为它已执行。

但两份既有成功证据均否定“必须先 P10 才能 RR placed”的普遍断言：

- accepted N_ref：RR cross/placed **同为6155**；P10 entry=6160；6155/6160 RR knee N仍−37.8，随后6161/6168/6175才为−34.6/−29.3/−27.2（P10旧 layer跨到P11仍连续）。这不是本轮同 HEAD 新配对评估。
- 原 `recording_motion_contract.json` 的真实事件：RR cross=7398/61.65 s；TOP_LOADED=7865/65.541667 s，force15.8335 N、gap−1.8314 mm；P10 source从67.233333 s才开始。先加载后展开有实际传感事件支持，不是只看段名。

## 最小下一步

以正在运行的**普通 P07 真实前驱 PPO 采样**为主，核对后续封存更新的 mean/REQUEST/actual 与 RR 接触结果，之后固定 checkpoint 评估；不因这份报告改 gate、reward、sigma 或暂停优化。

如后续仍重复同类 hover，可将“提前原 P10 RR 膝序列”列为**一次隔离、非学习信用的诊断候选**，而非默认修复：仅在同回合真实 RR cross、当前合法 XY/lateral、有效 AIR/continuation、非 ground、已有两条真实其他支撑且无 safety abort，并且原 P09 late 已实际下发后，按原时序一次性接续这一个 knee owner；保留全部 residual、mapper/history/限速/hard limits，绝不跳阶段或制造 placed。须记录消耗过的源事件，正常进入 P10 时复用其进度，不重播；不启动其他 P10/RL 准备动作。当前最终 target/实测一致、合法 over-top AIR，使它可作为可解释的有限干预，但现有日志**不能预证展开会降低真实 gap 或完成放置**。若实施需另行审查隔离接线与收尾，不在本报告中准备或应用生产 patch。

[结构化逐拍证据](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_p05_hip_only_continuation_v1/CP218496_RR_capture_source_readonly.json)。CPU helper 已退出。

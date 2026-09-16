# 2048 决策训练：奖励与信用核验

范围：已封存的自然 P01 训练块，2048 决策、16 次完成更新（1280–1295），最终计数 170240。只读 manifest、advantage_audit、completed_episodes 和 optimizer_updates；没有加载模型/rollout tensor、读取完整 native 流或改生产代码。训练进程完成不等于任务成功。

## P02 的真实信用

P02 共 516 样本（25.2%），覆盖 7/16 次更新。raw GAE 均值 −0.3830，395 负/121 正；保存的标准化优势均值 +0.2164，360 正/156 负。P02 即时 reward 均值 +0.001248（233 正/283 负），old value 均值 −19.8204，return 均值 −20.2035。

标准化范围是每次完整 128 样本 rollout，并非 P02 自己。尤其 update 1282/1290/1295 中，P02 相对于其他更差样本可全部得到正标准化优势，即使多数 raw GAE 为负。这与现有 PPO 规则一致，不是“负回报被错误翻正”的证据。16 次更新的 summary 仿射检查最大误差 3.54e−7，无非有限指标；保存优势总体均值接近 0，population std 约 0.996086，与 128 样本的 sample-std 标准化相符。

只有 5 次自然 P01 入口、4 个已完成 episode；四次都产生 FR crossing 和 placement，但没有完整越障成功，也没有 P02 terminal。不能将 516 简单称为“没有 P02 学习”，也不能由计数认定已覆盖确定性策略的失稳路径。

## 真实终止与阶段交接

| Episode | 阶段 / 原因 | 真实末区间 ticks | Stored reward = return | Raw GAE |
|---|---|---:|---:|---:|
| 0 | P09 / FALL | 8 | −42.254707 | −10.928190 |
| 1 | P09 / BODY_COLLISION | 7 | −42.664707 | −10.804804 |
| 2 | P05 / INCOMPLETE_CONTROLLER_BLOCKED（recovery exhausted） | 2 | −41.742973 | −17.071898 |
| 3 | P05 / INCOMPLETE_CONTROLLER_BLOCKED（task deadline） | 8 | −41.224113 | −18.594786 |

四个 terminal 都是 done=true、timeouts=false、bootstrap=false；terminal event 均 −40、next potential 均 0。return 与 float32 stored reward 相等，raw GAE 与 float32(reward−old value) 相等，后继 value 和后继 GAE 的系数均为 0。这里确有 7/2 tick 短末区间，不把它们写成 8 tick。重置后的新 episode 不参与终止步的回报。

25 个普通阶段切换全部 done=false、bootstrap=true：P01→P02 为 5 次，P02→P03/P03→P04/P04→P05 各 4 次，P05→P06/P06→P07/P07→P08/P08→P09 各 2 次。16 个 rollout 的尾步都非终止；尾部 critic 值没有单独记录，未额外 forward 或猜测其数值。

## 当前含义与最小下一步

根代理报告已封存 C170240 仍在 P02 因 RR hard limit 终止；本报告没有读取 C 源文件，精确 tick 由其正式提取结果给出。这说明随机 FR 放置尚未转化为确定性 mean 轨迹恢复，不说明所有探索都无效，更不能据 GAE 的符号推断某个 wheel 的因果。

优先复用现有 P02 的 sample、old_mean/std 和 C 实际动作/支撑轨迹，核对成功采样是否持续偏离条件均值。不同闭环轨迹不能伪装成同状态对照；未执行的 mean 动作也没有可声称的实测 return。若再续训，关注更多自然 P01 独立入口、完整轨迹以及失败区域的实际覆盖，而非只增加决策总数；保留所有 12 通道、物理参数、普通阶段 bootstrap 和完整任务回报。不能把 deterministic 动作塞入 Gaussian storage 并沿用采样 log-prob，也不能在 FR capture 时人为 done。

本审计没有发现必须先修的信用链错误，也没有支持永久 mask、改 gain/physics 或再改 functional-carry 折扣的证据。尚无唯一根因或保证成功的修复。

证据：[完整 JSON](<C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_2048_reward_credit_summary.json>)；源日志：[advantage_audit](<C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/train/20260915T0246086426747Z_ga9c52261ba87_56a6fc5cd8bd42bd9d91b7a75f5ce321/advantage_audit.jsonl>)、[completed_episodes](<C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/train/20260915T0246086426747Z_ga9c52261ba87_56a6fc5cd8bd42bd9d91b7a75f5ce321/completed_episodes.jsonl>)、[manifest](<C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/train/20260915T0246086426747Z_ga9c52261ba87_56a6fc5cd8bd42bd9d91b7a75f5ce321/run_manifest.json>)。


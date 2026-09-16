# RL−3 零残差 B：P13 完成观察被旧恢复脉冲打断

结论：证据支持修复 P13 nominal 的终态所有权，不支持改验收或把原结果改成成功。B 已真实完成四腿放置，并在短暂停轮时进入当前物理 controlled 条件；旧 P13 恢复姿态＋轮脉冲随后继续执行，使最后 1 s 观察不通过。

只读已封存 source 的 tick8600–8857：258 条物理/native、34 条决策和34条height记录，另读终态 manifest；未扫全程重新聚合、未改生产/物理/原结果、未启动仿真。全部258tick的 raw、projected和same-prestate native policy effect 均为0，mask仍全12为1。

## 精确时序

P13 进入 tick6728；RR crossing/placement 在6155，RL placement在6727。source 本地事件 16.733333/16.8/17.8 s 分别对应 source tick2008/2016/2136；按实际 dispatch→post-observation坐标，应为 native tick8737/8745/8865。

| 实际 tick | N / 真实状态 | 原验收含义 |
|---|---|---|
| 8736 | 四轮 N=+.3；实测最大轮速 .31745 rad/s | region/support已真，但尚未controlled |
| 8737 | 四轮 N=0，servo N不变；最大轮速 .09106，body v .02788，ω .04883 | 开始固定1s观察 |
| 8744 | N仍停轮；最大轮速 .08420，body v .02336，ω .05430 | 当前controlled；观察已过7/120s |
| 8745 | 旧source原子切换恢复servo N＋wheel [1.09,−.72,−.43,−.39] | 真实轮速立即 [1.09012,−.71672,−.52935,−.49358]，失去controlled |
| 8752 | 脉冲继续；body v .25825 m/s | 运动状态明显改变 |
| 8857 | 脉冲仍在；maxwheel1.00725，body v .01826，ω .01074 | 满1s，region/support仍真，但controlled=false，失败 |
| 8865（未执行） | 原参考最终停轮时刻 | 比观察终点晚8tick，不能拿预测当已完成 |

15个实际保存的观察 endpoint 都由 tick−elapsed×120 反推出起点8737；120Hz物理速度谓词也只在8737–8744这8tick连续为真。任务字段只在实际已保存 endpoint 使用，没有15Hz前向填充。

失败原文是 “not currently controlled at fixed post-completion observation end”；termination_source=POST_COMPLETION_LOSS，而 post_completion_loss_observed=false。因此本次不是区域丢失：终态仍有3个真实TOP支持，轮速没有停止。最后1tick（8856→8857）的决策未返回，未补造reward。

8737前的servo N为 [−18.5,−31.4,3.7,31.1,−6.9,−18.7,−6.9,−27.2]。8745变为 [.5,−.7,3.7,.4,−2.6,−3.9,−.5,−6]，由现有mapper按1.25°/tick执行，不是瞬移。参考home-like动作与轮脉冲同时发生，尚未隔离二者各自贡献；但脉冲与实测超速的直接时序已明确。

## 最小候选与边界

在同tick当前 P13 valid/VERIFIED、allplaced、region/support/controlled均真且观察已开始时，让 final-stop nominal owner 接管：捕获当时的 current nominal servo8（不是home、actual q、旧入口），四轮 N=0。应在过期恢复组生效前建立所有权，防止旧group和 endpoint home override再次覆盖；保留mapper/ACK/HISTORY、正常残差和全部12通道。

有效接管后不要因下一tick transient uncontrolled重新放回旧脉冲。这个 latch 只保存 nominal 所有权，不锁定success：原evaluator仍检查完整1s、区域丢失、安全和终点速度；不能清loss标记、重启计时或放宽阈值。reset清除owner。

正例就是8744：FL当时AIR、不承载，但FR/RL/RR三个真实TOP支持有效，所以不能额外要求四腿全部承载或回到home角。反例包括8736（仍超速）、非P13、缺placement、未知/过期证据、区域/真实支持不足，以及只有旧post_started历史而当前已失控；这些都不能新触发接管。已接管后若状态失败，仍应由原规则失败，不能把owner当成功证书。

生产位置：[当前完成判定](<C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:837>)、[nominal派发](<C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:2060>)、[endpoint home覆盖](<C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:2126>)。任务配置明确physical_stable_pose，home仅建议/严格质量诊断。

本候选尚未实施或证明成功，仍需定向正反例和真实零残差重跑。[JSON证据](<C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/p13_RL3_completion_interruption.json>)。


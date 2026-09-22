# P05 有界停滞终止的训练信用核对（只读）

结论：**未发现明确的 done/Phi/GAE 接线错误，也没有致命概率或非法执行问题证据；不构成继续训练的门禁。** 本次只读生产代码与封存 det 日志，未制造 PPO 样本、更新网络或启动 Isaac。下列 4 个生产模块及 2 个选定 YAML 的当前 SHA 与 det 运行 manifest 绑定值全部一致。

## 这次评估到底结束于什么

CP187904 det run `91d8d06aaced49ebb0ad9d5a5c067b31`，decision763，tick6096→6099，最后一步实际只有 **3/120=.025 s**。物理 evaluator 仍 valid，未触发 collision 等物理失败；语义 supervisor 在 P05 发出 `INCOMPLETE_CONTROLLER_BLOCKED`，source=`LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。这是**任务未完成**，不是碰撞、录像错误或外部 wall-clock 截断。

P05 原限30 s，当前 progress=.85，许可延长=min(10,30×.5)×.85²=7.225 s；阶段年龄达到37.225 s，episode时间50.825 s。延长取当前进度，不是不断重置的隐藏计时器；`stall_diagnostic=true` 本身仅作诊断，真正终止条件是这个有界期限。[实际结局](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_conditioned_hip_wheel_v1/CP187904_deterministic_terminal_appendix.json)、[supervisor](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:1380)

## 同类真实训练样本会走的通路

| 层 | 当前实现 |
| --- | --- |
| Core终态 | `_terminal_reason` 接受 semantic task reason，即使物理 evaluator reason 为 null；物理循环立即停，`done=true`；返回 `terminated=true,truncated=false`。普通阶段切换没有 reason，因此不结束 episode。[env](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_env.py:47)、[返回](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_env.py:234) |
| RSL封装 | 拒绝任何 external truncation；`done=step.terminated`，`extras.time_outs=false`；真实末态先保存，可在完整rollout尾延后reset。下一episode不能把末态替换成“未结束”。[wrapper](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:237) |
| 持久化及入buffer | 实际采样 raw action、旧 mean/std/logp/value、reward、terminal先绑定并落盘，之后调用官方 `process_env_step`；校验存储 raw/distribution 未变。前缀不算学习。[collector](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:1773)、[入buffer](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:1834) |
| Bootstrap | 官方 timeout reward补偿乘 `time_outs=0`，故为0。GAE同时以 `1-done=0` 消除 next-V 和下一episode advantage；无跨reset泄漏，即使最后计算的critic观测已reset也被mask掉。[官方PPO](C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/rsl_rl/algorithms/ppo.py:175)、[GAE](C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/rsl_rl/algorithms/ppo.py:187) |

## Phi 与成本：终态惩罚不是漏掉了

当前 reward：`5*(.9985*Phi_after-Phi_before) + event - .02*实际dt + quality`；**任何有 reason 的终态，reward 的 Phi_after 强制为0**，success event=+40，其余终态 event=−40。因此停滞未完成的标签与物理失败不同，但此版确实使用同一−40终态成本；这是明确配置，不是 reason 丢失。[reward](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_reward.py:436)、[配置](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/configs/ppo_task_conditioned_hip_wheel_v1/reward_config.yaml:42)

本次 det 日志末拍已实际计算：

- Phi_before=.41119746455768813；shaping=**−2.0559873227884404**；
- terminal event=**−40**；time cost=**−.0005**（按3个实际物理步，不按完整8步）；
- 加权质量成本=0；合计 **−42.05648732278844**。

末态 task snapshot 仍记录测量型 global potential=.40865785317518555，不是reward忘记归零：reward明确用0、done明确禁止bootstrap，观测值不等于终态回报势函数。缺测几何会标null并保留任务终止；此拍几何实际记录为0，不是缺测伪零。[原始末行](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T0627142354953Z_gee5a9651591d_91d8d06aaced49ebb0ad9d5a5c067b31/source/video_policy_decisions.jsonl)

## 能说明什么，不能说明什么

真实采到该终态时，未标准化的末拍 `delta=r−V_old`，末拍 return=r；相同rollout内此前动作经γλ回传。rollout=128、γ=.9985、λ=.99，旧rollout不会被事后重新写GAE，跨更早批次依靠critic学习；这是有限采样/信用传播范围，不是跨阶段切断。全rollout advantage还会标准化，**负reward不保证每条最终标准化advantage均为负**；没有实际采样的旧value和整批数据，不能计算或声称本次eval已经产生了GAE/梯度。[官方标准化](C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/rsl_rl/algorithms/ppo.py:205)

非终止停滞期间没有额外固定pose或stall bonus/penalty，仍有时间成本及实际global-Phi变化；P05本批几何成本可为0。不能由此断言策略已经学会避免停滞，也不据此提出reward扫参。

这次录像路径调用同一 `core.step`，但 manifest明示 **optimizer_updates=0**，上述reward只是评估结果。[评估调用](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_video.py:661)、[评估记录](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_video.py:766) 下一轮自然P01训练如真实再出现该结局，可核对同拍 `dones=1/time_outs=0/Phi_after=0/event=-40` 与实际rollout/advantage；若没出现就准确记为未采到，不能移植这条冻结录像充当失败学习样本。

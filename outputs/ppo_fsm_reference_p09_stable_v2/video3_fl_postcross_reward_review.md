# Video3：FL 越沿后奖励与放置约束核查

结论：**未发现本窗口已确认的刷 lift 奖励、加载惩罚回退或关闭 FL 下降通道的实现缺陷；支持完成当前后腿课程后继续自然 P01 学习。** 这只支持继续采样，不保证下一块就能放置或完整成功；不调整 reward、超参、验收标准或控制器。

## 实际奖励，而非仅看符号

对象为已完成145920/seed4001/video3 的 [policy ledger](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260910T1138465345417Z_g7db0d17f398d_9bd793c700bd4040b4c517a0cd625ebb/source/video_policy_decisions.jsonl)。首次758行解析的逐行回传被工具截断，未用不完整输出计算总和；获主控明确许可后，只做了一次累计标量补充扫描，以下来自该完整摘要。未读活动训练。

表中窗口为决策628–758，共131个“C5024及之后的末态”：其首个奖励覆盖5016→5024，故累计8.7s。各项是**未折扣的实际已加权奖励**，不是重新计算的优化 return。

| 项 | 131决策合计 | 单决策范围 |
|---|---:|---:|
| task_progress | −42.557903412 | [−42.036937772, +0.045749225] |
| body_stability | −0.000874396 | [−0.000067626, −0.000001183] |
| contact_motion_quality | −8.063e−10 | [−8.063e−10, 0] |
| control_smoothness | −0.027218198 | [−0.000213251, −0.000104169] |
| control_regularization | 0 | [0, 0] |
| 总奖励 | **−42.585996006** | [−42.037057091, +0.045538943] |

非终止130决策合计 **−0.548938915**，只有2次总奖励为正；其中C当步为+0.045538943。严格从C后下一步算，130决策/8.633333s含终态合计−42.631534949，仅1次微小正值（最大+0.000958753）。不能将这些局部正增量一概解释成 FL 抬升奖金：全局势能还含其他腿的当前几何与准备状态。

[奖励实现](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_reward.py:196) 为 `5*(0.9985*Phi_after-Phi_before)`，另减实际时间/质量成本；没有独立 lift、阶段标签或非零动作奖金。131行公式和五项求和的最大绝对误差均0。C当步Phi为0.397468472→0.407496228；最后非终止Phi为0.407254221。终止时物理快照Phi仍0.409896095，但**奖励明确用Phi_after=0**：shaping−2.036271105、事件−40、4-tick时间成本−0.000666667，总计−42.037057091，time_outs=false、bootstrap=false。两种Phi的差异是任务终止定义，不是漏奖励。

## 进度、下降与动作能力

[现有物理诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video3_fl_placement_diagnosis.md) 和此次摘要一致：131末态AIR、TOP/placement均0，台面净空29.539–33.017mm；P05 phase_progress恒0.7，期限30+10×0.7²=34.9s。该阶段进度不是全局奖励势能；[placed谓词](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:1017) 的0.7表示已有lift/crossing但还没有capture，不是成功或只能保持腾空。

[全局势能](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:1067) 在Q+C后保留已完成的unload份额，真实接触加载不再扣掉它；后续lift份额不会因继续抬高而增加。[capture软项](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:1173) 为 `0.5*XY*0.025/(0.025+abs(clearance))+0.5*TOP_fraction`，使接近表面增加势能，AIR最多拿一半，其余需要真实TOP。固定非负Phi的悬停项为负，回到同一增广物理/历史状态的折扣闭环不能重复累积该shaping；现有 [hover/cycle反例测试](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/tests/unit/test_semantic_capture_approach.py:308) 也覆盖此意图，本轮未启动测试进程。实际动作改变后的整体奖励仍受全身运动与接触成本影响，不能把静态几何偏导当作成功控制证明。

全部131末态FL hip/knee mask为1、headroom clip为0；[P05范围](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/configs/ppo_fsm_reference_p09_stable_v2/execution_profile.yaml:38) 为±18°/±24°，现场余量包含整个范围。实际residual为hip7.468–7.738°、knee20.532–20.648°，没有触顶；两方向仍开放，**不等于已证明某个关节方向能安全产生笛卡尔下降**。nominal仍[22.8,−13.4]°；既有诊断已证实目标下发与跟踪，策略没有形成真实落地。

测得transfer_fraction恒1，当前角色窗口减轻必要运动成本；该项不直接奖励AIR，TOP形成后会逐步恢复成本。真实下降可有平滑/落地代价，这是待权衡的训练信号，不是本窗口证实的奖励反向错误。结合已经发生的Q→C、开放执行链和真实未放置终止，下一自然P01训练块能再次覆盖前驱准备及FL收尾，并让后续结果影响准备动作；**无需先改权重或设成功门禁**。保持真实完整评估、当前策略未学会与软件缺陷三者分开。

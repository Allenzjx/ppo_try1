# 本轮交付：zero 成功，Residual PPO 尚未完整成功

生产提交 `6c2121b68654eaaa2afa7c19e9d02ae770493d2f`，原 FSM、成功 zero 和历史数据保留。所有运行已自然封存，无活动 Isaac 实例或计划中的后台训练。最新 CP 是训练候选，不能标为成功模型。

## 实际结果与视频

| 运行 | 结果 | 时间 / 第一未完成任务 |
|---|---|---|
| 改进 zero，N+0，无 learned checkpoint | 完整 SUCCESS，四腿越沿/放置并受控停车 | 73.808333s；保留原成功基线另存 |
| CP178432，重新加载后自然 P01、全部12残差通道 | TASK INCOMPLETE；无碰撞、数值异常或硬限违规 | 49.225s；P05 FL越沿后未放置，末净空3.470128mm、AIR、承载0N |
| 512-decision 后腿训练课程 | 非终态样本，不是成功episode | 到P12；RR曾越沿/放置，后退回ground，RL未完成 |

视频均为真实 Isaac、正常速度15fps、<=200s，无插帧/跨run拼接。对比采用相同机位、场景和seed；zero与新RR分支的RR资格/nominal接续规则不同，不能称完全相同控制器或用缺失后半程的PPO轨迹排名稳定性。PPO短侧结束后明确标注冻结，不是继续仿真。全机轮廓在所检关键帧内，远侧轮接触仍可能被机身自身遮挡。

- [zero 完整成功视频](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/video_review_v1/zero_refine_6c2121b_run042420/zero_Nplus0_full_review_clean.mp4)
- [正式 PPO 完整失败视频及四轮实测速度](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/video_review_v1/cp178432_run051034/PPO_full_actual_wheels_RR.mp4)
- [P05 FL 失败段：净空、接触、承载、关节和四轮速度](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/video_review_v1/cp178432_run051034/PPO_P05_FL_capture_failure_actual_30s.mp4)
- [zero vs PPO 同机位对比](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/video_review_v1/cp178432_run051034/zero_vs_PPO_same_camera.mp4)

## 实际训练与 checkpoint

从原最新CP177792恢复，新增 **640 policy decisions / 5 PPO updates / 100 optimizer steps**，累计 **178432 / 1359 / 27180**。先自然P01训练128，再以真实P01→P06前缀采集512；335个前缀决策排除训练信用。739次正式视频评估决策也不计训练。

新增样本：P01=2，P02=126，P03/P04/P05=0，P06=166，P07=1，P08=1，P09=214，P10=1，P11=19，P12=110，P13=0。P05新增样本为零是本轮采样缺口，不意味着祖先网络从未训练过它。

[最新 checkpoint](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_residual_rr_fix_v1/checkpoints/history/checkpoint_step_000178432.pt)

SHA256 `19acb91f64a2b728e54f02d50dcaf187e11fd432ff68901ba91ff0af58f6a665`。同372维/full12/quarter温度.25/rho.9，有效Adam LR1e-5，Identity normalizer；保留兼容actor/critic/Adam/RNG/counters，旧未完成rollout清空后真实重采样。保存/官方重新加载已验证。reward epsilon仍0，没有为画面稳定新增奖励、改HISTORY或关闭wheel residual。

## 已实施及尚未解决

- zero：终态源home目标改为0.5s平滑过渡，保留1s观察和停车阈值；新机位不改变传感器/执行器/物理。实际成功，但最终RR轮速距阈值仅0.010504rad/s，不能宣称所有稳定性指标更好。
- RR：新自由AIR增高资格，接触上升不生成新资格，ground撤销；窄范围待执行源事件依据真实当前状态接续。保留普通阶段的连续动作、全12残差与历史，不要求RR自身先动作或虚构FL承载。
- 实际后腿数据记录撤销/重获资格和短暂越沿放置；随后失守。七个新源许可事件当时均合法放行，因此不能把后缀越沿归因于等待保护生效，也不能称RR完整修复成功。
- 正式P01首先卡在FL放置；源下降已执行并到端点，但全身状态仍未接触。后段knee跟踪误差均值.00254°，hip反馈周期摆动的误差均值.676°/最大1.362°，不能说每拍完美跟踪；也未发现持续下探命令被明显驱动失灵阻塞。FL自身有效残差较小，不能单凭间隙归因其大幅抬住。没有证据把问题改称mask/漏写/传感器或录像失败。原P05局部恢复时间37.225s耗尽，剩余总任务时间不用于绕开阶段规则。
- 最新P02四轮都有指令和实测转动，未发现前三轮被mask或抵消。FR悬空时旋转不等于提供地面牵引。

新严格完整P01评估为0/1成功，未创建成功checkpoint指针；尚未证明稳定性优于zero。下一优先是P05真实接触放置能力，再验证RR连续carry和放置后的保持，次级前腿稳定性优化继续延期。

## 报告与机器可读证据

- [生产修改清单与测试范围](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/residual_rr_fix_v1/changes_summary.md)
- [RR诊断（含原问题、修复与真实后腿样本）](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/diagnostics_v1/residual_RR_diagnosis.md)
- [最新完整P01的FL失败执行链诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/diagnostics_v1/formal_P01_CP178432_failure.md)
- [后腿训练时序诊断](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/diagnostics_v1/training_RR_carry_178432.md)
- [实际训练与评估记账](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/residual_rr_fix_v1/RECOVERY.md)
- [版本迁移和续训方案](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/residual_rr_fix_v1/training_resume_plan.md)
- [正式评估结果与质量数据 JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/residual_rr_fix_v1/evaluation_results.json)
- [P02四轮逐通道表](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/video_review_v1/cp178432_run051034_p02/p02_wheel_chain.md)

本失败轨迹的roll/pitch RMS为6.954382°/5.749541°，角速度P95为.018862/.017192rad/s；缺少P06–P13且长期驻留P05，所以固定全程质量分数为null，不与成功zero作优越性排序。

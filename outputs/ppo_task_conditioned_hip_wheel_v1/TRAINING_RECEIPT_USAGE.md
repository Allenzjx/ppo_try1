# 新版本已封存训练块 receipt

`training_receipt.py`复用旧输出目录中的只读摘要逻辑，增加新experiment/policy、P01–P13零补齐、prefix单列、实际geometry质量系数/贡献、checkpoint新branch计数、terminal和首未完成阶段。旧工具和生产不修改。

在项目根目录运行；把`ACTUAL_SEALED_RUN_ID`替换为**实际已自然结束或已完成update边界停止**的正式新版本训练run。它不是现在已有run的声明，也不要求等待未来run。

```powershell
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' 'outputs/ppo_task_conditioned_hip_wheel_v1/training_receipt.py' --run 'runs/ppo_task_conditioned_hip_wheel_v1/train/ACTUAL_SEALED_RUN_ID' --output 'outputs/ppo_task_conditioned_hip_wheel_v1/ACTUAL_SEALED_RUN_ID_receipt.json'
```

输出采用exclusive-create，不覆盖旧receipt。脚本不导入torch或仿真；不等待、不修改run、不续训，也不更改checkpoint。只接受新版本正式N1/128-step rollout、`SUCCEEDED`或`STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`封存状态；拒绝活跃、不完整、旧experiment、混合policy或不一致记录。

重点字段：

- `new_decisions / new_ppo_updates / new_optimizer_steps`：本run实际完成量，与update ledger、封存manifest和末CP ancestry核对。`final_checkpoint.branch_counts`是新branch累计量，`lifetime_counters`是历史总量，不能相加。
- `actual_request_phase_coverage`：只按learner实际发出动作时的phase_id计数，P01–P13全部列出；请求P08课程不保证P08实际样本，也不算prefix在前驱阶段的运动。
- `prefix`：单列行为decisions/ticks、attempt结果和原provenance，learner_credit固定0；两种实际prefix日志kind均支持，若有prefix_evidence则交叉核对。prefix未持久化为独立日志时，该项为null，不伪称读过。
- `actual_front_quality_by_task_substate`：沿用实际每物理步front beta范围、dt、tilt/rate和weighted cost。
- `actual_geometry_quality_by_phase`：实际每步geometry beta范围、已知cost与缺失terminal测量数。unknown测量仍为null/缺失计数，不能当测得的零。
- `actual_quality_contributions`：front+geometry实际cost与负body_stability family算术交叉核对。`recorded_quality_epsilon_counts`只报告日志原值；不把配置值冒充实际采样系数。
- `terminal_outcomes / first_terminal_outcome / last_sample_outcome`：termination原因为原字段；首未完成阶段按日志`completed_stage_ids`排序得出。预算边界单独标为任务未完成状态，不当任务失败或成功。
- `natural_P01_full_task_success_count / suffix_success_count`：按正式episode ledger分开；训练执行成功不是越障成功。
- `final_checkpoint`：仅最后保存CP做一次字节hash与既有roundtrip manifest核对，列出本run checkpoint路径；不重新加载模型、不重复审计历史CP。

测试：`candidate/test_training_receipt_metadata.py`，**10个metadata合成正反例通过**。测试包含自然/前缀计数隔离、零覆盖阶段、geometry算术、安全终止时的缺失测量、未封存run、错误新branch/旧experiment、缺失likelihood证据和prefix credit泄漏；这些是临时目录内假数据，不是模型、真实训练或物理成功证据。

当前没有为尚未产生的正式run生成receipt。

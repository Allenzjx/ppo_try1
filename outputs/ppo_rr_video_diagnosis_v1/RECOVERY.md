# Recovery — video-first diagnosis

状态截至2026-09-14 05:45 UTC：两块训练均已完成并保存，自然P01重载视频评估也已封存，当前无本项目Isaac训练/评估进程。评估为P02/7.325秒/RR knee HARD_JOINT_LIMIT，非成功；视频已正常导出并展示。
评估run：`runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260914T0540369315910Z_g4b2c038887c4_85756e634ebf4aa2999aa1273ba6077b`。
后续启动前仍应重新检查此run最终manifest、最新pointer与项目Isaac进程，不能并行启动另一个实例。

## 已保存的真实状态

- 项目：`C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1`
- HEAD：`4b2c038887c4109c639eea720f9e60de2c6d8d93`，生产src/config/scripts未改。
- Latest：`outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000167424.pt`
- 累计167424 policy decisions /1273 PPO updates /25460 optimizer steps；保存/重载round-trip=true。
- checkpoint SHA256：`9eb4bbdb2cb93afcc4fc04ab1295b92bde06167a72af9febe09804e9e5603a44`。
- 本轮实际新增640 /5 /100。896个教师决策、所有视频/干预数据均不计入PPO。
- Block1自然P01：512/4/80；block2物理P06前驱：128/1/20。后者消费原生停止请求，状态STOPPED_AT_VERIFIED_UPDATE_BOUNDARY；原请求剩余384未运行，不预扣为已完成。
- 阶段样本：P01=2,P02=101,P03=8,P04=1,P05=400,P06=115,P07=1,P08=1,P09=11,P10–P13=0。
- 后腿第一回合FALL，第二回合在P06非终止更新边界结束。无后缀成功、无全程成功，不新增或覆盖best。

Block1：`runs/ppo_fsm_reference_p09_stable_v2/train/20260914T0508225745866Z_g4b2c038887c4_bb20474c8a244881b0bd2a2e769e056d`。
Block2：`runs/ppo_fsm_reference_p09_stable_v2/train/20260914T0518470549569Z_g4b2c038887c4_98748e39f68e441a8ff2ab2fe47d79da`。
原生run/checkpoint manifest为最终计数权威；首回合receipt仅是当时未更新的32行快照，此后已包含在128行更新中。

## 恢复方式

在上述项目目录的PowerShell运行。先查看最新pointer和manifest；若未来已有更新有效模型，使用实际最新，不退回这里的数字。

```powershell
$continuationPointer = Get-Content -Raw -LiteralPath .\outputs\ppo_fsm_reference_p09_stable_v2\checkpoints\checkpoint_last_pointer.json | ConvertFrom-Json
Get-Content -Raw -LiteralPath $continuationPointer.manifest
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 4b2c038887c4109c639eea720f9e60de2c6d8d93 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P06 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Decisions 128 -Seed 1001 -NumEnvs 1 -Device cuda:0 -Checkpoint $continuationPointer.checkpoint -CheckpointIntervalUpdates 1
```

这是复用本轮P06课程的恢复命令，不是承诺该课程高效或成功。每次reset的真实P01前缀约29.87s，不能把其448个教师决策算入学习。既有P05较早交接是可研究的下一个课程选项，但本轮没有运行或宣称其收益。不要在未完成rollout内部更改课程。

自然P01独立确定性视频复评命令（无教师，最多200s，不是训练）：

```powershell
$evaluationPointer = Get-Content -Raw -LiteralPath .\outputs\ppo_fsm_reference_p09_stable_v2\checkpoints\checkpoint_last_pointer.json | ConvertFrom-Json
& .\scripts\run_semantic_video.ps1 -Command eval -ExpectedHead 4b2c038887c4109c639eea720f9e60de2c6d8d93 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage full_episode -Mode semantic_residual_eval -MaxDecisions 3000 -Seed 4001 -Device cuda:0 -Checkpoint $evaluationPointer.checkpoint
```

复评非成功时原脚本可能返回非零exit，仍应保留原始物理结果，并用本轮诊断导出器交付视频；不得把task_success改true。

## 保留与兼容性

原166784 checkpoint、历史164736视频、所有B0/C0/D原run及before视频保持不变。只有从旧d4来源166784到当前4b首次恢复使用`checkpoint166784_provenance_only_migration.json`：没有生产文件差异，保留actor/critic/Adam/identity normalizer/RNG/预算，丢弃旧未完成rollout。当前4b checkpoint直接恢复，不能再次套用旧迁移。372维观测、Full12、HISTORY rho0.9、reward/action/物理保持本轮冻结版本。

保留dirty工作树：正常训练已更新checkpoint_last、pointer、resume_state并新增immutable历史文件，本轮输出目录新增诊断/媒体工具。不要git clean/reset、删history或覆盖原FSM。没有后台自动续训、提交、推送或新建工程。完整交付与结论见RESULTS.md；物理结果与媒体有效性见video_manifest.json。

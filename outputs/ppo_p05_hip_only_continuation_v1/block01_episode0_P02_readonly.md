# 新执行版本首个训练 episode：P02 只读核对

运行：`20260922T0415552437349Z_g0001c3138b0b_7b34b44b4b9d49ddb087585dec7acef6`，HEAD `0001c3138b0b38a7278edc288b8ddc7444815770`。范围仅 completed episode 0 与对应前 **236** 条 policy 审计记录；没有修改活动训练。

## 精确终止原因

这是 **P02 真实未完成后的局部任务期限终态**，未发现本轮补全器或 pending 调度造成的新入口／执行错误。

- 终点 tick **1888**，episode **15.733333 s**；P02 从 tick 16 开始，stage_age **15.6 s**。
- `termination_source=LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。原限制 15 s，加当前进度宽限 **0.5945546494 s**；15.6 ≥ 15.5945546494。
- 进度项：`lifted_FR=0`、`clear_FR=0`、`approach_FR=0.844668917`。宽限确为 `7.5 × (0.844668917/3)^2`。
- P02 `entry_valid=true`、无 entry reasons。独立物理 evaluator `valid=true`、`termination_reason=null`：不是机身碰撞、数值中止或基础设施错误，也不是“成功”。
- FR 当前真实 ground=true，台面 gap **−50.001815 mm**，前缘距离 **−48.832771 mm**，无 crossing、placement。obstacle pair 同时存在、surface 为 `OBSTACLE_AMBIGUOUS`，没有被写成 TOP 或承载已验证。
- 真实 lift qualification 曾于 **21** 发生、**109** 接地撤销；又于 **1462** 发生、**1646** 接地撤销。这些不是被遗漏的历史事件；后一次仍低于台面，未完成必要净空与前送。

## 不是源动作从未下发，也不是物理冻结

P02 源仅更新 FR knee，有限 source end 为 45.9°；源 P01 四轮 +0.3 rad/s 从其 source time 0.333333 s 保持至 13.266667 s 的明确四轮 stop。

本 episode 的完整 policy 决策末端记录中：四轮 N 从 tick **48** 起均 +0.3；至 tick **1600** 的首个 stop 后决策记录均 0。它们不是 P01→P02 时丢失，而是较早 P01 finite source 的真实结束。后续 `_approach_assist_required` 要求当前 FR qualification、至少原规定的其他支撑与台面上方 15 mm 净空；此 episode 没有维持这些物理条件，所以不追加前送。policy wheel residual 仍有效，不是 wheel residual/nominal 共用 mask 清零。

236/236 决策 residual permission mask 全 1、capture-assist owner 全 false；1888/1888 physical tick native audit 通过。FR 实际关节在动：例如同一 native 坐标 knee position 从 tick 1880 的 **0.795643091 rad** 变化到 1888 的 **0.821117759 rad**，最终 target 分别 45.424080° 与 50.609970°。`recent_joint_motion_deg=0` 是 evaluator 在尚未越沿且当前接地时清空本次抬升证据窗口的结果，不表示 PhysX 或 actuator 没有步进。

结论限于本 episode：符合已有的随机前段净空失败类型，没有发现需要热修或阻止当前 2048 自然 P01 更新的执行链缺陷；不能以此一例证明所有策略／控制语义均完美。

## 当前已有的合法 P04/P05 课程选项（仅建议，未启动）

当前 CLI 和 prefix adapter 已支持本版本 389 维 actor，以及自然 P01 的冻结 checkpoint-policy roll-in。当前块自然结束并封存后，可从**实际最新兼容、完整 update checkpoint** 使用：

```powershell
& 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\scripts\run_semantic_ppo.ps1' `
  -Command train -Stage phase_suffix -SemanticVersion v3 `
  -ExperimentId p05_hip_only_continuation_v1 `
  -ExpectedHead 0001c3138b0b38a7278edc288b8ddc7444815770 `
  -NumEnvs 1 -Device cuda:0 -FromPhase P04 `
  -TeacherOffsetDecisions 0 -PrefixSource checkpoint_policy `
  -Checkpoint $latestCompatibleCheckpoint -Decisions 512
```

`$latestCompatibleCheckpoint` 必须先由最新 pointer/manifest 实际解析，不能用旧 CP 数字覆盖。本建议未固定 seed，也未实际执行。可将 `-FromPhase P04` 改为 `P05`；**P04 offset 0 优先保留前驱准备给 learner**，避免短 P04 阶段因非零 offset 已结束而错过入口。

合法性质：

- prefix 在同一 `SemanticEpisodeEnv` 真实从 P01 连续执行；源 actor 由这次恢复 checkpoint 独立冻结，新的公共补全器启用规则同版保留，训练 learner 不切换底层控制器。
- reset roll-in 的 `policy_credit=False`，不进入 PPO storage；任务时钟包含 prefix，进入 learner 时不重置 episode、动作历史或物理状态。
- 达到真实目标 phase 后才开启 learner credit；如果前缀物理终止、超出有界前缀预算或错过目标 phase，只做**一次 fresh P01 fallback**，不递归寻找幸运入口。
- prefix 的成败、行为决策和物理步单独记录在 `prefix_evidence.jsonl`；suffix 成功不能命名为完整自然 P01 PPO 成功。
- 不需要也不应再传 `NewMdpWarmStart`、`PolicyDistributionMigration` 或旧迁移 JSON。

依据：当前 run 的 `completed_episodes.jsonl` / `residual_and_projection_audit.jsonl`，以及 `semantic_supervisor.py`、`semantic_cli.py`、`semantic_checkpoint_prefix.py`、`semantic_checkpoint_prefix_policy.py` 与 `scripts/run_semantic_ppo.ps1`。

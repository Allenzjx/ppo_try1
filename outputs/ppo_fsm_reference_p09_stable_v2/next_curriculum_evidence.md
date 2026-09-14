# 下一课程的有界只读核查

## 结论：优先 P10，不用 P11 规避 RR 放置

建议下一轮课程机会为：**当前 P07 块完成后，先 P10-origin 128 决策；随后 P04-origin 在同一次 run 连续训练 512 决策（4 个保持原长度的 128 rollout）**。前者给 P10→P11→P12→P13 的 RL 准备、载荷转移、抬升越沿、全身收尾学习机会；后者让策略参与 FL 的 P04 准备→P05 放置入口，并让未终止的 P05 物理轨迹继续跨越 rollout/update 边界，针对刚观察到的 P05 后退失配。当前 P07-origin 块已经保留 RR 的前驱准备机会。这里不调整 rollout length、gamma/lambda、reward、action cap、网络或 optimizer 超参数。

这些是**接下来的课程机会，不是任务总预算、最多运行次数或停止条件**。之后继续 natural-P01 学习与重新加载后的完整评估，并按真实阶段样本、终止结果和可用资源调整后续课程；不能因上述机会已经各运行一次就结束续训，也不以请求的决策数冒充已优化或已保存计数。

P11-origin 不是替代 P10 可达性的办法：它同样依赖 RR 放置，而且把 P10 的 FR 接收侧准备交给教师，不如 P10 保留完整尾段。P10 的 128 只是一个 rollout 块，不保证本次实际采到四个阶段；按真实 `curriculum_start.actual_phase`、`requested_phase_still_active_at_credit` 和 `phase_id` 分别统计。教师 P10 后缀即使完成，也只能称 `SUFFIX_SUCCESS`，不是 natural-P01 当前策略完整成功。

若 P10 本次前缀因 RR 未放置而 miss，保留 miss 和自动 natural-P01 fallback 的真实训练数据，不把它计成 P10 样本；不尝试以 P11 绕过同一个物理条件，也不重立 A/非零探针成功门禁。后续训练继续依据实际证据安排，不把一次前缀 miss 当作训练任务结束。

## P10/P11 入口的实际条件

`stage_task_spec.yaml:217–233` 与 `semantic_supervisor.py:83,1017,1061,1221`：

- P09 的完成谓词是 `placed_RR`，不是“RR 曾 AIR/曾 Q”。RR 要有真实 Q/C/P 事件，当前不接地、位于平台 XY，且当前为 TOP 或可用 AIR 且台面间隙≥0；仍在地面或仅有过时 P 历史不够。
- P10 入口同时要求 physical_valid、FR/FL 已放置以及上述 RR 当前放置可用；P10 学习 RL 前缘接近和接收侧准备。
- P11 也要求这套放置条件，并且必须先实际完成 P10 准备，或 RL 已存在合格下游运动被连续接管；随后才学习 RL 载荷转移。不能靠把 CLI 改成 P11 跳过 P09。
- P12 处理 RL 越沿放置；P13 要求四腿已放置再完成全身通过与受控停稳。普通阶段改变不会结束 episode 或切断已开始策略部分的 GAE。

`semantic_prefix.py:118–224,319–420` 的 teacher 路径从真实 P01 reset 开始，每个物理 tick 先由当前语义监督器更新任务状态；教师自身同名 FSM 阶段不决定课程入口。到目标**语义**阶段后的 8-tick 边界才开始 receipt 绑定的连续 takeover，不还原关节快照、不强制设置任务阶段。

默认最多 1800 个前缀决策（120 s），其中 takeover 上限 30 决策（2 s），目标 offset=0；前缀时间仍计入 200 s 物理任务时限。教师先结束、真实任务先终止、目标阶段在 offset 前结束、或前缀预算耗尽都会记录 miss，并执行一次 fresh P01 fallback。P10/P11 都可能因 RR 未达到新语义 P 状态而不可达；该情况是课程分布未就绪，不是损坏网络或零 PPO 学习的理由。

还有一个需在报告中保持的限制：`READY` 打开 credit 时保存“目标是否仍 active”，不会强行把已经自然前进的阶段改回去；takeover 的短暂时间可能使实际首个学习阶段晚于 CLI 目标。offset=0 有助于保留短阶段机会，但必须按真实记录确认。READY 之后代码显式保留 source-derived P10 的有效 normal-drive correction，不把教师旧 bias taper 与新 source bias 叠加或无条件清掉。

## P04→P05 的连续学习机会与实际限制

`stage_task_spec.yaml:163–180` 的 P04 入口要求真实 FR 已放置，完成条件为 `edge_proximity_FL` 与 `role_prepared_FL`；P05 承接 FL 越沿并受控放置。P04 不要求固定关节姿态、完全静止或固定支撑组合。`semantic_supervisor.py:1233–1259` 还允许已有合格 FL 下游运动连续进入 P05，不让它落地后重做准备。

已完成的 seed2001 natural-P01 评估仅有 2 个 P04 决策，说明该阶段可能很短；P04-origin 保持 `TeacherOffsetDecisions 0`，仍须核对真正的 credit-start 阶段，不能保证每次请求都采到很多 P04 样本。教师给出的 FR 放置状态也不等于当前策略自然 P01 到达的状态分布，所以后缀训练不能替代 natural-P01 学习与完整评估。

`semantic_env.py:137–236` 只由真实任务/安全终止原因设置 done；P04→P05 沿用动作、观测及 reward 历史。P04/P05 的 Full12 caps 相同，普通交接无范围收缩，允许的 residual 与 mapper 状态继续。`semantic_training.py:1107–1260` 在同一次 run 的完整 128-rollout/update 边界保存和更新，但不会因此 reset 物理环境；只有真实 episode 终止才重新执行合法前缀。

原建议单次 P04 128 只有 8.533 秒策略物理时间，通常看不到 P05 的 30 秒局部截止终局。非终止 rollout 尾部通过 critic bootstrap，**不是 done 或阶段 GAE 截断 bug**；但 rollout 以外的后期放置/失败不会直接进入该块有限长度 GAE。若每 128 后退出并另启 run，新 run 会 fresh prefix，不恢复尚未完成的物理 P05，因此可能反复只采早期片段。

现改为同一次 P04 run 的 512 决策，保留四个原 128 rollout 的连续物理轨迹，让后续 P05 放置或失败进入实际训练数据。若每次决策完整执行 8 tick，策略部分最多 34.133 秒；部分终止 tick、较长 P04、多个真实失败后的重置以及既有有界局部延展都可能改变覆盖范围，**不保证 512 内发生 P05 终局或成功**。同一 run 跨 update 边界仍使用原 critic bootstrap，不声称把四块拼成一个 512 长度 GAE。完成后按真实结果继续训练，不能将非终止预算尾部标作成功或任务失败。

## 已有真实记录支持什么，不支持什么

以下是首次只读核查保存的**带日期历史快照，不是当前进度计数**：本轮 P06 块第一段教师前缀已实际 accepted，448 教师决策、tick3584/29.866667 s、semantic P06，0 策略 credit。P07 run 为 `train/20260910T0647306226372Z_g06716a88bcc9_c6cfcadd350b418a9969f7dfc35406a5`，started manifest 请求 128 决策。**2026-09-10 06:53:31 UTC 当时尾记录仍是教师 P05/tick2544/21.2 s，policy_credit=false**；不能把请求128写成已更新128。当次核查尚未看到本轮 P10/P11 accepted 记录。之后进展以已有 `progress_snapshots/` 和实际完成 manifest/checkpoint 为准，本次文档修订不推算新的动态计数。

冻结 source Trial043 的原始 `leg_crossing_events.jsonl`（4.8 KB，未重跑）记录：RR ACTIVE_LIFT tick6918/57.65 s，FRONT_FACE_CROSSED 7109/59.241667 s，TOP_LOADED 7579/63.158333 s；RL ACTIVE_LIFT 7718/64.316667 s，C 8381/69.841667 s，P 8410/70.083333 s。已保存 source command 审计中 P09=872 tick、P10=26 tick，P10 非零 normal bias=15 tick，首次7794/64.95 s。这说明现有冻结教师确实曾生成 RR 放置和后段运动，P10 是有物理根据的可尝试分布，不是从标签臆造的入口。

但旧事件分类不等同于本轮新监督器的每项 current-valid 判据；Trial043 历史成功也不能替代本轮 accepted/评估。未重放旧 471 MB 原始观测流，不声称证明新 P10 一定可达，也没有改变旧 trial 原始 INCOMPLETE 结果。

## P05 reward：确实总体惩罚这段后退，但不是逐步 body-x 单调罚

核查 seed2001 natural-P01 deterministic eval `validation/20260910T0624345531274Z_g28609010db4e_2d880bd470694bcea9a6028c0902774e` 的固定635条 audit；P05 为451条，其中450非终止。后退以连续决策真实 `body_forward_m` 下降识别，不使用 CoM 代替机身。

| 固定样本集合 | 样本数 | total<0 / >0 | total 合计 | PBRS 合计 |
| --- | --- | --- | --- | --- |
| P05 非终止 | 450 | 416 / 34 | -1.769817 | -1.030729 |
| 其中机身后退 | 383 | 363 / 20 | -1.587650 | -0.961810 |
| nominal wheel 归零后的非终止 | 314 | 297 / 17 | -1.185425 | -0.669308 |

末 INCOMPLETE 决策 reward=-41.152515：terminal event=-40，下一 Phi=0，PBRS=-1.152062，且无 bootstrap。旧观察到的任务未完成没有被标成 SUCCESS，也不会因未碰撞领取成功奖励。

当前公式是 `5*(0.9985*Phi_after-Phi_before) - 0.02*dt + terminal_event`，再加有界的姿态/接触/实际 drive 平滑负代价。`physical_potential()` 与阶段名称无关；P05 尚未四腿放置时，Phi 来自四腿 workspace、transfer、initial/lift、carry、capture，而全身 finish 份额只在四腿 P 后启用。没有独立“body 向后速度为负就罚”的项，因此不能声称每一步后退都严格负回报。

20 个后退但 total>0 的样本不是阶段标签 bonus 的证据。例如 audit191/tick1528，机身仅后退0.514 mm，同时新增 FL 真实 Q，Phi 从0.274113增到0.353875，total=+0.394201。合法全身协调本来可以短时后移；它也不保证后续放置成功。当前长期后退总体受罚、终端未完成强负回报，未发现显式标签/参考模仿奖励冲突；但总体 Phi 可抵消单个方向退步，不能把“没有每步专罚”或一次跟踪不足直接归因为唯一失败原因。

## 可执行现有命令（只列出，未执行）

须等当前 Isaac 退出并在完整 rollout/update 边界保存，再依次执行；launcher 已有单进程、HEAD、运行代码清洁检查。以下读取 `checkpoint_last.pt` 的已验证 immutable pointer，不回退固定141568。条件是前一块的新 checkpoint 已发布为同一 logging-only HEAD `06716a88bcc9ee2fff615bcf2681dd59d97142e5`；若当前块未能发布，仍须采用既有已审查的 runtime-identity migration 恢复，不能将旧 HEAD checkpoint 当成无需迁移的新 checkpoint。

**若下一运行前已应用并提交视频/记录兼容补丁，以下固定 HEAD 命令不能直接照抄。**须改用实际提交的新 HEAD，并从实际最新、已核验的 immutable checkpoint 生成且验证对应 `video_review` 精确迁移计划，第一次加载时传其真实 `-ResumeMigration` 路径；该日志/视频兼容迁移保留 actor/critic/std、Adam、normalizer、RNG 和计数，不使用 `-NewMdpWarmStart`。一旦新 HEAD 已发布兼容 checkpoint，后续同 HEAD 普通恢复不再重复旧迁移。视频是否成功发布本身不是继续 optimizer 的门禁。

```powershell
Set-Location 'C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1'
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead '06716a88bcc9ee2fff615bcf2681dd59d97142e5' -Stage phase_suffix -Decisions 128 -Seed 1001 -NumEnvs 1 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -FromPhase P10 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Device cuda:0 -CheckpointIntervalUpdates 1 -Checkpoint '.\outputs\ppo_fsm_reference_p09_stable_v2\checkpoints\checkpoint_last.pt'
# P10 首次机会结束并核验最新保存后再执行；P04 在同一次 run 连续完成四个原 128 rollout，不是四次重启。
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead '06716a88bcc9ee2fff615bcf2681dd59d97142e5' -Stage phase_suffix -Decisions 512 -Seed 1001 -NumEnvs 1 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -FromPhase P04 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Device cuda:0 -CheckpointIntervalUpdates 1 -Checkpoint '.\outputs\ppo_fsm_reference_p09_stable_v2\checkpoints\checkpoint_last.pt'
```

命令只改变下一课程的合法 reset 分布，保留 actor/critic/std、Adam、identity normalizer、RNG 和 lifetime/spent-budget；不加 NewMdpWarmStart、不重复分布迁移、不改变 reward/action semantics。fresh rollout 重新采集；由现有 CLI 校验新课程 epoch 与生命周期预算。

上述课程之后继续 natural-P01 学习、保存和重新加载后的完整评估，并结合后腿/前驱实际样本继续调整课程。完整 P01 成功、真实权限/资源边界等以用户任务条件和实际证据判断，不由本文件列出的两条命令限定续训总量。

本报告范围：上述带日期训练前缀片段、635条已完成评估 audit、现行 prefix/supervisor/env/training/reward/launcher 静态代码，以及小型历史 crossing/source 摘要。本次输出文档更新没有跑 Python/Torch/Isaac、没有修改生产或 Builder、没有创建新审计文件或 CSV、没有泛化重设计其他阶段。

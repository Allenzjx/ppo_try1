# 已完成自然 P01 C 摘要：已确认 schema 与边界

只读参考：新 B2 `diagnostics/20260909T0246187023798Z_g2f942e824f82_a77db36275dd4bc4be422482922fde92` 与旧 C8 `ppo_transfer_roles_v1/validation/20260908T0953095796695Z_gd7479d9fc41c_708d4bd483f4419382addb1c487cc75e`。二者只用于确认格式，不作为新 C 结果；B2 外部900决策窗口未完整终止，旧 C8 不属新namespace。未重新扫大raw/native流。

- 最终 `run_manifest.json` 必须 `command=eval/lifecycle=SUCCEEDED/completed_at_utc`，且 `result` 与 `evaluation_manifest.json` 相同。这里 SUCCEEDED 仅执行完成，任务可失败或未完成。
- 评估为 `wlr50_clean.semantic_evaluation.v1`。原字段包括 `mode/checkpoint/checkpoint_sha256/deterministic_policy/from_phase/seed/policy_decisions/observed_physics_ticks/duration_s/optimizer_updates_during_evaluation/task_success/controller_task_success/termination_reason/window_ended_before_task_terminal`。
- `telemetry` 已含 `episodes/decisions/physics_ticks/observation_dimension/phase_decisions`；`quality_metrics` 已含 `global/phases/transfer_capture_split/fixed_quality_score`。不需解析带巨型nested单元格的CSV。阶段缺采样可为0个样本，但质量必须UNAVAILABLE，不能0分或完美。
- 末 `residual_and_projection_audit.jsonl` 的 `semantic_task` 含真实stage、completed_stage_ids、completion_values、entry_reasons、termination_source/local_timeout、physical_evaluator、pending_capture。仅末行的 `actuator_target_effect_audit_summary` 覆盖末1–8tick，不是全episode。
- `physical_task_evaluation` 的新共同字段含 `valid/evaluator_version/run_validity/physical_evidence_status`、traversal/task-controlled/strict-recovery及post观察。历史Q/C/P与末态真实接触/承载分别输出，不重放或改判。
- 原始物理只读首/末一行，用于tick0/末态、finite、body/current support、实际初态。普通eval未保存settle次数/level-reference校准明细，脚本将这些parity项标UNAVAILABLE；不从seed相同推出动态初态配对。
- 检查点只核实际文件存在及sidecar的记录路径/SHA与eval绑定、roundtrip声明和runtime/policy一致；**不加载PT、不重复计算大文件hash，不冒充新的独立checkpoint完整性证明**。
- 当前manifest没有全native/全4writes汇总；轻量报告将全程验证标UNAVAILABLE并单列末步实际证据。它不是新的审计门禁。

脚本 `../summarize_completed_full_evaluation.py` 仅接受新namespace `validation` 下自然P01、N1、无prefix、deterministic residual C且真实内部终止的完成运行。拒绝active、prior_B、旧namespace、suffix、无检查点、非零optimizer或外部短窗口；不会创建占位评估或更新主manifest。输出目录必须是新experiment的outputs下一个尚未存在的独占目录。

用法（在真实新C完成后；本次未对新C执行）：

```powershell
& 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' -B `
  'outputs/ppo_all_stage_acceptance_v1/summarize_completed_full_evaluation.py' `
  --run-dir 'runs/ppo_all_stage_acceptance_v1/validation/<真实已完成run>' `
  --output-dir 'outputs/ppo_all_stage_acceptance_v1/reports/<该run>_full_evaluation'
```

脚本只用stdlib，不import生产模块或Torch。输出 `full_evaluation_summary.json/.md`，JSON内 `latest_full_evaluation` 可由主线程审阅后引用；不自行把一个run选择为全局最新，不改训练计数/成功率。质量是实际窗口描述，不是相对FSM优势或唯一因果解释。

准备验证：`--help`、AST语法、模块载入后无Torch、质量缺项/真实零值区分、task failure/safety/incomplete/unverified分类，以及B2/旧C8在读取流之前被namespace规则拒绝，均已用stdlib执行通过；没有创建评估输出。第一次内联检查命令仅因shell引号失败，改为正确here-string后通过，未影响源日志。**尚无符合新协议的完成C被本脚本端到端处理**，首次真实执行仍需主线程核对摘要与final manifest；不把准备检查称真实评估通过。

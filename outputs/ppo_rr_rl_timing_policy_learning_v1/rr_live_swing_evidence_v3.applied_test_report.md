# RL live-swing v3：已应用与定向回归

主代理确认 CP221568 DET 已自然封存、无 Isaac 运行后授权应用。未提交；基底 HEAD 为44219b4fdc4d36d33be489b833c03b897766045b，由主代理统一冻结。

## 实际修改

- semantic_rear_policy_timing.py：新增 LIVE_SWING_MODE 与 RECAPTURE_MODES，v3明确继承v2 recapture。
- semantic_supervisor.py：v3 RL真实本次抬升资格 latch；任意ground撤销并清当前attempt/短窗/initial，不删除历史cross/placed；实际新资格可恢复。暴露current字段与原RL观测槽，同419维；RR retention继承v2。v1/v2不启用新字段/语义。
- tests/unit/test_semantic_rl_live_swing_evidence_v3.py：12个真实TaskEvaluator合成传感器用例；不直接伪造RL current资格字段。
- 经主代理额外明确批准，将旧current-profile测试的配置断言由v2改为LIVE_SWING_MODE。其他v1/v2、安全和物理反例未删改。

## 实际结果

使用base Python：C:\Users\kskzz\miniconda3\python.exe。

定向6文件 **176 passed in 12.25s**：

1. test_semantic_rl_live_swing_evidence_v3.py
2. test_semantic_rear_policy_timing_control.py
3. test_semantic_rr_recapture_current_support_v2.py
4. test_semantic_p05_capture_handoff_v2.py
5. test_semantic_p05_capture_continuation_v1.py
6. test_semantic_p05_completed_source_recovery_v3.py

命令：python -m pytest -o pythonpath=src -o addopts= [上述六文件] -q。
新文件单独 **12 passed in 0.87s**；git diff --check通过。

第一次收集因base环境未安装本项目且无src路径失败，使用pytest显式pythonpath修正，无系统环境改动。随后唯一失败为旧current-profile断言写死v2，已按主代理授权同步版本；全部回归重跑通过。

未运行Isaac/模型/训练，不改reward、actor或物理能力；没有commit。迁移代理已获知CONTROL_REVIEW_COMPLETE可置True。后续从新学CP221568迁移并重新采样，单测不等于实际越障改善。

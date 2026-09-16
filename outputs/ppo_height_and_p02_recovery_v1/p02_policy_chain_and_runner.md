# P01–P02 只读结论与临时诊断入口

旧版正式 C168192（887 ticks / 7.391667s）尚未完成 P02：RR knee target −5.433995°，actual −60.017312°，不是 policy 发出 −60°。FR 有 tick26 抬升历史，但终态净空 −15.815mm、前缘距离 −150.055mm，C/P 未成立。RR 当时仍有实测 ground normal force16.1583N。

同 N 的 B2/C 887 个早期 tick 请求完全一致；12 通道 final target 在 tick1 就不同，四轮实际速度也在 tick1 开始明显不同。RR 连续12 tick 跟踪误差达到1/2/5°，分别起于35/47/152。此窗口 RR mapped N 始终0、nominal geometry 未激活；current-policy 同前态目标效应与跨轨迹误差已分列于 p02_policy_chain_readonly.json。

HISTORY 逆算只用于诊断：111个确定性 RR raw 全负，推算 network mean 同样全负，无“网络已要求反向但 HISTORY 压住”证据。没有加载 head/sigma 张量；本次确定性评估不采样 sigma。当前模型仍明显改变 FL 支撑膝、RL hip 与四轮，不应仅责怪 RR 直接残差或再次修改已生效的 carry reward。

C168192 未保存同次实际 PhysX actuator getters，必须记 unknown。旧 RR_OFF 首动作前实测600/60/2.7Nm/约5rad/s只属于那个诊断。隐式驱动 computed/applied effort 是近似 PD 估计，不是独立 PhysX 电机力矩；接触力已实测，不能说所有力信息未知。

## 新增输出文件；生产代码没有修改

- run_p02_mask_diagnostic.py：复用官方 semantic_video_cli 的 reset、checkpoint/migration loader、物理评价器、录像及 finally 关闭。复用旧 run_rr_channels_off_diagnostic.py 的 live_probe 函数；不执行其 RR-mask main，也不修改旧文件。
- p02_mask_runner_cpu_receipt.json：mask正反例、实际 raw HISTORY/编码特征检查、官方参数原样传递与 wrapper 恢复检查。没有启动 Isaac、加载checkpoint或生成物理证据。
- p02_policy_chain_readonly.json：既有 B2/C168192 的有界数据与完整证据路径。

唯一新增 CLI 参数：

- --diagnostic-mask wheels4：临时置零raw8..11；先运行这一项。
- --diagnostic-mask fl_knee：临时置零raw1；必要时另起自然reset运行。
- --diagnostic-max-decisions 240：最多16s真实仿真或更早真实终止；官方任务 horizon仍200s。边界只是诊断停止，不是P02完成门。

每次记录 policy_proposed_raw_full12、diagnostic_dispatched_raw_full12、previous_actual_dispatched_raw_full12 和实际372维输入的 HISTORY片段。其余通道仍由冻结actor对改变后的真实闭环状态计算，不声称它们等于原C的开环序列。生产mask不变，结束后不留永久mask。任何结果均标 diagnostic_only / formal_P02_evidence=false，不使用 ppo_p02_verified 命名。

## 调用模板（根代理锁定新生产commit及迁移后再运行）

以下PowerShell命令中的大写占位值需替换为根代理已核验的实际值：

    & 'C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe' 'C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/run_p02_mask_diagnostic.py' --diagnostic-mask wheels4 --diagnostic-max-decisions 240 eval --run-dir 'FRESH_UNIQUE_DIAGNOSTIC_RUN_PATH' --expected-head 'PINNED_FULL_40_CHARACTER_COMMIT' --checkpoint 'VERIFIED_LATEST_CHECKPOINT_PATH' --resume-migration 'MATCHING_REVIEWED_MIGRATION_JSON' --semantic-version v3 --experiment-id fsm_reference_p09_stable_v2 --mode semantic_residual_eval --stage full_episode --from-phase P01 --num-envs 1 --seed 4001 --max-decisions 3000 --device cuda:0 --no-headless

无迁移的精确兼容版本才可省略 --resume-migration。第二项只更换 mask选择与唯一run目录。expected-head/checkpoint/migration均原样交给官方preflight；wrapper不绕过版本验证。

录像直接由当前官方capture产生，原始物理失败/未完成字段不改成成功。提前到诊断边界时保留 DIAGNOSTIC_BOUNDED_WINDOW 与真实物理状态；所有writer仍经官方finally关闭。结果source/run manifests添加诊断说明，另有 diagnostic_policy_decisions.jsonl 与 live_preaction_drive_getters.json。


# CP199168 receiving profile：N8 只读可行性

结论：**当前没有可安全直接调用的 N8 正常续训路径。** 现成 N8 是旧 v2/324/P01 批量骨架，不能把 `-NumEnvs 1` 改成8后用于当前 receiving/372 checkpoint。不是要求增加任务成功门；继续 N1 和既定视频/有限诊断，不需等待本项。未执行 Python、测试、smoke、Isaac 或优化，未改生产。

实际源：[CP199168 manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_task_conditioned_hip_wheel_v1/checkpoints/history/checkpoint_step_000199168_manifest.json)，SHA `e98957a4072b01eff9d9f20838b07dd0890bc997ef7a268061b929ae8aa5b673`，199168 decisions /1521 updates /30420 optimizer steps，v3、372、N1、receiving actor，seed1001。

| 确定阻点 | 当前源码 |
|---|---|
| 并非只限制首次 sigma 迁移 | [validate_request:170](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_cli.py:170) 对**所有 v3 N≠1**拒绝；首次 receiving migration 的 N1 guard是另一个独立限制。既有 receiving checkpoint普通恢复也不能 N8。 |
| 默认配置/网络不匹配 | [dispatch_vector:1003](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_cli.py:1003) 未传当前实验配置；vector env无参加载旧 schema/reward，vector runner没有传 receiving policy/layout，默认 v2/旧 actor。不能用旧324网络替代或重置均值。 |
| 实际执行链未适配 | [vector backend:89](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_vector_backend.py:89) 明确拒绝当前 `same_tick_post_mapper_servo_margin_v1`；控制器仍加载旧task spec。当前 nominal geometry、前ACK tracking/headroom不能靠删guard保留。 |
| 恢复协议仍是旧324拓扑 | [topology/source_num_envs:3895](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_migration.py:3895) 的 role372仅N1，v3 topology也仅N1；[execution factor:4000](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_migration.py:4000)明确输出324。loader还要求同372布局的严格因子，因此不能借旧1→8 factor绕过。 |
| 当前采样审计是N1 | [actual request:1593](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:1593) 检查实际 head形状(1,2,12)，同一现行训练路径会调用它；需真实逐行H/cap/sigma证据，不能关闭审计假装兼容。 |

现有可复用证据：20260905旧[v2 N8 smoke](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v2/interface_smoke/20260905T0818372067281Z_gd19c655713bf_08f8a99ba7eb46009bdc197b41d626ad/vector_smoke_manifest.json)确实记录2次reset、1024总决策、0优化、各行原生target/物理隔离通过，并明确不要求任务成功；但只覆盖各行 P01=2/P02=126，绑定旧 d19c6557/runtime，不兼容当前649ccd9。现有 `test_semantic_vector_draft.py` 和 `test_semantic_vector_cli_integration.py` 覆盖单场景批量step、peer final-observation bootstrap及官方旧324 checkpoint 1→8→1权重/Adam/RNG恢复；本次仅检查测试源码，没有重跑优化测试，更不是当前372物理证明。

未来可复用单Isaac/8行骨架，不必另造训练系统；但要有明确的当前配置/控制/372采样接线与严格拓扑迁移，保留权重、完整Adam/LR、Identity、RNG、aux7/8和累计计数、清空旧rollout。当前不存在该已验证路线，本轮不实施。整组reset会截断未失败peer（已有正确critic bootstrap，但不会补出后腿物理样本）；128×8使2048总样本只有2次update，而N1是16次，不能承诺同学习强度或8倍净提速。

资源只作旁证：活动录像期间 RTX4080 Laptop，12282MiB总/4923MiB已用/7075MiB空闲、GPU23%；系统可见31.70GiB、空闲4.83GiB。旧N8 smoke记录设备采样占用峰值3.38GiB，WDDM进程峰值不可用。这既不证明当前N8容量不足，也不证明当前配置能安全容纳；未再启动资源竞争任务。

当前合法普通恢复仍是：在现有录像自然结束后的正常边界，用 `run_semantic_ppo.ps1 -Command train -ExpectedHead 649ccd906421d06c8b5c699f28101730910e885c -SemanticVersion v3 -ExperimentId task_conditioned_hip_wheel_v1 -NumEnvs 1 -Stage full_episode -FromPhase P01 -Seed 1001 -Device cuda:0 -Checkpoint <实际最新兼容checkpoint> -Decisions <剩余预算内>`；exact同runtime恢复不重传已完成sigma迁移，不重跑aux。没有可提供的合法现成 N8 命令。

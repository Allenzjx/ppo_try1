# RR 固定等待门与消费者只读审计

审计对象：生产 HEAD **28609010db4e57c5b34304a4ae2563c69f9d00b9**，激活配置
configs/ppo_fsm_reference_p09_stable_v2。审计时 src/configs/scripts/tests 工作树干净。
只读生产及现有测试，不导入 Torch/Isaac，不运行测试；未应用任何延期补丁。
以下路径以 C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1 为根。
旧 outputs/runs、延期源码副本、旧精确入口工程不作为激活实现证据。

## 结论及必须保留的区别

当前 RR 功能性抬升、名义动作派发、reward、termination 和自然 P01 独立评估中，
未发现“RR 先悬空/稳住 1 秒才允许动作或达标”的门。
生产 PPO 文件内 air_duration_s、edge_contact_duration_s 仅在
src/wlr50_clean/ppo/semantic_supervisor.py:839 及 :840 生成，:841 明确为诊断量，
没有消费者读取它们来判定抬升、推进阶段、奖励、结束、提升或视频成功。

这**不等于删除了所有时间条件**：

- stage_task_spec.yaml:60 的两帧连续物理采样用于抗单帧 dropout；不是 1 秒保持。
- :57 的 0.5 秒 history window 保存最近实测动作/几何响应；EDGE 失去近期响应会失去
  current_lift_valid，但不会仅因“等够时间”获得资格。
- :25 的 minimum_evidence_s=0.06666666666666667 是载荷转移/准备的短物理证据窗口。
  semantic_transfer_roles.py:138、:191、:193、:203、:206 使用该窗口，不能描述成完全无时间证据。
  已具备当前 RR 功能性抬升会走 supervisor.py:992 的准备绕过和 :1240 连续接管，
  不需要落地后重做准备时钟；该窗口不封锁全12维动作。
- supervisor.py:1247 在现有 15 Hz 决策格点推进阶段，每观测最多推进一个阶段。
  它是 120 Hz 物理下的8-tick调度同步，不是 RR 专用停稳等待。
- stage_task_spec.yaml:80 的 **1 秒 post_completion_observation_s** 是完整全身越障、
  当前平台区域及受控状态之后的 P13 收尾观察。:100 的 stable_duration_s=0.5
  在 all_stage 路径保留为 strict_recovery_quality 诊断，不应冒充 RR 抬升门。

## 激活生产路径及消费者

| 路径/行号 | 核对结果 |
|---|---|
| configs/ppo_fsm_reference_p09_stable_v2/stage_task_spec.yaml:7 | 启用 functional_lift_edge_v2；:8 启用 successful_fsm_derived_v2，:9 启用 all_stage_v1。 |
| src/wlr50_clean/ppo/semantic_supervisor.py:598、:616 | RR 明确用新 functional 判定覆盖旧一般 leg 初始/合格抬升计算，不仅增加诊断。 |
| src/wlr50_clean/ppo/semantic_supervisor.py:776 | _observe_functional_rr 由离地/实测3mm初始及8mm建立、全身动作响应、两帧真实其他支撑、安全包络判定；:815–826 不比较悬空秒数。:831 安全有效时允许继续探索，即使当前尚未建立有效抬升。 |
| src/wlr50_clean/ppo/semantic_supervisor.py:992、:1005、:1021 | role_prepared/transfer_ready/lifted/placed 的 RR 消费者使用 current_lift_valid 或当前真实可用 placement；历史抬升不能冒充放置，历史放置也不能冒充当前支撑。 |
| configs/ppo_fsm_reference_p09_stable_v2/stage_task_spec.yaml:208 | P09 完成仍为 placed_RR (:213)，而不是 RR 初始离地或悬停。P10 :220 仍要求当前可用 placed_RR。 |
| src/wlr50_clean/ppo/semantic_supervisor.py:1711 | 源关节及全身 owner 继续执行。仅额外 P01-derived rolling 建议检查当前 AIR 与几何；:1729 标注 fixed_lift_timer_gate=False。EDGE 不加盲目 wheel 推墙，但原源动作与 residual 继续。 |
| src/wlr50_clean/ppo/semantic_nominal_geometry.py:103 | RR 下落建议使用实测地面相对抬升及边缘距离，未消费 timer；:108 明确全12 residual 独立可用。 |
| src/wlr50_clean/ppo/semantic_reward.py:138、:153、:164、:196 | reward 根据实测 transfer、接触事件、actual-drive平滑和物理 potential；不读取 RR air/edge duration。0.25秒 rebound_window 是最近落地后无主动动作变化的反弹识别，不是抬升等待或硬失败。终端奖励仅使用同一 success/reason。 |
| src/wlr50_clean/ppo/semantic_env.py:48 | termination 先检查真实安全/任务失败，再读取 task success/reason 和200秒总时限。普通 phase switch 不在终止条件内。 |
| src/wlr50_clean/ppo/semantic_supervisor.py:865、:871、:878 | 1秒由全身 controlled_now 启动，观察结束仍须当前受控、未丢失平台区域、无安全失败。:891 的旧稳定/回零量是 strict_recovery_quality；:905 明确区分。:1276 只允许已启动P13收尾完成，:1284 不延长200秒硬期限。 |
| src/wlr50_clean/ppo/semantic_cli.py:393、:725 | 当前普通 B/C 和 A 独立评估都通过 _request_paths 传同一当前 task_spec/quality_score；PhysicalEvaluationRecorder 在 semantic_legacy_evaluation.py:59、:93 实例化/逐tick调用同一 TaskEvaluator，没有新增 RR 等待门。 |
| src/wlr50_clean/ppo/semantic_metrics.py:255 | 指标按实际物理阶段聚合，所有阶段未采样则不给完整质量分；:271 明确 score_is_not_success。当前 semantic CLI 不导入旧 checkpoint_promotion/final_reporting/stability_metrics 的历史推广门，也没有现成自动“更优”提升器。不能把“没有此RR门”写成“更优提升已通过”。 |

### 视频消费者的独立限制（未在本审计修复）

生产 semantic_video.py:91 的 video_configuration 仍仅调用 version_paths(semantic_version)，
未路由到当前 experiment_id。semantic_video_cli.py:75 的 build_video_core 无 experiment_id，
:31 构造 checkpoint runner 也未传372 observation_layout。
因此该旧视频入口**尚不能代表当前实验已接通的视频路径**；不能用它证明当前372视频成功。

该实现没有 RR 1 秒门：:411 使用配置对应的独立 PhysicalEvaluationRecorder，
:582 重放同一 TaskEvaluator，:590 要求真实任务成功。
semantic_video.py:33 的 POST_TICKS=184（约1.533秒）和 :246 的 post-success hold 是
成功后的录像上下文与稳定性复核，不是 RR 阶段前的等待；quality 不以这段尾帧稀释。
捕获/发布还要求实际完整文件/时间线 (:560、:656)，它们不启动或封锁训练 optimizer。
这些视频路径/372兼容问题属于既有延期项，本次不扩展补丁。

## 现有测试的具体正反例（本次只读，未重跑）

- tests/unit/test_p09_functional_edge_lift_v2.py:58：同一已抬起状态额外0/17/145 ticks，
  跨过1秒也不改变 AIR 功能性资格；:72 证明无需 RR 自身关节变化。
- :82：真实 EDGE/AMBIGUOUS 抬升可建立，但不改名为 AIR、不宣称越沿/放置。
- :93：长时间卡 EDGE 且无近期响应不获资格，仍可继续调整；:127 即使等150 ticks也不能
  把无关节需求/响应的纯轮墙面状态变成有效抬升。
- :112：单帧 dropout、同tick重复读不能累积资格；:138 P08→P09保留同一attempt。
- :168：真实早放置不需要空中保持；P10仍检查当前可用，不以历史TOP代替当前承载。
- tests/unit/test_functional_rr_nominal_geometry_v2.py:103：AIR时长2/120秒和1.5秒生成相同
  额外rolling建议；EDGE没有额外盲推。
- tests/unit/test_all_stage_finish_progress.py:73、:100、:120：P13收尾观察单独验证，
  不能把局部截止延展变成超过200秒的全局任务延展。

## 源 FSM 派生 N 的能力清单已经如实公开

同目录 first_execution_divergence.md:24 记录原源的 Full12/overlap、MotionExecutor、
成熟 mapper 载荷追踪、normal tuning 与有限 recovery。
:60–65 逐项记录此次保留/恢复的源 correction/time_scale、P10有限post-mapper bias、
单成熟mapper/无额外逻辑pre-slew、未完成owner连续交接、同一zero/nonzero N、prefix taper。
:51 明确源成功 P09 的872 ticks非零drive-feedback为0，故不接回未触发的旧精确膝角/回弹门。
:67 明确新N是成功FSM派生版本，不是完整原A逐位复现；:69 明确未复用源P13旧reference重试，
当前 continuous nominal + PPO 的P13真实可达性/恢复效果仍待实测。
未把未复用能力隐藏，也未把离线证明或现有测试称作本轮完整PPO成功。


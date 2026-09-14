# d4e4600：下一次原FSM / 已保存PPO配对视频入口

只读代码核查；当前HEAD实读为 `d4e46006b382093b960f2c428081d8a1c595bb6a`。未运行Isaac、测试或媒体验证，未读活跃训练/旧视频/模型字节；以下是现有入口能力，不是新的A结果。

结论：**可直接使用同一当前视频入口、experiment与seed4001，分别跑legacy_fsm_eval(A)与semantic_residual_eval(C)**。共享物理场景、自然P01/沉降配方、测量与外部验收；但冻结A的内部控制/入口/终止规则有意保留，不能称内部停止逻辑或初始物理状态逐bit完全等价。

## 已核实的共同条件

| 项目 | A与C共同路径/值 | 代码证据 |
|---|---|---|
|入口|v3、`fsm_reference_p09_stable_v2`、eval、seed4001、N1、fresh process、natural P01；禁止suffix、teacher offset、new-MDP与policy-distribution迁移|`semantic_video_cli.py:65–77`|
|场景/USD|两者使用同一个`_load_live_dependencies`的create_scene/reset_scene/RobotAdapter；A只替换测量reader，不换controller/physics。USD固定`C:/robotics_sim/wlr_robot/usd/wlr_robot_drive_test.usd`，创建场景时校验源锁SHA|`semantic_video_cli.py:80–110`；`isaac_fsm_backend.py:265–295`；`scene_factory.py:21–37,190–225`|
|spawn/physics|spawn(0,0,.04)m、identity rotation；dt1/120s、render stride8、physics devicecuda:0、gravity−9.81；相同摩擦/驱动/求解器配置|`infrastructure/scene_factory.py:21–61,203–209,282–302`。CLI Device控制PPO设备，不覆写场景固定physics device|
|沉降|自然reset180ticks/1.5s，ZERO Full12、相同atomic apply→physics step→readback；首次reset不恢复后腿快照|`isaac_fsm_backend.py:503–518,739–781`；`semantic_backend.py:144–188`|
|预动作与编码|当前experiment：extra preaction0、pre frames0、post-success额外ticks0；3次shader-only render不产生控制/物理step。不是历史180+64追加预动作配方|`semantic_video.py:574–624,714–723`|
|实测评价|A、C均用`SemanticSensorReader(all_stage_v1)`；同一experiment的task_spec与quality_score送入`PhysicalEvaluationRecorder/TaskEvaluator`，从tick0开始、逐physics tick观测；质量计算用raw body orientation，不用A校准相对姿态降低RMS|`semantic_video_cli.py:88–109`；`semantic_physical_sensing.py:172–186`；`semantic_video.py:624–637`；`semantic_legacy_evaluation.py:59–123`；`semantic_metrics.py:197–214`|
|物理终点|共同observer在TaskEvaluator首个success或failure的准确tick停止，包括不足8ticks末决策；没有碰撞但任务未完不能成功。成功还需独立物理流重放通过|`semantic_video.py:445–460,639–676,789–843`|
|上限/视频|两条core均有200s时限；视频固定MaxDecisions3000且不允许Decisions短窗口。15fps、1280×720、同相机、最多3000帧，保留真实不足8tick末间隔及其显示量化|`ppo_termination_v2.yaml:3`；`semantic_env.py:72`；`semantic_video_cli.py:72–76`；`semantic_video.py:29–33,60–84`|

绑定的是相同配置与物理执行配方，**没有实际运行A前不能声称已取得相同初态/配对成功证据**。真正配对时应对照两份runtime_contract/6配置绑定、natural_reset_proof、源tick0物理观测及scene配置。代码内的实际spawn与USD锁验证不应被“只看seed一样”替代。

## 有意差异与N/A（不能伪装成全相同）

- A保留`SensorFsmController`及`configs/fsm_states.yaml`、`recording_motion_contract.json`、成熟mapper和原入口检查；C为SemanticControllerAdapter + 当前nominal/residual。A原phase mask/观测/reward不用于学得策略，raw residual强制全0，模型/checkpoint/normalizer/optimizer恢复均N/A。
- A在沉降末30ticks/0.25s保留原level calibration；C用versioned fixed chassis axes、不做初始姿态校准。两者执行相同180个沉降物理ticks，A多的是读数/控制参考语义，不是追加物理预动作。当前共同质量评价另外从raw姿态计算。自然reset标记也有意不同：A=`normal_p01_reset`/requested_phase=None，C=`semantic_natural_P01`/P01；都必须reset_count1、options{}、prime0、tick0、无已恢复快照（`semantic_video.py:466–518`）。
- C必须official load已保存模型，conditional deterministic actor（`stochastic_output=False`），训练seed从checkpoint读、视频环境seed4001；两种seed角色不得混淆。C不执行optimizer update；A没有模型load。失败分支可能在末尾unchanged-model断言之前退出，不把初始load hash冒称最终复核（`semantic_video_cli.py:17–59`；`semantic_video.py:671–686`）。
- **外部成功/物理失败标准相同，但内部提前结束规则不同。** A原FSM阻塞或原controller结束可使core.done；C有当前任务阶段deadline。capture不会为了共同评价擅自继续已停止的控制器；两者只要没达到共同物理成功都会保留diagnostic，而非继承controller“成功”标签（`residual_direct_env.py:855–913`；`semantic_supervisor.py:1266–1290`；`semantic_video.py:638–676`）。200s限长共同，A旧TIMEOUT/truncated命名与C有限任务deadline命名不强行等同。
- 旧A历史运行不是这次当前配对视频；不重分类旧A，不借历史Trial043替代这次真实结果。

## 主控后续可用命令（本次未执行）

原FSM A不需要、并且**禁止**checkpoint（`semantic_cli.py:187–190`）；也不传ResumeMigration：

```powershell
& 'C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/scripts/run_semantic_video.ps1' -Command eval -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -ExpectedHead d4e46006b382093b960f2c428081d8a1c595bb6a -Stage full_episode -Seed 4001 -MaxDecisions 3000 -Mode legacy_fsm_eval -Device cuda:0
```

C使用同一命令的`-Mode semantic_residual_eval -Checkpoint <主控选定、已保存并核验的immutable官方pt路径>`。不要使用训练中的内存权重；若该checkpoint本身已是当前d4完整runtime，**不传迁移参数**。如未来HEAD/runtime与它不符，现有preflight严格拒绝，需另有既存已审查的合法迁移；不能用视频入口绕过new-MDP续训。此处不提前虚构下一次checkpoint文件名。

PS launcher本身检查无其他Isaac/PPO进程、ExpectedHead相符且生产路径clean，并持有单进程锁；由主控在当前训练实际正常结束后顺序执行。A产物路径`video_eval/baseline_A/<run>/source`，C为`video_eval/validation/<run>/source`。

验收最小文件集合：各自`semantic_video_source_manifest.json`（natural_reset_proof、共同配置、C真实checkpoint_load_provenance）、`physical_observations.jsonl`（tick0及完整真实终点）、`video_policy_decisions.jsonl`（A全0/C真实执行及末partial ticks）、`native_tick_audit.jsonl`、`viewport_buffer_video_manifest.json`与frame ledger。captured/文件PASS不等于任务成功；失败仍保留原片/源manifest。若只有已知MP4容器时长异常，后续可独立处理无损诊断/合规发布，但不改任务判定或把媒体处理设为optimizer门禁。

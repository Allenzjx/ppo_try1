# Shared training / physical parity review

基线：`285307603f43bfc2846dada5f3ede73b0dc5f536`，来源 CP138496 / 1047 updates / HISTORY372。已完整读取本轮 `0610eac7-775f-41a0-a772-223fe0147ef9/pasted-text.txt`。以下基线结论来自源码及现有小型 JSON/MD；没有重跑 R/A/B/C、加载 PT、重新哈希大资产或重新认证全部阶段。并行实现中的新代码不由这些旧证据自动认证。

## 1. 确证缺陷及范围

**角色计时借用：可由调用顺序复现，不是已证实的真实失败唯一原因。** `TransferRoleTracker.observe` 的旧 `response_since` 同时由 `max(measured_transfer, space_response)>0` 维持。只有接收侧相对机体收缩、无定向 CoM/减载/initial 的准备动作可先积满短窗；随后第一次有效 load drop 到卸载范围，便立即获得成熟的 transfer。旧测试只覆盖“旧窗口没有任何准备响应→新减载”，没有覆盖这个空间准备正响应反例。它违反新规范 P11 的当前尝试证据要求。

已获授权的窄修复仅在 `physical_acceptance_version: all_stage_v1` 启用：准备响应与真实转移响应独立计时；后者仅允许原有 measured-transfer 路径（定向 CoM、实测减载或当前 AIR initial），不因阶段变化清零。新模式遇任意明确 `load_fraction_valid=False`，角色不可用并中断两时钟；有限 placeholder 不是卸载证据。旧未启用模式保留原义。准备不是 FK 可达性或静力稳定证明，时序证据也不是电机功或因果动力学证明。

**终止归因标签混用。** 基线 backend 将所有 `INCOMPLETE_CONTROLLER_BLOCKED` 标为 timeout / `TASK_DEADLINE_OR_STALL`，但 stall 实际不终止；该结果码还承载 RR_FIRST 顺序违规。新 `termination_source` 应由具体触发点给出，不能由该聚合码反推“卡住”或阶段截止。此处是分类缺陷，未证明状态写入、控制或传感链损坏。

## 2. P01–P13 局部时限的真实接线

|阶段|基线局部时限(s)|
|---|---:|
|P01|30|
|P02|15|
|P03|20|
|P04|20|
|P05|30|
|P06|40|
|P07|15|
|P08|20|
|P09|30|
|P10|20|
|P11|20|
|P12|30|
|P13|60|

唯一普通阶段截止调用在 `TaskStageSupervisor.observe_and_update`：每个真实 physics observation 更新 evaluator，符合 entry/completion 的普通切换仅在8 tick边界发生；切换先于截止检查，可在同刻完成后合法进入下一阶段。随后若当前 stage age 到表中上限，或整回合 age 到200s，统一产生 `INCOMPLETE_CONTROLLER_BLOCKED`。6s/.01 stall 仅诊断，`terminates:false`。基线没有按净进展/当前合法任务过程给予有限延续的分类路径。

这套旧有限任务定义内部自洽，但没有完成新规范要求的逐阶段时限审查/可恢复截止分类。不能直接将全部 deadline 改 truncation，也不能以统一增大 timeout 代替该审查。原 FSM 的 WAIT_ENTRY/Recording 动作源结束是另一条控制器路径，不是这些 task-stage 定时器。

## 3. 观测、PBRS、GAE：未发现这里的接线错误

- 372 的原324列保留 phase one-hot、`task_times=(now, remaining_task_time_s, stage_elapsed_s)`，固定 scale200；原固定局部时限可由 phase+age恢复，整体剩余时间显式可见。角色48列仍是当前消费者摘要，不是完整 deque/锚点重建。
- 新计时仍由既有 `continued_response_fraction` 表示真正 transfer 成熟度；`preparation_progress/preparation_ready`、workspace/motion 分别表达准备进展。独立准备时间仅诊断，不声称372是严格完整 Markov 状态；如后续另加影响决策的 retry/隐藏计时，需要重新检查可观测性。
- 实际 `gamma=.9985, lambda=.99, 15Hz decision, 120Hz physics, rollout128`。reward 每个已发 policy action 计算一次 `F=5*(gamma*Phi_next-Phi_before)`；body/contact/smooth/time cost 按实际 physics dt 积分，time cost `.02/s`，终止 ±40。不是每个120Hz tick重复乘 gamma。
- 真终止的 `Phi_next=0`、`terminated=True/truncated=False`、`time_outs=False`；最后一个决策可少于8tick。普通 phase 切换不 done，不重置 evaluator/bridge/filter/nominal/reward历史。官方 GAE 的128批末用真实后继观测 bootstrap；没有发现普通阶段切断或将 reset 后下一 episode value 串进真终止。
- 匹配 gamma 的 PBRS 满足 `sum(gamma^t F_t)=5*(-Phi_0+gamma^T Phi_T)`；真终止最后项为0。局部正/负 dense 信号合法，不等于改变最终折扣目标，也不保证数十秒后的成功信用容易学到。已有 official-RSL GAE 单测覆盖有限终止/128非终止bootstrap/普通phase标签；本审查未重跑它们。
- N1 当前 adapter 拒绝外部 truncation。vector peer-barrier 是独立路径：在 reset 前用各 peer 真实 final observation 计算 `gamma*V`，只对非真终止 peer补偿并切批；不能把它当当前 N1 suffix 训练已经验证。正式评估到外部 MaxDecisions 会记录 `window_ended_before_task_terminal`，不自动算成功。

## 4. R/A/B/C：哪些共用，哪些未配对

|路径|已核实证据|边界|
|---|---|---|
|R / Recording|现有 `recording_evidence.json` 和 environment lock：持久Full12、缺通道hold、显式stop、canonical单位/顺序；命令源与派生120Hz记录有不同时间范围|没有重新运行原工程；不能据命令/lock声明证明其本次动态初始化、传感历史或全身动力学与B/C相同|
|A / 冻结 FSM|现有旧v2 A source manifest：seed4001、180 settle+64额外零动作、实测level标定；旧 P10 WAIT_ENTRY/历史关节与反弹入口未满足|与当前版本、seed、初始化时长和评价器均非配对；旧A未完成不是当前PPO优劣结论|
|B / 同版零残差|`SemanticIsaacBackend` 自然P01 reset，180 settle+0，固定轴参考；同scene/reset/adapter/reader和task spec路径|raw0不是恢复旧A控制器，也不表示所有native/名义动作是0|
|C / 同版策略|和B相同backend/evaluator配置入口，仅策略raw action来源不同；确定性正式评估无prefix/optimizer|ACK及四写审计只证明调用/目标链，不单独证明真实接触、载荷转移或任务成功|

公共 scene factory / `_reset_physics_lifecycle` / `RobotAdapter.from_scene` / live sensing / `SensorReader` 来自同依赖入口；每tick唯一 native写→physics step→readback→真实sensor observation→controller。声明锁包含同机器人USD、120Hz、gravity−9.81、solver8/2、servo600/60、tracking gain8/max10/每4tick、原硬限及摩擦；本次没有重哈希或重新测量这些物理参数，声明一致不等于动态状态一致。

B/C 的版本化软件差异包含 continuous semantic nominal、post-mapper residual、geometry修正、2° headroom、previous-ACK requested tracking reference；它们不是 frozen A 原控制器。原冻结 mapper/servo硬限及最终1.25°/tick链未因本次只读审查改写。没有从所读代码/小manifest确证第二次mapper、反向时间或绕过真实sensor的损坏；也没有物理新运行去宣告“全部通过”。

注意旧 `existing_A_metrics_reference.json` 的 `maximum_commanded_wheel_speed_rad_s=0` 缺少当前同链可信语义，不能解释为A实际一直零轮命令；旧 schema名含v2、旧物理valid和视频封装问题也不能直接替代新版本真实性判断。旧A后续P11–P13未访问，稳定性缺段仍应null；Recording缺完整角动量/能量/力矩历史时保持不可用。

## 5. 实施/验证收据

本子任务生产范围仅 `src/wlr50_clean/ppo/semantic_transfer_roles.py`，新增专属 `tests/unit/test_semantic_transfer_response_maturity.py`。其余时限/分类/传感实现归主线程和对应owner，本报告不代替他们的最终回归/真实验证。

首个测试调用因未设PYTHONPATH发生3个collection error，不是测试通过。补齐项目src路径后初版三文件76项通过；扩展承载有效性用例时，23个新用例因并行 all-stage constructor 增加必需配置而失败，未视作计时逻辑失败或忽略错误。按主线程决定，专属测试以真实旧 TaskEvaluator 生成当前物理字段、独立启用新 tracker，隔离另一个 owner 正在修改的接触验证；未知载荷标志明确注入，不能把这些测试声称为新 all-stage detector 验收。

最终命令范围为 `test_semantic_transfer_response_maturity.py`（25项）、既有 `test_semantic_transfer_roles.py` 与 `test_semantic_role_observation.py`，合计 **83项通过，exit0，命令耗时约3.38s**。覆盖四腿空间借龄拒绝/独立持续转移、已成熟定向响应允许、旧字典兼容、当前支持中断、承载不可用、旧硬QCP一致、阶段不清历史、372现有槽位成熟度。所有CPU调用隐藏CUDA、限制OMP线程1；未运行Isaac、GPU或优化器。已停止Python及编辑，未提交。

计时修复不把原滑窗命令/实测运动的关联启发式升级为因果证明，也不保存完整接触/力积分历史；接触可信度、固定方向锚点及真实仿真跨阶段效果仍须由共享实现和后续真实记录验证。

结论：角色借龄和聚合终止来源混用是具体可复现问题；阶段截止是旧定义自洽但新规范尚未分类到位。尚无此次只读证据证明 R/A/B/C 底层执行链损坏或某条物理轨迹因此必败。未访问阶段、未配对历史与未运行的验证均不记为通过。

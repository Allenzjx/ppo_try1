# RR 最小候选实施建议（生产未改）

依据：`residual_RR_diagnosis.md/json`、`rr_lift_qualification_replay_fixture.json`、`rr_carry_predicate_replay.json`。全部来自旧封存原输入，尚未证明修改后的物理结果。当前 zero 自然运行结束前不实施。

## 1. 资格：不把接触爬高计为自由抬升

范围：`semantic_supervisor.py:TaskEvaluator.__init__ / _observe_functional_rr` 及所需诊断输出。

- 跟踪连续真实 AIR 段；起点高度取该段之前最近真实接触的轮底高度。任何ground/obstacle接触结束该AIR段，不能将接触期间累计的升高计入新自由增高。
- 新 qualified 需当前AIR、连续样本达到现有2、自由增高达到现有8 mm、既有whole-body response、其他真实支撑和安全条件；不要求RR自身关节变化或FL接触，不增固定hover时长。
- ground重新接触撤销该attempt已有资格。有效自由抬升已建立后，允许满足现有几何/支撑条件的edge/TOP连续调整，但未知接触本身不能新建资格。
- 初始3 mm仍是探索进展，不能独立充当crossing信用。保留原`ground_relative_lift_m`含义，新增明确free-rise诊断，不能静默改现有字段含义。

## 2. 动作时序：只暂缓尚未消费的源段

范围：同文件 `NominalMotionProvider._sequence_permission`；必要的私有状态/诊断；`_continuous_advisory`仍走现有单一执行链。

**P09待执行knee waypoint**：从实际contract识别该carry序列的knee waypoint，不硬编码历史目标角。尚未在top时，若当前没有有效free-lift，或当前AIR且gap低于既有15 mm、真实轮底正在下降，则暂不消费这条source时钟。保留当前nominal，不回放旧姿态；所有12个residual、mapper/HISTORY、真实物理与其他准备层继续。恢复条件来自当前实际状态，不是静止或历史入口相似度。

**P09首次四轮正向原子组**：source 1.9333 s对应组需要当前有效free-lift/AIR或合法current TOP continuation；**不加15 mm滚动门槛**。显式stop保持其写明通道所有权。只检查将消费该原子组的边界，不每tick强制全轮归零，不拆坏其四轮原子性。

注意：现有 `_rr_top_continuation_allowed` 还要求 source endpoint/late-group-wait，不能直接拿它判断尚未开始的首次wheel组，否则其时序前置条件会恒拒绝。可抽取仅物理TOP资格的小helper（当前Q、已验证TOP bearing、无ground、lateral/允许gap及真实其他支撑），由不同owner条件包裹；不能只凭surface TOP或历史Q兜底。

暂停后的实际动作仍可能掉落/超时；不把predicate触发视为修复成功，也不屏蔽policy来强制复刻zero。

## 3. 已有夹具能证明什么

- 自由抬升组件：zero5434应获资格；CP4633应获资格、4731ground后撤销；CP4940接触上升不得获新资格，晚AIR最多0.516 mm不能洗白。夹具覆盖zero1085 ticks、CP1173 ticks；不是从P01重演整个evaluator。
- source原输入predicate：CP首次would-hold=4697，前观测gap9.892 mm、距前缘−228.135 mm、vz−0.14419 m/s；zero六个knee waypoint全不触发。
- wheel组：CP4833前ground/Q=false应等；zero5641前AIR/Q=true应放行，虽gap为−12.008 mm。zero近corner5936 gap−7.219 mm时无待执行knee waypoint，不应被该窄规则阻塞。
- 加定向合成正反例：whole-body lift且RR关节不动可获Q；FL AIR不能记承载；已有Q轻触不自动失败；新contact-rise不能Q；显式stop不被等候吞掉；phase交接不reset、无额外write；没有knee新waypoint时不因普通下落暂停所有owner。

## 4. 不扩大本候选

暂不修改固定base几何投影、reward、sigma/温度、动作范围或网络容量。实际轮底下降/机身速度可输出诊断，但不得把局部FK或新增预测包装成物理保证。隔离新版任务语义/时序迁移，保留兼容权重、原候选、checkpoint和zero；清旧未完rollout后真实采样、更新、重新加载并从P01验证。

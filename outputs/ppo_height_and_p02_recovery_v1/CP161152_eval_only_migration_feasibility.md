# CP161152 → 当前 a9c：只读迁移可行性

这不是migration plan，也未调用builder/validator、加载张量、保存模型、修改latest或训练旧checkpoint。当前16次更新及维护后的最新模型仍先评估。仅复用本次10份旧评估元数据、当前训练启动合同及现有迁移消费者的窄范围规则。

CP161152旧正式评估记录372维/12raw、HISTORY rho=.9、identity-RSL/fixed-schema政策合同；其action_schema与observation_schema记录指纹均和当前相同，包含既有尺度布局。没有应清空网络、重置normalizer或新增观察维度的依据。元数据兼容不等于已经执行过当前loader验证。

需要分别覆盖的变化：

1. 原 d4 nominal → 部分顺序/连续owner调度。现有 `_timing_only_factor`（semantic_migration.py:1261）只允许nominal.sequence_semantics加入和受审NominalMotionProvider范围；不能借此混入reward/height。
2. 当前functional carry/capture-settle body消费。现有 `_body_reward_factor`（1390）只允许两config键和窄body消费者变化，需要completed同N参考（不要求成功）及同一source checkpoint绑定的timing ancestor。已经生成的167424 body plan不能直接当161152的plan。
3. 当前RR bounded v3 geometry、FL−4 preparation与晚期组入口。现有 `_height_recovery_factor`（1592）要求source已有timing，保护reward/observation/action等配置；task只允许height_recovery块变化，code受exact reviewed hashes及范围约束。当前168192→a9c height plan来源和合同都不是161152。

关键限制：当前入口1712明确height factor不能与timing/body等顶层混用。直接把161152送入当前单height方案会因旧timing/reward差异超范围；不能仅靠“都是372”宣称现成方案已可运行。若未来确需比较，必须另行审阅来源绑定的组合/阶段迁移机制；本次未构建该机制，未创建中间checkpoint，也没有为规避validator而改代码。

评估方式本身可以指定历史checkpoint并使用新的独立evaluation run目录，不需要回滚或写official latest。若经审阅的eval-only迁移可用，继续保持冻结权重、0更新、自然P01和当前统一验收；不要把旧run的FR放置当新规则结果。旧optimizer预算/normalizer/HISTORY状态不得伪造为当前latest计数，也不复用旧rollout。

此为以后必要时的备用可行性说明，不是当前训练门禁。现阶段只保留161152作为最近的旧前腿实证候选，先完成正在运行的训练及最新checkpoint正式评估。


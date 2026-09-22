# 可选辅助学习的方法标注：隔离候选，不影响当前纯 PPO 导出

当前缺口已确认：review 的 checkpoint 校验认可 task branch 的原 PPO 起点／计数，但不会检查 branch 内的 auxiliary_mean_learning；quantity pair 的 top-level 新字段检查也不足以识别嵌套 ledger。故若未来采用 aux，直接走旧入口会误标 pure PPO。

本候选仅提供 100 行以内的默认拒绝守卫。它先复用旧 sealed_source／checked_review 的封存、checkpoint、素材哈希和完整帧检验，再读取真实 .pt infos，与 sidecar 精确一致后递归检查辅助记录。nested auxiliary_mean_learning、未知 aux 标记、祖先／列表内记录、空或零计数 ledger 均拒绝旧 pure PPO 路径；不会把 presence 当成有效辅助学习证据。未修改两个现有媒体模块，未运行编码或物理。

启用前根需选择：纯 PPO 媒体通过这个独立 wrapper；或在现有 quantity helper 实际 infos 比对后增加 require_pure_infos(infos) 单点守卫。仅有候选文件并不会自动保护绕过 wrapper 的旧入口，所以 aux 启用前必须把正式媒体入口接上守卫。当前 CP192512 无 aux，旧纯 PPO 导出不受影响。

未来真正批准专用 aux adapter 的最小条件（现在未实现，也不编造 ledger schema）：

- 精确核对 branch['auxiliary_mean_learning'] 的官方持久化 schema、父 checkpoint／数据证据哈希、独立 aux steps、实际允许的 FL mean rows、保留的其他权重／Adam 策略和 RNG；PPO decisions/updates 不因 aux 增加；fresh PPO 后继 checkpoint 保留 ledger。
- `PPO + LIMITED AUX`（中文可写“PPO + 有限辅助学习”）同时进入画面固定标题、文件名和 media/pair receipt。列出 PPO 数量与 aux steps 两个独立计数，不把 aux 计入 PPO；stochastic/deterministic 模式另行标注。
- 保留旧视频的封存、真实 checkpoint/素材 hash、camera／场景 seed4001／评价配置／natural-P01 entry、正常速度、全程帧和短侧冻结标识。不能先调用旧 export 生成“PPO FULL12”再只改 JSON 名称；需要在独立 adapter 中替换标题文本入口并沿用原校验／编码流程。
- quantity-only 的控制等价凭证仍单独成立，但学习方法声明必须另列 aux；不能靠 same runtime 或 budget factor 给 aux 合法性背书。无效、未知或缺失继承 ledger 一律拒绝并要求专用 adapter。

这里只准备拒绝误标的候选，不批准任何 aux 数据、动作、损失或训练，也不声称带 aux checkpoint 已存在。

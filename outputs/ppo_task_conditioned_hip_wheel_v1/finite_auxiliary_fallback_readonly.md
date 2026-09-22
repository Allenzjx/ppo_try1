# 有限辅助学习后备：只读可行性，不启用

依第12节，仅为固定CP评估仍失败后的备选；1536 decisions/12 updates与两次失败不自动证明“长期未吸收”。当前纯PPO/视频照常。未写辅助代码/标签，未改生产/仿真。

## 保存接口：有底层接点，没有现成完整辅助流程

`semantic_training.save_semantic_checkpoint:509` 可存任意infos，经官方runner实际重载验证actor/critic/完整Adam/Identity/RNG；`load_semantic_checkpoint:752` 返回这些infos，要求空rollout、无pending transition。官方RSL `PPO.save/load` 已保留同一个optimizer；actor确定性forward提供含真实HISTORY的条件均值。**无需另造框架、清空网络或Adam。**

但**没有现成aux入口/账本**；`train_semantic:1904` 仅继承白名单，任意aux字段下一次存档会丢。最小衔接：完整PPO更新后的空storage → 独立预算aux → 现有保存/重载 → 新采集rollout。须持续保留数据/代码hash、源CP、aux loss/steps、预算及祖先；不冒增PPO计数，另列aux与合计。相同决策数另命名CP，不覆盖原件。生产变更需显式版本/契约，不能冒用archive-only或旧sigma迁移。

## 哪些片段够格

| 已封存片段 | 可以支持 | 不能支持 |
| --- | --- | --- |
| RR-B：qualified后FL/RL−3 | 后抬升carry/释放保持候选，真实放置 | 不教初始卸载；RR放置慢.183s，不泛化任意入口 |
| RR-A：准备期RL−3 | 时机反例 | qualified后.517s回地，非持续carry正例 |
| FL−3诊断 | 执行与gap响应证据 | 最小gap仍11.322mm，无捕获/P06 |
| 正式update1420捕获 | 真触顶/捕获局部样本 | 7个P06决策后掉载，非可靠保持；正A不等于好动作 |

**数据未就绪：** `direction_probe.py:204–218` 仅probe_entry有完整372；逐拍有注入raw/执行/后态，但缺完整pre-action372。不能将后续−3配给唯一入口或拿请求摘要冒充全观测。除非能证明各观测组/真实历史可重建，否则需有限新配对采集，不能造零历史；旧CP185856的Phi语义也不能直接重标为新版本。

## 最小风险与退出条件

只监督有证据的相关通道，不把其他通道标签设0。共享MLP及既有Adam动量会影响其他输出/σ，不能声称只改hip；须核验critic、分布、双组LR/Adam及真实HISTORY，不reset或直接放大梯度。aux不入PPO buffer，有限预算退出后无teacher/zero fallback，仍须同一模型自然P01验收。**现接口/数据不足以无改动直接开启合规aux；不是当前训练或视频门禁。**

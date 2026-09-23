# 一次有限前段保留，不是 PPO 或物理成功

实际源 CP222592/v9：222592 decisions /1704 PPO /34080 Adam；原分支2048/16/320均保留。读取检查通过；未改网络。42项定向测试通过。

当前94/93真实局部成功前段样本的原始 raw 标签保留，reset row0排除。初始负梯度使全部训练/验证 FL、FR、RR wheel REQUEST方向为正，RL为负；不能把它称为四轮强制同向。FL每单位SGD学习率的平均REQUEST切线约1.19e-7/1.20e-7 rad/s，最大1.388e-7。真实选定梯度范数2.083e-5，HISTORY链式0.1因子未重缩放。

选择单次固定32×1000独立SGD预算；没有自适应LR搜索。初始切线×总LR32000对应FL约0.0038（最大0.00444）rad/s、最大FL knee约0.206deg。这只是冻结初始梯度的局部估算，不是最终拟合、闭环HISTORY积累或恢复预测。它针对已观测到的P02前送降低，不改controller/reward/timeout。

每一步相对原CP的累计约束：各joint REQUEST≤2deg，各wheel≤0.05rad/s，全12维Gaussian双向KL≤0.5，绝对log-sigma改变≤0.25。后者保留既有有限拟合的分布边界，非物理成功/相似度验收。第一次不满足就回退该512参数提案并停止，不增预算重试。

仅actor第一层P01/P02两列可改；均值和sigma可改变，公开记录。P03–P13同输入Gaussian必须位级不变。PPO Adam、critic、LR1e-5、Identity、全RNG、旧AUX/迁移谱系及PPO计数保留。新AUX单独入当前RRbranch ledger，不发布latest指针。官方保存/重载之后清空rollout，重新采集真实PPO，再自然P01确定性完整评估。

旧AUXFR1曾未恢复确定性P02，本次也无成功保证，不追加新policy或部署隐藏teacher。

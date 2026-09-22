# Block06：普通 PPO 的局部学习方向（只读、有界）

范围严格为实际 CP209920_AUXFR1 → CP211968 两端、rollout/update1606和1621、其256条样本及末段一条前驱、既有254条历史det P02输入。没有拟合/GPU/物理/生产改动。可复查脚本 `review_block06_endpoints.py`，完整数字在 `block06_two_endpoint_readonly.json`。脚本不写文件，不构造优化器；JSON由本次报告保存。未扫描其他历史块或中间权重。

## 普通 PPO 确实继续偏离这组局部可行参考

同一历史det P02输入、同一HISTORY下的 raw mean MAE（不是最终控制或物理效果）：

| 通道 | 源AUXFR1 | block06末 | 末减源的平均raw μ |
| --- | ---: | ---: | ---: |
| FL knee | .034034 | .037122 | −.003089 |
| FR knee | .003259 | .007159 | −.010418 |
| FL wheel | .026491 | .046792 | −.020301 |
| RL hip | .011548 | .013399 | +.001850 |

首/末rollout各128个真实输入也呈相同方向，不是拿不同轨迹直接相减。首rollout两条P01输入：FLknee/FRknee/FLwheel平均再负移 `.002605/.007911/.014305`。只读 functional 参数组合仅恢复旧首层P01/P02列、保留末端其余权重，P01首拍FRknee漂移仍为`−.00804281`（全部新权重`−.00804593`）、FLwheel仍`−.01473920`（全部`−.01473857`）。因此这次P01同输入变化几乎由**其他共享参数的变化**留下；这不是对训练时每一梯度的因果分摊，也不证明全部物理失败都由该漂移造成。

## 信用分配：不是几何成本直接压住任务

- 选定256条的 task-space geometry weighted cost **全部0**。首/末128条姿态quality总成本仅`.009302/.006446`，时间成本`.170667/.170000`；末rollout含真实`−40`终止事件。没有证据支持此刻再削弱几何/稳定性reward。
- 首rollout里35条“FR净空下降且前缘距离退步”的P02样本，20条standardized advantage为正；相反，已有正净空且前进的73条中34条为负。例如209927净空`+5.50mm`、该拍前进`20.57mm`、reward`+.13141`，rawGAE却`−.39760`、标准化优势`−4.8550`，其最后一次minibatch前logp相对采样时降低`.30464`。其GAE分解：potential`+.16366`、time`−.08773`、body`−.00503`、critic TD`−.46851`。局部几何进展并不保证后续回报，故这不是“正确动作被错罚”的充分证据，但明确不是geometry cost造成符号翻转。
- 末rollout终止样本211888：old V=`−7.01400`、实际terminal return=`−40.95105`，rawGAE=`−33.93705`，no-bootstrap。此处有可直接量化的价值低估损失；首rollout无终止，其value“错误”不能靠这份局部截断数据独立判定。
- 末rollout的前48条来自失败末段，后80条来自新episode。whole-rollout rawGAE均值`−9.75543`、样本std`12.96315`。新episode中58条FR净空低于障碍顶的样本，其standardized advantage **58/58为正**，但rawGAE只有30/58为正；实际最后一次minibatch前logp有37/58增大。211889 P01首拍rawGAE`−.26641`被归一化为`+.73200`，该拍FLwheel创新`−.119812`，实际mean-head loss梯度和为`+.33892`，对应梯度下降局部推动该输出负向。共享权重、其他样本、clip及Adam使这不能直接等同最终参数变化。

这些是原始PPO中的**相对优势/有限horizon/价值估计**效应，不是成功重标记或实现mask bug。gamma=.9985、lambda=.99，GAE衰减gamma*lambda=.988515，e-fold约86.6 decisions/5.77秒；128步rollout约8.53秒，而失败约21–22秒。早期入口主要靠critic/bootstrapping接到远端失败，当前value有明显不准之处。首末两块数据不足以推出“全局归一化一定错误”或“直接增大lambda一定修复”。分解的非终止tail value由已存return代数反推，不冒称独立测量；FP64分解与原FP32 GAE最大误差首`3.16e−6`、末`1.39e−5`。actual minibatch advantages逐行匹配，每样本恰用5次。

## 最小判断与下一步

**优先处理共享均值漂移/入口与成功分布丢失，不凭本审查立即改reward。**普通PPO不是没更新，而是当前仅失败front数据、稀少P01、有限horizon/value误差会给均值持续提供并不等同“更接近局部成功参考”的梯度；当前真实block07继续，不能打断它把审查做门禁。

Root明确 P03+逐位不变不是用户硬要求后，下一次有限AUX采用现有 `mlp.4.weight[:12,:] + bias[:12]` 的**12个mean输出行**是合适的最小子空间：固定trunk及12个logσ行，既保留当前389/真实.9 HISTORY和全部兼容权重，也让同输入σ精确不变、mean具有现有hidden特征的状态依赖。它不是重置head、新增policy容量或教师部署。历史P02实际raw加已核验P01第二拍作为监督，P01单行独立标记，不伪造同阶段holdout；P03–P06以显式mean保护loss和全Gaussian/REQUEST trust约束，**不保证均值逐位不变或未来轨迹不变**。

32步仍可作为有限上限，但应由该子空间的真实梯度/单位LR响应选择预算，不搬首层LR500；不能先承诺有意义或成功。保持全部旧Adam/critic/Identity/RNG，单独记录新增AUX、保留已有完整ledger、清空未完成rollout。后续若普通PPO再次抹掉同输入改善，应据真实信用分配/覆盖决定训练方案，而不是无限重复相同AUX。本次仅设计，不执行该更新。

# v9 / v10 真实 PPO 与独立 AUX 核对

JSON-only 封存审计通过；未运行 Torch、测试、仿真或读取当前录像流。完整字段与证据路径见同名 JSON。

| 真实运行 | 已计决策 | PPO | Adam | 输入阶段 |
|---|---:|---:|---:|---|
| v9 P07 前驱课程 | 640 | 5 | 100 | P07=2，P08=2，P09=636 |
| v10 自然 P01 | 128 | 1 | 20 | P01=2，P02=126 |
| 合计 | 768 | 6 | 120 | 其他阶段本两块均为0 |

v9 原计划2048，仅消费640，剩1408不计。两次 successful_nominal 前缀各645决策，共1290决策／10320物理步，均明确零 PPO 信用。第一集537 learner决策，78.758333秒在P09真实未完成：RR历史合格／跨沿为真，当前无TOP、未放置，gap29.968060mm。第二集103 learner决策在P09／49.866667秒于完整更新边界截断，非终态，不称失败或成功。

独立 front-retention AUX 为32 accepted／32 attempted；PPO和Adam计数新增均为0。官方独立重载通过；其32/32账本已在本次真实 v10 普通 PPO 保存中整对象保留。旧四事件103/104账本、旧历史元数据、各分支起点及命名迁移回执均保留。原 AUX execution 中“尚未验证下一次PPO carry”是当时记录，不覆盖历史文件；此次证据补齐它。

v10 完成一次真实更新1705，最终222720／1705／34100，RR分支2176／17／340；有效LR=1e-5。actor、critic、Adam哈希均变化，RNG正常推进；Identity normalizer哈希不变，官方 save_load_round_trip=true。当前审计仅复核这些保存证据，未重复加载张量或使用GPU。祖先分支仍从220544计数，未借用另一主线640。

128条记录全部满足：raw在抽样、policy_request、applied及native审计中相同；μ／σ与原分布相同；一次随机抽样、无额外抽样；logp逐条相同。用记录值双精度重算 Gaussian logp 的最大差为3.3862e-6。reward按原float32存储转换128/128一致（最大转换差2.0400e-9），所有1024物理步native审计通过，12通道mask全1，FL/RR assist均未取得owner，RR追加21特征全部为0。

P01→P02普通切换没有done；128条无终态，尾端使用正常bootstrap。8.533333秒P02是预算边界partial，不能称本次完整越障成功，也没有v10后腿学习样本。当前自然P01录像结果不在本审计范围。

普通resume的通用临时 `resume_migration` 字段未继续复制，`resume_ancestry` 更新为此次恢复来源；正式命名的v10、v9及全部旧迁移回执整对象不变，不构成谱系丢失。

最终checkpoint：`checkpoint_step_000222720.pt`，记录SHA `069a71f547427b69491ff749dccc14fcf403565d81b3c37db3c6ac9b1e739e55`。

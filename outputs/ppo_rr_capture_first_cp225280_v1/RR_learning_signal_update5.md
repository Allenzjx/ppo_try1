# 第5次真实更新：RR学习信号（已完成CPU只读分析）

输入：train_v2_fresh512_5f8487b，rollout_0005.pt / likelihood_0005.json。
主线程明确确认 Isaac 正常退出后运行；分析进程 exit0，约5秒。
此后只用标准库读取结果。没有 PPO/AUX、权重写入或物理执行。

## 计数与核验

新增512个真实 PPO 决策、1个 PPO update、20个 Adam step；1995个真实前缀决策 credit=0；
全部512为P09，跨2次捕获入口，AUX=0。
累计local2560/5/100；task_v2 512/1。
CP227840 SHA：
d1a6147bf4ee210b845fcb55e25b25c9d35b4a15d5e5d09124b08a5d752a00d9。

保存的观测、raw action、old logp、mean、std、reward、done 与决策日志逐值一致。
CPU重算 logp 最大误差3.82e-6，GAE归一化4.77e-7；
加载采样前checkpoint重算相同448输入的conditional mean最大误差2.39e-7。
没有将旧447观测随意追加新资格字段。

## 真正改变了什么

使用本次512中的同一组486个“当前AIR、合法顶部XY、同attempt资格有效”输入，
分别运行更新前CP227328和更新后CP227840；没有拿不同rollout的均值直接比较。

| 固定输入上的指标 | RR hip | RR knee |
|---|---:|---:|
| conditional mean变化均值（raw单位） | -0.0056817 | -0.0021691 |
| 486输入中负向变化数 | 486 | 486 |
| local raw head mean，更新前 | +0.0174039 | -0.0907003 |
| local raw head mean，更新后 | -0.0394133 | -0.1123917 |
| 更新后sigma均值 | 0.282107 | 0.293058 |

hip向优先候选负方向移动；knee仍向负方向移动，不支持“已学会正向knee展开”的说法。
这些是网络参数作用的固定输入证据，不是物理放置成功证明。conditional与local head
的变化不同，保留了既有单次HISTORY的实际数学关系，未将HISTORY抹掉。

## Reward / GAE给正向knee什么信号

“正向探索”按当拍raw sample减当拍conditional mean定义，而非绝对raw符号。

| 512个真实样本 | 高于当拍knee均值 | 低于当拍knee均值 |
|---|---:|---:|
| 数量 | 252 | 260 |
| 实测knee每decision变化均值 | +0.22295° | -0.21684° |
| gap减小均值 | +0.45596mm | -0.29217mm |
| raw GAE均值 | -7.15437 | -6.82036 |
| 归一化advantage均值 | -0.01711 | +0.01658 |
| mean-score × advantage均值 | -0.32601 | -0.04148 |

全512的knee mean-score × advantage均值为-0.18152。即较正向探索与当拍几何下降
存在关联，但本批未来回报/GAE没有形成正向膝均值的净训练推动；不可因为当前gap变小，
就重标负优势为正或声称PPO已学对。

第347号样本（0-based）在89.725s以INCOMPLETE_CONTROLLER_BLOCKED结束，
terminal_event=-40；该样本恰落在高于knee均值组，前序GAE也受后续结果影响。
这不证明该动作导致终止。241个AIR合法gap下降样本的平均优势仍为-0.04009。
所有512个决策端点均为AIR，没有新增TOP/承载保持样本。未据稀疏端点否认瞬时native接触，
但它们不能提供已记录的连续捕获成功学习信号。

knee最后派发的headroom裁剪375/512（73.24%），tanh绝对值≥.95为21/512。
不能把增加sigma等同增加有效控制探索。

## 真实LR与KL

实际LR从2.25e-5降至1e-5；全局KL均值0.0187674，clip fraction=0.2234375。
RR两通道占KL 9.20%，其他10通道占90.80%；其他10的均值项KL=0.0128533。
这支持“全局KL受到其他通道显著驱动”的事实，不证明分组优化或重参数化一定更好。
本次没有改分布、Adam、LR算法、reward或进行AUX。

## Pending tracking缺陷是另一个独立贡献

全部512样本P09 late648仍pending，逻辑N的RR始终[-6.9°, -37.8°]。
其中53个决策末端还宣告RR tracking，459个末端tracking已结束。
第一episode source stop后mappedN RR knee=-44.05°，即比源N负6.25°；
第二episode stop后为-47.8°，即负10°，后续维持。
既有源事件对齐审查的旧pending补偿为-1.25°；第一episode额外负5°符合错误继承tracking。
第二episode差异说明还受真实闭环反馈影响，不能机械将所有差额归为同一个固定bug。

无tracking的459末端仍继承负向mappedN knee偏置，均值偏差-7.36928°。
因此必须区分源tracking/历史补偿与policy作用；本批controller bias为0。
此事实支持最小owner继承修复，不支持把后续所有回弹都归因于早先切换。
源owner证据沿用v2_episode1_tracking_owner_followup.json，未重新跑参考。

## 可复现操作

标准库测试：test_rr_learning_signal_stdlib.py（12通过），
test_rr_learning_signal_update5_stdlib.py（3通过）。

分析命令仅在确认Isaac退出且没有formal video进程时运行：

    C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe outputs\ppo_rr_capture_first_cp225280_v1\analyze_rr_learning_signal_update5.py --isaac-stopped --output <NEW_JSON_PATH>

已生成RR_learning_signal_update5.json，不覆盖结果、不再次加载模型。


# 首块 update1420：真实FL捕获与保持信用

只核验已封存 rollout1420：global186113–186240，P05 **94**条/P06 **34**条；所有`done=False`。读取前384条日志仅为这128条提供真实前态；未读取后续活跃尾、未新增采样/optimizer/仿真、未改reward或生产。详见同名JSON及`audit_update1420_FL_credit.py`。

结论：**新的rolling-retention确实进入了真实reward，但这次更新尚不能称为吸收了稳定FL保持。** 第一拍掉载产生显著负reward，然而该rollout的掉载段normalized A仍全部为正。FL hip网络均值只小幅负移，相关样本的概率改变很大一部分来自sigma及其它通道；不能以联合likelihood提高代替FL动作学会。

## 真实事件、Phi与advantage

首次qualified placed事件为物理tick **2793**；其所属决策global **186206**在tick2800结束，P05→P06。当时FL真实TOP支撑，bearing4.35024N，gap−0.478mm，前缘距离+86.364mm。前一条186205在tick2792已经触顶支撑，但尚未满足连续TOP样本的历史placed条件；不能把两者合并成同一拍。

| global/tick | 事件 | reward | Phi前→后 | raw GAE | normalized A |
|---|---|---:|---|---:|---:|
| 186205/2792 | 已触顶、未qualified placed | +.106425 | .415765→.437974 | −.545362 | +.299911 |
| 186206/2800 | qualified capture，P05→P06 | +.162353 | .437974→.471418 | −1.083124 | −.526786 |
| 186207/2808 | 首个P06，FL支撑 | −.004830 | .471418→.471426 | +.359751 | +1.691336 |
| 186213/2856 | 仍支撑 | −.004805 | — | +.165880 | +1.393300 |
| 186214/2864 | 首次P06掉载，gap+.832mm | **−.160355** | **.477545→.446410** | **+.195881** | **+1.439420** |
| 186215/2872 | 仍AIR，gap+3.889mm | −.041423 | .446410→.439051 | +.293354 | +1.589265 |
| 186240/3072 | rollout末仍AIR，gap+16.438mm | −.002322 | .427943→.428387 | +.003950 | +1.144366 |

P06前7条支撑，首次掉载至本rollout尾27条均无FL支撑。后27条23个即时reward为负，raw GAE26正/1负，**normalized A27条全正**。P05的94条raw GAE全负，normalized A11正/83负。保存returns−values与实际adv标准化复算逐值相同；普通phase过渡没有done。

这不是即时reward没有生效或凭此即可认定GAE有bug：掉载拍的真实TD residual约−.094105，后续GAE递推项约+.289985，相加为+.195881。capture拍的value从−19.868677到下一拍−21.502028，TD residual−1.438744，加后续项+.355619后为−1.083124。保留这些当前critic/尾bootstrap引起的信用事实，不把“A正/负”改写为物理成功/失败标签。

## rolling-retention是否真的作用

对同一日志物理状态，调用生产的纯retention函数，另以**仅去掉rolling-retention项**的旧retention计算作算术对照；不推进状态、不重算critic/GAE、不生成替代轨迹。

首次掉载时FL原retention约`.999818→.391475`，不带新项的旧retention仍为`1→1`；FR保持1。FL原有capture份额对应Phi变化由−.000007726变为−.025862297。按真实`potential_weight=5、gamma=.9985`，该拍新项贡献 **−.129078884**；总reward−.160355076中，其余项约−.031276192。因此新项不是空配置，也没有再发一个捕获bonus；它占这次即时负reward幅度约80.5%。body几何质量在这两个关键拍均为0，不是掉载惩罚来源。

后27条新retention的算术reward差均值−.006975，累计约−.188333；部分靠近顶面恢复时为正，这是potential恢复，不是永久逐拍接触罚款。此差仅是在**相同已发生物理状态**上的项分解。移除该项还会改变globalPhi观测和未来policy输入，不能把“总reward减去此差”称为真实反事实轨迹或GAE。

## FL hip/knee：分别看实际请求、梯度与概率

| 样本 | FL network mean hip/knee | HISTORY hip/knee | conditional mean hip/knee | 实际REQUEST hip/knee（°） |
|---|---|---|---|---|
| capture186206 | +.005661 / −.087773 | −.167425 / −.184806 | −.150116 / −.175103 | **−2.765634 / −7.117582** |
| 首掉载186214 | +.018745 / −.101100 | −.018403 / −.126064 | −.014688 / −.123567 | **+.495667 / −4.489661** |

capture时FL hip确实给出了负REQUEST，但network hip mean仍略正，负conditional mean主要继承自真实负HISTORY，不能直接说网络已经学会负hip均值。首掉载拍的状态判定仍为动作前rolling，下一拍才转recovery；这是pre-action observation语义，不是漏切。

实际head梯度来自本次official loss的真实autograd hook。对全128×5次使用，mean分支与`−0.1*A*ratio/32*(raw−mu)/sigma²`（strict clip时为0）复算最大差 **1.518e−7**；不是离线伪造的目标梯度。

- capture186206：A−.526786，5次均非strict-clipped；最后一次hip/knee mean输出梯度 **−.003551 / −.054779**。就这条样本的局部梯度而言，梯度下降推向更正mean，不能称为正向强化了负落脚动作。最后联合ratio1.185323（logp+.170015），FL hip/knee单独logp **+.062073 / +.003469**，其余10通道+.104474。
- capture的FL hip logp+.062073中，先只替换mean的代数贡献仅 **+.000325**，再替换sigma约 **+.061748**。这里确实主要是缩sigma提高近mean样本的密度，**仅限这条样本/这次更新**；不能扩展为所有学习都是sigma吸收。
- 首掉载186214：A+1.439420；第一次实际hip mean输出梯度−.223981，与当拍raw hip比mean更正一致。5次中1次strict-clipped，其余mean梯度非零。最后联合ratio反而.959498，FL hip/knee单独logp **−.041880 / +.020155**；不能用正A或联合概率来保证每个通道的实际更新方向。
- 首个支撑P06样本186207最后联合logp+.283821，FL hip/knee却分别−.026203/−.039841，其余10通道+.349865，再次说明joint likelihood不能代表FL吸收。

这些是“某样本、某次minibatch、优化前”的实际输出梯度和密度变化；共享网络、其它样本、entropy、参数裁剪及完整Adam共同决定最终权重，不能把局部梯度直接当独立Adam参数变化。

## 同一保存观测：CP186112→CP186240

只做CPU functional forward，保持同一数值372输入和HISTORY，无采样/梯度/optimizer，RNG不变。初CP conditional mean与collection最大误差在1e−6内。不是自然确定性入口，也不跨版本宣称同一物理态编码相同。

| 真实观测组 | 平均FL hip network Δ | 平均FL hip conditional Δ | 平均FL knee conditional Δ | hip sigma末/初 | knee sigma末/初 |
|---|---:|---:|---:|---:|---:|
| P05 94条 | −.002045 | −.0002045 | −.0021629 | .93954 | .98494 |
| P06支撑7条 | −.001564 | −.0001564 | −.0019802 | .94075 | .98349 |
| P06掉载27条 | −.002118 | −.0002118 | −.0021315 | .93967 | .98417 |

capture观测上，末CP相较初CP的`cap*tanh(mean)`请求差仅 **FL hip−.002432°、knee−.043609°**；首掉载观测约 **−.003927° / −.066927°**。这是条件均值映射，不是机器人实际位移。相比hip微小均值变化，learned sigma约收缩6%更明显；本次不能宣布已把有利的hip方向可靠学进mean。

当前可交付的事实是：真实capture已出现，新retention对真实掉载有明显reward响应，真实PPO梯度正常；**稳定保持/全程越障仍未由此rollout证明**。继续当前混合课程和真实学习，不据单个update热改reward、rho或LR，不新增训练门禁。

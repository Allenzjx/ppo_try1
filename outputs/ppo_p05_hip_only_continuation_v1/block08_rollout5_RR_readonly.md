# Block08 第五个 rollout：RR 首次越沿后的落脚信用

仅检查封存 update1637 / decisions213889–214016 的128条实际输入、前一物理 endpoint、对应 likelihood，以及 CP213888→CP214016；不扫描其他 rollout，不拟合，不改生产、奖励或σ，不运行物理。当前仍在训练，本文不替代完整 episode 结果或后续自然 P01 评估。

## 物理覆盖与 retirement 实际消费

- 128输入全部 P09；episode时间54.0667–62.5333秒。首次RR越沿在213924 / 56.4秒，此时gap24.968mm、front+13.131mm。128 endpoints均有qualified lift，93已cross，真实TOP与placed均0。
- receiver retirement实际启用：92个 learner inputs / 93 endpoints；其中90 inputs / 91 endpoints改变Φ（receiver原本已为1的两点无差值）。末端FL receiver workspace proxy仅0.074016，但当前RR preparation receiver半份按规则保留为1；相对关闭此支路的同状态纯重算，末端Φ多0.0098386，本128步shaping合计多0.0445873。不是只命中predicate而未消费，也不代表FL空中承载。
- 连续postcross92步，gap范围36.532–67.918mm，末端51.593mm；较crossing时净增26.626mm，未完成放置。38个下降步平均−2.930mm；54个上升步平均+2.555mm。

## 奖励与优势：有落脚导向，但这批远距信用很弱

加权body/geometry/contact/smoothness/regularization成本在全部128步均为0，**没有证据以quality竞争为理由改reward**。总reward−0.635034 = potential shaping−0.464368 + time−0.170667，无terminal event。

| 当前合格postcross子集 | 下降38步 | 上升54步 |
| --- | ---: | ---: |
| capture shaping合计 | +0.0489565 | −0.0721745 |
| 总reward合计 | −0.112504 | −0.483320 |
| raw GAE均值 | +0.597584 | +0.598073 |
| normalized advantage均值 | −0.474668 | −0.473360 |
| normalized advantage为正 | 11/38 | 20/54 |
| 最后一次minibatch步前logp相对旧策略增加 | 13/38 | 21/54 |

下降有即时几何shaping优势，但这批后续GAE对下降/上升几乎不区分。所有128个rawGAE均正（均值0.775382、样本std0.374575），标准化后67正61负，这是正常相对优势，不是实现bug。crossing那拍reward+0.0786605、rawGAE+1.551317、normalized+2.071506，关键越沿事件有明显正信用。

当前RR AIR且xy内的capture对全局Φ贡献为 `0.02125*0.025/(0.025+abs(gap_m))`。保持其他量不变，gap50/66mm下降1mm分别仅贡献未折扣reward+0.0004786/+0.0003243；一拍时间成本−0.001333，当前总Φ折扣项约−0.0046。该远距曲线非零但明显弱；负即时总reward不等于下降被错罚，折扣势函数项不是独立稳定性成本，也不能只要求每拍reward转正。

38下降步capture shaping+0.048957、time−0.050667；其余未折扣物理进展+0.064110、总Φ折扣项−0.176896，共得总reward−0.112504。92个postcross样本的平均GAE分解：capture−0.001615、非capture shaping−0.163735、time−0.045009、critic TD+0.808221，合计约+0.597871。这里没有终止/真实完整回报，不能据此断言critic错误或reward反向。非终止尾bootstrap由已存return代数反推；FP64分解与原FP32 GAE最大差2.274e−5。

## 此次实际 PPO 更新的同输入均值变化

以下全部使用相同128个真实389输入/HISTORY，对比紧邻两端checkpoint。REQUEST定义仅 `当前cap*tanh(conditional μ)`，不是最终mapper/滤波/限位后的执行量，也不是Gaussian经tanh的期望值。

| 通道 | raw μ平均Δ | REQUEST平均Δ | 下降步REQUEST平均Δ | 上升步REQUEST平均Δ |
| --- | ---: | ---: | ---: | ---: |
| RR hip | −0.00044718 | −0.006435° | −0.007116° | −0.007584° |
| RR knee | −0.00011897 | −0.003819° | −0.003165° | −0.002163° |
| FL hip | +0.00105465 | +0.030019° | +0.029199° | +0.029132° |
| FL knee | +0.00297927 | +0.034840° | +0.032172° | +0.034584° |
| RL hip | −0.00049497 | −0.009355° | −0.009457° | −0.009914° |
| FR knee | +0.00115107 | +0.103274° | +0.108743° | +0.116808° |
| FL wheel | +0.00144343 | +0.00079519 rad/s | +0.00078312 | +0.00090629 |
| FR wheel | +0.00161578 | +0.00190094 rad/s | +0.00191271 | +0.00186308 |
| RL wheel | −0.00001481 | −0.00001316 rad/s | −0.00003972 | −0.00005662 |
| RR wheel | +0.00084156 | +0.00047074 rad/s | +0.00048908 | +0.00049594 |

RR hip raw μ原均值0.772132，更新后0.771685；RR knee原−0.320151，后−0.320270。其σ平均分别0.232774→0.228872、0.022646→0.022819。FL/FR/RL/RR wheel σ平均Δ分别−0.0164554/−0.0004720/−0.0019164/−0.0009811。完整分布、raw创新、实际loss梯度和逐组数据见同名JSON。不能把这些关节符号直接解释为轮端上升/下降，更不能把单样本loss梯度之和当作Adam后的唯一原因。

实际每样本使用5次；本更新20 Adam steps、LR1e−5、KL均值0.018906、clip fraction0.317188，actor确实变化。CPU重算旧μ最大绝对差1.19e−7；σ绝对差1.25e−6、相对差2.20e−6，符合CPU/CUDA浮点重算误差。

**有限结论：当前实现确实收到首次RR越沿样本并消费retirement；没有quality罚项竞争。RR远距capture即时信号存在但弱，实际单次更新RR REQUEST变动很小，且下降/上升没有形成明显GAE分离。先完成正在进行的本块和自然P01重载评估；该单个非终止rollout不足以独立授权调reward、σ、硬限或宣布学会落脚。**

复查脚本：`review_block08_rr_rollout5.py`；完整结果：`block08_rollout5_RR_readonly.json`。全部CPU进程已退出；审查新增PPO/AUX/physics credit均0。

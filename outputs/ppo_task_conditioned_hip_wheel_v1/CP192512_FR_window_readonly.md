# CP192512 det：已完成FR窗口的临时对照

仅分析活动视频`78f9ffd75d504f0e8447f44e61d134ce`已完整写出的0→1738（FR qualified24/cross1729/placed1738），**不是整次评估结果**。本次最终视频尚未封存，未来任务结果=null。复用此前当前B的0→1502及CP189952 det的0→1796相同物理窗口指标，不重扫旧run。

|物理窗口指标|当前B|CP189952 det|CP192512 det|
|---|---:|---:|---:|
|FR捕获用时 s|12.5167|14.9667|14.4833|
|roll /pitch角RMS rad|.231919 /.183521|.202405 /.176131|.201981 /.173810|
|roll/pitch-rate RMS rad/s|.117889|.092906|.098530|
|峰值合成倾角 rad|.309464|.285318|.285290|
|body collider最低z：最小 /均值 mm|91.272 /94.389|91.577 /94.052|93.161 /96.113|
|body collider最低z：捕获端点 mm|106.170|99.338|103.405|
|FR首次AIR gap≥15mm至越沿前平均gap mm|98.762|88.925|89.006|
|同窗FR最大gap mm|102.132|93.637|94.723|
|实测base净前送 mm|179.136|166.858|170.891|

相对上一CP189952：新FR快0.4833 s，机身几何最低/平均/捕获端点分别高1.584/2.060/4.067 mm，FR该窗口平均gap近似不变（+0.081 mm）；因此这次相对旧CP的改善不是靠更慢或更低的机身。但是**角速率RMS反而高6.05%**，峰值倾角基本不变，不应笼统说所有稳定性指标都更好。

相对B：新FR仍慢1.9667 s，rate RMS低16.42%、峰值倾角低7.81%，但FR平均gap少9.757 mm，捕获端点body collider低2.765 mm、机身净前送少8.245 mm；尽管最低/平均body几何高约1.9/1.7 mm，也不能称无代价、全窗口或全任务优于B。较低rate与较长用时共现，未用固定速度因果试验将两者分离。

RL/RR hip的整个事件窗均值（最终target /实测actual，deg）：

|关节|当前B|CP189952 det|CP192512 det|
|---|---:|---:|---:|
|RL hip|38.063 /37.901|36.291 /36.001|35.484 /35.213|
|RR hip|0 /.647|3.272 /4.372|1.769 /3.016|

新CP的RL/RR实际hip均值较旧CP分别下降.789/1.357°；RR实测峰值3.507°（旧4.845°，B1.890°）。这是完整原始关节读回，不是从target推算，不能将全身结果单因果归给两个hip。髋安装点世界高度未由此工具读取，维持null，不从body/CoM补造。

三者物理seed4001，本次首个真实policy request确认`deterministic_conditional_mean`。新runtime完整hash与旧版本不同：仅记录到training/cli/migration及execution_profile四文件变化；git diff确认execution_profile仅full_episode训练数量100000→131072。其余受登记的物理/控制实现及动作、观测、奖励、阶段配置hash一致，不伪称完整runtime hash相同。

活动文件只绑定实际消费的完整前缀：physical_observations截至1738共1739行、43,515,374 bytes；stage_transition_evidence截至1738共7行、23,712 bytes。JSON分别保存prefix SHA256，**不是仍增长文件的全文件封存hash**。角rate用wrapped相邻120Hz导数的时间均方根，没有再除以时长；角RMS为梯形时间均方根。collider z不是base z，AABB下界不是精确mesh净空。未读取后续未完成窗口、未改活动文件/编码/物理。

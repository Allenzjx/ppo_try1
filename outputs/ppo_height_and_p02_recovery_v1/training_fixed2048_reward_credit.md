# 固定配置2048块：P02终止覆盖与真实信用

已封存自然P01训练：CP172544→174592，2048决策 / 16次更新（1314–1329）/ 320 optimizer steps。只读16条优势审计、16条完成更新、4条completed episode、薄receipt，以及最后一条决策（有界尾部读取）；未扫153MB决策全流、未加载模型、未读取运行中的C174592。

## 覆盖结论

**P02有726真实样本、覆盖9次更新，但P02 terminal=0。** 本块未覆盖正式C172544的“P02 RR hardlimit”终止类型，不能把它说成已增加该mean失败轨迹的直接终止信用。非终止采样是否接近那条失败轨迹的状态邻域，现有薄统计无法确定，不能反过来断言从未接近。

五个自然P01入口均记录FR crossing/placement：四个完整episode加一个373决策的非终止尾。它们不是五个完整任务成功。

| Episode | 决策数 | FR Q / C / P tick | 终态 | 末区间实际ticks |
|---|---:|---|---|---:|
| 0 | 247 | 19 / 1202 / 1254 | P05 FALL | 7 |
| 1 | 608 | 18 / 1006 / 1046 | P05 BLOCKED，task deadline | 8 |
| 2 | 619 | 31 / 1195 / 1234 | P05 BLOCKED，task deadline | 4 |
| 3 | 201 | 17 / 997 / 1048 | P05 FALL | 1 |
| 4（预算尾，未终止） | 373 | 18 / 1538 / 1572 | P05，tick2984，bootstrap=true | 非terminal |

固定τ=.5的上一1024块曾有1个P02 hardlimit terminal；本块没有再次出现。随机轨迹通过FR与确定性均值恢复不是同一证据，不能因此宣布温度有效或失效。

## P02优势与真实终止

726个P02样本：raw GAE均值−0.236522，378正/348负，范围[−6.701344,.946136]；stored normalized advantage均值+.105092，460正/266负。reward均值−.000205（307正/419负），old value均值−18.153271，return均值−18.389793，均无非有限数。

标准化针对每个完整128样本rollout，不是按phase。例如update1315里P02的28个raw GAE全负、均值−4.460088，但28个标准化优势全正、均值+1.020893，因为该rollout包含更差的其他阶段。这与现有规则相符，不是负信用被误写的证据。按summary复算仿射标准化最大差3.30e−7；整体stored均值接近0，population std约.996086，符合sample-std标准化。

四个真实terminal全部P05，event=−40、next potential=0、bootstrap=false、timeouts=false。stored reward分别−41.328667/−41.065910/−41.297939/−41.585560，均等于return；raw GAE分别−18.762339/−21.096102/−20.001476/−19.758064，标准化优势也全负。float32 reward−old value与记录raw GAE一致；reset后的下一episode没有接入这些终止步的回报。真实末区间7/8/4/1tick，未伪称全部8tick。

20次普通阶段切换（P01→02、02→03、03→04、04→05各5次）全部terminal=false、bootstrap=true。16个rollout尾都非终止；按现有代码使用官方last value续接，数值未独立记录，未额外critic forward。

## 首拍均值核对与解释边界

引用根代理的 `checkpoint172544_first_mean_train_vs_formal.json`：同CP的训练首拍recorded conditional mean与正式C首拍deterministic raw在12通道完全相等，最大差0；实际训练sampled raw立即不同。这排除了“首输出均值加载不同”这一具体解释，**不证明全部372输入相同，也不证明后续HISTORY、身体状态或更新后权重相同**。

当前较窄结论是：本块随机FR capture覆盖充分可见，但仍未采到旧正式mean的P02硬限终止结果；不能把随机动作的return归给未执行的mean，也不能由GAE符号推某个wheel的单一原因。先等C174592正式封存结果，本审计不提出新reward/HISTORY/ρ/τ/physics/mask变更。

精确run路径、逐update P02统计和终止一致性见同名JSON。阶段预算与CP保存链复用checkpoint_entry的薄receipt，不重复建立新验收门禁。

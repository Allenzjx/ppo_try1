# CP182528确定性P05失败：终止、控制链与课程入口

封存source：`20260918T0751563353604Z_g3a50657a96c9_9132b92ae22044568b20624cf72afc7f/source`。自然P01、deterministic conditional mean；FR placed1556，FL crossed2586，但FL始终未placed。末tick6036 / **50.3s**，INCOMPLETE_CONTROLLER_BLOCKED，不是完整成功。

## 为什么在50.3s终止

P05从tick1568 / 13.066667s开始，末阶段年龄37.233333s。实际terminal detail为LOCAL_BOUNDED_RECOVERY_EXHAUSTED：当拍completion_values.placed_FL=.85，局部时限为 **30 + min(10,30×.5)×.85² = 37.225s**，在120Hz首个满足超限的物理拍终止。`.85=.35×lift + .35×crossing + .30×(.5×top_geometry)`，并不表示FL已接触；没有top样本，placed未成立。stall_diagnostic=true，但spec的stall terminates=false，不是stall直接杀死任务。不能沿用旧CP180480的34.9s阶段预算或48.3667s总时长。代码：semantic_supervisor.py:1091、1358–1385；stage_task_spec.yaml的local_timeout_policy及stall_diagnostic。

P05源关节端点为tick2737 / 22.808333s：首dispatch1569加1168源ticks（9.733333s）。这是代码与记录的nominal servo8共同推导，**不是独立已记录的P05 endpoint flag**。端点后全部nominal servo8与源末端精确一致；轮子仍可能按合法pending-capture前送规则运行，不能由servo末端推断wheel同拍全停。

## 端点后仍有动作表达和实测响应

以下为末tick6036同拍canonical度；actual为步进后实测，mapper反馈测量是步进前，二者未混用。

| FL通道 | base μ | conditional/raw | N | mapped N | filtered REQUEST=effective | final | actual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hip | .235219 | .236316 | 22.800000 | 22.212065 | +4.176237 | 26.388302 | 26.857991 |
| knee | .024800 | .024138 | −13.400000 | −12.150000 | +.579196 | −11.570804 | −11.577368 |

端点后3300个物理样本：FL REQUEST与headroom有效残差差0，FL无headroom clipping，真实joint索引/派发校验全部通过；最终slew在194拍起作用，其余也不是仅凭mask判定。FL始终AIR、障碍力0N，gap最小16.147mm、均值19.530mm；末gap19.282771mm/front100.035432mm。

最后6s：hip/knee REQUEST均值+4.169973°/+.582724°，实际跟踪final的RMS误差 **.600767°/.004963°**；FL gap范围19.283–20.521mm，仍无接触。mapper hip在22.212065/23.462065°间变化，存在约1.25°反馈目标波动；这不是“所有补偿都零”或“永无部分反馈交互”。但持续残差确实进入最终目标并有实测响应，不支持“FL动作被完全丢失/屏蔽”的解释。源端点附近曾有短暂跟踪滞后：tick2737 hip target26.9555°、actual32.9430°，不能用最后6s的较小误差覆盖该瞬态。

末拍全身：FR TOP/support，bearing12.25910N；RL/RR分别地面support、14.59385/1.86842N，FL不承载。端点后FR均为障碍接触，RL均地面，RR3298/3300拍地面、2拍AIR。机身z在.092253–.093668m，最后6s线速度均值.026810m/s、角速度均值.071254rad/s；不是低高度FALL或数值爆炸。其他腿末final/actual（hip,knee）为FR(.053528,46.288193)/(1.127927,46.686603)，RL(6.024082,2.470306)/(6.977750,1.707492)，RR(5.383069,−4.403269)/(5.475271,−4.123665)。这是全身闭环状态，不能凭FL hip修正正号或旧−2°探针孤立归因。

结论：**当前确定性策略在有表达/派发能力且实际跟踪的条件下维持了未触地构型，任务目标未完成**。没有证据把它称为底层mask问题，也未证明单一关节或mapper反馈就是全部原因；P06没有进入，本轮失败不能归因于P05→P06首拍cap转换。

## 下一次真实prefix：建议P05 / checkpoint_policy / offset200

当前完整确定性记录P05起tick1568，三个合法候选如下。均FL crossed=true、placed=false、AIR/support=false、0N、within_top_xy/within_lateral_span=true。

| Offset decisions | 真实tick / 秒 | P05年龄s | FL gap mm | front mm |
| --- | --- | ---: | ---: | ---: |
| 150 | 2768 / 23.066667 | 10.000000 | 18.245510 | 64.992014 |
| **200** | **3168 / 26.4** | **13.333333** | **19.743285** | **100.402359** |
| 240 | 3488 / 29.066667 | 16.000000 | 19.246189 | 100.400511 |

必须纠正预期：200/240都比150更晚，**并不更接近源端点时间，也没有更小的垂直gap**。选200的理由是已到较成熟的端点后平台、XY余量充分，仍未接触，且比240少耗2.667s前缀与局部任务预算；240仅少约.497mm gap，不足以证明更好入口。不能把约20mm悬空叫即将必然接触。

此建议仅来自当前固定CP的真实完整评估状态；**seed1001及下一权重的checkpoint-policy prefix尚未运行/接受**。实际执行仍需记录accepted/miss、交接tick、未placed/XY和继承状态；不能预支receipt、改成N前缀或从placed快照偷换课程。

数据：[CP182528_P05_diagnosis.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/CP182528_P05_diagnosis.json)。复用既有分层解析脚本得到的小型同拍摘录与窗口统计；额外模型前向/优化/仿真=0，未修改生产、reward、DELIVERY或RECOVERY，也未读取活动随机评估。

## 最后6s的真实频谱与局部相位（不外推全episode）

仅tick5317–6036，共720个均匀120Hz物理样本，去均值、Hann窗单边periodogram，频率分辨率1/6Hz。policy另外使用89个**原生**均匀15Hz完整决策样本（start5320–6024，REQUEST取对应end5328–6032；末半决策排除），没有插值或上采样。表中“基波幅”是`2|FFT|/sum(Hann)`，不是方波总幅度。

| 信号 | 原生采样 | 观测主频 | 主频正弦幅 / 全窗峰峰值 |
| --- | --- | --- | --- |
| FL hip N | 120Hz | 常数22.8°，无主频 | 0 / 0° |
| mapped N − N（mapper补偿） | 120Hz | **15.000Hz** | .816602° / 1.250000° |
| filtered REQUEST实际执行序列 | 120Hz | 最低频格.166667Hz | .003018° / .017619° |
| conditional mean（raw单位） | 15Hz | 最低频格.168539Hz | .000177 / .001034 |
| filtered REQUEST决策末原样本 | 15Hz | 最低频格.168539Hz | .003020° / .017619° |
| 最终目标 | 120Hz | **15.000Hz** | .816537° / 1.267619° |
| actual q | 120Hz | **15.000Hz** | .087279° / .217480° |

低频格仅覆盖约一个周期，故应描述为小幅慢漂移，**不能确认存在稳定的.168Hz周期振荡**。15Hz policy记录的Nyquist为7.5Hz，不能从它推出更高频率；实际120Hz REQUEST序列是执行日志，不是上采样均值。mapper/final的15Hz结论来自真实记录的PSD主峰，而非把30Hz反馈采样或15Hz决策频率直接当成振荡频率。

mapper/final的15Hz邻近三频格分别含85.355%/85.359%的去DC窗功率；actual q为97.230%。机身world角速度x/y/z也在15Hz有主峰，幅分别.001983/.003278/.000835rad/s，但该峰邻域只占26.780%/46.539%/54.189%功率，不能把全部机身波动归结于这一峰。

用11个1s、50%重叠Hann窗计算15Hz cross-spectrum：actual q相对final相位 **−63.151°**，各窗范围−63.160°至−63.141°，该频率增益.106889、相干约1。按记录时序换算的表观相位滞后为11.695ms；**不是独立测出的纯延迟或完整开环传递函数**，相位有整周期歧义，且command为本步目标、actual为本步结束测量。body omega-x相对final相位约−129.60°，各窗−134.98°至−120.75°、相干.975，也只表示闭环同频关联，不能据此分离四腿/接触的因果贡献。

本窗口的可见高频角度分量主要已经出现在mapper补偿/最终请求中，而policy均值与filtered REQUEST变化很小，实测关节对15Hz目标有衰减跟随。因此证据支持优先理解该**执行反馈闭环周期**，不支持立即把新增CAPS/策略差分惩罚当作已证实的修复；这类惩罚不能直接证明会移除由mapper产生的周期。也不能由此排除policy通过姿态/载荷间接影响闭环，或推断其他阶段无需平滑。此节没有新增控制干预、调参或稳定性改善宣称。

小型统计：[CP182528_last6s_spectrum.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fl_capture_quality_v1/CP182528_last6s_spectrum.json)。仅CPU读取已封存源，额外模型前向/优化/物理运行均0。

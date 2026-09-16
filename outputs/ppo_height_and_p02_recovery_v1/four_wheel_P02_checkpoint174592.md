# C174592：同配置续训后的正式四轮结果

**812 tick / 6.766667 s，P02 RR knee HARD_JOINT_LIMIT。** RR最终目标−4.246801°、实际−60.022797°；FR Q=tick24，C/P未完成，body collision=false。已从P01实际重载评估，失败仍在，不是完整PPO成功。

C174592与C172544同4a58 runtime、同372/all12 HISTORY温度策略、同τ=.5，之间增加2048真实决策/16更新。两者以及同4a成功zero的实际初态q/qd/base/CoM/contact/geometry完全相等；new/old共同723tick的source N Full12完全相等，new/zero共同812tick也相等。后续物理状态不同，不能强配成同状态反事实。

## Tick400 原生/逻辑轮速

单位rad/s，顺序FL/FR/RL/RR。四轮N=mapped N均+.3、controller bias=0。Full12 indices与live joint ids均8/9/10/11，对应四个`*_ankle`，接触对象对应`*_wheel`；canonical向前为正，转native的符号为[−1,+1,−1,+1]。

| 通道 | raw | projected | 最终canonical target | 最终native target | 实测native qd | 实测canonical qd | ground normal N |
|---|---:|---:|---:|---:|---:|---:|---:|
| FL | −1.216746 | −.503215 | −.203215 | +.203215 | +.014348 | −.014348 | 12.3886 |
| FR | −.503217 | −.278786 | +.021214 | +.021214 | +.021147 | +.021147 | 0，AIR |
| RL | −.468976 | −.262423 | +.037577 | −.037577 | −.053483 | +.053483 | 2.6842 |
| RR | +1.150096 | +.490672 | +.790672 | +.790672 | +.517062 | +.517062 | 14.4494 |

native qd直接来自同tick的`robot.data.joint_vel`，不是目标或独立扭矩推测。新C的103个真实height采样点全部与canonical qd按符号转换精确一致；无height行的中间tick只标记推导native，不伪补采样。

| Tick400对照 | canonical final4 | canonical actual qd4 | RR target / actual ° | FR gap mm | base-origin / CoM高 mm |
|---|---|---|---:|---:|---:|
| C174592 | [−.20322,.02121,.03758,.79067] | [−.01435,.02115,.05348,.51706] | −4.284 / −22.691 | 78.435 | 70.975 / 155.220 |
| C172544 | [−.22423,.01380,.04067,.79809] | [−.04513,.01280,.04450,.69198] | −4.371 / −28.388 | 73.624 | 66.922 / 150.424 |
| same4a zero | [.3,.3,.3,.3] | [.24424,.29813,.34175,.31376] | 0 / −.607 | 100.780 | 72.502 / 164.641 |

此刻三者FR均AIR、承载0，FL/RL/RR是实测ground支撑。FR轮转动不能称承载推进。

## 抵消与执行链

新C tick41–812的772个正N tick中，前三轮raw全负、RR全正；FL最终目标负向756 tick，FR绝对目标<.05为644 tick，RL为620 tick，RR高于N为772 tick。**前三轮正向N被抵消、RR被增强的模式仍在。** .05只是描述性近停阈值，正常stop/差速允许；这不是“只有RR实际牵引”的证明。

812tick中all12 phase mask全开且actual native mapping/setter/counterfactual验证通过；sourceN与mapped wheelN差0，wheel bias0。计入P01→P02首tick继承hold，tanh×.6、每tick .015 residual slew的projected复算最大差1.11e−16；native final与N+residual组合差≤2.92e−8（float32）。四轮均按既有resolver/符号转换/全量velocity setter写入，没有发现N遗漏、错误mask对象、重复幅度或单位异常。ACK正确不等于协同合理。

source owner边界：既有P01 changed-channel层跨P02继续，P02 current-FR approach assist也可提供同一个完整+.3建议。当前决策日志的`source_partial_order.layers=[]`，没有唯一逐tick owner字段；因此仅确认完整N实际保持，不能把每tick+.3武断归到其中一个来源。源路径和已审计代码位置见JSON，不改成功zero。

raw每8tick保持，阶段切换沿用残差；现有源码回写实际raw HISTORY。正式日志没有完整372输入tensor，未额外forward或声称独立证明其全历史。

## RR与任务时序

相对zero，tick1 all12目标已不同；相对旧C，tick1四轮final仍相同，但伺服目标/接触/实际响应已经不同。new/old四轮final首次差异为tick5/2/2/5，不能只凭wheel目标解释全部首次分歧。

RR误差连续12tick超过1°的起点是26（旧27）；2°起46（旧43）；5°起159（旧63）。base原点下降3mm起40，FR gap峰97.696mm在tick140，持续比峰低5mm起262，CoM低于初态3mm起458。5°误差和部分下降较旧C推迟、tick400误差减小，是局部差异，**并没有恢复FR越沿/放置或避免最终硬限**。

新C RR全813物理行都有真实ground contact；FR自tick2为空中。终态FR gap−7.551mm、front-distance−171.007mm，未C/P；同tick812 zero的RR实际−.616°、FR gap98.964mm。zero原完整8857tick/P13成功保持不变。接触力和轮速已知，但没有独立瞬时驱动力矩或单轮净牵引因果分解，不能制造唯一底层原因。

本轮真实续训及正式评估已完成，结果如上；本报告不再提出新σ、reward、HISTORY、mask或physics变更。训练的随机FR捕获不能替代本次正式P02失败。全部分析只读，仅封存三run的有界前缀，未触碰生产。

# 59e首回合：443样本，P12有限恢复耗尽

只读已封存episode 0及前443条完整残差记录（global229633–230075）；未读取第二回合。run：`20260924T1121437392178Z_g59e868f3e223_63b8d42e71504d8197a9228aabc8625b`。

**终止原因：** tick9755 / 81.291667s，`INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。P12 age30.025s超过30.019656s有效限额（30s基础＋0.019656s当前进度余量）。物理判定器仍VALID且无独立安全／wheel-only失败。它是实际任务terminal，不能bootstrap，不是外部截断。

777个successful_nominal前缀decision零PPO credit，tick6216交接。443个learner端点全部P12。此检查时尚未完成512样本块，**新增完整更新计0**；补齐后的优化归封存报告，不在这里预支。

## 真实事件

- RR lift5427、cross/placed6133和RL lift6212来自前缀，不是learner新成果；RL6212在6247落地撤销。
- learner RL lift6302→6501落地撤销、6861→6885落地撤销；之后无新资格。当前资格端点31（继承3、新事件28）。无qualified edge recovery、合法capture区、TOP承重、cross或placed。
- RR在7756 / 64.633333s发生真实地面撤销；此后无合法TOP承重恢复。早段TOP恢复来自AIR／XY丢失，不是GROUND重捕获。历史placed保留，不等于当前可用支撑。
- 第一处RR/FL同时丢失TOP的learner端点6248；并非直到最终超时才掉载。

## 端点样本覆盖（可重叠，不当作连续物理步数）

| 当前证据 | RR | FL | RL | FR |
|---|---:|---:|---:|---:|
|合法TOP承重|86|84|0|420|
|AIR|38|358|36|23|
|GROUND|250|0|405|0|
|当前有效后腿资格|170|不适用|31|不适用|

其他接触／XY外状态不强行归为合法TOP。RR可落脚AIR准备30、实际RR承重前侧准备86。既有短窗“FR轴正投影＋RL载荷比例下降”24，只是投影代理，**不是已验证侧向FR转移或合格RL卸载**。

| 终点腿 | 实测接触／力 | gap(mm) | 前缘距离(mm) |
|---|---|---:|---:|
|RR|GROUND，5.696N|−49.995|−87.269|
|RL|GROUND，12.826N|−50.730|−174.665|
|FL|AIR，0N|+95.926|+87.106|
|FR|合法TOP，9.706N|−0.276|+385.373|

最后存储reward=−43.061016、old value=−10.295301；实际done=true。这次真实终止在新512的同一采集块内，仍须等完整512及官方优化审计后再比较raw GAE／归一化advantage。不能从该冻结actor回合宣称新512已产生学习收益。

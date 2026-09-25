# 最新59e封存512：膝关节真实REQUEST→headroom→FINAL多样性

范围仅 `20260924T1121437392178Z_g59e868f3e223_63b8d42e71504d8197a9228aabc8625b` 的完整512条learner记录，不是旧f6d首118样本。只读stdlib；不加载模型，不读取当前DET，不改生产。

**结论：FL knee存在明确的多请求→相同−58°目标现象；FR/RR knee本块没有相同的−58°／headroom饱和证据。** 当前RR落地窗口最明显：109种不同raw、109种filtered REQUEST最终都为−58°。不能把所有膝关节统称被限位，也不能把这109个结果单独归因owner。

## 裁剪与owner分开

这是15Hz决策末端实际记录的最后派发值／证据，不是完整4091个物理步的逐tick裁剪次数。RR GROUND/TOP由该决策末实测状态分组；TOP承重要求传感器、合法XY、当前接触与已有0.2N阈值，非历史placed。

| 窗口 | 通道 | n | same-tick headroom clipped | FINAL=−58° | 输入owner active | owner二次clip精确次数 |
|---|---|---:|---:|---:|---:|---|
|全部|FL knee|512|289|170|397|N/A，receipt未输出|
|全部|FR knee|512|0|0|不适用|不属owner通道|
|全部|RR knee|512|0|0|不适用|不属owner通道|
|RR GROUND|FL knee|250|195|109|250|N/A|
|RR GROUND|FR knee|250|0|0|不适用|不属owner通道|
|RR GROUND|RR knee|250|0|0|不适用|不属owner通道|
|RR合法TOP承重|FL knee|111|33|33|14|N/A|
|RR合法TOP承重|FR knee|111|0|0|不适用|不属owner通道|
|RR合法TOP承重|RR knee|111|0|0|不适用|不属owner通道|

`headroom clipped`直接取 `actuator_target_effect_audit.policy_headroom_evidence.clipped_servo_indices`。owner active只取policy当拍输入`rear_owner_observed_features[9]`，不是末tick精确投影receipt。训练记录的完整effect audit、简化tick列表均未序列化`rear_owner_recovery_evidence`（512条全部缺失）；不能把缺失当0，也不能把active当clip。生产接口会内部核验owner变换，但此次日志无法给其二次cap/headroom裁剪次数。FR knee(index3)、RR knee(index7)不在owner的(0,1,4,5)通道集合。

所有512条all12 mask为1，final dispatch verification通过，P12期间FL assist未拥有这三个目标通道。这些只排除相应接线问题，不等于动作合理或物理成功。

## 实际数值多样性

下表为不同数值个数（四舍五入到1e−6，完整精度亦在JSON）。raw是无单位Gaussian样本；REQUEST是当次实际滤波／限速后输入headroom的残差°；effective仍是**owner之前**的headroom有效残差°；FINAL是实际下发canonical目标°，不是实测关节角。

| 窗口 | 通道 | raw种数 | REQUEST种数 | effective种数 | FINAL种数 | FINAL范围(°) |
|---|---|---:|---:|---:|---:|---|
|全部|FL knee|512|481|203|295|−58.000…−16.338|
|全部|FR knee|512|419|419|419|−51.929…+82.807|
|全部|RR knee|512|512|512|512|−51.250…−29.037|
|RR GROUND|FL knee|250|248|54|140|−58.000…−33.874|
|RR GROUND|FR knee|250|250|250|250|−11.242…+25.492|
|RR GROUND|RR knee|250|250|250|250|−51.250…−46.026|
|RR合法TOP承重|FL knee|111|105|74|73|−58.000…−20.338|
|RR合法TOP承重|FR knee|111|86|86|86|−51.929…+78.807|
|RR合法TOP承重|RR knee|111|111|111|111|−51.099…−29.037|

RR GROUND中FL FINAL=−58°的109行：raw范围−2.979585…−1.504287；实际REQUEST −35.814571…−32.613119°（109种）；headroom effective唯一−26.6°；pre-owner candidate与FINAL均唯一−58°，109行都记录headroom clip，输入owner均active。这是多个实际raw/request被映射为相同最终目标的直接证据，但各行历史和状态不同，不是固定单状态的探针。

两个直接日志例：global229878/tick8184的raw−1.504287、REQUEST−32.613119°；global229979/tick8992的raw−2.979585、REQUEST−35.814571°；两者pre-owner effective−26.6°，candidate/FINAL−58°。

另一方面，全部289次FL headroom clip只有170次最终−58°；GROUND下195次只有109次最终−58°。在全部512行中FL pre-owner candidate与FINAL不同193次（GROUND141，合法TOP3）；这包含下游owner／最终slew的作用，缺receipt不能再精确拆分。故不能把pre-owner的headroom饱和次数直接当最终执行饱和，或把它们全部解释成owner冻结。

这里只读实际同拍字段，没有从独立重算nominal与FINAL相减推断policy作用。未新增sigma、reward、owner或限位修订；此次数据支持优先关注FL负端行为坍缩，同时保留FR/RR仍有目标多样性的事实。

# P02 四轮通路：确有策略抵消，不是 nominal 丢失

正式 C171520 的 tick41–693，source N 与 mapped N 始终完整保留四轮 `+.3 rad/s`。最终目标却被 residual 改成 **FL 多数反转、FR/RL 接近停转、RR 大幅正转**。这是目标层的 RR 主导，不是“all12 mask=1 所以没问题”，也不是已测得只有 RR 提供牵引。

tick400 的同一物理时钟数据如下；canonical 速度统一为向前正，native 符号按关节方向转换。

| 通道 / Full12 index / live joint_id | N | raw | tanh×.6 = projected | 最终 canonical 目标 | 实测 canonical qd | ground normal N |
|---|---:|---:|---:|---:|---:|---:|
| FL ankle / 8 / 8 | .30000 | −1.14184 | −.48902 | −.18902 | −.05643 | 11.6015 |
| FR ankle / 9 / 9 | .30000 | −.49572 | −.27525 | .02475 | .02486 | 0，AIR |
| RL ankle / 10 / 10 | .30000 | −.53515 | −.29359 | .00641 | −.01693 | 3.0047 |
| RR ankle / 11 / 11 | .30000 | 1.25857 | .51040 | .81040 | .68424 | 13.8777 |

653 个正 N tick 中：FL 最终目标负向 **644** tick；FR 绝对目标<.05 为 **525** tick；RL 为 **565** tick；RR **653/653** 都高于 nominal。这里 .05 只是描述性近停阈值，不是任务门禁。减速、正常 stop、差速本身均允许；当前证据显示的是持续偏置与失败共存。

## 从输入到最后写入

- `command_batch.py:28,57,166,198`：索引8–11对应 FL/FR/RL/RR ankle joint，contact body 对应 `*_wheel`；canonical→native signs为 `[-1,+1,-1,+1]`。本次 startup 实际 `robot.joint_names` 将其解析为 ids8–11，无模糊 alias。native audit未独立保留完整 ACK id数组，故 ids证据明确来自live名称+精确resolver。
- `semantic_backend.py:65` 与实际 execution profile：P01/P02每轮 residual cap=.6，覆盖初始化 .12；这是 raw 经 tanh 后的 residual 幅度，**不是乘到整个 N 的 mask/scale**。phase/runtime/safety mask合成作用于 residual。
- `action_projection.py:514`：tanh×scale→mask→120Hz residual slew；每tick最多 .015rad/s。`PhaseTransitionBridge:351` 保留跨阶段残差。tick16末 projected 为 `[-.13217,-.05101,-.05822,+.14781]`，tick17虽然换成新 raw，projected 仍保持上一值一tick，随后继续 slew；不是清零。计入此既有 hold 后，693tick逐通道复算与实际 projected 最大差 **2.22e−16**。
- `semantic_residual_adapter.py:112,157–172`：wheel nominal不经过servo mapper变化；最终为 `clamp(N+controller_bias+projected)`，随后四轮全量转换、`set_joint_velocity_target(...wheel_ids)`、一次`write_data_to_sim()`。本次 wheel controller bias=0；N与mapped N差=0；final native反转为canonical后与N+residual最大差 **2.92e−8 rad/s**（float32 dispatch）。693/693 actual mapping/setter/counterfactual验证通过。没有在这些证据里发现漏写、错误对象mask、符号或单位错配。
- `semantic_env.py:149,163` 每个决策raw保持8tick并回写实际raw HISTORY；C日志验证了693tick的决策内raw保持，但没有保存完整372输入，所以不能独立重测其 HISTORY tensor/head。轮raw对应原 HISTORY slice203:207，未新增隐藏历史。历史 M诊断则明确验证过实际masked raw/HISTORY。

## 时序和物理响应

P01末tick16、P02起tick17，N四轮都是0，策略已请求前三轮负向、RR正向。RR 1°误差持续段起点tick29，此时N仍0；tick41当前FR任务反馈才添加四轮+.3。故并非只有 nominal开始滚动后才出现RR误差。

tick41最终四轮目标 `[.03282,.18061,.16316,.60573]`；到tick63变为 `[−.02526,.15377,.12903,.67868]`，RR knee target/actual为−2.3629°/−7.3922°。这些早期点FR与RL均AIR，真正ground支持为FL/RR；不能将悬空FR/RL轮速当成承载推进。

tick400 FR仍AIR，FR top净空63.164mm，body原点高63.237mm，RR实际−31.0156°；tick693 FR净空−5.570mm，body原点48.206mm，RR目标−5.1739°/实际−60.0061°，P02 HARD_JOINT_LIMIT，FR C/P未完成。终态FL/RL/RR ground法向力约11.422/3.057/15.379N，FR为0。接触力和轮速已记录，但没有独立瞬时关节驱动力矩或各轮净牵引因果分解。

## 已有 wheel4-off 诊断的正确使用

复用历史 CP168192/3fc 的 M诊断，不重跑：1920tick四轮raw/projected/direct policy effect均0，source nominal非零1880tick，其他8伺服仍有作用。tick400四轮N/目标均+.3，实测qd `[.16110,.30048,.35927,.25957]`；RR target/actual−1.2995°/−1.9398°，FR净空94.924mm。其FR crossing1672、placement1686；到16秒为诊断窗口结束，不是完整任务成功。

它支持“去掉那次 wheel residual 后闭环行为明显不同”，但同时改了HISTORY与后续其他8通道，checkpoint/head也不同于当前C171520，不能冒充当前冻结策略的纯wheel因果隔离。当前较窄结论是：**三轮正向nominal确被策略明显抵消而RR得到增强，执行链在按该请求工作；协同是否合理仍不能由通道开启或ACK通过保证。** 不因此永久mask、不要求固定四轮同速、不更改reward/ρ/physics。

详细同步证据及原run路径见同目录 `four_wheel_P02_path.json`。全部分析只读，无新增仿真或生产改动。

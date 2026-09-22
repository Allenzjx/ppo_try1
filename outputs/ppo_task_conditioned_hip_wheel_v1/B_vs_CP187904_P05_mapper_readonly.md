# 同版 B 与 CP187904 det：P05 mapper 周期的有限只读对照

**B 已完成的 P05 没有重现 det 失败窗口的持续15Hz补偿周期。直接可见的区别是动作owner、接触和构型：B 捕获后退役 tracking，det 未捕获而继续 tracking。不能据此把周期判定为“所有zero状态的共性”，也不能把残差项本身确定为唯一致振原因。**

范围：同HEAD `ee5a9651591d` 的 B `video_eval/prior_B/20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source`，只读已完整写出的P05及紧接P06窗口（native t1512–2920、physical t2300–2920、截至t2728的少量policy和截至P06的transition）；未读取/依赖仍在执行的P13终态。对照为 `video_eval/validation/20260921T0627142354953Z_gee5a9651591d_91d8d06aaced49ebb0ad9d5a5c067b31/source`，复用已封存诊断，仅补读固定det t3000–3600的native记录。本报告中的B“成功”只指已确认的FL捕获/P05完成，不替代完整run结论。

## 实际时序与周期

B进入P05 t1512；FL越沿2468、本次所查物理窗口首次台面接触2676、placed2677、P06进入2680。全部已查B native full12 residual均严格0。角度为canonical servo °；N为真正mapper后baseline，不是离线nominal重算。

| 窗口 | FL hip source / mapped N ° | 实际eligible反馈数 | mapped N lag4 / lag8 RMS差 ° | FL实际接触 |
|---|---|---:|---:|---|
| B t2300–2648，349tick | 48.2 / 恒49.45 | 0 | 0 / 0 | 349tick均AIR |
| B t2649–2677，29tick | 45→24.9 / 48.2→22.4 | 3，单向下降 | 3.993 / 8.209，源下降不是周期 | 2676、2677两tick接触 |
| B t2678–2920，243tick | 24.9 / 恒22.4 | 0 | 0 / 0 | 243tick均台面接触，5.127–10.940N |
| det t3000–3600，601tick | 22.8 / 22.210518↔23.460518 | 150 | 1.25 / 精确0 | 全AIR，无FL放置 |

整个B P05（发往P05的t1513–2680）hip/knee eligible反馈仅10/6次；可见的是源waypoint转换中的有限追踪，没有det晚段150拍等幅来回。B长保持段hip compensation=+1.25°，**保持补偿值不等于仍在反馈更新**：该段scheduled=false。B落脚时t2673补偿0→−1.25、2677再到−2.5；两次都是向同一方向，不是振荡。

B t2677确认捕获，`capture_owner_hold`记录held nominal `[24.9,−13.4]`、retired P01–P05 layers；t2678 native明确 `ended_tracking=true, scheduled=false`。后续mapped hip保持22.4°，没有恢复旧source入口或清掉已生效的−2.5°补偿。此时actual hip31.002°逐步收敛到约23°，与有载静差同存；截至2920仍保持真实台面接触。B接触后实际knee范围[−12.247,−11.104]°。

det晚段仍 `scheduled=true / ended_tracking=false`，每4tick更新一次，8tick重复；actual hip有约.206°同频响应，knee约−18.8°且FL悬空。其source22.8°、knee残差约−6.66°、接触/载荷和完整历史均不同于B，因此这不是固定物理状态的单变量实验。不能把B捕获后退役tracking的事实泛化为“未捕获时也应该停止tracking”，更不能放宽捕获判据。

## 一次限定的离线算术敏感性检查

保持det真实记录的actual、source、旧补偿c不变，只在**各拍独立重算**时把feedback参考中的previous FL hip REQUEST设0；gain仍8、限幅10°、slew1.25°和所有输入记录均不变：

```text
desired = clip(8*recorded_e0, -10, 10)
one_step_c = recorded_c + clip(desired-recorded_c, -1.25, 1.25)
```

150次中150次仍朝真实补偿同方向变化；93次与真实新补偿一致，57次幅度变化，最大新补偿差.260667°，单步幅度仍.989333–1.25°。例如t3201去掉−.007462°的previous R后desired=.971353°，仍上跳1.25°；t3205去掉−.002924°后desired=−.675297°，仍下跳1.25°。

这只说明**对已经形成的记录轨迹，当拍极小hip residual参考项不是这些反向反馈步的唯一来源**。它不是把PPO移除后重跑的zero；保留了过去残差造成的状态、其余11通道、recorded c，并且每拍独立、不把反事实结果递推到下一拍。故不能据此宣称零残差闭环必然振荡/不振荡，或任何候选控制器能捕获成功。没有运行gain2、没有扫参、没有更新HISTORY。

## 对下一步的有限判断

证据支持将这项问题准确描述为“未捕获、持续tracking构型中的离散反馈极限环”，值得在后续资源允许时做**一次隔离、同入口的物理验证**；尚不支持认定单位/符号bug，也不支持立即替换生产控制器。值得测试的理由是实际target和actual均有确定周期，不是因为它已被证明导致P05失败。若将来验证，必须同时看真实接触/保持和后继行为，不以更小target波动替代任务完成。

当前成功zero/此次B的owner退役语义不修改；此前gain2仅为未采用提案，本次不选参数、不扩大门禁。根任务的正式训练可独立继续。本任务仅只读/CPU算术，新增真实decisions、updates和optimizer steps均0。

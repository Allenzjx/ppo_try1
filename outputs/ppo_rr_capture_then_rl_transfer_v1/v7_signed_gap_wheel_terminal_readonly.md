# v7：负 gap 与 wheel release 的实际时序

只读封存 source `validation/20260923T0934264919029Z_g60abc00957c0_ef405598c8954e6280e844c4c29ed041/source`。manifest确认 physical success=false；122.625s / episode tick14715，P09 `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。仅一次 CPU1 标准库尾读 physical/native/capture各241行（14475–14715，末2s），video最后32行，共约27.9MB。0新增physics/actor/PPO；helper已退出。完整逐拍值和读取尾段hash在同名JSON；没有读取全部历史。

## 首处分歧已证实，但不是整个失败的唯一因果证明

`semantic_rr_carry_wheel.py:build_rr_carry_wheel_context` 的 AIR 许可仍要求 `gap >= 0`。v7 assist/context接受有符号几何带，wheel子层却在过零时从 `forward_floor` 切至 `release_slew`，重新向负FL policy候选靠近。**envelope_active始终true，不代表仍在执行非负floor。** source零速stop、当前支撑与Q没有在该边界失效。

以下 tick 为 episode输出/物理后态；这一拍控制读取前一个 episode tick。native绝对dispatch另有+179初始化偏移：episode14601对应native14780，不能混称同一时钟。

| episode输出tick | 输入gap µm | 输出gap µm | wheel action | FL FINAL rad/s | FL实测 rad/s |
|---:|---:|---:|---|---:|---:|
|14599|+8.690|+2.611|forward_floor|+.000125789|+.0221593|
|14600|+2.611|−3.537|forward_floor|+.000037812|+.0228351|
|14601|−3.537|−8.149|release_slew|−.014962188|+.0120192|
|14602|−8.149|−10.720|release_slew|−.029962188|−.00195783|
|14604|−12.228|+1.746|release_slew|−.059962188|−.0234536|
|14605|+1.746|+14.537|forward_floor|−.044962188|−.0159715|
|14701|−2.866|+5.541|release_slew|−.149966396|−.0604862|
|14715|−5.769|−10.906|release_slew|−.044949089|+.0688590|

首个release时原source N四轮仍 `[0,0,0,0]`；policy REQUEST/candidate `[-.937242351,+.062755975,+.000027775,-.088525526]`，mask `[1,1,1,1]`。FL FINAL恰从前一FINAL按 −1.8/120=−.015 rad/s推进，不是某个遗漏mask、源stop新触发或native覆盖。source endpoint/committed ACK验证仍true、fresh/finite wheel owners均空、FL/FR/RL当前支持；原stop source tick864、当前sample5693。实际native wheel buffer `[+.014962188,+.062755972,−.000027775,−.088525526]`，与 canonical符号映射 `[-,+,-,+]` 一致，float32读回验证true；ID未持久化，保留null。

末2s共有15次gap过零、15次wheel action切换，51拍release；FL FINAL范围[−.149966,+.010092]，实测范围[−.064444,+.092427] rad/s。回正后也须按原slew返非负，所以 `forward_floor` 标签不意味着此拍FINAL已经非负。不能用最终target负号冒充该拍实际负转（终态实际仍正，见表）。

## 全身/捕获证据

- 末2s RR gap从+.743625mm到−.010906mm，最小−.024447mm；RR前缘深度22.9745→16.8737mm（−6.1008mm）。同窗body世界位移[−.80496,+.05735,−.12314]mm，massCOM位移[−.96192,+.11178,−.09298]mm；不可把全部RR退回归因于单轮，因为关节与身体仍连续变化。
- 终态RR FINAL hip/knee=[−1.785913,−12.361176]°，actual=[−2.114141,−12.293048]°，target−actual=[+.328228,−.068129]°。assist实际到53°累计travel、43s exposure后BLOCKED `finite_search_travel_or_margin`；没有额外预算或接触信用，RR placed仍false。
- 241拍RR current-Q均true，raw ground/obstacle `force_w_n`及normal均0、active false，无TOP或support。**另一个真实摩擦读回不为零**：`tangential_force_n=norm(friction_force_w_n)`（contact_classifier.py:181）自14609起103拍非零，末值/最大.225898N；末contact point=[.54025024,−.31676686,.05115822]m。它与零normal分别保留，不用它伪造TOP/bearing，也不能据此声称所有原始接触量为零或传感器损坏。
- 终态FL/FR verified TOP support分别3.161964/13.438872N，RL ground support11.855742N。RR无接触/放置，未进入P10或RL流程。
- 终态四轮 canonical candidate=[−.938946,+.062419,+.001321,−.089272]；FINAL=[−.044949,+.062419,+.001321,−.089272]；actual=[+.068859,+.029389,+.049255,−.098883] rad/s。native FINAL=[+.044949,+.062419,−.001321,−.089272]，最后setter为真实 `robot._joint_pos_target_sim/robot._joint_vel_target_sim_after_existing_write_data_to_sim`。

## 窄修订依据与验证边界

有充分数据支持修复 **已获准capture有符号带内、仍真实AIR的wheel许可不一致**：共用当前已冻结geometry band，不因微小负gap释放到负FL候选；gap<=0时前送gain仍0，但保持已有非负floor0及原1.8 slew。真正TOP/contact、GROUND/wall、越界、失Q/支持或fresh source stop仍按其原退出/让权规则，不能把负gap当TOP，不能加预算或延时。

最小反例：同一真实pre14600状态允许floor0而非release；正负微gap往返不切release；真正TOP及带外仍release/bypass；新stop仍胜出；同raw/logp/all12/previousFINAL限速不变。该修订只修已证的分支不一致，**尚未证明会形成RR normal contact、完成捕获或改善整体稳定性**。未来物理结果仍需独立核对。

## 原始摩擦读回的有限源码核对

`isaac_fsm_backend.py:2014–2027` 在唯一 `sim.step(render=False)` 后 readback/read；`sensing/sensor_reader.py:122–160` 每个body调用 `sensor.update(dt, force_recompute=True)`，随后从同一 `sensor.data` 按相同exact pair索引读取 force matrix、history、point、friction。当前调用链没有在normal与friction之间再次物理步进，也没有本地沿用上一拍friction的分支。

本机实际 `C:/robotics_sim/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py:364–414` 在同一个 `_update_buffers_impl` 内依次读取：`get_contact_force_matrix(dt)`、`get_contact_data(dt)` 的点/count/start、`get_friction_data(dt)` 的独立friction/count/start。friction按patch求和（`avg=False`），point求均值；unpack每次新建聚合数组，count=0明确写default0（479–492），没有该层“无新contact仍保留旧friction”的实现证据。

因此当前能证实的是 **normal/resultant pair `force_w_n` 为0，而独立friction patch聚合非零**；它们来自不同provider API/独立counts，不是把同一个向量同时说成零与非零。尾流没有保存两套原始count/start、独立provider时间戳或friction向量，所以不能判断底层patch寿命/采样是否错位，也不能判定数值就是陈旧；仅凭.2259N不足以确认提取损坏。该发现不构成新的训练门禁、承载认定或放宽TOP判据理由。

输出JSON SHA256：`e093978a84e4d901ccc52924c3d5a96e4aff5723a2a49f00f20912d8dde9f53e`。追加本节仅小文件文本核对；所有CPU helper已退出。

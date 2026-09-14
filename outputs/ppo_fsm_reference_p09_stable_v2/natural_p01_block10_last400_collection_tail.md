# Block10 末400条：第二次RR初始离地与非终止采样尾

只解析同run `train/20260910T1233487014760Z_g7db0d17f398d_9984d1c7ed954d8fb2769e551a23fc6e` 的 audit1649–2048及第7条completed，不重复分析此前1648条。固定global147953–148352；原始bytes119,413,909–147,205,652（含端点，27,791,744 bytes），SHA256 `1ff5c89dc82e5e657327a194788c4d754c36e8787b4f0416c3f8a71b95ff1ea7`。未读取活动video4。

| 范围 | 实际相位样本 | 原结果及前腿事件 |
|---|---|---|
| ep6：237条，global147953–148189 | P01=2/P02=196/P03=6/P04=13/P05=20 | P05 FALL，tick1892/15.766667 s；FR Q4/C1589/P1631，FL Q1820/C0/P0 |
| ep7：163条，global148190–148352 | P01=2/P02=115/P03=5/P04=1/P05=40 | **非终止collection tail**，P05 tick1304/10.866667 s；FR Q28/C935/P971，FL Q1051、1211/C0/P0 |

**第二次RR I定位：** ep6的 **P05 tick1822 / 15.183333 s / global148180**，`whole_body_initial_clearance` excursion3.103388 mm、全身实测关节运动19.045737°、命令运动20.006133°。其所在决策末tick1824为AIR、initial=true，但established/current_lift_valid均false；台面净空-48.118 mm、距前缘-541.459 mm。整个ep6 RR没有Q、C或P；终态已GROUND、initial=false。它是有测量证据的**初始离地而非成立抬升**，不能漏计，也不能说发生于P09或已有后腿成功。ep7 RR无任何I/Q/C/P。加上已报告ep0的I1/Q1，ep1–5皆零，整块恰为 **RR I2/Q1/C0/P0**；末400的RR current-valid决策末态数为0。

**ep6 FALL依据：** 终态372维policy/critic相同、finite_fallback=false；按既有固定schema及locked底面0回解base_z=**0.014902249 m<0.015 m**，gravity_z=-0.968845189，线/角速度模长0.090829/0.462861，未触发翻转或速度爆炸阈值，实际8个关节在各自硬限内。这里是实测float32观测回解，不声称读取了原始双精度传感流。保留低高度FALL；第一未完成任务为FL保持净空、越沿并放置：末态FL AIR/有效载荷0，距前缘-60.102 mm、台面净空-40.385 mm，历史Q不是C/P。

**ep7尾部不是成功或失败终态：** `terminal=false/reason=null`，physical VERIFIED、entry_valid=true；当前P05 age2.666667 s，只是2048决策预算结束。FL先Q1051、1124回地撤销、再Q1211；末态AIR/有效载荷0、距台面+2.211 mm但仍在前缘-18.858 mm，尚未C/P，第一未完成任务仍是FL实际越沿与放置。RR末态GROUND，距前缘-571.879 mm，没有抬升资格。尾部未保存terminal_observation，不推造该时刻body高度或重力。

末400执行3196/3196 native-effect ticks verified、own-phase3188、8个普通交接hold，禁止的episode内state write=0。按主控完成块汇总，2048样本最终完成16次PPO更新/320优化步，最新148352/1124；不能把预算末尾163条称为第8个完成回合，或把跨更新随机训练称为固定策略成功评估。仅新增本报告，未改主报告、生产、checkpoint或进程。

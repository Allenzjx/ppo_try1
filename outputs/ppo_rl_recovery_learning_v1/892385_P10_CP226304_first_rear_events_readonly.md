# CP226304 P10 前缀：首个后腿事件（窄只读）
  
来源：`runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T1717242165088Z_g892385cba8a7_afdc21482bbc480a9692fb63fe8ad858`，固定完整文件边界 **5,607,220 bytes**；仅 global226305–226345 的41条学生决策，tick6144–6464。HEAD892385，seed1001，后腿辅助OFF。读取采用open/seek/tell，不等待writer。此表不是完整回合结果或新增优化器计数。

真实nominal前缀在 **6133完成RR cross/placed**；P10学生接管点6136。前缀RR成果不计为学生学习。RL在接管时没有有效抬升资格。

| tick / 秒 | RR当前接触、载荷 / 合法XY | RL当前状态 | FR knee FINAL° |
| --- | --- | --- | ---: |
| 6144 / 51.200 | TOP 15.840N / 是 | GROUND，无资格 | 34.077 |
| 6200 / 51.667 | TOP 13.272N / 是 | AIR，无资格；gap−43.459mm | 23.482 |
| 6208 / 51.733 | TOP 6.313N / 是 | GROUND_AND_OBSTACLE，无资格；front−51.873mm | 19.482 |
| 6240 / 52.000 | AIR 0N / 是 | GROUND 15.766N，无资格 | 11.482 |
| 6288 / 52.400 | AIR 0N / 是 | GROUND 12.212N，无资格 | 21.559 |
| 6344 / 52.867 | GROUND 4.306N / 否 | GROUND，无资格 | 14.644 |

RL在6176、6197只有whole_body_initial_clearance，**41条内没有qualified lift、cross或placed**。首次碰边6208是仍带ground的接触，不是已有效抬升后的合法边缘恢复。

RR首个失载端点6240，AIR计数2/free-air参考6238；端点gap−0.546mm，仍在合法XY。6256–6280短暂TOP恢复3.83–4.14N；6288再次AIR，6296离开XY，6344落回ground。gap负值不替代传感器AIR/TOP判定。

## 第一差异的层级证据

四轮顺序均为FL/FR/RL/RR，速度为canonical rad/s。

| 端点 | source N | filtered REQUEST | FINAL | actual |
| --- | --- | --- | --- | --- |
| RL首次碰边6208 | [−1.070,0,0,0] | [−.409,+.208,−.191,−.443] | [−1.479,+.208,−.191,−.443] | [−1.563,+.207,−.018,−.442] |
| RR首次失载6240 | [−.300,−.300,−.300,−.300] | [−.749,+.618,−.272,−.517] | [−1.049,+.318,−.572,−.817] | [−1.049,+.286,−.523,−.816] |

四轮N−.3首见6216，**晚于第一次RL碰边**。它是P12在source .4667s开始、2.6667s停止的2.2s有限脉冲；截至6464尚未到stop，不应误称等待锁死stop。代码保留独立wheel时钟，RL依赖等待仅暂停RL关节游标。

6232→6240 source关节N未变。FL mapper变化(+2.5,−2.5)°、REQUEST变化(−4,−4)°，FINAL变为(−41.999,−49.049)°。RR hip/k FINAL=(3.086,−38.117)°，实测=(3.193,−38.060)°；同拍FL/RR为AIR，FR/RL承载16.044/15.766N。

FR knee在41条内**没有−58°饱和**，FINAL范围−4.932至37.577°。首RR失载前后6232→6240 REQUEST反而由−22.595回升至−18.595°，FINAL7.482→11.482°，actual14.614→12.576°；不能套用上一轮FRk负限饱和解释。

6240最终派发all12 FINAL等于headroom candidate，决策输入owner active全0；下一输入6248显示owner激活。没有完整120Hz owner receipt，因此仅能排除最后派发上额外owner目标覆盖，不能声称整8拍不存在owner状态变化。随后P12记录holding_RL_joint_lane，符合当前RR失载后的支持依赖。

关节actual来自endpoint−1跟踪证据，接触/几何为endpoint。多腿策略、mapper和有限source轮脉冲同时变化；此只读观测**不证明唯一原因、不算物理单通道对照、不算RL成功**。原回合与真实训练继续，未修改生产或运行配置。

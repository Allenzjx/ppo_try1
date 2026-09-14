# Block10：固定 episode1–5 汇总（全部保留原失败）

只读 run `train/20260910T1233487014760Z_g7db0d17f398d_9984d1c7ed954d8fb2769e551a23fc6e` 的前1648条audit/前6条completed；不再诊断已单独报告的ep0，不读取第七回合。本报告实际分析audit443–1648、global146747–147952，共1206条；原始bytes33,272,974–119,413,908（含端点，86,140,935 bytes），SHA256 `194ab1537e647bd36a2df245f4aac19968b91d39f43fcb3bf82e0273d40e5c8f`。

## 回合结果与实际任务事件

阶段列固定按 **P01/P02/P03/P04/P05/P06** 顺序；这些回合P07–P13均无样本。Q/C/P为本自然P01回合新事件次数，不是末态history bit或教师信用；Q可因回地撤销而重新产生，多Q不表示多次任务成功。

| ep | 决策数；阶段计数 | 原终态（tick / s） | FR Q/C/P | FL Q/C/P | 第一个未完成任务 |
|---|---|---|---|---|---|
| 1 | 239；2/92/4/1/140/0 | P05 FALL；1911 / 15.925 | 1/1/1 | 6/0/0 | FL保持净空并越沿；未放置 |
| 2 | 136；2/134/0/0/0/0 | P02 FALL；1084 / 9.033333 | 1/0/0 | 0/0/0 | FR净空/靠近前缘；clear_FR=0，未C/P |
| 3 | 424；2/122/5/1/155/139 | P06 BODY_COLLISION；3389 / 28.241667 | 1/1/1 | 1/1/1 | 后腿靠近/准备尚未完成；rear_approach=0.599 |
| 4 | 185；2/104/4/1/74/0 | P05 FALL；1474 / 12.283333 | 1/1/1 | 3/0/0 | FL重建有效净空并越沿；未放置 |
| 5 | 222；2/100/4/1/94/21 | P06 FALL；1776 / 14.8 | 3/1/1 | 1/1/1 | 后腿靠近/准备尚未完成；rear_approach=0.149 |

五回合合计FR Q/C/P=7/4/4、FL=11/2/2；**RR I/Q/C/P全部为0，current_lift_valid末态样本数也为0**。实际后腿样本只有P06=160，不能将阶段未达到的后腿事件补成成功。ep1 FL六次Q中五次回地撤销；ep4三次Q全部回地撤销；ep5 FR三次Q中两次回地撤销。

## 终止依据：实测高度与BODY判定分开

全部终态policy/critic各372维且一致，finite_fallback=false。按既有schema固定scale与locked obstacle bottom=0，从relative bottom回解base_z；gravity直接取projected_gravity_chassis。下表为**实测float32观测回解**，不是原始双精度传感流：

| ep | base_z (mm) | gravity_z | 支持的终止依据 |
|---|---:|---:|---|
| 1 | 14.945403 | -0.959482 | 实际高度<15 mm |
| 2 | 14.462739 | -0.973756 | 实际高度<15 mm |
| 3 | 29.407188 | -0.977356 | 不触发低高度/翻转；另有BODY_CONTACT |
| 4 | 14.793128 | -0.973055 | 实际高度<15 mm |
| 5 | 13.992690 | -0.981110 | 实际高度<15 mm，vz=-0.200385 m/s |

五个末态线速度均≤0.464426 m/s、角速度≤1.099098 rad/s，低于5/20安全阈值；gravity均未触发>-0.30翻转分支，8个实测关节各在自身硬限内。故四次FALL有独立低高度证据，不能因接触腿存在而改判或放宽15 mm阈值，也没有新实际硬限中止。

ep3 audit/completed记录 `BODY_COLLISION`，physical evaluator VERIFIED/valid=true、source=`BODY_CONTACT`、reason=`central body/obstacle collision`。生产消费authoritative BODY detector；该检测限定verified exact base_link/Obstacle活跃对，需持续≥2tick或live穿透≥1 mm，腿/wheel接触不够。本训练audit未保存原始pair持续次数/penetration细项，不能独立指定本次是哪一分支；body AABB最小z=0.050005953 m也不是signed penetration。保留原BODY失败，不以缺失细项声称探测器出错。

## FL放置与末态承载，尤其ep5

- ep3：FL Q1072/C1333/P2277，tick2280由真实placed_FL进入P06；终态仍TOP/support=true，bearing12.083372 N、有效载荷比例0.532291。RR仍GROUND、front=-325.244 mm，无I/Q；FR/RL AIR不能虚构为当前承载。
- ep5：FL Q916/C1096/P1604，tick1608进入P06，证明已发生真实放置；**不证明末态仍承载**。P之后22个决策末态只有7个TOP/support、15个AIR。终态FL在前缘+234.446 mm、距台面+35.388 mm，当前AIR、接触对inactive、bearing=0/load=0均验证有效，support=false。
- ep5末态RR也为AIR，但ground-relative excursion仅0.764 mm，initial/established/current valid均false，本回合没有RR I或Q事件；不能把8个AIR physics samples或全身移动当作已成立抬升。RR仍在前缘-364.135 mm、台面净空-50.449 mm，无C/P。当前有效支撑来自FR TOP（13.147944 N）和RL GROUND（12.397503 N），不是悬空FL/RR；也不从这些载荷推造CoM运动方向。

上述五回合共9634/9634 native-effect tick verified、own-phase9615，19个普通交接hold，禁止的episode内state write=0；没有新增下发/重置故障证据。固定前12条optimizer日志1109–1120/global146432–147840均20步、有限非零梯度。前六回合1648采样中1536条已有更新记录，余112条在本固定切点尚未优化；不预支后来更新或修改checkpoint指针。训练回合跨中途更新，不能视为固定策略评估，更不能由局部FL放置推出全程成功或稳定性优于FSM。

仅新增本报告；未改生产、主报告、官方checkpoint或进程，未扫描活动后续回合。

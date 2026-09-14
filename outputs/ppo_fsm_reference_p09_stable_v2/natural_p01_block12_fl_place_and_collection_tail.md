# Block12 自然P01：一次低高度FALL，另一次FL已承载的非终止尾

固定单次解析run `train/20260910T1400283678624Z_g7db0d17f398d_5404e719ad11483d97e9759a2fc96e4d` 的512条audit及首条completed，global149377–149888；原始36,139,777 bytes，SHA256 `b98b4200be3ceb7a3e11e8538db329d978881e63c7fb602821d8aa214d5d8e15`。两次实际起点均自然P01 tick0，无教师信用。未读取活动video5、官方checkpoint或大physical流。

| 回合 | 新policy与发出动作相位 | 实际物理/native | 结果 |
|---|---|---|---|
| ep0 | 193；P01=2/P02=115/P03=5/P04=11/P05=60 | 1537/1537 verified，own1533 | P05 FALL，tick1537/12.808333 s |
| ep1 | 319；P01=2/P02=98/P03=6/P04=16/P05=176/P06=21 | 2552/2552 verified，own2547 | P06 tick2552/21.266667 s，nonterminal collection tail |

**ep0原FALL。** 终态372维policy/critic一致、finite_fallback=false；按既有固定schema及locked底面0回解base_z=**14.733754 mm<15 mm**、gravity_z=-0.971530199，线/角速度模长0.151719/0.790033，8个实测关节在各自硬限内。低高度分支成立，不是翻转或速度异常，不放宽阈值；这是实测float32观测回解，不冒称独立双精度物理流复核。FL Q1138/1214/1394/1475，1194/1318/1418三次回地撤销，未C/P；末态FL AIR、有效载荷0，距前缘-76.724 mm、台面净空-47.737 mm，首个未完成任务仍为FL保持净空、越沿并放置。FR新Q39/C944/P971不能代替FL任务。

**ep1自然策略真实FL放置。** FL I1066/Q1071后1249回地撤销，再I1301/Q1305；C与P均记录于 **tick2382/19.85 s**，tick2384/19.866667 s由placed_FL=1进入P06。这是本回合新事件，不是教师或上一回合历史。

| 当前物理状态 | FL真实承载 | 其余腿与未完成任务 |
|---|---|---|
| P06入口tick2384 | TOP，13.005286 N、有效load0.443089，front+0.299 mm、台面净空-1.548 mm | FR AIR不承载；RL/RR GROUND，RR load0.484399 |
| 采样尾tick2552 | TOP，13.833345 N、有效load0.455123，front+59.324 mm、台面净空-0.320 mm | FR仍AIR；RL/RR GROUND，RR14.391926 N/load0.473500，front-535.947 mm |

P06的21个决策末态FL均TOP/support=true、无AIR或载荷未知；可以确认这些采样点及尾部的**当前承载**，不只是历史placed，也不外推为未来持续稳定。反之，FR虽历史Q29/C847/P847，此时却AIR/载荷0，不能称前两腿都承载。尾部P06 rear_approach=0，RR initial/established/current valid均false，第一未完成任务是后腿靠近与为RR后续运动建立入口，不是完整任务成功。

整块RR **I/Q/C/P全部0**，AIR末态1、GROUND511、current-valid0；唯一AIR不表示有效抬升。RR load-valid为507/512，五个未知值不填0或认作已卸载。尾部没有terminal372观测，不补造此时base高度/重力或CoM迁移方向。

**连续性与保存边界。** 九次普通阶段切换全部decisionTerminal=false、bootstrap=true；只有ep0真实FALL为done且bootstrap=false。最终ep1 terminal=false/reason=null/time_outs=false/bootstrap=true，entry_valid及physical VERIFIED，不把正常预算尾变成超时失败、额外完成回合或成功。全块4089 native-effect verified、own4080，差额9为普通handoff hold；禁止的episode内state write=0，未发现新增下发/阶段reset缺陷。

按主控完成块验真，14:09:47.1593132Z正常exit0，全部512样本完成4次PPO更新/80优化步，已保存149888/1136/22720。随机训练中途更新及局部放置不等于固定策略评估成功，更不证明稳定性优于FSM。仅新增本报告，不改生产、四主报告、CSV或checkpoint。

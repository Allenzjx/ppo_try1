# CP194560 stochastic：RR 回地后 P09 停滞，未完成任务

PPO+LIMITEDAUX 固定 checkpoint 随机评估，物理 seed4001、policy seed4101、optimizer updates=0。run `1e424676283e49bdbee680a08111d59e` 自然封存于 **8219 tick / 68.491667 s**，P09 `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。独立物理 evaluator **valid=true、termination=null、success=false**；不是机身碰撞，也不是成功。time_outs=false、terminal bootstrap=false。source验收报“episode did not meet common physical task”，与真实任务未完成一致，不据此称writer故障。

FR qualified/cross/placed=21/1647/1660；FL=1737/2747/**2986**。FL捕获是真实事件，但不代表后续持续承载。

| 物理窗口 | 实际结果 |
|---|---|
| FL越沿→捕获 2747→2986 | 1.9917 s，真实观察到捕获；15Hz端点没有刚好落在capture tick，不能用端点TOP=0否认120Hz事件 |
| FL捕获→P07 2986→4600 | 1615个raw观测，FL obstacle108、AIR1507；当前TOP+verified bearing+support为13/202保存端点（6.44%）；最大gap63.108 mm |
| P06 2992→4600 | 13.4 s，**实际base净前送249.360 mm**；FL AIR1507/1609 raw、TOP承载13/202端点，末gap37.413 mm/0 N |
| RR准备→终点 4600→8219 | 未完成，30.1583 s；RR当前lift有效仅5/454端点，FL全部3620个raw观测AIR；不得把较低RMS当完整成功 |

P06真实导数 RMS=.046728 rad/s、peak tilt=.159870 rad；body collider最低世界z120.122 mm、保守障碍AABB分离下界70.122 mm。上述位移来自base位置，不是轮端移动或轮速均值，也不证明轮驱牵引的因果占比。早期活动前缀只含3000→4600端点，得到+243.914 mm；最终从真实P06入口2992计算+249.360 mm，差异是起点口径，未改动原预览。

RR实际顺序：**4878初始free-air上升 → 4887 qualified（40.725 s）→ 4928回地撤销（41.0667 s，before-cross）→ 8219控制器停滞结束**。qualified与回地事件相隔仅41tick/0.3417 s；从未cross/placed。末RR GROUND、current_lift_valid=false、active_attempt=false，前沿距离−175.324 mm、gap−49.976 mm；历史4887时间戳仍存在，但历史active布尔已false。工具的首未完成字段`RR:active_lift`不表示从未合格，实际尚缺**保持有效抬起并完成RR越沿/受控放置**。

末FL仍AIR、gap186.416 mm/0 N；FR当前TOP13.492 N，RL/RR当前ground11.706/3.893 N。不能把FL历史placed或CoM方向说成FL当前承载。RR准备窗body collider最低53.776 mm、AABB下界3.776 mm；未触发独立碰撞事件，但没有完成任务，不把“没有撞”改成成功。

指标沿用真实120Hz相邻角度导数的时间积分RMS，不再除时长；接触端点比例不冒充连续时间比例。未提取首次FL AIR精确tick或未记录量保持null。最终JSON约13KB，仅唯一事件、四窗口聚合和末状态；没有重复累计history/逐endpoint明细。旧大型前缀产物保留，不作破坏性删除。本次无旧raw重扫、GPU导入、物理运行或生产修改；CPU helper已退出，后续训练未等待本报告。

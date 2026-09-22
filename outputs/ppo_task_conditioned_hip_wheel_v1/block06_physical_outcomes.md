# Block06：三条已结束训练回合的物理结果

源：`runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T0727365666248Z_gee5a9651591d_6c75964ec21a46e296fcc856b2ee2064`。只读 `completed_episodes.jsonl`、前三个terminal前的decision audit及optimizer日志；未分析后续8个决策，不扫旧库/视频。

三条均真实完成FR/FL placed，却未维持RR抬起、更未完成RR越沿/放置，最终P09机身碰撞：semantic `BODY_COLLISION`，独立物理 `TASK_FAILURE_BODY_COLLISION`、valid=true、success=false。第一未完成任务均为P09 RR有效抬起保持→越沿/放置；RL后续任务未到达。

|回合（日志index）|decisions / global范围|终点tick /秒|FR placed /FL placed|RR曾qualified→GROUND撤销|
|---|---|---|---|---|
|0|624 /187905–188528|4988 /41.5667|1629 /3042|4916→4968|
|1|707 /188529–189235|5653 /47.1083|1750 /3866|5584→5624|
|2|709 /189236–189944|5668 /47.2333|2076 /3908|5402→5438|

终态三者 `history.active_lift.RR=false`、`front_edge_crossed.RR=false`、`placed.RR=false`；旧event_ticks仍记录曾qualified的时间，不能记作当前有效资格或后腿成功。保存的RR current_lift_valid=true端点分别为4920–4960（每8tick，6点）、5584–5616（5点）、5408–5432（4点）；首次随后保存的false分别4968、5624、5440。精确丢失tick=null，仅有最后true与首个false的区间证据，不能把qualified→GROUND的时长全部称为有效保持。回合2的5440已再次AIR却仍invalid；三次终点也有近地AIR回弹，不等于重新取得资格。

## FL保持与P06实际前送

|回合|FL placed→P07保存端点：TOP/bearing、AIR|最大FL gap mm|P06→P07 tick /时长|body实测净前送 mm|
|---|---|---:|---|---:|
|0|24/200、176/200|33.855|3048→4640 /13.2667 s|254.635|
|1|1/179、178/179|47.219|3872→5296 /11.8667 s|231.235|
|2|7/151、144/151|38.125|3912→5112 /10.0000 s|211.581|

这是约15 Hz实际evaluator端点，不是连续接触持续率；具体捕获tick可能位于决策内部，分母从窗口内首个保存端点起算。TOP/bearing需当前真实support，不使用历史placed代替。P06位移是进入/退出事件端点的 `body_forward_m` 差，不是轮速推算，也不证明推进主要来自轮驱。三条在P07入口和全部列出的RR资格/失效/终态快照中FL均AIR，不能虚构FL承载。

## body几何下降先于碰撞

下表来自已写reward的120 Hz真实collider几何审计；100/80/60/50 mm只是离线描述用标线，没有新增控制阈值。

|回合|P07入口collider最低z mm|首次低于100 /80 /60 /50 mm的tick|RR落地撤销tick|碰撞终态tick|
|---|---:|---|---:|---:|
|0|122.023|4949 /4963 /4980 /4987|4968|4988|
|1|127.158|5597 /5619 /5637 /5652|5624|5653|
|2|128.831|5422 /5440 /5459 /5667|5438|5668|

即RR曾合格后，body几何已下降，RR接着落地失效，再继续低位并最终触发真实机身碰撞；回合2在低于60 mm后还持续约1.74 s才终止。RR阶段collider最低z分别49.858/49.995/49.994 mm，body/障碍AABB保守分离下界均降至0。终止来自物理碰撞标签，不是仅凭z或AABB阈值推断；这些时序也不单独证明某个关节/轮残差的因果责任。

粒度限制：body collider z不是base z；AABB下界不是精确mesh净空。完整raw contact/逐物理步current_lift_valid、RR hip世界安装点高度在本训练日志中未提供，相关精确量为null，不重放或补造。

三条回合分别跨越4/6/5次actor真实更新，因此这是持续学习中的混合策略轨迹，不能给固定CP排名，也不能凭越来越短的P06用时声称性能改善。只生成本短报告，没有改生产或启动Isaac，不影响CP189952自然P01视频。

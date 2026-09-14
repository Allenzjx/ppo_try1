# Video7：RR实测硬限终止；FL承载未被全局载荷无效标签否定

来源：`video_eval/validation/20260910T1711073721697Z_gd4e46006b382_c8a157341180454faab7e89333997585/source`。一次读取 `video_policy_decisions.jsonl` **49,217,823 bytes / 732行**；只追加必要尾部：physical bytes[146,154,302,146,678,590)（20条完整记录）、native bytes[80,815,956,81,340,244)（37条完整记录），以及1.12MB source manifest的任务评价字段。未读庞大quality分项、活动训练/视频或checkpoint。主线程已核实154240官方权重、372/HISTORY/d4、自然P01初始化；本次评估 **0 PPO updates**。

## 真实终止，而非deadline

共 **732 issued decisions，731正常environment step**。最后决策732请求P06，从5848执行至 **5853，仅5个物理步** 后由外层PhysicalEndpoint安全终止，没有正常step_info返回。总5853 ticks/**48.775s**；不把前一条5848的未终止语义快照当最终物理评价。

manifest的5853真实评价：`termination_reason=SAFETY_ABORT`、`termination_source=HARD_JOINT_LIMIT`、`reason=hard joint limit: rear_right_knee`、run_validity=VALID、physical_evidence_status=CONTACT_BEARING_UNVERIFIED、success=false。不是超时，不是PPO训练信用，也不是无碰撞即成功。

物理尾部 `joints.rear_right_knee.position_deg` 和 `actual_full12[7]` 一致，规范角依次为：

|物理tick|实测RR knee（°）|实际下发target（°）|
|---|---:|---:|
|5848|−59.982129652596|−5.931935866840|
|5849|−59.999553465168|−5.938500352311|
|5850|−59.992702785431|−5.938500352311|
|5851|−59.987607464310|−5.938500352311|
|5852|−59.997483917850|−5.938500352311|
|5853|**−60.001520559649**|−5.938500352311|

硬下限−60°，5853实际越限 **0.001520559649°**。target未越限，不能据target判实际安全，也不是命令显示舍入错误。最终native RR knee nominal/基线为0°，residual/combined bias为−5.938500352311°；真实float32 position target为+.107403099537rad。后腿负向映射及standing offset使其物理rad为正，并非错误反向下发。

731正常step的5848 ticks全部native verified；精确5849–5853的5条native亦verified、setter/dispatch一致、mapping/前ACK独立验证真，state writes均0。合计 **5853/5853已验证**。约54°跟踪误差的事实不能单独归因为某条控制命令或软件缺陷；不改物理硬限。

## 第一未完成任务与当前支撑

issued阶段计数 **P01=2/P02=182/P03=6/P04=25/P05=149/P06=368**，P07–P13=0。FL **I1782/Q1786/C1968/P2910**，P06入口2912。RR正常决策与最终评价的事件历史均为空，**I/Q/C/P全0**，并非曾有效抬升后被交接清除。

未完成的是 **P06的双后腿接近任务**：`rear_approach=min(edge_proximity_RL,edge_proximity_RR)`。正常端点最大进度仅.578899，最后5848为.562873；最终5853 RR front **−50.363805mm**（GROUND_AND_OBSTACLE），RL front **−334.366348mm**，RL仍远于目标区间下沿−225mm（含既定5mm测量容差）。RR接近立面不等于两后腿接近完成，更不是RR抬升或越沿。

**未发现“FL放置后当前承载失效”的已存端点。** 从2912至5848共368个正常决策端点，FL始终TOP/support/bearing_verified真；5853独立最终评价仍TOP、within_top_xy=true、ground=false，bearing **8.446005N**，front+87.330226mm。原始FL障碍pair verified、force_z=8.446005N、ground pair inactive，支持当时实际顶面承载。没有逐tick扫描整个放置后物理流，因此不保证所有中间单拍绝无接触间断。

全局CONTACT_BEARING_UNVERIFIED首次正常端点为5672，来自RR **OBSTACLE_AMBIGUOUS / bearing_verified=false**，而非FL。终态RR同时有ground竖直11.033138N与obstacle横向−5.540527N；全局load_fraction_valid=false使归一化载荷不宜当精确承载比例，但不能否定FL独立且已验证的力/接触事实。FR终态AIR、RL为GROUND；不虚构固定四轮承载。

## 晚捕获桥接：时间较晚，但并未满足桥接启动条件

P05入口1720，原源wheel停止时刻 **1720+1088=2808**；P2910确实发生在其后。不过此前已有独立几何退出：2824端点FL front+96.826mm、N四轮+.3；2832已front+109.711mm、N四轮0。2904尚未P时front+127.389mm、N已经0；2912当前P成立但front仍+125.801mm。

因此capture并未突然把正在滚动的N清零：既有`front<.10m`条件早已不满足，且capture前`previous nominal wheels != +.3`。v2桥接明确不能凭新P重新启动早已停止的建议。这不是已修复的“捕获关窗导致剩余1–7拍短缺口”的实训覆盖；也不把0重新判成无条件切换清零。

|端点tick|source→end|N四轮|projected residual四轮（rad/s）|
|---|---|---|---|
|2904|P05→P05|[0,0,0,0]|[−.510820,−.187503,+.137449,+.474191]|
|2912|P05→P06|[0,0,0,0]|[−.507332,−.186670,+.136464,+.473502]|
|2920|P06→P06|[+.3,+.3,+.3,+.3]|[−.605687,−.291670,+.137227,+.470720]|

P05→P06普通交接非done，下一决策bridge明确保留全12 residual，无禁止通道丢弃或范围裁剪；+.3 wheel action jump来自P06新owner首条真实滚动建议，residual没有被清零。以上N是决策最后source步的记录，不能冒充所有中间物理tick完整动作曲线。

结论：本例有真实FL越沿放置和当前承载，但双后腿接近未完成，最终RR实测硬限安全终止。没有完整PPO成功或稳定性优势结论；未发现新的明确执行缺陷，不将此次失败或晚桥未激活设置成续训门禁。

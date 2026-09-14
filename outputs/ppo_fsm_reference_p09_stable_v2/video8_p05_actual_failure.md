# Video8：P03→P04→P05实际未完成诊断

结论：FR真实完成I/Q/C/P；**FL整段没有I/Q/C/P事件**，并非末帧false掩盖了此前成功。P04→P05动作继承正常，FL nominal、residual、final target及实际关节均有变化；记录有明显跟踪不足，但没有确认新的mask/clip/reset/native下发错误。P05最终耗尽当前进度对应的有限任务预算，保持原INCOMPLETE_CONTROLLER_BLOCKED结果。

范围：`video_eval/validation/20260910T2100033647207Z_gd4e46006b382_e98a9290c0f44825aa705e61f7a57377/source`，固定722条`video_policy_decisions.jsonl`、47,812,908bytes，以及`semantic_video_source_manifest.json`少量指定字段。诊断读取说明：首次脚本因误用不存在的`run_manifest.json`文件名而退出、缓存丢失，故对**相同固定722条**必要重读一次，完成于2026-09-10T21:18:33.913Z；这是诊断脚本错误，不是本次视频运行故障。未读quality大对象、未扫描physical/native完整流，未读活动训练或CP；后续取数只使用内存缓存。

## 完整物理终态与阶段

source manifest：seed4001、role C、semantic_residual_eval、自然P01、0 optimizer updates、无额外pre/post物理步。**722 issued/722 returned；721×8+1×4=5772ticks，48.1s**。末决策5768→5772执行4ticks并正常返回；`interrupted_final_decision_ticks=0`仅指没有外层PhysicalEndpoint中断，不能把末拍改算8ticks。录像722帧/48.133333s，最后真实终态5772被保留，33.333ms只是末帧显示量化。

|策略来源阶段|实际决策数|物理区间|
|---|---:|---|
|P01|2|0–16|
|P02|158|16–1280|
|P03|6|1280–1328|
|P04|95|1328–2088|
|P05|461|2088–5772|

P06–13均0。FR **I16/Q25/C1291/P1327**，没有撤Q事件；Q25实测gain8.618321mm、joint motion4.231532°。1328进入P04时FR确为TOP、support/bearing_verified=true、1.719305N；1336曾AIR/0N，说明历史P不是每拍承载，终态FR恢复TOP/verified9.310717N。

FL事件表为空；722个端点的`initial_clearance`、`active_attempt`均false，history Q/C/P均无。FL仅7个AIR端点，其recent clearance gain最大**0.063555mm**；全体端点最大gain**0.184618mm**（4904，OBSTACLE_AMBIGUOUS，非已验证AIR），最佳top gap仍**−49.934296mm**。可以说没有成立当前定义的有效初始抬升，不能扩大成“轮端完全没有微小运动”。P05接触组合：GROUND9、GROUND+FRONT_WALL31、GROUND+OBSTACLE_AMBIGUOUS405、AIR7、OBSTACLE_AMBIGUOUS9；无TOP。

终态FL front **−48.893120mm**、top gap **−50.814238mm**、GROUND+OBSTACLE_AMBIGUOUS。虽`support=true`、bearing数值0.506526N，**bearing_verified=false**，不能把它当可靠载荷分配。首先未完成的物理任务是获得有效FL抬升并继续越沿/放置，不是仅差最后放置。

## P04→P05真实连续性与实际控制能力

1328 P03→P04按FR放置通过；2088 P04→P05按FL edge_proximity=1、role_prepared=1通过，均entry valid、done=false。2088 FL仍GROUND，准备阶段通过不等于已抬起。准备诊断有实测whole-body motion5.660849°、CoM向接收侧位移1.337788mm等证据，但不证明静态稳定或后续动作一定有效。

两个handoff均hold=true，全部12通道residual保留、forbidden/scale-clipped为空，最大servo residual差3.56e−15°、wheel5.56e−17rad/s。P03→P04 wheel请求跳变1.09rad/s来自新owner建议变化，residual连续；P04→P05四轮N在**2080/2088/2096本来均0**，F一直保留非零residual，不是此次切换把wheel/residual清零。未到P05→P06，不能声称本片段覆盖了晚捕获桥分支。

FL hip/knee对照；raw为无单位latent，其余为规范角°；A来自实测role joint-margin回解，不从N或F反推：

|tick|N|raw|projected R|final F|actual A|
|---|---|---|---|---|---|
|2088|0 / −22.9|.08787 / 1.25608|1.57766 / 20.39950|1.57766 / −.69221|.77424 / −2.49020|
|2096|0 / −26.1|.09119 / 1.25831|1.63683 / 20.41432|1.63683 / −5.68568|1.21247 / −4.08734|
|2504|48.2 / −36.7|.17616 / 1.56976|3.13847 / 22.00770|52.58847 / −15.94230|14.18249 / −21.14907|
|3240|24.9 / −13.4|—|3.38468 / 22.02016|39.64065 / 9.87016|29.60299 / −32.38664|
|5772|22.8 / −13.4|.18723 / 1.55181|3.33128 / 21.93800|25.95095 / 9.78800|26.02770 / 9.19137|

正的knee residual持续抵消部分负nominal；这是实际策略请求方向，不是mask禁用。P03/P04/P05的FL headroom裁剪端点分别**0/0/0**；最终slew修改分别**2/0/4**（只统计记录末端，不冒充每个物理tick统计）。P05最大target-actual差**42.256805°**，发生在3240的FL knee；2504 hip也有明显差异。与此同时，末段F/A已较接近但轮端仍未有效抬起，因此既不能说“完全未下发/无响应”，也不能将整次失败单因归咎跟踪误差。地面/障碍接触与跟踪差同时存在，只支持相关性，未作新的执行故障因果诊断。

终态四轮N全0，F=`[−.326078,−.258873,+.223614,+.450242]rad/s`，仍有实际residual控制请求。5772/5772 native ticks verified且actual-effect、own5768（4次正常handoff），全部记录端点mask全开、dispatch/mapping一致、状态写入检查零；没有已确认新的动作继承或native执行错误。命令连续不保证nominal的时机/幅度适合这个新状态，也不是稳定性优于FSM的证据。

## Local deadline：明确预算来源

P05从2088/17.4s开始，到5772/48.1s，age=**30.700000s**。当前配置`local_timeout_policy`为maximum_extension=10s、fraction=0.5；supervisor使用现有公式
`allowance=min(10,30×0.5)×clip(current_progress)^2`。
末progress/placed_FL shaping=0.2644370391995249，得到**0.699269477006s**；fixed_post_window_allowance=0，effective limit=**30.699269477006s**。age刚超过该预算，source=`LOCAL_BOUNDED_RECOVERY_EXHAUSTED`，classification=`finite_task_terminal_not_external_truncation`；不是200s全局截止或录像提前截断。配置来源`configs/ppo_fsm_reference_p09_stable_v2/stage_task_spec.yaml`，计算位于`semantic_supervisor.py`的当前进度平方有限延长分支。

物理evaluator本身`termination_reason=null`、reason空、VALID，但CONTACT_BEARING_UNVERIFIED；语义任务才产生INCOMPLETE_CONTROLLER_BLOCKED，bootstrap=false。原失败与完整录像保留，不把“没有独立安全/碰撞终止”当成功，不调整标签/阈值或将该诊断设为optimizer门槛。

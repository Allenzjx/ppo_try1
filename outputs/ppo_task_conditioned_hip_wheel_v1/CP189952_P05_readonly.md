# CP189952 deterministic：P05有限FL诊断的未干预对照

封存run `a6d92def5d74491580cccd936a1edf18`：6276 tick /52.30 s，首个未完成任务是FL捕获。FL lift/cross=1836/2888，placed=null。语义终止 `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`；独立物理判定 valid=true、termination=null、success=false，**不是机身碰撞，也不是成功**。

既有 `FL_minus3` 最早decision入口为**推导的2984 tick /24.8667 s**。P05源首tick1809；既有有限源长1168 tick、无P05 sequence pause，故首endpoint对应2977；源FL终值22.8/−13.4°也在2977首次出现。2976尚未endpoint，2984已经实际满足P05、有效无物理终止、FL已越沿/未placed/AIR/within_top_xy。日志未直接保存此endpoint flag，因此不把推导写成直接测量；这里只复用有限源时钟，未重放mapper或计算独立N差。

## 同拍关节链：数值顺序均为FL hip /knee，单位deg

|episode tick（时间s）|2984（24.8667）|6200（51.6667）|6276（52.3000）|
|---|---:|---:|---:|
|源/N|22.800 /−13.400|22.800 /−13.400|22.800 /−13.400|
|同次派发mapped baseline|21.550 /−12.150|23.450 /−12.150|22.200 /−12.150|
|PPO filtered REQUEST|+2.438 /−8.135|+2.576 /−8.068|+2.578 /−8.064|
|同拍有效修正|+2.438 /−8.135|+2.576 /−8.068|+2.578 /−8.064|
|最终target|23.988 /−20.285|26.026 /−20.218|24.778 /−20.214|
|实测位置|28.209 /−20.464|25.459 /−20.218|25.258 /−20.214|
|实测速度deg/s|−42.161 /+1.064|+5.200 /−0.287|−5.291 /−0.349|
|body collider最低世界z mm|134.939|135.052|134.566|
|保守body/障碍AABB分离下界mm|107.234|85.052|84.566|
|FL gap mm /承载N|28.047 /0|22.047 /0|21.165 /0|

三个点的残差许可均12维全1、clipped_servo_indices为空，派发目标/映射核验均通过。源/N≠mapped baseline；后者取真实同次调用的 `policy_headroom_evidence`，不是另跑一条N。最终target取同tick原始物理日志command，位置/速度取步进后实测。原始native command clock为episode tick+179，JSON分别保留两个时钟，没有混对时刻。

**方向事实：** 源FL hip已有从48.2°向22.8°的下放，actual也确实产生负速度；不能说“负向hip运动尚未出现”。但2984–6276共3293个物理采样中，PPO hip REQUEST始终是+2.416至+2.578°，同拍有效量相同，负REQUEST样本0、hip headroom裁剪0。因而当前证据支持“策略持续正向hip附加修正，并保留约−8° knee修正”，不支持“策略已请求负hip却被mask拒绝”。这一段实测hip范围24.320–28.209°，FL gap最小18.483 mm，始终未捕获。它不是−3干预会奏效的证明，也不把源下降的实测负速度冒充PPO负残差。

## 四轮快照：native轴原始符号、rad/s

顺序FL /FR /RL /RR；canonical→native符号为− /+ /− /+。以下是轴转速，不是轮端平移或牵引估计。

|tick|N四轮canonical|最终target native|实测角速度native|
|---|---|---|---|
|2984|.300 /.300 /.300 /.300|−.249574 /+.340570 /−.495946 /+.132664|−.249434 /+.190655 /−.461916 /+.128213|
|6200|0 /0 /0 /0|+.047330 /+.035731 /−.212052 /−.165305|+.047494 /−.014239 /−.229518 /−.140789|
|6276|0 /0 /0 /0|+.047168 /+.035635 /−.212237 /−.165203|+.047479 /−.064494 /−.154362 /−.115900|

最后写入证据为 `robot._joint_pos_target_sim / robot._joint_vel_target_sim_after_existing_write_data_to_sim`；actual另从同tick HeightDiagnostics真实native关节速度、按startup名称索引提取。因此末段FR存在正target但负实测速，不能宣称每轮完美跟踪或由某轮提供主牵引。这里只给入口/末段必要快照，未重复全轮审计；末段N为零的事实也不被误标成残差mask。

产物 `CP189952_P05_readonly.json` 保留精确数值和来源；提取脚本只读封存run、输出此目录，未运行Isaac、未改生产、未做干预。当前单−3实际效果仍未测。

# 有界建议：可以检验小幅 FL hip–knee 耦合，但尚无“已修复”证据

**证据足以支持一次小型双通道物理诊断；不足以证明膝修正方向、幅值或成功保证。** 不建议据此继续放大单hip、立即调reward，或部署固定bias。本建议不影响当前P04 PPO正常完成。

事实来自已保存摘要，无旧raw重扫：

- CP194560 det的3293拍末段hip REQUEST全部负，末有效hip/knee=−2.007/−8.108°，actual=20.678/−20.266°。相对CP192512，actual hip低3.909°，knee也多负1.640°，但gap仍16.253 mm、body collider最低点末值近似相同；不存在“负hip没发出”的证据。
- 同CP189952的有限−6单hip干预确实执行：hold actual hip均值19.223°、最小gap6.118 mm、均gap10.252 mm，但无FL接触；其他11通道保持在线，knee与机身仍有闭环响应。它证明局部AIR接近，不证明继续加大hip必然落脚。
- 成功N在2468→2677完成FL越沿→捕获，此后到P07的311/311端点为当前TOP承载。其捕获时body collider最低点123.248 mm、FL actual hip31.564°；不同状态能完成任务，不能把这个角度或更低body照抄为目标。N只是可行性证据。

局部几何推测：hip与knee共同改变轮端相对安装点的位置；新策略更负knee与更负hip同时出现，可能使轮端净变化相互补偿，也可能是机身姿态/其他腿/轮反馈的结果。**当前摘要没有局部FK雅可比或双通道干预，因此“膝抵消hip”仍只是待测假设。** 旧N的FL knee及全身同状态对照不完整，不能用缺失值拟合方向。

若主代理在当前训练自然结束后选择测试，可优先考虑单一、明确标记的假设例：在真实P05 endpoint、已cross、AIR、within_top_xy、未placed时，以当拍ACK的REQUEST为各自锚点，**hip −1° / knee +1°**（即让当前膝残差略少负）。这是小幅同状态附近的方向试验，不是已验证梯度，不是追逐N角度。若改用训练后的最新CP，应使用该CP的真实入口与对照，不能把CP194560对照当作同策略反事实。不要顺带改变其他髋、wheel、reward或HISTORY。

保留既有12-decision quintic ramp；entry累计75包含ramp（63纯hold），或capture+24先触发release；release12、follow30。人工只选2通道，其他10维每拍保持原在线det输出；真实H随实际动作推进，不清零、不递归每拍追加偏置。观测判据是REQUEST/effective/target/actual、FL真实gap/XY/接触及承载、body collider与四安装点、其余支撑、release后的保持。仅AIR gap下降不能变成capture成功或正保持标签；单次耦合实验也不能独立分解hip和knee各自因果贡献。

接口审查：底层`outputs/ppo_task_conditioned_hip_wheel_v1/direction_probe.py:58–95`已从`raw=list(baseline)`开始，仅迭代offsets中的通道；既有RR案例也含两通道，因此数值核心可支持索引0/1，其他10维能保持同拍raw不变，容量检查仍保留。**现有FL−6 CLI/receipt不能直接传双通道**：`direction_probe_FL_minus6_candidate.py:32,83,86,88,122`固定单hip、拒绝delta[1:]、单通道selector和−6标签。未来实现需另建明确双通道候选case及真实2通道receipt/selector与定向CPU测试，不修改旧脚本，不伪装−6标签、不放宽全局校验。测试至少核验锚点非递减、其他10维逐样本不变、真实H未重置、正确hash/两个实际通道、现有限幅/错误拒绝及原时序。

本轮仅只读建议：未实现候选、未运行物理/CPU Python helper、未改生产/配置/reward，未发放任何训练信用。当前P04训练不是该诊断的等待对象或门槛。

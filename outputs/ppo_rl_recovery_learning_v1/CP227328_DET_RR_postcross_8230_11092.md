# CP227328 DET：RR 越沿至终止的窄窗口

**首个未完成任务仍是 P09 的 RR 真实顶部捕获，不是 RL 已开始后的放置失败。** RR qualified7143/cross8230，但没有 placed；终止11092/92.433333s，`INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`。本次不证明某一关节或轮速是唯一原因。

来源为 sealed `20260924T1842131087811Z_g892385cba8a7_c074d1ebec6947cc830fb3e0ec86dcec/source`；CP227328、HEAD892385、自然P01确定性，FL assist ON、后腿 assist OFF。只读8230–11092：359个决策端点、2863个native audit及物理传感器tick；无模型导入、仿真、PPO/AUX更新或生产修改。完整向量、权限、传感器证据见同名JSON。

## 同拍关节链与真实响应

下表角度°，间隙/前缘mm。整个决策窗口RR source N hip/knee固定 **−6.900/−37.800**；同拍mapped baseline固定 **−8.150/−39.050**，controller bias为0。REQUEST是过滤后的物理残差，不是raw Gaussian值；raw与HISTORY相关输入另存JSON。knee可用残差下界−18.950，FINAL保留2°物理安全余量。

| tick / s | RR REQUEST h/k | 有效 knee修正 | FINAL h/k | 实测 h/k | gap / 前缘 |
| --- | --- | ---: | --- | --- | --- |
| 8232 / 68.600 | +15.998842 / −19.223550 | −18.950000 | +7.848842 / −58.000000 | +7.590489 / −57.758828 | 56.958 / +0.292 |
| 8640 / 72.000 | +16.132165 / −19.170122 | −18.950000 | +7.982165 / −58.000000 | +7.714239 / −57.763063 | 57.041 / +34.067 |
| 9384 / 78.200 | +16.175519 / −19.073182 | −18.950000 | +8.025519 / −58.000000 | +7.761247 / −57.766368 | 57.650 / +96.591 |
| 9600 / 80.000 | +16.195420 / −18.993274 | −18.950000 | +8.045420 / −58.000000 | +7.781376 / −57.767393 | 54.050 / +100.248 |
| 9632 / 80.267 | +16.218518 / −18.945016 | −18.945016 | +8.068518 / −57.995016 | +7.802178 / −57.761594 | 54.921 / +99.794 |
| 11092 / 92.433 | +16.268045 / −18.950603 | −18.950000 | +8.118045 / −58.000000 | +7.857102 / −57.755939 | 55.836 / +99.729 |

- RR hip REQUEST范围+15.998842..+16.279357，将负向mapped N抵消成FINAL+7.848842..+8.129357；没有形成新的负向hip调整。尾跟踪误差约−0.261°，不能称为大幅跟踪失效。
- RR knee **359个不同REQUEST**，范围−19.290797..−18.904415；**315/359**端点及2511/2863物理tick的FINAL为−58°。另44个端点解除裁剪，但FINAL仅到−57.954415°，总活动范围仅0.045585°。不能说本次全部请求都映射成完全相同动作，也不能说尾强烈越限：尾裁掉的仅 **0.000602710°**，全窗最大裁剪0.340797°。
- 所有2863传感器tick RR contact_class=AIR，ground/obstacle pair均未激活；359端点均合法XY、current qualified，但TOP/真实bearing均0。gap范围53.694–58.337mm，未消除。CoM相对首行Δxyz=+98.333/−2.446/−0.820mm，仅为世界坐标位移，不等于已完成向FR侧转移。

## 四轮：尾拍完整通路

顺序FL/FR/RL/RR，rad/s。尾N及mapped N四轮均+0.300；无wheel headroom改写。native实测列是对**已存canonical传感器值**用已验证轴符号[−1,+1,−1,+1]逆变换，非单独序列化的raw native传感器tensor。

| wheel | REQUEST=effective | FINAL canonical | 下发 native（直接记录） | 实测 canonical | 实测 native（逆变换） |
| --- | ---: | ---: | ---: | ---: | ---: |
| FL | −1.014460206 | −0.714460206 | +0.714460194 | −0.816235244 | +0.816235244 |
| FR | −0.074909612 | +0.225090388 | +0.225090384 | +0.064980850 | +0.064980850 |
| RL | −0.016654579 | +0.283345421 | −0.283345431 | +0.046808720 | −0.046808720 |
| RR | −0.106261230 | +0.193738770 | +0.193738773 | +0.192976996 | +0.192976996 |

FL在2863tick内最终目标、实测canonical速度均负，确为policy抵消，不是符号误读或其余轮mask。FR/RL尾实测比目标小，但不能仅凭该差值分解负载/驱动因果；RR AIR自转不计牵引。**N并非全窗恒+.3**：9600四轮N=0，诊断为`no_extra_wall_push_source_and_residual_adjustment_continue`，FINAL直接保留原负残差；9632已恢复+.3。不能把这个合法nominal调整说成丢失policy通道。各选定tick的canonical/native四轮完整链均在JSON。

时间对齐：native dispatch tick=episode tick+179；尾11271派发读取前一物理状态11091，再获得11092步后传感器读回。表中实际关节值来自**同一11092传感器行**，不是用前拍tracking evidence替代。

## 控制权、准备与接触限制

- 359/359 decision输入`phase_mask12=全1`、`owner17=全0`、`cooperative_prep_allowed=true`；FINAL与同拍headroom candidate全12精确相等。2863/2863 native tick也验证mask全1、真实派发映射通过。**endpoint owner17 receipt未序列化（N/A）**，不能把输入全0扩大为每个子步owner均已直接观测。
- P09 late source clock全窗648，358端点等待`current_RR_bearing_before_FL_RL_transfer`；未见late组取得控制权。尾改为`no_live_physical_readiness`是终止拍状态，不能反推此前等待原因。FR/FL/RL policy准备自由度仍在，未被整个锁住；允许探索不等于已学会准备。
- 尾FL knee：N−13.400→mapped−12.150→REQUEST−33.118132→FINAL−45.268132→actual−45.608456；FR knee：+31.100→+31.275638→−51.764566→−20.488927→−20.181433。RL h/k：N28.200/0、REQUEST−12.385073/+0.657084、FINAL15.814927/+0.657084、actual17.259544/+0.752403。全窗FINAL变化幅度仅FLk0.101°、FRk0.361°、RLh0.251°/RLk0.090°，没有产生新的明显准备运动；这些值不证明某个替代角度必然成功。
- 首次decision端点RL `GROUND_AND_OBSTACLE`、全局`CONTACT_BEARING_UNVERIFIED`均为9384；尾FL/FR真实bearing为1.937/13.379N，RR0，RL的GROUND+OBSTACLE混合接触不能视为已验证顶部支撑或有效RL抬升。尾两种RL传感器pair本身均verified，ground normal6.633N、obstacle normal8.698N；“bearing unverified”不是本报告认定传感器文件失效。RL全359端点无qualified，未形成后续越沿资格。

结论限于本次证据：已越沿但RR始终悬空；正RRhip残差、几乎停在负边界的RRk、持续FL负转和近静止的其他准备关节同时存在。源强转移等待有真实支撑依据；没有mask错误证据。不能由这些相关现象宣布单轴因果或声称probe已执行。

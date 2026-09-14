# P10 block8：前 110 决策的 RL 控制链诊断

结论：此固定片段有实际 RL 动作与两次初始离地 I，但尚无 Q/C/P，也没有任务 terminal。未发现新的明确软件下发、mask、owner 或事件判定缺陷。与旧 P10 块的区别包含膝 residual 方向和实测髋响应，不能将不足单因归给跟踪、接触或某个命令，也不据此改参数或要求探针成功。

## 固定范围与信用

Run `runs/ppo_fsm_reference_p09_stable_v2/train/20260910T1036035708863Z_g69aeaca777dc_1c97c4bee3f548da83b7860abcf8d541`，生产 `69aeaca777dc`，起点 CP145408。本报告只解析 `residual_and_projection_audit.jsonl` 第 1–110 行，global 145409–145518，bytes `0–10356669`（含端点，10356670 bytes），SHA256 `eaeb88403202930195674ecd8824eb80ad8237898777b040b750a375e915ed3f`；未追读活动尾部。

教师实到 P10 tick7584/63.2s；策略末端为 tick7592–8464/63.266667–70.533333s。P10=1、P11=1、P12=108。教师 RR Q6912/C7109/P7579，以及教师 RL I7/I1722，均为零新增策略信用。此片段尚未到首个128决策更新边界；不把未来更新、终止或后续事件计入。

## 实际请求与响应

单位：关节角度为 canonical degrees；raw 是无界高斯 latent，不是角度。实际关节来自同 tick `transfer_roles.FR.receiver_workspace_state.joint_range_margin_deg`（FR 的接收侧为 RL）：hip=negative_margin−135、knee=negative_margin−60，并非由 target 假造。下列简写为 [RL hip, RL knee]。

| row / tick | nominal | projected residual | 最终 drive target | 实测 joint | RL 物理状态 |
|---|---|---|---|---|---|
| 1 / 7592 | [15.4,19.4] | [−3.799,+4.000] | [7.851,27.150] | [14.612,20.565] | GROUND |
| 21 / 7752 | [.5,35.3] | [−23.435,−5.308] | [−24.185,31.242] | [−6.788,31.235] | GROUND，台面净空−50.926mm |
| 47 / 7960 | [19.6,35.3] | [−22.722,−6.315] | [−1.872,30.235] | [−9.825,30.663] | AIR，但尚无 Q |
| 64 / 8096 | [19.6,−18.7] | [−19.144,−12.000] | [.456,−31.950] | [10.023,−27.410] | GROUND + FRONT_WALL |
| 73 / 8168 | [−10.1,−18.7] | [−21.739,+19.239] | [−33.089,−.711] | [−27.304,−4.868] | GROUND |
| 110 / 8464 | [−10.1,−18.7] | [−19.594,−14.222] | [−28.738,−34.172] | [−30.501,−33.302] | GROUND + FRONT_WALL |

对照已完成旧报告 `p10_block128_rl_and_p13.md`：旧 block5 在相同 row21/tick7752、相同 nominal [.5,35.3] 时，residual=[−19.942,+16.574]、最终 target=[−20.692,53.124]、实测=[−18.727,49.892]，已经 Q7750。新块膝请求方向与旧块明显不同，实测膝也约低18.66°；髋 target 更负但实测尚比旧块少向负方向约11.94°。有真实跟踪差异，不能由单个受载时刻证明 setter 失败。旧 C8089/P8164 与新片段对应时刻的 GROUND 是不同物理结果；旧后缀中的 RL 越沿放置并非完整后缀成功，旧块 P13 仍未受控；更不是自然 P01 全程成功。

## 控制能力、速率与 owner

- RL hip：raw 110/110 为负（−2.325至−.160）；projected residual 均为负，均值−21.278°、范围[−23.545,−3.799]°，当前 cap24°。不是 residual 请求很小。实测范围[−34.192,+14.612]°，有显著真实运动。
- RL knee：raw 72/110负，projected 78/110负；projected范围[−22.005,+20.730]°（cap36°），实测范围[−33.302,+35.667]°。这不是单调抬升轨迹，角度正负也不能独立代表世界系升降。
- 两关节 mask 关闭=0/110、post-mapper headroom 裁剪=0/110；RL captured-owner suppression=0/110。P12 nominal 正常推进；没有 RL 历史 P，因此不会被“已放置”owner 错误冻结。
- 末端 projected 与 `tanh(raw)*cap` 不同的决策：hip3/110、knee51/110；这是包含既有120Hz residual slew 的实际投影结果，不能把 raw 当已生效量。最终 candidate 与实际 final 因后续处理不同：各2/110；最后一个物理 tick 达到既有1.25°/tick slew幅值：hip4、knee3。以上只是已存每决策末端的统计，不是全部880tick的逐通道限速频率；没有持续 headroom 截断证据。
- RL wheel nominal 为−.3 rad/s共33行、其余0；projected residual正向96行，最终 target正向89行。row21 residual+.276837基本抵消nominal−.3，最终−.023163、实测+.017525；row110 nominal0、target+.430799、实测+.296273。轮子确有运动，不等于足端已越沿，也不能用 command 推导 lift。

880个物理tick全部 native effect verified，无 forbidden in-episode state writes。own-phase request=878；其余2个是普通 P10→P11/P11→P12 已验证 handoff-hold。两个普通交接 residual step≤2.665e−15°/2.776e−17rad/s。初次教师→策略 P09→P10 记录的0.5°/.015rad/s是策略开始时的正常非零 residual 首步，不是后续普通切换清零。没有新 mapper/filter reset 证据，也不因 nominal owner 请求变化建议双重 pre-slew。

## 接触、I/Q/C/P 与第一个未完成任务

RL 新 I7954（全身实测运动5.157°）、I8311（8.152°），没有新 Q/C/P；两个 I 不能合并成持续合格抬升。110个末端中 GROUND=93、AIR=3、FRONT_WALL=14、obstacle_pair_active=49（这些计数可重叠），load_fraction_valid=false=61。首个障碍接触末端在 row3/tick7608，为 OBSTACLE_AMBIGUOUS；最接近前缘仍为−47.661mm（tick8376），最大台面净空也仅−45.899mm（tick8160，非AIR）。这是地面/前壁约束下未形成有效净空的证据，不是由受力数据证明的唯一阻挡原因。

末端 RL 为 ground + FRONT_WALL、front−48.254mm、台面净空−50.790mm；载荷未验证，不把日志中残留数值比例当真实 bearing。真实 body_forward 几何从首行+.306217m到末行+.100106m，净后退206.111mm；这是 body 相对障碍位置，不以 CoM 替代。未存在 terminal tensor，未解码或补造不存在的 body height/姿态值。

第一个未完成任务是 P12：RL 有效抬起、越过前缘并受控放置。保留未完成结果，继续合法采样，不把当前失败表现变为新的 optimizer 门槛。此报告仅输出诊断；无生产、配置、checkpoint、进程或主报告改动。

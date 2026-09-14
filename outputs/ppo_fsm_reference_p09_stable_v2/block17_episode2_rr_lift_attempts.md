# Block17 ep2：两次真实 RR 抬升，均回地；非终止采集尾

范围：run `train/20260910T1839014552998Z_gd4e46006b382_3bf26c40dc774ea38185c4b77938fded`，2026-09-10T19:22:24.448Z 固定 audit **行20–128 / 字节[1618867,11074249)**。前19行只定位换行、不解析；未读后续活动 run、physical 大流、CP 或旧 run。109条 global **154900–155008**：**P07=1/P08=10/P09=98**。教师信用入口 P07/tick5912、49.266667s、prefix attempt2；末6784、56.533333s。新增策略物理872ticks/7.266667s，教师前缀无PPO信用。

## I/Q/撤销与连续交接

P07→P08 在5920（rear edge与preparation均1），P08→P09 在6000（RR edge与transfer均1）。两次均正常目标分支、非lift takeover、done=false；handoff保留所有12通道residual，forbidden/scale-clipped为空，最大servo residual差3.56e−15°、wheel1.12e−16rad/s。6000 RR虽AIR但仅0.252312mm抬升，无I/Q；两次Q均在**P09**建立，不是继承教师的RR资格。

|尝试|真实事件tick（120Hz）|建立证据|记录端点current有效区间|
|---|---|---|---|
|1|I6010 → Q6015 → 回地6099|I抬升3.098010mm、whole-body joint motion36.168094°；Q抬升9.195954mm、RR joint motion11.104622°|6016–6096，每8ticks，共11端点|
|2|I6536 → Q6539 → 回地6655|I抬升4.140954mm、whole-body motion10.793240°；Q抬升8.481279mm、RR joint motion2.333239°|6544–6648，每8ticks，共14端点|

每次回地各产生 `qualification_revoked_ground_before_cross` 与 `current_lift_revoked_ground` 两种标签，是**同一次物理撤销**，不能加成4次。RR事件总I2/Q2/回地撤销2、current有效25端点、C0/P0。历史 `event_ticks.active_lift.RR=6015` 是首个建立tick；第二次Q6539保存在事件表中，不能用旧首tick覆盖新尝试。

FL在5920 TOP/verified承载5.396426N，6000 TOP/4.165329N；但**25个RR current有效端点全部FL AIR、support=false、bearing=0且verified**，独立其他支撑均FR/RL。功能抬升有body_control_evidence=true，但这只是当前安全与实测支撑条件，不证明静态稳定，也不能虚构FL正在承载。第二次Q的自身joint变化较小仍被承认；不要求先出现大RR关节动作。

## 真实净空/前送与回退

下表front为轮端相对前缘距离，gap为台面净空，均mm；不是由命令反推。

|tick|front / gap|状态与意义|
|---|---|---|
|6016|−231.285 / −39.904|尝试1 current有效，但尚低于台面且离前缘远|
|6072|−313.605 / +53.822|第一尝试最大ground lift104.645mm，却更远离前缘；大抬升不等于carry|
|6088→6096|−315.212 / +13.892 → −299.633 / −33.911|这两个端点之间失去台面净空，6099回地|
|6544|−85.609 / −33.785|尝试2建立后的首记录端点|
|6584→6592|−77.632 / +3.045 → **−63.390 / +3.301**|真实短暂前送且current有效；6592为本段最近前缘，仍未越沿|
|6600|−67.113 / −3.102|已开始退离前缘并失去正台面净空，Q当前仍有效|
|6648→6656|−159.606 / −39.669 → −164.626 / −50.523|轮端持续回退，6655回地撤Q|
|6784|−114.054 / −50.230|GROUND，无当前I/Q/C/P，尚在P09|

6592→6648，`body_forward_m` 从0.142834降至0.126970m（真实body前进量减少15.864mm），轮端front退96.216mm；两者都变动，不能将轮端全部后退单因归为body或某关节。6592线速0.083036m/s、角速0.161014rad/s，支持当时短暂可用功能状态；没有越沿C、没有放置P，不能称受控carry任务完成，更不是完整后缀/自然P01成功。

## N / residual / final / actual：执行继续，但未维持任务结果

RR hip/knee单位°；N为记录raw nominal，R为projected residual，F为final target，A为实测joint-margin回解的规范角。它们不是同一量，也不能仅凭角度变化方向判断轮端升降。

|tick|N|R|F|A|
|---|---|---|---|---|
|6072|52.4 / 0|−4.164 / +5.643|49.486 / 6.597|41.978 / 7.170|
|6096|55.6 / −11.3|−10.733 / +0.590|46.117 / −10.710|46.404 / −2.301|
|6544|−6.9 / −37.8|+4.630 / −1.562|+0.732 / −47.112|−0.008 / −46.684|
|6592|−6.9 / −37.8|−2.499 / +2.465|−19.649 / −44.808|−13.555 / −49.204|
|6600|−6.9 / −37.8|+1.501 / −0.751|−9.649 / −39.801|−13.662 / −44.811|
|6648|−6.9 / −37.8|+13.146 / +15.479|+4.996 / −23.571|+3.623 / −25.621|

第二尝试期间RR nominal保持同一对值，residual、最终target和实际关节继续变化；没有第二次source重播或因Q建立自动重置的证据。6592raw latent RR=[+0.077005,+0.068589]，其投影/继承后的R并不等于原始latent缩放；当拍候选target[-10.648613,−36.584649]与F有最终slew差，不能把N+R直接当落地执行值。所有109记录端点mask全开；RR headroom裁剪仅6472一个端点，不在两次Q建立/回地窗口。

四轮顺序FL/FR/RL/RR。6592 N=[.3,.3,.3,.3]，R=[+.287561,−.559030,+.055434,+.372993]，F=[+.587561,−.259030,+.355434,+.672993]，实测=[+.587246,−.327094,+.601762,+.674745]rad/s：不同方向的动作确实下发并产生响应。6600 N仍全+.3，故净空初次转负并非同拍四轮N清零；6648 N全0、6656 N=[−1.07,0,0,0]发生于同一P09内部，residual/实测均继续非零，不能据此误称普通阶段切换清零。本范围未重读源contract逐条判定这些后续wheel owner事件的时机适配性。

必须保留的控制限制：**6592/6600/6608/6616/6624** 的 nominal几何审计明确为 `degraded_bypass_infeasible_box_downward`，delta=0、`mathematical_contract_verified=false`、`box_vs_downward_halfspace`。例如6592使用源6591的gap+3.802569mm，nominal一阶原始z变化−12.421721mm，给定box内无可行下移约束解；已标明 `physical_motion_guaranteed=false`、`degraded_bypass_is_clearance_guarantee=false`、residual_constraint_applied=false。因此不能声称该段几何保护保证净空，也不能把这个已记录降级分支直接当作新执行损坏或计时强制早降的因果证明。carry诊断仍声明source owners continued、无固定lift timer/15mm gate；建议按时序继续不证明对新状态时机必然合适。

## 正确的非终止尾与证据限度

109条均非terminal；末6784为physical VALID/VERIFIED、termination_reason=null、task_outcome_label=null、time_outs=false、`terminal_bootstrap_allowed=true`。这是外部采集到128整块的尾部，不是新任务失败、done或成功；全部128已保存/1update的账由根快照确认，本报告仅109条事实，不重复计前19条。872/872 native verified且actual-effect、own870（两次handoff hold）、状态写入检查全部零，未发现新增owner/mask/reset/dispatch中断。

末FL恢复TOP并实承载12.042629N，RR GROUND17.267050N，FR/RL AIR；RR可观测其他支持只剩FL，body_control_evidence=false，但bounded motion adjustment仍允许。历史FL放置不替代任何时刻的当前承载，末RR GROUND也不能抹去此前两次真实Q。后续首个未完成任务仍是RR维持有效抬升并前送越沿/放置。两次尝试都在本块一次更新前采得，不能称为已经学习改善，不新增门禁或修改生产。

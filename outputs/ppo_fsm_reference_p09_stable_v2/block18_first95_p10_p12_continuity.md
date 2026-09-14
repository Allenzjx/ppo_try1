# Block18：前95条 P10→P12 固定实训截点

结论：教师RR放置入口有当前TOP承载证据，P10→P11→P12动作连续；前95条策略没有产生新的RL I/Q/C/P。RL出现AIR与前壁接触，但均不等于有效抬升、越沿或放置；当前任务仍未完成，训练继续，不设新门禁。

范围：`train/20260910T1921276505520Z_gd4e46006b382_2f570e3174ba4fe3a1035ca1c6162b8b`。2026-09-10T19:40:49.729Z固定解析 audit **前95完整行/字节[0,8879867)**，global **155009–155103**；未追活动后续、读physical大流或CP。样本来源 **P10=1/P11=1/P12=93**。实际curriculum标记P10/tick**7584**、63.2s、teacher_initialized_suffix、prefix attempt0；第95条到**8344/69.533333s**，新增760physics ticks/6.333333s。教师前缀不计PPO样本；本截点不是已完成整块或已优化95条的证明。

## 教师RR与新RL事实分开

继承教师RR **Q6912/C7109/P7579**，均早于7584信用入口。首个策略末端7592，RR为真实TOP、within_top_xy=true、support/bearing_verified=true、承载**11.234232N**，front+93.749586mm、gap−0.683486mm；7600/7608仍TOP承载13.041386/13.505063N。它不是仅有历史P的假承载。动作前7584的完整物理张量不在这95条中，精确reset/load由根独立receipt核验，不用7592冒充7584传感采样。

RR在94/95端点真实TOP并verified承载；仅**8320 AIR/support=false/0N**，仍within_top_xy/current_lift_valid及currently_usable=true。这里“当前可用”不等于当拍承载，不能写成95帧全程着地。8344恢复TOP、承载12.113061N；这些仍是教师RR历史的后续当前状态，不能计作本段新学出的RR放置事件。

RL首端7592为GROUND/NONE，front−76.475587mm、gap−50.010977mm、实承载2.968851N；当前initial=false、Q/C/P=false。其历史I7/I1722属于教师早期，**信用区间内新I/Q/C/P均0**。95个端点无TOP、initial_clearance均false；12个AIR端点的recent gain最大仅1.573514mm（7704），不能凭AIR记抬升成立。

以下分类由RL原始ground/air/obstacle_pair/contact_surface联合列出，而非另建任务判定：

|RL记录接触组合|端点数|
|---|---:|
|仅GROUND|54|
|AIR|12|
|GROUND + FRONT_WALL|10|
|FRONT_WALL，无GROUND|3|
|GROUND + OBSTACLE_AMBIGUOUS|12|
|OBSTACLE_AMBIGUOUS，无GROUND|4|

最近前缘为**8232：front−47.721353mm、gap−51.103448mm，GROUND+FRONT_WALL**，仍未越沿。该帧support标记true、bearing值2.214032N但**bearing_verified=false**，不得称已证真实承载或用不可靠归一化载荷推断协同。全片段RL load_fraction_valid=false有29端点。8344回到纯GROUND、bearing_verified=true/2.164174N、front−62.818583mm、gap−49.990645mm；第一未完成物理任务仍是P12取得有效RL抬升、保持净空并前送越沿/放置。

## 连续交接与实际动作

7592 P10→P11：entry valid且FR/FL/RR placed历史成立，RL edge_proximity/role_prepared均1；7600 P11→P12：RL edge_proximity/transfer_ready均1。均普通物理目标分支、done=false，不是RL已抬升。两次handoff_hold=true，12通道residual保留、forbidden/scale-clipped为空；最大servo residual差2.67e−15°、wheel2.78e−17rad/s。源owner请求servo跳变5.3°/3.2°不等于residual清零。

RR以外的RL目标也确实改变。RL hip/knee单位°，A由实测role joint-margin按规范零点回解：

|tick|raw nominal|projected residual|final target|actual|
|---|---|---|---|---|
|7592|15.4 / 19.4|−2.509 / +0.488|9.141 / 23.638|14.874 / 19.636|
|7608|15.4 / 22.6|−8.405 / −5.365|3.245 / 15.985|8.219 / 22.186|
|8344|−10.1 / −18.7|−1.081 / −29.095|−13.023 / −49.045|−7.703 / −48.146|

7592/7600/7608四轮N连续为`[-1.07,0,0,0]`，不是阶段切换清零；各轮residual非零并保留。8344源N四轮0但F来自持续residual=`[+.410941,+1.200000,−.999467,−.523271]rad/s`；同阶段后期N=0不能据此说执行链停摆。实际关节响应不等于轮端按某角度方向抬起，也不构成任务成功。

**760/760 native ticks verified且actual-effect、own758**（两次正常handoff），95端点mask全开、dispatch mapping一致、状态写入检查全零。未发现新控制器reset/动作继承/下发缺陷。截止8344 physical VALID/VERIFIED、termination_reason=null、terminal=false、bootstrap_allowed=true；没有安全中止或完成结论。保留运行中的学习结果，不把后续更新/终止计入这个固定截点。

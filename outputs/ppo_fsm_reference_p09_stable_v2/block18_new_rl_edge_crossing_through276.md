# Block18：RL新越沿C9256已核实，但未放置（截止第276条）

结论：**同一P10教师入口后的真实PPO动作产生RL新I4/Q4/C1/P0**，不是教师继承C。第四次尝试在9256越沿，之后轮端与body均有真实回退，P始终false；9736历史C为true不意味着当下仍在前缘上方或已放置。没有将RL不存在的RR专用current_lift_valid/established字段补成false。

固定范围：run `train/20260910T1921276505520Z_gd4e46006b382_2f570e3174ba4fe3a1035ca1c6162b8b`；2026-09-10T19:44:55.987Z只解析audit **行96–276、字节[8879867,26977002)**，前95未重读、原报告不改。181条global **155104–155284**，端点8352–9792，全部P12、无terminal/阶段切换，1448新增物理ticks。所有curriculum标记相同：P10/tick7584、prefix_attempt_index0，tick/decision连续，没有新episode/reset；首新行RL仍无历史Q/C/P，末行C事件tick新增9256。

## 新的主动抬升与越沿事件

|尝试|I / Q事件tick|Q的测量证据|随后|
|---|---|---|---|
|1|8369 / 8374|upward excursion8.364011mm，RL joint motion4.755118°，当时top gap−42.024013mm|8633地面接触撤Q|
|2|8819 / 8822|8.308042mm，3.148181°，gap−42.023976mm|9019地面接触撤Q|
|3|9042 / 9051|8.542458mm，7.328636°，gap−41.455145mm|9111地面接触撤Q|
|4|9149 / 9153|8.442027mm，10.000022°，gap−41.668375mm|**9256新C**，截至9792无P|

I事件还保留whole-body实际动作与命令变化证据，例如第四次whole-body motion31.458804°、command motion42.264630°；不是因AIR标签或仅发出命令便认可抬升。RL事件采用通用`active_lift`历史、`active_attempt`、`initial_clearance`及事件表。`event_ticks.active_lift.RL=8374`只记首次Q，不能拿它覆盖后来的8822/9051/9153新尝试。三次地面撤Q是三个事件，不套用RR专有双标签计数。

**C9256/global155217（本回合第209次策略决策，77.133333s）**：RL `active_attempt=true`、initial=true、history Q=true、AIR、ground=false、within_top_xy/lateral=true、crossing_evidence_status=VERIFIED；front **+0.675026mm**，top clearance **+43.959226mm**，近期RL动作23.562339°、whole-body动作133.986809°。因此有实际主动尝试与越沿几何，不是教师继承C，也不是仅把前壁接触标签当作越沿。当拍RL仍无TOP接触、support=false/0N；不能称放置或全任务成功。全局load_fraction_valid=false不推翻这组独立几何/动作证据，也不能被用来编造承载分配。

## C后真实回退与未完成的放置

|tick|RL front / top gap（mm）|当前事实|
|---|---|---|
|9248|−7.839557 / +57.767469|还未C|
|9256|+0.675026 / +43.959226|首次C，AIR未放置|
|9272|**+16.005400 / +12.891061**|本固定范围最向前端点，仍无TOP/P|
|9296|−8.007642 / +22.528302|C后首次记录回到前缘后侧，历史C保留|
|9656|−137.821880 / −1.769515|C后首次记录top gap转负|
|9736|−158.899870 / −36.900571|AIR、无TOP/P，当前几何不再满足放置位置|
|9792|−136.310216 / −13.879320|仍AIR、active_attempt=true、历史Q/Ctrue/Pfalse|

9256→9736，body_forward_m从0.149439696降至0.009266261m（后退**140.173435mm**）；RL轮端front净退**159.574896mm**。因此“越过后又在前缘后”有真实整体运动及局部构型变化证据，不是C事件凭空消失。过程不单调，不能把全部轮端变化只归于body或一个关节。C之后本范围没有RL地面接触端点，也没有TOP端点，故不是“放置后再次落地”；当前首个未完成任务是重新取得可放置的真实位置并受控接触台面。

RL hip/knee的raw N在下列两个端点均为`[-10.1,−18.7]°`，R/F/实测A继续变化：

|tick|projected residual|final target|实际规范角（joint-margin回解）|
|---|---|---|---|
|9256|−20.595829 / −14.368880|−32.424021 / −34.318880|−30.184290 / −37.579531|
|9736|−16.865463 / −17.144870|−27.543225 / −58.000000|−25.349131 / −56.962625|

9256四轮N全0，但F=`[-.761645,+.294166,−.121770,+.409610]`rad/s，实测=`[-.796935,+.339067,−.121458,+.398301]`。9736 N也全0，F=`[-.469923,−1.094554,+.610294,+.546131]`、实测=`[-.641551,−1.092220,+.611327,+.425069]`。9296/9600中间N还曾全+.3。动作和响应未停，不能把同一P12内N为0误称跨阶段重置；这些测量也不足以将回退单因归于nominal或某次residual选择。

## RR历史放置不等于持续当前TOP承载

前95报告已区分RR8320短AIR与8344恢复TOP。本新增范围：8352–8376又AIR/0N；8384短暂合格TOP承载5.308388N。**8392–8560仍有真实TOP表面反力/support/bearing_verified=true，但within_top_xy=false、任务top_contact=false**，不能把承载全写成丢失，也不能以surface=TOP冒充仍满足放置几何。8568AIR/0N；8576开始记录GROUND实承载14.825277N；8648首次记录GROUND_AND_OBSTACLE、bearing_verified=false（虽数值10.380219N，不能当已验证承载）。此后有地面与混合接触交替，不是全程同一模式。

C9256当拍FL TOP已验证13.325760N、FR TOP已验证0.768960N；RR是GROUND_AND_OBSTACLE、top=false、bearing_verified=false，不能把其数值7.571862N算成确定TOP支撑。9736 FL TOP已验证8.548618N，FR AIR/0N，RR仍混合接触且承载归属未知；9792同类RR状态。RR教师P7579历史保留，并不说明后来维持了可用顶部放置，更不能将其算成本段新RR成功。

## 截点与结论边界

181条/1448ticks均native verified且actual-effect、own1448，mask全开、native dispatch/mapping一致、状态写入检查零，没有新阶段交接或reset。9792为VALID、CONTACT_BEARING_UNVERIFIED、无termination、done=false、time_outs=false、bootstrap_allowed=true；训练仍运行，不记成终态失败或成功。这里确认的是教师前缀之后策略真实产生的新RL越沿事件及后续未放置，不是固定策略评估、完整后缀成功、自然P01成功或由某一次PPO更新导致改进的证明。

# 第9机会：短暂历史placed，不是当前捕获成功

固定快照上界227,176,139 bytes；只解析第9机会byte[173,109,593,227,159,151)，372个active样本，未读第10前缀。末10948/91.233333s/P11为INCOMPLETE_CONTROLLER_BLOCKED，local/full均false；372端点合法TOP/bearing均0、记录hold均0。

## 历史placed从哪里来

RR历史placed记录在8218，首次在8224决策端可见。但8224已AIR/0N、consecutiveAIR=5、free_air_reference_tick=8219。所选物理判定器只在top_surface接触、非GROUND且合法XY时累计TOP；minimum_top_samples=2。因此历史事件结合前后端点支持8217–8218至少两次短暂native TOP接触，随后即丢失。它不是凭P11标签伪造的历史，但也不是0.5s可用承载。该endpoint日志没有这两tick实际force或hold峰值，不能把max_logged_hold=0升级成物理上绝无瞬时触碰。

| tick / phase | gap / front mm | TOP / ground / eligible | source / mappedN / generic RR° | FINAL / actual RR° |
|---|---|---|---|---|
| 8216 / P09→P09 | -0.910 / -0.099 | False / False / True | [-6.900, -37.800] / [-8.150, -39.050] / [+0.000, +0.000] | [-4.336, -21.566] / [-6.629, -24.374] |
| 8224 / P09→P09 | -0.798 / -1.996 | False / False / True | [-6.900, -37.800] / [-8.150, -39.050] / [+0.000, +0.000] | [-0.336, -25.566] / [-4.525, -24.230] |
| 8240 / P09→P10 | +4.129 / +3.114 | False / False / True | [-6.900, -37.800] / [-8.150, -39.050] / [+0.000, +0.000] | [+0.657, -33.566] / [-2.481, -28.810] |
| 8248 / P10→P11 | +8.577 / -3.035 | False / False / True | [-6.900, -29.300] / [-8.150, -34.600] / [+0.000, +0.750] | [+4.157, -31.866] / [-0.145, -30.001] |
| 8256 / P11→P11 | +3.260 / -18.472 | False / False / True | [-6.900, -27.200] / [-8.150, -27.200] / [+0.000, +0.000] | [+7.657, -21.866] / [+2.753, -27.847] |
| 8408 / P11→P11 | -49.759 / -48.847 | False / False / True | [-6.900, -27.200] / [-8.150, -27.200] / [+0.000, +0.000] | [-0.460, -20.863] / [-2.309, -15.546] |
| 8416 / P11→P11 | -51.198 / -48.830 | False / True / False | [-6.900, -27.200] / [-8.150, -27.200] / [+0.000, +0.000] | [-4.460, -24.863] / [-1.662, -18.223] |
| 10948 / P11→P11 | -50.202 / -80.595 | False / False / False | [-6.900, -27.200] / [-8.150, -27.200] / [+0.000, +0.000] | [+8.488, -22.490] / [+8.381, -27.926] |

P09→P10在8240发生，completion_values仅placed_RR=1。8216→8224→8232→8240期间RR source[−6.9,−37.8]、mappedN[−8.15,−39.05]和generic0未变，但FINAL knee从−21.566依次回折至−25.566/−29.566/−33.566°；这是本次sample/执行限速后的实际请求变化。首次接触丢失在P10以前，不能归为P10提前展开造成。

第一拍P10到8248：source knee−29.3、mapped−34.6（本回合不是旧例的−32.1）、generic+0.75，FINAL knee−31.866，比上一拍展开1.700°，actual仍因跟踪/物理滞后进一步回折至−30.001°。8256进入P11接续source/mapped knee−27.2，FINAL展开到−21.866°，此时轮心已退至前缘−18.472mm、失去合法XY。动作和载荷有耦合，本表不能指定某个单轴为以后掉地的唯一原因。

## 第一次GROUND与资格保护

首次native current_lift_revoked_ground=8414；最近决策端8408仍非GROUND、eligible=true，8416已GROUND、eligible=false。该时刻比第一P10动作晚约1.4s，RR在XY外，gap≈−51.198mm，并有OBSTACLE_AMBIGUOUS边缘/立面pair。掉地后继续保留active任务和历史cross/placed，但历史不替代当前资格或支撑。

本回合GROUND端点共64，首次GROUND后eligible=true端点192。首个重新eligible端点8552：free-air reference8536、连续AIR16tick、实测unsupported_free_lift=11.929mm，current_lift_valid/lift_established均true，但仍XY外、TOPfalse/0N。这不同于只沿用旧placed，不能说掉地后资格一直为false，也不把单个initial_clearance事件自动当新qualified。末当前TOPfalse/0N、XY区域外75.595mm、hold0，故没有本地成功。P09 late始终未消费，未启动P12新卸载。

边缘短接触允许连续处理，不等同跳过主动抬升的纯wheel爬升；本次有既有真实RR资格，最后是普通任务未完成，不追加未被观测证明的纯wheel违规。局部成功仍要求当前合法TOP+真实bearing+本次有效抬升谱系+连续0.5s，阶段P11和旧placed都不能单独满足。

窄代码依据：semantic_supervisor.py的loaded/TOP计数与placed；semantic_rr_capture_local_task.py的currentTOP/current-attempt和连续hold；当前task_spec的minimum_top_samples=2。未重扫全仓库或原始native日志，未修改运行/生产或创建AUX标签。

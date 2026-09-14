# Video10：P05 未取得 FL 有效抬升，有限恢复期限耗尽

等待两份最终清单完成后才读取：run `video_eval/validation/20260910T2329210669061Z_gd4e46006b382_b1afaf16892b450b81d986e7148862d0` 的run_manifest完成时间为2026-09-10 23:44:57.680495 UTC。随后在23:46:27.187–23:46:27.432 UTC **一次**扫描封存 `source/video_policy_decisions.jsonl`，664行、43,650,044 bytes。除最终manifest外未打开巨型physical/quality/native流文件，未读后续训练、checkpoint或重跑仿真。

## 真实终态

自然P01、seed4001，optimizer updates=0；664 issued全部正常returned，663×8+末5=**5309 ticks/44.241667s**。阶段样本P01=2/P02=169/P03=10/P04=22/P05=461。终态P05 `INCOMPLETE_CONTROLLER_BLOCKED` / **`LOCAL_BOUNDED_RECOVERY_EXHAUSTED`**；共同物理任务验收未通过，lifecycle=`DIAGNOSTIC_FAILURE`，不是成功视频。

物理evaluator自身termination_reason/source均null，run VALID、`CONTACT_BEARING_UNVERIFIED`。本次不是BODY_COLLISION或安全中止；未碰撞但未完成任务仍是任务未完成。

## 第一未完成任务与四腿事件

- **FL全程I0/Q0/回地撤销0/C0/P0**，不是仅凭末态false推断：全部历史事件逐次去重检查，无FL初始/资格事件；initial_clearance与active_attempt在664个端点均false。11个短AIR端点不能代替有效抬升，记录最大recent_clearance_gain仅0.584277mm。
- FL端点最高台面净空仍−49.957580mm，最靠前仍−47.670182mm（两者不是同tick）。第一未完成任务为**FL有效主动/全身协同抬升并维持净空**，越沿与真实放置亦未发生。
- FR新I16（P01）/Q25（P02）/C1380/P1441（P03）。末态FR实际TOP、bearing_verified=true、bearing=5.715072N；不能把总载荷比例未知等同于独立FR顶部接触不存在。
- RR全程无I/Q/C/P、current Q始终false。末RR GROUND、front−615.728055mm、净空−50.445124mm。另见RL I7、I1412，无RL Q，不能误算为RR进展。

末FL为GROUND+`OBSTACLE_AMBIGUOUS`，front−48.501101mm、净空−51.608694mm、top_contact=false、within_top_xy=false。虽然support字段为true、bearing_force=11.961435N、load_fraction=0.324477，**bearing_verified=false且load_fraction_valid=false**，不能将数值认作可信顶部承载或载荷分配。接触分类限制保留；独立几何也未显示达到净空或越沿，因此不能只把失败归为传感器标签问题。

## P04→P05 Full12 连续性和执行链

实际切换P03→04=1448，P04→05=1624，入口均valid且普通切换未终止。所有四次阶段交接的下一拍审计都显示**全12通道previous/carried residual一致**，forbidden/phase-scale-clipped列表为空，已有handoff hold生效。

1616/1624/1632的nominal四轮均+0.3。1624 residual四轮=[−0.451916,−0.253257,+0.257756,+0.480225]、final=[−0.151916,+0.046743,+0.557756,+0.780225]；1632继续按新采样变化，没有普通切换清零wheel/residual证据。这里是实际记录的决策端点与交接证据，不虚构逐物理tick命令曲线。

5309 native ticks全部verified/native effect有效，own-phase effect=5305；每决策setter/dispatch和mapping一致，全部无episode内状态写入。P05 FL hip/knee mask开放461/461，headroom clipping=0。

末FL N=[22.8,−13.4]°、R=[3.232420,+21.608639]°、final=[26.284926,9.458639]°；由实测joint-margin回解canonical actual≈[25.969422,7.840684]°。实际跟踪这些命令并没有形成有效抬升。没有确认新的mask/owner清零或native下发缺陷，但不把正knee residual、末端跟踪差或接触不确定性任一项认定为唯一原因。

## 不是固定录像截断

P05进入1624/13.533333s；5304时age=30.666667s尚未终止。5309时age=30.708333s，超过当时 **30+0.702902442=30.702902442s** 的有效期限，progress=0.265123074、fixed_post_window_allowance=0、terminal bootstrap=false。末5ticks为正常返回的真实有限期限终态；`interrupted_final_decision_ticks=0`不代表末步执行8ticks。

最终视频664帧，正常速度编码44.266667s，相对物理时长多0.025s为末帧显示量化；无额外物理ticks或PRE/POST帧。文件封存/录像返回与任务成功分别判断。原失败保留；本报告不改变生产、checkpoint、主报告、任务/安全标准，也不增加续训门禁。

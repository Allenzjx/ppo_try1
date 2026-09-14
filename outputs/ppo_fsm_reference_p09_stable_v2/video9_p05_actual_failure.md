# Video9：FL 短暂抬升后回地，越沿未完成

仅一次流式解析已完成 run `video_eval/validation/20260910T2235582002787Z_gd4e46006b382_0f64bf3b35f74953994f8ca191952ee8/source/video_policy_decisions.jsonl` 的631行、42,148,194 bytes（2026-09-10 22:52:01.957–22:52:28.923 UTC）。不读取block22、checkpoint或完整physical/quality/native流；checkpoint159104的加载身份及质量指标由主控核验。

631次issued均正常returned：630×8+末5=**5045 ticks/42.041667s**。阶段样本 P01=2/P02=140/P03=7/P04=21/P05=461；终态P05 `INCOMPLETE_CONTROLLER_BLOCKED`。正常return不是任务成功；主控记录的source共同物理验收未通过及lifecycle `DIAGNOSTIC_FAILURE`不应改写。

## 真实 I/Q/C/P 与当前接触

- FR：**I15（P01）、Q24（P02）、C1147/P1188（P03）**；末态仍TOP、bearing_verified=true、bearing=6.171N。
- FL：**I1429、Q1434、回地撤销1472，均P05；C/P始终false**，之后无新I/Q。Q1434事件实测向上变化9.027936mm、台面净空−40.944224mm，不可因为末态Qfalse写成全片从未抬起。
- FL只有2个AIR决策端点；1432 AIR，1440–1464为`OBSTACLE_AMBIGUOUS`、bearing_verified=false，1472出现GROUND并撤销。全部决策端点最靠前仍−48.413780mm，最高台面净空仍−41.522834mm；端点极值不是同tick，也不能替代物理事件tick。没有越沿或顶面放置证据。
- 末5045 FL为GROUND+`OBSTACLE_AMBIGUOUS`、front−48.735258mm、净空−50.019038mm、top_contact=false、within_top_xy=false。虽字段support=true、bearing_force=7.876735N、load_fraction=0.225821，但**bearing_verified=false、load_fraction_valid=false**，这些数值不能证明可信承载分配。FR直接TOP/bearing证据仍成立，不能因总载荷比例未知把独立接触证据全清空。

最早始终未完成的后续物理任务是**FL保持净空并前送越沿，继而放置**；终态还需重新取得当前抬升状态。传感归类存在承载不确定性，但“前缘始终未越过”的几何证据独立存在，不把此结果简单归为传感器故障。

## 连续交接与实际下发

P01→02=16、P02→03=1136、P03→04=1192、P04→05=1360，各次入口valid、未因普通切换终止。所有四次下一拍jump审计显示全12通道previous/carried residual完全一致，forbidden/phase-scale-clipped均空；既有handoff hold生效。1352/1360/1368 nominal四轮均+0.3，residual保持非零并连续更新，没有P04→05 wheel清零证据。

5045 native ticks全部verified/native effect成立，own-phase effect=5041；各决策setter/dispatch及mapping一致，无episode内状态写入。P05 FL hip/knee mask开放461/461，FL两通道headroom clip=0。末态FL N=[22.8,−13.4]°、R=[1.663488,+21.109379]°、final=[24.618407,8.959379]°；由实测关节margin回解canonical actual≈[24.422547,7.812419]°。N不能跳过mapper/geometry与残差直接当实际关节角。末端跟踪接近也未产生有效净空/越沿；不能仅凭正knee residual认定单一失败原因。

所读证据未确认新的mask封锁、residual重置或native下发缺陷。它证明这个checkpoint的该次实际闭环轨迹未完成任务，不证明所有失败原因已经被排除。

## 真实有限恢复期限，不是视频固定截断

P05于1360/11.333333s进入。5040时stage age=30.666667s，未终止；5045时age=30.708333s，超过当时 **30+0.700731308=30.700731308s** 的effective limit（progress=0.264713299；fixed_post_window_allowance=0）。source明确为`LOCAL_BOUNDED_RECOVERY_EXHAUSTED`、`finite_task_terminal_not_external_truncation`，terminal bootstrap=false；末决策实际执行5ticks后正常返回，不是200秒视频包装提前切断。

末物理evaluator自身termination_reason/source均null，run VALID、`CONTACT_BEARING_UNVERIFIED`；不是BODY_COLLISION或安全中止，但无碰撞不等于越障成功。原P05任务未完成结果保留；不调整奖励、标准、参数、生产代码或训练门禁。

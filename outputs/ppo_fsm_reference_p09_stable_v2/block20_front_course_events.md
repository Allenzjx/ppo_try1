# Block20 前腿课程：固定 512 条事件核验

仅一次读取已完成 run `train/20260910T2117208453706Z_gd4e46006b382_85a629fc55c0447bb5b517ff2dbc278b/residual_and_projection_audit.jsonl` 的 43,606,526 bytes、512 行（2026-09-10 21:35:52.717–21:36:34.330 UTC），global 157569–158080；未读取 block21、checkpoint 或大物理流。

| 回合 | 策略样本 / 分阶段 | 策略物理 ticks | 结尾 |
|---|---|---:|---|
| ep0 | 496；P03=7/P04=5/P05=484 | 3965 | tick5621，P05 `INCOMPLETE_CONTROLLER_BLOCKED`，`LOCAL_BOUNDED_RECOVERY_EXHAUSTED` |
| ep1 | 16；P03=9/P04=7 | 128 | tick1784/P04，非终止采集尾，bootstrap=true |

两回合均从真实教师 P03/tick1656 接管；前缀不计策略信用。全块 4093 native ticks 均 verified，其中 own-phase request effect=4090；所有决策无 episode 内状态写入。不是从自然 P01 的完整策略成功。

## FL：确实多次抬起，但没有越沿和放置

按 `(episode, leg, event, physics_tick)` 去重，并排除 credit 起点前教师事件：FL **I=19、Q=16、回地撤销=16、C=0、P=0**，全部新事件在 ep0/P05；ep1 无新 FL I/Q/C/P。首 Q1940，末 Q5291；末次在终态 tick5621 回地撤销。不能把末态 active=false 写成全程从未抬起。

全范围 FL 最靠前的决策端点仍为 −32.545 mm，虽然另一个端点的最大台面净空达到 +83.047 mm，二者不是同一时刻，且均不能代替 C/P。终态 FL GROUND、front −61.092 mm、clearance −50.313 mm，实测地面 bearing=14.314 N（verified），不是顶面放置或顶部承载。首次始终未完成的后续任务是有效越沿、继而放置；末态还已失去当次抬升资格。

## RR：唯一 Q 在 P05，不是 rear crossing

RR 新 I1927 → **Q1934（P05，line35/global157603 所覆盖物理区间）** → 回地撤销1959；第二 I4435 后未建立新 Q。总 I2/Q1/C0/P0，`qualification_revoked_ground_before_cross` 与 `current_lift_revoked_ground` 同属 tick1959 的一次回地，不重复计数。

Q1934 事件记录真实向上变化 8.277498 mm、台面净空 −42.291495 mm、目标腿 joint-motion proxy 3.145214°。最近保存的决策端点如下（并非精确 Q tick 的原始接触流）：

| tick | RR contact / current Q | 相对地面抬升 | RR front / 台面净空 |
|---:|---|---:|---:|
| 1936 | AIR / true | 9.664 mm | −627.682 / −40.905 mm |
| 1944 | AIR / true | 12.359 mm | −642.924 / −38.210 mm |
| 1952 | AIR / true | 7.350 mm | −657.759 / −43.219 mm |
| 1960 | GROUND / false | 0 mm | −670.571 / −50.326 mm |

这三个有效端点 FL 都 AIR、bearing=0、support=false；真实其他支撑是 FR TOP 与 RL GROUND，均 verified。以1936为例，FR/RL bearing=15.403/15.979 N，RR 自身/全身近期关节运动 proxy=3.212/29.485°，body_control_evidence=true。这是全身响应与当前支撑证据，不证明 FL 承载，也不证明已经形成受控越沿轨迹。全块 RR 最靠前端点是更早的 tick1912/GROUND，front **−601.245 mm**、clearance −50.985 mm。RR 在远离前缘处短暂取得 Q 后回地，C/P 从未成立。

## 首前驱交接

ep0 P03→P04 在1712，P04→P05 在1752；都 done=false。下一拍 jump audit 两次均显示 previous residual 与 carried residual 全12通道相等，forbidden/phase-scale clipped 列表为空，沿用已有 handoff hold，无 residual 清零。P04→P05 前后记录的 nominal 四轮均 +0.3；tick1752 residual 四轮为 [−0.028845, −0.055771, +0.444874, +0.325027]，正常继承后继续变化。这里证明已记录端点和交接审计连续，不虚构未读取的逐物理 tick 命令曲线。没有据此新增执行缺陷或训练门槛；保留原任务未完成结果，生产、语义和参数均未改。

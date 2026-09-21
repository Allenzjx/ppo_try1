# Block12 首次随机捕获：发生在本块首个 optimizer 更新前

仅查 run `20260918T0909562623513Z_g3a50657a96c9_d0f934ba1eae46958f7c8b880fdac987` 的首个完整 prefix、前86条 learner（截止tick3896）及第一个 optimizer 回执；无模型前向/额外优化/仿真。

**实际 CP183552 前缀401决策到 tick3208/P05，accepted=true、miss=null、无 fallback。** 全部 prefix credit=false，learner 明示 prefix 不入 storage；第一条真实信用是 global183553，raw HISTORY 精确继承。真实 FL placed 在 **tick3748 /31.233333s**，落在第68条 learner/global183620；P06在3752接管。首 optimizer1400 到 global183680（128条）才执行，其更新前 actor hash 与源CP183552相同。因此捕获前新增 optimizer=0，不能称这是本批更新学出的均值能力。

已有同CP确定性评估 tick3208 的 gap24.990mm/AIR；本次 prefix 末 FL raw/final 与该记录相同。prefix回执本身缺少该tick净空字段，不能冒称独立读到了25mm；本次**首条真实 learner 终拍3216**已记录 gap23.497mm/0N。所恢复的是当前约25mm悬空入口，不是旧3mm探针，也不是从已 placed 快照起步。

下表 hip/knee 均按 FL 顺序；μ/sample 为 raw空间，REQUEST/final/actual为canonical °。4个代表行的 REQUEST=有效残差，实际派发验证均通过、无headroom clip。

| 终拍tick | conditional μ H/K | sample H/K | REQUEST H/K ° | final H/K ° | actual* H/K ° | FL gap mm / 承载N |
|---|---|---|---|---|---|---|
| 3216 | +.24131/−.06357 | +.18354/−.05581 | +3.267/−1.338 | 25.475/−13.488 | 26.571/−13.594 | +23.497 / 0 |
| 3744 | +.07510/−.04543 | −.24064/+.02475 | −2.933/+.594 | 16.476/−11.556 | 21.857/−11.840 | +1.677 / 0 |
| 3752 | −.19488/+.01176 | −.34677/−.12697 | −6.003/−3.031 | 12.156/−15.181 | 18.899/−12.260 | −1.647 / 11.434 |
| 3896 | +.31319/−.10890 | +.30156/−.15612 | +9.368/−5.575 | 27.527/−17.725 | 27.773/−16.439 | +2.450 / 0 |

*训练日志的 actual 来自**每条最后一次 dispatch 前**的原生关节读回，表中canonical值由同拍记录的 `nominal−current_actual_canonical_error` 推得；不是终拍contact/gap时刻的post-step q。原生rad及时间语义保存在JSON；post-step actual q缺失，未补造。

捕获所在动作的 base μ 仍为 hip +.217008/knee−.105093；conditional/sample因真实历史及随机创新不同，不能把负sample等同于负base均值。3752时FL TOP、其余实测承载为FR4.809/RL2.647/RR12.290N；与首learner时FR12.245/RL14.534/RR1.890N明显不同，**是全通道/全身闭环过程，不是单hip因果试验**。3896时FL已重新AIR/0N，历史placed仍真，不能声称持续承载或全任务成功。

结论仅为：**当前 checkpoint 的随机闭环可以从这个真实未捕获入口恢复接触，且发生于本块任何学习更新之前。** 首更新后来已完成，但不改写上述时序；当前确定性均值是否改善仍须其独立评估。本helper已自然退出，无生产修改。

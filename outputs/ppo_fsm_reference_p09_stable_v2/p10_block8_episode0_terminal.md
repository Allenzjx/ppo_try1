# P10 block8 episode0：456 决策终局补充

这是 `p10_block8_first110_control.md` 之后的独立、有界终局报告，不修改其前110条固定片段结论。结果仍为 **P12 未完成**；后段确有4次 RL Q，但全部重新触地撤销，没有 C/P，没有 P13，也不是完整后缀或自然 P01 成功。

## 固定范围与真实信用

Run `runs/ppo_fsm_reference_p09_stable_v2/train/20260910T1036035708863Z_g69aeaca777dc_1c97c4bee3f548da83b7860abcf8d541`，生产69aeaca。只读 audit 前456行（episode0），global145409–145864；固定 bytes `0–45360334`，45360335 bytes，SHA256 `447b91a5bd0f7e6e4fc2356d1f899fdece3ce57eaf98f16be67ee8c4410a238b`。不解析第二前缀或后续策略行。

P10=1/P11=1/P12=454/P13=0，真实策略物理tick=3647（教师 credit-start7584至terminal11231），native effect 3647/3647 verified，无 forbidden state writes；own-phase request3645，其余2个为已有普通 handoff hold。教师 RR Q/C/P和教师 RL I7/I1722均零新增策略信用。

此终态时 **456 sampled / 384 optimized**；optimizer log只有更新1102/1103/1104，分别在global145536/145664/145792，每次20 steps，有限非零梯度与actor改变均已记录。保存点依据主控快照为145792/1104/22080；最后72个样本（global145793–145864）虽已持久化audit，但尚未优化，不能称为已入checkpoint。本报告不计未来更新。

## RL 新事件：末端 false 不代表全程未 Q

新 I 共11次：7954、8311、8743、9121、9155、9577、9745、9883、9978、10105、10728。前两次已在前110报告中记录，不重复增加总账。

| 新 Q tick / 秒 | 同尝试随后地面撤销 tick / 秒 | Q 时实测自身关节运动 |
|---|---|---:|
| 9754 / 81.283333 | 9869 / 82.241667 | 2.614° |
| 9891 / 82.425000 | 9947 / 82.891667 | 4.928° |
| 10111 / 84.258333 | 10205 / 85.041667 | 3.294° |
| 10735 / 89.458333 | 10756 / 89.633333 | 6.047° |

对应 I 的全身实测运动为6.583°/16.031°/5.482°/7.891°（I与Q不是相同tick，不相加）。后段有36个决策末端 `history.active_lift.RL=true`，最终全部撤销。历史 `event_ticks.active_lift.RL=9754` 只是首次Q时间，不能当成终态仍Q。

最大台面净空曾达 **+1.550mm**（row283/tick9848），但同刻前缘距离 **−88.413mm**，未越沿。整个episode最近前缘仍为前110中tick8376的−47.661mm。新Q表明后来出现了不同的真实抬起状态，不能只沿用前110“尚无Q”的结论，也不能把净空、AIR或Q等同于后腿完成。

## 后段运动与执行链

第111–456行共346决策：RL GROUND262、AIR74、FRONT_WALL11（接触统计可重叠），load未知115。RL两关节mask关闭0、headroom裁剪0、RL captured-owner suppression0。没有新增明确下发/owner缺陷；既有P12 nominal后段[-10.1,−18.7]°下，residual与实际全身响应能产生Q，但没有维持并完成前送放置。

真实 body_forward 相对障碍几何：row110 +.100106m，row256 +.103653m，row384 −.010509m，终态 +.079154m。因此不是全程单调后退：中段后退后，末72决策又前进约89.663mm；但相对row110仍净退20.953mm，相对策略首行净退227.064mm。此处使用body几何，不以CoM替代。

终态 RL nominal[-10.1,−18.7]°、residual[−19.052,+3.543]°、final target[−29.553,−16.407]°、实测[−28.939,−14.057]°。实际角度依据同tick FR角色接收侧RL的关节margin反算，不由命令推导。四轮nominal均0，final targets FL/FR/RL/RR=[−.238495,−.519606,+.641628,+.472959]rad/s，实测[+.013729,−.519048,+.626905,+.272614]；不能只凭轮子转向或追踪误差归结唯一成因。

## 最终未完成条件

terminal tick11231 /93.591666667s，P12 age30.258333333s；local nominal limit30s，current-progress allowance .252950277512s，effective limit30.252950277512s。age已经超过有效期限，source=`LOCAL_BOUNDED_RECOVERY_EXHAUSTED`，reason=`INCOMPLETE_CONTROLLER_BLOCKED`，是有限任务终止而非外部截断。没有启用固定悬停计时，也不以历史峰值进展持续续期。

末端 physical evaluator valid=true/status=VERIFIED，RL GROUND、contact_surface=NONE、front−109.118mm、台面净空−51.253mm、有效load .108971，当前Q/C/P均false。终态不是BODY_COLLISION或独立安全中止；没有安全事故并不等于任务成功。首个未完成任务仍为P12的RL有效抬起/前送越沿/受控放置。

reward breakdown total=−43.230024938（写入训练标量float32为−43.230026245），其中terminal event=−40、potential shaping=−3.227039422；terminal bootstrap=false。更新跨越此episode，不能将后段Q直接归因于学习改进，也不能据此宣称FSM比较结论。

本次只新增本报告；未改前110报告、生产、配置、checkpoint、进程或主manifest，未运行新Isaac。

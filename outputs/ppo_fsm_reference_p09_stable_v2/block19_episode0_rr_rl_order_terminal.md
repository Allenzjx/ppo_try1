# Block19 ep0：自然P01的RR新C与RL顺序中止（固定首580条）

结论：自然P01轨迹实际完成FL Q/C/P，并在P09反复尝试后记录 **RR C4615**；25个物理ticks后，**4640因“RL crossed before RR placement”中止**。它不是“RL比RR更早越沿”：RR已先C，但尚无P。RL末帧确有TOP接触/承载及前缘正距离，然而现行判定没有把RL C/P写入历史。报告保留原 `INCOMPLETE_CONTROLLER_BLOCKED / RR_FIRST_ORDER`，不预判顺序约束合理性、不修改规则或结果。

范围：run `train/20260910T2021542525340Z_gd4e46006b382_6707fa943a204563ab1699572bde18a9`，2026-09-10T20:36:12.181Z固定解析audit前580行、字节`[0,52484848)`，completed首行`[0,107296)`；未读第二回合、physical大流、CP或Torch。episode0/seed1001，global **155521–156100**，天然P01首个动作0→8，无教师curriculum字段。580决策/4640ticks/**38.666667s**；source counts：P01=2、P02=96、P03=7、P04=1、P05=51、P06=55、P07=1、P08=1、P09=366，P10–13=0。

## 自然前驱与跨阶段连续性

FL **I910/Q915/C1222/P1256**，均本回合真实产生。Q915在P05：upward excursion8.139658mm、joint motion8.003566°。C后1224仍AIR/0N；1256才真实TOP/verified承载29.689896N并进P06，1264承载降至3.317294N，不能把瞬时高载荷当长期稳定。

阶段边界tick：P01→02=16、P02→03=784、P03→04=840、P04→05=848、P05→06=1256、P06→07=1696、P07→08=1704、P08→09=1712。RR **Q1691在P06建立**，1696/1704/1712 current有效连续接管；1696 rear_approach仅0.501375，仍按当前有效抬升接管，不退回固定入口。FL在1696/1704 AIR/0N，1712恢复TOP/verified3.948287N；RR抬升并非虚构FL始终承载。

8次普通handoff全done=false、hold=true，全部12通道residual保留、forbidden/scale-clipped为空，最大servo residual差1.43e−14°/wheel1.12e−16rad/s。没有普通切换清零或回报切断证据。全4640 native ticks verified且actual-effect、own4632（8次正常handoff），native dispatch/mapping一致、状态写入检查零。

## RR与RL的真实尝试（事件tick，不把AIR当Q）

RR总 **I7/Q5/回地撤Q4/C1/P0**：

|RR I → Q|所在阶段|撤销/结果|
|---|---|---|
|910，仅I|P05|未建立Q|
|1676 → 1691|P06|2014于P09回地撤销|
|2622，仅I|P09|未建立Q|
|2865 → 2875|P09|2888回地撤销|
|3831 → 3858|P09|3906回地撤销|
|4368 → 4373|P09|4474回地撤销|
|4550 → 4610|P09|**C4615**，没有回地/放置|

四次RR地面撤销各有qualification-revoked与current-revoked两种标签，同tick只计一次。最后Q4610实测upward excursion106.757771mm、RR joint motion43.131571°、top gap+60.423001mm；4608虽然已AIR且gap+54.616699mm，但body_control_evidence/current仍false，不能据大高度提前把Q记到4608。4616/4624/4632 current=true、history Q/Ctrue/Pfalse；末4640 current=false与整帧顺序abort相伴，RR仍AIR且无新回地事件，不应当成第五次地面撤Q。

RL总 **I8/Q4/回地撤Q3**：独立I9(P01)、I886(P05)、I2532/I4069(P09)均无后续Q；其余在P09为 **I2166/Q2181→撤2288；I3054/Q3061→撤3097；I3265/Q3274→撤3457；I4506/Q4506**。最后一次I/Q同tick：upward excursion15.335095mm、RL joint motion10.785907°、whole-body motion62.243459°。这些是通用RL历史/事件，不造RR专属current-lift字段。

## RR先C；RL实际顶部接触，但C/P未获登记

|记录tick|RR front / gap（mm）|RL front / gap（mm）|判定|
|---|---|---|---|
|4608|−5.385693 / +54.616699|−6.768104 / −0.900348|RR尚无Q/C，RL Qtrue/Cfalse|
|4616|+1.175407 / +73.499345|−4.208919 / −0.718083|RR历史C事件已记4615；RL有TOP接触但未C|
|4624|+7.626537 / +94.922263|−5.492565 / −0.625012|无terminal|
|4632|+11.724177 / +114.658457|−6.882545 / −1.063076|RR current有效，无terminal|
|4640|+13.735655 / +127.178217|**+0.484291 / −0.066950**|RR_FIRST_ORDER中止；RR Ctrue/Pfalse，RL Cfalse/Pfalse|

4640 RL `active_attempt=true`、initial=true、Q历史true、ground=false、within_top_xy/lateral=true、top_contact=true、support/bearing_verified=true、8.789127N；连续TOP样本计数6。FL/FR同时真实TOP承载7.084160/11.047212N；RR AIR/0N。真实TOP接触与“已写入任务P”必须分开：终态RL `crossing_evidence_status=PENDING`，`event_ticks.front_edge_crossed`只有FR837/FL1222/RR4615，**没有RL C tick**。因此可报告“4640出现RL越沿几何并触发顺序约束”，不能伪造正式RL C=4640或改成已完成RL放置。

末决策从4632执行到4640，native记录明确覆盖episode ticks **4633–4640**，八拍全部真实执行；其中command ticks4812–4819是另一绝对计数，不能混用。physical evaluator与semantic/decision endpoint均记4640，前一端点4632无中止：**持久化的规则触发tick为4640，恰好等于本次决策末端**。这里未读取各子步物理完整状态，不从连续TOP计数反推更早的越沿触发tick。

## 实际未完成任务、终态与结论限制

按已保存状态，首个未完成的序列任务是**RR的受控放置**：RR已C但仍在台面上方127.178mm的AIR状态。RL已经真实接触台面、越过前缘，却被记录为顺序中止而没有正式C/P；是否应当允许这种全身协同顺序，由根按用户规范与RR_FIRST_ORDER实现另行核查，本报告不作规则有效性结论。

末帧四轮N全0但projected residual与F均非零：F=`[+.072791,+.584064,−.925663,+.365113]rad/s`。RR最终target `[+3.230325,−58]°`、实际规范角`[−2.774526,−57.702035]°`；RL最终target`[−1.893565,−5.021060]°`、实际`[+16.007886,+2.258512]°`。命令持续、存在真实跟踪差，不把target或joint方向当轮端高度，也不把同阶段N=0当继承截断。

终态372 policy张量finite_fallback=false：按locked bottom=0及relative-plane分量回解base_z **0.093365707m**，gravity z−0.964536250，线速0.162004m/s、角速0.779339rad/s，八个实际规范servo均在硬限内。这是float32持久化观测回解，不是新增原始双精度流；支持本次记录不是低高度FALL、翻转、速度爆炸或实际关节硬限中止。physical VALID/VERIFIED、source RR_FIRST_ORDER，原task outcome仍INCOMPLETE_CONTROLLER_BLOCKED/full_success=false，terminal=true、bootstrap=false。没有碰撞不是完整成功；此次自然P01新RR越沿也不是固定策略评估或稳定性优于FSM的证明。

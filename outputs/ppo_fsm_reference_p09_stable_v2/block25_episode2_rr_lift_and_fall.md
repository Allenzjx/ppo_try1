# Block25 episode2：RR Q后保持AIR，低机身高度FALL，未完成越沿

固定范围：`train/20260911T0053577123642Z_gd4e46006b382_6f25b311ef674a58984702fd575ad5d4`，audit **655–1149**、global **163343–163837**，逻辑字节 `[48440980,86433425)`；2026-09-11 01:32:35.206–01:32:35.578 UTC 单次解析，前654行只定位，另读completed第3行。未读其他回合、大物理流、CP或媒体，无生产修改。

## 本回合直接证据，不由全块差额推断

自然P01首端点8，初始事件历史为空，无教师后缀。495样本来源阶段：**P01=2、P02=116、P03=6、P04=12、P05=129、P06=214、P07=1、P08=1、P09=14**。494×8+末7=3959 physics ticks，全部native verified/actual effect；own-phase=3951，495条均无episode内状态写入，全部mask开放、setter/native mapping通过。

新FR Q22/C948/P990、新FL Q1116/C1888/P2117均发生在当前自然轨迹。2120 P05→P06时FL确实TOP、verified bearing=5.713405N，后来再悬空，不能沿用该历史承载。

RR本回合仅 **I1/Q1/回地撤销0/current有效端点11/C0/P0**：

- **I3857，P09**：事件净空增量3.297607mm、全身关节运动14.062691°、command motion27.897957°。
- **Q3865，P09**：事件净空增量8.563951mm、RR近期自身关节运动4.216224°、台面净空−42.932303mm。
- 3872–3952每8tick一个端点，共11条均AIR/current=true/body_control_evidence=true；没有回地或Q撤销事件。末3959仍AIR、historical Q/established=true，current=false的原因是`physical_safety_abort`，不是回地或历史丢失。

真实P06→P07=3832、P07→P08=3840、P08→P09=3848，三处RR都GROUND/I/Q/current=false；这次Q并非前驱已完成的抬升，不能套用另一回合的事件链。8个普通阶段切换均done=false，全部Full12 residual原值继承，forbidden/phase-scale丢弃列表为空。

## 净空与支撑：建立Q不等于稳定完成

| 端点tick | RR地面相对净空 | RR front / 台面净空 | 当前状态 |
|---|---:|---:|---|
|3872|16.627441mm|−154.718515 / −34.868813mm|AIR/Q有效|
|3912|137.198873mm|−143.308577 / +85.702618mm|AIR/Q有效|
|**3944**|**251.389365mm**|−114.215967 / +199.893111mm|本范围最高记录端点，AIR/Q有效|
|3952|247.249033mm|−101.964402 / +195.752778mm|AIR/Q有效|
|**3959**|234.506688mm|−89.110282 / +183.010434mm|AIR，但安全终止/current失效|

最大高度不是更优或稳定的证明；最靠前的终态仍距前缘89.110282mm，C/P未发生。首未完成任务是保持可用全身状态继续前送越沿，继而受控放置；不是完成越沿后才失败。

Q附近3864/3872，FL均AIR/0N，FR TOP与RL GROUND实测承载均verified；3872分别14.119455N、15.232713N。11个有效端点中FR/RL全部有效承载，FL为7个AIR、4个TOP端点；3952的FL TOP=3.829342N。终态FL/FR TOP与RL GROUND仍分别4.566780N、11.543465N、12.998924N，均verified，但这不排除低机身高度。body-control字段是当时安全包络/短窗支撑证据，不是静态稳定证书或未来成功保证。未保存Q精确tick的完整姿态/速度快照，不能虚构其安全余量；后续FALL也不能追溯抹去已成立的Q。

nominal四轮3832为+.3，3840为0，有同tick几何退役证据：RR front−.202451m、RL−.209733m，`measured_fraction=1 / wheel_gain=0`；final四轮仍非零。3872虽Q有效但靠近前缘且低于台面，carry附加wheel建议=false；3912已高于台面且AIR有效，建议=true。不是切阶段清零residual或固定悬停门。

RR final target从3872 `[5.835594,7.188996]` 到3952 `[67.723924,−13.820955]`、3959 `[68.622147,−14.770955]`°；末实测RR约`[65.324503,−11.432357]`°，与target不同。native下发有效不等于任务完成，不从关节正负或大抬升单独推导失控因果。

## 原始安全终态

completed episode2：**495决策，3959tick/32.991667s，P09 FALL / PHYSICAL_SAFETY / SAFETY_ABORT，VALID / VERIFIED**；full_task_success=false，time_outs=false、terminal_bootstrap_allowed=false。

末372维policy观测全部有限、finite_fallback=false。按当前schema与锁定障碍底面z=0回解：base_z≈**14.297605mm < 15mm**；projected gravity=`[.061385509,.345341802,−.936467230]`，线速模长.242298m/s、角速模长.494483rad/s。符合`isaac_fsm_backend.py:6520`低高度FALL分支，而非gravity_z>−.30或速度爆炸；八个实测servo均未越硬限。高度为**float32相对几何+environment lock推导**，不是额外读取的原始双精度坐标。

结论：一次真实Q及11个有效端点保留，但大净空和三腿接触不能使低高度FALL变为成功；没有回地撤销、没有RR越沿/放置。未发现新的明确交接/dispatch缺陷，不修改规则、不设置训练门禁。

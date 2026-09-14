# Block22 ep0：自然 P01 完成前腿放置后，P06 机身碰撞

固定读取 run `train/20260910T2250463225566Z_gd4e46006b382_afc16caf6fe34ce38d6b58033ec82d59` 的 audit前234行、bytes [0,16712213)，global159105–159338（2026-09-10 22:57:57.097–22:57:57.220 UTC 单次解析），及completed第一行。没有解析活动ep1以后的数据或读取checkpoint/物理大流。

首决策P01 tick0→8，事件历史为空，全回合无curriculum prefix字段；这是**自然P01随机训练轨迹**，不是教师后缀，也不是video9的固定策略确定性评估。234决策：P01=2/P02=90/P03=7/P04=1/P05=45/P06=89。1865 native ticks全部verified/native effect有效，own-phase effect=1860，记录中无episode内状态写入；末决策1864→1865只执行1tick后真实终止。

## 新物理事件与当前承载

| 腿 | 本回合新 I / Q / C / P tick |
|---|---|
| FR | I10 / Q15（P01） / C743 / P789（P03） |
| FL | I873 / Q879 / C1020 / P1153（均P05） |
| RR | I1404（P06，向上变化3.305518mm）；无Q/C/P |

这些前腿事件都由本回合策略产生，不继承教师信用。另有RL Q1842，不能误算为RR Q。

FL放置历史并不证明持续承载：1152已TOP、verified bearing=3.057N；P1153后，阶段交接端点1160实际为AIR/support=false/bearing=0；1168重新TOP、0.372N，1736 TOP、12.537N，碰撞前1864仍TOP、3.733N。终态1865 FL又AIR、bearing=0、load_fraction_valid=false，虽然历史C/P继续保留。

终态FR/RL/RR也记录AIR、support=false、bearing=0；RR仍无有效抬升，front−356.385630mm、台面净空−50.511244mm，不能把这一次末端AIR视为新Q。首次未完成阶段任务是**P06后腿接近/入口准备**：终态 `rear_approach=0.474457480`，尚未进入P07，更未完成RR越沿放置。

## P05→P06 连续性与 d4 桥覆盖范围

P03→P04=792、P04→P05=800、P05→P06=1160，都不是episode终止。1168 jump audit明确全12通道previous residual与carried residual相等，forbidden/phase-scale-clipped列表为空。1152/1160/1168的nominal四轮均+0.3；1160 residual=[−0.452942,−0.166336,+0.364822,+0.551796]，final四轮因此仍连续非零。没有普通交接清零或native下发损坏的新证据。

`capture_owner_hold`确实记录FL observed_capture_tick=1153、held nominal=[48.2,−36.7]°及旧servo owners退休。但**不能将其等同于新capture-to-handoff轮桥的分支激活记录**。当前日志没有`captured_handoff`逐tick分支标记；静态代码的该分支还要求每个源tick当前有效TOP承载、非政策边界、先前rolling和无新wheel事件。

这是早捕获：P05自800开始，P1153时age=2.941667s；冻结 `configs/fsm_states.yaml` 的P05 wheel stop声明在source onset=9.066667s。因此本回合没有覆盖“旧source stop之后再捕获”的关键空档情形。**端点连续已证实，逐tick桥分支是否曾冗余激活无法由所读记录单独确认**；不据此要求额外探针或新增训练门禁。

## 原碰撞失败保留

completed首行确认ep0、seed1001、234decisions、15.541667s、BODY_COLLISION、full_task_success=false。物理评价明确 `TASK_FAILURE_BODY_COLLISION` / `BODY_CONTACT` / `central body/obstacle collision`，tick1865，run VALID；聚合physical evidence status为`CONTACT_BEARING_UNVERIFIED`，不是把所有接触量都宣称verified。没有读取到具体collider/pair、原始BODY接触力或signed distance，故不从姿态或同时失载推断碰撞部件/单一原因。

这是前腿新事件和连续动作的有效实训证据，但完整任务仍因机身碰撞失败。未修改生产、配置、超参数、checkpoint、主ledger或安全标准；不将随机训练回合当成确定性评估改善保证。

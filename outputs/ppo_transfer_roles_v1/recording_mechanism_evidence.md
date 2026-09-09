# Recording 连续角色与动作证据（离线；未改生产）

已完整读新附件。中间数据见 [recording_evidence.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_transfer_roles_v1/recording_evidence.json)：v010 的142个Fast segment、五个重叠窗口及完整12通道start/end/delta；另8版本做有限相似命令检索。`segment_columns` 对应 `segments` 行，单位前8为度、后4为rad/s；原step/event索引、batch、命令速度、servo/wheel持续时间、轮积分、活动及保持通道均保留。未生成CSV、未跑Recording/FSM/Isaac。

## 1. 三种时钟与证据不得混合

复用成熟 `height_based_obstacle_replay/fsm_50mm_recording_derived_v3/recording_fast_plan.py::fast_plan_rows`，它直接调用 `playback.plan_from_steps(profile='fast')`，不是重写解析器。9版本原命令映射错误均0。该parser恢复稀疏命令的持久12维状态，Fast归一化为motion_only，按150°/s与原记录wheel活动时长规划；v010规划82.467333s。人工录制duration、Fast计划时间、派生合同实际120Hz运行时间分列，不能互称。

COMMAND_OBSERVED：原JSON绝对命令、同batch和持续目标；遗漏通道保持，明确wheel stop才停。v010所有step尾确有停轮，不能为“连续”删掉这些原事件，也不能把下一step的非零servo保持当新delta重复累加。planner的segment时长不证明实际关节届时已到位。

MEASURED_RESPONSE：原step前后物理q/adapter快照、派生合同既有轮几何和normal_force。原JSON的grounded_reference诊断可能是缓存spawn信息，本次没有拿它当实时载荷。合同旧contact分类与当前v3硬Q/C/P不同，不直接重授资格。

MECHANISM_HYPOTHESIS：向对角侧转移、释放接收侧连杆空间、接触反作用的解释。当前提取没有质量加权全身CoM/完整接触wrench；身体或轮心位移不能冒充CoM，scalar normal force不能证明水平冲量。

## 2. 五个连续窗口

窗口有意重叠以保留接管前准备，不是相加的训练量；完整角度表在JSON。以下wheel积分顺序FL/FR/RL/RR，单位rad，属于Fast命令积分而非实测位移。

| v010 source steps | 连续命令线索（绝对°） | wheel积分 |
|---|---|---|
| 1–6：FR准备至接触建立 | RL hip0→37.6，再17.5→6.9；FR knee0→45.9；step4同batch含FL knee−22.9、FL wheel−.79及RL+.61 | 4.252/5.200/5.932/5.200 |
| 4–10：新FR支撑至FL捕获、轮行 | FR knee保持45.9；FL[0,−22.9]→[48.2,−36.7]→[22.8,−13.4]；step8全轮+.3持续8.2s，step10再+.3持续25.5333s | 10.492/11.440/12.172/11.440 |
| 10–18：RR完整准备与放置 | FL hip22.8→49.2→38.6；FR knee45.9→31.1并FR wheel−.63；RL hip6.9→31.2；RR hip0→55.6、knee0→−37.8，再hip→−6.9 | 6.754/8.050/8.680/8.680 |
| 18–24：RR接触至RL放置 | step18同时FL[38.6,−13.4]→[−18.5,−31.4]、RL[31.2,0]→[15.4,19.4]、FL wheel−1.07；随后RR knee−37.8→−27.2；FR hip0→12.2→3.7；RL knee19.4→35.3→−18.7、hip15.4→.5→31.2→−10.1 | −2.586/−.660/−.660/−.660 |
| 25–26：前进及恢复 | 全轮+.3持续16.7333s；随后8servo恢复建议与四轮[1.09,−.72,−.43,−.39]同batch持续1s，再显式stop | 6.110/4.300/4.590/4.630 |

step12同batch servo窗口.098667s、wheel1s；step18分别.380667s/1.8s；step26分别.204667s/1s。它们不是孤立摆腿。对应的真实跨step响应仍应读既有测量，不能仅凭这些时长虚构同步力学效果。

## 3. 实测支持与P05/P06缺口

派生合同P05末（32.1s）FL front+90.674mm、gap+13.075mm、AIR/0N；短尾仍AIR/0N。P06开始32.1667s仍AIR/0N、gap+4.743mm；P06末FL为TOP/5.510N。合同也明确FL载荷在连续P06轮行内建立。因此“先placed_FL才允许任何完成捕获所需轮行”的依赖确有反例；这支持开放pending-capture协作建议，**不支持AIR直接placed或降低接触门槛**。并不证明每次P05失败都由该缺口造成。

RR准备P08末：FL AIR/0N、FR TOP10.947N、RL GROUND14.649N、RR AIR/0N；P09入口仍FL AIR而FR/RL约13.344/16.148N。RR放置后P10入口FL/RR约13.507/15.999N，FR AIR/0N；P11末FR gap+79.607mm且AIR/0N，FL/RR仍12.543/14.364N。这与两种对角接收侧开空间、中间支撑交替相容，不要求接收脚已承载，不构成50/50或永久支撑规则。

## 4. FR knee“约−40°”与版本边界

9份命令没有FR knee绝对−40°；最小绝对为v003−10.2、v005−3.9，其余0。v010后腿前step12是45.9→31.1（delta−14.8），原快照physical q46.269→31.205°，recorded adapter target31.225°，字段不得混为一值。v008准备step11为55.4→22.6（−32.8）；其−47.6是最终恢复48→.4。v009−42.3也是最终恢复44.8→2.5，不是深弯入口。

UI/JSON为相对standing的canonical绝对命令；FR sign+1，当前native物理target=rad(standing+最终canonical)，physical q另测。原knee hard[−60,210]°、reserve[−58,208]°，−40本身合法；但当前后腿阶段cap±36，在nominal31.1/45.9时请求区间仅[−4.9,67.1]/[9.9,81.9]。实际可达域须再按当拍mapper/controller/headroom/slew核验，不能用这两个静态区间冒充native证明，更不能改硬限凑角度。

既有策略报告将v003/008/009/010记录为RR_FIRST且任务核心完成、姿态未完成；v007/011只在已停止回放中观察到RR先过、RL未完成。v005/006/012在后腿前停止，实测顺序不可补造。本次引用范围没有已验证RL_FIRST，不能因此称RL_FIRST无效；尤其初始RL hip调整是FR准备，不是RL首先越障。全部历史失败原样保留，不重跑或平均出跨版本“成功轨迹”。

来源：现工程 `configs/recording_motion_contract.json`；成熟工程 `reports/50MM_VERSION_STRATEGY_CLUSTERS.md`、`50MM_REPLAY_TASK_SUCCESS_TABLE.md`；原9份 `versions/*/accepted_steps.jsonl`。精确路径与每条索引在JSON。

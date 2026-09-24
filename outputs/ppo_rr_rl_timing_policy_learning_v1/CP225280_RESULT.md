# CP225280：真实续训与确定性重载结果

本轮协同准备版本已完成 **2048 个有效 policy decisions /16 次 PPO updates /320 次 Adam minibatch steps**，新增 AUX=0。保存并独立重载后，自然 P01 连续评估到 **91.166667 秒**；结果为 **INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED**。RR 抬升、越沿保留，但 RR 捕获和 RL 越障未完成。不是完整成功，也不证明稳定性优于 N。

控制版本 `49eb23163a6e20bc56301dbafb59b137ecebce66`；422 维观测、12 维动作、120 Hz 物理/15 Hz 决策、实际学习率1e-5。后腿任务补全、几何接管和强制 wheel 前向均 OFF；已有 FL capture assist ON，继承的有限 AUX 谱系另行披露。模型、actor/critic、Adam、Identity normalizer、HISTORY 保留；版本迁移时清空未完成 rollout，迁移本身不计学习。

## 最新视频与 checkpoint

正确字幕版为 `video_review/CP225280_deterministic_cooperative_prep_v4_review_v2/`。完整片、同次连续后腿节选、历史 N 同视角对比分别为：

- `CP225280_DET_full_policy_rear_no_assist_INCOMPLETE.mp4`
- `CP225280_DET_RR_to_RL_preparation_REAR_OFF_FL_ON_INCOMPLETE.mp4`
- `N_vs_CP225280_DET_same_camera.mp4`

原始仿真唯一来源：`runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T0214316632402Z_g49eb23163a6e_530cc3c96d834dc3ba84b0b19d8db4f1/source`。正常15 fps、完整失败尾段，不跨run拼接、不加速。N_ref是保留的73.808333秒历史成功，不是新版本同构B；结束后明确定格，不算额外物理时间。首次未发布的v1字幕把终止后源建议与最后已提交基线混淆，保留并标记QA_SUPERSEDED；v2只改证据显示，无新仿真或学习。

v2三片均完成全解码、连续单调PTS和零黑帧检查；完整/对比1368帧、91.2秒，节选527帧、35.133334秒。主代理已目视检查完整末帧、节选入口和对比末帧，最后提交基线、post-step源建议、P09 source holding及RL立面接触标注均分开显示。完整片SHA `53fab7e51b79109a7388ca989f757fb642b8982945719291826319e63a7bbe30`；其余完整receipt在v2目录。

Checkpoint：`branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000225280.pt`。

- SHA256：`21897a4b2d2a85f01092c187c1b1ac824d8e61b01edaf732a6385ca951ab1dcf`
- sidecar SHA256：`6e05b4e279c1d8ac99447f91d69b7db3f6dd6afaab6a326e610283f783b1f185`
- lifetime counters：225280 decisions /1725 PPO updates /34500 Adam steps。
- 本版本起点CP223232；选定分支自CP220544累计4736/37/740。其他分支的训练不借给本模型。
- 最新不等于最优：没有将此未完成模型标为优于已接受的CP221696，未覆盖成功N、原FSM、旧checkpoint或旧视频。

## 自然 P01 的实际结果

| 事件 | 真实时间 / 结果 |
|---|---|
| FR放置 | 23.008333 s |
| FL放置 / P06入口 | 43.041667 /43.066667 s；原FL辅助贡献保留 |
| RR有效抬升 | 58.291667 s |
| RR越沿 | 66.491667 s |
| RR触顶、当前承载、放置 | 均未取得；末帧AIR，gap56.118904 mm，前缘后99.886480 mm，0 N |
| RL | 无有效抬升/越沿/放置；末帧GROUND_AND_OBSTACLE，接触面FRONT_WALL，前缘前50.054040 mm |
| 第一未完成任务 | RR真实顶部捕获，随后才谈可用RR/FL支撑与RL转移 |

P09 late组本次没有启动：source cursor648、status=holding、late_group_start_tick=null。因此本回合的RR高悬不能归因于“late组已经提前执行”。另一方面，policy全12通道仍开放；FR/FL/RL准备并没有被统一mask成零。

末帧FL knee实际−45.179749°，FR knee−18.932307°，RR hip+7.460255°、knee−57.753849°。这是实测，不是仅凭角度证明原因。四个对应髋部安装点世界高度FL/FR/RL/RR分别180.863/218.445/175.128/212.764 mm；左−右均高差−37.609 mm，后−前−5.707 mm。未形成用户提出的温和前右低位候选，更未证明受控向FR转移。CoM局部固定方向投影不是整次转移或FR承载证明。

## 四轮通路：末拍的原始证据分层

以下均为canonical forward-positive，速度rad/s；policy raw/条件均值是无量纲。Episode end tick10940；最后提交执行器receipt native tick11119，二者不是同一时钟。N列是该receipt中组合前基线，不是单独执行过的反事实N轨迹。终止后source hint四轮为0，不能拿它与此前实际final相减归因。

| 轮 | 最后提交基线N | residual许可 | policy raw=条件均值 | post-rate residual请求 | 最终下发 | 步进后实测 |
|---|---:|---:|---:|---:|---:|---:|
| FL |+.300000|1|−1.218602|−1.007090|−.707090|−.866686|
| FR |+.300000|1|−.045668|−.054763|+.245237|+.026399|
| RL |+.300000|1|−.001973|−.001973|+.298027|−.046110|
| RR |+.300000|1|−.172220|−.102323|+.197677|+.196367|

FL持续反转仍是policy对正向N的抵消，不是只有RR有指令，也不是残差mask清掉N。FR/RL最终指令非零而实测较小，RL同时接触立面；这属于需要区分的实际跟踪/接触现象，尚未独立证明某一驱动或载荷是唯一原因。RR在AIR旋转不计牵引。最终写入仍是原有原子Full12执行路径。

## 实际样本与训练质量

| 块 | 有效决策 / PPO更新 | 实际后腿请求阶段 | 协同prep-active端点 |
|---|---|---|---:|
| 真实N前缀到P07，后续学生采样 |384 /3|P07=1,P08=1,P09=64,P10=2,P11=8,P12=308|239|
| 真实N前缀到P10，后续学生采样 |384 /3|P10=1,P11=1,P12=382|143|
| 自然P01 |1280 /10|P07=1,P08=1,P09=451；其余后腿阶段0|0|
| 总计 |2048 /16|P07=2,P08=2,P09=515,P10=3,P11=9,P12=690,P13=0|382|

全部P01–P13计数：4/348/4/1/279/191/2/2/515/3/9/690/0。前缀没有PPO信用，也没有episode内传送。P07学生确有RR触顶和短RL资格，但首次触顶发生在本课程首次更新之前，不能说是这些新更新学会了；P10最初RR放置属于N前缀。两块之后均失去持续RR支撑，RL未越沿。

自然训练回合FR/FL完成，RR仅75.233333–75.358333 s短暂有效抬升，后真实落地撤销，85.141667 s局部任务终止；随后自然reset两拍填完最后128 rollout。这是变化中的随机训练轨迹，不是上述冻结CP225280视频。最后value loss300.089757（有限），实际terminal不bootstrap；新episode预算尾正确bootstrap。两个384后腿块均为非terminal预算尾，正确bootstrap但不足以证明完整后续结果。不能把正GAE当成功。

## 生产改动与仍存在的缺口

- `semantic_cooperative_preparation.py`、supervisor/reward：以已有任务potential预算增加FL可用行程/RL轮端空间代理；声明小的无进展FL反转软先验，未改目标。RR越沿后旧FL接收奖励退出仍保留。
- `semantic_rear_cooperative_prep_{actor,sigma,profile}.py`及训练注册：相同422可观测状态条件下扩大指定准备通道探索；采样与current likelihood共用kernel，保存原raw/logp；无mean head重置或后腿补全器。
- `semantic_hip_mount_geometry.py`、height diagnostics：真实USD同一base_link四安装点，只读世界高度，无物理写入、无高度硬门或自动姿态目标。
- `semantic_cooperative_prep_migration.py`及配置：完整权重/优化器/normalizer/RNG迁移、空rollout；训练/评估相同执行实现。未改资产、摩擦、重力、spawn、碰撞体或执行器能力。
- 定向生产、监督器、分布/likelihood、迁移、四安装点和视频谱系测试已通过，具体分组见COOPERATIVE_PREPARATION_V4.md及封存测试记录；不把合成测试计成机器人成功。

尚未修好的具体问题：本次确定性RR捕获失败、FR/FL准备未产生充分物理效果、FL持续反转、RL接近立面。11个P07首次捕获准备样本的RR knee不同raw全部投影到−58°保护目标（并非−60°真实硬限），而后续更大总体窗口因nominal基线改变不再饱和；不能用平均值掩盖此入口缺口。详见cooperative_RR_channels_append.md。当前counterroll软项还排除了source=0窗口；只读反事实筛选不等于新奖励已实施。

另外，已下发的P09 late远端目标在掉载后可能继续被追踪；仅暂停source clock并不撤回它们。这是其他后缀中确认的控制限制，未在本版暗中修复，也不是本次late未启动回合的已证实失败原因。

本版不证明比N稳定。描述性姿态/速度数据在CP225280_descriptive_stability.md；不同任务结果、时长与窗口不作优越性排名。原N_ref、历史模型与全部失败证据保留。

# CP227328：前段恢复，后腿真实续训，完整任务仍未完成

2026-09-24。保护分支 `cp225280_front_preserved_v1`，控制版本
`892385cba8a7089b52567bb7f558c018fac82a77`。以下是已封存实际结果，
不是计划或完整越障成功声明。

## 最新视频

三片位于 `video_review/CP227328_deterministic_front_preserved_v1_review/`：

- `CP227328_DET_full_RL_completion_attempt_INCOMPLETE.mp4`
- `CP227328_DET_RR_capture_detail_INCOMPLETE.mp4`
- `N_vs_CP227328_DET_same_camera.mp4`

主片从自然P01开始，由同一个保存后重载学生连续执行；无教师前缀、模型
拼接或剪掉失败尾段。1387帧/15fps，物理92.433333s/11092ticks；编码
92.466667s的差额仅为最后4-tick区间的帧显示量化。详情为同次[860,1387)
帧，35.133334s，明确标记RR CAPTURE INCOMPLETE / RL NOT REACHED。
N是保留的历史成功对照，不是本版新跑的同构B；结束后明示定格，不计后续
稳定物理时间。三片完整decode、连续单调PTS、black-like帧数0；主任务已
目视contact sheet与detail首帧。完整绑定和QA见该目录 `export_receipt.json`。

原P05 FL辅助ON；后腿capture/geometry/强制前向辅助OFF；已声明、可观察的
issued-owner suspension/projection ON。没有后腿隐藏teacher或自动设角。

## 1. FR放置后前部降低是否确认？如何恢复？

CP230144的退化已由FR真实放置事件对齐确认：约+0.508s到+5.975s，FR/FL
对应刚体安装点分别降低44.663/26.813mm；约+6s时FL尚未越沿，gap已为
−6.808mm。不是“已在顶部，只差下探”。N相同、mapper差异很小，策略及
实际跟踪已在FR放置前后发生变化；未把单轴或画面倾斜说成唯一已证因果。

最小恢复采用经过439维接口兼容核验的CP225280完整状态独立分支，加有限
前段Gaussian KL保留约束；不改USD、机身/障碍/物理能力、不增加下探脚本，
不设置固定高度或零pitch。原网络/critic/Adam/Identity normalizer/RNG保留，
旧未完成rollout不借用；原成功N、旧CP225280和旧最新CP231680均未覆盖。

新CP227328在FR placed+5.991667s：FR/FL安装点世界高度191.251/198.267mm，
CoM z162.349mm、pitch+0.114637°，FL gap54.037mm/front−116.203mm。
旧CP225280对应安装点无法同run核验，仍标N/A，没有补造。

本次FR placed2761=23.008333s；FL crossed4316=35.966667s、placed5232=
43.600s，首个P06请求5240=43.666667s。FL辅助在捕获处贡献hip−26.164793°、
knee+0.174035°；这是控制器贡献，不能全归给网络。P06首拍FL短暂AIR，
随后5248 TOP5.531N、5352 TOP4.133N；不把历史placed当连续承载。
详细对齐见 `CP227328_DET_front_preservation_sealed.md/json`。

## 2. 实际有多少后腿样本进入优化？

新保护分支累计 **2048真实学生决策 / 4 PPO / 80 Adam**。阶段样本：
P07/P08/P09/P10/P11/P12 = 6/6/430/7/31/1568；P01–P06/P13新增
on-policy样本为0。13条真实连续nominal前缀共9259决策/74072ticks，全部
排除PPO信用。前段保留采用100个真实fit状态、99个heldout，2560次离线
KL曝光进入上述80个原Adam小批次；额外PPO样本/AUX optimizer steps均0，
不是把前缀或回放冒充前段新训练。新自然P01评估才是前段保持的物理证据。

| 非互斥物理窗口 | 已进入PPO的决策端点 |
| --- | ---: |
| RR可落脚区域AIR准备 | 294 |
| RR实际TOP承载下的准备 | 202 |
| 固定FR方向投影及RL载荷份额下降代理 | 109 |
| 学生新取得RL有效离地资格 | 30（3个事件） |
| 继承前缀RL资格，单列不冒称新发现 | 12 |
| RL越沿/顶部放置 | 0 / 0 |

FR方向代理不等于已证明受控侧向转移，表中窗口不能相加。早期P07块有两次
学生RR首次捕获，随后均掉载；最新P10块的首次RR placed均由前缀取得，
第三回合后来重新AIR、合法TOP加载只算支撑恢复，不伪称新的完整越沿。

最后P10块1024决策/2PPO/40Adam，轨迹59/111/269/452/133；前四分别机身
碰障、FL knee实际硬限、FL knee实际硬限、局部未完成。最后133为正确
bootstrap的非terminal预算尾。每raw正式曝光5次；old logp复核误差≤2.60e−6。
未放宽硬限、机身碰障或纯轮爬升规则。

紧急Steer收到时仍在运行的旧分支另在完整边界收尾512/1PPO/20Adam，保存
CP231680；它不进入此学生谱系。合计完成的边界也不能称5次新分支更新。
详见 `892385_P10_CP226304_course_coverage.md/json` 与
`CP227328_branch_physical_coverage_recheck.md/json`。

## 3. 新学生完成哪一步？第一个剩余问题是什么？

新学生独立自然P01完成FR/FL放置和P06，RR qualified7143=59.525s、
crossed8230=68.583333s。最终P09未完成：RR合法顶部XY/AIR/0N，gap55.836mm；
从未placed。RL没有有效离地、越沿或放置，末为GROUND_AND_OBSTACLE，
bearing_verified=false，不能当顶部支撑。

终止为 `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`，
P09实际限时34.9s（30s加4.9s可观察进度预算）。没有机身碰障/NaN/硬限
安全终止；物理run有效，但末接触证据仍保留`CONTACT_BEARING_UNVERIFIED`。
源`DIAGNOSTIC_FAILURE`仅因未完成共同任务，不是录像损坏。
**第一处剩余问题是RR捕获和可继续利用的支撑，不是FL前段。RL目标未完成。**

FL持续反向也未宣称解决：末canonical N轮速四个均+0.3，FL策略附加修正
−1.014460，最终−0.714460、实测−0.816235rad/s。这是本次实际策略抵消，
不能称mask丢失，也没有为评估秘密改方向。同步窄表见
`CP227328_DET_RR_postcross_8230_11092.md/json`：越沿后359决策中RR knee有315次
FINAL为−58°，其余44次最大仅恢复0.045585°；尾裁剪只有0.000603°，不照搬
旧CP的强烈超限解释。RR hip正残差抵消负向N；准备许可开放、输入owner17
为0，FINAL等于同拍candidate，不能称整个四腿准备被mask。近静止的准备
关节、持续FL反转与RR悬空同时出现，但尚未由独立物理探针对单轴因果定论。

本次全段roll/pitch RMS=7.925824°/5.492047°，peak=11.655274°/9.973630°；
无P10–P13覆盖，不能据此称稳定性优于完整成功N。

## Checkpoint、生产修改与恢复

最新checkpoint：
`outputs/ppo_rr_rl_timing_policy_learning_v1/branches/cp225280_front_preserved_v1/checkpoints/history/checkpoint_step_000227328.pt`

SHA256 `5b871af2df8d6b9c0180f7a51863b18b2153148d8dc2a94876f21f5bd73791b4`；
sidecar `0200daa9a22652e1f1c64e5c5ff82a0c977f0418c508158455b3a5ce85bac308`。
227328/1729/34580，save/load=true，实际自适应LR3.375e−5。保存状态没有
episode物理快照；恢复时重新生成真实前缀，不能teleport。

生产修改集中在`semantic_front_preservation.py`、`semantic_front_replay.py`、
`semantic_training.py`及两处受验证身份加载分派；版本固定后未热改源码、
reward或动作含义。14新增CPU/真实优化集成测试、41既有定向测试通过；
身份/数据集/导出正反例另外计账，不算物理样本。最新导出命名16项测试通过。

原始run：`video_eval/validation/20260924T1842131087811Z_g892385cba8a7_c074d1ebec6947cc830fb3e0ec86dcec`。
source manifest SHA `4cab08446b22e19af76866cfc4d7a80cde4ea8d57ead29ea9f3677437c160ead`；
run manifest SHA `171989a9e5fc21f5a4efe57c351a21ea369979b1784a13d515ed4231588140ab`。
正常训练和评估进程均已退出；没有承诺或启动后台续训。活动状态与精确恢复
点见 `RECOVERY.md`，保留CP225792/CP226304前段候选和所有旧产物。

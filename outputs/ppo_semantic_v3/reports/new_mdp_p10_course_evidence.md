# 当前版本 P10/offset0 课程：已有证据与覆盖边界（只读）

当前读取 HEAD：`c34262abffc16847ff32d15ecbf790dd60803e0a`。这是一项课程可行性核对，**不是已选定的新运行、成功预测或优化器门禁**。主线程当前 P06/offset240 训练未被本报告读取、监控或干预；新 continuous-stop 分支尚无 P13 实际样本的状态来自本任务开始时主线程确认，不以旧运行补记。

固定证据：旧 HEAD `68631e932c7deb08a7a3f2a2787b79fa7eb569ef` 的 `runs/ppo_semantic_v3/train/20260906T1453012844004Z_g68631e932c7d_4793a1e4c0664f928b384744693c15fc`，即62848→66944完整P10块。复用已完成的 `p10_capture_block_66944.md`、`p10_capture_initial_512.md`、`p10_capture_first_three_episodes.md`；本次只额外读取首个实际prefix/credit收据和首68条策略行中的6个固定阶段边界样本，不重扫240MB全audit、不重算checkpoint哈希。

## 1. 旧 P10 初始化真实提供了什么

五次prefix均 accepted，每次948教师决策/7584ticks，在episode tick7584/63.2s接管P10，尚余136.8s。教师累计4740决策/37920ticks全部排除PPO信用；每次原200s任务时钟连续，接管不重新计时。未发生miss/fallback是这五次旧运行事实，不是新版本保证。

首收据绑定source_control_tick7583、command_physics_tick7763、handoff_tick7584（物理底层reset偏移与episode时钟不可混用），teacher_calls7584，controller_bias_full12全0；首个策略物理tick7585，首策略决策g62849结束于7592。`from_P01_current_policy=false`，requested/actual都P10，接管时仍处P10。旧教师最后nominal为 `[-18.5,-31.4,0,31.1,15.4,19.4,-6.9,-37.8,-1.07,0,0,0]`；这不是全零静态reset。

| 接管时历史 | 当前测量 | 信用边界 |
|---|---|---|
| FR Q71/C1665/P1695；FL Q2461/C3115/P3583 | FL真TOP/load .425281；FR是AIR/load0 | 前腿主动过程与首次放置均为教师；历史placed不是当前四腿都承载 |
| RR Q6938/C7109/P7579 | RR真TOP/front+91.978mm/load .484429 | RR抬腿、越沿、放置在接管前完成，不可算本回合PPO学会 |
| RL无hard Q/C/P | RL GROUND/front−87.234mm/load .090290，另有FL/RR两个当前支持 | RL尚需真正抬起、越沿、捕获；但workspace与卸载准备已经很好 |

首收据 `workspace_RL=1/support_RL=1`；RL载荷 .090290 低于已有 .20 卸载容差，因此现有 `load_ready_RL` 公式也是1。PPO后续P10/P11的过渡是真连续数据，但**不是从不足workspace/高载荷状态学出完整准备动作的证明**。当前整体速度也不为零：body linear .200818m/s、angular .438908rad/s、最大轮速1.098043rad/s；需要策略接住真实动态状态。

## 2. P10→P13 没有阶段物理reset

首回合固定边界直接重读如下；行内均 `terminal=false`、8ticks且native全验证。

| 策略global / 请求阶段 | 结束tick / 新阶段 | 当前任务证据 |
|---|---|---|
|62849 /P10 |7592 /P11 |workspace_RL=1、support_RL=1；首PPO tick7585，无教师信用 |
|62850 /P11 |7600 /P12 |workspace_RL=1、load_ready_RL=1；P10→P11首tick7593为连续hold，之后7tick own-policy effect |
|62851 /P12 |7608 /P12 |P11→P12首tick7601连续hold，之后7tick own effect |
|62914 /P12 |8112 /P12 |尚未达到真实placed；completion .925不是成功 |
|62915 /P12 |8120 /P13 |RL Q7959/C8070/P8113均在策略接管后；placed_RL=1 |
|62916 /P13 |8128 /P13 |首tick8121连续hold，之后7tick own effect；阶段开始不等任务终止 |

这三次内部handoff的已记录最大servo/wheel跳变量仅浮点舍入量级（≤3.553e−15deg、≤1.735e−18rad/s），残差携带未清零，`hard_safety_modified=false`。PPO/价值估计不会把普通阶段变更当episode done；现有GAE/γ=.995/λ=.95保持跨阶段连续，真正deadline/物理失败才按既有有限任务terminal语义处理。首教师→PPO动作进入仍受既有每tick .5deg/.015rad/s残差slew限制，不应把内部hold误说成首策略没有实际作用。

当前 `semantic_prefix.py` 仍由未修改旧教师真实运行，并让当前TaskStageSupervisor逐tick影子评估；只在有效当前目标阶段、合法决策边界用最后实际收据创建semantic provider。复用同一supervisor/history/时钟，teacher bias若非零先在reset-only takeover中退去，READY才开信用。`from_live_prefix`不导入旧label当新成功，不强制某个历史入口。

P10接管创建的新provider没有一条真实运行过的semantic P06 layer；不能伪造P06 tail重播或把教师经历当该tail的实跑。最后收据nominal/后续channel owner与slew继续有效；这类suffix不是从当前策略自然P01到达P10的等价物。

## 3. 若未来另行选择2048，可针对哪些缺口

旧完整4096行的PPO阶段分布为P10/P11/P12/P13=5/5/706/3380，其余0。四次terminal分别967、964、453、966决策：三次P13 incomplete，一次P12 incomplete；最后746决策是非terminal P13尾段，成功0。

仅作**旧数据的截窗算术**，其前2048行包含旧episode0的967、episode1的964及episode2前117：P10/P11/P12/P13=3/3/242/1800。此数不是新版本2048的采样承诺；新策略、奖励、名义建议、prefix接受情况与终止时机均可能改变访问。2048恰好为16个128-decision rollout的请求规模，不代表16更新已经发生。

| 当前潜在覆盖目标 | 已有支持与限制 |
|---|---|
| P12 RL主动抬腿→越沿→首次capture | 旧5回合有4次策略段RL Q/C/P，另1次Q后落地撤销、无C/P。当前capture-approach可在已qualified+crossed而未placed的RL状态提供连续表面接近信用，仍需真实接触取得另一半capture信用/硬placed。是否实际访问须看新日志。 |
| P13连续受控停止 | 旧块大量P13、包括finite nominal四轮全0段，说明此初始化路径过去能提供收尾样本。当前`per_wheel_four_type_threshold_ratio_v1`按四轮实测速度均值、四轮命令均值、body linear/angular四类型连续聚合，旧68631数据没有执行这个新公式，不能补作新分支训练证据。 |
| 维持已完成腿当前区域/支持、RL落下后的身体稳定 | 历史四腿placed后，旧terminal/尾段仍出现FR或RL AIR及区域退出；当前retention与安全/稳定成本可继续作用。AIR本身不是硬失败，不能新增四腿必须同时TOP的门。 |
| P10 workspace、P11载荷转移 | 存在真实阶段过渡和动作，但旧每回合只各1决策且教师已准备充分。此课程对严重workspace不足、RR失去支撑后的重新准备，覆盖证据很弱。 |

它**不补足**自然P01前腿首次capture、P06两后腿workspace建立、P07/P08准备或P09 RR独立Q/C/P。尤其最新C72320在P05缺FL首次capture：P10教师已完成FL放置，不能以RL后段或P13结果声称修复了该前段缺口。新capture-approach对前腿的作用仍需其自己的自然P01实际证据。

## 4. 具体风险与解释限制

- 旧首两P13 terminal的四轮nominal已经0，但最大实际filtered canonical命令仍 .330504/.461021rad/s、最大实测轮速 .359397/.421291rad/s；stable_for_s=0。旧零nominal固定173行中4轮命令全≤.02为0/173，静态 `0.6*tanh(mean)` 也0/173达标，不能只归咎std或用独立Gaussian概率当真实成功率。当前硬command .02、实测wheel .25、body linear .05/angular .30及 .5s稳定条件未变。
- 历史RL Q/C/P不保证保持区域/当前支持。旧episode2在8054 GROUND撤销RL资格且RR也退回GROUND，P12未完成；尾段RL又是AIR/load0。这些是真控制风险，不可凭完成历史免除安全判定或把问题一概归到P13停止。
- 旧初始化偏向“RR刚placed、RL已接近且低载荷”的窄分布。当前shared actor后段训练可能改变前段输出，但没有controlled遗忘证明；后段成功若将来发生也只按teacher-initialized suffix计，不替代保存checkpoint的自然P01固定mean评估。
- 旧136.8s剩余时间包括后续P10/P11/P12/P13全过程；当前阶段20/20/30/60s和200s整任务时钟仍在，不因预算2048而延长。教师耗时不进优化预算但是真实计算成本。prefix若未到目标或目标已越过，现有失败记录/一次fresh-P01 fallback可能改变实际阶段分布，不可先填预期P13样本。
- 当前capture/stop都是软进度版本差异，P06 tail/几何建议也已不同于68631。即使旧教师动作保持冻结，也不把旧完整episode宣称为当前MDP配对结果；新结果改善不能无对照拆成纯PPO或某一个reward的因果收益。

结论：**P10/offset0是已有工程中可用于后段RL capture与P13新停止信用实测的有界候选，不是全任务能力的捷径，也不是当前P06训练的前置条件。** 是否选择及何时执行由主线程另决策；本报告只提出上述可观察范围，不要求A/B先成功、额外probe或新增训练门禁。已在固定窗口结束；仅PowerShell读取并新增本报告，无Python/Isaac/生产或历史修改、无持续监控。

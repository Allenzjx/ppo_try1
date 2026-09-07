# 134849 运行首回合：P06 接近不足的固定只读诊断

2026-09-06。仅 PowerShell 读取完成数据/现有代码并新增本报告；没有 Python、Isaac、参数修改或训练控制操作。当前运行继续，**不将一次 P06 incomplete 作为启动门禁**。

## 范围与真实结果

运行：`runs/ppo_semantic_v3/train/20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189`，runtime `68631e932c7deb08a7a3f2a2787b79fa7eb569ef`，seed 1001，P01 当前策略自然起点训练。

只读取 `completed_episodes.jsonl` 第一条及 `residual_and_projection_audit.jsonl` 前 **939 行，global 54,657–55,595**。首回合在 **62.6 s / tick 7,512 / P06** 真正终止，`INCOMPLETE_CONTROLLER_BLOCKED`，task/full-task success 均 false；physical evaluator valid=true、physical failure reason=null。它是未完成阶段任务，不是接口崩溃或成功。

另读到完整 optimizer update **400 / global 55,680**，已覆盖上述首回合全部数据；该更新记录 20 optimizer steps、actor 改变、有限非零梯度。本报告不统计其中第二回合的 85 行，不将当前运行剩余计划写为完成量，也不声称已另行验证 55,680 checkpoint 保存。

| 请求阶段 | 首回合实际决策数 | 转入下一阶段的实际 tick / time |
|---|---:|---|
| P01 | 1 | 8 / .066667 s |
| P02 | 186 | 1,496 / 12.466667 s |
| P03 | 4 | 1,528 / 12.733333 s |
| P04 | 1 | 1,536 / 12.8 s |
| P05 | 147 | 2,712 / 22.6 s |
| P06 | 600 | 未转移；7,512 / 62.6 s 终止 |

FR Q/C/P=45/1,504/1,524，FL=1,611/2,640/2,709。RR/RL 均无 qualified/cross/placed；若干 initial clearance 事件不补足 qualified。本回合没有到 P07–P13。

## 第一个未完成目标：两个后轮的 workspace

P06 仍要求 `rear_approach=min(workspace_RR,workspace_RL)`。现有合法 front 区间为 **[−.22,+.06] m**，本回合 P06 两腿横向始终有效，但决策末均未到其前向区间。

| 实际样本 | RR front mm | RL front mm | P06 进度 / global φ |
|---|---:|---:|---|
| P06 入口，tick 2,712，尚为 P05 请求末帧 | −531.820 | −558.636 | 0 / .444613884 |
| P06 首决策末，tick 2,720 | −529.915 | −557.006 | 0 / .444032970 |
| RR 最近的决策末，tick 5,376 / 44.8 s | **−256.393** | −354.662 | .461350936 / .474210304 |
| RL 最近的决策末，tick 5,632 / 46.933333 s | −300.969 | **−350.562** | .477751691 / .449519900 |
| 有限源末附近，tick 5,776 / 48.133333 s | −274.953 | −357.110 | .451561052 / .472424667 |
| nominal 已零，tick 5,792 / 48.266667 s | −284.565 | −357.329 | .450682599 / .470365173 |
| 终止，tick 7,512 / 62.6 s | −395.201 | −456.720 | .053120393 / .453736695 |

最近的两个样本不是同一时刻，不能拼成一个更好构型。RR 最近时仍差 **36.393 mm**，RL 最近时仍差 **130.562 mm**；终止分别差 175.201/236.720 mm。终止前确有接近后再退回，不是从始至终物理位置完全不动。

终止时 RR/RL 均 GROUND、净空约 −50.157/−49.712 mm，载荷 .042494/.490895；FR 当前 TOP/load .466611；**FL 当前 AIR/load 0，但历史 placed=true**。当前 support_count=3。身体线速度 .070438 m/s、角速度 .211276 rad/s、最大实测轮速 .449873 rad/s 均有限。历史完成与当前承载未混同。

## P06 轮建议为何退出：有限源片段，不是 measured retirement

600 个 P06 决策末的 `nominal_provider_diagnostics.p06_rolling_retirement.peak_fraction` 最小/最大均 **0**，`wheel_gain=1`。峰值在每个真实物理 tick 单调保留，故不是中途退休后被还原。终止诊断明确 `terminal_no_new_retirement`，source tick/time 空值，没有伪造终止后的新测量。

该诊断标注 `current_nominal_suggestion_before_slew_not_applied_target`：是当前下一名义建议，不能直接冒充最后已执行 target。下面的 nominal/actual 则来自最后已执行 tick 的 audit。

`configs/recording_motion_contract.json` 的真实 P06 source：t=0 `wheel all 0.300`，t=**25.533333333331882 s** `wheel stop`，有限 end_full12 四轮零。`NominalMotionProvider._continuous_advisory` 每 tick 推进 layer 的 `MotionExecutor`；P06 gain 只乘其轮贡献。本回合 gain=1，所以 source 到时结束不依赖 workspace 完成。

| 实际决策末 tick / time | nominal 四轮 rad/s | 实际 canonical 轮 target，FL/FR/RL/RR，rad/s |
|---|---|---|
| 5,768 / 48.066667 s | .3 ×4 | [.322791, .036542, .067257, .293495] |
| 5,776 / 48.133333 s | .3 ×4 | [.296440, .151546, .046773, .389766] |
| 5,784 / 48.2 s | .1 ×4 | [.060847, −.168454, −.033227, .140308] |
| 5,792 / 48.266667 s | 0 ×4 | [.010401, −.148454, −.075079, .059383] |
| 7,512 / 62.6 s | 0 ×4 | [−.084287, −.304677, −.178466, .109251] |

首个精确零的**已记录决策末**是 tick 5,792；本训练日志没有在此输出每个子 tick 的完整向量，故不复制旧评估的 5,789 为本次实测首零 tick。nominal 随后零到终止，共 **216 个决策 / 14.4 s**。这是有限建议离场，不是 PPO 被 mask 或强置零；残差与 actual targets 持续非零。

首回合 native audit verified **7,512/7,512 ticks**，own-phase effect **7,507**（五次阶段交接首 tick 排除），939 行无 in-episode state-write 违规。实际 canonical target 不是轮速测量，也不是 native 左右符号映射后的轮轴坐标；不能由某轮符号或四轮平均直接推出车身方向/净位移。

## 实际控制平均值：策略仍在作用

以下均为采样决策末的算术均值，轮序 FL/FR/RL/RR。`old_distribution_mean` 是各次真实采样时的 actor mean，跨持续更新，不是冻结末 checkpoint 的确定性新评估。raw 是归一化单位；projected residual 与 canonical target 是 rad/s。

| 范围 / 量 | FL | FR | RL | RR |
|---|---:|---:|---:|---:|
| P06 600 行 raw actor mean | .105842 | −.267841 | −.280341 | −.024139 |
| P06 600 行实际 sampled raw | .098864 | −.267924 | −.280228 | −.022715 |
| P06 600 行 projected residual | .059248 | −.156311 | −.157300 | −.013830 |
| P06 600 行 actual target | .250706 | .035148 | .034158 | .177629 |
| nominal 零后 216 行 raw actor mean | .092602 | −.255847 | −.269004 | −.030574 |
| nominal 零后 216 行 residual = actual target | .042065 | −.151982 | −.151518 | −.017842 |
| nominal 零后 216 行 actual target 绝对值均值 | .078819 | .158508 | .191946 | .084281 |

P06 nominal 的决策末均值四轮均为 .191458 rad/s。FR/RL 的负残差在实际命令层抵消了相当部分正 nominal；这是可观察的相加关系，不是已经证明它导致后腿未到位。接触、轮滑、身体/关节协同及采样更新均未被单独控制。

## 完整有符号 reward 累积

下表直接加总已加权 families，包含真实终止惩罚，不删除失败最后一步。另列 P06 非终止 599 行，以免把终止事件错判成持续质量成本。

| 实际范围 | task | body | contact | smooth | regularization | 总 reward |
|---|---:|---:|---:|---:|---:|---:|
| 首回合 939 行 | −50.536246 | −.480722 | −.003013 | −2.567166 | 0 | −53.587146 |
| P06 全 600 行 | −49.937069 | −.266537 | −.002016 | −1.762615 | 0 | −51.968236 |
| P06 非终止 599 行 | −7.666205 | −.266091 | −.002016 | −1.759339 | 0 | −9.693651 |

P06 task 的分解为 potential shaping **−9.137068617**、时间成本 **−.8**、终止事件 **−40**。其中终止一步 shaping −2.269530660，task −42.270863994；真实终止 next-φ=0、无 bootstrap，不能拿终止 frame 仍存在的物理 φ=.453736695 代替 reward 的 next-φ。

P06 当前物理 φ 范围 .425–.492657016，不是旧 C46464 的长期固定 .4675。旧报告确认的 RL 前置 workspace 被排除问题，不能原样搬到当前启用 preparation credit 的版本。

### 正前进但净成本为负的真实例，不是符号错误

tick **5,768→5,776**，RR 向前 4.577965 mm、RL 向前 .699982 mm；两腿仍未 qualified。FR/FL capture retention 都为 1，RR unload 已饱和，RR/RL initial=false。按生产公式手工 double 算术重算：

- RR workspace **.761876125→.780187983**；RL workspace **.448761124→.451561052**。
- `φ=.425 + .02125*(workspace_RR + workspace_RL + 1)`，重算 .471976041532709→.472424666992976，与记录一致。
- 两腿接近贡献 `Δφ=+.000448625460267`，正向项 `5Δφ=+.002243127301335`。
- 原有折扣项 `−5*.005*φ_after=−.011810616674824`，故实际 shaping **−.009567489373490**；再减时间 .001333333，task **−.010900822706823**。
- 当步 body **−.000440247**、contact **−.00000000197**、smooth **−.002799015**，总 **−.014140086463510**。

这同时证明当前 RL workspace 确实进入奖励、方向没有反号，以及正位移不保证单步净 reward 正。折扣 potential / 时间成本和运动质量成本是不同项；仅靠这个和式或负 advantage 相关性，不能判定存在错误惩罚、奖励主因或最优策略应后退。potential-based shaping 的序列/终止约定仍须完整考虑，不能把一步标量当控制因果实验。

## 有界结论与后续边界

可证明的是：本回合两腿 workspace 均未达到；有限 P06 nominal 在任务尚未完成时退出，retirement 从未激活；零 nominal 后实际残差持续作用且随后总体后退；当前 preparation 方向有实际正信用，body/contact/smooth 非零，但未发现奖励符号/重复计奖或人为禁用策略的具体错误。

与旧 `eval_46464_diagnosis.md` 的共同点是同一有限 source 离场机制；不同点是旧确定性 seed2001 评估 RL 最近时距下界仅 7.394 mm，而本次随机训练 seed1001 的最近差距为 130.562 mm，策略/版本/轨迹不同，不能作 paired 因果比较。`p06_rolling_retirement_live.md` 已验证过另一真实轨迹中的测量退火；那次激活不代表本次也激活。

当前应继续已安排采样，并在固定边界比较是否反复出现“源到时停、workspace 仍远”的现象。若后续选择实验，**nominal 对未完成接近目标的反馈调度**或**reward 相对尺度**只能作为待验证的独立候选；本报告不选择/实施两者，不要求先成功再训练，不改硬任务、动作范围、P13 control、entropy/std。一次首回合 P06 未完成不足以证明当前 capture 修订有害或应立即改运行。

仅本独占报告新增；第二回合不纳入本结论。写完停止。

# Capture retention P01 整块：实际完成至 62848

**训练执行 SUCCEEDED；实际 8192 决策 / 64 PPO updates / 1280 optimizer steps。任务成功 0，不能把执行完成称为任务成功。**

运行：[20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189)，P01 / N1 / seed 1001 / heteroscedastic_log_v1。已完成的 `run_manifest.json`、`training_manifest.json`、64 行 update 日志、8192 行动作审计、9 行 completed episode 日志和最终 sidecar 交叉对账。只读 PowerShell；未运行 Python/Isaac，没有修改生产、旧报告或主报告。另行启动的固定均值 P01 评估不纳入这里。

## 1. Source、保存与预算

| 项目 | 实际值 |
|---|---:|
| 本块 source → final global | 54656 → 62848 |
| 累计 PPO updates | 392 → 456 |
| 累计 optimizer steps | 7840 → 9120 |
| 本块请求 / 实际 / 未消耗 / rounding overrun | 8192 / 8192 / 0 / 0 |
| 本块 wall time | 3060.860190999927 s（约 51.0143 min） |
| 本块 physics ticks | 65525 |
| 已完成回合 / 未终结尾段 | 9 / 1 |
| task success | 0 |

本次是显式 capture 新 MDP warm-start，**不是 exact-MDP resume**。`new_mdp_warm_start.json` 记录 source=54656、保留全部 actor（含已学 std）/critic、固定 324 维观测预处理及 RNG；Adam moments 重置，初始 LR=3e−5，随后沿用原自适应逻辑。旧 rollout buffer 和物理状态均不继承。记录的运行差异只有 `semantic_supervisor.py` 与 v3 `stage_task_spec.yaml`，即已审核的 capture 版本边界；本报告未再次修改或放宽契约。

最终实际保存：[checkpoint_step_000062848.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000062848.pt)，以及 [对应 sidecar](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000062848_manifest.json)。最终 pointer 指向该 immutable history checkpoint；sidecar 的 `save_load_round_trip=true`、global=62848、ppo_updates=456、optimizer_steps=9120。

- checkpoint SHA：`e5c91c4f443c4fcc05671eb4d5ac517d33b6b65449c08b2be9d55bd9cef0a68b`
- sidecar SHA：`1c7b502faadd8a15f15e8c54ec9047dccb57419a5861860dc572e9875f7398ae`

哈希匹配沿用主线程实际文件核验；本次只读取 pointer/sidecar 声明，不重复计算哈希或加载 PyTorch。`physical_env_state_saved=false`，最后未终结回合的物理连续状态不是可恢复存档。

| v3 已消耗课程预算 | Source | Final | 本块增加 |
|---|---:|---:|---:|
| full_episode | 16896 | 25088 | 8192 |
| phase_suffix | 27648 | 27648 | 0 |
| smoke（该 v3 ledger） | 0 | 0 | 0 |

预算原点继续保留 10112：`10112 + 25088 + 27648 = 62848`。这里的 v3 smoke=0 不抹去原点之前的历史训练。当前 sampling 为 `P01_full_task_only_initial_version`；core decisions 与 policy decisions 同为 8192、core ticks 与已计信用 ticks 同为 65525，无 teacher prefix 文件或额外前缀决策混入本块信用。

## 2. 所有实际阶段样本

| 阶段 | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 | P13 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 已优化决策 | 11 | 1770 | 36 | 10 | 1479 | 3856 | 8 | 5 | 564 | 1 | 2 | 450 | 0 |

合计 8192，来自每行 action-source `phase_id`，与 final telemetry 完全一致；不是 nominal 层数、阶段标签奖励或计划份额。P12 的 450 条仅来自 episode 7；本块没有 P13 样本。

## 3. 九个完成回合与真实非终结尾段

`Q/C/P` 表示 qualified/crossed/placed 的 episode-local physics tick，`—` 表示没有该硬事件；initial clearance 不等于 Q。

| 回合 | global 范围 | 决策 / ticks | 秒 / 结束阶段 | 原始结果；第一未完成 |
|---|---|---:|---|---|
| 0 | 54657–55595 | 939 / 7512 | 62.6 / P06 | INCOMPLETE；rear_approach=.053120393 |
| 1 | 55596–56272 | 677 / 5409 | 45.075 / P09 | BODY_COLLISION；RR 尚无 Q/C/P，placed_RR=.265268526 |
| 2 | 56273–57203 | 931 / 7448 | 62.066667 / P06 | INCOMPLETE；rear_approach=.437048632 |
| 3 | 57204–58126 | 923 / 7384 | 61.533333 / P06 | INCOMPLETE；rear_approach=0 |
| 4 | 58127–59061 | 935 / 7480 | 62.333333 / P06 | INCOMPLETE；rear_approach=0 |
| 5 | 59062–60080 | 1019 / 8152 | 67.933333 / P09 | INCOMPLETE；RR 曾多次 Q，仍无 C/P，placed_RR=.416115265 |
| 6 | 60081–60671 | 591 / 4728 | 39.4 / P09 | BODY_COLLISION；RR 尚无 Q/C/P，placed_RR=.290757170 |
| 7 | 60672–61799 | 1128 / 9024 | 75.2 / P12 | INCOMPLETE；RL 尚无 Q/C/P，placed_RL=.094106716 |
| 8 | 61800–62417 | 618 / 4940 | 41.166667 / P09 | BODY_COLLISION；RR 尚无 Q/C/P，placed_RR=.242220231 |
| 9 尾段 | 62418–62848 | 431 / 3448 | 截止 28.733333 / P06 | **未终结**；rear_approach=.392679889 |

前九回合共 **7761** 决策，尾段 **431**，合计 8192。完成结果是 **6 INCOMPLETE_CONTROLLER_BLOCKED + 3 BODY_COLLISION**；不得把预算结束的第十回合追加成一次失败或成功。

| 回合 | FR Q/C/P | FL Q/C/P | RR Q/C/P | RL Q/C/P |
|---|---|---|---|---|
| 0 | 45 / 1504 / 1524 | 1611 / 2640 / 2709 | — / — / — | — / — / — |
| 1 | 66 / 1348 / 1368 | 1450 / 2488 / 2559 | — / — / — | — / — / — |
| 2 | 59 / 1445 / 1455 | 1534 / 2579 / 2642 | — / — / — | — / — / — |
| 3 | 61 / 1386 / 1394 | 1496 / 2527 / 2578 | — / — / — | — / — / — |
| 4 | 60 / 1471 / 1488 | 1573 / 2613 / 2676 | — / — / — | — / — / — |
| 5 | 51 / 1508 / 1528 | 1608 / 2581 / 2713 | 5642，7774，8125 / — / — | — / — / — |
| 6 | 61 / 1372 / 1385 | 1472 / 2442 / 2576 | — / — / — | — / — / — |
| 7 | 48 / 1428 / 1450 | 1539 / 2344 / 2651 | 5346 / 5396 / 5397 | — / — / — |
| 8 | 58 / 1392 / 1410 | 1505 / 2328 / 2620 | — / — / — | — / — / — |
| 9 尾段 | 46 / 1485 / 1497 | 1604 / 2442 / 2684 | — / — / — | — / — / — |

Episode 5 的 RR 在 **7565、7917** 两次 GROUND-before-cross 撤销资格，随后分别重新 Q7774、Q8125；`event_ticks.active_lift` 只保留首 Q5642，以上重试来自真实 `lift_attempt_events`，不是重复奖励推断。终止时 RR 仍 AIR/front −162.220 mm、clearance −.653 mm，未 crossed/placed。

Episode 7 真实完成 RR 子事件，但终止时 RR 已 **GROUND**、front **−218.364 mm**、clearance **−48.568 mm**、load **.148809**；历史 placed 不能代表当前 TOP。RL GROUND/front −146.225 mm、clearance −51.132 mm、load .521812，尚无硬 Q/C/P；FR TOP/load .329378，FL AIR/load0。

最后尾段 P06：RR/RL 都 GROUND，front 分别 −371.830 / −350.181 mm；FR TOP/load .417068，FL AIR/load0。虽前腿已有本回合 Q/C/P，后腿准备仍未完成；没有从未来评估或旧回合补写后腿历史。

### 三次碰撞的末 tick 分账

| 回合 / global | 末决策实际 ticks | 相对 8 tick 少 | bootstrap / time_outs |
|---|---:|---:|---|
| 1 / 56272 | 1 | 7 | false / false |
| 6 / 60671 | 8 | 0 | false / false |
| 8 / 62417 | 4 | 4 | false / false |

因此 `8192×8 − 7 − 0 − 4 = 65525`。是三次碰撞中两次短末决策，**不是三次都短**；没有遗失决策，也不是外部截断 bootstrap。

## 4. 真实 actor / optimizer / raw Gaussian / native 审计

64 条 update 为 **393–456**，global 每次严格 +128、每次 optimizer steps=20。63 个相邻 `actor_before == previous_actor_after` 全部相等，64 次 actor changed 与 finite nonzero gradient 均 true；首尾 actor hash 也与 final training manifest 一致。

- before：`cca90f60744f0e00068068c9160cd06648a129a881df9cb010380a17dc716f5d`
- after：`7d8b3ea0b862a587ed65877e84da836033c14c8275a3d46f694ebd8cb885f56b`

| 64 次实际 update 指标 | 最小 | 最大 | 均值 |
|---|---:|---:|---:|
| KL mean | .013904703 | .038417591 | .022491779 |
| clip fraction | .2203125 | .36875 | .290405273 |
| entropy | −5.974983287 | −4.019726896 | −5.084613438 |
| value loss | .000692650 | 107.946147346 | 14.371102476 |
| surrogate loss | −.055927213 | −.017727511 | −.033599248 |
| optimizer LR | .000010000 | .000067500 | .000010898 |

所有上述指标有限。日志中的 gradient_norm_min 范围 1.000253596–1.414212881，gradient_norm_max 范围 1.012757994–1.414213637；这里仅按原记录含义报告，不冒称独立重算的 clipping 前范数。Gaussian differential entropy 可为负，不表示非法分布。

8192 条均保存有限的 12 维 raw action、old mean、old std（std>0）、old log-prob、old value 和 reward；8192 条 raw action 与对应 applied audit 的原始 raw 字段逐值相等。所有 action/channel 的 old std 最小 **.0895875394**、最大 **.2509717345**。原始 Gaussian log-prob 范围 **−8.6293277740 ～ +10.8755626678**；这是连续密度 log 值，不是离散概率。

用实际 raw/mean/std 独立按未 tanh 的 12 维 Gaussian 求和，double 重算与记录的 float32 log-prob 最大差 **1.5668302049e−6**（g56341，日志 −.600914001465，重算 −.600915568295）。未错误使用 applied action 或添加 tanh Jacobian。这是日志数值一致性核验，不是重新启动官方 PPO 或重新加载 storage/checkpoint。

全部 **8192** 行 native `all_ticks_verified=true`；逐 tick 审计列表长度等于实际 physics_ticks，列表内 verified 均 true，累计 verified tick count **65525**。全部 **8192** 行四个 in-episode root-pose/root-velocity/force-or-impulse/gravity 写入计数为 0，且 `no_in_episode_state_writes_verified=true`。没有把 reset 或 teacher 的物理步骤充当 PPO 信用。

## 5. 当前 capture 保持分支的实际覆盖

| 腿 | 已 placed 腿-决策行 | R<1 | AIR 且 R=1 | AIR 且 R<1 | GROUND 且 R<1 | 最小 R |
|---|---:|---:|---:|---:|---:|---:|
| FR | 6385 | 0 | 229 | 0 | 0 | 1 |
| FL | 4896 | 2 | 3646 | 2 | 0 | .979323420 |
| RR | 454 | 448 | 3 | 55 | 352 | .086782488 |
| RL | 0 | 不适用 | 不适用 | 不适用 | 不适用 | 不适用 |

R<1 的剩余接触情形可能是 obstacle pair/非 TOP；不能把它们统称 GROUND。该表只对已经真实 placed 的腿计保持度，后腿尚未放置时仍沿用原进度分支。

保留此前三个独立有界收据，不覆盖其历史结论：

- [首 1024 真实奖励核验](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/capture_retention_live_reward.md)：g54657–55680，新旧 phi/shaping 全部相等，当时 R<1 尚未 exercise。
- [3072 决策窗口](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/capture_retention_window_3072.md)：g54657–57728，仍无 R<1；三个终局与第四回合尾段均按当时边界报告。
- [真实主动分支](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/capture_retention_live_active_branch.md)：仅 g61340–61440，RR 放置→退回→短暂区域恢复→GROUND，完整新旧 phi/shaping 独立重算误差 0；报告明确 t5397 事件与 t5400 实际 TOP 几何的证据边界。

本块证明新保持信号确实参与实际优化样本，不再对所有已 placed 腿的区域回退完全不敏感；但 episode 7 仍退回并在 P12 incomplete，不能声称已学会保持或优于 baseline。总体为训练执行完成、0 次完整任务成功；当前 checkpoint 的独立固定均值 P01 评估须另按实际结果报告。

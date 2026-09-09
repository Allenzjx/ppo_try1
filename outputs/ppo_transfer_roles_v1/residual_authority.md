# Transfer roles：残差执行幅度与边界

状态：配置及直接相关 CPU 测试已完成；未运行 Isaac，未证明物理响应或任务成功。本页中的 −40° 只是 FR knee 候选目标，不是历史唯一姿态、接管条件或硬成功条件。九个 Recording 版本中未找到 FR knee absolute −40°；absolute / delta / native articulation 必须继续区分，见 `recording_mechanism_evidence.md`。

## 本次限定变化

- P01–P05 四轮 requested cap：0.3 → 0.6 rad/s；P06–P13 保留前轮 1.2、后轮 0.6，不在后续阶段缩小范围。
- P06–P13 **仅 FR knee（canonical index 3）**：36 → 112°；P01–P05 FR knee 仍为 24°，其余七个 servo 不变。
- 保持 12 通道全开放、120/15 Hz、tanh、原 Gaussian/history kernel、rho、噪声参数、轮请求限速 1.8 rad/s²、servo 请求限速 60°/s、最终原 mapper/drive 限速 1.25°/tick。没有修改 adapter、projector、冻结 A 或任何硬限。

## 为什么 112，而不是仅按 nominal 选择 96

真实顺序是唯一 nominal mapper → nominal geometry（只涉及 active rear pair，不能更改 FR）→ 有界 controller bias + policy → 同 tick headroom → 原最终硬限与 slew。

FR knee nominal 为 45.9° 时，合法 mapper carry-over correction 可为 +10°，独立 controller correction 也可为 +10°：baseline = 65.9°。目标 −40° 需要 **requested residual −105.9°**。96° cap 的最负极限只能给最终预限速目标 −30.1°；即使 controller=0，所需 −95.9° 已接近 96 上限，latent 约 −3.78。112° cap 对最保守组合只需有限 latent `atanh(−105.9/112) ≈ −1.79`。

CPU 测试先让真实 frozen mapper 从零自行发展出 +10° correction，再结束 tracking；真实测量未收敛，合法 carry-over 保持。随后启用生产 previous-ACK-request reference 路由（当前该通道 inactive，仍读真实 q），从零请求按 0.5°/tick 累积。没有直接改 mapper/history 或绕过 slew。

| nominal FR knee | controller bias | mapper native | baseline | 到 −40° 的请求 | 实际测试所需 request ticks / 秒 |
|---:|---:|---:|---:|---:|---:|
| 31.1 | −10 | 41.1 | 31.1 | −71.1 | 143 / 1.1917 |
| 31.1 | 0 | 41.1 | 41.1 | −81.1 | 163 / 1.3583 |
| 31.1 | +10 | 41.1 | 51.1 | −91.1 | 183 / 1.5250 |
| 45.9 | −10 | 55.9 | 45.9 | −85.9 | 172 / 1.4333 |
| 45.9 | 0 | 55.9 | 55.9 | −95.9 | 192 / 1.6000 |
| 45.9 | +10 | 55.9 | 65.9 | −105.9 | 212 / 1.7667 |

这些是**目标缓冲区**抵达时间，不是实际关节 q 抵达时间。测试使用固定 measured q 的 CPU articulation-buffer fixture，未模拟惯性、接触、稳定性或阻挡响应。实际 active tracking、nominal change、后续 policy变化都可改变基线和结果。

FR sign 为 +1；CPU fixture 的 standing 为 0.75 rad，最终 canonical −40° 的 articulation target 是 `0.75 + radians(−40)`，不是 articulation −40°。knee 原硬限仍是 [−60,210]°，reserved band 仍 [−58,208]°。baseline=65.9 时 ±112 请求给出 [−46.1,177.9]°，并不宣称可访问全部 reserved band；baseline=0、request=−112 会被原 headroom 裁为 −58。最终 slew 仍可让 actual target 暂时不同于候选目标。

## 早期轮子的有限 latent 可达性

当 pending nominal=+0.3 rad/s、无附加 wheel bias 时，cap0.6 的未滤波 canonical 区间极限为 [−0.3,+0.9]，有限 latent 下两端为开区间。0.3 cap 无法有限地抵消 +0.3，更不能反向。

| 期望 canonical 净命令 | 请求 | raw = atanh(request/0.6) | 从零请求的 ticks |
|---:|---:|---:|---:|
| −0.02 | −0.32 | 约 −0.595 | 22 |
| 0 | −0.30 | −0.549306 | 20 |
| +0.02 | −0.28 | 约 −0.506 | 19 |

测试覆盖 P01/P05 × 四轮 × 三个目标，并核对实际 float32 staging/dispatch 的左轮反号与右轮正号。它证明本执行范围可表达这些目标，不证明 PPO 会给出对应 raw、保持足够久或获得更好控制。原 ±2.0943951023931953 rad/s wheel hard limit 未更改。

## 必要的两列观测尺度迁移

旧 requested-history scale4、clip20 只能无损编码 ±80°，不能完整描述新的 FR knee 请求。已在 parent 明确授权下只把 `previous_residual_full12[3]` 与 `previous_previous_residual_full12[3]` 的 scale4→6，实际列分别 **210、222**。新 cap112 编码为 ±18.6667，不触 clip20；其他所有列、顺序、维度324、raw history 195:207、raw clip20 不变。

这是显式新 MDP/观测数值版本边界，不能普通 exact resume。旧未饱和范围内对应第一层权重列乘1.5可补偿输入缩放，**但完整 actor/critic、normalizer、RNG 和模型数值比较/迁移由主集成负责**；本文件仅测试 scale、列位与标量补偿方向，不声称已完成该迁移。

## 测试收据与范围

`tests/unit/test_semantic_transfer_residual_authority.py` 新增38项；原 `test_semantic_front_wheel_authority.py` 77项仅更新早期 cap fixture 数值一行。合计 **115 passed / 0 failed / 0 skipped，12.838s**。外部 JUnit：`C:/robotics_sim/wlr_robot/transfer_residual_authority_trial01.xml`。

真实 production mapper/reference/headroom/final-drive 经 CPU 四 float32 buffer 审计 verified：每 tick 仅一次 mapper advance、一次 articulation dispatch；审计不推进 mapper/不写目标，zero policy 与原 frozen dispatch 相等，大请求跨 P06→P07 不清零，撤回按原限速退场。没有 Isaac state-write 测试或物理成功声明。测试 Python 已退出；未提交。

## 后续真实 PPO 使用证据（第三段训练完成后补记）

上面的“未提交／未物理运行”是单元测试收据当时的状态。集成生产版本已提交为 `2f27c6f5065aee6fe7b17f165a876e6dc5307841`，501 项集成测试通过，按迁移协议从实际 123136-decisions checkpoint 续训；本节不是新的参数修改。

数据来自已完成运行 `20260907T2346092851638Z_g2f27c6f5065a_3ba4a92452b14e74ad1f3d5c8a03075d` 的一次流式汇总 `block3_summary/training_block_summary.json`。该段有 1024 个 PPO decisions、8 次更新；其中 P06 为 990 个政策样本。四次成功 P06 重置前缀共 1792 decisions，全部排除于 PPO 信用之外。

| P06 通道 | filtered requested residual 范围 | final canonical drive target 范围 | 最终净负／正样本 |
|---|---:|---:|---:|
| FR knee（deg） | −102.843614 ～ 77.627294 | −57.496562 ～ 122.974346 | 148／842 |
| FL wheel（rad/s） | −1.148854 ～ 1.041651 | −0.848854 ～ 1.341651 | 590／400 |
| FR wheel（rad/s） | −1.067307 ～ 0.816535 | −0.767307 ～ 1.116535 | 327／663 |
| RL wheel（rad/s） | −0.558245 ～ 0.273806 | −0.258245 ～ 0.573806 | 292／698 |
| RR wheel（rad/s） | −0.397656 ～ 0.587368 | −0.097656 ～ 0.887368 | 17／973 |

FR knee nominal 为 45.9°，四轮 nominal 均为 +0.3 rad/s；FR knee raw latent 范围 −2.849810 ～ 1.970061。以上表明扩大的请求范围实际被策略使用，经过执行层可形成负向 FR knee 目标和各轮净反向命令。不同字段极值不保证来自同一时刻，不能拼成单 tick 因果链。

范围仅覆盖 decision 末快照，按该 decision 的来源阶段归组；它们不是完整 120 Hz 范围、物理 measured q/速度，也不是已反左轴符号的 native float32 readback。native 审计另记：8171 个政策物理 ticks、3 次非 terminal handoff，记录的 native、四类状态写入、bootstrap 异常均为零。

该段实际结果为 P06 BODY_COLLISION、P09 FALL、P06 INCOMPLETE，另有 8-decisions 非终结 P06 尾段。RR/RL 资格出现后均撤销，没有后腿越沿或放置；不能把上述控制能力证据称为安全或任务成功。

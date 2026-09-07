# C66944 固定 mean 天然 P01 评估：真实 BODY_COLLISION

来源：`runs/ppo_semantic_v3/validation/20260906T1546561630471Z_g68631e932c7d_db836ec127c44b7e983ff9a307ef594a`。固定、已完整结束的单次评估；只读 PowerShell 核验，没有重放、Python、Isaac 或生产修改。

## 1. 完整结果，不把执行成功当任务成功

- `run_manifest.json.lifecycle = SUCCEEDED`；根已确认 session70942 exit0。评估执行完成，但 `task_success=false`、`controller_task_success=false`，终止 `BODY_COLLISION`，不是完成任务。
- checkpoint `checkpoint_step_000066944.pt`；evaluation manifest 记录的 checkpoint SHA256 为 `8d5896c5fb02b55ab09b9fcaf387c7a43c112795928c7a721ad31126cd0002bb`。本次读取记录，未重新加载/哈希张量。
- runtime source `68631e932c7deb08a7a3f2a2787b79fa7eb569ef`，seed2001，`deterministic_policy=true`、天然 `from_phase=P01`、`optimizer_updates_during_evaluation=0`。不是 P10 teacher suffix，也不是随机动作训练回合。
- 585 policy decisions、4674 physics ticks、38.95 s；严格等于 `584*8+2`。最后 decision585 只执行2个120Hz tick即真实终止，`terminal_bootstrap_allowed=false`、reward terminal next-phi=0。没有补齐剩余6tick来延后碰撞。
- 4675 行 physical observations 包括 reset tick0；4675 行 `all_finite=true`。末 physical evaluator `valid=true` 且 `termination_reason=TASK_FAILURE_BODY_COLLISION`，说明有效环境内真实物理任务失败，不是接口失败；不同层的 `BODY_COLLISION` 是同一次终止的简写。
- 阶段 decision 计数 P01–P13：`[1,175,3,1,149,243,1,1,11,0,0,0,0]`，总585。P09只执行82tick（0.683333 s）就碰撞，非 P09 deadline。

## 2. P06–P09 的实测连续链

| 交接完成 tick / time | 阶段变化 | RR front / RL front，m | FL 当前 load | RR 当前 load | RR clearance，m |
| --- | --- | --- | ---: | ---: | ---: |
| 2632 / 21.933333 s | P05→P06 | −.540142 / −.555194 | .197217 | .254881 | −.0502200 |
| 4576 / 38.133333 s | P06→P07 | −.219278 / −.159929 | 0 | .109679 | −.0500424 |
| 4584 / 38.200000 s | P07→P08 | −.217611 / −.158531 | 0 | .0799003 | −.0499920 |
| 4592 / 38.266667 s | P08→P09 | −.215860 / −.157400 | 0 | .107473 | −.0499891 |

P06花1944tick/16.2s将两后腿推进现有 workspace；P07、P08各8tick/一个decision。交接时 workspace_RR/workspace_RL/support_RR 或 load_ready_RR 等相应物理谓词已经满足；它们不要求重新达到旧A固定支撑/入口姿态。这里不能把阶段短本身叫作跳过任务或代码失败。`stage_transition_evidence.jsonl` 同时含同阶段历史事件，不能把每行都算成阶段切换。

上述 ordinary phase 交接的 decision termination 均 null、task_success=false，并保留 bootstrap；仅末 collision 才 true terminal。P07/P08的真实8tick audit均8/8 verified、7/8 own-phase-request native effect；P09首decision同样7/8，第二decision8/8。第一tick交接保持，随后本阶段请求确实影响目标，不是整decision raw被吞。

实际 native wheel 连续性（四轮 canonical 顺序 FL/FR/RL/RR；native方向符号遵循原adapter）：

- tick4576(P06)→4577(P07)：nominal四轮均 .268397937，residual四轮及 native targets 逐值相同；native为 `[-.145105645,+.403576255,-.555452406,+.479936182]`。4578开始新的建议/残差，native实际改变。
- 4584(P07)→4585(P08)同样逐值保持；4586新请求生效。
- 4592(P08)→4593(P09)同样逐值保持；4594新请求生效。
- decision末 nominal四轮从4576的 .268397937、4584的 .168208490、4592的 .071818839 到4600的全零，residual并未被清空。P09后续仍有其它建议：4672/4674四轮 nominal为 `[0,-.63,0,0]`，不能把整个P09描述成全轮零nominal。

这些是实际命令/时钟连续的外部证据；没有逐字段导出所有内部 mapper/filter 状态，故不声称对其做了内部状态逐位比较。没有发现普通阶段切换导致 episode reset 的证据。

## 3. FL 历史已放置，但当前并非持续承载；RR 尚未取得初始/硬资格

真实 hard-history 事件：FR qualified48 / cross1418 / placed1432；FL qualified1517 / cross2540 / placed2632。RR、RL均无 hard qualified/cross/placed。最终完整 `lift_attempt_events` 中也没有 RR `whole_body_initial_clearance`，不能把其末端小幅离地当作已经通过 initial 或 qualified。

FL在2632放置时实际 obstacle pair force 6.696514 N、load .197217，是真实接触，不只是历史标签。随后 exact contact 状态为：

- 2632–2690：仍有 obstacle 接触；2691开始AIR。
- 3355–3364：短暂重新OBSTACLE；3364 force .669941 N。
- 3365–4674：连续1310个tick为AIR、ground/obstacle pair均不活跃，终止仍 load0；本段采样覆盖约10.9167s。

到P06→07、P07→08、P08→09时FL分别在顶面上 +3.54894、+10.2483、+29.1630 mm，但已不承载。末端FL clearance +195.809 mm、front +.467972 m，仍在平台XY内但非TOP支撑。真实当前支撑在末端是 FR TOP load .440215 和 RL GROUND load .559785；RR/FL AIR。既有 FL placed/history 不表示当前 FL support=true。此处只陈述实际协调结果，不能由此推导必须固定FL承载的历史姿态门禁，或证明它独自造成碰撞。

RR到4672仍GROUND，clearance −47.5612 mm，ground force3.791929 N；4673在BODY接触同tick才转AIR，4674只累计2个AIR samples。最终RR bottom=3.659912 mm，相对50mm顶面clearance −46.3401 mm，front −.208736 m；current initial=false、hard qualified=false、cross/placed=false。末快照的短后缀 measured gain仅1.221128 mm，不能仅因绝对bottom略超3mm就授予整个主动过程资格。没有RR达到顶面或已跨沿的证据，也不是本次 `WHEEL_ONLY_CLIMB` 终止。RR刚离地与BODY首次受力同步，日志不能将此离地归因为成功主动抬腿。

## 4. BODY collision 的 exact-pair 正证及几何界限

全部4675 raw行中，仅4673、4674的 `contacts.base_link.obstacle.active=true`。两者 `sensor_body=base_link`、`other_body=/World/Obstacle`、`pair_verified=true`，来源 `isaaclab.ContactSensor.force_matrix_w`。不是任何腿/轮contact被误作BODY。

| tick / time | exact pair active | 连续计数 | 实际 force world XYZ，N | contact point world XYZ，m | detected |
| --- | --- | ---: | --- | --- | --- |
| 4672 / 38.933333s | false | 0 | [0,0,0] | [.58043325,−.24542674,.05080972] | false |
| 4673 / 38.941667s | true | 1 | [−3.20e−7,+3.98e−7,+31.2068920] | [.58056885,−.24553028,.05016085] | false |
| 4674 / 38.950000s | true | 2 | [+5.63e−7,+1.48e−7,+12.1347952] | [.58056676,−.24552134,.05000224] | true |

4672虽然有point记录，但力为0、activefalse，检测器没有把point单独当成真实碰撞。4673只一个active tick，明确等待持续证据；4674第二tick满足原条件，reason=`exact base_link/obstacle pair persisted`。两次实测力主要向上；point的z接近障碍顶面 .05m，x大于实际front .521312174m，且y在[−1,+1]内部。这与躯干碰到平台表面的物理记录相容，不是后腿撞立面的证据。

三个时刻 `geometry_penetration_m=0`。生产 `BodyCollisionDetector` 明确采用 `real_pair AND (persistent OR geometry_corroborated)`，持续阈值2tick；几何深度来自 `geometry.py:183` 的base bounds/AABB overlap。因此0只表示该几何指标没有提供穿透佐证，不否定已验证的exact-pair持续力，也不能用它宣称身体未碰到障碍。反之，两个contact样本不能证明真实网格深穿透、长期压住或之后必定翻倒。终止后没有第4675个执行tick；报告没有“碰撞后若继续”的物理证据。没有修改检测器/阈值。

## 5. Native/状态写入与学习信号边界

`native_tick_audit.jsonl` 恰4674条，4674/4674 verified；逐行四类 in-episode root pose、root velocity、force/impulse、gravity write 总和为0。最后2tick均有真实native effect和本decision own-phase effect，不是NaN fallback或未执行动作。

末端应用logical wheel commands为 `[-.092609848,-.543938388,+.201374062,+.183114398]` rad/s，native float32为 `[+.092609845,-.543938398,-.201374069,+.183114395]`。这里是目标，不冒充measured速度。末物理base linear norm .0379338 m/s、angular norm .150628 rad/s；BODY碰撞不是超大速度/非有限爆炸触发。短终止的后续phi=0/no bootstrap与原真终止约定一致，不在阶段交接重复终止。

结论：这是新保存C66944在天然P01固定mean执行中、前腿已真实完成但后腿尚未取得RR初始/硬lift资格时发生的、exact BODY pair持续接触导致的真实失败。P06 workspace进展、短P07/P08、本阶段raw实际作用、当前FL不承载与碰撞是可核实的时间链；目前不是一个由单一动作或奖励分量证明的因果解释。不能用此前P10训练到P13的suffix结果替代此full-P01结果，也不能拿P13 stop提案解释或修复这次尚未到P13的碰撞。

已停止只读分析；仅新增本报告，未运行Python/Isaac、未修改production/config/tests、未提交。

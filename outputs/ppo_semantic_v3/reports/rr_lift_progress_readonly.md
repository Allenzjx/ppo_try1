# RR initial→qualified 抬升高度平区：只读诊断与候选

## 结论

**存在明确的局部高度奖励平区。** 在前驱腿已经 placed、RR 已有真实 initial clearance 但尚未 qualified、workspace/unload 等其余量不变时，继续从约3 mm抬到低于障碍顶面的50 mm，不会持续增加当前 global physical potential。此结论来自生产公式，也被同一真实回合中不同高度、相同其它势函数条件的样本支持。

这不是“总 reward 为0”，也不意味着提高轮底完全没有其它动力学收益。姿态、接触、平滑性、workspace/unload、初始事件、最终资格和时限仍影响奖励。这里缺的是**已开始主动离地后，继续缩短到顶面距离的直接 task-progress 差分信号**。

本报告只读当前固定 runtime `cae1d6e7cfdd8383fec67e8f919cc823f9e65448` 和已完成首回合；无 Python、Isaac、生产修改或提交。4096训练块不因此暂停或增加门禁；先保存并进行计划中的 P01 评估，再决定下一版本。

## 生产路径为何形成平区

`src/wlr50_clean/ppo/semantic_supervisor.py:491–508` 的 `physical_potential` 对每条当前物理前驱已完成的未放置腿计算：

```text
leg_value = .1*workspace + .1*unload
          + .25*(.25*initial_boolean + .75*qualified_lift_boolean)
          + .35*carry + .2*capture
global_phi = .85*sum(leg_value)/4 + .15*finish
```

在 RR initial=true、qualified=false 时，carry=0；未跨线时capture=0。上述抬升项就是常数 `.25*.25 = .0625`，不读取 `clearance_m`、当前轮底高度或 `recent_clearance_gain_m`。在未触发其它条件的开区间内，高度偏导为0。

资格本身仍正确要求完整物理过程：`TaskEvaluator.observe:290–314` 的 initial 需要实际AIR、连续AIR样本、至少3mm向上增量和真实全身动作证据；qualified 还需要当前位置处于既有近前沿范围、真实AIR轮底≥顶面、至少8mm近期增量及动作证据。3mm是**过程内向上增量阈值**，不是要求所有场景的绝对轮底z恰为3mm；本文的3→50mm是本场景直观高度区间。

虽然 `predicate('lifted_*')` 在 `:451–462` 存在 dense 分量，它没有进入上述 global phi。并且，**单纯接入这个现有 predicate 仍不够**：其 `recent_clearance_gain/0.008` 达到8mm即饱和；动作证据和两帧AIR也很快饱和，仍不能区分后续8→50mm的目标距离。`predicate('clear_*')` 又只奖励正的顶面以上净空，无法补齐顶面以下区间。

## 已完成首回合中的直接证据

来源：`runs/ppo_semantic_v3/train/20260906T0545269489491Z_gcae1d6e7cfdd_1813c27a4d0e408aab8fde1f0c4010c6` 的 `residual_and_projection_audit.jsonl` 与 `completed_episodes.jsonl`。

首回合1027 decisions，global17793–18819，68.466667s 在 P09 以 `INCOMPLETE_CONTROLLER_BLOCKED` 结束，无任务成功。本次只读取该完整回合；读时完整 optimizer 边界已到19584，不把当前未完成 rollout 混入优化量。

筛选条件全部来自实际决策末 `physical_evaluator`：FR/FL已placed，RR未placed；RR initial=true、qualified=false、AIR且两接触均不active；RR在workspace横向/前向范围，零load fraction，至少两个其它支撑。149个满足条件的决策末记录，**phi全部是0.48078125000000005**。这组包括离地后下降但尚未重新接地的低高度样本，不能把149误写成149个新的有效向上抬升。

代表样本如下，均为 P09、其它支撑数2、whole-body actuation evidence=true、load fraction0：

| Global / tick / 时间s | 轮底离地高度mm | 相对顶面净空mm | global phi | task family |
|---|---:|---:|---:|---:|
| 18578 / 6288 / 52.4 | 3.964156 | −46.035844 | .48078125 | +.053053385 |
| 18811 / 8152 / 67.933333 | 8.005936 | −41.994064 | .48078125 | −.013352865 |
| 18665 / 6984 / 58.2 | 15.252691 | −34.747309 | .48078125 | −.013352865 |
| 18423 / 5048 / 42.066667 | 21.909323 | −28.090677 | .48078125 | −.013352865 |
| 18422 / 5040 / 42.0 | 29.539001 | −20.460999 | .48078125 | −.013352865 |

最高的决策末真实无接触高度是29.539001mm，仍未达到50mm顶面。这里直接核验的是持久化的决策末完整物理评估字段，不冒充另有每个120Hz tick的完整高度流。同期每tick native效应摘要不能用于重建未保存的高度向量。

初始事件那一行可有正 task reward，因此不声称所有行奖励都负或为零。其它平区行在没有事件/势变化时，现有 `semantic_reward.py:179–185` 给出：

```text
5 * (.995 - 1) * .48078125 - .02/15 = -.0133528645833
```

这是正常的折扣势与物理时间成本；其它四family也仍计算。将“高轮底行的总reward较负”与“抬升受惩罚”直接画等号不成立，关键是控制其它phi条件之后高度没有差分区分。

## 最小候选：只替换既有 lift-credit 中的未合格部分

可在下一显式 v3 new-MDP revision 中，仅使 global phi 的已有 lift-credit 对**真实进行中的主动AIR过程**使用连续顶面距离。不要另加第六family、每帧“向上就给钱”的奖励或phase bonus；不改现有 .25/.75比例、其余权重、hard qualification、cross/placed/success、deadline、nominal和caps。

一个无需新观测维数、也无需猜轮半径/地面高度的最小计算候选是：

```text
gap = max(0, -current.clearance_m)              # 实测轮底距当前障碍顶面的剩余距离
s = history.minimum_lift_gain_m                # 既有8mm物理尺度，不是新增成功阈值
height_progress = s / (s + gap)                # 任何有限gap都存在严格高度梯度
lift_credit = 1                               if existing qualified bit is true
            = .99 * height_progress          if eligible active AIR process
            = 0                              otherwise
```

把 `lift_credit` 放入原来的 `.25*(.25*initial + .75*lift_credit)`；保留 `.99` 为未qualified上限，只沿用既有dense predicate的“不伪装完成”约定。qualified=true仍只能由现有真实 evaluator 产生，soft值不能写回任何history位，也不能满足stage completion。

例：轮底高度3/8/15/29.539/40/49mm、顶面50mm时，未qualified的候选lift-credit约为 .144/.1584/.18419/.27828/.44/.88。高度升高则严格增加，无8→50mm截平。它的绝对增量仍受原有lift-family权重限制；这是补信号而非保证一次训练成功。此函数选择需要在下次版本中明确记录，不能说目前已经实施或验证。

`eligible active AIR process` 至少使用已有可信字段：独立 evaluator valid、无安全/任务失败、该腿前驱已实际placed、已由真实历史赚到initial、当前ground/obstacle pair都不active、连续AIR样本满足现有minimum、当前whole-body actuation evidence有效。可以复用既有near-front/lateral物理范围限制任务相关性；不能引入某个FL/RL角度、指定对角支撑、固定入场状态或历史耗时。

尤其不能使用 `initial` 一个旧bit单独授权所有后续高度。若已有初始小跳后接触墙面，再由墙面升高后短暂脱离，单靠“当前AIR + 旧initial”仍可能误计过去的接触抬高。针对这个边界，实施时必须从现有 `_samples[leg]` 的逐tick `(bottom, air, whole-body joints/commands)` 证明**当前无接触段的主动向上过程**，而不是复用接触段的 `recent_clearance_gain` 最大值；否则该候选仍不完整。允许墙边发生真正重新主动修正，不能将“之前接触过障碍”永久判死刑。所需过程判断应是当前测量的派生诊断，不另造成功标签或导入A历史。

本候选用当前高度而非历史最大高度：下降、回地、动作过程失效会撤销对应soft进度，避免赚到高度后永久保留。过程持续但高度静止应保持同值而不是持续积攒奖励；奖励仍只由原potential difference产生。不要因普通phase label改变而重置或重发这部分credit。

## 必须有的正反例（设计要求，未执行）

1. 用真实 `TaskEvaluator` 连续观察先赚到合法 initial，再让RR当前AIR高度依次3、8、15、30、49mm，其它workspace/unload/support/前驱固定：新phi严格增加；现有qualified/cross/placed仍false。不得手填initial/lift布尔掩盖接口问题。
2. 达到实测顶面后，缺近期真实增量、动作证据、连续AIR或近前沿条件之一，不能qualified；soft最多未完成上限。满足原条件才可得qualified=1，后续cross/placed和完整稳定终止仍必须另行满足。
3. 同样高度序列但一直ground或obstacle contact、只有wheel spin、无对应全身动作的 wall-only过程，不能得到相同soft高度分数。另测“先小跳赚initial→墙接触升高→被动脱离”不能继承接触高度；真正新AIR主动修正可重新积累，不永久禁止接触后尝试。
4. 被动弹跳、下降经过相同高度、未满足前驱的RL、非有限/未验证接触/几何、body collision/fall均不能通过软进度旁路。全部12残差保持，不能把零actuation等价于某个单关节不动。
5. 同一物理历史/高度仅换普通phase label，global phi相同。跨phase过程持续、不清当前真实证据；P06/P09 suffix仍区分teacher已形成的state与policy新获得的差分，不给A prefix PPO credit。
6. 完整“上升→下降→重新回到同一状态”没有新的终止事件时，不应靠重复逐帧height bonus形成可获利循环；检查真实discounted potential差分、终止phi0，禁止额外加一次高度family。
7. 旧v2和未opt-in v3仍使用原bool公式；新增诊断不改变324维编码和已有clearance物理单位。测试真实 reward calculator 的总差只来自现有task-progress potential项，不能重复reward整个family。

因此，当前证据足以支持“存在值得补齐的局部task-progress信号”，但尚不支持直接改变hard资格、降低顶面要求或宣称物理可达性已解决。候选先保留为下一保存/评估边界后的选择，不影响当前固定块训练。

## 补充：独立 AIR shaping 过程证据，不因0.5s滑窗到期撤销

### 最小状态与取得方式

不能每tick都用“当前0.5s窗口仍有≥3mm增量”作为高度进度的开关，否则合法抬高并维持净空的机器人会在0.5s后被撤分。这需要区分**取得当前过程资格**与**维持当前过程资格**。

建议每腿只增加一个独立的 `soft_air_actuation_earned: bool`，默认false；另记录 `earned_tick` 供审计（不参加公式）。AIR段起点/当段采样数可以直接由现有 `_air_count` 和 `_samples` 的连续AIR后缀得到，不必新增整段样本缓冲、历史最高高度、接触姿态或A轨迹状态。该bit仅用于未qualified的shaping权重，不写入 `initial_clearance`、`active_lift`、`front_edge_crossed`、`placed`，不进入stage completion或success判断。

取得条件必须在每个真实120Hz观察中计算，不能只看15Hz决策末：

1. 本腿现在确实无ground/obstacle接触，两对contact provenance已验证，数据有限、没有终止/安全失败。当前腿前驱已placed，任务相关几何合法。旧initial可以保留为必要条件，但**不构成充分证据**。
2. 从当前 `_samples[leg]` 向前取**截至本tick的最大连续AIR后缀**，遇到第一个ground或obstacle接触就停止。不能直接复用当前 `recent_clearance_gain_m` 或 `whole_body_actuation_evidence`，因为它们的原始0.5s窗口可能包含墙接触段。
3. 在这个纯AIR后缀内，有至少既有 `minimum_air_samples` 个样本；当前轮底高于这个后缀中较早的实测最低轮底至少既有 `minimum_initial_clearance_gain_m`（3mm）。用“当前高度减此前AIR最低高度”，而不是“窗口任何时刻的最大上升幅度”：只剩下降的被动轨迹不能凭早已过去的峰值取得资格。
4. 在**同一AIR后缀、同一向上段**内，重新计算既有全身动作响应证据：实际所有8关节累计运动达到既有尺度，或重力方向变化达到既有尺度；同时存在实际控制需求。只有wheel spin、没有对应全身姿态/关节响应不能通过。通过时置 `soft_air_actuation_earned=true`，只记录一次取得tick。

所有阈值复用现有物理证据尺度。这里没有要求RR自己的关节一定变化，也没有要求某个指定支撑组或关节角；来源可以是其它腿的支撑动作、全身运动和当前控制作用。这个条件是**软奖励证据**，不是一次额外task gate：不能据此禁止下发动作、跳过失败训练样本或阻止普通phase切换。

### 恒定关节目标仍可能在驱动上升

不能把 `command_motion > 0` 作为唯一取得条件。目标保持不变时，PD驱动仍可能使实际关节继续接近目标、转移载荷并抬升RR。现有 `_samples` 保存了全部关节实测位置、全部关节command、重力方向与wheel targets（`semantic_supervisor.py:269–289`），可在同一AIR后缀内沿用以下需求证据之一：目标变化达到现有尺度；或当前目标与实际位置仍有既有tracking误差；或存在既有非零wheel command需求。它还必须与同段真实全身响应和当前无接触向上增量同时成立。

因此正例必须包含：**command数组逐位恒定、RR关节不动、其它支撑关节在接近其恒定目标，RR实际向上**。该例应能取得soft过程资格；不能因新command变化为0而误判被动。也应覆盖恒定目标下身体姿态连续变化的全身作用路径，不能把“8关节均未发新目标”误作“执行器没有做功”。

### 维持、撤销与下降

一旦本连续AIR段真实取得资格：

- 只要仍处于该段、数据/安全有效，就**不再要求滑窗里持续存在新gain、新command或新joint运动**。保持净空1s、5s时资格不因窗口失忆而撤销。高度进度仍取当前gap，保持高度没有重复逐帧bonus。
- 当前下降使高度进度自然降低，但不必仅因瞬时负垂向速度清bit。这样微小纠正/下降后重新抬高不会人为重复“过程开始”；同一AIR段仍是同一段。下降至接触后则按下项撤销。
- **任何本腿ground或obstacle接触**立即清除此独立soft bit与取得tick，不等到phase变化或下一个policy decision。即使hard initial或qualified按原任务语义仍保留，也不能替soft重新授权。已有hard-qualified的合法TOP捕获仍通过原来的qualified=1分支，不因清soft bit降低hard资格。
- 真实episode reset、新建evaluator、安全/任务终止会清除或使其不再贡献；缺失/非法接触和几何不能默认为AIR。当前已有失败/异常处理继续拥有停机语义，shaping helper不得吞错继续credit。
- 普通phase变化、进入P09/P12、0.5s窗口滚动、恒定command都**不清除**此bit。离开任务相关ROI时可停止其当前几何贡献，但不得把phase label当作重置开关或完成标签。

这样，“初始小跳→墙接触抬高→被动脱离”会在墙接触时失去soft资格；旧initial无法恢复它。重新AIR的首样本高度只作为新段参考，**不能把接触阶段已经获得的高度变化算成新AIR增量**。真正主动修正后又发生≥3mm纯AIR向上及全身响应，可以重新取得资格，不将曾经接触障碍永久定为不可学。

这一保守取得方式也有明确边界：若上升全部发生在接触→首AIR采样之间，而之后没有可测的进一步AIR上升，现有离散观测不能证明本段无接触主动向上；soft部分会暂不取得。原hard initial/qualified规则不因此改变，已有hard qualified仍取得完整原credit。这不是新增最低起跳高度或动作禁令。

### 可识别性局限，不能夸大为动力学因果证明

当前 `JointObservation` 只有位置、速度、command及error，没有这段样本中的逐关节实际输出力矩/功率；目标误差不是实测做功。纯AIR向上、关节响应和命令需求的时间对应能排除“只有接触段升高、随后静止AIR”的明显反例，却**不能严格区分执行器持续做功与先前接触释放的弹性能/动量**。尤其有非零恒定目标、被动关节响应或惯性上升同时存在时，两个机制可在这些字段上不可区分。

因此不得宣称该bit证明了反事实意义的“高度完全由policy驱动”。可以诚实命名为 `measured_actuated_air_process_for_shaping`，报告所依据的AIR范围、当前增量、全身响应、command需求以及取得/接触撤销tick。若以后真实数据出现上述不可辨反例，再按实测执行器力矩/功率或其它物理证据单独评估；现在不新建昂贵前置门，不因无法完美归因而放宽hard wheel-only/qualified/success。

此bit属于显式新增的reward过程状态，必须通过task/审计诊断公开，不能冒充已有hard history位或声称任务MDP没变；本候选不新增324维actor编码。自然teacher prefix若沿用同一真实evaluator，应携带其实际过程状态到接管点，reward初始化于该真实当前phi，prefix期间不得记PPO credit；不能在handoff按目标phase捏造或清空这个bit来重新领分。

### 补充最小正反例

- 连续AIR主动上升取得资格后，恒定目标/恒定净空维持超过0.5s及2s：soft bit保持、phi不因过期下降、没有每tick正bonus；普通phase切换同样保持。
- 恒定目标下其它支撑关节持续接近目标、RR自身joint motion=0、RR无接触向上：取得资格；只有修改command但实际没有上升或全身响应，不取得。
- 墙接触后旧initial仍true，随即短暂AIR但无新段向上/无新段全身响应：soft bitfalse；把整个0.5s混合窗口的gain故意设很高也不能骗过真实后缀计算。
- 已取得资格后接触一tick、再AIR：先撤销，必须用新段证据重取；真正墙边主动修正的正例可以重取，不永久封禁。
- 同段下降降低当前高度分数，悬停不撤销资格；返回相同完整过程状态的循环只走原potential差分。hard initial/qualified/crossed/placed、终止结果在加入/不加入soft过程的相同观察序列中逐值相同。

以上仅是下一实现边界的明确方案，尚未执行或修改任何生产内容。

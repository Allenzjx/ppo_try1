# P06–P13 前轮 residual cap 1.2：只读候选核对

2026-09-06；未修改或执行候选，未启动 Python/Isaac，未新增训练或批准发布。当前固定训练块及其后完整评估仍由主线程完成。本报告仅针对 `execution_profile.yaml` 中 P06–P13 的 wheel `[FL,FR,RL,RR]=[1.2,1.2,.6,.6]` 候选；P01–P05、servo cap、residual/nominal slew、硬限、reward、nominal、actor 架构和异方差分布均不在候选改动范围。

## 1. 已有路径支持，不需新架构

- `configs/ppo_semantic_v3/execution_profile.yaml`：现有 `residual.phase_caps_full12` 已逐阶段逐通道配置。只涉及八行的 index8/9（FL/FR），不是把 index10/11 后轮一起放大；初始 fallback scalar 与 P01–P05 的四轮.3不应顺带修改。
- `src/wlr50_clean/ppo/semantic_backend.py:49` 的 `build_semantic_projector` 检查有序 P01–P13、每行12个有限正值、逐通道不在后阶段缩小、cap不超物理 range span。`.3 → 1.2` 前轮与`.3 → .6` 后轮均满足现有单调规则。轮速 span 来自 frozen ±2.0943951023931953，宽度4.1887902047863905；1.2低于此 loader 上界。这是配置允许性，不是最终实际 target 可达性证明。
- `action_projection.py:504–594` 已逐通道完成 `tanh → scale → mask → previous-residual slew → phase-cap → absolute interval`；没有四轮必须同cap的生产断言。`semantic_residual_adapter.py:25–120` 独立组合 policy residual 与原 controller bias，最终仍夹在 ±2.0943951023931953；controller-only 原包络不因此放松，一次 mapper advance/一次 write 不改变。`actuator_target_effect.py` 比较真实 dispatched float32，不写死.6。
- `configs/ppo_semantic_v3/action_schema.json` 仍是324输入/12动作既有顺序及“配置指定物理尺度”，无须新增动作维或换Gaussian语义。旧logprob依然针对 raw latent，不是裁剪后的实际 wheel target。

没有发现为表达该异步前后轮范围而必须修改上述生产 Python 的具体阻断；这不等于本候选测试或物理验证已通过。

## 2. 取消哪些强建议，为什么1.2不是临界饱和点

冻结源位置与当前78行事实见 `residual_wheel_nominal_cancellation_readonly.md`。−1.07属于P09 FL，不是P07 FR。下表忽略 residual slew，设 controller wheel bias=0、固定该 nominal：

| 固定 canonical nominal | 精确归零所需 residual | 候选 finite raw `atanh(residual/1.2)`，约 | 到cap剩余表达余量 |
|---|---:|---:|---:|
| P07 FR −.63 | +.63 | +.583 | .57 |
| P09 FL −1.07 | +1.07 | +1.430 | .13 |
| P13 FR −.72 | +.72 | +.693 | .48 |
| P13 FL +1.09 | −1.09 | −1.518 | .11 |

这四段原±.6不能完全取消，候选在表达层均可取消。1.2比最大1.09多.11，故无需数学上的 infinite raw/`tanh=1`才能归零。负nominal若要求净+.02，残差增加.02后也仍在候选范围内；这只是明确工程单位的抵消目标，不改用户停轮成功容差。

后轮不变；P12全轮−.3，以及P13 RL−.43/RR−.39原本已在±.6可取消范围内。P13 finite endpoint之后nominal归零，仍由真实速度、支持/ROI、命令与稳定持续等原条件判定，不把归零一个target等同整机成功。

**1.2仍受硬限交集约束。** 例如 FL nominal+1.09且residual同向+1.2请求+2.29，P09 FL−1.07且residual−1.2请求−2.27，都会触±2.094395硬限；无权把cap1.2宣称为所有nominal下完整±1.2实际余量。抵消向量本身净0/小正则不受这个硬限阻碍。

## 3. 限速与已学策略不变的准确含义

Residual rate 保持1.8rad/s²，即.015/tick、完整8tick最多.12；nominal另有3rad/s²。固定nominal且从零残差出发，仅为到达+.63/+.72/+1.07/−1.09就分别至少42/48/72/73tick。已有反向残差、handoff hold或短终止会改变实际可用时间。提高cap不加快 slew，也不保证尚未到达该范围时的目标有变化。

同一 raw 输出的前轮表达值从`.6*tanh(z)`变成`1.2*tanh(z)`，直接翻倍；经过 slew/硬限后不一定翻倍。**同一mean/std/actor权重不能保证旧物理输出不变。** 当前78行FR已学raw mean全部负，投影残差也全负；放大cap可能先扩大顺nominal的负请求，而非自动学会抵消。范围修订不是对现有失败成因的证明。

P13 nominal0、bias0、slew收敛时，.02命令停止区间对应的 raw 界由 `atanh(.02/.6)=.033345687` 缩到 `atanh(.02/1.2)=.016668210`，约减半；零附近同raw std对应物理扰动约翻倍。现有连续P13 shaping公式不变，但策略需适应更敏感的前轮停止尺度。这是风险分析，不使用独立Gaussian概率估算实际整机成功率，也不改后轮或硬成功容差。

## 4. 现有迁移路径与不可偷换的语义

执行profile属于runtime contract。直接 exact-resume 会在 `semantic_cli.py:220` 的 `_preflight_checkpoint` 拒绝源/现contract不同，不能把它伪装成 output-only 或 policy-distribution-only迁移。

现有显式 `--new-mdp-warm-start` 路径支持该配置边界：

1. `semantic_migration.py:58` 的 `build_v3_warm_start_record` 绑定六配置源/目标hash、冻结A/120Hz/15Hz/200s/运行库恒等及324固定预处理；保留源计数、spent stage budgets、原origin，不继承旧rollout/物理状态。执行范围变化列入 `runtime_changed_files`。
2. CLI在warm-start分支前已从已验metadata选择 `policy_version_from_metadata`；实际 runner 传同一 heteroscedastic-log policy version、`initialize_actor=False`。不是再次Gaussian→hetero conversion。
3. `semantic_training.py:512` 的 `_load_v3_warm_start` 先真实load并验证完整actor（含mean/logstd）、critic、optimizer源hash和identity normalizer，再显式fresh Adam **3e−5**、清rollout、restore源RNG。这不是保留原Adam moments，也不保证源adaptive LR相同；当前新MDP机制如此记录，不能在候选说明中省略。
4. `warm_start_source_execution_profile` 从源hash绑定的工作树/历史Git字节恢复旧profile；新initial名含源hash+目标commit/content hash，旧checkpoint/sidecar不可覆盖。`compare_warm_start_action` 保存同一当前观测、同raw、两个profile的各投影层，而不改环境/bridge历史。
5. 该比较只用**零残差历史、一个1/120s tick**；即使前轮scale翻倍，rate后两边仍可能同为.015，P01–P05比较更会本来完全相等。因此“initial projected差0”不能当成范围没有变化或长期物理等价。需要看scaled层及专门多tick CPU例。

候选来源必须等主线程固定实际结束checkpoint后再确定。本报告不预填源终点，不将任何架构/配置迁移算成PPO学习收益。A冻结动作不变；B原始zero-residual路径没有新命令，但没有为此伪造新的B rollout验证。

## 5. 现有测试中具体受影响点与最小增补

本次只读搜索，没有运行测试。

- 确定隐含“四轮.6”的用例：`tests/unit/test_semantic_p06_wheel_tail.py:337` 的 `test_real_projected_negative_residual_can_cancel_tail_with_one_frozen_dispatch` 使用四轮共同 `raw=-atanh(.5)`，并要求 residual四轮−.3、native全0。候选下前轮将收敛−.6而净−.3，原后两条数值断言不再成立。应保留显式旧profile测试，不削弱其断言；另用候选逐轮 `raw=[-atanh(.25),-atanh(.25),-atanh(.5),-atanh(.5)]` 验同一+.3 tail确可全轮归零，并保留一次advance/write、真实audit的原断言。
- `test_semantic_backend.py:24` 的四轮.015是**首tick rate**断言且默认v2，不是.6 cap假设，不应改；`:35`非零历史的.045也是slew断言。`test_semantic_continuous_stop_progress.py` 的测量速度.6、`test_semantic_p06_rolling_retirement.py` 的source−.63都是物理/nominal fixture，不应随候选改数值。现有retired-nominal residual-effect测试不要求四轮同cap，可原样保留。
- `test_semantic_v3_continuation.py` 已有新MDP/原预算/历史profile/旧profile等hash才断言scale无差的测试；不可移除那些绑定。`test_semantic_policy_cli.py` 有eval/video从保存metadata选型；但现有各独立测试不等价于“异方差v3 checkpoint + 仅前轮profile变化”的端到端CPU回归。

最小候选增补可集中为六组（不创建新测试框架）：

1. **配置精确diff**：对旧/候选副本断言只有 P06–P13 index8/9由.6→1.2，P01–P05、后轮、servo、120/15、slew、hard、nominal/reward/schema恒等。bad rows包括错长度/顺序、0/NaN/Inf、P07后缩cap、超span；每项仍fail-closed。
2. **表达与连续实现**：各13phase正负raw按真实cap断言；P05→P06四轮留历史/首hold不跳，8tick≤.12，非零→zero只slew衰减。不能只用一次大raw的.015验证1.2范围。
3. **旧不能、新能取消的成对例**：真实frozen nominal −.63/−1.07/−.72/+1.09，old最终不可0/new finite raw经足够连续tick可0及一个小正目标；后轮与P01–P05相同raw表达不变。使用真实adapter fixture核native sign/float32，nominal及controller bias不被改写。
4. **硬限制反例**：前述+2.29/−2.27请求夹紧、controller-only包络仍拒绝错误值；NaN/错误ID/shape拒绝、nominal geometry有无两分支以及一次advance/write audit保持。硬clip不得被“cap满足”绕过。
5. **P13精度例**：nominal0同raw旧新scaled前轮2倍而后轮不变；.02两侧浮点可分辨；较大target即使allplaced仍不满足原stop硬条件。原continuous-stop与finite/物理tests不放宽。
6. **hetero新MDP+重载**：官方runner小CPUfixture保存已有mean/logstd/critic/identity+非空Adam后，仅改前轮profile；验证新initial各权重/源RNG/原计数预算恒等、freshAdam/rollout、current sampling字段；首比较scaled的前两轮差异正确；保存后同新contract exact-load与eval/video选型保持。反例：未声明newMDP、源/目标profile被换字节、encoder变更、旧initial覆盖均拒绝。不需要实际任务成功才能测试这条软件路径。

## 6. 后续若单独选择发布，最小真实响应窗口

不在当前训练并行运行。已有 `semantic_workspace_probe.py` 可复用合法prefix、120Hz audit、无optimizer及逐段独立reset/evidence原语，但其现有CLI**不能原样声称覆盖本候选**：`old_range`选的是v2 profile而非当前v3 .6；`response_actions`同时刺激八servo和四轮，且raw magnitude≤1。1.2×tanh(1)=.913913，仍不足以抵消1.07/1.09。不能为探针篡改真实nominal或放松任何安全判定。

若主线程另授权一个小诊断，最少需要：

- 用同当前MDP的合法P07 prefix进入一个约2–4s窗口，比较当前v3 .6与候选的**单前轮**有限raw阶跃/回零；每段使用真实nominal及当前状态，记录实际prefix initial差异，不能把两个独立reset说成bitwise同态。确认真实出现的FR−.63窗口有正残差抵消、slew、native float32响应，同时保存q/轮速、pair/load/body状态；发生硬失败立即按原规则结束，诊断不是必须成功的新门槛。
- P09 FL−1.07、P13 FL+1.09/FR−.72若后来在合法连续轨迹自然出现，再用对应短窗口核强建议边界；若carry覆盖掉源建议或未到这些阶段，明确“未exercise”，不得注入假nominal或先要求完整成功才继续优化器。最初P07窗口只能验证其实际范围，不能覆盖全部四个静态极值。
- 最低后续P13自然样本应覆盖nominal归零后的前轮小raw/均值与实际速度/支持，而非仅展示更大target。现有原始native审计保持120Hz、完整字段、无第二write/advance；不为本候选减少证据频率。

本候选增加的是前轮表达范围，同时提高停止敏感度；是否采用、何时发布及实际预算仍未决定。本报告完成后停止，不监控后续结果。

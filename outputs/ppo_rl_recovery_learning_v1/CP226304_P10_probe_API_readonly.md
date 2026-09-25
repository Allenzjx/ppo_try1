# CP226304 P10 单通道诊断：既有API与最小候选（只读，未执行）

当前完整自然评估必须自然封存；本说明不授权启动探针，不改生产、配置或模型。

## 结论

**没有可直接执行的现成P10/+4探针命令。** `fr_knee_entry_offset_probe.py` 硬绑定CP230144/59e、自然P01同policy前缀、P05 age .5–.75s、FR TOP＋RL承载、固定入口REQUEST−4°；CLI仅有run-dir/max-decisions/execute。`student_entry_direction_probe.py` 虽允许指定CP/hash/HEAD，但强制P09合法AIR并同时替换FLk/FRk/RRh三个raw，也无nominal前缀。都不能只改命令参数冒充本次诊断。

**无需生产源码变化，但需要另行批准后新增一个outputs-only薄诊断driver及定向测试。** 复用组件如下：

- `semantic_video_cli.build_video_core(role='C', semantic_version='v3', experiment_id='rr_rl_timing_policy_learning_v1')` 与官方 `checkpoint_loader(args, contract)`：验证并加载CP226304的439/512新身份、Adam/normalizer等；诊断无更新。
- `CheckpointPolicyPrefixRequest(target_phase='P10', teacher_offset_decisions=0, source='successful_nominal')`、`CheckpointPolicyPrefixRslAdapter(...).install_prefix_policy(zero12, exact_nominal_provenance)`：与训练现有通路相同，在自然P01按同一N/mapper真实前进至P10，不teleport；nominal provenance包含当前execution/task/runtime哈希及439/full12。不要换成旧 `PrefixSemanticIsaacBackend` frozen-FSM takeover。
- 同一核心连续接管；复用纯 `replace_fr_request(original12, cap, request)` 仅替换raw[3]，通过现有投影/限速/mapper/HISTORY/原子写入。复用原probe的逐tick观察、原sample与实际applied分开记录、最终unchanged校验。新driver无PPO存储/更新，前缀和诊断均零学习信用。

精确绑定：CP226304路径为 `outputs/ppo_rr_rl_timing_policy_learning_v1/branches/cp225280_front_preserved_v1/checkpoints/history/checkpoint_step_000226304.pt`；SHA `c0f4727bcc4e1963b5e9d079158dde1f5f662390397b61049ff893c40e1910f6`，manifest SHA `4d1e103507e7083f9d1ecb6d5ff230bd3d106054da2af0ef85e0ce81b9c8569b`，HEAD `892385cba8a7089b52567bb7f558c018fac82a77`。物理seed4001/确定性与已有方向诊断一致；checkpoint内部训练seed仍1001，不改保存RNG或模型。原FL assist ON，后腿辅助OFF。

## 触发、预算与退出

在**P10首个学生决策**检查当前物理有效、RR合法TOP真实承载，并复用现有 `rear_dependency(...).support_transfer_permitted`（RR＋当前FL或FR桥接承载），不用历史placed替代承载。既有P07采样P10仅一个decision，因此P05式age≥.5s必定有漏触发风险。入口不合格则明确NOT_TRIGGERED，不移动到虚构状态；prefix miss/fresh-P01 fallback也不宣称P10入口。

保存当时 `core.bridge.previous_projected_residual_full12[3]`，与actor审计的previous filtered REQUEST对齐；这才是固定入口值。从它smoothstep到**入口＋4° REQUEST**（不是FR knee绝对+4°、不是每拍＋4°），1s ramp/1s hold/1s向**当前**学生请求release。其他11 raw逐拍保持当前学生值；干预改变真实历史，所以它们不等同未干预基线轨迹。

允许P10→P11→P12正常跨阶段，不因阶段变化清HISTORY或立刻结束；这些阶段FRk cap均112°。保持现有支撑许可；RR掉载/必要bridge消失、新物理异常、cap变化则120Hz latch，下一决策不再干预（已发8tick动作最多余7tick，原安全中止即时）。不自动重触发，不保持RR目标，不打开任何rear helper。

建议仅120个学生诊断决策（8s，含3s干预及释放后观察），另计真实前缀，仍受总200s/原安全限制。原900decisions是自然P01总预算，不能误当后缀预算。输出标 `N-prefix initialized diagnostic`，不是自然P01 PPO成功。恢复只意味着取消通道替换、继续同一实际HISTORY，不意味着物理状态瞬时回到未干预轨迹；磁盘模型/原run没有改动，无git rollback。

若获批准，最小**待实现**入口可绑定上述CP/HEAD/P10/+4，仅保留run-dir、max-student-decisions和显式execute参数；目前不存在该文件或可运行命令，勿直接调用旧脚本绕过绑定。

**因果限制：** “固定入口＋4”同时取消了原FRk请求的后续负漂。与未干预轨迹相比，效果不能全称为4°方向收益。若只问正向4°的独立增益，需固定入口0偏移的配对诊断；该控制也不是无干预。无需把这组对照变成继续PPO的门禁。

最低证据：prefix交接tick/状态与不重置历史；入口REQUEST、cap、原mean/sample/logp、applied/raw[3]；逐tick同拍source/mapped N→REQUEST→headroom→FINAL→actual；RR当前力/XY/gap、FR hip/knee跟踪、FL支撑、四轮source/最终/实测、RL资格/接触、实际CoM/机身位移；三秒及提前释放时间；模型/Adam/normalizer前后hash不变，实际更新0。必须测出FINAL/实测响应，不能仅inverse-tanh正确便声称物理改善。

## 更低集成风险的现成选项：先P10连续on-policy 512

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 892385cba8a7089b52567bb7f558c018fac82a77 -SemanticVersion v3 -ExperimentId rr_rl_timing_policy_learning_v1 -Stage phase_suffix -FromPhase P10 -TeacherOffsetDecisions 0 -PrefixSource successful_nominal -Decisions 512 -MaxDecisions 3000 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -CheckpointOutputBranch cp225280_front_preserved_v1 -Checkpoint 'outputs\ppo_rr_rl_timing_policy_learning_v1\branches\cp225280_front_preserved_v1\checkpoints\history\checkpoint_step_000226304.pt'
```

这是**已有可用命令，但本次未执行**。它不需新driver/生产变化，继承439/512、实际LR≈1e−5、Adam、HISTORY、前段KL保护；一块完整有效512预计到CP226816/PPO1728/Adam34560。阶段正常连续，P10不是固定512步姿态；可覆盖捕获后保持→P11准备→P12 RL，但只能按实际阶段/事件计样本。

相对探针，它的**集成与训练分布风险更低**：没有人工raw替换，直接产生合法on-policy学习样本。它**不保证物理风险更低或必定有RL有效样本**：当前负FRk分布和source脉冲可能重复掉载/硬限。P12首块仅一次新RL资格且无越沿，P07块虽两次学生RR捕获但RL资格0，故P10是合理中间课程、不是已证成功。

建议先完成当前自然评估，再优先考虑这一最小P10块；按完整更新保存后检查RR支撑保持、FRk饱和和RL实际资格。若重复同一饱和/掉载且数据仍不能区分原因，再决定是否批准单通道物理诊断。不要无检查无限重复P10，也不要让探针成为训练门槛。N前缀已捕获RR必须记为teacher成果，不能算学生首次捕获；自然P01覆盖仍须保留，前段保护不被后腿课程取代。

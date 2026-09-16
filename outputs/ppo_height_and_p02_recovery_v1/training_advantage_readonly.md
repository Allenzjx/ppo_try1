# 下个真实训练块：只读优势审计候选（未应用）

产物：

- [training_advantage_readonly.patch](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_advantage_readonly.patch)：apply_patch 格式，单一生产文件 semantic_training.py，4 个窄插入区。
- [training_advantage_readonly_tests.patch](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_advantage_readonly_tests.patch)：7 个定向 CPU 用例的新单元测试，亦未应用。

## 采样与时钟

每次现有 env.step 返回后，仅暂存本 rollout 的实际 decision request phase、end phase、episode endpoint tick、实际执行 N（1–8）及终止/时间元数据；不使用 curriculum/from_phase 充当样本阶段。teacher roll-in 不在 collector 的这些已发行 PPO 决策中。phase 映射按现有 storage 的 [time, env] 排列，global decision = rollout 首决策 + time×num_envs + env。

现有 compute_returns 完成后，直接复用原本就会保存的 CPU snapshot，读取 returns、old values、stored advantages、rewards、dones。raw GAE = returns − old values，减法保留 storage dtype，统计再用 double。该量不是重算 critic、episode_return，也不是反标准化后猜测的 advantage。

每完整 rollout 在 update 前写一行 advantage_audit.jsonl，含：

- 整体和实际 request phase 的样本数、终止数，以及 raw GAE、stored advantages、old values、returns、rewards 各自 min/max/mean/population std、正/负/零和非有限计数。
- 短终态的准确原始 phase、tick/N、真实 reward/return/GAE、bootstrap flag；普通 phase-change 单列且不标 done。
- 本次预期 update 编号及 global decision 范围；明确 pre-update、此条本身不证明优化已完成。须和现有 optimizer_updates.jsonl/最终 manifest 关联确认。
- rollout 尾的 terminal/nonterminal env 索引。真正 last_values 已由既有 compute_returns 使用，但本补丁不新增 forward 或伪装为另行测得的尾值，单独值为 null。
- 当前全 rollout 标准化模式有明确标签；若已有其他配置启用 minibatch 标准化，则正确标成“此时仍为 raw GAE”，不虚报 standardized。

小终态例如 N=3 仍是一个策略决策，按现有 once-per-issued-action gamma；补丁不把它扩成 8 tick，也不另乘 gamma 的分数次幂。

## 严格不变项

不修改 storage 张量、returns/GAE 计算、normalizer、372 输入、HISTORY/rho、sigma、reward、raw logprob、done、bootstrap、optimizer 或学习率；不新增 actor/critic 前向，不消耗采样 RNG。利用既有 snapshot，不再次复制整批 GPU storage。非有限统计输出 count/null 而非 JSON NaN，不修复数据或把坏数据改成有效值。

当优化前发生异常，可能已有 pre-update 行，因此它不能充当“更新完成”收据。未完成 rollout 不会输出完整优势审计，也不回填旧 rollout。

## 已完成候选验证与应用后测试

仅内存执行候选 helper + 测试正文，7 项通过；未 import 修改后的生产模块、未改磁盘生产文件、未启动 Isaac、未加载真实 checkpoint。

覆盖：raw/standardized 区分；T×N 实际 phase 归属；storage 与请求元数据和 CPU RNG 不变；3-tick HARD_JOINT_LIMIT 真终态；普通跨阶段非 done；非有限值只报告不修补；错 topology/缺 phase 拒绝冒用课程；非默认 normalization 标签。

另将 4 个补丁 hunk 在内存匹配当前生产文件，全部唯一匹配，完整候选文件 ast.parse 通过；没有写回生产。应用后建议运行新测试及原有 test_semantic_training.py 的真实 CPU update/failed-episode roundtrip 用例；原有原始 action/logprob、normalizer、终态断言保持。

适用下个现有 1024–4096 决策块（当前 rollout128×1即8–32行优势审计）；不创建新训练工程。待正在运行的 zero 退出、根代理 review 后再应用。若 runtime contract 因 instrumentation source 变化需要新身份绑定，应保留原兼容 actor/critic、Adam、normalizer/HISTORY，并在现有入口开始 fresh rollout；不能为日志变更重置网络。


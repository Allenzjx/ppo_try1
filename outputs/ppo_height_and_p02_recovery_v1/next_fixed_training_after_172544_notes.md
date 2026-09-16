# CP172544 后下一正式块：固定配置、补充真实 P02 失败覆盖

建议与根代理实际选择一致：从 CP172544 在 HEAD `4a58c0190ef744a2e40c3668d9042caf85389385` 续训 **自然 P01 连续 2048 decisions**（计划16个完整128样本 rollout / 320 optimizer steps），随后保存、重载，立即进行自然 P01 确定性正式 C。根代理已报告启动 session56645；本文不读取其活动训练数据、不把计划数当已完成数。

理由：τ=.5 首块只有3次自然入口、484个P02样本、1次P02真实终止；496个样本在P05。8次更新之后 C172544仍于723 ticks / 6.025 s在P02 HARD_JOINT_LIMIT，FR只有I/Q，无C/P。当前不是“没有更新”的问题，而是单次负终态与不同随机轨迹尚未产生可复现的确定性恢复。增加一个明确有界的16-update块，可在同一过程重新采样失败附近及其后果，并减少多次Isaac启动开销；不承诺2048会对应多少P02样本、独立入口或成功episode。完成后必须看实际C与分阶段样本，不自动无限重复。

## 已封存 τ1024：P02 失败确实进入更新

只读现有 receipt、8条 `advantage_audit.jsonl` 及已完成分布摘要；未加载模型/rollout张量，未重算回报。

- Update1307包含 global171678、episode第158个决策：P02 HARD_JOINT_LIMIT，真实末区间1256→1264共8 ticks，terminal=true、bootstrap=false。stored reward=return=`-40.96377182006836`，old V=`-19.387659072875977`，raw GAE=`-21.576112747192383`，标准化优势=`-2.252927780151367`。同编号完成optimizer记录有20 steps、actor changed、finite nonzero gradient；不是只有pre-update审计而未执行更新。
- 全484个P02样本分布在1306/1307/1308/1312/1313：raw GAE均值`-0.8851853756865194`（339正/145负）；stored优势均值`-0.13209838562593065`（302正/182负）。每个完整128-rollout共同标准化，不是P02自己标准化，也不是每个wheel独立优势。
- 1306较早126个P02样本的raw GAE均值+0.6925948461；该rollout结束时尚未看到第158决策的终态，使用官方critic tail bootstrap。之后发生失败不能倒填/重用旧rollout；更长总采样预算也不会把128步rollout变成无限实测回报。这是当前on-policy近似的限制，不是已证实的GAE实现错误。
- 9次普通phase交接全部非terminal且bootstrap=true；两次真实终止（P02 hard limit、P05 blocked）均不bootstrap。P05终态只有1 physics tick，保存 reward=return=`-41.080265045166016`。末尾249决策为非terminal P05尾段，不称完整episode或任务成功。
- 8次完成更新均actor changed、有限非零梯度；有效Adam LR范围1e-5..2.25e-5，最终1e-5；网络、Adam、identity normalizer、RNG、计数与保存重载链均有现有receipt支持。未发现需要先修的optimizer/credit bug；不能从总梯度范数大于1就推断单个模块梯度裁剪损坏。

C172544是冻结确定性评估，不进入PPO数据：91次issued/90次returned，末次720→723只有3 ticks且未返回，不能凭视频补造训练reward或old log-prob。训练中的共同动作负优势也不能单独证明“某一个轮的取消”是唯一因果；同闭环全部12通道都在更新。

## 保持配置及已有官方入口

保持τ=.5、HISTORY rho=.9、role372、raw12全开放、现有N/mapper/projection/reward/physics、γ=.9985、λ=.99、128 rollout、5 epochs×4 minibatches、seed1001、N1。有效LR从Adam恢复（当前1e-5），不能用runner配置默认3e-5覆盖。CP172544已属于当前policy版本/HEAD，不再使用旧171520温度迁移计划，也不warm-start或清空网络；每次合法新仿真从P01开始，storage fresh。

官方命令（记录根代理所选参数，本文不执行）：

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 4a58c0190ef744a2e40c3668d9042caf85389385 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage full_episode -FromPhase P01 -Decisions 2048 -Seed 1001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000172544.pt -CheckpointIntervalUpdates 10
```

`-Decisions`是该run新增训练采样预算，不是最大episode长度或P02样本数；`-MaxDecisions`不提供训练内短reset开关。wrapper实际传`--headless`，backend实际`step(render=False)`；不承诺实时倍率，仍受真实仿真、采集和写盘开销影响。当前full_episode预算已用88448/100000，剩11552；2048在现有预算内。source actor仍是现有网络，不做teacher cloning。

若以后需要更多独立自然入口，现有合法方法是多个完成run之间保存/重载，例如4×512或2×1024，不是在普通阶段交接/FR capture时伪造done；run末非终态按既有value bootstrap，下一run合法reset。它们改变采样覆盖并增加启动成本，也可能截去较晚真实终态；本轮先不再拆成4×256。固定HISTORY表示episode reset时历史归零合法，但不能在P01→P02→P03交接清历史。

有限后段维护可以在正式C之后从**刚实际保存的最新CP**执行 `-Stage phase_suffix -FromPhase P06 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Decisions 256`，其余同上。该路径真实P01→P06教师前驱，不从已抬RR快照开始；前驱数据不计PPO，P06交接后仍全12。此前256维护实际排除1344教师决策，成本显著；不能先用这个后缀掩盖P02失败。当前mean尚不能通过P02，不推荐checkpoint_policy prefix作为立即到达P06的手段。现phase_suffix预算剩26016。后缀成功不等于自然P01完整成功。

## 依据与边界

依据：`training_P01_temperature1024_actual.json`、`training_temperature1024_distribution.json/.md`、`p02_policy_recovery_checkpoint172544.json`、`training_2048_reward_credit_summary.md`；新credit仅用封存run `20260915T0556048565573Z_g4a58c0190ef7_df022d61cbd2428ab7f7c0dc6697fb3e/advantage_audit.jsonl`，与receipt完成update号配对。

P02 effective σ均值.08588是同决策条件创新尺度，不是边际/平稳sigma；z总体均值.00479、population std1.00209，12通道native作用均实际非零。目前没有充分证据再改温度、HISTORY、clip/reward或永久mask。成功zero保持SUCCEEDED；固定配置续训的收益只能由后续新checkpoint实际重载任务结果证明。

代码核对：`semantic_cli.py:145`请求与剩余预算校验、`:917`自然/真实前驱后缀分流；`semantic_training.py:1298`累计stage预算、`:1305`128-rollout完整更新、`:1391`实际请求phase而非teacher统计、`:1404`raw action/old distribution严格storage一致、`:1435`官方更新与保存；`scripts/run_semantic_ppo.ps1:93`参数转发/单Isaac流程。未改生产、未运行仿真或新模型forward。

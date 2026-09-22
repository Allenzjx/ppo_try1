# Block11：首个 P04 课程 rollout 交接核验

结论：首128样本核验通过；这是 successful_nominal 初始化的 suffix PPO，不是从P01完整策略成功。只检查 rollout001486 / CP194688，未改变运行、配置或生产。

- **前缀零信用**：实际188个 N+0 decisions、1504 physics ticks；全部raw/residual为零、policy_credit=false。在12.533333 s的真实P04接管，无fallback；任务时钟还剩187.466667 s，不是恢复历史快照。
- **交接连续**：末前缀native command tick1683→首learner1684；首动作物理区间1504→1512，P04→P05未done。保存372输入里的previous N/实际drive/mapper补偿匹配前缀末端，最大误差8.59e−7（float32编码）；127个后继的raw H逐值相等，filtered request/drive/N最大差2.69e−6。原有−3.75°等mapper补偿没有被清空。
- **N连续不等于N恒定**：前一nominal四轮为(-.79,.3,.51911,.3)，新P04当前N为(.3,.3,.3,.3) rad/s；旧值仍完整保存在previous_nominal/previous_drive。关节N延续(0,−22.9,0,45.9,17.5,0,0,0)°。这是不同current/previous调度值，不能把它们强制相等或当作残差；本检查未重新审计参考动作设计。
- **真实学习动作**：128条正式raw/mu/sigma/logprob逐值匹配原storage；实际P04=1、P05=127，全128为Gaussian单次采样非零动作，许可12通道均1。1024个物理tick的native派发审核通过；每decision末的实际native目标改变11–12通道，首decision为12。许可/派发通过不等于任务成功或每轮牵引相同。
- **计数隔离**：source194560→CP194688，新增128 decisions、1 PPO update、20 optimizer steps；只phase_suffix预算+128，188前缀不入buffer。aux ledger完整保持7 accepted /8 attempted，新aux=0。实际learner控制器为SemanticControllerAdapter，N提供器NominalMotionProvider；不是旧teacher backend。
- **Adam与LR**：源actor hash等于实际update before；Adam参数键及除LR外全部group字段保持，所有已存per-parameter step6620→6640，Identity normalizer hash不变。实际保存LR从1e−5变为2.25e−5，update1486记录同值、KL均值.014184395；生产官方PPO仍是desired_kl=.01的adaptive schedule。它按各minibatch KL更新LR，不能从整update平均KL反推逐步轨迹，也不能把配置初值3e−5当作实际恢复LR。没有手工改LR或重置Adam的证据。

CP194688 SHA: 5a6afb7cc5dabbe979df7af3014f068439e53a84ee20b41a3f739eedbff54405。

JSON含实际边界、误差、计数和固定首128行/首prefix区间哈希；不对仍在增长的整日志作最终封存声明。生产证据接口：semantic_checkpoint_prefix.py 的 _roll_in/step；semantic_env.py 的逐物理步_history更新；semantic_training.py 的实际collector/storage和独立budget/counter保存；安装版rsl_rl/algorithms/ppo.py:269–294的既有adaptive KL更新。未执行网络forward、optimizer或Isaac。

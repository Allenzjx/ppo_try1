# non_residual_refine_v1：隔离zero收尾候选

用户要求的 `non_residual_refine_v1` 在现有工程命名规则中映射为 `runs/ppo_non_residual_refine_v1/`、`outputs/ppo_non_residual_refine_v1/` 和 `configs/ppo_non_residual_refine_v1/`；只是隔离候选，不是重建训练工程。

该入口仅允许自然P01的 `semantic_prior_eval`（以及只读preflight），无policy checkpoint、无PPO训练/更新。实际运行是成功N加零policy residual，不能称为learned PPO。原成功zero的源码版本、配置、数据与视频全部保留。

六配置来自 `ppo_task_first_recovery_v1`，只有stage spec中的 `nominal.final_stop_owner` 选择独立 `source_home_after_physical_stop_v1`；reward epsilon0及其余五份配置原样保留。新mode的实现、实际终态home与camera结果由各自真实run记录，配置接通和测试通过本身不代表物理成功。

恢复点及保护基线引用见 `outputs/diagnostics_v1/checkpoint_and_protected_baseline_references.json`。不要在此目录写入旧运行的拼接数据或将旧zero成功视频冒充新收尾候选。

# 输出侧候选：迁移、CLI 与冻结 prefix

本目录不是已部署版本。生产仍为 `f2e552406ea74a5aae2803347d1c8bebe261910d`；没有改 `src/`、六份配置或真实 checkpoint，没有启动 Isaac、提交 Git 或增加正式训练计数。统一 patch 仅供根任务在三个旧版本视频完成后的安全边界审阅。

## 单因素边界

新增 `request_history_kernel_factor`，schema 为 `wlr50_clean.cap_transition_request_history_same372.v1`。仅允许 `fl_capture_quality_v1` / v3 / 372 / N1 / 旧 quarter kernel → 新 cap-transition REQUEST-history kernel。六份配置必须逐字相同，source/target runtime 仅允许六个明确文件变化；完整文件 hash、限定 AST 作用区、physical setter/reset/step 调用不变均绑定到 plan。

图必须仍为相邻 P01→…→P13→SUCCESS、120Hz/15Hz，obs 原有 stage/age/completed/raw/REQUEST 切片及固定尺度、13×12 cap 表必须一致。旧 actor 类与旧 HISTORY 函数不在许可改动区。CLI 主循环只增加冻结 prefix 的一个来源字典合并，不放宽执行主循环。

保留全部 actor/critic 张量、Adam moments/step/options、Identity、训练 RNG、全局计数、实际 LR1e−5；runner 静态声明 LR3e−5保持原值，恢复后有效 Adam LR仍1e−5。旧 rollout/pending action 必须为空，新数据重采。既有 ancestry 保留，新 factor 通过现有 resume_ancestry 持久记录。迁移本身 +0 decisions / +0 updates。

同权重行为**并非完全等价**：仅观测 gate 成立的 cap 增大首拍，conditional mean 的历史中心变为前拍 filtered REQUEST 在当前 cap 下的 inverse-tanh 值。σ仍 learned σ×.25；其他帧和通道旧 raw 中心。REQUEST 不是 final target、native effective residual 或实际关节运动。

## prefix 与恢复

迁移当块 prefix 的张量来自旧 CP，函数来自新 actor，因此增加独立 `source_policy_contract`、`effective_policy_contract`、源/有效 runtime hashes 和完整 plan path/hash/checkpoint binding。`policy_contract` 是实际执行 kernel，不能伪称源 CP 已保存新 kernel。新 CP exact-resume 不需迁移 plan，但 source/effective contract 与 runtime 必须一致。

精确 class、forward/get_latent、official Normal、Identity、参数hash以及冻结副本独立 storage 检查全部保留。前缀仍无 PPO 信用。旧三种 HISTORY prefix 原有行为不变。

## CPU 证据与限制

`test_request_migration_prefix.py` 16项通过：读取真实 CP180480 的 metadata/完整性，但不加载或重写其真实训练状态；对当前六份实际候选源码和真实六配置执行 AST/字节绑定、plan JSON 重建及篡改负例。测试用虚拟 TEST revision 与仅测试的版本字节读取映射，未建立真实目标 Git commit，生成的临时 plan **不可部署**。

另用官方 CPU RSL 和合成环境创建临时旧 quarter checkpoint（非用户 CP），填充 Adam 后真实迁移加载，逐项验证权重、Adam、normalizer、RNG、LR、计数；拒绝非空 storage/pending action/错误 actor。新 kernel 冻结 prefix 无 RNG 消耗、来源字段正确；128合成 decisions /20 minibatches 后实际 save/load 与普通 exact resume 通过，ancestry 保留。现有未修改 `semantic_video_cli.checkpoint_loader` 的 deterministic 与显式 stochastic 两入口在真实 gate 观测下输出匹配同 kernel，证明未遗漏入口，**不等于物理视频或任务成功**。

联合 actor/contract/audit/likelihood 与旧 .5/.25/HISTORY 回归为193项；另47项既有 prefix 回归通过，合计240项不同测试。`candidate_all_tests.xml` 是最终统一执行回执。仅 `conftest.py` 的包搜索覆盖与旧 reward 文件只读位置映射用于 CPU 隔离；生产文件不动。

采用前，根任务仍须在明确安全边界审查/apply、运行实际生产路径必要测试并提交真实 target HEAD，然后用正式 `runtime_contract` 和精确六文件 hashes 重新调用 `build_migration_plan(..., request_history_kernel_review={reason, reviewed_code_sha256})`。不能使用 CPU fixture 的虚拟 revision/临时 plan，不能据本候选宣称稳定性改善、任务完成或新增真实 PPO 更新。

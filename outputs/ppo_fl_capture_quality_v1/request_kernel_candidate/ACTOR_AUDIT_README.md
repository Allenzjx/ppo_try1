# 隔离候选：actor / distribution / training 审计部分

不是生产部署。仅复制源码并编辑本目录下的副本；生产HEAD仍为 `f2e552406ea74a5aae2803347d1c8bebe261910d`。当前运行和三个正式视频使用旧版本，未切换到本候选。无Isaac、无GPU模型、无新真实训练计数。

新版本：`cap_transition_request_history_heteroscedastic_log_temperature_quarter_v1`；新class：`SemanticCapTransitionQuarterHistoryMLPModel`。新factor：`request_history_kernel_factor`，schema `wlr50_clean.cap_transition_request_history_same372.v1`。同372，全12，rho=.9，temperature=.25；σ、N、cap、reward、physical/evaluator/mapper均不变。

只有 encoded stage age恰0、前驱已完成、且该通道cap增大时，把上一拍 filtered REQUEST 通过新cap的atanh逆变换作为rho中心。其余帧/通道使用旧raw；原始raw历史不改写。该REQUEST不是headroom/final-slew之后的实际动作，更不是关节实际运动。

已实现：

- 旧三个history类/旧history函数保持原样；追加纯观测中心函数与新class，单次MLP、单次draw，无私有历史/cache。
- 完整policy contract/runner配置识别新版本，显式声明mean改变；不是temperature-only迁移。
- training候选新增窄factor验证/loader接点，source限定旧quarter/FL/372/N1/实际Adam LR1e−5；保存兼容weights、Adam、Identity、RNG，拒绝非空rollout/pending action。
- actual request audit旧类仍原schema，新类增加gate/physical REQUEST/caps/有效center，原raw另存；actor与audit调用同一纯函数。真实分布一致性检查仍保留。
- likelihood audit继续观察官方PPO对stored raw的真实logp；仅新版本provenance补全相关观测索引，无额外forward/draw。

测试：`actor_request_tests.xml` 32/32；联同145项旧 HISTORY/.5/.25测试的 `actor_legacy_regression_tests.xml` 共177/177通过。新测试覆盖真实CP178688+rollout1362、请求审计开关RNG/sample精确一致、20minibatch官方PPO审计开关后weights/Adam/RNG逐位一致、first ratio≈1、同阶段/零history/P02→P03/输入错误/混合batch与padding/元数据混用拒绝。

测试进程通过 `conftest.py` 将candidate目录放在Python包搜索首位；没有写生产模块。旧回归第一次仅因复制源码的 `__file__` 相对资源位置缺少config而失败，测试随后只读映射该单个旧reward文件到原repo同文件，加载函数/配置字节/参数均不变。两个独立CPU optimizer使用深拷贝checkpoint字典，避免Adam.load_state_dict保留输入CPU tensor storage而污染对照。上述测试均不计入真实训练。

另一个受委派代理在本目录拥有migration/CLI/frozen-prefix三副本及其独立测试。最终统一patch仅供根任务审阅；必须等待这部分通过与根任务安全边界授权，不能单独启用本三文件片段。

`build_candidate_patch.py`仅生成review patch及hash回执，并执行只读`git apply --check`；不会apply。缺失正式target HEAD/真实runtime绑定时，候选factor证据不得冒称正式迁移授权。

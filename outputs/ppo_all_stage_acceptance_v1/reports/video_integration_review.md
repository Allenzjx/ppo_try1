# 当前 all-stage 视频接线只读复核

范围：当前 `semantic_video.py`、`semantic_video_cli.py`、`run_semantic_video.ps1` 及其直接调用的配置、HISTORY actor、共同评价和 reset 路径。未执行 Python/Torch、测试、Isaac、录制或旧视频重封装；不影响正在运行的 B2。以下是源码结论，不是当前 checkpoint 的真实视频验证。

## 结论：基础功能已存在，三处必要接线仍缺失

不能沿用旧报告“372 视频未实现”的笼统说法：HISTORY372 模型、严格 checkpoint 加载、确定性实际 kernel、自然 P01 单episode、共同 evaluator、原速录像和完整解码路径均已有实现；但当前视频入口不能直接正确录制新 all_stage_acceptance_v1 的 C。

| 位置 | 确切缺口 | 最小接线 |
|---|---|---|
| `semantic_video_cli.main`；`semantic_video.video_configuration/build/validate`；PS wrapper | 普通 CLI 已支持 `--experiment-id all_stage_acceptance_v1`，视频却只将 semantic_version 传入 runtime_contract/config 选择。PS 没有 ExperimentId，run/log 路径仍固定旧 semantic namespace。新 CP 会被路径或严格 runtime/config 校验拒绝；手动传 experiment-id 也不能弥补内部漏传。 | 将显式 experiment_id 贯穿 runtime_contract、core config、capture/source manifest、独立 replay validator；PS 使用同一 experiment namespace。保留缺省旧 v2/v3 行为、隔离目录和严格 hash/config 校验，不增迁移绕过。 |
| `semantic_video_cli.checkpoint_loader:30–33` | 已读取 checkpoint、解析实际 policy_version，但构造 runner 时漏传已预检 observation_layout。`SemanticHistoryMLPModel` 缺省 layout 严格要求324，真实372输入会拒绝，不是自动推导372。 | 复用普通 CLI `_observation_layout_options(args)` 或相同严格 resolver，向 construct_semantic_runner 传 role layout；provenance 明列 observation layout/维度与 policy contract。无 NewMdp、无padding、无模型转换。 |
| `semantic_video_cli.build_video_core:79–82` 的 A | A直接建立原 IsaacFSMBackend，没有普通 `_evaluation_legacy` 已有的 all-stage measurement reader 替换。选择新共同 task spec 后，旧 raw reader不足以提供当前姿态下 collider/body bounds 和 surface-bearing证据。 | 复用 `_evaluation_legacy` 的局部 dependencies.reader_from_scene→SemanticSensorReader 配方，仅测量适配器；冻结 A controller/mapper/reset/control 源不改。B/C 已由 SemanticIsaacBackend 的所选task spec选择该reader。 |

相关入口：`semantic_cli.version_paths/_request_paths/runtime_contract/_preflight_checkpoint`；`semantic_training.construct_semantic_runner`；`semantic_legacy_evaluation._evaluation_legacy`。无需新建视频系统或修旧 A 容器。

## 已有能力与边界

- `checkpoint_loader.action` 用真实 `TensorDict` 调用 `runner.alg.actor(..., stochastic_output=False)`，因此只要布局接对，走的是新 actor 的 `0.1*μ+0.9*previous_raw[195:207]`，不是旧 mean 或 JIT/ONNX 导出。当前372保留该slice；episode内观测由正常 core.step 更新，phase handoff 不额外清历史。loader使用正式checkpoint加载，结束比较actor/critic/optimizer/normalizer哈希，optimizer更新计数为0。
- `validate_video_args` 已要求 eval、seed4001、N1、FromPhase P01、offset0、无NewMdp、max_decisions3000、无额外decisions；core第一次自然reset、无snapshot、无prefix。录制不得使用 suffix 成功替代完整P01。
- `capture_semantic_video` 在120Hz实际tick上使用独立 `PhysicalEvaluationRecorder`，按共同success/真实终止停止；允许最后一个15Hz决策只完成1–8tick并显式记录。离线validator再逐tick用同TaskEvaluator重放，不靠旧FSM phase/success标签。新配置必须同时绑定capture和replay，不能只给controller换spec。
- v3 pre-roll取已有180次zero settle中的最后64tick（0.5333s），不增加physics、sensor读或controller step；原v2另外64tick hold/refresh的路径不应套到当前v3。记录器在app.close前finalize，源文件不拼接/加速，后续有frame ledger、PTS与完整decode检查。
- 200秒文件上限已存在：120Hz、15fps、`PRE_TICKS=64`、`POST_TICKS=184`、最多3000frames。公式 `1+floor((64+episode_ticks+184)/8)` 使当前完整context最多容纳 **23751个episode tick（197.925s）**；这不等于共同任务200秒的所有合法episode都能附带这些额外首尾帧。新共同success已经含固定1秒收尾，现视频还会额外物理hold184tick（1.5333s）。应在录制前明确版本化的视频context预算；不可截中间动作/加速，也不可把额外视频hold当成新增训练成功门槛。当前没有证据需要修改任务200秒。
- post-roll继续原mapper而非reset，但显式停止四轮并保持上一nominal/servo requested bias，不是继续采样HISTORY策略。它是无学习信用的真实录像context，需要诚实标注。当前已有headroom和tracking_reference mode传递；没有本次执行证据证明开启全部新mode后post-roll始终维持原target或稳定。几何投影激活的最后ACK、当前requested与effective不同等情况需一个定向接线测试，不要把旧假体测试泛称实际372全链验证。

## A/C同条件比较应保留什么、披露什么

已有比较发布器严格比较seed、camera、runtime_contract、semantic_version和pre-roll来源，并要求两条源各自通过共同评价与完整解码。这些是必要绑定，但不能证明动态初始化逐位一致。

二者可保持相同资产/场景、自然P01 reset请求、seed4001、180zero settle、相机、120/15Hz和新共同评价；A仍用冻结 controller/mapper，C使用新监督器和学习residual。**A在settle最后30tick读姿态并计算level参考，B/C使用配置固定chassis axes、校准样本数0**；reader缓存及初始控制状态也不是同一个对象或同一队列。不得为了“配对”偷偷把A改成C reset/supervisor，或注入C已达状态。

最终记录实际reset metadata、初始raw/current bounds、level reference来源/窗口、settle/prime/extra ticks及首个native ACK。可以称“同场景/seed/共同物理定义的A→C方法对照”，并列上述控制初始化差异；若声称严格动态初态配对则需要真实字段支持。B/C在同版本下的比较更直接隔离学习作用。历史A视频若runtime/初态不匹配只标非配对，当前A重跑失败也不能篡改历史成功。

## 最小定向测试清单（本次仅建议，未运行）

1. 扩 `test_semantic_video_v3.py`：all-stage selector从PS/CLI传到五配置、runtime、source与replay；错namespace/错spec拒绝，缺省旧版本保留。
2. 新/扩loader小CPU seam：真实HISTORY372 checkpoint严格加载，固定观测结果等于正式eval kernel、历史变化会影响mean、维度/contract错误拒绝，actor/critic/std/normalizer/optimizer不变；无前缀和optimizer更新。
3. A build seam：只替换measurement dependencies，controller/reset路径保持原实现；B/C选择同all-stage测量/schema。比较清单显式保留A校准与B/C固定参考差别，不改A。
4. 复用现有settle-tail测试，保180原step/原write，v3 extra0；新增当前tracking-reference首tick bootstrap及post-roll lastACK参考连续性。对geometry激活/非零headroom截断的最后ACK给正反例。
5. 新共同success含收尾的endpoint/replay一致；200秒frame预算边界与失败保留；后续失稳不得裁掉发布成功。继续用已有decode/PTS校验，不扩成训练前视频gate。

只需后三个视频生产文件及其专属定向测试；共同A测量初始化可复用现有配方，不修改冻结A。待取得并保存/重载真实完整成功CP后再实施必要录制接线及采集；本报告不声称当前已经产出可发布视频。

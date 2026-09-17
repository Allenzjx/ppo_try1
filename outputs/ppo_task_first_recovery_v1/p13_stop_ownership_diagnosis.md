# CP177024：P13 停车交接诊断

结果保留为 `INCOMPLETE_CONTROLLER_BLOCKED / POST_COMPLETION_LOSS`，不是完整成功。四腿已放置，无机身碰撞。直接终止原因是固定 post-completion 窗口末实测轮速不受控，不是 home error 32.08°；home 不是当前硬验收门槛。

| 事件 | 正式全通道 CP177024 | 保留成功 zero |
|---|---:|---:|
| 最后一腿 RL 放置 | 6057 | 6727 |
| P13 任务入口 | 6064 | 6728 |
| 物理完成后1秒窗口开始 | 6057（仍 P12） | 8737（P13） |
| P13 nominal 开始四轮 +0.3 rad/s | 6065 | 6729 |
| P13 nominal 发出 stop | 运行结束前没有 | 8737 |
| 窗口结束 | 6177，不通过 | 8857，成功 |

在 C6057，四腿历史、平台区域、承载与实测速度满足同一 evaluator 规则，最大 wheel 速度0.2441109 rad/s < 0.25，因此窗口真实启动。此前 P12 的四轮 nominal 已为0；P13 首拍却重新执行源建议 `wheel all 0.3`。源 P13 首个 authored stop 位于 +16.733333 s，远晚于本次剩余不到1秒窗口。

现 `_observe_final_stop_owner()` 只在“已有 P13 source layer 且此前 nominal 已发 stop”时允许接管。C6064 首次 P13 snapshot 当前 controlled、region、support 均有效，但 provider 尚处于 P12，因此不能继承已有的零速停车建议；6065 源推进重新获得所有权。后续 owner 始终未取得。这是**已完成物理目标与源建议停车交接的时序不兼容**，不是 nominal=0 被 residual 抵消，也不是 wheel mask 遗漏。

仍有独立策略/负载因素：窗口开始后 C6060（尚未重新推进）RL 实测0.2646529 rad/s已经短暂超过0.25，目标为0.1096958 rad/s，来自 wheel residual。不能因此预言仅修 nominal 就必然成功。终点 C6177：

| canonical wheel 顺序 | FL | FR | RL | RR |
|---|---:|---:|---:|---:|
| N rad/s | .3 | .3 | .3 | .3 |
| PPO 有效 residual rad/s | .0304263 | −.0256365 | .1096207 | .0007116 |
| 最终 target rad/s | .3304263 | .2743635 | .4096207 | .3007116 |
| 实测 rad/s | .3291455 | .0698579 | .4917325 | .2580642 |

机身线速度0.0357211 m/s < .05、角速度0.0625961 rad/s < .30；区域与支撑仍有效，FL/RL/RR 实测轮速超限。固定1秒判定按现有规则正常执行，不能把旧失败改成成功。

建议仅修共享 nominal stop 交接：首次 P13 调用即可依据**已经由真实物理 evaluator 启动的 post window**及当拍有效区域、四腿几何、可信承载、至少两个顶面支撑接管停车；不再要求 provider 先执行一个 P13 stop。持有上一拍 nominal servo，明确 nominal wheel stop，全部12维 PPO residual 继续叠加。窗口时长、起点、终止标准、硬失败保持不变；窗口内短暂 measured-control 丢失单独记录，不冒充当前受控。

成功 zero 的 post-start 在8737前一直为 false，8737与既有 authored stop 同拍，故该修复应不影响此前路径；需回放验证8737后的完整命令仍相等。修复后必须新版本迁移、真实重跑，不能以计算诊断代替成功。

证据：`c177024_p13_stop_window.json`；提取脚本：`check_p13_c177024_stop.py`。

## 安全边界后的实施与计算验证

仅修改 `NominalMotionProvider._observe_final_stop_owner()`，实现上述交接；`acquisition_semantics=post_window_triggered_nominal_stop_takeover_v2`。新 `current_post_window_takeover_eligibility` 允许在已启动窗口内接管，旧 `current_entry_eligibility` 明确作为严格当前受控状态诊断，不冒称低速仍成立。evaluator、源时钟、残差、mapper、历史、配置均未更改。

70项 stop-owner/source-partial-order/nominal-history 定向测试通过。完整保留 zero P13 输入重放为 tick6728–8857，共2130行实录物理输入、2129条对应下发记录：新旧 nominal Full12 差异0，旧计算输出与原记录差异0；阶段、tracking、controller bias相同；原/新 owner与post窗口均在8737开始。终点8857不是decision tick，task级result仍为空；此报告不拿该字段替代既有真实成功结果。

receipt：`retained_zero_p13_stop_replay_receipt.json`；脚本：`replay_retained_zero_p13_stop.py`。新supervisor SHA256：`04c2b9a8a6f71598cda8f9ca9b5c4282f14a061e13f49d8377a59746b518166a`。这证明原zero完整P13的命令等价，不是新物理运行，也不预言新PPO一定成功。

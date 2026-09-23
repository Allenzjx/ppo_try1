# 412 维 RR 捕获重基设计复核（仅静态）

结论：赞成“两项显式 candidate-entry baseline＋既有捕获锚点”的小扩展，不必为普通slew新增源时钟暂停。但以下边界必须明确，不能把仍饱和于-58°的旧绝对candidate在0.75秒内放回。封存末拍 pre-assist RR=[-0.3235,-58]，FINAL=[-1.7859,-25.3612]，尚无RR TOP；后续捕获/失接触规则属于待验证的新控制贡献。

## 正确的重基通路与观测布局

令 `u = geometry-corrected native + bounded controller + filtered requested residual`，必须取在 `project_semantic_servo_headroom` **之前**，而不是现有 `candidate_before_assist`（它已受到headroom投影）。真实合法TOP/bearing交接时：既有两anchor取上一实际FINAL，两新增baseline取同拍u。跟随目标为 `anchor + (u_now-u_entry)`；先对重基坐标施加原2°安全余量，再保留最后硬限、原slew及一次atomic派发。不能用未投影u取消安全余量，也不能把u称作未经过任何过滤的raw Gaussian。

新增两项应追加在410/411，**保留旧389–402 assist字段和403–409任务字段原位**；将assist14直接插成16会错移已训练任务列。mode/原因的新语义与不截断数值需版本化；actor/critic保留旧410列并零追加，Adam矩同列迁移，Identity normalizer扩展，旧未完rollout清空。raw sample/logp与请求HISTORY不被重基覆盖；同一状态的zero-current-policy反事实使用同一entry baseline，不重新生成baseline来抵消历史策略贡献。

## reserve许可：现有字段能做，但“fresh progress”不能偷换

40°之后最多一次12°/12 active-s、总52°/44s；near-top gap≤25mm，再同时要求当前有效AIR/Q/cross/XY、≥2实测其他支撑、两关节跟踪≤3°、剩余物理余量和真实下降窗口。任何接触停止下探；不因phase、anchor或contact重置累计budget。

注意当前 `window_elapsed_s` 除真实下降≥0.2mm外，还会在 `_anchor`、contact、HOLD→AIR及hip→knee切换清零，故 `<2s` **不独自证明**fresh下降。若坚持412，可用新增公开mode/blocked_reason组合编码“reserve已由真实下降获取资格”；仅真实peak-to-current下降分支发放，失资格/reanchor撤销，保证schema validator/context能从公开状态重建许可。不能新增未观测bool。若这种mode复用过于复杂，宁可再追加一个有明确时间语义的progress-credit字段成为413。近台面阈值只是reserve入口/当前保护，不是TOP或承载证明。

## contact loss与源节点：只有真正等待才停游标

在正常 `REBASED_FOLLOW` 中P10源正向增量和policy变化均实际进入目标，三节点照常各消费一次；不因为最终slew未到位就回放/宣称未消费。日志分开记源节点消费、u增量、重基请求、投影后FINAL和actual；输入的源正增量不保证总u增量必正，policy/controller可能抵消。

短暂失RR TOP但仍合法AIR/Q/cross、XY和其他当前支撑有效时，不立即抹掉重基baseline或退回旧candidate；可保留跟随并叠加剩余有进展的有限recapture（只增加公开knee anchor并计入原累计budget）。不能把lost bearing当新RL卸载许可。若需真正独占RR pair、保持上一FINAL等待跟踪/恢复，则仅暂停依赖它的尚未消费P10节点，其他有用前送/FR/FL准备继续；恢复时明确从上一FINAL与当前u重新锚定，不追赶或重放旧节点，预算不续发。重新锚定必须是日志可见事件，不可每拍都rebase而悄悄取消policy/source增量。

现有policy请求本身有cap，但 `u_now-u_entry` 仍可能含大幅源/mapper/controller变化；不能把原final slew当接触稳定保证。保留跟踪/当前支持反馈及重基后的余量投影，记录每个分量与首次接触损失。如果另加局部变化率/幅度保护，需公开其版本及实际projection，不能悄悄把RR residual置零。源节点已进入投影而受限，与源节点根本未获执行机会是两种不同情况。

## 退役：RL刚AIR不是撤去RR支撑的正确默认时机

当前代码用RL lift history触发retired，随后0.75秒回绝对candidate；不能直接沿用。RL刚起摆通常更需要RR支撑。建议保留可随u变化的重基RR支撑直到RL真实放置并形成当前支撑，或其他已证实不依赖RR的安全交接；这不是锁死RR角度。然后才向原candidate有限、反馈允许地释放偏置，继续观察当前RR接触/跟踪与其他支撑。若这段需要等待，禁止提前启动相关home撤支撑动作；若RL已AIR，RL落脚恢复必须继续。全球200秒和真实安全始终优先。没有当前物理证据时不能只因旧placed、source endpoint或phase标签退役。

最小反例：饱和-58的u入口下，正向P10增量仍改变FINAL；入口/失接触重锚不跳变；同一节点不重放；fresh窗口因contact重置不能获reserve；当前AIR历史placed不授予新RL卸载；RL新AIR不触发回旧hover；零当前残差反事实使用原baseline；412追加不移动旧7位任务输入。未实施任何生产改动。

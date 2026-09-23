# RR 有进展的有限捕获恢复与 P10 接续草案

仅静态设计；未改生产/配置、未运行 helper/Isaac。依据 root 提供的已封存 CP220544-v4 同控制反事实结果，不冒称新录像或成功：109.625 s / tick 13155，P09 `finite_search_travel_or_margin`；RR 当前 AIR、真实 lift+cross、force=0、未 placed；FL TOP，FR/FL 已 placed。RR hip=-1.785913°、knee=-25.361176°，累计 travel=40°、active elapsed=30 s，gap=13.4286 mm、front=50.0889 mm。因此是**仍有真实下降时用尽搜索行程**，不是已触地，也不是32 s总时间用尽。

## 1. 一次有限、反馈许可的 knee 余量，不续任意等待

封存测量：92.733 s 的 gap 41.287 mm / knee -42.253° → 96.733 s 的33.610/-38.253 → 102.400 s 的24.496/-32.586 →109.625 s 的13.429/-25.361。末段约1.53 mm/deg，朴素外推还需约8.8°；这是浮动机身的相关实测，不是恒定Jacobian或成功保证。

建议候选只在原40°边界之后启用**一次最多12° knee正向 / 12 active-s的剩余预算**（约9°实测外推＋3°有限不确定余量）。每拍仍要求：物理有效、当前RR有效AIR与lift/cross、合法顶部XY、实际其他承载≥2、hip/knee实测跟踪≤现有3°、现有2 s /0.2 mm下降反馈窗口有效，以及真实正向关节余量；任一失效即暂停该增量，任何真实接触立即停止下探。GROUND/墙接触不得延续旧AIR资格。保留1°/s，不恢复hip搜索，不复位累计travel/elapsed，不因phase/失接触再发一份reserve。剩余gap不下降、跟踪失败或几何退出时不能靠这12秒续命。

这个最小候选可沿用410维：`travel_used_deg >40`明确表示reserve；总硬上限52°/44 active-s（hip仍≤20°/12s，knee≤32°/32s），既有14项把已用预算、目标、窗口、blocked全公开，无新隐藏计数。必须同时版本化 `_search_exhaustion_reason`、snapshot validator、`advance`、context许可与manifest/观测语义；不能只改timeout或一处40。保留原窗口修复、全局200 s、物理硬限与唯一派发。12°是本条轨迹支持的有限候选而非永久扩大所有探索：若此候选也无实测进展，应停在证据处改动作功能，不重复充值。若以后改成运行时自适应斜率/预算，必须新增并观察斜率窗口及分配预算，不能塞入未公开缓存。

## 2. 捕获后按新落点交给 P10，不向旧悬空绝对值释放

当前 `apply_rr_capture_assist_snapshot` 将两个RR通道在0.75 s内混到动态 N+policy；P10 knee 三节点约0/7/14tick已消费完毕。若捕获需把knee继续正向到约-17°，原-34.6→-29.3→-27.2°绝对组明显可能重新收腿；不要将这一预测当已发生的触地丢失。本次仍无RR TOP，须后续真实TOP→bearing→P10验证。

最小接续建议是**公开的捕获锚定、相对P10任务增量**：在真实TOP/bearing且合法交接时锁存最后FINAL的RR pair，先持有P10尚未消费的源时钟，再将P10有效knee增量相对其源入口重基到捕获目标；新policy变化相对该交接拍的有效policy请求继续参与。RR hip暂由捕获锚点＋后续policy增量连续提供，不瞬时回旧绝对nominal，也不是永久锁角。首节点登记为显式重基零增量，后续节点各消费一次；暂停时不追赶，不在assist遮蔽节点后偷偷标成已执行，不重复播放。源group、原绝对值、重基delta、实际owner/FINAL、cursor均落日志。

这比“延长0.75 s”更能避免回旧hover，但属于声明过的控制接续，不是网络学到。当P10相对动作完成、当前RR承载/跟踪及接收准备允许时有限释放偏置；失接触暂停**新依赖RR承载的卸载**并允许recapture，不撤掉一切FR/FL准备。RL已真实AIR时继续其越沿/落脚，不把它冻住等RR复位。释放每步仍必须观察当前接触与真实支撑；不把old placed、phase或两点数量当稳定证明。

这里不应为强保410维隐藏状态：复用现有两个capture锚点还需公开交接基准（至少RR两通道有效candidate/policy-entry baseline）及暂停的P10源进度；若现有观测不能唯一重建cursor，再追加该一项。分轴独立释放则还需相应独立fraction。建议小幅schema扩展并保留兼容actor/critic列权重，新增列零初始化；明确Adam对应矩迁移、Identity normalizer扩展、旧未完成rollout清空后重新采样。B/C/训练/评估相同路径，raw Gaussian/logp/HISTORY不被辅助重写。

## 3. 有界验证点

只需定向反例：40°后仍下降可用一次reserve；无进展/失支撑/ground不可充值；首TOP发生在最后预算步仍可确认/正常handoff；P10隐藏时不消费、重基节点不重放；捕获后final不跳回旧hover；RR当前失接触不得启动新RL卸载、RL已AIR恢复不被冻住。随后真实自然P01继续至RL/终态，不在RR接触时截断。先保留本次失败证据与原候选，不改前段、几何辅助或late全身高度来混淆此项归因。

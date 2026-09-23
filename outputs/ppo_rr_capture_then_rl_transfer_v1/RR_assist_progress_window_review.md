# RR 助控 gap 进展窗口：只读设计候选

2026-09-22，当前 `a546` 冻结运行期间。仅阅读 `semantic_rr_capture_assist.py` 和 profile 源码及主代理提供的数值；没有读取大 live trace、运行 Python/测试/物理、修改生产或训练。

## 现行判定的准确含义

`RRHipOnlyCaptureAssist.advance()`（约239–263行）使用：

`improved = window_start_gap_m - current_gap_m >= 0.0002`

只有改善才把基准改成当前 gap，并把局部窗口 elapsed 清零。因此在连续 AIR 搜索中，这个字段实际接近“最后一次至少改善0.2 mm后的较低基准”，**不是窗口内最大 gap，也不是最近下降趋势**。gap 恶化不会刷新计时。跟踪、支撑、安全、当前资格等任何 reason 出现时不继续下探；局部 `window_elapsed_s` 只累计获准 DESCEND 的真实派发 exposure，不是持续墙钟/仿真时长。满2 s立刻输出 BLOCKED；BLOCKED 后若后来 gap 真正低于旧基准0.2 mm，现有代码可以重开局部窗口。

总 `travel_used_deg <= 20°`、`descent_elapsed_s <= 12 s` 是累计获准下探预算，contact/hold/release/re-anchor 不会将其清零。12 s不能被描述成从初始化开始的绝对仿真时限；global200与任务局部终止独立存在。mode=BLOCKED 不享有 DESCEND 局部恢复许可。

提供的实况：入口 hip11.964°/knee−51.611°、gap28.662 mm；约2 s后 hip7.964°、gap约64 mm。**这仅证明窗口内最终 gap 比入口更大，当前 `gap_not_improving` 与既有净改善准则一致。** 若中途实际是“28.662→65→64 mm”，现行基准确实会漏记最后1 mm的局部下降；若只是“28.662→64 mm持续上升/持平”，则没有已证的漏判。当前不能由 hip负移4°就认定负向 hip有效；late FL/RL 全身动作及 geometry 退出是同期混杂因素。

## 唯一最小候选：版本化的局部峰值下降准则

待 sealed 同拍轨迹确认存在局部下降后，才考虑一个单因素版本：把窗口参考定义为本局部窗口内合法测量的 `peak_gap`。

1. 只在原合法 AIR/XY/资格/实测其他支撑/跟踪条件满足的测量上更新 `peak_gap = max(peak_gap, gap)`。**上升或新的峰值只更新参考，不重置任何计时，也不直接获得继续下探许可。** 无效传感器/失去支撑/接触/硬安全等仍先处理。
2. `peak_gap - gap >= 0.2 mm` 才记为“当前全身状态下测得的局部净空下降”，重开原局部2 s窗，并以该拍 gap 开始下一窗。不是任务成功、TOP、承载或“RR hip 的因果贡献”，也不进 PPO reward/优势标签。
3. 原20°总量、12 s累计获准下探、双关节3°跟踪门、原速率/限位、contact停止、释放/退役、global200/物理安全全部不变。BLOCKED 只能在相同合法条件及真实下降证据下恢复；恶化本身不恢复。重复波动可能消耗更多既有预算，但不得无限刷新总量/总时间。

这是**改变助控继续/阻塞的控制语义**，不是修日志。若实现，峰值是影响动作的公开状态，必须进入 snapshot/validate/actor观测/保存重载；可在显式新版本中复用原单个 gap scalar 的位置并准确重命名其语义，保持410维数，但不能拿旧 v1 的 `window_start_gap_m` 名字静默解释成峰值。相应 strict 迁移需保留网络/Adam/RNG/谱系并弃旧 rollout；当前不准备或应用该迁移。

## 封存后选择前的最小证据与正反例

只需覆盖 assist入口到首次 BLOCKED及紧邻后段：按真实tick列 gap、窗口基准/elapsed、实际与target RR hip/knee、late首次派发/geometry退出、other supports、currentQ、contact。确定最大gap的tick，以及随后在原窗耗尽前是否至少下降0.2 mm、实际目标是否被跟踪。未记录到的峰值不能估算补造。

| 定向例（合成测试候选，未执行） | 期望 |
|---|---|
| 28.662→65→64 mm，合法且跟踪 | v1不算净改善；候选只承认65→64的局部下降，并保留此前总travel/exposure。 |
| 28.662→64 mm持续上升，或64 mm持平满2 s | 峰值增长不刷新时钟，仍BLOCKED。 |
| 64→63.9 mm（仅0.1 mm下降） | 不达阈值；不重开窗口。 |
| 反复上升/下降0.2 mm | 可形成多局部下降，但20°/12 s累计耗尽后不可再DESCEND；stage/contact/reset局部窗口不得清累计预算。 |
| 下降同时存在失去支撑、XY越界、资格无效、实际跟踪超3° | 不因此恢复下探；安全/原执行门优先。 |
| 真实接触/TOP、RL已qualified lift、进入释放/退役 | 原 contact停止和释放/退役优先，不能用peak下降跳回搜索。 |
| 同状态snapshot保存/恢复 | 峰值、局部时间、累计预算与下一拍动作完全一致，无隐藏窗口状态。 |

建议目前先保留真实BLOCKED结果并核对 sealed 轨迹；只有证明确有被入口基准掩盖的下降，才值得选择这项有限候选。它不能保证能够降低到TOP，更不能解决由全身构型、当前 knee 保持或其他腿动作造成的根本几何不足。

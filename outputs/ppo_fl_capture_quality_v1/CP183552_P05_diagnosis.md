# CP183552 确定性 P05 失败：相对 CP182528 的增量

封存 source：`runs/ppo_fl_capture_quality_v1/video_eval/validation/20260918T0851231634993Z_g3a50657a96c9_ceb7a3f60fb049a28fbd55c9143d710d/source`。CP183552、seed4001、deterministic conditional mean。结果为 **DIAGNOSTIC_FAILURE / INCOMPLETE_CONTROLLER_BLOCKED**，不是越障成功。

FR placed=1594；FL crossed=2588，始终没有 placed。终态 tick **5796 / 48.3s / P05**：FL AIR、gap **25.992981mm**、front **152.231935mm**、0N。物理 evaluator 本身仍 valid，无碰撞/安全失败终止；直接来源是 `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`，本次 `stall_diagnostic=false`。

## 为什么是 48.3s，而旧评估是 50.3s

当前 placed_FL 进展公式为 `.35*lift + .35*crossing + .30*(.5*top_geometry + .5*top_samples_fraction)`（`semantic_supervisor.py:1091`）。两次均已 lift/cross、无 top 接触样本：

- 本次 gap 25.993mm **超过配置 top_gap_max=25mm**，`within_top_xy=true` 但 `top_geometry=false`，所以 progress=.70。允许时长为 `30+min(10,30*.5)*.70²=34.9s`；P05 进入 tick1608/13.4s，恰在 `13.4+34.9=48.3s` 结束。
- CP182528 gap19.283mm、`top_geometry=true`，progress=.85，允许 `30+10*.85²=37.225s`；入口 tick1568/13.066667s，物理采样第一次达到该时限是 tick6036/50.3s。

**不是 front>0.15m 直接导致这次时限变化**；该 near-front 上限用于其他 lift/soft-credit 条件，而这里 qualified lift/crossing 历史已成立。实际差异是当前垂直几何份额及入口时刻；没有换用旧终止时间或修改判据。

## 末段 FL 同拍控制链

下表为 tick5796；所有物理角度均为 canonical servo °，actual 是该物理步后读回值。base/conditional/raw 为本次真实请求记录，无额外网络前向。

| FL通道 | base μ | conditional=raw | N | mapped N | filtered REQUEST=有效残差 | final | actual |
|---|---:|---:|---:|---:|---:|---:|---:|
| hip | +.248478 | +.244155 | 22.8 | 23.468834 | +4.309496 | 27.778330 | 27.194547 |
| knee | −.070625 | −.067146 | −13.4 | −12.15 | −1.609089 | −13.759089 | −13.755439 |

不是只凭 mask=1 判断：选取的 7 个同拍记录均有 setter/映射/原生派发验证，headroom clip 空；FL REQUEST=effective，最终 target 等于限速前 candidate，没有该拍残差被覆盖或清零的证据。该结论仅覆盖所取记录，不冒充全程逐tick新审计。

| tick | hip REQUEST / final / actual ° | knee REQUEST / final / actual ° | FL gap mm |
|---|---|---|---:|
| 5776 | +4.299382 / 26.518216 / 26.987655 | −1.597017 / −13.747017 / −13.749327 | 25.3940 |
| 5784 | +4.300348 / 26.519182 / 26.988552 | −1.598401 / −13.748401 / −13.750683 | 25.4306 |
| 5792 | +4.301344 / 26.520178 / 26.989509 | −1.599853 / −13.749853 / −13.752080 | 25.4627 |
| 5796 | +4.309496 / 27.778330 / 27.194547 | −1.609089 / −13.759089 / −13.755439 | 25.9930 |

相较 CP182528 末 hip REQUEST +4.176237°，本次仍保留约 +4.3° 非零确定性残差；knee 已从旧 +0.579196° 改为本次 −1.609089°，但任务结果依然悬空。可确认“均值策略仍未形成有效捕获”，不能据此把 hip 或 knee 单独定为充分原因。mapped hip 在最后半个 decision 改变1.25°，因此不能将 final 的变化全归于 policy。

终态 body z=.099009m；其他接触对反力分别 FR 障碍12.1614N、RL 地面14.6324N、RR 地面1.84585N。它们是实际 contact-pair 反力，不把 FL 悬空虚称承载。当前 P06 从未进入，不能把 P05→P06 cap 交接当作此次直接失败原因。

## P05 offset200 的实际已观察状态

P05 进入 **tick1608**，首次 P05 actuator dispatch 为1609。200个决策后对应 **tick3208 / 26.733333s**：FL qualified/crossed=true、placed=false、AIR/0N，gap **24.990360mm**、front **103.050057mm**、within_top_xy=true、top_geometry=true、phase_progress=.85。hip REQUEST +4.259996°，final26.467722° / actual26.938539°；knee REQUEST−1.522614°，final−13.672614° / actual−13.674444°。

这是明确的“越沿后仍未接触”入口，不是从已 placed 状态开始；但约25mm也不能称最后几毫米近接触。它只证明固定CP183552、seed4001完整评估里的实际可达状态，**不是训练seed1001的 prefix receipt 或保证**；当前 block12 以其真正执行的前缀回执为准，本诊断未读取活动训练。

仅生成本说明及同名 JSON（7个物理代表行）；无新仿真、CUDA、模型前向、optimizer、频谱分析或生产修改。CPU helper 已自然退出。

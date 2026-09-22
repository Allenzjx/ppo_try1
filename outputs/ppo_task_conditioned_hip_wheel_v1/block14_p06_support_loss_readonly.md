# Block14：P06 首次 FL 当前支撑丢失固定窗口（只读）

结论：FL 的历史 placed 事件是真实 tick **2957**，P05→P06 在 tick **2960**。P06 第18个学习请求（global **197508**，作用 tick 3097–3104）发生第一次当前 TOP 支撑丢失：3096 端点仍为 TOP，3104 端点为 AIR 且 `consecutive_air_samples=8`，所以首个 AIR/失去当前 TOP 的精确物理 tick 是 **3097**。3120 的 `consecutive_air_samples=24` 严格证明连续 AIR 到3120；3128 的 `consecutive_top_samples=3` 只证明当时这段 TOP streak 从3126开始。3121–3125的逐tick接触未落盘，所以首次recontact不能伪造为3126，首次失载持续时间只界定为 **24–29 ticks / 0.2000–0.2417s**。这不是把历史 placed 撤销：窗口内 FL `placed_history=true` 67/67，而当前端点仅 TOP 26/67、AIR 41/67；FR 则 TOP 67/67。

## 固定读取边界

- run：`runs\ppo_task_conditioned_hip_wheel_v1\train\20260921T1355291518150Z_g649ccd906421_aac78369d59846409cb2d4039083e87a`；仅读取仍增长 audit 的前 **437** 条完整行，到 global 197557 / episode tick 3496 立即停止；未读取其后内容，也没有全文件 hash。
- tick3496 位于已经完成并落盘的 rollout/update1509 边界内（完整边界 global197632）；只检查 rollout/likelihood 文件存在和字节数，未加载 `.pt`、torch、模型或 Isaac。
- 分析窗口是行370–437：P05→P06 过渡端点2960，加67个P06端点2968–3496。前64个P06请求止于tick3472。

## 当前接触与历史事件

|事件/端点|FL 当前状态|FL gap|FL bearing|FR 当前状态|body forward|phase progress|task potential|
|---|---|---:|---:|---|---:|---:|---:|
|placed/P06过渡端点 2960|TOP|-0.939 mm|6.802 N|TOP|-176.805 mm|0.000000|0.471824|
|首个P06端点 2968|TOP|-0.939 mm|4.967 N|TOP|-172.625 mm|0.000000|0.472468|
|最后TOP端点 3096|TOP|-0.612 mm|1.486 N|TOP|-141.187 mm|0.092979|0.478650|
|首个AIR端点 3104|AIR|+0.397 mm|0 N|TOP (12.310 N)|-140.646 mm|0.099915|0.454932|
|重获TOP端点 3128|TOP|-0.007 mm|4.880 N|TOP|-140.563 mm|0.116269|0.477577|
|固定尾端 3496|AIR|+29.864 mm|0 N|TOP (11.750 N)|-99.257 mm|0.301896|0.434131|

3104 的 FL 是 `within_top_xy=true`、AIR、force=0，不是落到 ground；FR 仍承载。第一次失载当步，body 仍前送 **+0.541 mm**、phase progress **+0.006937**，但 task potential **-0.023718**。首个P06端点到首次AIR端点累计前送 **+31.980 mm**。到tick3496虽累计前送 **+73.368 mm**，FL 已连续 AIR 127 physics ticks（从3370到3496）且 gap **29.864 mm**；因此“body前送”不能代替“FL当前承载”。

## 首次失载的同拍六通道控制链

顺序固定为 **FL hip / FL knee / FL wheel / FR wheel / RL wheel / RR wheel**；前两项单位deg，四轮单位rad/s。“REQUEST”是策略物理 residual；“actual”是保存的同次 dispatch tracking 前关节位置（转canonical）与决策端点 canonical wheel qd。final逐通道等于 mapped baseline + effective REQUEST（最大误差 0.0e+00）。Hip/knee 的 mapped baseline 仍受 source mapper slew，因此不能拿 `nominal + REQUEST` 冒充 final。

|端点|当前FL|nominal|mapped baseline|REQUEST|final|actual|
|---|---|---|---|---|---|---|
|3096|TOP|+24.900 / -13.400 / +0.300 / +0.300 / +0.300 / +0.300|+22.400 / -12.150 / +0.300 / +0.300 / +0.300 / +0.300|-6.235 / -11.133 / +0.065 / -0.022 / +0.279 / -0.089|+16.165 / -23.283 / +0.365 / +0.278 / +0.579 / +0.211|+17.204 / -23.393 / +0.322 / +0.195 / +0.808 / +0.143|
|3104|AIR|+24.900 / -13.400 / +0.300 / +0.300 / +0.300 / +0.300|+22.400 / -12.150 / +0.300 / +0.300 / +0.300 / +0.300|-4.972 / -11.360 / +0.185 / +0.095 / +0.297 / -0.118|+17.428 / -23.510 / +0.485 / +0.395 / +0.597 / +0.182|+17.210 / -23.420 / +0.485 / +0.343 / +0.753 / +0.094|
|3128|TOP|+24.900 / -13.400 / +0.300 / +0.300 / +0.300 / +0.300|+22.400 / -12.150 / +0.300 / +0.300 / +0.300 / +0.300|-6.826 / -10.564 / -0.175 / +0.074 / +0.152 / -0.136|+15.574 / -22.714 / +0.125 / +0.374 / +0.452 / +0.164|+15.346 / -22.585 / +0.152 / +0.108 / +0.651 / +0.170|
|3496|AIR|+24.900 / -13.400 / +0.300 / +0.300 / +0.300 / +0.300|+22.400 / -12.150 / +0.300 / +0.300 / +0.300 / +0.300|-9.822 / -6.508 / -0.263 / -0.260 / +0.127 / -0.178|+12.578 / -18.658 / +0.037 / +0.040 / +0.427 / +0.122|+12.640 / -17.515 / +0.036 / -0.174 / +0.520 / +0.157|

在首次失载端点3104：FL hip/knee final 为 **+17.428° / −23.510°**，保存的实际位置为 **+17.210° / −23.420°**；四轮 final 为 **+.485 / +.395 / +.597 / +.182 rad/s**，实际为 **+.485 / +.343 / +.753 / +.094 rad/s**。与3096相比，FL实际hip/knee只变化约 **+0.006° / −0.027°**，接触却从TOP转AIR；这是同步事实，不单独证明哪个控制通道导致失载。

## H / cap transition

首个P06请求 tick2961–2968 的 cap（同一六通道顺序）从 **+18.000 / +24.000 / +1.000 / +0.600 / +1.000 / +0.600** 变成 **+32.000 / +36.000 / +1.200 / +1.200 / +1.000 / +0.600**；gate 为 `[True, True, True, True, False, False]`，即 FL hip/knee、FL/FR wheel gate=true，未改cap的RL/RR wheel为false。上一P05物理REQUEST **-2.103 / -5.799 / -0.609 / -0.026 / +0.272 / -0.166** 被原样记录为 `previous_filtered_request`（最大差 1.90e-07），转换后的 H 是 **-0.066 / -0.162 / -0.559 / -0.021 / +0.279 / -0.285**。该gate只出现在首个P06请求，之后到3496全部false。

首次失载请求3104没有cap切换：H **-0.197 / -0.320 / +0.210 / -0.019 / +0.287 / -0.149** 与上一请求3096的 selected raw逐项相等（最大差 0.0e+00）；在rho=.9下 conditional mean为 **-0.203 / -0.322 / +0.171 / -0.017 / +0.277 / -0.157**，本次 selected raw为 **-0.157 / -0.327 / +0.215 / +0.079 / +0.307 / -0.199**。`tanh(raw)×cap` 的 bounded candidate 是 **-4.972 / -11.360 / +0.254 / +0.095 / +0.297 / -0.118**；随后既有projection把FL wheel按request rate limit从 **+0.254** 限到上表 **+0.185 rad/s**，其余五个所列通道没有该差异。也就是说，3097掉载不是cap transition同拍突变；它发生在P06固定cap、常规H递推的第18个请求。P06没有触发只适用于P10–P12且RR历史placed的receiving-wheel ×3 gate。

## 与已保存 B_current 的简短比较

这里只读既有 `currentB_event_windows.json` 汇总，不重扫Recording。B 是 `nominal_without_learned_residual`：FL placed2677、P06 entry2680，同样相隔3ticks；其完整 P06 rear-approach 2680–5160 中，逐physics FL obstacle contact **2481/2481、AIR 0**，保存端点 FL TOP/support **311/311、AIR 0**，FL gap最大仅 **0.0143 mm**。所以当前P06的41/67 AIR端点以及3370–3496连续AIR，不是成功N窗口中观察到的正常短暂载荷变化，可严格称为“相对B的support-retention degradation”。但两次轨迹没有时间对齐或状态匹配，这不是raw差分或因果counterfactual，不能据此单独归因某个policy通道或更新。

## 边界

逐physics consecutive counter允许精确定位首次 AIR tick3097、严格连续AIR到3120，以及3126开始的当前TOP streak；它不能恢复3121–3125未保存的中间接触，因此 `first_exact_recontact_tick=null`。其余表格仍是15 Hz端点。这里没有扭矩、slip、接触点连续轨迹或因果干预；也不覆盖reward/GAE/likelihood（由独立审计负责）。本审计没有加载checkpoint/torch、没有运行Isaac、没有修改生产或训练门禁。详细数值和全部选定端点在同名JSON；重现入口为本脚本。

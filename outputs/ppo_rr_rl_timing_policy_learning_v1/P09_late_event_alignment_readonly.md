# P09 late：相同源动作，不同真实支撑入口

2026-09-23。仅限已指定 CP218496 报告、CP220544 v7 选定证据及成功 N 对应窗口。没有重跑 Recording/Isaac、没有改原始日志、没有新增学习信用。成功 N 是历史 N_ref，不是与新控制版同构配对。最新正式要求的 rear-assists-OFF 验收优先于旧 DESIGN_sequence 中增加捕获辅助的建议。

## 1. 第一处任务时序差异

| 条件/事件 | 成功 N_ref | CP218496 DET | CP220544 v7 DET＋声明的 RR assist |
|---|---|---|---|
| RR 真实越沿 | episode tick6155 /51.2917s | 6662 /55.5167s | 9555 /79.625s |
| 当时 placed / 接触 | placed 同拍6155；之后 RR 可测支撑 | 未 placed，AIR，gap23.664mm | 未 placed，AIR，gap27.510mm，bearing0N |
| P09 late 首次实际名义组 | **6156 /51.3000s**，已有上一拍 placed | **6663 /55.5250s**，仍未落脚 | **9556 /79.6333s**，仍未落脚 |
| P10 RR knee 后续节点 | entry6160；6161−34.6、6168−29.3、6175−27.2° | 未启动，RR未placed | 未启动，RR未placed |
| 实际结束 | 73.8083s，完整成功 | 81.2333s，P09未完成 | 122.625s，P09未完成 |

N 的6154–6157与v7的9554–9557已用只读二分定位读取 native_tick_audit 的四拍小窗口确认。两者前一拍都仍为carry组，下一拍才切late。CP218496复用已封存232拍窗口报告。没有把 v7 dispatch physics tick（episode+179）和 episode tick 混为同一时钟。

相同 P09 late 源值变化（servo为canonical度，wheel为canonical forward-positive rad/s；顺序FL/FR/RL/RR）：

| 通道 | carry源目标 | late源目标 | 能确认的控制功能 |
|---|---:|---:|---|
| FL hip/knee | 38.6 /−13.4 | −18.5 /−31.4 | 大幅重构FL；不是只改RR的下降动作 |
| FR hip/knee | 0 /31.1 | 0 /31.1 | 保持已有源值；不等于最终/实际值保持 |
| RL hip/knee | 28.2 /0 | 15.4 /19.4 | RL也被改变；仅凭腿名不能断言其唯一功能是卸载 |
| RR hip/knee | −6.9 /−37.8 | −6.9 /−37.8 | RR源目标未给新下降路径 |
| 四轮 | .3 /.3 /.3 /.3 | −1.07 /0 /0 /0 | 前送组切FL反向脉冲，再由源显式stop结束 |

结论是“成功参考已具有RR支撑才执行同一late组，而两个C仅凭越沿便执行”，不是“C已经跳到了P10/P11”。这支持把依赖RR支撑的late重构与RR自身carry/capture区分。它本身不证明整组late每个通道都应永久停住，也不证明RR单一角度是充分解。

## 2. 实际控制权与目标/响应，不能把辅助轨迹当网络学习

CP218496关键RR pair（hip/knee，度）；mapped N与当拍policy REQUEST分别取原日志，不从两个独立nominal重算值做差：

| tick | mapped N | policy REQUEST | geometry/其他owner | final | actual | RR gap mm |
|---:|---|---|---|---|---|---:|
|6656|−9.400/−39.050|+16.545/−16.601|原geometry +10/+10仍在|17.145/−45.651|16.822/−45.457|22.516|
|6664|−9.400/−39.050|+16.557/−16.621|6658 geometry退出；final经slew|8.407/−54.421|14.459/−47.838|25.112|
|6680|−9.400/−41.550|+16.572/−16.462|knee有效残差−16.450，最终硬边界|7.172/−58|8.174/−54.981|57.101|
|9744|−6.022/−37.347|+16.641/−15.954|无新RR下降owner|10.619/−53.301|10.340/−53.085|69.700|

6657进入原TOP XY容差，6658原+10/+10建议撤出，6662越沿，6663late，属于不同事件。目标变化与全身重构接近，但不能把gap抬高全部归给其中一项。后期actual跟踪final，未看到RR actuator输出丢失的证据。正hip残差确实抵消了负nominal，负knee残差与裁剪也确实存在。

v7：RR辅助在 **9516** 已取得6/7最终owner；9555越沿时仍HOLD/BLOCKED且真实AIR。该时刻原policy投影请求为RR `[+16.347,−16.274]°`，但RR最终target是辅助保存的 `[+18.214,−45.361]°`，actual `[+17.901,−45.190]°`。源RR仍 `[−6.9,−37.8]°`。当前小证据集没有9555独立mapped-N servo值，保留缺失，不能用final减REQUEST补造它。

v7在9773（81.4417s，源stop已提交）RR gap62.281mm仍AIR；后续assist把final变到末拍 `[−1.786,−12.361]°`，actual `[−2.114,−12.293]°`，gap−0.0109mm，但仍AIR、bearing0、placed=false。耗用53°/约43s辅助预算，没有完成放置。几何gap接近0不等于传感器接触；原始normal/resultant为0，但独立切向摩擦记录有非零微量值，不能说所有接触量都恒为0。

四轮也分清source、策略与后投影：v7越沿9555时source `[.3,.3,.3,.3]`，REQUEST `[−.925586,+.073696,−.014549,−.091077]`，final `[−.625586,+.373696,+.285451,+.208923]`，actual `[−.741399,+.311649,+.339303,+.208007]`，wheel mask全1，尚未wheel投影。故FL反转在late脉冲之前已由policy抵消前送形成；不是其他三轮被mask。末拍source0、REQUEST FL−.938946，但postprojection final FL−.044949，属于声明的控制干预，不能称策略自行修正。

## 3. FR/FL接收侧事实与尚待验证的方向

成功N在6160/6168/6176/6248，FR knee final约+29.994°、actual约+30°；P11新源owner是FR hip序列，并非新FR knee节点。相应FL knee actual从−13.05→−17.32→−23.56→−31.52°，是在**向更负**变化；不能将这段历史说成当前C的FL从−58°“正向恢复”的实验证明。N在6160已有RR障碍接触反力约9.688N，6176约11.136N；当前C AIR不具备同样反作用条件。原记录O表示障碍接触，不单凭O字符替代严格TOP分类。

下一次有限真实诊断应检验，而非把+30/−20硬编码为成功标签：

1. 在RR真实捕获/当前支撑形成后，FR接收构型变化是否使mass-weighted CoM沿**该次转移起点固定FR方向**的投影增加；同时单列FR脚位移，不能用脚自己靠近CoM冒充转移。
2. FL是否实际离开负边界并获得双向可用行程，同时RR/FL传感器接触保持；若仅nominal角改变、final被policy/裁剪/owner抵消或AIR未承载，应明确归到相应层，而非宣称已形成支撑架构。

## 4. 精确来源

- N：`runs/ppo_task_conditioned_hip_wheel_v1/video_eval/prior_B/20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source`。事件复用 `CP218496_RR_capture_source_readonly.json:accepted_N_ref`；实际构型/力复用 `N_P10_P12_actual_vs_v5_RR_precontact.md`，late首次额外读取native6154–6157。
- CP218496：`outputs/ppo_p05_hip_only_continuation_v1/CP218496_RR_capture_source_readonly.{md,json}`，其source末段为 `20260922T1818481605380Z_g6ac7b553d792_d3e1bceb3e8e462ca7eddd06eb20a21f/source`。
- CP220544 v7：`outputs/ppo_rr_capture_then_rl_transfer_v1/video_review/CP220544_ancestor_signed_contact_v7_review/CP220544_v7_selected_fourwheel_evidence.json`、`v7_signed_gap_wheel_terminal_readonly.md`；额外读取对应source native9554–9557。

补充当前P05结果边界：CP222720录像55.4s因viewport callback_count=0停止，是CAPTURE_ABORT/P05未完成，不是自然task terminal或局部超时。P05 v3最小接续修复只把已完成源后的反馈起始年龄下界30改0，保留绝对end40与所有source/接触/净空门；该修复139项CPU定向测试通过，尚不构成真实P05或完整越障成功。

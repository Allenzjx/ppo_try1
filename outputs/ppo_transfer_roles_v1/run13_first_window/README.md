# Run13：首次可见窗口（13 决策，非最终结果）

固定范围 **g135425–135437、episode tick5953–6056，共104 ticks**；源阶段 P07=1、P08=1、P09=11。只保留实际可见13条，未为凑24条等待后续前缀。13条均非terminal、reason=null、bootstrap允许、physical valid=true；**本窗口没有可见终止行，不能从下一前缀推定本回合原因**。这些是已发出的policy样本，不宣称已优化。

## 接管与物理状态

首次 frozen-FSM prefix 为744决策/5952 ticks，全部排除policy信用；P07 offset0在tick5952/49.6s接管。原source控制tick5951→原生ACK6131，只有一次articulation写入；首policy tick5953→ACK6132，之后104 ticks连续且native−episode恒179，无时钟归零。

接管RR为真实GROUND，AIR=false，support=true，load=.247819573，front−205.029mm、clear−50.920mm；硬Q/C/P全部false。FL为真实TOP/support=true，load=.199119297，clear+.256mm，教师历史Q2461/C3115/P3583；不能把教师事件算新增policy成功。

## FL命令与跨阶段连续性

单位：servo为canonical角度deg；m为原生mapper转换后的逻辑角度，包含mapper补偿，区别于另加的controller bias（本表均0）。本窗口FL requested=effective，未触发headroom裁剪。

| global | source→end | tick | FL nominal | requested/effective | mapper m | final canonical |
|---|---|---:|---:|---:|---:|---:|
| 135425 | P07→P08 | 5960 | 26.3 | 4.000 | 26.3 | 30.300 |
| 135426 | P08→P09 | 5968 | 29.8 | 0.947 | 29.8 | 30.747 |
| 135427 | P09→P09 | 5976 | 33.3 | -2.553 | 33.3 | 30.747 |
| 135428 | P09→P09 | 5984 | 37.3 | -6.350 | 37.3 | 30.950 |
| 135429 | P09→P09 | 5992 | 41.3 | -6.172 | 41.3 | 35.128 |
| 135430 | P09→P09 | 6000 | 45.3 | -2.172 | 45.3 | 43.128 |
| 135431 | P09→P09 | 6008 | 49.2 | 1.828 | 49.2 | 51.028 |
| 135432 | P09→P09 | 6016 | 49.2 | 5.828 | 51.7 | 57.528 |
| 135433 | P09→P09 | 6024 | 49.2 | 9.828 | 54.2 | 64.028 |
| 135434 | P09→P09 | 6032 | 49.2 | 5.828 | 54.2 | 60.028 |
| 135435 | P09→P09 | 6040 | 49.2 | 7.600 | 51.7 | 59.300 |
| 135436 | P09→P09 | 6048 | 49.2 | 4.536 | 49.2 | 53.736 |
| 135437 | P09→P09 | 6056 | 49.2 | 0.536 | 46.7 | 47.236 |

接管nominal22.8°→首决策末源nominal26.3°，之后相邻8-tick末值增量为3.5/3.5/4/4/4/3.9/0…°；**与60°/s（.5°/tick）相符**。但日志只有决策末源nominal（tick末−1）及每tick验证摘要，不能独立证实全部中间nominal逐值≤.5°。原生最终servo上次target→本次target的13个完整末样本，全部通道最大步长1.25°，不是把nominal限速当物理关节限速。

P07→P08在tick5961、P08→P09在5969确有hold：原requested残差完整继承，无禁用/phasecap丢弃；applied action jump最大约1.78e−15（浮点误差量级），未无条件清零。104/104物理ticks原生效果验证通过、104/104有实际policy效果，其中两次hold不计本阶段新raw效果，故own-phase效果102 ticks。13份完整末audit均mapping/setter verified；四类statewrite计数全0，无日志越界/硬安全改写证据。compact字段不足以独立重建所有tick全部float32目标。

FL raw范围−.20110…+.48603，requested−6.34979…+9.82760°，m26.3…54.2°，final30.3…64.02760°。g135427用negative residual使m33.3°对应final30.74694°；其后正残差也真实放大命令。轮通道同样非零：FL轮nominal+.3，实际canonical净命令−.54403…+.32990rad/s；原生左轴float32符号在独立audit里核验，不能把canonical double命令当native读回。

## 实际接触，不能替代任务结果

首policy末tick5960起的13个末采样FL均AIR/support=false/load0，clear+6.974…+137.764mm；最后+102.004mm。接管时的FL TOP载荷不能沿用到这些采样。实际FL q（末次dispatch前真实传感值，经standing换算）约23.696→53.865°，不是目标瞬时跟踪证明。

RR末采样GROUND10次、AIR3次（5984/6016/6056），clear仍−50.920…−48.318mm；全窗口RR/RL硬Q/C/P均false，无新增后腿initial/qualified/cross/placed事件。短暂AIR不等于有效抬升资格，更不等于任务成功。

来源：该run第一段prefix与固定前13 policy rows；未读取后续policy轨迹、optimizer、checkpoint或已缓冲外终止。JSON receipt保留精确字段及范围。无生产修改、Python/Torch/Isaac或测试执行。


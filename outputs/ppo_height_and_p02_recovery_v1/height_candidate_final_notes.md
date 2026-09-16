# 已封存高度候选：供最终报告引用的限定结论

仅复用 `height_event_comparison_FL4_RL3.payload.json`、`B_HEIGHT_FL4.aggregate.json`、`B_HEIGHT_RL3.aggregate.json`；未读取当前运行的 zero，也未扫描原始大日志。以下显示数值四舍五入至 6 位；原 JSON 保留完整精度。角度均为相对 standing 的 canonical 度；高度、距离表使用 mm。

## Source 缩减、mapper 输出与实际关节必须分开

- **FL−4**：P07 source owner 在 observation 5160 把目标 **37.6→33.6°**，下一实际 dispatch/post tick 5161 的 nominal 确为33.6°；但该拍 mapper/final 仅 **23.65°**，actual **22.865556°**。RL−3 候选同一拍未缩减 FL，source37.6°，mapper/final同为23.65°、actual也同为22.865556°。这直接证明 source−4° 不等于首拍 mapped/actual−4°。
- **RL−3**：P08 observation5360 的 owner **14.3→11.3°**，post5361 的 nominal确为11.3°；该拍 mapper/final **8.15°**、actual **8.502398°**。5360 物理行仍是上一拍 RL source6.9°、actual8.435797°，不能把未来 source 目标贴到上一物理状态。现有 B2 摘要没有 post5361 的独立 mapped 值；不补造该拍“实际减3°”对照。
- 到各自首次 RR Q 事件，FL−4 的 FL source/mapped/final 为 **34.6/34.6/34.6°**，actual34.538807°；RL−3 的 RL 三者为 **28.2/28.2/28.2°**，actual27.829945°。B2 相应 source/final/actual 为 FL **38.6/38.6/38.554664°**、RL **31.2/31.2/30.073025°**（B2 独立 mapped 字段未留存）。所以事件对齐 actual 差分别 **−4.015858°**、**−2.243080°**，不是 RL 必然少3°。
- owner 注释可被后来的 source owner 接管。RL−3 的旧 P08 注释在6720仍写28.2°，但下一 dispatch6721的 RL nominal实际为 **−6.9°**；FL−4终态旧 owner 注释34.6°，实际当前source已为−18.5°。不能把历史注释当持续实际减幅。

## 首次 RR Q：事件对齐，不是同一时刻或单因素因果试验

| 候选 / Q tick | body collider 最低z | base-origin z | RR upper-link原点z | RR轮底相对台面gap | RR knee actual / 距−60°余量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B2 / 5425 | 90.402921 | 55.094443 | 195.526183 | −41.823266 | 0.145153 / 60.145153° |
| FL−4 / 5442 | 87.402916 | 54.158680 | 187.133819 | −41.413802 | −0.066878 / 59.933122° |
| RL−3 / 5434 | 95.461485 | 60.048178 | 197.310746 | −41.990862 | 0.133087 / 60.133087° |

相对B2，在这些事件处 FL−4 的 body/base/upper-link z 分别为 **−3.000006/−0.935763/−8.392364 mm**，RL−3 为 **+5.058564/+4.953735/+1.784563 mm**。这只是观测差：B2几何控制版本不同；FL−4（3fc0a8c）与RL−3（a9c52261）也不是同HEAD。不能称为仅改变某髋角所造成的高度因果效应。

**这些 Q tick 的真实 USD mount 均未知。** FL−4只有5440/5448、RL−3只有5432/5440的邻近独立 height samples，未插值；上表 upper-link原点是另一项120Hz实测量，不能冒充USD mount。A 的旧 ACTIVE_LIFT/Q 语义不同，不与新Q硬对齐；其现有窗口也无真实USD mount记录。

## 各自终态：位置与任务状态不同，不能据较高终态宣称更稳定

| 候选 / terminal tick | 结果 | body最低z / base z | 真实USD RR mount z | RR轮link-origin z / 独立最低碰撞点z | RR front / 台面gap | RR knee actual / 下限余量 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| B2 / 6374 | HARD_JOINT_LIMIT | 108.694340 / 102.383897 | 未知 | 未知 / 未知 | −19.751605 / −5.733861 | −60.000011 / −0.000011° |
| FL−4 / 8844 | P09 INCOMPLETE_CONTROLLER_BLOCKED | 129.728024 / 108.257242 | 189.875711 | 48.839796 / −1.133383 | −64.778146 / −51.133383 | −28.893031 / 31.106969° |
| RL−3 / 8857 | P13 INCOMPLETE_CONTROLLER_BLOCKED | 190.853088 / 149.339080 | 255.652605 | 99.472217 / 49.237898 | +266.788307 / −0.762102 | −6.030535 / 53.969465° |

FL−4终态RR为真实GROUND、承载13.429890N，未cross/place；RL−3终态RR为真实TOP、承载11.368696N，RR历史Q/C/P已有记录，且当前initial lift与qualified lift有效。RL−3的RR C/P事件均在6155，front **+0.097934 mm**、gap **−0.268861 mm**、knee actual **−39.446992°**、下限余量 **20.553008°**；该事件USD mount仍未知（邻近样本6152/6160）。A原参考cross7109的gap29.976310mm、place7579的gap−1.783198mm是另一条事件轨迹，不能强求同一入口姿态。

FL−4/RL−3终态独立碰撞点与USD mount均恰在各自terminal tick采样；body独立最低z与已记录值一致。轮link-origin不是最低碰撞点，也不自动等于几何轮心；body世界最低z不是机身到障碍物的最小分离距离。上述 knee 余量仅为该tick实际q到−60°下限，不是全episode最小余量，也不是允许PPO residual区间。

两候选全程12维 raw/projected/direct native residual的记录最大值均为0，故这里只是nominal诊断，**不是PPO学习收益**。两次task_success均false；较高的RL−3终态来自已到平台区域的不同任务状态，不能与FL−4仍在地面未越沿的终态组成稳定性优越证明。媒体/接受校验异常另列，不替代真实任务终止原因。

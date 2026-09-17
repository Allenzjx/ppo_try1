# CP177152：成功完整运行中的 P02 四轮证据

结论：本次不是只有 RR 有指令，也不是只有 RR 实测转动。P02 正向 N 窗口内，没有旧策略“FL反向、FR/RL近乎抵消、RR加强”的长期模式；四轮保留协作，FR悬空时旋转不等于地面牵引。无需再为此改变 mask 或强制等速。

来源为已封存完整正式运行 `20260916T0550249947408Z_g00050a2b1452_086805789a6e491392ef4656c7708c2c`，保存/重载 CP177152 后自然 P01 开始，最终6116ticks / 50.966667秒、P13真实 SUCCESS。这里只读前缀日志至P02末tick1360，完整结果另引用既有正式receipt，未由P02局部结果推导成功。旧CP174592、CP176768表与成功zero全部保留；未改正在运行的512训练。

## Tick400 / 3.333333秒

四轮 full source/N 与实际 mapper 后 N 均为+0.3 rad/s，controller bias均0，实际 residual mask均1；mask作用于**附加 residual**，不是nominal或actuator写入选择。按FL/FR/RL/RR，动作索引/live ankle IDs为8/9/10/11，canonical→native符号为−/+/−/+。

| 轮 | policy raw | 限速/投影后有效修正 rad/s | 最终canonical | 最终native | 实测native qd | 实测canonical qd | 接触 / 地面法向力N |
|---|---:|---:|---:|---:|---:|---:|---|
| FL | .023455 | .014070 | .314070 | −.314070 | −.240914 | .240914 | GROUND /13.68145 |
| FR | .081661 | .048888 | .348888 | .348888 | .349504 | .349504 | AIR /0 |
| RL | .134468 | .080198 | .380198 | −.380198 | −.403698 | .403698 | GROUND /2.56478 |
| RR | −.016863 | −.010117 | .289883 | .289883 | .290567 | .290567 | GROUND /12.34999 |

这是同tick原生 `robot.data.joint_vel` 与物理日志，不是从轮端位移推测电机转动。此拍RR knee target/actual为−.194519° / −.635943°；FR轮底高于障碍顶面 .118485m；base-origin/CoM世界高度 .076035 / .168043m。

保留成功zero同tick四轮target均+.3、实测 `[.24424,.29813,.34175,.31376]`，RR knee `0/−.607°`，FRgap .100780m。旧CP174592同tick最终target `[−.20322,.02121,.03758,.79067]`。本次与旧策略差异发生在**policy残差/实际合成target**，不是其他三轮被mask。不同闭环轨迹的等时间比较不是同状态反事实，不能独立归因于某一次更新。

## 整个实际 P02 正向 N 窗口

P02为tick17–1360；tick17–40源/N四轮均0，tick41–1360四轮均+.3，1320个真实物理tick，采样覆盖11.0秒（时间戳 .341667–11.333333秒）。不是要求全阶段每帧四轮非零。

| 轮 | 最终target范围 rad/s | 均值 | 实测qd范围 rad/s | 均值 | target反向 / abs<.05 / 低于N一半 |
|---|---|---:|---|---:|---|
| FL | [.304370,.317295] | .314101 | [−.073750,.489759] | .249180 | 0 / 0 / 0 tick |
| FR | [.320563,.348888] | .346469 | [.320593,.352647] | .346961 | 0 / 0 / 0 tick |
| RL | [.333514,.380307] | .377189 | [−1.286249,.715259] | .423806 | 0 / 0 / 0 tick |
| RR | [.289521,.296387] | .290260 | [.030025,.432303] | .305151 | 0 / 0 / 0 tick |

前三轮 residual均小正、RR小负，整1320tick没有“前三轮target低于各自N一半且RR加强”的区间（0秒）；也没有任何单轮target低于N一半。阈值仅是可复核的描述，不是新成功门槛。旧CP174592在772tick正N窗口中FL反向756tick、FR近零644tick、RL近零620tick、RR高于N772tick，不能与本次混称。

最终target非零仍不代表完美物理跟踪：FL实测abs<.05发生于tick235/249/562/571/580，各仅1tick，共.041667秒；RR为tick382单拍，共.008333秒，FR/RL无近零拍。RL仍有瞬时反向峰−1.286249，不能隐去，也不能仅凭此归因为mask。没有新增接触/净牵引因果试验。

1360个前缀tick的实际all12 mask均1、source N与mapped wheel N差0、wheel controller bias0；同拍 `N + 有效residual` 与native最终target符号还原最大误差1.48891e−8。170个真实原生qd读回点与canonical符号还原差0。派发及actual mapping均核验通过，实际目标取自既有全量wheel setter / `write_data_to_sim`之后。日志没有唯一逐tick源动作owner字段，因此不凭changed_channels虚构owner；完整N持续值、最终写入读回与物理转动分别保留。

FR真实主动抬升tick22、越沿1374、放置1397，tick1400再次有TOP承载2.79427N。四轮标注视频复用已交付 `videos/ppo_task_recovery_cp177152_full_success_wheel_qd.mp4`，前段视频为 `videos/ppo_task_recovery_cp177152_p01_to_fr_capture.mp4`；均不是wheel残差屏蔽诊断。

精确数据、绑定、逐区间统计及原zero/旧策略表见 `four_wheel_P02_checkpoint177152.json`。复用既有 `analyze_176768_p02_wheels.py`，仅输出侧增加run/media/checkpoint参数与连续物理tick统计；没有模型forward、GAE重算、再次物理运行或生产变更。

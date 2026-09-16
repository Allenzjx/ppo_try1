# Wheels4 诊断：P02 有界控制链核验

此为已结束的诊断，不是正式全12 PPO成功：240 decisions / 1920 ticks / 16s，因 DIAGNOSTIC_BOUNDED_WINDOW 停止，零更新、模型未变。真实 FR 抬升27、越沿1672、放置1686，后续到达P05。原C168192在887发生RR硬限中止。本报告仅对原C/当前零残差≤887共同窗口与诊断完整P02进行分析；没有新增仿真或生产修改。

1920个native tick的wheel raw、projected、同prestate direct target effect均为0；生产mask仍是12个1，其余8通道真实目标效应持续存在。1880tick仍有非零nominal wheel，因此不是wheel全停。240行实际raw→下一HISTORY及372观察片段逐行严格相等；final多一次边界检查，共241。目标canonical/native重建最大误差1.705e−6，未发现重复应用或单位错配。原C与诊断前887tick N完全一致，初始actual q/qd/base/CoM/contact整个对象严格相等；其他servo raw自tick9开始随改变的闭环/HISTORY而不同，不能声称这是“其他动作固定”的实验。

共同P02窗口，原C四轮目标均值依FL/FR/RL/RR为[-.08857,.05905,.12458,.76381]rad/s。诊断保持source同向目标（启动后全+.3）。原C直接wheel残差的四轮平均量跨tick均值−.07702rad/s，而逐tick差分RMS均值.32631rad/s；是明显差速模式，不只是整体前进速度改变。但该干预同时移除了平均与差分部分，且改变后续HISTORY/servo闭环，尚未分离二者因果。

诊断完整P02共1640ticks/205decisions：RR knee实际最低−5.012°、target最低−1.887°、tracking最低−3.150°；原C共同窗口实际最低−60.017°、target最低−5.770°、tracking最低−54.583°。tick400原C RR source/mapped0、projected/final−5.712°、actual−23.910°，诊断final−1.299°/actual−1.940°。两者同prestate一步direct effect约−1.25°是最终slew下的反事实定义，不等于总残差相等。

这不是“两边都失去所有支撑”：tick400两者FL/RL/RR均真实GROUND。原C三腿normal force为11.666/3.334/13.677N，对应wheel-body vx为−.000845/.005318/.024244m/s；诊断12.647/3.301/12.832N，vx为.010231/.013121/.014583m/s。可据实说差速指令、实际腿轮运动和负载路径分化，不可声称已分解出唯一受力因果或轮胎滑移机制。

本次实际getter：servo K600/D60、effort2.7Nm、speed约5rad/s；wheel K0/D20、effort1Nm、speed2.094395rad/s；未发现配置错配。原C没有同轮getter或effort缓存，必须标unknown。新诊断RR隐式PD估计56/205端点被裁剪，其中24..304有36个连续采样端点；即便如此仍避免旧C的大幅失控。估计来自前一dispatch的最近compute，不是postpose重算，也不是独立测得PhysX驱动力矩；Isaac的ImplicitActuator本身只估算并返回原控制目标，仿真内部处理隐式PD。

结论：没有必须先修复的执行链故障证据；支持按既定授权继续全12、原幅度/rho/reward的P01恢复训练与后腿维护，并启用只读GAE审计。不要永久mask、强制四轮同速或据此改物理gain。随后以重新加载后的正式全12结果验收。平均分量/差分分量分离干预若以后必要可提高诊断辨别力，但本轮未运行，也不是optimizer启动门槛。

逐tick验证统计、选定端点与来源见同目录 p02_wheels4_completed_comparison.json。native几何字段缺失未被冒充zero：P01–P05不启用后腿geometry的依据是当前ACTIVE代码范围P09/P12，而非缺日志推断。


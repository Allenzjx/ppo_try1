# RR 悬空与真实捕获：有限几何对照

只读已封存数据；未启动 Isaac、加载模型、拟合或创建辅助训练样本。当前 CP214400 运行未被修改。

## 三个来源，不混称

| 实测量 | CP213376_AUXMEAN3 det，60–81.5 s | block03 PPO 捕获前，t8720 | 旧 accepted N，RR 捕获 t6155 |
|---|---:|---:|---:|
| RR 轮底距障碍顶面，mm | 均值 50.491；范围 49.866–51.827 | +0.360 | −0.269 |
| RR hip 安装点世界高，mm | 均值 224.005 | null：未记录世界位姿 | 197.794 |
| 轮底相对 RR 安装点世界竖直量，mm | −123.514 | null | −148.062 |
| 真实 body collider 最低点世界高，mm | 117.742 | 100.072 | 89.168 |
| RR 实际 hip / knee，deg，canonical | +9.275 / −48.657 | +8.774 / −50.664 | −8.228 / −39.447 |
| RR 轮心相对 base，body-frame z，mm | +21.654 | +25.280 | −6.337 |
| 当前接触 | FL/FR TOP；RL ground；RR AIR | FL/FR TOP；RL ground+obstacle ambiguous；RR AIR | FL/FR/RR obstacle 接触；RL ground；RR 当拍已获 placement |

失败列几何为 2,581 个 120 Hz 真值样本均值，不是一个合成姿态；接触由其中 324 个同步 decision-end evaluator 核对，FL/FR/RL 均为当前 verified support，RR 全部 AIR/无承载。body collider 高度不是 base 原点高度，更不是安装点高度。

N 是指定旧版本几何参考，不是 PPO 示范，也不是本轮同版本配对控制。block03 是混合训练策略的单次真实 RR 捕获，不代表完整越障成功。

## 可精确分解的部分：旧 N 捕获 → 当前失败窗口

每个当前实测姿态分别与 N 的 t6155 比较，再求均值。RR 额外悬空 **+50.760 mm** 分解为：

- RR 安装点世界高：**+26.212 mm**，其中 base 原点竖直平移 +22.158 mm、机身姿态移动安装点 +4.053 mm。
- 轮心相对安装点的实测局部构型变化，在 N 姿态投影：**+26.489 mm**。
- 机身姿态对该相对轮心向量的旋转项：**−1.424 mm**。
- 轮心到真实 collider 最低点的包络差：**−0.517 mm**；障碍顶面差 0。

逐状态恒等式闭合误差小于 7×10⁻¹⁴ mm。安装点来自两份 startup 中一致的真实 USD body0/localPos0，加各自已记录 base_link 位姿；与两份 height diagnostics 独立记录核对误差小于 10⁻⁹ mm。局部项使用轮心，轮轴转动/姿态引起的轮底包络单列，不把轮底最低点当刚性轮端固定点。

这是一种明确顺序的几何分解，不是动力学因果分摊。相对旧 N，**不是只有全身抬得过高，也不是只有 RR 局部构型不足，两项量级相当**；不能据此指定历史关节角为目标。

## 更相关的 PPO 捕获证据：不能只责 RR 局部关节

block03 t8488→8720（1.9333 s）中，RR gap **22.369→0.360 mm**，真实 body collider 最低点 **135.324→100.072 mm**；RR 实际 hip **6.205→8.774°**、knee **−48.627→−50.664°**，轮心 body-frame z 却 **+19.780→+25.280 mm**。同时 RL hip **6.014→18.070°**，FR knee **+0.460→−11.147°**。这是全身构型与支撑都在变化的捕获，不是孤立 RR 伸腿记录；collider 下降量不能冒称 hip 安装点下降量。

当前失败末端 t9780 与真实捕获前 t8720 的 RR 轮心在 body frame 内仅相距 **5.866 mm**，而 gap 相差 **50.005 mm**；RR 两关节也相近。仅这一局部轮心构型变化在任意固定姿态下的竖直贡献绝对值不超过 5.866 mm，不能单独解释约 50 mm 差距。全身世界位姿/机身姿态与轮底包络必然还参与，但成功训练日志未存绝对 base 位姿、quaternion、RR mount 和轮底包络，故**精确世界分解为 null**，不能把差值全分配给高度或某一关节。未计算缺失 RR knee 枢轴/轴向的瞬时 Jacobian。

支撑也不等同：捕获前 30 个样本 FL 只有 4 个当前 verified TOP，FR 为 30/30；RL 29/30 为 obstacle ambiguous 且 bearing 未验证，不能用其 ground contact 位或历史 placement 伪称可靠承载。t8736–8808 的 10 个样本 RR 才全部当前 TOP/verified support（gap −0.950 至 −1.059 mm），FL 10/10、FR 9/10、RL 5/10 verified support。RR t8726 的 placement 历史并不使 t8728 的 AIR 自动成为持续承载；已知 t8816 又 AIR，本报告不外推保持。

当前失败窗口内八个实际关节基本冻结；RR hip 全幅 0.067°、knee 全幅 0.071°，21.5 s 仅净缩小 gap 0.860 mm。这是已经停滞的不同全身状态，不是已经完成放置。证据支持关注全身放置几何和动态支撑接续；不支持只加 RR 膝幅值、固定历史姿态，或将旧样本某个带噪声 RR 输出指定为唯一修复。

## 可复查数据

精简 JSON：`RR_capture_height_geometry_readonly.json`（约 58 KB，无累计 history）；脚本：`rr_height_geometry_readonly.py`。JSON 保存指定窗口原始完整行的 SHA256、必要端点、各实测统计和分解项，未把文件读取范围或 decision-end 状态冒充整个 run。训练 residual/target 是截至该 end_tick 的动作；同 tick 状态是下一次动作的 input，不把当拍 residual 与下一拍状态错位为同拍因果。

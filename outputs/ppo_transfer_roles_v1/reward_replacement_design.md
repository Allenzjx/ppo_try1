# 五类稠密奖励：替换设计（只读）

完整读附件；依据4a7bdbe生产（等同e846），不评价WIP。旧跌倒报告未完成，已暂停。反例不证明失败因果。

索引：S=`semantic_supervisor.py`；R=`semantic_reward.py:SemanticRewardCalculator.evaluate`；配置=`ppo_semantic_v3/reward_config.yaml`。

| 现有family/信号 | 冲突或已有保护 | 最小处理与合成反例 |
|---|---|---|
| task_progress：S.physical_potential的`.1*workspace+.1*unload` | 仅前缘区间、另两腿支撑和目标低载，不代表接收空间/有效转移 | 原准备份额替换为角色空间与实测转移，不叠加。掉载但CoM反向不能算准备；只移接收轮靠近固定CoM不能算CoM平移。 |
| body_stability | attitude乘`1−.8*f`，非全阶段等权；f仅看低载，角速/角加速度仍全权重 | 替换S.observe_and_update的physical_transfer_fraction来源：角色转移准备允许倾斜，capture/settle渐恢复；不能等AIR才减罚。合法腾空与失稳掉载不可只凭同一低载同等减罚。 |
| contact_motion_quality | 无一般receiver AIR罚；反弹需近期触地、向上反转、全身command excursion≤.1；无大力奖 | 保留保守过滤，不当因果证明。无接触下压无冲量；主动command变化不因向上速度自动罚。冲击按触地下落速度超额计，载荷仅缩放；低速接触不因载荷本身扣分。不整项删除。 |
| control_smoothness | applied_only已去nominal/residual重复收费；实际一阶/二阶差分互补 | 保留：匀速坡道一阶非零、二阶零；重分nominal/residual但actual不变，成本应不变。不新增平滑项。 |
| control_regularization | 权重0且disabled | 保持关闭；不奖励非零残差。 |

## 跨项边界

1. S._current_capture_retention保留`.8`历史、`.2`当前区域；平台上方AIR可满额，**不是载荷信用**，support仍false。合法receiver AIR不一律罚，失效交接不伪称准备；角色层暴露当前退化，不强制receiver先承载。
2. S._current_lift_credit及TaskEvaluator已用全身actuation+真实净空；RR自身角变化小仍可得初始进展，不能直接给Q/C/P。保留此保护。
3. S._current_capture_progress已有Q+C后接近梯度，unload退休为1，加载不丢抬升/越沿信用。C118016信号非平坦；不能靠加奖掩盖P05/P06动作依赖。
4. R用`5*(.9985*Phi_next−Phi_before)`；真实终态Phi=0/失败−40，普通phase无奖，按dt计质量。C119040相邻Phi差0、handoff非done。替换后保持每腿预算/参考/历史连续，仅换标签不得重置势、重复领奖。

最小替换仅**准备代理、低载减罚系数**。保留两点动态支撑、真实capture与安全失败；不拒绝替代构型，不动gamma/lambda/rho，不加CoM靠脚奖/大力奖/第六family。仅设计，未实施或实验。

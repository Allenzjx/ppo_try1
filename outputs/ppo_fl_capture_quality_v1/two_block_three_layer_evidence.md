# 两个封存块的三层证据（1408 个已优化样本）

只读 block01 `041224…541c` 的 512 个样本（updates1360–1363）和 block02 `042155…693f` 的 896 个样本（1364–1370）。后者在 verified update boundary 停止；没有把未优化尾部或 N prefix 计入 PPO。未重新 forward、未重新仿真、未修改 reward/参数。完整逐通道统计见同名 JSON；下表采用已保存的实际 collection 请求，不是固定最终 checkpoint 的新评估。

**策略层**展示 FL hip 的带符号 base μ→conditional μ，以及实际 effective σ 的均值/峰值（均为 raw latent 单位）；全部 12 通道统计另存 JSON。**执行层**为 8 servo/4 wheel 的平均绝对值，单位是 deg / rad/s：`tanh×当前phase cap请求 → 已滤波请求 → 同pretick实际native target差`。最后一项来自该决策末次真实 dispatch 的独立审计，而非重新计算 N 的差值。

| 请求阶段 / n | FL hip：base μ→conditional μ；σ均/峰 | 全12 raw绝对峰；至少1通道 abs(tanh)≥.95 的决策 | 执行层：servo° / wheel rad/s | 机身角速均 / P95 / 峰 rad/s | 实际接触与历史（决策末端） |
| --- | --- | --- | --- | --- | --- |
| P01 / 4 | −.0715→+.0346；.0550/.0638 | .1175；0/4 | .745/.0272 → .745/.0272 → .629/.0272 | .120/.190/.198 | FL/RR均地面承载；样本极少 |
| P02 / 362 | −.0852→−.0567；.0639/.1282 | .6255；0/362 | 2.252/.0490 → 2.248/.0490 → .993/.0490 | .096/.209/.970 | FR AIR 362；FL/RR承载362 |
| P05 / 693 | +.0728→+.0958；.1319/1.0442 | 2.7066；5/693 | 3.654/.1398 → 3.627/.1370 → 1.123/.1370 | .117/.261/.446 | FL AIR657，TOP15，placed历史4 |
| P06 / 93 | +.0811→−.2260；.2115/2.1812 | 2.9367；5/93 | 10.440/.1814 → 9.082/.1540 → 1.221/.1540 | .192/.585/.810 | FL placed历史93，但TOP62、AIR30 |
| P09 / 225 | +.1407→+.2504；.1829/3.6710 | 13.6957；19/225 | 8.068/.2103 → 7.626/.1860 → 1.208/.1860 | .289/.592/1.411 | FL placed历史225，但TOP31、AIR194；RR AIR72、承载153、cross/place均0 |

表中包含全部目标阶段；其他阶段 P03=9、P04=16、P07=3、P08=3，合计仍为1408。接触计数不是独立 episode 成功率；P05 的地面承载、TOP 和历史 placed 必须分开。所有目标阶段的同pretick审计均 verified；所查决策末次 dispatch 无 servo headroom clipping。**native Δ只有约1°不代表残差只有约1°或执行无效**：比较的两个分支共用已经含历史作用的真实 previous-final 和同一 slew 限制，它测的是当前请求的同拍边际目标作用，不是累积姿态影响。它也不是实际关节位移或负载反应。

局部宽尾的明确实例：block01 首次 FL 捕获前 tick2152，FL hip σ=.666886、base μ=+.12610、conditional μ=−.77106、raw=−2.70659；下一拍 σ=1.04420。P06 的 FL wheel σ最高2.81260。block02 的 P09 tick3680/global179638，FL wheel σ=5.14973、base μ=.51773、conditional μ=4.94292、raw=13.69568。实际 rho=.9、temperature=.25 均已保存核对；固定温度不意味着 learned/effective σ 对所有状态同样小。饱和是局部尾部现象，不应把上述极值当成全部样本常态。

四次已完成 episode 都取得真实 FL placed（ticks2157、2917、2768、2828）；后三次取得 RR active_lift（3166、3037、2956），但没有 RR cross/place，终止分别为 P06 BODY_COLLISION、P09 FALL、P09 FALL、P09 BODY_COLLISION。block01 第二次和 block02 第四次的当前样本已经完成优化，但 episode 尚未终止，不是成功 episode。

这支持接下来用**同一固定 checkpoint**分别录制 deterministic 与 stochastic 视频：展示当拍 base/conditional/σ、raw与投影、实际接触和动作；不把“base μ较小”当作确定性闭环一定稳定。这里混合了多个更新中的旧 collection 网络、不同自然/真实prefix入口和不同实际状态，没有构成严格配对因果试验，也不支持宣称某个新温度/参数更好。保留原训练与安全标准，先看真实轨迹。

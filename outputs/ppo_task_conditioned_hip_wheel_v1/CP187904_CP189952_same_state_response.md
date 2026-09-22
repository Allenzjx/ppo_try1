# 最终 CP189952：同一批 37 态的只读响应对照

**最终未推翻此前方向判断：network mean 确实改变，但不能称为已纠正全身动作；sigma 扩大更明显。** 本次仍只载入 CP187904 与最终 CP189952 两个模型，使用上一报告完全相同的 37 个真实保存的 372 维状态（逐态 SHA 全部一致），不重构假 HISTORY。没有 CUDA、Isaac、采样、优化器调用或 pointer 变更。旧 CP189440 JSON／MD 保留，旧 JSON SHA 仍为 `4c36842def154587fb2861115baf88af9229ddfec998fe8530986d1922734b5b`。

- 源 CP187904：1433 updates / 28660 optimizer steps，SHA `e3a405312514c8c99dd55195a7f8648cff5c44af48e33c4a7e7429a2a2d2eb4c`。
- 最终 CP189952：1449 updates / 28980 optimizer steps，SHA `cf89d3c206874a5d239b21e622752f7f294f3219b4b587853c43840a1de570d2`。相对源新增 2048 decisions / 16 updates / 320 steps。
- 复算核对通过：source network mean 最大误差 5.96e−8，conditional mean／sigma／真实 HISTORY 核对最大误差均 1.49e−8；CPU RNG 未变。逐态数据包含全部 12 通道、真实输入、动作后接触标签、cap/B/gate 与源 rollout 哈希。

## 方向与幅度

表中是相同状态、相同 HISTORY 下，最终相对源 CP 的平均确定性 `REQUEST = cap × tanh(mu)` 变化（deg）；它是 filter/rate/headroom 前的策略请求，不是下发 target、实测关节位移或随机动作期望。

| 实际保存状态窗口 | FL hip | FL knee | RL hip | RR hip | RR knee | FR knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 首回合 FR（9） | +.179 | −.123 | −.059 | +.074 | −.022 | −.211 |
| FL 捕获／首次 AIR（10） | +.369 | −.128 | −.111 | +.116 | −.001 | −.724 |
| P06 后段（6） | +.433 | −.228 | −.165 | +.175 | −.017 | −1.417 |
| 首回合 P09（6） | +.467 | −.166 | −.178 | +.154 | −.042 | −1.252 |
| 次回合 P09（6） | +.443 | −.207 | −.152 | +.189 | +.029 | −1.414 |

捕获到 P09 四个窗口，FL hip network mean 增量 +.136–.146，RL hip 增量 −.063 至 −.081，RR hip 增量 +.067–.099；RR knee 保持负值，变化较小。仍使用 `mu = .1×network_mean + .9×真实HISTORY`，固定 HISTORY 项没有变化，因此这是实际网络响应变化，不是 HISTORY 被替换。这里的 `Δmu=.1Δbase` 只适用于固定同态 HISTORY；闭环中真实 HISTORY 会随动作累积、状态也会改变，因此小的同态 REQUEST 差不是长期物理偏移上界或稳定性证明。

具体到次回合 P09，平均 network mean：FL hip `.02684→.17011`，RL hip `−.28389→−.35117`，RR hip `.49792→.59671`，RR knee `−.30151→−.29260`；最终平均 REQUEST 为 `+5.984° / −5.770° / +10.713° / −10.674°`。对比已保留 CP189440 的同态结果，最后 512 decisions 后 FL hip 又向正增加约 .12–.16°，RL hip 又向负约 .095–.131°，RR hip 的正请求回撤约 .046–.058°，但相对 CP187904 仍更正。不能因此宣称已形成诊断中的成功全身协同，也不能把原始符号直接翻译成身体高度改善。

四轮 mean 方向判断未反转：FL wheel network mean 转负、RL wheel 更正；FR 略减、RR 保持负且变化小。捕获到 P09 各窗口的平均 ΔREQUEST 范围（FL／FR／RL／RR，rad/s）：`−.01024…−.01144 / −.00129…−.00428 / +.00977…+.01175 / −.00029…+.00018`。不是 mask 或 actuator 跟踪结论。

## sigma 与均值贡献

同态有效 sigma 比值（最终／源，捕获到 P09）：FL hip ×1.337–1.449、FL knee ×1.356–1.436、RL hip ×1.175–1.187、RR hip ×1.010–1.045、RR knee ×1.044–1.085；FL wheel ×1.428–1.599，FR wheel ×.921–.964，RL wheel ×1.076–1.127，RR wheel ×.889–.925。状态/B/cap 不变，变化来自网络 log-sigma head。

闭式 `KL(source||final)` 分解的窗口平均（12 通道求和；均值项／sigma 项）：FR `.16043/.28323`，捕获 `.24375/.44472`，P06 `.51757/.54573`，P09 首 `.31401/.59020`、次 `.39325/.59279`。sigma 项在本次五窗口均较大，但均值项仍占约 35–49%，并非“只学 sigma”。这是冻结同态 Gaussian 分解，不是实际 PPO update KL 或梯度因果归因。

结论边界：定向 37 态不是完整状态分布；未产生新物理行为、GAE 或训练 credit。是否真正保持 FL 接触、提高 RR 空间或解决 P09 碰撞，以 root 正在运行的最终 CP 闭环评估为准，不以这份报告作成功或新设计门槛。

完整文件：`CP187904_CP189952_same_state_response.json`，SHA `2adff31a370a7b0418ec1966cc8a5078b466cd99824f228b5085b3aea41835bd`。复用工具：`compare_CP187904_CP189440_readonly.py 189952`；本次仅为输出工具增加两个授权固定目标的选择，不修改生产。

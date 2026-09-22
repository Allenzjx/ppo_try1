# CP187904 → CP189440：真实同状态响应，只读 CPU 对照

**结论：均值确实改变了，不是只改 sigma；但本组数据不显示方向已被全面纠正。** 捕获／P06／两次 P09 窗口中，FL hip 的 network mean 向正方向移动，RR hip 原本为正且进一步增加；RL hip 原本为负且略更负，RR knee 仍为负、变化较小。不能把这些原始关节符号直接解释成“已提高身体”或“已避免碰撞”。

固定来源：当前 natural-P01 run `20260921T0727365666248Z_gee5a9651591d_6c75964ec21a46e296fcc856b2ee2064`，取 **37 个真实保存的 372 维 input states**：首回合 FR 9、FL 捕获／首次 AIR 10、P06 后段 6、首／次回合 P09 失败接近段各 6。只载入两个 CP，不采样、不调用 optimizer、不启动 Isaac、不使用 CUDA、不改 pointer；CPU RNG 未变。终态样本的输入是动作前状态，JSON 的接触／tick 标签明确是该动作后的实际结果，两者没有混写。

| 固定 checkpoint | decisions / PPO updates / optimizer steps | SHA256 |
| --- | --- | --- |
| `checkpoints/history/checkpoint_step_000187904.pt` | 187904 / 1433 / 28660 | `e3a405312514c8c99dd55195a7f8648cff5c44af48e33c4a7e7429a2a2d2eb4c` |
| `checkpoints/history/checkpoint_step_000189440.pt` | 189440 / 1445 / 28900 | `4261a388a51e6e6a5763845c209240cf9a692d81c49a469e2bb575d1a37cee6b` |

## 均值方向与 HISTORY

两个 CP 使用完全相同的原始记录 HISTORY center（独立从真实保存 observation 核对，最大 CPU/GPU 差 1.49e−8）。`mu = .1 × network_mean + .9 × HISTORY`，HISTORY 项在本对照严格不变。因此下列同状态变化来自网络，而非闭环历史被替换。source CP 在首批原始采样上的 network/conditional mean/sigma 复算最大误差分别 5.96e−8／1.49e−8／1.49e−8。

以下是各窗口的平均 **Δ[cap × tanh(mu)]**，单位 deg；这是 mapper/filter/headroom 之前的确定性 REQUEST，不是 actuator target、真实关节变化或随机动作的期望。

| 窗口 | FL hip | FL knee | RL hip | RR hip | RR knee | FR knee |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 首回合 FR | +0.110 | −0.036 | −0.001 | +0.081 | −0.007 | −0.197 |
| FL 捕获／首次 AIR | +0.245 | −0.015 | −0.016 | +0.162 | +0.038 | −0.739 |
| P06 后段 | +0.275 | −0.090 | −0.034 | +0.221 | −0.001 | −1.398 |
| 首回合 P09 | +0.312 | −0.039 | −0.056 | +0.203 | +0.008 | −1.249 |
| 次回合 P09 | +0.281 | −0.077 | −0.026 | +0.248 | +0.038 | −1.418 |

例如次回合 P09 的窗口平均 network mean：FL hip `.02684→.11769`，RL hip `−.28389→−.29540`，RR hip `.49792→.62712`，RR knee `−.30151→−.28999`；对应固定 `.9×HISTORY` 为 `+.17245 / −.21079 / +.42165 / −.27647`。最终平均确定性 REQUEST 仍为 `FL hip +5.822°、RL hip −5.644°、RR hip +10.772°、RR knee −10.665°`。所以网络方向真实进入了 mu，但 0.1 权重与真实历史共同决定最终中心；不能称为已学到诊断中的全身协同。FR knee 的更大变化也必须纳入后续闭环解释，而非只看 RR。

四轮平均 ΔREQUEST，单位 rad/s，顺序 FL／FR／RL／RR：

| 窗口 | 四轮变化向量 |
| --- | --- |
| 首回合 FR | `−.005916 / −.001325 / +.002833 / −.000238` |
| FL 捕获／首次 AIR | `−.011666 / −.001786 / +.007275 / +.000246` |
| P06 后段 | `−.012781 / −.004462 / +.008068 / −.000092` |
| 首回合 P09 | `−.012484 / −.002740 / +.009082 / +.000012` |
| 次回合 P09 | `−.011837 / −.004380 / +.008277 / +.000176` |

FL wheel network mean 在这些窗口转负，RL wheel 更正，RR wheel 仍负且变化很小；这是策略同态响应，不是 mask／指令丢失诊断。

## sigma 也显著改变，但不能全局归因成“只有 sigma”

捕获到 P09 四窗口内，FL hip sigma ×1.218–1.316、FL knee ×1.263–1.335、RL hip ×1.183–1.199；RR hip ×.995–1.031、RR knee ×1.026–1.057。FR knee ×1.295–1.369。此处状态 gate/B/cap 固定，变化来自保存网络的 log-sigma head，不是更换采样尺度配置。

37 态按窗口的闭式 Gaussian `KL(source||target)` 分解（均值项／sigma 项，12 通道求和再按窗口平均）为：FR `.10085/.14740`，捕获 `.21640/.23844`，P06 `.45668/.30367`，P09 首 `.29018/.34700`、次 `.37065/.34894`。两项总体同量级；有的通道／窗口主要扩 sigma，有的均值贡献更大。**这不是实际 PPO update 的 KL，也不是梯度因果归因。**

边界：37 态是定向小样本，不是完整分布评估；它们由采集时不同策略产生，在两个固定 CP 上离线前向，不进入 buffer，不产生 GAE 或新学习。不能据此声明 CP189440 已解决 FL 保持、P09 碰撞或获得完整成功；须以独立闭环评估判定。

完整逐态 12 通道的 network mean／0.1 项、真实 HISTORY／0.9 项、conditional mean、cap×tanh REQUEST、learned/effective sigma、gate、原始 observation 哈希与源 rollout 哈希见同目录 `CP187904_CP189440_same_state_response.json`（SHA256 `4c36842def154587fb2861115baf88af9229ddfec998fe8530986d1922734b5b`）。可复现脚本：`compare_CP187904_CP189440_readonly.py`。仅上述输出新增，生产和训练状态未改。

# Video6：真实 FL 放置后退回地面，P06 期限未完成

固定 run `20260910T1528556872035Z_g7db0d17f398d_a5db22d8e4b545969459f6ea059a6c87`，2026-09-10 15:52:36.790432Z 完成。实际官方加载 checkpoint **151936**，无迁移，372维 history policy、deterministic `conditional_mean`、seed4001；自然 P01 首次 reset，options={}、prime/pre-action ticks=0、无教师前缀、0 optimizer updates。初始加载 checkpoint/manifest/actor hashes 见同名 JSON；未重新哈希模型。物理未成功分支在 `check_model()` 前退出，**未执行末尾模型不变哈希断言**，不能把初始加载凭据当最终比对。

1176 决策 / 9408 physics ticks / 78.4 秒。P01–P06 样本分别 **2/170/7/39/358/600**，P07–P13=0。终止为 `P06 / INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_TASK_DEADLINE`：P06 从 tick4608 /38.4s 开始，stage_age=40.00000000000001，effective_limit=40+0+0=40s，rear_approach=0。不是200秒全局窗口、外部截断，也不是 BODY、wheel-only 或独立安全中止。作业 `DIAGNOSTIC_FAILURE` 与上述物理未完成一致，不能发布成功或稳定性改善结论。

## FL：两个抬升尝试，第二次确实放置

| 事件 | tick / 秒 | 实际证据 |
|---|---:|---|
| 首次 I / Q | 1805 /15.041667；1809 /15.075 | 测得向上3.318 /9.327mm；Q不是TOP |
| Q撤销 | 1843 /15.358333 | `qualification_revoked_ground_before_cross`，实际回地；1848端点GROUND |
| 新 I / Q | 3890 /32.416667；3937 /32.808333 | 新向上5.544 /8.103mm；旧event_ticks.active_lift=1809是历史首时间，不能冒充本次Q |
| C | 4486 /37.383333 | exact front +0.028554mm、gap +1.132817mm、数值载荷分数0；不把越沿当放置 |
| P | 4604 /38.366667 | exact front +13.512734mm、gap −0.275764mm、数值载荷分数0.148173 |
| P05→P06 | 4608 /38.4 | 实际TOP、非GROUND、support=true、bearing_verified/load_fraction_valid=true；承载5.405339N，载荷分数0.188347 |

生产放置规则是已C后连续2个真实TOP样本（非地面、在同一顶面区域），并非仅历史膝角或摆腿幅度。P4604的精确力/有效性字段没有在小事件文件中单独存储；表中精确事件几何来自 `stage_transition_evidence.jsonl`，5.405339N则明确来自**后4 ticks的4608决策端点**。二者不可冒称同一时刻。4616第一个P06端点仍TOP、有效承载7.942325N。

## P06：已完成历史保留，但当前支撑退化

- 首个失去TOP的决策端点 **4880 /40.666667s**（发生于4872–4880之间）：front −5.437013mm、gap −1.500815mm，仍有有效承载11.579612N。这时不能称完全无支撑。
- **5024 /41.866667s** 首次观测FL support=false、承载0、bearing_verified=false、全局 `CONTACT_BEARING_UNVERIFIED`；front −33.177134mm、gap −14.527399mm。
- **5168 /43.066667s** 首次回GROUND（5160–5168之间），front −49.268361mm、gap −50.827798mm。以后没有重新TOP；P06共33 TOP、36个非TOP非GROUND、531 GROUND端点。
- 末尾 FL GROUND 且 obstacle surface=`OBSTACLE_AMBIGUOUS`，front −47.565785mm、gap −50.430918mm。虽有ground分量/support与数值bearing=12.369541N，但 `bearing_verified=false`、`load_fraction_valid=false`，**不能把其0.377640数值分数当已验证全身载荷分配**。已放置/C历史仍true，不代表当前仍在顶面。
- RR 的完整事件列表 **没有 I/Q/C/P**。此前P05有短暂AIR端点，但不满足有效抬升事件，不能把AIR等同Q；P06全部600个端点GROUND/current_lift_valid=false。末front −467.174691mm、gap −50.049706mm、ground-relative lift=0。

## 实际动作与核验边界

所有1176个端点12通道mask均开；9408 native ticks逐一连续、全部 verified/actual_effect，own_phase_effect=9403；no_in_episode_state_writes=true。**执行链VERIFIED不等于末端接触承载VERIFIED**。P05的358端点满足tanh(raw)×cap；P06为598/600，另外2条有投影/变化率链作用（最大full12分量差7.457008），没有逐tick证明其具体原因，不称“完全无下游投影”，也不凭此认定缺陷。

P06 nominal端点四轮通常+.3rad/s，FL nominal hip/knee=22.8/−13.4deg。可核验的前后动作如下（角度deg；wheel顺序FL/FR/RL/RR，rad/s）：

| tick | FL residual hip/knee | FL实际target hip/knee | 实际四轮 |
|---:|---|---|---|
| 4608，交接端点 | +4.9654/+22.0177 | 27.2205/9.8677 | −.6187/−.1447/−.0236/+.5188 |
| 4616，第1个P06端点 | +8.4654/+25.5177 | 30.7205/13.3677 | −.4237/+.0503/+.2788/+.8176 |
| 4880，失TOP | −1.4793/+27.1522 | 20.7758/15.0022 | −.6047/+.0614/+.0302/+.6419 |
| 9408，终止 | −.3175/+27.5615 | 21.9375/15.4115 | −.6870/+.0955/+.1067/+.7128 |

交接4608的 **nominal四轮确为0**，前一端点4600及4616均+.3；actual wheels及residual并没有全清零。残差端点连续非零不证明所有单tick都零跳变，更不能隐去这次nominal变化。本次未额外重建单tick bridge，不能把它或负FL轮速直接归因为退回地面的唯一原因。未证明控制、概率或reward符号缺陷；不提出新的成功门禁。

部分轨迹roll/pitch RMS=0.081844/0.112217rad、对应角速度RMS=0.055600/0.057289rad/s，仅描述这段失败轨迹，不与FSM宣称全程稳定性更好。

## 诊断录像：无损容器修复，仍为 incomplete

原capture 1176帧/15fps，全解码PASS、unique1176、black0；原container duration=286331153s错误。单独packet-copy生成：

- `video_diagnostics/video6_151936_p01_incomplete.mp4`：严格容器78.4s，110380439B；SHA256 `4b13e0b62fdba5a6f95829345dde39fd46aa4a3556ed089f4a21a4c04298c70e`。
- `.remux_receipt.json`：逐帧checksum、绝对PTS、PTS差分及key flags严格一致，全部1176帧重新全解码通过；原文件110515863B及原manifest前后字节/hash不变。
- 首末图：`video6_151936_decoded_01.png`、`video6_151936_decoded_02.png`，均在 `video_diagnostics/`；frame0/PTS0与frame1175/PTS78.333333，供主线程视觉核验。

无重编码、裁剪、倍速、插帧、拼接或success publisher。系统Python未从PATH找到ffmpeg的首次检查没有写媒体；随后使用本次capture记录的现有ffmpeg完成处理，未安装软件。完整原录像、manifest及失败结论全部保留。

范围：一次本次已完成policy ledger扫描、小manifest及131KB事件文件；未读其他视频历史或独立巨大physical/native流，未碰生产、checkpoint、主报告、当前训练进程。附同名JSON为定向证据摘要。

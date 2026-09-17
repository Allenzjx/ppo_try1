# 512块第三episode：P06真实接触失败，未见执行矛盾

只读run `20260916T0628168580885Z_gfc14a68c037a_fe364dae15434293be866a2fe834d015` 的3条已结束episode摘要及尾部100条中筛出的第三episode94条（global177470–177563），未扫描后续物理记录、未读活内存/重算GAE/forward。当前固定块及生产未修改。

第三episode全程P06，N从P01到tick2680交接；94 decisions后tick3429 / 28.575s终止 **BODY_COLLISION**，物理source=BODY_CONTACT，reason=`central body/obstacle collision`。这是独立于前两个P09 FALL的失败，没有提前接管或阶段跳跃。RR末拍仍有承载、非AIR、current_lift_valid=false，距前缘−.40197m。

现行物理评价器只在权威 `body_collision.detected` 为真时给出此任务失败，不是因reward低或阶段超时。终态机身bbox底z=.05000413m、前x=.641611m，与5cm障碍顶面接触及前缘重叠相容。训练摘要未保存完整body接触力向量，因此不能凭此报告具体接触冲量；但现有接触判定/几何/连续下沉没有矛盾，不能改判。

后段实际变化清楚：tick3312→3344→3376→3408→3424，FR knee final为66.063→82.063→98.063→114.063→122.063°，body底world-z从.123335→.108951→.087103→.066874→.052108m。支撑由FR/RL/RR、FL/RR，随后FL/FR/RR变化，终拍仅FL/RR。末拍FR knee final119.563°，actual117.734°（前1物理tick）；实际在跟随较大的目标，不能仅称传感器错或servo没执行。

末拍FR knee合成分项为source N45.9°、mapped N45.958976°、controller bias0、有效residual73.604336°，总119.563312°。四轮final `[−.438445,.383036,.329517,.270201]` rad/s，actual `[−.464093,.381835,.328550,.226509]`，FL反向已实际发生。mask全12个1，映射及native目标读回相符；最后5物理tick全verified/no非法state writes，无finite fallback。不能把source-N与最终servo差异误称丢失或偷偷叠加。

为何不同于成功CP177152：

- 成功评估是旧best固定权重、自然P01起、全程deterministic conditional mean；P06已包含前阶段策略塑造的姿态、接触、mapper/residual和非零HISTORY。既有媒体首P06采样为tick2656，四轮target约 `[.41676,.33653,.43727,.28057]`。
- 此课程先由成功N、零raw从P01真实roll-in，tick2680才接管P06；不是best策略的相同入口或静态快照。它保留真实physical/controller状态，但policy-history由N零raw形成。阶段名相同不能保证身体状态或命令历史相同。
- 当前训练是0.25随机innovation，与成功评估没有随机innovation不同；rho=.9会延续已采样的raw。小创新与有记忆的动作映射仍可累积显著关节/轮差速，低温不等于确定性评估。
- 网络也不是CP177152：已做首128及后续updates。本episode在global177536（本episode第67拍，tick3216）完成update1352，随后以更新网络继续同一真实状态。这是正常on-policy分块边界，不等于整个94拍来自一个冻结模型；不应拿新actor重算旧动作来解释。

结论：**有效探索接触失败，未确认执行实现错误**；不同入口、采样与网络变化共同构成合理解释空间，但不是已分离的因果证明。保留best及失败样本，按授权继续固定块，结束后保存/重载正式评估；不热改N/mask/reward/guard，也不把单次失败设为optimizer门槛。待完整块结果再决定是否需要针对性入口/载荷诊断。

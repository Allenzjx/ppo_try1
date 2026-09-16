# 最近10次正式 C：前腿备用 checkpoint 元数据

仅检查当前experiment validation最近10个非launcher、已完成run/source manifest；不读tensor/轨迹，不重算hash，不运行Isaac，不更改latest。当前训练后的最新checkpoint仍是首选。

| Checkpoint decisions | Source HEAD | FR越沿 tick | FR放置 tick | 全run观测 ticks | 备注 |
|---:|---|---:|---:|---:|---|
| 161152 | d4e46006b382 | 1380 | 1441 | 5309 | 最近的前腿实证备用 |
| 159104 | d4e46006b382 | 1147 | 1188 | 5045 | 旧规则下备用，未排序稳定性 |
| 157568 | d4e46006b382 | 1291 | 1327 | 5772 | 旧规则下备用，未排序稳定性 |
| 154240 | d4e46006b382 | 1484 | 1518 | 5853 | 后续另有硬限失败 |
| 151936 | 7db0d17f398d | 1388 | 1429 | 9408 | 旧规则下备用，未排序稳定性 |

较新的168192、167424两次、166784、164736均未记录FR越沿/放置并以HARD_JOINT_LIMIT结束，不作为此次前腿备用候选。全部10次均不是完整越障成功。

5候选均记录自然P01从tick0起、official deterministic C reload、372/12、HISTORY rho=.9、0 optimizer更新、无干预或教师offset。其action/observation/quality配置指纹与当前启动manifest相同；task/execution/reward不同。旧formal manifest没有新诊断wrapper的post-capture frozen布尔字段：缺失是unknown，不是false，也不在本次虚构重新验证。逐tick全12mask/非零动作没有重新扫描；这里只声明其已有all12执行合同。

evaluator_version都写all_stage_v1，并不保证语义相同。候选来源d4或7db，早于当前nominal顺序、carry reward、RR有界geometry、FLheight及晚期原子组入口等修订；不能把历史FR成功直接算作当前规则验证。151936–161152只证明这些既有权重曾在旧正式自然P01闭环中完成FR越沿/放置，不能称稳定性最佳或完整PPO成功。

若最新完成训练后的正式评估仍退化，161152是最近的独立前腿备用候选；须采用明确兼容迁移与当前完整评估，不能自动覆盖现行latest或沿用旧rollout。其余候选未按稳定性排名。此查找不阻断当前训练，也不扩大搜索范围。

具体checkpoint/source manifest路径与已记录配置指纹见 recent_formal_front_fallback_candidates.json。部分旧物理manifest终止字段为空，仅保留“不完整/diagnostic failure”；本次没有猜测其精确后续阻塞原因。


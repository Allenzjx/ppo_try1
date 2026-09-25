# 隔离候选：N1 / rear-owner439 的 512 步收集边界

状态：**只准备，未应用生产，未加载任何模型，未新增 PPO/AUX/物理样本**。目标仅将新收集块由128改为512。默认/全部历史profiles仍128；gamma=.9985、lambda=.99、网络、实际Adam/LR、reward、控制、HISTORY和观测不变。

## 最小修改面

仅五个现有runtime文件，完整候选位于 `tree/src/wlr50_clean/ppo/`；`collection512.patch` / `collection512.apply_patch` 是同一修改。原始字节与候选SHA在 `source_target_sha256.json`。应用后须按实际文件字节重建runtime contract，不能把隔离目录SHA当生产运行证据。

- `semantic_return_profile.py`：collection marker与数学return身份分离。无marker只能128；新marker只能512+当前v3 return。PBRS仍校验version/gamma/lambda，不改reward文件中的历史rollout字段。
- `semantic_training.py`：显式factory参数，仅rear-owner439/v3允许512；独立检查实际N1、actions[512,1,12]、policy/critic[512,1,439]。collector已有动态batch/计数保持；新receipt随每次checkpoint继承。
- `semantic_front_retention439.py`：旧65a validator只增加一个入口分派，其原128/20公式及AUX验证不改。新outer先对绑定的真实65a源运行原完整祖先链，再独立检验新段512/20和不可变AUX/旧receipt。零更新publication保留actor/critic、完整Adam、LR、normalizer/RNG，新建空storage。
- `semantic_cli.py` / `semantic_video_cli.py`：普通train/eval与正式视频通过同一已发布checkpoint marker选择构造参数；无新checkpoint不改变默认。不增加新CLI命令。

旧自动curriculum仍锁定128，本首试**不用该调度器，也不宣称执行它**；继续已有官方train单块调用。新块每次512决策/+1 PPO/+20 Adam；5 epochs / 4 minibatches保持。实际更少更新、较大minibatch是实验的一部分。

## 合法边界后的最小使用顺序（尚未执行）

1. 等当前65a P07运行正常封存，核验**当时实际最新**完整checkpoint/sidecar/pointer，不能固定CP229120 AUX32或当前读到的中间编号。
2. 审查/apply这五处后冻结新commit与完整runtime contract。无reward/config/物理文件修改。显式调用现有模块新增的单个API：
   `publish_collection512_checkpoint(checkpoint, contract, output_checkpoint, reason=..., expected_source_sha256=..., expected_manifest_sha256=...)`。
   它要求真实latest pointer、源65a完整更新、唯一同branch输出；不自动提升pointer、不运行teacher/优化、不伪称环境状态续接。旧128先正常加载，新512精确转移全部学习状态；保存后再用新512正常loader重载。
3. publication返回零增量和新文件哈希。用**该新checkpoint**走既有官方train入口，首试显式512或其整数倍；不要将512称为四次PPO。成功保存后由原有checkpoint流程更新pointer。
4. 下一轮评估/视频继续同一checkpoint和普通loader，辅助声明不变。半衰期仍约4秒，长块不保证30秒前动作获得强终止信号，也不保证能力改善。

## 已做与未做验证

已做：`test_collection512_stdlib.py` **19 tests passed**，五个candidate AST均可解析。覆盖旧默认精确等价、marker/长度/布尔反例、PBRS gamma/lambda仍严格、512/20计数、拒绝128或四次伪计数、AUX/旧receipt不可变、零credit身份、branch累计量、正常/视频构造选择和旧128公式保留。测试只使用stdlib；lineagefixture的历史源验证调用是显式mock，**不是实checkpoint祖先链或Torch身份验收**。

仍必须在Isaac退出后做真实source publication/reload身份验证及最小固定输入mean/sigma/value、完整Adam/RNG、空buffer验收。首个真实512块核对raw样本/oldlogp、每样本5轮、实际20步、phase不done、真实terminal断开GAE、尾bootstrap和teacherprefix0credit。不得因这些检查尚未运行而声称已续训512或已解决P12恢复。

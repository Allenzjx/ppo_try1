# 单通道独立候选：尚未执行

只用于检验：**在 P12 源四轮 stop 已实际派发、RR/FL 当前均合法 TOP bearing 的窗口，削弱 FL 持续负向 residual 是否减少身体/RR 向后退，而不破坏支撑。** 无改善或更早失去支撑可否证这个方向；不预设 FL 是唯一原因，更不提供 AUX 成功标签。

`fl_wheel_residual_half_probe.py` 复用旧 `student_entry_direction_probe.py` 的不可变 checkpoint 校验、普通官方 loader/conditional-mean kernel、后腿辅助 OFF 核对和 JSON/actual-audit；旧脚本未修改。新脚本默认拒绝执行，只有明确 reviewed flag 才可能启动。checkpoint/manifest SHA 与完整 frozen HEAD 必须由调用者指定，不自动读取可能过时的 last pointer，也不做任何模型迁移。

## 生命周期和干预

- 单一 ordinary core：`_CheckpointPolicyCreditCore` 安装 observer → 唯一自然 P01 reset → tick-0 official checkpoint loader → `successful_nominal` 零残差真实连续前缀到 P10 → 同一 core 普通 student suffix。无快照/teleport/重建控制器。前缀失败就在 result 回执终止诊断，不触发 helper 的自动 fresh-P01 fallback。
- 候选仅支持当前 439 维 `role422_rear_owner_recovery_v1`、12 动作、120 Hz/15 Hz；runtime 精确相同才加载，所有阶段 all12 许可继续核验。名义前缀/诊断 suffix/干预全部 PPO、teacher sample、optimizer、AUX credit 为 **0**。manifest 声明成功 nominal 初始化行为存在，但没有 learned teacher、隐藏后腿补全或训练标签。
- 不以 N=0 证明 stop：读取 P12 当前确实发出的四轮 atomic group，再等下一实际 physics dispatch 的 ACK/native audit 验证；随后才允许触发。其间任何后续 wheel event 或 nonzero nominal 会使旧 stop 失效。
- 仅一次有限窗口，至多 3 秒/45 个完整决策。每次从**当拍新鲜原始 student raw**计算 `raw[8] = atanh(0.5*tanh(original_raw[8]))`，仅当其为负；正/零和其他 11 维原样传入。不是每拍在上次输出上继续减半，不改 N、mapper、source owner 或 HISTORY。
- 当前 RR/FL 任一失去合法 TOP、实测 bearing、XY/净空条件，或离开 P12，即逐 physics tick 锁存退出。普通 core 的每个决策固定 8 ticks，因此**下一决策不再干预，可能有至多 7 个剩余 ticks 延迟**；原安全仍逐 tick 终止。为保持 ordinary kernel 不插入中途 raw 改写。退出后正常 student 同一回合继续至原终态/200 秒。
- 3 秒只界定修改 raw 的时窗；普通 mapper/filter/HISTORY 的既有效应可在退出后连续衰减，绝不谎称 final target 在退出那一 tick 突然恢复。

每拍保存原始 raw、original request audit、实际 raw、changed index、tanh 前后值；逐物理 tick 保存 source N、actual final target、ACK/readback、真实接触/几何、rear-owner evidence。最终目标与另算 nominal 的差**不**冒充当拍 policy 作用。DET audit 没有 stochastic selected log-prob；干预 action 不进入任何 PPO likelihood/storage。

## 离线验证范围

`test_fl_wheel_residual_half_probe.py`：25 个纯 stdlib 测试通过（2026-09-24），包括单维 tanh 变换、输入不变、源 stop 未来/实际回执区分、陈旧/重复 ACK、current support 正反例、历史 placed 不代替 bearing、一次性/时长/支撑丢失退出、zero credit、普通单 core 生命周期和实际 raw 审计。仅标准 Python，不导入 Torch/PXR/Isaac/生产模块，不启动当前训练之外的新物理实例。

这些测试只证明接线/门控语义；**未验证物理收益、未证明可用作 AUX 标签、不是正式 PPO 成功**。是否执行仍由 root 在当前训练正常保存结束后决定。

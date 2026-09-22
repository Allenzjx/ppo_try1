# Budget-only 首个真实 update 核验

仅只读 CP192000、首个新 CP192128、已保存 rollout/update 和当前生产代码；未创建 runner、未重放优化器、未使用 CUDA／Isaac。

- 首次真实续训完成 **128 decisions / 1 PPO update / 20 optimizer steps**：全局 192000/1465/29300 → **192128/1466/29320**。128 个实际样本为 P01=2、P02=126，无 teacher prefix 样本。
- full_episode 已用量 **99968→100096**，suffix=81920、smoke=0 不变；新上限 131072，余量 30976。原 task 分支起点 185856/1417/28340 保留，分支累计变为 6272/49/980；其他原分支及预算起点亦逐项相同。
- 首次实际 update 的 `actor_parameter_sha256_before` **精确等于源 CP actor hash**；update 后 actor 确实变化。两 CP 的 actor/critic 全 tensor hash、完整 Adam hash、内嵌 infos 与 sidecar 均重新核验通过。
- 实际 Adam 是 **1 个 param group、12 个 state entries**，组配置全部保留、LR=1e-5；每个参数的 Adam step **6220→6240**，两阶动量均有限且非零。Adam 内部 step 与全历史 optimizer_steps 是不同计数，不能混写。Identity normalizer hash 未变。
- 正式 `checkpoint192000_quantity_continuation.json` 经官方 validator 再验；SHA `2f551128effcb1f6f039ced3d5c3487b8860408ffc71b736a1471975fce1023c`。新 CP 内持久化预算 receipt 与其完整精确一致，`new_mdp=false`、`kernel_changed=false`。
- 实际 post-load 身份记录为 `training_after_actual_reset_and_checkpoint_load`，09:57:15.768685 UTC。此前必经 loader 对实际 actor/critic/full Adam/normalizer 的 hash 校验、全部 param-group LR 与算法 scalar 同步，以及源 RNG 恢复。未另存 update 前 Adam/RNG 快照，因此不把 update 后状态冒充恢复瞬间的直接快照。
- 保存 RNG seed 仍 1001；Python／NumPy／torch CPU 状态与源相同，CUDA 状态已随真实采样／更新推进。未重放 CUDA RNG，也不声称 RNG 在训练后仍不变。
- 熵退火仍用旧常量总和 **210000**，源码语句未变；在 global=192128 的系数按实际执行语句为 **0.0013404190476190478**（系数未单独日志记录）。日志 `entropy=-26.6794967651` 是分布熵，不是系数。

首 update KL=0.0136234、clip_fraction=0.278125、value_loss=0.0237871；这证明实际更新，不证明越障成功。

首新 checkpoint：`outputs/ppo_task_conditioned_hip_wheel_v1/checkpoints/history/checkpoint_step_000192128.pt`，SHA `6463fae5d8e15984104074145fdb259a2f98028a228ed4fb8e6bbd53421e9eda`。当前主线继续运行；未修改生产。

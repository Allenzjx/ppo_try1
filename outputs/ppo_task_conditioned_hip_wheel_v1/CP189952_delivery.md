# CP189952：第二轮固定模型完整尝试（工作继续）

## 已交付视频

- `CP189952_deterministic_review/CP189952_deterministic_P01_full.mp4`：正常速度，P01自然起步，52.30秒，P05任务未完成。
- `CP189952_N_pair/N_vs_CP189952_deterministic.mp4`：同控制版本、场景seed4001的N+0与上述det对照。N完整成功73.808333秒；右侧结束后323帧明确标为定格，不是额外物理证据。
- `CP189952_stochastic_seed4101_review/CP189952_stochastic_P01_full_seed4101.mp4`：同CP，policy seed4101，42.391667秒，P09真实机身碰撞。

三条视频均全帧解码通过，root已看首尾图并交付；同模型/控制版本/场景/相机绑定由`CP189952_same_model_modes.json`核验。原片、失败等待、原run和旧CP187904视频全部保留。所有C正式评估均全12通道、无人工偏置/教师/zero fallback、零optimizer更新。

## 已验证结果

确定性run：`video_eval/validation/20260921T0801290800396Z_gee5a9651591d_a6d92def5d74491580cccd936a1edf18`。
FR lift/cross/placed=24/1785/1796ticks；FL lift/cross=1836/2888，未placed。
终点tick6276，`INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`；FL AIR、gap21.165051mm、承载0N。
独立物理评价有效、未报告硬碰撞，但任务未完成，不能称成功。外层`DIAGNOSTIC_FAILURE`不表示文件损坏。

随机run：`video_eval/validation/20260921T0819434957323Z_gee5a9651591d_b66847e1dd544fdba49da0c6bb528346`。
FR placed1684、FL placed2988，但后续FL承载不足；tick5087真实`TASK_FAILURE_BODY_COLLISION`。
RR从未取得qualified lift、从未越沿/放置；终点虽AIR，`current_lift_valid=false`，不能称有效抬起。
首次未可靠保持的前序任务为FL捕获后接续；终止时尚未完成RR有效卸载。无完整或后缀成功。

FR同物理事件窗口：

| 指标 | 当前N+0 | CP187904 det | CP189952 det |
| --- | ---: | ---: | ---: |
| 捕获时间s | 12.5167 | 13.5000 | 14.9667 |
| roll/pitch-rate RMS rad/s | .117889 | .104231 | .092906 |
| body collider最低z mm | 91.272 | 92.512 | 91.577 |
| 首次15mm AIR至越沿前平均FR gap mm | 98.762 | 96.127 | 88.925 |

姿态角速度更低，伴随变慢和净空下降；不是无代价稳定性优势，也未证明RL回收单独造成该变化。RL hip实际均值与旧det基本相同，RR hip实际均值反而增加.627°。完整定义见`CP189952_FR_window_readonly.md/json`；RMS未除以时长，qd不当作牵引速度。

## 实际训练与模型

最新保存模型：`checkpoints/history/checkpoint_step_000189952.pt`，SHA256
`cf89d3c206874a5d239b21e622752f7f294f3219b4b587853c43840a1de570d2`。
累计189952 policy decisions /1449 PPO updates /28980 optimizer steps；本分支自185856新增4096/32/640，保存重载已验证。
最新并不等于最佳成功模型：尚无本分支完整P01 PPO成功模型。

| P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 | P11 | P12 | P13 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 | 1077 | 20 | 7 | 1217 | 1357 | 5 | 8 | 393 | 0 | 0 | 0 | 0 |

此前真实前缀2984决策独立排除；视频和诊断也不算训练。第二块2048全部自然P01；三次完整回合均真实FR/FL捕获后FL保持不足、RR短暂qualified后回地并P09碰撞，所有失败进入更新。最后8决策仅预算边界，不称终态。

采用代码仍为`ee5a9651591d20bea48be8ceba075fa66594cb36`：状态sigma、任务质量/空间下界及其训练/保存/加载/视频接线；未再修改nominal、mapper、评价器或物理。
训练继续使用372输入、12维、rho=.9、LR1e-5、128rollout、5epoch×4minibatch、完整Adam/Identity。迁移仅在原完整边界完成，旧rollout不复用；具体12通道B表与reward见`TRAINING_DESIGN.md`。

## 后续实际工作

同CP随机原片已完成。08:35UTC启动一次40秒上限的FL hip−3°独立诊断，只用于当前真实响应，不记PPO/aux信用、不作为optimizer门禁。再补现有P09+35真实N前缀carry课程并恢复自然P01采样；不注入AIR快照，不把后缀当全程。尚未开启辅助监督、重置网络或放大HISTORY梯度。

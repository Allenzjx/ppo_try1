# 新REQUEST-history kernel：真实采样/更新审计

## 首个完整update1376（128个真实决策，已完成）

Run：`20260918T0627100878188Z_g3a50657a96c9_92db59ba0d774fa78e4cc53d5fad8c77`。实际768预算/P04/checkpoint_policy/offset0/seed1001；本节严格只核验其中**第一批128**，不包含后续尚未封存的数据。

- 已完成新增128 decisions、1 PPO update、20 optimizer minibatches；CP180608累计180608/1376/27520。实际覆盖 **P04=1、P05=127**。
- 128/128真实request使用新policy版本。raw、旧conditional μ、effective σ、old logp与已保存rollout逐张量**完全相同**；request logp亦与采样日志相同。原始raw-history、previous-filtered-REQUEST分别从同一保存372 observation的195:207、207:219读取，匹配零误差，没有用shuffle邻居。
- 依据保存的phase one-hot、age20、completed158:171及真实cap表纯算术重建gate/center：gate逐项一致，center和conditional公式最大误差均0。`previous_filtered_REQUEST`不是post-headroom/final-slew command或actual q。
- **实测正例：0个cap-entry gated decisions。** age=0有P04、P05两个真实反例：predecessor completed=true，但cap未增加，全部gate=false；旧raw中心精确保留。其余同阶段样本也精确保留旧中心。这证明此批“不该触发时不触发”，不证明P06入口的正向分支已物理验证。
- 第一真实optimizer minibatch的32样本（尚未参数更新）ratio范围 **0.9999980330–1.0000019073**，最大偏离1为1.96695e−6，strict clipped branch=0。**不是声称全部128样本都在初始参数下测试过。** 共20个真实minibatch，每个样本实际使用5次，全部新history-source provenance正确。
- CPU仅以保存μ/σ/raw重算Gaussian logp，最大误差3.81470e−6；未调用actor、未采样、未额外优化。实际request记录每决策1次抽样、额外forward/RNG=0；实际所有physics tick执行校验通过。
- update1376有有限非零梯度、actor参数实际改变；LR=1e−5，KL均值0.01967256，value loss0.08437086。更新前actor hash仍是CP180480 `b5d7e834…f545`，更新后为`e07a490c…cedc4`。
- P03→P04与P04→P05的真实执行交接记录保留REQUEST，最大servo residual尾差分别8.88e−16/2.66e−15；无禁止通道丢弃。**此批没有P05→P06，不能把1377的后续入口归入1376。**

迁移/prefix绑定：源CP180480 SHA`ec48951d…b05a`；旧runtime `f6caf0c2…a1b9c`、新runtime `e1e2a003…7ca4`分别记录；prefix实际加载新kernel、参数字节保持源CP、独立storage与块内冻结=true；plan SHA`fff0eeced774ff663e66e492d4b00698bc0029f2315366568ac60462c0b8b3a9`与文件一致。不是把旧source contract冒充新kernel。

本次保存CP180608 SHA`98f03edadabaebd925b0ad436d2b62c23adc56b1db8e009e825c7ec8f268f09d`，manifest SHA`ba84944f25f5088baa4f65f990892392410d4d184db1e94f68305aa732cd9162`；manifest记录`save_load_round_trip=true`、继承正式迁移及180480/1375/27500来源计数。未另做actor reload/forward来消耗训练资源。

证据：`update1376_actual_request_kernel_audit.json`及原run的`rollout_001376.pt`、`update_001376_likelihood.json`；核验脚本`audit_first_request_kernel_update.py`。生产diff=0。这里确认真实接线/学习发生，不是任务成功或稳定性改善结论。

## 第二个完整update1377：真实P06正例与后续FALL

仅追加已完成的第二批128，global180609–180736，20次真实optimizer steps；不纳入1378活动数据。实际请求 **P04=1、P05=43、P06=75、P07=1、P08=1、P09=7**。本批所有存储/请求/概率/索引检查通过；raw、旧μ/σ/logp保存完全一致，CPU重建center最大误差1.49e−8、conditional误差7.45e−9（CPU/GPU float32尾差），保存Gaussian logp差3.81e−6。首个未更新minibatch32样本ratio范围0.9999980330–1.0000038147，strict clip=0，每个样本仍实际优化5次。

### 唯一门控正例与负例

- 唯一gate：本批index24、global180633、P06首决策tick2824→2832，age=0、P05 completed=true。前8个servo及FL/FR wheel cap增加，10通道gate=true；RL/RR wheel cap未增加，仍用原raw。P07/P08/P09以及新episode P04/P05的age=0全部正确不触发；其余同阶段center与原raw精确相同。
- FL真实placed tick2818。global180632的P05→P06决策结束tick2824时，FL TOP/support=true，bearing4.968506N、gap−0.227184mm；首P06决策末bearing5.613691N。这是checkpoint-policy前缀后PPO采样episode的真实捕获，**不是自然P01完整PPO成功**。
- P05→P06执行桥保留所有REQUEST，servo最大尾差8.88e−16°、wheel2.78e−17rad/s，handoff_hold=true；没有禁止通道丢弃或phase-scale clip。不能把分布中心改变叫动作清零或直接重置实际驱动。
- FR knee：前一filtered REQUEST=−6.605833°，旧raw=−.282528；新center=−.0590492。真实base=−.269934、conditional=−.0801377、sigma=.0857316、sample=−.0774727；该决策末REQUEST/effective=−8.659629°，mapped N45.321863°，final36.662235°。同一保存base/innovation仅换回旧中心的算术为−30.420560°请求；这不是额外actor前向或已运行反事实。
- **下一拍就不再门控**：global180634 raw=−.281818，REQUEST继续按原slew到−12.659629°；180635到−16.659629°。故真实正例证明只在合法入口抑制重表达突增，未消除后续负均值/噪声或持续全身作用。

### 捕获、跨阶段回报与失败学习的真实方向

| global / 事件 | done | reward | saved return | old V | 标准化advantage |
|---|---|---:|---:|---:|---:|
| 180632 / FL捕获，P05→P06 | false | +.220479 | −29.272243 | −23.304527 | +.429982 |
| 180633 / 入口gate | false | −.006863 | −29.590567 | −24.236658 | +.561452 |
| 180708 / P07→P08 | false | +.002384 | −40.886673 | −26.267391 | −1.423072 |
| 180709 / P08→P09 | false | +.003928 | −41.094284 | −26.714457 | −1.371784 |
| 180716 / tick3495 FALL | true | −42.647350 | −42.647350 | −28.365738 | −1.350748 |

普通切换不置done，gamma=.9985、lambda=.99继续计算；本批唯一done是index107的安全终止，bootstrap=false，terminal return=reward，随后新episode从真实prefix接管，不能跨该episode连接回报。入口原始GAE=`return−V=−5.353909`，捕获为−5.967716，但整批原始GAE均值−7.975224，经官方整批标准化后二者成为正advantage。这是相对排序，不是发现了概率接线错误；不能声称未来FALL使此前每个动作都被负向训练。

gate样本实际在minibatch 3/7/11/15/19使用，advantage均+.561452；ratio=.994630/1.019900/1.038177/1.046010/1.053107，均未进入strict clipped branch。最后相对于采样旧策略的logp上升，说明该样本概率总体提高。首minibatch近1检查**没有包含这个gate样本**，其第一次出现时已经过3个optimizer步骤，不能篡称该正例也在未更新权重下ratio=1。

第一episode最终global180716、tick3495/29.125s为FALL；FL placed=2818；RR qualified lift=3421，尚无RR crossed/placed。终止拍FR knee REQUEST=−40.536016°、mapped N24.85°、final−15.686016°。**新kernel没有修好完整任务**；此次入口改善不能解释为后续负动作及全身支撑风险已经解决。本审计不改reward、cap、温度或评价条件。

update1377实际actor参数改变、有限非零梯度，LR1e−5、KL均值.02742449；CP180736累计180736/1377/27540。checkpoint SHA`275b270f50f7c42c478793e5988b58fed4fbc773b2cb1b23c2d86a3b4fdbde9d`；manifest SHA`732d56769e23317ed8e136f824601551bc12ce94686a23e802f31c97299a02c9`，记录save/load roundtrip=true。

新增小型证据：`update1377_actual_request_kernel_audit.json`、`update1377_handoff_reward.json`。只读取前256条已完成决策、前2条update/advantage和封存1377 tensors/hooks；CPU1thread，额外actor前向/随机抽样/优化/仿真全部0，生产未改。

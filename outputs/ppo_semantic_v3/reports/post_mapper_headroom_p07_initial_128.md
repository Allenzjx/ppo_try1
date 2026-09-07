# Post-mapper headroom：P07 首128真实更新（有界，非整块完成）

固定核验范围：global **76417–76544**，PPO update **563**；本轮新增 **128 decisions / 1 update / 20 optimizer steps**。只读截止于已保存 immutable `checkpoint_step_000076544.pt`，不纳入其后的采样或更新。2048是本轮请求预算，不是本报告已完成量。

运行：`runs/ppo_semantic_v3/train/20260906T1928297576378Z_ge99fde1b3e83_639e48bdc9c4499eae7e839939fe0e71`。HEAD `e99fde1b3e8366f0ff1d484140b82877df745f0c`；P07 / offset0 / N1 / seed1001；显式 `NewMdpWarmStart`，不是同MDP exact resume。本报告仅PowerShell读取JSON/JSONL，不加载tensor、不运行Python/CUDA/Isaac、不重复大checkpoint哈希，也未改生产或历史报告。

## 1. 实际initial warmstart收据

源 immutable checkpoint：`checkpoint_step_000076416.pt`，源HEAD `c34262abffc16847ff32d15ecbf790dd60803e0a`。实际initial sidecar为：

`outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000076416_s8273a81e70e0_ge99fde1b3e83_9977a6a445bd6eb42f8fdc384e5bd52ac4eccd226bac44d50d94b401bc8e2226_manifest.json`

其stage为 `initial_v3_warm_start`、`save_load_round_trip=true`，记录checkpoint SHA `a3fee31de93c4ef1678badfcc6ca9945918d4844bd2ce94144bb80e3a1db163e`。此处引用实际运行保存的收据，不宣称本报告重新计算了文件hash或独立torch.load。

源与initial逐字段比较均相等：actor全参数hash、critic hash、normalizer hash、完整JSON的training RNG、policy contract、runner config、global/updates/optimizer计数、stage ledger及new-MDP origin。

| 项目 | 源 → initial实际收据 |
|---|---|
| actor（含learned heteroscedastic distribution参数） | `0f4edbbdec4bd7f38dc43de6c43d10a4eeead80836abbc3d0020fd7caf277494`，相等 |
| critic | `ca5a1825b47a43996efdbdc395ab9cf612432d265442135518fda3574c939ff0`，相等 |
| normalizer | `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`，相等 |
| policy | `heteroscedastic_log_v1` / HeteroscedasticGaussianDistribution / learned log-std / 324 obs、12 raw action，完整contract相等 |
| 累计计数 | 76416 / 562 PPO updates / 11240 optimizer steps，initial零加计数 |
| requested stage ledger | full_episode29184 / phase_suffix37120 / smoke0，initial不消费 |
| new-MDP origin | 10112，未另起或抹除累计账本 |
| Adam | `fa500e1e9652179107d05bd2e37dd22901c13090be222d8ae68b66747cfd84e3` → `ebddf13dce59639efaf4e3951e4a4db11f59a2b1e7a9f70fe1168320ba716614`；源记录LR1e−5 → initial3e−5 |

本次warmstart record明确 `reset_all_moments`、`old_rollout_buffer_inherited=false`、`physical_state_inherited=false`，恢复已验证training RNG后重新采样；initial `physical_env_state_saved=false`，topology为N1、fresh128 decisions/env、无peer reset、task timeout不bootstrap。这里区分“实际保存的迁移声明及state hashes”与本报告未做的独立tensor内部检查。

`implemented_reset_sampling`及topology均为 **`natural_P01_A_teacher_prefix_then_semantic_suffix_N1.v2:P07:offset_0`**，不是源课程描述残留。initial action comparison在实际P07/t5952，以零projector history比较源/新profile；六组12维逻辑输出差全0，`actual_environment_or_bridge_history_modified=false`。它只证明该接管状态下的逻辑比较，不是全轨迹或native等价证明。

契约差异：源/目标29项frozen-A文件hash记录无差异；action schema、observation schema、reward config、task spec、quality score均未变。执行profile改变，另声明8个PPO执行/审计/接线文件变更（action_projection、actuator_target_effect、isaac_fsm_backend、semantic_backend、semantic_headroom、semantic_residual_adapter、semantic_vector_backend、semantic_video）。本报告未重复读取29个文件做hash。目标仅版本化post-mapper servo margin；CPU/GPU audit优化候选未接线。

## 2. 首个真实teacher→P07接管

首prefix实际 **744 decisions / 5952 physics ticks**，所有prefix决策 `policy_credit=false`、raw及projected residual全零、5952 verified native ticks、无episode-state写入。`policy_credit_start`本身也是非PPO样本；截至本固定128窗口只使用这一个prefix，无fallback。

实际credit开始：**P07，t5952，49.6s**，`requested_phase_still_active_at_credit=true`、`from_P01_current_policy=false`，并非直接写入历史姿态或已抬起RR的snapshot。接管receipt source control tick5951、command dispatch tick6131（控制时钟与含180个基础tick的dispatch时钟分开记录），bias12项全0、每次articulation write=1；保存的是当前真实mapper/最终drive，非新置零的mapper历史。

实际start raw/contact与共同物理evaluator证据：

| 当前腿 | 接触与真实载荷 | front / clearance | 硬Q/C/P |
|---|---|---|---|
| RR | GROUND，ground pair active/verified，normal7.094826N，obstacle inactive；load fraction0.247819573 | −205.028680mm / −50.919979mm | 全false |
| RL | GROUND，ground pair active/verified，normal8.689030N，obstacle inactive；load fraction0.303504492 | −219.366219mm / −50.144145mm | 全false |

教师形成的前腿历史：FR Q71/C1665/P1695，FL Q2461/C3115/P3583；这些事件在PPO接管前，不是当前策略新增成果。RL teacher history中有早先initial-clearance事件但未硬Q/C/P，不能据历史initial字段说接管时AIR。P07/P08准备仍进入真实策略样本，未把suffix标成完整P01训练或full-task success。

## 3. 首128优化窗口、实际物理执行与新headroom

完整逐行global连续，无旧run/重复global混入：

| 源phase | 已优化decisions |
|---|---:|
| P07 | 1 |
| P08 | 1 |
| P09 | 126 |
| 合计 | 128 |

全窗口 **1024 ticks**，无短tick；online native summary为1024 verified、1024 actual-native-effect、1022 own-phase-request effect（两处phase过渡不据此归到新phase本次own request）。128行均明确 `prefix_teacher_data_in_ppo_storage=false`。四项in-episode root-pose/root-velocity/force-or-impulse/gravity写入最大值均0；无terminal，本窗口是首回合未终结的128决策段。

审计粒度区分：1024tick完整验证计数来自每tick在线审计汇总；本报告独立PowerShell公式重算的是JSONL保留的 **128个决策末完整headroom/native记录**，没有将其冒充1024份完整数学离线重算。

全部128决策末记录存在 `same_tick_post_mapper_servo_margin_v1`，原request与projection的逐通道差最大0。以各记录实际geometry-corrected native+c和原hip/knee2°reserve，重新计算servo区间及effective residual，最大误差0；wheel residual不变。沿同一记录的previous-final、原hard limits和1.25°slew，再用接管receipt所绑定standing转换及原servo/wheel轴符号、float32 cast重建，决策末actual native readback最大差0。在线记录的setter==dispatch、expected==actual、same-tick均true。

78/128决策末在新真实headroom处裁剪，均为canonical index7（RR knee），不是“取消所有裁剪”。例如g76434/t6096：nominal=native=−35.1°、c0、request−26.626177°，区间下界−22.9°，effective−22.9°。最终canonical drive为−57.726177°，不等于静态candidate−58°，因为实际final slew历史仍生效。

也有真实旧余量误裁现在未裁的例子：**g76437/t6120**，nominal−37.8°、真实native−36.55°、c0、request−21.205714°；旧unmapped-nominal区间会截成−20.2°，新effective保留−21.205714°，实际canonical drive−57.755714°。这是同一实际状态的算术反事实，不是重放旧物理轨迹，也不证明此改动导致任务进展。

geometry证据在104个决策末出现：identity61、projected_exact_forward31、projected_relaxed_forward10、degraded_bypass_infeasible_box_downward2；其余24末tick无geometry context。PPO effect仍相对于同geometry的zero-current-policy，geometry另与raw-nominal-zero分账；没有改成以未投影nominal计算PPO effect。线性preview/target重建不是实际净空保证。

## 4. 真实optimizer与保存边界

`optimizer_updates.jsonl`的首完整update563记录：

- 20 optimizer steps；actor before与源/initial的`0f4edbbd…`精确相等，after为`8972e907d96e5c0d30924eb2c0b477499c5f8039c5886f8be828c3129411d5d9`，`actor_parameters_changed=true`。
- `finite_nonzero_gradient_observed=true`，gradient norm1.007971358–1.414213245；KL0.02228605766，clip fraction0.2703125，entropy−3.752481234，value loss0.01032084497，surrogate−0.01751874499。
- update结束LR1e−5；initial为3e−5，经原adaptive机制变化。没有逐minibatch LR轨迹，不能断言20步始终同一LR。
- 128行Gaussian raw/mean/std均为12维有限数据、std>0，std范围0.0949091613–0.2482156903。以已记录raw/old-mean/old-std重算12维Gaussian log-prob，最大差1.21891549e−6（double离线重算相对记录float32计算）；未用filtered/effective动作代替PPO raw sample。

已保存step76544 sidecar：**76544 / 563 PPO updates / 11260 optimizer steps**，`save_load_round_trip=true`；记录checkpoint SHA **`2e68520e004f255bafe8c62ba5838fac699e8fe7f45b8bcee678118ee801b5aa`**。actor after与update记录相等，normalizer仍源hash；预算ledger为full29184 / suffix37248 / smoke0，正好新增128个suffix请求。本报告不重复hash验证、不把运行中last pointer可能后移当作此窗口的统计边界。

## 5. 截止窗口的任务状态（不预判终局）

g76544末为P09/t6976/58.133333s，physical evaluator valid=true，termination=null，task_success=false。PPO信用段RR出现initial-clearance t6007及t6349、硬qualified t6405；截至窗口没有RR cross/place，RL无新增硬Q/C/P。RR当前AIR、load0，front−119.539675mm、clearance−6.379976mm，虽历史active_lift=true，不能称已越沿/放置或当前净空合格。RL仍GROUND，load0.541831253，front−326.331061mm、clearance−51.856646mm。FL当前AIR/load0、在台面XY上方，不能把教师placed历史当当前承载。

因此本报告只确认：新版本真实initial迁移、grounded P07前驱接管、首个官方PPO更新/保存和headroom原生下发审计已运行。128个已优化样本全部来自一个尚未终结的后缀回合；既不是2048训练完成，也不是自然P01完整任务成功，更不证明sigma或headroom带来的因果收益。固定证据读取及报告写入至此停止。

# Independent read-only review — unwired native-audit candidate

结论：**未发现相对当前生产版本新增的、可由源码确定的CPU原型阻断问题或fail-closed检查丢失。** 这不是CUDA真实等价批准，也不是提速结论。候选仍未接线；当前训练生产代码保持不变。

审查范围为README、完整candidate与test源码、已有JUnit，以及生产`actuator_target_effect.py`和其实际dispatch调用位置。仅PowerShell读取和本报告写入；未重复161项CPU测试，未运行Python/CUDA/Isaac。已读取XML为161 tests、0 failure/error/skip、2.639s；该时间不是性能benchmark。生产audit小文件SHA实读仍为`7ec65a2e425681d2e172caa409ae0735983d451bdf7346fa0cecba63cb268721`，与候选测试pin一致。

## 已保留的关键语义

| 审查点 | 静态核验 |
|---|---|
| 同tick状态 | 候选没有新的mapper/controller/sensor调用、apply、setter、write或sim.step；previous-final、raw native、controller/combined bias都来自当次传参，没有模块级缓存。 |
| 真实调用时点 | 生产`isaac_fsm_backend.py:1932–1977`先记录t−1 actual final，执行唯一`_atomic_apply`及其既有`write_data_to_sim`，再做audit，最后`sim.step`。候选未来若接线必须留在此同一位置；当前未改调用者。 |
| 四个target绑定 | candidate97–125仍检查Tensor、rank2、N1、float32、canonical IDs数量/唯一性并实际索引；四个选中snapshot均做finite检查，staged servo/wheel必须等于dispatch servo/wheel。 |
| expected重建 | 重用同一frozen bounded slew/hard limits/standing offset/sign/radian conversion；expected with-residual float32目标必须等于实际dispatch，不用逻辑double动作冒充native readback。 |
| PPO counterfactual | with-r与no-current-r共享同一corrected nominal、controller correction和t−1实际final，changed仍为float32 numeric inequality，不依据raw是否非零。 |
| geometry第三分支 | candidate187–203额外重建raw nominal+controller、zero-current-policy分支。PPO delta仍actual−geometry-zero；geometry delta仍geometry-zero−raw-zero，未将g归给policy。现有完整字段、delta一致性、单rear-pair和Mapping检查均保留。 |
| ±0与量化 | 使用`!=`而非bitwise effect；+0/−0不算作用。记录实际target来自snapshot，CPU测试以JSON字符串验证符号。极小servo residual量化成同一float32仍不算非零作用。 |
| 重入/新tick | 每次独立4次selected clone，无复用上一tick的target或history。已有successive-dispatch测试也覆盖这一点。 |

不能把“一次packed `.cpu()`”称为四个buffer的原子快照。它们依旧是四次顺序selection/clone，只是在单一owner、无中途target写入的前提下属于同一dispatch；这与原始调用者的约束一致。

## 有意差异：仍fail-closed，但不是完整错误行为恒等

1. 候选先验证四份结构再一次transfer，然后按原次序做finite检查。因此“前一个NaN+后一个错误dtype”等多故障情况下报错优先级可变。README和test320明确记录，两个版本都拒绝，未看到绕过。
2. candidate111–112新增同device要求。它比原来明确，不会接受混device输入；但CPU/GPU混合、不同CUDA设备和不支持设备上的错误类型/时机没有实测。不要宣称这些失败路径已逐字等价。
3. 审计本身并没有新增geometry context tick/列校验；它保留原有ACK审计范围。真实context与本次dispatch tick、具体rear pair的绑定仍由生产`semantic_residual_adapter.py`在advance之前/之后完成。候选没有去掉这些上游检查，也不能被描述为独立重新验证了完整kinematic context。

## 未解决的GPU等价范围（非已证实故障）

**数值执行位置已改变。** candidate127–139及170、187–199将expected/counterfactual float32构造、比较及两个delta减法放到CPU，而生产在目标device上执行。CPU161项全字典/JSON相等能证实这些固定CPU输入；不能推导实际CUDA返回所有字段相同。

后续若选择进一步验证，已有计划的最小数值样本应同时包含：

- 当前真实普通/geometry三分支target，以及±0、tiny residual量化成0；
- subnormal目标本身，和**两个正常非零float32目标相减得到subnormal delta**的邻近边界；后者不能只靠“subnormal vs zero”fixture覆盖；
- 当前CPU运行环境的denormal/FTZ/DAZ行为、CUDA cast/比较/减法结果。仅Torch dtype字符串为float32不能约束这些执行细节。

这只是对已有GPU未验证清单的精确补充，不建议本轮新增测试框架、放宽字段或以近似`allclose`取代当前equal/JSON契约。出现差异时应先区分actual/counterfactual值、changed flags与delta记录是哪一项变化，不能只比较最终count。

**copy完成不等于自动建立外部producer stream依赖。** candidate116的blocking `.cpu()`可等待本copy完成，但如果PhysX/其他CUDA stream在未建立依赖时仍可能改源buffer，它不会凭common-device自行证明读到正确dispatch时点。当前CPU测试的`.cpu()` spy也只数Python调用，不数DMA或GPU kernel。后续真实设备比较必须维持原单owner、dispatch→snapshot→physics顺序，并确认实际stream可见性；不要增后台copy、nonblocking或跨tick缓存后仍沿用本审查。

## 测试与文档评价

现有测试使用真实frozen adapter/mapper/setter cast和真实geometry helper（synthetic J），同时比较完整dict与exact JSON；positive审计期间禁止advance/apply/set/write/update/额外joint read，before/after检查buffer bytes、mapper/ACK/request、写入计数。负例覆盖4targets、IDs、expected/staged/dispatch、bias/slew/phase、geometry证据。范围与README基本相符。

未发现需要修改当前原型源码的确定性finding。README已经正确声明CPU-only、stream/FTZ风险和未测总成本；上述“正常数相减得到subnormal”与“blocking copy不替代producer依赖”可作为后续独占GPU比较的具体注意点，不是本次生产变更许可。

最终状态：仅审查输出，**no blocking finding for the unwired CPU prototype；CUDA真实等价与端到端速度均未验证**。不承诺节省PCIe事务、不承诺提速，不改迁移或当前训练路径；本报告完成后停止。

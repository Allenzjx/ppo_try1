# RR 当前掉载恢复 v2：待应用草稿

状态：**仅草稿，未应用生产、未运行测试/模型/Isaac**。7个修改hunk已对当前生产源码逐一确认唯一精确上下文，替换后源码和新增测试在内存中AST解析通过；这不等于行为测试通过。

应用顺序：`rr_recapture_current_support_v2.production.patch`，然后`rr_recapture_current_support_v2.tests.patch`。文件使用本工程编辑工具的 `*** Begin Patch` 格式，可将内容直接交给 `apply_patch`，不是git unified-diff格式。配置及checkpoint迁移由root/迁移agent单独处理。

生产范围仅3文件：

- `semantic_rear_policy_timing.py`：新增`RECAPTURE_MODE = "rr_recapture_current_support_v2"`和`MODES`；`public_timing(..., mode=MODE)`默认旧v1。
- `semantic_supervisor.py`：新mode许可、provider显式传mode；仅RR历史placed后的当前保持份额新增opt-in分支。
- `semantic_backend.py`：profile接受v1/v2，保留profile/task必须一致、后腿capture/geometry/wheel projection必须OFF。

## 行为差异

新v2下，已经开始late、已经RR placed或处于P10–P12时，如果RR缺乏真实当前承载、RL尚非有效当前AIR且未placed，则公共九字段重新选择`rr_carry_capture=true`、`rl_prep_transfer=false`。这是任务请求，不修改`current_lift_valid`、当前接触或历史资格。来源时钟、owner、源派发、P09 late/P12 dependency规则完全不动；不添加接触等待/载荷阈值。

RR已placed但RL尚未有效AIR/placed时，保持原`.8`历史信用；仅原`.2`retention改为既有捕获公式：`.5 * bounded_XY_gap_proximity + .5 * current_contact_fraction`。几何部分要求当前合法XY/侧向、非ground，且是无障碍接触的AIR或严格TOP；低于既有signed gap下界不给该项。contact部分必须经当前严格TOP、非AIR、合法XY、实测support、已验证有限force且达到原noise floor，并复用原minimum_top_samples。AIR即使几何gap零也最多获得一半retention；高悬空按既有gap尺度递减，不能为1。

当前RL已有效AIR或RL已placed时恢复旧RR retention，保留合法动态后续任务。FR/FL/RL自身retention、新/旧placed事件、任务验收、真实安全中止都不改。没有第二次placement事件或额外reward系数。

## 训练/模型约束

同一个419维模型、同一个九字段、同一个Gaussian/HISTORY/sigma kernel，actor/profile合同字节不改。状态门值改变会按原kernel自然改变局部sigma许可，不添加新的采样或deterministic动作路径。不新增运行时关节目标、capture assist、固定FR/RR姿态、轮速整形。

这是观测/奖励语义变化；必须由root在完整update边界封存最新模型后明确迁移，保留兼容权重/optimizer/normalizer，清空未完成rollout并重新采样。草稿不猜最新CP、不回退、不把reward变更当作模型已经学会恢复。

## 待运行定向测试

新增测试文件覆盖：v1默认与早期行为不变；P09late/P10/P11/P12掉载恢复可见但时钟/输入历史不变；真实bearing、合法RL AIR、RL placed的继续；AIR零/低/高gap的有界保持梯度与原`.8/.2`预算；force零/NaN/未验证、XY/ground/wall/下界负例；provider传mode、profile保留rearOFF。没有运行这些测试；根任务在安全保存边界应用后再执行。

补充最小跨模块正反例（仍仅草稿、未运行）：同一组合fixture在P09 late启动后丢失RR承载，检查公共carry/retention与P12独立RL游标一致；P09轮stop仍只发一次、不重播late，重新取得RR承载后RL游标逐拍恢复而不追赶。对照为已经合法在途RL AIR：继续其RL动作和既有RR保持语义。P09 stop按该源实际事件计数，不能因后来P12合法wheel owner仍在输出而误称它未发stop。

下一真实块建议验证“掉载→再捕获→接续保持”，不宣称本修订已解决整个RR→RL依赖。暂不追加笼统post-touch P09/P10暂停：P09 late在5.4秒只有一次joint/full12目标，之后7.2秒是wheel stop；停时钟会保留已发关节目标且可能延长反转。P10仅含RR knee节点，尚无充分证据将它全判为危险RL卸载；强制先承载才允许它可能阻止有用捕获/加载。保留P11接收准备及P12当前支撑门，若真实保持仍失败，再做有明确归因的小对照，不添加固定载荷/静止时长门或隐藏后腿目标保持器。

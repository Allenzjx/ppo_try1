# CP222720 v10：P05 源结束后的正净空机会被年龄门错过

只读小窗口与静态分析；唯一录像仍活跃时未改生产、未测试/仿真、未更新模型。源：`video_eval/validation/20260923T1241379571238Z_g99dff5fd366e_d4c5e1bc1e8a4bfeb2b946c7f12630dd/source`。使用固定偏移随机 seek 定位 JSONL 后读取29.2667–32.4秒48条decision及该小段22条重叠相邻记录，并读取几条实时尾行；不是全历史扫描。所有动作值来自实际 journal，未离线重算 N。

## 先后顺序与归因

| episode t / tick | 实际记录 |
|---|---|
| 29.33–30.40 s | N 四轮均+.3；FINAL FL约−.465、FR+.317、RL+.344、RR+.139 rad/s。FL是policy抵消后反向，不是mask遗漏；FL当时AIR不计地面牵引。 |
| 30.600 / 3672 | P05 source_tick1088有四轮新事件；该decision N仍记录+.3。30.6667 /3680下一decision N已全0。不能把decision行当作该源事件精确actuator提交时刻。 |
| 30.733–31.133 s | 源FL knee−36.7→−13.4、hip48.2→45→24.9。前缘距离短暂到−0.271mm（31.0667），但未真正非负/cross；随后下降、退回。assist全程WAIT，并未HOLD接管。 |
| 31.2667 /3752 | 首个所读decision的source endpoint=true、prior=false；gap+15.248mm、front−18.589mm，年龄/先前endpoint两项拒绝。 |
| 31.3333 /3760 | endpoint/prior均true、无新wheel owner、三条其他实际支撑、FL合格AIR且gap+8.953mm/front−24.169mm。**唯一拒绝项是age9.8不在[30,40)**。 |
| 31.47–32.4 s | 同样只被年龄门拒绝，仍有约2–4mm正gap、source已完整结束。 |
| 40.600 /4872 | 所读尾行仍正gap+.1756mm，front−29.216mm；N全0，FINAL约[−.748,+.0139,+.0358,−.1485]。这是本次最后读到的正值，不声称全轨迹最后正gap。 |
| 41.9333 /5032 | age20.4、gap−.8805mm；年龄和正gap两项都拒绝。43.2667尾行仍如此，mask四轮全1、N全0、FINAL等于经投影residual，独立dispatch audit通过。 |

第一处分歧顺序是**有限源四轮stop，然后源关节展开/hip下调在真实cross前执行**；并非辅助器误HOLD。源完整结束后本来另有state-feedback恢复，但它被绑定到任务最大年龄30秒才开始；此回合正净空窗口提前出现并丧失。现有capture-pending/FL assist要求真实cross和合法XY，当前cross=false，正确没有虚构捕获入口。v2虽名为recross，也允许first_approach；失败不是仅缺少历史cross造成。

## 唯一最小候选（staged，待主代理审查）

新显式模式 `p05_completed_source_first_approach_recross_v3`：**仅把现有恢复年龄下界30改为0**，仍要求完整source已发出后的相邻tick、无新wheel owner、合格AIR/正gap/合法横向与前缘区域/未placed/实际其他支撑，并保留v2的first-approach与same-uninterrupted-AIR-recross区分。绝对结束仍30+10=40；不改timeout、源下降轨迹、轮速+.3幅值、actor/reward/physics、不提前consume源动作，不覆盖stop当拍。v1/v2旧模式仍有原[30,40)窗口。

这会针对31.3333之后已证的有效执行机会，不声称修好30.6的stop或31.1以前的下降顺序；亦不保证+.3能抵消所有policy影响。早期FL反向仍可能存在；当前支持轮FR/RL/RR恢复原+.3建议后实际是否推进、净空是否保持、是否形成真实TOP，必须下一真实同版本验证。若已经到非正gap，该候选仍不会强推立面，不能叫无限wheel-only恢复。

`production.patch`只涉及supervisor注册/age下界与当前RR实验spec选择；profile/migration由独立候选处理。`test_semantic_p05_completed_source_recovery_v3.py`为未运行草稿，覆盖新窗口、旧v1/v2、完整endpoint/相邻tick、stop优先、正gap/支撑、同AIR跨度、绝对end及仅wheel建议变化；`legacy_v2_test_compat.patch`只让旧v2套件显式选历史模式，不删除其反例。全局200秒与真实安全判据未改。

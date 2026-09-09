# B1：已命中现行 wheel-only 条件，未发现漏记全身抬升

来源：`prior_B/20260908T0800275970113Z_gd7479d9fc41c_e857a8e4844d4ea984ff8f4e385d3d19`，HEAD `d7479d9fc41cacd98740a10fb47c2fee64fd74b7`。无 checkpoint、自然 P01、seed2001、零残差；758决策/6057ticks/50.475s，P09 `WHEEL_ONLY_CLIMB`，物理记录 valid=true，实际 task=false、optimizer=0、非外部截窗。

## 决定性事件

| tick | RR实测状态 | front / gap（mm） | 判定意义 |
|---|---|---|---|
| 5320 / 5328 / 5336 | 均GROUND | front −207.519 / −206.702 / −205.279 | P06→P07→P08→P09，各前驱过渡相隔8tick，并非已有RR硬资格 |
| 5596 | exact obstacle active，3.523N | −49.304 / −50.579 | 首次该窗口障碍接触，不据接触点单独推断面/因果 |
| 5993 | AIR，initial被接受 | −17.618 / −5.236 | 向上3.824mm；own运动仅1.683°，但全身19.499°、命令67.495°已被认可 |
| 6016 | 该窗口最高AIR | −6.339 / −0.886 | 仍未满足bottom≥top；不是要求再高15mm才可取得硬Q |
| 6019 | 再次obstacle active，14.918N | −5.460 / −1.365 | 5992–6018连续27个AIR tick结束 |
| 6057 | 首次center越沿，真实TOP | +0.264 / −0.867 | exact pair13.758N、load0.503990，当前crossing geometry成立，但历史Q缺失 |

完整末态RR事件史只有5993 initial，没有 qualified、GROUND-revoked、cross或placed。限定5320–6057的738行原始窗口中，118行AIR、185行GROUND、435行obstacle；AIR bottom≥top为0行。5993后没有GROUND。因此是**从未取得资格**，不是已取得后又落地撤销。

`TaskEvaluator`已使用全身发力/运动证据；硬Q还要求连续AIR≥2、窗口抬升≥8mm、bottom≥top及现有前缘范围。终止时own12.735°、全身91.964°均不是被忽略的零运动。初始离地和Q不是同一事件。首次越沿已有当前合法接触几何，失败条件是Q=false。`consecutive_top_samples=1`统计的是center≥front后的loaded tick，不是说此前没有障碍接触；尚无合法cross，更不能把当前支撑当placed。

FL末态真实TOP/support/load0.496010；body collision=false。末个1tick的native verified，raw/projected全12零、无四类state writes；未重复全6057tick native审计。所查738行geometry/contact pair及finite全部有效。未发现需要改判或修传感链的证据。

范围：final manifest、末raw/audit及一次固定P07–P09原始窗口；不改生产/规则，不运行Python/Torch/仿真。一次transition输出过大被工具截断，结论使用独立白名单字段与固定窗口。近零净空不等于已证明测量误差；contact point可能在active=false时存在，不作为接触成立依据。`wheel-only`是当前任务标准命中，**不证明轮驱动是唯一物理原因**，也不是A/C优越性或优化器门禁。

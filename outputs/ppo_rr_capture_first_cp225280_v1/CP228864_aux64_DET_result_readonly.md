# CP228864 AUX64 确定性评估：前缀保留，RR膝脱离裁剪但仍未捕获

已封存：10940tick / 91.166667s / 1368帧，error=null；INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED，local/full success均false。标签为DET、finite RR AUX training lineage、rear helpers OFF（原FL hip-only assist ON）。本次新增PPO/AUX/训练credit全部0。

前缀998请求严格local0、一次HISTORY、无local forward；与接受的CP225280 anchor对齐998行，raw/FINAL最大差均0、missing0。FR lift24/cross2751/place2761；FL lift2774/cross4269/place5165；RR lift6995/cross7979，全部同tick。

## 本次真实RR动作（不是旧“370拍都−58”）

激活gate7979的FINAL=[+7.485,−58.000]°；首gate可见7984 actual=[+7.223,−57.760]°。以下范围仅370个真正active-policy决策端点，不包含前缀最后一拍。

| RR | FINAL范围 deg | actual范围 deg | FINAL相对gate范围 deg | actual相对7984范围 deg |
|---|---|---|---|---|
| hip | [+5.366, +7.285] | [+5.108, +7.127] | [-2.120, -0.200] | [-2.115, -0.096] |
| knee | [-58.000, -56.699] | [-57.765, -56.470] | [+0.000, +1.301] | [-0.004, +1.290] |

仅4/370个FINAL knee仍等于−58°；首次脱离为8024/66.866667s（−57.924482°），约67.53s已到−57.25°附近。展开确实执行，但整个FINAL膝变化范围仅约1.30°，不能把脱离裁剪等同已完成下降。

| tick / s | RR FINAL / actual deg | gap mm | raw conditional mean hip,knee | local raw head hip,knee |
|---|---|---|---|---|
| 7992 / 66.600000 | [+7.285, -58.000] / [+7.127, -57.760] | 58.956159 | [+0.764, -0.602] | [-0.145, +0.061] |
| 8024 / 66.866667 | [+6.653, -57.924] / [+6.544, -57.729] | 58.034436 | [+0.720, -0.582] | [-0.139, +0.066] |
| 8104 / 67.533333 | [+5.909, -57.252] / [+5.714, -57.066] | 57.579437 | [+0.671, -0.557] | [-0.133, +0.070] |
| 8224 / 68.533333 | [+5.474, -56.935] / [+5.219, -56.711] | 55.230760 | [+0.644, -0.545] | [-0.127, +0.076] |
| 10940 / 91.166667 | [+5.804, -56.742] / [+5.540, -56.518] | 57.408820 | [+0.665, -0.538] | [-0.109, +0.071] |

active最小gap=55.230760mm（8224/68.533333s），末57.408820mm。source RR=[−6.9,−37.8]、mappedN=[−8.15,−39.05]全程恒定，generic RR bias=0、actual tracking列表空，P09 late保持pending。实际均值/动作已有变化，但未把轮端从高空推进到TOP；不单轴归因。

## 真实TOP与停止/FL反转

仅对封存capture_assist_ticks做一次完整布尔扫描：10940行、tick1…10940连续，RR top_surface_contact=true为0，字段缺失0。gate7979起2962个native tick也无GROUND。因此不是仅凭15Hz漏看TOP；本次确无记录TOP，hold=0。末资格false伴随任务live=false，不解释成GROUND撤资格。

P09 source864 stop首次见于8200/68.333s，但同拍合成N四轮仍[+.3,+.3,+.3,+.3]；8920/74.333s合成N才全0，不能把source stop与最终合成stop混用。FL在370个active端点FINAL和实测canonical速度均负；尾N四轮0，FINAL=[−1.005577,−.038372,+.039282,−.059130]，实测=[−1.060134,−.037648,+.060978,−.059827] rad/s（FL,FR,RL,RR）。这是策略残差持续反向，不是缺失nominal或mask证据。

第一未完成任务：RR真实顶部捕获及可用承载0.5s。不是动作角度仍完全锁死，也不是phase标签或无碰撞就成功。新视频由主任务导出，本报告不替代目视QA或声明全程成功。

详细来源：`CP228864_aux64_DET_sealed_actualmetrics.json`（既有分析器）与 `CP228864_aux64_DET_result_readonly.json`（本次范围/计数）。

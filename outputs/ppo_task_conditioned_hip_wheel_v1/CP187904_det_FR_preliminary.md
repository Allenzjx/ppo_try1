# CP187904 deterministic：FR 已完成窗口预览

仅比较真实 P01→FR 捕获窗口；本轮视频仍可能在运行，未等待或伪造最终 manifest。候选为自然 P01、实际 `deterministic_conditional_mean`，FR lift/cross/placed 为 tick 23/1607/1620。

|指标|本轮 CP187904 det，0→1620|旧成功 B，0→1502|变化|
|---|---:|---:|---:|
|FR 捕获时间|13.500 s|12.5167 s|+0.9833 s|
|roll/pitch rate RMS|0.104231 rad/s|0.117889 rad/s|−11.586%|
|峰值合成倾角|0.290952 rad|0.309464 rad|−5.982%|
|机身 collider 最低世界 z|92.512 mm|91.272 mm|+1.241 mm|
|body/障碍 AABB 保守分离下界最小值|227.336 mm|217.320 mm|+10.016 mm|
|首次 AIR 且 FR gap≥15 mm|tick 65|tick 66|−1 tick|
|此后至越沿前 FR gap 均值|96.127 mm|98.762 mm|−2.636 mm|
|此后至越沿前 FR gap 最小值|15.334 mm|15.330 mm|+0.004 mm|

初步表现是倾角/角速率更小、捕获更慢，FR 空中平均净空略低；不能把时间代价隐去后声称全面更优。净空最小值受“首次达到 15 mm”窗口起点影响，不是独立的裕量改善证据。

RMS 由完整 120 Hz 物理记录的 wrapped 相邻欧拉 roll/pitch 导数求时间均方根，**没有再除时长**。最低 collider z 不是 base z，AABB 下界不是精确 mesh 距离。全窗口 FR gap 从初始地面约 −50.84 mm 到最终接触，不能把初始负值误叫摆腿失败。

旧 B 属不同保存版本，仅是明确标注的 preliminary 参考；新同版本 B 尚未运行，本表不是最终配对统计，也不说明 P05 或整次越障成功。原始指标见 `CP187904_det_FR_preliminary.json`，有限读取复现脚本为 `video_fr_preview_readonly.py`。

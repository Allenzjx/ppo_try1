# 独立跨 HEAD、N 控制路径未变的比较入口

入口为 `cross_head_unchanged_n_pair.py`；原 `paired_event_media.py`、strict pair函数、旧receipt均不改。仅接受已封存并已导出review的 B f2e5524 / deterministic C 3a50657，其他HEAD组合拒绝。当前仅完成help与只读代码范围自检，尚未读取活动C、渲染新比较或声称其结果。

它复用原单源完整性校验和独立event quality重算，但不伪造same-version条件。新receipt schema为 `wlr50_clean.explicit_cross_HEAD_unchanged_N_media_pair.v1`，写明实际两个HEAD/runtime、精确六文件差异及hash、六配置逐字相同、scene锁定配置及构建/控制代码相同、camera/seed/逻辑P01入口相同。B无CP提前返回、ZERO12实际路径及共享helper未改有对应源码检查。**B未在3a重跑**；不声称bitwise实测初态、轨迹等价或未追踪外部asset的独立重测。

输出完整episode比较及P01→两侧各自真实FR捕获的相同事件窗口。1×、同P01物理时间原点，不按阶段伸缩、不删内部停顿；短侧结束后定格并持续标“NO PHYSICS / METRIC CREDIT”。质量指标来自各原始source的真实事件窗口，不计算定格帧。片头（第1帧）和全程两面板均标CROSS-HEAD、unchanged N/control与B NOT re-run，不增加伪物理片头时长。若FR捕获事件缺失，当前入口按原事件质量有效性拒绝，而不制造完整FR比较。

待C封存且review/quality已有真实路径后运行：

```powershell
python outputs/ppo_fl_capture_quality_v1/cross_head_unchanged_n_pair.py `
  --b-receipt outputs/ppo_fl_capture_quality_v1/B_control_review/B_control_Nplus0.media.json `
  --c-receipt <已封存CP182528的review-media.json> `
  --b-quality outputs/ppo_fl_capture_quality_v1/B_control_event_quality.json `
  --c-quality <同一C源的event-quality.json> `
  --destination outputs/ppo_fl_capture_quality_v1/CP182528_cross_HEAD_unchanged_N_pair
```

输出只可进入本outputs根下新目录（不可覆盖）；receipt名明确为 `cross_HEAD_pair_receipt.json`，不会写 `strict_same_version_contract`。单次描述性差异不是鲁棒性／优越性结论，不以质量推断任务成功。实际生成后仍需查看解码preview确认双面板文字与腿名清晰；此说明不是媒体已通过QA。

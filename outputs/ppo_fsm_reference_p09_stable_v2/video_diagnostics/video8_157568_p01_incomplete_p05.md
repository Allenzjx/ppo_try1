# Video8：P05未完成的无损诊断副本

执行完成：2026-09-10 21:18:40Z（receipt为等价−04:00时间），Python helper及ffmpeg remux均exit0。仅处理run `20260910T2100033647207Z_gd4e46006b382_e98a9290c0f44825aa705e61f7a57377`，未重验旧录像。

结果仍为主控已核验的 **P05 / INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED**，不是成功视频，也不是新增物理安全失败。媒体helper从原source manifest再次核对checkpoint157568、722 issued/722 returned环境step、5772实际ticks/48.1s；physical evaluator自身termination_reason=null、task_success=false。721条完整8ticks加末4ticks；末条是返回的环境step，不能沿用video7的中断末步说明。此次未重扫policy/physical大流来重做任务诊断。

原MP4为68152245 bytes，SHA256 `2e876123d1e9b22d29dc90b415b7935a931bdcf4a9efdaa0f9b8a4a18308fe3f`；15fps、722帧全decode PASS，但container duration=286331153s错误。已用`-c:v copy`独立remux，不覆盖原件、不裁剪/拼接/倍速/插帧/重编码。

输出 `video8_157568_p01_incomplete_p05.mp4`：68069277 bytes，SHA256 `74aa9cecd5ec5e28e0edf5c846425c476d6dad91eabe2f701cbbe8426044d6e4`。严格容器校验PASS，ffmpeg显示48.13s；722/15精确媒体时长48.133333s，源与副本均如此。比物理48.1s多的1/30s是原生末4tick图像显示量化，不是增加物理时间或帧。

验证范围：全部722个encoded payload SHA256、精确有理数PTS/DTS/duration、packet关键flag（25关键帧），以及完整decoded frame checksum/绝对PTS/PTS差分序列逐项相同。原timebase1/15→副本1/15360仅改变刻度表示，秒数不变。两者均1280×720/H.264/yuv420p、unique722、black-like0、native15fps连续单调；首末PTS=0/48.066667s。ordered normalized packet记录SHA256=`8fdd14308b0cc094efa04739b4bef6f237c595516259c799a5e3cddef786c473`。

原视频hash前后实测一致；原source manifest与capture manifest的bytes前后一致，SHA256分别`de0a38afc54d2ca1b1a635c7e063a5fab6b97a915761cc9e7fc590026e643c54`、`2225116c098c6613e0249c6955970e696f5dfc61d7cafdaa26f82859cb574cea`。未调用success publisher，不修改生产/模型/主ledger，不干预并行训练。

同目录交付：

- `video8_157568_p01_incomplete_p05.remux_receipt.json`：完整命令、原/副本证据与hash。
- `video8_157568_decoded_01.png`：源等价frame0 / PTS0。
- `video8_157568_decoded_02.png`：源等价frame721 / PTS48.066667。
- `remux_video8_diagnostic.py`：video7 helper的最小输出层副本，旧helper不变；nonoverwrite保留。

PNG结构检查通过，均为真实解码原帧等价图像，未加标注；主控随后亲自视觉核验。媒体完整不等于任务成功或稳定性改善，媒体处理不是optimizer启动门禁。

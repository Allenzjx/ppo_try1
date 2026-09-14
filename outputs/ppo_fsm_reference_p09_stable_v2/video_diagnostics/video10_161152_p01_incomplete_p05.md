# Video10：P05未完成的无损诊断副本

2026-09-10 23:46:51Z完成（receipt为等价−04:00时间）。仅处理`20260910T2329210669061Z_gd4e46006b382_b1afaf16892b450b81d986e7148862d0/source`；Python helper和ffmpeg remux实际exit0，CPU媒体子进程串行执行，未启动第二Isaac或导入Torch。

任务仍为主控已核验的 **P05 / INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED**，没有完整越障成功，不把DIAGNOSTIC_FAILURE等同视频I/O失败。helper从source manifest再次核对实际checkpoint161152、664 issued/664 returned、5309 ticks/44.241666667s、task_success=false；physical evaluator自身termination_reason=null。663×8+末5=5309，末决策正常返回，不是假造完整8ticks或观察器中断。

原MP4为62672616 bytes，SHA256 `f174f65cecb47e9e5611c3aaf25d149b7137d9446bc9bade330edb60a067691b`；664帧/15fps/full decode PASS，但原container duration=286331153s无效。因此只以stream-copy生成新的`video10_161152_p01_incomplete_p05.mp4`，不覆盖原件。新片62596276 bytes，SHA256 `005fc942735dd8973a12cd882f7617192e9036e8d9530a566de9d79f10f4bdce`，严格容器时长验收PASS，ffmpeg显示44.27s。

原/副本媒体时长均为664/15=44.266666667s；比5309/120的物理时间长0.025s，是保留原生末5tick图像的显示量化，不是增加物理时间或帧。首末PTS为0/44.2s。

完整核验：664个压缩packet payload SHA256、精确有理数PTS/DTS/duration、packet key flags（23关键帧）逐项相同；全部decoded frame checksum、绝对PTS与PTS差分序列也完全一致。两者timebase1/15→1/15360仅变表示，秒数不变。ordered normalized packet记录SHA256=`8555874ed78aca6a17f139adfb4fd997d0a15b845afa97017a8debb6439c1943`。两片都是1280×720/H.264/yuv420p、full decode PASS、unique664、black-like0、15fps连续单调。无重编码、裁剪、拼接、变速、插帧或额外帧。

原片bytes/hash前后不变；原semantic source与capture manifest分别逐bytes前后一致，SHA256为`d14f9ae8b71812b214982a9400fd87ab6f95b3846574b472ce5dc88c9bc37abc`、`18eb4a2ad723c21eb6abdba0e905be6a4e027b7776fc9da4a316625e3ecd4d02`。所有写入仅在outputs/video_diagnostics；未改生产/四主报告/ledger/源目录，未重验video9或旧媒体/模型，未调用success publisher。

同目录交付：

- `video10_161152_p01_incomplete_p05.mp4`
- `video10_161152_p01_incomplete_p05.remux_receipt.json`（完整命令与验证）
- `video10_161152_decoded_01.png`（原帧等价frame0 / PTS0）
- `video10_161152_decoded_02.png`（原帧等价frame663 / PTS44.2）
- `remux_video10_diagnostic.py`（现有helper最小独立副本，nonoverwrite）

两张PNG均通过结构/1280×720检查；由主控随后亲自视觉核验。本媒体诊断不构成任务成功、稳定性改善或训练门禁。

# Video9：P05未完成诊断副本，不是成功视频

2026-09-10 22:52:40Z完成（receipt保存等价−04:00时间）。唯一处理来源为`20260910T2235582002787Z_gd4e46006b382_0f64bf3b35f74953994f8ca191952ee8/source`。Python helper及ffmpeg remux实际exit0；所有媒体子进程顺序执行，无并发CPU媒体任务，不启动Isaac/Torch，不干预训练。

原source manifest核对：实际加载checkpoint159104，631 issued且631 environment-step returned；630×8+末5=5045 physics ticks / 42.041666667s，task_success=false，physical evaluator自身termination_reason=null。主控已确认P05任务未完成；本次仅媒体核验，不重做任务诊断、不补造success。

原片59244888 bytes、SHA256 `3f9a6158657af9fc1bfbd5fe112a3e62a9f06d9912365e2e38747be5f3e74a05`，631帧/15fps/full decode通过，但container duration仍为无效286331153s。因此以`-c:v copy`生成独立`video9_159104_p01_incomplete_p05.mp4`：59172296 bytes、SHA256 `0946cf8d74fd491d90a10a71db778736e8b7795a7818af3a5b659f2e40496dcf`，strict container校验PASS，显示42.07s。

实际631/15媒体时长42.066666667s，比物理时间长0.025s，保留原生末5tick帧的显示量化，不宣称媒体/物理时长严格相等。首/末PTS=0/42s。

验证完整范围：631个encoded payload SHA256、精确有理数PTS/DTS/duration、packet key flags（22关键帧）逐项一致；全部decoded frame checksum、绝对PTS、PTS差分亦一致。容器timebase1/15→1/15360仅改变表示；ordered normalized packet记录SHA256=`252646e7777d196ad720d73e9084e948a6485a1c3d68ce562850c214f7a57147`。两片均1280×720/H.264/yuv420p、full decode PASS、unique631、black-like0、15fps连续单调。无重编码、裁剪、拼接、变速、插帧或额外帧。

原视频bytes/hash前后完全一致；原semantic source与capture manifest也逐bytes保持，SHA256分别`58a67cd7ef82e64dd2d1552047e8cf59fe6cb6f35bccc40c2c9652101dad19f2`、`fcc67d2d322a02c0982bffc21a53bdcce0dc67af25472971fffb08f1dd02e8b7`。未调用success publisher，未重验video8或其他旧媒体/模型，未改生产/主报告/ledger。

同目录交付：

- `video9_159104_p01_incomplete_p05.mp4`
- `video9_159104_p01_incomplete_p05.remux_receipt.json`（完整命令及原/副本验证）
- `video9_159104_decoded_01.png`（原帧等价frame0 / PTS0）
- `video9_159104_decoded_02.png`（原帧等价frame630 / PTS42）
- `remux_video9_diagnostic.py`（唯一新增最小输出helper，旧helper不变，nonoverwrite）

首末PNG结构检查通过、无额外标注；由主控随后亲自视觉检查。此次媒体完整性不构成任务成功、稳定性改善或optimizer门禁。

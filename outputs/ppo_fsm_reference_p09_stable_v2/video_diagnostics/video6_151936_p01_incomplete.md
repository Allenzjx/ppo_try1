# Video6 / CP151936：仅失败诊断副本

完整自然P01评估在78.4s于P06任务期限未完成；不是成功视频。原物理末证据为VALID / CONTACT_BEARING_UNVERIFIED，不重分类BODY、wheel-only或成功。

`video6_151936_p01_incomplete.mp4` 是原1176帧录像的packet-copy容器修复：无重编码、裁剪、倍速、插帧或拼接。重新全解码与原逐帧checksum、绝对PTS、差分、key flags完全一致；1176 unique帧、black0，容器78.4s。原录像/manifest前后不变。

校验：`video6_151936_p01_incomplete.remux_receipt.json`。首末帧：`video6_151936_decoded_01.png` / `video6_151936_decoded_02.png`。任务诊断：上级目录 `video6_151936_p06_diagnosis.md` / `.json`。未调用成功发布路径；首末PNG待主线程视觉核验。

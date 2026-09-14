# Video8：主控首末帧及诊断副本检查

主控实际查看 video8_157568_decoded_01.png（第0帧）与 video8_157568_decoded_02.png（第721帧）。机器人与障碍物均在画面内，首帧略有渲染模糊但可辨认，末帧清晰；未见黑帧、遮挡标签或裁掉失败尾部。末帧并未呈现整机完成通过。图像仅用于视觉完整性检查，接触/承载与任务分类仍来自物理日志，不从截图伪造腿角色、CoM或因果。

已读无损副本回执：全部722个压缩包的payload/exact PTS/DTS/duration/key flags及解码帧/PTS序列一致；full decode=true、unique722、black0。原MP4保留68,152,245 bytes/SHA256 2e876123d1e9b22d29dc90b415b7935a931bdcf4a9efdaa0f9b8a4a18308fe3f，但容器时长286331153秒异常，不能称原容器验收通过。独立副本为68,069,277 bytes/SHA256 74aa9cecd5ec5e28e0edf5c846425c476d6dad91eabe2f701cbbe8426044d6e4，1280×720/H264/yuv420p/15fps，正常容器48.13秒。

722/15=48.133333秒显示时长与5772/120=48.1秒物理时长相差既有末帧量化0.033333秒；未重编码、补帧、裁剪、拼接或变速。原source manifest保持不变（de0a38afc54d2ca1b1a635c7e063a5fab6b97a915761cc9e7fc590026e643c54），capture manifest亦保留。主控未再次运行大媒体验证或重复哈希；检查范围为已有执行回执与实际首末图。

此片只命名为 video8_157568_p01_incomplete_p05.mp4：自然P01/已保存模型重载/确定性，P05 LOCAL_BOUNDED_RECOVERY_EXHAUSTED，仍非完整成功，更不是FSM配对改善证明。722步均返回，最后仅4个物理tick；interrupted_final_decision_ticks=0表示未通过PhysicalEndpoint异常打断，不表示末步执行了8ticks。


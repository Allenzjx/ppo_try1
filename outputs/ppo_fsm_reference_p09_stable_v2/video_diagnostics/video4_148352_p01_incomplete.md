# Video4：未完成任务的可播放诊断副本

[播放视频](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video4_148352_p01_incomplete.mp4) · [核验回执](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video4_148352_p01_incomplete.remux_receipt.json)

2026-09-10 13:50:47 UTC核验完成。checkpoint148352从自然P01运行至P05未完成；本视频不是成功越障或稳定性改善证据。仅将原H.264码流无重编码复制到独立MP4，修复容器duration：286331153s→49.93s（FFmpeg显示精度，严格≤200s）。无trim、stitch、速度变换或插帧。

完整解码PASS：749帧、15fps、1280×720、unique749、black0。源/副本逐帧checksum、绝对PTS、PTS差分、keyflag序列完全相等。媒体49.933333s；物理5988ticks/49.9s，原有末帧显示量化0.033333s保持不变。

原视频71,343,870bytes，SHA256前后均为 `9b7c7e49a17651715f0b8cba311f3890a4aa44b036fa8e15f987f05ea4a71423`。副本71,257,622bytes，SHA256为 `4ec158b560c57ab1fa7efc3586b062e5905fc96830ca6fd58ea76460022d215c`。源manifest逐字节未改；其他原始材料保留。未调用success publisher，未修改原helper、生产、checkpoint或训练进程。

从该副本解码提取的PNG供主控视觉核验（此处不预先宣称人工看图已完成）：[第一帧/index0/PTS0](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video4_148352_decoded_01.png)；[最后一帧/index748/PTS49.866667](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video4_148352_decoded_02.png)。PNG提取不改变视频码流。

初次生成后仅PNG签名检查的转义字面量写错；修正output-only检查后，以verify-existing重新核验已生成文件，没有覆盖/重新生成媒体。过程如实记录在receipt中；这不是录像内容错误或任务判定变化。

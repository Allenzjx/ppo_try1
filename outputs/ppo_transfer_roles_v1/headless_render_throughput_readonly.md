# Headless 非录像训练：无渲染入口已在使用

结论：**正常训练没有每120Hz tick的 `render=True` 或应用帧率等待可直接关掉；不建议为此切换路径。** 已用源码和已完成 block8 小摘要/manifest核对，未benchmark、未读轨迹、未运行 Python/Torch/Isaac、未改生产。N1训练不等待本项。

调用链为 `train_semantic → RslAdapter.step → SemanticEpisodeEnv.step`（每决策最多8物理tick）→ `SemanticIsaacBackend.step_physics → IsaacFSMBackend.step_physics`：一次真实dispatch及native审计后，[isaac_fsm_backend.py:1993](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/isaac_fsm_backend.py:1993) 明确 `scene.sim.step(render=False)`，随后 `adapter.update_readback()`、sensor reader、controller更新。settle/prime同样用False。训练CLI以 `AppLauncher(headless=..., enable_cameras=False)` 启动，只在启动处显式 `app.update()`；视频的 `render_video_frame()` 是独立调用，不接入普通train。SceneCfg的 `render_interval=8` 不会把这个False自动变成每8tick渲染。

底层也核实：本机 [IsaacLab SimulationContext.step:541](C:/robotics_sim/IsaacLab/source/isaaclab/isaaclab/sim/simulation_context.py:541) 在正常playing状态直接 `super().step(render=False)`；安装的 [Isaac Sim core SimulationContext.step:672](C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/isaacsim/exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py:672) 的False分支直接 `_physics_context._step`，不调用 `_app.update()`。其physics step是 `simulate(dt,time)` 后 `fetch_results()`，这是物理结果同步，不是显示帧率sleep，不能去掉而仍声称同tick传感器/控制一致。

本机 headless experience [isaaclab.python.headless.kit:95](C:/robotics_sim/IsaacLab/apps/isaaclab.python.headless.kit:95) 已设 `app.runLoops.main.rateLimitEnabled=false`。无GUI/相机/livestream/XR时，IsaacLab默认 `NO_GUI_OR_RENDERING`，render自身也是空操作。暂停timeline分支可能刷新app；reset/启动也有生命周期update/render，这不等于正常每tick等待。这里没有读取运行进程的动态Carb设置，故不声称排除所有外部扩展/环境变量影响；但明确训练调用链本身没有待关闭的渲染循环。

传感器无需靠画面刷新：`robot.update(dt)` 推进读回时间戳；ArticulationData从PhysX tensor读取关节/链接状态，链接姿态getter按需更新kinematics。 [sensor_reader.py:121](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/sensing/sensor_reader.py:121) 每tick `ContactSensor.update(dt, force_recompute=True)`；本机ContactSensor从PhysX contact view取力、点和摩擦，不是RTX相机。geometry使用实时link positions配合静态collider cache。Fabric刷新到Hydra/可视变换与这些原生读取不同；未来若改为XForm/Fabric世界变换或启用摄像头，就不能沿用此结论。**不删除readback/contact update，不关Fabric模块，不绕过原物理step。**

已完成block8记录：headless=true；1024信用决策/8165信用物理tick，另3584前缀决策/28672前缀tick；训练wall=1739.0041s，前缀累计roll-in=1483.9228s，reset累计=30.9836s。这些计时范围不同（包括初始化边界），不能直接相减作完整成本拆分；没有per-tick render/physics/audit分解，不能据此把耗时归因渲染或承诺关闭某项提速。当前无渲染方式已经在用，保持现路径即可；本结论不增加任何任务成功或启动门禁。

# Outputs-only hip 安装点诊断候选

`hip_mount_geometry.py` 不在生产路径中，不修改422观察、动作、reward或许可；未部署到实际仿真。

接口：

- `resolve_hip_mounts(actual_provider)`：合法 reset/只读窗口，从 actual live USD stage 解析四个唯一命名 hip Joint。严格要求同一带RigidBodyAPI的 `/World/WLRRobot/base_link` body0、authored finite localPos0，保存实际root/session/property-layer来源。不存在URDF/对称性/上腿原点fallback；任意点缺失，整体invalid，mounts=null。
- `measure_hip_mounts(resolution, raw_observation)`：只取本拍 `bodies.base_link` 的 world link pose，输出四点world xyz/z、左右/前后均高差、FR world z、tick/time/source；不接触活动upper-link数据，不创建额外timer，不做任何物理调用。
- 正号约定：left−right>0为左高；rear−front>0为后高。资产安装点自身的高度差保留，绝不自动当作“应为0”的姿态目标。

调用方继续保存现有runtime资产hash receipt，不要把layer路径替代字节绑定。读取失败只显示N/A及reason，不应终止已运行的物理任务或发出动作。下一次实际live USD解析尚未执行；CP221696只直接记录了RR localPos0，另外三点仍须真实解析后才能发布verified四点诊断。

已执行纯标准库测试：

```powershell
& 'C:/Program Files/Python313/python.exe' -m unittest discover -s outputs/ppo_rr_rl_timing_policy_learning_v1/drafts/cooperative_prep_v4 -p test_hip_mount_geometry.py -v
```

**17 passed，0.001s，进程已退出。** 覆盖共同平移、roll/pitch/yaw符号、quaternion符号/幅值归一化、唯一性、同父刚体、finite/authored/provenance、缺点N/A、不读运动upper link、不修改输入。全部使用合成USD接口与姿态，**零真实仿真／训练信用**，没有Torch/PXR/Isaac初始化。

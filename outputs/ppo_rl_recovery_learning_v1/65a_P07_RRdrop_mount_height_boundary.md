# 65a P07：RR loaded/drop 五时刻髋高度证据边界

只读封存run `20260924T1046432416199Z_g65a9255be6d9_bff8c2f6d5574a42925e6b9d8f382a94` 的6272、6320、6328、6360、6416；不读当前512运行、不启动仿真/Torch/PXR。

**四个同刚体髋安装点世界z、左−右平均z、后−前平均z、FR角点世界z：本次均N/A。**
该train目录没有height diagnostic/physical-observation文件；五条JSON没有
`same_rigid_body_hip_mount_definitions` / `same_rigid_body_hip_mount_geometry` 或可绑定的原始同拍base_link pose。
现有 `semantic_hip_mount_geometry.py` 确实使用实际USD joint body0/localPos0，再由实测base_link pose转换；但其read-only recorder没有为这次训练输出结果。
actor schema有raw quaternion和CoM-relative-base编码，但本次未解码rollout张量，也没有本run已验证四个mount的解析receipt，不能凭代码存在补成实测高度。未以移动upper-link原点、关节角或相机左右替代。

下表CoM为日志mass-weighted世界坐标，Δ固定相对tick6272，单位mm；FR h/k是实测关节canonical°，**不是刚体FR角点高度**。
FR实际角由同拍receiver_workspace range margin与已记录硬限还原（hip lower=−135°、knee lower=−60°；代码margin=actual−lower），未从target当actual。

| tick / s | FR h/k 实测° | CoM Δx / Δy / Δz mm | RR / FL / RL 实测载荷N | RR当前 |
| --- | --- | --- | --- | --- |
| 6272 / 52.267 | 36.15 / -20.58 | 0.00 / 0.00 / 0.00 | 4.288 / 3.090 / 10.362 | 合法TOP |
| 6320 / 52.667 | 17.06 / -42.95 | 80.24 / 6.23 / 5.90 | 3.952 / 12.011 / 8.466 | 合法TOP |
| 6328 / 52.733 | 14.59 / -43.92 | 88.51 / 6.96 / 4.25 | 0.000 / 7.577 / 11.732 | AIR |
| 6360 / 53.000 | 18.61 / -39.69 | 69.17 / 11.55 / 9.18 | 0.000 / 6.269 / 10.290 | AIR |
| 6416 / 53.467 | 17.50 / -12.26 | 6.68 / 19.24 / 32.55 | 0.000 / 1.110 / 14.301 | AIR |

6272时固定FR方向约(+0.8273,−0.5618,0)。至6320有明显+x前送，但Δy为+6.23mm，与该固定FR侧向的负y相反；至6416变为+19.24mm。FR轴点积可因+x为正，不能把它称为已完成FR侧横向转移。RR6328失TOP；RL全过程仍GROUND，未取得qualified unload。FR本身五时刻均TOP、约11.55–14.72N；这与“FR关节负角必然撑高/阻碍转移”的单因果说法不同，尚不能从本表验证髋角点高低假设。

有限成功N事件表可复用以下实测字段，但同样未保存四mount高度或base quaternion，因此本轮不重扫原N物理日志：

| N tick / s | FR h/k 实测° | CoM Δx / Δy / Δz mm（自6152） | RR / FL / RL 载荷N |
| --- | --- | --- | --- |
| 6152 / 51.267 | 0.51 / 30.26 | 0.00 / 0.00 / 0.00 | 6.484 / 5.928 / 9.591 |
| 6201 / 51.675 | 8.95 / 29.87 | 48.49 / 2.51 / 25.75 | 13.999 / 14.454 / 0.785 |
| 6233 / 51.942 | 4.67 / 29.87 | 73.37 / -7.57 / 36.27 | 14.724 / 15.695 / 0.000 |

N在6233已出现固定起点CoM负y位移，并有RR/FL约14.72/15.69N、RL AIR0N。它展示可接续的支撑与运动差异，但不同run/不同构型不能证明复制FR +30°就足够。这里没有新增姿态目标、硬门、资产更改或运行要求。

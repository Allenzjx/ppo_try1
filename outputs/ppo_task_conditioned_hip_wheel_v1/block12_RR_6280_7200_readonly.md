# Block12：RR短暂放置后反复失接触；7200不是GROUND

仅分析活动run `3f4b1021c62b4daf84d7c5e53a91e9bf` 已落盘6280–7200（116端点），读取在7200立即停止。真实N前缀到5456接管，RR qualified5434属于前缀；PPO后缀继续cross5991/place6285，不能称完整P01策略成功。7200仍P11、非terminal，physical valid=true、termination=null、success=false。

## 首次保持丢失与回撤

6288、6296都是当前TOP/verified bearing/support（1.232/1.852 N）。6304首存AIR/0 N，故首次失TOP在 **(6296,6304]**，而不是假定精确6304；当拍current_lift_valid仍true。RR前沿距离从6288的+5.994 mm变为6304的−2.151 mm，仍在XY容差内；6312首次存为outside（−7.191 mm）。随后6336重回XY，6352又真实TOP并进入P11，6360再AIR；它不是一次单调撤回。

6432为AIR、gap+.351 mm/front+19.356 mm，之后仍多次AIR/TOP。**7029有直接ground撤销事件，7032首存GROUND/support=.607 N**。后续GROUND/AIR/模糊障碍接触交替；7200必须按当前字段报告：

| 7200 RR字段 | 真实值 |
|---|---|
| contact_mode / contact_surface | OBSTACLE_AMBIGUOUS |
| air / ground_contact | false / **false** |
| obstacle_pair_active / bearing_verified | true / false |
| support / bearing_force | false / 0 N |
| front / gap | −48.372 / −51.807 mm |
| current_lift_valid / historical placed | false / true |

因此“确实曾在7029回地”与“7200当前不是有效ground/TOP承载”同时成立。不能从air=false、负gap或历史placed推断当前地面/顶部支撑。

6288→7200的115端点中，RR TOP29、AIR69、ground14，其余为模糊状态；不是连续时长比例。真实body净后退 **25.426 mm**，RR中心净后退 **54.366 mm**。但最初6288→6312，body仍前进 **7.585 mm**而RR中心回退13.185 mm；轮端运动与机身位移不同，不能把最初撤回直接称为机身整体倒退。

## 四轮：负残差仍在，未见mask吞掉N

四轮顺序FL/FR/RL/RR，canonical rad/s。整个已读窗N与同拍mapped均为0，所以下表effective=final target；不是用独立重算N推残差。

| tick | effective / final四轮 | 实测canonical qd四轮 |
|---|---|---|
| 6288 | −.5353,−.0652,−.0785,−.0855 | −.5369,−.3158,−.2031,−.0220 |
| 6304 | −.5763,−.0807,−.1330,−.0502 | −.5763,−.0713,−.1189,−.0488 |
| 7032 | −.5183,−.1363,+.0851,−.0551 | −.5200,−.6544,−.2240,+.0543 |
| 7200 | −.1908,−.0407,+.1168,−.1188 | −.1905,−.0806,+.1862,−.0681 |

放置后RR target114/115为负（mean−.11846），actual105/115为负（mean−.14104）。策略负向作用与实际反转都存在；7032负target对应正actual，也显示接触/负载/全身运动下不能把ACK当完美跟踪。12维许可全1，所读派发均verified，final−mapped−effective误差0；独立final-stop owner未取得。无证据将此归因mask丢失，但没有扭矩/滑移反事实，不能唯一归因某个wheel导致后退。

源stop精确tick **未在这个窗内观察到，留null**：6280在P09且尚未capture时四轮N已经0；P09→P10=6288、P10→P11=6352时均保持0。P06层已retired/wheel_gain0，额外rolling suggestion未启用；不能说是这两个阶段切换刚把轮速清零，也不能凭P06原始rolling=.3说它仍有当前所有权。完整原子source-stop owner ledger和raw native实测qd缺失为null。

本报告只对固定窗负责，不等待或判定后续终态，不改生产/判定/reward或增加门槛。前缀hash只覆盖消费到7200的完整行，不是活动全文件hash；CPU helper已退出，无新物理运行。

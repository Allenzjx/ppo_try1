# 首个真实终态：回报与 reset 边界

只读核验已完成update1282/global168576内的首个terminal；2048训练块尚未宣称完成。episode0：358 decisions、2864 ticks、23.8667s，P09/FALL（PHYSICAL_SAFETY）。

terminal为global168550，实际执行2856→2864，完整8个physics ticks，并非短于8。done=true、time_outs=false、terminal_bootstrap_allowed=false；后一实际决策global168551已从P01 tick0→8开始新episode。

实际存储标量精确满足：

- reward = returns = −42.25470733642578
- oldV = −31.32651710510254
- rawGAE = reward − oldV = −10.928190231323242
- stored standardized advantage = −1.210860252380371

原始calculator double total为−42.254709145785796，与float32存储相差约1.81e−6，未把二者强写成相等。reward内terminal event−40，next potential0。done=1使官方递推中的nextvalue与未来GAE系数均为0；下一episode的实际oldV−17.194881439208984未进入前episode终态return。

本rollout后26条新episode样本令rollout尾部本身为非terminal，这不改变中间真实terminal的零bootstrap。未加载rollout/CP张量，也未另算critic尾值。规范化终态标量与全rollout已记录均值/标准差的仿射关系误差约1.87e−7。

同一rollout内P05→P06→P07→P08→P09仍为普通非terminal切换。以上只证实这次真实终态与reset边界，不虚构短N终态覆盖，也不作为新的训练门槛。


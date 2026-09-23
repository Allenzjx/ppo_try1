# 封存后固定 receiver 方向诊断：复用现有入口

状态：仅文本准备，当前真实 PPO 训练期间**未执行**。不新增诊断框架，不修改既有分析器、生产、配置或测试。等新 natural-P01 评估的 Isaac 退出、source manifest封存后，才允许下述单CPU调用。

## 复用与明确边界

已存在 `diagnose_c53119a_v5_video_draft.py` 主函数及其hash绑定的 `diagnose_57ae41e_sealed_fixed_com.py` 公共函数，已实现所需两个独立锚点、真正mass-weighted CoM、receiver/body位移分列、当前contact/bearing、yaw、缺失RL标记、自然P01/单episode/0optimizer及sealed hash检查。

已有完整主入口硬绑定旧source/HEAD/CP，因此不能直接运行旧CLI并把结果称为新评估。下面仅在独立Python进程内显式替换三个**输入选择常量**；原文件不变，原manifest/fullstream/hash/自然P01/0更新检查全部仍执行。不是替换生产规则或模拟状态。分析器旧输出格式名带`v5`，本次必须标为“legacy analysis format reused; actual control version comes from this source runtime contract”，不能把格式名当作新数据的控制版本。

分析器SHA256：`931837422464fe31e439f42248e29d202effa180053922f8872980f6bff38131`。
公共函数SHA256（主入口自动验证）：`b521aabc6ddc208ab65d2357aed97a8b2e3cadd42e2b6433e5129992a781ade1`。

## 预先固定的参考定义

- RR→FL：同一个natural-P01 episode首次实际P07调度入口的raw物理tick；RL→FR：首次实际P10入口tick。两个参考分别读取该拍 `center_of_mass.valid/position_w_m` 与对应wheel center。
- `d0 = normalize((receiver0_world − massCOM0_world).xy)`；固定后不随脚移动、普通phase切换、yaw或接触闪断重锚。分别输出 `dot(COM−COM0,d0)`、`dot(vCOM,d0)`、receiver自身投影、base自身投影；单位m/m·s⁻¹，表格显示mm时明确转换。
- **实际调度入口参考不等于精确物理卸载开始**。若真实转移提前发生，明确该固定窗未覆盖前序，不事后寻找更有利起点。若要报告真实力学onset，需先明确并审核独立事件定义；现有滑动role方向不能冒充固定起点。
- 世界对角投影包含向前行进分量，并受整体yaw影响；yaw0/当前yaw/差值单列，不把正投影等同侧向载荷成功。receiving脚AIR与其CoM方向仍分开。
- P10未到达：`anchors.P10.valid=false / reason=not reached`，报告 `RL_direction=not_reached`；不得用P07、终态FR位置或另一run代替。锚点缺测/COM无效/方向退化也为null，不猜。
- 每个里程碑保留真实current腿接触/verified bearing与history placed，二者不互代。后续GROUND导致尝试失效时，原锚点仍是原全段世界参考，不得将之后的投影宣称为同一次连续有效载荷转移；现入口不自动创造新尝试锚点。

## 待封存后的唯一调用（现在不要运行）

先由owner确认该评估进程退出，并从新封存manifest解析准确SOURCE、runtime完整HEAD、实际加载checkpoint SHA；不是盲用训练祖先SHA。固定新唯一输出目录（必须不存在）。工作目录为本仓库根。

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
# 用未来实际封存值填入；以下占位符不得执行。
$fixedSource='<SEALED_NATURAL_P01_SOURCE_ABSOLUTE_PATH>'
$fixedHead='<ACTUAL_FULL_RUNTIME_HEAD>'
$fixedCheckpointSha='<ACTUAL_LOADED_CHECKPOINT_SHA256>'
$fixedOutput='<NEW_UNIQUE_OUTPUT_ABSOLUTE_DIRECTORY>'
& C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe -c "import hashlib,importlib.util,sys;from pathlib import Path;p=Path('outputs/ppo_rr_capture_then_rl_transfer_v1/diagnose_c53119a_v5_video_draft.py');assert hashlib.sha256(p.read_bytes()).hexdigest()=='931837422464fe31e439f42248e29d202effa180053922f8872980f6bff38131';s=importlib.util.spec_from_file_location('sealed_direction_review',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.SOURCE=Path(sys.argv[1]).resolve();m.HEAD=sys.argv[2];m.CP_SHA=sys.argv[3];m.main(Path(sys.argv[4]).resolve())" $fixedSource $fixedHead $fixedCheckpointSha $fixedOutput
```

只读各指定流一遍、只保留固定锚点和有限实际milestones/末段摘要；没有physics、actor/optimizer、视频编码。既有主函数同时读取stage/decision/native/capture/physical和可选height流以完成证据对齐；这不是仅几行tail查询，等物理退出后执行。若只需两个CoM数字，不必再额外遍历其它历史库或Recording。

实际交付应摘出P07/P10各自锚点及RR TOP/placed、P10/P11/P12/terminal等已发生状态的投影/receiver motion/yaw/current support表，保留完整失败结尾。旧schema/文件名仅格式版本，真实源、HEAD、checkpoint、任务成败由JSON记录；不存在RL则明确未到达。若新writer schema不匹配，停止分析并报告，不能靠绕过manifest或拼接其它run补齐。

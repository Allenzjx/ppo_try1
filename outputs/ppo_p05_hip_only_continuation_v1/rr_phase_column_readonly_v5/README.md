# P09 column sensitivity — static preparation only

This directory contains **no executor, budget, fitting method, checkpoint writer, or AUX event**. At preparation time the CP218496 video is still running: the helper has not been imported, forwarded, tested, or executed. Run only after video sealing and explicit root review.

`inspect_rr_phase_column.py` binds CP218496 (`6f7c572a…f9e227`) and current runtime `6ac7b553…`. Its only positive inputs are the seven directly saved block09 rollout1655 observations/actions at decisions **216238–216244**. Targets are actual full12 raw actions, not stored means, nominal or final drives. Existing sealed credit evidence binds actual RR crossing/capture and subsequent TOP/P10/P11 continuation; later contact loss is explicitly excluded. These adjacent samples are off-policy AUX candidates from one trajectory, **not independent validation, a new rollout, or full-task success**.

The existing actor's first-layer P09 column `mlp.0.weight[:,8]` has 256 values. Exact Identity-normalized nonP09 inputs have `X[8]=0`, so changing only that column cannot change the same-input full Gaussian in P01–P08/P10–P13. Algebraic synthetic probes check this without claiming real state coverage. P09 mean **and** logσ may change; future visited states and HISTORY are not invariant. No new architecture or normalization is introduced, and the true `.1` conditional-mean derivative remains intact.

The helper reports current raw mean-target MSE, mean/logσ/σ/full12 REQUEST, loss gradient, and JVP along the negative gradient and its unit-L2 direction. Optional `--include-seven-state-sigma-nullspace` computes an 84×256 logσ Jacobian, numerical rank, and the loss-gradient projection into its local numerical nullspace. This is **pure derivative linear algebra**, not an implemented optimizer or proof of finite-step/all-P09 σ invariance. Numerical rank uses a reported float32-precision tolerance with a double-precision SVD.

Old/current control MDPs are explicitly different: the bound identity migration changes P05 nominal recovery only; its new P05-only gate is off on all seven P09 inputs. Observation codec, reward, actor kernel, caps, sigma scheduling, capture assist, and task acceptance are unchanged by that migration. Current whole-body/HISTORY distribution need not resemble the old trajectory. The P05 helper's `WAIT/initialized==0` positive-label gate is intentionally **not** reused for these later P09 inputs.

Source actor/critic/Adam/normalizer/RNG/counters remain protected; only a CPU actor copy is instantiated, with full process RNG restoration. The only possible file output is a new inspection JSON inside this directory. No gradient result constitutes authorization or a learning-rate/trust-budget choice.

Future reviewed command (do not run while the video is active):

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' 'outputs/ppo_p05_hip_only_continuation_v1/rr_phase_column_readonly_v5/inspect_rr_phase_column.py' --include-seven-state-sigma-nullspace --report 'outputs/ppo_p05_hip_only_continuation_v1/rr_phase_column_readonly_v5/CP218496_seven_capture_sensitivity.json'
```

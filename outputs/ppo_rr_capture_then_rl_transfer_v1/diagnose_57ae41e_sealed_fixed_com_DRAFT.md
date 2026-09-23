# Fixed CoM / RR actual-response diagnostic — preparation and execution status

Executed after explicit natural-seal/Isaac-exit authorization on 2026-09-23, CPU-only with one-thread settings. Exit 0; 10 compact milestones. Actual results and source-stream hashes: `headless_57ae41e_fixed_COM_review/fixed_COM_RR_response.json`; concise interpretation: `headless_57ae41e_fixed_COM_review/ACTUAL_SUMMARY.md`. No production/config changes, actor forward, optimization, or new physics. The preparation account below describes the earlier frozen-runtime boundary.

This output-only standard-library script is restricted to the declared `57ae41eb4b57` headless evaluation. Preparation used production writer code and only bounded first rows from one already sealed historical headless run. No Python/helper, simulator, model, optimizer, tests, video encoding, live broad scan, or production edits were performed to prepare it.

Run it **only after the owner confirms natural sealing and Isaac exit**. It additionally requires completed run/evaluation manifests, matching runtime contracts, zero evaluation optimizer updates, and contiguous raw/native/decision tails matching the sealed extent. It parses each of four JSONL streams once, hashes the bytes actually read, and refuses incomplete/changing streams. Windows live file `Length=0` is not evidence that a writer is inactive; the current run's actual lines were independently confirmed by the root agent. No observer-wiring fix is proposed.

## Exact evidence and clocks

- First P07 entry fixes mass COM0 → FL wheel center; first P10 fixes mass COM0 → FR wheel center. They are scheduler-entry references, not asserted exact unloading onset. No moving-foot re-anchor. Missing P10 or invalid COM/receiver means unavailable, never a rolling-tracker substitute.
- The reader joins exact episode ticks across stage, native and raw streams. A native row is the dispatch that produces that post-step physical row; its controller/global dispatch tick can have a reset/settle offset. Assist/wheel context carries its own prior observation tick, retained separately.
- It selects RR real lift/cross/place events, P07/P10/P11/P12/P13 entries, first wheel-envelope/projection, first assist modes, knee-search/budget transitions, first pre-dispatch TOP/contact, and terminal. These are a small milestone set, not full trace republication.
- `native_tick_audit.nominal_full12` is a **top-level sibling** of `native_audit`, not inside it. Wheel evidence is `native_audit.rr_carry_wheel_evidence`, including source stop proof, gain, selected bearing channels, prior FINAL, candidate/desire/output/controller delta. Native actuator target arrays are kept separately from post-step canonical measured wheel velocity.
- `current_legs` is retained only at the exact 15 Hz decision endpoint. At other ticks, raw contact pairs plus actual pre-dispatch verified FL/FR/RL support and RR TOP-bearing context remain available; no later endpoint is silently used. Historical placed never substitutes for current bearing.
- Headless raw rows contain actual mass COM, body pose, wheel geometry, joint state, body link origins, and raw contact pairs. They do **not** carry USD hip joint mount localPos0, so true RR hip mount is explicitly null; the upper-link origin is separately named. Native joint IDs are null if not persisted in the audit. No inferred traction or pose-causality claim.

The following was the reviewed command (now completed). A rerun requires a **new** output destination; existing reports are not overwritten:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& C:/Users/kskzz/miniconda3/envs/env_isaaclab/python.exe outputs/ppo_rr_capture_then_rl_transfer_v1/diagnose_57ae41e_sealed_fixed_com.py --owner-confirmed-sealed --output outputs/ppo_rr_capture_then_rl_transfer_v1/headless_57ae41e_fixed_COM_review
```

The produced JSON/Markdown report is diagnostic only: zero physics, policy decisions, optimizer updates, auxiliary updates, or reward/GAE credit.

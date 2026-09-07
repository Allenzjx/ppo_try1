# Previous-ACK REQUEST tracking reference — versioned control revision

Committed runtime **42b91e857a0a2da256dee85e8092642ca8e64aeb** (parent2677995), seven production/config files plus three test files; no remote push. Production worktree was clean when real smoke started.

## Actual predecessor and motivation

Source runtime2677995544c974c03d8b1e41d77375b45e323c9e completed block27 at **101376 policy decisions /757 PPO updates /15140 optimizer steps**. The immutable source is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000101376.pt`. Its saved round-trip check is true. Source block27 added8192/64/1280; no prefix credit. Reloaded natural-P01 C101376 actually ended966dec/7728ticks/64.4s in P06 incomplete, physical-valid/null hard-failure, zero optimizer updates. It did not complete the obstacle task. See `p01_block_101376.md` and `eval_101376_diagnosis.md`.

The bounded C93184 control investigation (`fl_contact_control_contrast_93184.md`) confirmed real q-driven reverse hip tracking feedback:16 sampled updates exactly follow the original gain8/clamp/slew law. At t5256 FLhip nominal22.8°, mapper correction−6.186242°, intact request+6.327937°, final target22.941695°, actual q23.412275°. This demonstrates partial opposition, **not a single cause of failure**; successful FL placements also had opposite hip feedback, and their knees/wheels/support differed. The new mode is a testable control-semantic change, not a claim that fixing hip feedback solves P06 or rear placement.

## Production changes

- `semantic_tracking_reference.py`: read-only pre-dispatch capture and pure previous-ACK filtered REQUEST reference calculation. It validates actual q, prior native ACK/request composition, mapper/write/sample clocks, and explicit reset bootstrap. It never steps physics or mutates sensor/mapper state.
- `semantic_residual_adapter.py`: the unique original mapper advance receives a local computational reference only on eligible scheduled feedback channels; authored nominal and actual sensor data are unchanged. Current residual zero does not erase nonzero previous request. The original zero-history/no-geometry dispatch path remains available.
- `semantic_backend.py`: explicit opt-in configuration and the reset-derived bootstrap tick. Ordinary phases do not reset reference history.
- `isaac_fsm_backend.py`: capture independent real pre-dispatch evidence before the one actual write, only when the new mode is enabled.
- `actuator_target_effect.py`: independently rebuild the reference from that captured context and source tracking names; reject missing/tampered provenance and clock mismatch. The existing four float32 target-buffer checks and same-history zero-current-residual comparison remain.
- `semantic_video.py`: successful post-roll retains the enabled reference mode and prior request; it cannot bootstrap a new history after success.
- `configs/ppo_semantic_v3/execution_profile.yaml`: opt-in `previous_ack_requested_servo_reference_v1`, revision `continuous_whole_body_exploration_v3_previous_request_reference`.

Frozen A mapper/controller/protected files are not edited. The inherited backend's new branch is opt-in; v2/A have no reference mode. No task evaluator, reward, phase goal, nominal schedule, physical parameter, 324-column schema, wheel cap, servo cap or slew value changes in this revision.

## Exact limits of the reference

With current real canonical error e0 and previous ACK REQUEST r-, desired original correction c0=clip(K*e0,±C), reference correction c1=clip(K*(e0+r-),±C). Clamp c1 into `[max(-C,min(c0,0,L-n)), min(C,max(c0,0,U-n))]`, where L/U inset the frozen servo hard bounds by2°. Transform only this computational error input, then call the frozen mapper once. Inactive, ended-tracking, nominal-changing, not-yet-reached and non-feedback channels retain actual q exactly.

The requested value is already represented by the current 324-column previous-residual history (servo/4, largest36°→9<clip20); it is **not** previous effective residual or physical displacement. A prior request rejected by headroom still expresses intent. No extra effective-residual hidden state is added; this does not assert that the entire robot system is fully observable.

The reserve bounds **desired mapper correction**, not every final native target. Old correction must still slew back, and original baseline/controller/geometry exceptions remain. The test n130°/old compensation5° gives desired3° but next correction3.75°, native133.75°: retained old slew is explicitly not called a new global2° guarantee. Actual hard limits, original1.25°/tick final servo slew, original controller±10° envelope and current post-mapper residual headroom remain enforced. No predictive collision guarantee is claimed.

## Verification before physics

**505 distinct CPU tests passed**, zero final failures/errors:

- `tracking_reference_final_specialized_20260906.xml`:137 tests (89 pure helper,40 real mapper/CPU float32 dispatch,8 profile/extra wiring),2.963s.
- `tracking_reference_existing_regression_20260906.xml`:189 tests,13.336s.
- `tracking_reference_continuity_regression_20260906.xml`:179 tests,16.933s.

Receipts live under `C:/robotics_sim/wlr_robot/`. Earlier failed/intermediate specialized receipts are preserved and are not added to the final count. Tests cover all signs, reserve/hard edges, requested-versus-effective separation, exact zero-history comparisons, current-zero history, one mapper/write, ended-tracking real q, phase history continuity, stale/forged ACK/context, and all four actual CPU float32 buffers. These are not Isaac physical results or task-success evidence.

## Migration and next actual run

This is a new MDP/control-semantic boundary, not an ordinary exact-runtime resume. Use the actual latest valid source101376, preserving actor including learned state-dependent std, critic, identity-normalizer state, unchanged preprocessing, training RNG, lifetime counters and prior stage budget spending. Existing explicit NewMdpWarmStart resets Adam moments with initial learning rate3e-5; old unfinished rollout and physical state are not inherited. Collect fresh legal-reset data under the new runtime. The existing initial-action comparison concerns same-state logical projection only, not changed mapper dynamics or a physical trajectory equivalence proof.

At this report's initial publication the new revision has **no Isaac or PPO-update credit yet**. Real smoke/response and continuation receipts must be appended only after completion. No A5/5 or complete manual-probe success gate is introduced. Original A remains an incomplete historical-entry-blocked run, not retroactively successful. No full-P01 PPO success, successful PPO video, paired stability advantage or final delivery is claimed.

## Completed real smoke and first saved update

Real Isaac smoke `interface_checks/20260907T0131475778559Z_g42b91e857a0a_976e83ff5565408d9804fc6c2e3e3707` exited0, executionSUCCEEDED/functional_passed. Actual128 diagnostic decisions, two independent legal resets at decisions1/65,1024 physics ticks/native verified,956 actual/own-phase effect ticks,120 decisions with nonzero target effect, four state-write totals0. Both episode windows are64dec/4.266667s; final P02, taskfalse/nonterminal. Optimizer updates and optimized-policy credit are **zero**. All128 decision-end reference receipts independently verified. Compact tick summaries do not retain per-channel feedback-reference-used evidence for all1024 ticks; no active-reference count or physical improvement is invented from endpoint samples.

Training block28 is genuinely running, `train/20260907T0134080107746Z_g42b91e857a0a_a862752b4ec84772b24f061befcd1676`, N1seed1001/naturalP01, requested4096, checkpoint interval8 updates. It is a fresh legal-reset NewMdpWarmStart from101376, not a completed4096 budget yet. Initial migration receipt records preserved actor/std/critic/normalizer/RNG and discarded old rollout/physical state, fresh Adam3e-5. By the first bounded verification the optimizer had actually reached101632/759; these are live counts, not the final block ledger.

The first immutable new-control checkpoint **checkpoint_step_000101504.pt** was saved at101504/758/15160 with roundtriptrue, actual+128dec/+1PPO/+20optimizer. Pointer-recorded SHA82c9aa5f1f00766f4315bf085a3c813ecdfb60bef6fddaadd4a4ba897e69e3db; sidecar3b37d0fbc5e75f61b3218ed741ddc38cf31533ab4fad6ae5389f039424395050. Update-end LR1e-5 is recorded separately from the fresh3e-5 initial LR; no claim that every minibatch used one constant rate. First-save detailed audit is separate; no new full-P01 evaluation or task success is credited here.

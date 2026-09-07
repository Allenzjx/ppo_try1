# Dense soft workspace potential — versioned learning signal

Runtime **f1a9bbf650b1b8f80d5169e09173fcfc68797d99**, parent42b91e8. Three production/config files and three test files committed locally; no remote push. Source latest valid checkpoint **103168 /771 PPO updates /15420 optimizer steps**, not the stale10112 restore point. Source is `checkpoints/history/checkpoint_step_000103168.pt`, SHA005d8794fc4fbae5616c478321ddd1d0e2d8f3082fd94caec22d3acbb6a6c0c3, recorded roundtriptrue. Its full/suffix spent budgets are45312/47744; originalv3 origin10112 is retained.

## Narrow motivation and change

The completed C101376 P06 trace has600 decision endpoints, both rear legs laterally in span but always farther than.25m outside the existing[-.22,+.06]m workspace interval. The old clipped workspace component is zero at all600 endpoints. Other potential and reward components are **not** all zero. This is a verified component plateau, not a demonstrated sole cause of failure. The six measured first/closest/end distances are retained in `workspace_progress_plateau_101376.md` and production arithmetic tests.

`semantic_workspace_potential.py` adds a pure, bounded soft value `.25/(.25+d)`, where d is distance to that existing interval; out-of-lateral-span remains0. Stable arithmetic handles finite extremes and rejects nonfinite/malformed input. Far gradients decay and may ultimately underflow; this does not remove every plateau.

`semantic_supervisor.py` calls it at exactly two sites inside `physical_potential`: ordinary unplaced-leg workspace and the existing predecessor-blocked workspace-only preparation credit. Each retains its old `.85/4*.1=.02125` total-potential share. All legs use the same formula. Predecessor gating for unload/lift/cross/capture, actual Q/C/P history, placed retention and finish branches are unchanged.

`stage_task_spec.yaml` selects `workspace_interval_reciprocal_potential_v1` and revision `continuous_whole_body_v3_dense_workspace_potential`. The public workspace predicate, entry/completion predicates, phase transition, nominal scheduling and evaluator are unchanged. A soft value may round1 immediately outside the interval; it is never used as a replacement hard gate. AIR still cannot become load-bearing by geometric overlap or historical placement alone.

Action/controller modes, bounds and slew, policy architecture and std, 324-column schema/scales, five reward-family weights, gamma.995/PBRS, terminal semantics and rollout/GAE remain unchanged. Existing observation contains a Phi scalar whose meaning changes, so this is explicitly **new MDP/reward semantics**, not exact-runtime resume.

## Actual verification

The final selected regression inventory contains610 JUnit entries including44 subtest entries (566 top-level parametrized cases/methods). In the first run609 entries passed; one optional real historical-checkpoint test correctly rejected a CUDA RNG device-count mismatch in the deliberately GPU-hidden test environment. A second attempted environment clearing still exposed no device and retained that failure. No code/assertion was weakened. A clean process verified one availableCUDA device and reran the19-case continuation file: **19/19 passed**,5.825s. These reruns are not added as new distinct coverage.

Receipts under `C:/robotics_sim/wlr_robot/`: `workspace_potential_final_regression_20260906.xml` (28.122s, original environment failure preserved), `workspace_potential_real_rng_regression_20260906.xml` (failed clearing attempt preserved), and `workspace_potential_real_rng_regression_visible_20260906.xml` (final19pass). Independent specialist runs are also retained but not double-counted. Coverage includes real evaluator continuity P06→P07→P08→P09, unchanged nominal/entry/completion/nextafter boundaries, exact existing workspace weight, no fabricated events, placed AIR retention, absorbing terminalPhi0, single PBRS application, 324 layout with only existingPhi scalar changed, compatible heteroscedastic migrations, and tracking-reference dispatch.

Historical preparation-credit tests explicitly select their original workspace formula to preserve all old numeric and hard-condition assertions. The independent new38-case integration file exercises the actual production default.

## Real training status at publication

Block29 actually launched in Isaac, run `train/20260907T0205293896232Z_gf1a9bbf650b1_f859147a01004e878cecae656230e34c`, N1seed1001/naturalP01/full_episode, requested4096 decisions/checkpoint interval8 updates. Native PythonPID148380/session72444. NewMdpWarmStart strictly verifies source weights/old optimizer before retaining actor including learned state-dependent std, critic, identity-normalizer, RNG and counters; it then creates fresh Adam initialLR3e-5, empty rollout and legal reset. No old partial rollout/physical snapshot is inherited. No A5/5 or successful-manual-probe optimizer gate is added. At publication the initialization receipts exist, but **no completed4096 budget or new update is claimed yet**.

Latest completed full evaluation is **C103168**,660dec/5273ticks/43.9416667s, P05 incomplete FL capture. FL qualified1754/crossed2791, no placed event; finalAIR0load and+16.635617mm gap, physicalvalid/nullhardfailure. All5273native rows verify; reference actually used1306ticks. This remains a task noncompletion, not a video/physics-chain failure. See `eval_103168_diagnosis.md`. No full/suffix PPO success, successful PPO video or paired superiority is claimed. Original A and all prior failure/incomplete/history artifacts remain intact.

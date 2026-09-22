# CP203776 feedback-v2: bounded P07–P09 obstruction check

Current natural-P01 deterministic run `20260922T0631083720315Z_ga802b24d78df_aced5ae61b384b9f8a8a79db9a30914a`, source under `runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/`. Read-only window: episode ticks **7368–8392**, 61.400–69.933333 s, runtime `a802b24d78df`. This is an intermediate diagnosis, not the final episode result. No production/config edits, controller replay, extra Isaac process or state writes were performed for this report.

## First remaining obstruction

**P09 has really started; RR lifts briefly, then loses carry clearance before crossing.** The source is not waiting for old FL placement or stage entry. Its local clock is held at pending knee waypoint **96 / 0.8 s**, whose source target would be RR knee **−16.6°**. Previous knee waypoints −4.9° and −11.3° were actually issued. The first hold occurs while RR is still qualified AIR but low and descending; after RR touches ground, the same pending event correctly reports missing current free lift. This is a measured lift-retention/whole-body trajectory failure under the current controller, not an observed missing-actuator or residual-mask failure. The audit does not prove that this readiness rule is an optimal task policy or that bypassing it would safely succeed.

| Episode tick / time | Actual source or physical evidence |
|---|---|
| 7368 / 61.400 s | P07 label and source start. |
| 7376 / 61.466667 s | P08 label; source waits for P07 groups and measured response. |
| 7384 / 61.533333 s | P09 label; source initially waits for RL response and actual RR unload. |
| 7408 / 61.733333 s | P07 FR atomic group starts. |
| 7568 / 63.066667 s | P08 source starts after its predecessor condition resolves. |
| 7616 / 63.466667 s | P09 source starts; FR and RL are real other supports. First completed native RR hip command appears at 7617. |
| 7641 / 63.675 s; 7647 / 63.725 s | RR initial free lift, then qualified measured lift (8.827 mm free rise). |
| 7697 / 64.141667 s; 7705 / 64.208333 s | Completed native ticks first issue knee −4.9°, then −11.3°. |
| 7700 / 64.166667 s | RR maximum gap in this window: **+18.918 mm**, still **218.307 mm before the front edge**. |
| 7712 / 64.266667 s | Source local 96 holds: `current_AIR_clearance_decreasing_before_pending_knee`; gap +13.365 mm, measured bottom velocity **−0.106024 m/s**, current qualification true, two verified other supports, fresh evidence. |
| 7740 / 64.500 s; 7741 / 64.508333 s | Current qualification revoked, then real ground contact (15.358 N). No crossing. |
| 8392 / 69.933333 s | Still local 96: `current_free_lift_before_pending_knee`, current qualification false, GROUND. Entry valid, no entry reasons, EXECUTION; P01–P08 actually completed. |

The authored hip-return sequence begins at local 136 and four-wheel +0.3 rad/s at local 232; neither has been reached. This is not a lost +0.3 command: it remains downstream of the unpassed knee waypoint. P07/P08 source clocks continue (1025/825 at tick 8392), while only P09 remains at 96. The P06 wheel contribution has been geometrically retired (gain 0); its zero nominal is not a mask applied to PPO. Existing wheel residuals continue.

The current code path is `semantic_supervisor.py:_sequence_permission` (line 1937) → `_rr_pending_carry_readiness` (line 2049). It holds pending authored events, not the mapper, residual history or simulator. The initial rejection depends on current low **and decreasing** AIR clearance, not a historical pose, fixed settling duration or a requirement for all feet to be planted. FL is airborne and is not counted as bearing.

## Same-tick RR command path

Canonical joint degrees. `N` and mapper values below come from the matching **native audit**, not the next-observation nominal in the capture row. Baseline is the recorded geometry-corrected, controller-inclusive value used before adding the current PPO residual. This distinction matters: at 7696–7741 the baseline differs from the raw mapper by up to −10° hip / +10° knee; that difference must not be falsely attributed to PPO. Requested and headroom-effective PPO residuals are identical at these ticks. Capture-assist ownership and final slew/clamp indices are empty at these four samples.

| Tick | N hip / knee | Raw mapper hip / knee | Actual pre-PPO baseline hip / knee | PPO requested = effective hip / knee | Final hip / knee | Measured hip / knee |
|---:|---:|---:|---:|---:|---:|---:|
| 7696 | 55.600 / 0.000 | 56.850 / 0.000 | 46.850 / 0.000 | +12.998344 / −14.390008 | 59.848344 / −14.390008 | 54.618787 / −14.398367 |
| 7712 | 55.600 / −11.300 | 56.850 / −11.300 | 46.850 / −1.300 | +13.018571 / −14.407786 | 59.868571 / −15.707786 | 58.201051 / −14.882464 |
| 7741 | 55.600 / −11.300 | 56.850 / −21.300 | 46.850 / −11.300 | +13.088833 / −14.471683 | 59.938833 / −25.771683 | 59.643614 / −21.694610 |
| 8389 | 55.600 / −11.300 | 56.850 / −11.811401 | 56.850 / −11.811401 | +12.626129 / −13.949608 | 69.476129 / −25.761009 | 69.678207 / −25.182280 |

At episode tick 8389 (raw dispatch tick 8568), RR final-minus-measured errors are −0.202078° hip / −0.578729° knee. RR knee actual is 34.818° above hard minimum −60°; final is 32.239° above reserved minimum −58°. Hip also has ample numerical headroom. This excludes a frozen joint or current hard-limit saturation, but angular margin alone does not establish a viable wheel trajectory.

All **1025** native audit rows in 7368–8392 record verified mapping, matching setter/dispatch targets, independently verified previous-ACK tracking and all-one residual permission masks; zero headroom clips and zero in-episode root/velocity/force/gravity writes. Capture assist owns only FL indices 0/1 for the first 91 ticks, then no channels from tick 7459 onward. RR is never assist-owned. These are the run's original recorded audit results, not a new full replay audit.

## Whole-body context, not a single-joint causal claim

| Tick | RR gap / front mm | Body z mm | FL gap mm |
|---:|---:|---:|---:|
| 7616 | −50.066 / −161.597 | 108.826 | 65.692 |
| 7700 | +18.918 / −218.307 | 113.506 | 92.333 |
| 7712 | +13.365 / −219.082 | 117.205 | 102.116 |
| 7741 | −51.031 / −204.341 | 136.502 | 153.445 |
| 8389 | −50.743 / −197.748 | 140.309 | 167.619 |

RR moves farther behind the edge during the lift. Its clearance then falls while the body and FL rise; several joints, geometry correction and support loads evolve together. These logs therefore do not isolate RR knee, hip residual or body height as a unique causal lever.

At tick 8389: FL AIR, **0 N** (historical placement is not current bearing); FR TOP 13.161 N; RL ground 11.904 N; RR ground 3.023 N. Support margin is +51.135 mm with three actual contacts, and RR has two genuine other supports. Actual FL hip/knee is 31.986/−32.285°, FR 7.591/−1.775°, RL 15.588/2.520°. Body velocity is [0.001539, −0.010122, 0.016832] m/s, not motionless.

At the same tick, nominal wheel targets are all zero; final canonical FL/FR/RL/RR wheel targets are [−0.434184, +0.089028, +0.026445, −0.098231] rad/s, and measured angular velocities [−0.433905, −0.072749, −0.000831, +0.032628] rad/s. The loaded wheels' rate mismatch is visible; native command delivery is verified, but perfect loaded tracking is not claimed. It is not correct to say only RR has a wheel command.

**Outcome through the bounded endpoint:** real RR lift was acquired but not maintained or advanced to crossing. No RR placement, RL traversal, full-task success or stability superiority is established. Preserve this run and await its separate final result; no hot fix, gate bypass, reward change or additional physics run is proposed by this read-only report.

## Sealed final result (appended after episode end)

The same run sealed at **2026-09-22 07:04:41 UTC**. Its final decision 1374 executes one physics tick, ending at **tick 10985 / 91.541666667 s**, P09 **`INCOMPLETE_CONTROLLER_BLOCKED`**, source **`LOCAL_BOUNDED_RECOVERY_EXHAUSTED`**. This is a finite task terminal, not an external truncation: `time_outs=false`, terminal bootstrap false, P09 local limit 30 s plus 0.000405 s measured-progress allowance. Entry remains valid with no entry reasons. The physical evaluator remains `valid=true`, `run_validity=VALID`, `physical_evidence_status=VERIFIED`, with no physical safety termination (`termination_reason=null`); the semantic supervisor, not a sensor failure or hard-safety abort, ends this incomplete attempt.

Final event ledger: FR lift/cross/place at ticks **23 / 2207 / 2218**; FL lift/cross at **2247 / 3352**. P05's local deadline advances the scheduler to P06 at **5832 / 48.600 s with FL still unplaced and no completion credit**. Actual FL placement arrives later at **5856 / 48.800 s**, completing P05 without resetting physics or rewinding the scheduler. This run therefore physically exercises pending continuation, unlike the earlier two deterministic attempts. Historical placement does not certify sustained bearing: final FL is AIR, 167.727 mm top gap, 0 N.

RR initially clears at 7641 and qualifies at **7647**, but returns to ground at **7741**; final active-lift/current-validity bits are false while the recorded qualification event tick is retained. There is **no RR crossing or placement** and no RL traversal. Final RR is GROUND, gap **−50.803 mm**, front distance **−196.363 mm**, force **3.197 N**. P01–P08 are completed, P09 is the first unfinished task; full traversal and controlled-final success remain false.

The manifest's lifecycle is `DIAGNOSTIC_FAILURE`, with `source_acceptance_error="SemanticVideoError: episode did not meet common physical task"`. The launcher exit 1 represents this physical task nonacceptance, not a missing recording. The sealed source records **1374 actual frames at 15 fps**, encoded 91.600 s including 0.058333 s final-display quantization, zero extra pre/post physics/frames, no speed modification or stitching. Evaluation and controller migration add **zero PPO updates**. Raw artifacts are retained; this appendix does not claim a separate decode or annotated-export QA pass.

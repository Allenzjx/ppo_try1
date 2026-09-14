# P04 training: first two completed episodes only

Read-only bounded diagnosis, 2026-09-10, production `64c03243ac05`. Run: `runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0852458310353Z_g64c03243ac05_aafdb9864bdc42dda72ef793af390efd`.

Evidence boundary: `residual_and_projection_audit.jsonl` lines **1–318**, immutable byte range **0–23,622,511** inclusive; exact SHA-256 **0998b2c2c8813d31634c2fa05078fec1f163f0e4cca98ea68f364735c79e2f17**. Also the first two completed-episode records and first two optimizer records. No later policy rows or active-run state used. No production/process changes or simulator/tests launched.

## Results and exact sample accounting

| Episode | Policy globals | P04 | P05 | P06 | Executed physics ticks | Actual terminal |
|---|---|---:|---:|---:|---:|---|
| 0 | 141825–142005 | 1 | 166 | 14 | 1444 | FALL, P06, tick 3140 / 26.166667 s |
| 1 | 142006–142142 | 1 | 136 | 0 | 1093 | FALL, P05, tick 2789 / 23.241667 s |
| Total | 318 decisions | 2 | 302 | 14 | 2537 | Two safety terminals; zero success |

Each teacher prefix reaches real P04 at tick 1696 / 14.133333 s; prefix time is not policy credit. Suffix durations are 12.033333 and 9.108333 s. Teacher FR already has Q23/C1665/P1695. **Teacher FL has no I/Q/C/P at policy entry**; FL is grounded. Teacher RL I7 is not new policy evidence.

## First episode: fresh FL lift attempts, crossing, and placement

New FL I/Q attempts occur at **1725/1755**, **2572/2577**, and **2660/2664**; the first two qualifications are revoked on ground at **2422** and **2630**. Consequently C2821 / 23.508333 s follows the third attempt, not uninterrupted control since first-ever Q1755. P3025 / 25.208333 s is also new policy-sampled evidence. The first-ever Q timestamp remains 1755 by design; the full event list is needed to identify the current attempt.

First I has measured 3.283 mm lift gain, 10.448° whole-body actual joint motion and 17.680° command motion; first Q has 8.888 mm gain and 23.852° FL own joint response. By decision-end 1760, FL is AIR, load 0 valid, non-supporting, with FR/RL/RR supporting. The independent 0.5 s CoM projection toward receiving RR is **−4.157 mm**, instantaneous +0.000046 m/s: do not turn measured lifting into a claim that CoM necessarily moved toward RR. Whole-body response, CoM direction, and actual support are separate facts.

The third Q has 10.062 mm gain and only 3.123° own-joint motion: this is not a required large isolated FL-joint excursion. At tick 3024, FL is directly observed TOP, support true, load .336918 valid, front +29.530 mm and gap −.940 mm; P is latched at 3025. By P05→P06 at **3032**, FL is briefly AIR, +1.836 mm above top, front +33.583 mm, load 0, with three other supports. Historical placement is real but does not mean FL is still bearing at this boundary. At terminal 3140, FL is again TOP/support with load .113418 valid; all four measured supports exist, but that alone does not prevent a body safety failure.

RR also has an incidental I2490/Q2504 followed by ground revocation2578, no C/P. These are full-body exploration events during P05, not completion of the rear-leg course.

## Second episode: FL lift is not yet crossing

New FL I/Q at **1724/1729**, revoked2271; **2456/2464**, revoked2482; **2594/2605**, still historical Q at terminal. No FL C or P is recorded. End FL is AIR/non-supporting, front **−73.280 mm**, gap **−48.165 mm**, load 0 valid: initial lift/long AIR history does not demonstrate usable carry or completed crossing. FR/RL/RR support; short-window CoM toward RR is **−19.707 mm**, current projected velocity −.076668 m/s. RR briefly I2200/Q2209 then ground revocation2254; no rear C/P.

## Boundary and native control chain

Both P04→P05 transitions occur at tick1704 from physical proximity + role preparation, with valid FR placement; FL has no Q then. P04 is preparation, not a supposed completed FL lift or fixed posture. Episode 0 P05→P06 occurs at3032 with real historical FL placement. Neither ordinary transition sets terminal or discards return continuity.

Across **all 2537 physics ticks**, actual native target effect and own-phase request effect are verified; across all318 decision-end records the mask is 12/12 enabled, setter/mapping equality and previous-ack verification pass, no forbidden state writes are reported. Previous acknowledgement is consistently dispatch_tick−1, bootstrap remains180 within both physical episodes, and command clock is episode-physics clock+179. This offset is not a reset defect.

P04→P05 carries the previous full12 residual exactly to numerical precision (max step ≤1.34e−15°); the new nominal knee request changes −3.2°, wheels do not jump. P05→P06 carries residuals (≤3.56e−15°), while the new rolling owner legitimately changes wheel requests **+0.3 rad/s**. No residual/filter clearing or mapper restart is evidenced. Startup residual zero belongs to a new teacher-initialized episode, not to an ordinary phase change.

Episode 0 final nominal wheels are [.3,.3,.3,.3]; residual [−.139092,−.414737,−.652841,+.563961] gives target [.160908,−.114737,−.352841,.863961], with actual [.164712,−.420240,−.463641,.855999] rad/s. Episode 1 final targets [−.679184,.306713,.213756,.871765] have actual [−.679219,.201117,.258602,.795748]. These are real, uneven command/response directions; no single tracking-error or nominal-only causal explanation is established. The bounded evidence does not identify a hidden downlink defect.

## FALL trigger: what is and is not retained

Both final env verdicts are **FALL**, and supervisor reports `SAFETY_ABORT`, source `PHYSICAL_SAFETY`, reason “fall or physics explosion”. Physical evaluation is VALID; terminal observations are finite without fallback, not timeout or full success. Actual body linear/angular speeds are **.106727 m/s / .569233 rad/s** and **.212589 m/s / .457857 rad/s**, far below the explosion speed limits 5 / 20.

Production `_fall_and_explosion` (`isaac_fsm_backend.py:6502–6526`) defines FALL as **base_z < .015 m OR projected_gravity_b_z > −.30**; semantic supervisor enforces the same limits (`semantic_supervisor.py:476`). These specific two measured quantities and the backend `physics_guard_values` are **not persisted in the retained decision/terminal records**. There is no raw physical-observation stream in this training run. Therefore the available evidence establishes a physical FALL classification but cannot independently select which of those two branches fired or give its exact violating value. No CoM, BODY-bounds center, commanded joint value, or reward scalar is substituted for a missing root/IMU measurement.

Separately measured 0.5 s body displacement at the terminal is [+64.146,+5.483,−51.912] mm for episode0 and [+25.364,+.416,−20.814] mm for episode1: real descent is observed, but this does not by itself prove an absolute-height threshold. First unfinished task is P06 rear-approach preparation for episode0, and P05 FL crossing/placement for episode1; both end by safety abort, not merely incomplete collection.

## Optimizer context, not improvement evidence

Within this boundary, update1074 follows global141952 and update1075 follows142080, each20 optimizer steps, changed actor hash, finite nonzero gradients. Logged learning rates are 1e−5 and1.5e−5 (existing scheduler behavior, not changed here); gradient ranges are .9999998–1.4142136 and1.4142133–1.4142136. Thus **318 sampled decisions, 2 completed updates /40 optimizer steps, with62 samples beyond the second update boundary** at this snapshot. Cumulative published update position at142080 is1075 /21,500 optimizer steps; this report does not assert a newer checkpoint or credit later live samples.

Episode0 spans update1074 (C/P occur afterward); episode1 spans update1075. They are not two fixed-policy matched trials. Their events are valid on-policy samples, but neither cross-episode outcome nor the new FL placement proves learning improvement, stability superiority, natural-P01 completion, or a successful PPO checkpoint.

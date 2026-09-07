# C113920 final evaluation — FL crossed, but remained unloaded above the platform

Run `validation/20260907T0523186087751Z_gf4bfe2560bfd_b5694e9e04f14a45aa8af9de9841fe5b`, runtime f4bfe2560bfd, checkpoint113920, N1/seed2001/natural P01/deterministic policy mean. Final manifest lifecycle is **SUCCEEDED execution**, and root confirmed process exit0. Physical **task_success=false**: **673 decisions /5384 physics ticks /44.866666667s**, **P05 INCOMPLETE_CONTROLLER_BLOCKED**, stage age30s. The physical evaluator is valid, with null physical termination and no safety-failure reason. Optimizer updates during evaluation0; `window_ended_before_task_terminal=false`.

Scope: final JSON manifests, the completed673-row decision ledger, terminal raw observation5384, selected decision-end trajectory points and existing quality statistics. No checkpoint/PT load, repeated hash, Python/GPU/Isaac, next training data or production/config/test change. An overly broad diagnostic JSON print was truncated; only the subsequent compact selected fields are used below.

## First missing task and actual history

Source-phase counts P01–P13 are **[1,217,4,1,450,0,0,0,0,0,0,0,0]**. P02/P03/P04/P05 enter at ticks8/1744/1776/1784. All four ordinary handoffs have null termination, bootstrap allowed and time_outs=false. No P06–P13 sample occurs.

| Leg | Qualified lift tick | Front crossing tick | Placed tick | Terminal current contact / load |
|---|---:|---:|---:|---|
| FR | 48 | 1756 | 1770 | TOP / .419189504 |
| FL | 1865 | 2902 | **Absent** | **AIR / 0** |
| RL | Absent | Absent | Absent | GROUND / .485710921 |
| RR | Absent | Absent | Absent | GROUND / .095099575 |

FL's whole-body initial clearance is measured at1813/15.108333s (upward3.233150mm), followed by genuine qualified lift at1865/15.541667s (excursion50.536592mm, current above-top gap+.439172mm). Crossing2902 is retained. Neither is placement. RL's initial-clearance event12 is not hard qualification; RR/RL have no hard Q/C/P in the final history.

The first unfinished task is **actual loaded FL platform capture**, not an old entry requirement, missing qualification or missing historical crossing. Terminal `placed_FL=.85` is fractional predicate progress, not a placed bit. Entry remains valid. FL current frontdistance is **+1.866046mm**, clearance **+21.597487mm**, within_top_xy/top_geometry/lateral=true, but obstacle_pair_active=false, consecutive_TOP_samples0, load0 and support=false. Thus geometry alone is already satisfied while actual supporting contact is absent.

The runtime's hard placement code still requires crossed history and at least two consecutive measured loaded TOP samples; loaded TOP combines verified obstacle contact, current top geometry and nonnegative frontdistance. No pose imitation, historical support assumption or relaxed gate is needed to explain this noncompletion. The stored final AIR streak is3587 ticks ending5384, i.e.1798–5384 inclusive; this is evaluator substep evidence, not an independently rescanned full raw-contact history.

## Selected FL trajectory, not 120Hz extrema

All points below are post-cross decision endpoints, AIR/load0/unplaced. Extrema refer only to these sampled endpoints.

| Tick | FL front mm | Top gap mm | Top geometry | Interpretation |
|---:|---:|---:|:---:|---|
| 2904 | +4.095498 | +84.265327 | false | First sampled point after real crossing2902 |
| 2912 | +14.976088 | +89.516928 | false | Airborne above platform |
| 2936 | **+29.984597** | +80.120703 | false | Maximum sampled post-cross frontdistance |
| 2968 | +10.174516 | +24.453213 | true | First sampled post-cross top geometry, still no contact |
| 3000 | +9.866658 | +23.031031 | true | AIR/load0 |
| 4000 | +4.221622 | +21.197808 | true | AIR/load0 |
| 4104 | +2.400461 | **+20.940815** | true | Smallest sampled post-cross absolute gap, not touchdown |
| 5000 | +1.837674 | +22.139502 | true | AIR/load0 |
| 5384 | +1.866046 | +21.597487 | true | Deadline, no placement |

Of450 P05 endpoints,449 are AIR, one reports support, none has obstacle contact or true TOP. There are311 crossed/within-XY endpoints and302 top-geometry endpoints. Consequently many geometrically eligible observations exist without a measured placement; this is not evidence that placement was silently credited or that a contact threshold should change. The observed retreat from the maximum sampled frontdistance does not by itself identify a unique action, reward or mechanical cause.

## Terminal raw contact and support check

Raw5384 is finite with no listed data-quality issue. Verified FL wheel geometry has center `[.5231782198,.0810348094,.1215977594]`m and bottom z **.071597487302m**, against obstacle front x .521312173774m and top z .05m. These independently reproduce the positive frontdistance and21.597487mm top gap. Both exact FL ground/obstacle pairs are verified inactive, each force `[0,0,0]`N, with no obstacle contact point. Its measured wheel velocity is−.070880584rad/s and command−.070537980rad/s; these do not imply support.

Current support is explicitly **FR/RL/RR**, count3/valid, not FL. FR obstacle force vector is `[-1.060409188,.025519710,12.099638939]`N, recorded pair normal magnitude12.146043856N, with contact point `[.6193878651,−.3300130665,.0511977375]`m. RL/RR verified ground normals are14.073506355/2.755516529N. The support polygon's measured CoM projection is inside by24.824628605mm. Body collision detected/real_pair_active/persistent are allfalse, penetration0, reason `no exact base_link/obstacle contact`.

This is a real task deadline, not a recorded body collision, nonfinite fallback or hard-joint failure. Absence of these failures does not establish stability superiority or complete success.

## Native receipts, clocks and terminal return

The completed ledger reconciles673 eight-tick decisions to5384 physical ticks, with zero episode tick/time discontinuities. Existing summaries record **5384 verified native ticks /5384 actual-effect ticks /5380 own-phase-request-effect ticks**, excluding the four handoff holds from own-request counts. All673 endpoint checks for verification, setter/dispatch equality, actual mapping, same-tick counterfactual and independent previous-ACK reference pass. Four in-episode state-write totals are all0; physical-invalid snapshots0 and finite fallbacks0.

Native and episode clocks are distinct: the first decision ends at episode8/native187, the last at episode5384/native5563, the same179-tick offset. The inspected first/last native previous-ACK ticks186/5562 precede dispatch187/5563 by one; mapper feedback tick equals the corresponding dispatch and previous feedback sample equals previous ACK. This uses existing receipts and endpoint checks, not a new reconstruction of every native tensor or a full native-log scan.

Only the true terminal disables bootstrap, with time_outs=false, pre-potential .4117728063251514 and absorbing next-potential0. Its PBRS term is−2.058864031625757, terminal event−40, total reward−42.060892962183644. The evaluator's still-descriptive physical potential .41177546389328445 is not substituted for absorbing reward Phi. Whole evaluation return is−43.3083633786969; no optimizer or new PPO credit is generated by this evaluation.

## Existing quality diagnostics and interpretation limit

| Recorded diagnostic | Whole44.866667s | P05-only30s |
|---|---:|---:|
| Roll RMS rad | .132057458 | .045701604 |
| Pitch RMS rad | .100492381 | .036295600 |
| Body x angular-speed RMS rad/s | .085108205 | .071155400 |
| Body y angular-speed RMS rad/s | .068493432 | .039125730 |
| Angular-acceleration RMS rad/s² | 4.051394856 | 2.949093374 |

Whole-window angular-acceleration peak106.876237298rad/s²; nonfinite skipped ticks0. `fixed_quality_score=null`, `all_phases_sampled=false`; the existing P05-only score .4589909713 is not a full-task or paired-FSM score. These are incomplete-window diagnostics, not proof of causal improvement from the return-profile change.

Compared only as historical facts, C109824 also had FL Q/C without P, but ended outside top XY with gap+37.810539mm; C113920 ends geometrically over the top with gap+21.597487mm and still no contact. The saved weights differ and these are not controlled paired runs, so neither geometry difference nor phase counts establish gamma causality or stable improvement. No new gate, detector change, successful-video claim or parameter proposal is made.

Final conclusion: **FL genuinely crossed but did not load onto the platform before P05's deadline.** Current contact evidence, not historical Q/C or soft progress .85, determines the missing placement. Report complete; only the final evaluation section is appended to master, leaving block33 and all training counts unchanged.

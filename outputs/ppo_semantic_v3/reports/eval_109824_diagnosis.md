# C109824 final — FL crossed, retreated airborne, and never captured

Run `runs/ppo_semantic_v3/validation/20260907T0425049534146Z_gf1a9bbf650b1_c287e9492402483f946f83dd50cb40ca`, saved109824, runtime f1a9bbf650b1, N1/seed2001/naturalP01/fixed deterministic mean. Root confirmed the evaluation process exited. Final lifecycle **SUCCEEDED is execution completion, not task success**.

Actual **668 decisions /5344 physics ticks /44.533333333s**, P05 `INCOMPLETE_CONTROLLER_BLOCKED`, stage age30s. Taskfalse, physicalvalidtrue/null hard physical failure, optimizer updates0, window_ended_before_task_terminal=false. This is genuine task noncompletion, not a body collision, hard joint failure, recording error or interrupted evaluation.

Scope: final manifests, the completed668-row decision ledger, terminal raw observation5344, existing quality metrics and selected post-cross decision endpoints. Runtime predicate/config attribution uses **`git show f1a9bbf650b1:...`**, not current work-in-progress production files. No checkpoint/hash/PT/Python/GPU/Isaac, new training, full raw/native rescan, production/config/test edit or future result was used.

## First unfinished physical task

Source-phase counts P01–P13 are **[1,213,3,1,450,0,0,0,0,0,0,0,0]**. P02/P03/P04/P05 enter at8/1712/1736/1744 ticks. Every one of the four phase handoffs is nonterminal, bootstrap=true, time_outs=false; no P06–P13 samples occur.

| Leg | Hard qualified lift | Front crossing | Real placed | Terminal current state / load |
|---|---:|---:|---:|---|
| FR | 51 | 1724 | 1734 | TOP /.420857335 |
| FL | 1825 | 2889 | **Absent** | **AIR /0** |
| RR | — | — | — | GROUND /.074029053 |
| RL | — | — | — | GROUND /.505113612 |

First missing task is **current FL platform capture/placement**, not qualification or historical crossing. Entry is still valid; `placed_FL=.7` retains the qualified-lift/crossing components but not current top geometry/contact. At the terminal FL frontdistance is **−69.295879771mm**, topgap **+37.810539277mm**, within_top_xy=false, top_geometry=false, obstacle pair inactive, consecutive_TOP_samples0. Its Q/C history is not erased by airborne retreat, but neither is it a placed wheel.

All450 P05 decision endpoints are FL AIR, trueTOP0 and obstacle-active0. The final evaluator records3604 consecutiveAIR ticks ending5344, i.e.1741–5344 inclusive. This recorded substep streak is distinguished from an independently scanned raw-contact history.

The runtime's unchanged hard placement path requires crossed history plus at least2 consecutive measured loaded TOP samples: verified obstacle contact, current top rectangle/gap geometry and nonnegative frontdistance. The recorded terminal state lacks these conditions. Existing top-gap bounds are−15..+25mm and XY measurement tolerance5mm; no reference joint angle, fixed historical entry, proposed return-profile behavior or new gate is substituted for actual capture.

## Selected post-cross trajectory

These extrema are over **post-cross decision endpoints only**, not full120Hz extrema. All listed points are AIR/load0 without obstacle contact.

| Tick | FL front mm | FL gap mm | Within top XY | Meaning |
|---:|---:|---:|:---:|---|
| 2896 | **+.981334** | **+90.758955** | true | Maximum sampled post-cross frontdistance and gap |
| 2904 | −2.118942 | +71.556648 | true | First sampled retreat behind front; still within5mm tolerance |
| 2936 | −19.341167 | **+34.568970** | false | Closest sampled post-cross absolute topgap; already outside XY |
| 3000 | −18.572983 | +37.639966 | false | Airborne, no capture |
| 4000 | −42.598751 | +38.231691 | false | Further retreat |
| 5344 | **−69.295880** | **+37.810539** | false | P05 deadline, no placement |

Thus the small historical crossing did not persist as current arrival over the platform. Even the closest sampled post-cross height is9.568970mm above the existing+25mm geometric bound, and that point is outside XY with no contact. These measurements do not identify a unique controller, action, reward or physical cause.

## Terminal raw safety and support

Raw5344 independently has all_finite=true and verified FL geometry. FL bottomz=.087810539277m above obstacle top.05m matches the recorded gap; both exact ground/obstacle pairs are verified inactive with0N. FR's obstacle pair is verified active with normal12.220765495N; RL/RR ground normals are14.667381287/2.149639845N. Current support is FR/RL/RR, count3, valid; the measured robot-CoM projection is inside its support polygon by23.543145773mm. FL supplies no support.

Body collision detected/real_pair_active/persistent are allfalse, penetration0, reason `no exact base_link/obstacle contact`. The physical task evaluator is valid with no hard physical termination; no finite-observation fallback or skipped nonfinite quality tick is recorded. These facts distinguish this task deadline from a safety abort, but do not turn it into full success or a stability advantage.

## Clock/native completeness and existing quality measurements

All668 decisions contain8 physical ticks, summing5344 without gaps; recorded simulation time matches tick/120. Existing per-decision native summaries total **5344 verified/effect ticks**, own-phase effect5340 (four handoff holds excluded). All endpoint setter/dispatch, actual-mapping, same-tick counterfactual and independent previous-ACK reference checks pass. Four in-episode state-write totals0, invalid physical snapshots0, clock discrepancies0, finite fallbacks0. This is a reconciliation of existing receipts, not a new independent native-target reconstruction or full native-log scan. Evaluation optimizer updates remain0.

Existing `quality_metrics` use actual physical ticks excluding video padding:

| Recorded diagnostic | Whole44.533333s window | P05-only30s window |
|---|---:|---:|
| Roll RMS rad | .128124142 | .043530859 |
| Pitch RMS rad | .103887734 | .059450296 |
| Body x angular-speed RMS rad/s | .083892281 | .075333779 |
| Body y angular-speed RMS rad/s | .065767172 | .047403508 |
| Angular-acceleration RMS rad/s² | 4.320426690 | 3.164561746 |

Whole-window angular-acceleration peak is126.811845455rad/s². `fixed_quality_score=null`, `all_phases_sampled=false`, skipped nonfinite ticks0. These are diagnostics for an incomplete, non-paired physical window; they are **not superior-to-FSM results or evidence of causal improvement**. Unsampled phases are not scored as perfect. No new video, suffix/full task success or promoted checkpoint is claimed.

Final conclusion: FL earned genuine Q/C but never made loaded platform capture, then retreated airborne until P05's deadline. No execution mismatch is established in this scoped check. C107264 and all other historical outcomes, including old A, remain unchanged; the current return-profile work in progress receives no rollout or success credit. Report and master append completed; writing stopped and master ownership released.

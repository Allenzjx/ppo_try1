# Continuous whole-body Residual PPO v3 — live implementation record

Status: real training and saved-checkpoint evaluation executed; no full-task trained success is claimed.

Current completed-boundary summary: **103,168 lifetime optimized decisions /771 PPO updates /15,420 optimizer steps**. Twenty-eight finalized blocks add **93,056 decisions /727 PPO updates /14,540 optimizer steps** since verified origin10,112/44/880. Latest42b91e8 tracking-reference new-MDP naturalP01 block from101,376 finalized **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY /exit0** after actual **1,792 decisions /14 updates /280 optimizer steps**; original planned4,096 leaves2,304 unconsumed, rounding0, training-loop wall723.9258038001135s. Final requested_policy_decisions1792 is the finalized stop amount, not a rewrite of planned4096. Immutable `checkpoints/history/checkpoint_step_000103168.pt` records roundtriptrue and CP SHA `005d8794fc4fbae5616c478321ddd1d0e2d8f3082fd94caec22d3acbb6a6c0c3`; pointer/sidecar/source receipts agree without rehashing. Spending full_episode45,312/phase_suffix47,744/smoke0/origin10,112. One completed957-decision P06 incomplete episode plus835 optimized decisions in an unfinished P06 tail; no teacher/prefix budget, rearhardQ/C/P or P07–13 samples. All14,336 native ticks verify/effect, own14,326/fourstatewrites0;10phase changes nonterminal. All14 actor updates change with finite nonzero gradients/continuous receipts. Early stop was requested at a complete update before a separately versioned workspace-potential-density revision, not an optimizer failure or task-success gate. **No suffix or full-task success is recorded.** Every earlier block, stop request, bounded report and failed A/B/C outcome remains intact.

Latest completed deterministic natural-P01 evaluation is **checkpoint103,168: P05 INCOMPLETE_CONTROLLER_BLOCKED, not successful**. Run `runs/ppo_semantic_v3/validation/20260907T0147389603120Z_g42b91e857a0a_a502f3cc872645a496b2fa1d6968b077` reloads the fixed mean under42b91e8/N1seed2001/naturalP01, without teacher or optimizer updates. Actual **660 decisions /5,273 ticks /43.941666667s**, P05age30.008333s; execution SUCCEEDED/exit0, taskfalse/windowEndedfalse, physicalvalidtrue/nullhardfailure. First missing is actual FLcapture (`placed_FL=.85`): FLQ1754/C2791/noP, finalAIR/load0/gap+16.635617mm/front+5.441311mm. FRQ48/C1644/P1659/currentTOP/load.347264298; rears nohardQ/C/P, currentGROUND. All5273native ticks verify/effect, own5269/fourstatewrites0; last decision1tick explains660×8−7. One full native whitelist scan confirms reference_used on1306physicalticks/2729channel-samples, despite only1of660decision-end snapshots showing use; it is not valid to infer inactivity from endpoints. `eval_103168_diagnosis.md` records the actual FLhip feedback/float32 dispatch example and limits its interpretation. C101376/P06, C93184/P05, oldA and every earlier result remain preserved. No causal control improvement, full/suffix success, paired stability benefit or successfulvideo is claimed.

The current completed training/evaluation runtime is **42b91e857a0a2da256dee85e8092642ca8e64aeb**. Block28 introduced previous-ACK requested-servo tracking reference through explicit NewMdpWarmStart from101,376, preserving learned network/identity-normalizer/RNG/origin/budgets with freshAdam and rollout. `tracking_reference_block_103168.md` records the complete-update stop after1792actual/2304unconsumed; implementation and first128 reports remain bounded history. Latest completed full evaluation is now C103,168/P05 INCOMPLETE under42b91e8. The actual native feedback use is established, but no direct old-mode paired physics or causal stability advantage follows. The separately developed workspace-potential-density revision has no test/training/outcome credit in this ledger. Source A/frozen machinery and every prior failed/incomplete result retain their status.

## Current twenty-eight-run sampling and outcome summary

All previous blocks and reports remain preserved. Block27 retains `p01_block_101376.md` and its six-completed-episode audit; finalized block28 is `tracking_reference_block_103168.md`, with the first128 report and stop request retained. Latest completed full evaluation is C103,168 (`eval_103168_diagnosis.md`), P05 INCOMPLETE/physicalvalid/nullhardfailure/0optimizerupdates. C101376/P06, C93184/P05 and every older A/B/C outcome remain unchanged. The next planned production revision receives no future sample or outcome credit. The twenty-eight finalized blocks contribute:

| Training start UTC / requested curriculum | Optimized decisions | PPO updates | Optimizer steps | Actual endpoint |
|---|---:|---:|---:|---|
| 03:01:33 / P06 | 1,536 | 12 | 240 | Checkpoint11,648; stopped at verified update |
| 03:36:31 / P06 | 512 | 4 | 80 | Checkpoint12,160; requested block finished |
| 03:44:55 / P01 | 512 | 4 | 80 | Checkpoint12,672; requested block finished |
| 04:06:10 / P06 | 2,176 | 17 | 340 | Checkpoint14,848; stopped at verified update |
| 04:31:06 / P10 | 1,920 | 15 | 300 | Checkpoint16,768; requested block finished |
| 05:15:50 / P09 | 1,024 | 8 | 160 | Checkpoint17,792; stopped at verified update;1,024 unspent |
| 05:45:26 / P01 | 4,096 | 32 | 640 | Checkpoint21,888; requested block finished;3 incomplete plus unfinished tail |
| 06:36:17 / P06 | 6,144 | 48 | 960 | Checkpoint28,032; requested block finished;7 incomplete plus unfinished P13 tail |
| 07:48:08 / P10 | 2,048 | 16 | 320 | Checkpoint30,080; requested block finished;2 incomplete plus unfinished tail |
| 08:19:49 / P01 | 8,192 | 64 | 1,280 | Checkpoint38,272; requested block finished;7 incomplete plus unfinished tail |
| 09:46:28 / P06, policy-distribution boundary | 8,192 | 64 | 1,280 | Checkpoint46,464; requested block finished;10 incomplete plus unfinished P06 tail |
| 11:33:23 / natural P01, workspace-shaping boundary | 4,096 | 32 | 640 | Checkpoint50,560; requested block finished;3 P09 incomplete plus unfinished P09 tail |
| 12:11:30 / P10, same-MDP exact resume | 2,048 | 16 | 320 | Checkpoint52,608; requested block finished;2 P13 incomplete plus unfinished P13 tail |
| 13:08:47 / P06, nominal-geometry new-MDP boundary | 2,048 | 16 | 320 | Checkpoint54,656; requested block finished;P09/P12 incomplete plus unfinished P09 tail |
| 13:48:49 / natural P01, capture-retention new-MDP boundary | 8,192 | 64 | 1,280 | Checkpoint62,848; requested block finished;6 incomplete/3 BODY_COLLISION plus unfinished P06 tail |
| 14:53:01 / P10, same-MDP exact resume | 4,096 | 32 | 640 | Checkpoint66,944; requested block finished;3 P13/1 P12 incomplete plus unfinished P13 tail |
| 16:11:14 / P06, continuous-stop new-MDP boundary | 1,280 | 10 | 200 | Checkpoint68,224; stopped at verified update;2,816 unspent;2 P06 incomplete plus unfinished P06 tail |
| 16:50:13 / natural P01, P06 measured-tail new-MDP boundary | 4,096 | 32 | 640 | Checkpoint72,320; requested block finished;1 P09 collision/3 P06 incomplete plus unfinished P06 tail |
| 17:38:50 / P06 offset240, capture-approach new-MDP boundary | 768 | 6 | 120 | Checkpoint73,088; stopped at verified update;1,280 unconsumed;2 P06 incomplete plus46-decision unfinished tail |
| 18:05:56 / P07 offset0, same-MDP exact resume | 3,328 | 26 | 520 | Checkpoint76,416; stopped at verified update;768 unconsumed;1 P09 collision/7 P09 incomplete plus85-decision unfinished tail |
| 19:28:29 / P07 offset0, post-mapper-headroom new-MDP boundary | 2,048 | 16 | 320 | Checkpoint78,464; requested block finished;4 P09 incomplete plus240-decision unfinished tail |
| 20:07:18 / naturalP01, same-MDP exact resume | 4,096 | 32 | 640 | Checkpoint82,560; requested block finished;1 P09 incomplete/4 P09 collisions plus392-decision unfinished P06 tail |
| 20:46:00 / P10offset0, same-MDP exact resume | 2,432 | 19 | 380 | Checkpoint84,992; stopped at verified update;1,664 unconsumed;1 P13/3 P12 incomplete plus103-decision unfinished P13 tail |
| 22:11:50 / P06offset0, frozen-checkpoint-prefix new-MDP boundary | 2,048 | 16 | 320 | Checkpoint87,040; requested block finished;6 P09 collisions plus109-decision unfinished P06 tail |
| 23:01:40 / naturalP01, front-wheel-range new-MDP boundary | 2,048 | 16 | 320 | Checkpoint89,088; requested block finished;2 P09 collisions plus467-decision unfinished P06 tail |
| 23:26:10 / P06offset0, same-MDP frozen-checkpoint-prefix resume | 4,096 | 32 | 640 | Checkpoint93,184; requested block finished;4 collisions/3 incomplete plus466-decision unfinished P09 tail |
| 00:17:59 / naturalP01, same-MDP ordinary resume | 8,192 | 64 | 1,280 | Checkpoint101,376; requested block finished;5 P06 incomplete/1 P09 incomplete/2 P09 collisions plus499-decision unfinished P06 tail |
| 01:34:08 / naturalP01, tracking-reference new-MDP diagnostic | 1,792 | 14 | 280 | Checkpoint103,168; verified-update stop,2,304 planned decisions unconsumed;1 P06 incomplete plus835-decision unfinished P06 tail |
| **Finalized twenty-eight-block total** | **93,056** | **727** | **14,540** | **No suffix or full-task success** |

Current twenty-eight-block phase credits exclude all teacher/checkpoint initialization actions; block28 has only natural-P01 policy credit:

| Policy-request phase | First twenty-seven blocks | Latest tracking-reference block | All twenty-eight blocks |
|---|---:|---:|---:|
| P01 |53 |2 |55 |
| P02 |9,403 |394 |9,797 |
| P03 |190 |8 |198 |
| P04 |51 |2 |53 |
| P05 |7,369 |308 |7,677 |
| P06 |33,326 |1,078 |34,404 |
| P07 |130 |0 |130 |
| P08 |97 |0 |97 |
| P09 |22,479 |0 |22,479 |
| P10 |40 |0 |40 |
| P11 |37 |0 |37 |
| P12 |7,361 |0 |7,361 |
| P13 |10,728 |0 |10,728 |
| **Total** |**91,264** |**1,792** |**93,056** |

The following twenty-seven-block phase ledger is preserved history, not an alternate current total:

| Policy-request phase | First twenty-six blocks | Latest natural-P01 block | All twenty-seven blocks |
|---|---:|---:|---:|
| P01 |44 |9 |53 |
| P02 |7,642 |1,761 |9,403 |
| P03 |154 |36 |190 |
| P04 |41 |10 |51 |
| P05 |6,034 |1,335 |7,369 |
| P06 |28,768 |4,558 |33,326 |
| P07 |121 |9 |130 |
| P08 |93 |4 |97 |
| P09 |22,009 |470 |22,479 |
| P10 |40 |0 |40 |
| P11 |37 |0 |37 |
| P12 |7,361 |0 |7,361 |
| P13 |10,728 |0 |10,728 |
| **Total** |**83,072** |**8,192** |**91,264** |

The following twenty-six-block phase ledger is preserved history, not an alternate current total:

| Policy-request phase | First twenty-five blocks | Latest same-MDP P06 prefix block | All twenty-six blocks |
|---|---:|---:|---:|
| P01 |44 |0 |44 |
| P02 |7,642 |0 |7,642 |
| P03 |154 |0 |154 |
| P04 |41 |0 |41 |
| P05 |6,034 |0 |6,034 |
| P06 |25,743 |3,025 |28,768 |
| P07 |101 |20 |121 |
| P08 |85 |8 |93 |
| P09 |21,420 |589 |22,009 |
| P10 |39 |1 |40 |
| P11 |34 |3 |37 |
| P12 |6,911 |450 |7,361 |
| P13 |10,728 |0 |10,728 |
| **Total** |**78,976** |**4,096** |**83,072** |

The following twenty-five-block phase ledger is preserved history, not an alternate current total:

| Policy-request phase | First twenty-four blocks | Latest natural-P01 front-wheel block | All twenty-five blocks |
|---|---:|---:|---:|
| P01 |41 |3 |44 |
| P02 |7,061 |581 |7,642 |
| P03 |143 |11 |154 |
| P04 |38 |3 |41 |
| P05 |5,585 |449 |6,034 |
| P06 |24,777 |966 |25,743 |
| P07 |86 |15 |101 |
| P08 |83 |2 |85 |
| P09 |21,402 |18 |21,420 |
| P10 |39 |0 |39 |
| P11 |34 |0 |34 |
| P12 |6,911 |0 |6,911 |
| P13 |10,728 |0 |10,728 |
| **Total** |**76,928** |**2,048** |**78,976** |

The following twenty-four-block phase ledger is preserved history, not an alternate current total:

| Policy-request phase | First twenty-three blocks | Latest checkpoint-prefix P06 block | All twenty-four blocks |
|---|---:|---:|---:|
| P01 |41 |0 |41 |
| P02 |7,061 |0 |7,061 |
| P03 |143 |0 |143 |
| P04 |38 |0 |38 |
| P05 |5,585 |0 |5,585 |
| P06 |22,806 |1,971 |24,777 |
| P07 |76 |10 |86 |
| P08 |77 |6 |83 |
| P09 |21,341 |61 |21,402 |
| P10 |39 |0 |39 |
| P11 |34 |0 |34 |
| P12 |6,911 |0 |6,911 |
| P13 |10,728 |0 |10,728 |
| **Total** |**74,880** |**2,048** |**76,928** |

The following twenty-three-block phase ledger is preserved history, not an alternate current total:

| Policy-request phase | First twenty-two blocks | Latest P10/headroom block | All twenty-three blocks |
|---|---:|---:|---:|
| P01 |41 |0 |41 |
| P02 |7,061 |0 |7,061 |
| P03 |143 |0 |143 |
| P04 |38 |0 |38 |
| P05 |5,585 |0 |5,585 |
| P06 |22,806 |0 |22,806 |
| P07 |76 |0 |76 |
| P08 |77 |0 |77 |
| P09 |21,341 |0 |21,341 |
| P10 |34 |5 |39 |
| P11 |29 |5 |34 |
| P12 |5,424 |1,487 |6,911 |
| P13 |9,793 |935 |10,728 |
| **Total** |**72,448** |**2,432** |**74,880** |

The following twenty-two-block phase ledger is preserved history,not an alternate current total:

| Policy-request phase | First twenty-one blocks | Latest naturalP01 block | All twenty-two blocks |
|---|---:|---:|---:|
| P01 |35 |6 |41 |
| P02 |5,959 |1,102 |7,061 |
| P03 |121 |22 |143 |
| P04 |32 |6 |38 |
| P05 |4,693 |892 |5,585 |
| P06 |21,236 |1,570 |22,806 |
| P07 |71 |5 |76 |
| P08 |72 |5 |77 |
| P09 |20,853 |488 |21,341 |
| P10 |34 |0 |34 |
| P11 |29 |0 |29 |
| P12 |5,424 |0 |5,424 |
| P13 |9,793 |0 |9,793 |
| **Total** |**68,352** |**4,096** |**72,448** |

The following twenty-one-block phase ledger is preserved history,not an alternate current total:

| Policy-request phase | First twenty blocks | Latest P07/headroom block | All twenty-one blocks |
|---|---:|---:|---:|
| P01 |35 |0 |35 |
| P02 |5,959 |0 |5,959 |
| P03 |121 |0 |121 |
| P04 |32 |0 |32 |
| P05 |4,693 |0 |4,693 |
| P06 |21,236 |0 |21,236 |
| P07 |66 |5 |71 |
| P08 |67 |5 |72 |
| P09 |18,815 |2,038 |20,853 |
| P10 |34 |0 |34 |
| P11 |29 |0 |29 |
| P12 |5,424 |0 |5,424 |
| P13 |9,793 |0 |9,793 |
| **Total** |**66,304** |**2,048** |**68,352** |

The following twenty-block phase ledger is preserved history,not an alternate current total:

| Policy-request phase | First nineteen blocks | Latest P07/offset0 block | All twenty blocks |
|---|---:|---:|---:|
| P01 |35 |0 |35 |
| P02 |5,959 |0 |5,959 |
| P03 |121 |0 |121 |
| P04 |32 |0 |32 |
| P05 |4,693 |0 |4,693 |
| P06 |21,236 |0 |21,236 |
| P07 |57 |9 |66 |
| P08 |57 |10 |67 |
| P09 |15,506 |3,309 |18,815 |
| P10 |34 |0 |34 |
| P11 |29 |0 |29 |
| P12 |5,424 |0 |5,424 |
| P13 |9,793 |0 |9,793 |
| **Total** |**62,976** |**3,328** |**66,304** |

The following nineteen-block phase ledger is preserved history,not an alternate current total:

| Policy-request phase | First eighteen blocks | Latest P06/offset240 block | All nineteen blocks |
|---|---:|---:|---:|
| P01 |35 |0 |35 |
| P02 |5,959 |0 |5,959 |
| P03 |121 |0 |121 |
| P04 |32 |0 |32 |
| P05 |4,693 |0 |4,693 |
| P06 |20,468 |768 |21,236 |
| P07 |57 |0 |57 |
| P08 |57 |0 |57 |
| P09 |15,506 |0 |15,506 |
| P10 |34 |0 |34 |
| P11 |29 |0 |29 |
| P12 |5,424 |0 |5,424 |
| P13 |9,793 |0 |9,793 |
| **Total** |**62,208** |**768** |**62,976** |

The following eighteen-block phase ledger is preserved history,not an alternate current total:

| Policy-request phase | First seventeen blocks | Latest natural-P01 block | All eighteen blocks |
|---|---:|---:|---:|
| P01 |30 |5 |35 |
| P02 |5,065 |894 |5,959 |
| P03 |103 |18 |121 |
| P04 |27 |5 |32 |
| P05 |3,953 |740 |4,693 |
| P06 |18,046 |2,422 |20,468 |
| P07 |56 |1 |57 |
| P08 |56 |1 |57 |
| P09 |15,496 |10 |15,506 |
| P10 |34 |0 |34 |
| P11 |29 |0 |29 |
| P12 |5,424 |0 |5,424 |
| P13 |9,793 |0 |9,793 |
| **Total** |**58,112** |**4,096** |**62,208** |

The following seventeen-block phase ledger is preserved history,not an alternate current total:

| Policy-request phase | First sixteen blocks | Latest P06 block | All seventeen blocks |
|---|---:|---:|---:|
| P01 |30 |0 |30 |
| P02 |5,065 |0 |5,065 |
| P03 |103 |0 |103 |
| P04 |27 |0 |27 |
| P05 |3,953 |0 |3,953 |
| P06 |16,766 |1,280 |18,046 |
| P07 |56 |0 |56 |
| P08 |56 |0 |56 |
| P09 |15,496 |0 |15,496 |
| P10 |34 |0 |34 |
| P11 |29 |0 |29 |
| P12 |5,424 |0 |5,424 |
| P13 |9,793 |0 |9,793 |
| **Total** |**56,832** |**1,280** |**58,112** |

The following sixteen-block phase ledger is preserved history, not an alternate current total:

| Policy-request phase | First fifteen blocks | Latest P10 block | All sixteen blocks |
|---|---:|---:|---:|
| P01 |30 |0 |30 |
| P02 |5,065 |0 |5,065 |
| P03 |103 |0 |103 |
| P04 |27 |0 |27 |
| P05 |3,953 |0 |3,953 |
| P06 |16,766 |0 |16,766 |
| P07 |56 |0 |56 |
| P08 |56 |0 |56 |
| P09 |15,496 |0 |15,496 |
| P10 |29 |5 |34 |
| P11 |24 |5 |29 |
| P12 |4,718 |706 |5,424 |
| P13 |6,413 |3,380 |9,793 |
| **Total** |**52,736** |**4,096** |**56,832** |

The following fifteen-block phase ledger is preserved history, not an alternate current total; its last natural-P01 block has no teacher:

| Policy-request phase | First fourteen blocks | Capture-retention P01 block | All fifteen blocks |
|---|---:|---:|---:|
| P01 |19 |11 |30 |
| P02 |3,295 |1,770 |5,065 |
| P03 |67 |36 |103 |
| P04 |17 |10 |27 |
| P05 |2,474 |1,479 |3,953 |
| P06 |12,910 |3,856 |16,766 |
| P07 |48 |8 |56 |
| P08 |51 |5 |56 |
| P09 |14,932 |564 |15,496 |
| P10 |28 |1 |29 |
| P11 |22 |2 |24 |
| P12 |4,268 |450 |4,718 |
| P13 |6,413 |0 |6,413 |
| **Total** |**44,544** |**8,192** |**52,736** |

The following fourteen-block phase ledger is preserved history, not an alternate current total:

| Policy-request phase | First thirteen blocks | Latest geometry P06 block | All fourteen blocks |
|---|---:|---:|---:|
| P01 |19 |0 |19 |
| P02 |3,295 |0 |3,295 |
| P03 |67 |0 |67 |
| P04 |17 |0 |17 |
| P05 |2,474 |0 |2,474 |
| P06 |11,938 |972 |12,910 |
| P07 |45 |3 |48 |
| P08 |47 |4 |51 |
| P09 |14,325 |607 |14,932 |
| P10 |17 |11 |28 |
| P11 |21 |1 |22 |
| P12 |3,818 |450 |4,268 |
| P13 |6,413 |0 |6,413 |
| **Total** |**42,496** |**2,048** |**44,544** |

The following thirteen-block phase ledger is preserved history, not an alternate current total:

| Policy-request phase | First twelve blocks | Latest P10 block | All thirteen blocks |
|---|---:|---:|---:|
| P01 |19 |0 |19 |
| P02 |3,295 |0 |3,295 |
| P03 |67 |0 |67 |
| P04 |17 |0 |17 |
| P05 |2,474 |0 |2,474 |
| P06 |11,938 |0 |11,938 |
| P07 |45 |0 |45 |
| P08 |47 |0 |47 |
| P09 |14,325 |0 |14,325 |
| P10 |14 |3 |17 |
| P11 |14 |7 |21 |
| P12 |3,623 |195 |3,818 |
| P13 |4,570 |1,843 |6,413 |
| **Total** |**40,448** |**2,048** |**42,496** |

The following twelve-block phase ledger is preserved history, not an alternate current total. Its latest natural-P01 block contributed no teacher data:

| Policy-request phase | First eleven blocks | New natural-P01 block | All twelve blocks |
|---|---:|---:|---:|
| P01 | 15 | 4 | 19 |
| P02 | 2,564 | 731 | 3,295 |
| P03 | 51 | 16 | 67 |
| P04 | 13 | 4 | 17 |
| P05 | 1,887 | 587 | 2,474 |
| P06 | 10,563 | 1,375 | 11,938 |
| P07 | 39 | 6 | 45 |
| P08 | 42 | 5 | 47 |
| P09 | 12,957 | 1,368 | 14,325 |
| P10 | 14 | 0 | 14 |
| P11 | 14 | 0 | 14 |
| P12 | 3,623 | 0 | 3,623 |
| P13 | 4,570 | 0 | 4,570 |
| **Total** | **36,352** | **4,096** | **40,448** |

The following eleven-block phase ledger is preserved history, not an alternate current total:

| Policy-request phase | First nine runs | Latest natural-P01 block | First ten runs | Policy P06 block | All eleven blocks |
|---|---:|---:|---:|---:|---:|
| P01 | 6 | 9 | 15 | 0 | 15 |
| P02 | 1,068 | 1,496 | 2,564 | 0 | 2,564 |
| P03 | 20 | 31 | 51 | 0 | 51 |
| P04 | 5 | 8 | 13 | 0 | 13 |
| P05 | 720 | 1,167 | 1,887 | 0 | 1,887 |
| P06 | 5,218 | 2,120 | 7,338 | 3,225 | 10,563 |
| P07 | 21 | 8 | 29 | 10 | 39 |
| P08 | 21 | 9 | 30 | 12 | 42 |
| P09 | 7,380 | 2,892 | 10,272 | 2,685 | 12,957 |
| P10 | 8 | 1 | 9 | 5 | 14 |
| P11 | 8 | 1 | 9 | 5 | 14 |
| P12 | 923 | 450 | 1,373 | 2,250 | 3,623 |
| P13 | 4,570 | 0 | 4,570 | 0 | 4,570 |
| **Total** | **19,968** | **8,192** | **28,160** | **8,192** | **36,352** |

The following phase ledger preserves the preceding nine-run boundary, not an alternate current total:

| Policy-request phase | First seven runs | P06 lift-credit block | First eight runs | Latest P10 block | All nine runs |
|---|---:|---:|---:|---:|---:|
| P01 | 6 | 0 | 6 | 0 | 6 |
| P02 | 1,068 | 0 | 1,068 | 0 | 1,068 |
| P03 | 20 | 0 | 20 | 0 | 20 |
| P04 | 5 | 0 | 5 | 0 | 5 |
| P05 | 720 | 0 | 720 | 0 | 720 |
| P06 | 3,115 | 2,103 | 5,218 | 0 | 5,218 |
| P07 | 12 | 9 | 21 | 0 | 21 |
| P08 | 12 | 9 | 21 | 0 | 21 |
| P09 | 4,898 | 2,482 | 7,380 | 0 | 7,380 |
| P10 | 2 | 3 | 5 | 3 | 8 |
| P11 | 2 | 3 | 5 | 3 | 8 |
| P12 | 137 | 585 | 722 | 201 | 923 |
| P13 | 1,779 | 950 | 2,729 | 1,841 | 4,570 |
| **Total** | **11,776** | **6,144** | **17,920** | **2,048** | **19,968** |

The earlier first-seven-run phase ledger and P10 details are retained below as history, not alternate current totals.

The P10 block contains contiguous credited decisions14,849–16,768 and PPO updates82–96. Two real teacher prefixes each contribute948 behavior decisions /7,584 native ticks, reaching actual P10 at63.2s with136.8s remaining. Their combined1,896 decisions /15,168 ticks are excluded from PPO storage and totals. Credited suffix physics totals15,360 ticks. Frozen teacher achievement of the preceding FR/FL/RR sequence is not counted as PPO learning those phases.

| Policy-request phase | P10 block | P09 block | Preserved first-six-run subtotal | New P01 block | All seven runs |
|---|---:|---:|---:|---:|---:|
| P01 | 0 | 0 | 2 | 4 | 6 |
| P02 | 0 | 0 | 366 | 702 | 1,068 |
| P03 | 0 | 0 | 4 | 16 | 20 |
| P04 | 0 | 0 | 1 | 4 | 5 |
| P05 | 0 | 0 | 139 | 581 | 720 |
| P06 | 0 | 0 | 2,045 | 1,070 | 3,115 |
| P07 | 0 | 0 | 8 | 4 | 12 |
| P08 | 0 | 0 | 8 | 4 | 12 |
| P09 | 0 | 1,024 | 3,187 | 1,711 | 4,898 |
| P10 | 2 | 0 | 2 | 0 | 2 |
| P11 | 2 | 0 | 2 | 0 | 2 |
| P12 | 137 | 0 | 137 | 0 | 137 |
| P13 | 1,779 | 0 | 1,779 | 0 | 1,779 |
| **Total** | **1,920** | **1,024** | **7,680** | **4,096** | **11,776** |

P10 block episode0 supplied970 policy decisions (P10=1/P11=1/P12=68/P13=900). It terminated at127.866667s with `INCOMPLETE_CONTROLLER_BLOCKED`, P13 age60s, despite72.133333s remaining on the whole-task clock. Episode1 supplied950 decisions (P10=1/P11=1/P12=69/P13=879) and was still nonterminal in P13 at126.533333s when the optimizer block ended. The latter is not an additional completed failure or success. Both have actual RL qualified lift, crossing and placement **after** policy credit began: episode0 ticks7,910/8,058/8,144; episode1 ticks7,730/8,072/8,151. These are current-policy suffix events, not physical final-stop success, and do not establish fresh-P01 RR preparation.

No P13 decision-end sample meets all final-control conditions: `final_controlled=true` is0/1,779 and recorded stable duration is0s throughout. Wheel-command≤0.02rad/s fails all1,779 samples; measured wheel speed fails852, body linear speed1,176 and angular speed172. Region fails4 samples; support fails none. There are327 samples where only command fails among those conditions, but many other samples have real measured-motion violations as well. Thus neither suffix failure is explained solely by small command noise or only by a running nominal tail. No success threshold or old result is changed by this report.

### Previous natural-P01 block finalized:4,096 actual; no task success

`runs/ppo_semantic_v3/train/20260906T0545269489491Z_gcae1d6e7cfdd_1813c27a4d0e408aab8fde1f0c4010c6` is finalized `SUCCEEDED`, meaning its optimizer budget completed, not that a physical task succeeded. Both requested and actual budgets are4,096, with0 unconsumed and0 rounding overrun. Its4,096 audit rows are exactly contiguous globals17,793–21,888, with32 new PPO updates /640 optimizer steps. Every update records actor change and finite nonzero gradients. Manifest telemetry agrees with the phase counts above, with32,768 credited physics ticks, three completed episodes and `success_count=0`.

All four current-policy starts are natural P01 at decision1/tick8, with teacher offset0 and no prefix evidence file. The actual P01 execution branch uses no teacher roll-in; core counters equal PPO counters4,096/32,768, rather than hiding extra preparation inside the credit. All native8-tick summaries verify and report zero in-episode state writes. P07/P08 each supply only four decisions, correctly reflecting one-decision transitions per episode, not extended training in those labels. No P10–P13 decisions occur in this run.

| Natural-P01 episode | Credited global range | Decisions | Actual outcome |
|---|---|---:|---|
| 0 | 17,793–18,819 | 1,027 | P09 incomplete at tick8,216/68.466667s |
| 1 | 18,820–19,864 | 1,045 | P09 incomplete at tick8,360/69.666667s |
| 2 | 19,865–20,939 | 1,075 | P09 incomplete at tick8,600/71.666667s |
| 3 / unfinished tail | 20,940–21,888 | 949 | Nonterminal P09 at tick7,592/63.266667s |

The three terminals report `INCOMPLETE_CONTROLLER_BLOCKED`, P09 stage age about30s, no task/full-task success and no terminal bootstrap. The949-decision tail is optimized data, not an additional terminal failure or success. Its RR obtains genuine `qualified_measured_upward_lift` at tick5,032/41.933333s (global21,568): measured upward excursion46.205mm, joint motion16.105deg, clearance+0.799mm, front−32.270mm, AIR without ground/obstacle contact. There are52 qualified decision-end samples. The decision-end clearance peak is+9.101mm at tick5,040, and the closest qualified sample remains before the front plane at−23.166mm. Ground contact at tick5,446 revokes eligibility before crossing; RR crossing and placement remain absent throughout the entire run. These real exploration observations do not guarantee deterministic reproduction by the saved actor.

The new-MDP warm start from17,792 retains actor/std, critic and compatible identity normalization, resets Adam moments with initial learning rate3e−5, and carries neither old rollout storage nor physical state. The final checkpoint records21,888/136/2,720 and origin10,112; actor digest changes from `b599ff4c14b6262ca0c8dbe241c34826a75afb96d7b2abe7553497fb9d1afa48` to `238d76693f7f6533e84b3d862288dacbf30f88b27e52647ce96a90ffc1cb69df`. Full-episode spending grows512→4,608, while suffix spending stays7,168. Final KL0.0214036 and clip fraction0.26875 are training diagnostics only. The saved checkpoint does not contain physical episode state and requires a legal reset. Its completed non-successful natural-P01 evaluation is tracked separately above and below.

### Applied P06 rolling-prior retirement: implementation and real target evidence

Runtime `cae1d6e7cfdd8383fec67e8f919cc823f9e65448` changes only two production files, `configs/ppo_semantic_v3/stage_task_spec.yaml` and `src/wlr50_clean/ppo/semantic_supervisor.py`, plus targeted tests for this nominal revision. Root recorded227 focused tests passed and a separate54-test continuation/CLI/prefix batch passed; overlapping batches are not summed as unique tests or physical training credit. The change is a nominal/prior implementation revision, not itself learned improvement.

`p06_rolling_retirement_live.md` documents the first768 optimized decisions through global18,560, a subset of this finalized4,096-decision run. Outside the rear workspace the inherited P06 rolling prior remains0.3rad/s; entering the workspace retires that layer continuously, with peak-progress memory preventing it from restarting when the body moves backward. Actual P09 early-lift nominal FL/RL/RR wheel values fall to0.0486123663rad/s; the native zero-residual counterfactual is float32±0.0486123674, while the actual native targets separately include current-policy residual. P07's own FR−0.63rad/s adjustment remains visible. Later P09-owned0.3rad/s rolling remains available even after the P06 layer is fully retired: retiring the old layer does not mean commanding all wheels to stop.

These are real logged target changes, not merely a diagnostic scalar. The diagnostics describe the next suggestion, while last-applied native vectors are checked through the matching applied audit;15Hz endpoint vectors are not presented as a saved120Hz full-vector stream. The observed fade is implementation-effect evidence only, not a paired stability comparison against an old deterministic trajectory. That early snapshot contains no qualified RR lift and does not physically prove the qualified-lift carry override; the later fourth-episode lift and ground revocation above remain exploration outcomes, not completion.

### Previous P09 block finalized:1,024 actual,1,024 unspent

Preserved six-run boundary: checkpoint17,792 /104 PPO updates /2,080 optimizer steps, v3 additions7,680/60/1,200, recorded SHA-256 `9fc9826c695e6e3e63d81737e211836064d2527adf2a06bbed4ed5b54a51a68e`, `save_load_round_trip=true`. This P09 run spent1,024 of its originally planned2,048 decisions; its1,024 unspent decisions are not retroactively credited by the separate4,096-decision P01 run. The earlier4,096 allocation was separately completed as2,176 P06 plus1,920 P10 under unchanged `a475bad8f9a8c4cd0a6f23f6d621c00b06a02d47`, with full saved model/Adam/normalizer resume between those two runs.

`runs/ppo_semantic_v3/train/20260906T0515507234089Z_g314f5a8d6bf6_fa38b1752ea547a899ae53e6b7435bc6` ended `STOPPED_AT_VERIFIED_UPDATE_BOUNDARY` in both final manifests. Original plan2,048; finalized requested/actual1,024; unconsumed1,024; rounding overrun0. All1,024 audit rows are contiguous globals16,769–17,792 and belong to P09, with no rows beyond the optimized boundary. They account for8,192 current-policy physics ticks and8 real PPO updates /160 optimizer steps; every update records changed actor parameters and finite nonzero gradients. This is not2,048 completed decisions or evidence that a changed nominal proposal has already been trained.

Three accepted real teacher prefixes each supply761 behavior decisions /6,088 ticks, reaching current semantic P09 at50.733333s with149.266667s remaining. Total excluded teacher credit is2,283 decisions /18,264 ticks; there is no miss/fallback. Physical core including prefixes totals3,307 decisions /26,456 ticks, exactly2,283+1,024 and18,264+8,192, **not one continuous episode duration**. All credited rows explicitly exclude teacher data from storage, retain `teacher_initialized_suffix` scope, pass native tick-audit summaries and record zero in-episode state writes. Initial RR front−188.353mm/clearance−50.517mm, ground contact and no lift/cross/place history show that this prefix was preparation, not a pre-solved RR traversal. The nominal label P09 refers to the semantic supervisor and does not mean the frozen teacher already executed its recorded P09 lift.

| Current-policy episode | Credited global range | Decisions | Actual outcome |
|---|---|---:|---|
| 0 | 16,769–17,218 | 450 | P09 incomplete at tick9,688/80.733333s; stage age30s |
| 1 | 17,219–17,668 | 450 | P09 incomplete at tick9,688/80.733333s; stage age30s |
| 2 / unfinished tail | 17,669–17,792 | 124 | Nonterminal P09 at tick7,080/59.0s; no appended success/failure |

Both completed episodes report `INCOMPLETE_CONTROLLER_BLOCKED`, not a new physical safety failure: physical evaluation remains valid with no physical termination reason, while task/full-task success and terminal bootstrap are false. The first endpoint RR is front−69.327mm, clearance−50.404mm, one AIR sample/load0, with FL also not currently loaded on top. The second has RR front−52.720mm, clearance−49.702mm, ground contact/load0.467096; FL remains TOP/load0.527724. The unfinished tail has RR front−60.900mm, clearance−46.959mm,15 AIR samples and initial-clearance history, but still no qualified lift/cross/placement. None of those observations is suffix success; the124-decision tail is already optimized data but not a completed task outcome.

The final sidecar records checkpoint17,792/104 updates/2,080 optimizer steps, original v3 origin10,112, `save_load_round_trip=true`, and spent budgets `full_episode=512`, `phase_suffix=7168`, `smoke=0`. Actor digest changes from `30a1327a42338ea837db28edeb24eb3da749838ead64065759595f33a421737a` to `b599ff4c14b6262ca0c8dbe241c34826a75afb96d7b2abe7553497fb9d1afa48`. Final update KL0.0192466 and clip fraction0.21875 are optimization diagnostics, not stability superiority. Physical episode state is not saved; any later run uses a legal reset rather than an in-place continuation claim. At that historical boundary, the latest fresh-P01 evaluation remained the failed16,768 run; no17,792 checkpoint evaluation or video was inferred from saving it.

### Historical checkpoint38,272 fresh-P01 result: RR physical failure, not improvement

The finalized evaluation binds immutable `checkpoint_step_000038272.pt`, SHA-256 `f8b6e19233283c0678f336072d5febffe94329c097b5662bcb1b400e5b0ddde3`, runtime `64abc5357d00763419fe51c8126db8454fcf7697`, natural P01, seed2001 and deterministic policy. Actual741 policy decisions advance5,923 physics ticks over49.358333333s, with0 optimizer updates and `window_ended_before_task_terminal=false`. The three-tick final interval is not rounded to741×8. Run execution completed normally, but `task_success=false`, `controller_task_success=false` and the terminal phase is P09.

The shared physical evaluation remains `valid=true`, `success=false`, with `TASK_FAILURE_WHEEL_ONLY_CLIMB` and the recorded reason: "RR crossed front without uninterrupted above-top active lift and current AIR/TOP geometry". Its event/history ledger contains FR/FL qualified lift, crossing and placement, but no RR qualified/cross/placed and no RL traversal. This is not the earlier qualified-AIR pending-boundary case, nor a successful task hidden by a reporting error. The8,192-decision training block's occasional RR/RL lift events do not establish deterministic success for its final checkpoint. The finalized `eval_38272_diagnosis.md` retains the detailed stream evidence; no duplicate large-audit scan was performed for this master update.

The previous C28,032 result reached a later stage but also failed to complete the task; both immutable outcomes are retained. No stability superiority, paired improvement or causal allocation between nominal control and PPO is inferred from this new failure alone. The current B comparison was executed separately under the same runtime and seed, as finalized next; it is diagnostic evidence, not a new prerequisite to optimize or a demand to rerun A until successful.

### Current B zero-residual result: P09 incomplete, execution intact

Current B run `runs/ppo_semantic_v3/prior_B/20260906T0918387231314Z_g64abc5357d00_e8d6438326ce4f87b7d3138fba1ab953` is now finalized `SUCCEEDED`. Its mode is `semantic_prior_eval`, v3 under runtime `64abc5357d00763419fe51c8126db8454fcf7697`, seed2001, natural P01 and N1, with `checkpoint=null` and zero residual. Actual1,126 decisions advance9,008 ticks over75.066666667s; evaluation optimizer updates=0, task/controller success=false and `window_ended_before_task_terminal=false`. The terminal controller result is P09 `INCOMPLETE_CONTROLLER_BLOCKED`. The prior mode's `deterministic_policy=false` field does not denote a sampled PPO actor: no checkpoint/actor is loaded.

The independent physical evaluator remains `valid=true`, `success=false`, with physical termination reason=null. RR never has hard qualification, crossing or placement; FR/FL histories are present, RR/RL traversal is incomplete. Terminal RR is currently GROUND, not AIR/TOP, at front−57.446089mm and clearance−50.011973mm, load fraction0.486531. This is a genuine incomplete physical task under the current nominal/prior controller, not a claimed sensor/interface fault or broken software execution chain.

B therefore does not complete RR traversal in this episode; C38,272 also does not complete it and instead reaches the earlier49.358333s wheel-only failure. The shared runtime/seed/P01 setup supplies a useful control-versus-residual diagnostic, not identical evolving physical states or a demonstrated stability improvement. Neither longer elapsed time nor a different failure class is promoted to success. No B success requirement is added before future optimization, and no old A/C result or artifact is rewritten. A possible state-dependent policy-logstd revision is only a proposed next step at this reporting boundary: no implementation, migration or training outcome is credited here.

### Historical checkpoint28,032 fresh-P01 result: RR traversal, incomplete RL continuation

The finalized evaluation binds immutable `checkpoint_step_000028032.pt`, SHA-256 `9e1cc652ae19cbee87f0612051b62675a6397671b92d5ff8ccb18da344b355cb`, deterministic policy, natural P01, seed2001 and0 optimizer updates. It observes1,203 policy decisions and9,617 actual physics ticks over80.141667s; the final partial interval is not rounded to1,203×8. `window_ended_before_task_terminal=false` and the controller ends in P12 with `INCOMPLETE_CONTROLLER_BLOCKED`. A normal execution exit and `lifecycle=SUCCEEDED` do not override `task_success=false`. Physical evaluation remains `valid=true`, `success=false`, `termination_reason=null`: this is an incomplete task at the controller deadline, not a newly labelled physical safety failure or interface error.

RR's real qualified/cross/placed events are ticks5,820/5,998/5,999. These arise within the natural-P01 current-policy episode, not a teacher-supplied history. Nevertheless, terminal RR is back on ground: front−63.631829mm, clearance−49.474134mm, load0.200458, not current TOP support. RL qualifies at tick6,387/53.225s with measured above-top clearance+1.917975mm, but `qualification_revoked_ground_before_cross` at tick6,445/53.708333s clears that eligibility. RL has no cross or placed event, and ends on ground with front−50.526467mm, clearance−50.453657mm and load0.374668. Historical qualified-event timestamps are retained as evidence, not misreported as current unrevoked qualification. FL is actually TOP/load0.215841 at terminal; that endpoint alone does not establish continuous support throughout the attempt.

The final `eval_28032_diagnosis.md` reconciles9,618 finite raw observations including tick0 and all9,617 verified/native-effect ticks, with zero in-episode state writes. The episode has1,202 complete8-tick decisions plus the final1-tick interval. Actual policy sampling reaches P09 for90 decisions, P10/P11 for one each and P12 for451; P13 has0. P11→P12 occurs at tick6,016/50.133333s, and the terminal P12 age is30.008333s. Recorded continuous progress is not substituted for the unmet `placed_RL` completion predicate.

The RR crossing is a legitimate observed contact chain after qualified AIR, not a newly relabelled wheel-only result: RR qualifies at front−21.374mm/clearance+0.442mm, then makes its first post-qualification obstacle contact at tick5,956; at crossing tick5,998 it has front+0.137mm/clearance−1.871mm and obstacle14.885N. It is only just over the edge. At tick6,028/50.233333s it retreats to front−0.151mm, and at6,356/52.966667s it returns to ground with20.833N ground force. RR is already on ground when RL qualifies. The RL no-contact peak at tick6,402/53.35s is genuinely above top by21.624mm, **but still249mm before the front plane**. Its ground return at6,445 clears qualification before the later obstacle contact at6,561. The unresolved task is converting that lifted geometry into forward carry/placement while retaining the earlier support, not a lack of any observed RL height.

Current FL support and historical placement remain separate. Across the720-tick P09 window, FL is AIR569 ticks and has obstacle contact151; across3,601 P12 ticks it has obstacle contact3,579 and AIR22. At RL's peak, FL actually carries14.471N while RR is on ground. Recorded CoM.x decreases from0.703298m at RR placement to0.658373m at RL's peak; this is an association with the measured retreat/support change, not a new CoM gate or causal proof. Actual rear-leg projected residual peaks remain below their configured caps (P09 RR hip/knee1.931665/9.405639deg; P12 RL2.267327/2.195876deg). Native mapped targets, logical nominal suggestions and achieved measured joints are kept distinct; these figures do not prove that a particular larger command would succeed.

This is real deterministic RR subgoal progress compared with the earlier incomplete RR trajectories, but not whole-task success, a matched A/C stability claim or isolated proof of reward causality. The preceding block changed shaping and trained the actor, so an engineering revision and learned changes are not separately attributed by this outcome alone. The saved stochastic P13 suffix tails remain distinct from this fresh-P01 P12 outcome.

### Historical checkpoint21,888 fresh-P01 result: valid measured failure, not success

The finalized evaluation binds the exact21,888 checkpoint/SHA above, `deterministic_policy=true`, `from_phase=P01`, seed2001 and `optimizer_updates_during_evaluation=0`. The observed6,001 physics ticks are the actual751-decision episode, including its partial terminal interval;751×8 is not substituted for the measured tick count. `window_ended_before_task_terminal=false`, task/controller success=false and `physical_failure_is_not_interface_failure=true`: this is a completed physical non-success, not an interrupted evaluation, checkpoint-load problem or interface failure. The shared evaluator is `valid=true`, with `TASK_FAILURE_WHEEL_ONLY_CLIMB`.

Terminal RR front distance is+0.127259mm and clearance−1.586838mm, with verified obstacle pair, top geometry/TOP contact and load fraction0.4218877. RR active-lift qualification, crossing and placement remain false; `crossing_geometry_pending=false`. The only RR lift-attempt event is `whole_body_initial_clearance` at tick5,815/48.458333s: measured upward excursion10.566510mm, own RR joint motion0.933222deg and whole-body joint motion7.018738deg. This event is initial clearance, **not qualified above-top lift**. No RR qualified-lift event exists in this evaluation. FR/FL qualified lift, crossing and placement are present; RL/RR traversal is not complete.

The finalized `eval_21888_diagnosis.md` checks all6,002 observation rows including tick0. P09 spans ticks5,329–6,001 (673 actual ticks),85 decisions. RR first obstacle contact is tick5,551/46.258333s, front−49.683mm, clearance−49.662mm, with ground8.964N and obstacle14.528N. This precedes the only initial-clearance event by264 ticks. Before first contact, the best no-contact clearance is−48.942mm; the best no-contact clearance over all P09 remains−1.7851mm at tick5,972. Even the maximum including contact samples remains below top at−1.552856mm. First geometric crossing occurs at terminal tick6,001 with obstacle13.703N, not after a qualified above-top lift. No qualification-revocation event is missing: RR never qualified in this evaluation. These body-pair forces establish contact ordering, not an independently identified contact-face normal or complete propulsion causality.

Historical FL placement is not continuous current support. FL has no ground/obstacle force for the first496 P09 ticks (4.133333s), including RR's first obstacle contact and initial-clearance event. Across673 ticks,534 have no active FL contact and139 have obstacle contact; at terminal, FL really is loaded on top with load fraction0.402397 and obstacle13.070232N. That endpoint does not erase the earlier unloading. The recorded support projection has444 valid-inside,101 valid-outside and128 unavailable samples; unavailable is not outside. These remain measured diagnostics, not new success or admission gates.

All6,001 native audits verify, with zero in-episode state writes; all673 P09 ticks contain a real native target effect. RR projected hip/knee residual maxima use9.355%/27.304% of their caps; no channel reaches its configured cap in that window. This is not actor saturation, nor proof that every failed state is recoverable within those caps. Actual post-mapper/native targets remain distinct from nominal suggestions and measured joints. Later nonzero wheel suggestions are not automatically attributed to the already retired P06 layer. The diagnostic report preserves these command/mapping distinctions without claiming an interface defect or a paired performance gain.

The stochastic fourth training episode's genuine RR qualification therefore did not reproduce as a successful deterministic RR traversal by this checkpoint. That observation does not retrospectively erase the training lift evidence, but neither run supplies whole-task/suffix success or a same-condition stability-improvement claim. The historical A P10 entry-blocked recording and all earlier C results remain unchanged. A future reward/nominal revision or planned continuation is not counted as an executed update here.

### Historical checkpoint16,768 fresh-P01 result and remaining gap

The new evaluation's RR has no initial-clearance or qualified-lift event anywhere in the episode; RR active-lift/crossed/placed remain false. P06→P07, P07→P08 and P08→P09 occur at ticks5,240/5,248/5,256. Before obstacle contact, the best P09 RR clearance is−47.990mm (only about2.01mm above ground); first obstacle contact occurs at tick5,474 with front distance−49.608mm, clearance−50.058mm, ground force3.084N and obstacle-pair force32.725N. At terminal tick5,926 the wheel center crosses to+0.401mm, clearance remains−2.220mm and obstacle force is13.196N. No nonfinite sample or body collision is present in the diagnosed P06–P09 window. This evidence supports the recorded RR failure and is not the previously repaired qualified-AIR/microscopic-negative-clearance boundary case. Geometry and exact body-pair forces suggest contact-supported ascent, but do not supply an independently verified surface/contact-point classification or isolate the entire source of propulsion.

P09 has84 decisions /670 actual ticks, with670/670 native audits verified and actual target effects, no in-episode state writes and no phase-cap saturation. RR hip residual peaks at10.253% of its24-degree cap; RR knee at21.245% of its36-degree cap. This actor did not exhaust the configured residual range, but that is not proof the range can recover every failed state. The completed P10 block therefore extends real RL/P13 sampling without resolving the fresh-P01 RR preparation/crossing gap or controlled final-stop task. Current evidence remains **zero suffix success, zero fresh-P01 success, no demonstrated stability superiority, and no new paired video**. The new curriculum's requested budget is intentionally not prefilled as optimized work.

Historical checkpoint-bound update (2026-09-06 UTC): **12,672 lifetime optimized decisions / 64 PPO updates / 1,280 optimizer steps**. At that boundary V3 had added **2,560 / 20 / 400** to origin10,112/44/880. Its saved-and-reloaded C evaluation failed in P09 at49.4583s; no full task success or stability improvement was claimed. These launch-time statements are retained as chronological records, not current totals.

## Applied AIR-pending / physical-stop revision — historical launch record

Runtime `a475bad8f9a8c4cd0a6f23f6d621c00b06a02d47`, task revision `continuous_whole_body_v3_pending_air_and_physical_stop`: qualified, unrevoked AIR with insufficient current crossing geometry remains unfinished, never automatically crossed/placed/successful. The opt-in v3 evaluator still rejects ground-revoked lift, lateral bypass and RL-before-RR. The old v2 default behavior remains unchanged. V3 final completion and final progress use physical stop rather than eight-joint home similarity; the home suggestion and diagnostic remain. Targeted physical counterexamples: 169 passed; independent continuation/CLI/prefix regressions: 54 passed. These overlapping CPU batches are not physical training credit.

Read-only replay of unchanged recorded observations with the same new v3 evaluator: A old capture 7,845 rows remains incomplete (FR/FL/RR placed, RL not); old B 9,121 rows remains incomplete (only FR/FL placed); C12,672 5,936 rows remains RR wheel-only failure. None contained a pending-geometry tick. This is common-rule re-evaluation of old measured trajectories, not new A/B/C rollouts, not a counterfactual trajectory and not a current-nominal paired comparison. Original labels/files are unchanged.

Historical launch snapshot: `runs/ppo_semantic_v3/train/20260906T0406107605483Z_ga475bad8f9a8_bdc6c747c7434adb9c18c72ac1cbdf9a`, P06 real-teacher preparation, offset0, N1, seed1001, 4,096 requested decisions with complete128-decision updates and checkpoint cadence4. Warm start from then-latest12,672 retained actor/critic/std and compatible identity normalization, explicitly reset Adam, preserved lifetime counts and spent v3 budgets, and collected a fresh rollout. The subsequently completed allocation is recorded above:2,176 P06 plus1,920 P10. Separate RL coverage is not evidence of learning the skipped RR task. The detailed original launch/update records below are retained.

## Protected recovery point

- Existing project/remote/branch preserved; initial clean HEAD `1ca1e3756dcb7e8c6c689bb7f983c4a76c613003` is a docs-only successor of `00b94fb`.
- Latest verified source: `outputs/ppo_semantic_v2/checkpoints/history/checkpoint_step_000010112.pt`.
- SHA-256: `7397162192242fde4d7532b3ab529b49bea7b41689f766f3ee8c7b27ae3940fb`.
- Source lifetime: 10,112 optimized policy decisions, 44 official PPO updates, 880 optimizer steps. V3 physical training additions: not yet run at this report revision.
- Original FSM/controller/mapper, scene, robot, obstacle, actuator settings and all 29 protected files retain their original hashes. No old run or checkpoint is rewritten.

## A diagnostic and actual C evidence

The latest A viewport source ended at 65.3667 s / physics tick 7,844 in P10 WAIT_ENTRY. Its frozen reference entry gate rejected RR knee position (actual −45.8583 degrees versus historical −50.3976; error 4.5393 exceeds 2 degrees) and historical rebound velocity. The independent physical evaluator remained valid, with no physical failure recorded: FR/FL/RR had placed, RL had not crossed/placed. The 989-frame video source itself validated. Classification: **incomplete due to frozen A controller entry blocker**, not collision, wheel-only climb, or a codec failure. The old result remains unchanged.

C's saved 10,112 checkpoint evaluation remains an incomplete P09 deadline at 75.4667 s, with no P10–P13 coverage. Detailed P07–P09 evidence and the nine hypotheses are in the adjacent continuity diagnosis. In particular, small deterministic C residuals were not saturated: restrictive exploration capacity is a configuration fact, not proof that clipping caused this trajectory.

## Versioned production changes

- `semantic_supervisor.py`: v3 whole-body measured actuation plus verified initial AIR clearance; no requirement for target-leg joint change. Current above-top crossing qualification remains separate, is revoked by ground contact before crossing, and preserves append-only retry evidence. FL placed history and FL current contact/load remain separate.
- V3 nominal: only changed advisory channels are owned by a new phase; unfinished predecessor channel suggestions continue. No mapper, live residual, actuator history or physical state reset at ordinary transitions. An initial joint-sign-based rear pause was found incorrect during real training and has been removed in the next revision (evidence below). Finite joint suggestions remain advisory; the existing rolling suggestion can advance a genuinely cleared leg. These are B/C prior changes, **not PPO learning**.
- V3 potential uses the same physical/history definition regardless of phase label, with no phase-label bonus. Ordinary phase edges do not terminate the rollout or GAE.
- `semantic_prefix.py`: real reset-only frozen A roll-in to P06–P13, explicit actual-ACK takeover and continuous observation/controller/mapper history. Teacher data is excluded from PPO storage and counts. A failed initialization is logged and followed by one fresh-P01 fallback; no counterfeit historical observations or success latches.
- `semantic_reward.py`: departure is not automatically rebound. Recent touchdown plus upward reversal under an unchanged whole-body applied command supplies conservative rebound evidence; ambiguous/actuated departures and contact chatter remain diagnostic. Nominal/residual differences are diagnostic, while applied first and second differences supply smoothness costs. Magnitude regularization is disabled. Physical transfer weighting avoids a label-induced attitude-cost jump.

## Configuration and migration

Same 324 observation features, fixed scales and identity normalizers; same 12 Gaussian latent actions and official installed RSL-RL. Actor, critic and learned standard deviation are retained. Critic is recalibrated by new on-policy updates; Adam moments are deliberately reset with initial learning rate 3e-5. Old incomplete rollout is not inherited. This is a new-MDP warm start, not bitwise continuation or unchanged physical policy output.

P01–P05 requested symmetric residual caps: front hip/knee ±18/24 degrees, rear hip/knee ±12/18 degrees, each wheel ±0.3 rad/s. P06–P13: every hip/knee ±24/36 degrees and every wheel ±0.6 rad/s. The unchanged hard actuator limits intersect these ranges, producing state-dependent asymmetric usable headroom. Residual slew stays 60 degrees/s and 1.8 rad/s². Caps do not shrink at a phase edge; no all-channel large-noise requirement exists. Exploration std and PPO KL are separate from these ranges.

Start with N1 precursor/suffix blocks so synchronous N8 peer resets do not interrupt rear preparation. Follow with P10/P11 and full-P01 blocks; report **actual** coverage, not target mixture percentages as executed facts. Each run fixes its curriculum and configuration for every complete rollout/update.

## Verification and interpretation

Focused continuity/reward regression after the two shaping fixes: 153 passed, zero failures. Separate official-RSL migration/continuation batch: 106 passed; prefix/native integration batch: 47 passed (overlapping batches must not be summed as unique tests). CPU fixtures are not physical training, nor evidence of full task success.

## First real workspace response and interface defect

Run: `runs/ppo_semantic_v3/interface_checks/20260906T0233067086105Z_g2248cd1_workspace`, runtime `2248cd1940ad7a37d7a283f277732a366519db8d`. The actual A prefix reached P06 on all three independent resets. Nominal-zero and old-range segments each completed 64 decisions / 512 native ticks, with zero and 512 nonzero actuator-effect ticks respectively. New-range completed 20 audited native ticks (two complete decisions and four partial ticks) before the existing feedback-bias interface rejected a servo offset beyond its ±10-degree mapper-compensation envelope. This is an **execution interface failure**, not collision, wheel-only climbing, or completed workspace validation. Source results remain immutable. The final run manifest is FAILED; Kit immediate shutdown masked the Python exception in the process exit status, so exit code zero is not accepted as success.

The PPO post-mapper residual must be separated from bounded frozen mapper feedback. Original RobotAdapter and FSM files remain protected; the repair belongs to the semantic PPO execution boundary and must retain native hard limits, slew limits and one atomic Full12 write. A short repaired response will verify this boundary before optimizer use. No requirement for the diagnostic action to finish the task is introduced.

Two reward edge cases were fixed before the first v3 PPO update: the command change dispatched on a touchdown tick is retained when deciding whether a later departure is passive rebound; transfer weighting excludes future legs whose measured placement prerequisites are not yet satisfied. Both have positive and negative regression tests, and neither alters the frozen A controller.

## Repaired native response and launched continuation

Committed runtime: `26fa541abd1b12665a3585aa39e0cf0f40f0e81c`. V3 explicitly opts into `independent_post_mapper_residual.v1`; v2 keeps its prior path. The new PPO-only adapter keeps the original mapper and final physical limits, separately validates bounded controller feedback, and exposes controller and residual components in the ACK. Native/adapter/prefix regressions: 91 passed; the final two probe-seam tests also passed. All 29 protected files still matched before this commit.

Fresh real response: `runs/ppo_semantic_v3/interface_checks/20260906T0257098749051Z_g26fa541_workspace_repair`. Lifecycle SUCCEEDED, 32 response decisions / 256 native ticks, all P06, no task termination. The end projected residual is alternating hip +11.0908 degrees / knee −16.6362 degrees, with all wheels +0.2772703 rad/s. Native audit is verified with all 12 channels changed. This proves execution of this declared stimulus beyond the old feedback envelope; it is neither a whole-task success nor proof that every state can use the full symmetric requested range.

Real training has been launched at `runs/ppo_semantic_v3/train/20260906T0301338311562Z_g26fa541abd1b_5f5c057082944a4784c57990e0471190`: N1, seed1001, P06 real-teacher precursor, offset0, phase_suffix, 2,048 requested policy decisions, 128-decision rollout, 5 epochs × 4 minibatches, gamma .995, GAE lambda .95, PPO clip .2, initial Adam learning rate 3e-5 with adaptive target KL .01. Checkpoint cadence is every complete update. Teacher prefix remains outside optimizer credit. At launch, no new optimized decisions are yet claimed; append checkpoint-bound actual counts when complete.

### First verified real update

Update45 completed: +128 optimized policy decisions, +1 PPO update, +20 optimizer steps; lifetime10,240/45/900. All first-rollout samples are P06, not P07–P13. Actor SHA changed from `37fb5761909b4c1510328a0ccbff7b813a13364035c3fd83ec92a6a25b4b0c83` to `20849e14deadb5624bf2f70ba7a9e33ccc065a56768675c363b60fa925c93aae`; finite nonzero gradients verified. Mean KL .0186419, clip fraction .303125, adaptive learning rate1e-5 after the update (initial3e-5 is not claimed as the constant actual rate). Saved-and-roundtrip-verified `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000010240.pt`. The run continues; subsequent partially collected data is not included in optimized counts. A new full-P01 evaluation is still pending the completed training block.

### First block: preserved at a complete update before correcting the carry pause

Final lifecycle `STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`: 1,536 actually optimized decisions of 2,048 planned (512 not spent), 12 new official PPO updates, 240 optimizer steps; lifetime **11,648 decisions / 56 updates / 1,120 optimizer steps**. Latest immutable checkpoint: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000011648.pt`. The training loop took1,011.319s excluding its initial prefix setup. Four real P06 teacher roll-ins total1,792 teacher decisions /14,336 teacher ticks, excluded from PPO credit. Three physical BODY_COLLISION terminations, zero task successes. Last unfinished episode ended only at the saved training boundary; it is not counted as success or failure.

| Policy-credit phase | Optimized decisions |
|---|---:|
| P01–P05 | 0 |
| P06 | 999 |
| P07 | 4 |
| P08 | 4 |
| P09 | 529 |
| P10–P13 | 0 |

The first suffix episode covered414 decisions /3,306 physical ticks and ended at57.4167s in P09. The shared evaluator reported `central body/obstacle collision`, valid sensors, no bootstrap; no raw collision-force magnitude was retained in the compact training log. RR produced five initial-clearance events but never attained active above-top qualification, crossing, or placement. FL was AIR/load0 after P08→P09 and could not be credited as support. Native dispatch, no-state-write and residual handoff audits passed; finite predecessor P07/P08 suggestions continued under P09.

**Implementation correction prompted by these samples:** the nominal code incorrectly classified the first numeric decrease of a rear knee as descent, and held the whole layer until that wheel was already near/crossed the front plane. In current A's P09, between phase+.6667s and+1.1333s, RR knee0→−37.8 degrees accompanies wheel forward travel75.97mm while body x changes only−.013mm; clearance remains95.56mm, yet the wheel is still64.16mm before the plane. The later hip segment crosses at+1.6083s. Historical P12 evidence likewise shows115.86mm airborne forward travel during decreasing RL knee, with54.79mm remaining clearance. Thus neither the first knee decrease nor first hip decrease is a valid general descent trigger. The pause obstructed a suggestion that creates its own release condition. Past collision labels and training data remain unchanged; the defect is not used to relabel a failure.

The next revision removes this phase-layer pause altogether. It does not replace it with a historical time/angle/pose gate, does not force a unique motion, and does not change physical success/safety rules. Far-front and near-front regression cases now receive the same finite joint suggestions; actual task evaluation still determines crossing/placement. 155 continuity/supervisor/observation/reward/environment tests passed. Resume will preserve checkpoint11,648, retain already-spent v3 budgets and origin10,112, reset Adam explicitly, and discard any unfinished old-MDP rollout. Full-P01 evaluation remains pending after the corrected continuation, not falsely inferred from suffix training.

Success requires a new saved-and-reloaded checkpoint completing one continuous natural-P01 episode. Suffix success, first full success and demonstrated stability improvement remain three separate claims. There are no `improved` or full-success artifacts yet.

Method boundaries: retain the installed implementation of [PPO](https://arxiv.org/abs/1707.06347); use [curriculum initialization](https://proceedings.mlr.press/v78/florensa17a.html) as motivation, not a convergence guarantee. The finite 200-second task includes observed remaining time and is terminal; external truncation remains distinct, as described in [Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/). V3 N1 uses one gamma per policy decision; a shortened final physical interval has actual-dt costs, zero terminal potential and no bootstrap.

## Verified three-run totals and fresh C evaluation — 2026-09-06 UTC

The following values were read from each immutable `training_manifest.json` and independently matched to its `run_manifest.json.result`. Actual decisions, update counts, lifetime decision totals and per-phase counts agree. The corresponding `optimizer_updates.jsonl` files contain12,4,4 actual update records. Every run reports changed actor parameters and finite nonzero gradients. A run lifecycle of `SUCCEEDED` means its requested optimizer work finished, **not** that the robot completed the task.

| Actual training run under `runs/ppo_semantic_v3/train/` | Start / stage | Runtime | Actual decisions | PPO updates | Optimizer steps | Lifetime decisions | Final lifecycle |
|---|---|---|---:|---:|---:|---:|---|
| `20260906T0301338311562Z_g26fa541abd1b_5f5c057082944a4784c57990e0471190` | P06 teacher precursor / phase_suffix | `26fa541` | 1,536 | 12 | 240 | 11,648 | STOPPED_AT_VERIFIED_UPDATE_BOUNDARY |
| `20260906T0336310729329Z_g5036e11622ac_be84c27beb4744c781ff28ce0aa48fd9` | P06 teacher precursor / phase_suffix | `5036e11` | 512 | 4 | 80 | 12,160 | SUCCEEDED |
| `20260906T0344557730220Z_g5036e11622ac_474a218c14914a65b7d80eadcdbae170` | Fresh P01 / full_episode | `5036e11` | 512 | 4 | 80 | 12,672 | SUCCEEDED |
| **V3 actual addition** | | | **2,560** | **20** | **400** | | |

The first run spent1,536 of2,048 planned decisions; its unused512 are not counted. Runs2 and3 each spent their requested512. The teacher prefix contributes2,240 additional behavior decisions /17,920 native ticks across the first two runs; none enter these optimized totals. Run2 ended at its training budget with an unfinished P09 episode and no terminal event. Run3 completed one recorded P05 failure and began a second episode. All three runs report zero task successes; historical failed episodes remain present and their rewards are not rewritten.

| Policy-credit phase | Run1 | Run2 | Run3 | Total |
|---|---:|---:|---:|---:|
| P01 | 0 | 0 | 2 | 2 |
| P02 | 0 | 0 | 366 | 366 |
| P03 | 0 | 0 | 4 | 4 |
| P04 | 0 | 0 | 1 | 1 |
| P05 | 0 | 0 | 139 | 139 |
| P06 | 999 | 227 | 0 | 1,226 |
| P07 | 4 | 1 | 0 | 5 |
| P08 | 4 | 1 | 0 | 5 |
| P09 | 529 | 283 | 0 | 812 |
| P10 / P11 / P12 / P13 | 0 each | 0 each | 0 each | **0 each** |

Latest immutable checkpoint: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000012672.pt`, SHA-256 `2c05f08e5d5910506646137517351ae6df5f3e65c34b6d4297635bb00c6b7134`. Its manifest records `global_policy_decisions=12672`, `ppo_updates=64`, `optimizer_steps=1280`, `new_mdp_origin_global_policy_decisions=10112`, and `save_load_round_trip=true`. Actual checkpoint and manifest hashes both match `checkpoint_last_pointer.json`; the manifest hash is `87bb491fce76aafe1360456dcf0b829be3bc16380ef9068d929a6f7c1d9a9d7b`.

### New deterministic C result: not successful

Evaluation: `runs/ppo_semantic_v3/validation/20260906T0349586939252Z_g5036e11622ac_86eb1a3b6506415c92de0cb07efdbe08/evaluation_manifest.json`. It binds the exact12,672 checkpoint/hash above, runtime `5036e11622ac2e9b9eca86a3991f18781fe155e0`, deterministic policy, seed2001, natural P01 and **zero optimizer updates during evaluation**. The evaluation process finalized normally, but `task_success=false`:742 policy decisions /5,935 actual physics ticks /49.4583333s, terminal P09, `WHEEL_ONLY_CLIMB` (`TASK_FAILURE_WHEEL_ONLY_CLIMB` in the shared physical evaluator). The final partial decision is included honestly;742×8 is not reported as the actual tick count.

RR at termination: front distance+0.375870mm, bottom/top clearance−1.157461mm, current verified obstacle pair and top geometry, load fraction0.489026. Its current active-lift qualification is false; recorded history has only RR's initial-clearance attempt at tick5,322/44.35s, with no qualified above-top lift, crossing or placement. FR/FL crossing and placement histories are present; RL/RR are not placed. Sensor evaluation remains valid. Thus this is a non-success under the actual shared evaluator, not a checkpoint-load, sensor or video error. It is distinct from the qualified AIR-only FL edge case below; correcting that FL case cannot be credited as solving this RR failure or producing a new full-task success. P10–P13 have still not been reached by credited v3 training or this C evaluation.

### Current evaluator corrections in progress; not PPO gains

Run3's first P01 episode was labelled `WHEEL_ONLY_CLIMB` at P05, tick2,650/22.0833333s (332 decisions). Its unchanged `completed_episodes.jsonl.terminal_info.semantic_task.physical_evaluator` shows FL front+0.178519mm, clearance−0.104683mm, **24 consecutive AIR samples**, no ground or obstacle contact, load0 and an active-lift qualification established at tick1,634. A barely negative collider-bottom reading with no contact is not evidence of wheel-driven climbing. The strict AIR `bottom>=top` crossing branch wrongly converted this unresolved geometry into immediate failure. A versioned correction is in progress to leave such qualified AIR crossings pending rather than grant success or invent wheel-only contact. Actual loaded top/active-process evidence and safety checks remain required; the old failure/reward stays recorded under its original runtime.

Separately, review found that v3 final control still required the eight measured servos to approach `[0]*8` within8 degrees, and that final task-progress potential rewarded this home resemblance. The upcoming v3-only correction removes that hard historical-home condition and progress term. The home nominal remains an optional motion suggestion and its error remains diagnostic. Actual body/wheel speed, controlled wheel commands, physical support, full ROI, stable duration and safety conditions are unchanged; v2 defaults remain unchanged. These prior/evaluator corrections are implementation changes, **not learned performance improvements**. The intended next training run preserves checkpoint12,672 weights with explicit versioned new-MDP migration and fresh rollout credit; no unexecuted continuation is included in the totals above.

### Diagnostic A video remains available; publication is not a training gate

The original A P10 historical-entry incomplete result described earlier is unchanged. Its source video already exists at `runs/ppo_semantic_v2/video_eval/baseline_A/20260905T0902190128552Z_g00b94fbb276a_e0ddaa746bc84bddb8e4f80506e09fac/source/actual_viewport_video.mp4` (98,808,532 bytes, SHA-256 `e3f76cdcded1d680232dc7f113be2bd7b7c7dfcfc8b56c3725de4deb79fa2a18`). The adjacent `viewport_buffer_video_manifest.json` reports valid989-frame,15fps,H.264,yuv420p full decode. Its semantic source manifest correctly retains `success_candidate=false`, `diagnostic_only=true`. Technical video validity is not task success, and this old extra64-tick-pre-roll capture is not a new same-condition v3 A success baseline.

The current opt-in v3 video CLI can likewise execute and preserve an A failure/incomplete raw recording: `capture_semantic_video` finalizes the recorder and writes raw/technical/source artifacts in its `finally`/finalization path; `semantic_video_cli` writes a final `DIAGNOSTIC_FAILURE` run manifest, then returns2. The PowerShell wrapper surfaces that nonzero result and preserves the exact run and launcher logs. This is intentional diagnostic retention, not a successful publication. The standalone success-publication and success-comparison functions remain strict and are **not invoked by training, required to start/resume PPO, or a demand to rerun A until5/5**. A failed raw source may be handed off with an explicit incomplete/failure label and its technical manifest; a neutral diagnostic comparison publisher has not been implemented. No failed A or C file is renamed as success/improved.

## In-progress P06 block: migration and first saved update verified

Run `runs/ppo_semantic_v3/train/20260906T0406107605483Z_ga475bad8f9a8_bdc6c747c7434adb9c18c72ac1cbdf9a` starts from checkpoint12,672 under runtime `a475bad8f9a8c4cd0a6f23f6d621c00b06a02d47`. Its `run_manifest.started.json` is RUNNING and requests4,096 P06 phase_suffix decisions; this is a **plan, not a completed training total**. Read-only verification used this run's `new_mdp_warm_start.json`, `new_mdp_initial_action_comparison.json`, the source checkpoint manifest and the new immutable pre-update checkpoint manifest. No Python/native process was launched for this review.

The warm-start record binds source SHA-256 `2c05f08e5d5910506646137517351ae6df5f3e65c34b6d4297635bb00c6b7134`, source versionv3 and lifetime counters12,672/64/1,280. Its only changed runtime files are `configs/ppo_semantic_v3/stage_task_spec.yaml` and `src/wlr50_clean/ppo/semantic_supervisor.py`; the execution/action/observation/reward/quality configurations retain their recorded source hashes. The immutable initial checkpoint is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000012672_s2c05f08e5d59_ga475bad8f9a8_e59d6eda4c09e2daacde30e7b11777d3dacf1fcaf058d883ee318f495a00ef74.pt`, recorded SHA-256 `b3bc6aea0887b470d00380e03ccaf10a27ce0505a5981608e8d7ab3869fc2a7e`, with `save_load_round_trip=true`.

Source and pre-update manifest parameter hashes agree exactly: actor `1132c1ad8e50dfea55e52c67b8335a824905bf64345df48cb217994fff1c8948`, critic `10b88950396c6ac09ef6afdf1fa3efafdc7e198ee375efa294c389a2cf7ba47a`, identity normalizers `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`. The actor digest covers every named actor parameter, including its learned distribution parameters; the migration explicitly preserves learned std rather than reinitializing it. The initial manifest still declares fixed versioned preprocessing and identity RSL normalization. Adam is deliberately reconstructed with empty moments at3e-5 after verifying the official saved-state restore: its state hash changes from source `9178c6cd84e53c1cd57eb0068954a74da9f31dc50c78036d07951a21ba4a8b2a` to initial `288adbbc80645732a151e1b41a8e33524b35bec244c5d1301ff503f78cae2b8a`. The initial manifest's retained `last_update=64` is source history, not a newly executed update.

Spent budgets remain exactly `{full_episode:512, phase_suffix:2048, smoke:0}` in both source and initial target manifests; original v3 origin10,112 and lifetime counters are preserved. `old_rollout_buffer_inherited=false`, `physical_state_inherited=false`, `physical_env_state_saved=false`, and `exact_mdp_resume=false`. The loader requires fresh128×1×12 storage atstep0 with no pending transition, verifies the saved network/normalizer state, creates the new Adam optimizer and restores the verified training RNG for fresh rollouts. This is an explicit new-MDP boundary, not bitwise physical continuation or reclaimed spent budget.

The same-state initial action comparison is at P06 tick3,584/29.8666667s after real teacher roll-in, observation dimension324 and projection dt1/120s. It records zero optimizer updates and no actual environment/bridge-history modification. All12 channels of each old/new logical applied, bounded, scaled, masked, rate-projected and safe-projected target difference are zero, consistent with unchanged action execution profiles. The raw deterministic actor mean begins `[0.0122202961, -0.0721314773, -0.0676235184, 0.0174939297, ...]`. This is a same-observation logical projection check, explicitly **not a native-dispatch test or proof of an unchanged future physical trajectory**.

At this verification snapshot, the first actual new optimizer record is update65: **+128 decisions /+1 PPO update /+20 optimizer steps**, saved as `checkpoint_step_000012800.pt` with round-trip verified lifetime **12,800/65/1,300**, original v3 origin10,112 and spent phase_suffix2,176/full_episode512/smoke0. Actor hash changes to `82f619249c10b97849869a03c2eb73f6ba8363b77a75873f4d8f0d67d2ae10e9`, finite nonzero gradients are recorded (norm1.000152–1.227801), mean KL0.01104824, clip fraction0.165625 and adaptive post-update learning rate4.5e-5. This advances verified v3 additions to2,688 decisions/21 updates/420 optimizer steps at this snapshot only. The rest of the planned4,096 block is not yet credited here, and no task success is inferred from this optimizer update or the evaluator corrections.

### P06 block completed at update81; remaining budget reassigned to P10

The same04:06:10 run now has matching finalized `run_manifest.json` and `training_manifest.json`: `STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`, `planned_requested_policy_decisions=4096`, `actual_policy_decisions=2176`, `requested_policy_decisions=2176`, `unconsumed_requested_policy_decisions=1920`, `rounding_overrun=0`. Its17 actual `optimizer_updates.jsonl` records contribute340 optimizer steps. Actor parameters changed and finite nonzero gradients were recorded. The reported training wall-time field is1,241.4286s. A saved update boundary is not an episode-success boundary.

| Credited phase in this P06 block | Actual decisions |
|---|---:|
| P06 | 819 |
| P07 | 3 |
| P08 | 3 |
| P09 | 1,351 |
| P01–P05 and P10–P13 | 0 each |
| **Total** | **2,176** |

Four real P06 teacher-prefix attempts were accepted, each448 teacher decisions /3,584 native ticks: **1,792 teacher decisions /14,336 teacher ticks**, all excluded from PPO credit. No prefix miss or fallback is recorded. Credited policy physics totals17,401 ticks; teacher plus policy totals31,737 ticks. Their sum is not rounded to decisions×8 because one terminal policy interval is partial.

The run completed three **incomplete**, non-success P09 episodes, not three new physical safety failures. Every endpoint's shared physical evaluator remains valid with `termination_reason=null`, while the semantic controller reports `INCOMPLETE_CONTROLLER_BLOCKED` at its P09 deadline. Their recorded data are:

| Episode | Policy decisions after prefix | Episode time including real prefix | Terminal tick | P09 age | RR front distance | RR clearance |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 658 | 73.733333s | 8,848 | 30.000000s | −49.810mm | −49.054mm |
| 1 | 684 | 75.466667s | 9,056 | 30.000000s | −57.791mm | −33.245mm |
| 2 | 717 | 77.608333s | 9,313 | 30.008333s | −56.055mm | −48.062mm |

All three endpoints have RR active-lift qualification, crossing and placement false. The first has current RR ground contact; the latter two do not. No full-P01 task success or teacher-initialized task success is recorded. The fourth episode contributed117 policy decisions before the training boundary and did not terminate: it is **neither a success nor an additional completed failure**. The earlier C12,672 wheel-only evaluation and the old A historical-entry incomplete result remain unchanged and are not replaced by these training diagnostics.

The saved immutable boundary is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000014848.pt`, recorded SHA-256 `b03ec4fca80d473524b06878e7c6a50f3071d98402a9a62d7881b7a0c9d56461`, with `save_load_round_trip=true`. Its manifest records14,848 lifetime decisions /81 PPO updates /1,620 optimizer steps and original v3 origin10,112. Spent budgets are `phase_suffix=4224`, `full_episode=512`, `smoke=0`. Thus completed v3 additions are now **4,736 decisions /37 PPO updates /740 optimizer steps**. Actual cumulative phase credits are P01=2, P02=366, P03=4, P04=1, P05=139, P06=2,045, P07=8, P08=8, P09=2,163, and P10–P13=0 each. The last real update81 reports KL0.01749473, clip fraction0.2609375 and learning rate1e-5; these optimizer diagnostics are not task-success evidence.

New invocation `runs/ppo_semantic_v3/train/20260906T0431068891234Z_ga475bad8f9a8_12c8718349444d16934274ffb641768c/run_manifest.started.json` is RUNNING, requests **P10 phase_suffix /1,920 decisions**, and binds the same `a475bad8f9a8c4cd0a6f23f6d621c00b06a02d47` runtime and checkpoint14,848. Its arguments explicitly have `new_mdp_warm_start=false` and `resume_migration=null`: this selects the unchanged-MDP full saved-state/Adam resume workflow, not another optimizer reinitialization. Only the next run's completed update evidence can credit that remaining budget. The budget ledger therefore reads **4,096 planned =2,176 completed P06 +1,920 reassigned P10**, not4,096 already trained or a fresh extra1,920 on top. Physical initialization, curriculum target and prefix time remain separately reported from optimizer credit.

## Continuous wheel-command stop progress — implementation record only

Committed runtime `314f5a8d6bf6b1d2791f6b5b021a0603446a0b5f` adds the opt-in v3 setting `final.stop_command_progress=reciprocal_physical_stop_tolerance`. The production delta is limited to `configs/ppo_semantic_v3/stage_task_spec.yaml` and `src/wlr50_clean/ppo/semantic_supervisor.py`, with targeted regression tests. It adds **one continuous command-stop term to the existing finish potential**, not another reward family or phase bonus. For maximum actual commanded wheel magnitude `u` and the existing physical stop tolerance `tau=0.02 rad/s`, this term is1 when `u<=tau` and `tau/u` otherwise. Thus progress remains responsive above the tolerance instead of becoming uniformly zero; it does not label above-tolerance commands stopped. The original measured wheel/body rates, physical success predicate,0.5s stable duration, support/ROI/safety requirements, nominal suggestions and residual caps are unchanged. The default v2 setting remains unchanged.

Root's recorded focused validation is172 physical/reward regression tests passed plus a separate54-test continuation/CLI/prefix batch passed. These may overlap and are **not added together as a unique-test total**. They are CPU implementation checks, not new physical task successes or optimized decisions. Correcting this shaping omission is an engineering change, not a PPO performance gain.

The actual ongoing invocation is `runs/ppo_semantic_v3/train/20260906T0515507234089Z_g314f5a8d6bf6_fa38b1752ea547a899ae53e6b7435bc6`, requesting P09 phase_suffix /2,048 decisions. Read-only PowerShell inspection of its `new_mdp_warm_start.json` and immutable initial-checkpoint manifest confirms source `checkpoint_step_000016768.pt`, source SHA-256 `016c7f7a8b25fef2e8893900e4ca1a34c0c37c321a03eeced5f6643222ada8a7`. The record lists only the two production files above as changed. Source and target spent budgets are identical: `full_episode=512`, `phase_suffix=6144`, `smoke=0`; lifetime16,768/96/1,920 and original v3 origin10,112 are preserved.

Source and initial actor hashes both equal `30a1327a42338ea837db28edeb24eb3da749838ead64065759595f33a421737a`; critic hashes both equal `4bb1970e50c6ed9fcbeab2fb5ef1dcd7707e90f9c92430193556ac5947da1f57`; identity-normalizer hashes both equal `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`. Learned std is retained with all actor parameters. Adam is explicitly reset at3e-5: source optimizer digest `14f80ae9b8be24020ac9f2e2a3e75d248d82e633ccc9bd83f4022723ba00c54c` changes to initial `288adbbc80645732a151e1b41a8e33524b35bec244c5d1301ff503f78cae2b8a`. The migration records `old_rollout_buffer_inherited=false`, `physical_state_inherited=false` and `exact_mdp_resume=false`; it uses fresh rollout storage rather than reusing old transition credit.

Immutable pre-update checkpoint: `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000016768_s016c7f7a8b25_g314f5a8d6bf6_0bf82d3af21b470e0600f58ff8b3d113695b85c6b40061d6e7271e116b8a6a18.pt`, recorded SHA-256 `275c37596d300f0f1c26a51db0170aea1669590995d54a498f51f6b0008f4982`, `save_load_round_trip=true`. This initialization is not itself an optimizer update. The run remains in progress at this documentation boundary; its planned2,048 decisions and any subsequently produced partial rollout are **not included in this report's current completed-run counters**. Actual new counts will be updated from finalized evidence separately. The earlier A/C outcomes and absence of a validated new paired video remain unchanged.

## Actuated AIR and current-clearance shaping — preserved launch record

Runtime `64abc5357d00763419fe51c8126db8454fcf7697` enables v3 `lift_credit_semantics=measured_air_process_current_top_gap`, revision `continuous_whole_body_v3_actuated_air_current_clearance_credit`. Its production diff is limited to `configs/ppo_semantic_v3/stage_task_spec.yaml` and `src/wlr50_clean/ppo/semantic_supervisor.py`, plus the dedicated `tests/unit/test_semantic_lift_clearance_credit.py`. It changes the existing progress potential's lift credit, not the hard initial-clearance/qualified-lift/cross/placed history, physical success rules, carry eligibility, nominal provider or residual caps. No prior failed trajectory is relabelled by this report, and this engineering correction is not a PPO performance gain.

Soft AIR credit is earned from a fresh consecutive AIR suffix, excluding earlier ground/wall motion, after existing predecessor-placement and task-region checks. It requires chronological upward motion using the existing3mm initial-clearance scale, measured joint-command demand or tracking error, and measured joint/gravity response. Once earned it persists during the same uninterrupted AIR process, so hovering does not lose credit merely because a sliding-window motion signal expires; contact or failure clears the soft state. These measured timing and command/response conditions are **heuristic actuation evidence, not a causal proof of motor work**. No soft flag is imported into the hard qualification or success history.

The shaping value depends on current top clearance, not an unreduced historical peak: with the existing lift scale `s=0.008m`, its clearance fraction is `s / (s + max(0, -current_clearance))`. An eligible but not hard-qualified AIR process receives0.99 times this fraction; current AIR, no ground/obstacle contact, initial-clearance evidence and task-region conditions still apply. A hard-qualified but not crossed lift also uses current clearance; completed qualified crossing retains its completed credit. Hard carry progress continues to require the original hard active-lift history, so soft credit does not unlock carry or declare traversal. The disabled/default legacy mode retains its earlier semantics.

Root recorded251 related regression tests passed and a separate54-test migration/continuation batch passed. They may overlap and are **not summed as an independent unique-test count**. CPU tests do not constitute physical task success, optimized samples, or evidence that the revised shaping improves stability.

Actual launched run: `runs/ppo_semantic_v3/train/20260906T0636173277212Z_g64abc5357d00_6614a15089014b2ebef51695b9b8cc22`. Read-only PowerShell inspection of `run_manifest.started.json` confirms `lifecycle=RUNNING`, v3, `stage=phase_suffix`, `from_phase=P06`, teacher offset0, seed1001, N1, `decisions=6144` and `new_mdp_warm_start=true`. The embedded warm-start record binds source `checkpoint_step_000021888.pt`, SHA-256 `05bf403d742426582e8c3cada15ee215b8e0c9f4be5b1abe65eb2c8cf91873d8`, and lists only the two production files above as changed. Execution, action, observation, quality and reward-configuration file hashes remain equal across that recorded transition; the shaping implementation/spec changes are nevertheless an explicit new-MDP boundary.

That launch record declares preservation of actor parameters including learned std, critic weights and compatible identity normalization (324 observation features/12 raw actions), with critic recalibration on fresh on-policy data. Adam moments are explicitly reset at initial learning rate3e−5. Old rollout storage and physical state are not inherited; this is not exact-MDP or bitwise physical continuation. Source and target spent-stage ledgers agree at `full_episode=4608`, `phase_suffix=7168`, `smoke=0`, preserving original v3 origin10,112 and source lifetime counters. Real teacher roll-in is excluded from PPO credit. This section verifies the actual launch's declared migration contract, not a new optimizer boundary or a completed prefix success.

At this documentation boundary, **6,144 is planned only**. The seven completed-run totals remain11,776 new decisions /92 updates /1,840 optimizer steps, lifetime21,888/136/2,720. No eighth completed run, new success, new deterministic evaluation or video is credited. Finalized training/update evidence will supply the actual consumed budget and outcomes after the run ends; subsequent partial rollouts are not prefilled here.

## P06 lift-credit block finalized:6,144 actual, no suffix success

The06:36:17 run is now finalized `SUCCEEDED` in its run/training manifests: planned=requested=actual6,144, unconsumed0 and rounding overrun0. This supersedes the immediately preceding launch-time plan without rewriting that historical record. `p06_lift_credit_block_next.md` reconciles all6,144 audit rows, globals21,889–28,032,48 complete128-decision updates and960 new optimizer steps. All48 updates record actor change and finite nonzero gradients; the actor digest changes from `238d76693f7f6533e84b3d862288dacbf30f88b27e52647ce96a90ffc1cb69df` to `f5ea5f3f2e4e51f201b7525da4885948a1fe5c588e202ff60fa049d2e9e189de`. Source21,888/136/2,720 therefore advances exactly to28,032/184/3,680. This is genuine optimization, not a task-success result.

All eight real teacher-prefix attempts are accepted with no fallback, each448 decisions /3,584 ticks: **3,584 teacher decisions /28,672 ticks**, excluded from PPO storage and global counts. Their raw PPO actions and projected residuals are zero, with verified native audits and zero in-episode state writes. Every current-policy credit starts at P06 after29.866667s of real preparation; the200s task clock includes that prefix. Current-policy physics totals **49,138 ticks**, not6,144×8: episodes1 and2 each have a final1-tick decision, globals23,364 and24,089, accounting for the14-tick difference. All credited ticks verify and have a real native effect; all6,144 policy rows exclude teacher credit and report zero in-episode state writes. Total physical core including prefixes is9,728 decisions /77,810 ticks over eight separate episodes, not one continuous trajectory. The training wall field is3,282.2445895s; separately recorded reset30.4051242s and roll-in1,240.1457101s are not blindly added to it.

| Episode index | Optimized global range | Decisions | Actual ending stage/time | Outcome and limitation |
|---|---|---:|---|---|
| 0 | 21,889–22,643 | 755 | P09/80.200000s | Incomplete; no RR qualified/cross/place |
| 1 | 22,644–23,364 | 721 | P09/77.875000s | Incomplete; RR qualified at tick6,230 but did not cross/place |
| 2 | 23,365–24,089 | 725 | P09/78.141667s | Incomplete; no RR qualified/cross/place |
| 3 | 24,090–24,855 | 766 | P12/80.933333s | Incomplete; RR crossed/placed, later back on ground; RL not qualified |
| 4 | 24,856–25,551 | 696 | P09/76.266667s | Incomplete; no RR qualified/cross/place |
| 5 | 25,552–26,868 | 1,317 | P13/117.666667s | Incomplete; both rear legs crossed/placed, controlled stop not achieved |
| 6 | 26,869–27,576 | 708 | P09/77.066667s | Incomplete; no RR qualified/cross/place |
| 7 / unfinished tail | 27,577–28,032 | 456 | P13/60.266667s | Nonterminal; both rear legs crossed/placed, not controlled |

The first seven episodes contribute5,688 decisions and all formally report `INCOMPLETE_CONTROLLER_BLOCKED`; the456-decision tail is already optimized but not a completed failure or success. There are five P09 incompletes, one P12 incomplete and one P13 incomplete. Both `success_count` and `teacher_initialized_task_success_count` are0. RR qualified history occurs in four episodes, with crossing/placement in three; RL qualified/cross/placed occurs in two, including the unfinished tail. These are current-policy suffix subevents, not a fresh-P01 success rate or evidence of a reliably successful final actor.

### Episode3: RR traversal occurred, then current support was lost

RR qualified/cross/placed at ticks5,742/5,998/6,092, within the current PPO segment. The qualified event has measured above-top clearance+0.719452mm, followed by a real airborne crossing and loaded placement. Phase transitions continue through P10/P11 into P12 without reset or imported historical entry. This is actual RR subgoal progress, not a teacher-prefix achievement. However, by the first observed decision-end ground return at tick7,352, RR is front−50.138410mm with clearance−50.441361mm and load0.518881. At the P12 terminal it remains on ground, while FL is AIR/load0. RL has soft/initial attempts but never hard qualification, crossing or placement. Append-only RR placed history does not certify lasting top support.

### Episode5: both rear legs reach P13; nominal ends but stopping still fails

RR qualified/cross/placed at ticks6,170/6,350/6,368 and RL at6,710/6,852/6,914, all after current-policy credit begins. P12→P13 occurs at tick6,920/57.666667s. Continuous handoffs preserve real actuator/physical history; these rear-leg events do not imply that PPO learned the teacher-supplied FR/FL prefix.

Of900 P13 decision-end samples, the first272 still have finite nominal motion; from75.866667s through117.666667s, **all12 nominal channels remain zero** for628 samples. In that zero-nominal segment, region and support pass all628, but controlled stop passes none: body linear speed exceeds0.05m/s in383 samples, angular speed exceeds0.30rad/s in44, actual wheel speed exceeds0.25rad/s in183, and wheel command exceeds0.02rad/s in626. The two small-command samples still have body speeds0.118558/0.130289m/s. These are15Hz endpoint counts, not a claim that every120Hz intermediate state was stored in that audit. This incomplete episode is not explained by an indefinitely running nominal layer or command noise alone.

At its formal P13 terminal,117.666667s/stage age60s, body linear speed0.068714m/s and maximum wheel command0.133108rad/s fail the unchanged0.05/0.02 tolerances. Angular0.111517rad/s and actual wheel maximum0.163151rad/s pass their existing limits. Nominal is all zero. Current RR and FL are AIR/load0; RL and FR are loaded TOP. Home error14.4632deg is diagnostic, not a failure condition. The result remains `INCOMPLETE_CONTROLLER_BLOCKED`, not controlled-stop or suffix success.

### Final tail: a second double-rear traversal, still nonterminal P13

Episode7's456 optimized decisions include P06=261, P07/P08/P10/P11=1 each, P09=73, P12=68 and **P13=50**. RR qualified/cross/placed at ticks6,066/6,263/6,267; RL at6,407/6,756/6,832. P13 begins at tick6,832/56.933333s and the final saved data row is tick7,232/60.266667s, stage age3.333333s, `termination=null`. It is not an eighth completed episode and is not merely an unfinished P12 attempt.

At that final row both rear legs are actually TOP, loads RR0.135796/RL0.509111; FR is TOP/load0.355094 and FL is AIR/load0. Region/support pass, but controlled=false and stable duration0: body speed0.081593m/s, actual wheel maximum0.408138rad/s and command maximum0.516135rad/s exceed existing tolerances. Finite early-P13 nominal wheels remain0.3rad/s, with residuals recorded separately. This does not establish permanent rolling or imply that more time would necessarily yield success.

The immutable final checkpoint for this P06 block is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000028032.pt`, SHA-256 `9e1cc652ae19cbee87f0612051b62675a6397671b92d5ff8ccb18da344b355cb`; root's verified sidecar SHA-256 is `fc0ce42fd77f8a3add28fbb8af4b3304f7e244b6c3cfeb6352014d86b5ba8d64`. It records `save_load_round_trip=true`, spent full/suffix4,608/13,312 and original v3 origin10,112. Checkpoint metadata explicitly states `physical_env_state_saved=false` and `resume_physics=legal_reset_not_bitwise_continuation`; reloading training state does not resume the tail's P13 physical state in place. The seven earlier run totals, old checkpoints, A result and completed C21,888 failure remain intact. The separate C28,032 natural-P01 evaluation has since completed with the P12 incomplete result recorded above; it is not inferred from the training tail.

## Same-MDP P10 continuation from28,032 — preserved launch record

Actual new invocation: `runs/ppo_semantic_v3/train/20260906T0748083849450Z_g64abc5357d00_525e380c18ae4039ad93d764e7a06f50`. Read-only inspection of `run_manifest.started.json` confirms `RUNNING`, unchanged runtime `64abc5357d00763419fe51c8126db8454fcf7697`, v3 P10 phase_suffix, offset0, seed1001, N1, **2,048 planned decisions**, and checkpoint cadence4 completed updates. Source is the immutable28,032 checkpoint. `new_mdp_warm_start=false`, `resume_migration=null`, `_warm_start_record=null` and `_migration_record=null` select the unchanged-MDP full checkpoint resume path, not another reward migration or optimizer restart.

The saved actor/critic/std, normalizer, Adam state and training RNG are continued through that path without an explicit new-MDP moment reset; physical initialization is still a legal fresh teacher-prefix episode, not saved physical-state restoration. Teacher preparation remains excluded from PPO credit. At this launch snapshot, source spending remains full_episode4,608/phase_suffix13,312, lifetime28,032/184/3,680 and v3 additions17,920/140/2,800. The2,048 request is not a ninth completed run, a task-success gate or precredited training. Actual consumption, checkpoint and outcomes will be added only after the run's finalized evidence is supplied.

## P10 block finalized:2,048 actual, two P13 incompletes and a tail

The07:48:08 run is now finalized `SUCCEEDED` with planned=requested=actual2,048, unconsumed0 and rounding overrun0. Final manifest/sidecar confirm16 new PPO updates /320 optimizer steps, checkpoint30,080/200/4,000, `save_load_round_trip=true`, and immutable `checkpoint_step_000030080.pt` SHA-256 `5524ae5dfa51009786398fe3c5da5a5065e34792e2a1ec24c972552161ffb82e`. Full-episode spending stays4,608; suffix spending grows13,312→15,360. Reported wall time is1,450.9099527s. This supersedes the launch-only snapshot without replacing the earlier evidence. Execution completion is explicitly not physical task success.

The exact actual phase ledger is P10=3, P11=3, P12=201, P13=1,841, totaling2,048 decisions /16,384 credited ticks. Three real P10 prefixes each contribute948 teacher decisions /7,584 ticks: **2,844 teacher decisions /22,752 ticks**, excluded from PPO credit. Including those prefixes, physical core totals4,892 decisions /39,136 ticks across three episodes. The task clock includes the63.2s teacher preparation; it is not reset to zero at policy takeover. These are separate episodes, not one combined continuous rollout.

The finalized `p10_block_30080.md` verifies all2,048 contiguous policy rows and all16 complete updates, with finite nonzero gradients and actor changes at each update. Actor digest changes from `f5ea5f3f2e4e51f201b7525da4885948a1fe5c588e202ff60fa049d2e9e189de` to `2ffcaf809bb6050334294648d28df85e7a1e5c4c5142010908cc9901956c6e99`. All16,384 current-policy ticks have verified native effects; all22,752 teacher ticks separately verify, with teacher raw/projected residual zero and all in-episode state-write counts zero. Run-manifest start/end elapsed time is1,848.783665s, distinct from the1,450.9099527s training wall field; those timing scopes are not added together.

| Episode index | Optimized global range | Decisions | Actual outcome |
|---|---|---:|---|
| 0 | 28,033–29,000 | 968 | P13 incomplete at127.733333s |
| 1 | 29,001–29,969 | 969 | P13 incomplete at127.800000s |
| 2 / unfinished tail | 29,970–30,080 | 111 | Nonterminal P13 at70.600000s |

Both completed episodes formally report `INCOMPLETE_CONTROLLER_BLOCKED`, not suffix success; both success counters remain0. The111-decision tail contains P10/P11=1 each, P12=68 and P13=41, and is optimized data but not a third terminal result. `p10_block_30080.md` preserves the bounded earlier snapshots and the final episode audit. The historical29,056 checkpoint snapshot's second-episode56-decision tail is not confused with this final third-episode tail.

### Teacher RR history versus new current-policy RL events

All three roll-ins already supply RR qualified/cross/placed history at ticks6,938/7,109/7,579, before policy credit starts at7,584. Those RR and earlier FR/FL events are teacher achievements, not newly learned actions in this P10 block. New RL qualified/cross/placed events occur after takeover: episode0 ticks7,912/8,066/8,122; episode1 ticks7,960/8,064/8,132; tail episode2 ticks7,789/8,075/8,139. These are real current-policy RL subevents, but do not establish fresh-P01 rear preparation or task completion.

Both completed episodes finish after the finite nominal has ended, with all12 nominal channels zero and controlled stop still unmet. The first episode's900 P13 decision-end samples split into272 finite-nominal and628 all-zero samples; in the zero segment, body speed exceeds0.05m/s in379 samples and wheel command exceeds0.02rad/s in627. Its only below-command-tolerance sample still has body speed0.088244m/s. At terminal, body speed0.059984m/s and command0.112195rad/s fail those unchanged tolerances, while angular0.231826rad/s and measured wheel maximum0.159298rad/s pass theirs. This is not evidence of indefinitely inherited nominal rolling or a failure caused only by home resemblance; home is diagnostic.

At the two terminal endpoints RR is currently AIR/load0, RL is TOP/load0.543160 and0.479869 respectively, and FL is AIR/load0. Preserved RR placed history is not continuous current support. The final tail instead has both rear legs currently TOP, loads RR0.105939/RL0.514618, with FL still AIR; it remains uncontrolled and nonterminal. Its finite early-P13 nominal wheels are still0.3rad/s, which does not imply permanent rolling or eventual success. None of these endpoints is renamed as successful or improved.

The second episode's628 zero-nominal P13 samples likewise have region/support valid throughout but no sampled controlled stop. Terminal body speed0.118722m/s, angular0.349317rad/s and wheel command0.113466rad/s exceed their existing tolerances; actual wheel speed0.147229rad/s passes. The final tail is only2.733333s into P13: body0.045953m/s and angular0.199087rad/s are below their limits, but actual wheel0.348126rad/s and command0.339161rad/s remain above theirs, with stable duration0. A brief low body-speed endpoint is not0.5s controlled-stop success, and an unfinished early-nominal window is not a completed failure.

The same-MDP continuation preserved the existing learning state rather than reinitializing Adam at a reward boundary. Checkpoint30,080 saves training state, not the third episode's physical P13 state; a later invocation still performs a legal physical reset. The latest completed natural-P01 deterministic evaluation remains C28,032's P12 incomplete. Saving this30,080 checkpoint is not a substitute for a new full-task evaluation or paired stability video.

## Same-MDP natural-P01 training from30,080 — preserved launch record

Actual new invocation: `runs/ppo_semantic_v3/train/20260906T0819495765677Z_g64abc5357d00_b22cdf1222cd4e759de72734f0d5fa7a`. Read-only `run_manifest.started.json` confirms `RUNNING`, unchanged runtime `64abc5357d00763419fe51c8126db8454fcf7697`, v3 `full_episode`, natural P01, offset0, seed1001, N1, **8,192 planned decisions**, and checkpoint cadence4 completed updates. Source is immutable30,080; `new_mdp_warm_start=false` and `resume_migration=null` select full same-MDP saved-state/Adam/RNG continuation, not another migration. Natural-P01 policy sampling does not inherit teacher credit from the preceding P10 block.

This is a launch record only. Current completed totals remain nine runs /19,968 new decisions /156 new PPO updates /3,120 new optimizer steps, lifetime30,080/200/4,000. Source spent budgets remain full_episode4,608/phase_suffix15,360/smoke0 until actual new update accounting is finalized. The requested8,192 decisions and any arithmetic planned endpoint are **not** recorded as executed work, a tenth completed run, or task success. Subsequent actual training and deterministic evaluation outcomes will be reported from their own completed evidence.

## Natural-P01 block finalized:8,192 actual, no full-task success

The08:19:49 run is now finalized `SUCCEEDED`, with planned=requested=actual8,192, unconsumed0 and rounding overrun0. This is completion of the optimizer allocation, not physical task success. Source30,080/200/4,000 advances by64 complete PPO updates /1,280 optimizer steps to immutable38,272/264/5,280, with `save_load_round_trip=true`. The finalized checkpoint SHA is `f8b6e19233283c0678f336072d5febffe94329c097b5662bcb1b400e5b0ddde3`; root's actual checkpoint/sidecar pointer verification agrees. Full-episode spending grows4,608→12,800, while suffix spending remains15,360 and origin remains10,112. The run's training wall field is2,998.4590719s. No new MDP change, Adam reinitialization or teacher credit is introduced by this same-runtime continuation.

Manifest telemetry records all8,192 current-policy decisions /65,529 physical ticks across eight natural-P01 episodes, with success_count0 and seven `INCOMPLETE_CONTROLLER_BLOCKED` terminals. There is no teacher prefix to subtract or credit: the complete policy phase ledger is P01=9, P02=1,496, P03=31, P04=8, P05=1,167, P06=2,120, P07=8, P08=9, P09=2,892, P10=1, P11=1, P12=450 and P13=0. Labels with more than one sampled decision at an entry are preserved as actual data, not normalized to an assumed one-sample transition. This phase ledger sums to8,192 and matches the final training manifest.

The finalized `p01_block_38272.md` verifies contiguous globals30,081–38,272, all64 updates with finite nonzero gradients and actor changes, and an unbroken adjacent actor/global chain. Actor digest changes from `2ffcaf809bb6050334294648d28df85e7a1e5c4c5142010908cc9901956c6e99` to `78e5636f01ee6bc239cc15009a92a25b340d38074196ca5b6ca6745a19b56a8b`. All65,529 physical ticks have verified native target effects, with zero in-episode state writes. The sole short terminal interval is episode1/global32,310:1 actual tick instead of8, explaining8,192×8−7 exactly; these are not missing or unaudited ticks. Eight natural-P01 starts and matching core/policy counts establish zero teacher credit without treating absent suffix-only fields as an explicit false flag.

The seven completed episodes contribute1,181+1,049+1,098+1,049+1,053+987+1,080=7,497 decisions. The final695 decisions, globals37,578–38,272, belong to an eighth nonterminal episode, ending at P09 tick5,560/46.333333s. They are optimized samples, not a completed failure or success. Only the first episode reaches P10–P12; the remaining six terminal episodes end in P09. No episode reaches P13 in this natural-P01 block, regardless of earlier successful rear-leg subevents in teacher-initialized P10 training.

### Natural-P01 rear-leg events and the remaining placement problem

Episode0 supplies1,181 decisions and ends P12 incomplete at78.733333s. RR genuinely qualifies/crosses/places at ticks5,516/5,807/5,829. Later it retreats behind the edge and is observed on ground, while history keeps the completed event. RL qualifies at5,939 and reaches a measured decision-end AIR clearance of+115.489mm at tick6,240, yet that high point is still200.830mm before the front plane. Ground contact at6,632 revokes RL qualification before crossing; it never crosses/places. The endpoint has both rear legs on ground and FL AIR/load0. High clearance alone does not complete forward carry or preserve RR top support.

Episode2 records RR qualification at5,576 and crossing at5,777 but never placement; its terminal RR is AIR before the obstacle. Episodes1,3,4,5 and6 never obtain RR hard qualification, crossing or placement, despite repeated initial-clearance attempts. These P09 incompletes are not equivalent repeated RL-stage experiments: their preceding RR placement is unfinished. More30-second P09 waiting samples are not deeper task progress, and the changing online-policy trajectories do not establish monotonic improvement.

In the final tail, RR qualifies at tick5,387 but `qualification_revoked_ground_before_cross` occurs at5,496. At the final row RR is on ground, front approximately−47.889mm and clearance−49.904mm, with FL AIR/load0. There is no RR crossing/placement and no RL hard-qualified lift in that tail. Its remaining episode outcome is unknown; the checkpoint boundary does not manufacture a terminal event or preserve this physical state for in-place continuation.

Across the seven completed episodes RR qualifies and crosses in two episodes and reaches placement in one; the tail adds a qualified event that is subsequently revoked. RL qualifies only in episode0 and has no crossing/placement anywhere in this block. Episode5 and6 end with both rear legs on ground while FL is actually TOP, loads0.246908 and0.284216 respectively; their continuing RR gap is not evidence that FL is always unloaded. In the final tail RR load is0.044044 and RL is also on ground/load0.573729, while FR is TOP/load0.382227. These current contacts remain distinct from historical placements and do not support a complete-task success claim.

`p01_block_38272.md` retains the earlier bounded31,616 and36,224 snapshots separately from this final block. Its current-policy action samples are not conditional actor means, and FSM nominal suggestions are not actor means either. Different online updates and measured entry/support states do not isolate exploration noise from mean-policy changes; no causal blame or improvement is inferred from these non-paired trajectories. Original incomplete labels, safety predicates and all prior checkpoint/run evidence remain unchanged.

This finalized block brings the ten-run v3 additions to28,160 decisions /220 PPO updates /4,400 optimizer steps and lifetime38,272/264/5,280. The separate C38,272 natural-P01 deterministic evaluation has since completed with the P09 RR wheel-only failure recorded above, and the subsequent current B zero-residual evaluation completed with P09 incomplete. Neither contributes optimizer decisions or a successful task result. The preceding C28,032 P12 incomplete and all training episode records remain unchanged; no later policy revision, migration, training or video is precredited here.

## Policy-distribution revision from38,272 — CPU evidence and preserved launch snapshot

Committed runtime `49749aa527a4a00901e434104821d7e8241ea8da` introduces an explicit policy-architecture boundary from official RSL `GaussianDistribution` with state-independent scalar std (`gaussian_scalar_v1`) to official `HeteroscedasticGaussianDistribution` with a state-dependent log-std head (`heteroscedastic_log_v1`). The existing hidden layers and mean-output rows are retained exactly; the new log-std output weights start at zero and their biases are `log(source learned sigma)`, without sigma clipping. Critic weights and compatible identity normalization are retained. The input remains324 features and the raw output remains12 Gaussian latents before the unchanged tanh/projection path. This is not a new physical MDP, reward revision, success-threshold change, or demonstrated PPO improvement.

The actual `policy_migration_38272_preflight.json` binds source `checkpoint_step_000038272.pt`, SHA-256 `f8b6e19233283c0678f336072d5febffe94329c097b5662bcb1b400e5b0ddde3`, to the committed target runtime. Its exact five production-file delta is `scripts/run_semantic_ppo.ps1`, `src/wlr50_clean/ppo/semantic_cli.py`, `src/wlr50_clean/ppo/semantic_policy_distribution.py`, `src/wlr50_clean/ppo/semantic_training.py` and `src/wlr50_clean/ppo/semantic_video_cli.py`. Physics, all six v3 configurations, reward, nominal motion, supervisor/evaluator and frozen A remain unchanged. The record states `physical_mdp_changed=false`, `reward_changed=false`, `policy_architecture_changed=true` and `exact_optimizer_resume=false`; policy conversion cannot silently stand in for `NewMdpWarmStart` or a physical-state restore.

Root's final combined CPU batch reports **273 passed**, with JUnit artifact `semantic_policy_distribution_final_cpu.xml`; this is one combined batch, not the sum of overlapping earlier test counts. Separately, `policy_mapping_38272_cpu.json` evaluates the actual saved checkpoint on512 archived physical observation vectors from four preserved rollouts. Hidden/mean tensors and critic are exact; measured maximum mean difference and value difference are both0, maximum sigma difference is `1.4901161193847656e-08`, and maximum absolute KL is `2.842170943040401e-14`. Source learned sigma spans approximately0.143311–0.171699. The artifact records no sampled stochastic action, no written checkpoint, no added optimizer update, no added policy decision and no added physics tick. These are forward-equivalence checks on historical observations, not a new rollout, all-phase physical equivalence, bitwise future-trajectory equivalence or task success.

The conversion contract creates **fresh Adam moments at the source's actual effective learning rate1e-5**, also synchronizing the PPO adaptive-LR scalar. It restores the verified source training RNG after constructing and mapping both models and checking their outputs. The old rollout buffer and physical state are discarded; new rollout data must come from a legal reset. Source and target records preserve global38,272 /264 PPO updates /5,280 optimizer steps, original v3 origin10,112, and spent budgets `full_episode=12800`, `phase_suffix=15360`, `smoke=0`. Every migration-added decision/update/optimizer counter is0. The new std-head parameterization prevents exact old-Adam continuation, but does not reset budget spending or erase earlier failures.

Actual live invocation: `runs/ppo_semantic_v3/train/20260906T0946289788164Z_g49749aa527a4_d866c10e684d4248b10521cb7d32bf09`. Read-only PowerShell inspection of its `run_manifest.started.json` confirms `lifecycle=RUNNING`, `command=train`, the exact committed target above, v3/N1/seed1001, `stage=phase_suffix`, `from_phase=P06`, teacher offset0, **8,192 planned decisions**, and checkpoint interval4 completed updates. Source is the immutable38,272 checkpoint; `policy_distribution_migration=true`, `new_mdp_warm_start=false`, `resume_migration=null`, and the resolved policy is `heteroscedastic_log_v1`. The embedded migration record agrees with the separate preflight's five-file delta, source counters, unchanged spending and fresh-Adam/source-RNG contract.

Switching the next sampling curriculum from natural P01 to teacher-initialized P06 occurs after a complete verified PPO update boundary. It changes the curriculum's initial-state sampling, not the physical transition/reward/task definitions; real teacher preparation remains excluded from PPO credit. It is not an in-place continuation of the previous physical episode. At this report boundary only the launch and declared migration are verified: formal live initial-checkpoint/round-trip evidence and new completed optimizer updates remain to be checked from their own artifacts. **No part of the8,192 plan is added to completed totals here.** The ten completed blocks remain28,160 new decisions /220 updates /4,400 optimizer steps, lifetime38,272/264/5,280. Latest completed C and B evaluations remain their recorded non-successes; neither new success nor paired video/stability improvement is claimed or required as an optimizer-launch gate.

## Running policy P06 block — fixed actual boundary40,832

This bounded update supersedes the preceding launch-only status without erasing it. The same09:46:28 run remains active. Its separately saved initial policy checkpoint is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_policy_from_000038272_sf8b6e1923328_g49749aa527a4_b6fb79ec05cd8925253063170b07aedd10afc0c8d0cc2cd1bb6e37825c365391_p54871747db0d.pt`, recorded SHA-256 `052ca0b75082fa4c34c1112bb4a3c127c427a71bd0d735b4e572dce1efd5e251`, with `stage=initial_policy_distribution_migration` and `save_load_round_trip=true`. It retains38,272/264/5,280, original origin10,112 and the unchanged source-stage spending. This is now actual saved live migration evidence, not merely the preflight declaration or the CPU-only mapping test.

The initial sidecar records exact retained hidden/mean weights and critic/identity normalization, `fresh_Adam_no_old_moments` at1e-5, rollout step0 and restored source RNG. On the real current observation the mapping comparison gives mean/value differences0, std difference `1.4901161193847656e-08` and absolute KL `4.263256414560601e-14`; these live single-observation numbers are distinct from the512-history-observation CPU result above. Mapping itself adds no physics/sample/update credit. Actual P06 teacher preparation has already occurred before that observation, so the mapping's zero added-physics field must not be mistaken for zero prefix time or physical reset cost.

The immutable40,832 sidecar binds the new official heteroscedastic-log policy to the same runtime and records284 PPO updates /5,680 optimizer steps, `save_load_round_trip=true`, and checkpoint SHA-256 `4afab9654294f1ac83aec6079b3b472a4c617e46111b6b74b4a2af11d29e73f4`. This is exactly **2,560 optimized decisions /20 complete updates /400 optimizer steps** after source38,272. Budget spending is full_episode12,800, phase_suffix17,920, smoke0, origin10,112. Of the8,192 request,5,632 is outside this fixed saved-boundary account; it is not asserted to remain unexecuted at every later instant of the active run. No planned46,464 checkpoint or eventual run completion is credited.

The first three completed episodes provide801,822 and704 credited decisions, respectively, and terminate **P12 incomplete, P12 incomplete, P09 incomplete** (`INCOMPLETE_CONTROLLER_BLOCKED`). Their2,327 decisions plus the fourth episode's233-decision nonterminal tail account for2,560 exactly. These are teacher-initialized suffix results, not natural-P01 success. Current-policy RR qualified/cross/placed subevents in the first two episodes do not imply sustained top support or completion; the third episode's RR qualifies at tick5,949 but does not cross before ground revokes eligibility at6,453. The second episode's RL qualifies at6,729 and6,893, with ground revocations at6,799 and7,225 and no crossing/placement. Repeated lift attempts or historical placement bits are not successes. `policy_p06_block_46464.md` preserves the earlier38,784 and39,296 snapshots and the detailed bounded episode ledger; the number in that report's filename is a planned endpoint, not a completed result.

Completed additions since origin are now30,720 decisions /240 PPO updates /4,800 optimizer steps, while the number of finalized training blocks stays ten. Latest reloaded deterministic full-P01 evaluation remains C38,272's49.358333s/P09 wheel-only failure; neither this partial stochastic block nor policy-equivalence tests replace a new full evaluation. No successful new video or paired improvement is claimed.

## Final-delivery checklist at40,832 — evidence gaps, not training gates

- A reloadable latest training checkpoint and migration provenance exist, but no verified complete natural-P01 task-success checkpoint exists. `checkpoint_first_full_success`, success-video and improved names must remain absent until their respective physical-success or paired-improvement evidence exists. A first genuine full success should be preserved without waiting for an arbitrary improvement percentage.
- Consolidated, version-matched A/B/C result tables with explicit denominators, approximately five predefined paired validation runs, and a separately locked candidate/test set are not yet complete. Phase quality must separate active transfer from captured/stable windows and include completion/time, attitude/rates, acceleration, clearance/lift-cross-place events, contact/bounce, slip, applied-action changes and P13 stop behavior. Unvisited phases are missing data, not zero-error samples. A→C and B→C comparisons remain separate.
- The training history documents real teacher prefixes, sampled phases and concrete rear-leg gaps, but the final consolidated natural-entry geometry/velocity/contact distributions and ensuing continuous P07–P10/P10–P13 crossing/placement rates are still outstanding. Residual range/projection limits and the actual native workspace response already have `metrics/residual_workspace_audit.csv`, its manifest and the probe reports; final presentation must preserve their state-dependent and partial-probe limitations rather than call all configured authority physically validated.
- No current same-condition v3 A/C source pair or required `fsm_baseline_clean.mp4`, `ppo_success_clean.mp4`, `fsm_vs_ppo_success.mp4` is delivered. The production recorder already supports v3 and observes the last64 ticks of the existing180-tick settle without adding64 warmup ticks; `posttrain_video_route.md` predates that implementation and its missing-v3-interface statements are historical, not current code blockers. Real source/PTS/decode/frame/SHA validation and outcome-labelled comparison remain to be executed. The standalone neutral-outcome comparison helper is only a prepared artifact, not an executed video result. An incomplete A is a labelled diagnostic, not a demand to rerun A until success or5/5.
- The final handoff still needs one manifest/index containing real absolute checkpoint/config/normalization/nominal/action/metric/video paths and checksums, plus the exact recovery command for the eventual stopped, verified boundary. A post-conversion checkpoint on unchanged49749 uses ordinary full saved-state resume with its actual policy inferred from metadata; it must not reapply `-PolicyDistributionMigration` or `-NewMdpWarmStart`. Current live training is not stopped or duplicated to manufacture that final handoff.

These gaps describe unfinished user delivery, not extra software/physics prerequisites for continuing authorized PPO. CPU tests, interface probes, suffix subevents and optimizer execution success remain distinct from whole-task success and stability improvement.

## Policy-architecture P06 block finalized:8,192 actual, no suffix success

The09:46:28 run under `49749aa527a4a00901e434104821d7e8241ea8da` is now finalized `SUCCEEDED` in both run and training manifests; root records process exit0. Planned=requested=actual8,192, unconsumed0 and rounding overrun0. Source38,272/264/5,280 advances by64 complete PPO updates /1,280 optimizer steps to **46,464/328/6,560**. The training wall field is4,776.261459799949s. This supersedes the launch and bounded40,832/45,440 statuses without rewriting their historical observations. `SUCCEEDED` means the optimizer allocation executed, not that any physical episode completed the task.

Final immutable checkpoint is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000046464.pt`, checkpoint SHA-256 `fc1fd6710965d627aff6e6ecc91e02c5ed9c144f7637cba9a50cfa079cf5513d` and sidecar SHA-256 `10b34e9c076a83c22f4af4871c6d837795cc12c59200db21d2df2bc0d5d19a40`, both verified by root against the actual saved artifacts. The sidecar records `save_load_round_trip=true`, actual policy `heteroscedastic_log_v1`, actor hash `571f19b7ccc64593456eb4a90cd9163c38b273d2e65305b2b87f27b6c41df205`, original v3 origin10,112, and spent full_episode12,800/phase_suffix23,552/smoke0. The initial conversion actor hash was `3ba295ffd216eb1c2e87f476559f36c8ff89e122979c607bcb41583597fba517`; the final hash is a learned state after real updates, not merely a renamed old checkpoint. Architecture migration, CPU equivalence and its zero-credit initial publication remain separately documented above.

Actual policy phase counts are P06=3,225, P07=10, P08=12, P09=2,685, P10=5, P11=5 and P12=2,250; P01–P05 and P13 receive0 current-policy samples in this suffix block. These8,192 credits advance **65,529 physical ticks**, while11 real teacher prefixes each use448 decisions /3,584 ticks. The combined teacher account is **4,928 decisions /39,424 ticks**, excluded from PPO storage/global counts. Physical core including preparation is13,120 decisions /104,953 ticks across11 separate episodes. The200s task clock includes each real prefix; counts across resets are not presented as one continuous episode. Manifest reset wall38.28408839995973s and roll-in wall1,807.6189648001455s have distinct scopes and are not added blindly to the training wall field.

There are **10 formal `INCOMPLETE_CONTROLLER_BLOCKED` episodes and an unfinished eleventh P06 tail**. Completed episodes contribute8,032 credited decisions; the final160 decisions, globals46,305–46,464, are nonterminal optimized tail data, not an eleventh failure or a success. Both fresh-P01-scoped `success_count` and `teacher_initialized_task_success_count` are0. The first eight completed episodes account for6,308 decisions; episode8 adds916 and ends P12 incomplete at90.933333s, and episode9 adds808 and ends P09 incomplete at83.733333s. No episode supplies P13 policy data. Preserved RR qualified/cross/placed events in some suffix episodes and isolated RL qualified events are not task completion or fresh-P01 rear-leg competence; ground revocations and subsequent current contacts remain part of the actual result.

The finalized stream audit confirms all65,529 credited ticks have verified native target effects, with zero in-episode state writes. The sole short interval is episode3/global41,342:1 actual tick rather than8, accounting exactly for8,192×8−7; no missing ticks are filled in. All64 updates have actor changes, finite nonzero gradients and consistent adjacent actor/global hash chains. Raw-action log-probability recomputation from the recorded12-channel means/stds differs by at most approximately1.95144e-6 under the separate double-versus-float calculation. These are learning/execution integrity checks, not physical success or evidence that the state-dependent exploration is beneficial.

Among the10 completed episodes, RR has qualified lift in9, crossing in6 and placement in5; **all5 episodes with RR placement eventually terminate with RR currently on ground**. RL has6 qualified attempts across episodes1,6,7 and8, all revoked before crossing, with no RL crossing/placement anywhere in the block. Episode8 records RR qualified/cross/placed at ticks6,964/7,154/7,289 and RL qualification7,470 revoked by ground at7,498; it still ends P12 incomplete. Episode9 records RR qualification6,805 revoked at6,876, without crossing/placement, and ends P09 incomplete. Earlier episode1 includes RL qualification/revocation while RR is actually TOP, so RL failure cannot uniformly be attributed to RR first losing support. The final160-decision P06 tail ends at tick4,864/40.533333s with RR/RL on ground and neither rear leg carrying qualified/cross/placed history in that episode. No unfinished tail is promoted to a terminal result.

The finalized phase table at the top now totals **36,352 new decisions /284 new PPO updates /5,680 new optimizer steps** across11 completed blocks, lifetime46,464/328/6,560. `policy_p06_block_46464.md` is the detailed episode/native/distribution ledger; its older38,784,39,296,40,832 and45,440 appendices preserve what was known at those earlier boundaries. Their planned/bounded labels are not used to contradict the actual final8,192 outcome.

## Checkpoint46,464 natural-P01 evaluation — preserved launch snapshot

Actual invocation `runs/ppo_semantic_v3/validation/20260906T1109596110104Z_g49749aa527a4_df39981a0f5c40cd921698578dec5e67` has a verified `run_manifest.started.json` with `lifecycle=RUNNING`. It selects saved immutable46,464 on the same49749 runtime, v3/N1/seed2001, `semantic_residual_eval`, natural P01, teacher offset0 and the checkpoint-selected heteroscedastic-log architecture. Neither `new_mdp_warm_start` nor `policy_distribution_migration` is enabled. This is the fixed saved-mean deterministic evaluation path, not continuation of training noise or reuse of the last suffix physical state.

Until this run produces its own final result, **latest completed full-P01 evaluation remains C38,272:741 decisions /5,923 ticks /49.358333s, P09 RR WHEEL_ONLY_CLIMB**, with the separately preserved current-B P09 incomplete. No outcome, duration, new optimizer credit, complete-task success or improvement is inferred for C46,464. There is still no new successful video or validated same-condition A/C comparison; final-delivery gaps remain evidence to produce, not additional gates that undo authorized training.

## C46,464 finalized: valid P06 incomplete, not a physical collision/climb failure

The11:09:59 natural-P01 evaluation is now finalized `SUCCEEDED` in its run manifest, with939 decisions /7,512 ticks /62.6s and0 optimizer updates. The saved46,464 checkpoint, same49749 runtime, seed2001 and `heteroscedastic_log_v1` fixed mean are explicit; no teacher prefix, policy conversion or new-MDP warm start was used during evaluation. This supersedes the preserved RUNNING snapshot above. Both task success and controller success are false, termination is `INCOMPLETE_CONTROLLER_BLOCKED`, and `window_ended_before_task_terminal=false`: the controller reached its actual P06 deadline, rather than a recorder/window truncation. Shared physical evaluation remains `valid=true`, `success=false`, `termination_reason=null` and empty physical-failure reason. The absence of a collision/climb failure is not completion.

The939 current-policy decisions divide into P01=1, P02=185, P03=4, P04=1, P05=148 and **P06=600**, with no P07–P13 samples. FR qualified/cross/placed at ticks53/1,498/1,516; FL at1,610/2,568/2,711. Both rear legs remain without hard qualified/cross/placed history. The early RL initial-clearance event at tick9 is not qualified rear traversal. Prior placed history does not establish current FL support: at terminal FL is AIR, obstacle/ground pair inactive, clearance+4.132395mm and load0. FR remains TOP/load0.434381; RR and RL are on ground with loads0.070784 and0.494834 respectively.

The first unmet P06 workspace item is RL's front distance **−0.2368337775962388m**,16.833778mm short of the existing−0.22m lower bound. RR's−0.21683358805346842m is3.166412mm inside that particular lower bound. These measured distances identify the incomplete task goal; they do not prove a causal motor limitation, justify reducing its threshold, or establish that changing a reward will fix it. Final region/support/control checks remain false. C46,464 stopping at P06 and earlier C38,272 failing at P09 are distinct non-successes; the different duration or absence of a wheel-only event is not reported as stability improvement. No new successful video exists.

## Unplaced-leg workspace shaping — preserved implementation-plan snapshot

The selected revision is limited to opt-in workspace progress for an **unplaced leg even before all predecessors are placed**. This changes the shaping signal, not permission to mark a lift, carry, unload, crossing, placement or final task success out of order. Existing hard histories, physical task/evaluator criteria, nominal suggestions, action projection/caps and learned policy architecture are to remain unchanged. It is not an evaluator/nominal repair, and it must not retroactively relabel C46,464 or any earlier result. Production changes and targeted tests are being implemented separately; combined verification and a new committed runtime are not claimed complete in this snapshot.

If that revision passes review, the planned explicit new-MDP path starts from immutable46,464 and keeps all learned heteroscedastic mean/log-std weights, critic and compatible identity normalization. Adam moments are intentionally reset at3e-5 instead of preserving the source's actual effective1e-5; training RNG and original origin10,112 are retained, with fresh rollout storage and legal physical reset. Existing full_episode12,800/phase_suffix23,552 spending and lifetime46,464/328/6,560 are not reset or reissued. Root's planned next natural-P01 allocation is4,096 decisions **after** verification; there is no started-manifest path or executed budget to record yet. The eleven-block actual additions therefore remain36,352/284/5,680, with no added migration/training/evaluation-success credit from this implementation work.

## Committed workspace shaping and real P01 launch —9d70, initial boundary verified

Runtime `9d70aae58243b9fb2248d64561959e6a67901b44` changes only `src/wlr50_clean/ppo/semantic_supervisor.py` and `configs/ppo_semantic_v3/stage_task_spec.yaml` in the production inventory, plus three test files. The exact two-file runtime delta is independently present in this launch's new-MDP record. The opt-in unplaced-leg workspace credit does not unlock hard lift/carry/unload or crossing/placement order, change nominal motion, enlarge action caps, alter physical failure/success criteria, or replace the learned heteroscedastic network. The shaping and its derived potential feature are intentionally new semantics; the prior C46,464 incomplete record remains unchanged.

PowerShell inspection of `C:/robotics_sim/wlr_robot/semantic_preparation_all_cpu.xml` confirms **634 tests,0 failures,0 errors,0 skipped,81.469s**, for the final combined30-semantic-module batch. These are CPU regression results, not new physics, optimized samples or task success. Root's separate current-helper diagnostic on the archived C46,464 tick7,504 observation gives potential old0.4675→new0.4873191238379249. Holding the other diagnostic inputs fixed and varying only RL front distance shows the intended workspace gradient:

| Diagnostic RL front distance m | New potential | RL workspace goal progress |
|---|---:|---:|
| −0.240 | 0.48705 | 0.92 |
| −0.230 | 0.48790 | 0.96 |
| −0.220 | 0.48875 | 1.00 |
| −0.215 | 0.48875 | 1.00 |

This is an offline helper sensitivity check, with no physics advancement and no new PPO action. It shows a changed numerical credit and its plateau, not that the real robot can reach the workspace or that the policy will improve. The hard workspace boundary itself remains−0.22m.

Actual run: `runs/ppo_semantic_v3/train/20260906T1133239688401Z_g9d70aae58243_1e3e34207a9442c387f5123a55a5e121`. Its `run_manifest.started.json` is `RUNNING` and binds v3/N1/seed1001, `stage=full_episode`, natural P01, teacher offset0, **4,096 planned decisions**, checkpoint interval4 updates, `new_mdp_warm_start=true` and `policy_distribution_migration=false`. It selects `heteroscedastic_log_v1` directly from immutable source46,464, SHA-256 `fc1fd6710965d627aff6e6ecc91e02c5ed9c144f7637cba9a50cfa079cf5513d`; this is not a second Gaussian-to-log-std conversion or teacher-initialized P06 roll-in.

The actual separate initial is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000046464_sfc1fd6710965_g9d70aae58243_d098c4e8ec88d40a9cf2246b629cdd0b0e695dfd1363564994ab66f3aebfe55e.pt`, recorded checkpoint SHA-256 `b52f11f195bd2ee1539ee1c72bb6f2542795f8091d37fbb22486d41effb68545`, `stage=initial_v3_warm_start` and `save_load_round_trip=true`. Read-only source/initial sidecar comparison confirms:

| Saved state | Source and initial evidence |
|---|---|
| Actor, including learned mean/log-std head | Same hash `571f19b7ccc64593456eb4a90cd9163c38b273d2e65305b2b87f27b6c41df205` |
| Critic | Same hash `e2625f6d407c0b741fd0a6fd48019bd29eaac903dcef99e11356172770dec55f` |
| Identity normalizer | Same hash `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552` |
| Training RNG | Complete serialized state values equal between source and initial |
| Lifetime counters |46,464 decisions /328 PPO updates /6,560 optimizer steps, unchanged |
| Original origin and spending | origin10,112; full_episode12,800 /phase_suffix23,552 /smoke0, unchanged |

The optimizer digest deliberately changes from source `ca143b116e5c869bc7e21e0be46c28fd68929f05750b10ac7dc6da48be7a0870` to initial `ebddf13dce59639efaf4e3951e4a4db11f59a2b1e7a9f70fe1168320ba716614`, and recorded effective LR changes1e-5→3e-5. The verified loader first strictly restores/hashes the saved networks and optimizer, then creates fresh Adam moments, requires fresh128×1×12 rollout storage and restores source RNG before publication. No optimizer step uses old moments after this declared new-MDP boundary. `physical_env_state_saved=false` and `resume_physics=legal_reset_not_bitwise_continuation` remain explicit; no earlier physical episode or spent rollout is resumed in place. This documentation comparison reads sidecars and the implemented loader contract, not a concurrent Python deserialization of the live checkpoint.

The initial sidecar also binds this run's `new_mdp_initial_action_comparison.json`, SHA-256 `ce7cde8de111c6b34435938cf5d0057ccd7bf9f1150dc913164398aa19049b88`. At P01/tick0 it feeds the **same current NEW-MDP observation** and current nominal into two fresh logical projectors; old and new execution-profile SHA are identical (`79f99a0ac8434ca7b99778a4f49669b64db2af15e1f6e069aa527e74054fb004`), and all recorded old/new residual/projected/applied differences are0. It records no environment/bridge-history modification and0 optimizer updates. This establishes same-input/profile consistency only: it does not compare the old-MDP potential observation with the revised one. Because the derived potential feature can differ despite unchanged324-feature ordering/scaling and preserved weights, the same physical state may produce a different mean or std; neither action equivalence across MDPs nor bitwise future behavior is claimed.

At this launch/initial-checkpoint boundary **4,096 is still a plan**, not a twelfth completed block or a completed50,560 checkpoint. The eleven completed blocks and actual totals remain36,352 new decisions /284 updates /5,680 optimizer steps, lifetime46,464/328/6,560. Latest completed deterministic P01 remains C46,464's62.6s/P06 incomplete. New optimized decisions and task outcomes await their own saved-update/final-run evidence; no new success or video is credited here.

## Natural-P01 preparation block finalized:4,096 actual, three P09 incompletes

The11:33:23 run `runs/ppo_semantic_v3/train/20260906T1133239688401Z_g9d70aae58243_1e3e34207a9442c387f5123a55a5e121` is now finalized `SUCCEEDED` execution in its run/training manifests. Planned=requested=actual4,096, unconsumed0 and rounding overrun0. It performs **32 actual PPO updates /640 optimizer steps**, advancing source46,464/328/6,560 to **50,560/360/7,200**. Reported training wall time is1,553.4874382000417s. This final evidence supersedes the initial plan and the preserved48,512 bounded snapshot; it is not a claim that a physical task succeeded.

The final immutable checkpoint is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000050560.pt`, SHA-256 `6ccf5b26784e4db7056b1ba6521d0ac34c3c49cf915ff618d4410c9f14d7fd99`, with sidecar SHA-256 `69bb996011ce3f12730bcfc0511887174631e669dd5600366ec1e5bf74f88702`; root verified both actual hashes and `save_load_round_trip=true`. The sidecar binds9d70 and `heteroscedastic_log_v1`, with final actor hash `2f2bfb60ae85605afdbf50d919d6b6b16b49f33400bf9dc70d404abb13d9c207`. Original origin10,112 is unchanged; full-episode spending grows12,800→16,896 and suffix spending remains23,552, with smoke0. The earlier46,464 checkpoint and both migration lineages remain intact.

All **4,096 credited decisions /32,768 physical ticks** come from four real natural-P01 current-policy episodes, with no teacher prefix. Manifest sampling is `P01_full_task_only_initial_version`, and physical core/policy counts agree instead of hiding preparation outside the policy ledger. The exact phase ledger is P01=4, P02=731, P03=16, P04=4, P05=587, P06=1,375, P07=6, P08=5 and P09=1,368; P10–P13=0. It sums to4,096 and matches the top-level twelve-block table. This is actual policy sampling of the front-leg and whole-body preparation, not teacher credit from the preceding P06 suffix run.

Three episodes formally terminate in **P09 `INCOMPLETE_CONTROLLER_BLOCKED`**, totaling3,377 credited decisions. The remaining719 decisions, globals49,842–50,560, are an unfinished fourth P09 episode; they are optimized samples but not a fourth completed failure or success. Manifest `success_count=0`. All four natural episodes reach P09, but none produces P10–P13 policy data. Reaching P09 after the preparation revision does not establish successful RR/RL continuation, final stopping, or paired PPO stability improvement. `p01_preparation_block_50560.md` owns the detailed episode/entry/native ledger; its prior48,512 account remains a bounded historical subset of this final block.

The twelve completed blocks now contribute **40,448 new policy decisions /316 PPO updates /6,320 optimizer steps** since origin10,112/44/880. The saved50,560 actor is a real updated recovery point, not a full-task-success checkpoint; saving and reloading it does not restore the unfinished physical P09 state. The workspace-shaping change and any observed progress are not retroactively attributed solely to learned improvement or used to relabel previous C/A failures.

## C50,560 fresh-P01 evaluation — preserved launch snapshot, superseded below

Actual new invocation is `runs/ppo_semantic_v3/validation/20260906T1200142665931Z_g9d70aae58243_bb0d667703da4eca80a5c99292560192`. Its read-only started manifest confirms `RUNNING`, same9d70 runtime, v3/N1/seed2001, `semantic_residual_eval`, natural P01, teacher offset0, immutable50,560 and checkpoint-selected `heteroscedastic_log_v1`. Neither `new_mdp_warm_start` nor `policy_distribution_migration` is enabled. It evaluates the fixed saved deterministic mean rather than continuing stochastic training or the last episode's physical state.

No outcome, duration, successful task/video or optimizer credit is inferred for this live evaluation. **Latest completed natural-P01 evaluation remains C46,464:939 decisions /7,512 ticks /62.6s, valid P06 incomplete**, with the earlier C38,272 wheel-only failure, current-B incomplete and historical A diagnosis preserved. Existing non-successful A evidence is not turned into a new prerequisite or a demand for repeated successful A runs. There is still no new successful or validated same-condition A/C comparison video.

## C50,560 finalized: measured P06 completion, then P09 incomplete

The preceding launch snapshot is superseded by finalized execution `SUCCEEDED` and the actual1,166-decision /9,328-tick /77.733333333s C50,560 result. Saved fixed-mean natural-P01 seed2001 evaluation performs0 optimizer updates; task/controller success=false, physical evaluation valid=true with no physical failure, and actual terminal reason `INCOMPLETE_CONTROLLER_BLOCKED`. The window did not end before the task terminal. P01–P13 request counts are1,186,4,1,150,372,1,1,450,0,0,0,0. `eval_50560_diagnosis.md` binds the source checkpoint, exact run and bounded live measurements.

P06→P07 occurs at tick5,712/47.6s with RR/RL front distances−219.232258/−217.444714mm, both satisfying the unchanged−220mm workspace minimum. P07/P08 each take one decision; P09 begins at5,728/47.733333333s and lasts450 decisions/30s. Its first unfinished task is RR placement, with the prerequisite above-top active lift already absent. RR initial-clearance events at6,198/6,495/6,604 have real motion evidence but **no Q/C/P**. Across the actual120Hz P09 window, RR's highest AIR wheel-bottom clearance is−6.232581mm at6,312, and its closest front position remains−22.421267mm at6,317. The final RR is GROUND/front−122.911480mm/load0.05007624; RL is GROUND/front−280.198526mm/load0.47864871. FL is AIR/load0 at entry, all three initial events and terminal, but has small genuine contact load0.02521791 at RR's peak; this is not evidence of uninterrupted FL support or a motor-causal explanation.

From logged decision-end tick6,640 onward,337 nominal vectors are static `[-18.5,-31.4,0,31.1,15.4,19.4,-6.9,-37.8,0,0,0,0]`: four wheels zero, not all12 channels zero. Residuals and verified distinct actual native targets continue through terminal. This finite nominal retirement does not constitute a codec/recording failure or close the residual interface. It also does not complete the missing RR task. No new suffix/full success, improved-policy claim or successful/paired video follows from this evaluation, and training counters remain unchanged.

## Same-MDP P10 continuation from50,560 — preserved launch snapshot

Actual started manifest `runs/ppo_semantic_v3/train/20260906T1211309264074Z_g9d70aae58243_42ace02dcac84aaa83ee7e1ee578f58e/run_manifest.started.json` is `RUNNING` and binds same9d70/v3/N1/seed1001, source `checkpoint_step_000050560.pt`, checkpoint-selected `heteroscedastic_log_v1`, `stage=phase_suffix`, `from_phase=P10`, teacher offset0,2,048 planned decisions and checkpoint interval4 updates. `new_mdp_warm_start=false`, `policy_distribution_migration=false` and `resume_migration=null`: this is exact-resume continuation with a new sampling start, not new learned weights/MDP or a fresh-Adam migration. Actual restoration/update results await their own evidence; none is inferred from the launch arguments alone.

Teacher preparation and its earlier rear-leg histories remain outside PPO decision/reward/optimizer credit; reaching P10 via that prefix is not a natural-P01 learned success. No planned52,608 endpoint, new block completion, phase counts or success is prefilled. The current completed ledger remains twelve blocks,50,560/360/7,200 lifetime and40,448/316/6,320 added since origin10,112/44/880.

## Same-MDP P10 block finalized at52,608 — two P13 incompletes plus tail

The12:11:30 run is now finalized `SUCCEEDED` execution; root confirms process exit0. Its final manifests supersede the preceding launch-only snapshot and `p10_block_52608.md`'s preserved51,584 subset. Actual=requested=planned2,048, unconsumed0 and rounding overrun0;16 updates /320 optimizer steps advance source50,560/360/7,200 to **52,608/376/7,520**. All16 update records show changed actor parameters and finite nonzero gradients. Training wall time is1,459.4089381000958s. Final checkpoint and sidecar hashes are bound in the top summary; root measured both actual files and confirmed save-load roundtrip.

Final policy phase counts are P10=3, P11=7, P12=195, P13=1,843 (all other phases0), summing to2,048 decisions /16,384 ticks. Three accepted reset-only teacher prefixes each execute948 decisions /7,584 ticks, totaling2,844 /22,752, outside PPO credit. The physical core consequently contains4,892 decisions /39,136 ticks; none of the extra teacher data is added to policy budgets. Current spending is full_episode16,896 /phase_suffix25,600 /smoke0 and original origin10,112 remains unchanged.

| Actual episode | Policy globals / decisions | Terminal or saved-tail outcome |
|---|---|---|
|0 |50,561–51,530 /970 |P13 `INCOMPLETE_CONTROLLER_BLOCKED`;127.866666667s/tick15,344 |
|1 |51,531–52,500 /970 |P13 `INCOMPLETE_CONTROLLER_BLOCKED`;127.866666667s/tick15,344 |
|2 tail |52,501–52,608 /108 |P13 at70.4s/tick8,448; termination=null, unfinished;43 P13 samples |

New RL Q/C/P ticks are7,957/8,076/8,142,7,982/8,133/8,138 and7,956/8,071/8,102 respectively, all after the7,584 policy takeover. RR Q/C/P6,938/7,109/7,579 and the earlier FR/FL history are teacher events in each episode and are excluded. New RL placement is real policy suffix experience, not policy completion of all four legs from P01.

Both completed episodes reach P13's60s deadline with all12 nominal targets zero, historical all-placed=true and current region/support=true, but controlled=false/stable0. At both terminals FL/RR are TOP while FR/RL are AIR/load0. Episode0 command/body linear/angular are0.178885413rad/s /0.124574697m/s /0.335342168rad/s; episode1 is0.188053472rad/s /0.060123169m/s /0.086366561rad/s. Thus the0.02rad/s commanded stop bound is missed in both, body linear speed exceeds0.05m/s in both, and episode0 also exceeds0.30rad/s body angular speed. The report's detailed episode0 zero-nominal μ/σ statistics are measured diagnostics, not independent-Gaussian physical-success estimates or proof of a single failure cause.

The final108-decision tail is different: only2.866667s into P13, its four nominal wheels are still+0.3rad/s and servo suggestions remain nonzero. FR/RL/RR currently contact TOP while FL is AIR/load0; region/support=true but controlled=false, with actual wheel command0.437127583rad/s and measured wheel speed0.909590364rad/s. It is not a completed third failure or evidence of permanently inherited nominal motion. Manifest success counts remain0. **No suffix or full-task success and no new successful/paired video are credited**.

Thirteen finalized blocks now add42,496 decisions /332 updates /6,640 optimizer steps relative to origin10,112/44/880. Latest completed saved fixed-mean natural-P01 evaluation remains **C50,560 P09 incomplete,1,166 decisions /9,328 ticks /77.733333333s**. C52,608 has not been given a full-P01 evaluation in this record, and saving it does not restore a physical P13 tail.

## Separate nominal-geometry revision — preserved READY implementation plan

Root has started preparing the next `nominal_geometry` execution revision after this completed9d70 block. At this report boundary the prospective production changes are **not yet committed, not test-verified and not enabled in a new rollout**. There is no new runtime identity, CPU-pass claim, migration execution, training count or task improvement to record. The current learned52,608 checkpoint and every historical result remain unchanged; any eventual controller/prior change must be labeled separately from PPO learning. This planning record is not an extra A/probe success gate.

## Nominal geometry committed, initial state preserved, first128-decision update verified

The preceding READY plan is superseded by actual commit `4d268fc547b704c590c5f21563ea4b1970b021a1`: six production files (versioned v3 execution profile, nominal geometry/projection helpers, semantic backend/dispatch and additive native audit) plus four test files. PowerShell read of `C:/robotics_sim/wlr_robot/semantic_geometry_final_cpu.xml` verifies **841 tests /0 failures /0 errors /0 skipped /83.416s**. Earlier partial207/124/72-test receipts overlap and are not added as unique counts. These are CPU/interface checks, not task-success evidence.

Independent protection audit in `nominal_geometry_implementation_audit.md` measured the old52,608 checkpoint and sidecar hashes as unchanged6956b02a…8582 /353a9b32…4caa and verified all29 frozen files against the original inventory, source checkpoint and prior run-start bindings, with0 discrepancies. The geometry layer changes only an eligible current-state rear nominal suggestion, not the old FSM files, task evaluator, phase clock or historical-entry gates; its zero-residual B behavior changes too. Any subsequent controller/prior effect remains separately attributed from PPO learning.

Actual new run `runs/ppo_semantic_v3/train/20260906T1308472858273Z_g4d268fc547b7_dcefe730da37448599db863a5f261d47` binds4d268fc/v3/N1/seed1001, P06/teacher offset0, phase_suffix2,048 requested and checkpoint interval4. `new_mdp_warm_start=true`, `policy_distribution_migration=false`, `resume_migration=null`, with checkpoint-selected `heteroscedastic_log_v1`. It is an explicit new execution/MDP boundary, not exact same-MDP Adam continuation or another Gaussian conversion.

Actual initial checkpoint is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000052608_s6956b02aaec6_g4d268fc547b7_64b915174f5f4edd7dde25494fbceb18f907c1979a0731c0e46b1beb6047a9fe.pt`. Its actual SHA-256 is **`3f0854ff48013b52ca8572a0fe5446f9bb8adf6334f8f10b3af4e5c3e9ae1310`**, matching the initial manifest, and `save_load_round_trip=true`. The paired source/initial metadata gives:

| State | Actual initial migration evidence |
|---|---|
| Actor including learned heteroscedastic mean/log-std |Digest unchanged `7b7996a8555baa1289a842b6301020ad30120c773d10f43b507fdffd87a5c3d5` |
| Critic |Digest unchanged `180db1155f101fac765a73b6f3434929786530c33ca3a0b44713808283d22457` |
| Identity normalizer |Digest unchanged `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552` |
| Stored RNG |Complete source/initial `training_rng_state` objects compare equal |
| Initial lifetime/origin/budgets |52,608/376/7,520; origin10,112; full_episode16,896/phase_suffix25,600/smoke0, unchanged |
| Adam |Source `5e8b4906a15b55620bc033809af5a9933fd80142cd8c495a4d9bbfaf60708137` → fresh `ebddf13dce59639efaf4e3951e4a4db11f59a2b1e7a9f70fe1168320ba716614`; moments reset |
| Initial learning rate |Source effective1e-5 → new initial3e-5 |
| Rollout/physical state |Neither inherited; source RNG restored for fresh new-MDP rollout after legal reset; teacher remains outside PPO credit |

The initial preservation evidence asserts retained network/state, not identical closed-loop actions under a changed nominal execution rule. No credit is created by the migration itself. The initial checkpoint's source hashes bind the unmodified52,608 payload and sidecar; its unchanged324×12 network contract retains the learned heteroscedastic architecture.

The actual first update record is **PPO update377 /global52,736 /20 optimizer steps**, adding128 new decisions. Actor digest changes from the preserved7b7996a8… to `2f3752cbf394c64324b819c1966cf502bf8ff242547ce09bfdb9e27d9e72fc2c`, with `actor_parameters_changed=true`, `finite_nonzero_gradient_observed=true` and gradient norms1.002740108–1.366367730. The update receipt and saved sidecar record an actual effective LR of **1e-5**; the3e-5 value above is the verified initialization rate, not a promise of constant LR throughout the block.

Immutable `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000052736.pt` has actual SHA-256 **`3ef19a21a812663373a517dbf3aced46bdc6d1738f3f2e2c6701035ad6609617`**, matching its manifest and `save_load_round_trip=true`. It binds the new4d268fc run and52,736/377/7,540, with current full_episode16,896 /phase_suffix25,728 /smoke0. This is a real saved optimization boundary, not a completed fourteenth block, physical task outcome or proof that the new P09/P12 geometry intervention improved a trajectory. The thirteen completed-block tables above remain finalized-history totals; this separate running contribution is **128/1/20 only**. Later updates, remaining requested decisions, full evaluation and video outcomes are deliberately not rolled forward in this report snapshot.

## Nominal-geometry P06 finalized:2,048 actual, P09/P12 incompletes plus tail

The preceding52,736 first-update snapshot is superseded by the completed run's final manifests, not erased or counted again. Run13:08:47 under4d268fc is `SUCCEEDED` execution, with root-confirmed exit0, planned=requested=actual2,048, unconsumed0, rounding_overrun0 and wall-time1,137.38724800013s. Sixteen updates377–392 perform320 optimizer steps, advancing source52,608/376/7,520 to **54,656/392/7,840**. The final checkpoint and sidecar hashes are bound in the top summary and root measured both actual files with roundtriptrue.

The complete read-only audit finds2,048 consecutive policy globals52,609–54,656,16,384 physics ticks,16,384 verified/effect ticks,16,372 own-phase-request-effect ticks and zero state-write/teacher-in-storage issues. The12 incoming handoff ticks explain the own-effect difference. Recorded raw actions/means/std are12-dimensional and finite with positive std; Gaussian latent log-probability recomputation has maximum absolute error1.3468938224736604e-6 at53,823. All16 actor hash links and expected128-decision/20-optimizer update increments match, every update records changed actor/finite nonzero gradients, and the last actor digest equals the sidecar. PPO configuration is5 learning epochs ×4 minibatches; all actual update LR receipts are1e-5, distinct from new-MDP initialization3e-5. These are execution/bookkeeping checks, not physical-success probabilities.

The final P06–P12 policy ledger is **972,3,4,607,11,1,450**, P13=0, with P01–P05=0. Three separately accepted teacher prefixes total1,344 decisions /10,752 ticks (each448 /3,584); they have zero raw/projected residual and no policy credit. Physical core totals3,392 decisions /27,136 ticks. Current spending is full_episode16,896 /phase_suffix27,648 /smoke0; origin10,112 remains. All FR/FL Q/C/P before3,584 are teacher history, not PPO completion from P01.

| Actual episode | Policy globals / decisions | Actual endpoint and first unfinished task |
|---|---|---|
|0 |52,609–53,374 /766 |P09 incomplete,80.933333333s/tick9,712; RR carry/cross/placement absent |
|1 |53,375–54,269 /895 |P12 incomplete,89.533333333s/tick10,744; RL crossing/placement absent after interrupted qualification |
|2 tail |54,270–54,656 /387 |P09 unfinished,55.666666667s/tick6,680;10 P09 samples, no terminal |

Episode0 RR qualifies three times at6,999/7,913/8,460 and ground-before-cross revokes it at7,065/8,008/9,683; no RR crossing/placement occurs. It ends with RR/RL GROUND, while FL/FR are genuinely TOP. Episode1 earns current-policy RR Q/C/P6,112/6,536/7,048, then RL Q7,514→ground revoke7,713 without RL crossing/placement. At its terminal both rear wheels are GROUND, FR TOP and FL AIR/load0; RR historical placement is not sustained current support. Both completed outcomes are actual `INCOMPLETE_CONTROLLER_BLOCKED` with valid physical evaluator/no physical-failure reason and task/full success=false.

The third tail's RR qualifies at6,678, just two physics ticks before the saved observation; RR is AIR/clearance+2.853037mm/front−234.929320mm, with no crossing/placement. FR is TOP/load0.46764057, RL GROUND/load0.53235943 and FL AIR/load0/clearance+119.173070mm. This new qualification is a real current-policy subevent, not a completed success or third terminal failure. `nominal_geometry_p06_block_54656.md` holds exact per-episode phases and terminal details.

Actual geometry use is separately bounded: `nominal_geometry_live_effect.md` has81 selected decision-end geometry audits (42 identity,24 projected,15 explicit degraded bypass) and separates actual-PPO versus nominal-geometry float32 effects. `nominal_geometry_live_transition.md` covers the first640 decisions/5,120 ticks of ordinary P06→P09 continuity. Neither subset is expanded into an unperformed whole-run geometry or causal audit. Degraded bypass is no clearance guarantee; B nominal changed too, so these non-success trajectories cannot be labeled PPO-only improvement.

Fourteen finalized blocks now contribute **44,544 new decisions /348 PPO updates /6,960 optimizer steps** since origin10,112/44/880. There is still no suffix/full-task success or new successful/paired video. Main task has started a fresh saved54,656 natural-P01 evaluation, but no outcome is prefilled here; latest completed full-P01 result remains **C50,560 P09 incomplete,1,166 decisions /9,328 ticks /77.733333333s**. Historical A failures remain unchanged and are not new optimizer gates.

## C54,656 finalized: P05 FL placement incomplete; no rear geometry projection executed

The preceding pending-evaluation statement is superseded by actual run `runs/ppo_semantic_v3/validation/20260906T1331439070323Z_g4d268fc547b7_95bda06bb52647c698f017a562c4a662`. Root confirms exit0 and the finalized execution is `SUCCEEDED`. Same4d268fc/seed2001 natural-P01 deterministic saved54,656 evaluation performs0 optimizer updates and ends after **640 decisions /5,120 ticks /42.666666667s**, P05 `INCOMPLETE_CONTROLLER_BLOCKED`. Physical valid=true, failure reason empty/termination=null; `window_ended_before_task_terminal=false` proves an actual deadline rather than a capture truncation. Training totals remain54,656/392/7,840.

The exact phase ledger is P01=1, P02=184, P03=4, P04=1, P05=450; P06–P13=0. P05 starts at1,520/12.666666667s and lasts30s. FL Q/C=1,597/2,625 without placement; FR Q/C/P=49/1,490/1,505. Terminal FL front+26.042524mm and clearance+1.846840mm satisfy the current top geometry region but FL is AIR, obstacle_pair=false, load0 and consecutive_top_samples0. The first incomplete task is loaded FL placement after crossing, not a rear workspace or historical joint-entry gate. FR remains TOP/load0.42973977, while RL/RR remain GROUND and have no Q/C/P. `eval_54656_diagnosis.md` preserves the exact source/phase/contact evidence.

All5,120 native ticks report verified/effect with zero state-write issues. There are0 geometry-bearing decision-end audits and no geometry-adjustment/evidence fields anywhere in the native audit stream. The helper is restricted to P09/P12, neither reached; consequently no direct executed geometry projection can explain this P05 terminal. This does not prove the changed training MDP had no indirect influence on learned weights, and no matched causal ablation is claimed. C50,560's earlier P09 incomplete remains historical, not an improvement benchmark silently reinterpreted after this result.

Main task is now preparing a separate current-capture reward-only revision. No revised parameters, commit/test result, migration execution or new training samples are claimed here. Fourteen finalized-block counts remain unchanged; there is no suffix/full success or new successful/same-condition paired video, and no additional A success gate.

## Capture-retention revision committed; actual new-MDP initialization from54,656

The preceding preparing-only snapshot is superseded by commit `68631e932c7deb08a7a3f2a2787b79fa7eb569ef`: two production files (semantic supervisor and v3 stage task spec) plus a new55-case test file. The actual final combined JUnit `C:/robotics_sim/wlr_robot/semantic_capture_retention_all_cpu.xml` records **896 tests /0 failures /0 errors /0 skipped /85.719s**. CPU tests are not extra training or physical success. The opt-in `current_platform_region_after_placement` reuses the existing placed-leg capture share: historical placed progress is0.8+0.2×current-region retention rather than permanently1. Retention decreases continuously outside the same measured platform XY rectangle or below its existing top-gap envelope. AIR above the platform is still fully eligible without contact/load/fixed pose/height ceiling. Historical Q/C/P, predecessor ordering, unplaced formulas and all hard success/failure/stop conditions remain unchanged, as do nominal/geometry, caps, mapper, environment and frozen A. It is a reward/potential change, not a new physical gate or nominal fix.

Real run `runs/ppo_semantic_v3/train/20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189` is started RUNNING with v3 N1/seed1001, natural P01/full_episode, teacher offset0,8,192 requested decisions, checkpoint cadence4 and explicit new-MDP warm start from54,656. This section fixes only the initial checkpoint snapshot, not later collected/optimized samples. The initial curriculum has `prefix_request=null`; old physical state and rollout are not inherited. Source P06→new natural-P01 sampling is explicit, not teacher history credited as current PPO success.

Actual immutable initial file `checkpoint_initial_v3_from_000054656_sf24b35714c5b_g68631e932c7d_4514185096bcd8d85cee235533c911e3d7d2f1609d3f0a21efa5ccebf58a771a.pt` has measured SHA-256 **`4726e5a3ec9ce2028c2bb3971de5f3fefb469f35a1e01ef02b1ce5f06ee05c04`**, matching the sidecar with roundtriptrue; actual sidecar SHA is `87feb0065f2fc55f3b2de7417941b959befd3c54960163e0fef96caae8526a67`. Source54,656 remains bound by f24b3571… and sidecar609e2686…. Whole actor digest `cca90f60744f0e00068068c9160cd06648a129a881df9cb010380a17dc716f5d`, critic `1ed87c38bd86953a2e5d1b91fc96ef0581650d1f5bcf08d8a9f00e34455dce6d` and identity-normalizer `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552` all equal the source. The official heteroscedastic mean/log-std actor remains the same architecture; complete stored RNG JSON also compares equal.

Adam changes from source `04bbd35fe486bb9d449143d202e6c27e13c571a4ac6ce55b2ca29fc576dee0c7` at1e-5 to fresh initial `ebddf13dce59639efaf4e3951e4a4db11f59a2b1e7a9f70fe1168320ba716614` at3e-5. The existing loader verifies empty fresh storage/no pending transition and verified source state before resetting all Adam moments and restoring source RNG. Initialization preserves **54,656/392/7,840**, full_episode16,896 /phase_suffix27,648 /smoke0 and original origin10,112. This is not exact-MDP optimizer continuation and does not reset the lifetime/spent-budget ledger.

The initial P01/tick0 same-state logical action comparison records0 optimizer updates, no environment/history mutation and zero differences across all12 channels of the old/new projection outputs under identical physical execution-profile hashes. It uses the same raw mean and one same new observation, not a separately reconstructed old-MDP observation or native trajectory. Potential is an observation feature, so unchanged schema/weights do not prove old/new-MDP behavior bitwise equal. Detailed bindings and limitations are in `capture_retention_implementation.md`. Fourteen completed blocks and latest full C54,656 P05 incomplete remain unchanged; no new successful/paired video or future training/evaluation outcome is claimed.

## Fifteenth block finalized:8,192 natural-P01 decisions; capture feedback active, no task success

This final section supersedes, without erasing or counting twice, the preceding68631e9 initialization-only snapshot. Actual run `runs/ppo_semantic_v3/train/20260906T1348490960247Z_g68631e932c7d_e52b74960e7d4242a8da0dfe545bf189` has finalized `SUCCEEDED` execution and root-confirmed process exit0. Its `training_manifest.json` records planned=requested=actual8,192, unconsumed0, rounding_overrun0, wall-time3,060.860190999927s,64 new PPO updates and1,280 new optimizer steps. Source54,656/392/7,840 advances to immutable **62,848/456/9,120**. The final sidecar records roundtriptrue and origin10,112; current spending is full_episode25,088 /phase_suffix27,648 /smoke0. The top summary records root-verified actual checkpoint/sidecar hashes; no duplicate hash or full-audit scan was performed for this master update.

Manifest phase counts are exactly `11,1770,36,10,1479,3856,8,5,564,1,2,450,0` for P01–P13, summing to8,192. Actual policy/core physics totals **65,525 ticks**, not blindly8,192×8; terminal partial intervals remain included at their actual lengths. Core telemetry records10 episodes,9 completed,success_count0,6 `INCOMPLETE_CONTROLLER_BLOCKED` and3 `BODY_COLLISION`. This is natural-P01/full_episode sampling without teacher-prefix credit or suffix initialization. The9 small completed-episode records and the final audit row give the exact account below; the431-decision tail is optimized data, not another completed outcome.

| Natural-P01 episode | Credited global range | Decisions | Actual endpoint |
|---|---|---:|---|
| 0 |54,657–55,595 |939 |P06 incomplete /62.600000s |
| 1 |55,596–56,272 |677 |P09 BODY_COLLISION /45.075000s |
| 2 |56,273–57,203 |931 |P06 incomplete /62.066667s |
| 3 |57,204–58,126 |923 |P06 incomplete /61.533333s |
| 4 |58,127–59,061 |935 |P06 incomplete /62.333333s |
| 5 |59,062–60,080 |1,019 |P09 incomplete /67.933333s |
| 6 |60,081–60,671 |591 |P09 BODY_COLLISION /39.400000s |
| 7 |60,672–61,799 |1,128 |P12 incomplete /75.200000s |
| 8 |61,800–62,417 |618 |P09 BODY_COLLISION /41.166667s |
| 9 / unfinished tail |62,418–62,848 |431 |Nonterminal P06 /28.733333s /tick3,448 |

The first9 episodes total7,761 decisions;7,761+431=8,192. The final audit row has global62,848,decision_count431,physics_tick3,448,phase/end_phase P06,terminal=false,termination_reason=null and task_success=false. It is neither an appended success nor an additional completed failure. Recorded BODY_COLLISION outcomes remain hard-failure labels and are not changed by this report; `capture_run_episode1_collision.md` explicitly distinguishes the available sensor/task evidence from unavailable per-body contact-point/raw-force detail. No independent proof of each collision's contact geometry is fabricated.

### What the real capture-retention branch demonstrates—and does not

`capture_retention_live_active_branch.md` fixes episode7 globals61,340–61,440,101 optimized decisions /808 ticks. RR really qualifies/crosses/places at episode ticks5,346/5,396/5,397; the decision-end t5,400 sample is actual TOP with five consecutive TOP samples,front+2.514723mm,clearance−2.389780mm and load0.145403. This is a current-policy natural-P01 history, not teacher credit. Later that bounded window records89 already-placed rows with retention R<1; independently recomputed new potential/shaping matches the logged values, while old frozen-placed progress would remain1. Retention falls to0.966744694 and briefly recovers to1 at t5,472 through current permitted region geometry, then falls toward0.188 with RR back on GROUND. The recovery does **not** create a new TOP/cross/placed event or prove pure top-face contact; the report preserves the limits of current body-pair and geometry evidence.

The formula therefore has a real active signal path after placement, including a genuine short geometry-retention recovery, rather than being permanently dormant. It does not show that the trained policy learned stable capture, that RR retained support, that RL completed, or that stability improved against a matched baseline. Episode7 ultimately terminates P12 incomplete, and P13 is never sampled in the whole block. The new reward changes the MDP/potential and its observation feature; this implementation effect is not itself PPO improvement.

`capture_run_episode5_rr_carry.md` separately preserves another real attempt: RR Q at5,642,7,774 and8,125, with the first two revoked by GROUND at7,565/7,917 and no crossing/placement. The closest15Hz recorded front is−31.0573mm at t5,864 with+8.1393mm clearance; highest clearance+30.6526mm occurs at a different state,t5,800/front−55.0412mm. FL is genuinely TOP at both extrema, but late body retreat and intermittent FL AIR are also measured. Geometry receipt binding passes in its available contexts; its same-state two-joint first-order nominal/controller/geometry/residual target decomposition is explicitly not a causal physical rollout. Neither this incomplete episode nor the three collision episodes is silently recast as a success or as proof that a particular next fix must be adopted.

Finalized15-block additions are now **52,736 decisions /412 PPO updates /8,240 optimizer steps** since origin10,112/44/880. The freshly reloaded saved62,848 natural-P01/seed2001 fixed-mean evaluation, session59959, is currently RUNNING at `runs/ppo_semantic_v3/validation/20260906T1440514266953Z_g68631e932c7d_945a0a5ecd3243c9a82d56baf14c968b`; its started manifest records RUNNING. No future evaluation result is prefilled. **Latest completed full-P01 evaluation remains C54,656 P05 incomplete**. All original A/B/C results, prior bounded snapshots and failed videos remain preserved; no suffix/full success, paired stability improvement, or new successful/same-condition paired video is credited. Any P06 nominal-tail proposal remains separate, unadopted design at this reporting boundary and is not implied by finishing this training block.

## C62,848 finalized: real RR traversal, P11 RL preparation incomplete

The preceding running-evaluation snapshot is now superseded by actual finalized run `runs/ppo_semantic_v3/validation/20260906T1440514266953Z_g68631e932c7d_945a0a5ecd3243c9a82d56baf14c968b`, session59959/exit0. Execution `SUCCEEDED` does not override task/controller success=false. The immutable62,848 checkpoint is bound by SHA `e5c91c4f443c4fcc05671eb4d5ac517d33b6b65449c08b2be9d55bd9cef0a68b`; evaluation is fixed deterministic mean, natural P01,seed2001,0 optimizer updates. It runs **993 decisions /7,944 physics ticks /66.2s**, ending P11 `INCOMPLETE_CONTROLLER_BLOCKED`,stage_age20s and `window_ended_before_task_terminal=false`. Shared physical evaluation valid=true,success=false,termination_reason=null: the task is genuinely incomplete at its current measured-goal deadline, not a collision, sensor/interface error, video truncation or restored historical exact-entry veto.

Actual phase counts P01–P13 are `1,174,3,1,150,246,1,1,114,2,300,0,0`, summing to993. The first unmet P11 requirements are measured workspace_RL=**0.289348919** and load_ready_RL=**0.637392508**. P12/RL lift-and-placement and P13 final stop are not reached. RR's actual natural-P01 Q/C/P ticks are5,276/5,521/5,528; these are neither teacher events nor evidence of sustained current capture. RR later retreats to GROUND and then AIR. Its terminal current AIR has clearance−45.452739mm/front−255.709675mm/load0, not TOP support, despite retained completed-event history. RL has no hard Q/C/P and is currently GROUND,front−397.662770mm/clearance−50.443178mm/load0.490085993. Terminal FR and FL are actually TOP with load0.501039555 and0.008874451 respectively; current support is not inferred from their historical placed flags. The separately owned `eval_62848_diagnosis.md` records the detailed trace; this master update reads only the final manifest and endpoint, not another large raw audit.

This replaces C54,656 as latest **completed** full-P01 evaluation, without deleting that P05 incomplete result or any earlier A/B/C evidence. RR traversing in this fresh evaluation is a real observed event, but later-stage progression and retained history do not establish full success or paired stability superiority. Evaluation adds no optimizer decisions or steps: completed totals remain **62,848/456/9,120**,15 completed blocks and origin-relative52,736/412/8,240. No successful or same-condition paired video is added.

### Same-MDP P10 continuation launched; planned credit only

Started run `runs/ppo_semantic_v3/train/20260906T1453012844004Z_g68631e932c7d_4793a1e4c0664f928b384744693c15fc`, session69590, records lifecycle RUNNING and command=train under unchanged `68631e932c7deb08a7a3f2a2787b79fa7eb569ef`. Its actual arguments bind saved62,848,heteroscedastic_log_v1,v3,N1,seed1001,from_phase=P10,teacher_offset_decisions=0,stage=phase_suffix,decisions=4,096 and checkpoint_interval_updates=4. `new_mdp_warm_start=false`, `policy_distribution_migration=false`, resume migration=null and warm-start record=null. This is same-MDP exact model/Adam/RNG continuation at a verified update boundary, not a new reward/architecture migration or fresh-Adam reset.

Only launch and the4,096 plan are recorded here; no collected samples, new optimized boundary,16th completed block,66,944 endpoint or488 update count is credited in advance. Teacher prefix actions/events remain outside PPO storage/phase credit and preceding teacher leg traversal cannot be attributed to the suffix policy. Any future successful suffix would still not establish natural-P01 full-task success. Reporting stops at this fixed launch boundary.

## Sixteenth block finalized:4,096 P10 suffix decisions; no suffix/full success

The preceding launch-only snapshot is superseded, without deleting it or counting it twice, by finalized run `runs/ppo_semantic_v3/train/20260906T1453012844004Z_g68631e932c7d_4793a1e4c0664f928b384744693c15fc`, session69590. Its actual `training_manifest.json` is `SUCCEEDED` execution with planned=requested=actual4,096,unconsumed0,rounding_overrun0,wall-time2,805.885992700001s. Under unchanged `68631e932c7deb08a7a3f2a2787b79fa7eb569ef`, same-MDP exact saved62,848 resume with Adam retained performs **32 new PPO updates /640 optimizer steps**, ending **66,944/488/9,760**. There is no reward/nominal/config migration in this block. Origin10,112 and prior budgets remain: final full_episode25,088 /phase_suffix31,744 /smoke0.

Immutable `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000066944.pt` is bound by checkpoint SHA-256 `8d5896c5fb02b55ab09b9fcaf387c7a43c112795928c7a721ad31126cd0002bb` and sidecar `1f6b8a0be46e7cbf097807a263a4c854921eeb393e908f22f833ba891ca9d99a`; actual pointer fields agree,root has measured both hashes and the sidecar records `save_load_round_trip=true`. The independent single-block audit reports32 continuous actor-hash/counter links,changed actor parameters and finite nonzero gradients for every update. This master does not repeat its large-audit/hash scan or reinterpret optimizer execution as physical success.

PPO-request counts are P01–P09=0,P10=5,P11=5,P12=706,P13=3,380,total4,096. Actual credited physics is **32,761 ticks**, all verified by the single-block audit, not rounded to32,768; episode2/global65,232 ends after1tick instead of8, explaining the7-tick difference. There are5 accepted real teacher prefixes,each948 behavior decisions/7,584ticks,total4,740/37,920,with no prefix miss. These are excluded from PPO storage/phase credit. The physical core including preparation totals8,836 decisions/70,681ticks, exactly4,740+4,096 and37,920+32,761, not one continuous episode duration. Teacher-supplied FR/FL/RR Q/C/P is not attributed to current-policy training.

| Suffix episode | Credited global range | Policy decisions | Actual outcome including prefix time |
|---|---|---:|---|
| 0 |62,849–63,815 |967 |P13 incomplete /127.666667s |
| 1 |63,816–64,779 |964 |P13 incomplete /127.466667s |
| 2 |64,780–65,232 |453 |P12 incomplete /93.341667s;final interval1tick |
| 3 |65,233–66,198 |966 |P13 incomplete /127.600000s |
| 4 / unfinished tail |66,199–66,944 |746 |Nonterminal P13 /112.933333s /episode tick13,552 |

The4 completed episodes total3,350 decisions;3,350+746=4,096. All4 terminate `INCOMPLETE_CONTROLLER_BLOCKED`; telemetry has fresh-P01 success_count0 and teacher_initialized_task_success_count0. The final row is global66,944,decision_count746,phase/end_phase P13,terminal=false,termination_reason=null; the tail is already optimized but not a fifth completed failure or success. Recorded policy RL Q/C/P occurs in episode3 at7,950/8,066/8,105 and the tail at7,960/8,072/8,111, after current-policy takeover; those are legitimate suffix events, not final controlled-stop success. At the tail boundary RL is currently AIR/load0 while RR is TOP/load about0.53717, so historical placed flags do not establish present all-leg support or task completion.

Sixteen finalized blocks contribute **56,832 new decisions /444 PPO updates /8,880 optimizer steps** since verified origin10,112/44/880. This block is no longer RUNNING. Latest completed natural-P01 deterministic evaluation remains **C62,848 P11 incomplete,993 decisions/7,944ticks/66.2s**, not overwritten by suffix behavior. Newly reloaded saved66,944 natural-P01/N1/seed2001 fixed-mean evaluation is RUNNING,session70942,at `runs/ppo_semantic_v3/validation/20260906T1546561630471Z_g68631e932c7d_db836ec127c44b7e983ff9a307ef594a`; its actual started arguments bind66,944,from_phase P01,semantic_residual_eval,unchanged68631e9 and no migration. No result,success,improvement or video is prefilled. All older outcomes and bounded reports remain unchanged.

## C66,944 finalized: verified persistent BODY contact in early P09

The preceding pending-evaluation snapshot is superseded by finalized run `runs/ppo_semantic_v3/validation/20260906T1546561630471Z_g68631e932c7d_db836ec127c44b7e983ff9a307ef594a`, session70942/exit0. Same68631e9,N1,seed2001,natural P01,deterministic fixed mean,immutable66,944 checkpoint SHA `8d5896c5fb02b55ab09b9fcaf387c7a43c112795928c7a721ad31126cd0002bb`,optimizer updates0. Execution `SUCCEEDED` is not task success: **585 decisions/4,674ticks/38.95s/P09 BODY_COLLISION**, task/controller successfalse and valid shared physical evaluator termination `TASK_FAILURE_BODY_COLLISION`. The last policy interval is2ticks, exactly584×8+2; it ends immediately on the real terminal with no value bootstrap, not an artificially completed8tick interval or P09 deadline.

P01–P13 policy counts are **[1,175,3,1,149,243,1,1,11,0,0,0,0]**, totaling585. Actual P09 lasts82physics ticks/0.683333s after entry4,592. FR Q/C/P=48/1,418/1,432 and FL=1,517/2,540/2,632. RR/RL never qualify/cross/place, and RR's final lift-attempt ledger has no initial-clearance event. RR is still GROUND at4,672 and only AIR at4,673–4,674 alongside the collision, without above-top geometry or adequate qualified lift. The later P10/P13 suffix training cannot substitute for this full-P01 outcome, and a P13 stop proposal does not explain a failure before P13 was visited.

The independently reviewed `eval_66944_diagnosis.md` verifies exact `contacts.base_link.obstacle` pair,`sensor_body=base_link`,`other_body=/World/Obstacle`,pair_verified=true. Tick4,673 is the first active sample,world-z force **31.2068920N**,persistent count1,detected=false; tick4,674 has **12.1347952N**,count2,detected=true. Existing reason is `exact base_link/obstacle pair persisted`. Tick4,672 has zero force and active=false even though a point is present, so point presence alone was not mistaken for collision. `geometry_penetration_m=0` means that AABB metric supplies no extra penetration evidence; it does not negate the independent verified contact-force persistence. Conversely these two samples do not prove deep mesh penetration, long subsequent contact or any unexecuted post-terminal motion.

FL's historical placement is not sustained current load: it becomes AIR at2,691, briefly contacts obstacle at3,355–3,364, then remains AIR for1,310ticks from3,365 through4,674. It is unladen at the P06/P07/P08 handoffs and collision endpoint. This is genuine measured coordination context, not proof FL alone caused collision or justification for a new fixed-load/old-entry gate. All **4,674/4,674 native audit ticks verify**; in-episode root-pose,root-velocity,force/impulse andgravity writes total0. Full raw finite rows include4,675 observations with reset tick0. The endpoint is a real task failure under valid physical sensing, not a software-interface failure or a mislabeled P13 stop result.

Latest completed full-P01 evaluation is now **C66,944 BODY_COLLISION**. C62,848's earlier P11 incomplete and all old A/B/C reports/manifests remain preserved. Evaluations add no training credit: completed totals stay **66,944/488/9,760**,16 blocks,origin-relative56,832/444/8,880 and spent full25,088/suffix31,744. `p10_capture_block_66944.md` is the independent final training ledger, including32 continuous updates and12-dimensional Gaussian LP audit; it does not turn its suffix events into full-P01 success. No paired improvement or new successful/same-condition paired video is claimed.

### Selected continuous-stop revision: CPU verification only, not yet trained

Main task has selected and locally implemented a new opt-in P13 progress branch in the supervisor and v3 task spec: `final.stop_progress_semantics=per_wheel_four_type_threshold_ratio_v1`. The old reciprocal `stop_command_progress` mode remains a supported compatibility path/config entry; the selected new branch replaces the control-progress contribution rather than adding a duplicate one. At this reporting boundary production changes are **under CPU verification, uncommitted and not physically executed**. No final combined-test result, new runtime HEAD, migration checkpoint, training run or optimized count is prefilled. It is preparation for an explicit new-MDP boundary, not a claim that current66,944 already learned the revised progress. This report does not alter any hard physical success/failure/stable-time condition or the saved C66,944 outcome.

### Continuous-stop revision committed and real P06 launch: fixed initial boundary

The preceding CPU-only snapshot is superseded by commit **68cd9f5fca6c9ba06c38666fb55024b9ba04f9b0**. Four files were committed: two production files, `src/wlr50_clean/ppo/semantic_supervisor.py` and `configs/ppo_semantic_v3/stage_task_spec.yaml`, plus two focused test files. The opt-in `final.stop_progress_semantics=per_wheel_four_type_threshold_ratio_v1` replaces the previous control-progress contribution; the old reciprocal compatibility mode remains supported and is not added again. Hard physical task/safety/controlled-stop conditions, stable-time requirement, nominal advice and frozen A are unchanged. This is a selected reward/MDP revision, not evidence that an earlier checkpoint learned a better stop or that the earlier P09 collision was a P13 issue.

Root's completed full `tests/unit` run reports **2,318 passed,9 skipped,0 failures,0 errors**, elapsed427.59s. Direct light reading of `C:/robotics_sim/wlr_robot/semantic_continuous_stop_all_cpu.xml` confirms2,327 total test cases,9 skips and0 failures/errors, with JUnit suite-time427.553s. The9 skips are Windows symbolic-link privilege limitations. The85 newly added continuous-stop tests are already included in those counts; focused legacy and new test totals are not added to inflate the combined result. These are CPU regression results, not new physical success or video evidence.

Actual started manifest `runs/ppo_semantic_v3/train/20260906T1611147430138Z_g68cd9f5fca6c_846e7b65ab224b86a53905c55fa11f85/run_manifest.started.json` records lifecycle `RUNNING`, command `train`, expected/source runtime68cd9f5,semantic v3,heteroscedastic_log_v1,N1,seed1001,from_phase P06,teacher_offset_decisions0,stage phase_suffix,decisions4,096,checkpoint_interval_updates4 and `new_mdp_warm_start=true`. Source is immutable `checkpoint_step_000066944.pt`, SHA `8d5896c5fb02b55ab09b9fcaf387c7a43c112795928c7a721ad31126cd0002bb`. It is not same-MDP exact Adam resume or another policy-architecture migration. The migration record lists exactly the two production changes above; action/execution/observation/reward-config/quality files remain hash-identical. Its source and target ledgers both retain full_episode25,088/phase_suffix31,744/smoke0 and original origin10,112. Real reset-only teacher preparation remains excluded from PPO credit, and its P06 suffix cannot substitute for natural-P01 evaluation.

The actual saved initial sidecar is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_initial_v3_from_000066944_s8d5896c5fb02_g68cd9f5fca6c_b7d80703ee87e23902156fa0b2a5a9016ba1b66d21fc9d04444508497079f494_manifest.json`. It records initial checkpoint SHA **f21138c5e64052e2e59005b7ac353f5c74352a44ccaddc29046556400297f2f7** and `save_load_round_trip=true`; this report reads the saved evidence and does not duplicate root's file hashing. Direct source/initial sidecar comparisons find actor parameter digest `ebef32f5df36bbf666f622ff817d2c01b33b77e52798a3f05f7ba2ca766aee1e`, critic digest `95cd581b97b790dde8e3fda797b463db9e99599fd29cea18cb766465e9ceb0fd`, identity-normalizer digest `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`, the complete serialized `training_rng_state`, lifetime66,944/488/9,760 and spent-budget ledger all exactly equal. The preserved actor contains its learned state-dependent log-std; there is no reinitialization to scalar Gaussian. Actual optimizer state digest changes from source `ac0a9ed0d9bc8d382d4bf53936d3f55191cf972479a79bc6b40798ca76f954ca` to fresh `ebddf13dce59639efaf4e3951e4a4db11f59a2b1e7a9f70fe1168320ba716614`, and saved effective LR changes1e-5 to3e-5. The explicit loader discards old rollout/physical state, restores the verified source RNG after construction/load and starts fresh storage; inherited source `last_update` metadata is history, not a new update at initialization.

The newly written `new_mdp_initial_action_comparison.json` binds P06/tick3,584/29.866667s,324-dimensional current observation,optimizer_updates0 and `actual_environment_or_bridge_history_modified=false`. All12-channel new-minus-old values are0 for bounded/scaled/masked/rate-projected/safe-projected/applied logical actions, with unchanged execution-profile hash. Its declared scope is **same-state logical output/projected target, not native dispatch or subsequent trajectory**: it is not a paired rollout, a claim that changed reward/potential produces identical old-MDP behavior, or evidence of task success.

This is a fixed **initial migration and RUNNING launch** account, not a rolling optimizer audit. The sixteen completed blocks and their current summary remain **66,944/488/9,760**, origin-relative56,832/444/8,880; the planned4,096 is not credited in advance and no seventeenth completed block is entered. Latest completed reloaded full-P01 evaluation remains **C66,944:585 decisions/4,674ticks/38.95s/P09 BODY_COLLISION**. No new suffix/full success, stability-improved checkpoint or successful paired video is claimed.

## Seventeenth block finalized at68,224: requested complete-update diagnostic stop

The preceding initial/RUNNING snapshot is preserved and superseded by finalized run `runs/ppo_semantic_v3/train/20260906T1611147430138Z_g68cd9f5fca6c_846e7b65ab224b86a53905c55fa11f85`, session31153, runtime **68cd9f5fca6c9ba06c38666fb55024b9ba04f9b0**. Both final run/result lifecycle fields are `STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`. Root used the existing `stop_after_update.request.json` mechanism to save after a complete PPO update; there was no mid-rollout termination, active-runtime change or success-conditioned discarding of experience. This is a finalized seventeenth block, **not completion of the originally planned4,096 decisions and not task success**.

Direct final-manifest fields are actual_policy_decisions1,280,final requested_policy_decisions1,280,**planned_requested_policy_decisions4,096**,unconsumed_requested_policy_decisions2,816,rounding_overrun0,ppo_updates_this_run10,optimizer_steps_this_run200 and wall_time_s830.2300018998794. The finalized requested field records the consumed shortened block; the independent planned field and original started manifest preserve the4,096 plan. Immutable `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000068224.pt` records **68,224/498/9,960** and `save_load_round_trip=true`. Root's actual measured checkpoint/sidecar SHA values are respectively **0bc86a834f7baeb6fb35a5c27d9d31dab6738f7656f528dc8cf101866af26897** and **0e2afd936ab377a9b13cc356ac658bccc334c31bd519d3da7c5e7d4422816b7e**, matching pointer/sidecar; this report reads the finalized binding without rehashing the files. Spending is full_episode25,088/phase_suffix33,024/smoke0,origin10,112. `25,088+33,024+10,112=68,224`; unused2,816 is not assigned to an unexecuted future experiment.

Independent finalized `continuous_stop_p06_diagnostic_block_68224.md` supplies the full stream ledger; its earlier `continuous_stop_p06_initial_67072.md` remains an unchanged bounded initial/first-update record. All1,280 actual policy requests,global66,945–68,224,are P06,with no missing or repeated global IDs. Every other phase has0 new samples, including P13. Actual episode accounting is:

| Episode | Actual globals | Credited decisions / physics ticks | Last episode tick / seconds | Actual outcome |
|---|---|---:|---|---|
| 0 |66,945–67,545 |601 /4,801 |8,385 /69.875 |P06 INCOMPLETE_CONTROLLER_BLOCKED |
| 1 |67,546–68,146 |601 /4,801 |8,385 /69.875 |P06 INCOMPLETE_CONTROLLER_BLOCKED |
| 2 / unfinished tail |68,147–68,224 |78 /624 |4,208 /35.066667 |P06 nonterminal,termination_reason=null |

Completed-episode samples1,202 plus78 already optimized tail samples equal1,280. The two terminal decisions,global67,545/68,146,each execute1tick; thus policy ticks are **1,280×8−14=10,226**, not missing telemetry. The two completed episodes have P06 stage_age40.008333s and physical failure=null,not collision or200s total timeout. Tail stage_age5.2s/termination=null remains unfinished,not a third failure or success. Its partial return is not represented as a completed-episode return.

Three accepted real reset-only teacher prefixes each contribute448decisions/3,584ticks,total **1,344/10,752**,with no fallback. Teacher raw/projected residuals remain zero,policy_credit=false,and prefix_teacher_data_in_ppo_storage=false on all policy rows. Complete core2,624decisions/20,978ticks separates teacher1,344/10,752 from PPO1,280/10,226. All10,226 policy native ticks and10,752 teacher native ticks verify; root-pose/root-velocity/force-or-impulse/gravity write counters are0. Teacher FR/FL Q/C/P history is preserved as preparation,not newly learned PPO credit.

RR/RL have **no qualified/crossed/placed events in any of these three policy segments**. The first unfinished physical stage is P06 workspace preparation. At the two terminal endpoints RR is AIR but below top (front−445.091/−544.209mm,clear−43.718/−48.141mm,load0); RL is GROUND (front−557.033/−639.969mm,load0.536903/0.518840). Tail RR remains AIR below top(front−362.026mm,clear−48.450mm) and RL GROUND(front−479.900mm,load0.513904). AIR alone is not hard qualification. P13 was never visited,so this block **has not physically exercised the newly selected continuous-stop progress branch or demonstrated stopping success**. Future nominal-tail investigation is a separate versioned experimental plan,not a proven causal explanation or an implemented change credited here.

Optimizer evidence contains exactly updates489–498/global67,072–68,224,each128 policy decisions and20 optimizer steps. All10 actor before/after links are continuous,changed and finite-nonzero-gradient verified; the final actor-after matches checkpoint68,224. Initial fresh Adam LR3e-5 is preserved in the earlier migration evidence; all10 logged update-end LR values are1e-5. No per-minibatch LR log exists,so this is not a claim of constant LR throughout all optimization. Migration itself contributes0 new optimizer credit.

Seventeen finalized blocks therefore contribute **58,112 decisions/454 PPO updates/9,080 optimizer steps** since verified origin10,112/44/880. Latest completed natural-P01 evaluation remains **C66,944/P09 BODY_COLLISION**,585decisions/4,674ticks/38.95s. Newly reloaded68,224 natural-P01/N1/seed2001 fixed-mean evaluation,session18632,has actual `RUNNING` started manifest at `runs/ppo_semantic_v3/validation/20260906T1628525452104Z_g68cd9f5fca6c_fac838181e114f68954319480dae7b77`; command `eval`,mode semantic_residual_eval,checkpoint68,224,no new-MDP migration. Its outcome is not yet credited here. No full/suffix success,new successful video or paired stability improvement is asserted; every earlier failure/incomplete/bounded snapshot is retained.

## C68,224 finalized: real FL lift/crossing without physical capture

The preceding pending-evaluation snapshot is superseded by finalized `runs/ppo_semantic_v3/validation/20260906T1628525452104Z_g68cd9f5fca6c_fac838181e114f68954319480dae7b77`,session18632/exit0. Actual runtime68cd9f5,N1,seed2001,naturalP01,deterministic saved heteroscedastic fixed mean,checkpoint68,224 SHA `0bc86a834f7baeb6fb35a5c27d9d31dab6738f7656f528dc8cf101866af26897`,optimizer_updates_during_evaluation0. Direct final-manifest fields confirm **632 policy decisions/5,056ticks/42.13333333333333s**,P05 `INCOMPLETE_CONTROLLER_BLOCKED`,task/controller successfalse,and `window_ended_before_task_terminal=false`. Run lifecycle `SUCCEEDED` means execution completed,not task success. The independent physical evaluator is valid with termination_reason=null: no recorded body collision,wheel-only or other hard physical failure. This is a true30s P05 stage deadline terminal,not an early video/codec cutoff; bootstrap remainsfalse.

Actual phase counts P01–P13 are **[1,177,3,1,450,0,0,0,0,0,0,0,0]**. FR earns Q49/C1,434/P1,448. P05 starts1,456; FL last contacts ground at1,468,then remains AIR throughout **1,469–5,056 inclusive,3,588 raw samples**. It has initial clearance1,483,qualified lift1,532 and front crossing2,561,but no placed event. Across the full P05 window,FL obstacle-active samples are0; all450 decision-end TOP-sample counters are0. Historical FR placement remains real,but it cannot substitute for present FL contact/support.

Independent finalized `eval_68224_diagnosis.md` identifies the closest measured FL bottom-to-top sample at tick2,657/22.141667s:front **+27.759853mm**,clearance **+0.402172mm**,AIR,obstacle force0 and verified pair inactive. An available contact-point field on an inactive zero-force pair is not touchdown evidence. Neither this sub-millimetre cached-collider geometry nor the missing touchdown alone proves a sensor bug. Final FL is still AIR,front+29.225651mm,clearance+2.212522mm,load0; FR is currentTOP/load0.424179 while RR/RL areGROUND/load0.076910/0.498911. Rear legs have no Q/C/P. The first unfinished physical task is **sustained actual FL placement/capture**,not a historical angle/entry condition.

`placed_FL` reaches0.85 at decision329/tick2,632 and at304 of450 P05 decision-end samples. Its existing components are0.35 qualified lift+0.35 crossing+0.30×0.5 top geometry without TOP samples=0.85. It is partial task progress,not a placed latch or85% task-success probability. The exact capture criterion is retained. Residual actuation remains live after the finite nominal advice: final logical wheel residuals are approximately[-0.00598130,-0.01066752,+0.02495894,+0.06776555]rad/s,with the corresponding recorded native float32 targets after the existing sign mapping; this is control evidence,not proof that any one residual component caused the lack of contact.

All **5,056/5,056 native ticks verify**; own-phase effect5,052 excludes the four ordinary incoming-handoff-first ticks. All5,057 raw rows including reset are finite and all four in-episode state-write categories remain0. No optimizer update or physical-state rewriting occurred during evaluation. **P06–P13 were never entered**,so C68,224 cannot be explained by the separately proposed P06 nominal-tail change or used as a physical test of that future branch. Its difference from C66,944 is an observed outcome difference across saved policies,not controlled evidence of PPO stability improvement.

Latest completed full-P01 evaluation is now **C68,224/P05 incomplete**. All prior C66,944 BODY_COLLISION,C62,848 P11 incomplete and earlier A/B/C evidence remain unchanged. Training accounting stays **17 finalized blocks,68,224/498/9,960**,origin-relative58,112/454/9,080,spent full25,088/suffix33,024. The seventeenth P06 block remains actual1,280/10/200 with original4,096 plan and2,816 unconsumed,not retroactively completed by evaluation. The selected P06 measured-workspace nominal-tail revision is presently uncommitted and untrained; this report does not prefill a new HEAD,training count,checkpoint or future outcome. No full/suffix success or new successful/paired video is claimed.

## Eighteenth block finalized:73e937 measured P06 tail, natural-P01 training to72,320

The earlier uncommitted-tail and first-update snapshots are superseded by committed **73e937039708a7306b2b2c941e1f9108b8fa8b3d** and finalized run `runs/ppo_semantic_v3/train/20260906T1650134938238Z_g73e937039708_7681a31f01714528b75509300dfac00f`. Only3 production files changed: `semantic_supervisor.py`,`configs/ppo_semantic_v3/stage_task_spec.yaml`,and `semantic_prefix.py` for read-only current inner-task diagnostic forwarding. `nominal.p06_wheel_tail_semantics=measured_workspace_retirement_after_finite_source` extends only the existing P06-owned.3rad/s source suggestion after its finite endpoint while retaining measured retirement,monotonic peak,newer owners,terminal handling and physical slew. It does not force a PPO action,reset a mapper,restore an old entry or change hard task success. It changes B's nominal as well as C's,so cross-version outcomes cannot be called PPO-only improvement. Frozen A's29-file map and action/execution/observation/reward/quality configurations remain equal in the recorded migration; old A's actual incomplete/nonpassing outcome remains unchanged and is not an optimizer gate.

CPU evidence is **998 disjoint passed cases/0 failures/errors/skips**: `C:/robotics_sim/wlr_robot/semantic_p06_tail_semantic_regression_cpu.xml` has956cases/93.668s,and `semantic_p06_wheel_tail_20260906.xml` has42cases/4.994s. A light XML classname+test-name comparison finds0 shared node IDs. The earlier126-case existing-modes and66-case legacy/prefix subsets are not added again. These tests preserve old finite-tail fixtures,324-dimensional encoding and read-only getter behavior; they are not physical success evidence.

`p06_tail_p01_initial_update.md` remains a fixed initial/first128 record,not rewritten as the final run. The new initial checkpoint from68,224 records SHA `7047bca41eabae1fe55c86cfdc1931f3b26bb4912abc39f6cb69e60f03d7feae` and roundtriptrue; source/initial full actor including learned log-std,critic,identity normalizer,complete RNG,policy/runner contract,68,224/498/9,960 and full25,088/suffix33,024 all match. This is explicit `NewMdpWarmStart`,not exact-Adam resume: Adam is fresh3e-5,old rollout and physical state are not inherited,and migration adds0 decisions. The initial `implemented_reset_sampling` string still describes source P06,but current sampling/topology,first physical tick and trained checkpoints identify actual naturalP01; the caveat is preserved,not misrepresented as teacher credit. Final72,320 sampling fields both identifyP01,prefix_request=null.

Final run and training manifests are lifecycle **SUCCEEDED execution**,requested=planned=actual4,096,unconsumed0,rounding_overrun0,no stop request,wall_time_s **1,553.9316320000216**. Source68,224/498/9,960 plus4,096/32/640 gives **72,320/530/10,600**. Immutable `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000072320.pt` has `save_load_round_trip=true`,SHA **96d26c928dcd9d07f95bf4f33cd5fade64220de7bdffbb5b581f0dcadc64c10f**; its sidecar SHA is **67503b620a98eebe12607838f3b1213fbd759992810c80b3ff676dc9e9d554f8**. Root's actual file hashes match publication pointer/sidecar; this report does not repeat hashing. Final spentfull29,184/suffix33,024/smoke0 plusorigin10,112 equals72,320. The prior short seventeenth block remains1,280 actual/2,816 unused; no historical plan is relabeled completed by this later request.

Independent finalized `p06_tail_p01_block_final.md` verifies all4,096 rows and32 update records. PhaseP01–P13 counts are **[5,894,18,5,740,2,422,1,1,10,0,0,0,0]**. Every episode starts naturalP01,teacher decisions/ticks0,and core=policy4,096. Completed episodes and unfinished tail are:

| Episode | Actual globals | Decisions / physics ticks / seconds | Recorded outcome |
|---|---|---|---|
|0 |68,225–68,872 |648 /5,177 /43.141667 |P09 BODY_COLLISION |
|1 |68,873–69,806 |934 /7,472 /62.266667 |P06 INCOMPLETE_CONTROLLER_BLOCKED |
|2 |69,807–70,737 |931 /7,448 /62.066667 |P06 INCOMPLETE_CONTROLLER_BLOCKED |
|3 |70,738–71,674 |937 /7,496 /62.466667 |P06 INCOMPLETE_CONTROLLER_BLOCKED |
|4 / unfinished tail |71,675–72,320 |646 /5,168 /43.066667 |P06 terminal=false |

The4 completed episodes contribute3,450 decisions; tail646 is already optimized but not a fifth completed failure/success. Only global68,872 executes1terminal tick,so total **32,761=4,096×8−7**. All32,761 native ticks verify and have real residual effects; own-phase effects32,733 are a distinct count excluding ordinary incoming hold ticks. All4 state-write counters remain0. The32 actor updates499–530 are changed/finite-nonzero-gradient verified,with31 adjacent hashes continuous and each20 optimizer steps. Final actor-after matches checkpoint72,320. Raw Gaussian LP re-evaluation using recorded12-dimensional actions/means/std has maximum absolute difference1.5555665e-6; std remains positive finite. Recorded value loss ranges.000620815–106.662969,not evidence of convergence. All32 update-end LR values are1e-5; initial3e-5 and per-minibatch-unobserved LR are not conflated.

All5 policy episodes genuinely earn front FR/FL Q/C/P from current natural-P01 data,not teacher history. Neither rear leg earns any hard Q/C/P in any episode. Repeated rear initial-clearance events are not qualified lift or successful attempts. Episode0 visits P09 and collides before RR qualified lift; the valid detector's BODY_COLLISION label is retained,while the compact training log cannot independently supply exact body-pair force/persistence as a full raw-evaluation trace can. The three P06 incomplete episodes fail present rear workspace preparation,with terminal RR/RL front positions approximately[-380.249,−461.729],[-437.673,−553.447],[-474.081,−578.060]mm. The final nonterminal tail still has RR AIR below top/front−444.717mm and RL GROUND/front−550.118mm,with no rear Q/C/P; its later outcome is not inferred.

`p06_tail_p01_first_episode_readonly.md` shows episode0 P06 fromtick2,560 to5,088,only2,528ticks<finite-source3,064ticks: **new endpoint tail never activates there**. FL had genuine current TOP/contact/load at placement2,560; existing retirement reaches peak1/gain0 at5,096 and does not restart on retreat. Later P07 FR-wheel ownership and residuals remain active into P09. Its rear-stage visit or collision is not attributed to an unexecuted tail branch.

`p06_tail_p01_second_episode_readonly.md` shows episode1 genuinely activates the tail: first current diagnostic global69,589/tick5,736,then tick5,744's consumed nominal and same-tick zero-residual native wheel counterfactual[-.3000000119,+.3000000119,−.3000000119,+.3000000119] verify a real post-endpoint source baseline. Peakretirement stays0,217 current active diagnostic snapshots occur,and P06 still ends incomplete; rear approach maximum is only.033084. PPO can oppose this advice,including low mean FR/RL target components; this observation is not a causal proof of failure. Across the full block there are651 `live_endpoint_tail` decision-end snapshots,not651 successes or automatically complete physical intervals. Suggestions are before later ownership/slew and are not interchangeable with the last applied command. No tail activation creates qualification,placement or guaranteed progress.

Eighteen finalized blocks now add **62,208 decisions/486 PPO updates/9,720 optimizer steps** since original10,112/44/880. No suffix/full-task success or improved/paired video has been produced. Latest completed full-P01 remains **C68,224 P05 incomplete,632decisions/5,056ticks/42.133333s**. New saved72,320 natural-P01/N1/seed2001 deterministic fixed-mean evaluation is **RUNNING** at `runs/ppo_semantic_v3/validation/20260906T1717048576808Z_g73e937039708_b1b7a8788eb545698c65882674df3153`; actual started fields bindcheckpoint72,320,semantic_residual_eval,no new-MDP migration. Its final outcome is not prefilled. No future curriculum or training credit is assigned by this report.

## C72,320 finalized: FL active lift and crossing, still no sustained capture

The preceding RUNNING snapshot is superseded by finalized `runs/ppo_semantic_v3/validation/20260906T1717048576808Z_g73e937039708_b1b7a8788eb545698c65882674df3153`,session27768/exit0. Saved checkpoint72,320 is evaluated using the fixed heteroscedastic mean,N1,seed2001,naturalP01,no teacher and0 optimizer updates. Actual **647 decisions/5,176ticks/43.13333333333333s** end at P05 stage_age30s with `INCOMPLETE_CONTROLLER_BLOCKED`. Lifecycle SUCCEEDED denotes completed execution; task/controller successfalse. Shared physical evaluation is valid with termination_reason=null: no recorded collision,wheel-only or sensing/interface failure. PhaseP01–P13 is **[1,191,4,1,450,0,0,0,0,0,0,0,0]**.

Independent finalized `eval_72320_diagnosis.md` verifies FR Q48/C1,546/P1,561 and FL Q1,652/C2,690 with **no FL placed event**. FL remains uninterrupted AIR from1,588 through5,176 inclusive,3,589 raw samples; P05 has0 obstacle-active samples/zero obstacle force,and all450 decision-end TOP counters are0. Closest recorded FL bottom to top is tick2,774,clearance **+2.871861mm**,front **+9.146396mm**,AIR and verified obstacle pair inactive/force0. The presence of an inactive contact-point field does not mean touchdown. Final FL remains AIR,clearance+5.506163mm/front+11.367742mm/load0; FR is currentTOP/load.430306,RR/RL areGROUND. Neither rear leg has hard Q/C/P.

The first unfinished physical task is **actual FL placement/capture**,not a historical joint-angle/clock-entry requirement. `placed_FL=.85` at304 decision-end samples remains.35qualification+.35crossing+.15topgeometry with no TOP samples,not a placement latch or85% success. All5,176 native ticks verify actual effects,own-phase5,172 excludes ordinary incoming hold ticks; state-write categories are0,all5,177 raw observations are finite,and body detected/exact-pair-active/persistent flags are allfalse. Control remains live after finite nominal advice,so missing contact is not a codec cutoff or absent action path.

This trajectory never entersP06 and has0 new-tail activations. Its outcome cannot be assigned causally to an unexecuted P06-tail branch,and stochastic training captures do not replace deterministic evaluation. Latest completed full-P01 is now **C72,320/P05 incomplete**; C68,224 and all older results remain unchanged. Accounting stays **18 finalized blocks,72,320/530/10,600**,origin-relative62,208/486/9,720,full29,184/suffix33,024. No full/suffix success,paired stability improvement or successful new paired video is claimed. The next capture-approach reward revision is only being implemented; no new commit,training count or outcome is credited here.

## Nineteenth block finalized: capture-approach P06/offset240 diagnostic to73,088

Committed runtime **c34262abffc16847ff32d15ecbf790dd60803e0a** follows the earlier pending implementation snapshot. The new opt-in first-capture potential splits the existing.2 capture share between current post-qualified/post-cross surface proximity and actual TOP-sequence credit,and retires its existing local unload preparation share after Q+C so legal loading need not reverse progress. It does not grant contact,placement or success from geometry. Placed-leg retention and continuous P13 stop remain unchanged. The same commit corrects two descriptive CLI initial `implemented_reset_sampling` fields,not historical checkpoints or runtime validation. Three production files are supervisor,task YAML and CLI; four test files complete the seven-file commit. CPU evidence is53 focused capture cases plus1,004 disjoint broader cases=**1,057 passed/0 failures/skips**. Earlier43 CLI and other subsets overlap and are not added again. The complete scope/formulas/tests remain in `capture_approach_implementation.md`.

Run `runs/ppo_semantic_v3/train/20260906T1738509257629Z_gc34262abffc1_04bbf6558046407990f85358d738cc17` usesN1/seed1001/P06offset240/phase_suffix from72,320. Fixed `capture_approach_p06_offset240_initial.md` directly compares source and new initial: full actor/std,critic,identity normalizer,RNG,policy/runner configuration,72,320/530/10,600,full29,184/suffix33,024 andorigin10,112 are retained. Source LR1e-5 becomes fresh Adam initial3e-5; old rollout and physical state are not inherited. Initial SHA recorded9061fcd2b42c339cc27c770f7b5b09cd9ce08a59e7f76a6a6b3f84ed5db43221/roundtriptrue. Corrected initial sampling fields now identifyP06offset240; an unrelated initial top-level `phase_suffix_curriculum_implemented=false` source-value residue is disclosed in that report,while topology/actual sampling and later normal saves correctly say suffixtrue. No immutable evidence is rewritten to hide that descriptive residue.

The run-specific normal stop request is honored at a complete update boundary. Final lifecycle is **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY**,not optimizer failure or task success. Final `requested_policy_decisions=actual_policy_decisions=768` is distinct from preserved `planned_requested_policy_decisions=2048` and `unconsumed_requested_policy_decisions=1280`; overrun0. Actual **768 decisions/6 PPO updates/120 optimizer steps**,recorded wall **831.2335137999617s**,produce **73,088/536/10,720**. Root verified actual final checkpoint/sidecar hashes against pointer: `checkpoint_step_000073088.pt` SHA **3efd9531d6bf797a2852107c1183ac8be2b96bcf5eaa3ed59358f0cd9d44ffef**,sidecar **43156d6dc98d44de7ea9d0b127ea9cf06535961318d850c72eeedd7ead25aa14**,roundtriptrue. No repeated hashing was done here.

Independent final `capture_approach_p06_offset240_block_final.md` verifies the complete policy ledger: **P06=768,all other phases0**. Three accepted actual prefixes each use688decisions/5,504ticks,total **2,064/16,512**,excluded from storage and all policy counters. PPO ticks **6,130=768×8−14**; core2,832decisions/22,642ticks includes the teacher rather than converting it to PPO credit. Each real handoff isP06 at5,504/45.866667s,offset240 from observedP06entry3,584,with154.133333s left on the original200s clock. Teacher FR Q71/C1,665/P1,695 and FL Q2,461/C3,115/P3,583 remain preparation only.

| Episode | Actual policy globals | Decisions / PPO ticks | Last episode tick / time | Outcome |
|---|---|---:|---|---|
|0 |72,321–72,681 |361 /2,881 |8,385 /69.875s |P06 age40.008333s incomplete |
|1 |72,682–73,042 |361 /2,881 |8,385 /69.875s |P06 age40.008333s incomplete |
|2 / unfinished tail |73,043–73,088 |46 /368 |5,872 /48.933333s |P06 age19.066667s,terminal=false |

The two true deadlines execute1tick at their final decisions,accounting exactly for14 fewer physics ticks. Both physical evaluators are valid with null failure reason; neither is a collision/wheel-only classification. The46 tail decisions are already optimized,but administrative stopping is not a third task failure or success. RR/RL have no hard Q/C/P anywhere in the policy segments. RR initial-clearance counts20/21/2 are not qualifications. First unfinished geometry is current RL workspace: best completed-episode fronts−222.931918/−226.973500mm against the unchanged−220mm lower bound; terminal fronts−227.768984/−229.859379mm. FL is currentlyAIR/load0 at both terminals despite teacher placement history. The nonterminal tail still has RL front−260.933575mm. No unrecorded120Hz maxima or contact forces are inferred from compact training data.

All6,130 policy native ticks and own-phase effects verify,with all four in-episode state-write categories0. Six actor updates531–536 are changed/finite-nonzero-gradient verified and hash-continuous from source1587569b…4300 tofinal5031de52…4589,each20 optimizer steps. Three immutable saves72,448/72,832/73,088 record roundtriptrue. All6 logged update-end LR values are1e-5; initial3e-5 and unseen minibatch LR are not conflated. This block visits no later stage and has **0 first-capture branch activations,0 finite-source endpoints and0 new-tail active snapshots**. Its optimization cannot be represented as learned capture or tested P13 stopping.

Nineteen finalized blocks add **62,976 decisions/492 updates/9,840 optimizer steps** since10,112/44/880. Spentfull29,184/suffix33,792 plusorigin10,112 equals73,088. All prior stopped budgets,failed/incomplete A/B/C results and early bounded reports remain unchanged; no future curriculum budget is consumed on paper.

## C73,088 finalized: actual capture-approach exposure, FL still lacks touchdown

Final `runs/ppo_semantic_v3/validation/20260906T1758410256842Z_gc34262abffc1_1c3cc10c00874db8a1b189eb741a4832` reloadscheckpoint73,088 atc34262a,N1,seed2001,naturalP01,fixed heteroscedastic mean,no teacher and0 optimizer updates. Execution is SUCCEEDED,while task/controller successfalse. Actual **649 decisions/5,192ticks/43.266666666666666s**,all8ticks each,end atP05 stageage30s with `INCOMPLETE_CONTROLLER_BLOCKED`. Physicalvalidtrue/termination_reason=null identifies incomplete FL capture,not a body collision,sensor/interface failure or video cutoff. PhaseP01–P13 is **[1,194,3,1,450,0,0,0,0,0,0,0,0]**.

Final `eval_73088_diagnosis.md` confirms FR Q48/C1,570/P1,584,including real first/second obstacle contacts1,583/1,584. FL Q1,669/C2,708 is likewise genuineAIR lift/crossing,but no placed event occurs. FL remainsAIR1,605–5,192 inclusive,3,588 raw samples; P05 obstacle-active/force samples are0. Its nearest measured gap is **+2.763440mm at2,791**,front+3.200832mm,AIR/force0; finalgap+5.563301mm/front+10.349873mm/load0. Inactive contact-point fields are not touchdown evidence. FR is currentTOP/load.426240811,RL/RR areGROUND/load.500309320/.073449869. Rear Q/C/P never occur because those stages were not reached.

Unlike the P06-only training block,the **new first-capture approach formula is actually exercised in this evaluation**:1 FR and311 FL decision-end snapshots (310nonterminal+1terminal FL diagnostic),with0 current real-contact fraction on those samples. FL's capture value spans.117010143–.450217285; at2,792 actualgap+2.764372mm givescapture.450217285/phi.401634235 but stillnoTOP. The hard `placed_FL=.85` remains qualification/crossing/topgeometry partial progress,not placement. Terminal diagnosticphi is not bootstrapped: actual potential_after0,terminalevent−40,totalreward−42.001499329. The diagnostic report's649-row PBRS recomputation has maximum mismatch0. Exposure is evaluation data,not new PPO credit or evidence that the preceding six P06-only updates trained this branch.

All5,192 native ticks verify,own-phase effects5,188 exclude ordinary incomingholdticks,and all four state-write categories remain0. Fullraw5,193rows are finite/continuous. **P06–P13 are absent**,so neither the P06 tail,rear geometry advice nor continuousP13 stopping is physically exercised by this evaluation. C72,320/C68,224 P05 incomplete and older failures remain preserved; minor gap/time differences across changedruntime/weights do not establish paired causal improvement. Latest completed full-P01 is now **C73,088/P05 incomplete**. Training remains19 finalizedblocks,73,088/536/10,720; no suffix/fullsuccess,new successfulvideo or future P07 budget is credited.

## Twentieth block finalized: P07 continuous suffix to76,416

Run `runs/ppo_semantic_v3/train/20260906T1805562366558Z_gc34262abffc1_6e47bada5505418e811416a50bd871f5`,same c34262a/N1/seed1001/P07offset0,exact-resumes73,088. It actually optimizes **3,328 decisions/26 PPO updates/520 optimizer steps** before honoring the normal `stop_after_update.request.json` at a complete verified boundary. Original plan4,096,actual/finalized requested3,328,unconsumed768,rounding0; wall3367.614808900049s. Lifecycle is **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY**,not original-budget completion,optimizer failure or task success. Final immutable checkpoint76,416/562/11,240 records SHA8273a81e70e08109c4ae5249034a767ec5d43c17e7c60b0fbb3b1c864971a697/roundtriptrue; root matched actual checkpoint and sidecar hashes to pointer. No duplicate rehash was used here.

Complete `p07_continuous_block_76416.md` reconciles every policy row73,089–76,416 with final manifests: P07=9/P08=10/P09=3,309,all other phases0. Episode decisions are **78,452,452,452,453,452,452,452**,total3,243 completed; episode8 contributes **85 optimized but nonterminal tail decisions**,globals76,332–76,416. The first episode retains P09 BODY_COLLISION; the next seven retain P09 INCOMPLETE_CONTROLLER_BLOCKED. None is relabeled success. Tail endsP09 at6,632ticks/55.266667s,stageage5.533333s,not a ninth task failure. RR Q/C appears in completed episodes3–7 but none ever places RR; RL has no Q/C/P. Tail RR Q6,498 remainsAIR/front−43.575611mm/clearance+29.762970mm/load0 and has not crossed. FL/FR are then currentTOP/load.063642849/.459865905; RL isGROUND/load.476491247. Teacher placement history is not substituted for current support.

Nine accepted prefixes each744decisions/5,952ticks add **6,696 teacher decisions/53,568ticks**,excluded from policy storage and counters. Teacher raw/projected actions are zero,allnativeverified,no-statewrites; FR Q71/C1,665/P1,695 and FL Q2,461/C3,115/P3,583 are teacher preparation. PPO **26,622ticks=3,328×8−2**; the unique short interval is firstcollision/global73,166 with6ticks. Core totals10,024decisions/80,190ticks include teacher separately. All26,622 individual native records verify/effect,own26,604 excludes18 incomingholdticks; all four state-write categories0. The **18 ordinary phase transitions remain nonterminal,time_outs=false,bootstrap-allowed**,corroborated by the env→officialPPO done path; no phase-specific GAE terminal mask is introduced. This is an audit of emitted flags and producer logic,not a new tensor recomputation. All26 update records537–562 are actor-changed/finite-nonzero-gradient,20optimizersteps each,128global increments,hash-continuous from source to saved actor. Five epochs×four minibatches remain configured; update-end LR1e-5 is not a claim about unseen per-minibatch rates.

The stop is motivated by **confirmed premature same-tick servo headroom preclipping**,not refusal to optimize failed episodes. `post_mapper_servo_headroom_readonly.md` preserves184 strict identity-geometry episode3 boundaries with actual remaining post-mapper margin. Its terminal−16.2 residual is explicitly slew recovery,not falsely called saturation. The report supports a separate single-factor execution-range correction,not proof that it caused every failure and not permission to relax physical hard limits. The initial128,episode0 andepisodes1–2 diagnostic files remain intact. Front-wheel1.2 and GPU-copy prototypes remain unpublished/unwired; neither CPU result is added to actual task credit.

Twenty finalized blocks add **66,304 decisions/518 updates/10,360 optimizer steps** since10,112/44/880; full29,184/suffix37,120/smoke0 reconcile to76,416. Original A failure and every prior failed/incomplete result remain preserved; no suffix/full success or paired stability improvement is claimed.

## C76,416 finalized: P05 FL capture still absent

Final unchanged-c34262a evaluation `runs/ppo_semantic_v3/validation/20260906T1908330051233Z_gc34262abffc1_fd1eba3dea714afca139357945d47440` reloads checkpoint76,416 for N1/seed2001/naturalP01/fixed heteroscedastic mean with no teacher and0 optimizer updates. Execution SUCCEEDED/exit0; taskfalse. Actual **648 decisions/5,184ticks/43.2s**,phase[1,193,3,1,450,0,0,0,0,0,0,0,0],endP05stageage30s `INCOMPLETE_CONTROLLER_BLOCKED`. Physicalvalidtrue/physicalreasonnull and window_ended_before_task_terminal=false distinguish incomplete capture from collision,interface or externally truncated evaluation.

`eval_76416_diagnosis.md` directly verifies FR Q47/C1,562/P1,576,including actual pair-verified obstacle forces41.5870933533N/7.8683032990N at1,575/1,576. FL Q1,661/C2,700 is real but no placement occurs: rawFL remainsAIR1,597–5,184 inclusive,3,588samples,with0 P05 obstacle-active samples and0 nonzero normal-force samples. Closest absolute gap after center crossing is+5.271797mm at2,784/front+4.638377mm,AIR/force0. TerminalFLclear+6.397297mm/front+5.655411mm/load0/topgeometrytrue/TOPfalse; `placed_FL=.85` is partial geometry/history score,not capture. FR remains currentTOP/load.428756311. Rear Q/C/P remain absent.

All5,185rawobservations arefinite/continuous,bodydetected/realpair/persistent0. All5,184individualnativeaudits verify actual reconstruction/staged-dispatch/same-tickcounterfactual and fourstatewritecategories0; decisionown-phase5,180 excludesfourholds. Firstunfinished task is **FL real surface contact/capture**; residualdispatch remains live after finite nominal advice. No P06–P13 sample exists,so this result does not physically exercise a rear headroom correction or P13 stop. Latestcompleted full-P01 is nowC76,416/P05incomplete; C73,088 and all prior results remain unchanged. Training counts remain76,416/562/11,240. No next-version run,success or video is prefilled.

## Twenty-first block finalized: post-mapper headroom P07 to78,464

Committed runtime **e99fde1b3e8366f0ff1d484140b82877df745f0c**,run `runs/ppo_semantic_v3/train/20260906T1928297576378Z_ge99fde1b3e83_639e48bdc9c4499eae7e839939fe0e71`,N1/seed1001/P07offset0,uses explicit NewMdpWarmStart from76,416. The independent fixed `post_mapper_headroom_p07_initial_128.md` preserves actual source/initial receipts: actor including learned std,critic,identity normalizer,RNG,policy/runner configuration,76,416/562/11,240,budget29,184/37,120 andorigin10,112 are retained. Fresh Adam initial3e-5 replaces old moments/source LR1e-5; old rollout and physical state are not inherited. The execution correction moves servo2° headroom to the actual same-tick mapped/geometry-corrected nominal plus controller bias; true hard limits/final slew remain. It is not a reward,nominal,hard-task-criterion or frozen-A change and is not called PPO-only improvement. The fixed first128 output/headroom proof remains bounded,not overwritten by final statistics.

Final run/training manifests and root-confirmed processexit0 report **SUCCEEDED execution**,actual=requested=planned2,048,unconsumed0/rounding0. Actual **2,048 decisions/16 PPO updates/320 optimizer steps** yield **78,464/578/11,560**. Recorded wall1933.4150088999886s is the training-loop field: later resets/prefixes are inside it,the first prefix is before it; this is neither an all-prefix-excluded policy benchmark nor all-process total. Final immutable `checkpoint_step_000078464.pt` sidecar records SHAd09fa40910d8d7986d9813d3f8659b45c84d8b19daceffc70f6477a8de2ae644/roundtriptrue. Root verified finalpointer/counters/roundtrip; no additional checkpoint hashing is claimed here.

Complete `post_mapper_headroom_p07_block_78464.md` reconciles contiguous policyglobals76,417–78,464. Actual phase credits **P07=5/P08=5/P09=2,038**,all other phases0. Four completed episodes each452decisions/3,616PPOticks endP09 at9,568ticks/79.733333s,stageage30s: globals76,417–76,868;76,869–77,320;77,321–77,772;77,773–78,224. All four are `INCOMPLETE_CONTROLLER_BLOCKED`,physicalvalidtrue/nullphysicalfailure. Fifth tail **240decisions/1,920ticks**,globals78,225–78,464,endsP09 at7,872/65.6s with terminationnull. Its240samples are optimized,not a fifth failure or success. The requested training budget is complete but tasksuccess remains0.

RR qualification ticks are6,405/6,423/6,108/6,043 across the completed episodes. Only episodes2/3 cross,at6,200/6,170; no episode places RR and RL has no hard Q/C/P. Terminal RR fronts−161.272105/−250.108179/−225.922015/−347.104829mm demonstrate that crossed history is not current forward placement. Episodes0/1/2 terminateRR AIR/load0,while episode3 isGROUND/load.029237532; episode2 currentFL isTOP/load.140769404 but other completed terminalsFLAIR/load0. TailRR Q6,306 has no crossing/placement and isAIR/front−108.987954mm/clearance−3.039366mm/load0; tailFL isAIR/load0. No support or contact is inferred from teacher placement history or a partial `placed_RR` scalar.

Five accepted actual teacher prefixes each744decisions/5,952ticks total **3,720/29,760**,excluded from storage and policy counters. Every prefix raw/projected residual is0,policy_creditfalse,all29,760nativeverified/noepisodewrites. Front FR Q71/C1,665/P1,695 and FL Q2,461/C3,115/P3,583 remain teacher events. Policy **16,384ticks=2,048×8** has no short interval; fullcore5,768decisions/46,144ticks includes the teacher separately. All16,384 individual policy native audits agree with summaries and verify/effect; own16,374 excludes10handoffholds. Fourstatewritecategories0,teacher_initialized_suffix scope preserved. Ten ordinary P07→P08/P08→P09 transitions are nonterminal/time_outsfalse/bootstrapallowed,not GAE phase boundaries. Numerical tensor GAE is not separately recomputed by this report.

Sixteen optimizer rows563–578 each have20steps,actorchanged/finite-nonzero-gradient,128global increments and all15neighborhashlinks matching. Firstactorbefore equals source0f4edbbd…7494; lastafter equals savedcc5caa5d…36ea. Final identitynormalizer remains sourcec230b0db…4552. Fiveepochs×fourminibatches remain configured;16recorded update-end LRvalues1e-5 are not statements about unlogged per-minibatch rates. Saves76,544/76,928/77,440/77,952/78,464 retain separateimmutable receipts.

Twenty-one finalized blocks add **68,352 decisions/534 updates/10,680 optimizer steps** since10,112/44/880. Full29,184/suffix39,168/smoke0 plusorigin reconcile to78,464. All20previousblocks/stoppedbudgets and A/B/C failures remain. Latest completed full natural-P01 remainsC76,416/P05incomplete; no C78,464 evaluation is fabricated. Frontwheel1.2/GPUcopy candidates remain unpublished. Root has started **block22 same-MDP naturalP01 requested4,096** from78,464,without NewMdpWarmStart,in `runs/ppo_semantic_v3/train/20260906T2007180722147Z_ge99fde1b3e83_4cec420c803f4493a1cdb29abf662bd0`. This is launch-only/RUNNING; none of its logs were scanned and none of its future counts are credited here. No A/probe-success gate,full/suffixsuccess,paired improvement or successful video is claimed.

## Twenty-second block finalized: same-MDP naturalP01 to82,560

Unchanged **e99fde1b3e8366f0ff1d484140b82877df745f0c**,run `runs/ppo_semantic_v3/train/20260906T2007180722147Z_ge99fde1b3e83_4cec420c803f4493a1cdb29abf662bd0`,N1/seed1001/full_episode,naturalP01,exact-resumes78,464. Started arguments explicitly set NewMdpWarmStartfalse and policy-distribution-migrationfalse; no new warm-start or teacher-prefix artifact is created. Source/final runtime hashes match. The previous new-MDP ancestry remains provenance,not a repeated Adam reset or another execution/reward revision in this block.

Root confirmed processexit0 and savedpointer82,560/610/12,200/roundtriptrue. Actual=requested=planned **4,096 decisions/32 PPO updates/640 optimizer steps**,unconsumed0/rounding0,recorded `wall_time_s=1553.7430133000016`; lifecycle **SUCCEEDED execution**,not task success. Immutable `checkpoint_step_000082560.pt` sidecar records SHAcc8d148603553b4fd0fe0019bae0fb51bc7fde116451f4274c961399614391f5/roundtriptrue; this report did not rehash or load tensors. Full33,280/suffix39,168/smoke0 plus unchangedorigin10,112 reconcile to82,560.

One complete policy-ledger pass in `headroom_p01_block_82560.md` verifies globals78,465–82,560 and actual phasecounts **[6,1102,22,6,892,1570,5,5,488,0,0,0,0]**. Five completed episodes are:

| Episode | Policy globals | Decisions / physics ticks | Terminal stage / seconds | Actual outcome |
|---|---|---:|---|---|
|0 |78,465–79,502 |1,038 /8,304 |P09 /69.2 |INCOMPLETE_CONTROLLER_BLOCKED |
|1 |79,503–80,284 |782 /6,250 |P09 /52.083333 |BODY_COLLISION |
|2 |80,285–80,872 |588 /4,699 |P09 /39.158333 |BODY_COLLISION |
|3 |80,873–81,517 |645 /5,159 |P09 /42.991667 |BODY_COLLISION |
|4 |81,518–82,168 |651 /5,207 |P09 /43.391667 |BODY_COLLISION |
|5 / nonterminal tail |82,169–82,560 |392 /3,136 |P06 /26.133333 |termination=null |

Completed decisions3,704 plus392optimized tail equal4,096. The tail is not a sixth failure or success. Episode0's physical evaluator is valid/nullfailure; the four collisions retain valid physical readings and `TASK_FAILURE_BODY_COLLISION`,not interface errors. Compact training evidence is not expanded into invented exact-pair force histories or a proved collision cause. All six episode starts are actualP01/decision1/tick8; teacher count0. Consequently FR/FL Q/C/P in each segment are real current-policy events,not inherited prefix credit. FL placementticks2,654/2,654/2,734/2,687/2,775/2,711 are recorded. Only episode0 has RRhardQ5,525/C5,610; noRRplacement anywhere and noRLhardQ/C/P. Episode0 endsRR AIR/load0/front−275.748905mm/clearance+19.609050mm while FLisTOP/load.068578912; fourcollisionterminalsFLAIR/load0 andRRwithoutqualification. TailRR isGROUND/front−488.535039mm/clearance−50.264737mm/load.222947229 andFLisTOP/load.258650657. Historical crossing and partial progress are not current support or fulltask success.

All **32,755 individual native ticks** verify/effect and match summaries; own32,710 excludes45incomingholdticks. Fourstatewritecategories0. Exactdeficit13 from4,096×8 is explained only by fourterminaldecisions:global80,284=2ticks,80,872=3ticks,81,517=7ticks,82,168=7ticks. No teacher ticks inflate PPO/core counts. All45ordinary phase transitions remain nonterminal/time_outsfalse/bootstrapallowed; no phase-specific GAE cut is introduced. Thirty-tworolloutfiles exist but were not tensor-loaded or numerically re-audited here.

All32optimizer records579–610 have128global increments,20steps,actorchanged andfinite-nonzero-gradients. Their31adjacenthashlinks match sourcecc5caa5d…36ea tofinal605d1772…17ad. Identitynormalizer remains sourcec230b0db…4552. Configured5epochs×4minibatches equals20steps/update; actualsum **32×20=640**. All32recorded update-endLRs are1e-5,without an unlogged per-minibatch claim. Nine immutable saves span78,592–82,560. The naturally sampled front captures and RRcross are not deterministic evaluation or paired proof of headroom benefit.

Twenty-two finalized blocks add **72,448 decisions/566 PPO updates/11,320 optimizer steps** since10,112/44/880; every earlier finalized/stopped block and failed A/B/C result remains. Root has launched **C82,560 naturalP01 evaluation**,currentlyRUNNING; it was not scanned or prefilled by this update. Latestcompleted full-P01 remainsC76,416/P05incomplete. No success/probe gate,full/suffixsuccess,paired stability superiority,new successfulvideo or later training budget is credited. The stale pre-final active21 paragraph has been removed from the current header while its original bounded evidence remains preserved in the initial128 report and historical block21 section.

## C82,560 finalized — early P09 verified body collision, no RR qualification

This later finalized evaluation supersedes only the pending/current-evaluation status in the preceding time slice; every training figure and historical failed A/B/C result remains unchanged. `runs/ppo_semantic_v3/validation/20260906T2034252033630Z_ge99fde1b3e83_4253287d07a646d6abe47a4f3354075d` completed SUCCEEDED execution/exit0 on e99fde1 with saved82,560,deterministic fixed mean,N1seed2001,naturalP01,no teacher and0 optimizer updates. Actual **764decisions /6,107ticks /50.8916666667s**,P09 BODY_COLLISION; validphysical=true and `TASK_FAILURE_BODY_COLLISION`,not a timeout or interface failure. Phase counts are **[1,189,3,1,151,407,1,1,10,0,0,0,0]**. Lastdecision3ticks explains total764×8−5 and is a genuine terminal without bootstrap.

Real FR Q45/C1,530/P1,544 and FL Q1,628/C2,557/P2,755 precede P06 completion6,016,P07→P086,024 andP08→P096,032. Collision follows after only75P09ticks/.625s. **RR has no initial-clearance event and no qualified/cross/place history**; RL also has noQ/C/P. Of75P09rawticks,RR has60ground-active and15AIR; maximumAIRclearance remains−44.732659mm belowtop. This is not a legal rear crossing subsequently lost.

First exact verified `base_link`/`/World/Obstacle` pair activates at6,106/50.883333s withFz32.738990784N/consecutive1; the next at6,107 hasFz12.651303291N/consecutive2 and detected/persistent=true. Both record geometrypenetration0,which does not negate measured persistent contact. Contact positions havez0.050076775/0.049994137m near top0.05m; no unobserved face/cause is assigned. Terminal derived roll/pitch≈12.1676°/−15.3936°,CoMinside=true; neither that CoM diagnostic nor base-originheight disproves bodycollision.

FL was genuinely obstacle-active on3,238P06samples,including13.070292N at6,016; from6,023 it is continuouslyAIR for85samples,including allP09ticks withforce0. Terminal FLclearance+177.357038mm/load0 andRR AIR−44.732659mm/load0 differ from FR TOP/load.397502763 andRL GROUND/load.602497237. Historical FLplaced is not current support. FinalRRhip native55.6+residual5.453242569→drive61.053242569°,RRknee0+3.917300189→3.917300189°; FRwheel−.63+.168737963→−.461262037rad/s. Finalgeometry adjustment isidentity,not a guarantee about the cause of the collision.

Once-read raw6,108observations and native6,107ticks are finite/contiguous and allnativechecks verified; own6,099 excludes8incomingholds. Staged/dispatch/frozen-reconstruction/counterfactualchecks pass and allfour statewrites remain0. Full targeted evidence is in `eval_82560_diagnosis.md`. Training stays **82,560 /610 /12,200**,22finalizedblocks; no new full/suffixsuccess,paired stability improvement or successfulvideo is claimed.

## Twenty-third block finalized — P10 continuation stopped after19 complete updates

Run `runs/ppo_semantic_v3/train/20260906T2046002070189Z_ge99fde1b3e83_aa661e98d21b4ca484288779f0a064eb`,unchanged e99fde1/N1/seed1001/P10offset0/phase_suffix,finalized **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY**,rootconfirmed exit0/CLOSED. Immediate source82,560/610/12,200 resumes without NewMdpWarmStart,policy-distribution migration or resume-migration plan. Source/final runtime-contract records match; policy/Adam continuation is not a new MDP or a reset to initial learning. The recorded stop request asks to rebalance curriculum after repeated P12 incompletes and evaluate the actual latest saved checkpoint,without any failed-A/manual-probe success gate or mid-rollout kill.

Actual **2,432decisions /19PPOupdates /380optimizersteps →84,992 /629 /12,580**. Originalplanned4,096,finalizedrequested2,432,unused1,664,rounding0. Final `checkpoint_step_000084992.pt` SHA `ebabc03d1d3611bbe4ab1c05feddbbe3102eedb71ae9d5924cd8f77f2826ea01` and save_load_round_trip=true are bound by its immutable sidecar; root actually checked file/sidecar/pointer agreement,no hash recomputation here. Actual immediate-source spending33,280full/39,168suffix becomes33,280/41,600,origin10,112/smoke0. An older retained `source_stage_requested_decisions`29,184/37,120 is ancestry from the e99 new-MDP boundary,not this block's immediate start ledger.

One final policy pass confirms contiguous globals82,561–84,992 and phase P01–P09=0each/P10=5/P11=5/P12=1,487/P13=935. Final `p10_headroom_block_84992.md` binds these rows to manifests,completed episodes and compact native evidence; the earlier first-P13 window remains separately preserved in `p13_headroom_first_episode_readonly.md`.

| Episode | Globals | Decisions /policy ticks | Endpoint /physical episode time | Actual outcome |
|---|---|---:|---|---|
|0 |82,561–83,530 |970 /7,760 |P13 /127.866667s |INCOMPLETE_CONTROLLER_BLOCKED |
|1 |83,531–83,983 |453 /3,617 |P12 /93.341667s |INCOMPLETE_CONTROLLER_BLOCKED |
|2 |83,984–84,436 |453 /3,617 |P12 /93.341667s |INCOMPLETE_CONTROLLER_BLOCKED |
|3 |84,437–84,889 |453 /3,617 |P12 /93.341667s |INCOMPLETE_CONTROLLER_BLOCKED |
|4 /unfinished tail |84,890–84,992 |103 /824 |P13 /70.066667s |termination=null |

Completed2,329 plus103already-optimized tail equals2,432. TailP10/P11/P12/P13=1/1/66/35; it is not a fifth failure or success. All four completed physical evaluations are valid with nullphysicalfailure—task incompletes,not software execution or sensing failures. Episode0 real current-policy RLQ7,961/C8,063/P8,138 reachesP13 but does not stop. Episode1 has noRLQ; episode2 Q9,228 is revoked atground9,588 withoutcross/place. Episode3 Q7,974→ground8,015,Q8,995→ground9,640,Q10,044 remainsqualified atterminal but has noRLcross/place; the first-Qeventtick must not conceal intervening revocations. Finaltail RLQ7,958/C8,066/P8,128 is actual policy credit but only partial-task progress.

All five starts have teacher948decisions/7,584ticks reachingP10at63.2s; actualpolicydecision1 ends7,592. FRQ71/C1,665/P1,695,FLQ2,461/C3,115/P3,583 andRRQ6,938/C7,109/P7,579 are already present in every start and are **teacher events,not PPO achievements in this suffix**. Prefix files contain4,740teacherdecisionrows plus15start/result/creditmetadata rows,allpolicy_credit=false; everypolicyrow has teacher-in-storage=false andteacher_initialized_suffix scope. Prefix4,740/37,920 pluspolicy2,432/19,435 reconciles core7,172/57,355 over5starts.

All **19,435 compact native tick records** verify/effect; own19,423 excludes12incomingholds,with allfour recorded state-write category sums0. Three genuine one-tick terminal decisions atglobals83,983/84,436/84,889 explain19,435=2,432×8−21. All12ordinary phase changes are nonterminal,time_outsfalse/bootstrapallowed: no phase-label GAE cut. No rollouttensor contents were loaded. Finaltail currentlyRR/RL/FR TOP withload.190907/.440328/.368765,FLAIR0; region/supporttrue butcontrolledfalse/stable0,bodylinear.082898533m/s,wheelmax.393794000rad/s,commandmax.351548247rad/s. Allplacedhistory is not currentfour-wheel support or fullstop success.

All19optimizerrecords611–629 advance128decisions/20steps,allactorschange and reportfinite-nonzerogradients;the source-to-first link and18adjacentupdate links have no mismatch. Sourceactor605d1772…17ad reaches8eb3d28a…63d56c,normalizer remainsc230b0db…4552. Configured5epochs×4minibatches agrees with **19×20=380**. All recordedupdate-endLRs/final/sourceactualLR are1e−5; configuredbase3e−5 is not asserted as a minibatch measurement.

Recorded training-loop wall **2,288.737341400003s**; reset telemetry records23.338547800202s andteacher roll-in1,713.106228699675s acrossall5prefixes. Firstprefixprecedesloop,laterprefix/resetoverlapsloop; these are not summed/subtracted into invented optimizer-onlytime. Final23blocksadd **74,880decisions /585PPOupdates /11,700optimizersteps** since10,112/44/880; allprevious22blocks,failedA/B/Cresults,initialboundedreports andunusedbudgetsremain. Rootnowruns saved84,992 naturalP01eval onunchangede99; thisreportdoesnotreadorprefillitsoutcome. LatestCOMPLETEDfull-P01remains **C82,560/P09BODY_COLLISION**. No suffix/fullsuccess,newsuccessfulvideo or pairedstabilityimprovement is claimed.

## C84,992 finalized — natural-P01 RR placement, RL never qualified, P12 incomplete

This finalized evaluation supersedes the immediately preceding pending-evaluation snapshot without changing any training ledger. `runs/ppo_semantic_v3/validation/20260906T2132007864973Z_ge99fde1b3e83_62520cdf88e549ae9929b9916bd92336` completed SUCCEEDED execution,rootconfirmed session88717exit0/CLOSED. Saved84,992/unchangede99,N1seed2001,deterministicfixedmean,naturalP01,no teacher,0optimizerupdates. Actual **1,353decisions /10,824ticks /90.2s** end atP12age30s with INCOMPLETE_CONTROLLER_BLOCKED,taskfalse/physicalvalidtrue/nullphysicalfailure. Phase counts **[1,197,3,1,150,457,1,1,88,3,1,450,0]** reconcile exactly;200sglobalhorizonnotreached.

Current-policy FRQ45/C1,594/P1,608,FLQ1,693/C2,655/P2,816 and **RRQ7,149/C7,184/P7,191** are realnaturalP01events. RL has initial-clearance attempts but nohardQ/C/P. P06→P076,472/P07→P086,480/P08→P096,488;P09→P107,192/P10→P117,216/P11→P127,224. AtC'sP10entry59.933333s,RRisactuallyobstacleloaded19.921835N/front+8.248333mm andRLisGROUND/front−245.673683mm,workspace0.897305. At7,208RLworkspace0.999970931doesnotcomplete;7,216passesworkspacewhileload_readystill0.971185;7,224RLload.037123passeswithRR+FLgenuinesupport,RLstillGROUND0.976787N. AIRbegins7,225,notpre-earnedqualification. This actualCstate/history differs from knownA-teacherP10at7,584/63.2s; phase label alone is not equalphysicalinitialization and neither olddelay noroldposture is required.

All3,600P12rawticks containRL3,450GROUND/150AIR/0obstacle/0frontcross. HighestAIR is **7,254:clearance−33.886751mm/front−232.675191mm**,whileRRandFLstillhaveactualobstacleforces16.437492/14.939277N. FourP12initialevents7,236/7,303/7,580/7,597neverbecomehardlift. ClosestAIRcenter7,225remains−195.286330mm; closestanycenter7,800isGROUND/front−165.922996mm,notcarrysuccess. Finalplaced_RL=0,firstmissingRLactive-lift/cross/place;P13neverentered.

RR'sP7,191 isreal21.465583Nobstaclecontact,butcenterfirstretreatsbehindfront7,281;firstno-obstaclecontact7,428isAIRgap,notclaimofpermanentloss;firstGROUND7,595has30.748957N. TerminalRR/RLareGROUND(front−445.589181/−503.132364mm,loads.145687/.497784),FLAIRsince7,793for3,032samples(front−168.880131mm/clearance+21.041979mm/load0),FRaloneTOP/load.356529. BodyXretreats.467113361mfromP12entrytoendpoint. HistoricalRRplaceddoesnotguaranteecurrentplatformsupport;region/supportavailablefalseatend. This establishesregressionanduncompletedRLtask,notisolatedreward/mapper/nominalcausality.

Once-readraw10,825samplesandnative10,824recordsarecontiguous;allrawfinite,bodydetected0/exactbodypairactive0. Nativeverified10,824,own10,813,allfourstatewrites0;terminalbootstrapfalse. Finalfinitewheel-nominal0stilldeliverscanonicalwheelresidual/drive[+.028734387,−.324726297,−.364317100,+.106808831]rad/s withrealfloat32setterreadback;terminalheadroomclippedservoindicesempty. These endpointfactsarenotawhole-episodeabsence-of-clippingclaim. `eval_84992_diagnosis.md` contains the bounded entrywindow and completefinalevidence.

Training remains **84,992 /629 /12,580**,23finalizedblocks,origin-relative74,880/585/11,700. C82,560andallolderA/B/Cfailed/incompleteresultsremainpreserved. No newlyselectedreward/mapperchange,certifiedsaved-C-prefixrollout,full/suffixsuccess,newsuccessfulvideo or pairedstabilityimprovement is credited here.


## Twenty-fourth block finalized — frozen-checkpoint P06 initialization to87,040

Run `runs/ppo_semantic_v3/train/20260906T2211507063967Z_ga580fc2add81_e539358648084c70bf6b28037515480c` finalized both training/run manifests **SUCCEEDED execution**; root confirmed session99244 closed/exit0 and pointer/sidecar87,040/645/12,900/roundtriptrue. It runs a580fc2/v3/N1/seed1001/P06offset0 with `prefix_source=checkpoint_policy` and an explicit NewMdpWarmStart from saved84,992. Actual=requested=planned **2,048 decisions /16 PPO updates /320 optimizer operations**, unconsumed0 and rounding0. Final full33,280/suffix43,648/smoke0 and origin10,112 give87,040; all previous unused budgets remain history.

Immutable `checkpoint_step_000087040.pt` has recorded SHA `66fdb6b0ca8be83c31566a0aa6701834d5d8159d4614b4f43251b00a51388729` and save_load_round_trip=true. Once-read final policy ledger is contiguous84,993–87,040, with phase **P06=1,971/P07=10/P08=6/P09=61; all other phases0**. Final `p06_checkpoint_prefix_block_87040.md` binds manifests, episodes, compact native records, fixed-prefix provenance and optimizer receipts. The earlier `checkpoint_prefix_first_live_85120.md` remains a fixed first128 slice, not a substitute for the final audit.

| Episode | Policy globals | Credited decisions / physics ticks | Physical endpoint | Result |
|---|---|---:|---|---|
|0 |84,993–85,325 |333 /2,659 |P09 /45.625s |BODY_COLLISION |
|1 |85,326–85,597 |272 /2,173 |P09 /41.575s |BODY_COLLISION |
|2 |85,598–85,951 |354 /2,831 |P09 /47.058333s |BODY_COLLISION |
|3 |85,952–86,276 |325 /2,593 |P09 /45.075s |BODY_COLLISION |
|4 |86,277–86,573 |297 /2,370 |P09 /43.216667s |BODY_COLLISION |
|5 |86,574–86,931 |358 /2,859 |P09 /47.291667s |BODY_COLLISION |
|6, unfinished |86,932–87,040 |109 /872 |P06 /30.733333s |Nonterminal tail |

Six completed failures total1,939decisions;109tail decisions are already optimized but not a seventh terminal. All completed outcomes remain physically valid recorded BODY_COLLISION, not software lifecycle failures. RR/RL have no hardQ/C/P in any completed episode or tail; RR only has a nonqualifying initial-clearance event in episode4/t5137. All six completed terminals have FL AIR/load0 and RR AIR belowtop. The compact training ledger does not independently reconstruct body-pair force/persistence or prove collision causality. At final tailt3688, FL is currentTOP/load.498042404, FR AIR/load0, RL AIR/front−429.949921mm/clearance−39.166557mm/load0, RR GROUND/front−437.383782mm/clearance−49.780226mm/load.501957596.

There are **seven accepted frozen-source84,992 prefixes**, each352deterministic decisions/2,816ticks, nofallback: total2,464decisions/19,712ticks. Aggregate prefix phases P01=7/P02=1,379/P03=21/P04=7/P05=1,050 are excluded from PPO. Each credit seam is actual P06/t2816/23.4666666667s, with176.5333333333s remaining on the same physical task horizon. Source-policy FRQ45/C1594/P1608 and FLQ1693/C2655(first)-or-2648(later)/P2816 are prefix-owned, not new PPO learning. The source actor remains fixed throughout16updates; equal prefix lengths do not imply bitwise-identical physical states. Physical core totals4,512decisions/36,069ticks reconcile exactly without treating seven episodes as one duration.

All **16,357 policy compact-native ticks** verify/effect, own16,332, and allfour state-write categories0. Terminal intervals at85,325/85,597/85,951/86,276/86,573/86,931 are3/5/7/1/2/3ticks, explaining27fewer ticks than2,048×8. Prefixnative19,712also verifies/effects, own19,684, fourwrites0. All18 ordinary P06→P07→P08→P09 transitions are nonterminal/time_outsfalse/bootstrapallowed: no phase-label GAE cut. Checkpoint-prefix actions remain excluded from PPO storage, and no old recorded transition is used as training data.

Initial source→target receipts preserve the full learned heteroscedastic actor/std, critic, identity normalizer, complete RNG, original84,992/629/12,580 and full33,280/suffix41,600 budgets. Adam intentionally resets from source effective1e-5 to fresh3e-5; old rollout/physical state are not inherited. All16optimizer rows630–645 advance128decisions/20steps, actorchanged/finite-nonzero-gradients, with source-to-first and15adjacent hashes matching. Sourceactor8eb3d28a…63d56c reachesfinalded137cc…fad164, while finalcurriculum still pins the frozen8eb3 source. Normalizer remainsc230b0db…4552. Configured5epochs×4minibatches matches16×20=320; update-endLR1e-5 is not a claim about unlogged per-minibatch LR.

Recorded training-loop wall **1,487.5502931000665s**; reset telemetry26.942712999880314s and roll-in864.1623346996494s cover allseveninitializations. Firstprefixprecedesloop, laterprefixesoverlaploop; these times are not summed/subtracted into invented optimizer-only throughput. Twenty-four finalized blocks now add **76,928decisions /601PPOupdates /12,020optimizersteps** since10,112/44/880. Latest completed full-P01 evaluation remains **C84,992/P12 incomplete**; root's newly running saved87,040 evaluation is not read or prefilled. No suffix/fullsuccess, new successfulvideo or pairedstabilityimprovement is claimed. Frozen A's nonpassing result remains unchanged and is not a training gate.

## Completed natural-P01 C87040 evaluation — no new training credit

Finalized evaluation `validation/20260906T2240147110525Z_ga580fc2add81_3846a3f7a0764d6a836c153999750ea6`, a580fc2, saved87,040 fixedmean/N1/seed2001, finishes **877dec/7,011ticks/58.425s/P09 BODY_COLLISION**, validtrue/`TASK_FAILURE_BODY_COLLISION`/taskfalse/optimizer0. Root confirmed SUCCEEDED execution/exit0/CLOSED. This supersedes the preceding block24 historical note that C87,040 was pending; it does not alter any block24 or older training ledger.

Actual phases P01–P13 = **1,195,3,1,151,515,1,1,9,0,0,0,0**. P06→P07 at6,928, P07→P08 at6,936, P08→P09 at6,944 are continuous nonterminal transitions, not episode resets or frozen-pose entry waits. The exact base_link–Obstacle pair first activates7,010 with16.68274498N and point[.54515606,−.23316331,.05080922]m;7,011 has25.99365425N and point[.54584968,−.23312132,.04995762]m, verifiedpairtrue/consecutive2/persistenttrue. Geometrypenetration0 does not invalidate the real contact.

FR Q45/C1,579/P1,592 and FLQ1,675/C2,607/P2,805 are real current-policy full-episode history. RR has no initial event or Q/C/P, so the failure is not a legal RR crossing later lost; RL has initial attempts only. At P09entry RR is GROUND4.04331N/front−219.60559mm, FL AIR+8.43823mm/force0. FL's final73ticks remain AIR; terminalFLclear+169.88762mm/load0, RRAIRclear−46.14172mm/front−234.94169mm/load0. TerminalRL's normalizedload1.0 corresponds to just.299468N absolutegroundforce, not robust support.

During the67tick P09 window, base position changes+41.969/−9.348/+10.743mm, but the live mass-weighted CoM changes+17.443/−26.761/+2.821mm; terminalroll12.4498°/pitch−13.4330°. NewentryCoMz150.820mm differs from oldC84,992 entry155.226mm. The preserved C84,992 +72tick sample had no bodypair despite FLAIR and nominalFRwheel−.63, and its laterRRplacement still ended P12incomplete. NewC87,040 terminates+67, so no+72 sample exists; neither equalphase labels nor these differences isolate a collision cause.

At collision, canonical nominal wheels[0,−.63,0,0] plus actualresidual[−.200230,+.192053,+.259151,+.102329] yield[−.200230,−.437947,+.259151,+.102329]rad/s. All7,011 native rows verify/effect and setter/mapping agreement, own7,003; allfourstatewritecounts0. Lastdecision has3ticks and all7,012rawobservations are finite/contiguous. The full report `eval_87040_diagnosis.md` records the exact12-channel raw/request/mapped/finalfloat32 receipts and physical comparison. The structural front-range revision/probe is separate, with no success/causal credit. Lifetime counters remain **87,040/645/12,900**, twenty-four finalizedblocks; no subsequent planned count is added.


## Finalized block25 — natural-P01 front-wheel range,89088

Run `train/20260906T2301404237753Z_g2677995544c9_2288cdc238fb40bf96f60d1b04de349d` is final SUCCEEDED execution; root confirms session31307 CLOSED/exit0. Actual=requested=planned2,048 decisions/16PPO/320optimizer steps, unused0/rounding0, loop wall745.9372223000973s. Source87,040/645/12,900 becomes89,088/661/13,220; full35,328/suffix43,648/smoke0/origin10,112. Pointer/immutable sidecar bind checkpoint_step_000089088.pt, checkpointSHA `b5befddddbe2e877b9df76926ba3271e17a4e0eb79a352040aae1e01acd69242`, pointer-recorded sidecarSHA `a2090a54580e22b1fda7d440bcea786cd3ff506b07307053d39a4e8b33a46f73`, recordedroundtriptrue; no hashes are recomputed by this report.

Actual policy phaseP01–P13 = **3,581,11,3,449,966,15,2,18,0,0,0,0**. Global87,041–89,088 is contiguous. There is0teacher/prefix credit: current sampling is naturalP01, prefixrequestnull, three episode starts all credited tick8, and core counts equal policy2,048/16,376ticks. All16,376native ticks verify/effect, own16,355, fourstatewrites0. Short terminal decisions87,777=6ticks and88,621=2ticks explain exactly2,048×8−8=16,376. All21phase transitions remain nonterminal/timeoutfalse/bootstraptrue; no artificial phase GAE terminal is introduced.

| Episode | Global range | Decisions / ticks | Actual outcome |
|---|---|---|---|
| 0 | 87041–87777 | 737 /5894 | P09 BODY_COLLISION49.116667s |
| 1 | 87778–88621 | 844 /6746 | P09 BODY_COLLISION56.216667s |
| 2 | 88622–89088 | 467 /3736 | P06 nonterminal tail31.133333s |

Both physical terminal evaluations are valid with TASK_FAILURE_BODY_COLLISION; tailvalidtrue/reasonnull. Two completed episodes sum1,581 plus467 unfinished decisions, success0. RR has only initial5859/6682 in the two terminals and no hardQ/C/P; RL has no hardQ/C/P anywhere. FR/FL genuine current-policy Q/C/P occurs in each natural episode, not teacher history. The terminal first unfinished task is placed_RR (.2836998344/.3226526113), interrupted by collision; FL is currently AIR/load0 in both. Tail first unfinished rear_approach is.0862031167; RR GROUNDfront−427.50952mm and RL AIRfront−448.44922mm, while FL currentlyTOP/load.546019. None is full/suffixsuccess or a third terminal failure.

The sole recorded runtime-file delta is the execution-profile front-wheel range. Source and newinitial actorincludingheterostd/critic/identitynormalizer/trainingRNG/origin/spentledger receipts match exactly; initial87,040/645/12,900 adds no work. Adam resets fromsource1e−5 toinitial3e−5 with changedstatehash and no oldrollout/physical-state inheritance. All16actor updates change parameters with finite nonzero gradients; source-before,15adjacentlinks/finalactor and16×20=320steps agree. Update-end LR1e−5 does not imply everyminibatch'srate. Five immutable saves recordroundtriptrue and unchanged identitynormalizer. Complete single-pass policy/native/episode and scalarGaussian logprob checks are in `p01_front_wheel_block_89088.md` (maximumLPerror1.279569455e−6).

This is finalized block25, not a proven control success or causal collision repair. The range revision changes physical action interpretation, so it is not attributed solely to PPO learning. Latest completed full eval stays C87,040/P09 BODY_COLLISION; root's separate reloaded C89,088 evaluation is not inspected or prefilled. All earlier failed A/B/C outcomes and old budget boundaries remain preserved.


## Completed natural-P01 C89088 evaluation — P06 incomplete, no training credit

Run `validation/20260906T2316032937892Z_g2677995544c9_7a1d828322994ae5a270a90d2aac627c` under2677995/saved89,088/N1/seed2001/fixedmean/naturalP01 is final SUCCEEDED execution; root confirms session90638 CLOSED/exit0. Actual950dec/7,600ticks/63.333333s, P06age40s, `INCOMPLETE_CONTROLLER_BLOCKED`, taskfalse/physicalvalidtrue/nullphysicalfailure/optimizer0. This supersedes the preceding block25 pending-evaluation note without changing its training counts or any historical outcome.

Phases P01–P13=1,193,4,1,151,600,0,0,0,0,0,0,0. P06 entry2,800/23.333333s→deadline7,600/63.333333s. The existing completion is min(workspace_RL,workspace_RR): lateralvalidtrue for both, butRRfront−236.756679mm remains16.756679mm behind−220mm, and RLfront−226.803657mm is6.803657mm short. Terminalrear_approach=.9329732840471127, limited byRR. Best15Hz decision-end pair progress occurs atterminal; no stronger120Hz-extremum claim is made. This is actual remaining workspace, not an old pose/force/clock-entry guard or a new gate.

FRQ45/C1,563/P1,577 and FLQ1,669/C2,604/P2,793 are real naturalpolicy history. RRhasnoinitial/QCP; RLhas10initialhints/nohardQCP. CurrentFLTOP/load.473384/obstaclepair12.867535N differs from FRhistoryplaced but currentlyAIR/load0/+20.832359mm. RRGROUND/load.526616/14.314496N and RLAIR/load0 remain beforeworkspace. Finalrawfinite/exactbodypairinactive0force/bodydetectedfalse is consistent with nullphysicalfailure, but does not establish stability superiority over a later-stage failed rollout.

All7,600native ticks independently verify setter/mapping and summaries report7,600effect/7,595own; allfourstatewrites0. Fivephasechanges are continuous nonterminal/bootstraptrue; terminalbootstrapfalse is P06incomplete, not the200s globaltimeout. Lastpreterminal3snapshots retain live_endpoint_tail+.3nominalwheels; terminal preserves recordedlastdispatch. Terminalfrontnominal[.3,.3], projected[−.318750175,+.506907105], actual[−.018750175,+.806907105]rad/s cannotproveusebeyondold±.6residual. There is no additionalrange sweep. Fullbounded evidence is in `eval_89088_diagnosis.md`.

Current finalizedtrainingcount remains89,088/661/13,220 across25blocks, not93,184. Rootstarts same-HEAD ordinaryresume89,088, P06offset0/checkpoint_policy/N1seed1001 planned4,096 in `train/20260906T2326102309749Z_g2677995544c9_5826da63aa384e5d8f6e7ea4a025cfaf`, session10060; noNewMdpWarmStart and nofuturecountsarecredited. Prefixteacher/resetactionsremainoutsidePPOcredit. C87,040collision and olderA/B/Cresults are preserved; nofull/suffixsuccessorpairedstabilityimprovement is established.


## Finalized block26 — same-MDP P06 checkpoint-prefix,93184

Run `train/20260906T2326102309749Z_g2677995544c9_5826da63aa384e5d8f6e7ea4a025cfaf` is finalSUCCEEDED execution; root confirms session10060 CLOSED/exit0 and PID162828 gone. Actual=requested=planned4,096dec/32PPO/640optimizersteps, unused0/rounding0, loop wall2385.0470698999707s. Source89,088/661/13,220 becomes93,184/693/13,860. Full35,328/suffix47,744/smoke0/origin10,112; across26blocks sinceorigin,83,072dec/649PPO/12,980optimizersteps added. Latestpointer andimmutable sidecar agree oncheckpoint_step_000093184.pt/SHA `f82d1a53f1580b730d7472d18ebd966d4503f83165a0e8d6d416dc2c1f3e0656`, recordedsidecarSHA `58546c94fad050444b3996abf752d2f7d63f84981bc79c0d49fbd02151e077ab`, roundtriptrue. No fresh hashes are computed here.

Policy phaseP01–P13 = **0,0,0,0,0,3025,20,8,589,1,3,450,0**. All4,096rows global89,089–93,184 are continuous and suffix-scoped; teacher/checkpoint-prefix storage flagsfalse. Seven terminals sum3,630dec plus466optimized nonterminaltail:

| Episode | Global range | Decisions / policy ticks | Actual endpoint |
|---|---|---|---|
| 0 | 89089–89449 | 361 /2882 | P09 BODY_COLLISION47.35s |
| 1 | 89450–90307 | 858 /6864 | P12 incomplete80.533333s |
| 2 | 90308–90629 | 322 /2575 | P09 BODY_COLLISION44.791667s |
| 3 | 90630–90988 | 359 /2865 | P09 BODY_COLLISION47.208333s |
| 4 | 90989–91337 | 349 /2786 | P09 BODY_COLLISION46.55s |
| 5 | 91338–92118 | 781 /6241 | P09 incomplete75.341667s |
| 6 | 92119–92718 | 600 /4800 | P06 incomplete63.333333s |
| 7 | 92719–93184 | 466 /3728 | P09 tail54.4s/tick6528, nonterminal |

Onlyep1 earns RRQ5903/C6026/P6027 aftercredit; laterRR isGROUND andP12placed_RL0. Ep5 earns trueRRQ5942 butnoC/P, terminalRRAIR+68.138571mm/front−398.640823mm/placed_RR.35. All8episodes have noRLhardQ/C/P. Fourbodycollision endpoints haveRRunqualified; ep6 firstmissingrear_approach.203984786 andtailplaced_RR.334105256. FLisAIR atallterminal endpoints exceptep5GROUND/load.162764, neverTOP; tailFLAIR/load0. PrefixfrontQ/C/P andhistoricalRRplacement cannotbe substitutedforcurrent support. NoP13/suffix/fullsuccess.

Eight accepted350dec/2800tick prefixes total2,800dec/22,400ticks (phaseP01–P05=8/1544/32/8/1208). EachcurrentP06 credit starts2800/23.333333s, remaining176.666667s; source89088/frozenactor733b1c…33312bd remainsfixed, nofallback. Everyprefixrecord policy_creditfalse; all22,400nativeverify/effect, own22,368/fourwrites0. Policy32,741ticks verify/effect, own32,709/fourwrites0; fivepartial terminal intervals2/7/1/2/1 yield27missingticks vs4096×8. All24phase changes are nonterminal/timeoutfalse/bootstraptrue. Combinedphysicalcore6,896dec/55,141ticks is exactlypolicy+prefix, notextraPPO. Resetwall27.6551573s/rollin985.7647375s are separate telemetry, not disjoint additions toloopwall (firstprefixbeforeloop,laterprefixesinside).

This remains2677995 ordinaryresume with sourceAdam retained, notanewMDPmigration. All32updates662–693 changeactorwithfinitegradients; source-before,31adjacentlinks/finalactor agree,32×20=640actualsteps. Update-endLR1e−5 doesnotproveeveryminibatch'srate. Ninepublishedsavesrecordroundtriptrue andunchangedidentitynormalizer. Finalactor `fc9e30f050fde45629ee4964cf352c7f5dc7f26fd1265c7b746a01fbb3e5dfc5`; finalcritic `20fb832b985a0ac88db76655c678bc44306aca03e01cf91ed5565d2b6ddd62e7`. Samecompletepolicy pass checks finitepositive12D distribution andLPmaxerror1.456163938e−6. Detailsare `p06_same_mdp_block_93184.md`; `p06_same_mdp_first_89216.md` and1219rowcontrastremainunchanged.

Block26 isfullyfinishedasanoptimizerbudget, nottasksuccess. LatestcompletedfullP01evalremainsC89,088/P06incomplete; planned/pendingreloadedC93,184naturalP01 resultisnotreadorprecredited. All26traininghistoriesandoldfailedA/B/Cresultsarepreserved.


## Final C93,184 natural-P01 evaluation — FL capture incomplete

This final result supersedes only the earlier pending-C93,184 status, without changing the twenty-six-block training ledger: **93,184/693/13,860**, additional83,072/649/12,980 since origin10,112/44/880; spending full35,328/suffix47,744 remains unchanged.

Run `validation/20260907T0009333548816Z_g2677995544c9_869bd7c54b5f4b688d9582b8a4e507ae` finalized SUCCEEDED execution/exit0 but taskfalse: **657 decisions/5256ticks/43.8s, P05 INCOMPLETE_CONTROLLER_BLOCKED**, naturalP01, fixed mean, N1seed2001, optimizer0, physicalvalidtrue/nullhardfailure/windowEndedfalse. Source-phase counts P01–P13 are **[1,201,4,1,450,0,0,0,0,0,0,0,0]**. P02/P03/P04/P05 begin at ticks8/1616/1648/1656; P05 then reaches30s without FL placement.

FR Q45/C1627/P1641 and currentTOP; FL Q1731/C2763 but neverplaced; neither rear has hardQ/C/P. First missing value `placed_FL=.85` is .35qualified+.35crossed+.15geometry, not an actual TOP-contact result. FinalFL is AIR/load0, topgap+2.410625489mm/front+17.634275983mm, with recorded3587consecutiveAIR samples. P05 has449AIR/0TOP/0obstacle-active decision-end snapshots out of450. Closest qualified+crossed+topXY decision endpoint is tick4264, gap+1.549241515mm/front+18.195870946mm; selected raw4260–4268 confirms localminimum4264 and9finite geometry-verified AIR/zero-force samples. No global120Hz gap minimum or independently scanned fullAIR interval is asserted.

Final actual supports are FR obstacle(load.421856345/normal12.094018331N), RL ground(.498455901/14.290018082N), RR ground(.079687754/2.284533978N). Supportcount3, CoM inside/+20.423134mm margin do not substitute for absentFLcapture. Finalrawfinite/bodycollisionfalse; a stale non-null inactiveFL contactpoint is not counted as support. All5256 native ticks verify/effect, own5252/fourstatewrites0, continuous657×8 ledger/no finitefallback. This evaluation never reaches P06–13, so the P06 tail/rear geometry/frontwheel1.2 opt-in are not directly exercised. No causal claim about policy forgetting or the prior training block follows.

See `eval_93184_diagnosis.md` for the bounded final diagnosis. Historical C89,088/P06 and every prior A/B/C result remain unchanged. The subsequent planned same-MDP P01 full8192 ordinary resume93,184 is not included in actual counts; no new stream was read. No successful full/suffix episode, paired stability benefit or new successful video is recorded.


## Finalized block27 — natural-P01 same-MDP checkpoint101,376

Run `train/20260907T0017590892338Z_g2677995544c9_bf91484d9ac74584b18ac862afe2f22c` finalized SUCCEEDED execution/exit0, source93,184/693/13,860 → **101,376/757/15,140**. Actual=requested=planned8192/64/1280, unconsumed0/rounding0; training-loop wall2950.030770600075s. This is ordinary same-MDP resume under2677995, not a new migration/Adam reset. Full spending43,520/suffix47,744/smoke0; since origin10,112/44/880,27 finalized blocks add91,264/713/14,260. No C101,376 evaluation outcome is inferred.

Immutable `checkpoints/history/checkpoint_step_000101376.pt` and pointer/sidecar record checkpoint SHA `b4f23b0ff604a278277cbd1ac20d4659f4f4c27e01c5fb4dac3e4f4329e83ce5`; pointer sidecar SHA `09fced18ae3084e9426c0640d8da17b5ccc25552e3837954b7bd558ab6bbf4e1`. All9 save boundaries record roundtriptrue/unchanged identity normalization. No repeated hashing/PT/GPU loading. All64 update actors changed with positive finite gradients, source/63adjacent/final actor-chain agreement and exact128decision/20optimizer increments. Update-end LR≈1e−5 is not a statement about every minibatch.

Full phase vector P01–P13 is **[9,1761,36,10,1335,4558,9,4,470,0,0,0,0]**. Eight terminals have954/941/959/867/1251/812/946/963 decisions, totaling7693:5 P06 INCOMPLETE,1 P09 INCOMPLETE,2 P09 BODY_COLLISION. Episode8 is a **499-decision nonterminal P06 tail**, g100878–101376/tick3992/33.266667s, not a ninth failure or success. The first6-completed-episode audit remains a fixed historical slice.

All9 episodes earn current-policy frontQ/C/P, but terminal/tail FL is always AIR/load0. Only episode4 earns RR Q6998; ground-before-cross9033 revokes current eligibility, with no RR C/P. All other episodes/tail have no RR hardQ/C/P; allRLhardQ/C/P absent. P06 incomplete endpoints have rear_approach0/0/.103842277/.904461283/0; the unfinished tail has0. Episode6 rear frontsRR−235.781666mm/RL−243.884679mm still miss the−220mm workspace bound. Tail rear frontsRR−498.459917/RL−501.087083mm, bothGROUND; FR TOP/load.419073340, FL AIR/gap+2.268118mm. Historical front placement is not current whole-body support.

All65,529 policy/core ticks verify/effect, own65,475/fourstatewrites0, with0sequence/audit/nonfinite-distribution/physicalvalidity anomalies. The sole short interval is episode7 terminal g100877=1tick, explaining8192×8−7. All54 phase changes are nonterminal/bootstrap-allowed. Teacher credit0;9 naturalP01 episodes and no reset-prefix budget. Incomplete endpoints have null physical hard failure; two collision reasons remain TASK_FAILURE_BODY_COLLISION. No raw collision-force mechanics are invented from compact data.

See `p01_block_101376.md`; `p01_block_27_completed_prefix_audit.md` is unchanged. Latest completed deterministic fullP01 evaluation remains C93,184/P05 incomplete. Root's running C101,376 is not read/precredited. No full/suffix success, paired stability benefit or successful video is claimed; A and all earlier outcomes are preserved.


## Final C101,376 natural-P01 evaluation — rear workspace not reached

This final result supersedes the pending-C101,376 status only. Training remains27 finalized blocks: **101,376/757/15,140**, additional91,264/713/14,260 since10,112/44/880; full43,520/suffix47,744 unchanged.

Run `validation/20260907T0108106916731Z_g2677995544c9_3cfbdf407efc46619d280462e8ac47a4` finalized SUCCEEDED execution/exit0 but taskfalse: **966dec/7728ticks/64.4s/P06 INCOMPLETE_CONTROLLER_BLOCKED**, P06age40s, N1seed2001/fixedmean/naturalP01/optimizer0/windowEndedfalse. Physical evaluation valid/nullhardfailure. Phase vector **[1,210,4,1,150,600,0,0,0,0,0,0,0]**, P06 entered2928/24.4s. Actual first missing predicate is rear_approach0 throughout600 P06 decision-end samples.

Current rear frontsRR−823.927995mm/RL−855.566439mm miss the historical2677995−220mm lower workspace bound by603.927995/635.566439mm; lateral span is valid. Independently closest P06 decision-end frontsRR−492.873205/RL−530.957533mm also remain outside, without claiming a global120Hz minimum. Body-forward field2936→7728 moves−.238605406→−.567992438m; no single-actuator causal conclusion follows.

FR Q47/C1699/P1717 and FLQ1813/C2828/P2922 are real current-policy history, while RR/RL have nohardQ/C/P. FinalFL AIR/load0/gap+13.695089mm/front−340.180662mm; finalFR/RL/RR are GROUND, notTOP. Raw exact-ground normal forces12.796389580/14.219820023/1.809562564N support loads.443921832/.493302311/.062775857; FL pairs inactive0N. Final supportcount3/CoM-inside margin+15.124766mm, allfinite/bodycollisionfalse. Historical placement does not mean current capture. P06 decision-end FL AIR587/TOP9 and FR TOP166 counts are sampled, not complete raw contact-history scans.

All7728 native ticks verify/effect, own7723/fourstatewrites0, continuous966×8 ledger and no audit/physicalvalidity/finitefallback anomaly. Terminal nominal wheels+.3 plus residual[−.447268688,−.697472758,−.313221581,+.237757316] gives actualcanonical[−.147268688,−.397472758,−.013221581,+.537757316]rad/s. FR residual itself exceeds old.6, but does not establish why task completion failed.217 decision-end P06tail snapshots are live_endpoint_tail; retirement peak0/gain1, terminal no-new-tail status does not erase the already applied+.3nominal. P07–P13 remain unvisited.

See `eval_101376_diagnosis.md`; all earlier C93184/A outcomes and training histories are preserved. No collision here is not proof of improved stability. The separately developed tracking-reference revision has no evaluated/tested/trained result in this record; no future counts, success or video are prefilled.


## Finalized block28 — tracking-reference verified-update stop103,168

Run `train/20260907T0134080107746Z_g42b91e857a0a_a862752b4ec84772b24f061befcd1676` under42b91e857a0a2da256dee85e8092642ca8e64aeb finalized **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY /exit0**. Source101,376/757/15,140 → **103,168/771/15,420**, actual1792/14/280; originalplanned4096/unconsumed2304/rounding0. Final requested1792 reflects the saved stopping boundary, while planned4096 remains recorded. Training-loop wall723.9258038001135s. Fullbudget45,312/suffix47,744/smoke0/origin10,112. Since origin10,112/44/880,28 finalized blocks add93,056/727/14,540.

The retained stop_after_update request explicitly asks for a complete saved update before a separately versioned workspace-potential-density revision. It is not an optimizer/interface failure, half-rollout interruption, task-success requirement or completed original4096 budget. Tracking-reference is a new-MDP control-semantic revision, not ordinary same-MDP resume: source learned actor/std/critic/normalizer/RNG/budgets retained, freshAdam initial3e−5, no inherited rollout/physical state. First101504 migration/update details remain in `tracking_reference_first_101504.md`; no initial same-state logical comparison is renamed native trajectory equivalence.

Phase P01–P13 = **[2,394,8,2,308,1078,0,0,0,0,0,0,0]**. Episode0 g101377–102333 is957dec/7656ticks/63.8s P06 INCOMPLETE; episode1 tail g102334–103168 is835dec/6680ticks/55.666667s P06 **nonterminal**. Their respective P01–P06 counts are1/205/4/1/146/600 and1/189/4/1/162/478. No teacher credit and no P07–13 samples. Physicalvalidtrue throughout, terminal hardfailure null, tasksuccess0. Only one completed failure, not two.

Front current-policy Q/C/P: episode0 FR42/1661/1680,FL1788/2802/2849; tailFR46/1522/1547,FL1642/2692/2853. RearhardQ/C/P allabsent. RR initial-clearance hints27/11 and one RLinitial each are not qualification. Both endpoints rear_approach0; completedRRfront−668.114852mm/RL−560.309321mm, tailRR−665.977013/RL−595.236626mm. TailrearGROUND/load.043618544/.528435259; FL AIR/load0/gap+34.787869mm, FR GROUND/load.427946197. Historical front placement does not provide currentTOP support or rear transfer.

All14,336 native ticks verify/effect, own14,326/fourstatewrites0; continuous1792×8 sequence,10phase changes nonterminal/bootstraptrue,0audit/nonfinite-distribution/physicalvalidity/fallback anomalies. All1792 decision-end reference receipts independently verify previous-ACK mode; this does not count every channel's active reference at every compact120Hz tick. All14 updates758–771 changeactor with nonzero finite gradients and continuous source/13adjacent/final hash chains, exact128decisions/20optimizer increments. Gradientnorm1.0003439493–1.4142136024; update-endLR≈1e−5, separately from initial3e−5.

Three updated saves101504/102400/103168 record roundtriptrue/normunchanged. Immutable103168 CP recordedSHA `005d8794fc4fbae5616c478321ddd1d0e2d8f3082fd94caec22d3acbb6a6c0c3`, pointer-sidecarSHA `0c5ce571870dad9e18018dab8d9dfff1f742af7675d4d996335a562343232838` agree with final receipts; no rehash/PT/GPU. Full ledger: `tracking_reference_block_103168.md`. Prior tracking implementation/first128 and every earlier report remain intact.

Latest completed full evaluation remains C101,376/P06 incomplete under2677995. The separately running C103,168 fixed-mean naturalP01 evaluation is not read or prefilled. No new workspace revision/training count, full/suffix success, paired stability improvement or successful video is claimed; all earlier A/B/C failures and budget histories are preserved.


## Final C103,168 natural-P01 evaluation — FL capture absent; reference used

This final result replaces pending-C103,168 status only. Training remains28 finalized blocks, **103,168/771/15,420**, additional93,056/727/14,540 since10,112/44/880; full45,312/suffix47,744 unchanged.

Run `validation/20260907T0147389603120Z_g42b91e857a0a_a502f3cc872645a496b2fa1d6968b077` finalized SUCCEEDED execution/exit0 but taskfalse: **660dec/5273ticks/43.941666667s/P05 INCOMPLETE**, naturalP01/fixedmean/N1seed2001/optimizer0/windowfalse/physicalvalidtrue/nullhardfailure. P05age30.008333s; final1tick interval explains660×8−7. Phase vector **[1,203,4,1,451,0,0,0,0,0,0,0,0]**, P05entry1672/13.933333s.

FR Q48/C1644/P1659/currentTOP, FLQ1754/C2791 but no placement, both rears nohardQ/C/P. Firstmissing `placed_FL=.85` is qualified+crossed+geometry partial credit, not actual TOPcapture. FinalFL rawgap+16.635616692mm/front+5.441311429mm, AIR/load0/verifiedground+obstacleforce0. Closest qualified+crossed+topXY decision-end gap+16.587552996mm at3016, not a global120Hz minimum. P05 endpointsFLAIR449/TOP0/obstacleactive0 outof451; final recordedAIRstreak3581 (not independently rescanned raw). FinalFR obstacle8.971070996N/load.347264298; RL/RRground11.351406097/5.511076927N, load.439405515/.213330188. Rawfinite/bodycollisionfalse/support3/CoMinside+24.222089mm do not replace missingFLcapture.

All5273native rows verify/effect/reference-independent-check, sequence/clockerrors0/fourstatewrites0; own5269. True `reference_used` ticks **1306** (P02=395/P03=8/P04=2/P05=901), channel-use2729 (RLhip796/FRknee1305/FLknee12/FLhip616); first9/last5273. Only1of660decision-end snapshots shows use, so endpointfalse is not evidence of route inactivity.

Actual P05FLhip exampleepisode1709/dispatch1888/prevACK1887: realphysicalq.101337097585rad, nominal4.8°, canonicalerror−.926836436°, previousREQUEST+4.542589552°. Desiredoriginalc0−7.414691489° versus referencec1bounded+10°; oldcomp0 follows original1.25° slew, native6.05°, currentbias+4.542589552°, finalfloat32target.186260506511rad. Same-history zero-current-residual target.142627283931rad retains priorreference and is not an executed old-mode comparison. This is measured reference/dispatch participation, not causal physical improvement.

See `eval_103168_diagnosis.md` for scope and clock/contact details. Prior C101376/C93184/A and all28 training blocks are preserved. No later workspace-potential revision result, future optimizer credit, full/suffix success, paired stability benefit or successfulvideo is inferred.

## Workspace soft-potential revision and actual block29 start

Runtime f1a9bbf650b1b8f80d5169e09173fcfc68797d99 changes exactly three production/config files: stage_task_spec.yaml, semantic_supervisor.py, new semantic_workspace_potential.py. The existing workspace share now has reciprocal distance sensitivity beyond the old clipped range; public entry/completion, Q/C/P, nominal scheduling, action/controller limits and five reward weights remain unchanged. ExistingPhi observation content changes, so this is an explicit new-MDP boundary. See `workspace_potential_revision.md` for the exact scope and preserved test failures caused by CUDA-hidden RNG-device mismatch, followed by a passing visible-device19-case continuation rerun. No RNG or task validation was weakened.

Actual block29 `train/20260907T0205293896232Z_gf1a9bbf650b1_f859147a01004e878cecae656230e34c` is running naturalP01/N1seed1001, requested4096, source103168/771/15420. First saved update **103296/772/15440**, roundtriptrue, CP SHA3e37fca3b32e14b3e0e1b93f8a16c6ece45efefac95bfa0239fd5f514b2b5626. Exact source/initial actor includingstd, critic, normalizer and trainingRNG match; freshAdam3e-5 and emptyrollout/legalreset, counters/budgets retained. All first1024 native ticks verify, no teacher credit/state writes; first128 P01=1/P02=127. `workspace_potential_first_103296.md` documents this fixed boundary, not a completed4096 allocation.

First completed new episode is947dec/63.1333333s/P06 INCOMPLETE, physicalvalid/nullhardfailure, both rear hardQ/C/P absent. Subsequent collection/update continues. This live note does not finalize block29 or modify the previous28-block ledger. No successful checkpoint/video or paired improvement is claimed.

## Finalized block29 — workspace soft-potential, checkpoint107,264

This final append supersedes only the preceding live block29 status; all bounded reports and earlier ledger entries remain unchanged. Run [20260907T0205293896232Z_gf1a9bbf650b1_f859147a01004e878cecae656230e34c](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_semantic_v3/train/20260907T0205293896232Z_gf1a9bbf650b1_f859147a01004e878cecae656230e34c), HEAD f1a9bbf650b1b8f80d5169e09173fcfc68797d99, finalized **SUCCEEDED execution / exit0**, natural P01 / N1 / seed1001. Source103,168/771/15,420 → **107,264/803/16,060**; actual=requested=planned4096/32/640, unused0/rounding0, wall2110.9012369001284s. Since origin10,112/44/880, **29 finalized blocks add97,152/759/15,180**. Spent budgets are full49,408/suffix47,744/smoke0, retaining origin10,112.

The explicit new-MDP initial preserves learned actor including heteroscedastic std, critic, normalizer, RNG and lifetime/budget counters, with fresh Adam3e−5 and fresh rollout/legal reset. It is not ordinary same-MDP Adam continuation. Five updated saves103296/104192/105216/106240/107264 all record roundtriptrue, unchanged identity normalizer, matching update actor and fixed target runtime. Final checkpoint recorded SHA `54449971c1233785133c33b4f362f85e6d449b7c73be4d4ea589d2db54e288d0` matches pointer/sidecar fields; pointer-sidecar SHA `ab23007a4df8aa5be7598e5b738eaf29cad7c0912ec8de2625da993b9f1dda63`. No repeated file hashing or PT loading.

Actual source-phase P01–P13 vector is **[6,931,22,5,1383,1749,0,0,0,0,0,0,0]**. Completed episodes are947/P06,641/P05,938/P06,630/P05 decisions, all INCOMPLETE_CONTROLLER_BLOCKED with physicalvalidtrue/nullhardfailure; their ticks are7576/5128/7504/5033. Episode4 is a **940-decision nonterminal P06 tail**, g106325–107264/tick7520/62.666667s, fully optimized, not a fifth failure or success. The sole short interval is ep3 terminal g106324/P05=1tick, explaining **32761=4096×8−7**; it is incomplete, not collision or missing data. No teacher/prefix credit; five natural P01 episodes. P07–13 remain unvisited, not assigned zero quality scores.

FR hard Q/C/P across episodes0–4:52/1587/1611,55/1500/1515,60/1475/1508,52/1397/1419,55/1558/1600. FL:1729/2749/2775;1612/noC/noP;1610/2636/2702;1508/2669/noP;1694/2730/3123. Every RR/RL hard Q/C/P remains absent. P06 incomplete endpoints and the P06 tail have public rear_approach0; the new reciprocal soft component does not alter that hard condition or imply whole reward0. Final tail RR/RL fronts−.6417162384/−.4799942973m are GROUND with loads.0479227600/.4834657616; FR is GROUND/load.4686114784 and FL AIR/load0/gap+.0341335287m. Historical front placement is not current TOP support. No full/suffix success is recorded.

All32761 native ticks verify/effect, own32738; all4096 decision-end mapping/setter/counterfactual and independent previous-ACK reference checks pass, four in-episode state-write types are0. No sequence, physical-validity, finite-distribution or finite-fallback anomalies. All23 phase transitions are nonterminal;4092 nonterminal decisions bootstrap normally, four real terminals mask GAE/bootstrap and set nextPhi0, with time_outsfalse throughout. Official RSL gamma.995/lambda.95 remains in use; no policy storage is cut merely by a phase label. All32 updates changed actor with nonzero finite gradients, exact128/20 increments and complete source/adjacent/final actor-hash chain. Recorded update-end LR is1e−5 for31 updates and≈1.5e−5 at780; no per-minibatch constant-LR claim.

Full audit: [workspace_block_107264.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/workspace_block_107264.md). The earlier [first2048 authority report](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/workspace_first2048_authority.md) remains restricted to its722 P06 samples and is not extrapolated to this block's1749. Latest completed evaluation recorded here remains C103,168/P05 incomplete; C107,264 and any next course are not read or prefilled. No paired physical improvement, successful checkpoint/video, or task success is inferred from completion of this optimizer budget.


## Final C107,264 natural-P01 evaluation — P05 FL capture incomplete

This append supersedes the preceding pending-C107,264 statement only. The completed block29 ledger remains **107,264/803/16,060**,29 finalized blocks/additional97,152/759/15,180 since10,112/44/880, full49,408/suffix47,744 unchanged. No new training samples or optimizer steps are credited.

Run `validation/20260907T0242264425838Z_gf1a9bbf650b1_be40bfe9223b49a4bec2468f632992d7` finalized SUCCEEDED execution/exit0 but taskfalse: **666dec/5328ticks/44.4s/P05 INCOMPLETE_CONTROLLER_BLOCKED**, P05age30s, saved107264/N1seed2001/fixedmean/naturalP01/optimizer0/windowfalse. Physicalvalidtrue/nullhardfailure. Phase P01–P13 **[1,210,4,1,450,0,0,0,0,0,0,0,0]**; P05entry1728/14.4s. Four phase handoffs remain nonterminal/bootstraptrue/not timeouts.

FR Q50/C1700/P1713/currentTOP; FLQ1808/C2854 but noP; RR/RL nohardQ/C/P. Actual first missing `placed_FL=.7` retains genuine qualification/historical crossing but lacks current top geometry and actual capture. FinalFL AIR/load0/front−28.900113274mm/gap+27.966563972mm/topXYfalse/TOPfalse. P05 decision endsFLAIR449/TOP0/obstacleactive0 outof450; final evaluator3589consecutiveAIR is recordedhistory, not an independently rescanned rawinterval.

Post-cross sampled FL maximumfront+17.998222mm at2880/gap+83.767285mm; first negative-front decision at2912/gap+27.496492mm; closest sampled post-cross topgap+24.940958mm at2920/front−3.040310mm stillAIR/0force, despite topgeometrytrue. These are decision-end extrema, not full120Hz extrema. The terminal has further retreated, so prior crossing is not current topXY/capture.

Terminalrawfinite/bodycollisionfalse, FL verifiedinactivepairs0N. FR verifiedobstacle normal12.345253479N/load.425757948; RL/RR ground14.913726807/1.736963391N/load.514338384/.059903668. Current supportFR/RL/RR count3, CoMinside margin+25.354172mm, not FLsupport. All5328native rows verify/effect/setter-mapping-counterfactual/reference checks, own5324/fourstatewrites0, sequence/clockerrors0; complete666×8 ledger/no finitefallback. No native execution anomaly is shown.

Detailed fixed-scope diagnosis: `eval_107264_diagnosis.md`. C103168/P05, C101376/P06, oldA and every prior result are retained. P06 was not visited; no causal new-workspace-reward improvement, full/suffix success, paired stability advantage or successfulvideo is claimed. The subsequent P06 training is not read/precredited. Existing report headers/history were not rewritten in this append-only update.

## Finalized block30 — same-MDP frozen-FSM P06 course, checkpoint108,288

This append finalizes only block30 and preserves the preceding29 blocks and C107,264 evaluation. Run `train/20260907T0253258293143Z_gf1a9bbf650b1_a6d3b0a9bb214f398e3579e3f350e7ba`, HEAD f1a9bbf650b1b8f80d5169e09173fcfc68797d99, N1/seed1001/P06/frozen_fsm/offset0, ended **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY / exit0**. The run-specific stop reallocates future sampling toward existing P07 preparation; it is not a fault, new-MDP change, success reclassification or optimizer-success gate. Source107,264/803/16,060 → **108,288/811/16,220**, actual1024/8/160, planned2048/unconsumed1024/rounding0, wall820.5672665999737s.

The30 finalized blocks add **98,176 policy decisions /767 PPO updates /15,340 optimizer steps** since origin10,112/44/880. Budget totals are full49,408/suffix48,768/smoke0;49,408+48,768+10,112=108,288. Ordinary same-runtime resume retains learned actor includingstd, critic, Adam/identitynormalizer/RNG through verified loading; no newAdam was requested. All8 source/adjacent/final actor fingerprints agree, all8 updates report changedactor/nonzero finitegradients and20steps, recorded update-end LR1e−5. Saved107392/107776/108288 sidecars have roundtriptrue, correct updateactor/runtime/source_run and unchangednormalizer. This report did not independently loadPT or rehashcheckpoint files.

Both actual frozen-teacher prefixes were accepted without fallback, each448dec/3584ticks/P06 at29.8666666667s with170.1333333333s remaining. **896teacher decisions/7168ticks are entirely excluded from PPO**; all prefix raw actions0, all ticksverified/policy_creditfalse/no-state-write evidencecomplete. Actual PPO is1024 P06 decisions/8185ticks; core total1920dec/15353ticks. Teacher P01–P05 and FR/FL Q/C/P are not current-policy phase credit.

Episode0 completes601PPOdec/4801ticks, g107265–107865, physicaltick8385/time69.875s/P06age40.0083333 INCOMPLETE_CONTROLLER_BLOCKED, validtrue/nullphysicalfailure. Its1tick terminal explains8185=1024×8−7; nextPhi0/bootstrapfalse. Episode1 is an **optimized423decision/3384tick nonterminaltail**, g107866–108288, physicaltick6968/time58.0666667s/P06age28.2/reasonnull. It is not a second failure. All1023 nonterminals retainbootstrap; time_outsfalse throughout. Both endpoints have hardrear_approach0, RR/RL noQ/C/P. RRinitial bit occurs at1/4 decision endpoints respectively, not qualification or lift-attempt counts.

FR Q71/C1665/P1695 and FL Q2461/C3115/P3583 remain teacher histories. Ep0 terminal FL AIR/load0/gap+.027621274m; FR/RR/RL GROUND loads.474690755/.019982036/.505327210, rear fronts−.616933395/−.457951647m. Ep1tail FL AIR/load0/gap+.061886956m; FR/RR/RL GROUND loads.401007866/.090074168/.508917967, rear fronts−.591817241/−.479690552m. FL support endpoints8 per episode, AIR593/415; historical placed does not mean currentTOPsupport. Sampled closest RR/RL fronts remain outsideworkspace (ep0−.443071593/−.412201140m; ep1−.443473619/−.450031270m), not asserted120Hz extrema.

All8185 credited native ticks verify/effect/own-requesteffect;1024 endpoint independent previousACKreference/clock checks pass, physicalvalidity/raw-request checks pass and fourstatewrite types0. Nominal wheels remain.3 at inspected endpoints while PPO residuals remainactive: terminal actualcanonical wheels `[.073469557,−.463073697,.096016382,.563654234]`; tail `[−.109038664,−.056877179,.198379665,.713864152]`. This is executed policy authority, not a causal explanation or a reason to mask actions.

Full final audit: [p06_frozen_block_108288.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p06_frozen_block_108288.md); the bounded [first128 report](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p06_frozen_first_107392.md) remains unchanged. Latest completed naturalP01 evaluation remains C107,264/P05 incomplete. No subsequent P07 data, unconsumedplan, full/suffix success, paired stability improvement or successfulvideo is precredited.

## Finalized block31 — same-MDP frozen-FSM P07 course, checkpoint108,800

This append finalizes only block31; prior blocks, bounded reports and C107,264 evaluation remain intact. Run `train/20260907T0313083674401Z_gf1a9bbf650b1_98167177c0ff4cd6b15dd57e491fa8c2`, same HEAD f1a9bbf650b1b8f80d5169e09173fcfc68797d99, N1/seed1001/P07/frozen_fsm/offset0, ended **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY / exit0**. Source108,288/811/16,220 → **108,800/815/16,300**; actual512/4/80, planned1024/unconsumed512/rounding0, wall1809.5637302000541s. The stop reallocates later sampling toward existing P10, not a fault, new-MDP revision, success gate or consumption of the remaining plan.

The31 finalized blocks add **98,688 decisions /771 PPO updates /15,420 optimizer steps** since origin10,112/44/880. Spent full49,408 remains unchanged, suffix48,768→49,280, smoke0;49,408+49,280+10,112=108,800. Current new_mdp_warm_start=false and same-runtime source-bound resume retain learned actor/std, critic, Adam, identity normalizer and RNG via the verified load path, with fresh rollout/legal physical reset. Older new-MDP ancestry in the sidecar is not a new Adam reset in block31. No independent PT load or repeated file hash was performed for this report.

All4 updates812–815 add128decisions/20optimizersteps, change actor, report nonzero finite gradients and maintain the full source/adjacent/final actor-fingerprint chain; recorded update-endLR1e−5. Actual saves **108416 and108800** both record roundtriptrue, correct update actor/runtime/source_run and unchanged normalizer. Final actor fingerprint `8e029ded1cf5155188ebea42331fdd177ca6c11eda8de757703448ec239bfdc3`; immutable sidecars, not a moving active-run pointer, supply these receipts.

Actual credited phases **P07=6/P08=6/P09=500**, all others0. Six accepted physical frozen-teacher prefixes each744dec/5952ticks/P07 at49.6s, no fallback, give **4464 teacher decisions/35712 ticks excluded from PPO**. Credited512dec/4082ticks plus teacher equals physical core4976dec/39794ticks. Every prefix raw action0/policy_creditfalse/no-state-write verificationtrue, all35712 native ticksverify; teacher P01–P06 and front Q/C/P are not current-policy credit.

Five completed episodes are **452/P09 INCOMPLETE**, then **11/11/11/12 decisions/P09 BODY_COLLISION**; the final **15-decision P09 tail is nonterminal and already optimized**, not a sixth failure. Exact credited ticks3616/82/83/87/94/120; short collision intervals2/3/7/6 explain4082=512×8−14. Episode0 endsg108740/tick9568/79.7333333s/P09age30, validtrue/nullphysicalfailure. Collisions endg108751/108762/108773/108785 at6034/6035/6039/6046, validtrue/TASK_FAILURE_BODY_COLLISION. Tailg108786–108800 ends6072/50.6s, reasonnull. All507 nonterminals bootstrap; five real terminals do not, time_outsfalse throughout. Full/suffix task success0.

Episode0 genuinely earns **RR Q6588/C6655 but noP** after teacher handoff; RL has no hardQ/C/P anywhere, and RR has none in episodes1–5. First missing ep0 task is current RR capture: terminalRR AIR/load0/front−150.050845mm/gap+115.399794mm; FL trulyTOP/load.274080278, RL GROUND/load.322329096. This is earned crossing followed by high unloaded retreat, not placement or ground-revoked qualification. Four collision endpointsRR AIR/unqualified and FL AIR/load0 retain the actual safety classification; compact records do not independently recover terminal raw pair/force/penetration. TailRR/RL areGROUND loads.059105413/.494532070, fronts−208.140191/−212.666061mm; FLAIR/load0. Teacher FRQ71/C1665/P1695 and FLQ2461/C3115/P3583 never imply ongoing support or new PPO credit.

All4082 credited native ticks verify/effect, own4070;512 decision-end audits verify, physicalvaliditytrue, fourstatewrite totals0/teacher-storageflags0/sequenceerrors0. See [p07_block_108800.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p07_block_108800.md), preserving the bounded [first16](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p07_first_handoffs_108304.md), [episode0](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p07_episode0_rear_chain_108740.md) and [ep1–2](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p07_early_body_collision_evidence.md) reports. No block32 data/planned credit, full-task success, paired improvement or successfulvideo is read or asserted. Latest completed naturalP01 evaluation recorded here remains C107,264/P05 incomplete.

## Finalized block32 — same-MDP frozen-FSM P10 course, checkpoint109,824

This append finalizes only block32, preserving all earlier evidence. Run `train/20260907T0351509044004Z_gf1a9bbf650b1_f18d36a5a1bd43de83a723a754e0bc78`, same HEAD f1a9bbf650b1b8f80d5169e09173fcfc68797d99, N1/seed1001/P10/frozen_fsm/offset0, finalized **SUCCEEDED execution /exit0**, not task success. Source108,800/815/16,300 → **109,824/823/16,460**; actual=planned=requested1024/8/160, unconsumed0/rounding0, wall1528.7083147999365s.

The32 finalized blocks add **99,712 decisions /779 updates /15,580 optimizer steps** since origin10,112/44/880. Budget full49,408 unchanged, suffix49,280→50,304, smoke0;49,408+50,304+10,112=109,824. Ordinary same-MDP resume retains actor/learnedstd, critic, Adam, identitynormalizer and RNG via the verified load path, no newAdam request, fresh rollout/legal reset. All8 updates816–823 add128/20, changeactor with finite nonzero gradients and an uninterrupted source/adjacent/final fingerprint chain; update-endLR≈1e−5. Actual saves108928/109312/109824 all roundtriptrue/updateactor/source/runtime/normmatch; finalactor `db277144127e0653082346a04ce45156586465db9b27187c5b7c3c8c30768b70`. No independent PT load or repeated hash.

Credited phases **P10=4/P11=104/P12=916**, all others0. Four accepted real prefixes each948dec/7584ticks/P10 at63.2s, no fallback, give **3792teacher decisions/30336ticks excluded from PPO**. Actual PPO1024dec/8171ticks, physical core4816dec/38507ticks. Teacher raw0/policy_creditfalse/no-state-write verificationtrue/nativeverified30336; precredit histories do not count as policy accomplishments.

Completed ep0/1 each453dec/3617ticks, g108801–109253 and109254–109706, P10/P11/P12=1/1/451, end11201/93.341666667s/P12age30.0083333 INCOMPLETE_CONTROLLER_BLOCKED, physicalvalidtrue/nullhardfailure/placed_RL0. Ep2 g109707–109818 is112dec/889ticks, phases1/99/12, end8473/70.608333333s/P12 HARD_JOINT_LIMIT. Ep3 g109819–109824 is **6dec/48ticks nonterminal P12 tail**, phases1/3/2, end7632/63.6s, not a fourth failure. Three1tick terminal intervals explain8171=1024×8−21. All1021 nonterminals bootstrap, three true terminals do not, time_outsfalse. No P13/full/suffix success.

Teacher FRQ71/C1665/P1695, FLQ2461/C3115/P3583 and RRQ6938/C7109/P7579 precede every creditstart. **RL has no hard Q/C/P anywhere in this block.** Ep0/1 terminalRL GROUND/front−289.307814/−344.773751mm and RR formerlyplaced nowGROUND/front−259.365198/−370.278370mm; FLAIR/load0. Missing RL capture and lost current RR platform support are real state facts, not new teacher successes. Ep2 FLTOP/load.729325546, RRAIR/load0, RLGROUND/load.025228574; tailFL/RR trueTOP while RL obstacle-supported but TOPfalse/noQCP. Historical placement is not current support.

Ep2 retained failure detail is **SAFETY_ABORT / `hard joint limit: front_left_knee`**, outward reasonHARD_JOINT_LIMIT. Both task evaluator and live guard inspect measured canonical joint position, not the logical action. HardFLknee[−60,210]°, existing2° reservedtarget[−58,208]°. Pre-dispatch actualphysicalq−1.0412137508392334rad, converted to degrees then minus standingoffset+.02684400080484552°, gives canonical **−59.683997494879° at the preceding state**. Same-tick logicalnominal−31.4 plusrequested−28.761027355832 is not finaldispatch: native−36.4/controller0/headroomeffective−21.6 yields previous/currentfinal **−58/−58°**, verifiedfloat32setter−1.0118224620819092rad. Therefore this is not evidence that finalcommand exceededhardlimits. Exact post-step offendingq/qdot/torque/overshoot are absent fromcompactterminal; precedingnear-lower-boundq does not substitute for them or establish unique cause. The safetyfailure remains true, event−40/Phi_after0/totalreward−43.3770424967/bootstrapfalse, not reclassified by training execution success.

All8171 credited native ticks verify/effect, own8163,1024 endpointauditsverify/physicalvalidtrue, fourstatewrites0/teacher-storageflags0/sequenceerrors0. Full bounded ledger and limit-evidence distinctions: [p10_block_109824.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p10_block_109824.md), retaining [first16](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p10_first_handoffs_108816.md) and [first128](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p10_first128_rl_authority.md). Active C109824 evaluation is not read/prefilled. Latest completed naturalP01 evaluation recorded here remains C107,264/P05 incomplete; no later credit, task success, paired improvement or successfulvideo is asserted.

## Completed C109,824 natural-P01 evaluation — FL capture still incomplete

This append replaces the preceding launch-only C109824 status with its actual completed evaluation, without changing any of the32 finalized training blocks or earlier failed/incomplete A/C evidence. Run `validation/20260907T0425049534146Z_gf1a9bbf650b1_c287e9492402483f946f83dd50cb40ca`, runtime f1a9bbf650b1, reloaded checkpoint109824, N1/seed2001/naturalP01/fixed deterministic mean, finalized **SUCCEEDED execution /taskfalse**. Actual **668 decisions /5344ticks /44.533333333s**, **P05 INCOMPLETE_CONTROLLER_BLOCKED**, P05 age30s; physicalvalidtrue/nullhardfailure, optimizerupdates0, windowEndedfalse.

Phase vector P01–P13 is **[1,213,3,1,450,0,0,0,0,0,0,0,0]**. P02/P03/P04/P05 entryticks8/1712/1736/1744 are genuine nonterminal handoffs/bootstraptrue/time_outsfalse. FRQ51/C1724/P1734; **FLQ1825/C2889 but noP**; RR/RL nohardQ/C/P. First missing task is FL actual capture, `placed_FL=.7`: terminalFL AIR/load0/front−69.295879771mm/gap+37.810539277mm, currenttopXY/geometryfalse/TOPcount0. All450P05 decision endpoints areAIR/obstacleactive0; final recordedAIR streak1741–5344 is3604ticks. Closest post-cross decision-end gap2936 is+34.568970mm atfront−19.341167mm/outsideXY; maximum sampledpostcrossfront2896 isonly+.981334mm whilegap+90.758955mm. Neither earliercross nor geometricproximity is actual placement.

Terminalraw isfinite and confirmsFL exactpairsinactive/0N; FRtrueTOP/load.420857335, RL/RRGROUND loads.505113612/.074029053, support3/valid androbotCoM polygonmargin+23.543145773mm. Bodycollisiondetected/active/persistentfalse; that is not an improvement claim. Existing receipts reconcile all5344nativeverified/effect ticks, own5340, fourstatewrites0, noN1clock/fallback discrepancy. Existing whole-window quality hasroll/pitchRMS .128124142/.103887734rad andangularaccelerationRMS4.320426690rad/s², but **fixedqualityscore=null/allphasessampledfalse**; noFSM superiority, pairedstability advantage orcausalgain is inferred.

See [eval_109824_diagnosis.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/eval_109824_diagnosis.md). Latest completed naturalP01 result is now C109824/P05incomplete. Training counts remain **109824/823/16460**, with32 finalized blocks adding99712/779/15580 since10112/44/880; full49408/suffix50304 unchanged. Any current return-profile production work is unexecuted in this historical run and receives no additional count, success orimprovement credit. C107264 and oldA results are retained unchanged; master append finished and writing stopped.

## Finalized block33 — new return-horizon natural P01, checkpoint113,920

This append finalizes only block33 and preserves all32 previous block ledgers, initial/first128 evidence and failed/incomplete A/C results. Run `train/20260907T0453293778943Z_gf4bfe2560bfd_8719fa76781746e09c31b9b78dc14df8`, HEAD f4bfe2560bfd228541f7829d830fa441054eab1d, N1/seed1001/naturalP01, finalized **SUCCEEDED execution /exit0**, not physical task success. Source109,824/823/16,460 → **113,920/855/17,100**; actual=requested=planned4096/32/640, unconsumed0/rounding0, recorded loopwall1717.9151240000501s.

The33 finalized blocks add **103,808 policy decisions /811 PPO updates /16,220 optimizer steps** since10112/44/880. Budget full49,408→53,504, suffix50,304 unchanged, smoke0/origin10112;53,504+50,304+10,112=113,920. This explicit new-MDP return-profile boundary uses gamma.9985/lambda.99/rollout128 while preserving learned hetero actor/std, critic, identitynormalizer, RNG and lifetime accounting through the already audited initial publication. It does not reset spent budgets or establish a physical improvement. Initial details are retained separately in [return_horizon_initial_109824.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/return_horizon_initial_109824.md).

Exactly4096 contiguous rows g109825–113920 reconcile to phase vector **[8,1124,27,10,1727,1200,0,0,0,0,0,0,0]**. All six starts are naturalP01 with nullprefixrequest and no teacher credit. Five completed episodes have **651/647/931/947/641 decisions**, respectively P05/P05/P06/P06/P05 INCOMPLETE_CONTROLLER_BLOCKED, physicalvalidtrue/nullhardfailure. Their global endpoints are110475/111122/112053/113000/113641. The sixth is an **optimized279-decision nonterminaltail** g113642–113920,2232ticks/18.6s/P05age5.333333333, phases2/192/4/1/80 inP01–05—not a sixth failure. No task success or P07–P13 sample exists.

FR Q/C/P is recorded in all six episodes. FL: ep0Q1677/noC/P; ep1Q1637/C2725/noP; ep2Q1535/C2428/P2643; ep3Q1752/C2707/P2774; ep4Q1624/C3300/noP; tailQ1679/noC/P. **RR/RL have no recorded hardQ/C/P throughout.** P05 missingtasks are FL crossing plus capture inep0, and actual capture afterQ+C inep1/4; all terminalFL areAIR/load0. Ep2/3 P06 begins2648/2776 but endsrear_approach0, RR/RL fronts−658.168/−485.279mm and−509.642/−361.791mm. Both have FL AIR/load0 despite historicalP; RR isAIR belowtop/unqualified, RL GROUND. No history is substituted for current support or complete task success.

All **32761 native ticks** verify/effect; own32735 plus26handoffholdticks reconcile. The sole1tick interval is ep1/g111122 P05 deadline, explaining4096×8−7. All26 phasehandoffs remainnonterminal/bootstraptrue;4091 nonterminalrowsbootstrap, five true terminals do not and havePhi_after0, time_outsfalse. Fourstatewritetotals0, nofallback, selected finite/native/rawbinding/clock checks0discrepancies. These are logged execution/mask checks, not PT/GAE replay or proof of greater stability.

All32 updates824–855 changeactor, reportfinite nonzero gradients and20optimizersteps, with source/31adjacent/final actor-chain checks matching. Allfive saves109952/110848/111872/112896/113920 recordroundtriptrue, correctupdateactor/source_run andunchangedidentitynormalizer. Finalpointer/sidecar identify113920/855/17100, recordedCP SHA `54bbc7de91a4d7a7461c52b2165abf6286127918522cc52606a96bb140499a6a`, finalactor `fb377ac16b8ec08d5592bbd7e70b3f766ca251abc9f1453aa72db0bf54cd1253`. No repeatedCP hash orPT load was performed.

Full bounded audit: [return_horizon_block_113920.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/return_horizon_block_113920.md). Latest completed fullP01 evaluation recorded here remains **C109824/P05 incomplete**. The separately launched C113920 evaluation and any nextcourse were not read or prefilled. Earlier A failures, no full/suffix success, no paired stability-improvement proof and no new successfulvideo remain unchanged. Block33 report/master append finished; writing stopped.

## Completed C113,920 natural-P01 evaluation — FL geometry reached, loaded capture absent

This append replaces only the launch-only C113920 evaluation status; block33 and all33 training ledgers remain unchanged. Run `validation/20260907T0523186087751Z_gf4bfe2560bfd_b5694e9e04f14a45aa8af9de9841fe5b`, runtime f4bfe2560bfd, checkpoint113920, N1/seed2001/naturalP01/fixed deterministic mean, finalized **SUCCEEDED execution /exit0 /taskfalse**. Actual **673 decisions /5384ticks /44.866666667s**, **P05 INCOMPLETE_CONTROLLER_BLOCKED**, age30s; physicalvalidtrue/nullhardfailure, optimizerupdates0, windowEndedfalse.

Phase vector P01–P13 is **[1,217,4,1,450,0,0,0,0,0,0,0,0]**. Four handoffs at8/1744/1776/1784 remain nonterminal/bootstraptrue/time_outsfalse. FRQ48/C1756/P1770; **FLQ1865/C2902 but noP**; RR/RL nohardQ/C/P. First missing task is genuine FL loaded capture: terminal `placed_FL=.85` is fractional progress, not placement. FL is **AIR/load0/supportfalse/front+1.866046mm/gap+21.597487mm**, with current topXY/topgeometry true but obstacle contact false/TOPcount0. Of450P05 endpoints,449AIR, trueTOP0, obstacleactive0,302topgeometry. Stored terminalAIRstreak3587 ticks reinforces the missing contact; Q/C and geometry do not fabricate support.

The verified terminal raw reproduces FL bottomz .071597487302m over .05m obstacle top and both exact FL pairs inactive/0N. FR trueTOP/load.419189504; RL/RR GROUND loads.485710921/.095099575. Support is FR/RL/RR, count3/valid, CoM polygon margin+24.824628605mm. Bodycollision detected/active/persistentfalse, penetration0. Native receipts reconcile5384verified/effect ticks, own5380, fourstatewrites0, no episodeclock gap/physical-invalid snapshot/finitefallback. First/last native clocks187/5563 correspond to episode8/5384 with the same179-tick offset and immediately preceding ACKs. These are existing receipt checks, not independent PT/native tensor replay.

The true terminal has absorbingPhi0/bootstrapfalse, PBRS−2.0588640316 and event−40; whole eval return−43.3083633787. Quality roll/pitchRMS .132057458/.100492381rad, angularaccelerationRMS4.051394856rad/s²; **fixedqualityscore=null/allphasessampledfalse**. This incomplete, non-paired observation is not a gamma-causal or stable-superiority claim. Compared with historical C109824, FL now ends geometrically over the platform but still unloaded; neither result is complete task success.

Full bounded evidence: [eval_113920_diagnosis.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/eval_113920_diagnosis.md). Latest completed naturalP01 result is now **C113920/P05 incomplete**. Training counts stay113920/855/17100, with33 finalized blocks adding103808/811/16220 since10112/44/880; full53504/suffix50304 unchanged. No new training, future credit, successfulvideo or revised gate is read or asserted. Report/master append completed; writing stopped and master ownership released.

## Finalized block34 — same-MDP frozen-FSM P06 offset160, checkpoint115,968

This append finalizes only block34, preserving all33 earlier ledgers, historical A/C outcomes and the first16/handoff report. Run `train/20260907T0531193843603Z_gf4bfe2560bfd_4b60e01ba2614a4a90ad4c6a7c06a5e1`, same HEAD f4bfe2560bfd228541f7829d830fa441054eab1d, N1/seed1001/P06/frozen_fsm/offset160, finalized **SUCCEEDED execution /exit0**, not task success. Source113920/855/17100 → **115968/871/17420**; actual=requested=planned2048/16/320, unused0/rounding0. Recorded training-loopwall1824.8636562000029s; roll-in/reset telemetry1231.783564399695/23.788317599799484s is kept in its own scope, not subtracted to invent exclusive policy time.

The34 finalized blocks add **105856 decisions /827 PPO updates /16540 optimizer steps** since10112/44/880. Full53504 unchanged, suffix50304→52352, smoke0/origin10112;53504+52352+10112=115968. This is ordinary same-MDP continuation, retaining actor/learnedstd, critic, Adam/identitynormalizer/RNG via verified source loading—not a fresh-Adam or return-profile revision. Gamma.9985/lambda.99/128 remain recorded. All16 updates856–871 changeactor with finite nonzero gradients and20steps; source/15adjacent/final fingerprints match. Five saves114048/114432/114944/115456/115968 recordroundtriptrue/updateactor/source_run/normalizer consistency. FinalCP/sidecar pointer SHA receipts are `a4c6320ed34755b97b4c6f06a9e5a78eda76c3a1e6350cef4b0f126eb8ed145f` / `188e398ec6c227673edb07be0ade8781292cc195d861dd5df97b1c9d72cff84f`; no repeatedfilehash or PT load.

All2048 contiguous policy rows g113921–115968 are **P06**, all otherphases0. Five actual prefixes each608dec/4864ticks are accepted withoutfallback, targetP06first3584+160×8offset, handoff4864/40.533333333s/remaining159.466666667s. **3040teacher decisions/24320ticks are excluded from PPO**; all raw/projectedpolicyresidual0, policy_creditfalse/nativeverified/no-state-write verificationtrue. Physical core5088dec/40676ticks equals prefix plus2048PPOdec/16356ticks. Each handoff RR/RL isGROUND/nohardQCP and FLtrueTOP/load.192344755. Teacher FRQ71/C1665/P1695 and FLQ2461/C3115/P3583 remain exactly preserved, not new policy achievements.

Four true completedepisodes each **441dec/3521ticks** endg114361/114802/115243/115684 atphysicaltick8385/69.875s/P06age40.008333333 INCOMPLETE_CONTROLLER_BLOCKED, physicalvalidtrue/nullhardfailure. Each finalinterval is1tick, explaining16356=2048×8−28. The final **284dec/2272tick optimized nonterminaltail** g115685–115968 endsP06/tick7136/59.466666667s/age29.6/reasonnull, not a fifth failure. All2044nonterminalsbootstrap, fourterminalsPhi_after0/no bootstrap, time_outsfalse. All16356native ticksverified/effect/own, fourstatewrites0, no finitefallback/native/clock/rawbinding discrepancy; no creditedphasechange or teacher-storageflag occurred.

No current-policy RR/RL hardQ/C/P is recorded. P06 firstmissing remains real rear approach/workspace: endrear_approach .241408515/.320172924/.528050799/.586994011/.604297656 acrossfourterminals+tail, not completion. EndRR/RL fronts(mm) are−409.648/−333.519;−389.957/−311.072;−337.987/−257.438;−323.251/−240.891;tail−318.926/−253.851. Allfive endFL areAIR/load0 despite teacherplacement; acrosspolicyendpoints FLTOP61/AIR1987. RR episode0endsAIR belowtop/unqualified, othersGROUND; RLalwaysGROUND attheseends, FRtrueTOP. Neither partialprogress nor lowAIR is promoted toqualification orsuccess, and no unique failurecause is inferred.

See [p06_offset160_block_115968.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p06_offset160_block_115968.md), preserving [p06_offset160_first_113936.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p06_offset160_first_113936.md). Latest completed naturalP01 evaluation remains **C113920/P05 incomplete**. No new P07 run was read or credited, no full/suffix success, paired improvement or successfulvideo is asserted. This final report/master append is complete and writing stopped.

## Finalized block35 — same-MDP frozen-FSM P07, checkpoint116,992

This append preserves all34 earlier ledgers and historical A/C outcomes. Run `train/20260907T0607442246056Z_gf4bfe2560bfd_0be30949b7cb42deb69981fa628ac0fc`, HEAD f4bfe2560bfd228541f7829d830fa441054eab1d, N1/seed1001/P07/frozen_fsm/offset0, finalized **SUCCEEDED execution / exit0**, not task success. Source115968/871/17420 → **116992/879/17580**; actual=requested=planned1024/8/160, unused0/rounding0. Recorded training-loop wall1351.645459299907s; roll-in1231.6701696000528s and reset22.60671910014935s retain their own scope and are not subtracted to invent exclusive policy time.

The35 finalized blocks add **106880 decisions /835 PPO updates /16700 optimizer steps** since10112/44/880. Full53504 unchanged, suffix52352→53376, smoke0/origin10112;53504+53376+10112=116992. This is ordinary same-MDP continuation, not a fresh-Adam or return-profile migration. Gamma.9985/lambda.99/128 remain recorded. All8 updates872–879 have changed actor fingerprints, finite nonzero gradients and20 optimizer steps; source/seven-adjacent/final actor links are exact. Three immutable saves116096/116480/116992 record roundtriptrue, source_run/actor/update/normalizer consistency. Final pointer CP/sidecar SHA receipts are `7d1beab38b91640a361d5537d1b7e26a845def233dca7775c9de2a7693133b11` / `db241f81977c01f0830bbf88adf92a3463055533b01b0b69aed623c0269ba11f`; no repeated filehash or PT load was performed.

All1024 contiguous policy rows g115969–116992 have P01–P13 counts **[0,0,0,0,0,0,4,5,1015,0,0,0,0]**. Four accepted measured prefixes each744dec/5952ticks reach P07 tick5952/49.6s, remaining150.4s, RR/RL GROUND/nohardQCP, FL/FR exact obstacle contact. **2976 teacher decisions/23808 ticks are excluded**; all teacher raw/projected residual0, creditfalse, nativeverified/no-state-write verificationtrue. Teacher FR Q71/C1665/P1695 and FL Q2461/C3115/P3583 remain unchanged and are not PPO achievements. Core4000dec/32000ticks equals prefix plus1024PPOdec/8192ticks.

There are **three completed episodes**: ep0 g115969–115981=13dec/104ticks/P09 BODY_COLLISION at50.466666667s; ep1 g115982–116434=453dec/3624ticks/P09 incomplete at79.8s; ep2 g116435–116886=452dec/3616ticks/P09 incomplete at79.733333333s. The two incomplete episodes have valid physical evidence and null hard-failure reason; the collision retains TASK_FAILURE_BODY_COLLISION, not a software-execution error. The **106dec/848tick optimized nonterminaltail** g116887–116992 ends P09 tick6800/56.666666667s and is not a fourth failure or success. All8192 native ticks are verified/effect,8184 own-phase +8 handoff-hold; four state-write counts0, no clock/binding/finite-fallback mismatch. All1021 nonterminals bootstrap, all3 terminals have Phi_after0/no bootstrap, and all8 ordinary phase handoffs remain nonterminal.

RR has true current-policy Q6889 in ep1 and Q6702 in ep2, but **no RR crossing or placement**. Closest decision-end front distances are−25.133/−12.074mm, while highest AIR clearances are+94.249/+115.679mm; endpoint fronts retreat to−154.281/−126.578mm with RR AIR/load0. Thus the first unmet rear task is crossing/capture, not simply failure to lift. Tail RR Q6442 is revoked on GROUND6545 and requalified6693, still no C/P. Ep0 has no RRQCP; RL has no hardQCP anywhere. FL ends ep0 AIR/load0, ep1/ep2/tail TOP with loads.034164/.170602/.169048, but history is not uninterrupted current support. No P10–P13 policy samples, full/suffix success, paired improvement or successfulvideo is claimed.

See [p07_block_116992.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p07_block_116992.md). Latest completed naturalP01 evaluation remains **C113920/P05 incomplete**. No later run/evaluation was inspected or credited. This append is final and report writing has stopped.

## Finalized block36 — same-MDP frozen-FSM P10, checkpoint118,016

This append preserves all35 previous ledgers and historical A/C outcomes. Run `train/20260907T0639140541486Z_gf4bfe2560bfd_d832dad59b2f4187bceccc6c054be076`, same HEAD f4bfe2560bfd228541f7829d830fa441054eab1d, N1/seed1001/P10/frozen_fsm/offset0, finalized **SUCCEEDED execution / exit0**, not task success. Source116992/879/17580 → **118016/887/17740**; actual=requested=planned1024/8/160, unused0/rounding0. Training-loop wall1204.9456783998758s. Roll-in1178.596724000061s/reset19.640224100090563s retain their distinct timing scopes, not exclusive-policy throughput.

Across36 finalized blocks the additions since10112/44/880 are **107904 decisions /843 PPO updates /16860 optimizer steps**. Full53504 unchanged, suffix53376→54400, origin10112/smoke0;53504+54400+10112=118016. This is ordinary same-MDP continuation, not fresh-Adam or a return-profile revision. All8 updates880–887 have changed actors, finite nonzero gradients and20 optimizer steps, exact source/seven-adjacent/final fingerprint links; recorded update-end LR1e-5. Saves117120/117504/118016 each record roundtriptrue, actor/update/source-run/normalizer consistency. Latest pointer CP/sidecar SHA receipts are `57f8c5e70820235fdb558711f639222009fea2021a2490be0256bb707a42b091` / `61af30bb8930697952713cc15aa25a5d98dddb1fac8c04cc26bb547ecb6b7163`, without repeated hashing or PT loads.

All1024 contiguous policy rows g116993–118016 have phase counts **[0,0,0,0,0,0,0,0,0,3,29,992,0]**. Ep0 g116993–117445=453dec/3617ticks and ep1 g117446–117917=472dec/3776ticks are **P12 INCOMPLETE_CONTROLLER_BLOCKED**, at93.341666667/94.666666667s inclprefix, physicalvalidtrue/nullhardfailure. The **99dec/792tick nonterminaltail** g117918–118016 endsP12/tick8376/69.8s, not a third failure. Only g117445 has1tick, so native8185=1024×8−7; all8185verified/effect,8179own+6handoff-hold, fourstatewrites0 and no inspected clock/binding/finite-fallback discrepancy. All1022nonterminalsbootstrap, two terminalsPhi_after0/no bootstrap, timeoutflagsfalse; all6 ordinary phase handoffs remain nonterminal.

Three accepted prefixes each948dec/7584ticks reach actualP10/63.2s, remaining136.8s. **2844 teacher decisions/22752ticks are excluded from PPO**, with raw/projected0, creditfalse/nativeverified/no-state-write verificationtrue. Core3868dec/30937ticks conserves teacher plus policy. FR Q71/C1665/P1695, FL Q2461/C3115/P3583 and RR Q6938/C7109/P7579 are teacher achievements, not new PPO credit. At handoff RR/FL are trueTOP but FR AIR; RL GROUND/nohardQCP.

No new credited RR/RL hardQ/C/P appears. **RL has not obtained above-top qualification/crossing/capture**; highest decision-end AIR clearances remain−37.073/−45.145/−47.070mm across two episodes+tail. End RL fronts are−110.731/−257.357/−93.873mm, allGROUND. RR endsGROUND despite historicalteacherplacement; FL endsAIR/load0 despite historicalplacement, FRtrueTOP. Partial placed_RL .15622/0/.18572 is not hard completion. No P13 sample, body-collision failure, full/suffix success or paired improvement is asserted.

See [p10_block_118016.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p10_block_118016.md), preserving [p10_first128_117120.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/p10_first128_117120.md). Latest completed naturalP01 evaluation remains **C113920/P05 incomplete**. No following evaluation was read or credited. Final report/master append complete; writing stopped.

## Completed C118,016 natural-P01 evaluation — FL capture remains incomplete

This append supplies the actual finalized C118016 evaluation only; every preceding training ledger and historical result remains unchanged. Run `validation/20260907T0707251670686Z_gf4bfe2560bfd_b8a1a4c5cb244de4951774a4483803e5`, runtime f4bfe2560bfd, saved118016, N1/seed2001/naturalP01/fixed deterministic mean, ended **SUCCEEDED execution /exit0 /taskfalse**. Actual **669 decisions /5352ticks /44.6s**, **P05 INCOMPLETE_CONTROLLER_BLOCKED**, age30s; physicalvalidtrue/nullhardfailure, optimizerupdates0, windowEndedfalse.

Phase vector P01–P13 **[1,213,4,1,450,0,0,0,0,0,0,0,0]**; four nonterminal handoffs enter P02/P03/P04/P05 at8/1712/1744/1752. FRQ48/C1723/P1740, **FLQ1843/C2869/noP**, RR/RL nohardQ/C/P. First missing task is FL actual placement: terminalFL AIR/load0/supportfalse/front+6.726686mm/gap+31.968420mm, withinXYtrue but topgeometryfalse/TOPcount0; `placed_FL=.7` is not completion. P05 has447AIR/3GROUND endpoints,0 obstacle/TOP/topgeometry; closest sampled post-cross gap+25.384521mm is stillAIR/load0. This is not a sensor-bug or threshold-change conclusion.

Finite terminalraw verifies wheel bottom .081968419822m above .05m top and both exact FL pairs inactive/0N. FRtrueTOP/load.418400197, RL/RRGROUND loads.507382481/.074217323; three valid supports excludeFL, CoM polygon margin+19.472737008mm. Bodycollision detected/active/persistentfalse, penetration0. Existing native receipts reconcile5352verified/effect,5348own, all endpoint verification checks passed, fourstatewrites0/no clock discrepancy/finitefallback. All668 nonterminals bootstrap; only final deadline has absorbingPhi0, PBRS−2.0495000124/event−40/no bootstrap. Whole eval return−43.2976834164.

Whole-window roll/pitchRMS .128848634/.097607314rad and angularaccelerationRMS4.092257956rad/s² are incomplete-window diagnostics: **fixedqualityscore=null/allphasessampledfalse**, P06–P13 unsampled. No causal explanation, stable superiority, full/suffix success or successfulvideo is asserted. See [eval_118016_diagnosis.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/eval_118016_diagnosis.md). Latest completed naturalP01 evaluation is now **C118016/P05 incomplete**; all36 training ledgers and their totals are untouched. No new training or proposed code revision receives credit here. Report/master append complete; writing stopped and master ownership released.

## #37 finalized history-kernel P06/offset160 — 119040, no task success

Run `train/20260907T0736098630371Z_ge8462f066908_a07ce6b30c64439e8e8735a0926c4cc6`, He8462f066908, N1/seed1001/frozen-FSM P06 offset160, has final `SUCCEEDED` training manifest and root-confirmed exit0. Source118016/887/17740 → **119040/895/17900**, actual **1024 policy decisions /8 PPO updates /160 optimizer steps**, planned1024/unconsumed0/rounding0, wall677.5841695999261s. This is the explicit `history_conditioned_heteroscedastic_log_v1` conditional-policy boundary: learned weights/std/critic/identity norm/RNG and budgets preserved, fresh Adam initially3e-5/empty rollout; physical/reward configuration unchanged. Initial details remain in [history_kernel_initial_118016.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/history_kernel_initial_118016.md).

Across37 finalized blocks, actual additions since10112/44/880 are **108928 decisions /851 PPO updates /17020 optimizer steps**. Full53504 unchanged, suffix54400→55424, origin10112/smoke0;53504+55424+10112=119040. Eight update records888–895 each contain20 optimizer steps, changed actor and finite nonzero gradients, with exact source/adjacent/final fingerprint continuity. Update-end LR is1e-5. Actual regular saves118144/118528/119040 all have roundtriptrue and match their update actor/source-run; final sidecar records119040/895/17900 and the history policy. These are recorded save/load receipts and JSON comparisons, not an independent PT load or repeated hash here.

Contiguous credit g118017–119040 has P01–P13 phase counts **[0,0,0,0,0,323,2,2,697,0,0,0,0]**. Exactly one completed episode: **578 decisions /4624ticks**, g118017–118594, P06/P07/P08/P09=126/1/1/450, **P09 INCOMPLETE_CONTROLLER_BLOCKED** at79.0666666667s/taskage30, physicalvalidtrue/nullphysicalfailure. The following **446 decisions /3568ticks** g118595–119040 are a **nonterminal tail**, P06/P07/P08/P09=197/1/1/247; it endsP09/tick8432/time70.2666666667s/age16.4666666667s and is not a second failure. No P10–P13 PPO coverage or full/suffix success is recorded.

Two accepted live frozen prefixes each608dec/4864ticks reach actualP06 at40.5333333333s with159.4666666667s remaining. All **1216 teacher decisions /9728ticks** are excluded from PPO; raw/projected0, creditfalse/nativeverified/no-state-write verificationtrue. Physical core2240dec/17920ticks conserves teacher+PPO. Front FRQ71/C1665/P1695 and FLQ2461/C3115/P3583 are teacher history, not new PPO events. First credited ep0 FL isTOP/supporttrue, but first credited ep1 FL isAIR/supportfalse despite historicalplacement; endpoint contact is not the exact pre-action handoff sample.

All **8192 credited native ticks verified/effect**, **8186 own-phase effects +6 handoff holds**; fourstatewrites0, no inspected clock/summary mismatch or finitefallback. All six ordinary P06→07→08→09 transitions are nonterminal/bootstraptrue. All1023 nonterminals permit bootstrap, while the one true terminal has absorbingPhi0/event−40/no bootstrap. Adjacent credited Phi is continuous within each episode, excluding legitimate reset boundaries.

Ep0 RR has initial-clearance attempts but nohardQ/C/P; RL briefly Q7766 then GROUNDrevoked7825/noC/P. In the **nonterminal tail**, RL Q6150 is revoked6243; **RR newly earns Q6917 and C7576 after creditstart4864, but noP**. Tail末RR isGROUND/front−83.052mm/clear−47.649mm/load.140340 despite retained historicalQ/C; RL isGROUND/front−165.762mm/load.427162/noC/P, FL isAIR/load0/no support despite its historicalplacement. First unfinished remains RR capture; recorded crossing is not stable placement or successfultask. No causal improvement or new rule is asserted.

See [history_p06_block_119040.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/history_p06_block_119040.md) for fixed-window native/events/reward detail. Latest completed naturalP01 result in this append remains **C118016/P05 incomplete**; no following evaluation was read or credited. All prior master paragraphs are preserved. Report/master append complete; writing stopped and master ownership released.

## Completed C119,040 natural-P01 evaluation — crossed FL never captured

This append updates only the latest completed full evaluation; all37 finalized training blocks, totals119040/895/17900, and historical A/C outcomes remain unchanged. Run `validation/20260907T0752476310693Z_ge8462f066908_f4de72f296d54255bccbc7bd5b92c2ac`, runtimee8462f066908, saved119040, N1/seed2001/naturalP01/history-conditioned deterministic conditional mean, finalized **SUCCEEDED execution /root-confirmed exit0 /taskfalse**. Actual **658 decisions /5264ticks /43.8666666667s**, **P05 INCOMPLETE_CONTROLLER_BLOCKED**, age30s; physicalvalidtrue/nullhardfailure, optimizer0, windowEndedfalse.

Phase vector **[1,203,3,1,450,0,0,0,0,0,0,0,0]**. Four forward handoffs enterP02/P03/P04/P05 at8/1632/1656/1664, allnonterminal/bootstraptrue. FRQ60/C1642/P1654; **FLQ1750/C2806/noP**, RL/RRnohardQCP. First missing is realFLplatformcapture: endFLAIR/load0/supportfalse/front−43.662306mm/gap+54.577028mm, outsideXY/TOPgeometryfalse/TOPcount0; partialplaced_FL .7 is not success. CompleteP05 raw3600ticks contains11GROUND+3589continuousAIR(t1676–5264),0FLobstaclecontact/0topgeometry. Atactualcross2806front+.356380mm/gap+94.134619mm; furthestfront+2.146367mm@2813, closestpostcrossrawgap+31.452109mm@2855alreadyfront−18.278834mm. Crossinghistory does not imply latercapture/current support.

All5264nativeverified/effect; decision summaries5260own+4handoffholds, fourstatewrites0; all5265rawfinite/clockcontinuous, bodydetected/realpair/persistent0. FRcurrentTOP/load.423923333, RL/RRGROUND loads.488419982/.087656685; valid3-supportpolygon excludesFL, CoM margin+28.431977mm. All657nonterminalsbootstrap, one realdeadlineabsorbingPhi0/no bootstrap, time_outsfalse. TerminalPBRS−2.026310057585334/event−40/total−42.028362451227565; completeevalreturn−43.24933885826473.

Incomplete-window roll/pitchRMS .120451125/.095246567rad and angularaccelerationRMS4.438567835rad/s² are diagnostics only; fixedqualityscore=null/allphasessampledfalse. No execution anomaly, full/suffix success, paired improvement or successfulvideo is claimed. See [eval_119040_diagnosis.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/eval_119040_diagnosis.md). Latest completed naturalP01 evaluation is now **C119040/P05 incomplete**; new same-MDP naturalP01 training is not read or credited here. Report/master append final, writing stopped and master released.

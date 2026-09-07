# P07 continuous suffix — final verified-update boundary 76,416

Status: **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY**; eight task failures/incompletes and an optimized nonterminal tail. No suffix/full-task success. This is a finalized partial training budget, not an optimizer failure or completion of the original request.

## Final execution and accounting

Run `runs/ppo_semantic_v3/train/20260906T1805562366558Z_gc34262abffc1_6e47bada5505418e811416a50bd871f5`, unchanged runtime `c34262abffc16847ff32d15ecbf790dd60803e0a`, N1/seed1001/P07/offset0/phase_suffix, exact-resumes source checkpoint73,088. Final `run_manifest.json` and `training_manifest.json` agree with the complete policy stream:

- Original planned request4,096; actual/finalized requested3,328; unconsumed768; rounding_overrun0.
- Actual26 PPO updates/520 optimizer steps; source73,088/536/10,720 → **76,416/562/11,240**. Recorded wall3367.614808900049s.
- Final spent ledger: full_episode29,184/phase_suffix37,120/smoke0; task origin10,112 remains unchanged. The source checkpoint's actual ledger was29,184/33,792; inherited `source_stage_requested_decisions` is older provenance, not the current starting budget.
- `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000076416.pt`, recorded checkpoint SHA `8273a81e70e08109c4ae5249034a767ec5d43c17e7c60b0fbb3b1c864971a697`, sidecar `save_load_round_trip=true`. Root independently matched checkpoint and sidecar actual hashes to their pointer; no duplicate hashing was performed for this report.

Policy-request phase counts P01–P13 are **[0,0,0,0,0,0,9,10,3309,0,0,0,0]**, summing3,328. All globals73,089–76,416 are contiguous, with no extra collected row after the finalized boundary.

## Episode ledger and actual rear-leg evidence

Q/C/P below mean the evaluator's actual qualified lift/front crossing/placement events, not initial-clearance hints. Times/ticks are the continuous physical episode clock, including the reset-only teacher prefix.

| Episode | Policy globals | Decisions / policy ticks | Last physical tick / seconds | Outcome | RR Q / C / P |
|---|---|---:|---|---|---|
|0 |73,089–73,166 |78 /622 |6,574 /54.783333 |P09 BODY_COLLISION |6,374 /none /none |
|1 |73,167–73,618 |452 /3,616 |9,568 /79.733333 |P09 incomplete |6,395 /none /none |
|2 |73,619–74,070 |452 /3,616 |9,568 /79.733333 |P09 incomplete |6,386 /none /none |
|3 |74,071–74,522 |452 /3,616 |9,568 /79.733333 |P09 incomplete |6,063 /6,173 /none |
|4 |74,523–74,975 |453 /3,624 |9,576 /79.8 |P09 incomplete |6,051 /6,158 /none |
|5 |74,976–75,427 |452 /3,616 |9,568 /79.733333 |P09 incomplete |6,103 /6,187 /none |
|6 |75,428–75,879 |452 /3,616 |9,568 /79.733333 |P09 incomplete |6,101 /6,180 /none |
|7 |75,880–76,331 |452 /3,616 |9,568 /79.733333 |P09 incomplete |6,049 /6,165 /none |
|8, nonterminal tail |76,332–76,416 |85 /680 |6,632 /55.266667 |P09; no termination |6,498 /none /none |

Eight completed episodes consume3,243 decisions; the final85 are already optimized but are neither a ninth failure nor a success. Seven incomplete reasons remain `INCOMPLETE_CONTROLLER_BLOCKED`; the first collision is not reclassified. Five episodes cross RR's front edge, but **none places RR**. RL has no Q/C/P in any of these policy segments. Front FR/FL Q/C/P are teacher history and cannot be credited as learned by this suffix. Episode1's final current RR qualification is false despite its historical Q event; historical qualification is not current support.

At the tail boundary, P09 age5.533333s has unfinished `placed_RR=0.623742680254632`. RR is currently AIR, front−43.575611mm/clearance+29.762970mm/load0, qualified but not crossed/placed; RL is GROUND, front−266.063895mm/load0.476491247. FL and FR are current TOP with loads0.063642849/0.459865905. These are boundary measurements, not a success projection. No later P10–P13 policy samples exist.

## Teacher exclusion, native targets and continuity

Nine accepted reset-only prefixes each execute744 decisions/5,952 ticks: **6,696 teacher decisions/53,568 ticks**. The prefix stream contains6,696 decision rows plus nine start/result/credit-start records each. Every teacher decision has raw and projected residual all zero, `policy_credit=false`, native audit verified, and no in-episode state writes. FR Q71/C1,665/P1,695 and FL Q2,461/C3,115/P3,583 precede the actual P07 takeover at5,952; these are preparation, not PPO credit.

The complete core therefore contains10,024 decisions/80,190 ticks, whereas PPO contains only3,328 decisions/**26,622 ticks**. The sole partial action is episode0/global73,166 with6 ticks, explaining26,622=3,328×8−2 exactly. All26,622 per-tick records independently match their decision summaries and are verified/actual-native-effect true. Own-phase effect26,604 excludes18 actual incoming handoff-hold ticks. All policy rows report the four in-episode root-pose/root-velocity/force-or-impulse/gravity write counts zero, verified no writes, teacher data absent from PPO storage, and `task_result_scope=teacher_initialized_suffix`.

All **18 ordinary P07→P08/P08→P09 decision transitions** remain nonterminal, `time_outs=false`, `terminal_bootstrap_allowed=true`; there is no phase-boundary terminal/GAE mask. Code corroboration: `semantic_env.py:196,226–236` derives done solely from a task termination reason and returns external truncation false; `semantic_training.py:177–205` rejects external truncation and resets only on done; `:723–729` passes the actual dones to official PPO and computes returns after each full128-decision rollout. This verifies the stored audit flags plus their producer path, not an independent numerical reread of tensor GAE values. Ordinary PPO rollout boundaries are retained; they are not phase-terminal resets.

The26 optimizer records537–562 each have20 steps, changed actor parameters and finite nonzero gradients. All global increments are128, the25 adjacent actor hash links match, first before-hash equals source actor5031de52…4589, and final after-hash equals saved actor0f4edbbd…7494. Runner configuration retains5 epochs×4 minibatches. Every recorded update-end learning rate is1e-5; no claim is made about unlogged per-minibatch rates. Rollout artifacts are preserved and were not loaded with Python in this audit.

## Stop rationale and attribution boundary

The existing `stop_after_update.request.json` asked to finish and optimize the current complete on-policy rollout, verify its save, then stop. `post_mapper_servo_headroom_readonly.md` supplies the fixed episode3 evidence:184 strict identity-geometry boundaries exhibit logical-nominal safety preclipping even though the same-tick mapped target retains physical headroom. The terminal episode3 row itself is **slew recovery, not saturation**, and remains described that way. This evidence motivates a separately versioned servo-headroom correction; it does not establish that the preclip caused every task outcome, relax actual hard limits, or erase failed data.

Earlier `p07_continuous_initial_128.md`, `p07_continuous_first_episode_diagnosis.md`, `p07_completed_episodes_1_2_diagnosis.md` and the headroom report remain intact. The front-wheel1.2 candidate and GPU-copy performance candidate are unpublished/unwired, and their CPU experiments are not training or dynamic feasibility evidence. At this report boundary, freshly reloaded C76,416 natural-P01 evaluation is running; **latest completed full-P01 remains C73,088/P05 incomplete**. No future evaluation result, success or video is prefilled.

Subsequent finalized evaluation addendum: C76,416 has now completed648decisions/5,184ticks/43.2s,P05 incomplete,physicalvalidtrue/null physicalfailure,0optimizerupdates. `eval_76416_diagnosis.md` preserves the independent actual FL Q/C/no-contact/no-placement evidence. The initial evaluation-running time slice above is historical; latest completed evaluation is now C76,416,with no additional training credit.

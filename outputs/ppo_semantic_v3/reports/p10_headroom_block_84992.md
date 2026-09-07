# Final block23 — P10 same-MDP continuation, checkpoint84,992

Final lifecycle: **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY**, root confirmed process exit0/CLOSED and final pointer/checkpoint/sidecar hash agreement and save/load round trip. This finalizes the actual diagnostic block, not the originally planned4,096 decisions and not a task success. No new evaluation result is assumed.

Run: `runs/ppo_semantic_v3/train/20260906T2046002070189Z_ge99fde1b3e83_aa661e98d21b4ca484288779f0a064eb`.

## Actual budget and save boundary

- Unchanged runtime `e99fde1b3e8366f0ff1d484140b82877df745f0c`, semantic v3, heteroscedastic actor, N1/seed1001/P10 offset0, phase_suffix. Source is immutable82,560/610/12,200. Arguments have new_mdp_warm_start=false, policy_distribution_migration=false and resume_migration=null; source/final runtime-contract JSON compares equal.
- Actual **2,432 policy decisions /19 PPO updates /380 optimizer steps**, final **84,992 /629 /12,580**. Finalized requested_policy_decisions=2,432; planned_requested_policy_decisions=4,096; unconsumed=1,664; rounding_overrun=0.
- Final stage spending is **full_episode33,280 /phase_suffix41,600 /smoke0**, origin10,112. The actual immediate source sidecar is33,280/39,168, so only2,432 suffix credit is added. The retained `source_stage_requested_decisions` ancestry field29,184/37,120 is an older new-MDP boundary record, not the immediate resumed checkpoint's current spending; it is not used as the block23 start ledger.
- Final `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000084992.pt`, recorded SHA `ebabc03d1d3611bbe4ab1c05feddbbe3102eedb71ae9d5924cd8f77f2826ea01`, sidecar `save_load_round_trip=true`. Root did the real file-hash verification; this read-only report did not repeat it or load tensors.
- The existing stop request explicitly asks for the next complete on-policy update, preservation of unspent budget and a reloaded natural-P01 evaluation before choosing another change. Reason: rebalance curriculum after repeated P12 incompletes. No mid-rollout process kill, discarded failed episode, failed-A prerequisite, manual-probe success gate or optimizer-error classification is introduced.

## One final policy-ledger pass

All2,432 JSON policy rows are contiguous globals82,561–84,992 with no gap/extra row. Manifest and stream phase counts match:

| Policy-request phase | Actual decisions |
|---|---:|
| P01–P09 |0 each |
| P10 |5 |
| P11 |5 |
| P12 |1,487 |
| P13 |935 |
| Total |2,432 |

| Episode | Policy global range | Decisions /policy ticks | Endpoint physical tick /seconds including prefix | Outcome |
|---|---|---:|---|---|
|0 |82,561–83,530 |970 /7,760 |15,344 /127.866667 |P13 INCOMPLETE_CONTROLLER_BLOCKED |
|1 |83,531–83,983 |453 /3,617 |11,201 /93.341667 |P12 INCOMPLETE_CONTROLLER_BLOCKED |
|2 |83,984–84,436 |453 /3,617 |11,201 /93.341667 |P12 INCOMPLETE_CONTROLLER_BLOCKED |
|3 |84,437–84,889 |453 /3,617 |11,201 /93.341667 |P12 INCOMPLETE_CONTROLLER_BLOCKED |
|4 /unfinished tail |84,890–84,992 |103 /824 |8,408 /70.066667 |P13, termination=null |

Four completed episodes sum2,329 decisions;103 already-optimized tail decisions reconcile to2,432. The tail is not a fifth failure or a success. It contains P10=1/P11=1/P12=66/P13=35; episode0 contains1/1/68/900, and each other completed episode1/1/451/0. All four completed physical evaluator snapshots are valid with physical termination_reason=null: task-duration incompletes, not collision/sensing/interface failures.

## Teacher history versus current-policy rear work

Five accepted reset-only prefixes each execute948 decisions/7,584 ticks, reaching P10 at63.2s; every actual policy segment begins decision1/tick7,592 with takeover boundary7,584. Prefix records total4,755:5 starts,4,740 teacher decisions,5 results and5 policy-credit-start metadata records; all `policy_credit` flags false. All2,432 policy rows report teacher-in-storage=false and `teacher_initialized_suffix` scope.

Identical front/RR event history is already present at all five policy starts: FR Q71/C1,665/P1,695; FL Q2,461/C3,115/P3,583; RR Q6,938/C7,109/P7,579. These are real teacher events but **not PPO-earned RR or front-leg achievements** in this block. Earlier RL initial-clearance events at7/1,723 also belong to the teacher, not current-policy qualification.

| Episode | Current-policy RL hard events after7,584 | Terminal /tail interpretation |
|---|---|---|
|0 |Q7,961 /C8,063 /P8,138 |Real RL placement; P13 stop remains incomplete |
|1 |No Q/C/P |First missing rear completion is RL lift/cross/place |
|2 |Q9,228 →ground-before-cross revocation9,588; no C/P |Retained event tick is history, current qualification=false |
|3 |Q7,974→revocation8,015; Q8,995→revocation9,640; Q10,044; no C/P |Current qualification=true at terminal, but no crossing/placement |
|4 tail |Q7,958 /C8,066 /P8,128 |Real RL placement before incomplete sampling window ends; not task success |

The episode3 event_ticks field keeps its first RLQ7,974; the chronological lift-attempt list proves subsequent revocations/requalifications. It must not be summarized as an uninterrupted first lift.

Current contacts differ from old placement history. Episode1/2 end with RR back on GROUND (loads0.532892/0.080307), RL respectively AIR/GROUND and no RL crossing. Episode3 ends RR TOP/load0.402428, RL AIR/front−71.801450mm/clearance−28.641356mm/load0 despite a retained current qualification; its first incomplete target remains placed_RL. Thus the three P12 outcomes do not share a single proved physical cause.

Episode0 ends FL/RL/RR TOP, FR AIR; region/support=true but controlled=false, stable0. Its detailed eight-second, zero-nominal-wheel stop diagnosis is preserved separately in `p13_headroom_first_episode_readonly.md`, not generalized to every episode. Final unfinished tail has RR/RL/FR TOP loads0.190907/0.440328/0.368765, FL AIR/load0; region/support=true, controlled=false, stable0, body speed0.082898533m/s, measured-wheel maximum0.393794000rad/s and command maximum0.351548247rad/s. Historical all-placed is not all-wheel contact or successful controlled stop.

## Native, write and trajectory continuity evidence

Policy physics total **19,435**, exactly2,432×8−21. Globals83,983/84,436/84,889 are genuine one-tick terminal intervals, each accounting for seven fewer ticks; every other policy decision executes8ticks. Compact native summaries and all19,435 individual compact records agree: **verified19,435 /actual-native-effect19,435 /own-phase-effect19,423**. Twelve incoming hold ticks are excluded from own credit. This is the actual recorded verifier result, not an independent full-rate re-creation of unlogged float32 buffers.

The four recorded state-write category sums are0: root pose, root velocity, force/impulse and gravity. All12 ordinary phase changes are nonterminal, time_outs=false and terminal_bootstrap_allowed=true; no phase-label GAE cut is introduced by these transitions. Genuine task terminal intervals retain their terminal semantics. Rollout tensor contents were not loaded or re-audited in this report.

Teachers separately contribute **4,740 decisions /37,920 physics ticks**, so physical core accounting is **7,172 decisions /57,355 ticks** across five starts. Teacher simulation time316s and policy161.958333s sum477.958333s; prefix time remains part of each physical task horizon, not optimizer credit.

## Update and elapsed-time records

Nineteen optimizer records611–629 each advance global decisions by128 and perform20 optimizer steps. All actor-before/after links are contiguous from source `605d17729aec86d8d9a3e4470368591040c7ae049fb01f805acb81d2124617ad` to final `8eb3d28a16b8da10a7660d3ae2cbe2b8642bb7b14b043909183663297263d56c`, each actor changes and every record reports finite nonzero gradients. Zero link/counter/unchanged-actor/gradient anomalies were found. Recorded gradient norm extrema are1.002103764 and1.414213639. Configured5epochs×4minibatches agrees with **19×20=380** actual steps.

All19 update-end learning rates and final sidecar are1e−5, matching the immediate source's recorded rate; runner configuration base3e−5 is not claimed as the actual rate of every minibatch. Identity normalizer SHA remains source `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`. This is same-MDP continuation, not fresh-Adam or a new architecture migration. Physical state is not saved; future resume uses legal reset, not bitwise physical continuation.

Recorded `wall_time_s` is **2,288.737341400003s** for the training loop. Reset telemetry separately records reset_wall_time23.338547800202s and roll_in_wall_time1,713.106228699675s over the five prefixes. The first prefix precedes the loop; later reset/prefix work occurs inside it. These fields overlap and are **not summed or subtracted into an invented optimizer-only or end-to-end throughput number**.

## Final status

Twenty-three finalized blocks now add **74,880 policy decisions /585 PPO updates /11,700 optimizer steps** after origin10,112/44/880, yielding84,992/629/12,580. The first22 blocks, all failed/incomplete A/B/C results, initial/first-episode diagnostics and unused budgets remain preserved. No suffix/full-task success, new successful video or paired stability improvement is claimed.

Root has started saved84,992 natural-P01 evaluation under the same runtime. Its new run is not read or precredited here; latest completed deterministic P01 remains C82,560/P09 BODY_COLLISION. This report is final and does not continue monitoring.

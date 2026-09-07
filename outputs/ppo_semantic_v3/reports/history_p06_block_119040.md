# #37 completed history-kernel P06/offset160 block — 119040

Fixed run: `train/20260907T0736098630371Z_ge8462f066908_a07ce6b30c64439e8e8735a0926c4cc6`, HEAD `e8462f06690873c8ac4c4fec453f451f5d10d0e8`. The root confirmed exit0/process exit; published `training_manifest.json` is `SUCCEEDED`. This is execution/training completion, **not task success**. N1, seed1001, frozen-FSM reset-only P06 prefix, offset160, `history_conditioned_heteroscedastic_log_v1`.

Review scope: final JSON manifests/sidecars, one completed 1024-row policy-audit scan, one completed prefix-evidence scan, the one completed-episode record, and eight small update records. No PT/Torch/Python/GPU/Isaac or repeated file hashing. An early active-file `OpenText` sharing error was a read-tool issue, corrected by read/write sharing for schema inspection; it is not a training failure. No following evaluation was read.

## Actual ledger and saves

| Quantity | Verified result |
|---|---:|
| Source global / PPO updates / optimizer steps | 118016 / 887 / 17740 |
| Added credited policy / PPO / optimizer | **1024 / 8 / 160** |
| Final global / PPO / optimizer | **119040 / 895 / 17900** |
| Planned / consumed / unconsumed / rounding | 1024 / 1024 / 0 / 0 |
| Wall time | 677.5841695999261 s |
| v3 origin | 10112, unchanged |
| Full-episode spent | 53504, unchanged |
| Phase-suffix spent | 54400 → 55424 |
| Smoke spent | 0, unchanged |

Across the 37 finalized blocks, additions since origin10112/update44/optimizer880 are **108928 / 851 / 17020**. `53504 + 55424 + 10112 = 119040`. No teacher decision or unissued planned decision enters this ledger.

The eight update records are 888–895, global118144/118272/118400/118528/118656/118784/118912/119040, each20 optimizer steps. Every record reports changed actor parameters and finite nonzero gradients. Source→first, all seven adjacent updates, and final actor fingerprints match. Source actor SHA `49515718e9c0b6d9bccde781973e7395a5bcc8b7be55fed16f108f236945baad`; final `46eb02e6bef26b5240f6283a893fe3b5fa8438c047541b2672cd22b478edce7b`. Recorded gradient norms span1.0772437015–1.4142135705; these are existing diagnostic aggregates, not a new gradient-bound assertion. All recorded update-end learning rates are **1e-5**, distinct from the verified fresh-Adam initial3e-5.

Three actual regular saves, separate from the migration initial:

| Saved global | PPO / optimizer | Suffix spent | Roundtrip and actor/update match |
|---:|---:|---:|---|
| 118144 | 888 / 17760 | 54528 | true / true |
| 118528 | 891 / 17820 | 54912 | true / true |
| 119040 | 895 / 17900 | 55424 | true / true |

Each sidecar points to this source run and records the new history-kernel policy. These are publisher load/save receipts plus metadata comparisons, not independent tensor loads here. The initial weight/std/critic/identity-normalizer/RNG preservation, fresh Adam/storage, six unchanged configuration hashes, and explicit conditional-policy-only transition are documented in [history_kernel_initial_118016.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/history_kernel_initial_118016.md). Gamma .9985/lambda .99,324/12, rollout128 and the physical/reward configuration remain unchanged at this boundary.

## Credited samples, true terminal and nonterminal tail

All1024 policy records are contiguous **g118017–119040**. Phase counts P01–P13 are **[0,0,0,0,0,323,2,2,697,0,0,0,0]**. P10–P13 receive no PPO samples in this block.

| Physical episode | Credited globals | Decisions / ticks | P06 / P07 / P08 / P09 | End state |
|---|---|---:|---:|---|
| 0, completed | 118017–118594 | 578 / 4624 | 126 / 1 / 1 / 450 | P09 deadline, `INCOMPLETE_CONTROLLER_BLOCKED`, task time79.0666666667s, age30s |
| 1, **nonterminal tail** | 118595–119040 | 446 / 3568 | 197 / 1 / 1 / 247 | P09, tick8432/time70.2666666667s, age16.4666666667s, no termination |

Only one completed-episode record exists. Its physical evaluator is valid, with no physical failure reason. The tail is **not a second failure**, a completed episode, or a full/suffix success. Both fresh-P01-current-policy success count and teacher-initialized suffix success count are0.

All six ordinary transitions remain nonterminal and bootstrap-eligible:

| Episode | P06→P07 | P07→P08 | P08→P09 |
|---|---|---|---|
| 0 | g118142, tick5872 | g118143, tick5880 | g118144, tick5888 |
| 1 | g118791, tick6440 | g118792, tick6448 | g118793, tick6456 |

P07/P08 each have one issued decision per episode. No stage handoff clears the GAE episode. There are1023 nonterminal bootstrap-eligible rows; the sole true terminal has `potential_after=0`, terminal event−40, and no bootstrap. Adjacent credited potential-before/previous-potential-after values match within1e-10 throughout each episode, including ordinary phase transitions. Prefix resets are intentionally excluded from that adjacency check. Last-row nonterminal bootstrap eligibility is recorded; no independent PT inspection of its numeric critic bootstrap was performed.

## Teacher exclusion and physical continuity

Two accepted prefixes each use **608 decisions /4864 ticks**, reaching actualP06 at40.5333333333s with159.4666666667s task time remaining. The requested P06 phase is still active at each credit start; neither prefix falls back. All1216 prefix-decision rows have `policy_credit=false`, zero raw/projected PPO residual, native verification complete, and no-state-write verification true.

Teacher totals are **1216 decisions /9728 ticks**, excluded from storage and policy/optimizer credit. Physical core conservation is **2240 decisions /17920 ticks = teacher1216/9728 + PPO1024/8192**. Prefix roll-in wall500.3966283002s and reset wall17.2174996000s are separate timing scopes; the block's wall time is not an isolated policy-throughput benchmark.

FR Q71/C1665/P1695 and FL Q2461/C3115/P3583 are already present in the first credited observation of both episodes; they are teacher-created history, not newly earned PPO events. At the first credited endpoint both rear legs have no hardQ/C/P. Current contact must be read separately: first ep0 endpoint FL is TOP/supporttrue/load.33647, whereas first ep1 endpoint FL is AIR/supportfalse/load0 despite the same historical placement. These are post-first-action endpoints, not a claim about the exact pre-action handoff contact sample.

All **8192 credited physical ticks** are native-verified and have actual native target effect; **8186** have own-phase-request effect and **6** are recorded handoff holds. All four in-episode root-pose/root-velocity/force-or-impulse/gravity-write counters are0. The fixed scan found no summary-count mismatch, in-episode physical/command-clock discontinuity, or terminal finite-observation fallback. This verifies actual execution receipts, not every hidden physical state or causal effectiveness of a target. Compact training logs are not a complete120Hz raw-sensor stream.

## Newly observed rear events and current geometry

Episode0: RR initial-clearance attempts begin at tick6390 and recur, but **no RR hardQ/C/P** is recorded. RL obtains a new hard qualification at7766, then it is revoked on GROUND before crossing at7825; no RL C/P. Thus an initial lift or even incidental RL qualification does not complete the scheduled RR capture task. The terminal has `placed_RR=0.0578027082`, not completion.

Episode1 tail: RL hardQ6150 is revoked on GROUND6243, with no C/P. RR has initial-clearance attempts at5768/6528/6672/6794/6908, then **new PPO hardQ6917 and crossing7576**. There is **no RR placement**, no RL crossing/placement, and no P10 entry. These rear events occur after the physical credit-start tick4864, unlike the teacher's front-leg events.

| Endpoint | RR current state | RL current state | FL current state |
|---|---|---|---|
| Episode0 true terminal | AIR, supportfalse/load0, front−279.384mm, clearance−48.006mm; noQ/C/P | GROUND, supporttrue/load.571458, front−325.857mm, clearance−51.779mm; Q revoked/noC/P | AIR, supportfalse/load0, clearance+127.540mm; historicalQ/C/P remains |
| Episode1 nonterminal tail | **GROUND**, supporttrue/load.140340, front−83.052mm, clearance−47.649mm; historicalQ/Ctrue, Pfalse | GROUND, supporttrue/load.427162, front−165.762mm, clearance−51.800mm; Q revoked/noC/P | AIR, supportfalse/load0, clearance+67.625mm; historicalQ/C/P remains |

FR is currentTOP/supporttrue at both endpoints (loads.428542/.432498). The RR crossing is real recorded progress but does not guarantee retained placement or current forward geometry: the tail has already retreated behind the front plane and returned to GROUND. First unfinished task remains RR capture. Across this fixed block, decision-end maxima are RR clearance+93.635292mm/front+4.505220mm and RL clearance+27.607681mm/front−133.464482mm; these extrema can occur at different samples and are not a reconstructed single pose or120Hz peak.

## Reward ledger and limits

Summed signed published families over credited rows only: task_progress **−46.8974818126**, body_stability **−1.0626589732**, contact_motion_quality **−0.01236752420**, control_smoothness **−2.6778724955**, control_regularization **0**; total **−50.6503800132**. This combines a true-terminal episode with a nonterminal tail and is not a complete-episode performance comparison or causal attribution. The true terminal's beforePhi=.4785685476/afterPhi0/event−40 preserves absorbing-state semantics.

Conclusion: the selected history-kernel course completed its actual1024/8/160 budget with valid native evidence and continuous ordinary handoffs. It produced a new RR Q/C in a nonterminal tail, not placement, later-stage coverage, or task success. No paired improvement, sensor bug, or reward/nominal intervention is inferred. Following C119040 evaluation is outside this report.

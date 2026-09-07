# P06 offset-240 capture-approach diagnostic block — final actual ledger

## Final status and consumed budget

Run: `runs/ppo_semantic_v3/train/20260906T1738509257629Z_gc34262abffc1_04bbf6558046407990f85358d738cc17`, runtime `c34262abffc16847ff32d15ecbf790dd60803e0a`, N1 seed 1001, source checkpoint 72320. Both finalized manifests report **`STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`**, responding to the run-specific stop request. This is a deliberate completed-update stop, not optimizer failure or task success.

| Ledger item | Actual final value |
| --- | ---: |
| Original planned request | 2048 policy decisions |
| Consumed policy decisions | **768** |
| Unconsumed original request | **1280** |
| PPO updates in this run | **6** |
| Optimizer steps in this run | **120** |
| Rounding overrun | 0 |
| Lifetime policy decisions | 72320 → **73088** |
| Lifetime PPO updates | 530 → **536** |
| Lifetime optimizer steps | 10600 → **10720** |
| Recorded training-manifest wall time | 831.2335138 s |

The finalized `requested_policy_decisions` and `actual_policy_decisions` both equal 768; the separate `planned_requested_policy_decisions=2048` and `unconsumed_requested_policy_decisions=1280` preserve the original request. No expected or planned endpoint was substituted for these completed values.

The final checkpoint budget ledger is **phase_suffix 33792 = 33024 + 768**, full_episode 29184, smoke 0. With preserved new-MDP origin 10112, `33792 + 29184 + 10112 = 73088`. This revision did not reset lifetime or already-spent training budget.

## PPO, teacher preparation and physics are separate

Exactly **768 policy audit rows**, global **72321–73088**, and six complete rollout files `rollout_000531.pt` through `rollout_000536.pt` are present. All credited phase labels are **P06=768**; P01–P05 and P07–P13 contribute zero PPO decisions in this block.

Three reset-only prefix attempts were accepted, each 688 decisions / 5504 physics ticks, with actual P06 still active at credit onset and offset 240. They contribute **2064 prefix decisions / 16512 prefix ticks**, all excluded from PPO. Every policy audit row has `prefix_teacher_data_in_ppo_storage=false`.

| Scope | Decisions | Physics ticks |
| --- | ---: | ---: |
| Credited PPO only | **768** | **6130** |
| Excluded prefix preparation | **2064** | **16512** |
| Physical core total | **2832** | **22642** |

The core's broader phase histogram includes the teacher's P01–P05; it must not be presented as policy phase coverage. The two one-tick terminal decisions account for `768 × 8 − 14 = 6130` credited ticks. Prefix time remains inside each real task's physical horizon.

## Episodes and the nonterminal tail

| Episode | PPO global window | PPO decisions / ticks | Last physical tick / time | Last result |
| --- | --- | --- | --- | --- |
| 0 | 72321–72681 | 361 / 2881 | 8385 / 69.875 s | P06 age 40.008333, `INCOMPLETE_CONTROLLER_BLOCKED` |
| 1 | 72682–73042 | 361 / 2881 | 8385 / 69.875 s | P06 age 40.008333, `INCOMPLETE_CONTROLLER_BLOCKED` |
| 2, partial task | 73043–73088 | **46 / 368** | 5872 / 48.933333 s | P06 age 19.066667, **terminal=false**, no task result |

Episodes 0 and 1 have valid physical evaluators and null physical failure reasons. Neither is a body-collision or wheel-only classification. Both are genuine stage deadlines, with task success false, absorbing next potential zero and no terminal bootstrap.

The third episode's 46 decisions are **already included in the completed rollout/update**, not discarded or merely planned samples. Its task remains unfinished and unobserved beyond that point; the administrative stop is not a third failure or success. The checkpoint records `physical_env_state_saved=false` and `resume_physics=legal_reset_not_bitwise_continuation`, so the saved model does not promise physical continuation of this partial task.

## Hard events, current support and first unfinished task

All three physical episodes share the accepted prefix's recorded front events: FR Q71/C1665/P1695 and FL Q2461/C3115/P3583, before the credit start at tick 5504. These remain valid history but are **not PPO-acquired Q/C/P**. Neither rear leg obtains hard qualified lift, crossing or placement anywhere in the credited block.

RR does show measured initial-clearance processes after credit: 20 events in episode 0, 21 in episode 1 and 2 in the partial episode. RL has no new initial-clearance event after credit. Initial clearance is not qualification or placement. There are no post-credit stage transitions, no later channel-owner execution sample, and **zero new first-capture branch activations**: front legs are already placed, RR lacks hard Q+C, and RL is predecessor-blocked.

| Episode | Best recorded paired approach | RL front at best (mm) | Final RR / RL front (mm) | Current last support |
| --- | --- | ---: | --- | --- |
| 0 | .988272 at tick 8032 | **-222.931918** | -168.085810 / -227.768984 | FL AIR, FR TOP, RL+RR GROUND |
| 1 | .972106 at tick 6664 | **-226.973500** | -125.195887 / -229.859379 | FL+RR AIR, FR TOP, RL GROUND |
| 2 partial | .850895 at tick 5824 | -257.276323 | -174.730566 / -260.933575 | FL AIR, FR TOP, RL+RR GROUND |

Against the existing -220 mm workspace lower bound, RL's best retained shortfall is 2.931918 mm in episode 0 and 6.973500 mm in episode 1. Both completed episodes first remain incomplete at **current RL workspace approach**, despite RR having reached its workspace. The partial episode is still approaching. These are decision-end geometry samples; no unrecorded 120 Hz trajectory is fabricated.

FL's historical placement does not imply present support. At the two real terminals FL load is zero; episode-0 FR/RL/RR loads are .456670/.518466/.024864, and episode-1 FR/RL loads are .509040/.490960 with RR also AIR. No fixed support or posture requirement is inferred from those observations.

The finite P06 nominal wheel suggestion remains `[.3,.3,.3,.3]` at the inspected terminal/partial endpoints, while residuals remain active. For example, episode-0 terminal actual logical wheels are `[+.477868,+.042291,-.107440,+.344069]`; episode-1 terminal `[+.482874,-.011636,-.037279,+.219641]`. Nominal is not actual net control. This supports ongoing policy influence but does not establish a unique cause of incomplete approach. All source-endpoint and post-finite-tail active flags are zero in this block, so it supplies no new tail-extension outcome.

## Actual optimizer chain and saved sidecars

All six update records report changed actor parameters and finite nonzero gradients, with 20 optimizer steps each. Their global/update pairs are:

`72448/531 → 72576/532 → 72704/533 → 72832/534 → 72960/535 → 73088/536`.

The source checkpoint sidecar actor digest matches the first update's before-digest. Every update's after-digest equals the next before-digest; no link is missing. The recorded chain begins `1587569b6d1510508833bfca0f9d1ce128cc3fe47838ff38510a93dff2744300` and ends `5031de525ff8d43dfcc9ea315a5efc4aba8999e51607644255aa94e765c64589`. The final sidecar has that same actor digest. The six recorded effective optimizer learning rates are all 1e-5; this report changes none.

Three `.pt` checkpoints and their corresponding `_manifest.json` sidecars exist in `outputs/ppo_semantic_v3/checkpoints/history/`:

| Checkpoint step | Lifetime update / optimizer steps | Sidecar save-load round trip |
| --- | --- | --- |
| 000072448 | 531 / 10620 | true |
| 000072832 | 534 / 10680 | true |
| **000073088** | **536 / 10720** | **true** |

Each sidecar names this source run and the actual P06 offset-240 sampling. Final policy contract remains 324 observations, 12 raw actions, `heteroscedastic_log_v1`; fixed-schema/identity-normalizer semantics remain recorded, with the same normalizer digest in these three saves. Latest `.pt` size is 3,844,767 bytes. Its **recorded** SHA-256 is `3efd9531d6bf797a2852107c1183ac8be2b96bcf5eaa3ed59358f0cd9d44ffef`.

These are sidecar/update-record checks and file-existence/size checks. I did **not** rehash or load checkpoint tensors or independently perform their round trip; the root agent owns the actual checkpoint hash/evaluation verification.

## Native audit and conclusion

All **6130 credited physics ticks** have verified actual native effects and own-phase request effects. Every credited decision verifies zero in-episode root-pose, root-velocity, force/impulse and gravity writes. The compact training record does not contain a full raw physical/contact stream; it cannot support invented per-tick forces, exact contact points or unrecorded kinematic maxima.

This finalized block successfully executed and saved **768 actual optimization decisions**, while consuming only part of its original request. It produced P06-only policy coverage, two true approach deadlines, and one nonterminal task tail—not full-task success and not a test of the new post-cross capture reward. The next sampling/evaluation choice is separate; no new gate, dynamics/reward change or success reclassification is proposed here.

Sources: finalized training/run manifests, all 768 policy audits, two completed-episode records, six optimizer-update records, prefix telemetry, and the three named checkpoint sidecars. Only PowerShell reads and this independent report write were performed. Reporting is complete; no Python, new simulation, repeated large hashes, production edits or historical artifact changes.

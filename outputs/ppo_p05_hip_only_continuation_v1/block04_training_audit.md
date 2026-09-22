# Block04 — sealed P06-frontier training audit

Run: `20260922T0814225788964Z_ga802b24d78df_790387038d25495ea5f001f934d3ade2`; frozen source runtime `a802b24d78df5f1b8f0caf914ce7a8a68a98db8e`. Audit reads sealed recorded evidence only, not the subsequently revised runtime. CPU helper exited 0 in 6.168 s; no Isaac, policy inference or production/configuration changes.

## Outcome and actual learning

Stopped cleanly at the requested verified update boundary: **2048 of 4096 planned decisions consumed**, 2048 unconsumed. This is `STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`, not a failed training process and not task success.

- Added **2048 policy decisions / 16 PPO updates / 320 Adam steps**, updates **1574–1589**.
- Lifetime checkpoint counts: **207872 / 1589 / 31780**.
- P05 branch cumulative: **8192 / 64 / 1280**; feedback-v2 branch actually learned **4096 / 32 / 640**. Migration itself is not learning.
- Two completed learner episodes remained incomplete in P09. Third episode **really placed RR**, advanced through P10 into P11, and was nonterminal when the completed update boundary was saved.
- This is frozen-CP205824 natural-P01 prefix → P06 learner training. **No complete natural-P01 PPO success is established.**

## Exact sampled phases

Request = actual phase of the action stored in PPO. Endpoint = phase after its physical steps. Prefix actions are physical behavior but **uncredited**, not PPO rows.

| Phase | Learner requests | Learner endpoints | Uncredited prefix requests |
|---|---:|---:|---:|
| P01 | 0 | 0 | 6 |
| P02 | 0 | 0 | 1005 |
| P03 | 0 | 0 | 12 |
| P04 | 0 | 0 | 3 |
| P05 | 0 | 0 | 660 |
| P06 | 548 | 545 | 0 |
| P07 | 6 | 6 | 0 |
| P08 | 3 | 3 | 0 |
| P09 | 1354 | 1356 | 0 |
| P10 | 1 | 1 | 0 |
| P11 | 136 | 137 | 0 |
| P12 | 0 | 0 | 0 |
| P13 | 0 | 0 | 0 |
| Total | 2048 | 2048 | 1686 |

All three prefixes were independently accepted at **562 decisions / 4496 physics ticks / 37.466667 s** each; all prefix ledger rows declare `policy_credit=false`. All 2048 learner receipts say neither teacher nor checkpoint-prefix data enters PPO storage. Saved rollout phase one-hots and original Gaussian action/mean/std/logp/reward/done rows match the learner audit exactly.

Actual learner physics: **16377 ticks / 136.475 s**; prefix physics: **13488 ticks / 112.4 s**; combined task physics **29865 ticks / 248.875 s across three episodes**. These exclude reset/settling. The task horizon includes its own prefix; there is no single >200-second episode.

## RR physical-event coverage

| RR fact | Actual policy inputs | Physical endpoints |
|---|---:|---:|
| Current qualified lift | 1401 | 1404 |
| Real earned crossing history | 1082 | 1085 |
| Current real TOP (loaded legal region, no ground) | 35 | 36 |
| Real placement history | 137 | 138 |
| Crossed but not yet placed | 945 | 947 |

Input Q/C/P are read directly from the saved 389-dimensional rollout, not inferred from phase labels. Exact input TOP uses 2045 preceding recorded endpoints plus the three prefix-handoff inputs (global decisions 205825, 206546, 207246): saved RR ground-pair active flag at index 137 equals 1, which excludes TOP under the recorded all-stage definition. **No unknown TOP inputs are filled with guessed values.**

Endpoint `top_surface_contact` count is **39**, versus usable current `top_contact` **36**; do not merge these distinct predicates. RR placed inputs are actually consumed by updates 1588 (9) and 1589 (128), **685 optimization sample uses** over five epochs; crossed inputs yield 5410 uses and current qualified inputs 7005. Exact known TOP input sample uses are 175. These are repeated minibatch uses, not extra policy decisions.

## Episode evidence

| Learner episode | Credited decisions / ticks | Full physical age including prefix | End state | RR Q / cross / place event ticks |
|---|---:|---:|---|---|
| 0 | 721 / 5764 | 85.500000 s | P09 terminal, INCOMPLETE_CONTROLLER_BLOCKED | 6299 / 7120 / absent |
| 1 | 700 / 5597 | 84.108333 s | P09 terminal, INCOMPLETE_CONTROLLER_BLOCKED | 6157 / 6979 / absent |
| 2 | 627 / 5016 | 79.266667 s | P11 **partial/nonterminal** | 6195 / 7105 / **8415** |

The first two ended with current qualified RR AIR, no TOP/placement, respectively gap **60.116879 mm / 102.294136 mm**, front distance **140.869800 mm / 78.408423 mm**, and zero bearing force. Their first unfinished RR task is placement, not missing crossing.

Third learner RR placement occurred at **tick 8415 / 70.125 s**. Last endpoint **9512 / 79.266667 s**: RR current lift valid, real TOP, no ground, gap **-0.004477 mm**, front **+53.443078 mm**, bearing **1.292089 N**; FR/FL/RR placed history true, RL false. No RL qualified/crossed/placed event is recorded. P11 continuation has not been evaluated to a final task outcome; it must not be labeled success or RL task failure merely because the update boundary stopped sampling.

Both complete episodes have the exact same real prefix events FR Q/C/P = 23/2710/2721 and FL Q/C/P = 2738/3857/4492, also retained in episode 2. The frozen prefix is not credited to the learner.

## Execution and optimizer integrity

- All **16377** credited native ticks verified; final mapping/dispatch matches and all 12 residual permits are open. No in-episode state writes recorded.
- FL capture assist owns only FL hip/knee on **581** endpoints; **1467** endpoints have all 12 policy channels unmodified by assist. No RR assist. Pending/allow-continuation/scheduler-advanced-pending endpoint counts are all zero in these suffix samples.
- Every rollout is finite, `[128,1,389]` for actor/critic observations and `[128,1,12]` raw action. Actor/critic observations match. Original raw Gaussian actions and distribution fields match the collected audit; one actual draw, no extra random draw.
- All 16 updates have finite diagnostics, nonzero gradient and changed actor hashes. Each update has 20 actual minibatches and each of its 128 samples is used exactly five times. Independent CPU raw-logp maximum discrepancy: **7.62939453125e-6**.
- Source→target Adam step delta is exactly **320 for all 12 parameter-state entries**; 6 actor and 6 critic parameter tensors changed. LR **1e-5**, Identity normalizers unchanged in type/semantics.
- Actor before: `c5c5dffa5df0725aeeb082b600b0eaf3052a9520c61ca5ed3b0939e920053c1f`; after: `1b845daa87f1c78b0044546712a243bfdd6ca0491c3418d9eabf0844c625cda9`.
- Recorded native save/load round-trip: **true**. Independently loaded final checkpoint on CPU, serialized to an in-memory buffer, reloaded and checked exact nested state equality. This fresh CPU round-trip is **not** an Isaac resume or a newly claimed physical evaluation.

## Immutable checkpoint and hashes

Final pre-reward-revision checkpoint: [checkpoint_step_000207872.pt](checkpoints/history/checkpoint_step_000207872.pt)

- Checkpoint SHA-256: `8f15de6495adc49c6d7e19000bc56ca612ce9861efaa9811bfab5bd057ea9cf5`
- Checkpoint manifest SHA-256: `100abb15f3d25db15a51817da5b37266efe3fe254bfcf596ccab1ded083bc717`
- Training manifest SHA-256: `76b489c06c15ac5122f638511ea2895f35dfcbfa631f4b8805a768f166e54881`
- Frozen prefix/source checkpoint CP205824 SHA-256: `c1615e5d6987306d41dcdbbd4c3767bd0d28d30fda559d149e7c5b2ade886735`

[Machine-readable audit](block04_training_audit.json) includes per-update physical-input quantities, both complete episodes, the partial episode, prefix exclusion and exact execution counts. [CPU helper](audit_block04_cpu.py) imports only stdlib and torch and reuses recorded old native verification flags; it does not reconstruct old audits with modified runtime classes.

No new AUX, sigma, nominal or physics modification is claimed by this audit. The inherited limited-AUX lineage and explicit FL capture assist mean this should not be relabeled an unassisted pure-policy run. Subsequent reward/observation-scalar migration and natural-P01 evaluation belong to the next separately versioned result.

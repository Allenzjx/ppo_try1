# Sealed front audit — CP221184 v4 versus CP220544 v2

Scope: read-only CPU analysis of sealed sources, from natural P01 through the first P06 or terminal. No actor forward, physics, training, production/configuration edit, or new front optimization. This is not an A/B claim with identical whole controllers. It does not describe the later headless CP220544-v4 counterfactual, whose result is not assumed here.

## Outcome and exact provenance

| Candidate | Frozen source/runtime | Result |
|---|---|---|
| CP220544, deterministic, declared capture assist | `20260923T0356402230589Z_g26db2a1946e8_c1a7c95ca1db41d69882389e1e97cd8d/source`; `26db2a1946e8` | Front prefix reached P06 at 51.000 s with FL capture still pending; whole 104.033 s episode later reached RR/P09, without complete success. Whole manifest eventually records FR/FL placed. |
| CP221184, deterministic, declared capture assist/v4 | `20260923T0546444511040Z_g632295bc49cf_9a60bc74b9b14f6eb7debd0d2e92d914/source`; `632295bc49cf` | 60.733333 s, 7,288 physics ticks, 911 decisions; P05 incomplete. `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`, P05 age 40.000 s; FR placed, FL/RR/RL not placed. No RR capture intervention occurred. |

Source directories are under `runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation/`. CP221184's physical evaluator did not declare a collision/safety failure; the last decision's semantic-task scheduler supplies the local terminal above. Its finalized viewport/full-decode evidence is valid (911 frames, 60.733334 s, 15 fps); this is task non-completion, not a missing/invalid video. The raw container duration anomaly is not the decoded episode duration.

CP hashes: CP220544 `a32a4f01afbfbcda06ba59f7e439bb5ab9aa1dbefb3d8e3093e0dd86c827e85e`; CP221184 `2fed192f9c5141a40f769893d0266ec3b95c91d55a4b3091acc411cd99672f90`. Both seed 4001, natural P01, deterministic conditional mean, 410 observations/12 actions, same camera and normalizer hash. The actor hashes differ. CP221184 includes **640 real rear-only decisions / 5 PPO updates / 100 Adam steps** (P07=3, P08=2, P09=635); excluded teacher prefixes are not front on-policy rehearsal. The later v4 migration itself adds no learning or AUX updates.

## First divergence and FL qualification

| Event | CP220544 v2 | CP221184 v4 |
|---|---:|---:|
| First applied action | Tick 1: same actual source N and mapped nominal as v4, different raw policy/final target | Raw policy already differs on first decision (recorded at end tick 8, applied starting tick 1); max raw difference 0.003536938, max final-target difference 0.031533496 degrees |
| P03 / P05 entry | 19.733333 / 20.066667 s | 20.400000 / 20.733333 s |
| FL active-lift qualification earned | Tick 2409, 20.075000 s | Tick 2500, 20.833333 s |
| First later AIR interruption | None before analyzed first P06 | Tick 3621, 30.175000 s: AIR and soft-AIR qualification flags drop; not yet ground. AIR returns at 30.400000 s. Do not infer wall contact from these flags alone. |
| First ground after earned lift | None before analyzed first P06 | Tick 3680, 30.666667 s: real GROUND; active-lift qualification revoked, no prior FL crossing; force 2.565825 N; front distance -64.515 mm, bottom-to-top gap -51.276 mm |
| Crossing / terminal | FL crossing history earned 29.533333 s; P06 soft continuation at 51 s is not itself placement | No FL crossing; terminal FL front -67.508 mm, gap -49.999 mm; GROUND, no current valid lift |

Interpretation correction to the original adjacent audit: its `source_N` field outside `native` is the **post-observation/next-frame nominal**. Its first difference at tick 2368 reflects old P03 versus new P02, not a proven first same-dispatch source mismatch. For actual pre-dispatch N use `selected.<candidate>[...].native.source_N`. The first raw output difference predates that phase divergence, but full actor input vectors were not persisted; no identical-input replay was performed. Later actions depend on both different learned weights and different evolving state/HISTORY. These data do not isolate the weights as the sole cause.

## Four wheels at tick 3680 / 30.666667 s

Canonical forward-positive values; raw is dimensionless Gaussian request, all other wheel values rad/s. All four permission masks are 1 and apply to residuals. The actual pre-dispatch N is **[0,0,0,0] in both runs**: the source stop is present. Effective residual is the producer's filtered/projection receipt, not an independently recomputed nominal subtraction. At this sample it equals final canonical wheel target. Native target signs are shown separately.

| Run / wheel | Raw policy | Effective residual = final canonical target | Final native target | Measured canonical velocity |
|---|---:|---:|---:|---:|
| CP220544 / FL | -0.913987 | -0.723040 | +0.723040 | -0.723026 |
| CP220544 / FR | +0.036685 | +0.022001 | +0.022001 | -0.099511 |
| CP220544 / RL | +0.058472 | +0.058406 | -0.058406 | +0.108231 |
| CP220544 / RR | -0.224930 | -0.132727 | -0.132727 | -0.051851 |
| CP221184 / FL | -0.886168 | -0.709496 | +0.709496 | -0.677156 |
| CP221184 / FR | +0.047316 | +0.028368 | +0.028368 | -0.114881 |
| CP221184 / RL | -0.033983 | -0.033970 | +0.033970 | +0.038917 |
| CP221184 / RR | -0.259031 | -0.152033 | -0.152033 | -0.143427 |

Owner at this sample is nominal hold/stop plus policy residual through the verified final atomic dispatch. Both FL/RR capture-assist owner masks are all false. All analyzed native receipts verify the write, both residual masks remain all ones, and the v4 RR wheel shaping contribution is zero throughout the front prefix (RR assist WAIT, guidance false). Four distinct nonzero requests and measured velocities exist; neither only-RR commands nor a three-wheel mask is supported. FR/RL velocity may disagree with target sign because commanded velocity is not measured tracking/traction; ACK alone does not resolve load coupling. FL reverse is present in both runs and is **not alone a newly introduced difference**.

At phase-aligned P05 entry, both N wheels are `[+0.3,+0.3,+0.3,+0.3]`; canonical final targets are old `[-0.065813,+0.332859,+0.292058,+0.201930]`, new `[-0.053011,+0.337609,+0.247302,+0.191899]`. Thus wheel inheritance initially exists, while policy can cancel/reverse an individual nominal. This does not prescribe globally forward-only wheels.

## Joint response and physical divergence at the same tick

Degrees; these same-time samples have different phase ages and physical states, not a matched-state experiment.

| Channel | CP220544 target / actual | CP221184 target / actual |
|---|---:|---:|
| FL hip | 19.8421 / 19.2901 | 18.7017 / 19.0086 |
| FL knee | -32.1287 / -32.1318 | -31.8795 / -32.0153 |
| FR hip | 6.3267 / 9.8290 | 6.1861 / 27.4746 |
| FR knee | 33.7583 / 34.4399 | 33.4346 / 34.2247 |
| RL hip | 1.3104 / 2.5338 | 1.3277 / 2.6132 |
| RL knee | 1.4566 / 1.0366 | 1.2584 / 0.9112 |
| RR hip | 7.2420 / 7.3943 | 7.1442 / 7.1767 |
| RR knee | -8.3297 / -8.0585 | -8.6480 / -8.4187 |

Both actual pre-dispatch joint N vectors are `[22.8,-13.4,0,45.9,6.9,0,0,0]`. FR hip target tracking error is +3.5023 degrees old versus +21.2885 degrees new; this substantial response difference is not proof of an index/writer bug or of a single-joint causal mechanism. New body z is 0.060873 m versus old 0.095104 m; new mass-weighted CoM z is 0.135524 m versus old 0.160461 m. Old FL is AIR with valid crossing history and +8.013 mm gap; new FL is ground/preedge. These records establish the coupled physical divergence, not its unique origin.

## What is / is not established

Scoped config/source comparison finds retained front nominal and P05 capture logic/reward/schema; task-spec/profile changes concern RR feedback, the RR handoff window and RR carry-wheel shaping. However shared adapter/backend files also changed, so whole-controller equivalence must not be claimed. The recorded new RR interventions are inactive before P06. There is no observed front nominal erasure, residual-mask leak or actuator-write loss in the audited receipts.

The appropriate next evidence is the already separated, same-control CP220544-v4 natural-P01 physical counterfactual, not another broad front-reward redesign. No result or video is claimed for that run here. Preserve CP221184 as the latest learned lineage and CP220544 as the previously front-valid candidate; neither larger update count nor an older video establishes current control-version task success.

Evidence: `CP221184_vs_CP220544_front_divergence.json` retains selected native receipts and all event transitions. Its original `.md` remains unchanged and is superseded by the nominal-timing clarification above. The existing script was revised to draft v2 output names after the original CPU execution, but **v2 has not been executed**; no v2 artifact is used here. No production or configuration changes were made for this audit.

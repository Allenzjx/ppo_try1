# Block 02: first sealed batch, read-only audit

Run: `20260922T0523430327244Z_g0001c3138b0b_f01402fe07fa4d85892560249297522b`.
Scope: first prefix plus sealed `rollout_001542.pt`, first 128 applied-decision records, and completed optimizer-update record 1542. CPU only; no runtime/config/checkpoint edits or simulation started. No claims about later, unsealed data.

## Real prefix and credit boundary

- Frozen source CP201728, source SHA recorded in prefix provenance: `a4eb243ce07ad4a9ed1cf2f7f9a22e951181bdb6e0716361b8eaeb173acfd5b6`. Source counts 201728 decisions / 1541 updates. Independent frozen parameter/buffer storage is recorded; first optimizer actor-before hash matches the frozen-prefix actor hash.
- 408 physical prefix decisions: P01 2, P02 254, P03 3, P04 1, P05 148. Every action is a contiguous 8-tick physical interval; total 3264 ticks. Every prefix row explicitly has `policy_credit=false`; in-episode state-write audits pass.
- Real prefix first reached P06 at tick 3264 / 27.2 s. Learner starts there, with RR current qualification, crossing history and placement history all false. The first credited endpoint is tick 3272 / 27.266667 s, global decision 201729.
- This is checkpoint-policy-initialized suffix training, **not full-P01 current-policy training or full-task success**. Prefix data are explicitly excluded from PPO storage.
- Windows directory-entry `Length` was stale; direct file contents establish the evidence above. The initial zero-length observation was not evidence loss.

## First sealed learner batch

128 new decisions, global 201729–201856; 1024 credited physics ticks; final endpoint 35.733333 s. All 128 requested and endpoint stages are P06. No terminal samples.

| Actual observation coverage | Samples |
| --- | ---: |
| RR currently qualified | 0 |
| RR crossed history | 0 |
| RR placed history | 0 |
| RR crossed but unplaced | 0 |
| Pending scheduler handoff | 0 |

This batch covers predecessor preparation, **not yet actual RR crossing/landing**. Later-stage labels must not substitute for those physical flags.

## Original Gaussian and observable assist

- Observation shape `(128, 1, 389)`; current policy contract `p05_hip_only_capture_assist_history_v1`.
- Stored raw Gaussian actions match decision raw actions, selected samples and native audit raw actions exactly in float32. Stored conditional mean, standard deviation and old log probability likewise match collection records.
- Independent CPU `Normal(mean,std).log_prob(raw).sum(-1)` differs from stored old log probability by at most `3.814697265625e-6`. Every decision records one sample and zero extra random draws.
- All 12 assist observation features exactly match preceding authoritative `state_after`, including the first learner observation against the last frozen-prefix action. All five continuation features match policy-request evidence; samples 2–128 also match preceding actual supervisor state.
- Observed assist modes: HOLD 32, DESCEND 46, BLOCKED 50. Endpoint modes: HOLD 31, DESCEND 46, BLOCKED 51; the difference is the normal within-decision state transition.
- All 128 endpoints have only FL hip/knee owner indices `[0,1]`. The other ten pre/post-assist candidate channels are unchanged. Original policy samples are unchanged and all 12 residual-permission bits are 1. Final FL knee equals the assist hold target exactly (maximum error 0 degrees). This is assisted execution, not pure-policy FL capture.
- Native execution verification passes for every credited endpoint.

## Actual update

Completed update 1542: 128 new decisions, 20 optimizer steps, LR `1e-5`, finite nonzero gradients and changed actor parameters. Actor hashes: before `84f3319f19d3b5f4ec43eaa42da5577bfbb3c4da2ca9b9f0cc032cfa03916f89`; after `dba22af0f21e6e309749fd285209a72087549686a49f62e8673f64abe5440ef7`. KL 0.0266004; clip fraction 0.3296875. Optimizer completion is not physical task completion.

## Follow-up: first completed episode only

One-pass audit of the first 756 credited decision records (global 201729–202484), first completed-episode record, and first three next-reset prefix actions. No Torch, runtime edits or physics calls. Counts below are decision-request or post-action endpoint samples, **not every physical tick**.

| Phase | Requested samples | Endpoint samples |
| --- | ---: | ---: |
| P06 | 294 | 293 |
| P07 | 1 | 1 |
| P08 | 1 | 1 |
| P09 | 460 | 461 |
| All others | 0 | 0 |
| Total | 756 | 756 |

RR actual endpoint coverage: current valid lift 17; same-attempt lift established 18; crossed history 0; classified TOP contact 0; top-surface contact 0; placed history 0; crossed-but-unplaced 0. Policy-request current-qualification count is also 17; request placement history count is 0. Actual endpoint contact modes: AIR 33, FRONT_WALL 1, GROUND 498, GROUND_AND_OBSTACLE 137, OBSTACLE_AMBIGUOUS 87. These are mutually exclusive contact-mode classifications; the separate ground-contact boolean is true for 635 endpoints. Pending scheduler handoff count is 0.

First RR valid-lift endpoint: global 202057, P09, tick 5896 / 49.133333 s. Wheel-bottom clearance relative to platform top is still **−38.3126 mm**, and front distance is **−172.4618 mm**. Valid initial lift therefore does not establish platform clearance or crossing. This episode contains rear preparation and brief lift evidence but **no actual RR crossed/landing-state training coverage**, despite 460 requested P09 samples.

Credited suffix physics totals 6045 ticks / 50.375 s (last action legitimately has 5 ticks). Added to the uncredited 3264-tick / 27.2 s prefix, actual episode duration is 9309 ticks / **77.575 s**; physical decision count including prefix is 1164 = 408 + 756. Every credited record retains `prefix_checkpoint_policy_data_in_ppo_storage=false`.

The last decision remains `terminal=true`, P09 `INCOMPLETE_CONTROLLER_BLOCKED`, `terminal_bootstrap_allowed=false`; exactly one terminal record exists. Both its journal and completed-episode record retain `task_success=false` and `full_task_success=false`. Final RR is GROUND, unqualified/uncrossed/unplaced, top gap −50.1948 mm, front distance −56.0169 mm. This verifies the recorded terminal, not an as-yet-unsealed rollout tensor.

The next reset is visible in prefix lines 413–416: a new `checkpoint_prefix_start`, followed by actual P01 tick 8, P01→P02 tick 16, and P02 tick 24. All have `policy_credit=false`; actions retain `task_result_scope=checkpoint_policy_initialization_excluded` and `full_task_success=false`. No later episode or remaining block data were audited.

## Later bounded audit: first actual RR-crossed learner batch

Sealed rollout 1550, global 202753–202880, endpoint times 45.133333–53.6 s. Requested stages: P06 9, P07 1, P08 1, P09 117. First actual RR crossing endpoint in this run is global 202866, tick 6320 / 52.666667 s.

Actual policy-input coverage: RR qualified 85, crossed-but-unplaced **14** (global 202867–202880), placed 0. Endpoint coverage: qualified 86, crossed 15, TOP/placed 0. The 14 crossed observations have platform clearance 29.9093–80.5311 mm. Endpoint and input counts differ because crossing occurs during action 202866.

All 14 crossed observations occur in the official likelihood audit and are used exactly five times across 20 optimizer minibatches. All 128 raw actions, conditional means/sigmas and old log probabilities match sealed storage and collection records; independent CPU Gaussian logp error is at most 3.8147e-6. One draw/action, prefix excluded, native verification and all12 permission checks pass. Completed update 1550 has changed actor parameters, finite gradients, LR 1e-5. This is genuine on-policy crossed-state training, **not TOP placement or full-task success**.

| Quantity, crossed-input subset only | RR hip | RR knee |
| --- | --- | --- |
| Raw conditional mean | 0.180127…0.686488 | −0.469000…−0.364051 |
| Raw conditional sigma | 0.078537…0.112250 | 0.016526…0.020868 |
| Mean mapped through cap·tanh, degrees | 4.276893…14.297307 | −15.746067…−12.555984 |
| Local pre-slew physical 1σ, degrees | 1.240398…2.524686 | 0.492156…0.635058 |
| Mapper+controller baseline, degrees | −10.65…−5.65 | −39.05…−36.306173 |
| Actual requested = effective residual, degrees | 3.237150…14.641700 | −16.114471…−12.324371 |
| Candidate = final target, degrees | −4.912850…8.991700 | −55.153486…−50.709771 |
| Headroom-clipped endpoints / assist owners | 0 / 0 | 0 / 0 |

The candidate/final equality above holds per matched endpoint in the crossed subset; it is not inferred from equal aggregate ranges. Across the whole batch, ordinary final slew can differ (hip −0.729898…0.335813 degrees; knee 0…1.25 degrees). Only FL [0,1] is assist-owned in 20 early endpoints; the remaining 108 have no assist owners. RR pre/post-assist candidates are identical in all 128 endpoints. Mapper baseline is not relabeled as the source nominal.

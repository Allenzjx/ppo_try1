# Final block30 — frozen-FSM P06 course, checkpoint108288

Run `train/20260907T0253258293143Z_gf1a9bbf650b1_a6d3b0a9bb214f398e3579e3f350e7ba`, runtime `f1a9bbf650b1b8f80d5169e09173fcfc68797d99`, N1/seed1001/P06/frozen_fsm/offset0. Both final manifests record `STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`; root confirmed session24163 exit0 and PID184612 gone. The stop request explicitly reallocates sampling from P06-only preparation to the existing P07 pre-lift entry. It is not infrastructure failure, task success, a new MDP or an optimizer-success prerequisite.

Scope: finalized small manifests and sidecars, all1024 credited audit rows, both complete prefix attempts, the single completed-episode row and all8 optimizer records. No subsequent P07 run, Python/PT/tensor load, GPU/Isaac, checkpoint hash recomputation, production/config/test edit or future credit. One overbroad small-manifest display included nested result content and was tool-truncated; a subsequent compact whitelist read provided the relevant scalar fields. This did not affect training or the complete audit-row scan.

## Actual budget and saved chain

| Quantity | Actual |
|---|---:|
| Planned request | 2048 decisions |
| Consumed/requested final allocation | 1024 decisions |
| Unconsumed plan | 1024 decisions |
| Rounding overrun | 0 |
| New PPO updates / optimizer steps | 8 / 160 |
| Lifetime global / PPO / optimizer | 107264/803/16060 → **108288/811/16220** |
| Manifest wall time | 820.5672665999737s |
| Full-episode budget spent | 49408, unchanged |
| Suffix budget spent | 47744 → 48768 |
| Smoke budget spent | 0 |

The fixed origin remains10112/44/880. **Thirty finalized training blocks now add98176 decisions,767 PPO updates and15340 optimizer steps**: `108288−10112`, `811−44`, `16220−880`. The budget identity is `49408+48768=98176`, plus origin10112 gives108288. The unused1024 are not charged, and future P07 work is not included.

This run is ordinary same-runtime resume, not another warm start: started arguments set NewMdpWarmStart=false and both migration options null; source/target runtime and policy contracts match. The learned source actor including std is unchanged at first optimizer entry. Source critic/Adam/normalizer restoration and RNG retention are supported by the fail-closed production load path and run receipts, not a separate tensor experiment by this report. First128 details and precise source fingerprints remain in `p06_frozen_first_107392.md`.

Actual update chain: `107392/804 → 107520/805 → 107648/806 → 107776/807 → 107904/808 → 108032/809 → 108160/810 → 108288/811`. Each records20 optimizer steps, changed actor and finite nonzero gradients; every source/adjacent actor fingerprint matches, with no broken chain. All recorded update-end effective LRs are1e-5; this does not assert every minibatch used constant LR. Final actor fingerprint is `851b1fbdec25ecfc071da55ea236fc522c152a3628ceba77799853239363fcae`.

Saved pairs at107392/107776/108288 record round-trip=true, actor matching the corresponding actual update, unchanged identity normalizer, exact source runtime and this run's source path. Their suffix counters are47872/48256/48768, full49408 throughout. Final immutable pair is `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000108288.pt` and `_manifest.json`. No mutable later pointer was used to precredit future work.

## Two physical prefixes, zero teacher PPO credit

Both attempts are accepted, miss=null, actual/requestedP06 still active at credit start. Each executes448 zero-raw teacher/takeover decisions and3584 physics ticks, opens at29.8666666667s with170.1333333333s remaining, and has no fallback. All prefix records declare policy_credit=false; all7168 prefix ticks verify and all prefix no-state-write evidence is complete.

| Stream | Decisions | Physics ticks |
|---|---:|---:|
| Excluded teacher initialization | 896 | 7168 |
| Actual PPO credit | 1024 | 8185 |
| Same physical cores combined | 1920 | 15353 |

The final telemetry exactly matches these counts. Every credited record has `prefix_teacher_data_in_ppo_storage=false` and raw request matching its applied audit. Source-only teacher P01–P05 and their FR/FL events are not added to the policy phase histogram. All credited source-phase samples are **P06=1024**, P01–P05/P07–P13=0.

## Completed episode and optimized nonterminal tail

| Episode | Credited global window | PPO decisions / ticks | Final physical time | Actual state |
|---|---|---:|---:|---|
| 0 | 107265–107865 | 601 /4801 | tick8385 /69.875s | P06 age40.0083333, INCOMPLETE_CONTROLLER_BLOCKED |
| 1 tail | 107866–108288 | 423 /3384 | tick6968 /58.0666667s | P06 age28.2, terminal=false/reason=null |

Episode0's final decision is1 physics tick, so `1024×8−7=8185`; it is a real task deadline, not a missing dispatch. Public rear_approach remains0 at both endpoints. The completed episode is physically valid with null physical-failure reason; no collision/wheel-only/safety event is relabelled. It receives absorbing nextPhi0 and no bootstrap. All1023 nonterminal decisions retain bootstrap and all1024 time_outs flags are false. The423-tail decisions have already participated in completed updates; administrative stopping neither discards them nor invents a second task failure.

RR/RL hard Q/C/P are absent in both terminal/tail histories and all read decision endpoints. FR Q71/C1665/P1695 and FL Q2461/C3115/P3583 are retained from each real teacher prefix. RR's initial-clearance bit appears at1 endpoint in episode0 and4 in episode1; these are endpoint counts, not counted lift attempts, qualifications or crossings. RL's initial bit never appears in this fixed decision set.

Closest sampled rear fronts remain outside workspace: episode0 RR−.443071593/RL−.412201140m; episode1 RR−.443473619/RL−.450031270m. These are decision-end extrema, not continuous120Hz geometric extrema. P06 never completes before either stopping boundary.

| Endpoint | Leg | Current contact / support | Load | Front (m) | Clearance (m) |
|---|---|---|---:|---:|---:|
| Ep0 terminal | FL | AIR / false | 0 | +.010608080 | +.027621274 |
| Ep0 terminal | FR | GROUND / true | .474690755 | −.089766618 | −.050457065 |
| Ep0 terminal | RR | GROUND / true | .019982036 | −.616933395 | −.050206042 |
| Ep0 terminal | RL | GROUND / true | .505327210 | −.457951647 | −.049270274 |
| Ep1 tail | FL | AIR / false | 0 | −.008318480 | +.061886956 |
| Ep1 tail | FR | GROUND / true | .401007866 | −.048531201 | −.050225725 |
| Ep1 tail | RR | GROUND / true | .090074168 | −.591817241 | −.049859668 |
| Ep1 tail | RL | GROUND / true | .508917967 | −.479690552 | −.050023248 |

FL is support at8/AIR593 sampled endpoints in episode0 and support8/AIR415 in episode1. Historical placed=true for FL/FR is not current TOP capture/support. Neither endpoint provides rear qualification, and this report does not reinterpret initial-clearance hints as hard success.

## Native continuity and visible control authority

All8185 credited ticks verify, have actual-native effect and own-phase-request effect. All1024 decision-end previous-ACK reference checks and their clock relations pass; all physical evaluators are valid, raw requests match and all four forbidden state-write types are0. Detailed per-tick reference-use counts are not available in compact training logs and are not inferred. Prefix and credit clocks remain separate in accounting but continuous inside each physical episode; new full resets occur only between tasks.

At both endpoint rows, nominal wheels are `[.3,.3,.3,.3]`, but residuals remain active. Ep0 terminal residuals are `[−.226530443,−.763073697,−.203983618,+.263654234]`, producing canonical wheel targets `[+.073469557,−.463073697,+.096016382,+.563654234]`; ep1 tail residuals are `[−.409038664,−.356877179,−.101620335,+.413864152]`, with targets `[−.109038664,−.056877179,+.198379665,+.713864152]`. These readings show policy influence and opposition/assistance to nominal, not a causal proof that one component caused the incomplete approach or a reason to clear residuals.

The final telemetry has one completed task, zero task/full/suffix successes and one unfinished task tail. The first actual missing work remains P06 rear workspace/approach; no P07+ PPO coverage or successful checkpoint/video is claimed. The planned reallocation is a curriculum choice at a verified update boundary, not a newly imposed execution gate.

# Explicit role-observation append: verified recovery boundary

Production commit: `db991aa68103642c179e130fe4cce6ee892c89ab` (after continuous-role implementation `2f27c6f5065aee6fe7b17f165a876e6dc5307841`). This revision does not change physics, action authority, reward, phase conditions or deadlines.

## Why the existing324 inputs were insufficient

The original phase-progress scalar and aggregate potential do not uniquely expose all four new role outputs. A concrete P05 counterexample has FL Q+C but no placement, identical current geometry/hard bits/phase progress/total potential, yet different short support/response history and therefore different motion-fraction-dependent body costs. The explicit append exposes these consumed summaries. It is not a claim that all sliding-window state is reconstructible, that prior on-policy samples were invalid, or that this alias uniquely caused an observed failure.

## Exact observation and network treatment

The first324 columns, their order and scales are unchanged from commit2f27c6f. The earlier4-to6 scale compensation at columns210/222 is **not repeated**. Append48 at columns324–371: FL, FR, RL, RR each receive valid, workspace/preparation/transfer progress, motion fraction, preparation/transfer readiness, fixed direction XY, and short-support/response/window evidence fractions. Missing evidence is explicitly invalid, not invented support. The layout is versioned `diagonal_transfer_state_v1`; unmarked/mixed schemas are rejected.

Both actor and critic first-layer matrices expand from256×324 to256×372. All original columns and all other tensors are copied, new48 columns are zero initialized, learned heteroscedastic standard deviation and history kernel remain unchanged. Action dimension12, previous-raw slice195:207, rho0.9, gamma0.9985 and lambda0.99 remain fixed. This is not a network restart.

## Actual CUDA migration and first update

- Immutable source: `checkpoints/history/checkpoint_step_000130304.pt`,130304 decisions /983 PPO updates /19660 optimizer steps, SHA256 `b0b36465e3ccb52017fb8e3e803517eebbe6f1d8aad2a78da110bba0da7fdb89`.
- Run: `20260908T0230082471288Z_gdb991aa68103_b9b719b139e44396ab233876cbdbf87b`, real CUDA Isaac, natural P01, seed1001, N1, requested1536 fresh decisions.
- Official old-layout source runner verifies source actor/critic/Adam/identity normalizer and RNG. Source Adam is not installed into differently shaped target parameters; target starts fresh Adam at3e-5. Complete training RNG and lifetime/spent-budget counters are preserved. No old or partial rollout is inherited.
- Actual initial same-state mean maximum difference7.450580596923828e-9, standard-deviation difference0, value difference0. Deterministic projected action difference is0 on all12 channels. These are floating-point functional checks, not a guarantee of identical future physical trajectories.
- First completed new update:130432 decisions /984 PPO updates /19680 optimizer steps; finite nonzero gradients and changed actor. Published `checkpoint_step_000130432.pt` passed save/load round trip, SHA256 `733a097c95a3fd461f2cf081d9650ab781ebf45da687c18a93abb9d36d604a39`.
- This document records the first verified boundary. The authoritative last-pointer may already be newer; never restore130432 over a later valid checkpoint. Normal372 continuation retains its current Adam/normalizer/RNG and does not repeat NewMdpWarmStart.

## Verification and scope

Integrated CPU regression:618 passed,0 failures/errors/skips,67.846s, `C:/robotics_sim/wlr_robot/role_append_integration_trial02.xml`. Includes compatible tensor mapping, new-column gradients/update, normal372 resume, old324 compatibility, strict contracts/selected-configuration binding, policy/prefix/CLI, phase/role/reward/return tests. CPU tests are not substitutes for CUDA RNG restoration; the real CUDA migration above supplies that evidence. The initial failed integration receipt is preserved separately.

No new complete P01 evaluation or success video is implied by the migration/update. The last formal outcomes before this revision remain C1 P02 incomplete and C2 P05 incomplete (0/2). Historical A remains P10 incomplete and unpaired.

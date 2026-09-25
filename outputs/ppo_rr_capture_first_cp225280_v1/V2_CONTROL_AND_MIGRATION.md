# RR-first continuation: narrow v2 change record

Runtime: `5f8487b76f27eb165a329a0b6f3096f54ebb4d48`.
This is the existing repository/branch, not a rebuilt training stack.

## What remains protected

The packaged frozen CP225280 actor executes the actual feedback prefix through
RR lift/cross and the measured capture activation gate. It is not a prerecorded
trajectory or a checkpoint switch. Before that gate, local mean and additional
exploration contribute exactly zero. Frozen prior parameters, buffers and
Identity normalizer are excluded from Adam; the separate critic cannot update
them. The accepted nominal/mapper/HISTORY/FL assist configuration is retained.

After activation all12 local channels remain available, including the other
legs and wheels. Neither qualification loss nor a first TOP sample disables
the local correction. The raw prior/local means are composed before exactly
one HISTORY operation; actual Gaussian samples and their likelihood enter PPO.
No RR automatic descent, rear geometry helper or forced wheel direction exists
in the formal path. Existing FL hip-only capture assist remains ON and labelled.

## Confirmed issues and changes

| Evidence | Narrow implementation |
|---|---|
| Synthetic GROUND→unqualified TOP could reuse old placed/crossed in the local-v1 hold test. Actual first stochastic success did not exploit it. | Append explicit same-attempt eligibility, require it for local contact credit/hold/success; retain literal sensor TOP/bearing and recovery control. |
| P09 late targets can begin after short RR bearing and continue after losing it. | Keep its unconsumed FL/RL four-servo+FL-wheel event pending during this local capture task. Preserve earlier carry, wheel stops, P10 RR knee and P11 FR preparation. |
| Episode5 actually began P12 at RR hold0.0167s; RL then unloaded before RR lost contact at0.4833s. This does not prove single-cause failure. | Keep only a not-yet-started new P12 unload pending; preserve an already qualified RL swing and already-started stop clock. |

These are source/task changes, not learned gains. The pending source adapter is
for a task ending at RR capture/hold. Do not claim that it already provides a
full RR→RL continuation controller. Policy may still move any12 channels; this
is not a per-leg lock or proof that every sampled motion is useful.

## Migration and credit

The old2048/4/80 package and its manifests remain immutable. V2 appends one
input column to the local actor and critic:447→448; the prior stays439 and its
original422 columns remain unchanged. Old weights, learned heads, Adam steps,
moments and actual LR2.25e-5 are preserved. Only the appended input/Adam columns
start at zero. Identity normalizers and full saved RNG are preserved; no old
unfinished rollout is reused. Actual CUDA fixed-input mean/log-sigma/critic
comparison errors were0 at migration, before new learning.

The migrated package has zero v2 PPO credit. Its filename includes the runtime
revision; later training has separate `task_v2_policy_decisions` and
`task_v2_ppo_updates` in addition to cumulative local counters. Prior225280,
real uncredited prefix decisions, local PPO and AUX are separate ledgers.

The first70e startup failed during controller construction with no decision or
update. Its overly restrictive carry-mode assertion was corrected without
disabling the accepted carry mode; the failure files are retained. Actual
factory tests now cover this configuration.85 directed tests cover the original
81 cases plus4 production-source compatibility cases; these are wiring/math
evidence, not physical success.

## Learning evidence so far

The first2048 active samples produced4 PPO updates/80 Adam steps and one true
stochastic RR local hold before the first update. This is not deterministic
learning success. Later mean changes did not retain the useful knee-positive
direction;67.6% of rows were AIR recovery after contact. See
`RR_learning_signal_first2048.md`. The v2 run collects fresh512 active samples
before a reloaded deterministic naturalP01 video. No AUX or mean/sigma reset is
being substituted for this new on-policy data.

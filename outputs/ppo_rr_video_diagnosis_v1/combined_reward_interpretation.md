# B0 versus C0: recorded reward at the same physical time

## Scope and verification

Both reward prefixes cover **122 returned decisions / 976 physics ticks / 8.133333333333333 s**, starting at natural P01. These are signed undiscounted sums of the actual recorded reward components, not a new learning run or a counterfactual trajectory. The root reviewed `extract_reward_windows.mjs` before the C0 and common-time B0 extraction. Family-sum error is zero and maximum component reconstruction error is 2.7755575615628914e-17 in both outputs; reward aliases agree.

- B0: `B0_reward_0_976ticks.json` / `.md`.
- C0: `C0_reward_0_20s_bound.json` / `.md`: the requested upper bound was 20 s, but C0 stopped earlier. Its unreturned decision 123 spans tick 976–983 and has no recorded reward. These seven ticks are excluded from both compared reward prefixes. **No terminal -40 was fabricated.**
- Physical endpoint values below come only from the six tick 0/976/983 records in the already generated `rr_bounded_before.records.json`, not a new replay or an original-stream rescan. Calibrated attitude is labeled there `derived_same_tick_using_SHA256_verified_recorded_fixed_calibration`.

## Exact common-time signed comparison

| Recorded component | B0 zero residual | C0 loaded PPO | C0 minus B0 |
| --- | ---: | ---: | ---: |
| Potential shaping | +0.532875632 | +0.293800748 | -0.239074884 |
| Time | -0.162666667 | -0.162666667 | 0 |
| Body attitude | -0.038108920 | -0.010821611 | +0.027287310 |
| Euler roll/pitch rates | -0.000542194 | -0.000475790 | +0.000066404 |
| Angular acceleration | -0.006501020 | -0.005938633 | +0.000562387 |
| Contact quality | -0.000011567 | -0.000020098 | -0.000008531 |
| Applied first difference | -0.010348619 | -0.014740015 | -0.004391396 |
| Applied second difference | -0.016778141 | -0.022411975 | -0.005633835 |
| Residual regularization | 0 | 0 | 0 |
| Recorded terminal event in compared prefix | 0 | 0 | 0 |
| **Total** | **+0.297918504** | **+0.076725958** | **-0.221192545** |

C0's integrated **weighted attitude penalty is 71.6034705% smaller**. All body costs together give C0 a +0.027916100 relative advantage. But its potential shaping is 0.239074884 worse and applied smoothness costs 0.010025231 more. Consequently the combined recorded reward **does not prefer C0** on this aligned prefix. The smaller level penalty does not overwhelm the task-progress loss in these data.

The attitude integral includes the existing physical-transfer weight and clipped normalization; it is not raw roll/pitch RMS. B0 endpoint `f` ranges 0.222222222–1 (weight 0.2–0.822222222); C0 endpoint `f` ranges 0.875–1 (weight 0.2–0.3). These endpoint ranges cannot quantify exactly how much of the integrated attitude difference came from physical angles versus within-decision weighting. No such attribution is invented.

## Lower tilt cost coexists with lower body/CoM and worse clearance

Both reset records are identical for the listed quantities: base-origin z=0.09901974350214005 m, CoM z=0.15462384420150221 m, calibrated roll/pitch=(-0.00003884937598026426,-0.012019878918690496) rad.

| Tick 976 measurement | B0 | C0 |
| --- | ---: | ---: |
| Base-origin z, m | 0.0728161633014679 | 0.05084144324064255 |
| Mass-weighted CoM z, m | 0.16463154098721813 | 0.1167990134517 |
| Calibrated roll, rad | -0.23742317573637434 | +0.07427248953860635 |
| Calibrated pitch, rad | -0.1896062670650352 | -0.11106249988982593 |
| Actual body angular speed, rad/s, decision endpoint | 0.01479370374725722 | 0.2023275873157821 |
| FR bottom-to-top gap, m | +0.09913702309190804 | -0.013606880156445983 |
| FR front distance, m | -0.06812774794268961 | -0.12690177219081278 |

At tick 976, C0 base origin is 21.9747201 mm lower than B0 and its CoM is 47.8325275 mm lower. Relative to its identical reset, C0 base/CoM fell 48.1783003/37.8248307 mm. B0 CoM instead rose 10.0076968 mm. C0's final roll/pitch magnitudes are smaller, but its angular speed is larger and FR is lower and farther from the edge. Thus **more level is not equivalent to more stable or closer to finishing the task**. Base-origin height is not a base-obstacle clearance measurement and must not be relabeled as a collision proof.

The C0 physical stop is seven ticks later: at 983, base z=0.050464555621147156 m, CoM z=0.11601119924216335 m. B0 at that same tick is 0.0727531686425209/0.1646053483032719 m. The root independently identified C0's stop as P02 SAFETY_ABORT/HARD_JOINT_LIMIT; the reward-prefix extractor records the actual `SAFETY_ABORT` callback reason and does not infer a missing last-step reward.

Both common-prefix endpoints have FR I/Q/AIR=true but no FR C/P. B0 I/Q occur at ticks 15/24; C0 at 17/26. B0 later earns FR C/P at 1486/1502, outside this common-time comparison; C0's early termination is not compared with B0's later success segment as if both had equal duration.

## What the evidence does and does not justify

There is a **component-level tension**: the attitude term charges less for C0 even as its body/CoM and FR clearance drop. This makes level-only stability reasoning inappropriate and provides a measured reason to inspect the interaction between body-motion allowance, retained clearance and progress. It does **not** prove the entire reward favors sinking, nor that the actor learned to sink to collect reward. Its common-time total is worse; in the last returned decision (968–976), C0 potential shaping is already -0.010801677118644631 and total reward -0.012434155309735805, while B0 is +0.01285654863237895 and +0.010955608805409705 respectively.

All 122 C0 raw actions are nonzero; B0 has zero throughout. Neither prefix records endpoint servo headroom clipping (0/122 each). This does not prove absence of intra-decision clipping or actual motor-force saturation; neither is available from these decision fields. No residual-magnitude regularizer or extra imitation term explains the result.

These runs evaluate a frozen actor, with zero optimizer updates. Reward-only bookkeeping changes in evaluation cannot alter that actor's outputs, and this offline extraction is not PPO learning. Any actual reward/control repair must be independently justified, versioned and followed by fresh on-policy collection; these data alone do not authorize a speculative gain, torque, gate or weight change.

All source videos, checkpoints and production files remain unchanged. Overlapping event windows in the individual reports must not be summed into another episode total.

# Isolated FR-bearing geometry diagnostic — NOT EXECUTED

Prepared only. Sole active formal Isaac episode remains untouched. No production files, model parameters, reward, masks, HISTORY semantics or physical capacity changed. Stdlib tests do not prove physical improvement.

## One question, one channel

The sealed CP229120/CP225280 age2/4/6 table shows identical source commands but about +4.2 degrees higher FR knee FINAL in the failing run, plus increasing FR hip actual-minus-target error under measured TOP load. This candidate tests the **direction** of a modest FR knee reduction, not the causal sufficiency of FR knee, not a fixed FR posture and not a global front repair.

Use **CP230144**, SHA `376ab4b6fe6f2a6a82bedd1190ba0814dc5e5b396127c48c34cf7b1d4a8265f1`, sidecar `30dc90ece977823762eb7f19b79b944f41b20e14c5231c45014ab58d17db3ba9`, exact HEAD `59e868f3e223c589e7645a0f5d63f91fa6119fb6`. Same seed4001, deterministic conditional mean, natural P01 prefix under this same checkpoint. Official video checkpoint loader resolves collection512 and performs ordinary lineage/load verification; there is no new training runner, optimizer or migration. Original FL helper remains ON, all rear task helpers OFF, all12 channels permitted.

At the first decision in **P05 age0.5–0.75s** with current FR legal TOP bearing and current RL bearing, freeze `bridge.previous_projected_residual_full12[3]` as the entry REQUEST. Verify it against the actor's logged filtered-request observation. Command only FR knee raw through inverse-tanh of:

- 1s smoothstep from that entry REQUEST to **entry minus4 degrees**;
- 1s hold that same fixed entry-minus4 REQUEST;
- 1s smoothstep blend to the **current** student REQUEST, then cease intervention.

The four degrees are a one-time physical-unit REQUEST delta, never subtracted repeatedly and never a promise of a four-degree FINAL or actual joint change. The remaining11 raw values are exactly the current student values. Current caps, source N, mapper, rate limits, headroom, final slew and one-write dispatch remain authoritative. If entry-minus4 lies outside the current cap, do not trigger; no saturation-based substitute is introduced.

First loss of support/phase eligibility is latched at120Hz; no next modified decision is issued. The already-running immutable eight-tick action can have up to7 ticks remaining; ordinary safety may end it earlier. No mid-decision source or raw mutation is used. No retrigger occurs.

Actual intervened raw and projected REQUEST naturally enter ordinary HISTORY. Subsequent student outputs therefore condition on the **real intervened trajectory**; do not call them the untouched-baseline actions. Return of channel ownership does not erase history or guarantee instant restoration of baseline targets. Original raw, original Gaussian density, applied raw, effective projection and native ACK are recorded separately. Applied actions have **no PPO log-probability/storage credit**. No teacher model is loaded, and no diagnostic row becomes an AUX target.

## Reviewed launch candidate — root must decide after the active episode exits

```powershell
& 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe' `
  'outputs/ppo_rl_recovery_learning_v1/fr_knee_entry_offset_probe.py' `
  --run-dir 'runs/ppo_rr_rl_timing_policy_learning_v1/direction_probe/CP230144_FRk_entry_minus4_v1' `
  --max-decisions 900 `
  --execute-reviewed-real-diagnostic
```

Default900 decisions = at most60simseconds; a smaller positive budget is accepted. An ordinary physical terminal ends earlier. Continue after the three-second intervention to observe FL clearance, forward approach and capture; any later event is still a **declared-intervention diagnostic**, never formal learned-policy success. No camera/video is enabled by this diagnostic, matching the existing direction-probe path; its deliverable is synchronized physical/control evidence. The formal continuous video is a separate preserved run.

Expected readout: source and mapped N, raw original/applied, FR requested/effective/FINAL/actual knee; FR hip FINAL/actual and current load; all four wheel FINAL/actual; FL gap/front/contact; CoM/base and current support at matched P05 ages2/4/6s. Verify the intended offset reached FINAL before attributing changed body geometry. A lack of improvement is evidence against this particular direction/entry/duration, not proof of a unique alternative.

## Focused non-physical checks

`test_fr_knee_entry_offset_probe.py` covers zero-delta bitwise identity, both request directions, onlychannel3, fixed entry/no accumulation, smooth ramp/hold/release, support-loss latch/no retrigger, stale/absent support, current cap rejection, early-window-only trigger, no automatic execution/no Torch import, official collection512 loader and zero-credit declarations. These are candidate wiring checks only; actual checkpoint loading and floating-body response remain unexecuted.

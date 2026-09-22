# First formal training block: completed FR prefix only

Run: `runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T0459512815083Z_gee5a9651591d_8e120f68dc7c4598b08dc79fcd3d34a4`. Only complete written decisions 185857–186060 were read, stopping at the first recorded FR placement. FR active lift=19, cross=1611, placed=1628 / 13.5667 s. The containing final decision ends at tick 1632, four ticks after the physical event. No P05 or whole-block outcome is inferred.

This is a **stochastic learning rollout, not fixed-checkpoint evaluation**. PPO update 1418 occurs after decision 185984 / physics tick 1024, and its receipt confirms actor parameters changed. The FR prefix therefore includes actions from before and after an optimizer update. Successful zero B is only a descriptive task-window reference; this is not a paired deployment-performance claim.

The 204 audited decisions all declare `training_style_conditional_gaussian`, all actual residual permission masks are full 12-channel ones, and all native dispatch receipts verify. Each of the 12 physical REQUEST channels is nonzero in all 204 decisions; this is not the earlier wheel-residual-off diagnostic. These checks establish live dispatch, not perfect actuator tracking.

## New reward is actually active

All consumed rows use `task_conditioned_hip_wheel_quality_v1`, epsilon/body-family weight 0.06. The original front tilt/rate contribution remains effectively 0.03→0.015 per second; its 1599 real P01/P02 physics samples produce total weighted cost 0.0167915697. Actual body-family reward through the containing decision is -0.0167915697. Per-decision decomposition into front tilt, front rate and geometry has maximum absolute error 1.08e-19.

The new obstacle-geometry component is evaluated in all 1599 eligible samples with coefficient 0.03/s; every sample has valid live collider bounds and obstacle planes. It has **zero actual penalty** because the minimum conservative separation is 261.124 mm, well above the 20 mm margin. The recomputed clipped-square cost exactly matches the saved audit. This proves the path is active, not that a nonzero geometry gradient has already improved this FR rollout.

P03 is outside the declared front and geometry quality windows. Its 29 physical ticks from 1600 through capture at 1628 explicitly contain outside-window/null geometry audits and no front tilt/rate audit. Consequently full P01→FR-capture 120 Hz RMS and exact whole-window geometry minimum are left null; those samples are not filled with zero or called sensor failures. The full-prefix FR leg gap and later collider bounds are available only at decision endpoints in this training log. Four-hip installation heights are not recorded here and are not invented.

## Comparable P01/P02 quality windows

Each column spans its own complete P01/P02 physics interval, ending immediately before P03. These are distinct-duration task windows, not matched fixed-time trajectories. Rate RMS is `sqrt(time_mean((roll_rate^2 + pitch_rate^2)/2))`; tilt is `hypot(roll,pitch)`. Collider values are real collision bounds, never base-origin height.

| Metric | Learning rollout | Successful zero B |
|---|---:|---:|
| 120 Hz intervals | 1599 | 1471 |
| P01/P02 duration, s | 13.3250 | 12.2583 |
| Body rate RMS, rad/s | 0.082128 | 0.081015 |
| Peak tilt, rad | 0.303750 | 0.309464 |
| Roll RMS, rad | 0.204707 | 0.232688 |
| Pitch RMS, rad | 0.176495 | 0.184428 |
| Collider minimum world z, min / mean, mm | 89.269 / 94.509 | 91.272 / 94.274 |
| Conservative body-obstacle separation, min, mm | 261.124 | 257.062 |
| Weighted front quality cost | 0.016792 | 0.018082 |
| Real FR placement time, s | 13.5667 | 12.5167 |

The rollout has slightly lower peak tilt (-1.85%) but slightly higher rate RMS (+1.37%), a collider minimum approximately 2.00 mm lower, and FR placement 1.05 s later. A smaller angle or cost alone therefore does not establish better task performance or uniform clearance improvement.

For the common 15 Hz gap diagnostic, start at each run's first measured AIR endpoint with at least 15 mm top-plane gap and stop before its FR crossing. The rollout starts at tick 64 and has minimum/mean gap 23.457/88.893 mm across 194 endpoints; B starts at tick 72 and has 27.458/98.940 mm across 177 endpoints. Thus sampled gap is retained but generally smaller. Exact 120 Hz FR gap is unavailable in the current training log; the source B has a separate dense raw stream, which is not used to fabricate the missing training counterpart.

Evidence: `block01_FR_quality_audit.json`; reproducible read-only script: `training_fr_readonly.py`. For later video selection, this prefix is useful as a completed FR example with real full-channel stochastic learning and auditable reward, but must retain its learning-rollout label. No production code, active config, model or running simulator was modified by this analysis.

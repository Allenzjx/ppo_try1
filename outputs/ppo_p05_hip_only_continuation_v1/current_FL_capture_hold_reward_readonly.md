# Current FL approach, capture and rolling-retention reward: read-only check

Scope: frozen runtime `0001c3138b0b38a7278edc288b8ddc7444815770`; current `p05_hip_only_continuation_v1` reward/task specifications; episode 1 of training run `20260922T0415552437349Z_g0001c3138b0b_7b34b44b4b9d49ddb087585dec7acef6`; the new deterministic video run `20260922T0349194627694Z_g0001c3138b0b_608ca7ebf2074e70af33a21f8423fb00`. No runtime edits, configuration changes, or physics launched for this check.

## Conclusion

FL is **not ignored after its first real placement**. Loss of actual bearing during the early P06 rolling window reduces the active physical task potential, even though `contact_motion_quality` has weight zero. Both the current stochastic training episode and deterministic video show the resulting negative reward. No additional reward term is proposed as a missing-signal fix.

Historical `placed_FL` remains true. Current support, TOP/AIR state, gap, and retention remain separate measured quantities; history does not invent current bearing.

## Production computation

Source: `src/wlr50_clean/ppo/semantic_supervisor.py`, `physical_potential` (line 1228), `_current_capture_progress` (1333), `_current_capture_retention` (1366); `src/wlr50_clean/ppo/semantic_reward.py` (436–453).

Before real FL placement, qualified lift and edge crossing permit a capture share based on actual legal-XY approach and consecutive TOP observations:

`capture_FL = 0.5 * proximity + 0.5 * clipped_consecutive_TOP_fraction`

`proximity = 0.5 / (1 + positive_gap / 0.025) + 0.5 / (1 + positive_gap / 0.003)`

Outside the legal top/lateral region the descent proximity credit is zero. This shaping does not itself set `placed_FL`, reward a specified hip angle, or require a static posture. Existing workspace/carry/transfer shares continue to depend on physical geometry. Later forward preparation can increase rear-leg workspace/transfer progress; there is no blanket forward-speed or fixed-wheel-command reward.

After placement, the FL contribution to global potential is:

`Phi_FL = (0.85 / 4) * (0.8 + 0.2 * R_FL)`

`R_FL = G * ((1 - w) + w * U)`

`G = min(clipped_platform_XY_retention, current_below_top_penetration_retention)`

`w = clip((-0.22 - max(RR.front_distance, RL.front_distance)) / 0.05, 0, 1)`

`U = 0.5 / (1 + max(FL.clearance, 0) / 0.003) + 0.5 * actual_bearing_indicator`

The indicator requires **current** TOP contact, top-surface contact, support, verified bearing, not AIR, and not ground contact. It is not a force-magnitude reward. The rolling term applies only after both fronts really placed and before RR is currently qualified-lifted or already placed; otherwise retention is just `G`.

The active reward consumes this potential as:

`task_reward = 5 * (0.9985 * Phi_after - Phi_before) + terminal_event - 0.02 * elapsed_seconds`

`task_progress` weight is 1.0. Thus the variable FL-retention component alone contributes `0.2125 * (0.9985 * R_after - R_before)` when the placed status is unchanged. The separate contact-quality family is zero-weighted; it does not switch off this task-potential signal.

## Actual decision-endpoint evidence

Ticks are episode-local 120 Hz physics ticks; gap is the recorded wheel clearance relative to platform top, in mm. The first stochastic episode ended before FL; this table uses episode 1 (the next episode), not suffix teacher credit.

| Run / tick | Phase | FL current state | Gap mm | R_FL | Total potential shaping | Actual reward |
|---|---|---|---:|---:|---:|---:|
| Stochastic ep1 / 3208 | P06 | TOP, actual support; real placed event at 3205 | -0.493044 | 1.000000 | +0.136687 | +0.135354 |
| Stochastic ep1 / 3312 | P06 | TOP, actual support | -0.001124 | 1.000000 | -0.005115 | -0.006448 |
| Stochastic ep1 / 3320 | P06 | AIR, historical placement retained | +1.023091 | 0.372848 | -0.142731 | -0.144064 |
| Deterministic / 3792 | P06 | TOP; real placed event at 3789 | -0.105432 | 1.000000 | +0.126015 | +0.124682 |
| Deterministic / 3800 | P06 | AIR, historical placement retained | +0.157809 | 0.475013 | -0.101599 | -0.102932 |

For stochastic 3312→3320, the FL retention component is **-0.133389**, not the entire -0.142731 potential change. For deterministic 3792→3800 it is **-0.111711**, not the entire -0.101599 potential change. Other physical-potential components also change.

Further stochastic losses 3384→3392 and 3456→3464 produce FL retention contributions -0.107747 and -0.106409 respectively. All 242 P06 decision endpoints have zero weighted contact-quality cost; the sum of the P06 collider-geometry cost is also zero. Consequently the demonstrated bearing-loss signal is the active task potential, not a hidden contact penalty or body-geometry penalty.

Evidence comes from the already-written `residual_and_projection_audit.jsonl` per-decision `applied_audit` fields (physical evaluator, semantic task and reward). `R_FL` was recomputed read-only using the production retention helper and each recorded evaluator state. These are observed rewards, not a claim about the sign of the eventual GAE or policy gradient.

## Release during rear preparation: important qualification

There is no permanent P07-label-based exemption. The rolling-contact weight retires on current rear geometry or current qualified RR lift/real RR placement. In stochastic ep1:

- P06 tick 5104: rear distance -219.261 mm, so `w=0`, even though FL is AIR with 53.610 mm gap.
- P07/P08/P09 entry endpoints 5144/5152/5160: `w=0`; the needed FL release in that approach receives no rolling-bearing loss from this term.
- P09 tick 5888: rear retreats to -220.197 mm while RR is not currently qualified; `w` reactivates to only 0.003948. At 5984, rear is -219.842 mm and `w=0` again. The total shaping -0.083113 at 5888 must **not** be attributed to the tiny reactivated FL term.

This is a current physical-context signal, not a requirement to keep FL on the platform throughout every rear task. Reactivation after retreat is observable behavior; whether it conflicts with a useful recovery trajectory should be assessed after the running block and reloaded full evaluation, not assumed from a phase label or changed mid-block.

## What this does and does not establish

Approach, real capture and early rolling retention all have learning signals. Holding an already-good state need not give a positive reward every tick: discounting and elapsed-time cost remain. Contact loss reduces potential and recapture can restore it; a brief loss/recovery may have largely offsetting potential-based shaping. With the contact-quality family at zero there is no additional persistent duration-integrated contact-loss cost from that family.

These observations establish reward consumption, not that the present policy has learned to hold or that a reward change alone would repair frozen actions. Continue the current authorized update block and reload/evaluate before considering any targeted objective change. No new gate, fixed pose, all-feet-static requirement, or additional contact penalty is recommended by this missing-signal check.

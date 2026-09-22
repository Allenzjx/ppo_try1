# CP192512 → immediate aux → CP194560: fixed-real-state comparison

**The sampled fixed-state FL direction was retained and strengthened by the
subsequent2048 real PPO decisions, not cancelled or only absorbed by sigma.**
This is direct actor-mean evidence, not joint log-probability attribution. It
does not by itself establish FL capture or natural-trajectory success/safety.

CPU-only, zero optimizer/policy draws. Reused the original75 P05 AIR approach
inputs,24 P01–P04 holdouts and26 late states (P06=8/P08=2/P09=8/P12=8).
All125 complete372 observations, encoded real HISTORY and caps were identical
across the three forwards. No zero-H reconstruction or new PPO buffer data.
Immediate aux had7 accepted/8 attempted independent steps; CP194560 then added
2048 actual PPO decisions /16 updates /320 Adam steps. The aux ledger persists.

## FL: means over the75 P05 approach states

| Quantity | CP192512 | Immediate aux | CP194560 |
|---|---:|---:|---:|
| Network mean, raw | +.115834 | -.002673 | -.124343 |
| Conditional mean, raw | -.142445 | -.154296 | -.166463 |
| REQUEST, degrees | -2.539786 | -2.747333 | -2.959649 |
| Effective raw sigma | .123578 | .123578 | .131031 |

Aux changed mean REQUEST by-.207547°, then PPO by another-.212316° (total
-.419863°). On the63 hold states with fixed H=-.20039523, network means were
+.118925 → +.000519 → -.120922, conditional means -.168463 → -.180304 →
-.192448, REQUEST -3.003973 → -3.210750 → -3.421921°. Thus the newer negative
network direction is not merely inherited negative HISTORY. Whether it forms
the right future HISTORY/support configuration remains a physical question.

## Full12 PPO changes: CP194560 minus immediate aux,75 P05 states

| Channel | Mean conditional raw μ change | Mean sigma ratio |
|---|---:|---:|
| FL hip | -.012167 | 1.06044 |
| FL knee | -.005747 | 1.01612 |
| FR hip | -.003253 | .86481 |
| FR knee | -.001781 | 1.00380 |
| RL hip | -.010763 | 1.09721 |
| RL knee | +.002479 | 1.02220 |
| RR hip | +.003217 | .87028 |
| RR knee | -.006336 | .91800 |
| FL wheel | -.013513 | .99469 |
| FR wheel | -.006593 | .95427 |
| RL wheel | -.001626 | 1.05399 |
| RR wheel | +.007372 | .93621 |

In physical REQUEST units the larger non-FL-hip joint mean changes are FL knee
-.12331°, RL hip-.11074°, RR knee-.10348°. Wheel mean changes are
FL-.013498,FR-.003947,RL-.001537,RR+.004083 rad/s. These are requested residual
changes, not actual actuator motion. Immediate aux itself left other11 means
and all12 sigmas exactly unchanged on all125 inputs; these additional changes
come from the subsequent PPO checkpoint difference.

## Shared-row effects outside P05

| Fixed cohort | n | Aux FL REQUEST change | Further PPO change | Total vs CP192512 |
|---|---:|---:|---:|---:|
| P01–P04 holdout | 24 | -.14236° | -.14585° | -.28822° |
| P06 | 8 | -.35162° | -.39640° | -.74803° |
| P08 | 2 | -.37978° | -.41229° | -.79207° |
| P09 | 8 | -.35572° | -.41166° | -.76738° |
| P12 | 8 | -.30266° | -.34790° | -.65055° |

The largest total absolute late-state FL REQUEST difference is .835061°;
front-holdout maximum is .403743°. The aux-only .25° front-holdout bound was not
a bound on subsequent unrestricted full12 PPO or on other phases. These numbers
are neither a new gate nor evidence that the shared changes are physically good.
Actual future trajectory effects remain unknown here and are explicitly null
in the JSON. Same-state mean effects are not a causal estimate of how much aux,
versus ordinary PPO learning, caused the later natural behavior.

Complete per-state full12 outputs, source/selection/rollout hashes and statistics:
`CP194560_fixed_state_comparison.json`; reproducible one-shot helper:
`CP194560_fixed_state_comparison.py`. CP194560 SHA256:
`20d327d6c9b62a1b459dad61abbae7181a7b1dcae23ca26a5f51ba63e7399aca`.
Sealed aux modules, production code/configuration and physical runs were untouched.

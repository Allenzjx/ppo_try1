# Sealed P02 four-wheel chain

Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_residual_rr_fix_v1\video_eval\validation\20260917T0510347247858Z_g6c2121b68654_35383f247031424e9b890de765bc2b13\source`

Smoke-only: False; recorded checkpoint decisions: 178432. No new physics/optimizer.

## Actual tick400 (3.333333s)

| Leg | N | mapped N | raw latent | mask | filtered residual | same-state target effect | final canonical / native | measured canonical / native | contact |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| FL | +0.30000 | +0.30000 | +0.08800 | 1 | +0.05267 | +0.05267 | +0.35267 / -0.35267 | +0.45378 / -0.45378 | GROUND |
| FR | +0.30000 | +0.30000 | +0.00981 | 1 | +0.00588 | +0.00588 | +0.30588 / +0.30588 | +0.30808 / +0.30808 | AIR |
| RL | +0.30000 | +0.30000 | +0.08298 | 1 | +0.04967 | +0.04967 | +0.34967 / -0.34967 | +0.42431 / -0.42431 | GROUND |
| RR | +0.30000 | +0.30000 | +0.00607 | 1 | +0.00364 | +0.00364 | +0.30364 / +0.30364 | +0.30958 / +0.30958 | GROUND |

All velocities/corrections are rad/s; raw is dimensionless. Mask acts on additive residual, not N.
RR knee target/actual: -1.71785/-2.13171deg; FR gap +0.104389m; body origin z 0.073995m.

- FL: no listed cancellation/near-zero predicate; not proof of good traction.
- FR: no listed cancellation/near-zero predicate; not proof of good traction.
- RL: no listed cancellation/near-zero predicate; not proof of good traction.
- RR: no listed cancellation/near-zero predicate; not proof of good traction.

## Window and evidence limits

{"P01_tail_start": 1, "P02_first": 17, "P02_last": 1392, "P03_first": 1393, "last_read_included_tick": 1408}

Full held N is actual recorded value. No unique source owner or wheel-specific final writer ID is logged; changed_channels does not reassign owner. Native staged/dispatched audit verifies its named write buffers, not an independent motor-force causal proof.

Projected/filtered residual and same-pre-tick zero-current-policy target delta are both retained. Raw tanh*cap is diagnostic only; no independent N recalculation, history reset, or different-run subtraction is called the current policy effect.

Detailed P02 and handoff-window per-wheel counts, masks, bound source-prefix digests, measured values and actual dispatch checks are in `p02_wheel_chain.json`. Descriptive 0.05/half-N comparisons are not new task thresholds.

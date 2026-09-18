# Sealed P02 four-wheel chain

Source: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_non_residual_refine_v1\video_eval\prior_B\20260917T0424208857504Z_g6c2121b68654_13804a86a302480ab61ad2af0dcb8dcd\source`

Smoke-only: True; recorded checkpoint decisions: None. No new physics/optimizer.

## Actual tick400 (3.333333s)

| Leg | N | mapped N | raw latent | mask | filtered residual | same-state target effect | final canonical / native | measured canonical / native | contact |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| FL | +0.30000 | +0.30000 | +0.00000 | 1 | +0.00000 | -0.00000 | +0.30000 / -0.30000 | +0.24424 / -0.24424 | GROUND |
| FR | +0.30000 | +0.30000 | +0.00000 | 1 | +0.00000 | +0.00000 | +0.30000 / +0.30000 | +0.29813 / +0.29813 | AIR |
| RL | +0.30000 | +0.30000 | +0.00000 | 1 | +0.00000 | -0.00000 | +0.30000 / -0.30000 | +0.34175 / -0.34175 | GROUND |
| RR | +0.30000 | +0.30000 | +0.00000 | 1 | +0.00000 | +0.00000 | +0.30000 / +0.30000 | +0.31376 / +0.31376 | GROUND |

All velocities/corrections are rad/s; raw is dimensionless. Mask acts on additive residual, not N.
RR knee target/actual: +0.00000/-0.60671deg; FR gap +0.100780m; body origin z 0.072502m.

- FL: no listed cancellation/near-zero predicate; not proof of good traction.
- FR: no listed cancellation/near-zero predicate; not proof of good traction.
- RL: no listed cancellation/near-zero predicate; not proof of good traction.
- RR: no listed cancellation/near-zero predicate; not proof of good traction.

## Window and evidence limits

{"P01_tail_start": 1, "P02_first": 17, "P02_last": 1472, "P03_first": 1473, "last_read_included_tick": 1488}

Full held N is actual recorded value. No unique source owner or wheel-specific final writer ID is logged; changed_channels does not reassign owner. Native staged/dispatched audit verifies its named write buffers, not an independent motor-force causal proof.

Projected/filtered residual and same-pre-tick zero-current-policy target delta are both retained. Raw tanh*cap is diagnostic only; no independent N recalculation, history reset, or different-run subtraction is called the current policy effect.

Detailed P02 and handoff-window per-wheel counts, masks, bound source-prefix digests, measured values and actual dispatch checks are in `p02_wheel_chain.json`. Descriptive 0.05/half-N comparisons are not new task thresholds.

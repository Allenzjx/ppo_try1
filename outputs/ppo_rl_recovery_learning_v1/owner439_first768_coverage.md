# Optimizer-completed physical coverage

Actual new decisions / PPO / Adam: 768 / 6 / 120.

| Run | Window | aligned input | endpoint |
| --- | --- | ---: | ---: |
| 20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7 | RR_reachable_AIR_preparation | 48 | 48 |
| 20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7 | RR_actual_bearing_front_preparation | 28 | 28 |
| 20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7 | positive_FR_axis_CoM_body_projection_with_RL_fraction_decline | 22 | 22 |
| 20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7 | qualified_RL_edge_recovery | 0 | 0 |
| 20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7 | RL_qualified_AIR_capture_region | 0 | 0 |
| 20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7 | RL_actual_TOP_bearing | 0 | 0 |
| 20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7 | RL_placed_history | 0 | 0 |
| 20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7 | P13_actual_stage | 0 | 0 |
| 20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7 | all_task_complete | 0 | 0 |

Counts are nonexclusive; missing first-prefix/episode input alignment is explicitly N/A.
Positive local fixed-FR-axis projections of both CoM and body displacement, plus declining RL load fraction while RR and a front leg currently bear. NOT verified lateral/rightward transfer, NOT absolute RL force decline, NOT qualified RL unloading. Forward movement alone can make the projection positive; a load fraction can decline because other legs load more. World xyz components and current absolute forces are reported separately; yaw/body-frame decomposition is not inferred from the projection.
JSON contains original reward/value/standardized advantage examples, exact terminal raw GAE, and per-phase raw-GAE statistics. Nonterminal raw GAE is not guessed.
Nonterminal final rollout boundaries are partial episodes, not failures or successes. No tensor, policy, reward, or runtime was modified.

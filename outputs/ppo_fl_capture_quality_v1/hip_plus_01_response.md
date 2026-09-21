# hip_plus — sealed external FL diagnostic

Started: True; complete: True; original terminal: None.
First obstacle contact: None; placed decision: None; P06 decision: None.

| Window | ticks | FL gap min/mean/max mm | hip target mean / actual mean deg | obstacle-contact samples |
|---|---|---|---|---|
| pre | [2592, 2608] | 22.235 / 40.414 / 60.519 | 26.0375 / 35.4852 | 0 |
| ramp | [2609, 2672] | 2.811 / 8.205 / 20.742 | 23.1512 / 24.0181 | 0 |
| hold | [2673, 2928] | 5.867 / 7.209 / 8.944 | 24.2712 / 24.2186 | 0 |
| release | [2929, 2992] | 4.113 / 5.490 / 6.333 | 24.0143 / 24.0559 | 0 |
| follow | [2993, 3088] | 2.676 / 3.260 / 4.434 | 23.2278 / 23.2600 | 0 |

External controllability intervention, not trained-PPO or formal full success.
Other ten policy channels react on actual observations; not a rigid-base isolated response.
Mapper pre-state, filtered request, headroom-effective offset, final target and physical response are separate.
P06 entry and current contact retention are reported separately from historical placement.

Full raw execution chain and final current support are in the paired JSON.

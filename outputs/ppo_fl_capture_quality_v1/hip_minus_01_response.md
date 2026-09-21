# hip_minus — sealed external FL diagnostic

Started: True; complete: True; original terminal: None.
First obstacle contact: None; placed decision: None; P06 decision: None.

| Window | ticks | FL gap min/mean/max mm | hip target mean / actual mean deg | obstacle-contact samples |
|---|---|---|---|---|
| pre | [2592, 2608] | 22.235 / 40.414 / 60.519 | 26.0375 / 35.4852 | 0 |
| ramp | [2609, 2672] | 0.882 / 4.397 / 20.733 | 21.1370 / 22.5460 | 0 |
| hold | [2673, 2928] | 0.394 / 0.933 / 1.631 | 21.2540 / 21.2056 | 0 |
| release | [2929, 2992] | 0.440 / 1.175 / 2.497 | 21.6084 / 21.4069 | 0 |
| follow | [2993, 3088] | 2.189 / 3.105 / 3.811 | 22.3998 / 22.2629 | 0 |

External controllability intervention, not trained-PPO or formal full success.
Other ten policy channels react on actual observations; not a rigid-base isolated response.
Mapper pre-state, filtered request, headroom-effective offset, final target and physical response are separate.
P06 entry and current contact retention are reported separately from historical placement.

Full raw execution chain and final current support are in the paired JSON.

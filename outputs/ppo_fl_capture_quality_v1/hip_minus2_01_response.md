# hip_minus — sealed external FL diagnostic

Started: True; complete: True; original terminal: None.
First obstacle contact: 2843; placed decision: 2848; P06 decision: 2848.

| Window | ticks | FL gap min/mean/max mm | hip target mean / actual mean deg | obstacle-contact samples |
|---|---|---|---|---|
| pre | [2592, 2608] | 22.235 / 40.414 / 60.519 | 26.0375 / 35.4852 | 0 |
| ramp | [2609, 2672] | -0.338 / 3.800 / 20.732 | 20.8501 / 22.3145 | 0 |
| hold | [2673, 2848] | -0.530 / -0.043 / 0.583 | 20.7397 / 20.7153 | 6 |
| release | [2849, 2912] | -0.914 / 0.268 / 2.457 | 20.3836 / 20.3014 | 42 |
| follow | [2913, 3008] | -1.098 / 3.471 / 10.347 | 21.1868 / 21.0001 | 22 |

External controllability intervention, not trained-PPO or formal full success.
Other ten policy channels react on actual observations; not a rigid-base isolated response.
Mapper pre-state, filtered request, headroom-effective offset, final target and physical response are separate.
P06 entry and current contact retention are reported separately from historical placement.

Full raw execution chain and final current support are in the paired JSON.

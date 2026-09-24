# CP221568 v2 versus CP220544 v7: bounded P01/P02 evidence

Read-only, 2026-09-23. No model forward, Torch, Isaac, production edits, optimizer/AUX updates or new physical evaluation. Old v7 is a verified front reference, not a complete no-assist task success.

Sources:
- Old CP220544 SHA `47fdec0614ed2683a0eae6fdef736598400f3c6833473b551a8c93f6f0d8f430`, runtime `60abc00957c0`, run `runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation/20260923T0934264919029Z_g60abc00957c0_ef405598c8954e6280e844c4c29ed041/source`.
- New CP221568 SHA `1827b5d59935b31e1e27cc7ae0c364dd0203a9d7f77a7d9e9502189bbd31c7d5`. Runtime `44219b4fdc4d`, run `runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260923T2135184159404Z_g44219b4fdc4d_aceeacf1f768481da3b5ba4f71afebc8/source`.
- Read old through tick2394 (19.95s; FR placement), new through terminal tick2692 (22.433333s). Earliest-difference comparison stops tick2368, before stage divergence. No later old rear trajectory or active training logs inspected.

## First difference, not an inferred cause

Initial tick0 physical records are identical except geometry computation wall time. The first dispatched raw request differs at tick1 (0.008333s): maximum raw-channel difference0.004397046 (RL hip). Source and mapped nominal are still identical. The first final-servo difference is0.0692481deg (FL hip); maximum first final-wheel difference0.00143903rad/s (FR). Actual geometry already differs by micrometres after that physics step.

Mapped nominal first differs at tick105/0.875s in FR knee (43.5560874 versus43.5839729deg); subsequent mapper compensation is closed-loop, not a new source event. All12 source commands match exactly through tick2368. First source difference is tick2369 because old has legitimately entered P03 while new remains P02.

At all2368 compared dispatches in each run: residual mask all1, verified native write, no FL capture-assist owner and no RR capture-assist owner. Old RR-wheel envelope is inactive with zero controller delta2368/2368. Rear geometry correction is P09/P12-only; disabling it is not a P02 geometry-off explanation. No P01/P02 source/acceptance change was found in the bounded60abc→44219 control diff.

## RR_WAIT14 migration hypothesis ruled out in this front window

Actual old296 P01/P02 decision receipts and new337 receipts all have:
- FL assist12 = zero; FL continuation5 = zero.
- RR assist14 = zero; RR transfer7 = zero.
- New rear timing9 = zero337/337; new rear sigma multipliers all1.
- First previous-raw/history-center12 are zero in both; caps and cap-entry state agree.

Thus there is no old nonzero WAIT14 becoming zero after rear-assist OFF in these runs. The schema diff appends nine fields and retains the old410 layout. Full actor input vectors are not persisted here, so this audit does not pretend to independently prove every old410 input column equal. The specific14-column hypothesis is disproved by the actual receipts, not merely defaults.

First base-mean wheels (FL,FR,RL,RR) already differ:
old [-0.277308,0.050564,-0.007573,-0.162296];
new [-0.301895,0.026579,-0.027024,-0.178976].

## Closed-loop command and response

Canonical forward-positive wheel order below is FL,FR,RL,RR. Native actuator signs are [-,+,-,+]. Source nominal wheels are all0 for40 ticks, then all+0.3rad/s for2328 ticks in both compared prefixes. AIR FR wheel rotation is not ground traction.

Mean over the same4.0–19.733333s physical window:

| Wheel | N target | Old final | New final | Old measured | New measured |
|---|---:|---:|---:|---:|---:|
| FL | +0.300 | +0.101530 | +0.058037 | +0.162437 | +0.137066 |
| FR | +0.300 | +0.335591 | +0.319289 | +0.335996 | +0.319659 |
| RL | +0.300 | +0.266541 | +0.256772 | +0.396635 | +0.431379 |
| RR | +0.300 | +0.218740 | +0.201642 | +0.162274 | +0.129052 |

Units rad/s. All four have commands and measured rotation. The FL command difference is about−0.04349rad/s, not a lost nominal channel. Policy cancellation and subsequent physical feedback differ; rotation alone does not determine slip/traction or isolate the cause of body progress.

| Time(s) | Old FR front(mm) | New FR front(mm) | Old FR gap(mm) | New FR gap(mm) |
|---:|---:|---:|---:|---:|
| 4.000 | −134.444 | −133.657 | 84.200 | 81.389 |
| 8.000 | −102.481 | −109.968 | 84.203 | 82.550 |
| 12.000 | −70.335 | −83.955 | 83.963 | 81.723 |
| 16.000 | −38.856 | −57.709 | 83.296 | 81.948 |
| 19.200 | −15.612 | −37.006 | 86.062 | 80.107 |
| 19.733 | −9.654 | −36.072 | 83.873 | 81.896 |

At19.733s new baseX is26.589mm behind old. Old enters P03 at that tick, crosses at19.858333s and has actual placed_FR at19.941667s. Source P03 then differs intentionally. New never reaches the unchanged P02 approach acceptance band. The declared−5mm nominal approach boundary with5mm measurement tolerance accepts front≥−10mm; old measured/evaluator values agree at−9.654mm. This is not a mismatched sensor timestamp. An≈80mm gap alone is not the P02 failure: old had a similar gap before legitimate P03 handoff.

## HISTORY limits the meaning of a same-input one-step comparison

`semantic_history_actor.py:59` computes conditional_mu=0.1*network_mu+0.9*history; the task-conditioned actor uses the same kernel. `semantic_env.py:163` stores the raw request as next observable previous_raw, not the final actuator or measured motion. At enlarged cap entry the explicitly observable previous filtered request can replace this history center; no hidden cache is needed.

An identical-input comparison fixes history, so a network-mean difference is attenuated by0.1 on that one call. In a simplified repeated constant-difference, no-clipping/no-cap-switch recurrence, history can accumulate toward roughly ten times that initial conditional difference. This is a mechanism explanation, NOT a linear-system proof or measured amplification factor for this robot: feedback geometry, network inputs, tanh, mapper and physical projection also evolve. A1e−3 same-input difference cannot disprove the observed≈0.0435rad/s FL closed-loop separation or imply finite front-retention fitting is useless. No AUX efficacy claim is made here.

## Terminal is a budget cutoff despite continuing physical progress

| Time(s) | FR front(mm) | FR gap(mm) | BaseX(mm) | Current wheel supports |
|---:|---:|---:|---:|---|
| 19.433333 | −36.472 | 81.159 | 125.455 | FL,RL,RR |
| 20.433333 | −33.426 | 83.662 | 128.738 | FL,RL,RR |
| 21.433333 | −27.567 | 82.946 | 134.439 | FL,RL,RR |
| 22.433333 | −20.346 | 83.224 | 141.755 | FL,RL,RR |

Last3s: FR advances16.126mm and baseX16.300mm; last1s advances7.221mm/7.316mm. FR AIR361/361 ticks, minimum gap80.892mm, measured wheel support count2–3. All46 sampled decision task records report stall=false and physical_evaluator.termination_reason=null. This is not proof of static stability, but it excludes an asserted stall/safety-abort explanation for this recorded cutoff.

Exact terminal:
- `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`.
- P02 age22.300s; nominal budget15s; progress0.9862058335.
- allowance=min(10,15*0.5)*progress²=7.294514596s; effective budget22.294514596s.
- lifted_FR=1, clear_FR=1, approach_FR=0.9586175006; still10.346mm short of existing accepted approach boundary.
- Supervisor budget branch (`semantic_supervisor.py:1681–1703`) tests age, not stall. Stall is calculated afterwards and is diagnostic only. No physical task completion was awarded.

The failure therefore contains TWO real facts: lower closed-loop forward progress than v7 AND an engineered finite local cutoff while useful, collision-free approach was still occurring. Merely extending a fixed timeout or relabeling success is not justified. A versioned, observable, progress-earned continuation can be evaluated independently of limited front-policy retention; it must keep P03 entry and global200s unchanged, reject oscillation/stall/contact-loss counterexamples, and not hide a second controller or swap front/rear models.

## Scope and next falsifiable checks

No source/mask/assist wiring repair is supported by this paired prefix. If remaining migration parity requires proof, compare the one exact initial old410/current419 encoded state/history (first410 columns) before model fitting; the logged zero assist groups already resolve the stated WAIT14 concern. Any teacher retention must preserve the new learned checkpoint lineage, use declared bounded data and distinguish parameter changes from controller continuation. Re-evaluate one natural P01 deterministic episode with identical training/evaluation control, not a front/late checkpoint switch. No success or future P03/P05 recovery is inferred from this evidence.


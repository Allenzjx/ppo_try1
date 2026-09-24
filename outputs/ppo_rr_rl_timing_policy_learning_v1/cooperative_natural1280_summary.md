# Natural P01 / 1280 sealed audit — CP225280

## Outcome and credit

**Front sequence retained; RR/RL task not completed in the natural training trajectory.** Episode1 reached FR placement, FL placement/P06, then P09. RR had one short real qualified lift that was revoked on ground; no RR crossing/TOP/current bearing, and no qualified RL swing/crossing/TOP. This was an evolving stochastic on-policy training trajectory, **not a frozen CP225280 deterministic evaluation**.

- Exact sealed run: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_rl_timing_policy_learning_v1\train\20260924T0149526403839Z_g49eb23163a6e_b74e867ba96841109c0616c26526f9ba`.
- Natural block: **decisions224001–225280; +1280 / +10 PPO updates / +200 optimizer steps**, all completed. Lifetime **225280 /1725 /34500**.
- Together with the separately sealed P07 384 and P10 384 reports: this course **2048 /16 /320**. No prefix is counted as learner.
- Training and run manifests both `SUCCEEDED`: collection, updates, save succeeded—not physical task success.
- Request-phase P01…P13 counts: **4 / 348 / 4 / 1 / 279 / 191 / 1 / 1 / 451 / 0 / 0 / 0 / 0**. P10–P13 received no natural samples.
- Actual physics ticks10233 rather than10240: one terminal policy action executed1 instead of8 ticks. Both natural episodes started at P01/tick8; no teleport or teacher prefix was used.
- Final checkpoint: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_rl_timing_policy_learning_v1\branches\ancestor220544_recapture_v2\checkpoints\history\checkpoint_step_000225280.pt`.
- SHA from sealed checkpoint sidecar: `21897a4b2d2a85f01092c187c1b1ac824d8e61b01edaf732a6385ca951ab1dcf`; file exists, not redundantly rehashed.
- Actual sidecar-byte SHA: `6e05b4e279c1d8ac99447f91d69b7db3f6dd6afaab6a326e610283f783b1f185`, matches the parent seal.

Natural schema has **no prefix-wrapper storage flags or prefix counters**. They were not defaulted to false/0: the audit verifies `implemented_sampling=P01_full_task_only_initial_version`, `phase_suffix_curriculum_implemented=false`, no prefix telemetry fields, each real reset at P01/tick8, and all1280 decisions included in completed128-step updates. Prefix wrappers add those flags only when used (`semantic_prefix.py:416`, `semantic_checkpoint_prefix.py:343`); the training manifest explicitly reports whether `prefix_request` existed (`semantic_training.py:2371`).

## Physical progression in episode1

| Event | Exact event tick / time | Physical evidence / limitation |
| --- | --- | --- |
| FR qualified lift | 21 / .175 s | Measured upward excursion8.415 mm. |
| P02→P03 | 2800 /23.333333 s | Existing lifted/clear/approach goals satisfied; no lowered criterion. |
| FR crossed / placed | 2810 /23.416667 s; **2826 /23.55 s** | First sampled post-placement t2832 TOP13.409 N. |
| FL qualified lift | 2858 /23.816667 s | Measured upward excursion8.552 mm. |
| FL crossed / placed | 3975 /33.125 s; **5066 /42.216667 s** | P05→P06 at5072/42.266667 s. First post-placement endpoint is already AIR0 N; historical placement is not sustained load. |
| Enter P07→P08→P09 | 6600/55.0;6608/55.066667;6616/55.133333 s | Continuous progression. No P10 entry. |
| RR qualified lift | **9028 /75.233333 s** | True continuous unsupported AIR rise8.026 mm, but bottom remains42.502 mm below platform top at qualification; this is initial lift, not obstacle clearance. |
| RR ground revocation | **9043 /75.358333 s** | Current and pre-cross historical lift revoked; crossed/placed remained false. Qualification lasted15 ticks/.125 s. |
| First incomplete task terminal | **10217 /85.141667 s**, global225278 | P09 local task deadline; RR placement still0. This is not global200 s or a collision/NaN classification. |

RR AIR40/1280 endpoint samples, but **current-qualified only2**, TOP0, crossing0, placed0, current bearing0. At first qualified endpoint t9032 RR front−247.690 mm, gap−40.553 mm, AIR0 N; it is not near a capturable top region. RL AIR30 endpoint samples occurs without a qualified RL lift; RL qualified/cross/TOP/placed all0. AIR alone is not credited as the necessary lift.

FR and FL `current_lift_valid` fields are **absent** in this evaluator's per-leg front schema, so front “qualified endpoint” counts are not reported as zero. Their measured qualification **events** above are present. Contact fields exist: FR TOP925 endpoints; FL TOP20. FL later being AIR is not by itself a failure or fictitious support—the report keeps its current contact separate from historical placed.

### Last episode1 state, not the reset tail

At terminal: RR GROUND3.087 N, front−261.854 mm, gap−50.728 mm, current lift=false; RL GROUND10.506 N, front−223.227 mm, gap−50.152 mm. FR remains TOP13.451 N. FL AIR0 N, gap+181.531 mm, with4863 consecutive AIR physics ticks. Historical FR/FL placement remains true but FL is not bearing.

P09 source cursor **88**, status `holding`, reason `no_live_physical_readiness`; late group never starts (`late_group_start_tick=null`). Current RR carry/reachable/contact/bearing and RL-transfer-ready all false. Final canonical targets include RR hip+72.796°, RR knee−25.013°, FL knee−43.416°, FL wheel−.862481 rad/s; measured FL wheel−.861391 rad/s. These are observed targets, not proof of their independent causal contribution.

## Cooperative sampling and learning validity

**Prep gate0/1280, FL extra0/1280.** Therefore the new state-gated cooperative multiplier never activates here; the carry-and-reachable RR×4 multiplier also does not activate. The natural trajectory did not reach this experiment's intended cooperative preparation region. It would be incorrect to claim these1280 samples demonstrated effective new prep exploration. Rear capture assists remained off, RR-assist14 zero, one policy draw and no extra forward per decision; all native audit tick summaries verified.

Counterroll soft cost eligible0/10233 physical samples, charged0. This trajectory's lack of qualified/reachable rear preparation is an additional eligibility restriction; source-sign exclusion from the separate P10 report cannot be assumed the only cause here.

All1280 stored raw samples/logp match their policy receipts. All200 optimizer minibatches cover each128-step rollout sample five times, using the same observed-gate sigma kernel. Independent scalar Gaussian max error: old/sample logp0.000003588, current logp0.000003803. This checks raw-action likelihood, not the probability of a clipped executed target. All updates record finite nonzero gradients, changed actor parameters and learning rate1e−5.

| PPO update | Credited decisions | KL mean | Value loss | Raw GAE mean | Terminals |
| --- | --- | --- | --- | --- | --- |
| 1716 | 224001–224128 | 0.02639 | 0.03599 | -0.13979 | 0 |
| 1717 | 224129–224256 | 0.01576 | 0.00770 | -0.12200 | 0 |
| 1718 | 224257–224384 | 0.01717 | 0.01408 | 0.08885 | 0 |
| 1719 | 224385–224512 | 0.02333 | 0.09430 | -0.40986 | 0 |
| 1720 | 224513–224640 | 0.02731 | 0.01923 | 0.05534 | 0 |
| 1721 | 224641–224768 | 0.02743 | 0.01318 | 0.04931 | 0 |
| 1722 | 224769–224896 | 0.01662 | 0.02843 | 0.23752 | 0 |
| 1723 | 224897–225024 | 0.02110 | 0.10837 | 0.38934 | 0 |
| 1724 | 225025–225152 | 0.01529 | 0.19075 | 0.50276 | 0 |
| 1725 | 225153–225280 | 0.02197 | 300.08976 | -15.87824 | 1 |

The final value-loss spike **300.089757** is finite, coincides with the real task-terminal return, and must not be hidden by an overall average. Terminal sample old V−10.902531, reward=return−42.268894, raw GAE−31.366364. Last block raw GAE mean−15.878240 versus earlier−.409859…+.502757. The terminal reward is −40 event plus−2.268699 potential shaping and a small geometry cost. This is observable training pressure, not proof the policy learned a correction; further attribution would require subsequent evaluation.

## Terminal versus collection budget and GAE

Episode1 uses1278 decisions. Its terminal source is **`LOCAL_TASK_DEADLINE`**, age30.008333 s in P09, limit30 s, allowance0; remaining global time114.858333 s. Supervisor checks global/local finite deadlines at `semantic_supervisor.py:1844–1850`. It is not an external data truncation: terminal bootstrap explicitly false and terminal return equals reward. `stall_diagnostic=false` does not override the separate local timer. Current RR qualification/capture is unavailable; this report does not propose merely adding timeout.

After that terminal, training naturally resets and samples **two P01 decisions** (global225279–225280, ending P02/t16/.133333 s) to finish the last128-step rollout. Those two samples are real learner decisions, not a second completed attempt. The overall1280 collection budget then seals the checkpoint.

Normal phase changes remain nonterminal. All ten rollout **tails** are nonterminal and use official compute-returns bootstrap, including the fresh second episode at the last tail; the prior failed episode is cut by its own terminal. Separate numerical last bootstrap values are not logged here and were not re-forwarded.

Sampling block1 uses CP224000 behavior; subsequent blocks use their preceding saved update through CP225152. CP225280 is saved only after update1725, so this report does not substitute the evolving training trajectory for the latest saved-model DET video.

## Concrete blocker and scope

The first remaining task is **RR lift/carry toward a legal top region and durable capture**, before RL transfer. The natural path retains FR/FL completion but fails well before RR capture: only a short8 mm initial lift at approximately248 mm before the front edge, then ground revocation and no eligible cooperative prep samples. This differs from the P10 teacher-captured loss scenario and must not reuse that explanation as established cause.

Outputs-only stdlib audit; no Torch/Isaac/model queries, new physics, production/reward edits, or active DET reads. Full bounded facts and null/absent-field distinctions are in the companion JSON.


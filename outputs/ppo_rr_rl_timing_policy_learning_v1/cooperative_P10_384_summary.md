# Sealed cooperative P10 / 384 — read-only physical and likelihood summary

## Scope and credit

Only sealed run `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\runs\ppo_rr_rl_timing_policy_learning_v1\train\20260924T0130272021180Z_g49eb23163a6e_5e6b94b24cce4a2fb9edc117565be855`, its 384 decision receipts, three optimizer/advantage/likelihood records, and relevant frozen code were read. Active P01 course was not read or changed. No Torch, Isaac, model forwards, physics, or production edits were executed.

- Frozen source: `49eb23163a6e20bc56301dbafb59b137ecebce66`; same 422-dimensional cooperative policy.
- Learner decisions **223617–224000: +384 / +3 PPO updates / +60 optimizer steps**. Lifetime 224000 / 1715 / 34300 (last total supplied by sealed parent ledger).
- N+0 continuous prefix: **767 decisions, 6136 physics ticks / 51.133333 s, PPO credit 0**. RR qualified at t5427; RR crossed/placed at **t6133 / 51.108333 s**, inside teacher prefix. This run does not demonstrate learner-acquired initial RR placement.
- Saved CP224000: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1\outputs\ppo_rr_rl_timing_policy_learning_v1\branches\ancestor220544_recapture_v2\checkpoints\history\checkpoint_step_000224000.pt`.
- Checkpoint hash from its sealed sidecar: `24fa50fba9212ddbda123380f9746769654af169c65fc48e09380c6d60ef5d9d`; checkpoint exists, not redundantly rehashed. Sidecar bytes independently match `ac2424852f57a9a8953c4054f80ba6ffd293994d337424e6febd472b803a3eb3`.
- Sampling blocks use CP223616, CP223744, CP223872 respectively. CP224000's final update is **not physically executed within this run**.

| Block / behavior CP | Credited decisions | Physical seconds | RR TOP / current qualified endpoints | RL AIR / current qualified endpoints | Prep-gate samples |
| --- | --- | --- | --- | --- | --- |
| 1 / CP223616 | 223617–223744 | 51.133–59.667 | 45 / 125 | 4 / 2 | 117 |
| 2 / CP223744 | 223745–223872 | 59.667–68.200 | 28 / 33 | 7 / 15 | 26 |
| 3 / CP223872 | 223873–224000 | 68.200–76.733 | 0 / 0 | 0 / 0 | 0 |

All counts are **15 Hz endpoint samples**, not contact durations. P10=1, P11=1, P12=382. Aggregate RR TOP/current bearing=73, RR AIR=96, ground-contact=213, current qualified=158. RL AIR=11, current qualified=17; qualification and AIR are separate (some qualified endpoints have non-ground edge contact). RL cross/TOP/placed=0.

## RR contact loss, qualification, and current support

| Event | Evidence |
| --- | --- |
| Learner first endpoint t6144 / 51.2 s | RR TOP 15.338 N; front +4.814 mm, gap −0.209 mm. Placement predates learner. |
| First loss | Last sampled TOP t6200: 12.321 N, gap −0.798 mm. At t6208: AIR 0 N, gap +2.849 mm, seven consecutive AIR ticks. **First AIR t6202 / 51.683333 s is counter-derived**, not an invented instantaneous force reading. |
| Intermittent recaptures | TOP returns at endpoint t6232, t6264, t6536, t7120. Presence is not sustained completion. |
| Last TOP before ground | t7384 / 61.533333 s: TOP 0.775 N, front +16.950 mm; consecutive TOP=272 ticks. t7392 has AIR count8, so **last AIR onset t7385 / 61.541667 s**. |
| Ground/revocation | **Exact event t7496 / 62.466667 s**: GROUND 49.517 N, front −91.077 mm, gap −50.324 mm; `current_lift_revoked_ground`. This is 111 ticks / 0.925 s after the counter-derived last AIR onset. |
| Final t9208 / 76.733333 s | RR GROUND, current lift false, TOP false, front −75.225 mm, gap −49.989 mm, force2.704 N. RL GROUND 15.028 N, front −68.581 mm; FR10.949 N, FL0 N. No RL crossing/placement. |

Exact AIR onset derivation uses evaluator's contiguous 120 Hz updates, same-tick idempotence and reset-on-non-AIR counter (`semantic_supervisor.py:746–750,830`). Raw endpoint brackets and event receipts are retained in JSON. Historical RR crossed/placed remain true throughout all384, while **current qualification is revoked at ground**; history is not asserted as current support.

P09 late began at **t6114 / 50.95 s during teacher prefix**, before historical placement t6133. On first loss, P09 source advances735→743 (t6200→6208), remains active; P12 changes active→`holding_RL_joint_lane`, transfer permission true→false and actual-current-RL-swing true→false. Wheel source clock continues. On last loss it similarly advances P09 source1919→1927 while P12 holds its RL joint lane. This confirms the already-declared limit: P12 current-support gate is active, but source-clock holding does not retract previously issued P09 remote servo goals. No new rear controller guard was added here; coincidence does not prove what caused loss.

At final state, public recapture-task request remains true, but physical `rr_lift_carry=false`, current RR lift=false, reachable=false, current bearing=false, and RL-prep=false. Prep-gate=false. Public task/recapture intent is **not** a sensor assertion of current AIR or bearing.

RL learner attempts: qualified t6188 / 51.566667 s → ground revoked t6202 / 51.683333 s; re-qualified t7184 / 59.866667 s → ground revoked t7297 / 60.808333 s. Both are real short attempts but **neither crosses or touches TOP**.

## Actual exploration authority, including clamps

All384 observed-gate multipliers matched the shared scalar kernel. Prep gate active143 (117/26/0 by block); FL additional precontact multiplier active0. RR hip multiplier remains ×4 **only under the existing carry-and-reachable condition**, never increased. No rear capture assist, extra model forward or extra draw; RR-assist14 was zero. All channel masks remain enabled, but actual post-rate/headroom effects below—not masks alone—show authority.

Table is only the143 prep-active samples. Raw/μ/σ are latent Gaussian values; servo request/final units °, wheel units rad/s. “Clamp” is rate / headroom / final-servo-slew endpoint counts; raw remains the likelihood variable.

| Channel/index | Raw range | Conditional μ range | σ range | Requested after residual rate limit | Final target range | Clamp counts |
| --- | --- | --- | --- | --- | --- | --- |
| FR_knee [3] | -2.125…0.629 | -1.957…0.532 | 0.163…0.679 | -90.434…12.283 | -58.000…42.361 | 133 / 1 / 0 |
| FL_knee [1] | -2.227…0.019 | -2.173…-0.081 | 0.141…0.457 | -35.172…-0.583 | -58.000…-36.312 | 65 / 70 / 0 |
| RL_hip [4] | -1.434…0.188 | -1.347…0.110 | 0.108…0.220 | -21.419…2.310 | -34.855…28.129 | 32 / 0 / 2 |
| RL_knee [5] | -1.096…0.490 | -0.988…0.436 | 0.049…0.197 | -28.695…16.351 | -25.251…52.901 | 31 / 0 / 0 |
| FL_wheel [8] | -6.065…0.366 | -5.610…0.259 | 0.027…0.951 | -1.200…0.210 | -1.277…-0.545 | 22 / 0 / 0 |

- FL knee: all384 conditional means negative; requested residual negative384/384. **271/384 final targets at −58°**, including70/143 prep samples. Raw samples are all384 distinct; only107 distinct final targets. This is substantial collapse at the real headroom boundary, not lack of stochastic raw variation. In prep, final FL knee never exceeds −36.312°.
- FR knee: prep sigma is wider and requests vary, but133/143 are residual-rate-limited; final mean during prep −10.676°, range −58…+42.361°. This is not evidence of reliable receiving-space motion.
- RL hip: prep32 rate clamps, two final slew clamps, no headroom clipping; RL knee31 rate clamps, no headroom clipping. Their final targets vary, yet physical RL crossing/TOP remains0.
- All384 rate clamps FRknee162, FLknee86, RLhip34, RLknee31, FLwheel132. Headroom clamps FRknee1, FLknee271, others0; final servo slew clamps RLhip3 only among these channels. This does **not** imply the whole controller or every interior physics tick is unclamped.

Residual-rate audit uses `tanh(raw) * current_cap`, actual previous-filtered request,60°/s or1.8rad/s² and the logged tick count. Two actual one-tick handoff holds reduce available evolution from8→7 ticks at decisions223618/223619. Maximum reconstruction difference **4.627e−6** from float32 inputs. Same-tick headroom candidate→prior-final/hard-bound/slew reconstruction error **0**. No independently recomputed nominal was subtracted to manufacture policy effect. Source anchors: `action_projection.py:515–542`, `semantic_env.py:147–166`, `semantic_residual_adapter.py:279–291`, `robot_adapter.py:709–733`.

## FL reverse after source stop: first divergence

Wheel numbers below are canonical forward-positive rad/s, including the wheel slots of the headroom mapper vector; these are not raw actuator-joint signs. The verified native conversion is FL = −canonical (`command_batch.py:57–61,213,234`), so negative canonical FL is physical reverse despite positive native FL joint velocity. Raw native actuator velocity is not reprinted or claimed directly read here. “Stop” means the actual merged N and same-tick mapped N have become0, not a claim based only on source event labels.

| Tick/s | N / mapped N | Conditional μ / σ / raw | Desired policy residual | Post-rate / headroom / final | Actual wheel rad/s |
| --- | --- | --- | --- | --- | --- |
| 6472 / 53.933333 | −0.3 / −0.3 | −1.137796 / .109974 / −1.089861 | −.956193 | −.956193 / −.956193 / −1.256193 | −1.421570 |
| **6480 / 54.000000** | **0 / 0** | **−1.105158 / .169821 / −1.067218** | **−.946093** | **−.946093 / −.946093 / −.946093** | **−.869012** |
| 6488 / 54.066667 | 0 / 0 | −1.083589 / .222156 / −1.109845 | −.964809 | −.964809 / −.964809 / −.964809 | −.930739 |

At t6480 the original current raw request itself asks for reverse, and is reached by post-rate/final. This is **policy reversal, not a missing nominal stop, not only retained slew, and not wheel-mask loss**.

| Scope | n | Raw<0 | Conditional μ<0 | Requested<0 | Final<0 | Actual<0 | Residual rate-clamped |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All | 384 | 312 (81.25%) | 322 (83.85%) | 322 (83.85%) | 327 (85.16%) | 327 (85.16%) | 132 |
| Prep active | 143 | 138 (96.50%) | 140 (97.90%) | 138 (96.50%) | 143 (100%) | 143 (100%) | 22 |
| N_FL=0 | 342 | 275 (80.41%) | 283 (82.75%) | 285 (83.33%) | 285 (83.33%) | 285 (83.33%) | 120 |

All384 FL base-network means stay negative (−1.535…−.477, mean−1.080); conditional μ differs through HISTORY. N0 FL sigma range.0271…2.9403; final−1.19999…+1.18991, actual−1.29316…+1.19032. Thus it does sometimes reverse sign but does not establish reliable forward cooperation.

Logged counterroll reward eligible=0 /3072 physical samples, positive cost=0, total cost=0 across all three rollouts. The predicate intentionally requires currently qualified RR AIR, measured FL support and **source FL>0** (`semantic_cooperative_preparation.py:153–162`); it does not charge source-stop windows merely because policy reverses. Aggregate zero charge is a measured fact, not proof each failed gate was the same, nor a reward change recommendation.

## PPO receipt and end condition

| Update | KL | Clip fraction | Value loss | Surrogate loss | Raw GAE mean | Old V / return mean | Reward mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1713 | 0.03135 | 0.38125 | 0.57040 | -0.04203 | 0.73914 | -12.76228 / -12.02315 | -0.006374 |
| 1714 | 0.02611 | 0.35313 | 0.10572 | -0.03751 | 0.19004 | -12.85813 / -12.66809 | -0.008530 |
| 1715 | 0.02678 | 0.32813 | 0.54747 | -0.03668 | 0.77131 | -13.05828 / -12.28697 | -0.005788 |

All128 samples per block finite; actor hashes change each update, nonzero finite gradients recorded, learning rate1e−5. Sixty minibatches cover each sample exactly five times. Original raw/logp/μ/σ match stored rollout; current likelihood uses the same observed-gate sigma kernel. Independent scalar Gaussian maximum error: old/sample logp2.092e−6, current logp2.929e−6. This validates the **raw** likelihood, not likelihood of a clipped target. Mostly positive raw GAE—even in block3 with RR ground/RL uncompleted—is critic-relative, not evidence of task success.

The run ended at its **384-decision collection budget**, lifecycle SUCCEEDED means collection/save succeeded, **not robot-task success**. All384 terminals=false/task-success=false; ordinary P10→P11→P12 changes did not end GAE. All three tails were nonterminal with official compute-returns bootstrap enabled; separate numerical last-values were not logged here and were not rerun. Global200 s was not reached.

### Concrete current blocker, without causal overclaim

RR teacher capture is not maintained through RL preparation: intermittent TOP is followed by loss of region/ground and correct qualification revocation; RL only achieves transient lift attempts and remains uncrossed. FL knee stays heavily clipped at its negative boundary and the stopped-source FL wheel predominantly continues policy-requested reverse. These are concurrent measured limitations, not proof any one channel alone caused the failure. This audit changed no target, reward, gate, policy, or active run.

## Bounded next-change recommendation (not implemented)

The143 prep samples show residual rate-clipping FR knee93.0%, FL knee45.5%, RL hip22.4%, RL knee21.7%; FL knee additionally reaches its−58° headroom boundary49.0%. Clamp categories overlap and cannot be summed. These are actual limiting rates, **not** a counterfactual estimate of how much new sigma caused or lost useful exploration. FL knee has67 distinct final targets from143 distinct raw draws; FR knee97; RL hip/knee142 each. Useful physical progress must remain the test, not target diversity.

The zero nominal /285-of342 reverse samples plus zero logged counterroll eligibility establish a **coverage gap in that particular soft term**: its deliberate source>0 requirement excludes stop windows. They do not establish that every reverse sample was harmful, that all exclusions had only that cause, or that the whole reward lacks task-progress signals. A minimal candidate is to separate measured task need for RR forward approach/capture from whether N happens to be positive. Preserve intentional source reverse and actual beneficial forward/legally descending effects; consider a local soft cost only with valid current task evidence, real FL support, actual target and measured speed both reverse, and insufficient physical progress. Before changing it, count that candidate eligibility on these receipts and add beneficial-reverse/source-reverse/current-invalid negative examples. Do not turn the entire P12 ground/recapture interval into a blanket no-reverse mode.

For servos, first check that the existing task potential assigns useful credit to restoring FL range and receiving geometry; a further raw-sigma increase would not address the measured negative-mean/headroom collapse. Do not relax the physical slew or hard limits on the basis of this audit. This is an evidence-based design candidate only; no mean shift, abs, global reverse ban, new angle target, reward edit, or live-run change was made.

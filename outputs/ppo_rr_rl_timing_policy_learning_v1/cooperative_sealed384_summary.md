# Sealed cooperative P07 course: actual384 summary

Run: `20260924T0038084596548Z_g49eb23163a6e_baf0b6006fea46a3a731db4c4d63d633`. Frozen control HEAD49eb23163a6e. Only this sealed course was read; no Torch/model/Isaac execution or production edits.

**Saved CP223616, cumulative update1712. Actual increment:384 policy decisions /3 PPO updates /60 optimizer steps.** SHA256 `327f62688c41cb0d6137fb35de64592c12030389aa352c7199fd1f780079d632`. Continuous successful-N P01→P07 prefix645 decisions /43.000s /tick5160 has **zero PPO credit**. One continuous physical episode, no teleport/reset during these384. Training budget ended at68.600s, not task termination: task success0, task terminal0, completed episodes0. The process lifecycle SUCCEEDED means the training request completed.

## Coverage, separated by sampling parameters

Each row contains128 on-policy decisions. Contact/qualification counts are15Hz decision endpoints, not summed120Hz contact duration.

| Block | Network during collection | Credited decisions | Physical time(s) | Prep gate | RR TOP/current-bearing | RL qualified | RL AIR | P12 RL lane held |
|---|---|---|---|---|---|---|---|---|
| 1 | CP223232 | 223233–223360 | 43.000–51.533 | 29 | 8 | 4 | 5 | 46 |
| 2 | CP223360 | 223361–223488 | 51.533–60.067 | 89 | 26 | 0 | 1 | 102 |
| 3 | CP223488 | 223489–223616 | 60.067–68.600 | 121 | 79 | 0 | 5 | 49 |

Total stages: P07=1, P08=1, P09=64, P10=2, P11=8, P12=308. Total RR current-qualified307, TOP/current-bearing113, RL current-qualified4, RL TOP/crossed/placed0. Rear capture assists and rear wheel projection remained off. Extra FL-wheel sigma factor appeared only11 requests in block1; RR-hip remains the declared ×4, not further broadened.

**Attribution:** RR placed at5682/47.350s and RL qualified at5772/48.100s occurred during block1, before any update in this course. They are saved CP223232 parameters with the new stochastic sigma/control version—not newly learned CP223360 behavior. Both rear events are later than learner start5160 and are not teacher-prefix achievements. Block2 runs parameters after update1710; block3 after1711, but physical state/time also change. Increased contact counts therefore are **not a paired causal learning comparison**. CP223616's final update1712 had no subsequent execution inside this training run; its separate deterministic evaluation is outside this audit.

## Contact progress and actual remaining blocker

- First sampled RR TOP5688/47.400s:0.364N; lost by5696. Stronger TOP5768–5816:13.50–15.74N,54 consecutive physical TOP samples by5816. Lost by5824.
- Later blocks repeatedly reacquired RR: TOP force1.115–11.785N in block2,0.391–16.729N in block3. There are14 sampled gains and14 losses across the course; JSON preserves all28 transition brackets. Consecutive endpoints can miss shorter interior contacts, so these are not exhaustive120Hz event counts.
- RL qualification lasted5772→5802 (48.100→48.350s); it returned to ground before crossing and current qualification was revoked. Later initial-clearance event7566/63.050s did **not** become a new qualified lift. No RL TOP, crossing or placement.
- Last recorded TOP endpoint8112/67.600s:1.352N; lost in(8112,8120]. At8232/68.600s RR is **qualified AIR, inside legal XY, gap8.587mm,0N**, with109 consecutive AIR samples. RR historical placed is retained but not misrepresented as present support.
- RL at68.600s:GROUND, current-qualified=false, front-distance−54.389mm, gap−50.847mm,13.466N. Real other supports are FR+FL+RL. First unmet P12 task remains `placed_RL` (bounded progress score0.254819, not placement).
- P12 reports `holding_RL_joint_lane`, `support_transfer_permitted=false`, `rl_current_swing=false`; its wheel source clock continues. Current recapture/carry preparation remains allowed, not whole-body frozen. Last actor request at8224 sees rear flags[1,0,0,0,*,*,*,0,1], carry=true, reachable=true, RLprep=false, cooperativeprep=true, parentreceiving=true, FL-extra=false. Thus current loss/recapture permission is observable; **no support is invented**.
- Final canonical FL wheel target−1.07278rad/s; logged measured FL−1.10664rad/s. Other final wheels FR+.44837,RL−.15156,RR−.08790. Body-forward metric fell from.33246m at60.067s to.17568m at68.600s. These are measured/commanded concomitant facts, not proof FL reversal alone caused regression.

**Concrete blocker:** RR contact is not retained sufficiently to finish the next RL transfer/qualified swing. At the end RR remains a reachable AIR candidate needing recapture, while RL bears ground load and has no live swing exception. StageP12 and historical placed_RR do not equal successful continuation. Existing P09 late targets are not withdrawn after loss; that known control limitation remains distinct from current P12 cursor gating.

## P09 late: initial light-touch hypothesis checked

P09 late actually starts **tick6202 /51.683s**, during later reacquisition—not at first light TOP5688 and not during the initial stronger segment. Its source clock was only281 at first light TOP and361–409 during the strong segment; the late boundary is648. It waits at648 through6200.

Force values below are measured endpoint forces in N; all FR/FL contacts listed are TOP. Exact6202 instantaneous force is not in the per-physics audit summaries, so it is **not reconstructed from6208**.

| Physics tick/time(s) | P09 source tick / late start | FR N | FL N | RR N | RL N |
|---|---|---|---|---|---|
| 5688 / 47.400 | 281 / not started | 14.207 | 3.532 | 0.364 | 10.613 |
| 5768 / 48.067 | 361 / not started | 0.000 | 14.716 | 15.158 | 0.000 |
| 5816 / 48.467 | 409 / not started | 0.000 | 12.765 | 13.886 | 1.883 |
| 6200 / 51.667 | 648 / not started | 14.699 | 3.275 | 0.000 | 11.235 |
| 6208 / 51.733 | 655 / 6202 | 9.307 | 6.131 | 4.164 | 8.109 |
| 6216 / 51.800 | 663 / 6202 | 7.687 | 9.112 | 7.317 | 8.133 |
| 6224 / 51.867 | 671 / 6202 | 10.401 | 5.932 | 2.568 | 5.633 |
| 6232 / 51.933 | 679 / 6202 | 15.093 | 4.149 | 0.000 | 9.029 |

6200→6208 merged source nominal changes FL hip/knee from38.6/−13.4° to−18.5/−31.4°. RL nominal is already0.5/35.3° and unchanged; four nominal wheels are0 on both endpoints. No new−1.07 FL source pulse appears in this narrow endpoint window.

After RR loss at6232, P12's RL lane pauses, but P09 remainsactive/source679. FL mapped-N hip continues11.1→1.1° and final hip2.548→−7.452° from6224→6232; final FL knee remains−58°. This is direct evidence that **pausing a source cursor does not withdraw its previously issued remote target**. It does not show that the old target was the sole cause of contact loss. Same-tick mapped-N/final vectors and FR/FL/RR/RL forces are in JSON `P09_late_bounded_window`.

## PPO numbers and reward observations

All losses/targets below are actual ledger values, not model re-evaluation.

| Completed update/CP | Mean KL | Clip fraction | Value loss | Surrogate loss | Mean old V | Mean return target | Mean raw GAE | Raw GAE positive/negative |
|---|---|---|---|---|---|---|---|---|
| 1710 / CP223360 | 0.03252 | 0.3375 | 1.16814 | -0.02422 | -12.7714 | -11.8035 | 0.9678 | 116/12 |
| 1711 / CP223488 | 0.01662 | 0.2328 | 0.36541 | -0.02461 | -12.8341 | -12.2307 | 0.6034 | 127/1 |
| 1712 / CP223616 | 0.02104 | 0.2781 | 0.28838 | -0.04340 | -13.4247 | -12.9740 | 0.4506 | 114/14 |

All3 have20 optimizer steps, LR1e−5, finite nonzero gradients and changed actor hashes; all128 advantages/values/returns per block finite. Stored advantages are normalized (mean≈0,std≈.99609). Mostly positive raw GAE with negative returns means outcomes exceeded this critic's old predictions, **not that the task succeeded**. No terminal samples; ordinary phase changes stayed nonterminal; each tail uses the logged official nonterminal compute_returns bootstrap (last V was not independently re-evaluated).

FL counterroll fee is genuinely active but narrow:

| Block | Physical samples | Eligible | Positive fee samples / decisions | Integrated logged fee |
|---|---|---|---|---|
| 1 | 1024 | 107 | 44 / 8 | 0.001610034 |
| 2 | 1024 | 0 | 0 / 0 | 0.000000000 |
| 3 | 1024 | 0 | 0 / 0 | 0.000000000 |

Per-decision fee exactly equals sum(raw_cost×physical_dt×.01), and `semantic_reward.py:462` subtracts it in unweighted task_progress. Later two blocks have source FL nominal0 at every logged endpoint and all their120Hz fee samples ineligible. The existing fee requires positive source FL suggestion (`semantic_cooperative_preparation.py:154–160`); a prep exploration gate alone does not imply fee eligibility. It intentionally does not penalize every stop-conditioned reversal. No reward tuning was performed.

## Sampling / execution accounting

384 original raw/old-logp/mean/sigma receipts match exactly;60 minibatches,5 exposures per sample, unchanged shared cooperative multiplier rule. Independent scalar Normal max errors: original 0.000002662950414844545, minibatch 0.0000025216514742965046. No extra policy draws or teacher-prefix credit. Native target audit verifies3,072 physical ticks; this validates audited target execution, not sustained physical tracking or task success. Final checkpoint is saved, but this training run is not a deterministic evaluation/video claim.

Files: `cooperative_sealed384_summary.json` retains block targets/advantages, endpoint contact transitions, final observed flags, and the narrow late-group window. The prior `cooperative_first128_summary.md/json` remains unchanged.


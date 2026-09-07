# Block25 final — natural-P01 front-wheel-range training to89088

Finalized source: `runs/ppo_semantic_v3/train/20260906T2301404237753Z_g2677995544c9_2288cdc238fb40bf96f60d1b04de349d`. Read-only PowerShell ledger check; no simulator, tensor loading, fresh hashing or production changes. Root independently confirmed session31307 CLOSED/exit0. Both final manifests record `SUCCEEDED` execution. This does not mean task success.

## Actual budget and immutable save

Actual=requested=planned **2,048 policy decisions /16 PPO updates /320 optimizer steps**, unconsumed0, rounding0, recorded loop wall **745.9372223000973s**. Source87040/645/12900 becomes **89088/661/13220**. Spending is full_episode35,328/phase_suffix43,648/smoke0; the original10,112 origin is retained. Across25 finalized blocks since10,112/44/880, actual additional work is **78,976 decisions /617 updates /12,340 optimizer steps**.

The latest pointer names immutable `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000089088.pt` and its matching sidecar. Recorded checkpoint SHA is `b5befddddbe2e877b9df76926ba3271e17a4e0eb79a352040aae1e01acd69242`; pointer-recorded sidecar SHA is `a2090a54580e22b1fda7d440bcea786cd3ff506b07307053d39a4e8b33a46f73`. The sidecar matches global89,088/661/13,220, this exact source run, final actor and stage ledger, and records save/load roundtriptrue. These are compared recorded receipts, not newly recomputed hashes. Saved boundaries87,168/87,552/88,064/88,576/89,088 all record roundtriptrue and the same identity-normalizer hash.

## Exact policy and episode ledger

Policy-request phase counts P01–P13 are **3,581,11,3,449,966,15,2,18,0,0,0,0**, summing2,048. The actual audit is contiguous global87,041–89,088, with no additional rows. Three natural-P01 starts each begin credited physical tick8. `curriculum_epoch.prefix_request=null`, sampling is `P01_full_task_only_initial_version`, and core decisions/ticks exactly equal policy decisions/ticks: **no teacher or checkpoint-policy prefix actions occur in this block**. Front-leg achievements below are therefore genuine current-policy history, not transferred teacher credit.

| Episode | Global decision range | Decisions | Physics ticks | Final time / phase | Outcome |
|---|---|---:|---:|---|---|
| 0 | 87041–87777 | 737 | 5894 | 49.116667s / P09 | BODY_COLLISION, taskfalse |
| 1 | 87778–88621 | 844 | 6746 | 56.216667s / P09 | BODY_COLLISION, taskfalse |
| 2 | 88622–89088 | 467 | 3736 | 31.133333s / P06 | Nonterminal optimized tail; reasonnull |

Completed-episode files independently agree on the two terminal counts/times. The two completed episodes total1,581 decisions; tail467 is neither a third failure nor a success. All three endpoint shared physical evaluations are valid; the first two physical reasons are `TASK_FAILURE_BODY_COLLISION`, and the tail reason isnull. This training run retains the compact applied/physical evaluation evidence, not a separate complete raw body-pair observation stream; exact body-force/contact-point magnitudes are not invented here.

All **16,376 policy physics ticks** have native verification and actual target effect; own-phase effect16,355. All four in-episode state-write counters are0. The only short decisions are global87,777/6ticks and88,621/2ticks, both real terminal:2,048×8−2−6=16,376. Every one of21 policy phase changes is nonterminal, timeoutfalse and bootstraptrue; phases are not turned into artificial GAE terminals. The two real collisions retain their task terminal semantics.

## First missing physical task and actual rear-leg history

| Episode | FR Q/C/P | FL Q/C/P | RR attempt / hard result | First unfinished task at endpoint |
|---|---|---|---|---|
| 0 | 65 /1522 /1545 | 1642 /2614 /2750 | initial5859 only; no Q/C/P | P09 `placed_RR=0.2836998344`; interrupted by body collision |
| 1 | 42 /1643 /1664 | 1749 /2772 /2866 | initial6682 only; no Q/C/P | P09 `placed_RR=0.3226526113`; interrupted by body collision |
| 2, unfinished | 47 /1511 /1539 | 1642 /2458 /2749 | no RR initial or Q/C/P | P06 `rear_approach=0.0862031167`, still running at budget boundary |

RL has initial-clearance hints in all three episodes but **no hard Q/C/P**, and RR likewise has no hard Q/C/P anywhere. Initial hints and continuous partial values are not active-lift qualification or successful placement. No P10–P13 samples are present.

At episode0 collision, FL is AIR/load0/clearance+194.7857mm, RR AIR/load0/front−194.10595mm/clearance−44.83692mm, and RL AIR/load0. At episode1 collision, FL is AIR/load0/+149.73042mm and RR AIR/load0/front−179.04925mm/clearance−44.70106mm; RL is GROUND with normalized load.805693. Historical FL placement does not imply current support. At the nonterminal tail, FL is actually TOP/load.546019 with obstacle pair active, RR GROUND/load.453981/front−427.50952mm, and RL AIR/load0/front−448.44922mm. The tail has not yet completed rear workspace approach; it is not an evaluated failure of a later-stage skill.

## Version boundary and real optimizer evidence

This is the selected **2677995544c974c03d8b1e41d77375b45e323c9e** new-MDP boundary: the recorded runtime file delta is only `configs/ppo_semantic_v3/execution_profile.yaml` (P06–P13 front-wheel residual caps .6→1.2, rear wheels unchanged). The other five selected configuration receipts are equal. The same raw actor output can therefore produce a different physical action; this change is not behavior-bitwise equivalence or proven treatment of the earlier collision.

Source87040 and new immutable initial sidecar have equal complete actor (including learned heteroscedastic std), critic, identity-normalizer and training-RNG receipts, unchanged origin/spent ledger and zero added counters. Source Adam learning rate1e−5 becomes fresh Adam3e−5; the optimizer-state hash changes and `old_rollout_buffer_inherited=false`/`physical_state_inherited=false`. The initial snapshot records roundtriptrue and SHA `8f749455697b3e379af32cb2aaa8632b4bc5937b9cbcda7b94b7fbb2b9ac12a6`. Initial sampling is current natural P01, not the source checkpoint's P06 prefix course. RNG receipt equality is checked; the migration record explicitly restores it before fresh sampling.

All16 update records are contiguous646–661/global increments128, each records20 actual optimizer steps (=5 epochs×4 minibatches), changed actor parameters and finite nonzero gradients. The source→first-before,15 adjacent links, and final-after→manifest actor hash all match. Recorded gradient extrema are1.0009003732–1.4142136425. All update-end learning-rate records are1e−5; this does not claim a constant rate at every minibatch. Final actor receipt is `733b1c1efaa2fddd8c5754f2ec94cb288f951fc0934503fb02acee51f33312bd`. Identity-normalizer hash remains `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`.

The same single policy-stream pass checks finite positive12-dimensional old standard deviations/finite means and recomputes stored diagonal-Gaussian log probabilities in CPU scalar arithmetic; maximum absolute difference is1.279569455e−6. There is no alternate training execution or policy update in this analysis.

**Conclusion:** the selected range version completed its requested optimizer budget and produced a verified saved checkpoint, but both completed episodes still have a real P09 body-collision termination before RR qualification. No full-task/suffix success, stability improvement, cap-collision causal claim or successful video is established. Root's separate reloaded C89088 natural-P01 evaluation is pending outside this report; latest completed full evaluation remains C87040 BODY_COLLISION until its own final evidence appears.

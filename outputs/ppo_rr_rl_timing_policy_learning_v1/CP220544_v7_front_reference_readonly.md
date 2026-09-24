# Front reference facts before current CP222080 DET closes

Bounded read-only comparison; no model forward or new physical run. Current recording outcome is not asserted here.

Source: CP220544 v7 SHA `47fdec0614ed2683a0eae6fdef736598400f3c6833473b551a8c93f6f0d8f430`, sealed deterministic natural-P01 run `20260923T0934264919029Z_g60abc00957c0_ef405598c8954e6280e844c4c29ed041/source` in the old RR-capture namespace.

| Verified old event | Tick | Time |
|---|---:|---:|
| P01→P02 | 16 | 0.133333 s |
| P02→P03 (FR not yet crossed/placed) | 2368 | 19.733333 s |
| Actual FR front-edge-crossed history | 2383 | 19.858333 s |
| Actual FR placed history | 2393 | 19.941667 s |
| P03→P04 with placed_FR=1 | 2400 | 20.000000 s |

These come from sealed `stage_transition_evidence.jsonl`; `phase_metrics.csv` independently records P02 duration19.6s and P03 duration0.266667s. P02 handoff is not itself touchdown. Existing bounded comparison records old FR at8s: front−102.481mm/gap84.203mm; at12s: front−70.335mm/gap83.963mm. Thus current reported10s front≈−95mm/gap≈79mm is not by itself a failure or proof of matched trajectories. No exact10s paired causal inference was performed.

## Configuration compatibility, not complete runtime equivalence

Compared old committed60abc config to current419 config: P01/P02/P03 full stage entries match; Full12 action schema matches after removing only namespace/explicit assist-description metadata; the complete `residual` block matches. Physics120Hz, policy15Hz, mapper, caps, HISTORY and source-derived front control remain preserved by the published410→419 contract.

Whole profiles are intentionally different: rear geometry/hip-knee/wheel assists are now OFF, P05 recovery is v3, rear source timing and actual curriculum changed. Old geometry correction is confined to P09/P12, not this front window. New rear sigma gates depend on rear task states and are not front-specific multipliers. Original419 publication preserves old410 columns with new9 zero columns, but **CP222080 subsequently received1536/12/240 real updates**, including shared network parameters. Neither equal configuration nor initial zero-append guarantees current actions/closed-loop trajectories equal the old v7 run. Old v7 later failed P09 and is only a verified front reference, not a full-task or no-assist RR success.

## Stochastic training is not deterministic evaluation

The new512-decision natural-P01 block has P01=4/P02=508. Its completed-episode journal contains one physical terminal: episode0,15.658333s, P02 `INCOMPLETE_CONTROLLER_BLOCKED`; the other episode was a nonterminal sampling-boundary partial. The actual first learner request records `sampling_draws=1`, whereas deterministic evaluation uses the conditional mean. A failed sampled trajectory cannot prove the deterministic policy must fail; positive training-budget completion cannot prove task success either. P07-prefix success is also not natural-P01 student success.

Minimum next step: let the current fixed-checkpoint DET reach its actual sealed result. If front regression is confirmed, retain CP222080 and its genuine learning history. An explicitly selected initial419/front-validated ancestor may be evaluated/continued in a separately authorized output branch without resetting weights or borrowing1536 credit. Do not overwrite existing same-global-step history files or promote an ancestor over the current latest pointer. No branch selection, migration record or further training was performed by this check.

Additional evidence: `outputs/ppo_rr_capture_then_rl_transfer_v1/front_v7_v8_bounded_comparison.json`, old source transition/phase files, new natural-block `20260923T1908118345669Z_gfa4b98ed506e_1e77c2dd353745ebb86708c5095a34c9/completed_episodes.jsonl`, and the existing initial migration receipt. No later live logs were scanned.

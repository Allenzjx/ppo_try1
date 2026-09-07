# C54,656 natural-P01 evaluation — P05 incomplete, geometry not reached

Status: **finalized execution `SUCCEEDED`; task/controller success=false**. Root confirms process exit0. Run: `runs/ppo_semantic_v3/validation/20260906T1331439070323Z_g4d268fc547b7_95bda06bb52647c698f017a562c4a662`.

## Saved policy and actual outcome

Runtime `4d268fc547b704c590c5f21563ea4b1970b021a1`, seed2001, `semantic_residual_eval`, deterministic=true, natural P01, source `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000054656.pt`, SHA-256 `f24b35714c5b4881f78713739640f2130f8678216e8394d1fd41c91734c9aec7`. This is the saved heteroscedastic policy's fixed mean, not stochastic training or a resumed physical suffix.

Actual **640 decisions /5,120 physics ticks /42.6666666667s**, ending in P05 `INCOMPLETE_CONTROLLER_BLOCKED`, with0 optimizer updates. The shared physical evaluator is valid=true, success=false, termination_reason=null, reason empty. `window_ended_before_task_terminal=false`: P05 reaches its actual30s deadline, not an output-window truncation. Execution success is not task success; no body-collision, wheel-only-climb or sensor/interface failure is reported.

| Policy phase | Decisions | Outgoing transition tick / time |
|---|---:|---|
| P01 |1 |8 /0.066666667s |
| P02 |184 |1,480 /12.333333333s |
| P03 |4 |1,512 /12.6s |
| P04 |1 |1,520 /12.666666667s |
| P05 |450 |No completion; terminal5,120 /42.666666667s |
| P06–P13 |0 |Not visited |
| **Total** |**640** |**5,120 ticks** |

The manifest and complete640-row decision ledger agree. All5,120 native ticks report verified actual effect, with0 in-episode state-write issues. No new training decision/update credit is added by this evaluation.

## First incomplete physical task: FL placement after lift and crossing

FR Q/C/P=49/1,490/1,505; FL Q/C=**1,597/2,625**, but **no FL placement**. FL's qualification and crossing are genuine; they are not enough to mark current loaded TOP placement. At terminal:

| Leg | Current measured contact | Front distance | Bottom clearance above obstacle top | Load fraction |
|---|---|---:|---:|---:|
| FL |AIR; no ground or obstacle pair |+26.042524mm |+1.846840mm |0 |
| FR |TOP, obstacle pair active |+125.895146mm |−1.495869mm |0.42973977 |
| RL |GROUND |−556.250427mm |−49.373323mm |0.50701932 |
| RR |GROUND |−547.211381mm |−50.556457mm |0.06324091 |

FL has `top_geometry=true` but `top_contact=false`, obstacle_pair_active=false and consecutive_top_samples0. Its wheel geometry lies within the top region, but the live exact contact/load evidence does not establish placement. The unchanged evaluator requires crossing followed by qualifying loaded top contact for `minimum_top_samples=2`; no such placement event is recorded. The P05 completion value0.85 is continuous diagnostic progress, **not** a successful boolean or permission to advance. FR is currently TOP, while both rear wheels remain on ground and have no Q/C/P.

This identifies the first unmet task directly: FL must complete loaded placement after crossing. It is not a P06 rear workspace deficit, P09 knee-sign/entry pause, or P13 stop-command failure. The report does not infer that FL was permanently airborne at every intervening tick merely from the final AIR state, nor substitute CoM position for measured FL contact.

## Nominal geometry did not execute in this evaluation

The selected execution profile contains the new geometry opt-in, but the implementation activates only for RR/P09 or RL/P12. **Neither stage is visited**. The complete640 decision-end native audits contain0 geometry-bearing records; a streaming search of the entire `native_tick_audit.jsonl` finds no `nominal_geometry_evidence`, `geometry_adjusted_native_full12` or `nominal_geometry_adjustment_full12` fields.

Thus there is no direct geometry adjustment to blame for this evaluation's P05 terminal. In particular, this result cannot be described as a rear-leg projection stopping FL capture, or a restored knee-sign pause. The policy weights were trained in a changed execution/MDP setting, so absence of a projection in this rollout is **not** proof that earlier training-data changes have no indirect influence. There is no matched causal ablation here; neither improvement nor regression is assigned to a specific intervention on this evidence alone.

The preceding P06 training block's bounded live geometry effects remain genuine and separately scoped in `nominal_geometry_live_effect.md`; they are not effects observed in this natural-P01 evaluation. C50,560's earlier P09 incomplete and this C54,656 P05 incomplete remain distinct actual non-successes, not relabeled successful or paired-improved results.

## Reporting boundary

Latest completed full-P01 evaluation is now **C54,656 P05 incomplete**. Training counters remain **54,656/392/7,840**, fourteen finalized blocks and44,544/348/6,960 additions since origin10,112/44/880. No new successful/same-condition paired video or suffix/full-task success is claimed.

Main task is preparing a separate current-capture reward-only revision. This report records no new parameters, commit/test receipt, migration execution, training samples or result for that prospective change. All old evidence remains intact. Sources are this run's final manifests, stage transitions, full decision ledger and lexical native-audit geometry check; no Python, extra Isaac, production edit or commit was performed by this report task.

# First P13 episode after C82,560 — fixed tail-window diagnosis

Status: completed episode0, **P13 INCOMPLETE_CONTROLLER_BLOCKED**, not success. The enclosing P10 training block continues; this report does not update its optimized-boundary ledger, master report or future outcomes. No production/config changes, Python, Isaac, tensors or hashes were used.

Run: `runs/ppo_semantic_v3/train/20260906T2046002070189Z_ge99fde1b3e83_aa661e98d21b4ca484288779f0a064eb`, runtime e99fde1, source82,560, N1/seed1001/P10 offset0, same-MDP sampled-policy training.

## Fixed completed episode and credit boundary

Actual `completed_episodes.jsonl` episode_index0 records970 policy decisions, task_success=false/full_task_success=false, duration127.8666666667s, `INCOMPLETE_CONTROLLER_BLOCKED`. The terminal decision is global83,530: policy globals82,561–83,530. It is a real P13 age60s deadline with physical evaluation valid=true and physical termination_reason=null, not a collision or software/codec failure.

Teacher preparation ends at tick7,584/63.2s after948 decisions. The first policy tick interval ends7,592. Teacher data is not stored for PPO credit and the result scope remains `teacher_initialized_suffix`, not natural-P01 success.

| Transition | Tick | Episode time (s) |
|---|---:|---:|
| Teacher reaches P10 / policy takeover |7,584 |63.2 |
| P10→P11 |7,592 |63.266667 |
| P11→P12 |7,600 |63.333333 |
| P12→P13 |8,144 |67.866667 |
| P13 incomplete terminal |15,344 |127.866667 |

These transitions account for1 P10 +1 P11 +68 P12 +900 P13 decisions=970. The continuous post-teacher interval is7,760 ticks; teacher7,584 ticks remain separately excluded.

| Leg | Qualified Q | Cross C | Place P | Credit ownership |
|---|---:|---:|---:|---|
| FR |71 |1,665 |1,695 |Teacher; not learned in this suffix |
| FL |2,461 |3,115 |3,583 |Teacher; not learned in this suffix |
| RR |6,938 |7,109 |7,579 |Teacher; not learned in this suffix |
| RL |7,961 |8,063 |8,138 |Actual current-policy events after takeover |

## Selected last120 P13 decisions

Only **decisions851–970 /globals83,411–83,530**, plus decision850 as the residual-slew predecessor, are used for numerical tail statistics. Their executed ticks are14,385–15,344 (960ticks/8s), with decision endpoints14,392–15,344,119.933333–127.866667s. All following state/command frequencies are **120 decision-end observations**, not fabricated full-rate sensor traces. Reward costs are the existing eight-tick integrated records. Compact native evidence covers all960 actual ticks, verified=true/own-phase=true; all four in-episode state-write categories are0.

The first streaming aggregation encountered the compact native-record schema (it does not contain full target arrays) and stopped without publishing results. Recovery parsed only the first row and the fixed tail/predecessor, not the second episode or another full-episode JSON pass. Full native float32 target arrays are available at decision ends; compact per-tick verification is kept distinct from independent full-rate array reconstruction.

## Nominal is zero; the learned distribution still requests rolling

All four nominal wheels, mapped native nominal wheels and controller-wheel biases are0 at every selected endpoint. Final nominal Full12 is entirely0. Actual canonical drive wheels equal their projected residuals at all120 endpoints; the nominal is not forcing a residual cancellation problem in this tail.

Units: raw mean/std are latent Gaussian values; transformed requests and actual drive are rad/s. `0.6*tanh(mean)` is a static transformation of each stored behavior distribution mean on its actual observation, **not a fresh deterministic rollout** and not a claim that an average Gaussian action equals its transformed mean.

| Wheel | Raw mean range | Mean raw std | 0.6*tanh(mean) range | Actual canonical drive range | Mean actual drive |
|---|---|---:|---|---|---:|
| FL |+0.142340…+0.298172 |0.180630 |+0.084832…+0.173783 |−0.052411…+0.293432 |+0.119257 |
| FR |−0.903871…−0.598349 |0.231059 |−0.430907…−0.321524 |−0.521882…−0.175228 |−0.376855 |
| RL |−1.133905…−0.692578 |0.214770 |−0.487411…−0.359781 |−0.539609…−0.312469 |−0.443782 |
| RR |−0.185808…+0.206329 |0.204941 |−0.110219…+0.122070 |−0.197215…+0.298844 |+0.017950 |

Raw sampled ranges FL[−0.098966,+0.636998],FR[−1.415151,−0.092626],RL[−1.468797,−0.534715],RR[−0.385846,+0.546741] produce tanh-request ranges respectively[−0.059187,+0.337711],[−0.533150,−0.055417],[−0.539609,−0.293386],[−0.220664,+0.298844]. FR and RL **never even sample a request within ±0.02rad/s** in this window; their stored distribution means likewise never imply a near-stop request. Neither the sampled requests, transformed means nor actual commands have all four wheels within stop tolerance on any selected decision.

### Projection/rate limit is real, but does not explain away those requests

Current P13 cap is±0.6rad/s per wheel; residual slew is1.8rad/s², or±0.015 per120Hz tick /±0.12 per full decision. For all480 wheel-channel endpoints, the logged projected residual matches the following recurrence within1e−8, including the actual preceding decision850:

`r_end = clamp(0.6*tanh(raw), r_previous_end−0.12, r_previous_end+0.12)`.

Sample request differs from the final residual on FL36/FR29/RL19/RR47 decisions. Thus it would be false to say there is no clipping: **slew limiting is active**. However, no additional wheel headroom/safety clipping is needed to explain these endpoints. With zero wheel nominal/bias, the actual maximum magnitude0.539608629rad/s is below both requested0.6 and the projector's exact physical bound2.094395102rad/s (`action_projection.py:68`), leaving at least1.554786474rad/s physical margin. Zero lies inside both intervals; this is not a static inability to cancel a nonzero nominal. These observations do not establish which policy update caused the behavior or prove a future zero request would meet every physical stop condition.

Within-tolerance counts for individual FL/FR/RL/RR wheels are: transformed mean0/0/0/31; sampled request10/0/0/18; actual drive4/0/0/20 (each denominator120). No independent-Gaussian probability is treated as an actual success rate.

## Several current physical conditions are missing

| Condition / existing hard threshold | Passes out of120 endpoints | Recorded range |
|---|---:|---|
| All historical placements |120 |true |
| final_region_valid |110 |10 geometry/region misses remain |
| final_support_available |120 |true; at least two genuine TOP supports |
| Maximum commanded wheel speed≤0.02rad/s |0 |0.314782…0.539609 |
| Maximum measured wheel speed≤0.25rad/s |0 |0.302466…0.588188 |
| Body linear speed≤0.05m/s |45 |0.006154…0.188356 |
| Body angular speed≤0.30rad/s |104 |0.017306…0.485069 |
| final_controlled |0 |false |
| final_stable_for_s≥0.5s |0 |maximum recorded duration0 |

Final region=true/support=true, but body linear speed0.075553194m/s, measured-wheel maximum0.464481831rad/s and commanded-wheel maximum0.427047335rad/s still exceed their thresholds. Body angular speed0.076404233rad/s passes. It is not accurate to describe the terminal as failing only a nominal/cap constraint, nor to assume every other stop prerequisite was always satisfied.

Historical all-placed is not current four-wheel support. Tail endpoints: FL TOP120/load mean0.439967; RR TOP120/load mean0.498006; RL TOP94/AIR26/load mean0.060601; FR TOP2/AIR118/load mean0.001426. Terminal FL/RL/RR are genuinely TOP with loads0.433494/0.094963/0.471543; FR is AIR/load0 with16.035198mm bottom clearance. These measured states support the reported support=true without pretending all four wheels remain loaded.

Final wheel raw sample is `[+0.353007466,−0.357944489,−0.890712678,+0.347154081]`, stored mean `[+0.234656036,−0.706217706,−0.856802821,+0.087344520]`, std `[0.182672530,0.232782036,0.204211220,0.199586242]`. Projected/actual canonical drive is `[+0.203424010,−0.206041320,−0.427047335,+0.118204626]rad/s`. Frozen wheel signs yield real float32 setter targets `[−0.203424007,−0.206041321,+0.427047342,+0.118204623]`. Last RR requested sample is rate-limited; do not substitute the sampled request for its actual dispatch.

## Recorded reward and narrowly scoped source facts

The existing four-type stop progress (`semantic_supervisor.py:584–618`) averages measured wheel, applied-command wheel, body-linear and body-angular ratios `T/(T+abs(value))`; it replaces the legacy aggregate rather than adding a second command term. In this tail its reconstructed stop component is0.333092–0.623345 (mean0.448187). Recorded whole_task_success soft goal is0.766618–0.824669 (mean0.789637); global potential0.964993–0.973700. Those are shaping values, not hard success or stable-time credit.

| Weighted reward family | Sum over120 decisions | Mean per decision |
|---|---:|---:|
| task_progress |−47.890039552 |−0.399083663 |
| body_stability |−0.045015301 |−0.000375128 |
| contact_motion_quality |−0.000432948 |−0.000003608 |
| control_smoothness |−0.326483493 |−0.002720696 |
| control_regularization |0 |0 |

Task_progress includes the last genuine terminal event−40 and potential-to-zero shaping−4.854994305; terminal task_progress is−44.856327638. Excluding that terminal, its119-decision mean is−0.025493377. The terminal is not removed from training or reclassified. Existing diagnostics record37 touchdown events, confirmed-post-touchdown-rebound0, nominal-first-difference0, integrated actual-first-difference5.173573307 and second-difference1.356096560; contact chatter0.666667 is diagnostic, not an extra configured reward family.

Source inspection only: `semantic_reward.py:165–191` and current `reward_config.yaml` use applied first/second differences for control_smoothness; residual magnitude regularization is disabled and its weight0. There is **no separately accumulated absolute final-wheel-command cost** in these current families. Applied wheel commands already enter the continuous stop potential and unchanged hard stop predicate. Body stability uses attitude/rate/angular acceleration, contact quality uses actual contact-motion terms. This fact is not evidence that PBRS is erroneous or that these observed correlations prove a reward conflict. No new cost, gate, coefficient or architecture is implemented or endorsed here.

## Conclusion and evidence limits

RL lift/cross/place is real current-policy suffix progress. The final eight seconds have zero wheel nominal, but persistent nonzero learned mean/sample rolling requests, correctly rate-limited into actual drive. Both command and measured-wheel stop limits fail throughout the selected endpoints; body motion and occasional region misses also matter. This establishes a concrete current control/state pattern, **not a causal proof of why PPO learned it or a predicted benefit from a candidate cost change**. Full-task and suffix success remain false. The enclosing block, later episodes and optimizer ledger were not analyzed or credited by this report.

Sources: this run's actual first `completed_episodes.jsonl` entry, fixed selected records from `residual_and_projection_audit.jsonl`, current `semantic_env.py:24–46`, `action_projection.py:568–630`, `semantic_reward.py`, `semantic_supervisor.py`, `configs/ppo_semantic_v3/execution_profile.yaml`, `reward_config.yaml` and `stage_task_spec.yaml`. All historical evidence is untouched.

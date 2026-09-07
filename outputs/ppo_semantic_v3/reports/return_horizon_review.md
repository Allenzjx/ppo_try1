# Return-horizon review — current f1a9 configuration, read-only

Scope: current local production/configuration and installed official RSL implementation, plus JavaScript arithmetic only. No active block32 stream, historical trajectory scan, checkpoint/PT/hash, Python/GPU/Isaac execution or parameter change. This is a later-choice aid, not a gate or an empirical diagnosis of the unique failure cause.

## Actual units and wiring

- [Reward configuration](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/configs/ppo_semantic_v3/reward_config.yaml:8): gamma=.995, physics120Hz, policy15Hz, task200s, terminal potential0, potential weight5, success event+40/failure event−40, time cost.02 per actual physical second. Task family weight1; body/contact/smoothness/regularization weights.4/.2/.1/0. The ±40 is the **event**, not the entire terminal reward, which also contains potential removal and elapsed-time/quality costs.
- [Runner config](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:101) uses matching gamma=.995, lambda=.95, rollout128, with [the wrapper](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/rl_library_wrapper.py:637) passing these values to official PPO. Lambda is a return-estimator parameter, not an additional discount in the environment objective.
- [Environment](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_env.py:147) executes up to8 actual1/120s ticks per issued action, stops early on true termination, then calls reward evaluation **once**. [Reward](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_reward.py:174) integrates quality costs using each actual dt, and computes `F=5*(.995*Phi_next−Phi_before)` once per decision. Time cost is.001333333333 per normal8-tick decision, smaller for a shortened terminal interval. It does not charge.02 per tick or apply gamma eight times.
- Ordinary phase changes do not set done or reset the global potential. Actual task success, failure and task deadline are terminal, nextPhi0, no bootstrap. The explicit convention is one gamma per issued policy action, including a shortened final action; this is decision-discounting, not exact continuous-time discounting.

No mismatch among the inspected current values or 120Hz/15Hz aggregation was found. This is a source-level conclusion, not a new physical test.

## Pure-arithmetic delay table

For normal-duration decisions, `n=15*seconds`; gamma*lambda=.94525. The first column is the relative objective weight of the same reward moved n decisions later. The second is the **formal coefficient of one TD residual n positions later in an uninterrupted GAE trace**, not the actual contribution assigned across saved rollout boundaries.

| Delay (s) | Decisions n | gamma^n | (gamma*lambda)^n |
|---:|---:|---:|---:|
| 5 | 75 | .6866430932 | .01465552743 |
| 10 | 150 | .4714787374 | 2.147844842e−4 |
| 20 | 300 | .2222921998 | 4.613237465e−8 |
| 30 | 450 | .1048060457 | 9.908518294e−12 |
| 60 | 900 | .01098430722 | 9.817873479e−23 |
| 120 | 1800 | .0001206550051 | 9.639063964e−45 |

Gamma's half-life is9.21884s; the formal direct TD-trace half-life is.82069s. They describe different quantities. Rollout128 spans8.533333s at normal intervals; gamma^128=.5264466124 and (.94525)^128=.000741268956.

[Official local PPO](C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/rsl_rl/algorithms/ppo.py:187) computes `last_values=critic(next_observation)`, then backward `delta=r+gamma*(1−done)*V_next−V` and `A=delta+gamma*lambda*(1−done)*A_next`. A starts at0 at each finite rollout end. [Training](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_training.py:745) calls this on the actual next observation after128 transitions, then updates; ordinary nonterminal physical state continues into the next collection.

Consequently the table's long direct traces are **not literally carried across128-step batches**. Nevertheless,128 is not a hard8.53s cutoff of the value objective: nonterminal critic bootstrap can represent rewards far beyond the batch. True terminals mask both bootstrap and trace; reset observations cannot leak value backward through done. Whole-batch advantage normalization also means table coefficients are not a direct prediction of final normalized actor-gradient magnitudes.

The main training limitation supported by this wiring is reliance on an accurately learned critic to transmit later rear crossing/placement/stop consequences back across many batches to earlier preparation. Weak direct trace or imperfect value estimation can limit that transmission. Neither this arithmetic nor a short rollout proves learning is impossible, that a particular reward conflict is the root cause, or that a particular parameter change will fix it.

## Correct PBRS telescoping and its interpretation

Let a trajectory have transitions t=0,…,T−1, true terminal state s_T, matching objective gamma, and shaped reward `r'_t=r_t+5*(gamma*Phi(s_(t+1))−Phi(s_t))`. Here r includes the event, actual-time cost and the other reward families. Then exactly:

`sum(t=0..T−1) gamma^t * F_t = 5*(gamma^T*Phi(s_T)−Phi(s_0)) = −5*Phi(s_0)`.

Thus `G'=G−5*Phi(s_0)` for a true terminal withPhi0. For a fixed start, matching-gamma PBRS alone does **not** add a permanent preference for a failed trajectory that temporarily reached more stages, and an absorbing terminal must not retain its preterminal progress potential. It redistributes local feedback: increasing physical progress can produce a useful dense transition signal for approximate value/policy learning, while later retreat and terminal potential removal balance it in the full discounted sum. This does not mean all failed trajectories have equal total return; their base events, durations and physical-quality costs differ.

At a nonterminal rollout boundary H the partial shaping sum instead contains `+5*gamma^H*Phi(s_H)`. Correct shaped-value bootstrap, `V'(s_H)=V(s_H)−5*Phi(s_H)`, cancels that boundary term in the bootstrapped target. In practice the critic is learned, not exact: a dense local reward does not guarantee correct long-range credit. A negative individual PBRS increment, or its negative terminal correction, is not by itself a bug. No extra stage/FL/AIR/entry bonus is justified by this review.

## Failure-delay effect: theoretical, not observed reward hacking

A transparent counterexample isolates discounting from PBRS. Suppose the same start has unavoidable failure after T normal decisions, no other quality costs, constant per-decision time cost `c=.02/15`, and terminal event−40 on transitionT−1. Its shaped discounted return is:

`J_fail(T)=−c*(1−gamma^T)/(1−gamma)−40*gamma^(T−1)−5*Phi(s_0)`.

Delaying that failure one normal decision changes return by

`J_fail(T+1)−J_fail(T)=gamma^(T−1)*[40*(1−gamma)−c*gamma] = .198673333333*gamma^(T−1) > 0`.

So in this deliberately restricted example, delaying a negative event is favored despite the small time cost; PBRS does not create or remove this preference. For an otherwise identical success event+40, the one-step change is `−.201326666667*gamma^(T−1)`, favoring earlier success. Actual behavior includes state-dependent quality costs, possibility of future success, finite200s deadlines, varying starts and occasionally shortened terminal decisions; this is not a comparison of actual trajectories. It does not establish that the policy deliberately stalls, that all late failures outrank successes, or that a measured reward exploit occurred. The loader's failure-cost bound is not a proof of every possible long-delay success/failure ordering.

No gamma/lambda/rollout/event/time/entropy/std/reward modification is selected. No simulation, optimizer count or future training credit is added. Only this report is written; review complete and stopped.

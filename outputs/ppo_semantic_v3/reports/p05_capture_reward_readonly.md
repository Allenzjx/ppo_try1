# P05 FL capture reward continuity — bounded read-only analysis

Status: code/formula inspection and selected rows from completed C68224 only. No Python, simulator, optimizer, production/config edits, or new physical intervention. This is not a training gate or a selected reward revision. The master remains `outputs/ppo_semantic_v3/reports/training_report.md`; it is deliberately not updated during the currently running73e937 P01 block.

## 1. Answer and attribution boundary

**Yes: with measured top XY fixed, FL still AIR, qualified+crossed, no placement and zero TOP samples, `placed_FL` is exactly0.85 throughout +0.4mm to +25mm top clearance. More importantly, the actual v3 global reward potential also has no continuous positive-clearance approach-to-touchdown term in this state.** This is an algebraic plateau, not proof that it caused the failed rollout or that touching the surface is safely reachable by one chosen joint command.

Completed C68224 used68cd9f5, naturalP01/seed2001/fixed learned mean,632decisions/5056ticks/42.133333s, P05 incomplete, physicalvalid=true/failure=null,optimizer0. It earned FL Q1532/C2561 but never placement. The independent [completed diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/eval_68224_diagnosis.md) retains the full evidence. Current73e937 changes the P06 nominal tail and prefix diagnostic forwarding; the inspected68cd→73e diff does not change the P05 predicate/potential/reward formulas below. C68224 never entered P06. Neither that future tail branch nor its newly running training can retrospectively explain this P05 result.

## 2. Predicate versus actual reward potential

[TaskStageSupervisor.predicate](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:620) computes, before actual placement:

`placed_FL = min(.99, .35*lift + .35*crossing + .30*(.5*top_geometry + .5*clip(top_samples/2)))`.

For qualified+crossed AIR, `lift=crossing=1`, `top_samples=0`. The existing top geometry accepts measured XY and clearance[-15,+25]mm, inclusive. Therefore +0.4→+25mm gives .35+.35+.15=.85 with zero clearance derivative inside that interval. Above+25mm it drops to.70 if all other conditions remain fixed; this is a geometry-membership step, not a smooth descent incentive. At one genuinely loaded TOP sample it becomes.925; at the second actual consecutive loaded sample the placement event makes it1. Those are physical events, not values that may be inferred from distance alone. The [real loaded predicate](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:440) still requires obstacle pair active, valid top geometry and front distance≥0; no change is proposed here.

However, **.85 is not itself the reward potential**. The selected [global physical potential](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:648) uses an unfinished, predecessor-eligible leg contribution:

`V_FL = .1*workspace + .1*unload + .25*(.25*initial + .75*lift_credit) + .35*carry + .2*capture`, with `Phi=.85*sum(V_leg)/4 + .15*finish`.

After FL Q+C, `lift_credit=1` and `carry=1`; AIR gives `capture=clip(top_samples/2)=0`. In this rollout FL is unloaded with other supports available, so `unload=1`; workspace is fixed and valid. Thus `V_FL=.8`, independent of positive clearance. The [lift-credit branch](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_supervisor.py:694) explicitly returns1 after qualified crossing; its below-top soft lift scale does not supply a post-cross descent gradient. GlobalPhi need not even step down at+25mm as the separate phase predicate does. Other bodies/legs may change Phi in a real movement, so “fixed-state clearance-only plateau” must not be generalized to every physical trajectory.

The earlier capture-retention term applies **only after `history.placed=true`**. It cannot provide the missing first-touchdown approach term here. It also intentionally allows AIR above the platform after placement; it is not a permanent-contact reward. P13 finish contributes0 until all four placements, so its new stop formula is inactive in this P05 state. The code emits no phase-label transition bonus, reference imitation reward, or nonzero-residual bonus.

## 3. Selected actual reward rows, not a counterfactual rollout

Source: completed `validation/20260906T1628525452104Z_g68cd9f5fca6c_fac838181e114f68954319480dae7b77/residual_and_projection_audit.jsonl`. Only7 selected decision rows were decoded for this analysis; the prior independent raw-contact diagnosis was reused.

| Decision / tick | FL clearance (mm) | placed_FL / Phi | Task-progress family | Body / applied smoothness families | Contact family |
|---|---:|---|---:|---|---:|
|329 /2632 |+21.662717 |.85 /.3825 |−.0108958333 |−.0000140543 /−.0003470179 |0 |
|332 /2656 |+.493055 |.85 /.3825 |−.0108958333 |−.0000939663 /−.0006574721 |0 |
|333 /2664 |+1.215441 |.85 /.3825 |−.0108958333 |−.0001128029 /−.0006696866 |0 |
|400 /3200 |+1.929556 |.85 /.3825 |−.0108958333 |−.0000780290 /−.0006250016 |0 |
|631 /5048 |+2.222275 |.85 /.3825 |−.0108958333 |−.0000719045 /−.0006250006 |0 |

All listed FL samples are AIR/obstacle inactive/load0/top_samples0. Raw tick2657 is the closer +.402172mm sample,front+27.759853mm,still no force/contact; it is between decision-end records and must not be relabeled a touchdown. Actual neighboring XY/pose/action values vary, so these rows support the plateau but do not constitute a controlled clearance-only experiment.

At constantPhi=.3825, the [reward formula](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_reward.py:180) gives `5*(.995*Phi-Phi)=−.0095625`, plus time cost `−.02*(8/120)=−.0013333333`. This exactly explains task-progress−.0108958333. Waiting is not positive reward harvesting. Terminal632/t5056 uses absorbingPhi0,shaping−1.9125,event−40,time−.0013333333:task family−41.9138333333,total−41.9145307198. This gives a delayed incentive to avoid incompletion, but no local geometric direction from +25mm toward contact.

## 4. What existing terms can favor or oppose a gentle capture

- **Actual capture improves task potential.** With all other eligible FL terms held fixed, first TOP sample adds.1 to its leg contribution, or.02125 toPhi; two-sample placement replaces the incomplete contribution with completed-region credit. For the observed starting `V_FL=.8` and retained region1, final placement gives a net.2 leg/.0425Phi increase before any downstream preparation changes. This is delayed/contact-count progress, not a smooth distance-to-contact slope.
- **Unload and transfer weighting are state-dependent.** Before placement, `load_ready_FL` stays1 for load≤.20 but can fall above it. AIR has physical_transfer_fraction1,discounting attitude/contact terms. Loading can increase their weight; body rates/acceleration are still charged independently. These are possible short-horizon tradeoffs, not proof of a trained reluctance to touch down. Once placed, the incomplete unload contribution is replaced by completed-region credit.
- **Contact cost does not penalize quiet contact just for existing.** [Touchdown impact](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/src/wlr50_clean/ppo/semantic_reward.py:127) only taxes pre-contact downward speed beyond.05m/s; load scales that impact,not an independent static force cost. Slip is taxed only above.15m/s. A sufficiently slow nonslipping landing need not incur either. Rebound is charged only under the conservative recent-touchdown/unchanged-command condition; contact chatter is diagnostic in this selected mode. Aggregate `touchdown_events=1` at some selected rows refers to all four wheels, **not FL**; FL remains AIR and the actual total contact family is0 there.
- **Smoothness can oppose target changes, but is not a command-magnitude penalty.** Current mode taxes first/second differences of actual mapped drive targets,not the separate nominal/residual difference diagnostics. At d400/d631 nominal difference is0 and residual-difference integrals are around1e−8,while mapped-drive smoothness still contributes about−.000625. Therefore one cannot assign all this cost to exploratory residual movement; frozen feedback also changes actual targets. A different static residual has no direct magnitude cost: control_regularization is disabled/weight0. The code has no explicit “keep the leg AIR” reward.
- **Safety remains physical.** A harder/faster descent can add impact, slip, body collision or lost support. Neither the formal plateau nor a small positive cached geometric gap authorizes forcing contact, removing the two-sample capture requirement, or declaring sensor error. Current full failure penalties are retained.

These statements identify terms and algebra, not their causal contribution to this policy's learned value or actual success probability. No gradient tracing, action intervention, or new rollout was performed.

## 5. Finite P05 nominal and remaining physical controls

The frozen `recording_motion_contract.json` P05 source has active_duration9.733333s. Its last knee target reaches−13.4° at9.4s; the subsequent hip advice45.0→24.9→22.8° ends at9.733333s. The source's physical-purpose text explicitly records that historical top load latches during the immediately continuous P06 wheel advance. That source commentary is historical context,not permission to grant semantic placement without contact or replay an exact posture.

Actual C68224 at d320/t2560 still has nominal FL[48.2,−30.4]°. By d329/t2632 it has the final full12 **[22.8,−13.4,0,45.9,6.9,0,0,0,0,0,0,0]**,which is unchanged in every later selected row through terminal. Thus near the closest recorded gap the nominal source has reached its finite final target; the servos themselves are **not disabled** and their measured response need not be motionless. At d631,FL recent joint excursion is still3.061016° and whole-body excursion10.375765°. Source endpoint is advice,not a residual cutoff.

All12 residual channels remain open in P05 under the existing [execution profile](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/configs/ppo_semantic_v3/execution_profile.yaml:23): front hip/knee caps18°/24°,rear12°/18°,wheel.3rad/s; residual slew60°/s and1.8rad/s². At120Hz that is at most.5°/.015rad/s per tick before other projection constraints. These are requested ranges intersected with mapper/absolute headroom and physical safety,not guaranteed usable Cartesian displacement.

Actual near t2656 projected FL residual is[+4.668433,+.081677]°; final it is[+5.316688,+.555592]°. Final wheel residuals are[−.00598130,−.01066752,+.02495894,+.06776555]rad/s,not disabled. Final reported mapped logical FL drive targets are[22.830140,−11.594408]°; they are not simply nominal+residual because frozen mapping/feedback remains in the path. No Jacobian or physical perturbation here establishes which sign/magnitude of FL hip/knee,other-leg support adjustment or rolling would safely lower the real wheel. Their availability does not prove sufficient capture authority. The current geometry-advisory implementation is explicitly scoped toP09/RR andP12/RL,not P05.

## 6. Bounded possible follow-up, not selected or implemented

1. After the active block and its saved P01 evaluation, a CPU-only counterexample test can hold a **validated post-cross/pre-placement** state fixed and sweep positive gap through.4/1/5/15/25mm. Assert both present predicate/Phi plateau and unchanged hard Q/C/P/termination,then inspect any separately proposed shaping candidate. This is a formula diagnostic,not a physical rollout or a new optimizer gate.
2. If later authorized, compare a small pair of physical interventions at an equivalent real natural-prefix P05 capture window: unchanged actor/action versus one bounded smooth target perturbation. Keep nominal,hard capture,contact sensing,source checkpoint and all unrelated rewards fixed; record native targets,real support/contact/body motion and target-to-geometry response. Do not infer intervention effect from direct state rewriting or merely changing logged clearance.
3. Only if that evidence warrants a reward-only experiment, consider reusing part of the existing capture share for a smooth bounded post-cross approach-to-surface signal,explicitly retaining an unresolved-contact remainder until the actual loaded samples occur. Reject candidates that award placement by distance,pressure a pre-cross leg downward,penalize legitimate later AIR motion,or double-count the current capture weight. Do not simultaneously change nominal/caps/clock or claim PPO-only improvement across an MDP revision.

Nothing here alters the running73e937 experiment or selects the next change. C68224 remains a valid P05 incomplete trajectory,not successful capture,not definitive sensor error,and not a causal test of the P06-tail repair. Analysis complete; no further process remains running from this task.

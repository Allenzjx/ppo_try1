# Nominal-geometry P06 block finalized at54,656 — no task success

Status: **FINAL `SUCCEEDED` training execution; two task incompletes plus an unfinished tail**. Root confirms process exit0. Actual2,048 decisions /16 PPO updates /320 optimizer steps advance source52,608/376/7,520 to **54,656/392/7,840**. This is neither suffix nor natural-P01 success. Earlier52,736/53,120/53,248 bounded evidence remains historical subsets, not additional spending.

## Bound run, checkpoint and execution epoch

Run: `runs/ppo_semantic_v3/train/20260906T1308472858273Z_g4d268fc547b7_dcefe730da37448599db863a5f261d47`. Runtime `4d268fc547b704c590c5f21563ea4b1970b021a1`, v3/N1/seed1001, P06/teacher offset0, `phase_suffix`, requested2,048, save cadence4 updates. The actual new-MDP initial preservation and first update are documented in `nominal_geometry_implementation_audit.md` and the master report; source actor/learned std, critic, identity normalizer and RNG were retained, with fresh Adam and rollout. This is not a same-MDP exact resume or a policy-distribution conversion.

Final run/training manifests record planned=requested=actual2,048, unconsumed0, rounding_overrun0 and wall time **1,137.38724800013s**. The checkpoint `outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000054656.pt` is bound by SHA-256 `f24b35714c5b4881f78713739640f2130f8678216e8394d1fd41c91734c9aec7`; sidecar SHA-256 is `609e2686d8288ff3ec01ee7197fb23a31f59fefb58113c26cb0b700c4e9a6566`. Root measured both final files; this audit independently reads the matching checkpoint SHA and `save_load_round_trip=true` from the sidecar.

Lifetime54,656/392/7,840, original origin10,112 and actual spending **full_episode16,896 /phase_suffix27,648 /smoke0** are preserved. The curriculum epoch is fixed P06, reset sampling `natural_P01_A_teacher_prefix_then_semantic_suffix_N1.v2:P06:offset_0`; changes are permitted only between complete rollout updates. Ordinary phase transitions do not end an episode. No future budget or evaluation outcome is filled in.

## Optimizer, action-density and native audit

A read-only PowerShell pass checks all2,048 policy records, globals **52,609–54,656**, with no gaps or extra rows. Policy ticks total16,384, all verified and all reporting actual native effect; own-phase-request-effect ticks total16,372. The12-tick difference corresponds to one incoming hold tick across the12 ordinary phase transitions in the three sampled episodes, not12 missing dispatches. All per-decision no-state-write flags are true and all teacher-in-PPO-storage flags are false.

Updates **377–392** each advance128 decisions and execute20 optimizer steps. The16-update actor before/after hash chain has0 mismatches, starts at the preserved source actor and ends at the final sidecar actor digest `cca90f60744f0e00068068c9160cd06648a129a881df9cb010380a17dc716f5d`. Every update reports changed actor parameters and finite nonzero gradients. Final critic digest is `1ed87c38bd86953a2e5d1b91fc96ef0581650d1f5bcf08d8a9f00e34455dce6d`; identity normalizer remains `c230b0db34453fa8047231a56598e8f56fcc7e13d9af6f863b64662b833f4552`.

The recorded PPO configuration has5 learning epochs ×4 minibatches, consistent with20 optimizer steps per rollout update; gamma0.995, GAE lambda0.95 and clip0.2 remain. Initial LR3e-5 is distinguished from the **actual1e-5 effective LR in all16 update receipts** under the configured adaptive schedule.

All2,048 stored raw actions, distribution means and standard deviations have12 finite components with positive std. Recomputing the diagonal Gaussian latent log density from each recorded raw action/mean/std gives maximum absolute difference **1.3468938224736604e-6**, at global53,823. This is an action-probability bookkeeping check, not an estimate of task-success probability; no product of per-channel stop probabilities or causal sigma claim is made.

## Policy versus teacher ledger

| Episode | Current-policy globals | P06 | P07 | P08 | P09 | P10 | P11 | P12 | P13 | Outcome |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
|0 |52,609–53,374 |313 |1 |2 |450 |0 |0 |0 |0 |766 decisions; P09 incomplete |
|1 |53,375–54,269 |284 |1 |1 |147 |11 |1 |450 |0 |895 decisions; P12 incomplete |
|2 tail |54,270–54,656 |375 |1 |1 |10 |0 |0 |0 |0 |387 decisions; P09 unfinished |
|**Total** |**52,609–54,656** |**972** |**3** |**4** |**607** |**11** |**1** |**450** |**0** |**2,048** |

P01–P05 have0 current-policy samples. Three accepted teacher prefixes each execute448 decisions /3,584 ticks: **1,344 teacher decisions /10,752 ticks**, separately verified to have zero raw/projected residual and policy_credit=false. Full physical core totals **3,392 decisions /27,136 ticks**. Current PPO credit begins after each actual P06 takeover at3,584/29.866666667s; no exact historical snapshot is inherited.

Teacher FR Q/C/P=71/1,665/1,695 and FL=2,461/3,115/3,583 precede takeover and are not PPO credit. The RR/RL events below occur after policy takeover and belong to the current suffix, but do not by themselves constitute a full-P01 policy success.

## Actual terminal tasks and rear-leg histories

### Episode0: RR carry/placement still incomplete

It terminates after766 current-policy decisions, tick9,712 /80.933333333s, P09 age30s, `INCOMPLETE_CONTROLLER_BLOCKED`. Physical evaluation is valid with no physical-failure reason; task/full success=false. The outgoing target remains `placed_RR`, not a blocked old P10 joint-entry guard.

RR achieves **three** qualified upward lifts, then returns to ground before crossing each time:

| RR qualified tick | Revoked-on-ground tick | Crossing/placement |
|---:|---:|---|
|6,999 |7,065 |None |
|7,913 |8,008 |None |
|8,460 |9,683 |None |

At terminal RR is GROUND, front−103.930440mm, clearance−50.016994mm, load0.10415228. RL remains GROUND/front−244.953212mm/load0.33404614. FL and FR are current TOP with loads0.16129241/0.40050918. Thus transient above-top qualification is not retained across the ground-before-cross interruptions, and real front support at terminal does not complete RR carry/placement. RL has no post-takeover qualified/cross/placed event in this episode.

### Episode1: RR placement earned but current support lost; RL placement unfinished

It terminates after895 decisions, tick10,744 /89.533333333s, P12 age30s, `INCOMPLETE_CONTROLLER_BLOCKED`; physical valid=true/no physical-failure reason, task/full success=false. RR Q/C/P is **6,112/6,536/7,048**, all current-policy suffix events. RL qualifies at7,514, but ground-before-cross revokes it at7,713; **no RL crossing or placement** follows. The first unfinished P12 task is actual RL crossing/placement after that interrupted lift, not a historical angle or motion-clock gate.

At terminal both rear wheels are GROUND: RR front−134.861019mm/clearance−49.490298mm/load0.10858906; RL front−93.030747mm/clearance−51.167959mm/load0.54039526. FR is TOP/load0.35101568; FL is AIR/load0 with clearance+14.328865mm. RR's stored placed history is genuine, but cannot be described as sustained current RR support during the whole RL attempt.

### Episode2 tail: a new RR qualification just before the saved boundary

The387-decision tail ends at tick6,680 /55.666666667s in P09, with only10 P09 policy samples, terminal=false/termination=null. RR qualifies at **6,678/55.65s**, two physics ticks before the saved observation, but has no crossing/placement. At the boundary RR is AIR, clearance+2.853037mm and front−234.929320mm; the front plane remains far ahead. RL is GROUND/load0.53235943, FR is TOP/load0.46764057, and FL is AIR/load0 with clearance+119.173070mm. This is newly measured qualification, not a completed third failure, successful rear-leg traversal or task success.

## Actual geometry exposure: reuse bounded proof without overstating it

`nominal_geometry_live_effect.md` already analyzes globals52,925–53,030:81 geometry-bearing decision-end audits out of106 selected records. Its status counts are42 identity,8 projected-exact-forward,16 projected-relaxed-forward and15 explicit degraded bypasses. All81 selected audits verify real setter equality, exact g accounting and separate same-state PPO-versus-geometry float32 target differences; recorded live COM/link velocity identity errors remain small. **This subset is not relabeled as an all16,384-tick geometry analysis.** Degraded bypasses have no clearance guarantee, and native inverse-target adjustment degrees are not actual one-tick physical joint displacement.

`nominal_geometry_live_transition.md` separately binds the first640 decisions/5,120 native ticks through53,248. It verifies continuing P06→P07→P08→P09 returns, one-tick incoming residual holds followed by own-request effects, current measured load-driven short phases, and no episode reset at ordinary handoff. Its observations of FL AIR/load0 are current contact facts, not claims that FL is always airborne in later episodes.

These existing live reports prove exercised interfaces and bounded continuity. They do not guarantee next-tick height under contact, prove motor-specific causality, or attribute any old/new trajectory difference solely to PPO. Nominal geometry changes zero-residual B as well as C. The completed block still has **zero suffix/full-task successes and zero P13 samples**; no improved-checkpoint or successful-video label follows.

## Delivery state

Fourteen finalized blocks now add **44,544 decisions /348 PPO updates /6,960 optimizer steps** since origin10,112/44/880. The previous52,608 recovery files and bounded migration/update records remain preserved. A new saved54,656 natural-P01 evaluation was launched by the main task; it is pending in this report, not an outcome. Latest completed full-P01 evaluation remains **C50,560: P09 incomplete,1,166 decisions /9,328 ticks /77.733333333s**.

This final audit used only read-only PowerShell and the named outputs reports. No Python, extra Isaac process, production edit, historical-run rewrite or commit was performed by this report task. New evaluation results are left for the main task's actual completion notice.

### Subsequent saved-policy evaluation completed

The pending evaluation above subsequently finalized: C54,656 natural-P01/seed2001 under4d268fc, run `validation/20260906T1331439070323Z_g4d268fc547b7_95bda06bb52647c698f017a562c4a662`,640 decisions /5,120 ticks /42.666666667s, **P05 `INCOMPLETE_CONTROLLER_BLOCKED`**, taskfalse and0 optimizer updates. FL qualified/crossed but never placed; no P09/P12 geometry invocation occurred. Full details and attribution limits are in `eval_54656_diagnosis.md`. This subsequent evaluation changes the latest completed evaluation status, not the finalized training counts or the outcomes above.

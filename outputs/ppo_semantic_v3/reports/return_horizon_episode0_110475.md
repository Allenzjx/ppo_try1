# Return-profile block33: fixed first episode, g109825–110475

Scope is only completed episode0 of `train/20260907T0453293778943Z_gf4bfe2560bfd_8719fa76781746e09c31b9b78dc14df8`, runtime f4bfe2560bfd. The first completed-episode record and **651 contiguous policy rows** were read, stopping atg110475 without reading the next episode. No checkpoint, optimizer-update stream, PT/hash, Python/pytest/GPU/Isaac, production/test/master edit or proposed new gate. These are executed episode decisions, **not a claim that a651-decision optimizer batch or final block checkpoint exists**.

## Actual outcome and phase coverage

Episode0 ends at **tick5208 /43.4s /P05 INCOMPLETE_CONTROLLER_BLOCKED**, stageage30s, taskfalse. The physical evaluator remains valid with termination_reason=null and empty physical failure detail: this is a real task deadline/noncompletion, not a body collision, wheel-only failure, hard joint limit or invalid observation. All651 decisions execute8 ticks, totaling5208.

Source-phase counts P01–P13 are **[1,195,4,1,450,0,0,0,0,0,0,0,0]**. P05 begins at1608/13.4s and remains the first unfinished front-leg task; no P06/rear course is reached. FR genuinely earns Q48/C1596/P1597. FL earns Q1677 but **never C or P**. Its current active qualification is later revoked before crossing; the historical event tick1677 remains an audit record, not proof that current qualification stayed true.

## FL: qualified AIR, short of crossing, then actual ground revocation

The recorded FL event sequence is:

- Initial whole-body clearance at1615/13.458333s, upward excursion3.704995mm.
- Hard qualified upward lift at **1677/13.975s**, recorded excursion43.485139mm and current above-top clearance+1.025220mm. This is genuine qualification, not just an initial hint.
- `qualification_revoked_ground_before_cross` at **5165/43.041667s**.
- A new whole-body initial-clearance event at5177/43.141667s and soft AIR evidence earned at5177. It does not re-earn hard qualification: currentFL Q/C/P are allfalse at the terminal.

These exact event ticks come from the evaluator's recorded120Hz history. The selected geometry/contact values below are **decision endpoints**, not reconstructed raw substeps or global120Hz extrema.

| Tick /time(s) | FL front(mm) | FL top gap(mm) | Current contact/load | Current hardQ | Meaning |
|---|---:|---:|---|---|---|
| 1608 /13.4 | −159.807887 | −47.088197 | AIR/0 | false | P05 entrance |
| 1672 /13.933333 | −165.787574 | −4.811409 | AIR/0 | false | Before exactQ1677 |
| 1680 /14.0 | −157.975342 | +7.107993 | AIR/0 | true | Qualified but far behind front |
| **2760 /23.0** | **−12.300905** | **+56.293731** | **AIR/0** | true | Closest sampled P05 front, also closest among qualified samples |
| 4672 /38.933333 | −67.496773 | **+.095022** | AIR/0 | true | Closest sampled absolute top gap, but still behind front |
| 5160 /43.0 | −66.776809 | −43.229873 | AIR/0 | true | Before ground revocation |
| 5168 /43.066667 | −69.082853 | −49.278224 | **GROUND/.155553022** | **false** | Endpoint corroborates recorded ground revocation5165 |
| 5176 /43.133333 | −68.604019 | −46.327266 | AIR/0 | false | New AIR, soft evidence not yet earned |
| 5184 /43.2 | −66.508171 | −36.096128 | AIR/0 | false | New soft process earned, not hardQ |
| **5208 /43.4** | **−70.140627** | **−26.042777** | **AIR/0, supportfalse** | **false** | Actual incomplete terminal |

P05's450 sampled endpoints have FL AIR444, GROUND1, obstacle-pair-active5, **trueTOP0**, support6 and currenthardQ436. The six support endpoints are not sustained platform capture. The closest sampled front remains12.300905mm behind the front plane; the near-zero gap happens much later,67.496773mm behind it. Combining these two distinct extrema into one feasible capture state would be incorrect. Exact crossing history remains absent, which is stronger than merely not seeing a crossing at the decision endpoints.

Terminal FL is outside topXY (outside-distance65.140627mm), within lateral span, obstacleinactive, load0/supportfalse, consecutiveAIR38 and consecutiveTOP0. `placed_FL=.5737539034516153` is a fractional task-progress value, **not placement or crossed history**. Total support_count3 does not make this airborne FL a support leg. Its current soft AIR credit is true but neither hardQ nor success has been restored.

## Rear hints are not a rear task success

RR has exactly three recorded `whole_body_initial_clearance` events at1606/3338/3460 (13.383333/27.816667/28.833333s), excursions3.198292/3.231999/3.060192mm. The first uses own-joint motion1.387553° together with whole-body motion10.707832° and command motion17.297366°; this is consistent with whole-body initial evidence and is not a requirement for RR alone to move2°. **No RR hardQ/C/P is earned**. RL has an initial hint at8, likewise no hardQ/C/P.

Terminal RR is GROUND/supporttrue/load.084245147, front−638.104026mm/gap−50.205770mm, initialfalse. A transient whole-body rear unloading hint while working on FL does not mean the rear phase or later success was reached.

## Actual signed reward decomposition

These are sums of the recorded **weighted environment families**, including the true terminal; not discounted returns, advantages, entropy terms or optimizer losses. P05 contains exactly30 physical seconds, the whole episode43.4s.

| Family | Whole651-decision episode | P05's450 decisions |
|---|---:|---:|
| task_progress | −42.713945458 | −43.196690831 |
| body_stability | −.347628082 | −.169066611 |
| contact_motion_quality | −.001284765 | −.000352746 |
| control_smoothness | −1.672307177 | −1.175663009 |
| control_regularization | 0 | 0 |

Whole-episode double-precision family total and recorded episode_return agree at **−44.735165481705**; summing the PPO-facing float32 reward values gives−44.735165942433, a small representation difference, not another cost. Task decomposition is exact: whole episodePBRS−1.845945457544, time−.868, terminalevent−40; P05PBRS−2.596690830910, time−.6, event−40. Smoothness is the largest non-task cost here; this is a magnitude observation, not proof of a causal conflict.

The new discount is actually used once per decision: every row satisfies `F=5*(.9985*Phi_after−Phi_before)` with maximum parsed-formula error0. Local examples retain both positive and negative signals:

- At1680, Phi.308198422→.350764883, F+.210201570 and totalreward≈+.205901459.
- At the closest-front endpoint2760, Phi.394485427→.395844128 gives F+.003824674, but time/quality costs yield totalreward≈−.000311867. Positive progress plus negative net reward is observable, **not by itself a bug or unique explanation**; globalPhi includes other body/leg features too.
- At5168, the sampled ground revocation interval has Phi.344982902→.270966466, F−.372114426 and total≈−.376787841.
- At5184, the new legitimate soft AIR process has Phi.270974733→.291517083, F+.100525372 and total≈+.096290596; hardQ remainsfalse.

At terminal5208, **event−40, Phi_after0, no bootstrap**, prePhi.2947719022294969 and potential removal−1.4738595111474844. Task family−41.475192844481 includes time−.001333333333; body−.000344883972 and smooth−.002302221346 remain separate. The PPO-facing finalreward is−41.477840423584. Physical validity and a nonzero preterminal task score do not allow the deadline to bootstrap.

As a consistency check on this exact failed episode, initialPhi=.06804292995522962. Pure PowerShell arithmetic gives discounted sum `sum(.9985^t * F_t)=−.34021464977614774`, matching `−5*Phi_initial=−.3402146497761481` to≈3.3e−16. Thus local dense progress is present, and its matching-gamma absorbing-terminal telescoping is intact; the unsigned/signed family sums above must not be mistaken for this discounted identity. No event bonus or duplicate PBRS contribution is inferred.

## Non-paired comparison and evidence limits

The finalized [C109824 diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_semantic_v3/reports/eval_109824_diagnosis.md) is a different experiment: old f1a9 runtime, fixed deterministic mean, seed2001,668dec/44.533333s/P05 incomplete, FLQ1825/C2889/noP; all450 P05 endpoints AIR, terminalfront−69.295880mm/gap+37.810539mm. Current #33 episode uses stochastic training under the new return profile and seed1001: FLQ1677/noC/noP, late realGROUND revocation, terminalAIR below top. Both lack actual FL placement. Differences in seed, sampling, training/update context and reward revision prevent a paired gamma-effect or stable-improvement claim.

All5208 recorded native ticks verify, own-request effect5204, four in-episode state-write totals0, physical-validity and global-sequence errors0. These compact receipts are not a complete raw contact/force/kinematic reconstruction. The initial read failed on Windows file sharing and produced no usable rows; the successful shared-read retry remained fixed to the same651 rows. Its empty intermediate counters are not training results.

The first unfinished task is FL forward crossing followed by capture, after a genuine but later revoked qualification—not a missing initial-lift event or a rear hard success. No subsequent episode, final planned block budget, checkpoint outcome, parameter change, detector change or extra training gate is asserted. Only this report is written; review complete and stopped.

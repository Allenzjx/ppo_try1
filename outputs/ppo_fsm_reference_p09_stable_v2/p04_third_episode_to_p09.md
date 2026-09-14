# P04 episode 2: FL placement, continuous approach to P09, BODY_COLLISION

Read-only diagnosis, 2026-09-10, production `64c03243ac05`; no production/process changes or simulator/test execution. Run: `runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0852458310353Z_g64c03243ac05_aafdb9864bdc42dda72ef793af390efd`.

Fixed policy evidence is audit **lines319–708**, bytes **23,622,512–55,893,900 inclusive**, 32,271,389 bytes; exact SHA-256 **634fe7dcafd30c7e7ec2bc4507f2744008664f35c093e745ad263f05e2f44634**. Only this episode's390 policy rows enter the analysis. Prefix result/credit-start lines644/645 confirm accepted real teacher P04 at tick1696 /14.133333 s (212 prefix decisions, zero PPO credit, not an exact historical pose or current-policy P01). Inherited FR Q23/C1665/P1695; FL has no teacher I/Q/C/P.

## Actual phases and new FL events

| Issued phase | Policy samples | Actual transition tick / time (s) |
|---|---:|---|
| P04 | 1 | →P05 1704 /14.200000 |
| P05 | 145 | →P06 2864 /23.866667 |
| P06 | 206 | →P07 4512 /37.600000 |
| P07 | 2 | →P08 4528 /37.733333 |
| P08 | 6 | →P09 4576 /38.133333 |
| P09 | 30 | BODY_COLLISION4809 /40.075000 |

Globals142143–142532,390 decisions and3113 executed physics ticks; suffix25.941667 s. The last action executes only one tick before the terminal, not eight fabricated ticks. The first three completed episodes total **708** samples (P04=3/P05=447/P06=220/P07=2/P08=6/P09=30); this is a fixed subtotal, not the evolving live-run total.

New FL I/Q attempts: **1727/1731**, **1770/1783**, **1980/1986**, **2611/2618**. The first three Qs are revoked by ground at1755,1873,2568. The fourth reaches **C2828 /23.566667 s** and **P2864 /23.866667 s**. First-ever Q1731 is historical, not the start of one uninterrupted successful attempt. Initial I1727 has3.848 mm lift gain and10.030° whole-body actual response; the final Q2618 has8.193 mm gain and3.216° own-joint response. These are measured whole-body plus joint events, not a requirement for a large isolated swing-joint movement.

At P2864 the FL observation itself is TOP/support, valid load .198576, front+31.050 mm, gap−1.429 mm. At2872 it remains supporting with load .104929. FL placement is new policy-sampled evidence, not inherited teacher success.

## What the rear transitions actually established

P06→P07 meets rear-approach geometry with previously placed FR/FL; RR is still GROUND, front−202.323 mm, lift0. Only FL/RR are supports at4512, so RR body-control evidence is false then. P06 completion is proximity, not a claim that RR is currently lifted or ready to be statically balanced.

By P07→P08 at4528, measured supports are FL/FR/RR, FL is TOP and actually bearing (load .522762), RR remains GROUND; rear proximities and RR preparation pass. The independent CoM projection toward FL over0.5 s is+63.503 mm, current+.117774 m/s. This time support and CoM movement have separate positive evidence.

P08→P09 at4576 meets proximity+transfer, **not RR Q**. CoM projection toward FL is+19.640 mm, current+.087570 m/s, with a valid RR load decrease .358180. However FL is now AIR/load0/non-supporting; current supports are FR/RL/RR. Thus movement toward the FL side is real but FL bearing is not. RR remains GROUND/lift0. On the next P09 action, the same motion history continues; no re-landing/re-lifting of an established RR attempt is involved because no such attempt exists here. Phase readiness is not an invariant claim that the same support arrangement persists later.

## RR: one earlier initial event, no Q or rear completion

The whole episode has **one** RR initial event: **I3087 /25.725 s in P06**, gain3.011946 mm, whole-body actual response8.117°, command motion13.391°. At3088 AIR lift is3.399477 mm, own response3.250°, other supports FR/RL while FL is AIR. At3096 the initial-now threshold is no longer met (lift.975494 mm); by3104 it is GROUND, attemptfalse, lift0. The exact ground-reset tick is not separately logged. **RR never reaches Q in this episode**, so there is no Q-revocation event to count. No RR C/P or current-valid decision end occurs anywhere in the390 samples.

Within P09 specifically: **28 GROUND ends,2 AIR ends;0 initial-now,0 Q/C/P**. At4808 lift is.839523 mm; at4809 it is1.617864 mm, front−170.489 mm and gap−48.380 mm. Both are small AIR observations, not qualified initial lift. Current I/established/Q-valid are false; body-control evidence and motion allowance become false on the collision tick. Duration .016667 s is diagnostic, not a hover test. P09 entry is valid, but its first uncompleted physical work is establishing a usable RR lift/carry attempt, then crossing and placement—not “holding a qualified lift longer”.

## Continuous downlink and terminal evidence

All3113 physics ticks have verified native and own-phase action effect; all390 decision-end masks are12/12, setter/mapping equality and previous-ack verification pass, and no prohibited state writes are reported. Ordinary handoffs use the carry hold and preserve residuals (servo step ≤7.11e−15°, wheel residual step ≤2.23e−16); no forbidden-channel drop or phase-scale clipping occurs. Nominal request changes are −3.2° atP04→05 and maximum12.7°/8.5°/4.2° atP06→07→08→09; these are owner requests, not reset evidence. P08→09 wheel request changes only .000243187 rad/s; P05→06 has no wheel jump in this episode. Do not generalize another episode's +.3 jump to this one.

At4809 the evaluator reports **TASK_FAILURE_BODY_COLLISION**, reason “central body/obstacle collision”, source **BODY_CONTACT**, with VALID/VERIFIED physical evidence; the env verdict is BODY_COLLISION. Production derives this failure from authoritative `observation.body_collision.detected` (`semantic_supervisor.py:471–474`), not elapsed time, nominal motion, phase label, or video packaging. The training record does not retain the raw contact-pair manifold, signed BODY-to-obstacle distance, or collision force; those values remain **unknown**, not inferred from body bounds. BODY bounds are not fully in the platform region (outside distance .180353 m), so no completed traversal is claimed.

Terminal supports are **FR TOP/load .441722** and **RL GROUND/load .558278**; FL AIR/load0 despite historicalP, RR AIR/load0. Body speed .124667 m/s and angular speed .749139 rad/s; measured short-window body displacement [−27.909,+5.201,−24.013] mm shows retreat/descent, not net forward carry. CoM projection toward FL is−1.511 mm over the window while instantaneous projected velocity is+.068609 m/s; neither fact creates support under airborne FL.

Final wheel nominal [.3,.3,.3,.3] plus residual [−1.144709,.133521,.179161,.471327] yields target [−.844709,.433521,.479161,.771327] rad/s, with actual [−.844621,.422509,.498289,.771618]. Real signed response exists. This bounded audit identifies no hidden downlink/reset defect, but does not assign the collision to one command, one leg, or tracking error alone.

Updates1076/1077/1078 at globals142208/142336/142464 each record20 optimizer steps, actor change and finite nonzero gradients; the episode spans these updates. Its FL placement and deeper stage progression are observations, not evidence of causal learning improvement or stability superiority. No rear policy success or natural-P01 full success is established; the original failed result is retained. Current training may continue beyond this fixed report boundary.

# CP209920 deterministic P01→P02 front loss — read-only

## Scope and outcome

Only the sealed source run `20260922T1004553890489Z_g5fd88852bf20_a9c7b7134b6c4b448234ac34c0a3dc08/source` was read: 233 decisions, 1,864 real physics ticks, 15.533333 s, no P03 or P05. Full task **INCOMPLETE**, not a successful front-leg/P05 demonstration.

Terminal: `INCOMPLETE_CONTROLLER_BLOCKED` / `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`. P02 entry remained valid with no entry reasons. Stage age 15.400000 s exceeded its finite local limit 15 + current-progress allowance 0.396598 = 15.396598 s. End completion `lifted_FR=0, clear_FR=0, approach_FR=0.689868`. Physical evaluator stayed VALID/VERIFIED, its failure reason stayed null; all raw observations finite and no in-episode state writes. Therefore this is task noncompletion/control behavior, **not a recorded body-collision, wheel-wall-ascent, fall, hard-limit, or numerical safety abort**. That does not establish stability or success.

## First obstruction and source chronology

| Episode tick / time (s) | Recorded fact | Interpretation |
| --- | --- | --- |
| 13 / 0.108333 | FR initial measured whole-body clearance | Real initial lift, not successful P02 clearance |
| 17 / 0.141667 | P02 source starts; P01 layer continues | No fixed-entry or predecessor deadlock |
| 22 / 0.183333 | FR qualified lift; gap −42.638 mm | Qualification is upward-lift evidence, not “above obstacle” |
| 41 / 0.341667 | All four source wheel suggestions become +0.3 rad/s | Inherited P01 rolling is present in P02 |
| 73 / 0.608333 | Source FR knee reaches its 45.9° held endpoint | Source knee command was delivered, not waiting on a mask |
| 93 / 0.775000 | **Maximum all-tick FR gap +13.106667 mm** | Never reaches the current +15 mm P02 clearance goal |
| 196 / 1.633333 | FR ground contact 17.331825 N; gap −50.289813 mm; front −108.144071 mm; qualification revoked before crossing | First true qualification loss; direct ground-pair evidence, not label inference |
| 1593 / 13.275000 | All four nominal wheels become zero after last +0.3 at tick1592 /13.266667 | Finite source stop occurs **11.641667 s after** recontact; cannot explain initial loss |
| 1864 / 15.533333 | FR gap −50.089900 mm, front −87.532994 mm, ground load1.947482 N | No re-earned lift/cross/place; local task recovery exhausted |

P02 uses the source-derived FR-knee motion while retaining P01 RL hip 37.6°. The recorded source-partial-order diagnostic has no P07–P09 layers, successful-FSM source entry/completion gates are disabled, and capture/final-stop owners were never acquired. This is not a P05 pending-capture or old placed-FL barrier.

After the finite wheel stop, the existing P02 rolling-extension predicate is correctly false: FR qualification was revoked and current gap is below +15 mm (it additionally checks other measured supports and approach position). This physical-goal-dependent extension did not suppress the original rolling pulse. The source ends/holds while a genuine FR lift/clearance task remains unsatisfied; no source readiness bypass is justified by these records.

## Same-tick four-wheel path

Order **FL / FR / RL / RR**; all entries rad/s. “Effective policy” is the logged same-tick projected post-history/filter/rate-limit residual, **not** a separately recomputed nominal difference. In these samples mapper wheel baseline equals N, assist correction is zero, and final=baseline+effective. Actual is post-step recorded wheel-joint angular velocity, not wheel-center translation.

| Tick / time | N = wheel mapper baseline | Effective policy residual | Final command | Actual angular velocity |
| --- | --- | --- | --- | --- |
| 96 / 0.8000 | [0.300, 0.300, 0.300, 0.300] | [-0.20260, -0.00047, 0.00266, -0.07416] | [0.09740, 0.29953, 0.30266, 0.22584] | [0.04088, 0.29939, 0.30177, 0.40014] |
| 196 / 1.6333 | [0.300, 0.300, 0.300, 0.300] | [-0.28682, -0.00262, 0.00431, -0.09752] | [0.01318, 0.29738, 0.30431, 0.20248] | [-0.11064, 3.21749, 0.29541, 0.37224] |
| 1592 / 13.2667 | [0.300, 0.300, 0.300, 0.300] | [-0.32626, -0.00482, 0.01209, -0.10992] | [-0.02626, 0.29518, 0.31209, 0.19008] | [0.12120, 0.30735, 0.31255, 0.11433] |
| 1593 / 13.2750 | [0.000, 0.000, 0.000, 0.000] | [-0.32621, -0.00483, 0.01213, -0.10992] | [-0.32621, -0.00483, 0.01213, -0.10992] | [-0.18986, 0.03001, 0.01324, -0.19403] |
| 1864 / 15.5333 | [0.000, 0.000, 0.000, 0.000] | [-0.32911, -0.00482, 0.01322, -0.11086] | [-0.32911, -0.00482, 0.01322, -0.11086] | [-0.11604, -0.00966, 0.01386, -0.13534] |

At the first FR recontact, all four final wheel commands are nonzero; FR and RL still receive approximately +0.30, RR +0.20. **FL** has +0.30 nominal almost cancelled by −0.28682 policy to +0.01318; later policy actually reverses FL final to negative. Thus it is not “only RR has wheel command”. Actual motion is also not RR-only: FR's 3.21749 rad/s is a transient at ground impact; later FR/RL continue about +0.30. FL actual sign/magnitude differs from its tiny forward target under contact; this is an observed tracking/load coupling, not permission loss. After the source stop, FR/RL residuals are small because their learned residual is small, not because a residual mask removed nominal.

All 1,864 native audits verify both dispatch target equality and actual mapping, with last source `robot._joint_pos_target_sim/robot._joint_vel_target_sim_after_existing_write_data_to_sim`. All12 permissions open, no assist owners, no reserved-headroom servo clips. These dispatch and measured-response checks—not the all-one mask alone—rule out a demonstrated dropped-channel/index/second-write fault in this run. Native actuator axes are signed: at tick196 native wheel targets are [-0.01318, 0.29738, -0.30431, 0.20248] (left wheels negative), matching the canonical final via the recorded mapping.

## FR knee and inherited RL hip

Joint cells: **N / actual same-tick mapper / effective policy / final / post-step actual**, degrees. The native mapper may differ from N due to established tracking/HISTORY; its contribution is not attributed to PPO.

| Tick | FR knee | RL hip | FR / RL gap (mm) |
| --- | --- | --- | --- |
| 96 | 45.900 / 47.150 / -3.228 / 43.922 / 43.832 | 37.600 / 38.850 / -2.979 / 35.871 / 35.759 | 12.960 / 13.021 |
| 128 | 45.900 / 44.822 / -3.906 / 40.917 / 41.827 | 37.600 / 38.850 / -3.374 / 35.476 / 35.339 | 9.123 / 11.491 |
| 196 | 45.900 / 46.546 / -4.830 / 41.716 / 41.187 | 37.600 / 38.850 / -3.812 / 35.038 / 34.828 | -50.290 / 62.060 |
| 1864 | 45.900 / 45.332 / -5.783 / 39.549 / 40.095 | 37.600 / 38.850 / -4.233 / 34.617 / 34.385 | -50.090 / 71.555 |

FR knee receives a progressively negative policy correction (−3.228° at tick96, −4.830° at recontact, −5.783° at end); its actual angle follows the final near40–44°, rather than the unchanged source45.9°. Simultaneously FL knee policy becomes −12.118° at96, −16.343° at196 and −18.316° at end; full-body support geometry changes while the FR wheel falls. These are concrete policy/controller/geometry contributions, not a proof that one channel alone caused the fall.

RL has distinct `rear_left_hip`/`rear_left_wheel` sensing and inherited nominal37.6°, not FR/RL field swapping. At tick196 RL hip actually34.828° and wheel gap+62.060 mm with0 N; at end RL gap+71.555 mm and0 N. FR is actually grounded at the same time. Therefore “FR stays down while RL is up” is a real emergent state of this controller/policy, not a mistaken leg label or evidence that only the RR wheel spins.

## Conclusion and limits

This deterministic evaluation repeats the **early loss-of-FR-lift class** seen in the bounded block05 episode2 snapshot, not block05 episodes0/1's retained-clearance/insufficient-forward-approach class. The first failure precedes source stop; source rolling survives the handoff and native writes match. Learned negative FR-knee/FL-wheel corrections and whole-body support changes are present and materially alter the source behavior. A single observational trajectory cannot allocate causal percentages between policy, mapper tracking and mechanics, or prove the zero counterfactual would succeed.

No source/config/reward/checkpoint changes, no new simulation, no retrospective Recording scan. Only the sealed logs and current relevant predicates were read. The stdlib CPU extractor exited0; no helper remains running.

Evidence: `video_policy_decisions.jsonl`, `native_tick_audit.jsonl`, `physical_observations.jsonl`, `stage_transition_evidence.jsonl`; same-run output helper `audit_CP209920_DET_cpu.py`. Relevant production logic: `semantic_supervisor.py:658` ground-before-cross revocation; `:2482` P02 approach extension; `:2650` source start/tick and `:2666` measured extension.


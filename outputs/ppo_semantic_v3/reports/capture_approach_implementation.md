# Post-cross first-capture progress revision

Historical prelaunch time slice: Implementation in progress after the completed73e9370 natural-P014096 block and C72320 evaluation. The actual latest saved model is72320/530/10600; no newer training count is claimed here. The finalized diagnostic/evaluation status is recorded in the final section below.

## Evidence and scope

C68224 and C72320 both genuinely qualified and crossed FL, but their fixed means did not obtain actual FL platform contact. C72320 ended P05 incomplete at647 decisions/5176ticks/43.133333s, physical evaluator valid with no physical failure, optimizer0. Its closest recorded FL gap was+2.87186mm with zero obstacle force/contact; it never reached P06. Conversely, all five stochastic P01 training episodes in the completed4096 block had real front-leg Q/C/P. This is evidence of a repeatable incomplete fixed-mean capture and real sampled capture capability, not proof of a particular joint-direction intervention or a sensor bug.

Local formula inspection also confirmed that post-qualified/post-cross, pre-placement AIR has no positive-clearance first-capture gradient: the original capture term is only consecutive TOP count. The separate `placed_FL=.85` predicate is not itself the v3 reward potential. Existing PBRS and the final incomplete penalty do penalize waiting; the issue is a flat local approach direction, not positive hover reward farming. See `p05_capture_reward_readonly.md` and the two completed evaluation diagnoses.

## Production change

`semantic_supervisor.py` adds explicit `CAPTURE_APPROACH_MODE=post_cross_current_surface_proximity_plus_real_contact_v1`. Its loader requires the existing global physical potential, current-platform geometry/retention mode, and a positive existing top-gap scale. The v3 task YAML opts in under a new revision. No mode or None retains the old formula exactly.

For an eligible **unplaced** leg after both real hard qualification and front crossing:

`xy = clip(1 - current_top_xy_outside_distance / .25, 0, 1)`

`proximity = xy * T / (T + abs(current_clearance))`, where `T=existing top_gap_max_m=.025m`.

`capture = .5*proximity + .5*clip(consecutive_real_TOP_samples / existing_minimum_TOP_samples)`.

This replaces, rather than adds to, the existing `.2*capture` share. It reuses the existing `.25m` carry/retention decay and measured geometry; no joint sign, pose, phase clock or new spatial threshold is a dispatch gate. AIR alone can earn at most half this capture share. Before crossing/qualification, it supplies no downward approach credit. Placed legs bypass this helper and keep their existing current-region retention, including legitimate AIR above the platform.

The same opt-in branch retires the preceding preparation share by assigning the existing local `.1*unload` contribution its completed value after hard Q+C. This is **task-progress bookkeeping**, not a claim that the measured `load_ready` predicate previously reached1 or that an AIR leg supports weight. The current load/support predicates, goal features, observations and histories remain measured and unchanged. Before Q+C and before predecessor eligibility, the old preparation calculation is unchanged.

That small retirement is needed to avoid creating a touchdown conflict while splitting capture credit. At fixed near-zero gap/XY, hover can have leg progress.9. On the first real TOP sample, a physically possible load fraction.8 with two other supports gives current `load_ready=(1-.8)/.8=.25`; without the retirement, the new split would decrease the leg progress to.875 despite legal loading. With it, the same sequence is.9→.95→1 at real placement in the retained region. Current `load_ready` still remains.25 in that example. No support, touchdown, Q/C/P, stage transition or success is fabricated by the local progress value.

All weights outside this first-capture/preparation transition, reward families, terminal±40, time cost, gamma/lambda, PBRS absorbing terminal, ordinary-phase continuity, action ranges/slew, nominal scheduling including the completed P06 tail revision, geometry actuation, physical limits/sensors and A/FSM remain unchanged. The separate P13 continuous-stop mode is unchanged. Changes in the existing potential scalar and its downstream observation history are an explicit new MDP despite the same324-dimensional schema and12-dimensional policy output.

`semantic_cli.py` additionally corrects a descriptive initial-metadata bug: both new-MDP and distribution-migration initial records now set `implemented_reset_sampling` from the current env configuration, matching the already-correct sampling/topology fields. This is two construction-time assignments, not a new loader restriction, reset path or change to old embedded checkpoints/sidecars.

## Tests and migration boundary

The new dedicated capture tests have actually passed53 cases in4.37s, including real evaluator Q→C before geometry sweeps, four-leg symmetry, signed gap/XY, unresolved AIR contact, first/second TOP, current-load/predicate preservation, predecessor and qualification guards, placed AIR retention, old-mode compatibility, identical hard events/transitions,324-feature semantics and PBRS hover/cycle/terminal counterexamples (`C:/robotics_sim/wlr_robot/semantic_capture_approach_20260906.xml`). The broader semantic regression, explicitly excluding that new file, then passed1004 tests in96.55s (`C:/robotics_sim/wlr_robot/semantic_capture_approach_regression_cpu.xml`). These disjoint sets total1057 passed, with no skips or failures. The focused43 CLI run overlaps the1004 and is not added again; earlier progress-focused runs also overlap. These are CPU contract/serialization/formula tests, not new physical task success.

Legacy retention fixtures explicitly disable the newer approach flag when constructing the older retention-only configuration. One stop-validator negative fixture also removes newer capture dependencies to keep testing its intended stop error. No original physical assertion or hard-success threshold is relaxed.

After tests and a scoped runtime commit, continue from actual72320 with compatible actor including learned std, critic, identity normalizer, training RNG and lifetime/spent budgets preserved. The existing NewMdpWarmStart resets Adam moments at initial3e-5, uses an empty rollout and a fresh physical reset; no old rewards, partial buffer or simulated state are reused.

The selected next bounded course is **P06 teacher offset240 / N1 / seed1001 /2048 requested new policy decisions**, followed by another saved-model natural-P01 fixed evaluation. This deliberately increases rear-execution exposure after the completed P01 block's P06-heavy samples (2422 P06 but only10 P09). Existing real teacher evidence has a P06 interval3584–5952; offset240 is at5504 and leaves56 historical decisions before P07, with no recorded RR qualification yet. The next run must report its own current handoff contact/gap/load and actual phase; old compact records do not prove current GROUND. Prefix actions remain excluded from PPO, and an unexpected legitimate fallback is recorded rather than hidden. A suffix outcome cannot be called full PPO success. Future blocks continue alternating full-task and rear/predecessor coverage; none is credited in this note.

Risks: proximity can guide a new policy but does not prove a safe Cartesian action or convergence. Quiet geometry near the surface is not real contact, loaded contact can still cause physical impact/collision, and post-placement retention/ultimate stopping remain separate tasks. No new success checkpoint, video or stability superiority has been produced by this implementation note.

## Actual runtime freeze and launch

Independent final production review found no blocking issue. All seven scoped files (three production/config plus four tests) were committed as `c34262abffc16847ff32d15ecbf790dd60803e0a`. Root verified no Python/Isaac process before starting the real2048-decision P06/offset240 block at `runs/ppo_semantic_v3/train/20260906T1738509257629Z_gc34262abffc1_04bbf6558046407990f85358d738cc17`, with the planned N1/seed1001/NewMdpWarmStart/source72320 settings and checkpoint interval4. Only preserved output reports remain untracked. This append records the actual launch, not completion or even a completed new optimizer update; prefix/handoff and first saved-update evidence are still pending.

## Actual finalized diagnostic and reloaded P01 evaluation

This final append supersedes the earlier implementation/launch-pending time slices without deleting them. Committedc34262a run `20260906T1738509257629Z_gc34262abffc1_04bbf6558046407990f85358d738cc17` honored a normal complete-update stop: **768 actual/6 PPO/120 optimizer steps** of2,048 planned,**1,280 unconsumed**,wall831.2335137999617s. Source72,320/530/10,600 becomes **73,088/536/10,720**; spentfull29,184/suffix33,792,origin10,112 preserved. Final checkpointSHA3efd9531d6bf797a2852107c1183ac8be2b96bcf5eaa3ed59358f0cd9d44ffef,sidecarSHA43156d6dc98d44de7ea9d0b127ea9cf06535961318d850c72eeedd7ead25aa14 androundtriptrue were root-verified; this append does not repeat hashing.

Actual initial evidence is separately fixed in `capture_approach_p06_offset240_initial.md`: actor/learnedstd,critic,identity normalizer,RNG,runner/policy and spentcounters preserved; freshAdam3e-5/empty inherited rollout,not exactAdam resume. Initial artifact SHA9061fcd2…3221 belongs to the new initial checkpoint,not the original72,320 model. Correctedinitial sampling strings nameP06offset240; the disclosed unrelated top-levelsuffixfalse residue is not rewritten.

Final `capture_approach_p06_offset240_block_final.md` verifiesP06=768/allotherphases0,policy6,130native+owneffectticks/no4statewrites,and3accepted prefixes2,064decisions/16,512ticks excluded from credit. Two361-decision P06deadline episodes plus46optimizednonterminaltail are not threefailures or originalbudgetcompletion. Neither rearleg earned hardQ/C/P; all frontQ/C/P were teacher preparation. **No new first-capture branch,finite-source endpoint or tail extension was actually visited during these optimized samples.** Firstunfinished task was currentRLworkspace,not capture.

Reloaded73,088 naturalP01/N1/seed2001fixedmean evaluation `20260906T1758410256842Z_gc34262abffc1_1c3cc10c00874db8a1b189eb741a4832` completedexecutionSUCCEEDED buttaskfalse: **649decisions/5,192ticks/43.2666666667s/P05INCOMPLETE**,physicalvalidtrue/nullfailure,optimizer0. FRQ48/C1,570/P1,584; FLQ1,669/C2,708 butnoP,uninterruptedAIR1,605–5,192,0P05obstacleforce/contact. ClosestFLgap+2.763440mm/front+3.200832mm stillAIR. `eval_73088_diagnosis.md` gives fullraw contact/native evidence.

The new capture formula **did** appear in this evaluation:1FR/311FL decision-end eligible snapshots,the latterincluding1terminaldiagnostic. FLcapture.117010143–.450217285 withrealcontactfraction0; hardcapture remained unmet. Terminalreward usesabsorbingpotential0,not diagnosticphi. All5,192native verify/statewrites0; P06–P13 neverentered. This distinguishes actual branch exposure from optimized branch training or demonstrated physical improvement. C72,320 and all older incomplete/failed evidence remain unchanged. No full/suffixsuccess or improvedvideo is produced; no futurecoursebudget is credited.

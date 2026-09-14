# Residual PPO recovery and current evidence

## Current fixed state — video6 complete; block14 stop requested

Training is fixed to the [completed block13 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T152830791441Z.json), observed 2026-09-10 15:28:30.791441Z; checkpoint byte verification is parent 15:28:32.2542108Z. Video6 completed at15:52:36.790432Z; the separate block14 active snapshot is parent16:08:30Z. These are dated receipts, not a continuously polled pointer. Block14 has210sampled/one completed PPO update1153, but its newer checkpoint has not yet been independently verified byparent and adds no credit to the completed-block ledger here.

Latest independently verified saved model: **151936 policy decisions / 1152 PPO updates / 23040 optimizer steps**, [checkpoint](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000151936.pt) and [official immutable manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000151936_manifest.json). Parent independently checked both byte hashes and the official save/load roundtrip; this reporter did not read a live pointer, sidecar, tensor, active stream or repeat a model/media hash.

- Checkpoint SHA256: `ba52ffcd874c227c41632d8a9b3b58a4980b715ec6181eba0113d84e837f61c9`
- Manifest SHA256: `58c34b5f2ee649d708ae6629b5e4c6292490dbdf2a86804f83d8e8fc1772302d`
- Actor after actual final update1152: `c878c6b1fe68297997601df5f7d7d38ff4a65a2d42276bf9873b36449c7b4236`. This is the update receipt, not an invented manifest actor key.

Production remains HEAD `7db0d17f398d393ce026b6990bd2566d53366407`, runtime `5eb1536f17f3037639015e714f17ae8e805489c35ca3cf5c4318546ef3a4ac93`, same372-dimensional policy contract and existing configuration. Block13 resumed149888 without migration. Compatible weights, optimizer, identity normalizer and counters were retained; no network restart or reward/control change occurred in this report update.

The final optimizer LR is **1e-5**, not a permanently fixed learning rate. The existing adaptive scheduler logged update1147 ending5.0625e-5 and update1149 ending2.25e-5; many other logged values are1e-5. These are selected observations, not a complete per-update trace, and no manual configuration change was made.

Across **13 completed blocks**, the new-version ledger is **11392 decisions / 89 PPO updates / 1780 optimizer steps**, all optimized and saved, unfinished rollout0. The earlier block13 first128/150016 receipt is now included exactly once in the full2048 block; it is not extra credit. Latest completed actor full-P01 evaluation is now **video6 /151936**, P06 incomplete: trueFL placement was earned, then currentTOP was lost and RR never earned I/Q. This is not full traversal success; final run validity isVALID but contact-bearing evidence isunverified.

### Completed block13: actual natural-P01 continuation

Run `20260910T1451368221058Z_g7db0d17f398d_19bb6607a9384f08b00d341ba21060b7` completed normally, exit0 at15:28:03.3359421Z. It requested full_episode/P01/2048, seed1001, N1, cuda:0, checkpoint cadence1, source149888, no teacher prefix or migration. Parent had verified its actual natural-P01 reset and checkpoint-load boundary at14:52:02.632145Z, P01/tick0/prefixnull. This was real current-policy simulation and PPO updating.

Actual command (historical execution, not a request to launch alongside video6):

```powershell
.\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 7db0d17f398d393ce026b6990bd2566d53366407 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage full_episode -FromPhase P01 -Decisions 2048 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000149888.pt
```

**2048 decisions /16 PPO updates /320 optimizer steps** were saved. Source phases are P01=16/P02=835/P03=35/P04=25/P05=746/P06=266/P07=6/P08=3/P09=116/P10–P13=0. All16365 policy-native ticks have verified target effects,16323 own-phase plus42 inherited handoff ticks; forbidden state writes0.

Seven true terminals remain one HARD_JOINT_LIMIT and six FALLs. The final191-sample collection tail is nonterminal, already learned/saved, not an eighth failed or successful episode. RR new I6/Q4, ground revocations3, current-valid decision endpoints95, crossing0/placement0. Initial lift or historicalQ is not crossing/placement.

- [Episode0 RL-knee hard limit](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/natural_p01_block13_episode0_rl_knee_limit.md): measured knee−60.061543° crossed the hard bound while target−6.758895° remained in range. This is not a target-rounding explanation or a uniquely diagnosed policy/actuator cause. RR I3367/Q3371 earned inP06 carried through P07→P08→P09; FL actually changed fromTOP toAIR during handoff and cannot be counted as bearing while airborne. RR Q was physically revoked on ground at3441; noC/P.
- [Episode1 P09 FALL](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/natural_p01_block13_episode1_p09_fall.md): measured body height14.778882mm supports the safety abort, not numerical explosion. RR Q2040 earned inP06 remained current throughP09 until abort; RR still behind the edge, noC/P. Actual residual and wheel continuity is retained.
- [Episodes2–7 and final tail](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/natural_p01_block13_episodes2_to7.md): ep2–6 have measured low-height FALLs; ep2's P06 Q2728 continuously entersP09 and is revoked at3106. Ep6 fails inP06, notP07. Ep7's191 samples endP05/tick1528/12.733333s, terminal=false/bootstrap=true. FL has historicalQ1341/C1458/Pfalse; contact-surface labelTOP and12.620559N do not establish validplacement: front−14.666mm, within-top-XY=false and actual top-contact predicate=false. FR isAIR; no newRR I/Q/C/P. Do not turn a contact label into a successful placement.

These bounded diagnoses establish real physics and continuous handoff evidence; they do not establish a new dispatch/reset defect, a new reward revision, successful rear placement or stable full traversal.

### Video6 completed: true FL placement, then P06 deadline

Run `video_eval/validation/20260910T1528556872035Z_g7db0d17f398d_a5db22d8e4b545969459f6ea059a6c87` finalized15:52:36.790432Z as `DIAGNOSTIC_FAILURE`. Parent read its [final source manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260910T1528556872035Z_g7db0d17f398d_a5db22d8e4b545969459f6ea059a6c87/source/semantic_video_source_manifest.json), confirming official151936 load, matching initial CP/sidecar/actor hashes, no migration,372-dimensional conditional-mean policy, seed4001, naturalP01/tick0/decision0/done=false/reset1/options{}/prime0/pre-action0, no teacher and **0 optimizer updates /0 training credit**.

It ran **1176 decisions /9408 ticks /78.4s**, phase counts2/170/7/39/358/600 forP01–P06; P07–P13 were not reached. P06 entered at4608/38.4s and ended at9408/78.4s: stage_age40.00000000000001, effective limit40+0+0=40s, rear_approach0, `INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_TASK_DEADLINE`. This is finite task termination, time_outs=false/bootstrap=false, not global200s expiry or external truncation. No BODY, wheel-only or independent safety terminal was recorded.

The [bounded diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video6_151936_p06_diagnosis.md) and [fixed scalar JSON](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video6_151936_p06_diagnosis.json) distinguish:

- FL first I1805/Q1809 was revoked on ground1843; new I3890/Q3937 preceded C4486 and **actual P4604**. P05→P06 endpoint4608 was realTOP/support, verified5.405339N/load0.188347. The exact P4604 event geometry and4608 endpoint force are different instants; exact force/validity at4604 is not separately stored.
- FL first sampled lostTOP at4880, but still had verified11.579612N then; it was not yet wholly unsupported. At5024 support=false/load evidence unverified; by5168 it wasGROUND and never regainedTOP. P06 endpoints are33TOP/36neitherTOPnorGROUND/531GROUND.
- TerminalFL isGROUND, notTOP, surfaceOBSTACLE_AMBIGUOUS, front−47.565785mm/gap−50.430918mm. Its numeric12.369541N and0.377640 fraction are **not verified load-distribution evidence**: bearing_verified=false/load_fraction_valid=false. HistoricalC/P remains true but does not imply currenttop support.
- RR earned **no I/Q/C/P**. All600P06 endpoints areGROUND/current_lift_valid=false; briefP05 AIR was not effective lift. Terminalfront−467.174691mm, gap−50.049706mm. First unfinished task is P06 rear approach/preparation, with FL support having degraded.

Final physical run validity is **VALID**, physical evidence status **CONTACT_BEARING_UNVERIFIED**, success=false. All9408 native effects areverified/continuous,9403own and no episode statewrites; execution verification doesnot make contactbearing verified. All12 endpoint masks wereopen. P06 matchedtanh(raw)×cap598/600 endpoints; two had downstream projection/rate-chain effects, not proven defects or proof of unrestricted projection.

The4608 endpoint nominal wheels changed from+.3 at4600 to0 then back+.3 at4616; actual wheels and residual were not allzero. Parent separately confirmed the small P05 capture-wheel overlay shutdown gap and requested a graceful training stop before its fix. **This does not prove that the gap caused FL retreat or the failed task**, and this reporter made no control change.

Failure acceptance occurs before `check_model()`: **the end-of-evaluation unchanged-model hash check was not executed**. Initial official load hashes and0updates are not an executed final comparison. Partial roll/pitch RMS0.081844/0.112217rad describe only the78.4s incomplete trajectory and do not demonstrate superiority toFSM.

The separate [playable diagnostic MP4](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video6_151936_p01_incomplete.mp4) has1176unique1280×720/15fps frames, black0, full decodePASS, valid78.4s container/media duration,110380439B. Its [packet-copy receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video6_151936_p01_incomplete.remux_receipt.json) verifies exact decoded checksums, absolute/deltaPTS and key flags; SHA256 `4b13e0b62fdba5a6f95829345dde39fd46aa4a3556ed089f4a21a4c04298c70e`. Original110515863B video and source manifest remain unchanged despite the original286331153s invalid container duration. No re-encode/trim/stitch/speed change/interpolation or success publication. Parent viewed first/last PNGs and displayed the playableMP4; reporter read only completed receipts, not media/raw or repeat hashes.

### Historical video5 and separate N0 diagnostic

**Video5 /149888 is a preserved historical actor full-start evaluation**: finalized14:26:43.677313Z, official initial load verified/no migration, naturalP01,800decisions/6396ticks/53.3s,0updates. It ended P05 `INCOMPLETE_CONTROLLER_BLOCKED`; first unfinished task is true controlledFL placement. FL I1989/Q1992/C2587/Pfalse,477 post-crossing endpointsAIR/load0. Its effectiveP05 limit was37.225s=30+7.225, actual37.233333s, not the older34.9s. Physical evidence VALID/VERIFIED without physical hard terminal. No terminal model-hash check or successful-video publication was performed. Original800-frame recording's invalid container duration was not repaired here. See [video5 bounded diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video5_149888_p05_diagnosis.md). Historicalvideo4/148352 and its lossless diagnostic remux remain unchanged; neither is a success or paired stability comparison.

**N0 is a separate completed no-actor diagnostic, not A or a PPO checkpoint evaluation.** Its [fixed final snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T145119085643Z.json) contains1108 decisions/8861ticks/73.841667s, P09 incomplete, window_ended=false, VALID/VERIFIED and no safety/BODY terminal. FL earned I1550/Q1559/C2468/P2677; RR I5205/5487,Q5509, real ground revocation6243, C/Pfalse. P09 local recovery was exhausted: age30.7083333s>effective30.7052647347s=30+0.7052647347+fixedpost0. Terminal bootstrap=false/time_outs=false; final5ticks verified. This is neither global200s expiry nor a60s cutoff.

The [N0 continuity report](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/n0_first864_p09_continuity_snapshot.md) retains its earlier nonterminal scope, and [progress-aligned N0/video5 comparison](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/n0_vs_video5_p05_progress_comparison.md) shows that trueFL placement is reachable under currentN without requiring PPO to copy one pose. N0's complete P09 nominal sequences were actually issued; no new execution defect or optimizer gate was established. N0 adds no actor updates or training credit.

### Completed ledger and evidence publication

Completed source-phase counts: **P01=70/P02=3708/P03=139/P04=122/P05=4622/P06=1611/P07=35/P08=25/P09=420/P10=3/P11=3/P12=579/P13=55**, sum11392. Policy-native physics90967. **47 true terminals**: FALL31/BODY_COLLISION11/HARD_JOINT_LIMIT4/INCOMPLETE_CONTROLLER_BLOCKED1. Twelve nonterminal collection tails are not extra terminals. RR current-valid endpoints452 and cumulative newly earnedC1/P0; endpoint counts are not counts of new lift events and can include physically inherited prefix state.

The new [block13 CSV](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_block13_000151936.csv) is2048×263, completed-source149593016B, all credited/optimized/saved=true, teacher0, missing raw-body values left blank in2048 rows. Parent completed engine/recalculation/all-scalar verification, independent PowerShell readback and PNG QA; see [receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_block13_000151936_authoring_qa/receipt.json) and [visual review](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_block13_000151936_authoring_qa/visual_review.md). This reporter read those QA receipts, not CSV/raw.

Disjoint publication is **13752 rows =11392 saved training +1252 earlier nonvideo evaluation +1108 N0 diagnostic**. N0 rows are zero_residual_eval=true with credited/optimized/saved=false and exact physical joins1108/1108, no nearest substitution. Video3/4/5/6 and overlapping older CSV1/2 are not added. PriorCSV and QA artifacts are unchanged; former block13 first128 is counted once, not added again.

### Block14 active snapshot and requested graceful stop

Parent16:08:30Z reported a real P06-origin prefix training opportunity, request1024 from151936: **210 sampled decisions and one completed PPO update1153**. A newer checkpoint has not yet been independently verified byparent at this reporting boundary. Do not add these samples/update to the13completed-block saved ledger, infer its pending count or present a guessed new model hash. This reporter was not supplied a runID/load-identity receipt for this small update and did not read active files.

Parent requested a graceful stop at a complete-update boundary so the confirmed smallP05 capture-wheel overlay shutdown gap can be fixed. Stop completion and final checkpoint verification are **not yet claimed** here. Do not interrupt the process or start another Isaac job from this report. The gap is a bounded implementation issue, not proof of the failed trajectory's unique cause, and not a new success gate.

### Recovery after the active process and necessary fix

After the current process has actually exited, resolve the actual newest valid official checkpoint, currentHEAD and the fix/migration state. The latest independently verified model in this report is151936; a later valid saved checkpoint must take precedence. Do not resume an old151936 command blindly or reuse the old145920 exact migration. If runtime code changes, use the applicable precise compatibility/migration plan rather than assuming same-HEAD resume.

Preserve compatible weights/Adam/identitynormalizer/RNG/counters and legally reset physical state; discard an incomplete rollout only if recovery requires it. This reporter neither applied the fix nor changed production, checkpoints or process state. Continue based on actual evidence/resources, without A5/5, N0 success, fixed-pose or probe-success optimizer gates.

### Preserved reports and limits

Before this update the four prior files were archived byte-for-byte and archive/source SHA matched: [training report](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/report_history/training_and_evaluation_before_completed_video6_20260910T155236790432Z.json), [runtime identity](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/report_history/runtime_identity_before_completed_video6_20260910T155236790432Z.json), [publication index](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/report_history/checkpoint_publication_index_before_completed_video6_20260910T155236790432Z.json), [RECOVERY](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/report_history/RECOVERY_before_completed_video6_20260910T155236790432Z.md). The previous currenttop is preserved in those archives, not appended repeatedly; every existing historical section below is retained unchanged.

The index remains `READ_ONLY_PUBLICATION_INDEX_NOT_OFFICIAL_RESUME_METADATA`. Only the four public report files were changed. Official pointer/sidecar/resume state, immutable checkpoints, production, reward, CSV/media and active processes were not modified. Full-P01 successful PPO checkpoint, successful ≤200s PPO video, matched FSM video, comparison video and demonstrated stability superiority remain unachieved.

## Historical recovery state — block8 saved and video3 launch-only snapshot

The following prior text is retained as historical evidence. Its request-only video3 and no-active-training statements are superseded by the completed evaluation and active block9 snapshot above.

### Historical current-state heading — recorded before video3 completion

Checkpoint accounting is fixed at **2026-09-10T11:36:41.705167Z**. Block8 finished
SUCCEEDED/exit0 and Isaac exited at11:35:19Z per parent. This is completed training,
not physical task success. The new video3 started at11:38:47.227154Z; its actual
policy load/actions/outcome have not been reviewed here. Root owns that process.

### Saved checkpoint source differs from current production

Latest immutable saved checkpoint:
[checkpoint_step_000145920.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000145920.pt).
Checkpoint SHA256:
e5a62122d091c9d826880f8c0df2338846d8d7dab7ad0cb08944b8db8a14fbfc.
Official immutable manifest SHA256:
8de12f0d7d8e24cd7bcd0c43c632fa178ac6088bd9df1ac959e50ee48da55b1d.
Parent recomputed both hashes and verified the official roundtrip; this report
read immutable metadata and the fixed completed snapshot, not tensor data.

Saved lifetime: **145920 decisions /1105 PPO updates /22100 optimizer steps**.
New since140544/1063/21260: **5376 /42 /840**, all sampled/optimized/saved,
zero unfinished rollout. Effective saved LR1e-5, identity normalizer retained.
Saved source is **69aeaca777dcc8653e60f19da56ae1cd002e5271** /
runtime21ad0d667c84e5c53e5e8aa37239421f8dcc4f717f5bcf6d1c817c2dd95293a4.

Current production is separately **7db0d17f398d393ce026b6990bd2566d53366407** /
runtime5eb1536f17f3037639015e714f17ae8e805489c35ca3cf5c4318546ef3a4ac93.
There is no claimed7db checkpoint.145920 is latest_not_best, not a success policy.
The publication index remains a read-only reference, never official resume metadata.

### Completed P10 block8

Run20260910T1036035708863Z_g69aeaca777dc_1c97c4bee3f548da83b7860abcf8d541
actually saved **512 decisions /4 PPO updates /80 optimizer steps**.
Saved phase samples P10=2/P11=2/P12=508, P13=0. All4093 physical policy ticks
have verified native effects:4089 own-phase ticks and four ordinary handoff-hold
ticks, with no forbidden state writes. Prefix events have zero policy credit.

Episode0 had456 decisions/3647 policy ticks and ended P12
INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED:
RL I11/Q4, all four Q attempts subsequently ground-revoked, C/P0.
Episode1 had56 decisions/446 policy ticks and ended P12 HARD_JOINT_LIMIT:
actual rear_right_knee−60.025815543° despite target−41.584803440°.
RL had no new I/Q/C/P in that second suffix. Unknown load stays unknown, not zero.
See [episode0 diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p10_block8_episode0_terminal.md),
[episode1 diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p10_block8_episode1_hard_limit.md),
and [completed block8 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T113641705167Z.json).
Their older384-saved/128-pending timing statements remain historical; this final
snapshot now confirms all512 saved. Neither episode completed the RL task or P13.

The old collector ran an unused third63.2s teacher prefix before the last update
saved. It contributes **zero** policy decisions, episodes, events or update credit.
RR C/P present in suffix observations was teacher history, not new policy success.
Actual block8 load identity was10:52:49.789459Z/PID19228 atP10/tick7584/READY
after real reset and source145408 loading; it does not identify the new video process.

Cumulative saved phase counts:
**P01=33/P02=1690/P03=59/P04=42/P05=2139/P06=548/P07=24/P08=17/
P09=184/P10=3/P11=3/P12=579/P13=55**.
There are **26 real task/safety/incomplete terminals**:
8 BODY_COLLISION task failures,17 independent safety aborts (FALL15/HARD2),
and1 finite incomplete-task terminal. Seven nonterminal collection tails remain.
No collision and no safety abort still do not imply task completion.

### Terminal-tail scheduling revision and exact recovery migration

Only semantic_training.py changed in7db: for supported N1, identity-normalized,
nonrecurrent/no-RND collection, a true terminal on the last rollout tick is
persisted, stored and updated before reset; a verified checkpoint is forced,
and reset occurs only when another rollout begins. Mid-rollout terminals and
ordinary nonterminal phase transitions retain their prior behavior.
The six configs, physics, task/reward, nominal, action mapper, HISTORY372/12
policy kernel and hyperparameters are unchanged. This is not a new MDP.

[Regression XML](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/rollout_tail_reset_regression_v1.xml)
records376 CPU tests passed,0 failures/errors/skips,51.315s. This includes official
PPO terminal-return/save seams; it does not prove real Isaac reset RNG or future
physical trajectories are bitwise identical.

Resuming saved69aeaca checkpoint145920 undercurrent7db requires the exact
[terminal-tail migration](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/migrations/terminal_tail_reset_from_000145920_to_7db0d17.json),
SHA d38f63f0c88a581bfae8bccd2e641e9c7be546dbcca425f5c26212e2793d190e.
It preserves actor/critic/std, Adam and actual LR, normalizer, checkpoint RNG,
counters and spent budgets; discard old rollout and start legal P01 or a declared
zero-credit prefix. No NewMdpWarmStart, new network or unconditional RNG reset.
Do not omit this plan merely because the observation dimension remains372.

### Video3 is ACTIVE_REQUEST_ONLY, not an evaluation result

Run20260910T1138465345417Z_g7db0d17f398d_9bd793c700bd4040b4c517a0cd625ebb
was launched with parent PID43176, source145920 and the exact migration above.
[Started manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260910T1138465345417Z_g7db0d17f398d_9bd793c700bd4040b4c517a0cd625ebb/run_manifest.started.json):
semantic_residual_eval/C, naturalP01/full_episode, seed4001/N1, visible capture,
MaxDecisions3000, no short Decisions argument, no teacher prefix or warm start.
Setup had0actions at the launch report; later sampled actions and exact load proof
are unreviewed, not asserted zero forever. Video evaluation has no optimizer
updates or checkpoint writes. Do not launch a concurrent recovery process.

Latest **completed naturalP01 evaluation still uses142848**, seed2001:
617 decisions/41.075s, P05 LOCAL_TASK_DEADLINE, first unfinished FL crossing/
controlled placement, no BODY/FALL, task_success=false.145920 roundtrip and the
new video request do not replace that missing latest-checkpoint full evaluation.

After the active process ends, any recovery command must use actual current HEAD,
the immutable145920 source and its7db migration unless a newer verified checkpoint
is deliberately selected. A same69aeaca resume or old142848 migration is not the
current recovery route. Neither successful recording nor task completion alone
would establish FSM-relative stability improvement.

### Timeline publications and preserved history

The unchanged3556-row consolidated CSV plus2560-row block7 delta remain available.
The new [block8 delta](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_block8_000145920.csv)
adds512×263 withteacher0, P10=2/P11=2/P12=508, all512 saved/optimized.
Its [QA review](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_block8_000145920_authoring_qa/visual_review.md)
records scalar/PowerShell/visual verification and no error literals; missing
raw body fields remain blank for512 rows. The three disjoint files total
**6628 rows =5376 saved train+1252 completed nonvideo eval**, excluding video3.
No old CSV was overwritten or extra sheet generated here.

All four previous main reports are archived before this update with suffix
before_completed_block8_20260910T113641705167Z; historical diagnostic and
checkpoint timing claims remain unchanged in those files and below.

## Historical recovery state — prior completed-block7 snapshot retained verbatim

The following older current-status labels are historical, superseded by the
completed-block8 accounting and distinct current7db video request above.

## Current recovery state — block7 complete; P10 curriculum request active

Saved accounting is fixed at **2026-09-10T10:35:47.134148Z**. The separate
block8 started-manifest status was read at10:38:03Z. No active pointer or
ongoing policy/prefix stream was polled for this report. Root owns the active
simulation; do not launch a concurrent Isaac process or repeat completed work.

### Verified completed block7 and current checkpoint

The naturalP01 run
20260910T0949490247355Z_g69aeaca777dc_1471cdeb04d243babf3b12b94ca7a42e
is **STOPPED_AT_VERIFIED_UPDATE_BOUNDARY**. It planned4096 decisions but actually
collected, optimized and saved **2560 /20 PPO updates /400 optimizer steps**.
The final manifest records requested_policy_decisions=2560,
planned_requested_policy_decisions=4096, unconsumed_requested_policy_decisions=1536,
rounding_overrun=0. The unused1536 are not samples, updates, missing data or a
required automatic rerun. This was a curriculum allocation at a complete update
boundary, not a task-success declaration or optimizer gate.

Latest immutable checkpoint at this boundary:
[checkpoint_step_000145408.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000145408.pt).
Checkpoint SHA256:
c5d66b2aea266a138dea607b8880a8bfab3526a3b22e985cd289c6f348565d61.
Official immutable manifest SHA256:
142d5d5298de1e309890b16450891b0913684402e0f93087c34706d9dc94bbc8.
Parent computed both byte hashes and verified the official roundtrip; the report
read the immutable sidecar and completed snapshot without tensor loading.
Alias/resume-state hashes and later active pointers were not reverified here.

Saved lifetime: **145,408 decisions /1,101 PPO updates /22,020 optimizer steps**.
New since140544/1063/21260: **4864 /38 /760**, all sampled, optimized and saved.
There is no unfinished rollout at this completed boundary. Source HEAD remains
69aeaca777dcc8653e60f19da56ae1cd002e5271, runtime
21ad0d667c84e5c53e5e8aa37239421f8dcc4f717f5bcf6d1c817c2dd95293a4.
Saved adaptive LR1e-5; identity normalizer hashc230b0db is retained.
The publication index is read-only, not official loader metadata.
145408 is **latest_not_best**, not a success checkpoint and not yet fully evaluated.

Block7 saved phase samples:
**P01=24/P02=1265/P03=42/P04=19/P05=989/P06=178/P07=3/P08=3/P09=37**,
P10–P13=0. Cumulative seven-run saved phase samples:
**P01=33/P02=1690/P03=59/P04=42/P05=2139/P06=548/P07=24/P08=17/
P09=184/P10=1/P11=1/P12=71/P13=55**.
These replace the earlier partial saved128/unknown-phase account. Its historical
snapshot is preserved, not retroactively relabeled.

Block7 has10 true safety terminals: **FALL9/HARD_JOINT_LIMIT1**, no full-task
success. The hard-limit record identifies front_right_knee; it is an independent
safety stop, not one of the two task-failure categories. The last completed
episode ended P09/FALL at21.116667s. The final episode's **57 decisions at
P02/tick456/3.8s are nonterminal**, ending only when the verified update completed.
They are saved data, not another failure or success. Across seven finalized runs:
24 real task/safety terminals: eight BODY_COLLISION task failures and16 independent
safety aborts (FALL15/HARD_JOINT_LIMIT1), plus seven nonterminal collection tails.

RR recorded I9/Q6 events and30 current-valid decision ends during block7,
but **no new crossing or placement**. Maximum measured lift200.227mm occurs at
a FALL terminal with current_lift_valid=false and load_fraction_valid=false;
it is not stable carry, and unknown normalized load is not zero load.
The actual2560 actions were nonzero with20442 verified native-effect ticks,
20386 own-phase request-effect ticks, and no forbidden root/velocity/force/gravity
writes in retained evidence. These counters do not identify a unique fall cause.
See [completed block7 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T103547134148Z.json)
and [final training manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0949490247355Z_g69aeaca777dc_1471cdeb04d243babf3b12b94ca7a42e/training_manifest.json).

The [bounded later-episode diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/natural_p01_block7_later_episodes.md)
confirms episode7's actual FR knee−60.002540312° despite target−58°.
For the five FALL records in episodes4–9, terminal372 float32 observations plus
locked obstacle planes imply base height below.015m; these are derived values,
not raw double-precision base measurements. Within that bounded evidence, no new
explicit execution/evaluation defect was found; neither lift differences nor
later phase visits establish learning or stability improvement.

### Latest completed full naturalP01 evaluation remains142848

Latest completed deterministic naturalP01/seed2001 evaluation still loaded142848:
**617 decisions /4929 physics ticks /41.075s**, zero training or optimizer credit.
It ended **P05 INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_TASK_DEADLINE**, first
unfinished FL front-edge crossing and controlled placement, no BODY/FALL.
It was not an external short-window cutoff or full traversal success.
145408's save/load roundtrip does not replace this missing latest-checkpoint
evaluation. Both older completed evaluations, failed-video initialization evidence,
and the [eval2 diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/seed2001_142848_p05_diagnosis.md)
remain intact. Quality RMS describes incomplete trajectories only; no FSM-relative
stability superiority or successful video is claimed.

### Active block8: P10-origin512, request only

Run20260910T1036035708863Z_g69aeaca777dc_1c97c4bee3f548da83b7860abcf8d541
started at10:36:04.060860Z, requested **phase_suffix/P10/512 decisions**, seed1001/
N1/CUDA0, checkpoint interval1, teacher offset0 and frozen_fsm prefix, starting
from immutable145408. It uses the same69aeaca runtime and ordinary resume:
no ResumeMigration, NewMdpWarmStart or policy-distribution migration.
See [started manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/train/20260910T1036035708863Z_g69aeaca777dc_1c97c4bee3f548da83b7860abcf8d541/run_manifest.started.json).

The parent reported teacher prefix without policy credit at dispatch. This report
has not reviewed new actual identity, credit start, samples, updates or saves.
Those counts remain **unknown**, not an asserted zero for all later time;512 is
only a request. Prefix physical events never enter policy credit. Earlier block7
09:50:14Z/PID36972 identity remains proof of that completed run, not proof of
this new P10 run's loaded objects. Root will provide its actual receipt separately.

After the active process ends, inspect the actual latest pointer and completed
update ledger before choosing recovery or remaining requests. Do not use this
fixed145408 index to overwrite a later checkpoint. Ordinary same-HEAD resume
from145408 requires no migration; if deliberately falling back to source142848
under69aeaca, its exact
[video-only plan](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/migrations/video_natural_reset_from_000142848_to_69aeaca.json)
is still required. That plan preserves weights/std/Adam/LR/normalizer/RNG/counters
and spent budgets, discards rollout storage, and is not a new MDP.
The six configs/control/nominal/reward/observation kernel remain unchanged.

Conditional non-video full evaluation reference after normal process exit and
same-HEAD verification; do not run it concurrently:

~~~powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 69aeaca777dcc8653e60f19da56ae1cd002e5271 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -Stage full_episode -FromPhase P01 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000145408.pt
~~~

### Publication and history

Latest consolidated CSV remains
[p09_evidence_timeline_000142848_eval2.csv](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000142848_eval2.csv):
**3556×263 =2304 saved train+635+617 eval**, eight completed runs through142848.
It excludes all of now-completed block7, active block8 and the zero-action video.
The consolidated CSV and prior CSVs/QA are untouched; no full-task-success video
is published by this report.

An additional disjoint [block7 delta CSV](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_block7_000145408.csv)
contains2560×263, all2560 optimized/saved, teacher0 and10 terminal rows.
Its [QA review](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_block7_000145408_authoring_qa/visual_review.md)
records parent visual inspection and independent PowerShell readback. Together
the unchanged3556-row consolidated CSV and this delta cover6116 rows:
4864 saved training+1252 evaluation, without overlap or any active block8 rows.

Separate [block8 episode0 diagnostic](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p10_block8_episode0_terminal.md),
fixed at11:03:20Z:456 sampled/384 optimized/72 not yet optimized; P12 task
incomplete via LOCAL_BOUNDED_RECOVERY_EXHAUSTED. RL I11/Q4, all4 Q revoked
on ground return, C/P0. Its145792 checkpoint is only a snapshot claim without
parent byte rehash; this supplement does not replace the main10:35:47 verified
145408 checkpoint or its accounting, and does not establish full-task success.

All four previous main reports were archived before updating under
[report_history](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/report_history),
with suffix before_completed_block7_20260910T103547134148Z. Older timestamps,
requests and evidence remain historical, not current instructions.

## Historical recovery state — prior09:54:44 snapshot retained verbatim

The following sections preserve earlier states. The completed10:35:47 accounting
and new P10 request above supersede their current-status labels.

## Current recovery state — fixed09:54:44; natural P01 training running

This authoritative section is a **fixed 2026-09-10T09:54:44.201940Z snapshot**,
not a live pointer. Later updates are intentionally excluded. Root owns the
active process/session54301; do not start another Isaac process, stop it, or
repeat the4096 request from this document.

Production is69aeaca777dcc8653e60f19da56ae1cd002e5271, runtime
21ad0d667c84e5c53e5e8aa37239421f8dcc4f717f5bcf6d1c817c2dd95293a4.
Control/MDP semantics and all six configs remain the28609010 version.
Only semantic_video.py changed at this revision; exact SAME372/12 policy
compatibility is retained. See the
[applied reset-proof receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_natural_reset_applied_69aeaca.md).
Its CPU regression XML records316 passed/0 failed/0 errors/0 skipped in20.092s.
The first173-test receipt with one launcher-environment assertion failure is
preserved. CPU success is not a live repaired-video or physical-success result.

### Saved, sampled and active accounting

Latest checkpoint **at this fixed boundary**:
[checkpoint_step_000142976.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000142976.pt).
SHA256: 29349ad04cdf557e6e45d6050a247807ab8c45d0825cedbf0860efaecd189563.
Official immutable sidecar SHA256:
01081c0a9f030f4854791f8c33187db3b4a44b72bf620329fd6b866ccde6e810.
Root computed both hashes and verified the official save/load roundtrip; this
report read the immutable metadata, not checkpoint tensors or active rawstreams.
Alias/resume-state byte hashes were not independently rechecked for this update.

Lifetime saved: **142,976 decisions /1,082 PPO updates /21,640 optimizer steps**.
New saved since140544/1063/21260: **2,432 /19 /380**. Saved source/runtime are
69aeaca/21ad0d66 above; adaptive LR1e-5 and normalizer hashc230b0db remain recorded.
It is latest_not_best, not a full-task-success checkpoint, and has not yet had
its own naturalP01 full evaluation.

The new naturalP01 training run
20260910T0949490247355Z_g69aeaca777dc_1471cdeb04d243babf3b12b94ca7a42e
requested4096 decisions, seed1001/N1/CUDA0, checkpoint interval1, unchanged
128-decision PPO rollouts. At the specified snapshot it had **245 sampled /
128 optimized and saved /117 unfinished-rollout decisions**, one PPO update
and20 optimizer steps. Cumulative sampled is2549, distinct from2432 saved.
Sampled phase counts for this active run only:
**P01=6/P02=179/P03=3/P04=1/P05=56**, P06–P13=0.
The saved128 phase split was not separately read; these245 phase counts must
not be relabeled saved. The report retains the six completed runs' exact2304
saved phase counts separately, plus128 saved with phase allocation unreviewed.

Its first episode ended in a real P05 FALL after166 decisions/11.008333s.
The terminal atglobal143014 is beyond saved142976, so this is sampled terminal
evidence, not yet a saved terminal at the fixed boundary. It is not a completed
4096 request or a task success. The117 unfinished rollout contains the last38
decisions of that episode and79 decisions of the next episode; none receives
extra optimizer/checkpoint credit here. RR I was seen, but no qualified Q/C/P
in this245-row snapshot.
See [fixed active snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T095444201940Z.json).

Actual loading is now proven, not merely requested:
[live identity](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0949490247355Z_g69aeaca777dc_1471cdeb04d243babf3b12b94ca7a42e/live_runtime_identity.json)
records09:50:14.775516Z/PID36972, training_after_actual_reset_and_checkpoint_load,
P01/tick0 and initialized true. Actual controller/nominal/config paths match
69aeaca/current project; logger control/import calls are0/0. This is not
source-FSM trajectory-equivalence or USD resolver proof.

### Latest completed full naturalP01 evaluation:142848, still P05 incomplete

The completed deterministic seed2001 evaluation loaded142848 under64c0324:
**617 decisions /4929 physics ticks /41.075s**, zero optimizer and training credit.
Phases: P01=2/P02=158/P03=4/P04=2/P05=451, P06–P13=0.
Task result: **INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_TASK_DEADLINE**.
The job's SUCCEEDED lifecycle only means it ran to completion.
There was no BODY_COLLISION/FALL, no external short-window cutoff and no
traversal success. First unfinished task remains **FL front-edge crossing and
controlled placement inP05**. Local age30.008333s exceeded the current30s limit;
current progress and allowance were0;158.925s remained in the global task budget.

FL produced fresh I1374/Q1378, then ground-return revoke2805, with C/P still0.
Nearest decision-end front distance was−21.437mm with+26.887mm top clearance;
maximum clearance+56.245mm occurred at a different tick. Actual base x showed
P05 net retreat .233751m; this is body position, not CoM. One local reward
interval gained transfer-evidence credit while geometry slightly regressed,
but this does not prove deliberate reward exploitation or a unique cause.
See [eval2 diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/seed2001_142848_p05_diagnosis.md)
and [eval2 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T094646793551Z.json).
Roll/pitch RMS .0922318/.1283252rad describe this incomplete trajectory only;
higher lift, nearer edge or lower RMS does not establish FSM superiority.
The older141568/635-decision evaluation remains preserved. The separate09:30
video attempt is still a zero-action initialization diagnostic before policy
load, not this completed evaluation and not a physical task failure.

### Migration and recovery after the active process finishes

The actual initial142848→69aeaca launch used
[migration plan](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/migrations/video_natural_reset_from_000142848_to_69aeaca.json),
SHA42d4eea45cd9d5ee39246477fd09fa265065108e8565c2586f8073d5f01ce51e.
It preserves actor/critic/std, Adam and actual LR, normalizer, RNG, counters and
spent budgets; old rollout is discarded and fresh legal P01 data collected.
It is video instrumentation only, **not NewMdpWarmStart**. No six-config,
physics, nominal, reward, observation or policy-kernel change is authorized.

A real69aeaca checkpoint142976 now exists, so ordinary same-HEAD continuation
from it does not reuse the source142848 migration. If recovery must instead
use old142848, its exact plan above remains required. After normal process exit,
inspect the actual latest pointer/sidecar and completed updates before choosing
the remaining training request; do not repeat4096 or assume this fixed128 is
the final saved total. Later pointers may already exist.

A conditional, non-video full evaluation entry using this fixed69aeaca
checkpoint is below. It must not run concurrently with root's active process:

~~~powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 69aeaca777dcc8653e60f19da56ae1cd002e5271 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -Stage full_episode -FromPhase P01 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000142976.pt
~~~

### Fixed CSV scope and remaining deliverables

Newest completed export:
[p09_evidence_timeline_000142848_eval2.csv](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000142848_eval2.csv),
**3556 rows ×263 columns =2304 saved train +635+617 eval**, eight completed runs.
It excludes the active seventh training run, including saved142976, and the
zero-action video diagnostic. Teacher credit0. All1252 evaluation rows use
exact physical joins;2304 missing training body attitude/linear rows remain
blank, not replaced by command or CoM. The parent verified every scalar and
reviewed the QA image; see
[visual review](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000142848_eval2_authoring_qa/visual_review.md).
Earlier CSVs/QA and report snapshots are preserved.
No fullP01 success, successful ≤200s PPO video, paired FSM/PPO video or
full-task stability superiority is claimed.

## Historical recovery state — block6/eval1 snapshot retained verbatim

The following sections describe earlier timestamps and requests. They are
preserved evidence, superseded by the fixed09:54:44 accounting above.

## Current recovery state — block6 saved; video C initialization diagnostic

Fixed evidence snapshot: **2026-09-10T09:32:56.5152480Z**. The latest saved
checkpoint uses source HEAD64c03243ac05e43e7a5843eaee252eef1136c925 and runtime
hash99ea40879c1ba6e2618bfba901d2be77edafacb039114f641725f3e325249dee.
Its control/MDP semantics remain the28609010 revision;64c0324 added video-only
runtime changes, not a new MDP or a new372-dimensional policy kernel.

Latest immutable checkpoint:
[checkpoint_step_000142848.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000142848.pt).
SHA256: 7fb0a5a1607125d477c7e49c05422a40e3cb33d72dcad83c1fb1aeaf5e59eaa1.
Official sidecar SHA256: fb89869010f6fd100e4d0c37f3e5c2ba43e79df4312ec805f92a5b86bca4344e.
Saved lifetime counters: **142,848 decisions /1,081 PPO updates /21,620 optimizer steps**.
Relative to the preserved140544/1063/21260 baseline: **2,304 /18 /360** newly
sampled, optimized and saved; teacher prefixes and the video diagnostic add no
credit. There is no unsaved training tail in this six-block accounting.
The final saved adaptive LR is1e-5. Official save/load roundtrip is true;
immutable/alias checkpoint hashes, official/resume sidecar hashes and the
unchanged pointer were verified without Torch or tensor loading.
[checkpoint_last_manifest.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last_manifest.json)
is a read-only publication index, never replacement loader metadata.
This is **latest_not_best**, not a full-traversal-success checkpoint.

### Completed P04 block6 and bounded physical evidence

Run20260910T0852458310353Z_g64c03243ac05_aafdb9864bdc42dda72ef793af390efd
completed its1024-decision request: **1024 decisions /8 PPO updates /160 optimizer
steps**, using eight unchanged128-decision rollouts, source141824 and no migration.
Actual phase samples: **P04=5/P05=761/P06=220/P07=2/P08=6/P09=30**; others0.
There were three real safety terminals: FALL atP06/t26.166667s, FALL atP05/
t23.241667s, and BODY_COLLISION atP09/t40.075s. The fourth episode's316-decision
tail ended atP05/t35.2s because the collection budget ended: **no terminal,
no FL placement, neither success nor another failure**. Its retained
CONTACT_BEARING_UNVERIFIED and RR load_fraction_valid=false mean load is unknown,
not zero.

The actual P04 identity was recorded **08:57:20.957634Z, PID23900**, after real
reset and checkpoint141824 loading, atP04/tick1696/READY. Active semantic
controller, nominal provider, source MotionExecutor, mapper, sensing and config
paths match this project/runtime;14 module paths were checked on disk. This is
real process evidence, not a started-manifest assumption or offline reconstruction.
It does not prove source-FSM trajectory equivalence or USD dependency resolution.
See [actual identity](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0852458310353Z_g64c03243ac05_aafdb9864bdc42dda72ef793af390efd/live_runtime_identity.json).

The [first-two-episode diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p04_first_two_completed_episodes.md)
covers318 decisions, including one fresh FL placement and two real FALL terminals.
The retained evidence lacks exact base_z/projected_gravity_z, so it cannot
independently choose the violated FALL subpredicate. Episodes span existing PPO
updates and are not matched frozen-actor trials; the existing transient adaptive
LR1.5e-5 was not a new hyperparameter setting.
The [third-episode diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p04_third_episode_to_p09.md)
covers390 decisions: fresh FL Q2618/C2828/P2864, then P07/P08 preparation and
P09 BODY_COLLISION, but no qualified RR lift/crossing/placement. At P08→P09,
CoM moved toward FL while FL was AIR; this is not FL bearing. RR's earlier
incidental I inP06 was not qualified Q. The first unfinished RR task was usable
lift/carry, then crossing and placement—not holding an already-qualified lift
for a fixed duration. Exact body-contact manifold/gap/force is not retained and
is not inferred. Neither bounded diagnosis proves a unique causal action chain.
See [completed block6 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T092950976322Z.json).

Cumulative saved phase samples are:
P01=9/P02=425/P03=17/P04=23/P05=1150/P06=370/P07=21/P08=14/
P09=147/P10=1/P11=1/P12=71/P13=55.
The six blocks have14 safety terminals (FALL6/BODY_COLLISION8) and six
nonterminal collection tails. Prefix RR events remain excluded.

### Evaluation boundary and zero-action video diagnostic

The latest **completed naturalP01 evaluation still loaded141568**, seed2001:
635 decisions/42.283333s, P05 LOCAL_TASK_DEADLINE, first unfinished FL
crossing/placement; no BODY_COLLISION/FALL.142848 has a save/load roundtrip,
but no completed naturalP01 evaluation yet.

The new C/seed4001/P01/no-teacher attempt
20260910T0930063395586Z_g64c03243ac05_4bb994da17ed47bea5c9e0b04c3a2a41
ended at09:30:41.953417Z with lifecycle DIAGNOSTIC_FAILURE:
SemanticVideoError: video must own the first natural reset of a fresh process.
The final manifest records reset_count1/options{}, training_phase_snapshot="P01",
checkpoint_load_provenance=null, **0 issued actions /0 environment steps /
0 optimizer updates**, physical_episode=null and physical_task_success=null.
This is an initialization-guard/P01-marker mismatch **before policy loading**,
not a PPO task failure or another completed naturalP01 evaluation.
See [final diagnostic manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation/20260910T0930063395586Z_g64c03243ac05_4bb994da17ed47bea5c9e0b04c3a2a41/run_manifest.json).
No new checkpoint or successful video resulted. Root owns a minimal video-only
guard repair after the process exited; this snapshot has not verified a repaired
HEAD, test result or rerun. Do not blindly repeat the failed64c0324 video command.

### Recovery entry and remaining work

Do not rerun the already-completed P04 request1024. Before any new process, read
the actual pointer and current committed runtime, and coordinate with root.
On **unchanged clean64c0324 only**, after confirming no active simulation, the
existing non-video naturalP01 evaluation entry remains:

~~~powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 64c03243ac05e43e7a5843eaee252eef1136c925 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -Stage full_episode -FromPhase P01 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000142848.pt
~~~

This is a conditional recovery reference, not authorization to launch a second
process or a substitute for root's video repair. If the actual committed
runtime changes, use its explicit reviewed video/instrumentation-only SAME372
migration, preserving actor/critic/std, Adam moments and actual LR, normalizer,
RNG, lifetime counters and spent budgets. Such a runtime-only migration is
**not NewMdpWarmStart**; do not reset compatible states or bypass the runtime
identity check. No future HEAD or migration path is invented here.
The remaining workflow includes naturalP01 training/evaluation and genuinely
paired FSM/PPO video evidence; no suffix success substitutes for a full task.

The newest fixed CSV remains
[p09_evidence_timeline_000141696.csv](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000141696.csv),
**1787 rows =1152 saved training+635 earlier evaluation**, cutoff141696.
It excludes completed blocks5/6 and the zero-action video diagnostic. Its QA
and the original1659-row CSV/QA are preserved. The
[applied-video report](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_applied_64c0324.md)
and231-pass CPU receipt remain limited to their stated test scope; the real
initialization diagnostic is additional evidence, not hidden by those tests.
No full-task stability superiority, successful PPO video or paired comparison
video is claimed.

## Historical recovery state — prior block5 snapshot retained verbatim

The following prior sections describe their own older timestamps and plans;
they are preserved evidence, not current commands or current counters.

## Current recovery state — block5 saved; P04 request1024 active

Publication verified **2026-09-10T08:57:47.0058826Z**. Production and the latest
saved checkpoint both use `64c03243ac05e43e7a5843eaee252eef1136c925`; control/MDP
semantics remain28609010. No new migration, normalizer reset, Adam reset, policy
kernel change or new-MDP warm start is required for the current same-HEAD resume.

Latest immutable checkpoint:
`outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000141824.pt`.
SHA256: `73fb969d5af9b233172e161c7b2ef0aa8a8011aac04115cc05aca054606ba000`.
Official sidecar SHA256: `5cd29a7107453cbdc20d4b9790ed5607be4e88d84e13bfdbf62c1d249628e13a`.
Saved lifetime counters: **141,824 / 1,073 PPO / 21,460 optimizer**.
New saved totals since140544/1063/21260: **1,280 / 10 / 200**. Effective LR remains
`1e-5`. The official save/load roundtrip passed; immutable and alias checkpoint
hashes, official/resume-state hashes and the unchanged pointer were verified.
`checkpoint_last_manifest.json` is still only a read-only publication index.

### Completed P10 block and first unfinished task

Block5 actually collected128 policy decisions and completed1 PPO update/20
optimizer steps. Credit by source phase: **P10=1, P11=1, P12=71, P13=55**.
There were no task terminals. Its128-decision tail ended at P13/tick8608/
71.733333s; on-policy time was8.533333s after the teacher prefix.

This is now real loading/credit evidence, not merely a started manifest:
`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0828133987091Z_g64c03243ac05_8f4efe6ab2af4675b6e3f42469a8acfa/live_runtime_identity.json`
records **08:47:15.728149Z, PID22760, P10/tick7584, READY**, after actual reset and
checkpoint141696 loading. Runtime64c0324/hash99ea4087 matches; actual classes,
configuration paths and source project agree. Policy credit began at63.2s while
the requested P10 was still active. Teacher RR Q6912/C7109/P7579 were already
present and receive **no new policy event credit**.

The policy suffix produced fresh **RL I7746/Q7750/C8089/P8164**. At the end,
all placement histories and the final region were valid, but **final_controlled
was false** and the existing final observation had not started. All55 P13 sampled
ends passed region/support/evidence; wheel speed failed the existing limit in
55/55, body speed in46/55 and angular speed in28/55. Final body speed .115041m/s
and FL/RL/RR wheel speeds .892218/.383814/.504495rad/s remained too high; angular
speed .284159rad/s passed. RL being currently AIR is not an extra failure gate:
FL/FR/RR provide actual TOP support and the region test already passes.

P13 was sampled for only3.666667s. Its source nominal wheels were still +.3rad/s
throughout these55 rows, within the source0–16.733333s rolling segment and before
the finite17.8s nominal home/zero tail. This is an observed nominal/finish-goal
tension, not a dispatch-failure verdict or proof that a future stop will succeed.
Residuals remained continuous at transitions; new RL events occurred before the
single update, so they do not prove update1073 improved the policy.
See [completed P10/P13 diagnosis](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p10_block128_rl_and_p13.md)
and [block5 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T085154904868Z.json).
Neither this suffix nor its nonterminal tail is complete naturalP01 success.

### Active P04 request and current recovery command

The new started manifest is
`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0852458310353Z_g64c03243ac05_aafdb9864bdc42dda72ef793af390efd`.
It requests **P04-origin1024 decisions**, seed1001/N1/CUDA0, offset0,
`frozen_fsm` prefix and checkpoint interval1. The request was expanded from the
earlier512 plan to allow **eight unchanged128-decision rollouts** and a real
P05 task outcome; this does not change PPO hyperparameters or MDP semantics.
The started timestamp is08:52:46.550594Z. For this new active P04 run, this report
has not inspected actual loading/credit-start or policy samples; requested1024
is not sampled or saved credit. Root owns session3185. Do not launch another
Isaac, touch its process, or interrupt its unfinished rollout.

After the active process exits normally, inspect the actual latest pointer and
completed-update accounting first. If it still points to141824 and this request
has not saved any update, the same-HEAD restart entry is:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 64c03243ac05e43e7a5843eaee252eef1136c925 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P04 -Decisions 1024 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

No `-ResumeMigration`, `-NewMdpWarmStart`, or policy-distribution migration flag.
If some P04 updates have already been saved, use the actual latest checkpoint
and determine the remaining authorized request; do not repeat1024 blindly.
The follow-up plan remains naturalP01 training and full deterministic evaluation
from the actual latest saved checkpoint, not an exclusive suffix curriculum.

**Latest saved141824 is not the latest completed naturalP01 evaluation.** The
latest completed full-episode evaluation still loaded **141568**, seed2001,
and ended after635 decisions/42.283333s at P05 `LOCAL_TASK_DEADLINE`, with FL
crossing/placement unfinished and no BODY_COLLISION/FALL.141824 has a verified
save/load roundtrip but has not yet completed its own naturalP01 evaluation.
The unchanged same-HEAD evaluation command after training/process exit is:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 64c03243ac05e43e7a5843eaee252eef1136c925 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -Stage full_episode -FromPhase P01 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

The newest CSV remains
[p09_evidence_timeline_000141696.csv](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000141696.csv):
**1787 rows =1152 saved training+635 earlier evaluation**, through141696 only.
It excludes completed block5 and the active P04 run until another final export.
Its QA and the older1659-row CSV/QA remain unchanged. Applied video64c0324 and
the231-pass CPU receipt remain valid within their stated test scope; no current
success video, fullP01 checkpoint success or FSM stability improvement is claimed.

## Earlier recovery snapshots — retained, superseded for current accounting

The following sections retain their original timestamps, old planned512 request,
then-pending P10 load, saved counts and commands. They are historical records,
not the latest operating instruction. The block5/P04 section above supersedes
them; old partial samples and teacher events are never counted again.

## Current recovery state — video-only revision 64c0324

Status recorded **2026-09-10T08:30:24.047381Z**. Production HEAD is
`64c03243ac05e43e7a5843eaee252eef1136c925`; the control/MDP semantics remain
`28609010db4e57c5b34304a4ae2563c69f9d00b9`. The latest saved checkpoint is from
the intervening `06716a88` runtime. These three boundaries must not be conflated.

The fourth block, P07-origin 128 decisions, completed one real PPO update and
20 optimizer steps, with an actual save/load roundtrip. Current immutable source:
`outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000141696.pt`.
Checkpoint SHA256: `70ae4fedc9210b7e48d4ec33a825aa9b5e47f6acf7d80e064914f03920ff4395`.
Official sidecar SHA256: `852500f344b85b4ee398d79e49abf93dff34f38261f2489079473519b6a19d06`.
Saved lifetime counts are **141,696 / 1,072 PPO / 21,440 optimizer**; relative to
the unchanged 140,544 / 1,063 / 21,260 baseline, **1,152 / 9 / 180** are saved.
The actual pointer, immutable/alias checkpoint hashes and official/resume-state
hashes were checked at the timestamp above. The publication index is only an
index; `resume_state.json` and the immutable official sidecar remain authoritative.

Block4 phase credit is P07=7, P08=7, P09=114. Its six completed episodes remain
BODY_COLLISION failures. Episode6 has a 57-decision nonterminal tail ending
P09/tick6368/53.066667s, with RR not placed. The first lift attempt was revoked
on ground return at6055; the new edge attempt qualified at6249, continued through
actual contact and AIR, and crossed at6328. The final RR position had retreated
outside the top-placement region. Crossing history is not current placement or
full traversal. All these samples preceded the block's one update; their events
do not establish that the updated policy has improved.
See [block4 evidence](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p07_edge_crossing_block128.md)
and [completed snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T082308229118Z.json).

### Active P10 request and safe continuation

The started manifest records a new real P10-origin request, **128 decisions,
seed1001, N1, CUDA0**, teacher offset0 and `frozen_fsm` prefix:
`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0828133987091Z_g64c03243ac05_8f4efe6ab2af4675b6e3f42469a8acfa`.
Its started timestamp is 2026-09-10T08:28:14.072875Z. This report has verified
the startup configuration only, **not actual policy loading, live identity or
policy credit-start**. No requested or prefix decisions are added to saved counts.
Do not start another Isaac or interrupt the root-owned active process.

The exact video-only migration is
`outputs/ppo_fsm_reference_p09_stable_v2/migrations/video_from_000141696_to_64c0324.json`.
It binds source141696/06716a8 to64c0324 and five reviewed production paths only.
All six configs, physical control, nominal, reward, preprocessing, N1 topology
and HISTORY372 kernel are unchanged. It preserves weights, learned std, Adam
moments, actual LR `1e-5`, normalizers, RNG, counters and spent budgets; old
rollout storage is discarded and reset/prefix physics is recollected legally.
**Do not use `-NewMdpWarmStart` or the older141568 instrumentation plan.**

Only after normal process exit and checking the actual latest pointer, if the
source still is141696, the approved restart command for this request is:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 64c03243ac05e43e7a5843eaee252eef1136c925 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P10 -Decisions 128 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000141696.pt -ResumeMigration outputs/ppo_fsm_reference_p09_stable_v2/migrations/video_from_000141696_to_64c0324.json
```

After a new64c0324 checkpoint is actually saved, ordinary same-HEAD continuation
uses its latest pointer with no migration or warm-start flag. The root's next
sampling plan is **P04 continuous512, then naturalP01 training and evaluation**;
this is a plan, not completed credit or an exclusive P10 curriculum. The planned
P04 command, only once a same-HEAD checkpoint exists and the active run has exited:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 64c03243ac05e43e7a5843eaee252eef1136c925 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P04 -Decisions 512 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

The completed naturalP01 evaluation still refers to141568, not141696:
635 decisions/42.283333s, P05 `INCOMPLETE_CONTROLLER_BLOCKED` from
`LOCAL_TASK_DEADLINE`, no BODY_COLLISION/FALL, FL crossing/placement unfinished.
The new saved policy requires its own naturalP01 evaluation; no success or FSM
stability superiority has been established. Once a latest same-HEAD checkpoint
exists, the unchanged deterministic evaluation entry is:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 64c03243ac05e43e7a5843eaee252eef1136c925 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -Stage full_episode -FromPhase P01 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

### Current report artifacts

The [new fixed CSV](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000141696.csv)
contains1787 rows/263 columns:1152 saved training decisions from four blocks plus
the635-decision141568 evaluation. Active P10 is excluded. Parent scalar and
[visual QA](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline_000141696_authoring_qa/visual_review.md)
passed; missing measurements remain blank with valid flags. The prior1659-row
CSV and QA are retained unchanged.

[Applied video record](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/video_applied_64c0324.md):
the video-only patch is now applied, with **231 passed,0 failed,0 errors,0 skipped**
in `video_current_experiment_regression_v2.xml` (17.411s). This includes a real
CPU HISTORY372 update/save/reload and preserved Adam/RNG/normalizer/kernel checks.
The first failed v1 receipt remains intact. CPU success is not live video or
full-task success. No PPO success or comparison video is claimed.

## Preserved prior recovery snapshots — not current commands

Everything below is retained history. Its earlier headings saying "current",
12/50/61 sample fragments, old saved counters, and deferred/unapplied video
status describe their own timestamps. The64c0324 section above and actual latest
pointer supersede them; never add historical samples twice.

## Current recovery state — logging-only revision 06716a8

Production HEAD: `06716a88bcc9ee2fff615bcf2681dd59d97142e5`.
Project: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1`.

The current run is
`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0647306226372Z_g06716a88bcc9_c6cfcadd350b418a9969f7dfc35406a5`.
Its actual started manifest requests **128 P07-origin policy decisions**, N1,
seed1001, CUDA0, teacher offset0, `frozen_fsm` prefix, checkpoint interval1.
Do not start a second Isaac process. The teacher prefix and any pending rollout
are not saved policy credit; use a subsequently published checkpoint pointer,
not the requested128 count, to determine whether this block has completed.

### In-progress snapshot — 2026-09-10T07:56:44.591846Z

One `summarize_live.py --kind train --write` inspection recorded **61 sampled
policy decisions**, phase counts **P07=5, P08=5, P09=51**, and470 on-policy
physics ticks. The five completed episodes contain12/12/12/14/11 policy
decisions; all terminated in **P09 BODY_COLLISION**, with the physical evaluator
reporting VALID/VERIFIED and BODY_CONTACT. The first uncompleted phase is P09;
RR has no crossing/placement. Recorded RR initial-lift evidence appears in2
decision samples, but lift-established/current-valid/crossing/placement counts
are all0. Diagnostic AIR duration is not a qualification or success gate.

This run has **0 new PPO updates, 0 optimizer steps and 0 saved decisions**.
The latest logged global decision141629 is a sampled counter, not a checkpoint.
The active process remains owned by the main task and continues collecting the
requested128-decision rollout; this snapshot neither stops nor completes it.
Teacher prefixes are excluded. Individual files were read at separate byte
boundaries, so this is a timestamped lower bound rather than an atomic live state.
No repeated polling, Torch load, checkpoint tensor rehash or Isaac launch was
performed by this report update.

Evidence: [07:56:44 snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T075644591846Z.json).
The original12-decision first-episode fragment remains in the report, and the
[07:41:59 snapshot with50 sampled decisions](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/progress_snapshots/20260910T074159118501Z.json)
is retained unchanged. These fragments are not summed or credited a second time.

The completed [CSV evidence snapshot](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_evidence_timeline.csv)
contains1659 rows/263 columns (1024 completed training+635 deterministic evaluation),
with [visual QA](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/p09_timeline_authoring_qa/visual_review.md).
It excludes all current06716a8 P07 samples. The
[deferred-video parent review](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/deferred_video/parent_review_06716a88.md)
records passed Python/PowerShell syntax checks, but **pytest and checkpoint
roundtrip tests have not run and the production video patch is not applied**.
The reviewed draft must not be imported into this active process and is not an
optimizer gate or evidence of successful traversal/video publication.

The latest verified checkpoint before this run is
`outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000141568.pt`.
SHA256: `1fa8de42754f04c5931b28f775f286c03a7791afeaf789fed0b51d4479607055`.
Sidecar SHA256: `8a24472f7f9dd7589522ca7cd08328342811702e6ac2efcfba00ce1ec169bf4b`.
Actual saved lifetime counts: **141,568 / 1,071 PPO / 21,420 optimizer**.
The experiment has added **1,024 saved policy decisions / 8 PPO updates /
160 optimizer steps** relative to140544. Source effective Adam LR is `1e-5`.
The save/load roundtrip passed. Latest is not best or complete-traversal success.

The new revision changes actual-object/path logging and exact SAME372 resume
compatibility only. It does **not** change the control/MDP semantics, six selected
configurations, actor kernel, nominal, reward, observation preprocessing, or
residual ranges from28609010. This is **not another new-MDP warm start**:
actor, critic, learned std, Adam moments and effective LR, identity normalizers,
RNG, lifetime counters and spent budgets are preserved. Old rollout storage is
discarded, and reset/prefix physics is recollected legally. The explicit plan is
`outputs/ppo_fsm_reference_p09_stable_v2/migrations/runtime_identity_from_000141568_to_06716a8.json`.
Its scope is a reviewed logging-only claim, not a claim that two physical
trajectories are identical.

The first saved-policy deterministic naturalP01 evaluation is **complete**:
`runs/ppo_fsm_reference_p09_stable_v2/validation/20260910T0624345531274Z_g28609010db4e_2d880bd470694bcea9a6028c0902774e`.
It loaded141568, seed2001, no teacher prefix, and made zero optimizer updates.
It ran **635 decisions / 5,074 physics ticks / 42.283333s**, completing P01–P04
but not FL crossing/placement in P05. Recorded terminal:
`INCOMPLETE_CONTROLLER_BLOCKED`, with semantic source `LOCAL_TASK_DEADLINE`.
P05 elapsed30.016667s against effective30s, with progress allowance0 and global
time still157.716667s. This was a finite task terminal, **not** an external
diagnostic cutoff, BODY_COLLISION, or FALL. FL was back on ground, crossing and
placement false. Full traversal remains incomplete. Evaluation samples are not
training credit. Window quality and complete evidence are in
`training_and_evaluation_manifest.json`; no full-task superiority over FSM or
success video is claimed.

### Recovery commands

Run commands only after the active Isaac run has exited or saved and stopped at
a complete update boundary. Check `checkpoints/checkpoint_last_pointer.json`
first. If it still points to141568 under28609010, the current approved startup is:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 06716a88bcc9ee2fff615bcf2681dd59d97142e5 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P07 -Decisions 128 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000141568.pt -ResumeMigration outputs/ppo_fsm_reference_p09_stable_v2/migrations/runtime_identity_from_000141568_to_06716a8.json
```

Do not pass `-NewMdpWarmStart`; that would reset Adam again. The migration plan is
bound to exactly141568 and must not be reused for a later checkpoint. Once the
current run has published a verified06716a8 checkpoint, ordinary continuation
uses its actual latest pointer with **no migration flag**:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 06716a88bcc9ee2fff615bcf2681dd59d97142e5 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P07 -Decisions 128 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

That128 block is an example continuation, not authorization to assume a completed
block or an exclusive long-term P07 curriculum. Retain naturalP01 and later-stage
sampling according to the actual unfinished tasks. After a new same-HEAD saved
checkpoint exists, deterministic naturalP01 evaluation uses:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 06716a88bcc9ee2fff615bcf2681dd59d97142e5 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -Stage full_episode -FromPhase P01 -TeacherOffsetDecisions 0 -PrefixSource frozen_fsm -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

`live_runtime_identity.json` is emitted only after the actual reset/prefix and
checkpoint restore. The current run's real file appeared at
**2026-09-10T07:00:48.739846Z**, PID28624, P07/tick5912, prefix `READY`:
`runs/ppo_fsm_reference_p09_stable_v2/train/20260910T0647306226372Z_g06716a88bcc9_c6cfcadd350b418a9969f7dfc35406a5/live_runtime_identity.json`.
Its06716a8 revision and runtime-content digest match the started manifest. Actual
Python is the expected `env_isaaclab` executable, cwd is this PPO project, and all
14 recorded `wlr50_clean` modules resolve within its `src`. Active objects are
`SemanticControllerAdapter`, `NominalMotionProvider`, source `MotionExecutor`,
`RobotAdapter`, `ServoTargetMapper`, and `SemanticSensorReader` with exact-pair
contacts and semantic collider geometry. Actual backend task/profile paths point
to the new namespace; loaded FSM/contract paths point to this project's frozen
`configs/fsm_states.yaml` and `configs/recording_motion_contract.json`.
The active robot's stored asset config is
`C:\robotics_sim\wlr_robot\usd\wlr_robot_drive_test.usd`.

Recovery is explicitly absent on semanticN: a loaded `RecoveryPlanner` class is
not an active selected recovery object, just as a loaded `SensorFsmController`
class does not make the active controller originalA. The logger performed zero
control calls and zero imports, with no historical-run backfill. This actual
object/path evidence does not certify USD-resolved dependencies, fullA controller
equivalence, policy updates, task success, or matching physical trajectories.

## Historical notes — preserved snapshots, not current commands

The text below is retained for chronology. Its old HEADs, provisional run status
and commands are superseded by the current section and actual latest pointer.
In particular, references below to pending evaluation or unapplied logging
drafts describe earlier snapshots, not the present06716a8 runtime.

Historical production revision: `28609010db4e57c5b34304a4ae2563c69f9d00b9`.
Project: `C:\robotics_sim\wlr_robot\fsm_base_on_recording_ppo_phase_v1`.

The initial source is the immutable all-stage checkpoint at
`outputs/ppo_all_stage_acceptance_v1/checkpoints/history/checkpoint_step_000140544.pt`.
It contains **140,544 saved policy decisions / 1,063 PPO updates / 21,260 optimizer steps**.
SHA256: `c5ab9153671cff596fb7dc1059b8064adf7322c3abd42bc442c580be13ead49c`.

The previous run logged an additional update at 140,672 / 1,064 but did not save it.
That update and its remaining rollout are not recovered or counted as this experiment's learning.
The previous run remains unfinalized, with no process present at this revision's startup.
Its exact interruption cause is not established.

The new-MDP boundary preserves actor, critic, learned distribution, normalizers,
training RNG, lifetime counters and spent stage budgets. Adam moments are cleared;
the verified source effective learning rate is retained (`1e-5`, configured initial
default remains `3e-5`). Observation shape remains 372; the existing RR lift bit
now expresses current validity. Source-derived nominal and event semantics changed.
Every new rollout is collected under this committed revision.

## Verified continuation checkpoint and current run

**First fixed-version check completed:** immutable
`checkpoints/history/checkpoint_step_000141568.pt`, SHA256
`1fa8de42754f04c5931b28f775f286c03a7791afeaf789fed0b51d4479607055`.
Save/load round trip passed. Lifetime counts are **141,568 policy decisions /
1,071 PPO updates / 21,420 optimizer steps**; this revision has added
**1,024 / 8 / 160**, all saved, without crediting teacher prefixes or old unsaved
data. The final 384-decision P01 block
`20260910T0616115212299Z_g28609010db4e_f2de0b3725a5425e9cbd9bff946c1dd9`
ended at a complete update boundary, with its one episode nonterminal at 25.6 s,
P05; FL placement was still unfinished. This collection boundary is neither
task success nor a task-timeout label.

Aggregated policy phase credit across the three completed blocks:
P01=9, P02=425, P03=17, P04=18, P05=389, P06=150, P07=12, P08=1,
P09=3, P10=P11=P12=P13=0. Later-stage opportunities remain required.
The seed-2001 deterministic natural-P01 full evaluation command below has now
been launched from this checkpoint with no teacher prefix. Its real final result
is not yet known at this update. It must run to a real terminal or the 200 s task
limit; do not turn a diagnostic window cutoff into a physical failure.
The 1,024-step first check is not the task's total budget or a stopping condition.
After evaluation, continue predecessor and later-stage training from the actual
latest checkpoint, retaining natural P01 learning and validation.

**Latest completed boundary (06:16 UTC):** the P06 block stopped normally at a
verified update boundary, with **128** actual credited decisions, **1** PPO update,
**20** optimizer steps; 896 of its originally requested decisions were not used.
The latest checkpoint is `checkpoints/history/checkpoint_step_000141184.pt`, SHA256
`33083ddb33dc5f7e1cb1092bc25787d9a18c5fd67ef5b49043a6337d8175b7b0`.
Round-trip verification passed; lifetime **141,184 / 1,068 / 21,360**.
New-version saved learning totals are **640 / 5 / 100**.
P06-block phase credit is P06=112, P07=12, P08=1, P09=3, all others=0.
It had one BODY_COLLISION (P09) and two FALL terminals (P07, P06), no success.
The last three samples were a nonterminal P06 tail, not another completed episode.

A same-version, ordinary-checkpoint-resume natural P01 training block of 384
decisions was launched next. No new-MDP flag, weight reset, or production change
is used. If it completes, the first fixed-version check will reach 1,024 saved
decisions; use actual manifests rather than assuming completion. Then immediately
perform the natural P01 deterministic reload evaluation below, followed by further
training according to actual unfinished tasks. No resource boundary is claimed.

Snapshot: 2026-09-10 05:54 UTC (later pointer/manifest takes precedence).
The completed natural-P01 training block
`20260910T0530598363441Z_g28609010db4e_4c4cee3253cf41af9c521d5cb576ccbd`
added **512 saved decisions / 4 PPO updates / 80 optimizer steps**.
Its immutable checkpoint is
`checkpoints/history/checkpoint_step_000141056.pt`, SHA256
`b8079ebad977391395b762cce38fd6d187d8a11206960d61a5934eef0e06f7cc`.
The saved manifest reports round-trip verification and lifetime counters
**141,056 / 1,067 / 21,340**; adaptive optimizer LR is `1.5e-5` at this checkpoint.
The completed episodes fell at P06 (17.9583 s) and P02 (5.45 s); a final P05
episode was unfinished at the collection boundary. None is full success.

The next run
`20260910T0542011958077Z_g28609010db4e_b88e6f85c2e1480ea33d2fbcf28d6139`
is collecting 1,024 P06-predecessor decisions from that verified checkpoint.
Its first 82 credited samples reached P09, recorded RR initial and established
lift (five current-valid decision-end samples), then terminated with verified
BODY_COLLISION at 35.3333 s, before RR crossing or placement. These 82 samples
were persisted before the next teacher reset but were not yet a complete update
at this snapshot; do not count them as saved learning. Teacher-prefix steps are
not policy samples. No deterministic natural-P01 reload evaluation has completed
under this revision yet. The planned evaluation below must still run.

The live process must finish or stop safely before any production revision.
The deferred runtime-identity drafts under `deferred_runtime_identity/` are
not applied, not tested, and must not be added to PYTHONPATH.

At 06:02 UTC the P06 run had 102 logged decisions, no complete new update,
and two real terminals: P09 BODY_COLLISION after 82 decisions, then P07 FALL
after 20. A pinned `stop_after_update.request.json` was added to this exact run
to save and stop at its next complete update (not immediately, no kill).
The 1,024-decision request will therefore not all be consumed by this run.
After its verified completion, use the actual latest pointer and collect a
natural P01 block sufficient to bring this revision to at least 1,024 new saved
decisions, then run the deterministic P01 evaluation below. Continue learning
with P06/P07 predecessor and later-stage opportunities afterward. This is a
curriculum allocation decision, not a task stop or a fabricated resource limit.

Before starting, verify no project Isaac process is already running and check
`checkpoints/checkpoint_last_pointer.json` in this directory. When it exists,
its immutable checkpoint and manifest, not this initial source number, are authoritative.
Do not run a second Isaac process or hot-edit production during collection.

Initial migration command (only before a new-namespace checkpoint exists):

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 28609010db4e57c5b34304a4ae2563c69f9d00b9 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage full_episode -Decisions 512 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -NewMdpWarmStart -Checkpoint outputs/ppo_all_stage_acceptance_v1/checkpoints/history/checkpoint_step_000140544.pt
```

After the first verified update, continue from the new namespace (no migration flag):

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command train -ExpectedHead 28609010db4e57c5b34304a4ae2563c69f9d00b9 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Stage phase_suffix -FromPhase P06 -Decisions 1024 -Seed 1001 -NumEnvs 1 -Device cuda:0 -CheckpointIntervalUpdates 1 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

Natural P01 deterministic evaluation, with normalizer/optimizer updates disabled:

```powershell
& .\scripts\run_semantic_ppo.ps1 -Command eval -ExpectedHead 28609010db4e57c5b34304a4ae2563c69f9d00b9 -SemanticVersion v3 -ExperimentId fsm_reference_p09_stable_v2 -Mode semantic_residual_eval -MaxDecisions 3000 -Seed 2001 -NumEnvs 1 -Device cuda:0 -Checkpoint outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/checkpoint_last.pt
```

Teacher-prefix physics and zero-residual checks are excluded from on-policy counts.
Latest is not best, and local progress is not full-task success. No full-success
checkpoint or success video is claimed by this recovery document.

# Height and P02 recovery — completed-run handoff

Latest sealed checkpoint: **174592 policy decisions /1329 PPO updates /26580 optimizer steps**. This round, from168192, has actually completed **6400 /50 /1000**. The unchanged2048 continuation and immediate official C174592 evaluation are both finalized. All physical processes have ended per root; no active training, unexecuted budget, resource-exhaustion claim, or new production change is implied.

The zero-residual run remains a verified physical **full-task success** at8857ticks/P13. The latest formal all12 PPO still fails at812ticks/6.766667s/P02. It has not recovered FR crossing/placement, completed full traversal, or demonstrated stability superiority.

## Current recovery identity and safety boundary

- HEAD `4a58c0190ef744a2e40c3668d9042caf85389385`.
- Runtime `169a0415974a16a762f753fe3b3d6dcd702a783aebf184f4a9ab551a6052d0a9`.
- Latest [checkpoint_step_000174592.pt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000174592.pt), [immutable manifest](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/checkpoint_step_000174592_manifest.json). Recorded SHA `1da3e970d4be999efe5d26a9bb09f52126ea85bbafdab0ad59b9bb6dcecf1881`; official save/load roundtrip=true and formal reload verified.
- Current policy `history_conditioned_heteroscedastic_log_temperature_v1`, τ=.5 conditional innovation scale, HISTORY rho=.9. Final effective Adam LR1e-5; do not replace it with runner default LR.
- 372 observations/all12 actions,120Hz physics/15Hz decisions, physical model/hard limits/evaluator/reward/residual caps unchanged. Ordinary phase changes are not terminals.
- Any later same-version continuation must use actual latest174592 (or a newer valid sealed checkpoint), not old172544/171520 migration. Preserve compatible actor/critic/Adam/normalizer/RNG/counters, fresh rollout, legal P01 reset. No saved PhysX-contact state or tick-exact physical continuation is claimed.

Completed continuation run: `train/20260915T0633266376940Z_g4a58c0190ef7_d19a81607e6044a281a010de29469635`. [Decision receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/continuation_after_wheel_audit_172544.json) now records completion and the failed C result. [next_fixed_training_after_172544_notes.md](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/next_fixed_training_after_172544_notes.md) is the historical pre-run rationale, not a pending command. No further run is represented as started here.

## Actual training, not task-success credit

| Completed block | Source → final checkpoint | Added decisions / PPO / optimizer | Evidence |
| --- | --- | --- | --- |
| a9c natural P01 |168192→170240|2048 /16 /320|[receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_P01_2048_actual.json)|
| eec6 four separate256 legal-reset P01 blocks |170240→171264|1024 /8 /160|[receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_P01_four256_actual.json)|
| eec6 P06 predecessor maintenance |171264→171520|256 /2 /40|[receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_P06_256_actual.json)|
|4a58 τ=.5 natural P01 |171520→172544|1024 /8 /160|[receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_P01_temperature1024_actual.json)|
|4a58 unchanged natural P01 |172544→174592|2048 /16 /320|[receipt](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_P01_fixed2048_after_wheel_audit_actual.json)| 

The four256 runs have separate legal resets, the same seed and a sequentially updated checkpoint chain; they are not statistically independent trials.

This round’s actual requested-phase samples: **P01=43/P02=2456/P03=113/P04=142/P05=3170/P06=401/P07=3/P08=3/P09=69/P10–P13=0**, sum6400. The maintenance teacher’s1344 decisions/10752ticks remain excluded. Neither the latest2048 nor prior temperature1024 added P06+ samples.

Latest2048 alone: P01=10/P02=726/P03=35/P04=59/P05=1218,16372 real physics ticks, teacher0.16 complete updates changed actor with finite nonzero gradients; normalizer unchanged, official chain/roundtrip verified. **Actual LR range1e-5..3.375e-5, final1e-5**, not “always1e-5.” [Actual distribution/mask audit](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_fixed2048_distribution.json) verifies2048/2048 all12 masks and16372/16372 native ticks; τ only.5. Target effect is not measured torque or proof of good motion.

Five natural entries all have FR C/P history, but four completed episodes failed **P05 FALL×2/BLOCKED×2**; the373-decision P05 tail is nonterminal, not a fifth full-task success. Critically, latest P02 samples=726 but **P02 terminals=0**. Do not claim this block observed the formal mean’s P02 hard-limit termination or that all sampled states were proven near it. [Credit audit](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/training_fixed2048_reward_credit.md) verifies20 ordinary phase switches nonterminal/bootstrap and four actual P05 terminals with reward=return/no bootstrap; short terminal intervals7/8/4/1ticks remain real.

The earlier temperature1024 did contain one P02 hard-limit terminal in completed update1307, plus one P05 blocked and a249-decision nonterminal tail. Recorded conditional first mean from CP172544 matched its formal first raw exactly, while sampled raw differed; this does not prove all372 inputs or later histories identical. No GAE, HISTORY, sigma or sole-wheel root cause is established.

## Full zero success and newest formal PPO outcome

**Zero4a58 full physical success is unchanged.** Run `prior_B/20260915T0525295619176Z_g4a58c0190ef7_615f8fe3cfc64cff81f73ac6a0d302e6`, seed4001, naturalP01,8857ticks/73.808333s/P13, task=true, POST_COMPLETION_OBSERVATION. Fixed1s complete=true/loss=false, region/support/current-control true. RR I5428/Q5434/C6155/P6155; RL Q6251/C6658/P6727. It has zero policy/optimizer credit and does not use temperature.

The final-stop owner enters observationpre8737; first completed decision showing it8744; next held dispatchpost8738. N8737–8857 held, actual native wheel targets0, old8745 recoverypulse absent. Source clocks and normalmapper/feedback continue; actualq is not frozen. [B_HEIGHT_FINAL_STOP.aggregate.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/B_HEIGHT_FINAL_STOP.aggregate.json) and [P13_final_stop_owner_success.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/P13_final_stop_owner_success.json).

True stopped terminal: bodycollidermin172.999229mm, USD RRmount241.153101mm, RRwheel **link-origin**100.192405mm/collidermin49.940715mm, kneeactual−27.371109715°/lowerheadroom32.628890285°. Wheel geometric center unknown/null. Do not use olda9c moving heights.

**Formal C174592 remains a P02 safety abort.** Run `validation/20260915T0710472104000Z_g4a58c0190ef7_253ae24ad12c4830ad3e3ae5316ac659`, same runtime/seed4001, official reloaded174592, naturalP01/all12 deterministic/no-mask/no-teacher/0updates.812ticks/6.766667s, HARD_JOINT_LIMIT rear_right_knee; requested−4.246801342° vsactual−60.022796599°. FR Q24 but C/Pfalse; first unfinished task is FR front-edge crossing. FRgap−7.551mm/front−171.007mm; bodycollision=false. [Versioned formal evidence](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/p02_policy_recovery_checkpoint174592.json) and [Latest canonical evidence](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/p02_policy_recovery_evidence.json) agree exactly. Wrapper exit1 is valid physical failure, not a crash.

## Playable media, real-time wheel data and fair quality

All listed videos pass full decode/PTS, exact frameledger/runtime/checkpoint binding,15fps and≤200s. Original footage and every older C remain. First encoded frame is actualtick8, not fabricatedtick0. The successful zero’s fixed camera leaves part of the robot outside the view late in the run; this visual coverage limit is retained, without rerunning the camera or reversing its logged physical success.

- [Zero full success](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/zero_RLminus3_final_stop_latest.mp4) —1108real frames.
- [Newest PPO full failed evaluation](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/PPO_checkpoint174592_full12.mp4) —102real frames, complete812tick endpoint.
- [Strict same-version pair](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/zero_RLminus3_vs_PPO_checkpoint174592.mp4) —C’s1006 extra display frames say RUN ENDED, not physical data.
- [Zero P01–P03 wheel evidence](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/zero_RLminus3_P01_P03_wheel_qd.mp4) —188frames/tick8–1504, continuous excerpt of the separately retained full success.
- [Newest PPO complete wheel evidence](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/PPO_checkpoint174592_full12_wheel_qd.mp4) —102frames/tick8–812, failure retained.
- [Historical masked diagnostic](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/videos/p02_wheels4_mask_diagnostic_16s_wheel_qd.mp4) —240frames/16s, distinct diagnostic family, not formalPPO.

ASCII FL/FR/RL/RR show native joint rotation rad/s, actual dispatched target and actual `robot.data.joint_vel`, exact same frame tick. Canonical→native signs[-1,+1,-1,+1]; physical wheel velocity fields themselves are canonical. Rotation is not wheel-body translation. Same-name .wheel_data.json retains joint/prim binding and source log lines; no interpolation or15Hz contact forward-fill. Root inspected the new full-video endpoints and panel101; all fields are clear. Old C172544’s independent pixel audit remains historical, not a newly rerun C174592 pixel test.

[Latest matched quality](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/B_FINAL_STOP_C174592_matched_height_quality.json) uses real inclusive0..812ticks/813 observations each, independent height samplesB102/C103, not an invented B812 sample. FullB0..8857/C0..812 windows stay separate. Shared roll RMS B/C=.227216/.116905rad, pitch=.179952/.117051rad; C’s more horizontal body still collapses: collider minimum63.278mm and USDmount122.578mm versus B’s sampled91.411/201.497mm. No stability superiority follows. Frozen video frames never enter statistics.

## Latest four-wheel control finding

[Latest audit](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/four_wheel_P02_checkpoint174592.md) confirms equal actual initial q/qd/base/CoM/contact/geometry and sourceN Full12 across all812 sharedticks with successfulzero (also723 with C172544). At tick400, canonical N=.3 all4, projected=[−.503215,−.278786,−.262423,+.490672], final=[−.203215,+.021214,+.037577,+.790672]; directnativeqd=[+.014348,+.021147,−.053483,+.517062]. FR AIR has no bearing; otherthree realground.

All812 composition/mapping/setter/counterfactual checks pass,103 direct nativeqd sign checks have0error. N/mappedwheelN equal, wheelbias0, projection error≤1.11e−16/nativefloat32 error≤2.92e−8. FL final is reversed756 of772 positiveN ticks; FR nearstop644/RL620; RR enhanced772. .05 is only a descriptive threshold. This is policy cancellation and loaded response, not N omission, and ACK/all12enablement does not prove coordination. Unique per-tick source owner is unknown in current emptylayers log; do not assign all+.3 to one source.

RR5° sustained-error onset moves63→159 vsC172544 and tick400actual improves−28.388→−22.691°, but1° onset is27→26 and final hardlimit still occurs. These local changes are not FR/P02 recovery. No forced same-wheel-speed, permanentmask or new reward/physics change is introduced.139CPU wheel/adjacent tests (19new) remain wiring evidence, not success.

## Preserved historical failures and causality limits

OldA/frozenFSM/raw conformance and existing physical re-adjudication remain distinct; neither history nor filenames changes this run’s result. OldB2 P09 final target−60°/actual−60.000011° was a nominal-geometry issue; oldC168192 P02 target≈−5.434°/actual≈−60.017° is tracking failure, not a−60° policy request.

FL−4+newgeometry zero ended8844/P09 blocked, RR Q attempts5442/5764 were revoked by real contact; no C/P despite31.107° knee headroom. RL−3+late-entry a9c zero reached all-leg placement but failed fixed P13 endpoint: oldstop8737→recoverypulse8745→end8857 with controlled=false, region/support true and post_completion_loss_observed=false. That old failure is not overwritten by new4a success. RR C/P6155 precedes late-group post6156, so do not attribute capture to that later dispatch. Height amplitudes and geometry/entry timing differ across candidates; they are not isolated single-factor causal experiments.

Historical C170240/171264/171520/172544 ended P02 at752/751/693/723ticks respectively; all original full videos and formal JSON are retained. The16s wheels4 mask at frozenCP168192 reached P05 after FR C1672/P1686, but changes closed-loop HISTORY/servo paths and has0 training credit. It is not an all12 success or unique wheel-cause proof.

[Historical C172544 wheel audit](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/four_wheel_P02_checkpoint172544.md) additionally verifies the same actual initial state and all12 N across the shared723ticks with successful zero. At tick400 canonical final=[−.224233,+.013801,+.040672,+.798092]; direct measured native qd=[+.045131,+.012796,−.044501,+.691982]. FR AIR has zero bearing; FL/RL/RR have real GROUND loads. All723 projection/native joins and92 direct native-qd sign checks pass, but this is not reasonable-coordination proof. That C172544 terminal FR gap remains+.320mm/front−181.033mm with no C/P; do not substitute old C’s negative gap. [Historical C171520 audit](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/four_wheel_P02_path.md) stays separately labeled.139 CPU wheel/adjacent tests passed (19 new, not summed with overlapping suites); no permanent mask or fixed-four-wheel-speed requirement was introduced.

## Implementation and artifact boundaries

Production changes are shared nominal height/feasibility/source scheduling, read-only height/advantage diagnostics, P13 final-stop owner and versioned exploration temperature, not a new training project. Exact historical migration receipts retain compatible weights, full Adam, normalizer, RNG and counters, with fresh rollout. CPU groups232/157/125/28, a9c92, eec owner26/migration36/adjacent266, and temperature384 pass/1explicit deselect/0skips overlap; never sum them as distinct tests or physical samples.

Parent authored [31×80 comparison CSV](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/height_transfer_candidate_comparison.csv) from [v2 measured event payload](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/height_event_comparison_final.payload.json), with typed/roundtrip/preview QA reported passed. No unfinished-run values were fabricated. Current source and counters are in [latest_run_results.json](C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1/outputs/ppo_height_and_p02_recovery_v1/latest_run_results.json); research’s nine-source table and all immutable historical manifests/videos are retained.

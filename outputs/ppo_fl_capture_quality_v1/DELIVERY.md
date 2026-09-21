# FL capture/front quality: latest saved-checkpoint evidence

Latest completed deterministic video is CP183552; the latest completed
same-checkpoint deterministic/stochastic pair remainsCP182528, below.
The natural-P01 block11 has now sealed:1024 decisions/8updates, reaching
CP183552/update1399. This branch has5120 real optimized decisions/40updates/
800optimizersteps. CP183552 natural-P01 deterministic video is now sealed
P05 incomplete. Block12 (2048 requested P05 checkpoint_policy/offset200) has
stopped safely after768 actual decisions/6updates/120steps: CP184320/update1405,
branch5888/46/920. The unconsumed1280 are not credited. Two terminated episodes
capturedFL then collided atP09; no fulltask success. LatesttrainingCP is not yet
an evaluated model. A single-channel physical-innovation version is being
integrated only after this complete-update save; successfulN stays unchanged.
The explicit sigma version is now committed at e73542c and realtraining started
fromCP184320:1024 P05 checkpoint-policy/offset200 decisions requested, not yet
credited. Only P06+ FRknee innovation sigma changes; no deterministic scaling,
newN primitive, actor reset, action-cap reduction or acceptance relaxation.
Earlier CP180480 both modes and B_control remain below.
No complete PPO success is claimed. Accepted N_ref and successful B are intact.

Block11 actual phase counts P01..P09 are4/392/7/2/307/268/1/1/42.
Both natural episodes truly capturedFL (ticks2848 and2847); the first later
endedP09 BODY_COLLISION at5283/44.025s, while the second reachedbudgetP06/tick2904
withFL againAIR/7.949mm/0N. See block11_1024_receipt.json. P01/P02 body-family
cost actually optimized was−.036705387638, not just a nonzero configuration.
The narrow block11_capture_credit.md audit confirms correct reward/action/
storage alignment and actualminibatch use, but not mean-policy capture learning:
joint-action probability improvement does not establish FLhip mean improvement.

## Newest deterministic video: CP183552 (09:12 UTC)

[Complete natural-P01 evaluation, P05 incomplete](C_CP183552_deterministic_review/ppo_latest_full_episode_incomplete.mp4)

Saved/reloadedCP183552, full12, no intervention, same3a/scene4001, optimizer0.
FR placed1594/13.283333s, FL crossed2588 but neverplaced; actualterminal
5796/48.3s, INCOMPLETE_CONTROLLER_BLOCKED. FL AIR/25.992981mm/0N.
725frames/15fps,48.3333media seconds,1280x870,full-decode valid/continuous,
no black frames; root inspectedfirst/last andpublishedtheMP4 immediately.
Fullstall/failuretail and actualfour-wheel jointvelocities are retained.
This is not deterministiccapture improvement or fullsuccess.

PredeclaredFR event-window RMS=.107908446278rad/s,tiltpeak=.293049357523rad,
duration13.283333s. Against realB .117889399212/.309464301689/12.516667s:
RMS−8.466%,tiltpeak−5.304%,time+0.766667s/+6.125%. Bwasnot rerunat3a;
the existing f2e→3a unchanged-N/controller/config/scene-path disclosure applies,
not a sameHEAD/identicaltrajectory claim. NewexplicitcrossHEADpair is now
exported separately; oldstrictpair rule and sourcereceipts remainunchanged.

Publishednewoutputs (rootinspectedfullpairfirst/lastpreviews):

- [P05 continuous incomplete excerpt](C_CP183552_P05_clip/ppo_P05_capture_after_training_incomplete.mp4):525frames/35s,ticks1608–5796, noFLplaced/P06.
- [Explicit cross-HEAD full comparison](CP183552_cross_HEAD_unchanged_N_pair/cross_HEAD_unchanged_N_full_incomplete.mp4):1108frames/73.8667s,C383terminalfreeze frames clearlylabelled/no metriccredit.
- [FR event-window comparison](CP183552_cross_HEAD_unchanged_N_pair/cross_HEAD_unchanged_N_P01_to_FR_capture.mp4):200frames/13.3333s,B12freeze frames; no phasewarp.

Allnormal15fps/full-decodevalid/continuous/zero black frames. Bnotrerun and
notsamebuild disclosures retained in cross_HEAD_pair_receipt.json andeverypanel.

Auxiliary duration check (not a newpreselected primary metric or task-progress
alignment): for normalizedclock u=t/T, RMS(dtheta/du)=T*RMS(dtheta/dt).
Those derived values areB1.47558231346 andC1.43338386139, a−2.859783% difference.
Thus slower completion explains part ofthe rawrate difference; this arithmetic
does not establish speed-independent causal benefit or statisticalrobustness.
Tiltpeak, realFRcapture and actualduration remain separatelyreported.
MissingFLcapture windows are null, notzero. CP183552 currentlydet-only;
do not pairitwithCP182528stochasticas a sampling-mode-onlycomparison.

Incremental CP183552_P05_diagnosis.md/json confirms same-tickFL REQUEST remains
effective intheselectedsamples (terminalhip+4.309496deg,knee−1.609089deg),
with actualtracking and no demonstrated newmask/overwrite. Itdoesnotisolate
onejointas thesolecause. Timeout48.3s comes from gap>25mm reducingtop_geometry
progressfrom.85to.70, hence30+10*.70²=34.9s afterP05entry13.4s; front>.15m
isnotitsdirectcause. Placementstillrequiresrealcontact; AIRwasnotrelabeled.

## Latest playable CP182528 (2026-09-18 08:14 UTC)

[Deterministic full episode — P05 incomplete, 50.30 physical seconds](C_CP182528_deterministic_review/ppo_latest_full_episode_incomplete.mp4)

- [P05 continuous excerpt, stall and terminal retained](C_CP182528_P05_clip/ppo_P05_capture_after_training_incomplete.mp4)
- [Explicit cross-HEAD unchanged-N full comparison](CP182528_cross_HEAD_unchanged_N_pair/cross_HEAD_unchanged_N_full_incomplete.mp4)
- [FR event-window comparison with duration disclosed](CP182528_cross_HEAD_unchanged_N_pair/cross_HEAD_unchanged_N_P01_to_FR_capture.mp4)

Full and event comparisons are15fps/1x, full-decode valid. Root inspected the
full-pair first/last previews: both HEAD banners and terminal freeze labels are
visible. The independent cross_HEAD_pair_receipt.json binds real source records;
it neither alters old source receipts nor relaxes the original strict pair rule.

Saved-and-reloaded naturalP01, full12, no intervention, optimizer_updates0,
same3a HEAD and scene4001. FR placed1556/12.966667s; FLcross2586 but neverplaced.
Terminal6036/50.3s: INCOMPLETE_CONTROLLER_BLOCKED atP05, FL AIR/19.282771mm/0N.
The writer sealed the actual failure. Full decode755frames/15fps, continuous
timestamps, no black frames, original whole-body view and full stall tail;
root inspected first/last previews and published the MP4 before stochastic ended.

FR event-window Euler-rate RMS=.10970819665rad/s and tiltpeak=.30155585155rad,
duration12.966667s. Existing realB RMS=.11788939921,peak=.30946430169,
duration12.516667s: descriptive differences about−6.9%/−2.6%/+0.45s.
This is a single front-window observation, not full-task superiority,
statistical robustness, or speed-independent causal proof. B was recorded at
f2e, C at3a; six configs and N/control/physics paths are unchanged by the
learning-kernel-only delta, but B was not rerun at3a and it is NOT a same-HEAD
pair. The separate explicit cross-HEAD comparison preserves that disclosure;
the original strict-pair rule and old receipts are not altered.
Source: CP182528_deterministic_quality.json and the sealed review receipt.

### Same-checkpoint stochastic result, now sealed (08:35 UTC)

[Stochastic full episode — P09 low-height safety stop](C_CP182528_stochastic_review/ppo_smooth_exploration_stochastic_labeled_incomplete.mp4)

[Explicitly stochastic FL capture and continuous P06 excerpt](C_CP182528_stochastic_P05_capture_labeled/ppo_P05_capture_stochastic_after_training.mp4)

Same immutableCP182528/actorhash/HEAD3a/sixconfigs/scene4001/camera verified in
CP182528_policy_modes_verified.json; policyseed4101, no intervention/full12.
FR placed1564/13.033333s; FL placed2761/23.008333s and P06 started2768.
P07/08/09 began4888/4896/4904. RR qualified5167 then revoked onground5394;
qualified again6335 then revoked6381. RR never crossed or placed. Terminal
6699/55.825s P09 SAFETY_ABORT: base_linkz .0148815438m < unchanged.015m
low-height FALL limit. It is not a physical explosion or complete task success.

Fullvideo838frames/15fps,55.8667media seconds,full-decode valid, no black frames,
continuous timestamps; root inspectedfirst/last and publishedbefore further
training completed. The27.6667s FL excerpt is contiguousticks1576–4888, retains
the wholeP06 segment and does not include/misrepresent laterP09 failure as success.
FL is AIR again at excerptend despite historyplaced=true; no continuing-bearing
claim. Fullvideo retains the actual safety tail.

Stochastic FR RMS=.11356750413rad/s,peak=.30926054009rad,duration13.033333s.
FL lift→capture9.508333s,RMS=.06849677444,peak=.10697096291;
cross→capture2.316667s,RMS=.06727121123,peak=.05299332355.
Do not use this stochasticcapture to claim deterministicP05 solved. Current
deterministic first unfinished task remainsFL capture; stochastic first
unfinished downstream task isRR carry/cross/place.

Confirmed P05 diagnosis and measured last6s15Hz mapper/final/actual-q spectrum
are in CP182528_P05_diagnosis.md and CP182528_last6s_spectrum.json. Policymean
had only slow small drift in its native15Hz samples; no CAPS/filter/temp change
was justified by that interval. No new physical execution bug was established.

## Latest saved training boundary (2026-09-18 07:52 UTC)

First4096 new decisions are now fully optimized:32PPOupdates/640optimizersteps.
Actual latestCP182528/update1391/27820steps, save_load_round_trip=true.
Checkpoint SHA81f392a726b425a0bd8e004db11513a4acdb5e2413aec676f8a1b5b835ebc47e;
manifest SHA3e61d497dd8e44207fc8caa8d9d60539b025df3db4d15ac34231d81242d06e6a.
Block09 added128P06. Block10 addedP10/P11/P12=1/1/126, no learner P13,
and ended at budget inP12 withoutRL capture; RR had returnedGROUND despite
historicalplacement. No full-task success and no newly proven best checkpoint.
Natural-P01 deterministic video evaluation of this savedCP is now running,
same3a HEAD,seed4001,full12,no intervention. Stochastic sameCP follows.
First4096 is an evaluation checkpoint, not a declared convergence or stop budget.

### Earlier saved boundary (07:28 UTC)

At07:28, CP182272/update1389 was the latest completed checkpoint, not the older
CP180480 used in the videos below. This turn has now added3840 real learner
decisions,30updates,600optimizersteps. Block07 added768 with P05/P06/P07/P08/P09
counts175/518/2/2/71 and three real FL captures; two later P09 BODY_COLLISION
terminations, one P06 budget end. See block07_summary.md; historic placement
does not establish continuing support or full-task success.

Block08 naturally completed256 decisions,2updates from naturalP01, actual
P01/P02/P03/P04/P05 counts2/210/4/1/39. FR placed tick1721, no terminal before
the complete update boundary. Actual body quality contribution was
−.0145646431, including P02 FUNCTIONAL_CARRY988 physics samples with beta.015/s;
quality did not vanish during transfer or carry. See block08_256_receipt.json.
Next real N-prefix maintenance P06/128 then P10/128, followed by latest saved
checkpoint natural-P01 deterministic and explicitly stochastic video evaluation.
No new fixed-checkpoint capability claim precedes those evaluations.

## Playable, saved-and-reloaded CP180480

- [Deterministic full episode, 48.4 s](C_CP180480_deterministic_review_contact_labels/ppo_latest_full_episode_incomplete.mp4)
- [Continuous P05 excerpt, 35 s](C_CP180480_P05_clip_contact_labels/ppo_P05_capture_after_training_incomplete.mp4)
- [Same-CP stochastic full episode, 28.07 s](C_CP180480_stochastic_review/ppo_smooth_exploration_stochastic_labeled_incomplete.mp4)
- [Stochastic FL capture and P06 continuation excerpt, 13.93 s](C_CP180480_stochastic_P05_capture_labeled/ppo_P05_capture_stochastic_after_training.mp4)
- [Same-version N+0, complete success, 73.808 s](B_control_review/B_control_Nplus0.mp4)
- [Full normal-speed N+0 vs deterministic PPO](CP180480_B_control_pair/nominal_vs_ppo_latest_incomplete.mp4)
- [P01 to real FR placement, normal-speed pair](CP180480_B_control_pair/ppo_P01_P02_stability_vs_nominal.mp4)

Both policy modes are normal 1x/15fps, single-episode native video with actual four-wheel
angular velocities, FL gap/contact/bearing and placement history. Full decode,
continuous timestamps and first/last frame inspection passed. The deterministic excerpt
retains the entire P05 stall and failure tail; no frames were spliced from another
run. The first draft annotation used a nonexistent contact_mode key and showed
N/A; the contact_labels copies above read actual air/contact_surface instead.
Only the output renderer changed; source footage, task result and production
remain identical. Draft media are preserved, not silently overwritten.

Formal C source:
`runs/ppo_fl_capture_quality_v1/video_eval/validation/20260918T0524353774569Z_gf2e552406ea7_0d8dd902152a4ad68a84118544c00b3c`.
Runtime f2e552406ea74a5aae2803347d1c8bebe261910d; natural P01, full12,
deterministic conditional mean, seed4001, no teacher or diagnostic intervention.

FR was actually placed at tick1603 / 13.358333 s. FL crossed at2439 but never
placed. Final tick5804 / 48.366667 s: INCOMPLETE_CONTROLLER_BLOCKED at P05,
FL AIR, 25.728854 mm gap, 0 N bearing. This is genuine task noncompletion, not
media corruption. The launcher reports DIAGNOSTIC_FAILURE for that outcome;
the native writer and source manifest sealed successfully.

Predeclared complete FR window: body roll/pitch Euler-rate RMS .1094489165 rad/s,
peak roll/pitch norm .3016974385 rad, duration13.358333s. Missing FL windows are
null. The same-version comparison below supplies a limited single-pair observation.

## Actual training, not requested curriculum counts

The separately labeled stochastic episode used policyseed4101 with the same
immutable CP180480, scene4001, runtime, camera and configuration (verified in
CP180480_policy_modes_verified.json). FR placed1568; FL placed2995/24.958333s,
then P06 started3000/25s. It terminated3368/28.066667s at P09 on PHYSICAL_SAFETY:
saved base_link z=.014471285m is below the existing .015m FALL boundary;
projected gravity z=-.935036 and linear/angular speeds .159809/.293707 do not
support the explosion branch. RR had lift3234 but no crossing/placement.
This is real full12 stochastic capture, not deterministic capability or full
task success. The short stochastic clip covers P05 and P06 through tick3240;
the subsequent P09 safety-stop tail remains in the linked full video.

Stochastic FR-window RMS=.119050597rad/s, tiltpeak=.314247446rad,
duration13.066667s. FL lift-to-capture11.508333s and cross-to-capture5.966667s.
Do not substitute these stochastic metrics for deterministic quality claims.

## First check block counters and evaluated checkpoint

Origin CP178432/update1359; first evaluated CP180480/update1375. Added2048 actual
learner decisions,16 PPO updates,320 optimizer steps. Actual P01–P09 counts:
6,488,9,18,988,226,6,13,294; P10–P13 each0. P03–P06=1241/2048 (60.596%).
True prefixes, physical probes and CPU synthetic updates receive no credit.
See [verified training summary](first2048_training_summary.md).

Checkpoint used in all C videos above:
`checkpoints/history/checkpoint_step_000180480.pt`
SHA256 ec48951dbc345d196c7ffa14fc6fa7aba7c33723292c11e8d500ef640ca2b05a.
Save/reload and actual optimizer participation verified. It is not a newly
proven best checkpoint: deterministic P05 capture remains unsolved.

Accepted N_ref remains untouched at73.808333s SUCCESS, including original
source/config/video. The isolated request-history change was applied only after
these three runs sealed; it is not part of CP180480 or any video above.
See RECOVERY.md for the active process and exact continuation state.

## What the next isolated change can and cannot fix

The sealed stochastic handoff audit separates two effects. At P05→P06, FR knee
raw was still positive (.08195→.06389); changing cap24→112 alone contributed
+7.1959 degrees before filtering. Subsequently the learned base mean stayed
negative and the request reached −54.9219 degrees. The first-entry REQUEST
history candidate addresses only the range-coordinate inheritance, not that
later learned mean or the coupled P09 nominal/support changes. It is not a
P05-capture fix by itself. Its migration adds no physical training credit.
See CP180480_stochastic_handoff.md and NEXT2048_CURRICULUM.md. No policy reset,
reward/temperature change, fixed hip-down script or nominal switch is proposed.

## Same-version pair and continuation

B_control run `20260918T0555572376368Z_gf2e552406ea7_1894338b350241fcbbac4868c5f7f3fb`
naturally sealed SUCCESS:8857ticks/73.808333s, all legs placed and controlled stop.
Strict pair receipt verifies same f2e HEAD/runtime, six configs, scene seed4001
and camera. It does not claim bitwise identical initial measured states.

| FR preparation→real placement | N+0 | deterministic CP180480 |
|---|---:|---:|
| Roll/pitch Euler-rate RMS (rad/s) | .1178893992 | .1094489165 |
| Peak roll/pitch norm (rad) | .3094643017 | .3016974385 |
| Duration (s) | 12.516667 | 13.358333 |

RMS is about7.16% lower and peak about2.51% lower, but duration is .841667s
longer (+6.72%). This is one observed front-window improvement, not statistical
robustness, speed-independent causal proof, or complete-task superiority. The
deterministic policy still fails later FL capture; stochastic C has worse FR
RMS than B. Missing windows stay null. Added video freeze frames are explicitly
marked and excluded from physical metrics.

After all three runs ended, HEAD3a50657a96c9beca6178d7f549750f21025a9fc6
applied the six-file REQUEST-history change plus one portable unit test.
438 targeted production tests passed, no skips/failures. All six configuration
files and N/mapper/physical/evaluator/reward paths remain unchanged. The formal
migration plan is checkpoint180480_request_history_migration.json; it preserves
weights, Adam, Identity, RNG, actual LR1e−5 and counters with fresh rollout.
Block06 completed768 P04/checkpoint_policy decisions under the new kernel,
seed1001: P04=3/P05=394/P06=301/P07=2/P08=2/P09=66. Two actual FL captures
were followed by P09 FALL29.125s and BODY_COLLISION42.491667s. No full success.
Newest sealed checkpoint beforeblock07 is
`checkpoints/history/checkpoint_step_000181248.pt`, SHA
49617e25e4ec1bb4636e43f2790ccb81bfaa5f2ca91c50df2cfa2dc99e834027,
update1381/27620optimizersteps; this turn +2816dec/+22updates/+440steps.
All completed-block actual coverage: P01=6/P02=488/P03=9/P04=21/P05=1382/
P06=527/P07=8/P08=15/P09=360; P10–P13=0. See block06_768_receipt.json.
Active block07/session4210 requests768 P05/checkpoint_policy/offset150,
exact resume fromCP181248. Real prefixes remain uncredited. Later planned
front-quality and rear-maintenance blocks will use actual latest checkpoints.
See RECOVERY.md and subsequent receipts for current state, not old video names.

Mapper clarification from existing physical probes (no new intervention): the
tracking reference uses canonical measurement minus the previous filtered
REQUEST, so its error is N−q+previous_REQUEST, not an uncorrected nominal-only
error. Two ±1.5deg holds each supplied256ticks/64feedback samples, no additional
reference clipping; final-target mean difference3.017141deg and actual-q mean
difference3.012993deg for a3deg REQUEST difference. Opposite-sign transient
compensation exists, but does not by itself prove harmful cancellation; mean
compensation was only+.035/+.052deg. This excludes complete/persistent loss
in those probes, not every possible transient interaction. No new mapper bug
was established and the mapper was not changed. Code: semantic_tracking_reference.py
and the previous_servo_residual_deg path in semantic_residual_adapter.py.

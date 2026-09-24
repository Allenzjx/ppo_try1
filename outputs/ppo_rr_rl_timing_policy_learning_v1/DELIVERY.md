# Rear-policy learning delivery — work in progress

## Latest real delivery — CP225280

The frozen cooperative version49eb has actually added **2048 decisions /16 PPO updates /320 Adam steps, new AUX0**, followed by saved/reloaded deterministic naturalP01 evaluation. Full-state model225280/1725/34500 is preserved in the selected branch.422observations/Full12,120Hz/15Hz,LR1e-5, rear task assists OFF /original FL assist ON. See `CP225280_RESULT.md` for the compact physical/control table, code changes, true sample coverage, exactcheckpoint/hash and limitations.

Latest videos are in `video_review/CP225280_deterministic_cooperative_prep_v4_review_v2/`: full91.166667s physical episode/91.2s video, same-run rear preparation detail, and historicalN comparison. Versionv2 corrects the distinction between the last committed N baseline and post-step source hint, source-clock holding versus a public dependency flag, and RL ground+wall contact labels. The unreleasedv1 is retained as QA-superseded. No simulation/model/output outcome was changed by the caption correction.

**Actual result: INCOMPLETE.** FR placed23.008333s, FL placed43.041667s/P06 at43.066667s, RR qualified58.291667s/crossed66.491667s. RR never contacted/bore onTOP; terminalgap56.118904mm,0N. RL neverqualified/crossed/placed and reachesground+frontwall contact. First remaining task is RR capture. P09 late neverstarted in this episode, so its earlier known issued-target-after-loss limitation is not asserted as this episode's cause. FL policy reverse and insufficient physical preparation persist. This model is latest, not certified improved or stable-superior; historicalN success73.808333s remains unchanged and its ended comparison side is explicitlyfrozen.

Real cooperative prep-active samples total382 (P07suffix239, P10suffix143, natural0). Natural training had a shortRR lift and realfailure; its final update's finitevalue-loss spike300.089757 is reported, not hidden. Full role/phase counts, raw-likelihood checks, currentcontactversushistory and separateprefixcredit are in the sealed reports. No AUX or policy-learning credit is assigned to diagnostics, media export, migration or the originalFL controller.

## Current delivery — CP223616 cooperative preparation, 2026-09-23

Saved and independently reloaded CP223616, frozen control49eb23163a6e,422 observations/Full12, deterministic naturalP01. New cooperative-version learning at this checkpoint is384 real decisions/3 PPO updates/60 Adam steps, new AUX0; full selected-branch continuation3072/24/480. Full actor/critic/Adam/LR1e-5/Identity/HISTORY preserved. Formal rear task assists remain OFF; existing FL capture assist remains ON. The remaining P10=384 then naturalP01=1280 course is running separately; unsealed budgets are not yet earned counts.

New videos under `video_review/CP223616_deterministic_cooperative_prep_v4_review/`:

- `CP223616_DET_full_policy_rear_no_assist_INCOMPLETE.mp4`
- `CP223616_DET_RR_to_RL_preparation_REAR_OFF_FL_ON_INCOMPLETE.mp4`
- `N_vs_CP223616_DET_same_camera.mp4`

Actual physical85.108333s/tick10213,1277frames/display85.133334s at normal15fps; no sped-up or removed failure tail. FR placement and FL→P06 at41.333333s were retained. RR qualified52.191667s/crossed58.35s, but acquired no real TOP contact/load; terminal AIR gap54.774910mm/front100.124mm/0N. RL did not qualify/cross/place. First unfinished physical task: RR capture. Outcome `INCOMPLETE_CONTROLLER_BLOCKED`, not collision or recording corruption, and not full-task success or stability superiority. All3 videos passed full decode, PTS continuity and black-frame checks; root visually inspected full60s, terminal and comparison terminal. Historical N_ref retains73.808333s success and explicit post-end freeze, not a fresh same-version paired B.

The measured four-mount heights and fixed-start localCoM→FR projection are diagnostics, not support proof or posture targets. Final FL/FR knee actual−44.334/−17.060deg; final four-wheel targets inFL/FR/RL/RR order [−1.0205,+.0205,+.0558,−.0986]rad/s, measured[−1.1055,−.00233,+.05578,−.10007]. Persistent FL counterroll has **not** been corrected by these3 updates. See `CP223616_cooperative_RR_RL_window_audit.md/.json`, `cooperative_sealed384_summary.md/.json`, and `successful_N_RR_RL_physical_reference.md/.json`. The earlier stochastic P07 suffix's RR TOP/load and short RL lift are real but did not persist and are not this deterministic result.

## Preserved prior delivery — CP221696

Learned CP221696 was saved/reloaded and migrated to explicit422 P02 measured-progress revision `d7e97ee7b7e493d4f3ff34f9c8762f73550ad7bd`. Its videos and history below are preserved, not the latest weights or activity.

### Prior deterministic video evaluation

`video_review/CP221696_deterministic_p02_progress_v1_review/` contains the authoritative CP221696 natural-P01 evaluation, same-run RR-attempt detail and historical-N comparison:

- `CP221696_DET_full_policy_rear_no_assist_INCOMPLETE.mp4`
- `CP221696_DET_RR_attempt_to_actual_tail_INCOMPLETE.mp4`
- `N_vs_CP221696_DET_same_camera.mp4`

Actual physical duration was93.433333s through tick11212; the normal15fps artifact contains1402 continuous frames and displays93.466667s without trimming the failure tail. FL reached P06 at43.933333s. The episode ended in P09 as `INCOMPLETE_CONTROLLER_BLOCKED` / `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`, not a body or ground collision: RR was AIR at0N with gap+44.355306mm and front distance+100.079182mm, never acquired current TOP support or placement, and RL was not reached. Front FL assist was ON, rear task assist was OFF, and this migration added no PPO or AUX credit. Full, detail and pair passed full decode, continuous monotonic PTS and zero black-like frames; root visually checked the full and comparison endpoints. Historical N is explicitly the older frozen reference, not a fresh same-controller B.

## Previous completed deterministic video evaluation (CP221568)

`video_review/CP221568_deterministic_recapture_v2_review/` contains the previous saved/reloaded CP221568 natural-P01 attempt, same-run detail and historical-N comparison:

- `CP221568_DET_full_policy_rear_no_assist_INCOMPLETE.mp4`
- `CP221568_DET_RR_NOT_REACHED_P02_failure_detail_INCOMPLETE.mp4`
- `N_vs_CP221568_DET_same_camera.mp4`

Actual result: P02 incomplete at22.433333s/tick2692, `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`; FR front distance−20.345625mm/gap83.223920mm, no FR placement and no RR/RL task reach. This is task noncompletion, not media failure or a rear-placement test. Front FL assist was ON; rear task assist, rear geometry correction and rear wheel projection were OFF. The original capture callback supplied337 continuous frames (22.466667s display quantization); no failure tail was removed. Historical N remains the preserved73.808333s success and is explicitly labelled non-fresh. Ended sides are labelled frozen, not extra physical stability time. Full/detail/pair passed full decode, continuous monotonic PTS and zero black-like frames; root checked the full and comparison endpoints visually. The older CP222080 video and its distinct22.525s/P02 failure remain unchanged in their original directory.

## What was implemented

`semantic_supervisor.py` removes the isolated P05 age30 gate after the real source endpoint, permits same-air FL recross recovery, and keeps actual contact authoritative. The opt-in rear timing implementation gives RR capture policy control before the dependent P09 late FL/RL group. P12 continually checks present RR support for a fresh RL unload while allowing a legitimate ongoing RL swing to continue. Pending local clocks wait without resetting physical history or phase-terminal GAE.

Rear task assistance is OFF in common training/evaluation. Observed rear task state and source clocks extend410 inputs to419 with old network/Adam state retained and zero-initialized new columns. Local RR hip and FR/FL knee exploration gates use the same distribution in collection and likelihood evaluation. Full12 residual permission is supplemented by actual target/dispatch evidence; it is not the sole proof of control.

The recapture revision44219b4fdc4d reopens the observed RR task after loss of current support, and replaces full historical-placement retention at arbitrary AIR height with bounded current gap/contact retention. Current frozen45862675a18a inherits that behavior and fixes measured current RL swing evidence. Neither adds a joint-target teacher, physical capability, new contact bonus or fixed user-angle goal.

## Learning and model selection

The first frozen course really added1536 decisions /12 PPO updates /240 Adam steps, AUX0, producing CP222080. Its above deterministic front regression is preserved, not relabelled as rear success.

The current explicit `ancestor220544_recapture_v2` branch starts from the verified compatible CP220544 lineage. It does not inherit the other branch's1536 decisions/12 PPO updates or claim a larger historical counter. Its complete course added **1024 decisions /8 PPO updates /160 Adam steps**, producing CP221568 (221568/1696/33920), SHA256 `1827b5d59935b31e1e27cc7ae0c364dd0203a9d7f77a7d9e9502189bbd31c7d5`. Exact optimized phases: P01=2,P02=254,P03–P06=0,P07=1,P08=1,P09=510,P10=1,P11=1,P12=254,P13=0. Prefixes earn zero credit. All network/critic/Adam/LR/Identity/RNG state is retained; semantic migrations use fresh rollout data. Main latest pointer remains untouched. The above new video evaluates this exact learned checkpoint, not the initial ancestor.

P07 learner samples produced no RR TOP/placement despite actual negative hip responses. The P10 learner suffix began from N's real RR support, retained intermittent TOP bearing (34/256 endpoints), then lost support (222 AIR endpoints); RL did not cross or place. These are training outcomes, not natural-P01 student success. Whole-turn real work across both explicitly separate branches is2560 decisions/20 PPO/400 Adam; only1024/8/160 belongs to the selected CP221568 lineage increment.

No new AUX fitting has run. The bounded outputs-only zero-update ancestor query is complete: student wheel REQUEST means were only about0.001–0.0023rad/s more negative than the same419 ancestor on the selected real front inputs. This is not enough to explain the closed-loop regression or justify fitting. Teacher queries are not executed actions or success labels; actor hashes and CPU RNG were unchanged. P07/P09/P12 same-input protection checks passed; no actual P03–P06 learner rows were invented. Source selection, tests, migrations, prefixes, deterministic evaluation and video export add zero PPO samples.

## Frozen v3 continuation in progress

The sealed P10 audit proved that RL never received the current-lift/motion-continuation fields consumed by rear dependency (only RR produced them). Committed45862675a18a publishes measured same-attempt RL qualification, clears it on every actual ground return, preserves historical events, and exposes it through the existing RL observation slot. It keeps419 dimensions and inherits v2 recapture/retention. Directed control tests176 passed and migration/branch tests58 passed, all zero robot credit.

Official CUDA publication and fresh reload of `branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_rear_live_swing_CP221568_g45862675a18a.pt` passed (SHAc98c0fb806c2df822f18d59b6e2287254ade4702beafed8d67b8eef21c1ac10f). It retains the learned CP221568 full state/counters; the semantic migration adds zero updates and starts fresh rollout storage. A bounded existing-CLI course `live_swing512_g45862675a18a` has started: P07-prefix256, P10-prefix128, naturalP01128. These are planned, not yet credited. No v3 deterministic result is claimed. Production remains frozen during physics; the latest completed video above remains explicitly v2.

## Not yet achieved

Sustained learned RR capture, usable RR/FL support, natural-P01 learned RL placement, full no-rear-assist success and stability superiority are not established. The new branch must be saved/reloaded and evaluated with one frozen policy from P01. Exact run/pointer/counter recovery is in `RECOVERY.md` and the live course ledger; this document does not promise unattended continuation.

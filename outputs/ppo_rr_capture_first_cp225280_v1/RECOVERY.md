# RR capture first: independent CP225280 composite continuation

Latest instruction: protect the accepted P01-to-RR-crossing precursor, learn
actual RR capture/hold first. RL-specialized courses are not used in this route.
All previous branches, checkpoints, successful N and delivered videos remain.

## Frozen implementation boundary

- The six control configuration files are exact copies of accepted49eb files;
  current code selects retained v3 rear timing and v4 preparation. No owner
  projection, rear capture helper, geometry helper or forced-forward wheels.
- The original422 policy input is retained. Seventeen explicit absent-owner
  zeros adapt the zero-learning439 CP225280 publication, whose original422
  weights are checked tensor-by-tensor at initialization. Eight public local
  task fields make447. The frozen prior is inside each composite checkpoint.
- Local mean starts zero. Before current qualified AIR + original RR crossing
  + legal top XY, local actor and its random sampling are not executed. The
  frozen prior remains closed-loop. After activation, all12 channels learn;
  gate remains latched through contact/drop until actual episode reset.
- Raw heads combine before exactly one existing HISTORY. Only active Gaussian
  samples/log probabilities enter official RSL-RL PPO. New Adam owns only the
  local128x128 actor and independent128x128 critic. Prior Adam is historical.
- RR-only potential/real contact/verified bearing hold reward replaces the
  global reward for this explicitly local training task. Native frame reads
  update contact hold; gamma=.9985 applies once per15Hz decision, lambda=.99.
  A local RR hold success is not global obstacle success. Ordinary phases and
  collection tails are not fake terminals; budget tails bootstrap.
-512 active decisions per update;5 epochs x4 minibatches; new local initial
  LR3e-4 adaptive KL. Prior actualLR1e-5 is neither silently continued nor lost:
  its optimizer is retained in the immutable historical checkpoint only.
- The old execution-profile sampling_target/curriculum path is deliberately
  not read by this custom route. `local_training.json` and the physical gate
  define the new course; real frozen P01 prefixes have zero PPO credit.

## Entry points

`python -m wlr50_clean.ppo.semantic_rr_capture_local`
supports initialize, diagnostic, train and eval; each requires pinned HEAD and
a fresh run directory under `runs/ppo_rr_capture_first_cp225280_v1`.

The diagnostic alone replaces RR raw channels to seek a ramped entry-FINAL
hip−25°/knee+20° candidate through unchanged physical projection and slew.
Its log identifies previous-ACK inverse estimate versus actual final response.
It holds the first contacted FINAL target instead of continuing the angle ramp.
It is not a policy evaluation, PPO sample, auxiliary learning, or hidden helper.

## Current execution receipt (2026-09-24, first formal training running)

Production is pinned to `ecf205e80094693938057589fc91f7f31082ef89`.
The initial composite was saved and actually reloaded on CUDA at
`checkpoints/history/checkpoint_CP225280_local000000.pt`; local PPO decisions,
updates, optimizer steps and auxiliary updates are all zero. Never initialize
the local head again when resuming: load the last complete composite pointer.

Sealed direction diagnostic (normal exit0):
`runs/ppo_rr_capture_first_cp225280_v1/diagnostic_hipminus25_kneeplus20_ecf205e`.
This is NOT formal deterministic learned success and has zero PPO credit.
It ended naturally at93.5 s /tick11220 with `INCOMPLETE_CONTROLLER_BLOCKED`:
RR never achieved TOP or placed, final gap7.429624 mm, bearing0 N. The final
measured RR angles were(-17.723827,-37.744195) degrees. No runtime error or
model mutation occurred. Source/run manifests and the full failure tail are
sealed. Do not change this result into success because the gap became small.

The frozen natural prefix reproduced the accepted physical FR placement at
23.008333 s, FL placement at43.041667 s, RR qualified lift at58.291667 s and
RR crossing/capture gate at66.491667 s. The independent candidate then moved
the actual committed RR targets from approximately(+7.485,-58) degrees to
(-17.515,-38) degrees. At70.733333 s, measured angles were(-17.717,-37.747),
gap4.741 mm, still AIR and0 N. This proves a measured descent response, NOT
TOP contact or usable support. The entire998-decision inactive prefix matched
original raw requests and committed FINAL targets with max absolute delta0.

Formal training started after the diagnostic process exited, in
`runs/ppo_rr_capture_first_cp225280_v1/train_first2048_ecf205e`.
It collects the first2048 activated on-policy RR decisions using this same
initialized composite; diagnostic target overrides are absent. Afterwards:
save/reload and a natural P01 deterministic evaluation. Prefix decisions
are separately counted and excluded from optimizer storage. The new display
counter CP227328/local2048 must NOT be confused with the historical soft-KL
branch's CP227328. All run manifests and checkpoint pointers are authoritative.

### First stochastic training episode (before any new optimizer update)

The first189 activated samples reached a genuine local success at tick9496 /
79.133333 s. First sampled TOP/bearing endpoint was tick9440. At the local
terminal, RR was placed, had61 consecutive native TOP samples, continuous
verified bearing hold0.5 s, and measured bearing force13.260898 N. Gap was
-0.611323 mm. The logged load fraction had `load_fraction_valid=false` and
must NOT be cited as a validated load percentage. The nominal task label had
continued through P10 to P11; the new local gate stayed active throughout.

This is stochastic on-policy exploration with the initialized local mean,
not a learned deterministic evaluation and not full obstacle success. No
rear target override/assist was present. There were still zero completed PPO
updates at this receipt;189 samples remain in the first512 rollout while the
next real frozen prefix runs. Do not discard/reinitialize the live rollout.
Evidence: `train_first2048_live_episode1.json`, actual decision log and the
later update/checkpoint manifests. The headless training episode has no video;
never substitute the independent diagnostic video for this trajectory.

### Completed PPO boundaries during the live2048 block

-512 credited decisions /1 PPO update /20 Adam steps were saved and actually
  reloaded as `checkpoint_CP225792_local000512.pt`. Actual adaptive LR was
  .00045. Official20 minibatches covered512 rows five times; raw dispatch and
  stored old likelihood matched. Counts:2 capture opportunities,1 completed
  local success,1995 uncredited prefix decisions, AUX0.
-1024 credited decisions /2 PPO updates /40 Adam steps were subsequently
  saved and reloaded as `checkpoint_CP226304_local001024.pt`. Counts:3 capture
  opportunities,2 completed local episodes,1 local success,2992 prefix, AUX0.
  These are training checkpoints, not yet natural-P01 deterministic videos.
-Episode2 ended `INCOMPLETE_CONTROLLER_BLOCKED` at103.575 s after557 credited
  samples. Its maximum native-observer bearing hold was.35 s, below the same
  .5 s local success condition. No criterion was relaxed to call it success.
-Episode3 ended `INCOMPLETE_CONTROLLER_BLOCKED` at105.0 s after578 credited
  samples; RR returned to GROUND. Old AIR/placed history is not success. The
  first three episodes contain1324 credited samples in total, but only1024
  have been optimized at this receipt. Episode4's real frozen prefix has
  reachedP06. Source/configuration are still the pinned version above.
-The running block continues beyond1024; only the pointer and complete update
  manifests define recoverable optimizer boundaries. Pending samples are not
  a completed PPO update. Do not kill the normal run or modify its production
  source/configuration while it is collecting.

### Narrow handoff issue under investigation (no live production change)

Do NOT infer RL unloading from P12 alone: in episode3 the P12 RL source lane
was held at source tick0 and RL remained GROUND. The confirmed earlier issue
is that the P09 late group can start after brief current RR bearing, then its
cursor continues when RR loses contact. This can continue FL/RL target pursuit
without proving P12 strong-unload execution. Read
`episode2_first_RR_drop_readonly.md` and the pending isolated guard proposal.
Any correction must be after a normal version boundary, preserve the accepted
prefix, keep helpful P10 RR knee capture actions and explicit stops available,
and not turn FINAL (which contains policy) into nominal then add policy twice.

## Safety and resumption

### Live-block update, 2026-09-25

The third completed boundary is1536 /3 PPO /60 Adam, saved and reloaded at
`checkpoints/history/checkpoint_CP226816_local001536.pt`
(SHA256 `63bbd57f36630cbda8c7855b2efded7a7b6862b096316094c9b75ff061aaf847`).
Actual adaptive LR is now1e-5, not the initialization value. Update3 mean KL
is.0316092596; JSON-only per-channel reconstruction attributes94.70% to the
other10 channels and5.30% to RR hip/knee. See
`RR_channel_KL_updates1to3_json_only.json`. Do not interpret this as strong RR
mean learning or change exploration before checking the real learning signal.

Episode4 ended incomplete at94.566667s;1746 samples were collected in the first
four opportunities. Its short TOP hold peaked at.058333s and RR later returned
to GROUND. Episode5 is a new real frozen prefix. The block still targets2048;
unoptimized pending rows are not a fourth completed update.

A directed synthetic counterexample found a local-v1 acceptance hole: after
GROUND invalidates the current lift, old crossed/placed plus unqualified new
TOP could satisfy the local hold predicate. This has NOT been observed as a
false success in the current run. The snapshot audit found exactly one actual
success (9496), with current `lift_established=true`; it remains valid. At that
valid TOP, `current_lift_valid=false`, so requiring that AIR/control-availability
field would wrongly reject the positive case. Use the evaluator's same-attempt
establishment instead; every GROUND revokes it and fresh measured AIR earns it.

`staged_capture_lineage_v2/` contains an UNAPPLIED task fix and9 stdlib tests.
It exposes one extra public eligibility bit (448 total) and restricts qualified
hold/success/contact-potential, without altering real TOP/bearing readings or
clearing the active learning gate. Formal delivery must use the corrected
contract after a complete-block boundary, transparent447-to448 weight/Adam
migration and fresh compatible sampling. Do not silently patch the live run.
`staged_late_guard_v2/` is a separate UNAPPLIED, physically unverified source
counterfactual; it defers the unconsumed five-channel late group while retaining
authored stop dispatch. No claim that the full group always harms holding.

The original same-version deterministic evaluation plan is superseded by the
confirmed acceptance-hole repair before formal evaluation. No previous run,
manifest, training credit or result label is retroactively rewritten.

One Isaac process at a time. Do not import Torch/PXR in a parallel helper during
native execution. Do not change source/config/model meaning within a rollout.
All checkpoints are immutable and published only at complete update boundaries.
No robot asset, mass, friction, gravity, obstacle, spawn or actuator capability
was changed. Original task/safety termination rules remain in the underlying
environment. Training/evaluation use the identical control and local task path.

## Complete2048 boundary and cold v2 integration (2026-09-25)

The ecf205 run exited normally, exit0, lifecycle COMPLETE. Final counters:
2048 activated RR decisions /4 PPO updates /80 Adam steps /4986 uncredited
live frozen-prefix decisions /0 AUX. Five opportunities, four completed local
episodes, one stochastic local success; the fifth ended at a nonterminal
collection tail with correct bootstrap. No deterministic RR success is claimed.
Sealed source: `checkpoints/history/checkpoint_CP227328_local002048.pt`, SHA256
`afb7e0e2970c0e7b1c544d4b9f385cbb30e94aa5a8eeebe576c3323ce655d806`.
Its actual final learning rate is2.25e-5. The separate older soft-KL CP227328
is NOT this checkpoint. See `train_first2048_complete.json`.

After confirmed Isaac exit, the actual stored rollout/GAE/fixed-input review
found a useful knee-positive mean shift in update1, but reversal in updates2–4.
67.6% of rows are AIR recapture after TOP and28.5% have RR knee headroom clipping.
No sigma/mean reset or AUX is justified by these counts alone. Preserve them.
See `RR_learning_signal_first2048.md`.

Production v2 now appends one explicitly observed same-attempt eligibility bit
(448 total) and requires it for qualified contact potential/hold/success, not
for permission to act. Current sensor TOP/bearing remain factual even when the
attempt is unqualified. GROUND revokes lift establishment; historical placed
cannot qualify a new unlifted contact. The recorded valid success remains valid.

The isolated route also defers the UNCONSUMED P09 five-channel late group and
the not-yet-started P12 unload while the RR-local task is active. The original
stop carrier remains available; P10 RR knee capture and P11 FR preparations
are not held. All12 local policy channels stay available. Already valid RL AIR
and already-started P12 stop clocks retain existing handling. This is declared
source scheduling, NOT a rear actuator helper or learned contribution. Do not
reuse this local-terminal source design as a completed full-RL controller.

81 directed tests pass, including actual immutable2048 checkpoint migration,
exact old model/Adam columns, new zero input columns, source-pending and sensor
positive/negative cases, and actual temporary save/load. Production migration
publishes a new `_lineage448_v2` filename. Prior439, local learned heads, actual
Adam steps/moments/LR, Identity normalizers and full saved RNG are preserved.
Old incomplete rollout is never reused. `task_v2_*` counters start at0; old
2048/4/80 remain the cumulative branch history. Fresh planned check:512 active
decisions, one complete update, then reloaded natural-P01 deterministic video.
Physical validation of v2 is still pending at this receipt.

### v2 startup correction and current run

The first70e6918 physical startup failed before its first decision, while
constructing the source adapter. Its constructor had incorrectly rejected the
accepted `current_free_lift_before_pending_knee_and_roll_v1` mode. This was an
implementation error, not a task failure or a lost training update. Preserve
`runs/.../train_v2_fresh512_70e6918/failure.json`; its decision/update logs are
empty. No episode was killed. Isaac closed itself; shell exit0 alone is not a
completion receipt. Require a sealed lifecycle COMPLETE manifest.

Fixed and committed revision:
`5f8487b76f27eb165a329a0b6f3096f54ebb4d48`.
The accepted carry mode is retained, including knee source ticks80..120,
roll232, late648, and authored stop864.25 source tests now pass, including
actual factory/accepted configuration and stop single-dispatch beyond source
endpoint. Earlier pending-source tests used a carry-disabled stub and missed
this real configuration; the production-config regression test now covers it.

No new update had occurred, so the same complete2048 source was republished
under the corrected runtime, preserving learned weights/Adam/RNG, without a
reset. Immutable new filename:
`checkpoints/history/checkpoint_CP227328_local002048_lineage448_v2_g5f8487b76f27.pt`.
All three actual CUDA comparison errors (mean/log-sigma/critic) were0 before
fresh training. The initial70e migration remains historical, not overwritten.

Current real run: `runs/ppo_rr_capture_first_cp225280_v1/train_v2_fresh512_5f8487b`.
Target512 fresh activated samples /one complete update. Root PTY3468.
Do not modify source/config while it is live. No parallel Torch/PXR process.
After normal completion, reload its new checkpoint for the formal naturalP01
deterministic video, with original FL assist ON and rear task helpers OFF.

## Complete2560, tracking repair and first trained deterministic evaluation

The 5f8487b fresh512 run naturally completed, exit0/lifecycle COMPLETE.
Its actual new counts are512 RR samples (all P09),1 PPO update/20 Adam steps,
1995 uncredited real frozen-prefix decisions,0 AUX. Cumulative branch counts:
2560/5/100,6981 prefix,7 opportunities,5 completed episodes,1 earlier stochastic
local success. The new512 had no TOP success; its second episode ended at the
nonterminal collection tail. No deterministic success has yet been claimed.
CP227840 SHA:d1a6147bf4ee210b845fcb55e25b25c9d35b4a15d5e5d09124b08a5d752a00d9.
Actual final adaptive LR=1e-5. See train_v2_first512_complete.json.

Read-only source analysis found an unintended v2 side effect: the unconsumed
late full12 sample granted RR tracking although the real pre-late sample647
had no tracking responsibility. The original wheel-stop864 later cleared it.
This was accidental acquisition, NOT loss of old continuous RR tracking.
At the first episode's minimum-gap window mappedN was already fixed; subsequent
RR knee refolding and FL/RL contact changes cannot be attributed solely to that
earlier source transition. See v2_episode1_tracking_owner_followup.md and the
timestamped response table. Preserve the whole affected run, not relabel it.

Only after confirmed Isaac exit, commit1e10d39c9a80c017cb7e4a0035d00c40a5b4dd81
restores the actual previous sample's tracking list inside this pending carrier.
Explicit wheel stop remains exactly once; P10/P11 remain independent; all12
policy channels, physical mapping and HISTORY stay unchanged. No rear helper.
The same448 explicit rebind validates this three-file control-only change and
preserves all actor/critic/Adam hashes, effective LR, Identity, full RNG, counts
and original447 migration receipt. No previous rollout is retained/reused.
Tests:25 existing source tests+6 integrated tracking cases+6 actual tensor
rebind cases+3 route integration cases pass. Synthetic tests earn no PPO credit.

Rebound checkpoint:
checkpoints/history/checkpoint_CP227840_local002560_lineage448_v2_g1e10d39c9a80.pt
SHA81aa94a31039730c7ad87a45b53de22a12361047d324d4d5700b835d7cddbdb5;
sidecar SHAa34999a1a2cdc963b8e7f4d1914c5564ed4a1920987293a6599fcc3313d95956.
The tracking repair has ZERO subsequent PPO samples at this publication.
Do not claim its benefit was learned by update5, which preceded the repair.

Current formal eval: runs/ppo_rr_capture_first_cp225280_v1/
eval_CP227840_DET_tracking_1e10d39, root session7942. Natural P01, reloaded
single packaged frozen CP225280 prior+local RR policy, deterministic, FL assist
ON, rear task helpers OFF, no diagnostic overrides, normal-speed video.
Wait for natural completion; no hot edits and no parallel Torch/PXR. Only sealed
source can be exported with export_formal_from_sealed.py --execute.

### CP227840 formal result and fresh post-repair sampling

Evaluation7942 naturally exited0, COMPLETE/error=null:10940 native ticks,
91.1666667s,1368 recorded decision intervals. RR local/full success bothfalse.
Termination is LOCAL_BOUNDED_RECOVERY_EXHAUSTED, not body collision/wheel-only
climb/safety abort. Last RR gap58.1095076mm,0N,no TOP,no hold. The first998
inactive requests have exact zero local contribution and match accepted CP225280
raw/FINAL (max deltas0/0). FR placement23.008333s,FL43.041667s,RR qualified
lift58.291667s andcross66.491667s exactly match the original event records.
The repaired capture source mapped RR remains−8.15/−39.05°, but policy still
requests enough negative knee residual to reach FINAL−58°. No automatic rear
descent was introduced. This is a real evaluated incomplete checkpoint.

Media export running to video_review/CP227840_DET_tracking_fixed_review,
root session63240. Source is sealed; do not reuse unfinished older MP4s.

Next real train started only after the eval process exited:
runs/ppo_rr_capture_first_cp225280_v1/train_tracking_fixed512_1e10d39,
root session98501. Resume latest CP227840 at same committed1e10d39 runtime,
512 fresh active RR samples, original2560/5/100 history retained. This will be
the first PPO block under pending_source_tracking_inheritance_v1. No source,
reward, distribution, LR reset, AUX or rollout-semantic change. No parallel
Torch/PXR until it naturally completes. Prefix remains real CP225280 and
uncredited. Do not pre-count a sixth update before a sealed manifest exists.

### First real success under the tracking repair (pending update6)

The current run's first opportunity completed local RR capture at tick8312 /
69.266667s after41 active on-policy samples and998 uncredited live prefix
decisions. Native-observer TOP first tick8252, placed8253, terminal61 consecutive
TOP samples/0.5s/12.605516N. Current-attempt eligibility remained true; noGROUND.
P09 late and new P12 RL unload remain pending. FirstTOP preceded the ordinary
P10 knee unfolding; P10 contributes after contact (one existing +0.75deg generic
knee-bias endpoint, not a rear capture helper). See
tracking_fixed_first_stochastic_capture_readonly.md/json for the actual chain.

This is a headless stochastic LOCAL success, not deterministic/full-task success
or update6 learning: these41 rows are still pending in the new512 rollout.
The second frozen prefix began normally. No video exists for this training run.
Do not reuse the previous failed deterministic video as this trajectory.

### Update6 sealed; finite AUX64; fresh sampling (latest)

Run train_tracking_fixed512_1e10d39 naturally exited0/COMPLETE. New512 RR-task
samples/1 PPO update/20 Adam,1995 uncredited prefix,0 AUX. Phase labels104 P09,
2 P10,2 P11,404 P12 do NOT imply RL training: all optimize only localRR.
First opportunity41 samples achieved0.5s/12.6055N; second471 reached TOP but
dropped after0.4167s and is a nonterminal97.8667s budget tail, correctly
bootstrapped. Cumulative3072/6/120,8976 prefix,9 opportunities,2 local stochastic
successes (one older v1). CP228352 SHA46cc0c4939c5f6eb61ab3a629eac37a3fd8db31f44b9463890cf11f85c6714e4;
sidecar96f024bbdb87b9ec36fab6a39710df84aa55b4caafaff82e2937fb19ab4a9ee3.
This pre-AUX checkpoint has NOT been physically evaluated deterministically.

Read-only actual Tensor/storage checks passed (RR_learning_signal_update6_tracking_fixed.json).
Source analyzer initially referred to nonexistent semantic_policy.py; repaired
to the actual HISTORY/P05/rear-owner actor dependencies before successful run.
Success41 normalized advantages all positive. Same-input130 eligibleAIR mean
shifts: hip−0.006006,knee+0.003466 raw; other10 contribute95.92% of globalKL.
129/512 knee headroom-clipped; actual adaptive PPO LR=1.5e−5. Direction improved
slightly, not a demonstration of deterministic contact.

At the cold boundary commit0ff03eafeeb75ba8505a99478cea2a6243cf93f5 adds ONLY
optional finite AUX training helper and route ledger/path/provenance support.
No control/reward/observation/distribution/physics changes. Exact1e→0ff rebind
preserves prior/actor/critic/Adam/RNG/counts before separately declared learning.
Tests20 stdlib/mock+3 CPU tensor pass; actual CUDA load/fit/save/reload also passes.

One finite AUX trial: run aux64_rr_mean_CP228352_0ff03ea,64 independent SGD
steps,LR0.005,max conditional shift budget0.5sigma; all64 accepted. Eleven real
rows8224→8312,15.739mm→contact/hold; NOT the full41 trajectory or frozenprefix.
Only local final mean rows6/7+bias changed. Exact critic/PPO Adam/effectiveLR/
prior/trunk/std/other10/RNG maintained. PPO counts unchanged. Observed maximum
reference mean shift0.05855sigma. AIR knee raw mean0.021146→0.033826; AIR hip
0.202334→0.206040 (wrong direction for this descent subset). Aggregate loss
improvement is NOT physical success or proof of59mm-entry capability.

Latest saved/reloaded CP:
checkpoints/history/checkpoint_CP228352_local003072_aux000064_lineage448_v2_g0ff03eafeeb7.pt
SHAc9bebd75202635e7bfc3efa6f943597e357b2597d2450240bd786820b407e60b;
sidecar9d9ac5f0187e3d6a568b13876259ebe9041903b260a0edba1a522446df09cbe1.
3072 decisions/6 PPO/120 PPO Adam;64 separately-accounted AUX SGD. Unevaluated.
Detailed receipt, full source package, recipe and separate AUX optimizer state
are sealed under the AUX run and referenced in the checkpoint ledger.

Current sole Isaac run: train_after_aux64_fresh512_0ff03ea, session36141,
512 fresh active samples from this exact CP and frozen0ff runtime. No teacher
action at deployment, originalFL assistON, rear helpersOFF. Do not hot-edit or
run concurrent Torch/PXR. Let the complete update save, then reload from natural
P01 for a new deterministic video. Never use CP227840's old video as this model.

### Update7 still collecting — two actual opportunities (2026-09-25 07:49 UTC)

Same sole session36141/source0ff; no production/config/model changes. First
opportunity423 active rows (85 P09/338 P10) ended94.700s incomplete after one
brief TOP endpoint72.2s/22.526964N, stored native hold0.008333s. First subsequent
lost-bearing endpoint remains legalXY; next exitsXY, then real GROUND8737
revokes this lift attempt. Not the older knee-refolding failure: this P10
transition continues knee unfolding. Source stop is real; FL residual remains
reverse. See train_after_aux64_first_opportunity_readonly.md/json.

Second opportunity46 active rows (38 P09/1 P10/2 P11/5 P12) achieved actual
RR local success69.533333s/t8344: eligible legalTOP,67 continuous native samples,
hold0.55s,10.053564N,load_fraction0.330140 valid,noGROUND. P09late and newP12
unload stay pending. This is random sampling before update7, not deterministic
or full-task success. Source tracking/P10 unfolding/current all12 policy all
contribute; AUX causation is not established. See
train_after_aux64_second_success_readonly.md/json; its22-row real descent/hold
window is availability evidence only, NOT a new fit or teacher package.

Total469/512 pending new active rows, with43 still required. Third natural
frozen prefix is running; do not interrupt it, credit it as PPO, or count
update7 before complete save. Latest sealed remains CP228352_aux64 (3072/6/120,
AUX64). Next complete CP expected228864/local3584, then run prepared update7
Tensor analyzer only after process exit, followed immediately by naturalP01
saved/reloaded DET video. Optional gain10 coordinate drafts remain ONLY in
staged_rr_mean_coordinates_only; do not claim deployment or use their source
contracts until a later cold decision after new deterministic evidence.

### Update7 COMPLETE; CP228864 DET now running (2026-09-25 07:57 UTC)

Session36141 exited0 naturally, no Python/Isaac process remained; run_manifest
COMPLETE verified. New512 RR rows/1 PPO/20 Adam,2992 uncredited prefix,0 newAUX.
Actual optimizer rows:166 P09/339 P10/2 P11/5 P12, all RR-only objective.
Episodes423 incomplete,46 capture+hold success,43 nonterminal budget tail at
69.333333s/gap42.0799mm. Cumulative3584/7/140,AUX64,prefix11968,12opportunities,
8completed,3 stochastic successes (including older v1); no new RL curriculum.

CP checkpoint_CP228864_local003584_aux000064_lineage448_v2_g0ff03eafeeb7.pt
SHA2996c9886b68ce716828fe7be84c7dcd28089ddb49f233699fa86bd3b18b8ee5;
sidecar cd9c71db1ccf1e295d84697dee619b2e9670c90aa1daa176459d7dcf27c446c0.
Actual LR1e-5; save/load verified, no deterministic result yet.

Prepared update7 Tensor analyzer actually ran after Isaac exit,5.66s/PASS:
RR_learning_signal_update7_after_aux64.json.512 storage rows exact,20 real
minibatches/five exposures each, Gaussian error3.81e-6,GAE normalization0,
loaded collection mean error2.38e-7,std5.96e-8; frozen prior exact.163 eligible
AIR inputs mean shifts hip−0.003417623/knee+0.003209849;144 final-knee headroom
clipped. Global KL0.0204605,RR contribution4.7942%,other1095.2058%; current
source/model/reward unchanged. This is numerical learning evidence, not DET
success or proof that gain10 would improve physical performance.

Immediately started sole Isaac DET session54703:
runs/ppo_rr_capture_first_cp225280_v1/eval_CP228864_aux64_DET_0ff03ea
using the above saved CP and exact0ff. NaturalP01,same composite all episode,
originalFL assistON,rear helpersOFF,AUX64 training lineage disclosed. Console
outputs/ppo_rr_capture_first_cp225280_v1/eval_CP228864_aux64_DET_0ff03ea.console.log.
Do not hot-edit, do not concurrentTorch/PXR, do not call older videos this CP.
Finish source sealing, export with existing AUX-aware wrapper, inspect media,
deliver new full/detail/CP225280 comparison before optional further revision.

### DET sealed and delivered; RR-mean coordinate migration and fresh block (2026-09-25 08:44 UTC)

Session54703 completed naturally.0ff CP228864 DET:91.166667s/10940 ticks/1368
frames, errornull, INCOMPLETE_CONTROLLER_BLOCKED, RR/fullfalse.10940 native
TOP booleans allfalse, minimum active gap55.230760mm, final57.408820mm/0N.
998 pre-gate raw/FINAL requests exactly match accepted CP225280 and all front/
RR lift/cross event ticks match. RR knee FINAL now leaves-58 after4 active
decisions, but actual max unfolding only1.290069deg; RR hip max reduction2.115deg.
FL active FINAL/actual remain reverse; no mask or forced direction was added.
See CP228864_aux64_DET_result_readonly.md/json. New full/detail/CP225280 videos
were exported, decoded/visually checked and delivered from
video_review/CP228864_aux64_DET_0ff03ea_review. Raw native container timestamp
duration was malformed; exporter used actual1368 continuous frames at15fps,
without time acceleration or removing the failure tail. This video is0ff only.

At a verified no-Isaac cold boundary, production commit
0d0f8948999222f9b8710b0eb913a9419537be83 adds RR-only mean parameter coordinates:
local_training.json, semantic_rr_capture_local.py, semantic_rr_capture_local_actor.py,
semantic_rr_capture_local_aux.py, new semantic_rr_mean_coordinates.py. No changes
to source motion/reward/task/obs448/std/action12/HISTORY/physical limits. Final
local RR mean rows6/7 divide10; matching actual Adam exp_avg*10, exp_avg_sq*100
(AMSGrad same), same Parameters/optimizer/options/step/LR. Mean gain10 cancels
the row change initially; future optimization dynamics intentionally differ.
Do not describe this as new physical action gain or optimizer-equivalent training.
Prior/critic/other10means/std/RNG and AUX64 lineage exact; no added learning credit.
Old AUX SGD recipe now explicitly rejects nonidentitygain; do not reuse .005 blindly.

Actual cold CUDA run migrate_rr_mean10_CP228864_0d0f894 COMPLETE,8.63s. Receipt
coordinate_receipt.json allPASS: raw mean1.19209e-7 maxerror, effective localdelta
2.98e-8, same-action rawlogp0, exact inactive/protected outputs. Both CPU Tensor
tests and70 relevant production tests passed; media25 tests passed. New checkpoint
was freshly strictly reloaded before pointer publication:
checkpoint_CP228864_local003584_aux000064_lineage448_v2_g0d0f89489992.pt
SHA ea6a93db6342a5eaf09cf119c914ebc6e3ddba730e384b7751914cfdfb443101;
sidecar6c02b9d5c099fc3a8a7332e2df726a174b72acbf87359d61441b597aacbadc48.
Counts3584/7/140,AUX64,actualLR1e-5 unchanged. Physically UNEVALUATED version.

New sole Isaac PID74580/session32484:
train_rr_mean10_fresh512_0d0f894, source0d0f894, above explicitcheckpoint,
512 new active RR decisions requested. Console train_rr_mean10_fresh512_0d0f894.console.log.
At08:44UTC normal frozen prefix ongoing; no update8 claimed. Do not hot-edit or
start concurrentTorch/PXR. Let full update naturally save, use newly prepared
gain-aware update8 Tensor analyzer only after process exit, then new saved/reloaded
naturalP01 DET/full/detail/comparison. Rearhelpers OFF, FLassistON, AUX64 history
only. Do not use delivered0ff video as validation of newgain/source/nextweights.

Live first opportunity in this block ended normally with RR_CAPTURE_HOLD_LOCAL_SUCCESS:
143 active rows (135P09/1P10/1P11/6P12), first TOP endpoint9064, terminal9128/
76.066667s,66 consecutive nativeTOP samples,0.541667s hold,11.394116N/current
loadfraction0.408725,eligible/legalXY/noGROUND. Actual RR at terminal
[-16.954412,-20.734461]deg; FINAL[-11.511427,-20.479928]. Snapshot:
train_gain10_first_opportunity_snapshot.json. This is stochastic collection
BEFORE update8; gain migration initially preserved action function, so it is
not evidence that future coordinate-scaled optimization caused contact.143 rows
are pending, no update8/checkpoint yet. Second real frozen prefix continues.

### Update8 COMPLETE; CP229376 naturalP01 DET started (2026-09-25 09:23 UTC)

Former session32484 exited0 naturally; no Python/Isaac remained before analysis.
Run train_rr_mean10_fresh512_0d0f894 COMPLETE. Actual512 newRR samples/1PPO/
20Adam,1995 prefix credit0,newAUX0. First143 capture+hold success as above;
second369 no loggedTOP, nonterminal budget tail at10928/91.066667s,gap25.277119mm.
Tail bootstrapped, not a fabricated failure/done. Actual phases504P09/1P10/1P11/
6P12 remain RR-only. Cumulative4096/8/160,AUX64,prefix13963,14opportunities,
9completed,4stochastic successes includingonev1; taskv2 cumulative2048/4PPO.

Saved CP checkpoint_CP229376_local004096_aux000064_lineage448_v2_g0d0f89489992.pt
SHA cd6d724af6901d03ee235f3b5d3160b4f9db86195d0cf423b3e1e159625368da;
sidecar da5917b6a2affeb9fb0d1d52597d89e0aa15928221de060d5acdc157646f989c.
Actual LR1e-5, evaluated=false. Actual gain-aware Tensor analyzer ran only after
Isaac exit,5.31s/PASS; RR_learning_signal_update8_gain10.json. Exact512 storage
obs/raw/logp/mean/std/reward/done,20 actualminibatches/fiveexposures, Gaussian
error3.81e-6,normalizedGAE0,loadedcollectionmean1.19e-7; same frozen prior.
485 eligibleAIR inputs all hipnegative/kneepositive mean shifts, average
[-0.017138386,+0.008015373] raw conditional, notdegrees; effective localhead
[-0.171383873,+0.080153739] beforeoneHISTORY, alreadygain-applied. GlobalKL
0.014084862,RR10.4974%,other1089.5026%. Not a deterministic success claim or
controlled causal comparison against different earlier datasets.

Immediately launched sole Isaac session59631:
eval_CP229376_gain10_aux64_DET_0d0f894, expectedHEAD0d0f8948999222f9b8710b0eb913a9419537be83,
explicitnewCP, naturalP01, samepackagedpolicy, FLassistON/rearhelpersOFF,
finiteAUX64 history/noevaluationlearning. Console has samebasename underoutputs.
Do not hot-edit or run concurrentTorch/PXR. After normaltermination seal/export
with existing gain-aware wrapper; visually inspect and deliver full/detail/
CP225280samecamera. All older source/weights/videos remain unchanged.

### CP229376 DET COMPLETE; same-version continuous2048 resumed (2026-09-25 10:01 UTC)

Session59631 naturally exited0; one video continuousP01 91.166667s/10940ticks/
1368frames, errornull, INCOMPLETE_CONTROLLER_BLOCKED, RR/fullfalse. RR remains
AIR/0N, finalgap51.159650mm; actual[2.855070,-54.383540]deg versus
FINAL[3.119494,-54.635472]. No claim of capture from angle/gap improvement.
CP229376_DET_P06_prefix_live.json and CP229376_DET_RR_activation_live.json prove
all998 inactive raw/FINAL requests exactly match acceptedCP225280; FRplaced2761,
FLplaced5165,RRqualified6995/cross7979 unchanged. Same originalFLassistON, rearOFF.
Source manifest SHA0de9aa536ac614a27fde5ef808290d931c49841b13c3f31666b680a29e2dfb5f;
run manifest44e5db838d6f1b44c0217db3b98cdbebeb7c86dc8c5295235d7876926ec358c5.

Gain-aware export plan accepted exact checkpoint/coordinate/AUX provenance;
CPU-only exporter session79455 writes new
video_review/CP229376_gain10_aux64_DET_0d0f894_review, all normal speed with failure
tail retained, original acceptedCP225280 comparison. Inspect decoded preview and
receipt before delivery. No second physical run for postprocessing.

After verified Isaac exit and without source/config/reward changes, immediately
started sole Isaac session86454:
train_gain10_continuous2048_from_CP229376_0d0f894 --decisions2048,
explicit checkpoint_CP229376_local004096_aux000064_lineage448_v2_g0d0f89489992.pt.
Console uses samebasename underoutputs. Four512 PPO updates requested, not yet
credited. Natural frozenprefixes produce actual uncaptured RR entries; active
all12 captures/holds. Preserve surviving physical state/HISTORY between complete
updates; real terminals alone start a fresh prefix. No new AUX, optimizer reset,
sigma change, RL curriculum or rear helper. Same actual LR/state inherited.
Check actual per-update saves/counts; checkpoint expected totals are not success.
No hot edits/concurrentTorch/PXR. Last verified physical result is the above
CP229376 incomplete; do not call later weights evaluated without a new run.

CP229376 export79455 has exited0. Root inspected allthree decoded terminal
previews (wholebody/RR/obstacle visible, correctprior/local4096/source/AUX/rearOFF
labels; comparison reference explicitly frozen/notnewphysics). Full1368@15fps,
91.2 encodedseconds, decodevalid, black0, continuousPTS, allfailuretail retained;
SHA7be8e4ac18b3531934b63f5a4bedfac1ea87f91cd7a51c233b250f862d810d2d.
Detail371frames,24.733334s fromsameepisode[997,1368),SHA
cc250c531a6bb00329772c3deb1b303cbe5b79ff3721d734b49b807ff5f2adbe.
Full inline video plus detail/comparison links delivered immediately in commentary.
Sole training process PID74420/session86454 continues; export is no longer live.

Continuous2048 post-run analyzer is prepared and main-agent reviewed:
analyze_rr_learning_signal_continuous2048.py. Python313 stdlib9 tests PASS;
no Torch/PXR/live-run analysis used. It preserves whole-episode TOP/reacquisition
history across512 boundaries, checks nonterminal continuity, pairs each rollout
with its own actual collector/update checkpoint, and normalizes GAE per512.
Final budget tail is not relabelled success/failure. Independent bootstrap
recurrence is explicitly not claimed. Run its --isaac-stopped Tensor path only
after session86454 exits naturally and its real COMPLETE manifest exists.
No production src/config change; sole Isaac continues unchanged.

Live continuous2048 first opportunity ended with real stochastic local success:
87 active rows after998 credit0 prefix. First loggedTOP8624; endpoint8680/
72.333333s,63 consecutive nativeTOP samples,0.516667s hold,13.282523N,
load fraction0.490504 valid,legalXY/currentattempteligible,GROUND0.
ActualRR[-16.409038,-23.345733]deg; final[-15.030061,-14.040685].
This is CP229376 collection BEFORE any new optimizer update, not DET success.
continuous2048_first_opportunity_snapshot.json is a fixed-byte stdlib read;
its sparse-endpoint limits remain explicit. Next real prefix started normally;
87 pending rows are not yet a completed update/checkpoint. Do not startRL.

Continuous block opportunities2 and3 also ended normally as stochastic local
RR successes, still BEFORE update9. Opportunity2:294 active,TOP10272 to
10328/86.066667s,61 nativeTOP,hold0.5s,13.665306N,actual[4.250560,-31.310392].
Opportunity3:124 active,TOP8904 to8968/74.733333s,66 nativeTOP,hold0.541667s,
12.275890N,actual[-5.802261,-18.696573]. Each terminallegalXY/currenteligible
with realTOP/bearing,GROUNDfalse. PhaseP12 is RR hold here, notRL curriculum.
505 pending onpolicy samples total (87+294+124), no new complete PPO/Adam yet.
Fourth real frozenprefix started; cannot manufacture the remaining7 samples.
Fixed-byte snapshots second/third_opportunity and first_two_capture_comparison
record exact limits and contribution. No newAUX, source edit or secondIsaac.
FreeC49.32GiB stayed stable across the two latest checks; no resource blocker.

### Continuous block update9 saved; same episode continues (2026-09-25 11:18 UTC)

New actual512/1PPO/20Adam; phase490P09/3P10/8P11/11P12.3989 prefixcredit0.
CP229888_local004608_aux000064_lineage448_v2_g0d0f89489992.pt has independent
SHAe60c82d31267049a8f00ebb64c0e117e250c82485d4013355334ca9d20a518bf,
sidecar1266ed34ca1a2f8c7d44bdb1c06a41a1b32c7d29940c2c1d41daadc9ff4f1234.
Actualcounts4608/9/180,AUX64,prefix17952,opportunities18,completed12,
local_successes7 (includes historicalv1),taskv22560/5. ActualadaptiveLR1.5e-5;
no manual hyperparameterchange. Save/loadroundtrip/emptyrollouttrue.
Stdlib update9 snapshot:512 rows each5 exposures/20officialminibatches,
all2560 oldlogp matches,selected=issued/nativeverified512/512. NoTorch was run.
KL0.0116964528,clip0.1625,value_loss350.03065,surrogate-0.01898533.
Fourth episode active tail8032/66.933333s nonterminal continues afterupdate;
do not reset its history or count it as a completedfailed/success episode.
SoleIsaac74420/session86454 still collecting nextrollout, nohot edits.
NewCP229888 UNEVALUATED; latestDET/video remainsCP229376 incomplete.
Prepared GAIN10_AUX_FALLBACK_CANDIDATE.md is read-only, NOT implemented or
credited. Current2048block continuesPPO with no additional AUX or rearhelpers.

Fourth opportunity ended INCOMPLETE_CONTROLLER_BLOCKED at10804/90.033333s:
354 active rows, no loggedTOP endpoint/maxhold0, finalgap50.074995mm/0N,
actualRR[2.131127,-57.759306],FINAL[-2.215252,-58]. Its first7 active rows
belong to update9's collectorCP229376; remaining347 are underCP229888 and pending
update10. This is not a standalone fixedpolicy evaluation or new fulltask
success/failure claim. Total block859 sampled,512 optimized,347 pending.
Fifth realprefix started normally. Fourth_opportunity_snapshot preserves limits.

Fifth opportunity (entirely collected withCP229888) succeeded stochastically:
90 active rows; firstTOP8632, terminal8696/72.466667s,67 nativeTOPsamples,
hold0.55s,14.361286N/loadfraction0.483496valid,legalXY/eligible/GROUNDfalse.
ActualRR[-8.240393,-30.206125],FINAL[-6.485059,-29.115913],endgap-0.226836mm.
Totalblock949 sampled/512 optimized/437 pending;4 newlocal successes and1
incomplete,notDETresults. Sixth realprefix startsnormally; no newupdate10 yet.
Snapshotcontinuous2048_fifth_opportunity_snapshot.json, no newAUX or sourcechange.

Sixth opportunity (CP229888) also ended stochasticRR local success:35 active,
firstTOP8192,terminal8256/68.8s,67 nativeTOPsamples,hold0.55s,12.713815N,
loadfraction0.446240valid,legalXY/eligible/GROUNDfalse. EndactualRR
[-21.281494,-18.029363],FINAL[-17.737522,4.610786],gap-0.495815mm.
Hip actual delta vsentry=-28.504918deg; this is actualsampletrajectory,notmean
or a new fixedangle reward/DETclaim. Final knee tracking lag is visible; do not
infer longer-term support robustness beyond the measuredlocalhold.
Block984 sampled,512 optimized,472 pending;5 stochastic local successes plus
1 incomplete. Seventh realprefix started; update10 stillnotearned.

### Update10 actual save, same episode continues (2026-09-25 12:18UTC)

CP230400_local005120_aux000064_lineage448_v2_g0d0f89489992.pt,
SHA804d990df6c7aa21e0acfe427066b068aace35495924eb10cfa97f6f3b2e777c,
sidecar87098425655b8855414827fb69a28da9107bbba30ecea5685e169e6040668710.
Actual bytehash checks/roundtrip/emptyrollout pass. New512/1PPO/20Adam;
phase496P09/2P10/2P11/12P12. Block1024/2/40,prefix6980credit0; total5120/10/200,
AUX64,prefix20943,opportunities21,completed15,local_successes9(includingoldv1),
taskv23072/6. ActualadaptiveLR1e-5,KL0.015652224,clip0.210546875.
Seventh episode tail8296/69.133333s was nonterminal/gap4.800530mm/noTOP;
physicalepisode/history continues afterupdate. FirstTOP8312,maxhold0.425s then
lostTOP; laterphaseP12/AIR/0N is not success. Snapshotupdate10 has512 unique
sample rows/fiveexposures each/20minibatches/2560 oldlogp matches/noextra draws,
selected=issued/nativeverified512/512. Tensor review still waits for soleIsaac
exit. NoAUX/source/rewardchanges. CP230400 UNEVALUATED;74420/session86454 continues.

Seventh opportunity recovered after the0.425s TOP thenAIR loss and ended real
stochasticlocal success at9032/75.266667s:132 active,total1116blockrows,
terminal68 nativeTOPsamples/0.558333s/12.786470N,loadfraction0.466124valid,
legalXY/currenteligible/GROUNDfalse. ActualRR[-5.710589,-37.184423],
FINAL[-3.313596,-5.892447],gap-0.225121mm. Trackinglag is not hidden.
First40 rows wereCP229888 collector; last92 CP230400 (pendingupdate11).
Do not label this mixedcollector episode as standalone fixedCP DET proof.
Block6 stochasticlocal successes/1incomplete;1024 optimized/92pending.
Eighth realprefix started normally. Snapshotseventh_opportunity preserves data.

Eighth opportunity, collected entirely with CP230400, reached stochastic local
capture+hold in29 active rows: firstTOP8144, terminal8208/68.4s,66 nativeTOP
samples/0.541667s/13.642917N/loadfraction0.483298valid, legalXY/currenteligible,
GROUNDfalse. ActualRR[-6.279818,-36.783891],FINAL[-2.624751,3.864464]; substantial
knee tracking lag is disclosed, not converted into long-duration stability.
Block1145 sampled/1024 optimized/121 pending;7 local successes/1 incomplete.
Ninth real frozen prefix started normally. No new optimizer/AUX/source changes.
Snapshotcontinuous2048_eighth_opportunity_snapshot.json is bounded read-only
evidence, not a new deterministic evaluation. Latest actualCP230400 remains
UNEVALUATED until the complete block exits and a separate naturalP01 run occurs.

Ninth opportunity, entirely CP230400, ended INCOMPLETE_CONTROLLER_BLOCKED at
10948/91.233333s after372 active rows. No legalTOP/bearing endpoint, maxhold0;
historical placed=true does NOT override currentTOPfalse/0N/outsideXY75.595mm.
GROUND was observed during the episode and revoked the old AIR qualification;
later free-air/requalification did not produce capture. EndactualRR
[8.380804,-27.926085],FINAL[8.488203,-22.489601],gap-50.202361mm outsideXY
is not a negative-gap successful landing. Block1517 sampled/1024 optimized/493
pending,7 stochasticlocal successes/2 incomplete. Tenth realprefix now starts;
no new update11 yet. Snapshotcontinuous2048_ninth_opportunity_snapshot.json.

Bounded episode9 follow-up (continuous2048_ninth_incomplete_readonly.md/json):
historicalplaced8218 plus TOP detector minimum2 and AIR-since8219 supports brief
nativeTOP8217-8218, not sustained bearing; individual force/hold peak unavailable
in decision-endpoint log. 8224 is alreadyAIR/0N; firstP10knee8248 follows the loss.
Pre-loss source/mapped unchanged while sampled FINALknee refolded. GROUND8414
revokes eligibility; real freeAIR rise11.929mm/16ticks permits neweligibility8552,
not mere reuse of oldplaced. Later legalTOP/hold still absent. No automatic
pure-wheel violation inferred from edge contact; lategroup unconsumed/noP12.

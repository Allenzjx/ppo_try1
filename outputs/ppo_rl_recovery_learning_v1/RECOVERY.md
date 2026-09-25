# RL recovery learning — current implementation boundary

## SEALED CP227328 delivery — no active Isaac (2026-09-24 15:27 local)

Training13695 exited normally0; DET86314 finished naturally and source sealed.
The launcher exits1 because lifecycle is DIAGNOSTIC_FAILURE: task incomplete,
not a damaged recording. 1387 decisions/11092 ticks/92.433333s; final interval
4ticks, encoded1387frames/92.466667s. FRplaced2761, FLcross4316/placed5232→P06;
RRqualified7143/cross8230 but neverplaced, gap55.836mm/AIR/0N at end; RL never
qualified/crossed/placed. First unmet task is RR capture and usable support.
P09 INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED; valid
physical run, no safety abort, final CONTACT_BEARING_UNVERIFIED for RL.

Latest saved/reloaded/evaluated CP227328 and hash/count identity are below.
Same protected branch cumulative2048/4PPO/80Adam, front-KL2560 exposures,
zero extra on-policy/AUX steps. No new rear helper. Exact branch coverage is
in CP227328_branch_physical_coverage_recheck.md/json and the P10 sealed report.
Original CP225280, old branchCP231680, CP225792/226304 and N_ref remain intact.

Latest concise delivery: CP227328_URGENT_STEER_DELIVERY.md.
Sealed same-tick rear table: CP227328_DET_RR_postcross_8230_11092.md/json;
359 postcross decision endpoints/2863 native+sensor ticks, no new simulation
or training. Do not reuse oldCP explanations: knee FINAL saturation315/359,
tail clip only0.000603deg; RRhip positive offset and FL reverse persist;
inputowner17 all0/prepallowed, no demonstrated all-leg mask fault.
Three QA-passed videos: video_review/CP227328_deterministic_front_preserved_v1_review/.
Full and detail names are INCOMPLETE; detail is RR_capture, not fictitious RL
progress. HistoricalN is explicitly frozen after its real end. Main agent
viewed the contact sheet and detail first frame; source manifests are unchanged.

No training/evaluation is active or promised in the background. Next resumption
must use the explicit cp225280_front_preserved_v1 checkpoint namespace and
official compatible loader, not old ancestor files sharing numeric step names.
The prepared direction probe has never run and contributes0 training/evidence.
All dated activity sections below are historical, not commands to restart them.

## CP227328 sealed; natural-P01 DET active (2026-09-24 14:43 local)

P10 session13695 exited naturally0 after actual1024 student decisions/2PPO/
40Adam. Latest same-branch CP227328/PPO1729/Adam34580, binarySHA
`5b871af2df8d6b9c0180f7a51863b18b2153148d8dc2a94876f21f5bd73791b4`,
sidecarSHA `0200daa9a22652e1f1c64e5c5ff82a0c977f0418c508158455b3a5ce85bac308`.
Pointer/binary/sidecar match; official save/load roundtrip=true. The existing
adaptive optimizer now reports actualLR3.375000000000001e-5 (not the1e-5
launch value). New protected branch cumulative2048/4PPO/80Adam, with2560
offline front-replay row exposures in those80 official minibatches, zero
additional on-policy samples or separate AUX optimizer steps. The old branch
CP231680's separate512/1PPO/20Adam boundary is retained, not in this lineage.

Sole Isaac is session86314/PID72620, frozen892385, saved-reloaded CP227328,
DET naturalP01/seed4001/full_episode/max3000 decisions/200s. No teacher/model
splice. FL assistON, rear task capture/geometry/forced-forward assistanceOFF,
declared issued-owner suspension/projectionON. No production/reward/action
change since892385. Actual source:
`runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T1842131087811Z_g892385cba8a7_c074d1ebec6947cc830fb3e0ec86dcec/source`.
This checkpoint's full evaluation is NOT yet complete; keep verified CP226304 and
its three videos. CPU-only sealed training analytics may run; no independent
Torch/PXR/model helper, another Isaac, or hot runtime changes.

Live milestone15:01 local: CP227328 naturalP01 has FR placed2761=23.008333s,
FL crossed4316=35.966667s, FL placed5232=43.6s and enteredP06. At45.666667s
it remains physically valid/nonterminal. The same student continues rear tasks;
this confirms front completion only, not a complete evaluation/traversal. Original
FL assist contribution remains disclosed; the current video is still being written.

All activity sections below are historical snapshots, not current processes.

## ACTIVE P10 continuous RL block (2026-09-24 14:20 local)

First512 update is now sealed: CP226816/PPO1728/Adam34560, actualLR1e-5,
official save/load roundtrip=true. BinarySHA
`8fcf3bdf87525c4448a39bf67ee9e6ac0e4e7d7f932d5912e05de272bb23199a`,
sidecarSHA `401c41bc28c47cd90e6d46dc12dd5e5e7a7aa2724937fe42bd0cb4da6ba89085`.
Pointer, binary and sidecar independently match. New front-preserved branch
cumulative1536 student decisions/3PPO/60Adam; replay1920 row exposures in
those60 minibatches, zero additional on-policy or AUX optimizer steps.
This is NOT yet a physical evaluation of CP226816. Evaluated CP226304 remains
protected. Same session13695 continues its second512 buffer, with no hot edits.

CP226304 natural-P01 session5269 exited naturally1: sealed task incomplete,
not recording corruption. 11044 ticks/92.033333s/1381 decisions; FR/FL placed,
RR qualified7108/cross8152 but still AIR/0N/gap55.335mm, RL never qualified.
First unmet task is RR capture/bearing. The same run's full/detail/historical-N
package has been exported, decoded and delivered. The authoritative directory
is `video_review/CP226304_deterministic_front_preserved_v1_review`; the detail
is labelled RR capture incomplete / RL not reached, not RL completion.

Sole Isaac is now session13695/PID10172, same frozen892385, run:
`runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T1717242165088Z_g892385cba8a7_afdc21482bbc480a9692fb63fe8ad858`.
From CP226304, phase_suffix/P10/offset0/successful_nominal,1024 requested
student decisions,512 per update/save every update, seed1001/N1/cuda0.
Plan `CP226304_P10_nominal_prefix1024_candidate.md` is now launched; planned
CP226816/227328 are NOT earned until complete update receipts appear.
Latest sealed checkpoint remains CP226304/1727/34540 at launch. No new
production/reward/distribution change, migration, assist or AUX. Prefixes
zero PPO credit. Do not run Torch/PXR/second Isaac or hot-edit runtime.

Historical 14:11 snapshot:439 new student decisions collected, then pending.
Three student episodes ended after59/111/269 decisions: BODY_COLLISION,
HARD_JOINT_LIMIT, HARD_JOINT_LIMIT. The fourth real nominal prefix is active,
with zero PPO credit. Do not read the last terminal row as the current state.
That pending buffer is now covered by the full512 receipt above. Do not count
the second buffer until its own complete receipt appears.

No exporter/second Isaac is running. An outputs-only diagnostic driver is
prepared but has NEVER been executed; it is not a production rear helper or
a training/evaluation gate. All older activity sections below are historical;
they do not authorize restarts.

## CP226304 sealed; new naturalP01 video active (2026-09-24 12:42 local)

P07 session47934 exited naturally0 after the full512/1PPO/20Adam. New same
front-preserved branch CP226304/PPO1727/Adam34540, source CP225792. Checkpoint
SHA c0f4727bcc4e1963b5e9d079158dde1f5f662390397b61049ff893c40e1910f6;
sidecarSHA4d1e103507e7083f9d1ecb6d5ff230bd3d106054da2af0ef85e0ce81b9c8569b.
Pointer/binary/sidecar match, official save/load roundtrip=true, actualLR1e-5.
New branch cumulative1024 decisions/2PPO/40Adam; replay1280 exposures/40same
Adam minibatches/0extraPPO or AUX steps. Last heldoutmeanKL .0061758384,
not a physical retention guarantee. Sealed coverage is complete in
`892385_P07_CP225792_512_coverage.md/json`, including cumulative1024 credit.

Sole Isaac session5269 now runs saved-reloaded CP226304 DET naturalP01,
seed4001/full_episode/max3000 decisions/200s, frozen892385, same FLassistON /
rearOFF / declared owner projection. No teacher prefix or model splice. Keep
the physically verified CP225792 candidate and its v2 three-video package.
Do not run Torch/PXR/model helpers or hot-edit production while5269 is active.
Exact source:
`runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T1641350708339Z_g892385cba8a7_de3f11b3a669411abf3e6658002008cc/source`.
At12:49 local it was decision270/sim18.0s/P02, valid/no terminal; do not
mistake this live prefix status for the sealed CP225792 evaluation below.

All dated sections below are historical snapshots, not additional active
processes. Only5269 is active at this boundary; never restart those old runs.


## Active P07 learner continuation from front-preserved CP225792 (2026-09-24 11:28 local)

Sole Isaac session47934/PID65540, frozen892385; run
`train/20260924T1527536734451Z_g892385cba8a7_f00e6fb48d074b26a2a98898505632e6`.
Same branch `cp225280_front_preserved_v1`, actual CP225792 SHA17a7e849...,
phase_suffix/P07/offset0/successful_nominal,512 requested learner decisions,
seed1001/N1/cuda:0/save every full update. No source/reward/config/mean reset,
no teacher credit, no rear helper; same frontKL/FLassist/history/limits.
Purpose: learner RR capture followed by RL preparation, not guaranteed RL
coverage. New counters are pending the complete update; do not label this run
as CP226304 yet. Keep CP225792 as the evaluated front-restored candidate.
No Torch/PXR/model helpers or production changes until this run exits normally.
CPU-only Pillow/libx264 export QA may run concurrently, never a second Isaac.

First P07 student episode ended after74 decisions at47.916667s/P09 with real
HARD_JOINT_LIMIT (RR knee measured−60.058109° vs−60° hard lower bound).
FINAL remained−58°; native mapping/dispatch verified. This is actual response
undershoot beyond an in-range target, not a sent out-of-range command. The
existing safety abort/reset worked and was not loosened. Second real prefix
is in progress; buffered74 decisions are not an optimizer update yet.

Later live status (12:21 local):272 learner decisions buffered; fifth real N
prefix in progress. Completed episodes:74/RR knee hard limit,113/FR knee hard
limit,48/body collision,37/body collision. Episode1's RR qualified5371,
cross5666 and placed5697 occurred after5160 handoff; RR subsequently lost load,
and RL did not qualify before that safety abort. No second PPO update or new
checkpoint is claimed yet. Do not confuse the previous terminal row with the
currently advancing teacher prefix.


## CP225792 naturalP01 sealed; rear continuation next (2026-09-24 11:24 local)

Session34489 has exited naturally; no Isaac process remains at this boundary.
Formal saved-reloaded CP225792/892385 naturalP01 DET completed1392 decisions,
11132 physics ticks/92.766667s. FR/FL placed, RR qualified7198 and crossed8308,
but RR not placed/0N/gap58.382mm at end; RL not qualified/crossed/placed.
The source is sealed with lifecycle DIAGNOSTIC_FAILURE only because
`SemanticVideoError: episode did not meet common physical task`; preserve this
classification, not a successful task or a viewport callback failure. Final
interval4ticks gives1392 encodedframes/92.8s; no episode tail was removed.
Artifact export is active outputs-only. Original source manifests are immutable.

P07 nominal-prefix512 continuation from this same CP225792 is prepared, not
yet launched. It needs no control/reward/migration change; preserve frontKL,
seed1001/439/512/Adam/LR, frontFLassistON/rearOFF. The purpose is to include
actual learner RR capture followed by RL preparation rather than only a
P12 prefix with RR already placed. Report reached states, not intended labels.
Exact command: CP225792_P07_nominal_prefix512_candidate.md.
Original CP225792 is now a physically verified front-restored candidate;
retain it even if a later checkpoint has a larger training counter.


## CP225792 saved; naturalP01 video active (2026-09-24 10:48 local)

Training session30343 exited normally0 after512 fresh rear decisions/1 PPO/20 Adam.
New branch cp225280_front_preserved_v1: CP225792/1726/34520, save/load verified.
CheckpointSHA17a7e849afc242856f035350d814349be1e101b49385c42ab9c90966a84fc5a7;
sidecarSHAd42847f6c90612366bdb16ca574e11403430ac368ed487a69a0d904b73d8cc6d.
Actual replay:20 minibatches/640 front exposures,0 extra PPO/AUX steps.
Exposure phases P01=7/P02=224/P03=14/P04=7/P05=196/P06=192.
Fit/heldout mean KL after update .007811164/.007886615; this measures drift,
not physical front retention. Same saved Adam/LR approximately1e-5/Identity.
First442 student episode81.2667s/P12 incomplete; second70 nonterminal tail
56.4667s, RR actualTOP13.484N, RL obstacle contact/unqualified/not crossed.
Both real nominal prefixes777 each are excluded; initial RL qualification6212
precedes6216 handoff and is not student credit. New learner RL6437 did qualify,
reach61.475mm above top but still159.412mm before edge, then lose qualification.

Sole Isaac video session34489 now evaluates this exact saved student naturally
fromP01, DET seed4001, full_episode max3000 decisions/200s, same892385 runtime,
FLassistON/rearOFF/owner projection declared. No teacher prefix/model splice.
Do not run Torch/PXR/helpers or edit production while34489 is active.
Publication receipt: staged_cp225280_front_branch/front_preserved_CP225280_g892385cba8a7_publication.json
(SHA29bd2ba487088f98d76944566959057a8ffb280ab72ae51eae47f65eff0e0379).
Old branchCP231680 and all its updates remain separate and preserved.


## Active CP225280 front-preserved rear block (2026-09-24 10:12 local)

Pinned runtime892385cba8a7089b52567bb7f558c018fac82a77. Sole Isaac session30343,
run train/20260924T1412116439544Z_g892385cba8a7_791e04b05a7a4df9a493a606cd958a57.
New independent branch cp225280_front_preserved_v1, from published CP225280
identity checkpoint4240785e0fb18f1e91c2b0b8ce2592304bf84cca20a4c6853fe707371b9b5d9c;
sidecar8b37a904ce5b0c32a48f125f1c76ba0326dee69756c8f56189821595b49de8e3.
Full-state publication/reload and199 real-input Gaussian identity passed,0 learning.
Source counters225280/1725/34500; all legacy later CP231680 data stays untouched.
Command: train v3 rr_rl_timing_policy_learning_v1 phase_suffix512/P12/offset8,
successful_nominal continuous prefix, seed1001, N1/cuda:0, save every update.
Expected per completed block512 actual samples/1 PPO/20 Adam plus640 front replay
exposures (0 PPO/0 extra AUX credit). Actual new results pending sealed update.
No new front reward; rear assistsOFF/FLassistON/owner projection declared.
Do not hot edit production or run Torch/PXR/model helpers while30343 is active.
Next: saved-reloaded student naturalP01 DET video, no policy splice.


## Urgent front-preservation switch: old update sealed (2026-09-24)

Session68580 exited normally0, STOPPED_AT_VERIFIED_UPDATE_BOUNDARY. Its P10
course added512 actual learner decisions /1 PPO /20 Adam, not the requested1024.
Saved old branch CP231680 /1763 /35260; checkpointSHA
1893d4c35c51cb20254c5353d19cb05a642e68c7e9b53bfffcf0d25887e94922,
sidecarSHA f85a60aebc321258ab1bb366491f93fede34a3a9ffb493d60cf9ea78e9122b2f.
Learner phase counts P10=2/P11=2/P12=508. Real nominal prefixes1534 receive
zero PPO credit. First episode81.275s/P12 incomplete, not RL success.
Old-branch total since225280 is6400 decisions/38 PPO/760 Adam; AUX32 separate.
All old checkpoints, videos and pointers remain preserved.

Production switch now applies the independently staged CP225280 front-preserved
identity and front-only Gaussian KL in the existing PPO optimizer steps. It does
not change N, mapper, HISTORY, FL assist, rear assist OFF, physics or new reward.
Compatibility/numeric tests and zero-credit publication precede the first new
rear512 block. Independent branch counters begin225280/1725/34500, not231680;
this is explicit user-authorized source selection, never relabelled later learning.
No current Isaac process. Older active-status paragraphs below are historical.


## Active fixed59e P10 continuation (2026-09-24 09:19 local)

Single Isaac session68580 runs
`train/20260924T1319527689424Z_g59e868f3e223_77de1c96e9684016876ea29d7c101116`
from immutable CP231168, phase_suffix/P10 offset0 with continuous real
successful_nominal prefix, seed1001, N1/cuda:0,1024 requested learner decisions.
Prefix has zero PPO credit. No production/reward/distribution change; frontFL
assistON/rearOFF. Keep one Isaac; no Torch/PXR/model helpers while68580 active.
Actual new count is pending complete updates. Urgent user steer supersedes the
old2048 plan: stop_after_update.request.json is now present, checked only after
the complete512 PPO update and verified save. No kill/hot edit. Preserve this
branch's new rear data/model, then use an independently routed compatible
CP225280 front-preserving branch, keeping later rear implementation where
validated. Do not require CP230144 descendants to relearn allFL beforeRL work.
Pending readonly tasks: event-aligned FRplaced/frontmount descent comparison,
CP225280 full-state/interface/HISTORY/N/mapper/FLassist migration. Next official
video must be one saved-reloaded student fromP01, no policy splice.

## Sealed fixed59e naturalP01 continuation (2026-09-24 09:19 local)

Single Isaac session17276 exited naturally0, lifecycleSUCCEEDED, after
`train/20260924T1257554297942Z_g59e868f3e223_8a6cf8fd89554916abdc8a6b9f6fe1ac`
from immutable CP230144, full_episode/P01, preserved seed1001, N1/cuda:0,
1024 requested decisions with512 collection and save every update.
No production/reward/distribution changes; front FL helperON, rear helpersOFF.
Actual1024 decisions/2 PPO/40 Adam saved; currentCP231168/1762/35240.
Checkpoint SHAa859f84973e3fced3c79b03af7042bedfc68e96996da83b60791494bae2b39c2,
sidecar SHAfd6dc56a0968ffdc333df89d3175d7809fc7d81494318f6476ec08029e8572b0.
First243-sample episode ended P02 incomplete. Second781-sample episode had
FR placed, FL/RR/RL not placed; nonterminal P05 at52.066667s budget tail.
This is not formal DET and not a task success. Current-request accumulated
learning sinceCP225280 is5888 decisions/37 PPO/740 Adam, AUX32/32 separate.
The previous125638 launcher/session40718 failed preflight before simulation:
seed1002 differed from saved1001. Its stderr is retained,0 sampling/update
credit. Resume now preserves the checkpoint seed/RNG; no validator bypass.

First complete update is now1761, CP230656, +512 decisions/+1 PPO/+20 Adam.
Pointer binds checkpoint SHA27412d8f4e73481c0a53e180f51e375e3194787aa9148a9b382dc45f8880aa29
and sidecar SHA5b65f4df975be217a6494f5a30cb1ccf59477849c2aff44d1bda72a8d99acdba.
LR remains1e-5; no new AUX. The first243-sample episode ended P02 incomplete,
not safety failure. Second episode placedFR, currently P05 FL not placed.
CP230656 is not yet a formal DET evaluation and is not claimed better.

## Sealed isolated diagnostic (not PPO or a formal policy evaluation)

Single Isaac session2202 exited naturally with exit0 after running
`runs/ppo_rr_rl_timing_policy_learning_v1/direction_probe/CP230144_FRk_entry_minus4_v1`
from the same CP230144/59e naturalP01 state. The reviewed FR-knee-only REQUEST
candidate executed ticks2096–2456 (17.466667–20.466667s): fixed entry
−4.369524768° to entry−4°, 1s ramp/1s hold/1s release, 45 modified decisions.
It released, then the same student reached the ordinary P05 incomplete terminal
at56.933333s/854 decisions. Other11 raw channels used the current student; actual
intervened HISTORY is retained. No model/optimizer update and no PPO/AUX credit.
Front FL helperON, rear helpersOFF; production remains frozen59e. The first50 unmodified
decision records matched the formal video numerically; this is a limited causal
check, not an exact-entry gate or a statement of later equivalence. The sealed
manifest asserts frozen_learned_state_unchanged=true and zero PPO/AUX credit.
The tested hold window worsened FL clearance and CoM height; no successful AUX
target is derived. See CP230144_FRk_probe_through2560.md/json for measured deltas.

## CP230144 formal DET retry sealed — actual P05 task incomplete

SameHEAD/CP/seed4001 naturalP01 retry
`video_eval/validation/20260924T1206270979619Z_g59e868f3e223_6daab84a9021442ca9d5a14aea51c95e/source`
sealed854 decisions /6832ticks /56.933333s. P05 age40s terminated
`INCOMPLETE_CONTROLLER_BLOCKED / LOCAL_BOUNDED_RECOVERY_EXHAUSTED`;
physical evaluator VALID with no safety termination. FR placed; FL/RR/RL not.
FL qualified2048, first post-peak gap<0 at2602, actualGROUND revocation3196;
no later qualified event. FinalFL AIR0N is unqualified, active_attempt=false,
gap−51.185mm and outsideXY, NOT a newly valid lift. P05 assist wasWAIT and
uninitialized for all600 decisions, never reached capture entry/XY/TOP;
not descent-assist budget exhaustion. Source wheelSTOP appearedpoststep3120,
actualnative dispatch waszero from3121; policy residual still commanded wheels.
See `CP230144_DET_P05_failure.md/json` and updated `FINAL_DELIVERY.md`.
Three same-run videos are sealed under
`video_review/CP230144_deterministic_collection512_review/`: full and truthful
RR_NOT_REACHED predecessor detail have854 frames/56.933334s; historical-N pair
has1108 frames/73.866667s. All fully decode, have continuous monotonicPTS and
zero black-like frames; root viewed full/pair final QA images. See export_receipt.json.
Independent FR-knee direction probe is separate from this formal episode and
adds no PPO/AUX credit; no unsealed probe result is asserted here.

## CP230144 video retry boundary — artifact error is not task failure

First formal naturalP01/DET/seed4001 attempt at
`video_eval/validation/20260924T1200016881929Z_g59e868f3e223_682c728eb6c54cdfa895ae0c3a080c98`
sealed `DIAGNOSTIC_FAILURE` at2026-09-24T12:03:59.293531Z. Physical execution
was936ticks/7.8s with117 issued decisions/capture requests, but the actual
completed video is116frames/7.733334s. That partial source is retained.
Exact source acceptance error:
`VIDEO_OR_ARTIFACT_ERROR: encode failed: RuntimeError: viewport callback_count=0 after 3 waits, expected 1`.
This is a viewport/media callback failure, not a physical task failure or a
new P01/P02 regression. It adds0 PPO updates and does not alter CP230144.
Main task ran one sameHEAD/sameCP/seed naturalP01 retry (session39333),
run `20260924T1206270979619Z_g59e868f3e223_6daab84a9021442ca9d5a14aea51c95e`;
its actualtask result is recorded above; videoexport/QA is separate. Preserve
this failed first attempt separately and do not relabel it as a full evaluation.
`FINAL_DELIVERY.md` consolidates sealed training/formalDET evidence, not a
task-success declaration. Main task controls the soleIsaac resource.

## CP230144: first collection512 block sealed (59e)

Frozen HEAD `59e868f3e223c589e7645a0f5d63f91fa6119fb6`; training PID33872
exited naturally with exit0. Run
`train/20260924T1121437392178Z_g59e868f3e223_63b8d42e71504d8197a9228aabc8625b`.
Official single-block command: phase_suffix/P12 +8 successful_nominal prefix,
512 actual learner decisions, N1/cuda:0/seed1001/checkpoint every update.
Sealed lifetime230144 /1760 PPO /35200 Adam; this block adds512 /1 /20,
not4 PPO updates. Turn totals from225280 are4864 /35 /700, plus separately
32 accepted /32 attempted AUX; this block adds no AUX. Latest checkpoint:
`outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000230144.pt`
SHA `376ab4b6fe6f2a6a82bedd1190ba0814dc5e5b396127c48c34cf7b1d4a8265f1`;
sidecarSHA `30dc90ece977823762eb7f19b79b944f41b20e14c5231c45014ab58d17db3ba9`.
Both actual file hashes match the branch latest pointer.

Two continuous nominal prefixes contribute1554 decisions and zero PPO credit.
All512 learner samples areP12: first443 end in real bounded-recovery terminal,
second69 remain nonterminal and bootstrap. First episode RL obtains two fresh
qualifications but no cross/place; RR ground event7756 has no later legalTOP
recovery. Second episode only inherits prefixRL qualification6212, revoked6250;
its56.4s tail has FR TOP, FL/RR AIR, RL GROUND. This is not task success.
CPU saved-rollout check verified all512 raw/reward/V/logp/adv identities,
20 minibatches and exactly5 exposures per sample. No model was loaded.
See `59e_P12_512_coverage.md/json` and `59e_collection512_vs72e_sealed.json`.
CurrentCP naturalP01 deterministic video/acceptance is a separate main-task run;
do not reuse old video as its evaluation.

Source is `checkpoint_collection512_CP229632_g59e868f3e223.pt`, SHA
`a79384986711c4466559544f27b55f6f3094f9433b180e35c541993786a967d3`;
sidecar `90fbe1837f541eca96a589cd2f1a2c72255856d61b6d37d52517a9c06504a9a0`.

Publication/reload passed; 12 real fixed inputs have bitwise-identical
mean/sigma/value, complete Adam/LR/Identity/RNG and AUX32 preserved.
Publication adds0 decisions/PPO/Adam/AUX. Fresh512 storage, same gamma/lambda,
5 epochs/4 minibatches; the completed full512 added1 PPO/20 Adam, NOT4 PPO/80 Adam.
No rear task/geometry/forward assist added; original FL assist staysON.
The offline CPU helper exited after its check; no helper is left running.
The failed912 publication is retained, not deployed. See
`staged_collection512/DEPLOYED_BOUNDARY.md` and the valid publication receipt.

## CP229632: post-AUX P07 block sealed; real learner RR capture, RL not yet qualified

65a run `train/20260924T1046432416199Z_g65a9255be6d9_bff8c2f6d5574a42925e6b9d8f382a94`
naturally completed512 learner decisions /4 PPO updates1756–1759 /80 Adam,
using the existing128-step collector. PID33224 exited; no task terminal was
fabricated. Lifetime229632/1759/35180, LR1e-5, roundtriptrue.
Verified checkpointSHA
`74f6d2753e14f64b64af11321e10ece0c8658a0af4e34ffdfd60fe524f535762`;
sidecarSHA
`a882a6c3150dd180aa81255fcbffa0272914d267b0edbf972920c20e63028da7`.
Actual turn totals from225280 are4352 decisions /34 PPO /680 Adam, plus
separately32 accepted /32 attempted AUX; this P07 block itself added no AUX.

645 continuous successful_nominal prefix decisions endtick5160/43s, zero PPO
credit. RR qualification5432, crossing6185 and realplacement6270 all occur
after student handoff, not during the prefix. Seven sampled endpoints retain
legalRR TOP bearing, then RR loses contact and ground event6543 revokes its
current qualification; no later legalTOP restoration. RL has no qualified
lift/recovery/cross/place. Phase samplesP07=1/P08=1/P09=137/P10=1/P11=2/P12=370.
The77.1333s P12 tail is nonterminal with bootstrap; FR TOP, FL AIR, RR currently
qualifiedAIR outside the landing region, RL GROUND. HistoricalRR placement is
not current support. See `65a_P07_postaux512_coverage.md/json`.

This is suffix training progress, not naturalP01 deterministic success and not
isolated proof of an AUX contribution. CP229632 has no corresponding formal
video yet. The collection512 candidate is a separate subsequent boundary;
do not reinterpret these four historical128 updates as one512 update.

## 65a accounting identity and finite front-retention439 AUX published; physical evaluation pending

Runtime `65a9255be6d9fd4e19590a48a3a650ae606b04f7` adds only the explicit
front-retention439 identity/ledger accounting paths. Its identity publication
preserved CP229120 counters `229120/1755/35100` and all learned state with zero
PPO/AUX credit. A single separately accounted finite retention fit then completed
`32 accepted / 32 attempted` and passed the official independent save/reload:
checkpoint SHA
`3b55427ae443accbae34413498189036f907f36832e769350435a95b41cc0148`,
sidecar SHA
`7af9749c7e810f70a230b561727bbfb2efcba9b9f7abebd2386f167634890a52`.
Lifetime PPO counters remain exactly `229120/1755/35100`; these 32 AUX steps are
not PPO, teacher deployment, RR capture success, or a physical-evaluation result.
The bounded fit reported maximum REQUEST changes `0.035654°` (joints) and
`0.000918925 rad/s` (wheels), maximum full-Gaussian KL `0.0122388`, and maximum
absolute log-sigma change `0.0561200`, all within its predeclared fixed trust
envelope. A new real P07 `successful_nominal` prefix continuation is active as
root session52106; it has no credited outcome here until its next complete PPO
boundary and no deterministic video yet.

## CP229120 sealed; formal 72e natural-P01 DET exported and decoded

The ordinary full_episode/P01 block naturally sealed all1536 requested learner
decisions: +12 PPO/+240 Adam. The block includes a102.4s nonterminal P09
training tail. RR acquired a real short qualification at tick10211/85.0917s,
visible for11 decision endpoints, then was revoked to GROUND at
tick10301/85.8417s; it never crossed or placed and was not sustained/recovered
into capture. At the102.4s tail RR isGROUND and FL isAIR. FR/FL `placed`
records are historical events, not evidence that both front legs were currently
bearing during the RR window. This is training evidence, not a formal
deterministic result or task terminal. Immutable checkpoint
`checkpoint_step_000229120.pt` is lifetime229120/1755/35100, LR1e-5,
save_load_round_trip=true. Checkpoint SHA is
`a02bfd50e17cad3bebe083493b0da9611cfe607fa18dad6a4ed83abdecb91f1c`;
sidecar SHA is
`7c2b8958f695387f9c186978ec6a4b80df496220cfbd4d0eabfe62dc6e683036`.
Turn total is now3840 decisions /30 PPO /600 Adam /newAUX0.

The same-checkpoint deterministic full-P01 run sealed naturally at57.866667s /
tick6944 with868 encoded frames. It ended in P05 at
`INCOMPLETE_CONTROLLER_BLOCKED` / `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`; RR and RL
windows were never reached. This is a real predecessor failure, not an RR/RL
detail, task success, media error or safety abort. The sealed source/run manifest
hashes are respectively
`454dfc33130d732d78d9c6f3a25a569065199fd1847eec30f09097f5520c46a4` and
`4a90890968a075dc5ebd8ea1b0f08d0c6c9ad54d91d765b3f85d26f260211546`.

The immutable review directory is
`video_review/CP229120_deterministic_rr_retention_72e_review/`: full and detail
each decode868/868 frames at15fps and57.866667s with zero black-like frames;
the detail explicitly says `RR NOT REACHED | PREDECESSOR FAILURE TAIL`. The
1108-frame historical-N comparison freezes only the shorter candidate after its
own endpoint; those240 frozen frames are not physical evidence. All three files
have monotonic continuous PTS and full-decode validation in `export_receipt.json`.

## P12 block sealed CP227584; natural P01 training launched

P12 run exited0 normally with `STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`.
Actual1024 learner decisions /8 PPO /160 Adam, episodes452+453+119;
the last119 are a nonterminal tail at the legal update boundary, not success.
Checkpoint `checkpoint_step_000227584.pt`, lifetime227584/1743/34860,
LR1e-5, save_load_round_trip=true. Both pointer hashes verified:
checkpoint7969bdd11f60acb2aa1aaf09185aa6a76b3951ae2e4f6dd2f1989e11e4630df5;
sidecar24ca80245bd9208094f329bbcb545e280c5c686bacbec387845f23073b172f7d.
Turn optimized credit now2304 decisions /18 PPO /360 Adam /newAUX0.

After sole oldIsaac32440 exited and clean production was checked, official
same72e full_episode/P01 train launched from that immutableCP227584:
requested1536, seed1001,N1,cuda:0,checkpoint interval1. Exact active run is
`train/20260924T0938059083357Z_g72e63592bdf4_889c2d62e7fe433ea0dc3e99787a26ca`;
its live runtime identity records sole Isaac PID19200 and HEAD72e63592bdf4.
No migration or warm start. PrefixSource=frozen_fsm and
`mode=semantic_prior_eval` are CLI/default labels; the started receipt is
`stage=full_episode`, `from_phase=P01`, `teacher_offset_decisions=0`, and the
first issued row is ordinary learner global227585, not an uncredited teacher
prefix. That row has all12 phase mask enabled, one conditional-Gaussian draw,
12 raw/applied channels, and saved selected-raw log probability
22.961448669433594 exactly equal to the rollout old log probability. It records
rear task assists disabled and owner suspension explicitly not a rear teacher.
Frozen execution config keeps the P05 hip-only FL assist enabled (inactive in
this initial P01 row); RR capture assist remains null. No new deterministic
video exists yet. Expected end counters and growing P01 rows are not credit.

## P12 course handoff queued at next complete update; first check now3840

At update1742/global227456 the P12 block has optimized896 decisions /7PPO /
140Adam. Two complete learner episodes contain452+453=905 rows;9 pending
rows are not yet optimized. Third continuous nominal prefix is running.
The existing pinned `stop_after_update.request.json` was submitted2026-09-24
05:26 local. It is expected to close the next real128 rollout at1024 block
decisions, not at a partial buffer or artificial task terminal. Wait for the
actual checkpoint and natural process exit before starting another Isaac.

Rebalanced first inspection block: earlierP07768 + P10512 + P12 planned1024
+ naturalP01 planned1536 =3840 new decisions. This protects natural predecessor
coverage and avoids another nominal reset solely to fill the earlier1536
suffix allocation. These future counts are NOT yet credit. Same72e runtime,
weights/Adam/LR/normalization/HISTORY preserved, no new AUX or deployed probe.

Second episode ended81.958333s / `LOCAL_BOUNDED_RECOVERY_EXHAUSTED`, no RL
cross/place and no safety abort. Fresh learner RL qualifications6323/6747/6880
all returnedGROUND. A learner-only6688→6752 window has CoM(+28.91,+8.08,+1.50)mm,
body forward+35.54mm and RL bearing10.59→0N, with RR legalTOP8/9 and FL9/9.
Its positive FR-axis projection+19.92mm is mainly forward: dy has the opposite
sign to the fixed FR direction. This is unload/forward progress, not proven
FR-side lateral transfer. RR firstGROUND7064 then never restored TOP/current
qualification. Newest checkpoint is still unevaluated from naturalP01.

## P12 first episode sealed; three more real updates, no rear success claim

Current 72e P12+8 run remains active, sole Isaac PID32440. CP226944 and its
sidecar were hash-verified against the pointer: checkpoint
`b59bd23205fab96c154ae689fc6cef7b45cdcaabaa6a0c2918e9c13a342bd992`,
sidecar `2b3bb7bd97255c73c97966279aad2a59d81f8bf11f793bc56555c0267d135056`.
Its save/load round trip is true. This block has completed384 decisions /
3 PPO updates1736–1738 /60 Adam; turn total1664/13/260, new AUX0.
The first episode collected452 learner decisions;68 pending rows are not
additional optimized credit. Latest checkpoint has no natural-P01 video yet.

The actual accepted P12 handoff was tick6216/51.8s, after777 nominal decisions,
all zero PPO credit. RR placement6133 and initial RL qualification6212 are
prefix-owned. Student requalifications6294 and7124 both later returnGROUND;
RL cross/place remain false. First episode ends tick9832/81.933333s at
`LOCAL_BOUNDED_RECOVERY_EXHAUSTED`, P12 age30.6667s versus30+0.66007 allowance.
`stall_diagnostic=false`, no safety termination. Do not describe this as an
RL-edge collision or hard-limit abort, or a stall-detector termination.

RR leaves legal TOP XY at6368 while still sensorTOP13.13N; it is not yetAIR.
At6448 RR andFL are both AIR at the sampled endpoint; first RR GROUND endpoint
is7432. CoM6224→9832 changes(-170.91,-71.36,-26.85)mm; fixed initial FR-axis
projection is-105.02mm, not successful FR-directed transfer. Early retreat
includes the source's finite four-wheel reverse pulse; later sourceSTOP plus
negative policy is a separate issue. Late FL isAIR, so its rotation cannot be
claimed as ground traction. Second continuous prefix is running unchanged.

## CP226560 sealed; P12+8 real continuation active

The P10 course naturally exited0 at its complete512-decision boundary,
`STOPPED_AT_VERIFIED_UPDATE_BOUNDARY`; no process kill or production hot edit.
The final128 rollout includes the127 pending rows plus one real learner
decision after a fourth continuous nominal prefix. The boundary is not a
task-success or terminal claim; nonterminal tail bootstrap remains normal.

Actual checkpoint:
`../ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000226560.pt`
SHA868981b1a31c3bba8bc97f747e4e01a7c9d5b1963dccc8fcec2e725c19be87c5;
sidecarSHA974452660a4b224996d20fab0e61f0bcef1f02648ae085a5e46bc5fd875dbad8.
Lifetime226560/1735PPO/34700Adam, LR1e-5, save_load_round_trip true.
This72e block adds512/4/80; whole turn adds1280/10/200, newAUX0.
It has NOT yet been natural-P01 deterministic video evaluated.

The next official CLI run is active on unchanged72e:
`train/20260924T0835078527430Z_g72e63592bdf4_eb09a77b75b0420997a80291feb869d8`.
Requested1536, phase_suffix/P12, successful_nominal, offset8, seed1001/N1,
checkpoint interval1, cuda:0. Sole Isaac PID32440 at launch; previous9760 exited.
No migration/warm-start/head reset. Current source is immutableCP226560 above.
Actual prefix entry and RL coverage are pending physical evidence, not inferred
from the phase label or offset. NaturalP011536 follows this block.

Only outputs-side preparation is concurrent:72e video metadata export, an
unexecuted isolated post-stop FL-wheel half-residual probe (25 stdlib tests),
and a conditional finite front-prefix-retention439 AUX candidate. The latter
is NOT deployed or fitted; production, optimizer, reward and input meaning are
unchanged. No new rear task assistant or forced-forward rule is enabled.

## P10 complete-update course handoff requested; not yet a new checkpoint

1734/CP226432 is confirmed complete: this72e block384 optimized decisions,
3 PPO updates/60 Adam; turn total1152/9/180, newAUX0. A stop-after-update
request was submitted only after1734 completed, so the next normal complete
update is the intended512-decision boundary. It does not hot-edit runtime or
fabricate done. The third episode ended at block sample511; the existing
adapter is producing another real prefix for the final learner sample.
Unoptimized127-row tail is not added to optimized credit above.

Course reallocation after actual save/Isaac exit: retain the actual latest
compatible72e checkpoint; P10 actual512, then successful_nominal P12+8
decision-offset1536, then naturalP011536. Together with earlierP07768 this
keeps the planned4352 decision total. P12+8 is a real continuous predecessor,
not a snapshot, stable-entry certification or learner lift credit. Old N
suggests an RR/FL bearing plus earlyRL AIR window; current receipt/contact
must establish what entry actually occurred. Prefix fallback, if any, must
be reported as naturalP01 coverage instead. No same-version runtime change.

Second P10 episode62 learner decisions ended55.208333s with rear_right_knee
HARD_JOINT_LIMIT, not the first episode'sFR knee. RL qualification6219 was
revoked6335; a second6529 attempt was revoked6621; neither crossed/placed.
The latter window retained actualRR contact force even while its XY left the
legal top zone; do not call itAIR. Source stops occurred while FL policy kept
negative wheel requests under realFL bearing. This motivates only a prepared,
NOTEXECUTED independent post-stop FL-wheel magnitude diagnostic, not a global
forward rule, deployed rear assist, newAUX or formalpolicy-success claim.

## 72e P10 continuation: first256 optimized, course still active

As of2026-09-24 local03:55 the same sole Isaac course is still running;
no production/config/observation/action change during its rollout.
Completed1732–1733 add256 real decisions/2 PPO/40 Adam, zero new AUX.
Turn total against225280 is1024/8/160. Immutable CP226304 is hash-verified:
`../ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000226304.pt`
SHA6e2c383a59c2bf854b59866cf969cf7e777b9a01764376e2cb16f54cdcf24dd4,
sidecarSHAadb555f129792cd017a8cf7f6b9fc9c9fb95e7db8f97ad75a84b72b0b87736fc.
Lifetime226304/1733/34660, LR1e-5, save_load_round_trip true.
This new checkpoint has NOT been natural-P01 video evaluated yet.

Successful_nominal prefix endedtick6136/51.1333s,767 uncredited decisions.
RR placement6133 is prefix-owned. RL qualification6198 is after learner
handoff, but in the first rollout before the new optimizer update; it is not
evidence that the new reward already taught that movement. It persisted for
two sampled endpoints, then current qualification was revoked atground6214.
First256 endpoints:31 realRR TOP bearing,120 reachable qualifiedRR AIR,
2 RLqualified/1 edge-recovery,0 RLcross/TOP/placed/P13/fullcompletion.
See72e_first256_completed_readonly.md/json; overlapping windows are not summed.
Training has no episode video. The most recent delivered formal video remains
the sealed f6d CP226048 incomplete84.8s run below, not a72e video.

Current first episode has now ended at72.0417s/tick8645 after314 learner
decisions, with real front_right_knee HARD_JOINT_LIMIT. Its unoptimized tail
is not credited above. Normal reset/prefix continues. Do not relabel this as
RL-edge collision or claim removing a nonexistent collision termination.
P10 requested2048 remains in progress; naturalP01 requested1536 is next.

## CP226048 natural-P01 sealed; reward-only72e continuation started

Formal f6d DET CP226048 video sealed at84.8s/1272 decisions, P09
INCOMPLETE_CONTROLLER_BLOCKED. FR placed24.075s, FL placed45.758333s;
RR never acquired qualified lift/crossing/placement, RL likewise absent.
At end RR ground2.98674N/gap-50.965mm/front-245.029mm; FL AIR gap180.727mm.
The P09 source clock held at80, waiting current free lift before pending knee;
late FL/RL owner was not issued. Do not attribute this earlier failure to an
active owner-dropout suspension, or claim that this latest model is better.
Lifecycle DIAGNOSTIC_FAILURE means task nonacceptance; sealed video exists.
Three decoded/verified15fps videos are in
`video_review/CP226048_deterministic_owner439_recovery_review/`.
Full84.8s and detail30.2s preserve the failure; historicalN is explicitly not
fresh same-versionB, frozen after its73.808s endpoint. New rear task assistsOFF,
FLassistON, owner suspension/projectionON; no newAUX. Original references retained.

After Isaac exited, production frozen at72e63592bdf412d375abfda29b03cf456fbf4f7e.
Six runtime paths: semantic_reward.py reward-local current-retention replacement,
new semantic_rr_retention_migration.py and existing CLI/migration/training/namespace
wiring. Shared task potential/439input/HISTORY/rawGaussian/caps/physics unchanged.
Only already-placedRR recovery outside old legalAIR/TOP gains bounded geometric
distinction in its existing share; this does NOT fix the above first-lift failure.
Normal import tests cover retention/contact/input/terminal, actual source plans,
CLI entry, full-state publication/reload, synthetic128-row receipt inheritance,
owner/edge/proxy/prefix paths. Test-only RSL-RL density initialization was corrected;
one old profile assertion now expects already-authorized edgev4, safety negatives
remain. Actual frozenCP same-input deterministic/stochastic/raw-density/value
equivalence test passed; synthetic work is zero physical PPO/AUX credit.

Immutable zero-credit publication:
`../ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_rr_retention_CP226048_g72e63592bdf4.pt`
SHA15be034b14cfc8741a9dc83e12b6da2220af6d97ccee06b8da109c0e9b706113;
sidecarSHA80f350f6bb6f5b188f8eec4ea3fa5053de62cf81944014ff5b344a8c44e1f3ed.
Independent reload verified all439 learned weights/Adam/LR/Identity/RNG/counts;
226048 decisions/1731 PPO/34620 Adam unchanged. Old owner receipt preserved.
Fresh empty rollout, fresh legal continuous prefix; no simulator snapshot claim.
Main pointer not promoted by publication.

Started existing-CLI P10 successful_nominal-prefix2048 block:
`train/20260924T0731401970292Z_g72e63592bdf4_8abca665b0f64439bdd233a9d189e513`.
Seed1001/N1/128rollout/LR1e-5/120Hz physics/15Hz decisions, checkpoint everyupdate.
Requested2048 is NOT yet completed credit. NaturalP01 learner1536 is next planned
coverage; no second Isaac, no hot edits. This turn's currently completed real
credit remains768 decisions/6 PPO/120 Adam/0 AUX until new sealed updates exist.

## CP226048 sealed; natural-P01 DET video currently running

The first real course stopped normally at a verified complete update boundary,
not a fabricated task terminal:768 new policy decisions /6 PPO /120 Adam,
zero AUX. Immutable latest learned checkpoint is
`../ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000226048.pt`,
SHA256 `fbd27dea193153c7150881b1258dbf5ce249cfd5d1529e22b066be0abf0da973`;
sidecar SHA256 `93f41386de6d46fc204f2a22cb2c712ea83eb9b1d6b29c32a7276e2d9da6df53`.
Lifetime226048 decisions /1731 PPO /34620 Adam; save/load roundtriptrue.
Actual requested-phase samples P07=4,P08=4,P09=276,P10=7,P11=11,P12=466,
P13=0, natural-P01 policy samples0 so far. All reset prefixes are uncredited.
`owner439_first768_coverage.md/json` reports physical windows:48 RR reachable
AIR,28 real RR-bearing prep,22 positive FR-axis projection plus RL fraction
decline (NOT lateral transfer/qualified unloading), and zero qualified RL
edge-recovery/capture/TOP/placed. Many P12 samples are RR-ground recovery,
not successful RL training. This evidence motivated earlier evaluation and
subsequent P10-heavy plus longer natural-P01 sampling, not a task-success claim.

Active formal video: `video_eval/validation/20260924T0651033031701Z_gf6d1d2df8d87_b494c761b75a4755bd13d45572d8a2c1`.
It uses the immutableCP226048, naturalP01, deterministic mean, seed4001,
unchanged frozenf6d, FLassistON/ownerprojectionON/rearcompletionOFF, no probe.
At this entry the video is NOT sealed and has no claimed outcome. No new
reward or diagnostic candidate has been deployed. Single Isaac resource only.

Pending after evidence: continue from the actual latest compatible state, with
P10 real support predecessor2048 and naturalP01 long enough to include rear
outcomes (planned1536, not a convergence guarantee). A return-ground retention
candidate and same439 identity-migration design exist only in outputs documents;
they are not runtime changes or learning gains. Never reapply append439 to the
original225280 to discard these6 real updates.

## First real update saved; course still active

At local02:18 on2026-09-24, immutable CP225408 and its sidecar exist and the
complete optimizer row confirms225408 lifetime decisions/1726 PPO. This is
+128 decisions/+1 PPO/+20 Adam relative to this turn's source; no AUX.
`../ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000225408.pt`
is not yet certified by a natural-P01 deterministic video. The course remains
active; do not interpret these counters as the final4096 result.

First completed sampled episode:118 decisions from successful-N P07 prefix,
RR placed at tick5850 with later real TOP bearing, then FL knee actual
-60.0115665deg triggered unchanged HARD_JOINT_LIMIT at6103/50.858333s/P12.
RL had initial clearance but no qualified lift/crossing/placement. This is
suffix progress, not natural-P01 success. See `first_owner_episode118_readonly.md`.
RR's two earlier brief losses preceded the late source owner; sampled owner
active4 remainedfalse. Thus this episode does not physically validate a
post-late-owner support-loss suspension. Six FL knee and24 RR knee distinct
raw samples landed at FINAL -58deg; actual tracking/loads must be distinguished
from source, residual and final target. The second sampled episode ended
at45.375s/P09 with actual central BODY_COLLISION; it was not relabeled or hidden.

Monitoring correction: Windows directory Length metadata can stay0 while the
writer has appended megabytes. `live_progress.py` uses an independent direct
open/seek-END/tell snapshot and only complete JSON rows. Prefix progress is
separate from optimized samples and cannot raise PPO counters.

## Current boundary: f6d publication; real 4096 course running

The exact439 suffix-prefix interface repair passed74 focused tests (9 new
interface cases,55 existing prefix cases,10 actor/migration cases). Frozen HEAD:
`f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b`. No physics, reward or control change
relative to the preceding b0ff publication was made for this interface repair.
The new full-state publication was independently reloaded:
`../ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_rear_owner_CP225280_gf6d1d2df8d87.pt`.
SHA256 `fbe28e3718a5796afc2271e7b10aa04017369593b5c62e0bde1973a20f68032f`.
Actor/critic/Adam/normalizer hashes, training RNG, LR and all counters match the
b0ff439 publication exactly. Still225280 decisions/1725 PPO/34500 Adam; migration
and republication earn zero learning credit. Both immutable versions remain.

Course `20260924_owner439_4096_course_v3` is ACTIVE at this entry, with serial
P07=1024, P10=2048, naturalP01=1024 policy decisions. The first actual Isaac child
is `train/20260924T0550107325587Z_gf6d1d2df8d87_44bb94e9c5a84f44bfb4e14785d144d7`.
The coordinator uses base Python; only the official child uses Isaac Python.
No new PPO update is confirmed yet. Prefix data is not policy credit. Production
is frozen throughout the block; this status is not a completed-course result.

Diagnostic correction: the older probe analysis mislabeled mapper nominal
`native_drive_target_full12` as final. It is superseded by
`student_entry_direction_probe_v2_analysis_ACK_FINAL.json` and
`ANALYSIS_SUPERSEDED.md`. The real atomic ACK `drive_target_full12` changed during
the3-decision probe: FL knee -44.371→-32.871deg, FR knee -18.825→-7.325deg,
RR hip +6.985→-4.515deg; RR knee stayed at -58deg. Actual responses confirm
these requests reached the execution path, but neither the short active window
nor the later15.168mm gap produced RR TOP contact. No successful AUX label or
PPO learning is inferred from it. Original evidence and superseded analyses stay.

The sections below retain earlier chronological status, not current blockers.

## Actual completed diagnostic / pending prefix adapter fix

The isolated `20260924_owner439_student_entry_v2` diagnostic sealed naturally at
tick10940 /91.166667s/P09, `INCOMPLETE_CONTROLLER_BLOCKED`. Front FR/FL placement,
FL→P06 and valid RR lift/crossing were retained. Candidate raw overrides occurred
only from66.533333 to66.733333s (3 decisions,24 physical ticks), then released
when front distance became negative. RR gap temporarily approached15mm after
release but returned near56mm, with no sustained TOP capture. This is NOT a
formal policy evaluation or successful AUX label. Model state unchanged; new
PPO/AUX counts still0. All original diagnostic evidence is preserved.

First4096 course launch `.../curriculum/20260924_owner439_4096_course_v1` failed
before child physics because the standard single-process guard matched the
orchestrator interpreter's `env_isaaclab` path. The guard was not changed;
the pure orchestration helper now runs under base Python313.
Second launch `.../curriculum/20260924_owner439_4096_course_v2` reached the real
runner but rejected439 in the suffix prefix adapter before reset/sampling.
Child: `train/20260924T0544133172435Z_gb0ff99729b1d_f5be68206c454e82afdcd679af8309f7`.
Both failed startup ledgers remain intact; neither earns policy decisions or PPO.
The missing exact439 layout/provenance acceptance is being repaired with focused
tests. No running rollout is being edited. The published439 weights/Adam have
not learned since migration; the corrected publication must verify their exact
identity rather than discard/reset learned state.

Next requested block is4096 (1024 P07 +2048 P10 +1024 naturalP01), keeping128-step
rollouts. The prior sealed384-decision suffix blocks ended without a physical
terminal;42 contact/qualification changes were already inside rollouts, so
there is no evidence to treat128 alone as the cause. See
`smallcredit_horizon_readonly.md` for unmodified reward/value/advantage evidence.

Latest boundary: production frozen at `b0ff99729b1df84c4648f9d53b1d372f1e9152af`.
Full-state 439 successor of CP225280 has been published and independently
reloaded; see `IMPLEMENTATION_BOUNDARY.md` and
`rear_owner_CP225280_gb0ff99729b1d_publication.json`. Counts remain
225280/1725/34500; no migration learning credit. The first isolated diagnostic
`20260924_owner439_student_entry_v1` exited at tick1 (0.008333s) because its
outputs-only raw-observation JSON writer did not handle the dataclass. No
intervention activated, and no PPO/AUX update occurred. Error run is preserved;
this is logging/infrastructure error, not evidence of task failure.

The entries below retain design-time chronology; actual subsequent training
and physical results must be appended only after sealing.

2026-09-24. Continuing the existing project; no new training stack, physics asset,
network reset, optimizer reset, or current-rollout interruption. At entry HEAD
`ccd90e9b973719d20e1a9e610b58e827e9d35130` was clean and no Isaac process existed.
The parent archival commit and all old artifacts are preserved. No ancestor/current
AGENTS.md was present in C:/, C:/robotics_sim, wlr_robot or this project.

Verified immutable source CP225280:
`../ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000225280.pt`.
SHA256 `21897a4b2d2a85f01092c187c1b1ac824d8e61b01edaf732a6385ca951ab1dcf`;
sidecar `6e05b4e279c1d8ac99447f91d69b7db3f6dd6afaab6a326e610283f783b1f185`.
Actual source counts225280 decisions /1725 PPO /34500 Adam steps.
This is latest saved, not certified superior to acceptedCP221696 or successful N.
Source DET remains incomplete: RR crossed but no TOP capture, lastgap56.119mm;
RL wall contact did not terminate the episode. No new physical result is claimed here.

## Implementation being verified

RR candidate/capture + cooperative preparation → current RR/FL support → FR-directed
transfer → RL swing/edge recovery → capture → whole-body finish.

- Preserve actual RR bearing requirement for strong source transfer. A currently
  qualified same-attempt RL EDGE contact can retain a recovery task without being
  labeled AIR, support, placed or success. Ground return revokes current evidence.
- Exit previously issued P09-late FL/RL and P12-dependent RL servo owners on lost
  prerequisites. Use committed FINAL and adjacent residual request as entry anchors;
  changed policy requests retain bidirectional bounded control. Exclude stale N pull
  on those owned channels only, before unchanged hard limits/final slew/atomic write.
  A lawful RL AIR continuation does not release unsupported FL strong transfer.
  This is disclosed control-layer interference, not learned rear capture or a teacher
  trajectory, and cannot undo already-realized body momentum or guarantee support.
- Wheels/source stops are not suspended; later source/capture owners retire the hold.
  New anchors/flags are public:422→439, old numeric codec/kernel retained as prefix.
  FL assist remainsON; rear capture/geometry assistants and forced wheel forwardOFF.
- Replace two saturated preparation proxies using existing potential shares. FL uses
  remaining protected range relative to actual36° request capacity; RL uses wheel-to-
  top/lateral-corridor deficit instead of distance away from the obstacle. These do
  not prove force leverage, whole-linkage clearance, transfer or actual bearing.
- Preserve rawGaussian sample/log likelihood, HISTORY, all12channels, optimizer/LR,
  Identity normalizers, existing sigma, finite task200s and original safety.
  A future migration will zero-append only17 input columns/Adam moments, discard old
  rollout, and save/reload before new real collection. No migration learning credit.

## Next execution boundary

Finish targeted wiring/owner/edge/probability tests; freeze exact runtime and publish
full-state439 successor; perform a short isolated learner-entry direction diagnostic
(zeroPPO/AUX credit), then real serial rear-heavy PPO blocks and naturalP01 reload
evaluation/video. Diagnostic candidates are not training-success labels until their
real consequences are checked. Do not run a second Isaac or edit a running block.

No physical/PPO/AUX credit has yet been added in this directory. Pending budgets
are plans only. An incomplete physical attempt must preserve its actual failure tail.
